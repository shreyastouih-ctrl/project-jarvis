"""Unit tests for telephony state machine, call screening proxy, urgency classification, and debrief logging."""

from __future__ import annotations

import asyncio
import time
import pytest

from jarvis.core.event_bus import EventBus, EventType
from jarvis.core.state_machine import AssistantState, AssistantStateMachine
from jarvis.telephony.call_engine import (
    ActiveCall,
    CallDebriefRecord,
    CallEngine,
    UrgencyClassifier,
)
from jarvis.telephony.simulator import (
    CallSimulator,
    SCENARIO_EMERGENCY_REACTOR_BREACH,
    SCENARIO_HIGH_URGENT_DEFENSE,
    SCENARIO_NORMAL_STARK_EXPO,
    SCENARIO_SILENT_CALLER,
    SCENARIO_USER_TAKEOVER,
)
from jarvis.telephony.state import (
    InvalidTelephonyStateTransitionError,
    TelephonyState,
    TelephonyStateMachine,
    UrgencyLevel,
)


# ==============================================================================
# CLASS 1: BUSY MODE STATE MACHINE (5 Tests)
# ==============================================================================
class TestBusyModeStateMachine:
    """Verifies telephony state machine lifecycle, transitions, and assistant state sync."""

    def test_initial_state_available_mode(self, telephony_engine: CallEngine) -> None:
        """Verify initial available state and empty call history."""
        assert telephony_engine.busy_mode is False
        assert telephony_engine.current_state == TelephonyState.AVAILABLE
        assert len(telephony_engine.get_call_history()) == 0

    def test_toggle_busy_mode_transitions(self, telephony_engine: CallEngine) -> None:
        """Verify toggle sets busy mode flag and transitions telephony state machine."""
        telephony_engine.set_busy_mode(True)
        assert telephony_engine.busy_mode is True
        assert telephony_engine.current_state == TelephonyState.BUSY

        telephony_engine.set_busy_mode(False)
        assert telephony_engine.busy_mode is False
        assert telephony_engine.current_state == TelephonyState.AVAILABLE

    def test_state_machine_legal_transitions(self) -> None:
        """Verify canonical sequence of legal state transitions."""
        sm = TelephonyStateMachine(TelephonyState.AVAILABLE)
        assert sm.current_state == TelephonyState.AVAILABLE

        sm.transition_to(TelephonyState.BUSY, reason="busy_toggle")
        sm.transition_to(TelephonyState.SCREENING, reason="call_incoming")
        sm.transition_to(TelephonyState.TAKING_MESSAGE, reason="message_turn")
        sm.transition_to(TelephonyState.URGENT_ESCALATION, reason="urgent_detected")
        sm.transition_to(TelephonyState.USER_TAKEOVER, reason="user_intervened")
        sm.transition_to(TelephonyState.CALL_COMPLETED, reason="call_ended")
        sm.transition_to(TelephonyState.AVAILABLE, reason="reset")

        assert sm.current_state == TelephonyState.AVAILABLE
        history = sm.get_history()
        assert len(history) == 8  # 1 init + 7 transitions

    def test_state_machine_illegal_transitions_rejected(self) -> None:
        """Verify illegal direct state transitions raise InvalidTelephonyStateTransitionError."""
        sm = TelephonyStateMachine(TelephonyState.AVAILABLE)
        with pytest.raises(InvalidTelephonyStateTransitionError):
            sm.transition_to(TelephonyState.USER_TAKEOVER)

        with pytest.raises(InvalidTelephonyStateTransitionError):
            sm.transition_to(TelephonyState.CALL_COMPLETED)

    @pytest.mark.asyncio
    async def test_busy_mode_assistant_state_machine_sync(self, event_bus: EventBus) -> None:
        """Verify toggling busy mode synchronizes the central assistant state machine."""
        assistant_sm = AssistantStateMachine(event_bus)
        engine = CallEngine(event_bus=event_bus, state_machine=assistant_sm, default_busy=False)

        engine.set_busy_mode(True)
        await asyncio.sleep(0.01)
        assert assistant_sm.current_state == AssistantState.BUSY

        engine.set_busy_mode(False)
        await asyncio.sleep(0.01)
        assert assistant_sm.current_state == AssistantState.IDLE


# ==============================================================================
# CLASS 2: INCOMING CALL HANDLING (4 Tests)
# ==============================================================================
class TestIncomingCallHandling:
    """Verifies incoming call routing, screening initialization, and duplicate rejection."""

    def test_incoming_call_in_available_mode_rings(self, telephony_engine: CallEngine) -> None:
        """Verify incoming call in available mode rings and does not screen."""
        telephony_engine.set_busy_mode(False)
        resp = telephony_engine.handle_incoming_call("call-avail-1", "Tony Stark", "+1-555-0100")

        assert "incoming call" in resp.lower()
        call = telephony_engine.active_calls["call-avail-1"]
        assert call["state"] == "ringing"
        assert len(call["transcript"]) == 0

    def test_incoming_call_in_busy_mode_enters_screening(self, telephony_engine: CallEngine) -> None:
        """Verify incoming call in busy mode intercepts call with polite British screening greeting."""
        telephony_engine.set_busy_mode(True)
        greeting = telephony_engine.handle_incoming_call("call-busy-1", "Pepper Potts", "+1-555-0101")

        assert "unavailable" in greeting.lower()
        call = telephony_engine.active_calls["call-busy-1"]
        assert call["state"] == "screening"
        assert len(call["transcript"]) == 1
        assert call["transcript"][0]["role"] == "jarvis"
        assert call["transcript"][0]["speaker"] == "jarvis"

    @pytest.mark.asyncio
    async def test_call_incoming_event_bus_emission(
        self, telephony_engine: CallEngine, event_bus: EventBus
    ) -> None:
        """Verify handle_incoming_call publishes CALL_INCOMING event to EventBus."""
        telephony_engine.handle_incoming_call("call-ev-1", "Justin Hammer", "+1-555-0199")

        ev = await event_bus.wait_for(EventType.CALL_INCOMING, timeout=1.0)
        assert ev is not None
        assert ev.get("call_id") == "call-ev-1"
        assert ev.get("caller") == "Justin Hammer"
        assert ev.get("number") == "+1-555-0199"

    def test_duplicate_active_call_id_rejected(self, telephony_engine: CallEngine) -> None:
        """Verify registering a duplicate active call ID raises ValueError."""
        telephony_engine.handle_incoming_call("call-dup-1", "Happy Hogan", "+1-555-0144")
        with pytest.raises(ValueError, match="already active"):
            telephony_engine.handle_incoming_call("call-dup-1", "Happy Hogan", "+1-555-0144")


# ==============================================================================
# CLASS 3: MULTI-TURN SCREENING CONVERSATION (5 Tests)
# ==============================================================================
class TestMultiTurnScreeningConversation:
    """Verifies dialogue progression, transcript collation, event updates, and silent caller handling."""

    def test_initial_jarvis_screening_greeting(self, telephony_engine: CallEngine) -> None:
        """Verify opening greeting is polite, mentions Mr. Stark's unavailability, and asks for message."""
        telephony_engine.set_busy_mode(True)
        greeting = telephony_engine.handle_incoming_call("call-greet-1", "Pepper Potts", "+1-555-0101")

        assert "good day" in greeting.lower() or "unavailable" in greeting.lower()
        assert "message" in greeting.lower()

    def test_single_turn_caller_message_recording(self, telephony_engine: CallEngine) -> None:
        """Verify caller message is recorded and Jarvis confirms message taking."""
        telephony_engine.set_busy_mode(True)
        telephony_engine.handle_incoming_call("call-msg-1", "Pepper Potts", "+1-555-0101")

        reply = telephony_engine.process_caller_utterance(
            "call-msg-1", "Please tell Mr. Stark the board meeting is rescheduled to 3 PM."
        )
        assert "noted" in reply.lower() or "message" in reply.lower()

        active = telephony_engine.active_calls["call-msg-1"]
        # Transcript: initial greeting + caller utterance + jarvis response = 3 turns
        assert len(active["transcript"]) == 3
        assert active["transcript"][1]["role"] == "caller"
        assert active["transcript"][1]["text"] == "Please tell Mr. Stark the board meeting is rescheduled to 3 PM."
        assert active["transcript"][2]["role"] == "jarvis"

    def test_multi_turn_structured_screening_dialogue(self, telephony_engine: CallEngine) -> None:
        """Verify multiple sequential dialogue turns preserve chronological transcript order."""
        telephony_engine.set_busy_mode(True)
        telephony_engine.handle_incoming_call("call-multi-1", "Justin Hammer", "+1-555-0199")

        telephony_engine.process_caller_utterance("call-multi-1", "I am calling about the Stark Expo contract.")
        telephony_engine.process_caller_utterance("call-multi-1", "Our legal team sent the updated terms.")
        telephony_engine.process_caller_utterance("call-multi-1", "Please have him call me back today.")

        active = telephony_engine.active_calls["call-multi-1"]
        # 1 greeting + 3 caller turns + 3 jarvis turns = 7 turns total
        assert len(active["transcript"]) == 7
        roles = [t["role"] for t in active["transcript"]]
        assert roles == ["jarvis", "caller", "jarvis", "caller", "jarvis", "caller", "jarvis"]

    @pytest.mark.asyncio
    async def test_screening_update_event_emission_per_turn(
        self, telephony_engine: CallEngine, event_bus: EventBus
    ) -> None:
        """Verify each screening turn publishes CALL_SCREENING_UPDATE to EventBus."""
        telephony_engine.set_busy_mode(True)
        telephony_engine.handle_incoming_call("call-ev-screen", "Col. Rhodes", "+1-555-0144")

        # Flush greeting event
        await event_bus.wait_for(EventType.CALL_SCREENING_UPDATE, timeout=1.0)

        telephony_engine.process_caller_utterance("call-ev-screen", "Need a quick update on the defense brief.")
        ev = await event_bus.wait_for(EventType.CALL_SCREENING_UPDATE, timeout=1.0)

        assert ev is not None
        assert ev.get("call_id") == "call-ev-screen"
        assert len(ev.get("transcript", [])) >= 3

    def test_empty_or_inaudible_caller_utterance_prompt(self, telephony_engine: CallEngine) -> None:
        """Verify two consecutive silent caller turns trigger 'still there' then 'goodbye'."""
        telephony_engine.set_busy_mode(True)
        telephony_engine.handle_incoming_call("call-sil-1", "Unknown Caller", "+1-555-0000")

        # Turn 1: silent
        r1 = telephony_engine.process_caller_utterance("call-sil-1", "")
        assert "still there" in r1.lower()

        # Turn 2: silent dots
        r2 = telephony_engine.process_caller_utterance("call-sil-1", "...")
        assert "goodbye" in r2.lower()

        active = telephony_engine.active_calls["call-sil-1"]
        assert active["urgency"] == UrgencyLevel.LOW.value


# ==============================================================================
# CLASS 4: URGENCY CLASSIFICATION AND ESCALATION (6 Tests)
# ==============================================================================
class TestUrgencyClassificationAndEscalation:
    """Verifies rule-based NLP classification across urgency tiers and monotonicity."""

    def test_urgency_low_telemarketer_classification(self) -> None:
        """Verify sales, warranty, and spam utterances classify as Low urgency."""
        urgency, is_silent = UrgencyClassifier.classify(
            "We are calling regarding your car's extended warranty and a special promotional offer."
        )
        assert urgency == UrgencyLevel.LOW.value
        assert is_silent is False

    def test_urgency_normal_routine_meeting_classification(self) -> None:
        """Verify routine business inquiries classify as Normal urgency."""
        urgency, is_silent = UrgencyClassifier.classify(
            "Hi Tony, confirming lunch tomorrow at 12:30 PM."
        )
        assert urgency == UrgencyLevel.NORMAL.value
        assert is_silent is False

    def test_urgency_high_priority_deadline_classification(self) -> None:
        """Verify priority deadlines and urgent defense inquiries classify as High urgency."""
        urgency, is_silent = UrgencyClassifier.classify(
            "Tony, need your urgent sign-off on the defense brief ASAP before 5 PM."
        )
        assert urgency == UrgencyLevel.HIGH.value
        assert is_silent is False

    def test_urgency_emergency_critical_keywords_classification(self, telephony_engine: CallEngine) -> None:
        """Verify reactor breach, casualty, and critical life safety classify as Emergency."""
        telephony_engine.set_busy_mode(True)
        telephony_engine.handle_incoming_call("call-urg-emerg", "Facility Guard", "+1-555-9110")

        resp = telephony_engine.process_caller_utterance(
            "call-urg-emerg", "Critical emergency! Reactor breach detected in sector 4!"
        )
        assert "emergency" in resp.lower() or "alerting" in resp.lower()
        active = telephony_engine.active_calls["call-urg-emerg"]
        assert active["urgency"] == UrgencyLevel.EMERGENCY.value

    def test_emergency_state_escalation_and_hud_event(self, telephony_engine: CallEngine) -> None:
        """Verify emergency urgency transitions call state to urgent_escalation."""
        telephony_engine.set_busy_mode(True)
        telephony_engine.handle_incoming_call("call-esc-1", "Security", "+1-555-9111")

        telephony_engine.process_caller_utterance("call-esc-1", "Intruder detected in the workshop armor vault!")
        active = telephony_engine.active_calls["call-esc-1"]

        assert active["state"] == TelephonyState.URGENT_ESCALATION.value
        assert telephony_engine.current_state == TelephonyState.URGENT_ESCALATION

    def test_monotonic_urgency_escalation_no_downgrade(self) -> None:
        """Verify emergency urgency level never downgrades upon subsequent polite phrases."""
        u1, _ = UrgencyClassifier.classify("Critical reactor breach!", current_urgency="Normal")
        assert u1 == UrgencyLevel.EMERGENCY.value

        u2, _ = UrgencyClassifier.classify("Thank you very much, good day.", current_urgency=u1)
        assert u2 == UrgencyLevel.EMERGENCY.value  # Monotonic invariant maintained

    def test_silent_turns_after_emergency_preserve_urgency(self, telephony_engine: CallEngine) -> None:
        """Verify that once a call reaches Emergency or High, silent turns do not downgrade urgency to Low."""
        telephony_engine.set_busy_mode(True)
        telephony_engine.handle_incoming_call("call-emerg-silent", "Dr. Banner", "+1-555-9111")

        # Turn 1: Emergency declared
        telephony_engine.process_caller_utterance("call-emerg-silent", "Critical emergency! The Hulk is out of control!")
        active = telephony_engine.active_calls["call-emerg-silent"]
        assert active["urgency"] == UrgencyLevel.EMERGENCY.value

        # Turn 2: Silent turn 1
        r1 = telephony_engine.process_caller_utterance("call-emerg-silent", "")
        assert "still there" in r1.lower()
        assert active["urgency"] == UrgencyLevel.EMERGENCY.value

        # Turn 3: Silent turn 2
        r2 = telephony_engine.process_caller_utterance("call-emerg-silent", "...")
        assert "goodbye" in r2.lower()
        assert active["urgency"] == UrgencyLevel.EMERGENCY.value

        # Debrief verification
        deb = telephony_engine.end_call("call-emerg-silent")
        assert deb.urgency == UrgencyLevel.EMERGENCY.value
        assert "Follow up with Dr. Banner" in deb.action_item

        # Also verify High urgency is preserved across silent turns
        telephony_engine.handle_incoming_call("call-high-silent", "Col. Rhodes", "+1-555-0144")
        telephony_engine.process_caller_utterance("call-high-silent", "Tony, urgent deadline for defense brief!")
        assert telephony_engine.active_calls["call-high-silent"]["urgency"] == UrgencyLevel.HIGH.value

        telephony_engine.process_caller_utterance("call-high-silent", "")
        telephony_engine.process_caller_utterance("call-high-silent", "   ")
        deb_high = telephony_engine.end_call("call-high-silent")
        assert deb_high.urgency == UrgencyLevel.HIGH.value
        assert "Follow up with Col. Rhodes" in deb_high.action_item


# ==============================================================================
# CLASS 5: USER CALL TAKEOVER (4 Tests)
# ==============================================================================
class TestUserCallTakeover:
    """Verifies manual takeover hand-off from screening and graceful disconnected handling."""

    def test_takeover_from_active_screening(self, telephony_engine: CallEngine) -> None:
        """Verify user takeover transitions active call state to user_connected."""
        telephony_engine.set_busy_mode(True)
        telephony_engine.handle_incoming_call("call-to-1", "Happy Hogan", "+1-555-1234")

        telephony_engine.take_over("call-to-1")
        active = telephony_engine.active_calls["call-to-1"]

        assert active["taken_over"] is True
        assert active["state"] == "user_connected"
        assert telephony_engine.current_state == TelephonyState.USER_TAKEOVER

    def test_takeover_from_urgent_escalation(self, telephony_engine: CallEngine) -> None:
        """Verify takeover from emergency escalation state transitions smoothly to user_connected."""
        telephony_engine.set_busy_mode(True)
        telephony_engine.handle_incoming_call("call-to-emerg", "Guard", "+1-555-9110")
        telephony_engine.process_caller_utterance("call-to-emerg", "Emergency in sector 7!")

        telephony_engine.take_over("call-to-emerg")
        active = telephony_engine.active_calls["call-to-emerg"]

        assert active["taken_over"] is True
        assert active["state"] == "user_connected"

    @pytest.mark.asyncio
    async def test_takeover_event_bus_notification(
        self, telephony_engine: CallEngine, event_bus: EventBus
    ) -> None:
        """Verify takeover publishes CALL_SCREENING_UPDATE with state='user_connected'."""
        telephony_engine.set_busy_mode(True)
        telephony_engine.handle_incoming_call("call-to-ev", "Col. Rhodes", "+1-555-0144")

        # Flush incoming & greeting events
        await event_bus.wait_for(EventType.CALL_SCREENING_UPDATE, timeout=1.0)

        telephony_engine.take_over("call-to-ev")
        ev = await event_bus.wait_for(EventType.CALL_SCREENING_UPDATE, timeout=1.0)

        assert ev is not None
        assert ev.get("state") == "user_connected"

    def test_takeover_nonexistent_or_disconnected_call_is_safe(self, telephony_engine: CallEngine) -> None:
        """Verify takeover on a nonexistent or already ended call does not raise an exception."""
        telephony_engine.set_busy_mode(True)
        telephony_engine.handle_incoming_call("call-disc-test", "Col. Rhodes", "+1-555-2222")
        telephony_engine.end_call("call-disc-test")

        # Safely ignored on ended call
        telephony_engine.take_over("call-disc-test")
        assert "call-disc-test" not in telephony_engine.active_calls

        # Safely ignored on never-existing call
        telephony_engine.take_over("call-never-existed")

    def test_utterances_after_takeover_do_not_override_takeover_state(
        self, telephony_engine: CallEngine
    ) -> None:
        """Verify caller utterances after user takeover return empty string and do not mutate takeover state."""
        telephony_engine.set_busy_mode(True)
        telephony_engine.handle_incoming_call("call-to-guard", "Pepper Potts", "+1-555-0100")
        telephony_engine.take_over("call-to-guard")

        assert telephony_engine.current_state == TelephonyState.USER_TAKEOVER
        active = telephony_engine.active_calls["call-to-guard"]
        assert active["taken_over"] is True

        # Utterance sent while user is on the call
        reply = telephony_engine.process_caller_utterance(
            "call-to-guard", "Tony, are you on the line? Critical breach!"
        )
        assert reply == ""  # Jarvis does not proxy autonomous responses
        assert active["taken_over"] is True
        assert active["state"] == "user_connected"
        assert telephony_engine.current_state == TelephonyState.USER_TAKEOVER

        # Check transcript contains the caller message
        last_turn = active["transcript"][-1]
        assert last_turn["role"] == "caller"
        assert "Critical breach!" in last_turn["text"]


# ==============================================================================
# CLASS 6: CALL DEBRIEF AND HISTORY LOGGING (6 Tests)
# ==============================================================================
class TestCallDebriefAndHistoryLogging:
    """Verifies structured debrief creation, action item generation, history filtering, and event publication."""

    def test_end_call_returns_valid_debrief_record(self, telephony_engine: CallEngine) -> None:
        """Verify ending a call produces a complete CallDebriefRecord."""
        telephony_engine.set_busy_mode(True)
        telephony_engine.handle_incoming_call("call-deb-1", "Colonel Rhodes", "+1-555-0144")
        telephony_engine.process_caller_utterance("call-deb-1", "Review the urgent defense brief immediately.")

        debrief = telephony_engine.end_call("call-deb-1")

        assert isinstance(debrief, CallDebriefRecord)
        assert debrief.call_id == "call-deb-1"
        assert debrief.caller_name == "Colonel Rhodes"
        assert debrief.caller_number == "+1-555-0144"
        assert debrief.duration_seconds > 0.0
        assert "defense brief" in debrief.summary
        assert debrief.urgency in (UrgencyLevel.HIGH.value, UrgencyLevel.EMERGENCY.value)
        assert "Follow up with Colonel Rhodes" in debrief.action_item
        assert len(debrief.transcript) >= 3

    def test_debrief_action_item_generation_by_urgency(self, telephony_engine: CallEngine) -> None:
        """Verify action item phrasing corresponds to classified urgency level."""
        # Low urgency
        telephony_engine.set_busy_mode(True)
        telephony_engine.handle_incoming_call("call-low-deb", "Telemarketer", "+1-555-0000")
        telephony_engine.process_caller_utterance("call-low-deb", "Calling about your car warranty discount.")
        debrief_low = telephony_engine.end_call("call-low-deb")
        assert "No immediate action required" in debrief_low.action_item

        # Normal urgency
        telephony_engine.handle_incoming_call("call-norm-deb", "Pepper Potts", "+1-555-0101")
        telephony_engine.process_caller_utterance("call-norm-deb", "Confirming tomorrow's dinner reservations.")
        debrief_norm = telephony_engine.end_call("call-norm-deb")
        assert "Review message from Pepper Potts" in debrief_norm.action_item

        # High urgency
        telephony_engine.handle_incoming_call("call-high-deb", "General Ross", "+1-555-7777")
        telephony_engine.process_caller_utterance("call-high-deb", "Urgent pentagon briefing ASAP.")
        debrief_high = telephony_engine.end_call("call-high-deb")
        assert "Follow up with General Ross" in debrief_high.action_item

    @pytest.mark.asyncio
    async def test_call_debrief_created_event_emission(
        self, telephony_engine: CallEngine, event_bus: EventBus
    ) -> None:
        """Verify ending a call publishes CALL_DEBRIEF_CREATED event with full debrief payload."""
        telephony_engine.set_busy_mode(True)
        telephony_engine.handle_incoming_call("call-ev-deb", "Happy Hogan", "+1-555-1234")
        telephony_engine.process_caller_utterance("call-ev-deb", "Manifest signed and ready.")

        debrief = telephony_engine.end_call("call-ev-deb")
        ev = await event_bus.wait_for(EventType.CALL_DEBRIEF_CREATED, timeout=1.0)

        assert ev is not None
        assert ev.get("call_id") == "call-ev-deb"
        assert ev.get("caller_name") == "Happy Hogan"
        assert ev.get("urgency") == debrief.urgency

    def test_call_history_chronological_logging(self, telephony_engine: CallEngine) -> None:
        """Verify multiple calls are logged in strict chronological order."""
        telephony_engine.set_busy_mode(True)
        for i in range(3):
            cid = f"call-hist-{i}"
            telephony_engine.handle_incoming_call(cid, f"Caller {i}", f"+1-555-000{i}")
            telephony_engine.process_caller_utterance(cid, f"Message turn from caller {i}")
            telephony_engine.end_call(cid)

        history = telephony_engine.get_call_history()
        assert len(history) == 3
        assert [r.call_id for r in history] == ["call-hist-0", "call-hist-1", "call-hist-2"]

    def test_call_history_filtering_and_limits(self, telephony_engine: CallEngine) -> None:
        """Verify history query filtering by caller number and limit slicing."""
        telephony_engine.set_busy_mode(True)
        telephony_engine.handle_incoming_call("c1", "Pepper", "+1-555-0101")
        telephony_engine.end_call("c1")

        telephony_engine.handle_incoming_call("c2", "Rhodey", "+1-555-0144")
        telephony_engine.end_call("c2")

        telephony_engine.handle_incoming_call("c3", "Pepper", "+1-555-0101")
        telephony_engine.end_call("c3")

        pepper_calls = telephony_engine.get_call_history(caller_number="+1-555-0101")
        assert len(pepper_calls) == 2
        assert all(c.caller_name == "Pepper" for c in pepper_calls)

        latest_one = telephony_engine.get_call_history(limit=1)
        assert len(latest_one) == 1
        assert latest_one[0].call_id == "c3"

        telephony_engine.clear_call_history()
        assert len(telephony_engine.get_call_history()) == 0

    def test_end_call_nonexistent_raises_key_error(self, telephony_engine: CallEngine) -> None:
        """Verify calling end_call on a nonexistent call ID raises KeyError."""
        with pytest.raises(KeyError, match="not active"):
            telephony_engine.end_call("nonexistent-call-id")

    def test_multi_call_end_call_preserves_urgent_escalation(
        self, telephony_engine: CallEngine
    ) -> None:
        """Verify ending one call does not reset engine to BUSY when another Emergency call remains active."""
        telephony_engine.set_busy_mode(True)
        telephony_engine.handle_incoming_call("call-A", "Normal Caller", "+1-555-1111")
        telephony_engine.handle_incoming_call("call-B", "Emergency Caller", "+1-555-9999")

        # Escalate Call B to Emergency
        telephony_engine.process_caller_utterance("call-B", "Critical emergency! Reactor core overload!")
        assert telephony_engine.current_state == TelephonyState.URGENT_ESCALATION

        # End Call A (normal call)
        telephony_engine.end_call("call-A")

        # State should STILL be URGENT_ESCALATION because Call B is active and emergency
        assert telephony_engine.current_state == TelephonyState.URGENT_ESCALATION
        assert "call-B" in telephony_engine.active_calls

        # End Call B
        telephony_engine.end_call("call-B")

        # Now that no calls remain, state should transition to BUSY
        assert telephony_engine.current_state == TelephonyState.BUSY
        assert len(telephony_engine.active_calls) == 0

    def test_multi_call_end_call_preserves_user_takeover(
        self, telephony_engine: CallEngine
    ) -> None:
        """Verify ending one call preserves USER_TAKEOVER when another taken-over call remains active."""
        telephony_engine.set_busy_mode(True)
        telephony_engine.handle_incoming_call("call-1", "Caller 1", "+1-555-0001")
        telephony_engine.handle_incoming_call("call-2", "Caller 2", "+1-555-0002")

        telephony_engine.take_over("call-2")
        assert telephony_engine.current_state == TelephonyState.USER_TAKEOVER

        # End Call 1
        telephony_engine.end_call("call-1")

        # State should remain USER_TAKEOVER because call-2 is still active and taken over
        assert telephony_engine.current_state == TelephonyState.USER_TAKEOVER

        # End Call 2
        telephony_engine.end_call("call-2")
        assert telephony_engine.current_state == TelephonyState.BUSY


# ==============================================================================
# CLASS 7: CALL SIMULATOR PRESET SCENARIOS (2 Tests)
# ==============================================================================
class TestCallSimulatorPresets:
    """Verifies headless simulator execution against standard Iron Man dialogue scenarios."""

    def test_call_simulator_sync_scenarios(self, telephony_engine: CallEngine) -> None:
        """Verify synchronous execution of all preset scenarios."""
        sim = CallSimulator(engine=telephony_engine)

        # 1. Emergency reactor breach
        deb_emerg = sim.run_simulation(SCENARIO_EMERGENCY_REACTOR_BREACH)
        assert deb_emerg.urgency == UrgencyLevel.EMERGENCY.value
        assert "Facility Guard" in deb_emerg.caller_name

        # 2. High urgent defense
        deb_high = sim.run_simulation(SCENARIO_HIGH_URGENT_DEFENSE)
        assert deb_high.urgency == UrgencyLevel.HIGH.value
        assert "defense brief" in deb_high.summary

        # 3. Normal stark expo
        deb_norm = sim.run_simulation(SCENARIO_NORMAL_STARK_EXPO)
        assert deb_norm.urgency == UrgencyLevel.NORMAL.value

        # 4. Silent caller
        deb_silent = sim.run_simulation(SCENARIO_SILENT_CALLER)
        assert deb_silent.urgency == UrgencyLevel.LOW.value

        # 5. User takeover
        deb_to = sim.run_simulation(SCENARIO_USER_TAKEOVER)
        assert deb_to.call_id == SCENARIO_USER_TAKEOVER.call_id

    @pytest.mark.asyncio
    async def test_call_simulator_async_scenarios(self, telephony_engine: CallEngine) -> None:
        """Verify asynchronous execution with simulated speech delays."""
        sim = CallSimulator(engine=telephony_engine)
        deb_emerg = await sim.run_simulation_async(SCENARIO_EMERGENCY_REACTOR_BREACH)
        assert deb_emerg.urgency == UrgencyLevel.EMERGENCY.value
        assert deb_emerg.duration_seconds > 0.0

