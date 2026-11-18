"""Adversarial Concurrency & Stress Probe: Telephony Subsystem."""

import asyncio
import time
import pytest
from jarvis.core.event_bus import EventBus, EventType
from jarvis.core.state_machine import AssistantState, AssistantStateMachine
from jarvis.telephony.call_engine import CallEngine, UrgencyLevel
from jarvis.telephony.state import TelephonyState


class TestAdversarialTelephony:
    """Stress testing concurrent calls, urgency escalation races, and takeover preemption."""

    @pytest.mark.asyncio
    async def test_multiple_concurrent_incoming_calls(self):
        """Stress: 10 concurrent incoming calls received simultaneously while in busy mode."""
        bus = EventBus()
        sm = AssistantStateMachine(event_bus=bus)
        engine = CallEngine(event_bus=bus, state_machine=sm, default_busy=True)

        async def arrive_call(call_id: str, caller: str):
            await asyncio.sleep(0.001)
            return engine.handle_incoming_call(call_id, caller, f"555-010{call_id[-1]}")

        tasks = [arrive_call(f"call_{i}", f"Caller_{i}") for i in range(10)]
        greetings = await asyncio.gather(*tasks)

        assert len(engine.active_calls) == 10
        assert all("unavailable" in g for g in greetings)
        assert engine.current_state == TelephonyState.SCREENING

        # End all 10 calls concurrently
        async def finish_call(call_id: str):
            await asyncio.sleep(0.001)
            return engine.end_call(call_id)

        debriefs = await asyncio.gather(*[finish_call(f"call_{i}") for i in range(10)])
        assert len(debriefs) == 10
        assert len(engine.active_calls) == 0
        assert len(engine.get_call_history()) == 10
        assert engine.current_state == TelephonyState.BUSY
        await bus.close()

    @pytest.mark.asyncio
    async def test_emergency_escalation_race_and_multi_call_state_preservation(self):
        """Race: Call 1 (normal) and Call 2 (emergency) active. Call 1 ends. Call 2 preserves Emergency."""
        bus = EventBus()
        sm = AssistantStateMachine(event_bus=bus)
        engine = CallEngine(event_bus=bus, state_machine=sm, default_busy=True)

        engine.handle_incoming_call("c1", "Pepper", "555-0101")
        engine.handle_incoming_call("c2", "Rhodey", "555-0102")

        # C1 speaks normal message
        r1 = engine.process_caller_utterance("c1", "Just checking in on the schedule.")
        assert "noted" in r1.lower()
        assert engine.active_calls["c1"].urgency == UrgencyLevel.NORMAL.value

        # C2 speaks critical emergency message
        r2 = engine.process_caller_utterance("c2", "We have an armor failure and reactor breach! Emergency!")
        assert "critical emergency" in r2.lower()
        assert engine.active_calls["c2"].urgency == UrgencyLevel.EMERGENCY.value
        assert engine.current_state == TelephonyState.URGENT_ESCALATION

        # End C1: Telephony state must NOT revert to BUSY or SCREENING, it MUST remain URGENT_ESCALATION
        d1 = engine.end_call("c1")
        assert d1.urgency == UrgencyLevel.NORMAL.value
        assert engine.current_state == TelephonyState.URGENT_ESCALATION, (
            f"Telephony state downgraded prematurely to {engine.current_state} while C2 is still emergency!"
        )

        # C2 caller speaks subsequent casual comment: urgency must NOT downgrade
        r2_followup = engine.process_caller_utterance("c2", "Can you hear me? Thanks.")
        assert engine.active_calls["c2"].urgency == UrgencyLevel.EMERGENCY.value
        assert engine.current_state == TelephonyState.URGENT_ESCALATION

        # End C2
        d2 = engine.end_call("c2")
        assert d2.urgency == UrgencyLevel.EMERGENCY.value
        assert engine.current_state == TelephonyState.BUSY
        await bus.close()

    @pytest.mark.asyncio
    async def test_immediate_takeover_preempts_screening_dialogue(self):
        """Race: User takes over call while caller utterances arrive."""
        bus = EventBus()
        sm = AssistantStateMachine(event_bus=bus)
        engine = CallEngine(event_bus=bus, state_machine=sm, default_busy=True)

        engine.handle_incoming_call("c_takeover", "Happy Hogan", "555-0103")

        # Caller utterance 1 before takeover
        resp1 = engine.process_caller_utterance("c_takeover", "Jarvis, is Tony there?")
        assert len(resp1) > 0

        # User initiates takeover
        engine.take_over("c_takeover")
        assert engine.active_calls["c_takeover"].taken_over is True
        assert engine.current_state == TelephonyState.USER_TAKEOVER

        # Caller utterance 2 after takeover: Jarvis must stay silent (return empty string)
        resp2 = engine.process_caller_utterance("c_takeover", "Hey Tony, can you hear me now?")
        assert resp2 == "", f"Expected empty Jarvis response after takeover, got: '{resp2}'"

        # End call and verify debrief
        debrief = engine.end_call("c_takeover")
        assert debrief.call_id == "c_takeover"
        assert len(debrief.transcript) >= 3
        await bus.close()

    @pytest.mark.asyncio
    async def test_duplicate_and_disconnected_call_operations_safety(self):
        """Edge Case: Duplicate incoming call ID, duplicate takeover, and double end-call."""
        bus = EventBus()
        engine = CallEngine(event_bus=bus, default_busy=True)

        engine.handle_incoming_call("c_dup", "Caller", "555-0104")

        # Duplicate handle_incoming_call must raise ValueError
        with pytest.raises(ValueError):
            engine.handle_incoming_call("c_dup", "Caller", "555-0104")

        # Multiple take_over calls on active call must be safe and idempotent
        engine.take_over("c_dup")
        engine.take_over("c_dup")

        # Takeover on non-existent / disconnected call must not crash
        engine.take_over("non_existent_call_id")

        # First end_call succeeds
        engine.end_call("c_dup")

        # Second end_call raises KeyError cleanly
        with pytest.raises(KeyError):
            engine.end_call("c_dup")

        await bus.close()
