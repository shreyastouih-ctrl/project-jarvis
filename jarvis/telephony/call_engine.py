"""Autonomous AI Call Screening Engine, Urgency Classifier, and Debrief Logger."""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Optional, Protocol

from jarvis.core.event_bus import EventBus, EventType
from jarvis.core.state_machine import AssistantState, AssistantStateMachine
from jarvis.telephony.state import TelephonyState, TelephonyStateMachine, UrgencyLevel

logger = logging.getLogger(__name__)


def _safe_create_task(coro: Any) -> Optional[asyncio.Task]:
    """Safely schedule a coroutine task on the active event loop if one exists."""
    try:
        loop = asyncio.get_running_loop()
        return loop.create_task(coro)
    except RuntimeError:
        try:
            coro.close()
        except Exception:
            pass
        return None


@dataclass
class CallDebriefRecord:
    """Structured record generated upon call conclusion."""

    call_id: str
    caller_name: str
    caller_number: str
    timestamp: str
    duration_seconds: float
    summary: str
    urgency: str  # "Low", "Normal", "High", "Emergency"
    action_item: str
    transcript: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        """Convert record to a dictionary."""
        return {
            "call_id": self.call_id,
            "caller_name": self.caller_name,
            "caller_number": self.caller_number,
            "timestamp": self.timestamp,
            "duration_seconds": self.duration_seconds,
            "summary": self.summary,
            "urgency": self.urgency,
            "action_item": self.action_item,
            "transcript": list(self.transcript),
        }


@dataclass
class ActiveCall:
    """Represents a live call supporting both attribute and dict-like indexing."""

    call_id: str
    caller_name: str
    caller_number: str
    start_time: float
    state: str = "screening"
    urgency: str = "Normal"
    transcript: list[dict[str, Any]] = field(default_factory=list)
    taken_over: bool = False
    silent_turns: int = 0

    def __getitem__(self, key: str) -> Any:
        if hasattr(self, key):
            return getattr(self, key)
        raise KeyError(key)

    def __setitem__(self, key: str, value: Any) -> None:
        setattr(self, key, value)

    def __contains__(self, key: str) -> bool:
        return hasattr(self, key)

    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default)

    def keys(self) -> list[str]:
        return [
            "call_id", "caller_name", "caller_number", "start_time",
            "state", "urgency", "transcript", "taken_over", "silent_turns"
        ]

    def values(self) -> list[Any]:
        return [getattr(self, k) for k in self.keys()]

    def items(self) -> list[tuple[str, Any]]:
        return [(k, getattr(self, k)) for k in self.keys()]

    def to_dict(self) -> dict[str, Any]:
        return {k: getattr(self, k) for k in self.keys()}


class UrgencyClassifier:
    """Rule-based natural language urgency classifier with monotonicity and silent detection."""

    EMERGENCY_KEYWORDS = {
        "emergency", "reactor breach", "critical", "crash", "hospital",
        "attack", "police", "911", "intruder", "life-or-death",
        "casualty", "meltdown", "suit breach", "armor failure", "fatal",
    }

    HIGH_KEYWORDS = {
        "urgent", "asap", "immediately", "priority", "important",
        "defense brief", "deadline", "shield", "pentagon", "classified",
        "board meeting", "time sensitive",
    }

    LOW_KEYWORDS = {
        "warranty", "telemarketer", "special offer", "promotion",
        "discount", "newsletter", "no rush", "whenever you can",
        "at your leisure", "unsolicited",
    }

    @classmethod
    def classify(cls, text: str, current_urgency: str = "Normal") -> tuple[str, bool]:
        """Classify utterance into UrgencyLevel.

        Returns (urgency_str, is_silent).
        """
        stripped = text.strip() if text else ""
        if not stripped or stripped in ("...", "..", ".", "silence"):
            return UrgencyLevel.LOW.value, True

        lower = text.lower()

        # Emergency check
        if any(kw in lower for kw in cls.EMERGENCY_KEYWORDS):
            return UrgencyLevel.EMERGENCY.value, False

        # High priority check
        if any(kw in lower for kw in cls.HIGH_KEYWORDS):
            # Monotonicity: do not downgrade if already Emergency
            if current_urgency == UrgencyLevel.EMERGENCY.value:
                return UrgencyLevel.EMERGENCY.value, False
            return UrgencyLevel.HIGH.value, False

        # Low check
        if any(kw in lower for kw in cls.LOW_KEYWORDS):
            if current_urgency in (UrgencyLevel.EMERGENCY.value, UrgencyLevel.HIGH.value):
                return current_urgency, False
            return UrgencyLevel.LOW.value, False

        # Default Normal (preserve higher prior urgency)
        if current_urgency in (UrgencyLevel.EMERGENCY.value, UrgencyLevel.HIGH.value):
            return current_urgency, False
        return UrgencyLevel.NORMAL.value, False


class CallEngineProtocol(Protocol):
    """Protocol contract for CallEngine implementations."""

    busy_mode: bool
    def set_busy_mode(self, enabled: bool) -> None: ...
    def handle_incoming_call(self, call_id: str, caller_name: str, caller_number: str) -> str: ...
    def process_caller_utterance(self, call_id: str, text: str) -> str: ...
    def take_over(self, call_id: str) -> None: ...
    def end_call(self, call_id: str) -> CallDebriefRecord: ...
    def get_call_history(self) -> list[CallDebriefRecord]: ...


class CallEngine:
    """Autonomous AI Call Screening Proxy for Jarvis."""

    def __init__(
        self,
        event_bus: Optional[EventBus] = None,
        state_machine: Optional[AssistantStateMachine] = None,
        default_busy: bool = False,
    ) -> None:
        self.event_bus = event_bus
        self.assistant_sm = state_machine
        self._busy_mode = default_busy
        self.telephony_sm = TelephonyStateMachine(
            TelephonyState.BUSY if default_busy else TelephonyState.AVAILABLE
        )
        self.active_calls: dict[str, ActiveCall] = {}
        self.history: list[CallDebriefRecord] = []

    @property
    def busy_mode(self) -> bool:
        """Return whether busy mode is enabled."""
        return self._busy_mode

    @property
    def current_state(self) -> TelephonyState:
        """Return current telephony state machine state."""
        return self.telephony_sm.current_state

    def set_busy_mode(self, enabled: bool) -> None:
        """Toggle busy mode on or off."""
        self._busy_mode = enabled
        target = TelephonyState.BUSY if enabled else TelephonyState.AVAILABLE
        if self.telephony_sm.current_state in (TelephonyState.AVAILABLE, TelephonyState.BUSY):
            self.telephony_sm.transition_to(target, reason="user_toggle")

        if self.assistant_sm:
            _safe_create_task(
                self.assistant_sm.transition_to(
                    AssistantState.BUSY if enabled else AssistantState.IDLE,
                    reason="telephony_busy_toggle",
                )
            )

    def handle_incoming_call(self, call_id: str, caller_name: str, caller_number: str) -> str:
        """Handle an incoming phone call."""
        if call_id in self.active_calls:
            raise ValueError(f"Call {call_id} is already active")

        call = ActiveCall(
            call_id=call_id,
            caller_name=caller_name,
            caller_number=caller_number,
            start_time=time.time(),
            state="screening" if self._busy_mode else "ringing",
            urgency="Normal",
        )
        self.active_calls[call_id] = call

        # 1. Publish EventType.CALL_INCOMING
        if self.event_bus:
            _safe_create_task(
                self.event_bus.publish_event(
                    EventType.CALL_INCOMING,
                    call_id=call_id,
                    caller=caller_name,
                    caller_name=caller_name,
                    number=caller_number,
                    caller_number=caller_number,
                )
            )

        # 2. If busy, Jarvis intercepts and conducts screening
        if self._busy_mode:
            self.telephony_sm.transition_to(TelephonyState.SCREENING, reason="incoming_call_busy")
            greeting = "Good day, Mr. Stark is currently unavailable. May I take a message?"
            now_str = time.strftime("%H:%M:%S")
            entry = {
                "role": "jarvis",
                "speaker": "jarvis",
                "text": greeting,
                "time": now_str,
                "timestamp": time.time(),
            }
            call.transcript.append(entry)

            if self.event_bus:
                _safe_create_task(
                    self.event_bus.publish_event(
                        EventType.CALL_SCREENING_UPDATE,
                        call_id=call_id,
                        state="screening",
                        transcript=list(call.transcript),
                        urgency=call.urgency,
                    )
                )
            return greeting

        return f"Incoming call from {caller_name} ({caller_number})."

    def process_caller_utterance(self, call_id: str, text: str) -> str:
        """Process spoken utterance from caller during screening."""
        if call_id not in self.active_calls:
            raise KeyError(f"Call {call_id} not active")

        call = self.active_calls[call_id]
        now_str = time.strftime("%H:%M:%S")

        # If user has already taken over, record transcript and do not proxy autonomous responses
        if call.taken_over:
            call.transcript.append({
                "role": "caller",
                "speaker": "caller",
                "text": text,
                "time": now_str,
                "timestamp": time.time(),
            })
            if self.event_bus:
                _safe_create_task(
                    self.event_bus.publish_event(
                        EventType.CALL_SCREENING_UPDATE,
                        call_id=call_id,
                        state=call.state,
                        transcript=list(call.transcript),
                        urgency=call.urgency,
                    )
                )
            return ""

        # Record caller turn
        call.transcript.append({
            "role": "caller",
            "speaker": "caller",
            "text": text,
            "time": now_str,
            "timestamp": time.time(),
        })

        urgency, is_silent = UrgencyClassifier.classify(text, current_urgency=call.urgency)

        if is_silent:
            call.silent_turns += 1
            if call.silent_turns == 1:
                response = "Pardon me, are you still there, sir?"
            else:
                # Preserve Emergency or High urgency across silent turns
                if call.urgency not in (UrgencyLevel.EMERGENCY.value, UrgencyLevel.HIGH.value):
                    call.urgency = UrgencyLevel.LOW.value
                response = "I am unable to hear you. Goodbye."
        else:
            call.silent_turns = 0
            call.urgency = urgency
            if urgency == UrgencyLevel.EMERGENCY.value:
                call.state = TelephonyState.URGENT_ESCALATION.value
                self.telephony_sm.transition_to(
                    TelephonyState.URGENT_ESCALATION, reason="emergency_detected", force=True
                )
                response = "I detect a critical emergency. Alerting Mr. Stark immediately."
            elif urgency == UrgencyLevel.HIGH.value:
                response = "Understood, sir. I have noted this message and marked it as high priority."
            else:
                response = "I have noted that message. I will ensure he receives it."

        # Record Jarvis response
        call.transcript.append({
            "role": "jarvis",
            "speaker": "jarvis",
            "text": response,
            "time": now_str,
            "timestamp": time.time(),
        })

        if self.event_bus:
            _safe_create_task(
                self.event_bus.publish_event(
                    EventType.CALL_SCREENING_UPDATE,
                    call_id=call_id,
                    state=call.state,
                    transcript=list(call.transcript),
                    urgency=call.urgency,
                )
            )

        return response

    def take_over(self, call_id: str) -> None:
        """User takes over screened call. Safe on non-active/disconnected calls."""
        if call_id not in self.active_calls:
            logger.info("Takeover ignored for disconnected or nonexistent call %s", call_id)
            return

        call = self.active_calls[call_id]
        call.taken_over = True
        call.state = TelephonyState.USER_TAKEOVER.value  # "user_connected"
        self.telephony_sm.transition_to(TelephonyState.USER_TAKEOVER, reason="user_takeover", force=True)

        if self.event_bus:
            _safe_create_task(
                self.event_bus.publish_event(
                    EventType.CALL_SCREENING_UPDATE,
                    call_id=call_id,
                    state="user_connected",
                    transcript=list(call.transcript),
                    urgency=call.urgency,
                )
            )

    def end_call(self, call_id: str) -> CallDebriefRecord:
        """End call and generate structured CallDebriefRecord."""
        if call_id not in self.active_calls:
            raise KeyError(f"Call {call_id} not active")

        call = self.active_calls.pop(call_id)
        duration = max(0.5, round(time.time() - call.start_time, 2))

        caller_msgs = [
            t["text"]
            for t in call.transcript
            if (t.get("role") == "caller" or t.get("speaker") == "caller") and t["text"].strip()
        ]
        summary = (
            f"Call with {call.caller_name}: " + ("; ".join(caller_msgs) if caller_msgs else "No message left")
        )

        if call.urgency in (UrgencyLevel.HIGH.value, UrgencyLevel.EMERGENCY.value):
            action_item = f"Follow up with {call.caller_name} ({call.caller_number})"
        elif call.urgency == UrgencyLevel.LOW.value:
            action_item = "No immediate action required"
        else:
            action_item = f"Review message from {call.caller_name} ({call.caller_number})"

        debrief = CallDebriefRecord(
            call_id=call_id,
            caller_name=call.caller_name,
            caller_number=call.caller_number,
            timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            duration_seconds=duration,
            summary=summary,
            urgency=call.urgency,
            action_item=action_item,
            transcript=list(call.transcript),
        )

        self.history.append(debrief)
        if not self.active_calls:
            self.telephony_sm.transition_to(
                TelephonyState.BUSY if self._busy_mode else TelephonyState.AVAILABLE,
                reason="call_ended",
                force=True,
            )
        else:
            # Update telephony_sm based on remaining active calls
            if any(c.urgency == UrgencyLevel.EMERGENCY.value for c in self.active_calls.values()):
                self.telephony_sm.transition_to(TelephonyState.URGENT_ESCALATION, reason="remaining_emergency", force=True)
            elif any(c.taken_over for c in self.active_calls.values()):
                self.telephony_sm.transition_to(TelephonyState.USER_TAKEOVER, reason="remaining_takeover", force=True)
            else:
                self.telephony_sm.transition_to(TelephonyState.SCREENING, reason="remaining_active_calls", force=True)

        if self.event_bus:
            _safe_create_task(
                self.event_bus.publish_event(
                    EventType.CALL_DEBRIEF_CREATED,
                    call_id=call_id,
                    caller_name=debrief.caller_name,
                    caller_number=debrief.caller_number,
                    summary=debrief.summary,
                    urgency=debrief.urgency,
                    action_item=debrief.action_item,
                    transcript=debrief.transcript,
                    duration_seconds=debrief.duration_seconds,
                    timestamp=debrief.timestamp,
                )
            )

        return debrief

    def get_call_history(
        self,
        caller_number: Optional[str] = None,
        caller_name: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> list[CallDebriefRecord]:
        """Return chronological list of completed call debrief records with optional filtering."""
        records = list(self.history)
        if caller_number is not None:
            records = [r for r in records if r.caller_number == caller_number]
        if caller_name is not None:
            records = [r for r in records if r.caller_name == caller_name]
        if limit is not None and limit > 0:
            records = records[-limit:]
        return records

    def clear_call_history(self) -> None:
        """Clear call history log."""
        self.history.clear()
