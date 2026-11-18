"""Telephony lifecycle state machine and urgency levels."""

from __future__ import annotations

import logging
import time
from enum import Enum
from typing import Optional

logger = logging.getLogger(__name__)


class TelephonyState(str, Enum):
    """Lifecycle states of the Jarvis Telephony Subsystem."""

    AVAILABLE = "available"
    BUSY = "busy"
    SCREENING = "screening"
    TAKING_MESSAGE = "taking_message"
    URGENT_ESCALATION = "urgent_escalation"
    USER_TAKEOVER = "user_connected"       # Value "user_connected" aligns with E2E assertions
    USER_CONNECTED = "user_connected"      # Explicit alias
    CALL_COMPLETED = "call_completed"


class UrgencyLevel(str, Enum):
    """Urgency classification levels for screened phone calls."""

    LOW = "Low"
    NORMAL = "Normal"
    HIGH = "High"
    EMERGENCY = "Emergency"


class InvalidTelephonyStateTransitionError(ValueError):
    """Raised when an invalid telephony state transition is attempted."""

    def __init__(self, current_state: TelephonyState, target_state: TelephonyState, reason: str = "") -> None:
        msg = f"Cannot transition telephony state from {current_state.value} to {target_state.value}"
        if reason:
            msg += f" (reason: {reason})"
        super().__init__(msg)
        self.current_state = current_state
        self.target_state = target_state
        self.reason = reason


# Directed transition graph
VALID_TELEPHONY_TRANSITIONS: dict[TelephonyState, set[TelephonyState]] = {
    TelephonyState.AVAILABLE: {
        TelephonyState.BUSY,
        TelephonyState.SCREENING,
    },
    TelephonyState.BUSY: {
        TelephonyState.AVAILABLE,
        TelephonyState.SCREENING,
    },
    TelephonyState.SCREENING: {
        TelephonyState.TAKING_MESSAGE,
        TelephonyState.URGENT_ESCALATION,
        TelephonyState.USER_TAKEOVER,
        TelephonyState.CALL_COMPLETED,
        TelephonyState.BUSY,
        TelephonyState.AVAILABLE,
    },
    TelephonyState.TAKING_MESSAGE: {
        TelephonyState.URGENT_ESCALATION,
        TelephonyState.USER_TAKEOVER,
        TelephonyState.CALL_COMPLETED,
        TelephonyState.BUSY,
        TelephonyState.AVAILABLE,
    },
    TelephonyState.URGENT_ESCALATION: {
        TelephonyState.USER_TAKEOVER,
        TelephonyState.CALL_COMPLETED,
        TelephonyState.BUSY,
        TelephonyState.AVAILABLE,
    },
    TelephonyState.USER_TAKEOVER: {
        TelephonyState.CALL_COMPLETED,
        TelephonyState.BUSY,
        TelephonyState.AVAILABLE,
    },
    TelephonyState.CALL_COMPLETED: {
        TelephonyState.AVAILABLE,
        TelephonyState.BUSY,
        TelephonyState.SCREENING,
    },
}


class TelephonyStateMachine:
    """Manages telephony subsystem state transitions and audit history."""

    def __init__(self, initial_state: TelephonyState = TelephonyState.AVAILABLE, max_history: int = 100) -> None:
        self._state = initial_state
        self._max_history = max_history
        self._history: list[tuple[float, TelephonyState, str]] = [
            (time.time(), initial_state, "initialization")
        ]

    @property
    def current_state(self) -> TelephonyState:
        return self._state

    @property
    def is_available(self) -> bool:
        return self._state == TelephonyState.AVAILABLE

    @property
    def is_busy(self) -> bool:
        return self._state == TelephonyState.BUSY

    @property
    def is_screening(self) -> bool:
        return self._state in (TelephonyState.SCREENING, TelephonyState.TAKING_MESSAGE, TelephonyState.URGENT_ESCALATION)

    def can_transition(self, target_state: TelephonyState) -> bool:
        if target_state == self._state:
            return True
        allowed = VALID_TELEPHONY_TRANSITIONS.get(self._state, set())
        return target_state in allowed

    def transition_to(self, target_state: TelephonyState, reason: str = "", force: bool = False) -> TelephonyState:
        old_state = self._state
        if old_state == target_state:
            return self._state

        if not force and not self.can_transition(target_state):
            logger.warning("Invalid telephony transition: %s -> %s (%s)", old_state.value, target_state.value, reason)
            raise InvalidTelephonyStateTransitionError(old_state, target_state, reason)

        self._state = target_state
        now = time.time()
        self._history.append((now, target_state, reason))
        if len(self._history) > self._max_history:
            self._history.pop(0)

        logger.info("Telephony transition: %s -> %s [%s]", old_state.value, target_state.value, reason)
        return self._state

    def get_history(self) -> list[tuple[float, TelephonyState, str]]:
        return list(self._history)
