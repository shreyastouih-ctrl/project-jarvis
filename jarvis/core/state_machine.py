"""Core Assistant State Machine managing state transitions, history, and bus dispatch."""

from __future__ import annotations

import asyncio
import logging
import time
from enum import Enum
from typing import Optional

from jarvis.core.event_bus import EventBus, EventType

logger = logging.getLogger(__name__)


class AssistantState(str, Enum):
    """The canonical states of the Jarvis AI Assistant."""

    IDLE = "IDLE"
    LISTENING = "LISTENING"
    PROCESSING = "PROCESSING"
    SPEAKING = "SPEAKING"
    BUSY = "BUSY"


class InvalidStateTransitionError(ValueError):
    """Raised when an invalid state transition is requested."""

    def __init__(self, current_state: AssistantState, target_state: AssistantState, reason: str = "") -> None:
        msg = f"Cannot transition from {current_state.value} to {target_state.value}"
        if reason:
            msg += f" (reason: {reason})"
        super().__init__(msg)
        self.current_state = current_state
        self.target_state = target_state
        self.reason = reason


# Directed graph of permissible state transitions
VALID_TRANSITIONS: dict[AssistantState, set[AssistantState]] = {
    AssistantState.IDLE: {
        AssistantState.LISTENING,
        AssistantState.PROCESSING,
        AssistantState.SPEAKING,
        AssistantState.BUSY,
    },
    AssistantState.LISTENING: {
        AssistantState.PROCESSING,
        AssistantState.IDLE,
        AssistantState.BUSY,
    },
    AssistantState.PROCESSING: {
        AssistantState.SPEAKING,
        AssistantState.IDLE,
        AssistantState.LISTENING,
        AssistantState.BUSY,
    },
    AssistantState.SPEAKING: {
        AssistantState.IDLE,
        AssistantState.LISTENING,  # Barge-in interruption
        AssistantState.BUSY,
    },
    AssistantState.BUSY: {
        AssistantState.IDLE,
        AssistantState.PROCESSING,
        AssistantState.SPEAKING,
        AssistantState.LISTENING,
    },
}


class AssistantStateMachine:
    """State machine governing the assistant lifecycle and broadcasting state changes."""

    def __init__(
        self,
        event_bus: Optional[EventBus] = None,
        initial_state: AssistantState = AssistantState.IDLE,
        max_history: int = 100,
    ) -> None:
        self._state = initial_state
        self._event_bus = event_bus
        self._max_history = max_history
        self._history: list[tuple[float, AssistantState, str]] = [
            (time.time(), initial_state, "initialization")
        ]
        self._lock = asyncio.Lock()

    @property
    def current_state(self) -> AssistantState:
        """Return the current assistant state."""
        return self._state

    @property
    def is_idle(self) -> bool:
        return self._state == AssistantState.IDLE

    @property
    def is_listening(self) -> bool:
        return self._state == AssistantState.LISTENING

    @property
    def is_processing(self) -> bool:
        return self._state == AssistantState.PROCESSING

    @property
    def is_speaking(self) -> bool:
        return self._state == AssistantState.SPEAKING

    @property
    def is_busy(self) -> bool:
        return self._state == AssistantState.BUSY

    def can_transition(self, target_state: AssistantState) -> bool:
        """Check if transitioning to target_state is permitted from current_state."""
        if target_state == self._state:
            return True
        allowed = VALID_TRANSITIONS.get(self._state, set())
        return target_state in allowed

    async def transition_to(
        self,
        target_state: AssistantState,
        reason: str = "",
        force: bool = False,
    ) -> AssistantState:
        """Execute a state transition with validation, history logging, and event bus broadcast."""
        async with self._lock:
            old_state = self._state
            if old_state == target_state:
                logger.debug("State already in %s, skipping transition", target_state.value)
                return self._state

            if not force and not self.can_transition(target_state):
                logger.warning(
                    "Invalid transition attempted: %s -> %s (reason: %s)",
                    old_state.value,
                    target_state.value,
                    reason,
                )
                raise InvalidStateTransitionError(old_state, target_state, reason)

            self._state = target_state
            now = time.time()
            self._history.append((now, target_state, reason))
            if len(self._history) > self._max_history:
                self._history.pop(0)

            logger.info("State transition: %s -> %s [%s]", old_state.value, target_state.value, reason)

        # Broadcast state change event outside the lock
        if self._event_bus:
            await self._event_bus.publish_event(
                EventType.STATE_CHANGED,
                old_state=old_state.value,
                new_state=target_state.value,
                reason=reason,
                timestamp=now,
            )

        return self._state

    def get_history(self) -> list[tuple[float, AssistantState, str]]:
        """Return a copy of the state transition history."""
        return list(self._history)
