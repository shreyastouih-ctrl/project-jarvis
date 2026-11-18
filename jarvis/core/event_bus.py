"""Asynchronous Publish/Subscribe Event Bus with error isolation and typed events."""

from __future__ import annotations

import asyncio
import inspect
import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Awaitable, Callable, Dict, List, Optional, Union

logger = logging.getLogger(__name__)


class EventType(str, Enum):
    """Canonical event types for the Jarvis AI Assistant."""

    STATE_CHANGED = "state_changed"             # payload: {"old_state": str, "new_state": str}
    SPEECH_STARTED = "speech_started"           # payload: {"timestamp": float}
    SPEECH_ENDED = "speech_ended"               # payload: {"audio_data": bytes, "duration_ms": float}
    TRANSCRIPT_READY = "transcript_ready"       # payload: {"text": str, "confidence": float}
    INTENT_DISPATCHED = "intent_dispatched"     # payload: {"intent": str, "model_id": str, "tier": int}
    TTS_SYNTHESIS_STARTED = "tts_started"       # payload: {"text": str}
    TTS_AUDIO_CHUNK = "tts_audio_chunk"         # payload: {"chunk": bytes, "is_final": bool}
    BARGE_IN_TRIGGERED = "barge_in_triggered"   # payload: {"generation_id": int, "timestamp": float}
    DEVICE_STATE_CHANGED = "device_changed"     # payload: {"device_id": str, "state": str, "traits": dict}
    CALL_INCOMING = "call_incoming"             # payload: {"call_id": str, "caller": str, "number": str}
    CALL_SCREENING_UPDATE = "call_screening"   # payload: {"call_id": str, "state": str, "transcript": list}
    CALL_DEBRIEF_CREATED = "call_debrief"       # payload: CallDebriefRecord dict
    TELEMETRY_UPDATE = "telemetry_update"       # payload: {"cpu": float, "ram": float, "net_ping": float, "audio_level": float}


@dataclass
class Event:
    """Encapsulation of a bus event."""

    event_type: EventType
    payload: dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)

    def get(self, key: str, default: Any = None) -> Any:
        """Helper to get payload properties."""
        return self.payload.get(key, default)


EventHandler = Union[
    Callable[[Event], Awaitable[None]],
    Callable[[Event], None],
]


class EventBus:
    """Async Pub/Sub event bus supporting typed events, filtering, and error isolation."""

    def __init__(self, max_history: int = 500) -> None:
        self._subscribers: dict[Optional[EventType], list[EventHandler]] = defaultdict(list)
        self._history: list[Event] = []
        self._max_history = max_history
        self._lock = asyncio.Lock()
        self._waiters: list[tuple[EventType, Optional[Callable[[Event], bool]], asyncio.Future[Event]]] = []
        self._running = True

    def subscribe(self, event_type: Optional[EventType], handler: EventHandler) -> None:
        """Subscribe a callback to a specific event type or wildcard (None for all events)."""
        if handler not in self._subscribers[event_type]:
            self._subscribers[event_type].append(handler)
            logger.debug("Subscribed %s to %s", handler, event_type)

    def unsubscribe(self, event_type: Optional[EventType], handler: EventHandler) -> None:
        """Unsubscribe a callback."""
        if handler in self._subscribers[event_type]:
            self._subscribers[event_type].remove(handler)
            logger.debug("Unsubscribed %s from %s", handler, event_type)

    async def publish(self, event: Event) -> None:
        """Publish an event to all registered subscribers with strict error isolation."""
        if not self._running:
            return

        async with self._lock:
            self._history.append(event)
            if len(self._history) > self._max_history:
                self._history.pop(0)

        # Notify waiting futures
        self._resolve_waiters(event)

        # Collect applicable handlers: specific event_type + wildcard (None)
        handlers: list[EventHandler] = []
        handlers.extend(self._subscribers.get(event.event_type, []))
        handlers.extend(self._subscribers.get(None, []))

        if not handlers:
            return

        # Execute all handlers in parallel with error isolation
        tasks = [self._invoke_safe(handler, event) for handler in handlers]
        await asyncio.gather(*tasks, return_exceptions=True)

    async def publish_event(self, event_type: EventType, **payload: Any) -> Event:
        """Convenience method to create and publish an event."""
        event = Event(event_type=event_type, payload=payload)
        await self.publish(event)
        return event

    async def _invoke_safe(self, handler: EventHandler, event: Event) -> None:
        """Execute a single handler with exception isolation."""
        try:
            if inspect.iscoroutinefunction(handler):
                await handler(event)
            else:
                res = handler(event)
                if inspect.isawaitable(res):
                    await res
        except Exception as exc:
            logger.exception("Exception in event handler %s for event %s: %s", handler, event.event_type, exc)

    def _resolve_waiters(self, event: Event) -> None:
        """Check and resolve any matching futures waiting for this event."""
        remaining: list[tuple[EventType, Optional[Callable[[Event], bool]], asyncio.Future[Event]]] = []
        for target_type, predicate, fut in self._waiters:
            if fut.done():
                continue
            if event.event_type == target_type:
                matched = False
                if predicate is None:
                    matched = True
                else:
                    try:
                        matched = bool(predicate(event))
                    except Exception as exc:
                        logger.exception("Exception in wait_for predicate for event %s: %s", event.event_type, exc)
                        matched = False
                if matched:
                    fut.set_result(event)
                    continue
            remaining.append((target_type, predicate, fut))
        self._waiters = remaining

    async def wait_for(
        self,
        event_type: EventType,
        timeout: float = 5.0,
        predicate: Optional[Callable[[Event], bool]] = None,
    ) -> Event:
        """Wait asynchronously for a specific event type with timeout."""
        loop = asyncio.get_running_loop()
        fut: asyncio.Future[Event] = loop.create_future()
        self._waiters.append((event_type, predicate, fut))
        try:
            return await asyncio.wait_for(fut, timeout=timeout)
        except asyncio.TimeoutError:
            if fut in [w[2] for w in self._waiters]:
                self._waiters = [w for w in self._waiters if w[2] is not fut]
            raise TimeoutError(f"Timed out waiting for event {event_type} after {timeout}s")

    def get_history(self, event_type: Optional[EventType] = None) -> list[Event]:
        """Return a copy of recent events, optionally filtered by event type."""
        if event_type is None:
            return list(self._history)
        return [e for e in self._history if e.event_type == event_type]

    def clear_history(self) -> None:
        """Clear recorded event history."""
        self._history.clear()

    async def close(self) -> None:
        """Gracefully close the event bus and cancel any pending waiters."""
        self._running = False
        for _, _, fut in self._waiters:
            if not fut.done():
                fut.cancel()
        self._waiters.clear()
        self._subscribers.clear()
