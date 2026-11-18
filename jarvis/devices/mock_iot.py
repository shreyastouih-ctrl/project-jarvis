"""Simulated IoT gateway / hub executing asynchronous actuations and event broadcasts."""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Optional

from jarvis.core.event_bus import EventBus, EventType
from jarvis.devices.registry import DeviceRegistry

logger = logging.getLogger(__name__)


class MockIoTHub:
    """Simulated IoT Hub bridge (acting as HomeAssistant, Hue Bridge, or Zigbee Hub)."""

    def __init__(
        self,
        registry: DeviceRegistry,
        event_bus: Optional[EventBus] = None,
        latency_ms: float = 0.0,
        simulated_latency_ms: Optional[float] = None,
    ) -> None:
        self.registry = registry
        self.event_bus = event_bus or registry.event_bus
        self.latency_ms = simulated_latency_ms if simulated_latency_ms is not None else latency_ms
        self._offline_devices: set[str] = set()
        self._failure_modes: dict[str, str] = {}
        self._history: list[dict[str, Any]] = []

    @property
    def simulated_latency_ms(self) -> float:
        """Alias for latency_ms for testing backwards-compatibility."""
        return self.latency_ms

    @simulated_latency_ms.setter
    def simulated_latency_ms(self, value: float) -> None:
        self.latency_ms = value

    def simulate_offline(self, entity_id: str, offline: bool = True) -> None:
        """Mark a device as offline or online."""
        if offline:
            self._offline_devices.add(entity_id)
        else:
            self._offline_devices.discard(entity_id)

    def simulate_failure(self, entity_id: str, error_message: str) -> None:
        """Inject a simulated failure message for a device."""
        self._failure_modes[entity_id] = error_message

    def clear_failures(self) -> None:
        """Clear all injected failures and offline devices."""
        self._offline_devices.clear()
        self._failure_modes.clear()

    async def execute(self, entity_id: str, action: str, **params: Any) -> dict[str, Any]:
        """Asynchronously execute a device action with optional simulated latency and failure modes."""
        start_time = time.perf_counter()

        if self.latency_ms > 0:
            await asyncio.sleep(self.latency_ms / 1000.0)

        # Check offline simulation
        if entity_id in self._offline_devices:
            result = {
                "success": False,
                "error": f"Device {entity_id} is unreachable / offline",
                "entity_id": entity_id,
                "device_id": entity_id,
            }
            duration = time.perf_counter() - start_time
            self._record_history(entity_id, action, params, result, duration)
            return result

        # Check failure mode simulation
        if entity_id in self._failure_modes:
            result = {
                "success": False,
                "error": self._failure_modes[entity_id],
                "entity_id": entity_id,
                "device_id": entity_id,
            }
            duration = time.perf_counter() - start_time
            self._record_history(entity_id, action, params, result, duration)
            return result

        # Execute through registry
        result = self.registry.execute_command(entity_id, action, **params)

        # Explicitly broadcast DEVICE_STATE_CHANGED if event bus available, action succeeded,
        # and registry has not already published via its own event_bus
        if result.get("success") and self.event_bus and not self.registry.event_bus:
            dev = self.registry.get(entity_id)
            if dev:
                await self.event_bus.publish_event(
                    EventType.DEVICE_STATE_CHANGED,
                    device_id=entity_id,
                    entity_id=entity_id,
                    state=dev.state,
                    traits=dict(dev.attributes),
                    timestamp=dev.last_updated,
                )

        duration = time.perf_counter() - start_time
        self._record_history(entity_id, action, params, result, duration)
        return result

    def _record_history(
        self,
        entity_id: str,
        action: str,
        params: dict[str, Any],
        result: dict[str, Any],
        duration: float,
    ) -> None:
        self._history.append({
            "timestamp": time.time(),
            "entity_id": entity_id,
            "action": action,
            "params": params,
            "result": result,
            "duration_ms": duration * 1000.0,
        })

    def get_history(self) -> list[dict[str, Any]]:
        """Return execution history."""
        return list(self._history)

    def clear_history(self) -> None:
        """Clear execution history."""
        self._history.clear()
