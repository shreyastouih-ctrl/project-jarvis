"""In-memory smart device registry supporting entity storage, actuation, and event publishing."""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Optional, Tuple

from jarvis.core.event_bus import EventBus, EventType
from jarvis.devices.intent_parser import DeviceIntentParser
from jarvis.devices.schema import DeviceAction, DeviceEntity, DeviceNotFoundError

logger = logging.getLogger(__name__)

# Allowed actions per device domain
_DOMAIN_ALLOWED_ACTIONS: dict[str, set[str]] = {
    "light": {
        DeviceAction.TURN_ON.value,
        DeviceAction.TURN_OFF.value,
        DeviceAction.TOGGLE.value,
        DeviceAction.SET_BRIGHTNESS.value,
        DeviceAction.QUERY.value,
    },
    "switch": {
        DeviceAction.TURN_ON.value,
        DeviceAction.TURN_OFF.value,
        DeviceAction.TOGGLE.value,
        DeviceAction.QUERY.value,
    },
    "thermostat": {
        DeviceAction.SET_TEMPERATURE.value,
        DeviceAction.TURN_ON.value,
        DeviceAction.TURN_OFF.value,
        DeviceAction.QUERY.value,
    },
    "climate": {
        DeviceAction.SET_TEMPERATURE.value,
        DeviceAction.TURN_ON.value,
        DeviceAction.TURN_OFF.value,
        DeviceAction.QUERY.value,
    },
    "lock": {
        DeviceAction.LOCK.value,
        DeviceAction.UNLOCK.value,
        DeviceAction.QUERY.value,
    },
    "fan": {
        DeviceAction.TURN_ON.value,
        DeviceAction.TURN_OFF.value,
        DeviceAction.TOGGLE.value,
        DeviceAction.SET_SPEED.value,
        DeviceAction.QUERY.value,
    },
    "media_player": {
        DeviceAction.SET_VOLUME.value,
        DeviceAction.VOLUME_UP.value,
        DeviceAction.VOLUME_DOWN.value,
        DeviceAction.MUTE.value,
        DeviceAction.UNMUTE.value,
        DeviceAction.TURN_ON.value,
        DeviceAction.TURN_OFF.value,
        DeviceAction.QUERY.value,
    },
    "system": {
        DeviceAction.LAUNCH_APP.value,
        DeviceAction.QUERY.value,
        DeviceAction.TURN_ON.value,
        DeviceAction.TURN_OFF.value,
    },
}


class DeviceRegistry:
    """In-memory registry managing smart device entities and executing commands."""

    def __init__(
        self,
        event_bus: Optional[EventBus] = None,
        seed_defaults: bool = True,
    ) -> None:
        self.event_bus = event_bus
        self._devices: dict[str, DeviceEntity] = {}
        self._parser = DeviceIntentParser(registry=self)
        if seed_defaults:
            self.seed_default_entities()

    def seed_default_entities(self) -> None:
        """Seed default smart devices for testing and out-of-the-box operations."""
        defaults = [
            DeviceEntity("light.living_room", "light", "Living Room Light", "off", {"brightness": 100}),
            DeviceEntity("light.workshop", "light", "Workshop Light", "on", {"brightness": 80}),
            DeviceEntity("light.office", "light", "Office Light", "on", {"brightness": 100}),
            DeviceEntity("switch.main_power", "switch", "Main Power", "on", {}),
            DeviceEntity(
                "thermostat.main",
                "thermostat",
                "Main Thermostat",
                "cool",
                {"temperature": 21.0, "target_temperature": 21.0, "min_temp": 16.0, "max_temp": 30.0},
            ),
            DeviceEntity("lock.front_door", "lock", "Front Door", "locked", {}),
            DeviceEntity("fan.bedroom", "fan", "Bedroom Fan", "off", {"speed": 0}),
            DeviceEntity(
                "media_player.system",
                "media_player",
                "System Audio",
                "playing",
                {"volume": 0.75, "muted": False},
            ),
            DeviceEntity(
                "system.arc_reactor",
                "system",
                "Arc Reactor Core",
                "nominal",
                {"power_level": 100.0, "status": "online"},
            ),
        ]
        for d in defaults:
            self._devices[d.entity_id] = d

    def register(self, entity: DeviceEntity) -> None:
        """Register a new device entity or update existing."""
        self._devices[entity.entity_id] = entity
        logger.debug("Registered device entity: %s", entity.entity_id)

    def unregister(self, entity_id: str) -> bool:
        """Remove a device entity from the registry. Returns True if removed, False otherwise."""
        if entity_id in self._devices:
            del self._devices[entity_id]
            logger.debug("Unregistered device entity: %s", entity_id)
            return True
        return False

    def get(self, entity_id: str) -> Optional[DeviceEntity]:
        """Retrieve a device entity by its entity_id."""
        return self._devices.get(entity_id)

    def list_all(self, domain: Optional[str] = None) -> list[DeviceEntity]:
        """List all devices, optionally filtered by domain."""
        if domain is None:
            return list(self._devices.values())
        return [d for d in self._devices.values() if d.domain == domain]

    def count(self, domain: Optional[str] = None) -> int:
        """Return count of registered devices."""
        return len(self.list_all(domain))

    def clear(self) -> None:
        """Clear all registered devices."""
        self._devices.clear()

    def update_state(
        self,
        entity_id: str,
        state: str,
        attributes: Optional[dict[str, Any]] = None,
    ) -> DeviceEntity:
        """Update the state and traits of a registered entity and dispatch event."""
        if entity_id not in self._devices:
            raise DeviceNotFoundError(f"Device {entity_id} not found in registry")

        dev = self._devices[entity_id]
        dev.state = state
        if attributes:
            dev.attributes.update(attributes)
        dev.last_updated = time.time()

        self._publish_state_changed(entity_id, dev)
        return dev

    def execute_command(self, entity_id: str, action: str, **params: Any) -> dict[str, Any]:
        """Execute a state change command on a device entity and publish event."""
        if entity_id not in self._devices:
            return {
                "success": False,
                "error": f"Device {entity_id} not found",
                "entity_id": entity_id,
                "device_id": entity_id,
            }

        dev = self._devices[entity_id]

        # Domain action compatibility check
        allowed = _DOMAIN_ALLOWED_ACTIONS.get(dev.domain)
        if allowed is not None and action not in allowed:
            return {
                "success": False,
                "error": f"Action '{action}' is not supported for domain '{dev.domain}'",
                "entity_id": entity_id,
                "device_id": entity_id,
            }

        if action == DeviceAction.TURN_ON.value:
            dev.state = "on"
            if "brightness" in params:
                dev.attributes["brightness"] = max(1, min(100, int(params["brightness"])))
        elif action == DeviceAction.TURN_OFF.value:
            dev.state = "off"
        elif action == DeviceAction.TOGGLE.value:
            dev.state = "off" if dev.state == "on" else "on"
        elif action == DeviceAction.SET_BRIGHTNESS.value:
            raw_b = params.get("brightness", 100)
            dev.attributes["brightness"] = max(1, min(100, int(raw_b)))
            dev.state = "on"
        elif action == DeviceAction.SET_TEMPERATURE.value:
            raw_t = params.get("temperature", 21.0)
            t = max(16.0, min(30.0, float(raw_t)))
            dev.attributes["temperature"] = t
            dev.attributes["target_temperature"] = t
        elif action == DeviceAction.LOCK.value:
            dev.state = "locked"
        elif action == DeviceAction.UNLOCK.value:
            dev.state = "unlocked"
        elif action == DeviceAction.SET_VOLUME.value:
            raw_v = float(params.get("volume", 0.5))
            vol = raw_v / 100.0 if raw_v > 1.0 else raw_v
            dev.attributes["volume"] = max(0.0, min(1.0, round(vol, 2)))
        elif action == DeviceAction.VOLUME_UP.value:
            step = float(params.get("step", 0.1))
            current = float(dev.attributes.get("volume", 0.5))
            dev.attributes["volume"] = min(1.0, round(current + step, 2))
        elif action == DeviceAction.VOLUME_DOWN.value:
            step = float(params.get("step", 0.1))
            current = float(dev.attributes.get("volume", 0.5))
            dev.attributes["volume"] = max(0.0, round(current - step, 2))
        elif action == DeviceAction.MUTE.value:
            dev.attributes["muted"] = True
            dev.attributes["volume"] = 0.0
        elif action == DeviceAction.UNMUTE.value:
            dev.attributes["muted"] = False
        elif action == DeviceAction.LAUNCH_APP.value:
            dev.attributes["last_launched"] = params.get("app_name", "app")
            dev.state = "launched"
        elif action == DeviceAction.SET_SPEED.value:
            dev.attributes["speed"] = int(params.get("speed", 1))
        elif action == DeviceAction.QUERY.value:
            pass  # Read-only query

        # Merge any additional params
        for k, v in params.items():
            if k not in ("brightness", "temperature", "volume", "step", "speed", "app_name"):
                dev.attributes[k] = v

        dev.last_updated = time.time()
        self._publish_state_changed(entity_id, dev)

        return {
            "success": True,
            "entity_id": entity_id,
            "device_id": entity_id,
            "state": dev.state,
            "attributes": dict(dev.attributes),
            "action": action,
        }

    def _publish_state_changed(self, entity_id: str, dev: DeviceEntity) -> None:
        """Safely schedule event dispatch on the event bus."""
        if not self.event_bus:
            return
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(
                self.event_bus.publish_event(
                    EventType.DEVICE_STATE_CHANGED,
                    device_id=entity_id,
                    entity_id=entity_id,
                    state=dev.state,
                    traits=dict(dev.attributes),
                    timestamp=dev.last_updated,
                )
            )
        except RuntimeError:
            pass  # No running event loop in synchronous test

    def parse_natural_command(self, utterance: str) -> Tuple[str, str, dict[str, Any]]:
        """Convenience bridge method delegating to DeviceIntentParser."""
        return self._parser.parse_natural_command(utterance)

    parse_command = parse_natural_command
