"""Data schemas, enums, and exceptions for smart device entities and commands."""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional


class DeviceDomain(str, Enum):
    """Supported smart home entity domains."""

    LIGHT = "light"
    SWITCH = "switch"
    THERMOSTAT = "thermostat"
    CLIMATE = "climate"
    LOCK = "lock"
    FAN = "fan"
    MEDIA_PLAYER = "media_player"
    SYSTEM = "system"


class DeviceState(str, Enum):
    """Canonical states for device entities."""

    ON = "on"
    OFF = "off"
    LOCKED = "locked"
    UNLOCKED = "unlocked"
    COOL = "cool"
    HEAT = "heat"
    IDLE = "idle"
    NOMINAL = "nominal"
    PLAYING = "playing"
    PAUSED = "paused"
    ERROR = "error"
    OFFLINE = "offline"
    LAUNCHED = "launched"


class DeviceAction(str, Enum):
    """Canonical actions executable on devices."""

    TURN_ON = "turn_on"
    TURN_OFF = "turn_off"
    TOGGLE = "toggle"
    SET_BRIGHTNESS = "set_brightness"
    SET_TEMPERATURE = "set_temperature"
    LOCK = "lock"
    UNLOCK = "unlock"
    SET_VOLUME = "set_volume"
    VOLUME_UP = "volume_up"
    VOLUME_DOWN = "volume_down"
    MUTE = "mute"
    UNMUTE = "unmute"
    LAUNCH_APP = "launch_app"
    QUERY = "query"
    SET_SPEED = "set_speed"


class DeviceError(Exception):
    """Base exception for smart device errors."""


class DeviceNotFoundError(DeviceError, KeyError):
    """Raised when an entity ID is not found in the registry."""


class DeviceOfflineError(DeviceError):
    """Raised when actuation is attempted on an unreachable device."""


class InvalidDeviceActionError(DeviceError, ValueError):
    """Raised when an unsupported action is requested for an entity domain."""


_ENTITY_ID_RE = re.compile(r"^[a-z0-9_]+\.[a-z0-9_]+$")


@dataclass
class DeviceEntity:
    """HomeAssistant-compatible entity model representing a smart device."""

    entity_id: str                              # e.g., "light.living_room"
    domain: str = ""                            # "light", "switch", "thermostat", "lock", etc.
    name: str = ""                              # Human-readable name
    state: str = "off"                          # "on", "off", "locked", etc.
    attributes: dict[str, Any] = field(default_factory=dict)
    last_updated: float = field(default_factory=time.time)

    def __post_init__(self) -> None:
        if not self.entity_id or not _ENTITY_ID_RE.match(self.entity_id):
            raise ValueError(
                f"Invalid entity_id format: '{self.entity_id}'. "
                "Must be 'domain.object_id' with lowercase alphanumeric and underscore characters."
            )

        prefix = self.entity_id.split(".", 1)[0]
        if self.domain:
            if self.domain != prefix:
                raise ValueError(
                    f"Domain mismatch: entity_id prefix '{prefix}' does not match specified domain '{self.domain}'"
                )
        else:
            self.domain = prefix

        if not self.name or not self.name.strip():
            raise ValueError("Device name cannot be empty or whitespace.")

        if self.attributes is None:
            self.attributes = {}

    def to_dict(self) -> dict[str, Any]:
        """Convert entity to a JSON-serializable dictionary."""
        return {
            "entity_id": self.entity_id,
            "domain": self.domain,
            "name": self.name,
            "state": self.state,
            "attributes": dict(self.attributes),
            "last_updated": self.last_updated,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DeviceEntity:
        """Instantiate DeviceEntity from a dictionary."""
        return cls(
            entity_id=data["entity_id"],
            domain=data.get("domain", data["entity_id"].split(".", 1)[0]),
            name=data.get("name", data["entity_id"]),
            state=data.get("state", "off"),
            attributes=dict(data.get("attributes", {})),
            last_updated=data.get("last_updated", time.time()),
        )


@dataclass
class DeviceCommand:
    """Structured representation of a parsed natural language command."""

    entity_id: Optional[str] = None
    domain: Optional[str] = None
    action: str = ""
    parameters: dict[str, Any] = field(default_factory=dict)
    confidence: float = 1.0
    raw_text: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Convert command to a dictionary."""
        return {
            "entity_id": self.entity_id,
            "domain": self.domain,
            "action": self.action,
            "parameters": dict(self.parameters),
            "confidence": self.confidence,
            "raw_text": self.raw_text,
        }
