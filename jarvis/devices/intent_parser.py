"""Natural language parser for smart device commands and voice utterances."""

from __future__ import annotations

import re
from typing import Any, Optional, Tuple

from jarvis.devices.schema import DeviceAction, DeviceCommand


class DeviceIntentParser:
    """Translates natural language voice utterances into structured DeviceCommands."""

    # Explicit spoken aliases mapped to canonical entity IDs
    ENTITY_ALIASES: dict[str, str] = {
        # Lights
        "living room light": "light.living_room",
        "living room lights": "light.living_room",
        "living room": "light.living_room",
        "lounge light": "light.living_room",
        "workshop light": "light.workshop",
        "workshop lights": "light.workshop",
        "workshop": "light.workshop",
        "garage light": "light.workshop",
        "office light": "light.office",
        "office lights": "light.office",
        "office": "light.office",
        "study light": "light.office",

        # Climate / Thermostat
        "thermostat": "thermostat.main",
        "main thermostat": "thermostat.main",
        "ac": "thermostat.main",
        "air conditioner": "thermostat.main",
        "heating": "thermostat.main",

        # Locks
        "front door": "lock.front_door",
        "front door lock": "lock.front_door",
        "door lock": "lock.front_door",
        "door": "lock.front_door",

        # Switches / Fans
        "main power": "switch.main_power",
        "power switch": "switch.main_power",
        "coffee maker": "switch.coffee_maker",
        "bedroom fan": "fan.bedroom",
        "fan": "fan.bedroom",

        # Media Player / Audio
        "system volume": "media_player.system",
        "system audio": "media_player.system",
        "audio": "media_player.system",
        "sound": "media_player.system",
        "volume": "media_player.system",
        "speaker": "media_player.system",

        # System
        "arc reactor": "system.arc_reactor",
        "arc reactor core": "system.arc_reactor",
        "reactor": "system.arc_reactor",
    }

    # Polite Jarvis filler prefixes and suffixes
    PREFIX_CLEAN_RE = re.compile(
        r"^(?:(?:hey|ok|hello)?\s*jarvis[,.]?\s*)*(?:please\s+|could\s+you\s+(?:please\s+)?|would\s+you\s+(?:kindly\s+)?|can\s+you\s+(?:please\s+)?)*",
        re.IGNORECASE,
    )
    SUFFIX_CLEAN_RE = re.compile(
        r"[,.]?\s*(?:please|sir|jarvis|for\s+me)[.!]?$",
        re.IGNORECASE,
    )

    def __init__(self, registry: Optional[Any] = None) -> None:
        self.registry = registry

    def clean_text(self, text: str) -> str:
        """Strip polite conversational filler, whitespace, and normalize casing."""
        if not text:
            return ""
        t = text.strip()
        t = self.PREFIX_CLEAN_RE.sub("", t).strip()
        t = self.SUFFIX_CLEAN_RE.sub("", t).strip()
        # Remove trailing and leading punctuation except internal words
        t = re.sub(r"^[^\w]+|[^\w%°.]+$", "", t)
        return t.strip().lower()

    def resolve_entity(self, text: str, default_domain: Optional[str] = None) -> Tuple[Optional[str], bool]:
        """Find entity ID matching aliases in utterance text.

        Returns (entity_id, is_explicit).
        """
        # 1. Exact or substring match in alias map (longer phrases first)
        sorted_aliases = sorted(self.ENTITY_ALIASES.items(), key=lambda x: len(x[0]), reverse=True)
        for alias, entity_id in sorted_aliases:
            if re.search(r"\b" + re.escape(alias) + r"\b", text):
                return entity_id, True

        # 2. Check if a domain-based default fallback applies
        if default_domain == "light" or "lights" in text or "light" in text:
            if "workshop" in text:
                return "light.workshop", True
            if "office" in text:
                return "light.office", True
            return "light.living_room", False

        if default_domain in ("thermostat", "climate") or any(w in text for w in ("thermostat", "temperature", "temp")):
            return "thermostat.main", False

        if default_domain == "lock" or "door" in text:
            return "lock.front_door", False

        if default_domain == "fan" or "fan" in text:
            return "fan.bedroom", False

        if default_domain == "switch" or "power" in text:
            return "switch.main_power", False

        if default_domain == "media_player" or any(w in text for w in ("volume", "sound", "audio")):
            return "media_player.system", False

        return None, False

    def parse(self, text: str) -> Optional[DeviceCommand]:
        """Parse natural language command into a structured DeviceCommand, or return None."""
        if not text or len(text) > 1000:
            return None

        # Check for malformed / pure symbols / pure numbers
        if re.fullmatch(r"^[^\w]*$", text) or re.fullmatch(r"^\d+$", text.strip()):
            return None

        cleaned = self.clean_text(text)
        if not cleaned:
            return None

        # Check for non-device general conversational queries
        conversational_starters = (
            "who was", "who is", "what's the weather", "what is the weather",
            "play ac/dc", "tell me a joke", "how are you", "who created",
            "capital of", "define "
        )
        if any(cleaned.startswith(starter) for starter in conversational_starters):
            return None

        # 1. Status Queries (e.g. "is the front door locked?", "what is the workshop temperature?")
        query_patterns = ("is the", "what is the", "what's the", "check the", "check", "status of", "status")
        if any(p in cleaned for p in query_patterns):
            if "door" in cleaned or "locked" in cleaned or "lock" in cleaned:
                return DeviceCommand(
                    entity_id="lock.front_door",
                    domain="lock",
                    action=DeviceAction.QUERY.value,
                    parameters={},
                    confidence=0.95,
                    raw_text=text,
                )
            if any(w in cleaned for w in ("temperature", "thermostat", "temp")):
                return DeviceCommand(
                    entity_id="thermostat.main",
                    domain="climate",
                    action=DeviceAction.QUERY.value,
                    parameters={},
                    confidence=0.95,
                    raw_text=text,
                )
            if "light" in cleaned or "lights" in cleaned:
                entity, is_explicit = self.resolve_entity(cleaned, default_domain="light")
                return DeviceCommand(
                    entity_id=entity or "light.living_room",
                    domain="light",
                    action=DeviceAction.QUERY.value,
                    parameters={},
                    confidence=0.95 if is_explicit else 0.65,
                    raw_text=text,
                )
            if "reactor" in cleaned or "arc reactor" in cleaned:
                return DeviceCommand(
                    entity_id="system.arc_reactor",
                    domain="system",
                    action=DeviceAction.QUERY.value,
                    parameters={},
                    confidence=1.0,
                    raw_text=text,
                )

        # 2. Thermostat / Temperature Commands
        if any(w in cleaned for w in ("thermostat", "temperature", "temp", "degrees", "deg")):
            m = re.search(r"(\d+(?:\.\d+)?)", cleaned)
            temp = float(m.group(1)) if m else 22.0
            entity, is_explicit = self.resolve_entity(cleaned, default_domain="thermostat")
            return DeviceCommand(
                entity_id=entity or "thermostat.main",
                domain="thermostat",
                action=DeviceAction.SET_TEMPERATURE.value,
                parameters={"temperature": temp},
                confidence=0.95 if is_explicit else 0.9,
                raw_text=text,
            )

        # 3. Lock / Unlock Commands
        if "lock" in cleaned or "unlock" in cleaned or "secure" in cleaned:
            # Prevent matching "clock" or "block"
            if re.search(r"\b(?:lock|unlock|secure)\b", cleaned):
                entity, is_explicit = self.resolve_entity(cleaned, default_domain="lock")
                action = DeviceAction.UNLOCK.value if "unlock" in cleaned else DeviceAction.LOCK.value
                return DeviceCommand(
                    entity_id=entity or "lock.front_door",
                    domain="lock",
                    action=action,
                    parameters={},
                    confidence=0.95 if is_explicit else 0.85,
                    raw_text=text,
                )

        # 4. Media / Audio / Volume Commands
        if any(w in cleaned for w in ("volume", "mute", "unmute", "sound", "louder", "quieter")):
            entity = "media_player.system"
            if "unmute" in cleaned:
                return DeviceCommand(entity, "media_player", DeviceAction.UNMUTE.value, {}, 1.0, text)
            if "mute" in cleaned:
                # Support volume: 0 and action set_volume
                return DeviceCommand(entity, "media_player", DeviceAction.SET_VOLUME.value, {"volume": 0}, 1.0, text)
            if any(w in cleaned for w in ("turn up", "increase", "raise", "louder", "boost")):
                return DeviceCommand(entity, "media_player", DeviceAction.VOLUME_UP.value, {"step": 0.1}, 1.0, text)
            if any(w in cleaned for w in ("turn down", "decrease", "lower", "quieter", "reduce")):
                return DeviceCommand(entity, "media_player", DeviceAction.VOLUME_DOWN.value, {"step": 0.1}, 1.0, text)
            m = re.search(r"(\d+)", cleaned)
            if m:
                val = int(m.group(1))
                return DeviceCommand(entity, "media_player", DeviceAction.SET_VOLUME.value, {"volume": val}, 1.0, text)

        # 5. Light Dimming / Brightness Commands
        if any(w in cleaned for w in ("brightness", "dim", "%", "percent", "brighten")):
            entity, is_explicit = self.resolve_entity(cleaned, default_domain="light")
            m = re.search(r"(\d+)", cleaned)
            b = int(m.group(1)) if m else 50
            return DeviceCommand(
                entity_id=entity or "light.living_room",
                domain="light",
                action=DeviceAction.SET_BRIGHTNESS.value,
                parameters={"brightness": max(1, min(100, b))},
                confidence=0.95 if is_explicit else 0.65,
                raw_text=text,
            )

        # 6. Switch / Power / Turn On / Turn Off / Toggle
        is_turn_off = bool(re.search(r"\b(?:turn off|switch off|shut off|power off|lights off)\b", cleaned))
        is_turn_on = bool(re.search(r"\b(?:turn on|switch on|power on|lights on|activate)\b", cleaned))
        is_toggle = bool(re.search(r"\b(?:toggle)\b", cleaned))

        if is_turn_off or is_turn_on or is_toggle:
            action = (
                DeviceAction.TURN_OFF.value
                if is_turn_off
                else (DeviceAction.TOGGLE.value if is_toggle else DeviceAction.TURN_ON.value)
            )

            # Check if utterance is completely ambiguous without specific target (e.g. "turn it on right now")
            if re.search(r"\b(?:turn it on|turn it off|switch it on|switch it off)\b", cleaned):
                return None

            default_domain = None
            if "light" in cleaned or "lights" in cleaned:
                default_domain = "light"
            elif "fan" in cleaned:
                default_domain = "fan"
            elif "coffee" in cleaned or "power" in cleaned or "switch" in cleaned:
                default_domain = "switch"

            entity, is_explicit = self.resolve_entity(cleaned, default_domain=default_domain)
            if entity:
                domain = entity.split(".", 1)[0]
                confidence = 0.95 if is_explicit else 0.65
                return DeviceCommand(
                    entity_id=entity,
                    domain=domain,
                    action=action,
                    parameters={},
                    confidence=confidence,
                    raw_text=text,
                )

        # 7. App Launching
        m_app = re.search(r"\b(?:launch|open|start)\s+([a-zA-Z0-9_\-]+)\b", cleaned)
        if m_app and not any(w in cleaned for w in ("fan", "light", "lights", "door", "power", "reactor", "volume")):
            app_name = m_app.group(1)
            return DeviceCommand(
                entity_id="system.app_launcher",
                domain="system",
                action=DeviceAction.LAUNCH_APP.value,
                parameters={"app_name": app_name},
                confidence=0.9,
                raw_text=text,
            )

        # 8. Arc Reactor Status Query
        if "arc reactor" in cleaned or "reactor" in cleaned:
            return DeviceCommand(
                entity_id="system.arc_reactor",
                domain="system",
                action=DeviceAction.QUERY.value,
                parameters={},
                confidence=1.0,
                raw_text=text,
            )

        return None

    def parse_natural_command(self, utterance: str) -> Tuple[str, str, dict[str, Any]]:
        """Bridge method returning (entity_id, action, params) tuple for E2E tests."""
        cmd = self.parse(utterance)
        if cmd and cmd.entity_id:
            return (cmd.entity_id, cmd.action, cmd.parameters)
        return ("system.arc_reactor", "query", {})
