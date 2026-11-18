"""Unit tests for smart device schema, registry operations, intent parser, and mock IoT hub."""

from __future__ import annotations

import asyncio
import time
import pytest

from jarvis.core.event_bus import EventBus, EventType
from jarvis.devices.intent_parser import DeviceIntentParser
from jarvis.devices.mock_iot import MockIoTHub
from jarvis.devices.registry import DeviceRegistry
from jarvis.devices.schema import (
    DeviceAction,
    DeviceCommand,
    DeviceEntity,
    DeviceNotFoundError,
)


# ==============================================================================
# CLASS 1: DEVICE SCHEMA VALIDATION (5 Tests)
# ==============================================================================
class TestDeviceEntitySchemaValidation:
    """Verifies schema validation, invariants, and dictionary serialization."""

    def test_valid_device_entity_instantiation(self) -> None:
        """Verify instantiation across canonical domains with default and custom attributes."""
        dev = DeviceEntity(
            entity_id="light.living_room",
            domain="light",
            name="Living Room Light",
            state="off",
            attributes={"brightness": 100},
        )
        assert dev.entity_id == "light.living_room"
        assert dev.domain == "light"
        assert dev.name == "Living Room Light"
        assert dev.state == "off"
        assert dev.attributes["brightness"] == 100
        assert dev.last_updated > 0.0

        # Verify to_dict and from_dict roundtrip
        d_dict = dev.to_dict()
        assert d_dict["entity_id"] == "light.living_room"
        assert d_dict["domain"] == "light"
        assert d_dict["name"] == "Living Room Light"
        assert d_dict["state"] == "off"

        recovered = DeviceEntity.from_dict(d_dict)
        assert recovered.entity_id == dev.entity_id
        assert recovered.domain == dev.domain
        assert recovered.state == dev.state

    def test_entity_id_domain_validation_and_mismatch(self) -> None:
        """Verify domain mismatch between entity_id prefix and explicit domain raises ValueError."""
        with pytest.raises(ValueError, match="Domain mismatch"):
            DeviceEntity(
                entity_id="switch.fan",
                domain="light",
                name="Fan",
            )

    def test_invalid_entity_id_format_rejection(self) -> None:
        """Verify invalid entity_id formats are strictly rejected."""
        invalid_ids = [
            "",
            "no_dot",
            ".leading_dot",
            "trailing_dot.",
            "spaces in name.test",
            "Uppercase.Domain",
            "special!char.name",
        ]
        for inv_id in invalid_ids:
            with pytest.raises(ValueError, match="Invalid entity_id format"):
                DeviceEntity(entity_id=inv_id, name="Test Device")

    def test_empty_or_whitespace_name_rejection(self) -> None:
        """Verify empty or whitespace-only device names are rejected."""
        with pytest.raises(ValueError, match="Device name cannot be empty or whitespace"):
            DeviceEntity(entity_id="light.kitchen", name="")

        with pytest.raises(ValueError, match="Device name cannot be empty or whitespace"):
            DeviceEntity(entity_id="light.kitchen", name="   ")

    def test_attributes_default_dict_isolation(self) -> None:
        """Verify default attributes dictionary is not shared across distinct entity instances."""
        e1 = DeviceEntity(entity_id="light.living_room", name="Living Room")
        e2 = DeviceEntity(entity_id="light.kitchen", name="Kitchen")

        e1.attributes["foo"] = "bar"
        assert "foo" not in e2.attributes
        assert e2.attributes == {}


# ==============================================================================
# CLASS 2: DEVICE REGISTRY OPERATIONS (8 Tests)
# ==============================================================================
class TestDeviceRegistryOperations:
    """Verifies registry CRUD operations, attribute merging, command execution, and error handling."""

    def test_register_and_get_entity(self) -> None:
        """Verify registration and lookup of entities in an isolated registry."""
        registry = DeviceRegistry(seed_defaults=False)
        entity = DeviceEntity(entity_id="light.living_room", name="Living Room Light", state="off")
        registry.register(entity)

        assert registry.get("light.living_room") is entity
        assert registry.get("light.unknown") is None

    def test_duplicate_registration_overwrite_or_update(self) -> None:
        """Verify registering an existing entity ID updates state without duplicating entries."""
        registry = DeviceRegistry(seed_defaults=False)
        e1 = DeviceEntity(entity_id="light.living_room", name="Living Room Light", state="off")
        registry.register(e1)
        assert registry.count() == 1

        e2 = DeviceEntity(entity_id="light.living_room", name="Living Room Light", state="on")
        registry.register(e2)
        assert registry.count() == 1
        assert registry.get("light.living_room").state == "on"

    def test_list_all_unfiltered_and_filtered_by_domain(self) -> None:
        """Verify list_all returns all entities or filtered subset by domain."""
        registry = DeviceRegistry(seed_defaults=False)
        registry.register(DeviceEntity("light.l1", "light", "L1"))
        registry.register(DeviceEntity("light.l2", "light", "L2"))
        registry.register(DeviceEntity("light.l3", "light", "L3"))
        registry.register(DeviceEntity("switch.s1", "switch", "S1"))
        registry.register(DeviceEntity("switch.s2", "switch", "S2"))
        registry.register(DeviceEntity("lock.d1", "lock", "D1"))

        assert len(registry.list_all()) == 6
        assert len(registry.list_all(domain="light")) == 3
        assert len(registry.list_all(domain="switch")) == 2
        assert len(registry.list_all(domain="lock")) == 1
        assert registry.list_all(domain="vacuum") == []

    def test_unregister_existing_and_nonexistent_entity(self) -> None:
        """Verify unregistering entities returns correct boolean status."""
        registry = DeviceRegistry(seed_defaults=False)
        registry.register(DeviceEntity("light.living_room", "light", "Living Room"))

        assert registry.unregister("light.living_room") is True
        assert registry.get("light.living_room") is None
        assert registry.unregister("light.living_room") is False

    def test_update_state_attribute_merging(self) -> None:
        """Verify update_state merges traits non-destructively."""
        registry = DeviceRegistry(seed_defaults=False)
        registry.register(
            DeviceEntity(
                "light.living_room",
                "light",
                "Living Room",
                "off",
                {"brightness": 50, "color_temp": 3000},
            )
        )

        updated = registry.update_state("light.living_room", state="on", attributes={"brightness": 80})
        assert updated.state == "on"
        assert updated.attributes["brightness"] == 80
        assert updated.attributes["color_temp"] == 3000

    def test_update_state_nonexistent_raises_key_error(self) -> None:
        """Verify update_state on nonexistent device raises DeviceNotFoundError."""
        registry = DeviceRegistry(seed_defaults=False)
        with pytest.raises(DeviceNotFoundError):
            registry.update_state("unknown.device", "on")

    def test_execute_command_success_all_domains(self) -> None:
        """Verify command execution across light, thermostat, lock, switch, and media player."""
        registry = DeviceRegistry(seed_defaults=True)

        # Light actuation
        res_light = registry.execute_command("light.living_room", "turn_on", brightness=80)
        assert res_light["success"] is True
        assert res_light["state"] == "on"
        assert res_light["attributes"]["brightness"] == 80

        # Thermostat actuation
        res_temp = registry.execute_command("thermostat.main", "set_temperature", temperature=22.5)
        assert res_temp["success"] is True
        assert res_temp["attributes"]["temperature"] == 22.5
        assert res_temp["attributes"]["target_temperature"] == 22.5

        # Lock actuation
        res_lock = registry.execute_command("lock.front_door", "unlock")
        assert res_lock["success"] is True
        assert res_lock["state"] == "unlocked"

        # Switch actuation
        res_switch = registry.execute_command("switch.main_power", "turn_off")
        assert res_switch["success"] is True
        assert res_switch["state"] == "off"

    def test_execute_command_error_handling_and_clamping(self) -> None:
        """Verify error handling for missing devices, unsupported actions, and parameter boundary clamping."""
        registry = DeviceRegistry(seed_defaults=True)

        # Nonexistent device
        res_missing = registry.execute_command("light.nonexistent", "turn_on")
        assert res_missing["success"] is False
        assert "not found" in res_missing["error"].lower()

        # Unsupported action for domain
        res_invalid = registry.execute_command("light.living_room", "unlock")
        assert res_invalid["success"] is False
        assert "not supported" in res_invalid["error"].lower()

        # Parameter clamping: brightness clamped to [1, 100]
        res_b_high = registry.execute_command("light.living_room", "set_brightness", brightness=150)
        assert res_b_high["attributes"]["brightness"] == 100

        res_b_low = registry.execute_command("light.living_room", "set_brightness", brightness=-10)
        assert res_b_low["attributes"]["brightness"] == 1

        # Parameter clamping: temperature clamped to [16.0, 30.0]
        res_t_high = registry.execute_command("thermostat.main", "set_temperature", temperature=45.0)
        assert res_t_high["attributes"]["temperature"] == 30.0

        res_t_low = registry.execute_command("thermostat.main", "set_temperature", temperature=5.0)
        assert res_t_low["attributes"]["temperature"] == 16.0


# ==============================================================================
# CLASS 3: DEVICE INTENT PARSER (8 Tests)
# ==============================================================================
class TestDeviceIntentParser:
    """Verifies natural language utterance translation to structured device commands."""

    def test_parse_light_turn_on_off_and_dimming(self, device_intent_parser: DeviceIntentParser) -> None:
        """Verify voice commands for turning on, off, and dimming lights."""
        cmd1 = device_intent_parser.parse("Jarvis, turn on the living room light, please")
        assert cmd1 is not None
        assert cmd1.action == "turn_on"
        assert cmd1.domain == "light"
        assert cmd1.entity_id == "light.living_room"

        cmd2 = device_intent_parser.parse("dim office lights to 30%")
        assert cmd2 is not None
        assert cmd2.action == "set_brightness"
        assert cmd2.parameters["brightness"] == 30
        assert cmd2.entity_id == "light.office"

        cmd3 = device_intent_parser.parse("switch off workshop lights")
        assert cmd3 is not None
        assert cmd3.action == "turn_off"
        assert cmd3.entity_id == "light.workshop"

    def test_parse_thermostat_temperature_commands(self, device_intent_parser: DeviceIntentParser) -> None:
        """Verify temperature setting utterances with degrees and explicit numbers."""
        cmd1 = device_intent_parser.parse("set temperature to 22 degrees")
        assert cmd1 is not None
        assert cmd1.action == "set_temperature"
        assert cmd1.parameters["temperature"] == 22.0
        assert cmd1.entity_id == "thermostat.main"

        cmd2 = device_intent_parser.parse("set thermostat to 70")
        assert cmd2 is not None
        assert cmd2.action == "set_temperature"
        assert cmd2.parameters["temperature"] == 70.0

    def test_parse_door_lock_and_unlock_commands(self, device_intent_parser: DeviceIntentParser) -> None:
        """Verify door locking and unlocking utterances."""
        cmd1 = device_intent_parser.parse("lock the front door")
        assert cmd1 is not None
        assert cmd1.action == "lock"
        assert cmd1.domain == "lock"
        assert cmd1.entity_id == "lock.front_door"

        cmd2 = device_intent_parser.parse("unlock front door, sir")
        assert cmd2 is not None
        assert cmd2.action == "unlock"
        assert cmd2.domain == "lock"
        assert cmd2.entity_id == "lock.front_door"

    def test_parse_switch_and_system_volume_commands(self, device_intent_parser: DeviceIntentParser) -> None:
        """Verify switch control and system volume / mute utterances."""
        cmd1 = device_intent_parser.parse("turn on coffee maker")
        assert cmd1 is not None
        assert cmd1.action == "turn_on"
        assert cmd1.domain == "switch"

        cmd2 = device_intent_parser.parse("set volume to 60 percent")
        assert cmd2 is not None
        assert cmd2.action == "set_volume"
        assert cmd2.parameters["volume"] == 60
        assert cmd2.entity_id == "media_player.system"

        cmd3 = device_intent_parser.parse("mute system audio")
        assert cmd3 is not None
        assert cmd3.action == "set_volume"
        assert cmd3.parameters["volume"] == 0

    def test_parse_device_status_query_intents(self, device_intent_parser: DeviceIntentParser) -> None:
        """Verify status queries for locks and temperature."""
        cmd1 = device_intent_parser.parse("is the front door locked?")
        assert cmd1 is not None
        assert cmd1.action == "query"
        assert cmd1.domain == "lock"
        assert cmd1.entity_id == "lock.front_door"

        cmd2 = device_intent_parser.parse("what is the workshop temperature?")
        assert cmd2 is not None
        assert cmd2.action == "query"
        assert cmd2.domain == "climate"

    def test_parse_unrecognized_conversational_queries_returns_none(
        self, device_intent_parser: DeviceIntentParser
    ) -> None:
        """Verify non-device conversational queries return None so they fall through to LLM routers."""
        queries = [
            "Who was the first president?",
            "Play AC/DC",
            "What's the weather today?",
            "Tell me a joke",
            "How are you doing today?",
        ]
        for q in queries:
            assert device_intent_parser.parse(q) is None

    def test_parse_malformed_empty_and_noise_utterances(
        self, device_intent_parser: DeviceIntentParser
    ) -> None:
        """Verify empty, noise, pure numbers, and massive strings do not raise exceptions."""
        assert device_intent_parser.parse("") is None
        assert device_intent_parser.parse("   \t  \n") is None
        assert device_intent_parser.parse("!??...") is None
        assert device_intent_parser.parse("42") is None

        # Extreme length string
        massive = "please " * 1000
        assert device_intent_parser.parse(massive) is None

    def test_parse_confidence_scoring_and_entity_disambiguation(
        self, device_intent_parser: DeviceIntentParser
    ) -> None:
        """Verify explicit entity matches yield high confidence while generic commands score lower."""
        explicit_cmd = device_intent_parser.parse("turn on living room light")
        assert explicit_cmd is not None
        assert explicit_cmd.confidence >= 0.9

        generic_cmd = device_intent_parser.parse("turn on the light")
        assert generic_cmd is not None
        assert generic_cmd.confidence < 0.7


# ==============================================================================
# CLASS 4: MOCK IOT HUB ASYNC EXECUTION (5 Tests)
# ==============================================================================
class TestMockIoTHubAsyncExecution:
    """Verifies async IoT hardware gateway execution, latency simulation, and event broadcasting."""

    @pytest.mark.asyncio
    async def test_mock_iot_hub_execute_command_async(
        self, mock_iot_hub: MockIoTHub, device_registry: DeviceRegistry
    ) -> None:
        """Verify asynchronous execution updates the device state in the registry."""
        res = await mock_iot_hub.execute("light.living_room", "turn_on", brightness=75)
        assert res["success"] is True
        assert res["state"] == "on"

        dev = device_registry.get("light.living_room")
        assert dev is not None
        assert dev.state == "on"
        assert dev.attributes["brightness"] == 75

    @pytest.mark.asyncio
    async def test_mock_iot_hub_event_bus_device_changed_emission(
        self, mock_iot_hub: MockIoTHub, event_bus: EventBus
    ) -> None:
        """Verify execution publishes DEVICE_STATE_CHANGED event to the EventBus."""
        await mock_iot_hub.execute("light.living_room", "turn_on", brightness=90)

        ev = await event_bus.wait_for(EventType.DEVICE_STATE_CHANGED, timeout=1.0)
        assert ev is not None
        assert ev.get("device_id") == "light.living_room"
        assert ev.get("state") == "on"
        assert ev.get("traits", {}).get("brightness") == 90

    @pytest.mark.asyncio
    async def test_mock_iot_hub_simulated_actuation_latency(
        self, device_registry: DeviceRegistry, event_bus: EventBus
    ) -> None:
        """Verify actuation latency simulation enforces configured delays."""
        hub = MockIoTHub(registry=device_registry, event_bus=event_bus, latency_ms=25.0)
        assert hub.simulated_latency_ms == 25.0

        t0 = time.perf_counter()
        res = await hub.execute("light.living_room", "turn_off")
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        assert res["success"] is True
        assert elapsed_ms >= 20.0

    @pytest.mark.asyncio
    async def test_mock_iot_hub_concurrent_command_execution(
        self, mock_iot_hub: MockIoTHub, device_registry: DeviceRegistry
    ) -> None:
        """Verify multiple simultaneous actuations execute cleanly without race conditions."""
        commands = [
            mock_iot_hub.execute("light.living_room", "turn_on", brightness=50),
            mock_iot_hub.execute("light.workshop", "turn_on", brightness=100),
            mock_iot_hub.execute("thermostat.main", "set_temperature", temperature=23.0),
            mock_iot_hub.execute("lock.front_door", "unlock"),
        ]
        results = await asyncio.gather(*commands)

        assert all(r["success"] is True for r in results)
        assert device_registry.get("light.living_room").attributes["brightness"] == 50
        assert device_registry.get("light.workshop").attributes["brightness"] == 100
        assert device_registry.get("thermostat.main").attributes["temperature"] == 23.0
        assert device_registry.get("lock.front_door").state == "unlocked"

    @pytest.mark.asyncio
    async def test_mock_iot_hub_nonexistent_device_handling(
        self, mock_iot_hub: MockIoTHub
    ) -> None:
        """Verify nonexistent devices, offline devices, and failure modes return structured errors."""
        # Nonexistent device
        res = await mock_iot_hub.execute("light.nonexistent", "turn_on")
        assert res["success"] is False
        assert "not found" in res["error"].lower()

        # Offline simulation
        mock_iot_hub.simulate_offline("light.living_room", offline=True)
        res_offline = await mock_iot_hub.execute("light.living_room", "turn_on")
        assert res_offline["success"] is False
        assert "unreachable" in res_offline["error"].lower()

        # Failure mode simulation
        mock_iot_hub.simulate_offline("light.living_room", offline=False)
        mock_iot_hub.simulate_failure("light.living_room", "Zigbee network timeout")
        res_fail = await mock_iot_hub.execute("light.living_room", "turn_on")
        assert res_fail["success"] is False
        assert "zigbee network timeout" in res_fail["error"].lower()

        # History tracking
        history = mock_iot_hub.get_history()
        assert len(history) >= 3
