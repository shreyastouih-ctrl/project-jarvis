"""Pytest fixtures for Jarvis AI Assistant headless test suite."""

from __future__ import annotations

import asyncio
from typing import AsyncGenerator

import pytest

from jarvis.audio.capture import VirtualAudioPipeline
from jarvis.audio.stt import MockSTTEngine
from jarvis.audio.tts import MockTTSEngine
from jarvis.audio.vad import VADDetector
from jarvis.config import SystemConfig
from jarvis.core.event_bus import EventBus
from jarvis.core.orchestrator import JarvisOrchestrator
from jarvis.core.state_machine import AssistantStateMachine


@pytest.fixture
def system_config() -> SystemConfig:
    """Return default test configuration in headless mode."""
    config = SystemConfig()
    config.headless = True
    config.audio.sample_rate = 16000
    config.audio.chunk_size = 512
    config.vad.pre_speech_buffer_ms = 1000
    config.vad.min_speech_frames = 2
    config.vad.silence_hangover_ms = 300
    return config


@pytest.fixture
async def event_bus() -> AsyncGenerator[EventBus, None]:
    """Provide a clean EventBus and guarantee async close."""
    bus = EventBus()
    yield bus
    await bus.close()


@pytest.fixture
def state_machine(event_bus: EventBus) -> AssistantStateMachine:
    """Provide an AssistantStateMachine connected to the event bus."""
    return AssistantStateMachine(event_bus)


@pytest.fixture
def virtual_pipeline(system_config: SystemConfig) -> VirtualAudioPipeline:
    """Provide an active VirtualAudioPipeline for audio injection."""
    pipeline = VirtualAudioPipeline(system_config.audio)
    pipeline.start()
    yield pipeline
    pipeline.stop()
    pipeline.clear()


@pytest.fixture
def vad_detector(system_config: SystemConfig) -> VADDetector:
    """Provide an adaptive VADDetector."""
    return VADDetector(system_config.vad)


@pytest.fixture
def mock_stt() -> MockSTTEngine:
    """Provide a deterministic MockSTTEngine."""
    return MockSTTEngine()


@pytest.fixture
def mock_tts(system_config: SystemConfig) -> MockTTSEngine:
    """Provide a MockTTSEngine with instant barge-in support."""
    return MockTTSEngine(system_config.tts)


@pytest.fixture
async def orchestrator(
    system_config: SystemConfig,
    event_bus: EventBus,
    state_machine: AssistantStateMachine,
    virtual_pipeline: VirtualAudioPipeline,
    vad_detector: VADDetector,
    mock_stt: MockSTTEngine,
    mock_tts: MockTTSEngine,
) -> AsyncGenerator[JarvisOrchestrator, None]:
    """Provide a fully wired JarvisOrchestrator."""
    orch = JarvisOrchestrator(
        config=system_config,
        event_bus=event_bus,
        state_machine=state_machine,
        audio_capture=virtual_pipeline,
        vad=vad_detector,
        stt=mock_stt,
        tts=mock_tts,
    )
    yield orch
    if orch.is_running:
        await orch.stop()


# ---------------------------------------------------------------------------
# Milestone 3 Fixtures: Smart Devices & Telephony
# ---------------------------------------------------------------------------

from jarvis.devices.intent_parser import DeviceIntentParser
from jarvis.devices.mock_iot import MockIoTHub
from jarvis.devices.registry import DeviceRegistry
from jarvis.telephony.call_engine import CallEngine
from jarvis.telephony.simulator import CallSimulator


@pytest.fixture
def device_registry(event_bus: EventBus) -> DeviceRegistry:
    """Provide a seeded DeviceRegistry connected to EventBus."""
    return DeviceRegistry(event_bus=event_bus, seed_defaults=True)


@pytest.fixture
def mock_iot_hub(device_registry: DeviceRegistry, event_bus: EventBus) -> MockIoTHub:
    """Provide an active MockIoTHub connected to registry and event bus."""
    return MockIoTHub(registry=device_registry, event_bus=event_bus)


@pytest.fixture
def device_intent_parser(device_registry: DeviceRegistry) -> DeviceIntentParser:
    """Provide a DeviceIntentParser aware of registered device entities."""
    return DeviceIntentParser(registry=device_registry)


@pytest.fixture
def telephony_engine(event_bus: EventBus) -> CallEngine:
    """Provide a clean CallEngine connected to EventBus."""
    return CallEngine(event_bus=event_bus)


@pytest.fixture
def call_simulator(telephony_engine: CallEngine) -> CallSimulator:
    """Provide a headless CallSimulator for scripted testing."""
    return CallSimulator(engine=telephony_engine)

