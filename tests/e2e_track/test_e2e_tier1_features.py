"""Tier 1 E2E Feature Coverage Test Suite for Jarvis AI Assistant.

Authoritative source: ORIGINAL_REQUEST.md (§R1–R5), PROJECT.md, and TEST_INFRA.md.
27 Equivalence-Partitioned Test Cases verifying standard operating conditions
across all five core feature areas:
- Class TestTier1VoiceAndAudio (6 tests: R1)
- Class TestTier1HUDAndTelemetry (5 tests: R2)
- Class TestTier1DevicesAndTelephony (6 tests: R3)
- Class TestTier1ModelCategorizationAndLatency (5 tests: R4)
- Class TestTier1InstallerAndPackaging (5 tests: R5)
"""

from __future__ import annotations

import asyncio
from pathlib import Path
import re
import tomllib
from typing import Any

import pytest
from starlette.testclient import TestClient

from jarvis.audio.capture import VirtualAudioPipeline
from jarvis.audio.stt import MockSTTEngine
from jarvis.audio.tts import MockTTSEngine, format_jarvis_speech
from jarvis.audio.vad import VADDetector
from jarvis.config import SystemConfig
from jarvis.core.event_bus import EventBus, EventType
from jarvis.core.orchestrator import JarvisOrchestrator
from jarvis.core.state_machine import AssistantState, AssistantStateMachine
from tests.e2e_track.conftest import (
    AsyncEventCollector,
    AudioStreamGenerator,
    CallDebriefRecord,
    DeterministicModelRouter,
    InMemoryDeviceRegistry,
    HeadlessTelephonyEngine,
    ModelLatencyMetrics,
    SyntheticModelProvider,
    classify_model_tier,
    collect_system_telemetry,
)


# ==============================================================================
# CLASS 1: VOICE & AUDIO SUBSYSTEM (Requirement R1, 6 Tests)
# ==============================================================================
class TestTier1VoiceAndAudio:
    """Verifies continuous background audio capture, VAD, state machine transitions,
    sub-30ms barge-in preemption, persona formatting, and British TTS speech synthesis.
    """

    def test_r1_background_audio_capture_lifecycle(
        self,
        e2e_virtual_pipeline: VirtualAudioPipeline,
        audio_generator: AudioStreamGenerator,
    ) -> None:
        """Feature 1: Verify VirtualAudioPipeline start/stop lifecycle and chunk streaming."""
        pipeline = e2e_virtual_pipeline
        assert pipeline.is_active is True

        # Inject 10 chunks of 16kHz audio (10 * 32ms = 320ms)
        pcm_data = audio_generator.speech_like(320.0)
        injected_count = pipeline.inject_pcm_data(pcm_data)
        assert injected_count == 10

        # Read chunks and verify dimensions (512 samples * 2 bytes = 1024 bytes)
        for _ in range(injected_count):
            chunk = pipeline.read_chunk(timeout=0.1)
            assert len(chunk) == 1024

        pipeline.stop()
        assert pipeline.is_active is False

    def test_r1_adaptive_vad_speech_detection_and_circular_buffer(
        self,
        e2e_system_config: SystemConfig,
        audio_generator: AudioStreamGenerator,
    ) -> None:
        """Feature 2: Verify adaptive VAD detection and 1000ms pre-speech circular ring buffer."""
        vad = VADDetector(e2e_system_config.vad)
        silence_chunk = audio_generator.silence(32.0)
        assert len(silence_chunk) == 1024

        # Feed 35 frames of silence (~1120ms > 1000ms buffer)
        for _ in range(35):
            frame_res = vad.process_frame(silence_chunk)
            assert frame_res.is_speech is False
            assert frame_res.speech_started is False

        # Pre-speech circular buffer must reach configured capacity (~31 chunks)
        assert len(vad.pre_speech_buffer) >= 30

        # Feed 5 frames of high-amplitude tone (440Hz, amp=0.7)
        tone_chunk = audio_generator.tone(32.0, frequency=440.0, amplitude=0.7)
        speech_started_seen = False
        for _ in range(5):
            res = vad.process_frame(tone_chunk)
            if res.speech_started:
                speech_started_seen = True
            assert res.energy_rms > res.threshold

        assert speech_started_seen is True
        assert vad.is_in_speech is True

    @pytest.mark.asyncio
    async def test_r1_state_machine_voice_interaction_transitions(
        self,
        e2e_system_config: SystemConfig,
        e2e_event_bus: EventBus,
        event_collector: AsyncEventCollector,
    ) -> None:
        """Feature 5: Full canonical state transition sequence (IDLE -> LISTENING -> PROCESSING -> SPEAKING -> IDLE)."""
        pipeline = VirtualAudioPipeline(e2e_system_config.audio)
        stt = MockSTTEngine()
        stt.set_next_transcript("Jarvis, report status")
        tts = MockTTSEngine(e2e_system_config.tts)
        state_machine = AssistantStateMachine(e2e_event_bus)
        orchestrator = JarvisOrchestrator(
            config=e2e_system_config,
            event_bus=e2e_event_bus,
            state_machine=state_machine,
            audio_capture=pipeline,
            stt=stt,
            tts=tts,
        )

        await orchestrator.start()
        assert state_machine.is_idle

        # 1. Calibrate noise floor baseline
        for _ in range(10):
            pipeline.inject_synthetic_audio(duration_ms=32, amplitude=0.01)

        # 2. Inject speech tone frames (4 frames = ~128ms)
        for _ in range(4):
            pipeline.inject_synthetic_audio(duration_ms=32, amplitude=0.8, frequency=300.0)

        # Wait for SPEECH_STARTED
        speech_start_ev = await event_collector.wait_for(EventType.SPEECH_STARTED, timeout=2.0)
        assert speech_start_ev is not None

        # 3. Inject silence hangover (12 frames = ~384ms > 300ms)
        for _ in range(12):
            pipeline.inject_silence(duration_ms=32)

        # Wait for SPEECH_ENDED
        speech_end_ev = await event_collector.wait_for(EventType.SPEECH_ENDED, timeout=2.0)
        assert speech_end_ev is not None

        # Wait for TRANSCRIPT_READY
        transcript_ev = await event_collector.wait_for(EventType.TRANSCRIPT_READY, timeout=2.0)
        assert transcript_ev.get("text") == "Jarvis, report status"

        # Wait for TTS audio chunk
        tts_chunk_ev = await event_collector.wait_for(EventType.TTS_AUDIO_CHUNK, timeout=2.0)
        assert len(tts_chunk_ev.get("chunk", b"")) > 0

        # Wait for speech playback to finish and return to IDLE
        await asyncio.sleep(0.5)
        assert state_machine.is_idle

        # Verify state transition sequence in event history
        state_changes = [e.get("new_state") for e in event_collector.events_of_type(EventType.STATE_CHANGED)]
        assert "LISTENING" in state_changes
        assert "PROCESSING" in state_changes
        assert "SPEAKING" in state_changes
        assert "IDLE" in state_changes

        await orchestrator.stop()

    @pytest.mark.asyncio
    async def test_r1_sub_30ms_barge_in_preemption(
        self,
        e2e_system_config: SystemConfig,
        e2e_event_bus: EventBus,
        event_collector: AsyncEventCollector,
    ) -> None:
        """Feature 3: Verify sub-30ms barge-in preemption upon user speech onset during assistant speech."""
        pipeline = VirtualAudioPipeline(e2e_system_config.audio)
        tts = MockTTSEngine(e2e_system_config.tts)
        state_machine = AssistantStateMachine(e2e_event_bus)
        orchestrator = JarvisOrchestrator(
            config=e2e_system_config,
            event_bus=e2e_event_bus,
            state_machine=state_machine,
            audio_capture=pipeline,
            tts=tts,
        )

        await orchestrator.start()

        # Begin assistant speech output
        speak_task = asyncio.create_task(
            orchestrator.speak("I am currently analyzing all telemetry diagnostics, sir.")
        )
        await asyncio.sleep(0.05)
        assert state_machine.is_speaking

        # User interrupts by speaking loudly (inject 3 frames)
        for _ in range(3):
            pipeline.inject_synthetic_audio(duration_ms=32, amplitude=0.9, frequency=400.0)

        # Wait for BARGE_IN_TRIGGERED
        barge_in_ev = await event_collector.wait_for(EventType.BARGE_IN_TRIGGERED, timeout=2.0)
        assert barge_in_ev is not None

        interruption_latency = barge_in_ev.get("interruption_latency_ms")
        assert interruption_latency < 30.0, f"Barge-in latency {interruption_latency}ms must be < 30ms"

        # Allow orchestrator event loop step to complete transition
        await asyncio.sleep(0.05)
        assert state_machine.is_listening or state_machine.current_state in (AssistantState.LISTENING, AssistantState.PROCESSING)

        await speak_task
        await orchestrator.stop()

    def test_r1_jarvis_persona_courtesy_and_brevity(self) -> None:
        """Feature 4: Verify Jarvis persona formatting: removes markdown, limits to 1-2 sentences, courteous tone."""
        raw_markdown = (
            "```python\nimport sys\nprint('status')\n```\n"
            "**Diagnostic Complete**: All systems are operating within optimal parameters, sir.\n"
            "### Extended Analysis\n"
            "The arc reactor output is steady at 100%. We have detected no external intrusions.\n"
            "Additional telemetry reports can be generated upon your request."
        )
        formatted = format_jarvis_speech(raw_markdown)

        # Verify markdown stripped
        assert "```" not in formatted
        assert "**" not in formatted
        assert "###" not in formatted

        # Verify brevity (1-2 sentences)
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", formatted) if s.strip()]
        assert 1 <= len(sentences) <= 2

        # Verify courtesy
        assert "sir" in formatted.lower()

        # Verify fallback on empty string
        empty_formatted = format_jarvis_speech("")
        assert "sir" in empty_formatted.lower()

    @pytest.mark.asyncio
    async def test_r1_british_tts_voice_configuration_and_synthesis(
        self,
        e2e_system_config: SystemConfig,
    ) -> None:
        """Feature 4: Verify British voice TTS configuration parameters and streaming audio synthesis."""
        config = e2e_system_config.tts
        assert config.edge_voice == "en-GB-RyanNeural"
        assert config.sapi_voice == "Microsoft David Desktop"
        assert config.persona_name == "Jarvis"

        mock_tts = MockTTSEngine(config)
        chunks = []
        async for chunk in mock_tts.synthesize_stream("Right away, sir.", generation_id=1):
            chunks.append(chunk)

        assert len(chunks) > 0
        assert all(len(c) == 1024 for c in chunks)
        assert mock_tts.last_ttft_ms >= 0.0


# ==============================================================================
# CLASS 2: HUD & TELEMETRY INTERFACE (Requirement R2, 5 Tests)
# ==============================================================================
class TestTier1HUDAndTelemetry:
    """Verifies Iron Man holographic HUD static assets and palette (strictly NOT Ultron red),
    FastAPI HTTP root endpoint, WebSocket real-time telemetry streaming, Arc Reactor state modulation,
    and live hardware telemetry gauges.
    """

    def test_r2_hud_static_assets_and_color_palette(self) -> None:
        """Feature 6: Verify static HUD assets exist with Stark cyan/gold palette and NO Ultron red."""
        root_dir = Path(__file__).resolve().parent.parent.parent
        static_dir = root_dir / "jarvis" / "ui" / "static"
        assert static_dir.exists() and static_dir.is_dir()

        index_html = static_dir / "index.html"
        hud_css = static_dir / "hud.css"
        arc_reactor_js = static_dir / "arc_reactor.js"
        visualizer_js = static_dir / "visualizer.js"

        assert index_html.exists() and index_html.stat().st_size > 0
        assert hud_css.exists() and hud_css.stat().st_size > 0
        assert arc_reactor_js.exists() and arc_reactor_js.stat().st_size > 0
        assert visualizer_js.exists() and visualizer_js.stat().st_size > 0

        css_content = hud_css.read_text(encoding="utf-8")
        # Verify Iron Man Stark palette: Cyan #00E5FF, Electric Blue #0077B6, Gold #FFD700
        assert "#00E5FF" in css_content or "#0077B6" in css_content or "#FFD700" in css_content
        # Strictly NOT Ultron red as primary theme
        assert "#FF0000" not in css_content
        assert "#E60000" not in css_content
        assert "crimson" not in css_content.lower()

    def test_r2_fastapi_server_root_endpoint(self, hud_client: TestClient) -> None:
        """Feature 6: Verify FastAPI HUD root endpoint returns valid HTML with HUD visualizer containers."""
        response = hud_client.get("/")
        assert response.status_code == 200
        assert "text/html" in response.headers.get("content-type", "")
        content = response.text
        assert "arc-reactor" in content or "telemetry" in content

    def test_r2_websocket_telemetry_streaming(self, hud_client: TestClient) -> None:
        """Feature 10: Verify real-time WebSocket telemetry packet contains bounded float metrics."""
        with hud_client.websocket_connect("/ws/telemetry") as ws:
            msg_state = ws.receive_json()
            assert msg_state.get("type") == "state"

            msg_telemetry = ws.receive_json()
            assert msg_telemetry.get("type") == "telemetry"
            for key in ["cpu", "ram", "net_ping", "audio_level"]:
                assert key in msg_telemetry
                val = msg_telemetry[key]
                assert isinstance(val, (int, float))

            assert 0.0 <= msg_telemetry["cpu"] <= 100.0
            assert 0.0 <= msg_telemetry["ram"] <= 100.0
            assert msg_telemetry["net_ping"] >= 0.0
            assert msg_telemetry["audio_level"] >= 0.0

    def test_r2_arc_reactor_state_modulation_events(self, hud_client: TestClient) -> None:
        """Feature 7: Verify WebSocket broadcasts state changes for Arc Reactor rotation speed modulation."""
        with hud_client.websocket_connect("/ws/telemetry") as ws:
            _ = ws.receive_json()  # Initial state
            _ = ws.receive_json()  # Initial telemetry

            # Modulate assistant state to SPEAKING
            ws.send_json({"action": "set_state", "state": "SPEAKING"})
            state_msg = ws.receive_json()
            assert state_msg.get("type") == "state"
            assert state_msg.get("state") == "SPEAKING"

    def test_r2_real_time_hardware_telemetry_gauges(self) -> None:
        """Feature 9: Verify host hardware telemetry collector returns live CPU, RAM, and network metrics."""
        telemetry = collect_system_telemetry()
        assert isinstance(telemetry, dict)
        assert "cpu" in telemetry and "ram" in telemetry and "net_ping" in telemetry
        assert 0.0 <= telemetry["cpu"] <= 100.0
        assert 0.0 <= telemetry["ram"] <= 100.0
        assert telemetry["net_ping"] >= 0.0


# ==============================================================================
# CLASS 3: SMART DEVICES & TELEPHONY MANAGEMENT (Requirement R3, 6 Tests)
# ==============================================================================
class TestTier1DevicesAndTelephony:
    """Verifies smart device registry CRUD, natural language command parsing, actuation & event dispatch,
    busy-mode toggle, autonomous AI call screening proxy, and structured CallDebriefRecord with urgency tagging.
    """

    def test_r3_device_registry_entity_crud(self, device_registry: InMemoryDeviceRegistry) -> None:
        """Feature 11: Verify smart device entity catalog retrieval, listing, and domain schema."""
        all_devs = device_registry.list_all()
        assert len(all_devs) >= 5

        living_room = device_registry.get("light.living_room")
        assert living_room is not None
        assert living_room.domain == "light"
        assert living_room.name == "Living Room Light"

        thermostat = device_registry.get("thermostat.main")
        assert thermostat is not None
        assert thermostat.domain == "thermostat"

        lock = device_registry.get("lock.front_door")
        assert lock is not None
        assert lock.domain == "lock"

        # Nonexistent entity returns None safely
        assert device_registry.get("light.non_existent") is None

    def test_r3_natural_language_device_command_parsing(self, device_registry: InMemoryDeviceRegistry) -> None:
        """Feature 12: Verify natural language voice utterances translate to structured device actions."""
        entity, action, params = device_registry.parse_natural_command("turn on living room light")
        assert entity == "light.living_room"
        assert action == "turn_on"

        entity, action, params = device_registry.parse_natural_command("set thermostat to 22")
        assert entity == "thermostat.main"
        assert action == "set_temperature"
        assert params.get("temperature") == 22.0

        entity, action, params = device_registry.parse_natural_command("lock front door")
        assert entity == "lock.front_door"
        assert action == "lock"

    @pytest.mark.asyncio
    async def test_r3_device_actuation_and_event_publishing(
        self,
        device_registry: InMemoryDeviceRegistry,
        event_collector: AsyncEventCollector,
    ) -> None:
        """Feature 13: Verify device execution updates state and publishes DEVICE_STATE_CHANGED event."""
        res = device_registry.execute_command("light.living_room", "turn_on", brightness=80)
        assert res["success"] is True
        assert res["state"] == "on"
        assert res["attributes"]["brightness"] == 80

        dev = device_registry.get("light.living_room")
        assert dev.state == "on"
        assert dev.attributes["brightness"] == 80

        ev = await event_collector.wait_for(EventType.DEVICE_STATE_CHANGED, timeout=2.0)
        assert ev is not None
        assert ev.get("device_id") == "light.living_room"
        assert ev.get("state") == "on"

    def test_r3_busy_mode_state_toggle(self, telephony_engine: HeadlessTelephonyEngine) -> None:
        """Feature 14: Verify global busy-mode controller toggle."""
        assert telephony_engine.busy_mode is False
        telephony_engine.set_busy_mode(True)
        assert telephony_engine.busy_mode is True

        telephony_engine.set_busy_mode(False)
        assert telephony_engine.busy_mode is False

    @pytest.mark.asyncio
    async def test_r3_autonomous_ai_call_screening_proxy(
        self,
        telephony_engine: HeadlessTelephonyEngine,
        event_collector: AsyncEventCollector,
    ) -> None:
        """Feature 15: Verify autonomous AI call screening answers calls and conducts screening dialogue."""
        telephony_engine.set_busy_mode(True)
        call_id = "call-101"
        telephony_engine.handle_incoming_call(call_id, "Justin Hammer", "+1-555-0199")

        # Verify incoming call event
        incoming_ev = await event_collector.wait_for(EventType.CALL_INCOMING, timeout=2.0)
        assert incoming_ev.get("caller") == "Justin Hammer"

        # Verify screening state and opening Jarvis greeting
        active = telephony_engine.active_calls[call_id]
        assert active["state"] == "screening"
        assert len(active["transcript"]) >= 1
        assert "unavailable" in active["transcript"][0]["text"].lower()

        # Caller provides utterance; Jarvis acknowledges and takes message
        reply = telephony_engine.process_caller_utterance(call_id, "I am calling about the Stark Expo contract.")
        assert "message" in reply.lower() or "noted" in reply.lower()
        assert len(active["transcript"]) >= 3

    @pytest.mark.asyncio
    async def test_r3_call_debrief_record_and_urgency_classification(
        self,
        telephony_engine: HeadlessTelephonyEngine,
        event_collector: AsyncEventCollector,
    ) -> None:
        """Feature 16 & 18: Verify call debrief compilation, urgency tagging, and event emission."""
        telephony_engine.set_busy_mode(True)
        call_id = "call-102"
        telephony_engine.handle_incoming_call(call_id, "Colonel Rhodes", "+1-555-0144")

        # Feed urgent message
        reply = telephony_engine.process_caller_utterance(
            call_id, "Please have Tony review the urgent defense brief immediately."
        )
        assert any(w in reply.lower() for w in ["priority", "urgent", "noted", "emergency", "alerting"])

        debrief = telephony_engine.end_call(call_id)
        assert isinstance(debrief, CallDebriefRecord)
        assert debrief.call_id == call_id
        assert debrief.caller_name == "Colonel Rhodes"
        assert debrief.urgency in ["Low", "Normal", "High", "Emergency"]
        assert debrief.urgency in ["High", "Emergency"]  # Urgent keyword escalation
        assert "defense brief" in debrief.summary
        assert len(debrief.action_item) > 0
        assert len(debrief.transcript) >= 2

        # Verify debrief event emitted
        debrief_ev = await event_collector.wait_for(EventType.CALL_DEBRIEF_CREATED, timeout=2.0)
        assert debrief_ev.get("call_id") == call_id
        assert debrief_ev.get("urgency") == debrief.urgency


# ==============================================================================
# CLASS 4: MODEL CATEGORIZATION & LATENCY SELECTION (Requirement R4, 5 Tests)
# ==============================================================================
class TestTier1ModelCategorizationAndLatency:
    """Verifies 4-tier model taxonomy engine, benchmark latency metrics collection,
    fastest-path dynamic selection, generation metadata, and synthetic CI provider execution.
    """

    def test_r4_four_tier_taxonomy_categorization(self) -> None:
        """Feature 19: Verify 4-tier taxonomy engine bins latencies into Tier 0, 1, 2, 3."""
        assert classify_model_tier(25.0) == 0    # Tier 0: Direct Reflex (<50ms)
        assert classify_model_tier(90.0) == 1    # Tier 1: Edge SLM (<150ms)
        assert classify_model_tier(320.0) == 2   # Tier 2: Cloud Fast (<500ms)
        assert classify_model_tier(750.0) == 3   # Tier 3: Deep Reasoning (>=500ms)

    def test_r4_latency_benchmark_metrics_collection(
        self,
        model_router: DeterministicModelRouter,
    ) -> None:
        """Feature 20: Verify benchmark harness probes DNS, TCP, TTFT, TAT, and TPS metrics."""
        metrics_list = model_router.benchmark_all()
        assert len(metrics_list) == 4

        for m in metrics_list:
            assert isinstance(m, ModelLatencyMetrics)
            assert m.tier in (0, 1, 2, 3)
            assert m.dns_ms >= 0.0
            assert m.tcp_ms >= 0.0
            assert m.ttft_ms >= 0.0
            assert m.tat_ms >= m.ttft_ms
            assert m.tps > 0.0
            assert m.healthy is True

    def test_r4_fast_path_model_selection(
        self,
        model_router: DeterministicModelRouter,
    ) -> None:
        """Feature 21: Verify fast-path selector dynamically picks lowest TTFT viable candidate."""
        model_router.set_latency("tier0-direct-reflex", ttft_ms=45.0)
        model_router.set_latency("tier1-edge-slm", ttft_ms=110.0)
        model_router.set_latency("tier2-cloud-fast", ttft_ms=250.0)

        fastest = model_router.select_fastest()
        assert fastest == "tier0-direct-reflex"

        # Explicit tier requirement selection
        tier2_fastest = model_router.select_fastest(required_tier=2)
        assert tier2_fastest == "tier2-cloud-fast"

    def test_r4_fast_path_text_generation_and_metadata(
        self,
        model_router: DeterministicModelRouter,
    ) -> None:
        """Feature 21: Verify generation returns response text, selected model ID, and latency measurement."""
        response, model_id, latency_ms = model_router.generate("Status of power grid")
        assert len(response) > 0
        assert "sir" in response.lower()
        assert model_id == "tier0-direct-reflex"
        assert latency_ms > 0.0

    def test_r4_synthetic_model_provider_deterministic_ci(
        self,
        synthetic_model_provider: SyntheticModelProvider,
    ) -> None:
        """Feature 22: Verify deterministic synthetic model provider operates headlessly in CI."""
        response, model_id, latency_ms = synthetic_model_provider.generate("Diagnostic check")
        assert response is not None and len(response) > 0
        assert model_id in synthetic_model_provider.models
        assert latency_ms <= 50.0


# ==============================================================================
# CLASS 5: INSTALLER & GITHUB PACKAGING (Requirement R5, 5 Tests)
# ==============================================================================
class TestTier1InstallerAndPackaging:
    """Verifies existence, syntax, execution safety, and documentation of installer artifacts
    (install.ps1, install.bat, INSTALL.md) and pyproject.toml packaging metadata.
    """

    def test_r5_installer_artifacts_existence(self) -> None:
        """Feature 25: Verify install.ps1, install.bat, and INSTALL.md exist and are non-empty."""
        root_dir = Path(__file__).resolve().parent.parent.parent

        ps1 = root_dir / "install.ps1"
        bat = root_dir / "install.bat"
        md = root_dir / "INSTALL.md"

        assert ps1.exists() and ps1.is_file() and ps1.stat().st_size > 0
        assert bat.exists() and bat.is_file() and bat.stat().st_size > 0
        assert md.exists() and md.is_file() and md.stat().st_size > 0

    def test_r5_powershell_installer_syntax_and_structure(self) -> None:
        """Feature 25: Verify install.ps1 contains strict error handling, venv creation, and host logging."""
        root_dir = Path(__file__).resolve().parent.parent.parent
        ps1 = root_dir / "install.ps1"
        content = ps1.read_text(encoding="utf-8")

        assert "$ErrorActionPreference = 'Stop'" in content
        assert "python -m venv" in content or "$VENV_DIR" in content
        assert "pip install" in content
        assert "Write-Host" in content

    def test_r5_batch_installer_syntax_and_entrypoint(self) -> None:
        """Feature 25: Verify install.bat batch script syntax, python presence check, and errorlevel handling."""
        root_dir = Path(__file__).resolve().parent.parent.parent
        bat = root_dir / "install.bat"
        content = bat.read_text(encoding="utf-8")

        assert "@echo off" in content.lower()
        assert "where python" in content.lower()
        assert "if errorlevel" in content.lower()
        assert "pip install" in content.lower()

    def test_r5_install_documentation_completeness(self) -> None:
        """Feature 25: Verify INSTALL.md provides Prerequisites, Installation, Running, and Troubleshooting."""
        root_dir = Path(__file__).resolve().parent.parent.parent
        md = root_dir / "INSTALL.md"
        content = md.read_text(encoding="utf-8")

        assert "## Prerequisites" in content
        assert "Python" in content
        assert "## Installation" in content
        assert "install.ps1" in content
        assert "install.bat" in content
        assert "## Running Jarvis" in content
        assert "## Troubleshooting" in content

    def test_r5_packaging_metadata_and_entrypoints(self) -> None:
        """Feature 25: Verify pyproject.toml packaging metadata, jarvis CLI entrypoint, and dependencies."""
        root_dir = Path(__file__).resolve().parent.parent.parent
        pyproject_path = root_dir / "pyproject.toml"
        assert pyproject_path.exists()

        with open(pyproject_path, "rb") as f:
            data = tomllib.load(f)

        project = data.get("project", {})
        assert project.get("name") == "jarvis-assistant"
        assert ">=3.11" in project.get("requires-python", "")

        scripts = project.get("scripts", {})
        assert scripts.get("jarvis") == "run_jarvis:main"

        deps = project.get("dependencies", [])
        dep_str = " ".join(deps).lower()
        assert "fastapi" in dep_str
        assert "sounddevice" in dep_str
        assert "psutil" in dep_str
        assert "edge-tts" in dep_str
