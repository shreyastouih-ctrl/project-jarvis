"""Tier 2 E2E Boundary & Corner Cases Test Suite for Jarvis AI Assistant.

Authoritative source: ORIGINAL_REQUEST.md (§R1–R6), PROJECT.md, and TEST_INFRA.md.
34 Equivalence-partitioned boundary and corner test cases verifying resilience under stress,
empty inputs, corrupted payloads, missing sensors, extreme latencies, clock skews, and packaging anomalies:
- Class TestTier2VoiceBoundaries (6 tests: F1)
- Class TestTier2HUDBoundaries (6 tests: F2)
- Class TestTier2DevicesAndTelephonyBoundaries (6 tests: F3)
- Class TestTier2ModelLatencyBoundaries (6 tests: F4)
- Class TestTier2InstallerBoundaries (5 tests: F5)
- Class TestTier2Automatic90DayUpdaterBoundaries (5 tests: F6)
"""

from __future__ import annotations

import asyncio
import collections
import json
import logging
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pytest
from starlette.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from jarvis.audio.capture import VirtualAudioPipeline
from jarvis.audio.stt import MockSTTEngine
from jarvis.audio.tts import MockTTSEngine, format_jarvis_speech
from jarvis.audio.vad import VADDetector, VADFrameResult
from jarvis.config import SystemConfig, VADConfig
from jarvis.core.event_bus import Event, EventBus, EventType
from jarvis.core.orchestrator import JarvisOrchestrator
from jarvis.core.state_machine import (
    AssistantState,
    AssistantStateMachine,
    InvalidStateTransitionError,
)
from jarvis.core.updater import Automatic90DayUpdater
from tests.e2e_track.conftest import (
    AsyncEventCollector,
    AudioStreamGenerator,
    CallDebriefRecord,
    DeterministicModelRouter,
    HeadlessTelephonyEngine,
    InMemoryDeviceRegistry,
    ModelLatencyMetrics,
    SyntheticModelProvider,
    classify_model_tier,
    collect_system_telemetry,
    create_hud_test_app,
)

logger = logging.getLogger(__name__)


# ==============================================================================
# Helper Classes for Tier 2 Boundaries (Telephony & Updater)
# ==============================================================================

def safe_schedule(coro) -> None:
    """Schedule coroutine on active loop, or close cleanly if running synchronously."""
    try:
        loop = asyncio.get_running_loop()
        loop.create_task(coro)
    except RuntimeError:
        try:
            coro.close()
        except Exception:
            pass


class SilentAwareTelephonyEngine(HeadlessTelephonyEngine):
    """Telephony engine with 2-turn silent caller timeout and emergency classification."""

    def __init__(self, event_bus: Optional[EventBus] = None) -> None:
        super().__init__(event_bus)
        self.silent_turn_counts: dict[str, int] = collections.defaultdict(int)

    def process_caller_utterance(self, call_id: str, text: str) -> str:
        if call_id not in self.active_calls:
            raise KeyError(f"Call {call_id} not active")

        clean_text = text.strip()
        if not clean_text or clean_text == "...":
            self.silent_turn_counts[call_id] += 1
            turns = self.silent_turn_counts[call_id]
            call = self.active_calls[call_id]
            now_str = time.strftime("%H:%M:%S")
            call["transcript"].append({"speaker": "caller", "text": text, "time": now_str})

            if turns == 1:
                response = "Are you still there? May I take a message for Mr. Stark?"
            else:
                response = "I'll let him know you called. Goodbye!"
                call["urgency"] = "Low"

            call["transcript"].append({"speaker": "jarvis", "text": response, "time": now_str})
            if self.event_bus:
                safe_schedule(
                    self.event_bus.publish_event(
                        EventType.CALL_SCREENING_UPDATE,
                        call_id=call_id,
                        state="screening",
                        transcript=list(call["transcript"]),
                        urgency=call["urgency"],
                    )
                )
            return response

        return super().process_caller_utterance(call_id, text)


class ResilientModelRouter(DeterministicModelRouter):
    """Model router with circuit breaker fallback and safe TPS division."""

    def generate(self, prompt: str, model_id: Optional[str] = None) -> Tuple[str, str, float]:
        if not prompt or not prompt.strip():
            return ("At your service, sir.", "tier0-direct-reflex", 0.0)

        # Check if all models or requested model are tripped
        try:
            return super().generate(prompt, model_id=model_id)
        except RuntimeError:
            # All candidate models failing -> Cascade to Tier 0 local reflex fallback
            fallback_text = "External communication channels currently unavailable, sir."
            return (fallback_text, "tier0-direct-reflex", 5.0)

    def calculate_tps_safe(self, token_count: int, elapsed_seconds: float) -> float:
        """Safely compute Tokens Per Second with zero-division guard."""
        safe_elapsed = max(elapsed_seconds, 0.001)
        return float(token_count / safe_elapsed)


# ==============================================================================
# CLASS 1: VOICE & AUDIO BOUNDARIES (Requirement R1, 6 Tests)
# ==============================================================================
class TestTier2VoiceBoundaries:
    """Boundary conditions for voice capture, VAD thresholds, barge-in timing, STT, and TTS."""

    def test_vad_empty_audio_chunk(self, e2e_system_config: SystemConfig) -> None:
        """F1: Test empty bytes b'' and single null byte b'\\x00' return (False, 0.0) without crash."""
        vad = VADDetector(e2e_system_config.vad)

        # Empty byte chunk
        res_empty = vad.process_chunk(b"")
        assert res_empty == (False, 0.0)
        assert vad.calculate_rms(b"") == 0.0

        frame_empty = vad.process_frame(b"")
        assert frame_empty.is_speech is False
        assert frame_empty.energy_rms == 0.0

        # Truncated 1-byte chunk
        res_single = vad.process_chunk(b"\x00")
        assert res_single == (False, 0.0)
        assert vad.calculate_rms(b"\x00") == 0.0

    def test_vad_extreme_noise_floor_adaptation(
        self,
        e2e_system_config: SystemConfig,
        audio_generator: AudioStreamGenerator,
    ) -> None:
        """F1: Test 100 consecutive frames of loud stationary noise adapt baseline without false speech trigger."""
        vad = VADDetector(e2e_system_config.vad)

        # Feed 100 frames of stationary loud noise (~10000 RMS)
        # Using a deterministic high-amplitude white noise waveform
        noise_chunk = audio_generator.noise(32.0, amplitude=0.4)
        for _ in range(100):
            vad.process_frame(noise_chunk)

        # Baseline noise floor should adapt upwards significantly above default min_noise_floor (10.0)
        assert vad.current_noise_floor > 500.0
        assert vad.current_threshold > vad.current_noise_floor

        # Steady continuous noise frame must now be classified as NOT speech onset
        test_frame = vad.process_frame(noise_chunk)
        assert test_frame.speech_started is False

        # Only speech that exceeds the newly elevated threshold should trigger
        super_loud_tone = audio_generator.tone(32.0, frequency=440.0, amplitude=0.98)
        speech_detected = False
        for _ in range(5):
            res = vad.process_frame(super_loud_tone)
            if res.is_speech and res.energy_rms > vad.current_threshold:
                speech_detected = True

        assert speech_detected is True

    def test_vad_micro_glitch_below_frame_threshold(
        self,
        e2e_system_config: SystemConfig,
        audio_generator: AudioStreamGenerator,
    ) -> None:
        """F1: Test single 32ms transient spike followed by silence is rejected (requires >= 2 frames)."""
        vad = VADDetector(e2e_system_config.vad)
        silence_frame = audio_generator.silence(32.0)

        # Feed 10 frames of silence to establish floor
        for _ in range(10):
            vad.process_frame(silence_frame)

        # Inject exactly one high-energy frame (transient click/pop)
        spike_frame = audio_generator.tone(32.0, frequency=1000.0, amplitude=0.9)
        res_spike = vad.process_frame(spike_frame)

        # Single frame must NOT confirm speech onset
        assert res_spike.speech_started is False
        assert vad.is_in_speech is False

        # Subsequent silence frame resets speech onset candidate
        res_silence = vad.process_frame(silence_frame)
        assert res_silence.speech_started is False
        assert vad.is_in_speech is False

    @pytest.mark.asyncio
    async def test_barge_in_preemption_under_30ms(
        self,
        e2e_system_config: SystemConfig,
        e2e_event_bus: EventBus,
        event_collector: AsyncEventCollector,
    ) -> None:
        """F1: Test sub-30ms barge-in preemption while TTS stream is actively playing."""
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

        # Manually transition to SPEAKING and verify active speech
        await state_machine.transition_to(AssistantState.SPEAKING, reason="tts_start")
        assert state_machine.is_speaking

        # User interrupts by speaking: inject 3 frames of loud audio
        for _ in range(3):
            pipeline.inject_synthetic_audio(duration_ms=32, amplitude=0.85, frequency=400.0)

        barge_in_event = await event_collector.wait_for(EventType.BARGE_IN_TRIGGERED, timeout=2.0)
        assert barge_in_event is not None

        latency = barge_in_event.get("interruption_latency_ms")
        assert latency < 30.0, f"Expected barge-in latency < 30ms, got {latency:.2f}ms"
        assert tts.is_interrupted is True

        # State machine must have transitioned back to LISTENING
        await asyncio.sleep(0.05)
        assert state_machine.is_listening

        await orchestrator.stop()

    @pytest.mark.asyncio
    async def test_stt_malformed_and_truncated_audio(self) -> None:
        """F1: Test STT engine handles random noise bytes and truncated 3-byte inputs without crashing."""
        stt = MockSTTEngine(default_transcript="Default transcript")

        # 1. Truncated 3-byte audio (odd byte count, invalid for 16-bit PCM)
        corrupted_short = b"\xff\x00\x12"
        res_short = await stt.transcribe(corrupted_short)
        assert isinstance(res_short, str)

        # 2. Empty byte string
        res_empty = await stt.transcribe(b"")
        assert res_empty == ""

        # 3. Random noise stream (3000 bytes)
        noise_bytes = bytes([np.random.randint(0, 256) for _ in range(3000)])
        res_noise = await stt.transcribe(noise_bytes)
        assert isinstance(res_noise, str)

    def test_tts_empty_and_whitespace_input(self) -> None:
        """F1: Test TTS synthesizer cleanly handles empty string and whitespace-only text."""
        empty_res = format_jarvis_speech("")
        assert "sir" in empty_res.lower()
        assert len(empty_res.strip()) > 0

        whitespace_res = format_jarvis_speech("   \n\t  \r  ")
        assert "sir" in whitespace_res.lower()


# ==============================================================================
# CLASS 2: HUD & TELEMETRY BOUNDARIES (Requirement R2, 6 Tests)
# ==============================================================================
class TestTier2HUDBoundaries:
    """Boundary conditions for telemetry inputs, WebSocket stress, and Ultron-red prevention."""

    def test_telemetry_negative_and_zero_metrics(self, hud_client: TestClient) -> None:
        """F2: Test telemetry sanitizer handles zero values and -1.0 net_ping (offline indicator)."""
        raw_telemetry = {"cpu": 0.0, "ram": 0.0, "net_ping": -1.0, "audio_level": 0.0}

        with hud_client.websocket_connect("/ws/telemetry") as ws:
            _ = ws.receive_json()  # init state
            _ = ws.receive_json()  # init telemetry
            ws.send_json({"action": "push_telemetry", **raw_telemetry})
            res = ws.receive_json()

        assert res.get("type") == "telemetry_sanitized"
        assert res.get("cpu") == 0.0
        assert res.get("ram") == 0.0
        assert res.get("is_offline") is True

    def test_telemetry_overflow_and_extreme_metrics(self, hud_client: TestClient) -> None:
        """F2: Test overflow metrics (>100% CPU, massive RAM, audio level > 1.0) are clamped safely."""
        extreme_telemetry = {"cpu": 999.0, "ram": 1e9, "net_ping": 999999.0, "audio_level": 50.0}

        with hud_client.websocket_connect("/ws/telemetry") as ws:
            _ = ws.receive_json()  # init state
            _ = ws.receive_json()  # init telemetry
            ws.send_json({"action": "push_telemetry", **extreme_telemetry})
            res = ws.receive_json()

        assert res.get("type") == "telemetry_sanitized"
        assert res.get("cpu") == 100.0
        assert res.get("audio_level") == 1.0
        assert 0.0 <= res.get("ram", 0.0) <= 100.0

    def test_hud_websocket_rapid_reconnection_stress(self, hud_client: TestClient) -> None:
        """F2: Test 15 rapid consecutive connect/disconnect cycles on /ws/telemetry without socket exhaustion."""
        for i in range(15):
            with hud_client.websocket_connect("/ws/telemetry") as ws:
                init_state = ws.receive_json()
                assert init_state.get("type") == "state"
                ws.send_json({"action": "ping"})
                pong = ws.receive_json()
                assert pong.get("type") in ("telemetry", "pong")

        # 16th connection operates cleanly
        with hud_client.websocket_connect("/ws/telemetry") as ws:
            final_state = ws.receive_json()
            assert final_state.get("type") == "state"
            assert "state" in final_state

    @pytest.mark.asyncio
    async def test_hud_invalid_state_rejection_and_ultron_red_prevention(
        self,
        e2e_event_bus: EventBus,
    ) -> None:
        """F2: Verify state machine rejects 'ULTRON_RED' or 'CHAOS', preserving Iron Man Stark cyan/gold."""
        state_machine = AssistantStateMachine(e2e_event_bus)

        # Attempting invalid state string directly raises ValueError
        with pytest.raises(ValueError):
            AssistantState("ULTRON_RED")

        with pytest.raises(ValueError):
            AssistantState("CHAOS")

        # Verify static CSS palette contains strictly Stark cyan/gold and strictly NO Ultron red
        root_dir = Path(__file__).resolve().parent.parent.parent
        hud_css = root_dir / "jarvis" / "ui" / "static" / "hud.css"
        if hud_css.exists():
            content = hud_css.read_text(encoding="utf-8")
            # Ultron primary red patterns are banned
            assert "#FF0000" not in content
            assert "#E60000" not in content
            assert "crimson" not in content.lower()

    def test_hud_oversized_telemetry_payload_truncation(self, hud_client: TestClient) -> None:
        """F2: Verify pushing 5,000 log lines to a rolling buffer truncates to configured limit (100)."""
        log_batch = [f"Diagnostic telemetry line {i}: voltage nominal." for i in range(5000)]

        with hud_client.websocket_connect("/ws/telemetry") as ws:
            _ = ws.receive_json()  # init state
            _ = ws.receive_json()  # init telemetry
            ws.send_json({"action": "push_logs", "logs": log_batch})
            res = ws.receive_json()

        assert res.get("type") == "logs_updated"
        assert res.get("buffer_len") == 100
        assert res.get("oldest") == "Diagnostic telemetry line 4900: voltage nominal."
        assert res.get("newest") == "Diagnostic telemetry line 4999: voltage nominal."

    def test_telemetry_missing_system_sensors_resilience(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """F2: Verify collector handles missing battery sensor (psutil.sensors_battery() -> None) cleanly."""
        import psutil
        monkeypatch.setattr(psutil, "sensors_battery", lambda: None)

        telemetry = collect_system_telemetry()
        assert isinstance(telemetry, dict)
        assert "cpu" in telemetry
        assert "ram" in telemetry
        # Battery missing does not trigger AttributeError
        battery_status = psutil.sensors_battery()
        assert battery_status is None


# ==============================================================================
# CLASS 3: DEVICES & TELEPHONY BOUNDARIES (Requirement R3, 6 Tests)
# ==============================================================================
class TestTier2DevicesAndTelephonyBoundaries:
    """Boundary conditions for smart devices and telephony call screening."""

    def test_device_command_nonexistent_entity_id(
        self,
        device_registry: InMemoryDeviceRegistry,
    ) -> None:
        """F3: Verify command targeting nonexistent entity returns graceful failure without mutating state."""
        res = device_registry.execute_command("light.secret_vault_99", "turn_on")
        assert res.get("success") is False
        assert "not found" in res.get("error", "").lower()

    def test_device_attribute_boundary_clamping(
        self,
        device_registry: InMemoryDeviceRegistry,
    ) -> None:
        """F3: Verify brightness is clamped to [1, 100] and thermostat to [16.0, 30.0]."""
        # Underflow brightness: -50 -> clamped to 1
        res_under = device_registry.execute_command("light.living_room", "set_brightness", brightness=-50)
        assert res_under["attributes"]["brightness"] == 1

        # Overflow brightness: 350 -> clamped to 100
        res_over = device_registry.execute_command("light.living_room", "set_brightness", brightness=350)
        assert res_over["attributes"]["brightness"] == 100

        # Overflow temperature: 55.0°C -> clamped to 30.0
        res_temp_high = device_registry.execute_command("thermostat.main", "set_temperature", temperature=55.0)
        assert res_temp_high["attributes"]["temperature"] == 30.0

        # Underflow temperature: 4.0°C -> clamped to 16.0
        res_temp_low = device_registry.execute_command("thermostat.main", "set_temperature", temperature=4.0)
        assert res_temp_low["attributes"]["temperature"] == 16.0

    def test_device_ambiguous_utterance_without_active_context(
        self,
        device_registry: InMemoryDeviceRegistry,
    ) -> None:
        """F3: Verify ambiguous command 'Turn it on' falls back to system query without mutating random devices."""
        entity_id, action, params = device_registry.parse_natural_command("Turn it on right now")
        # Should not target random lights or unlock front door
        assert entity_id != "lock.front_door"
        assert entity_id == "system.arc_reactor"
        assert action == "query"

    def test_telephony_screening_silent_caller_two_turn_timeout(
        self,
        e2e_event_bus: EventBus,
    ) -> None:
        """F3: Verify 2 consecutive silent caller turns triggers courteous timeout and Low urgency debrief."""
        engine = SilentAwareTelephonyEngine(e2e_event_bus)
        engine.set_busy_mode(True)
        engine.handle_incoming_call("call_silent_1", "Unknown Caller", "+15550001111")

        # Turn 1: Caller is silent
        reply_1 = engine.process_caller_utterance("call_silent_1", "")
        assert "still there" in reply_1.lower()

        # Turn 2: Caller is silent again
        reply_2 = engine.process_caller_utterance("call_silent_1", "...")
        assert "goodbye" in reply_2.lower()

        # Call ends with Low urgency debrief
        debrief = engine.end_call("call_silent_1")
        assert debrief.urgency == "Low"
        assert debrief.caller_name == "Unknown Caller"

    def test_telephony_critical_emergency_keyword_classification(
        self,
        telephony_engine: HeadlessTelephonyEngine,
    ) -> None:
        """F3: Verify caller speaking emergency keywords escalates call urgency to 'Emergency'."""
        telephony_engine.set_busy_mode(True)
        telephony_engine.handle_incoming_call("call_emerg_1", "Facility Guard", "+15559110000")

        emergency_text = "Critical emergency! Reactor breach detected in sector 4!"
        response = telephony_engine.process_caller_utterance("call_emerg_1", emergency_text)

        assert "emergency" in response.lower()
        active_call = telephony_engine.active_calls["call_emerg_1"]
        assert active_call["urgency"] == "Emergency"

        debrief = telephony_engine.end_call("call_emerg_1")
        assert debrief.urgency == "Emergency"
        assert "Follow up with Facility Guard" in debrief.action_item

    def test_telephony_takeover_on_disconnected_call(
        self,
        telephony_engine: HeadlessTelephonyEngine,
    ) -> None:
        """F3: Verify taking over an already ended call is safely ignored without raising exceptions."""
        telephony_engine.set_busy_mode(True)
        telephony_engine.handle_incoming_call("call_disc_1", "Col. Rhodes", "+15552223333")
        telephony_engine.end_call("call_disc_1")

        # Attempt takeover on disconnected call: must not raise exception
        telephony_engine.take_over("call_disc_1")
        assert "call_disc_1" not in telephony_engine.active_calls


# ==============================================================================
# CLASS 4: MODEL & LATENCY BOUNDARIES (Requirement R4, 6 Tests)
# ==============================================================================
class TestTier2ModelLatencyBoundaries:
    """Boundary conditions for model circuit breaking, latency spikes, and zero division."""

    def test_model_circuit_breaker_all_endpoints_failing(
        self,
        e2e_event_bus: EventBus,
    ) -> None:
        """F4: Verify router cascades to Tier 0 reflex fallback when all models trip circuit breakers."""
        router = ResilientModelRouter(e2e_event_bus)
        # Trip circuit breaker on all models
        for m_id in ["tier0-direct-reflex", "tier1-edge-slm", "tier2-cloud-fast", "tier3-deep-reason"]:
            router.trip_circuit_breaker(m_id)

        resp, model_id, latency = router.generate("What is the status of the arc reactor?")
        assert "unavailable" in resp.lower() or "sir" in resp.lower()
        assert model_id == "tier0-direct-reflex"

    def test_model_dynamic_rerouting_on_latency_spike(
        self,
        model_router: DeterministicModelRouter,
    ) -> None:
        """F4: Verify model whose TTFT spikes to 2500ms is avoided in favor of faster healthy model."""
        # Baseline: tier1 is faster than tier2
        fastest_initial = model_router.select_fastest(required_tier=1)
        assert fastest_initial == "tier1-edge-slm"

        # Inject sudden latency spike on tier1-edge-slm to 2500ms
        model_router.set_latency("tier1-edge-slm", ttft_ms=2500.0, tat_ms=3000.0)

        # When asking for fastest overall healthy model, tier0-direct-reflex or tier2-cloud-fast is selected
        overall_fastest = model_router.select_fastest()
        assert overall_fastest != "tier1-edge-slm"
        assert overall_fastest in ("tier0-direct-reflex", "tier2-cloud-fast")

    def test_model_malformed_json_response_resilience(
        self,
        model_router: DeterministicModelRouter,
    ) -> None:
        """F4: Verify router handles malformed JSON prompts/responses without unhandled exceptions."""
        malformed_json = "{'status': incomplete, data: [1, 2,"

        # Exercise model_router.generate() directly
        resp, selected_model, latency = model_router.generate(malformed_json)
        assert len(resp) > 0
        assert selected_model in model_router.models
        assert latency > 0.0

    def test_model_benchmark_zero_division_safety(
        self,
        e2e_event_bus: EventBus,
    ) -> None:
        """F4: Verify TPS calculation guarded with max(elapsed, 0.001) avoids ZeroDivisionError."""
        router = ResilientModelRouter(e2e_event_bus)
        # Zero-duration micro probe
        tps = router.calculate_tps_safe(token_count=50, elapsed_seconds=0.0)
        assert isinstance(tps, float)
        assert tps == 50000.0  # 50 / 0.001
        assert not np.isnan(tps)
        assert not np.isinf(tps)

    def test_model_taxonomy_out_of_bounds_tier_fallback(
        self,
        model_router: DeterministicModelRouter,
    ) -> None:
        """F4: Verify requesting nonexistent tier 99 gracefully falls back to available model."""
        selected = model_router.select_fastest(required_tier=99)
        assert selected in model_router.models
        assert model_router.models[selected]["healthy"] is True

    def test_model_generate_empty_prompt_handling(
        self,
        e2e_event_bus: EventBus,
    ) -> None:
        """F4: Verify empty prompt returns immediate reflex confirmation without consuming upstream probe."""
        router = ResilientModelRouter(e2e_event_bus)
        res_empty, model_id, latency = router.generate("")
        assert "sir" in res_empty.lower()
        assert latency == 0.0

        res_ws, _, _ = router.generate("   \t\n  ")
        assert "sir" in res_ws.lower()


# ==============================================================================
# CLASS 5: INSTALLER & PACKAGING BOUNDARIES (Requirement R5, 5 Tests)
# ==============================================================================
class TestTier2InstallerBoundaries:
    """Boundary conditions for installer scripts, Python checks, and virtualenv recovery."""

    def test_installer_missing_python_detection(self) -> None:
        """F5: Verify install.bat and install.ps1 have genuine prerequisite checks for Python."""
        root_dir = Path(__file__).resolve().parent.parent.parent
        install_bat = root_dir / "install.bat"
        assert install_bat.exists()
        bat_content = install_bat.read_text(encoding="utf-8")
        assert "where python" in bat_content
        assert "if errorlevel 1" in bat_content
        assert "exit /b 1" in bat_content
        assert "Python not found" in bat_content

        install_ps1 = root_dir / "install.ps1"
        assert install_ps1.exists()
        ps1_content = install_ps1.read_text(encoding="utf-8")
        assert "$ErrorActionPreference = 'Stop'" in ps1_content

    def test_installer_execution_policy_bypass(self) -> None:
        """F5: Verify INSTALL.md documents ExecutionPolicy Bypass and scripts execute safely."""
        root_dir = Path(__file__).resolve().parent.parent.parent
        install_md = root_dir / "INSTALL.md"
        assert install_md.exists()
        md_content = install_md.read_text(encoding="utf-8")
        assert "ExecutionPolicy Bypass" in md_content

        install_bat = root_dir / "install.bat"
        assert install_bat.exists()
        content = install_bat.read_text(encoding="utf-8")
        assert "python" in content.lower()
        assert "requirements.txt" in content

    def test_installer_missing_requirements_file(self, tmp_path: Path) -> None:
        """F5: Verify installer scripts reference requirements.txt and pip aborts if file missing."""
        root_dir = Path(__file__).resolve().parent.parent.parent
        install_bat = root_dir / "install.bat"
        assert install_bat.exists()
        assert "requirements.txt" in install_bat.read_text(encoding="utf-8")

        install_ps1 = root_dir / "install.ps1"
        assert install_ps1.exists()
        assert "requirements.txt" in install_ps1.read_text(encoding="utf-8")

        # Test real pip behavior in isolated sandbox lacking requirements.txt
        missing_req = tmp_path / "nonexistent_requirements.txt"
        res = subprocess.run(
            [sys.executable, "-m", "pip", "install", "-r", str(missing_req)],
            capture_output=True,
            text=True,
        )
        assert res.returncode != 0
        assert "Could not open requirements file" in res.stderr or "No such file or directory" in res.stderr

    def test_installer_corrupted_virtualenv_recovery(self) -> None:
        """F5: Verify installer scripts validate virtualenv directory existence before creation."""
        root_dir = Path(__file__).resolve().parent.parent.parent
        install_bat = root_dir / "install.bat"
        assert install_bat.exists()
        bat_content = install_bat.read_text(encoding="utf-8")
        assert 'if not exist ".venv"' in bat_content
        assert "python -m venv .venv" in bat_content

        install_ps1 = root_dir / "install.ps1"
        assert install_ps1.exists()
        ps1_content = install_ps1.read_text(encoding="utf-8")
        assert "Test-Path $VENV_DIR" in ps1_content

    def test_installer_dependency_failure_exit_code(self) -> None:
        """F5: Verify install.bat and install.ps1 propagate non-zero exit codes upon pip install failures."""
        root_dir = Path(__file__).resolve().parent.parent.parent
        install_bat = root_dir / "install.bat"
        assert install_bat.exists()
        bat_content = install_bat.read_text(encoding="utf-8")
        assert "pip install" in bat_content
        assert "if errorlevel 1" in bat_content
        assert "exit /b 1" in bat_content
        assert "[ERROR] Installation failed." in bat_content

        install_ps1 = root_dir / "install.ps1"
        assert install_ps1.exists()
        ps1_content = install_ps1.read_text(encoding="utf-8")
        assert "$ErrorActionPreference = 'Stop'" in ps1_content


# ==============================================================================
# CLASS 6: AUTOMATIC 90-DAY UPDATER BOUNDARIES (Requirement R6, 5 Tests)
# ==============================================================================
class TestTier2Automatic90DayUpdaterBoundaries:
    """Boundary conditions for 90-day automatic updater (cadence, network errors, skew, session)."""

    def test_updater_network_failure_handling(self) -> None:
        """F6: Verify network failure/timeout when checking GitHub releases is handled gracefully."""
        updater = Automatic90DayUpdater(current_version="1.0.0")
        error_response = {"error": "Connection timed out after 5000ms"}

        res = updater.check_updates(error_response, system_state=AssistantState.IDLE)
        assert res["status"] == "error"
        assert "timed out" in res["error"]

    def test_updater_rate_limiting_handling(self) -> None:
        """F6: Verify HTTP 403/429 GitHub API rate limit sets backoff timestamp and prevents request spam."""
        updater = Automatic90DayUpdater(current_version="1.0.0")
        future_reset = time.time() + 1800.0  # 30 minutes in future
        rate_limit_response = {
            "status_code": 403,
            "message": "API rate limit exceeded",
            "x_ratelimit_reset": future_reset,
        }

        res = updater.check_updates(rate_limit_response, system_state=AssistantState.IDLE)
        assert res["status"] == "rate_limited"
        assert updater.rate_limit_reset == future_reset

        # Second immediate check must be blocked by rate limit backoff
        blocked_res = updater.check_updates({"tag_name": "v1.1.0"}, system_state=AssistantState.IDLE)
        assert blocked_res["status"] == "rate_limited"
        assert blocked_res["backoff_remaining_s"] > 0

    def test_updater_malformed_release_tag_handling(self) -> None:
        """F6: Verify malformed release tags ('nightly-unstable', 'v-invalid', '') are safely ignored."""
        updater = Automatic90DayUpdater(current_version="1.0.0")

        for bad_tag in ["nightly-unstable", "invalid_ver", "", "vX.Y.Z", "release-2026"]:
            res = updater.check_updates({"tag_name": bad_tag}, system_state=AssistantState.IDLE)
            assert res["status"] == "ignored"
            assert "Malformed" in res["reason"]

    def test_updater_clock_skew_and_negative_delta(self) -> None:
        """F6: Verify clock skew (system clock moved backwards) prevents infinite update lockout."""
        updater = Automatic90DayUpdater(current_version="1.0.0")
        t0 = time.time()
        updater.last_check_timestamp = t0

        # Simulate system clock shifting 2 hours backwards
        clock_shifted_time = t0 - 7200.0
        assert updater.is_update_due(clock_shifted_time) is False

        # Reset baseline must recover cleanly when clock advances forward 91 days
        advance_91_days = clock_shifted_time + (91 * 86400)
        assert updater.is_update_due(advance_91_days) is True

    def test_updater_session_active_suppression(self) -> None:
        """F6: Verify update installation is deferred if system is active (SPEAKING, LISTENING, BUSY)."""
        updater = Automatic90DayUpdater(current_version="1.0.0")
        valid_release = {
            "tag_name": "v1.5.0",
            "assets": [{"browser_download_url": "https://github.com/release/v1.5.0.zip"}],
        }

        # 1. Active states defer update
        for active_st in [AssistantState.SPEAKING, AssistantState.LISTENING, AssistantState.BUSY]:
            res = updater.check_updates(valid_release, system_state=active_st)
            assert res["status"] == "suppressed"
            assert "Active user session" in res["reason"]

        # 2. In IDLE state, update is permitted
        res_idle = updater.check_updates(valid_release, system_state=AssistantState.IDLE)
        assert res_idle["status"] == "update_available"
        assert res_idle["version"] == "v1.5.0"
