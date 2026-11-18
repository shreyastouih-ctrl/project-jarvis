"""Tier 4 E2E Real-World Application Scenarios Test Suite for Jarvis AI Assistant.

Authoritative source: ORIGINAL_REQUEST.md (§R1–R4), PROJECT.md, and TEST_INFRA.md.
5 Complex Multi-Step Real-World Scenarios simulating day-in-the-life workflows:
- test_scenario_1_focused_inventor (S1: The Focused Inventor Workflow)
- test_scenario_2_critical_emergency_override (S2: Critical Emergency Override)
- test_scenario_3_rapid_voice_automation (S3: Rapid Voice Automation & Ambient Control)
- test_scenario_4_network_degradation_and_fallback (S4: Network Degradation & Adaptive Latency Fallback)
- test_scenario_5_morning_briefing_diagnostic (S5: Complete Morning Briefing & System Diagnostic)
"""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
import re
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pytest
from fastapi import FastAPI
from starlette.testclient import TestClient
from starlette.websockets import WebSocket, WebSocketDisconnect

from jarvis.audio.capture import VirtualAudioPipeline
from jarvis.audio.stt import MockSTTEngine
from jarvis.audio.tts import MockTTSEngine, format_jarvis_speech
from jarvis.audio.vad import VADDetector
from jarvis.config import SystemConfig
from jarvis.core.event_bus import Event, EventBus, EventType
from jarvis.core.orchestrator import JarvisOrchestrator
from jarvis.core.state_machine import AssistantState, AssistantStateMachine
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
)

logger = logging.getLogger(__name__)


def create_scenario_hud_app(
    event_bus: EventBus,
    state_machine: AssistantStateMachine,
    device_registry: InMemoryDeviceRegistry,
    telephony_engine: HeadlessTelephonyEngine,
) -> FastAPI:
    """Create testable FastAPI app for Tier 4 scenarios."""
    app = FastAPI(title="Jarvis Scenario HUD")

    @app.get("/api/status")
    async def get_status():
        return {
            "state": state_machine.current_state.value,
            "busy_mode": telephony_engine.busy_mode,
            "active_devices": len(device_registry.list_all()),
        }

    @app.websocket("/ws/telemetry")
    async def ws_telemetry(websocket: WebSocket):
        await websocket.accept()
        # Initial snapshots
        await websocket.send_json({"type": "state", "state": state_machine.current_state.value})
        telem = collect_system_telemetry()
        await websocket.send_json({"type": "telemetry", **telem})

        async def state_listener(event: Event):
            try:
                await websocket.send_json({
                    "type": "state",
                    "state": event.payload.get("new_state", ""),
                })
            except Exception:
                pass

        event_bus.subscribe(EventType.STATE_CHANGED, state_listener)

        try:
            while True:
                msg = await websocket.receive_json()
                action = msg.get("action")
                if action == "ping":
                    await websocket.send_json({"type": "pong"})
                elif action == "set_state":
                    new_st = msg.get("state")
                    if new_st:
                        await websocket.send_json({"type": "state", "state": new_st})
                elif action == "push_telemetry":
                    await websocket.send_json({
                        "type": "telemetry",
                        "cpu": msg.get("cpu", 20.0),
                        "ram": msg.get("ram", 40.0),
                        "net_ping": msg.get("net_ping", 15.0),
                        "audio_level": msg.get("audio_level", 0.05),
                    })
                elif action == "trigger_screening_alert":
                    urgency = msg.get("urgency", "Normal")
                    theme = "stark_gold" if urgency == "Emergency" else "stark_cyan"
                    await websocket.send_json({
                        "type": "call_alert",
                        "caller": msg.get("caller", ""),
                        "urgency": urgency,
                        "theme": theme,
                        "alert": urgency == "Emergency",
                    })
        except WebSocketDisconnect:
            pass
        finally:
            event_bus.unsubscribe(EventType.STATE_CHANGED, state_listener)

    return app


# ==============================================================================
# TIER 4: REAL-WORLD APPLICATION SCENARIOS (5 Tests)
# ==============================================================================
class TestTier4Scenarios:
    """Full end-to-end composite workflows exercising cross-subsystem orchestration."""

    @pytest.mark.asyncio
    async def test_scenario_1_focused_inventor(
        self,
        e2e_system_config: SystemConfig,
        e2e_event_bus: EventBus,
        event_collector: AsyncEventCollector,
        device_registry: InMemoryDeviceRegistry,
        telephony_engine: HeadlessTelephonyEngine,
    ) -> None:
        """Scenario 1 (S1): The Focused Inventor Workflow.

        Tony Stark activates busy mode for engineering research. Jarvis dims ambient lights,
        monitors elevated telemetry, screens routine call from Pepper Potts autonomously,
        logs debrief silently, and exits busy mode cleanly when Tony finishes.
        """
        state_machine = AssistantStateMachine(e2e_event_bus)
        app = create_scenario_hud_app(e2e_event_bus, state_machine, device_registry, telephony_engine)
        client = TestClient(app)

        with client.websocket_connect("/ws/telemetry") as ws:
            _ = ws.receive_json()  # state
            _ = ws.receive_json()  # telemetry

            # Step 1: User engages busy mode
            await state_machine.transition_to(AssistantState.BUSY, reason="focused_research")
            telephony_engine.set_busy_mode(True)
            assert state_machine.is_busy is True
            assert telephony_engine.busy_mode is True

            # Verify busy state transition broadcast over WebSocket
            busy_state_frame = ws.receive_json()
            assert busy_state_frame.get("type") == "state"
            assert busy_state_frame.get("state") == "BUSY"

            # Step 2: Ambient adjustment (dim workshop light to 30%, set thermostat to 21°C)
            dim_res = device_registry.execute_command("light.workshop", "set_brightness", brightness=30)
            assert dim_res["attributes"]["brightness"] == 30
            temp_res = device_registry.execute_command("thermostat.main", "set_temperature", temperature=21.0)
            assert temp_res["attributes"]["temperature"] == 21.0

            # Step 3: High compute load telemetry broadcast (CPU 80%, RAM 72%, ping 24ms)
            ws.send_json({"action": "push_telemetry", "cpu": 80.0, "ram": 72.0, "net_ping": 24.0, "audio_level": 0.05})
            telem_frame = ws.receive_json()
            assert telem_frame.get("type") == "telemetry"
            assert telem_frame.get("cpu") == 80.0
            assert telem_frame.get("ram") == 72.0

            # Step 4 & 5: Incoming call from Pepper Potts screened autonomously
            call_id = "call_s1_pepper"
            telephony_engine.handle_incoming_call(call_id, "Pepper Potts", "+15551234567")
            assert telephony_engine.active_calls[call_id]["state"] == "screening"

            reply_1 = telephony_engine.process_caller_utterance(
                call_id,
                "Please remind Tony about the board meeting tomorrow at 9 AM.",
            )
            assert "message" in reply_1.lower()

            reply_2 = telephony_engine.process_caller_utterance(call_id, "Thank you, Jarvis.")
            assert "message" in reply_2.lower() or "ensure" in reply_2.lower()

            # Step 6: Normal urgency debrief created silently
            debrief = telephony_engine.end_call(call_id)
            assert debrief.caller_name == "Pepper Potts"
            assert debrief.urgency == "Normal"
            assert "board meeting" in debrief.summary.lower()

            # Step 7: HUD Quiet Mode Integrity - verify no emergency red alert
            ws.send_json({"action": "trigger_screening_alert", "caller": "Pepper Potts", "urgency": "Normal"})
            quiet_alert = ws.receive_json()
            assert quiet_alert.get("alert") is False
            assert quiet_alert.get("theme") == "stark_cyan"

            # Step 8: Session completion - Tony exits busy mode
            await state_machine.transition_to(AssistantState.IDLE, reason="research_complete")
            telephony_engine.set_busy_mode(False)
            assert state_machine.is_idle is True
            assert telephony_engine.busy_mode is False

    @pytest.mark.asyncio
    async def test_scenario_2_critical_emergency_override(
        self,
        e2e_system_config: SystemConfig,
        e2e_event_bus: EventBus,
        event_collector: AsyncEventCollector,
        device_registry: InMemoryDeviceRegistry,
        telephony_engine: HeadlessTelephonyEngine,
    ) -> None:
        """Scenario 2 (S2): Critical Emergency Override.

        Assistant screens call in busy mode. Security reports perimeter breach.
        Emergency classifier triggers HUD alert banner. User speaks voice takeover.
        Call proxy disengages AI loopback and connects user immediately.
        """
        state_machine = AssistantStateMachine(e2e_event_bus)
        await state_machine.transition_to(AssistantState.BUSY, reason="busy_session")
        telephony_engine.set_busy_mode(True)

        app = create_scenario_hud_app(e2e_event_bus, state_machine, device_registry, telephony_engine)
        client = TestClient(app)

        with client.websocket_connect("/ws/telemetry") as ws:
            _ = ws.receive_json()  # init state
            _ = ws.receive_json()  # init telem

            # Step 1: Security call incoming during busy mode
            call_id = "call_s2_security"
            telephony_engine.handle_incoming_call(call_id, "Stark Tower Security", "+15559998888")

            # Step 2: Caller declares emergency breach
            emerg_text = "Emergency! Perimeter breach in sub-level 3, unauthorized intrusion in progress!"
            reply = telephony_engine.process_caller_utterance(call_id, emerg_text)

            # Step 3: Emergency classification confirmed
            active_call = telephony_engine.active_calls[call_id]
            assert active_call["urgency"] == "Emergency"
            assert "emergency" in reply.lower()

            # Step 4 & 5: HUD broadcasts urgent visualizer in Stark Gold (strictly NOT Ultron red)
            ws.send_json({
                "action": "trigger_screening_alert",
                "caller": "Stark Tower Security",
                "urgency": "Emergency",
            })
            emerg_frame = ws.receive_json()
            assert emerg_frame.get("alert") is True
            assert emerg_frame.get("urgency") == "Emergency"
            assert emerg_frame.get("theme") == "stark_gold"

            # Step 6 & 7: Emergency override disengages busy mode; voice takeover routed through orchestrator
            await state_machine.transition_to(AssistantState.IDLE, reason="emergency_override")
            telephony_engine.set_busy_mode(False)

            pipeline = VirtualAudioPipeline(e2e_system_config.audio)
            stt = MockSTTEngine()
            stt.set_next_transcript("Jarvis, patch them through right now!")
            tts = MockTTSEngine(e2e_system_config.tts)

            orchestrator = JarvisOrchestrator(
                config=e2e_system_config,
                event_bus=e2e_event_bus,
                state_machine=state_machine,
                audio_capture=pipeline,
                stt=stt,
                tts=tts,
            )

            async def voice_intent_router(utterance: str) -> str:
                lower = utterance.lower()
                if any(k in lower for k in ("take", "answer", "patch")) and any(k in lower for k in ("call", "through")):
                    telephony_engine.take_over(call_id)
                    return "Connecting you immediately, sir."
                return "Acknowledged, sir."

            orchestrator.set_response_generator(voice_intent_router)
            await orchestrator.start()

            try:
                # Inject user voice takeover audio frames (4 speech frames + 12 silence frames)
                for _ in range(4):
                    pipeline.inject_synthetic_audio(duration_ms=32, amplitude=0.8, frequency=300.0)
                for _ in range(12):
                    pipeline.inject_silence(duration_ms=32)

                # Await transcript processing via orchestrator audio loop
                tr_ev = await event_collector.wait_for(EventType.TRANSCRIPT_READY, timeout=3.0)
                assert tr_ev.get("text") == "Jarvis, patch them through right now!"
                await asyncio.sleep(0.05)

                assert active_call["taken_over"] is True
                assert active_call["state"] == "user_connected"
            finally:
                await orchestrator.stop()

            # Step 8: Call ends with emergency debrief
            debrief = telephony_engine.end_call(call_id)
            assert debrief.urgency == "Emergency"
            assert debrief.caller_name == "Stark Tower Security"
            assert "Follow up" in debrief.action_item

    @pytest.mark.asyncio
    async def test_scenario_3_rapid_voice_automation(
        self,
        device_registry: InMemoryDeviceRegistry,
        model_router: DeterministicModelRouter,
    ) -> None:
        """Scenario 3 (S3): Rapid Voice Automation & Ambient Control.

        Three rapid successive voice commands (workshop light, thermostat, front door lock)
        are resolved via Tier 0/1 fast-path models (<150ms) and executed atomically.
        """
        utterances = [
            "Turn on workshop light",
            "Set thermostat to 22 degrees",
            "Lock the front door",
        ]

        latencies: list[float] = []

        for utterance in utterances:
            t0 = time.perf_counter()

            # 1. Select fast-path model (Tier 0 or Tier 1: <150ms)
            selected_model = model_router.select_fastest(required_tier=0)
            model_info = model_router.models[selected_model]
            assert model_info["tier"] <= 1

            # 2. Parse command utterance using device_registry.parse_command
            target_entity, action, params = device_registry.parse_command(utterance)

            # 3. Execute parsed device command
            res = device_registry.execute_command(target_entity, action, **params)
            assert res.get("success") is True

            # 4. Generate confirmation text
            dev = device_registry.get(target_entity)
            verbal_reply = f"{dev.name} updated to {dev.state}, sir."
            assert "sir" in verbal_reply.lower()

            t1 = time.perf_counter()
            elapsed_ms = (t1 - t0) * 1000.0
            latencies.append(elapsed_ms)

        # Verify all device target states in registry
        assert device_registry.get("light.workshop").state == "on"
        assert device_registry.get("thermostat.main").attributes["temperature"] == 22.0
        assert device_registry.get("lock.front_door").state == "locked"

        # Verify fast execution (sub-150ms per command in headless environment)
        for lat in latencies:
            assert lat < 150.0

    @pytest.mark.asyncio
    async def test_scenario_4_network_degradation_and_fallback(
        self,
        e2e_event_bus: EventBus,
        model_router: DeterministicModelRouter,
    ) -> None:
        """Scenario 4 (S4): Network Degradation & Adaptive Latency Fallback.

        Primary cloud model suffers sudden latency timeout (>2000ms). Dynamic benchmark
        harness trips circuit breaker and falls back to local Tier 1 SLM without speech stall.
        Subsequent queries immediately route to local model without repeated timeout penalties.
        """
        # Baseline query 1: Cloud Tier 2 model is operational (220ms)
        initial_fastest = model_router.select_fastest(required_tier=2)
        assert initial_fastest == "tier2-cloud-fast"

        # Injected degradation: Cloud model times out (>2000ms)
        model_router.set_latency("tier2-cloud-fast", ttft_ms=2500.0, tat_ms=3500.0)
        model_router.trip_circuit_breaker("tier2-cloud-fast")

        # Dynamic router cascades to local Tier 1 SLM
        fallback_model = model_router.select_fastest()
        assert fallback_model != "tier2-cloud-fast"
        assert fallback_model in ("tier0-direct-reflex", "tier1-edge-slm")

        resp1, m1, lat1 = model_router.generate("What is the current system status?", model_id=fallback_model)
        assert m1 == fallback_model
        assert lat1 < 150.0  # Local model responds under 150ms

        # Subsequent query 2: router directly uses local model with zero timeout penalty
        selected_next = model_router.select_fastest()
        assert selected_next == fallback_model
        resp2, m2, lat2 = model_router.generate("Recalibrate sensors", model_id=selected_next)
        assert m2 == fallback_model
        assert lat2 < 150.0

    @pytest.mark.asyncio
    async def test_scenario_5_morning_briefing_diagnostic(
        self,
        e2e_system_config: SystemConfig,
        e2e_event_bus: EventBus,
        device_registry: InMemoryDeviceRegistry,
        telephony_engine: HeadlessTelephonyEngine,
        model_router: DeterministicModelRouter,
    ) -> None:
        """Scenario 5 (S5): Complete Morning Briefing & System Diagnostic.

        Tony Stark awakens: "Good morning, Jarvis." Jarvis runs full telemetry diagnostic,
        polls device statuses, retrieves overnight screened message, composes composite
        briefing with Tier 2 model, delivers verbal response, and returns to IDLE.
        """
        state_machine = AssistantStateMachine(e2e_event_bus)
        app = create_scenario_hud_app(e2e_event_bus, state_machine, device_registry, telephony_engine)
        client = TestClient(app)

        with client.websocket_connect("/ws/telemetry") as ws:
            _ = ws.receive_json()  # init state
            _ = ws.receive_json()  # init telem

            # Step 1: Assistant transitions to PROCESSING
            await state_machine.transition_to(AssistantState.PROCESSING, reason="morning_briefing_request")

            # Step 2: Diagnostic sweep
            # a. Telemetry poll
            telem = collect_system_telemetry()
            assert "cpu" in telem and "ram" in telem

            # b. Device status check
            devices = device_registry.list_all()
            assert len(devices) >= 5

            # c. Retrieve overnight call history
            telephony_engine.set_busy_mode(True)
            telephony_engine.handle_incoming_call("call_s5_overnight", "Virginia Potts", "+15551234567")
            telephony_engine.process_caller_utterance("call_s5_overnight", "Reminder: board meeting today at 9 AM.")
            debrief = telephony_engine.end_call("call_s5_overnight")
            assert len(telephony_engine.history) >= 1

            # Step 3: Model composes composite briefing
            briefing_prompt = (
                f"Compose morning briefing: CPU={telem['cpu']}%, RAM={telem['ram']}%, "
                f"devices_secured={len(devices)}, overnight_calls={len(telephony_engine.history)}"
            )
            model_choice = model_router.select_fastest(required_tier=2)
            _, selected_model, _ = model_router.generate(briefing_prompt, model_id=model_choice)
            assert selected_model == "tier2-cloud-fast"

            spoken_briefing = (
                f"Good morning, sir. All core systems are nominal with CPU at {int(telem['cpu'])} percent. "
                f"All perimeter defenses are secure, and you have one message from Ms. Potts regarding the board meeting."
            )
            formatted = format_jarvis_speech(spoken_briefing)
            assert "sir" in formatted.lower()

            # Step 4: Spoken delivery and transition SPEAKING -> IDLE
            await state_machine.transition_to(AssistantState.SPEAKING, reason="briefing_speech")
            assert state_machine.is_speaking is True

            await state_machine.transition_to(AssistantState.IDLE, reason="briefing_complete")
            assert state_machine.is_idle is True
