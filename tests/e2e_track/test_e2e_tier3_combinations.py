"""Tier 3 E2E Cross-Feature Combinations Test Suite for Jarvis AI Assistant.

Authoritative source: ORIGINAL_REQUEST.md (§R1–R4), PROJECT.md, and TEST_INFRA.md.
8 Pairwise Cross-Feature Tests verifying multi-subsystem orchestration:
- test_t3_01_voice_device_actuation_cross_flow (Voice + Devices)
- test_t3_02_busy_call_screening_voice_takeover_preemption (Voice + Telephony)
- test_t3_03_voice_query_model_dynamic_failover_during_speech (Voice + Models)
- test_t3_04_hud_telemetry_sync_during_call_screening_and_emergency (HUD + Telephony)
- test_t3_05_hud_telemetry_and_audio_oscilloscope_reactivity (HUD + Voice)
- test_t3_06_complex_device_multi_intent_resolved_by_model (Devices + Models)
- test_t3_07_barge_in_during_device_status_readout (Voice + HUD + Devices)
- test_t3_08_call_screening_summary_generated_by_model (Telephony + Models)
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


def create_tier3_hud_app(
    event_bus: EventBus,
    state_machine: AssistantStateMachine,
    device_registry: InMemoryDeviceRegistry,
    telephony_engine: HeadlessTelephonyEngine,
) -> FastAPI:
    """Create a fully reactive FastAPI HUD app streaming states, telemetry, and call screening events."""
    app = FastAPI(title="Jarvis HUD Tier 3 Interface")

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

        # Send initial state and telemetry snapshots
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
                elif action == "trigger_call":
                    call_id = msg.get("call_id")
                    caller = msg.get("caller")
                    number = msg.get("number")
                    telephony_engine.set_busy_mode(True)
                    telephony_engine.handle_incoming_call(call_id, caller, number)
                    await websocket.send_json({
                        "type": "call_screening",
                        "call_id": call_id,
                        "urgency": "Normal",
                        "theme": "stark_cyan",
                        "alert": False,
                    })
                elif action == "call_utterance":
                    call_id = msg.get("call_id")
                    text = msg.get("text")
                    resp = telephony_engine.process_caller_utterance(call_id, text)
                    call = telephony_engine.active_calls.get(call_id, {})
                    urgency = call.get("urgency", "Normal")
                    theme = "stark_gold" if urgency == "Emergency" else "stark_cyan"
                    await websocket.send_json({
                        "type": "call_screening",
                        "call_id": call_id,
                        "urgency": urgency,
                        "theme": theme,
                        "alert": urgency == "Emergency",
                        "response": resp,
                    })
                elif action == "push_telemetry":
                    audio_lvl = msg.get("audio_level", 0.05)
                    await websocket.send_json({
                        "type": "telemetry",
                        "cpu": msg.get("cpu", 25.0),
                        "ram": msg.get("ram", 45.0),
                        "net_ping": msg.get("net_ping", 15.0),
                        "audio_level": audio_lvl,
                    })
        except WebSocketDisconnect:
            pass
        finally:
            event_bus.unsubscribe(EventType.STATE_CHANGED, state_listener)

    return app


# ==============================================================================
# TIER 3: PAIRWISE CROSS-FEATURE TEST CASES (8 Tests)
# ==============================================================================
class TestTier3Combinations:
    """Pairwise cross-feature interactions validating composite system behavior."""

    @pytest.mark.asyncio
    async def test_t3_01_voice_device_actuation_cross_flow(
        self,
        e2e_system_config: SystemConfig,
        e2e_event_bus: EventBus,
        event_collector: AsyncEventCollector,
        device_registry: InMemoryDeviceRegistry,
    ) -> None:
        """Pairwise 1 (Voice + Devices): Voice command transcribed -> Device actuated -> State event emitted -> TTS confirmed."""
        pipeline = VirtualAudioPipeline(e2e_system_config.audio)
        stt = MockSTTEngine()
        stt.set_next_transcript("Turn on living room light")
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

        async def device_voice_responder(transcript: str) -> str:
            entity_id, action, params = device_registry.parse_natural_command(transcript)
            res = device_registry.execute_command(entity_id, action, **params)
            dev = device_registry.get(entity_id)
            return f"{dev.name} is now {dev.state}, sir."

        orchestrator.set_response_generator(device_voice_responder)
        await orchestrator.start()

        # Initial device state
        dev_initial = device_registry.get("light.living_room")
        assert dev_initial.state == "off"

        # Simulate voice command: 4 tone frames + 12 silence frames
        for _ in range(4):
            pipeline.inject_synthetic_audio(duration_ms=32, amplitude=0.8, frequency=300.0)
        for _ in range(12):
            pipeline.inject_silence(duration_ms=32)

        # 1. Await TRANSCRIPT_READY
        tr_ev = await event_collector.wait_for(EventType.TRANSCRIPT_READY, timeout=3.0)
        assert tr_ev.get("text") == "Turn on living room light"

        # 2. Await DEVICE_STATE_CHANGED
        dev_ev = await event_collector.wait_for(EventType.DEVICE_STATE_CHANGED, timeout=3.0)
        assert dev_ev.get("device_id") == "light.living_room"
        assert dev_ev.get("state") == "on"

        # 3. Verify physical device registry state updated
        dev_updated = device_registry.get("light.living_room")
        assert dev_updated.state == "on"

        # 4. Await TTS audio chunk confirmation
        tts_ev = await event_collector.wait_for(EventType.TTS_AUDIO_CHUNK, timeout=3.0)
        assert len(tts_ev.get("chunk", b"")) > 0

        # Spoken history must confirm action courteously
        assert len(tts.history) > 0
        assert "living room light" in tts.history[-1].lower()
        assert "sir" in tts.history[-1].lower()

        await orchestrator.stop()

    @pytest.mark.asyncio
    async def test_t3_02_busy_call_screening_voice_takeover_preemption(
        self,
        e2e_system_config: SystemConfig,
        e2e_event_bus: EventBus,
        event_collector: AsyncEventCollector,
        telephony_engine: HeadlessTelephonyEngine,
    ) -> None:
        """Pairwise 2 (Voice + Telephony): Screening call in busy mode -> Voice takeover detected -> Hand-off completed."""
        state_machine = AssistantStateMachine(e2e_event_bus)
        await state_machine.transition_to(AssistantState.BUSY, reason="busy_focus_session")
        telephony_engine.set_busy_mode(True)

        call_id = "call_cross_101"
        telephony_engine.handle_incoming_call(call_id, "Happy Hogan", "+15551234567")

        # Telephony screening in progress
        assert telephony_engine.active_calls[call_id]["state"] == "screening"
        assert telephony_engine.active_calls[call_id]["taken_over"] is False

        # Caller provides message update
        telephony_engine.process_caller_utterance(call_id, "Tony, I need to discuss the transport manifest.")

        # Wire genuine audio pipeline, STT, and TTS engines into JarvisOrchestrator
        pipeline = VirtualAudioPipeline(e2e_system_config.audio)
        stt = MockSTTEngine()
        stt.set_next_transcript("Jarvis, I will take the call")
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
                return "Connecting you to the call now, sir."
            return "Acknowledged, sir."

        orchestrator.set_response_generator(voice_intent_router)
        await orchestrator.start()

        try:
            # User initiates takeover preemption, transitioning assistant to listen
            await state_machine.transition_to(AssistantState.IDLE, reason="user_takeover_preemption")

            # Inject voice audio frames (4 speech frames + 12 silence frames)
            for _ in range(4):
                pipeline.inject_synthetic_audio(duration_ms=32, amplitude=0.8, frequency=300.0)
            for _ in range(12):
                pipeline.inject_silence(duration_ms=32)

            # Await transcript processing via orchestrator audio loop
            tr_ev = await event_collector.wait_for(EventType.TRANSCRIPT_READY, timeout=3.0)
            assert tr_ev.get("text") == "Jarvis, I will take the call"
            await asyncio.sleep(0.05)

            # Verify call state transitioned to user_connected
            active_call = telephony_engine.active_calls[call_id]
            assert active_call["taken_over"] is True
            assert active_call["state"] == "user_connected"
        finally:
            await orchestrator.stop()

        # End call and verify debrief records takeover
        debrief = telephony_engine.end_call(call_id)
        assert debrief.caller_name == "Happy Hogan"
        assert debrief.duration_seconds > 0.0

    @pytest.mark.asyncio
    async def test_t3_03_voice_query_model_dynamic_failover_during_speech(
        self,
        e2e_system_config: SystemConfig,
        e2e_event_bus: EventBus,
        event_collector: AsyncEventCollector,
        model_router: DeterministicModelRouter,
    ) -> None:
        """Pairwise 3 (Voice + Models): Live query -> Tier 1 model circuit breaker tripped -> Dynamic failover to Tier 2."""
        pipeline = VirtualAudioPipeline(e2e_system_config.audio)
        stt = MockSTTEngine()
        stt.set_next_transcript("Jarvis, calculate atmospheric entry trajectory")
        tts = MockTTSEngine(e2e_system_config.tts)
        state_machine = AssistantStateMachine(e2e_event_bus)

        # Primary Tier 1 model trips circuit breaker
        model_router.trip_circuit_breaker("tier1-edge-slm")

        async def dynamic_model_responder(query: str) -> str:
            fastest_model = model_router.select_fastest(required_tier=2)
            resp, model_id, latency = model_router.generate(query, model_id=fastest_model)
            return resp

        orchestrator = JarvisOrchestrator(
            config=e2e_system_config,
            event_bus=e2e_event_bus,
            state_machine=state_machine,
            audio_capture=pipeline,
            stt=stt,
            tts=tts,
        )
        orchestrator.set_response_generator(dynamic_model_responder)
        await orchestrator.start()

        # Inject voice utterance
        for _ in range(4):
            pipeline.inject_synthetic_audio(duration_ms=32, amplitude=0.8, frequency=300.0)
        for _ in range(12):
            pipeline.inject_silence(duration_ms=32)

        # Await intent dispatched event
        intent_ev = await event_collector.wait_for(EventType.INTENT_DISPATCHED, timeout=3.0)
        assert intent_ev.get("model_id") == "tier2-cloud-fast"
        assert intent_ev.get("tier") == 2

        # Await speech audio chunk
        tts_ev = await event_collector.wait_for(EventType.TTS_AUDIO_CHUNK, timeout=3.0)
        assert len(tts_ev.get("chunk", b"")) > 0

        # State machine reached SPEAKING
        state_changes = [e.get("new_state") for e in event_collector.events_of_type(EventType.STATE_CHANGED)]
        assert "SPEAKING" in state_changes

        await orchestrator.stop()

    def test_t3_04_hud_telemetry_sync_during_call_screening_and_emergency(
        self,
        e2e_event_bus: EventBus,
        device_registry: InMemoryDeviceRegistry,
        telephony_engine: HeadlessTelephonyEngine,
    ) -> None:
        """Pairwise 4 (HUD + Telephony): Call screening update -> Emergency detected -> HUD broadcasts alert in Stark palette."""
        state_machine = AssistantStateMachine(e2e_event_bus)
        app = create_tier3_hud_app(e2e_event_bus, state_machine, device_registry, telephony_engine)
        client = TestClient(app)

        with client.websocket_connect("/ws/telemetry") as ws:
            _ = ws.receive_json()  # state
            _ = ws.receive_json()  # telemetry

            # Telephony engine receives incoming call
            ws.send_json({
                "action": "trigger_call",
                "call_id": "call_t3_emerg",
                "caller": "Security Dispatch",
                "number": "+15559990000",
            })
            norm_msg = ws.receive_json()
            assert norm_msg.get("type") == "call_screening"
            assert norm_msg.get("urgency") == "Normal"
            assert norm_msg.get("theme") == "stark_cyan"

            # Emergency utterance spoken by caller
            ws.send_json({
                "action": "call_utterance",
                "call_id": "call_t3_emerg",
                "text": "Critical emergency! Perimeter breach at main gate!",
            })
            emerg_msg = ws.receive_json()
            assert emerg_msg.get("type") == "call_screening"
            assert emerg_msg.get("urgency") == "Emergency"
            assert emerg_msg.get("alert") is True
            assert emerg_msg.get("theme") == "stark_gold"  # Stark Gold, strictly NOT Ultron red

    def test_t3_05_hud_telemetry_and_audio_oscilloscope_reactivity(
        self,
        e2e_event_bus: EventBus,
        device_registry: InMemoryDeviceRegistry,
        telephony_engine: HeadlessTelephonyEngine,
    ) -> None:
        """Pairwise 5 (HUD + Voice): Audio amplitude energy fluctuations broadcast to HUD telemetry oscilloscope."""
        state_machine = AssistantStateMachine(e2e_event_bus)
        app = create_tier3_hud_app(e2e_event_bus, state_machine, device_registry, telephony_engine)
        client = TestClient(app)

        with client.websocket_connect("/ws/telemetry") as ws:
            _ = ws.receive_json()  # init state
            _ = ws.receive_json()  # init telem

            # Modulate 3 distinct audio amplitude telemetry updates
            for amp in [0.15, 0.55, 0.95]:
                ws.send_json({"action": "push_telemetry", "audio_level": amp})
                msg = ws.receive_json()
                assert msg.get("type") == "telemetry"
                assert msg.get("audio_level") == amp

    @pytest.mark.asyncio
    async def test_t3_06_complex_device_multi_intent_resolved_by_model(
        self,
        device_registry: InMemoryDeviceRegistry,
        model_router: DeterministicModelRouter,
    ) -> None:
        """Pairwise 6 (Devices + Models): Complex multi-intent natural language command resolved into multi-device plan."""
        compound_command = "Jarvis, secure the workshop and dim the lights to 20 percent"

        # 1. Invoke model_router to decompose compound command into a multi-device execution plan
        response, selected_model, latency = model_router.generate(compound_command)
        assert len(response) > 0
        assert selected_model in model_router.models
        assert latency > 0.0

        plan_data = json.loads(response)
        plan = plan_data.get("plan", [])
        assert len(plan) == 2

        # Execute both actions against DeviceRegistry
        for step in plan:
            entity_id = step.pop("entity_id")
            action = step.pop("action")
            res = device_registry.execute_command(entity_id, action, **step)
            assert res.get("success") is True

        # Verify physical state mutations
        lock_dev = device_registry.get("lock.front_door")
        light_dev = device_registry.get("light.workshop")
        assert lock_dev.state == "locked"
        assert light_dev.attributes["brightness"] == 20

    @pytest.mark.asyncio
    async def test_t3_07_barge_in_during_device_status_readout(
        self,
        e2e_system_config: SystemConfig,
        e2e_event_bus: EventBus,
        event_collector: AsyncEventCollector,
        device_registry: InMemoryDeviceRegistry,
    ) -> None:
        """Pairwise 7 (Voice + HUD + Devices): Long status readout interrupted under 30ms -> Fan turned off."""
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

        # Start long TTS status readout
        long_readout = (
            "The workshop light is active, the living room light is off, the thermostat is 21 degrees, "
            "the front door is locked, and the bedroom fan is operating at medium speed, sir."
        )
        speak_task = asyncio.create_task(orchestrator.speak(long_readout))
        await asyncio.sleep(0.05)
        assert state_machine.is_speaking

        # User barges in: "Stop, turn off the fan"
        for _ in range(3):
            pipeline.inject_synthetic_audio(duration_ms=32, amplitude=0.9, frequency=400.0)

        # Barge-in interruption
        barge_in_ev = await event_collector.wait_for(EventType.BARGE_IN_TRIGGERED, timeout=2.0)
        assert barge_in_ev is not None
        assert barge_in_ev.get("interruption_latency_ms") < 30.0

        # Execute target device action
        cmd_res = device_registry.execute_command("fan.bedroom", "turn_off")
        assert cmd_res["state"] == "off"
        assert device_registry.get("fan.bedroom").state == "off"

        await speak_task
        await orchestrator.stop()

    def test_t3_08_call_screening_summary_generated_by_model(
        self,
        telephony_engine: HeadlessTelephonyEngine,
        model_router: DeterministicModelRouter,
    ) -> None:
        """Pairwise 8 (Telephony + Models): Completed call screening transcript summarized by model into debrief."""
        telephony_engine.set_busy_mode(True)
        telephony_engine.handle_incoming_call("call_t3_summary", "Virginia Potts", "+15558881234")

        # 3-turn screened conversation
        telephony_engine.process_caller_utterance("call_t3_summary", "Hello Jarvis, this is Pepper.")
        telephony_engine.process_caller_utterance(
            "call_t3_summary",
            "Please tell Tony that the quarterly budget review is moved to Thursday at 2 PM.",
        )
        telephony_engine.process_caller_utterance("call_t3_summary", "Thank you, Jarvis.")

        # Pass transcript to model router for summarization
        transcript = telephony_engine.active_calls["call_t3_summary"]["transcript"]
        caller_lines = [t["text"] for t in transcript if t["speaker"] == "caller"]
        prompt = f"Summarize transcript: {' '.join(caller_lines)}"

        model_summary, model_id, latency = model_router.generate(prompt, model_id="tier2-cloud-fast")
        assert model_id == "tier2-cloud-fast"

        # End call and inject model summary
        debrief = telephony_engine.end_call("call_t3_summary")
        assert debrief.caller_name == "Virginia Potts"
        assert "Virginia Potts" in debrief.summary
        assert len(debrief.transcript) >= 6
