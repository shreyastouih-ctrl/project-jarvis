"""Pytest fixtures and test harnesses for the opaque-box E2E test suite (Tiers 1-4).

Authoritative source: ORIGINAL_REQUEST.md, PROJECT.md, and TEST_INFRA.md.
Integrates with:
- jarvis.core.event_bus
- jarvis.core.state_machine
- jarvis.core.orchestrator
- jarvis.audio.capture
- jarvis.audio.vad
- jarvis.audio.stt
- jarvis.audio.tts
- In-memory DeviceRegistryProtocol
- Headless CallEngineProtocol
- Synthetic ModelRouterProtocol
- FastAPI HUD WebSocket & REST TestClient
"""

from __future__ import annotations

import asyncio
import collections
import logging
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, AsyncGenerator, Callable, Dict, Iterator, List, Optional, Tuple, Union

import numpy as np
import pytest
from fastapi import FastAPI
from starlette.testclient import TestClient
from starlette.websockets import WebSocket, WebSocketDisconnect

# Core and Audio Subsystem imports
from jarvis.audio.capture import AudioCaptureProtocol, VirtualAudioPipeline
from jarvis.audio.stt import MockSTTEngine, STTEngineProtocol
from jarvis.audio.tts import MockTTSEngine, TTSEngineProtocol, format_jarvis_speech
from jarvis.audio.vad import VADDetector, VADFrameResult, VADProtocol
from jarvis.config import AudioConfig, SystemConfig, TTSConfig, VADConfig
from jarvis.core.event_bus import Event, EventBus, EventType
from jarvis.core.orchestrator import JarvisOrchestrator
from jarvis.core.state_machine import AssistantState, AssistantStateMachine

logger = logging.getLogger(__name__)


def safe_create_task(coro) -> None:
    """Safely schedule a coroutine task if an asyncio event loop is currently running."""
    try:
        loop = asyncio.get_running_loop()
        loop.create_task(coro)
    except RuntimeError:
        # Synchronous test execution without active event loop: close coroutine cleanly
        try:
            coro.close()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# 1. Headless Async Event Collector
# ---------------------------------------------------------------------------

class AsyncEventCollector:
    """Thread-safe and async collector capturing EventBus events for test assertions."""

    def __init__(self, bus: EventBus) -> None:
        self.bus = bus
        self.events: list[Event] = []
        self._lock = asyncio.Lock()
        # Wildcard subscription captures all event types
        self.bus.subscribe(None, self._record_event)

    async def _record_event(self, event: Event) -> None:
        async with self._lock:
            self.events.append(event)

    def events_of_type(self, event_type: Union[EventType, str]) -> list[Event]:
        """Return all recorded events matching the target event type."""
        target = event_type.value if isinstance(event_type, EventType) else str(event_type)
        return [
            e for e in self.events
            if (e.event_type.value if isinstance(e.event_type, EventType) else str(e.event_type)) == target
        ]

    def has_event(self, event_type: Union[EventType, str]) -> bool:
        """Check if at least one matching event has been received."""
        return len(self.events_of_type(event_type)) > 0

    def get_last(self, event_type: Optional[Union[EventType, str]] = None) -> Optional[Event]:
        """Return the most recent event matching event_type, or latest overall."""
        if event_type is None:
            return self.events[-1] if self.events else None
        matches = self.events_of_type(event_type)
        return matches[-1] if matches else None

    async def wait_for(
        self,
        event_type: Union[EventType, str],
        timeout: float = 3.0,
        predicate: Optional[Callable[[Event], bool]] = None,
    ) -> Event:
        """Wait for an event to arrive (checks already-collected events first)."""
        target = event_type.value if isinstance(event_type, EventType) else str(event_type)
        # Check already received events
        for e in self.events:
            ev_str = e.event_type.value if isinstance(e.event_type, EventType) else str(e.event_type)
            if ev_str == target:
                if predicate is None or predicate(e):
                    return e

        # Await on the underlying bus
        ev_enum = event_type if isinstance(event_type, EventType) else EventType(event_type)
        return await self.bus.wait_for(ev_enum, timeout=timeout, predicate=predicate)

    def clear(self) -> None:
        """Clear recorded events."""
        self.events.clear()

    def teardown(self) -> None:
        """Unsubscribe handler from the bus."""
        self.bus.unsubscribe(None, self._record_event)


# ---------------------------------------------------------------------------
# 2. Deterministic Audio Stream Generator
# ---------------------------------------------------------------------------

class AudioStreamGenerator:
    """Generates mathematically exact 16kHz 16-bit mono PCM waveforms."""

    def __init__(self, sample_rate: int = 16000, sample_width: int = 2) -> None:
        self.sample_rate = sample_rate
        self.sample_width = sample_width

    def silence(self, duration_ms: float) -> bytes:
        """Generate zero-amplitude PCM silence."""
        total_samples = int((duration_ms / 1000.0) * self.sample_rate)
        return b"\x00" * (total_samples * self.sample_width)

    def tone(self, duration_ms: float, frequency: float = 440.0, amplitude: float = 0.5) -> bytes:
        """Generate pure sine tone."""
        total_samples = int((duration_ms / 1000.0) * self.sample_rate)
        t = np.arange(total_samples) / float(self.sample_rate)
        peak = int(32767 * max(0.0, min(1.0, amplitude)))
        waveform = (peak * np.sin(2.0 * np.pi * frequency * t)).astype(np.int16)
        return waveform.tobytes()

    def speech_like(self, duration_ms: float, base_freq: float = 220.0, amplitude: float = 0.7) -> bytes:
        """Generate multi-harmonic speech-like waveform (f0, 2*f0, 3*f0) for VAD triggering."""
        total_samples = int((duration_ms / 1000.0) * self.sample_rate)
        t = np.arange(total_samples) / float(self.sample_rate)
        h1 = np.sin(2.0 * np.pi * base_freq * t)
        h2 = 0.5 * np.sin(2.0 * np.pi * (2 * base_freq) * t)
        h3 = 0.25 * np.sin(2.0 * np.pi * (3 * base_freq) * t)
        combined = (h1 + h2 + h3) / 1.75
        peak = int(32767 * max(0.0, min(1.0, amplitude)))
        waveform = (peak * combined).astype(np.int16)
        return waveform.tobytes()

    def noise(self, duration_ms: float, amplitude: float = 0.05) -> bytes:
        """Generate low-amplitude white noise for background floor testing."""
        total_samples = int((duration_ms / 1000.0) * self.sample_rate)
        peak = int(32767 * max(0.0, min(1.0, amplitude)))
        return np.random.randint(-peak, peak, size=total_samples, dtype=np.int16).tobytes()

    def inject_speech_burst(
        self,
        pipeline: VirtualAudioPipeline,
        speech_ms: float = 500.0,
        trailing_silence_ms: float = 400.0,
    ) -> Tuple[int, int]:
        """Inject speech frames followed by sufficient silence hangover to trigger speech completion."""
        s_chunks = pipeline.inject_pcm_data(self.speech_like(speech_ms))
        sil_chunks = pipeline.inject_pcm_data(self.silence(trailing_silence_ms))
        return (s_chunks, sil_chunks)


# ---------------------------------------------------------------------------
# 3. Isolated In-Memory Device Registry
# ---------------------------------------------------------------------------

@dataclass
class DeviceEntity:
    """HomeAssistant-compatible entity model."""
    entity_id: str
    domain: str
    name: str
    state: str
    attributes: dict[str, Any] = field(default_factory=dict)


class InMemoryDeviceRegistry:
    """In-memory smart device registry implementing DeviceRegistryProtocol."""

    def __init__(self, event_bus: Optional[EventBus] = None) -> None:
        self.event_bus = event_bus
        self._devices: dict[str, DeviceEntity] = {}
        self._seed_default_entities()

    def _seed_default_entities(self) -> None:
        defaults = [
            DeviceEntity("light.living_room", "light", "Living Room Light", "off", {"brightness": 100}),
            DeviceEntity("light.workshop", "light", "Workshop Light", "on", {"brightness": 80}),
            DeviceEntity("switch.main_power", "switch", "Main Power", "on", {}),
            DeviceEntity("thermostat.main", "thermostat", "Main Thermostat", "cool", {"temperature": 21.0, "target_temperature": 21.0}),
            DeviceEntity("lock.front_door", "lock", "Front Door", "locked", {}),
            DeviceEntity("fan.bedroom", "fan", "Bedroom Fan", "off", {"speed": 0}),
            DeviceEntity("system.arc_reactor", "system", "Arc Reactor Core", "nominal", {"power_level": 100.0, "status": "online"}),
        ]
        for d in defaults:
            self._devices[d.entity_id] = d

    def get(self, entity_id: str) -> Optional[DeviceEntity]:
        return self._devices.get(entity_id)

    def list_all(self) -> list[DeviceEntity]:
        return list(self._devices.values())

    def update_state(self, entity_id: str, state: str, attributes: Optional[dict[str, Any]] = None) -> DeviceEntity:
        if entity_id not in self._devices:
            raise KeyError(f"Device {entity_id} not found in registry")
        dev = self._devices[entity_id]
        dev.state = state
        if attributes:
            dev.attributes.update(attributes)
        if self.event_bus:
            safe_create_task(
                self.event_bus.publish_event(
                    EventType.DEVICE_STATE_CHANGED,
                    device_id=entity_id,
                    state=state,
                    traits=dict(dev.attributes),
                )
            )
        return dev

    def execute_command(self, entity_id: str, action: str, **params: Any) -> dict[str, Any]:
        if entity_id not in self._devices:
            return {"success": False, "error": f"Device {entity_id} not found"}
        dev = self._devices[entity_id]
        if action == "turn_on":
            dev.state = "on"
            if "brightness" in params:
                dev.attributes["brightness"] = max(1, min(100, int(params["brightness"])))
        elif action == "turn_off":
            dev.state = "off"
        elif action == "toggle":
            dev.state = "off" if dev.state == "on" else "on"
        elif action == "set_brightness":
            raw_b = params.get("brightness", 100)
            dev.attributes["brightness"] = max(1, min(100, int(raw_b)))
            dev.state = "on"
        elif action == "set_temperature":
            raw_t = params.get("temperature", 21.0)
            dev.attributes["temperature"] = max(16.0, min(30.0, float(raw_t)))
        elif action == "lock":
            dev.state = "locked"
        elif action == "unlock":
            dev.state = "unlocked"

        for k, v in params.items():
            if k not in ("brightness", "temperature"):
                dev.attributes[k] = v

        if self.event_bus:
            safe_create_task(
                self.event_bus.publish_event(
                    EventType.DEVICE_STATE_CHANGED,
                    device_id=entity_id,
                    state=dev.state,
                    traits=dict(dev.attributes),
                )
            )
        return {"success": True, "entity_id": entity_id, "state": dev.state, "attributes": dev.attributes}

    def parse_natural_command(self, utterance: str) -> Tuple[str, str, Dict[str, Any]]:
        """Parse natural language command into structured (entity_id, action, params) tuple."""
        text = utterance.lower().strip()
        # Thermostat commands
        if "thermostat" in text or "temperature" in text:
            m = re.search(r"(\d+(?:\.\d+)?)", text)
            temp = float(m.group(1)) if m else 22.0
            return ("thermostat.main", "set_temperature", {"temperature": temp})

        # Lock commands
        if "lock" in text:
            if "unlock" in text:
                return ("lock.front_door", "unlock", {})
            return ("lock.front_door", "lock", {})

        # Light commands
        if "light" in text:
            target = "light.workshop" if "workshop" in text else "light.living_room"
            if "brightness" in text or "dim" in text or "%" in text:
                m = re.search(r"(\d+)", text)
                b = int(m.group(1)) if m else 80
                return (target, "set_brightness", {"brightness": b})
            elif "off" in text:
                return (target, "turn_off", {})
            elif "on" in text:
                return (target, "turn_on", {})

        # Switch / power commands
        if "power" in text or "switch" in text:
            if "off" in text:
                return ("switch.main_power", "turn_off", {})
            return ("switch.main_power", "turn_on", {})

        return ("system.arc_reactor", "query", {})

    parse_command = parse_natural_command


# ---------------------------------------------------------------------------
# 4. Headless Telephony Engine
# ---------------------------------------------------------------------------

@dataclass
class CallDebriefRecord:
    """Structured call debrief record."""
    call_id: str
    caller_name: str
    caller_number: str
    timestamp: str
    duration_seconds: float
    summary: str
    urgency: str  # "Low", "Normal", "High", "Emergency"
    action_item: str
    transcript: list[dict[str, str]]


class HeadlessTelephonyEngine:
    """Autonomous call screening and telephony engine adhering to CallEngineProtocol."""

    def __init__(self, event_bus: Optional[EventBus] = None) -> None:
        self.busy_mode: bool = False
        self.event_bus = event_bus
        self.active_calls: dict[str, dict[str, Any]] = {}
        self.history: list[CallDebriefRecord] = []
        self._emergency_keywords = {"emergency", "crash", "critical", "hospital", "urgent", "reactor breach"}

    def set_busy_mode(self, enabled: bool) -> None:
        self.busy_mode = enabled

    def handle_incoming_call(self, call_id: str, caller_name: str, caller_number: str) -> None:
        call = {
            "call_id": call_id,
            "caller_name": caller_name,
            "caller_number": caller_number,
            "start_time": time.time(),
            "transcript": [],
            "urgency": "Normal",
            "state": "screening" if self.busy_mode else "ringing",
            "taken_over": False,
        }
        self.active_calls[call_id] = call

        if self.event_bus:
            safe_create_task(
                self.event_bus.publish_event(
                    EventType.CALL_INCOMING,
                    call_id=call_id,
                    caller=caller_name,
                    number=caller_number,
                )
            )

        if self.busy_mode:
            greeting = "Good day, Mr. Stark is currently unavailable. May I take a message?"
            call["transcript"].append({"speaker": "jarvis", "text": greeting, "time": time.strftime("%H:%M:%S")})
            if self.event_bus:
                safe_create_task(
                    self.event_bus.publish_event(
                        EventType.CALL_SCREENING_UPDATE,
                        call_id=call_id,
                        state="screening",
                        transcript=list(call["transcript"]),
                    )
                )

    def process_caller_utterance(self, call_id: str, text: str) -> str:
        if call_id not in self.active_calls:
            raise KeyError(f"Call {call_id} not active")
        call = self.active_calls[call_id]
        now_str = time.strftime("%H:%M:%S")
        call["transcript"].append({"speaker": "caller", "text": text, "time": now_str})

        lower_text = text.lower()
        if any(kw in lower_text for kw in self._emergency_keywords):
            call["urgency"] = "Emergency"
            response = "I detect a critical emergency. Alerting Mr. Stark immediately."
        elif "important" in lower_text or "asap" in lower_text:
            call["urgency"] = "High"
            response = "Understood, sir. I have marked this as high priority."
        else:
            response = "I have noted that message. I will ensure he receives it."

        call["transcript"].append({"speaker": "jarvis", "text": response, "time": now_str})

        if self.event_bus:
            safe_create_task(
                self.event_bus.publish_event(
                    EventType.CALL_SCREENING_UPDATE,
                    call_id=call_id,
                    state="screening",
                    transcript=list(call["transcript"]),
                    urgency=call["urgency"],
                )
            )
        return response

    def take_over(self, call_id: str) -> None:
        if call_id not in self.active_calls:
            return
        call = self.active_calls[call_id]
        call["taken_over"] = True
        call["state"] = "user_connected"
        if self.event_bus:
            safe_create_task(
                self.event_bus.publish_event(
                    EventType.CALL_SCREENING_UPDATE,
                    call_id=call_id,
                    state="user_connected",
                    transcript=list(call["transcript"]),
                )
            )

    def end_call(self, call_id: str) -> CallDebriefRecord:
        if call_id not in self.active_calls:
            raise KeyError(f"Call {call_id} not active")
        call = self.active_calls.pop(call_id)
        duration = max(0.5, time.time() - call["start_time"])
        caller_msgs = [t["text"] for t in call["transcript"] if t["speaker"] == "caller"]
        summary = f"Call with {call['caller_name']}: " + ("; ".join(caller_msgs) if caller_msgs else "No message left")
        action = f"Follow up with {call['caller_name']} ({call['caller_number']})" if call["urgency"] in ("High", "Emergency") else "No immediate action required"
        debrief = CallDebriefRecord(
            call_id=call_id,
            caller_name=call["caller_name"],
            caller_number=call["caller_number"],
            timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ"),
            duration_seconds=round(duration, 2),
            summary=summary,
            urgency=call["urgency"],
            action_item=action,
            transcript=list(call["transcript"]),
        )
        self.history.append(debrief)
        if self.event_bus:
            safe_create_task(
                self.event_bus.publish_event(
                    EventType.CALL_DEBRIEF_CREATED,
                    call_id=call_id,
                    caller_name=debrief.caller_name,
                    caller_number=debrief.caller_number,
                    summary=debrief.summary,
                    urgency=debrief.urgency,
                    action_item=debrief.action_item,
                    transcript=debrief.transcript,
                )
            )
        return debrief


# ---------------------------------------------------------------------------
# 5. Deterministic Synthetic Model Provider & Router
# ---------------------------------------------------------------------------

@dataclass
class ModelLatencyMetrics:
    """Latency benchmark metrics for candidate models."""
    model_id: str
    tier: int               # 0, 1, 2, 3
    dns_ms: float
    tcp_ms: float
    ttft_ms: float
    tat_ms: float
    tps: float
    healthy: bool


def classify_model_tier(ttft_ms: float) -> int:
    """Classify model response latency into the 4-Tier taxonomy."""
    if ttft_ms < 50.0:
        return 0  # Tier 0 Direct Reflex (<50ms)
    elif ttft_ms < 150.0:
        return 1  # Tier 1 Edge SLM (<150ms)
    elif ttft_ms < 500.0:
        return 2  # Tier 2 Cloud Fast (<500ms)
    else:
        return 3  # Tier 3 Deep Reasoning (>=500ms)


class DeterministicModelRouter:
    """Deterministic model router implementing ModelRouterProtocol with mock latency profiles."""

    def __init__(self, event_bus: Optional[EventBus] = None) -> None:
        self.event_bus = event_bus
        self.models: dict[str, dict[str, Any]] = {
            "tier0-direct-reflex": {"tier": 0, "dns_ms": 1.0, "tcp_ms": 1.0, "ttft_ms": 15.0, "tat_ms": 25.0, "tps": 200.0, "healthy": True},
            "tier1-edge-slm":      {"tier": 1, "dns_ms": 2.0, "tcp_ms": 3.0, "ttft_ms": 85.0, "tat_ms": 110.0, "tps": 80.0, "healthy": True},
            "tier2-cloud-fast":    {"tier": 2, "dns_ms": 15.0, "tcp_ms": 25.0, "ttft_ms": 220.0, "tat_ms": 350.0, "tps": 60.0, "healthy": True},
            "tier3-deep-reason":   {"tier": 3, "dns_ms": 20.0, "tcp_ms": 30.0, "ttft_ms": 650.0, "tat_ms": 950.0, "tps": 40.0, "healthy": True},
        }
        self.circuit_tripped: set[str] = set()

    def set_latency(self, model_id: str, ttft_ms: float, tat_ms: Optional[float] = None) -> None:
        if model_id in self.models:
            self.models[model_id]["ttft_ms"] = ttft_ms
            self.models[model_id]["tat_ms"] = tat_ms or (ttft_ms + 50.0)

    def set_healthy(self, model_id: str, healthy: bool) -> None:
        if model_id in self.models:
            self.models[model_id]["healthy"] = healthy

    def trip_circuit_breaker(self, model_id: str) -> None:
        self.circuit_tripped.add(model_id)

    def reset_circuit_breaker(self, model_id: str) -> None:
        self.circuit_tripped.discard(model_id)

    def benchmark_all(self) -> list[ModelLatencyMetrics]:
        results = []
        for m_id, info in self.models.items():
            results.append(ModelLatencyMetrics(
                model_id=m_id,
                tier=info["tier"],
                dns_ms=info["dns_ms"],
                tcp_ms=info["tcp_ms"],
                ttft_ms=info["ttft_ms"],
                tat_ms=info["tat_ms"],
                tps=info["tps"],
                healthy=info["healthy"] and (m_id not in self.circuit_tripped),
            ))
        return results

    def select_fastest(self, required_tier: Optional[int] = None) -> str:
        candidates = [
            (m_id, info) for m_id, info in self.models.items()
            if info["healthy"] and m_id not in self.circuit_tripped
            and (required_tier is None or info["tier"] == required_tier)
        ]
        if not candidates:
            # Fallback to any healthy model
            healthy = [
                (m_id, info) for m_id, info in self.models.items()
                if info["healthy"] and m_id not in self.circuit_tripped
            ]
            if not healthy:
                raise RuntimeError("All candidate models are unhealthy or circuit-broken")
            candidates = healthy
        candidates.sort(key=lambda x: x[1]["ttft_ms"])
        return candidates[0][0]

    def generate(self, prompt: str, model_id: Optional[str] = None) -> Tuple[str, str, float]:
        selected = model_id or self.select_fastest()
        info = self.models[selected]
        if not info["healthy"] or selected in self.circuit_tripped:
            raise RuntimeError(f"Model {selected} is unhealthy or circuit-broken")
        latency = info["ttft_ms"]

        p_lower = prompt.lower()
        if "plan" in p_lower or ("secure" in p_lower and "workshop" in p_lower):
            plan = [
                {"entity_id": "lock.front_door", "action": "lock"},
                {"entity_id": "light.workshop", "action": "set_brightness", "brightness": 20},
            ]
            import json
            response = json.dumps({"plan": plan, "reply": "Securing workshop and setting lights to 20 percent, sir."})
        elif "summarize" in p_lower:
            response = "Summary of conversation: Caller requested message delivery to Tony."
        else:
            response = f"Acknowledged, sir. Processed via {selected}."

        if self.event_bus:
            safe_create_task(
                self.event_bus.publish_event(
                    EventType.INTENT_DISPATCHED,
                    intent=prompt,
                    model_id=selected,
                    tier=info["tier"],
                )
            )
        return (response, selected, latency)


class SyntheticModelProvider(DeterministicModelRouter):
    """Deterministic synthetic model provider for headless offline CI testing."""
    pass


# ---------------------------------------------------------------------------
# 6. System Telemetry Collector Function
# ---------------------------------------------------------------------------

def collect_system_telemetry() -> dict[str, float]:
    """Collect live system telemetry metrics (CPU, RAM, network latency)."""
    import psutil
    cpu = float(psutil.cpu_percent(interval=None))
    ram = float(psutil.virtual_memory().percent)
    net_ping = 12.0
    audio_level = 0.05
    return {
        "cpu": cpu,
        "ram": ram,
        "net_ping": net_ping,
        "audio_level": audio_level,
    }


# ---------------------------------------------------------------------------
# 7. FastAPI HUD Server App & TestClient
# ---------------------------------------------------------------------------

def create_hud_test_app(
    event_bus: EventBus,
    state_machine: AssistantStateMachine,
    registry: InMemoryDeviceRegistry,
    telephony: HeadlessTelephonyEngine,
):
    """Create testable FastAPI application for HUD endpoints."""
    from fastapi.responses import HTMLResponse, JSONResponse

    app = FastAPI(title="Jarvis HUD Interface")
    rolling_log_buffer: collections.deque = collections.deque(maxlen=100)

    @app.get("/", response_class=HTMLResponse)
    async def get_index():
        return (
            "<!DOCTYPE html><html><head><title>Jarvis HUD</title></head>"
            "<body style='background:#070A10;color:#00E5FF;'>"
            "<div id='arc-reactor'></div><div class='telemetry-gauge'></div>"
            "</body></html>"
        )

    @app.get("/api/status")
    async def get_status():
        return {
            "state": state_machine.current_state.value,
            "busy_mode": telephony.busy_mode,
            "active_devices": len(registry.list_all()),
            "status": "online",
        }

    @app.get("/api/telemetry")
    async def get_telemetry():
        return collect_system_telemetry()

    @app.post("/api/telemetry/push")
    async def push_telemetry_post(payload: dict[str, Any]):
        raw_cpu = float(payload.get("cpu", 0.0))
        raw_ram = float(payload.get("ram", 0.0))
        raw_ping = float(payload.get("net_ping", 0.0))
        raw_audio = float(payload.get("audio_level", 0.0))
        sanitized_cpu = max(0.0, min(100.0, raw_cpu))
        sanitized_ram = max(0.0, min(100.0, raw_ram if raw_ram <= 100.0 else raw_ram % 100.0))
        sanitized_audio = max(0.0, min(1.0, raw_audio))
        is_offline = raw_ping < 0
        return {
            "type": "telemetry_sanitized",
            "cpu": sanitized_cpu,
            "ram": sanitized_ram,
            "audio_level": sanitized_audio,
            "is_offline": is_offline,
        }

    @app.post("/api/telemetry/logs")
    async def push_logs_post(payload: dict[str, Any]):
        lines = payload.get("logs", [])
        for line in lines:
            rolling_log_buffer.append(line)
        return {
            "buffer_len": len(rolling_log_buffer),
            "maxlen": rolling_log_buffer.maxlen,
            "oldest": rolling_log_buffer[0] if rolling_log_buffer else None,
            "newest": rolling_log_buffer[-1] if rolling_log_buffer else None,
        }

    @app.websocket("/ws/telemetry")
    async def ws_telemetry(websocket: WebSocket):
        await websocket.accept()
        # Broadcast initial state snapshot
        await websocket.send_json({
            "type": "state",
            "state": state_machine.current_state.value,
        })
        # Broadcast initial telemetry snapshot
        telem = collect_system_telemetry()
        await websocket.send_json({
            "type": "telemetry",
            **telem,
        })

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
                if action == "toggle_busy":
                    telephony.set_busy_mode(msg.get("busy", True))
                    await websocket.send_json({"type": "busy_updated", "busy": telephony.busy_mode})
                elif action == "set_state":
                    new_st = msg.get("state")
                    if new_st:
                        await websocket.send_json({"type": "state", "state": new_st})
                elif action == "ping":
                    await websocket.send_json({"type": "pong"})
                elif action == "push_telemetry":
                    raw_cpu = float(msg.get("cpu", 0.0))
                    raw_ram = float(msg.get("ram", 0.0))
                    raw_ping = float(msg.get("net_ping", 0.0))
                    raw_audio = float(msg.get("audio_level", 0.0))
                    sanitized_cpu = max(0.0, min(100.0, raw_cpu))
                    sanitized_ram = max(0.0, min(100.0, raw_ram if raw_ram <= 100.0 else raw_ram % 100.0))
                    sanitized_audio = max(0.0, min(1.0, raw_audio))
                    is_offline = raw_ping < 0
                    await websocket.send_json({
                        "type": "telemetry_sanitized",
                        "cpu": sanitized_cpu,
                        "ram": sanitized_ram,
                        "audio_level": sanitized_audio,
                        "is_offline": is_offline,
                    })
                elif action == "push_logs":
                    lines = msg.get("logs", [])
                    for line in lines:
                        rolling_log_buffer.append(line)
                    await websocket.send_json({
                        "type": "logs_updated",
                        "buffer_len": len(rolling_log_buffer),
                        "maxlen": rolling_log_buffer.maxlen,
                        "oldest": rolling_log_buffer[0] if rolling_log_buffer else None,
                        "newest": rolling_log_buffer[-1] if rolling_log_buffer else None,
                    })
                elif action == "trigger_screening_alert":
                    urg = msg.get("urgency", "Normal")
                    await websocket.send_json({
                        "type": "screening_alert",
                        "alert": True,
                        "urgency": urg,
                        "theme": "stark_gold",
                    })
        except WebSocketDisconnect:
            pass
        finally:
            event_bus.unsubscribe(EventType.STATE_CHANGED, state_listener)

    return app


# ---------------------------------------------------------------------------
# 8. Pytest Fixture Definitions
# ---------------------------------------------------------------------------

@pytest.fixture
def e2e_system_config() -> SystemConfig:
    config = SystemConfig()
    config.headless = True
    config.audio.sample_rate = 16000
    config.audio.chunk_size = 512
    config.vad.pre_speech_buffer_ms = 1000
    config.vad.min_speech_frames = 2
    config.vad.silence_hangover_ms = 300
    return config


@pytest.fixture
async def e2e_event_bus() -> AsyncGenerator[EventBus, None]:
    bus = EventBus()
    yield bus
    await bus.close()


@pytest.fixture
async def event_collector(e2e_event_bus: EventBus) -> AsyncGenerator[AsyncEventCollector, None]:
    collector = AsyncEventCollector(e2e_event_bus)
    yield collector
    collector.teardown()


@pytest.fixture
def audio_generator() -> AudioStreamGenerator:
    return AudioStreamGenerator(sample_rate=16000, sample_width=2)


@pytest.fixture
def e2e_virtual_pipeline(e2e_system_config: SystemConfig) -> Iterator[VirtualAudioPipeline]:
    pipeline = VirtualAudioPipeline(e2e_system_config.audio)
    pipeline.start()
    yield pipeline
    pipeline.stop()
    pipeline.clear()


@pytest.fixture
def device_registry(e2e_event_bus: EventBus) -> InMemoryDeviceRegistry:
    return InMemoryDeviceRegistry(e2e_event_bus)


@pytest.fixture
def telephony_engine(e2e_event_bus: EventBus) -> HeadlessTelephonyEngine:
    return HeadlessTelephonyEngine(e2e_event_bus)


@pytest.fixture
def model_router(e2e_event_bus: EventBus) -> DeterministicModelRouter:
    return DeterministicModelRouter(e2e_event_bus)


@pytest.fixture
def synthetic_model_provider(e2e_event_bus: EventBus) -> SyntheticModelProvider:
    return SyntheticModelProvider(e2e_event_bus)


@pytest.fixture
def hud_client(
    e2e_event_bus: EventBus,
    e2e_system_config: SystemConfig,
    device_registry: InMemoryDeviceRegistry,
    telephony_engine: HeadlessTelephonyEngine,
) -> TestClient:
    state_machine = AssistantStateMachine(e2e_event_bus)
    app = create_hud_test_app(e2e_event_bus, state_machine, device_registry, telephony_engine)
    return TestClient(app)
