"""FastAPI Holographic HUD Server and Real-Time WebSocket Telemetry Bridge."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
import logging
from pathlib import Path
import time
from typing import Any, Optional, Union

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
import psutil

from jarvis.core.event_bus import Event, EventBus, EventType
from jarvis.core.state_machine import AssistantState, AssistantStateMachine

logger = logging.getLogger(__name__)


@dataclass
class TelemetryData:
    """Hardware telemetry snapshot data container."""

    cpu: float
    ram: float
    battery: float
    net_ping: float
    audio_level: float
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        """Convert telemetry metrics into cross-compatible dual payload format."""
        return {
            "type": "telemetry",
            "timestamp": self.timestamp,
            "cpu": self.cpu,
            "ram": self.ram,
            "battery": self.battery,
            "net_ping": self.net_ping,
            "audio_level": self.audio_level,
            "cpu_percent": self.cpu,
            "ram_percent": self.ram,
            "battery_percent": self.battery,
            "data": {
                "cpu_percent": self.cpu,
                "ram_percent": self.ram,
                "battery_percent": self.battery,
                "net_ping": self.net_ping,
                "audio_level": self.audio_level,
            },
        }


class HUDConnectionManager:
    """Thread-safe and async-safe WebSocket connection pool for HUD clients."""

    def __init__(self) -> None:
        self._connections: set[WebSocket] = set()

    @property
    def count(self) -> int:
        """Return number of active WebSocket connections."""
        return len(self._connections)

    async def connect(self, websocket: WebSocket) -> None:
        """Accept and register an active WebSocket client."""
        await websocket.accept()
        self._connections.add(websocket)
        logger.debug("HUD WebSocket client connected (active: %d)", len(self._connections))

    async def disconnect(self, websocket: WebSocket) -> None:
        """Safely deregister a WebSocket client."""
        self._connections.discard(websocket)
        logger.debug("HUD WebSocket client disconnected (active: %d)", len(self._connections))

    async def send_json(self, websocket: WebSocket, data: dict[str, Any]) -> None:
        """Send JSON payload to a specific client with error handling."""
        try:
            await websocket.send_json(data)
        except Exception as exc:
            logger.debug("Failed to send message to WebSocket client: %s", exc)
            await self.disconnect(websocket)

    async def broadcast_json(self, data: dict[str, Any]) -> None:
        """Broadcast JSON payload to all active clients, pruning dead connections."""
        targets = list(self._connections)
        if not targets:
            return

        dead: list[WebSocket] = []
        for ws in targets:
            try:
                await ws.send_json(data)
            except Exception:
                dead.append(ws)

        if dead:
            for ws in dead:
                self._connections.discard(ws)

    async def close_all(self) -> None:
        """Cleanly close all active WebSocket connections on server shutdown."""
        targets = list(self._connections)
        self._connections.clear()
        for ws in targets:
            try:
                await ws.close(code=1000)
            except Exception:
                pass


class TelemetrySampler:
    """Asynchronous background hardware telemetry collector and broadcaster."""

    def __init__(
        self,
        event_bus: Optional[EventBus] = None,
        connection_manager: Optional[HUDConnectionManager] = None,
        interval: float = 1.0,
    ) -> None:
        self.event_bus = event_bus
        self.connection_manager = connection_manager
        self.interval = interval
        self._running = False
        self._task: Optional[asyncio.Task[None]] = None
        self._latest_audio_level: float = 0.04
        self._latest_sample: Optional[dict[str, Any]] = None

        # Prime psutil CPU measurement to avoid 0.0 on subsequent calls
        try:
            psutil.cpu_percent(interval=None)
        except Exception:
            pass

    def set_audio_level(self, level: float) -> None:
        """Update current audio energy RMS level."""
        self._latest_audio_level = max(0.0, min(1.0, float(level)))

    def sample(self) -> dict[str, Any]:
        """Collect instantaneous host telemetry with safe fallbacks and bounded clamping."""
        try:
            raw_cpu = float(psutil.cpu_percent(interval=None))
            cpu = max(0.0, min(100.0, raw_cpu))
        except Exception:
            cpu = 15.0

        try:
            raw_ram = float(psutil.virtual_memory().percent)
            ram = max(0.0, min(100.0, raw_ram))
        except Exception:
            ram = 40.0

        try:
            bat = psutil.sensors_battery()
            if bat is not None and bat.percent is not None:
                battery = max(0.0, min(100.0, float(bat.percent)))
            else:
                battery = 100.0  # Mains power / Arc Reactor 100%
        except Exception:
            battery = 100.0

        net_ping = 12.0
        audio = max(0.0, min(1.0, self._latest_audio_level))

        telem = TelemetryData(
            cpu=cpu,
            ram=ram,
            battery=battery,
            net_ping=net_ping,
            audio_level=audio,
            timestamp=time.time(),
        )
        data = telem.to_dict()
        self._latest_sample = data
        return data

    def get_latest_or_sample(self) -> dict[str, Any]:
        """Return cached sample if available, otherwise sample immediately."""
        if self._latest_sample is not None:
            return self._latest_sample
        return self.sample()

    async def start(self) -> None:
        """Start asynchronous telemetry sampling loop."""
        if self._running:
            return
        self._running = True
        self._task = asyncio.create_task(self._sampling_loop(), name="hud_telemetry_sampler")

    async def stop(self) -> None:
        """Gracefully stop telemetry sampling loop."""
        if not self._running:
            return
        self._running = False
        if self._task and not self._task.done():
            self._task.cancel()
            try:
                task_loop = self._task.get_loop()
                current_loop = asyncio.get_running_loop()
                if task_loop is current_loop:
                    await self._task
            except (asyncio.CancelledError, RuntimeError):
                pass
            self._task = None

    async def _sampling_loop(self) -> None:
        """Periodic background sampling loop."""
        while self._running:
            try:
                await asyncio.sleep(self.interval)
                sample_data = self.sample()

                # Broadcast to connected WebSocket clients
                if self.connection_manager and self.connection_manager.count > 0:
                    await self.connection_manager.broadcast_json(sample_data)

                # Publish to internal EventBus
                if self.event_bus:
                    await self.event_bus.publish_event(
                        EventType.TELEMETRY_UPDATE,
                        cpu=sample_data["cpu"],
                        ram=sample_data["ram"],
                        battery=sample_data["battery"],
                        net_ping=sample_data["net_ping"],
                        audio_level=sample_data["audio_level"],
                    )
            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.warning("Error in telemetry sampling loop: %s", exc)


class EventBusHUDBridge:
    """Subscribes to internal EventBus events and relays them to HUD WebSocket clients."""

    def __init__(
        self,
        event_bus: EventBus,
        connection_manager: HUDConnectionManager,
        telemetry_sampler: Optional[TelemetrySampler] = None,
    ) -> None:
        self.event_bus = event_bus
        self.connection_manager = connection_manager
        self.telemetry_sampler = telemetry_sampler

    def start(self) -> None:
        """Subscribe to all EventBus events."""
        self.event_bus.subscribe(None, self.on_bus_event)

    def stop(self) -> None:
        """Unsubscribe from EventBus events."""
        self.event_bus.unsubscribe(None, self.on_bus_event)

    async def on_bus_event(self, event: Event) -> None:
        """Translate EventBus event to WebSocket JSON message."""
        if self.connection_manager.count == 0:
            return

        ev_type = event.event_type

        # 1. State changes
        if ev_type == EventType.STATE_CHANGED:
            new_st = event.payload.get("new_state", "IDLE")
            old_st = event.payload.get("old_state", "IDLE")
            await self.connection_manager.broadcast_json({
                "type": "state",
                "state": new_st,
                "old_state": old_st,
                "reason": event.payload.get("reason", ""),
                "timestamp": event.timestamp,
                "data": {
                    "state": new_st,
                    "old_state": old_st,
                },
            })

        # 2. Transcripts (User and Assistant)
        elif ev_type == EventType.TRANSCRIPT_READY:
            await self.connection_manager.broadcast_json({
                "type": "transcript",
                "text": event.payload.get("text", ""),
                "speaker": event.payload.get("speaker", "user"),
                "confidence": event.payload.get("confidence", 1.0),
                "timestamp": event.timestamp,
                "data": event.payload,
            })

        elif ev_type == EventType.TTS_SYNTHESIS_STARTED:
            await self.connection_manager.broadcast_json({
                "type": "transcript",
                "text": event.payload.get("text", ""),
                "speaker": "jarvis",
                "timestamp": event.timestamp,
                "data": event.payload,
            })

        # 3. Speech onset and audio energy
        elif ev_type == EventType.SPEECH_STARTED:
            energy = float(event.payload.get("energy_rms", 0.0))
            if self.telemetry_sampler and energy > 0:
                self.telemetry_sampler.set_audio_level(min(1.0, energy / 100.0))
            await self.connection_manager.broadcast_json({
                "type": "speech_started",
                "energy_rms": energy,
                "timestamp": event.timestamp,
                "data": event.payload,
            })

        # 4. Telemetry update from EventBus
        elif ev_type == EventType.TELEMETRY_UPDATE:
            cpu = float(event.payload.get("cpu", 0.0))
            ram = float(event.payload.get("ram", 0.0))
            battery = float(event.payload.get("battery", 100.0))
            net_ping = float(event.payload.get("net_ping", 12.0))
            audio_level = float(event.payload.get("audio_level", 0.0))
            await self.connection_manager.broadcast_json({
                "type": "telemetry",
                "timestamp": event.timestamp,
                "cpu": cpu,
                "ram": ram,
                "battery": battery,
                "net_ping": net_ping,
                "audio_level": audio_level,
                "cpu_percent": cpu,
                "ram_percent": ram,
                "battery_percent": battery,
                "data": event.payload,
            })

        # 5. Smart device state changes
        elif ev_type == EventType.DEVICE_STATE_CHANGED:
            await self.connection_manager.broadcast_json({
                "type": "device_update",
                "device_id": event.payload.get("device_id", ""),
                "state": event.payload.get("state", ""),
                "traits": event.payload.get("traits", {}),
                "timestamp": event.timestamp,
                "data": event.payload,
            })

        # 6. Telephony call updates
        elif ev_type in (EventType.CALL_INCOMING, EventType.CALL_SCREENING_UPDATE, EventType.CALL_DEBRIEF_CREATED):
            await self.connection_manager.broadcast_json({
                "type": "call_update",
                "event": ev_type.value,
                "call_id": event.payload.get("call_id", ""),
                "caller": event.payload.get("caller", ""),
                "caller_name": event.payload.get("caller_name", event.payload.get("caller", "")),
                "number": event.payload.get("number", ""),
                "state": event.payload.get("state", ""),
                "urgency": event.payload.get("urgency", "Normal"),
                "transcript": event.payload.get("transcript", []),
                "last_utterance": event.payload.get("last_utterance", ""),
                "speaker": event.payload.get("speaker", "caller"),
                "timestamp": event.timestamp,
                "data": event.payload,
            })

        # 7. Barge-in preemption
        elif ev_type == EventType.BARGE_IN_TRIGGERED:
            await self.connection_manager.broadcast_json({
                "type": "barge_in",
                "generation_id": event.payload.get("generation_id", 0),
                "latency_ms": event.payload.get("interruption_latency_ms", 0.0),
                "timestamp": event.timestamp,
                "data": event.payload,
            })


def create_app(
    event_bus: Optional[EventBus] = None,
    state_machine: Optional[AssistantStateMachine] = None,
    static_dir: Optional[Union[str, Path]] = None,
    telemetry_interval: float = 1.0,
    device_registry: Any = None,
    telephony_engine: Any = None,
) -> FastAPI:
    """FastAPI application factory for Jarvis Holographic HUD."""
    if static_dir is None:
        resolved_static_dir = Path(__file__).resolve().parent / "static"
    else:
        resolved_static_dir = Path(static_dir)

    connection_manager = HUDConnectionManager()
    telemetry_sampler = TelemetrySampler(
        event_bus=event_bus,
        connection_manager=connection_manager,
        interval=telemetry_interval,
    )
    bridge: Optional[EventBusHUDBridge] = None
    if event_bus is not None:
        bridge = EventBusHUDBridge(event_bus, connection_manager, telemetry_sampler)
        bridge.start()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # Startup
        await telemetry_sampler.start()
        logger.info("Jarvis HUD server initialized with static dir: %s", resolved_static_dir)
        yield
        # Shutdown
        await telemetry_sampler.stop()
        if bridge:
            bridge.stop()
        await connection_manager.close_all()
        logger.info("Jarvis HUD server shut down cleanly")

    app = FastAPI(
        title="Jarvis Holographic HUD Interface",
        version="0.1.0",
        lifespan=lifespan,
    )

    app.state.connection_manager = connection_manager
    app.state.telemetry_sampler = telemetry_sampler
    app.state.event_bus = event_bus
    app.state.state_machine = state_machine
    app.state.device_registry = device_registry
    app.state.telephony_engine = telephony_engine
    app.state.static_dir = resolved_static_dir

    # Mount static assets
    if resolved_static_dir.exists():
        app.mount("/static", StaticFiles(directory=str(resolved_static_dir), html=True), name="static")

    # 1. Root HTML Endpoint
    @app.get("/", response_class=HTMLResponse)
    async def get_index():
        index_file = resolved_static_dir / "index.html"
        if index_file.exists():
            return FileResponse(index_file)
        return HTMLResponse(
            "<!DOCTYPE html><html><head><title>JARVIS HUD</title></head>"
            "<body style='background:#050B14;color:#00E5FF;'>"
            "<div id='arc-reactor'><canvas id='orbital-rings'></canvas></div>"
            "<div id='telemetry-panel'><canvas id='oscilloscope'></canvas></div>"
            "</body></html>"
        )

    # 2. REST Endpoints
    @app.get("/api/status")
    async def get_status():
        current_st = state_machine.current_state.value if state_machine else "IDLE"
        busy = (current_st == "BUSY")
        if telephony_engine and hasattr(telephony_engine, "busy_mode"):
            busy = telephony_engine.busy_mode
        dev_count = 0
        if device_registry and hasattr(device_registry, "list_all"):
            try:
                dev_count = len(device_registry.list_all())
            except Exception:
                pass
        return {
            "status": "online",
            "state": current_st,
            "version": "0.1.0",
            "busy_mode": busy,
            "active_devices": dev_count,
            "active_connections": connection_manager.count,
        }

    @app.get("/api/telemetry")
    async def get_telemetry():
        return telemetry_sampler.get_latest_or_sample()

    # 3. Dual WebSocket Endpoint Handler (/ws/hud and /ws/telemetry)
    async def hud_ws_handler(websocket: WebSocket) -> None:
        await connection_manager.connect(websocket)
        try:
            # 1. Initial State Snapshot Handshake
            current_st = state_machine.current_state.value if state_machine else "IDLE"
            await connection_manager.send_json(websocket, {
                "type": "state",
                "state": current_st,
                "timestamp": time.time(),
                "data": {"state": current_st},
            })

            # 2. Initial Telemetry Snapshot Handshake
            telem = telemetry_sampler.get_latest_or_sample()
            await connection_manager.send_json(websocket, {
                "type": "telemetry",
                "timestamp": time.time(),
                "cpu": telem["cpu"],
                "ram": telem["ram"],
                "battery": telem.get("battery", 100.0),
                "net_ping": telem.get("net_ping", 12.0),
                "audio_level": telem.get("audio_level", 0.0),
                "cpu_percent": telem["cpu"],
                "ram_percent": telem["ram"],
                "battery_percent": telem.get("battery", 100.0),
                "data": telem.get("data", telem),
            })

            # 3. Message Loop
            while True:
                msg = await websocket.receive_json()
                action = msg.get("action")

                if action == "set_state":
                    target_st_name = msg.get("state")
                    if target_st_name and state_machine:
                        try:
                            target_state = AssistantState(target_st_name)
                            await state_machine.transition_to(target_state, reason="ws_command")
                        except Exception as exc:
                            logger.warning("Failed state transition from WS (%s): %s", target_st_name, exc)
                    await connection_manager.send_json(websocket, {
                        "type": "state",
                        "state": target_st_name,
                        "timestamp": time.time(),
                        "data": {"state": target_st_name},
                    })

                elif action == "toggle_busy":
                    busy_val = bool(msg.get("busy", True))
                    if telephony_engine and hasattr(telephony_engine, "set_busy_mode"):
                        telephony_engine.set_busy_mode(busy_val)
                    if state_machine:
                        target = AssistantState.BUSY if busy_val else AssistantState.IDLE
                        try:
                            await state_machine.transition_to(target, reason="ws_busy_toggle")
                        except Exception:
                            pass
                    await connection_manager.send_json(websocket, {
                        "type": "busy_updated",
                        "busy": busy_val,
                        "timestamp": time.time(),
                        "data": {"busy": busy_val},
                    })

                elif action == "ping":
                    await connection_manager.send_json(websocket, {
                        "type": "pong",
                        "timestamp": time.time(),
                    })

                elif action == "device_command" and device_registry:
                    dev_id = msg.get("device_id")
                    cmd = msg.get("command")
                    params = msg.get("params", {})
                    res = {}
                    if hasattr(device_registry, "execute_command"):
                        res = device_registry.execute_command(dev_id, cmd, **params)
                    await connection_manager.send_json(websocket, {
                        "type": "device_command_result",
                        "device_id": dev_id,
                        "result": res,
                    })

                elif action == "takeover_call" and telephony_engine:
                    call_id = msg.get("call_id", "")
                    if hasattr(telephony_engine, "take_over"):
                        telephony_engine.take_over(call_id)
                    await connection_manager.send_json(websocket, {
                        "type": "call_takeover_acknowledged",
                        "call_id": call_id,
                        "timestamp": time.time(),
                    })

        except WebSocketDisconnect:
            pass
        except Exception as exc:
            logger.warning("WebSocket client session error: %s", exc)
        finally:
            await connection_manager.disconnect(websocket)

    app.add_api_websocket_route("/ws/hud", hud_ws_handler)
    app.add_api_websocket_route("/ws/telemetry", hud_ws_handler)

    return app
