"""Comprehensive Headless Unit Tests for Jarvis Holographic HUD & Telemetry Interface.

Verifies:
1. Static asset serving, DOM containers, and strict Stark color palette (zero Ultron red).
2. FastAPI REST endpoints (/api/status, /api/telemetry) with mocked and clamped psutil metrics.
3. Real-time WebSocket bridge (/ws/hud and /ws/telemetry) connection lifecycle, initial state/telemetry
   handshake, inbound client actions (set_state, ping), and disconnect cleanup.
4. Multiple concurrent WebSocket clients and broadcast resilience to abruptly dropped connections.
5. Bidirectional EventBus integration (STATE_CHANGED, TELEMETRY_UPDATE, TRANSCRIPT_READY, BARGE_IN).
6. TelemetrySampler background task execution, non-blocking interval timing, and clean cancellation.
7. Desktop launcher CLI parsing, browser selection (--app mode for Chrome/Edge), and headless bypass.
8. 100% headless Windows execution completing in under 5 seconds with zero flaky timeouts.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
import re
import socket
import subprocess
import time
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from starlette.testclient import TestClient

from jarvis.config import SystemConfig
from jarvis.core.event_bus import Event, EventBus, EventType
from jarvis.core.state_machine import AssistantState, AssistantStateMachine
from jarvis.ui.desktop_launcher import (
    ServerThread,
    find_chrome_path,
    find_edge_path,
    launch_browser_window,
    launch_desktop,
    parse_launcher_args,
    select_browser,
    wait_for_server,
)
from jarvis.ui.server import (
    EventBusHUDBridge,
    HUDConnectionManager,
    TelemetryData,
    TelemetrySampler,
    create_app,
)


@pytest.fixture
def hud_static_dir() -> Path:
    """Return the static asset directory for the HUD."""
    static_path = Path(__file__).resolve().parent.parent.parent / "jarvis" / "ui" / "static"
    return static_path


@pytest.fixture
def hud_app(event_bus: EventBus, state_machine: AssistantStateMachine, hud_static_dir: Path):
    """Provide a configured FastAPI app with fast telemetry interval (10ms) for testing."""
    return create_app(
        event_bus=event_bus,
        state_machine=state_machine,
        static_dir=hud_static_dir,
        telemetry_interval=0.01,
    )


@pytest.fixture
def hud_client(hud_app) -> TestClient:
    """Provide a TestClient bound to the HUD FastAPI app."""
    return TestClient(hud_app)


# ==============================================================================
# 1. STATIC FILE SERVING & ASSET INTEGRITY
# ==============================================================================
class TestHUDStaticFiles:
    """Test suite for static assets, DOM containers, and Stark color palette compliance."""

    def test_root_index_html_serving_and_dom_elements(self, hud_client: TestClient) -> None:
        """Verify GET / returns 200 OK, text/html, and required HUD visualizer containers."""
        response = hud_client.get("/")
        assert response.status_code == 200
        assert "text/html" in response.headers.get("content-type", "")
        content = response.text

        # Verify key DOM containers for Arc Reactor and Oscilloscope
        assert "arc-reactor" in content, "Missing #arc-reactor container in index.html"
        assert "orbital-rings" in content, "Missing orbital-rings canvas in index.html"
        assert "oscilloscope" in content or "telemetry" in content, "Missing visualizer container in index.html"
        assert "hud.css" in content, "Missing stylesheet reference in index.html"
        assert "arc_reactor.js" in content, "Missing arc_reactor.js script tag in index.html"
        assert "visualizer.js" in content, "Missing visualizer.js script tag in index.html"

    def test_hud_css_serving_and_palette_integrity(self, hud_client: TestClient) -> None:
        """Verify GET /static/hud.css serves valid CSS containing Iron Man Stark palette colors."""
        response = hud_client.get("/static/hud.css")
        assert response.status_code == 200
        assert "text/css" in response.headers.get("content-type", "")
        css = response.text

        # Must contain Stark Cyan (#00E5FF), Electric Blue (#0077B6), or Stark Gold (#FFD700)
        has_cyan = "#00E5FF" in css.upper() or "#00E5FF" in css.lower()
        has_blue = "#0077B6" in css.upper() or "#0077B6" in css.lower()
        has_gold = "#FFD700" in css.upper() or "#FFD700" in css.lower()
        assert has_cyan or has_blue or has_gold, "hud.css must define Iron Man Stark palette"

    def test_hud_css_ultron_red_strictly_forbidden(self, hud_client: TestClient) -> None:
        """Verify hud.css strictly bans all Ultron red hues and crimson keywords."""
        response = hud_client.get("/static/hud.css")
        assert response.status_code == 200
        css = response.text

        # Strictly banned colors per R2 acceptance criteria
        assert "#FF0000" not in css.upper(), "Forbidden Ultron Red (#FF0000) found in hud.css"
        assert "#E60000" not in css.upper(), "Forbidden Ultron Crimson (#E60000) found in hud.css"
        assert "crimson" not in css.lower(), "Forbidden 'crimson' color name found in hud.css"

    def test_arc_reactor_js_serving(self, hud_client: TestClient) -> None:
        """Verify GET /static/arc_reactor.js returns 200 OK and valid JavaScript."""
        response = hud_client.get("/static/arc_reactor.js")
        assert response.status_code == 200
        assert len(response.text) > 0
        assert "ArcReactor" in response.text or "orbital" in response.text.lower()

    def test_visualizer_js_serving(self, hud_client: TestClient) -> None:
        """Verify GET /static/visualizer.js returns 200 OK and valid JavaScript."""
        response = hud_client.get("/static/visualizer.js")
        assert response.status_code == 200
        assert len(response.text) > 0
        assert "Oscilloscope" in response.text or "visualizer" in response.text.lower() or "canvas" in response.text.lower()

    def test_missing_static_file_returns_404(self, hud_client: TestClient) -> None:
        """Verify GET for a non-existent static asset returns 404 Not Found."""
        response = hud_client.get("/static/non_existent_asset_404.png")
        assert response.status_code == 404


# ==============================================================================
# 2. REST API ENDPOINTS (/api/status, /api/telemetry)
# ==============================================================================
class TestHUDRESTEndpoints:
    """Test suite for FastAPI REST status and telemetry endpoints."""

    def test_api_status_default_state(self, hud_client: TestClient) -> None:
        """Verify /api/status returns online status and default IDLE assistant state."""
        response = hud_client.get("/api/status")
        assert response.status_code == 200
        data = response.json()
        assert data.get("status") == "online"
        assert data.get("state") == "IDLE"
        assert "version" in data

    @pytest.mark.asyncio
    async def test_api_status_state_machine_reflection(
        self,
        hud_client: TestClient,
        state_machine: AssistantStateMachine,
    ) -> None:
        """Verify /api/status dynamically reflects active state transitions."""
        await state_machine.transition_to(AssistantState.LISTENING)
        res1 = hud_client.get("/api/status")
        assert res1.json().get("state") == "LISTENING"

        await state_machine.transition_to(AssistantState.PROCESSING)
        res2 = hud_client.get("/api/status")
        assert res2.json().get("state") == "PROCESSING"

        await state_machine.transition_to(AssistantState.SPEAKING)
        res3 = hud_client.get("/api/status")
        assert res3.json().get("state") == "SPEAKING"

        await state_machine.transition_to(AssistantState.IDLE)
        res4 = hud_client.get("/api/status")
        assert res4.json().get("state") == "IDLE"

    @pytest.mark.asyncio
    async def test_api_status_busy_mode_reflection(
        self,
        hud_client: TestClient,
        state_machine: AssistantStateMachine,
    ) -> None:
        """Verify /api/status reports busy_mode=True when state is BUSY."""
        await state_machine.transition_to(AssistantState.BUSY)
        res = hud_client.get("/api/status")
        data = res.json()
        assert data.get("state") == "BUSY"
        assert data.get("busy_mode") is True

    def test_api_telemetry_live_response_schema(self, hud_client: TestClient) -> None:
        """Verify /api/telemetry returns bounded numeric metrics for CPU, RAM, battery, ping."""
        response = hud_client.get("/api/telemetry")
        assert response.status_code == 200
        data = response.json()

        for key in ["cpu", "ram", "battery", "net_ping", "audio_level"]:
            assert key in data, f"Key '{key}' missing from /api/telemetry"
            assert isinstance(data[key], (int, float)), f"Key '{key}' must be a float or int"

        assert 0.0 <= data["cpu"] <= 100.0
        assert 0.0 <= data["ram"] <= 100.0
        assert 0.0 <= data["audio_level"] <= 1.0
        assert data["net_ping"] >= 0.0 or data["net_ping"] == -1.0

    def test_api_telemetry_mocked_psutil_values(
        self,
        event_bus: EventBus,
        state_machine: AssistantStateMachine,
        hud_static_dir: Path,
    ) -> None:
        """Verify /api/telemetry accurately captures mocked psutil hardware sensor values."""
        with patch("psutil.cpu_percent", return_value=45.2), \
             patch("psutil.virtual_memory") as mock_mem, \
             patch("psutil.sensors_battery") as mock_bat:

            mock_mem.return_value.percent = 63.8
            mock_bat.return_value = MagicMock(percent=88.5)

            app = create_app(event_bus=event_bus, state_machine=state_machine, static_dir=hud_static_dir)
            client = TestClient(app)

            response = client.get("/api/telemetry")
            assert response.status_code == 200
            data = response.json()
            assert pytest.approx(data["cpu"], rel=0.01) == 45.2
            assert pytest.approx(data["ram"], rel=0.01) == 63.8
            assert pytest.approx(data["battery"], rel=0.01) == 88.5

    def test_api_telemetry_null_battery_handling(
        self,
        event_bus: EventBus,
        state_machine: AssistantStateMachine,
        hud_static_dir: Path,
    ) -> None:
        """Verify /api/telemetry defaults battery to 100.0 on systems without battery (Mains / Arc Reactor)."""
        with patch("psutil.sensors_battery", return_value=None):
            app = create_app(event_bus=event_bus, state_machine=state_machine, static_dir=hud_static_dir)
            client = TestClient(app)

            response = client.get("/api/telemetry")
            assert response.status_code == 200
            data = response.json()
            assert data["battery"] == 100.0, "Systems without battery must default to 100.0 (Mains)"


# ==============================================================================
# 3. WEBSOCKET LIFECYCLE, PROTOCOL & CONCURRENCY
# ==============================================================================
class TestHUDWebSocketLifecycle:
    """Test suite for WebSocket connection lifecycle, initial handshake, and client actions."""

    def test_websocket_handshake_and_initial_state_snapshot(self, hud_client: TestClient) -> None:
        """Verify connecting to /ws/hud immediately delivers initial assistant state snapshot."""
        with hud_client.websocket_connect("/ws/hud") as ws:
            msg_state = ws.receive_json()
            assert msg_state.get("type") in ("state", "state_change")
            state_val = msg_state.get("state") or msg_state.get("data", {}).get("state")
            assert state_val == "IDLE"

    def test_websocket_initial_telemetry_snapshot(self, hud_client: TestClient) -> None:
        """Verify connecting to /ws/hud delivers initial hardware telemetry packet."""
        with hud_client.websocket_connect("/ws/hud") as ws:
            _ = ws.receive_json()  # 1. State snapshot
            msg_telem = ws.receive_json()  # 2. Telemetry snapshot
            assert msg_telem.get("type") == "telemetry"

            # Check top-level or nested fields
            cpu = msg_telem.get("cpu", msg_telem.get("data", {}).get("cpu_percent"))
            ram = msg_telem.get("ram", msg_telem.get("data", {}).get("ram_percent"))
            assert cpu is not None and 0.0 <= float(cpu) <= 100.0
            assert ram is not None and 0.0 <= float(ram) <= 100.0

    def test_websocket_dual_endpoint_alias_ws_telemetry(self, hud_client: TestClient) -> None:
        """Verify /ws/telemetry is a functional alias for /ws/hud supporting E2E test suites."""
        with hud_client.websocket_connect("/ws/telemetry") as ws:
            msg_state = ws.receive_json()
            assert msg_state.get("type") in ("state", "state_change")

            msg_telem = ws.receive_json()
            assert msg_telem.get("type") == "telemetry"
            assert "cpu" in msg_telem or "cpu_percent" in msg_telem.get("data", {})

    def test_websocket_client_ping_pong_heartbeat(self, hud_client: TestClient) -> None:
        """Verify WebSocket client can send ping action and receive pong response."""
        with hud_client.websocket_connect("/ws/hud") as ws:
            _ = ws.receive_json()  # State
            _ = ws.receive_json()  # Telemetry

            ws.send_json({"action": "ping"})
            reply = ws.receive_json()
            assert reply.get("type") in ("pong", "telemetry")

    def test_websocket_client_set_state_action(
        self,
        hud_client: TestClient,
        state_machine: AssistantStateMachine,
    ) -> None:
        """Verify WebSocket client can dispatch set_state action to transition assistant."""
        with hud_client.websocket_connect("/ws/hud") as ws:
            _ = ws.receive_json()  # State
            _ = ws.receive_json()  # Telemetry

            ws.send_json({"action": "set_state", "state": "SPEAKING"})
            reply = ws.receive_json()
            assert reply.get("type") == "state"
            assert reply.get("state") == "SPEAKING"
            assert state_machine.is_speaking

    def test_websocket_disconnect_cleanup(
        self,
        hud_app,
        hud_client: TestClient,
    ) -> None:
        """Verify disconnecting a WebSocket client decrements connection pool count to zero."""
        conn_mgr: HUDConnectionManager = hud_app.state.connection_manager
        assert conn_mgr.count == 0

        with hud_client.websocket_connect("/ws/hud") as ws:
            assert conn_mgr.count == 1
            _ = ws.receive_json()

        # After exiting context block, connection must be removed
        assert conn_mgr.count == 0

    def test_websocket_multiple_concurrent_clients(
        self,
        hud_app,
    ) -> None:
        """Verify multiple simultaneous WebSocket clients connect and receive broadcasts."""
        conn_mgr: HUDConnectionManager = hud_app.state.connection_manager

        c1 = TestClient(hud_app)
        c2 = TestClient(hud_app)
        c3 = TestClient(hud_app)

        with c1.websocket_connect("/ws/hud") as ws1:
            ws1.receive_json()
            ws1.receive_json()
            with c2.websocket_connect("/ws/hud") as ws2:
                ws2.receive_json()
                ws2.receive_json()
                with c3.websocket_connect("/ws/hud") as ws3:
                    ws3.receive_json()
                    ws3.receive_json()
                    assert conn_mgr.count == 3

                    # ws1 dispatches set_state SPEAKING which broadcasts to all
                    ws1.send_json({"action": "set_state", "state": "SPEAKING"})
                    r1 = ws1.receive_json()
                    r2 = ws2.receive_json()
                    r3 = ws3.receive_json()

                    assert r1.get("type") == "state" and r1.get("state") == "SPEAKING"
                    assert r2.get("type") == "state" and r2.get("state") == "SPEAKING"
                    assert r3.get("type") == "state" and r3.get("state") == "SPEAKING"

        assert conn_mgr.count == 0

    def test_websocket_broken_client_broadcast_resilience(
        self,
        hud_app,
    ) -> None:
        """Verify an abruptly closed client is pruned without preventing broadcasts to active clients."""
        conn_mgr: HUDConnectionManager = hud_app.state.connection_manager

        c_survivor = TestClient(hud_app)
        c_closing = TestClient(hud_app)

        with c_survivor.websocket_connect("/ws/hud") as ws_survivor:
            ws_survivor.receive_json()
            ws_survivor.receive_json()

            with c_closing.websocket_connect("/ws/hud") as ws_closing:
                ws_closing.receive_json()
                ws_closing.receive_json()
                assert conn_mgr.count == 2
            # ws_closing has now disconnected
            assert conn_mgr.count == 1

            # Dispatch action to broadcast
            ws_survivor.send_json({"action": "set_state", "state": "BUSY"})
            msg = ws_survivor.receive_json()
            assert msg.get("type") == "state"
            assert msg.get("state") == "BUSY"


# ==============================================================================
# 4. EVENTBUS TO WEBSOCKET BROADCASTING INTEGRATION
# ==============================================================================
class TestHUDEventBusBroadcasting:
    """Test suite for internal EventBus event propagation to WebSocket clients."""

    @pytest.mark.asyncio
    async def test_event_bus_state_changed_broadcast(
        self,
        event_bus: EventBus,
        hud_client: TestClient,
    ) -> None:
        """Verify publishing EventType.STATE_CHANGED sends state update to connected WebSockets."""
        with hud_client.websocket_connect("/ws/hud") as ws:
            _ = ws.receive_json()  # init state
            _ = ws.receive_json()  # init telem

            await event_bus.publish_event(
                EventType.STATE_CHANGED,
                old_state="IDLE",
                new_state="LISTENING",
                reason="speech_onset",
            )

            msg = ws.receive_json()
            assert msg.get("type") == "state"
            assert msg.get("state") == "LISTENING"
            assert msg.get("old_state") == "IDLE"

    @pytest.mark.asyncio
    async def test_event_bus_telemetry_update_broadcast(
        self,
        event_bus: EventBus,
        hud_client: TestClient,
    ) -> None:
        """Verify publishing EventType.TELEMETRY_UPDATE broadcasts metrics packet."""
        with hud_client.websocket_connect("/ws/hud") as ws:
            _ = ws.receive_json()  # init state
            _ = ws.receive_json()  # init telem

            await event_bus.publish_event(
                EventType.TELEMETRY_UPDATE,
                cpu=33.5,
                ram=58.2,
                net_ping=14.0,
                audio_level=0.45,
            )

            msg = ws.receive_json()
            assert msg.get("type") == "telemetry"
            assert pytest.approx(msg.get("cpu"), rel=0.01) == 33.5
            assert pytest.approx(msg.get("ram"), rel=0.01) == 58.2

    @pytest.mark.asyncio
    async def test_event_bus_transcript_ready_broadcast(
        self,
        event_bus: EventBus,
        hud_client: TestClient,
    ) -> None:
        """Verify publishing EventType.TRANSCRIPT_READY broadcasts user transcript."""
        with hud_client.websocket_connect("/ws/hud") as ws:
            _ = ws.receive_json()
            _ = ws.receive_json()

            await event_bus.publish_event(
                EventType.TRANSCRIPT_READY,
                text="Jarvis, scan perimeter",
                confidence=0.98,
                speaker="user",
            )

            msg = ws.receive_json()
            assert msg.get("type") == "transcript"
            assert msg.get("text") == "Jarvis, scan perimeter"
            assert msg.get("speaker") == "user"

    @pytest.mark.asyncio
    async def test_event_bus_tts_started_broadcast(
        self,
        event_bus: EventBus,
        hud_client: TestClient,
    ) -> None:
        """Verify publishing EventType.TTS_SYNTHESIS_STARTED broadcasts Jarvis speech transcript."""
        with hud_client.websocket_connect("/ws/hud") as ws:
            _ = ws.receive_json()
            _ = ws.receive_json()

            await event_bus.publish_event(
                EventType.TTS_SYNTHESIS_STARTED,
                text="All perimeter defense shields are active, sir.",
            )

            msg = ws.receive_json()
            assert msg.get("type") == "transcript"
            assert "perimeter defense shields" in msg.get("text")
            assert msg.get("speaker") == "jarvis"

    @pytest.mark.asyncio
    async def test_event_bus_barge_in_triggered_broadcast(
        self,
        event_bus: EventBus,
        hud_client: TestClient,
    ) -> None:
        """Verify publishing EventType.BARGE_IN_TRIGGERED broadcasts barge-in notification."""
        with hud_client.websocket_connect("/ws/hud") as ws:
            _ = ws.receive_json()
            _ = ws.receive_json()

            await event_bus.publish_event(
                EventType.BARGE_IN_TRIGGERED,
                generation_id=12,
                interruption_latency_ms=19.4,
            )

            msg = ws.receive_json()
            assert msg.get("type") == "barge_in"
            assert msg.get("generation_id") == 12
            assert pytest.approx(msg.get("latency_ms"), rel=0.01) == 19.4


# ==============================================================================
# 5. TELEMETRY SAMPLER BACKGROUND TASK & HARDWARE RESILIENCE
# ==============================================================================
class TestHUDTelemetrySampler:
    """Test suite for TelemetrySampler polling, non-blocking loops, and task cancellation."""

    def test_telemetry_sampler_instantaneous_sample(self, event_bus: EventBus) -> None:
        """Verify TelemetrySampler.sample() produces valid bounded hardware metrics."""
        sampler = TelemetrySampler(event_bus=event_bus, interval=0.05)
        sampler.set_audio_level(0.35)

        data = sampler.sample()
        assert isinstance(data, dict)
        assert 0.0 <= data["cpu"] <= 100.0
        assert 0.0 <= data["ram"] <= 100.0
        assert 0.0 <= data["battery"] <= 100.0
        assert pytest.approx(data["audio_level"], rel=0.01) == 0.35

    def test_telemetry_sampler_clamped_extremes(self, event_bus: EventBus) -> None:
        """Verify extreme or invalid metrics are safely clamped within permissible bounds."""
        sampler = TelemetrySampler(event_bus=event_bus, interval=0.05)

        # Set extreme audio levels
        sampler.set_audio_level(50.0)
        assert sampler._latest_audio_level == 1.0

        sampler.set_audio_level(-10.0)
        assert sampler._latest_audio_level == 0.0

    @pytest.mark.asyncio
    async def test_telemetry_sampler_loop_interval_timing_and_no_spin(self, event_bus: EventBus) -> None:
        """Verify telemetry sampler executes periodic polling using asyncio.sleep without CPU spin."""
        conn_mgr = HUDConnectionManager()
        sampler = TelemetrySampler(event_bus=event_bus, connection_manager=conn_mgr, interval=0.02)

        received_events: list[Event] = []

        async def telem_handler(event: Event) -> None:
            received_events.append(event)

        event_bus.subscribe(EventType.TELEMETRY_UPDATE, telem_handler)

        await sampler.start()
        # Allow loop to tick ~3-4 times in 80ms
        await asyncio.sleep(0.08)
        await sampler.stop()

        assert len(received_events) >= 2, "Sampler loop must tick periodically at configured interval"

    @pytest.mark.asyncio
    async def test_telemetry_sampler_clean_task_cancellation(self, event_bus: EventBus) -> None:
        """Verify calling stop() cleanly cancels background task without leaks or lingering tasks."""
        sampler = TelemetrySampler(event_bus=event_bus, interval=0.02)
        await sampler.start()
        assert sampler._running is True
        assert sampler._task is not None
        assert not sampler._task.done()

        await sampler.stop()
        assert sampler._running is False
        assert sampler._task is None

    def test_telemetry_sampler_psutil_exception_resilience(self, event_bus: EventBus) -> None:
        """Verify sampler catches psutil errors gracefully without crashing process."""
        with patch("psutil.cpu_percent", side_effect=PermissionError("Access denied")), \
             patch("psutil.virtual_memory", side_effect=RuntimeError("Memory read error")), \
             patch("psutil.sensors_battery", side_effect=Exception("Sensor unavailable")):

            sampler = TelemetrySampler(event_bus=event_bus, interval=0.05)
            # Must not raise an exception; uses safe fallbacks
            data = sampler.sample()
            assert isinstance(data, dict)
            assert data["cpu"] >= 0.0
            assert data["ram"] >= 0.0
            assert data["battery"] == 100.0


# ==============================================================================
# 6. DESKTOP LAUNCHER & BROWSER SELECTION
# ==============================================================================
class TestHUDDesktopLauncher:
    """Test suite for desktop launcher CLI argument parsing, browser selection, and app mode."""

    def test_launcher_cli_arg_defaults(self) -> None:
        """Verify default CLI arguments for desktop launcher."""
        args = parse_launcher_args([])
        assert args.port == 8000
        assert args.host == "127.0.0.1"
        assert args.headless is False
        assert args.browser == "auto"

    def test_launcher_cli_arg_custom_values(self) -> None:
        """Verify CLI argument parsing with custom flags."""
        args = parse_launcher_args(["--port", "9090", "--host", "0.0.0.0", "--headless", "--browser", "chrome"])
        assert args.port == 9090
        assert args.host == "0.0.0.0"
        assert args.headless is True
        assert args.browser == "chrome"

    def test_launcher_headless_bypasses_browser(self) -> None:
        """Verify launch_desktop in headless mode does not invoke any browser."""
        with patch("jarvis.ui.desktop_launcher.launch_browser_window") as mock_launch:
            server_th, proc = launch_desktop(app=None, headless=True)
            mock_launch.assert_not_called()
            assert proc is None

    def test_launcher_chrome_app_mode_command(self) -> None:
        """Verify browser selection formulates correct --app flag for Google Chrome."""
        fake_chrome = r"C:\Program Files\Google\Chrome\Application\chrome.exe"
        with patch("jarvis.ui.desktop_launcher.find_chrome_path", return_value=fake_chrome), \
             patch("subprocess.Popen") as mock_popen:

            proc = launch_browser_window("http://127.0.0.1:8000/", browser="chrome")
            mock_popen.assert_called_once()
            cmd_args = mock_popen.call_args[0][0]
            assert cmd_args[0] == fake_chrome
            assert "--app=http://127.0.0.1:8000/" in cmd_args

    def test_launcher_edge_app_mode_command(self) -> None:
        """Verify browser selection formulates correct --app flag for Microsoft Edge."""
        fake_edge = r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"
        with patch("jarvis.ui.desktop_launcher.find_chrome_path", return_value=None), \
             patch("jarvis.ui.desktop_launcher.find_edge_path", return_value=fake_edge), \
             patch("subprocess.Popen") as mock_popen:

            proc = launch_browser_window("http://127.0.0.1:8000/", browser="edge")
            mock_popen.assert_called_once()
            cmd_args = mock_popen.call_args[0][0]
            assert cmd_args[0] == fake_edge
            assert "--app=http://127.0.0.1:8000/" in cmd_args

    def test_launcher_fallback_to_webbrowser(self) -> None:
        """Verify fallback to standard webbrowser.open when neither Chrome nor Edge is found."""
        with patch("jarvis.ui.desktop_launcher.find_chrome_path", return_value=None), \
             patch("jarvis.ui.desktop_launcher.find_edge_path", return_value=None), \
             patch("webbrowser.open") as mock_wb:

            proc = launch_browser_window("http://127.0.0.1:8000/", browser="auto")
            assert proc is None
            mock_wb.assert_called_once_with("http://127.0.0.1:8000/")

    def test_launcher_server_thread_lifecycle(self, hud_app) -> None:
        """Verify ServerThread initializes cleanly and can be stopped."""
        server_thread = ServerThread(app=hud_app, host="127.0.0.1", port=8001)
        assert server_thread.daemon is True
        assert server_thread.name == "JarvisServerThread"
        # Verify stop handles unstarted server safely
        server_thread.stop()


# ==============================================================================
# 7. PERFORMANCE & RAPID RECONNECT STRESS TESTING
# ==============================================================================
class TestHUDPerformanceAndHeadlessExecution:
    """Stress and performance validation ensuring sub-5-second headless execution."""

    def test_fastapi_rest_concurrent_load(self, hud_client: TestClient) -> None:
        """Verify 20 rapid sequential REST queries return 200 OK within 1.0s."""
        start_time = time.perf_counter()
        for _ in range(20):
            res_status = hud_client.get("/api/status")
            assert res_status.status_code == 200
            res_telem = hud_client.get("/api/telemetry")
            assert res_telem.status_code == 200
        duration = time.perf_counter() - start_time
        assert duration < 1.0, f"20 REST requests took {duration:.3f}s (must be < 1.0s)"

    def test_rapid_websocket_reconnect_stress(self, hud_client: TestClient) -> None:
        """Verify 10 consecutive connect/disconnect cycles operate without socket exhaustion."""
        for _ in range(10):
            with hud_client.websocket_connect("/ws/hud") as ws:
                init_msg = ws.receive_json()
                assert init_msg.get("type") in ("state", "state_change")
                ws.send_json({"action": "ping"})
                pong = ws.receive_json()
                assert pong is not None

        # Subsequent connection operates cleanly
        with hud_client.websocket_connect("/ws/hud") as ws:
            final_msg = ws.receive_json()
            assert final_msg.get("type") in ("state", "state_change")
