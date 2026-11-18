"""Desktop Launcher for Jarvis Holographic HUD Window."""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
import shutil
import socket
import subprocess
import threading
import time
from typing import Any, Optional
import webbrowser

from fastapi import FastAPI
import uvicorn

logger = logging.getLogger(__name__)


def find_chrome_path() -> Optional[str]:
    """Find Google Chrome executable on Windows."""
    candidates = [
        r"C:\Program Files\Google\Chrome\Application\chrome.exe",
        r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
        os.path.expandvars(r"%LOCALAPPDATA%\Google\Chrome\Application\chrome.exe"),
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    return shutil.which("chrome") or shutil.which("google-chrome")


def find_edge_path() -> Optional[str]:
    """Find Microsoft Edge executable on Windows."""
    candidates = [
        r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
        r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
        os.path.expandvars(r"%LOCALAPPDATA%\Microsoft\Edge\Application\msedge.exe"),
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    return shutil.which("msedge")


def select_browser(preferred: str = "auto") -> tuple[str, Optional[str]]:
    """Select browser mode ('app' or 'default') and executable path."""
    pref = (preferred or "auto").lower()
    if pref == "chrome":
        chrome = find_chrome_path()
        return ("app", chrome) if chrome else ("default", None)
    elif pref == "edge":
        edge = find_edge_path()
        return ("app", edge) if edge else ("default", None)
    elif pref == "auto":
        chrome = find_chrome_path()
        if chrome:
            return ("app", chrome)
        edge = find_edge_path()
        if edge:
            return ("app", edge)
        return ("default", None)
    else:
        return ("default", None)


def launch_browser_window(
    url: str,
    browser: str = "auto",
    window_size: tuple[int, int] = (1280, 800),
) -> Optional[subprocess.Popen[Any]]:
    """Launch browser in borderless desktop app mode or default browser."""
    mode, exe = select_browser(browser)
    if mode == "app" and exe:
        cmd = [
            exe,
            f"--app={url}",
            f"--window-size={window_size[0]},{window_size[1]}",
            "--window-position=100,100",
            "--disable-extensions",
            "--no-first-run",
            "--no-default-browser-check",
        ]
        try:
            logger.info("Launching desktop HUD window via: %s", exe)
            return subprocess.Popen(cmd)
        except Exception as exc:
            logger.warning("Failed to launch browser app mode (%s), falling back to default browser", exc)

    logger.info("Opening HUD in default system browser: %s", url)
    webbrowser.open(url)
    return None


class ServerThread(threading.Thread):
    """Run uvicorn FastAPI server in a dedicated background daemon thread."""

    def __init__(self, app: FastAPI, host: str = "127.0.0.1", port: int = 8000) -> None:
        super().__init__(daemon=True, name="JarvisServerThread")
        self.app = app
        self.host = host
        self.port = port
        self.server: Optional[uvicorn.Server] = None
        self._started = threading.Event()

    def run(self) -> None:
        """Run uvicorn server."""
        config = uvicorn.Config(
            app=self.app,
            host=self.host,
            port=self.port,
            log_level="warning",
            access_log=False,
        )
        self.server = uvicorn.Server(config)
        self._started.set()
        self.server.run()

    def stop(self) -> None:
        """Signal uvicorn server to stop."""
        if self.server:
            self.server.should_exit = True


def wait_for_server(host: str, port: int, timeout: float = 5.0) -> bool:
    """Poll TCP socket until server accepts connections or timeout expires."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with socket.create_connection((host, port), timeout=0.2):
                return True
        except (socket.error, OSError):
            time.sleep(0.05)
    return False


def parse_launcher_args(args: Optional[list[str]] = None) -> argparse.Namespace:
    """Parse CLI arguments for desktop launcher."""
    parser = argparse.ArgumentParser(description="Jarvis Desktop HUD Launcher")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind server (default: 8000)")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host interface (default: 127.0.0.1)")
    parser.add_argument("--headless", action="store_true", help="Run without opening browser GUI")
    parser.add_argument(
        "--browser",
        type=str,
        default="auto",
        choices=["auto", "chrome", "edge", "default"],
        help="Target browser for app mode (default: auto)",
    )
    parser.add_argument("--no-server", action="store_true", help="Launch GUI only without embedded server")
    return parser.parse_args(args)


def launch_desktop(
    app: Optional[FastAPI] = None,
    host: str = "127.0.0.1",
    port: int = 8000,
    browser: str = "auto",
    headless: bool = False,
    wait_timeout: float = 5.0,
) -> tuple[Optional[ServerThread], Optional[subprocess.Popen[Any]]]:
    """Master entrypoint: start server thread and launch desktop browser window."""
    server_thread: Optional[ServerThread] = None

    if app is not None:
        server_thread = ServerThread(app, host=host, port=port)
        server_thread.start()
        ready = wait_for_server(host, port, timeout=wait_timeout)
        if not ready:
            logger.error("HUD server failed to start within %.1fs on %s:%d", wait_timeout, host, port)

    browser_proc: Optional[subprocess.Popen[Any]] = None
    if not headless:
        url = f"http://{host}:{port}/"
        browser_proc = launch_browser_window(url, browser=browser)

    return (server_thread, browser_proc)
