"""Automatic 90-Day (3-Month) Updater for Jarvis AI Assistant.

Per Feature 26 (PROJECT.md) and User Request:
- Checks for updates automatically every 90 days (90 * 86400 seconds = 7,776,000s).
- Compares version against GitHub releases repository tags using semantic versioning.
- Defers/suppresses updates when user sessions are active (SPEAKING, LISTENING, PROCESSING, BUSY).
- Handles GitHub API rate limiting (HTTP 403, 429, x-ratelimit-reset) and network errors.
- Guards against clock skew (negative elapsed time).
- Emits events to EventBus upon update detection and checks.
- Prepares installer handoff for self-update.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
import re
import time
from typing import Any, Dict, Optional, Tuple

from jarvis.core.event_bus import Event, EventBus, EventType
from jarvis.core.state_machine import AssistantState

logger = logging.getLogger(__name__)


def _safe_create_task(coro: Any) -> Optional[asyncio.Task]:
    """Safely schedule a coroutine task on the active event loop if one exists."""
    try:
        loop = asyncio.get_running_loop()
        return loop.create_task(coro)
    except RuntimeError:
        try:
            coro.close()
        except Exception:
            pass
        return None


class Automatic90DayUpdater:
    """90-Day automatic updater following Feature 26 specification."""

    def __init__(
        self,
        event_bus: Optional[EventBus] = None,
        check_interval_days: int = 90,
        current_version: str = "1.0.0",
        repo: str = "shreyastouih-ctrl/project-jarvis",
    ) -> None:
        self.event_bus = event_bus
        self.check_interval_days = check_interval_days
        self.current_version = current_version
        self.repo = repo
        self.interval_seconds = check_interval_days * 86400  # 90 * 86400 = 7,776,000s
        self.last_check_timestamp: float = time.time()
        self.rate_limit_reset: Optional[float] = None
        self.last_check_result: Optional[Dict[str, Any]] = None

    @staticmethod
    def parse_semver(tag: Any) -> Optional[Tuple[int, int, int]]:
        """Parse semver tag e.g. 'v1.2.3' or '1.2.3' into (major, minor, patch)."""
        if not tag or not isinstance(tag, str):
            return None
        match = re.match(r"^v?(\d+)\.(\d+)\.(\d+)$", tag.strip())
        if not match:
            return None
        return (int(match.group(1)), int(match.group(2)), int(match.group(3)))

    def is_update_due(self, current_time: Optional[float] = None) -> bool:
        """Check if 90-day update cadence has elapsed with clock skew guard."""
        now = current_time if current_time is not None else time.time()
        delta = now - self.last_check_timestamp
        if delta < 0:
            # Clock skew detected (system clock shifted backwards)
            logger.warning("Clock skew detected: delta %.2f s < 0. Resetting check baseline.", delta)
            self.last_check_timestamp = now
            delta = 0.0
        return delta >= self.interval_seconds

    def check_updates(
        self,
        api_response: Any,
        system_state: AssistantState = AssistantState.IDLE,
        current_time: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Perform update check respecting system active session and rate limits."""
        if not isinstance(api_response, dict):
            res = {
                "status": "error",
                "error": f"Invalid API response payload type: {type(api_response).__name__}",
            }
            self.last_check_result = res
            return res

        now = current_time if current_time is not None else time.time()

        # 1. Rate limiting check
        if self.rate_limit_reset and now < self.rate_limit_reset:
            res = {
                "status": "rate_limited",
                "backoff_remaining_s": self.rate_limit_reset - now,
            }
            self.last_check_result = res
            return res

        # 2. Session active suppression: defer if speaking, listening, processing, or busy
        if system_state in (
            AssistantState.SPEAKING,
            AssistantState.LISTENING,
            AssistantState.PROCESSING,
            AssistantState.BUSY,
        ):
            res = {
                "status": "suppressed",
                "reason": f"Active user session in progress ({system_state.value}). Update deferred.",
            }
            self.last_check_result = res
            return res

        # 3. Network or API error handling
        if "error" in api_response:
            res = {
                "status": "error",
                "error": api_response["error"],
            }
            self.last_check_result = res
            return res

        # 4. Handle HTTP 403/429 rate limit payload
        if api_response.get("status_code") in (403, 429):
            reset_ts = api_response.get("x_ratelimit_reset", now + 3600.0)
            self.rate_limit_reset = reset_ts
            res = {
                "status": "rate_limited",
                "reset_timestamp": reset_ts,
            }
            self.last_check_result = res
            return res

        # 5. Parse release tag
        raw_tag = api_response.get("tag_name", "")
        parsed = self.parse_semver(raw_tag)
        if parsed is None:
            res = {
                "status": "ignored",
                "reason": f"Malformed or non-semver release tag '{raw_tag}'",
            }
            self.last_check_result = res
            return res

        current_parsed = self.parse_semver(self.current_version) or (1, 0, 0)
        self.last_check_timestamp = now

        if parsed > current_parsed:
            raw_assets = api_response.get("assets", [])
            download_url = ""
            if isinstance(raw_assets, list) and raw_assets:
                first = raw_assets[0]
                if isinstance(first, dict):
                    download_url = first.get("browser_download_url", "")
            res = {
                "status": "update_available",
                "version": raw_tag,
                "download_url": download_url,
            }
            self.last_check_result = res

            if self.event_bus:
                _safe_create_task(
                    self.event_bus.publish(
                        Event(
                            event_type=EventType.STATE_CHANGED,
                            payload={
                                "action": "update_available",
                                "version": raw_tag,
                                "download_url": download_url,
                                "current_version": self.current_version,
                            },
                        )
                    )
                )
            return res

        res = {"status": "up_to_date", "version": self.current_version}
        self.last_check_result = res
        return res

    async def check_github_releases(
        self,
        system_state: AssistantState = AssistantState.IDLE,
    ) -> Dict[str, Any]:
        """Query GitHub Releases API asynchronously for the latest release."""
        try:
            import urllib.request
            import json

            url = f"https://api.github.com/repos/{self.repo}/releases/latest"
            req = urllib.request.Request(
                url,
                headers={"User-Agent": "Jarvis-AI-Assistant-Updater"},
            )
            loop = asyncio.get_running_loop()
            
            def _fetch():
                with urllib.request.urlopen(req, timeout=5.0) as response:
                    return json.loads(response.read().decode("utf-8"))

            data = await loop.run_in_executor(None, _fetch)
            return self.check_updates(data, system_state=system_state)
        except Exception as exc:
            return self.check_updates({"error": str(exc)}, system_state=system_state)

    def trigger_installer_handoff(
        self,
        installer_path: Optional[Path] = None,
        expected_sha256: Optional[str] = None,
    ) -> bool:
        """Prepare or hand off update execution to installer script with integrity verification."""
        import hashlib
        path = installer_path or (Path(__file__).resolve().parent.parent.parent / "install.bat")
        if not path.exists():
            logger.error("Installer handoff target not found: %s", path)
            return False

        try:
            if path.stat().st_size == 0:
                logger.error("Installer handoff target is empty (0 bytes): %s", path)
                return False
        except OSError as exc:
            logger.error("Cannot access installer target %s: %s", path, exc)
            return False

        if expected_sha256:
            try:
                content = path.read_bytes()
                actual_sha256 = hashlib.sha256(content).hexdigest()
                if actual_sha256.lower() != expected_sha256.lower():
                    logger.error("Installer hash mismatch: expected %s, got %s", expected_sha256, actual_sha256)
                    return False
            except Exception as exc:
                logger.error("Failed to verify checksum for %s: %s", path, exc)
                return False

        logger.info("Installer handoff staged successfully for %s", path)
        return True
