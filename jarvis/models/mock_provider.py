"""Synthetic Latency Model Provider for offline CI testing and simulation.

Features:
1. Configurable latency distribution profiles per model/tier (DNS, TCP, TTFT, TAT, TPS).
2. Deterministic mode vs. Stochastic mode with Gaussian/Uniform jitter and optional seed.
3. Streaming async generator simulating chunk/token delivery.
4. Simulated failure injection: HTTP 500, HTTP 429, Connection Timeout, Socket Dropped.
5. Virtual-time fast execution mode for sub-millisecond unit test execution.
"""

from __future__ import annotations

import asyncio
import random
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, AsyncGenerator, Callable, Dict, List, Optional, Tuple, Type, Union


class MockFailureType(str, Enum):
    """Supported simulated failure modalities."""
    INTERNAL_ERROR = "internal_error"         # HTTP 500 Internal Server Error
    RATE_LIMIT = "rate_limit"                 # HTTP 429 Rate Limit Exceeded
    TIMEOUT = "timeout"                       # Request / Connection Timeout
    SOCKET_DROPPED = "socket_dropped"         # Connection Reset / Abrupt Socket Close


class MockProviderError(Exception):
    """Base exception for synthetic provider errors."""
    def __init__(self, message: str, status_code: int = 500, model_id: str = "") -> None:
        super().__init__(message)
        self.status_code = status_code
        self.model_id = model_id


class MockInternalServerError(MockProviderError):
    """Simulates HTTP 500 Internal Server Error."""
    def __init__(self, message: str = "500 Internal Server Error", model_id: str = "") -> None:
        super().__init__(message, status_code=500, model_id=model_id)


class MockRateLimitError(MockProviderError):
    """Simulates HTTP 429 Too Many Requests."""
    def __init__(self, message: str = "429 Rate Limit Exceeded", retry_after_s: float = 5.0, model_id: str = "") -> None:
        super().__init__(message, status_code=429, model_id=model_id)
        self.retry_after_s = retry_after_s


class MockConnectionTimeoutError(MockProviderError):
    """Simulates network or API timeout."""
    def __init__(self, message: str = "Connection timed out", model_id: str = "") -> None:
        super().__init__(message, status_code=504, model_id=model_id)


class MockSocketDroppedError(MockProviderError):
    """Simulates abrupt connection drop mid-stream."""
    def __init__(self, message: str = "Socket dropped unexpectedly by peer", model_id: str = "") -> None:
        super().__init__(message, status_code=502, model_id=model_id)


@dataclass
class SyntheticLatencyProfile:
    """Latency distribution profile for a synthetic model endpoint."""
    tier: int                         # 0, 1, 2, 3
    dns_ms: float                     # Baseline DNS lookup delay (ms)
    tcp_ms: float                     # Baseline TCP connect delay (ms)
    ttft_ms: float                    # Baseline Time-To-First-Token (ms)
    tat_ms: float                     # Baseline Total Turnaround Time (ms)
    tps: float                        # Tokens per second streaming rate
    total_tokens: int = 40            # Default token length for completion
    stochastic: bool = False          # Whether jitter is enabled
    jitter_std_dev: float = 0.05      # Jitter standard deviation ratio (0.05 = 5%)
    jitter_mode: str = "gaussian"     # "gaussian" or "uniform"


# Standard baseline profiles for the 4 tiers
DEFAULT_TIER_PROFILES: dict[int, SyntheticLatencyProfile] = {
    0: SyntheticLatencyProfile(tier=0, dns_ms=1.0, tcp_ms=1.0, ttft_ms=15.0, tat_ms=25.0, tps=200.0, total_tokens=15),
    1: SyntheticLatencyProfile(tier=1, dns_ms=2.0, tcp_ms=3.0, ttft_ms=85.0, tat_ms=120.0, tps=80.0, total_tokens=30),
    2: SyntheticLatencyProfile(tier=2, dns_ms=15.0, tcp_ms=25.0, ttft_ms=220.0, tat_ms=350.0, tps=60.0, total_tokens=50),
    3: SyntheticLatencyProfile(tier=3, dns_ms=20.0, tcp_ms=30.0, ttft_ms=650.0, tat_ms=950.0, tps=40.0, total_tokens=80),
}


@dataclass
class FailureSchedule:
    """Defines failure injection parameters for a model."""
    error_type: MockFailureType = MockFailureType.INTERNAL_ERROR
    fail_next_n: int = 0                       # Fail next N requests deterministically
    error_rate: float = 0.0                    # Probability of failure (0.0 to 1.0)
    fail_on_token_index: Optional[int] = None  # Fail mid-stream at token N
    custom_message: Optional[str] = None


class _MockGenerateResult(tuple):
    """3-tuple (response_text, model_id, latency_ms) supporting both sync and async await."""
    def __new__(cls, response: str, model_id: str, latency_ms: float):
        return super().__new__(cls, (response, model_id, latency_ms))

    def __await__(self):
        async def _coro():
            return self
        return _coro().__await__()


class SyntheticLatencyModelProvider:
    """Configurable synthetic latency provider for offline CI testing and simulation."""

    def __init__(
        self,
        seed: Optional[int] = None,
        virtual_clock: bool = True,
        time_scale: float = 0.0,
        event_bus: Optional[Any] = None,
    ) -> None:
        self.virtual_clock = virtual_clock
        self.time_scale = time_scale
        self.event_bus = event_bus
        self._seed = seed
        self._rng = random.Random(seed)
        self._profiles: dict[str, SyntheticLatencyProfile] = {}
        self._failure_schedules: dict[str, FailureSchedule] = {}
        self._healthy_state: dict[str, bool] = {}
        self._init_default_models()

    def _init_default_models(self) -> None:
        """Register default synthetic models across all 4 tiers."""
        self.register_model("tier0-direct-reflex", DEFAULT_TIER_PROFILES[0])
        self.register_model("tier1-ultra-fast", DEFAULT_TIER_PROFILES[1])
        self.register_model("tier2-balanced", DEFAULT_TIER_PROFILES[2])
        self.register_model("tier3-deep-reasoning", DEFAULT_TIER_PROFILES[3])

    @property
    def models(self) -> dict[str, SyntheticLatencyProfile]:
        """Dictionary of registered model profiles."""
        return self._profiles

    def register_model(
        self,
        model_id: str,
        profile: SyntheticLatencyProfile,
        healthy: bool = True,
    ) -> None:
        """Register or update a synthetic model profile."""
        self._profiles[model_id] = profile
        self._healthy_state[model_id] = healthy

    def configure_failure(
        self,
        model_id: str,
        fail_next_n: int = 1,
        error_type: MockFailureType = MockFailureType.INTERNAL_ERROR,
        error_rate: float = 0.0,
        fail_on_token_index: Optional[int] = None,
        custom_message: Optional[str] = None,
    ) -> None:
        """Configure simulated failure injection for a model."""
        self._failure_schedules[model_id] = FailureSchedule(
            error_type=error_type,
            fail_next_n=fail_next_n,
            error_rate=error_rate,
            fail_on_token_index=fail_on_token_index,
            custom_message=custom_message,
        )

    def clear_failures(self, model_id: Optional[str] = None) -> None:
        """Clear all or model-specific failure injection configs."""
        if model_id:
            self._failure_schedules.pop(model_id, None)
        else:
            self._failure_schedules.clear()

    def set_healthy(self, model_id: str, healthy: bool) -> None:
        """Set administrative health state for a model."""
        self._healthy_state[model_id] = healthy

    def is_healthy(self, model_id: str) -> bool:
        """Check if model is currently healthy."""
        return self._healthy_state.get(model_id, True)

    def _calculate_jitter(self, base_val: float, profile: SyntheticLatencyProfile) -> float:
        """Calculate jittered metric value in stochastic mode."""
        if not profile.stochastic or profile.jitter_std_dev <= 0:
            return base_val
        if profile.jitter_mode == "uniform":
            factor = self._rng.uniform(1.0 - profile.jitter_std_dev, 1.0 + profile.jitter_std_dev)
        else:
            factor = self._rng.gauss(1.0, profile.jitter_std_dev)
        return max(0.1, base_val * factor)

    def sample_metrics(self, model_id: str) -> tuple[float, float, float, float, float]:
        """Compute (dns_ms, tcp_ms, ttft_ms, tat_ms, tps) for a probe sample."""
        if model_id not in self._profiles:
            raise KeyError(f"Unknown model_id: {model_id}")
        p = self._profiles[model_id]
        dns = self._calculate_jitter(p.dns_ms, p)
        tcp = self._calculate_jitter(p.tcp_ms, p)
        ttft = self._calculate_jitter(p.ttft_ms, p)
        tat = max(ttft + 5.0, self._calculate_jitter(p.tat_ms, p))
        tps = max(1.0, self._calculate_jitter(p.tps, p))
        return (dns, tcp, ttft, tat, tps)

    def _check_and_raise_failure(self, model_id: str, is_mid_stream: bool = False, token_index: int = 0) -> None:
        """Check failure injection schedule and raise error if triggered."""
        sched = self._failure_schedules.get(model_id)
        if not sched:
            return

        should_fail = False
        if sched.fail_next_n > 0:
            if sched.fail_on_token_index is not None:
                if is_mid_stream and token_index >= sched.fail_on_token_index:
                    should_fail = True
                    sched.fail_next_n -= 1
            else:
                if not is_mid_stream:
                    should_fail = True
                    sched.fail_next_n -= 1
        elif sched.error_rate > 0.0 and self._rng.random() < sched.error_rate:
            if sched.fail_on_token_index is not None:
                if is_mid_stream and token_index >= sched.fail_on_token_index:
                    should_fail = True
            else:
                if not is_mid_stream:
                    should_fail = True

        if should_fail:
            msg = sched.custom_message
            if sched.error_type == MockFailureType.INTERNAL_ERROR:
                raise MockInternalServerError(msg or f"Simulated 500 Internal Error from {model_id}", model_id=model_id)
            elif sched.error_type == MockFailureType.RATE_LIMIT:
                raise MockRateLimitError(msg or f"Simulated 429 Rate Limit from {model_id}", model_id=model_id)
            elif sched.error_type == MockFailureType.TIMEOUT:
                raise MockConnectionTimeoutError(msg or f"Simulated Timeout from {model_id}", model_id=model_id)
            elif sched.error_type == MockFailureType.SOCKET_DROPPED:
                raise MockSocketDroppedError(msg or f"Simulated Socket Dropped from {model_id}", model_id=model_id)
            else:
                raise MockProviderError(msg or f"Simulated Error from {model_id}", model_id=model_id)

    async def _simulate_delay(self, delay_ms: float) -> None:
        """Simulate latency delay, respecting virtual clock and time scale."""
        if self.virtual_clock or self.time_scale <= 0.0:
            await asyncio.sleep(0)
        else:
            await asyncio.sleep((delay_ms / 1000.0) * self.time_scale)

    async def stream_generate(
        self,
        prompt: str,
        model_id: str,
    ) -> AsyncGenerator[str, None]:
        """Streaming async generator yielding simulated response tokens."""
        if model_id not in self._profiles:
            raise KeyError(f"Unknown model_id: {model_id}")

        self._check_and_raise_failure(model_id, is_mid_stream=False)

        profile = self._profiles[model_id]
        dns, tcp, ttft, tat, tps = self.sample_metrics(model_id)

        await self._simulate_delay(ttft)

        words = [
            "Certainly,", "sir.", "I", "have", "processed", "your",
            "request", f"'{prompt}'.", "All", "systems", "operating",
            "at", "peak", "efficiency."
        ]
        token_interval_ms = 1000.0 / tps if tps > 0 else 10.0

        for idx, word in enumerate(words):
            self._check_and_raise_failure(model_id, is_mid_stream=True, token_index=idx)
            await self._simulate_delay(token_interval_ms)
            yield word + " "

    async def generate_text(
        self,
        prompt: str,
        model_id: str,
    ) -> tuple[str, str, float]:
        """Generate full completion text and return (response, model_id, latency_ms)."""
        tokens: list[str] = []
        start_time = time.monotonic()
        async for chunk in self.stream_generate(prompt, model_id):
            tokens.append(chunk)
        elapsed_ms = (time.monotonic() - start_time) * 1000.0
        profile = self._profiles[model_id]
        reported_latency = profile.tat_ms if self.virtual_clock else elapsed_ms
        return ("".join(tokens).strip(), model_id, reported_latency)

    def generate(
        self,
        prompt: str,
        model_id: Optional[str] = None,
    ) -> _MockGenerateResult:
        """Synchronous/asynchronous generation returning (response, model_id, latency_ms)."""
        selected = model_id or "tier0-direct-reflex"
        if selected not in self._profiles:
            # Fallback to first available model
            selected = next(iter(self._profiles.keys()), "tier0-direct-reflex")

        self._check_and_raise_failure(selected, is_mid_stream=False)
        profile = self._profiles[selected]
        latency = profile.ttft_ms

        p_lower = prompt.lower()
        if "diagnostic" in p_lower or "status" in p_lower:
            response = "Diagnostic check complete, sir. All systems nominal."
        else:
            response = f"Acknowledged, sir. Processed via {selected}."

        return _MockGenerateResult(response, selected, latency)


# Canonical alias for compatibility
SyntheticModelProvider = SyntheticLatencyModelProvider
