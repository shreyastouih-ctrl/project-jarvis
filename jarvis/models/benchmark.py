"""Active Latency Benchmark Harness for Jarvis AI Assistant.

Measures:
- Network transport timing: DNS lookup time and TCP handshake connection time.
- Streaming inference timing: Time-To-First-Token (TTFT), Total Turnaround Time (TAT),
  and Tokens-Per-Second (TPS).
- Statistical smoothing: Exponential Moving Average (EMA), IQR outlier filtering, and jitter.
- Probing lifecycle: Periodic background runner, on-demand warm-up, concurrency control,
  and rate-limiting.
"""

from __future__ import annotations

import asyncio
import logging
import math
import socket
import time
import urllib.parse
from collections import deque
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Callable, Dict, List, Optional, Tuple, Union

import numpy as np

from jarvis.core.event_bus import EventBus, EventType
from jarvis.models.taxonomy import (
    DEFAULT_MODEL_CATALOG,
    ModelLatencyMetrics,
    ModelTier,
    classify_model_tier,
)

logger = logging.getLogger(__name__)


# ==============================================================================
# 1. DATA STRUCTURES & CONFIGURATION
# ==============================================================================

@dataclass
class LatencySample:
    """Represents a single raw latency probe observation."""
    timestamp: float
    dns_ms: float
    tcp_ms: float
    ttft_ms: float
    tat_ms: float
    tps: float
    token_count: int
    success: bool
    error_message: Optional[str] = None


@dataclass
class BenchmarkConfig:
    """Configuration options for LatencyBenchmarkHarness."""
    alpha: float = 0.25                  # EMA smoothing factor (0.0 < alpha <= 1.0)
    window_size: int = 20                # Rolling window size for IQR outlier filtering
    probe_interval_s: float = 30.0       # Periodic background probe interval in seconds
    min_probe_interval_s: float = 5.0    # Per-model rate limiting cooldown
    probe_timeout_s: float = 5.0         # Probe timeout in seconds
    max_concurrency: int = 3             # Max parallel probes across models
    outlier_filtering: bool = True       # Enable IQR outlier filtering
    iqr_multiplier: float = 1.5          # IQR multiplier for upper/lower bounds
    probe_prompt: str = "ping"           # Minimal payload for probe queries
    enable_network_probing: bool = True  # Enable real socket DNS/TCP probes
    failure_threshold: int = 3           # Consecutive failures to mark unhealthy


# ==============================================================================
# 2. STATISTICAL SMOOTHING & OUTLIER FILTERING ENGINE
# ==============================================================================

class RollingLatencyStats:
    """Maintains rolling samples, IQR outlier filtering, EMA smoothing, and jitter."""

    def __init__(self, alpha: float = 0.25, window_size: int = 20, iqr_multiplier: float = 1.5):
        self.alpha = max(0.01, min(1.0, alpha))
        self.window_size = max(4, window_size)
        self.iqr_multiplier = iqr_multiplier

        self.samples: deque[LatencySample] = deque(maxlen=self.window_size)
        self.consecutive_failures: int = 0
        self.total_probes: int = 0
        self.total_failures: int = 0

        # Smoothed metrics (EMA)
        self.ema_dns_ms: float = 0.0
        self.ema_tcp_ms: float = 0.0
        self.ema_ttft_ms: float = 0.0
        self.ema_tat_ms: float = 0.0
        self.ema_tps: float = 0.0

        self.jitter_ms: float = 0.0
        self.last_updated: float = 0.0
        self._initialized: bool = False

    def update(self, sample: LatencySample) -> None:
        """Update rolling stats with a new sample."""
        self.total_probes += 1
        self.last_updated = sample.timestamp

        if not sample.success:
            self.consecutive_failures += 1
            self.total_failures += 1
            return

        self.consecutive_failures = 0

        # Outlier detection using IQR (requires >= 4 samples for statistical validity)
        is_outlier = False
        valid_ttfts = [s.ttft_ms for s in self.samples if s.success]

        if len(valid_ttfts) >= 4:
            q1, q3 = np.percentile(valid_ttfts, [25, 75])
            iqr = q3 - q1
            # Zero-variance safeguard: enforce a minimum floor
            effective_iqr = max(float(iqr), 0.1 * float(q3), 5.0)
            lower = max(0.0, float(q1) - self.iqr_multiplier * effective_iqr)
            upper = float(q3) + self.iqr_multiplier * effective_iqr

            if sample.ttft_ms < lower or sample.ttft_ms > upper:
                is_outlier = True

        self.samples.append(sample)

        # Update EMA with outlier dampening
        if not self._initialized:
            self.ema_dns_ms = sample.dns_ms
            self.ema_tcp_ms = sample.tcp_ms
            self.ema_ttft_ms = sample.ttft_ms
            self.ema_tat_ms = sample.tat_ms
            self.ema_tps = sample.tps
            self._initialized = True
        else:
            # Discount outlier influence by 80% to avoid distorting EMA
            effective_alpha = self.alpha * 0.2 if is_outlier else self.alpha
            self.ema_dns_ms = effective_alpha * sample.dns_ms + (1.0 - effective_alpha) * self.ema_dns_ms
            self.ema_tcp_ms = effective_alpha * sample.tcp_ms + (1.0 - effective_alpha) * self.ema_tcp_ms
            self.ema_ttft_ms = effective_alpha * sample.ttft_ms + (1.0 - effective_alpha) * self.ema_ttft_ms
            self.ema_tat_ms = effective_alpha * sample.tat_ms + (1.0 - effective_alpha) * self.ema_tat_ms
            self.ema_tps = effective_alpha * sample.tps + (1.0 - effective_alpha) * self.ema_tps

        # Compute sample standard deviation as jitter
        refreshed_ttfts = [s.ttft_ms for s in self.samples if s.success]
        if len(refreshed_ttfts) > 1:
            self.jitter_ms = float(np.std(refreshed_ttfts))
        else:
            self.jitter_ms = 0.0


# ==============================================================================
# 3. TRANSPORT & STREAMING PROBING UTILITIES
# ==============================================================================

def parse_endpoint_url(endpoint_url: Optional[str]) -> Tuple[Optional[str], int, bool]:
    """Extract host, port, and SSL flag from endpoint URL."""
    if not endpoint_url or endpoint_url.startswith("local://") or endpoint_url.startswith("memory://"):
        return None, 0, False
    try:
        parsed = urllib.parse.urlparse(endpoint_url)
        host = parsed.hostname
        is_ssl = parsed.scheme in ("https", "wss")
        port = parsed.port or (443 if is_ssl else 80)
        return host, port, is_ssl
    except Exception:
        return None, 0, False


async def measure_dns_latency(host: str, port: int = 443, timeout_s: float = 2.0) -> float:
    """Measure DNS resolution latency using non-blocking asyncio getaddrinfo."""
    if not host or host in ("localhost", "127.0.0.1", "::1"):
        return 0.0
    t0 = time.perf_counter()
    try:
        loop = asyncio.get_running_loop()
        await asyncio.wait_for(
            loop.getaddrinfo(host, port, family=socket.AF_UNSPEC, type=socket.SOCK_STREAM),
            timeout=timeout_s,
        )
        return (time.perf_counter() - t0) * 1000.0
    except Exception as exc:
        logger.debug("DNS probe failed for %s: %s", host, exc)
        return -1.0


async def measure_tcp_latency(host: str, port: int = 443, timeout_s: float = 2.0) -> float:
    """Measure TCP handshake connection latency."""
    if not host or host in ("localhost", "127.0.0.1"):
        return 0.1
    t0 = time.perf_counter()
    try:
        reader, writer = await asyncio.wait_for(
            asyncio.open_connection(host, port),
            timeout=timeout_s,
        )
        tcp_ms = (time.perf_counter() - t0) * 1000.0
        writer.close()
        await writer.wait_closed()
        return tcp_ms
    except Exception as exc:
        logger.debug("TCP probe failed for %s:%d: %s", host, port, exc)
        return -1.0


async def measure_streaming_probe(
    stream_fn: Callable[[], AsyncIterator[str]],
    timeout_s: float = 5.0,
) -> Tuple[float, float, float, int]:
    """Execute streaming inference probe and measure TTFT, TAT, and TPS.

    Returns:
        (ttft_ms, tat_ms, tps, token_count)
    """
    t_dispatch = time.perf_counter()
    t_first_token: Optional[float] = None
    token_count = 0

    async def _stream_consumer() -> None:
        nonlocal t_first_token, token_count
        async for chunk in stream_fn():
            now = time.perf_counter()
            if t_first_token is None:
                t_first_token = now
            if chunk:
                tokens_in_chunk = max(1, len(chunk.split()))
                token_count += tokens_in_chunk

    await asyncio.wait_for(_stream_consumer(), timeout=timeout_s)
    t_finish = time.perf_counter()

    if t_first_token is None:
        t_first_token = t_finish

    ttft_ms = (t_first_token - t_dispatch) * 1000.0
    tat_ms = (t_finish - t_dispatch) * 1000.0

    gen_time_s = t_finish - t_first_token
    if gen_time_s > 0.001 and token_count > 0:
        tps = token_count / gen_time_s
    elif tat_ms > 0.001 and token_count > 0:
        tps = token_count / (tat_ms / 1000.0)
    else:
        tps = 0.0

    return ttft_ms, tat_ms, tps, token_count


# ==============================================================================
# 4. ACTIVE LATENCY BENCHMARK HARNESS
# ==============================================================================

@dataclass
class _ModelRegistration:
    model_id: str
    tier: int
    endpoint_url: Optional[str] = None
    probe_fn: Optional[Callable[[], AsyncIterator[str]]] = None
    last_probed: float = 0.0


class LatencyBenchmarkHarness:
    """Active Latency Benchmark Harness managing model telemetry and health."""

    def __init__(
        self,
        provider: Optional[Any] = None,
        config: Optional[BenchmarkConfig] = None,
        event_bus: Optional[EventBus] = None,
        alpha: Optional[float] = None,
    ) -> None:
        self.provider = provider
        self.event_bus = event_bus
        self.config = config or BenchmarkConfig()
        if alpha is not None:
            self.config.alpha = alpha

        self._models: dict[str, _ModelRegistration] = {}
        self._stats: dict[str, RollingLatencyStats] = {}
        self._semaphore = asyncio.Semaphore(self.config.max_concurrency)
        self._running: bool = False
        self._probe_task: Optional[asyncio.Task] = None

        # Auto-seed standard models or provider models
        self._initialize_models()

    def _initialize_models(self) -> None:
        """Register initial models from provider or default catalog."""
        if self.provider and hasattr(self.provider, "models"):
            for m_id, prof in self.provider.models.items():
                tier = getattr(prof, "tier", 0)
                self.register_model(m_id, int(tier))
        else:
            for m_id, prof in DEFAULT_MODEL_CATALOG.items():
                self.register_model(m_id, int(prof.tier), prof.endpoint_url)

    @property
    def is_running(self) -> bool:
        return self._running

    def register_model(
        self,
        model_id: str,
        tier: int,
        endpoint_url: Optional[str] = None,
        probe_fn: Optional[Callable[[], AsyncIterator[str]]] = None,
        baseline_metrics: Optional[ModelLatencyMetrics] = None,
    ) -> None:
        """Register a model for active benchmarking."""
        self._models[model_id] = _ModelRegistration(
            model_id=model_id,
            tier=tier,
            endpoint_url=endpoint_url,
            probe_fn=probe_fn,
        )
        if model_id not in self._stats:
            stats = RollingLatencyStats(
                alpha=self.config.alpha,
                window_size=self.config.window_size,
                iqr_multiplier=self.config.iqr_multiplier,
            )
            if baseline_metrics is not None:
                sample = LatencySample(
                    timestamp=time.time(),
                    dns_ms=baseline_metrics.dns_ms,
                    tcp_ms=baseline_metrics.tcp_ms,
                    ttft_ms=baseline_metrics.ttft_ms,
                    tat_ms=baseline_metrics.tat_ms,
                    tps=baseline_metrics.tps,
                    token_count=10,
                    success=baseline_metrics.healthy,
                )
                stats.update(sample)
            self._stats[model_id] = stats

    def unregister_model(self, model_id: str) -> None:
        """Unregister a model from benchmarking."""
        self._models.pop(model_id, None)
        self._stats.pop(model_id, None)

    def record_sample(
        self,
        model_id: str,
        ttft_ms: float,
        tat_ms: Optional[float] = None,
        dns_ms: float = 0.0,
        tcp_ms: float = 0.0,
        tps: float = 50.0,
        token_count: int = 10,
        success: bool = True,
        error_message: Optional[str] = None,
    ) -> None:
        """Directly record an observation sample for a model, registering it if absent."""
        if model_id not in self._models:
            tier = int(classify_model_tier(ttft_ms))
            self.register_model(model_id, tier)

        stats = self._stats[model_id]
        computed_tat = tat_ms if tat_ms is not None else (ttft_ms + 20.0)
        sample = LatencySample(
            timestamp=time.time(),
            dns_ms=dns_ms,
            tcp_ms=tcp_ms,
            ttft_ms=ttft_ms,
            tat_ms=computed_tat,
            tps=tps,
            token_count=token_count,
            success=success,
            error_message=error_message,
        )
        stats.update(sample)

    def get_smoothed_ttft(self, model_id: str) -> float:
        """Retrieve smoothed TTFT estimate (EMA) for target model."""
        stats = self._stats.get(model_id)
        if not stats:
            return 0.0
        return stats.ema_ttft_ms

    async def probe_model(self, model_id: str, force: bool = False) -> ModelLatencyMetrics:
        """Probe a specific model, applying rate-limiting, concurrency control, and timeout."""
        if model_id not in self._models:
            # Check default catalog or provider
            if self.provider and hasattr(self.provider, "models") and model_id in self.provider.models:
                tier = int(getattr(self.provider.models[model_id], "tier", 1))
            else:
                prof = DEFAULT_MODEL_CATALOG.get(model_id)
                tier = int(prof.tier) if prof else 1
            self.register_model(model_id, tier)

        reg = self._models[model_id]
        stats = self._stats[model_id]
        now = time.time()

        # Rate-limiting check
        if not force and stats._initialized and (now - reg.last_probed) < self.config.min_probe_interval_s:
            metrics = self.get_metrics(model_id)
            if metrics:
                return metrics

        async with self._semaphore:
            reg.last_probed = now
            dns_ms, tcp_ms = 0.0, 0.0

            # 1. Transport Probing
            if self.config.enable_network_probing and reg.endpoint_url:
                host, port, _ = parse_endpoint_url(reg.endpoint_url)
                if host:
                    dns_ms = await measure_dns_latency(host, port, timeout_s=self.config.probe_timeout_s)
                    tcp_ms = await measure_tcp_latency(host, port, timeout_s=self.config.probe_timeout_s)

            # 2. Streaming / Inference Probing
            success = True
            error_msg: Optional[str] = None
            ttft_ms, tat_ms, tps, token_count = 0.0, 0.0, 0.0, 0

            if self.provider and hasattr(self.provider, "sample_metrics"):
                try:
                    s_dns, s_tcp, s_ttft, s_tat, s_tps = self.provider.sample_metrics(model_id)
                    if dns_ms <= 0:
                        dns_ms = s_dns
                    if tcp_ms <= 0:
                        tcp_ms = s_tcp
                    ttft_ms = s_ttft
                    tat_ms = s_tat
                    tps = s_tps
                    token_count = max(1, int(tps * (tat_ms - ttft_ms) / 1000.0))
                except Exception as exc:
                    success = False
                    error_msg = str(exc)
                    ttft_ms = self.config.probe_timeout_s * 1000.0
                    tat_ms = self.config.probe_timeout_s * 1000.0
            elif reg.probe_fn is not None:
                try:
                    ttft_ms, tat_ms, tps, token_count = await measure_streaming_probe(
                        reg.probe_fn,
                        timeout_s=self.config.probe_timeout_s,
                    )
                except Exception as exc:
                    success = False
                    error_msg = str(exc)
                    logger.warning("Probe inference failed for %s: %s", model_id, exc)
                    ttft_ms = self.config.probe_timeout_s * 1000.0
                    tat_ms = self.config.probe_timeout_s * 1000.0
            else:
                # Default baseline based on tier
                base = {0: (1.0, 1.0, 15.0, 25.0, 200.0),
                        1: (2.0, 3.0, 85.0, 110.0, 80.0),
                        2: (15.0, 25.0, 220.0, 350.0, 60.0),
                        3: (20.0, 30.0, 650.0, 950.0, 40.0)}.get(reg.tier, (5.0, 5.0, 100.0, 150.0, 50.0))
                dns_ms, tcp_ms, ttft_ms, tat_ms, tps = base
                token_count = 10

            # 3. Record Sample and Update Rolling Stats
            sample = LatencySample(
                timestamp=now,
                dns_ms=max(0.0, dns_ms),
                tcp_ms=max(0.0, tcp_ms),
                ttft_ms=ttft_ms,
                tat_ms=tat_ms,
                tps=tps,
                token_count=token_count,
                success=success,
                error_message=error_msg,
            )
            stats.update(sample)

            metrics = self._build_metrics(model_id, reg.tier, stats)

            # 4. Telemetry publication
            if self.event_bus:
                try:
                    loop = asyncio.get_running_loop()
                    loop.create_task(
                        self.event_bus.publish_event(
                            EventType.TELEMETRY_UPDATE,
                            model_id=model_id,
                            tier=reg.tier,
                            ttft_ms=metrics.ttft_ms,
                            tat_ms=metrics.tat_ms,
                            tps=metrics.tps,
                            healthy=metrics.healthy,
                        )
                    )
                except Exception:
                    pass

            return metrics

    async def probe_all(self, force: bool = False) -> list[ModelLatencyMetrics]:
        """Probe all registered models concurrently within configured concurrency limits."""
        tasks = [self.probe_model(mid, force=force) for mid in list(self._models.keys())]
        if not tasks:
            return []
        results = await asyncio.gather(*tasks, return_exceptions=True)
        valid_metrics: list[ModelLatencyMetrics] = []
        for r in results:
            if isinstance(r, ModelLatencyMetrics):
                valid_metrics.append(r)
            elif isinstance(r, Exception):
                logger.error("Error during probe_all: %s", r)
        return valid_metrics

    async def benchmark_all(self, force: bool = False) -> list[ModelLatencyMetrics]:
        """Async method probing all registered models and returning complete metrics suite."""
        return await self.probe_all(force=force)

    def record_live_sample(
        self,
        model_id: str,
        ttft_ms: float,
        tat_ms: float,
        token_count: int,
        success: bool = True,
    ) -> None:
        """Passively record metrics from live user query executions."""
        stats = self._stats.get(model_id)
        if not stats:
            return
        now = time.time()
        gen_time_s = max(0.001, (tat_ms - ttft_ms) / 1000.0)
        tps = (token_count / gen_time_s) if token_count > 0 else 0.0

        sample = LatencySample(
            timestamp=now,
            dns_ms=stats.ema_dns_ms,
            tcp_ms=stats.ema_tcp_ms,
            ttft_ms=ttft_ms,
            tat_ms=tat_ms,
            tps=tps,
            token_count=token_count,
            success=success,
        )
        stats.update(sample)

    def get_metrics(self, model_id: str) -> Optional[ModelLatencyMetrics]:
        """Return the current smoothed metrics for a model."""
        reg = self._models.get(model_id)
        stats = self._stats.get(model_id)
        if not reg or not stats:
            return None
        return self._build_metrics(model_id, reg.tier, stats)

    def get_all_metrics(self) -> list[ModelLatencyMetrics]:
        """Return current smoothed metrics for all registered models."""
        return [self._build_metrics(mid, r.tier, self._stats[mid]) for mid, r in self._models.items()]

    def get_fastest_model(self, tier: Optional[int] = None) -> Optional[str]:
        """Select the model_id with the lowest smoothed TTFT among healthy models."""
        all_metrics = self.get_all_metrics()
        candidates = [m for m in all_metrics if m.healthy]
        if tier is not None:
            candidates = [m for m in candidates if m.tier == tier]
        if not candidates:
            return None
        candidates.sort(key=lambda m: (m.ttft_ms, m.tat_ms))
        return candidates[0].model_id

    def _build_metrics(self, model_id: str, tier: int, stats: RollingLatencyStats) -> ModelLatencyMetrics:
        """Construct ModelLatencyMetrics adhering to PROJECT.md interface contract."""
        healthy = (stats.consecutive_failures < self.config.failure_threshold) and stats.total_probes > 0
        return ModelLatencyMetrics(
            model_id=model_id,
            tier=tier,
            dns_ms=round(stats.ema_dns_ms, 2),
            tcp_ms=round(stats.ema_tcp_ms, 2),
            ttft_ms=round(stats.ema_ttft_ms, 2),
            tat_ms=round(stats.ema_tat_ms, 2),
            tps=round(stats.ema_tps, 2),
            healthy=healthy,
            jitter_ms=round(stats.jitter_ms, 2),
            sample_count=len(stats.samples),
            last_updated=stats.last_updated,
        )

    # --------------------------------------------------------------------------
    # Lifecycle Management
    # --------------------------------------------------------------------------
    async def start(self) -> None:
        """Start the periodic background probing loop."""
        if self._running:
            return
        self._running = True
        self._probe_task = asyncio.create_task(self._background_loop(), name="latency_benchmark_harness")
        logger.info("LatencyBenchmarkHarness background probing started (interval=%.1fs)", self.config.probe_interval_s)

    async def stop(self) -> None:
        """Stop background probing and release resources."""
        if not self._running:
            return
        self._running = False
        if self._probe_task:
            self._probe_task.cancel()
            try:
                await self._probe_task
            except asyncio.CancelledError:
                pass
            self._probe_task = None
        logger.info("LatencyBenchmarkHarness stopped")

    async def _background_loop(self) -> None:
        """Continuous background probing loop."""
        while self._running:
            try:
                await self.probe_all()
            except Exception as exc:
                logger.exception("Error in background probe loop: %s", exc)
            await asyncio.sleep(self.config.probe_interval_s)

    async def __aenter__(self) -> LatencyBenchmarkHarness:
        await self.start()
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        await self.stop()
