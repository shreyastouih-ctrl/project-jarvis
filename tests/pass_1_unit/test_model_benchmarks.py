"""Comprehensive headless unit tests for Jarvis Model Taxonomy, Benchmark Harness & Fast-Path Router.

Verifies 7 Core Milestone 4 Capabilities:
1. Test 1: Tier categorization logic (Tier 0, 1, 2, 3 based on TTFT and capability tags).
2. Test 2: LatencyBenchmarkHarness calculation accuracy (DNS, TCP, TTFT, TAT, TPS) and EMA convergence.
3. Test 3: Dynamic Model Selector chooses fastest healthy model matching capability tags and tier limits.
4. Test 4: Circuit breaker transitions from CLOSED -> OPEN after consecutive threshold failures, diverting requests.
5. Test 5: Circuit breaker transitions from OPEN -> HALF_OPEN after recovery cool-off, resetting to CLOSED on success.
6. Test 6: Fallback routing cascade when all preferred tier models are unavailable.
7. Test 7: Conformance to ModelRouterProtocol and jarvis.core.event_bus INTENT_DISPATCHED event dispatching.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any, AsyncGenerator

import pytest

from jarvis.core.event_bus import Event, EventBus, EventType
from jarvis.models.benchmark import LatencyBenchmarkHarness, ModelLatencyMetrics
from jarvis.models.mock_provider import (
    MockConnectionTimeoutError,
    MockFailureType,
    MockInternalServerError,
    MockRateLimitError,
    MockSocketDroppedError,
    SyntheticLatencyModelProvider,
    SyntheticLatencyProfile,
)
from jarvis.models.router import (
    CircuitBreaker,
    CircuitBreakerConfig,
    CircuitBreakerState,
    DynamicModelRouter,
    ModelRouterProtocol,
    NoViableModelError,
)
from jarvis.models.taxonomy import (
    DEFAULT_MODEL_CATALOG,
    ModelProfile,
    ModelTier,
    classify_model_tier,
)


# ==============================================================================
# FIXTURES
# ==============================================================================
@pytest.fixture
def mock_provider() -> SyntheticLatencyModelProvider:
    """Provide a deterministic synthetic latency provider in virtual-clock mode."""
    return SyntheticLatencyModelProvider(seed=42, virtual_clock=True)


@pytest.fixture
def benchmark_harness(mock_provider: SyntheticLatencyModelProvider) -> LatencyBenchmarkHarness:
    """Provide an active LatencyBenchmarkHarness backed by mock_provider."""
    return LatencyBenchmarkHarness(provider=mock_provider, alpha=0.3)


@pytest.fixture
def event_bus() -> EventBus:
    """Provide a clean EventBus."""
    return EventBus()


@pytest.fixture
def model_router(
    mock_provider: SyntheticLatencyModelProvider,
    benchmark_harness: LatencyBenchmarkHarness,
    event_bus: EventBus,
) -> DynamicModelRouter:
    """Provide a DynamicModelRouter configured with circuit breaking and event bus."""
    cb_config = CircuitBreakerConfig(
        failure_threshold=3,
        recovery_cooloff_seconds=0.05,  # Short cooloff for fast unit testing
        half_open_trial_count=1,
    )
    return DynamicModelRouter(
        provider=mock_provider,
        harness=benchmark_harness,
        event_bus=event_bus,
        circuit_breaker_config=cb_config,
    )


# ==============================================================================
# 1. TIER CATEGORIZATION LOGIC & TAXONOMY
# ==============================================================================
class TestModelTaxonomyAndCategorization:
    """Test 1: Verify correct classification of models into Tier 0, 1, 2, 3 based on TTFT and tags."""

    def test_tier_categorization_threshold_boundaries(self) -> None:
        """Verify exact boundary classifications for all 4 latency tiers."""
        # Tier 0: Direct Reflex (<50ms)
        assert classify_model_tier(0.0) == ModelTier.TIER_0_DIRECT_REFLEX
        assert classify_model_tier(25.0) == ModelTier.TIER_0_DIRECT_REFLEX
        assert classify_model_tier(49.9) == ModelTier.TIER_0_DIRECT_REFLEX

        # Tier 1: Ultra-Fast (<150ms)
        assert classify_model_tier(50.0) == ModelTier.TIER_1_ULTRA_FAST
        assert classify_model_tier(100.0) == ModelTier.TIER_1_ULTRA_FAST
        assert classify_model_tier(149.9) == ModelTier.TIER_1_ULTRA_FAST

        # Tier 2: Balanced Conversational (<500ms)
        assert classify_model_tier(150.0) == ModelTier.TIER_2_BALANCED
        assert classify_model_tier(300.0) == ModelTier.TIER_2_BALANCED
        assert classify_model_tier(499.9) == ModelTier.TIER_2_BALANCED

        # Tier 3: Deep Analytical Reasoning (>=500ms)
        assert classify_model_tier(500.0) == ModelTier.TIER_3_DEEP_REASONING
        assert classify_model_tier(950.0) == ModelTier.TIER_3_DEEP_REASONING
        assert classify_model_tier(5000.0) == ModelTier.TIER_3_DEEP_REASONING

    def test_model_profile_creation_and_attributes(self) -> None:
        """Verify ModelProfile retains capabilities, cost metadata, and endpoint details."""
        profile = ModelProfile(
            model_id="gemini-2.5-flash",
            tier=ModelTier.TIER_1_ULTRA_FAST,
            provider="google",
            endpoint_url="https://generativelanguage.googleapis.com",
            context_window=1048576,
            capabilities={"streaming", "function_calling", "vision"},
            cost_per_1m_input=0.075,
            cost_per_1m_output=0.30,
            default_healthy=True,
        )
        assert profile.model_id == "gemini-2.5-flash"
        assert profile.tier == 1
        assert "function_calling" in profile.capabilities
        assert profile.cost_per_1m_input == 0.075
        assert profile.default_healthy is True

    def test_default_catalog_contains_representative_entries(self) -> None:
        """Verify pre-configured catalog includes models across all tiers."""
        catalog = DEFAULT_MODEL_CATALOG
        assert len(catalog) >= 4

        # Check representation of each tier
        tiers_present = {int(m.tier) for m in catalog.values()}
        assert {0, 1, 2, 3}.issubset(tiers_present)


# ==============================================================================
# 2. BENCHMARK HARNESS CALCULATION & EMA CONVERGENCE
# ==============================================================================
class TestBenchmarkHarnessAndEMA:
    """Test 2: LatencyBenchmarkHarness calculation accuracy (DNS, TCP, TTFT, TAT, TPS) and EMA convergence."""

    @pytest.mark.asyncio
    async def test_metric_calculation_accuracy_and_invariants(
        self,
        benchmark_harness: LatencyBenchmarkHarness,
    ) -> None:
        """Verify calculated metrics satisfy latency invariants and protocol contract."""
        metrics = await benchmark_harness.probe_model("tier1-ultra-fast")
        assert isinstance(metrics, ModelLatencyMetrics)
        assert metrics.model_id == "tier1-ultra-fast"
        assert metrics.tier == 1
        assert metrics.dns_ms >= 0.0
        assert metrics.tcp_ms >= 0.0
        assert metrics.ttft_ms > 0.0
        assert metrics.tat_ms >= metrics.ttft_ms
        assert metrics.tps > 0.0
        assert metrics.healthy is True

    @pytest.mark.asyncio
    async def test_exponential_moving_average_convergence(
        self,
        mock_provider: SyntheticLatencyModelProvider,
    ) -> None:
        """Verify EMA mathematical formula: EMA_new = alpha * sample + (1 - alpha) * EMA_old."""
        alpha = 0.3
        harness = LatencyBenchmarkHarness(provider=mock_provider, alpha=alpha)

        # Feed 3 deterministic samples: 100ms, 120ms, 80ms
        samples = [100.0, 120.0, 80.0]
        expected_ema = samples[0]  # First sample initializes EMA: 100.0

        harness.record_sample("custom-test-model", ttft_ms=samples[0])
        assert pytest.approx(harness.get_smoothed_ttft("custom-test-model"), rel=1e-3) == expected_ema

        # Second sample: 0.3 * 120 + 0.7 * 100 = 36 + 70 = 106.0
        expected_ema = alpha * samples[1] + (1 - alpha) * expected_ema
        harness.record_sample("custom-test-model", ttft_ms=samples[1])
        assert pytest.approx(harness.get_smoothed_ttft("custom-test-model"), rel=1e-3) == expected_ema

        # Third sample: 0.3 * 80 + 0.7 * 106 = 24 + 74.2 = 98.2
        expected_ema = alpha * samples[2] + (1 - alpha) * expected_ema
        harness.record_sample("custom-test-model", ttft_ms=samples[2])
        assert pytest.approx(harness.get_smoothed_ttft("custom-test-model"), rel=1e-3) == expected_ema

    @pytest.mark.asyncio
    async def test_benchmark_all_collects_complete_model_suite(
        self,
        benchmark_harness: LatencyBenchmarkHarness,
    ) -> None:
        """Verify benchmark_all() probes all registered models."""
        all_metrics = await benchmark_harness.benchmark_all()
        assert len(all_metrics) >= 4
        model_ids = {m.model_id for m in all_metrics}
        assert "tier0-direct-reflex" in model_ids
        assert "tier1-ultra-fast" in model_ids
        assert "tier2-balanced" in model_ids
        assert "tier3-deep-reasoning" in model_ids


# ==============================================================================
# 3. DYNAMIC MODEL SELECTOR CHOOSES FASTEST HEALTHY MODEL
# ==============================================================================
class TestDynamicModelSelector:
    """Test 3: Dynamic Model Selector chooses fastest healthy model matching capability tags and tier limits."""

    def test_select_fastest_unconstrained(
        self,
        model_router: DynamicModelRouter,
    ) -> None:
        """Verify unconstrained selection picks the absolute fastest healthy model (Tier 0)."""
        fastest = model_router.select_fastest()
        assert fastest == "tier0-direct-reflex"

    def test_select_fastest_with_tier_constraint(
        self,
        model_router: DynamicModelRouter,
    ) -> None:
        """Verify tier-constrained selection filters candidate pool to specified tier."""
        selected_tier1 = model_router.select_fastest(required_tier=1)
        assert selected_tier1 == "tier1-ultra-fast"

        selected_tier2 = model_router.select_fastest(required_tier=2)
        assert selected_tier2 == "tier2-balanced"

        selected_tier3 = model_router.select_fastest(required_tier=3)
        assert selected_tier3 == "tier3-deep-reasoning"

    def test_select_fastest_with_capability_tags(
        self,
        model_router: DynamicModelRouter,
        mock_provider: SyntheticLatencyModelProvider,
    ) -> None:
        """Verify capability filtering routes to capable model even if slower."""
        # Register a specialized vision model in Tier 2
        mock_provider.register_model(
            "tier2-vision-model",
            SyntheticLatencyProfile(tier=2, dns_ms=10.0, tcp_ms=20.0, ttft_ms=210.0, tat_ms=300.0, tps=50.0),
        )
        model_router.register_catalog_entry(
            ModelProfile(
                model_id="tier2-vision-model",
                tier=ModelTier.TIER_2_BALANCED,
                provider="custom",
                endpoint_url="http://localhost",
                context_window=8192,
                capabilities={"vision"},
            )
        )

        selected = model_router.select_fastest(required_tags={"vision"})
        assert selected == "tier2-vision-model"


# ==============================================================================
# 4. CIRCUIT BREAKER TRANSITION CLOSED -> OPEN ON CONSECUTIVE FAILURES
# ==============================================================================
class TestCircuitBreakerTripping:
    """Test 4: Circuit breaker transitions CLOSED -> OPEN after consecutive failures, diverting to backup."""

    @pytest.mark.asyncio
    async def test_consecutive_failures_trip_circuit_breaker(
        self,
        model_router: DynamicModelRouter,
        mock_provider: SyntheticLatencyModelProvider,
    ) -> None:
        """Verify N consecutive failures transition circuit breaker state to OPEN."""
        target_model = "tier0-direct-reflex"
        cb = model_router.get_circuit_breaker(target_model)
        assert cb.state == CircuitBreakerState.CLOSED

        # Configure 3 consecutive 500 Internal Errors
        mock_provider.configure_failure(target_model, fail_next_n=3, error_type=MockFailureType.INTERNAL_ERROR)

        # Attempt 1: failure recorded, state remains CLOSED
        with pytest.raises(MockInternalServerError):
            await model_router.generate_with_model("test prompt", target_model)
        assert cb.consecutive_failures == 1
        assert cb.state == CircuitBreakerState.CLOSED

        # Attempt 2: failure recorded, state remains CLOSED
        with pytest.raises(MockInternalServerError):
            await model_router.generate_with_model("test prompt", target_model)
        assert cb.consecutive_failures == 2
        assert cb.state == CircuitBreakerState.CLOSED

        # Attempt 3: threshold reached (3), trips to OPEN!
        with pytest.raises(MockInternalServerError):
            await model_router.generate_with_model("test prompt", target_model)
        assert cb.consecutive_failures == 3
        assert cb.state == CircuitBreakerState.OPEN

    @pytest.mark.asyncio
    async def test_router_automatically_diverts_traffic_away_from_open_model(
        self,
        model_router: DynamicModelRouter,
        mock_provider: SyntheticLatencyModelProvider,
    ) -> None:
        """Verify dynamic router bypasses OPEN model and selects next fastest healthy candidate."""
        # Trip tier0 model to OPEN
        mock_provider.configure_failure("tier0-direct-reflex", fail_next_n=3)
        cb0 = model_router.get_circuit_breaker("tier0-direct-reflex")
        for _ in range(3):
            try:
                await model_router.generate_with_model("trip", "tier0-direct-reflex")
            except MockInternalServerError:
                pass
        assert cb0.state == CircuitBreakerState.OPEN

        # Next general selection should bypass tier0 and choose tier1!
        selected = model_router.select_fastest()
        assert selected == "tier1-ultra-fast"

        # General generate() succeeds using backup model without failing user request
        resp, used_model, latency = await model_router.generate("Status report")
        assert used_model == "tier1-ultra-fast"
        assert len(resp) > 0


# ==============================================================================
# 5. CIRCUIT BREAKER RECOVERY OPEN -> HALF_OPEN -> CLOSED
# ==============================================================================
class TestCircuitBreakerRecovery:
    """Test 5: Circuit breaker transitions OPEN -> HALF_OPEN after cool-off, resetting to CLOSED on success."""

    @pytest.mark.asyncio
    async def test_open_transitions_to_half_open_after_cooloff(
        self,
        model_router: DynamicModelRouter,
    ) -> None:
        """Verify cool-off expiration moves circuit breaker to HALF_OPEN."""
        target_model = "tier1-ultra-fast"
        cb = model_router.get_circuit_breaker(target_model)
        cb.trip()  # Manually trip to OPEN
        assert cb.state == CircuitBreakerState.OPEN

        # Before cool-off expires: remains OPEN
        assert not cb.can_attempt()

        # Wait for cool-off (configured to 0.05s in fixture)
        await asyncio.sleep(0.06)

        # After cool-off: transitions to HALF_OPEN and allows probe
        assert cb.state == CircuitBreakerState.HALF_OPEN
        assert cb.can_attempt()

    @pytest.mark.asyncio
    async def test_successful_probe_resets_half_open_to_closed(
        self,
        model_router: DynamicModelRouter,
        mock_provider: SyntheticLatencyModelProvider,
    ) -> None:
        """Verify successful probe in HALF_OPEN resets state to CLOSED and clears failures."""
        target_model = "tier1-ultra-fast"
        cb = model_router.get_circuit_breaker(target_model)
        cb.trip()
        await asyncio.sleep(0.06)
        assert cb.state == CircuitBreakerState.HALF_OPEN

        # Clear any failures in provider so probe succeeds
        mock_provider.clear_failures(target_model)

        # Execute probe call
        resp, used_model, _ = await model_router.generate_with_model("probe ping", target_model)
        assert used_model == target_model
        assert cb.state == CircuitBreakerState.CLOSED
        assert cb.consecutive_failures == 0

    @pytest.mark.asyncio
    async def test_failed_probe_in_half_open_retrips_to_open(
        self,
        model_router: DynamicModelRouter,
        mock_provider: SyntheticLatencyModelProvider,
    ) -> None:
        """Verify probe failure in HALF_OPEN immediately re-trips back to OPEN."""
        target_model = "tier2-balanced"
        cb = model_router.get_circuit_breaker(target_model)
        cb.trip()
        await asyncio.sleep(0.06)
        assert cb.state == CircuitBreakerState.HALF_OPEN

        # Inject failure for probe
        mock_provider.configure_failure(target_model, fail_next_n=1)

        with pytest.raises(MockInternalServerError):
            await model_router.generate_with_model("probe ping", target_model)

        assert cb.state == CircuitBreakerState.OPEN


# ==============================================================================
# 6. FALLBACK ROUTING WHEN ALL PREFERRED TIER MODELS ARE UNAVAILABLE
# ==============================================================================
class TestFallbackRouting:
    """Test 6: Fallback routing cascade when all preferred tier models are unavailable."""

    def test_fallback_to_adjacent_tier_when_preferred_tier_down(
        self,
        model_router: DynamicModelRouter,
    ) -> None:
        """Verify fallback cascades to Tier 0 or Tier 2 when Tier 1 is completely unavailable."""
        # Trip tier1 model
        cb1 = model_router.get_circuit_breaker("tier1-ultra-fast")
        cb1.trip()

        # Request with required_tier=1 falls back gracefully to fastest available healthy tier
        fallback_model = model_router.select_fastest(required_tier=1, allow_fallback=True)
        assert fallback_model in ("tier0-direct-reflex", "tier2-balanced")

    def test_all_models_unavailable_raises_no_viable_model_error(
        self,
        model_router: DynamicModelRouter,
    ) -> None:
        """Verify NoViableModelError is raised if every single candidate model is tripped."""
        for m_id in ["tier0-direct-reflex", "tier1-ultra-fast", "tier2-balanced", "tier3-deep-reasoning"]:
            cb = model_router.get_circuit_breaker(m_id)
            cb.trip()

        with pytest.raises(NoViableModelError):
            model_router.select_fastest()


# ==============================================================================
# 7. CONFORMANCE TO ModelRouterProtocol & EventBus INTENT_DISPATCHED EVENT
# ==============================================================================
class TestModelRouterProtocolAndEventBus:
    """Test 7: Conformance to ModelRouterProtocol and jarvis.core.event_bus INTENT_DISPATCHED event."""

    def test_protocol_structural_conformance(
        self,
        model_router: DynamicModelRouter,
    ) -> None:
        """Verify DynamicModelRouter implements all required methods of ModelRouterProtocol."""
        assert hasattr(model_router, "benchmark_all")
        assert callable(model_router.benchmark_all)
        assert hasattr(model_router, "select_fastest")
        assert callable(model_router.select_fastest)
        assert hasattr(model_router, "generate")
        assert callable(model_router.generate)

    @pytest.mark.asyncio
    async def test_generate_publishes_intent_dispatched_event(
        self,
        model_router: DynamicModelRouter,
        event_bus: EventBus,
    ) -> None:
        """Verify generate() publishes EventType.INTENT_DISPATCHED with complete metadata payload."""
        dispatched_events: list[Event] = []

        async def intent_handler(event: Event) -> None:
            dispatched_events.append(event)

        event_bus.subscribe(EventType.INTENT_DISPATCHED, intent_handler)

        prompt = "Jarvis, compute optimal flight vector"
        response, selected_id, latency_ms = await model_router.generate(prompt)

        assert len(response) > 0
        assert selected_id == "tier0-direct-reflex"
        assert latency_ms > 0.0

        # Wait for bus event dispatch
        ev = await event_bus.wait_for(EventType.INTENT_DISPATCHED, timeout=1.0)
        assert ev.get("intent") == prompt
        assert ev.get("model_id") == selected_id
        assert ev.get("tier") == 0


# ==============================================================================
# 8. SYNTHETIC MODEL PROVIDER ERROR INJECTION & TOKEN STREAMING
# ==============================================================================
class TestSyntheticModelProviderCapabilities:
    """Unit tests for synthetic provider error injection types and token streaming."""

    @pytest.mark.asyncio
    async def test_simulated_error_types_raised_accurately(
        self,
        mock_provider: SyntheticLatencyModelProvider,
    ) -> None:
        """Verify each MockFailureType raises corresponding typed exception."""
        # 1. 429 Rate Limit
        mock_provider.configure_failure("tier1-ultra-fast", fail_next_n=1, error_type=MockFailureType.RATE_LIMIT)
        with pytest.raises(MockRateLimitError) as exc_info:
            await mock_provider.generate_text("hi", "tier1-ultra-fast")
        assert exc_info.value.status_code == 429
        assert exc_info.value.retry_after_s > 0

        # 2. Timeout
        mock_provider.configure_failure("tier1-ultra-fast", fail_next_n=1, error_type=MockFailureType.TIMEOUT)
        with pytest.raises(MockConnectionTimeoutError) as exc_info:
            await mock_provider.generate_text("hi", "tier1-ultra-fast")
        assert exc_info.value.status_code == 504

        # 3. Socket Dropped
        mock_provider.configure_failure("tier1-ultra-fast", fail_next_n=1, error_type=MockFailureType.SOCKET_DROPPED)
        with pytest.raises(MockSocketDroppedError) as exc_info:
            await mock_provider.generate_text("hi", "tier1-ultra-fast")
        assert exc_info.value.status_code == 502

    @pytest.mark.asyncio
    async def test_streaming_async_generator_token_yield(
        self,
        mock_provider: SyntheticLatencyModelProvider,
    ) -> None:
        """Verify streaming generator yields words iteratively."""
        tokens = []
        async for token in mock_provider.stream_generate("System diagnostic", "tier0-direct-reflex"):
            tokens.append(token)

        assert len(tokens) >= 5
        full_text = "".join(tokens)
        assert "Certainly" in full_text
        assert "System diagnostic" in full_text

    @pytest.mark.asyncio
    async def test_mid_stream_socket_drop_failure(
        self,
        mock_provider: SyntheticLatencyModelProvider,
    ) -> None:
        """Verify socket drop triggered mid-stream aborts generator."""
        mock_provider.configure_failure(
            "tier1-ultra-fast",
            fail_next_n=1,
            error_type=MockFailureType.SOCKET_DROPPED,
            fail_on_token_index=3,
        )

        tokens = []
        with pytest.raises(MockSocketDroppedError):
            async for token in mock_provider.stream_generate("Stream test", "tier1-ultra-fast"):
                tokens.append(token)

        # Successfully yielded 3 tokens before abrupt failure
        assert len(tokens) == 3


# ==============================================================================
# 9. SUITE RUNTIME & HEADLESS EXECUTION PERFORMANCE
# ==============================================================================
class TestExecutionPerformance:
    """Verify execution time is strictly under 3.0 seconds."""

    @pytest.mark.asyncio
    async def test_virtual_clock_execution_is_subsecond(
        self,
        mock_provider: SyntheticLatencyModelProvider,
    ) -> None:
        """Verify virtual clock allows probing all tiers in under 50ms of CPU time."""
        start = time.monotonic()
        for _ in range(50):
            mock_provider.sample_metrics("tier3-deep-reasoning")
        duration = time.monotonic() - start
        assert duration < 0.05, f"Virtual clock metrics took {duration:.4f}s, expected < 0.05s"
