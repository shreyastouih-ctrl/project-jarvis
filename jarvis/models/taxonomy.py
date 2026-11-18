"""4-Tier Model Taxonomy Engine, Profile Specifications, and Model Registry for Jarvis AI Assistant."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum, IntEnum
from typing import Any, Dict, Iterable, List, Optional, Set, Union

logger = logging.getLogger(__name__)


class ModelTier(IntEnum):
    """4-Tier Model Taxonomy based on response latency thresholds.

    Inherits from IntEnum so tier comparisons (e.g. tier == 0) and integer lookups
    work seamlessly across both integer primitives and typed enum variants.
    """
    TIER_0_DIRECT_REFLEX = 0
    TIER_1_ULTRA_FAST = 1
    TIER_2_BALANCED = 2
    TIER_3_DEEP_REASONING = 3

    # Canonical aliases
    REFLEX = 0
    EDGE_SLM = 1
    CLOUD_FAST = 2
    DEEP_REASON = 3


class ModelCapability(str, Enum):
    """Standardized capability tags for candidate models."""
    STREAMING = "streaming"
    FUNCTION_CALLING = "function_calling"
    VISION = "vision"
    AUDIO = "audio"
    LOCAL = "local"
    CODE = "code"
    REASONING = "reasoning"
    REFLEX = "reflex"
    SYSTEM_COMMANDS = "system_commands"


@dataclass(frozen=True)
class TierDefinition:
    """Descriptive metadata and constraints for a taxonomy tier."""
    tier: ModelTier
    name: str
    max_latency_ms: float
    description: str
    typical_use: str


TIER_DEFINITIONS: dict[int, TierDefinition] = {
    0: TierDefinition(
        tier=ModelTier.TIER_0_DIRECT_REFLEX,
        name="Direct Reflex",
        max_latency_ms=50.0,
        description="Instant rule-based, cached, or local deterministic reflexes requiring zero network round-trip",
        typical_use="Device control, state confirmation, system status queries, audio preemption",
    ),
    1: TierDefinition(
        tier=ModelTier.TIER_1_ULTRA_FAST,
        name="Ultra-Fast SLM",
        max_latency_ms=150.0,
        description="Local edge SLMs (e.g. Llama 3.2 1B) or ultra-low latency edge cloud endpoints (e.g. Gemini 2.0 Flash)",
        typical_use="Snappy conversational dialogue, rapid clarification, live voice responses",
    ),
    2: TierDefinition(
        tier=ModelTier.TIER_2_BALANCED,
        name="Balanced Conversational",
        max_latency_ms=500.0,
        description="Standard high-quality cloud conversational models (e.g. Claude 3.5 Haiku, GPT-4o-mini)",
        typical_use="General conversation, smart phone call screening dialogue, nuanced queries",
    ),
    3: TierDefinition(
        tier=ModelTier.TIER_3_DEEP_REASONING,
        name="Deep Analytical Reasoning",
        max_latency_ms=float("inf"),
        description="High-reasoning models with chain-of-thought deliberation (e.g. DeepSeek-R1, OpenAI o1)",
        typical_use="Complex technical troubleshooting, multi-step planning, deep debrief analysis",
    ),
}


@dataclass
class ModelLatencyMetrics:
    """Latency benchmark metrics for candidate models, adhering to PROJECT.md interface contract."""
    model_id: str
    tier: int               # 0, 1, 2, 3
    dns_ms: float
    tcp_ms: float
    ttft_ms: float
    tat_ms: float
    tps: float
    healthy: bool
    jitter_ms: float = 0.0
    sample_count: int = 0
    last_updated: float = 0.0


def classify_model_tier(latency_ms: float) -> ModelTier:
    """Classify model response latency (in milliseconds) into the 4-Tier taxonomy.

    Strict threshold boundaries:
    - Tier 0: Direct Reflex (<50ms)
    - Tier 1: Ultra-Fast SLM (<150ms)
    - Tier 2: Balanced Conversational (<500ms)
    - Tier 3: Deep Analytical Reasoning (>=500ms)
    """
    if latency_ms < 50.0:
        return ModelTier.TIER_0_DIRECT_REFLEX
    elif latency_ms < 150.0:
        return ModelTier.TIER_1_ULTRA_FAST
    elif latency_ms < 500.0:
        return ModelTier.TIER_2_BALANCED
    else:
        return ModelTier.TIER_3_DEEP_REASONING


@dataclass
class ModelProfile:
    """Specification, capabilities, and pricing profile for an LLM/SLM/Reflex model."""
    model_id: str
    tier: Union[ModelTier, int]
    provider: str
    endpoint_url: str
    name: Optional[str] = None
    context_window: int = 8192
    capabilities: set[str] = field(default_factory=set)
    cost_per_1m_tokens: float = 0.0
    cost_per_1m_input: float = 0.0
    cost_per_1m_output: float = 0.0
    healthy: bool = True
    default_healthy: bool = True
    timeout_seconds: float = 5.0
    is_synthetic: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.name is None:
            self.name = self.model_id
        if isinstance(self.tier, int) and not isinstance(self.tier, ModelTier):
            self.tier = ModelTier(self.tier)
        if isinstance(self.capabilities, (list, tuple)):
            self.capabilities = set(self.capabilities)
        if self.cost_per_1m_input > 0.0 and self.cost_per_1m_tokens == 0.0:
            self.cost_per_1m_tokens = self.cost_per_1m_input
        elif self.cost_per_1m_tokens > 0.0 and self.cost_per_1m_input == 0.0:
            self.cost_per_1m_input = self.cost_per_1m_tokens
        if not self.healthy:
            self.default_healthy = False
        if not self.default_healthy:
            self.healthy = False

    @property
    def cost_per_1m_input_tokens(self) -> float:
        return self.cost_per_1m_input

    @property
    def cost_per_1m_output_tokens(self) -> float:
        return self.cost_per_1m_output

    def has_capability(self, capability: str) -> bool:
        """Check if model provides a specific capability tag."""
        return capability.lower() in {c.lower() for c in self.capabilities}

    def has_capabilities(self, required: Iterable[str]) -> bool:
        """Check if model satisfies all required capability tags."""
        current = {c.lower() for c in self.capabilities}
        return all(r.lower() in current for r in required)


# Pre-configured catalog containing deterministic synthetic endpoints and industry models
DEFAULT_MODEL_CATALOG: dict[str, ModelProfile] = {
    # Tier 0: Direct Reflex
    "tier0-direct-reflex": ModelProfile(
        model_id="tier0-direct-reflex",
        name="Jarvis Core Direct Reflex",
        tier=ModelTier.TIER_0_DIRECT_REFLEX,
        provider="reflex",
        endpoint_url="local://reflex",
        context_window=4096,
        capabilities={"reflex", "streaming", "system_commands"},
        cost_per_1m_input=0.0,
        cost_per_1m_output=0.0,
        default_healthy=True,
        is_synthetic=True,
    ),
    # Tier 1: Ultra-Fast SLM / Edge
    "tier1-ultra-fast": ModelProfile(
        model_id="tier1-ultra-fast",
        name="Ultra-Fast Edge SLM",
        tier=ModelTier.TIER_1_ULTRA_FAST,
        provider="local",
        endpoint_url="http://localhost:11434/api/generate",
        context_window=8192,
        capabilities={"streaming", "function_calling", "local"},
        cost_per_1m_input=0.0,
        cost_per_1m_output=0.0,
        default_healthy=True,
        is_synthetic=True,
    ),
    "tier1-edge-slm": ModelProfile(
        model_id="tier1-edge-slm",
        name="Edge SLM Candidate",
        tier=ModelTier.TIER_1_ULTRA_FAST,
        provider="local",
        endpoint_url="http://localhost:11434/api/generate",
        context_window=8192,
        capabilities={"streaming", "function_calling", "local"},
        cost_per_1m_input=0.0,
        cost_per_1m_output=0.0,
        default_healthy=True,
        is_synthetic=True,
    ),
    "gemini-2.5-flash": ModelProfile(
        model_id="gemini-2.5-flash",
        name="Google Gemini 2.5 Flash",
        tier=ModelTier.TIER_1_ULTRA_FAST,
        provider="google",
        endpoint_url="https://generativelanguage.googleapis.com",
        context_window=1048576,
        capabilities={"streaming", "function_calling", "vision", "audio"},
        cost_per_1m_input=0.075,
        cost_per_1m_output=0.30,
        default_healthy=True,
        is_synthetic=False,
    ),
    "gemini-2.0-flash": ModelProfile(
        model_id="gemini-2.0-flash",
        name="Google Gemini 2.0 Flash",
        tier=ModelTier.TIER_1_ULTRA_FAST,
        provider="google",
        endpoint_url="https://generativelanguage.googleapis.com/v1beta",
        context_window=1048576,
        capabilities={"streaming", "function_calling", "vision", "audio"},
        cost_per_1m_input=0.10,
        cost_per_1m_output=0.40,
        default_healthy=True,
        is_synthetic=False,
    ),
    # Tier 2: Balanced Conversational
    "tier2-balanced": ModelProfile(
        model_id="tier2-balanced",
        name="Balanced Conversational Cloud",
        tier=ModelTier.TIER_2_BALANCED,
        provider="google",
        endpoint_url="https://generativelanguage.googleapis.com/v1beta",
        context_window=128000,
        capabilities={"streaming", "function_calling"},
        cost_per_1m_input=0.15,
        cost_per_1m_output=0.60,
        default_healthy=True,
        is_synthetic=True,
    ),
    "tier2-cloud-fast": ModelProfile(
        model_id="tier2-cloud-fast",
        name="Cloud Fast Candidate",
        tier=ModelTier.TIER_2_BALANCED,
        provider="google",
        endpoint_url="https://generativelanguage.googleapis.com/v1beta",
        context_window=128000,
        capabilities={"streaming", "function_calling", "vision"},
        cost_per_1m_input=0.15,
        cost_per_1m_output=0.60,
        default_healthy=True,
        is_synthetic=True,
    ),
    "claude-3-5-haiku": ModelProfile(
        model_id="claude-3-5-haiku",
        name="Anthropic Claude 3.5 Haiku",
        tier=ModelTier.TIER_2_BALANCED,
        provider="anthropic",
        endpoint_url="https://api.anthropic.com/v1/messages",
        context_window=200000,
        capabilities={"streaming", "function_calling", "vision"},
        cost_per_1m_input=0.80,
        cost_per_1m_output=4.00,
        default_healthy=True,
        is_synthetic=False,
    ),
    "gpt-4o-mini": ModelProfile(
        model_id="gpt-4o-mini",
        name="OpenAI GPT-4o-mini",
        tier=ModelTier.TIER_2_BALANCED,
        provider="openai",
        endpoint_url="https://api.openai.com/v1/chat/completions",
        context_window=128000,
        capabilities={"streaming", "function_calling", "vision"},
        cost_per_1m_input=0.15,
        cost_per_1m_output=0.60,
        default_healthy=True,
        is_synthetic=False,
    ),
    # Tier 3: Deep Analytical Reasoning
    "tier3-deep-reasoning": ModelProfile(
        model_id="tier3-deep-reasoning",
        name="Deep Analytical Reasoner",
        tier=ModelTier.TIER_3_DEEP_REASONING,
        provider="openai",
        endpoint_url="https://api.openai.com/v1",
        context_window=64000,
        capabilities={"streaming", "reasoning", "code"},
        cost_per_1m_input=15.0,
        cost_per_1m_output=60.0,
        default_healthy=True,
        is_synthetic=True,
    ),
    "tier3-deep-reason": ModelProfile(
        model_id="tier3-deep-reason",
        name="Deep Reason Candidate",
        tier=ModelTier.TIER_3_DEEP_REASONING,
        provider="openai",
        endpoint_url="https://api.openai.com/v1",
        context_window=64000,
        capabilities={"streaming", "reasoning", "code"},
        cost_per_1m_input=15.0,
        cost_per_1m_output=60.0,
        default_healthy=True,
        is_synthetic=True,
    ),
    "deepseek-r1": ModelProfile(
        model_id="deepseek-r1",
        name="DeepSeek R1",
        tier=ModelTier.TIER_3_DEEP_REASONING,
        provider="deepseek",
        endpoint_url="https://api.deepseek.com/v1/chat/completions",
        context_window=64000,
        capabilities={"streaming", "reasoning", "code"},
        cost_per_1m_input=0.55,
        cost_per_1m_output=2.19,
        default_healthy=True,
        is_synthetic=False,
    ),
}


class ModelRegistry:
    """Catalog holding model specifications and operational status."""

    def __init__(self, include_defaults: bool = True) -> None:
        self._models: dict[str, ModelProfile] = {}
        if include_defaults:
            for profile in DEFAULT_MODEL_CATALOG.values():
                self.register(profile)

    def register(self, profile: ModelProfile) -> None:
        """Register or update a model profile."""
        self._models[profile.model_id] = profile

    def get(self, model_id: str) -> Optional[ModelProfile]:
        """Retrieve model profile by ID."""
        return self._models.get(model_id)

    def list_models(
        self,
        tier: Optional[Union[ModelTier, int]] = None,
        capabilities: Optional[Iterable[str]] = None,
        healthy_only: bool = False,
    ) -> list[ModelProfile]:
        """Filter models by tier, capabilities, and health status."""
        results = []
        for profile in self._models.values():
            if tier is not None and int(profile.tier) != int(tier):
                continue
            if healthy_only and not profile.default_healthy:
                continue
            if capabilities and not profile.has_capabilities(capabilities):
                continue
            results.append(profile)
        return results

    def unregister(self, model_id: str) -> bool:
        """Remove a model from the registry."""
        return self._models.pop(model_id, None) is not None

    def __len__(self) -> int:
        return len(self._models)


default_registry = ModelRegistry()
