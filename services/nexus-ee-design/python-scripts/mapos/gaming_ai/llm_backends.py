#!/usr/bin/env python3
"""
LLM Backends - OpenRouter-based replacements for neural network components.

This module provides LLM-based implementations that replace the PyTorch
neural network components, enabling the Gaming AI system to run without
any local GPU or ML infrastructure.

Components:
- LLMStateEncoder: Replaces PCBGraphEncoder (GNN)
- LLMValueEstimator: Replaces ValueNetwork (MLP)
- LLMPolicyGenerator: Replaces PolicyNetwork (action selection)
- LLMDynamicsSimulator: Replaces DynamicsNetwork (world model)

Author: Adverant Inc.
License: MIT
"""

import os
import json
import hashlib
import logging
import asyncio
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple
from functools import lru_cache

import numpy as np

try:
    import httpx
    HTTPX_AVAILABLE = True
except ImportError:
    HTTPX_AVAILABLE = False

from .config import GamingAIConfig, LLMConfig, DRCConfig, get_config

logger = logging.getLogger(__name__)


class LLMClient:
    """
    Async client for OpenRouter API.

    Handles authentication, retries, and error handling for LLM calls.
    """

    def __init__(self, config: Optional[LLMConfig] = None):
        """Initialize LLM client with configuration."""
        self.config = config or LLMConfig()
        self._client: Optional["httpx.AsyncClient"] = None

    async def _get_client(self) -> "httpx.AsyncClient":
        """Get or create async HTTP client."""
        if not HTTPX_AVAILABLE:
            raise ImportError("httpx required for LLM backend. Install with: pip install httpx")

        if self._client is None:
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(self.config.timeout_seconds),
                headers={
                    "Authorization": f"Bearer {self.config.api_key}",
                    "HTTP-Referer": "https://adverant.ai",
                    "X-Title": "MAPOS Gaming AI",
                    "Content-Type": "application/json",
                }
            )
        return self._client

    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        json_mode: bool = True
    ) -> str:
        """
        Generate text completion from LLM.

        Args:
            prompt: User prompt
            system_prompt: Optional system prompt
            json_mode: If True, request JSON output

        Returns:
            Generated text response
        """
        if not self.config.api_key:
            raise ValueError("OpenRouter API key not configured")

        client = await self._get_client()

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": self.config.model,
            "messages": messages,
            "max_tokens": self.config.max_tokens,
            "temperature": self.config.temperature,
        }

        if json_mode:
            payload["response_format"] = {"type": "json_object"}

        last_error = None
        for attempt in range(self.config.max_retries):
            try:
                response = await client.post(
                    f"{self.config.base_url}/chat/completions",
                    json=payload
                )
                response.raise_for_status()
                data = response.json()
                return data["choices"][0]["message"]["content"]

            except Exception as e:
                last_error = e
                logger.warning(f"LLM call attempt {attempt + 1} failed: {e}")
                if attempt < self.config.max_retries - 1:
                    await asyncio.sleep(self.config.retry_delay_seconds * (attempt + 1))

        raise RuntimeError(f"LLM call failed after {self.config.max_retries} attempts: {last_error}")

    async def close(self):
        """Close the HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None


@dataclass
class StateEmbedding:
    """Result from state encoding."""
    embedding: np.ndarray
    structural_hash: str
    drc_signature: np.ndarray
    semantic_features: Dict[str, float]


class LLMStateEncoder:
    """
    LLM-based PCB state encoding.

    Replaces PCBGraphEncoder (GNN) with deterministic hashing
    and LLM semantic analysis. No PyTorch required.
    """

    def __init__(
        self,
        config: Optional[GamingAIConfig] = None,
        llm_client: Optional[LLMClient] = None
    ):
        """Initialize state encoder."""
        self.config = config or get_config()
        self.llm_client = llm_client or LLMClient(self.config.llm)
        self.embedding_dim = self.config.board.hidden_dim
        self._semantic_cache: Dict[str, np.ndarray] = {}

    def encode_state(self, pcb_state: Any) -> np.ndarray:
        """
        Encode PCB state to fixed-size embedding without neural networks.

        Uses deterministic hashing + LLM semantic analysis.

        Args:
            pcb_state: PCB state object

        Returns:
            256D numpy array embedding
        """
        # Part 1: Structural fingerprint (64D, deterministic)
        structural = self._extract_structural_features(pcb_state)

        # Part 2: DRC violation signature (64D, deterministic)
        drc_features = self._extract_drc_signature(pcb_state)

        # Part 3: Topological features (64D, deterministic)
        topological = self._extract_topological_features(pcb_state)

        # Part 4: Hash-based embedding (64D, deterministic)
        hash_embedding = self._get_hash_embedding(pcb_state)

        # Combine: 64 + 64 + 64 + 64 = 256D embedding
        embedding = np.concatenate([
            structural,
            drc_features,
            topological,
            hash_embedding
        ]).astype(np.float32)

        return embedding

    async def encode_state_with_llm(self, pcb_state: Any) -> StateEmbedding:
        """
        Encode state with additional LLM semantic analysis.

        This provides richer embeddings by using LLM to understand
        design intent and strategy.
        """
        # Get deterministic embedding
        embedding = self.encode_state(pcb_state)

        # Get structural hash
        structural_hash = self._compute_state_hash(pcb_state)

        # Check cache for semantic features
        if structural_hash in self._semantic_cache:
            semantic_embedding = self._semantic_cache[structural_hash]
        else:
            # Get LLM semantic analysis
            semantic_features = await self._analyze_design_semantics(pcb_state)
            semantic_embedding = self._features_to_embedding(semantic_features)
            self._semantic_cache[structural_hash] = semantic_embedding

        # Replace last 64D with semantic embedding
        enhanced = np.concatenate([
            embedding[:192],
            semantic_embedding
        ]).astype(np.float32)

        return StateEmbedding(
            embedding=enhanced,
            structural_hash=structural_hash,
            drc_signature=embedding[64:128],
            semantic_features={}
        )

    def _extract_structural_features(self, pcb_state: Any) -> np.ndarray:
        """Extract deterministic structural features (no LLM)."""
        features = []

        # Component statistics
        components = getattr(pcb_state, 'components', [])
        if components:
            positions = np.array([[c.x, c.y] for c in components if hasattr(c, 'x')])
            if len(positions) > 0:
                features.extend([
                    len(components) / 100.0,  # Normalized count
                    np.mean(positions[:, 0]) / self.config.board.default_width,
                    np.std(positions[:, 0]) / self.config.board.default_width if len(positions) > 1 else 0,
                    np.mean(positions[:, 1]) / self.config.board.default_height,
                    np.std(positions[:, 1]) / self.config.board.default_height if len(positions) > 1 else 0,
                    np.ptp(positions[:, 0]) / self.config.board.default_width,  # X range
                    np.ptp(positions[:, 1]) / self.config.board.default_height,  # Y range
                ])
            else:
                features.extend([0] * 7)
        else:
            features.extend([0] * 7)

        # Trace statistics
        traces = getattr(pcb_state, 'traces', [])
        if traces:
            widths = [t.width for t in traces if hasattr(t, 'width')]
            lengths = []
            for t in traces:
                if hasattr(t, 'start_x') and hasattr(t, 'end_x'):
                    length = np.sqrt((t.end_x - t.start_x)**2 + (t.end_y - t.start_y)**2)
                    lengths.append(length)

            features.extend([
                len(traces) / 500.0,
                np.mean(widths) / self.config.board.trace_width_max if widths else 0,
                np.std(widths) / self.config.board.trace_width_max if len(widths) > 1 else 0,
                np.mean(lengths) / 100.0 if lengths else 0,
                np.sum(lengths) / 10000.0 if lengths else 0,
            ])
        else:
            features.extend([0] * 5)

        # Via statistics
        vias = getattr(pcb_state, 'vias', [])
        if vias:
            via_positions = np.array([[v.x, v.y] for v in vias if hasattr(v, 'x')])
            diameters = [v.diameter for v in vias if hasattr(v, 'diameter')]

            features.extend([
                len(vias) / 200.0,
                np.mean(diameters) / self.config.board.via_diameter_max if diameters else 0,
                np.std(via_positions[:, 0]) / self.config.board.default_width if len(via_positions) > 1 else 0,
                np.std(via_positions[:, 1]) / self.config.board.default_height if len(via_positions) > 1 else 0,
            ])
        else:
            features.extend([0] * 4)

        # Zone statistics
        zones = getattr(pcb_state, 'zones', [])
        features.append(len(zones) / 20.0)

        # Net statistics
        nets = getattr(pcb_state, 'nets', [])
        features.append(len(nets) / 100.0)

        # Pad to 64D
        while len(features) < 64:
            features.append(0.0)

        return np.array(features[:64], dtype=np.float32)

    def _extract_drc_signature(self, pcb_state: Any) -> np.ndarray:
        """Extract DRC violation pattern signature."""
        features = np.zeros(64, dtype=np.float32)

        # Try to get DRC results
        drc = None
        if hasattr(pcb_state, 'run_drc'):
            try:
                drc = pcb_state.run_drc()
            except Exception as e:
                logger.debug(f"DRC run failed: {e}")

        if drc is None:
            return features

        # Violation type distribution
        violations_by_type = getattr(drc, 'violations_by_type', {})
        violation_types = [
            'clearance', 'track_width', 'via_dangling', 'track_dangling',
            'shorting_items', 'silk_over_copper', 'solder_mask_bridge',
            'courtyard_overlap', 'drill_size', 'annular_ring'
        ]

        for i, vtype in enumerate(violation_types):
            count = violations_by_type.get(vtype, 0)
            max_val = getattr(self.config.drc, f'{vtype}_max', 100)
            features[i] = np.clip(np.log1p(count) / np.log1p(max_val), 0, 1)

        # Total violations (normalized)
        total = getattr(drc, 'total_violations', 0)
        features[10] = np.clip(np.log1p(total) / np.log1p(1000), 0, 1)

        # Error/warning counts
        features[11] = len(getattr(drc, 'errors', [])) / 100.0
        features[12] = len(getattr(drc, 'warnings', [])) / 100.0

        # Violation spatial distribution (if available)
        if hasattr(drc, 'violation_positions'):
            positions = np.array(drc.violation_positions)
            if len(positions) > 0:
                features[13] = np.mean(positions[:, 0]) / self.config.board.default_width
                features[14] = np.std(positions[:, 0]) / self.config.board.default_width if len(positions) > 1 else 0
                features[15] = np.mean(positions[:, 1]) / self.config.board.default_height
                features[16] = np.std(positions[:, 1]) / self.config.board.default_height if len(positions) > 1 else 0

        return features

    def _extract_topological_features(self, pcb_state: Any) -> np.ndarray:
        """Extract topological/connectivity features."""
        features = np.zeros(64, dtype=np.float32)

        # Layer utilization
        layers = getattr(pcb_state, 'layers', [])
        features[0] = len(layers) / 16.0  # Normalize to max 16 layers

        # Net connectivity
        nets = getattr(pcb_state, 'nets', {})
        if isinstance(nets, dict):
            net_sizes = [len(pins) for pins in nets.values() if isinstance(pins, (list, set))]
        else:
            net_sizes = [len(getattr(n, 'pins', [])) for n in nets]

        if net_sizes:
            features[1] = len(net_sizes) / 100.0
            features[2] = np.mean(net_sizes) / 10.0
            features[3] = np.max(net_sizes) / 50.0
            features[4] = np.std(net_sizes) / 10.0 if len(net_sizes) > 1 else 0

        # Power/ground plane presence
        zones = getattr(pcb_state, 'zones', [])
        power_zones = sum(1 for z in zones if hasattr(z, 'net_name') and 'VCC' in str(z.net_name).upper())
        ground_zones = sum(1 for z in zones if hasattr(z, 'net_name') and 'GND' in str(z.net_name).upper())
        features[5] = power_zones / 5.0
        features[6] = ground_zones / 5.0

        # Routing density estimate
        traces = getattr(pcb_state, 'traces', [])
        components = getattr(pcb_state, 'components', [])
        if components:
            features[7] = len(traces) / (len(components) + 1)  # Traces per component

        return features

    def _get_hash_embedding(self, pcb_state: Any) -> np.ndarray:
        """Generate embedding from state hash."""
        state_hash = self._compute_state_hash(pcb_state)
        hash_int = int(state_hash[:16], 16)

        # Use hash as seed for reproducible random embedding
        rng = np.random.Generator(np.random.PCG64(hash_int))
        return rng.standard_normal(64).astype(np.float32)

    def _compute_state_hash(self, pcb_state: Any) -> str:
        """Compute unique hash for PCB state."""
        if hasattr(pcb_state, 'get_hash'):
            return pcb_state.get_hash()

        # Create hash from available attributes
        hash_parts = []

        for attr in ['components', 'traces', 'vias', 'zones', 'nets']:
            obj = getattr(pcb_state, attr, None)
            if obj is not None:
                hash_parts.append(f"{attr}:{len(obj) if hasattr(obj, '__len__') else 0}")

        hash_str = "|".join(hash_parts)
        return hashlib.sha256(hash_str.encode()).hexdigest()

    async def _analyze_design_semantics(self, pcb_state: Any) -> Dict[str, float]:
        """Use LLM to analyze design semantics."""
        summary = self._create_state_summary(pcb_state)

        prompt = f"""Analyze this PCB design and rate the following characteristics on a 0-1 scale:

{summary}

Rate each characteristic:
1. routing_density: How dense is the routing? (0=sparse, 1=very dense)
2. thermal_focus: How much is thermal management emphasized? (0=none, 1=high focus)
3. signal_integrity: How focused on SI? (0=basic, 1=high-speed design)
4. manufacturing_complexity: How complex to manufacture? (0=simple, 1=complex)
5. cost_optimization: How cost-optimized? (0=premium, 1=highly optimized)
6. layer_utilization: How well are layers used? (0=poor, 1=excellent)
7. component_density: How dense are components? (0=sparse, 1=dense)
8. design_maturity: How mature/refined is the design? (0=draft, 1=production-ready)

Return JSON with these 8 values.
"""

        try:
            response = await self.llm_client.generate(prompt, json_mode=True)
            return json.loads(response)
        except Exception as e:
            logger.warning(f"LLM semantic analysis failed: {e}")
            return {
                "routing_density": 0.5,
                "thermal_focus": 0.5,
                "signal_integrity": 0.5,
                "manufacturing_complexity": 0.5,
                "cost_optimization": 0.5,
                "layer_utilization": 0.5,
                "component_density": 0.5,
                "design_maturity": 0.5,
            }

    def _create_state_summary(self, pcb_state: Any) -> str:
        """Create text summary of PCB state for LLM."""
        parts = ["PCB Design Summary:"]

        components = getattr(pcb_state, 'components', [])
        parts.append(f"- Components: {len(components)}")

        traces = getattr(pcb_state, 'traces', [])
        parts.append(f"- Traces: {len(traces)}")

        vias = getattr(pcb_state, 'vias', [])
        parts.append(f"- Vias: {len(vias)}")

        zones = getattr(pcb_state, 'zones', [])
        parts.append(f"- Zones: {len(zones)}")

        layers = getattr(pcb_state, 'layers', [])
        parts.append(f"- Layers: {len(layers)}")

        if hasattr(pcb_state, 'board_size'):
            w, h = pcb_state.board_size
            parts.append(f"- Board size: {w:.1f}mm x {h:.1f}mm")

        return "\n".join(parts)

    def _features_to_embedding(self, features: Dict[str, float]) -> np.ndarray:
        """Convert feature dict to 64D embedding."""
        keys = sorted(features.keys())
        embedding = np.zeros(64, dtype=np.float32)

        for i, key in enumerate(keys[:8]):
            value = features.get(key, 0.5)
            # Expand each value to 8D using basis functions
            base_idx = i * 8
            embedding[base_idx] = value
            embedding[base_idx + 1] = value ** 2
            embedding[base_idx + 2] = np.sin(value * np.pi)
            embedding[base_idx + 3] = np.cos(value * np.pi)
            embedding[base_idx + 4] = value * (1 - value)  # Entropy
            embedding[base_idx + 5] = abs(value - 0.5)  # Distance from neutral
            embedding[base_idx + 6] = 1 - value
            embedding[base_idx + 7] = value ** 0.5

        return embedding


@dataclass
class ValueEstimate:
    """Result from value estimation."""
    quality_score: float
    fixability: float
    manufacturability: float
    reliability: float
    confidence: float
    reasoning: str


class LLMValueEstimator:
    """
    LLM-based PCB quality estimation.

    Replaces ValueNetwork (MLP) with intelligent DRC analysis.
    Provides more nuanced quality assessment than simple heuristics.
    """

    def __init__(
        self,
        config: Optional[GamingAIConfig] = None,
        llm_client: Optional[LLMClient] = None
    ):
        """Initialize value estimator."""
        self.config = config or get_config()
        self.llm_client = llm_client or LLMClient(self.config.llm)
        self._cache: Dict[str, ValueEstimate] = {}

    def estimate_quality_fast(self, pcb_state: Any) -> float:
        """
        Fast quality estimate without LLM (deterministic).

        Uses heuristic based on DRC violations.

        Args:
            pcb_state: PCB state object with run_drc() method.
                       If None or invalid, returns 0.5 (unknown quality).

        Returns:
            Quality score between 0.01 and 0.99

        Raises:
            No exceptions - all errors are caught and logged, returning
            a safe default value.
        """
        # Handle None or invalid input gracefully
        if pcb_state is None:
            logger.warning(
                "estimate_quality_fast called with None pcb_state. "
                "This indicates a bug in the calling code - pcb_state should "
                "always be provided for accurate value estimation."
            )
            return 0.5

        # Verify pcb_state has required interface
        if not hasattr(pcb_state, 'run_drc'):
            logger.warning(
                f"estimate_quality_fast: pcb_state ({type(pcb_state).__name__}) "
                f"does not have run_drc() method. Cannot estimate quality."
            )
            return 0.5

        # Run DRC and compute quality
        drc = None
        try:
            drc = pcb_state.run_drc()
        except Exception as e:
            logger.warning(f"DRC execution failed in estimate_quality_fast: {e}")
            return 0.5

        if drc is None:
            logger.debug("DRC returned None - using default quality estimate")
            return 0.5

        # Get violation count safely
        violations = getattr(drc, 'total_violations', 0)
        if not isinstance(violations, (int, float)):
            logger.warning(f"Invalid violations type: {type(violations)}. Using 0.")
            violations = 0

        # Compute quality score
        if violations == 0:
            return 0.99
        elif violations > 1000:
            return 0.01
        else:
            # Logarithmic scale with configurable target
            target = self.config.ralph_wiggum.target_violations
            score = max(0.01, 1.0 - np.log1p(violations) / np.log1p(target * 10))
            logger.debug(f"Quality estimate: {score:.4f} (violations={violations})")
            return score

    async def estimate_quality(self, pcb_state: Any) -> ValueEstimate:
        """
        Full quality estimation using LLM analysis.

        Provides detailed quality assessment with multiple dimensions.
        """
        # Check cache
        state_hash = self._compute_hash(pcb_state)
        if state_hash in self._cache:
            return self._cache[state_hash]

        # Get DRC summary
        drc_summary = self._summarize_drc(pcb_state)

        # Fast path for extreme cases
        drc = getattr(pcb_state, '_last_drc', None)
        if drc:
            violations = getattr(drc, 'total_violations', 0)
            if violations == 0:
                result = ValueEstimate(
                    quality_score=0.99,
                    fixability=1.0,
                    manufacturability=0.95,
                    reliability=0.95,
                    confidence=1.0,
                    reasoning="No DRC violations detected."
                )
                self._cache[state_hash] = result
                return result

            if violations > 1000:
                result = ValueEstimate(
                    quality_score=0.01,
                    fixability=0.2,
                    manufacturability=0.1,
                    reliability=0.1,
                    confidence=0.9,
                    reasoning=f"Excessive DRC violations ({violations}). Major redesign needed."
                )
                self._cache[state_hash] = result
                return result

        # Use LLM for nuanced analysis
        prompt = f"""Evaluate the DRC quality of this PCB design:

{drc_summary}

Consider:
1. Critical violations (shorting) are much worse than warnings (silk overlap)
2. Clearance violations can be fixed by routing; shorting requires component moves
3. Thermal issues affect long-term reliability
4. Manufacturing issues affect yield and cost

Rate on 0-1 scale:
- quality_score: Overall design quality
- fixability: How easy to fix remaining issues (0=impossible, 1=trivial)
- manufacturability: Ready for manufacturing (0=unfeasible, 1=ready)
- reliability: Long-term design robustness (0=will fail, 1=robust)
- confidence: Your confidence in this assessment

Provide brief reasoning.

Return JSON:
{{"quality_score": 0.X, "fixability": 0.X, "manufacturability": 0.X, "reliability": 0.X, "confidence": 0.X, "reasoning": "..."}}
"""

        try:
            response = await self.llm_client.generate(prompt, json_mode=True)
            data = json.loads(response)

            result = ValueEstimate(
                quality_score=float(data.get("quality_score", 0.5)),
                fixability=float(data.get("fixability", 0.5)),
                manufacturability=float(data.get("manufacturability", 0.5)),
                reliability=float(data.get("reliability", 0.5)),
                confidence=float(data.get("confidence", 0.5)),
                reasoning=str(data.get("reasoning", ""))
            )

        except Exception as e:
            logger.warning(f"LLM value estimation failed: {e}")
            # Fall back to fast estimate
            fast_score = self.estimate_quality_fast(pcb_state)
            result = ValueEstimate(
                quality_score=fast_score,
                fixability=0.5,
                manufacturability=fast_score,
                reliability=fast_score,
                confidence=0.3,
                reasoning="Fallback heuristic estimate (LLM unavailable)"
            )

        self._cache[state_hash] = result
        return result

    def _summarize_drc(self, pcb_state: Any) -> str:
        """Create DRC summary for LLM analysis."""
        parts = ["DRC Analysis:"]

        drc = None
        if hasattr(pcb_state, 'run_drc'):
            try:
                drc = pcb_state.run_drc()
                pcb_state._last_drc = drc
            except Exception as e:
                parts.append(f"DRC execution failed: {e}")
                return "\n".join(parts)

        if drc is None:
            parts.append("No DRC data available")
            return "\n".join(parts)

        total = getattr(drc, 'total_violations', 0)
        parts.append(f"Total violations: {total}")

        violations_by_type = getattr(drc, 'violations_by_type', {})
        if violations_by_type:
            parts.append("\nViolations by type:")
            for vtype, count in sorted(violations_by_type.items(), key=lambda x: -x[1])[:8]:
                severity = self._get_violation_severity(vtype)
                parts.append(f"  - {vtype}: {count} ({severity})")

        errors = getattr(drc, 'errors', [])
        if errors:
            parts.append(f"\nCritical errors: {len(errors)}")

        warnings = getattr(drc, 'warnings', [])
        if warnings:
            parts.append(f"Warnings: {len(warnings)}")

        return "\n".join(parts)

    def _get_violation_severity(self, vtype: str) -> str:
        """Get human-readable severity for violation type."""
        critical = {'shorting_items', 'track_dangling', 'via_dangling'}
        high = {'clearance', 'track_width', 'annular_ring'}
        medium = {'solder_mask_bridge', 'courtyard_overlap'}

        if vtype in critical:
            return "CRITICAL"
        elif vtype in high:
            return "HIGH"
        elif vtype in medium:
            return "MEDIUM"
        else:
            return "LOW"

    def _compute_hash(self, pcb_state: Any) -> str:
        """Compute hash for caching."""
        if hasattr(pcb_state, 'get_hash'):
            return pcb_state.get_hash()
        return str(id(pcb_state))


@dataclass
class ModificationSuggestion:
    """A suggested PCB modification."""
    mod_type: str
    target: str
    action: str
    rationale: str
    violations_fixed: List[str]
    confidence: float
    priority: int
    parameters: Dict[str, Any]


class LLMPolicyGenerator:
    """
    LLM-based modification policy generator.

    Replaces PolicyNetwork (softmax over actions) with intelligent
    LLM reasoning about what modifications will improve the design.
    """

    MODIFICATION_TYPES = [
        "move_component",
        "rotate_component",
        "adjust_trace_width",
        "move_via",
        "add_via",
        "delete_via",
        "adjust_clearance",
        "adjust_zone",
        "fix_silkscreen",
        "fix_solder_mask",
        "add_thermal_via",
        "no_action"
    ]

    def __init__(
        self,
        config: Optional[GamingAIConfig] = None,
        llm_client: Optional[LLMClient] = None
    ):
        """Initialize policy generator."""
        self.config = config or get_config()
        self.llm_client = llm_client or LLMClient(self.config.llm)

    async def suggest_modifications(
        self,
        pcb_state: Any,
        drc_violations: Optional[Dict[str, int]] = None,
        top_k: int = 5,
        focus_area: Optional[str] = None
    ) -> List[ModificationSuggestion]:
        """
        Generate ranked list of modification suggestions.

        Args:
            pcb_state: Current PCB state. If None, returns empty list.
            drc_violations: Optional violation counts by type
            top_k: Number of suggestions to return
            focus_area: Optional area to focus on

        Returns:
            List of modification suggestions ranked by expected impact.
            Returns empty list if pcb_state is None or invalid.
        """
        # Handle None or invalid input gracefully
        if pcb_state is None:
            logger.warning(
                "suggest_modifications called with None pcb_state. "
                "Cannot generate suggestions without a valid PCB state."
            )
            return []

        # Get violations if not provided
        if drc_violations is None:
            drc_violations = self._get_violations(pcb_state)

        # If no violations, return no_action suggestion
        if not drc_violations or sum(drc_violations.values()) == 0:
            logger.info("No DRC violations found - no modifications needed")
            return [ModificationSuggestion(
                mod_type="no_action",
                target="",
                action="Design has no DRC violations",
                rationale="No modifications needed - design is clean",
                violations_fixed=[],
                confidence=1.0,
                priority=1,
                parameters={}
            )]

        # Build analysis
        analysis = self._analyze_state(pcb_state, drc_violations)

        prompt = f"""You are a PCB design expert. Analyze this design and suggest modifications.

{analysis}

{"Focus on: " + focus_area if focus_area else ""}

Suggest {top_k} modifications to improve this design. For each:

1. **type**: One of: {', '.join(self.MODIFICATION_TYPES)}
2. **target**: What to modify (component ref, net name, or general area)
3. **action**: Specific change (e.g., "move +5mm X", "increase to 0.3mm")
4. **rationale**: Why this helps
5. **violations_fixed**: Which violation types this addresses
6. **confidence**: 0-1 confidence this will help
7. **priority**: 1-10 importance
8. **parameters**: Specific values for the modification

Rank by expected impact (confidence * violations fixed).

Return JSON array:
[
  {{
    "type": "...",
    "target": "...",
    "action": "...",
    "rationale": "...",
    "violations_fixed": ["clearance", "track_width"],
    "confidence": 0.8,
    "priority": 8,
    "parameters": {{"delta_x": 5.0}}
  }}
]
"""

        try:
            response = await self.llm_client.generate(prompt, json_mode=True)
            data = json.loads(response)

            suggestions = []
            for item in data:
                if self._validate_suggestion(item, pcb_state):
                    suggestions.append(ModificationSuggestion(
                        mod_type=item.get("type", "no_action"),
                        target=item.get("target", ""),
                        action=item.get("action", ""),
                        rationale=item.get("rationale", ""),
                        violations_fixed=item.get("violations_fixed", []),
                        confidence=float(item.get("confidence", 0.5)),
                        priority=int(item.get("priority", 5)),
                        parameters=item.get("parameters", {})
                    ))

            # Sort by expected impact
            suggestions.sort(
                key=lambda x: x.confidence * len(x.violations_fixed) * x.priority,
                reverse=True
            )

            return suggestions[:top_k]

        except Exception as e:
            logger.warning(f"LLM policy generation failed: {e}")
            return self._get_fallback_suggestions(pcb_state, drc_violations, top_k)

    def get_action_probabilities(
        self,
        pcb_state: Any,
        suggestions: List[ModificationSuggestion]
    ) -> np.ndarray:
        """
        Convert suggestions to action probability distribution.

        This provides compatibility with code expecting PolicyNetwork output.
        """
        probs = np.zeros(len(self.MODIFICATION_TYPES), dtype=np.float32)

        for sugg in suggestions:
            if sugg.mod_type in self.MODIFICATION_TYPES:
                idx = self.MODIFICATION_TYPES.index(sugg.mod_type)
                probs[idx] += sugg.confidence * sugg.priority

        # Normalize to probability distribution
        if probs.sum() > 0:
            probs = probs / probs.sum()
        else:
            probs[-1] = 1.0  # Default to no_action

        return probs

    def _get_violations(self, pcb_state: Any) -> Dict[str, int]:
        """Get violation counts from PCB state."""
        drc = None
        if hasattr(pcb_state, 'run_drc'):
            try:
                drc = pcb_state.run_drc()
            except Exception:
                pass

        if drc is None:
            return {}

        return getattr(drc, 'violations_by_type', {})

    def _analyze_state(self, pcb_state: Any, violations: Dict[str, int]) -> str:
        """Create state analysis for LLM."""
        parts = ["PCB Design Analysis:"]

        # Violation summary
        total = sum(violations.values())
        parts.append(f"\nTotal violations: {total}")

        if violations:
            parts.append("\nViolations by type (sorted by severity):")
            for vtype, count in sorted(violations.items(), key=lambda x: -x[1])[:10]:
                parts.append(f"  - {vtype}: {count}")

        # Component info
        components = getattr(pcb_state, 'components', [])
        parts.append(f"\nComponents: {len(components)}")

        # Trace info
        traces = getattr(pcb_state, 'traces', [])
        parts.append(f"Traces: {len(traces)}")

        # Via info
        vias = getattr(pcb_state, 'vias', [])
        parts.append(f"Vias: {len(vias)}")

        return "\n".join(parts)

    def _validate_suggestion(self, item: Dict, pcb_state: Any) -> bool:
        """Validate a suggestion is valid for this PCB."""
        mod_type = item.get("type", "")
        if mod_type not in self.MODIFICATION_TYPES:
            return False

        target = item.get("target", "")
        if not target and mod_type != "no_action":
            return False

        return True

    def _get_fallback_suggestions(
        self,
        pcb_state: Any,
        violations: Dict[str, int],
        top_k: int
    ) -> List[ModificationSuggestion]:
        """Generate fallback suggestions without LLM."""
        suggestions = []

        # Suggest based on violation types
        if violations.get('clearance', 0) > 0:
            suggestions.append(ModificationSuggestion(
                mod_type="adjust_clearance",
                target="global",
                action="increase clearance by 0.05mm",
                rationale="Reduce clearance violations",
                violations_fixed=["clearance"],
                confidence=0.6,
                priority=7,
                parameters={"delta": 0.05}
            ))

        if violations.get('track_width', 0) > 0:
            suggestions.append(ModificationSuggestion(
                mod_type="adjust_trace_width",
                target="thin_traces",
                action="increase minimum width to 0.2mm",
                rationale="Fix track width violations",
                violations_fixed=["track_width"],
                confidence=0.5,
                priority=6,
                parameters={"min_width": 0.2}
            ))

        if violations.get('silk_over_copper', 0) > 0:
            suggestions.append(ModificationSuggestion(
                mod_type="fix_silkscreen",
                target="all",
                action="move silkscreen away from copper",
                rationale="Fix silkscreen overlap violations",
                violations_fixed=["silk_over_copper"],
                confidence=0.7,
                priority=5,
                parameters={}
            ))

        if not suggestions:
            suggestions.append(ModificationSuggestion(
                mod_type="no_action",
                target="",
                action="no changes needed",
                rationale="No obvious improvements identified",
                violations_fixed=[],
                confidence=0.3,
                priority=1,
                parameters={}
            ))

        return suggestions[:top_k]


@dataclass
class DynamicsPrediction:
    """Result from dynamics prediction."""
    violations_fixed: List[str]
    violations_created: List[str]
    net_improvement: int
    side_effects: List[str]
    confidence: float
    best_case: str
    worst_case: str


class LLMDynamicsSimulator:
    """
    LLM-based dynamics simulation.

    Replaces DynamicsNetwork (world model) with LLM reasoning
    about the effects of modifications.
    """

    def __init__(
        self,
        config: Optional[GamingAIConfig] = None,
        llm_client: Optional[LLMClient] = None
    ):
        """Initialize dynamics simulator."""
        self.config = config or get_config()
        self.llm_client = llm_client or LLMClient(self.config.llm)

    async def simulate_modification(
        self,
        pcb_state: Any,
        modification: ModificationSuggestion
    ) -> DynamicsPrediction:
        """
        Simulate the effect of a modification without applying it.

        Args:
            pcb_state: Current PCB state. If None, returns fallback prediction.
            modification: Proposed modification

        Returns:
            Prediction of what will happen. Returns fallback prediction
            if pcb_state is None or invalid.
        """
        # Handle None or invalid input gracefully
        if pcb_state is None:
            logger.warning(
                "simulate_modification called with None pcb_state. "
                "Using fallback prediction based on modification alone."
            )
            return self._get_fallback_prediction(modification)

        if modification is None:
            logger.warning("simulate_modification called with None modification.")
            return DynamicsPrediction(
                violations_fixed=[],
                violations_created=[],
                net_improvement=0,
                side_effects=["No modification provided"],
                confidence=0.0,
                best_case="No change",
                worst_case="No change"
            )

        current_drc = self._summarize_drc(pcb_state)

        prompt = f"""PCB design has these violations:
{current_drc}

Proposed modification:
- Type: {modification.mod_type}
- Target: {modification.target}
- Action: {modification.action}
- Expected violations fixed: {modification.violations_fixed}

Predict the outcomes:

1. violations_fixed: List violations that will be fixed
2. violations_created: List new violations that might be created
3. net_improvement: Expected net reduction in violations (negative if worse)
4. side_effects: Unexpected consequences
5. confidence: 0-1 confidence in prediction
6. best_case: Best possible outcome
7. worst_case: Worst possible outcome

Return JSON:
{{"violations_fixed": [...], "violations_created": [...], "net_improvement": N, "side_effects": [...], "confidence": 0.X, "best_case": "...", "worst_case": "..."}}
"""

        try:
            response = await self.llm_client.generate(prompt, json_mode=True)
            data = json.loads(response)

            return DynamicsPrediction(
                violations_fixed=data.get("violations_fixed", []),
                violations_created=data.get("violations_created", []),
                net_improvement=int(data.get("net_improvement", 0)),
                side_effects=data.get("side_effects", []),
                confidence=float(data.get("confidence", 0.5)),
                best_case=data.get("best_case", ""),
                worst_case=data.get("worst_case", "")
            )

        except Exception as e:
            logger.warning(f"LLM dynamics simulation failed: {e}")
            return self._get_fallback_prediction(modification)

    def _summarize_drc(self, pcb_state: Any) -> str:
        """Create DRC summary."""
        parts = []

        drc = None
        if hasattr(pcb_state, 'run_drc'):
            try:
                drc = pcb_state.run_drc()
            except Exception:
                pass

        if drc is None:
            return "No DRC data available"

        violations = getattr(drc, 'violations_by_type', {})
        total = sum(violations.values())
        parts.append(f"Total violations: {total}")

        for vtype, count in sorted(violations.items(), key=lambda x: -x[1])[:5]:
            parts.append(f"  - {vtype}: {count}")

        return "\n".join(parts)

    def _get_fallback_prediction(
        self,
        modification: ModificationSuggestion
    ) -> DynamicsPrediction:
        """Generate fallback prediction without LLM."""
        return DynamicsPrediction(
            violations_fixed=modification.violations_fixed,
            violations_created=[],
            net_improvement=len(modification.violations_fixed),
            side_effects=["Prediction based on heuristics only"],
            confidence=0.3,
            best_case=f"Fix {len(modification.violations_fixed)} violation types",
            worst_case="No improvement or slight regression"
        )


# Convenience function for getting LLM backends
def get_llm_backends(
    config: Optional[GamingAIConfig] = None
) -> Tuple[LLMStateEncoder, LLMValueEstimator, LLMPolicyGenerator, LLMDynamicsSimulator]:
    """
    Get all LLM backend instances with shared configuration.

    Returns:
        Tuple of (encoder, value_estimator, policy_generator, dynamics_simulator)
    """
    config = config or get_config()
    llm_client = LLMClient(config.llm)

    return (
        LLMStateEncoder(config, llm_client),
        LLMValueEstimator(config, llm_client),
        LLMPolicyGenerator(config, llm_client),
        LLMDynamicsSimulator(config, llm_client)
    )
