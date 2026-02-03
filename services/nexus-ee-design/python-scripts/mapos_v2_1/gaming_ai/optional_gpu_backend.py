#!/usr/bin/env python3
"""
Optional GPU Backend - Third-party GPU inference for neural network acceleration.

This module provides optional GPU acceleration via third-party providers:
- RunPod: Dedicated GPU instances
- Modal: Serverless GPU functions
- Replicate: Model inference API
- Together AI: Inference APIs

Falls back to LLM backends if GPU is unavailable or fails.

Author: Adverant Inc.
License: MIT
"""

import os
import json
import logging
import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Any, Optional, List
from enum import Enum

import numpy as np

from .config import GamingAIConfig, InferenceProvider, get_config
from .llm_backends import (
    LLMStateEncoder,
    LLMValueEstimator,
    LLMPolicyGenerator,
    LLMDynamicsSimulator,
    LLMClient,
    StateEmbedding,
    ValueEstimate,
    ModificationSuggestion,
    DynamicsPrediction,
)

logger = logging.getLogger(__name__)


class GPUProviderError(Exception):
    """Error from GPU provider API."""
    pass


class GPUBackend(ABC):
    """Abstract base class for GPU inference backends."""

    @abstractmethod
    async def encode_state(self, state_dict: Dict[str, Any]) -> np.ndarray:
        """Encode PCB state to embedding."""
        pass

    @abstractmethod
    async def estimate_value(self, embedding: np.ndarray) -> float:
        """Estimate quality value from embedding."""
        pass

    @abstractmethod
    async def get_policy(self, embedding: np.ndarray, drc_context: Dict) -> np.ndarray:
        """Get action probabilities from embedding."""
        pass

    @abstractmethod
    async def predict_dynamics(
        self,
        embedding: np.ndarray,
        action: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Predict next state from current state and action."""
        pass

    @abstractmethod
    async def is_available(self) -> bool:
        """Check if this backend is available."""
        pass


class RunPodBackend(GPUBackend):
    """RunPod GPU backend for neural network inference."""

    def __init__(self, endpoint_id: str, api_key: Optional[str] = None):
        """
        Initialize RunPod backend.

        Args:
            endpoint_id: RunPod serverless endpoint ID
            api_key: RunPod API key (from env if not provided)
        """
        self.endpoint_id = endpoint_id
        self.api_key = api_key or os.environ.get("RUNPOD_API_KEY")
        self.base_url = f"https://api.runpod.ai/v2/{endpoint_id}"
        self._client: Optional[Any] = None

    async def _get_client(self):
        """Get HTTP client."""
        if self._client is None:
            try:
                import httpx
                self._client = httpx.AsyncClient(
                    timeout=60.0,
                    headers={"Authorization": f"Bearer {self.api_key}"}
                )
            except ImportError:
                raise GPUProviderError("httpx required for RunPod backend")
        return self._client

    async def _run_job(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Run a job on RunPod endpoint."""
        client = await self._get_client()

        # Start job
        response = await client.post(
            f"{self.base_url}/run",
            json={"input": input_data}
        )
        response.raise_for_status()
        job_data = response.json()
        job_id = job_data.get("id")

        if not job_id:
            raise GPUProviderError("No job ID returned from RunPod")

        # Poll for completion
        max_attempts = 60
        for _ in range(max_attempts):
            status_response = await client.get(f"{self.base_url}/status/{job_id}")
            status_response.raise_for_status()
            status_data = status_response.json()

            status = status_data.get("status")
            if status == "COMPLETED":
                return status_data.get("output", {})
            elif status in ("FAILED", "CANCELLED"):
                error = status_data.get("error", "Unknown error")
                raise GPUProviderError(f"RunPod job failed: {error}")

            await asyncio.sleep(1)

        raise GPUProviderError("RunPod job timed out")

    async def encode_state(self, state_dict: Dict[str, Any]) -> np.ndarray:
        """Encode PCB state using RunPod GPU."""
        result = await self._run_job({
            "action": "encode_state",
            "pcb_state": state_dict
        })
        return np.array(result.get("embedding", []), dtype=np.float32)

    async def estimate_value(self, embedding: np.ndarray) -> float:
        """Estimate value using RunPod GPU."""
        result = await self._run_job({
            "action": "estimate_value",
            "embedding": embedding.tolist()
        })
        return float(result.get("value", 0.5))

    async def get_policy(self, embedding: np.ndarray, drc_context: Dict) -> np.ndarray:
        """Get policy using RunPod GPU."""
        result = await self._run_job({
            "action": "get_policy",
            "embedding": embedding.tolist(),
            "drc_context": drc_context
        })
        return np.array(result.get("policy", []), dtype=np.float32)

    async def predict_dynamics(
        self,
        embedding: np.ndarray,
        action: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Predict dynamics using RunPod GPU."""
        result = await self._run_job({
            "action": "predict_dynamics",
            "embedding": embedding.tolist(),
            "action": action
        })
        return result

    async def is_available(self) -> bool:
        """Check if RunPod endpoint is available."""
        if not self.api_key or not self.endpoint_id:
            return False

        try:
            client = await self._get_client()
            response = await client.get(f"{self.base_url}/health")
            return response.status_code == 200
        except Exception as e:
            logger.debug(f"RunPod availability check failed: {e}")
            return False


class ModalBackend(GPUBackend):
    """Modal serverless GPU backend."""

    def __init__(self, app_name: str = "mapos-inference"):
        """
        Initialize Modal backend.

        Args:
            app_name: Modal app name
        """
        self.app_name = app_name
        self._stub = None

    async def _get_stub(self):
        """Get Modal stub."""
        if self._stub is None:
            try:
                import modal
                self._stub = modal.Stub(self.app_name)
            except ImportError:
                raise GPUProviderError("modal required for Modal backend. Install with: pip install modal")
        return self._stub

    async def encode_state(self, state_dict: Dict[str, Any]) -> np.ndarray:
        """Encode state using Modal GPU."""
        # Note: This is a simplified example. Real implementation would
        # use Modal's function decorators and proper deployment.
        raise NotImplementedError(
            "Modal backend requires deployment. See Modal documentation."
        )

    async def estimate_value(self, embedding: np.ndarray) -> float:
        """Estimate value using Modal GPU."""
        raise NotImplementedError("Modal backend requires deployment")

    async def get_policy(self, embedding: np.ndarray, drc_context: Dict) -> np.ndarray:
        """Get policy using Modal GPU."""
        raise NotImplementedError("Modal backend requires deployment")

    async def predict_dynamics(
        self,
        embedding: np.ndarray,
        action: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Predict dynamics using Modal GPU."""
        raise NotImplementedError("Modal backend requires deployment")

    async def is_available(self) -> bool:
        """Check if Modal is available."""
        try:
            import modal
            return True
        except ImportError:
            return False


class ReplicateBackend(GPUBackend):
    """Replicate model inference backend."""

    def __init__(
        self,
        model_version: str = "adverant/pcb-encoder:latest",
        api_key: Optional[str] = None
    ):
        """
        Initialize Replicate backend.

        Args:
            model_version: Replicate model version string
            api_key: Replicate API key
        """
        self.model_version = model_version
        self.api_key = api_key or os.environ.get("REPLICATE_API_TOKEN")

    async def _run_prediction(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Run prediction on Replicate."""
        try:
            import replicate
        except ImportError:
            raise GPUProviderError("replicate required. Install with: pip install replicate")

        output = await replicate.async_run(
            self.model_version,
            input=input_data
        )
        return output

    async def encode_state(self, state_dict: Dict[str, Any]) -> np.ndarray:
        """Encode state using Replicate."""
        result = await self._run_prediction({
            "action": "encode",
            "pcb_state": json.dumps(state_dict)
        })
        return np.array(result.get("embedding", []), dtype=np.float32)

    async def estimate_value(self, embedding: np.ndarray) -> float:
        """Estimate value using Replicate."""
        result = await self._run_prediction({
            "action": "value",
            "embedding": embedding.tolist()
        })
        return float(result.get("value", 0.5))

    async def get_policy(self, embedding: np.ndarray, drc_context: Dict) -> np.ndarray:
        """Get policy using Replicate."""
        result = await self._run_prediction({
            "action": "policy",
            "embedding": embedding.tolist(),
            "drc_context": drc_context
        })
        return np.array(result.get("policy", []), dtype=np.float32)

    async def predict_dynamics(
        self,
        embedding: np.ndarray,
        action: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Predict dynamics using Replicate."""
        return await self._run_prediction({
            "action": "dynamics",
            "embedding": embedding.tolist(),
            "action_data": action
        })

    async def is_available(self) -> bool:
        """Check if Replicate is available."""
        if not self.api_key:
            return False

        try:
            import replicate
            return True
        except ImportError:
            return False


class TogetherBackend(GPUBackend):
    """Together AI inference backend."""

    def __init__(
        self,
        model_name: str = "adverant/pcb-encoder",
        api_key: Optional[str] = None
    ):
        """
        Initialize Together AI backend.

        Args:
            model_name: Together AI model name
            api_key: Together AI API key
        """
        self.model_name = model_name
        self.api_key = api_key or os.environ.get("TOGETHER_API_KEY")
        self.base_url = "https://api.together.xyz/v1"

    async def encode_state(self, state_dict: Dict[str, Any]) -> np.ndarray:
        """Encode state using Together AI."""
        raise NotImplementedError("Together AI custom model deployment required")

    async def estimate_value(self, embedding: np.ndarray) -> float:
        """Estimate value using Together AI."""
        raise NotImplementedError("Together AI custom model deployment required")

    async def get_policy(self, embedding: np.ndarray, drc_context: Dict) -> np.ndarray:
        """Get policy using Together AI."""
        raise NotImplementedError("Together AI custom model deployment required")

    async def predict_dynamics(
        self,
        embedding: np.ndarray,
        action: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Predict dynamics using Together AI."""
        raise NotImplementedError("Together AI custom model deployment required")

    async def is_available(self) -> bool:
        """Check if Together AI is available."""
        return bool(self.api_key)


class OptionalGPUInference:
    """
    Unified interface for optional GPU inference with LLM fallback.

    Provides transparent switching between GPU and LLM backends.
    """

    def __init__(self, config: Optional[GamingAIConfig] = None):
        """
        Initialize optional GPU inference.

        Args:
            config: Gaming AI configuration
        """
        self.config = config or get_config()
        self._gpu_backend: Optional[GPUBackend] = None
        self._llm_backends = None
        self._gpu_available: Optional[bool] = None

        # Initialize LLM fallbacks
        self._init_llm_fallbacks()

        # Initialize GPU backend if configured
        if self.config.use_neural_networks:
            self._init_gpu_backend()

    def _init_llm_fallbacks(self):
        """Initialize LLM fallback backends."""
        llm_client = LLMClient(self.config.llm)
        self._encoder_fallback = LLMStateEncoder(self.config, llm_client)
        self._value_fallback = LLMValueEstimator(self.config, llm_client)
        self._policy_fallback = LLMPolicyGenerator(self.config, llm_client)
        self._dynamics_fallback = LLMDynamicsSimulator(self.config, llm_client)

    def _init_gpu_backend(self):
        """Initialize GPU backend based on configuration."""
        provider = self.config.neural_network.provider
        endpoint = self.config.neural_network.endpoint_id
        api_key = self.config.neural_network.api_key

        if provider == InferenceProvider.RUNPOD:
            if endpoint:
                self._gpu_backend = RunPodBackend(endpoint, api_key)
            else:
                logger.warning("RunPod configured but no endpoint ID provided")

        elif provider == InferenceProvider.MODAL:
            self._gpu_backend = ModalBackend()

        elif provider == InferenceProvider.REPLICATE:
            self._gpu_backend = ReplicateBackend(api_key=api_key)

        elif provider == InferenceProvider.TOGETHER:
            self._gpu_backend = TogetherBackend(api_key=api_key)

        elif provider == InferenceProvider.LOCAL:
            logger.info("Local PyTorch inference configured - using existing neural networks")

    async def is_gpu_available(self) -> bool:
        """Check if GPU backend is available."""
        if self._gpu_available is not None:
            return self._gpu_available

        if self._gpu_backend is None:
            self._gpu_available = False
            return False

        try:
            self._gpu_available = await self._gpu_backend.is_available()
        except Exception as e:
            logger.warning(f"GPU availability check failed: {e}")
            self._gpu_available = False

        return self._gpu_available

    async def encode_state(self, pcb_state: Any) -> np.ndarray:
        """
        Encode PCB state with GPU acceleration if available.

        Falls back to LLM-based encoding if GPU unavailable.
        """
        # Try GPU first if available
        if self.config.use_neural_networks and await self.is_gpu_available():
            try:
                state_dict = self._state_to_dict(pcb_state)
                return await self._gpu_backend.encode_state(state_dict)
            except Exception as e:
                logger.warning(f"GPU encoding failed, falling back to LLM: {e}")
                if not self.config.neural_network.fallback_to_llm:
                    raise

        # Use LLM fallback
        return self._encoder_fallback.encode_state(pcb_state)

    async def estimate_value(self, pcb_state: Any) -> ValueEstimate:
        """
        Estimate PCB quality with GPU acceleration if available.

        Falls back to LLM-based estimation if GPU unavailable.
        """
        if self.config.use_neural_networks and await self.is_gpu_available():
            try:
                embedding = await self.encode_state(pcb_state)
                value = await self._gpu_backend.estimate_value(embedding)

                return ValueEstimate(
                    quality_score=value,
                    fixability=0.5,  # GPU doesn't provide these
                    manufacturability=0.5,
                    reliability=0.5,
                    confidence=0.8,
                    reasoning="GPU-based neural network estimate"
                )
            except Exception as e:
                logger.warning(f"GPU value estimation failed: {e}")
                if not self.config.neural_network.fallback_to_llm:
                    raise

        # Use LLM fallback
        return await self._value_fallback.estimate_quality(pcb_state)

    async def suggest_modifications(
        self,
        pcb_state: Any,
        drc_violations: Optional[Dict[str, int]] = None,
        top_k: int = 5
    ) -> List[ModificationSuggestion]:
        """
        Get modification suggestions with GPU acceleration if available.

        Falls back to LLM-based suggestions if GPU unavailable.
        """
        if self.config.use_neural_networks and await self.is_gpu_available():
            try:
                embedding = await self.encode_state(pcb_state)
                drc_context = drc_violations or {}
                policy = await self._gpu_backend.get_policy(embedding, drc_context)

                # Convert policy to suggestions
                return self._policy_to_suggestions(policy, pcb_state, top_k)
            except Exception as e:
                logger.warning(f"GPU policy generation failed: {e}")
                if not self.config.neural_network.fallback_to_llm:
                    raise

        # Use LLM fallback
        return await self._policy_fallback.suggest_modifications(
            pcb_state, drc_violations, top_k
        )

    async def predict_modification_outcome(
        self,
        pcb_state: Any,
        modification: ModificationSuggestion
    ) -> DynamicsPrediction:
        """
        Predict modification outcome with GPU acceleration if available.

        Falls back to LLM-based prediction if GPU unavailable.
        """
        if self.config.use_neural_networks and await self.is_gpu_available():
            try:
                embedding = await self.encode_state(pcb_state)
                action = {
                    "type": modification.mod_type,
                    "target": modification.target,
                    "parameters": modification.parameters
                }
                result = await self._gpu_backend.predict_dynamics(embedding, action)

                return DynamicsPrediction(
                    violations_fixed=result.get("violations_fixed", []),
                    violations_created=result.get("violations_created", []),
                    net_improvement=result.get("net_improvement", 0),
                    side_effects=result.get("side_effects", []),
                    confidence=result.get("confidence", 0.5),
                    best_case=result.get("best_case", ""),
                    worst_case=result.get("worst_case", "")
                )
            except Exception as e:
                logger.warning(f"GPU dynamics prediction failed: {e}")
                if not self.config.neural_network.fallback_to_llm:
                    raise

        # Use LLM fallback
        return await self._dynamics_fallback.simulate_modification(pcb_state, modification)

    def _state_to_dict(self, pcb_state: Any) -> Dict[str, Any]:
        """Convert PCB state to dictionary for API calls."""
        if hasattr(pcb_state, 'to_dict'):
            return pcb_state.to_dict()

        # Manual conversion
        result = {}

        for attr in ['components', 'traces', 'vias', 'zones', 'nets']:
            obj = getattr(pcb_state, attr, None)
            if obj is not None:
                if hasattr(obj, '__len__'):
                    result[attr] = [
                        item.to_dict() if hasattr(item, 'to_dict') else str(item)
                        for item in obj
                    ]
                else:
                    result[attr] = str(obj)

        return result

    def _policy_to_suggestions(
        self,
        policy: np.ndarray,
        pcb_state: Any,
        top_k: int
    ) -> List[ModificationSuggestion]:
        """Convert policy array to modification suggestions."""
        from .llm_backends import LLMPolicyGenerator

        mod_types = LLMPolicyGenerator.MODIFICATION_TYPES
        suggestions = []

        # Get top-k actions by probability
        top_indices = np.argsort(policy)[-top_k:][::-1]

        for idx in top_indices:
            if idx < len(mod_types):
                mod_type = mod_types[idx]
                prob = float(policy[idx])

                suggestions.append(ModificationSuggestion(
                    mod_type=mod_type,
                    target="auto",
                    action=f"Apply {mod_type} modification",
                    rationale="GPU-based policy suggestion",
                    violations_fixed=[],
                    confidence=prob,
                    priority=int(prob * 10),
                    parameters={}
                ))

        return suggestions


def get_inference_backend(
    config: Optional[GamingAIConfig] = None
) -> OptionalGPUInference:
    """
    Get the unified inference backend.

    This is the main entry point for inference operations.
    Handles GPU/LLM switching transparently.

    Args:
        config: Optional configuration

    Returns:
        OptionalGPUInference instance
    """
    return OptionalGPUInference(config)
