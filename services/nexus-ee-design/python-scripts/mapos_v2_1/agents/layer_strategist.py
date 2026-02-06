"""
MAPO v2.0 Layer Assignment Strategist Agent

LLM agent that provides strategic guidance for layer assignments.
Uses Claude Opus 4.6 via OpenRouter for semantic reasoning about:
- Net-to-layer mapping
- Via minimization strategies
- Signal integrity layer considerations
- Power/ground plane utilization

Author: Claude Opus 4.6 via MAPO v2.0
"""

import asyncio
import json
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from enum import Enum
import os

try:
    import httpx
except ImportError:
    httpx = None

logger = logging.getLogger(__name__)


# =============================================================================
# Data Structures
# =============================================================================

class LayerPreference(Enum):
    """Layer preference types."""
    INNER = "inner"       # Prefer inner layers
    OUTER = "outer"       # Prefer outer layers
    ADJACENT_GROUND = "adjacent_ground"  # Adjacent to ground plane
    ADJACENT_POWER = "adjacent_power"    # Adjacent to power plane
    ANY = "any"           # No preference


@dataclass
class NetLayerHint:
    """Layer assignment hint for a single net."""
    net_name: str
    preferred_layers: List[int]
    avoid_layers: List[int]
    preference_type: LayerPreference
    reasoning: str
    confidence: float = 0.8


@dataclass
class LayerGrouping:
    """Grouping of nets that should share layers."""
    group_name: str
    net_names: List[str]
    recommended_layer: int
    reasoning: str


@dataclass
class LayerHintsResult:
    """Result from layer strategist analysis."""
    net_hints: List[NetLayerHint]
    groupings: List[LayerGrouping]
    power_layer_recommendation: Optional[int]
    ground_layer_recommendation: Optional[int]
    overall_strategy: str
    confidence: float = 0.8


# =============================================================================
# Layer Assignment Strategist Agent
# =============================================================================

class LayerAssignmentStrategistAgent:
    """
    Opus 4.6 agent for layer assignment strategy.

    Analyzes net characteristics and provides strategic hints for:
    - Which layers each net should use
    - How to group related nets
    - Via minimization strategies
    - Power/ground plane utilization
    """

    SYSTEM_PROMPT = """You are an expert PCB layer assignment strategist. Given a set of nets and a layer stackup, you provide strategic guidance for optimal layer assignments.

Your analysis considers:
1. **Signal Integrity**: High-speed signals adjacent to reference planes
2. **Via Minimization**: Group related nets on same layer when possible
3. **Net Types**: Power/ground on planes, signals on routing layers
4. **Layer Balance**: Distribute routing evenly across layers
5. **Manufacturing**: Minimize layer count when possible

For each net, provide:
- Preferred layers (ordered by preference)
- Layers to avoid (with reasons)
- Confidence level (0.0-1.0)

Also identify net groupings that should share layers to minimize vias.

Output as structured JSON with:
- net_hints: Array of hints per net
- groupings: Array of net groups
- power_layer_recommendation: Layer index for power
- ground_layer_recommendation: Layer index for ground
- overall_strategy: Text description of approach
- confidence: Overall confidence score"""

    def __init__(
        self,
        openrouter_api_key: Optional[str] = None,
        model: str = "anthropic/claude-opus-4-5-20251101",
        timeout: float = 60.0
    ):
        self.api_key = openrouter_api_key or os.environ.get("OPENROUTER_API_KEY")
        self.model = model
        self.timeout = timeout
        self.api_url = "https://openrouter.ai/api/v1/chat/completions"

    async def get_layer_hints(
        self,
        nets: List[Dict[str, Any]],
        layers: List[Dict[str, Any]],
        stackup_info: Optional[Dict[str, Any]] = None
    ) -> LayerHintsResult:
        """
        Get layer assignment hints from LLM.

        Args:
            nets: List of net info dicts with name, type, length, priority
            layers: List of layer info dicts with index, name, type
            stackup_info: Optional additional stackup information

        Returns:
            LayerHintsResult with hints for each net and groupings
        """
        if not self.api_key:
            logger.warning("No OpenRouter API key, returning default hints")
            return self._default_hints(nets, layers)

        if httpx is None:
            logger.warning("httpx not installed, returning default hints")
            return self._default_hints(nets, layers)

        prompt = self._build_prompt(nets, layers, stackup_info)

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(
                    self.api_url,
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json",
                        "HTTP-Referer": "https://github.com/adverant/nexus-ee-design",
                        "X-Title": "MAPO v2.0 Layer Strategist"
                    },
                    json={
                        "model": self.model,
                        "messages": [
                            {"role": "system", "content": self.SYSTEM_PROMPT},
                            {"role": "user", "content": prompt}
                        ],
                        "temperature": 0.3,
                        "max_tokens": 4000
                    }
                )

                if response.status_code != 200:
                    logger.error(f"OpenRouter API error: {response.status_code}")
                    return self._default_hints(nets, layers)

                result = response.json()
                content = result["choices"][0]["message"]["content"]

                return self._parse_response(content, nets, layers)

        except Exception as e:
            logger.error(f"Layer strategist API call failed: {e}")
            return self._default_hints(nets, layers)

    def _build_prompt(
        self,
        nets: List[Dict[str, Any]],
        layers: List[Dict[str, Any]],
        stackup_info: Optional[Dict[str, Any]]
    ) -> str:
        """Build the analysis prompt."""
        prompt = f"""Analyze the following PCB design and provide layer assignment recommendations.

## Layer Stackup

{json.dumps(layers, indent=2)}

## Nets to Route

{json.dumps(nets, indent=2)}

"""
        if stackup_info:
            prompt += f"""## Additional Stackup Info

{json.dumps(stackup_info, indent=2)}

"""

        prompt += """## Task

Provide layer assignment strategy as JSON:

```json
{
  "net_hints": [
    {
      "net_name": "string",
      "preferred_layers": [0, 1],
      "avoid_layers": [2],
      "preference_type": "adjacent_ground",
      "reasoning": "High-speed signal needs ground reference",
      "confidence": 0.9
    }
  ],
  "groupings": [
    {
      "group_name": "USB signals",
      "net_names": ["USB_D+", "USB_D-"],
      "recommended_layer": 0,
      "reasoning": "Differential pair should stay on same layer"
    }
  ],
  "power_layer_recommendation": 2,
  "ground_layer_recommendation": 1,
  "overall_strategy": "Description of approach...",
  "confidence": 0.85
}
```

Focus on:
1. Minimizing total vias
2. Keeping differential pairs on same layer
3. High-speed signals adjacent to ground
4. Balanced layer utilization"""

        return prompt

    def _parse_response(
        self,
        content: str,
        nets: List[Dict[str, Any]],
        layers: List[Dict[str, Any]]
    ) -> LayerHintsResult:
        """Parse LLM response into structured result."""
        try:
            # Extract JSON from response
            import re
            json_match = re.search(r'```json\s*(.*?)\s*```', content, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
            else:
                # Try to find raw JSON
                json_start = content.find('{')
                json_end = content.rfind('}') + 1
                if json_start >= 0 and json_end > json_start:
                    json_str = content[json_start:json_end]
                else:
                    return self._default_hints(nets, layers)

            data = json.loads(json_str)

            # Parse net hints
            net_hints = []
            for hint_data in data.get("net_hints", []):
                try:
                    pref_type = LayerPreference(hint_data.get("preference_type", "any"))
                except ValueError:
                    pref_type = LayerPreference.ANY

                net_hints.append(NetLayerHint(
                    net_name=hint_data.get("net_name", ""),
                    preferred_layers=hint_data.get("preferred_layers", []),
                    avoid_layers=hint_data.get("avoid_layers", []),
                    preference_type=pref_type,
                    reasoning=hint_data.get("reasoning", ""),
                    confidence=hint_data.get("confidence", 0.8)
                ))

            # Parse groupings
            groupings = []
            for group_data in data.get("groupings", []):
                groupings.append(LayerGrouping(
                    group_name=group_data.get("group_name", ""),
                    net_names=group_data.get("net_names", []),
                    recommended_layer=group_data.get("recommended_layer", 0),
                    reasoning=group_data.get("reasoning", "")
                ))

            return LayerHintsResult(
                net_hints=net_hints,
                groupings=groupings,
                power_layer_recommendation=data.get("power_layer_recommendation"),
                ground_layer_recommendation=data.get("ground_layer_recommendation"),
                overall_strategy=data.get("overall_strategy", ""),
                confidence=data.get("confidence", 0.8)
            )

        except Exception as e:
            logger.error(f"Failed to parse layer strategist response: {e}")
            return self._default_hints(nets, layers)

    def _default_hints(
        self,
        nets: List[Dict[str, Any]],
        layers: List[Dict[str, Any]]
    ) -> LayerHintsResult:
        """Generate default hints without LLM."""
        signal_layers = [l["index"] for l in layers if l.get("type") in ["signal", "mixed"]]
        ground_layers = [l["index"] for l in layers if l.get("type") == "ground"]
        power_layers = [l["index"] for l in layers if l.get("type") == "power"]

        net_hints = []
        for net in nets:
            net_type = net.get("type", "digital")

            if net_type in ["high_speed", "differential", "clock"]:
                # High-speed: prefer layers adjacent to ground
                preferred = []
                for gl in ground_layers:
                    for sl in signal_layers:
                        if abs(gl - sl) == 1:
                            preferred.append(sl)
                preferred = list(set(preferred)) or signal_layers[:1]

                net_hints.append(NetLayerHint(
                    net_name=net.get("name", ""),
                    preferred_layers=preferred,
                    avoid_layers=[],
                    preference_type=LayerPreference.ADJACENT_GROUND,
                    reasoning="High-speed signal needs ground reference plane",
                    confidence=0.7
                ))

            elif net_type == "power":
                net_hints.append(NetLayerHint(
                    net_name=net.get("name", ""),
                    preferred_layers=power_layers or signal_layers,
                    avoid_layers=[],
                    preference_type=LayerPreference.ANY,
                    reasoning="Power net prefers power plane or wide traces",
                    confidence=0.6
                ))

            else:
                # Default: any signal layer
                net_hints.append(NetLayerHint(
                    net_name=net.get("name", ""),
                    preferred_layers=signal_layers,
                    avoid_layers=[],
                    preference_type=LayerPreference.ANY,
                    reasoning="Standard signal routing",
                    confidence=0.5
                ))

        return LayerHintsResult(
            net_hints=net_hints,
            groupings=[],
            power_layer_recommendation=power_layers[0] if power_layers else None,
            ground_layer_recommendation=ground_layers[0] if ground_layers else None,
            overall_strategy="Default layer assignment: high-speed adjacent to ground, balanced distribution",
            confidence=0.5
        )


# =============================================================================
# Factory Function
# =============================================================================

def create_layer_strategist(
    openrouter_api_key: Optional[str] = None,
    model: str = "anthropic/claude-opus-4-5-20251101"
) -> LayerAssignmentStrategistAgent:
    """Factory function to create a layer strategist agent."""
    return LayerAssignmentStrategistAgent(
        openrouter_api_key=openrouter_api_key,
        model=model
    )


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    "LayerAssignmentStrategistAgent",
    "LayerHintsResult",
    "NetLayerHint",
    "LayerGrouping",
    "LayerPreference",
    "create_layer_strategist"
]
