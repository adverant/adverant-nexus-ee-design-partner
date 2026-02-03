#!/usr/bin/env python3
"""
Conflict Resolver Agent - LLM-driven routing conflict resolution.

This agent uses Claude Opus 4.5 via OpenRouter to resolve routing conflicts
when CBS (Conflict-Based Search) identifies nets competing for the same
routing resources.

Part of the MAPO v2.0 Enhancement: "Opus 4.5 Thinks, Algorithms Execute"
"""

import json
import os
import sys
import asyncio
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any, Tuple, Set
from enum import Enum, auto
from pathlib import Path

# Add parent directory to path for local imports
SCRIPT_DIR = Path(__file__).parent.parent.resolve()
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

try:
    from pcb_state import PCBState, DRCResult
    from generator_agents import GeneratorAgent, AgentConfig, AgentRole, GenerationResult
except ImportError:
    pass


class ConflictType(Enum):
    """Types of routing conflicts."""
    SPATIAL = auto()         # Two traces want same physical space
    CLEARANCE = auto()       # Traces too close together
    LAYER_CONTENTION = auto()  # Multiple nets want same layer segment
    VIA_PLACEMENT = auto()   # Vias competing for same location
    CROSSING = auto()        # Nets must cross, need resolution
    RESOURCE = auto()        # General resource contention


class ResolutionStrategy(Enum):
    """Strategies for resolving conflicts."""
    RIPUP_REROUTE_A = auto()    # Rip-up net A, reroute
    RIPUP_REROUTE_B = auto()    # Rip-up net B, reroute
    LAYER_CHANGE_A = auto()     # Move net A to different layer
    LAYER_CHANGE_B = auto()     # Move net B to different layer
    DETOUR_A = auto()           # Detour net A around conflict
    DETOUR_B = auto()           # Detour net B around conflict
    WIDTH_ADJUST = auto()       # Adjust trace widths
    SPLIT_ROUTE = auto()        # Split one net into multiple segments
    DEFER = auto()              # Defer to human/higher authority


@dataclass
class RoutingConflict:
    """A routing conflict between nets or resources."""
    conflict_id: str
    conflict_type: ConflictType
    net_a: str
    net_b: Optional[str]  # None for single-net conflicts (e.g., DRC)
    location: Tuple[float, float]  # (x, y) of conflict
    severity: str  # LOW, MEDIUM, HIGH, CRITICAL
    resources: List[Tuple[float, float]]  # Contested grid points
    description: str
    current_routes: Dict[str, List[Tuple[float, float]]] = field(default_factory=dict)


@dataclass
class ConflictResolution:
    """A resolution for a routing conflict."""
    conflict_id: str
    strategy: ResolutionStrategy
    affected_nets: List[str]
    actions: List[Dict[str, Any]]  # Specific actions to take
    estimated_cost: float  # Via count increase, length increase, etc.
    trade_offs: Dict[str, Any]
    reasoning: str
    confidence: float


@dataclass
class ResolutionPlan:
    """Complete plan for resolving multiple conflicts."""
    resolutions: List[ConflictResolution]
    total_estimated_cost: float
    success_probability: float
    warnings: List[str]
    reasoning: str


class ConflictResolverAgent(GeneratorAgent):
    """
    Opus 4.5 agent specialized in routing conflict resolution.

    When CBS (Conflict-Based Search) identifies conflicts:
    1. Analyzes conflict severity and type
    2. Proposes multiple resolution strategies
    3. Evaluates trade-offs (via count vs. length vs. layers)
    4. Recommends optimal resolution

    Integrates with existing agents via debate mechanism.
    """

    SYSTEM_PROMPT = """You are a routing conflict resolution specialist for MAPO v2.0.
When multiple nets conflict for the same routing resources, you determine the
best resolution strategy while minimizing overall design impact.

CONFLICT TYPES:
- SPATIAL: Two traces want same physical space
- CLEARANCE: Traces too close (violates design rules)
- LAYER_CONTENTION: Multiple nets want same layer segment
- VIA_PLACEMENT: Vias competing for same location
- CROSSING: Nets must cross each other
- RESOURCE: General resource contention

RESOLUTION STRATEGIES:
1. RIPUP_REROUTE_A/B: Remove one net's route and reroute it
   - Cost: Routing time, potential length increase
   - Use when: One net has more routing flexibility

2. LAYER_CHANGE_A/B: Move one net to a different layer
   - Cost: Via addition (signal integrity impact)
   - Use when: Layer has available capacity

3. DETOUR_A/B: Route around the conflict zone
   - Cost: Increased trace length
   - Use when: Space available for detour

4. WIDTH_ADJUST: Reduce trace width if clearance issue
   - Cost: Impedance change, current capacity
   - Use when: Width reduction is acceptable

5. SPLIT_ROUTE: Break net into multiple layer segments
   - Cost: Multiple vias
   - Use when: No single-layer path exists

DECISION FACTORS:
- Net priority (power > clock > high-speed > general)
- Current via count for each net
- Layer utilization
- Signal integrity requirements
- Manufacturing constraints

OUTPUT FORMAT: Return JSON:
{
  "resolutions": [
    {
      "conflict_id": "conflict_001",
      "strategy": "LAYER_CHANGE_A",
      "affected_nets": ["DATA0"],
      "actions": [
        {
          "action": "add_via",
          "net": "DATA0",
          "location": [75.0, 50.0],
          "from_layer": "F.Cu",
          "to_layer": "In1.Cu"
        },
        {
          "action": "reroute_segment",
          "net": "DATA0",
          "from": [75.0, 50.0],
          "to": [90.0, 50.0],
          "layer": "In1.Cu"
        }
      ],
      "estimated_cost": {
        "via_count": 2,
        "length_increase_mm": 5.0,
        "layer_transitions": 2
      },
      "trade_offs": {
        "pros": ["Avoids congested area", "No impact on net B"],
        "cons": ["Adds 2 vias", "Slight length increase"]
      },
      "reasoning": "DATA0 is lower priority, layer 2 has capacity",
      "confidence": 0.85
    }
  ],
  "total_estimated_cost": 2.5,
  "success_probability": 0.90,
  "warnings": ["Layer 2 approaching 60% utilization"],
  "reasoning": "Overall resolution plan explanation"
}"""

    def __init__(self):
        """Initialize the Conflict Resolver Agent."""
        config = AgentConfig(
            name="ConflictResolver",
            role=AgentRole.GENERAL,
            model="anthropic/claude-opus-4-5-20251101",
            temperature=0.3,
            max_tokens=8192,
            focus_areas=[
                "conflict_resolution", "rip_up_reroute", "layer_assignment",
                "trade_off_analysis", "routing_optimization"
            ],
            constraints={
                "max_via_penalty": 10,
                "max_length_penalty_mm": 50,
            }
        )
        super().__init__(config)

    def get_system_prompt(self) -> str:
        """Get the system prompt for conflict resolution."""
        return self.SYSTEM_PROMPT

    def get_focus_violations(self, drc_result: DRCResult) -> List[Dict]:
        """Not used for conflict resolution - returns empty list."""
        return []

    async def resolve_conflict(
        self,
        conflict: RoutingConflict,
        current_solution: Dict[str, Any],
        constraints: Optional[Dict[str, Any]] = None
    ) -> ConflictResolution:
        """
        Resolve a single routing conflict.

        Args:
            conflict: The routing conflict to resolve
            current_solution: Current routing solution state
            constraints: Optional routing constraints

        Returns:
            ConflictResolution with recommended actions
        """
        prompt = self._build_resolution_prompt(conflict, current_solution, constraints)
        response = await self._call_openrouter(prompt)
        resolution = self._parse_resolution_response(response, conflict)
        return resolution

    async def resolve_conflicts(
        self,
        conflicts: List[RoutingConflict],
        current_solution: Dict[str, Any],
        constraints: Optional[Dict[str, Any]] = None
    ) -> ResolutionPlan:
        """
        Resolve multiple routing conflicts.

        Args:
            conflicts: List of routing conflicts
            current_solution: Current routing solution state
            constraints: Optional routing constraints

        Returns:
            ResolutionPlan with all resolutions and overall analysis
        """
        if not conflicts:
            return ResolutionPlan(
                resolutions=[],
                total_estimated_cost=0,
                success_probability=1.0,
                warnings=[],
                reasoning="No conflicts to resolve"
            )

        # For complex cases with many conflicts, send all at once
        if len(conflicts) > 5:
            prompt = self._build_multi_conflict_prompt(conflicts, current_solution, constraints)
            response = await self._call_openrouter(prompt)
            return self._parse_plan_response(response, conflicts)

        # For simpler cases, resolve one by one
        resolutions = []
        total_cost = 0

        for conflict in conflicts:
            resolution = await self.resolve_conflict(conflict, current_solution, constraints)
            resolutions.append(resolution)
            total_cost += resolution.estimated_cost

        # Calculate overall success probability
        confidences = [r.confidence for r in resolutions]
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0.5

        return ResolutionPlan(
            resolutions=resolutions,
            total_estimated_cost=total_cost,
            success_probability=avg_confidence,
            warnings=[],
            reasoning=f"Resolved {len(conflicts)} conflicts individually"
        )

    def _build_resolution_prompt(
        self,
        conflict: RoutingConflict,
        current_solution: Dict[str, Any],
        constraints: Optional[Dict] = None
    ) -> str:
        """Build prompt for single conflict resolution."""
        prompt = f"""Resolve this routing conflict:

CONFLICT DETAILS:
  ID: {conflict.conflict_id}
  Type: {conflict.conflict_type.name}
  Net A: {conflict.net_a}
  Net B: {conflict.net_b or 'N/A (single-net conflict)'}
  Location: ({conflict.location[0]:.2f}, {conflict.location[1]:.2f})
  Severity: {conflict.severity}
  Description: {conflict.description}

CONTESTED RESOURCES:
{json.dumps(conflict.resources[:20], indent=2)}

CURRENT ROUTES:
{json.dumps(conflict.current_routes, indent=2)}

CONSTRAINTS:
{json.dumps(constraints or {}, indent=2)}

Provide a resolution with:
1. Recommended strategy
2. Specific actions to take
3. Estimated cost (vias, length)
4. Trade-off analysis
5. Confidence in the resolution
"""
        return prompt

    def _build_multi_conflict_prompt(
        self,
        conflicts: List[RoutingConflict],
        current_solution: Dict[str, Any],
        constraints: Optional[Dict] = None
    ) -> str:
        """Build prompt for multiple conflict resolution."""
        conflict_summaries = []
        for c in conflicts:
            conflict_summaries.append({
                'id': c.conflict_id,
                'type': c.conflict_type.name,
                'nets': [c.net_a, c.net_b] if c.net_b else [c.net_a],
                'location': c.location,
                'severity': c.severity,
                'resource_count': len(c.resources)
            })

        prompt = f"""Resolve these {len(conflicts)} routing conflicts as a coordinated plan:

CONFLICTS:
{json.dumps(conflict_summaries, indent=2)}

CONSTRAINTS:
{json.dumps(constraints or {}, indent=2)}

Provide a coordinated resolution plan that:
1. Minimizes total via count
2. Avoids creating new conflicts
3. Considers inter-dependencies between conflicts
4. Prioritizes high-severity conflicts
5. Maintains routing efficiency

Return resolutions for ALL conflicts in a single plan.
"""
        return prompt

    async def _call_openrouter(self, prompt: str) -> str:
        """Call Claude Opus 4.5 via OpenRouter."""
        api_key = os.environ.get('OPENROUTER_API_KEY')

        if not api_key:
            return self._generate_heuristic_response()

        try:
            import httpx

            async with httpx.AsyncClient(timeout=60.0) as client:
                response = await client.post(
                    "https://openrouter.ai/api/v1/chat/completions",
                    headers={
                        "Authorization": f"Bearer {api_key}",
                        "Content-Type": "application/json",
                        "HTTP-Referer": "https://adverant.ai",
                        "X-Title": "MAPO Conflict Resolver"
                    },
                    json={
                        "model": "anthropic/claude-opus-4-5-20251101",
                        "temperature": 0.3,
                        "max_tokens": 8192,
                        "messages": [
                            {"role": "system", "content": self.SYSTEM_PROMPT},
                            {"role": "user", "content": prompt}
                        ]
                    }
                )

                if response.status_code == 200:
                    data = response.json()
                    return data['choices'][0]['message']['content']
                else:
                    print(f"OpenRouter API error: {response.status_code}")
                    return self._generate_heuristic_response()

        except Exception as e:
            print(f"OpenRouter call failed: {e}")
            return self._generate_heuristic_response()

    def _generate_heuristic_response(self) -> str:
        """Generate heuristic response when LLM unavailable."""
        return json.dumps({
            "resolutions": [{
                "conflict_id": "unknown",
                "strategy": "DETOUR_A",
                "affected_nets": [],
                "actions": [],
                "estimated_cost": {"via_count": 0, "length_increase_mm": 10},
                "trade_offs": {"pros": ["Simple"], "cons": ["May not work"]},
                "reasoning": "Heuristic: detour around conflict",
                "confidence": 0.5
            }],
            "total_estimated_cost": 1.0,
            "success_probability": 0.6,
            "warnings": ["Using heuristic resolution - LLM unavailable"],
            "reasoning": "Heuristic-based resolution"
        })

    def _parse_resolution_response(
        self,
        response: str,
        conflict: RoutingConflict
    ) -> ConflictResolution:
        """Parse LLM response into ConflictResolution."""
        try:
            import re
            json_match = re.search(r'\{[\s\S]*\}', response)
            if json_match:
                data = json.loads(json_match.group())
            else:
                data = json.loads(self._generate_heuristic_response())

            # Get first resolution (for single conflict)
            resolutions = data.get('resolutions', [])
            if resolutions:
                r = resolutions[0]
            else:
                r = data

            # Parse strategy
            strategy_str = r.get('strategy', 'DETOUR_A')
            try:
                strategy = ResolutionStrategy[strategy_str]
            except KeyError:
                strategy = ResolutionStrategy.DETOUR_A

            # Parse estimated cost
            cost_data = r.get('estimated_cost', {})
            if isinstance(cost_data, dict):
                estimated_cost = (
                    cost_data.get('via_count', 0) * 1.0 +
                    cost_data.get('length_increase_mm', 0) * 0.1 +
                    cost_data.get('layer_transitions', 0) * 0.5
                )
            else:
                estimated_cost = float(cost_data) if cost_data else 1.0

            return ConflictResolution(
                conflict_id=r.get('conflict_id', conflict.conflict_id),
                strategy=strategy,
                affected_nets=r.get('affected_nets', [conflict.net_a]),
                actions=r.get('actions', []),
                estimated_cost=estimated_cost,
                trade_offs=r.get('trade_offs', {}),
                reasoning=r.get('reasoning', ''),
                confidence=r.get('confidence', 0.7)
            )

        except json.JSONDecodeError as e:
            print(f"JSON parse error in resolution response: {e}")
            return ConflictResolution(
                conflict_id=conflict.conflict_id,
                strategy=ResolutionStrategy.DEFER,
                affected_nets=[conflict.net_a],
                actions=[],
                estimated_cost=0,
                trade_offs={},
                reasoning="Parse error - deferring",
                confidence=0.3
            )

    def _parse_plan_response(
        self,
        response: str,
        conflicts: List[RoutingConflict]
    ) -> ResolutionPlan:
        """Parse LLM response into ResolutionPlan."""
        try:
            import re
            json_match = re.search(r'\{[\s\S]*\}', response)
            if json_match:
                data = json.loads(json_match.group())
            else:
                data = json.loads(self._generate_heuristic_response())

            resolutions = []
            for r in data.get('resolutions', []):
                strategy_str = r.get('strategy', 'DETOUR_A')
                try:
                    strategy = ResolutionStrategy[strategy_str]
                except KeyError:
                    strategy = ResolutionStrategy.DETOUR_A

                cost_data = r.get('estimated_cost', {})
                if isinstance(cost_data, dict):
                    estimated_cost = (
                        cost_data.get('via_count', 0) * 1.0 +
                        cost_data.get('length_increase_mm', 0) * 0.1
                    )
                else:
                    estimated_cost = float(cost_data) if cost_data else 1.0

                resolutions.append(ConflictResolution(
                    conflict_id=r.get('conflict_id', ''),
                    strategy=strategy,
                    affected_nets=r.get('affected_nets', []),
                    actions=r.get('actions', []),
                    estimated_cost=estimated_cost,
                    trade_offs=r.get('trade_offs', {}),
                    reasoning=r.get('reasoning', ''),
                    confidence=r.get('confidence', 0.7)
                ))

            return ResolutionPlan(
                resolutions=resolutions,
                total_estimated_cost=data.get('total_estimated_cost', 0),
                success_probability=data.get('success_probability', 0.7),
                warnings=data.get('warnings', []),
                reasoning=data.get('reasoning', '')
            )

        except json.JSONDecodeError as e:
            print(f"JSON parse error in plan response: {e}")
            return ResolutionPlan(
                resolutions=[],
                total_estimated_cost=0,
                success_probability=0.5,
                warnings=["Parse error"],
                reasoning="Parse error - empty plan"
            )

    async def critique(self, proposal: Any) -> Dict[str, Any]:
        """Critique a resolution proposal from another agent."""
        prompt = f"""Review this conflict resolution proposal:

PROPOSAL:
{json.dumps(proposal if isinstance(proposal, dict) else str(proposal), indent=2)}

Analyze for:
1. Feasibility of the resolution
2. Impact on other nets
3. Layer utilization concerns
4. Signal integrity issues
5. Missing considerations

Return JSON:
{{
  "issues": ["issue1", "issue2"],
  "severity": "LOW|MEDIUM|HIGH|CRITICAL",
  "suggestions": ["suggestion1", "suggestion2"],
  "approval_vote": true/false
}}"""

        response = await self._call_openrouter(prompt)

        try:
            import re
            json_match = re.search(r'\{[\s\S]*\}', response)
            if json_match:
                return json.loads(json_match.group())
        except:
            pass

        return {
            "issues": [],
            "severity": "LOW",
            "suggestions": [],
            "approval_vote": True
        }

    async def vote(self, proposal: Any, response: Any) -> bool:
        """Vote on whether to accept a proposal after debate."""
        return True


# Convenience function
def create_conflict_resolver() -> ConflictResolverAgent:
    """Create a new Conflict Resolver Agent instance."""
    return ConflictResolverAgent()


# Main entry point for testing
if __name__ == '__main__':
    async def test_resolver():
        """Test the conflict resolver."""
        print("\n" + "="*60)
        print("CONFLICT RESOLVER AGENT - Test")
        print("="*60)

        agent = create_conflict_resolver()

        # Sample conflict
        conflict = RoutingConflict(
            conflict_id="conflict_001",
            conflict_type=ConflictType.SPATIAL,
            net_a="DATA0",
            net_b="CLK",
            location=(75.0, 50.0),
            severity="HIGH",
            resources=[(75.0, 50.0), (77.54, 50.0), (80.08, 50.0)],
            description="DATA0 and CLK routes cross at (75, 50)",
            current_routes={
                "DATA0": [(50, 50), (75, 50), (100, 50)],
                "CLK": [(75, 30), (75, 50), (75, 70)]
            }
        )

        current_solution = {
            "layer_utilization": {"F.Cu": 0.45, "In1.Cu": 0.30},
            "via_count": {"DATA0": 0, "CLK": 2}
        }

        print(f"\nResolving conflict: {conflict.conflict_id}")
        print(f"  Type: {conflict.conflict_type.name}")
        print(f"  Nets: {conflict.net_a} vs {conflict.net_b}")
        print(f"  Location: {conflict.location}")

        resolution = await agent.resolve_conflict(conflict, current_solution)

        print(f"\nResolution:")
        print(f"  Strategy: {resolution.strategy.name}")
        print(f"  Affected Nets: {resolution.affected_nets}")
        print(f"  Estimated Cost: {resolution.estimated_cost:.2f}")
        print(f"  Confidence: {resolution.confidence:.0%}")
        print(f"  Reasoning: {resolution.reasoning}")

        if resolution.actions:
            print(f"\n  Actions:")
            for action in resolution.actions:
                print(f"    - {action}")

    asyncio.run(test_resolver())
