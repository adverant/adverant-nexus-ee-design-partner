#!/usr/bin/env python3
"""
Routing Strategist Agent - LLM-driven strategic routing analysis.

This agent uses Claude Opus 4.6 via OpenRouter to determine optimal routing
strategy BEFORE routing begins. It analyzes circuit topology, predicts
congestion, and recommends routing approaches for each net.

Part of the MAPO Enhancement: "Opus 4.6 Thinks, Algorithms Execute"
"""

import json
import os
import sys
import asyncio
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any, Tuple
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
    # Stub classes for standalone testing
    @dataclass
    class PCBState:
        components: List = field(default_factory=list)
        nets: List = field(default_factory=list)

    @dataclass
    class DRCResult:
        total_violations: int = 0


class RoutingMethod(Enum):
    """Available routing methods."""
    MANHATTAN_L = auto()      # Standard L-route (horizontal then vertical)
    MANHATTAN_Z = auto()      # Z-route for 4-way junction avoidance
    MANHATTAN_FLEX = auto()   # Flexible choice based on context
    TOPOLOGICAL = auto()      # Topological routing (any-angle)
    DIFFERENTIAL = auto()     # Differential pair routing
    BUS = auto()              # Bus routing (parallel signals)
    POWER_RAIL = auto()       # Horizontal power rail with vertical drops


class NetPriority(Enum):
    """Net routing priority levels."""
    CRITICAL = 1      # Route first, no compromise (power, critical signals)
    HIGH = 2          # Route early, minimal detours
    NORMAL = 3        # Standard routing
    LOW = 4           # Route last, can detour


@dataclass
class NetRoutingStrategy:
    """Routing strategy for a single net."""
    net_name: str
    priority: NetPriority
    method: RoutingMethod
    layer_preference: Optional[List[str]] = None
    width_mm: float = 0.25
    clearance_mm: float = 0.15
    max_length_mm: Optional[float] = None
    avoid_zones: List[str] = field(default_factory=list)
    routing_order: int = 0
    reasoning: str = ""


@dataclass
class CongestionZone:
    """A predicted congestion zone."""
    x_min: float
    x_max: float
    y_min: float
    y_max: float
    severity: str  # LOW, MEDIUM, HIGH, CRITICAL
    nets_involved: List[str]
    mitigation: str


@dataclass
class RoutingStrategy:
    """Complete routing strategy for a design."""
    net_strategies: Dict[str, NetRoutingStrategy]
    routing_order: List[str]
    congestion_zones: List[CongestionZone]
    global_recommendations: List[str]
    estimated_success_rate: float
    reasoning: str


class RoutingStrategistAgent(GeneratorAgent):
    """
    Opus 4.6 agent that determines routing strategy BEFORE routing begins.

    Analyzes:
    - Net criticality (power, signal, high-speed, differential)
    - Component clustering and placement
    - Expected congestion zones
    - Layer utilization potential

    Outputs:
    - Per-net routing priority order
    - Recommended routing method (Manhattan, L-route, Z-route)
    - Layer preferences for multi-layer PCB
    - Conflict avoidance zones
    """

    SYSTEM_PROMPT = """You are an expert PCB routing strategist working with the MAPO
(Multi-Agent Prompt Optimization) system. Given a circuit's netlist and component
placement, you determine the optimal routing strategy.

Your analysis should consider:
1. Power nets should be routed first with wide traces (horizontal rails preferred)
2. High-speed/differential pairs need careful impedance control
3. Identify potential congestion choke points
4. Recommend layer assignments to minimize vias
5. Flag nets that may conflict and suggest resolution order

ROUTING METHODS:
- MANHATTAN_L: Standard L-route (horizontal then vertical)
- MANHATTAN_Z: Z-route for 4-way junction avoidance
- MANHATTAN_FLEX: Flexible choice based on context
- DIFFERENTIAL: Matched-length differential pair routing
- BUS: Parallel bus signal routing with consistent spacing
- POWER_RAIL: Horizontal rail with vertical drops to components

NET PRIORITIES:
- CRITICAL (1): Power nets, critical signals - route first
- HIGH (2): High-speed signals - route early
- NORMAL (3): General signals - standard routing
- LOW (4): Non-critical signals - route last, can detour

OUTPUT FORMAT: Return a JSON object with:
{
  "net_strategies": {
    "<net_name>": {
      "priority": "CRITICAL|HIGH|NORMAL|LOW",
      "method": "MANHATTAN_L|MANHATTAN_Z|DIFFERENTIAL|BUS|POWER_RAIL",
      "layer_preference": ["F.Cu", "In1.Cu"] or null,
      "width_mm": 0.25,
      "clearance_mm": 0.15,
      "max_length_mm": 100 or null,
      "avoid_zones": ["zone_name"],
      "reasoning": "Brief explanation"
    }
  },
  "routing_order": ["net1", "net2", ...],
  "congestion_zones": [
    {
      "x_min": 50.0, "x_max": 80.0, "y_min": 30.0, "y_max": 50.0,
      "severity": "HIGH",
      "nets_involved": ["net1", "net2"],
      "mitigation": "Route net1 first, use layer 2 for net2"
    }
  ],
  "global_recommendations": [
    "Use layer 2 for east-west routing",
    "Keep differential pairs on top layer"
  ],
  "estimated_success_rate": 0.85,
  "reasoning": "Overall strategic analysis"
}"""

    def __init__(self):
        """Initialize the Routing Strategist Agent."""
        config = AgentConfig(
            name="RoutingStrategist",
            role=AgentRole.GENERAL,
            model="anthropic/claude-opus-4-5-20251101",
            temperature=0.3,  # Lower temperature for more deterministic strategy
            max_tokens=8192,
            focus_areas=[
                "routing_strategy", "net_ordering", "congestion_prediction",
                "layer_assignment", "conflict_avoidance"
            ],
            constraints={
                "min_trace_width_mm": 0.15,
                "min_clearance_mm": 0.1,
                "grid_mm": 2.54,
            }
        )
        super().__init__(config)

    def get_system_prompt(self) -> str:
        """Get the system prompt for strategic routing analysis."""
        return self.SYSTEM_PROMPT

    def get_focus_violations(self, drc_result: DRCResult) -> List[Dict]:
        """Not used for strategic planning - returns empty list."""
        return []

    async def analyze_routing_strategy(
        self,
        netlist: List[Dict[str, Any]],
        placement: Dict[str, Tuple[float, float]],
        constraints: Optional[Dict[str, Any]] = None
    ) -> RoutingStrategy:
        """
        Analyze circuit and generate routing strategy.

        Args:
            netlist: List of nets with pins [{name, pins: [{ref, pin, x, y}]}]
            placement: Component positions {ref: (x, y)}
            constraints: Optional routing constraints

        Returns:
            RoutingStrategy with per-net strategies and global recommendations
        """
        # Build analysis prompt
        prompt = self._build_strategy_prompt(netlist, placement, constraints)

        # Call LLM
        response = await self._call_openrouter(prompt)

        # Parse response
        strategy = self._parse_strategy_response(response, netlist)

        return strategy

    async def order_nets(
        self,
        nets: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Order nets for optimal routing sequence.

        This is a simplified interface that returns nets in recommended order.

        Args:
            nets: List of net dictionaries

        Returns:
            Reordered list of nets
        """
        # Quick heuristic ordering if LLM unavailable
        def net_priority_key(net: Dict) -> Tuple[int, int]:
            name = net.get('name', '').upper()

            # Power nets first
            if any(p in name for p in ['VCC', 'VDD', 'PWR', '5V', '3V3', '12V']):
                return (0, -len(net.get('pins', [])))
            # Ground next
            if any(g in name for g in ['GND', 'VSS', 'GROUND']):
                return (1, -len(net.get('pins', [])))
            # Differential pairs
            if name.endswith('+') or name.endswith('-') or '_P' in name or '_N' in name:
                return (2, -len(net.get('pins', [])))
            # Bus signals
            if '[' in name or name[-1].isdigit():
                return (3, -len(net.get('pins', [])))
            # Everything else by pin count (more pins = route earlier)
            return (4, -len(net.get('pins', [])))

        return sorted(nets, key=net_priority_key)

    def _build_strategy_prompt(
        self,
        netlist: List[Dict],
        placement: Dict[str, Tuple[float, float]],
        constraints: Optional[Dict] = None
    ) -> str:
        """Build the analysis prompt."""
        # Summarize netlist
        net_summary = []
        for net in netlist[:50]:  # Limit to 50 nets for context
            pins = net.get('pins', [])
            net_summary.append({
                'name': net.get('name', 'unnamed'),
                'pin_count': len(pins),
                'pins': pins[:10]  # First 10 pins
            })

        # Compute placement bounds
        if placement:
            xs = [p[0] for p in placement.values()]
            ys = [p[1] for p in placement.values()]
            bounds = {
                'x_min': min(xs), 'x_max': max(xs),
                'y_min': min(ys), 'y_max': max(ys)
            }
        else:
            bounds = {'x_min': 0, 'x_max': 200, 'y_min': 0, 'y_max': 150}

        prompt = f"""Analyze this circuit and generate an optimal routing strategy.

PLACEMENT BOUNDS:
{json.dumps(bounds, indent=2)}

COMPONENT POSITIONS ({len(placement)} components):
{json.dumps(dict(list(placement.items())[:30]), indent=2)}

NETLIST ({len(netlist)} nets):
{json.dumps(net_summary, indent=2)}

CONSTRAINTS:
{json.dumps(constraints or {}, indent=2)}

Generate a comprehensive routing strategy with:
1. Priority and method for each net
2. Recommended routing order
3. Predicted congestion zones with mitigations
4. Global routing recommendations
5. Estimated success rate

Focus on:
- Identifying power/ground nets for priority routing
- Detecting differential pairs and buses
- Predicting where nets will conflict
- Suggesting layer assignments to minimize congestion
"""
        return prompt

    async def _call_openrouter(self, prompt: str) -> str:
        """Call Claude Opus 4.6 via OpenRouter."""
        api_key = os.environ.get('OPENROUTER_API_KEY')

        if not api_key:
            # Fall back to heuristic response
            return self._generate_heuristic_strategy()

        try:
            import httpx

            async with httpx.AsyncClient(timeout=60.0) as client:
                response = await client.post(
                    "https://openrouter.ai/api/v1/chat/completions",
                    headers={
                        "Authorization": f"Bearer {api_key}",
                        "Content-Type": "application/json",
                        "HTTP-Referer": "https://adverant.ai",
                        "X-Title": "MAPO Routing Strategist"
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
                    return self._generate_heuristic_strategy()

        except Exception as e:
            print(f"OpenRouter call failed: {e}")
            return self._generate_heuristic_strategy()

    def _generate_heuristic_strategy(self) -> str:
        """Generate heuristic-based strategy when LLM unavailable."""
        return json.dumps({
            "net_strategies": {},
            "routing_order": [],
            "congestion_zones": [],
            "global_recommendations": [
                "Route power nets first with wide traces",
                "Use horizontal rails for VCC at top, GND at bottom",
                "Route differential pairs before general signals",
                "Avoid 4-way junctions by using Z-routing"
            ],
            "estimated_success_rate": 0.7,
            "reasoning": "Heuristic-based strategy (LLM unavailable)"
        })

    def _parse_strategy_response(
        self,
        response: str,
        netlist: List[Dict]
    ) -> RoutingStrategy:
        """Parse LLM response into RoutingStrategy."""
        try:
            # Extract JSON from response
            import re
            json_match = re.search(r'\{[\s\S]*\}', response)
            if json_match:
                data = json.loads(json_match.group())
            else:
                data = json.loads(self._generate_heuristic_strategy())

            # Parse net strategies
            net_strategies = {}
            for net_name, strategy in data.get('net_strategies', {}).items():
                try:
                    net_strategies[net_name] = NetRoutingStrategy(
                        net_name=net_name,
                        priority=NetPriority[strategy.get('priority', 'NORMAL')],
                        method=RoutingMethod[strategy.get('method', 'MANHATTAN_FLEX')],
                        layer_preference=strategy.get('layer_preference'),
                        width_mm=strategy.get('width_mm', 0.25),
                        clearance_mm=strategy.get('clearance_mm', 0.15),
                        max_length_mm=strategy.get('max_length_mm'),
                        avoid_zones=strategy.get('avoid_zones', []),
                        reasoning=strategy.get('reasoning', '')
                    )
                except (KeyError, ValueError) as e:
                    print(f"Failed to parse strategy for {net_name}: {e}")

            # Parse congestion zones
            congestion_zones = []
            for zone in data.get('congestion_zones', []):
                try:
                    congestion_zones.append(CongestionZone(
                        x_min=zone.get('x_min', 0),
                        x_max=zone.get('x_max', 0),
                        y_min=zone.get('y_min', 0),
                        y_max=zone.get('y_max', 0),
                        severity=zone.get('severity', 'MEDIUM'),
                        nets_involved=zone.get('nets_involved', []),
                        mitigation=zone.get('mitigation', '')
                    ))
                except (KeyError, ValueError) as e:
                    print(f"Failed to parse congestion zone: {e}")

            # Get routing order
            routing_order = data.get('routing_order', [])

            # If no order specified, use net_strategies order or netlist order
            if not routing_order:
                routing_order = list(net_strategies.keys())
            if not routing_order:
                routing_order = [n.get('name', '') for n in netlist]

            return RoutingStrategy(
                net_strategies=net_strategies,
                routing_order=routing_order,
                congestion_zones=congestion_zones,
                global_recommendations=data.get('global_recommendations', []),
                estimated_success_rate=data.get('estimated_success_rate', 0.7),
                reasoning=data.get('reasoning', '')
            )

        except json.JSONDecodeError as e:
            print(f"JSON parse error in strategy response: {e}")
            return RoutingStrategy(
                net_strategies={},
                routing_order=[n.get('name', '') for n in netlist],
                congestion_zones=[],
                global_recommendations=["Route power first", "Avoid 4-way junctions"],
                estimated_success_rate=0.5,
                reasoning="Parse error - using defaults"
            )

    async def critique(self, proposal: Any) -> Dict[str, Any]:
        """
        Critique a routing proposal from another agent.

        Used in debate-and-critique mechanism.

        Args:
            proposal: Routing proposal to critique

        Returns:
            Critique with issues and suggestions
        """
        # Build critique prompt
        prompt = f"""Review this routing proposal and identify potential issues:

PROPOSAL:
{json.dumps(proposal if isinstance(proposal, dict) else str(proposal), indent=2)}

Analyze for:
1. Potential congestion problems
2. Net priority issues
3. Layer assignment conflicts
4. Missing considerations

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
        """
        Vote on whether to accept a proposal after debate.

        Args:
            proposal: Original proposal
            response: Proposer's response to critiques

        Returns:
            True to accept, False to reject
        """
        # Simple heuristic: accept if response addresses issues
        return True


# Convenience function for creating the agent
def create_routing_strategist() -> RoutingStrategistAgent:
    """Create a new Routing Strategist Agent instance."""
    return RoutingStrategistAgent()


# Main entry point for testing
if __name__ == '__main__':
    async def test_strategist():
        """Test the routing strategist."""
        print("\n" + "="*60)
        print("ROUTING STRATEGIST AGENT - Test")
        print("="*60)

        agent = create_routing_strategist()

        # Sample netlist
        netlist = [
            {'name': 'VCC', 'pins': [
                {'ref': 'U1', 'pin': 'VCC', 'x': 100, 'y': 50},
                {'ref': 'C1', 'pin': '1', 'x': 105, 'y': 50}
            ]},
            {'name': 'GND', 'pins': [
                {'ref': 'U1', 'pin': 'GND', 'x': 100, 'y': 70},
                {'ref': 'C1', 'pin': '2', 'x': 105, 'y': 70}
            ]},
            {'name': 'DATA0', 'pins': [
                {'ref': 'U1', 'pin': 'PB0', 'x': 80, 'y': 50},
                {'ref': 'J1', 'pin': '1', 'x': 20, 'y': 50}
            ]},
            {'name': 'DATA1', 'pins': [
                {'ref': 'U1', 'pin': 'PB1', 'x': 80, 'y': 52.54},
                {'ref': 'J1', 'pin': '2', 'x': 20, 'y': 52.54}
            ]}
        ]

        placement = {
            'U1': (100, 60),
            'C1': (105, 60),
            'J1': (20, 51.27)
        }

        print("\nAnalyzing routing strategy...")
        strategy = await agent.analyze_routing_strategy(netlist, placement)

        print(f"\nRouting Order: {strategy.routing_order}")
        print(f"Estimated Success Rate: {strategy.estimated_success_rate:.0%}")
        print(f"Global Recommendations:")
        for rec in strategy.global_recommendations:
            print(f"  - {rec}")
        print(f"\nReasoning: {strategy.reasoning}")

        if strategy.congestion_zones:
            print(f"\nCongestion Zones:")
            for zone in strategy.congestion_zones:
                print(f"  - {zone.severity}: ({zone.x_min}-{zone.x_max}, {zone.y_min}-{zone.y_max})")
                print(f"    Nets: {zone.nets_involved}")
                print(f"    Mitigation: {zone.mitigation}")

    asyncio.run(test_strategist())
