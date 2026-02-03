#!/usr/bin/env python3
"""
Congestion Predictor Agent - LLM-driven congestion prediction without GPU/ML.

This agent uses Claude Opus 4.5 via OpenRouter to predict routing congestion
BEFORE actual routing begins, using semantic analysis of component placement
and netlist topology.

Part of the MAPO v2.0 Enhancement: "Opus 4.5 Thinks, Algorithms Execute"
"""

import json
import os
import sys
import asyncio
import math
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
    pass


class CongestionSeverity(Enum):
    """Congestion severity levels."""
    LOW = auto()       # Minor congestion, normal routing should work
    MEDIUM = auto()    # Moderate congestion, may need detours
    HIGH = auto()      # Significant congestion, consider layer changes
    CRITICAL = auto()  # Severe congestion, requires special handling


@dataclass
class CongestionRegion:
    """A predicted congestion region on the board."""
    x_min: float
    x_max: float
    y_min: float
    y_max: float
    severity: CongestionSeverity
    pin_density: float  # pins per mm²
    net_crossings: int  # estimated net crossing count
    involved_nets: List[str]
    involved_components: List[str]
    mitigation_strategy: str
    layer_escape_recommendation: Optional[str] = None

    @property
    def center(self) -> Tuple[float, float]:
        """Get center of the region."""
        return ((self.x_min + self.x_max) / 2, (self.y_min + self.y_max) / 2)

    @property
    def area(self) -> float:
        """Get area of the region in mm²."""
        return (self.x_max - self.x_min) * (self.y_max - self.y_min)


@dataclass
class CongestionPrediction:
    """Complete congestion prediction for a design."""
    regions: List[CongestionRegion]
    global_congestion_score: float  # 0-1, higher = more congested
    routing_feasibility: float  # 0-1, estimated probability of successful routing
    hotspots: List[Tuple[float, float]]  # (x, y) of most critical points
    recommended_detours: List[Dict[str, Any]]
    layer_usage_recommendation: Dict[str, float]  # layer: utilization percentage
    reasoning: str


class CongestionPredictorAgent(GeneratorAgent):
    """
    Opus 4.5 agent that predicts routing congestion without ML/GPU.

    Uses semantic understanding of:
    - Pin density in regions
    - Net crossing patterns
    - Component adjacency
    - Historical patterns from similar designs

    Outputs:
    - Congestion heatmap (region-based)
    - Recommended routing detours
    - Layer escape suggestions for dense areas
    """

    SYSTEM_PROMPT = """You are a PCB congestion prediction expert working with MAPO v2.0.
Your task is to analyze component placement and netlist topology to predict where
routing congestion will occur BEFORE actual routing begins.

ANALYSIS FACTORS:
1. Pin Density: Count pins per unit area - higher density = more congestion
2. Net Crossings: Analyze which nets must cross each other
3. BGA/QFP Fanout: Large ICs with many pins create escape routing challenges
4. Connector Bottlenecks: Multiple signals converging at connectors
5. Power Distribution: Wide power traces competing for space
6. Component Clustering: Dense component groups create routing channels

CONGESTION SEVERITY:
- LOW: < 0.5 pins/mm², < 3 net crossings in region
- MEDIUM: 0.5-1.0 pins/mm², 3-10 net crossings
- HIGH: 1.0-2.0 pins/mm², 10-20 net crossings
- CRITICAL: > 2.0 pins/mm² or > 20 net crossings

MITIGATION STRATEGIES:
1. Route critical/long nets first through congested area
2. Use layer transitions to escape dense regions
3. Route around congestion zone (detour)
4. Use bus routing to reduce effective wire count
5. Spread connections across multiple layers

OUTPUT FORMAT: Return JSON:
{
  "regions": [
    {
      "x_min": 50.0, "x_max": 80.0, "y_min": 30.0, "y_max": 60.0,
      "severity": "HIGH",
      "pin_density": 1.5,
      "net_crossings": 15,
      "involved_nets": ["VCC", "DATA0", "CLK"],
      "involved_components": ["U1", "U2"],
      "mitigation_strategy": "Route CLK first on layer 2, escape DATA via layer 3",
      "layer_escape_recommendation": "Use inner layers for east-west escape"
    }
  ],
  "global_congestion_score": 0.65,
  "routing_feasibility": 0.80,
  "hotspots": [[75.0, 45.0], [120.0, 80.0]],
  "recommended_detours": [
    {
      "net": "DATA_BUS",
      "avoid_region": {"x_min": 60, "x_max": 90, "y_min": 40, "y_max": 70},
      "suggested_path": "Route north then east"
    }
  ],
  "layer_usage_recommendation": {
    "F.Cu": 0.75,
    "In1.Cu": 0.50,
    "In2.Cu": 0.45,
    "B.Cu": 0.60
  },
  "reasoning": "Overall congestion analysis explanation"
}"""

    def __init__(self):
        """Initialize the Congestion Predictor Agent."""
        config = AgentConfig(
            name="CongestionPredictor",
            role=AgentRole.GENERAL,
            model="anthropic/claude-opus-4-5-20251101",
            temperature=0.2,  # Low temperature for consistent predictions
            max_tokens=8192,
            focus_areas=[
                "congestion_prediction", "pin_density", "net_crossings",
                "layer_escape", "routing_feasibility"
            ],
            constraints={
                "grid_mm": 2.54,
                "min_region_size_mm": 10.0,
            }
        )
        super().__init__(config)

    def get_system_prompt(self) -> str:
        """Get the system prompt for congestion prediction."""
        return self.SYSTEM_PROMPT

    def get_focus_violations(self, drc_result: DRCResult) -> List[Dict]:
        """Not used for congestion prediction - returns empty list."""
        return []

    async def predict_congestion(
        self,
        placement: Dict[str, Tuple[float, float]],
        netlist: List[Dict[str, Any]],
        pin_positions: Optional[Dict[str, List[Tuple[float, float]]]] = None
    ) -> CongestionPrediction:
        """
        Predict routing congestion based on placement and netlist.

        Args:
            placement: Component positions {ref: (x, y)}
            netlist: List of nets [{name, pins: [{ref, pin, x, y}]}]
            pin_positions: Optional pre-computed pin positions

        Returns:
            CongestionPrediction with regions and recommendations
        """
        # Compute pin positions if not provided
        if pin_positions is None:
            pin_positions = self._compute_pin_positions(placement, netlist)

        # Compute heuristic analysis first (fast, no LLM)
        heuristic_regions = self._compute_heuristic_congestion(
            placement, netlist, pin_positions
        )

        # Build prompt with heuristic analysis
        prompt = self._build_prediction_prompt(
            placement, netlist, pin_positions, heuristic_regions
        )

        # Call LLM for refined prediction
        response = await self._call_openrouter(prompt)

        # Parse response
        prediction = self._parse_prediction_response(response, heuristic_regions)

        return prediction

    def _compute_pin_positions(
        self,
        placement: Dict[str, Tuple[float, float]],
        netlist: List[Dict]
    ) -> Dict[str, List[Tuple[float, float]]]:
        """Compute all pin positions from placement and netlist."""
        pin_positions = {}

        for net in netlist:
            net_name = net.get('name', '')
            positions = []
            for pin in net.get('pins', []):
                # Try to get position from pin data
                x = pin.get('x')
                y = pin.get('y')

                # If not in pin, use component position
                if x is None or y is None:
                    ref = pin.get('ref', '')
                    if ref in placement:
                        x, y = placement[ref]

                if x is not None and y is not None:
                    positions.append((float(x), float(y)))

            if positions:
                pin_positions[net_name] = positions

        return pin_positions

    def _compute_heuristic_congestion(
        self,
        placement: Dict[str, Tuple[float, float]],
        netlist: List[Dict],
        pin_positions: Dict[str, List[Tuple[float, float]]]
    ) -> List[Dict]:
        """Compute heuristic-based congestion regions without LLM."""
        regions = []

        # Skip if no data
        if not placement or not pin_positions:
            return regions

        # Get board bounds
        all_positions = list(placement.values())
        for positions in pin_positions.values():
            all_positions.extend(positions)

        if not all_positions:
            return regions

        x_min = min(p[0] for p in all_positions)
        x_max = max(p[0] for p in all_positions)
        y_min = min(p[1] for p in all_positions)
        y_max = max(p[1] for p in all_positions)

        # Add margin
        margin = 10.0
        x_min -= margin
        x_max += margin
        y_min -= margin
        y_max += margin

        # Divide into grid cells (20mm x 20mm)
        cell_size = 20.0
        n_cols = max(1, int((x_max - x_min) / cell_size) + 1)
        n_rows = max(1, int((y_max - y_min) / cell_size) + 1)

        # Count pins per cell
        cell_pins = {}
        cell_nets = {}

        for net_name, positions in pin_positions.items():
            for x, y in positions:
                col = int((x - x_min) / cell_size)
                row = int((y - y_min) / cell_size)
                col = max(0, min(col, n_cols - 1))
                row = max(0, min(row, n_rows - 1))

                key = (row, col)
                cell_pins[key] = cell_pins.get(key, 0) + 1
                if key not in cell_nets:
                    cell_nets[key] = set()
                cell_nets[key].add(net_name)

        # Identify high-density cells
        for (row, col), pin_count in cell_pins.items():
            cell_area = cell_size * cell_size
            density = pin_count / cell_area

            # Determine severity
            if density > 2.0:
                severity = "CRITICAL"
            elif density > 1.0:
                severity = "HIGH"
            elif density > 0.5:
                severity = "MEDIUM"
            else:
                continue  # Skip low congestion cells

            # Create region
            rx_min = x_min + col * cell_size
            rx_max = rx_min + cell_size
            ry_min = y_min + row * cell_size
            ry_max = ry_min + cell_size

            regions.append({
                'x_min': rx_min,
                'x_max': rx_max,
                'y_min': ry_min,
                'y_max': ry_max,
                'severity': severity,
                'pin_density': round(density, 2),
                'pin_count': pin_count,
                'net_count': len(cell_nets.get((row, col), set())),
                'nets': list(cell_nets.get((row, col), set()))[:10]
            })

        # Sort by severity
        severity_order = {'CRITICAL': 0, 'HIGH': 1, 'MEDIUM': 2, 'LOW': 3}
        regions.sort(key=lambda r: severity_order.get(r['severity'], 4))

        return regions[:10]  # Return top 10 regions

    def _build_prediction_prompt(
        self,
        placement: Dict[str, Tuple[float, float]],
        netlist: List[Dict],
        pin_positions: Dict[str, List[Tuple[float, float]]],
        heuristic_regions: List[Dict]
    ) -> str:
        """Build the congestion prediction prompt."""
        # Compute board bounds
        all_positions = list(placement.values())
        if all_positions:
            x_min = min(p[0] for p in all_positions)
            x_max = max(p[0] for p in all_positions)
            y_min = min(p[1] for p in all_positions)
            y_max = max(p[1] for p in all_positions)
        else:
            x_min, x_max, y_min, y_max = 0, 200, 0, 150

        # Summarize netlist
        net_summary = []
        for net_name, positions in list(pin_positions.items())[:30]:
            net_summary.append({
                'name': net_name,
                'pin_count': len(positions),
                'span_x': max(p[0] for p in positions) - min(p[0] for p in positions) if positions else 0,
                'span_y': max(p[1] for p in positions) - min(p[1] for p in positions) if positions else 0
            })

        prompt = f"""Analyze this PCB layout and predict routing congestion.

BOARD BOUNDS:
  x: {x_min:.1f} to {x_max:.1f} mm
  y: {y_min:.1f} to {y_max:.1f} mm

COMPONENT COUNT: {len(placement)}

COMPONENT POSITIONS (sample):
{json.dumps(dict(list(placement.items())[:20]), indent=2)}

NET STATISTICS ({len(pin_positions)} nets):
{json.dumps(net_summary, indent=2)}

HEURISTIC ANALYSIS (pre-computed congestion regions):
{json.dumps(heuristic_regions, indent=2)}

Based on this data, provide a refined congestion prediction with:
1. Validated/adjusted congestion regions with mitigation strategies
2. Global congestion score (0-1)
3. Routing feasibility estimate
4. Hotspot coordinates
5. Recommended detours for difficult nets
6. Layer usage recommendations

Consider:
- Net crossing patterns (nets spanning opposite directions)
- BGA/QFP escape routing challenges
- Power net width requirements
- Signal integrity constraints (avoid routing near power)
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
                        "X-Title": "MAPO Congestion Predictor"
                    },
                    json={
                        "model": "anthropic/claude-opus-4-5-20251101",
                        "temperature": 0.2,
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
            "regions": [],
            "global_congestion_score": 0.5,
            "routing_feasibility": 0.7,
            "hotspots": [],
            "recommended_detours": [],
            "layer_usage_recommendation": {
                "F.Cu": 0.60,
                "B.Cu": 0.55
            },
            "reasoning": "Heuristic-based prediction (LLM unavailable)"
        })

    def _parse_prediction_response(
        self,
        response: str,
        heuristic_regions: List[Dict]
    ) -> CongestionPrediction:
        """Parse LLM response into CongestionPrediction."""
        try:
            import re
            json_match = re.search(r'\{[\s\S]*\}', response)
            if json_match:
                data = json.loads(json_match.group())
            else:
                data = json.loads(self._generate_heuristic_response())

            # Parse regions
            regions = []
            for r in data.get('regions', heuristic_regions):
                try:
                    severity_str = r.get('severity', 'MEDIUM').upper()
                    severity = CongestionSeverity[severity_str]

                    regions.append(CongestionRegion(
                        x_min=r.get('x_min', 0),
                        x_max=r.get('x_max', 0),
                        y_min=r.get('y_min', 0),
                        y_max=r.get('y_max', 0),
                        severity=severity,
                        pin_density=r.get('pin_density', 0),
                        net_crossings=r.get('net_crossings', 0),
                        involved_nets=r.get('involved_nets', r.get('nets', [])),
                        involved_components=r.get('involved_components', []),
                        mitigation_strategy=r.get('mitigation_strategy', ''),
                        layer_escape_recommendation=r.get('layer_escape_recommendation')
                    ))
                except (KeyError, ValueError) as e:
                    print(f"Failed to parse region: {e}")

            # If no regions from LLM, use heuristic regions
            if not regions and heuristic_regions:
                for r in heuristic_regions:
                    severity_str = r.get('severity', 'MEDIUM').upper()
                    try:
                        severity = CongestionSeverity[severity_str]
                    except KeyError:
                        severity = CongestionSeverity.MEDIUM

                    regions.append(CongestionRegion(
                        x_min=r.get('x_min', 0),
                        x_max=r.get('x_max', 0),
                        y_min=r.get('y_min', 0),
                        y_max=r.get('y_max', 0),
                        severity=severity,
                        pin_density=r.get('pin_density', 0),
                        net_crossings=r.get('net_count', 0) * 2,
                        involved_nets=r.get('nets', []),
                        involved_components=[],
                        mitigation_strategy="Route critical nets first"
                    ))

            # Parse hotspots
            hotspots = []
            for h in data.get('hotspots', []):
                if isinstance(h, (list, tuple)) and len(h) >= 2:
                    hotspots.append((float(h[0]), float(h[1])))

            return CongestionPrediction(
                regions=regions,
                global_congestion_score=float(data.get('global_congestion_score', 0.5)),
                routing_feasibility=float(data.get('routing_feasibility', 0.7)),
                hotspots=hotspots,
                recommended_detours=data.get('recommended_detours', []),
                layer_usage_recommendation=data.get('layer_usage_recommendation', {}),
                reasoning=data.get('reasoning', '')
            )

        except json.JSONDecodeError as e:
            print(f"JSON parse error in prediction response: {e}")
            return CongestionPrediction(
                regions=[],
                global_congestion_score=0.5,
                routing_feasibility=0.7,
                hotspots=[],
                recommended_detours=[],
                layer_usage_recommendation={},
                reasoning="Parse error - using defaults"
            )

    def get_cost_modifier(
        self,
        prediction: CongestionPrediction,
        x: float,
        y: float
    ) -> float:
        """
        Get routing cost modifier for a point based on congestion prediction.

        Used by PathFinder router to adjust costs in congested regions.

        Args:
            prediction: Congestion prediction
            x, y: Point coordinates

        Returns:
            Cost multiplier (1.0 = normal, >1.0 = avoid, <1.0 = prefer)
        """
        base_cost = 1.0

        for region in prediction.regions:
            if (region.x_min <= x <= region.x_max and
                region.y_min <= y <= region.y_max):
                # Point is in congested region
                if region.severity == CongestionSeverity.CRITICAL:
                    base_cost *= 3.0
                elif region.severity == CongestionSeverity.HIGH:
                    base_cost *= 2.0
                elif region.severity == CongestionSeverity.MEDIUM:
                    base_cost *= 1.5

        return base_cost


# Convenience function
def create_congestion_predictor() -> CongestionPredictorAgent:
    """Create a new Congestion Predictor Agent instance."""
    return CongestionPredictorAgent()


# Main entry point for testing
if __name__ == '__main__':
    async def test_predictor():
        """Test the congestion predictor."""
        print("\n" + "="*60)
        print("CONGESTION PREDICTOR AGENT - Test")
        print("="*60)

        agent = create_congestion_predictor()

        # Sample placement
        placement = {
            'U1': (100, 50),
            'U2': (100, 80),
            'C1': (90, 50),
            'C2': (90, 55),
            'C3': (90, 60),
            'C4': (110, 50),
            'C5': (110, 55),
            'R1': (80, 60),
            'R2': (80, 65),
            'J1': (20, 60),
        }

        # Sample netlist
        netlist = [
            {'name': 'VCC', 'pins': [
                {'ref': 'U1', 'pin': 'VCC', 'x': 95, 'y': 45},
                {'ref': 'U2', 'pin': 'VCC', 'x': 95, 'y': 75},
                {'ref': 'C1', 'pin': '1', 'x': 90, 'y': 48},
                {'ref': 'C4', 'pin': '1', 'x': 110, 'y': 48},
            ]},
            {'name': 'GND', 'pins': [
                {'ref': 'U1', 'pin': 'GND', 'x': 105, 'y': 55},
                {'ref': 'U2', 'pin': 'GND', 'x': 105, 'y': 85},
                {'ref': 'C1', 'pin': '2', 'x': 90, 'y': 52},
            ]},
            {'name': 'DATA0', 'pins': [
                {'ref': 'U1', 'pin': 'PB0', 'x': 85, 'y': 50},
                {'ref': 'J1', 'pin': '1', 'x': 20, 'y': 55},
            ]},
            {'name': 'DATA1', 'pins': [
                {'ref': 'U1', 'pin': 'PB1', 'x': 85, 'y': 52},
                {'ref': 'J1', 'pin': '2', 'x': 20, 'y': 58},
            ]},
        ]

        print("\nPredicting congestion...")
        prediction = await agent.predict_congestion(placement, netlist)

        print(f"\nGlobal Congestion Score: {prediction.global_congestion_score:.2f}")
        print(f"Routing Feasibility: {prediction.routing_feasibility:.0%}")
        print(f"Congestion Regions: {len(prediction.regions)}")

        for i, region in enumerate(prediction.regions):
            print(f"\n  Region {i+1}: {region.severity.name}")
            print(f"    Bounds: ({region.x_min:.1f}-{region.x_max:.1f}, {region.y_min:.1f}-{region.y_max:.1f})")
            print(f"    Pin Density: {region.pin_density:.2f} pins/mm²")
            print(f"    Nets: {region.involved_nets[:5]}")
            print(f"    Mitigation: {region.mitigation_strategy}")

        if prediction.hotspots:
            print(f"\nHotspots: {prediction.hotspots}")

        print(f"\nReasoning: {prediction.reasoning}")

    asyncio.run(test_predictor())
