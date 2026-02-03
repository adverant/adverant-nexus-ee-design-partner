#!/usr/bin/env python3
"""
Signal Integrity Advisor Agent - LLM-driven SI guidance without EM simulation.

This agent uses Claude Opus 4.5 via OpenRouter combined with analytical formulas
to provide signal integrity guidance for impedance-critical routing.

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


class SignalType(Enum):
    """Types of signals for SI analysis."""
    SINGLE_ENDED = auto()    # Standard single-ended signal
    DIFFERENTIAL = auto()    # Differential pair
    CLOCK = auto()           # Clock signal (sensitive to jitter)
    HIGH_SPEED = auto()      # High-speed data (>100MHz)
    POWER = auto()           # Power delivery
    LOW_SPEED = auto()       # Low-speed GPIO, etc.


class ProtocolType(Enum):
    """Common protocols with SI requirements."""
    USB = auto()          # USB: 90Ω differential
    ETHERNET = auto()     # Ethernet: 100Ω differential
    LVDS = auto()         # LVDS: 100Ω differential
    HDMI = auto()         # HDMI: 100Ω differential
    PCIE = auto()         # PCIe: 85Ω differential
    DDR = auto()          # DDR: Single-ended ~50Ω
    CAN = auto()          # CAN: 120Ω differential
    SPI = auto()          # SPI: No impedance requirement
    I2C = auto()          # I2C: No impedance requirement
    UART = auto()         # UART: No impedance requirement
    GPIO = auto()         # GPIO: General purpose


@dataclass
class LayerStackup:
    """PCB layer stackup definition."""
    layers: List[Dict[str, Any]]  # {name, type, thickness_mm, dielectric_constant}
    total_thickness_mm: float
    copper_weight_oz: float = 1.0

    @classmethod
    def default_4_layer(cls) -> 'LayerStackup':
        """Create default 4-layer stackup."""
        return cls(
            layers=[
                {'name': 'F.Cu', 'type': 'signal', 'thickness_mm': 0.035, 'er': 1.0},
                {'name': 'prepreg1', 'type': 'dielectric', 'thickness_mm': 0.2, 'er': 4.3},
                {'name': 'In1.Cu', 'type': 'plane', 'thickness_mm': 0.035, 'er': 1.0},
                {'name': 'core', 'type': 'dielectric', 'thickness_mm': 1.0, 'er': 4.5},
                {'name': 'In2.Cu', 'type': 'plane', 'thickness_mm': 0.035, 'er': 1.0},
                {'name': 'prepreg2', 'type': 'dielectric', 'thickness_mm': 0.2, 'er': 4.3},
                {'name': 'B.Cu', 'type': 'signal', 'thickness_mm': 0.035, 'er': 1.0},
            ],
            total_thickness_mm=1.6,
            copper_weight_oz=1.0
        )


@dataclass
class TraceGeometry:
    """Trace geometry for SI calculation."""
    width_mm: float
    thickness_mm: float = 0.035  # 1oz copper
    height_mm: float = 0.2      # Height above reference plane


@dataclass
class DifferentialPairGeometry:
    """Differential pair geometry."""
    trace_width_mm: float
    spacing_mm: float
    thickness_mm: float = 0.035
    height_mm: float = 0.2


@dataclass
class ImpedanceResult:
    """Result of impedance calculation."""
    impedance_ohm: float
    target_ohm: float
    within_tolerance: bool
    recommended_width_mm: float
    formula_used: str
    notes: List[str]


@dataclass
class SIGuidance:
    """Signal integrity guidance for routing."""
    net_name: str
    signal_type: SignalType
    protocol: Optional[ProtocolType]
    target_impedance_ohm: float
    recommended_width_mm: float
    recommended_spacing_mm: Optional[float]  # For differential pairs
    max_length_mm: Optional[float]
    length_matching_tolerance_mm: Optional[float]  # For diff pairs
    via_recommendations: List[str]
    routing_layer_preference: List[str]
    notes: List[str]


class SignalIntegrityAdvisorAgent(GeneratorAgent):
    """
    Opus 4.5 agent for signal integrity guidance.

    Advises on:
    - Trace width for target impedance (given stackup)
    - Differential pair spacing and routing
    - Length matching requirements
    - Via stubs and their impact
    - Reference plane continuity

    Uses analytical formulas + domain expertise (no GPU simulation).
    """

    SYSTEM_PROMPT = """You are a signal integrity expert for MAPO v2.0. Guide routing
decisions for high-speed signals using analytical methods without EM simulation.

IMPEDANCE FORMULAS (IPC-2141 based):

Microstrip (outer layer, trace over plane):
  Z0 ≈ 87/√(εr+1.41) × ln(5.98h/(0.8w+t))
  Where: h=height to plane, w=trace width, t=trace thickness, εr=dielectric constant

Stripline (inner layer, trace between planes):
  Z0 ≈ 60/√εr × ln(4h/(0.67π(0.8w+t)))
  Where: h=distance to nearest plane

Differential Microstrip:
  Zdiff ≈ 2×Z0×(1-0.48×exp(-0.96×s/h))
  Where: s=edge-to-edge spacing

Differential Stripline:
  Zdiff ≈ 2×Z0×(1-0.347×exp(-2.9×s/h))

PROTOCOL REQUIREMENTS:
| Protocol | Type | Target Z (Ω) | Notes |
|----------|------|--------------|-------|
| USB 2.0  | Diff | 90 ± 10%     | D+/D- pair |
| USB 3.x  | Diff | 90 ± 10%     | TX/RX pairs |
| Ethernet | Diff | 100 ± 10%    | MDI pairs |
| LVDS     | Diff | 100 ± 10%    | Low voltage |
| HDMI     | Diff | 100 ± 10%    | TMDS pairs |
| PCIe     | Diff | 85 ± 15%     | TX/RX pairs |
| DDR      | SE   | 40-60        | DQ, DQS |
| CAN      | Diff | 120          | CANH/CANL |

LENGTH MATCHING:
- USB 3.x: ± 5 mil (0.127mm) within pair
- PCIe: ± 5 mil within pair, ± 0.5" between pairs
- DDR: ± 25 mil for DQ to DQS
- Ethernet: ± 50 mil within pair

OUTPUT FORMAT: Return JSON:
{
  "guidance": [
    {
      "net_name": "USB_D+",
      "signal_type": "DIFFERENTIAL",
      "protocol": "USB",
      "target_impedance_ohm": 90,
      "recommended_width_mm": 0.15,
      "recommended_spacing_mm": 0.15,
      "max_length_mm": 100,
      "length_matching_tolerance_mm": 0.127,
      "via_recommendations": [
        "Use via-in-pad where possible",
        "Minimize via stubs on inner layers"
      ],
      "routing_layer_preference": ["F.Cu", "B.Cu"],
      "notes": [
        "Maintain 90Ω ± 10% differential impedance",
        "Keep D+ and D- tightly coupled"
      ]
    }
  ],
  "stackup_analysis": {
    "microstrip_z0": 50.2,
    "stripline_z0": 48.5,
    "recommended_diff_spacing": 0.15
  },
  "general_recommendations": [
    "Route high-speed signals on outer layers for better impedance control",
    "Avoid routing over plane splits"
  ]
}"""

    # Standard protocol impedances
    PROTOCOL_IMPEDANCE = {
        ProtocolType.USB: 90,
        ProtocolType.ETHERNET: 100,
        ProtocolType.LVDS: 100,
        ProtocolType.HDMI: 100,
        ProtocolType.PCIE: 85,
        ProtocolType.DDR: 50,
        ProtocolType.CAN: 120,
        ProtocolType.SPI: None,
        ProtocolType.I2C: None,
        ProtocolType.UART: None,
        ProtocolType.GPIO: None,
    }

    def __init__(self):
        """Initialize the Signal Integrity Advisor Agent."""
        config = AgentConfig(
            name="SignalIntegrityAdvisor",
            role=AgentRole.SIGNAL_INTEGRITY,
            model="anthropic/claude-opus-4-5-20251101",
            temperature=0.2,
            max_tokens=8192,
            focus_areas=[
                "impedance_control", "differential_pairs", "length_matching",
                "via_design", "reference_planes", "crosstalk"
            ],
            constraints={
                "impedance_tolerance_pct": 10,
                "min_trace_width_mm": 0.1,
            }
        )
        super().__init__(config)

    def get_system_prompt(self) -> str:
        """Get the system prompt for SI guidance."""
        return self.SYSTEM_PROMPT

    def get_focus_violations(self, drc_result: DRCResult) -> List[Dict]:
        """Filter for SI-related violations."""
        focus_types = {'clearance', 'track_width'}
        violations = []
        for v in drc_result.top_violations:
            if v.get('type', '') in focus_types:
                violations.append(v)
        return violations

    def calculate_microstrip_impedance(
        self,
        width_mm: float,
        height_mm: float,
        thickness_mm: float = 0.035,
        er: float = 4.3
    ) -> float:
        """
        Calculate microstrip impedance using IPC-2141 formula.

        Args:
            width_mm: Trace width
            height_mm: Height above reference plane
            thickness_mm: Copper thickness
            er: Dielectric constant

        Returns:
            Impedance in ohms
        """
        w = width_mm
        h = height_mm
        t = thickness_mm

        if w <= 0 or h <= 0:
            return 50.0  # Default

        # IPC-2141 microstrip formula
        z0 = (87 / math.sqrt(er + 1.41)) * math.log(5.98 * h / (0.8 * w + t))
        return max(10, min(150, z0))  # Clamp to reasonable range

    def calculate_stripline_impedance(
        self,
        width_mm: float,
        height_mm: float,
        thickness_mm: float = 0.035,
        er: float = 4.5
    ) -> float:
        """
        Calculate stripline impedance using IPC-2141 formula.

        Args:
            width_mm: Trace width
            height_mm: Distance to nearest reference plane
            thickness_mm: Copper thickness
            er: Dielectric constant

        Returns:
            Impedance in ohms
        """
        w = width_mm
        h = height_mm
        t = thickness_mm

        if w <= 0 or h <= 0:
            return 50.0  # Default

        # IPC-2141 stripline formula
        z0 = (60 / math.sqrt(er)) * math.log(4 * h / (0.67 * math.pi * (0.8 * w + t)))
        return max(10, min(150, z0))

    def calculate_differential_impedance(
        self,
        z0: float,
        spacing_mm: float,
        height_mm: float,
        is_microstrip: bool = True
    ) -> float:
        """
        Calculate differential impedance from single-ended.

        Args:
            z0: Single-ended impedance
            spacing_mm: Edge-to-edge spacing
            height_mm: Height to reference plane
            is_microstrip: True for microstrip, False for stripline

        Returns:
            Differential impedance in ohms
        """
        s = spacing_mm
        h = height_mm

        if h <= 0:
            return 2 * z0

        if is_microstrip:
            # Microstrip coupling
            coupling = 1 - 0.48 * math.exp(-0.96 * s / h)
        else:
            # Stripline coupling
            coupling = 1 - 0.347 * math.exp(-2.9 * s / h)

        return 2 * z0 * coupling

    def calculate_width_for_impedance(
        self,
        target_z0: float,
        height_mm: float,
        er: float = 4.3,
        thickness_mm: float = 0.035,
        is_microstrip: bool = True
    ) -> float:
        """
        Calculate trace width for target impedance.

        Uses iterative search to find optimal width.

        Args:
            target_z0: Target impedance
            height_mm: Height to reference plane
            er: Dielectric constant
            thickness_mm: Copper thickness
            is_microstrip: Microstrip or stripline

        Returns:
            Recommended trace width in mm
        """
        # Binary search for width
        width_min = 0.05
        width_max = 2.0

        for _ in range(20):  # 20 iterations for precision
            width = (width_min + width_max) / 2

            if is_microstrip:
                z0 = self.calculate_microstrip_impedance(width, height_mm, thickness_mm, er)
            else:
                z0 = self.calculate_stripline_impedance(width, height_mm, thickness_mm, er)

            if z0 > target_z0:
                width_min = width  # Need wider trace (lower Z)
            else:
                width_max = width  # Need narrower trace (higher Z)

        return round(width, 3)

    async def get_si_guidance(
        self,
        nets: List[Dict[str, Any]],
        stackup: Optional[LayerStackup] = None,
        constraints: Optional[Dict[str, Any]] = None
    ) -> List[SIGuidance]:
        """
        Get SI guidance for a list of nets.

        Args:
            nets: List of nets with name and optional protocol
            stackup: PCB layer stackup (uses default if None)
            constraints: Optional constraints

        Returns:
            List of SIGuidance for each net
        """
        if stackup is None:
            stackup = LayerStackup.default_4_layer()

        # Pre-compute stackup parameters
        stackup_params = self._analyze_stackup(stackup)

        # Build prompt
        prompt = self._build_si_prompt(nets, stackup, stackup_params, constraints)

        # Call LLM
        response = await self._call_openrouter(prompt)

        # Parse response
        guidance_list = self._parse_si_response(response, nets, stackup_params)

        return guidance_list

    def _analyze_stackup(self, stackup: LayerStackup) -> Dict[str, Any]:
        """Analyze stackup and compute key parameters."""
        # Find outer layer height (to first plane)
        outer_height = 0
        for layer in stackup.layers[1:]:
            if layer['type'] == 'dielectric':
                outer_height = layer['thickness_mm']
                break

        # Find average dielectric constant
        dielectric_layers = [l for l in stackup.layers if l['type'] == 'dielectric']
        avg_er = sum(l.get('er', 4.3) for l in dielectric_layers) / max(1, len(dielectric_layers))

        # Calculate reference impedances
        microstrip_z0 = self.calculate_microstrip_impedance(0.2, outer_height, 0.035, avg_er)
        stripline_z0 = self.calculate_stripline_impedance(0.2, outer_height, 0.035, avg_er)

        return {
            'outer_height_mm': outer_height,
            'avg_er': avg_er,
            'microstrip_z0_at_0.2mm': microstrip_z0,
            'stripline_z0_at_0.2mm': stripline_z0,
            'recommended_50ohm_width': self.calculate_width_for_impedance(50, outer_height, avg_er),
            'recommended_90ohm_diff_width': self.calculate_width_for_impedance(45, outer_height, avg_er),
            'recommended_100ohm_diff_width': self.calculate_width_for_impedance(50, outer_height, avg_er),
        }

    def _build_si_prompt(
        self,
        nets: List[Dict],
        stackup: LayerStackup,
        stackup_params: Dict,
        constraints: Optional[Dict] = None
    ) -> str:
        """Build SI guidance prompt."""
        prompt = f"""Provide signal integrity guidance for these nets.

STACKUP PARAMETERS:
{json.dumps(stackup_params, indent=2)}

NETS TO ANALYZE:
{json.dumps(nets[:30], indent=2)}

CONSTRAINTS:
{json.dumps(constraints or {}, indent=2)}

For each net:
1. Identify signal type and protocol
2. Determine target impedance
3. Calculate recommended trace width
4. Specify length matching requirements (if applicable)
5. Provide via and routing recommendations

Focus on practical, manufacturable guidance.
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
                        "X-Title": "MAPO SI Advisor"
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
            "guidance": [],
            "stackup_analysis": {},
            "general_recommendations": [
                "Use 0.2mm trace width for general signals",
                "Use 0.15mm width with 0.15mm spacing for differential pairs"
            ]
        })

    def _parse_si_response(
        self,
        response: str,
        nets: List[Dict],
        stackup_params: Dict
    ) -> List[SIGuidance]:
        """Parse LLM response into SIGuidance list."""
        guidance_list = []

        try:
            import re
            json_match = re.search(r'\{[\s\S]*\}', response)
            if json_match:
                data = json.loads(json_match.group())
            else:
                data = {'guidance': []}

            for g in data.get('guidance', []):
                try:
                    # Parse signal type
                    signal_type_str = g.get('signal_type', 'SINGLE_ENDED')
                    try:
                        signal_type = SignalType[signal_type_str]
                    except KeyError:
                        signal_type = SignalType.SINGLE_ENDED

                    # Parse protocol
                    protocol_str = g.get('protocol')
                    protocol = None
                    if protocol_str:
                        try:
                            protocol = ProtocolType[protocol_str]
                        except KeyError:
                            pass

                    guidance_list.append(SIGuidance(
                        net_name=g.get('net_name', ''),
                        signal_type=signal_type,
                        protocol=protocol,
                        target_impedance_ohm=g.get('target_impedance_ohm', 50),
                        recommended_width_mm=g.get('recommended_width_mm', 0.2),
                        recommended_spacing_mm=g.get('recommended_spacing_mm'),
                        max_length_mm=g.get('max_length_mm'),
                        length_matching_tolerance_mm=g.get('length_matching_tolerance_mm'),
                        via_recommendations=g.get('via_recommendations', []),
                        routing_layer_preference=g.get('routing_layer_preference', []),
                        notes=g.get('notes', [])
                    ))
                except Exception as e:
                    print(f"Failed to parse guidance: {e}")

        except json.JSONDecodeError as e:
            print(f"JSON parse error: {e}")

        # If no guidance from LLM, generate heuristic guidance
        if not guidance_list:
            for net in nets:
                net_name = net.get('name', '').upper()

                # Detect protocol from name
                if 'USB' in net_name:
                    guidance_list.append(SIGuidance(
                        net_name=net.get('name', ''),
                        signal_type=SignalType.DIFFERENTIAL,
                        protocol=ProtocolType.USB,
                        target_impedance_ohm=90,
                        recommended_width_mm=stackup_params.get('recommended_90ohm_diff_width', 0.15),
                        recommended_spacing_mm=0.15,
                        max_length_mm=100,
                        length_matching_tolerance_mm=0.127,
                        via_recommendations=["Minimize vias"],
                        routing_layer_preference=["F.Cu", "B.Cu"],
                        notes=["90Ω differential impedance required"]
                    ))
                elif 'ETH' in net_name or 'MDI' in net_name:
                    guidance_list.append(SIGuidance(
                        net_name=net.get('name', ''),
                        signal_type=SignalType.DIFFERENTIAL,
                        protocol=ProtocolType.ETHERNET,
                        target_impedance_ohm=100,
                        recommended_width_mm=stackup_params.get('recommended_100ohm_diff_width', 0.15),
                        recommended_spacing_mm=0.18,
                        max_length_mm=None,
                        length_matching_tolerance_mm=1.27,
                        via_recommendations=[],
                        routing_layer_preference=["F.Cu", "B.Cu"],
                        notes=["100Ω differential impedance"]
                    ))
                else:
                    # General signal
                    guidance_list.append(SIGuidance(
                        net_name=net.get('name', ''),
                        signal_type=SignalType.SINGLE_ENDED,
                        protocol=None,
                        target_impedance_ohm=50,
                        recommended_width_mm=0.2,
                        recommended_spacing_mm=None,
                        max_length_mm=None,
                        length_matching_tolerance_mm=None,
                        via_recommendations=[],
                        routing_layer_preference=[],
                        notes=[]
                    ))

        return guidance_list


# Convenience function
def create_si_advisor() -> SignalIntegrityAdvisorAgent:
    """Create a new Signal Integrity Advisor Agent instance."""
    return SignalIntegrityAdvisorAgent()


# Main entry point for testing
if __name__ == '__main__':
    async def test_si_advisor():
        """Test the SI advisor."""
        print("\n" + "="*60)
        print("SIGNAL INTEGRITY ADVISOR AGENT - Test")
        print("="*60)

        agent = create_si_advisor()

        # Test impedance calculations
        print("\n--- Impedance Calculations ---")

        z_microstrip = agent.calculate_microstrip_impedance(0.2, 0.2, 0.035, 4.3)
        print(f"Microstrip (w=0.2mm, h=0.2mm): {z_microstrip:.1f} Ω")

        z_stripline = agent.calculate_stripline_impedance(0.2, 0.2, 0.035, 4.5)
        print(f"Stripline (w=0.2mm, h=0.2mm): {z_stripline:.1f} Ω")

        width_50 = agent.calculate_width_for_impedance(50, 0.2, 4.3)
        print(f"Width for 50Ω microstrip: {width_50:.3f} mm")

        width_90 = agent.calculate_width_for_impedance(45, 0.2, 4.3)  # For 90Ω diff
        print(f"Width for 45Ω (90Ω diff): {width_90:.3f} mm")

        # Test SI guidance
        print("\n--- SI Guidance ---")

        nets = [
            {'name': 'USB_D+'},
            {'name': 'USB_D-'},
            {'name': 'ETH_MDI0_P'},
            {'name': 'CLK'},
            {'name': 'DATA0'},
        ]

        guidance = await agent.get_si_guidance(nets)

        for g in guidance:
            print(f"\n{g.net_name}:")
            print(f"  Type: {g.signal_type.name}")
            if g.protocol:
                print(f"  Protocol: {g.protocol.name}")
            print(f"  Target Z: {g.target_impedance_ohm} Ω")
            print(f"  Width: {g.recommended_width_mm} mm")
            if g.recommended_spacing_mm:
                print(f"  Spacing: {g.recommended_spacing_mm} mm")
            if g.notes:
                print(f"  Notes: {g.notes}")

    asyncio.run(test_si_advisor())
