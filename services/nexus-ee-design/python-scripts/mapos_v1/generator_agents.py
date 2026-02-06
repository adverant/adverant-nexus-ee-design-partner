#!/usr/bin/env python3
"""
Generator Agents - Specialized LLM agents for PCB optimization.

This module implements domain-specific agents that generate PCB modifications:
1. SignalIntegrityAgent: Focuses on trace routing, impedance, crosstalk
2. ThermalPowerAgent: Focuses on heat dissipation, current paths
3. ManufacturingAgent: Focuses on DFM, cost optimization, yield

Each agent uses Claude Opus 4.6 with specialized prompts and domain knowledge.
Inspired by Constitutional AI's multi-agent debate and AlphaFold's MSA processing.
"""

import json
import os
import sys
import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any, Tuple
from enum import Enum, auto
from pathlib import Path

# Add script directory to path for local imports
SCRIPT_DIR = Path(__file__).parent.resolve()
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from pcb_state import (
    PCBState, PCBModification, ModificationType,
    ParameterSpace, DRCResult
)


class AgentRole(Enum):
    """Roles for specialized agents."""
    SIGNAL_INTEGRITY = auto()
    THERMAL_POWER = auto()
    MANUFACTURING = auto()
    GENERAL = auto()


@dataclass
class AgentConfig:
    """Configuration for a generator agent."""
    name: str
    role: AgentRole
    model: str = "claude-opus-4-6-20260206"
    temperature: float = 0.7
    max_tokens: int = 4096
    focus_areas: List[str] = field(default_factory=list)
    constraints: Dict[str, Any] = field(default_factory=dict)


@dataclass
class GenerationResult:
    """Result from an agent's generation attempt."""
    agent_name: str
    modifications: List[PCBModification]
    reasoning: str
    confidence: float
    focus_violations: List[str]
    estimated_improvement: float


class GeneratorAgent(ABC):
    """
    Base class for PCB optimization agents.

    Each agent specializes in a domain and generates modifications
    based on its expertise. Agents compete via tournament judging.
    """

    def __init__(self, config: AgentConfig):
        """
        Initialize generator agent.

        Args:
            config: Agent configuration
        """
        self.config = config
        self.name = config.name
        self.role = config.role
        self.model = config.model
        self.elo_rating = 1200  # Starting Elo rating
        self.generation_count = 0
        self.successful_fixes = 0

    @abstractmethod
    def get_system_prompt(self) -> str:
        """Get the system prompt for this agent's specialty."""
        pass

    @abstractmethod
    def get_focus_violations(self, drc_result: DRCResult) -> List[Dict]:
        """Filter DRC violations relevant to this agent's focus."""
        pass

    async def generate_modifications(
        self,
        pcb_state: PCBState,
        drc_result: DRCResult,
        max_modifications: int = 5
    ) -> GenerationResult:
        """
        Generate candidate modifications for the PCB.

        Args:
            pcb_state: Current PCB state
            drc_result: Current DRC results
            max_modifications: Maximum modifications to generate

        Returns:
            GenerationResult with modifications and reasoning
        """
        self.generation_count += 1

        # Get violations this agent cares about
        focus_violations = self.get_focus_violations(drc_result)

        if not focus_violations:
            return GenerationResult(
                agent_name=self.name,
                modifications=[],
                reasoning="No violations in my focus area",
                confidence=0.0,
                focus_violations=[],
                estimated_improvement=0.0
            )

        # Build the prompt
        prompt = self._build_generation_prompt(pcb_state, drc_result, focus_violations)

        # Call LLM
        response = await self._call_llm(prompt)

        # Parse response into modifications
        modifications, reasoning, confidence = self._parse_response(response)

        # Estimate improvement
        estimated_improvement = len(modifications) * 0.1 * confidence

        return GenerationResult(
            agent_name=self.name,
            modifications=modifications[:max_modifications],
            reasoning=reasoning,
            confidence=confidence,
            focus_violations=[str(v) for v in focus_violations[:5]],
            estimated_improvement=estimated_improvement
        )

    def _build_generation_prompt(
        self,
        pcb_state: PCBState,
        drc_result: DRCResult,
        focus_violations: List[Dict]
    ) -> str:
        """Build the generation prompt with state context."""
        # Summarize PCB state
        state_summary = f"""
PCB STATE SUMMARY:
- Components: {len(pcb_state.components)}
- Traces: {len(pcb_state.traces)}
- Vias: {len(pcb_state.vias)}
- Zones: {len(pcb_state.zones)}
- Nets: {len(pcb_state.nets)}

CURRENT PARAMETERS:
{json.dumps(pcb_state.parameters, indent=2)}

DRC RESULTS:
- Total violations: {drc_result.total_violations}
- Errors: {drc_result.errors}
- Warnings: {drc_result.warnings}
- Unconnected: {drc_result.unconnected}

TOP VIOLATION TYPES:
{json.dumps(dict(sorted(drc_result.violations_by_type.items(), key=lambda x: -x[1])[:5]), indent=2)}

VIOLATIONS IN YOUR FOCUS AREA ({len(focus_violations)}):
{json.dumps(focus_violations[:10], indent=2)}
"""

        # Parameter bounds
        bounds_info = "\nPARAMETER BOUNDS:\n"
        for name, bounds in ParameterSpace.get_all_bounds().items():
            bounds_info += f"  {name}: [{bounds.min_value}, {bounds.max_value}] mm\n"

        return f"""
{state_summary}
{bounds_info}

Generate {3-5} specific modifications to reduce the violations in your focus area.
Each modification should be:
1. Targeted at a specific violation
2. Within the parameter bounds
3. Justified with reasoning

Return your response as JSON:
{{
  "reasoning": "High-level reasoning about the approach",
  "confidence": 0.0-1.0,
  "modifications": [
    {{
      "type": "MOVE_COMPONENT|ADJUST_TRACE_WIDTH|ADJUST_VIA_SIZE|ADJUST_CLEARANCE|ADJUST_ZONE",
      "target": "component_ref or net_name or zone_name",
      "parameters": {{"key": "value"}},
      "justification": "Why this helps"
    }}
  ]
}}
"""

    async def _call_llm(self, prompt: str) -> str:
        """Call the LLM with the given prompt."""
        # Check for API key
        api_key = os.environ.get('ANTHROPIC_API_KEY')

        if api_key:
            # Use real Anthropic API
            try:
                import anthropic
                client = anthropic.Anthropic(api_key=api_key)

                message = client.messages.create(
                    model=self.model,
                    max_tokens=self.config.max_tokens,
                    temperature=self.config.temperature,
                    system=self.get_system_prompt(),
                    messages=[{"role": "user", "content": prompt}]
                )

                return message.content[0].text
            except Exception as e:
                print(f"LLM call failed: {e}")
                return self._generate_fallback_response(prompt)
        else:
            # Generate heuristic-based response
            return self._generate_fallback_response(prompt)

    def _generate_fallback_response(self, prompt: str) -> str:
        """Generate a heuristic-based response when API unavailable."""
        # Extract violation types from prompt
        import re
        violations = re.findall(r'"(\w+)":\s*(\d+)', prompt)

        modifications = []

        # Generate modifications based on agent role
        if self.role == AgentRole.SIGNAL_INTEGRITY:
            modifications = [
                {
                    "type": "ADJUST_CLEARANCE",
                    "target": "signal_clearance",
                    "parameters": {"param_name": "signal_clearance", "value": 0.18},
                    "justification": "Increase signal clearance to reduce crosstalk"
                },
                {
                    "type": "ADJUST_TRACE_WIDTH",
                    "target": "signal",
                    "parameters": {"width": 0.22},
                    "justification": "Wider traces for better impedance control"
                }
            ]
        elif self.role == AgentRole.THERMAL_POWER:
            modifications = [
                {
                    "type": "ADJUST_VIA_SIZE",
                    "target": "GND",
                    "parameters": {"diameter": 1.0, "drill": 0.5},
                    "justification": "Larger thermal vias for heat dissipation"
                },
                {
                    "type": "ADJUST_ZONE",
                    "target": "GND_POWER",
                    "parameters": {"clearance": 0.3, "thermal_gap": 0.4},
                    "justification": "Better thermal connection to ground plane"
                }
            ]
        elif self.role == AgentRole.MANUFACTURING:
            modifications = [
                {
                    "type": "ADJUST_CLEARANCE",
                    "target": "zone_clearance",
                    "parameters": {"param_name": "zone_clearance", "value": 0.25},
                    "justification": "Improve manufacturability with larger zone clearance"
                },
                {
                    "type": "ADJUST_VIA_SIZE",
                    "target": "signal",
                    "parameters": {"diameter": 0.8, "drill": 0.4},
                    "justification": "Standard via size for better yield"
                },
                # MAPOS Phase 2: Footprint-level DFM fixes
                {
                    "type": "ADJUST_SOLDER_MASK",
                    "target": "all_smd_components",
                    "parameters": {"expansion_mm": -0.03},
                    "justification": "Reduce solder mask opening to create mask dam between pads"
                },
                {
                    "type": "MOVE_SILKSCREEN",
                    "target": "all_silk_over_copper",
                    "parameters": {"offset_y_mm": 1.5, "element": "reference"},
                    "justification": "Move silkscreen references away from copper pads"
                },
                {
                    "type": "ADJUST_SILKSCREEN",
                    "target": "all_components",
                    "parameters": {"text_size_mm": 0.8, "move_to_courtyard": True},
                    "justification": "Shrink and reposition silkscreen to avoid overlap"
                }
            ]

        return json.dumps({
            "reasoning": f"Heuristic-based modifications for {self.role.name} optimization",
            "confidence": 0.6,
            "modifications": modifications
        })

    def _parse_response(self, response: str) -> Tuple[List[PCBModification], str, float]:
        """Parse LLM response into modifications."""
        try:
            # Try to extract JSON from response
            import re
            json_match = re.search(r'\{[\s\S]*\}', response)
            if json_match:
                data = json.loads(json_match.group())
            else:
                return [], "Failed to parse response", 0.0

            reasoning = data.get('reasoning', '')
            confidence = float(data.get('confidence', 0.5))
            raw_mods = data.get('modifications', [])

            modifications = []
            for mod in raw_mods:
                try:
                    mod_type = ModificationType[mod['type']]
                    modifications.append(PCBModification(
                        mod_type=mod_type,
                        target=mod['target'],
                        parameters=mod['parameters'],
                        description=mod.get('justification', ''),
                        source_agent=self.name,
                        confidence=confidence
                    ))
                except (KeyError, ValueError) as e:
                    print(f"Failed to parse modification: {e}")
                    continue

            return modifications, reasoning, confidence

        except json.JSONDecodeError as e:
            print(f"JSON parse error: {e}")
            return [], "JSON parse error", 0.0

    def update_elo(self, opponent_elo: float, won: bool, k_factor: float = 32) -> float:
        """
        Update Elo rating after a match.

        Args:
            opponent_elo: Opponent's Elo rating
            won: Whether this agent won
            k_factor: Elo K-factor

        Returns:
            New Elo rating
        """
        expected = 1 / (1 + 10**((opponent_elo - self.elo_rating) / 400))
        actual = 1 if won else 0
        self.elo_rating += k_factor * (actual - expected)
        return self.elo_rating

    def record_success(self, violations_fixed: int) -> None:
        """Record successful fix for statistics."""
        self.successful_fixes += violations_fixed

    @property
    def success_rate(self) -> float:
        """Get agent's success rate."""
        if self.generation_count == 0:
            return 0.0
        return self.successful_fixes / self.generation_count

    def __repr__(self) -> str:
        return f"{self.name}(Elo={self.elo_rating:.0f}, success={self.success_rate:.1%})"


class SignalIntegrityAgent(GeneratorAgent):
    """
    Agent specializing in signal integrity optimization.

    Focus areas:
    - Trace length matching (differential pairs)
    - Impedance control
    - Crosstalk reduction
    - Return path continuity
    """

    def __init__(self):
        config = AgentConfig(
            name="SignalIntegritySpecialist",
            role=AgentRole.SIGNAL_INTEGRITY,
            focus_areas=[
                "clearance", "trace_width", "differential_pairs",
                "impedance", "crosstalk", "return_path"
            ],
            constraints={
                "min_clearance_mm": 0.1,
                "target_impedance_ohm": 50,
                "max_length_mismatch_pct": 5
            }
        )
        super().__init__(config)

    def get_system_prompt(self) -> str:
        return """You are a signal integrity specialist optimizing PCB layouts.

EXPERTISE:
- High-speed digital signal routing
- Differential pair matching
- Impedance control (50Ω single-ended, 100Ω differential)
- Crosstalk and noise reduction
- Return path optimization

FOCUS AREAS:
1. Trace length matching for differential pairs
2. Impedance-controlled routing
3. Adequate spacing between signals
4. Clean reference planes
5. Via placement for layer transitions

CONSTRAINTS:
- Minimum clearance: 0.1mm (IPC Class 2)
- Maximum trace length mismatch: 5%
- Maintain continuous return paths
- Avoid routing over splits in reference planes

Generate modifications that improve signal integrity while reducing DRC violations.
Prioritize changes that address multiple issues simultaneously."""

    def get_focus_violations(self, drc_result: DRCResult) -> List[Dict]:
        """Filter for signal integrity related violations."""
        focus_types = {
            'clearance', 'track_width', 'track_dangling',
            'silk_over_copper', 'copper_edge_clearance'
        }

        violations = []
        for v in drc_result.top_violations:
            vtype = v.get('type', '')
            if vtype in focus_types:
                violations.append(v)

        # Also check by_type counts
        for vtype in focus_types:
            if vtype in drc_result.violations_by_type:
                count = drc_result.violations_by_type[vtype]
                if count > 0 and not any(v.get('type') == vtype for v in violations):
                    violations.append({'type': vtype, 'count': count})

        return violations


class ThermalPowerAgent(GeneratorAgent):
    """
    Agent specializing in thermal and power integrity optimization.

    Focus areas:
    - Heat dissipation paths
    - Current distribution
    - Thermal via placement
    - Ground plane effectiveness
    """

    def __init__(self):
        config = AgentConfig(
            name="ThermalPowerSpecialist",
            role=AgentRole.THERMAL_POWER,
            focus_areas=[
                "thermal_vias", "power_planes", "current_capacity",
                "heat_spreading", "ground_integrity"
            ],
            constraints={
                "max_current_density_A_mm2": 35,
                "thermal_via_spacing_mm": 1.5,
                "min_power_trace_mm": 1.0
            }
        )
        super().__init__(config)

    def get_system_prompt(self) -> str:
        return """You are a thermal and power integrity specialist.

EXPERTISE:
- Power distribution network design
- Thermal management
- High-current trace sizing
- Via thermal resistance
- Ground plane optimization

FOCUS AREAS:
1. Heat dissipation from power components
2. Current carrying capacity
3. Thermal via arrays
4. Power/ground plane integrity
5. Voltage drop minimization

BOARD CONTEXT:
- High voltage: +58V main bus
- High current: Up to 100A total
- Power nets: +58V, +12V, +5V, +3V3
- Critical thermal components: MOSFETs, regulators

CONSTRAINTS:
- Max current density: 35 A/mm² (external layers)
- Thermal via spacing: 1.0-2.0mm grid
- Minimum power trace: 1.0mm for low current, 2.0mm+ for high

Generate modifications that improve thermal performance and power delivery
while reducing DRC violations related to power nets."""

    def get_focus_violations(self, drc_result: DRCResult) -> List[Dict]:
        """Filter for thermal/power related violations."""
        focus_types = {
            'shorting_items', 'via_dangling', 'zone_clearance',
            'annular_width', 'hole_clearance'
        }

        violations = []
        for v in drc_result.top_violations:
            vtype = v.get('type', '')
            desc = v.get('description', '').lower()

            # Check type or description for power-related issues
            if vtype in focus_types:
                violations.append(v)
            elif any(net in desc for net in ['+58v', '+12v', '+5v', 'gnd', 'power', 'vbus']):
                violations.append(v)

        return violations


class ManufacturingAgent(GeneratorAgent):
    """
    Agent specializing in DFM (Design for Manufacturing) optimization.

    Focus areas:
    - Solder mask apertures
    - Silkscreen placement
    - Via drill sizes
    - Courtyard clearances
    - Assembly considerations
    """

    def __init__(self):
        config = AgentConfig(
            name="ManufacturingSpecialist",
            role=AgentRole.MANUFACTURING,
            focus_areas=[
                "solder_mask", "silkscreen", "courtyard",
                "via_drill", "assembly", "cost"
            ],
            constraints={
                "min_drill_mm": 0.3,
                "min_annular_ring_mm": 0.1,
                "min_silk_clearance_mm": 0.15,
                "layer_count": 10
            }
        )
        super().__init__(config)

    def get_system_prompt(self) -> str:
        return """You are a DFM (Design for Manufacturing) specialist.

EXPERTISE:
- PCB fabrication constraints
- Assembly process optimization
- Solder paste/mask design
- Silkscreen layout
- Cost optimization

FOCUS AREAS:
1. Solder mask bridge prevention
2. Silkscreen legibility and placement
3. Component courtyard clearances
4. Via drill optimization for yield
5. Panelization considerations

MANUFACTURING CONSTRAINTS:
- Minimum drill: 0.3mm
- Minimum annular ring: 0.1mm (IPC Class 2)
- Minimum solder mask dam: 0.1mm
- Silkscreen to copper clearance: 0.15mm
- 10-layer board (affects via cost)

Generate modifications that improve manufacturability and reduce
fabrication-related DRC violations. Prioritize changes that:
- Reduce via count where possible
- Improve silkscreen clarity
- Prevent solder bridging
- Maintain assembly clearances"""

    def get_focus_violations(self, drc_result: DRCResult) -> List[Dict]:
        """Filter for manufacturing related violations."""
        focus_types = {
            'solder_mask_bridge', 'silk_over_copper', 'silk_overlap',
            'courtyards_overlap', 'lib_footprint_mismatch',
            'lib_footprint_issues', 'hole_clearance'
        }

        violations = []
        for v in drc_result.top_violations:
            vtype = v.get('type', '')
            if vtype in focus_types:
                violations.append(v)

        for vtype in focus_types:
            if vtype in drc_result.violations_by_type:
                count = drc_result.violations_by_type[vtype]
                if count > 0 and not any(v.get('type') == vtype for v in violations):
                    violations.append({'type': vtype, 'count': count})

        return violations


class AgentPool:
    """
    Pool of generator agents for parallel generation.

    Manages multiple agents and coordinates their generation attempts.
    """

    def __init__(self, agents: Optional[List[GeneratorAgent]] = None):
        """
        Initialize agent pool.

        Args:
            agents: List of agents (creates defaults if None)
        """
        if agents is None:
            self.agents = [
                SignalIntegrityAgent(),
                ThermalPowerAgent(),
                ManufacturingAgent()
            ]
        else:
            self.agents = agents

    async def generate_all(
        self,
        pcb_state: PCBState,
        drc_result: DRCResult,
        max_modifications_per_agent: int = 5
    ) -> List[GenerationResult]:
        """
        Generate modifications from all agents in parallel.

        Args:
            pcb_state: Current PCB state
            drc_result: Current DRC results
            max_modifications_per_agent: Max modifications per agent

        Returns:
            List of GenerationResults from all agents
        """
        tasks = [
            agent.generate_modifications(
                pcb_state, drc_result, max_modifications_per_agent
            )
            for agent in self.agents
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Filter out exceptions
        valid_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                print(f"Agent {self.agents[i].name} failed: {result}")
            else:
                valid_results.append(result)

        return valid_results

    def get_top_agents(self, n: int = 3) -> List[GeneratorAgent]:
        """Get top N agents by Elo rating."""
        return sorted(self.agents, key=lambda a: a.elo_rating, reverse=True)[:n]

    def get_agent_by_role(self, role: AgentRole) -> Optional[GeneratorAgent]:
        """Get agent by role."""
        for agent in self.agents:
            if agent.role == role:
                return agent
        return None

    def add_agent(self, agent: GeneratorAgent) -> None:
        """Add an agent to the pool."""
        self.agents.append(agent)

    def get_statistics(self) -> Dict[str, Any]:
        """Get pool statistics."""
        return {
            'agent_count': len(self.agents),
            'total_generations': sum(a.generation_count for a in self.agents),
            'total_successes': sum(a.successful_fixes for a in self.agents),
            'agents': [
                {
                    'name': a.name,
                    'role': a.role.name,
                    'elo': a.elo_rating,
                    'generations': a.generation_count,
                    'successes': a.successful_fixes
                }
                for a in sorted(self.agents, key=lambda x: x.elo_rating, reverse=True)
            ]
        }


def create_default_agent_pool() -> AgentPool:
    """Create default agent pool with all specialists."""
    return AgentPool()


async def run_agent_generation(
    pcb_path: str,
    output_json: Optional[str] = None
) -> List[GenerationResult]:
    """
    Run all agents on a PCB file.

    Args:
        pcb_path: Path to PCB file
        output_json: Optional path to save results

    Returns:
        List of GenerationResults
    """
    from pcb_state import PCBState

    print(f"\nLoading PCB: {pcb_path}")
    state = PCBState.from_file(pcb_path)

    print(f"Running DRC...")
    drc = state.run_drc()
    print(f"  Violations: {drc.total_violations}")

    print(f"\nInitializing agent pool...")
    pool = create_default_agent_pool()

    print(f"Generating modifications from {len(pool.agents)} agents...")
    results = await pool.generate_all(state, drc)

    print(f"\nResults:")
    for result in results:
        print(f"  {result.agent_name}:")
        print(f"    Modifications: {len(result.modifications)}")
        print(f"    Confidence: {result.confidence:.2f}")
        print(f"    Focus violations: {len(result.focus_violations)}")

    if output_json:
        output_data = {
            'pcb_path': pcb_path,
            'drc_violations': drc.total_violations,
            'results': [
                {
                    'agent': r.agent_name,
                    'modifications': [m.to_dict() for m in r.modifications],
                    'reasoning': r.reasoning,
                    'confidence': r.confidence
                }
                for r in results
            ]
        }
        with open(output_json, 'w') as f:
            json.dump(output_data, f, indent=2)
        print(f"\nSaved results to: {output_json}")

    return results


if __name__ == '__main__':
    import sys

    if len(sys.argv) < 2:
        print("Usage: python generator_agents.py <path_to.kicad_pcb> [output.json]")
        sys.exit(1)

    pcb_path = sys.argv[1]
    output_json = sys.argv[2] if len(sys.argv) > 2 else None

    print("\n" + "="*60)
    print("GENERATOR AGENTS - Multi-Agent PCB Optimization")
    print("="*60)

    asyncio.run(run_agent_generation(pcb_path, output_json))
