"""
MAPO v2.1 Schematic - Smoke Test Validator

Integrates the existing SmokeTestAgent into the Gaming AI optimization loop.
Provides fitness scoring based on smoke test results.

Author: Nexus EE Design Team
"""

import asyncio
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Add parent paths for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from agents.smoke_test import SmokeTestAgent, SmokeTestResult, SmokeTestSeverity

from ..core.schematic_state import (
    SchematicState,
    ValidationResults,
    FitnessScores,
    SymbolQuality,
)

logger = logging.getLogger(__name__)


@dataclass
class SmokeTestValidationResult:
    """
    Result from smoke test validation in optimization loop.
    
    Extends SmokeTestResult with fitness scoring for Gaming AI.
    """
    # Original smoke test result
    smoke_test: SmokeTestResult
    
    # Fitness components
    wiring_fitness: float = 0.0
    power_fitness: float = 0.0
    connectivity_fitness: float = 0.0
    
    # Issue breakdown
    fatal_issues: List[str] = field(default_factory=list)
    error_issues: List[str] = field(default_factory=list)
    warning_issues: List[str] = field(default_factory=list)
    
    # Recommendations for improvement
    recommendations: List[str] = field(default_factory=list)
    
    @property
    def passed(self) -> bool:
        return self.smoke_test.passed
    
    @property
    def combined_fitness(self) -> float:
        """Combined smoke test fitness score."""
        return (
            self.wiring_fitness * 0.4 +
            self.power_fitness * 0.3 +
            self.connectivity_fitness * 0.3
        )
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "passed": self.passed,
            "fitness": {
                "wiring": self.wiring_fitness,
                "power": self.power_fitness,
                "connectivity": self.connectivity_fitness,
                "combined": self.combined_fitness,
            },
            "issues": {
                "fatal": self.fatal_issues,
                "error": self.error_issues,
                "warning": self.warning_issues,
            },
            "recommendations": self.recommendations,
        }


class SmokeTestValidator:
    """
    Smoke Test Validator for MAPO v2.1 Gaming AI optimization.
    
    Wraps the existing SmokeTestAgent and provides:
    - Fitness scoring for optimization
    - Integration with SchematicState
    - Recommendations for mutation guidance
    """
    
    def __init__(self, config: Optional[Any] = None):
        """
        Initialize the smoke test validator.
        
        Args:
            config: Optional SchematicMAPOConfig
        """
        self.config = config
        self._agent: Optional[SmokeTestAgent] = None
    
    def _ensure_agent(self) -> SmokeTestAgent:
        """Get or create the smoke test agent."""
        if self._agent is None:
            self._agent = SmokeTestAgent()
        return self._agent
    
    async def validate(
        self,
        state: SchematicState,
    ) -> SmokeTestValidationResult:
        """
        Run smoke test validation on a schematic state.
        
        Args:
            state: SchematicState to validate
        
        Returns:
            SmokeTestValidationResult with fitness scores
        """
        agent = self._ensure_agent()
        
        # Convert state to KiCad schematic content
        kicad_sch = self._state_to_kicad_sexp(state)
        
        # Convert components to BOM items
        bom_items = self._state_to_bom(state)
        
        # Detect power sources
        power_sources = self._detect_power_sources(state)
        
        logger.info(f"Running smoke test on state {state.uuid[:8]}...")
        
        # Run smoke test
        result = await agent.run_smoke_test(
            kicad_sch_content=kicad_sch,
            bom_items=bom_items,
            power_sources=power_sources,
        )
        
        # Compute fitness scores
        validation_result = self._compute_fitness(result, state)
        
        logger.info(
            f"Smoke test complete: passed={validation_result.passed}, "
            f"fitness={validation_result.combined_fitness:.2f}"
        )
        
        return validation_result
    
    def _state_to_kicad_sexp(self, state: SchematicState) -> str:
        """
        Convert SchematicState to KiCad S-expression format.
        
        This generates a minimal but valid KiCad schematic that can
        be analyzed by the smoke test agent.
        """
        lines = [
            "(kicad_sch",
            "  (version 20211123)",
            "  (generator mapo_v2_1)",
            "",
            '  (paper "A4")',
            "",
            "  (lib_symbols",
        ]
        
        # Add library symbols
        for symbol_id, symbol_content in state.lib_symbols.items():
            lines.append(f"    {symbol_content}")
        
        lines.append("  )")
        lines.append("")
        
        # Add component instances
        for comp in state.components:
            x, y = comp.position
            lines.extend([
                f"  (symbol",
                f'    (lib_id "{comp.symbol_id}")',
                f"    (at {x} {y} {comp.rotation})",
                f'    (uuid "{comp.uuid}")',
                f"    (property \"Reference\" \"{comp.reference}\")",
                f"    (property \"Value\" \"{comp.value}\")",
                f"  )",
            ])
        
        # Add wires
        for wire in state.wires:
            for segment in wire.segments:
                x1, y1 = segment.start
                x2, y2 = segment.end
                lines.append(f"  (wire (pts (xy {x1} {y1}) (xy {x2} {y2})))")
        
        # Add junctions
        for junction in state.junctions:
            x, y = junction.position
            lines.append(f"  (junction (at {x} {y}))")
        
        # Add net labels for named nets
        net_positions = {}
        for conn in state.connections:
            if conn.net_name and conn.net_name not in net_positions:
                # Find position from a wire on this net
                for wire in state.wires:
                    if wire.net_name == conn.net_name and wire.segments:
                        x, y = wire.segments[0].start
                        net_positions[conn.net_name] = (x, y)
                        break
        
        for net_name, (x, y) in net_positions.items():
            lines.append(f'  (label "{net_name}" (at {x} {y-2.54} 0))')
        
        lines.append(")")
        
        return "\n".join(lines)
    
    def _state_to_bom(self, state: SchematicState) -> List[Dict[str, Any]]:
        """Convert state components to BOM items."""
        bom = []
        for comp in state.components:
            bom.append({
                "reference": comp.reference,
                "part_number": comp.part_number,
                "manufacturer": comp.manufacturer,
                "category": comp.category,
                "value": comp.value,
                "footprint": comp.footprint,
            })
        return bom
    
    def _detect_power_sources(self, state: SchematicState) -> List[Dict[str, Any]]:
        """Detect power sources from connections and components."""
        power_sources = []
        
        # Common power net patterns
        power_nets = {"VCC", "VDD", "VIN", "V+", "5V", "3V3", "12V", "48V"}
        
        for conn in state.connections:
            if conn.net_name:
                net_upper = conn.net_name.upper()
                if any(p in net_upper for p in power_nets):
                    # Infer voltage from net name
                    voltage = 5.0  # Default
                    if "3V3" in net_upper or "3.3V" in net_upper:
                        voltage = 3.3
                    elif "12V" in net_upper:
                        voltage = 12.0
                    elif "48V" in net_upper:
                        voltage = 48.0
                    
                    power_sources.append({
                        "net": conn.net_name,
                        "voltage": voltage,
                        "current_limit": 1.0,  # Conservative default
                    })
        
        # Deduplicate
        seen = set()
        unique_sources = []
        for ps in power_sources:
            if ps["net"] not in seen:
                seen.add(ps["net"])
                unique_sources.append(ps)
        
        return unique_sources
    
    def _compute_fitness(
        self,
        result: SmokeTestResult,
        state: SchematicState,
    ) -> SmokeTestValidationResult:
        """
        Compute fitness scores from smoke test result.
        
        Fitness is computed based on:
        - Wiring fitness: Proportion of successful connectivity checks
        - Power fitness: Power rail and ground connectivity
        - Connectivity fitness: No floating nodes, valid current paths
        """
        # Count issues by severity
        fatal_issues = []
        error_issues = []
        warning_issues = []
        
        for issue in result.issues:
            msg = f"{issue.test_name}: {issue.message}"
            if issue.component:
                msg += f" ({issue.component})"
            
            if issue.severity == SmokeTestSeverity.FATAL:
                fatal_issues.append(msg)
            elif issue.severity == SmokeTestSeverity.ERROR:
                error_issues.append(msg)
            elif issue.severity == SmokeTestSeverity.WARNING:
                warning_issues.append(msg)
        
        # Compute wiring fitness
        # Penalize based on severity of issues
        fatal_penalty = len(fatal_issues) * 0.3
        error_penalty = len(error_issues) * 0.1
        warning_penalty = len(warning_issues) * 0.02
        
        wiring_fitness = max(0.0, 1.0 - fatal_penalty - error_penalty - warning_penalty)
        
        # Compute power fitness
        power_checks = [
            result.power_rails_ok,
            result.ground_ok,
            result.no_shorts,
            result.power_dissipation_ok,
        ]
        power_fitness = sum(1 for c in power_checks if c) / len(power_checks)
        
        # Compute connectivity fitness
        connectivity_checks = [
            result.no_floating_nodes,
            result.current_paths_valid,
        ]
        connectivity_fitness = sum(1 for c in connectivity_checks if c) / len(connectivity_checks)
        
        # Also consider wiring completeness from state
        if state.wiring_completeness < 1.0:
            connectivity_fitness *= state.wiring_completeness
        
        # Extract recommendations
        recommendations = []
        for issue in result.issues:
            if issue.recommendation:
                recommendations.append(issue.recommendation)
        
        return SmokeTestValidationResult(
            smoke_test=result,
            wiring_fitness=wiring_fitness,
            power_fitness=power_fitness,
            connectivity_fitness=connectivity_fitness,
            fatal_issues=fatal_issues,
            error_issues=error_issues,
            warning_issues=warning_issues,
            recommendations=recommendations,
        )
    
    def compute_overall_fitness(
        self,
        state: SchematicState,
        validation_result: SmokeTestValidationResult,
    ) -> FitnessScores:
        """
        Compute overall fitness scores combining multiple factors.
        
        Args:
            state: Current schematic state
            validation_result: Smoke test validation result
        
        Returns:
            FitnessScores for Gaming AI optimization
        """
        # Correctness: ERC + symbol quality + connection completeness
        erc_score = 1.0 if validation_result.smoke_test.no_shorts else 0.5
        
        # Symbol quality score
        total_components = len(state.components)
        if total_components > 0:
            verified = sum(1 for c in state.components if c.quality == SymbolQuality.VERIFIED)
            fetched = sum(1 for c in state.components if c.quality == SymbolQuality.FETCHED)
            placeholder = sum(1 for c in state.components if c.quality == SymbolQuality.PLACEHOLDER)
            
            symbol_score = (verified * 1.0 + fetched * 0.8) / total_components
            # Heavy penalty for placeholders
            symbol_score -= placeholder * 0.3
            symbol_score = max(0.0, min(1.0, symbol_score))
        else:
            symbol_score = 0.0
        
        correctness = (erc_score * 0.5 + symbol_score * 0.3 + state.wiring_completeness * 0.2)
        
        # Wiring: From smoke test validation
        wiring = validation_result.wiring_fitness
        
        # Simulation: From smoke test pass/fail
        if validation_result.passed:
            simulation = 1.0
        else:
            simulation = validation_result.combined_fitness * 0.5
        
        # Cost: Placeholder for now (could compute from BOM)
        cost = 0.7  # Default good score
        
        return FitnessScores(
            correctness=correctness,
            wiring=wiring,
            simulation=simulation,
            cost=cost,
        )
    
    async def validate_and_score(
        self,
        state: SchematicState,
    ) -> Tuple[SchematicState, SmokeTestValidationResult]:
        """
        Validate state and return updated state with fitness scores.
        
        Args:
            state: SchematicState to validate
        
        Returns:
            Tuple of (updated_state, validation_result)
        """
        # Run smoke test
        validation_result = await self.validate(state)
        
        # Compute overall fitness
        fitness = self.compute_overall_fitness(state, validation_result)
        
        # Create validation results
        validation = ValidationResults(
            erc_errors=tuple(validation_result.fatal_issues),
            erc_warnings=tuple(validation_result.warning_issues),
            erc_passed=len(validation_result.fatal_issues) == 0,
            bp_violations=tuple(),
            bp_score=validation_result.wiring_fitness,
            smoke_test_passed=validation_result.passed,
            smoke_test_errors=tuple(validation_result.error_issues),
            placeholder_count=state.placeholder_count,
            verified_count=sum(1 for c in state.components if c.quality == SymbolQuality.VERIFIED),
            fetched_count=sum(1 for c in state.components if c.quality == SymbolQuality.FETCHED),
        )
        
        # Update state with validation results
        updated_state = state.with_validation(validation, fitness)
        
        return updated_state, validation_result
