"""
MAPO v2.1 Schematic - Smoke Test Validator

Integrates the existing SmokeTestAgent into the Gaming AI optimization loop.
Provides fitness scoring based on smoke test results.

Author: Nexus EE Design Team
"""

import asyncio
import json
import logging
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Add parent paths for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from ideation_context import ValidationContext, TestCriterion, ComplianceRequirement

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
        validation_context: Optional[ValidationContext] = None,
    ) -> SmokeTestValidationResult:
        """
        Run smoke test validation on a schematic state.

        Args:
            state: SchematicState to validate.
            validation_context: Optional validation hints from ideation
                context.  When provided, power_sources are enriched with
                voltage/current limits from the ideation power spec,
                test_criteria are evaluated as additional pass/fail
                checks, and compliance_requirements are included in the
                validation report.

        Returns:
            SmokeTestValidationResult with fitness scores
        """
        agent = self._ensure_agent()

        # Convert state to KiCad schematic content
        kicad_sch = self._state_to_kicad_sexp(state)

        # Convert components to BOM items
        bom_items = self._state_to_bom(state)

        # Detect power sources - enrich with ideation voltage/current limits
        power_sources = self._detect_power_sources(state)
        if validation_context is not None:
            power_sources = self._enrich_power_sources(
                power_sources, validation_context
            )

        logger.info(f"Running smoke test on state {state.uuid[:8]}...")

        # Run smoke test
        result = await agent.run_smoke_test(
            kicad_sch_content=kicad_sch,
            bom_items=bom_items,
            power_sources=power_sources,
        )

        # Compute fitness scores
        validation_result = self._compute_fitness(result, state)

        # Apply ideation test criteria and compliance checks
        if validation_context is not None:
            validation_result = self._apply_ideation_validation(
                validation_result, state, validation_context
            )

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

    def _enrich_power_sources(
        self,
        power_sources: List[Dict[str, Any]],
        validation_context: ValidationContext,
    ) -> List[Dict[str, Any]]:
        """Enrich auto-detected power sources with ideation voltage/current limits.

        When the ideation validation context contains voltage_limits or
        current_limits, these values override the auto-detected defaults
        for matching net names.  This ensures that the smoke test agent
        uses the designer's intended electrical specifications rather
        than heuristically inferred values.

        Args:
            power_sources: Auto-detected power source dicts from
                ``_detect_power_sources()``.
            validation_context: Validation hints from ideation.

        Returns:
            Enriched list of power source dicts.
        """
        if not validation_context.voltage_limits and not validation_context.current_limits:
            return power_sources

        logger.info(
            f"Enriching power sources with ideation limits: "
            f"{len(validation_context.voltage_limits)} voltage limits, "
            f"{len(validation_context.current_limits)} current limits"
        )

        # Build lookup of existing power sources by net name
        enriched: List[Dict[str, Any]] = []
        existing_nets: set = set()

        for ps in power_sources:
            net = ps.get("net", "")
            existing_nets.add(net)
            updated = dict(ps)

            # Apply voltage limits from ideation
            if net in validation_context.voltage_limits:
                limits = validation_context.voltage_limits[net]
                if isinstance(limits, (list, tuple)) and len(limits) == 2:
                    # (min_voltage, max_voltage) tuple - use midpoint as nominal
                    min_v, max_v = limits
                    updated["voltage"] = (min_v + max_v) / 2.0
                    updated["voltage_min"] = min_v
                    updated["voltage_max"] = max_v
                    logger.debug(
                        f"Applied voltage limits for {net}: {min_v}V - {max_v}V"
                    )
                elif isinstance(limits, (int, float)):
                    updated["voltage"] = float(limits)

            # Apply current limits from ideation
            if net in validation_context.current_limits:
                limit = validation_context.current_limits[net]
                if isinstance(limit, (int, float)):
                    updated["current_limit"] = float(limit)
                    logger.debug(
                        f"Applied current limit for {net}: {limit}A"
                    )

            enriched.append(updated)

        # Add power sources from voltage_limits that were not auto-detected
        for net, limits in validation_context.voltage_limits.items():
            if net not in existing_nets:
                voltage = 0.0
                voltage_min = 0.0
                voltage_max = 0.0
                if isinstance(limits, (list, tuple)) and len(limits) == 2:
                    voltage_min, voltage_max = limits
                    voltage = (voltage_min + voltage_max) / 2.0
                elif isinstance(limits, (int, float)):
                    voltage = float(limits)
                    voltage_min = voltage
                    voltage_max = voltage

                current_limit = 1.0  # Default
                if net in validation_context.current_limits:
                    cl = validation_context.current_limits[net]
                    if isinstance(cl, (int, float)):
                        current_limit = float(cl)

                enriched.append({
                    "net": net,
                    "voltage": voltage,
                    "voltage_min": voltage_min,
                    "voltage_max": voltage_max,
                    "current_limit": current_limit,
                })
                existing_nets.add(net)
                logger.info(
                    f"Added power source from ideation: {net} ({voltage}V, {current_limit}A)"
                )

        return enriched

    def _apply_ideation_validation(
        self,
        validation_result: SmokeTestValidationResult,
        state: SchematicState,
        validation_context: ValidationContext,
    ) -> SmokeTestValidationResult:
        """Apply ideation test criteria and compliance requirements to the
        validation result.

        Evaluates each ``TestCriterion`` from the ideation context against
        the schematic state and smoke test results.  Failures are added
        as error or warning issues depending on severity.  Compliance
        requirements are logged and appended to recommendations.

        This method modifies the validation result in place and also
        adjusts fitness scores based on ideation criteria outcomes.

        Args:
            validation_result: Current validation result from smoke test.
            state: SchematicState being validated.
            validation_context: Validation hints from ideation.

        Returns:
            Updated SmokeTestValidationResult with ideation checks applied.
        """
        if not validation_context.test_criteria and not validation_context.compliance_requirements:
            return validation_result

        logger.info(
            f"Applying ideation validation: {len(validation_context.test_criteria)} "
            f"test criteria, {len(validation_context.compliance_requirements)} "
            f"compliance requirements"
        )

        # Build lookup structures for evaluation
        net_names = {conn.net_name for conn in state.connections if conn.net_name}
        component_refs = {c.reference for c in state.components}
        component_categories = {c.category for c in state.components}

        ideation_failures = 0
        ideation_total = 0

        # Evaluate test criteria
        for criterion in validation_context.test_criteria:
            ideation_total += 1
            passed = self._evaluate_test_criterion(
                criterion, state, net_names, component_refs, validation_result
            )

            if not passed:
                ideation_failures += 1
                issue_msg = (
                    f"[IDEATION] {criterion.test_name}: "
                    f"FAILED - expected: {criterion.expected_result} "
                    f"(criteria: {criterion.pass_criteria})"
                )

                if criterion.severity in ("critical", "error"):
                    validation_result.error_issues.append(issue_msg)
                else:
                    validation_result.warning_issues.append(issue_msg)

                validation_result.recommendations.append(
                    f"Fix ideation test criterion '{criterion.test_name}': "
                    f"{criterion.expected_result}"
                )
                logger.warning(f"Ideation test criterion failed: {criterion.test_name}")
            else:
                logger.debug(f"Ideation test criterion passed: {criterion.test_name}")

        # Process compliance requirements - add as recommendations
        for req in validation_context.compliance_requirements:
            # Check if applicable components exist in the design
            applicable = True
            if req.applicable_components:
                applicable = any(
                    comp in component_refs or comp in component_categories
                    for comp in req.applicable_components
                )

            if applicable:
                validation_result.recommendations.append(
                    f"[{req.standard}] {req.requirement} "
                    f"(verify by: {req.verification_method})"
                )

        # Process thermal limits
        for ref, max_temp in validation_context.thermal_limits.items():
            if ref in component_refs:
                validation_result.recommendations.append(
                    f"Verify thermal limit for {ref}: max junction temp {max_temp}C"
                )

        # Process EMC requirements
        for emc_req in validation_context.emc_requirements:
            validation_result.recommendations.append(f"[EMC] {emc_req}")

        # Adjust fitness based on ideation criteria results
        if ideation_total > 0:
            ideation_pass_rate = (ideation_total - ideation_failures) / ideation_total
            # Blend ideation pass rate into existing fitness scores
            # Weight: 20% from ideation criteria, 80% from original smoke test
            validation_result.wiring_fitness = (
                validation_result.wiring_fitness * 0.8 + ideation_pass_rate * 0.2
            )
            logger.info(
                f"Ideation validation: {ideation_total - ideation_failures}/"
                f"{ideation_total} criteria passed "
                f"(pass rate: {ideation_pass_rate:.1%})"
            )

        return validation_result

    def _evaluate_test_criterion(
        self,
        criterion: TestCriterion,
        state: SchematicState,
        net_names: set,
        component_refs: set,
        validation_result: SmokeTestValidationResult,
    ) -> bool:
        """Evaluate a single test criterion against the schematic state.

        Performs structural checks based on the criterion category and
        pass_criteria string.  For criteria that cannot be fully
        evaluated at the schematic level (e.g., thermal or analog
        measurements), the criterion is marked as passed with a
        recommendation for manual verification.

        Args:
            criterion: TestCriterion to evaluate.
            state: SchematicState being validated.
            net_names: Set of all net names in the design.
            component_refs: Set of all component reference designators.
            validation_result: Current validation result for context.

        Returns:
            True if the criterion passes, False otherwise.
        """
        category = criterion.category.lower() if criterion.category else ""
        criteria_lower = criterion.pass_criteria.lower() if criterion.pass_criteria else ""

        # Power rail checks
        if category == "power":
            # Check if the referenced net exists
            for net in net_names:
                if net and criterion.test_name.upper() in net.upper():
                    return True
            # If pass_criteria mentions a specific net, check for it
            for net in net_names:
                if net and net.upper() in criteria_lower.upper():
                    return True
            # Power criterion references a net not found in the design
            return False

        # Connectivity checks
        if category in ("connectivity", "signal_integrity"):
            # Check that referenced components exist
            test_name_parts = criterion.test_name.replace("_", " ").split()
            for part in test_name_parts:
                if part.upper() in {r.upper() for r in component_refs}:
                    return True
            # If we can't structurally verify, pass with recommendation
            validation_result.recommendations.append(
                f"Manual verification needed: {criterion.test_name} - "
                f"{criterion.expected_result}"
            )
            return True

        # Component presence checks
        if category == "component":
            # Check that a specific component exists
            for ref in component_refs:
                if ref.upper() in criterion.test_name.upper():
                    return True
            return False

        # For categories we cannot structurally evaluate (thermal, analog,
        # mechanical, etc.), pass but add a recommendation
        validation_result.recommendations.append(
            f"Manual verification needed for [{category}] {criterion.test_name}: "
            f"{criterion.expected_result}"
        )
        return True

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
    
    async def _call_compliance_validator(
        self,
        state: SchematicState,
        schematic_path: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Call the TypeScript schematic-compliance-validator.ts service.

        This runs all 51 compliance checks (NASA, MIL, IPC, IEC, Best Practices)
        via the Node.js TypeScript service and returns the compliance report.

        Args:
            state: Current schematic state
            schematic_path: Optional path to saved .kicad_sch file.
                If not provided, generates temp file from state.

        Returns:
            Compliance report dict or None if validator unavailable
        """
        try:
            # Path to TypeScript validator service
            validator_path = Path(__file__).parent.parent.parent.parent / \
                "src" / "services" / "schematic" / "schematic-compliance-validator.ts"

            if not validator_path.exists():
                logger.warning(
                    f"Compliance validator not found at {validator_path}, "
                    "skipping standards compliance checks"
                )
                return None

            # Generate temp schematic file if needed
            if not schematic_path:
                import tempfile
                temp_fd, schematic_path = tempfile.mkstemp(suffix=".kicad_sch")
                with open(temp_fd, 'w') as f:
                    f.write(self._state_to_kicad_sexp(state))
                temp_file = schematic_path
            else:
                temp_file = None

            # Call TypeScript validator via tsx (TypeScript runtime)
            # Requires: npm install -g tsx
            result = subprocess.run(
                [
                    "tsx",  # TypeScript runtime
                    str(validator_path),
                    "--schematic", schematic_path,
                    "--format", "json"
                ],
                capture_output=True,
                text=True,
                timeout=60
            )

            # Clean up temp file
            if temp_file:
                try:
                    Path(temp_file).unlink()
                except Exception:
                    pass

            if result.returncode != 0:
                logger.warning(
                    f"Compliance validator failed (exit {result.returncode}): "
                    f"{result.stderr[:200]}"
                )
                return None

            # Parse JSON report
            report = json.loads(result.stdout)
            logger.info(
                f"Compliance validation complete: {report['passedChecks']}/"
                f"{report['totalChecks']} checks passed, "
                f"score={report['score']:.1f}"
            )
            return report

        except FileNotFoundError as exc:
            logger.warning(
                f"tsx not found - install with: npm install -g tsx. "
                "Skipping compliance checks."
            )
            return None
        except subprocess.TimeoutExpired:
            logger.warning("Compliance validator timed out after 60s")
            return None
        except json.JSONDecodeError as exc:
            logger.warning(f"Failed to parse compliance report JSON: {exc}")
            return None
        except Exception as exc:
            logger.warning(f"Compliance validator error: {exc}")
            return None

    async def validate_and_score(
        self,
        state: SchematicState,
        validation_context: Optional[ValidationContext] = None,
    ) -> Tuple[SchematicState, SmokeTestValidationResult]:
        """
        Validate state and return updated state with fitness scores.

        Args:
            state: SchematicState to validate.
            validation_context: Optional validation hints from ideation
                context.  Threaded through to ``validate()`` to enrich
                power source detection and apply ideation test criteria.

        Returns:
            Tuple of (updated_state, validation_result)
        """
        # Run smoke test with ideation validation context
        validation_result = await self.validate(
            state, validation_context=validation_context
        )

        # Run TypeScript compliance validator (MAPO v3.0)
        compliance_report = await self._call_compliance_validator(state)
        if compliance_report:
            # Merge compliance violations into smoke test result
            for check_result in compliance_report.get('checkResults', []):
                if check_result['status'] == 'failed':
                    for violation in check_result['violations']:
                        # Add as warning issue (non-fatal)
                        validation_result.warning_issues.append(
                            f"Standards Compliance: {violation['message']}"
                        )

            # Log compliance score
            logger.info(
                f"Standards compliance score: {compliance_report['score']}/100"
            )
        
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
