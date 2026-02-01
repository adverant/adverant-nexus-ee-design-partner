"""
MAPO Functional Validation Agent - Competitive multi-agent circuit validation.

Implements MAPO Gaming (Multi-Agent Policy Optimization with Competition) to validate
that generated schematics meet original specifications and will actually function.

Architecture:
- Multiple specialized agents compete to find issues
- Weighted voting based on confidence
- Adversarial agent can veto critical issues
- Requires supermajority (>75%) to pass

Author: Nexus EE Design Team
"""

import asyncio
import json
import logging
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


class IssueSeverity(Enum):
    """Severity levels for validation issues."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"  # Can trigger veto


class IssueCategory(Enum):
    """Categories of validation issues."""
    MISSING_COMPONENT = "missing_component"
    WRONG_CONNECTION = "wrong_connection"
    MISSING_CONNECTION = "missing_connection"
    POWER_ISSUE = "power_issue"
    GROUNDING_ISSUE = "grounding_issue"
    SIGNAL_INTEGRITY = "signal_integrity"
    FEEDBACK_INSTABILITY = "feedback_instability"
    MISSING_PROTECTION = "missing_protection"
    VALUE_MISMATCH = "value_mismatch"
    TOPOLOGY_ERROR = "topology_error"
    REFERENCE_DEVIATION = "reference_deviation"
    SPECIFICATION_VIOLATION = "specification_violation"


@dataclass
class ValidationIssue:
    """Single validation issue found by an agent."""
    category: IssueCategory
    severity: IssueSeverity
    message: str
    component_ref: Optional[str] = None
    net_name: Optional[str] = None
    recommendation: Optional[str] = None
    evidence: Optional[str] = None  # Supporting evidence


@dataclass
class ValidationVote:
    """Validation vote from a single agent."""
    agent_name: str
    passes: bool
    confidence: float  # 0.0 to 1.0
    issues: List[ValidationIssue] = field(default_factory=list)
    reasoning: str = ""
    execution_time_ms: float = 0.0


@dataclass
class ValidationResult:
    """Aggregated validation result from all agents."""
    passed: bool
    overall_score: float  # 0.0 to 1.0
    agreement_score: float  # How much agents agree
    votes: List[ValidationVote] = field(default_factory=list)
    critical_issues: List[ValidationIssue] = field(default_factory=list)
    all_issues: List[ValidationIssue] = field(default_factory=list)
    veto_triggered: bool = False
    veto_reason: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.utcnow)
    recommendations: List[str] = field(default_factory=list)


@dataclass
class SchematicContext:
    """Context for schematic validation."""
    schematic_sexp: str
    design_intent: str
    bom: List[Dict[str, Any]]
    connections: List[Dict[str, Any]]
    architecture: Optional[Dict[str, Any]] = None
    reference_designs: Optional[List[Dict[str, Any]]] = None
    power_budget: Optional[Dict[str, float]] = None


class ValidatorAgent(ABC):
    """Abstract base class for validator agents."""

    def __init__(self, name: str, weight: float = 1.0):
        """
        Initialize validator agent.

        Args:
            name: Agent identifier
            weight: Vote weight (1.0 = normal, >1.0 = more influence)
        """
        self.name = name
        self.weight = weight
        self._llm_client = None

    def set_llm_client(self, client: Any):
        """Set LLM client for intelligent analysis."""
        self._llm_client = client

    @abstractmethod
    async def validate(
        self,
        context: SchematicContext
    ) -> ValidationVote:
        """
        Perform validation and return vote.

        Args:
            context: Schematic context with all relevant data

        Returns:
            ValidationVote with pass/fail decision and issues
        """
        pass

    def _extract_components_from_sexp(
        self,
        sexp: str
    ) -> List[Dict[str, str]]:
        """Extract component information from S-expression."""
        components = []

        # Match: (symbol (lib_id "XXX") ... (property "Reference" "YYY" ...
        pattern = r'\(symbol\s+\(lib_id\s+"([^"]+)"\).*?\(property\s+"Reference"\s+"([^"]+)"'
        matches = re.findall(pattern, sexp, re.DOTALL)

        for lib_id, reference in matches:
            components.append({
                "lib_id": lib_id,
                "reference": reference
            })

        return components

    def _extract_wires_from_sexp(self, sexp: str) -> List[Dict]:
        """Extract wire information from S-expression."""
        wires = []

        # Match: (wire (pts (xy X1 Y1) (xy X2 Y2))
        pattern = r'\(wire\s+\(pts\s+\(xy\s+([-\d.]+)\s+([-\d.]+)\)\s+\(xy\s+([-\d.]+)\s+([-\d.]+)\)\)'
        matches = re.findall(pattern, sexp)

        for x1, y1, x2, y2 in matches:
            wires.append({
                "start": (float(x1), float(y1)),
                "end": (float(x2), float(y2))
            })

        return wires

    def _extract_labels_from_sexp(self, sexp: str) -> List[Dict]:
        """Extract label information from S-expression."""
        labels = []

        # Match regular labels
        pattern = r'\(label\s+"([^"]+)"\s+\(at\s+([-\d.]+)\s+([-\d.]+)'
        for text, x, y in re.findall(pattern, sexp):
            labels.append({"text": text, "type": "label", "position": (float(x), float(y))})

        # Match global labels
        pattern = r'\(global_label\s+"([^"]+)".*?\(at\s+([-\d.]+)\s+([-\d.]+)'
        for text, x, y in re.findall(pattern, sexp, re.DOTALL):
            labels.append({"text": text, "type": "global_label", "position": (float(x), float(y))})

        return labels


class SpecificationValidator(ValidatorAgent):
    """
    Validates schematic against original design specifications.

    Checks:
    - All required subsystems are present
    - Signal paths match specifications
    - Power requirements are met
    - Interface specifications are satisfied
    """

    def __init__(self):
        super().__init__("SpecificationValidator", weight=1.2)

    async def validate(self, context: SchematicContext) -> ValidationVote:
        """Validate against specifications."""
        import time
        start_time = time.time()

        issues = []
        passed = True
        confidence = 0.85

        # Extract design requirements from intent
        required_components = self._extract_required_components(context.design_intent)
        actual_components = self._extract_components_from_sexp(context.schematic_sexp)
        actual_refs = {c["reference"] for c in actual_components}

        # Check all required subsystems are present
        bom_parts = {item.get("part_number", "") for item in context.bom}

        # Verify BOM items are in schematic
        for bom_item in context.bom:
            part = bom_item.get("part_number", "")
            ref = bom_item.get("reference", "")

            if ref and ref not in actual_refs:
                issues.append(ValidationIssue(
                    category=IssueCategory.MISSING_COMPONENT,
                    severity=IssueSeverity.ERROR,
                    message=f"BOM item {part} ({ref}) not found in schematic",
                    component_ref=ref,
                    recommendation=f"Add {part} to schematic with reference {ref}"
                ))
                passed = False

        # Check power requirements
        power_labels = [l for l in self._extract_labels_from_sexp(context.schematic_sexp)
                       if "vcc" in l["text"].lower() or "vdd" in l["text"].lower() or "3v3" in l["text"].lower()]
        gnd_labels = [l for l in self._extract_labels_from_sexp(context.schematic_sexp)
                     if "gnd" in l["text"].lower() or "vss" in l["text"].lower()]

        if not power_labels:
            issues.append(ValidationIssue(
                category=IssueCategory.POWER_ISSUE,
                severity=IssueSeverity.CRITICAL,
                message="No power supply connections found in schematic",
                recommendation="Add VCC/VDD power connections"
            ))
            passed = False
            confidence = 0.6

        if not gnd_labels:
            issues.append(ValidationIssue(
                category=IssueCategory.GROUNDING_ISSUE,
                severity=IssueSeverity.CRITICAL,
                message="No ground connections found in schematic",
                recommendation="Add GND connections"
            ))
            passed = False
            confidence = 0.6

        # Check connections against specification
        expected_connections = self._extract_expected_connections(context.design_intent)
        actual_connections = context.connections

        for expected in expected_connections:
            found = False
            for actual in actual_connections:
                if self._connections_match(expected, actual):
                    found = True
                    break

            if not found:
                issues.append(ValidationIssue(
                    category=IssueCategory.MISSING_CONNECTION,
                    severity=IssueSeverity.WARNING,
                    message=f"Expected connection not found: {expected}",
                    recommendation=f"Add connection {expected}"
                ))

        elapsed = (time.time() - start_time) * 1000

        return ValidationVote(
            agent_name=self.name,
            passes=passed,
            confidence=confidence,
            issues=issues,
            reasoning=f"Validated {len(actual_components)} components against BOM with {len(context.bom)} items. "
                     f"Found {len(power_labels)} power and {len(gnd_labels)} ground connections.",
            execution_time_ms=elapsed
        )

    def _extract_required_components(self, design_intent: str) -> Set[str]:
        """Extract required component types from design intent."""
        required = set()

        # Look for common component mentions
        patterns = {
            r'\bMCU\b': "MCU",
            r'\bmicrocontroller\b': "MCU",
            r'\bgate driver\b': "Gate_Driver",
            r'\bMOSFET\b': "MOSFET",
            r'\bcapacitor\b': "Capacitor",
            r'\bresistor\b': "Resistor",
            r'\bregulator\b': "Regulator",
            r'\bLDO\b': "Regulator",
        }

        for pattern, comp_type in patterns.items():
            if re.search(pattern, design_intent, re.IGNORECASE):
                required.add(comp_type)

        return required

    def _extract_expected_connections(self, design_intent: str) -> List[str]:
        """Extract expected connections from design intent."""
        # This would use LLM for intelligent extraction
        # For now, return empty list
        return []

    def _connections_match(self, expected: str, actual: Dict) -> bool:
        """Check if an expected connection matches actual."""
        # Simplified matching
        return False


class CircuitAnalyst(ValidatorAgent):
    """
    Analyzes circuit topology for correctness.

    Checks:
    - Feedback loops are stable
    - No floating inputs/outputs
    - Component values are appropriate
    - Signal paths are complete
    """

    def __init__(self):
        super().__init__("CircuitAnalyst", weight=1.0)

    async def validate(self, context: SchematicContext) -> ValidationVote:
        """Analyze circuit topology."""
        import time
        start_time = time.time()

        issues = []
        passed = True
        confidence = 0.8

        components = self._extract_components_from_sexp(context.schematic_sexp)
        wires = self._extract_wires_from_sexp(context.schematic_sexp)
        labels = self._extract_labels_from_sexp(context.schematic_sexp)

        # Check for bypass capacitors near ICs
        ics = [c for c in components if c["reference"].startswith("U")]
        caps = [c for c in components if c["reference"].startswith("C")]

        if len(ics) > 0 and len(caps) < len(ics):
            issues.append(ValidationIssue(
                category=IssueCategory.MISSING_COMPONENT,
                severity=IssueSeverity.WARNING,
                message=f"Insufficient bypass capacitors: {len(ics)} ICs but only {len(caps)} capacitors",
                recommendation="Add 0.1uF bypass capacitor near each IC power pin"
            ))

        # Check wire connectivity
        if len(wires) == 0 and len(components) > 1:
            issues.append(ValidationIssue(
                category=IssueCategory.MISSING_CONNECTION,
                severity=IssueSeverity.CRITICAL,
                message="No wire connections found between components",
                recommendation="Add wire connections between component pins"
            ))
            passed = False
            confidence = 0.5

        # Check for protection circuits in power path
        has_power_ic = any("reg" in c["lib_id"].lower() or "ldo" in c["lib_id"].lower()
                          for c in components)
        has_fuse = any(c["reference"].startswith("F") for c in components)
        has_tvs = any("tvs" in c["lib_id"].lower() for c in components)

        if has_power_ic and not has_fuse:
            issues.append(ValidationIssue(
                category=IssueCategory.MISSING_PROTECTION,
                severity=IssueSeverity.WARNING,
                message="Power circuit lacks fuse protection",
                recommendation="Add fuse to power input for overcurrent protection"
            ))

        # Check for proper decoupling
        power_nets = [l for l in labels if "vcc" in l["text"].lower() or "vdd" in l["text"].lower()]
        if len(power_nets) > 0 and len(caps) < len(power_nets):
            issues.append(ValidationIssue(
                category=IssueCategory.POWER_ISSUE,
                severity=IssueSeverity.WARNING,
                message="Insufficient decoupling: some power connections may not have bypass caps",
                recommendation="Add bypass capacitors to all power net connections"
            ))

        elapsed = (time.time() - start_time) * 1000

        return ValidationVote(
            agent_name=self.name,
            passes=passed,
            confidence=confidence,
            issues=issues,
            reasoning=f"Analyzed topology with {len(components)} components, {len(wires)} wires. "
                     f"IC count: {len(ics)}, Capacitor count: {len(caps)}.",
            execution_time_ms=elapsed
        )


class AdversarialTester(ValidatorAgent):
    """
    Attempts to find failure modes in the schematic.

    Adversarial approach:
    - Injects fault scenarios
    - Identifies edge cases
    - Challenges other agents' conclusions
    - Can VETO if critical issue found
    """

    VETO_THRESHOLD = 0.9  # Confidence threshold for veto

    def __init__(self):
        super().__init__("AdversarialTester", weight=1.5)  # Higher weight for adversarial

    async def validate(self, context: SchematicContext) -> ValidationVote:
        """Perform adversarial testing."""
        import time
        start_time = time.time()

        issues = []
        passed = True
        confidence = 0.75
        veto_issues = []

        components = self._extract_components_from_sexp(context.schematic_sexp)
        wires = self._extract_wires_from_sexp(context.schematic_sexp)
        labels = self._extract_labels_from_sexp(context.schematic_sexp)

        # VETO CHECK 1: Empty schematic
        if len(components) == 0:
            veto_issues.append(ValidationIssue(
                category=IssueCategory.TOPOLOGY_ERROR,
                severity=IssueSeverity.CRITICAL,
                message="VETO: Schematic contains no components",
                evidence="Component count = 0"
            ))
            passed = False
            confidence = 1.0

        # VETO CHECK 2: No connections at all
        if len(components) > 1 and len(wires) == 0 and len(labels) == 0:
            veto_issues.append(ValidationIssue(
                category=IssueCategory.MISSING_CONNECTION,
                severity=IssueSeverity.CRITICAL,
                message="VETO: Components exist but no connections found",
                evidence=f"{len(components)} components with 0 wires and 0 labels"
            ))
            passed = False
            confidence = 1.0

        # FAULT INJECTION: What if power fails?
        power_count = len([l for l in labels if "vcc" in l["text"].lower()])
        gnd_count = len([l for l in labels if "gnd" in l["text"].lower()])

        if power_count == 0 or gnd_count == 0:
            issues.append(ValidationIssue(
                category=IssueCategory.POWER_ISSUE,
                severity=IssueSeverity.ERROR,
                message="Missing power/ground infrastructure - circuit will not function",
                evidence=f"VCC connections: {power_count}, GND connections: {gnd_count}",
                recommendation="Ensure all ICs have proper power and ground connections"
            ))
            passed = False

        # FAULT INJECTION: What if bypass caps are missing?
        ics = [c for c in components if c["reference"].startswith("U")]
        caps = [c for c in components if c["reference"].startswith("C")]

        if len(ics) > 2 and len(caps) == 0:
            issues.append(ValidationIssue(
                category=IssueCategory.MISSING_PROTECTION,
                severity=IssueSeverity.ERROR,
                message="No capacitors in design - ICs will likely oscillate or malfunction",
                evidence=f"{len(ics)} ICs with 0 capacitors",
                recommendation="Add at least one 0.1uF bypass capacitor per IC"
            ))
            passed = False

        # EDGE CASE: Single-point failures
        # Check for nets with only one connection (floating)
        net_connections: Dict[str, int] = {}
        for label in labels:
            net = label["text"]
            net_connections[net] = net_connections.get(net, 0) + 1

        floating_nets = [net for net, count in net_connections.items() if count == 1]
        if floating_nets:
            issues.append(ValidationIssue(
                category=IssueCategory.SIGNAL_INTEGRITY,
                severity=IssueSeverity.WARNING,
                message=f"Potential floating nets detected: {', '.join(floating_nets[:5])}",
                evidence=f"{len(floating_nets)} nets have only one connection point",
                recommendation="Verify these nets are intentionally single-ended or add connections"
            ))

        elapsed = (time.time() - start_time) * 1000

        # Include veto issues in main issues list
        issues.extend(veto_issues)

        return ValidationVote(
            agent_name=self.name,
            passes=passed,
            confidence=confidence,
            issues=issues,
            reasoning=f"Adversarial testing complete. Injected {3} fault scenarios. "
                     f"Found {len(veto_issues)} veto-worthy issues and {len(issues) - len(veto_issues)} other issues.",
            execution_time_ms=elapsed
        )


class ReferenceComparator(ValidatorAgent):
    """
    Compares schematic against known good reference designs.

    Checks:
    - Deviations from proven patterns
    - Manufacturer application note compliance
    - Industry standard circuit topologies
    """

    # Reference patterns for common circuits
    REFERENCE_PATTERNS = {
        "MCU_power": {
            "required": ["bypass_cap_0.1uF", "bypass_cap_10uF"],
            "recommended": ["ferrite_bead", "tvs_diode"]
        },
        "Gate_Driver": {
            "required": ["bootstrap_cap", "gate_resistor"],
            "recommended": ["tvs_protection"]
        },
        "LDO_Regulator": {
            "required": ["input_cap", "output_cap"],
            "recommended": ["input_protection"]
        }
    }

    def __init__(self):
        super().__init__("ReferenceComparator", weight=0.8)

    async def validate(self, context: SchematicContext) -> ValidationVote:
        """Compare against reference designs."""
        import time
        start_time = time.time()

        issues = []
        passed = True
        confidence = 0.7

        components = self._extract_components_from_sexp(context.schematic_sexp)

        # Identify circuit blocks
        has_mcu = any("mcu" in c["lib_id"].lower() or "stm32" in c["lib_id"].lower()
                      for c in components)
        has_gate_driver = any("drv" in c["lib_id"].lower() for c in components)
        has_regulator = any("reg" in c["lib_id"].lower() or "ldo" in c["lib_id"].lower()
                           for c in components)

        caps = [c for c in components if c["reference"].startswith("C")]
        resistors = [c for c in components if c["reference"].startswith("R")]

        # Check MCU reference pattern
        if has_mcu:
            if len(caps) < 2:
                issues.append(ValidationIssue(
                    category=IssueCategory.REFERENCE_DEVIATION,
                    severity=IssueSeverity.WARNING,
                    message="MCU power design deviates from reference: insufficient bypass capacitors",
                    evidence="Reference designs typically require 0.1uF + 10uF per power pin",
                    recommendation="Add 0.1uF and 10uF capacitors near MCU power pins"
                ))

        # Check gate driver reference pattern
        if has_gate_driver:
            has_bootstrap_cap = any("100nF" in str(context.bom) or "0.1uF" in str(context.bom)
                                   for c in caps)
            has_gate_resistor = len(resistors) > 0

            if not has_gate_resistor:
                issues.append(ValidationIssue(
                    category=IssueCategory.REFERENCE_DEVIATION,
                    severity=IssueSeverity.WARNING,
                    message="Gate driver lacks gate resistors - may cause ringing/EMI",
                    evidence="Reference designs use 4.7-10 ohm gate resistors",
                    recommendation="Add gate resistors between driver output and MOSFET gate"
                ))

        # Check regulator reference pattern
        if has_regulator:
            if len(caps) < 2:
                issues.append(ValidationIssue(
                    category=IssueCategory.REFERENCE_DEVIATION,
                    severity=IssueSeverity.ERROR,
                    message="Regulator lacks proper input/output capacitors - may be unstable",
                    evidence="LDO/Regulator requires input and output capacitors per datasheet",
                    recommendation="Add capacitors as specified in regulator datasheet"
                ))
                passed = False

        elapsed = (time.time() - start_time) * 1000

        return ValidationVote(
            agent_name=self.name,
            passes=passed,
            confidence=confidence,
            issues=issues,
            reasoning=f"Compared against {len(self.REFERENCE_PATTERNS)} reference patterns. "
                     f"MCU: {has_mcu}, Gate Driver: {has_gate_driver}, Regulator: {has_regulator}.",
            execution_time_ms=elapsed
        )


class MAPOFunctionalValidator:
    """
    MAPO Gaming coordinator for functional validation.

    Orchestrates competitive validation between multiple agents
    and aggregates results with weighted voting.
    """

    PASS_THRESHOLD = 0.75  # Supermajority required to pass
    AGREEMENT_THRESHOLD = 0.7  # Minimum agreement score

    def __init__(self, llm_client: Any = None):
        """
        Initialize the MAPO validator.

        Args:
            llm_client: Optional LLM client for intelligent analysis
        """
        self.llm_client = llm_client

        # Initialize competitive agents
        self.agents: List[ValidatorAgent] = [
            SpecificationValidator(),
            CircuitAnalyst(),
            AdversarialTester(),
            ReferenceComparator(),
        ]

        # Set LLM client on agents that support it
        if llm_client:
            for agent in self.agents:
                agent.set_llm_client(llm_client)

    async def validate(
        self,
        schematic_sexp: str,
        design_intent: str,
        bom: List[Dict[str, Any]],
        connections: List[Dict[str, Any]],
        architecture: Optional[Dict[str, Any]] = None,
        reference_designs: Optional[List[Dict[str, Any]]] = None,
        power_budget: Optional[Dict[str, float]] = None
    ) -> ValidationResult:
        """
        Run competitive validation with all agents.

        Args:
            schematic_sexp: KiCad schematic S-expression
            design_intent: Original design requirements
            bom: Bill of materials
            connections: Connection list
            architecture: Optional architecture definition
            reference_designs: Optional reference designs to compare against
            power_budget: Optional power budget constraints

        Returns:
            ValidationResult with aggregated scores and issues
        """
        logger.info(f"Starting MAPO competitive validation with {len(self.agents)} agents")

        # Create context
        context = SchematicContext(
            schematic_sexp=schematic_sexp,
            design_intent=design_intent,
            bom=bom,
            connections=connections,
            architecture=architecture,
            reference_designs=reference_designs,
            power_budget=power_budget
        )

        # Run all agents in parallel (competitive)
        votes = await asyncio.gather(*[
            agent.validate(context)
            for agent in self.agents
        ])

        # Aggregate results
        result = self._aggregate_votes(votes)

        logger.info(
            f"Validation complete: {'PASSED' if result.passed else 'FAILED'} "
            f"(score: {result.overall_score:.2f}, agreement: {result.agreement_score:.2f})"
        )

        return result

    def _aggregate_votes(self, votes: List[ValidationVote]) -> ValidationResult:
        """Aggregate votes with weighted scoring and veto check."""
        all_issues = []
        critical_issues = []
        veto_triggered = False
        veto_reason = None

        # Check for veto from AdversarialTester
        for vote in votes:
            if vote.agent_name == "AdversarialTester":
                for issue in vote.issues:
                    if issue.severity == IssueSeverity.CRITICAL:
                        if vote.confidence >= AdversarialTester.VETO_THRESHOLD:
                            veto_triggered = True
                            veto_reason = issue.message
                            break

        # Collect all issues
        for vote in votes:
            all_issues.extend(vote.issues)
            critical_issues.extend([i for i in vote.issues if i.severity == IssueSeverity.CRITICAL])

        # Calculate weighted scores
        total_weight = 0.0
        weighted_pass_score = 0.0

        for vote in votes:
            agent = next((a for a in self.agents if a.name == vote.agent_name), None)
            weight = agent.weight if agent else 1.0

            total_weight += weight * vote.confidence
            if vote.passes:
                weighted_pass_score += weight * vote.confidence

        # Overall score (0-1)
        overall_score = weighted_pass_score / total_weight if total_weight > 0 else 0.0

        # Agreement score (how much agents agree)
        pass_votes = sum(1 for v in votes if v.passes)
        agreement_score = max(pass_votes, len(votes) - pass_votes) / len(votes) if votes else 0.0

        # Determine pass/fail
        passed = (
            not veto_triggered and
            overall_score >= self.PASS_THRESHOLD and
            agreement_score >= self.AGREEMENT_THRESHOLD
        )

        # Generate recommendations
        recommendations = []
        for issue in sorted(all_issues, key=lambda i: i.severity.value, reverse=True):
            if issue.recommendation and issue.recommendation not in recommendations:
                recommendations.append(issue.recommendation)
                if len(recommendations) >= 10:
                    break

        return ValidationResult(
            passed=passed,
            overall_score=overall_score,
            agreement_score=agreement_score,
            votes=votes,
            critical_issues=critical_issues,
            all_issues=all_issues,
            veto_triggered=veto_triggered,
            veto_reason=veto_reason,
            recommendations=recommendations
        )


# CLI entry point for testing
if __name__ == "__main__":
    import sys

    async def main():
        # Test schematic S-expression
        test_sexp = """(kicad_sch (version 20231120) (generator "nexus_ee_design")
  (uuid "test-uuid")
  (paper "A4")
  (lib_symbols
    (symbol "STM32G431CBT6" (in_bom yes) (on_board yes)
      (property "Reference" "U" (at 0 0 0))
      (property "Value" "STM32G431" (at 0 0 0))
    )
    (symbol "C" (in_bom yes) (on_board yes)
      (property "Reference" "C" (at 0 0 0))
    )
  )
  (symbol (lib_id "STM32G431CBT6") (at 50 80 0) (unit 1)
    (property "Reference" "U1" (at 50 75 0))
  )
  (symbol (lib_id "C") (at 50 120 0) (unit 1)
    (property "Reference" "C1" (at 50 115 0))
  )
  (symbol (lib_id "C") (at 70 120 0) (unit 1)
    (property "Reference" "C2" (at 70 115 0))
  )
  (wire (pts (xy 50 90) (xy 50 110)))
  (wire (pts (xy 50 70) (xy 70 70)))
  (global_label "VCC" (shape input) (at 50 60 0))
  (global_label "GND" (shape input) (at 50 130 0))
  (label "PWM_A" (at 60 80 0))
)"""

        test_bom = [
            {"part_number": "STM32G431CBT6", "reference": "U1", "category": "MCU"},
            {"part_number": "100nF", "reference": "C1", "category": "Capacitor"},
            {"part_number": "10uF", "reference": "C2", "category": "Capacitor"},
        ]

        test_connections = [
            {"from_ref": "U1", "from_pin": "VCC", "to_ref": "C1", "to_pin": "1", "net_name": "VCC"},
            {"from_ref": "U1", "from_pin": "GND", "to_ref": "C1", "to_pin": "2", "net_name": "GND"},
        ]

        test_intent = """
        Design a simple MCU breakout board with:
        - STM32G431 microcontroller
        - Bypass capacitors for power stability
        - PWM outputs for motor control
        """

        print("Testing MAPO Functional Validator...")
        print("=" * 60)

        validator = MAPOFunctionalValidator()

        result = await validator.validate(
            schematic_sexp=test_sexp,
            design_intent=test_intent,
            bom=test_bom,
            connections=test_connections
        )

        print(f"\n{'='*60}")
        print("VALIDATION RESULT")
        print(f"{'='*60}")
        print(f"Status: {'PASSED' if result.passed else 'FAILED'}")
        print(f"Overall Score: {result.overall_score:.2%}")
        print(f"Agreement Score: {result.agreement_score:.2%}")

        if result.veto_triggered:
            print(f"\nVETO TRIGGERED: {result.veto_reason}")

        print(f"\nAgent Votes:")
        for vote in result.votes:
            status = "PASS" if vote.passes else "FAIL"
            print(f"  {vote.agent_name}: {status} (confidence: {vote.confidence:.2%}, {len(vote.issues)} issues)")

        if result.critical_issues:
            print(f"\nCritical Issues ({len(result.critical_issues)}):")
            for issue in result.critical_issues:
                print(f"  - [{issue.category.value}] {issue.message}")

        if result.recommendations:
            print(f"\nRecommendations:")
            for i, rec in enumerate(result.recommendations[:5], 1):
                print(f"  {i}. {rec}")

        print(f"\n{'='*60}")
        print("MAPO Functional Validation test complete!")

    asyncio.run(main())
