"""
Standards Compliance Agent - Enforces IEC/IEEE/IPC schematic standards.

Validates and enforces:
1. Reference designator conventions (IEC 60750, IEEE 315)
2. Net naming standards (functional, active-low conventions)
3. Label placement rules
4. Grid alignment
5. Junction dot requirements
6. Title block completeness

Author: Nexus EE Design Team
"""

import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


class ComplianceCheck(Enum):
    """Types of compliance checks."""
    REF_DESIGNATOR_PREFIX = "Reference designator uses correct IEC/IEEE prefix"
    REF_DESIGNATOR_SEQUENCE = "Reference designators are sequentially numbered"
    REF_DESIGNATOR_UNIQUE = "No duplicate reference designators"
    NET_NAME_DESCRIPTIVE = "Net names are descriptive (not auto-generated)"
    NET_NAME_CONSISTENT = "Net names are consistent across connections"
    NET_NAME_ACTIVE_LOW = "Active-low signals use proper convention (_N or _B)"
    POWER_LABELED = "Power and ground pins have labels"
    GRID_ALIGNED = "All component pins are on 100mil grid"
    JUNCTION_DOTS = "Junction dots present at wire connections"
    NO_FOUR_WAY = "No 4-way wire junctions"
    BYPASS_ADJACENT = "Bypass capacitors are adjacent to ICs"
    TITLE_BLOCK = "Title block is complete"


class ViolationSeverity(Enum):
    """Severity levels for compliance violations."""
    ERROR = "error"       # Must fix
    WARNING = "warning"   # Should fix
    INFO = "info"         # Suggestion


@dataclass
class ComplianceViolation:
    """A standards compliance violation."""
    check: ComplianceCheck
    severity: ViolationSeverity
    component_ref: Optional[str]
    description: str
    fix_suggestion: str
    location: Optional[Tuple[float, float]] = None


@dataclass
class ComplianceReport:
    """Complete standards compliance report."""
    passed: bool
    total_checks: int
    passed_checks: int
    violations: List[ComplianceViolation] = field(default_factory=list)
    fixes_applied: List[str] = field(default_factory=list)
    score: float = 0.0  # 0-1 compliance score

    def to_dict(self) -> Dict:
        return {
            "passed": self.passed,
            "score": f"{self.score:.1%}",
            "total_checks": self.total_checks,
            "passed_checks": self.passed_checks,
            "error_count": len([v for v in self.violations if v.severity == ViolationSeverity.ERROR]),
            "warning_count": len([v for v in self.violations if v.severity == ViolationSeverity.WARNING]),
            "violations": [
                {
                    "check": v.check.value,
                    "severity": v.severity.value,
                    "component": v.component_ref,
                    "description": v.description,
                    "fix": v.fix_suggestion,
                }
                for v in self.violations
            ]
        }


class StandardsComplianceAgent:
    """
    Validates schematic compliance with IEC/IEEE/IPC standards.

    Checks:
    1. Reference designators follow IEC 60750 / IEEE 315
    2. Net names are descriptive and consistent
    3. Power connections are properly labeled
    4. Grid alignment is correct (100 mil)
    5. Junction dots are present at connections
    6. No 4-way wire junctions
    """

    # IEC 60750 / IEEE 315 Reference Designator Prefixes
    VALID_PREFIXES = {
        "A": ["Assembly", "Subassembly"],
        "AT": ["Attenuator", "Isolator"],
        "B": ["Blower", "Motor"],
        "BT": ["Battery"],
        "C": ["Capacitor"],
        "CB": ["Circuit Breaker"],
        "D": ["Diode", "LED", "Zener", "TVS"],
        "DS": ["Display", "Light"],
        "E": ["Antenna"],
        "F": ["Fuse"],
        "FB": ["Ferrite Bead"],
        "G": ["Generator", "Oscillator"],
        "H": ["Hardware"],
        "HY": ["Circulator", "Directional Coupler"],
        "J": ["Connector", "Jack", "Header"],
        "JP": ["Jumper"],
        "K": ["Relay", "Contactor"],
        "L": ["Inductor", "Coil", "Ferrite"],
        "LS": ["Speaker", "Buzzer"],
        "M": ["Motor"],
        "MK": ["Microphone"],
        "MP": ["Mechanical Part"],
        "P": ["Plug", "Connector"],
        "PS": ["Power Supply"],
        "Q": ["Transistor", "MOSFET", "BJT", "IGBT"],
        "R": ["Resistor"],
        "RT": ["Thermistor"],
        "RN": ["Resistor Network"],
        "S": ["Switch"],
        "T": ["Transformer"],
        "TC": ["Thermocouple"],
        "TP": ["Test Point"],
        "U": ["IC", "MCU", "OpAmp", "Gate_Driver", "Amplifier", "Comparator"],
        "V": ["Vacuum Tube"],
        "VR": ["Varistor"],
        "W": ["Wire", "Cable"],
        "X": ["Socket"],
        "Y": ["Crystal", "Oscillator"],
        "Z": ["Network", "Filter"],
    }

    # Category to prefix mapping
    CATEGORY_PREFIX = {
        "MCU": "U",
        "IC": "U",
        "OpAmp": "U",
        "Gate_Driver": "U",
        "Amplifier": "U",
        "Comparator": "U",
        "Regulator": "U",
        "Power": "U",
        "CAN_Transceiver": "U",
        "MOSFET": "Q",
        "BJT": "Q",
        "Transistor": "Q",
        "IGBT": "Q",
        "Capacitor": "C",
        "Resistor": "R",
        "Inductor": "L",
        "Diode": "D",
        "LED": "D",
        "TVS": "D",
        "Zener": "D",
        "Connector": "J",
        "Crystal": "Y",
        "Thermistor": "RT",
        "Fuse": "F",
        "Relay": "K",
        "Transformer": "T",
        "Ferrite": "FB",
        "Switch": "S",
        "Test_Point": "TP",
    }

    # Grid unit (100 mil = 2.54mm)
    GRID_UNIT = 2.54

    # Power net patterns
    POWER_NET_PATTERNS = [
        r"^VCC$", r"^VDD$", r"^AVCC$", r"^DVCC$",
        r"^V\d+V?\d*$", r"^\+\d+V\d*$", r"^-\d+V\d*$",
        r"^VIN$", r"^VOUT$", r"^VBAT$",
    ]

    GROUND_NET_PATTERNS = [
        r"^GND$", r"^AGND$", r"^DGND$", r"^PGND$",
        r"^VSS$", r"^0V$", r"^COM$",
    ]

    # Active-low suffixes
    ACTIVE_LOW_SUFFIXES = ["_N", "_B", "_L", "N", "B"]

    def __init__(self):
        """Initialize the standards compliance agent."""
        self.violations: List[ComplianceViolation] = []

    def validate(
        self,
        sheet: Any,  # SchematicSheet
        bom: Optional[List[Dict]] = None,
        auto_fix: bool = False,
    ) -> ComplianceReport:
        """
        Validate schematic sheet against standards.

        Args:
            sheet: SchematicSheet to validate
            bom: Optional BOM for category info
            auto_fix: Automatically fix violations where possible

        Returns:
            ComplianceReport with all violations
        """
        self.violations = []
        fixes_applied = []
        total_checks = 0
        passed_checks = 0

        # Build category lookup
        categories = {}
        if bom:
            for item in bom:
                ref = item.get("reference")
                if ref:
                    categories[ref] = item.get("category", "Other")

        # Run all compliance checks
        checks = [
            (ComplianceCheck.REF_DESIGNATOR_PREFIX,
             self._check_ref_prefixes, sheet.symbols, categories),
            (ComplianceCheck.REF_DESIGNATOR_UNIQUE,
             self._check_ref_unique, sheet.symbols),
            (ComplianceCheck.REF_DESIGNATOR_SEQUENCE,
             self._check_ref_sequence, sheet.symbols),
            (ComplianceCheck.NET_NAME_DESCRIPTIVE,
             self._check_net_names, sheet.labels),
            (ComplianceCheck.POWER_LABELED,
             self._check_power_labels, sheet.symbols, sheet.labels),
            (ComplianceCheck.GRID_ALIGNED,
             self._check_grid_alignment, sheet.symbols),
            (ComplianceCheck.JUNCTION_DOTS,
             self._check_junction_dots, sheet.wires, sheet.junctions),
            (ComplianceCheck.NO_FOUR_WAY,
             self._check_no_four_way, sheet.wires, sheet.junctions),
        ]

        for check_info in checks:
            check_type = check_info[0]
            check_func = check_info[1]
            args = check_info[2:]

            total_checks += 1
            check_violations = check_func(*args)

            if not check_violations:
                passed_checks += 1
            else:
                self.violations.extend(check_violations)

                # Auto-fix if enabled
                if auto_fix:
                    for v in check_violations:
                        fixed = self._apply_fix(v, sheet)
                        if fixed:
                            fixes_applied.append(
                                f"Fixed: {v.check.value} for {v.component_ref or 'sheet'}"
                            )

        # Calculate score
        score = passed_checks / total_checks if total_checks > 0 else 0.0

        # Determine pass/fail
        error_count = len([v for v in self.violations if v.severity == ViolationSeverity.ERROR])
        passed = error_count == 0

        return ComplianceReport(
            passed=passed,
            total_checks=total_checks,
            passed_checks=passed_checks,
            violations=self.violations,
            fixes_applied=fixes_applied,
            score=score,
        )

    def _check_ref_prefixes(
        self,
        symbols: List[Any],
        categories: Dict[str, str]
    ) -> List[ComplianceViolation]:
        """Check reference designator prefixes match IEC/IEEE standards."""
        violations = []

        for symbol in symbols:
            ref = symbol.reference
            if not ref:
                continue

            # Extract prefix (letters before number)
            match = re.match(r"^([A-Z]+)", ref)
            if not match:
                violations.append(ComplianceViolation(
                    check=ComplianceCheck.REF_DESIGNATOR_PREFIX,
                    severity=ViolationSeverity.ERROR,
                    component_ref=ref,
                    description=f"Invalid reference designator format: {ref}",
                    fix_suggestion="Use format: PREFIX + NUMBER (e.g., U1, R10)",
                ))
                continue

            prefix = match.group(1)

            # Check if prefix is valid
            if prefix not in self.VALID_PREFIXES:
                violations.append(ComplianceViolation(
                    check=ComplianceCheck.REF_DESIGNATOR_PREFIX,
                    severity=ViolationSeverity.WARNING,
                    component_ref=ref,
                    description=f"Non-standard prefix '{prefix}' in {ref}",
                    fix_suggestion=f"Use IEC/IEEE standard prefix from: {list(self.VALID_PREFIXES.keys())}",
                ))
                continue

            # Check if prefix matches category
            category = categories.get(ref, "Other")
            expected_prefix = self.CATEGORY_PREFIX.get(category)

            if expected_prefix and prefix != expected_prefix:
                violations.append(ComplianceViolation(
                    check=ComplianceCheck.REF_DESIGNATOR_PREFIX,
                    severity=ViolationSeverity.WARNING,
                    component_ref=ref,
                    description=f"Prefix '{prefix}' doesn't match category '{category}' (expected: {expected_prefix})",
                    fix_suggestion=f"Change {ref} to {expected_prefix}X",
                ))

        return violations

    def _check_ref_unique(self, symbols: List[Any]) -> List[ComplianceViolation]:
        """Check for duplicate reference designators."""
        violations = []
        seen: Dict[str, int] = {}

        for symbol in symbols:
            ref = symbol.reference
            if not ref:
                continue

            if ref in seen:
                seen[ref] += 1
                violations.append(ComplianceViolation(
                    check=ComplianceCheck.REF_DESIGNATOR_UNIQUE,
                    severity=ViolationSeverity.ERROR,
                    component_ref=ref,
                    description=f"Duplicate reference designator: {ref} (occurrence {seen[ref]})",
                    fix_suggestion=f"Rename to unique designator",
                ))
            else:
                seen[ref] = 1

        return violations

    def _check_ref_sequence(self, symbols: List[Any]) -> List[ComplianceViolation]:
        """Check reference designators are sequentially numbered."""
        violations = []

        # Group by prefix
        prefix_numbers: Dict[str, List[int]] = {}

        for symbol in symbols:
            ref = symbol.reference
            if not ref:
                continue

            match = re.match(r"^([A-Z]+)(\d+)$", ref)
            if match:
                prefix = match.group(1)
                number = int(match.group(2))

                if prefix not in prefix_numbers:
                    prefix_numbers[prefix] = []
                prefix_numbers[prefix].append(number)

        # Check for gaps
        for prefix, numbers in prefix_numbers.items():
            numbers.sort()
            for i, num in enumerate(numbers):
                expected = i + 1
                if num != expected:
                    violations.append(ComplianceViolation(
                        check=ComplianceCheck.REF_DESIGNATOR_SEQUENCE,
                        severity=ViolationSeverity.INFO,
                        component_ref=f"{prefix}*",
                        description=f"Non-sequential numbering for {prefix}: expected {expected}, found {num}",
                        fix_suggestion=f"Renumber {prefix} components sequentially",
                    ))
                    break  # Only report first gap per prefix

        return violations

    def _check_net_names(self, labels: List[Any]) -> List[ComplianceViolation]:
        """Check net names are descriptive."""
        violations = []

        for label in labels:
            name = label.text

            # Check for auto-generated names
            if re.match(r"^Net-\(", name) or re.match(r"^N\d+$", name):
                violations.append(ComplianceViolation(
                    check=ComplianceCheck.NET_NAME_DESCRIPTIVE,
                    severity=ViolationSeverity.WARNING,
                    component_ref=None,
                    description=f"Auto-generated net name: {name}",
                    fix_suggestion="Use descriptive functional name (e.g., UART_TX, SPI_MOSI)",
                    location=label.position,
                ))

        return violations

    def _check_power_labels(
        self,
        symbols: List[Any],
        labels: List[Any]
    ) -> List[ComplianceViolation]:
        """Check power pins have labels."""
        violations = []

        # Collect existing power labels
        power_labels = set()
        ground_labels = set()

        for label in labels:
            name = label.text.upper()
            for pattern in self.POWER_NET_PATTERNS:
                if re.match(pattern, name):
                    power_labels.add(label.position)
            for pattern in self.GROUND_NET_PATTERNS:
                if re.match(pattern, name):
                    ground_labels.add(label.position)

        # Check each symbol's power pins
        for symbol in symbols:
            for pin in symbol.pins:
                pin_name = pin.name.upper()

                # Check power pins
                is_power = any(re.match(p, pin_name) for p in self.POWER_NET_PATTERNS)
                is_ground = any(re.match(p, pin_name) for p in self.GROUND_NET_PATTERNS)

                if is_power or is_ground:
                    pin_pos = symbol.get_absolute_pin_position(pin.name)
                    if pin_pos:
                        # Check if there's a label near this pin
                        has_label = any(
                            abs(lp[0] - pin_pos[0]) < 5 and abs(lp[1] - pin_pos[1]) < 5
                            for lp in (power_labels | ground_labels)
                        )

                        if not has_label:
                            violations.append(ComplianceViolation(
                                check=ComplianceCheck.POWER_LABELED,
                                severity=ViolationSeverity.WARNING,
                                component_ref=symbol.reference,
                                description=f"Power pin {pin_name} on {symbol.reference} has no label",
                                fix_suggestion=f"Add {'VCC' if is_power else 'GND'} label",
                                location=pin_pos,
                            ))

        return violations

    def _check_grid_alignment(self, symbols: List[Any]) -> List[ComplianceViolation]:
        """Check all pins are on 100 mil grid."""
        violations = []

        for symbol in symbols:
            x, y = symbol.position

            # Check if position is on grid
            x_grid = x / self.GRID_UNIT
            y_grid = y / self.GRID_UNIT

            if abs(x_grid - round(x_grid)) > 0.01 or abs(y_grid - round(y_grid)) > 0.01:
                violations.append(ComplianceViolation(
                    check=ComplianceCheck.GRID_ALIGNED,
                    severity=ViolationSeverity.ERROR,
                    component_ref=symbol.reference,
                    description=f"{symbol.reference} is off-grid at ({x:.2f}, {y:.2f})",
                    fix_suggestion=f"Snap to ({round(x_grid) * self.GRID_UNIT:.2f}, {round(y_grid) * self.GRID_UNIT:.2f})",
                    location=(x, y),
                ))

        return violations

    def _check_junction_dots(
        self,
        wires: List[Any],
        junctions: List[Any]
    ) -> List[ComplianceViolation]:
        """Check junction dots exist at wire connections."""
        violations = []

        # Count wire endpoints at each position
        endpoint_counts: Dict[Tuple[float, float], int] = {}

        for wire in wires:
            for point in [wire.start, wire.end]:
                key = (round(point[0], 2), round(point[1], 2))
                endpoint_counts[key] = endpoint_counts.get(key, 0) + 1

        # Check for missing junctions
        junction_positions = set(
            (round(j.position[0], 2), round(j.position[1], 2))
            for j in junctions
        )

        for pos, count in endpoint_counts.items():
            if count >= 3 and pos not in junction_positions:
                violations.append(ComplianceViolation(
                    check=ComplianceCheck.JUNCTION_DOTS,
                    severity=ViolationSeverity.WARNING,
                    component_ref=None,
                    description=f"Missing junction dot at ({pos[0]}, {pos[1]}) with {count} wires",
                    fix_suggestion="Add junction dot at this position",
                    location=pos,
                ))

        return violations

    def _check_no_four_way(
        self,
        wires: List[Any],
        junctions: List[Any]
    ) -> List[ComplianceViolation]:
        """Check for 4-way wire junctions (should be avoided)."""
        violations = []

        # Count wire endpoints at each position
        endpoint_counts: Dict[Tuple[float, float], int] = {}

        for wire in wires:
            for point in [wire.start, wire.end]:
                key = (round(point[0], 2), round(point[1], 2))
                endpoint_counts[key] = endpoint_counts.get(key, 0) + 1

        # Check for 4-way junctions
        for pos, count in endpoint_counts.items():
            if count >= 4:
                violations.append(ComplianceViolation(
                    check=ComplianceCheck.NO_FOUR_WAY,
                    severity=ViolationSeverity.WARNING,
                    component_ref=None,
                    description=f"4-way junction at ({pos[0]}, {pos[1]}) - {count} wires",
                    fix_suggestion="Split into two 3-way junctions for clarity",
                    location=pos,
                ))

        return violations

    def _apply_fix(self, violation: ComplianceViolation, sheet: Any) -> bool:
        """Attempt to automatically fix a violation."""
        # Grid alignment fix
        if violation.check == ComplianceCheck.GRID_ALIGNED:
            for symbol in sheet.symbols:
                if symbol.reference == violation.component_ref:
                    x, y = symbol.position
                    symbol.position = (
                        round(x / self.GRID_UNIT) * self.GRID_UNIT,
                        round(y / self.GRID_UNIT) * self.GRID_UNIT
                    )
                    return True

        # Junction dot fix
        if violation.check == ComplianceCheck.JUNCTION_DOTS:
            if violation.location:
                from agents.schematic_assembler import Junction
                sheet.junctions.append(Junction(position=violation.location))
                return True

        return False


# CLI test
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("Standards Compliance Agent test")
    print("Run with actual SchematicSheet objects from schematic assembler")
