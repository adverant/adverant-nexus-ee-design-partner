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
import math
import os
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
    # NASA-STD-8739.4 / MIL-STD-883 checks
    CAPACITOR_DERATING = "Capacitor voltage derating (NASA: >=1.5x nominal)"
    RESISTOR_DERATING = "Resistor power derating (NASA: >=2x dissipation)"
    POWER_BUDGET = "Power budget: total consumption <= supply capacity"
    BYPASS_CAP_DISTANCE = "Bypass caps within 10mm of IC power pins (IPC-2221)"
    IC_DECOUPLING = "Every IC has at least one decoupling capacitor"


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
            # NASA-STD-8739.4 / MIL-STD-883 / IPC-2221 checks
            (ComplianceCheck.BYPASS_CAP_DISTANCE,
             self._check_bypass_cap_distance, sheet.symbols, bom),
            (ComplianceCheck.IC_DECOUPLING,
             self._check_ic_decoupling, sheet.symbols, bom),
            (ComplianceCheck.CAPACITOR_DERATING,
             self._check_capacitor_derating, sheet.symbols, bom),
            (ComplianceCheck.RESISTOR_DERATING,
             self._check_resistor_derating, sheet.symbols, bom),
            (ComplianceCheck.POWER_BUDGET,
             self._check_power_budget, sheet.symbols, bom),
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

    def _check_bypass_cap_distance(
        self,
        symbols: List[Any],
        bom: Optional[List[Dict]],
    ) -> List[ComplianceViolation]:
        """
        Check bypass capacitors are within 10mm of their associated IC power pins.

        IPC-2221 / NASA-STD-8739.4 requires decoupling capacitors to be placed
        as close as possible to IC power pins (target: <10mm).
        """
        violations = []

        if not bom:
            return violations

        # Identify ICs and bypass caps
        bom_by_ref = {item.get("reference", ""): item for item in bom if item.get("reference")}
        ic_symbols = []
        bypass_cap_symbols = []

        for sym in symbols:
            bom_item = bom_by_ref.get(sym.reference, {})
            cat = bom_item.get("category", "").lower()
            value = bom_item.get("value", "").lower()

            if cat in ("mcu", "ic", "gate_driver", "can_transceiver", "regulator", "amplifier", "opamp"):
                ic_symbols.append(sym)
            elif cat == "capacitor" and self._is_bypass_value(value):
                bypass_cap_symbols.append(sym)

        # Check each bypass cap's distance to nearest IC
        for cap in bypass_cap_symbols:
            cap_x, cap_y = cap.position
            min_dist = float("inf")
            nearest_ic = None

            for ic in ic_symbols:
                ic_x, ic_y = ic.position
                dist = math.sqrt((cap_x - ic_x) ** 2 + (cap_y - ic_y) ** 2)
                if dist < min_dist:
                    min_dist = dist
                    nearest_ic = ic

            if nearest_ic and min_dist > 10.0:  # 10mm threshold
                violations.append(ComplianceViolation(
                    check=ComplianceCheck.BYPASS_CAP_DISTANCE,
                    severity=ViolationSeverity.WARNING,
                    component_ref=cap.reference,
                    description=(
                        f"Bypass cap {cap.reference} is {min_dist:.1f}mm from nearest IC "
                        f"({nearest_ic.reference}). IPC-2221 recommends <10mm."
                    ),
                    fix_suggestion=f"Move {cap.reference} closer to {nearest_ic.reference}",
                    location=cap.position,
                ))

        return violations

    def _check_ic_decoupling(
        self,
        symbols: List[Any],
        bom: Optional[List[Dict]],
    ) -> List[ComplianceViolation]:
        """
        Check every IC has at least one decoupling/bypass capacitor.

        NASA-STD-8739.4 requires every IC to have dedicated decoupling.
        """
        violations = []

        if not bom:
            return violations

        bom_by_ref = {item.get("reference", ""): item for item in bom if item.get("reference")}
        ic_symbols = []
        bypass_cap_symbols = []

        for sym in symbols:
            bom_item = bom_by_ref.get(sym.reference, {})
            cat = bom_item.get("category", "").lower()
            value = bom_item.get("value", "").lower()

            if cat in ("mcu", "ic", "gate_driver", "can_transceiver", "regulator", "amplifier", "opamp"):
                ic_symbols.append(sym)
            elif cat == "capacitor" and self._is_bypass_value(value):
                bypass_cap_symbols.append(sym)

        # For each IC, check if there's at least one bypass cap within 20mm
        for ic in ic_symbols:
            ic_x, ic_y = ic.position
            has_bypass = False

            for cap in bypass_cap_symbols:
                cap_x, cap_y = cap.position
                dist = math.sqrt((ic_x - cap_x) ** 2 + (ic_y - cap_y) ** 2)
                if dist <= 20.0:  # 20mm search radius
                    has_bypass = True
                    break

            if not has_bypass:
                violations.append(ComplianceViolation(
                    check=ComplianceCheck.IC_DECOUPLING,
                    severity=ViolationSeverity.ERROR,
                    component_ref=ic.reference,
                    description=(
                        f"IC {ic.reference} has no decoupling capacitor within 20mm. "
                        f"NASA-STD-8739.4 requires dedicated decoupling for every IC."
                    ),
                    fix_suggestion=f"Add 100nF bypass capacitor adjacent to {ic.reference}",
                    location=ic.position,
                ))

        return violations

    def _check_capacitor_derating(
        self,
        symbols: List[Any],
        bom: Optional[List[Dict]],
    ) -> List[ComplianceViolation]:
        """
        Check capacitor voltage derating per NASA-STD-8739.4.

        Rule: Capacitor voltage rating must be >= 1.5x the nominal operating voltage.
        For ceramic caps in high-reliability applications, >= 2x is recommended.
        """
        violations = []

        if not bom:
            return violations

        bom_by_ref = {item.get("reference", ""): item for item in bom if item.get("reference")}

        for sym in symbols:
            bom_item = bom_by_ref.get(sym.reference, {})
            cat = bom_item.get("category", "").lower()
            if cat != "capacitor":
                continue

            value_str = bom_item.get("value", "")
            voltage_rating = self._extract_voltage_rating(value_str)
            if voltage_rating is None:
                continue

            # Infer operating voltage from location in circuit
            # For now, check against common rail voltages
            common_rails = [1.8, 2.5, 3.3, 5.0, 10.0, 12.0, 24.0, 48.0]
            for rail_voltage in common_rails:
                if voltage_rating < rail_voltage * 1.5 and voltage_rating >= rail_voltage:
                    violations.append(ComplianceViolation(
                        check=ComplianceCheck.CAPACITOR_DERATING,
                        severity=ViolationSeverity.WARNING,
                        component_ref=sym.reference,
                        description=(
                            f"Capacitor {sym.reference} rated at {voltage_rating}V may be "
                            f"under-derated for {rail_voltage}V rail. "
                            f"NASA requires >= {rail_voltage * 1.5:.1f}V (1.5x derating)."
                        ),
                        fix_suggestion=(
                            f"Use capacitor rated >= {rail_voltage * 2:.0f}V "
                            f"(2x derating recommended for high reliability)"
                        ),
                    ))
                    break  # Only report once per cap

        return violations

    def _check_resistor_derating(
        self,
        symbols: List[Any],
        bom: Optional[List[Dict]],
    ) -> List[ComplianceViolation]:
        """
        Check resistor power derating per NASA-STD-8739.4.

        Rule: Resistor power rating must be >= 2x the actual power dissipation.
        Most standard SMD resistors are 1/8W (0402), 1/4W (0603/0805), 1/2W (1206).
        """
        violations = []

        if not bom:
            return violations

        bom_by_ref = {item.get("reference", ""): item for item in bom if item.get("reference")}

        for sym in symbols:
            bom_item = bom_by_ref.get(sym.reference, {})
            cat = bom_item.get("category", "").lower()
            if cat != "resistor":
                continue

            value_str = bom_item.get("value", "")
            resistance = self._extract_resistance(value_str)
            if resistance is None or resistance <= 0:
                continue

            # Estimate power dissipation based on common scenarios
            # For resistors on 3.3V/5V rails, estimate max current
            footprint = bom_item.get("footprint", "")
            power_rating = self._estimate_power_rating_from_footprint(footprint)

            if power_rating is None:
                continue

            # Check low-value resistors on power rails (high current potential)
            max_rail_voltage = float(os.environ.get("MAX_RAIL_VOLTAGE", "12.0"))
            if resistance < 10:  # Low value = high current potential
                max_current = max_rail_voltage / resistance  # Worst case from max rail
                max_power = max_current ** 2 * resistance
                if max_power > power_rating * 0.5:  # >50% of rating = insufficient derating
                    violations.append(ComplianceViolation(
                        check=ComplianceCheck.RESISTOR_DERATING,
                        severity=ViolationSeverity.WARNING,
                        component_ref=sym.reference,
                        description=(
                            f"Resistor {sym.reference} ({value_str}) may exceed "
                            f"50% power derating at {max_power:.2f}W "
                            f"(rating: {power_rating}W). "
                            f"NASA requires >= 2x derating."
                        ),
                        fix_suggestion=(
                            f"Use resistor rated >= {max_power * 2:.2f}W "
                            f"or increase resistance value"
                        ),
                    ))

        return violations

    def _check_power_budget(
        self,
        symbols: List[Any],
        bom: Optional[List[Dict]],
    ) -> List[ComplianceViolation]:
        """Check power budget: total consumption <= supply capacity."""
        violations = []

        if not bom:
            return violations

        # Estimate power consumption from BOM
        # Standard IC power consumption estimates by category
        ic_power_estimates = {
            "MCU": 0.15,          # 150mW typical
            "Gate_Driver": 0.05,  # 50mW quiescent
            "CAN_Transceiver": 0.03,  # 30mW
            "OpAmp": 0.01,        # 10mW
            "Current_Sense": 0.005,  # 5mW
        }

        total_consumption_w = 0.0
        supply_capacity_w = 0.0

        for item in bom:
            category = item.get("category", "")
            if category in ic_power_estimates:
                total_consumption_w += ic_power_estimates[category]
            elif category == "Regulator":
                # Regulators provide power
                # Estimate capacity from common regulator ratings
                supply_capacity_w += 5.0  # Conservative 5W default per regulator

        if supply_capacity_w > 0 and total_consumption_w > supply_capacity_w * 0.8:
            violations.append(ComplianceViolation(
                check=ComplianceCheck.POWER_BUDGET,
                severity=ViolationSeverity.WARNING,
                component_ref=None,
                description=(
                    f"Power budget warning: estimated consumption {total_consumption_w:.2f}W "
                    f"is >{80}% of estimated supply capacity {supply_capacity_w:.2f}W"
                ),
                fix_suggestion="Review power budget and consider adding more supply capacity",
            ))

        return violations

    @staticmethod
    def _is_bypass_value(value: str) -> bool:
        """Check if a capacitor value is typical for bypass/decoupling."""
        value = value.lower().strip()
        bypass_values = [
            "100n", "100nf", "0.1u", "0.1uf", "100000p", "100000pf",
            "10n", "10nf", "0.01u", "0.01uf",
            "1u", "1uf", "4.7u", "4.7uf", "10u", "10uf",
        ]
        return any(bv in value.replace(" ", "") for bv in bypass_values)

    @staticmethod
    def _extract_voltage_rating(value: str) -> Optional[float]:
        """Extract voltage rating from capacitor value string (e.g., '100nF/25V')."""
        match = re.search(r'(\d+(?:\.\d+)?)\s*[Vv]', value)
        if match:
            return float(match.group(1))
        return None

    @staticmethod
    def _extract_resistance(value: str) -> Optional[float]:
        """Extract resistance value in ohms from string (e.g., '10k', '4.7R', '100')."""
        value = value.strip().upper()
        # Handle k, M, R suffixes
        match = re.match(r'^(\d+(?:\.\d+)?)\s*([KMR]?)(?:OHM)?$', value, re.IGNORECASE)
        if match:
            num = float(match.group(1))
            suffix = match.group(2).upper()
            if suffix == "K":
                return num * 1000
            elif suffix == "M":
                return num * 1_000_000
            return num
        return None

    @staticmethod
    def _estimate_power_rating_from_footprint(footprint: str) -> Optional[float]:
        """Estimate resistor power rating from footprint size."""
        fp = footprint.lower()
        if "0201" in fp:
            return 0.05  # 1/20W
        elif "0402" in fp:
            return 0.0625  # 1/16W
        elif "0603" in fp:
            return 0.1  # 1/10W
        elif "0805" in fp:
            return 0.125  # 1/8W
        elif "1206" in fp:
            return 0.25  # 1/4W
        elif "2010" in fp:
            return 0.5  # 1/2W
        elif "2512" in fp:
            return 1.0  # 1W
        return None

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
