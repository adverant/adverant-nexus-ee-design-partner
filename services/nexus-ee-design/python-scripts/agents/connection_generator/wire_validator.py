"""
Wire Validator - IPC-2221 compliance validation for generated wiring.

Validates generated wires against IPC-2221 standards before accepting them:
- Conductor spacing (voltage-dependent)
- Conductor width (current-dependent)
- Bend angles (minimum 45 degrees)
- Wire crossings (target < 10)
- High-speed signal routing
- Differential pair matching

Author: Nexus EE Design Team
"""

import math
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import yaml


@dataclass
class ValidationViolation:
    """A single IPC-2221 violation."""
    rule_id: str
    severity: str  # 'error' | 'warning'
    message: str
    wire_id: Optional[int] = None
    suggestion: str = ""


@dataclass
class WireValidationReport:
    """Validation results for generated wires."""
    passed: bool
    violations: List[ValidationViolation]
    warnings: List[ValidationViolation]
    crossings_count: int
    ipc_2221_compliant: bool
    total_wires: int = 0


class WireValidator:
    """
    Validates generated wires against IPC-2221 standards.

    Usage:
        validator = WireValidator()
        report = validator.validate(wires, voltage_map, current_map, signal_types)
        if not report.passed:
            # Handle violations
    """

    def __init__(self, rules_path: Optional[str] = None):
        """
        Initialize validator with IPC-2221 rules.

        Args:
            rules_path: Path to ipc_2221_rules.yaml. If None, uses default location.
        """
        if rules_path is None:
            # Default to same directory as this file
            this_dir = os.path.dirname(os.path.abspath(__file__))
            rules_path = os.path.join(this_dir, "ipc_2221_rules.yaml")

        with open(rules_path) as f:
            self.rules = yaml.safe_load(f)

    def validate(
        self,
        wires: List[Dict],
        voltage_map: Dict[str, float],
        current_map: Dict[str, float],
        signal_types: Dict[str, str]
    ) -> WireValidationReport:
        """
        Validate wires against IPC-2221 rules.

        Args:
            wires: List of wire dictionaries with format:
                {
                    "net_name": str,
                    "start_point": {"x": float, "y": float},
                    "end_point": {"x": float, "y": float},
                    "waypoints": [{"x": float, "y": float}, ...],  # optional
                    "width": float,  # mm
                    "signal_type": str,  # power|signal|clock|high_speed|etc
                }
            voltage_map: Net name -> voltage (V)
            current_map: Net name -> current (A)
            signal_types: Net name -> type ('power'|'signal'|'clock'|...)

        Returns:
            WireValidationReport with pass/fail and violations
        """
        violations = []
        warnings = []

        # Check 1: Conductor spacing
        spacing_violations = self._check_conductor_spacing(wires, voltage_map)
        violations.extend([v for v in spacing_violations if v.severity == "error"])
        warnings.extend([v for v in spacing_violations if v.severity == "warning"])

        # Check 2: Acute angles
        angle_violations = self._check_angles(wires)
        violations.extend([v for v in angle_violations if v.severity == "error"])
        warnings.extend([v for v in angle_violations if v.severity == "warning"])

        # Check 3: Wire crossings
        crossings_count = self._count_crossings(wires)
        if crossings_count > 10:
            warnings.append(ValidationViolation(
                rule_id="minimize_crossings",
                severity="warning",
                message=f"{crossings_count} wire crossings (target < 10)",
                suggestion="Reroute to reduce crossings"
            ))

        # Check 4: High-speed signal crossings
        hs_violations = self._check_high_speed_crossings(wires, signal_types)
        violations.extend(hs_violations)

        # Check 5: Conductor width for current
        width_violations = self._check_conductor_width(wires, current_map)
        violations.extend([v for v in width_violations if v.severity == "error"])
        warnings.extend([v for v in width_violations if v.severity == "warning"])

        # Check 6: Differential pair matching
        diff_violations = self._check_differential_pairs(wires, signal_types)
        violations.extend(diff_violations)

        passed = len(violations) == 0
        ipc_2221_compliant = passed and len(warnings) == 0

        return WireValidationReport(
            passed=passed,
            violations=violations,
            warnings=warnings,
            crossings_count=crossings_count,
            ipc_2221_compliant=ipc_2221_compliant,
            total_wires=len(wires)
        )

    def _check_conductor_spacing(
        self,
        wires: List[Dict],
        voltage_map: Dict[str, float]
    ) -> List[ValidationViolation]:
        """Check minimum spacing between conductors."""
        violations = []

        for i, wire1 in enumerate(wires):
            for j, wire2 in enumerate(wires[i+1:], start=i+1):
                min_distance = self._min_distance_between_wires(wire1, wire2)
                required_spacing = self._get_required_spacing(
                    wire1['net_name'],
                    wire2['net_name'],
                    voltage_map
                )

                if min_distance < required_spacing:
                    violations.append(ValidationViolation(
                        rule_id="ipc_spacing",
                        severity="error",
                        message=(
                            f"Spacing violation: {min_distance:.2f}mm < {required_spacing:.2f}mm "
                            f"required between {wire1['net_name']} and {wire2['net_name']}"
                        ),
                        suggestion=f"Increase spacing to {required_spacing:.2f}mm"
                    ))

        return violations

    def _check_angles(self, wires: List[Dict]) -> List[ValidationViolation]:
        """Check that all bend angles are >= 45 degrees."""
        violations = []

        for i, wire in enumerate(wires):
            waypoints = wire.get('waypoints', [])
            if not waypoints:
                continue

            # Build full path: start -> waypoints -> end
            points = [wire['start_point']] + waypoints + [wire['end_point']]

            # Check angles at each waypoint
            for j in range(1, len(points) - 1):
                p1 = points[j - 1]
                p2 = points[j]
                p3 = points[j + 1]

                angle = self._calculate_angle(
                    (p1['x'], p1['y']),
                    (p2['x'], p2['y']),
                    (p3['x'], p3['y'])
                )

                if angle < 45.0:
                    violations.append(ValidationViolation(
                        rule_id="no_acute_angles",
                        severity="error",
                        message=f"Acute angle {angle:.1f}° on {wire['net_name']} (min 45°)",
                        wire_id=i,
                        suggestion="Use 45° or 90° bends only"
                    ))

        return violations

    def _check_high_speed_crossings(
        self,
        wires: List[Dict],
        signal_types: Dict[str, str]
    ) -> List[ValidationViolation]:
        """Check that high-speed signals don't cross other wires."""
        violations = []

        for i, wire1 in enumerate(wires):
            signal_type = signal_types.get(wire1['net_name'], 'signal')
            if signal_type not in ['clock', 'high_speed']:
                continue

            for j, wire2 in enumerate(wires):
                if i == j:
                    continue

                if self._wires_cross(wire1, wire2):
                    violations.append(ValidationViolation(
                        rule_id="high_speed_shielding",
                        severity="error",
                        message=f"High-speed signal {wire1['net_name']} crosses {wire2['net_name']}",
                        wire_id=i,
                        suggestion="Reroute high-speed signal to avoid crossings"
                    ))

        return violations

    def _check_conductor_width(
        self,
        wires: List[Dict],
        current_map: Dict[str, float]
    ) -> List[ValidationViolation]:
        """Check conductor width is adequate for current."""
        violations = []

        for i, wire in enumerate(wires):
            net_name = wire['net_name']
            current = current_map.get(net_name, 0.1)  # Default 100mA
            required_width = self._get_required_width(current)
            actual_width = wire.get('width', 0.25)

            if actual_width < required_width:
                violations.append(ValidationViolation(
                    rule_id="conductor_width",
                    severity="warning",
                    message=(
                        f"{net_name} width {actual_width:.2f}mm insufficient "
                        f"for {current:.1f}A (need {required_width:.2f}mm)"
                    ),
                    wire_id=i,
                    suggestion=f"Increase wire width to {required_width:.2f}mm"
                ))

        return violations

    def _check_differential_pairs(
        self,
        wires: List[Dict],
        signal_types: Dict[str, str]
    ) -> List[ValidationViolation]:
        """Check differential pair matching (length, spacing)."""
        violations = []

        # Find differential pairs (nets ending in _P/_N or +/-)
        diff_pairs: Dict[str, List[Dict]] = {}

        for wire in wires:
            net_name = wire['net_name']
            signal_type = signal_types.get(net_name, 'signal')

            if signal_type != 'differential':
                continue

            # Extract base name
            if net_name.endswith('_P') or net_name.endswith('_N'):
                base = net_name[:-2]
            elif net_name.endswith('+') or net_name.endswith('-'):
                base = net_name[:-1]
            else:
                base = net_name

            if base not in diff_pairs:
                diff_pairs[base] = []
            diff_pairs[base].append(wire)

        # Check each pair
        for base, pair_wires in diff_pairs.items():
            if len(pair_wires) != 2:
                violations.append(ValidationViolation(
                    rule_id="differential_pair_spacing",
                    severity="error",
                    message=f"Differential pair {base} has {len(pair_wires)} wires (expected 2)",
                    suggestion="Ensure both _P and _N nets are routed"
                ))
                continue

            # Check length matching
            length1 = self._wire_length(pair_wires[0])
            length2 = self._wire_length(pair_wires[1])
            length_mismatch = abs(length1 - length2)

            max_mismatch = self.rules['routing_rules'][5]['max_length_mismatch']
            if length_mismatch > max_mismatch:
                violations.append(ValidationViolation(
                    rule_id="differential_pair_spacing",
                    severity="error",
                    message=(
                        f"Differential pair {base} length mismatch {length_mismatch:.2f}mm "
                        f"(max {max_mismatch}mm)"
                    ),
                    suggestion="Match trace lengths by adding serpentine routing"
                ))

        return violations

    def _min_distance_between_wires(
        self,
        wire1: Dict,
        wire2: Dict
    ) -> float:
        """
        Calculate minimum distance between two wires.

        Simplified: Uses point-to-point distance between endpoints.
        More sophisticated version would check all segments.
        """
        p1_start = (wire1['start_point']['x'], wire1['start_point']['y'])
        p1_end = (wire1['end_point']['x'], wire1['end_point']['y'])
        p2_start = (wire2['start_point']['x'], wire2['start_point']['y'])
        p2_end = (wire2['end_point']['x'], wire2['end_point']['y'])

        distances = [
            self._distance(p1_start, p2_start),
            self._distance(p1_start, p2_end),
            self._distance(p1_end, p2_start),
            self._distance(p1_end, p2_end),
        ]

        # If wires have waypoints, check those too
        for wp1 in wire1.get('waypoints', []):
            for wp2 in wire2.get('waypoints', []):
                distances.append(self._distance(
                    (wp1['x'], wp1['y']),
                    (wp2['x'], wp2['y'])
                ))

        return min(distances)

    def _distance(self, p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
        """Euclidean distance between two points."""
        return math.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)

    def _calculate_angle(
        self,
        p1: Tuple[float, float],
        p2: Tuple[float, float],
        p3: Tuple[float, float]
    ) -> float:
        """
        Calculate angle at p2 formed by p1-p2-p3.
        Returns angle in degrees.
        """
        # Vectors
        v1 = (p1[0] - p2[0], p1[1] - p2[1])
        v2 = (p3[0] - p2[0], p3[1] - p2[1])

        # Dot product and magnitudes
        dot = v1[0] * v2[0] + v1[1] * v2[1]
        mag1 = math.sqrt(v1[0]**2 + v1[1]**2)
        mag2 = math.sqrt(v2[0]**2 + v2[1]**2)

        if mag1 == 0 or mag2 == 0:
            return 180.0  # Straight line

        # Angle in radians
        cos_angle = dot / (mag1 * mag2)
        cos_angle = max(-1.0, min(1.0, cos_angle))  # Clamp to [-1, 1]
        angle_rad = math.acos(cos_angle)

        # Convert to degrees
        return math.degrees(angle_rad)

    def _get_required_spacing(
        self,
        net1: str,
        net2: str,
        voltage_map: Dict[str, float]
    ) -> float:
        """Get required IPC-2221 spacing based on voltage."""
        voltage1 = voltage_map.get(net1, 0)
        voltage2 = voltage_map.get(net2, 0)
        max_voltage = max(abs(voltage1), abs(voltage2))

        for rule in self.rules['conductor_spacing']['voltage_classes']:
            if max_voltage <= rule['max_voltage']:
                return rule['min_spacing']

        # Default for very high voltage
        return 6.4

    def _get_required_width(self, current: float) -> float:
        """Get required conductor width for current capacity."""
        for rule in self.rules['conductor_width']['current_capacity']:
            if current <= rule['current_amps']:
                return rule['min_width']

        # For very high current, extrapolate
        return 5.0

    def _count_crossings(self, wires: List[Dict]) -> int:
        """Count total wire crossings."""
        count = 0
        for i, wire1 in enumerate(wires):
            for j, wire2 in enumerate(wires[i+1:], start=i+1):
                if self._wires_cross(wire1, wire2):
                    count += 1
        return count

    def _wires_cross(self, wire1: Dict, wire2: Dict) -> bool:
        """
        Check if two wires cross each other.

        Simplified: Checks if line segments intersect.
        Only checks main segments (start->end), not waypoints.
        """
        # Extract line segments
        p1_start = (wire1['start_point']['x'], wire1['start_point']['y'])
        p1_end = (wire1['end_point']['x'], wire1['end_point']['y'])
        p2_start = (wire2['start_point']['x'], wire2['start_point']['y'])
        p2_end = (wire2['end_point']['x'], wire2['end_point']['y'])

        return self._segments_intersect(p1_start, p1_end, p2_start, p2_end)

    def _segments_intersect(
        self,
        p1: Tuple[float, float],
        p2: Tuple[float, float],
        p3: Tuple[float, float],
        p4: Tuple[float, float]
    ) -> bool:
        """
        Check if line segment p1-p2 intersects with p3-p4.
        Uses cross product method.
        """
        def ccw(A, B, C):
            return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])

        # Segments intersect if endpoints are on opposite sides
        return ccw(p1, p3, p4) != ccw(p2, p3, p4) and ccw(p1, p2, p3) != ccw(p1, p2, p4)

    def _wire_length(self, wire: Dict) -> float:
        """Calculate total wire length including waypoints."""
        points = [wire['start_point']] + wire.get('waypoints', []) + [wire['end_point']]

        total_length = 0.0
        for i in range(len(points) - 1):
            p1 = (points[i]['x'], points[i]['y'])
            p2 = (points[i + 1]['x'], points[i + 1]['y'])
            total_length += self._distance(p1, p2)

        return total_length


# Utility functions for formatting validation reports

def format_validation_report(report: WireValidationReport) -> str:
    """Format validation report for logging."""
    lines = []
    lines.append("=" * 80)
    lines.append("IPC-2221 WIRE VALIDATION REPORT")
    lines.append("=" * 80)
    lines.append(f"Total Wires: {report.total_wires}")
    lines.append(f"Wire Crossings: {report.crossings_count}")
    lines.append(f"Status: {'✅ PASSED' if report.passed else '❌ FAILED'}")
    lines.append(f"IPC-2221 Compliant: {'✅ YES' if report.ipc_2221_compliant else '⚠️  NO (warnings present)'}")
    lines.append("")

    if report.violations:
        lines.append(f"ERRORS ({len(report.violations)}):")
        lines.append("-" * 80)
        for v in report.violations:
            lines.append(f"  [{v.rule_id}] {v.message}")
            if v.suggestion:
                lines.append(f"    → {v.suggestion}")
        lines.append("")

    if report.warnings:
        lines.append(f"WARNINGS ({len(report.warnings)}):")
        lines.append("-" * 80)
        for w in report.warnings:
            lines.append(f"  [{w.rule_id}] {w.message}")
            if w.suggestion:
                lines.append(f"    → {w.suggestion}")
        lines.append("")

    lines.append("=" * 80)
    return "\n".join(lines)
