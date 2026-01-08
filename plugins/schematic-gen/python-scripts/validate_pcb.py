#!/usr/bin/env python3
"""
PCB Design Rule Check (DRC) Validation

Validates KiCad PCB files against design rules:
1. Clearance - minimum distance between traces
2. Trace width - minimum width per net class
3. Via size - minimum drill and annular ring
4. Connectivity - all pads must connect to nets
5. Overlap detection - traces on same layer can't cross

This is REAL validation, not simulated.

Usage:
    python validate_pcb.py --path /path/to/board.kicad_pcb --json
"""

import argparse
import json
import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Set
from collections import defaultdict
import math


@dataclass
class Violation:
    """A DRC violation."""
    rule: str
    severity: str  # "error", "warning"
    message: str
    location: Optional[Tuple[float, float]] = None
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Trace:
    """A PCB trace segment."""
    start: Tuple[float, float]
    end: Tuple[float, float]
    width: float
    layer: str
    net_id: int
    uuid: str


@dataclass
class Via:
    """A via."""
    x: float
    y: float
    drill: float
    size: float
    net_id: int
    uuid: str


@dataclass
class Pad:
    """A component pad."""
    x: float
    y: float
    size: Tuple[float, float]  # width, height
    shape: str
    layers: List[str]
    net_id: int
    net_name: str
    component_ref: str
    pad_number: str
    uuid: str


@dataclass
class Zone:
    """A copper zone (pour)."""
    net_id: int
    net_name: str
    layer: str
    uuid: str


class DRCValidator:
    """
    Validates PCB design against design rules.

    Design Rules (defaults):
    - Clearance: 0.15mm minimum
    - Trace width: 0.25mm minimum (signal), varies by net class
    - Via drill: 0.3mm minimum
    - Via annular ring: 0.15mm minimum
    - Overlap: traces can't overlap on same layer
    """

    def __init__(self, pcb_path: str):
        self.path = pcb_path
        self.content = ""

        # Design rules
        self.rules = {
            'clearance_min': 0.15,  # mm
            'trace_width_min': 0.25,  # mm
            'trace_width_power': 1.0,  # mm for power nets
            'via_drill_min': 0.3,  # mm
            'via_annular_ring_min': 0.15,  # mm
        }

        # Net classes (name -> min trace width)
        self.net_classes = {
            'GND': 0.5,
            'VCC': 0.5,
            '+3.3V': 0.3,
            '+5V': 0.5,
            '+12V': 1.0,
            '+48V': 2.0,
            '+58V': 3.0,
            'default': 0.25,
        }

        # Parsed elements
        self.traces: List[Trace] = []
        self.vias: List[Via] = []
        self.pads: List[Pad] = []
        self.zones: List[Zone] = []
        self.nets: Dict[int, str] = {}  # net_id -> net_name

    def load(self):
        """Load PCB file."""
        if not os.path.exists(self.path):
            raise FileNotFoundError(f"PCB not found: {self.path}")

        with open(self.path, 'r', encoding='utf-8') as f:
            self.content = f.read()

    def validate(self) -> Dict[str, Any]:
        """
        Run full DRC validation.

        Returns:
            Dictionary with validation results and violations
        """
        self.load()
        self._parse_pcb()

        violations: List[Violation] = []

        # Run all checks
        violations.extend(self._check_trace_width())
        violations.extend(self._check_clearance())
        violations.extend(self._check_via_size())
        violations.extend(self._check_connectivity())
        violations.extend(self._check_overlaps())

        # Categorize violations
        errors = [v for v in violations if v.severity == 'error']
        warnings = [v for v in violations if v.severity == 'warning']

        return {
            'passed': len(errors) == 0,
            'error_count': len(errors),
            'warning_count': len(warnings),
            'violations': [
                {
                    'rule': v.rule,
                    'severity': v.severity,
                    'message': v.message,
                    'location': v.location,
                    'details': v.details
                }
                for v in violations
            ],
            'summary': {
                'trace_count': len(self.traces),
                'via_count': len(self.vias),
                'pad_count': len(self.pads),
                'net_count': len(self.nets),
                'zone_count': len(self.zones)
            }
        }

    def _parse_pcb(self):
        """Parse PCB elements."""
        self._parse_nets()
        self._parse_traces()
        self._parse_vias()
        self._parse_pads()
        self._parse_zones()

    def _parse_nets(self):
        """Parse net definitions."""
        net_pattern = r'\(net\s+(\d+)\s+"([^"]*)"\)'
        for match in re.finditer(net_pattern, self.content):
            net_id = int(match.group(1))
            net_name = match.group(2)
            self.nets[net_id] = net_name

    def _parse_traces(self):
        """Parse trace segments."""
        segment_pattern = (
            r'\(segment\s+'
            r'\(start\s+([\d.-]+)\s+([\d.-]+)\)\s+'
            r'\(end\s+([\d.-]+)\s+([\d.-]+)\)\s+'
            r'\(width\s+([\d.-]+)\)\s+'
            r'\(layer\s+"([^"]+)"\)\s+'
            r'\(net\s+(\d+)\)'
            r'.*?'
            r'\(uuid\s+"([^"]+)"\)'
        )

        for match in re.finditer(segment_pattern, self.content, re.DOTALL):
            self.traces.append(Trace(
                start=(float(match.group(1)), float(match.group(2))),
                end=(float(match.group(3)), float(match.group(4))),
                width=float(match.group(5)),
                layer=match.group(6),
                net_id=int(match.group(7)),
                uuid=match.group(8)
            ))

    def _parse_vias(self):
        """Parse vias."""
        via_pattern = (
            r'\(via\s+'
            r'\(at\s+([\d.-]+)\s+([\d.-]+)\)\s+'
            r'\(size\s+([\d.-]+)\)\s+'
            r'\(drill\s+([\d.-]+)\)\s+'
            r'.*?'
            r'\(net\s+(\d+)\)'
            r'.*?'
            r'\(uuid\s+"([^"]+)"\)'
        )

        for match in re.finditer(via_pattern, self.content, re.DOTALL):
            self.vias.append(Via(
                x=float(match.group(1)),
                y=float(match.group(2)),
                size=float(match.group(3)),
                drill=float(match.group(4)),
                net_id=int(match.group(5)),
                uuid=match.group(6)
            ))

    def _parse_pads(self):
        """Parse component pads."""
        # Find all footprints and their pads
        footprint_pattern = r'\(footprint\s+"[^"]*".*?\(property\s+"Reference"\s+"([^"]+)"'

        # Simplified pad extraction
        pad_pattern = (
            r'\(pad\s+"([^"]+)".*?'
            r'\(at\s+([\d.-]+)\s+([\d.-]+)'
            r'.*?'
            r'\(net\s+(\d+)\s+"([^"]*)"\)'
            r'.*?'
            r'\(uuid\s+"([^"]+)"\)'
        )

        # Find footprints and extract their pads
        for fp_match in re.finditer(r'\(footprint\s+"([^"]*)".*?\(at\s+([\d.-]+)\s+([\d.-]+)', self.content, re.DOTALL):
            fp_name = fp_match.group(1)
            fp_x = float(fp_match.group(2))
            fp_y = float(fp_match.group(3))

            # Get the full footprint block
            start = fp_match.start()
            depth = 0
            end = start
            for i, char in enumerate(self.content[start:]):
                if char == '(':
                    depth += 1
                elif char == ')':
                    depth -= 1
                    if depth == 0:
                        end = start + i + 1
                        break

            fp_block = self.content[start:end]

            # Get reference
            ref_match = re.search(r'\(property\s+"Reference"\s+"([^"]+)"', fp_block)
            ref = ref_match.group(1) if ref_match else "?"

            # Find pads in this footprint
            for pad_match in re.finditer(pad_pattern, fp_block, re.DOTALL):
                pad_num = pad_match.group(1)
                pad_x = float(pad_match.group(2))
                pad_y = float(pad_match.group(3))
                net_id = int(pad_match.group(4))
                net_name = pad_match.group(5)
                uuid = pad_match.group(6)

                self.pads.append(Pad(
                    x=fp_x + pad_x,
                    y=fp_y + pad_y,
                    size=(1.0, 1.0),  # Simplified
                    shape="rect",
                    layers=["F.Cu"],
                    net_id=net_id,
                    net_name=net_name,
                    component_ref=ref,
                    pad_number=pad_num,
                    uuid=uuid
                ))

    def _parse_zones(self):
        """Parse copper zones."""
        zone_pattern = (
            r'\(zone\s+'
            r'\(net\s+(\d+)\)\s+'
            r'\(net_name\s+"([^"]*)"\)'
            r'.*?'
            r'\(layer\s+"([^"]+)"\)'
            r'.*?'
            r'\(uuid\s+"([^"]+)"\)'
        )

        for match in re.finditer(zone_pattern, self.content, re.DOTALL):
            self.zones.append(Zone(
                net_id=int(match.group(1)),
                net_name=match.group(2),
                layer=match.group(3),
                uuid=match.group(4)
            ))

    def _check_trace_width(self) -> List[Violation]:
        """Check trace widths against net class requirements."""
        violations = []

        for trace in self.traces:
            net_name = self.nets.get(trace.net_id, "")

            # Get minimum width for this net
            min_width = self.net_classes.get(net_name, self.net_classes['default'])

            # Check for power nets
            if any(p in net_name.upper() for p in ['VCC', 'VDD', '+', 'PWR', 'POWER']):
                min_width = max(min_width, self.rules['trace_width_power'])

            if trace.width < min_width:
                violations.append(Violation(
                    rule='trace_width',
                    severity='warning' if trace.width >= self.rules['trace_width_min'] else 'error',
                    message=f"Trace width {trace.width}mm < minimum {min_width}mm for net '{net_name}'",
                    location=trace.start,
                    details={
                        'actual_width': trace.width,
                        'required_width': min_width,
                        'net': net_name,
                        'layer': trace.layer
                    }
                ))

        return violations

    def _check_clearance(self) -> List[Violation]:
        """Check clearance between traces."""
        violations = []
        min_clearance = self.rules['clearance_min']

        # Check trace-to-trace clearance on same layer
        for i, trace1 in enumerate(self.traces):
            for trace2 in self.traces[i+1:]:
                # Skip if different layers
                if trace1.layer != trace2.layer:
                    continue

                # Skip if same net
                if trace1.net_id == trace2.net_id:
                    continue

                # Calculate minimum distance between trace segments
                dist = self._segment_distance(
                    trace1.start, trace1.end,
                    trace2.start, trace2.end
                )

                # Account for trace widths
                clearance = dist - (trace1.width / 2) - (trace2.width / 2)

                if clearance < min_clearance:
                    violations.append(Violation(
                        rule='clearance',
                        severity='error',
                        message=f"Clearance {clearance:.3f}mm < minimum {min_clearance}mm",
                        location=trace1.start,
                        details={
                            'actual_clearance': clearance,
                            'required_clearance': min_clearance,
                            'net1': self.nets.get(trace1.net_id, "?"),
                            'net2': self.nets.get(trace2.net_id, "?"),
                            'layer': trace1.layer
                        }
                    ))

        return violations

    def _check_via_size(self) -> List[Violation]:
        """Check via drill and annular ring sizes."""
        violations = []

        for via in self.vias:
            # Check drill size
            if via.drill < self.rules['via_drill_min']:
                violations.append(Violation(
                    rule='via_drill',
                    severity='error',
                    message=f"Via drill {via.drill}mm < minimum {self.rules['via_drill_min']}mm",
                    location=(via.x, via.y),
                    details={
                        'actual_drill': via.drill,
                        'required_drill': self.rules['via_drill_min']
                    }
                ))

            # Check annular ring
            annular_ring = (via.size - via.drill) / 2
            if annular_ring < self.rules['via_annular_ring_min']:
                violations.append(Violation(
                    rule='via_annular_ring',
                    severity='error',
                    message=f"Via annular ring {annular_ring:.3f}mm < minimum {self.rules['via_annular_ring_min']}mm",
                    location=(via.x, via.y),
                    details={
                        'actual_ring': annular_ring,
                        'required_ring': self.rules['via_annular_ring_min'],
                        'via_size': via.size,
                        'via_drill': via.drill
                    }
                ))

        return violations

    def _check_connectivity(self) -> List[Violation]:
        """Check that all pads are connected to nets."""
        violations = []

        for pad in self.pads:
            if pad.net_id == 0 or not pad.net_name:
                violations.append(Violation(
                    rule='connectivity',
                    severity='warning',
                    message=f"Pad {pad.component_ref}.{pad.pad_number} not connected to any net",
                    location=(pad.x, pad.y),
                    details={
                        'component': pad.component_ref,
                        'pad': pad.pad_number
                    }
                ))

        return violations

    def _check_overlaps(self) -> List[Violation]:
        """Check for overlapping traces on same layer."""
        violations = []

        for i, trace1 in enumerate(self.traces):
            for trace2 in self.traces[i+1:]:
                # Skip if different layers
                if trace1.layer != trace2.layer:
                    continue

                # Skip if same net (traces can overlap on same net)
                if trace1.net_id == trace2.net_id:
                    continue

                # Check for intersection
                if self._segments_intersect(trace1.start, trace1.end, trace2.start, trace2.end):
                    violations.append(Violation(
                        rule='overlap',
                        severity='error',
                        message=f"Traces intersect on layer {trace1.layer}",
                        location=trace1.start,
                        details={
                            'net1': self.nets.get(trace1.net_id, "?"),
                            'net2': self.nets.get(trace2.net_id, "?"),
                            'layer': trace1.layer
                        }
                    ))

        return violations

    def _segment_distance(self, p1: Tuple[float, float], p2: Tuple[float, float],
                          p3: Tuple[float, float], p4: Tuple[float, float]) -> float:
        """Calculate minimum distance between two line segments."""
        # Simplified: use minimum of endpoint distances
        distances = [
            self._point_distance(p1, p3),
            self._point_distance(p1, p4),
            self._point_distance(p2, p3),
            self._point_distance(p2, p4),
            self._point_to_segment_distance(p1, p3, p4),
            self._point_to_segment_distance(p2, p3, p4),
            self._point_to_segment_distance(p3, p1, p2),
            self._point_to_segment_distance(p4, p1, p2),
        ]
        return min(distances)

    def _point_distance(self, p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
        """Calculate distance between two points."""
        return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

    def _point_to_segment_distance(self, p: Tuple[float, float],
                                    s1: Tuple[float, float], s2: Tuple[float, float]) -> float:
        """Calculate distance from point to line segment."""
        # Vector from s1 to s2
        dx = s2[0] - s1[0]
        dy = s2[1] - s1[1]
        length_sq = dx*dx + dy*dy

        if length_sq == 0:
            return self._point_distance(p, s1)

        # Project point onto line
        t = max(0, min(1, ((p[0] - s1[0]) * dx + (p[1] - s1[1]) * dy) / length_sq))

        # Closest point on segment
        closest = (s1[0] + t * dx, s1[1] + t * dy)
        return self._point_distance(p, closest)

    def _segments_intersect(self, p1: Tuple[float, float], p2: Tuple[float, float],
                            p3: Tuple[float, float], p4: Tuple[float, float]) -> bool:
        """Check if two line segments intersect."""
        def ccw(a, b, c):
            return (c[1] - a[1]) * (b[0] - a[0]) > (b[1] - a[1]) * (c[0] - a[0])

        return (ccw(p1, p3, p4) != ccw(p2, p3, p4)) and (ccw(p1, p2, p3) != ccw(p1, p2, p4))


def main():
    parser = argparse.ArgumentParser(
        description='Validate KiCad PCB against design rules'
    )
    parser.add_argument(
        '--path', '-p',
        type=str,
        required=True,
        help='Path to .kicad_pcb file'
    )
    parser.add_argument(
        '--json',
        action='store_true',
        help='Output as JSON'
    )
    parser.add_argument(
        '--strict',
        action='store_true',
        help='Treat warnings as errors'
    )

    args = parser.parse_args()

    try:
        validator = DRCValidator(args.path)
        result = validator.validate()

        if args.json:
            print(json.dumps(result, indent=2))
        else:
            # Human-readable output
            status = "PASSED" if result['passed'] else "FAILED"
            print(f"DRC Validation: {status}")
            print(f"  Errors: {result['error_count']}")
            print(f"  Warnings: {result['warning_count']}")
            print(f"\nBoard Statistics:")
            print(f"  Traces: {result['summary']['trace_count']}")
            print(f"  Vias: {result['summary']['via_count']}")
            print(f"  Pads: {result['summary']['pad_count']}")
            print(f"  Nets: {result['summary']['net_count']}")

            if result['violations']:
                print(f"\nViolations:")
                for v in result['violations']:
                    prefix = "[ERROR]" if v['severity'] == 'error' else "[WARN]"
                    loc = f" at ({v['location'][0]:.2f}, {v['location'][1]:.2f})" if v['location'] else ""
                    print(f"  {prefix} {v['rule']}: {v['message']}{loc}")

        # Exit with error if failed
        if args.strict:
            sys.exit(0 if result['error_count'] == 0 and result['warning_count'] == 0 else 1)
        else:
            sys.exit(0 if result['passed'] else 1)

    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
