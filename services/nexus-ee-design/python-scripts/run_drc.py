#!/usr/bin/env python3
"""
DRC Runner - Design Rule Check for KiCad PCB files.

This script runs Design Rule Checks on KiCad PCB files using
either the KiCad CLI or direct pcbnew API access.

Usage: run_drc.py <pcb_file_path>

Output (JSON to stdout):
  - passed: boolean
  - totalViolations: number
  - violations: array of {id, code, severity, message, location}
  - warnings: number
  - timestamp: ISO string

Author: Nexus EE Design Team
"""

import json
import os
import re
import subprocess
import sys
import tempfile
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


# KiCad CLI paths by platform
if sys.platform == 'darwin':
    KICAD_CLI = '/Applications/KiCad/KiCad.app/Contents/MacOS/kicad-cli'
    KICAD_PYTHON = '/Applications/KiCad/KiCad.app/Contents/Frameworks/Python.framework/Versions/3.9/bin/python3'
else:
    KICAD_CLI = '/usr/bin/kicad-cli'
    KICAD_PYTHON = '/usr/bin/python3'


class DRCViolation:
    """A single DRC violation."""

    def __init__(
        self,
        code: str,
        severity: str,
        message: str,
        location: Optional[Dict[str, Any]] = None
    ):
        self.id = str(uuid.uuid4())[:8]
        self.code = code
        self.severity = severity
        self.message = message
        self.location = location or {}

    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.id,
            'code': self.code,
            'severity': self.severity,
            'message': self.message,
            'location': self.location
        }


class DRCRunner:
    """Run DRC checks on KiCad PCB files."""

    def __init__(self, pcb_path: Path):
        self.pcb_path = pcb_path
        self.violations: List[DRCViolation] = []
        self.warnings_count = 0

    def run(self) -> Dict[str, Any]:
        """Run DRC and return results."""
        if not self.pcb_path.exists():
            return self._error_result(f"PCB file not found: {self.pcb_path}")

        # Try KiCad CLI first
        if Path(KICAD_CLI).exists():
            result = self._run_kicad_cli_drc()
        else:
            # Fall back to direct file analysis
            result = self._run_static_analysis()

        return result

    def _run_kicad_cli_drc(self) -> Dict[str, Any]:
        """Run DRC using KiCad CLI."""
        try:
            # Create temp file for DRC output
            with tempfile.NamedTemporaryFile(
                mode='w',
                suffix='.rpt',
                delete=False
            ) as f:
                report_path = f.name

            # Run kicad-cli drc
            cmd = [
                KICAD_CLI,
                'pcb', 'drc',
                '--output', report_path,
                '--format', 'report',
                str(self.pcb_path)
            ]

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=120
            )

            # Parse DRC report
            if Path(report_path).exists():
                self._parse_drc_report(Path(report_path))
                os.unlink(report_path)

            # Check exit code
            if result.returncode != 0:
                # Non-zero could mean violations found (which is ok)
                # Parse stderr for additional info
                if result.stderr:
                    self._parse_stderr(result.stderr)

            return self._build_result()

        except subprocess.TimeoutExpired:
            return self._error_result("DRC timed out after 120 seconds")
        except FileNotFoundError:
            # KiCad CLI not found, fall back
            return self._run_static_analysis()
        except Exception as e:
            return self._error_result(f"DRC error: {str(e)}")

    def _parse_drc_report(self, report_path: Path):
        """Parse KiCad DRC report file."""
        content = report_path.read_text()

        # Parse violations from report
        # Format: [error|warning]: <message> @ <location>
        violation_pattern = re.compile(
            r'\[(error|warning)\]:\s*(.+?)(?:\s*@\s*(.+))?$',
            re.MULTILINE
        )

        for match in violation_pattern.finditer(content):
            severity = 'error' if match.group(1) == 'error' else 'warning'
            message = match.group(2).strip()
            location_str = match.group(3)

            # Parse location if present
            location = {}
            if location_str:
                # Try to extract coordinates
                coord_match = re.search(
                    r'\((\d+\.?\d*)\s*mm?,\s*(\d+\.?\d*)\s*mm?\)',
                    location_str
                )
                if coord_match:
                    location = {
                        'x': float(coord_match.group(1)),
                        'y': float(coord_match.group(2))
                    }

            # Determine DRC code
            code = self._classify_violation(message)

            if severity == 'warning':
                self.warnings_count += 1
            else:
                self.violations.append(DRCViolation(
                    code=code,
                    severity=severity,
                    message=message,
                    location=location
                ))

    def _parse_stderr(self, stderr: str):
        """Parse additional errors from stderr."""
        # Look for error patterns
        error_pattern = re.compile(r'Error:\s*(.+)', re.IGNORECASE)
        for match in error_pattern.finditer(stderr):
            self.violations.append(DRCViolation(
                code='CLI_ERROR',
                severity='error',
                message=match.group(1).strip()
            ))

    def _classify_violation(self, message: str) -> str:
        """Classify violation into a DRC code."""
        message_lower = message.lower()

        if 'clearance' in message_lower:
            return 'CLEARANCE'
        elif 'unconnected' in message_lower or 'open net' in message_lower:
            return 'UNCONNECTED'
        elif 'short' in message_lower:
            return 'SHORT'
        elif 'overlap' in message_lower:
            return 'OVERLAP'
        elif 'drill' in message_lower:
            return 'DRILL'
        elif 'via' in message_lower:
            return 'VIA'
        elif 'track' in message_lower or 'trace' in message_lower:
            return 'TRACK'
        elif 'pad' in message_lower:
            return 'PAD'
        elif 'silk' in message_lower:
            return 'SILKSCREEN'
        elif 'mask' in message_lower:
            return 'SOLDER_MASK'
        elif 'courtyard' in message_lower:
            return 'COURTYARD'
        elif 'zone' in message_lower:
            return 'ZONE'
        elif 'net' in message_lower:
            return 'NET'
        else:
            return 'OTHER'

    def _run_static_analysis(self) -> Dict[str, Any]:
        """Run static analysis on PCB file (no KiCad CLI)."""
        try:
            content = self.pcb_path.read_text()
            self._analyze_pcb_content(content)
            return self._build_result()
        except Exception as e:
            return self._error_result(f"Static analysis error: {str(e)}")

    def _analyze_pcb_content(self, content: str):
        """Analyze PCB content for common issues."""
        # Check for unrouted nets (no tracks)
        has_tracks = '(segment' in content or '(arc' in content
        footprints = re.findall(r'\(footprint\s+"([^"]+)"', content)

        if footprints and not has_tracks:
            self.violations.append(DRCViolation(
                code='UNCONNECTED',
                severity='error',
                message=f'No traces found - {len(footprints)} components are unrouted'
            ))

        # Check for missing net assignments
        nets = set(re.findall(r'\(net\s+(\d+)\s+"([^"]+)"', content))
        if len(nets) < 2:
            self.warnings_count += 1

        # Check for components without pads
        for fp in footprints:
            # This is a simple heuristic check
            if 'via' not in fp.lower() and 'test' not in fp.lower():
                pad_count = content.count(f'(pad ')
                if pad_count == 0:
                    self.violations.append(DRCViolation(
                        code='PAD',
                        severity='warning',
                        message=f'Footprint may be missing pads: {fp}'
                    ))
                    break  # Don't spam violations

        # Check for edge cuts
        if '(layer "Edge.Cuts")' not in content and 'Edge.Cuts' not in content:
            self.warnings_count += 1

        # Check for copper zones without net
        zone_pattern = re.compile(r'\(zone\s+\(net\s+0\)', re.DOTALL)
        unassigned_zones = zone_pattern.findall(content)
        if unassigned_zones:
            self.violations.append(DRCViolation(
                code='ZONE',
                severity='error',
                message=f'{len(unassigned_zones)} copper zone(s) have no net assigned'
            ))

    def _build_result(self) -> Dict[str, Any]:
        """Build the final result dictionary."""
        total_violations = len(self.violations)
        passed = total_violations == 0

        return {
            'passed': passed,
            'totalViolations': total_violations,
            'violations': [v.to_dict() for v in self.violations],
            'warnings': self.warnings_count,
            'timestamp': datetime.now().isoformat()
        }

    def _error_result(self, message: str) -> Dict[str, Any]:
        """Build error result."""
        return {
            'passed': False,
            'totalViolations': 1,
            'violations': [{
                'id': str(uuid.uuid4())[:8],
                'code': 'ERROR',
                'severity': 'error',
                'message': message,
                'location': {}
            }],
            'warnings': 0,
            'timestamp': datetime.now().isoformat()
        }


def main():
    """Main entry point."""
    if len(sys.argv) < 2:
        print(json.dumps({
            'passed': False,
            'totalViolations': 1,
            'violations': [{
                'id': 'arg_error',
                'code': 'ERROR',
                'severity': 'error',
                'message': 'Usage: run_drc.py <pcb_file_path>',
                'location': {}
            }],
            'warnings': 0,
            'timestamp': datetime.now().isoformat()
        }))
        sys.exit(1)

    pcb_path = Path(sys.argv[1])
    runner = DRCRunner(pcb_path)
    result = runner.run()
    print(json.dumps(result))


if __name__ == '__main__':
    main()
