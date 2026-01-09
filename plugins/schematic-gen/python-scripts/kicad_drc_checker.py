#!/usr/bin/env python3
"""
KiCad DRC (Design Rule Check) Validator

Uses KiCad's actual DRC engine to validate PCB designs against
real industry design rules. This provides ground-truth validation
that cannot be gamed by adjusting scoring thresholds.

The DRC checks include:
- Clearance violations (track-to-track, track-to-pad, etc.)
- Minimum track width violations
- Minimum drill size violations
- Unconnected nets
- Copper pour issues
- Silkscreen overlaps
- And many more IPC-2221 related checks

Usage:
    from kicad_drc_checker import KiCadDRCChecker, run_drc_check

    checker = KiCadDRCChecker()
    result = checker.run_drc("/path/to/board.kicad_pcb")

    if result.passed:
        print("DRC PASSED - no violations")
    else:
        print(f"DRC FAILED - {len(result.errors)} errors, {len(result.warnings)} warnings")
"""

import json
import os
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple


@dataclass
class DRCViolation:
    """Single DRC violation."""
    severity: str  # "error" or "warning"
    type: str  # violation type (clearance, track_width, etc.)
    message: str
    location: Optional[Tuple[float, float]] = None
    items: List[str] = field(default_factory=list)


@dataclass
class DRCResult:
    """Complete DRC result."""
    passed: bool
    errors: List[DRCViolation]
    warnings: List[DRCViolation]
    error_count: int
    warning_count: int
    violations_by_type: Dict[str, int]
    summary: str
    raw_output: str = ""


class KiCadDRCChecker:
    """
    KiCad Design Rule Check validator.

    Uses kicad-cli to run actual DRC checks against PCB files.
    This is ground-truth validation - if DRC fails, the board
    has real design rule violations.
    """

    def __init__(self, kicad_cli_path: Optional[str] = None):
        """
        Initialize DRC checker.

        Args:
            kicad_cli_path: Optional path to kicad-cli. If not provided,
                           will search in PATH.
        """
        self.kicad_cli = kicad_cli_path or shutil.which('kicad-cli')

        if not self.kicad_cli:
            raise RuntimeError(
                "kicad-cli not found in PATH. Install KiCad 7.0+ or provide path."
            )

    def run_drc(self, pcb_path: str, output_format: str = "json") -> DRCResult:
        """
        Run DRC check on a PCB file.

        Args:
            pcb_path: Path to .kicad_pcb file
            output_format: Output format ("json" or "report")

        Returns:
            DRCResult with all violations

        Raises:
            FileNotFoundError: If PCB file doesn't exist
            RuntimeError: If kicad-cli fails
        """
        pcb_path = Path(pcb_path)
        if not pcb_path.exists():
            raise FileNotFoundError(f"PCB file not found: {pcb_path}")

        # Create temp file for DRC output
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as tmp:
            output_path = tmp.name

        try:
            # Run kicad-cli drc
            cmd = [
                self.kicad_cli,
                'pcb',
                'drc',
                '--output', output_path,
                '--format', 'json',
                '--severity-all',  # Include all severities
                str(pcb_path)
            ]

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=120
            )

            # kicad-cli returns non-zero if there are DRC errors
            # This is expected behavior, not a failure

            # Read DRC output
            if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
                with open(output_path, 'r') as f:
                    drc_data = json.load(f)
                return self._parse_drc_json(drc_data, result.stdout + result.stderr)
            else:
                # Try parsing stdout/stderr for DRC info
                return self._parse_drc_text(result.stdout + result.stderr)

        except subprocess.TimeoutExpired:
            raise RuntimeError("DRC check timed out after 120 seconds")
        except json.JSONDecodeError as e:
            # Fall back to text parsing
            return self._parse_drc_text(result.stdout + result.stderr if result else "")
        finally:
            # Clean up temp file
            if os.path.exists(output_path):
                os.unlink(output_path)

    def _parse_drc_json(self, drc_data: Dict, raw_output: str) -> DRCResult:
        """Parse JSON DRC output from kicad-cli."""
        errors = []
        warnings = []
        violations_by_type = {}

        # Parse violations from JSON
        for violation in drc_data.get('violations', []):
            severity = violation.get('severity', 'error')
            vtype = violation.get('type', 'unknown')
            message = violation.get('description', 'No description')

            # Extract location if available
            location = None
            if 'pos' in violation:
                pos = violation['pos']
                location = (pos.get('x', 0), pos.get('y', 0))

            # Extract affected items
            items = []
            for item in violation.get('items', []):
                item_desc = item.get('description', str(item))
                items.append(item_desc)

            v = DRCViolation(
                severity=severity,
                type=vtype,
                message=message,
                location=location,
                items=items
            )

            if severity == 'error':
                errors.append(v)
            else:
                warnings.append(v)

            violations_by_type[vtype] = violations_by_type.get(vtype, 0) + 1

        # Also check for unconnected items
        for unconnected in drc_data.get('unconnected', []):
            v = DRCViolation(
                severity='error',
                type='unconnected_net',
                message=f"Unconnected net: {unconnected.get('net', 'unknown')}",
                items=unconnected.get('items', [])
            )
            errors.append(v)
            violations_by_type['unconnected_net'] = violations_by_type.get('unconnected_net', 0) + 1

        # Generate summary
        passed = len(errors) == 0
        summary = self._generate_summary(errors, warnings, violations_by_type)

        return DRCResult(
            passed=passed,
            errors=errors,
            warnings=warnings,
            error_count=len(errors),
            warning_count=len(warnings),
            violations_by_type=violations_by_type,
            summary=summary,
            raw_output=raw_output
        )

    def _parse_drc_text(self, output: str) -> DRCResult:
        """Parse text DRC output when JSON is not available."""
        errors = []
        warnings = []
        violations_by_type = {}

        # Parse text output for violation patterns
        error_patterns = [
            (r'Error:\s*(.+)', 'error'),
            (r'DRC ERROR:\s*(.+)', 'error'),
            (r'Clearance violation:\s*(.+)', 'clearance'),
            (r'Track width violation:\s*(.+)', 'track_width'),
            (r'Unconnected:\s*(.+)', 'unconnected_net'),
        ]

        warning_patterns = [
            (r'Warning:\s*(.+)', 'warning'),
            (r'DRC WARNING:\s*(.+)', 'warning'),
        ]

        for pattern, vtype in error_patterns:
            for match in re.finditer(pattern, output, re.IGNORECASE):
                v = DRCViolation(
                    severity='error',
                    type=vtype,
                    message=match.group(1).strip()
                )
                errors.append(v)
                violations_by_type[vtype] = violations_by_type.get(vtype, 0) + 1

        for pattern, vtype in warning_patterns:
            for match in re.finditer(pattern, output, re.IGNORECASE):
                v = DRCViolation(
                    severity='warning',
                    type=vtype,
                    message=match.group(1).strip()
                )
                warnings.append(v)
                violations_by_type[vtype] = violations_by_type.get(vtype, 0) + 1

        # Check for success patterns
        if 'DRC violations count: 0' in output or 'No DRC errors' in output.lower():
            passed = True
        else:
            passed = len(errors) == 0

        summary = self._generate_summary(errors, warnings, violations_by_type)

        return DRCResult(
            passed=passed,
            errors=errors,
            warnings=warnings,
            error_count=len(errors),
            warning_count=len(warnings),
            violations_by_type=violations_by_type,
            summary=summary,
            raw_output=output
        )

    def _generate_summary(
        self,
        errors: List[DRCViolation],
        warnings: List[DRCViolation],
        violations_by_type: Dict[str, int]
    ) -> str:
        """Generate human-readable DRC summary."""
        lines = []

        if not errors and not warnings:
            lines.append("DRC PASSED - No violations found")
            return "\n".join(lines)

        if errors:
            lines.append(f"DRC FAILED - {len(errors)} error(s)")
        else:
            lines.append(f"DRC PASSED with {len(warnings)} warning(s)")

        if violations_by_type:
            lines.append("\nViolations by type:")
            for vtype, count in sorted(violations_by_type.items()):
                lines.append(f"  - {vtype}: {count}")

        if errors[:5]:  # Show first 5 errors
            lines.append("\nTop errors:")
            for err in errors[:5]:
                lines.append(f"  [{err.type}] {err.message}")

        return "\n".join(lines)

    def get_detailed_report(self, result: DRCResult) -> str:
        """Generate detailed DRC report."""
        lines = [
            "=" * 60,
            "KICAD DRC REPORT",
            "=" * 60,
            "",
            result.summary,
            "",
        ]

        if result.errors:
            lines.append("-" * 40)
            lines.append("ERRORS (must fix before fabrication):")
            lines.append("-" * 40)
            for i, err in enumerate(result.errors, 1):
                lines.append(f"\n{i}. [{err.type}] {err.message}")
                if err.location:
                    lines.append(f"   Location: ({err.location[0]:.2f}, {err.location[1]:.2f}) mm")
                if err.items:
                    lines.append(f"   Affected: {', '.join(err.items[:3])}")

        if result.warnings:
            lines.append("")
            lines.append("-" * 40)
            lines.append("WARNINGS (should review):")
            lines.append("-" * 40)
            for i, warn in enumerate(result.warnings, 1):
                lines.append(f"\n{i}. [{warn.type}] {warn.message}")

        lines.append("")
        lines.append("=" * 60)

        return "\n".join(lines)


def run_drc_check(pcb_path: str, verbose: bool = True) -> DRCResult:
    """
    Convenience function to run DRC check.

    Args:
        pcb_path: Path to .kicad_pcb file
        verbose: If True, print results to stdout

    Returns:
        DRCResult
    """
    try:
        checker = KiCadDRCChecker()
        result = checker.run_drc(pcb_path)

        if verbose:
            print(checker.get_detailed_report(result))

        return result

    except RuntimeError as e:
        print(f"DRC CHECK FAILED: {e}")
        # Return a failed result
        return DRCResult(
            passed=False,
            errors=[DRCViolation(
                severity='error',
                type='drc_tool_error',
                message=str(e)
            )],
            warnings=[],
            error_count=1,
            warning_count=0,
            violations_by_type={'drc_tool_error': 1},
            summary=f"DRC tool error: {e}"
        )


def main():
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(description='Run KiCad DRC check')
    parser.add_argument('pcb_file', help='Path to .kicad_pcb file')
    parser.add_argument('--json', action='store_true', help='Output as JSON')
    parser.add_argument('--quiet', action='store_true', help='Only output pass/fail')

    args = parser.parse_args()

    result = run_drc_check(args.pcb_file, verbose=not args.quiet and not args.json)

    if args.json:
        output = {
            'passed': result.passed,
            'error_count': result.error_count,
            'warning_count': result.warning_count,
            'violations_by_type': result.violations_by_type,
            'errors': [
                {
                    'type': e.type,
                    'message': e.message,
                    'location': e.location,
                    'items': e.items
                }
                for e in result.errors
            ],
            'warnings': [
                {
                    'type': w.type,
                    'message': w.message
                }
                for w in result.warnings
            ]
        }
        print(json.dumps(output, indent=2))
    elif args.quiet:
        print("PASS" if result.passed else "FAIL")

    # Exit with appropriate code
    import sys
    sys.exit(0 if result.passed else 1)


if __name__ == '__main__':
    main()
