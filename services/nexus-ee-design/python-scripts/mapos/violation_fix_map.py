#!/usr/bin/env python3
"""
Violation Fix Mapping - Maps DRC violation types to concrete fix operations.

This is the CRITICAL bridge between LLM suggestions and actual pcbnew operations.
Without this mapping, the Gaming AI generates suggestions that never execute.

The mapping provides:
1. Violation type normalization (handles different naming conventions)
2. Ordered list of fix operations to try for each violation type
3. Execution via subprocess to avoid pcbnew import issues
4. Success rate tracking for prioritization

Usage:
    from violation_fix_map import get_fixes_for_violation, apply_fixes_for_violations

    # Get fixes for a specific violation type
    fixes = get_fixes_for_violation('clearance')

    # Apply all appropriate fixes for a list of violations
    result = apply_fixes_for_violations(pcb_path, violations)
"""

import os
import sys
import json
import shutil
import subprocess
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime


def find_kicad_python() -> Optional[str]:
    """
    Find Python interpreter with pcbnew available.

    Searches platform-specific paths for KiCad's bundled Python.
    """
    # Check if we're already running with pcbnew
    try:
        import pcbnew
        return sys.executable
    except ImportError:
        pass

    # Platform-specific KiCad Python paths
    candidates = [
        # macOS KiCad 8.0
        '/Applications/KiCad/KiCad.app/Contents/Frameworks/Python.framework/Versions/3.9/bin/python3',
        '/Applications/KiCad/KiCad.app/Contents/Frameworks/Python.framework/Versions/3.11/bin/python3',
        # Linux (system KiCad)
        '/usr/bin/python3',
        # Docker/K8s container
        '/opt/kicad/bin/python3',
        '/usr/lib/kicad/bin/python3',
        # Snap KiCad
        '/snap/kicad/current/usr/bin/python3',
        # System Python (might have pcbnew installed)
        shutil.which('python3'),
        sys.executable,
    ]

    for path in candidates:
        if path and Path(path).exists():
            # Verify pcbnew is importable
            try:
                result = subprocess.run(
                    [path, '-c', 'import pcbnew; print("ok")'],
                    capture_output=True,
                    text=True,
                    timeout=10,
                    env={**os.environ, 'DISPLAY': '', 'KICAD_NO_TRACKING': '1'}
                )
                if result.returncode == 0 and 'ok' in result.stdout:
                    return path
            except (subprocess.TimeoutExpired, Exception):
                continue

    return None


@dataclass
class FixOperation:
    """
    A concrete fix operation that can be executed on a PCB file.

    Attributes:
        name: Human-readable name for logging
        script: Python script filename (relative to this file's directory)
        operation: Subcommand/operation to pass to script
        params: Additional CLI parameters
        success_rate: Historical success rate (0.0-1.0) for prioritization
    """
    name: str
    script: str
    operation: str = ""
    params: Dict[str, str] = field(default_factory=dict)
    success_rate: float = 0.5

    def execute(self, pcb_path: Path, kicad_python: Optional[str] = None) -> Dict[str, Any]:
        """
        Execute this fix operation on a PCB file.

        Args:
            pcb_path: Path to the .kicad_pcb file
            kicad_python: Path to Python with pcbnew (auto-detected if None)

        Returns:
            Dict with 'success' key and operation results
        """
        if kicad_python is None:
            kicad_python = find_kicad_python()

        if not kicad_python:
            return {
                'success': False,
                'error': 'KiCad Python not found - cannot execute pcbnew operations'
            }

        cmd = self._build_command(pcb_path, kicad_python)

        try:
            # Set headless environment
            env = os.environ.copy()
            env['DISPLAY'] = ''
            env['KICAD_NO_TRACKING'] = '1'

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=120,  # 2 minute timeout for complex operations
                env=env
            )

            if result.returncode != 0:
                return {
                    'success': False,
                    'error': result.stderr or f'Script returned exit code {result.returncode}',
                    'stdout': result.stdout,
                    'command': ' '.join(cmd)
                }

            # Try to parse JSON output
            try:
                output = json.loads(result.stdout)
                if 'success' not in output:
                    output['success'] = True
                return output
            except json.JSONDecodeError:
                # Non-JSON output is still success if returncode was 0
                return {
                    'success': True,
                    'output': result.stdout,
                    'raw': True
                }

        except subprocess.TimeoutExpired:
            return {
                'success': False,
                'error': f'Operation timed out after 120 seconds',
                'command': ' '.join(cmd)
            }
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'command': ' '.join(cmd)
            }

    def _build_command(self, pcb_path: Path, kicad_python: str) -> List[str]:
        """Build the subprocess command."""
        script_dir = Path(__file__).parent
        script_path = script_dir / self.script

        if not script_path.exists():
            raise FileNotFoundError(f"Fix script not found: {script_path}")

        cmd = [kicad_python, str(script_path)]

        if self.operation:
            cmd.append(self.operation)

        cmd.append(str(pcb_path))
        cmd.append('--json')

        for key, value in self.params.items():
            if value is True:
                cmd.append(f'--{key}')
            elif value is not False and value is not None:
                cmd.extend([f'--{key}', str(value)])

        return cmd


# =============================================================================
# THE CRITICAL MAPPING: violation_type → list of fix operations to try
# =============================================================================
# Each violation type maps to an ordered list of fixes. The system tries
# each fix in order until one succeeds (reducing violations for that type).
# =============================================================================

VIOLATION_FIX_MAP: Dict[str, List[FixOperation]] = {

    # =========================================================================
    # ZONE ISSUES
    # =========================================================================
    'zone_net_assignment_error': [
        FixOperation(
            name='fix_zone_nets',
            script='kicad_pcb_fixer.py',
            operation='zone-nets',
            success_rate=0.95
        ),
    ],
    'zone_fill_missing': [
        FixOperation(
            name='fill_zones',
            script='kicad_zone_filler.py',
            success_rate=0.99
        ),
    ],
    'zone_clearance': [
        FixOperation(
            name='adjust_zone_clearances',
            script='footprint_dfm_fixer.py',
            params={'fix-zone-clearances': True},
            success_rate=0.70
        ),
    ],

    # =========================================================================
    # VIA ISSUES
    # =========================================================================
    'via_dangling': [
        FixOperation(
            name='remove_dangling_vias',
            script='kicad_pcb_fixer.py',
            operation='dangling-vias',
            success_rate=0.90
        ),
    ],
    'dangling_via': [  # Alternate naming
        FixOperation(
            name='remove_dangling_vias',
            script='kicad_pcb_fixer.py',
            operation='dangling-vias',
            success_rate=0.90
        ),
    ],

    # =========================================================================
    # TRACK ISSUES
    # =========================================================================
    'track_dangling': [
        FixOperation(
            name='remove_dangling_tracks',
            script='kicad_dangling_track_fixer.py',
            params={'remove': True},
            success_rate=0.80
        ),
        FixOperation(
            name='extend_dangling_tracks',
            script='kicad_dangling_track_fixer.py',
            params={'extend': True},
            success_rate=0.60
        ),
    ],
    'track_width': [
        FixOperation(
            name='adjust_power_traces',
            script='kicad_trace_adjuster.py',
            operation='power',
            params={'width': '2.0'},
            success_rate=0.85
        ),
    ],

    # =========================================================================
    # CLEARANCE ISSUES
    # =========================================================================
    'clearance': [
        FixOperation(
            name='fix_design_settings',
            script='kicad_pcb_fixer.py',
            operation='design-settings',
            success_rate=0.60
        ),
    ],
    'clearance_violation': [
        FixOperation(
            name='fix_design_settings',
            script='kicad_pcb_fixer.py',
            operation='design-settings',
            success_rate=0.60
        ),
    ],

    # =========================================================================
    # SILKSCREEN ISSUES
    # =========================================================================
    'silk_over_copper': [
        FixOperation(
            name='move_silk_to_fab',
            script='kicad_silk_fixer.py',
            params={'no-backup': True},
            success_rate=0.95
        ),
    ],
    'silk_overlap': [
        FixOperation(
            name='resize_silk',
            script='footprint_dfm_fixer.py',
            params={'fix-silk-overlap': True},
            success_rate=0.85
        ),
    ],
    'silkscreen_overlap': [
        FixOperation(
            name='resize_silk',
            script='footprint_dfm_fixer.py',
            params={'fix-silk-overlap': True},
            success_rate=0.85
        ),
    ],

    # =========================================================================
    # SOLDER MASK ISSUES
    # =========================================================================
    'solder_mask_bridge': [
        FixOperation(
            name='tent_vias',
            script='kicad_mask_fixer.py',
            params={'no-backup': True},
            success_rate=0.90
        ),
    ],
    'mask_aperture': [
        FixOperation(
            name='fix_mask_expansion',
            script='kicad_mask_fixer.py',
            params={'no-backup': True, 'reduce-expansion': True},
            success_rate=0.80
        ),
    ],

    # =========================================================================
    # UNCONNECTED/ORPHAN ISSUES
    # =========================================================================
    'unconnected_items': [
        FixOperation(
            name='assign_orphan_nets',
            script='kicad_net_assigner.py',
            success_rate=0.70
        ),
    ],
    'unconnected': [
        FixOperation(
            name='assign_orphan_nets',
            script='kicad_net_assigner.py',
            success_rate=0.70
        ),
    ],

    # =========================================================================
    # FOOTPRINT ISSUES
    # =========================================================================
    'lib_footprint_issues': [
        FixOperation(
            name='check_footprint_library',
            script='kicad_footprint_updater.py',
            params={'check': True},
            success_rate=0.70
        ),
    ],
    'lib_footprint_mismatch': [
        FixOperation(
            name='check_footprint_library',
            script='kicad_footprint_updater.py',
            params={'check': True},
            success_rate=0.70
        ),
    ],
    'footprint_type_mismatch': [
        FixOperation(
            name='check_footprint_library',
            script='kicad_footprint_updater.py',
            params={'check': True},
            success_rate=0.50
        ),
    ],

    # =========================================================================
    # COURTYARD ISSUES
    # =========================================================================
    'courtyard_overlap': [
        # Note: Courtyard overlap usually requires component movement
        # which is complex. For now, flag for manual review.
    ],

    # =========================================================================
    # DRILL ISSUES
    # =========================================================================
    'drill_size': [
        FixOperation(
            name='fix_drill_sizes',
            script='kicad_pcb_fixer.py',
            operation='design-settings',  # Sets minimum drill in design rules
            success_rate=0.75
        ),
    ],
    'annular_ring': [
        FixOperation(
            name='fix_annular_ring',
            script='kicad_pcb_fixer.py',
            operation='design-settings',  # Sets minimum annular ring
            success_rate=0.70
        ),
    ],

    # =========================================================================
    # SPECIAL CASES
    # =========================================================================
    'shorting_items': [
        # Shorts require manual intervention - cannot auto-fix safely
    ],
    'net_conflict': [
        # Net conflicts require schematic review
    ],
}


def normalize_violation_type(violation_type: str) -> str:
    """
    Normalize violation type string for consistent matching.

    Handles:
    - Different casing (Clearance vs clearance)
    - Dashes vs underscores (silk-over-copper vs silk_over_copper)
    - Spaces (Silk Over Copper)
    - Trailing words (_violation, _error, etc.)
    """
    normalized = violation_type.lower()
    normalized = normalized.replace('-', '_').replace(' ', '_')

    # Remove common suffixes
    for suffix in ['_violation', '_error', '_issue', '_warning']:
        if normalized.endswith(suffix):
            normalized = normalized[:-len(suffix)]

    return normalized


def get_fixes_for_violation(violation_type: str) -> List[FixOperation]:
    """
    Get ordered list of fix operations to try for a violation type.

    Args:
        violation_type: The DRC violation type string

    Returns:
        List of FixOperation objects to try, ordered by success rate
    """
    normalized = normalize_violation_type(violation_type)

    # Direct match
    if normalized in VIOLATION_FIX_MAP:
        return VIOLATION_FIX_MAP[normalized]

    # Partial match - check if any key is contained in the normalized type
    for key in VIOLATION_FIX_MAP:
        if key in normalized or normalized in key:
            return VIOLATION_FIX_MAP[key]

    # No match found
    return []


def apply_suggestion_to_pcb(
    pcb_path: Path,
    suggestion: Any,
    kicad_python: Optional[str] = None
) -> Dict[str, Any]:
    """
    Apply an LLM suggestion to a PCB file by mapping it to a real fix operation.

    This is the bridge between Gaming AI suggestions and real pcbnew operations.

    Args:
        pcb_path: Path to PCB file
        suggestion: ModificationSuggestion or dict with 'mod_type' field
        kicad_python: Python executable with pcbnew (auto-detected if None)

    Returns:
        Dict with 'success' key and operation results
    """
    # Extract mod_type from suggestion
    if hasattr(suggestion, 'mod_type'):
        mod_type = suggestion.mod_type
    elif isinstance(suggestion, dict):
        mod_type = suggestion.get('mod_type', suggestion.get('type', ''))
    else:
        mod_type = str(suggestion)

    # Map suggestion types to fix operation keys
    SUGGESTION_TO_VIOLATION_MAP = {
        'fix_silkscreen': 'silk_over_copper',
        'adjust_silkscreen': 'silk_over_copper',
        'fix_solder_mask': 'solder_mask_bridge',
        'adjust_clearance': 'clearance',
        'increase_clearance': 'clearance',
        'adjust_trace_width': 'track_width',
        'widen_traces': 'track_width',
        'fix_zone_nets': 'zone_net_assignment_error',
        'fill_zones': 'zone_fill_missing',
        'remove_dangling_vias': 'via_dangling',
        'remove_dangling_tracks': 'track_dangling',
        'fix_dangling': 'track_dangling',
        'assign_nets': 'unconnected_items',
        'fix_unconnected': 'unconnected_items',
        'update_footprint': 'lib_footprint_issues',
        'fix_footprint': 'lib_footprint_issues',
    }

    # Normalize and map
    normalized_mod = normalize_violation_type(mod_type)
    violation_key = SUGGESTION_TO_VIOLATION_MAP.get(normalized_mod, normalized_mod)

    fixes = get_fixes_for_violation(violation_key)

    if not fixes:
        return {
            'success': False,
            'error': f'No fix operation mapped for suggestion type: {mod_type}',
            'normalized': normalized_mod,
            'violation_key': violation_key
        }

    # Try first fix (highest success rate)
    fix = fixes[0]
    result = fix.execute(Path(pcb_path), kicad_python)
    result['fix_name'] = fix.name
    result['fix_script'] = fix.script

    return result


def apply_fixes_for_violations(
    pcb_path: Path,
    violations: List[Dict],
    dry_run: bool = False,
    kicad_python: Optional[str] = None
) -> Dict[str, Any]:
    """
    Apply appropriate fixes for each violation type found in DRC results.

    Args:
        pcb_path: Path to PCB file
        violations: List of violation dicts with 'type' field
        dry_run: If True, just report what would be done
        kicad_python: Python executable with pcbnew

    Returns:
        Summary of fixes attempted and results
    """
    if kicad_python is None:
        kicad_python = find_kicad_python()

    results = {
        'violations_processed': 0,
        'fixes_attempted': 0,
        'fixes_successful': 0,
        'fixes_by_type': {},
        'errors': [],
        'kicad_python': kicad_python,
        'timestamp': datetime.now().isoformat(),
    }

    # Group violations by type
    by_type: Dict[str, List[Dict]] = {}
    for v in violations:
        vtype = v.get('type', 'unknown')
        if vtype not in by_type:
            by_type[vtype] = []
        by_type[vtype].append(v)

    # Process each violation type
    for vtype, vlist in by_type.items():
        results['violations_processed'] += len(vlist)

        fixes = get_fixes_for_violation(vtype)

        if not fixes:
            results['errors'].append(f"No fix available for violation type: {vtype}")
            results['fixes_by_type'][vtype] = {
                'count': len(vlist),
                'fixes_tried': [],
                'success': False,
                'reason': 'no_fix_mapped'
            }
            continue

        results['fixes_by_type'][vtype] = {
            'count': len(vlist),
            'fixes_tried': [],
            'success': False,
        }

        # Try fixes in order until one succeeds
        for fix in fixes:
            results['fixes_attempted'] += 1

            if dry_run:
                results['fixes_by_type'][vtype]['fixes_tried'].append({
                    'operation': fix.name,
                    'script': fix.script,
                    'dry_run': True,
                    'would_run': fix._build_command(pcb_path, kicad_python or 'python3'),
                })
                continue

            try:
                fix_result = fix.execute(pcb_path, kicad_python)

                results['fixes_by_type'][vtype]['fixes_tried'].append({
                    'operation': fix.name,
                    'script': fix.script,
                    'result': fix_result,
                })

                if fix_result.get('success'):
                    results['fixes_successful'] += 1
                    results['fixes_by_type'][vtype]['success'] = True
                    break  # Stop trying more fixes for this type

            except Exception as e:
                results['fixes_by_type'][vtype]['fixes_tried'].append({
                    'operation': fix.name,
                    'script': fix.script,
                    'error': str(e),
                })

    return results


def get_available_fixers() -> Dict[str, bool]:
    """
    Check which fixer scripts are available.

    Returns:
        Dict mapping script name to availability status
    """
    script_dir = Path(__file__).parent

    scripts = [
        'kicad_pcb_fixer.py',
        'kicad_zone_filler.py',
        'kicad_silk_fixer.py',
        'kicad_mask_fixer.py',
        'kicad_trace_adjuster.py',
        'kicad_net_assigner.py',
        'kicad_dangling_track_fixer.py',
        'kicad_footprint_updater.py',
        'footprint_dfm_fixer.py',
    ]

    return {
        script: (script_dir / script).exists()
        for script in scripts
    }


def get_violation_coverage() -> Dict[str, List[str]]:
    """
    Get coverage report of which violation types have fix operations.

    Returns:
        Dict with 'covered' and 'uncovered' lists
    """
    # Known violation types from KiCad DRC
    all_violation_types = [
        'clearance',
        'track_width',
        'track_dangling',
        'via_dangling',
        'zone_net_assignment_error',
        'zone_fill_missing',
        'zone_clearance',
        'silk_over_copper',
        'silk_overlap',
        'solder_mask_bridge',
        'mask_aperture',
        'unconnected_items',
        'lib_footprint_issues',
        'lib_footprint_mismatch',
        'courtyard_overlap',
        'drill_size',
        'annular_ring',
        'shorting_items',
        'net_conflict',
    ]

    covered = []
    uncovered = []

    for vtype in all_violation_types:
        fixes = get_fixes_for_violation(vtype)
        if fixes:
            covered.append(vtype)
        else:
            uncovered.append(vtype)

    return {
        'covered': covered,
        'uncovered': uncovered,
        'coverage_pct': len(covered) / len(all_violation_types) * 100 if all_violation_types else 0
    }


# =============================================================================
# CLI Interface
# =============================================================================

def main():
    """CLI interface for violation fix mapping."""
    import argparse

    parser = argparse.ArgumentParser(
        description='Violation Fix Mapping - Map DRC violations to fix operations'
    )

    subparsers = parser.add_subparsers(dest='command', help='Commands')

    # Coverage command
    coverage_parser = subparsers.add_parser('coverage', help='Show violation coverage')

    # Fixers command
    fixers_parser = subparsers.add_parser('fixers', help='Show available fixer scripts')

    # Lookup command
    lookup_parser = subparsers.add_parser('lookup', help='Lookup fixes for a violation type')
    lookup_parser.add_argument('violation_type', help='Violation type to look up')

    # Apply command
    apply_parser = subparsers.add_parser('apply', help='Apply fixes to a PCB file')
    apply_parser.add_argument('pcb_path', help='Path to .kicad_pcb file')
    apply_parser.add_argument('--dry-run', action='store_true', help='Show what would be done')
    apply_parser.add_argument('--json', action='store_true', help='JSON output')

    args = parser.parse_args()

    if args.command == 'coverage':
        coverage = get_violation_coverage()
        print(f"\nViolation Fix Coverage: {coverage['coverage_pct']:.1f}%")
        print(f"\nCovered ({len(coverage['covered'])}):")
        for v in coverage['covered']:
            fixes = get_fixes_for_violation(v)
            print(f"  - {v}: {[f.name for f in fixes]}")
        print(f"\nUncovered ({len(coverage['uncovered'])}):")
        for v in coverage['uncovered']:
            print(f"  - {v}")

    elif args.command == 'fixers':
        fixers = get_available_fixers()
        print("\nAvailable Fixer Scripts:")
        for script, available in fixers.items():
            status = "OK" if available else "MISSING"
            print(f"  [{status}] {script}")

    elif args.command == 'lookup':
        fixes = get_fixes_for_violation(args.violation_type)
        if fixes:
            print(f"\nFixes for '{args.violation_type}':")
            for fix in fixes:
                print(f"  - {fix.name}")
                print(f"    Script: {fix.script}")
                print(f"    Operation: {fix.operation or '(default)'}")
                print(f"    Success rate: {fix.success_rate*100:.0f}%")
        else:
            print(f"\nNo fixes mapped for '{args.violation_type}'")

    elif args.command == 'apply':
        pcb_path = Path(args.pcb_path)
        if not pcb_path.exists():
            print(f"Error: PCB file not found: {pcb_path}")
            sys.exit(1)

        # Run DRC first to get violations
        # This requires kicad-cli
        print("Note: Run DRC first to get violations, then pass them to apply_fixes_for_violations()")
        print("Example:")
        print("  from violation_fix_map import apply_fixes_for_violations")
        print("  result = apply_fixes_for_violations(pcb_path, drc_violations)")

    else:
        parser.print_help()


if __name__ == '__main__':
    main()
