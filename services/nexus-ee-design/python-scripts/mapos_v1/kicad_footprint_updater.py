#!/usr/bin/env python3
"""
KiCad Footprint Updater

Handles lib_footprint_issues and lib_footprint_mismatch DRC violations by:
1. Identifying footprints that differ from library versions
2. Analyzing the differences (pads, courtyard, 3D model, etc.)
3. Providing options to update from library or accept local changes

Uses pcbnew API directly.

Part of MAPOS Gaming AI Remediation - Phase 2

IMPORTANT NOTE:
Full footprint synchronization requires schematic-to-PCB synchronization which
is a complex operation best done through KiCad's "Update PCB from Schematic".
This script provides analysis and limited fixes for common issues.
"""

import pcbnew
from pathlib import Path
import json
import argparse
import shutil
from datetime import datetime
from typing import List, Dict, Optional, Tuple, Set
from dataclasses import dataclass, field


@dataclass
class FootprintDifference:
    """Represents a difference between board and library footprint."""
    reference: str
    library_id: str
    difference_type: str  # 'pad_count', 'pad_position', 'courtyard', '3d_model', 'attributes', 'unknown'
    description: str
    severity: str  # 'info', 'warning', 'error'
    auto_fixable: bool = False
    fix_action: Optional[str] = None


@dataclass
class FootprintInfo:
    """Information about a footprint on the board."""
    reference: str
    value: str
    library_nickname: str
    library_item_name: str
    full_library_id: str
    position: Tuple[float, float]  # mm
    rotation: float  # degrees
    layer: str
    pad_count: int
    has_courtyard: bool
    has_3d_model: bool
    attributes: int
    is_dnp: bool


class FootprintUpdater:
    """Analyze and update footprints from library."""

    def __init__(self, pcb_path: str):
        """
        Initialize the updater.

        Args:
            pcb_path: Path to .kicad_pcb file
        """
        self.pcb_path = Path(pcb_path)
        if not self.pcb_path.exists():
            raise FileNotFoundError(f"PCB file not found: {pcb_path}")

        self.board = pcbnew.LoadBoard(str(self.pcb_path))
        self._footprint_table: Optional[pcbnew.FP_LIB_TABLE] = None

    def _get_footprint_table(self) -> Optional[pcbnew.FP_LIB_TABLE]:
        """Get the footprint library table."""
        if self._footprint_table is not None:
            return self._footprint_table

        try:
            self._footprint_table = pcbnew.GetGlobalFootprintTable()
            return self._footprint_table
        except Exception:
            # Library table may not be available in headless mode
            return None

    def get_footprint_info(self, footprint) -> FootprintInfo:
        """
        Extract information about a footprint.

        Args:
            footprint: pcbnew footprint object

        Returns:
            FootprintInfo dataclass
        """
        lib_id = footprint.GetFPID()
        pos = footprint.GetPosition()

        # Count pads
        pad_count = len(list(footprint.Pads()))

        # Check for courtyard
        has_courtyard = False
        for item in footprint.GraphicalItems():
            layer = item.GetLayer()
            layer_name = pcbnew.LayerName(layer).lower()
            if 'courtyard' in layer_name:
                has_courtyard = True
                break

        # Check for 3D model
        has_3d_model = len(footprint.Models()) > 0

        return FootprintInfo(
            reference=footprint.GetReference(),
            value=footprint.GetValue(),
            library_nickname=str(lib_id.GetLibNickname()),
            library_item_name=str(lib_id.GetLibItemName()),
            full_library_id=str(lib_id.GetUniStringLibId()),
            position=(pos.x / 1_000_000, pos.y / 1_000_000),
            rotation=footprint.GetOrientationDegrees(),
            layer=pcbnew.LayerName(footprint.GetLayer()),
            pad_count=pad_count,
            has_courtyard=has_courtyard,
            has_3d_model=has_3d_model,
            attributes=footprint.GetAttributes(),
            is_dnp=footprint.IsDNP() if hasattr(footprint, 'IsDNP') else False,
        )

    def find_library_issues(self) -> List[FootprintDifference]:
        """
        Find footprints with potential library mismatch issues.

        This performs heuristic analysis since full library comparison
        requires access to the library files.

        Returns:
            List of FootprintDifference objects
        """
        issues = []

        for footprint in self.board.GetFootprints():
            info = self.get_footprint_info(footprint)

            # Check for missing library reference
            if not info.library_nickname or info.library_nickname == '':
                issues.append(FootprintDifference(
                    reference=info.reference,
                    library_id=info.full_library_id,
                    difference_type='unknown',
                    description='Footprint has no library reference',
                    severity='warning',
                    auto_fixable=False,
                ))
                continue

            # Check for missing courtyard (common DRC issue)
            if not info.has_courtyard:
                issues.append(FootprintDifference(
                    reference=info.reference,
                    library_id=info.full_library_id,
                    difference_type='courtyard',
                    description='Footprint missing courtyard layer',
                    severity='info',
                    auto_fixable=True,
                    fix_action='add_courtyard',
                ))

            # Check for unusual attributes
            attrs = info.attributes
            # Common KiCad attributes:
            # FP_SMD = 0x01, FP_THROUGH_HOLE = 0x02, FP_EXCLUDE_FROM_POS_FILES = 0x04
            # FP_EXCLUDE_FROM_BOM = 0x08, FP_DNP = 0x10

            # Detect potentially inconsistent attributes
            smd_pads = 0
            th_pads = 0
            for pad in footprint.Pads():
                if pad.GetAttribute() == pcbnew.PAD_ATTRIB_SMD:
                    smd_pads += 1
                elif pad.GetAttribute() == pcbnew.PAD_ATTRIB_PTH:
                    th_pads += 1

            # SMD footprint with through-hole pads or vice versa
            is_smd_attr = (attrs & 0x01) != 0
            is_th_attr = (attrs & 0x02) != 0

            if smd_pads > 0 and th_pads > 0:
                # Mixed technology - this is valid but unusual
                pass
            elif smd_pads > 0 and is_th_attr and not is_smd_attr:
                issues.append(FootprintDifference(
                    reference=info.reference,
                    library_id=info.full_library_id,
                    difference_type='attributes',
                    description='Footprint has SMD pads but Through-Hole attribute',
                    severity='warning',
                    auto_fixable=True,
                    fix_action='fix_attributes',
                ))
            elif th_pads > 0 and is_smd_attr and not is_th_attr:
                issues.append(FootprintDifference(
                    reference=info.reference,
                    library_id=info.full_library_id,
                    difference_type='attributes',
                    description='Footprint has Through-Hole pads but SMD attribute',
                    severity='warning',
                    auto_fixable=True,
                    fix_action='fix_attributes',
                ))

            # Check for DNP without the flag
            if 'DNP' in info.value.upper() and not info.is_dnp:
                issues.append(FootprintDifference(
                    reference=info.reference,
                    library_id=info.full_library_id,
                    difference_type='attributes',
                    description='Value contains "DNP" but DNP attribute not set',
                    severity='info',
                    auto_fixable=True,
                    fix_action='set_dnp',
                ))

        return issues

    def analyze_footprints(self) -> Dict:
        """
        Analyze all footprints in the board.

        Returns:
            Dict with analysis results
        """
        footprints = []
        libraries_used: Set[str] = set()

        for fp in self.board.GetFootprints():
            info = self.get_footprint_info(fp)
            footprints.append({
                'reference': info.reference,
                'value': info.value,
                'library': info.library_nickname,
                'footprint': info.library_item_name,
                'position_mm': info.position,
                'rotation': info.rotation,
                'layer': info.layer,
                'pad_count': info.pad_count,
                'has_courtyard': info.has_courtyard,
                'has_3d_model': info.has_3d_model,
            })
            if info.library_nickname:
                libraries_used.add(info.library_nickname)

        issues = self.find_library_issues()

        return {
            'success': True,
            'operation': 'analyze',
            'total_footprints': len(footprints),
            'libraries_used': sorted(list(libraries_used)),
            'footprints': footprints,
            'issues_found': len(issues),
            'issues': [
                {
                    'reference': i.reference,
                    'library_id': i.library_id,
                    'type': i.difference_type,
                    'description': i.description,
                    'severity': i.severity,
                    'auto_fixable': i.auto_fixable,
                }
                for i in issues
            ],
            'summary': {
                'info': sum(1 for i in issues if i.severity == 'info'),
                'warning': sum(1 for i in issues if i.severity == 'warning'),
                'error': sum(1 for i in issues if i.severity == 'error'),
                'auto_fixable': sum(1 for i in issues if i.auto_fixable),
            },
        }

    def add_courtyard(self, footprint, margin_mm: float = 0.25) -> bool:
        """
        Add a courtyard rectangle to a footprint based on its bounding box.

        Args:
            footprint: pcbnew footprint object
            margin_mm: Margin around pads in mm

        Returns:
            True if courtyard was added
        """
        # Get bounding box of pads
        min_x = float('inf')
        min_y = float('inf')
        max_x = float('-inf')
        max_y = float('-inf')

        for pad in footprint.Pads():
            pos = pad.GetPosition()
            size = pad.GetSize()

            pad_min_x = pos.x - size.x // 2
            pad_min_y = pos.y - size.y // 2
            pad_max_x = pos.x + size.x // 2
            pad_max_y = pos.y + size.y // 2

            min_x = min(min_x, pad_min_x)
            min_y = min(min_y, pad_min_y)
            max_x = max(max_x, pad_max_x)
            max_y = max(max_y, pad_max_y)

        if min_x == float('inf'):
            return False  # No pads

        # Add margin
        margin_nm = int(margin_mm * 1_000_000)
        min_x -= margin_nm
        min_y -= margin_nm
        max_x += margin_nm
        max_y += margin_nm

        # Determine courtyard layer based on footprint layer
        fp_layer = footprint.GetLayer()
        if fp_layer == pcbnew.F_Cu:
            courtyard_layer = pcbnew.F_CrtYd
        else:
            courtyard_layer = pcbnew.B_CrtYd

        # Create courtyard rectangle
        # Use PCB_SHAPE for rectangle
        rect = pcbnew.PCB_SHAPE(footprint)
        rect.SetShape(pcbnew.SHAPE_T_RECT)
        rect.SetStart(pcbnew.VECTOR2I(min_x, min_y))
        rect.SetEnd(pcbnew.VECTOR2I(max_x, max_y))
        rect.SetLayer(courtyard_layer)
        rect.SetWidth(int(0.05 * 1_000_000))  # 0.05mm line width

        footprint.Add(rect)
        return True

    def fix_attributes(self, footprint) -> bool:
        """
        Fix footprint attributes based on pad types.

        Args:
            footprint: pcbnew footprint object

        Returns:
            True if attributes were fixed
        """
        smd_pads = 0
        th_pads = 0

        for pad in footprint.Pads():
            attr = pad.GetAttribute()
            if attr == pcbnew.PAD_ATTRIB_SMD:
                smd_pads += 1
            elif attr == pcbnew.PAD_ATTRIB_PTH:
                th_pads += 1

        current_attrs = footprint.GetAttributes()

        # Clear SMD/TH flags
        new_attrs = current_attrs & ~0x03

        # Set appropriate flag
        if smd_pads > 0 and th_pads == 0:
            new_attrs |= 0x01  # FP_SMD
        elif th_pads > 0 and smd_pads == 0:
            new_attrs |= 0x02  # FP_THROUGH_HOLE
        elif smd_pads > 0 and th_pads > 0:
            # Mixed - set both (or could choose dominant)
            new_attrs |= 0x01 | 0x02

        if new_attrs != current_attrs:
            footprint.SetAttributes(new_attrs)
            return True

        return False

    def set_dnp_flag(self, footprint) -> bool:
        """
        Set the DNP (Do Not Populate) flag on a footprint.

        Args:
            footprint: pcbnew footprint object

        Returns:
            True if flag was set
        """
        if hasattr(footprint, 'SetDNP'):
            footprint.SetDNP(True)
            return True
        else:
            # Older KiCad - use attributes
            current_attrs = footprint.GetAttributes()
            new_attrs = current_attrs | 0x10  # FP_DNP
            if new_attrs != current_attrs:
                footprint.SetAttributes(new_attrs)
                return True
        return False

    def apply_auto_fixes(self, backup: bool = True, save: bool = True) -> Dict:
        """
        Apply all auto-fixable fixes.

        Args:
            backup: Create backup before modifying
            save: Save changes to file

        Returns:
            Dict with fix results
        """
        if backup:
            self._create_backup()

        issues = self.find_library_issues()
        auto_fixable = [i for i in issues if i.auto_fixable]

        fixes_applied = []
        fixes_failed = []

        # Group by footprint reference
        fp_map = {fp.GetReference(): fp for fp in self.board.GetFootprints()}

        for issue in auto_fixable:
            fp = fp_map.get(issue.reference)
            if not fp:
                fixes_failed.append({
                    'reference': issue.reference,
                    'action': issue.fix_action,
                    'error': 'Footprint not found',
                })
                continue

            try:
                success = False

                if issue.fix_action == 'add_courtyard':
                    success = self.add_courtyard(fp)
                elif issue.fix_action == 'fix_attributes':
                    success = self.fix_attributes(fp)
                elif issue.fix_action == 'set_dnp':
                    success = self.set_dnp_flag(fp)

                if success:
                    fixes_applied.append({
                        'reference': issue.reference,
                        'action': issue.fix_action,
                        'description': issue.description,
                    })
                else:
                    fixes_failed.append({
                        'reference': issue.reference,
                        'action': issue.fix_action,
                        'error': 'Fix returned False',
                    })

            except Exception as e:
                fixes_failed.append({
                    'reference': issue.reference,
                    'action': issue.fix_action,
                    'error': str(e),
                })

        if save and fixes_applied:
            pcbnew.SaveBoard(str(self.pcb_path), self.board)

        return {
            'success': True,
            'operation': 'apply_auto_fixes',
            'issues_found': len(issues),
            'auto_fixable': len(auto_fixable),
            'fixes_applied': len(fixes_applied),
            'fixes_failed': len(fixes_failed),
            'applied': fixes_applied,
            'failed': fixes_failed,
            'file_saved': save and len(fixes_applied) > 0,
            'backup_created': backup,
        }

    def update_from_library(self, references: Optional[List[str]] = None, backup: bool = True, save: bool = True) -> Dict:
        """
        Update footprints from library.

        NOTE: This is a LIMITED implementation. Full footprint update
        from library is a complex operation that:
        1. Requires access to library files
        2. Must preserve position, rotation, and pad assignments
        3. Should update nets properly

        For complete updates, use KiCad's "Update PCB from Schematic" feature.

        Args:
            references: List of references to update (None = all)
            backup: Create backup before modifying
            save: Save changes to file

        Returns:
            Dict with update results
        """
        if backup:
            self._create_backup()

        fp_table = self._get_footprint_table()

        updated = []
        failed = []
        flagged = []

        for footprint in self.board.GetFootprints():
            ref = footprint.GetReference()

            if references and ref not in references:
                continue

            lib_id = footprint.GetFPID()
            lib_nick = str(lib_id.GetLibNickname())
            fp_name = str(lib_id.GetLibItemName())

            try:
                if fp_table is None:
                    # No library access - flag for manual review
                    flagged.append({
                        'reference': ref,
                        'library': lib_nick,
                        'footprint': fp_name,
                        'status': 'flagged_for_review',
                        'reason': 'Library table not available in headless mode',
                        'recommendation': 'Use KiCad "Update PCB from Schematic" feature',
                    })
                    continue

                # Try to load footprint from library
                try:
                    lib_fp = fp_table.LoadEnumeratedFootprint(lib_id.GetLibNickname(), fp_name)

                    if lib_fp is None:
                        failed.append({
                            'reference': ref,
                            'error': f'Footprint {fp_name} not found in library {lib_nick}',
                        })
                        continue

                    # Compare and update if different
                    # This is where a full implementation would:
                    # 1. Compare all properties
                    # 2. Update graphics, pads, etc.
                    # 3. Preserve position and orientation

                    # For now, just flag for review
                    flagged.append({
                        'reference': ref,
                        'library': lib_nick,
                        'footprint': fp_name,
                        'status': 'library_found',
                        'recommendation': 'Manual review recommended - full sync requires schematic',
                    })

                except Exception as e:
                    failed.append({
                        'reference': ref,
                        'error': f'Failed to load from library: {e}',
                    })

            except Exception as e:
                failed.append({
                    'reference': ref,
                    'error': str(e),
                })

        return {
            'success': True,
            'operation': 'update_from_library',
            'footprints_checked': len(list(self.board.GetFootprints())),
            'updated': len(updated),
            'flagged_for_review': len(flagged),
            'failed': len(failed),
            'updated_list': updated,
            'flagged_list': flagged,
            'failed_list': failed,
            'note': 'Full footprint sync requires KiCad "Update PCB from Schematic" feature',
            'file_saved': save and len(updated) > 0,
            'backup_created': backup,
        }

    def _create_backup(self) -> Path:
        """Create a backup of the PCB file."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        backup_path = self.pcb_path.with_suffix(f'.backup_{timestamp}.kicad_pcb')
        shutil.copy2(self.pcb_path, backup_path)
        return backup_path


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description='Analyze and fix footprint library issues in KiCad PCB files',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze footprints (no changes)
  python kicad_footprint_updater.py board.kicad_pcb --analyze

  # Apply auto-fixable fixes (courtyards, attributes)
  python kicad_footprint_updater.py board.kicad_pcb --fix

  # Check against library (limited without GUI)
  python kicad_footprint_updater.py board.kicad_pcb --check-library

  # JSON output for scripting
  python kicad_footprint_updater.py board.kicad_pcb --analyze --json

NOTE: Full footprint library synchronization requires KiCad's
"Update PCB from Schematic" feature which is not available
in headless/scripting mode.
        """,
    )

    parser.add_argument('pcb_path', help='Path to .kicad_pcb file')

    action_group = parser.add_mutually_exclusive_group()
    action_group.add_argument(
        '--analyze', '-a',
        action='store_true',
        help='Analyze footprints and find issues (default)',
    )
    action_group.add_argument(
        '--fix', '-f',
        action='store_true',
        help='Apply auto-fixable fixes',
    )
    action_group.add_argument(
        '--check-library', '-c',
        action='store_true',
        help='Check footprints against library (limited)',
    )

    parser.add_argument(
        '--no-backup',
        action='store_true',
        help='Skip creating backup file',
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be done without making changes',
    )
    parser.add_argument(
        '--json',
        action='store_true',
        help='Output results as JSON',
    )
    parser.add_argument(
        '--references', '-r',
        nargs='+',
        help='Specific footprint references to process',
    )

    args = parser.parse_args()

    try:
        updater = FootprintUpdater(args.pcb_path)

        backup = not args.no_backup
        save = not args.dry_run

        if args.fix:
            result = updater.apply_auto_fixes(backup=backup, save=save)
        elif args.check_library:
            result = updater.update_from_library(
                references=args.references,
                backup=backup,
                save=save,
            )
        else:
            # Default: analyze
            result = updater.analyze_footprints()

        # Add dry run flag to result
        if args.dry_run:
            result['dry_run'] = True

        # Output
        if args.json:
            print(json.dumps(result, indent=2, default=str))
        else:
            # Human-readable output
            print(f"\n{'='*60}")
            print(f"Footprint Updater - {result['operation']}")
            print(f"{'='*60}")

            print(f"Total footprints: {result.get('total_footprints', result.get('footprints_checked', 0))}")

            if result['operation'] == 'analyze':
                print(f"Issues found: {result['issues_found']}")
                if result.get('summary'):
                    print(f"  - Info: {result['summary']['info']}")
                    print(f"  - Warning: {result['summary']['warning']}")
                    print(f"  - Error: {result['summary']['error']}")
                    print(f"  - Auto-fixable: {result['summary']['auto_fixable']}")

                if result.get('issues'):
                    print(f"\nIssues:")
                    for issue in result['issues'][:10]:  # Show first 10
                        severity_icon = {'info': 'ℹ️', 'warning': '⚠️', 'error': '❌'}.get(issue['severity'], '•')
                        print(f"  {severity_icon} {issue['reference']}: {issue['description']}")

            elif result['operation'] == 'apply_auto_fixes':
                print(f"Fixes applied: {result['fixes_applied']}")
                print(f"Fixes failed: {result['fixes_failed']}")

            elif result['operation'] == 'update_from_library':
                print(f"Updated: {result['updated']}")
                print(f"Flagged for review: {result['flagged_for_review']}")
                print(f"Failed: {result['failed']}")
                print(f"\nNote: {result.get('note', '')}")

            if result.get('dry_run'):
                print("\n[DRY RUN - No changes made]")
            elif result.get('file_saved'):
                print(f"\nChanges saved to: {args.pcb_path}")

            print(f"{'='*60}\n")

        return 0 if result['success'] else 1

    except FileNotFoundError as e:
        error = {'success': False, 'error': str(e)}
        if args.json:
            print(json.dumps(error, indent=2))
        else:
            print(f"Error: {e}", file=__import__('sys').stderr)
        return 1
    except Exception as e:
        error = {'success': False, 'error': str(e), 'type': type(e).__name__}
        if args.json:
            print(json.dumps(error, indent=2))
        else:
            print(f"Error ({type(e).__name__}): {e}", file=__import__('sys').stderr)
        return 1


if __name__ == '__main__':
    exit(main())
