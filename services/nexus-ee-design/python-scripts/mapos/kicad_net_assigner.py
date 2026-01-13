#!/usr/bin/env python3
"""
KiCad Net Assigner - Assigns nets to orphan pads using pcbnew API.

This script should be run with a Python that has pcbnew available:
- macOS: KiCad's bundled Python
- Linux/Docker: System Python with kicad-python3 package

Part of MAPOS (Multi-Agent PCB Optimization System) for the Nexus EE Design Partner plugin.
"""

import sys
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional

# Cross-platform pcbnew import
try:
    from kicad_paths import KICAD_SITE_PACKAGES
    if KICAD_SITE_PACKAGES not in sys.path:
        sys.path.insert(0, KICAD_SITE_PACKAGES)
except ImportError:
    # Fallback: try macOS path if on macOS
    if sys.platform == 'darwin':
        mac_site = '/Applications/KiCad/KiCad.app/Contents/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages'
        if mac_site not in sys.path:
            sys.path.insert(0, mac_site)

try:
    import pcbnew
except ImportError as e:
    print(f"ERROR: Cannot import pcbnew: {e}", file=sys.stderr)
    print("Ensure pcbnew is available in your Python environment.", file=sys.stderr)
    sys.exit(1)


# Net assignment rules based on component type and pad number
NET_ASSIGNMENT_RULES = {
    # MOSFETs: High-side and low-side configurations
    # For N-channel MOSFETs: Gate (G), Drain (D), Source (S)
    # Typical pinout: 1=Gate, 2=Drain, 3=Source (or Tab)
    'MOS': {
        '3': 'GND',  # Source/Tab typically to GND for low-side
        '4': 'GND',  # Tab connection to GND
    },

    # Decoupling capacitors: Usually between power and GND
    'C': {
        '1': None,  # Leave unassigned - could be any power rail
        '2': 'GND',  # Typically one side goes to GND
    },

    # Test points
    'TP': {
        '1': None,  # Test points should remain unassigned
    },

    # Connectors: Usually power input
    'J': {
        '1': None,  # Connector pins need schematic for proper assignment
        '2': None,
    },
}


def find_orphan_pads(board) -> List[Dict]:
    """
    Find all pads that have no net assigned (net code 0).

    Returns:
        List of dicts with pad information
    """
    orphans = []

    for footprint in board.GetFootprints():
        ref = footprint.GetReference()
        fp_name = footprint.GetFPIDAsString()

        for pad in footprint.Pads():
            net = pad.GetNet()
            if net.GetNetCode() == 0:
                pos = pad.GetPosition()
                orphans.append({
                    'reference': ref,
                    'pad_number': pad.GetNumber(),
                    'footprint': fp_name,
                    'position': (pcbnew.ToMM(pos.x), pcbnew.ToMM(pos.y)),
                    'pad_type': str(pad.GetAttribute()),
                    'layers': str(pad.GetLayerSet().Seq()),
                })

    return orphans


def assign_net_to_pad(board, footprint, pad, net_name: str) -> bool:
    """
    Assign a net to a pad.

    Args:
        board: The pcbnew board object
        footprint: The footprint containing the pad
        pad: The pad to assign
        net_name: Name of the net to assign

    Returns:
        True if successful, False otherwise
    """
    net = board.FindNet(net_name)
    if net is None:
        print(f"  Warning: Net '{net_name}' not found in board", file=sys.stderr)
        return False

    pad.SetNet(net)
    return True


def get_component_prefix(reference: str) -> str:
    """Extract component prefix from reference (e.g., 'MOS' from 'MOS1')."""
    prefix = ''
    for char in reference:
        if char.isalpha():
            prefix += char
        else:
            break
    return prefix


def assign_orphan_pad_nets(pcb_path: str, save: bool = True, dry_run: bool = False) -> dict:
    """
    Assign nets to orphan pads based on component type heuristics.

    Args:
        pcb_path: Path to the .kicad_pcb file
        save: Whether to save the board after assignment
        dry_run: If True, only report what would be done

    Returns:
        Dict with assignment statistics
    """
    pcb_path = Path(pcb_path).resolve()
    if not pcb_path.exists():
        raise FileNotFoundError(f"PCB file not found: {pcb_path}")

    print(f"Loading board: {pcb_path}")
    board = pcbnew.LoadBoard(str(pcb_path))

    # Find all orphan pads
    orphans = find_orphan_pads(board)
    print(f"Found {len(orphans)} orphan pads")

    # Group by component prefix
    by_prefix = {}
    for orphan in orphans:
        prefix = get_component_prefix(orphan['reference'])
        if prefix not in by_prefix:
            by_prefix[prefix] = []
        by_prefix[prefix].append(orphan)

    print("\nOrphan pads by component type:")
    for prefix, items in sorted(by_prefix.items()):
        print(f"  {prefix}: {len(items)} pads")

    # Assign nets based on rules
    assignments = []
    skipped = []

    for footprint in board.GetFootprints():
        ref = footprint.GetReference()
        prefix = get_component_prefix(ref)

        rules = NET_ASSIGNMENT_RULES.get(prefix, {})

        for pad in footprint.Pads():
            if pad.GetNet().GetNetCode() != 0:
                continue  # Skip pads that already have nets

            pad_num = pad.GetNumber()
            assigned_net = rules.get(pad_num)

            if assigned_net:
                if dry_run:
                    print(f"  Would assign {ref}.{pad_num} -> {assigned_net}")
                    assignments.append({
                        'reference': ref,
                        'pad': pad_num,
                        'net': assigned_net,
                        'status': 'would_assign'
                    })
                else:
                    success = assign_net_to_pad(board, footprint, pad, assigned_net)
                    status = 'assigned' if success else 'failed'
                    print(f"  {ref}.{pad_num} -> {assigned_net} ({status})")
                    assignments.append({
                        'reference': ref,
                        'pad': pad_num,
                        'net': assigned_net,
                        'status': status
                    })
            else:
                skipped.append({
                    'reference': ref,
                    'pad': pad_num,
                    'reason': 'no_rule'
                })

    # Save the board
    if save and not dry_run and assignments:
        print(f"\nSaving board to: {pcb_path}")
        pcbnew.SaveBoard(str(pcb_path), board)
        print("Board saved")

    result = {
        'orphan_pads_found': len(orphans),
        'assignments_made': len([a for a in assignments if a['status'] == 'assigned']),
        'assignments_failed': len([a for a in assignments if a['status'] == 'failed']),
        'skipped': len(skipped),
        'assignments': assignments,
        'success': True,
        'dry_run': dry_run
    }

    return result


def report_orphan_pads(pcb_path: str) -> dict:
    """
    Report all orphan pads without making changes.

    Args:
        pcb_path: Path to the .kicad_pcb file

    Returns:
        Dict with orphan pad information
    """
    board = pcbnew.LoadBoard(str(pcb_path))
    orphans = find_orphan_pads(board)

    # Group by component prefix
    by_prefix = {}
    for orphan in orphans:
        prefix = get_component_prefix(orphan['reference'])
        if prefix not in by_prefix:
            by_prefix[prefix] = []
        by_prefix[prefix].append(orphan)

    return {
        'total_orphan_pads': len(orphans),
        'by_component_type': {k: len(v) for k, v in by_prefix.items()},
        'orphan_pads': orphans
    }


def main():
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description='Assign nets to orphan pads in KiCad PCB files',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
This script must be run with KiCad's bundled Python interpreter.

Examples:
  %(prog)s board.kicad_pcb              # Assign nets and save
  %(prog)s board.kicad_pcb --dry-run    # Show what would be done
  %(prog)s board.kicad_pcb --report     # Report orphan pads only

Note: Run with:
  /Applications/KiCad/KiCad.app/Contents/Frameworks/Python.framework/Versions/3.9/bin/python3
        '''
    )
    parser.add_argument('pcb_path', help='Path to KiCad PCB file')
    parser.add_argument('--no-save', action='store_true',
                       help='Do not save the board after assignment')
    parser.add_argument('--dry-run', action='store_true',
                       help='Show what would be done without making changes')
    parser.add_argument('--report', action='store_true',
                       help='Report orphan pads without making changes')
    parser.add_argument('--json', action='store_true',
                       help='Output results as JSON')

    args = parser.parse_args()

    try:
        if args.report:
            result = report_orphan_pads(args.pcb_path)
        else:
            result = assign_orphan_pad_nets(
                args.pcb_path,
                save=not args.no_save,
                dry_run=args.dry_run
            )

        if args.json:
            print(json.dumps(result, indent=2))
        else:
            if not args.report:
                print(f"\nSummary:")
                print(f"  Orphan pads found: {result['orphan_pads_found']}")
                print(f"  Assignments made: {result['assignments_made']}")
                print(f"  Skipped: {result['skipped']}")

        return 0 if result.get('success', True) else 1

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
