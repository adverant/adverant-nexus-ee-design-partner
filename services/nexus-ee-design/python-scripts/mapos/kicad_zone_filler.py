#!/usr/bin/env python3
"""
KiCad Zone Filler - Uses native pcbnew API to fill zones.

This script should be run with a Python that has pcbnew available:
- macOS: KiCad's bundled Python
- Linux/Docker: System Python with kicad-python3 package

Part of MAPOS (Multi-Agent PCB Optimization System) for the Nexus EE Design Partner plugin.
"""

import sys
import os
import json
from pathlib import Path
from datetime import datetime

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


def fill_all_zones(pcb_path: str, save: bool = True) -> dict:
    """
    Fill all zones in a KiCad PCB file using the ZONE_FILLER API.

    Args:
        pcb_path: Path to the .kicad_pcb file
        save: Whether to save the board after filling

    Returns:
        Dict with zone fill statistics
    """
    pcb_path = Path(pcb_path).resolve()
    if not pcb_path.exists():
        raise FileNotFoundError(f"PCB file not found: {pcb_path}")

    print(f"Loading board: {pcb_path}")
    board = pcbnew.LoadBoard(str(pcb_path))

    # Get all zones
    zones = board.Zones()
    zone_count = len(zones)

    if zone_count == 0:
        print("No zones found in board")
        return {'zones_filled': 0, 'zones_total': 0, 'success': True}

    print(f"Found {zone_count} zones")

    # Collect zone info before fill
    zone_info = []
    for zone in zones:
        info = {
            'name': zone.GetZoneName(),
            'net': zone.GetNetname(),
            'layer': board.GetLayerName(zone.GetFirstLayer()),
            'priority': zone.GetAssignedPriority(),
        }
        zone_info.append(info)
        print(f"  Zone: {info['name']} on {info['layer']} (net: {info['net']})")

    # Create zone filler
    print("\nFilling zones...")
    filler = pcbnew.ZONE_FILLER(board)

    # Fill all zones
    try:
        filler.Fill(zones)
        print("Zone fill completed successfully")
    except Exception as e:
        print(f"Zone fill error: {e}", file=sys.stderr)
        return {'zones_filled': 0, 'zones_total': zone_count, 'success': False, 'error': str(e)}

    # Save the board
    if save:
        print(f"\nSaving board to: {pcb_path}")
        pcbnew.SaveBoard(str(pcb_path), board)
        print("Board saved")

    return {
        'zones_filled': zone_count,
        'zones_total': zone_count,
        'zones': zone_info,
        'success': True,
        'pcb_path': str(pcb_path)
    }


def get_zone_statistics(pcb_path: str) -> dict:
    """
    Get statistics about zones in a PCB file.

    Args:
        pcb_path: Path to the .kicad_pcb file

    Returns:
        Dict with zone statistics
    """
    board = pcbnew.LoadBoard(str(pcb_path))
    zones = board.Zones()

    stats = {
        'total_zones': len(zones),
        'zones_by_layer': {},
        'zones_by_net': {},
        'filled_areas': []
    }

    for zone in zones:
        layer = board.GetLayerName(zone.GetFirstLayer())
        net = zone.GetNetname()

        # Count by layer
        stats['zones_by_layer'][layer] = stats['zones_by_layer'].get(layer, 0) + 1

        # Count by net
        stats['zones_by_net'][net] = stats['zones_by_net'].get(net, 0) + 1

        # Get filled area info
        filled_polys = zone.GetFilledPolysList(zone.GetFirstLayer())
        area_count = filled_polys.OutlineCount() if filled_polys else 0

        stats['filled_areas'].append({
            'name': zone.GetZoneName(),
            'layer': layer,
            'net': net,
            'outline_count': area_count
        })

    return stats


def main():
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description='Fill zones in KiCad PCB files using pcbnew API',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
This script must be run with KiCad's bundled Python interpreter.

Examples:
  %(prog)s board.kicad_pcb
  %(prog)s board.kicad_pcb --no-save
  %(prog)s board.kicad_pcb --stats

Note: Run with:
  /Applications/KiCad/KiCad.app/Contents/Frameworks/Python.framework/Versions/3.9/bin/python3
        '''
    )
    parser.add_argument('pcb_path', help='Path to KiCad PCB file')
    parser.add_argument('--no-save', action='store_true',
                       help='Do not save the board after filling')
    parser.add_argument('--stats', action='store_true',
                       help='Print zone statistics and exit')
    parser.add_argument('--json', action='store_true',
                       help='Output results as JSON')

    args = parser.parse_args()

    try:
        if args.stats:
            result = get_zone_statistics(args.pcb_path)
        else:
            result = fill_all_zones(args.pcb_path, save=not args.no_save)

        if args.json:
            print(json.dumps(result, indent=2))
        else:
            if not args.stats:
                print(f"\nResult: {result['zones_filled']}/{result['zones_total']} zones filled")
                print(f"Success: {result['success']}")

        return 0 if result.get('success', True) else 1

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


if __name__ == '__main__':
    sys.exit(main())
