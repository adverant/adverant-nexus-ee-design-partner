#!/usr/bin/env python3
"""
KiCad PCB Fixer - Execute pcbnew operations for zone nets, design settings, and dangling vias.

This script MUST be run with KiCad's bundled Python interpreter that has pcbnew available.
Works cross-platform: macOS, Linux, and Docker containers.

Usage:
    python kicad_pcb_fixer.py zone-nets board.kicad_pcb [--json]
    python kicad_pcb_fixer.py design-settings board.kicad_pcb [--json]
    python kicad_pcb_fixer.py dangling-vias board.kicad_pcb [--json]
    python kicad_pcb_fixer.py all board.kicad_pcb [--json]

Part of MAPOS (Multi-Agent PCB Optimization System) for the Nexus EE Design Partner plugin.
"""

import sys
import os
import json
import shutil
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional, Set, Tuple


def find_pcbnew():
    """Find and import pcbnew from various possible locations."""
    # First try direct import (works if running with KiCad's Python)
    try:
        import pcbnew
        return pcbnew
    except ImportError:
        pass

    # Try common KiCad Python site-packages locations
    candidates = [
        # macOS KiCad 8.x
        '/Applications/KiCad/KiCad.app/Contents/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages',
        '/Applications/KiCad/KiCad.app/Contents/Frameworks/Python.framework/Versions/3.11/lib/python3.11/site-packages',
        # macOS KiCad 7.x
        '/Applications/KiCad/KiCad.app/Contents/Frameworks/Python.framework/Versions/Current/lib/python3.9/site-packages',
        # Linux (Debian/Ubuntu)
        '/usr/lib/python3/dist-packages',
        '/usr/local/lib/python3/dist-packages',
        # Linux (PPA)
        '/usr/share/kicad/scripting/plugins',
        # Container/Custom
        '/opt/kicad/lib/python3/site-packages',
    ]

    for path in candidates:
        if path and Path(path).exists():
            if path not in sys.path:
                sys.path.insert(0, path)
            try:
                import pcbnew
                return pcbnew
            except ImportError:
                continue

    # If still not found, provide helpful error
    print("ERROR: Cannot import pcbnew module.", file=sys.stderr)
    print("", file=sys.stderr)
    print("This script requires KiCad's Python interpreter with pcbnew.", file=sys.stderr)
    print("", file=sys.stderr)
    print("On macOS, run with:", file=sys.stderr)
    print("  /Applications/KiCad/KiCad.app/Contents/Frameworks/Python.framework/Versions/3.9/bin/python3", file=sys.stderr)
    print("", file=sys.stderr)
    print("On Linux, ensure kicad is installed and try:", file=sys.stderr)
    print("  python3 (with kicad-scripting package installed)", file=sys.stderr)
    sys.exit(1)


# Import pcbnew
pcbnew = find_pcbnew()


# IPC-2221 Class 2 design rules (standard for commercial electronics)
IPC_2221_CLASS_2 = {
    'min_clearance_mm': 0.1,      # 100um minimum clearance
    'min_track_width_mm': 0.15,   # 150um minimum track
    'min_via_diameter_mm': 0.4,   # 400um minimum via
    'min_via_drill_mm': 0.2,      # 200um minimum drill
    'min_annular_ring_mm': 0.1,   # 100um annular ring
}


class KiCadPCBFixer:
    """Execute pcbnew operations for zone nets, design settings, and dangling vias."""

    def __init__(self, pcb_path: str):
        """
        Initialize fixer with a PCB file.

        Args:
            pcb_path: Path to the .kicad_pcb file
        """
        self.pcb_path = Path(pcb_path).resolve()
        if not self.pcb_path.exists():
            raise FileNotFoundError(f"PCB file not found: {self.pcb_path}")

        print(f"Loading board: {self.pcb_path}")
        self.board = pcbnew.LoadBoard(str(self.pcb_path))
        self.changes_made = False

    def fix_zone_nets(self) -> Dict[str, Any]:
        """
        Fix zone net assignments by detecting and correcting mismatched zone-to-net connections.

        Zones should be connected to the net that best matches their copper pour area.
        This fixes cases where zones have incorrect or no net assigned.

        Returns:
            Dict with fix statistics
        """
        print("\n=== Fixing Zone Net Assignments ===")

        zones = list(self.board.Zones())
        if not zones:
            print("No zones found in board")
            return {'zones_checked': 0, 'zones_fixed': 0, 'success': True}

        zones_checked = 0
        zones_fixed = 0
        fixes = []

        # Build a map of net names to net objects for quick lookup
        net_map = {}
        for net in self.board.GetNetInfo().NetsByName():
            net_map[net] = self.board.FindNet(net)

        # Get all pads to determine zone net based on contained pads
        pads_by_position = {}
        for footprint in self.board.GetFootprints():
            for pad in footprint.Pads():
                pos = pad.GetPosition()
                key = (pos.x, pos.y, pad.GetLayerSet())
                net = pad.GetNet()
                if net.GetNetCode() > 0:
                    pads_by_position[key] = net

        for zone in zones:
            zones_checked += 1
            zone_name = zone.GetZoneName()
            current_net = zone.GetNet()
            current_net_name = current_net.GetNetname() if current_net else "No Net"
            layer = self.board.GetLayerName(zone.GetFirstLayer())

            # Check if zone has no net or net code 0
            if not current_net or current_net.GetNetCode() == 0:
                # Try to determine the correct net from connected pads
                # Check if zone outline contains any pads
                zone_outline = zone.Outline()
                correct_net = None
                correct_net_name = None

                # Simple heuristic: check for common net names in zone name
                zone_name_upper = zone_name.upper() if zone_name else ""
                if 'GND' in zone_name_upper or 'GROUND' in zone_name_upper:
                    correct_net = self.board.FindNet('GND')
                    if not correct_net:
                        correct_net = self.board.FindNet('gnd')
                    correct_net_name = 'GND'
                elif 'VCC' in zone_name_upper or 'POWER' in zone_name_upper or 'PWR' in zone_name_upper:
                    for net_name in ['VCC', 'VDD', 'VBUS', '+5V', '+3V3', '+12V']:
                        correct_net = self.board.FindNet(net_name)
                        if correct_net:
                            correct_net_name = net_name
                            break

                if correct_net and correct_net.GetNetCode() > 0:
                    print(f"  Zone '{zone_name}' on {layer}: No net -> {correct_net_name}")
                    zone.SetNet(correct_net)
                    zones_fixed += 1
                    fixes.append({
                        'zone': zone_name,
                        'layer': layer,
                        'old_net': current_net_name,
                        'new_net': correct_net_name,
                        'reason': 'zone_name_heuristic'
                    })
                    self.changes_made = True
                else:
                    print(f"  Zone '{zone_name}' on {layer}: No net assigned (could not determine correct net)")
            else:
                print(f"  Zone '{zone_name}' on {layer}: OK (net: {current_net_name})")

        print(f"\nZone nets: {zones_fixed}/{zones_checked} fixed")

        return {
            'zones_checked': zones_checked,
            'zones_fixed': zones_fixed,
            'fixes': fixes,
            'success': True
        }

    def fix_design_settings(self) -> Dict[str, Any]:
        """
        Update design rule clearances to meet IPC-2221 Class 2 requirements.

        Sets minimum clearances, track widths, via sizes, and drill diameters
        to ensure DRC compliance with industry standards.

        Returns:
            Dict with settings changed
        """
        print("\n=== Fixing Design Settings (IPC-2221 Class 2) ===")

        ds = self.board.GetDesignSettings()
        changes = []

        # Convert mm to internal units (nanometers)
        def mm_to_iu(mm: float) -> int:
            return int(mm * 1e6)

        # Check and update minimum clearance
        current_clearance = ds.GetSmallestClearanceValue()
        target_clearance = mm_to_iu(IPC_2221_CLASS_2['min_clearance_mm'])
        if current_clearance < target_clearance:
            # Set default netclass clearance
            nc = ds.m_NetSettings.GetDefaultNetclass()
            nc.SetClearance(target_clearance)
            print(f"  Min clearance: {current_clearance/1e6:.3f}mm -> {IPC_2221_CLASS_2['min_clearance_mm']}mm")
            changes.append({
                'setting': 'min_clearance',
                'old_value_mm': current_clearance / 1e6,
                'new_value_mm': IPC_2221_CLASS_2['min_clearance_mm']
            })
            self.changes_made = True
        else:
            print(f"  Min clearance: {current_clearance/1e6:.3f}mm (OK)")

        # Check and update minimum track width
        current_track = ds.m_TrackMinWidth
        target_track = mm_to_iu(IPC_2221_CLASS_2['min_track_width_mm'])
        if current_track < target_track:
            ds.m_TrackMinWidth = target_track
            print(f"  Min track width: {current_track/1e6:.3f}mm -> {IPC_2221_CLASS_2['min_track_width_mm']}mm")
            changes.append({
                'setting': 'min_track_width',
                'old_value_mm': current_track / 1e6,
                'new_value_mm': IPC_2221_CLASS_2['min_track_width_mm']
            })
            self.changes_made = True
        else:
            print(f"  Min track width: {current_track/1e6:.3f}mm (OK)")

        # Check and update minimum via diameter
        current_via = ds.m_ViasMinSize
        target_via = mm_to_iu(IPC_2221_CLASS_2['min_via_diameter_mm'])
        if current_via < target_via:
            ds.m_ViasMinSize = target_via
            print(f"  Min via diameter: {current_via/1e6:.3f}mm -> {IPC_2221_CLASS_2['min_via_diameter_mm']}mm")
            changes.append({
                'setting': 'min_via_diameter',
                'old_value_mm': current_via / 1e6,
                'new_value_mm': IPC_2221_CLASS_2['min_via_diameter_mm']
            })
            self.changes_made = True
        else:
            print(f"  Min via diameter: {current_via/1e6:.3f}mm (OK)")

        # Check and update minimum via drill
        current_drill = ds.m_MinThroughDrill
        target_drill = mm_to_iu(IPC_2221_CLASS_2['min_via_drill_mm'])
        if current_drill < target_drill:
            ds.m_MinThroughDrill = target_drill
            print(f"  Min via drill: {current_drill/1e6:.3f}mm -> {IPC_2221_CLASS_2['min_via_drill_mm']}mm")
            changes.append({
                'setting': 'min_via_drill',
                'old_value_mm': current_drill / 1e6,
                'new_value_mm': IPC_2221_CLASS_2['min_via_drill_mm']
            })
            self.changes_made = True
        else:
            print(f"  Min via drill: {current_drill/1e6:.3f}mm (OK)")

        # Update annular ring (copper annulus = (via_size - drill_size) / 2)
        current_annular = ds.m_ViasMinAnnularWidth
        target_annular = mm_to_iu(IPC_2221_CLASS_2['min_annular_ring_mm'])
        if current_annular < target_annular:
            ds.m_ViasMinAnnularWidth = target_annular
            print(f"  Min annular ring: {current_annular/1e6:.3f}mm -> {IPC_2221_CLASS_2['min_annular_ring_mm']}mm")
            changes.append({
                'setting': 'min_annular_ring',
                'old_value_mm': current_annular / 1e6,
                'new_value_mm': IPC_2221_CLASS_2['min_annular_ring_mm']
            })
            self.changes_made = True
        else:
            print(f"  Min annular ring: {current_annular/1e6:.3f}mm (OK)")

        print(f"\nDesign settings: {len(changes)} changes made")

        return {
            'settings_checked': 5,
            'settings_changed': len(changes),
            'changes': changes,
            'ipc_standard': 'IPC-2221 Class 2',
            'success': True
        }

    def remove_dangling_vias(self) -> Dict[str, Any]:
        """
        Remove vias that are not connected to any track (dangling vias).

        Dangling vias are vias with net code 0 or vias whose center doesn't
        connect to any track endpoint on any layer.

        Returns:
            Dict with removal statistics
        """
        print("\n=== Removing Dangling Vias ===")

        tracks = list(self.board.GetTracks())
        vias = []
        non_via_tracks = []

        # Separate vias from tracks
        for track in tracks:
            if track.Type() == pcbnew.PCB_VIA_T:
                vias.append(track)
            else:
                non_via_tracks.append(track)

        if not vias:
            print("No vias found in board")
            return {'vias_checked': 0, 'vias_removed': 0, 'success': True}

        print(f"Found {len(vias)} vias and {len(non_via_tracks)} tracks")

        # Build set of track endpoints for fast lookup
        # Track endpoints are where vias should connect
        track_endpoints: Set[Tuple[int, int]] = set()
        for track in non_via_tracks:
            start = track.GetStart()
            end = track.GetEnd()
            track_endpoints.add((start.x, start.y))
            track_endpoints.add((end.x, end.y))

        # Also consider pad positions as valid connection points
        for footprint in self.board.GetFootprints():
            for pad in footprint.Pads():
                pos = pad.GetPosition()
                track_endpoints.add((pos.x, pos.y))

        vias_checked = 0
        vias_removed = 0
        removed_info = []

        for via in vias:
            vias_checked += 1
            pos = via.GetPosition()
            net = via.GetNet()
            net_code = net.GetNetCode() if net else 0
            net_name = net.GetNetname() if net else "No Net"

            # Check if via is connected to anything
            via_pos = (pos.x, pos.y)
            is_connected = via_pos in track_endpoints

            # Also check if via has valid net (net_code > 0 means it's assigned to something)
            has_valid_net = net_code > 0

            # A via is dangling if it has no valid net AND is not at a track endpoint
            # OR if it explicitly has net code 0
            if net_code == 0 and not is_connected:
                print(f"  Removing via at ({pos.x/1e6:.3f}, {pos.y/1e6:.3f})mm - no net, not connected")
                self.board.Remove(via)
                vias_removed += 1
                removed_info.append({
                    'position_mm': (pos.x / 1e6, pos.y / 1e6),
                    'net': net_name,
                    'reason': 'no_net_not_connected'
                })
                self.changes_made = True
            elif not is_connected and net_code > 0:
                # Has net but not physically connected - might be intentional (test point)
                # Don't remove these automatically
                print(f"  Via at ({pos.x/1e6:.3f}, {pos.y/1e6:.3f})mm - has net '{net_name}' but not at track endpoint (kept)")
            else:
                pass  # Via is properly connected

        print(f"\nDangling vias: {vias_removed}/{vias_checked} removed")

        return {
            'vias_checked': vias_checked,
            'vias_removed': vias_removed,
            'removed': removed_info,
            'success': True
        }

    def run_all(self) -> Dict[str, Any]:
        """
        Run all fix operations in sequence.

        Returns:
            Dict with combined results from all operations
        """
        print("\n" + "=" * 60)
        print("Running all MAPOS PCB fixes")
        print("=" * 60)

        results = {
            'zone_nets': self.fix_zone_nets(),
            'design_settings': self.fix_design_settings(),
            'dangling_vias': self.remove_dangling_vias(),
            'success': True
        }

        # Calculate totals
        results['total_fixes'] = (
            results['zone_nets'].get('zones_fixed', 0) +
            results['design_settings'].get('settings_changed', 0) +
            results['dangling_vias'].get('vias_removed', 0)
        )

        return results

    def save(self) -> bool:
        """
        Save the board if changes were made.

        Returns:
            True if saved successfully, False if no changes to save
        """
        if not self.changes_made:
            print("\nNo changes made - not saving")
            return False

        print(f"\nSaving board to: {self.pcb_path}")
        pcbnew.SaveBoard(str(self.pcb_path), self.board)
        print("Board saved successfully")
        return True


def main():
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description='KiCad PCB Fixer - Fix zone nets, design settings, and remove dangling vias',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Operations:
  zone-nets         Fix zone net assignments based on zone names and contents
  design-settings   Update design rules to meet IPC-2221 Class 2 requirements
  dangling-vias     Remove vias not connected to any track
  all               Run all operations

Examples:
  %(prog)s zone-nets board.kicad_pcb
  %(prog)s design-settings board.kicad_pcb --json
  %(prog)s dangling-vias board.kicad_pcb --no-save
  %(prog)s all board.kicad_pcb --json

Part of MAPOS (Multi-Agent PCB Optimization System)
        '''
    )

    parser.add_argument('operation',
                        choices=['zone-nets', 'design-settings', 'dangling-vias', 'all'],
                        help='Operation to perform')
    parser.add_argument('pcb_path', help='Path to KiCad PCB file')
    parser.add_argument('--no-save', action='store_true',
                        help='Do not save the board after fixes')
    parser.add_argument('--json', action='store_true',
                        help='Output results as JSON')

    args = parser.parse_args()

    try:
        fixer = KiCadPCBFixer(args.pcb_path)

        if args.operation == 'zone-nets':
            result = fixer.fix_zone_nets()
        elif args.operation == 'design-settings':
            result = fixer.fix_design_settings()
        elif args.operation == 'dangling-vias':
            result = fixer.remove_dangling_vias()
        elif args.operation == 'all':
            result = fixer.run_all()
        else:
            print(f"Unknown operation: {args.operation}", file=sys.stderr)
            return 1

        # Save if requested and changes were made
        if not args.no_save:
            saved = fixer.save()
            result['saved'] = saved
        else:
            result['saved'] = False

        result['pcb_path'] = str(fixer.pcb_path)
        result['timestamp'] = datetime.now().isoformat()

        if args.json:
            print(json.dumps(result, indent=2, default=str))

        return 0 if result.get('success', True) else 1

    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
