#!/usr/bin/env python3
"""
KiCad Trace Adjuster - Adjust trace widths for power integrity using pcbnew API.

This script MUST be run with KiCad's bundled Python interpreter that has pcbnew available.
Works cross-platform: macOS, Linux, and Docker containers.

Usage:
    python kicad_trace_adjuster.py net board.kicad_pcb GND 2.0 [--json]
    python kicad_trace_adjuster.py power board.kicad_pcb --width 2.0 [--json]

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


# Common power net name patterns
POWER_NET_PATTERNS = [
    'GND', 'AGND', 'DGND', 'PGND', 'EARTH', 'GROUND', 'VSS', 'GNDA', 'GNDD',
    'VCC', 'VDD', 'VBUS', 'VIN', 'VOUT', 'VSYS',
    '+1V0', '+1V2', '+1V5', '+1V8', '+2V5',
    '+3V', '+3V3', '+3.3V', '+3V3D', '+3V3A',
    '+5V', '+5V0', '+5VD', '+5VA', 'USB_5V',
    '+9V', '+12V', '+15V', '+24V', '+48V', '+58V',
    '-5V', '-12V', '-15V', '-24V',
    'VBAT', 'VBATT', 'VCELL',
    'PWR', 'POWER', 'VPWR',
]

# Minimum recommended trace widths for current (IPC-2152)
# Values in mm for 1oz copper, 10C temperature rise
CURRENT_TRACE_WIDTH = {
    0.5: 0.3,    # 0.5A -> 0.3mm minimum
    1.0: 0.5,    # 1A -> 0.5mm minimum
    2.0: 0.8,    # 2A -> 0.8mm minimum
    3.0: 1.0,    # 3A -> 1.0mm minimum
    5.0: 1.5,    # 5A -> 1.5mm minimum
    10.0: 2.5,   # 10A -> 2.5mm minimum
}


class TraceAdjuster:
    """Adjust trace widths for power integrity."""

    def __init__(self, pcb_path: str):
        """
        Initialize adjuster with a PCB file.

        Args:
            pcb_path: Path to the .kicad_pcb file
        """
        self.pcb_path = Path(pcb_path).resolve()
        if not self.pcb_path.exists():
            raise FileNotFoundError(f"PCB file not found: {self.pcb_path}")

        print(f"Loading board: {self.pcb_path}")
        self.board = pcbnew.LoadBoard(str(self.pcb_path))
        self.changes_made = False

    def _mm_to_iu(self, mm: float) -> int:
        """Convert millimeters to KiCad internal units (nanometers)."""
        return int(mm * 1e6)

    def _iu_to_mm(self, iu: int) -> float:
        """Convert KiCad internal units (nanometers) to millimeters."""
        return iu / 1e6

    def _is_power_net(self, net_name: str) -> bool:
        """Check if a net name matches power net patterns."""
        if not net_name:
            return False
        net_upper = net_name.upper()
        for pattern in POWER_NET_PATTERNS:
            if pattern.upper() in net_upper or net_upper == pattern.upper():
                return True
        return False

    def adjust_net_traces(self, net_name: str, width_mm: float, save: bool = True) -> Dict[str, Any]:
        """
        Adjust all traces for a specific net to a target width.

        Only widens traces - does not narrow them (to preserve intentional design decisions).

        Args:
            net_name: Name of the net to adjust
            width_mm: Target trace width in millimeters
            save: Whether to save the board after adjustment

        Returns:
            Dict with adjustment statistics
        """
        print(f"\n=== Adjusting Traces for Net: {net_name} ===")
        print(f"Target width: {width_mm}mm")

        # Find the net
        net = self.board.FindNet(net_name)
        if not net:
            print(f"Warning: Net '{net_name}' not found in board")
            # Try case-insensitive search
            for netinfo in self.board.GetNetInfo().NetsByName():
                if netinfo.upper() == net_name.upper():
                    net = self.board.FindNet(netinfo)
                    print(f"  Found net with different case: {netinfo}")
                    break

        if not net:
            return {
                'net': net_name,
                'target_width_mm': width_mm,
                'traces_checked': 0,
                'traces_adjusted': 0,
                'error': f"Net '{net_name}' not found",
                'success': False
            }

        net_code = net.GetNetCode()
        target_width_iu = self._mm_to_iu(width_mm)

        traces_checked = 0
        traces_adjusted = 0
        adjustments = []

        # Iterate through all tracks
        for track in self.board.GetTracks():
            # Skip vias
            if track.Type() == pcbnew.PCB_VIA_T:
                continue

            # Check if track belongs to target net
            if track.GetNetCode() != net_code:
                continue

            traces_checked += 1
            current_width = track.GetWidth()

            # Only widen traces, never narrow them
            if current_width < target_width_iu:
                old_width_mm = self._iu_to_mm(current_width)
                track.SetWidth(target_width_iu)
                traces_adjusted += 1
                self.changes_made = True

                # Get track position for logging
                start = track.GetStart()
                end = track.GetEnd()
                layer = self.board.GetLayerName(track.GetLayer())

                adjustments.append({
                    'layer': layer,
                    'old_width_mm': old_width_mm,
                    'new_width_mm': width_mm,
                    'start_mm': (self._iu_to_mm(start.x), self._iu_to_mm(start.y)),
                    'end_mm': (self._iu_to_mm(end.x), self._iu_to_mm(end.y)),
                })

                print(f"  Track on {layer}: {old_width_mm:.3f}mm -> {width_mm}mm")

        print(f"\nNet '{net_name}': {traces_adjusted}/{traces_checked} traces adjusted")

        # Save if requested
        if save and self.changes_made:
            print(f"\nSaving board to: {self.pcb_path}")
            pcbnew.SaveBoard(str(self.pcb_path), self.board)
            print("Board saved successfully")

        return {
            'net': net_name,
            'net_code': net_code,
            'target_width_mm': width_mm,
            'traces_checked': traces_checked,
            'traces_adjusted': traces_adjusted,
            'adjustments': adjustments,
            'saved': save and self.changes_made,
            'success': True
        }

    def adjust_power_traces(self, width_mm: float, save: bool = True) -> Dict[str, Any]:
        """
        Widen all power traces (GND, VCC, etc.) to a target width.

        Automatically detects power nets by pattern matching and widens all
        traces that are below the target width.

        Args:
            width_mm: Target trace width in millimeters
            save: Whether to save the board after adjustment

        Returns:
            Dict with adjustment statistics
        """
        print(f"\n=== Adjusting Power Traces ===")
        print(f"Target width: {width_mm}mm")

        # Find all power nets in the board
        power_nets = []
        for net_name in self.board.GetNetInfo().NetsByName():
            if self._is_power_net(net_name):
                net = self.board.FindNet(net_name)
                if net and net.GetNetCode() > 0:
                    power_nets.append((net_name, net.GetNetCode()))

        if not power_nets:
            print("No power nets found in board")
            return {
                'target_width_mm': width_mm,
                'power_nets_found': 0,
                'traces_checked': 0,
                'traces_adjusted': 0,
                'success': True
            }

        print(f"Found {len(power_nets)} power nets:")
        for net_name, net_code in power_nets:
            print(f"  - {net_name} (net code: {net_code})")

        target_width_iu = self._mm_to_iu(width_mm)
        power_net_codes = {net_code for _, net_code in power_nets}

        traces_checked = 0
        traces_adjusted = 0
        nets_affected = set()
        adjustments = []

        # Iterate through all tracks
        for track in self.board.GetTracks():
            # Skip vias
            if track.Type() == pcbnew.PCB_VIA_T:
                continue

            # Check if track belongs to a power net
            net_code = track.GetNetCode()
            if net_code not in power_net_codes:
                continue

            traces_checked += 1
            current_width = track.GetWidth()

            # Only widen traces, never narrow them
            if current_width < target_width_iu:
                old_width_mm = self._iu_to_mm(current_width)
                track.SetWidth(target_width_iu)
                traces_adjusted += 1
                self.changes_made = True

                # Get track info
                net = track.GetNet()
                net_name = net.GetNetname() if net else "Unknown"
                nets_affected.add(net_name)
                layer = self.board.GetLayerName(track.GetLayer())

                adjustments.append({
                    'net': net_name,
                    'layer': layer,
                    'old_width_mm': old_width_mm,
                    'new_width_mm': width_mm,
                })

                print(f"  {net_name} on {layer}: {old_width_mm:.3f}mm -> {width_mm}mm")

        print(f"\nPower traces: {traces_adjusted}/{traces_checked} adjusted")
        print(f"Nets affected: {', '.join(sorted(nets_affected)) if nets_affected else 'none'}")

        # Save if requested
        if save and self.changes_made:
            print(f"\nSaving board to: {self.pcb_path}")
            pcbnew.SaveBoard(str(self.pcb_path), self.board)
            print("Board saved successfully")

        return {
            'target_width_mm': width_mm,
            'power_nets_found': len(power_nets),
            'power_nets': [name for name, _ in power_nets],
            'traces_checked': traces_checked,
            'traces_adjusted': traces_adjusted,
            'nets_affected': list(nets_affected),
            'adjustments': adjustments,
            'saved': save and self.changes_made,
            'success': True
        }

    def report_trace_widths(self) -> Dict[str, Any]:
        """
        Report trace widths by net, without making changes.

        Returns:
            Dict with trace width statistics
        """
        print("\n=== Trace Width Report ===")

        by_net: Dict[str, Dict[str, Any]] = {}

        for track in self.board.GetTracks():
            # Skip vias
            if track.Type() == pcbnew.PCB_VIA_T:
                continue

            net = track.GetNet()
            net_name = net.GetNetname() if net else "No Net"
            width_mm = self._iu_to_mm(track.GetWidth())

            if net_name not in by_net:
                by_net[net_name] = {
                    'trace_count': 0,
                    'min_width_mm': width_mm,
                    'max_width_mm': width_mm,
                    'is_power': self._is_power_net(net_name),
                }

            by_net[net_name]['trace_count'] += 1
            by_net[net_name]['min_width_mm'] = min(by_net[net_name]['min_width_mm'], width_mm)
            by_net[net_name]['max_width_mm'] = max(by_net[net_name]['max_width_mm'], width_mm)

        # Sort by net name
        sorted_nets = sorted(by_net.items(), key=lambda x: (not x[1]['is_power'], x[0]))

        print("\nPower Nets:")
        for net_name, info in sorted_nets:
            if info['is_power']:
                print(f"  {net_name}: {info['trace_count']} traces, "
                      f"width: {info['min_width_mm']:.3f}-{info['max_width_mm']:.3f}mm")

        print("\nSignal Nets (sample):")
        signal_count = 0
        for net_name, info in sorted_nets:
            if not info['is_power']:
                signal_count += 1
                if signal_count <= 10:  # Only show first 10 signal nets
                    print(f"  {net_name}: {info['trace_count']} traces, "
                          f"width: {info['min_width_mm']:.3f}-{info['max_width_mm']:.3f}mm")

        if signal_count > 10:
            print(f"  ... and {signal_count - 10} more signal nets")

        return {
            'total_nets': len(by_net),
            'power_nets': sum(1 for info in by_net.values() if info['is_power']),
            'signal_nets': sum(1 for info in by_net.values() if not info['is_power']),
            'by_net': by_net,
            'success': True
        }


def main():
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description='KiCad Trace Adjuster - Adjust trace widths for power integrity',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Operations:
  net <pcb> <net_name> <width_mm>    Adjust traces for a specific net
  power <pcb> --width <mm>           Widen all power traces (GND, VCC, etc.)
  report <pcb>                       Report trace widths without changes

Examples:
  %(prog)s net board.kicad_pcb GND 2.0
  %(prog)s net board.kicad_pcb "+3V3" 1.0 --json
  %(prog)s power board.kicad_pcb --width 2.0
  %(prog)s power board.kicad_pcb --width 1.5 --no-save --json
  %(prog)s report board.kicad_pcb --json

Part of MAPOS (Multi-Agent PCB Optimization System)
        '''
    )

    subparsers = parser.add_subparsers(dest='operation', help='Operation to perform')

    # Net-specific adjustment
    net_parser = subparsers.add_parser('net', help='Adjust traces for a specific net')
    net_parser.add_argument('pcb_path', help='Path to KiCad PCB file')
    net_parser.add_argument('net_name', help='Name of the net to adjust')
    net_parser.add_argument('width_mm', type=float, help='Target trace width in mm')
    net_parser.add_argument('--no-save', action='store_true', help='Do not save changes')
    net_parser.add_argument('--json', action='store_true', help='Output as JSON')

    # Power traces adjustment
    power_parser = subparsers.add_parser('power', help='Widen all power traces')
    power_parser.add_argument('pcb_path', help='Path to KiCad PCB file')
    power_parser.add_argument('--width', type=float, required=True,
                              help='Target trace width in mm')
    power_parser.add_argument('--no-save', action='store_true', help='Do not save changes')
    power_parser.add_argument('--json', action='store_true', help='Output as JSON')

    # Report mode
    report_parser = subparsers.add_parser('report', help='Report trace widths')
    report_parser.add_argument('pcb_path', help='Path to KiCad PCB file')
    report_parser.add_argument('--json', action='store_true', help='Output as JSON')

    args = parser.parse_args()

    if not args.operation:
        parser.print_help()
        return 1

    try:
        adjuster = TraceAdjuster(args.pcb_path)

        if args.operation == 'net':
            result = adjuster.adjust_net_traces(
                args.net_name,
                args.width_mm,
                save=not args.no_save
            )
        elif args.operation == 'power':
            result = adjuster.adjust_power_traces(
                args.width,
                save=not args.no_save
            )
        elif args.operation == 'report':
            result = adjuster.report_trace_widths()
        else:
            print(f"Unknown operation: {args.operation}", file=sys.stderr)
            return 1

        result['pcb_path'] = str(adjuster.pcb_path)
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
