#!/usr/bin/env python3
"""
KiCad Headless Runner - Execute pcbnew operations in headless or headed mode.

This module provides cross-platform support for running KiCad pcbnew operations:
- macOS: Uses native display or Xquartz
- Linux/K8s: Uses Xvfb virtual display (headless)
- Docker: Runs with Xvfb sidecar

For K8s deployment, use with the kicad-xvfb sidecar container that provides
the virtual display at DISPLAY=:99

Usage:
    python3 kicad_headless_runner.py fill-zones board.kicad_pcb
    python3 kicad_headless_runner.py assign-nets board.kicad_pcb
    python3 kicad_headless_runner.py run-drc board.kicad_pcb

Environment Variables:
    DISPLAY: X11 display (default :99 for headless)
    KICAD_PYTHON: Path to KiCad's Python (auto-detected)
    KICAD_HEADLESS: Set to "1" to force Xvfb mode
"""

import os
import sys
import json
import subprocess
import tempfile
import shutil
from pathlib import Path
from typing import Dict, Any, Optional, Tuple


# Auto-detect KiCad Python paths by platform
def get_kicad_paths() -> Tuple[str, str]:
    """Get KiCad Python executable and site-packages path."""
    if sys.platform == 'darwin':  # macOS
        python_path = "/Applications/KiCad/KiCad.app/Contents/Frameworks/Python.framework/Versions/3.9/bin/python3"
        site_packages = "/Applications/KiCad/KiCad.app/Contents/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages"
    elif sys.platform == 'linux':  # Linux/K8s
        # Check common Linux installation paths
        for python_path in [
            "/usr/bin/python3",
            "/usr/local/bin/python3",
        ]:
            if Path(python_path).exists():
                break
        site_packages = "/usr/lib/python3/dist-packages"  # Debian/Ubuntu
        if not Path(site_packages).exists():
            site_packages = "/usr/lib64/python3/site-packages"  # Fedora
    else:
        raise RuntimeError(f"Unsupported platform: {sys.platform}")

    return python_path, site_packages


KICAD_PYTHON, KICAD_SITE_PACKAGES = get_kicad_paths()


class KiCadHeadlessRunner:
    """
    Run KiCad pcbnew operations in headless or headed mode.

    For K8s deployment, this expects an Xvfb sidecar container providing
    a virtual display. The typical K8s setup is:

    ```yaml
    containers:
      - name: kicad-worker
        image: adverant/kicad-worker:latest
        env:
          - name: DISPLAY
            value: ":99"
      - name: xvfb
        image: adverant/xvfb:latest
        command: ["Xvfb", ":99", "-screen", "0", "1920x1080x24"]
    ```
    """

    def __init__(self, pcb_path: str, headless: bool = None):
        """
        Initialize the runner.

        Args:
            pcb_path: Path to the KiCad PCB file
            headless: Force headless mode (auto-detected if None)
        """
        self.pcb_path = Path(pcb_path).resolve()
        if not self.pcb_path.exists():
            raise FileNotFoundError(f"PCB file not found: {pcb_path}")

        # Determine if we need headless mode
        if headless is None:
            # Auto-detect: headless if DISPLAY is not set or is :99
            display = os.environ.get('DISPLAY', '')
            self.headless = not display or display == ':99' or os.environ.get('KICAD_HEADLESS') == '1'
        else:
            self.headless = headless

        # Ensure DISPLAY is set for Xvfb
        if self.headless and 'DISPLAY' not in os.environ:
            os.environ['DISPLAY'] = ':99'

    def _run_pcbnew_script(self, script: str, timeout: int = 300) -> Dict[str, Any]:
        """
        Execute a pcbnew Python script.

        Args:
            script: Python script to execute
            timeout: Execution timeout in seconds

        Returns:
            Dict with execution results
        """
        # Create a temporary script file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(script)
            script_path = f.name

        try:
            env = os.environ.copy()
            env['PYTHONPATH'] = KICAD_SITE_PACKAGES

            result = subprocess.run(
                [KICAD_PYTHON, script_path],
                capture_output=True,
                text=True,
                timeout=timeout,
                env=env
            )

            # Try to parse JSON output
            output = result.stdout.strip()
            try:
                return json.loads(output)
            except json.JSONDecodeError:
                return {
                    'success': result.returncode == 0,
                    'stdout': output,
                    'stderr': result.stderr,
                    'returncode': result.returncode
                }

        finally:
            Path(script_path).unlink(missing_ok=True)

    def fill_zones(self) -> Dict[str, Any]:
        """
        Fill all zones in the PCB using ZONE_FILLER.

        Returns:
            Dict with zone fill results
        """
        script = f'''
import sys
import os
import json

# Suppress KiCad telemetry
os.environ["KICAD_NO_TRACKING"] = "1"

# Add pcbnew to path
sys.path.insert(0, "{KICAD_SITE_PACKAGES}")

try:
    import pcbnew

    board = pcbnew.LoadBoard("{self.pcb_path}")
    if board is None:
        print(json.dumps({{"success": False, "error": "Failed to load board"}}))
        sys.exit(1)

    zones = board.Zones()
    zone_count = len(zones)

    zone_info = []
    for zone in zones:
        zone_info.append({{
            "name": zone.GetZoneName(),
            "net": zone.GetNetname(),
            "layer": board.GetLayerName(zone.GetFirstLayer())
        }})

    if zone_count > 0:
        filler = pcbnew.ZONE_FILLER(board)
        filler.Fill(zones)
        pcbnew.SaveBoard("{self.pcb_path}", board)

    print(json.dumps({{
        "success": True,
        "zones_filled": zone_count,
        "zones": zone_info
    }}))

except Exception as e:
    print(json.dumps({{"success": False, "error": str(e)}}))
    sys.exit(1)
'''
        return self._run_pcbnew_script(script)

    def assign_orphan_nets(self) -> Dict[str, Any]:
        """
        Assign nets to orphan pads (pads with no net assigned).

        Returns:
            Dict with assignment results
        """
        script = f'''
import sys
import os
import json

os.environ["KICAD_NO_TRACKING"] = "1"
sys.path.insert(0, "{KICAD_SITE_PACKAGES}")

try:
    import pcbnew

    board = pcbnew.LoadBoard("{self.pcb_path}")
    if board is None:
        print(json.dumps({{"success": False, "error": "Failed to load board"}}))
        sys.exit(1)

    gnd_net = board.FindNet("GND")
    assigned = []

    for footprint in board.GetFootprints():
        ref = footprint.GetReference()

        for pad in footprint.Pads():
            if pad.GetNet().GetNetCode() == 0:  # No net
                pad_num = pad.GetNumber()

                # MOSFET source/tab pads typically GND
                if ref.startswith("MOS") and pad_num in ["3", "4"]:
                    if gnd_net:
                        pad.SetNet(gnd_net)
                        assigned.append({{
                            "ref": ref,
                            "pad": pad_num,
                            "net": "GND"
                        }})

    pcbnew.SaveBoard("{self.pcb_path}", board)

    print(json.dumps({{
        "success": True,
        "pads_assigned": len(assigned),
        "assignments": assigned
    }}))

except Exception as e:
    print(json.dumps({{"success": False, "error": str(e)}}))
    sys.exit(1)
'''
        return self._run_pcbnew_script(script)

    def get_board_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the PCB board.

        Returns:
            Dict with board statistics
        """
        script = f'''
import sys
import os
import json

os.environ["KICAD_NO_TRACKING"] = "1"
sys.path.insert(0, "{KICAD_SITE_PACKAGES}")

try:
    import pcbnew

    board = pcbnew.LoadBoard("{self.pcb_path}")
    if board is None:
        print(json.dumps({{"success": False, "error": "Failed to load board"}}))
        sys.exit(1)

    # Count elements
    footprints = list(board.GetFootprints())
    zones = list(board.Zones())
    tracks = [t for t in board.GetTracks() if isinstance(t, pcbnew.PCB_TRACK)]
    vias = [t for t in board.GetTracks() if isinstance(t, pcbnew.PCB_VIA)]

    # Count orphan pads
    orphan_pads = 0
    for fp in footprints:
        for pad in fp.Pads():
            if pad.GetNet().GetNetCode() == 0:
                orphan_pads += 1

    print(json.dumps({{
        "success": True,
        "footprints": len(footprints),
        "zones": len(zones),
        "tracks": len(tracks),
        "vias": len(vias),
        "orphan_pads": orphan_pads
    }}))

except Exception as e:
    print(json.dumps({{"success": False, "error": str(e)}}))
    sys.exit(1)
'''
        return self._run_pcbnew_script(script, timeout=60)


def main():
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description='Run KiCad pcbnew operations in headless mode',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Commands:
  fill-zones    Fill all zones using ZONE_FILLER
  assign-nets   Assign nets to orphan pads
  stats         Get board statistics

Environment:
  DISPLAY       X11 display (default :99 for headless)
  KICAD_HEADLESS=1  Force headless mode

For K8s deployment, use with Xvfb sidecar container.
        '''
    )
    parser.add_argument('command', choices=['fill-zones', 'assign-nets', 'stats'],
                       help='Command to execute')
    parser.add_argument('pcb_path', help='Path to KiCad PCB file')
    parser.add_argument('--headless', action='store_true',
                       help='Force headless mode')
    parser.add_argument('--json', action='store_true',
                       help='Output as JSON')

    args = parser.parse_args()

    try:
        runner = KiCadHeadlessRunner(args.pcb_path, headless=args.headless)

        if args.command == 'fill-zones':
            result = runner.fill_zones()
        elif args.command == 'assign-nets':
            result = runner.assign_orphan_nets()
        elif args.command == 'stats':
            result = runner.get_board_stats()
        else:
            print(f"Unknown command: {args.command}", file=sys.stderr)
            return 1

        if args.json:
            print(json.dumps(result, indent=2))
        else:
            if result.get('success'):
                print(f"Success: {result}")
            else:
                print(f"Failed: {result.get('error', 'Unknown error')}", file=sys.stderr)
                return 1

        return 0 if result.get('success') else 1

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


if __name__ == '__main__':
    sys.exit(main())
