#!/usr/bin/env python3
"""
FreeRouting Integration for MAPOS

This module provides integration with FreeRouting (https://freerouting.org) for
handling complex dangling track cases that can't be fixed by simple extend/remove.

FreeRouting is an open-source auto-router that can:
1. Route partially completed PCBs
2. Rip-up and reroute problematic nets
3. Complete unfinished traces

Uses KiCad's Specctra DSN/SES export/import for interoperability.

Part of MAPOS Gaming AI Remediation - Phase 4
"""

import os
import sys
import json
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import List, Dict, Optional, Any, Tuple
from dataclasses import dataclass
from datetime import datetime

# Try to import kicad_paths for cross-platform support
try:
    from kicad_paths import KICAD_CLI
except ImportError:
    KICAD_CLI = None


@dataclass
class FreeRoutingResult:
    """Result of a FreeRouting operation."""
    success: bool
    nets_rerouted: int = 0
    passes_used: int = 0
    duration_seconds: float = 0.0
    error: Optional[str] = None
    log_output: Optional[str] = None


class FreeRoutingBridge:
    """
    Bridge between MAPOS and FreeRouting for complex routing cases.

    FreeRouting operates via:
    1. KiCad exports PCB to Specctra DSN format
    2. FreeRouting routes/reroutes the DSN
    3. FreeRouting outputs Specctra SES (session) file
    4. KiCad imports SES back into the PCB

    Prerequisites:
    - kicad-cli (for DSN/SES export/import)
    - FreeRouting JAR file (freerouting-X.X.X.jar)
    - Java runtime (for FreeRouting)
    """

    # FreeRouting JAR search paths
    FREEROUTING_PATHS = [
        '/opt/freerouting/freerouting.jar',
        '/usr/local/share/freerouting/freerouting.jar',
        '/usr/share/freerouting/freerouting.jar',
        '/app/freerouting.jar',  # Docker/K8s
        Path.home() / '.local/share/freerouting/freerouting.jar',
        Path.home() / 'freerouting.jar',
    ]

    def __init__(self, pcb_path: str):
        """
        Initialize the FreeRouting bridge.

        Args:
            pcb_path: Path to the .kicad_pcb file
        """
        self.pcb_path = Path(pcb_path)
        if not self.pcb_path.exists():
            raise FileNotFoundError(f"PCB file not found: {pcb_path}")

        self.kicad_cli = self._find_kicad_cli()
        self.freerouting_jar = self._find_freerouting()
        self.java = self._find_java()

    def _find_kicad_cli(self) -> Optional[str]:
        """Find kicad-cli executable."""
        if KICAD_CLI and Path(KICAD_CLI).exists():
            return KICAD_CLI

        candidates = [
            '/Applications/KiCad/KiCad.app/Contents/MacOS/kicad-cli',  # macOS
            '/usr/bin/kicad-cli',  # Linux
            '/usr/local/bin/kicad-cli',
            shutil.which('kicad-cli'),
        ]

        for path in candidates:
            if path and Path(path).exists():
                return path
        return None

    def _find_freerouting(self) -> Optional[str]:
        """Find FreeRouting JAR file."""
        # Check environment variable first
        env_path = os.environ.get('FREEROUTING_JAR')
        if env_path and Path(env_path).exists():
            return env_path

        for path in self.FREEROUTING_PATHS:
            if Path(path).exists():
                return str(path)

        return None

    def _find_java(self) -> Optional[str]:
        """Find Java runtime."""
        candidates = [
            os.environ.get('JAVA_HOME', '') + '/bin/java',
            '/usr/bin/java',
            '/usr/local/bin/java',
            shutil.which('java'),
        ]

        for path in candidates:
            if path and Path(path).exists():
                return path
        return None

    def is_available(self) -> Dict[str, bool]:
        """
        Check if FreeRouting integration is available.

        Returns:
            Dict with availability status for each component
        """
        return {
            'kicad_cli': self.kicad_cli is not None,
            'freerouting': self.freerouting_jar is not None,
            'java': self.java is not None,
            'all_available': all([
                self.kicad_cli is not None,
                self.freerouting_jar is not None,
                self.java is not None,
            ])
        }

    def export_dsn(self, output_path: Path) -> bool:
        """
        Export PCB to Specctra DSN format.

        Args:
            output_path: Path to write the DSN file

        Returns:
            True if export succeeded
        """
        if not self.kicad_cli:
            raise RuntimeError("kicad-cli not found - cannot export DSN")

        result = subprocess.run([
            self.kicad_cli, 'pcb', 'export', 'specctra',
            '--output', str(output_path),
            str(self.pcb_path)
        ], capture_output=True, text=True, timeout=60)

        return result.returncode == 0 and output_path.exists()

    def import_ses(self, ses_path: Path, output_path: Optional[Path] = None) -> bool:
        """
        Import Specctra SES file back into PCB.

        Args:
            ses_path: Path to the SES file
            output_path: Optional output PCB path (defaults to original)

        Returns:
            True if import succeeded
        """
        if not self.kicad_cli:
            raise RuntimeError("kicad-cli not found - cannot import SES")

        if output_path is None:
            output_path = self.pcb_path

        result = subprocess.run([
            self.kicad_cli, 'pcb', 'import', 'specctra',
            '--output', str(output_path),
            str(ses_path)
        ], capture_output=True, text=True, timeout=60)

        return result.returncode == 0

    def reroute_nets(
        self,
        net_names: Optional[List[str]] = None,
        max_passes: int = 50,
        thread_count: int = 4,
        timeout_seconds: int = 300
    ) -> FreeRoutingResult:
        """
        Rip-up and reroute specific nets (or all unrouted) using FreeRouting.

        Args:
            net_names: List of net names to reroute (None = route all unrouted)
            max_passes: Maximum routing passes
            thread_count: Number of threads for parallel routing
            timeout_seconds: Maximum time for routing

        Returns:
            FreeRoutingResult with operation details
        """
        availability = self.is_available()
        if not availability['all_available']:
            missing = [k for k, v in availability.items() if not v and k != 'all_available']
            return FreeRoutingResult(
                success=False,
                error=f"Missing components: {', '.join(missing)}"
            )

        start_time = datetime.now()

        with tempfile.TemporaryDirectory() as tmpdir:
            dsn_path = Path(tmpdir) / 'board.dsn'
            ses_path = Path(tmpdir) / 'board.ses'
            log_path = Path(tmpdir) / 'freerouting.log'

            # Step 1: Export DSN
            try:
                if not self.export_dsn(dsn_path):
                    return FreeRoutingResult(
                        success=False,
                        error="Failed to export DSN from KiCad"
                    )
            except Exception as e:
                return FreeRoutingResult(
                    success=False,
                    error=f"DSN export error: {e}"
                )

            # Step 2: Run FreeRouting
            cmd = [
                self.java, '-jar', self.freerouting_jar,
                '-de', str(dsn_path),  # Design file input
                '-do', str(ses_path),  # Session file output
                '-mp', str(max_passes),  # Max passes
                '-mt', str(thread_count),  # Thread count
                '-dct',  # Disable component transform (preserve positions)
            ]

            # Add net filter if specified
            if net_names:
                cmd.extend(['-inc', ','.join(net_names)])

            try:
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=timeout_seconds,
                    env={**os.environ, 'DISPLAY': ''}  # Headless mode
                )

                log_output = result.stdout + '\n' + result.stderr

                if result.returncode != 0:
                    return FreeRoutingResult(
                        success=False,
                        error=f"FreeRouting failed with code {result.returncode}",
                        log_output=log_output
                    )

            except subprocess.TimeoutExpired:
                return FreeRoutingResult(
                    success=False,
                    error=f"FreeRouting timed out after {timeout_seconds}s"
                )
            except Exception as e:
                return FreeRoutingResult(
                    success=False,
                    error=f"FreeRouting execution error: {e}"
                )

            # Step 3: Check if SES was created
            if not ses_path.exists():
                return FreeRoutingResult(
                    success=False,
                    error="FreeRouting did not produce output SES file",
                    log_output=log_output if 'log_output' in locals() else None
                )

            # Step 4: Import SES back into PCB
            try:
                # First backup the original PCB
                backup_path = self.pcb_path.with_suffix(
                    f'.backup_freerouting_{datetime.now().strftime("%Y%m%d_%H%M%S")}.kicad_pcb'
                )
                shutil.copy2(self.pcb_path, backup_path)

                if not self.import_ses(ses_path):
                    # Restore backup on failure
                    shutil.copy2(backup_path, self.pcb_path)
                    return FreeRoutingResult(
                        success=False,
                        error="Failed to import SES back into KiCad"
                    )
            except Exception as e:
                return FreeRoutingResult(
                    success=False,
                    error=f"SES import error: {e}"
                )

            duration = (datetime.now() - start_time).total_seconds()

            # Parse routing results from log
            passes_used = 0
            try:
                if 'Pass' in log_output:
                    # Count completed passes from log
                    passes_used = log_output.lower().count('pass')
            except:
                pass

            return FreeRoutingResult(
                success=True,
                nets_rerouted=len(net_names) if net_names else 0,
                passes_used=passes_used,
                duration_seconds=duration,
                log_output=log_output
            )

    def route_remaining(self, max_passes: int = 100) -> FreeRoutingResult:
        """
        Route all remaining unconnected nets.

        This is useful for completing partially routed PCBs.

        Args:
            max_passes: Maximum routing passes

        Returns:
            FreeRoutingResult with operation details
        """
        return self.reroute_nets(net_names=None, max_passes=max_passes)

    def optimize_routes(self, max_passes: int = 30) -> FreeRoutingResult:
        """
        Optimize existing routes without adding new connections.

        Uses FreeRouting's optimization passes to clean up trace paths.

        Args:
            max_passes: Maximum optimization passes

        Returns:
            FreeRoutingResult with operation details
        """
        # FreeRouting's -oit (optimize routes) flag
        availability = self.is_available()
        if not availability['all_available']:
            missing = [k for k, v in availability.items() if not v and k != 'all_available']
            return FreeRoutingResult(
                success=False,
                error=f"Missing components: {', '.join(missing)}"
            )

        start_time = datetime.now()

        with tempfile.TemporaryDirectory() as tmpdir:
            dsn_path = Path(tmpdir) / 'board.dsn'
            ses_path = Path(tmpdir) / 'board.ses'

            try:
                if not self.export_dsn(dsn_path):
                    return FreeRoutingResult(
                        success=False,
                        error="Failed to export DSN"
                    )

                cmd = [
                    self.java, '-jar', self.freerouting_jar,
                    '-de', str(dsn_path),
                    '-do', str(ses_path),
                    '-mp', str(max_passes),
                    '-oit',  # Optimize input traces
                    '-dct',  # Don't change positions
                ]

                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=300,
                    env={**os.environ, 'DISPLAY': ''}
                )

                if result.returncode == 0 and ses_path.exists():
                    backup_path = self.pcb_path.with_suffix('.backup_optimize.kicad_pcb')
                    shutil.copy2(self.pcb_path, backup_path)

                    if self.import_ses(ses_path):
                        return FreeRoutingResult(
                            success=True,
                            passes_used=max_passes,
                            duration_seconds=(datetime.now() - start_time).total_seconds(),
                            log_output=result.stdout
                        )

                return FreeRoutingResult(
                    success=False,
                    error=f"Optimization failed: {result.stderr}"
                )

            except Exception as e:
                return FreeRoutingResult(
                    success=False,
                    error=str(e)
                )


def get_dangling_nets_from_drc(drc_json_path: str) -> List[str]:
    """
    Extract net names with dangling tracks from a DRC JSON report.

    Args:
        drc_json_path: Path to kicad-cli DRC JSON output

    Returns:
        List of net names that have dangling track violations
    """
    try:
        with open(drc_json_path) as f:
            drc_data = json.load(f)

        dangling_nets = set()

        for violation in drc_data.get('violations', []):
            if violation.get('type') == 'track_dangling':
                # Try to extract net name from violation details
                items = violation.get('items', [])
                for item in items:
                    net = item.get('net', '')
                    if net:
                        dangling_nets.add(net)

        return list(dangling_nets)

    except Exception:
        return []


def main():
    """CLI interface for FreeRouting integration."""
    import argparse

    parser = argparse.ArgumentParser(
        description='FreeRouting Integration for MAPOS - Auto-routing for complex cases',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Check FreeRouting availability
  python freerouting_integration.py board.kicad_pcb --check

  # Route all remaining unconnected nets
  python freerouting_integration.py board.kicad_pcb --route-remaining

  # Reroute specific nets
  python freerouting_integration.py board.kicad_pcb --reroute GND +3V3 VCC

  # Optimize existing routes
  python freerouting_integration.py board.kicad_pcb --optimize
        """
    )

    parser.add_argument('pcb_path', help='Path to .kicad_pcb file')

    action_group = parser.add_mutually_exclusive_group()
    action_group.add_argument(
        '--check', '-c',
        action='store_true',
        help='Check FreeRouting availability'
    )
    action_group.add_argument(
        '--route-remaining', '-r',
        action='store_true',
        help='Route all remaining unconnected nets'
    )
    action_group.add_argument(
        '--reroute',
        nargs='+',
        metavar='NET',
        help='Reroute specific nets'
    )
    action_group.add_argument(
        '--optimize', '-o',
        action='store_true',
        help='Optimize existing routes'
    )

    parser.add_argument(
        '--max-passes', '-p',
        type=int,
        default=50,
        help='Maximum routing passes (default: 50)'
    )
    parser.add_argument(
        '--json',
        action='store_true',
        help='Output results as JSON'
    )

    args = parser.parse_args()

    try:
        bridge = FreeRoutingBridge(args.pcb_path)

        if args.check:
            availability = bridge.is_available()
            if args.json:
                print(json.dumps(availability, indent=2))
            else:
                print("\nFreeRouting Availability:")
                for component, available in availability.items():
                    if component != 'all_available':
                        status = "OK" if available else "MISSING"
                        print(f"  [{status}] {component}")
                print(f"\n  Ready: {'YES' if availability['all_available'] else 'NO'}")
            return 0 if availability['all_available'] else 1

        elif args.route_remaining:
            print(f"Routing remaining unconnected nets (max {args.max_passes} passes)...")
            result = bridge.route_remaining(max_passes=args.max_passes)

        elif args.reroute:
            print(f"Rerouting nets: {', '.join(args.reroute)}")
            result = bridge.reroute_nets(
                net_names=args.reroute,
                max_passes=args.max_passes
            )

        elif args.optimize:
            print(f"Optimizing existing routes (max {args.max_passes} passes)...")
            result = bridge.optimize_routes(max_passes=args.max_passes)

        else:
            # Default: check availability
            availability = bridge.is_available()
            if args.json:
                print(json.dumps(availability, indent=2))
            else:
                print("\nUse --help for available commands")
                print(f"FreeRouting ready: {'YES' if availability['all_available'] else 'NO'}")
            return 0

        # Output result
        if args.json:
            print(json.dumps({
                'success': result.success,
                'nets_rerouted': result.nets_rerouted,
                'passes_used': result.passes_used,
                'duration_seconds': result.duration_seconds,
                'error': result.error,
            }, indent=2))
        else:
            print(f"\n{'='*60}")
            print(f"FreeRouting Result: {'SUCCESS' if result.success else 'FAILED'}")
            print(f"{'='*60}")
            if result.success:
                print(f"  Nets rerouted: {result.nets_rerouted}")
                print(f"  Passes used: {result.passes_used}")
                print(f"  Duration: {result.duration_seconds:.1f}s")
            else:
                print(f"  Error: {result.error}")
            print(f"{'='*60}\n")

        return 0 if result.success else 1

    except FileNotFoundError as e:
        error = {'success': False, 'error': str(e)}
        if args.json:
            print(json.dumps(error, indent=2))
        else:
            print(f"Error: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        error = {'success': False, 'error': str(e), 'type': type(e).__name__}
        if args.json:
            print(json.dumps(error, indent=2))
        else:
            print(f"Error ({type(e).__name__}): {e}", file=sys.stderr)
        return 1


if __name__ == '__main__':
    sys.exit(main())
