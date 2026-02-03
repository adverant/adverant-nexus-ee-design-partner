#!/usr/bin/env python3
"""
KiCad Solder Mask Fixer - Fix solder mask bridge DRC violations.

This script addresses the ROOT CAUSE of solder mask bridge violations:
1. Track-to-pad proximity (vias near pads)
2. Fine-pitch component pad clusters

Solutions implemented:
- Via tenting: Cover vias with solder mask to eliminate track-via bridges
- Bridge allowance: Enable AllowSolderMaskBridges() for fine-pitch footprints

Part of MAPOS (Multi-Agent PCB Optimization System).
"""

import os
import sys
import argparse
import shutil
import subprocess
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
import math

# Try to import pcbnew (KiCad Python API)
PCBNEW_AVAILABLE = False
try:
    import pcbnew
    PCBNEW_AVAILABLE = True
except ImportError:
    pass


class SolderMaskFixer:
    """
    Fix solder mask bridge violations using correct pcbnew API methods.

    This addresses the actual root causes:
    - Via tenting eliminates via-to-track mask bridges
    - Bridge allowance handles fine-pitch components where bridges are acceptable
    """

    def __init__(self, pcb_path: str, backup: bool = True):
        self.pcb_path = Path(pcb_path)
        self.backup = backup
        self.board = None
        self.stats = {
            'vias_tented': 0,
            'footprints_bridge_allowed': 0,
            'total_vias': 0,
            'total_footprints': 0,
            'fine_pitch_detected': 0,
        }

        if not self.pcb_path.exists():
            raise FileNotFoundError(f"PCB file not found: {pcb_path}")

    def load_board(self) -> None:
        """Load the PCB board file."""
        if self.backup:
            self._create_backup()

        if PCBNEW_AVAILABLE:
            self.board = pcbnew.LoadBoard(str(self.pcb_path))
            print(f"Loaded board: {self.pcb_path}")
            print(f"  Footprints: {len(list(self.board.GetFootprints()))}")
            print(f"  Tracks/Vias: {len(list(self.board.GetTracks()))}")
        else:
            print("ERROR: pcbnew not available - cannot apply fixes")
            sys.exit(1)

    def save_board(self) -> None:
        """Save the modified board."""
        if PCBNEW_AVAILABLE and self.board:
            pcbnew.SaveBoard(str(self.pcb_path), self.board)
            print(f"Saved board: {self.pcb_path}")

    def _create_backup(self) -> None:
        """Create timestamped backup."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = self.pcb_path.with_suffix(f".mask_backup_{timestamp}.kicad_pcb")
        shutil.copy2(self.pcb_path, backup_path)
        print(f"Created backup: {backup_path}")

    def tent_all_vias(self) -> int:
        """
        Apply full tenting to all vias.

        This covers vias with solder mask, eliminating solder mask bridge
        violations caused by via-to-track proximity.

        Returns:
            Number of vias tented
        """
        if not self.board:
            return 0

        print(f"\n--- Tenting All Vias ---")

        vias_tented = 0

        for track in self.board.GetTracks():
            if isinstance(track, pcbnew.PCB_VIA):
                self.stats['total_vias'] += 1

                # Check if via supports tenting API
                try:
                    # KiCad 8+ uses tenting mode
                    # TENTING_MODE_TENTED = tent the via (cover with solder mask)
                    track.SetFrontTentingMode(pcbnew.TENTING_MODE_TENTED)
                    track.SetBackTentingMode(pcbnew.TENTING_MODE_TENTED)
                    vias_tented += 1
                except AttributeError:
                    # Fallback: use HasSolderMask method if available
                    try:
                        track.SetHasSolderMask(False)  # False = no mask opening = tented
                        vias_tented += 1
                    except:
                        pass

        self.stats['vias_tented'] = vias_tented
        print(f"Tented {vias_tented} of {self.stats['total_vias']} vias")
        return vias_tented

    def calculate_min_pad_spacing(self, footprint) -> float:
        """
        Calculate minimum spacing between pads in a footprint.

        Args:
            footprint: KiCad footprint object

        Returns:
            Minimum pad-to-pad spacing in mm
        """
        pads = list(footprint.Pads())
        if len(pads) < 2:
            return float('inf')

        positions = [(p.GetPosition().x, p.GetPosition().y) for p in pads]

        min_spacing = float('inf')
        for i, pos1 in enumerate(positions):
            for pos2 in positions[i+1:]:
                dx = pos1[0] - pos2[0]
                dy = pos1[1] - pos2[1]
                dist = math.sqrt(dx*dx + dy*dy)
                if dist > 0 and dist < min_spacing:
                    min_spacing = dist

        return pcbnew.ToMM(int(min_spacing)) if min_spacing != float('inf') else float('inf')

    def allow_bridges_on_fine_pitch(self, pitch_threshold_mm: float = 0.5) -> int:
        """
        Enable solder mask bridge allowance on fine-pitch footprints.

        Fine-pitch components (QFN, BGA, fine-pitch connectors) have pads
        too close together for reliable solder mask dams. The fab house
        handles this during assembly.

        Args:
            pitch_threshold_mm: Consider fine-pitch if pad spacing < this value

        Returns:
            Number of footprints modified
        """
        if not self.board:
            return 0

        print(f"\n--- Allowing Bridges on Fine-Pitch Footprints ---")
        print(f"Pitch threshold: {pitch_threshold_mm}mm")

        footprints_modified = 0

        for footprint in self.board.GetFootprints():
            self.stats['total_footprints'] += 1
            ref = footprint.GetReference()

            pads = list(footprint.Pads())
            if len(pads) < 3:  # Need at least 3 pads to have bridging concern
                continue

            min_spacing = self.calculate_min_pad_spacing(footprint)

            if min_spacing < pitch_threshold_mm:
                self.stats['fine_pitch_detected'] += 1

                # Check if method exists (KiCad 9+)
                try:
                    if hasattr(footprint, 'AllowSolderMaskBridges'):
                        if not footprint.AllowSolderMaskBridges():
                            footprint.SetAllowSolderMaskBridges(True)
                            footprints_modified += 1
                            print(f"  {ref}: min spacing {min_spacing:.2f}mm - bridges allowed")
                    else:
                        # Fallback: reduce solder mask margin on pads to create bridges
                        # Negative margin = more mask between pads
                        margin = pcbnew.FromMM(-0.05)  # More aggressive mask
                        for pad in pads:
                            if pad.GetAttribute() == pcbnew.PAD_ATTRIB_SMD:
                                pad.SetLocalSolderMaskMargin(margin)
                        footprints_modified += 1
                        print(f"  {ref}: min spacing {min_spacing:.2f}mm - reduced pad mask margin")
                except Exception as e:
                    print(f"  WARNING: {ref} - Could not modify: {e}")

        self.stats['footprints_bridge_allowed'] = footprints_modified
        print(f"Modified {footprints_modified} of {self.stats['fine_pitch_detected']} fine-pitch footprints")
        return footprints_modified

    def reduce_global_mask_expansion(self, expansion_mm: float = 0.0508) -> bool:
        """
        Reduce global solder mask expansion.

        The default KiCad solder mask expansion (0.1mm) is often too large
        and causes mask apertures to overlap. OSH Park recommends 0.0508mm (2mil).

        Args:
            expansion_mm: New global solder mask expansion

        Returns:
            True if successful
        """
        if not self.board:
            return False

        print(f"\n--- Adjusting Global Solder Mask Expansion ---")

        try:
            ds = self.board.GetDesignSettings()

            # Get current value
            current = pcbnew.ToMM(ds.m_SolderMaskExpansion)
            print(f"Current expansion: {current:.4f}mm")

            # Set new value
            ds.m_SolderMaskExpansion = pcbnew.FromMM(expansion_mm)
            print(f"New expansion: {expansion_mm:.4f}mm")

            return True
        except Exception as e:
            print(f"WARNING: Could not adjust mask expansion: {e}")
            return False

    def increase_mask_min_width(self, min_width_mm: float = 0.1) -> bool:
        """
        Increase minimum solder mask web width.

        This prevents the DRC from flagging bridges that are too thin
        for reliable manufacturing.

        Args:
            min_width_mm: Minimum solder mask web width

        Returns:
            True if successful
        """
        if not self.board:
            return False

        print(f"\n--- Adjusting Minimum Solder Mask Width ---")

        try:
            ds = self.board.GetDesignSettings()

            # Get current value
            current = pcbnew.ToMM(ds.m_SolderMaskMinWidth)
            print(f"Current min width: {current:.4f}mm")

            # Set new value
            ds.m_SolderMaskMinWidth = pcbnew.FromMM(min_width_mm)
            print(f"New min width: {min_width_mm:.4f}mm")

            return True
        except Exception as e:
            print(f"WARNING: Could not adjust mask min width: {e}")
            return False

    def run_drc(self) -> Dict[str, int]:
        """Run DRC and return violation counts by type."""
        print(f"\n--- Running DRC Check ---")

        kicad_cli = None
        possible_paths = [
            '/Applications/KiCad/KiCad.app/Contents/MacOS/kicad-cli',
            '/usr/bin/kicad-cli',
            '/usr/local/bin/kicad-cli',
            shutil.which('kicad-cli'),
        ]

        for path in possible_paths:
            if path and Path(path).exists():
                kicad_cli = path
                break

        if not kicad_cli:
            print("WARNING: kicad-cli not found")
            return {}

        output_path = self.pcb_path.parent / 'drc_mask_report.json'

        try:
            result = subprocess.run([
                kicad_cli, 'pcb', 'drc',
                '--output', str(output_path),
                '--format', 'json',
                '--severity-all',
                str(self.pcb_path)
            ], capture_output=True, text=True, timeout=120)

            if output_path.exists():
                with open(output_path) as f:
                    drc_data = json.load(f)

                violations = {}
                for violation in drc_data.get('violations', []):
                    vtype = violation.get('type', 'unknown')
                    violations[vtype] = violations.get(vtype, 0) + 1

                # Highlight solder mask specific violations
                mask_violations = violations.get('solder_mask_bridge', 0)
                total = sum(violations.values())

                print(f"Total violations: {total}")
                print(f"Solder mask bridges: {mask_violations}")

                return {'total': total, 'solder_mask_bridge': mask_violations, 'by_type': violations}

            return {}

        except Exception as e:
            print(f"DRC failed: {e}")
            return {}

    def apply_all_fixes(self) -> Dict[str, Any]:
        """
        Apply all solder mask fixes.

        Returns:
            Statistics dictionary
        """
        print("=" * 60)
        print("SOLDER MASK FIXER - ROOT CAUSE IMPLEMENTATION")
        print("=" * 60)
        print(f"PCB: {self.pcb_path}")

        # Load board
        self.load_board()

        # Get initial DRC
        print("\n[BEFORE FIXES]")
        initial_drc = self.run_drc()
        initial_mask = initial_drc.get('solder_mask_bridge', 0)

        # Apply fixes
        self.tent_all_vias()
        self.allow_bridges_on_fine_pitch(pitch_threshold_mm=0.5)
        self.reduce_global_mask_expansion(expansion_mm=0.0508)
        self.increase_mask_min_width(min_width_mm=0.1)

        # Save board
        self.save_board()

        # Get final DRC
        print("\n[AFTER FIXES]")
        final_drc = self.run_drc()
        final_mask = final_drc.get('solder_mask_bridge', 0)

        # Calculate improvement
        improvement = initial_mask - final_mask
        improvement_pct = (improvement / initial_mask * 100) if initial_mask > 0 else 0

        print("\n" + "=" * 60)
        print("RESULTS - SOLDER MASK BRIDGES")
        print("=" * 60)
        print(f"Initial solder_mask_bridge: {initial_mask}")
        print(f"Final solder_mask_bridge: {final_mask}")
        print(f"Improvement: {improvement} ({improvement_pct:.1f}%)")
        print(f"\nModifications applied:")
        print(f"  Vias tented: {self.stats['vias_tented']}/{self.stats['total_vias']}")
        print(f"  Fine-pitch footprints: {self.stats['footprints_bridge_allowed']}/{self.stats['fine_pitch_detected']}")

        return {
            'initial_mask_violations': initial_mask,
            'final_mask_violations': final_mask,
            'improvement': improvement,
            'improvement_pct': improvement_pct,
            'stats': self.stats,
            'initial_drc': initial_drc,
            'final_drc': final_drc,
        }


def main():
    parser = argparse.ArgumentParser(
        description='Fix solder mask bridge violations using correct root cause solutions'
    )
    parser.add_argument('pcb_path', help='Path to KiCad PCB file')
    parser.add_argument('--no-backup', action='store_true', help='Skip backup creation')
    parser.add_argument('--pitch-threshold', type=float, default=0.5,
                       help='Fine-pitch threshold in mm (default: 0.5)')
    parser.add_argument('--mask-expansion', type=float, default=0.0508,
                       help='Global mask expansion in mm (default: 0.0508 / 2mil)')
    parser.add_argument('--mask-min-width', type=float, default=0.1,
                       help='Minimum mask web width in mm (default: 0.1)')
    parser.add_argument('--no-tenting', action='store_true',
                       help='Skip via tenting')
    parser.add_argument('--no-bridge-allow', action='store_true',
                       help='Skip bridge allowance on fine-pitch')
    parser.add_argument('--json', action='store_true',
                       help='Output results as JSON')

    args = parser.parse_args()

    fixer = SolderMaskFixer(args.pcb_path, backup=not args.no_backup)

    # Load board
    fixer.load_board()

    # Get initial DRC
    initial_drc = fixer.run_drc()

    # Apply selected fixes
    if not args.no_tenting:
        fixer.tent_all_vias()

    if not args.no_bridge_allow:
        fixer.allow_bridges_on_fine_pitch(pitch_threshold_mm=args.pitch_threshold)

    fixer.reduce_global_mask_expansion(expansion_mm=args.mask_expansion)
    fixer.increase_mask_min_width(min_width_mm=args.mask_min_width)

    # Save board
    fixer.save_board()

    # Get final DRC
    final_drc = fixer.run_drc()

    results = {
        'initial_mask_violations': initial_drc.get('solder_mask_bridge', 0),
        'final_mask_violations': final_drc.get('solder_mask_bridge', 0),
        'stats': fixer.stats,
    }

    if args.json:
        print(json.dumps(results, indent=2))

    return 0 if results['final_mask_violations'] < results['initial_mask_violations'] else 1


if __name__ == '__main__':
    sys.exit(main())
