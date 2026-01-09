#!/usr/bin/env python3
"""
Zone Filler - Fill copper zones in KiCad PCB files.

This script fills unfilled zones in KiCad PCB files to generate the actual
copper polygons. This is necessary for:
1. Proper visualization (filled zones appear as solid copper)
2. Accurate copper coverage calculations
3. Fabrication (Gerber files need filled zones)

Usage:
    python zone_filler.py --pcb board.kicad_pcb --output board_filled.kicad_pcb
    python zone_filler.py --pcb board.kicad_pcb --in-place
"""

import argparse
import re
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

# Try to import pcbnew (KiCad Python API)
try:
    import pcbnew
    HAS_PCBNEW = True
except ImportError:
    HAS_PCBNEW = False


def fill_zones_with_pcbnew(pcb_path: str, output_path: Optional[str] = None) -> bool:
    """
    Fill all zones in a PCB file using KiCad's pcbnew API.

    This is the preferred method as it uses KiCad's actual zone filling
    algorithm, including proper thermal relief and clearance calculations.

    Args:
        pcb_path: Path to input .kicad_pcb file
        output_path: Path for output file (or None for in-place)

    Returns:
        True if successful
    """
    if not HAS_PCBNEW:
        print("Error: pcbnew not available", file=sys.stderr)
        return False

    try:
        print(f"Loading PCB: {pcb_path}")
        board = pcbnew.LoadBoard(pcb_path)

        print("Filling zones...")
        filler = pcbnew.ZONE_FILLER(board)
        zones = board.Zones()

        print(f"Found {zones.size()} zones to fill")

        # Fill all zones
        zones_to_fill = [zones[i] for i in range(zones.size())]
        filler.Fill(zones_to_fill)

        # Count filled zones
        filled_count = 0
        for zone in zones_to_fill:
            if zone.GetFilledPolysList(zone.GetFirstLayer()).OutlineCount() > 0:
                filled_count += 1

        print(f"Filled {filled_count}/{len(zones_to_fill)} zones")

        # Save the board
        out_path = output_path or pcb_path
        pcbnew.SaveBoard(out_path, board)
        print(f"Saved filled PCB to: {out_path}")

        return True

    except Exception as e:
        print(f"Error filling zones: {e}", file=sys.stderr)
        return False


def generate_simple_fill(zone_polygon: List[Tuple[float, float]],
                         clearance: float = 0.2) -> List[Tuple[float, float]]:
    """
    Generate a simple filled polygon for a zone.

    This is a simplified fallback when pcbnew is not available.
    It creates a solid fill without thermal reliefs.

    Args:
        zone_polygon: List of (x, y) points defining zone outline
        clearance: Clearance from zone edge

    Returns:
        List of (x, y) points for filled polygon (inset by clearance)
    """
    # Simple inset - just reduce each coordinate
    # A proper implementation would use polygon offsetting
    if len(zone_polygon) < 3:
        return zone_polygon

    # Calculate centroid
    cx = sum(p[0] for p in zone_polygon) / len(zone_polygon)
    cy = sum(p[1] for p in zone_polygon) / len(zone_polygon)

    # Inset each point toward centroid by clearance
    filled = []
    for x, y in zone_polygon:
        dx = cx - x
        dy = cy - y
        dist = (dx**2 + dy**2) ** 0.5
        if dist > 0:
            scale = clearance / dist
            filled.append((x + dx * scale, y + dy * scale))
        else:
            filled.append((x, y))

    return filled


def fill_zones_sexp(pcb_path: str, output_path: Optional[str] = None) -> bool:
    """
    Fill zones by modifying the S-expression directly.

    This is a fallback method that adds simple filled_polygon entries
    to zones that don't have them. It won't handle thermal reliefs or
    proper clearances, but it will make zones visible.

    Args:
        pcb_path: Path to input .kicad_pcb file
        output_path: Path for output file (or None for in-place)

    Returns:
        True if successful
    """
    try:
        print(f"Loading PCB (S-expression mode): {pcb_path}")

        with open(pcb_path, 'r') as f:
            content = f.read()

        # Find all zones and their outlines
        zone_pattern = r'\(zone\s+.*?\(polygon\s+\(pts\s+(.*?)\)\s*\).*?\)'
        zones = list(re.finditer(zone_pattern, content, re.DOTALL))

        print(f"Found {len(zones)} zones")

        # Check which zones already have filled_polygon
        filled_pattern = r'filled_polygon'
        has_filled = filled_pattern in content

        if has_filled:
            print("Zones already have filled polygons")
            if output_path and output_path != pcb_path:
                shutil.copy(pcb_path, output_path)
            return True

        # Add filled polygons to each zone
        # This is a simplified approach - just copies the outline as fill
        modified_content = content

        for match in zones:
            zone_text = match.group(0)
            pts_text = match.group(1)

            # Parse points
            point_pattern = r'\(xy\s+([\d.]+)\s+([\d.]+)\)'
            points = [(float(m.group(1)), float(m.group(2)))
                     for m in re.finditer(point_pattern, pts_text)]

            if len(points) < 3:
                continue

            # Generate filled polygon (same as outline for simplicity)
            # Format each point on its own line for readability
            filled_pts = "\n\t\t\t\t".join(f"(xy {x:.3f} {y:.3f})" for x, y in points)

            # Find the zone's layer
            layer_match = re.search(r'\(layer\s+"([^"]+)"\)', zone_text)
            layer = layer_match.group(1) if layer_match else "F.Cu"

            # Create filled_polygon entry (properly indented)
            filled_polygon = f'''
		(filled_polygon
			(layer "{layer}")
			(pts
				{filled_pts}
			)
		)
'''

            # Insert before the final closing ) of the zone
            # Find where to insert (before the last two closing parens)
            # The zone structure ends with:  )  ) at the end
            zone_end_idx = zone_text.rfind('\t)')  # Find the indented closing
            if zone_end_idx == -1:
                zone_end_idx = zone_text.rfind(')')

            new_zone = zone_text[:zone_end_idx] + filled_polygon + zone_text[zone_end_idx:]

            modified_content = modified_content.replace(zone_text, new_zone)

        # Save the modified content
        out_path = output_path or pcb_path
        with open(out_path, 'w') as f:
            f.write(modified_content)

        print(f"Added filled polygons to {len(zones)} zones")
        print(f"Saved to: {out_path}")

        return True

    except Exception as e:
        print(f"Error filling zones (S-expression): {e}", file=sys.stderr)
        return False


def fill_zones(pcb_path: str, output_path: Optional[str] = None,
               use_pcbnew: bool = True) -> bool:
    """
    Fill zones in a PCB file.

    Tries pcbnew first (for proper fills), falls back to S-expression
    modification if pcbnew is not available.

    Args:
        pcb_path: Path to input .kicad_pcb file
        output_path: Path for output file (or None for in-place)
        use_pcbnew: Try to use pcbnew first

    Returns:
        True if successful
    """
    if use_pcbnew and HAS_PCBNEW:
        return fill_zones_with_pcbnew(pcb_path, output_path)
    else:
        print("Note: Using S-expression fallback (pcbnew not available)")
        return fill_zones_sexp(pcb_path, output_path)


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description='Fill copper zones in KiCad PCB files'
    )
    parser.add_argument(
        '--pcb', '-p',
        type=str,
        required=True,
        help='Path to .kicad_pcb file'
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        help='Output path (default: modify in place)'
    )
    parser.add_argument(
        '--in-place',
        action='store_true',
        help='Modify file in place (create backup)'
    )
    parser.add_argument(
        '--no-backup',
        action='store_true',
        help='Don\'t create backup file'
    )
    parser.add_argument(
        '--force-sexp',
        action='store_true',
        help='Force S-expression mode (skip pcbnew)'
    )

    args = parser.parse_args()

    pcb_path = args.pcb
    if not Path(pcb_path).exists():
        print(f"Error: PCB file not found: {pcb_path}", file=sys.stderr)
        sys.exit(1)

    # Handle in-place editing
    output_path = args.output
    if args.in_place:
        output_path = pcb_path
        if not args.no_backup:
            backup_path = pcb_path + ".backup"
            shutil.copy(pcb_path, backup_path)
            print(f"Created backup: {backup_path}")

    # Fill zones
    success = fill_zones(
        pcb_path,
        output_path,
        use_pcbnew=not args.force_sexp
    )

    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
