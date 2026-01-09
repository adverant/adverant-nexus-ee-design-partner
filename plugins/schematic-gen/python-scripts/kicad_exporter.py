#!/usr/bin/env python3
"""
KiCad Exporter - Automated Layer Image Export via KiCad CLI

This is the CORRECT way to export PCB layer images. DO NOT use Python-based
rendering which creates incorrect visualizations.

Exports:
- Individual layer SVGs/PNGs
- Layer composites
- PDFs for documentation
- Gerbers for manufacturing

Usage:
    python kicad_exporter.py --pcb board.kicad_pcb --output-dir ./output
    python kicad_exporter.py --pcb board.kicad_pcb --layers F.Cu,B.Cu --format png
    python kicad_exporter.py --schematic sheet.kicad_sch --format pdf
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass


@dataclass
class ExportConfig:
    """Configuration for KiCad exports."""
    kicad_cli: str = "/Applications/KiCad/KiCad.app/Contents/MacOS/kicad-cli"
    output_format: str = "svg"  # svg, pdf
    page_size_mode: int = 2  # 0=with frame, 1=page size, 2=board area only
    dpi: int = 300
    include_edge_cuts: bool = True


# Standard PCB layers to export
STANDARD_PCB_LAYERS = [
    ("F.Cu", "Front Copper"),
    ("B.Cu", "Back Copper"),
    ("In1.Cu", "Inner Layer 1"),
    ("In2.Cu", "Inner Layer 2"),
    ("In3.Cu", "Inner Layer 3"),
    ("In4.Cu", "Inner Layer 4"),
    ("F.SilkS", "Front Silkscreen"),
    ("B.SilkS", "Back Silkscreen"),
    ("F.Mask", "Front Solder Mask"),
    ("B.Mask", "Back Solder Mask"),
    ("F.Paste", "Front Paste"),
    ("B.Paste", "Back Paste"),
    ("Edge.Cuts", "Board Outline"),
    ("F.Fab", "Front Fabrication"),
    ("B.Fab", "Back Fabrication"),
]

# Common layer composites
LAYER_COMPOSITES = {
    "top_view": ["F.Cu", "F.SilkS", "Edge.Cuts"],
    "bottom_view": ["B.Cu", "B.SilkS", "Edge.Cuts"],
    "all_copper": ["F.Cu", "In1.Cu", "In2.Cu", "B.Cu", "Edge.Cuts"],
    "assembly_top": ["F.SilkS", "F.Fab", "Edge.Cuts"],
    "assembly_bottom": ["B.SilkS", "B.Fab", "Edge.Cuts"],
}


class KiCadExporter:
    """
    Exports PCB and schematic files using KiCad CLI.

    This is the canonical way to generate layer images - NOT Python rendering.
    """

    def __init__(self, config: Optional[ExportConfig] = None):
        self.config = config or ExportConfig()
        self._verify_kicad_cli()

    def _verify_kicad_cli(self):
        """Verify KiCad CLI is available."""
        if not Path(self.config.kicad_cli).exists():
            # Try to find it
            common_paths = [
                "/Applications/KiCad/KiCad.app/Contents/MacOS/kicad-cli",
                "/usr/bin/kicad-cli",
                "/usr/local/bin/kicad-cli",
                shutil.which("kicad-cli")
            ]
            for path in common_paths:
                if path and Path(path).exists():
                    self.config.kicad_cli = path
                    break
            else:
                raise FileNotFoundError(
                    "kicad-cli not found. Install KiCad or set KICAD_CLI env var."
                )

    def _run_cli(self, args: List[str]) -> Tuple[int, str, str]:
        """Run KiCad CLI command."""
        cmd = [self.config.kicad_cli] + args
        result = subprocess.run(cmd, capture_output=True, text=True)
        return result.returncode, result.stdout, result.stderr

    def _get_pcb_layers(self, pcb_path: str) -> List[str]:
        """Get list of layers defined in PCB file."""
        layers = []
        with open(pcb_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
            # Parse layer definitions
            import re
            layer_matches = re.findall(r'\((\d+)\s+"([^"]+)"\s+\w+', content)
            for _, layer_name in layer_matches:
                layers.append(layer_name)
        return layers

    def export_pcb_layer(
        self,
        pcb_path: str,
        output_path: str,
        layers: List[str],
        include_edge_cuts: bool = True
    ) -> bool:
        """
        Export specific PCB layers to SVG.

        Args:
            pcb_path: Path to .kicad_pcb file
            output_path: Output SVG path
            layers: List of layer names (e.g., ["F.Cu", "F.SilkS"])
            include_edge_cuts: Include board outline

        Returns:
            True if successful
        """
        if include_edge_cuts and "Edge.Cuts" not in layers:
            layers = layers + ["Edge.Cuts"]

        layer_str = ",".join(layers)

        args = [
            "pcb", "export", "svg",
            "--layers", layer_str,
            "--output", output_path,
            "--page-size-mode", str(self.config.page_size_mode),
            pcb_path
        ]

        returncode, stdout, stderr = self._run_cli(args)

        if returncode != 0:
            print(f"Error exporting {layer_str}: {stderr}", file=sys.stderr)
            return False

        return True

    def export_pcb_pdf(
        self,
        pcb_path: str,
        output_path: str,
        layers: Optional[List[str]] = None
    ) -> bool:
        """Export PCB to PDF."""
        args = [
            "pcb", "export", "pdf",
            "--output", output_path,
            pcb_path
        ]

        if layers:
            args.insert(3, "--layers")
            args.insert(4, ",".join(layers))

        returncode, stdout, stderr = self._run_cli(args)
        return returncode == 0

    def export_schematic_pdf(
        self,
        schematic_path: str,
        output_path: str
    ) -> bool:
        """Export schematic to PDF."""
        args = [
            "sch", "export", "pdf",
            schematic_path,
            "-o", output_path
        ]

        returncode, stdout, stderr = self._run_cli(args)
        return returncode == 0

    def export_gerbers(
        self,
        pcb_path: str,
        output_dir: str
    ) -> bool:
        """Export Gerber files for manufacturing."""
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        args = [
            "pcb", "export", "gerbers",
            "--output", output_dir,
            pcb_path
        ]

        returncode, stdout, stderr = self._run_cli(args)
        return returncode == 0

    def export_drill(
        self,
        pcb_path: str,
        output_dir: str
    ) -> bool:
        """Export drill files."""
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        args = [
            "pcb", "export", "drill",
            "--output", output_dir,
            pcb_path
        ]

        returncode, stdout, stderr = self._run_cli(args)
        return returncode == 0

    def export_all_layers(
        self,
        pcb_path: str,
        output_dir: str,
        format: str = "svg"
    ) -> Dict[str, str]:
        """
        Export all PCB layers to individual files.

        Args:
            pcb_path: Path to .kicad_pcb file
            output_dir: Output directory
            format: Output format (svg)

        Returns:
            Dict mapping layer name to output path
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Get actual layers in this PCB
        pcb_layers = self._get_pcb_layers(pcb_path)

        results = {}

        for layer_name, description in STANDARD_PCB_LAYERS:
            if layer_name not in pcb_layers and layer_name != "Edge.Cuts":
                continue

            # Skip inner layers if not present
            if layer_name.startswith("In") and layer_name not in pcb_layers:
                continue

            safe_name = layer_name.replace(".", "_")
            output_path = str(output_dir / f"{safe_name}.{format}")

            print(f"  Exporting {layer_name}...")
            success = self.export_pcb_layer(
                pcb_path,
                output_path,
                [layer_name],
                include_edge_cuts=(layer_name != "Edge.Cuts")
            )

            if success:
                results[layer_name] = output_path

        # Export composites
        for composite_name, layers in LAYER_COMPOSITES.items():
            # Only export if all layers exist
            if all(l in pcb_layers or l == "Edge.Cuts" for l in layers):
                output_path = str(output_dir / f"{composite_name}.{format}")
                print(f"  Exporting composite: {composite_name}...")
                success = self.export_pcb_layer(
                    pcb_path,
                    output_path,
                    layers,
                    include_edge_cuts=False
                )
                if success:
                    results[composite_name] = output_path

        return results

    def convert_svg_to_png(
        self,
        svg_path: str,
        png_path: Optional[str] = None,
        size: int = 2000
    ) -> Optional[str]:
        """
        Convert SVG to PNG using macOS Quick Look.

        Args:
            svg_path: Input SVG path
            png_path: Output PNG path (default: same as svg with .png)
            size: Max dimension in pixels

        Returns:
            PNG path if successful, None otherwise
        """
        if png_path is None:
            png_path = svg_path + ".png"

        # Use macOS qlmanage for conversion
        result = subprocess.run(
            ["qlmanage", "-t", "-s", str(size), "-o",
             str(Path(svg_path).parent), svg_path],
            capture_output=True,
            text=True
        )

        if result.returncode == 0:
            # qlmanage creates file with .svg.png extension
            temp_path = svg_path + ".png"
            if Path(temp_path).exists():
                if temp_path != png_path:
                    shutil.move(temp_path, png_path)
                return png_path

        return None


def export_full_design(
    pcb_path: str,
    output_dir: str,
    schematic_paths: Optional[List[str]] = None,
    png_size: int = 2000
) -> Dict[str, any]:
    """
    Export complete design with all layers and formats.

    This is the main entry point for the pipeline.

    Args:
        pcb_path: Path to .kicad_pcb file
        output_dir: Base output directory
        schematic_paths: Optional list of schematic files
        png_size: Size for PNG conversion

    Returns:
        Export results with paths and status
    """
    exporter = KiCadExporter()
    output_dir = Path(output_dir)

    results = {
        "pcb": pcb_path,
        "schematics": schematic_paths or [],
        "outputs": {},
        "errors": []
    }

    # Create subdirectories
    svg_dir = output_dir / "svg"
    png_dir = output_dir / "layer_images"
    pdf_dir = output_dir / "pdfs"
    gerber_dir = output_dir / "gerbers"

    for d in [svg_dir, png_dir, pdf_dir, gerber_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # Export all layers as SVG
    print("Exporting PCB layers...")
    svg_results = exporter.export_all_layers(pcb_path, str(svg_dir))
    results["outputs"]["svg"] = svg_results

    # Convert SVGs to PNGs
    print("Converting to PNG...")
    png_results = {}
    for layer_name, svg_path in svg_results.items():
        safe_name = layer_name.replace(".", "_")
        png_path = str(png_dir / f"{safe_name}.png")
        result = exporter.convert_svg_to_png(svg_path, png_path, png_size)
        if result:
            png_results[layer_name] = result
    results["outputs"]["png"] = png_results

    # Export PDF
    print("Exporting PCB PDF...")
    pcb_pdf = str(pdf_dir / f"{Path(pcb_path).stem}.pdf")
    if exporter.export_pcb_pdf(pcb_path, pcb_pdf):
        results["outputs"]["pcb_pdf"] = pcb_pdf

    # Export schematics
    if schematic_paths:
        print("Exporting schematic PDFs...")
        sch_pdfs = {}
        for sch_path in schematic_paths:
            sch_pdf = str(pdf_dir / f"{Path(sch_path).stem}.pdf")
            if exporter.export_schematic_pdf(sch_path, sch_pdf):
                sch_pdfs[sch_path] = sch_pdf
        results["outputs"]["schematic_pdfs"] = sch_pdfs

    # Export Gerbers
    print("Exporting Gerbers...")
    if exporter.export_gerbers(pcb_path, str(gerber_dir)):
        results["outputs"]["gerbers"] = str(gerber_dir)

    # Export drill files
    if exporter.export_drill(pcb_path, str(gerber_dir)):
        results["outputs"]["drill"] = str(gerber_dir)

    print(f"\nExport complete. Output directory: {output_dir}")
    return results


def main():
    parser = argparse.ArgumentParser(
        description='KiCad Exporter - Automated Layer Image Export'
    )
    parser.add_argument(
        '--pcb', '-p',
        type=str,
        help='Path to .kicad_pcb file'
    )
    parser.add_argument(
        '--schematic', '-s',
        type=str,
        nargs='*',
        help='Path(s) to .kicad_sch file(s)'
    )
    parser.add_argument(
        '--output-dir', '-o',
        type=str,
        default='./output',
        help='Output directory (default: ./output)'
    )
    parser.add_argument(
        '--layers', '-l',
        type=str,
        help='Specific layers to export (comma-separated)'
    )
    parser.add_argument(
        '--format', '-f',
        type=str,
        choices=['svg', 'pdf', 'gerber', 'all'],
        default='all',
        help='Export format (default: all)'
    )
    parser.add_argument(
        '--png-size',
        type=int,
        default=2000,
        help='PNG conversion size (default: 2000)'
    )
    parser.add_argument(
        '--json',
        action='store_true',
        help='Output results as JSON'
    )

    args = parser.parse_args()

    if not args.pcb and not args.schematic:
        parser.error("Specify --pcb and/or --schematic")

    results = export_full_design(
        pcb_path=args.pcb,
        output_dir=args.output_dir,
        schematic_paths=args.schematic,
        png_size=args.png_size
    )

    if args.json:
        print(json.dumps(results, indent=2))
    else:
        print("\n" + "=" * 60)
        print("EXPORT RESULTS")
        print("=" * 60)
        print(f"PCB: {results['pcb']}")
        print(f"Output: {args.output_dir}")
        print(f"SVG files: {len(results['outputs'].get('svg', {}))}")
        print(f"PNG files: {len(results['outputs'].get('png', {}))}")
        if results['outputs'].get('pcb_pdf'):
            print(f"PCB PDF: {results['outputs']['pcb_pdf']}")
        if results['outputs'].get('gerbers'):
            print(f"Gerbers: {results['outputs']['gerbers']}")


if __name__ == '__main__':
    main()
