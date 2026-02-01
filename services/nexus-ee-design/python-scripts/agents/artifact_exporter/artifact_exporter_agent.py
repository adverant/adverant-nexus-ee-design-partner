"""
Artifact Exporter Agent - Exports schematics to PDF/image and saves to NFS.

Provides automated export functionality for the MAPO pipeline:
- PDF export for documentation and review
- PNG/SVG export for web viewing and reports
- NFS share synchronization for terminal computer access

Uses KiCad CLI for native format exports.

Author: Nexus EE Design Team
"""

import asyncio
import json
import logging
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# KiCad CLI paths by platform
if sys.platform == 'darwin':
    KICAD_CLI = '/Applications/KiCad/KiCad.app/Contents/MacOS/kicad-cli'
elif sys.platform == 'win32':
    KICAD_CLI = r'C:\Program Files\KiCad\8.0\bin\kicad-cli.exe'
else:  # Linux/Docker
    KICAD_CLI = '/usr/bin/kicad-cli'


@dataclass
class ArtifactConfig:
    """Configuration for artifact export and storage."""
    # Base NFS share path (for terminal computer)
    nfs_base_path: str = "/Volumes/Nexus/plugins/ee-design-plugin/artifacts"

    # Local fallback path (when NFS not available)
    local_base_path: str = ""

    # Artifact subdirectories
    schematics_dir: str = "schematics"
    pdf_dir: str = "pdf"
    images_dir: str = "images"
    validation_dir: str = "validation-reports"

    # Export settings
    export_pdf: bool = True
    export_svg: bool = True
    export_png: bool = True
    png_dpi: int = 300

    # Project identification
    project_id: Optional[str] = None
    design_name: str = "schematic"

    def __post_init__(self):
        # Set local fallback to output directory if not specified
        if not self.local_base_path:
            self.local_base_path = str(Path(__file__).parent.parent.parent / "output" / "artifacts")


@dataclass
class ExportResult:
    """Result from artifact export operation."""
    success: bool
    schematic_path: Optional[Path] = None
    pdf_path: Optional[Path] = None
    svg_path: Optional[Path] = None
    png_path: Optional[Path] = None
    nfs_synced: bool = False
    nfs_paths: Dict[str, str] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "schematic_path": str(self.schematic_path) if self.schematic_path else None,
            "pdf_path": str(self.pdf_path) if self.pdf_path else None,
            "svg_path": str(self.svg_path) if self.svg_path else None,
            "png_path": str(self.png_path) if self.png_path else None,
            "nfs_synced": self.nfs_synced,
            "nfs_paths": self.nfs_paths,
            "errors": self.errors,
            "timestamp": self.timestamp,
        }


class ArtifactExporterAgent:
    """
    Agent for exporting schematic artifacts and syncing to NFS.

    Handles:
    - PDF export using KiCad CLI
    - SVG/PNG image export using KiCad CLI
    - Artifact syncing to NFS share for terminal computer
    """

    def __init__(self, config: Optional[ArtifactConfig] = None):
        """Initialize the artifact exporter."""
        self.config = config or ArtifactConfig()
        self._kicad_cli = self._find_kicad_cli()

        if self._kicad_cli:
            logger.info(f"KiCad CLI found: {self._kicad_cli}")
        else:
            logger.warning("KiCad CLI not found - exports will be limited")

    def _find_kicad_cli(self) -> Optional[str]:
        """Find KiCad CLI executable."""
        possible_paths = [
            KICAD_CLI,
            "/Applications/KiCad/KiCad.app/Contents/MacOS/kicad-cli",
            "/usr/local/bin/kicad-cli",
            "/usr/bin/kicad-cli",
            "kicad-cli",
        ]

        for path in possible_paths:
            if Path(path).exists():
                return path
            # Check if it's in PATH
            try:
                result = subprocess.run(
                    ["which", path],
                    capture_output=True,
                    text=True,
                    timeout=5
                )
                if result.returncode == 0 and result.stdout.strip():
                    return result.stdout.strip()
            except Exception:
                pass

        return None

    async def export_all(
        self,
        schematic_path: Path,
        project_id: Optional[str] = None,
        design_name: Optional[str] = None,
    ) -> ExportResult:
        """
        Export schematic to all configured formats and sync to NFS.

        Args:
            schematic_path: Path to the .kicad_sch file
            project_id: Optional project ID for organization
            design_name: Optional design name for file naming

        Returns:
            ExportResult with paths to all exported files
        """
        result = ExportResult(success=False, schematic_path=schematic_path)

        if not schematic_path.exists():
            result.errors.append(f"Schematic file not found: {schematic_path}")
            return result

        # Update config with provided values
        if project_id:
            self.config.project_id = project_id
        if design_name:
            self.config.design_name = design_name

        # Create output directories
        output_base = self._get_output_base()
        output_base.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name = f"{self.config.design_name}_{timestamp}"

        logger.info(f"Exporting artifacts for: {schematic_path}")
        logger.info(f"Output base: {output_base}")

        # Export PDF
        if self.config.export_pdf:
            pdf_result = await self._export_pdf(schematic_path, output_base, base_name)
            if pdf_result:
                result.pdf_path = pdf_result
                logger.info(f"PDF exported: {pdf_result}")
            else:
                result.errors.append("PDF export failed")

        # Export SVG
        if self.config.export_svg:
            svg_result = await self._export_svg(schematic_path, output_base, base_name)
            if svg_result:
                result.svg_path = svg_result
                logger.info(f"SVG exported: {svg_result}")
            else:
                result.errors.append("SVG export failed")

        # Export PNG (convert from SVG or use KiCad directly)
        if self.config.export_png:
            png_result = await self._export_png(
                schematic_path, output_base, base_name, result.svg_path
            )
            if png_result:
                result.png_path = png_result
                logger.info(f"PNG exported: {png_result}")
            else:
                result.errors.append("PNG export failed")

        # Copy schematic to output directory
        sch_output = output_base / f"{base_name}.kicad_sch"
        try:
            shutil.copy2(schematic_path, sch_output)
            result.schematic_path = sch_output
            logger.info(f"Schematic copied: {sch_output}")
        except Exception as e:
            result.errors.append(f"Failed to copy schematic: {e}")

        # Sync to NFS share
        nfs_result = await self._sync_to_nfs(result)
        result.nfs_synced = nfs_result

        # Determine overall success
        result.success = (
            result.schematic_path is not None and
            (result.pdf_path is not None or not self.config.export_pdf) and
            (result.svg_path is not None or not self.config.export_svg)
        )

        return result

    def _get_output_base(self) -> Path:
        """Get the base output directory."""
        if self.config.project_id:
            return Path(self.config.local_base_path) / self.config.project_id
        return Path(self.config.local_base_path)

    async def _export_pdf(
        self,
        schematic_path: Path,
        output_base: Path,
        base_name: str
    ) -> Optional[Path]:
        """Export schematic to PDF using KiCad CLI."""
        if not self._kicad_cli:
            logger.warning("KiCad CLI not available for PDF export")
            return None

        pdf_dir = output_base / self.config.pdf_dir
        pdf_dir.mkdir(parents=True, exist_ok=True)
        pdf_path = pdf_dir / f"{base_name}.pdf"

        try:
            cmd = [
                self._kicad_cli,
                "sch", "export", "pdf",
                "--output", str(pdf_path),
                str(schematic_path)
            ]

            logger.debug(f"Running: {' '.join(cmd)}")

            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await proc.communicate()

            if pdf_path.exists() and pdf_path.stat().st_size > 0:
                logger.info(f"PDF export successful: {pdf_path.stat().st_size} bytes")
                return pdf_path

            if proc.returncode != 0:
                logger.error(f"PDF export failed: {stderr.decode()}")
                return None

            logger.warning("PDF export completed but file not found")
            return None

        except Exception as e:
            logger.error(f"PDF export error: {e}")
            return None

    async def _export_svg(
        self,
        schematic_path: Path,
        output_base: Path,
        base_name: str
    ) -> Optional[Path]:
        """Export schematic to SVG using KiCad CLI."""
        if not self._kicad_cli:
            logger.warning("KiCad CLI not available for SVG export")
            return None

        images_dir = output_base / self.config.images_dir
        images_dir.mkdir(parents=True, exist_ok=True)
        svg_path = images_dir / f"{base_name}.svg"

        try:
            cmd = [
                self._kicad_cli,
                "sch", "export", "svg",
                "--output", str(images_dir),  # KiCad outputs to directory
                str(schematic_path)
            ]

            logger.debug(f"Running: {' '.join(cmd)}")

            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await proc.communicate()

            # KiCad creates SVG with schematic name, find it
            if proc.returncode == 0:
                # Look for generated SVG file
                for svg_file in images_dir.glob("*.svg"):
                    # Rename to our convention if needed
                    if svg_file.name != f"{base_name}.svg":
                        target = images_dir / f"{base_name}.svg"
                        shutil.move(str(svg_file), str(target))
                        return target
                    return svg_file

            logger.error(f"SVG export failed: {stderr.decode()}")
            return None

        except Exception as e:
            logger.error(f"SVG export error: {e}")
            return None

    async def _export_png(
        self,
        schematic_path: Path,
        output_base: Path,
        base_name: str,
        svg_path: Optional[Path] = None
    ) -> Optional[Path]:
        """Export schematic to PNG (via SVG conversion or KiCad CLI)."""
        images_dir = output_base / self.config.images_dir
        images_dir.mkdir(parents=True, exist_ok=True)
        png_path = images_dir / f"{base_name}.png"

        # Try converting from SVG first (higher quality)
        if svg_path and svg_path.exists():
            try:
                # Try cairosvg first
                import cairosvg
                cairosvg.svg2png(
                    url=str(svg_path),
                    write_to=str(png_path),
                    dpi=self.config.png_dpi
                )
                if png_path.exists():
                    logger.info(f"PNG converted from SVG: {png_path.stat().st_size} bytes")
                    return png_path
            except ImportError:
                logger.debug("cairosvg not available, trying alternative methods")
            except Exception as e:
                logger.debug(f"cairosvg conversion failed: {e}")

        # Try Inkscape as fallback
        if svg_path and svg_path.exists():
            try:
                inkscape_paths = [
                    "/Applications/Inkscape.app/Contents/MacOS/inkscape",
                    "/usr/bin/inkscape",
                    "inkscape"
                ]
                for inkscape in inkscape_paths:
                    try:
                        proc = await asyncio.create_subprocess_exec(
                            inkscape,
                            "--export-type=png",
                            f"--export-filename={png_path}",
                            f"--export-dpi={self.config.png_dpi}",
                            str(svg_path),
                            stdout=asyncio.subprocess.PIPE,
                            stderr=asyncio.subprocess.PIPE
                        )
                        await proc.communicate()
                        if png_path.exists():
                            logger.info(f"PNG converted via Inkscape: {png_path.stat().st_size} bytes")
                            return png_path
                    except FileNotFoundError:
                        continue
            except Exception as e:
                logger.debug(f"Inkscape conversion failed: {e}")

        # Last resort: use ImageMagick
        if svg_path and svg_path.exists():
            try:
                proc = await asyncio.create_subprocess_exec(
                    "convert",
                    "-density", str(self.config.png_dpi),
                    str(svg_path),
                    str(png_path),
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE
                )
                await proc.communicate()
                if png_path.exists():
                    logger.info(f"PNG converted via ImageMagick: {png_path.stat().st_size} bytes")
                    return png_path
            except Exception as e:
                logger.debug(f"ImageMagick conversion failed: {e}")

        logger.warning("PNG export failed - no conversion method available")
        return None

    async def _sync_to_nfs(self, result: ExportResult) -> bool:
        """Sync exported artifacts to NFS share."""
        nfs_base = Path(self.config.nfs_base_path)

        # Check if NFS is mounted
        if not nfs_base.exists():
            logger.warning(f"NFS share not available: {nfs_base}")
            # Try to create it as a local directory for testing
            try:
                nfs_base.mkdir(parents=True, exist_ok=True)
                logger.info(f"Created local artifacts directory: {nfs_base}")
            except PermissionError:
                logger.warning("Cannot create NFS directory - skipping sync")
                return False

        # Create project subdirectory
        if self.config.project_id:
            nfs_project = nfs_base / self.config.project_id
        else:
            nfs_project = nfs_base / "default"

        nfs_project.mkdir(parents=True, exist_ok=True)

        synced_count = 0

        # Sync schematic
        if result.schematic_path and result.schematic_path.exists():
            try:
                nfs_sch = nfs_project / self.config.schematics_dir
                nfs_sch.mkdir(parents=True, exist_ok=True)
                target = nfs_sch / result.schematic_path.name
                shutil.copy2(result.schematic_path, target)
                result.nfs_paths["schematic"] = str(target)
                synced_count += 1
                logger.info(f"Synced schematic to NFS: {target}")
            except Exception as e:
                logger.error(f"Failed to sync schematic: {e}")

        # Sync PDF
        if result.pdf_path and result.pdf_path.exists():
            try:
                nfs_pdf = nfs_project / self.config.pdf_dir
                nfs_pdf.mkdir(parents=True, exist_ok=True)
                target = nfs_pdf / result.pdf_path.name
                shutil.copy2(result.pdf_path, target)
                result.nfs_paths["pdf"] = str(target)
                synced_count += 1
                logger.info(f"Synced PDF to NFS: {target}")
            except Exception as e:
                logger.error(f"Failed to sync PDF: {e}")

        # Sync SVG
        if result.svg_path and result.svg_path.exists():
            try:
                nfs_images = nfs_project / self.config.images_dir
                nfs_images.mkdir(parents=True, exist_ok=True)
                target = nfs_images / result.svg_path.name
                shutil.copy2(result.svg_path, target)
                result.nfs_paths["svg"] = str(target)
                synced_count += 1
                logger.info(f"Synced SVG to NFS: {target}")
            except Exception as e:
                logger.error(f"Failed to sync SVG: {e}")

        # Sync PNG
        if result.png_path and result.png_path.exists():
            try:
                nfs_images = nfs_project / self.config.images_dir
                nfs_images.mkdir(parents=True, exist_ok=True)
                target = nfs_images / result.png_path.name
                shutil.copy2(result.png_path, target)
                result.nfs_paths["png"] = str(target)
                synced_count += 1
                logger.info(f"Synced PNG to NFS: {target}")
            except Exception as e:
                logger.error(f"Failed to sync PNG: {e}")

        # Create manifest file
        manifest = {
            "exported_at": result.timestamp,
            "design_name": self.config.design_name,
            "project_id": self.config.project_id,
            "artifacts": result.nfs_paths,
            "source_schematic": str(result.schematic_path) if result.schematic_path else None,
        }

        try:
            manifest_path = nfs_project / "latest_export.json"
            manifest_path.write_text(json.dumps(manifest, indent=2))
            logger.info(f"Export manifest written: {manifest_path}")
        except Exception as e:
            logger.warning(f"Failed to write manifest: {e}")

        return synced_count > 0

    def get_nfs_status(self) -> Dict[str, Any]:
        """Get NFS share mount status."""
        nfs_base = Path(self.config.nfs_base_path)

        return {
            "nfs_path": str(nfs_base),
            "mounted": nfs_base.exists(),
            "writable": nfs_base.exists() and os.access(nfs_base, os.W_OK),
            "fallback_path": self.config.local_base_path,
        }


# CLI entry point
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Export schematic artifacts")
    parser.add_argument("schematic_path", help="Path to .kicad_sch file")
    parser.add_argument("--project-id", help="Project ID for organization")
    parser.add_argument("--design-name", default="schematic", help="Design name")
    parser.add_argument("--nfs-path", help="NFS share base path")
    parser.add_argument("--no-pdf", action="store_true", help="Skip PDF export")
    parser.add_argument("--no-svg", action="store_true", help="Skip SVG export")
    parser.add_argument("--no-png", action="store_true", help="Skip PNG export")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    )

    config = ArtifactConfig(
        export_pdf=not args.no_pdf,
        export_svg=not args.no_svg,
        export_png=not args.no_png,
        project_id=args.project_id,
        design_name=args.design_name,
    )

    if args.nfs_path:
        config.nfs_base_path = args.nfs_path

    async def main():
        exporter = ArtifactExporterAgent(config)
        result = await exporter.export_all(
            Path(args.schematic_path),
            project_id=args.project_id,
            design_name=args.design_name,
        )

        print(json.dumps(result.to_dict(), indent=2))
        return 0 if result.success else 1

    sys.exit(asyncio.run(main()))
