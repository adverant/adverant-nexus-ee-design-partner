"""
Artifact Exporter Agent - Exports schematics to PDF/image and saves to NFS.

Provides automated export functionality for the MAPO pipeline:
- PDF export for documentation and review
- PNG/SVG export for web viewing and reports
- NFS share synchronization for terminal computer access

In containerized environments (K8s), exports via HTTP call to mapos-kicad-worker
which has KiCad CLI installed. On local machines (Mac/Linux), uses local KiCad CLI.

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
from urllib.parse import urljoin

# HTTP client for remote export
try:
    import httpx
    HTTPX_AVAILABLE = True
except ImportError:
    HTTPX_AVAILABLE = False

logger = logging.getLogger(__name__)

# Environment-based configuration for K8s deployment
KICAD_WORKER_URL = os.environ.get('KICAD_WORKER_URL', 'http://mapos-kicad-worker:8080')
ARTIFACT_STORAGE_PATH = os.environ.get('ARTIFACT_STORAGE_PATH', '')

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
    # Artifact storage path - uses env var in K8s (points to NFS mount)
    # Falls back to local path when running outside container
    artifact_base_path: str = field(
        default_factory=lambda: ARTIFACT_STORAGE_PATH or str(Path(__file__).parent.parent.parent / "output" / "artifacts")
    )

    # Remote KiCad worker URL (for containerized environments without local KiCad)
    kicad_worker_url: str = field(
        default_factory=lambda: KICAD_WORKER_URL
    )

    # Whether to prefer remote export (auto-detected based on local KiCad availability)
    prefer_remote_export: bool = False

    # Artifact subdirectories
    schematics_dir: str = "schematics"
    pdf_dir: str = "pdf"
    svg_dir: str = "svg"
    images_dir: str = "png"
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
        # Ensure artifact base path exists
        Path(self.artifact_base_path).mkdir(parents=True, exist_ok=True)


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

    In containerized environments:
    - Calls mapos-kicad-worker HTTP API for PDF/SVG/PNG export
    - Saves directly to NFS-mounted artifact storage path

    On local machines (Mac/Linux with KiCad):
    - Uses local KiCad CLI for exports
    - Saves to local artifact directory

    Handles:
    - PDF export for documentation and review
    - SVG/PNG image export for web viewing
    - Direct NFS storage in K8s (ARTIFACT_STORAGE_PATH mount)
    """

    def __init__(self, config: Optional[ArtifactConfig] = None):
        """Initialize the artifact exporter."""
        self.config = config or ArtifactConfig()
        self._kicad_cli = self._find_kicad_cli()
        self._http_client: Optional['httpx.AsyncClient'] = None

        if self._kicad_cli:
            logger.info(f"KiCad CLI found: {self._kicad_cli}")
            self.config.prefer_remote_export = False
        else:
            logger.info("KiCad CLI not found - will use remote export via kicad-worker")
            self.config.prefer_remote_export = True

        logger.info(f"Artifact storage path: {self.config.artifact_base_path}")

    async def _get_http_client(self) -> 'httpx.AsyncClient':
        """Get or create HTTP client for remote export."""
        if not HTTPX_AVAILABLE:
            raise RuntimeError("httpx not installed - cannot perform remote export")
        if self._http_client is None:
            self._http_client = httpx.AsyncClient(timeout=120.0)
        return self._http_client

    async def close(self):
        """Close HTTP client."""
        if self._http_client:
            await self._http_client.aclose()
            self._http_client = None

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
        schematic_path: Optional[Path] = None,
        schematic_content: Optional[str] = None,
        project_id: Optional[str] = None,
        design_name: Optional[str] = None,
    ) -> ExportResult:
        """
        Export schematic to all configured formats.

        In K8s: Saves directly to NFS-mounted artifact storage.
        On local: Saves to local output directory.

        Args:
            schematic_path: Path to the .kicad_sch file (for local export)
            schematic_content: Raw schematic content string (for remote export)
            project_id: Optional project ID for organization
            design_name: Optional design name for file naming

        Returns:
            ExportResult with paths to all exported files
        """
        result = ExportResult(success=False, schematic_path=schematic_path)

        # For remote export, we need content; for local we need path
        if self.config.prefer_remote_export and not schematic_content:
            if schematic_path and schematic_path.exists():
                schematic_content = schematic_path.read_text(encoding='utf-8')
            else:
                result.errors.append("Schematic content required for remote export")
                return result
        elif not self.config.prefer_remote_export and (not schematic_path or not schematic_path.exists()):
            result.errors.append(f"Schematic file not found: {schematic_path}")
            return result

        # Update config with provided values
        if project_id:
            self.config.project_id = project_id
        if design_name:
            self.config.design_name = design_name

        # Create output directories (this is the NFS path in K8s)
        output_base = self._get_output_base()
        output_base.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name = f"{self.config.design_name}_{timestamp}"

        logger.info(f"Exporting artifacts using {'remote' if self.config.prefer_remote_export else 'local'} export")
        logger.info(f"Output base (NFS in K8s): {output_base}")

        # Export PDF
        if self.config.export_pdf:
            if self.config.prefer_remote_export:
                pdf_result = await self._export_pdf_remote(schematic_content, output_base, base_name)
            else:
                pdf_result = await self._export_pdf(schematic_path, output_base, base_name)
            if pdf_result:
                result.pdf_path = pdf_result
                result.nfs_paths["pdf"] = str(pdf_result)
                logger.info(f"PDF exported: {pdf_result}")
            else:
                result.errors.append("PDF export failed")

        # Export SVG
        if self.config.export_svg:
            if self.config.prefer_remote_export:
                svg_result = await self._export_svg_remote(schematic_content, output_base, base_name)
            else:
                svg_result = await self._export_svg(schematic_path, output_base, base_name)
            if svg_result:
                result.svg_path = svg_result
                result.nfs_paths["svg"] = str(svg_result)
                logger.info(f"SVG exported: {svg_result}")
            else:
                result.errors.append("SVG export failed")

        # Export PNG
        if self.config.export_png:
            if self.config.prefer_remote_export:
                png_result = await self._export_png_remote(schematic_content, output_base, base_name)
            else:
                png_result = await self._export_png(
                    schematic_path, output_base, base_name, result.svg_path
                )
            if png_result:
                result.png_path = png_result
                result.nfs_paths["png"] = str(png_result)
                logger.info(f"PNG exported: {png_result}")
            else:
                result.errors.append("PNG export failed")

        # Copy schematic to output directory
        sch_dir = output_base / self.config.schematics_dir
        sch_dir.mkdir(parents=True, exist_ok=True)
        sch_output = sch_dir / f"{base_name}.kicad_sch"
        try:
            if schematic_content:
                sch_output.write_text(schematic_content, encoding='utf-8')
            elif schematic_path:
                shutil.copy2(schematic_path, sch_output)
            result.schematic_path = sch_output
            result.nfs_paths["schematic"] = str(sch_output)
            logger.info(f"Schematic saved: {sch_output}")
        except Exception as e:
            result.errors.append(f"Failed to save schematic: {e}")

        # In K8s, we're already writing to NFS mount, so mark as synced
        result.nfs_synced = True

        # Write manifest file
        await self._write_manifest(output_base, result)

        # Determine overall success
        result.success = (
            result.schematic_path is not None and
            (result.pdf_path is not None or not self.config.export_pdf) and
            (result.svg_path is not None or not self.config.export_svg)
        )

        return result

    def _get_output_base(self) -> Path:
        """Get the base output directory (NFS mount in K8s, local otherwise)."""
        base = Path(self.config.artifact_base_path)
        if self.config.project_id:
            return base / self.config.project_id
        return base

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

    async def _export_pdf_remote(
        self,
        schematic_content: str,
        output_base: Path,
        base_name: str
    ) -> Optional[Path]:
        """Export schematic to PDF via remote kicad-worker API."""
        pdf_dir = output_base / self.config.pdf_dir
        pdf_dir.mkdir(parents=True, exist_ok=True)
        pdf_path = pdf_dir / f"{base_name}.pdf"

        try:
            client = await self._get_http_client()
            export_url = urljoin(self.config.kicad_worker_url, "/v1/schematic/export")

            logger.info(f"Requesting PDF export from: {export_url}")

            response = await client.post(
                export_url,
                json={
                    "schematic_content": schematic_content,
                    "export_format": "pdf",
                    "design_name": base_name,
                }
            )

            if response.status_code != 200:
                logger.error(f"Remote PDF export failed: {response.status_code} - {response.text}")
                return None

            result = response.json()
            if not result.get("success") and result.get("errors"):
                logger.warning(f"PDF export warnings: {result.get('errors')}")

            # Download the exported file
            download_url = urljoin(self.config.kicad_worker_url, result["download_url"])
            logger.info(f"Downloading PDF from: {download_url}")

            download_response = await client.get(download_url)
            if download_response.status_code != 200:
                logger.error(f"PDF download failed: {download_response.status_code}")
                return None

            pdf_path.write_bytes(download_response.content)
            logger.info(f"Remote PDF export successful: {pdf_path.stat().st_size} bytes")
            return pdf_path

        except Exception as e:
            logger.error(f"Remote PDF export error: {e}")
            return None

    async def _export_svg_remote(
        self,
        schematic_content: str,
        output_base: Path,
        base_name: str
    ) -> Optional[Path]:
        """Export schematic to SVG via remote kicad-worker API."""
        svg_dir = output_base / self.config.svg_dir
        svg_dir.mkdir(parents=True, exist_ok=True)
        svg_path = svg_dir / f"{base_name}.svg"

        try:
            client = await self._get_http_client()
            export_url = urljoin(self.config.kicad_worker_url, "/v1/schematic/export")

            logger.info(f"Requesting SVG export from: {export_url}")

            response = await client.post(
                export_url,
                json={
                    "schematic_content": schematic_content,
                    "export_format": "svg",
                    "design_name": base_name,
                }
            )

            if response.status_code != 200:
                logger.error(f"Remote SVG export failed: {response.status_code} - {response.text}")
                return None

            result = response.json()
            if not result.get("success") and result.get("errors"):
                logger.warning(f"SVG export warnings: {result.get('errors')}")

            # Download the exported file
            download_url = urljoin(self.config.kicad_worker_url, result["download_url"])
            logger.info(f"Downloading SVG from: {download_url}")

            download_response = await client.get(download_url)
            if download_response.status_code != 200:
                logger.error(f"SVG download failed: {download_response.status_code}")
                return None

            svg_path.write_bytes(download_response.content)
            logger.info(f"Remote SVG export successful: {svg_path.stat().st_size} bytes")
            return svg_path

        except Exception as e:
            logger.error(f"Remote SVG export error: {e}")
            return None

    async def _export_png_remote(
        self,
        schematic_content: str,
        output_base: Path,
        base_name: str
    ) -> Optional[Path]:
        """Export schematic to PNG via remote kicad-worker API."""
        png_dir = output_base / self.config.images_dir
        png_dir.mkdir(parents=True, exist_ok=True)
        png_path = png_dir / f"{base_name}.png"

        try:
            client = await self._get_http_client()
            export_url = urljoin(self.config.kicad_worker_url, "/v1/schematic/export")

            logger.info(f"Requesting PNG export from: {export_url}")

            response = await client.post(
                export_url,
                json={
                    "schematic_content": schematic_content,
                    "export_format": "png",
                    "design_name": base_name,
                    "dpi": self.config.png_dpi,
                }
            )

            if response.status_code != 200:
                logger.error(f"Remote PNG export failed: {response.status_code} - {response.text}")
                return None

            result = response.json()
            if not result.get("success") and result.get("errors"):
                logger.warning(f"PNG export warnings: {result.get('errors')}")

            # Download the exported file
            download_url = urljoin(self.config.kicad_worker_url, result["download_url"])
            logger.info(f"Downloading PNG from: {download_url}")

            download_response = await client.get(download_url)
            if download_response.status_code != 200:
                logger.error(f"PNG download failed: {download_response.status_code}")
                return None

            png_path.write_bytes(download_response.content)
            logger.info(f"Remote PNG export successful: {png_path.stat().st_size} bytes")
            return png_path

        except Exception as e:
            logger.error(f"Remote PNG export error: {e}")
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

        svg_dir = output_base / self.config.svg_dir
        svg_dir.mkdir(parents=True, exist_ok=True)
        svg_path = svg_dir / f"{base_name}.svg"

        try:
            cmd = [
                self._kicad_cli,
                "sch", "export", "svg",
                "--output", str(svg_dir),  # KiCad outputs to directory
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
                for svg_file in svg_dir.glob("*.svg"):
                    # Rename to our convention if needed
                    if svg_file.name != f"{base_name}.svg":
                        target = svg_dir / f"{base_name}.svg"
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

    async def _write_manifest(self, output_base: Path, result: ExportResult) -> None:
        """Write manifest file for the export."""
        manifest = {
            "exported_at": result.timestamp,
            "design_name": self.config.design_name,
            "project_id": self.config.project_id,
            "artifacts": result.nfs_paths,
            "source_schematic": str(result.schematic_path) if result.schematic_path else None,
            "export_method": "remote" if self.config.prefer_remote_export else "local",
            "storage_path": str(output_base),
        }

        try:
            manifest_path = output_base / "latest_export.json"
            manifest_path.write_text(json.dumps(manifest, indent=2))
            logger.info(f"Export manifest written: {manifest_path}")
        except Exception as e:
            logger.warning(f"Failed to write manifest: {e}")

    def get_storage_status(self) -> Dict[str, Any]:
        """Get artifact storage status."""
        storage_base = Path(self.config.artifact_base_path)

        return {
            "storage_path": str(storage_base),
            "exists": storage_base.exists(),
            "writable": storage_base.exists() and os.access(storage_base, os.W_OK),
            "kicad_worker_url": self.config.kicad_worker_url,
            "prefer_remote_export": self.config.prefer_remote_export,
            "local_kicad_available": self._kicad_cli is not None,
        }


# CLI entry point
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Export schematic artifacts")
    parser.add_argument("schematic_path", help="Path to .kicad_sch file")
    parser.add_argument("--project-id", help="Project ID for organization")
    parser.add_argument("--design-name", default="schematic", help="Design name")
    parser.add_argument("--output-path", help="Artifact output path (overrides ARTIFACT_STORAGE_PATH)")
    parser.add_argument("--kicad-worker-url", help="Remote KiCad worker URL (overrides KICAD_WORKER_URL)")
    parser.add_argument("--force-remote", action="store_true", help="Force remote export even if local KiCad available")
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

    if args.output_path:
        config.artifact_base_path = args.output_path
    if args.kicad_worker_url:
        config.kicad_worker_url = args.kicad_worker_url
    if args.force_remote:
        config.prefer_remote_export = True

    async def main():
        exporter = ArtifactExporterAgent(config)
        try:
            result = await exporter.export_all(
                schematic_path=Path(args.schematic_path),
                project_id=args.project_id,
                design_name=args.design_name,
            )
            print(json.dumps(result.to_dict(), indent=2))
            return 0 if result.success else 1
        finally:
            await exporter.close()

    sys.exit(asyncio.run(main()))
