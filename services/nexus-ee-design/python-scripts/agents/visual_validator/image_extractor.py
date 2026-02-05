"""
Schematic Image Extractor - Extracts PNG images via kicad-worker for visual validation.

NO FALLBACKS - Only uses kicad-worker HTTP API. Verbose errors on failure.

This extractor is designed for the MAPO visual feedback loop where:
1. Schematic content is sent to kicad-worker
2. PNG image is returned for Opus 4.5 vision analysis
3. Errors are raised with full context for debugging

Author: Nexus EE Design Team
"""

import asyncio
import hashlib
import logging
import os
import tempfile
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import List, Optional
from urllib.parse import urljoin

try:
    import httpx
    HTTPX_AVAILABLE = True
except ImportError:
    HTTPX_AVAILABLE = False

logger = logging.getLogger(__name__)

# Environment configuration
KICAD_WORKER_URL = os.environ.get('KICAD_WORKER_URL', 'http://mapos-kicad-worker:8080')


class ImageExtractionError(Exception):
    """
    Raised when image extraction fails.

    Contains detailed error context for debugging.
    """

    def __init__(
        self,
        message: str,
        kicad_worker_url: str,
        http_status: Optional[int] = None,
        response_body: Optional[str] = None,
        schematic_size_bytes: Optional[int] = None,
        errors: Optional[List[str]] = None
    ):
        self.message = message
        self.kicad_worker_url = kicad_worker_url
        self.http_status = http_status
        self.response_body = response_body
        self.schematic_size_bytes = schematic_size_bytes
        self.errors = errors or []

        # Build verbose error message
        full_message = f"""
================================================================================
IMAGE EXTRACTION FAILED
================================================================================
Error: {message}
KiCad Worker URL: {kicad_worker_url}
HTTP Status: {http_status or 'N/A'}
Schematic Size: {schematic_size_bytes or 'N/A'} bytes
Response Body: {response_body[:500] if response_body else 'N/A'}
Additional Errors: {', '.join(errors) if errors else 'None'}

TROUBLESHOOTING:
1. Verify kicad-worker is running: curl {kicad_worker_url}/health
2. Check K8s pod status: kubectl get pods -n nexus | grep kicad
3. View worker logs: kubectl logs -n nexus deployment/mapos-kicad-worker
4. Ensure KICAD_WORKER_URL env var is set correctly
================================================================================
"""
        super().__init__(full_message)


@dataclass
class ImageExtractionResult:
    """Result from schematic image extraction."""
    success: bool
    image_data: Optional[bytes] = None  # Raw PNG bytes for LLM vision
    image_path: Optional[Path] = None  # Path if saved to disk
    image_hash: str = ""  # MD5 hash for change detection
    resolution_dpi: int = 300
    extraction_method: str = "kicad_worker"
    extraction_time_ms: float = 0.0
    schematic_size_bytes: int = 0
    image_size_bytes: int = 0
    errors: List[str] = field(default_factory=list)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> dict:
        return {
            "success": self.success,
            "image_path": str(self.image_path) if self.image_path else None,
            "image_hash": self.image_hash,
            "resolution_dpi": self.resolution_dpi,
            "extraction_method": self.extraction_method,
            "extraction_time_ms": self.extraction_time_ms,
            "schematic_size_bytes": self.schematic_size_bytes,
            "image_size_bytes": self.image_size_bytes,
            "errors": self.errors,
            "timestamp": self.timestamp,
        }


class SchematicImageExtractor:
    """
    Extracts PNG images from KiCad schematics via kicad-worker HTTP API.

    NO FALLBACKS - Only uses kicad-worker. Raises verbose errors on failure.

    Usage:
        extractor = SchematicImageExtractor()
        result = await extractor.extract_png(schematic_content)
        if result.success:
            # result.image_data contains PNG bytes for LLM vision
            opus_response = await analyze_with_opus(result.image_data)
    """

    OPTIMAL_DPI = 300  # Optimal resolution for Opus 4.5 vision analysis
    DEFAULT_TIMEOUT = 120.0  # 2 minutes for large schematics

    def __init__(
        self,
        kicad_worker_url: Optional[str] = None,
        dpi: int = 300,
        timeout: float = 120.0,
        save_to_disk: bool = False,
        output_dir: Optional[Path] = None
    ):
        """
        Initialize the image extractor.

        Args:
            kicad_worker_url: URL of kicad-worker service. Uses env var if not provided.
            dpi: Resolution for PNG export (default 300 for Opus 4.5)
            timeout: HTTP request timeout in seconds
            save_to_disk: Whether to save PNG to disk (useful for debugging)
            output_dir: Directory for saved images (uses temp dir if not provided)
        """
        if not HTTPX_AVAILABLE:
            raise ImportError(
                "httpx is required for image extraction. "
                "Install with: pip install httpx"
            )

        self.kicad_worker_url = kicad_worker_url or KICAD_WORKER_URL
        self.dpi = dpi
        self.timeout = timeout
        self.save_to_disk = save_to_disk
        self.output_dir = output_dir or Path(tempfile.gettempdir()) / "mapo_images"
        self._http_client: Optional[httpx.AsyncClient] = None

        logger.info(
            f"SchematicImageExtractor initialized: "
            f"kicad_worker={self.kicad_worker_url}, dpi={self.dpi}"
        )

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client."""
        if self._http_client is None:
            self._http_client = httpx.AsyncClient(timeout=self.timeout)
        return self._http_client

    async def close(self):
        """Close HTTP client."""
        if self._http_client:
            await self._http_client.aclose()
            self._http_client = None

    async def extract_png(
        self,
        schematic_content: str,
        design_name: str = "schematic",
        iteration: Optional[int] = None
    ) -> ImageExtractionResult:
        """
        Extract PNG image from schematic content via kicad-worker.

        Args:
            schematic_content: Raw .kicad_sch S-expression content
            design_name: Name for the exported file
            iteration: Optional iteration number (for logging/naming)

        Returns:
            ImageExtractionResult with PNG bytes

        Raises:
            ImageExtractionError: If extraction fails (no fallback)
        """
        start_time = asyncio.get_event_loop().time()
        schematic_size = len(schematic_content.encode('utf-8'))

        iter_suffix = f"_iter{iteration}" if iteration is not None else ""
        export_name = f"{design_name}{iter_suffix}"

        logger.info(
            f"Extracting PNG: design={export_name}, "
            f"schematic_size={schematic_size} bytes, dpi={self.dpi}"
        )

        try:
            client = await self._get_client()
            export_url = urljoin(self.kicad_worker_url, "/v1/schematic/export")

            # Request PNG export from kicad-worker
            response = await client.post(
                export_url,
                json={
                    "schematic_content": schematic_content,
                    "export_format": "png",
                    "design_name": export_name,
                    "dpi": self.dpi,
                }
            )

            if response.status_code != 200:
                raise ImageExtractionError(
                    message=f"kicad-worker returned HTTP {response.status_code}",
                    kicad_worker_url=self.kicad_worker_url,
                    http_status=response.status_code,
                    response_body=response.text,
                    schematic_size_bytes=schematic_size
                )

            result_json = response.json()

            if not result_json.get("success", True):
                errors = result_json.get("errors", [])
                raise ImageExtractionError(
                    message="kicad-worker export failed",
                    kicad_worker_url=self.kicad_worker_url,
                    http_status=200,
                    response_body=str(result_json),
                    schematic_size_bytes=schematic_size,
                    errors=errors
                )

            # Download the exported PNG
            download_url = result_json.get("download_url")
            if not download_url:
                raise ImageExtractionError(
                    message="No download_url in kicad-worker response",
                    kicad_worker_url=self.kicad_worker_url,
                    http_status=200,
                    response_body=str(result_json),
                    schematic_size_bytes=schematic_size
                )

            full_download_url = urljoin(self.kicad_worker_url, download_url)
            logger.debug(f"Downloading PNG from: {full_download_url}")

            download_response = await client.get(full_download_url)

            if download_response.status_code != 200:
                raise ImageExtractionError(
                    message=f"PNG download failed with HTTP {download_response.status_code}",
                    kicad_worker_url=self.kicad_worker_url,
                    http_status=download_response.status_code,
                    response_body=download_response.text[:500],
                    schematic_size_bytes=schematic_size
                )

            image_data = download_response.content
            image_size = len(image_data)

            if image_size == 0:
                raise ImageExtractionError(
                    message="Downloaded PNG is empty (0 bytes)",
                    kicad_worker_url=self.kicad_worker_url,
                    http_status=200,
                    schematic_size_bytes=schematic_size
                )

            # Compute hash for change detection
            image_hash = hashlib.md5(image_data).hexdigest()

            # Calculate extraction time
            extraction_time = (asyncio.get_event_loop().time() - start_time) * 1000

            # Optionally save to disk for debugging
            image_path = None
            if self.save_to_disk:
                self.output_dir.mkdir(parents=True, exist_ok=True)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                image_path = self.output_dir / f"{export_name}_{timestamp}.png"
                image_path.write_bytes(image_data)
                logger.info(f"PNG saved to: {image_path}")

            logger.info(
                f"PNG extraction successful: "
                f"size={image_size} bytes, hash={image_hash[:8]}, "
                f"time={extraction_time:.0f}ms"
            )

            return ImageExtractionResult(
                success=True,
                image_data=image_data,
                image_path=image_path,
                image_hash=image_hash,
                resolution_dpi=self.dpi,
                extraction_method="kicad_worker",
                extraction_time_ms=extraction_time,
                schematic_size_bytes=schematic_size,
                image_size_bytes=image_size,
            )

        except ImageExtractionError:
            raise  # Re-raise our custom errors as-is
        except httpx.ConnectError as e:
            raise ImageExtractionError(
                message=f"Cannot connect to kicad-worker: {e}",
                kicad_worker_url=self.kicad_worker_url,
                schematic_size_bytes=schematic_size,
                errors=["Connection refused - is kicad-worker running?"]
            )
        except httpx.TimeoutException as e:
            raise ImageExtractionError(
                message=f"kicad-worker request timed out after {self.timeout}s: {e}",
                kicad_worker_url=self.kicad_worker_url,
                schematic_size_bytes=schematic_size,
                errors=[f"Timeout exceeded ({self.timeout}s) - schematic may be too large"]
            )
        except Exception as e:
            raise ImageExtractionError(
                message=f"Unexpected error during image extraction: {type(e).__name__}: {e}",
                kicad_worker_url=self.kicad_worker_url,
                schematic_size_bytes=schematic_size,
                errors=[str(e)]
            )

    async def extract_for_comparison(
        self,
        schematic_content: str,
        previous_hash: Optional[str] = None,
        design_name: str = "schematic",
        iteration: Optional[int] = None
    ) -> ImageExtractionResult:
        """
        Extract PNG and check if schematic visually changed.

        Useful for detecting visual stagnation in the validation loop.

        Args:
            schematic_content: Raw .kicad_sch content
            previous_hash: MD5 hash from previous extraction
            design_name: Name for export
            iteration: Current iteration number

        Returns:
            ImageExtractionResult with visual_changed flag in errors
        """
        result = await self.extract_png(schematic_content, design_name, iteration)

        if previous_hash and result.image_hash == previous_hash:
            result.errors.append("VISUAL_UNCHANGED: Image hash matches previous iteration")
            logger.warning(
                f"Schematic visually unchanged between iterations "
                f"(hash={result.image_hash[:8]})"
            )

        return result

    async def health_check(self) -> bool:
        """
        Check if kicad-worker is accessible.

        Returns:
            True if healthy, raises ImageExtractionError otherwise
        """
        try:
            client = await self._get_client()
            health_url = urljoin(self.kicad_worker_url, "/health")
            response = await client.get(health_url, timeout=10.0)

            if response.status_code == 200:
                logger.info(f"kicad-worker health check passed: {self.kicad_worker_url}")
                return True

            raise ImageExtractionError(
                message=f"Health check failed with HTTP {response.status_code}",
                kicad_worker_url=self.kicad_worker_url,
                http_status=response.status_code,
                response_body=response.text
            )
        except httpx.ConnectError as e:
            raise ImageExtractionError(
                message=f"Cannot connect to kicad-worker for health check: {e}",
                kicad_worker_url=self.kicad_worker_url,
                errors=["Connection refused - kicad-worker may not be running"]
            )
