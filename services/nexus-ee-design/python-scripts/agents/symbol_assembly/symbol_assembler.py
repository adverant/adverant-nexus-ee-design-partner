"""
Symbol Assembly Agent - Pre-Generation Symbol, Datasheet & Characterization Gathering.

Runs BEFORE schematic generation to gather all required symbols, datasheets,
and component documentation. Uses GraphRAG-first search strategy with
WebSocket progress streaming identical to schematic generation.

Search order (GraphRAG FIRST):
    1. Nexus Memory/GraphRAG
    2. KiCad Worker (kicad-worker API)
    3. SnapEDA API
    4. UltraLibrarian API
    5. Datasheet sources (DigiKey, Mouser, LCSC)
    6. Opus 4.6 LLM Generation (last resort)

All content extraction uses Opus 4.6 via OpenRouter - NEVER regex.
No fallbacks - verbose error logging only.

Author: Nexus EE Design Team
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import sys
import traceback
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import quote_plus

import httpx

# Add parent paths for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from progress_emitter import ProgressEmitter

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Environment configuration
# ---------------------------------------------------------------------------

OPENROUTER_API_KEY: str = os.environ.get("OPENROUTER_API_KEY", "")
OPENROUTER_BASE_URL: str = "https://openrouter.ai/api/v1"
OPENROUTER_MODEL: str = "anthropic/claude-opus-4.6"

# LLM routing: "claude_code_max" routes Anthropic models through the proxy,
# any other value (or empty) uses OpenRouter for everything.
LLM_ANTHROPIC_ROUTE: str = os.environ.get("LLM_ANTHROPIC_ROUTE", "claude_code_max")
LLM_CLAUDE_CODE_PROXY_URL: str = os.environ.get(
    "LLM_CLAUDE_CODE_PROXY_URL",
    "http://claude-code-proxy.nexus.svc.cluster.local:3100",
)
LLM_CLAUDE_CODE_PROXY_TIMEOUT: int = int(
    os.environ.get("LLM_CLAUDE_CODE_PROXY_TIMEOUT_SECONDS", "600")
)

# Map OpenRouter model IDs to native Anthropic model IDs for the proxy
OPENROUTER_TO_ANTHROPIC_MODEL_ID: dict[str, str] = {
    "anthropic/claude-opus-4.6": "claude-opus-4-6",
    "anthropic/claude-opus-4-6-20260206": "claude-opus-4-6",
    "anthropic/claude-sonnet-4-5": "claude-sonnet-4-5",
    "anthropic/claude-sonnet-4": "claude-sonnet-4",
    "anthropic/claude-3.5-sonnet": "claude-3-5-sonnet-20241022",
    "anthropic/claude-3.5-haiku": "claude-3-5-haiku-20241022",
}

NEXUS_API_KEY: str = os.environ.get("NEXUS_API_KEY", "")
NEXUS_API_URL: str = os.environ.get("NEXUS_API_URL", "https://api.adverant.ai")

# Internal GraphRAG service (direct cluster communication, no auth required)
GRAPHRAG_INTERNAL_URL: str = os.environ.get(
    "GRAPHRAG_INTERNAL_URL",
    os.environ.get(
        "NEXUS_GRAPHRAG_URL",
        "http://nexus-graphrag.nexus.svc.cluster.local:8090",
    ),
)
GRAPHRAG_COMPANY_ID: str = os.environ.get("GRAPHRAG_COMPANY_ID", "adverant")
GRAPHRAG_APP_ID: str = os.environ.get("GRAPHRAG_APP_ID", "ee-design")

KICAD_WORKER_URL: str = os.environ.get(
    "KICAD_WORKER_URL", "http://mapos-kicad-worker:8080"
)

ARTIFACT_STORAGE_PATH: str = os.environ.get("ARTIFACT_STORAGE_PATH", "/data/artifacts")

# JSON parsing retry/repair constants
MAX_JSON_PARSE_RETRIES: int = 2        # Repair retries per chunk (3 total attempts)
JSON_RETRY_BASE_DELAY: float = 2.0     # Exponential backoff base (seconds)
JSON_REPAIR_CONTEXT_CHARS: int = 2000  # Failed response chars to include in repair prompt

SNAPEDA_API_KEY: str = os.environ.get("SNAPEDA_API_KEY", "")
ULTRALIBRARIAN_API_KEY: str = os.environ.get("ULTRALIBRARIAN_API_KEY", "")


# ---------------------------------------------------------------------------
# Assembly phases for progress streaming
# ---------------------------------------------------------------------------


class AssemblyPhase(str, Enum):
    """Phases in symbol assembly (for progress streaming)."""

    ANALYZE = "analyze"
    GRAPHRAG = "graphrag"
    KICAD = "kicad"
    SNAPEDA = "snapeda"
    ULTRALIBRARIAN = "ultralibrarian"
    DATASHEET = "datasheet"
    LLM_GENERATE = "llm_generate"
    INGEST = "ingest"


# Phase progress ranges for overall progress calculation
ASSEMBLY_PHASE_RANGES: Dict[AssemblyPhase, Tuple[int, int]] = {
    AssemblyPhase.ANALYZE: (0, 10),
    AssemblyPhase.GRAPHRAG: (10, 30),
    AssemblyPhase.KICAD: (30, 45),
    AssemblyPhase.SNAPEDA: (45, 55),
    AssemblyPhase.ULTRALIBRARIAN: (55, 65),
    AssemblyPhase.DATASHEET: (65, 75),
    AssemblyPhase.LLM_GENERATE: (75, 90),
    AssemblyPhase.INGEST: (90, 100),
}


class AssemblyEventType(str, Enum):
    """Event types for symbol assembly (matches schematic generation pattern)."""

    PHASE_START = "phase_start"
    PHASE_COMPLETE = "phase_complete"
    SYMBOL_ASSEMBLY_START = "symbol_assembly_start"
    SYMBOL_SEARCH = "symbol_search"
    SYMBOL_FOUND = "symbol_found"
    SYMBOL_GENERATED = "symbol_generated"
    DATASHEET_DOWNLOADED = "datasheet_downloaded"
    CHARACTERIZATION_CREATED = "characterization_created"
    SYMBOL_ASSEMBLY_COMPLETE = "symbol_assembly_complete"
    ERROR = "error"


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class ComponentRequirement:
    """
    A component requirement extracted from ideation artifacts by Opus 4.6.

    Represents a single electronic component that needs its KiCad symbol
    gathered, its datasheet downloaded, and its characterization documented
    before schematic generation can begin.

    Attributes:
        part_number: Manufacturer part number (e.g. ``STM32G431CBT6``).
        manufacturer: Component manufacturer name (e.g. ``ST``).
        package: Physical package type (e.g. ``LQFP-48``, ``0603``).
        category: Component category such as ``MCU``, ``Resistor``, ``MOSFET``.
        value: Nominal value string (e.g. ``100nF``, ``10k``).
        description: Brief description of the component's role.
        quantity: Number of instances required.  Defaults to ``1``.
        subsystem: Name of the parent subsystem (e.g. ``Power Stage``).
        alternatives: List of alternative part numbers.
    """

    part_number: str
    manufacturer: str
    package: str
    category: str
    value: str = ""
    description: str = ""
    quantity: int = 1
    subsystem: str = ""
    alternatives: List[str] = field(default_factory=list)


@dataclass
class ComponentGatherResult:
    """
    Result of gathering symbol/datasheet for a single component.

    Collects the outcome of searching all sources (GraphRAG, KiCad, SnapEDA,
    UltraLibrarian, datasheet providers, LLM generation) for a given component.

    Attributes:
        part_number: The component's part number.
        manufacturer: The component's manufacturer.
        category: Component category.
        symbol_found: Whether a KiCad symbol was successfully obtained.
        symbol_source: Name of the source that provided the symbol.
        symbol_content: The raw KiCad S-expression symbol content.
        symbol_path: Filesystem path where the ``.kicad_sym`` file was written.
        datasheet_found: Whether a datasheet was located.
        datasheet_source: Name of the source that provided the datasheet.
        datasheet_url: URL of the datasheet.
        datasheet_path: Filesystem path where the PDF was saved.
        characterization_created: Whether a characterization doc was generated.
        characterization_path: Filesystem path of the characterization markdown.
        llm_generated: Whether the symbol was generated by Opus 4.6.
        pin_count: Number of pins in the generated/found symbol.
        errors: List of error messages encountered during gathering.
    """

    part_number: str
    manufacturer: str
    category: str

    # Symbol
    symbol_found: bool = False
    symbol_source: str = ""
    symbol_content: str = ""
    symbol_path: Optional[str] = None

    # Datasheet
    datasheet_found: bool = False
    datasheet_source: str = ""
    datasheet_url: str = ""
    datasheet_path: Optional[str] = None

    # Characterization (when no datasheet available)
    characterization_created: bool = False
    characterization_path: Optional[str] = None

    # LLM-generated
    llm_generated: bool = False
    pin_count: int = 0

    # Errors
    errors: List[str] = field(default_factory=list)

    @property
    def success(self) -> bool:
        """True if at least a symbol was found or generated."""
        return self.symbol_found

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary for JSON report."""
        return {
            "part_number": self.part_number,
            "manufacturer": self.manufacturer,
            "category": self.category,
            "symbol_found": self.symbol_found,
            "symbol_source": self.symbol_source,
            "symbol_path": self.symbol_path,
            "datasheet_found": self.datasheet_found,
            "datasheet_source": self.datasheet_source,
            "datasheet_url": self.datasheet_url,
            "datasheet_path": self.datasheet_path,
            "characterization_created": self.characterization_created,
            "characterization_path": self.characterization_path,
            "llm_generated": self.llm_generated,
            "pin_count": self.pin_count,
            "errors": self.errors,
            "success": self.success,
        }


@dataclass
class AssemblyReport:
    """
    Summary report of the entire symbol assembly run.

    Contains aggregate statistics and per-component results for the
    complete symbol gathering operation.

    Attributes:
        project_id: The project this assembly was run for.
        started_at: ISO-8601 timestamp when assembly began.
        completed_at: ISO-8601 timestamp when assembly finished.
        total_components: Total number of components extracted.
        symbols_found: Number of components that have symbols.
        symbols_from_graphrag: Count sourced from GraphRAG memory.
        symbols_from_kicad: Count sourced from KiCad worker.
        symbols_from_snapeda: Count sourced from SnapEDA.
        symbols_from_ultralibrarian: Count sourced from UltraLibrarian.
        symbols_llm_generated: Count generated by Opus 4.6.
        datasheets_downloaded: Count of datasheets saved.
        characterizations_created: Count of characterization docs created.
        errors_count: Total number of component-level errors.
        components: Per-component result dictionaries.
        errors: Aggregate error messages.
    """

    project_id: str
    operation_id: str = ""
    started_at: str = ""
    completed_at: str = ""
    total_components: int = 0
    symbols_found: int = 0
    symbols_from_graphrag: int = 0
    symbols_from_kicad: int = 0
    symbols_from_snapeda: int = 0
    symbols_from_ultralibrarian: int = 0
    symbols_llm_generated: int = 0
    datasheets_downloaded: int = 0
    characterizations_created: int = 0
    status: str = "running"
    errors_count: int = 0
    components: List[Dict[str, Any]] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)

    @property
    def success(self) -> bool:
        """True if all components were successfully gathered."""
        return self.errors_count == 0

    def to_dict(self) -> Dict[str, Any]:
        """Serialize for JSON output."""
        return {
            "project_id": self.project_id,
            "operationId": self.operation_id,
            "status": self.status,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "total_components": self.total_components,
            "symbols_found": self.symbols_found,
            "symbols_from_graphrag": self.symbols_from_graphrag,
            "symbols_from_kicad": self.symbols_from_kicad,
            "symbols_from_snapeda": self.symbols_from_snapeda,
            "symbols_from_ultralibrarian": self.symbols_from_ultralibrarian,
            "symbols_llm_generated": self.symbols_llm_generated,
            "datasheets_downloaded": self.datasheets_downloaded,
            "characterizations_created": self.characterizations_created,
            "errors_count": self.errors_count,
            "components": self.components,
            "errors": self.errors,
            "success": self.success,
        }


# ---------------------------------------------------------------------------
# Assembly progress emitter (wraps ProgressEmitter for assembly-specific use)
# ---------------------------------------------------------------------------


def _calc_overall(phase: AssemblyPhase, phase_progress: int) -> int:
    """
    Calculate overall progress percentage from phase and phase-specific progress.

    Args:
        phase: The current assembly phase.
        phase_progress: Progress within the current phase (0--100).

    Returns:
        Overall progress percentage (0--100).
    """
    start, end = ASSEMBLY_PHASE_RANGES.get(phase, (0, 100))
    return start + int((end - start) * phase_progress / 100)


def emit_progress(
    operation_id: str,
    phase: str,
    message: str,
    percent: int,
    *,
    event_type: str = "",
    **extra_data: Any,
) -> None:
    """
    Emit a ``PROGRESS:{json}`` line to stdout for Node.js WebSocket relay.

    This is the **identical** streaming pattern used by
    ``api_generate_schematic.py`` and ``progress_emitter.py``.

    Args:
        operation_id: Operation identifier for the WebSocket session.
        phase: Current assembly phase name (e.g. ``"graphrag"``).
        message: Human-readable message with tag prefix (e.g. ``"[GRAPHRAG] ..."``).
        percent: Overall progress percentage (0--100).
        event_type: Optional event type override.
        **extra_data: Additional key-value pairs to include in the event payload.
    """
    event: Dict[str, Any] = {
        "type": event_type or f"{phase}_progress",
        "operationId": operation_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "progress_percentage": max(0, min(100, percent)),
        "current_step": message,
        "phase": phase,
    }
    if extra_data:
        event["data"] = extra_data
    print(f"PROGRESS:{json.dumps(event)}", flush=True)


class AssemblyProgressEmitter:
    """
    Wraps the shared ``ProgressEmitter`` to emit symbol-assembly-specific
    events using the ``PROGRESS:{json}`` stdout format identical to
    schematic generation.

    Provides tag-specific convenience methods for each assembly phase:
    ``[ANALYZE]``, ``[GRAPHRAG]``, ``[KICAD]``, ``[SNAPEDA]``,
    ``[ULTRALIBRARIAN]``, ``[DATASHEET]``, ``[LLM]``, ``[ERROR]``,
    ``[INGEST]``.
    """

    def __init__(self, operation_id: str = "") -> None:
        self._emitter = ProgressEmitter(operation_id=operation_id)
        self._operation_id = operation_id

    # -- Human-readable tagged log lines --

    def log_analyze(self, message: str, phase_progress: int = 0) -> None:
        """Emit ``[ANALYZE]`` log line."""
        overall = _calc_overall(AssemblyPhase.ANALYZE, phase_progress)
        self._emitter.emit(
            event_type=AssemblyEventType.SYMBOL_ASSEMBLY_START.value,
            progress=overall,
            message=f"[ANALYZE] {message}",
            phase=AssemblyPhase.ANALYZE.value,
            phase_progress=phase_progress,
        )

    def log_graphrag(self, message: str, phase_progress: int = 0) -> None:
        """Emit ``[GRAPHRAG]`` log line."""
        overall = _calc_overall(AssemblyPhase.GRAPHRAG, phase_progress)
        self._emitter.emit(
            event_type=AssemblyEventType.SYMBOL_SEARCH.value,
            progress=overall,
            message=f"[GRAPHRAG] {message}",
            phase=AssemblyPhase.GRAPHRAG.value,
            phase_progress=phase_progress,
        )

    def log_kicad(self, message: str, phase_progress: int = 0) -> None:
        """Emit ``[KICAD]`` log line."""
        overall = _calc_overall(AssemblyPhase.KICAD, phase_progress)
        self._emitter.emit(
            event_type=AssemblyEventType.SYMBOL_SEARCH.value,
            progress=overall,
            message=f"[KICAD] {message}",
            phase=AssemblyPhase.KICAD.value,
            phase_progress=phase_progress,
        )

    def log_snapeda(self, message: str, phase_progress: int = 0) -> None:
        """Emit ``[SNAPEDA]`` log line."""
        overall = _calc_overall(AssemblyPhase.SNAPEDA, phase_progress)
        self._emitter.emit(
            event_type=AssemblyEventType.SYMBOL_SEARCH.value,
            progress=overall,
            message=f"[SNAPEDA] {message}",
            phase=AssemblyPhase.SNAPEDA.value,
            phase_progress=phase_progress,
        )

    def log_ultralibrarian(self, message: str, phase_progress: int = 0) -> None:
        """Emit ``[ULTRALIBRARIAN]`` log line."""
        overall = _calc_overall(AssemblyPhase.ULTRALIBRARIAN, phase_progress)
        self._emitter.emit(
            event_type=AssemblyEventType.SYMBOL_SEARCH.value,
            progress=overall,
            message=f"[ULTRALIBRARIAN] {message}",
            phase=AssemblyPhase.ULTRALIBRARIAN.value,
            phase_progress=phase_progress,
        )

    def log_datasheet(self, message: str, phase_progress: int = 0) -> None:
        """Emit ``[DATASHEET]`` log line."""
        overall = _calc_overall(AssemblyPhase.DATASHEET, phase_progress)
        self._emitter.emit(
            event_type=AssemblyEventType.DATASHEET_DOWNLOADED.value,
            progress=overall,
            message=f"[DATASHEET] {message}",
            phase=AssemblyPhase.DATASHEET.value,
            phase_progress=phase_progress,
        )

    def log_llm(self, message: str, phase_progress: int = 0) -> None:
        """Emit ``[LLM]`` log line."""
        overall = _calc_overall(AssemblyPhase.LLM_GENERATE, phase_progress)
        self._emitter.emit(
            event_type=AssemblyEventType.SYMBOL_GENERATED.value,
            progress=overall,
            message=f"[LLM] {message}",
            phase=AssemblyPhase.LLM_GENERATE.value,
            phase_progress=phase_progress,
        )

    def log_ingest(self, message: str, phase_progress: int = 0) -> None:
        """Emit ``[INGEST]`` log line."""
        overall = _calc_overall(AssemblyPhase.INGEST, phase_progress)
        self._emitter.emit(
            event_type=AssemblyEventType.SYMBOL_ASSEMBLY_COMPLETE.value,
            progress=overall,
            message=f"[INGEST] {message}",
            phase=AssemblyPhase.INGEST.value,
            phase_progress=phase_progress,
        )

    def log_error(self, message: str) -> None:
        """Emit ``[ERROR]`` log line."""
        self._emitter.error(
            f"[ERROR] {message}", error_code="SYMBOL_ASSEMBLY_FAILED"
        )

    def complete(self, result: Optional[Dict[str, Any]] = None) -> None:
        """Emit completion event."""
        event: Dict[str, Any] = {
            "type": AssemblyEventType.SYMBOL_ASSEMBLY_COMPLETE.value,
            "operationId": self._emitter.operation_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "progress_percentage": 100,
            "current_step": "Symbol assembly complete!",
        }
        if result:
            event["data"] = {"result": result}
        self._emitter._emit_line(event)


# ---------------------------------------------------------------------------
# Main assembler class
# ---------------------------------------------------------------------------


class SymbolAssembler:
    """
    Pre-generation symbol assembly agent.

    Analyzes ideation artifacts using Opus 4.6 to extract components,
    then gathers symbols, datasheets, and characterization documents
    from multiple sources with GraphRAG searched FIRST.

    Emits WebSocket progress events using the identical ``PROGRESS:{json}``
    stdout format as schematic generation.

    Example usage::

        assembler = SymbolAssembler(
            project_id="proj-abc",
            operation_id="op-123",
        )
        report = await assembler.run(
            project_id="proj-abc",
            artifacts=[...],
            operation_id="op-123",
        )

    Search order (GraphRAG FIRST):
        1. Nexus Memory / GraphRAG (``POST /api/memory/recall``)
        2. KiCad Worker API
        3. SnapEDA API
        4. UltraLibrarian API
        5. Datasheet sources (DigiKey / Mouser / LCSC)
        6. Opus 4.6 LLM symbol generation (last resort)
    """

    def __init__(
        self,
        project_id: str = "",
        operation_id: str = "",
        output_base_path: Optional[str] = None,
    ) -> None:
        """
        Initialize the symbol assembler.

        Args:
            project_id: Project identifier for organizing output artifacts.
            operation_id: Operation ID for WebSocket progress streaming.
            output_base_path: Base path for artifact storage.
                Defaults to ``ARTIFACT_STORAGE_PATH`` env var.
        """
        self.project_id = project_id
        self.operation_id = operation_id
        self._output_base = Path(output_base_path or ARTIFACT_STORAGE_PATH)

        # Build output directory structure
        self._output_dir = self._output_base / project_id / "symbol-assembly"
        self._symbols_dir = self._output_dir / "symbols"
        self._datasheets_dir = self._output_dir / "datasheets"
        self._characterizations_dir = self._output_dir / "characterizations"

        # Progress emitter
        self._progress = AssemblyProgressEmitter(operation_id=operation_id)

        # HTTP client (shared, lazily created)
        self._http: Optional[httpx.AsyncClient] = None

    async def _get_http(self) -> httpx.AsyncClient:
        """Get or create the shared async HTTP client."""
        if self._http is None or self._http.is_closed:
            self._http = httpx.AsyncClient(
                timeout=httpx.Timeout(60.0),
                follow_redirects=True,
            )
        return self._http

    async def close(self) -> None:
        """Close the HTTP client and release resources."""
        if self._http and not self._http.is_closed:
            await self._http.aclose()
            self._http = None

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    async def run(
        self,
        project_id: str,
        artifacts: List[Dict[str, Any]],
        operation_id: str,
    ) -> AssemblyReport:
        """
        Run the full symbol assembly pipeline.

        1. Analyzes ideation artifacts with Opus 4.6 to extract components.
        2. Searches each source in order for each component.
        3. Downloads datasheets.
        4. Generates missing symbols with LLM.
        5. Stores files to ``{artifact_storage_path}/{project_id}/symbol-assembly/``.
        6. Ingests to GraphRAG: ``POST https://api.adverant.ai/api/memory/store``.
        7. Returns ``AssemblyReport``.

        Args:
            project_id: Project identifier.
            artifacts: Raw ideation artifact dicts from the API.
            operation_id: Operation ID for WebSocket streaming.

        Returns:
            AssemblyReport with results for every component.
        """
        # Update instance state from call args (allows re-use)
        self.project_id = project_id
        self.operation_id = operation_id
        self._output_dir = self._output_base / project_id / "symbol-assembly"
        self._symbols_dir = self._output_dir / "symbols"
        self._datasheets_dir = self._output_dir / "datasheets"
        self._characterizations_dir = self._output_dir / "characterizations"
        self._progress = AssemblyProgressEmitter(operation_id=operation_id)

        report = AssemblyReport(
            project_id=project_id,
            operation_id=operation_id,
            started_at=datetime.now(timezone.utc).isoformat(),
        )

        try:
            # Ensure output directories exist
            self._symbols_dir.mkdir(parents=True, exist_ok=True)
            self._datasheets_dir.mkdir(parents=True, exist_ok=True)
            self._characterizations_dir.mkdir(parents=True, exist_ok=True)

            # ---- Phase 1: Analyze components ----
            self._progress.log_analyze(
                "Parsing ideation artifacts with Opus 4.6...",
                phase_progress=0,
            )
            components = await self.analyze_components(artifacts)
            report.total_components = len(components)
            self._progress.log_analyze(
                f"Parsing ideation artifacts with Opus 4.6... "
                f"Found {len(components)} components",
                phase_progress=100,
            )

            if not components:
                self._progress.log_error(
                    "No components extracted from ideation artifacts. "
                    "Verify that BOM, component_selection, or schematic_spec "
                    "artifacts are present and contain component data."
                )
                report.errors.append(
                    "No components extracted from ideation artifacts"
                )
                report.errors_count = 1
                report.completed_at = datetime.now(timezone.utc).isoformat()
                return report

            # ---- Phases 2-7: Gather symbols / datasheets ----
            results: List[ComponentGatherResult] = []
            for idx, comp in enumerate(components):
                result = await self._gather_component(comp, idx, len(components))
                results.append(result)

            # Tally results
            for r in results:
                report.components.append(r.to_dict())
                if r.symbol_found:
                    report.symbols_found += 1
                if r.symbol_source == "graphrag":
                    report.symbols_from_graphrag += 1
                elif r.symbol_source == "kicad":
                    report.symbols_from_kicad += 1
                elif r.symbol_source == "snapeda":
                    report.symbols_from_snapeda += 1
                elif r.symbol_source == "ultralibrarian":
                    report.symbols_from_ultralibrarian += 1
                if r.llm_generated:
                    report.symbols_llm_generated += 1
                if r.datasheet_found:
                    report.datasheets_downloaded += 1
                if r.characterization_created:
                    report.characterizations_created += 1
                if r.errors:
                    report.errors_count += 1
                    for err in r.errors:
                        report.errors.append(f"{r.part_number}: {err}")

            # ---- Phase 8: Ingest into GraphRAG ----
            await self._ingest_to_graphrag(results)

            # Write assembly_report.json
            report.status = "complete"
            report.completed_at = datetime.now(timezone.utc).isoformat()
            report_path = self._output_dir / "assembly_report.json"
            report_path.write_text(json.dumps(report.to_dict(), indent=2))
            logger.info(f"Assembly report written to {report_path}")

            # Emit completion
            self._progress.complete(
                {
                    "total_components": report.total_components,
                    "symbols_found": report.symbols_found,
                    "datasheets_downloaded": report.datasheets_downloaded,
                    "characterizations_created": report.characterizations_created,
                    "errors_count": report.errors_count,
                }
            )

            return report

        except Exception as exc:
            tb = traceback.format_exc()
            error_msg = f"Symbol assembly failed: {exc}\n{tb}"
            logger.error(error_msg)
            # Print to stderr so Node route handler captures the actual error
            print(f"ASSEMBLY_ERROR: {exc}", file=sys.stderr)
            self._progress.log_error(error_msg)
            report.errors.append(error_msg)
            report.errors_count += 1
            report.status = "error"
            report.completed_at = datetime.now(timezone.utc).isoformat()
            # Write error report to disk so GET endpoint returns error instead of 404
            try:
                report_path = self._output_dir / "assembly_report.json"
                report_path.parent.mkdir(parents=True, exist_ok=True)
                report_path.write_text(json.dumps(report.to_dict(), indent=2))
                logger.info(f"Error report written to {report_path}")
            except Exception as write_exc:
                logger.error(f"Failed to write error report: {write_exc}")
                print(f"ASSEMBLY_ERROR: Failed to write error report: {write_exc}", file=sys.stderr)
            return report
        finally:
            await self.close()

    # ------------------------------------------------------------------
    # Phase 1: Component analysis (Opus 4.6, NO regex)
    # ------------------------------------------------------------------

    async def analyze_components(
        self,
        artifacts: List[Dict[str, Any]],
    ) -> List[ComponentRequirement]:
        """
        Use Opus 4.6 to extract every electronic component from the supplied
        ideation artifacts. Routes through Claude Code Max proxy or OpenRouter
        based on LLM_ANTHROPIC_ROUTE configuration.

        ALL extraction uses Opus 4.6 -- **NEVER** regex.

        Args:
            artifacts: Raw ideation artifact dicts (each with ``name``,
                ``artifact_type``, ``category``, ``content`` keys).

        Returns:
            De-duplicated list of ``ComponentRequirement`` objects with
            part_number, manufacturer, package, and category populated.

        Raises:
            RuntimeError: If the LLM call fails or returns unparseable output.
        """
        if not self._should_use_proxy() and not OPENROUTER_API_KEY:
            raise RuntimeError(
                "No LLM endpoint available for Opus 4.6 content extraction. "
                "Either set LLM_ANTHROPIC_ROUTE=claude_code_max (to use Claude "
                "Code Max proxy) or provide OPENROUTER_API_KEY."
            )

        # Serialize artifacts into text parts for the LLM
        artifact_text_parts: List[str] = []
        for artifact in artifacts:
            name = artifact.get("name", "untitled")
            artifact_type = artifact.get("artifact_type", "unknown")
            category = artifact.get("category", "other")
            content = artifact.get("content", "")
            if content:
                artifact_text_parts.append(
                    f"--- Artifact: {name} (type={artifact_type}, "
                    f"category={category}) ---\n"
                    f"{content}\n"
                )

        if not artifact_text_parts:
            logger.warning("No artifact content to analyze")
            return []

        # Chunk artifacts into batches to avoid exceeding context window.
        # Each chunk is processed separately, then results are merged.
        # ~80K chars per chunk keeps prompts well within the 200K token limit.
        MAX_CHUNK_CHARS = 80_000
        chunks: List[List[str]] = []
        current_chunk: List[str] = []
        current_size = 0
        for part in artifact_text_parts:
            part_len = len(part)
            if current_chunk and current_size + part_len > MAX_CHUNK_CHARS:
                chunks.append(current_chunk)
                current_chunk = []
                current_size = 0
            current_chunk.append(part)
            current_size += part_len
        if current_chunk:
            chunks.append(current_chunk)

        logger.info(
            f"Split {len(artifact_text_parts)} artifacts into "
            f"{len(chunks)} analysis chunks "
            f"(max {MAX_CHUNK_CHARS} chars/chunk)"
        )

        extraction_prompt_template = (
            "You are an expert electronics engineer. Analyze the following "
            "ideation artifacts from an EE design project and extract EVERY "
            "electronic component referenced. Return a JSON array of objects, "
            "each with these exact fields:\n"
            "\n"
            "- part_number (string): The specific manufacturer part number "
            "(e.g., 'STM32G431CBT6', 'INA240A4PWR'). For generic passives "
            "use the part number if given, otherwise describe the value "
            "(e.g., '100nF_cap').\n"
            "- manufacturer (string): The manufacturer name "
            "(e.g., 'ST', 'TI', 'Infineon'). Empty string if unknown.\n"
            "- package (string): The package type "
            "(e.g., 'LQFP48', 'SOT-23', 'SMD0603'). Empty string if unknown.\n"
            "- category (string): One of: MCU, MOSFET, Gate_Driver, IC, "
            "Amplifier, Capacitor, Resistor, Inductor, Diode, LED, Connector, "
            "Power, Regulator, Crystal, Thermistor, CAN_Transceiver, TVS, "
            "Sensor, Transformer, Relay, Fuse, Other\n"
            "- value (string): The component value "
            "(e.g., '100nF', '10k', '650V/31A SiC'). Empty string if N/A.\n"
            "- description (string): Brief description of the component's "
            "role in the design.\n"
            "- quantity (integer): How many of this component are needed. "
            "Default 1.\n"
            "- subsystem (string): Which subsystem this belongs to "
            "(e.g., 'Power Stage', 'MCU Core'). Empty string if unknown.\n"
            "- alternatives (array of strings): Alternative part numbers if "
            "mentioned. Empty array if none.\n"
            "\n"
            "IMPORTANT RULES:\n"
            "- Include ALL components, even power symbols (VCC, GND) and "
            "passive components.\n"
            "- Do NOT skip any component mentioned anywhere in the artifacts.\n"
            "- If a component appears in multiple artifacts, include it only "
            "once with the most complete data.\n"
            "- Return ONLY the JSON array. Nothing else.\n"
            "- Do NOT include markdown formatting (no ```, no ```json).\n"
            "- Do NOT include any explanation, summary, or commentary.\n"
            "- Do NOT say things like 'The JSON array above contains...' "
            "or 'Here are the components...'.\n"
            "- Your entire response must be valid JSON starting with [ and "
            "ending with ].\n"
            "\n"
            "BAD (will cause a pipeline failure):\n"
            "  Here is the JSON array of components:\n"
            "  ```json\n"
            "  [{...}]\n"
            "  ```\n"
            "  The array above contains 32 entries...\n"
            "\n"
            "GOOD (correct format):\n"
            "  [{\"part_number\": \"STM32G474\", ...}]\n"
            "\n"
            "ARTIFACTS:\n"
        )

        # Process each chunk with retry/repair and collect raw component dicts
        all_raw_components: List[dict] = []
        failed_chunks: List[Tuple[int, str]] = []  # (index, prompt)

        for chunk_idx, chunk_parts in enumerate(chunks):
            chunk_text = "\n".join(chunk_parts)
            chunk_label = f"chunk {chunk_idx + 1}/{len(chunks)}"
            prompt = extraction_prompt_template + chunk_text

            logger.info(
                f"Analyzing {chunk_label}: "
                f"{len(chunk_parts)} artifacts, "
                f"{len(prompt)} chars prompt"
            )
            self._progress.log_analyze(
                f"Parsing ideation artifacts with Opus 4.6... "
                f"({chunk_label})",
                phase_progress=int((chunk_idx / len(chunks)) * 90),
            )

            raw_components = await self._extract_components_with_retry(
                prompt, chunk_label
            )

            if raw_components:
                all_raw_components.extend(raw_components)
                logger.info(
                    f"{chunk_label} yielded {len(raw_components)} components"
                )
            else:
                failed_chunks.append((chunk_idx, prompt))
                logger.warning(
                    f"{chunk_label} yielded 0 components after all retries"
                )

        # Second-pass retry for any chunks that failed completely
        if failed_chunks:
            logger.info(
                f"Retrying {len(failed_chunks)} failed chunk(s) "
                f"(second pass)"
            )
            for chunk_idx, prompt in failed_chunks:
                chunk_label = f"chunk {chunk_idx + 1}/{len(chunks)} (retry)"
                self._progress.log_analyze(
                    f"Retrying failed {chunk_label}...",
                    phase_progress=92,
                )
                raw_components = await self._extract_components_with_retry(
                    prompt, chunk_label
                )
                if raw_components:
                    all_raw_components.extend(raw_components)
                    logger.info(
                        f"{chunk_label} yielded {len(raw_components)} "
                        f"components on second pass"
                    )

        if not all_raw_components:
            raise RuntimeError(
                f"ALL {len(chunks)} chunk(s) failed component extraction "
                f"after retries. Zero components extracted from "
                f"{len(artifacts)} artifacts."
            )

        raw_components = all_raw_components

        components: List[ComponentRequirement] = []
        for raw in raw_components:
            if not isinstance(raw, dict):
                logger.warning(f"Skipping non-dict component entry: {raw}")
                continue
            components.append(
                ComponentRequirement(
                    part_number=str(raw.get("part_number", "")).strip(),
                    manufacturer=str(raw.get("manufacturer", "")).strip(),
                    package=str(raw.get("package", "")).strip(),
                    category=str(raw.get("category", "Other")).strip(),
                    value=str(raw.get("value", "")).strip(),
                    description=str(raw.get("description", "")).strip(),
                    quantity=int(raw.get("quantity", 1)),
                    subsystem=str(raw.get("subsystem", "")).strip(),
                    alternatives=list(raw.get("alternatives", [])),
                )
            )

        # Deduplicate by part_number (keep first occurrence)
        seen: Dict[str, int] = {}
        deduped: List[ComponentRequirement] = []
        for comp in components:
            key = comp.part_number.lower()
            if key not in seen:
                seen[key] = len(deduped)
                deduped.append(comp)

        logger.info(
            f"Extracted {len(deduped)} unique components from "
            f"{len(artifacts)} ideation artifacts"
        )
        return deduped

    # ------------------------------------------------------------------
    # Component extraction with retry + repair
    # ------------------------------------------------------------------

    _COMPONENT_SYSTEM_MESSAGE = (
        "You are a structured data extraction engine. You ONLY output valid "
        "JSON arrays. You never output prose, markdown, explanations, or "
        "commentary. Your entire response is always a single JSON array."
    )

    async def _extract_components_with_retry(
        self, prompt: str, chunk_label: str
    ) -> List[dict]:
        """Call Opus with system message + prefill, parse, retry on failure.

        Orchestrates:
        1. Initial call with system message + assistant prefill '['
        2. Parse response (try with prefill prepended, then without)
        3. On failure: up to MAX_JSON_PARSE_RETRIES repair retries
           with exponential backoff
        4. Returns [] only if all attempts exhausted
        """
        # --- Attempt 1: initial call with system + prefill ---
        response_text = await self._call_opus(
            prompt,
            system_message=self._COMPONENT_SYSTEM_MESSAGE,
            assistant_prefill="[",
        )

        # Try parsing with prefill prepended (model continues from '['),
        # then without (model may have ignored the prefill)
        for candidate in (f"[{response_text}", response_text):
            components = self._parse_component_json(candidate, chunk_label)
            if components:
                return components

        # --- Repair retries ---
        last_response = response_text
        for retry_idx in range(MAX_JSON_PARSE_RETRIES):
            delay = JSON_RETRY_BASE_DELAY * (2 ** retry_idx)
            logger.warning(
                f"JSON parse failed for {chunk_label}. "
                f"Repair retry {retry_idx + 1}/{MAX_JSON_PARSE_RETRIES} "
                f"after {delay}s backoff"
            )
            await asyncio.sleep(delay)

            repair_prompt = self._build_repair_prompt(prompt, last_response)

            self._progress.log_analyze(
                f"Repair retry {retry_idx + 1} for {chunk_label}...",
                phase_progress=91,
            )

            response_text = await self._call_opus(
                repair_prompt,
                system_message=self._COMPONENT_SYSTEM_MESSAGE,
                assistant_prefill="[",
            )

            for candidate in (f"[{response_text}", response_text):
                components = self._parse_component_json(
                    candidate, chunk_label
                )
                if components:
                    logger.info(
                        f"Repair retry {retry_idx + 1} succeeded for "
                        f"{chunk_label}: {len(components)} components"
                    )
                    return components

            last_response = response_text

        logger.error(
            f"All {1 + MAX_JSON_PARSE_RETRIES} attempts failed for "
            f"{chunk_label}"
        )
        return []

    def _build_repair_prompt(
        self, original_prompt: str, failed_response: str
    ) -> str:
        """Construct a repair prompt showing what went wrong."""
        truncated = failed_response[:JSON_REPAIR_CONTEXT_CHARS]
        return (
            "Your previous response could NOT be parsed as a JSON array. "
            "This is what you returned (first "
            f"{JSON_REPAIR_CONTEXT_CHARS} chars):\n\n"
            f"---\n{truncated}\n---\n\n"
            "That is WRONG. You must return ONLY a valid JSON array, "
            "starting with [ and ending with ]. No prose, no markdown "
            "fences, no explanation before or after.\n\n"
            "Here is the original request again. Respond with ONLY the "
            "JSON array:\n\n"
            f"{original_prompt}"
        )

    # ------------------------------------------------------------------
    # JSON parsing helpers for component extraction (multi-strategy)
    # ------------------------------------------------------------------

    def _parse_component_json(
        self, response_text: str, context_label: str
    ) -> List[dict]:
        """Parse LLM response text into a list of component dicts.

        Tries multiple extraction strategies in order of reliability.
        Returns an empty list on failure instead of raising, so the
        retry loop in _extract_components_with_retry can handle escalation.

        Args:
            response_text: Raw LLM response string.
            context_label: Label for error messages (e.g., "chunk 1/3").

        Returns:
            List of raw component dicts (empty on parse failure).
        """
        result, strategy = self._extract_json_array(response_text)
        if result is not None:
            logger.info(
                f"Parsed {len(result)} components from {context_label} "
                f"using strategy: {strategy}"
            )
            return result

        logger.warning(
            f"All JSON extraction strategies failed for {context_label}. "
            f"Response (first 2000 chars): {response_text[:2000]}"
        )
        return []

    def _extract_json_array(
        self, text: str
    ) -> Tuple[Optional[List[dict]], str]:
        """Try multiple strategies to extract a JSON array from text.

        Strategies tried in order:
        1. direct_parse - json.loads on stripped text
        2. markdown_fence - extract from ```json ... ``` fences
        3. bracket_balanced - walk brackets respecting string escapes
        4. greedy_largest - brute-force all bracket ranges

        Returns:
            (list_of_dicts, strategy_name) on success, (None, "") on failure.
        """
        stripped = text.strip()

        # Strategy 1: Direct parse
        try:
            parsed = json.loads(stripped)
            if isinstance(parsed, list):
                return parsed, "direct_parse"
        except (json.JSONDecodeError, ValueError):
            pass

        # Strategy 2: Markdown fence extraction (case-insensitive)
        fence_match = re.search(
            r'```(?:json|JSON)?\s*\n(.*?)```', stripped, re.DOTALL
        )
        if fence_match:
            try:
                parsed = json.loads(fence_match.group(1).strip())
                if isinstance(parsed, list):
                    return parsed, "markdown_fence"
            except (json.JSONDecodeError, ValueError):
                pass

        # Strategy 3: Bracket-balanced extraction
        balanced = self._extract_balanced_array(stripped)
        if balanced is not None:
            return balanced, "bracket_balanced"

        # Strategy 4: Greedy largest valid array
        largest = self._extract_largest_valid_array(stripped)
        if largest is not None:
            return largest, "greedy_largest"

        return None, ""

    def _extract_balanced_array(self, text: str) -> Optional[List[dict]]:
        """Extract a JSON array using bracket-depth walking.

        Walks from the first '[' tracking depth while respecting string
        literals and escape sequences. Returns the parsed array or None.
        """
        start = text.find('[')
        if start == -1:
            return None

        depth = 0
        in_string = False
        escape_next = False
        i = start

        while i < len(text):
            ch = text[i]

            if escape_next:
                escape_next = False
                i += 1
                continue

            if ch == '\\' and in_string:
                escape_next = True
                i += 1
                continue

            if ch == '"':
                in_string = not in_string
            elif not in_string:
                if ch == '[':
                    depth += 1
                elif ch == ']':
                    depth -= 1
                    if depth == 0:
                        candidate = text[start:i + 1]
                        try:
                            parsed = json.loads(candidate)
                            if isinstance(parsed, list):
                                return parsed
                        except (json.JSONDecodeError, ValueError):
                            pass
                        # This balanced segment didn't parse; keep looking
                        # for the next '[' after this position
                        next_start = text.find('[', i + 1)
                        if next_start == -1:
                            return None
                        start = next_start
                        i = next_start
                        depth = 0
                        continue

            i += 1

        return None

    def _extract_largest_valid_array(
        self, text: str
    ) -> Optional[List[dict]]:
        """Brute-force fallback: find all [ and ] positions, try parsing
        each range from outermost to innermost, keep the largest valid list.
        """
        open_positions = [i for i, c in enumerate(text) if c == '[']
        close_positions = [i for i, c in enumerate(text) if c == ']']

        if not open_positions or not close_positions:
            return None

        best: Optional[List[dict]] = None
        best_len = 0

        for op in open_positions:
            for cp in reversed(close_positions):
                if cp <= op:
                    continue
                candidate = text[op:cp + 1]
                # Skip very short candidates unlikely to be component arrays
                if len(candidate) < 10:
                    continue
                try:
                    parsed = json.loads(candidate)
                    if isinstance(parsed, list) and len(parsed) > best_len:
                        best = parsed
                        best_len = len(parsed)
                except (json.JSONDecodeError, ValueError):
                    continue

        return best

    # ------------------------------------------------------------------
    # Phase 2-7: Gather symbol + datasheet for a single component
    # ------------------------------------------------------------------

    async def _gather_component(
        self,
        comp: ComponentRequirement,
        index: int,
        total: int,
    ) -> ComponentGatherResult:
        """
        Gather symbol and datasheet for a single component using the ordered
        search chain: GraphRAG -> KiCad -> SnapEDA -> UltraLibrarian ->
        Datasheet -> LLM.

        No fallbacks: if ALL sources fail, emit ``[ERROR]`` with full details.

        Args:
            comp: The component requirement to gather data for.
            index: 0-based index in the component list.
            total: Total number of components.

        Returns:
            ``ComponentGatherResult`` with gathered data or errors.
        """
        result = ComponentGatherResult(
            part_number=comp.part_number,
            manufacturer=comp.manufacturer,
            category=comp.category,
        )

        def phase_pct() -> int:
            return int(((index + 1) / max(1, total)) * 100)

        # ---- Source 1: GraphRAG / Nexus Memory (FIRST) ----
        try:
            graphrag_result = await self.search_graphrag(comp)
            if graphrag_result:
                self._progress.log_graphrag(
                    f"Searching Nexus Memory for {comp.part_number}... "
                    f"FOUND (ingested "
                    f"{graphrag_result.get('last_success', 'previously')})",
                    phase_progress=phase_pct(),
                )
                result.symbol_found = True
                result.symbol_source = "graphrag"
                result.symbol_content = graphrag_result.get(
                    "symbol_content", ""
                )
                if graphrag_result.get("datasheet_url"):
                    result.datasheet_found = True
                    result.datasheet_source = "graphrag"
                    result.datasheet_url = graphrag_result["datasheet_url"]
                # Write symbol file
                if result.symbol_content:
                    sym_path = (
                        self._symbols_dir / f"{comp.part_number}.kicad_sym"
                    )
                    sym_path.write_text(result.symbol_content)
                    result.symbol_path = str(sym_path)
                return result
        except Exception as exc:
            error_detail = (
                f"GraphRAG search failed for {comp.part_number}: {exc}. "
                f"Nexus API URL: {NEXUS_API_URL}, "
                f"API key present: {bool(NEXUS_API_KEY)}"
            )
            logger.warning(error_detail)
            result.errors.append(error_detail)

        self._progress.log_graphrag(
            f"Searching Nexus Memory for {comp.part_number}... NOT FOUND",
            phase_progress=phase_pct(),
        )

        # ---- Source 2: KiCad Worker ----
        try:
            kicad_data = await self.search_kicad(comp)
            if kicad_data:
                lib_name = kicad_data.get("library", "Unknown")
                self._progress.log_kicad(
                    f"Searching KiCad libraries for {comp.part_number}... "
                    f"FOUND ({lib_name}:{comp.part_number})",
                    phase_progress=phase_pct(),
                )
                result.symbol_found = True
                result.symbol_source = "kicad"
                result.symbol_content = kicad_data.get("symbol_content", "")
                if kicad_data.get("datasheet_url"):
                    result.datasheet_found = True
                    result.datasheet_source = "kicad"
                    result.datasheet_url = kicad_data["datasheet_url"]
                if result.symbol_content:
                    sym_path = (
                        self._symbols_dir / f"{comp.part_number}.kicad_sym"
                    )
                    sym_path.write_text(result.symbol_content)
                    result.symbol_path = str(sym_path)
                return result
        except Exception as exc:
            error_detail = (
                f"KiCad Worker search failed for {comp.part_number}: {exc}. "
                f"Worker URL: {KICAD_WORKER_URL}"
            )
            logger.warning(error_detail)
            result.errors.append(error_detail)

        self._progress.log_kicad(
            f"Searching KiCad libraries for {comp.part_number}... NOT FOUND",
            phase_progress=phase_pct(),
        )

        # ---- Source 3: SnapEDA ----
        try:
            snapeda_data = await self.search_snapeda(comp)
            if snapeda_data:
                self._progress.log_snapeda(
                    f"Searching SnapEDA for {comp.part_number}... FOUND",
                    phase_progress=phase_pct(),
                )
                result.symbol_found = True
                result.symbol_source = "snapeda"
                result.symbol_content = snapeda_data.get("symbol_content", "")
                if snapeda_data.get("datasheet_url"):
                    result.datasheet_found = True
                    result.datasheet_source = "snapeda"
                    result.datasheet_url = snapeda_data["datasheet_url"]
                if result.symbol_content:
                    sym_path = (
                        self._symbols_dir / f"{comp.part_number}.kicad_sym"
                    )
                    sym_path.write_text(result.symbol_content)
                    result.symbol_path = str(sym_path)
                return result
        except Exception as exc:
            error_detail = (
                f"SnapEDA search failed for {comp.part_number}: {exc}. "
                f"API key present: {bool(SNAPEDA_API_KEY)}"
            )
            logger.warning(error_detail)
            result.errors.append(error_detail)

        self._progress.log_snapeda(
            f"Searching SnapEDA for {comp.part_number}... NOT FOUND",
            phase_progress=phase_pct(),
        )

        # ---- Source 4: UltraLibrarian ----
        try:
            ul_data = await self.search_ultralibrarian(comp)
            if ul_data:
                self._progress.log_ultralibrarian(
                    f"Searching UltraLibrarian for {comp.part_number}... FOUND",
                    phase_progress=phase_pct(),
                )
                result.symbol_found = True
                result.symbol_source = "ultralibrarian"
                result.symbol_content = ul_data.get("symbol_content", "")
                if ul_data.get("datasheet_url"):
                    result.datasheet_found = True
                    result.datasheet_source = "ultralibrarian"
                    result.datasheet_url = ul_data["datasheet_url"]
                if result.symbol_content:
                    sym_path = (
                        self._symbols_dir / f"{comp.part_number}.kicad_sym"
                    )
                    sym_path.write_text(result.symbol_content)
                    result.symbol_path = str(sym_path)
                return result
        except Exception as exc:
            error_detail = (
                f"UltraLibrarian search failed for {comp.part_number}: {exc}. "
                f"API key present: {bool(ULTRALIBRARIAN_API_KEY)}"
            )
            logger.warning(error_detail)
            result.errors.append(error_detail)

        self._progress.log_ultralibrarian(
            f"Searching UltraLibrarian for {comp.part_number}... NOT FOUND",
            phase_progress=phase_pct(),
        )

        # ---- Source 5: Datasheet search (DigiKey, Mouser, LCSC) ----
        try:
            ds_data = await self._search_datasheets(comp)
            if ds_data:
                result.datasheet_found = True
                result.datasheet_source = ds_data.get("source", "web")
                result.datasheet_url = ds_data.get("url", "")
                self._progress.log_datasheet(
                    f"Found datasheet for {comp.part_number} from "
                    f"{result.datasheet_source}",
                    phase_progress=phase_pct(),
                )
                # Download if URL available
                if result.datasheet_url:
                    dl_path = await self.download_datasheet(
                        comp.part_number, result.datasheet_url
                    )
                    if dl_path:
                        result.datasheet_path = dl_path
        except Exception as exc:
            error_detail = (
                f"Datasheet search failed for {comp.part_number}: {exc}"
            )
            logger.warning(error_detail)
            result.errors.append(error_detail)

        # ---- Source 6: LLM Generation (Opus 4.6) ----
        try:
            self._progress.log_llm(
                f"Generating symbol for {comp.part_number} using Opus 4.6...",
                phase_progress=phase_pct(),
            )
            llm_symbol = await self.generate_symbol_with_llm(comp)
            if llm_symbol:
                result.symbol_found = True
                result.symbol_source = "llm_generated"
                result.symbol_content = llm_symbol
                result.llm_generated = True
                result.pin_count = llm_symbol.count("(pin ")
                if result.symbol_content:
                    sym_path = (
                        self._symbols_dir / f"{comp.part_number}.kicad_sym"
                    )
                    sym_path.write_text(result.symbol_content)
                    result.symbol_path = str(sym_path)
                self._progress.log_llm(
                    f"Generating symbol for {comp.part_number} using "
                    f"Opus 4.6... COMPLETE ({result.pin_count} pins)",
                    phase_progress=phase_pct(),
                )
        except Exception as exc:
            error_detail = (
                f"LLM symbol generation failed for {comp.part_number}: {exc}. "
                f"OpenRouter API key present: {bool(OPENROUTER_API_KEY)}"
            )
            logger.error(error_detail)
            result.errors.append(error_detail)

        # If no datasheet was found, create characterization doc
        if not result.datasheet_found and result.symbol_found:
            try:
                char_path = await self._create_characterization(comp)
                if char_path:
                    result.characterization_created = True
                    result.characterization_path = str(char_path)
            except Exception as exc:
                error_detail = (
                    f"Characterization creation failed for "
                    f"{comp.part_number}: {exc}"
                )
                logger.warning(error_detail)
                result.errors.append(error_detail)

        # If STILL no symbol found after all sources, emit [ERROR]
        if not result.symbol_found:
            error_msg = (
                f"Failed to find symbol for {comp.part_number} - "
                f"Part not resolved by any source. "
                f"Searched: GraphRAG, KiCad, SnapEDA, UltraLibrarian, LLM. "
                f"Input: {{part_number: \"{comp.part_number}\", "
                f"manufacturer: \"{comp.manufacturer}\", "
                f"package: \"{comp.package}\"}}. "
                f"Action required: verify part number in BOM artifact."
            )
            self._progress.log_error(error_msg)
            result.errors.append(error_msg)

        return result

    # ------------------------------------------------------------------
    # Public source search methods
    # ------------------------------------------------------------------

    def _graphrag_internal_headers(self) -> Dict[str, str]:
        """Standard headers for internal GraphRAG requests (no auth required)."""
        return {
            "Content-Type": "application/json",
            "X-Company-ID": GRAPHRAG_COMPANY_ID,
            "X-App-ID": GRAPHRAG_APP_ID,
            "X-Service-Name": "nexus-ee-design",
            "X-Internal-Request": "true",
        }

    async def search_graphrag(
        self,
        component: ComponentRequirement,
    ) -> Optional[Dict[str, Any]]:
        """
        Search GraphRAG for previously ingested component resolution data.

        Uses the internal cluster API (no auth required):
        ``POST {GRAPHRAG_INTERNAL_URL}/internal/search``

        Searches for entities with matching part_number in domain 'technical'.
        Results are filtered by relevance score (>0.5) and content matching.

        Args:
            component: The component requirement to search for.

        Returns:
            A dict with ``symbol_content``, ``datasheet_url``,
            ``last_success``, ``confidence`` if found.  ``None`` otherwise.
        """
        http = await self._get_http()

        search_text = (
            f"{component.part_number} {component.manufacturer} "
            f"{component.package} schematic symbol KiCad"
        )

        search_url = f"{GRAPHRAG_INTERNAL_URL}/internal/search"
        try:
            response = await http.post(
                search_url,
                headers=self._graphrag_internal_headers(),
                json={
                    "text": search_text,
                    "domain": "technical",
                    "limit": 10,
                    "minScore": 0.4,
                },
                timeout=15.0,
            )
        except Exception as exc:
            logger.warning(
                f"GraphRAG search request failed for {component.part_number}: "
                f"{type(exc).__name__}: {exc}. URL: {search_url}"
            )
            return None

        if response.status_code != 200:
            logger.warning(
                f"GraphRAG search returned HTTP {response.status_code} "
                f"for {component.part_number}. URL: {search_url}. "
                f"Body: {response.text[:500]}"
            )
            return None

        data = response.json()
        if not data.get("success"):
            logger.warning(
                f"GraphRAG search unsuccessful for {component.part_number}: "
                f"{data.get('error', {}).get('message', 'unknown error')}"
            )
            return None

        results = data.get("data", {}).get("results", [])

        # Search through results for component_resolution entities
        pn_lower = component.part_number.lower()
        for result in results:
            content_str = result.get("content", "")
            metadata = result.get("metadata", {})
            score = result.get("score", 0)

            # Check metadata first (structured match)
            if metadata.get("part_number", "").lower() == pn_lower:
                return self._extract_component_from_result(
                    content_str, metadata, score
                )

            # Check content for embedded JSON with part_number
            if isinstance(content_str, str) and pn_lower in content_str.lower():
                try:
                    parsed = json.loads(content_str)
                    if isinstance(parsed, dict):
                        mem_pn = str(parsed.get("part_number", "")).lower()
                        if mem_pn == pn_lower or pn_lower in mem_pn:
                            return self._extract_component_from_result(
                                content_str, metadata, score, parsed
                            )
                except json.JSONDecodeError:
                    pass

        return None

    def _extract_component_from_result(
        self,
        content_str: str,
        metadata: Dict[str, Any],
        score: float,
        parsed: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Extract symbol data from a GraphRAG search result."""
        if parsed is None:
            try:
                parsed = json.loads(content_str)
            except (json.JSONDecodeError, TypeError):
                parsed = {}

        resolution = parsed.get("resolution", parsed)
        return {
            "symbol_content": resolution.get("symbol_content", ""),
            "source": resolution.get("source", "graphrag"),
            "datasheet_url": resolution.get("datasheet_url", ""),
            "last_success": resolution.get("last_success", ""),
            "confidence": resolution.get("confidence", score),
        }

    async def search_kicad(
        self,
        component: ComponentRequirement,
    ) -> Optional[Dict[str, Any]]:
        """
        Search KiCad symbol libraries via the kicad-worker API.

        Queries the internal KiCad worker service to find symbols in
        official KiCad libraries.

        Args:
            component: The component requirement to search for.

        Returns:
            Dict with ``symbol_content``, ``library``, ``datasheet_url``
            if found.  ``None`` otherwise.
        """
        http = await self._get_http()

        search_url = f"{KICAD_WORKER_URL}/v1/symbols/search"
        response = await http.get(
            search_url,
            params={"query": component.part_number, "limit": 5},
            timeout=15.0,
        )

        if response.status_code != 200:
            logger.debug(
                f"KiCad Worker search returned HTTP {response.status_code} "
                f"for {component.part_number}. Body: {response.text[:500]}"
            )
            return None

        data = response.json()
        results = data.get("results", [])

        if not results:
            return None

        # Use first (best) result
        best = results[0]
        lib_name = best.get("library", "")
        symbol_name = best.get("name", component.part_number)

        # Fetch full symbol content
        lib_url = (
            f"{KICAD_WORKER_URL}/v1/symbols/"
            f"{quote_plus(lib_name)}/{quote_plus(symbol_name)}"
        )
        sym_response = await http.get(lib_url, timeout=15.0)

        if sym_response.status_code != 200:
            logger.debug(
                f"KiCad Worker fetch returned HTTP {sym_response.status_code} "
                f"for {lib_name}:{symbol_name}"
            )
            return None

        sym_data = sym_response.json()
        return {
            "symbol_content": sym_data.get("content", ""),
            "library": lib_name,
            "datasheet_url": sym_data.get(
                "datasheet_url", best.get("datasheet", "")
            ),
        }

    async def search_snapeda(
        self,
        component: ComponentRequirement,
    ) -> Optional[Dict[str, Any]]:
        """
        Search SnapEDA API for KiCad symbols.

        Args:
            component: The component requirement to search for.

        Returns:
            Dict with ``symbol_content``, ``datasheet_url`` if found.
            ``None`` otherwise.
        """
        http = await self._get_http()

        headers: Dict[str, str] = {"Accept": "application/json"}
        if SNAPEDA_API_KEY:
            headers["Authorization"] = f"Bearer {SNAPEDA_API_KEY}"

        search_url = "https://www.snapeda.com/api/v1/parts/search"
        response = await http.get(
            search_url,
            params={"q": component.part_number, "limit": 5},
            headers=headers,
            timeout=20.0,
        )

        if response.status_code != 200:
            logger.debug(
                f"SnapEDA search returned HTTP {response.status_code} "
                f"for {component.part_number}. Body: {response.text[:500]}"
            )
            return None

        data = response.json()
        results = data.get("results", data.get("parts", []))

        if not results:
            return None

        # Find best match
        best = results[0]
        part_url = best.get("url", "")
        has_kicad = best.get("has_kicad_symbol", False) or best.get(
            "has_symbol", False
        )

        if not has_kicad:
            logger.debug(
                f"SnapEDA found {component.part_number} but no KiCad symbol "
                f"available"
            )
            return None

        # Try to download KiCad symbol
        download_url = best.get("kicad_symbol_url", "")
        if not download_url and part_url:
            download_url = f"{part_url.rstrip('/')}/kicad/"

        if download_url:
            try:
                dl_response = await http.get(
                    download_url,
                    headers=headers,
                    timeout=30.0,
                )
                if dl_response.status_code == 200:
                    content_type = dl_response.headers.get("content-type", "")
                    if "json" in content_type:
                        sym_data = dl_response.json()
                        return {
                            "symbol_content": sym_data.get(
                                "symbol", sym_data.get("content", "")
                            ),
                            "datasheet_url": best.get("datasheet_url", ""),
                        }
                    else:
                        return {
                            "symbol_content": dl_response.text,
                            "datasheet_url": best.get("datasheet_url", ""),
                        }
            except Exception as dl_exc:
                logger.warning(
                    f"SnapEDA symbol download failed for "
                    f"{component.part_number}: {dl_exc}"
                )

        return None

    async def search_ultralibrarian(
        self,
        component: ComponentRequirement,
    ) -> Optional[Dict[str, Any]]:
        """
        Search UltraLibrarian API for KiCad symbols.

        Args:
            component: The component requirement to search for.

        Returns:
            Dict with ``symbol_content``, ``datasheet_url`` if found.
            ``None`` otherwise.
        """
        if not ULTRALIBRARIAN_API_KEY:
            logger.debug(
                "No ULTRALIBRARIAN_API_KEY, skipping UltraLibrarian search"
            )
            return None

        http = await self._get_http()

        search_url = "https://app.ultralibrarian.com/api/v1/search"
        response = await http.get(
            search_url,
            params={
                "keyword": component.part_number,
                "format": "kicad",
                "limit": 5,
            },
            headers={
                "Authorization": f"Bearer {ULTRALIBRARIAN_API_KEY}",
                "Accept": "application/json",
            },
            timeout=20.0,
        )

        if response.status_code != 200:
            logger.debug(
                f"UltraLibrarian search returned HTTP {response.status_code} "
                f"for {component.part_number}. Body: {response.text[:500]}"
            )
            return None

        data = response.json()
        results = data.get("results", data.get("parts", []))

        if not results:
            return None

        best = results[0]
        download_url = best.get("download_url", best.get("kicad_url", ""))

        if not download_url:
            return None

        try:
            dl_response = await http.get(
                download_url,
                headers={
                    "Authorization": f"Bearer {ULTRALIBRARIAN_API_KEY}"
                },
                timeout=30.0,
            )
            if dl_response.status_code == 200:
                content_type = dl_response.headers.get("content-type", "")
                if "json" in content_type:
                    sym_data = dl_response.json()
                    return {
                        "symbol_content": sym_data.get(
                            "symbol", sym_data.get("content", "")
                        ),
                        "datasheet_url": best.get("datasheet_url", ""),
                    }
                else:
                    return {
                        "symbol_content": dl_response.text,
                        "datasheet_url": best.get("datasheet_url", ""),
                    }
        except Exception as dl_exc:
            logger.warning(
                f"UltraLibrarian download failed for "
                f"{component.part_number}: {dl_exc}"
            )

        return None

    async def download_datasheet(
        self,
        part_number: str,
        url: str,
    ) -> Optional[str]:
        """
        Download a datasheet PDF to the datasheets output directory.

        Args:
            part_number: Component part number (used for the filename).
            url: URL to download the PDF from.

        Returns:
            Filesystem path (as string) to the downloaded file,
            or ``None`` if the download failed.
        """
        http = await self._get_http()

        try:
            response = await http.get(url, timeout=60.0)
            if response.status_code == 200:
                ds_path = (
                    self._datasheets_dir / f"{part_number}_datasheet.pdf"
                )
                ds_path.write_bytes(response.content)
                size_mb = len(response.content) / 1024 / 1024
                logger.info(
                    f"Downloaded datasheet for {part_number}: "
                    f"{len(response.content)} bytes -> {ds_path}"
                )
                self._progress.log_datasheet(
                    f"Downloading {part_number} datasheet... "
                    f"COMPLETE ({size_mb:.1f}MB)"
                )
                return str(ds_path)
            else:
                logger.warning(
                    f"Datasheet download returned HTTP {response.status_code} "
                    f"for {part_number} from {url}"
                )
        except Exception as exc:
            logger.warning(
                f"Datasheet download failed for {part_number}: {exc}"
            )

        return None

    async def generate_symbol_with_llm(
        self,
        component: ComponentRequirement,
    ) -> str:
        """
        Use Opus 4.6 to generate a KiCad symbol when no symbol was found
        from any external source.

        Args:
            component: The component to generate a symbol for.

        Returns:
            The raw KiCad S-expression symbol content string.

        Raises:
            RuntimeError: If OPENROUTER_API_KEY is not set or if the LLM
                returns an invalid response.
        """
        if not self._should_use_proxy() and not OPENROUTER_API_KEY:
            raise RuntimeError(
                "No LLM endpoint available for symbol generation. "
                "Either set LLM_ANTHROPIC_ROUTE=claude_code_max or "
                "provide OPENROUTER_API_KEY."
            )

        prompt = (
            "You are an expert KiCad symbol designer. Generate a complete, "
            "valid KiCad 8.x S-expression symbol for the following "
            "component:\n"
            "\n"
            f"Part Number: {component.part_number}\n"
            f"Manufacturer: {component.manufacturer}\n"
            f"Category: {component.category}\n"
            f"Package: {component.package}\n"
            f"Value: {component.value}\n"
            f"Description: {component.description}\n"
            "\n"
            "Requirements:\n"
            "1. Use KiCad 8.x S-expression format\n"
            "2. Include ALL pins with correct names, numbers, types "
            "(input/output/bidirectional/passive/power_in/power_out)\n"
            "3. Include proper pin positions and orientations\n"
            "4. Include Reference, Value, Footprint, Datasheet properties\n"
            "5. Use standard pin names for this component type\n"
            "6. Body rectangle should fit all pins with proper spacing\n"
            "7. Use tab indentation (KiCad 8.x standard)\n"
            "\n"
            "Return ONLY the raw S-expression starting with (symbol ...), "
            "no markdown formatting, no explanation.\n"
        )

        response_text = await self._call_opus(prompt)

        # Clean response (strip markdown fences and preamble text)
        cleaned = response_text.strip()
        fence_match = re.search(r'```(?:\w*)\s*\n(.*?)```', cleaned, re.DOTALL)
        if fence_match:
            cleaned = fence_match.group(1).strip()
        elif not cleaned.startswith("(symbol"):
            # No fences, try to find S-expression start
            sym_start = cleaned.find("(symbol")
            if sym_start != -1:
                cleaned = cleaned[sym_start:]

        if not cleaned.startswith("(symbol"):
            logger.error(
                f"LLM symbol generation for {component.part_number} did not "
                f"return valid S-expression. Response starts with: "
                f"{cleaned[:100]}"
            )
            raise RuntimeError(
                f"LLM generated invalid symbol for {component.part_number}. "
                f"Expected S-expression starting with '(symbol', got: "
                f"'{cleaned[:100]}'"
            )

        return cleaned

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    async def _search_datasheets(
        self,
        comp: ComponentRequirement,
    ) -> Optional[Dict[str, Any]]:
        """
        Search for datasheets from DigiKey, Mouser, and LCSC.

        Uses Opus 4.6 to extract the best datasheet URL from search results
        rather than relying on fragile HTML parsing.

        Args:
            comp: The component requirement.

        Returns:
            Dict with ``source`` and ``url`` if found.  ``None`` otherwise.
        """
        http = await self._get_http()

        # Try known datasheet URL patterns for common manufacturers
        manufacturer_ds_patterns: Dict[str, str] = {
            "ST": (
                f"https://www.st.com/resource/en/datasheet/"
                f"{comp.part_number.lower()}.pdf"
            ),
            "TI": (
                f"https://www.ti.com/lit/ds/symlink/"
                f"{comp.part_number.lower()}.pdf"
            ),
            "Infineon": (
                f"https://www.infineon.com/dgdl/"
                f"{comp.part_number}+-DS-v01_00-EN.pdf"
            ),
            "NXP": (
                f"https://www.nxp.com/docs/en/data-sheet/"
                f"{comp.part_number}.pdf"
            ),
        }

        # Check manufacturer-specific URL first
        mfr = comp.manufacturer.upper() if comp.manufacturer else ""
        for mfr_key, ds_url in manufacturer_ds_patterns.items():
            if mfr_key.upper() in mfr or mfr in mfr_key.upper():
                try:
                    head_resp = await http.head(ds_url, timeout=10.0)
                    if head_resp.status_code == 200:
                        content_type = head_resp.headers.get(
                            "content-type", ""
                        )
                        if "pdf" in content_type or "octet" in content_type:
                            return {"source": mfr_key, "url": ds_url}
                except Exception:
                    pass

        # Try DigiKey search
        try:
            digikey_url = (
                f"https://www.digikey.com/en/products/filter?"
                f"keywords={quote_plus(comp.part_number)}"
            )
            digi_resp = await http.get(digikey_url, timeout=15.0)
            if (
                digi_resp.status_code == 200
                and "datasheet" in digi_resp.text.lower()
            ):
                ds_url = await self._extract_datasheet_url_with_llm(
                    comp.part_number, digi_resp.text[:5000], "DigiKey"
                )
                if ds_url:
                    return {"source": "DigiKey", "url": ds_url}
        except Exception as exc:
            logger.debug(
                f"DigiKey search failed for {comp.part_number}: {exc}"
            )

        # Try Mouser search
        try:
            mouser_url = (
                f"https://www.mouser.com/c/?q="
                f"{quote_plus(comp.part_number)}"
            )
            mouser_resp = await http.get(mouser_url, timeout=15.0)
            if (
                mouser_resp.status_code == 200
                and "datasheet" in mouser_resp.text.lower()
            ):
                ds_url = await self._extract_datasheet_url_with_llm(
                    comp.part_number, mouser_resp.text[:5000], "Mouser"
                )
                if ds_url:
                    return {"source": "Mouser", "url": ds_url}
        except Exception as exc:
            logger.debug(
                f"Mouser search failed for {comp.part_number}: {exc}"
            )

        # Try LCSC search
        try:
            lcsc_url = (
                f"https://www.lcsc.com/search?q="
                f"{quote_plus(comp.part_number)}"
            )
            lcsc_resp = await http.get(lcsc_url, timeout=15.0)
            if (
                lcsc_resp.status_code == 200
                and "datasheet" in lcsc_resp.text.lower()
            ):
                ds_url = await self._extract_datasheet_url_with_llm(
                    comp.part_number, lcsc_resp.text[:5000], "LCSC"
                )
                if ds_url:
                    return {"source": "LCSC", "url": ds_url}
        except Exception as exc:
            logger.debug(
                f"LCSC search failed for {comp.part_number}: {exc}"
            )

        return None

    async def _create_characterization(
        self,
        comp: ComponentRequirement,
    ) -> Optional[Path]:
        """
        Use Opus 4.6 to create a characterization document when no datasheet
        is available.  Synthesizes all available information about the
        component into a markdown reference document.

        Args:
            comp: The component to characterize.

        Returns:
            Path to the characterization markdown file, or ``None`` if
            creation failed.
        """
        if not self._should_use_proxy() and not OPENROUTER_API_KEY:
            logger.warning(
                f"Cannot create characterization for {comp.part_number}: "
                "no LLM endpoint available (set LLM_ANTHROPIC_ROUTE=claude_code_max "
                "or OPENROUTER_API_KEY)"
            )
            return None

        prompt = (
            "You are an expert electronics engineer. Create a comprehensive "
            "component characterization document for a component that has no "
            "datasheet available. Include all available information and "
            "reasonable engineering assumptions based on the component type "
            "and category.\n"
            "\n"
            f"Part Number: {comp.part_number}\n"
            f"Manufacturer: {comp.manufacturer}\n"
            f"Category: {comp.category}\n"
            f"Package: {comp.package}\n"
            f"Value: {comp.value}\n"
            f"Description: {comp.description}\n"
            f"Subsystem: {comp.subsystem}\n"
            "\n"
            "Create a markdown document with these sections:\n"
            "1. Component Overview\n"
            "2. Pin Configuration (table with pin number, name, type, "
            "description)\n"
            "3. Electrical Characteristics (based on category norms)\n"
            "4. Absolute Maximum Ratings\n"
            "5. Typical Application Circuit\n"
            "6. PCB Layout Recommendations\n"
            "7. Notes and Assumptions (clearly mark what is assumed vs "
            "known)\n"
            "\n"
            "Return the markdown document only, no additional commentary."
        )

        response_text = await self._call_opus(prompt)

        char_path = (
            self._characterizations_dir
            / f"{comp.part_number}_characterization.md"
        )
        char_path.write_text(response_text)
        logger.info(
            f"Created characterization for {comp.part_number}: {char_path}"
        )

        return char_path

    async def _extract_datasheet_url_with_llm(
        self,
        part_number: str,
        page_html: str,
        source: str,
    ) -> Optional[str]:
        """
        Use Opus 4.6 to extract a datasheet PDF URL from HTML search results.

        Args:
            part_number: The component part number being searched.
            page_html: First N chars of the search results HTML page.
            source: Name of the source site (DigiKey, Mouser, LCSC).

        Returns:
            Direct URL to the datasheet PDF, or ``None`` if not found.
        """
        if not self._should_use_proxy() and not OPENROUTER_API_KEY:
            return None

        prompt = (
            f"Extract the direct PDF datasheet URL for part number "
            f"'{part_number}' from this {source} search results page HTML "
            f"snippet. Return ONLY the URL string (starting with http or "
            f"https), no markdown, no explanation. If no datasheet URL is "
            f"found, return the single word 'NONE'.\n"
            "\n"
            f"HTML:\n{page_html}"
        )

        try:
            response = await self._call_opus(prompt)
            result = response.strip()
            if result.upper() == "NONE" or not result.startswith("http"):
                return None
            return result
        except Exception as exc:
            logger.debug(f"LLM datasheet URL extraction failed: {exc}")
            return None

    # ------------------------------------------------------------------
    # GraphRAG ingestion
    # ------------------------------------------------------------------

    async def _ingest_to_graphrag(
        self,
        results: List[ComponentGatherResult],
    ) -> None:
        """
        Store all gathered component data into GraphRAG via the internal API
        ``POST {GRAPHRAG_INTERNAL_URL}/internal/entities`` so future sessions
        can find these components via GraphRAG-first search.

        Each component is stored as a 'technical' domain entity with
        structured metadata for exact part_number matching and vector
        similarity search.

        Args:
            results: List of component gather results to ingest.
        """
        successful = [r for r in results if r.symbol_found]
        if not successful:
            logger.info("No successful results to ingest into GraphRAG")
            return

        self._progress.log_ingest(
            f"Storing {len(successful)} symbols to GraphRAG for future "
            f"sessions...",
            phase_progress=0,
        )

        http = await self._get_http()
        store_url = f"{GRAPHRAG_INTERNAL_URL}/internal/entities"
        stored = 0
        errors = 0

        for idx, result in enumerate(successful):
            try:
                # Build structured content for vector search and exact match
                entity_content = json.dumps(
                    {
                        "type": "component_resolution",
                        "part_number": result.part_number,
                        "manufacturer": result.manufacturer,
                        "category": result.category,
                        "resolution": {
                            "source": result.symbol_source,
                            "format": "kicad_sym",
                            "symbol_content": (
                                result.symbol_content[:10000]
                                if result.symbol_content
                                else ""
                            ),
                            "datasheet_url": result.datasheet_url,
                            "verified": result.symbol_source
                            in ("kicad", "snapeda", "ultralibrarian"),
                            "success_count": 1,
                            "last_success": datetime.now(
                                timezone.utc
                            ).isoformat(),
                            "confidence": (
                                0.9
                                if result.symbol_source != "llm_generated"
                                else 0.7
                            ),
                            "llm_generated": result.llm_generated,
                            "pin_count": result.pin_count,
                        },
                    }
                )

                entity_id = str(uuid.uuid5(
                    uuid.NAMESPACE_DNS,
                    f"component:{result.part_number}:{result.manufacturer}",
                ))

                response = await http.post(
                    store_url,
                    headers=self._graphrag_internal_headers(),
                    json={
                        "id": entity_id,
                        "domain": "technical",
                        "entityType": "component_resolution",
                        "content": entity_content,
                        "metadata": {
                            "part_number": result.part_number,
                            "manufacturer": result.manufacturer,
                            "category": result.category,
                            "package": result.package,
                            "source": result.symbol_source,
                            "project_id": self.project_id,
                            "pin_count": result.pin_count,
                            "event_type": "component_resolution",
                        },
                    },
                    timeout=30.0,
                )

                if response.status_code in (200, 201):
                    stored += 1
                    logger.debug(
                        f"Stored {result.part_number} to GraphRAG "
                        f"(entity: {entity_id})"
                    )
                else:
                    errors += 1
                    logger.warning(
                        f"GraphRAG store returned HTTP "
                        f"{response.status_code} for {result.part_number}. "
                        f"URL: {store_url}. "
                        f"Body: {response.text[:500]}"
                    )
            except Exception as exc:
                errors += 1
                logger.warning(
                    f"GraphRAG store failed for {result.part_number}: "
                    f"{type(exc).__name__}: {exc}. URL: {store_url}"
                )

            # Update progress
            pct = int(((idx + 1) / len(successful)) * 100)
            self._progress.log_ingest(
                f"Storing to GraphRAG... {idx + 1}/{len(successful)}",
                phase_progress=pct,
            )

        datasheets_count = sum(1 for r in results if r.datasheet_found)
        self._progress.log_ingest(
            f"Storing {stored} symbols + {datasheets_count} datasheets "
            f"to GraphRAG for future sessions... COMPLETE"
            + (f" ({errors} store errors)" if errors else ""),
            phase_progress=100,
        )

    # ------------------------------------------------------------------
    # LLM routing helpers
    # ------------------------------------------------------------------

    def _is_anthropic_model(self, model_id: str) -> bool:
        """Check if a model ID refers to an Anthropic model."""
        return model_id.startswith("anthropic/") or model_id.startswith("claude-")

    def _should_use_proxy(self) -> bool:
        """Whether to route Anthropic models through Claude Code Max proxy."""
        return (
            LLM_ANTHROPIC_ROUTE == "claude_code_max"
            and self._is_anthropic_model(OPENROUTER_MODEL)
        )

    def _get_proxy_model_id(self) -> str:
        """Translate OpenRouter model ID to native Anthropic model ID."""
        return OPENROUTER_TO_ANTHROPIC_MODEL_ID.get(
            OPENROUTER_MODEL, "claude-opus-4-6"
        )

    # ------------------------------------------------------------------
    # Opus 4.6 LLM call (routes via proxy or OpenRouter)
    # ------------------------------------------------------------------

    async def _call_opus(
        self,
        prompt: str,
        *,
        system_message: Optional[str] = None,
        assistant_prefill: Optional[str] = None,
    ) -> str:
        """
        Call Opus 4.6 via Claude Code Max proxy (preferred for Anthropic models)
        or OpenRouter (for non-Anthropic models / when proxy is not configured).

        Routing is controlled by LLM_ANTHROPIC_ROUTE env var:
        - "claude_code_max": Anthropic models go through the in-cluster proxy
        - any other value: all models go through OpenRouter

        Args:
            prompt: The user prompt to send.
            system_message: Optional system message for output enforcement.
            assistant_prefill: Optional assistant message prefix to steer
                the model's response format (e.g., "[" to force JSON array).

        Returns:
            The assistant response text.

        Raises:
            RuntimeError: If the LLM call fails for any reason.
        """
        if self._should_use_proxy():
            return await self._call_via_proxy(
                prompt,
                system_message=system_message,
                assistant_prefill=assistant_prefill,
            )
        return await self._call_via_openrouter(
            prompt,
            system_message=system_message,
            assistant_prefill=assistant_prefill,
        )

    async def _call_via_proxy(
        self,
        prompt: str,
        *,
        system_message: Optional[str] = None,
        assistant_prefill: Optional[str] = None,
    ) -> str:
        """Route an Anthropic model call through Claude Code Max proxy.

        Uses SSE streaming (stream=true) so the proxy sends HTTP headers
        immediately with keepalive pings. This prevents read timeouts on
        large prompts where the CLI takes 5-15 minutes to respond.
        """
        proxy_model = self._get_proxy_model_id()
        endpoint = f"{LLM_CLAUDE_CODE_PROXY_URL}/v1/chat/completions"

        logger.info(
            "Routing Anthropic model through Claude Code Max proxy (streaming): "
            f"model={proxy_model}, endpoint={endpoint}, "
            f"prompt_length={len(prompt)}"
        )

        http = await self._get_http()

        messages: list[dict[str, str]] = []
        if system_message:
            messages.append({"role": "system", "content": system_message})
        messages.append({"role": "user", "content": prompt})
        if assistant_prefill:
            messages.append({"role": "assistant", "content": assistant_prefill})

        payload = {
            "model": proxy_model,
            "messages": messages,
            "max_tokens": 8192,
            "temperature": 0.1,
            "stream": True,  # SSE streaming: headers arrive immediately
        }

        # With streaming, headers arrive instantly and keepalive pings flow
        # every 15s. Read timeout only needs to cover gaps between SSE events.
        proxy_timeout = httpx.Timeout(
            connect=30.0,
            read=float(LLM_CLAUDE_CODE_PROXY_TIMEOUT),
            write=60.0,
            pool=30.0,
        )

        try:
            content_parts: list[str] = []
            async with http.stream(
                "POST",
                endpoint,
                headers={"Content-Type": "application/json"},
                json=payload,
                timeout=proxy_timeout,
            ) as response:
                if response.status_code != 200:
                    error_body = (await response.aread()).decode()[:2000]
                    raise RuntimeError(
                        f"Claude Code Max proxy returned HTTP {response.status_code}. "
                        f"Model: {proxy_model}. "
                        f"Proxy URL: {endpoint}. "
                        f"Response body: {error_body}. "
                        f"Hint: Check proxy OAuth status via "
                        f"GET {LLM_CLAUDE_CODE_PROXY_URL}/auth/status"
                    )

                # Parse SSE events from the stream
                async for line in response.aiter_lines():
                    if not line or line.startswith(": keepalive"):
                        continue  # Skip keepalive pings and empty lines

                    if line.startswith("data: "):
                        data_str = line[6:]  # Strip "data: " prefix

                        if data_str == "[DONE]":
                            break

                        try:
                            chunk = json.loads(data_str)
                        except json.JSONDecodeError:
                            continue

                        # Check for error events
                        if "error" in chunk:
                            error_msg = chunk["error"].get("message", str(chunk["error"]))
                            raise RuntimeError(
                                f"Claude Code Max proxy error: {error_msg}"
                            )

                        # Extract content delta from streaming chunks
                        choices = chunk.get("choices", [])
                        if choices:
                            delta = choices[0].get("delta", {})
                            content = delta.get("content", "")
                            if content:
                                content_parts.append(content)

            result = "".join(content_parts)
            if not result:
                raise RuntimeError(
                    f"Claude Code Max proxy returned empty content via streaming. "
                    f"Model: {proxy_model}. "
                    f"Proxy URL: {endpoint}"
                )

            logger.info(
                f"Proxy streaming response received: "
                f"{len(result)} chars in {len(content_parts)} chunks"
            )
            return result

        except httpx.ReadTimeout as exc:
            raise RuntimeError(
                f"Claude Code Max proxy read timeout after {LLM_CLAUDE_CODE_PROXY_TIMEOUT}s. "
                f"The prompt may be too large for a single LLM call. "
                f"Proxy URL: {LLM_CLAUDE_CODE_PROXY_URL}. "
                f"Consider increasing LLM_CLAUDE_CODE_PROXY_TIMEOUT_SECONDS "
                f"(current: {LLM_CLAUDE_CODE_PROXY_TIMEOUT})"
            ) from exc
        except httpx.ConnectError as exc:
            raise RuntimeError(
                f"Claude Code Max proxy connection refused. "
                f"Proxy URL: {LLM_CLAUDE_CODE_PROXY_URL}. "
                f"Check proxy pod: kubectl get pods -n nexus -l app=claude-code-proxy"
            ) from exc
        except RuntimeError:
            raise  # Re-raise our own RuntimeErrors
        except Exception as exc:
            raise RuntimeError(
                f"Claude Code Max proxy request failed: {exc}. "
                f"Proxy URL: {LLM_CLAUDE_CODE_PROXY_URL}. "
                f"Check proxy pod: kubectl get pods -n nexus -l app=claude-code-proxy"
            ) from exc

    async def _call_via_openrouter(
        self,
        prompt: str,
        *,
        system_message: Optional[str] = None,
        assistant_prefill: Optional[str] = None,
    ) -> str:
        """Route an LLM call through OpenRouter."""
        if not OPENROUTER_API_KEY:
            raise RuntimeError(
                "OPENROUTER_API_KEY environment variable is required. "
                f"Model: {OPENROUTER_MODEL}. "
                f"LLM_ANTHROPIC_ROUTE={LLM_ANTHROPIC_ROUTE}. "
                "Hint: For Anthropic models, set LLM_ANTHROPIC_ROUTE=claude_code_max "
                "to use the Claude Code Max proxy instead of OpenRouter."
            )

        logger.info(
            f"Routing model through OpenRouter: model={OPENROUTER_MODEL}"
        )

        http = await self._get_http()

        messages: list[dict[str, str]] = []
        if system_message:
            messages.append({"role": "system", "content": system_message})
        messages.append({"role": "user", "content": prompt})
        if assistant_prefill:
            messages.append({"role": "assistant", "content": assistant_prefill})

        payload = {
            "model": OPENROUTER_MODEL,
            "messages": messages,
            "max_tokens": 8192,
            "temperature": 0.1,
        }

        response = await http.post(
            f"{OPENROUTER_BASE_URL}/chat/completions",
            headers={
                "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://nexus.adverant.ai",
                "X-Title": "Nexus EE Design - Symbol Assembly",
            },
            json=payload,
            timeout=120.0,
        )

        if response.status_code != 200:
            error_body = response.text[:2000]
            status = response.status_code
            hint = ""
            if status == 401:
                hint = (
                    " Hint: OpenRouter API key is invalid or expired. "
                    "Check OPENROUTER_API_KEY env var and OpenRouter dashboard."
                )
            elif status == 402:
                hint = " Hint: OpenRouter account has insufficient credits."
            elif status == 429:
                hint = " Hint: OpenRouter rate limit exceeded. Retry after backoff."
            raise RuntimeError(
                f"OpenRouter API returned HTTP {status}. "
                f"Model: {OPENROUTER_MODEL}. "
                f"Response body: {error_body}.{hint}"
            )

        data = response.json()
        choices = data.get("choices", [])
        if not choices:
            raise RuntimeError(
                f"OpenRouter API returned empty choices. "
                f"Model: {OPENROUTER_MODEL}. "
                f"Full response: {json.dumps(data)[:2000]}"
            )

        return choices[0].get("message", {}).get("content", "")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = [
    # Dataclasses
    "ComponentRequirement",
    "ComponentGatherResult",
    "AssemblyReport",
    # Enums
    "AssemblyPhase",
    "AssemblyEventType",
    # Progress
    "AssemblyProgressEmitter",
    "emit_progress",
    # Main class
    "SymbolAssembler",
]


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Symbol Assembly Agent - gather symbols, datasheets, and characterizations"
    )
    parser.add_argument("--project-id", required=True, help="Project identifier")
    parser.add_argument("--operation-id", required=True, help="Operation ID for progress tracking")
    parser.add_argument("--artifacts-json", required=True, help="Path to JSON file containing ideation artifacts")
    parser.add_argument("--output-dir", required=True, help="Output directory for assembly artifacts")

    args = parser.parse_args()

    # Configure logging to stderr (stdout is reserved for PROGRESS: events)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        stream=sys.stderr,
    )

    # Load artifacts from the JSON file
    artifacts_path = Path(args.artifacts_json)
    if not artifacts_path.exists():
        logger.error(f"Artifacts file not found: {artifacts_path}")
        sys.exit(1)

    try:
        artifacts = json.loads(artifacts_path.read_text())
    except (json.JSONDecodeError, OSError) as exc:
        logger.error(f"Failed to read artifacts file {artifacts_path}: {exc}")
        sys.exit(1)

    logger.info(
        f"Starting symbol assembly: project={args.project_id}, "
        f"operation={args.operation_id}, artifacts={len(artifacts)}"
    )

    async def _main() -> None:
        assembler = SymbolAssembler(
            project_id=args.project_id,
            operation_id=args.operation_id,
            output_base_path=str(Path(args.output_dir).parent.parent),
        )
        try:
            report = await assembler.run(
                project_id=args.project_id,
                artifacts=artifacts,
                operation_id=args.operation_id,
            )
            logger.info(
                f"Assembly complete: {report.symbols_found}/{report.total_components} "
                f"symbols found, {report.datasheets_downloaded} datasheets, "
                f"{report.errors_count} errors"
            )
            if report.errors_count > 0:
                for err in report.errors:
                    logger.warning(f"  Error: {err}")
        finally:
            await assembler.close()

    asyncio.run(_main())
