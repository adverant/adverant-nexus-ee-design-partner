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
import sys
import traceback
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
    os.environ.get("LLM_CLAUDE_CODE_PROXY_TIMEOUT_SECONDS", "300")
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

KICAD_WORKER_URL: str = os.environ.get(
    "KICAD_WORKER_URL", "http://mapos-kicad-worker:8080"
)

ARTIFACT_STORAGE_PATH: str = os.environ.get("ARTIFACT_STORAGE_PATH", "/data/artifacts")

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

        # Serialize artifacts into a text block for the LLM
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

        combined_text = "\n".join(artifact_text_parts)

        prompt = (
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
            "IMPORTANT:\n"
            "- Include ALL components, even power symbols (VCC, GND) and "
            "passive components.\n"
            "- Do NOT skip any component mentioned anywhere in the artifacts.\n"
            "- If a component appears in multiple artifacts, include it only "
            "once with the most complete data.\n"
            "- Return ONLY the JSON array, no markdown formatting, no "
            "explanation.\n"
            "\n"
            "ARTIFACTS:\n"
            f"{combined_text}"
        )

        response_text = await self._call_opus(prompt)

        # Parse the JSON response
        try:
            cleaned = response_text.strip()
            if cleaned.startswith("```"):
                first_newline = cleaned.index("\n")
                cleaned = cleaned[first_newline + 1:]
            if cleaned.endswith("```"):
                cleaned = cleaned[:-3]
            cleaned = cleaned.strip()

            raw_components = json.loads(cleaned)
        except (json.JSONDecodeError, ValueError) as parse_err:
            error_msg = (
                f"Failed to parse Opus 4.6 component extraction response as "
                f"JSON. Parse error: {parse_err}. "
                f"Response (first 2000 chars): {response_text[:2000]}"
            )
            logger.error(error_msg)
            raise RuntimeError(error_msg) from parse_err

        if not isinstance(raw_components, list):
            raise RuntimeError(
                f"Opus 4.6 returned non-array type "
                f"({type(raw_components).__name__}) for component extraction. "
                f"Response: {response_text[:2000]}"
            )

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

    async def search_graphrag(
        self,
        component: ComponentRequirement,
    ) -> Optional[Dict[str, Any]]:
        """
        Search Nexus Memory / GraphRAG for previously ingested component data.

        ``POST https://api.adverant.ai/api/memory/recall``
        ``Authorization: Bearer {NEXUS_API_KEY}``

        Query: ``"{part_number} {manufacturer} schematic symbol datasheet"``

        Args:
            component: The component requirement to search for.

        Returns:
            A dict containing ``symbol_content``, ``datasheet_url``,
            ``last_success``, ``confidence`` if found.  ``None`` otherwise.
        """
        if not NEXUS_API_KEY:
            logger.debug("No NEXUS_API_KEY, skipping GraphRAG search")
            return None

        http = await self._get_http()

        query = (
            f"{component.part_number} {component.manufacturer} "
            f"schematic symbol datasheet"
        )

        response = await http.post(
            f"{NEXUS_API_URL}/api/memory/recall",
            headers={
                "Authorization": f"Bearer {NEXUS_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "query": query,
                "limit": 5,
                "filters": {
                    "event_type": "component_resolution",
                },
            },
        )

        if response.status_code != 200:
            logger.warning(
                f"GraphRAG recall returned HTTP {response.status_code} "
                f"for {component.part_number}. Body: {response.text[:500]}"
            )
            return None

        data = response.json()
        memories = data.get("memories", [])

        for memory in memories:
            content = memory.get("content", "")
            if isinstance(content, str):
                try:
                    parsed = json.loads(content)
                except json.JSONDecodeError:
                    continue
            else:
                parsed = content

            # Check if this matches our component
            mem_pn = str(parsed.get("part_number", "")).lower()
            if (
                mem_pn == component.part_number.lower()
                or component.part_number.lower() in mem_pn
            ):
                resolution = parsed.get("resolution", parsed)
                return {
                    "symbol_content": resolution.get("symbol_content", ""),
                    "source": resolution.get("source", "graphrag"),
                    "datasheet_url": resolution.get("datasheet_url", ""),
                    "last_success": resolution.get("last_success", ""),
                    "confidence": resolution.get("confidence", 0.8),
                }

        return None

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

        # Clean response (strip markdown fences if present)
        cleaned = response_text.strip()
        if cleaned.startswith("```"):
            first_nl = cleaned.index("\n")
            cleaned = cleaned[first_nl + 1:]
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]
        cleaned = cleaned.strip()

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
        Store all gathered data into GraphRAG via
        ``POST https://api.adverant.ai/api/memory/store`` so future sessions
        can find these components via GraphRAG-first search.

        Args:
            results: List of component gather results to ingest.
        """
        if not NEXUS_API_KEY:
            logger.warning("No NEXUS_API_KEY, skipping GraphRAG ingestion")
            return

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
        stored = 0
        errors = 0

        for idx, result in enumerate(successful):
            try:
                memory_content = json.dumps(
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

                response = await http.post(
                    f"{NEXUS_API_URL}/api/memory/store",
                    headers={
                        "Authorization": f"Bearer {NEXUS_API_KEY}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "content": memory_content,
                        "event_type": "component_resolution",
                        "metadata": {
                            "part_number": result.part_number,
                            "manufacturer": result.manufacturer,
                            "category": result.category,
                            "source": result.symbol_source,
                            "project_id": self.project_id,
                        },
                    },
                )

                if response.status_code == 200:
                    stored += 1
                else:
                    errors += 1
                    logger.warning(
                        f"GraphRAG store returned HTTP "
                        f"{response.status_code} for {result.part_number}. "
                        f"Body: {response.text[:500]}"
                    )
            except Exception as exc:
                errors += 1
                logger.warning(
                    f"GraphRAG store failed for {result.part_number}: {exc}"
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

    async def _call_opus(self, prompt: str) -> str:
        """
        Call Opus 4.6 via Claude Code Max proxy (preferred for Anthropic models)
        or OpenRouter (for non-Anthropic models / when proxy is not configured).

        Routing is controlled by LLM_ANTHROPIC_ROUTE env var:
        - "claude_code_max": Anthropic models go through the in-cluster proxy
        - any other value: all models go through OpenRouter

        Args:
            prompt: The user prompt to send.

        Returns:
            The assistant response text.

        Raises:
            RuntimeError: If the LLM call fails for any reason.
        """
        if self._should_use_proxy():
            return await self._call_via_proxy(prompt)
        return await self._call_via_openrouter(prompt)

    async def _call_via_proxy(self, prompt: str) -> str:
        """Route an Anthropic model call through Claude Code Max proxy."""
        proxy_model = self._get_proxy_model_id()
        endpoint = f"{LLM_CLAUDE_CODE_PROXY_URL}/v1/chat/completions"

        logger.info(
            "Routing Anthropic model through Claude Code Max proxy: "
            f"model={proxy_model}, endpoint={endpoint}"
        )

        http = await self._get_http()

        payload = {
            "model": proxy_model,
            "messages": [
                {"role": "user", "content": prompt},
            ],
            "max_tokens": 8192,
            "temperature": 0.1,
        }

        try:
            response = await http.post(
                endpoint,
                headers={"Content-Type": "application/json"},
                json=payload,
                timeout=float(LLM_CLAUDE_CODE_PROXY_TIMEOUT),
            )
        except Exception as exc:
            raise RuntimeError(
                f"Claude Code Max proxy connection failed: {exc}. "
                f"Proxy URL: {LLM_CLAUDE_CODE_PROXY_URL}. "
                f"Check proxy pod status: kubectl get pods -n nexus -l app=claude-code-proxy"
            ) from exc

        if response.status_code != 200:
            error_body = response.text[:2000]
            raise RuntimeError(
                f"Claude Code Max proxy returned HTTP {response.status_code}. "
                f"Model: {proxy_model}. "
                f"Proxy URL: {endpoint}. "
                f"Response body: {error_body}. "
                f"Hint: Check proxy OAuth status via GET {LLM_CLAUDE_CODE_PROXY_URL}/auth/status"
            )

        data = response.json()
        choices = data.get("choices", [])
        if not choices:
            raise RuntimeError(
                f"Claude Code Max proxy returned empty choices. "
                f"Model: {proxy_model}. "
                f"Full response: {json.dumps(data)[:2000]}"
            )

        return choices[0].get("message", {}).get("content", "")

    async def _call_via_openrouter(self, prompt: str) -> str:
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

        payload = {
            "model": OPENROUTER_MODEL,
            "messages": [
                {"role": "user", "content": prompt},
            ],
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
