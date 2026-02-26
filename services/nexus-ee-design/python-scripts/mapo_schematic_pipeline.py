"""
MAPO Schematic Pipeline - Unified orchestrator for schematic generation and validation.

Integrates:
- Symbol Fetcher Agent (real symbols from SnapEDA, KiCad, manufacturers)
- GraphRAG Symbol Indexer (semantic search and caching)
- Schematic Assembler Agent (intelligent placement and routing)
- Vision Validator (LLM-based multi-expert validation)
- MAPO Loop (iterative refinement until quality threshold)

Version: 3.1
Author: Nexus EE Design Team
"""

import asyncio
import json
import logging
import os
import sys
import tempfile
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from agents.symbol_fetcher import SymbolFetcherAgent, FetchedSymbol
from agents.schematic_assembler import (
    SchematicAssemblerAgent,
    SchematicSheet,
    BOMItem,
    Connection,
    BlockDiagram,
    SymbolQuality,
)
from agents.connection_generator import ConnectionGeneratorAgent
from agents.layout_optimizer import LayoutOptimizerAgent
from agents.standards_compliance import StandardsComplianceAgent
from agents.wire_router import EnhancedWireRouter
from agents.functional_validator import MAPOFunctionalValidator
from agents.visual_validator import (
    DualLLMVisualValidator,
    ValidationLoop,
    SchematicImageExtractor,
    ProgressTracker,
    IssueToFixTransformer,
    SchematicFixApplicator,
    ImageExtractionError,
    StagnationError,
)
from agents.artifact_exporter import ArtifactExporterAgent, ArtifactConfig, ExportResult
from agents.smoke_test import SmokeTestAgent, SmokeTestResult
from validation.schematic_vision_validator import (
    SchematicVisionValidator,
    MAPOSchematicLoop,
    SchematicValidationReport,
)
from graphrag.symbol_indexer import SymbolGraphRAGIndexer, create_indexer
from progress_emitter import (
    ProgressEmitter,
    SchematicPhase,
    SchematicEventType,
    calculate_overall_progress,
)
from ideation_context import (
    IdeationContext,
    SymbolResolutionContext,
    ConnectionInferenceContext,
    PlacementContext,
    ValidationContext,
)
from checkpoint_manager import CheckpointManager
from retry_utils import retry_phase

logger = logging.getLogger(__name__)

MAPO_VERSION = "3.1"


class SchematicGenerationError(Exception):
    """
    Raised when schematic generation fails due to validation errors.

    This exception provides detailed information about what went wrong,
    which components failed, and what the user can do to resolve the issue.
    """

    def __init__(
        self,
        message: str,
        failed_components: Optional[List[Dict[str, Any]]] = None,
        validation_errors: Optional[List[str]] = None,
        suggestion: Optional[str] = None
    ):
        self.message = message
        self.failed_components = failed_components or []
        self.validation_errors = validation_errors or []
        self.suggestion = suggestion

        # Build detailed error message
        full_message = message

        if failed_components:
            component_list = "\n".join([
                f"  - {c.get('reference', '?')} ({c.get('part_number', '?')}): {c.get('error', 'Unknown error')}"
                for c in failed_components
            ])
            full_message += f"\n\nFailed components:\n{component_list}"

        if validation_errors:
            error_list = "\n".join([f"  - {e}" for e in validation_errors])
            full_message += f"\n\nValidation errors:\n{error_list}"

        if suggestion:
            full_message += f"\n\nSuggestion: {suggestion}"

        super().__init__(full_message)


@dataclass
class PipelineConfig:
    """Configuration for the MAPO schematic pipeline."""
    # Symbol fetching - default to project-local cache
    symbol_cache_path: Path = field(
        default_factory=lambda: Path(__file__).parent / "symbol_cache"
    )
    use_graphrag: bool = False  # Disabled by default (Neo4j often not available)
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = ""

    # Validation - MUST use Opus 4.6 only per user directive
    primary_model: str = "anthropic/claude-opus-4.6"
    verification_model: str = "anthropic/claude-opus-4.6"
    validation_threshold: float = 0.85
    max_iterations: int = 5

    # Output
    output_dir: Path = field(
        default_factory=lambda: Path(__file__).parent / "output"
    )

    # Artifact Export Configuration (auto-export to PDF/image and NFS sync)
    auto_export: bool = True  # Enable auto-export after generation
    export_pdf: bool = True
    export_svg: bool = True
    export_png: bool = True
    nfs_base_path: str = field(default_factory=lambda: os.environ.get('ARTIFACT_STORAGE_PATH', '/data/artifacts'))
    project_id: Optional[str] = None

    def __post_init__(self):
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.symbol_cache_path.mkdir(parents=True, exist_ok=True)


@dataclass
class PipelineResult:
    """Result from the MAPO schematic pipeline."""
    success: bool
    schematic_path: Optional[Path] = None
    sheets: List[SchematicSheet] = field(default_factory=list)
    validation_report: Optional[SchematicValidationReport] = None
    symbols_fetched: int = 0
    symbols_from_cache: int = 0
    symbols_generated: int = 0
    iterations: int = 0
    total_time_seconds: float = 0
    errors: List[str] = field(default_factory=list)

    # Smoke test results (circuit validation)
    smoke_test_result: Optional[SmokeTestResult] = None
    smoke_test_passed: bool = False

    # Artifact export results (PDF, image, NFS sync)
    export_result: Optional[ExportResult] = None
    pdf_path: Optional[Path] = None
    svg_path: Optional[Path] = None
    png_path: Optional[Path] = None
    nfs_synced: bool = False
    nfs_paths: Dict[str, str] = field(default_factory=dict)

    # Quality metrics (populated by phases for loop runner)
    visual_score: float = 0.0
    smoke_test_score: float = 0.0
    compliance_score: float = 0.0
    center_fallback_ratio: float = 0.0
    placeholder_ratio: float = 0.0
    overlap_count: int = 0
    connection_coverage: float = 0.0

    def to_dict(self) -> Dict:
        return {
            "success": self.success,
            "schematic_path": str(self.schematic_path) if self.schematic_path else None,
            "sheet_count": len(self.sheets),
            "validation_score": self.validation_report.overall_score if self.validation_report else None,
            "validation_passed": self.validation_report.passed if self.validation_report else False,
            "symbols_fetched": self.symbols_fetched,
            "symbols_from_cache": self.symbols_from_cache,
            "symbols_generated": self.symbols_generated,
            "iterations": self.iterations,
            "total_time_seconds": self.total_time_seconds,
            "errors": self.errors,
            # Smoke test results
            "smoke_test_passed": self.smoke_test_passed,
            "smoke_test": self.smoke_test_result.to_dict() if self.smoke_test_result else None,
            # Export results
            "pdf_path": str(self.pdf_path) if self.pdf_path else None,
            "svg_path": str(self.svg_path) if self.svg_path else None,
            "png_path": str(self.png_path) if self.png_path else None,
            "nfs_synced": self.nfs_synced,
            "nfs_paths": self.nfs_paths,
            # Quality metrics
            "visual_score": self.visual_score,
            "smoke_test_score": self.smoke_test_score,
            "compliance_score": self.compliance_score,
            "center_fallback_ratio": self.center_fallback_ratio,
            "placeholder_ratio": self.placeholder_ratio,
            "overlap_count": self.overlap_count,
            "connection_coverage": self.connection_coverage,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)


class MAPOSchematicPipeline:
    """
    Unified pipeline for schematic generation with MAPO validation.

    Usage:
        pipeline = MAPOSchematicPipeline()
        result = await pipeline.generate(
            bom=[...],
            design_intent="FOC ESC for brushless motor",
            connections=[...]
        )
    """

    def __init__(
        self,
        config: Optional[PipelineConfig] = None,
        progress_emitter: Optional[ProgressEmitter] = None,
    ):
        """
        Initialize the pipeline.

        Args:
            config: Pipeline configuration
            progress_emitter: Optional progress emitter for WebSocket streaming
        """
        self.config = config or PipelineConfig()
        self._progress = progress_emitter

        # Components will be initialized lazily
        self._symbol_fetcher: Optional[SymbolFetcherAgent] = None
        self._graphrag_indexer: Optional[SymbolGraphRAGIndexer] = None
        self._assembler: Optional[SchematicAssemblerAgent] = None
        self._connection_generator: Optional[ConnectionGeneratorAgent] = None
        self._layout_optimizer: Optional[LayoutOptimizerAgent] = None
        self._standards_compliance: Optional[StandardsComplianceAgent] = None
        self._functional_validator: Optional[MAPOFunctionalValidator] = None
        self._visual_validator: Optional[DualLLMVisualValidator] = None
        self._validator: Optional[SchematicVisionValidator] = None
        self._renderer: Optional[Any] = None  # KiCanvas renderer
        self._artifact_exporter: Optional[ArtifactExporterAgent] = None  # PDF/image export + NFS sync
        self._smoke_test: Optional[SmokeTestAgent] = None  # SPICE-based circuit validation
        self._checkpoint_mgr: Optional[CheckpointManager] = None  # Checkpoint persistence
        # Note: Gaming AI optimization uses stateless optimize_schematic() function

    def _emit_progress(
        self,
        phase: SchematicPhase,
        phase_progress: int,
        message: str,
        event_type: Optional[str] = None,
        **extra_data
    ) -> None:
        """
        Emit progress event if emitter is configured.

        Args:
            phase: Current phase
            phase_progress: Progress within phase (0-100)
            message: Human-readable message
            event_type: Optional specific event type
            **extra_data: Additional data to include
        """
        if self._progress:
            self._progress.emit_phase(
                phase=phase,
                phase_progress=phase_progress,
                message=message,
                event_type=event_type,
                **extra_data
            )

    def _start_phase(self, phase: SchematicPhase, message: Optional[str] = None) -> None:
        """Mark start of a phase."""
        if self._progress:
            self._progress.start_phase(phase, message)

    def _complete_phase(self, phase: SchematicPhase, message: Optional[str] = None) -> None:
        """Mark completion of a phase."""
        if self._progress:
            self._progress.complete_phase(phase, message)

    async def initialize(self):
        """Initialize all pipeline components."""
        logger.info("Initializing MAPO Schematic Pipeline...")

        # Initialize GraphRAG if enabled
        if self.config.use_graphrag:
            try:
                self._graphrag_indexer = await create_indexer(
                    neo4j_uri=self.config.neo4j_uri,
                    neo4j_user=self.config.neo4j_user,
                    neo4j_password=self.config.neo4j_password
                )
                logger.info("GraphRAG indexer connected")
            except Exception as e:
                logger.warning(f"GraphRAG initialization failed: {e}")
                self._graphrag_indexer = None

        # Initialize symbol fetcher
        self._symbol_fetcher = SymbolFetcherAgent(
            cache_path=self.config.symbol_cache_path,
            graphrag_client=self._graphrag_indexer
        )
        logger.info("Symbol fetcher initialized")

        # Initialize assembler
        self._assembler = SchematicAssemblerAgent(
            symbol_fetcher=self._symbol_fetcher,
            graphrag_client=self._graphrag_indexer
        )
        logger.info("Schematic assembler initialized")

        # Initialize connection generator
        self._connection_generator = ConnectionGeneratorAgent()
        logger.info("Connection generator initialized")

        # Initialize layout optimizer (Phase 12 - IPC-2221/IEEE 315 compliance)
        self._layout_optimizer = LayoutOptimizerAgent()
        logger.info("Layout optimizer initialized")

        # Initialize standards compliance checker (Phase 13 - IEC 60750/IEEE 315)
        self._standards_compliance = StandardsComplianceAgent()
        logger.info("Standards compliance agent initialized")

        # Initialize MAPO functional validator (Phase 15 - Competitive multi-agent)
        self._functional_validator = MAPOFunctionalValidator()
        logger.info("MAPO functional validator initialized")

        # Initialize dual-LLM visual validator (Phase 16 - Opus 4.6 + Kimi K2.5)
        self._visual_validator = DualLLMVisualValidator()
        logger.info("Dual-LLM visual validator initialized")

        # Initialize legacy validator
        self._validator = SchematicVisionValidator(
            primary_model=self.config.primary_model,
            verification_model=self.config.verification_model
        )
        self._validator.PASS_THRESHOLD = self.config.validation_threshold
        logger.info("Vision validator initialized")

        # Initialize artifact exporter for auto-export to PDF/image and NFS sync
        if self.config.auto_export:
            artifact_config = ArtifactConfig(
                artifact_base_path=self.config.nfs_base_path,  # NFS mount path for artifacts
                export_pdf=self.config.export_pdf,
                export_svg=self.config.export_svg,
                export_png=self.config.export_png,
                project_id=self.config.project_id,
            )
            self._artifact_exporter = ArtifactExporterAgent(artifact_config)
            storage_status = self._artifact_exporter.get_storage_status()
            logger.info(f"Artifact exporter initialized (path: {storage_status['storage_path']}, writable: {storage_status['writable']})")

        # Initialize smoke test agent (LLM-based circuit validation)
        self._smoke_test = SmokeTestAgent()
        logger.info("Smoke test agent initialized (LLM-based circuit validation)")

        logger.info("Pipeline initialization complete (all MAPO agents ready)")

    async def generate(
        self,
        bom: List[Dict[str, Any]],
        design_intent: str,
        connections: Optional[List[Dict[str, str]]] = None,
        block_diagram: Optional[Dict[str, Any]] = None,
        design_name: str = "schematic",
        reference_images: Optional[List[bytes]] = None,
        skip_validation: bool = False,
        ideation_context: Optional[IdeationContext] = None,
        resume_from_checkpoint: bool = False,
    ) -> PipelineResult:
        """
        Generate a validated schematic from BOM and design intent.

        Args:
            bom: List of component dictionaries with part_number, category, etc.
            design_intent: Natural language description of circuit function
            connections: Optional explicit connections between components
            block_diagram: Optional hierarchical block structure
            design_name: Name for the output schematic
            reference_images: Optional reference design images for comparison
            skip_validation: Skip the MAPO validation loop (for testing)
            ideation_context: Optional structured ideation context extracted
                from ideation artifacts via build_ideation_context().
                When provided, enriches every pipeline phase with structured
                data (BOM hints, explicit connections, placement hints,
                validation criteria) instead of relying solely on text.
            resume_from_checkpoint: If True, try to load checkpoint and skip
                completed phases. Saves hours of LLM token re-generation.

        Returns:
            PipelineResult with schematic and validation info
        """
        start_time = datetime.now()
        result = PipelineResult(success=False)

        # Initialize checkpoint manager using progress emitter's operation_id (if available)
        op_id = getattr(self._progress, 'operation_id', None) or f"pipeline-{os.getpid()}"
        self._checkpoint_mgr = CheckpointManager(op_id)

        # --- RESUME LOGIC: Load checkpoint and determine which phases to skip ---
        _resumed_checkpoint: Optional[Dict[str, Any]] = None
        _completed_phases: List[str] = []
        if resume_from_checkpoint:
            _resumed_checkpoint = self._checkpoint_mgr.load_checkpoint()
            if _resumed_checkpoint:
                _completed_phases = _resumed_checkpoint.get("completed_phases", [])
                resume_count = self._checkpoint_mgr.increment_resume_count()
                logger.info(
                    f"RESUMING from checkpoint: completed_phases={_completed_phases}, "
                    f"resume_attempt={resume_count}"
                )
                self._emit_progress(
                    SchematicPhase.SYMBOLS, 0,
                    f"Resuming from checkpoint (skipping: {', '.join(_completed_phases)})",
                    event_type="checkpoint_resume",
                    resumed_phases=_completed_phases,
                    resume_count=resume_count,
                )

                if resume_count > 3:
                    logger.warning(
                        f"Max resume attempts ({resume_count}) exceeded — starting fresh"
                    )
                    self._checkpoint_mgr.cleanup()
                    _resumed_checkpoint = None
                    _completed_phases = []
            else:
                logger.info("Resume requested but no checkpoint found — running from scratch")

        try:
            # Ensure initialized
            if not self._symbol_fetcher:
                await self.initialize()

            # Emit initial progress
            self._start_phase(SchematicPhase.SYMBOLS, f"Starting schematic generation: {len(bom)} components")

            # Convert BOM dictionaries to BOMItem objects
            bom_items = [
                BOMItem(
                    part_number=item.get('part_number', item.get('mpn', 'UNKNOWN')),
                    manufacturer=item.get('manufacturer'),
                    reference=item.get('reference'),
                    quantity=item.get('quantity', 1),
                    category=item.get('category', 'Other'),
                    value=item.get('value', ''),
                    footprint=item.get('footprint', ''),
                    description=item.get('description', '')
                )
                for item in bom
            ]

            # FAIL-FAST: Reject empty BOM before wasting hours on LLM calls
            if not bom_items:
                raise SchematicGenerationError(
                    message=(
                        f"BOM is empty — cannot generate schematic with 0 components. "
                        f"Input bom had {len(bom)} raw entries but 0 converted to BOMItem objects."
                    ),
                    suggestion=(
                        "1. Ensure ideation artifacts contain component/BOM data\n"
                        "2. Verify subsystem names match expected values in create_foc_esc_bom()\n"
                        "3. Pass BOM explicitly via the 'bom' parameter"
                    ),
                )

            # ========================================================
            # Phase 2b: Resolve symbols BEFORE connection generation
            # This gives the connection generator access to actual pin
            # names instead of relying on LLM-inferred pin guesses.
            # ========================================================
            pre_resolved_symbols: Optional[Dict[str, Any]] = None
            component_pins: Optional[Dict[str, List[Dict]]] = None

            # Skip symbol pre-resolution if resuming from assembly checkpoint
            # (symbols were already resolved and assembled in the prior run)
            if "assembly" not in _completed_phases:
                self._start_phase(
                    SchematicPhase.SYMBOLS,
                    "Resolving component symbols for pin-accurate connections..."
                )
                self._emit_progress(
                    SchematicPhase.SYMBOLS, 10,
                    f"Fetching symbols for {len(bom_items)} components...",
                    event_type=SchematicEventType.SYMBOL_RESOLVING.value
                )

                try:
                    pre_resolved_symbols, component_pins = (
                        await self._assembler.resolve_symbols_only(bom_items)
                    )

                    pins_total = sum(len(p) for p in component_pins.values())
                    logger.info(
                        f"Phase 2b: Resolved {len(pre_resolved_symbols)} symbols, "
                        f"{len(component_pins)} components with pin data "
                        f"({pins_total} total pins available for connection validation)"
                    )

                    self._complete_phase(
                        SchematicPhase.SYMBOLS,
                        f"Resolved {len(pre_resolved_symbols)} symbols "
                        f"({pins_total} pins extracted for connection validation)"
                    )
                except Exception as e:
                    # Symbol pre-resolution failed — log warning but continue.
                    # The assembler's assemble_schematic() will re-resolve internally
                    # and connections will be generated without pin data (old behavior).
                    logger.warning(
                        f"Phase 2b symbol pre-resolution failed: {e}. "
                        f"Falling back to legacy flow (connections without pin data)."
                    )
                    pre_resolved_symbols = None
                    component_pins = None

                    self._emit_progress(
                        SchematicPhase.SYMBOLS, 100,
                        f"Symbol pre-resolution failed — falling back to legacy flow",
                        event_type=SchematicEventType.ERROR.value
                    )

            # Convert connections if provided, otherwise auto-generate
            connection_objs = None
            if connections:
                connection_objs = [
                    Connection(
                        from_ref=conn.get('from_ref', ''),
                        from_pin=conn.get('from_pin', ''),
                        to_ref=conn.get('to_ref', ''),
                        to_pin=conn.get('to_pin', ''),
                        net_name=conn.get('net_name')
                    )
                    for conn in connections
                ]
                logger.info(f"Using {len(connection_objs)} explicit connections")
                self._emit_progress(
                    SchematicPhase.CONNECTIONS, 100,
                    f"Using {len(connection_objs)} explicit connections"
                )
            elif "connections" in _completed_phases and _resumed_checkpoint:
                # RESUME: Restore connections from checkpoint (skip LLM re-generation)
                checkpoint_data = _resumed_checkpoint.get("data", {})
                saved_connections = checkpoint_data.get("connections", [])
                if saved_connections:
                    connection_objs = [
                        Connection(
                            from_ref=c.get("from_ref", ""),
                            from_pin=c.get("from_pin", ""),
                            to_ref=c.get("to_ref", ""),
                            to_pin=c.get("to_pin", ""),
                            net_name=c.get("net_name"),
                        )
                        for c in saved_connections
                    ]
                    logger.info(
                        f"RESUMED {len(connection_objs)} connections from checkpoint "
                        f"(skipped LLM generation — saved hours of tokens)"
                    )
                    self._emit_progress(
                        SchematicPhase.CONNECTIONS, 100,
                        f"Restored {len(connection_objs)} connections from checkpoint",
                        event_type="checkpoint_resume",
                    )
                else:
                    logger.warning(
                        "Checkpoint says connections complete but no data found — regenerating"
                    )
                    _completed_phases.remove("connections")
                    # Fall through to normal generation below

            if connection_objs is None and "connections" not in _completed_phases:
                # Auto-generate connections from BOM and design intent
                self._start_phase(SchematicPhase.CONNECTIONS, "Generating connections via LLM...")
                logger.info("Auto-generating connections from BOM and design intent...")

                self._emit_progress(
                    SchematicPhase.CONNECTIONS, 20,
                    "Calling Claude Opus 4.6 for signal inference...",
                    event_type=SchematicEventType.CONNECTIONS_LLM_CALL.value
                )

                # Extract ideation connection hints (explicit connections, interfaces, power rails)
                seed_connections = None
                if ideation_context and ideation_context.has_connection_hints:
                    seed_connections = ideation_context.connection_inference
                    ci = seed_connections
                    logger.info(
                        f"IDEATION SEED DATA: {len(ci.explicit_connections)} explicit connections, "
                        f"{len(ci.interfaces)} interfaces, {len(ci.power_rails)} power rails, "
                        f"{len(ci.ground_nets)} ground nets, {len(ci.critical_signals)} critical signals. "
                        f"Passing to connection generator as seed_connections."
                    )
                else:
                    logger.warning(
                        "NO ideation connection hints available — connection generator will rely "
                        "entirely on LLM inference (lower quality). Ideation context: "
                        f"{'present but no hints' if ideation_context else 'None'}"
                    )

                try:
                    generated_connections = await retry_phase(
                        self._connection_generator.generate_connections,
                        bom=bom,
                        design_intent=design_intent,
                        component_pins=component_pins,
                        seed_connections=seed_connections,
                        max_retries=1,  # Single retry: each LLM call takes 30-80min via proxy
                        base_delay=10.0,
                        phase_name="connections",
                    )
                    connection_objs = [
                        Connection(
                            from_ref=gc.from_ref,
                            from_pin=gc.from_pin,
                            to_ref=gc.to_ref,
                            to_pin=gc.to_pin,
                            net_name=gc.net_name
                        )
                        for gc in generated_connections
                    ]
                    logger.info(f"Auto-generated {len(connection_objs)} connections")

                    # Compute connection coverage: fraction of BOM components with at least one connection
                    connected_refs = set()
                    for c in connection_objs:
                        connected_refs.add(c.from_ref)
                        connected_refs.add(c.to_ref)
                    # Use component_pins keys (reference designators like U1, R1, C1)
                    # because BOM dicts from create_foc_esc_bom() lack 'reference' keys
                    if component_pins:
                        bom_refs = set(component_pins.keys())
                    else:
                        bom_refs = {comp.get("reference", "") for comp in bom if comp.get("reference")}
                    if bom_refs:
                        matched = connected_refs & bom_refs
                        result.connection_coverage = len(matched) / len(bom_refs)
                    else:
                        # Fallback: if no ref designators available, use ratio of connected components
                        result.connection_coverage = min(1.0, len(connected_refs) / max(1, len(bom)))
                    logger.info(
                        f"Connection coverage: {result.connection_coverage:.0%} "
                        f"({len(connected_refs & bom_refs) if bom_refs else len(connected_refs)}/{len(bom_refs) if bom_refs else len(bom)} BOM components connected)"
                    )

                    self._complete_phase(
                        SchematicPhase.CONNECTIONS,
                        f"Generated {len(connection_objs)} connections"
                    )

                    # Checkpoint: save connections so we can skip re-generation on resume
                    try:
                        if hasattr(self, '_checkpoint_mgr') and self._checkpoint_mgr:
                            self._checkpoint_mgr.save_checkpoint(
                                phase="connections",
                                data={
                                    "connections": [
                                        {"from_ref": c.from_ref, "from_pin": c.from_pin,
                                         "to_ref": c.to_ref, "to_pin": c.to_pin,
                                         "net_name": c.net_name}
                                        for c in connection_objs
                                    ]
                                },
                                completed_phases=["connections"],
                            )
                    except Exception as cp_err:
                        logger.warning(f"Checkpoint save (connections) failed: {cp_err}")
                except Exception as e:
                    # CRITICAL: LLM connection generation failed
                    error_msg = (
                        f"CRITICAL: LLM connection generation FAILED: {e}\n"
                        f"  Check: Claude proxy connectivity, rate limits, prompt format."
                    )
                    logger.error(error_msg)
                    result.errors.append(f"LLM Connection Generation Failed: {str(e)}")

                    # Emit error event so frontend shows clear failure message
                    self._emit_progress(
                        SchematicPhase.CONNECTIONS, 100,
                        f"ERROR: Connection generation failed - {str(e)}",
                        event_type=SchematicEventType.ERROR.value
                    )

                    # FALLBACK: Try to build connections from seed data (ideation hints)
                    # This gives us at least power connections and explicit signal paths
                    connection_objs = []
                    if seed_connections and hasattr(seed_connections, 'explicit_connections'):
                        try:
                            from agents.connection_generator.connection_generator_agent import (
                                ConnectionGeneratorAgent,
                            )
                            fallback_gen = ConnectionGeneratorAgent()
                            components = fallback_gen._extract_component_info(
                                bom, component_pins
                            )
                            # Generate rule-based power + bypass connections
                            power_conns = fallback_gen._generate_power_connections(components)
                            bypass_conns = fallback_gen._generate_bypass_cap_connections(components)
                            seed_conns = fallback_gen._convert_seed_connections(
                                seed_connections, components, component_pins
                            )
                            all_fallback = power_conns + bypass_conns + seed_conns
                            connection_objs = [
                                Connection(
                                    from_ref=gc.from_ref,
                                    from_pin=gc.from_pin,
                                    to_ref=gc.to_ref,
                                    to_pin=gc.to_pin,
                                    net_name=gc.net_name,
                                )
                                for gc in all_fallback
                            ]
                            logger.warning(
                                f"FALLBACK: Using {len(connection_objs)} connections "
                                f"from seed data + rule-based generation "
                                f"({len(power_conns)} power, {len(bypass_conns)} bypass, "
                                f"{len(seed_conns)} seed)"
                            )
                            result.errors.append(
                                f"Using {len(connection_objs)} fallback connections "
                                f"(no LLM signal connections)"
                            )
                        except Exception as fallback_err:
                            logger.error(f"Seed connection fallback also failed: {fallback_err}")
                            result.errors.append(f"Seed fallback failed: {fallback_err}")
                    else:
                        logger.error(
                            "No seed connections available for fallback — "
                            "schematic will have ZERO wires"
                        )

            # Convert block diagram if provided
            block_diagram_obj = None
            if block_diagram:
                block_diagram_obj = BlockDiagram(
                    blocks=block_diagram.get('blocks', {}),
                    connections=block_diagram.get('connections', [])
                )

            logger.info(f"Starting schematic generation: {len(bom_items)} components")

            # Initialize sheets (may be populated by assembly or left empty on resume)
            sheets: List[SchematicSheet] = []

            # --- RESUME: If assembly already completed, skip to output generation ---
            _skip_to_output = False
            _resumed_schematic_content: Optional[str] = None
            if "assembly" in _completed_phases and _resumed_checkpoint:
                checkpoint_data = _resumed_checkpoint.get("data", {})
                _resumed_schematic_content = checkpoint_data.get("schematic_content")
                if _resumed_schematic_content and len(_resumed_schematic_content) > 100:
                    _skip_to_output = True
                    logger.info(
                        f"RESUMED schematic from checkpoint ({len(_resumed_schematic_content)} chars) "
                        f"— skipping symbol resolution, assembly, and layout phases"
                    )
                    self._emit_progress(
                        SchematicPhase.ASSEMBLY, 100,
                        "Restored assembled schematic from checkpoint",
                        event_type="checkpoint_resume",
                    )
                else:
                    logger.warning("Checkpoint says assembly complete but no schematic_content — re-assembling")

            if not _skip_to_output:
                # Phase 1: Assemble schematic (uses Enhanced Wire Router internally)
                self._start_phase(SchematicPhase.ASSEMBLY, "Assembling schematic structure...")
                sheets = await self._assembler.assemble_schematic(
                    bom=bom_items,
                    block_diagram=block_diagram_obj,
                    connections=connection_objs,
                    design_name=design_name,
                    pre_resolved_symbols=pre_resolved_symbols,
                )

                result.sheets = sheets
                self._complete_phase(
                    SchematicPhase.ASSEMBLY,
                    f"Assembled {len(sheets)} sheet(s) with {len(sheets[0].symbols) if sheets else 0} components"
                )

            # ========================================
            # VALIDATION GATE: Symbol Quality Check
            # (Skipped when resuming from assembly checkpoint)
            # ========================================
            if not _skip_to_output:
                # This gate prevents saving schematics with all-placeholder symbols
                if sheets and sheets[0].symbols:
                    placeholder_symbols = [
                        s for s in sheets[0].symbols
                        if s.quality == SymbolQuality.PLACEHOLDER
                    ]
                    valid_symbols = [
                        s for s in sheets[0].symbols
                        if s.quality != SymbolQuality.PLACEHOLDER
                    ]

                    # Log symbol resolution summary
                    logger.info(f"Symbol Resolution Summary:")
                    logger.info(f"  Total components: {len(sheets[0].symbols)}")
                    logger.info(f"  Valid symbols: {len(valid_symbols)}")
                    logger.info(f"  Placeholder symbols: {len(placeholder_symbols)}")

                    # Track statistics
                    for symbol in sheets[0].symbols:
                        if symbol.quality == SymbolQuality.VERIFIED:
                            result.symbols_fetched += 1
                        elif symbol.quality == SymbolQuality.CACHED:
                            result.symbols_from_cache += 1
                        elif symbol.quality in (SymbolQuality.LLM_GENERATED, SymbolQuality.PLACEHOLDER):
                            result.symbols_generated += 1

                    # Compute placeholder ratio for quality tracking
                    total_symbols = len(sheets[0].symbols)
                    placeholder_count = len(placeholder_symbols)
                    if total_symbols > 0:
                        result.placeholder_ratio = placeholder_count / total_symbols
                        if result.placeholder_ratio > 0.3:
                            logger.warning(
                                f"HIGH PLACEHOLDER RATIO: {result.placeholder_ratio:.0%} "
                                f"({placeholder_count}/{total_symbols}) symbols are placeholders"
                            )

                    # CRITICAL: Fail if ALL symbols are placeholders
                    if len(valid_symbols) == 0 and len(placeholder_symbols) > 0:
                        failed_components = [
                            {
                                "reference": s.reference,
                                "part_number": s.part_number,
                                "error": s.resolution_error or "All symbol sources failed"
                            }
                            for s in placeholder_symbols
                        ]

                        error_msg = (
                            f"Symbol resolution failed for all {len(bom_items)} components. "
                            f"No valid symbols could be fetched from any source."
                        )
                        logger.error(error_msg)

                        raise SchematicGenerationError(
                            message=error_msg,
                            failed_components=failed_components,
                            validation_errors=[
                                f"Tried sources: KiCad Worker, Local Cache, GitHub, SnapEDA, LLM",
                                f"All {len(placeholder_symbols)} components returned placeholders"
                            ],
                            suggestion=(
                                "1. Verify KICAD_WORKER_URL is accessible from this pod\n"
                                "2. Check that mapos-kicad-worker service is running\n"
                                "3. Configure OPENROUTER_API_KEY or ANTHROPIC_API_KEY for LLM fallback"
                            )
                        )

                    # WARN if any symbols are placeholders (but still proceed)
                    if len(placeholder_symbols) > 0:
                        logger.warning(
                            f"{len(placeholder_symbols)} of {len(sheets[0].symbols)} components "
                            f"are using placeholder symbols and need manual resolution"
                        )
                        for s in placeholder_symbols:
                            error_msg = f"Placeholder symbol for {s.reference} ({s.part_number})"
                            result.errors.append(error_msg)
                            logger.warning(f"  - {s.reference} ({s.part_number}): {s.resolution_error or 'Unknown'}")
                else:
                    # No symbols at all - this is also an error
                    raise SchematicGenerationError(
                        message="Schematic assembly produced no symbols",
                        validation_errors=["BOM may be empty or all components filtered out"],
                        suggestion="Verify BOM contains valid component entries"
                    )

            # Phase 2: Layout Optimization (IPC-2221/IEEE 315 signal flow)
            # (Skipped when resuming from assembly checkpoint)
            if not _skip_to_output and self._layout_optimizer:
                self._start_phase(SchematicPhase.LAYOUT, "Running layout optimization...")
                logger.info("Running layout optimization...")

                # Extract placement hints from ideation context
                placement_hints = ideation_context.placement if ideation_context else None
                if placement_hints:
                    logger.info(
                        f"Using placement hints: {len(placement_hints.subsystem_blocks)} subsystems, "
                        f"signal_flow={placement_hints.signal_flow_direction}"
                    )

                for i, sheet in enumerate(sheets):
                    self._emit_progress(
                        SchematicPhase.LAYOUT,
                        int((i / len(sheets)) * 80),
                        f"Optimizing layout for sheet {i + 1}/{len(sheets)}...",
                        event_type=SchematicEventType.LAYOUT_OPTIMIZING.value
                    )
                    optimization_result = self._layout_optimizer.optimize_layout(
                        symbols=sheet.symbols,
                        connections=connection_objs if connection_objs else [],
                        bom=bom,
                        placement_hints=placement_hints
                    )
                    # Apply optimized positions
                    for comp_ref, position in optimization_result.optimized_positions.items():
                        for symbol in sheet.symbols:
                            if symbol.reference == comp_ref:
                                symbol.position = position
                                break
                    success_str = "PASSED" if optimization_result.success else "needs work"
                    logger.info(f"Layout optimization: {success_str}, {len(optimization_result.improvements)} improvements")
                self._complete_phase(
                    SchematicPhase.LAYOUT,
                    f"Layout optimized: {len(optimization_result.improvements) if optimization_result else 0} improvements"
                )

                # Checkpoint: save after layout (positions updated)
                try:
                    if self._checkpoint_mgr:
                        self._checkpoint_mgr.save_checkpoint(
                            phase="layout",
                            data={"layout_complete": True},
                            completed_phases=["connections", "assembly", "layout"],
                        )
                except Exception as cp_err:
                    logger.warning(f"Checkpoint save (layout) failed: {cp_err}")

            # Phase 3: Standards Compliance Check (IEC 60750/IEEE 315)
            # (Skipped when resuming from assembly checkpoint)
            if not _skip_to_output and self._standards_compliance:
                logger.info("Running standards compliance check...")
                for sheet in sheets:
                    compliance_report = self._standards_compliance.validate(
                        sheet=sheet,
                        bom=bom,
                        auto_fix=True  # Auto-fix violations where possible
                    )
                    if not compliance_report.passed:
                        logger.warning(f"Standards compliance: {len(compliance_report.violations)} violations")
                        for violation in compliance_report.violations[:5]:
                            logger.warning(f"  - {violation.check.value}: {violation.description}")
                    else:
                        logger.info(f"Standards compliance: PASSED ({compliance_report.score:.1%})")

            # Generate output file
            output_path = self.config.output_dir / f"{design_name}.kicad_sch"

            if _skip_to_output and _resumed_schematic_content:
                # RESUME: Write checkpoint schematic directly to output file
                schematic_content = _resumed_schematic_content
                output_path.write_text(schematic_content)
                result.schematic_path = output_path
                logger.info(
                    f"Wrote resumed schematic to {output_path} "
                    f"({len(schematic_content)} chars from checkpoint)"
                )
            else:
                schematic_content = await self._assembler.generate_kicad_sch(sheets[0])
                output_path.write_text(schematic_content)
                result.schematic_path = output_path
                logger.info(f"Schematic assembled: {output_path}")

            # Validate: symbol instance count in output matches BOM
            import re as _re
            symbol_instances = len(_re.findall(r'\(symbol \(lib_id', schematic_content))
            bom_count = len(bom_items)
            if symbol_instances < bom_count * 0.8:
                logger.error(
                    f"OUTPUT VALIDATION FAILED: Only {symbol_instances}/{bom_count} symbol instances "
                    f"in generated schematic (< 80% threshold). Some BOM components were not placed."
                )
                if not hasattr(self, '_pipeline_errors'):
                    self._pipeline_errors = []
                self._pipeline_errors.append(
                    f"Component count validation: {symbol_instances}/{bom_count} placed"
                )
            else:
                logger.info(
                    f"OUTPUT VALIDATION: {symbol_instances}/{bom_count} symbol instances in schematic"
                )

            # Checkpoint: save assembled schematic (most critical artifact)
            # Include connection data so resume from assembly also has connections available
            try:
                if hasattr(self, '_checkpoint_mgr') and self._checkpoint_mgr:
                    checkpoint_data = {
                        "schematic_content": schematic_content,
                        "output_path": str(output_path),
                    }
                    # Preserve connection data for potential re-use on resume
                    if connection_objs:
                        checkpoint_data["connections"] = [
                            {"from_ref": c.from_ref, "from_pin": c.from_pin,
                             "to_ref": c.to_ref, "to_pin": c.to_pin,
                             "net_name": c.net_name}
                            for c in connection_objs
                        ]
                    self._checkpoint_mgr.save_checkpoint(
                        phase="assembly",
                        data=checkpoint_data,
                        completed_phases=["connections", "assembly"],
                    )
            except Exception as cp_err:
                logger.warning(f"Checkpoint save (assembly) failed: {cp_err}")

            # Phase 3.5: Smoke Test (LLM-based circuit validation)
            # Ensures circuit won't "smoke" when power is applied
            if self._smoke_test:
                self._start_phase(SchematicPhase.SMOKE_TEST, "Running circuit validation (smoke test)...")
                logger.info("Running LLM-based smoke test (circuit validation)...")

                self._emit_progress(
                    SchematicPhase.SMOKE_TEST, 30,
                    "Analyzing circuit topology...",
                    event_type=SchematicEventType.SMOKE_TEST_RUNNING.value
                )

                smoke_result = await retry_phase(
                    self._smoke_test.run_smoke_test,
                    kicad_sch_content=schematic_content,
                    bom_items=bom,
                    power_sources=None,  # Auto-detect from schematic
                    max_retries=3,
                    base_delay=10.0,
                    phase_name="smoke_test",
                )
                result.smoke_test_result = smoke_result
                result.smoke_test_passed = smoke_result.passed

                if smoke_result.passed:
                    logger.info("Smoke test: PASSED - circuit appears safe to power")
                    if self._progress:
                        self._progress.emit_smoke_test(passed=True)
                else:
                    fatal_count = sum(1 for i in smoke_result.issues
                                     if i.severity.value == "fatal")
                    error_count = sum(1 for i in smoke_result.issues
                                     if i.severity.value == "error")
                    logger.warning(
                        f"Smoke test: FAILED - {fatal_count} fatal issues, {error_count} errors"
                    )
                    for issue in smoke_result.issues[:5]:
                        logger.warning(f"  [{issue.severity.value.upper()}] {issue.message}")

                    if self._progress:
                        self._progress.emit_smoke_test(
                            passed=False,
                            issues_count=fatal_count + error_count
                        )

                    # Add smoke test issues to result errors
                    for issue in smoke_result.issues:
                        if issue.severity.value in ("fatal", "error"):
                            result.errors.append(f"[Smoke Test] {issue.message}")

            # Phase 4: MAPO Functional Validation (Competitive Multi-Agent)
            if not skip_validation and self._functional_validator:
                logger.info("Starting MAPO competitive functional validation...")

                functional_result = await self._functional_validator.validate(
                    schematic_sexp=schematic_content,
                    design_intent=design_intent,
                    bom=bom,
                    connections=[vars(c) for c in connection_objs] if connection_objs else []
                )

                if not functional_result.passed:
                    logger.warning(
                        f"Functional validation: FAILED (score: {functional_result.overall_score:.1%})"
                    )
                    if functional_result.veto_triggered:
                        logger.error(f"VETO: {functional_result.veto_reason}")
                    for issue in functional_result.critical_issues[:3]:
                        logger.warning(f"  - {issue.category.value}: {issue.message}")
                else:
                    logger.info(
                        f"Functional validation: PASSED (score: {functional_result.overall_score:.1%})"
                    )

            # Phase 5: Enhanced Visual Validation Loop with kicad-worker integration
            if not skip_validation and self._visual_validator:
                logger.info("Starting ENHANCED dual-LLM visual validation loop...")
                logger.info("CRITICAL: Using kicad-worker for image extraction (NO FALLBACKS)")

                # Initialize enhanced components
                # Save every iteration's image for debugging and Opus analysis review
                iteration_output_dir = output_path.parent / "validation_iterations"
                os.makedirs(iteration_output_dir, exist_ok=True)

                image_extractor = SchematicImageExtractor(
                    kicad_worker_url=os.environ.get('KICAD_WORKER_URL', 'http://mapos-kicad-worker:8080'),
                    dpi=300,
                    save_to_disk=True,
                    output_dir=iteration_output_dir,
                )

                progress_tracker = ProgressTracker(
                    stagnation_threshold=0.02,
                    max_stagnant_iterations=3,
                )

                issue_transformer = IssueToFixTransformer(
                    max_fixes=5,
                    min_confidence=0.6,
                    use_llm=True,
                )

                fix_applicator = SchematicFixApplicator(
                    validate_syntax=True,
                )

                # Create enhanced validation loop
                visual_loop = ValidationLoop(
                    validator=self._visual_validator,
                    image_extractor=image_extractor,
                    progress_tracker=progress_tracker,
                    issue_transformer=issue_transformer,
                    fix_applicator=fix_applicator,
                    target_score=self.config.validation_threshold,
                    max_iterations=self.config.max_iterations,
                    max_stagnant_iterations=3,
                    max_fixes_per_iteration=5,
                )

                try:
                    # Run enhanced visual validation loop
                    loop_result = await visual_loop.run(
                        schematic_path=str(output_path),
                        schematic_content=schematic_content,
                        specification=design_intent,
                        on_iteration=lambda i, r, p: self._emit_progress(
                            SchematicPhase.VISUAL_VALIDATION,
                            int(50 + (i / self.config.max_iterations) * 50),
                            f"Iteration {i}: score={r.combined_score:.1%}, "
                            f"progress={p.progress_score:.1%}" if p else f"Iteration {i}"
                        )
                    )

                    result.iterations = loop_result.iterations

                    # Update schematic content if fixes were applied
                    if loop_result.fixes_applied:
                        schematic_content = output_path.read_text(encoding='utf-8')
                        logger.info(f"Applied {len(loop_result.fixes_applied)} fixes during validation")

                    # Store visual validation score in result
                    result.visual_score = loop_result.final_score

                    if loop_result.final_passed:
                        logger.info(
                            f"Visual validation: PASSED after {loop_result.iterations} iterations "
                            f"(score: {loop_result.final_score:.1%})"
                        )
                    else:
                        logger.warning(
                            f"Visual validation: Did not reach target "
                            f"(final: {loop_result.final_score:.1%}, target: {self.config.validation_threshold:.1%})"
                        )

                except ImageExtractionError as e:
                    logger.error(f"IMAGE EXTRACTION FAILED - NO FALLBACK")
                    logger.error(str(e))
                    raise  # Re-raise to fail the pipeline

                except StagnationError as e:
                    logger.error(f"VALIDATION STAGNATION DETECTED")
                    logger.error(str(e))
                    # Capture best visual score achieved before stagnation
                    result.visual_score = progress_tracker.best_score
                    logger.info(f"Visual score (before stagnation): {result.visual_score:.1%}")
                    # Continue with current schematic, but log the issue
                    result.errors.append(f"Validation stagnated: {e.reason.value}")

                except (ValueError, RuntimeError, OSError, json.JSONDecodeError,
                        asyncio.TimeoutError, ConnectionError) as e:
                    logger.warning(
                        "WARNING: Visual validation FAILED (non-fatal) and was SKIPPED. "
                        "The schematic was already assembled and can be used, but visual "
                        "quality was NOT validated by LLM. "
                        "Error type: %s, Error: %s",
                        type(e).__name__, e
                    )
                    # Capture best visual score achieved before error
                    if progress_tracker and progress_tracker.best_score > 0:
                        result.visual_score = progress_tracker.best_score
                        logger.info(f"Visual score (before error): {result.visual_score:.1%}")
                    result.errors.append(f"Visual validation skipped: {type(e).__name__}: {e}")
                    # Continue with current schematic — visual validation is
                    # an improvement step, not a gate. The schematic was already
                    # assembled and can be used without validation.

            # Legacy validation (fallback if new validators unavailable)
            elif not skip_validation and self._validator:
                logger.info("Starting legacy MAPO validation loop...")

                # Render schematic to image for validation
                schematic_image = await self._render_schematic(schematic_content)

                if schematic_image:
                    # Run validation
                    report = await self._validator.validate_schematic(
                        schematic_image=schematic_image,
                        design_intent=design_intent,
                        reference_images=reference_images,
                        iteration=0
                    )

                    result.validation_report = report
                    result.iterations = 1

                    # If not passed, run MAPO loop
                    if not report.passed and self.config.max_iterations > 1:
                        final_content, final_report = await self._run_mapo_loop(
                            schematic_content,
                            design_intent,
                            reference_images,
                            report
                        )

                        # Update output
                        output_path.write_text(final_content)
                        result.validation_report = final_report
                        result.iterations = final_report.iteration + 1

                    logger.info(
                        f"Validation complete: score={report.overall_score:.2%}, "
                        f"passed={report.passed}"
                    )
                else:
                    logger.warning("Could not render schematic for validation")

            result.success = True

            # Phase 6: Auto-export to PDF/image and sync to NFS
            if self._artifact_exporter and result.schematic_path:
                self._start_phase(SchematicPhase.EXPORT, "Exporting schematic artifacts...")
                logger.info("Starting auto-export to PDF/image and NFS sync...")

                self._emit_progress(
                    SchematicPhase.EXPORT, 20,
                    "Generating PDF export...",
                    event_type=SchematicEventType.EXPORT_PDF.value
                )

                export_result = await self._artifact_exporter.export_all(
                    schematic_path=result.schematic_path,
                    project_id=self.config.project_id,
                    design_name=design_name,
                )

                result.export_result = export_result
                result.pdf_path = export_result.pdf_path
                result.svg_path = export_result.svg_path
                result.png_path = export_result.png_path
                result.nfs_synced = export_result.nfs_synced
                result.nfs_paths = export_result.nfs_paths

                if export_result.success:
                    logger.info(f"Auto-export complete:")
                    if export_result.pdf_path:
                        self._emit_progress(
                            SchematicPhase.EXPORT, 50,
                            f"PDF exported: {export_result.pdf_path}",
                            event_type=SchematicEventType.EXPORT_PDF.value
                        )
                        logger.info(f"  PDF: {export_result.pdf_path}")
                    if export_result.svg_path:
                        self._emit_progress(
                            SchematicPhase.EXPORT, 70,
                            f"SVG exported: {export_result.svg_path}",
                            event_type=SchematicEventType.EXPORT_SVG.value
                        )
                        logger.info(f"  SVG: {export_result.svg_path}")
                    if export_result.png_path:
                        logger.info(f"  PNG: {export_result.png_path}")
                    if export_result.nfs_synced:
                        self._emit_progress(
                            SchematicPhase.EXPORT, 90,
                            f"Synced to NFS storage",
                            event_type=SchematicEventType.EXPORT_NFS_SYNC.value
                        )
                        logger.info(f"  NFS synced: {export_result.nfs_paths}")
                    self._complete_phase(SchematicPhase.EXPORT, "Export complete")
                else:
                    logger.warning(f"Auto-export had errors: {export_result.errors}")
                    result.errors.extend(export_result.errors)
                    self._emit_progress(
                        SchematicPhase.EXPORT, 100,
                        f"Export completed with errors"
                    )

        except SchematicGenerationError as e:
            # Handle our custom validation errors with full details
            logger.error(f"Schematic generation failed: {e.message}")
            if e.failed_components:
                logger.error(f"  Failed components: {len(e.failed_components)}")
            if e.suggestion:
                logger.info(f"  Suggestion: {e.suggestion}")
            result.errors.append(str(e))
            result.success = False

            # Emit error progress
            if self._progress:
                self._progress.error(e.message, "VALIDATION_ERROR")

        except Exception as e:
            logger.error(f"Pipeline error: {e}", exc_info=True)
            result.errors.append(str(e))
            result.success = False

            # Emit error progress
            if self._progress:
                self._progress.error(str(e), "PIPELINE_ERROR")

        finally:
            result.total_time_seconds = (datetime.now() - start_time).total_seconds()

            # Propagate _pipeline_errors to result.errors (these were previously lost)
            if hasattr(self, '_pipeline_errors') and self._pipeline_errors:
                for pe in self._pipeline_errors:
                    if pe not in result.errors:
                        result.errors.append(pe)
                logger.warning(
                    f"Propagated {len(self._pipeline_errors)} pipeline errors to result"
                )

            # Clean up checkpoint on success (no longer needed)
            if result.success and self._checkpoint_mgr:
                try:
                    self._checkpoint_mgr.cleanup()
                    logger.info("Pipeline succeeded — checkpoint cleaned up")
                except Exception:
                    pass  # Non-critical

            # Emit completion if successful
            if result.success and self._progress:
                self._progress.complete({
                    "schematic_path": str(result.schematic_path) if result.schematic_path else None,
                    "component_count": sum(len(s.symbols) for s in result.sheets),
                    "total_time_seconds": result.total_time_seconds,
                    "smoke_test_passed": result.smoke_test_passed,
                    "nfs_synced": result.nfs_synced,
                })

        return result

    async def _render_schematic(self, schematic_content: str) -> Optional[bytes]:
        """Render schematic to PNG image for validation."""
        try:
            # Try using KiCad CLI for rendering
            kicad_cli = self._find_kicad_cli()
            if kicad_cli:
                return await self._render_with_kicad_cli(schematic_content, kicad_cli)

            # Fallback: Try using Puppeteer/KiCanvas
            return await self._render_with_kicanvas(schematic_content)

        except Exception as e:
            logger.warning(f"Schematic rendering failed: {e}")
            return None

    def _find_kicad_cli(self) -> Optional[str]:
        """Find KiCad CLI executable."""
        possible_paths = [
            "/Applications/KiCad/KiCad.app/Contents/MacOS/kicad-cli",
            "/usr/bin/kicad-cli",
            "/usr/local/bin/kicad-cli",
            "kicad-cli",
        ]

        for path in possible_paths:
            if os.path.exists(path) or path == "kicad-cli":
                return path

        return None

    async def _render_with_kicad_cli(
        self,
        schematic_content: str,
        kicad_cli: str
    ) -> Optional[bytes]:
        """Render using KiCad CLI."""
        with tempfile.NamedTemporaryFile(suffix=".kicad_sch", delete=False) as f:
            f.write(schematic_content.encode())
            sch_path = f.name

        png_path = sch_path.replace(".kicad_sch", ".png")

        try:
            proc = await asyncio.create_subprocess_exec(
                kicad_cli, "sch", "export", "svg",
                "-o", png_path.replace(".png", ".svg"),
                sch_path,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await proc.communicate()

            # Convert SVG to PNG if needed
            svg_path = png_path.replace(".png", ".svg")
            if os.path.exists(svg_path):
                # Try cairosvg for conversion
                try:
                    import cairosvg
                    cairosvg.svg2png(url=svg_path, write_to=png_path)
                except ImportError:
                    logger.warning("cairosvg not available for SVG to PNG conversion")
                    return None

            if os.path.exists(png_path):
                with open(png_path, "rb") as f:
                    return f.read()

        except Exception as e:
            logger.warning(f"KiCad CLI render failed: {e}")

        finally:
            # Cleanup
            for path in [sch_path, png_path, png_path.replace(".png", ".svg")]:
                try:
                    os.unlink(path)
                except OSError:
                    pass

        return None

    async def _render_with_kicanvas(self, schematic_content: str) -> Optional[bytes]:
        """Render using KiCanvas via headless browser."""
        try:
            # This would use Puppeteer/Playwright to render via KiCanvas
            # For now, return None to indicate rendering not available
            logger.info("KiCanvas rendering not implemented yet")
            return None

        except Exception as e:
            logger.warning(f"KiCanvas render failed: {e}")
            return None

    async def _run_mapo_loop(
        self,
        initial_content: str,
        design_intent: str,
        reference_images: Optional[List[bytes]],
        initial_report: SchematicValidationReport
    ) -> Tuple[str, SchematicValidationReport]:
        """Run the MAPO iterative refinement loop."""
        current_content = initial_content
        current_report = initial_report

        for iteration in range(1, self.config.max_iterations):
            logger.info(f"MAPO iteration {iteration + 1}/{self.config.max_iterations}")

            # Apply fixes from report
            if current_report.recommended_fixes:
                current_content = await self._apply_fixes(
                    current_content,
                    current_report.recommended_fixes
                )

            # Re-render and validate
            schematic_image = await self._render_schematic(current_content)
            if not schematic_image:
                logger.warning("Could not render schematic for re-validation")
                break

            current_report = await self._validator.validate_schematic(
                schematic_image=schematic_image,
                design_intent=design_intent,
                reference_images=reference_images,
                iteration=iteration
            )

            logger.info(
                f"Iteration {iteration + 1}: "
                f"score={current_report.overall_score:.2%}, "
                f"passed={current_report.passed}"
            )

            if current_report.passed:
                break

            if not current_report.recommended_fixes:
                logger.info("No more fixes available")
                break

        return current_content, current_report

    async def _apply_fixes(
        self,
        schematic_content: str,
        fixes: List[Dict]
    ) -> str:
        """Apply recommended fixes to schematic content."""
        modified_content = schematic_content

        for fix in fixes:
            fix_type = fix.get('fix_type', '')
            kicad_action = fix.get('kicad_action', '')

            if not kicad_action:
                continue

            try:
                # Apply the fix based on type
                if fix_type == 'add_component':
                    # Parse and add component
                    pass
                elif fix_type == 'modify_connection':
                    # Modify wire connections
                    pass
                elif fix_type == 'change_value':
                    # Update component value
                    issue_ref = fix.get('issue_ref', '')
                    if issue_ref:
                        # Find and update the component value
                        import re
                        pattern = rf'(\(property "Value" ")[^"]*(" \(at [^)]+\)[^)]*\))'
                        # This is a simplified fix - real implementation would be more sophisticated
                        pass
                elif fix_type == 'add_label':
                    # Add net label
                    pass

            except Exception as e:
                logger.warning(f"Failed to apply fix: {e}")

        return modified_content

    async def close(self):
        """Clean up resources."""
        if self._symbol_fetcher:
            await self._symbol_fetcher.close()

        if self._graphrag_indexer:
            await self._graphrag_indexer.close()


# Convenience function for simple usage
async def generate_schematic(
    bom: List[Dict],
    design_intent: str,
    connections: Optional[List[Dict]] = None,
    design_name: str = "schematic",
    skip_validation: bool = False
) -> PipelineResult:
    """
    Simple function to generate a validated schematic.

    Args:
        bom: List of component dictionaries
        design_intent: Description of circuit function
        connections: Optional net connections
        design_name: Name for output file
        skip_validation: Skip MAPO validation

    Returns:
        PipelineResult with schematic path and validation info
    """
    pipeline = MAPOSchematicPipeline()
    try:
        result = await pipeline.generate(
            bom=bom,
            design_intent=design_intent,
            connections=connections,
            design_name=design_name,
            skip_validation=skip_validation
        )
        return result
    finally:
        await pipeline.close()


# CLI entry point
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="MAPO Schematic Generation Pipeline")
    parser.add_argument("--bom", type=str, help="Path to BOM JSON file")

    # Design intent: support both direct string (--intent) and file (--intent-file)
    intent_group = parser.add_mutually_exclusive_group(required=True)
    intent_group.add_argument("--intent", type=str, help="Design intent description (direct string)")
    intent_group.add_argument("--intent-file", type=str, help="Path to file containing design intent (avoids E2BIG errors)")

    parser.add_argument("--output", type=str, default="schematic", help="Output name")
    parser.add_argument("--skip-validation", action="store_true", help="Skip MAPO validation")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")

    # Export options
    parser.add_argument("--no-export", action="store_true", help="Skip auto-export to PDF/image")
    parser.add_argument("--nfs-path", type=str, default="/Volumes/Nexus/plugins/ee-design-plugin/artifacts",
                        help="NFS share base path for artifacts")
    parser.add_argument("--project-id", type=str, help="Project ID for organizing artifacts")
    parser.add_argument("--no-pdf", action="store_true", help="Skip PDF export")
    parser.add_argument("--no-svg", action="store_true", help="Skip SVG export")
    parser.add_argument("--no-png", action="store_true", help="Skip PNG export")

    args = parser.parse_args()

    # Configure logging
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    )

    async def main():
        # Load BOM if provided
        if args.bom:
            with open(args.bom) as f:
                bom = json.load(f)
        else:
            # Default test BOM
            bom = [
                {"part_number": "STM32G431CBT6", "category": "MCU", "value": "STM32G431"},
                {"part_number": "DRV8323RS", "category": "Gate_Driver", "value": "DRV8323"},
                {"part_number": "CSD19505KCS", "category": "MOSFET", "value": "80V MOSFET"},
                {"part_number": "100uF", "category": "Capacitor", "value": "100uF/50V"},
                {"part_number": "10uF", "category": "Capacitor", "value": "10uF/25V"},
                {"part_number": "0.1uF", "category": "Capacitor", "value": "0.1uF"},
                {"part_number": "10k", "category": "Resistor", "value": "10k"},
                {"part_number": "1k", "category": "Resistor", "value": "1k"},
            ]

        # Load design intent from file or direct argument
        if args.intent_file:
            with open(args.intent_file, 'r', encoding='utf-8') as f:
                design_intent = f.read()
        else:
            design_intent = args.intent

        # Create pipeline config with export settings
        config = PipelineConfig(
            auto_export=not args.no_export,
            export_pdf=not args.no_pdf,
            export_svg=not args.no_svg,
            export_png=not args.no_png,
            nfs_base_path=args.nfs_path,
            project_id=args.project_id,
        )

        print(f"\n{'='*60}")
        print("MAPO Schematic Generation Pipeline")
        print(f"{'='*60}")
        print(f"Design Intent: {design_intent[:200]}..." if len(design_intent) > 200 else f"Design Intent: {design_intent}")
        print(f"Components: {len(bom)}")
        print(f"Skip Validation: {args.skip_validation}")
        print(f"Auto-Export: {config.auto_export}")
        print(f"NFS Path: {config.nfs_base_path}")
        print(f"{'='*60}\n")

        # Use custom pipeline with export config
        pipeline = MAPOSchematicPipeline(config)
        try:
            result = await pipeline.generate(
                bom=bom,
                design_intent=design_intent,
                design_name=args.output,
                skip_validation=args.skip_validation
            )
        finally:
            await pipeline.close()

        print(f"\n{'='*60}")
        print("RESULTS")
        print(f"{'='*60}")
        print(f"Success: {result.success}")
        print(f"Output: {result.schematic_path}")
        print(f"Symbols Fetched: {result.symbols_fetched}")
        print(f"  - From Cache: {result.symbols_from_cache}")
        print(f"  - Generated: {result.symbols_generated}")
        print(f"Iterations: {result.iterations}")
        print(f"Total Time: {result.total_time_seconds:.2f}s")

        if result.validation_report:
            print(f"\nValidation Score: {result.validation_report.overall_score:.2%}")
            print(f"Validation Passed: {result.validation_report.passed}")
            print(f"Critical Issues: {len(result.validation_report.critical_issues)}")

        # Show export results
        print(f"\nExport Artifacts:")
        if result.pdf_path:
            print(f"  PDF: {result.pdf_path}")
        else:
            print(f"  PDF: (not exported)")
        if result.svg_path:
            print(f"  SVG: {result.svg_path}")
        else:
            print(f"  SVG: (not exported)")
        if result.png_path:
            print(f"  PNG: {result.png_path}")
        else:
            print(f"  PNG: (not exported)")

        print(f"\nNFS Sync: {'SYNCED' if result.nfs_synced else 'NOT SYNCED'}")
        if result.nfs_paths:
            for artifact_type, path in result.nfs_paths.items():
                print(f"  {artifact_type}: {path}")

        if result.errors:
            print(f"\nErrors:")
            for error in result.errors:
                print(f"  - {error}")

        print(f"{'='*60}\n")

    asyncio.run(main())
