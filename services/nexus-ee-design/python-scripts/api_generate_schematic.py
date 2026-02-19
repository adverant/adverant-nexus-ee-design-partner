#!/usr/bin/env python3
"""
API Wrapper for MAPO Schematic Generation Pipeline.

This script provides a JSON-in, JSON-out interface for the TypeScript API
to call the MAPO schematic generation pipeline.

Usage:
    python api_generate_schematic.py --json '{
        "bom": [...],
        "design_intent": "...",
        "design_name": "...",
        "skip_validation": false
    }'

Or via stdin:
    echo '{"bom": [...], ...}' | python api_generate_schematic.py --stdin
"""

import asyncio
import argparse
import json
import logging
import os
import signal
import sys
import tempfile
from pathlib import Path
from typing import Dict, Any, List, Optional

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from mapo_schematic_pipeline import (
    MAPOSchematicPipeline,
    PipelineConfig,
    PipelineResult,
)
from progress_emitter import ProgressEmitter, init_progress
from ideation_context import IdeationContext
from ideation_extractors import build_ideation_context, ExtractionError
from ralph_loop_orchestrator import RalphLoopOrchestrator, RalphLoopResult
from checkpoint_manager import CheckpointManager

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    stream=sys.stderr  # Log to stderr, output JSON to stdout
)
logger = logging.getLogger(__name__)

# --- SIGTERM handler: save checkpoint and emit resumable progress before exit ---
_shutdown_event = asyncio.Event() if hasattr(asyncio, "Event") else None
_active_checkpoint_mgr: Optional["CheckpointManager"] = None
_active_output_path: Optional[Path] = None


def _sigterm_handler(signum: int, frame: Any) -> None:
    """Handle SIGTERM: save checkpoint, emit PROGRESS event, exit cleanly."""
    logger.warning("SIGTERM received — saving checkpoint and exiting gracefully")

    # Try to save a checkpoint with whatever we have on disk
    if _active_checkpoint_mgr and _active_output_path:
        try:
            # Check if assembly output already exists (most critical artifact)
            sch_files = list(_active_output_path.glob("*.kicad_sch"))
            if sch_files:
                content = sch_files[0].read_text()
                _active_checkpoint_mgr.save_checkpoint(
                    phase="interrupted",
                    data={"schematic_content": content, "output_path": str(sch_files[0])},
                    completed_phases=["assembly"],  # at minimum
                )
                logger.info(f"Checkpoint saved on SIGTERM: {sch_files[0]}")
        except Exception as exc:
            logger.error(f"Failed to save checkpoint on SIGTERM: {exc}")

    # Emit machine-readable progress so the TypeScript watchdog knows we saved state
    progress_msg = json.dumps({
        "type": "checkpoint_saved",
        "resumable": True,
        "progress_percentage": 0,
        "current_step": "Process interrupted — checkpoint saved",
    })
    print(f"PROGRESS:{progress_msg}", flush=True)

    sys.exit(0)


signal.signal(signal.SIGTERM, _sigterm_handler)


def create_foc_esc_bom(subsystems: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Create a comprehensive BOM for FOC ESC based on selected subsystems.

    This maps subsystem names to actual component definitions from the
    foc-esc-heavy-lift reference design.
    """
    # Component definitions by subsystem
    SUBSYSTEM_COMPONENTS = {
        "Power Input Stage": [
            {"part_number": "IPB200N25N3 G", "category": "MOSFET", "manufacturer": "Infineon", "value": "250V/66A N-FET", "description": "Reverse polarity protection"},
            {"part_number": "LTC4412ES6", "category": "IC", "manufacturer": "Analog Devices", "value": "Ideal Diode Controller", "description": "OR-ing controller"},
            {"part_number": "SL22_10010", "category": "Thermistor", "manufacturer": "Ametherm", "value": "10R NTC", "description": "Inrush limiter"},
            {"part_number": "744823347", "category": "Inductor", "manufacturer": "Wurth", "value": "470uH CM Choke", "description": "EMI filter"},
            {"part_number": "C5750X7R2A226M", "category": "Capacitor", "manufacturer": "TDK", "value": "22uF/100V", "quantity": 10, "description": "Bulk decoupling"},
            {"part_number": "FG28X7R1E222K", "category": "Capacitor", "manufacturer": "TDK", "value": "2.2nF Y1", "quantity": 2, "description": "Y-cap EMI"},
        ],
        "Gate Driver": [
            {"part_number": "UCC21530ADWRR", "category": "Gate_Driver", "manufacturer": "TI", "value": "6A/9A Isolated", "quantity": 6, "description": "Half-bridge driver"},
            {"part_number": "0603WAF4700T5E", "category": "Resistor", "manufacturer": "UniOhm", "value": "4.7R", "quantity": 12, "description": "Gate resistor"},
            {"part_number": "C2012X7R1E104K", "category": "Capacitor", "manufacturer": "TDK", "value": "100nF/25V", "quantity": 6, "description": "Bootstrap cap"},
            {"part_number": "C3216X7R1E106K", "category": "Capacitor", "manufacturer": "TDK", "value": "10uF/25V", "quantity": 6, "description": "VCC bypass"},
        ],
        "Power Stage": [
            {"part_number": "IMZA65R027M1H", "category": "MOSFET", "manufacturer": "Infineon", "value": "650V/31A SiC", "quantity": 6, "description": "Half-bridge switches"},
            {"part_number": "C5750X7R2A476M", "category": "Capacitor", "manufacturer": "TDK", "value": "47uF/100V", "quantity": 3, "description": "DC bus cap"},
            {"part_number": "1N4148WS", "category": "Diode", "manufacturer": "ON Semi", "value": "100V Fast", "quantity": 6, "description": "Flyback protection"},
        ],
        "MCU Core": [
            {"part_number": "STM32G431CBT6", "category": "MCU", "manufacturer": "ST", "value": "Cortex-M4 170MHz", "description": "Main controller"},
            {"part_number": "ABM8-8.000MHZ", "category": "Crystal", "manufacturer": "Abracon", "value": "8MHz", "description": "HSE crystal"},
            {"part_number": "C2012X7R1E105K", "category": "Capacitor", "manufacturer": "TDK", "value": "1uF/25V", "quantity": 4, "description": "VDD bypass"},
            {"part_number": "C2012X7R1E104K", "category": "Capacitor", "manufacturer": "TDK", "value": "100nF/25V", "quantity": 8, "description": "Decoupling"},
            {"part_number": "0603WAF1002T5E", "category": "Resistor", "manufacturer": "UniOhm", "value": "10k", "quantity": 4, "description": "Pullup/pulldown"},
        ],
        "Current Sensing": [
            {"part_number": "INA240A4PWR", "category": "Amplifier", "manufacturer": "TI", "value": "200V/V Gain", "quantity": 3, "description": "Current sense amp"},
            {"part_number": "WSL3637R0005FEA", "category": "Resistor", "manufacturer": "Vishay", "value": "0.5mR 3W", "quantity": 3, "description": "Shunt resistor"},
            {"part_number": "0603WAF1000T5E", "category": "Resistor", "manufacturer": "UniOhm", "value": "100R", "quantity": 6, "description": "RC filter"},
            {"part_number": "C2012X7R1E103K", "category": "Capacitor", "manufacturer": "TDK", "value": "10nF/25V", "quantity": 6, "description": "RC filter"},
        ],
        "Communication": [
            {"part_number": "TJA1051T/3", "category": "CAN_Transceiver", "manufacturer": "NXP", "value": "CAN FD", "description": "CAN transceiver"},
            {"part_number": "PRTR5V0U2X", "category": "TVS", "manufacturer": "Nexperia", "value": "ESD Protection", "description": "CAN ESD"},
            {"part_number": "0603WAF1200T5E", "category": "Resistor", "manufacturer": "UniOhm", "value": "120R", "description": "CAN termination"},
            {"part_number": "C2012X7R1E104K", "category": "Capacitor", "manufacturer": "TDK", "value": "100nF/25V", "quantity": 2, "description": "CAN bypass"},
            {"part_number": "USB4110-GF-A", "category": "Connector", "manufacturer": "GCT", "value": "USB-C", "description": "Debug connector"},
        ],
    }

    # Build BOM from selected subsystems
    bom = []
    for subsystem in subsystems:
        name = subsystem.get("name", "")
        if name in SUBSYSTEM_COMPONENTS:
            components = SUBSYSTEM_COMPONENTS[name]
            for comp in components:
                bom_item = comp.copy()
                bom_item["subsystem"] = name
                bom.append(bom_item)

    # Add power symbols if any subsystem selected
    if bom:
        bom.extend([
            {"part_number": "VCC", "category": "Power", "value": "VCC", "description": "Power rail"},
            {"part_number": "GND", "category": "Power", "value": "GND", "description": "Ground"},
        ])

    return bom


def create_design_intent(
    subsystems: List[Dict[str, Any]],
    project_name: str,
    ideation_artifacts: Optional[List[Dict[str, Any]]] = None
) -> str:
    """
    Create a detailed design intent from subsystem selection and ideation artifacts.

    If ideation artifacts are provided, they are used to enhance the design intent
    with detailed specifications, component rationale, and design calculations.
    """
    subsystem_names = [s.get("name", "Unknown") for s in subsystems]

    # Start with base intent
    intent_parts = [
        f"{project_name} - Schematic Design",
        "",
        "Selected Subsystems:",
        *[f"- {name}" for name in subsystem_names],
        "",
    ]

    # If we have ideation artifacts, use them to build comprehensive context
    if ideation_artifacts:
        # Group artifacts by category
        artifacts_by_category: Dict[str, List[Dict[str, Any]]] = {}
        for artifact in ideation_artifacts:
            category = artifact.get("category", "other")
            if category not in artifacts_by_category:
                artifacts_by_category[category] = []
            artifacts_by_category[category].append(artifact)

        # Add system overview first (if available)
        if "architecture" in artifacts_by_category:
            intent_parts.append("=== SYSTEM ARCHITECTURE ===")
            for artifact in artifacts_by_category["architecture"]:
                artifact_type = artifact.get("artifact_type", "")
                content = artifact.get("content", "")
                if artifact_type == "system_overview" and content:
                    intent_parts.append(f"\n{artifact.get('name', 'System Overview')}:")
                    # Include first 2000 chars of system overview
                    intent_parts.append(content[:2000])
                    if len(content) > 2000:
                        intent_parts.append("... [truncated]")
                elif artifact_type == "architecture_diagram" and content:
                    intent_parts.append(f"\nArchitecture Diagram ({artifact.get('name', '')}):")
                    intent_parts.append(content[:1000])
            intent_parts.append("")

        # Add schematic specifications
        if "schematic" in artifacts_by_category:
            intent_parts.append("=== SCHEMATIC SPECIFICATIONS ===")
            for artifact in artifacts_by_category["schematic"]:
                artifact_type = artifact.get("artifact_type", "")
                content = artifact.get("content", "")
                name = artifact.get("name", artifact_type)
                if content:
                    intent_parts.append(f"\n{name}:")
                    # Include schematic specs (important for connections)
                    intent_parts.append(content[:3000])
                    if len(content) > 3000:
                        intent_parts.append("... [truncated]")
            intent_parts.append("")

        # Add component selection / BOM context
        if "component" in artifacts_by_category:
            intent_parts.append("=== COMPONENT SELECTION ===")
            for artifact in artifacts_by_category["component"]:
                artifact_type = artifact.get("artifact_type", "")
                content = artifact.get("content", "")
                name = artifact.get("name", artifact_type)
                if content:
                    intent_parts.append(f"\n{name}:")
                    intent_parts.append(content[:2000])
                    if len(content) > 2000:
                        intent_parts.append("... [truncated]")
            intent_parts.append("")

        # Add any calculations
        has_calculations = any(
            a.get("artifact_type") == "calculations"
            for artifacts in artifacts_by_category.values()
            for a in artifacts
        )
        if has_calculations:
            intent_parts.append("=== DESIGN CALCULATIONS ===")
            for category_artifacts in artifacts_by_category.values():
                for artifact in category_artifacts:
                    if artifact.get("artifact_type") == "calculations":
                        content = artifact.get("content", "")
                        if content:
                            intent_parts.append(f"\n{artifact.get('name', 'Calculations')}:")
                            intent_parts.append(content[:1500])
                            if len(content) > 1500:
                                intent_parts.append("... [truncated]")
            intent_parts.append("")

        logger.info(f"Enhanced design intent with {len(ideation_artifacts)} ideation artifacts")
    else:
        # Default FOC ESC design intent if no artifacts provided
        intent_parts.extend([
            "This is a Field Oriented Control (FOC) Electronic Speed Controller for brushless motors.",
            "",
            "Design Requirements:",
            "1. Power Input: 48-60V battery input with reverse polarity protection",
            "2. Gate Drivers: Isolated half-bridge drivers with bootstrap supply",
            "3. Power Stage: SiC MOSFETs for high-efficiency switching",
            "4. MCU Core: STM32G4 microcontroller with FOC algorithm",
            "5. Current Sensing: Low-side shunt measurement with differential amplifiers",
            "6. Communication: CAN bus interface for control commands",
            "",
            "Key Connections:",
            "- VCC power rail to all IC power pins",
            "- GND reference to all ground pins",
            "- Phase outputs from power stage to motor connector",
            "- Current sense outputs to MCU ADC inputs",
            "- PWM signals from MCU to gate drivers",
            "- CAN bus to external connector",
            "",
            "Layout Considerations:",
            "- Keep power stage and gate drivers close together",
            "- Minimize current sense loop area",
            "- Proper decoupling for MCU",
            "- EMI filtering on input power",
        ])

    return "\n".join(intent_parts)


async def run_generation(
    bom: Optional[List[Dict[str, Any]]] = None,
    subsystems: Optional[List[Dict[str, Any]]] = None,
    design_intent: Optional[str] = None,
    design_name: str = "schematic",
    project_name: str = "FOC ESC",
    skip_validation: bool = False,  # Enable visual validation by default
    output_dir: Optional[str] = None,
    project_id: Optional[str] = None,  # Project ID for NFS export organization
    auto_export: bool = True,  # Enable auto-export to PDF/image and NFS
    operation_id: Optional[str] = None,  # Operation ID for WebSocket streaming
    ideation_artifacts: Optional[List[Dict[str, Any]]] = None,  # Ideation artifacts for context
    enable_ralph_loop: bool = False,  # Enable Ralph Loop iterative improvement
    ralph_max_iterations: int = 100,  # Max Ralph Loop iterations
    ralph_target_score: float = 100.0,  # Target score (0-100)
    ralph_plateau_threshold: int = 20,  # Stop if no improvement for N iterations
) -> Dict[str, Any]:
    """
    Run the MAPO schematic generation pipeline.

    Args:
        bom: Explicit BOM list (optional, will be generated from subsystems if not provided)
        subsystems: List of subsystem definitions to include
        design_intent: Natural language design description
        design_name: Name for output file
        project_name: Project name for title block
        skip_validation: Skip MAPO validation loop
        output_dir: Custom output directory
        project_id: Project ID for NFS export organization
        auto_export: Enable auto-export to PDF/image and NFS
        operation_id: Operation ID for WebSocket streaming
        ideation_artifacts: List of ideation artifacts to provide context for schematic generation

    Returns:
        Dictionary with generation results
    """
    global _active_checkpoint_mgr, _active_output_path

    try:
        # Create progress emitter for WebSocket streaming (if operation_id provided)
        progress_emitter = None
        if operation_id:
            progress_emitter = init_progress(operation_id)
            logger.info(f"Progress streaming enabled for operation: {operation_id}")

        # Build structured IdeationContext from artifacts (replaces create_design_intent)
        ideation_context: Optional[IdeationContext] = None
        if ideation_artifacts:
            logger.info(f"Building IdeationContext from {len(ideation_artifacts)} ideation artifacts")
            for artifact in ideation_artifacts:
                logger.info(f"  - {artifact.get('name', 'unknown')} ({artifact.get('artifact_type', 'unknown')})")
            try:
                ideation_context = await build_ideation_context(
                    raw_artifacts=ideation_artifacts,
                    subsystems=subsystems or [],
                    project_name=project_name,
                )
                logger.info(
                    f"IdeationContext built: bom={ideation_context.has_bom_artifacts}, "
                    f"connections={ideation_context.has_connection_hints}, "
                    f"placement={ideation_context.has_placement_hints}, "
                    f"validation={ideation_context.has_validation_criteria}"
                )
            except ExtractionError as exc:
                logger.error(f"IdeationContext extraction failed: {exc}")
                return {
                    "success": False,
                    "error": f"Ideation artifact extraction failed: {exc}",
                    "schematic_content": "",
                    "component_count": 0,
                    "errors": [str(exc)],
                }

        # Generate BOM from subsystems if not provided
        if bom is None and subsystems:
            bom = create_foc_esc_bom(subsystems)
            logger.info(f"Generated BOM with {len(bom)} components from {len(subsystems)} subsystems")
        elif bom is None:
            bom = []
            logger.warning("No BOM or subsystems provided, generating minimal schematic")

        # Use IdeationContext design_intent_text for backward compat, or legacy create_design_intent
        if design_intent is None:
            if ideation_context:
                design_intent = ideation_context.design_intent_text
            else:
                design_intent = create_design_intent(subsystems or [], project_name, ideation_artifacts)

        # Configure output directory
        if output_dir:
            output_path = Path(output_dir)
        else:
            output_path = Path(tempfile.mkdtemp(prefix="mapo_schematic_"))

        output_path.mkdir(parents=True, exist_ok=True)

        # Register checkpoint manager for SIGTERM handler
        checkpoint_mgr = CheckpointManager(operation_id or f"gen-{os.getpid()}")
        _active_checkpoint_mgr = checkpoint_mgr
        _active_output_path = output_path

        logger.info(f"Starting schematic generation: {len(bom)} components")
        logger.info(f"Output directory: {output_path}")

        # Configure pipeline with auto-export to PDF/image and NFS
        # nfs_base_path uses ARTIFACT_STORAGE_PATH env var (defaults to /data/artifacts in K8s)
        config = PipelineConfig(
            output_dir=output_path,
            validation_threshold=0.85,
            max_iterations=3 if not skip_validation else 1,
            # Enable auto-export to PDF/image and NFS sync
            auto_export=auto_export,
            export_pdf=True,
            export_svg=True,
            export_png=True,
            project_id=project_id,
        )

        # Choose between Ralph Loop (iterative improvement) or traditional single-pass pipeline
        if enable_ralph_loop:
            # Ralph Loop: Iterative improvement orchestrator
            logger.info(f"Using Ralph Loop orchestrator (max_iterations={ralph_max_iterations}, target={ralph_target_score}%)")

            orchestrator = RalphLoopOrchestrator(
                max_iterations=ralph_max_iterations,
                target_score=ralph_target_score,
                plateau_threshold=ralph_plateau_threshold,
            )

            try:
                ralph_result: RalphLoopResult = await orchestrator.run(
                    bom=bom,
                    design_intent=design_intent,
                    project_name=project_name,
                    design_type="foc_esc",
                    connections=None,  # Will be auto-generated
                    ideation_context=ideation_context,
                    operation_id=operation_id,
                )

                # Convert RalphLoopResult to PipelineResult format
                result = PipelineResult(
                    success=ralph_result.success,
                    schematic_path=Path(ralph_result.final_schematic_path) if ralph_result.final_schematic_path else None,
                    iterations=ralph_result.total_iterations,
                    total_time_seconds=ralph_result.total_duration_seconds,
                    errors=ralph_result.failure_analysis,
                )

                logger.info(
                    f"Ralph Loop completed: success={ralph_result.success}, "
                    f"iterations={ralph_result.total_iterations}, "
                    f"final_score={ralph_result.final_score:.1f}%"
                )

            finally:
                await orchestrator.close()
        else:
            # Traditional single-pass pipeline
            logger.info("Using traditional single-pass MAPO pipeline")

            pipeline = MAPOSchematicPipeline(config, progress_emitter=progress_emitter)
            try:
                result = await pipeline.generate(
                    bom=bom,
                    design_intent=design_intent,
                    design_name=design_name,
                    skip_validation=skip_validation,
                    ideation_context=ideation_context,
                )
            finally:
                await pipeline.close()

        # Read generated schematic content
        schematic_content = ""
        if result.schematic_path and result.schematic_path.exists():
            schematic_content = result.schematic_path.read_text()

        # Build response
        response = {
            "success": result.success,
            "schematic_path": str(result.schematic_path) if result.schematic_path else None,
            "schematic_content": schematic_content,
            "sheets": [
                {"name": s.name, "uuid": s.uuid, "component_count": len(s.symbols)}
                for s in result.sheets
            ],
            "component_count": sum(len(s.symbols) for s in result.sheets),
            "symbols_fetched": result.symbols_fetched,
            "symbols_from_cache": result.symbols_from_cache,
            "symbols_generated": result.symbols_generated,
            "iterations": result.iterations,
            "total_time_seconds": result.total_time_seconds,
            "errors": result.errors,
        }

        # Include validation results if available
        if result.validation_report:
            response["validation"] = {
                "overall_score": result.validation_report.overall_score,
                "passed": result.validation_report.passed,
                "critical_issues": len(result.validation_report.critical_issues),
            }

        # Include export results if auto-export was enabled
        if result.export_result:
            response["export"] = {
                "success": result.export_result.success,
                "pdf_path": str(result.pdf_path) if result.pdf_path else None,
                "svg_path": str(result.svg_path) if result.svg_path else None,
                "png_path": str(result.png_path) if result.png_path else None,
                "nfs_synced": result.nfs_synced,
                "nfs_paths": result.nfs_paths,
                "errors": result.export_result.errors if result.export_result.errors else [],
            }
            if result.nfs_synced:
                logger.info(f"Artifacts synced to NFS: {result.nfs_paths}")

        return response

    except Exception as e:
        logger.error(f"Generation failed: {e}", exc_info=True)

        # --- Partial result recovery ---
        # If assembly completed before the failure, return the schematic content
        # so the frontend can show a partial (but useful) result.
        partial_content = ""
        completed_phases: List[str] = []
        if _active_output_path:
            sch_files = list(_active_output_path.glob("*.kicad_sch"))
            if sch_files:
                try:
                    partial_content = sch_files[0].read_text()
                    completed_phases.append("assembly")
                    logger.info(f"Partial result recovered: {sch_files[0]} ({len(partial_content)} chars)")
                    # Also save checkpoint for potential future resume
                    if _active_checkpoint_mgr:
                        _active_checkpoint_mgr.save_checkpoint(
                            phase="partial_failure",
                            data={"schematic_content": partial_content, "error": str(e)},
                            completed_phases=completed_phases,
                        )
                except Exception as read_exc:
                    logger.warning(f"Failed to read partial result: {read_exc}")

        return {
            "success": False,
            "partial_success": bool(partial_content),
            "error": str(e),
            "schematic_content": partial_content,
            "completed_phases": completed_phases,
            "component_count": 0,
            "errors": [str(e)],
        }


def _patch_agent_providers(proxy_url: str) -> None:
    """
    Patch module-level OpenRouter constants in already-imported agent modules
    to route through the Claude Code Max proxy pod instead.

    This is needed because Python caches module imports, so env vars set after
    import don't affect module-level constants like OPENROUTER_BASE_URL.
    """
    proxy_chat_url = f"{proxy_url}/v1/chat/completions"

    # Patch schematic assembler
    try:
        import agents.schematic_assembler.assembler_agent as assembler_mod
        assembler_mod.OPENROUTER_BASE_URL = proxy_chat_url
        assembler_mod.OPENROUTER_API_KEY = "internal-proxy"
        logger.info("Patched schematic_assembler -> Claude Code Max proxy")
    except (ImportError, AttributeError) as e:
        logger.debug(f"Could not patch schematic_assembler: {e}")

    # Patch smoke test agent
    try:
        import agents.smoke_test.smoke_test_agent as smoke_mod
        smoke_mod.OPENROUTER_BASE_URL = proxy_chat_url
        smoke_mod.OPENROUTER_API_KEY = "internal-proxy"
        logger.info("Patched smoke_test_agent -> Claude Code Max proxy")
    except (ImportError, AttributeError) as e:
        logger.debug(f"Could not patch smoke_test_agent: {e}")

    # Patch visual validator (dual LLM)
    try:
        import agents.visual_validator.dual_llm_validator as dual_val_mod
        # dual_llm_validator uses instance variables, patched at init time via env var
        logger.info("Visual validator will use AI_PROVIDER env var at init time")
    except (ImportError, AttributeError) as e:
        logger.debug(f"Could not patch dual_llm_validator: {e}")

    # Patch issue_to_fix transformer
    try:
        import agents.visual_validator.issue_to_fix as fix_mod
        fix_mod.OPENROUTER_BASE_URL = f"{proxy_url}/v1"
        fix_mod.OPENROUTER_API_KEY = "internal-proxy"
        logger.info("Patched issue_to_fix -> Claude Code Max proxy")
    except (ImportError, AttributeError) as e:
        logger.debug(f"Could not patch issue_to_fix: {e}")

    # Patch symbol assembly
    try:
        import agents.symbol_assembly.symbol_assembler as sym_mod
        sym_mod.OPENROUTER_BASE_URL = proxy_chat_url
        sym_mod.OPENROUTER_API_KEY = "internal-proxy"
        logger.info("Patched symbol_assembler -> Claude Code Max proxy")
    except (ImportError, AttributeError) as e:
        logger.debug(f"Could not patch symbol_assembler: {e}")

    # Patch connection generator (already has native support, but patch for safety)
    try:
        import agents.connection_generator.connection_generator_agent as conn_mod
        conn_mod.LLM_BASE_URL = f"{proxy_url}/v1"
        conn_mod.AI_PROVIDER = "claude_code_max"
        logger.info("Patched connection_generator -> Claude Code Max proxy")
    except (ImportError, AttributeError) as e:
        logger.debug(f"Could not patch connection_generator: {e}")

    # Patch ideation extractors
    try:
        import ideation_extractors as ie_mod
        ie_mod.OPENROUTER_URL = proxy_chat_url
        ie_mod.OPENROUTER_MODEL = "anthropic/claude-opus-4-6"
        logger.info("Patched ideation_extractors -> Claude Code Max proxy")
    except (ImportError, AttributeError) as e:
        logger.debug(f"Could not patch ideation_extractors: {e}")


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="API Wrapper for MAPO Schematic Generation"
    )
    parser.add_argument(
        "--json",
        type=str,
        help="JSON input string with bom, design_intent, design_name, skip_validation"
    )
    parser.add_argument(
        "--stdin",
        action="store_true",
        help="Read JSON input from stdin"
    )
    parser.add_argument(
        "--pretty",
        action="store_true",
        help="Pretty print JSON output"
    )

    args = parser.parse_args()

    # Get input JSON
    if args.stdin:
        input_json = sys.stdin.read()
    elif args.json:
        input_json = args.json
    else:
        # Default test input
        input_json = json.dumps({
            "subsystems": [
                {"id": "1", "name": "MCU Core", "category": "Control"},
                {"id": "2", "name": "Current Sensing", "category": "Sensing"},
            ],
            "project_name": "Test FOC ESC",
            "design_name": "test_schematic",
            "skip_validation": False,
        })

    try:
        params = json.loads(input_json)
    except json.JSONDecodeError as e:
        result = {"success": False, "error": f"Invalid JSON input: {e}"}
        print(json.dumps(result))
        sys.exit(1)

    # Apply AI provider preference — default to claude_code_max (uses Max plan proxy)
    # Only falls back to openrouter if explicitly set to "openrouter"
    ai_provider = params.get("ai_provider", "claude_code_max")
    os.environ["AI_PROVIDER"] = ai_provider
    logger.info(f"AI provider: {ai_provider}")

    if ai_provider == "claude_code_max":
        proxy_url = os.environ.get(
            "CLAUDE_CODE_PROXY_URL",
            "http://claude-code-proxy.nexus.svc.cluster.local:3100"
        )
        logger.info(
            "Using Claude Code Max proxy pod for all LLM calls "
            f"(proxy URL: {proxy_url})"
        )
        # Patch module-level constants in already-imported agents
        # (Python caches imports, so env vars set after import don't affect module-level vars)
        _patch_agent_providers(proxy_url)

    # Run async generation
    result = asyncio.run(run_generation(
        bom=params.get("bom"),
        subsystems=params.get("subsystems"),
        design_intent=params.get("design_intent"),
        design_name=params.get("design_name", "schematic"),
        project_name=params.get("project_name", "FOC ESC"),
        skip_validation=params.get("skip_validation", False),
        output_dir=params.get("output_dir"),
        project_id=params.get("project_id"),
        auto_export=params.get("auto_export", True),
        operation_id=params.get("operation_id"),  # For WebSocket streaming
        ideation_artifacts=params.get("ideation_artifacts"),  # Ideation artifacts for context
        enable_ralph_loop=params.get("enable_ralph_loop", False),  # Ralph Loop iterative improvement
        ralph_max_iterations=params.get("ralph_max_iterations", 100),
        ralph_target_score=params.get("ralph_target_score", 100.0),
        ralph_plateau_threshold=params.get("ralph_plateau_threshold", 20),
    ))

    # Output JSON result to stdout
    indent = 2 if args.pretty else None
    print(json.dumps(result, indent=indent))

    # Exit with appropriate code
    sys.exit(0 if result.get("success", False) else 1)


if __name__ == "__main__":
    main()


# =============================================================================
# MAPO v2.1 Schematic Integration
# =============================================================================

async def run_generation_v2_1(
    bom: Optional[List[Dict[str, Any]]] = None,
    subsystems: Optional[List[Dict[str, Any]]] = None,
    design_intent: Optional[str] = None,
    design_name: str = "schematic",
    project_name: str = "FOC ESC",
    design_type: str = "foc_esc",
    max_iterations: int = 100,
    output_dir: Optional[str] = None,
    project_id: Optional[str] = None,
    ideation_artifacts: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """
    Run MAPO v3.0 schematic generation with:
    - Deep ideation artifact integration via IdeationContext
    - LLM-orchestrated Gaming AI optimization
    - Nexus-memory learning for symbols and wiring
    - Smoke test validation
    - Quality-diversity optimization via MAP-Elites + Red Queen

    Args:
        bom: Explicit BOM list
        subsystems: List of subsystem definitions (will be converted to BOM)
        design_intent: Natural language design description
        design_name: Name for output file
        project_name: Project name for title block
        design_type: Type of design for pattern matching
        max_iterations: Max Gaming AI iterations
        output_dir: Custom output directory
        project_id: Project ID for organization
        ideation_artifacts: List of ideation artifacts for structured context

    Returns:
        Dictionary with generation results
    """
    try:
        from mapos_v2_1_schematic import (
            SchematicMAPOOptimizer,
            SchematicMAPOConfig,
        )

        # Build structured IdeationContext from artifacts
        ideation_context: Optional[IdeationContext] = None
        if ideation_artifacts:
            logger.info(f"Building IdeationContext for v3.0 from {len(ideation_artifacts)} artifacts")
            try:
                ideation_context = await build_ideation_context(
                    raw_artifacts=ideation_artifacts,
                    subsystems=subsystems or [],
                    project_name=project_name,
                )
            except ExtractionError as exc:
                logger.error(f"IdeationContext extraction failed: {exc}")
                return {
                    "success": False,
                    "error": f"Ideation artifact extraction failed: {exc}",
                    "version": "3.0",
                    "errors": [str(exc)],
                }

        # Generate BOM from subsystems if not provided
        if bom is None and subsystems:
            bom = create_foc_esc_bom(subsystems)
            logger.info(f"Generated BOM with {len(bom)} components from {len(subsystems)} subsystems")
        elif bom is None:
            bom = []
            logger.warning("No BOM or subsystems provided")

        # Generate design intent: prefer IdeationContext, fallback to legacy
        if design_intent is None:
            if ideation_context:
                design_intent = ideation_context.design_intent_text
            else:
                design_intent = create_design_intent(subsystems or [], project_name)
        
        # Configure output directory
        if output_dir:
            output_path = Path(output_dir)
        else:
            output_path = Path(tempfile.mkdtemp(prefix="mapo_v2_1_schematic_"))
        
        output_path.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Starting MAPO v2.1 schematic generation: {len(bom)} components")
        logger.info(f"Output directory: {output_path}")
        
        # Create configuration
        config = SchematicMAPOConfig.from_env()
        config.output_dir = output_path
        config.project_id = project_id
        
        # Create optimizer and run
        optimizer = SchematicMAPOOptimizer(config)
        try:
            result = await optimizer.optimize(
                bom=bom,
                design_intent=design_intent,
                design_name=design_name,
                design_type=design_type,
                max_iterations=max_iterations,
                ideation_context=ideation_context,
            )
        finally:
            await optimizer.close()

        # Build response
        response = {
            "success": result.success,
            "version": "3.0",
            "schematic_path": str(result.schematic_path) if result.schematic_path else None,
            "schematic_content": result.schematic_content,
            "component_count": result.symbols_resolved,
            "symbols_from_memory": result.symbols_from_memory,
            "placeholders": result.placeholders,
            "connections_generated": result.connections_generated,
            "wires_routed": result.wires_routed,
            "total_iterations": result.total_iterations,
            "total_time_seconds": result.total_time_seconds,
            "final_fitness": result.final_fitness,
            "smoke_test_passed": result.smoke_test_passed,
            "errors": result.errors,
        }
        
        # Include state information if available
        if result.final_state:
            response["state"] = result.final_state.to_dict()
        
        return response
        
    except ImportError as e:
        logger.error(f"MAPO v2.1 not available: {e}")
        return {
            "success": False,
            "error": f"MAPO v2.1 not available: {e}",
            "version": "3.0",
        }
    except Exception as e:
        logger.error(f"MAPO v2.1 generation failed: {e}", exc_info=True)
        return {
            "success": False,
            "error": str(e),
            "version": "3.0",
            "errors": [str(e)],
        }


# Modified main to support v2.1
def main_v2():
    """CLI entry point with v2.1 support."""
    parser = argparse.ArgumentParser(
        description="API Wrapper for MAPO Schematic Generation (v2.1 support)"
    )
    parser.add_argument(
        "--json",
        type=str,
        help="JSON input string"
    )
    parser.add_argument(
        "--stdin",
        action="store_true",
        help="Read JSON input from stdin"
    )
    parser.add_argument(
        "--pretty",
        action="store_true",
        help="Pretty print JSON output"
    )
    parser.add_argument(
        "--v2",
        action="store_true",
        help="Use MAPO v2.1 pipeline"
    )
    
    args = parser.parse_args()
    
    # Get input JSON
    if args.stdin:
        input_json = sys.stdin.read()
    elif args.json:
        input_json = args.json
    else:
        input_json = json.dumps({
            "subsystems": [
                {"id": "1", "name": "MCU Core", "category": "Control"},
            ],
            "project_name": "Test",
            "design_name": "test_schematic",
        })
    
    try:
        params = json.loads(input_json)
    except json.JSONDecodeError as e:
        result = {"success": False, "error": f"Invalid JSON input: {e}"}
        print(json.dumps(result))
        sys.exit(1)
    
    # Check for v2.1 mode
    use_v2 = args.v2 or params.get("use_mapo_v2_1", False)
    
    if use_v2:
        result = asyncio.run(run_generation_v2_1(
            bom=params.get("bom"),
            subsystems=params.get("subsystems"),
            design_intent=params.get("design_intent"),
            design_name=params.get("design_name", "schematic"),
            project_name=params.get("project_name", "FOC ESC"),
            design_type=params.get("design_type", "foc_esc"),
            max_iterations=params.get("max_iterations", 100),
            output_dir=params.get("output_dir"),
            project_id=params.get("project_id"),
        ))
    else:
        result = asyncio.run(run_generation(
            bom=params.get("bom"),
            subsystems=params.get("subsystems"),
            design_intent=params.get("design_intent"),
            design_name=params.get("design_name", "schematic"),
            project_name=params.get("project_name", "FOC ESC"),
            skip_validation=params.get("skip_validation", False),
            output_dir=params.get("output_dir"),
            project_id=params.get("project_id"),
            auto_export=params.get("auto_export", True),
        ))
    
    indent = 2 if args.pretty else None
    print(json.dumps(result, indent=indent))
    sys.exit(0 if result.get("success", False) else 1)
