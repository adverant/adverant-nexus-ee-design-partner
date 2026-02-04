"""
Progress Emitter for WebSocket Streaming

Emits PROGRESS:{json} lines to stdout for Node.js to relay via WebSocket.
Follows the pattern established for all EE design services.

Usage:
    emitter = ProgressEmitter()
    emitter.emit("symbol_resolving", 10, "Resolving symbol: STM32G431")
    emitter.emit_phase("symbols", 15, "Found 5 of 10 symbols")
    emitter.complete({"schematic_id": "abc123"})
    emitter.error("Failed to connect to KiCad Worker")
"""

import json
import sys
from datetime import datetime, timezone
from typing import Any, Callable, Dict, Optional
from dataclasses import dataclass, field
from enum import Enum


class SchematicEventType(str, Enum):
    """Event types for schematic generation (matches TypeScript enum)."""
    # Phase lifecycle
    PHASE_START = "phase_start"
    PHASE_COMPLETE = "phase_complete"

    # Symbol resolution
    SYMBOL_RESOLVING = "symbol_resolving"
    SYMBOL_RESOLVED = "symbol_resolved"
    SYMBOL_FROM_MEMORY = "symbol_from_memory"
    SYMBOL_FROM_CACHE = "symbol_from_cache"
    SYMBOL_FROM_KICAD = "symbol_from_kicad"
    SYMBOL_FROM_SNAPEDA = "symbol_from_snapeda"
    SYMBOL_PLACEHOLDER = "symbol_placeholder"

    # Connection generation
    CONNECTIONS_GENERATING = "connections_generating"
    CONNECTIONS_LLM_CALL = "connections_llm_call"
    CONNECTIONS_GENERATED = "connections_generated"

    # Layout optimization
    LAYOUT_OPTIMIZING = "layout_optimizing"
    LAYOUT_ZONE_ASSIGNMENT = "layout_zone_assignment"
    LAYOUT_COMPLETE = "layout_complete"

    # Wire routing
    WIRING_START = "wiring_start"
    WIRING_NET = "wiring_net"
    WIRING_PROGRESS = "wiring_progress"
    WIRING_COMPLETE = "wiring_complete"

    # Assembly
    ASSEMBLY_START = "assembly_start"
    ASSEMBLY_PROGRESS = "assembly_progress"
    ASSEMBLY_COMPLETE = "assembly_complete"

    # Smoke test
    SMOKE_TEST_START = "smoke_test_start"
    SMOKE_TEST_RUNNING = "smoke_test_running"
    SMOKE_TEST_RESULT = "smoke_test_result"

    # Gaming AI (optimization)
    GAMING_AI_START = "gaming_ai_start"
    GAMING_AI_ITERATION = "gaming_ai_iteration"
    GAMING_AI_MUTATION = "gaming_ai_mutation"
    GAMING_AI_FITNESS = "gaming_ai_fitness"
    GAMING_AI_COMPLETE = "gaming_ai_complete"

    # Validation
    VALIDATION_START = "validation_start"
    VALIDATION_RUNNING = "validation_running"
    VALIDATION_RESULT = "validation_result"

    # Export
    EXPORT_START = "export_start"
    EXPORT_PDF = "export_pdf"
    EXPORT_SVG = "export_svg"
    EXPORT_PNG = "export_png"
    EXPORT_NFS_SYNC = "export_nfs_sync"
    EXPORT_COMPLETE = "export_complete"

    # Final states
    COMPLETE = "complete"
    ERROR = "error"


class SchematicPhase(str, Enum):
    """Phases in schematic generation (matches frontend phases)."""
    SYMBOLS = "symbols"
    CONNECTIONS = "connections"
    LAYOUT = "layout"
    WIRING = "wiring"
    ASSEMBLY = "assembly"
    SMOKE_TEST = "smoke_test"
    GAMING_AI = "gaming_ai"
    EXPORT = "export"


# Phase progress ranges for overall progress calculation
PHASE_PROGRESS_RANGES: Dict[SchematicPhase, tuple] = {
    SchematicPhase.SYMBOLS: (0, 25),
    SchematicPhase.CONNECTIONS: (25, 40),
    SchematicPhase.LAYOUT: (40, 55),
    SchematicPhase.WIRING: (55, 70),
    SchematicPhase.ASSEMBLY: (70, 80),
    SchematicPhase.SMOKE_TEST: (80, 90),
    SchematicPhase.GAMING_AI: (90, 95),
    SchematicPhase.EXPORT: (95, 100),
}


def calculate_overall_progress(phase: SchematicPhase, phase_progress: int) -> int:
    """
    Calculate overall progress from phase and phase-specific progress.

    Args:
        phase: Current phase
        phase_progress: Progress within current phase (0-100)

    Returns:
        Overall progress (0-100)
    """
    start, end = PHASE_PROGRESS_RANGES.get(phase, (0, 100))
    range_size = end - start
    return start + int((range_size * phase_progress) / 100)


@dataclass
class ProgressEmitter:
    """
    Emits progress events to stdout in PROGRESS:{json} format.

    Node.js parses these lines and relays them via WebSocket.
    """
    operation_id: str = ""
    current_phase: Optional[SchematicPhase] = None
    _total_events: int = field(default=0, init=False)

    def emit(
        self,
        event_type: str,
        progress: int,
        message: str,
        phase: Optional[str] = None,
        phase_progress: Optional[int] = None,
        **extra_data
    ) -> None:
        """
        Emit a progress event.

        Args:
            event_type: Type of event (from SchematicEventType)
            progress: Overall progress (0-100)
            message: Human-readable message
            phase: Current phase name
            phase_progress: Progress within current phase (0-100)
            **extra_data: Additional event-specific data
        """
        event = {
            "type": event_type,
            "operationId": self.operation_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "progress_percentage": max(0, min(100, progress)),
            "current_step": message,
        }

        if phase:
            event["phase"] = phase
        elif self.current_phase:
            event["phase"] = self.current_phase.value

        if phase_progress is not None:
            event["phase_progress"] = max(0, min(100, phase_progress))

        # Add any extra data
        if extra_data:
            event["data"] = extra_data

        self._emit_line(event)

    def emit_phase(
        self,
        phase: SchematicPhase,
        phase_progress: int,
        message: str,
        event_type: Optional[str] = None,
        **extra_data
    ) -> None:
        """
        Emit progress event with automatic overall progress calculation.

        Args:
            phase: Current phase
            phase_progress: Progress within phase (0-100)
            message: Human-readable message
            event_type: Event type (defaults to phase progress)
            **extra_data: Additional data
        """
        self.current_phase = phase
        overall_progress = calculate_overall_progress(phase, phase_progress)

        self.emit(
            event_type=event_type or f"{phase.value}_progress",
            progress=overall_progress,
            message=message,
            phase=phase.value,
            phase_progress=phase_progress,
            **extra_data
        )

    def start_phase(self, phase: SchematicPhase, message: Optional[str] = None) -> None:
        """Mark the start of a new phase."""
        self.current_phase = phase
        overall_progress = calculate_overall_progress(phase, 0)

        self.emit(
            event_type=SchematicEventType.PHASE_START.value,
            progress=overall_progress,
            message=message or f"Starting {phase.value.replace('_', ' ')}...",
            phase=phase.value,
            phase_progress=0,
        )

    def complete_phase(self, phase: SchematicPhase, message: Optional[str] = None) -> None:
        """Mark the completion of a phase."""
        overall_progress = calculate_overall_progress(phase, 100)

        self.emit(
            event_type=SchematicEventType.PHASE_COMPLETE.value,
            progress=overall_progress,
            message=message or f"Completed {phase.value.replace('_', ' ')}",
            phase=phase.value,
            phase_progress=100,
        )

    def emit_symbol_progress(
        self,
        current: int,
        total: int,
        part_number: str,
        source: str,
        success: bool = True,
    ) -> None:
        """Emit symbol resolution progress."""
        phase_progress = int((current / max(1, total)) * 100)

        if success:
            event_type = {
                "memory": SchematicEventType.SYMBOL_FROM_MEMORY,
                "cache": SchematicEventType.SYMBOL_FROM_CACHE,
                "kicad": SchematicEventType.SYMBOL_FROM_KICAD,
                "snapeda": SchematicEventType.SYMBOL_FROM_SNAPEDA,
            }.get(source.lower(), SchematicEventType.SYMBOL_RESOLVED)
        else:
            event_type = SchematicEventType.SYMBOL_PLACEHOLDER

        self.emit_phase(
            phase=SchematicPhase.SYMBOLS,
            phase_progress=phase_progress,
            message=f"{'Resolved' if success else 'Placeholder'}: {part_number} ({current}/{total})",
            event_type=event_type.value,
            component_name=part_number,
            component_index=current,
            total_components=total,
            source=source,
            success=success,
        )

    def emit_connection_progress(
        self,
        current: int,
        total: int,
        net_name: str,
    ) -> None:
        """Emit connection generation progress."""
        phase_progress = int((current / max(1, total)) * 100)

        self.emit_phase(
            phase=SchematicPhase.CONNECTIONS,
            phase_progress=phase_progress,
            message=f"Generated connection: {net_name} ({current}/{total})",
            event_type=SchematicEventType.CONNECTIONS_GENERATED.value,
            connections_generated=current,
            total_connections=total,
            net_name=net_name,
        )

    def emit_wiring_progress(
        self,
        wires_routed: int,
        total_wires: int,
        current_net: str = "",
    ) -> None:
        """Emit wire routing progress."""
        phase_progress = int((wires_routed / max(1, total_wires)) * 100)

        self.emit_phase(
            phase=SchematicPhase.WIRING,
            phase_progress=phase_progress,
            message=f"Routing wires: {wires_routed}/{total_wires}" + (f" ({current_net})" if current_net else ""),
            event_type=SchematicEventType.WIRING_PROGRESS.value,
            wires_routed=wires_routed,
            total_wires=total_wires,
            current_net=current_net,
        )

    def emit_gaming_ai_iteration(
        self,
        iteration: int,
        max_iterations: int,
        fitness: float,
        improvements: int = 0,
    ) -> None:
        """Emit Gaming AI optimization progress."""
        phase_progress = int((iteration / max(1, max_iterations)) * 100)

        self.emit_phase(
            phase=SchematicPhase.GAMING_AI,
            phase_progress=phase_progress,
            message=f"Optimization iteration {iteration}/{max_iterations} (fitness: {fitness:.3f})",
            event_type=SchematicEventType.GAMING_AI_ITERATION.value,
            iteration=iteration,
            max_iterations=max_iterations,
            fitness=fitness,
            improvements=improvements,
        )

    def emit_smoke_test(self, passed: bool, issues_count: int = 0) -> None:
        """Emit smoke test result."""
        self.emit_phase(
            phase=SchematicPhase.SMOKE_TEST,
            phase_progress=100,
            message=f"Smoke test: {'PASSED' if passed else 'FAILED'}" + (f" ({issues_count} issues)" if not passed else ""),
            event_type=SchematicEventType.SMOKE_TEST_RESULT.value,
            passed=passed,
            issues_count=issues_count,
        )

    def complete(self, result: Optional[Dict[str, Any]] = None) -> None:
        """
        Emit completion event.

        Args:
            result: Final result data to include
        """
        event = {
            "type": SchematicEventType.COMPLETE.value,
            "operationId": self.operation_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "progress_percentage": 100,
            "current_step": "Schematic generation complete!",
        }

        if result:
            event["data"] = {"result": result}

        self._emit_line(event)

    def error(self, message: str, error_code: str = "GENERATION_FAILED") -> None:
        """
        Emit error event.

        Args:
            message: Error message
            error_code: Error code for categorization
        """
        event = {
            "type": SchematicEventType.ERROR.value,
            "operationId": self.operation_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "progress_percentage": self._get_last_progress(),
            "current_step": f"Error: {message}",
            "error_message": message,
            "error_code": error_code,
        }

        if self.current_phase:
            event["phase"] = self.current_phase.value

        self._emit_line(event)

    def _emit_line(self, event: Dict[str, Any]) -> None:
        """Write PROGRESS: line to stdout."""
        self._total_events += 1
        line = f"PROGRESS:{json.dumps(event)}"
        print(line, flush=True)  # Ensure immediate output

    def _get_last_progress(self) -> int:
        """Get last emitted progress (for error events)."""
        if self.current_phase:
            return calculate_overall_progress(self.current_phase, 50)
        return 0


def create_progress_callback(
    emitter: ProgressEmitter
) -> Callable[[Dict[str, Any]], None]:
    """
    Create a progress callback function for use with MAPOSchematicPipeline.

    Args:
        emitter: ProgressEmitter instance

    Returns:
        Callback function that emits progress events
    """
    def callback(event: Dict[str, Any]) -> None:
        emitter.emit(
            event_type=event.get("type", "progress_update"),
            progress=event.get("progress_percentage", 0),
            message=event.get("current_step", "Processing..."),
            phase=event.get("phase"),
            phase_progress=event.get("phase_progress"),
            **{k: v for k, v in event.items()
               if k not in ("type", "progress_percentage", "current_step", "phase", "phase_progress")}
        )
    return callback


# Convenience functions for quick usage
_default_emitter: Optional[ProgressEmitter] = None


def init_progress(operation_id: str) -> ProgressEmitter:
    """Initialize the default progress emitter."""
    global _default_emitter
    _default_emitter = ProgressEmitter(operation_id=operation_id)
    return _default_emitter


def emit_progress(
    event_type: str,
    progress: int,
    message: str,
    **kwargs
) -> None:
    """Emit progress using the default emitter."""
    if _default_emitter:
        _default_emitter.emit(event_type, progress, message, **kwargs)
    else:
        # Fallback: just print the line without operation ID
        event = {
            "type": event_type,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "progress_percentage": progress,
            "current_step": message,
            **kwargs
        }
        print(f"PROGRESS:{json.dumps(event)}", flush=True)
