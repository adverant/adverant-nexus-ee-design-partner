"""
MAPO v2.1 Schematic - Main Optimizer Orchestrator

Coordinates all components for LLM-orchestrated Gaming AI schematic generation:
1. Symbol Resolution (memory-enhanced)
2. Connection Generation (memory-enhanced)
3. Wire Routing (EnhancedWireRouter)
4. Smoke Test Validation
5. Gaming AI Optimization (if needed)
6. Final Export

Philosophy: "Opus 4.6 Thinks, Gaming AI Explores, Algorithms Execute, Memory Learns"

Author: Nexus EE Design Team
"""

import asyncio
import logging
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Add parent paths for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from ideation_context import (
    IdeationContext,
    SymbolResolutionContext,
    ConnectionInferenceContext,
    PlacementContext,
    ValidationContext,
)

from agents.wire_router import EnhancedWireRouter
from agents.layout_optimizer import LayoutOptimizerAgent

from ..core.schematic_state import (
    SchematicState,
    SchematicSolution,
    ComponentInstance,
    Connection,
    Wire,
    WireSegment,
    Junction,
    ValidationResults,
    FitnessScores,
    SymbolQuality,
)
from ..core.config import SchematicMAPOConfig, get_config, set_config
from ..pipeline.symbol_resolution import MemoryEnhancedSymbolResolver
from ..pipeline.connection_inference import MemoryEnhancedConnectionGenerator
from ..validation.smoke_test_validator import SmokeTestValidator, SmokeTestValidationResult
from ..gaming_ai.llm_guided_map_elites import LLMGuidedSchematicMAPElites, MutationGuidance
from ..gaming_ai.llm_guided_red_queen import LLMGuidedRedQueen

logger = logging.getLogger(__name__)


@dataclass
class OptimizationResult:
    """Result from MAPO v2.1 schematic optimization."""
    success: bool
    final_state: Optional[SchematicState] = None
    schematic_path: Optional[Path] = None
    schematic_content: str = ""
    
    # Metrics
    total_iterations: int = 0
    total_time_seconds: float = 0.0
    final_fitness: float = 0.0
    smoke_test_passed: bool = False
    
    # Component resolution
    symbols_resolved: int = 0
    symbols_from_memory: int = 0
    placeholders: int = 0
    
    # Connection generation
    connections_generated: int = 0
    connections_from_patterns: int = 0
    wires_routed: int = 0
    
    # Errors
    errors: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "schematic_path": str(self.schematic_path) if self.schematic_path else None,
            "total_iterations": self.total_iterations,
            "total_time_seconds": self.total_time_seconds,
            "final_fitness": self.final_fitness,
            "smoke_test_passed": self.smoke_test_passed,
            "symbols_resolved": self.symbols_resolved,
            "symbols_from_memory": self.symbols_from_memory,
            "placeholders": self.placeholders,
            "connections_generated": self.connections_generated,
            "wires_routed": self.wires_routed,
            "errors": self.errors,
        }


class SchematicMAPOOptimizer:
    """
    MAPO v2.1 Schematic - Main Optimizer
    
    Pipeline:
    1. Symbol Resolution (memory-enhanced)
    2. Connection Generation (LLM-guided + memory patterns)
    3. Component Placement (Layout Optimizer)
    4. Wire Routing (EnhancedWireRouter)
    5. Smoke Test Validation
    6. Gaming AI Optimization (if smoke test fails)
    7. Store learned patterns
    8. Final Export
    """
    
    def __init__(self, config: Optional[SchematicMAPOConfig] = None):
        """Initialize optimizer with configuration."""
        self.config = config or get_config()
        set_config(self.config)
        
        # Pipeline components
        self._symbol_resolver: Optional[MemoryEnhancedSymbolResolver] = None
        self._connection_generator: Optional[MemoryEnhancedConnectionGenerator] = None
        self._wire_router: Optional[EnhancedWireRouter] = None
        self._layout_optimizer: Optional[LayoutOptimizerAgent] = None
        self._smoke_test_validator: Optional[SmokeTestValidator] = None
        
        # Gaming AI components
        self._map_elites: Optional[LLMGuidedSchematicMAPElites] = None
        self._red_queen: Optional[LLMGuidedRedQueen] = None
    
    def _ensure_symbol_resolver(self) -> MemoryEnhancedSymbolResolver:
        if self._symbol_resolver is None:
            self._symbol_resolver = MemoryEnhancedSymbolResolver(self.config)
        return self._symbol_resolver
    
    def _ensure_connection_generator(self) -> MemoryEnhancedConnectionGenerator:
        if self._connection_generator is None:
            self._connection_generator = MemoryEnhancedConnectionGenerator(self.config)
        return self._connection_generator
    
    def _ensure_wire_router(self) -> EnhancedWireRouter:
        if self._wire_router is None:
            self._wire_router = EnhancedWireRouter()
        return self._wire_router
    
    def _ensure_layout_optimizer(self) -> LayoutOptimizerAgent:
        if self._layout_optimizer is None:
            self._layout_optimizer = LayoutOptimizerAgent()
        return self._layout_optimizer
    
    def _ensure_smoke_test_validator(self) -> SmokeTestValidator:
        if self._smoke_test_validator is None:
            self._smoke_test_validator = SmokeTestValidator(self.config)
        return self._smoke_test_validator
    
    def _ensure_map_elites(self) -> LLMGuidedSchematicMAPElites:
        if self._map_elites is None:
            self._map_elites = LLMGuidedSchematicMAPElites(self.config)
        return self._map_elites
    
    def _ensure_red_queen(self) -> LLMGuidedRedQueen:
        if self._red_queen is None:
            self._red_queen = LLMGuidedRedQueen(self.config)
        return self._red_queen
    
    async def close(self):
        """Close all components."""
        if self._symbol_resolver:
            await self._symbol_resolver.close()
        if self._connection_generator:
            await self._connection_generator.close()
        if self._map_elites:
            await self._map_elites.close()
        if self._red_queen:
            await self._red_queen.close()
    
    async def optimize(
        self,
        bom: List[Dict[str, Any]],
        design_intent: str,
        design_name: str = "schematic",
        design_type: str = "foc_esc",
        max_iterations: int = 100,
        ideation_context: Optional[IdeationContext] = None,
    ) -> OptimizationResult:
        """
        Run the full MAPO v2.1 schematic optimization pipeline.

        Args:
            bom: Bill of materials
            design_intent: Natural language design description
            design_name: Name for output file
            design_type: Type of design for pattern matching
            max_iterations: Maximum Gaming AI iterations
            ideation_context: Optional structured ideation context extracted
                from ideation artifacts.  When provided, each pipeline stage
                receives domain-specific hints (preferred parts, explicit
                connections, placement constraints, validation criteria) that
                improve schematic quality.

        Returns:
            OptimizationResult with final schematic
        """
        start_time = time.time()
        result = OptimizationResult(success=False)

        try:
            logger.info(f"Starting MAPO v2.1 schematic optimization: {design_name}")
            logger.info(f"BOM: {len(bom)} items, Design type: {design_type}")

            if ideation_context is not None:
                logger.info(
                    "IdeationContext available: "
                    f"bom_artifacts={ideation_context.has_bom_artifacts}, "
                    f"connection_hints={ideation_context.has_connection_hints}, "
                    f"placement_hints={ideation_context.has_placement_hints}, "
                    f"validation_criteria={ideation_context.has_validation_criteria}, "
                    f"artifact_count={ideation_context.artifact_count}"
                )
                if ideation_context.extraction_errors:
                    for err in ideation_context.extraction_errors:
                        logger.warning(f"IdeationContext extraction error: {err}")
            else:
                logger.info("No IdeationContext provided - running in legacy mode")
            
            # Phase 1: Symbol Resolution with Memory
            logger.info("Phase 1: Symbol Resolution")
            resolution_hints = ideation_context.symbol_resolution if ideation_context else None
            resolved_symbols, lib_symbols = await self._resolve_symbols(bom, resolution_hints)
            result.symbols_resolved = len(resolved_symbols)
            result.symbols_from_memory = sum(1 for s in resolved_symbols if s.from_memory)
            result.placeholders = sum(1 for s in resolved_symbols if s.quality == SymbolQuality.PLACEHOLDER)

            # Check for critical placeholder count
            if result.placeholders > len(resolved_symbols) * 0.3:
                logger.warning(f"High placeholder count: {result.placeholders}/{result.symbols_resolved}")

            # Phase 2: Connection Generation with Memory Patterns
            logger.info("Phase 2: Connection Generation")
            components = [s.component for s in resolved_symbols]
            connection_hints = ideation_context.connection_inference if ideation_context else None
            connections = await self._generate_connections(
                components, design_intent, design_type, connection_hints
            )
            result.connections_generated = len(connections)

            # Phase 3: Component Placement
            logger.info("Phase 3: Component Placement")
            placement_hints = ideation_context.placement if ideation_context else None
            placed_components = await self._place_components(components, connections, placement_hints)
            
            # Phase 4: Wire Routing
            logger.info("Phase 4: Wire Routing")
            wires, junctions = await self._route_wires(placed_components, connections)
            result.wires_routed = len(wires)
            
            # Build initial state
            initial_state = SchematicState(
                components=tuple(placed_components),
                connections=tuple(connections),
                wires=tuple(wires),
                junctions=tuple(junctions),
                lib_symbols=lib_symbols,
                design_name=design_name,
                design_intent=design_intent,
            )
            
            # Phase 5: Smoke Test Validation
            logger.info("Phase 5: Smoke Test Validation")
            validator = self._ensure_smoke_test_validator()
            validation_ctx = ideation_context.validation if ideation_context else None
            validated_state, smoke_result = await validator.validate_and_score(
                initial_state, validation_context=validation_ctx
            )
            
            result.smoke_test_passed = smoke_result.passed
            result.final_fitness = validated_state.fitness.combined if validated_state.fitness else 0.0
            
            if smoke_result.passed:
                logger.info(f"Smoke test PASSED - fitness: {result.final_fitness:.3f}")
                # Store successful patterns
                await self._store_successful_patterns(validated_state, design_type, design_intent)
                result.final_state = validated_state
            else:
                # Phase 6: Gaming AI Optimization
                logger.info(f"Smoke test FAILED - starting Gaming AI optimization")
                logger.info(f"Issues: {smoke_result.fatal_issues + smoke_result.error_issues}")
                
                if self.config.map_elites_enabled:
                    optimized_state = await self._optimize_with_gaming_ai(
                        validated_state,
                        smoke_result,
                        max_iterations,
                        design_type,
                        design_intent,
                        ideation_context=ideation_context,
                    )
                    result.final_state = optimized_state
                    result.final_fitness = optimized_state.fitness.combined if optimized_state.fitness else 0.0
                    
                    # Re-check smoke test
                    if optimized_state.validation:
                        result.smoke_test_passed = optimized_state.validation.smoke_test_passed
                else:
                    result.final_state = validated_state
            
            # Phase 7: Generate output
            logger.info("Phase 7: Generate Output")
            if result.final_state:
                schematic_content = self._generate_kicad_schematic(result.final_state)
                result.schematic_content = schematic_content
                
                # Save to file
                output_path = self.config.output_dir / f"{design_name}.kicad_sch"
                output_path.write_text(schematic_content)
                result.schematic_path = output_path
                logger.info(f"Schematic saved to: {output_path}")
            
            result.success = result.final_state is not None
            result.total_iterations = result.final_state.iteration if result.final_state else 0
            
        except Exception as e:
            logger.error(f"Optimization failed: {e}", exc_info=True)
            result.errors.append(str(e))
        
        result.total_time_seconds = time.time() - start_time
        logger.info(f"Optimization complete in {result.total_time_seconds:.1f}s")
        logger.info(f"Success: {result.success}, Fitness: {result.final_fitness:.3f}")
        
        return result
    
    async def _resolve_symbols(
        self,
        bom: List[Dict[str, Any]],
        resolution_hints: Optional[SymbolResolutionContext] = None,
    ) -> Tuple[List, Dict[str, str]]:
        """Resolve symbols for all BOM items.

        Args:
            bom: Bill of materials list.
            resolution_hints: Optional symbol resolution hints from ideation
                context.  When provided, preferred parts, manufacturer priority,
                and package preferences are merged into the resolution process.
        """
        resolver = self._ensure_symbol_resolver()
        return await resolver.resolve_symbols(bom, resolution_hints=resolution_hints)
    
    async def _generate_connections(
        self,
        components: List[ComponentInstance],
        design_intent: str,
        design_type: str,
        connection_hints: Optional[ConnectionInferenceContext] = None,
    ) -> List[Connection]:
        """Generate connections with memory patterns.

        Args:
            components: Resolved component instances.
            design_intent: Natural language design description.
            design_type: Type of design for pattern matching.
            connection_hints: Optional connection inference hints from ideation
                context.  When provided, explicit connections are used as seed
                connections, interface definitions drive structured bus
                generation, and power rail data informs power net wiring.
        """
        generator = self._ensure_connection_generator()
        return await generator.generate_connections(
            components=components,
            design_intent=design_intent,
            design_type=design_type,
            connection_hints=connection_hints,
        )
    
    async def _place_components(
        self,
        components: List[ComponentInstance],
        connections: List[Connection],
        placement_hints: Optional[PlacementContext] = None,
    ) -> List[ComponentInstance]:
        """Place components using layout optimizer.

        Args:
            components: Resolved component instances.
            connections: Generated connections.
            placement_hints: Optional placement context from ideation.
                When provided, subsystem blocks supply position hints
                and grouping constraints, signal flow direction guides
                overall layout, and critical proximity / isolation zones
                are applied.
        """
        layout_optimizer = self._ensure_layout_optimizer()

        # Convert to format expected by layout optimizer
        bom_items = [
            {
                "reference": c.reference,
                "part_number": c.part_number,
                "category": c.category,
            }
            for c in components
        ]

        connection_dicts = [
            {
                "from_ref": c.from_ref,
                "to_ref": c.to_ref,
            }
            for c in connections
        ]

        # Build subsystem grouping and position hints from ideation context
        subsystem_groups: Dict[str, List[str]] = {}
        position_hints: Dict[str, str] = {}
        signal_flow: Optional[str] = None
        proximity_pairs: List[Tuple[str, str]] = []
        isolation_zones: List[Tuple[str, str]] = []

        if placement_hints is not None:
            logger.info(
                f"Applying placement hints: {len(placement_hints.subsystem_blocks)} "
                f"subsystem blocks, flow={placement_hints.signal_flow_direction}"
            )
            signal_flow = placement_hints.signal_flow_direction

            for block in placement_hints.subsystem_blocks:
                if block.components:
                    subsystem_groups[block.name] = block.components
                if block.position_hint:
                    position_hints[block.name] = block.position_hint

            proximity_pairs = list(placement_hints.critical_proximity)
            isolation_zones = list(placement_hints.isolation_zones)

        # Get optimized positions
        positions = await layout_optimizer.optimize_layout(
            bom_items=bom_items,
            connections=connection_dicts,
            subsystem_groups=subsystem_groups if subsystem_groups else None,
            position_hints=position_hints if position_hints else None,
            signal_flow=signal_flow,
            proximity_pairs=proximity_pairs if proximity_pairs else None,
            isolation_zones=isolation_zones if isolation_zones else None,
        )

        # Update component positions
        placed = []
        for comp in components:
            pos = positions.get(comp.reference, (50.0, 50.0))
            # Create new component with position (immutable)
            placed_comp = ComponentInstance(
                uuid=comp.uuid,
                part_number=comp.part_number,
                manufacturer=comp.manufacturer,
                category=comp.category,
                reference=comp.reference,
                value=comp.value,
                position=pos,
                rotation=comp.rotation,
                symbol_id=comp.symbol_id,
                footprint=comp.footprint,
                quality=comp.quality,
                resolution_source=comp.resolution_source,
                pins=comp.pins,
                description=comp.description,
            )
            placed.append(placed_comp)

        return placed
    
    async def _route_wires(
        self,
        components: List[ComponentInstance],
        connections: List[Connection],
    ) -> Tuple[List[Wire], List[Junction]]:
        """Route wires using EnhancedWireRouter."""
        router = self._ensure_wire_router()
        
        # Build component position map
        comp_positions = {c.reference: c.position for c in components}
        
        # Build pin position map
        pin_positions = {}
        for comp in components:
            for pin in comp.pins:
                # Calculate absolute pin position
                cx, cy = comp.position
                px, py = pin.position
                abs_x = cx + px
                abs_y = cy + py
                pin_positions[(comp.reference, pin.name)] = (abs_x, abs_y)
                pin_positions[(comp.reference, pin.number)] = (abs_x, abs_y)
        
        # Route each connection
        wires = []
        junctions = []
        net_points = {}  # net_name -> list of junction points
        
        for conn in connections:
            # Get pin positions
            from_pos = pin_positions.get((conn.from_ref, conn.from_pin))
            to_pos = pin_positions.get((conn.to_ref, conn.to_pin))
            
            if from_pos is None:
                from_pos = comp_positions.get(conn.from_ref, (0, 0))
            if to_pos is None:
                to_pos = comp_positions.get(conn.to_ref, (0, 0))
            
            # Route wire (Manhattan routing)
            segments = self._manhattan_route(from_pos, to_pos)
            
            net_name = conn.net_name or f"Net_{conn.from_ref}_{conn.from_pin}"
            
            wire = Wire(
                net_name=net_name,
                segments=tuple(segments),
                from_ref=conn.from_ref,
                from_pin=conn.from_pin,
                to_ref=conn.to_ref,
                to_pin=conn.to_pin,
            )
            wires.append(wire)
            
            # Track junction points for multi-point nets
            if net_name not in net_points:
                net_points[net_name] = []
            net_points[net_name].extend([from_pos, to_pos])
        
        # Create junctions where nets have multiple connections
        for net_name, points in net_points.items():
            if len(points) > 2:
                # Find common junction point (simplified: use center)
                unique_points = list(set(points))
                if len(unique_points) > 2:
                    # Create junction at first shared point
                    junctions.append(Junction(
                        position=unique_points[0],
                        net_name=net_name,
                    ))
        
        return wires, junctions
    
    def _manhattan_route(
        self,
        start: Tuple[float, float],
        end: Tuple[float, float],
    ) -> List[WireSegment]:
        """Simple Manhattan routing between two points."""
        x1, y1 = start
        x2, y2 = end
        
        # Horizontal first, then vertical
        mid_x = (x1 + x2) / 2
        
        segments = []
        
        if abs(x2 - x1) > 0.1:
            # First horizontal segment
            segments.append(WireSegment(start=(x1, y1), end=(mid_x, y1)))
            # Vertical segment
            if abs(y2 - y1) > 0.1:
                segments.append(WireSegment(start=(mid_x, y1), end=(mid_x, y2)))
            # Second horizontal segment
            segments.append(WireSegment(start=(mid_x, y2), end=(x2, y2)))
        elif abs(y2 - y1) > 0.1:
            # Just vertical
            segments.append(WireSegment(start=(x1, y1), end=(x2, y2)))
        else:
            # Same point, short wire
            segments.append(WireSegment(start=(x1, y1), end=(x2, y2)))
        
        return segments
    
    async def _optimize_with_gaming_ai(
        self,
        initial_state: SchematicState,
        smoke_result: SmokeTestValidationResult,
        max_iterations: int,
        design_type: str,
        design_intent: str,
        ideation_context: Optional[IdeationContext] = None,
    ) -> SchematicState:
        """
        Run MAP-Elites + Red Queen optimization until smoke test passes.

        Args:
            initial_state: Starting schematic state.
            smoke_result: Initial smoke test result.
            max_iterations: Maximum optimization iterations.
            design_type: Type of design for pattern matching.
            design_intent: Natural language design description.
            ideation_context: Optional structured ideation context.  When
                provided, the designer intent text is included in LLM
                mutation guidance prompts, and the validation context is
                threaded through re-validation calls so that ideation
                test criteria are evaluated at every iteration.
        """
        map_elites = self._ensure_map_elites()
        red_queen = self._ensure_red_queen()
        validator = self._ensure_smoke_test_validator()

        # Build supplementary designer intent from ideation context
        designer_intent_supplement = ""
        if ideation_context is not None:
            intent_text = ideation_context.design_intent_text
            if intent_text:
                designer_intent_supplement = (
                    "\n\n=== DESIGNER INTENT FROM IDEATION ===\n" + intent_text
                )
                logger.info(
                    "Including ideation designer intent in Gaming AI guidance "
                    f"({len(intent_text)} chars)"
                )

        validation_ctx = ideation_context.validation if ideation_context else None

        # Create initial solution
        solution = SchematicSolution(state=initial_state)
        solution.behavior_descriptor = map_elites.compute_behavior_descriptor(initial_state)
        map_elites.add_solution(solution)

        best_state = initial_state
        best_fitness = initial_state.fitness.combined if initial_state.fitness else 0.0

        issues = smoke_result.fatal_issues + smoke_result.error_issues

        for iteration in range(max_iterations):
            logger.info(f"Gaming AI iteration {iteration + 1}/{max_iterations}")

            # Select cells to explore
            cells = await map_elites.select_cells_for_exploration(issues, num_cells=3)

            for cell in cells:
                if cell.is_empty:
                    continue

                # Get LLM mutation guidance - enrich issues with designer intent
                enriched_issues = list(issues)
                if designer_intent_supplement:
                    enriched_issues.append(designer_intent_supplement)

                guidance = await map_elites.guide_mutation(cell.solution, enriched_issues)
                logger.debug(f"Mutation guidance: {guidance.mutation_type} - {guidance.description}")

                # Apply mutation
                mutated_state = await self._apply_mutation(cell.solution.state, guidance)

                # Re-validate with ideation validation context
                validated_state, new_smoke_result = await validator.validate_and_score(
                    mutated_state, validation_context=validation_ctx
                )
                
                # Create new solution
                new_solution = SchematicSolution(state=validated_state)
                new_solution.behavior_descriptor = map_elites.compute_behavior_descriptor(validated_state)
                new_solution.last_mutation = guidance.mutation_type
                
                # Add to archive
                map_elites.add_solution(new_solution)
                
                # Track best
                new_fitness = validated_state.fitness.combined if validated_state.fitness else 0.0
                if new_fitness > best_fitness:
                    best_fitness = new_fitness
                    best_state = validated_state
                    logger.info(f"New best fitness: {best_fitness:.3f}")
                
                # Check if smoke test passed
                if new_smoke_result.passed:
                    logger.info(f"Smoke test PASSED at iteration {iteration + 1}")
                    # Store successful patterns
                    await self._store_successful_patterns(validated_state, design_type, design_intent)
                    return validated_state
                
                issues = new_smoke_result.fatal_issues + new_smoke_result.error_issues
            
            # Periodic Red Queen round
            if self.config.red_queen_enabled and iteration % 10 == 9:
                solutions = [c.solution for c in map_elites.archive.archive.values() if not c.is_empty]
                if len(solutions) >= 3:
                    await red_queen.run_adversarial_round(solutions[:10])
        
        logger.warning(f"Max iterations reached without passing smoke test")
        return best_state
    
    async def _apply_mutation(
        self,
        state: SchematicState,
        guidance: MutationGuidance,
    ) -> SchematicState:
        """Apply a mutation to a state based on LLM guidance."""
        # For now, use a simplified mutation approach
        # A full implementation would have various mutation operators
        
        if guidance.mutation_type == "connection_fix" and guidance.target_connection:
            # Add or fix a connection
            from_ref, from_pin, to_ref, to_pin = guidance.target_connection
            new_conn = Connection(
                from_ref=from_ref,
                from_pin=from_pin,
                to_ref=to_ref,
                to_pin=to_pin,
                net_name=f"Net_{from_ref}_{from_pin}",
                inferred=True,
            )
            connections = list(state.connections) + [new_conn]
            return state.with_connections(connections)
        
        elif guidance.mutation_type == "wire_route":
            # Re-route wires
            components = list(state.components)
            connections = list(state.connections)
            wires, junctions = await self._route_wires(components, connections)
            return state.with_wires(wires, junctions)
        
        # Default: return state unchanged
        return state
    
    async def _store_successful_patterns(
        self,
        state: SchematicState,
        design_type: str,
        design_intent: str,
    ):
        """Store successful patterns in memory."""
        if not self.config.enable_memory:
            return
        
        generator = self._ensure_connection_generator()
        await generator.store_successful_connections(
            connections=list(state.connections),
            components=list(state.components),
            design_type=design_type,
            design_intent=design_intent,
            smoke_test_passed=True,
            validation_score=state.fitness.combined if state.fitness else 0.0,
        )
        logger.info("Stored successful wiring patterns in nexus-memory")
    
    def _generate_kicad_schematic(self, state: SchematicState) -> str:
        """Generate KiCad schematic S-expression from state."""
        lines = [
            "(kicad_sch",
            "  (version 20211123)",
            "  (generator mapo_v2_1_schematic)",
            "",
            '  (paper "A4")',
            "",
            "  (lib_symbols",
        ]
        
        # Add library symbols
        for symbol_id, symbol_content in state.lib_symbols.items():
            lines.append(f"    {symbol_content}")
        
        lines.append("  )")
        lines.append("")
        
        # Add component instances
        for comp in state.components:
            x, y = comp.position
            lines.extend([
                f"  (symbol",
                f'    (lib_id "{comp.symbol_id}")',
                f"    (at {x:.2f} {y:.2f} {comp.rotation})",
                f'    (uuid "{comp.uuid}")',
                f'    (property "Reference" "{comp.reference}" (at {x:.2f} {y+2.54:.2f} 0))',
                f'    (property "Value" "{comp.value}" (at {x:.2f} {y-2.54:.2f} 0))',
                f'    (property "Footprint" "{comp.footprint}" (at {x:.2f} {y:.2f} 0) (hide yes))',
                f"  )",
            ])
        
        # Add wires
        for wire in state.wires:
            for segment in wire.segments:
                x1, y1 = segment.start
                x2, y2 = segment.end
                lines.append(f"  (wire (pts (xy {x1:.2f} {y1:.2f}) (xy {x2:.2f} {y2:.2f})))")
        
        # Add junctions
        for junction in state.junctions:
            x, y = junction.position
            lines.append(f"  (junction (at {x:.2f} {y:.2f}))")
        
        # Add net labels
        added_labels = set()
        for conn in state.connections:
            if conn.net_name and conn.net_name not in added_labels:
                # Find a wire with this net to get position
                for wire in state.wires:
                    if wire.net_name == conn.net_name and wire.segments:
                        x, y = wire.segments[0].start
                        lines.append(f'  (label "{conn.net_name}" (at {x:.2f} {y-2.54:.2f} 0))')
                        added_labels.add(conn.net_name)
                        break
        
        lines.append(")")
        
        return "\n".join(lines)
