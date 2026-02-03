#!/usr/bin/env python3
"""
Refinement Loop - AlphaFold-style iterative refinement for PCB optimization.

This module implements an iterative refinement process inspired by AlphaFold:
1. Analyze current state and identify improvement opportunities
2. Generate targeted refinements
3. Evaluate and select best refinement
4. Iterate until convergence

The key insight from AlphaFold is that multiple rounds of refinement,
each using the output of the previous round, can progressively improve quality.
"""

import os
import sys
import json
import asyncio
import time
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any
from pathlib import Path

# Add script directory to path for local imports
SCRIPT_DIR = Path(__file__).parent.resolve()
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from pcb_state import PCBState, PCBModification, DRCResult, ModificationType
from generator_agents import AgentPool, GenerationResult


@dataclass
class RefinementAnalysis:
    """Analysis of current state for refinement."""
    total_violations: int
    top_violation_types: List[Tuple[str, int]]
    clustered_issues: List[Dict[str, Any]]
    priority_fixes: List[str]
    estimated_reducible: int


@dataclass
class RefinementStep:
    """Record of a single refinement step."""
    cycle: int
    before_violations: int
    after_violations: int
    improvement: int
    modification_applied: Optional[PCBModification]
    analysis: RefinementAnalysis
    duration_seconds: float


@dataclass
class RefinementResult:
    """Result of the refinement loop."""
    initial_violations: int
    final_violations: int
    total_improvement: int
    cycles_completed: int
    converged: bool
    steps: List[RefinementStep]
    final_state: PCBState


class RefinementLoop:
    """
    AlphaFold-inspired iterative refinement for PCB layouts.

    Key concepts from AlphaFold applied here:
    1. Recycling: Output of each iteration feeds into the next
    2. Progressive refinement: Each cycle targets remaining issues
    3. Convergence detection: Stop when improvements plateau
    4. Multi-scale analysis: Consider both local and global issues
    """

    def __init__(
        self,
        max_cycles: int = 10,
        convergence_threshold: float = 0.01,
        min_improvement: int = 5,
        agent_pool: Optional[AgentPool] = None,
        target_violations: int = 100
    ):
        """
        Initialize refinement loop.

        Args:
            max_cycles: Maximum refinement cycles
            convergence_threshold: Relative improvement threshold for convergence
            min_improvement: Minimum absolute improvement to continue
            agent_pool: Pool of LLM agents for guided refinement
            target_violations: Target violation count
        """
        self.max_cycles = max_cycles
        self.convergence_threshold = convergence_threshold
        self.min_improvement = min_improvement
        self.agent_pool = agent_pool or AgentPool()
        self.target_violations = target_violations

        # History
        self.steps: List[RefinementStep] = []

    async def refine(self, initial_state: PCBState) -> RefinementResult:
        """
        Run iterative refinement on a PCB state.

        Args:
            initial_state: Starting PCB state

        Returns:
            RefinementResult with final state and history
        """
        print(f"\n{'='*60}")
        print("ALPHAFOLD-STYLE REFINEMENT LOOP")
        print(f"{'='*60}")

        start_time = time.time()

        # Initialize
        current_state = initial_state.copy()
        initial_drc = current_state.run_drc()
        initial_violations = initial_drc.total_violations

        print(f"Initial violations: {initial_violations}")
        print(f"Target: {self.target_violations}")
        print(f"Max cycles: {self.max_cycles}")

        best_state = current_state
        best_violations = initial_violations
        converged = False

        for cycle in range(self.max_cycles):
            cycle_start = time.time()

            print(f"\n--- Cycle {cycle + 1}/{self.max_cycles} ---")

            # 1. Analyze current state
            analysis = await self._analyze_state(current_state)
            print(f"  Violations: {analysis.total_violations}")
            print(f"  Top issues: {analysis.top_violation_types[:3]}")
            print(f"  Estimated reducible: {analysis.estimated_reducible}")

            # 2. Generate targeted refinements
            refinements = await self._generate_refinements(current_state, analysis)
            print(f"  Generated {len(refinements)} refinement candidates")

            if not refinements:
                print("  No refinements generated - may have converged")
                break

            # 3. Evaluate candidates and select best
            candidates = []
            for ref in refinements:
                candidate_state = current_state.apply_modification(ref)
                candidate_drc = candidate_state.run_drc()
                candidates.append((candidate_state, candidate_drc, ref))

            # Sort by violations (ascending)
            candidates.sort(key=lambda x: x[1].total_violations)

            best_candidate = candidates[0]
            new_state, new_drc, applied_mod = best_candidate

            # 4. Check improvement
            before_violations = analysis.total_violations
            after_violations = new_drc.total_violations
            improvement = before_violations - after_violations

            cycle_duration = time.time() - cycle_start

            step = RefinementStep(
                cycle=cycle + 1,
                before_violations=before_violations,
                after_violations=after_violations,
                improvement=improvement,
                modification_applied=applied_mod if improvement > 0 else None,
                analysis=analysis,
                duration_seconds=cycle_duration
            )
            self.steps.append(step)

            print(f"  Best candidate: {after_violations} violations")
            print(f"  Improvement: {improvement:+d}")

            # 5. Check convergence
            if improvement <= 0:
                print("  No improvement - checking convergence")
                # Try a few more random modifications before declaring convergence
                if cycle > 2:
                    converged = True
                    print("  CONVERGED: No further improvement possible")
                    break
            elif improvement < self.min_improvement:
                relative_improvement = improvement / max(1, before_violations)
                if relative_improvement < self.convergence_threshold:
                    converged = True
                    print(f"  CONVERGED: Improvement below threshold ({relative_improvement:.3f} < {self.convergence_threshold})")
                    break

            # 6. Accept improvement and continue
            if improvement > 0:
                current_state = new_state

                if after_violations < best_violations:
                    best_state = new_state
                    best_violations = after_violations
                    print(f"  New best: {best_violations} violations")

            # 7. Check target
            if best_violations <= self.target_violations:
                print(f"  TARGET REACHED: {best_violations} <= {self.target_violations}")
                converged = True
                break

        total_time = time.time() - start_time

        # Final summary
        total_improvement = initial_violations - best_violations

        print(f"\n{'='*60}")
        print("REFINEMENT COMPLETE")
        print(f"{'='*60}")
        print(f"Cycles: {len(self.steps)}")
        print(f"Time: {total_time:.1f}s")
        print(f"Initial violations: {initial_violations}")
        print(f"Final violations: {best_violations}")
        print(f"Total improvement: {total_improvement} ({100*total_improvement/max(1,initial_violations):.1f}%)")
        print(f"Converged: {converged}")

        return RefinementResult(
            initial_violations=initial_violations,
            final_violations=best_violations,
            total_improvement=total_improvement,
            cycles_completed=len(self.steps),
            converged=converged,
            steps=self.steps,
            final_state=best_state
        )

    async def _analyze_state(self, state: PCBState) -> RefinementAnalysis:
        """
        Analyze current state for refinement opportunities.

        Uses both heuristics and LLM analysis to identify:
        - Violation clustering (spatially related issues)
        - Root causes vs symptoms
        - Priority order for fixes
        """
        drc = state.run_drc()

        # Sort violations by count
        sorted_types = sorted(
            drc.violations_by_type.items(),
            key=lambda x: x[1],
            reverse=True
        )

        # Cluster analysis (simplified)
        clusters = []
        for vtype, count in sorted_types[:5]:
            if count > 0:
                clusters.append({
                    'type': vtype,
                    'count': count,
                    'fixable': self._estimate_fixable(vtype, count)
                })

        # Priority fixes based on impact
        priority_fixes = []
        for vtype, count in sorted_types:
            if count > 10:
                priority_fixes.append(f"Address {count} {vtype} violations")

        # Estimate reducible violations
        estimated_reducible = sum(c['fixable'] for c in clusters)

        return RefinementAnalysis(
            total_violations=drc.total_violations,
            top_violation_types=sorted_types[:5],
            clustered_issues=clusters,
            priority_fixes=priority_fixes[:3],
            estimated_reducible=estimated_reducible
        )

    def _estimate_fixable(self, violation_type: str, count: int) -> int:
        """Estimate how many violations of this type are fixable."""
        # Fixability factors based on violation type
        fixability = {
            'clearance': 0.7,
            'via_dangling': 0.9,
            'track_dangling': 0.9,
            'silk_over_copper': 0.8,
            'silk_overlap': 0.8,
            'solder_mask_bridge': 0.6,
            'shorting_items': 0.5,
            'courtyards_overlap': 0.4,
            'unconnected': 0.3,
        }
        factor = fixability.get(violation_type, 0.5)
        return int(count * factor)

    async def _generate_refinements(
        self,
        state: PCBState,
        analysis: RefinementAnalysis
    ) -> List[PCBModification]:
        """
        Generate refinement candidates targeting the analysis results.

        Uses agent pool to generate intelligent refinements.
        """
        drc = state.run_drc()

        # Get agent suggestions
        try:
            results = await self.agent_pool.generate_all(
                state, drc, max_modifications_per_agent=3
            )

            all_mods = []
            for result in results:
                for mod in result.modifications:
                    # Weight modifications by agent confidence and relevance
                    mod.confidence *= result.confidence
                    all_mods.append(mod)

            # Sort by confidence
            all_mods.sort(key=lambda m: m.confidence, reverse=True)

            return all_mods[:10]  # Return top 10

        except Exception as e:
            print(f"  Agent generation failed: {e}")
            return self._generate_heuristic_refinements(state, analysis)

    def _generate_heuristic_refinements(
        self,
        state: PCBState,
        analysis: RefinementAnalysis
    ) -> List[PCBModification]:
        """Generate refinements using heuristics when LLM unavailable."""
        mods = []

        for cluster in analysis.clustered_issues:
            vtype = cluster['type']
            count = cluster.get('count', 0)

            if vtype == 'clearance':
                mods.append(PCBModification(
                    mod_type=ModificationType.ADJUST_CLEARANCE,
                    target='signal_clearance',
                    parameters={'param_name': 'signal_clearance', 'value': 0.18},
                    description="Increase signal clearance",
                    confidence=0.6
                ))
            elif vtype in ['via_dangling', 'track_dangling']:
                mods.append(PCBModification(
                    mod_type=ModificationType.ADJUST_ZONE,
                    target='GND',
                    parameters={'clearance': 0.25},
                    description="Adjust zone clearance for dangling elements",
                    confidence=0.5
                ))
            elif vtype == 'silk_over_copper':
                # MAPOS Phase 2: Move silkscreen away from copper
                mods.append(PCBModification(
                    mod_type=ModificationType.MOVE_SILKSCREEN,
                    target='all',
                    parameters={'offset_y_mm': 1.5, 'element': 'all'},
                    description=f"Move {count} silkscreen elements away from copper",
                    confidence=0.7
                ))
                # Also shrink silkscreen text to reduce overlap
                mods.append(PCBModification(
                    mod_type=ModificationType.ADJUST_SILKSCREEN,
                    target='all',
                    parameters={'text_size_mm': 0.8, 'move_to_courtyard': True},
                    description="Shrink silkscreen text and move outside courtyard",
                    confidence=0.65
                ))
            elif vtype == 'silk_overlap':
                # MAPOS Phase 2: Fix silkscreen overlap
                mods.append(PCBModification(
                    mod_type=ModificationType.ADJUST_SILKSCREEN,
                    target='all',
                    parameters={'text_size_mm': 0.7, 'text_thickness_mm': 0.1},
                    description=f"Reduce {count} silkscreen text sizes to prevent overlap",
                    confidence=0.65
                ))
            elif vtype == 'solder_mask_bridge':
                # MAPOS Phase 2: Fix solder mask bridge violations
                # Use negative expansion to create mask dam between pads
                mods.append(PCBModification(
                    mod_type=ModificationType.ADJUST_SOLDER_MASK,
                    target='all_smd',
                    parameters={'expansion_mm': -0.03},
                    description=f"Reduce solder mask opening on {count} pads to create mask dam",
                    confidence=0.7
                ))
            elif vtype == 'courtyards_overlap':
                # MAPOS Phase 2: Courtyard overlap handling
                mods.append(PCBModification(
                    mod_type=ModificationType.ADJUST_COURTYARD,
                    target='overlapping',
                    parameters={'expansion_mm': -0.1},
                    description=f"Reduce courtyard size on {count} overlapping components",
                    confidence=0.5
                ))
            elif vtype == 'shorting_items':
                # Zone net assignment issues
                mods.append(PCBModification(
                    mod_type=ModificationType.FIX_ZONE_NET,
                    target='misassigned_zones',
                    parameters={},
                    description=f"Fix {count} zone net assignments",
                    confidence=0.8
                ))
            elif vtype == 'unconnected':
                # Unconnected net handling - mostly informational
                mods.append(PCBModification(
                    mod_type=ModificationType.ADD_TRACE,
                    target='unconnected_nets',
                    parameters={'auto_route': True},
                    description=f"Route {count} unconnected nets",
                    confidence=0.3
                ))

        return mods


class IterativeRefinementOptimizer:
    """
    Full iterative refinement optimizer combining multiple techniques.

    Uses:
    1. Initial agent-based improvement
    2. AlphaFold-style refinement loop
    3. Final polishing pass
    """

    def __init__(
        self,
        pcb_path: str,
        target_violations: int = 100,
        max_cycles: int = 10,
        agent_pool: Optional[AgentPool] = None
    ):
        self.pcb_path = Path(pcb_path)
        self.target_violations = target_violations
        self.max_cycles = max_cycles
        self.agent_pool = agent_pool or AgentPool()

        self.refinement_loop = RefinementLoop(
            max_cycles=max_cycles,
            target_violations=target_violations,
            agent_pool=self.agent_pool
        )

    async def optimize(self) -> RefinementResult:
        """Run full optimization pipeline."""
        print(f"\n{'='*60}")
        print("ITERATIVE REFINEMENT OPTIMIZER")
        print(f"{'='*60}")
        print(f"PCB: {self.pcb_path.name}")
        print(f"Target: {self.target_violations} violations")

        # Load initial state
        initial_state = PCBState.from_file(str(self.pcb_path))
        initial_drc = initial_state.run_drc()

        print(f"\nInitial state:")
        print(f"  Violations: {initial_drc.total_violations}")
        print(f"  Errors: {initial_drc.errors}")
        print(f"  Unconnected: {initial_drc.unconnected}")

        # Run refinement loop
        result = await self.refinement_loop.refine(initial_state)

        return result

    def save_results(self, result: RefinementResult, output_path: str) -> None:
        """Save optimization results to file."""
        results = {
            'pcb_path': str(self.pcb_path),
            'initial_violations': result.initial_violations,
            'final_violations': result.final_violations,
            'total_improvement': result.total_improvement,
            'cycles_completed': result.cycles_completed,
            'converged': result.converged,
            'steps': [
                {
                    'cycle': s.cycle,
                    'before': s.before_violations,
                    'after': s.after_violations,
                    'improvement': s.improvement,
                    'duration': s.duration_seconds
                }
                for s in result.steps
            ],
            'final_modifications': [
                m.to_dict() for m in result.final_state.modifications
            ]
        }

        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"\nSaved results to: {output_path}")


async def run_refinement(
    pcb_path: str,
    max_cycles: int = 10,
    target_violations: int = 100,
    output_json: Optional[str] = None
) -> RefinementResult:
    """
    Run refinement loop on a PCB file.

    Args:
        pcb_path: Path to PCB file
        max_cycles: Maximum refinement cycles
        target_violations: Target violation count
        output_json: Optional path to save results

    Returns:
        RefinementResult
    """
    optimizer = IterativeRefinementOptimizer(
        pcb_path=pcb_path,
        target_violations=target_violations,
        max_cycles=max_cycles
    )

    result = await optimizer.optimize()

    if output_json:
        optimizer.save_results(result, output_json)

    return result


if __name__ == '__main__':
    import sys

    if len(sys.argv) < 2:
        print("Usage: python refinement_loop.py <path_to.kicad_pcb> [max_cycles] [target_violations]")
        sys.exit(1)

    pcb_path = sys.argv[1]
    max_cycles = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    target_violations = int(sys.argv[3]) if len(sys.argv) > 3 else 100

    asyncio.run(run_refinement(
        pcb_path,
        max_cycles=max_cycles,
        target_violations=target_violations,
        output_json=f"{pcb_path.rsplit('.', 1)[0]}_refinement_results.json"
    ))
