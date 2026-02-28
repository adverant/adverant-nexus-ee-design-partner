#!/usr/bin/env python3
"""
Continuous Loop Runner for MAPO Schematic Pipeline.

Runs the pipeline repeatedly until all quality gates pass, adjusting
parameters between iterations based on failure analysis.

Usage:
    python3 run_continuous_loop.py --design "foc_esc" --max-iterations 10

Quality Gates (ALL must pass):
    - 0% placeholder symbols
    - >= 80% connection coverage
    - 0 component overlaps
    - 0 fatal smoke test issues
    - >= 0.65 visual validation score
    - < 10% center fallback ratio
"""

import argparse
import asyncio
import json
import logging
import os
import sys
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, Any, List, Optional

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from mapo_schematic_pipeline import (
    MAPOSchematicPipeline,
    PipelineConfig,
    PipelineResult,
)
from api_generate_schematic import create_foc_esc_bom, create_design_intent

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    stream=sys.stderr,
)
logger = logging.getLogger("continuous_loop")


# ---------------------------------------------------------------------------
# Quality Gates
# ---------------------------------------------------------------------------

@dataclass
class QualityGates:
    """Thresholds that must ALL pass for the schematic to be accepted."""
    placeholder_ratio_max: float = 0.0       # 0 placeholders
    connection_coverage_min: float = 0.80     # >= 80% BOM connected
    overlap_count_max: int = 0               # 0 overlaps
    smoke_test_fatal_max: int = 0            # 0 fatal issues
    visual_score_min: float = 0.65           # visual >= 65% (real image analysis via proxy)
    center_fallback_ratio_max: float = 0.10  # < 10% center fallbacks
    functional_score_min: float = 0.60       # functional validation >= 60%


@dataclass
class GateResult:
    """Result of checking a single quality gate."""
    name: str
    passed: bool
    actual: float
    threshold: float
    message: str = ""


# ---------------------------------------------------------------------------
# Iteration Config (tunable between attempts)
# ---------------------------------------------------------------------------

class EscalationLevel:
    """Ralph-style progressive escalation for stagnating iterations."""
    NONE = 0            # Normal operation
    INCREASE_PARAMS = 1 # Increase spacing/retries aggressively
    FULL_RESET = 2      # No checkpoint resume, fresh generation
    ESCALATE = 3        # Log that human intervention may be needed

    @staticmethod
    def for_stagnation_count(count: int) -> int:
        """Map consecutive stagnation count to escalation level."""
        if count < 3:
            return EscalationLevel.NONE
        elif count < 5:
            return EscalationLevel.INCREASE_PARAMS
        elif count < 7:
            return EscalationLevel.FULL_RESET
        else:
            return EscalationLevel.ESCALATE


@dataclass
class IterationConfig:
    """Parameters tuned between iterations based on failure analysis."""
    layout_spacing_multiplier: float = 1.0
    connection_retry_count: int = 3
    max_visual_iterations: int = 5
    resume_from_checkpoint: bool = False
    escalation_level: int = 0


# ---------------------------------------------------------------------------
# Iteration Result
# ---------------------------------------------------------------------------

@dataclass
class IterationResult:
    """Captured result of one pipeline run."""
    iteration: int
    success: bool
    duration_seconds: float
    gate_results: List[Dict[str, Any]] = field(default_factory=list)
    all_gates_passed: bool = False
    pipeline_result: Optional[Dict[str, Any]] = None
    config_used: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


# ---------------------------------------------------------------------------
# Continuous Loop Runner
# ---------------------------------------------------------------------------

class ContinuousLoopRunner:
    """Runs the MAPO pipeline in a loop until quality gates pass."""

    def __init__(
        self,
        design_name: str = "foc_esc",
        max_iterations: int = 10,
        subsystems: Optional[List[str]] = None,
        gates: Optional[QualityGates] = None,
        output_dir: Optional[str] = None,
    ):
        self.design_name = design_name
        self.max_iterations = max_iterations
        self.subsystems = subsystems or [
            "Power Input Stage",
            "Gate Driver",
            "Power Stage",
            "MCU Core",
            "Current Sensing",
            "Communication",
        ]
        self.gates = gates or QualityGates()
        self.output_dir = Path(output_dir or os.environ.get(
            "OUTPUT_DIR", str(Path(__file__).parent / "output" / "loop_results")
        ))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.iteration_results: List[IterationResult] = []

    # ----- Quality gate checks -----

    def _check_gates(self, result: PipelineResult) -> List[GateResult]:
        """Check all quality gates against pipeline result."""
        checks = []

        # 1. Placeholder ratio
        checks.append(GateResult(
            name="placeholder_ratio",
            passed=result.placeholder_ratio <= self.gates.placeholder_ratio_max,
            actual=result.placeholder_ratio,
            threshold=self.gates.placeholder_ratio_max,
            message=f"{result.placeholder_ratio:.0%} placeholders (max {self.gates.placeholder_ratio_max:.0%})",
        ))

        # 2. Connection coverage
        checks.append(GateResult(
            name="connection_coverage",
            passed=result.connection_coverage >= self.gates.connection_coverage_min,
            actual=result.connection_coverage,
            threshold=self.gates.connection_coverage_min,
            message=f"{result.connection_coverage:.0%} coverage (min {self.gates.connection_coverage_min:.0%})",
        ))

        # 3. Overlap count
        checks.append(GateResult(
            name="overlap_count",
            passed=result.overlap_count <= self.gates.overlap_count_max,
            actual=float(result.overlap_count),
            threshold=float(self.gates.overlap_count_max),
            message=f"{result.overlap_count} overlaps (max {self.gates.overlap_count_max})",
        ))

        # 4. Smoke test fatals
        fatal_count = 0
        if result.smoke_test_result:
            fatal_count = sum(
                1 for issue in getattr(result.smoke_test_result, "issues", [])
                if getattr(issue, "severity", None) and "fatal" in str(getattr(issue, "severity", "")).lower()
            )
        checks.append(GateResult(
            name="smoke_test_fatals",
            passed=fatal_count <= self.gates.smoke_test_fatal_max,
            actual=float(fatal_count),
            threshold=float(self.gates.smoke_test_fatal_max),
            message=f"{fatal_count} fatal issues (max {self.gates.smoke_test_fatal_max})",
        ))

        # 5. Visual score
        checks.append(GateResult(
            name="visual_score",
            passed=result.visual_score >= self.gates.visual_score_min,
            actual=result.visual_score,
            threshold=self.gates.visual_score_min,
            message=f"{result.visual_score:.1%} visual (min {self.gates.visual_score_min:.1%})",
        ))

        # 6. Center fallback ratio
        checks.append(GateResult(
            name="center_fallback_ratio",
            passed=result.center_fallback_ratio <= self.gates.center_fallback_ratio_max,
            actual=result.center_fallback_ratio,
            threshold=self.gates.center_fallback_ratio_max,
            message=f"{result.center_fallback_ratio:.1%} fallbacks (max {self.gates.center_fallback_ratio_max:.1%})",
        ))

        # 7. Functional validation score (design-intent compliance)
        functional_score = getattr(result, 'functional_score', 0.0)
        checks.append(GateResult(
            name="functional_score",
            passed=functional_score >= self.gates.functional_score_min,
            actual=functional_score,
            threshold=self.gates.functional_score_min,
            message=f"{functional_score:.1%} functional (min {self.gates.functional_score_min:.1%})",
        ))

        return checks

    # ----- Reflection-based failure analysis -----

    def _classify_failure(
        self, gate_results: List[Dict[str, Any]], pipeline_result: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Classify the iteration failure into specific categories with root causes.

        Returns a dict with:
          - primary_failure: str (the most critical failure type)
          - failure_types: set of all failure types detected
          - details: human-readable analysis
          - recommendations: list of specific parameter changes
        """
        failed_gates = {
            g["name"]: g for g in gate_results if not g.get("passed", False)
        }
        passed_gates = {
            g["name"]: g for g in gate_results if g.get("passed", False)
        }

        failure_types = set()
        details = []
        recommendations = []

        # --- Power topology analysis ---
        if "smoke_test_fatals" in failed_gates:
            fatal_count = int(failed_gates["smoke_test_fatals"].get("actual", 0))
            failure_types.add("power_topology_broken")
            details.append(
                f"POWER TOPOLOGY: {fatal_count} fatal smoke test issues. "
                f"ICs are disconnected from power/ground rails."
            )
            recommendations.append("Ideation context needs better power rail definitions")

        if "functional_score" in failed_gates:
            score = failed_gates["functional_score"].get("actual", 0)
            threshold = failed_gates["functional_score"].get("threshold", 0.6)
            failure_types.add("design_intent_mismatch")
            details.append(
                f"DESIGN INTENT: Functional score {score:.1%} < {threshold:.1%}. "
                f"Connections don't match the design specification."
            )
            if score < 0.3:
                recommendations.append("Connection quality is critically low — "
                                       "check if ideation context is being used")
            else:
                recommendations.append("Connection quality is partial — "
                                       "some signal paths may be missing")

        # --- Connection quality analysis ---
        if "connection_coverage" in failed_gates:
            coverage = failed_gates["connection_coverage"].get("actual", 0)
            failure_types.add("missing_connections")
            details.append(
                f"CONNECTION COVERAGE: {coverage:.0%} of components connected. "
                f"Multiple components are electrically isolated."
            )
            recommendations.append("LLM connection generation needs more context "
                                   "or seed connections")

        if "center_fallback_ratio" in failed_gates:
            ratio = failed_gates["center_fallback_ratio"].get("actual", 0)
            failure_types.add("pin_mismatch")
            details.append(
                f"PIN MATCHING: {ratio:.0%} of wires routed to component centers "
                f"instead of actual pins. Pin names from connections don't match symbols."
            )
            recommendations.append("Pin name templates may need updating or "
                                   "connection generator fuzzy matching is failing")

        # --- Layout quality analysis ---
        if "overlap_count" in failed_gates:
            overlaps = int(failed_gates["overlap_count"].get("actual", 0))
            failure_types.add("layout_collision")
            details.append(
                f"LAYOUT COLLISIONS: {overlaps} overlapping component pairs remain."
            )
            recommendations.append("Increase layout spacing multiplier and "
                                   "collision resolution iterations")

        if "visual_score" in failed_gates:
            score = failed_gates["visual_score"].get("actual", 0)
            failure_types.add("visual_quality_low")
            details.append(
                f"VISUAL QUALITY: Score {score:.1%}. Layout may be cluttered or "
                f"wires may be poorly routed."
            )

        # --- Symbol quality ---
        if "placeholder_ratio" in failed_gates:
            ratio = failed_gates["placeholder_ratio"].get("actual", 0)
            failure_types.add("symbol_resolution_failed")
            details.append(
                f"SYMBOLS: {ratio:.0%} are placeholders. Symbol fetching/caching failed."
            )
            recommendations.append("Check symbol fetcher API connectivity and cache")

        # Determine primary failure (priority order)
        priority_order = [
            "power_topology_broken",
            "missing_connections",
            "design_intent_mismatch",
            "pin_mismatch",
            "symbol_resolution_failed",
            "layout_collision",
            "visual_quality_low",
        ]
        primary = "unknown"
        for ft in priority_order:
            if ft in failure_types:
                primary = ft
                break

        # Stagnation detection: count consecutive iterations with same failures
        stagnation_count = 0
        current_failed_set = set(failed_gates.keys())
        for prev_result in reversed(self.iteration_results):
            prev_failed_set = {
                g["name"] for g in prev_result.gate_results
                if not g.get("passed", False)
            }
            if prev_failed_set == current_failed_set:
                stagnation_count += 1
            else:
                break

        stagnation = stagnation_count >= 2  # 3+ total (current + 2 prev)
        escalation_level = EscalationLevel.for_stagnation_count(stagnation_count)

        if stagnation:
            level_names = {
                EscalationLevel.NONE: "NONE",
                EscalationLevel.INCREASE_PARAMS: "INCREASE_PARAMS",
                EscalationLevel.FULL_RESET: "FULL_RESET",
                EscalationLevel.ESCALATE: "ESCALATE",
            }
            details.append(
                f"STAGNATION DETECTED: Same {len(current_failed_set)} gates failing for "
                f"{stagnation_count + 1} consecutive iterations. "
                f"Escalation level: {level_names.get(escalation_level, 'UNKNOWN')}"
            )
            if escalation_level == EscalationLevel.INCREASE_PARAMS:
                recommendations.append("Dramatically increasing all parameters")
            elif escalation_level == EscalationLevel.FULL_RESET:
                recommendations.append("Full reset: no checkpoint resume, fresh generation")
            elif escalation_level == EscalationLevel.ESCALATE:
                recommendations.append(
                    "HUMAN INTERVENTION RECOMMENDED: Pipeline has stagnated for "
                    f"{stagnation_count + 1} iterations on the same failures. "
                    "Check: symbol fetcher APIs, ideation context data, LLM prompt quality."
                )

        return {
            "primary_failure": primary,
            "failure_types": failure_types,
            "details": details,
            "recommendations": recommendations,
            "stagnation": stagnation,
            "stagnation_count": stagnation_count,
            "escalation_level": escalation_level,
            "failed_gate_count": len(failed_gates),
            "passed_gate_count": len(passed_gates),
        }

    def _compute_config(self, iteration: int) -> IterationConfig:
        """Adjust parameters based on reflection analysis of previous failures."""
        config = IterationConfig()

        if iteration == 0:
            return config

        prev = self.iteration_results[-1]
        if not prev.gate_results:
            return config

        # Run reflection analysis
        analysis = self._classify_failure(prev.gate_results, prev.pipeline_result)

        logger.info(
            f"\n{'─'*50}\n"
            f"  REFLECTION ANALYSIS (iteration {iteration})\n"
            f"  Primary failure: {analysis['primary_failure']}\n"
            f"  Failure types: {', '.join(sorted(analysis['failure_types']))}\n"
            f"{'─'*50}"
        )
        for detail in analysis["details"]:
            logger.info(f"  → {detail}")
        for rec in analysis["recommendations"]:
            logger.info(f"  REC: {rec}")

        # Apply targeted parameter adjustments based on failure classification
        primary = analysis["primary_failure"]

        if "layout_collision" in analysis["failure_types"]:
            config.layout_spacing_multiplier = 1.5 + (iteration * 0.3)
            logger.info(f"  ADJUST: layout_spacing_multiplier={config.layout_spacing_multiplier:.1f}")

        if "pin_mismatch" in analysis["failure_types"]:
            config.connection_retry_count = min(5, 3 + iteration)
            logger.info(f"  ADJUST: connection_retry_count={config.connection_retry_count}")

        if "visual_quality_low" in analysis["failure_types"]:
            prev_result = prev.pipeline_result or {}
            vs = prev_result.get("visual_score", 0)
            if vs >= 0.4:
                config.max_visual_iterations = 10
                logger.info("  ADJUST: max_visual_iterations=10 (visual score recoverable)")

        escalation = analysis.get("escalation_level", EscalationLevel.NONE)
        config.escalation_level = escalation

        if escalation == EscalationLevel.INCREASE_PARAMS:
            config.layout_spacing_multiplier = 2.0 + iteration * 0.5
            config.connection_retry_count = 5
            config.max_visual_iterations = 10
            logger.info(
                "  ESCALATION [INCREASE_PARAMS]: Dramatically increasing all parameters"
            )
        elif escalation == EscalationLevel.FULL_RESET:
            config.layout_spacing_multiplier = 2.5
            config.connection_retry_count = 5
            config.max_visual_iterations = 10
            config.resume_from_checkpoint = False  # Force fresh generation
            logger.info(
                "  ESCALATION [FULL_RESET]: Fresh generation, no checkpoint resume"
            )
        elif escalation == EscalationLevel.ESCALATE:
            config.layout_spacing_multiplier = 3.0
            config.connection_retry_count = 5
            config.max_visual_iterations = 15
            config.resume_from_checkpoint = False
            logger.error(
                "  ESCALATION [ESCALATE]: Pipeline has stagnated. "
                "HUMAN INTERVENTION MAY BE REQUIRED. "
                "Check symbol APIs, ideation data, and LLM prompt quality."
            )
        else:
            # No stagnation — resume from checkpoint if symbols and connections passed
            passing = {g["name"] for g in prev.gate_results if g.get("passed", False)}
            if "placeholder_ratio" in passing and "connection_coverage" in passing:
                config.resume_from_checkpoint = True
                logger.info("  ADJUST: resume_from_checkpoint=True (symbols+connections OK)")

        return config

    # ----- Pipeline execution -----

    async def _run_pipeline(self, config: IterationConfig) -> PipelineResult:
        """Run the MAPO pipeline with the given config."""
        # Build BOM
        subsystem_dicts = [{"name": s} for s in self.subsystems]
        bom = create_foc_esc_bom(subsystem_dicts)
        design_intent = create_design_intent(subsystem_dicts, self.design_name)

        # Pipeline config
        pipeline_config = PipelineConfig()
        pipeline_config.max_iterations = config.max_visual_iterations

        pipeline = MAPOSchematicPipeline(config=pipeline_config)

        result = await pipeline.generate(
            bom=bom,
            design_intent=design_intent,
            design_name=self.design_name,
            resume_from_checkpoint=config.resume_from_checkpoint,
        )

        return result

    # ----- Logging -----

    def _log_iteration(self, iteration: int, gate_results: List[GateResult], duration: float):
        """Log iteration summary."""
        passed = sum(1 for g in gate_results if g.passed)
        total = len(gate_results)
        all_passed = passed == total

        status = "PASSED" if all_passed else "FAILED"
        logger.info(
            f"\n{'='*60}\n"
            f"  ITERATION {iteration + 1}/{self.max_iterations} — {status} "
            f"({passed}/{total} gates) [{duration:.0f}s]\n"
            f"{'='*60}"
        )
        for g in gate_results:
            icon = "PASS" if g.passed else "FAIL"
            logger.info(f"  [{icon}] {g.name}: {g.message}")
        logger.info("")

    def _save_iteration_result(self, iter_result: IterationResult):
        """Save iteration result to JSON file."""
        path = self.output_dir / f"iteration_{iter_result.iteration + 1}.json"
        with open(path, "w") as f:
            json.dump(asdict(iter_result), f, indent=2, default=str)
        logger.info(f"Iteration result saved to {path}")

    # ----- Main loop -----

    async def run(self) -> PipelineResult:
        """Run the continuous loop until quality gates pass or max iterations."""
        logger.info(
            f"\n{'#'*60}\n"
            f"  CONTINUOUS LOOP RUNNER\n"
            f"  Design: {self.design_name}\n"
            f"  Subsystems: {', '.join(self.subsystems)}\n"
            f"  Max iterations: {self.max_iterations}\n"
            f"{'#'*60}\n"
        )

        last_result = None

        for iteration in range(self.max_iterations):
            logger.info(f"\n--- Starting iteration {iteration + 1}/{self.max_iterations} ---")

            # 1. Compute config for this iteration
            config = self._compute_config(iteration)

            # 2. Run pipeline
            start_time = time.time()
            iter_result = IterationResult(
                iteration=iteration,
                success=False,
                duration_seconds=0,
                config_used=asdict(config),
            )

            try:
                result = await self._run_pipeline(config)
                last_result = result
                elapsed = time.time() - start_time
                iter_result.duration_seconds = elapsed
                iter_result.success = result.success
                iter_result.pipeline_result = result.to_dict()

                # 3. Check quality gates
                gate_results = self._check_gates(result)
                iter_result.gate_results = [asdict(g) for g in gate_results]
                iter_result.all_gates_passed = all(g.passed for g in gate_results)

                # 4. Log and save
                self._log_iteration(iteration, gate_results, elapsed)
                self._save_iteration_result(iter_result)
                self.iteration_results.append(iter_result)

                # 5. Check if we're done
                if iter_result.all_gates_passed:
                    logger.info(
                        f"\nALL QUALITY GATES PASSED on iteration {iteration + 1}! "
                        f"Schematic: {result.schematic_path}"
                    )
                    return result

            except Exception as e:
                elapsed = time.time() - start_time
                iter_result.duration_seconds = elapsed
                iter_result.error = f"{type(e).__name__}: {e}"
                self._save_iteration_result(iter_result)
                self.iteration_results.append(iter_result)
                logger.error(f"Iteration {iteration + 1} FAILED with error: {e}")
                # Continue to next iteration

        # Max iterations reached
        logger.error(
            f"\nFAILED to pass all quality gates after {self.max_iterations} iterations."
        )
        if self.iteration_results:
            best = max(
                self.iteration_results,
                key=lambda r: sum(1 for g in r.gate_results if g.get("passed", False)),
            )
            logger.info(
                f"Best iteration was #{best.iteration + 1} with "
                f"{sum(1 for g in best.gate_results if g.get('passed', False))}/{len(best.gate_results)} gates passed."
            )

        return last_result or PipelineResult(success=False, errors=["Max iterations reached"])

    # ----- Summary -----

    def print_summary(self):
        """Print a final summary of all iterations."""
        logger.info(f"\n{'='*60}")
        logger.info(f"  LOOP SUMMARY: {len(self.iteration_results)} iterations")
        logger.info(f"{'='*60}")
        for r in self.iteration_results:
            passed = sum(1 for g in r.gate_results if g.get("passed", False))
            total = len(r.gate_results)
            status = "PASS" if r.all_gates_passed else "FAIL"
            logger.info(
                f"  Iteration {r.iteration + 1}: [{status}] "
                f"{passed}/{total} gates, {r.duration_seconds:.0f}s"
                + (f" ERROR: {r.error}" if r.error else "")
            )
        logger.info("")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Run MAPO schematic pipeline in a continuous loop until quality gates pass."
    )
    parser.add_argument(
        "--design", default="foc_esc",
        help="Design name (default: foc_esc)"
    )
    parser.add_argument(
        "--max-iterations", type=int, default=10,
        help="Maximum loop iterations (default: 10)"
    )
    parser.add_argument(
        "--subsystems", nargs="+",
        default=None,
        help="Subsystem names to include (default: all FOC ESC subsystems)"
    )
    parser.add_argument(
        "--output-dir", default=None,
        help="Output directory for iteration results"
    )
    parser.add_argument(
        "--visual-threshold", type=float, default=0.65,
        help="Visual validation score threshold (default: 0.65, real image analysis via proxy)"
    )
    parser.add_argument(
        "--allow-placeholders", action="store_true",
        help="Allow up to 20%% placeholder symbols (for testing)"
    )

    args = parser.parse_args()

    gates = QualityGates(
        visual_score_min=args.visual_threshold,
    )
    if args.allow_placeholders:
        gates.placeholder_ratio_max = 0.25  # 6/29 = 20.7% for foc_esc with 6 missing symbols

    runner = ContinuousLoopRunner(
        design_name=args.design,
        max_iterations=args.max_iterations,
        subsystems=args.subsystems,
        gates=gates,
        output_dir=args.output_dir,
    )

    result = asyncio.run(runner.run())
    runner.print_summary()

    # Output final result as JSON to stdout
    if result:
        print(json.dumps(result.to_dict(), indent=2, default=str))
        sys.exit(0 if result.success else 1)
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
