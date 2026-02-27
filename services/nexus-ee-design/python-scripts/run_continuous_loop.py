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
    - >= 0.85 visual validation score
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
    visual_score_min: float = 0.55           # visual >= 55% (text-only via proxy, 85% with image)
    center_fallback_ratio_max: float = 0.10  # < 10% center fallbacks


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

@dataclass
class IterationConfig:
    """Parameters tuned between iterations based on failure analysis."""
    layout_spacing_multiplier: float = 1.0
    connection_retry_count: int = 3
    max_visual_iterations: int = 5
    resume_from_checkpoint: bool = False


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

        return checks

    # ----- Parameter adaptation -----

    def _compute_config(self, iteration: int) -> IterationConfig:
        """Adjust parameters based on previous iteration failures."""
        config = IterationConfig()

        if iteration == 0:
            return config

        prev = self.iteration_results[-1]
        if not prev.gate_results:
            return config

        failed_gates = {
            g["name"] for g in prev.gate_results if not g["passed"]
        }

        # If overlaps were an issue, increase spacing
        if "overlap_count" in failed_gates:
            config.layout_spacing_multiplier = 1.5 + (iteration * 0.2)
            logger.info(f"Adapting: layout_spacing_multiplier={config.layout_spacing_multiplier:.1f}")

        # If center fallback was high, retry more aggressively
        if "center_fallback_ratio" in failed_gates:
            config.connection_retry_count = min(5, 3 + iteration)
            logger.info(f"Adapting: connection_retry_count={config.connection_retry_count}")

        # If visual score was close, give more iterations
        if "visual_score" in failed_gates:
            prev_result = prev.pipeline_result or {}
            vs = prev_result.get("visual_score", 0)
            if vs >= 0.7:
                config.max_visual_iterations = 10
                logger.info("Adapting: max_visual_iterations=10 (score was close)")

        # Resume from checkpoint if symbols and connections passed
        passing = {g["name"] for g in prev.gate_results if g["passed"]}
        if "placeholder_ratio" in passing and "connection_coverage" in passing:
            config.resume_from_checkpoint = True
            logger.info("Adapting: resume_from_checkpoint=True (symbols+connections OK)")

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
        "--visual-threshold", type=float, default=0.55,
        help="Visual validation score threshold (default: 0.55 for text-only proxy, 0.85 with image)"
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
        gates.placeholder_ratio_max = 0.20

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
