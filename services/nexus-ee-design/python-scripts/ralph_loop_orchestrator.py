"""
Ralph Loop Orchestrator - MAPO v3.1 Iterative Improvement System

Named after Ralph Wiggum's self-referential "I'm a boy!" scene, this orchestrator
implements an iterative improvement pattern:

1. Generate schematic (iteration N)
2. Run smoke test + visual verification
3. If score < 100%, analyze what went wrong using Claude Opus 4.6
4. Feed failures back to MAPO as guidance
5. Regenerate schematic (iteration N+1)
6. Repeat until score = 100% OR max iterations (200) OR plateau detected

Philosophy: "Iterate Until Excellence or Exhaustion"

Author: Nexus EE Design Team
"""

import asyncio
import json
import logging
import os
import sys
import tempfile
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import httpx

# Add parent paths for imports
sys.path.insert(0, str(Path(__file__).parent))

from ideation_context import IdeationContext

# Import agents
from agents.smoke_test.smoke_test_agent import SmokeTestAgent, SmokeTestResult
from agents.visual_validator.dual_llm_validator import DualLLMVisualValidator, ComparisonResult

# Import MAPO optimizer
from mapos_v2_1_schematic.orchestrator.schematic_mapo_optimizer import (
    SchematicMAPOOptimizer,
    OptimizationResult,
)
from mapos_v2_1_schematic.core.config import SchematicMAPOConfig

logger = logging.getLogger(__name__)

# OpenRouter configuration for failure analysis
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_MODEL = "anthropic/claude-opus-4.6"  # Opus 4.6 for deep analysis


@dataclass
class IterationResult:
    """Results from a single Ralph loop iteration."""

    iteration: int
    schematic_path: str

    # Smoke test results
    smoke_test_passed: bool = False
    smoke_test_score: float = 0.0  # 0-100
    smoke_violations: List[Dict[str, Any]] = field(default_factory=list)

    # Visual verification results
    visual_test_passed: bool = False
    visual_test_score: float = 0.0  # 0-100
    visual_quality_issues: List[Dict[str, Any]] = field(default_factory=list)

    # Combined metrics
    overall_score: float = 0.0  # Weighted average of smoke + visual
    duration_seconds: float = 0.0
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    # Feedback for next iteration
    feedback_generated: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "iteration": self.iteration,
            "schematic_path": self.schematic_path,
            "smoke_test": {
                "passed": self.smoke_test_passed,
                "score": self.smoke_test_score,
                "violations": self.smoke_violations,
            },
            "visual_test": {
                "passed": self.visual_test_passed,
                "score": self.visual_test_score,
                "issues": self.visual_quality_issues,
            },
            "overall_score": self.overall_score,
            "duration_seconds": self.duration_seconds,
            "timestamp": self.timestamp,
            "feedback": self.feedback_generated,
        }


@dataclass
class RalphLoopResult:
    """Final results from Ralph loop execution."""

    success: bool
    final_schematic_path: str
    final_score: float
    total_iterations: int

    # Convergence metrics
    converged: bool  # True if reached 100% or plateaued gracefully
    convergence_iteration: Optional[int]  # When it reached target

    # Iteration history
    iterations: List[IterationResult] = field(default_factory=list)

    # Performance metrics
    total_duration_seconds: float = 0.0
    improvement_graph: Dict[int, float] = field(default_factory=dict)  # iteration -> score

    # Analysis
    failure_analysis: List[str] = field(default_factory=list)
    plateau_detected: bool = False
    plateau_iteration: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "success": self.success,
            "final_schematic_path": self.final_schematic_path,
            "final_score": self.final_score,
            "total_iterations": self.total_iterations,
            "converged": self.converged,
            "convergence_iteration": self.convergence_iteration,
            "total_duration_seconds": self.total_duration_seconds,
            "plateau_detected": self.plateau_detected,
            "plateau_iteration": self.plateau_iteration,
            "iterations": [it.to_dict() for it in self.iterations],
            "improvement_graph": self.improvement_graph,
            "failure_analysis": self.failure_analysis,
        }


class RalphLoopOrchestrator:
    """
    Ralph Loop Orchestrator - Iterative schematic improvement system.

    Wraps MAPO pipeline with smoke test + visual verification loop,
    feeding failures back to guide next iteration.
    """

    def __init__(
        self,
        max_iterations: int = 200,
        target_score: float = 100.0,
        plateau_threshold: int = 20,
        smoke_test_weight: float = 0.6,
        visual_test_weight: float = 0.4,
        config: Optional[SchematicMAPOConfig] = None,
    ):
        """
        Initialize Ralph loop orchestrator.

        Args:
            max_iterations: Hard cap on iterations (default: 200)
            target_score: Stop when reaching this score (0-100, default: 100.0)
            plateau_threshold: Stop if score doesn't improve for N iterations (default: 20)
            smoke_test_weight: Weight for smoke test in overall score (default: 0.6)
            visual_test_weight: Weight for visual test in overall score (default: 0.4)
            config: Optional SchematicMAPOConfig for MAPO optimizer
        """
        self.max_iterations = max_iterations
        self.target_score = target_score
        self.plateau_threshold = plateau_threshold
        self.smoke_test_weight = smoke_test_weight
        self.visual_test_weight = visual_test_weight
        self.config = config

        # Initialize components (lazy loading)
        self._smoke_tester: Optional[SmokeTestAgent] = None
        self._visual_verifier: Optional[DualLLMVisualValidator] = None
        self._mapo_optimizer: Optional[SchematicMAPOOptimizer] = None

        # HTTP client for LLM calls
        self._http_client: Optional[httpx.AsyncClient] = None

    def _ensure_smoke_tester(self) -> SmokeTestAgent:
        """Get or create smoke test agent."""
        if self._smoke_tester is None:
            self._smoke_tester = SmokeTestAgent()
        return self._smoke_tester

    def _ensure_visual_verifier(self) -> DualLLMVisualValidator:
        """Get or create visual verifier."""
        if self._visual_verifier is None:
            self._visual_verifier = DualLLMVisualValidator()
        return self._visual_verifier

    def _ensure_mapo_optimizer(self) -> SchematicMAPOOptimizer:
        """Get or create MAPO optimizer."""
        if self._mapo_optimizer is None:
            self._mapo_optimizer = SchematicMAPOOptimizer(config=self.config)
        return self._mapo_optimizer

    def _ensure_http_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client."""
        if self._http_client is None:
            self._http_client = httpx.AsyncClient(timeout=120.0)
        return self._http_client

    async def close(self):
        """Close all resources."""
        if self._mapo_optimizer:
            await self._mapo_optimizer.close()
        if self._http_client:
            await self._http_client.aclose()

    async def run(
        self,
        bom: List[Dict[str, Any]],
        design_intent: str,
        project_name: str,
        design_type: str = "foc_esc",
        connections: Optional[List[Dict]] = None,
        ideation_context: Optional[IdeationContext] = None,
        operation_id: Optional[str] = None,
    ) -> RalphLoopResult:
        """
        Run Ralph loop until reaching target quality or max iterations.

        Args:
            bom: Bill of materials
            design_intent: Design requirements text
            project_name: Project name for output files
            design_type: Type of design for pattern matching (default: "foc_esc")
            connections: Optional seed connections
            ideation_context: Optional IdeationContext with design hints
            operation_id: For WebSocket progress updates

        Returns:
            RalphLoopResult with final schematic and iteration history
        """
        start_time = time.time()
        iterations: List[IterationResult] = []
        improvement_graph: Dict[int, float] = {}
        failure_analysis: List[str] = []

        best_score = 0.0
        best_schematic_path = ""
        iterations_without_improvement = 0

        logger.info(f"=== RALPH LOOP START: {project_name} ===")
        logger.info(f"Target: {self.target_score}%, Max iterations: {self.max_iterations}")
        logger.info(f"Plateau threshold: {self.plateau_threshold} iterations")

        # Prepare feedback accumulator (grows with each iteration)
        accumulated_feedback = ""

        for iteration in range(1, self.max_iterations + 1):
            iter_start_time = time.time()

            logger.info(f"\n{'='*80}")
            logger.info(f"RALPH LOOP ITERATION {iteration}/{self.max_iterations}")
            logger.info(f"{'='*80}")

            # Emit progress
            if operation_id:
                await self._emit_progress(
                    operation_id,
                    f"Ralph iteration {iteration}: Generating schematic",
                    (iteration / self.max_iterations) * 100
                )

            # ============================================================
            # STEP 1: Generate/Regenerate Schematic
            # ============================================================
            try:
                logger.info(f"[Iteration {iteration}] Generating schematic...")

                # Build enhanced design intent with accumulated feedback
                enhanced_design_intent = design_intent
                if accumulated_feedback:
                    enhanced_design_intent = (
                        f"{design_intent}\n\n"
                        f"=== FEEDBACK FROM PREVIOUS ITERATIONS ===\n"
                        f"{accumulated_feedback}"
                    )
                    logger.info(f"Including feedback from {len(iterations)} previous iterations")

                # Run MAPO optimization
                optimizer = self._ensure_mapo_optimizer()
                optimization_result: OptimizationResult = await optimizer.optimize(
                    bom=bom,
                    design_intent=enhanced_design_intent,
                    design_name=f"{project_name}_iter_{iteration}",
                    design_type=design_type,
                    max_iterations=50,  # MAPO internal iterations (reduced for Ralph loop)
                    ideation_context=ideation_context,
                )

                if not optimization_result.success or not optimization_result.schematic_path:
                    logger.error(f"[Iteration {iteration}] Schematic generation failed")
                    failure_analysis.append(
                        f"Iteration {iteration}: Schematic generation failed - {optimization_result.errors}"
                    )
                    continue

                schematic_path = str(optimization_result.schematic_path)
                logger.info(f"[Iteration {iteration}] Schematic generated: {schematic_path}")

            except Exception as exc:
                logger.error(f"[Iteration {iteration}] Generation exception: {exc}", exc_info=True)
                failure_analysis.append(f"Iteration {iteration}: Generation exception - {exc}")
                continue

            # ============================================================
            # STEP 2: Run Smoke Test
            # ============================================================
            if operation_id:
                await self._emit_progress(
                    operation_id,
                    f"Ralph iteration {iteration}: Running smoke test",
                    (iteration / self.max_iterations) * 100
                )

            try:
                logger.info(f"[Iteration {iteration}] Running smoke test...")
                smoke_tester = self._ensure_smoke_tester()

                # Read schematic content
                with open(schematic_path, 'r') as f:
                    kicad_content = f.read()

                # Run smoke test
                smoke_result: SmokeTestResult = await smoke_tester.run_smoke_test(
                    kicad_sch_content=kicad_content,
                    bom_items=bom,
                    power_sources=self._detect_power_sources(bom),
                )

                # Calculate smoke test score (0-100)
                smoke_score = self._calculate_smoke_score(smoke_result)

                logger.info(
                    f"[Iteration {iteration}] Smoke test: "
                    f"{'PASS' if smoke_result.passed else 'FAIL'} "
                    f"(score: {smoke_score:.1f}%)"
                )

                # Extract violations
                smoke_violations = [issue.to_dict() for issue in smoke_result.issues]

            except Exception as exc:
                logger.error(f"[Iteration {iteration}] Smoke test exception: {exc}", exc_info=True)
                smoke_result = None
                smoke_score = 0.0
                smoke_violations = [{"severity": "fatal", "message": str(exc)}]

            # ============================================================
            # STEP 3: Run Visual Verification
            # ============================================================
            if operation_id:
                await self._emit_progress(
                    operation_id,
                    f"Ralph iteration {iteration}: Running visual verification",
                    (iteration / self.max_iterations) * 100
                )

            try:
                logger.info(f"[Iteration {iteration}] Running visual verification...")
                visual_verifier = self._ensure_visual_verifier()

                # Run visual validation
                comparison_result: ComparisonResult = await visual_verifier.validate_schematic(
                    schematic_path=schematic_path
                )

                # Calculate visual score (0-100)
                visual_score = comparison_result.combined_score * 100.0
                visual_passed = visual_score >= 90.0

                logger.info(
                    f"[Iteration {iteration}] Visual test: "
                    f"{'PASS' if visual_passed else 'FAIL'} "
                    f"(score: {visual_score:.1f}%)"
                )

                # Extract quality issues
                visual_issues = []
                for issue in comparison_result.agreed_issues:
                    visual_issues.append({
                        "category": issue.category.value,
                        "severity": issue.severity.value,
                        "description": issue.description,
                        "recommendation": issue.recommendation,
                    })

            except Exception as exc:
                logger.error(f"[Iteration {iteration}] Visual verification exception: {exc}", exc_info=True)
                comparison_result = None
                visual_score = 0.0
                visual_passed = False
                visual_issues = [{"severity": "error", "message": str(exc)}]

            # ============================================================
            # STEP 4: Calculate Overall Score
            # ============================================================
            overall_score = (
                smoke_score * self.smoke_test_weight +
                visual_score * self.visual_test_weight
            )

            iter_duration = time.time() - iter_start_time

            logger.info(f"[Iteration {iteration}] OVERALL SCORE: {overall_score:.1f}%")
            logger.info(
                f"  Breakdown: Smoke={smoke_score:.1f}% (weight={self.smoke_test_weight}), "
                f"Visual={visual_score:.1f}% (weight={self.visual_test_weight})"
            )

            # ============================================================
            # STEP 5: Generate Feedback for Next Iteration
            # ============================================================
            feedback = ""
            if overall_score < self.target_score:
                logger.info(f"[Iteration {iteration}] Generating feedback for next iteration...")
                feedback = await self._generate_feedback(
                    iteration=iteration,
                    smoke_result=smoke_result,
                    smoke_violations=smoke_violations,
                    visual_issues=visual_issues,
                    design_intent=design_intent,
                )

                # Accumulate feedback (keep last 3 iterations)
                if accumulated_feedback:
                    accumulated_feedback = f"{accumulated_feedback}\n\n{feedback}"
                    # Trim to last 3 iterations worth of feedback (rough heuristic)
                    lines = accumulated_feedback.split('\n')
                    if len(lines) > 100:  # Roughly 3 iterations
                        accumulated_feedback = '\n'.join(lines[-100:])
                else:
                    accumulated_feedback = feedback

            # ============================================================
            # STEP 6: Record Iteration
            # ============================================================
            iter_result = IterationResult(
                iteration=iteration,
                schematic_path=schematic_path,
                smoke_test_passed=smoke_result.passed if smoke_result else False,
                smoke_test_score=smoke_score,
                smoke_violations=smoke_violations,
                visual_test_passed=visual_passed,
                visual_test_score=visual_score,
                visual_quality_issues=visual_issues,
                overall_score=overall_score,
                duration_seconds=iter_duration,
                feedback_generated=feedback,
            )

            iterations.append(iter_result)
            improvement_graph[iteration] = overall_score

            # ============================================================
            # STEP 7: Check Convergence Criteria
            # ============================================================

            # Track best score
            if overall_score > best_score:
                best_score = overall_score
                best_schematic_path = schematic_path
                iterations_without_improvement = 0
                logger.info(f"✅ NEW BEST SCORE: {best_score:.1f}%")
            else:
                iterations_without_improvement += 1
                logger.info(
                    f"⚠️  No improvement for {iterations_without_improvement} iterations "
                    f"(plateau threshold: {self.plateau_threshold})"
                )

            # Check if target reached
            if overall_score >= self.target_score:
                total_duration = time.time() - start_time
                logger.info(f"\n🎉 TARGET REACHED at iteration {iteration}: {overall_score:.1f}%")

                return RalphLoopResult(
                    success=True,
                    final_schematic_path=best_schematic_path,
                    final_score=best_score,
                    total_iterations=iteration,
                    converged=True,
                    convergence_iteration=iteration,
                    iterations=iterations,
                    total_duration_seconds=total_duration,
                    improvement_graph=improvement_graph,
                    failure_analysis=failure_analysis,
                    plateau_detected=False,
                )

            # Check for plateau
            if iterations_without_improvement >= self.plateau_threshold:
                logger.warning(
                    f"\n⚠️  PLATEAU DETECTED at iteration {iteration}: "
                    f"No improvement for {self.plateau_threshold} iterations"
                )

                # Analyze plateau
                plateau_analysis = self._analyze_plateau(iterations)
                failure_analysis.append(f"Plateau detected at iteration {iteration}: {plateau_analysis}")

                total_duration = time.time() - start_time

                return RalphLoopResult(
                    success=best_score >= 90.0,  # Accept if >= 90%
                    final_schematic_path=best_schematic_path,
                    final_score=best_score,
                    total_iterations=iteration,
                    converged=True,
                    convergence_iteration=None,
                    iterations=iterations,
                    total_duration_seconds=total_duration,
                    improvement_graph=improvement_graph,
                    failure_analysis=failure_analysis,
                    plateau_detected=True,
                    plateau_iteration=iteration,
                )

        # ============================================================
        # Max Iterations Reached
        # ============================================================
        total_duration = time.time() - start_time
        logger.warning(
            f"\n⚠️  MAX ITERATIONS REACHED ({self.max_iterations}): "
            f"Best score = {best_score:.1f}%"
        )

        return RalphLoopResult(
            success=best_score >= 90.0,  # Accept if >= 90%
            final_schematic_path=best_schematic_path,
            final_score=best_score,
            total_iterations=self.max_iterations,
            converged=False,
            convergence_iteration=None,
            iterations=iterations,
            total_duration_seconds=total_duration,
            improvement_graph=improvement_graph,
            failure_analysis=failure_analysis,
            plateau_detected=False,
        )

    def _detect_power_sources(self, bom: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Detect power sources from BOM.

        Looks for power symbols (VCC, VDD, GND) and infers voltage levels.
        """
        power_sources = []

        for item in bom:
            part_number = item.get("part_number", "").upper()
            value = item.get("value", "").upper()

            # Check for power rail symbols
            if part_number in ("VCC", "VDD", "VIN", "V+"):
                # Try to infer voltage from value or default to 5V
                voltage = 5.0
                if "3.3V" in value or "3V3" in value:
                    voltage = 3.3
                elif "12V" in value:
                    voltage = 12.0
                elif "48V" in value:
                    voltage = 48.0

                power_sources.append({
                    "net": part_number,
                    "voltage": voltage,
                    "current_limit": 1.0,  # Conservative default
                })

        # Always add ground
        if not any(ps["net"] == "GND" for ps in power_sources):
            power_sources.append({
                "net": "GND",
                "voltage": 0.0,
                "current_limit": 100.0,  # Ground can handle high current
            })

        return power_sources

    def _calculate_smoke_score(self, smoke_result: SmokeTestResult) -> float:
        """
        Calculate smoke test score (0-100) from SmokeTestResult.

        Score is based on:
        - Pass/fail status
        - Number and severity of issues
        - Individual check results
        """
        if smoke_result.passed:
            return 100.0

        # Start with baseline score
        score = 50.0

        # Penalize based on issue severity
        for issue in smoke_result.issues:
            if issue.severity.value == "fatal":
                score -= 15.0
            elif issue.severity.value == "error":
                score -= 8.0
            elif issue.severity.value == "warning":
                score -= 3.0

        # Bonus for passing individual checks
        checks = [
            smoke_result.power_rails_ok,
            smoke_result.ground_ok,
            smoke_result.no_shorts,
            smoke_result.no_floating_nodes,
            smoke_result.power_dissipation_ok,
            smoke_result.current_paths_valid,
        ]
        passed_checks = sum(1 for c in checks if c)
        score += (passed_checks / len(checks)) * 30.0

        return max(0.0, min(100.0, score))

    async def _generate_feedback(
        self,
        iteration: int,
        smoke_result: Optional[SmokeTestResult],
        smoke_violations: List[Dict[str, Any]],
        visual_issues: List[Dict[str, Any]],
        design_intent: str,
    ) -> str:
        """
        Generate feedback for next iteration using Claude Opus 4.6.

        Analyzes failures and provides actionable guidance for MAPO optimizer.
        """
        if not OPENROUTER_API_KEY:
            logger.warning("OPENROUTER_API_KEY not set - using heuristic feedback")
            return self._generate_heuristic_feedback(smoke_violations, visual_issues)

        # Build comprehensive failure report
        violations_text = "\n".join([
            f"- [{v.get('severity', 'unknown')}] {v.get('message', 'No message')}"
            for v in smoke_violations[:10]  # Limit to top 10
        ])

        issues_text = "\n".join([
            f"- [{i.get('category', 'unknown')}] {i.get('description', 'No description')}"
            for i in visual_issues[:10]  # Limit to top 10
        ])

        prompt = f"""You are analyzing a failed schematic generation attempt (iteration {iteration}) to guide the next iteration.

DESIGN INTENT:
{design_intent[:1000]}

SMOKE TEST VIOLATIONS ({len(smoke_violations)} total):
{violations_text if violations_text else "None"}

VISUAL QUALITY ISSUES ({len(visual_issues)} total):
{issues_text if issues_text else "None"}

Provide specific, actionable guidance for the MAPO optimizer to fix these issues in the next iteration.

Focus on:
1. Component placement changes needed (specific reference designators)
2. Wire routing improvements (which nets need attention)
3. Voltage/current issues to resolve (specific power rails)
4. Layout organization changes (grouping, signal flow)

Output concise bullet points (max 5 items, each < 100 chars).
Format as plain text bullet points starting with "- ".
"""

        try:
            client = self._ensure_http_client()

            response = await client.post(
                OPENROUTER_BASE_URL,
                headers={
                    "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": OPENROUTER_MODEL,
                    "messages": [
                        {"role": "user", "content": prompt}
                    ],
                    "max_tokens": 500,
                    "temperature": 0.3,  # Low temperature for focused feedback
                }
            )

            if response.status_code != 200:
                logger.warning(f"LLM feedback failed: {response.status_code} - {response.text}")
                return self._generate_heuristic_feedback(smoke_violations, visual_issues)

            result = response.json()
            feedback = result.get("choices", [{}])[0].get("message", {}).get("content", "")

            if not feedback:
                logger.warning("Empty LLM feedback - using heuristic")
                return self._generate_heuristic_feedback(smoke_violations, visual_issues)

            logger.info(f"Generated LLM feedback ({len(feedback)} chars)")
            return feedback

        except Exception as exc:
            logger.error(f"LLM feedback exception: {exc}", exc_info=True)
            return self._generate_heuristic_feedback(smoke_violations, visual_issues)

    def _generate_heuristic_feedback(
        self,
        smoke_violations: List[Dict[str, Any]],
        visual_issues: List[Dict[str, Any]],
    ) -> str:
        """
        Generate heuristic feedback when LLM is unavailable.

        Falls back to rule-based feedback generation.
        """
        feedback_items = []

        # Analyze smoke violations
        for violation in smoke_violations[:5]:
            severity = violation.get("severity", "")
            message = violation.get("message", "")

            if "power" in message.lower() or "vcc" in message.lower():
                feedback_items.append("- Fix power rail connectivity issues")
            elif "ground" in message.lower() or "gnd" in message.lower():
                feedback_items.append("- Ensure proper ground connections")
            elif "short" in message.lower():
                feedback_items.append("- Remove short circuits between power and ground")
            elif "float" in message.lower():
                feedback_items.append("- Connect floating nodes to valid nets")

        # Analyze visual issues
        for issue in visual_issues[:5]:
            category = issue.get("category", "")

            if category == "wire_routing":
                feedback_items.append("- Improve wire routing: reduce crossings, use orthogonal paths")
            elif category == "component_placement":
                feedback_items.append("- Optimize component placement for signal flow")
            elif category == "label_placement":
                feedback_items.append("- Adjust label positions to avoid overlaps")
            elif category == "readability":
                feedback_items.append("- Improve overall schematic readability and organization")

        # Deduplicate
        feedback_items = list(dict.fromkeys(feedback_items))[:5]

        if not feedback_items:
            feedback_items = ["- Review overall design for common schematic issues"]

        return "\n".join(feedback_items)

    def _analyze_plateau(self, iterations: List[IterationResult]) -> str:
        """
        Analyze why the score plateaued.

        Looks for recurring patterns in failures.
        """
        if len(iterations) < 3:
            return "Insufficient iterations for analysis"

        # Get last 5 iterations
        recent = iterations[-5:]

        # Check for recurring violations
        violation_counts: Dict[str, int] = {}
        for iter_result in recent:
            for violation in iter_result.smoke_violations:
                key = violation.get("message", "unknown")[:50]  # First 50 chars
                violation_counts[key] = violation_counts.get(key, 0) + 1

        recurring = [k for k, v in violation_counts.items() if v >= 3]

        if recurring:
            return f"Recurring violations: {', '.join(recurring[:3])}"
        else:
            return "Score oscillating without consistent improvement pattern"

    async def _emit_progress(
        self,
        operation_id: str,
        message: str,
        progress_pct: float
    ):
        """
        Emit WebSocket progress update.

        In production, this would send updates via WebSocket manager.
        For now, just log to console in structured format.
        """
        progress_data = {
            "operation_id": operation_id,
            "message": message,
            "progress": progress_pct,
            "timestamp": datetime.now().isoformat(),
        }

        # Output as JSON to stdout for WebSocket manager to pick up
        print(f"PROGRESS:{json.dumps(progress_data)}", flush=True)

        # Also log normally
        logger.info(f"[{operation_id}] {message} ({progress_pct:.1f}%)")


# ============================================================
# CLI Interface (for testing)
# ============================================================

async def main():
    """CLI entry point for testing Ralph loop."""
    import argparse

    parser = argparse.ArgumentParser(description="Ralph Loop Orchestrator")
    parser.add_argument("--bom-json", required=True, help="Path to BOM JSON file")
    parser.add_argument("--design-intent", required=True, help="Design intent text")
    parser.add_argument("--project-name", required=True, help="Project name")
    parser.add_argument("--max-iterations", type=int, default=200, help="Max iterations")
    parser.add_argument("--target-score", type=float, default=100.0, help="Target score (0-100)")
    parser.add_argument("--output-json", help="Output results to JSON file")

    args = parser.parse_args()

    # Load BOM
    with open(args.bom_json, 'r') as f:
        bom = json.load(f)

    # Create orchestrator
    orchestrator = RalphLoopOrchestrator(
        max_iterations=args.max_iterations,
        target_score=args.target_score,
    )

    try:
        # Run loop
        result = await orchestrator.run(
            bom=bom,
            design_intent=args.design_intent,
            project_name=args.project_name,
        )

        # Output results
        result_dict = result.to_dict()

        if args.output_json:
            with open(args.output_json, 'w') as f:
                json.dump(result_dict, f, indent=2)
            print(f"Results written to {args.output_json}")
        else:
            print(json.dumps(result_dict, indent=2))

        # Exit code
        sys.exit(0 if result.success else 1)

    finally:
        await orchestrator.close()


if __name__ == "__main__":
    asyncio.run(main())
