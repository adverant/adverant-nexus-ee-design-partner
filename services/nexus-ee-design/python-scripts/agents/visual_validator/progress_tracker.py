"""
Progress Tracker - Tracks validation loop progress and detects stagnation.

Monitors score changes, issue counts, and visual changes across MAPO iterations
to detect when the schematic is NOT progressing despite fix attempts.

NO FALLBACKS - Strict stagnation detection with verbose reporting.

Author: Nexus EE Design Team
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .dual_llm_validator import ComparisonResult, VisualIssue

logger = logging.getLogger(__name__)


class StagnationRisk(Enum):
    """Risk level for stagnation."""
    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class StagnationReason(Enum):
    """Reasons for detected stagnation."""
    SCORE_PLATEAU = "score_plateau"  # Score not improving
    ISSUES_UNCHANGED = "issues_unchanged"  # Same issues persist
    VISUAL_UNCHANGED = "visual_unchanged"  # Image hash unchanged
    OSCILLATING = "oscillating"  # Score going up and down
    CRITICAL_UNRESOLVED = "critical_unresolved"  # Critical issues not fixed
    FIX_REGRESSION = "fix_regression"  # Fixes made score worse


class StagnationError(Exception):
    """
    Raised when stagnation is detected and loop should terminate.

    Contains detailed context for debugging why validation is stuck.
    """

    def __init__(
        self,
        reason: StagnationReason,
        consecutive_stagnant: int,
        history_summary: str,
        recommendations: List[str]
    ):
        self.reason = reason
        self.consecutive_stagnant = consecutive_stagnant
        self.history_summary = history_summary
        self.recommendations = recommendations

        full_message = f"""
================================================================================
VALIDATION STAGNATION DETECTED - LOOP TERMINATED
================================================================================
Reason: {reason.value}
Consecutive Stagnant Iterations: {consecutive_stagnant}

History Summary:
{history_summary}

Recommendations:
{chr(10).join(f"  - {r}" for r in recommendations)}

This indicates the MAPO loop cannot improve the schematic further with current
strategies. Manual intervention or different approach may be required.
================================================================================
"""
        super().__init__(full_message)


@dataclass
class ProgressSnapshot:
    """Snapshot of validation state at a single iteration."""
    iteration: int
    combined_score: float
    agreement_score: float
    total_issues: int
    critical_issues: int
    error_issues: int
    warning_issues: int
    agreed_issues: int
    unique_opus_issues: int
    unique_kimi_issues: int
    image_hash: str
    timestamp: datetime = field(default_factory=datetime.utcnow)

    # Computed delta from previous (populated by tracker)
    score_delta: float = 0.0
    issue_delta: int = 0
    critical_delta: int = 0

    def to_dict(self) -> dict:
        return {
            "iteration": self.iteration,
            "combined_score": self.combined_score,
            "agreement_score": self.agreement_score,
            "total_issues": self.total_issues,
            "critical_issues": self.critical_issues,
            "error_issues": self.error_issues,
            "warning_issues": self.warning_issues,
            "score_delta": self.score_delta,
            "issue_delta": self.issue_delta,
            "image_hash": self.image_hash[:8] if self.image_hash else "N/A",
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class ProgressAnalysis:
    """Analysis of progress across iterations."""
    is_progressing: bool
    progress_score: float  # 0.0 = fully stagnant, 1.0 = strong progress
    stagnation_risk: StagnationRisk
    stagnation_reason: Optional[StagnationReason] = None
    consecutive_stagnant_iterations: int = 0
    recommended_action: str = "continue"  # "continue" | "escalate" | "terminate"
    details: Dict[str, any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "is_progressing": self.is_progressing,
            "progress_score": self.progress_score,
            "stagnation_risk": self.stagnation_risk.value,
            "stagnation_reason": self.stagnation_reason.value if self.stagnation_reason else None,
            "consecutive_stagnant": self.consecutive_stagnant_iterations,
            "recommended_action": self.recommended_action,
            "details": self.details,
        }


class ProgressTracker:
    """
    Tracks progress across MAPO validation iterations.

    Detects stagnation patterns and recommends actions.

    NO FALLBACKS - Strict stagnation detection.

    Usage:
        tracker = ProgressTracker()
        for iteration in range(max_iterations):
            result = await validator.validate(...)
            image_result = await extractor.extract_png(...)

            analysis = tracker.record_and_analyze(result, image_result.image_hash)

            if analysis.recommended_action == "terminate":
                raise StagnationError(...)
    """

    # Stagnation thresholds
    SCORE_IMPROVEMENT_THRESHOLD = 0.02  # Less than 2% improvement = stagnant
    STAGNANT_ITERATIONS_WARNING = 2  # Warn after 2 stagnant iterations
    STAGNANT_ITERATIONS_TERMINATE = 3  # Terminate after 3 stagnant iterations

    # Progress scoring weights
    WEIGHT_SCORE_IMPROVEMENT = 0.40
    WEIGHT_ISSUE_REDUCTION = 0.30
    WEIGHT_CRITICAL_RESOLUTION = 0.30

    def __init__(
        self,
        stagnation_threshold: float = 0.02,
        max_stagnant_iterations: int = 3,
        detect_oscillation: bool = True
    ):
        """
        Initialize progress tracker.

        Args:
            stagnation_threshold: Score improvement below this is considered stagnant
            max_stagnant_iterations: Terminate after this many stagnant iterations
            detect_oscillation: Whether to detect score oscillation patterns
        """
        self.stagnation_threshold = stagnation_threshold
        self.max_stagnant_iterations = max_stagnant_iterations
        self.detect_oscillation = detect_oscillation

        self.history: List[ProgressSnapshot] = []
        self.consecutive_stagnant: int = 0
        self.best_score: float = 0.0
        self.best_iteration: int = 0

        logger.info(
            f"ProgressTracker initialized: "
            f"threshold={stagnation_threshold}, max_stagnant={max_stagnant_iterations}"
        )

    def reset(self):
        """Reset tracker for a new validation run."""
        self.history.clear()
        self.consecutive_stagnant = 0
        self.best_score = 0.0
        self.best_iteration = 0
        logger.info("ProgressTracker reset")

    def record_and_analyze(
        self,
        comparison_result: 'ComparisonResult',
        image_hash: str,
        issues: Optional[List['VisualIssue']] = None
    ) -> ProgressAnalysis:
        """
        Record current iteration and analyze progress.

        Args:
            comparison_result: Result from DualLLMVisualValidator
            image_hash: MD5 hash of rendered schematic image
            issues: Optional list of visual issues for detailed analysis

        Returns:
            ProgressAnalysis with stagnation detection
        """
        iteration = len(self.history)

        # Count issues by severity
        critical_count = 0
        error_count = 0
        warning_count = 0

        all_issues = (
            comparison_result.agreed_issues +
            comparison_result.unique_opus_issues +
            comparison_result.unique_kimi_issues
        )

        for issue in all_issues:
            severity = issue.severity.value if hasattr(issue.severity, 'value') else str(issue.severity)
            if severity == "critical":
                critical_count += 1
            elif severity == "error":
                error_count += 1
            elif severity == "warning":
                warning_count += 1

        # Create snapshot
        snapshot = ProgressSnapshot(
            iteration=iteration,
            combined_score=comparison_result.combined_score,
            agreement_score=comparison_result.agreement_score,
            total_issues=len(all_issues),
            critical_issues=critical_count,
            error_issues=error_count,
            warning_issues=warning_count,
            agreed_issues=len(comparison_result.agreed_issues),
            unique_opus_issues=len(comparison_result.unique_opus_issues),
            unique_kimi_issues=len(comparison_result.unique_kimi_issues),
            image_hash=image_hash,
        )

        # Calculate deltas from previous
        if self.history:
            prev = self.history[-1]
            snapshot.score_delta = snapshot.combined_score - prev.combined_score
            snapshot.issue_delta = snapshot.total_issues - prev.total_issues
            snapshot.critical_delta = snapshot.critical_issues - prev.critical_issues

        self.history.append(snapshot)

        # Update best score tracking
        if snapshot.combined_score > self.best_score:
            self.best_score = snapshot.combined_score
            self.best_iteration = iteration

        # Analyze progress
        analysis = self._analyze_progress(snapshot)

        # Log progress
        self._log_iteration(snapshot, analysis)

        return analysis

    def _analyze_progress(self, current: ProgressSnapshot) -> ProgressAnalysis:
        """Analyze progress based on current and historical snapshots."""

        # First iteration - no comparison possible
        if len(self.history) == 1:
            return ProgressAnalysis(
                is_progressing=True,
                progress_score=1.0,
                stagnation_risk=StagnationRisk.NONE,
                recommended_action="continue",
                details={"note": "First iteration - baseline established"}
            )

        prev = self.history[-2]

        # Calculate individual progress indicators
        score_improving = current.score_delta >= self.stagnation_threshold
        issues_reducing = current.issue_delta < 0
        critical_resolved = current.critical_delta < 0
        visual_changed = current.image_hash != prev.image_hash

        # Check for regression (score got worse)
        score_regressed = current.score_delta < -self.stagnation_threshold

        # Calculate progress score (0-1)
        progress_score = 0.0

        if score_improving:
            # Normalize improvement to 0-0.4 range
            normalized_improvement = min(current.score_delta / 0.1, 1.0)
            progress_score += self.WEIGHT_SCORE_IMPROVEMENT * normalized_improvement

        if issues_reducing:
            # Normalize issue reduction
            if prev.total_issues > 0:
                reduction_pct = abs(current.issue_delta) / prev.total_issues
                progress_score += self.WEIGHT_ISSUE_REDUCTION * min(reduction_pct, 1.0)

        if critical_resolved and prev.critical_issues > 0:
            resolution_pct = abs(current.critical_delta) / prev.critical_issues
            progress_score += self.WEIGHT_CRITICAL_RESOLUTION * min(resolution_pct, 1.0)

        # Determine if progressing
        is_progressing = (
            score_improving or
            (issues_reducing and current.issue_delta <= -2) or
            critical_resolved
        )

        # Update stagnation counter
        if is_progressing:
            self.consecutive_stagnant = 0
        else:
            self.consecutive_stagnant += 1

        # Determine stagnation reason
        stagnation_reason = None
        if not is_progressing:
            if not visual_changed:
                stagnation_reason = StagnationReason.VISUAL_UNCHANGED
            elif score_regressed:
                stagnation_reason = StagnationReason.FIX_REGRESSION
            elif current.critical_issues > 0 and current.critical_delta >= 0:
                stagnation_reason = StagnationReason.CRITICAL_UNRESOLVED
            elif self._detect_oscillation():
                stagnation_reason = StagnationReason.OSCILLATING
            else:
                stagnation_reason = StagnationReason.SCORE_PLATEAU

        # Determine risk level
        if self.consecutive_stagnant == 0:
            stagnation_risk = StagnationRisk.NONE
        elif self.consecutive_stagnant == 1:
            stagnation_risk = StagnationRisk.LOW
        elif self.consecutive_stagnant == 2:
            stagnation_risk = StagnationRisk.MEDIUM
        elif self.consecutive_stagnant >= self.max_stagnant_iterations:
            stagnation_risk = StagnationRisk.CRITICAL
        else:
            stagnation_risk = StagnationRisk.HIGH

        # Determine action
        if stagnation_risk == StagnationRisk.CRITICAL:
            recommended_action = "terminate"
        elif stagnation_risk in (StagnationRisk.HIGH, StagnationRisk.MEDIUM):
            recommended_action = "escalate"
        else:
            recommended_action = "continue"

        return ProgressAnalysis(
            is_progressing=is_progressing,
            progress_score=progress_score,
            stagnation_risk=stagnation_risk,
            stagnation_reason=stagnation_reason,
            consecutive_stagnant_iterations=self.consecutive_stagnant,
            recommended_action=recommended_action,
            details={
                "score_improving": score_improving,
                "issues_reducing": issues_reducing,
                "critical_resolved": critical_resolved,
                "visual_changed": visual_changed,
                "score_regressed": score_regressed,
                "score_delta": current.score_delta,
                "issue_delta": current.issue_delta,
                "critical_delta": current.critical_delta,
                "best_score": self.best_score,
                "best_iteration": self.best_iteration,
            }
        )

    def _detect_oscillation(self) -> bool:
        """Detect if score is oscillating (up-down-up-down pattern)."""
        if not self.detect_oscillation or len(self.history) < 4:
            return False

        # Check last 4 iterations for alternating pattern
        recent = self.history[-4:]
        deltas = [recent[i+1].score_delta for i in range(3)]

        # Oscillating if signs alternate: +, -, + or -, +, -
        signs = [d > 0 for d in deltas]
        oscillating = (
            (signs[0] != signs[1] and signs[1] != signs[2]) or
            all(abs(d) < self.stagnation_threshold for d in deltas)
        )

        return oscillating

    def _log_iteration(self, snapshot: ProgressSnapshot, analysis: ProgressAnalysis):
        """Log iteration progress."""
        delta_str = f"+{snapshot.score_delta:.3f}" if snapshot.score_delta >= 0 else f"{snapshot.score_delta:.3f}"
        issue_str = f"{snapshot.issue_delta:+d}" if snapshot.issue_delta != 0 else "0"

        status = "PROGRESSING" if analysis.is_progressing else f"STAGNANT ({self.consecutive_stagnant})"

        logger.info(
            f"[Iteration {snapshot.iteration}] "
            f"Score: {snapshot.combined_score:.3f} ({delta_str}) | "
            f"Issues: {snapshot.total_issues} ({issue_str}) | "
            f"Critical: {snapshot.critical_issues} | "
            f"Status: {status} | "
            f"Risk: {analysis.stagnation_risk.value}"
        )

        if analysis.recommended_action == "escalate":
            logger.warning(
                f"Stagnation risk elevated: {analysis.stagnation_reason.value if analysis.stagnation_reason else 'unknown'}"
            )
        elif analysis.recommended_action == "terminate":
            logger.error(
                f"STAGNATION CRITICAL - Recommending loop termination: "
                f"{analysis.stagnation_reason.value if analysis.stagnation_reason else 'max iterations'}"
            )

    def get_history_summary(self) -> str:
        """Get formatted summary of iteration history."""
        if not self.history:
            return "No iterations recorded"

        lines = ["Iteration | Score  | Delta  | Issues | Critical | Status"]
        lines.append("-" * 60)

        for i, snap in enumerate(self.history):
            delta = f"{snap.score_delta:+.3f}" if i > 0 else "  N/A "
            status = "OK" if snap.score_delta >= self.stagnation_threshold or i == 0 else "STAGNANT"
            lines.append(
                f"    {snap.iteration:3d}   | {snap.combined_score:.3f} | {delta} | "
                f"  {snap.total_issues:3d}  |    {snap.critical_issues:3d}     | {status}"
            )

        lines.append("-" * 60)
        lines.append(f"Best Score: {self.best_score:.3f} (iteration {self.best_iteration})")
        lines.append(f"Consecutive Stagnant: {self.consecutive_stagnant}")

        return "\n".join(lines)

    def get_recommendations(self) -> List[str]:
        """Get recommendations based on current state."""
        recommendations = []

        if not self.history:
            return ["Start validation loop"]

        latest = self.history[-1]

        if latest.critical_issues > 0:
            recommendations.append(
                f"Resolve {latest.critical_issues} critical issue(s) before continuing"
            )

        if self.consecutive_stagnant >= 2:
            recommendations.append(
                "Consider different fix strategy - current approach not improving score"
            )
            recommendations.append(
                "Review Opus 4.5 analysis for missed issues"
            )

        if self.best_iteration < len(self.history) - 1:
            recommendations.append(
                f"Consider rolling back to iteration {self.best_iteration} "
                f"(score {self.best_score:.3f})"
            )

        if latest.agreed_issues < latest.total_issues * 0.5:
            recommendations.append(
                "Low agreement between validators - review disagreements manually"
            )

        return recommendations

    def should_terminate(self) -> bool:
        """Check if loop should terminate due to stagnation."""
        return self.consecutive_stagnant >= self.max_stagnant_iterations

    def raise_if_stagnant(self):
        """Raise StagnationError if stagnation limit reached."""
        if self.should_terminate():
            latest = self.history[-1] if self.history else None
            reason = StagnationReason.SCORE_PLATEAU

            # Determine most likely reason
            if latest and len(self.history) >= 2:
                prev = self.history[-2]
                if latest.image_hash == prev.image_hash:
                    reason = StagnationReason.VISUAL_UNCHANGED
                elif latest.critical_issues > 0:
                    reason = StagnationReason.CRITICAL_UNRESOLVED

            raise StagnationError(
                reason=reason,
                consecutive_stagnant=self.consecutive_stagnant,
                history_summary=self.get_history_summary(),
                recommendations=self.get_recommendations()
            )
