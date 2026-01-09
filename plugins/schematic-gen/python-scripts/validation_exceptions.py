#!/usr/bin/env python3
"""
Validation Exceptions - Exception classes that CANNOT be ignored.

All validation failures raise exceptions, not return dicts.
This ensures the pipeline stops on quality failures instead of
silently continuing with broken output.

Design Principles:
1. Exceptions cannot be ignored by callers
2. Each exception type has specific context (score, issues, etc.)
3. Verbose error messages for debugging
4. Hierarchical structure for catch-all handling

Usage:
    from validation_exceptions import (
        ValidationFailure,
        EmptyBoardFailure,
        EmptySilkscreenFailure,
        QualityThresholdFailure,
        MissingDependencyFailure
    )

    # In validation code:
    if silkscreen_is_empty:
        raise EmptySilkscreenFailure(
            "No designators found on silkscreen layer",
            image_path="/path/to/F_SilkS.png",
            expected_designators=["R1", "R2", "C1", "U1"]
        )

    # In pipeline code:
    try:
        validate_design(pcb_path)
    except ValidationFailure as e:
        print(f"VALIDATION FAILED: {e}")
        sys.exit(1)
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from enum import Enum


class ValidationSeverity(Enum):
    """Severity levels for validation failures."""
    CRITICAL = "critical"    # Design cannot be manufactured
    ERROR = "error"          # Design has serious problems
    WARNING = "warning"      # Design needs improvement
    INFO = "info"            # Minor suggestions


@dataclass
class ValidationIssue:
    """A single validation issue with context."""
    message: str
    severity: ValidationSeverity
    location: Optional[str] = None  # e.g., "F_Cu layer", "component R1"
    fix_suggestion: Optional[str] = None
    coordinates: Optional[tuple] = None  # (x, y) in mm


class ValidationFailure(Exception):
    """
    Base validation failure - stops pipeline execution.

    All validation failures inherit from this class, allowing
    catch-all handling while preserving specific error types.
    """

    def __init__(
        self,
        message: str,
        score: float = 0.0,
        issues: Optional[List[ValidationIssue]] = None,
        context: Optional[Dict[str, Any]] = None
    ):
        self.message = message
        self.score = score
        self.issues = issues or []
        self.context = context or {}

        # Build detailed error message
        error_parts = [f"VALIDATION FAILED: {message}"]

        if score > 0:
            error_parts.append(f"Score: {score:.1f}/10")

        if issues:
            error_parts.append(f"Issues ({len(issues)}):")
            for issue in issues[:5]:  # Show first 5
                error_parts.append(f"  - [{issue.severity.value.upper()}] {issue.message}")
            if len(issues) > 5:
                error_parts.append(f"  ... and {len(issues) - 5} more")

        super().__init__("\n".join(error_parts))

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "type": self.__class__.__name__,
            "message": self.message,
            "score": self.score,
            "issues": [
                {
                    "message": i.message,
                    "severity": i.severity.value,
                    "location": i.location,
                    "fix_suggestion": i.fix_suggestion
                }
                for i in self.issues
            ],
            "context": self.context
        }


class EmptyBoardFailure(ValidationFailure):
    """
    PCB has no traces, footprints, or meaningful content.

    This is a CRITICAL failure - the board cannot be manufactured
    and indicates a fundamental problem with the generation pipeline.
    """

    def __init__(
        self,
        message: str,
        image_path: Optional[str] = None,
        file_size_bytes: Optional[int] = None,
        unique_colors: Optional[int] = None,
        copper_coverage: Optional[float] = None
    ):
        context = {}
        if image_path:
            context["image_path"] = image_path
        if file_size_bytes is not None:
            context["file_size_bytes"] = file_size_bytes
        if unique_colors is not None:
            context["unique_colors"] = unique_colors
        if copper_coverage is not None:
            context["copper_coverage_percent"] = copper_coverage * 100

        issues = [
            ValidationIssue(
                message=message,
                severity=ValidationSeverity.CRITICAL,
                location=image_path,
                fix_suggestion="Regenerate PCB with actual components and traces"
            )
        ]

        super().__init__(
            message=f"Empty board detected: {message}",
            score=0.0,
            issues=issues,
            context=context
        )


class EmptySilkscreenFailure(ValidationFailure):
    """
    Silkscreen layer has no designators or meaningful text.

    This is a CRITICAL failure - the board cannot be assembled
    without component reference designators.
    """

    def __init__(
        self,
        message: str,
        image_path: Optional[str] = None,
        expected_designators: Optional[List[str]] = None,
        found_designators: Optional[List[str]] = None
    ):
        context = {
            "image_path": image_path,
            "expected_designators": expected_designators or [],
            "found_designators": found_designators or []
        }

        missing = []
        if expected_designators and found_designators is not None:
            missing = set(expected_designators) - set(found_designators)
            context["missing_designators"] = list(missing)

        issues = [
            ValidationIssue(
                message=f"Silkscreen empty or missing designators: {message}",
                severity=ValidationSeverity.CRITICAL,
                location="F.SilkS layer",
                fix_suggestion="Add reference designators for all components"
            )
        ]

        if missing:
            for ref in list(missing)[:10]:
                issues.append(ValidationIssue(
                    message=f"Missing designator: {ref}",
                    severity=ValidationSeverity.ERROR,
                    location=f"Component {ref}"
                ))

        super().__init__(
            message=f"Empty silkscreen: {message}",
            score=0.0,
            issues=issues,
            context=context
        )


class RoutingFailure(ValidationFailure):
    """
    Routing quality is below acceptable threshold.

    This includes:
    - 90° angles (should be 45°)
    - Trace overlaps on same layer
    - Missing connections
    - Inadequate trace widths
    """

    def __init__(
        self,
        message: str,
        score: float,
        ninety_degree_count: int = 0,
        overlap_count: int = 0,
        missing_connections: Optional[List[str]] = None,
        issues: Optional[List[ValidationIssue]] = None
    ):
        context = {
            "ninety_degree_angles": ninety_degree_count,
            "trace_overlaps": overlap_count,
            "missing_connections": missing_connections or []
        }

        all_issues = issues or []

        if ninety_degree_count > 0:
            all_issues.append(ValidationIssue(
                message=f"Found {ninety_degree_count} 90° angles (should use 45°)",
                severity=ValidationSeverity.ERROR,
                fix_suggestion="Reroute traces using 45° angles for better signal integrity"
            ))

        if overlap_count > 0:
            all_issues.append(ValidationIssue(
                message=f"Found {overlap_count} trace overlaps on same layer",
                severity=ValidationSeverity.CRITICAL,
                fix_suggestion="Separate overlapping traces or use vias to change layers"
            ))

        super().__init__(
            message=f"Routing failure: {message}",
            score=score,
            issues=all_issues,
            context=context
        )


class QualityThresholdFailure(ValidationFailure):
    """
    Overall quality score is below the required threshold.

    This is raised when the design doesn't meet professional
    quality standards after maximum iteration attempts.
    """

    def __init__(
        self,
        message: str,
        score: float,
        threshold: float,
        iterations_attempted: int = 0,
        max_iterations: int = 100,
        issues: Optional[List[ValidationIssue]] = None
    ):
        context = {
            "achieved_score": score,
            "required_threshold": threshold,
            "iterations_attempted": iterations_attempted,
            "max_iterations": max_iterations,
            "score_gap": threshold - score
        }

        super().__init__(
            message=f"Quality threshold not met: {message} (Score: {score:.1f}/{threshold:.1f} after {iterations_attempted} iterations)",
            score=score,
            issues=issues,
            context=context
        )


class MissingDependencyFailure(ValidationFailure):
    """
    Required dependency is not available.

    This is raised when:
    - KiCad CLI is not installed
    - ANTHROPIC_API_KEY is not set
    - SKiDL libraries are not configured
    - pcbnew Python module is not available
    """

    def __init__(
        self,
        message: str,
        dependency_name: str,
        install_instructions: Optional[str] = None
    ):
        context = {
            "dependency": dependency_name,
            "install_instructions": install_instructions
        }

        issues = [
            ValidationIssue(
                message=f"Missing dependency: {dependency_name}",
                severity=ValidationSeverity.CRITICAL,
                fix_suggestion=install_instructions or f"Install {dependency_name}"
            )
        ]

        super().__init__(
            message=f"Missing dependency: {message}",
            score=0.0,
            issues=issues,
            context=context
        )


class ComponentPlacementFailure(ValidationFailure):
    """
    Component placement violates design rules.

    This includes:
    - Components too close together
    - Components outside board boundary
    - Overlapping footprints
    - Poor thermal placement
    """

    def __init__(
        self,
        message: str,
        score: float,
        overlapping_components: Optional[List[tuple]] = None,
        clearance_violations: Optional[List[Dict]] = None,
        issues: Optional[List[ValidationIssue]] = None
    ):
        context = {
            "overlapping_components": overlapping_components or [],
            "clearance_violations": clearance_violations or []
        }

        all_issues = issues or []

        if overlapping_components:
            for comp1, comp2 in overlapping_components[:5]:
                all_issues.append(ValidationIssue(
                    message=f"Components overlap: {comp1} and {comp2}",
                    severity=ValidationSeverity.CRITICAL,
                    fix_suggestion=f"Move {comp1} or {comp2} to eliminate overlap"
                ))

        super().__init__(
            message=f"Component placement failure: {message}",
            score=score,
            issues=all_issues,
            context=context
        )


class DRCFailure(ValidationFailure):
    """
    Design Rule Check (DRC) violations found.

    This includes KiCad's built-in DRC errors:
    - Clearance violations
    - Unconnected pins
    - Track width violations
    - Via size violations
    """

    def __init__(
        self,
        message: str,
        error_count: int,
        warning_count: int,
        drc_errors: Optional[List[Dict]] = None
    ):
        context = {
            "drc_error_count": error_count,
            "drc_warning_count": warning_count,
            "drc_errors": drc_errors or []
        }

        issues = []
        for err in (drc_errors or [])[:10]:
            issues.append(ValidationIssue(
                message=err.get("message", "DRC error"),
                severity=ValidationSeverity.ERROR if err.get("type") == "error" else ValidationSeverity.WARNING,
                location=err.get("location"),
                coordinates=err.get("coordinates")
            ))

        super().__init__(
            message=f"DRC failure: {message} ({error_count} errors, {warning_count} warnings)",
            score=0.0 if error_count > 0 else 5.0,
            issues=issues,
            context=context
        )


# Convenience function for raising with context
def fail_validation(
    failure_type: type,
    message: str,
    **kwargs
) -> None:
    """
    Raise a validation failure with the given type and context.

    Usage:
        fail_validation(EmptyBoardFailure, "No traces found", image_path="F_Cu.png")
    """
    raise failure_type(message, **kwargs)


# Export all exception classes
__all__ = [
    'ValidationSeverity',
    'ValidationIssue',
    'ValidationFailure',
    'EmptyBoardFailure',
    'EmptySilkscreenFailure',
    'RoutingFailure',
    'QualityThresholdFailure',
    'MissingDependencyFailure',
    'ComponentPlacementFailure',
    'DRCFailure',
    'fail_validation'
]
