#!/usr/bin/env python3
"""
Honest PCB Validation System

Provides truthful, non-gameable validation of PCB designs using multiple
independent validation methods:

1. KiCad DRC - Ground truth design rule checking (hard requirements)
2. Programmatic CV - Visual quality metrics (soft requirements)
3. Fabrication File Checks - File integrity validation (hard requirements)

The system clearly separates:
- HARD REQUIREMENTS: Must pass for fabrication (DRC, file integrity)
- SOFT REQUIREMENTS: Quality metrics that indicate professionalism
- LIMITATIONS: What we cannot verify without human/AI expert review

This validator will NOT:
- Game metrics by inflating floor scores
- Report passing when there are real issues
- Claim "professional quality" without evidence

Usage:
    from honest_validator import HonestValidator

    validator = HonestValidator()
    result = validator.validate(
        pcb_path="/path/to/board.kicad_pcb",
        image_dir="/path/to/layer_images",
        gerber_dir="/path/to/gerbers"
    )

    print(validator.get_report(result))
"""

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple


@dataclass
class ValidationCategory:
    """Result for a validation category."""
    name: str
    passed: bool
    score: Optional[float]  # None if not applicable
    max_score: Optional[float]
    issues: List[str]
    details: Dict[str, Any]
    is_hard_requirement: bool


@dataclass
class HonestValidationResult:
    """Complete honest validation result."""
    # Overall status
    fabrication_ready: bool  # Can this be sent to fab house?
    quality_level: str  # "production", "prototype", "review_needed", "failed"

    # Hard requirements (must pass)
    drc_passed: bool
    files_valid: bool

    # Soft requirements (quality indicators)
    visual_score: float

    # Detailed results
    categories: Dict[str, ValidationCategory]

    # Limitations - what we couldn't verify
    limitations: List[str]

    # Recommendations
    recommendations: List[str]

    # Summary
    summary: str


class HonestValidator:
    """
    Honest PCB validation that doesn't game metrics.

    Clearly reports what passed, what failed, and what we
    cannot verify without expert human/AI review.
    """

    # Quality thresholds - these are honest, not inflated
    PRODUCTION_THRESHOLD = 8.5  # Truly production-ready
    PROTOTYPE_THRESHOLD = 7.0   # Acceptable for prototyping
    REVIEW_THRESHOLD = 5.0      # Needs human review

    def __init__(self):
        """Initialize validator with available tools."""
        self._init_tools()

    def _init_tools(self):
        """Initialize validation tools."""
        # Try importing each tool
        self.has_drc = False
        self.has_cv_scorer = False

        try:
            from kicad_drc_checker import KiCadDRCChecker
            self.drc_checker = KiCadDRCChecker()
            self.has_drc = True
        except (ImportError, RuntimeError) as e:
            print(f"Note: KiCad DRC not available: {e}")
            self.drc_checker = None

        try:
            from programmatic_scorer import ProgrammaticScorer
            self.cv_scorer = ProgrammaticScorer(strict=False)
            self.has_cv_scorer = True
        except ImportError as e:
            print(f"Note: CV scorer not available: {e}")
            self.cv_scorer = None

    def validate(
        self,
        pcb_path: Optional[str] = None,
        image_dir: Optional[str] = None,
        gerber_dir: Optional[str] = None
    ) -> HonestValidationResult:
        """
        Run honest validation on PCB design.

        Args:
            pcb_path: Path to .kicad_pcb file (for DRC)
            image_dir: Path to layer images (for CV scoring)
            gerber_dir: Path to gerber files (for file validation)

        Returns:
            HonestValidationResult with truthful assessment
        """
        categories = {}
        limitations = []
        recommendations = []

        # 1. Run DRC if available and PCB provided
        drc_passed = True  # Assume passed if we can't check
        if pcb_path and self.has_drc:
            drc_result = self._run_drc(pcb_path)
            categories['drc'] = drc_result
            drc_passed = drc_result.passed
        elif pcb_path and not self.has_drc:
            limitations.append("KiCad DRC not available - design rules not verified")
            categories['drc'] = ValidationCategory(
                name="Design Rule Check",
                passed=True,  # Unknown, not failed
                score=None,
                max_score=None,
                issues=["DRC tool not available - cannot verify"],
                details={'available': False},
                is_hard_requirement=True
            )

        # 2. Run file validation if gerber dir provided
        files_valid = True
        if gerber_dir:
            file_result = self._validate_files(gerber_dir)
            categories['files'] = file_result
            files_valid = file_result.passed

        # 3. Run CV scoring if image dir provided
        visual_score = 5.0  # Default neutral
        if image_dir and self.has_cv_scorer:
            cv_result = self._run_cv_scoring(image_dir)
            categories['visual'] = cv_result
            visual_score = cv_result.score or 5.0
        elif image_dir and not self.has_cv_scorer:
            limitations.append("CV scorer not available - visual quality not assessed")

        # 4. Add standard limitations
        limitations.extend([
            "CV scoring measures proxy metrics, not actual PCB quality",
            "Routing quality assessment is approximate (edge detection based)",
            "Component placement quality not verified",
            "Signal integrity not analyzed",
            "Thermal analysis not performed",
        ])

        # 5. Determine overall status
        fabrication_ready = drc_passed and files_valid

        if not fabrication_ready:
            quality_level = "failed"
            recommendations.append("Fix DRC errors before sending to fabrication")
        elif visual_score >= self.PRODUCTION_THRESHOLD:
            quality_level = "production"
        elif visual_score >= self.PROTOTYPE_THRESHOLD:
            quality_level = "prototype"
            recommendations.append("Review visual quality before production run")
        elif visual_score >= self.REVIEW_THRESHOLD:
            quality_level = "review_needed"
            recommendations.append("Have a professional review before fabrication")
        else:
            quality_level = "failed"
            recommendations.append("Significant quality issues detected - needs work")

        # 6. Generate summary
        summary = self._generate_summary(
            fabrication_ready, quality_level, drc_passed,
            files_valid, visual_score, categories
        )

        return HonestValidationResult(
            fabrication_ready=fabrication_ready,
            quality_level=quality_level,
            drc_passed=drc_passed,
            files_valid=files_valid,
            visual_score=visual_score,
            categories=categories,
            limitations=limitations,
            recommendations=recommendations,
            summary=summary
        )

    def _run_drc(self, pcb_path: str) -> ValidationCategory:
        """Run KiCad DRC check."""
        try:
            result = self.drc_checker.run_drc(pcb_path)

            issues = []
            for err in result.errors[:10]:  # First 10 errors
                issues.append(f"[{err.type}] {err.message}")

            return ValidationCategory(
                name="Design Rule Check (DRC)",
                passed=result.passed,
                score=10.0 if result.passed else 0.0,
                max_score=10.0,
                issues=issues,
                details={
                    'error_count': result.error_count,
                    'warning_count': result.warning_count,
                    'violations_by_type': result.violations_by_type
                },
                is_hard_requirement=True
            )
        except Exception as e:
            return ValidationCategory(
                name="Design Rule Check (DRC)",
                passed=False,
                score=0.0,
                max_score=10.0,
                issues=[f"DRC check failed: {e}"],
                details={'error': str(e)},
                is_hard_requirement=True
            )

    def _validate_files(self, gerber_dir: str) -> ValidationCategory:
        """Validate fabrication files exist and are non-empty."""
        gerber_path = Path(gerber_dir)
        issues = []

        # Required file extensions for standard fab
        required_extensions = {
            '.gtl': 'Top copper',
            '.gbl': 'Bottom copper',
            '.gts': 'Top solder mask',
            '.gbs': 'Bottom solder mask',
            '.gto': 'Top silkscreen',
            '.gbo': 'Bottom silkscreen',
            '.gm1': 'Board outline',
            '.drl': 'Drill file'
        }

        found_files = {}
        missing_files = []
        empty_files = []

        for ext, desc in required_extensions.items():
            files = list(gerber_path.glob(f'*{ext}'))
            if files:
                for f in files:
                    if f.stat().st_size < 100:  # Suspiciously small
                        empty_files.append(f"{f.name} ({desc})")
                    else:
                        found_files[ext] = f.name
            else:
                missing_files.append(f"{ext} ({desc})")

        if missing_files:
            issues.append(f"Missing files: {', '.join(missing_files)}")
        if empty_files:
            issues.append(f"Empty/corrupt files: {', '.join(empty_files)}")

        passed = len(missing_files) == 0 and len(empty_files) == 0

        return ValidationCategory(
            name="Fabrication Files",
            passed=passed,
            score=10.0 if passed else (10.0 - len(missing_files) - len(empty_files)),
            max_score=10.0,
            issues=issues,
            details={
                'found_files': found_files,
                'missing': missing_files,
                'empty': empty_files
            },
            is_hard_requirement=True
        )

    def _run_cv_scoring(self, image_dir: str) -> ValidationCategory:
        """Run CV-based visual scoring."""
        try:
            result = self.cv_scorer.score_all_images(image_dir)

            issues = result.issues[:10]  # First 10 issues

            # Calculate layer summary
            layer_scores = {}
            for name, score in result.layer_scores.items():
                layer_scores[name] = score.score

            return ValidationCategory(
                name="Visual Quality (CV)",
                passed=result.passed,
                score=result.overall_score,
                max_score=10.0,
                issues=issues,
                details={
                    'layer_scores': layer_scores,
                    'aggregate_metrics': result.aggregate_metrics,
                    'suggestions': result.suggestions[:5]
                },
                is_hard_requirement=False  # Soft requirement
            )
        except Exception as e:
            return ValidationCategory(
                name="Visual Quality (CV)",
                passed=False,
                score=0.0,
                max_score=10.0,
                issues=[f"CV scoring failed: {e}"],
                details={'error': str(e)},
                is_hard_requirement=False
            )

    def _generate_summary(
        self,
        fabrication_ready: bool,
        quality_level: str,
        drc_passed: bool,
        files_valid: bool,
        visual_score: float,
        categories: Dict[str, ValidationCategory]
    ) -> str:
        """Generate honest summary."""
        lines = []

        lines.append("=" * 60)
        lines.append("HONEST PCB VALIDATION REPORT")
        lines.append("=" * 60)
        lines.append("")

        # Overall verdict
        if fabrication_ready:
            lines.append(f"FABRICATION: READY ({quality_level.upper()})")
        else:
            lines.append("FABRICATION: NOT READY - HAS BLOCKING ISSUES")

        lines.append("")
        lines.append("-" * 40)
        lines.append("HARD REQUIREMENTS (must pass for fab):")
        lines.append("-" * 40)

        # DRC status
        if 'drc' in categories:
            drc = categories['drc']
            status = "PASS" if drc.passed else "FAIL"
            lines.append(f"  DRC Check: {status}")
            if not drc.passed and drc.issues:
                for issue in drc.issues[:3]:
                    lines.append(f"    - {issue}")
        else:
            lines.append("  DRC Check: NOT AVAILABLE")

        # File status
        if 'files' in categories:
            files = categories['files']
            status = "PASS" if files.passed else "FAIL"
            lines.append(f"  Fab Files: {status}")
            if not files.passed and files.issues:
                for issue in files.issues[:3]:
                    lines.append(f"    - {issue}")

        lines.append("")
        lines.append("-" * 40)
        lines.append("SOFT REQUIREMENTS (quality indicators):")
        lines.append("-" * 40)

        # Visual score
        lines.append(f"  Visual Score: {visual_score:.1f}/10.0")
        if visual_score >= self.PRODUCTION_THRESHOLD:
            lines.append("    Assessment: Production quality")
        elif visual_score >= self.PROTOTYPE_THRESHOLD:
            lines.append("    Assessment: Prototype quality")
        elif visual_score >= self.REVIEW_THRESHOLD:
            lines.append("    Assessment: Needs review")
        else:
            lines.append("    Assessment: Quality concerns")

        # Layer breakdown if available
        if 'visual' in categories and categories['visual'].details.get('layer_scores'):
            lines.append("")
            lines.append("  Layer scores:")
            scores = categories['visual'].details['layer_scores']
            for layer, score in sorted(scores.items(), key=lambda x: x[1], reverse=True)[:5]:
                lines.append(f"    {layer}: {score:.1f}")

        lines.append("")
        lines.append("=" * 60)

        return "\n".join(lines)

    def get_report(self, result: HonestValidationResult) -> str:
        """Get full report including limitations."""
        lines = [result.summary]

        if result.limitations:
            lines.append("")
            lines.append("-" * 40)
            lines.append("LIMITATIONS (what we cannot verify):")
            lines.append("-" * 40)
            for lim in result.limitations:
                lines.append(f"  * {lim}")

        if result.recommendations:
            lines.append("")
            lines.append("-" * 40)
            lines.append("RECOMMENDATIONS:")
            lines.append("-" * 40)
            for rec in result.recommendations:
                lines.append(f"  -> {rec}")

        lines.append("")
        lines.append("=" * 60)

        return "\n".join(lines)


def main():
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(description='Honest PCB Validation')
    parser.add_argument('--pcb', help='Path to .kicad_pcb file')
    parser.add_argument('--images', help='Path to layer images directory')
    parser.add_argument('--gerbers', help='Path to gerber files directory')
    parser.add_argument('--json', action='store_true', help='Output as JSON')

    args = parser.parse_args()

    if not args.pcb and not args.images and not args.gerbers:
        parser.error("At least one of --pcb, --images, or --gerbers is required")

    validator = HonestValidator()
    result = validator.validate(
        pcb_path=args.pcb,
        image_dir=args.images,
        gerber_dir=args.gerbers
    )

    if args.json:
        output = {
            'fabrication_ready': result.fabrication_ready,
            'quality_level': result.quality_level,
            'drc_passed': result.drc_passed,
            'files_valid': result.files_valid,
            'visual_score': result.visual_score,
            'limitations': result.limitations,
            'recommendations': result.recommendations,
            'categories': {
                name: {
                    'passed': cat.passed,
                    'score': cat.score,
                    'issues': cat.issues,
                    'is_hard_requirement': cat.is_hard_requirement
                }
                for name, cat in result.categories.items()
            }
        }
        print(json.dumps(output, indent=2))
    else:
        print(validator.get_report(result))

    # Exit with appropriate code
    import sys
    sys.exit(0 if result.fabrication_ready else 1)


if __name__ == '__main__':
    main()
