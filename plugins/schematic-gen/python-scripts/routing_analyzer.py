#!/usr/bin/env python3
"""
Routing Analyzer - PCB Trace Routing Quality Analysis

Analyzes PCB copper layer images for routing quality using Claude Opus 4.5
via OpenRouter API (not direct Anthropic SDK):
- 45° vs 90° angle detection
- Trace overlap detection
- Routing cleanliness assessment
- Signal integrity concerns
- Professional appearance scoring

Environment Variables:
    OPENROUTER_API_KEY: Required - Your OpenRouter API key

Usage:
    python routing_analyzer.py --image F_Cu.png
    python routing_analyzer.py --image F_Cu.png --strict
    python routing_analyzer.py --dir ./layer_images --pattern "*Cu*.png"
"""

import argparse
import base64
import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

from validation_exceptions import (
    ValidationFailure,
    RoutingFailure,
    MissingDependencyFailure,
    ValidationIssue,
    ValidationSeverity
)

# Use OpenRouter client instead of direct Anthropic SDK
try:
    from openrouter_client import OpenRouterClient, OpenRouterAnthropicAdapter
    HAS_OPENROUTER = True
except ImportError:
    HAS_OPENROUTER = False


@dataclass
class RoutingIssue:
    """A specific routing issue found."""
    issue_type: str  # 'angle', 'overlap', 'clearance', 'width', 'aesthetics'
    severity: str  # 'critical', 'warning', 'info'
    description: str
    location: str  # Approximate location in image
    fix_suggestion: str


@dataclass
class RoutingAnalysisResult:
    """Complete routing analysis result."""
    overall_score: float
    passed: bool
    angle_score: float
    overlap_score: float
    cleanliness_score: float
    consistency_score: float
    signal_integrity_score: float

    # Counts
    ninety_degree_angles: int
    trace_overlaps: int
    clearance_violations: int

    # Issues
    critical_issues: List[RoutingIssue]
    warnings: List[RoutingIssue]
    recommendations: List[str]

    # Assessment
    professional_tier: str  # 'award-winning', 'professional', 'acceptable', 'below-average', 'amateur'
    summary: str


class RoutingAnalyzer:
    """
    Analyzes PCB routing quality using AI vision.

    Raises exceptions on critical routing failures - no silent passes.
    """

    def __init__(self, api_key: Optional[str] = None, strict: bool = False):
        """
        Initialize routing analyzer.

        Args:
            api_key: OpenRouter API key (defaults to OPENROUTER_API_KEY env var)
            strict: If True, raise exceptions on routing failures

        Raises:
            MissingDependencyFailure: If dependencies are missing
        """
        if not HAS_OPENROUTER:
            raise MissingDependencyFailure(
                "openrouter_client module required for routing analysis",
                dependency_name="openrouter_client",
                install_instructions="Ensure openrouter_client.py is in the same directory"
            )

        self.api_key = api_key or os.environ.get('OPENROUTER_API_KEY')
        if not self.api_key:
            raise MissingDependencyFailure(
                "OPENROUTER_API_KEY not set",
                dependency_name="OPENROUTER_API_KEY",
                install_instructions=(
                    "1. Get API key from https://openrouter.ai/keys\n"
                    "2. export OPENROUTER_API_KEY=your-key\n"
                    "3. Re-run this script"
                )
            )

        # Use OpenRouter with Anthropic-compatible interface
        self.client = OpenRouterAnthropicAdapter(api_key=self.api_key)
        self.strict = strict  # Stricter evaluation for production boards

    def _load_image(self, image_path: str) -> Tuple[str, str]:
        """Load image and return (base64_data, media_type)."""
        with open(image_path, 'rb') as f:
            data = base64.standard_b64encode(f.read()).decode('utf-8')

        ext = Path(image_path).suffix.lower()
        media_types = {
            '.png': 'image/png',
            '.jpg': 'image/jpeg',
            '.jpeg': 'image/jpeg'
        }
        return data, media_types.get(ext, 'image/png')

    def analyze_routing(self, image_path: str) -> RoutingAnalysisResult:
        """
        Analyze PCB copper layer routing quality.

        Args:
            image_path: Path to copper layer image

        Returns:
            RoutingAnalysisResult with detailed analysis
        """
        image_data, media_type = self._load_image(image_path)

        strictness_note = ""
        if self.strict:
            strictness_note = """
STRICT MODE ENABLED - This is a production board evaluation.
- ANY 90° angle is a failure
- ANY trace overlap is critical
- Professional standards apply throughout"""

        system_prompt = f"""You are an expert PCB routing quality analyst.
You specialize in evaluating trace routing for professional quality.

KEY EVALUATION CRITERIA:

1. ROUTING ANGLES (Critical)
   - Professional PCBs use 45° angles exclusively
   - 90° angles cause signal reflections and EMI
   - Count ALL instances of 90° angles
   - Even ONE 90° angle is a concern for high-speed designs

2. TRACE OVERLAPS (Critical)
   - Traces on the SAME layer must NEVER cross
   - Any overlap indicates a design rule violation
   - This is a manufacturing defect waiting to happen

3. ROUTING CLEANLINESS (Important)
   - Traces should follow logical paths
   - Avoid "spaghetti" routing (random, tangled traces)
   - Similar signals should route together
   - Ground returns should be obvious

4. TRACE WIDTH CONSISTENCY (Important)
   - Power traces should be wider than signals
   - Similar nets should have similar widths
   - Width changes should be gradual, not abrupt

5. SIGNAL INTEGRITY (Important)
   - Differential pairs should be matched length
   - High-speed signals should have clear return paths
   - Avoid stubs and antenna effects
{strictness_note}"""

        user_prompt = """Analyze this PCB copper layer image for routing quality.

Provide comprehensive JSON analysis:
{
    "overall_score": 1-10 (1=amateur, 10=award-winning),
    "passed": true/false (true if score >= 7 and no critical issues),

    "scores": {
        "angle_quality": 1-10 (10=all 45°, 1=all 90°),
        "overlap_free": 1-10 (10=no overlaps, 1=many overlaps),
        "cleanliness": 1-10 (10=pristine, 1=spaghetti),
        "width_consistency": 1-10 (10=consistent, 1=random),
        "signal_integrity": 1-10 (10=excellent, 1=poor)
    },

    "counts": {
        "ninety_degree_angles": number (estimate),
        "trace_overlaps": number,
        "clearance_violations": number,
        "spaghetti_regions": number
    },

    "critical_issues": [
        {
            "type": "angle|overlap|clearance|width|other",
            "description": "specific description",
            "location": "where in the image",
            "fix": "how to fix"
        }
    ],

    "warnings": [
        {
            "type": "category",
            "description": "description",
            "location": "where",
            "fix": "suggestion"
        }
    ],

    "recommendations": ["list of improvement suggestions"],

    "professional_tier": "award-winning|professional|acceptable|below-average|amateur",

    "summary": "One paragraph overall assessment"
}

Be thorough and specific. Count visible issues where possible.
An amateur design MUST be clearly identified as such."""

        response = self.client.messages.create(
            model="claude-opus-4-5-20251101",
            max_tokens=4096,
            system=system_prompt,
            messages=[{
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": media_type,
                            "data": image_data
                        }
                    },
                    {
                        "type": "text",
                        "text": user_prompt
                    }
                ]
            }]
        )

        response_text = response.content[0].text

        # Parse JSON response
        try:
            json_start = response_text.find('{')
            json_end = response_text.rfind('}') + 1
            if json_start >= 0 and json_end > json_start:
                result = json.loads(response_text[json_start:json_end])
            else:
                result = {}
        except json.JSONDecodeError:
            result = {}

        # Build RoutingIssue objects
        critical_issues = []
        for issue in result.get('critical_issues', []):
            critical_issues.append(RoutingIssue(
                issue_type=issue.get('type', 'unknown'),
                severity='critical',
                description=issue.get('description', ''),
                location=issue.get('location', 'unknown'),
                fix_suggestion=issue.get('fix', '')
            ))

        warnings = []
        for issue in result.get('warnings', []):
            warnings.append(RoutingIssue(
                issue_type=issue.get('type', 'unknown'),
                severity='warning',
                description=issue.get('description', ''),
                location=issue.get('location', 'unknown'),
                fix_suggestion=issue.get('fix', '')
            ))

        scores = result.get('scores', {})
        counts = result.get('counts', {})

        analysis_result = RoutingAnalysisResult(
            overall_score=result.get('overall_score', 0),
            passed=result.get('passed', False),
            angle_score=scores.get('angle_quality', 0),
            overlap_score=scores.get('overlap_free', 0),
            cleanliness_score=scores.get('cleanliness', 0),
            consistency_score=scores.get('width_consistency', 0),
            signal_integrity_score=scores.get('signal_integrity', 0),
            ninety_degree_angles=counts.get('ninety_degree_angles', 0),
            trace_overlaps=counts.get('trace_overlaps', 0),
            clearance_violations=counts.get('clearance_violations', 0),
            critical_issues=critical_issues,
            warnings=warnings,
            recommendations=result.get('recommendations', []),
            professional_tier=result.get('professional_tier', 'unknown'),
            summary=result.get('summary', '')
        )

        # STRICT MODE: Raise exception on critical routing failures
        if self.strict:
            # Trace overlaps are CRITICAL - short circuit
            if analysis_result.trace_overlaps > 0:
                issues = [
                    ValidationIssue(
                        message=f"Found {analysis_result.trace_overlaps} trace overlaps on same layer",
                        severity=ValidationSeverity.CRITICAL,
                        location=image_path,
                        fix_suggestion="Separate overlapping traces or use vias to change layers"
                    )
                ]
                raise RoutingFailure(
                    message=f"Trace overlaps detected in {Path(image_path).name}",
                    score=analysis_result.overall_score,
                    overlap_count=analysis_result.trace_overlaps,
                    issues=issues
                )

            # Excessive 90° angles in strict mode
            if analysis_result.ninety_degree_angles > 5:
                issues = [
                    ValidationIssue(
                        message=f"Found {analysis_result.ninety_degree_angles} 90° angles (should use 45°)",
                        severity=ValidationSeverity.ERROR,
                        location=image_path,
                        fix_suggestion="Reroute traces using 45° angles for better signal integrity"
                    )
                ]
                raise RoutingFailure(
                    message=f"Excessive 90° angles in {Path(image_path).name}",
                    score=analysis_result.overall_score,
                    ninety_degree_count=analysis_result.ninety_degree_angles,
                    issues=issues
                )

            # Overall score too low
            if analysis_result.overall_score < 5.0:
                raise RoutingFailure(
                    message=f"Routing quality too low ({analysis_result.overall_score}/10) in {Path(image_path).name}",
                    score=analysis_result.overall_score,
                    ninety_degree_count=analysis_result.ninety_degree_angles,
                    overlap_count=analysis_result.trace_overlaps
                )

        return analysis_result

    def analyze_multiple_layers(self, image_paths: List[str]) -> Dict[str, Any]:
        """Analyze multiple copper layer images."""
        results = {}
        all_scores = []
        all_critical = []

        for path in image_paths:
            print(f"  Analyzing {Path(path).name}...")
            result = self.analyze_routing(path)
            results[Path(path).name] = result
            all_scores.append(result.overall_score)
            all_critical.extend(result.critical_issues)

        avg_score = sum(all_scores) / len(all_scores) if all_scores else 0
        all_passed = all(r.passed for r in results.values())

        return {
            "layers_analyzed": len(results),
            "average_score": round(avg_score, 2),
            "all_passed": all_passed,
            "total_critical_issues": len(all_critical),
            "overall_verdict": "PASS" if all_passed and avg_score >= 7 else "FAIL",
            "layer_results": {
                name: {
                    "score": r.overall_score,
                    "passed": r.passed,
                    "tier": r.professional_tier,
                    "critical_count": len(r.critical_issues)
                }
                for name, r in results.items()
            }
        }

    def generate_report(self, result: RoutingAnalysisResult) -> str:
        """Generate a formatted text report."""
        lines = [
            "=" * 70,
            "PCB ROUTING QUALITY ANALYSIS REPORT",
            "=" * 70,
            "",
            f"Overall Score: {result.overall_score}/10",
            f"Status: {'PASS' if result.passed else 'FAIL'}",
            f"Professional Tier: {result.professional_tier.upper()}",
            "",
            "-" * 50,
            "DETAILED SCORES",
            "-" * 50,
            f"  Angle Quality (45° usage):     {result.angle_score}/10",
            f"  Overlap Free:                  {result.overlap_score}/10",
            f"  Routing Cleanliness:           {result.cleanliness_score}/10",
            f"  Width Consistency:             {result.consistency_score}/10",
            f"  Signal Integrity:              {result.signal_integrity_score}/10",
            "",
            "-" * 50,
            "ISSUE COUNTS",
            "-" * 50,
            f"  90° Angles Found:              {result.ninety_degree_angles}",
            f"  Trace Overlaps:                {result.trace_overlaps}",
            f"  Clearance Violations:          {result.clearance_violations}",
            ""
        ]

        if result.critical_issues:
            lines.extend([
                "-" * 50,
                f"CRITICAL ISSUES ({len(result.critical_issues)}) - MUST FIX",
                "-" * 50
            ])
            for issue in result.critical_issues:
                lines.extend([
                    f"  [{issue.issue_type.upper()}] {issue.description}",
                    f"    Location: {issue.location}",
                    f"    Fix: {issue.fix_suggestion}",
                    ""
                ])

        if result.warnings:
            lines.extend([
                "-" * 50,
                f"WARNINGS ({len(result.warnings)})",
                "-" * 50
            ])
            for issue in result.warnings:
                lines.extend([
                    f"  [{issue.issue_type.upper()}] {issue.description}",
                    f"    Location: {issue.location}",
                    ""
                ])

        if result.recommendations:
            lines.extend([
                "-" * 50,
                "RECOMMENDATIONS",
                "-" * 50
            ])
            for rec in result.recommendations:
                lines.append(f"  - {rec}")
            lines.append("")

        lines.extend([
            "-" * 50,
            "SUMMARY",
            "-" * 50,
            result.summary,
            ""
        ])

        return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description='Routing Analyzer - PCB Trace Routing Quality Analysis'
    )
    parser.add_argument(
        '--image', '-i',
        type=str,
        help='Copper layer image to analyze'
    )
    parser.add_argument(
        '--images',
        type=str,
        nargs='+',
        help='Multiple images to analyze'
    )
    parser.add_argument(
        '--dir', '-d',
        type=str,
        help='Directory of images'
    )
    parser.add_argument(
        '--pattern',
        type=str,
        default='*Cu*.png',
        help='File pattern for directory (default: *Cu*.png)'
    )
    parser.add_argument(
        '--strict',
        action='store_true',
        help='Enable strict mode for production boards'
    )
    parser.add_argument(
        '--json',
        action='store_true',
        help='Output as JSON'
    )

    args = parser.parse_args()

    if not args.image and not args.images and not args.dir:
        parser.error("Specify --image, --images, or --dir")

    if not HAS_OPENROUTER:
        print("ERROR: openrouter_client module not found")
        sys.exit(1)

    if not os.environ.get('OPENROUTER_API_KEY'):
        print("ERROR: OPENROUTER_API_KEY not set")
        print("Get your API key from https://openrouter.ai/keys")
        sys.exit(1)

    analyzer = RoutingAnalyzer(strict=args.strict)

    if args.dir:
        dir_path = Path(args.dir)
        images = list(dir_path.glob(args.pattern))
        if not images:
            print(f"No images matching {args.pattern} in {args.dir}")
            sys.exit(1)

        result = analyzer.analyze_multiple_layers([str(img) for img in images])

        if args.json:
            print(json.dumps(result, indent=2, default=str))
        else:
            print(f"\n{'='*60}")
            print("MULTI-LAYER ROUTING ANALYSIS")
            print(f"{'='*60}")
            print(f"Layers Analyzed: {result['layers_analyzed']}")
            print(f"Average Score: {result['average_score']}/10")
            print(f"Overall: {result['overall_verdict']}")
            print(f"Total Critical Issues: {result['total_critical_issues']}")
            print("\nPer-Layer Results:")
            for name, data in result['layer_results'].items():
                print(f"  {name}: {data['score']}/10 ({data['tier']}) - {'PASS' if data['passed'] else 'FAIL'}")

    elif args.images:
        result = analyzer.analyze_multiple_layers(args.images)
        if args.json:
            print(json.dumps(result, indent=2, default=str))
        else:
            print(f"Average Score: {result['average_score']}/10")
            print(f"Overall: {result['overall_verdict']}")

    else:
        result = analyzer.analyze_routing(args.image)

        if args.json:
            output = {
                "overall_score": result.overall_score,
                "passed": result.passed,
                "professional_tier": result.professional_tier,
                "scores": {
                    "angle_quality": result.angle_score,
                    "overlap_free": result.overlap_score,
                    "cleanliness": result.cleanliness_score,
                    "width_consistency": result.consistency_score,
                    "signal_integrity": result.signal_integrity_score
                },
                "counts": {
                    "ninety_degree_angles": result.ninety_degree_angles,
                    "trace_overlaps": result.trace_overlaps,
                    "clearance_violations": result.clearance_violations
                },
                "critical_issues": [
                    {
                        "type": i.issue_type,
                        "description": i.description,
                        "location": i.location,
                        "fix": i.fix_suggestion
                    }
                    for i in result.critical_issues
                ],
                "warnings": [
                    {
                        "type": i.issue_type,
                        "description": i.description,
                        "location": i.location
                    }
                    for i in result.warnings
                ],
                "recommendations": result.recommendations,
                "summary": result.summary
            }
            print(json.dumps(output, indent=2))
        else:
            print(analyzer.generate_report(result))


if __name__ == '__main__':
    main()
