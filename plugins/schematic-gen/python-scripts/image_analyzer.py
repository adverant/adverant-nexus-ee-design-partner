#!/usr/bin/env python3
"""
Image Analyzer for PCB Visual Defect Detection

Analyzes PCB layer images for visual defects using AI vision capabilities.
Detects overlapping elements, spacing issues, routing problems, and more.

Features:
- Overlap detection
- Spacing uniformity analysis
- Routing angle analysis (45° vs 90°)
- Component placement analysis
- Silkscreen completeness check
- Professional appearance scoring

Usage:
    python image_analyzer.py --image F_Cu.png --analysis routing
    python image_analyzer.py --image F_Silkscreen.png --analysis silkscreen
    python image_analyzer.py --dir ./layer_images --full-analysis
"""

import argparse
import base64
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from enum import Enum

try:
    import anthropic
    HAS_ANTHROPIC = True
except ImportError:
    HAS_ANTHROPIC = False


class AnalysisType(Enum):
    ROUTING = "routing"
    SILKSCREEN = "silkscreen"
    SPACING = "spacing"
    OVERLAPS = "overlaps"
    COMPOSITION = "composition"
    PROFESSIONAL = "professional"
    FULL = "full"


@dataclass
class AnalysisResult:
    """Result of a visual analysis."""
    analysis_type: AnalysisType
    score: float
    passed: bool
    issues: List[str]
    warnings: List[str]
    details: Dict[str, Any]


class ImageAnalyzer:
    """
    Analyzes PCB images for visual defects using Claude vision.
    """

    def __init__(self, api_key: Optional[str] = None):
        if not HAS_ANTHROPIC:
            raise ImportError("anthropic package required: pip install anthropic")

        self.api_key = api_key or os.environ.get('ANTHROPIC_API_KEY')
        if not self.api_key:
            raise ValueError("ANTHROPIC_API_KEY not set")

        self.client = anthropic.Anthropic(api_key=self.api_key)

    def _load_image(self, image_path: str) -> Tuple[str, str]:
        """Load image and return (base64_data, media_type)."""
        with open(image_path, 'rb') as f:
            data = base64.standard_b64encode(f.read()).decode('utf-8')

        ext = Path(image_path).suffix.lower()
        media_types = {
            '.png': 'image/png',
            '.jpg': 'image/jpeg',
            '.jpeg': 'image/jpeg',
            '.gif': 'image/gif',
            '.webp': 'image/webp'
        }
        return data, media_types.get(ext, 'image/png')

    def _analyze_with_prompt(
        self,
        image_path: str,
        system_prompt: str,
        user_prompt: str
    ) -> Dict[str, Any]:
        """Run analysis with specific prompts."""
        image_data, media_type = self._load_image(image_path)

        response = self.client.messages.create(
            model="claude-opus-4-6-20260206",
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

        try:
            json_start = response_text.find('{')
            json_end = response_text.rfind('}') + 1
            if json_start >= 0 and json_end > json_start:
                return json.loads(response_text[json_start:json_end])
        except json.JSONDecodeError:
            pass

        return {"raw_response": response_text}

    def analyze_routing(self, image_path: str) -> AnalysisResult:
        """
        Analyze PCB routing quality.

        Checks:
        - 45° vs 90° angles
        - Trace overlaps
        - Routing cleanliness
        - Trace width consistency
        """
        system_prompt = """You are a PCB routing quality analyzer.
Evaluate trace routing for professional quality standards.

Key criteria:
1. ANGLE QUALITY: Professional PCBs use 45° angles, not 90°
2. OVERLAPS: Traces on same layer should never cross
3. CLEANLINESS: Routing should be organized, not chaotic
4. CONSISTENCY: Similar nets should have similar trace widths"""

        user_prompt = """Analyze this PCB copper layer image for routing quality.

Return JSON with:
{
    "angle_score": 1-10 (10=all 45°, 1=all 90°),
    "overlap_score": 1-10 (10=no overlaps, 1=many overlaps),
    "cleanliness_score": 1-10 (10=very clean, 1=chaotic),
    "consistency_score": 1-10 (10=consistent widths, 1=inconsistent),
    "overall_score": 1-10,
    "passed": true/false (true if overall >= 7),
    "90_degree_angles_found": estimated count,
    "overlapping_traces_found": estimated count,
    "issues": ["list of specific issues"],
    "warnings": ["list of warnings"],
    "recommendations": ["list of improvements"]
}

Be specific about where issues are located (e.g., "center of board", "near U1")."""

        result = self._analyze_with_prompt(image_path, system_prompt, user_prompt)

        return AnalysisResult(
            analysis_type=AnalysisType.ROUTING,
            score=result.get('overall_score', 0),
            passed=result.get('passed', False),
            issues=result.get('issues', []),
            warnings=result.get('warnings', []),
            details=result
        )

    def analyze_silkscreen(self, image_path: str) -> AnalysisResult:
        """
        Analyze silkscreen layer completeness.

        Checks:
        - Reference designator presence
        - Polarity markings
        - Text readability
        - Assembly information
        """
        system_prompt = """You are a silkscreen quality inspector.
Evaluate PCB silkscreen/legend layer for completeness and quality.

Critical requirements:
1. ALL components must have reference designators (R1, C1, U1, etc.)
2. Polarized parts need polarity marks (diodes, electrolytics)
3. ICs need pin 1 indicators
4. Text must be readable (not too small, not overlapping)"""

        user_prompt = """Analyze this PCB silkscreen layer image.

Return JSON with:
{
    "designator_completeness": 1-10 (10=all present, 1=none),
    "polarity_markings": 1-10 (10=all marked, 1=none),
    "readability": 1-10 (10=clear, 1=illegible),
    "assembly_info": 1-10 (10=complete, 1=none),
    "overall_score": 1-10,
    "passed": true/false,
    "visible_designators": ["list of visible designators found"],
    "missing_elements": ["list of missing required elements"],
    "issues": ["list of critical issues"],
    "warnings": ["list of warnings"],
    "is_empty": true/false (true if silkscreen is mostly empty)
}

An EMPTY silkscreen is a CRITICAL failure."""

        result = self._analyze_with_prompt(image_path, system_prompt, user_prompt)

        return AnalysisResult(
            analysis_type=AnalysisType.SILKSCREEN,
            score=result.get('overall_score', 0),
            passed=result.get('passed', False),
            issues=result.get('issues', []),
            warnings=result.get('warnings', []),
            details=result
        )

    def analyze_spacing(self, image_path: str) -> AnalysisResult:
        """
        Analyze component spacing uniformity.

        Checks:
        - Spacing consistency between components
        - Crowding issues
        - Empty areas
        - Alignment
        """
        system_prompt = """You are a PCB layout spacing analyzer.
Evaluate component spacing and placement uniformity.

Key criteria:
1. UNIFORMITY: Similar components should have similar spacing
2. NO CROWDING: Components should not be crammed together
3. NO WASTED SPACE: Large empty areas indicate poor planning
4. ALIGNMENT: Components should align where possible"""

        user_prompt = """Analyze this PCB image for component spacing quality.

Return JSON with:
{
    "uniformity_score": 1-10 (10=uniform, 1=inconsistent),
    "crowding_score": 1-10 (10=no crowding, 1=severe crowding),
    "space_utilization": 1-10 (10=efficient, 1=wasted),
    "alignment_score": 1-10 (10=aligned, 1=random),
    "overall_score": 1-10,
    "passed": true/false,
    "crowded_areas": ["list of crowded regions"],
    "empty_areas": ["list of underutilized regions"],
    "issues": ["list of spacing issues"],
    "warnings": ["list of warnings"],
    "components_in_one_corner": true/false
}"""

        result = self._analyze_with_prompt(image_path, system_prompt, user_prompt)

        return AnalysisResult(
            analysis_type=AnalysisType.SPACING,
            score=result.get('overall_score', 0),
            passed=result.get('passed', False),
            issues=result.get('issues', []),
            warnings=result.get('warnings', []),
            details=result
        )

    def analyze_overlaps(self, image_path: str) -> AnalysisResult:
        """
        Detect overlapping elements in the design.

        Checks:
        - Trace overlaps on same layer
        - Silkscreen overlapping pads
        - Component overlaps
        - Via/pad overlaps
        """
        system_prompt = """You are a PCB overlap detector.
Find all instances of overlapping elements that violate design rules.

Types of overlaps to find:
1. Traces crossing/overlapping on same layer
2. Silkscreen text over pads or vias
3. Components overlapping each other
4. Any visual element intersection that shouldn't exist"""

        user_prompt = """Analyze this PCB image for overlapping elements.

Return JSON with:
{
    "trace_overlaps": number of trace overlaps found,
    "silkscreen_overlaps": number of silkscreen/pad overlaps,
    "component_overlaps": number of component overlaps,
    "other_overlaps": number of other overlaps,
    "total_overlaps": total count,
    "overall_score": 1-10 (10=no overlaps, 1=many overlaps),
    "passed": true/false (true if no critical overlaps),
    "overlap_locations": ["list of overlap locations"],
    "issues": ["list of critical overlaps"],
    "warnings": ["list of minor overlaps"]
}

ANY trace overlap on the same layer is a CRITICAL failure."""

        result = self._analyze_with_prompt(image_path, system_prompt, user_prompt)

        return AnalysisResult(
            analysis_type=AnalysisType.OVERLAPS,
            score=result.get('overall_score', 0),
            passed=result.get('passed', False),
            issues=result.get('issues', []),
            warnings=result.get('warnings', []),
            details=result
        )

    def analyze_professional_appearance(self, image_path: str) -> AnalysisResult:
        """
        Score overall professional appearance.

        Holistic evaluation of whether the design looks professional
        and production-ready.
        """
        system_prompt = """You are an expert PCB design judge evaluating professional quality.
Score designs like judging a design competition.

Scoring guide:
- 10: Award-winning, could be in a design magazine
- 8-9: Professional quality, ready for production
- 6-7: Acceptable, minor improvements needed
- 4-5: Below average, needs work
- 1-3: Amateur, would not pass professional review"""

        user_prompt = """Score this PCB design for professional appearance.

Return JSON with:
{
    "aesthetics_score": 1-10,
    "organization_score": 1-10,
    "craftsmanship_score": 1-10,
    "first_impression_score": 1-10,
    "overall_score": 1-10,
    "passed": true/false (true if overall >= 7),
    "professional_tier": "award-winning" | "professional" | "acceptable" | "below-average" | "amateur",
    "strongest_aspects": ["list of best features"],
    "weakest_aspects": ["list of worst features"],
    "issues": ["critical issues preventing professional quality"],
    "warnings": ["improvements needed"],
    "one_sentence_verdict": "Summary of design quality"
}

Be honest and rigorous. Amateur work must be identified as such."""

        result = self._analyze_with_prompt(image_path, system_prompt, user_prompt)

        return AnalysisResult(
            analysis_type=AnalysisType.PROFESSIONAL,
            score=result.get('overall_score', 0),
            passed=result.get('passed', False),
            issues=result.get('issues', []),
            warnings=result.get('warnings', []),
            details=result
        )

    def full_analysis(self, image_path: str) -> Dict[str, AnalysisResult]:
        """Run all analysis types on an image."""
        image_name = Path(image_path).name.lower()

        results = {}

        # Determine which analyses are relevant based on layer name
        if 'silkscreen' in image_name or 'silk' in image_name:
            results['silkscreen'] = self.analyze_silkscreen(image_path)
        elif 'cu' in image_name or 'copper' in image_name:
            results['routing'] = self.analyze_routing(image_path)
            results['overlaps'] = self.analyze_overlaps(image_path)
        else:
            # Run all for composite/unknown images
            results['routing'] = self.analyze_routing(image_path)
            results['spacing'] = self.analyze_spacing(image_path)
            results['overlaps'] = self.analyze_overlaps(image_path)
            results['professional'] = self.analyze_professional_appearance(image_path)

        return results

    def analyze_directory(self, directory: str, pattern: str = "*.png") -> Dict[str, Any]:
        """Analyze all images in a directory."""
        dir_path = Path(directory)
        images = list(dir_path.glob(pattern))

        if not images:
            return {"error": f"No images matching {pattern} in {directory}"}

        all_results = {}
        for img in images:
            print(f"Analyzing {img.name}...")
            all_results[img.name] = self.full_analysis(str(img))

        # Calculate summary statistics
        all_scores = []
        all_passed = []
        all_issues = []

        for img_name, analyses in all_results.items():
            for analysis_type, result in analyses.items():
                all_scores.append(result.score)
                all_passed.append(result.passed)
                all_issues.extend(result.issues)

        return {
            "directory": directory,
            "images_analyzed": len(images),
            "total_analyses": len(all_scores),
            "average_score": round(sum(all_scores) / len(all_scores), 2) if all_scores else 0,
            "pass_rate": round(sum(all_passed) / len(all_passed) * 100, 1) if all_passed else 0,
            "total_issues": len(all_issues),
            "unique_issues": list(set(all_issues)),
            "overall_passed": all(all_passed) if all_passed else False,
            "image_results": {
                k: {
                    t: {
                        "score": r.score,
                        "passed": r.passed,
                        "issues": r.issues
                    }
                    for t, r in v.items()
                }
                for k, v in all_results.items()
            }
        }


def main():
    parser = argparse.ArgumentParser(
        description='Image Analyzer for PCB Visual Defect Detection'
    )
    parser.add_argument(
        '--image', '-i',
        type=str,
        help='Image to analyze'
    )
    parser.add_argument(
        '--dir', '-d',
        type=str,
        help='Directory of images to analyze'
    )
    parser.add_argument(
        '--pattern',
        type=str,
        default='*.png',
        help='File pattern for directory (default: *.png)'
    )
    parser.add_argument(
        '--analysis', '-a',
        type=str,
        choices=[t.value for t in AnalysisType],
        default='full',
        help='Type of analysis to run'
    )
    parser.add_argument(
        '--json',
        action='store_true',
        help='Output as JSON'
    )

    args = parser.parse_args()

    if not args.image and not args.dir:
        parser.error("Specify --image or --dir")

    if not HAS_ANTHROPIC or not os.environ.get('ANTHROPIC_API_KEY'):
        print("ERROR: ANTHROPIC_API_KEY not set")
        print("Set API key: export ANTHROPIC_API_KEY=your-key")
        sys.exit(1)

    analyzer = ImageAnalyzer()

    if args.dir:
        results = analyzer.analyze_directory(args.dir, args.pattern)
    else:
        analysis_type = AnalysisType(args.analysis)
        if analysis_type == AnalysisType.FULL:
            results = analyzer.full_analysis(args.image)
            # Convert AnalysisResult objects to dicts
            results = {
                k: {
                    "score": v.score,
                    "passed": v.passed,
                    "issues": v.issues,
                    "warnings": v.warnings,
                    "details": v.details
                }
                for k, v in results.items()
            }
        elif analysis_type == AnalysisType.ROUTING:
            r = analyzer.analyze_routing(args.image)
            results = {"routing": {"score": r.score, "passed": r.passed, "issues": r.issues, "details": r.details}}
        elif analysis_type == AnalysisType.SILKSCREEN:
            r = analyzer.analyze_silkscreen(args.image)
            results = {"silkscreen": {"score": r.score, "passed": r.passed, "issues": r.issues, "details": r.details}}
        elif analysis_type == AnalysisType.SPACING:
            r = analyzer.analyze_spacing(args.image)
            results = {"spacing": {"score": r.score, "passed": r.passed, "issues": r.issues, "details": r.details}}
        elif analysis_type == AnalysisType.OVERLAPS:
            r = analyzer.analyze_overlaps(args.image)
            results = {"overlaps": {"score": r.score, "passed": r.passed, "issues": r.issues, "details": r.details}}
        elif analysis_type == AnalysisType.PROFESSIONAL:
            r = analyzer.analyze_professional_appearance(args.image)
            results = {"professional": {"score": r.score, "passed": r.passed, "issues": r.issues, "details": r.details}}

    if args.json:
        print(json.dumps(results, indent=2, default=str))
    else:
        print("\n" + "=" * 60)
        print("IMAGE ANALYSIS RESULTS")
        print("=" * 60)

        if 'error' in results:
            print(f"ERROR: {results['error']}")
        elif 'directory' in results:
            print(f"Directory: {results['directory']}")
            print(f"Images Analyzed: {results['images_analyzed']}")
            print(f"Average Score: {results['average_score']}/10")
            print(f"Pass Rate: {results['pass_rate']}%")
            print(f"Total Issues: {results['total_issues']}")
            print(f"Overall: {'PASS' if results['overall_passed'] else 'FAIL'}")

            if results.get('unique_issues'):
                print("\nUnique Issues Found:")
                for issue in results['unique_issues'][:10]:
                    print(f"  - {issue}")
        else:
            for analysis_name, data in results.items():
                print(f"\n{analysis_name.upper()}:")
                print(f"  Score: {data['score']}/10")
                print(f"  Status: {'PASS' if data['passed'] else 'FAIL'}")
                if data.get('issues'):
                    print("  Issues:")
                    for issue in data['issues'][:5]:
                        print(f"    - {issue}")


if __name__ == '__main__':
    main()
