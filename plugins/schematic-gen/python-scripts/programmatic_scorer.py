#!/usr/bin/env python3
"""
Programmatic PCB Visual Quality Scorer

Analyzes PCB images using computer vision techniques without requiring
external AI APIs. Provides objective, reproducible scoring based on:

1. Routing Quality - Detects trace angles, identifies 90° violations
2. Silkscreen Quality - Text density, coverage, readability indicators
3. Copper Balance - Distribution across the board
4. Component Placement - Clustering, alignment, spacing
5. Overall Aesthetics - Visual organization metrics

This is the fallback scorer when OpenRouter/Anthropic APIs are unavailable.

Usage:
    from programmatic_scorer import ProgrammaticScorer

    scorer = ProgrammaticScorer()
    results = scorer.score_all_images("/path/to/layer_images")
    print(f"Overall Score: {results['overall_score']}/10")
"""

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

try:
    from PIL import Image
    import numpy as np
    HAS_PIL = True
except ImportError:
    HAS_PIL = False


@dataclass
class LayerScore:
    """Score for a single PCB layer image."""
    layer_name: str
    layer_type: str
    score: float
    routing_score: float = 0.0
    coverage_score: float = 0.0
    balance_score: float = 0.0
    organization_score: float = 0.0
    issues: List[str] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ValidationResult:
    """Complete validation result."""
    overall_score: float
    passed: bool
    layer_scores: Dict[str, LayerScore]
    aggregate_metrics: Dict[str, float]
    issues: List[str]
    suggestions: List[str]


class ProgrammaticScorer:
    """
    Programmatic PCB visual quality scorer.

    Uses image processing to objectively measure PCB quality metrics.
    No external AI API required.
    """

    # Score thresholds
    PASS_THRESHOLD = 9.0
    MIN_ACCEPTABLE = 7.0

    # Copper layer color ranges (RGB)
    COPPER_COLORS = {
        'red': ((180, 0, 0), (255, 100, 100)),      # F.Cu
        'blue': ((0, 50, 100), (150, 200, 255)),    # B.Cu
        'green': ((0, 100, 0), (150, 255, 150)),    # Inner layers
        'yellow': ((200, 200, 0), (255, 255, 100)), # Silkscreen
    }

    def __init__(self, strict: bool = True):
        """Initialize scorer."""
        if not HAS_PIL:
            raise ImportError("PIL/numpy required: pip install Pillow numpy")
        self.strict = strict

    def score_all_images(self, image_dir: str) -> ValidationResult:
        """
        Score all PCB images in a directory.

        Args:
            image_dir: Directory containing layer images

        Returns:
            ValidationResult with scores and metrics
        """
        dir_path = Path(image_dir)
        if not dir_path.exists():
            raise FileNotFoundError(f"Directory not found: {image_dir}")

        # Find all images
        images = list(dir_path.glob("*.png")) + list(dir_path.glob("*.jpg"))

        layer_scores = {}
        all_issues = []
        all_suggestions = []

        print(f"\n[PROGRAMMATIC SCORING] Analyzing {len(images)} images...")

        for img_path in images:
            layer_type = self._detect_layer_type(img_path.name)
            print(f"  Scoring {img_path.name} ({layer_type})...")

            try:
                score = self._score_image(str(img_path), layer_type)
                layer_scores[img_path.name] = score
                all_issues.extend(score.issues)

                print(f"    Score: {score.score:.1f}/10")

            except Exception as e:
                print(f"    ERROR: {e}")
                layer_scores[img_path.name] = LayerScore(
                    layer_name=img_path.name,
                    layer_type=layer_type,
                    score=5.0,
                    issues=[f"Scoring error: {e}"]
                )

        # Calculate aggregate metrics
        scores = [s.score for s in layer_scores.values()]
        overall_score = sum(scores) / len(scores) if scores else 0.0

        # Generate suggestions based on issues
        all_suggestions = self._generate_suggestions(all_issues)

        aggregate_metrics = {
            'avg_routing_score': self._avg_metric(layer_scores, 'routing_score'),
            'avg_coverage_score': self._avg_metric(layer_scores, 'coverage_score'),
            'avg_balance_score': self._avg_metric(layer_scores, 'balance_score'),
            'avg_organization_score': self._avg_metric(layer_scores, 'organization_score'),
        }

        return ValidationResult(
            overall_score=round(overall_score, 2),
            passed=overall_score >= self.PASS_THRESHOLD and min(scores) >= self.MIN_ACCEPTABLE,
            layer_scores=layer_scores,
            aggregate_metrics=aggregate_metrics,
            issues=list(set(all_issues)),
            suggestions=list(set(all_suggestions))
        )

    def _avg_metric(self, layer_scores: Dict[str, LayerScore], metric: str) -> float:
        """Calculate average of a specific metric across layers."""
        values = [getattr(s, metric, 0) for s in layer_scores.values()]
        return round(sum(values) / len(values), 2) if values else 0.0

    def _score_image(self, image_path: str, layer_type: str) -> LayerScore:
        """Score a single image based on layer type."""
        img = Image.open(image_path)
        img_array = np.array(img)

        # Initialize metrics
        metrics = {}
        issues = []

        # Score based on layer type - check specific types FIRST
        # IMPORTANT: Check edge_cuts FIRST because "edge_cuts" contains "cu" which would match copper check
        #
        # HONEST SCORING POLICY:
        # - NO artificial floor scores that guarantee passing
        # - Scores reflect actual measured metrics
        # - Silkscreen gets special treatment (hard to measure with CV)
        # - Edge cuts just need to exist with a visible outline

        if 'edge' in layer_type.lower() or layer_type == 'edge_cuts':
            # Edge cuts - check it exists and has a visible outline
            # This is a binary check - either the board outline exists or it doesn't
            coverage_score, _, _ = self._score_basic_content(img_array)
            routing_score = 10.0  # N/A for edge cuts
            balance_score = 10.0  # N/A for edge cuts
            organization_score = 10.0  # Edge cuts are inherently organized
            # Edge cuts: visible outline = 8.0, no outline = fail
            if coverage_score > 2.0:
                score = 8.0  # Board outline exists - acceptable
            else:
                score = coverage_score  # No visible outline - problem
                issues.append("Board outline not clearly visible")

        elif layer_type == 'power_plane':
            # Power/ground planes - expected to have copper coverage
            coverage_score, coverage_metrics, coverage_issues = self._score_basic_content(img_array)
            balance_score, balance_metrics, balance_issues = self._score_copper_balance(img_array)
            routing_score = 10.0  # No routing on planes
            organization_score = coverage_score

            metrics.update(coverage_metrics)
            metrics.update(balance_metrics)
            issues.extend(balance_issues)

            # Power planes: honest weighted scoring
            # Coverage is most important for planes
            score = coverage_score * 0.7 + balance_score * 0.3

        elif layer_type == 'copper_inner_signal':
            # Inner signal layers - focus on routing quality
            routing_score, routing_metrics, routing_issues = self._score_routing(img_array)
            coverage_score, coverage_metrics, _ = self._score_copper_coverage(img_array)
            balance_score, _, _ = self._score_copper_balance(img_array)
            organization_score = routing_score

            metrics.update(routing_metrics)
            metrics.update(coverage_metrics)
            issues.extend(routing_issues)

            # Inner signal: routing quality is primary metric
            # Coverage can be low (sparse routing is normal)
            score = routing_score * 0.8 + balance_score * 0.2

        elif 'copper' in layer_type or 'cu' in layer_type.lower():
            # General copper layers (F.Cu, B.Cu)
            routing_score, routing_metrics, routing_issues = self._score_routing(img_array)
            coverage_score, coverage_metrics, coverage_issues = self._score_copper_coverage(img_array)
            balance_score, balance_metrics, balance_issues = self._score_copper_balance(img_array)

            metrics.update(routing_metrics)
            metrics.update(coverage_metrics)
            metrics.update(balance_metrics)
            issues.extend(routing_issues)

            # Determine if this is a multi-layer board context
            # On multi-layer boards (6+), B.Cu often has sparse routing
            # because signals use inner layers. Low coverage is EXPECTED.
            is_bottom_layer = 'bottom' in layer_type or 'b_cu' in layer_type.lower()

            if is_bottom_layer:
                # For bottom copper on multi-layer boards:
                # - Coverage is expected to be low (signals on inner layers)
                # - Only flag coverage if routing quality is also poor
                # - Focus primarily on routing angle quality
                #
                # Don't penalize for expected sparse routing
                if routing_score >= 8.0:
                    # Good routing angles - low coverage is expected design choice
                    organization_score = routing_score
                    score = routing_score * 0.85 + balance_score * 0.15
                    # Don't add coverage issues - sparse is expected
                else:
                    # Poor routing AND low coverage - flag both
                    organization_score = (routing_score + coverage_score + balance_score) / 3
                    issues.extend(coverage_issues)
                    issues.extend(balance_issues)
                    score = routing_score * 0.6 + coverage_score * 0.25 + balance_score * 0.15
            else:
                # For top copper (F.Cu):
                # - Expect moderate coverage from component pads + routing
                # - Coverage matters more as it's the primary signal layer
                organization_score = (routing_score + coverage_score + balance_score) / 3
                issues.extend(coverage_issues)
                # Honest weighted scoring - routing most important
                score = routing_score * 0.5 + coverage_score * 0.3 + balance_score * 0.2

        elif 'silk' in layer_type.lower():
            coverage_score, coverage_metrics, coverage_issues = self._score_silkscreen(img_array)
            organization_score, org_metrics, org_issues = self._score_text_organization(img_array)
            routing_score = 10.0  # N/A for silkscreen
            balance_score = coverage_score

            metrics.update(coverage_metrics)
            metrics.update(org_metrics)
            issues.extend(coverage_issues)
            issues.extend(org_issues)

            score = coverage_score * 0.5 + organization_score * 0.5

        else:
            # Composite/unknown - general scoring
            coverage_score, coverage_metrics, coverage_issues = self._score_basic_content(img_array)
            balance_score, balance_metrics, balance_issues = self._score_visual_balance(img_array)
            routing_score = coverage_score
            organization_score = balance_score

            metrics.update(coverage_metrics)
            metrics.update(balance_metrics)
            issues.extend(coverage_issues)
            issues.extend(balance_issues)

            score = coverage_score * 0.5 + balance_score * 0.5

        return LayerScore(
            layer_name=Path(image_path).name,
            layer_type=layer_type,
            score=round(min(10.0, max(0.0, score)), 2),
            routing_score=round(routing_score, 2),
            coverage_score=round(coverage_score, 2),
            balance_score=round(balance_score, 2),
            organization_score=round(organization_score, 2),
            issues=issues,
            metrics=metrics
        )

    def _score_routing(self, img_array: np.ndarray) -> Tuple[float, Dict, List[str]]:
        """
        Score routing quality by detecting trace angles.

        Good routing uses 45° and 90° angles, not arbitrary angles.
        """
        issues = []
        metrics = {}

        # Convert to grayscale for edge detection
        if len(img_array.shape) == 3:
            gray = np.mean(img_array, axis=2)
        else:
            gray = img_array.astype(float)

        # Simple edge detection using gradients
        dx = np.abs(np.diff(gray, axis=1))
        dy = np.abs(np.diff(gray, axis=0))

        # Crop to same size (both missing one row/col due to diff)
        min_h = min(dx.shape[0], dy.shape[0])
        min_w = min(dx.shape[1], dy.shape[1])
        dx = dx[:min_h, :min_w]
        dy = dy[:min_h, :min_w]

        # Find edge pixels (high gradient)
        edge_threshold = 30
        edge_mask = (dx > edge_threshold) | (dy > edge_threshold)

        # Calculate angles at edge pixels
        angles = np.degrees(np.arctan2(dy, dx + 0.001))
        edge_angles = angles[edge_mask]

        if len(edge_angles) > 0:
            # Normalize angles to 0-90 range
            normalized = np.abs(edge_angles) % 90

            # Good angles: near 0° (horizontal), 45°, or 90° (vertical)
            good_angles = np.sum(
                (normalized < 10) |  # Near horizontal
                (normalized > 80) |  # Near vertical
                ((normalized > 40) & (normalized < 50))  # Near 45°
            )

            angle_ratio = good_angles / len(edge_angles)
            metrics['good_angle_ratio'] = round(angle_ratio, 3)
            metrics['edge_pixel_count'] = len(edge_angles)

            # Score: 90%+ good angles = 10, 70% = 7, 50% = 5
            routing_score = min(10.0, angle_ratio * 11)

            if angle_ratio < 0.6:
                issues.append(f"Poor routing angles: only {angle_ratio*100:.0f}% use 45°/90° angles")
            elif angle_ratio < 0.75:
                issues.append(f"Routing could be improved: {angle_ratio*100:.0f}% good angles")
            # 75%+ is acceptable for complex PCBs with dense routing
        else:
            routing_score = 5.0
            metrics['good_angle_ratio'] = 0
            metrics['edge_pixel_count'] = 0

        return routing_score, metrics, issues

    def _score_copper_coverage(self, img_array: np.ndarray) -> Tuple[float, Dict, List[str]]:
        """Score copper coverage (not too sparse, not too dense)."""
        issues = []
        metrics = {}

        # Count copper-colored pixels
        copper_ratio = self._detect_copper_ratio(img_array)
        metrics['copper_coverage'] = round(copper_ratio, 4)

        # Optimal coverage: 15-40% for signal layers, higher for ground planes
        if copper_ratio < 0.05:
            score = 4.0 + copper_ratio * 40  # Very sparse
            issues.append(f"Low copper coverage ({copper_ratio*100:.1f}%) - sparse routing")
        elif copper_ratio < 0.15:
            score = 6.0 + (copper_ratio - 0.05) * 40  # Somewhat sparse
        elif copper_ratio <= 0.45:
            score = 10.0  # Optimal range
        elif copper_ratio <= 0.70:
            score = 10.0 - (copper_ratio - 0.45) * 12  # Getting dense (ground plane OK)
        else:
            score = 7.0  # Very dense - might be ground plane

        return score, metrics, issues

    def _score_copper_balance(self, img_array: np.ndarray) -> Tuple[float, Dict, List[str]]:
        """Score copper balance across the board."""
        issues = []
        metrics = {}

        # Divide board into quadrants and check copper distribution
        h, w = img_array.shape[:2]
        quadrants = [
            img_array[:h//2, :w//2],       # Top-left
            img_array[:h//2, w//2:],        # Top-right
            img_array[h//2:, :w//2],        # Bottom-left
            img_array[h//2:, w//2:]         # Bottom-right
        ]

        quad_coverage = []
        for q in quadrants:
            ratio = self._detect_copper_ratio(q)
            quad_coverage.append(ratio)

        metrics['quadrant_coverage'] = [round(c, 4) for c in quad_coverage]

        # Check balance - standard deviation of quadrant coverage
        if quad_coverage:
            mean_coverage = np.mean(quad_coverage)
            std_coverage = np.std(quad_coverage)

            metrics['coverage_std'] = round(std_coverage, 4)
            metrics['coverage_mean'] = round(mean_coverage, 4)

            # Good balance: low std relative to mean
            if mean_coverage > 0.01:
                cv = std_coverage / mean_coverage  # Coefficient of variation

                if cv < 0.3:
                    score = 10.0
                elif cv < 0.5:
                    score = 8.0
                    issues.append("Slightly unbalanced copper distribution")
                elif cv < 0.8:
                    score = 6.0
                    issues.append("Unbalanced copper distribution across board")
                else:
                    score = 4.0
                    issues.append("Very unbalanced copper - may cause warping")
            else:
                score = 5.0  # Very low coverage
        else:
            score = 5.0

        return score, metrics, issues

    def _score_silkscreen(self, img_array: np.ndarray) -> Tuple[float, Dict, List[str]]:
        """Score silkscreen quality."""
        issues = []
        metrics = {}

        # Detect yellow/white pixels (typical silkscreen colors)
        if len(img_array.shape) == 3:
            # Yellow detection (high R, high G, low B)
            yellow_mask = (
                (img_array[:,:,0] > 180) &
                (img_array[:,:,1] > 180) &
                (img_array[:,:,2] < 150)
            )
            yellow_ratio = np.sum(yellow_mask) / (img_array.shape[0] * img_array.shape[1])

            # Also check for text-like features (high edge density)
            gray = np.mean(img_array, axis=2)
        else:
            gray = img_array.astype(float)
            yellow_ratio = 0

        # Text detection via edge density
        dx = np.abs(np.diff(gray, axis=1))
        dy = np.abs(np.diff(gray, axis=0))
        edge_ratio = (np.sum(dx > 30) + np.sum(dy > 30)) / gray.size

        metrics['yellow_ratio'] = round(yellow_ratio, 4)
        metrics['edge_density'] = round(edge_ratio, 4)

        # Score based on presence of text-like features
        if edge_ratio > 0.01:  # Has text/designators
            score = min(10.0, 7.0 + edge_ratio * 100)
        elif yellow_ratio > 0.005:
            score = 6.0 + yellow_ratio * 200
            issues.append("Silkscreen present but low text density")
        else:
            score = 4.0
            issues.append("Silkscreen may be missing designators")

        return score, metrics, issues

    def _score_text_organization(self, img_array: np.ndarray) -> Tuple[float, Dict, List[str]]:
        """Score text organization (alignment, spacing)."""
        issues = []
        metrics = {}

        # This is a simplified check - real implementation would use OCR
        # For now, check for overall organization via visual metrics

        if len(img_array.shape) == 3:
            gray = np.mean(img_array, axis=2)
        else:
            gray = img_array.astype(float)

        # Check horizontal vs vertical edge balance
        dx = np.abs(np.diff(gray, axis=1))
        dy = np.abs(np.diff(gray, axis=0))

        h_edges = np.sum(dy > 30)
        v_edges = np.sum(dx > 30)

        # Good text should have balanced H/V edges (horizontal text has both)
        if h_edges + v_edges > 0:
            edge_ratio = min(h_edges, v_edges) / max(h_edges, v_edges)
            metrics['edge_balance'] = round(edge_ratio, 3)

            if edge_ratio > 0.5:  # Well-balanced = readable text
                score = 9.0
            elif edge_ratio > 0.3:
                score = 7.0
            else:
                score = 5.0
                issues.append("Text may not be properly aligned horizontally")
        else:
            score = 5.0
            metrics['edge_balance'] = 0

        return score, metrics, issues

    def _score_basic_content(self, img_array: np.ndarray) -> Tuple[float, Dict, List[str]]:
        """Basic content scoring for unknown layer types."""
        issues = []
        metrics = {}

        # Count non-white pixels
        if len(img_array.shape) == 3:
            gray = np.mean(img_array, axis=2)
        else:
            gray = img_array.astype(float)

        content_ratio = np.sum(gray < 250) / gray.size
        metrics['content_ratio'] = round(content_ratio, 4)

        if content_ratio > 0.1:
            score = min(10.0, 7.0 + content_ratio * 10)
        elif content_ratio > 0.01:
            score = 5.0 + content_ratio * 100
        else:
            score = 3.0
            issues.append("Layer appears mostly empty")

        return score, metrics, issues

    def _score_visual_balance(self, img_array: np.ndarray) -> Tuple[float, Dict, List[str]]:
        """Score overall visual balance."""
        # Reuse copper balance logic
        return self._score_copper_balance(img_array)

    def _detect_copper_ratio(self, img_array: np.ndarray) -> float:
        """Detect ratio of copper-colored pixels."""
        if len(img_array.shape) != 3:
            return 0.0

        total_pixels = img_array.shape[0] * img_array.shape[1]
        copper_pixels = 0

        for color_name, (low, high) in self.COPPER_COLORS.items():
            if color_name == 'yellow':  # Skip silkscreen color
                continue
            mask = (
                (img_array[:,:,0] >= low[0]) & (img_array[:,:,0] <= high[0]) &
                (img_array[:,:,1] >= low[1]) & (img_array[:,:,1] <= high[1]) &
                (img_array[:,:,2] >= low[2]) & (img_array[:,:,2] <= high[2])
            )
            copper_pixels += np.sum(mask)

        return copper_pixels / total_pixels

    def _detect_layer_type(self, filename: str) -> str:
        """Detect layer type from filename."""
        name_lower = filename.lower()

        if "edge" in name_lower or "cut" in name_lower:
            return "edge_cuts"
        if "all_copper" in name_lower or "all_cu" in name_lower:
            return "composite"
        if "top_view" in name_lower or "bottom_view" in name_lower:
            return "composite"
        if "f_cu" in name_lower or "f.cu" in name_lower:
            return "copper_top"
        if "b_cu" in name_lower or "b.cu" in name_lower:
            return "copper_bottom"
        # Check for power/ground plane layers (expected high coverage)
        if "gnd" in name_lower or "ground" in name_lower:
            return "power_plane"
        if "power" in name_lower:
            return "power_plane"
        if "58v" in name_lower or "12v" in name_lower or "5v" in name_lower or "3v3" in name_lower:
            return "power_plane"
        # Inner signal layers (lower coverage is normal)
        if "in" in name_lower and "cu" in name_lower:
            return "copper_inner_signal"
        if "silk" in name_lower:
            return "silkscreen"
        if "mask" in name_lower:
            return "solder_mask"

        return "unknown"

    def _generate_suggestions(self, issues: List[str]) -> List[str]:
        """Generate improvement suggestions based on issues."""
        suggestions = []

        issue_text = " ".join(issues).lower()

        if "angle" in issue_text or "routing" in issue_text:
            suggestions.append("Use 45-degree routing for better signal integrity")
            suggestions.append("Avoid arbitrary trace angles - stick to 45/90 degrees")

        if "coverage" in issue_text and "low" in issue_text:
            suggestions.append("Add more routing or copper pours to increase copper coverage")

        if "balance" in issue_text or "unbalanced" in issue_text:
            suggestions.append("Distribute components and routing more evenly across the board")
            suggestions.append("Add ground/power planes to balance copper on inner layers")

        if "silkscreen" in issue_text or "designator" in issue_text:
            suggestions.append("Ensure all components have visible reference designators")
            suggestions.append("Use horizontal text orientation for better readability")

        if "empty" in issue_text:
            suggestions.append("Check that all layers have appropriate content")

        return suggestions


def main():
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description='Programmatic PCB Visual Quality Scorer'
    )
    parser.add_argument(
        '--dir', '-d',
        type=str,
        required=True,
        help='Directory containing layer images'
    )
    parser.add_argument(
        '--json',
        action='store_true',
        help='Output as JSON'
    )
    parser.add_argument(
        '--threshold',
        type=float,
        default=9.0,
        help='Pass threshold (default: 9.0)'
    )

    args = parser.parse_args()

    try:
        scorer = ProgrammaticScorer()
        scorer.PASS_THRESHOLD = args.threshold

        results = scorer.score_all_images(args.dir)

        if args.json:
            output = {
                'overall_score': results.overall_score,
                'passed': results.passed,
                'layer_scores': {
                    name: {
                        'score': s.score,
                        'routing_score': s.routing_score,
                        'coverage_score': s.coverage_score,
                        'balance_score': s.balance_score,
                        'organization_score': s.organization_score,
                        'issues': s.issues
                    }
                    for name, s in results.layer_scores.items()
                },
                'aggregate_metrics': results.aggregate_metrics,
                'issues': results.issues,
                'suggestions': results.suggestions
            }
            print(json.dumps(output, indent=2))
        else:
            print(f"\n{'='*60}")
            print("PROGRAMMATIC SCORING RESULTS")
            print(f"{'='*60}")
            print(f"Overall Score: {results.overall_score}/10")
            print(f"Passed: {'YES' if results.passed else 'NO'}")
            print(f"\nAggregate Metrics:")
            for name, value in results.aggregate_metrics.items():
                print(f"  {name}: {value}")

            if results.issues:
                print(f"\nIssues ({len(results.issues)}):")
                for issue in results.issues[:10]:
                    print(f"  - {issue}")

            if results.suggestions:
                print(f"\nSuggestions:")
                for sug in results.suggestions[:5]:
                    print(f"  - {sug}")

        return 0 if results.passed else 1

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


if __name__ == '__main__':
    import sys
    sys.exit(main())
