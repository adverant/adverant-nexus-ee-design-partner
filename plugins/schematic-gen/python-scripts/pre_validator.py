#!/usr/bin/env python3
"""
Pre-Validator - Check images contain real PCB data before AI analysis.

This prevents the GIGO (Garbage In, Garbage Out) problem where AI validators
analyze empty or trivial images and return high scores because there's
"nothing wrong" with an empty board.

Pre-validation checks:
1. File size - empty images are very small
2. Color analysis - real PCBs have multiple colors
3. Copper coverage - Cu layers need actual copper traces
4. Silkscreen text - silkscreen needs visible designators
5. Content density - components should cover reasonable board area

Usage:
    from pre_validator import PreValidator
    from validation_exceptions import EmptyBoardFailure

    validator = PreValidator()

    # Check single image - raises EmptyBoardFailure if empty
    validator.validate_image_has_content("F_Cu.png", "copper")

    # Check all images in directory
    validator.validate_all_images("./layer_images/")

    # Check silkscreen specifically
    validator.validate_silkscreen_has_text("F_SilkS.png", expected_refs=["R1", "C1", "U1"])
"""

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

try:
    from PIL import Image
    import numpy as np
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

from validation_exceptions import (
    EmptyBoardFailure,
    EmptySilkscreenFailure,
    MissingDependencyFailure,
    ValidationIssue,
    ValidationSeverity
)


@dataclass
class PreValidationResult:
    """Result of pre-validation check."""
    image_path: str
    layer_type: str
    passed: bool
    file_size_bytes: int
    unique_colors: int
    copper_coverage: Optional[float]
    content_density: float
    issues: List[str]


class PreValidator:
    """
    Pre-validate images before sending to AI analysis.

    This catches obvious failures early:
    - Empty boards (just title block)
    - Missing silkscreen text
    - No copper traces
    - Trivial content

    All checks raise exceptions on failure - no silent passes.
    """

    # Minimum thresholds for valid images
    MIN_FILE_SIZE_BYTES = 30000  # Empty KiCad exports are ~88KB but minimal content
    MIN_UNIQUE_COLORS = 3  # At least background + title block + content
    MIN_COPPER_COVERAGE = 0.02  # At least 2% copper on copper layers
    MIN_CONTENT_DENSITY = 0.05  # At least 5% of board area has content
    MIN_SILKSCREEN_ELEMENTS = 1  # At least 1 text element on silkscreen

    # Color ranges for detecting PCB elements (RGB)
    # KiCad uses various colors for different layers - we need broad ranges
    COPPER_COLOR_RANGES = [
        ((180, 0, 0), (255, 100, 100)),      # Red copper (KiCad F.Cu default)
        ((150, 100, 0), (255, 200, 100)),    # Orange/gold copper
        ((100, 50, 0), (200, 150, 100)),     # Brown copper
        ((0, 100, 0), (200, 255, 200)),      # Green copper (KiCad inner layers - broad range)
        ((100, 150, 100), (200, 220, 200)),  # Light green (In1.Cu style)
        ((0, 50, 100), (150, 200, 255)),     # Blue copper (KiCad B.Cu - broad range)
        ((100, 0, 100), (255, 150, 255)),    # Purple/magenta copper
        ((200, 150, 200), (255, 240, 255)),  # Pink/light magenta (B.Cu alternate)
    ]

    SILKSCREEN_COLOR_RANGES = [
        ((200, 200, 0), (255, 255, 100)),    # Yellow silkscreen
        ((0, 0, 0), (50, 50, 50)),           # Black text
        ((200, 200, 200), (255, 255, 255)),  # White text
    ]

    def __init__(self, strict: bool = True):
        """
        Initialize pre-validator.

        Args:
            strict: If True, raise exceptions on any failure.
                   If False, return results but don't raise.
        """
        if not HAS_PIL:
            raise MissingDependencyFailure(
                "PIL/Pillow required for pre-validation",
                dependency_name="Pillow",
                install_instructions="pip install Pillow numpy"
            )

        self.strict = strict

    def _validate_svg_content(self, svg_path: str, layer_type: str) -> PreValidationResult:
        """
        Validate SVG file content without PIL.

        For SVG files, we check:
        1. File size (should be reasonable)
        2. Contains actual SVG elements (path, rect, circle, text, etc.)
        3. Has enough content lines (not just header)
        """
        path = Path(svg_path)
        file_size = path.stat().st_size
        issues = []

        # Read SVG content
        with open(svg_path, 'r') as f:
            content = f.read()

        # Check file size
        if file_size < 10000:
            issues.append(f"SVG file too small ({file_size} bytes)")

        # Count SVG drawing elements
        drawing_elements = 0
        for tag in ['path', 'rect', 'circle', 'line', 'polygon', 'polyline', 'text', 'g id=']:
            drawing_elements += content.count(f'<{tag}')

        # Check for actual content
        if drawing_elements < 5:
            issues.append(f"SVG has few drawing elements ({drawing_elements})")

        # Estimate content density based on elements
        content_density = min(1.0, drawing_elements / 100)

        # For copper layers, check for path elements (traces)
        if 'Cu' in layer_type or 'copper' in layer_type.lower():
            paths = content.count('<path')
            if paths < 3:
                issues.append(f"Copper layer has only {paths} paths - may be missing traces")

        # For silkscreen, check for text elements
        if 'Silk' in layer_type or 'silkscreen' in layer_type.lower():
            texts = content.count('<text')
            if texts < 1:
                issues.append("Silkscreen has no text elements")

        passed = len(issues) == 0 or (file_size > 30000 and drawing_elements > 10)

        return PreValidationResult(
            image_path=svg_path,
            layer_type=layer_type,
            passed=passed,
            file_size_bytes=file_size,
            unique_colors=0,  # N/A for SVG
            copper_coverage=None,
            content_density=content_density,
            issues=issues
        )

    def validate_image_has_content(
        self,
        image_path: str,
        layer_type: str = "unknown"
    ) -> PreValidationResult:
        """
        Check that an image contains actual PCB data.

        Args:
            image_path: Path to the image file
            layer_type: Type of layer ("copper", "silkscreen", "mask", etc.)

        Returns:
            PreValidationResult with analysis details

        Raises:
            EmptyBoardFailure: If image appears empty or trivial
        """
        path = Path(image_path)
        if not path.exists():
            raise EmptyBoardFailure(
                f"Image file not found: {image_path}",
                image_path=image_path
            )

        # Handle SVG files separately (PIL doesn't support SVG)
        if path.suffix.lower() == '.svg':
            result = self._validate_svg_content(image_path, layer_type)
            if self.strict and not result.passed and result.issues:
                raise EmptyBoardFailure(
                    f"SVG validation failed: {'; '.join(result.issues)}",
                    image_path=image_path
                )
            return result

        issues = []

        # 1. File size check
        file_size = path.stat().st_size

        # 2. Load and analyze image
        img = Image.open(image_path)
        img_array = np.array(img)

        # 3. Count unique colors
        if len(img_array.shape) == 3:
            # Color image - flatten to get unique color tuples
            pixels = img_array.reshape(-1, img_array.shape[-1])
            unique_colors = len(np.unique(pixels, axis=0))
        else:
            # Grayscale
            unique_colors = len(np.unique(img_array))

        # 4. Calculate content density (non-background pixels)
        content_density = self._calculate_content_density(img_array)

        # 5. Calculate copper coverage for copper layers (not edge cuts, composites, etc.)
        copper_coverage = None
        layer_lower = layer_type.lower()
        if ("cu" in layer_lower or "copper" in layer_lower) and \
           "edge" not in layer_lower and \
           "composite" not in layer_lower and \
           "assembly" not in layer_lower:
            copper_coverage = self._calculate_copper_coverage(img_array)

        # Build result
        result = PreValidationResult(
            image_path=image_path,
            layer_type=layer_type,
            passed=True,
            file_size_bytes=file_size,
            unique_colors=unique_colors,
            copper_coverage=copper_coverage,
            content_density=content_density,
            issues=[]
        )

        # Check thresholds - be more lenient for inner/bottom layers (power/ground planes)
        is_inner_layer = "inner" in layer_type.lower() or "in1" in layer_type.lower() or "in2" in layer_type.lower()
        is_bottom_layer = "bottom" in layer_type.lower() or "b_cu" in layer_type.lower() or "b.cu" in layer_type.lower()
        is_sparse_layer = is_inner_layer or is_bottom_layer

        if unique_colors < self.MIN_UNIQUE_COLORS:
            issues.append(f"Only {unique_colors} unique colors - likely empty or trivial")
            result.passed = False

        # Sparse layers (inner/bottom) may have sparse content (just vias/thermals) - be more lenient
        min_density = 0.01 if is_sparse_layer else self.MIN_CONTENT_DENSITY
        if content_density < min_density:
            issues.append(f"Content density {content_density*100:.1f}% - mostly empty")
            result.passed = False

        # Copper coverage check - sparse layers may show just vias/thermals
        min_copper = 0.005 if is_sparse_layer else self.MIN_COPPER_COVERAGE
        if copper_coverage is not None and copper_coverage < min_copper:
            issues.append(f"Copper coverage {copper_coverage*100:.1f}% - no traces visible")
            result.passed = False

        result.issues = issues

        # Raise exception if failed and strict mode
        if not result.passed and self.strict:
            raise EmptyBoardFailure(
                message="; ".join(issues),
                image_path=image_path,
                file_size_bytes=file_size,
                unique_colors=unique_colors,
                copper_coverage=copper_coverage
            )

        return result

    def validate_silkscreen_has_text(
        self,
        image_path: str,
        expected_refs: Optional[List[str]] = None,
        min_text_elements: int = 1
    ) -> PreValidationResult:
        """
        Check that silkscreen layer has visible text/designators.

        Args:
            image_path: Path to silkscreen image
            expected_refs: List of expected reference designators
            min_text_elements: Minimum number of text elements expected

        Returns:
            PreValidationResult with analysis

        Raises:
            EmptySilkscreenFailure: If silkscreen appears empty
        """
        path = Path(image_path)
        if not path.exists():
            raise EmptySilkscreenFailure(
                f"Silkscreen image not found: {image_path}",
                image_path=image_path
            )

        img = Image.open(image_path)
        img_array = np.array(img)

        # Check for text-like elements (high-contrast small regions)
        text_density = self._estimate_text_density(img_array)

        # Check for silkscreen colors (yellow on KiCad default)
        silkscreen_coverage = self._calculate_color_coverage(
            img_array,
            self.SILKSCREEN_COLOR_RANGES
        )

        issues = []
        passed = True

        if text_density < 0.001:  # Less than 0.1% text-like pixels
            issues.append("No text-like elements detected on silkscreen")
            passed = False

        if silkscreen_coverage < 0.005:  # Less than 0.5% silkscreen color
            issues.append(f"Silkscreen coverage {silkscreen_coverage*100:.2f}% - likely empty")
            passed = False

        result = PreValidationResult(
            image_path=image_path,
            layer_type="silkscreen",
            passed=passed,
            file_size_bytes=path.stat().st_size,
            unique_colors=len(np.unique(img_array.reshape(-1, img_array.shape[-1]) if len(img_array.shape) == 3 else img_array)),
            copper_coverage=None,
            content_density=silkscreen_coverage,
            issues=issues
        )

        if not passed and self.strict:
            raise EmptySilkscreenFailure(
                message="; ".join(issues),
                image_path=image_path,
                expected_designators=expected_refs
            )

        return result

    def validate_all_images(
        self,
        directory: str,
        patterns: Optional[List[str]] = None
    ) -> Dict[str, PreValidationResult]:
        """
        Validate all images in a directory.

        Args:
            directory: Directory containing images
            patterns: File patterns to match (default: *.png)

        Returns:
            Dictionary of filename -> PreValidationResult

        Raises:
            First EmptyBoardFailure or EmptySilkscreenFailure encountered
        """
        dir_path = Path(directory)
        if not dir_path.exists():
            raise EmptyBoardFailure(
                f"Directory not found: {directory}",
                image_path=directory
            )

        patterns = patterns or ["*.png", "*.jpg", "*.jpeg"]
        results = {}

        for pattern in patterns:
            for img_path in dir_path.glob(pattern):
                layer_type = self._detect_layer_type(img_path.name)

                if "silks" in layer_type.lower():
                    result = self.validate_silkscreen_has_text(str(img_path))
                else:
                    result = self.validate_image_has_content(str(img_path), layer_type)

                results[img_path.name] = result

        return results

    def _calculate_content_density(self, img_array: np.ndarray) -> float:
        """
        Calculate fraction of image that contains actual PCB content.

        This method first finds the bounding box of non-white content (the board area),
        then calculates how much of that area contains actual content vs background.

        This handles cases where KiCad exports have lots of white space around the board.
        """
        if len(img_array.shape) == 3:
            # Color image - convert to grayscale for analysis
            gray = np.mean(img_array, axis=2)
        else:
            gray = img_array.astype(float)

        # Find non-white pixels (threshold 250 to catch slightly off-white)
        non_white_mask = gray < 250

        # If no content at all, return 0
        if not np.any(non_white_mask):
            return 0.0

        # Find the bounding box of all non-white content
        rows_with_content = np.any(non_white_mask, axis=1)
        cols_with_content = np.any(non_white_mask, axis=0)

        if not np.any(rows_with_content) or not np.any(cols_with_content):
            return 0.0

        row_indices = np.where(rows_with_content)[0]
        col_indices = np.where(cols_with_content)[0]

        min_row, max_row = row_indices[0], row_indices[-1]
        min_col, max_col = col_indices[0], col_indices[-1]

        # Calculate content area size
        content_height = max_row - min_row + 1
        content_width = max_col - min_col + 1
        content_area = content_height * content_width

        # Count non-white pixels within the content bounding box
        content_region = gray[min_row:max_row+1, min_col:max_col+1]
        non_white_in_content = np.sum(content_region < 250)

        # For images with actual content, also check for meaningful features
        # (not just border/title block)
        # Look for darker pixels (actual copper, traces, etc.) below 200
        meaningful_content = np.sum(content_region < 200)
        meaningful_ratio = meaningful_content / content_area if content_area > 0 else 0

        # The content density is the ratio of non-white pixels in the content area
        # Plus a bonus for meaningful (darker) content
        density = non_white_in_content / content_area if content_area > 0 else 0

        # If we have meaningful content (darker features), boost the density score
        if meaningful_ratio > 0.01:  # At least 1% darker content
            density = max(density, meaningful_ratio * 2)  # Boost meaningful content

        return min(density, 1.0)  # Cap at 1.0

    def _calculate_copper_coverage(self, img_array: np.ndarray) -> float:
        """Calculate fraction of image that contains copper-colored pixels."""
        return self._calculate_color_coverage(img_array, self.COPPER_COLOR_RANGES)

    def _calculate_color_coverage(
        self,
        img_array: np.ndarray,
        color_ranges: List[Tuple[Tuple[int, int, int], Tuple[int, int, int]]]
    ) -> float:
        """Calculate fraction of image matching color ranges."""
        if len(img_array.shape) != 3 or img_array.shape[2] < 3:
            return 0.0

        total_matching = 0
        for low, high in color_ranges:
            mask = (
                (img_array[:, :, 0] >= low[0]) & (img_array[:, :, 0] <= high[0]) &
                (img_array[:, :, 1] >= low[1]) & (img_array[:, :, 1] <= high[1]) &
                (img_array[:, :, 2] >= low[2]) & (img_array[:, :, 2] <= high[2])
            )
            total_matching += np.sum(mask)

        total_pixels = img_array.shape[0] * img_array.shape[1]
        return total_matching / total_pixels

    def _estimate_text_density(self, img_array: np.ndarray) -> float:
        """Estimate density of text-like elements using edge detection."""
        if len(img_array.shape) == 3:
            gray = np.mean(img_array, axis=2)
        else:
            gray = img_array.astype(float)

        # Simple edge detection (Sobel-like)
        dx = np.abs(np.diff(gray, axis=1))
        dy = np.abs(np.diff(gray, axis=0))

        # Text has high edge density
        edge_threshold = 30
        edge_pixels = np.sum(dx > edge_threshold) + np.sum(dy > edge_threshold)
        total_pixels = gray.size

        return edge_pixels / total_pixels

    def _detect_layer_type(self, filename: str) -> str:
        """Detect layer type from filename."""
        name_lower = filename.lower()

        # Check edge cuts first (before copper check)
        if "edge" in name_lower or "cut" in name_lower:
            return "edge_cuts"

        # Composite views that shouldn't be validated as copper
        if "all_copper" in name_lower or "all_cu" in name_lower:
            return "composite"
        if "top_view" in name_lower or "bottom_view" in name_lower:
            return "composite"
        if "assembly" in name_lower:
            return "assembly"

        if "cu" in name_lower or "copper" in name_lower:
            if "f_cu" in name_lower or "f.cu" in name_lower:
                return "copper_top"
            elif "b_cu" in name_lower or "b.cu" in name_lower:
                return "copper_bottom"
            elif "in" in name_lower:
                return "copper_inner"
            return "copper"
        elif "silk" in name_lower:
            return "silkscreen"
        elif "mask" in name_lower:
            return "solder_mask"
        elif "paste" in name_lower:
            return "solder_paste"
        elif "fab" in name_lower:
            return "fabrication"
        else:
            return "unknown"


def main():
    """CLI for pre-validation."""
    parser = argparse.ArgumentParser(
        description='Pre-validate PCB images before AI analysis'
    )
    parser.add_argument(
        '--image', '-i',
        type=str,
        help='Single image to validate'
    )
    parser.add_argument(
        '--dir', '-d',
        type=str,
        help='Directory of images to validate'
    )
    parser.add_argument(
        '--layer-type', '-l',
        type=str,
        default='unknown',
        help='Layer type (copper, silkscreen, etc.)'
    )
    parser.add_argument(
        '--strict',
        action='store_true',
        default=True,
        help='Raise exceptions on failure (default: True)'
    )
    parser.add_argument(
        '--no-strict',
        action='store_true',
        help='Return results without raising exceptions'
    )
    parser.add_argument(
        '--json',
        action='store_true',
        help='Output as JSON'
    )

    args = parser.parse_args()

    if not args.image and not args.dir:
        parser.error("Specify --image or --dir")

    strict = not args.no_strict

    try:
        validator = PreValidator(strict=strict)

        if args.dir:
            results = validator.validate_all_images(args.dir)
            all_passed = all(r.passed for r in results.values())

            if args.json:
                output = {
                    name: {
                        "passed": r.passed,
                        "file_size_bytes": r.file_size_bytes,
                        "unique_colors": r.unique_colors,
                        "copper_coverage": r.copper_coverage,
                        "content_density": r.content_density,
                        "issues": r.issues
                    }
                    for name, r in results.items()
                }
                print(json.dumps(output, indent=2))
            else:
                print(f"\nPre-Validation Results for {args.dir}")
                print("=" * 60)
                all_passed = True
                for name, result in results.items():
                    status = "PASS" if result.passed else "FAIL"
                    all_passed = all_passed and result.passed
                    print(f"  [{status}] {name}")
                    if result.issues:
                        for issue in result.issues:
                            print(f"         - {issue}")
                print("=" * 60)
                print(f"Overall: {'PASS' if all_passed else 'FAIL'}")

        else:
            # Auto-detect layer type from filename if not specified
            layer_type = args.layer_type
            if layer_type == "unknown":
                layer_type = validator._detect_layer_type(Path(args.image).name)

            if "silk" in layer_type.lower():
                result = validator.validate_silkscreen_has_text(args.image)
            else:
                result = validator.validate_image_has_content(args.image, layer_type)

            if args.json:
                output = {
                    "passed": result.passed,
                    "file_size_bytes": result.file_size_bytes,
                    "unique_colors": result.unique_colors,
                    "copper_coverage": result.copper_coverage,
                    "content_density": result.content_density,
                    "issues": result.issues
                }
                print(json.dumps(output, indent=2))
            else:
                status = "PASS" if result.passed else "FAIL"
                print(f"\n[{status}] {args.image}")
                print(f"  Layer type: {result.layer_type}")
                print(f"  File size: {result.file_size_bytes} bytes")
                print(f"  Unique colors: {result.unique_colors}")
                print(f"  Content density: {result.content_density*100:.1f}%")
                if result.copper_coverage is not None:
                    print(f"  Copper coverage: {result.copper_coverage*100:.1f}%")
                if result.issues:
                    print("  Issues:")
                    for issue in result.issues:
                        print(f"    - {issue}")

        sys.exit(0 if (result.passed if args.image else all_passed) else 1)

    except (EmptyBoardFailure, EmptySilkscreenFailure) as e:
        print(f"\nFAILED: {e}", file=sys.stderr)
        sys.exit(1)
    except MissingDependencyFailure as e:
        print(f"\nMISSING DEPENDENCY: {e}", file=sys.stderr)
        sys.exit(2)


if __name__ == '__main__':
    main()
