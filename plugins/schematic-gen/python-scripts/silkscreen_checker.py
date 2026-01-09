#!/usr/bin/env python3
"""
Silkscreen Checker - PCB Silkscreen Layer Validation

Validates PCB silkscreen layer for completeness and quality using
Claude Opus 4.5 via OpenRouter API (not direct Anthropic SDK):
- Reference designator presence
- Polarity markings for diodes/capacitors
- Pin 1 indicators for ICs
- Text readability and placement
- Assembly information completeness

Environment Variables:
    OPENROUTER_API_KEY: Required - Your OpenRouter API key

Usage:
    python silkscreen_checker.py --image F_Silkscreen.png
    python silkscreen_checker.py --image F_Silkscreen.png --components components.json
    python silkscreen_checker.py --pcb board.kicad_pcb --extract-expected
"""

import argparse
import base64
import json
import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Any, List, Optional, Set

from validation_exceptions import (
    ValidationFailure,
    EmptySilkscreenFailure,
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
class ComponentInfo:
    """Information about a component for silkscreen validation."""
    reference: str
    value: str
    footprint: str
    is_polarized: bool = False
    has_pin1: bool = False
    component_type: str = "generic"


@dataclass
class SilkscreenCheckResult:
    """Result of silkscreen validation."""
    overall_score: float
    passed: bool
    designator_score: float
    polarity_score: float
    readability_score: float
    completeness_score: float
    found_designators: List[str]
    missing_designators: List[str]
    missing_polarity_marks: List[str]
    missing_pin1_marks: List[str]
    issues: List[str]
    warnings: List[str]
    recommendations: List[str]
    is_empty: bool


class SilkscreenChecker:
    """
    Validates PCB silkscreen layer for completeness and quality.

    Raises exceptions on validation failures - no silent passes.
    """

    def __init__(self, api_key: Optional[str] = None, strict: bool = True):
        """
        Initialize silkscreen checker.

        Args:
            api_key: OpenRouter API key (defaults to OPENROUTER_API_KEY env var)
            strict: If True, raise exceptions on failures

        Raises:
            MissingDependencyFailure: If dependencies are missing
        """
        if not HAS_OPENROUTER:
            raise MissingDependencyFailure(
                "openrouter_client module required for silkscreen validation",
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
        self.strict = strict

    def _load_image(self, image_path: str) -> tuple:
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

    def extract_expected_components(self, pcb_path: str) -> List[ComponentInfo]:
        """
        Extract expected components from a KiCad PCB file.

        Args:
            pcb_path: Path to .kicad_pcb file

        Returns:
            List of ComponentInfo objects
        """
        components = []

        with open(pcb_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()

        # Find all footprint blocks
        footprint_pattern = r'\(footprint\s+"([^"]+)"[^)]*\(at\s+[\d.-]+\s+[\d.-]+[^)]*\)'

        # More detailed parsing
        lines = content.split('\n')
        current_fp = None
        current_ref = None
        current_value = None

        for line in lines:
            # Start of footprint
            if '(footprint "' in line:
                match = re.search(r'\(footprint\s+"([^"]+)"', line)
                if match:
                    current_fp = match.group(1)

            # Reference designator
            if '(fp_text reference' in line or '(property "Reference"' in line:
                match = re.search(r'"([A-Z]+\d+)"', line)
                if match:
                    current_ref = match.group(1)

            # Value
            if '(fp_text value' in line or '(property "Value"' in line:
                match = re.search(r'"([^"]+)"', line)
                if match:
                    current_value = match.group(1)

            # End of footprint block - save component
            if current_ref and current_fp:
                comp_type = self._determine_component_type(current_ref, current_fp)
                is_polarized = comp_type in ['diode', 'led', 'electrolytic_cap', 'tantalum_cap']
                has_pin1 = comp_type in ['ic', 'connector', 'transistor']

                components.append(ComponentInfo(
                    reference=current_ref,
                    value=current_value or "",
                    footprint=current_fp,
                    is_polarized=is_polarized,
                    has_pin1=has_pin1,
                    component_type=comp_type
                ))

                current_ref = None
                current_value = None

        return components

    def _determine_component_type(self, reference: str, footprint: str) -> str:
        """Determine component type from reference and footprint."""
        ref_prefix = re.match(r'^([A-Z]+)', reference)
        if ref_prefix:
            prefix = ref_prefix.group(1)
            type_map = {
                'R': 'resistor',
                'C': 'capacitor',
                'L': 'inductor',
                'D': 'diode',
                'LED': 'led',
                'Q': 'transistor',
                'U': 'ic',
                'J': 'connector',
                'P': 'connector',
                'SW': 'switch',
                'F': 'fuse',
                'FB': 'ferrite_bead',
                'Y': 'crystal',
                'X': 'crystal'
            }
            if prefix in type_map:
                return type_map[prefix]

        # Check footprint for clues
        fp_lower = footprint.lower()
        if 'led' in fp_lower:
            return 'led'
        elif 'diode' in fp_lower:
            return 'diode'
        elif 'cap' in fp_lower and ('elec' in fp_lower or 'tant' in fp_lower or 'polar' in fp_lower):
            return 'electrolytic_cap'
        elif 'sot' in fp_lower or 'to-' in fp_lower:
            return 'transistor'
        elif 'qfp' in fp_lower or 'qfn' in fp_lower or 'bga' in fp_lower or 'soic' in fp_lower:
            return 'ic'
        elif 'conn' in fp_lower or 'header' in fp_lower or 'pin' in fp_lower:
            return 'connector'

        return 'generic'

    def check_silkscreen(
        self,
        image_path: str,
        expected_components: Optional[List[ComponentInfo]] = None
    ) -> SilkscreenCheckResult:
        """
        Validate silkscreen layer image.

        Args:
            image_path: Path to silkscreen layer image
            expected_components: Optional list of expected components

        Returns:
            SilkscreenCheckResult with detailed validation results
        """
        image_data, media_type = self._load_image(image_path)

        # Build component context if provided
        component_context = ""
        if expected_components:
            refs = [c.reference for c in expected_components]
            polarized = [c.reference for c in expected_components if c.is_polarized]
            pin1_needed = [c.reference for c in expected_components if c.has_pin1]

            component_context = f"""
Expected Components:
- All reference designators: {', '.join(refs)}
- Polarized components needing polarity marks: {', '.join(polarized) if polarized else 'None'}
- Components needing pin 1 indicators: {', '.join(pin1_needed) if pin1_needed else 'None'}
"""

        system_prompt = """You are a PCB silkscreen quality inspector.
Your job is to verify that the silkscreen layer has all required elements.

CRITICAL REQUIREMENTS:
1. Every component MUST have a visible reference designator
2. Polarized parts (diodes, LEDs, electrolytic caps) MUST have polarity marks
3. ICs and connectors MUST have pin 1 indicators
4. Text must be readable (proper size, not overlapping)

An EMPTY or nearly-empty silkscreen is a CRITICAL FAILURE."""

        user_prompt = f"""Analyze this PCB silkscreen layer image.

{component_context}

Provide detailed JSON analysis:
{{
    "is_empty": true/false (true if silkscreen is mostly empty - CRITICAL FAILURE),
    "found_designators": ["list of all designators visible in image"],
    "missing_designators": ["list of designators NOT found but expected"],
    "polarity_marks_found": ["list of components with visible polarity marks"],
    "missing_polarity_marks": ["polarized components without marks"],
    "pin1_marks_found": ["list of components with pin 1 indicators"],
    "missing_pin1_marks": ["ICs/connectors without pin 1 marks"],
    "text_readability": 1-10 (10=all clear, 1=illegible),
    "text_size_adequate": true/false,
    "overlapping_text": ["list of overlapping text instances"],
    "silkscreen_over_pads": ["list of silkscreen over pad instances"],
    "board_info_present": true/false (board name, revision, etc.),
    "overall_score": 1-10,
    "passed": true/false (overall >= 7 AND not empty),
    "designator_score": 1-10,
    "polarity_score": 1-10,
    "readability_score": 1-10,
    "completeness_score": 1-10,
    "issues": ["critical issues"],
    "warnings": ["warnings"],
    "recommendations": ["improvements"]
}}

Be thorough - examine every part of the image."""

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

        check_result = SilkscreenCheckResult(
            overall_score=result.get('overall_score', 0),
            passed=result.get('passed', False),
            designator_score=result.get('designator_score', 0),
            polarity_score=result.get('polarity_score', 0),
            readability_score=result.get('readability_score', 0),
            completeness_score=result.get('completeness_score', 0),
            found_designators=result.get('found_designators', []),
            missing_designators=result.get('missing_designators', []),
            missing_polarity_marks=result.get('missing_polarity_marks', []),
            missing_pin1_marks=result.get('missing_pin1_marks', []),
            issues=result.get('issues', []),
            warnings=result.get('warnings', []),
            recommendations=result.get('recommendations', []),
            is_empty=result.get('is_empty', False)
        )

        # STRICT MODE: Raise exception on empty silkscreen
        if self.strict and check_result.is_empty:
            expected_refs = [c.reference for c in expected_components] if expected_components else None
            raise EmptySilkscreenFailure(
                message="Silkscreen layer is empty or nearly empty - CRITICAL FAILURE",
                image_path=image_path,
                expected_designators=expected_refs,
                found_designators=check_result.found_designators
            )

        # STRICT MODE: Raise exception on significant missing designators
        if self.strict and check_result.missing_designators:
            missing_count = len(check_result.missing_designators)
            if missing_count > 0 and expected_components:
                expected_count = len(expected_components)
                if missing_count / expected_count > 0.5:  # More than 50% missing
                    raise EmptySilkscreenFailure(
                        message=f"More than 50% of designators missing ({missing_count}/{expected_count})",
                        image_path=image_path,
                        expected_designators=[c.reference for c in expected_components],
                        found_designators=check_result.found_designators
                    )

        return check_result

    def generate_report(self, result: SilkscreenCheckResult) -> str:
        """Generate a formatted report from check results."""
        lines = [
            "=" * 60,
            "SILKSCREEN VALIDATION REPORT",
            "=" * 60,
            "",
            f"Overall Score: {result.overall_score}/10",
            f"Status: {'PASS' if result.passed else 'FAIL'}",
            f"Empty Silkscreen: {'YES - CRITICAL FAILURE' if result.is_empty else 'No'}",
            "",
            "-" * 40,
            "SCORES",
            "-" * 40,
            f"  Designator Coverage: {result.designator_score}/10",
            f"  Polarity Markings: {result.polarity_score}/10",
            f"  Readability: {result.readability_score}/10",
            f"  Completeness: {result.completeness_score}/10",
            ""
        ]

        if result.found_designators:
            lines.extend([
                "-" * 40,
                f"FOUND DESIGNATORS ({len(result.found_designators)})",
                "-" * 40
            ])
            for ref in result.found_designators[:20]:
                lines.append(f"  {ref}")
            if len(result.found_designators) > 20:
                lines.append(f"  ... and {len(result.found_designators) - 20} more")
            lines.append("")

        if result.missing_designators:
            lines.extend([
                "-" * 40,
                f"MISSING DESIGNATORS ({len(result.missing_designators)}) - MUST FIX",
                "-" * 40
            ])
            for ref in result.missing_designators:
                lines.append(f"  {ref}")
            lines.append("")

        if result.missing_polarity_marks:
            lines.extend([
                "-" * 40,
                "MISSING POLARITY MARKS - MUST FIX",
                "-" * 40
            ])
            for ref in result.missing_polarity_marks:
                lines.append(f"  {ref}")
            lines.append("")

        if result.missing_pin1_marks:
            lines.extend([
                "-" * 40,
                "MISSING PIN 1 INDICATORS - MUST FIX",
                "-" * 40
            ])
            for ref in result.missing_pin1_marks:
                lines.append(f"  {ref}")
            lines.append("")

        if result.issues:
            lines.extend([
                "-" * 40,
                "CRITICAL ISSUES",
                "-" * 40
            ])
            for issue in result.issues:
                lines.append(f"  - {issue}")
            lines.append("")

        if result.warnings:
            lines.extend([
                "-" * 40,
                "WARNINGS",
                "-" * 40
            ])
            for warning in result.warnings:
                lines.append(f"  - {warning}")
            lines.append("")

        if result.recommendations:
            lines.extend([
                "-" * 40,
                "RECOMMENDATIONS",
                "-" * 40
            ])
            for rec in result.recommendations:
                lines.append(f"  - {rec}")
            lines.append("")

        return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description='Silkscreen Checker - PCB Silkscreen Layer Validation'
    )
    parser.add_argument(
        '--image', '-i',
        type=str,
        help='Silkscreen layer image to validate'
    )
    parser.add_argument(
        '--pcb', '-p',
        type=str,
        help='KiCad PCB file to extract expected components'
    )
    parser.add_argument(
        '--components', '-c',
        type=str,
        help='JSON file with expected components'
    )
    parser.add_argument(
        '--extract-expected',
        action='store_true',
        help='Extract expected components from PCB and save to JSON'
    )
    parser.add_argument(
        '--json',
        action='store_true',
        help='Output as JSON'
    )

    args = parser.parse_args()

    if not args.image and not (args.pcb and args.extract_expected):
        parser.error("Specify --image or --pcb with --extract-expected")

    if not HAS_OPENROUTER:
        print("ERROR: openrouter_client module not found")
        sys.exit(1)

    if not os.environ.get('OPENROUTER_API_KEY'):
        print("ERROR: OPENROUTER_API_KEY not set")
        print("Get your API key from https://openrouter.ai/keys")
        sys.exit(1)

    checker = SilkscreenChecker()

    # Extract expected components if requested
    if args.pcb and args.extract_expected:
        components = checker.extract_expected_components(args.pcb)
        output = [
            {
                "reference": c.reference,
                "value": c.value,
                "footprint": c.footprint,
                "is_polarized": c.is_polarized,
                "has_pin1": c.has_pin1,
                "component_type": c.component_type
            }
            for c in components
        ]
        output_path = Path(args.pcb).stem + "_components.json"
        with open(output_path, 'w') as f:
            json.dump(output, f, indent=2)
        print(f"Extracted {len(components)} components to {output_path}")
        return

    # Load expected components if provided
    expected = None
    if args.components:
        with open(args.components, 'r') as f:
            comp_data = json.load(f)
            expected = [
                ComponentInfo(
                    reference=c['reference'],
                    value=c.get('value', ''),
                    footprint=c.get('footprint', ''),
                    is_polarized=c.get('is_polarized', False),
                    has_pin1=c.get('has_pin1', False),
                    component_type=c.get('component_type', 'generic')
                )
                for c in comp_data
            ]
    elif args.pcb:
        expected = checker.extract_expected_components(args.pcb)

    # Run validation
    result = checker.check_silkscreen(args.image, expected)

    if args.json:
        output = {
            "overall_score": result.overall_score,
            "passed": result.passed,
            "is_empty": result.is_empty,
            "scores": {
                "designator": result.designator_score,
                "polarity": result.polarity_score,
                "readability": result.readability_score,
                "completeness": result.completeness_score
            },
            "found_designators": result.found_designators,
            "missing_designators": result.missing_designators,
            "missing_polarity_marks": result.missing_polarity_marks,
            "missing_pin1_marks": result.missing_pin1_marks,
            "issues": result.issues,
            "warnings": result.warnings,
            "recommendations": result.recommendations
        }
        print(json.dumps(output, indent=2))
    else:
        print(checker.generate_report(result))


if __name__ == '__main__':
    main()
