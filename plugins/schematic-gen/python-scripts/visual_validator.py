#!/usr/bin/env python3
"""
Visual Validation System for EE Design Outputs

Multi-persona visual validation using Claude Opus 4.5 vision capabilities
via OpenRouter API (not direct Anthropic SDK).

Validates PCB layouts, schematics, and generated images for professional quality.

Expert Personas:
1. PCB Aesthetics Expert - Visual appearance and professional look
2. Silkscreen Specialist - Designator placement and readability
3. Layout Composition Expert - Component grouping and signal flow
4. Manufacturing Visual Inspector - DFM visual checks
5. Human Perception Analyst - Cognitive load and visual hierarchy

Environment Variables:
    OPENROUTER_API_KEY: Required - Your OpenRouter API key

Usage:
    python visual_validator.py --images *.png --json
    python visual_validator.py --image F_Cu.png --persona PCB_AESTHETICS
    python visual_validator.py --dir ./output/layer_images --full-report
"""

import argparse
import base64
import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Any, List, Optional
from enum import Enum

from validation_exceptions import (
    ValidationFailure,
    QualityThresholdFailure,
    MissingDependencyFailure,
    ValidationIssue,
    ValidationSeverity
)

# Use OpenRouter client instead of direct Anthropic SDK
try:
    from openrouter_client import OpenRouterClient, OpenRouterAnthropicAdapter, get_openrouter_client
    HAS_OPENROUTER = True
except ImportError:
    HAS_OPENROUTER = False


class PersonaType(Enum):
    PCB_AESTHETICS = "pcb_aesthetics"
    SILKSCREEN = "silkscreen"
    LAYOUT_COMPOSITION = "layout_composition"
    MANUFACTURING = "manufacturing"
    HUMAN_PERCEPTION = "human_perception"


@dataclass
class VisualPersona:
    """Expert persona for visual design review."""
    name: str
    persona_type: PersonaType
    focus: str
    standards: List[str]
    system_prompt: str
    scoring_criteria: Dict[str, str]


# Define visual validation personas
VISUAL_PERSONAS = [
    VisualPersona(
        name="PCB Aesthetics Expert",
        persona_type=PersonaType.PCB_AESTHETICS,
        focus="Visual appearance, professional look, routing cleanliness",
        standards=["IPC-2221", "Professional PCB design aesthetics", "Award-winning layout principles"],
        system_prompt="""You are a PCB Aesthetics Expert with decades of experience designing award-winning circuit boards.

Your evaluation focuses on:
1. ROUTING QUALITY:
   - All traces should use 45° angles (no 90° angles except at pads)
   - Traces should be clean, organized, not chaotic
   - No unnecessary crossings or overlaps
   - Consistent trace widths for same net classes

2. VISUAL ORGANIZATION:
   - Components should be logically grouped
   - Signal flow should be apparent (left-to-right or top-to-bottom)
   - No "spaghetti" routing
   - Clean separation between analog/digital/power sections

3. PROFESSIONAL APPEARANCE:
   - Would this board look good in a product showcase?
   - Is the layout elegant or chaotic?
   - Does it show thoughtful engineering or rushed work?

4. SYMMETRY AND BALANCE:
   - Are similar components aligned?
   - Is the layout balanced visually?
   - Are there large empty areas that waste space?

Score each aspect 1-10 and provide specific issues found.
Be brutally honest - amateur designs MUST be flagged.""",
        scoring_criteria={
            "routing_angles": "45° angles used consistently (10) vs 90° angles everywhere (1)",
            "visual_organization": "Clean and logical (10) vs chaotic spaghetti (1)",
            "professional_look": "Award-winning quality (10) vs amateur/hobby (1)",
            "balance": "Well-balanced layout (10) vs components in one corner (1)"
        }
    ),

    VisualPersona(
        name="Silkscreen Specialist",
        persona_type=PersonaType.SILKSCREEN,
        focus="Silkscreen layer quality, designator visibility, assembly guides",
        standards=["IPC-7351", "Assembly documentation standards", "Silkscreen best practices"],
        system_prompt="""You are a Silkscreen Specialist focused on PCB silkscreen/legend quality.

Your evaluation focuses on:
1. REFERENCE DESIGNATORS:
   - ALL components MUST have visible designators (R1, C1, U1, etc.)
   - Designators must be readable (proper size, not too small)
   - Designators must NOT overlap with pads or vias
   - Designators should be near their components

2. POLARITY MARKINGS:
   - Diodes must have cathode indicators
   - Electrolytic capacitors must have polarity marks
   - ICs must have pin 1 indicators
   - Connectors must have pin 1 markings

3. ASSEMBLY INFORMATION:
   - Board name/revision visible
   - Date code or part number area
   - Test point labels if applicable
   - Connector labels (J1, J2, etc.)

4. READABILITY:
   - Text size minimum 0.8mm height for production
   - No overlapping text
   - Consistent orientation (readable from one direction)
   - High contrast (will print clearly)

If the silkscreen is EMPTY or nearly empty, this is a CRITICAL FAILURE.
Score each aspect 1-10 and list all missing elements.""",
        scoring_criteria={
            "designator_completeness": "All designators present (10) vs none/few (1)",
            "polarity_markings": "All polarized parts marked (10) vs none marked (1)",
            "readability": "Clear and readable (10) vs overlapping/tiny (1)",
            "assembly_info": "Complete assembly guides (10) vs no info (1)"
        }
    ),

    VisualPersona(
        name="Layout Composition Expert",
        persona_type=PersonaType.LAYOUT_COMPOSITION,
        focus="Overall board composition, component placement, thermal design",
        standards=["PCB layout best practices", "Thermal management guidelines", "Signal integrity principles"],
        system_prompt="""You are a Layout Composition Expert focusing on overall PCB organization.

Your evaluation focuses on:
1. COMPONENT PLACEMENT:
   - Logical grouping (power section, analog, digital, I/O)
   - Short critical signal paths
   - Heat-generating components distributed (not clustered)
   - Tall components not blocking airflow

2. SIGNAL FLOW:
   - Clear input-to-output flow
   - Minimal signal crossing between sections
   - Ground return paths considered
   - Power distribution visible

3. SPACE UTILIZATION:
   - Board space used efficiently
   - No large empty areas (unless for thermal)
   - Components not all crammed in one corner
   - Appropriate margins at edges

4. THERMAL CONSIDERATIONS:
   - Power components have thermal relief
   - Thermal vias visible under heat sources
   - No thermal hot spots (components too close)
   - Copper pour for heat spreading

5. MECHANICAL:
   - Mounting holes present and properly located
   - Keep-out areas respected
   - Connector placement accessible

Score each aspect 1-10 and describe composition issues.""",
        scoring_criteria={
            "component_grouping": "Logical functional groups (10) vs random placement (1)",
            "signal_flow": "Clear flow visible (10) vs chaotic (1)",
            "space_utilization": "Efficient use of space (10) vs poor utilization (1)",
            "thermal_design": "Good thermal management (10) vs no consideration (1)"
        }
    ),

    VisualPersona(
        name="Manufacturing Visual Inspector",
        persona_type=PersonaType.MANUFACTURING,
        focus="DFM visual checks, fiducials, test points, panelization",
        standards=["IPC-A-600 Class 2/3", "DFM guidelines", "Assembly process requirements"],
        system_prompt="""You are a Manufacturing Visual Inspector with PCB fabrication expertise.

Your evaluation focuses on:
1. DFM MARKERS:
   - Fiducial markers present (at least 3 for pick-and-place)
   - Tooling holes if required
   - Panelization markers/rails visible
   - Layer identification markers

2. TEST POINTS:
   - Test points for critical signals
   - Ground test points
   - Test point accessibility (not under components)
   - Test point labels

3. ASSEMBLY FEATURES:
   - Stencil-friendly pad shapes
   - Component orientation indicators
   - Solder paste considerations
   - No tomb-stoning risks (balanced pads)

4. FABRICATION:
   - Minimum trace widths met
   - Minimum clearances visible
   - Via sizes appropriate
   - Edge clearances respected

5. QUALITY INDICATORS:
   - IPC class markings if applicable
   - UL/certification markings area
   - Serial number area
   - Inspection points

Score each aspect 1-10 and identify manufacturing risks.""",
        scoring_criteria={
            "fiducials": "Proper fiducials present (10) vs none (1)",
            "test_points": "Adequate test coverage (10) vs no test points (1)",
            "dfm_compliance": "Fully DFM compliant (10) vs multiple violations (1)",
            "assembly_ready": "Ready for production (10) vs needs rework (1)"
        }
    ),

    VisualPersona(
        name="Human Perception Analyst",
        persona_type=PersonaType.HUMAN_PERCEPTION,
        focus="How humans perceive the design, cognitive load, visual hierarchy",
        standards=["Visual design principles", "Cognitive psychology", "UX principles applied to hardware"],
        system_prompt="""You are a Human Perception Analyst applying cognitive science to PCB review.

Your evaluation focuses on:
1. FIRST IMPRESSION:
   - What does someone think when first seeing this board?
   - Does it look professional or amateur?
   - Does it inspire confidence or concern?
   - Would you trust this in a critical application?

2. COGNITIVE LOAD:
   - Can you quickly understand the board layout?
   - Are sections clearly delineated?
   - Is there visual hierarchy (important things stand out)?
   - Or is it overwhelming and confusing?

3. INTUITIVE DESIGN:
   - Can you tell where power comes in?
   - Can you identify the main processor/controller?
   - Are I/O connectors logically placed?
   - Does the design "make sense"?

4. ERROR PRONENESS:
   - Are there areas that could cause assembly errors?
   - Could connectors be plugged in wrong?
   - Are polarized parts clearly marked?
   - Is pin 1 obvious?

5. MAINTAINABILITY:
   - Can components be accessed for rework?
   - Are test points reachable?
   - Is the design debuggable?
   - Would a technician appreciate or curse this board?

Score based on human factors, not just technical correctness.""",
        scoring_criteria={
            "first_impression": "Impressive/professional (10) vs concerning/amateur (1)",
            "cognitive_clarity": "Instantly understandable (10) vs confusing (1)",
            "intuitive_design": "Self-explanatory layout (10) vs cryptic (1)",
            "maintainability": "Easy to debug/repair (10) vs nightmare (1)"
        }
    ),
]


def get_client():
    """Get OpenRouter client with Anthropic-compatible interface.

    Uses OpenRouter API with Claude Opus 4.5 model.

    Raises:
        MissingDependencyFailure: If openrouter_client not available or API key not set
    """
    if not HAS_OPENROUTER:
        raise MissingDependencyFailure(
            "openrouter_client module required for visual validation",
            dependency_name="openrouter_client",
            install_instructions="Ensure openrouter_client.py is in the same directory"
        )

    api_key = os.environ.get('OPENROUTER_API_KEY')
    if not api_key:
        raise MissingDependencyFailure(
            "OPENROUTER_API_KEY environment variable not set",
            dependency_name="OPENROUTER_API_KEY",
            install_instructions=(
                "1. Get API key from https://openrouter.ai/keys\n"
                "2. export OPENROUTER_API_KEY=your-key\n"
                "3. Re-run this script"
            )
        )
    # Return adapter that provides Anthropic-compatible interface
    return OpenRouterAnthropicAdapter(api_key=api_key)


def load_image_base64(image_path: str) -> str:
    """Load image and convert to base64."""
    with open(image_path, 'rb') as f:
        return base64.standard_b64encode(f.read()).decode('utf-8')


def get_image_media_type(image_path: str) -> str:
    """Determine media type from file extension."""
    ext = Path(image_path).suffix.lower()
    media_types = {
        '.png': 'image/png',
        '.jpg': 'image/jpeg',
        '.jpeg': 'image/jpeg',
        '.gif': 'image/gif',
        '.webp': 'image/webp'
    }
    return media_types.get(ext, 'image/png')


def validate_image_with_persona(
    client,
    persona: VisualPersona,
    image_path: str,
    image_context: str = ""
) -> Dict[str, Any]:
    """
    Validate an image using a specific expert persona.

    Args:
        client: Anthropic client
        persona: Expert persona to use
        image_path: Path to image file
        image_context: Additional context about the image

    Returns:
        Validation results with scores and issues
    """
    image_data = load_image_base64(image_path)
    media_type = get_image_media_type(image_path)
    image_name = Path(image_path).name

    user_content = [
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
            "text": f"""Analyze this PCB design image: {image_name}

{f"Context: {image_context}" if image_context else ""}

Provide your expert evaluation as JSON with:
1. "scores": Dictionary of scores (1-10) for each criterion
2. "overall_score": Single overall score (1-10)
3. "critical_issues": List of critical problems (must fix)
4. "warnings": List of warnings (should fix)
5. "recommendations": List of improvements
6. "positive_aspects": What's done well
7. "pass_fail": "PASS" if overall >= 7, "FAIL" otherwise
8. "summary": One paragraph summary

Scoring criteria:
{json.dumps(persona.scoring_criteria, indent=2)}

Be rigorous and honest. Amateur work must be clearly identified as such."""
        }
    ]

    response = client.messages.create(
        model="claude-opus-4-5-20251101",
        max_tokens=4096,
        system=persona.system_prompt,
        messages=[{"role": "user", "content": user_content}]
    )

    response_text = response.content[0].text

    # Parse JSON from response
    try:
        json_start = response_text.find('{')
        json_end = response_text.rfind('}') + 1
        if json_start >= 0 and json_end > json_start:
            result = json.loads(response_text[json_start:json_end])
        else:
            result = {"raw_response": response_text, "overall_score": 0, "pass_fail": "FAIL"}
    except json.JSONDecodeError:
        result = {"raw_response": response_text, "overall_score": 0, "pass_fail": "FAIL"}

    return {
        "persona": persona.name,
        "persona_type": persona.persona_type.value,
        "focus": persona.focus,
        "image": image_path,
        "result": result
    }


def validate_all_personas(
    client,
    image_path: str,
    personas: Optional[List[PersonaType]] = None
) -> Dict[str, Any]:
    """Run all or selected personas on an image."""
    if personas:
        selected = [p for p in VISUAL_PERSONAS if p.persona_type in personas]
    else:
        selected = VISUAL_PERSONAS

    results = []
    for persona in selected:
        print(f"  Running {persona.name}...")
        result = validate_image_with_persona(client, persona, image_path)
        results.append(result)

    # Calculate aggregate scores
    scores = []
    for r in results:
        if 'result' in r and 'overall_score' in r['result']:
            scores.append(r['result']['overall_score'])

    avg_score = sum(scores) / len(scores) if scores else 0
    all_pass = all(r['result'].get('pass_fail') == 'PASS' for r in results if 'result' in r)

    result = {
        "image": image_path,
        "persona_results": results,
        "aggregate_score": round(avg_score, 2),
        "all_personas_pass": all_pass,
        "overall_verdict": "PASS" if all_pass and avg_score >= 7.0 else "FAIL"
    }

    return result


def validate_all_personas_strict(
    client,
    image_path: str,
    quality_threshold: float = 7.0,
    personas: Optional[List[PersonaType]] = None
) -> Dict[str, Any]:
    """
    Run all or selected personas on an image with strict quality enforcement.

    Args:
        client: Anthropic client
        image_path: Path to image file
        quality_threshold: Minimum score required (1-10)
        personas: Optional list of specific personas to use

    Returns:
        Validation results dict

    Raises:
        QualityThresholdFailure: If aggregate score is below threshold
    """
    result = validate_all_personas(client, image_path, personas)

    if result['overall_verdict'] == 'FAIL' or result['aggregate_score'] < quality_threshold:
        # Collect all issues from all personas
        issues = []
        for pr in result.get('persona_results', []):
            r = pr.get('result', {})
            for issue in r.get('critical_issues', []):
                issues.append(ValidationIssue(
                    message=f"[{pr['persona']}] {issue}",
                    severity=ValidationSeverity.CRITICAL,
                    location=image_path
                ))
            for warning in r.get('warnings', []):
                issues.append(ValidationIssue(
                    message=f"[{pr['persona']}] {warning}",
                    severity=ValidationSeverity.WARNING,
                    location=image_path
                ))

        raise QualityThresholdFailure(
            message=f"Visual validation failed for {Path(image_path).name}",
            score=result['aggregate_score'],
            threshold=quality_threshold,
            issues=issues
        )

    return result


def validate_directory(
    client,
    directory: str,
    file_pattern: str = "*.png",
    personas: Optional[List[PersonaType]] = None
) -> Dict[str, Any]:
    """Validate all images in a directory."""
    dir_path = Path(directory)
    images = list(dir_path.glob(file_pattern))

    if not images:
        return {"error": f"No images found matching {file_pattern} in {directory}"}

    print(f"Found {len(images)} images to validate")

    all_results = []
    for img in images:
        print(f"\nValidating {img.name}...")
        result = validate_all_personas(client, str(img), personas)
        all_results.append(result)

    # Calculate overall statistics
    pass_count = sum(1 for r in all_results if r['overall_verdict'] == 'PASS')
    fail_count = len(all_results) - pass_count
    avg_score = sum(r['aggregate_score'] for r in all_results) / len(all_results) if all_results else 0

    return {
        "directory": directory,
        "total_images": len(all_results),
        "passed": pass_count,
        "failed": fail_count,
        "average_score": round(avg_score, 2),
        "overall_verdict": "PASS" if fail_count == 0 and avg_score >= 7.0 else "FAIL",
        "image_results": all_results
    }


def run_fallback_validation(image_path: str) -> Dict[str, Any]:
    """Fallback when API is unavailable - RAISES EXCEPTION.

    This function no longer returns a fallback dict.
    Missing dependencies are critical failures.

    Raises:
        MissingDependencyFailure: Always - fallback is not acceptable
    """
    raise MissingDependencyFailure(
        "Visual validation requires OPENROUTER_API_KEY",
        dependency_name="OPENROUTER_API_KEY",
        install_instructions=(
            "1. Get API key from https://openrouter.ai/keys\n"
            "2. export OPENROUTER_API_KEY=your-key\n"
            "3. Re-run this script"
        )
    )


def main():
    parser = argparse.ArgumentParser(
        description='Visual Validation System for EE Design Outputs'
    )
    parser.add_argument(
        '--image', '-i',
        type=str,
        help='Single image to validate'
    )
    parser.add_argument(
        '--images',
        type=str,
        nargs='+',
        help='Multiple images to validate'
    )
    parser.add_argument(
        '--dir', '-d',
        type=str,
        help='Directory of images to validate'
    )
    parser.add_argument(
        '--pattern',
        type=str,
        default='*.png',
        help='File pattern for directory validation (default: *.png)'
    )
    parser.add_argument(
        '--persona',
        type=str,
        choices=[p.value for p in PersonaType],
        help='Run only specific persona'
    )
    parser.add_argument(
        '--json',
        action='store_true',
        help='Output as JSON'
    )
    parser.add_argument(
        '--list-personas',
        action='store_true',
        help='List available personas'
    )

    args = parser.parse_args()

    if args.list_personas:
        print("\nAvailable Visual Validation Personas:")
        print("=" * 60)
        for p in VISUAL_PERSONAS:
            print(f"\n{p.name}")
            print(f"  Type: {p.persona_type.value}")
            print(f"  Focus: {p.focus}")
            print(f"  Standards: {', '.join(p.standards)}")
            print("  Scoring Criteria:")
            for k, v in p.scoring_criteria.items():
                print(f"    - {k}: {v}")
        return

    # No fallback mode - missing dependencies raise exceptions
    try:
        client = get_client()
    except MissingDependencyFailure as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(2)

    # Determine which personas to use
    personas = None
    if args.persona:
        personas = [PersonaType(args.persona)]

    results = []

    if args.dir:
        result = validate_directory(client, args.dir, args.pattern, personas)
        results.append(result)
    elif args.images:
        for img in args.images:
            result = validate_all_personas(client, img, personas)
            results.append(result)
    elif args.image:
        result = validate_all_personas(client, args.image, personas)
        results.append(result)
    else:
        parser.error("Specify --image, --images, or --dir")

    # Output results
    if args.json:
        print(json.dumps(results, indent=2))
    else:
        for result in results:
            print("\n" + "=" * 70)
            print(f"VISUAL VALIDATION RESULTS")
            print("=" * 70)

            if 'error' in result:
                print(f"ERROR: {result['error']}")
                continue

            if 'directory' in result:
                print(f"Directory: {result['directory']}")
                print(f"Total Images: {result['total_images']}")
                print(f"Passed: {result['passed']}")
                print(f"Failed: {result['failed']}")
                print(f"Average Score: {result['average_score']}/10")
                print(f"Overall: {result['overall_verdict']}")

                for img_result in result.get('image_results', []):
                    print(f"\n  {Path(img_result['image']).name}: {img_result['overall_verdict']} ({img_result['aggregate_score']}/10)")
            else:
                print(f"Image: {result.get('image', 'N/A')}")
                print(f"Aggregate Score: {result.get('aggregate_score', 0)}/10")
                print(f"Overall: {result.get('overall_verdict', 'N/A')}")

                for pr in result.get('persona_results', []):
                    r = pr.get('result', {})
                    print(f"\n  {pr['persona']}: {r.get('overall_score', 0)}/10 - {r.get('pass_fail', 'N/A')}")
                    if r.get('critical_issues'):
                        print("    Critical Issues:")
                        for issue in r['critical_issues'][:3]:
                            print(f"      - {issue}")
                    if r.get('summary'):
                        print(f"    Summary: {r['summary'][:200]}...")


if __name__ == '__main__':
    main()
