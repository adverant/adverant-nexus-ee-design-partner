#!/usr/bin/env python3
"""
Example usage of Visual Verification Agent

Demonstrates how to integrate visual quality checking into MAPO pipeline.
"""

import asyncio
import json
import logging
import sys
from pathlib import Path

from visual_verifier import (
    VisualVerifier,
    VerificationError,
    verify_schematic
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def example_basic_usage():
    """Example 1: Basic verification of a single schematic."""
    print("\n" + "="*80)
    print("EXAMPLE 1: Basic Schematic Verification")
    print("="*80)

    # Path to test schematic
    schematic_path = "../../output/test_1.kicad_sch"

    try:
        # Verify schematic
        report = await verify_schematic(
            schematic_path,
            output_dir="/tmp/visual_verification"
        )

        # Print report
        print(report)

        # Check result
        if report.passed:
            logger.info("✓ Schematic passed visual quality check")
            return 0
        else:
            logger.error("✗ Schematic failed visual quality check")
            return 1

    except VerificationError as e:
        logger.error(f"Verification failed: {e}")
        return 1


async def example_batch_verification():
    """Example 2: Batch verify multiple schematics."""
    print("\n" + "="*80)
    print("EXAMPLE 2: Batch Verification")
    print("="*80)

    # Find all test schematics
    output_dir = Path("../../output")
    schematics = list(output_dir.glob("test_*.kicad_sch"))[:3]  # Limit to 3

    logger.info(f"Found {len(schematics)} schematics to verify")

    # Initialize verifier once
    verifier = VisualVerifier()

    results = []
    for schematic in schematics:
        logger.info(f"\nVerifying: {schematic.name}")

        try:
            report = await verifier.verify(
                str(schematic),
                output_dir="/tmp/visual_verification"
            )

            results.append({
                'schematic': schematic.name,
                'passed': report.passed,
                'score': report.overall_score,
                'issues': len(report.issues)
            })

            status = "✓ PASS" if report.passed else "✗ FAIL"
            logger.info(
                f"{status}: {schematic.name} - "
                f"Score: {report.overall_score:.1f}/100 "
                f"({len(report.issues)} issues)"
            )

        except VerificationError as e:
            logger.error(f"Failed to verify {schematic.name}: {e}")
            results.append({
                'schematic': schematic.name,
                'passed': False,
                'score': 0,
                'error': str(e)
            })

    # Summary
    print("\n" + "-"*80)
    print("BATCH VERIFICATION SUMMARY")
    print("-"*80)
    passed = sum(1 for r in results if r.get('passed', False))
    print(f"Total: {len(results)}")
    print(f"Passed: {passed}")
    print(f"Failed: {len(results) - passed}")

    # Detailed results
    print("\nDetailed Results:")
    for result in results:
        status = "✓" if result.get('passed', False) else "✗"
        score = result.get('score', 0)
        issues = result.get('issues', 'N/A')
        print(f"  {status} {result['schematic']:30} {score:5.1f}/100  ({issues} issues)")

    return 0 if passed == len(results) else 1


async def example_quality_gate_integration():
    """Example 3: Integration with MAPO pipeline as quality gate."""
    print("\n" + "="*80)
    print("EXAMPLE 3: MAPO Pipeline Integration")
    print("="*80)

    class SchematicPipeline:
        """Simulated MAPO pipeline with visual quality gate."""

        def __init__(self):
            self.visual_verifier = VisualVerifier()

        async def process_schematic(self, schematic_path: str):
            """Process schematic through full pipeline."""
            logger.info(f"Processing: {schematic_path}")

            # Step 1: Symbol resolution (simulated)
            logger.info("  [1/4] Symbol resolution... ✓")

            # Step 2: Connection inference (simulated)
            logger.info("  [2/4] Connection inference... ✓")

            # Step 3: Layout optimization (simulated)
            logger.info("  [3/4] Layout optimization... ✓")

            # Step 4: Visual quality verification (REAL)
            logger.info("  [4/4] Visual quality verification...")

            try:
                report = await self.visual_verifier.verify(
                    schematic_path,
                    output_dir="/tmp/mapo_pipeline"
                )

                if not report.passed:
                    logger.error(
                        f"  ✗ Quality gate FAILED: {report.overall_score:.1f}/100"
                    )
                    logger.error(f"  Issues identified: {len(report.issues)}")

                    # Log specific issues
                    for issue in report.issues[:3]:  # Top 3 issues
                        logger.error(
                            f"    - {issue.criterion_name}: "
                            f"{issue.issue_description}"
                        )

                    return False, report

                logger.info(
                    f"  ✓ Quality gate PASSED: {report.overall_score:.1f}/100"
                )
                return True, report

            except VerificationError as e:
                logger.error(f"  ✗ Verification error: {e}")
                return False, None

    # Run pipeline
    pipeline = SchematicPipeline()
    test_schematic = "../../output/test_1.kicad_sch"

    success, report = await pipeline.process_schematic(test_schematic)

    if success:
        logger.info("\n✓ Pipeline completed successfully")
        if report:
            # Save report for records
            report_path = "/tmp/mapo_pipeline/quality_report.json"
            with open(report_path, 'w') as f:
                json.dump(report.to_dict(), f, indent=2)
            logger.info(f"Quality report saved: {report_path}")
        return 0
    else:
        logger.error("\n✗ Pipeline failed quality gate")
        return 1


async def example_custom_rubric():
    """Example 4: Using custom quality rubric."""
    print("\n" + "="*80)
    print("EXAMPLE 4: Custom Quality Rubric")
    print("="*80)

    # Create custom rubric with stricter thresholds
    custom_rubric_path = "/tmp/custom_rubric.yaml"

    custom_rubric = """
# Custom rubric with stricter standards
rubric:
  - id: symbol_overlap
    name: "Symbol Overlap"
    weight: 0.20  # Higher weight for critical issue
    pass_criteria: "Zero overlapping symbols (zero tolerance)"
    scoring: "0 overlaps = 100, 1+ overlaps = 0"
    description: "Absolutely no symbol overlaps allowed"

  - id: wire_crossings
    name: "Wire Crossings"
    weight: 0.15
    pass_criteria: "Maximum 3 wire crossings total"
    scoring: "0-3 crossings = 100, 4-10 = 50, 11+ = 0"
    description: "Stricter crossing limits"

  - id: signal_flow
    name: "Signal Flow Direction"
    weight: 0.15
    pass_criteria: "100% left-to-right signal flow"
    scoring: "Perfect flow = 100, any deviation = 50"
    description: "Strict signal flow requirement"

  - id: professional_appearance
    name: "Professional Appearance"
    weight: 0.50  # Majority weight on overall quality
    pass_criteria: "Exceptional professional appearance"
    scoring: "Exceptional = 100, good = 70, acceptable = 40"
    description: "Must look like reference design"

thresholds:
  pass: 95  # Very strict pass threshold

image_settings:
  format: "png"
  density: 300
  background: "white"
"""

    # Write custom rubric
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write(custom_rubric)
        custom_rubric_path = f.name

    logger.info(f"Created custom rubric: {custom_rubric_path}")

    # Use custom rubric
    verifier = VisualVerifier(rubric_path=custom_rubric_path)

    test_schematic = "../../output/test_1.kicad_sch"

    try:
        report = await verifier.verify(test_schematic)

        print(f"\nCustom Rubric Results:")
        print(f"  Pass Threshold: 95/100")
        print(f"  Actual Score: {report.overall_score:.1f}/100")
        print(f"  Status: {'✓ PASS' if report.passed else '✗ FAIL'}")

        return 0 if report.passed else 1

    except VerificationError as e:
        logger.error(f"Verification failed: {e}")
        return 1


async def main():
    """Run all examples."""
    if len(sys.argv) > 1:
        example = sys.argv[1]
        examples = {
            '1': example_basic_usage,
            '2': example_batch_verification,
            '3': example_quality_gate_integration,
            '4': example_custom_rubric
        }

        if example in examples:
            return await examples[example]()
        else:
            print(f"Unknown example: {example}")
            print("Usage: python example_usage.py [1|2|3|4]")
            return 1

    # Run all examples
    print("\n" + "="*80)
    print("VISUAL VERIFICATION AGENT - EXAMPLES")
    print("="*80)

    results = []

    # Example 1
    result = await example_basic_usage()
    results.append(('Basic Usage', result))

    # Example 2
    result = await example_batch_verification()
    results.append(('Batch Verification', result))

    # Example 3
    result = await example_quality_gate_integration()
    results.append(('Pipeline Integration', result))

    # Example 4
    result = await example_custom_rubric()
    results.append(('Custom Rubric', result))

    # Summary
    print("\n" + "="*80)
    print("EXAMPLES SUMMARY")
    print("="*80)
    for name, result in results:
        status = "✓ SUCCESS" if result == 0 else "✗ FAILED"
        print(f"{status}: {name}")

    return 0 if all(r == 0 for _, r in results) else 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
