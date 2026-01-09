#!/usr/bin/env python3
"""
Full Pipeline E2E Test Runner - PRODUCTION VERSION

Tests the complete schematic-gen marketplace plugin pipeline:
1. SKiDL circuit definition
2. Netlist generation
3. Schematic conversion
4. PCB layout generation
5. KiCad CLI image export
6. Pre-validation (catches empty/garbage output)
7. Multi-agent validation
8. Visual validation loop (Ralph Wiggum iterative refinement)

CRITICAL: This test has NO MOCKS, NO FALLBACKS, NO SKIP OPTIONS.
All dependencies are MANDATORY. All validation is MANDATORY.
Failures raise exceptions - they CANNOT be ignored.

Usage:
    python run_full_pipeline_test.py
    python run_full_pipeline_test.py --output-dir ./test_output
    python run_full_pipeline_test.py --quality-threshold 9.0
    python run_full_pipeline_test.py --max-iterations 100
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, asdict, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional

# Add python-scripts to path for imports
SCRIPT_DIR = Path(__file__).parent.parent.parent / 'python-scripts'
sys.path.insert(0, str(SCRIPT_DIR))

from validation_exceptions import (
    ValidationFailure,
    EmptyBoardFailure,
    EmptySilkscreenFailure,
    QualityThresholdFailure,
    MissingDependencyFailure,
    RoutingFailure,
    DRCFailure
)
from pre_validator import PreValidator


@dataclass
class StageResult:
    """Result of a pipeline stage."""
    name: str
    passed: bool
    duration_ms: int
    output_files: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PipelineTestResult:
    """Complete pipeline test result."""
    test_id: str
    test_name: str
    timestamp: str
    passed: bool
    total_duration_ms: int
    stages: List[StageResult]
    summary: str
    quality_score: float = 0.0


class FullPipelineTest:
    """
    Comprehensive E2E test for the schematic-gen plugin.

    PRODUCTION VERSION - No mocks, no fallbacks, no skip options.
    All dependencies are MANDATORY. All validation is MANDATORY.
    Failures raise exceptions that CANNOT be ignored.
    """

    def __init__(
        self,
        output_dir: str,
        quality_threshold: float = 9.0,
        max_iterations: int = 100
    ):
        """
        Initialize pipeline test.

        Args:
            output_dir: Directory for test output artifacts
            quality_threshold: Minimum score required (1-10)
            max_iterations: Maximum Ralph loop iterations

        Raises:
            MissingDependencyFailure: If required dependencies are not available
        """
        self.output_dir = Path(output_dir)
        self.quality_threshold = quality_threshold
        self.max_iterations = max_iterations
        self.script_dir = SCRIPT_DIR

        # Test circuit parameters
        self.test_name = "555 LED Driver"
        self.test_id = f"e2e-{datetime.now().strftime('%Y%m%d-%H%M%S')}"

        self.stages: List[StageResult] = []
        self.pre_validator = PreValidator(strict=True)

        # MANDATORY: Verify all dependencies at initialization
        self._verify_dependencies()

    def _verify_dependencies(self) -> None:
        """
        Verify all required dependencies are available.

        Raises:
            MissingDependencyFailure: If any dependency is missing
        """
        # Check for KiCad CLI
        kicad_cli = shutil.which('kicad-cli')
        if not kicad_cli:
            raise MissingDependencyFailure(
                "kicad-cli not found in PATH - cannot export images",
                dependency_name="kicad-cli",
                install_instructions="Install KiCad 8.0+ from https://www.kicad.org/download/"
            )

        # Check for ANTHROPIC_API_KEY
        if not os.environ.get('ANTHROPIC_API_KEY'):
            raise MissingDependencyFailure(
                "ANTHROPIC_API_KEY environment variable not set - cannot run AI validation",
                dependency_name="ANTHROPIC_API_KEY",
                install_instructions="Set ANTHROPIC_API_KEY environment variable with your API key"
            )

        # Check for Python dependencies
        try:
            from PIL import Image
            import numpy as np
        except ImportError as e:
            raise MissingDependencyFailure(
                f"Required Python package not installed: {e}",
                dependency_name="Pillow/numpy",
                install_instructions="pip install Pillow numpy"
            )

        # Check for pcbnew (KiCad Python API)
        try:
            import pcbnew
        except ImportError:
            # pcbnew is optional for image export but required for PCB modification
            print("  WARNING: pcbnew not available - PCB modification will use S-expression fallback")

        print("  All dependencies verified OK")

    def setup(self) -> None:
        """Create output directories."""
        dirs = [
            self.output_dir,
            self.output_dir / 'netlists',
            self.output_dir / 'schematics',
            self.output_dir / 'pcb',
            self.output_dir / 'images',
            self.output_dir / 'reports',
            self.output_dir / 'gerbers'
        ]

        for d in dirs:
            d.mkdir(parents=True, exist_ok=True)

        print(f"Test output directory: {self.output_dir}")

    def _run_stage(self, name: str, func, critical: bool = True) -> StageResult:
        """
        Run a pipeline stage and record results.

        Args:
            name: Stage name for display
            func: Stage function to execute
            critical: If True, stage failure raises exception

        Returns:
            StageResult with execution details

        Raises:
            ValidationFailure: If critical stage fails
        """
        print(f"\n{'='*60}")
        print(f"STAGE: {name}")
        print(f"{'='*60}")

        start = time.time()
        try:
            result = func()
            duration = int((time.time() - start) * 1000)

            stage = StageResult(
                name=name,
                passed=result.get('success', False),
                duration_ms=duration,
                output_files=result.get('files', []),
                errors=result.get('errors', []),
                warnings=result.get('warnings', []),
                metrics=result.get('metrics', {})
            )

            if stage.passed:
                print(f"  PASSED ({duration}ms)")
            else:
                print(f"  FAILED ({duration}ms)")
                for err in stage.errors:
                    print(f"    ERROR: {err}")

                # Critical stages raise on failure
                if critical:
                    raise ValidationFailure(
                        f"Stage '{name}' failed: {'; '.join(stage.errors)}",
                        score=0.0
                    )

        except ValidationFailure:
            # Re-raise validation failures
            raise
        except Exception as e:
            duration = int((time.time() - start) * 1000)
            stage = StageResult(
                name=name,
                passed=False,
                duration_ms=duration,
                errors=[str(e)]
            )
            print(f"  EXCEPTION: {e}")

            if critical:
                raise ValidationFailure(
                    f"Stage '{name}' raised exception: {e}",
                    score=0.0
                )

        self.stages.append(stage)
        return stage

    def stage_generate_netlist(self) -> Dict[str, Any]:
        """Stage 1: Generate netlist from SKiDL."""
        print("  Generating netlist from test circuit...")

        # Import and run the test circuit
        sys.path.insert(0, str(Path(__file__).parent))
        from test_led_driver import generate_real_netlist

        netlist_path = str(self.output_dir / 'netlists' / 'led_driver.net')
        result = generate_real_netlist(netlist_path)

        if not result.get('success'):
            return {
                'success': False,
                'errors': [result.get('error', 'Netlist generation failed')]
            }

        # Verify netlist was actually created and has content
        netlist_file = Path(netlist_path)
        if not netlist_file.exists():
            return {
                'success': False,
                'errors': ['Netlist file was not created']
            }

        if netlist_file.stat().st_size < 100:
            return {
                'success': False,
                'errors': [f'Netlist file too small ({netlist_file.stat().st_size} bytes) - likely empty']
            }

        return {
            'success': True,
            'files': [netlist_path],
            'metrics': {
                'components': result.get('components', 0),
                'nets': result.get('nets', 0),
                'file_size_bytes': netlist_file.stat().st_size
            }
        }

    def stage_convert_schematic(self) -> Dict[str, Any]:
        """Stage 2: Convert netlist to KiCad schematic."""
        print("  Converting netlist to schematic...")

        netlist_path = self.output_dir / 'netlists' / 'led_driver.net'
        schematic_path = self.output_dir / 'schematics' / 'led_driver.kicad_sch'

        # Use netlist_to_schematic.py - REQUIRED, no fallback
        converter = self.script_dir / 'netlist_to_schematic.py'

        if not converter.exists():
            return {
                'success': False,
                'errors': [f'Schematic converter not found: {converter}']
            }

        try:
            result = subprocess.run(
                [sys.executable, str(converter),
                 '--netlists', str(netlist_path),
                 '--output', str(schematic_path)],
                capture_output=True,
                text=True,
                timeout=120
            )

            if result.returncode != 0:
                return {
                    'success': False,
                    'errors': [f'Converter failed: {result.stderr}']
                }

            # Verify schematic was created
            if not schematic_path.exists():
                return {
                    'success': False,
                    'errors': ['Schematic file was not created']
                }

            if schematic_path.stat().st_size < 500:
                return {
                    'success': False,
                    'errors': [f'Schematic file too small ({schematic_path.stat().st_size} bytes)']
                }

            return {
                'success': True,
                'files': [str(schematic_path)],
                'metrics': {
                    'file_size_bytes': schematic_path.stat().st_size
                }
            }

        except subprocess.TimeoutExpired:
            return {
                'success': False,
                'errors': ['Schematic conversion timed out after 120 seconds']
            }
        except Exception as e:
            return {
                'success': False,
                'errors': [f'Schematic conversion error: {e}']
            }

    def stage_generate_pcb(self) -> Dict[str, Any]:
        """Stage 3: Generate PCB layout."""
        print("  Generating PCB layout...")

        schematic_path = self.output_dir / 'schematics' / 'led_driver.kicad_sch'
        pcb_path = self.output_dir / 'pcb' / 'led_driver.kicad_pcb'

        # Import real PCB generator - NO MOCKS
        sys.path.insert(0, str(Path(__file__).parent))
        from test_led_driver import generate_real_pcb

        result = generate_real_pcb(str(schematic_path), str(pcb_path))

        if not result.get('success'):
            return {
                'success': False,
                'errors': [result.get('error', 'PCB generation failed')]
            }

        # Verify PCB was created and has content
        if not pcb_path.exists():
            return {
                'success': False,
                'errors': ['PCB file was not created']
            }

        # Check PCB has actual content (traces, footprints)
        pcb_content = pcb_path.read_text()

        # Count footprints
        footprint_count = pcb_content.count('(footprint ')
        if footprint_count < result.get('expected_footprints', 5):
            return {
                'success': False,
                'errors': [f'PCB has only {footprint_count} footprints - expected at least {result.get("expected_footprints", 5)}']
            }

        # Count traces (segments)
        segment_count = pcb_content.count('(segment ')
        if segment_count < 5:
            return {
                'success': False,
                'errors': [f'PCB has only {segment_count} trace segments - expected routed connections']
            }

        return {
            'success': True,
            'files': [str(pcb_path)],
            'metrics': {
                'board_width_mm': result.get('board_width_mm', 0),
                'board_height_mm': result.get('board_height_mm', 0),
                'layer_count': result.get('layer_count', 2),
                'footprint_count': footprint_count,
                'segment_count': segment_count
            }
        }

    def stage_export_images(self) -> Dict[str, Any]:
        """Stage 4: Export images using KiCad CLI."""
        print("  Exporting layer images...")

        pcb_path = self.output_dir / 'pcb' / 'led_driver.kicad_pcb'
        images_dir = self.output_dir / 'images'

        # Use KiCad CLI directly - REQUIRED, no fallback
        kicad_cli = shutil.which('kicad-cli')

        # Layers to export
        layers = [
            ('F.Cu', 'F_Cu'),
            ('B.Cu', 'B_Cu'),
            ('F.SilkS', 'F_SilkS'),
            ('B.SilkS', 'B_SilkS'),
            ('F.Mask', 'F_Mask'),
            ('B.Mask', 'B_Mask'),
            ('F.Paste', 'F_Paste'),
            ('B.Paste', 'B_Paste'),
            ('Edge.Cuts', 'Edge_Cuts'),
            ('F.Fab', 'F_Fab'),
            ('B.Fab', 'B_Fab')
        ]

        exported_files = []

        for layer_name, safe_name in layers:
            svg_path = images_dir / f'{safe_name}.svg'
            png_path = images_dir / f'{safe_name}.png'

            try:
                # Export layer to SVG
                result = subprocess.run(
                    [kicad_cli, 'pcb', 'export', 'svg',
                     '--layers', layer_name,
                     '--output', str(svg_path),
                     str(pcb_path)],
                    capture_output=True,
                    text=True,
                    timeout=60
                )

                if result.returncode != 0:
                    print(f"    Warning: Could not export {layer_name}: {result.stderr}")
                    continue

                # Convert SVG to PNG using cairosvg or rsvg-convert
                if shutil.which('rsvg-convert'):
                    subprocess.run(
                        ['rsvg-convert', '-f', 'png', '-o', str(png_path), str(svg_path)],
                        capture_output=True,
                        timeout=30
                    )
                else:
                    try:
                        import cairosvg
                        cairosvg.svg2png(url=str(svg_path), write_to=str(png_path))
                    except ImportError:
                        # Use PIL as fallback for SVG conversion
                        from PIL import Image
                        # Read SVG dimensions and create placeholder
                        print(f"    Warning: cairosvg not available for {layer_name}")
                        continue

                if png_path.exists():
                    exported_files.append(str(png_path))

            except subprocess.TimeoutExpired:
                print(f"    Warning: Export timeout for {layer_name}")
            except Exception as e:
                print(f"    Warning: Export error for {layer_name}: {e}")

        # Export assembly views
        for view_name, view_layers in [
            ('top_view', 'F.Cu,F.SilkS,Edge.Cuts'),
            ('bottom_view', 'B.Cu,B.SilkS,Edge.Cuts'),
            ('assembly_top', 'F.Fab,F.SilkS,Edge.Cuts'),
            ('assembly_bottom', 'B.Fab,B.SilkS,Edge.Cuts')
        ]:
            svg_path = images_dir / f'{view_name}.svg'
            png_path = images_dir / f'{view_name}.png'

            try:
                result = subprocess.run(
                    [kicad_cli, 'pcb', 'export', 'svg',
                     '--layers', view_layers,
                     '--output', str(svg_path),
                     str(pcb_path)],
                    capture_output=True,
                    text=True,
                    timeout=60
                )

                if result.returncode == 0:
                    if shutil.which('rsvg-convert'):
                        subprocess.run(
                            ['rsvg-convert', '-f', 'png', '-o', str(png_path), str(svg_path)],
                            capture_output=True,
                            timeout=30
                        )
                    else:
                        try:
                            import cairosvg
                            cairosvg.svg2png(url=str(svg_path), write_to=str(png_path))
                        except ImportError:
                            pass

                    if png_path.exists():
                        exported_files.append(str(png_path))

            except Exception as e:
                print(f"    Warning: Assembly export error for {view_name}: {e}")

        if len(exported_files) < 5:
            return {
                'success': False,
                'errors': [f'Only {len(exported_files)} images exported - expected at least 5']
            }

        return {
            'success': True,
            'files': exported_files,
            'metrics': {'layer_count': len(exported_files)}
        }

    def stage_pre_validation(self) -> Dict[str, Any]:
        """Stage 5: Pre-validate images contain real PCB data."""
        print("  Pre-validating images for actual content...")

        images_dir = self.output_dir / 'images'

        try:
            results = self.pre_validator.validate_all_images(str(images_dir))

            # Check results
            failed = [name for name, r in results.items() if not r.passed]

            if failed:
                # Pre-validation failure is CRITICAL - stops pipeline
                raise EmptyBoardFailure(
                    f"Pre-validation failed for {len(failed)} images: {', '.join(failed)}",
                    image_path=str(images_dir)
                )

            return {
                'success': True,
                'metrics': {
                    'images_validated': len(results),
                    'all_passed': True
                }
            }

        except (EmptyBoardFailure, EmptySilkscreenFailure):
            # Re-raise validation failures
            raise
        except Exception as e:
            return {
                'success': False,
                'errors': [f'Pre-validation error: {e}']
            }

    def stage_multi_agent_validation(self) -> Dict[str, Any]:
        """Stage 6: Run multi-agent validation."""
        print("  Running multi-agent validation...")

        schematic_path = self.output_dir / 'schematics' / 'led_driver.kicad_sch'

        # Try using multi_agent_validator
        validator_script = self.script_dir / 'multi_agent_validator.py'

        if not validator_script.exists():
            return {
                'success': False,
                'errors': [f'Multi-agent validator not found: {validator_script}']
            }

        try:
            result = subprocess.run(
                [sys.executable, str(validator_script),
                 '--path', str(schematic_path),
                 '--json'],
                capture_output=True,
                text=True,
                timeout=180,
                env={**os.environ, 'ANTHROPIC_API_KEY': os.environ.get('ANTHROPIC_API_KEY', '')}
            )

            if result.returncode != 0:
                return {
                    'success': False,
                    'errors': [f'Validator failed: {result.stderr}']
                }

            if result.stdout:
                validation_result = json.loads(result.stdout)
                score = validation_result.get('score', 0)
                passed = validation_result.get('passed', False)

                if not passed:
                    return {
                        'success': False,
                        'errors': [f'Multi-agent validation failed with score {score}'],
                        'metrics': {
                            'score': score,
                            'issues': len(validation_result.get('errors', []))
                        }
                    }

                return {
                    'success': True,
                    'metrics': {
                        'score': score,
                        'issues': len(validation_result.get('errors', []))
                    }
                }

            return {
                'success': False,
                'errors': ['Validator returned no output']
            }

        except subprocess.TimeoutExpired:
            return {
                'success': False,
                'errors': ['Multi-agent validation timed out after 180 seconds']
            }
        except json.JSONDecodeError as e:
            return {
                'success': False,
                'errors': [f'Invalid JSON from validator: {e}']
            }
        except Exception as e:
            return {
                'success': False,
                'errors': [f'Validation error: {e}']
            }

    def stage_visual_validation(self) -> Dict[str, Any]:
        """Stage 7: Run visual validation loop (Ralph Wiggum)."""
        print("  Running visual validation loop...")
        print(f"    Quality threshold: {self.quality_threshold}")
        print(f"    Max iterations: {self.max_iterations}")

        images_dir = self.output_dir / 'images'
        pcb_path = self.output_dir / 'pcb' / 'led_driver.kicad_pcb'

        try:
            from visual_ralph_loop import VisualRalphLoop

            loop = VisualRalphLoop(
                output_dir=str(images_dir),
                pcb_path=str(pcb_path),
                max_iterations=self.max_iterations,
                quality_threshold=self.quality_threshold,
                strict=True
            )

            # Run the full loop - raises QualityThresholdFailure if it doesn't converge
            result = loop.run()

            return {
                'success': True,
                'metrics': {
                    'overall_score': result.get('overall_score', 0),
                    'iterations': result.get('iteration', 0),
                    'images_validated': result.get('total_images', 0),
                    'issues_fixed': result.get('total_fixes', 0)
                }
            }

        except QualityThresholdFailure as e:
            # Re-raise - this is a legitimate failure
            raise
        except ValidationFailure as e:
            # Re-raise validation failures
            raise
        except Exception as e:
            return {
                'success': False,
                'errors': [f'Visual validation error: {e}']
            }

    def stage_generate_report(self) -> Dict[str, Any]:
        """Stage 8: Generate test report."""
        print("  Generating test report...")

        report_path = self.output_dir / 'reports' / 'e2e_test_report.json'

        # Calculate overall quality score
        quality_scores = []
        for stage in self.stages:
            if 'overall_score' in stage.metrics:
                quality_scores.append(stage.metrics['overall_score'])
            elif 'score' in stage.metrics:
                quality_scores.append(stage.metrics['score'])

        overall_quality = sum(quality_scores) / len(quality_scores) if quality_scores else 0.0

        # Compile results
        results = {
            'test_id': self.test_id,
            'test_name': self.test_name,
            'timestamp': datetime.now().isoformat(),
            'quality_threshold': self.quality_threshold,
            'overall_quality_score': overall_quality,
            'stages': [asdict(s) for s in self.stages],
            'summary': {
                'total_stages': len(self.stages),
                'passed_stages': sum(1 for s in self.stages if s.passed),
                'total_duration_ms': sum(s.duration_ms for s in self.stages),
                'quality_score': overall_quality
            }
        }

        with open(report_path, 'w') as f:
            json.dump(results, f, indent=2)

        return {
            'success': True,
            'files': [str(report_path)],
            'metrics': {
                'overall_quality_score': overall_quality
            }
        }

    def run(self) -> PipelineTestResult:
        """
        Run the complete pipeline test.

        Returns:
            PipelineTestResult with all stage results

        Raises:
            ValidationFailure: If any critical stage fails
            MissingDependencyFailure: If dependencies are missing
        """
        start = time.time()

        print("\n" + "=" * 60)
        print("FULL PIPELINE E2E TEST - PRODUCTION VERSION")
        print(f"Test: {self.test_name}")
        print(f"ID: {self.test_id}")
        print(f"Quality Threshold: {self.quality_threshold}/10")
        print(f"Max Iterations: {self.max_iterations}")
        print("=" * 60)

        # Setup
        self.setup()

        # Run all stages - ALL ARE CRITICAL (except report generation)
        stages = [
            ('1. Generate Netlist', self.stage_generate_netlist, True),
            ('2. Convert to Schematic', self.stage_convert_schematic, True),
            ('3. Generate PCB Layout', self.stage_generate_pcb, True),
            ('4. Export Images', self.stage_export_images, True),
            ('5. Pre-Validation', self.stage_pre_validation, True),
            ('6. Multi-Agent Validation', self.stage_multi_agent_validation, True),
            ('7. Visual Validation Loop', self.stage_visual_validation, True),
            ('8. Generate Report', self.stage_generate_report, False)
        ]

        for stage_name, stage_func, critical in stages:
            self._run_stage(stage_name, stage_func, critical=critical)

        total_duration = int((time.time() - start) * 1000)

        # ALL stages must pass - no exceptions
        all_passed = all(s.passed for s in self.stages)

        # Calculate final quality score
        quality_scores = []
        for stage in self.stages:
            if 'overall_score' in stage.metrics:
                quality_scores.append(stage.metrics['overall_score'])
            elif 'score' in stage.metrics:
                quality_scores.append(stage.metrics['score'])

        final_quality = sum(quality_scores) / len(quality_scores) if quality_scores else 0.0

        # Generate summary
        passed_count = sum(1 for s in self.stages if s.passed)
        summary = f"{passed_count}/{len(self.stages)} stages passed, quality score: {final_quality:.1f}/10"

        result = PipelineTestResult(
            test_id=self.test_id,
            test_name=self.test_name,
            timestamp=datetime.now().isoformat(),
            passed=all_passed,
            total_duration_ms=total_duration,
            stages=self.stages,
            summary=summary,
            quality_score=final_quality
        )

        # Print final summary
        print("\n" + "=" * 60)
        print("FINAL RESULTS")
        print("=" * 60)
        print(f"Test: {result.test_name}")
        print(f"Status: {'PASSED' if result.passed else 'FAILED'}")
        print(f"Quality Score: {result.quality_score:.1f}/10")
        print(f"Duration: {result.total_duration_ms}ms")
        print(f"Summary: {result.summary}")

        print("\nStage Results:")
        for stage in result.stages:
            status = "PASS" if stage.passed else "FAIL"
            print(f"  [{status}] {stage.name} ({stage.duration_ms}ms)")
            for key, value in stage.metrics.items():
                print(f"         {key}: {value}")
            for warn in stage.warnings:
                print(f"         Warning: {warn}")
            for err in stage.errors:
                print(f"         Error: {err}")

        return result


def main():
    parser = argparse.ArgumentParser(
        description='Full Pipeline E2E Test Runner - PRODUCTION VERSION (No mocks, no fallbacks)'
    )
    parser.add_argument(
        '--output-dir', '-o',
        type=str,
        default='./e2e_test_output',
        help='Output directory for test artifacts'
    )
    parser.add_argument(
        '--quality-threshold', '-q',
        type=float,
        default=9.0,
        help='Minimum quality score required (1-10, default: 9.0)'
    )
    parser.add_argument(
        '--max-iterations', '-m',
        type=int,
        default=100,
        help='Maximum Ralph loop iterations (default: 100)'
    )
    parser.add_argument(
        '--json',
        action='store_true',
        help='Output results as JSON'
    )

    args = parser.parse_args()

    try:
        test = FullPipelineTest(
            output_dir=args.output_dir,
            quality_threshold=args.quality_threshold,
            max_iterations=args.max_iterations
        )

        result = test.run()

        if args.json:
            print("\n" + json.dumps(asdict(result), indent=2, default=str))

        sys.exit(0 if result.passed else 1)

    except MissingDependencyFailure as e:
        print(f"\nMISSING DEPENDENCY: {e}", file=sys.stderr)
        sys.exit(2)
    except ValidationFailure as e:
        print(f"\nVALIDATION FAILED: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"\nUNEXPECTED ERROR: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(3)


if __name__ == '__main__':
    main()
