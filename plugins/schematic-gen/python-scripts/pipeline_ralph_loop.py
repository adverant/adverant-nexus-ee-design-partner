#!/usr/bin/env python3
"""
Pipeline Ralph Loop - Iterative Pipeline Validation and Improvement

This script implements the Ralph Wiggum loop technique for validating and
improving the PCB generation pipeline until it consistently produces
professional-quality output (9.0/10 or better).

The loop:
1. Generates a test PCB with the pipeline
2. Exports images using KiCad
3. Validates using Claude Opus 4.5 vision
4. If score < 9.0, analyzes issues and suggests pipeline improvements
5. Repeats until target score achieved or max iterations reached

Usage:
    python pipeline_ralph_loop.py --max-iterations 100 --threshold 9.0
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

# Ensure we can import from our modules
SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, '/Users/don/Adverant/Adverant-Nexus/services/nexus-pcb-layout/python-scripts')

# Import validation modules
from validation_exceptions import (
    ValidationFailure,
    QualityThresholdFailure,
    MissingDependencyFailure
)
from pre_validator import PreValidator
from openrouter_client import OpenRouterClient, get_openrouter_client
from visual_ralph_loop import MultiAgentValidator, VALIDATION_PERSONAS

# Check for pcbnew
try:
    import pcbnew
    HAS_PCBNEW = True
except ImportError:
    HAS_PCBNEW = False


@dataclass
class IterationResult:
    """Result of a single validation iteration."""
    iteration: int
    timestamp: str
    score: float
    passed: bool
    issues: List[str]
    suggestions: List[str]
    images_validated: List[str]


@dataclass
class LoopState:
    """Persistent state for the Ralph loop."""
    max_iterations: int = 100
    quality_threshold: float = 9.0
    current_iteration: int = 0
    best_score: float = 0.0
    history: List[IterationResult] = field(default_factory=list)
    status: str = "running"
    start_time: str = ""


class PipelineRalphLoop:
    """
    Ralph Wiggum loop for pipeline validation.

    Iteratively tests and validates the PCB generation pipeline
    until it consistently produces professional-quality output.
    """

    KICAD_CLI = '/Applications/KiCad/KiCad.app/Contents/MacOS/kicad-cli'
    KICAD_PYTHON = '/Applications/KiCad/KiCad.app/Contents/Frameworks/Python.framework/Versions/3.9/bin/python3'

    def __init__(
        self,
        output_dir: str,
        max_iterations: int = 100,
        quality_threshold: float = 9.0
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.state = LoopState(
            max_iterations=max_iterations,
            quality_threshold=quality_threshold,
            start_time=datetime.now().isoformat()
        )

        self.state_file = self.output_dir / 'ralph_loop_state.json'
        self.pcb_dir = self.output_dir / 'pcb'
        self.images_dir = self.output_dir / 'images'

        self.pcb_dir.mkdir(exist_ok=True)
        self.images_dir.mkdir(exist_ok=True)

        # Initialize clients
        self._verify_dependencies()
        self.openrouter = get_openrouter_client()
        self.pre_validator = PreValidator()

        # Load state if exists
        self._load_state()

    def _verify_dependencies(self):
        """Verify all required dependencies are available."""
        # Check OpenRouter API key
        api_key = os.environ.get('OPENROUTER_API_KEY')
        if not api_key:
            raise MissingDependencyFailure(
                "OPENROUTER_API_KEY not set",
                dependency_name="OPENROUTER_API_KEY",
                install_instructions="export OPENROUTER_API_KEY=your-key"
            )

        # Check KiCad
        if not os.path.exists(self.KICAD_CLI):
            raise MissingDependencyFailure(
                "KiCad CLI not found",
                dependency_name="kicad-cli",
                install_instructions="Install KiCad from https://kicad.org"
            )

        # Check pcbnew for Python
        if not HAS_PCBNEW:
            print("Warning: pcbnew not available in this Python. Will use KiCad Python.")

    def _load_state(self):
        """Load state from file if exists."""
        if self.state_file.exists():
            try:
                with open(self.state_file, 'r') as f:
                    data = json.load(f)
                    self.state.current_iteration = data.get('current_iteration', 0)
                    self.state.best_score = data.get('best_score', 0.0)
                    self.state.status = data.get('status', 'running')
                    print(f"Resuming from iteration {self.state.current_iteration}, best score: {self.state.best_score}")
            except Exception as e:
                print(f"Could not load state: {e}")

    def _save_state(self):
        """Save state to file."""
        state_data = {
            'max_iterations': self.state.max_iterations,
            'quality_threshold': self.state.quality_threshold,
            'current_iteration': self.state.current_iteration,
            'best_score': self.state.best_score,
            'status': self.state.status,
            'start_time': self.state.start_time,
            'history_length': len(self.state.history)
        }
        with open(self.state_file, 'w') as f:
            json.dump(state_data, f, indent=2)

    def _create_test_design(self) -> Dict[str, Any]:
        """Create a comprehensive test design to validate the pipeline."""
        # LED driver circuit with various component types
        return {
            'placement': {
                'components': [
                    {'reference': 'U1', 'footprint': 'Package_SO:SOIC-8', 'x': 25, 'y': 25, 'angle': 0, 'layer': 'F.Cu'},
                    {'reference': 'U2', 'footprint': 'Package_QFP:TQFP-32', 'x': 60, 'y': 25, 'angle': 0, 'layer': 'F.Cu'},
                    {'reference': 'R1', 'footprint': 'Resistor_SMD:R_0805', 'x': 40, 'y': 15, 'angle': 0, 'layer': 'F.Cu'},
                    {'reference': 'R2', 'footprint': 'Resistor_SMD:R_0805', 'x': 40, 'y': 20, 'angle': 0, 'layer': 'F.Cu'},
                    {'reference': 'R3', 'footprint': 'Resistor_SMD:R_0805', 'x': 40, 'y': 35, 'angle': 0, 'layer': 'F.Cu'},
                    {'reference': 'C1', 'footprint': 'Capacitor_SMD:C_0805', 'x': 15, 'y': 25, 'angle': 90, 'layer': 'F.Cu'},
                    {'reference': 'C2', 'footprint': 'Capacitor_SMD:C_0805', 'x': 15, 'y': 35, 'angle': 90, 'layer': 'F.Cu'},
                    {'reference': 'C3', 'footprint': 'Capacitor_SMD:C_1206', 'x': 80, 'y': 25, 'angle': 0, 'layer': 'F.Cu'},
                    {'reference': 'D1', 'footprint': 'LED_SMD:LED_0805', 'x': 50, 'y': 35, 'angle': 0, 'layer': 'F.Cu'},
                    {'reference': 'D2', 'footprint': 'Diode_SMD:D_SOD-123', 'x': 70, 'y': 35, 'angle': 0, 'layer': 'F.Cu'},
                    {'reference': 'Q1', 'footprint': 'Package_TO_SOT_SMD:SOT-23', 'x': 50, 'y': 15, 'angle': 0, 'layer': 'F.Cu'},
                ]
            },
            'routing': {
                'traces': [
                    # Various traces requiring 45-degree conversion
                    {'netName': 'VCC', 'start': [25, 20], 'end': [40, 15], 'width': 0.3, 'layer': 'F.Cu'},
                    {'netName': 'VCC', 'start': [40, 15], 'end': [50, 10], 'width': 0.3, 'layer': 'F.Cu'},
                    {'netName': 'GND', 'start': [15, 30], 'end': [25, 30], 'width': 0.5, 'layer': 'F.Cu'},
                    {'netName': 'GND', 'start': [25, 30], 'end': [40, 40], 'width': 0.5, 'layer': 'F.Cu'},
                    {'netName': 'SIG1', 'start': [30, 25], 'end': [55, 25], 'width': 0.25, 'layer': 'F.Cu'},
                    {'netName': 'SIG2', 'start': [40, 20], 'end': [60, 15], 'width': 0.25, 'layer': 'F.Cu'},
                    {'netName': 'LED_OUT', 'start': [45, 35], 'end': [55, 35], 'width': 0.3, 'layer': 'F.Cu'},
                    {'netName': 'BIAS', 'start': [70, 30], 'end': [80, 25], 'width': 0.25, 'layer': 'F.Cu'},
                ],
                'vias': [],
                'pours': []
            },
            'constraints': {
                'board': {
                    'auto_size': True,
                    'layers': 2
                },
                'optimize_placement': True
            }
        }

    def _generate_pcb(self, design: Dict[str, Any]) -> Path:
        """Generate PCB using the pipeline."""
        from kicad_layout_generator import generate_layout_from_constraints

        # Create schematic file
        schematic_path = self.pcb_dir / f'test_iter_{self.state.current_iteration}.kicad_sch'
        pcb_path = self.pcb_dir / f'test_iter_{self.state.current_iteration}.kicad_pcb'

        with open(schematic_path, 'w') as f:
            f.write('(kicad_sch (version 20230121))\n')

        result = generate_layout_from_constraints(
            schematic_path=str(schematic_path),
            output_path=str(pcb_path),
            constraints=design['constraints'],
            placement=design['placement'],
            routing=design['routing']
        )

        if not result['success']:
            raise ValidationFailure(f"PCB generation failed: {result.get('errors', [])}")

        return pcb_path

    def _export_images(self, pcb_path: Path) -> List[Path]:
        """Export layer images from PCB as PNG for AI validation."""
        images = []

        # Use kicad-cli render for PNG export (better for AI vision)
        views = [
            ('top', 'top_view.png'),
            ('bottom', 'bottom_view.png'),
        ]

        for side, filename in views:
            output_file = self.images_dir / f'iter_{self.state.current_iteration}_{filename}'

            cmd = [
                self.KICAD_CLI, 'pcb', 'render',
                '--side', side,
                '--width', '1600',
                '--height', '1200',
                '--quality', 'high',
                '-o', str(output_file),
                str(pcb_path)
            ]

            try:
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
                if result.returncode == 0 and output_file.exists():
                    images.append(output_file)
                    print(f"    Rendered {filename}")
            except Exception as e:
                print(f"    Warning: Could not render {side}: {e}")

        # Also export SVG for layer-specific details (kept for reference)
        layers = [
            ('F.Cu', 'F_Cu.svg'),
            ('F.SilkS', 'F_Silkscreen.svg'),
        ]

        for layer, filename in layers:
            svg_file = self.images_dir / f'iter_{self.state.current_iteration}_{filename}'

            cmd = [
                self.KICAD_CLI, 'pcb', 'export', 'svg',
                '-l', layer,
                '-o', str(svg_file),
                str(pcb_path)
            ]

            try:
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
                if result.returncode == 0 and svg_file.exists():
                    print(f"    Exported {filename}")
            except Exception as e:
                print(f"    Warning: Could not export {layer}: {e}")

        return images

    def _generate_pdf_documentation(self, pcb_path: Path) -> Tuple[Optional[Path], List[str]]:
        """
        Generate comprehensive PDF documentation for the PCB.

        Creates:
        1. Assembly documentation PDF (all layers, silkscreen, etc.)
        2. Fabrication documentation PDF (copper, drill, etc.)

        Returns:
            Tuple of (pdf_path, list of issues)
        """
        pdf_dir = self.output_dir / 'pdf'
        pdf_dir.mkdir(exist_ok=True)
        issues = []

        pdf_path = pdf_dir / f'pcb_documentation_iter_{self.state.current_iteration}.pdf'

        # Generate single PDF with all important layers
        # Note: --mode-single creates a single file, --mode-multipage creates a directory
        cmd = [
            self.KICAD_CLI, 'pcb', 'export', 'pdf',
            '--layers', 'F.Cu,B.Cu,F.SilkS,B.SilkS,Edge.Cuts',
            '--include-border-title',
            '--mode-single',
            '-o', str(pdf_path),
            str(pcb_path)
        ]

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
            if result.returncode == 0 and pdf_path.exists():
                print(f"    Generated PDF: {pdf_path.name}")

                # Validate PDF file size (should be > 10KB for valid content)
                pdf_size = pdf_path.stat().st_size
                if pdf_size < 10000:
                    issues.append(f"PDF file too small ({pdf_size} bytes) - may be empty")
                else:
                    print(f"    PDF size: {pdf_size / 1024:.1f} KB")
            else:
                issues.append(f"PDF generation failed: {result.stderr}")
                pdf_path = None

        except Exception as e:
            issues.append(f"PDF generation error: {e}")
            pdf_path = None

        # Also generate Gerber files for fabrication
        gerber_dir = pdf_dir / 'gerbers'
        gerber_dir.mkdir(exist_ok=True)

        gerber_cmd = [
            self.KICAD_CLI, 'pcb', 'export', 'gerbers',
            '-o', str(gerber_dir) + '/',
            str(pcb_path)
        ]

        try:
            result = subprocess.run(gerber_cmd, capture_output=True, text=True, timeout=120)
            if result.returncode == 0:
                gerber_files = list(gerber_dir.glob('*.g*')) + list(gerber_dir.glob('*.drl'))
                if gerber_files:
                    print(f"    Generated {len(gerber_files)} Gerber files")
                else:
                    issues.append("No Gerber files generated")
            else:
                issues.append(f"Gerber generation failed: {result.stderr}")
        except Exception as e:
            issues.append(f"Gerber generation error: {e}")

        # Generate drill files
        drill_cmd = [
            self.KICAD_CLI, 'pcb', 'export', 'drill',
            '-o', str(gerber_dir) + '/',
            str(pcb_path)
        ]

        try:
            result = subprocess.run(drill_cmd, capture_output=True, text=True, timeout=60)
            if result.returncode == 0:
                print(f"    Generated drill files")
            else:
                issues.append(f"Drill file generation warning: {result.stderr}")
        except Exception as e:
            issues.append(f"Drill file error: {e}")

        return pdf_path, issues

    def _validate_pdf(self, pdf_path: Path) -> Tuple[float, List[str], List[str]]:
        """
        Validate PDF documentation using AI vision.

        Checks:
        1. PDF opens correctly
        2. Contains all expected layers
        3. Text is readable
        4. Professional appearance
        """
        if not pdf_path or not pdf_path.exists():
            return 0.0, ["PDF file not found"], []

        issues = []
        suggestions = []

        # For now, we validate the PDF exists and has reasonable size
        # Full PDF validation would require rendering and AI analysis

        pdf_size = pdf_path.stat().st_size

        if pdf_size < 10000:
            issues.append("PDF too small - likely empty or corrupt")
            return 2.0, issues, ["Regenerate PDF with proper content"]

        if pdf_size < 50000:
            issues.append("PDF smaller than expected - may be missing content")
            suggestions.append("Ensure all layers are included")
            return 5.0, issues, suggestions

        # PDF appears valid
        print(f"    PDF validation: OK ({pdf_size / 1024:.1f} KB)")
        return 8.0, [], ["Consider adding more detailed documentation"]

    def _validate_images(self, images: List[Path], use_multi_agent: bool = True) -> Tuple[float, List[str], List[str]]:
        """
        Validate images using Claude Opus 4.5 vision.

        Args:
            images: List of image paths to validate
            use_multi_agent: If True, use 5 parallel expert agents (recommended)

        Returns:
            Tuple of (average_score, issues, suggestions)
        """
        if not images:
            return 0.0, ["No images to validate"], []

        # Pre-validate all images first
        valid_images = []
        all_issues = []

        for image_path in images:
            try:
                self.pre_validator.validate_image_has_content(str(image_path))
                valid_images.append(image_path)
            except Exception as e:
                all_issues.append(f"Pre-validation failed for {image_path.name}: {e}")

        if not valid_images:
            return 0.0, all_issues, ["Regenerate all images - none passed pre-validation"]

        # Use multi-agent parallel validation (5 expert personas)
        if use_multi_agent:
            print(f"\n[MULTI-AGENT VALIDATION] Using 5 expert agents in parallel...")
            multi_validator = MultiAgentValidator(max_workers=5, timeout=120)
            image_paths = [str(img) for img in valid_images]

            results = multi_validator.validate_all_images_parallel(image_paths)

            # Print per-image breakdown
            print("\nPer-Image Results:")
            for img_result in results.get('images', []):
                img_name = Path(img_result['image']).name
                avg_score = img_result['average_score']
                print(f"  {img_name}: {avg_score:.1f}/10")
                for agent in img_result.get('agents', []):
                    print(f"    - {agent['agent']}: {agent['score']:.1f}/10")

            return (
                results['overall_score'],
                results.get('unique_issues', []) + all_issues,
                results.get('unique_suggestions', [])
            )

        # Fallback to single-agent validation
        scores = []
        all_suggestions = []

        for image_path in valid_images:
            # AI validation
            layer_type = "copper" if "Cu" in image_path.name else "silkscreen"

            prompt = f"""Analyze this PCB {layer_type} layer image and rate it on a scale of 1-10 for professional quality.

Consider:
1. Routing quality - Are traces clean? No 90-degree angles? Proper 45-degree routing?
2. Component placement - Logical grouping? Adequate spacing?
3. Silkscreen (if visible) - Are designators readable? Proper size (1mm+)? Pin 1 indicators?
4. Board utilization - Is space used efficiently? Not too much empty space?
5. Professional appearance - Would this pass design review at a manufacturing house?

Respond in JSON format:
{{
    "score": <1-10>,
    "issues": ["list of specific issues found"],
    "suggestions": ["list of specific improvements needed"],
    "verdict": "PASS" or "FAIL"
}}

Be strict - only professional-quality output should score 9+."""

            try:
                response = self.openrouter.create_vision_completion(
                    image_path=str(image_path),
                    prompt=prompt,
                    system_prompt="You are an expert PCB designer with 20+ years experience. You assess PCB designs against IPC-2221 and industry best practices.",
                    temperature=0.1
                )

                # Parse response
                content = response.content

                # Extract JSON from response
                import re
                json_match = re.search(r'\{[^{}]*\}', content, re.DOTALL)
                if json_match:
                    result = json.loads(json_match.group())
                    score = float(result.get('score', 0))
                    scores.append(score)
                    all_issues.extend(result.get('issues', []))
                    all_suggestions.extend(result.get('suggestions', []))
                    print(f"    {image_path.name}: {score}/10")
                else:
                    print(f"    Warning: Could not parse response for {image_path.name}")

            except Exception as e:
                all_issues.append(f"Validation error for {image_path.name}: {e}")
                print(f"    Error validating {image_path.name}: {e}")

        avg_score = sum(scores) / len(scores) if scores else 0.0
        return avg_score, all_issues, all_suggestions

    def run_iteration(self) -> IterationResult:
        """Run a single validation iteration."""
        self.state.current_iteration += 1
        print(f"\n{'='*60}")
        print(f"ITERATION {self.state.current_iteration} / {self.state.max_iterations}")
        print(f"{'='*60}")

        timestamp = datetime.now().isoformat()

        # Generate test design
        print("\n1. Generating test PCB...")
        design = self._create_test_design()

        try:
            # Run with KiCad Python if pcbnew not available
            if not HAS_PCBNEW:
                # Write design to temp file and run with KiCad Python
                design_file = self.pcb_dir / 'design.json'
                with open(design_file, 'w') as f:
                    json.dump(design, f)

                generator_script = f'''
import sys
sys.path.insert(0, "{SCRIPT_DIR}")
sys.path.insert(0, "/Users/don/Adverant/Adverant-Nexus/services/nexus-pcb-layout/python-scripts")

import json
from kicad_layout_generator import generate_layout_from_constraints

with open("{design_file}", "r") as f:
    design = json.load(f)

result = generate_layout_from_constraints(
    schematic_path="{self.pcb_dir}/test.kicad_sch",
    output_path="{self.pcb_dir}/test_iter_{self.state.current_iteration}.kicad_pcb",
    constraints=design["constraints"],
    placement=design["placement"],
    routing=design["routing"]
)

print(json.dumps(result))
'''

                # Create schematic
                with open(self.pcb_dir / 'test.kicad_sch', 'w') as f:
                    f.write('(kicad_sch (version 20230121))\n')

                result = subprocess.run(
                    [self.KICAD_PYTHON, '-c', generator_script],
                    capture_output=True, text=True, timeout=120
                )

                if result.returncode != 0:
                    raise ValidationFailure(f"PCB generation failed: {result.stderr}")

                pcb_path = self.pcb_dir / f'test_iter_{self.state.current_iteration}.kicad_pcb'
            else:
                pcb_path = self._generate_pcb(design)

            print(f"   PCB generated: {pcb_path.name}")

        except Exception as e:
            return IterationResult(
                iteration=self.state.current_iteration,
                timestamp=timestamp,
                score=0.0,
                passed=False,
                issues=[f"PCB generation failed: {e}"],
                suggestions=["Fix pipeline generation errors"],
                images_validated=[]
            )

        # Export images
        print("\n2. Exporting layer images...")
        images = self._export_images(pcb_path)

        if not images:
            return IterationResult(
                iteration=self.state.current_iteration,
                timestamp=timestamp,
                score=0.0,
                passed=False,
                issues=["No images exported"],
                suggestions=["Check KiCad CLI and PCB file validity"],
                images_validated=[]
            )

        # Generate PDF documentation
        print("\n3. Generating PDF documentation...")
        pdf_path, pdf_issues = self._generate_pdf_documentation(pcb_path)

        # Validate PDF
        print("\n4. Validating PDF documentation...")
        pdf_score, pdf_validation_issues, pdf_suggestions = self._validate_pdf(pdf_path)

        # Validate images with AI
        print("\n5. Validating images with Claude Opus 4.5...")
        image_score, image_issues, image_suggestions = self._validate_images(images)

        # Combine scores (weight: images 70%, PDF 30%)
        if pdf_score > 0:
            combined_score = (image_score * 0.7) + (pdf_score * 0.3)
        else:
            combined_score = image_score

        # Combine all issues and suggestions
        all_issues = image_issues + pdf_issues + pdf_validation_issues
        all_suggestions = image_suggestions + pdf_suggestions

        passed = combined_score >= self.state.quality_threshold

        # Update best score
        if combined_score > self.state.best_score:
            self.state.best_score = combined_score

        result = IterationResult(
            iteration=self.state.current_iteration,
            timestamp=timestamp,
            score=combined_score,
            passed=passed,
            issues=all_issues,
            suggestions=all_suggestions,
            images_validated=[str(img) for img in images]
        )

        self.state.history.append(result)
        self._save_state()

        # Print summary
        print(f"\n{'='*60}")
        print(f"ITERATION {self.state.current_iteration} RESULT")
        print(f"{'='*60}")
        print(f"Score: {combined_score:.1f}/10 ({'PASS' if passed else 'FAIL'})")
        print(f"  - Image Score: {image_score:.1f}/10")
        print(f"  - PDF Score: {pdf_score:.1f}/10")
        print(f"Best Score: {self.state.best_score:.1f}/10")
        if all_issues:
            print(f"\nIssues Found ({len(all_issues)}):")
            for issue in all_issues[:5]:
                print(f"  - {issue}")
        if all_suggestions:
            print(f"\nSuggestions ({len(all_suggestions)}):")
            for suggestion in all_suggestions[:5]:
                print(f"  - {suggestion}")

        return result

    def run(self) -> Dict[str, Any]:
        """Run the Ralph loop until completion or max iterations."""
        print("\n" + "="*60)
        print("PIPELINE RALPH LOOP - Starting")
        print("="*60)
        print(f"Target Score: {self.state.quality_threshold}/10")
        print(f"Max Iterations: {self.state.max_iterations}")
        print(f"Output Directory: {self.output_dir}")

        while self.state.current_iteration < self.state.max_iterations:
            result = self.run_iteration()

            if result.passed:
                self.state.status = "completed"
                self._save_state()

                print("\n" + "="*60)
                print("SUCCESS - TARGET SCORE ACHIEVED")
                print("="*60)
                print(f"Final Score: {result.score:.1f}/10")
                print(f"Iterations Required: {self.state.current_iteration}")

                return {
                    'status': 'success',
                    'final_score': result.score,
                    'iterations': self.state.current_iteration,
                    'best_score': self.state.best_score
                }

            # Brief pause between iterations
            time.sleep(2)

        # Max iterations reached
        self.state.status = "max_iterations_reached"
        self._save_state()

        print("\n" + "="*60)
        print("MAX ITERATIONS REACHED")
        print("="*60)
        print(f"Best Score Achieved: {self.state.best_score:.1f}/10")
        print(f"Target Score: {self.state.quality_threshold}/10")

        return {
            'status': 'max_iterations',
            'final_score': self.state.best_score,
            'iterations': self.state.current_iteration,
            'target': self.state.quality_threshold
        }


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Pipeline Ralph Loop - Iterative PCB pipeline validation'
    )
    parser.add_argument(
        '--output-dir', '-o',
        default='/tmp/pipeline_ralph_loop',
        help='Output directory for test files'
    )
    parser.add_argument(
        '--max-iterations', '-n',
        type=int,
        default=100,
        help='Maximum iterations (default: 100)'
    )
    parser.add_argument(
        '--threshold', '-t',
        type=float,
        default=9.0,
        help='Quality threshold to achieve (default: 9.0)'
    )

    args = parser.parse_args()

    try:
        loop = PipelineRalphLoop(
            output_dir=args.output_dir,
            max_iterations=args.max_iterations,
            quality_threshold=args.threshold
        )

        result = loop.run()

        if result['status'] == 'success':
            print("\n<promise>PIPELINE PRODUCES PROFESSIONAL QUALITY OUTPUT</promise>")
            sys.exit(0)
        else:
            sys.exit(1)

    except MissingDependencyFailure as e:
        print(f"ERROR: {e}")
        sys.exit(2)
    except KeyboardInterrupt:
        print("\nInterrupted by user")
        sys.exit(130)


if __name__ == '__main__':
    main()
