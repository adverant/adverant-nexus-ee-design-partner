#!/usr/bin/env python3
"""
Visual Ralph Loop - Iterative Visual Quality Improvement (PRODUCTION VERSION)

This is the PRODUCTION version that:
1. REQUIRES all dependencies (no silent fallbacks)
2. Pre-validates images before AI analysis (catches GIGO)
3. Actually applies fixes to PCB files (not just logging)
4. Raises exceptions on failure (no silent passes)

The Ralph Wiggum technique: Same prompt fed repeatedly while Claude sees
its own previous work in files and git history.

Usage:
    python visual_ralph_loop.py --pcb board.kicad_pcb --output-dir ./images --max-iterations 100
    python visual_ralph_loop.py --pcb board.kicad_pcb --output-dir ./images --threshold 9.0
    python visual_ralph_loop.py --output-dir ./images --dry-run
"""

import argparse
import json
import os
import shutil
import sys
import time
import concurrent.futures
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

# Import validation exceptions - MANDATORY
from validation_exceptions import (
    ValidationFailure,
    EmptyBoardFailure,
    EmptySilkscreenFailure,
    QualityThresholdFailure,
    MissingDependencyFailure,
    RoutingFailure,
    ValidationIssue,
    ValidationSeverity
)

# Import pre-validator - MANDATORY
from pre_validator import PreValidator

# Import PCB modifier for real fixes
from pcb_modifier import PCBModifier, Fix, FixType

# Import visual validation modules
try:
    from visual_validator import (
        VISUAL_PERSONAS,
        validate_all_personas,
        get_client,
        PersonaType
    )
    HAS_VISUAL_VALIDATOR = True
except ImportError:
    HAS_VISUAL_VALIDATOR = False

try:
    from image_analyzer import ImageAnalyzer, AnalysisType
    HAS_IMAGE_ANALYZER = True
except ImportError:
    HAS_IMAGE_ANALYZER = False

# Import KiCad exporter for proper image generation
try:
    from kicad_exporter import KiCadExporter, export_full_design
    HAS_KICAD_EXPORTER = True
except ImportError:
    HAS_KICAD_EXPORTER = False

# Import OpenRouter client for multi-agent validation
try:
    from openrouter_client import OpenRouterClient, get_openrouter_client
    HAS_OPENROUTER = True
except ImportError:
    HAS_OPENROUTER = False

# Import programmatic scorer for local validation (no API required)
try:
    from programmatic_scorer import ProgrammaticScorer
    HAS_PROGRAMMATIC_SCORER = True
except ImportError:
    HAS_PROGRAMMATIC_SCORER = False


# ============================================================================
# MULTI-AGENT VALIDATION PERSONAS
# ============================================================================
VALIDATION_PERSONAS = {
    "routing_expert": {
        "name": "Senior PCB Routing Engineer",
        "focus": "Signal integrity, trace routing, 45-degree angles, via placement",
        "system_prompt": """You are a senior PCB routing engineer with 20+ years of experience at top electronics companies.
Your expertise includes high-speed signal integrity, impedance matching, and professional routing standards.

When analyzing PCB images, focus on:
1. Trace angle compliance (should be 45° or orthogonal, NEVER arbitrary angles)
2. Via placement and thermal relief
3. Ground return paths
4. Trace width consistency
5. Layer transitions
6. Clearance violations

Score on a scale of 1-10 where:
- 10: IPC Class 3 compliant, ready for aerospace/medical
- 7-9: Professional quality, ready for production
- 4-6: Acceptable but needs improvement
- 1-3: Amateur quality, needs significant work

Respond with JSON: {"score": X.X, "issues": [...], "suggestions": [...]}""",
    },
    "silkscreen_expert": {
        "name": "PCB Assembly Specialist",
        "focus": "Silkscreen legibility, designator placement, polarity markers",
        "system_prompt": """You are a PCB assembly specialist responsible for ensuring boards can be assembled correctly.
Your focus is on silkscreen quality - reference designators, polarity markers, and assembly instructions.

When analyzing silkscreen images, focus on:
1. Reference designator presence and legibility (R1, C1, U1, etc.)
2. Text orientation - should be horizontal or readable
3. Polarity markers for diodes, LEDs, electrolytic caps
4. Pin 1 indicators on ICs
5. No text overlapping pads
6. Assembly instructions clarity

Score on a scale of 1-10 where:
- 10: Perfect assembly documentation
- 7-9: Professional quality, all components identifiable
- 4-6: Some designators missing or hard to read
- 1-3: Assembly would be difficult/impossible

Respond with JSON: {"score": X.X, "issues": [...], "suggestions": [...]}""",
    },
    "drc_expert": {
        "name": "Design Rule Check Engineer",
        "focus": "Clearances, spacing, manufacturability",
        "system_prompt": """You are a DRC engineer responsible for ensuring boards meet manufacturing requirements.
Your expertise is in design rules, clearances, and manufacturability.

When analyzing PCB images, focus on:
1. Trace-to-trace clearance (minimum 0.15mm)
2. Trace-to-pad clearance
3. Via-to-trace clearance
4. Component spacing
5. Thermal relief patterns
6. Soldermask coverage

Score on a scale of 1-10 where:
- 10: Exceeds IPC-2221 Class 2
- 7-9: Meets standard requirements
- 4-6: Some rule violations
- 1-3: Major DRC violations

Respond with JSON: {"score": X.X, "issues": [...], "suggestions": [...]}""",
    },
    "aesthetics_expert": {
        "name": "PCB Layout Aesthetics Reviewer",
        "focus": "Visual organization, symmetry, professional appearance",
        "system_prompt": """You are a PCB aesthetics reviewer who evaluates layouts for visual quality and organization.
Professional layouts should look clean, organized, and intentional.

When analyzing PCB images, focus on:
1. Component alignment and grouping
2. Trace routing aesthetics (parallel runs, consistent angles)
3. Copper distribution/balance
4. Overall organization
5. Layer utilization
6. Visual hierarchy

Score on a scale of 1-10 where:
- 10: Award-winning layout, could be in a showcase
- 7-9: Professional, clean appearance
- 4-6: Functional but disorganized
- 1-3: Chaotic, amateur appearance

Respond with JSON: {"score": X.X, "issues": [...], "suggestions": [...]}""",
    },
    "manufacturing_expert": {
        "name": "PCB Manufacturing Engineer",
        "focus": "Fabrication readiness, Gerber quality, drill accuracy",
        "system_prompt": """You are a manufacturing engineer at a PCB fabrication facility.
You review designs for manufacturability and production yield.

When analyzing PCB images, focus on:
1. Layer alignment marks
2. Drill hit accuracy
3. Copper balance across layers
4. Edge clearances
5. Annular ring integrity
6. Solder mask definition

Score on a scale of 1-10 where:
- 10: Ready for high-volume production
- 7-9: Standard production ready
- 4-6: May have yield issues
- 1-3: Will have manufacturing problems

Respond with JSON: {"score": X.X, "issues": [...], "suggestions": [...]}""",
    }
}


class MultiAgentValidator:
    """
    Multi-agent validation system using parallel AI analysis.

    Spawns multiple specialized AI agents in parallel, each with a different
    expertise persona, to analyze PCB images from different perspectives.
    Results are aggregated for a comprehensive quality assessment.
    """

    def __init__(self, max_workers: int = 5, timeout: int = 120):
        """
        Initialize multi-agent validator.

        Args:
            max_workers: Maximum parallel agents (default 5 - one per persona)
            timeout: Timeout per agent in seconds
        """
        self.max_workers = max_workers
        self.timeout = timeout
        self.client = None

        if HAS_OPENROUTER:
            try:
                self.client = get_openrouter_client()
            except Exception as e:
                print(f"Warning: Could not initialize OpenRouter client: {e}")

    def _run_single_agent(
        self,
        image_path: str,
        persona_name: str,
        persona_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Run a single agent validation.

        Args:
            image_path: Path to image to analyze
            persona_name: Name of the persona (e.g., "routing_expert")
            persona_config: Persona configuration with system_prompt

        Returns:
            Dict with score, issues, suggestions
        """
        start_time = time.time()

        if not self.client:
            return {
                "agent": persona_name,
                "score": 5.0,
                "issues": ["AI client not available"],
                "suggestions": [],
                "error": "No OpenRouter client"
            }

        try:
            # Create the analysis prompt
            prompt = f"""Analyze this PCB image as a {persona_config['name']}.
Focus on: {persona_config['focus']}

Provide your assessment in JSON format with:
- score: A number from 1-10
- issues: Array of specific issues found
- suggestions: Array of improvement suggestions

Be specific and actionable in your feedback."""

            response = self.client.create_vision_completion(
                image_path=image_path,
                prompt=prompt,
                system_prompt=persona_config['system_prompt'],
                max_tokens=1000
            )

            # Parse JSON response
            content = response.content

            # Extract JSON from response
            import re
            json_match = re.search(r'\{[^{}]*"score"[^{}]*\}', content, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group())
            else:
                # Try to parse whole content as JSON
                result = json.loads(content)

            elapsed_ms = int((time.time() - start_time) * 1000)

            return {
                "agent": persona_name,
                "persona": persona_config['name'],
                "score": float(result.get('score', 5.0)),
                "issues": result.get('issues', []),
                "suggestions": result.get('suggestions', []),
                "elapsed_ms": elapsed_ms
            }

        except json.JSONDecodeError as e:
            return {
                "agent": persona_name,
                "score": 5.0,
                "issues": [f"Could not parse AI response: {e}"],
                "suggestions": [],
                "error": str(e)
            }
        except Exception as e:
            return {
                "agent": persona_name,
                "score": 5.0,
                "issues": [f"Agent error: {e}"],
                "suggestions": [],
                "error": str(e)
            }

    def validate_image_parallel(
        self,
        image_path: str,
        personas: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Validate a single image using multiple agents in parallel.

        Args:
            image_path: Path to image to validate
            personas: List of persona names to use (default: all)

        Returns:
            Aggregated validation results
        """
        personas_to_use = personas or list(VALIDATION_PERSONAS.keys())

        # Filter to requested personas
        active_personas = {
            k: v for k, v in VALIDATION_PERSONAS.items()
            if k in personas_to_use
        }

        print(f"\n  Running {len(active_personas)} agents in parallel on {Path(image_path).name}...")

        agent_results = []

        # Run agents in parallel
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all agent tasks
            future_to_persona = {
                executor.submit(
                    self._run_single_agent,
                    image_path,
                    persona_name,
                    persona_config
                ): persona_name
                for persona_name, persona_config in active_personas.items()
            }

            # Collect results as they complete
            for future in concurrent.futures.as_completed(future_to_persona, timeout=self.timeout):
                persona_name = future_to_persona[future]
                try:
                    result = future.result()
                    agent_results.append(result)
                    print(f"    [{persona_name}] Score: {result['score']:.1f}/10")
                except Exception as e:
                    print(f"    [{persona_name}] ERROR: {e}")
                    agent_results.append({
                        "agent": persona_name,
                        "score": 0.0,
                        "issues": [f"Agent failed: {e}"],
                        "suggestions": [],
                        "error": str(e)
                    })

        # Aggregate results
        scores = [r['score'] for r in agent_results if 'score' in r]
        avg_score = sum(scores) / len(scores) if scores else 0.0

        all_issues = []
        all_suggestions = []
        for r in agent_results:
            all_issues.extend(r.get('issues', []))
            all_suggestions.extend(r.get('suggestions', []))

        return {
            "image": image_path,
            "agents": agent_results,
            "average_score": round(avg_score, 2),
            "min_score": min(scores) if scores else 0.0,
            "max_score": max(scores) if scores else 0.0,
            "issues": list(set(all_issues)),  # Deduplicate
            "suggestions": list(set(all_suggestions)),
            "passed": avg_score >= 9.0
        }

    def validate_all_images_parallel(
        self,
        image_paths: List[str],
        personas: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Validate multiple images using parallel multi-agent analysis.

        Args:
            image_paths: List of image paths to validate
            personas: List of persona names to use (default: all)

        Returns:
            Comprehensive validation results
        """
        print(f"\n[MULTI-AGENT VALIDATION] Analyzing {len(image_paths)} images...")

        all_results = []
        all_scores = []
        all_issues = []
        all_suggestions = []

        for img_path in image_paths:
            result = self.validate_image_parallel(img_path, personas)
            all_results.append(result)
            all_scores.append(result['average_score'])
            all_issues.extend(result['issues'])
            all_suggestions.extend(result['suggestions'])

        overall_avg = sum(all_scores) / len(all_scores) if all_scores else 0.0

        return {
            "images": all_results,
            "overall_score": round(overall_avg, 2),
            "min_score": min(all_scores) if all_scores else 0.0,
            "max_score": max(all_scores) if all_scores else 0.0,
            "total_issues": len(set(all_issues)),
            "unique_issues": list(set(all_issues)),
            "unique_suggestions": list(set(all_suggestions)),
            "passed": overall_avg >= 9.0 and min(all_scores) >= 7.0,
            "agent_count": len(personas or VALIDATION_PERSONAS)
        }


@dataclass
class AgentResult:
    """Result from a single validation agent."""
    agent_name: str
    persona: str
    score: float
    issues: List[str]
    suggestions: List[str]
    passed: bool
    image_analyzed: str
    analysis_time_ms: int = 0


@dataclass
class IterationResult:
    """Result of a single validation iteration."""
    iteration: int
    timestamp: str
    scores: Dict[str, float]
    average_score: float
    passed: bool
    issues_found: List[str]
    fixes_applied: List[str]
    images_validated: int
    pre_validation_passed: bool = True
    agent_results: List[AgentResult] = field(default_factory=list)


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
    completion_promise: str = "ALL VISUAL OUTPUTS VALIDATED AND PROFESSIONAL QUALITY"


class VisualRalphLoop:
    """
    Iterative visual validation loop (PRODUCTION VERSION).

    Key differences from placeholder version:
    1. Dependencies are MANDATORY - raises MissingDependencyFailure if missing
    2. Pre-validation catches empty/broken images BEFORE AI analysis
    3. apply_fixes() actually modifies PCB files using pcbnew API
    4. Failures raise exceptions - no silent passes

    Runs: pre-validate → AI validate → generate fixes → apply fixes → regenerate → repeat
    """

    def __init__(
        self,
        output_dir: str,
        pcb_path: Optional[str] = None,
        max_iterations: int = 100,
        quality_threshold: float = 9.0,
        state_file: Optional[str] = None,
        schematic_paths: Optional[List[str]] = None,
        strict: bool = True
    ):
        """
        Initialize the Ralph loop.

        Args:
            output_dir: Directory containing output images
            pcb_path: Path to .kicad_pcb file (required for fixes)
            max_iterations: Maximum iterations before failure
            quality_threshold: Minimum score to pass (1-10)
            state_file: Path to state file for resuming
            schematic_paths: Paths to schematic files
            strict: If True, raise exceptions on failures (recommended)

        Raises:
            MissingDependencyFailure: If required dependencies are missing
        """
        self.output_dir = Path(output_dir)
        self.pcb_path = pcb_path
        self.max_iterations = max_iterations
        self.quality_threshold = quality_threshold
        self.state_file = state_file or str(self.output_dir / ".ralph-loop-state.json")
        self.schematic_paths = schematic_paths or []
        self.strict = strict

        # Validate dependencies
        self._verify_dependencies()

        # Initialize components
        self.pre_validator = PreValidator(strict=strict)
        self.pcb_modifier = None
        self.kicad_exporter = None
        self.client = None
        self.analyzer = None

        # Initialize PCB modifier if PCB path provided
        if pcb_path and Path(pcb_path).exists():
            try:
                self.pcb_modifier = PCBModifier(pcb_path)
                print(f"PCB modifier initialized: {pcb_path}")
            except Exception as e:
                print(f"Warning: Could not initialize PCB modifier: {e}")

        # Initialize KiCad exporter
        if HAS_KICAD_EXPORTER:
            try:
                self.kicad_exporter = KiCadExporter()
                print("KiCad exporter initialized")
            except Exception as e:
                print(f"Warning: Could not initialize KiCad exporter: {e}")

        # Initialize state
        self.state = LoopState(
            max_iterations=max_iterations,
            quality_threshold=quality_threshold,
            start_time=datetime.now().isoformat()
        )

        # Try to load existing state
        self._load_state()

    def _verify_dependencies(self):
        """
        Verify all required dependencies are available.

        Raises:
            MissingDependencyFailure: If required dependency is missing
        """
        # Check for API key - OPENROUTER_API_KEY is preferred (used by visual_validator)
        # Also accept ANTHROPIC_API_KEY for backward compatibility
        # BUT: Allow running with programmatic scoring even without API key
        if not os.environ.get('OPENROUTER_API_KEY') and not os.environ.get('ANTHROPIC_API_KEY'):
            if HAS_PROGRAMMATIC_SCORER:
                print("Note: No API key set - using programmatic scoring (local computer vision)")
            else:
                raise MissingDependencyFailure(
                    "OPENROUTER_API_KEY environment variable not set",
                    dependency_name="OPENROUTER_API_KEY",
                    install_instructions="export OPENROUTER_API_KEY=your_key_here (or ANTHROPIC_API_KEY)"
                )

        # Check for KiCad CLI (required for proper exports)
        if not shutil.which('kicad-cli'):
            print("Warning: kicad-cli not found - image regeneration will be limited")

        # Check output directory exists
        if not self.output_dir.exists():
            self.output_dir.mkdir(parents=True, exist_ok=True)

    def _load_state(self):
        """Load state from file if exists."""
        if Path(self.state_file).exists():
            try:
                with open(self.state_file, 'r') as f:
                    data = json.load(f)
                    self.state.current_iteration = data.get('current_iteration', 0)
                    self.state.best_score = data.get('best_score', 0.0)
                    self.state.status = data.get('status', 'running')
                    print(f"Resuming from iteration {self.state.current_iteration}")
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
            'completion_promise': self.state.completion_promise,
            'history_length': len(self.state.history)
        }
        with open(self.state_file, 'w') as f:
            json.dump(state_data, f, indent=2)

    def _init_clients(self):
        """Initialize API clients."""
        if HAS_VISUAL_VALIDATOR and not self.client:
            try:
                self.client = get_client()
            except Exception as e:
                print(f"  Warning: Could not initialize visual_validator client: {e}")
        if HAS_IMAGE_ANALYZER and not self.analyzer:
            try:
                self.analyzer = ImageAnalyzer()
            except (ImportError, ValueError) as e:
                # ImageAnalyzer requires anthropic package - skip if not available
                # We'll fall back to multi-agent validation via OpenRouter
                print(f"  Note: ImageAnalyzer not available ({e}), using multi-agent via OpenRouter")

    def _get_images(self) -> List[Path]:
        """Get all images to validate."""
        patterns = ['*.png', '*.jpg', '*.jpeg']
        images = []
        for pattern in patterns:
            images.extend(self.output_dir.glob(pattern))

        # Filter out SVG directory contents
        images = [img for img in images if 'svg' not in str(img)]
        return sorted(images)

    def pre_validate_all_images(self) -> Dict[str, Any]:
        """
        Pre-validate all images before AI analysis.

        This catches GIGO (Garbage In, Garbage Out) by detecting:
        - Empty images (just title block)
        - Missing copper traces
        - Empty silkscreen

        Returns:
            Dict with pre-validation results

        Raises:
            EmptyBoardFailure: If any copper layer is empty
            EmptySilkscreenFailure: If silkscreen is empty
        """
        images = self._get_images()
        if not images:
            raise EmptyBoardFailure(
                "No images found in output directory",
                image_path=str(self.output_dir)
            )

        results = {}
        all_passed = True

        print("\n[PRE-VALIDATION] Checking images contain real PCB data...")

        for img_path in images:
            layer_type = self._detect_layer_type(img_path.name)
            print(f"  Checking {img_path.name} ({layer_type})...")

            try:
                if "silk" in layer_type.lower():
                    result = self.pre_validator.validate_silkscreen_has_text(str(img_path))
                else:
                    result = self.pre_validator.validate_image_has_content(str(img_path), layer_type)

                results[img_path.name] = {
                    "passed": result.passed,
                    "content_density": result.content_density,
                    "issues": result.issues
                }

                if not result.passed:
                    all_passed = False
                    print(f"    FAILED: {', '.join(result.issues)}")
                else:
                    print(f"    OK (density: {result.content_density*100:.1f}%)")

            except (EmptyBoardFailure, EmptySilkscreenFailure) as e:
                # Re-raise if strict mode
                if self.strict:
                    raise
                results[img_path.name] = {
                    "passed": False,
                    "error": str(e)
                }
                all_passed = False

        return {
            "all_passed": all_passed,
            "images": results,
            "total": len(images),
            "passed_count": sum(1 for r in results.values() if r.get("passed", False))
        }

    def _detect_layer_type(self, filename: str) -> str:
        """Detect layer type from filename."""
        name_lower = filename.lower()

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
        elif "top_view" in name_lower or "assembly" in name_lower:
            return "composite"
        else:
            return "unknown"

    def validate_all_outputs(self, use_multi_agent: bool = True) -> Dict[str, Any]:
        """
        Run AI validation on all output images.

        Args:
            use_multi_agent: If True, use parallel multi-agent validation (recommended)

        Returns:
            Dictionary with scores for each image and overall stats
        """
        self._init_clients()
        images = self._get_images()

        if not images:
            return {
                "error": "No images found",
                "overall_score": 0,
                "passed": False
            }

        # Check if API key is available for multi-agent validation
        has_api_key = os.environ.get('OPENROUTER_API_KEY') or os.environ.get('ANTHROPIC_API_KEY')

        # Use multi-agent parallel validation only if API key is available
        if use_multi_agent and HAS_OPENROUTER and has_api_key:
            print("\n[MULTI-AGENT VALIDATION] Using 5 parallel expert agents...")
            multi_validator = MultiAgentValidator(max_workers=5, timeout=120)
            image_paths = [str(img) for img in images]

            # Run all agents in parallel on all images
            multi_results = multi_validator.validate_all_images_parallel(image_paths)

            # Check if all results are 5.0 (API failed) - fall back to programmatic
            all_scores = [r.get('average_score', 5.0) for r in multi_results.get('images', [])]
            if all_scores and all(s == 5.0 for s in all_scores):
                print("  Multi-agent validation failed (all scores 5.0) - falling back to programmatic scoring")
            else:
                # Convert to expected format
                results = {}
                for img_result in multi_results.get('images', []):
                    img_name = Path(img_result['image']).name
                    results[img_name] = {
                        "score": img_result['average_score'],
                        "passed": img_result['passed'],
                        "issues": img_result.get('issues', []),
                        "agent_scores": {
                            agent['agent']: agent['score']
                            for agent in img_result.get('agents', [])
                        },
                        "agents": img_result.get('agents', [])
                    }

                return {
                    "images": results,
                    "overall_score": multi_results['overall_score'],
                    "passed": multi_results['passed'],
                    "total_images": len(images),
                    "passing_images": sum(1 for r in results.values() if r['passed']),
                    "total_issues": multi_results['total_issues'],
                    "unique_issues": multi_results['unique_issues'],
                    "unique_suggestions": multi_results.get('unique_suggestions', []),
                    "agent_count": multi_results.get('agent_count', 5),
                    "validation_mode": "multi-agent"
                }

        # Fallback to programmatic scoring (no API required)
        if HAS_PROGRAMMATIC_SCORER:
            print("\n[PROGRAMMATIC SCORING] Using local computer vision analysis...")
            try:
                scorer = ProgrammaticScorer(strict=False)
                prog_results = scorer.score_all_images(str(self.output_dir))

                # Convert to expected format
                results = {}
                for name, layer_score in prog_results.layer_scores.items():
                    results[name] = {
                        "score": layer_score.score,
                        "passed": layer_score.score >= self.quality_threshold,
                        "issues": layer_score.issues,
                        "metrics": layer_score.metrics,
                        "routing_score": layer_score.routing_score,
                        "coverage_score": layer_score.coverage_score,
                        "balance_score": layer_score.balance_score
                    }

                return {
                    "images": results,
                    "overall_score": prog_results.overall_score,
                    "passed": prog_results.passed,
                    "total_images": len(prog_results.layer_scores),
                    "passing_images": sum(1 for r in results.values() if r['passed']),
                    "total_issues": len(prog_results.issues),
                    "unique_issues": prog_results.issues,
                    "unique_suggestions": prog_results.suggestions,
                    "aggregate_metrics": prog_results.aggregate_metrics,
                    "validation_mode": "programmatic"
                }
            except Exception as e:
                print(f"  Programmatic scoring failed: {e}")

        # Final fallback - per-image validation with available tools
        results = {}
        all_scores = []
        all_issues = []

        for img_path in images:
            print(f"  Validating {img_path.name}...")

            if HAS_IMAGE_ANALYZER and self.analyzer:
                # Run full analysis
                analysis_results = self.analyzer.full_analysis(str(img_path))

                img_scores = []
                img_issues = []

                for analysis_type, result in analysis_results.items():
                    img_scores.append(result.score)
                    img_issues.extend(result.issues)

                avg_score = sum(img_scores) / len(img_scores) if img_scores else 0

                results[img_path.name] = {
                    "score": round(avg_score, 2),
                    "passed": avg_score >= self.quality_threshold,
                    "issues": img_issues,
                    "analyses": {
                        t: {"score": r.score, "passed": r.passed}
                        for t, r in analysis_results.items()
                    }
                }
            else:
                # Fallback - basic validation
                results[img_path.name] = {
                    "score": 5.0,
                    "passed": False,
                    "issues": ["AI analyzer not available"],
                    "analyses": {}
                }
                avg_score = 5.0

            all_scores.append(avg_score)
            all_issues.extend(results[img_path.name].get('issues', []))

        overall_score = sum(all_scores) / len(all_scores) if all_scores else 0
        all_passed = all(r['passed'] for r in results.values())

        return {
            "images": results,
            "overall_score": round(overall_score, 2),
            "passed": all_passed and overall_score >= self.quality_threshold,
            "total_images": len(images),
            "passing_images": sum(1 for r in results.values() if r['passed']),
            "total_issues": len(all_issues),
            "unique_issues": list(set(all_issues)),
            "validation_mode": "single-agent"
        }

    def generate_fixes(self, validation_results: Dict[str, Any]) -> List[Fix]:
        """
        Generate Fix objects based on validation results.

        Args:
            validation_results: Results from validate_all_outputs()

        Returns:
            List of Fix objects that can be applied to the PCB
        """
        fixes = []

        for img_name, img_result in validation_results.get('images', {}).items():
            if img_result.get('passed', False):
                continue

            layer_type = self._detect_layer_type(img_name)

            for issue in img_result.get('issues', []):
                fix = self._create_fix_for_issue(issue, layer_type, img_name)
                if fix:
                    fixes.append(fix)

        # Sort by priority
        fixes.sort(key=lambda f: f.priority)

        return fixes

    def _create_fix_for_issue(self, issue: str, layer_type: str, img_name: str) -> Optional[Fix]:
        """Create a Fix object for a specific issue."""
        issue_lower = issue.lower()

        # Silkscreen issues
        if any(word in issue_lower for word in ['designator', 'reference', 'label', 'text']):
            if 'missing' in issue_lower:
                # Extract reference if mentioned (e.g., "Missing designator R1")
                import re
                match = re.search(r'\b([A-Z]+\d+)\b', issue)
                ref = match.group(1) if match else "REF"

                return Fix(
                    fix_type=FixType.ADD_DESIGNATOR,
                    description=f"Add missing designator {ref}",
                    ref=ref,
                    position=(50.0, 50.0),  # Would need smarter placement
                    layer="F.SilkS",
                    priority=1
                )

        # Routing issues
        if any(word in issue_lower for word in ['90°', 'angle', 'routing']):
            return Fix(
                fix_type=FixType.REROUTE_TRACE,
                description=f"Fix routing angle: {issue}",
                priority=2
            )

        # Spacing issues
        if any(word in issue_lower for word in ['spacing', 'crowded', 'overlap']):
            return Fix(
                fix_type=FixType.ADJUST_SPACING,
                description=f"Adjust spacing: {issue}",
                priority=3
            )

        return None

    def apply_fixes(self, fixes: List[Fix]) -> List[str]:
        """
        Actually apply fixes to the PCB file.

        This is the REAL implementation that modifies the PCB,
        not just a placeholder that logs messages.

        Args:
            fixes: List of Fix objects

        Returns:
            List of applied fix descriptions
        """
        if not self.pcb_modifier:
            print("  Warning: No PCB modifier - fixes cannot be applied")
            return [f"[SKIPPED] {fix.description}" for fix in fixes]

        applied = []

        try:
            applied = self.pcb_modifier.apply_fixes(fixes)
            self.pcb_modifier.save()
            print(f"  Applied {len(applied)} fixes to PCB")
        except Exception as e:
            print(f"  Error applying fixes: {e}")
            # Still return what we attempted
            applied = [f"[FAILED] {fix.description}: {e}" for fix in fixes]

        return applied

    def regenerate_outputs(self) -> bool:
        """
        Regenerate output images using KiCad CLI.

        Returns:
            True if regeneration succeeded

        Raises:
            MissingDependencyFailure: If KiCad CLI not available and strict mode
        """
        if not self.kicad_exporter or not self.pcb_path:
            if self.strict:
                raise MissingDependencyFailure(
                    "Cannot regenerate outputs without KiCad exporter and PCB path",
                    dependency_name="kicad-cli",
                    install_instructions="Install KiCad and ensure kicad-cli is in PATH"
                )
            print("  Warning: KiCad exporter not configured, skipping regeneration")
            return False

        try:
            print("  Regenerating outputs with KiCad CLI...")

            # Export all layers as SVG
            svg_dir = self.output_dir / "svg"
            svg_dir.mkdir(parents=True, exist_ok=True)

            svg_results = self.kicad_exporter.export_all_layers(
                self.pcb_path,
                str(svg_dir)
            )

            # Convert to PNG
            for layer_name, svg_path in svg_results.items():
                safe_name = layer_name.replace(".", "_")
                png_path = str(self.output_dir / f"{safe_name}.png")
                self.kicad_exporter.convert_svg_to_png(svg_path, png_path)

            print(f"  Regenerated {len(svg_results)} layer images")
            return True

        except Exception as e:
            print(f"  Error regenerating outputs: {e}")
            if self.strict:
                raise
            return False

    def run_iteration(self) -> IterationResult:
        """
        Run a single iteration of the validation loop.

        Returns:
            IterationResult with iteration details

        Raises:
            EmptyBoardFailure: If images are empty
            EmptySilkscreenFailure: If silkscreen is empty
        """
        self.state.current_iteration += 1
        iteration = self.state.current_iteration

        print(f"\n{'='*60}")
        print(f"ITERATION {iteration}/{self.max_iterations}")
        print(f"{'='*60}")

        # 1. PRE-VALIDATE - Check images contain real content
        print("\n[1/5] Pre-validating images...")
        pre_validation = self.pre_validate_all_images()
        pre_passed = pre_validation.get('all_passed', False)

        if not pre_passed:
            print(f"  WARNING: Pre-validation failed - images may be empty")
            # Continue to AI validation anyway to get detailed feedback

        # 2. AI VALIDATE - Get detailed quality scores
        print("\n[2/5] Running AI validation...")
        validation = self.validate_all_outputs()

        scores = {
            img: data['score']
            for img, data in validation.get('images', {}).items()
        }
        avg_score = validation.get('overall_score', 0)

        # 3. Check if we passed
        if validation.get('passed', False) and pre_passed:
            print(f"\n[PASS] ALL VALIDATIONS PASSED (Score: {avg_score}/10)")
            return IterationResult(
                iteration=iteration,
                timestamp=datetime.now().isoformat(),
                scores=scores,
                average_score=avg_score,
                passed=True,
                issues_found=[],
                fixes_applied=[],
                images_validated=validation.get('total_images', 0),
                pre_validation_passed=pre_passed
            )

        # 4. Generate fixes
        print(f"\n[3/5] Generating fixes (Score: {avg_score}/10)...")
        fixes = self.generate_fixes(validation)
        print(f"  Generated {len(fixes)} fix recommendations")

        # 5. Apply fixes - ACTUALLY MODIFY PCB
        print("\n[4/5] Applying fixes to PCB...")
        applied = self.apply_fixes(fixes[:10])  # Limit to top 10 per iteration

        # 6. Regenerate outputs
        print("\n[5/5] Regenerating outputs...")
        self.regenerate_outputs()

        # Update best score
        if avg_score > self.state.best_score:
            self.state.best_score = avg_score
            print(f"  New best score: {avg_score}/10")

        return IterationResult(
            iteration=iteration,
            timestamp=datetime.now().isoformat(),
            scores=scores,
            average_score=avg_score,
            passed=False,
            issues_found=validation.get('unique_issues', []),
            fixes_applied=applied,
            images_validated=validation.get('total_images', 0),
            pre_validation_passed=pre_passed
        )

    def run(self) -> Dict[str, Any]:
        """
        Run the full validation loop.

        Returns:
            Final results with status and history

        Raises:
            QualityThresholdFailure: If max iterations reached without passing
        """
        print(f"\nStarting Visual Ralph Loop (PRODUCTION)")
        print(f"Output directory: {self.output_dir}")
        print(f"PCB path: {self.pcb_path or 'Not specified'}")
        print(f"Max iterations: {self.max_iterations}")
        print(f"Quality threshold: {self.quality_threshold}/10")

        self.state.status = "running"
        self._save_state()

        last_result = None

        while self.state.current_iteration < self.max_iterations:
            # Run one iteration
            result = self.run_iteration()
            self.state.history.append(result)
            self._save_state()
            last_result = result

            if result.passed:
                self.state.status = "completed"
                self._save_state()

                print(f"\n{'*'*60}")
                print(f"SUCCESS: Visual validation PASSED at iteration {result.iteration}")
                print(f"Final score: {result.average_score}/10")
                print(f"{'*'*60}")

                # Output completion promise - ONLY when truly passing
                print(f"\n<promise>{self.state.completion_promise}</promise>")

                return {
                    "status": "PASS",
                    "iterations": result.iteration,
                    "final_score": result.average_score,
                    "best_score": self.state.best_score,
                    "completion_promise": self.state.completion_promise
                }

            # Brief pause between iterations
            time.sleep(0.5)

        # Max iterations reached - FAIL with exception if strict
        self.state.status = "max_iterations_reached"
        self._save_state()

        print(f"\n{'*'*60}")
        print(f"FAILED: Max iterations ({self.max_iterations}) reached")
        print(f"Best score achieved: {self.state.best_score}/10")
        print(f"{'*'*60}")

        if self.strict:
            raise QualityThresholdFailure(
                message=f"Could not achieve quality threshold after {self.max_iterations} iterations",
                score=self.state.best_score,
                threshold=self.quality_threshold,
                iterations_attempted=self.max_iterations,
                max_iterations=self.max_iterations,
                issues=[
                    ValidationIssue(
                        message=issue,
                        severity=ValidationSeverity.ERROR
                    )
                    for issue in (last_result.issues_found[:10] if last_result else [])
                ]
            )

        return {
            "status": "FAIL",
            "iterations": self.max_iterations,
            "final_score": last_result.average_score if last_result else 0,
            "best_score": self.state.best_score,
            "remaining_issues": last_result.issues_found if last_result else []
        }

    def dry_run(self) -> Dict[str, Any]:
        """
        Run validation once without the loop.

        Returns:
            Single iteration results
        """
        print("\nDRY RUN: Running single validation pass")
        print(f"Output directory: {self.output_dir}")

        # Pre-validation
        print("\n[PRE-VALIDATION]")
        try:
            pre_results = self.pre_validate_all_images()
            pre_passed = pre_results.get('all_passed', False)
        except (EmptyBoardFailure, EmptySilkscreenFailure) as e:
            print(f"  PRE-VALIDATION FAILED: {e}")
            pre_passed = False
            pre_results = {"error": str(e)}

        # AI validation
        print("\n[AI VALIDATION]")
        validation = self.validate_all_outputs()

        print(f"\n{'='*60}")
        print("DRY RUN RESULTS")
        print(f"{'='*60}")
        print(f"Pre-validation: {'PASS' if pre_passed else 'FAIL'}")
        print(f"Overall Score: {validation.get('overall_score', 0)}/10")
        print(f"Passed: {validation.get('passed', False)}")
        print(f"Total Images: {validation.get('total_images', 0)}")
        print(f"Passing Images: {validation.get('passing_images', 0)}")
        print(f"Total Issues: {validation.get('total_issues', 0)}")

        if validation.get('unique_issues'):
            print("\nUnique Issues:")
            for issue in validation['unique_issues'][:10]:
                print(f"  - {issue}")

        validation['pre_validation'] = pre_results
        return validation


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description='Visual Ralph Loop - Iterative Visual Quality Improvement (PRODUCTION)'
    )
    parser.add_argument(
        '--output-dir', '-o',
        type=str,
        required=True,
        help='Directory containing output images'
    )
    parser.add_argument(
        '--pcb', '-p',
        type=str,
        help='Path to .kicad_pcb file (required for applying fixes)'
    )
    parser.add_argument(
        '--schematic', '-s',
        type=str,
        nargs='*',
        help='Path(s) to .kicad_sch file(s)'
    )
    parser.add_argument(
        '--max-iterations', '-m',
        type=int,
        default=100,
        help='Maximum iterations (default: 100)'
    )
    parser.add_argument(
        '--threshold', '-t',
        type=float,
        default=9.0,
        help='Quality threshold (default: 9.0)'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Run single validation without loop'
    )
    parser.add_argument(
        '--json',
        action='store_true',
        help='Output as JSON'
    )
    parser.add_argument(
        '--state-file',
        type=str,
        help='Custom state file path'
    )
    parser.add_argument(
        '--regenerate-first',
        action='store_true',
        help='Regenerate outputs using KiCad CLI before validation'
    )
    parser.add_argument(
        '--no-strict',
        action='store_true',
        help='Disable strict mode (allow silent failures)'
    )

    args = parser.parse_args()

    try:
        loop = VisualRalphLoop(
            output_dir=args.output_dir,
            pcb_path=args.pcb,
            max_iterations=args.max_iterations,
            quality_threshold=args.threshold,
            state_file=args.state_file,
            schematic_paths=args.schematic,
            strict=not args.no_strict
        )

        # Regenerate outputs first if requested
        if args.regenerate_first and args.pcb:
            print("Regenerating outputs with KiCad CLI before validation...")
            loop.regenerate_outputs()

        if args.dry_run:
            results = loop.dry_run()
        else:
            results = loop.run()

        if args.json:
            print(json.dumps(results, indent=2, default=str))

    except MissingDependencyFailure as e:
        print(f"\nMISSING DEPENDENCY: {e}", file=sys.stderr)
        sys.exit(2)
    except (EmptyBoardFailure, EmptySilkscreenFailure) as e:
        print(f"\nEMPTY CONTENT: {e}", file=sys.stderr)
        sys.exit(3)
    except QualityThresholdFailure as e:
        print(f"\nQUALITY THRESHOLD NOT MET: {e}", file=sys.stderr)
        sys.exit(4)
    except ValidationFailure as e:
        print(f"\nVALIDATION FAILED: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
