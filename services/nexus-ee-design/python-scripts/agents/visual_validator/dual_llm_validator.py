"""
Dual-LLM Visual Validator - Real image-based schematic validation.

CRITICAL REQUIREMENT: This validator MUST export the actual schematic to a
PNG/PDF image and validate the VISUAL output, NOT mathematical analysis.

Architecture:
┌─────────────────────────────────────────────────────────────┐
│         KiCad Schematic (.kicad_sch)                        │
└─────────────────────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────────────────────┐
│         KiCad CLI Export to PNG (High Resolution)           │
│         kicad-cli sch export svg/pdf/png                    │
└─────────────────────────────────────────────────────────────┘
                    ↓
    ┌───────────────────────────┐  ┌───────────────────────────┐
    │  Claude Opus 4.6          │  │  Kimi K2.5                │
    │  (Ultrathinking Mode)     │  │  (Independent Analysis)   │
    │                           │  │                           │
    │  Extended reasoning       │  │  Alternative perspective  │
    │  Deep visual analysis     │  │  Cross-validation         │
    └───────────────────────────┘  └───────────────────────────┘
                    ↓                         ↓
    ┌───────────────────────────────────────────────────────────┐
    │              Comparison & Consensus Engine                │
    │  - Agreement analysis                                     │
    │  - Disagreement resolution                                │
    │  - Combined confidence score                              │
    │  - Generate fix recommendations                           │
    └───────────────────────────────────────────────────────────┘
                    ↓
    ┌───────────────────────────────────────────────────────────┐
    │              MAPO Gaming Loop                             │
    │  - If score < target: generate fixes                      │
    │  - Re-export and re-validate                              │
    │  - Iterate until compliance or max iterations             │
    └───────────────────────────────────────────────────────────┘

Author: Nexus EE Design Team
"""

import asyncio
import base64
import json
import logging
import os
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, TYPE_CHECKING

import httpx

# Centralized LLM provider configuration (supports Claude Code Max proxy + OpenRouter)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from llm_provider import get_llm_config, check_llm_available, log_provider_info

# New imports for enhanced visual feedback loop
from .image_extractor import SchematicImageExtractor, ImageExtractionError
from .progress_tracker import ProgressTracker, ProgressAnalysis, StagnationError
from .issue_to_fix import IssueToFixTransformer, SchematicFix, TransformResult
from .fix_applicator import SchematicFixApplicator, ApplyResult

logger = logging.getLogger(__name__)


class VisualIssueCategory(Enum):
    """Categories of visual issues."""
    COMPONENT_PLACEMENT = "component_placement"
    WIRE_ROUTING = "wire_routing"
    LABEL_PLACEMENT = "label_placement"
    POWER_DISTRIBUTION = "power_distribution"
    SIGNAL_FLOW = "signal_flow"
    READABILITY = "readability"
    STANDARDS_COMPLIANCE = "standards_compliance"
    MISSING_ELEMENTS = "missing_elements"
    OVERLAPPING = "overlapping"
    SPACING = "spacing"


class IssueSeverity(Enum):
    """Severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class VisualIssue:
    """Single visual issue identified by LLM."""
    category: VisualIssueCategory
    severity: IssueSeverity
    description: str
    location: Optional[str] = None  # e.g., "top-left quadrant", "near U1"
    recommendation: Optional[str] = None
    confidence: float = 0.8


@dataclass
class VisualAnalysis:
    """Analysis result from a single LLM."""
    model_name: str
    score: float  # 0.0 to 1.0
    issues: List[VisualIssue] = field(default_factory=list)
    strengths: List[str] = field(default_factory=list)
    overall_assessment: str = ""
    reasoning: str = ""  # Extended thinking/reasoning
    execution_time_ms: float = 0.0
    raw_response: Optional[str] = None


@dataclass
class ComparisonResult:
    """Result of comparing two LLM analyses."""
    agreement_score: float  # 0.0 to 1.0
    agreed_issues: List[VisualIssue] = field(default_factory=list)
    disagreed_issues: List[Tuple[VisualIssue, VisualIssue]] = field(default_factory=list)
    unique_opus_issues: List[VisualIssue] = field(default_factory=list)
    unique_kimi_issues: List[VisualIssue] = field(default_factory=list)
    combined_score: float = 0.0
    resolution_notes: List[str] = field(default_factory=list)


@dataclass
class ValidationLoopResult:
    """Result of the MAPO gaming validation loop."""
    final_passed: bool
    iterations: int
    initial_score: float
    final_score: float
    history: List[ComparisonResult] = field(default_factory=list)
    fixes_applied: List[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.utcnow)


def export_schematic_to_image(
    schematic_path: str,
    output_path: str,
    format: str = "png",
    dpi: int = 300,
    background: str = "white"
) -> bool:
    """
    Export KiCad schematic to image using KiCad CLI.

    CRITICAL: This function exports the ACTUAL schematic to a visual image.
    The image is what gets validated, NOT the S-expression text.

    Args:
        schematic_path: Path to .kicad_sch file
        output_path: Path for output image
        format: Output format (png, svg, pdf)
        dpi: Resolution in DPI (for raster formats)
        background: Background color

    Returns:
        True if export succeeded
    """
    try:
        # Build KiCad CLI command
        if format == "svg":
            cmd = [
                "kicad-cli", "sch", "export", "svg",
                "--output", output_path,
                "--no-background-color" if background == "transparent" else "",
                schematic_path
            ]
        elif format == "pdf":
            cmd = [
                "kicad-cli", "sch", "export", "pdf",
                "--output", output_path,
                schematic_path
            ]
        else:  # PNG - export via SVG then convert
            svg_path = output_path.replace(".png", ".svg")
            cmd = [
                "kicad-cli", "sch", "export", "svg",
                "--output", svg_path,
                schematic_path
            ]

        # Remove empty strings from command
        cmd = [c for c in cmd if c]

        logger.info(f"Exporting schematic: {' '.join(cmd)}")

        # Run KiCad CLI
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=60
        )

        if result.returncode != 0:
            logger.error(f"KiCad export failed: {result.stderr}")
            # Fallback: create a placeholder image for testing
            return _create_fallback_image(schematic_path, output_path)

        # Convert SVG to PNG if needed
        if format == "png" and os.path.exists(svg_path):
            try:
                # Try using ImageMagick or rsvg-convert
                convert_cmd = ["convert", "-density", str(dpi), svg_path, output_path]
                subprocess.run(convert_cmd, capture_output=True, timeout=30)
                os.remove(svg_path)

                # Enforce a 4MB / 3500px max so large paper sizes (A2, A1) don't
                # exceed Anthropic's 5MB per-image API limit.
                if os.path.exists(output_path):
                    file_size = os.path.getsize(output_path)
                    if file_size > 4 * 1024 * 1024:
                        resize_cmd = [
                            "convert", output_path,
                            "-resize", "3500x3500>",  # shrink only if larger
                            output_path
                        ]
                        subprocess.run(resize_cmd, capture_output=True, timeout=30)
                        new_size = os.path.getsize(output_path)
                        logger.info(
                            f"Image resized for API: {file_size // 1024}KB → "
                            f"{new_size // 1024}KB (max 3500px)"
                        )
            except Exception as e:
                logger.warning(f"SVG to PNG conversion failed: {e}, using SVG directly")
                os.rename(svg_path, output_path.replace(".png", ".svg"))

        return os.path.exists(output_path) or os.path.exists(output_path.replace(".png", ".svg"))

    except subprocess.TimeoutExpired:
        logger.error("KiCad export timed out")
        return _create_fallback_image(schematic_path, output_path)
    except FileNotFoundError:
        logger.error("KiCad CLI not found - please install KiCad 7.x+")
        return _create_fallback_image(schematic_path, output_path)
    except Exception as e:
        logger.error(f"Export failed: {e}")
        return _create_fallback_image(schematic_path, output_path)


def _create_fallback_image(schematic_path: str, output_path: str) -> bool:
    """Create a fallback image when KiCad CLI is not available."""
    try:
        # Read schematic content for text-based fallback
        with open(schematic_path, 'r') as f:
            content = f.read()

        # Create a simple text-based representation
        # In production, this would use a proper rendering library
        logger.warning("Using fallback image generation - KiCad CLI not available")

        # For now, return False to indicate proper export wasn't possible
        return False
    except Exception as e:
        logger.error(f"Fallback image creation failed: {e}")
        return False


class DualLLMVisualValidator:
    """
    Dual-LLM visual validator using Opus 4.6 and Kimi K2.5.

    CRITICAL: This validator analyzes RENDERED IMAGES, not text/S-expressions.
    """

    # Validation thresholds
    PASS_SCORE = 0.85
    AGREEMENT_THRESHOLD = 0.70
    MAX_LOOP_ITERATIONS = 10

    # Ultrathinking prompt for Opus 4.6
    OPUS_PROMPT = """You are analyzing a rendered schematic image for professional quality validation.

IMPORTANT: Engage EXTENDED THINKING mode. Use maximum reasoning depth for deep visual analysis.

Analyze this schematic image and evaluate:

1. COMPONENT PLACEMENT (signal flow)
   - Are components arranged left-to-right (inputs → outputs)?
   - Is VCC at top, GND at bottom?
   - Are bypass capacitors near their ICs?
   - Proper spacing between components?

2. WIRE ROUTING
   - Are wires orthogonal (90° only)?
   - Any 4-way junctions (should be avoided)?
   - Are wires overlapping or crossing excessively?
   - Clean Manhattan routing?

3. LABELING & READABILITY
   - Are all components labeled?
   - Reference designators visible and correct (R1, C1, U1)?
   - Net names clear and readable?
   - No overlapping text?

4. POWER DISTRIBUTION
   - Clear power rails visible?
   - VCC and GND properly distributed?
   - Bypass caps connected?

5. STANDARDS COMPLIANCE
   - Follows IEC/IEEE schematic conventions?
   - Professional appearance?

Score the schematic from 0.0 to 1.0 where:
- 1.0 = Production-ready, professional quality
- 0.8 = Good quality, minor issues
- 0.6 = Acceptable, needs improvement
- 0.4 = Poor quality, significant issues
- 0.2 = Unacceptable, major problems
- 0.0 = Completely invalid

RESPOND IN THIS JSON FORMAT:
{
    "score": 0.85,
    "overall_assessment": "Brief overall assessment",
    "issues": [
        {
            "category": "wire_routing",
            "severity": "warning",
            "description": "Issue description",
            "location": "near U1",
            "recommendation": "How to fix"
        }
    ],
    "strengths": ["List of things done well"],
    "reasoning": "Detailed reasoning for the score"
}"""

    # Kimi K2.5 prompt (similar but independent)
    # Enhanced prompt with progress context for iterative validation
    OPUS_PROGRESS_PROMPT = """You are analyzing a rendered schematic image for MAPO iterative validation.

ITERATION CONTEXT:
- Current iteration: {iteration}
- Previous score: {previous_score}
- Target score: {target_score}
- Previous issues remaining: {remaining_issues}

CRITICAL: Your analysis determines whether the schematic is PROGRESSING.

VISUAL ANALYSIS CHECKLIST:

1. COMPONENT PLACEMENT (25%)
   - Signal flow: inputs left → processing center → outputs right
   - Power rails: VCC at top, GND at bottom
   - Bypass capacitors: within 5mm of IC power pins
   - Component spacing: minimum 2.54mm (100 mil) grid alignment

2. WIRE ROUTING (25%)
   - Orthogonal routing: 90° angles only, no diagonals
   - No 4-way junctions: T-junctions only
   - Wire crossings: minimized, visually distinct
   - Power/ground routing: horizontal rails preferred

3. LABELING & READABILITY (20%)
   - All components labeled with reference designators
   - Net labels on important signals
   - No overlapping text
   - Values visible (resistance, capacitance)

4. POWER DISTRIBUTION (15%)
   - VCC/GND connections to all ICs
   - Bypass capacitors present
   - Power flags on power nets

5. STANDARDS COMPLIANCE (15%)
   - IEEE 315 symbol conventions
   - IEC 60750 documentation standards

RESPOND IN THIS JSON FORMAT:
{{
    "score": 0.85,
    "progress_assessment": {{
        "is_progressing": true,
        "improvement_from_previous": 0.05,
        "areas_improved": ["wire_routing", "labels"],
        "areas_regressed": [],
        "stagnation_risk": "low"
    }},
    "issues": [
        {{
            "category": "wire_routing",
            "severity": "warning",
            "description": "4-way junction detected near U1",
            "location": "coordinates (50, 100)",
            "recommendation": "Convert to T-junction with 1.27mm offset",
            "auto_fixable": true,
            "fix_operation": "ADD_JUNCTION",
            "fix_parameters": {{
                "x": 50.0,
                "y": 100.0
            }}
        }}
    ],
    "strengths": ["Good signal flow direction", "Power rails properly routed"],
    "reasoning": "Extended reasoning for the assessment"
}}

IMPORTANT: Include fix_operation and fix_parameters for auto-fixable issues.
Available fix_operations: MOVE_COMPONENT, ROTATE_COMPONENT, ADD_WIRE, ADD_JUNCTION, ADD_LABEL, ADD_NET_LABEL, ADD_POWER_FLAG, ADD_NO_CONNECT, ADD_COMPONENT"""

    KIMI_PROMPT = """Analyze this electronic schematic image and evaluate its professional quality.

Examine these aspects:

1. VISUAL LAYOUT
   - Component arrangement and spacing
   - Wire routing quality (orthogonal, no 4-way junctions)
   - Label placement and readability

2. DESIGN QUALITY
   - Signal flow direction (inputs left, outputs right)
   - Power distribution (VCC top, GND bottom)
   - Bypass capacitor placement

3. STANDARDS
   - Reference designator conventions
   - Net naming conventions
   - Professional appearance

Provide a score from 0.0 to 1.0 and list any issues found.

RESPOND IN THIS JSON FORMAT:
{
    "score": 0.85,
    "overall_assessment": "Brief assessment",
    "issues": [
        {
            "category": "component_placement",
            "severity": "warning",
            "description": "Issue",
            "location": "where",
            "recommendation": "fix"
        }
    ],
    "strengths": ["strengths"],
    "reasoning": "reasoning"
}"""

    def __init__(
        self,
        openrouter_api_key: Optional[str] = None
    ):
        """
        Initialize the dual-LLM validator using centralized llm_provider.

        All LLM calls go through httpx POST to /v1/chat/completions
        (OpenAI-compatible format). This works with both the Claude Code Max
        proxy and OpenRouter.

        Args:
            openrouter_api_key: OpenRouter API key override (optional, env var used otherwise)
        """
        # Resolve LLM configuration from centralized provider
        self._llm_config = get_llm_config()
        self._llm_chat_url = self._llm_config["base_url"]  # Already points to /v1/chat/completions
        self._llm_headers = self._llm_config["headers"]
        self._ai_provider = self._llm_config["provider"]

        # Allow explicit OpenRouter key override
        if openrouter_api_key:
            self._llm_headers["Authorization"] = f"Bearer {openrouter_api_key}"

        # Model identifiers
        self.opus_model = "anthropic/claude-opus-4.6"
        self.kimi_model = "google/gemini-2.5-pro"  # Alternative vision model via OpenRouter

        # Validate provider is available
        if not check_llm_available():
            raise RuntimeError(
                f"LLM provider '{self._ai_provider}' is not available. "
                f"For claude_code_max: ensure CLAUDE_CODE_PROXY_URL is reachable. "
                f"For openrouter: set OPENROUTER_API_KEY environment variable."
            )

        log_provider_info("DualLLMVisualValidator")
        logger.info(
            f"DualLLMVisualValidator initialized: provider={self._ai_provider}, "
            f"chat_url={self._llm_chat_url}, opus_model={self.opus_model}"
        )

    async def validate(
        self,
        schematic_path: str,
        specification: str = "",
        export_dpi: int = 300
    ) -> ComparisonResult:
        """
        Validate schematic using dual-LLM visual analysis.

        CRITICAL: This exports the schematic to an image and sends the
        actual image to both LLMs for visual analysis.

        Args:
            schematic_path: Path to .kicad_sch file
            specification: Original design specification
            export_dpi: Export resolution

        Returns:
            ComparisonResult with combined analysis
        """
        logger.info(f"Starting dual-LLM visual validation of {schematic_path}")

        # Step 1: Export schematic to image
        with tempfile.TemporaryDirectory() as tmpdir:
            image_path = os.path.join(tmpdir, "schematic.png")

            if not export_schematic_to_image(schematic_path, image_path, "png", export_dpi):
                logger.warning("Image export failed, attempting SVG export")
                image_path = os.path.join(tmpdir, "schematic.svg")
                if not export_schematic_to_image(schematic_path, image_path, "svg"):
                    logger.error("All export methods failed")
                    return self._create_fallback_result()

            # Step 2: Read and encode image
            image_data = self._read_image(image_path)

            if not image_data:
                logger.error("Failed to read exported image")
                return self._create_fallback_result()

            # Step 3: Run both LLMs in parallel
            opus_result, kimi_result = await asyncio.gather(
                self._analyze_with_opus(image_data, specification),
                self._analyze_with_kimi(image_data, specification),
                return_exceptions=True
            )

            # Handle exceptions
            if isinstance(opus_result, Exception):
                logger.error(f"Opus analysis failed: {opus_result}")
                opus_result = self._create_fallback_analysis("Claude Opus 4.6")

            if isinstance(kimi_result, Exception):
                logger.error(f"Kimi analysis failed: {kimi_result}")
                kimi_result = self._create_fallback_analysis("Kimi K2.5")

            # Step 4: Compare and reconcile results
            comparison = self._compare_analyses(opus_result, kimi_result)

            logger.info(
                f"Visual validation complete: "
                f"Opus={opus_result.score:.2f}, Kimi={kimi_result.score:.2f}, "
                f"Agreement={comparison.agreement_score:.2f}"
            )

            return comparison

    def _read_image(self, image_path: str) -> Optional[bytes]:
        """Read and return image bytes."""
        try:
            with open(image_path, 'rb') as f:
                return f.read()
        except Exception as e:
            logger.error(f"Failed to read image: {e}")
            return None

    async def _analyze_with_opus(
        self,
        image_data: bytes,
        specification: str
    ) -> VisualAnalysis:
        """
        Analyze schematic image with Claude Opus 4.6 via OpenAI-compatible API.

        CRITICAL: This sends the actual rendered IMAGE to the LLM.
        Uses httpx POST to /v1/chat/completions (works with Claude Code Max proxy
        and OpenRouter).
        """
        start_time = time.time()
        prompt = self.OPUS_PROMPT + (
            f"\n\nDesign Specification:\n{specification}" if specification else ""
        )

        logger.info(
            f"Opus analysis: POST {self._llm_chat_url} "
            f"model={self.opus_model} image_size={len(image_data)} bytes"
        )

        try:
            return await self._call_chat_completions(
                model=self.opus_model,
                image_data=image_data,
                prompt=prompt,
                model_display_name="Claude Opus 4.6",
                start_time=start_time
            )
        except Exception as e:
            logger.error(f"Opus analysis error: {e}")
            raise  # NO FALLBACK - raise the error

    async def _analyze_with_opus_enhanced(
        self,
        image_data: bytes,
        prompt: str,
        start_time: float
    ) -> VisualAnalysis:
        """
        Analyze schematic image with Claude Opus 4.6 using enhanced prompt.

        This version accepts a pre-formatted prompt with progress context.
        Uses httpx POST to /v1/chat/completions.
        NO FALLBACK - raises errors on failure.
        """
        logger.info(
            f"Opus enhanced analysis: POST {self._llm_chat_url} "
            f"model={self.opus_model} image_size={len(image_data)} bytes"
        )

        try:
            return await self._call_chat_completions(
                model=self.opus_model,
                image_data=image_data,
                prompt=prompt,
                model_display_name="Claude Opus 4.6",
                start_time=start_time
            )
        except Exception as e:
            logger.error(f"Opus enhanced analysis error: {e}")
            raise  # NO FALLBACK - raise the error

    async def _analyze_with_kimi(
        self,
        image_data: bytes,
        specification: str
    ) -> VisualAnalysis:
        """
        Analyze schematic image with second LLM validator (Gemini 2.5 Pro or Kimi K2.5).

        CRITICAL: This sends the actual rendered IMAGE to the LLM.

        For Claude Code Max proxy: uses Opus as the second validator (same proxy).
        For OpenRouter: uses Gemini 2.5 Pro as an independent second opinion.
        """
        start_time = time.time()
        prompt = self.KIMI_PROMPT + (
            f"\n\nDesign Specification:\n{specification}" if specification else ""
        )

        # Determine which model to use for the second validator
        if self._ai_provider == "claude_code_max":
            # Claude Code Max proxy only supports Anthropic models.
            # Use the same Opus model but with the independent KIMI_PROMPT
            # to get a different analytical perspective.
            second_model = self.opus_model
            display_name = "Claude Opus 4.6 (Second Validator)"
        else:
            # OpenRouter: use an independent vision model for true cross-validation
            second_model = self.kimi_model
            display_name = "Gemini 2.5 Pro (Second Validator)"

        logger.info(
            f"Second validator analysis: POST {self._llm_chat_url} "
            f"model={second_model} image_size={len(image_data)} bytes"
        )

        try:
            return await self._call_chat_completions(
                model=second_model,
                image_data=image_data,
                prompt=prompt,
                model_display_name=display_name,
                start_time=start_time
            )
        except Exception as e:
            logger.error(f"Second validator ({display_name}) analysis error: {e}")
            raise  # NO FALLBACK - raise the error

    async def _call_chat_completions(
        self,
        model: str,
        image_data: bytes,
        prompt: str,
        model_display_name: str,
        start_time: float,
        max_tokens: int = 4096,
        temperature: float = 0.0
    ) -> VisualAnalysis:
        """
        Call /v1/chat/completions endpoint with image data.

        This is the single unified method for ALL LLM calls. Works with both
        Claude Code Max proxy and OpenRouter.

        Args:
            model: Model identifier (e.g., "anthropic/claude-opus-4.6")
            image_data: Raw image bytes (PNG)
            prompt: Analysis prompt text
            model_display_name: Human-readable model name for logs/results
            start_time: time.time() when the call started
            max_tokens: Maximum response tokens
            temperature: Sampling temperature (0.0 for deterministic)

        Returns:
            VisualAnalysis parsed from the LLM response

        Raises:
            RuntimeError: If the API call fails or returns an error
        """
        # Send image with prompt to LLM for real visual analysis.
        # The Claude Code Max proxy now supports image passthrough via
        # direct Anthropic API calls (oauth-2025-04-20 beta).
        img_b64 = base64.b64encode(image_data).decode('utf-8')
        logger.info(
            f"Sending schematic image ({len(image_data)} bytes) for visual analysis "
            f"via {self._ai_provider} provider"
        )
        payload = {
            "model": model,
            "messages": [
                {
                    "role": "system",
                    "content": "You are an expert electronic schematic reviewer. Respond with valid JSON only."
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{img_b64}"
                            }
                        }
                    ]
                }
            ],
            "max_tokens": max_tokens,
            "temperature": temperature
        }

        try:
            async with httpx.AsyncClient(timeout=httpx.Timeout(180.0)) as client:
                logger.info(
                    f"POST {self._llm_chat_url} | model={model} | "
                    f"image={len(image_data)} bytes | max_tokens={max_tokens}"
                )

                response = await client.post(
                    self._llm_chat_url,
                    headers=self._llm_headers,
                    json=payload
                )

                elapsed_ms = (time.time() - start_time) * 1000

                if response.status_code != 200:
                    error_body = response.text[:500]
                    raise RuntimeError(
                        f"LLM API returned HTTP {response.status_code} from {self._llm_chat_url}. "
                        f"Provider: {self._ai_provider}, Model: {model}. "
                        f"Response body: {error_body}"
                    )

                result = response.json()

                if "choices" not in result or not result["choices"]:
                    raise RuntimeError(
                        f"LLM API returned no choices. "
                        f"Provider: {self._ai_provider}, Model: {model}. "
                        f"Full response: {json.dumps(result)[:500]}"
                    )

                content = result["choices"][0]["message"]["content"]
                logger.info(
                    f"LLM response received: model={model_display_name}, "
                    f"elapsed={elapsed_ms:.0f}ms, content_length={len(content)}"
                )

                return self._parse_llm_response(
                    content,
                    model_display_name,
                    elapsed_ms
                )

        except httpx.TimeoutException as e:
            raise RuntimeError(
                f"LLM API call timed out after 180s. "
                f"Provider: {self._ai_provider}, URL: {self._llm_chat_url}, Model: {model}. "
                f"Error: {e}"
            )
        except httpx.ConnectError as e:
            raise RuntimeError(
                f"Cannot connect to LLM API at {self._llm_chat_url}. "
                f"Provider: {self._ai_provider}, Model: {model}. "
                f"Ensure the proxy/service is running and accessible. "
                f"Error: {e}"
            )
        except RuntimeError:
            raise  # Re-raise our own RuntimeErrors
        except Exception as e:
            raise RuntimeError(
                f"Unexpected error calling LLM API. "
                f"Provider: {self._ai_provider}, URL: {self._llm_chat_url}, Model: {model}. "
                f"Error type: {type(e).__name__}, Error: {e}"
            )

    def _parse_llm_response(
        self,
        response: str,
        model_name: str,
        execution_time_ms: float
    ) -> VisualAnalysis:
        """Parse LLM response into VisualAnalysis."""
        try:
            # Extract JSON from response
            json_match = response
            if "```json" in response:
                json_match = response.split("```json")[1].split("```")[0]
            elif "```" in response:
                json_match = response.split("```")[1].split("```")[0]

            data = json.loads(json_match)

            issues = []
            for issue_data in data.get("issues", []):
                try:
                    issues.append(VisualIssue(
                        category=VisualIssueCategory(issue_data.get("category", "readability")),
                        severity=IssueSeverity(issue_data.get("severity", "warning")),
                        description=issue_data.get("description", ""),
                        location=issue_data.get("location"),
                        recommendation=issue_data.get("recommendation")
                    ))
                except ValueError:
                    issues.append(VisualIssue(
                        category=VisualIssueCategory.READABILITY,
                        severity=IssueSeverity.WARNING,
                        description=issue_data.get("description", "Unknown issue")
                    ))

            return VisualAnalysis(
                model_name=model_name,
                score=float(data.get("score", 0.5)),
                issues=issues,
                strengths=data.get("strengths", []),
                overall_assessment=data.get("overall_assessment", ""),
                reasoning=data.get("reasoning", ""),
                execution_time_ms=execution_time_ms,
                raw_response=response
            )

        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse LLM response as JSON: {e}")
            # Try to extract score from text
            score = 0.5
            if "score" in response.lower():
                import re
                score_match = re.search(r'score[:\s]+(\d+\.?\d*)', response.lower())
                if score_match:
                    score = min(1.0, float(score_match.group(1)))

            return VisualAnalysis(
                model_name=model_name,
                score=score,
                overall_assessment=response[:500] if response else "Parse error",
                execution_time_ms=execution_time_ms,
                raw_response=response
            )

    def _compare_analyses(
        self,
        opus: VisualAnalysis,
        kimi: VisualAnalysis
    ) -> ComparisonResult:
        """Compare and reconcile analyses from both LLMs."""
        # Calculate agreement score based on score difference and issue overlap
        score_diff = abs(opus.score - kimi.score)
        score_agreement = 1.0 - min(score_diff, 0.5) * 2  # 0.5 diff = 0% agreement

        # Compare issues
        agreed_issues = []
        opus_unique = list(opus.issues)
        kimi_unique = list(kimi.issues)

        for o_issue in opus.issues:
            for k_issue in kimi.issues:
                if self._issues_match(o_issue, k_issue):
                    agreed_issues.append(o_issue)
                    if o_issue in opus_unique:
                        opus_unique.remove(o_issue)
                    if k_issue in kimi_unique:
                        kimi_unique.remove(k_issue)
                    break

        # Issue agreement
        total_unique_issues = len(opus.issues) + len(kimi.issues)
        if total_unique_issues > 0:
            issue_agreement = len(agreed_issues) * 2 / total_unique_issues
        else:
            issue_agreement = 1.0

        # Combined agreement
        agreement_score = (score_agreement * 0.6 + issue_agreement * 0.4)

        # Combined score (weighted average - Opus has slightly higher weight due to ultrathinking)
        combined_score = opus.score * 0.6 + kimi.score * 0.4

        # Resolution notes
        resolution_notes = []
        if score_diff > 0.2:
            resolution_notes.append(
                f"Significant score difference: Opus={opus.score:.2f}, Kimi={kimi.score:.2f}. "
                "Manual review recommended."
            )

        if len(opus_unique) > 0:
            resolution_notes.append(
                f"Opus found {len(opus_unique)} unique issues not identified by Kimi"
            )

        if len(kimi_unique) > 0:
            resolution_notes.append(
                f"Kimi found {len(kimi_unique)} unique issues not identified by Opus"
            )

        return ComparisonResult(
            agreement_score=agreement_score,
            agreed_issues=agreed_issues,
            disagreed_issues=[],  # Could add if tracking specific disagreements
            unique_opus_issues=opus_unique,
            unique_kimi_issues=kimi_unique,
            combined_score=combined_score,
            resolution_notes=resolution_notes
        )

    def _issues_match(self, issue1: VisualIssue, issue2: VisualIssue) -> bool:
        """Check if two issues are about the same problem."""
        # Same category is a good indicator
        if issue1.category != issue2.category:
            return False

        # Check for similar description
        desc1_words = set(issue1.description.lower().split())
        desc2_words = set(issue2.description.lower().split())
        overlap = len(desc1_words & desc2_words)
        total = len(desc1_words | desc2_words)

        return overlap / total > 0.3 if total > 0 else False

    def _create_fallback_analysis(self, model_name: str) -> VisualAnalysis:
        """Create fallback analysis when LLM is unavailable."""
        return VisualAnalysis(
            model_name=model_name,
            score=0.5,
            issues=[
                VisualIssue(
                    category=VisualIssueCategory.READABILITY,
                    severity=IssueSeverity.WARNING,
                    description="LLM analysis unavailable - manual review required"
                )
            ],
            overall_assessment="Automated analysis unavailable",
            reasoning="LLM client not configured or API call failed"
        )

    def _create_fallback_result(self) -> ComparisonResult:
        """Create fallback result when image export fails."""
        return ComparisonResult(
            agreement_score=0.0,
            combined_score=0.0,
            resolution_notes=["Image export failed - cannot perform visual validation"]
        )


class ValidationLoop:
    """
    Enhanced MAPO Gaming validation loop with visual feedback.

    Uses kicad-worker for image extraction, Opus 4.6 for visual analysis,
    progress tracking for stagnation detection, and automatic fix application.

    NO FALLBACKS - Strict error handling with verbose reporting.
    """

    def __init__(
        self,
        validator: DualLLMVisualValidator,
        image_extractor: Optional[SchematicImageExtractor] = None,
        progress_tracker: Optional[ProgressTracker] = None,
        issue_transformer: Optional[IssueToFixTransformer] = None,
        fix_applicator: Optional[SchematicFixApplicator] = None,
        fix_generator: Optional[Callable] = None,  # Legacy support
        target_score: float = 0.85,
        max_iterations: int = 10,
        max_stagnant_iterations: int = 3,
        max_fixes_per_iteration: int = 5
    ):
        """
        Initialize the enhanced validation loop.

        Args:
            validator: DualLLMVisualValidator instance
            image_extractor: SchematicImageExtractor for kicad-worker integration
            progress_tracker: ProgressTracker for stagnation detection
            issue_transformer: IssueToFixTransformer for generating fixes
            fix_applicator: SchematicFixApplicator for applying fixes
            fix_generator: Legacy fix generator callable (deprecated)
            target_score: Target compliance score (default 85%)
            max_iterations: Maximum loop iterations
            max_stagnant_iterations: Terminate after this many stagnant iterations
            max_fixes_per_iteration: Maximum fixes to apply per iteration
        """
        self.validator = validator
        self.image_extractor = image_extractor or SchematicImageExtractor()
        self.progress_tracker = progress_tracker or ProgressTracker(
            max_stagnant_iterations=max_stagnant_iterations
        )
        self.issue_transformer = issue_transformer or IssueToFixTransformer(
            max_fixes=max_fixes_per_iteration
        )
        self.fix_applicator = fix_applicator or SchematicFixApplicator()
        self.fix_generator = fix_generator  # Legacy
        self.target_score = target_score
        self.max_iterations = max_iterations
        self.max_stagnant_iterations = max_stagnant_iterations
        self.max_fixes_per_iteration = max_fixes_per_iteration

        logger.info(
            f"ValidationLoop initialized: target={target_score:.0%}, "
            f"max_iter={max_iterations}, max_stagnant={max_stagnant_iterations}"
        )

    async def run(
        self,
        schematic_path: str,
        schematic_content: Optional[str] = None,
        specification: str = "",
        on_iteration: Optional[Callable] = None
    ) -> ValidationLoopResult:
        """
        Run the enhanced validation loop with visual feedback.

        NO FALLBACKS - Raises errors on failure.

        Args:
            schematic_path: Path to .kicad_sch file
            schematic_content: Optional raw schematic content (for content-based fixes)
            specification: Design specification
            on_iteration: Callback after each iteration

        Returns:
            ValidationLoopResult with final status and history

        Raises:
            ImageExtractionError: If kicad-worker fails to extract image
            StagnationError: If validation stagnates
        """
        logger.info(
            f"Starting enhanced MAPO validation loop "
            f"(target: {self.target_score:.0%}, max_iter: {self.max_iterations})"
        )

        # Reset progress tracker for new run
        self.progress_tracker.reset()

        # Load schematic content if not provided
        if schematic_content is None:
            schematic_content = Path(schematic_path).read_text(encoding='utf-8')

        history: List[ComparisonResult] = []
        fixes_applied: List[str] = []
        initial_score = 0.0
        current_content = schematic_content
        previous_score = 0.0
        previous_image_hash = ""

        for iteration in range(self.max_iterations):
            logger.info(f"=" * 60)
            logger.info(f"Validation iteration {iteration + 1}/{self.max_iterations}")
            logger.info(f"=" * 60)

            # Step 1: Extract image via kicad-worker (NO FALLBACK)
            logger.info("Step 1: Extracting image via kicad-worker...")
            image_result = await self.image_extractor.extract_png(
                schematic_content=current_content,
                design_name="validation",
                iteration=iteration
            )

            if not image_result.success:
                raise ImageExtractionError(
                    message="Image extraction failed during validation loop",
                    kicad_worker_url=self.image_extractor.kicad_worker_url,
                    schematic_size_bytes=len(current_content),
                    errors=image_result.errors
                )

            # Step 2: Run dual-LLM validation with image
            logger.info("Step 2: Running dual-LLM visual validation...")
            result = await self._validate_with_progress_context(
                image_data=image_result.image_data,
                specification=specification,
                iteration=iteration,
                previous_score=previous_score,
                remaining_issues=len(history[-1].agreed_issues) if history else 0
            )
            history.append(result)

            if iteration == 0:
                initial_score = result.combined_score

            logger.info(
                f"Iteration {iteration + 1} score: {result.combined_score:.2%} "
                f"(delta: {result.combined_score - previous_score:+.2%})"
            )

            # Save per-iteration image and Opus analysis report to disk
            try:
                iteration_dir = Path(schematic_path).parent / "validation_iterations"
                iteration_dir.mkdir(parents=True, exist_ok=True)

                # Save rendered image
                img_path = iteration_dir / f"iteration_{iteration + 1}.png"
                img_path.write_bytes(image_result.image_data)

                # Save analysis report
                report = {
                    "iteration": iteration + 1,
                    "combined_score": round(result.combined_score, 4),
                    "delta": round(result.combined_score - previous_score, 4),
                    "agreed_issues": [
                        {"category": str(i.category), "message": i.description, "confidence": i.confidence}
                        for i in result.agreed_issues
                    ] if hasattr(result, 'agreed_issues') else [],
                    "opus_unique_issues": [
                        {"category": str(i.category), "message": i.description, "confidence": i.confidence}
                        for i in result.unique_opus_issues
                    ] if hasattr(result, 'unique_opus_issues') else [],
                    "timestamp": datetime.now().isoformat(),
                }
                report_path = iteration_dir / f"iteration_{iteration + 1}_report.json"
                report_path.write_text(json.dumps(report, indent=2))

                logger.info(
                    f"Saved iteration {iteration + 1} image ({len(image_result.image_data)} bytes) "
                    f"and report to {iteration_dir}"
                )
            except Exception as save_err:
                logger.warning(f"Failed to save iteration artifacts: {save_err}")

            # Step 3: Track progress and detect stagnation
            logger.info("Step 3: Analyzing progress...")
            progress = self.progress_tracker.record_and_analyze(
                comparison_result=result,
                image_hash=image_result.image_hash
            )

            # Step 4: Check convergence criteria
            logger.info("Step 4: Checking convergence...")

            # 4a: Target achieved
            if result.combined_score >= self.target_score:
                logger.info(f"TARGET ACHIEVED: {result.combined_score:.2%} >= {self.target_score:.0%}")
                return ValidationLoopResult(
                    final_passed=True,
                    iterations=iteration + 1,
                    initial_score=initial_score,
                    final_score=result.combined_score,
                    history=history,
                    fixes_applied=fixes_applied
                )

            # 4b: Stagnation detected
            if progress.recommended_action == "terminate":
                logger.error(
                    f"STAGNATION DETECTED: {progress.consecutive_stagnant_iterations} "
                    f"consecutive stagnant iterations"
                )
                self.progress_tracker.raise_if_stagnant()

            # Step 5: Generate fixes from issues
            logger.info("Step 5: Generating fixes from issues...")
            all_issues = (
                result.agreed_issues +
                result.unique_opus_issues +
                result.unique_kimi_issues
            )

            if all_issues and iteration < self.max_iterations - 1:
                transform_result = await self.issue_transformer.transform(
                    issues=all_issues,
                    schematic_content=current_content,
                    max_fixes=self.max_fixes_per_iteration
                )

                if transform_result.fixes:
                    # Step 6: Apply fixes to schematic
                    logger.info(f"Step 6: Applying {len(transform_result.fixes)} fixes...")
                    apply_result = await self.fix_applicator.apply_fixes(
                        schematic_content=current_content,
                        fixes=transform_result.fixes
                    )

                    if apply_result.applied_fixes:
                        current_content = apply_result.modified_content
                        fixes_applied.extend(apply_result.applied_fixes)
                        logger.info(f"Applied {len(apply_result.applied_fixes)} fixes")

                        # Save updated content to path for validation
                        Path(schematic_path).write_text(current_content, encoding='utf-8')
                    else:
                        logger.warning("No fixes were successfully applied")
                else:
                    logger.warning("No fixable issues identified")

            # Update for next iteration
            previous_score = result.combined_score
            previous_image_hash = image_result.image_hash

            # Callback
            if on_iteration:
                try:
                    on_iteration(iteration + 1, result, progress)
                except Exception as e:
                    logger.warning(f"Iteration callback failed: {e}")

        # Max iterations reached
        final_score = history[-1].combined_score if history else 0.0
        logger.warning(f"MAX ITERATIONS REACHED. Final score: {final_score:.2%}")

        return ValidationLoopResult(
            final_passed=final_score >= self.target_score,
            iterations=self.max_iterations,
            initial_score=initial_score,
            final_score=final_score,
            history=history,
            fixes_applied=fixes_applied
        )

    async def _validate_with_progress_context(
        self,
        image_data: bytes,
        specification: str,
        iteration: int,
        previous_score: float,
        remaining_issues: int
    ) -> ComparisonResult:
        """
        Run validation with progress context in the prompt.

        Uses enhanced OPUS_PROGRESS_PROMPT for better fix suggestions.
        """
        # Format enhanced prompt with context
        enhanced_prompt = self.validator.OPUS_PROGRESS_PROMPT.format(
            iteration=iteration + 1,
            previous_score=f"{previous_score:.2%}" if previous_score > 0 else "N/A (first iteration)",
            target_score=f"{self.target_score:.0%}",
            remaining_issues=remaining_issues
        )

        # Add specification
        if specification:
            enhanced_prompt += f"\n\nDesign Specification:\n{specification}"

        # Run analysis with enhanced prompt
        start_time = time.time()

        opus_result = await self.validator._analyze_with_opus_enhanced(
            image_data=image_data,
            prompt=enhanced_prompt,
            start_time=start_time
        )

        # Run Kimi analysis in parallel (standard prompt)
        kimi_result = await self.validator._analyze_with_kimi(
            image_data=image_data,
            specification=specification
        )

        # Compare and return
        return self.validator._compare_analyses(opus_result, kimi_result)


# CLI entry point for testing
if __name__ == "__main__":
    import sys

    async def main():
        print("=" * 60)
        print("DUAL-LLM VISUAL VALIDATOR TEST")
        print("=" * 60)

        # Check for test schematic
        if len(sys.argv) > 1:
            schematic_path = sys.argv[1]
        else:
            # Use a default test path
            schematic_path = "/tmp/test_schematic.kicad_sch"
            print(f"No schematic provided, using: {schematic_path}")

        # Test image export
        print("\n1. Testing image export...")
        output_path = "/tmp/test_export.png"
        success = export_schematic_to_image(schematic_path, output_path)
        print(f"   Export success: {success}")

        # Test validator (will use fallback if no API keys)
        print("\n2. Testing dual-LLM validation...")
        validator = DualLLMVisualValidator()

        if os.path.exists(schematic_path):
            result = await validator.validate(schematic_path, "Test design specification")

            print(f"\nValidation Results:")
            print(f"   Combined Score: {result.combined_score:.2%}")
            print(f"   Agreement Score: {result.agreement_score:.2%}")
            print(f"   Agreed Issues: {len(result.agreed_issues)}")
            print(f"   Resolution Notes:")
            for note in result.resolution_notes:
                print(f"      - {note}")
        else:
            print(f"   Schematic not found: {schematic_path}")

        # Test validation loop
        print("\n3. Testing validation loop...")
        loop = ValidationLoop(validator, target_score=0.90, max_iterations=3)

        if os.path.exists(schematic_path):
            loop_result = await loop.run(schematic_path)

            print(f"\nLoop Results:")
            print(f"   Final Passed: {loop_result.final_passed}")
            print(f"   Iterations: {loop_result.iterations}")
            print(f"   Initial Score: {loop_result.initial_score:.2%}")
            print(f"   Final Score: {loop_result.final_score:.2%}")
        else:
            print("   Cannot run loop without schematic file")

        print("\n" + "=" * 60)
        print("VISUAL VALIDATOR TEST COMPLETE")
        print("=" * 60)

    asyncio.run(main())
