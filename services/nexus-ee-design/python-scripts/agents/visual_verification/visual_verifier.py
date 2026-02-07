"""
Visual Verification Agent for MAPO v3.1

Uses Claude Opus 4.5 to visually assess schematic quality and reject poor layouts.
Generates schematic images using KiCad CLI and scores them against a quality rubric.

Author: MAPO Team
Date: 2026-02-07
"""

import asyncio
import base64
import json
import logging
import os
import subprocess
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import httpx
import yaml

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class VerificationError(Exception):
    """Raised when visual verification fails."""
    pass


@dataclass
class QualityIssue:
    """A specific quality issue found in the schematic."""
    criterion_id: str
    criterion_name: str
    score: float  # 0-100 for this criterion
    issue_description: str
    suggestion: str

    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            'criterion_id': self.criterion_id,
            'criterion_name': self.criterion_name,
            'score': self.score,
            'issue_description': self.issue_description,
            'suggestion': self.suggestion
        }


@dataclass
class VisualQualityReport:
    """Complete visual quality assessment of a schematic."""
    passed: bool  # True if overall_score >= 90
    overall_score: float  # 0-100 weighted average
    criterion_scores: Dict[str, float]  # score per rubric item
    issues: List[QualityIssue]
    image_path: str
    model_used: str  # 'anthropic/claude-opus-4-5'
    timestamp: str
    overall_assessment: str = ""  # Summary from Opus
    production_ready: bool = False  # Direct assessment from Opus

    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            'passed': self.passed,
            'overall_score': self.overall_score,
            'criterion_scores': self.criterion_scores,
            'issues': [issue.to_dict() for issue in self.issues],
            'image_path': self.image_path,
            'model_used': self.model_used,
            'timestamp': self.timestamp,
            'overall_assessment': self.overall_assessment,
            'production_ready': self.production_ready
        }

    def __str__(self) -> str:
        """Human-readable report."""
        status = "✓ PASS" if self.passed else "✗ FAIL"
        lines = [
            f"\n{'='*80}",
            f"VISUAL QUALITY REPORT - {status}",
            f"{'='*80}",
            f"Overall Score: {self.overall_score:.1f}/100",
            f"Production Ready: {'Yes' if self.production_ready else 'No'}",
            f"Model: {self.model_used}",
            f"Timestamp: {self.timestamp}",
            f"\nAssessment: {self.overall_assessment}",
            f"\n{'-'*80}",
            f"CRITERION SCORES:",
            f"{'-'*80}"
        ]

        for criterion_id, score in self.criterion_scores.items():
            status_icon = "✓" if score >= 90 else "⚠" if score >= 70 else "✗"
            lines.append(f"  {status_icon} {criterion_id:25} {score:5.1f}/100")

        if self.issues:
            lines.extend([
                f"\n{'-'*80}",
                f"ISSUES IDENTIFIED ({len(self.issues)}):",
                f"{'-'*80}"
            ])
            for i, issue in enumerate(self.issues, 1):
                lines.extend([
                    f"\n{i}. {issue.criterion_name} (Score: {issue.score:.1f}/100)",
                    f"   Issue: {issue.issue_description}",
                    f"   Suggestion: {issue.suggestion}"
                ])

        lines.append(f"{'='*80}\n")
        return "\n".join(lines)


class VisualVerifier:
    """
    Visual quality assessment agent using Claude Opus 4.5.

    Generates schematic images from KiCad files and uses vision-enabled AI
    to assess quality against a comprehensive rubric.
    """

    def __init__(
        self,
        rubric_path: Optional[str] = None,
        api_key: Optional[str] = None
    ):
        """
        Initialize visual verifier with quality rubric.

        Args:
            rubric_path: Path to quality_rubric.yaml (defaults to same directory)
            api_key: OpenRouter API key (defaults to OPENROUTER_API_KEY env var)
        """
        # Locate rubric file
        if rubric_path is None:
            rubric_path = Path(__file__).parent / "quality_rubric.yaml"

        self.rubric_path = Path(rubric_path)
        if not self.rubric_path.exists():
            raise VerificationError(
                f"Quality rubric not found: {self.rubric_path}\n"
                f"Expected location: {Path(__file__).parent}/quality_rubric.yaml"
            )

        # Load rubric
        with open(self.rubric_path) as f:
            self.rubric_data = yaml.safe_load(f)

        self.rubric = self.rubric_data['rubric']
        self.thresholds = self.rubric_data.get('thresholds', {
            'pass': 90,
            'acceptable': 70,
            'poor': 50
        })
        self.image_settings = self.rubric_data.get('image_settings', {})

        # API configuration
        self.api_key = api_key or os.getenv('OPENROUTER_API_KEY')
        if not self.api_key:
            raise VerificationError(
                "OpenRouter API key not found. Set OPENROUTER_API_KEY environment variable."
            )

        self.model = "anthropic/claude-opus-4-5"
        logger.info(f"VisualVerifier initialized with rubric: {self.rubric_path}")

    async def verify(
        self,
        schematic_path: str,
        output_dir: str = "/tmp"
    ) -> VisualQualityReport:
        """
        Generate image and assess visual quality of schematic.

        Args:
            schematic_path: Path to .kicad_sch file
            output_dir: Directory for generated images

        Returns:
            VisualQualityReport with score and detailed issues

        Raises:
            VerificationError: If image generation or analysis fails
        """
        logger.info(f"Starting visual verification of: {schematic_path}")

        # Validate input
        schematic_path = Path(schematic_path)
        if not schematic_path.exists():
            raise VerificationError(
                f"Schematic file not found: {schematic_path}"
            )

        if not schematic_path.suffix == '.kicad_sch':
            raise VerificationError(
                f"Invalid file type: {schematic_path.suffix}. "
                f"Expected .kicad_sch file."
            )

        # Create output directory
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Step 1: Generate schematic image
        logger.info("Step 1: Generating schematic image...")
        image_path = await self._generate_image(schematic_path, output_dir)
        logger.info(f"Image generated: {image_path}")

        # Step 2: Analyze with Opus 4.5
        logger.info("Step 2: Analyzing with Claude Opus 4.5...")
        analysis = await self._analyze_with_opus(image_path)
        logger.info("Analysis complete")

        # Step 3: Build quality report
        logger.info("Step 3: Building quality report...")
        report = self._build_report(analysis, str(image_path))

        logger.info(
            f"Verification complete: {'PASS' if report.passed else 'FAIL'} "
            f"(Score: {report.overall_score:.1f}/100)"
        )

        return report

    async def _generate_image(
        self,
        schematic_path: Path,
        output_dir: Path
    ) -> Path:
        """
        Generate PNG image of schematic using KiCad CLI.

        Args:
            schematic_path: Path to .kicad_sch file
            output_dir: Output directory for images

        Returns:
            Path to generated PNG image

        Raises:
            VerificationError: If KiCad CLI fails
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name = f"schematic_{timestamp}"
        output_svg = output_dir / f"{base_name}.svg"
        output_png = output_dir / f"{base_name}.png"

        # Step 1: Export to SVG using KiCad CLI
        logger.info(f"Exporting to SVG: {output_svg}")
        try:
            result = subprocess.run(
                [
                    "kicad-cli", "sch", "export", "svg",
                    "--output", str(output_svg),
                    str(schematic_path)
                ],
                capture_output=True,
                text=True,
                timeout=30
            )
        except FileNotFoundError:
            raise VerificationError(
                "KiCad CLI not found. Please install KiCad 8.x and ensure "
                "'kicad-cli' is available in PATH.\n"
                "Installation: https://www.kicad.org/download/"
            )
        except subprocess.TimeoutExpired:
            raise VerificationError(
                f"KiCad export timed out after 30 seconds. "
                f"Schematic may be too complex: {schematic_path}"
            )

        if result.returncode != 0:
            raise VerificationError(
                f"KiCad export to SVG failed with exit code {result.returncode}\n"
                f"Command: kicad-cli sch export svg\n"
                f"Stderr: {result.stderr}\n"
                f"Stdout: {result.stdout}"
            )

        if not output_svg.exists():
            raise VerificationError(
                f"KiCad export completed but SVG file not created: {output_svg}"
            )

        # Step 2: Convert SVG to PNG using ImageMagick
        logger.info(f"Converting to PNG: {output_png}")
        density = self.image_settings.get('density', 300)
        background = self.image_settings.get('background', 'white')

        try:
            result = subprocess.run(
                [
                    "convert",
                    "-density", str(density),
                    "-background", background,
                    "-alpha", "remove",
                    "-flatten",
                    str(output_svg),
                    str(output_png)
                ],
                capture_output=True,
                text=True,
                timeout=30
            )
        except FileNotFoundError:
            # ImageMagick not available, use SVG directly
            logger.warning(
                "ImageMagick 'convert' command not found. Using SVG directly.\n"
                "For better quality, install ImageMagick: brew install imagemagick"
            )
            return output_svg
        except subprocess.TimeoutExpired:
            logger.warning("PNG conversion timed out, using SVG")
            return output_svg

        if result.returncode != 0:
            logger.warning(
                f"PNG conversion failed, using SVG: {result.stderr}"
            )
            return output_svg

        if output_png.exists():
            return output_png
        else:
            return output_svg

    async def _analyze_with_opus(
        self,
        image_path: Path
    ) -> Dict:
        """
        Send image to Claude Opus 4.5 for visual quality analysis.

        Args:
            image_path: Path to schematic image (PNG or SVG)

        Returns:
            Analysis results as dictionary with scores and issues

        Raises:
            VerificationError: If API call fails
        """
        # Read and encode image
        logger.info(f"Reading image: {image_path}")
        try:
            with open(image_path, 'rb') as f:
                image_data = base64.b64encode(f.read()).decode()
        except Exception as e:
            raise VerificationError(
                f"Failed to read image file: {image_path}\n"
                f"Error: {str(e)}"
            )

        # Determine media type
        if image_path.suffix.lower() == '.png':
            media_type = "image/png"
        elif image_path.suffix.lower() == '.svg':
            media_type = "image/svg+xml"
        else:
            raise VerificationError(
                f"Unsupported image format: {image_path.suffix}"
            )

        # Build analysis prompt
        prompt = self._build_analysis_prompt()

        # Call OpenRouter API with vision
        logger.info(f"Calling OpenRouter API with model: {self.model}")
        try:
            async with httpx.AsyncClient(timeout=120.0) as client:
                response = await client.post(
                    "https://openrouter.ai/api/v1/chat/completions",
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json",
                        "HTTP-Referer": "https://nexus.adverant.com",
                        "X-Title": "Adverant Nexus Visual Verifier"
                    },
                    json={
                        "model": self.model,
                        "messages": [
                            {
                                "role": "user",
                                "content": [
                                    {"type": "text", "text": prompt},
                                    {
                                        "type": "image_url",
                                        "image_url": {
                                            "url": f"data:{media_type};base64,{image_data}"
                                        }
                                    }
                                ]
                            }
                        ],
                        "temperature": 0.2,
                        "max_tokens": 4000,
                    }
                )
        except httpx.TimeoutException:
            raise VerificationError(
                "OpenRouter API request timed out after 120 seconds. "
                "The vision model may be overloaded. Please retry."
            )
        except httpx.RequestError as e:
            raise VerificationError(
                f"Network error calling OpenRouter API: {str(e)}"
            )

        # Check response status
        if response.status_code != 200:
            error_body = response.text
            raise VerificationError(
                f"OpenRouter API failed with status {response.status_code}\n"
                f"Response: {error_body}\n"
                f"Check API key and model availability."
            )

        # Parse response
        try:
            result = response.json()
            content = result['choices'][0]['message']['content']
        except (KeyError, IndexError, json.JSONDecodeError) as e:
            raise VerificationError(
                f"Failed to parse OpenRouter API response: {str(e)}\n"
                f"Response: {response.text}"
            )

        # Extract JSON from markdown code blocks if present
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            content = content.split("```")[1].split("```")[0].strip()

        # Parse JSON analysis
        try:
            analysis = json.loads(content)
        except json.JSONDecodeError as e:
            raise VerificationError(
                f"Failed to parse JSON analysis from Opus 4.5: {str(e)}\n"
                f"Content: {content}"
            )

        # Validate analysis structure
        if 'criterion_scores' not in analysis:
            raise VerificationError(
                "Invalid analysis format: missing 'criterion_scores' field\n"
                f"Analysis: {json.dumps(analysis, indent=2)}"
            )

        return analysis

    def _build_analysis_prompt(self) -> str:
        """
        Build structured prompt with quality rubric for Opus 4.5.

        Returns:
            Formatted prompt string with rubric and instructions
        """
        # Format rubric as structured text
        rubric_lines = []
        for item in self.rubric:
            rubric_lines.append(f"\n{item['id']} - {item['name']} (Weight: {item['weight']*100:.0f}%)")
            rubric_lines.append(f"  Criteria: {item['pass_criteria']}")
            rubric_lines.append(f"  Scoring: {item['scoring']}")
            rubric_lines.append(f"  Description: {item['description']}")

        rubric_text = "\n".join(rubric_lines)

        prompt = f"""You are a professional electrical engineer with 20+ years of experience reviewing schematics for production readiness. Your role is to assess this electronic schematic against industry best practices.

QUALITY RUBRIC:
{rubric_text}

INSTRUCTIONS:

For EACH criterion in the rubric above:
1. Carefully examine the schematic image
2. Evaluate against the pass_criteria specified
3. Assign a score 0-100 following the scoring guidelines
4. If score < 100, identify SPECIFIC issues (e.g., "U1 and U2 overlap", "12 wire crossings in power section")
5. Provide ACTIONABLE suggestions (e.g., "Move U2 down 2 grid units", "Reroute VCC net to reduce crossings")

SCORING CALIBRATION:
- 95-100: Exceptional quality, textbook example
- 90-94: Professional, production-ready
- 70-89: Acceptable but needs improvement
- 50-69: Poor quality, significant issues
- 0-49: Unacceptable, major problems

OUTPUT FORMAT (strict JSON):

{{
  "criterion_scores": {{
    "symbol_overlap": {{"score": 100, "issues": "None", "suggestion": ""}},
    "wire_crossings": {{"score": 70, "issues": "12 wire crossings found in power distribution section", "suggestion": "Reroute VCC and GND nets to minimize crossings, consider using net labels"}},
    "signal_flow": {{"score": 85, "issues": "Some signals flow right-to-left near U3", "suggestion": "Flip U3 horizontally to maintain left-to-right flow"}},
    "power_flow": {{"score": 100, "issues": "None", "suggestion": ""}},
    "functional_grouping": {{"score": 90, "issues": "Minor: decoupling caps could be closer to ICs", "suggestion": "Move C1, C2 adjacent to U1, U2 pins"}},
    "net_labels": {{"score": 75, "issues": "3 labels overlap with wires", "suggestion": "Reposition labels for CLK, DATA, RESET"}},
    "spacing": {{"score": 80, "issues": "Components tight in upper-right quadrant", "suggestion": "Spread components more evenly across canvas"}},
    "professional_appearance": {{"score": 88, "issues": "Overall good, minor alignment issues", "suggestion": "Align power symbols vertically"}}
  }},
  "overall_assessment": "This is a professional schematic with good signal flow and functional grouping. Main areas for improvement are reducing wire crossings in the power section and improving spacing in the upper-right area. With minor adjustments, this would be production-ready.",
  "production_ready": true
}}

CRITICAL REQUIREMENTS:
- Be objective and precise
- Use specific component references (U1, R5, etc.) when identifying issues
- Professional schematics typically score 85-95
- Only exceptional schematics score 95+
- Poor hobbyist schematics score 40-60
- Your assessment will be used for automated quality gates

Analyze the schematic now and return ONLY the JSON output."""

        return prompt

    def _build_report(
        self,
        analysis: Dict,
        image_path: str
    ) -> VisualQualityReport:
        """
        Convert Opus 4.5 analysis to structured VisualQualityReport.

        Args:
            analysis: Raw analysis dictionary from Opus
            image_path: Path to analyzed schematic image

        Returns:
            VisualQualityReport with scores and issues
        """
        # Extract overall assessment
        overall_assessment = analysis.get('overall_assessment', '')
        production_ready = analysis.get('production_ready', False)

        # Calculate weighted average score
        criterion_scores = {}
        issues = []
        total_weighted_score = 0.0

        for item in self.rubric:
            criterion_id = item['id']
            weight = item['weight']

            # Get score data from analysis
            if criterion_id not in analysis['criterion_scores']:
                logger.warning(
                    f"Criterion '{criterion_id}' missing from analysis, using 0"
                )
                score = 0
                score_data = {
                    'score': 0,
                    'issues': 'Not evaluated',
                    'suggestion': 'Re-run analysis'
                }
            else:
                score_data = analysis['criterion_scores'][criterion_id]
                score = score_data['score']

            criterion_scores[criterion_id] = score
            total_weighted_score += score * weight

            # Record issues for scores < 100
            if score < 100:
                issues.append(QualityIssue(
                    criterion_id=criterion_id,
                    criterion_name=item['name'],
                    score=score,
                    issue_description=score_data.get('issues', 'See analysis'),
                    suggestion=score_data.get('suggestion', 'No suggestion provided')
                ))

        # Overall score is weighted average
        overall_score = total_weighted_score

        # Pass/fail based on threshold
        pass_threshold = self.thresholds.get('pass', 90)
        passed = overall_score >= pass_threshold

        return VisualQualityReport(
            passed=passed,
            overall_score=overall_score,
            criterion_scores=criterion_scores,
            issues=issues,
            image_path=image_path,
            model_used=self.model,
            timestamp=datetime.now().isoformat(),
            overall_assessment=overall_assessment,
            production_ready=production_ready
        )


# Convenience function for CLI usage
async def verify_schematic(
    schematic_path: str,
    output_dir: str = "/tmp",
    rubric_path: Optional[str] = None
) -> VisualQualityReport:
    """
    Convenience function to verify a schematic.

    Args:
        schematic_path: Path to .kicad_sch file
        output_dir: Output directory for images
        rubric_path: Optional custom rubric path

    Returns:
        VisualQualityReport
    """
    verifier = VisualVerifier(rubric_path=rubric_path)
    report = await verifier.verify(schematic_path, output_dir)
    return report


if __name__ == "__main__":
    # CLI entry point for testing
    import sys

    if len(sys.argv) < 2:
        print("Usage: python visual_verifier.py <schematic.kicad_sch> [output_dir]")
        sys.exit(1)

    schematic = sys.argv[1]
    output = sys.argv[2] if len(sys.argv) > 2 else "/tmp"

    async def main():
        report = await verify_schematic(schematic, output)
        print(report)

        # Exit with appropriate code
        sys.exit(0 if report.passed else 1)

    asyncio.run(main())
