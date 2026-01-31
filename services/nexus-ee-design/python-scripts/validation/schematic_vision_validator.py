"""
Schematic Vision Validator - LLM Vision-based schematic validation.

Uses multiple expert personas to validate KiCad schematics through visual analysis.
Part of the MAPO-Schematic (Multi-Agent Pattern Orchestration) system.

Author: Nexus EE Design Team
"""

import asyncio
import base64
import json
import re
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict, Optional, Tuple, Any
from pathlib import Path
from datetime import datetime

try:
    import anthropic
except ImportError:
    anthropic = None

logger = logging.getLogger(__name__)


class ExpertPersona(Enum):
    """Expert personas for schematic validation."""
    CIRCUIT_ANALYST = "circuit_analyst"
    COMPONENT_VALIDATOR = "component_validator"
    CONNECTION_VERIFIER = "connection_verifier"
    MANUFACTURABILITY_REVIEWER = "manufacturability_reviewer"
    REFERENCE_COMPARATOR = "reference_comparator"


# Human-friendly names for experts (following patent plugin pattern)
EXPERT_FRIENDLY_NAMES = {
    ExpertPersona.CIRCUIT_ANALYST: "Circuit Topology Expert",
    ExpertPersona.COMPONENT_VALIDATOR: "Component Symbol Specialist",
    ExpertPersona.CONNECTION_VERIFIER: "Connectivity Analyst",
    ExpertPersona.MANUFACTURABILITY_REVIEWER: "DFM Specialist",
    ExpertPersona.REFERENCE_COMPARATOR: "Best Practices Reviewer",
}


def friendly_expert_name(expert: ExpertPersona) -> str:
    """Convert expert enum to human-readable name."""
    return EXPERT_FRIENDLY_NAMES.get(expert, expert.value)


@dataclass
class ValidationIssue:
    """Single validation issue identified by an expert."""
    component: str  # Reference designator or 'general'
    description: str
    severity: str  # critical, major, minor
    location: str  # Description of location in schematic
    expert: str  # Which expert found this

    def to_dict(self) -> Dict:
        return {
            "component": self.component,
            "description": self.description,
            "severity": self.severity,
            "location": self.location,
            "expert": self.expert
        }


@dataclass
class FixRecommendation:
    """Recommended fix for a validation issue."""
    issue_ref: str  # Component reference or 'general'
    fix_type: str  # add_component, remove_component, modify_connection, change_value, add_label
    description: str
    kicad_action: str  # Specific KiCad operation
    priority: str  # high, medium, low

    def to_dict(self) -> Dict:
        return {
            "issue_ref": self.issue_ref,
            "fix_type": self.fix_type,
            "description": self.description,
            "kicad_action": self.kicad_action,
            "priority": self.priority
        }


@dataclass
class ExpertValidationResult:
    """Result from a single expert validation."""
    expert: ExpertPersona
    score: float  # 0.0 - 1.0
    passed: bool
    issues: List[ValidationIssue] = field(default_factory=list)
    suggestions: List[str] = field(default_factory=list)
    confidence: float = 0.5
    raw_response: str = ""
    error: Optional[str] = None

    def to_dict(self) -> Dict:
        return {
            "expert": self.expert.value,
            "score": self.score,
            "passed": self.passed,
            "issues": [i.to_dict() for i in self.issues],
            "suggestions": self.suggestions,
            "confidence": self.confidence,
            "error": self.error
        }


@dataclass
class SchematicValidationReport:
    """Complete validation report from all experts."""
    overall_score: float
    passed: bool
    expert_results: List[ExpertValidationResult]
    critical_issues: List[ValidationIssue]
    recommended_fixes: List[FixRecommendation]
    iteration: int
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())

    def to_dict(self) -> Dict:
        return {
            "overall_score": self.overall_score,
            "passed": self.passed,
            "expert_results": [r.to_dict() for r in self.expert_results],
            "critical_issues": [i.to_dict() for i in self.critical_issues],
            "recommended_fixes": [f.to_dict() for f in self.recommended_fixes],
            "iteration": self.iteration,
            "timestamp": self.timestamp
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    def get_summary(self) -> str:
        """
        Get human-readable validation summary.
        Follows patent plugin box/table formatting pattern.
        """
        lines = [
            "",
            "=" * 60,
            "SCHEMATIC VALIDATION SUMMARY",
            "=" * 60,
            f"Iteration: {self.iteration + 1}",
            f"Timestamp: {self.timestamp}",
            f"Overall Score: {self.overall_score * 100:.1f}%",
            f"Status: {'✅ PASSED' if self.passed else '⚠️ NEEDS IMPROVEMENT'}",
            "",
            "EXPERT RESULTS:",
        ]

        # Sort by score descending
        sorted_results = sorted(self.expert_results, key=lambda r: r.score, reverse=True)
        for i, result in enumerate(sorted_results, 1):
            medal = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else f"#{i}"
            status = "✅" if result.passed else "⚠️"
            lines.append(
                f"  {medal} {friendly_expert_name(result.expert):30s} | "
                f"Score: {result.score * 100:5.1f}% | "
                f"Issues: {len(result.issues):2d} | "
                f"{status}"
            )

        if self.critical_issues:
            lines.extend([
                "",
                f"CRITICAL ISSUES ({len(self.critical_issues)}):",
            ])
            for issue in self.critical_issues[:5]:
                lines.append(f"  ❌ [{issue.component}] {issue.description}")
            if len(self.critical_issues) > 5:
                lines.append(f"  ... and {len(self.critical_issues) - 5} more")

        if self.recommended_fixes:
            lines.extend([
                "",
                f"RECOMMENDED FIXES ({len(self.recommended_fixes)}):",
            ])
            for fix in self.recommended_fixes[:5]:
                priority_emoji = "🔴" if fix.priority == "high" else "🟡" if fix.priority == "medium" else "🟢"
                lines.append(f"  {priority_emoji} [{fix.issue_ref}] {fix.description}")
            if len(self.recommended_fixes) > 5:
                lines.append(f"  ... and {len(self.recommended_fixes) - 5} more")

        lines.extend(["", "=" * 60])
        return "\n".join(lines)


class SchematicVisionValidator:
    """
    LLM Vision-based schematic validation using multiple expert personas.

    Implements MAPO (Multi-Agent Pattern Orchestration) for quality assurance.
    Uses Claude's vision capabilities to analyze rendered schematic images.
    """

    # Expert prompts for each persona
    EXPERT_PROMPTS = {
        ExpertPersona.CIRCUIT_ANALYST: """You are an expert circuit analyst reviewing a KiCad schematic.

Analyze this schematic image for:
1. **Circuit Topology**: Is the overall circuit topology correct for the intended function?
2. **Signal Flow**: Does signal flow logically (inputs → processing → outputs)?
3. **Power Distribution**: Are power rails properly distributed to all components?
4. **Feedback Loops**: Are feedback paths correctly implemented (if applicable)?
5. **Protection Circuits**: Are necessary protection elements present (ESD, overcurrent)?

Rate each aspect 0-100 and identify specific issues with component references.""",

        ExpertPersona.COMPONENT_VALIDATOR: """You are a component specialist validating KiCad schematic symbols.

Check each component for:
1. **Symbol Accuracy**: Do symbols match their actual component representation?
2. **Pin Mapping**: Are pins correctly labeled and positioned?
3. **Values/Ratings**: Are component values visible and appropriate?
4. **Reference Designators**: Are designators unique and follow conventions (R1, C1, U1)?
5. **Footprint Association**: Do components appear to have appropriate footprints?

List any components with incorrect or missing symbol data.""",

        ExpertPersona.CONNECTION_VERIFIER: """You are a connectivity specialist checking schematic wiring.

Verify:
1. **Wire Continuity**: Are all wires properly connected (no floating endpoints)?
2. **Junction Dots**: Are wire junctions clearly marked with dots?
3. **Net Labels**: Are important nets labeled consistently?
4. **Power Connections**: Are VCC and GND properly connected to all ICs?
5. **No-Connect Pins**: Are unused pins marked with no-connect flags?
6. **Bus Connections**: Are buses properly labeled and connected?

Identify any disconnected nets or ambiguous connections.""",

        ExpertPersona.MANUFACTURABILITY_REVIEWER: """You are a DFM specialist reviewing schematic for manufacturability.

Evaluate:
1. **BOM Completeness**: Are all components properly specified for ordering?
2. **Standard Values**: Are resistors/capacitors using standard E-series values?
3. **Package Selection**: Are packages appropriate for the design (SMD vs THT)?
4. **Second Sourcing**: Can critical components be second-sourced?
5. **Assembly Clarity**: Is the schematic clear enough for assembly?

Flag any components that may cause manufacturing issues.""",

        ExpertPersona.REFERENCE_COMPARATOR: """You are comparing this schematic against reference designs and best practices.

Evaluate:
1. **Best Practices**: Does the design follow manufacturer recommendations?
2. **Application Notes**: Are reference design patterns correctly applied?
3. **Critical Component Placement**: Are key components (decoupling caps, etc.) placed correctly?
4. **Known Issues**: Does the design avoid known problematic patterns?
5. **Industry Standards**: Does the design follow IPC and other relevant standards?

Identify deviations from reference designs that may cause issues."""
    }

    # Weights for overall score calculation
    EXPERT_WEIGHTS = {
        ExpertPersona.CIRCUIT_ANALYST: 0.30,
        ExpertPersona.COMPONENT_VALIDATOR: 0.25,
        ExpertPersona.CONNECTION_VERIFIER: 0.25,
        ExpertPersona.MANUFACTURABILITY_REVIEWER: 0.10,
        ExpertPersona.REFERENCE_COMPARATOR: 0.10
    }

    PASS_THRESHOLD = 0.85

    def __init__(
        self,
        primary_model: str = "claude-sonnet-4-20250514",
        verification_model: str = "claude-opus-4-20250514",
        api_key: Optional[str] = None
    ):
        """
        Initialize the validator.

        Args:
            primary_model: Model for expert validation (faster, cheaper)
            verification_model: Model for fix generation (more capable)
            api_key: Anthropic API key (defaults to ANTHROPIC_API_KEY env var)
        """
        self.primary_model = primary_model
        self.verification_model = verification_model

        if anthropic is None:
            raise ImportError("anthropic package required. Install with: pip install anthropic")

        self.client = anthropic.Anthropic(api_key=api_key) if api_key else anthropic.Anthropic()

    async def _emit_progress(
        self,
        callback: Optional[Any],
        event_type: str,
        message: str,
        level: str = "info",
        **extra_data
    ):
        """
        Emit progress event to callback and log.
        Following patent plugin pattern: dual logging to server AND callback.
        """
        # Always log to server
        log_method = getattr(logger, level, logger.info)
        log_method(f"[{event_type}] {message}")

        # Emit to callback if provided (for WebSocket streaming)
        if callback is not None:
            event = {
                "event_type": event_type,
                "message": message,
                "timestamp": datetime.utcnow().isoformat(),
                "level": level,
                **extra_data
            }
            if asyncio.iscoroutinefunction(callback):
                await callback(event)
            else:
                callback(event)

    async def validate_schematic(
        self,
        schematic_image: bytes,
        design_intent: str,
        reference_images: Optional[List[bytes]] = None,
        iteration: int = 0,
        parallel: bool = True,
        progress_callback: Optional[Any] = None
    ) -> SchematicValidationReport:
        """
        Run full validation pipeline with all expert personas.

        Args:
            schematic_image: PNG/JPEG bytes of rendered schematic
            design_intent: Description of what the circuit should do
            reference_images: Optional reference schematic images for comparison
            iteration: Current iteration number in MAPO loop
            parallel: Run experts in parallel (faster) vs sequential (more stable)
            progress_callback: Optional callback for progress events (WebSocket streaming)

        Returns:
            Comprehensive validation report
        """
        await self._emit_progress(
            progress_callback, "validation_init",
            f"🎯 Starting Schematic Validation (Iteration {iteration + 1}) — {len(ExpertPersona)} expert AI reviewers will analyze the circuit design.",
            iteration=iteration,
            num_experts=len(ExpertPersona),
            experts=[friendly_expert_name(e) for e in ExpertPersona]
        )

        # Run all experts
        if parallel:
            expert_results = await self._run_experts_parallel(
                schematic_image, design_intent, reference_images
            )
        else:
            expert_results = await self._run_experts_sequential(
                schematic_image, design_intent, reference_images
            )

        # Calculate overall score (weighted average)
        overall_score = sum(
            result.score * self.EXPERT_WEIGHTS[result.expert]
            for result in expert_results
            if result.error is None
        )

        # Aggregate critical issues
        critical_issues = []
        for result in expert_results:
            for issue in result.issues:
                if issue.severity == 'critical':
                    critical_issues.append(issue)

        # Generate fix recommendations
        recommended_fixes = await self._generate_fix_recommendations(
            expert_results, critical_issues
        )

        # Determine if passed
        passed = overall_score >= self.PASS_THRESHOLD and len(critical_issues) == 0

        report = SchematicValidationReport(
            overall_score=overall_score,
            passed=passed,
            expert_results=expert_results,
            critical_issues=critical_issues,
            recommended_fixes=recommended_fixes,
            iteration=iteration
        )

        # Emit expert rankings with human-readable format
        await self._emit_progress(
            progress_callback, "expert_rankings",
            f"📊 Expert Review Results — Overall: {overall_score*100:.0f}% compliance",
            overall_score=overall_score,
            passed=passed
        )

        # Rank experts by score
        sorted_results = sorted(expert_results, key=lambda r: r.score, reverse=True)
        for rank, result in enumerate(sorted_results, 1):
            medal = "🥇" if rank == 1 else "🥈" if rank == 2 else "🥉" if rank == 3 else f"#{rank}"
            status = "✅ Passed" if result.passed else "⚠️ Issues found"
            issues_text = f" ({len(result.issues)} issues)" if result.issues else ""

            await self._emit_progress(
                progress_callback, "expert_result",
                f"{medal} {friendly_expert_name(result.expert)}: {result.score*100:.0f}% — {status}{issues_text}",
                rank=rank,
                expert=result.expert.value,
                expert_friendly=friendly_expert_name(result.expert),
                score=result.score,
                passed=result.passed,
                issue_count=len(result.issues)
            )

        # Final status
        if passed:
            await self._emit_progress(
                progress_callback, "validation_passed",
                f"🎉 SUCCESS! Schematic achieved {overall_score*100:.0f}% compliance — exceeds the {self.PASS_THRESHOLD*100:.0f}% target!",
                level="success",
                overall_score=overall_score,
                target=self.PASS_THRESHOLD
            )
        else:
            await self._emit_progress(
                progress_callback, "validation_needs_improvement",
                f"⚠️ Schematic needs improvement: {overall_score*100:.0f}% compliance (target: {self.PASS_THRESHOLD*100:.0f}%), {len(critical_issues)} critical issues",
                level="warning",
                overall_score=overall_score,
                target=self.PASS_THRESHOLD,
                critical_issue_count=len(critical_issues)
            )

        return report

    async def _run_experts_parallel(
        self,
        schematic_image: bytes,
        design_intent: str,
        reference_images: Optional[List[bytes]]
    ) -> List[ExpertValidationResult]:
        """Run all experts in parallel."""
        tasks = [
            self._run_expert_validation(expert, schematic_image, design_intent, reference_images)
            for expert in ExpertPersona
        ]
        return await asyncio.gather(*tasks)

    async def _run_experts_sequential(
        self,
        schematic_image: bytes,
        design_intent: str,
        reference_images: Optional[List[bytes]]
    ) -> List[ExpertValidationResult]:
        """Run experts sequentially (more API-friendly)."""
        results = []
        for expert in ExpertPersona:
            result = await self._run_expert_validation(
                expert, schematic_image, design_intent, reference_images
            )
            results.append(result)
        return results

    async def _run_expert_validation(
        self,
        expert: ExpertPersona,
        schematic_image: bytes,
        design_intent: str,
        reference_images: Optional[List[bytes]]
    ) -> ExpertValidationResult:
        """Run single expert validation."""
        logger.debug(f"Running {expert.value} validation")

        try:
            # Build message content
            content = self._build_validation_message(
                expert, schematic_image, design_intent, reference_images
            )

            # Call Claude API
            response = self.client.messages.create(
                model=self.primary_model,
                max_tokens=2048,
                messages=[{"role": "user", "content": content}]
            )

            raw_response = response.content[0].text

            # Parse JSON response
            result_data = self._parse_expert_response(raw_response)

            # Build issues list
            issues = [
                ValidationIssue(
                    component=issue.get('component', 'general'),
                    description=issue.get('description', ''),
                    severity=issue.get('severity', 'minor'),
                    location=issue.get('location', ''),
                    expert=expert.value
                )
                for issue in result_data.get('issues', [])
            ]

            return ExpertValidationResult(
                expert=expert,
                score=result_data.get('score', 50) / 100.0,
                passed=result_data.get('passed', False),
                issues=issues,
                suggestions=result_data.get('suggestions', []),
                confidence=result_data.get('confidence', 0.5),
                raw_response=raw_response
            )

        except Exception as e:
            logger.error(f"Error in {expert.value} validation: {e}")
            return ExpertValidationResult(
                expert=expert,
                score=0.0,
                passed=False,
                error=str(e)
            )

    def _build_validation_message(
        self,
        expert: ExpertPersona,
        schematic_image: bytes,
        design_intent: str,
        reference_images: Optional[List[bytes]]
    ) -> List[Dict]:
        """Build the message content for Claude API."""

        # Determine image media type
        media_type = "image/png"
        if schematic_image[:3] == b'\xff\xd8\xff':
            media_type = "image/jpeg"

        content = [
            {
                "type": "text",
                "text": f"""Design Intent: {design_intent}

{self.EXPERT_PROMPTS[expert]}

Provide your analysis in the following JSON format:
{{
    "score": <0-100>,
    "passed": <true/false>,
    "issues": [
        {{
            "component": "<ref designator or 'general'>",
            "description": "<issue description>",
            "severity": "critical|major|minor",
            "location": "<description of location in schematic>"
        }}
    ],
    "suggestions": ["<improvement suggestion>", ...],
    "confidence": <0.0-1.0>
}}

Return ONLY valid JSON, no additional text."""
            },
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": media_type,
                    "data": base64.b64encode(schematic_image).decode()
                }
            }
        ]

        # Add reference images for reference comparator
        if reference_images and expert == ExpertPersona.REFERENCE_COMPARATOR:
            for i, ref_img in enumerate(reference_images[:2]):  # Max 2 references
                ref_media_type = "image/png"
                if ref_img[:3] == b'\xff\xd8\xff':
                    ref_media_type = "image/jpeg"

                content.append({
                    "type": "text",
                    "text": f"\nReference Design {i+1}:"
                })
                content.append({
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": ref_media_type,
                        "data": base64.b64encode(ref_img).decode()
                    }
                })

        return content

    def _parse_expert_response(self, response_text: str) -> Dict:
        """Parse JSON from expert response."""
        # Try to extract JSON from response
        json_match = re.search(r'\{[\s\S]*\}', response_text)
        if json_match:
            try:
                return json.loads(json_match.group())
            except json.JSONDecodeError:
                pass

        # Fallback
        logger.warning("Could not parse expert response as JSON")
        return {
            "score": 50,
            "passed": False,
            "issues": [],
            "suggestions": [],
            "confidence": 0.3
        }

    async def _generate_fix_recommendations(
        self,
        expert_results: List[ExpertValidationResult],
        critical_issues: List[ValidationIssue]
    ) -> List[FixRecommendation]:
        """Generate actionable fix recommendations from validation results."""

        # Consolidate all issues
        all_issues = []
        for result in expert_results:
            for issue in result.issues:
                all_issues.append(issue.to_dict())

        if not all_issues:
            return []

        try:
            prompt = f"""Based on these schematic validation issues, generate specific fix recommendations:

Issues:
{json.dumps(all_issues, indent=2)}

For each significant issue, provide a fix recommendation in this format:
{{
    "issue_ref": "<component or general>",
    "fix_type": "add_component|remove_component|modify_connection|change_value|add_label|other",
    "description": "<what to change>",
    "kicad_action": "<specific KiCad operation or S-expression modification>",
    "priority": "high|medium|low"
}}

Return a JSON array of recommendations. Focus on the most important fixes first.
Return ONLY valid JSON array, no additional text."""

            response = self.client.messages.create(
                model=self.verification_model,
                max_tokens=2048,
                messages=[{"role": "user", "content": prompt}]
            )

            result_text = response.content[0].text

            # Parse recommendations
            json_match = re.search(r'\[[\s\S]*\]', result_text)
            if json_match:
                fixes_data = json.loads(json_match.group())
                return [
                    FixRecommendation(
                        issue_ref=fix.get('issue_ref', 'general'),
                        fix_type=fix.get('fix_type', 'other'),
                        description=fix.get('description', ''),
                        kicad_action=fix.get('kicad_action', ''),
                        priority=fix.get('priority', 'medium')
                    )
                    for fix in fixes_data
                ]

        except Exception as e:
            logger.error(f"Error generating fix recommendations: {e}")

        return []

    def validate_schematic_sync(
        self,
        schematic_image: bytes,
        design_intent: str,
        reference_images: Optional[List[bytes]] = None,
        iteration: int = 0
    ) -> SchematicValidationReport:
        """
        Synchronous wrapper for validate_schematic.

        Use this when not in an async context.
        """
        return asyncio.run(
            self.validate_schematic(
                schematic_image, design_intent, reference_images, iteration,
                parallel=False  # Sequential is safer for sync wrapper
            )
        )


class MAPOSchematicLoop:
    """
    Multi-Agent Pattern Orchestration loop for schematic generation and validation.

    Loop:
    1. Generate/modify schematic
    2. Render to image
    3. Validate with Vision Validator
    4. If failed, apply fixes and repeat
    5. Continue until passed or max iterations
    """

    MAX_ITERATIONS = 5

    def __init__(
        self,
        validator: SchematicVisionValidator,
        renderer: Any = None,  # KiCanvasRenderer
        modifier: Any = None   # Schematic modifier function
    ):
        self.validator = validator
        self.renderer = renderer
        self.modifier = modifier
        self.history: List[SchematicValidationReport] = []

    async def run_loop(
        self,
        schematic_content: str,
        design_intent: str,
        reference_images: Optional[List[bytes]] = None
    ) -> Tuple[str, SchematicValidationReport]:
        """
        Run the MAPO validation loop.

        Args:
            schematic_content: Initial KiCad schematic content
            design_intent: Description of intended circuit function
            reference_images: Optional reference design images

        Returns:
            Tuple of (final schematic content, final validation report)
        """
        best_schematic = schematic_content
        best_score = 0.0
        current_schematic = schematic_content

        for iteration in range(self.MAX_ITERATIONS):
            logger.info(f"MAPO Loop iteration {iteration + 1}/{self.MAX_ITERATIONS}")

            # Render schematic to image
            if self.renderer:
                schematic_image = await self.renderer.render_to_png(current_schematic)
            else:
                raise ValueError("Renderer required for MAPO loop")

            # Validate
            report = await self.validator.validate_schematic(
                schematic_image,
                design_intent,
                reference_images,
                iteration=iteration
            )

            self.history.append(report)

            logger.info(
                f"Iteration {iteration + 1}: "
                f"score={report.overall_score:.2f}, passed={report.passed}"
            )

            # Track best
            if report.overall_score > best_score:
                best_score = report.overall_score
                best_schematic = current_schematic

            # Check termination
            if report.passed:
                logger.info(f"Validation passed at iteration {iteration + 1}")
                return current_schematic, report

            if not report.recommended_fixes:
                logger.info("No more fixes to apply, terminating")
                break

            # Apply fixes
            if self.modifier:
                current_schematic = await self.modifier(
                    current_schematic,
                    report.recommended_fixes
                )
            else:
                logger.warning("No modifier provided, cannot apply fixes")
                break

        logger.info(f"Max iterations reached, returning best (score={best_score:.2f})")
        return best_schematic, self.history[-1]

    def get_improvement_trend(self) -> List[float]:
        """Get the score trend across iterations."""
        return [report.overall_score for report in self.history]


# CLI entry point for testing
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 3:
        print("Usage: python schematic_vision_validator.py <image_path> <design_intent>")
        sys.exit(1)

    image_path = Path(sys.argv[1])
    design_intent = sys.argv[2]

    if not image_path.exists():
        print(f"Error: Image not found: {image_path}")
        sys.exit(1)

    # Read image
    with open(image_path, 'rb') as f:
        image_bytes = f.read()

    # Run validation
    validator = SchematicVisionValidator()
    report = validator.validate_schematic_sync(image_bytes, design_intent)

    # Output report
    print("\n" + "=" * 60)
    print("SCHEMATIC VALIDATION REPORT")
    print("=" * 60)
    print(f"\nOverall Score: {report.overall_score:.2%}")
    print(f"Passed: {report.passed}")
    print(f"\nExpert Results:")
    for result in report.expert_results:
        status = "✓" if result.passed else "✗"
        print(f"  {status} {result.expert.value}: {result.score:.2%} (confidence: {result.confidence:.2f})")
        if result.error:
            print(f"    ERROR: {result.error}")

    if report.critical_issues:
        print(f"\nCritical Issues ({len(report.critical_issues)}):")
        for issue in report.critical_issues:
            print(f"  - [{issue.component}] {issue.description}")

    if report.recommended_fixes:
        print(f"\nRecommended Fixes ({len(report.recommended_fixes)}):")
        for fix in report.recommended_fixes:
            print(f"  [{fix.priority.upper()}] {fix.issue_ref}: {fix.description}")

    print("\n" + "=" * 60)

    # Save full report
    output_path = image_path.with_suffix('.validation.json')
    with open(output_path, 'w') as f:
        f.write(report.to_json())
    print(f"Full report saved to: {output_path}")
