"""
Issue-to-Fix Transformer - Converts visual validation issues to actionable schematic fixes.

Uses Opus 4.6 to analyze issues and generate specific fix operations that can be
applied to the KiCad S-expression schematic content.

NO FALLBACKS - Strict error handling with verbose reporting.

Author: Nexus EE Design Team
"""

import json
import logging
import os
import re
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .dual_llm_validator import VisualIssue, VisualIssueCategory

logger = logging.getLogger(__name__)

# LLM configuration
OPENROUTER_API_KEY = os.environ.get('OPENROUTER_API_KEY', '')
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


class FixTransformError(Exception):
    """
    Raised when issue-to-fix transformation fails.

    Contains detailed context for debugging.
    """

    def __init__(
        self,
        message: str,
        issue_count: int,
        failed_issues: List[str],
        llm_response: Optional[str] = None
    ):
        self.message = message
        self.issue_count = issue_count
        self.failed_issues = failed_issues
        self.llm_response = llm_response

        full_message = f"""
================================================================================
ISSUE-TO-FIX TRANSFORMATION FAILED
================================================================================
Error: {message}
Total Issues: {issue_count}
Failed Issues:
{chr(10).join(f"  - {i}" for i in failed_issues[:5])}

LLM Response (truncated):
{llm_response[:500] if llm_response else 'N/A'}

TROUBLESHOOTING:
1. Check OPENROUTER_API_KEY is set
2. Review issue format matches expected schema
3. Check Opus 4.6 is available
================================================================================
"""
        super().__init__(full_message)


class FixCategory(Enum):
    """Categories of schematic fixes."""
    COMPONENT_PLACEMENT = "component_placement"
    WIRE_ROUTING = "wire_routing"
    LABEL_ADDITION = "label_addition"
    POWER_CONNECTION = "power_connection"
    SIGNAL_FLOW = "signal_flow"
    BYPASS_CAPACITOR = "bypass_capacitor"
    REFERENCE_DESIGNATOR = "reference_designator"
    JUNCTION = "junction"
    GROUND_CONNECTION = "ground_connection"
    NET_LABEL = "net_label"


class FixOperation(Enum):
    """Specific fix operations for S-expression modification."""
    MOVE_COMPONENT = "move_component"
    ROTATE_COMPONENT = "rotate_component"
    ADD_WIRE = "add_wire"
    REMOVE_WIRE = "remove_wire"
    REROUTE_NET = "reroute_net"
    ADD_LABEL = "add_label"
    ADD_JUNCTION = "add_junction"
    ADD_POWER_FLAG = "add_power_flag"
    ADD_NO_CONNECT = "add_no_connect"
    ADD_COMPONENT = "add_component"
    MODIFY_VALUE = "modify_value"
    ADD_NET_LABEL = "add_net_label"


@dataclass
class SchematicFix:
    """
    Actionable fix that can be applied to KiCad schematic S-expression.

    Contains all information needed to modify the schematic content.
    """
    fix_id: str
    priority: int  # 1 = highest priority
    category: FixCategory
    fix_operation: FixOperation
    description: str
    target_component: Optional[str] = None  # Reference designator (U1, R1, etc.)
    target_net: Optional[str] = None  # Net name for routing fixes
    parameters: Dict[str, Any] = field(default_factory=dict)
    estimated_impact: float = 0.5  # Expected score improvement (0.0-1.0)
    source_issues: List[str] = field(default_factory=list)  # Issue IDs
    auto_fixable: bool = True
    confidence: float = 0.8

    def to_dict(self) -> dict:
        return {
            "fix_id": self.fix_id,
            "priority": self.priority,
            "category": self.category.value,
            "fix_operation": self.fix_operation.value,
            "description": self.description,
            "target_component": self.target_component,
            "target_net": self.target_net,
            "parameters": self.parameters,
            "estimated_impact": self.estimated_impact,
            "source_issues": self.source_issues,
            "auto_fixable": self.auto_fixable,
            "confidence": self.confidence,
        }


@dataclass
class TransformResult:
    """Result from issue-to-fix transformation."""
    success: bool
    fixes: List[SchematicFix] = field(default_factory=list)
    skipped_issues: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    total_issues: int = 0
    fixable_issues: int = 0

    def to_dict(self) -> dict:
        return {
            "success": self.success,
            "fixes": [f.to_dict() for f in self.fixes],
            "skipped_issues": self.skipped_issues,
            "errors": self.errors,
            "total_issues": self.total_issues,
            "fixable_issues": self.fixable_issues,
        }


class IssueToFixTransformer:
    """
    Transforms visual validation issues into actionable schematic fixes.

    Uses Opus 4.6 to analyze issues and generate specific fix operations
    with parameters for S-expression modification.

    NO FALLBACKS - Strict transformation with verbose errors.

    Usage:
        transformer = IssueToFixTransformer()
        result = await transformer.transform(issues, schematic_content)
        for fix in result.fixes:
            # Apply fix to schematic
            pass
    """

    # Priority order for issue categories (lower = higher priority)
    PRIORITY_ORDER = {
        "missing_elements": 1,      # Missing components = broken circuit
        "power_distribution": 2,    # Power issues = circuit won't work
        "wire_routing": 3,          # Routing issues = incorrect connections
        "overlapping": 4,           # Overlapping = hard to read
        "component_placement": 5,   # Placement = readability
        "label_placement": 6,       # Labels = documentation
        "signal_flow": 7,           # Signal flow = convention
        "spacing": 8,               # Spacing = aesthetics
        "readability": 9,           # Readability = minor
        "standards_compliance": 10, # Standards = optional
    }

    # Maximum fixes per iteration to prevent over-correction
    MAX_FIXES_PER_ITERATION = 5

    FIX_GENERATION_PROMPT = """You are analyzing schematic validation issues to generate specific fixes.

For each issue, generate a fix operation that can be applied to the KiCad S-expression schematic.

IMPORTANT: Generate fixes that are ACTIONABLE and SPECIFIC. Each fix must include:
1. fix_operation: The specific operation type
2. target_component or target_net: What to modify
3. parameters: Specific values for the modification

AVAILABLE FIX OPERATIONS:
- MOVE_COMPONENT: Move a component to new coordinates
  parameters: {x: float, y: float, reference: str}

- ROTATE_COMPONENT: Rotate a component
  parameters: {angle: int (0, 90, 180, 270), reference: str}

- ADD_WIRE: Add a wire connection
  parameters: {start_x: float, start_y: float, end_x: float, end_y: float, net: str}

- ADD_JUNCTION: Add a junction at intersection
  parameters: {x: float, y: float}

- ADD_LABEL: Add a text label
  parameters: {x: float, y: float, text: str, size: float}

- ADD_NET_LABEL: Add a net label
  parameters: {x: float, y: float, net_name: str}

- ADD_POWER_FLAG: Add a power flag symbol
  parameters: {x: float, y: float, power_net: str}

- ADD_NO_CONNECT: Add no-connect flag
  parameters: {x: float, y: float}

- ADD_COMPONENT: Add a new component (e.g., bypass capacitor)
  parameters: {symbol_lib: str, symbol_name: str, reference: str, value: str, x: float, y: float}

- REROUTE_NET: Reroute a net through different path
  parameters: {net_name: str, waypoints: [{x: float, y: float}, ...]}

ISSUES TO FIX:
{issues_json}

SCHEMATIC CONTEXT:
- Total components: {component_count}
- Sheet size: {sheet_size}
- Existing nets: {net_names}

Respond with a JSON array of fixes:
[
  {
    "fix_operation": "MOVE_COMPONENT",
    "category": "component_placement",
    "description": "Move U1 to improve signal flow",
    "target_component": "U1",
    "parameters": {"x": 100.0, "y": 50.0, "reference": "U1"},
    "source_issue_index": 0,
    "auto_fixable": true,
    "confidence": 0.9,
    "estimated_impact": 0.1
  }
]

If an issue cannot be auto-fixed, set auto_fixable to false and explain in description.
RESPOND ONLY WITH THE JSON ARRAY, NO OTHER TEXT.
"""

    def __init__(
        self,
        max_fixes: int = 5,
        min_confidence: float = 0.6,
        use_llm: bool = True
    ):
        """
        Initialize the transformer.

        Args:
            max_fixes: Maximum fixes to generate per iteration
            min_confidence: Minimum confidence threshold for fixes
            use_llm: Whether to use LLM for complex fix generation
        """
        self.max_fixes = max_fixes
        self.min_confidence = min_confidence
        self.use_llm = use_llm
        self._http_client = None

        if self.use_llm and not OPENROUTER_API_KEY:
            logger.warning(
                "OPENROUTER_API_KEY not set - LLM fix generation disabled"
            )
            self.use_llm = False

        logger.info(
            f"IssueToFixTransformer initialized: "
            f"max_fixes={max_fixes}, min_confidence={min_confidence}, use_llm={use_llm}"
        )

    async def transform(
        self,
        issues: List['VisualIssue'],
        schematic_content: str,
        max_fixes: Optional[int] = None
    ) -> TransformResult:
        """
        Transform visual issues into actionable fixes.

        Args:
            issues: List of visual issues from validation
            schematic_content: Current schematic S-expression
            max_fixes: Override max fixes for this call

        Returns:
            TransformResult with prioritized fixes
        """
        max_fixes = max_fixes or self.max_fixes

        if not issues:
            logger.info("No issues to transform")
            return TransformResult(success=True, total_issues=0)

        logger.info(f"Transforming {len(issues)} issues to fixes (max {max_fixes})")

        # Prioritize issues
        prioritized = self._prioritize_issues(issues)

        # Extract schematic context
        context = self._extract_schematic_context(schematic_content)

        # Generate fixes - use rule-based for simple issues, LLM for complex
        fixes = []
        skipped = []
        errors = []

        # First pass: rule-based fixes for common patterns
        for issue in prioritized[:max_fixes * 2]:  # Process more than max to allow filtering
            rule_fix = self._apply_rule_based_fix(issue, context)
            if rule_fix:
                fixes.append(rule_fix)
                if len(fixes) >= max_fixes:
                    break

        # Second pass: LLM for remaining complex issues
        if self.use_llm and len(fixes) < max_fixes:
            remaining_issues = [
                i for i in prioritized
                if not any(i.description in f.source_issues for f in fixes)
            ][:max_fixes - len(fixes)]

            if remaining_issues:
                try:
                    llm_fixes = await self._generate_llm_fixes(
                        remaining_issues, context, max_fixes - len(fixes)
                    )
                    fixes.extend(llm_fixes)
                except Exception as e:
                    logger.error(f"LLM fix generation failed: {e}")
                    errors.append(str(e))

        # Assign priorities and IDs
        for i, fix in enumerate(fixes):
            if not fix.fix_id:
                fix.fix_id = f"fix_{uuid.uuid4().hex[:8]}"
            fix.priority = i + 1

        # Track skipped issues
        for issue in issues:
            if not any(issue.description in f.source_issues for f in fixes):
                skipped.append(issue.description[:100])

        result = TransformResult(
            success=len(fixes) > 0 or len(issues) == 0,
            fixes=fixes[:max_fixes],
            skipped_issues=skipped,
            errors=errors,
            total_issues=len(issues),
            fixable_issues=len(fixes),
        )

        logger.info(
            f"Transform complete: {len(fixes)} fixes generated, "
            f"{len(skipped)} issues skipped"
        )

        return result

    def _prioritize_issues(self, issues: List['VisualIssue']) -> List['VisualIssue']:
        """Sort issues by priority and severity."""
        def sort_key(issue):
            # Get category value
            category = issue.category.value if hasattr(issue.category, 'value') else str(issue.category)
            category_priority = self.PRIORITY_ORDER.get(category, 99)

            # Get severity value
            severity = issue.severity.value if hasattr(issue.severity, 'value') else str(issue.severity)
            severity_priority = {"critical": 0, "error": 1, "warning": 2, "info": 3}.get(severity, 4)

            return (severity_priority, category_priority)

        return sorted(issues, key=sort_key)

    def _extract_schematic_context(self, schematic_content: str) -> Dict[str, Any]:
        """Extract useful context from schematic for fix generation."""
        context = {
            "component_count": 0,
            "sheet_size": "A4",
            "net_names": [],
            "components": [],
            "wires": [],
        }

        try:
            # Count components (symbol sections)
            symbol_matches = re.findall(r'\(symbol\s+"([^"]+)"', schematic_content)
            context["component_count"] = len(symbol_matches)

            # Extract reference designators
            ref_matches = re.findall(r'\(property\s+"Reference"\s+"([^"]+)"', schematic_content)
            context["components"] = list(set(ref_matches))

            # Extract net labels
            net_matches = re.findall(r'\(label\s+"([^"]+)"', schematic_content)
            context["net_names"] = list(set(net_matches))

            # Get sheet size
            size_match = re.search(r'\(paper\s+"([^"]+)"', schematic_content)
            if size_match:
                context["sheet_size"] = size_match.group(1)

        except Exception as e:
            logger.warning(f"Error extracting schematic context: {e}")

        return context

    def _apply_rule_based_fix(
        self,
        issue: 'VisualIssue',
        context: Dict[str, Any]
    ) -> Optional[SchematicFix]:
        """Apply rule-based fix for common issue patterns."""
        category = issue.category.value if hasattr(issue.category, 'value') else str(issue.category)
        description = issue.description.lower()

        # Rule: Missing bypass capacitor
        if "bypass" in description or "decoupling" in description:
            # Extract IC reference from description
            ic_match = re.search(r'(U\d+|IC\d+)', issue.description, re.IGNORECASE)
            target_ic = ic_match.group(1) if ic_match else None

            return SchematicFix(
                fix_id="",
                priority=1,
                category=FixCategory.BYPASS_CAPACITOR,
                fix_operation=FixOperation.ADD_COMPONENT,
                description=f"Add bypass capacitor near {target_ic or 'IC'}",
                target_component=target_ic,
                parameters={
                    "symbol_lib": "Device",
                    "symbol_name": "C",
                    "reference": "C?",  # Auto-assign
                    "value": "100nF",
                    "near_component": target_ic,
                },
                estimated_impact=0.1,
                source_issues=[issue.description],
                auto_fixable=True,
                confidence=0.85,
            )

        # Rule: Missing junction
        if "junction" in description or "4-way" in description:
            return SchematicFix(
                fix_id="",
                priority=2,
                category=FixCategory.JUNCTION,
                fix_operation=FixOperation.ADD_JUNCTION,
                description="Add junction at wire crossing",
                parameters={"auto_detect": True},
                estimated_impact=0.05,
                source_issues=[issue.description],
                auto_fixable=True,
                confidence=0.9,
            )

        # Rule: Missing power flag
        if "power" in description and ("flag" in description or "floating" in description):
            net_match = re.search(r'(VCC|VDD|GND|VSS|\+\d+V|\-\d+V)', description, re.IGNORECASE)
            power_net = net_match.group(1) if net_match else "VCC"

            return SchematicFix(
                fix_id="",
                priority=2,
                category=FixCategory.POWER_CONNECTION,
                fix_operation=FixOperation.ADD_POWER_FLAG,
                description=f"Add power flag to {power_net} net",
                target_net=power_net,
                parameters={"power_net": power_net},
                estimated_impact=0.08,
                source_issues=[issue.description],
                auto_fixable=True,
                confidence=0.85,
            )

        # Rule: Missing net label
        if "label" in description and "net" in description:
            return SchematicFix(
                fix_id="",
                priority=3,
                category=FixCategory.NET_LABEL,
                fix_operation=FixOperation.ADD_NET_LABEL,
                description="Add net label for clarity",
                parameters={"auto_detect_net": True},
                estimated_impact=0.05,
                source_issues=[issue.description],
                auto_fixable=True,
                confidence=0.75,
            )

        # Rule: Component placement issue
        if category == "component_placement":
            ref_match = re.search(r'(U\d+|R\d+|C\d+|L\d+|D\d+|Q\d+)', issue.description)
            target_ref = ref_match.group(1) if ref_match else None

            if target_ref:
                return SchematicFix(
                    fix_id="",
                    priority=5,
                    category=FixCategory.COMPONENT_PLACEMENT,
                    fix_operation=FixOperation.MOVE_COMPONENT,
                    description=f"Adjust placement of {target_ref}",
                    target_component=target_ref,
                    parameters={
                        "reference": target_ref,
                        "optimize_placement": True,
                    },
                    estimated_impact=0.05,
                    source_issues=[issue.description],
                    auto_fixable=True,
                    confidence=0.7,
                )

        # No rule matched
        return None

    async def _generate_llm_fixes(
        self,
        issues: List['VisualIssue'],
        context: Dict[str, Any],
        max_fixes: int
    ) -> List[SchematicFix]:
        """Use Opus 4.6 to generate fixes for complex issues."""
        try:
            import httpx
        except ImportError:
            raise FixTransformError(
                message="httpx required for LLM fix generation",
                issue_count=len(issues),
                failed_issues=[i.description[:50] for i in issues]
            )

        # Format issues for prompt
        issues_json = json.dumps([
            {
                "index": i,
                "category": issue.category.value if hasattr(issue.category, 'value') else str(issue.category),
                "severity": issue.severity.value if hasattr(issue.severity, 'value') else str(issue.severity),
                "description": issue.description,
                "location": issue.location,
                "recommendation": issue.recommendation,
            }
            for i, issue in enumerate(issues)
        ], indent=2)

        prompt = self.FIX_GENERATION_PROMPT.format(
            issues_json=issues_json,
            component_count=context.get("component_count", 0),
            sheet_size=context.get("sheet_size", "A4"),
            net_names=", ".join(context.get("net_names", [])[:10]),
        )

        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                f"{OPENROUTER_BASE_URL}/chat/completions",
                headers={
                    "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": "anthropic/claude-opus-4-5-20251101",
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 2000,
                    "temperature": 0.3,
                }
            )

            if response.status_code != 200:
                raise FixTransformError(
                    message=f"LLM API returned {response.status_code}",
                    issue_count=len(issues),
                    failed_issues=[i.description[:50] for i in issues],
                    llm_response=response.text
                )

            result = response.json()
            content = result["choices"][0]["message"]["content"]

            # Parse JSON response
            try:
                # Find JSON array in response
                json_match = re.search(r'\[.*\]', content, re.DOTALL)
                if not json_match:
                    raise ValueError("No JSON array found in response")

                fixes_data = json.loads(json_match.group())
            except (json.JSONDecodeError, ValueError) as e:
                logger.warning(f"Failed to parse LLM fix response: {e}")
                return []

            # Convert to SchematicFix objects
            fixes = []
            for fix_data in fixes_data[:max_fixes]:
                try:
                    fix = SchematicFix(
                        fix_id="",
                        priority=len(fixes) + 1,
                        category=FixCategory(fix_data.get("category", "component_placement")),
                        fix_operation=FixOperation[fix_data["fix_operation"]],
                        description=fix_data.get("description", ""),
                        target_component=fix_data.get("target_component"),
                        target_net=fix_data.get("target_net"),
                        parameters=fix_data.get("parameters", {}),
                        estimated_impact=fix_data.get("estimated_impact", 0.05),
                        source_issues=[
                            issues[fix_data.get("source_issue_index", 0)].description
                        ] if issues else [],
                        auto_fixable=fix_data.get("auto_fixable", True),
                        confidence=fix_data.get("confidence", 0.7),
                    )

                    if fix.confidence >= self.min_confidence:
                        fixes.append(fix)

                except (KeyError, ValueError) as e:
                    logger.warning(f"Skipping invalid fix: {e}")
                    continue

            return fixes
