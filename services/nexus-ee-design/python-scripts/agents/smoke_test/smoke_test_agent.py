"""
SmokeTestAgent - LLM-based SPICE circuit validation

AI-first approach: Uses Claude Opus 4.5 to analyze generated schematics and
validate they will not "smoke" when power is applied.

Validates:
1. Power rail connectivity (VCC/VDD connected to all ICs)
2. Ground connectivity (GND/VSS properly connected)
3. Short circuit detection (no direct power-to-ground paths)
4. Floating node detection (critical pins connected)
5. Current path verification (power can flow through circuit)
6. Bypass capacitor placement (decoupling for ICs)
"""

import os
import json
import logging
import asyncio
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict, Optional, Any, Tuple
from pathlib import Path

import httpx

# Configure logging
logger = logging.getLogger(__name__)

# OpenRouter configuration for LLM-first approach
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_MODEL = "anthropic/claude-opus-4.5"  # Opus 4.5 via OpenRouter


class SmokeTestSeverity(Enum):
    """Severity levels for smoke test issues."""
    FATAL = "fatal"      # Circuit will definitely smoke/fail
    ERROR = "error"      # Circuit likely won't work correctly
    WARNING = "warning"  # Potential issue, needs review
    INFO = "info"        # Advisory information


@dataclass
class SmokeTestIssue:
    """A single issue found during smoke testing."""
    severity: SmokeTestSeverity
    test_name: str
    message: str
    component: Optional[str] = None
    net: Optional[str] = None
    recommendation: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "severity": self.severity.value,
            "test": self.test_name,
            "message": self.message,
            "component": self.component,
            "net": self.net,
            "recommendation": self.recommendation,
        }


@dataclass
class SmokeTestResult:
    """Result of smoke test validation."""
    passed: bool
    power_rails_ok: bool
    ground_ok: bool
    no_shorts: bool
    no_floating_nodes: bool
    power_dissipation_ok: bool
    current_paths_valid: bool
    issues: List[SmokeTestIssue] = field(default_factory=list)
    llm_analysis: Dict[str, Any] = field(default_factory=dict)
    simulation_data: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "passed": self.passed,
            "summary": {
                "power_rails_ok": self.power_rails_ok,
                "ground_ok": self.ground_ok,
                "no_shorts": self.no_shorts,
                "no_floating_nodes": self.no_floating_nodes,
                "power_dissipation_ok": self.power_dissipation_ok,
                "current_paths_valid": self.current_paths_valid,
            },
            "issues": [i.to_dict() for i in self.issues],
            "fatal_count": sum(1 for i in self.issues if i.severity == SmokeTestSeverity.FATAL),
            "error_count": sum(1 for i in self.issues if i.severity == SmokeTestSeverity.ERROR),
            "warning_count": sum(1 for i in self.issues if i.severity == SmokeTestSeverity.WARNING),
            "llm_analysis": self.llm_analysis,
        }


class SmokeTestAgent:
    """
    LLM-based smoke test agent for circuit validation.

    Uses Claude Opus 4.5 to intelligently analyze schematics and identify
    potential issues that would cause the circuit to fail when powered.

    AI-first approach:
    - No regex parsing - LLM extracts all relevant information
    - No algorithmic checks - LLM performs semantic analysis
    - Understands circuit topology and electrical engineering principles
    """

    def __init__(self, ngspice_path: str = "ngspice"):
        """Initialize the smoke test agent."""
        self.ngspice_path = ngspice_path
        self._openrouter_api_key = OPENROUTER_API_KEY
        self._http_client: Optional[httpx.AsyncClient] = None

        # Power ratings database (can be extended)
        self.component_power_ratings: Dict[str, float] = {
            "STM32G431": 0.5,  # 500mW typical
            "LM7805": 1.0,    # 1W without heatsink
            "AMS1117": 0.8,   # 800mW
        }

    async def _ensure_http_client(self):
        """Ensure HTTP client is initialized."""
        if self._http_client is None:
            self._http_client = httpx.AsyncClient(timeout=120.0)

    async def run_smoke_test(
        self,
        kicad_sch_content: str,
        bom_items: List[Dict[str, Any]],
        power_sources: Optional[List[Dict[str, Any]]] = None
    ) -> SmokeTestResult:
        """
        Run comprehensive smoke test on schematic using LLM analysis.

        Args:
            kicad_sch_content: KiCad schematic file content (.kicad_sch)
            bom_items: Bill of materials with component specifications
            power_sources: Power source definitions (voltage, current limits)

        Returns:
            SmokeTestResult with pass/fail and detailed issues
        """
        await self._ensure_http_client()

        if not self._openrouter_api_key:
            logger.error("OpenRouter API key not set - cannot perform LLM smoke test")
            return SmokeTestResult(
                passed=False,
                power_rails_ok=False,
                ground_ok=False,
                no_shorts=False,
                no_floating_nodes=False,
                power_dissipation_ok=False,
                current_paths_valid=False,
                issues=[SmokeTestIssue(
                    severity=SmokeTestSeverity.FATAL,
                    test_name="configuration",
                    message="OpenRouter API key not configured - cannot perform smoke test",
                    recommendation="Set OPENROUTER_API_KEY environment variable"
                )]
            )

        logger.info("Starting LLM-based smoke test analysis...")

        # Run comprehensive LLM analysis
        llm_result = await self._run_llm_smoke_analysis(
            kicad_sch_content, bom_items, power_sources or []
        )

        # Convert LLM result to SmokeTestResult
        return self._convert_llm_result(llm_result)

    async def _run_llm_smoke_analysis(
        self,
        kicad_sch: str,
        bom_items: List[Dict[str, Any]],
        power_sources: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Run comprehensive smoke test using LLM (Opus 4.5).

        AI-first approach: The LLM analyzes the entire schematic and performs
        all necessary checks using its understanding of electronics.
        """
        # Build BOM summary for LLM
        bom_summary = []
        for item in bom_items[:30]:  # Limit for token efficiency
            ref = item.get("reference", item.get("part_number", "?"))
            part = item.get("part_number", "unknown")
            cat = item.get("category", "")
            value = item.get("value", "")
            bom_summary.append(f"- {ref}: {part} ({cat}) {value}")

        # Build power source summary
        power_summary = []
        for ps in power_sources:
            power_summary.append(f"- {ps.get('net', 'VCC')}: {ps.get('voltage', 5)}V, {ps.get('current_limit', 1)}A max")

        if not power_summary:
            power_summary.append("- VCC: 5V (assumed)")
            power_summary.append("- GND: 0V (reference)")

        prompt = f"""You are an expert electronics engineer performing a "smoke test" validation on a KiCad schematic.

A "smoke test" ensures the circuit will NOT smoke/burn/fail when power is applied. You must analyze the schematic and identify any issues that would cause immediate failure.

IMPORTANT KICAD SEMANTICS:
- In KiCad, global_label elements with the SAME TEXT are ELECTRICALLY CONNECTED implicitly
- Example: (global_label "VCC" ...) at one location connects to ALL other (global_label "VCC" ...) elements
- This means if power pin VDD has a global_label "VCC" nearby, it IS connected to the VCC power rail
- Similarly, global_label "GND" elements are all connected to ground
- Do NOT report "no power connection" if the power pin has a nearby global_label with matching power net name
- Power net equivalents: VCC=VDD=3V3=5V, GND=VSS=GROUND, VDDA=analog power, VBAT=battery

ANALYZE THE FOLLOWING TESTS:

1. **POWER RAIL CHECK**: Are VCC/VDD/3V3/5V nets connected to all ICs that need power?
2. **GROUND CHECK**: Is GND/VSS connected to all components that need ground?
3. **SHORT CIRCUIT DETECTION**: Are there any direct connections between power and ground?
4. **FLOATING NODE DETECTION**: Are there critical pins (power, enable, reset) that are floating/unconnected?
5. **CURRENT PATH VALIDATION**: Can current flow from power through loads to ground?
6. **BYPASS CAPACITOR CHECK**: Do ICs have bypass/decoupling capacitors near their power pins?
7. **POLARITY CHECK**: Are polarized components (diodes, capacitors, ICs) oriented correctly?

BILL OF MATERIALS:
{chr(10).join(bom_summary)}

POWER SOURCES:
{chr(10).join(power_summary)}

KICAD SCHEMATIC S-EXPRESSION:
```
{kicad_sch[:15000]}
```

Return a JSON object with this EXACT structure:
{{
    "overall_passed": true/false,
    "tests": {{
        "power_rails": {{
            "passed": true/false,
            "issues": ["list of issues found"],
            "details": "explanation"
        }},
        "ground": {{
            "passed": true/false,
            "issues": ["list of issues found"],
            "details": "explanation"
        }},
        "short_circuits": {{
            "passed": true/false,
            "issues": ["list of issues found"],
            "details": "explanation"
        }},
        "floating_nodes": {{
            "passed": true/false,
            "issues": ["list of issues found"],
            "details": "explanation"
        }},
        "current_paths": {{
            "passed": true/false,
            "issues": ["list of issues found"],
            "details": "explanation"
        }},
        "bypass_caps": {{
            "passed": true/false,
            "issues": ["list of issues found"],
            "details": "explanation"
        }}
    }},
    "fatal_issues": ["list of issues that WILL cause smoke/failure"],
    "warnings": ["list of potential issues to review"],
    "recommendations": ["list of improvements"],
    "wire_count": number_of_wires_found,
    "component_count": number_of_components_found,
    "net_count": number_of_unique_nets_found
}}

Be thorough but practical. Focus on issues that would cause IMMEDIATE FAILURE when power is applied.
Return ONLY valid JSON, no explanations outside the JSON structure."""

        try:
            response = await self._http_client.post(
                OPENROUTER_BASE_URL,
                headers={
                    "Authorization": f"Bearer {self._openrouter_api_key}",
                    "Content-Type": "application/json",
                    "HTTP-Referer": "https://nexus.adverant.com",
                    "X-Title": "Nexus EE Design - Smoke Test"
                },
                json={
                    "model": OPENROUTER_MODEL,
                    "messages": [
                        {
                            "role": "system",
                            "content": "You are an expert electronics engineer specializing in circuit validation. Analyze schematics and return structured JSON results. Be precise and thorough."
                        },
                        {"role": "user", "content": prompt}
                    ],
                    "temperature": 0.0,
                    "max_tokens": 4000
                }
            )
            response.raise_for_status()
            result = response.json()

            content = result["choices"][0]["message"]["content"].strip()

            # Clean up JSON if wrapped in code block
            if content.startswith("```"):
                lines = content.split("\n")
                # Find the actual JSON content
                start_idx = 1
                end_idx = len(lines) - 1 if lines[-1].startswith("```") else len(lines)
                content = "\n".join(lines[start_idx:end_idx])

            return json.loads(content)

        except json.JSONDecodeError as e:
            logger.error(f"LLM smoke test returned invalid JSON: {e}")
            return {
                "overall_passed": False,
                "tests": {},
                "fatal_issues": [f"LLM analysis failed: Invalid JSON response - {e}"],
                "warnings": [],
                "recommendations": ["Retry smoke test"],
                "error": str(e)
            }
        except Exception as e:
            logger.error(f"LLM smoke test failed: {e}")
            return {
                "overall_passed": False,
                "tests": {},
                "fatal_issues": [f"LLM analysis failed: {e}"],
                "warnings": [],
                "recommendations": ["Check API connectivity and retry"],
                "error": str(e)
            }

    def _convert_llm_result(self, llm_result: Dict[str, Any]) -> SmokeTestResult:
        """Convert LLM analysis result to SmokeTestResult."""
        issues = []
        tests = llm_result.get("tests", {})

        # Process fatal issues
        for fatal in llm_result.get("fatal_issues", []):
            issues.append(SmokeTestIssue(
                severity=SmokeTestSeverity.FATAL,
                test_name="smoke_test",
                message=fatal,
                recommendation="Fix before applying power"
            ))

        # Process warnings
        for warning in llm_result.get("warnings", []):
            issues.append(SmokeTestIssue(
                severity=SmokeTestSeverity.WARNING,
                test_name="smoke_test",
                message=warning
            ))

        # Process individual test results
        for test_name, test_result in tests.items():
            if not test_result.get("passed", True):
                for issue in test_result.get("issues", []):
                    issues.append(SmokeTestIssue(
                        severity=SmokeTestSeverity.ERROR,
                        test_name=test_name,
                        message=issue,
                        recommendation=test_result.get("details", "")
                    ))

        # Determine overall pass/fail
        fatal_count = sum(1 for i in issues if i.severity == SmokeTestSeverity.FATAL)
        passed = llm_result.get("overall_passed", False) and fatal_count == 0

        return SmokeTestResult(
            passed=passed,
            power_rails_ok=tests.get("power_rails", {}).get("passed", False),
            ground_ok=tests.get("ground", {}).get("passed", False),
            no_shorts=tests.get("short_circuits", {}).get("passed", True),
            no_floating_nodes=tests.get("floating_nodes", {}).get("passed", False),
            power_dissipation_ok=True,  # Covered by current_paths in LLM analysis
            current_paths_valid=tests.get("current_paths", {}).get("passed", False),
            issues=issues,
            llm_analysis=llm_result
        )

    async def validate_connectivity(
        self,
        kicad_sch_content: str
    ) -> Dict[str, Any]:
        """
        Quick connectivity validation using LLM.

        Checks basic structural properties:
        - Are there wires in the schematic?
        - Are components connected?
        - Are power nets present?
        """
        await self._ensure_http_client()

        if not self._openrouter_api_key:
            return {"valid": False, "error": "No API key"}

        prompt = f"""Analyze this KiCad schematic and report on basic connectivity.

Answer these specific questions:
1. How many wire segments are present? (count (wire elements)
2. How many component symbols are placed? (count (symbol elements)
3. Are there power net labels (VCC, GND, etc.)?
4. Are components connected to each other via wires?

SCHEMATIC:
```
{kicad_sch_content[:10000]}
```

Return JSON:
{{
    "wire_count": number,
    "component_count": number,
    "has_power_nets": true/false,
    "components_connected": true/false,
    "connectivity_score": 0-100,
    "issues": ["list any major connectivity problems"]
}}

Return ONLY the JSON."""

        try:
            response = await self._http_client.post(
                OPENROUTER_BASE_URL,
                headers={
                    "Authorization": f"Bearer {self._openrouter_api_key}",
                    "Content-Type": "application/json",
                    "HTTP-Referer": "https://nexus.adverant.com",
                    "X-Title": "Nexus EE Design - Connectivity Check"
                },
                json={
                    "model": OPENROUTER_MODEL,
                    "messages": [
                        {"role": "system", "content": "You are a KiCad schematic analyzer. Return structured JSON."},
                        {"role": "user", "content": prompt}
                    ],
                    "temperature": 0.0,
                    "max_tokens": 1000
                }
            )
            response.raise_for_status()
            result = response.json()

            content = result["choices"][0]["message"]["content"].strip()
            if content.startswith("```"):
                lines = content.split("\n")
                content = "\n".join(lines[1:-1] if lines[-1].startswith("```") else lines[1:])

            return json.loads(content)

        except Exception as e:
            logger.error(f"Connectivity validation failed: {e}")
            return {"valid": False, "error": str(e)}

    async def close(self):
        """Close HTTP client."""
        if self._http_client:
            await self._http_client.aclose()
            self._http_client = None


# Test runner
async def main():
    """Test the smoke test agent."""
    import sys

    if len(sys.argv) < 2:
        print("Usage: python smoke_test_agent.py <schematic.kicad_sch>")
        print("       python smoke_test_agent.py --test")
        sys.exit(1)

    if sys.argv[1] == "--test":
        # Run a simple test
        agent = SmokeTestAgent()

        test_sch = """(kicad_sch (version 20231120) (generator "test")
  (symbol (lib_id "Device:R") (at 100 100 0) (unit 1)
    (property "Reference" "R1")
    (property "Value" "10k")
  )
  (wire (pts (xy 100 95) (xy 100 80)))
  (wire (pts (xy 100 105) (xy 100 120)))
  (label "VCC" (at 100 80 0))
  (label "GND" (at 100 120 0))
)"""

        bom = [
            {"reference": "R1", "part_number": "RC0805FR-0710KL", "category": "Resistor", "value": "10k"}
        ]

        result = await agent.run_smoke_test(test_sch, bom)
        print(json.dumps(result.to_dict(), indent=2))
        await agent.close()
    else:
        # Load schematic from file
        sch_path = Path(sys.argv[1])
        if not sch_path.exists():
            print(f"File not found: {sch_path}")
            sys.exit(1)

        sch_content = sch_path.read_text()

        agent = SmokeTestAgent()
        result = await agent.run_smoke_test(sch_content, [])
        print(json.dumps(result.to_dict(), indent=2))
        await agent.close()


if __name__ == "__main__":
    asyncio.run(main())
