"""
Ideation Artifact Generator Agent.

LLM-powered generation of pre-schematic design documentation.
Uses Claude Opus 4.6 or other LLMs via OpenRouter to generate
comprehensive design artifacts that provide context for schematic generation.

Author: Nexus EE Design Team
"""

import logging
import json
import os
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Dict, Any, List
from datetime import datetime

import httpx

logger = logging.getLogger(__name__)


class ArtifactType(str, Enum):
    """Types of ideation artifacts that can be generated."""
    SYSTEM_OVERVIEW = "system_overview"
    EXECUTIVE_SUMMARY = "executive_summary"
    ARCHITECTURE_DIAGRAM = "architecture_diagram"
    SCHEMATIC_SPEC = "schematic_spec"
    POWER_SPEC = "power_spec"
    MCU_SPEC = "mcu_spec"
    SENSING_SPEC = "sensing_spec"
    COMMUNICATION_SPEC = "communication_spec"
    CONNECTOR_SPEC = "connector_spec"
    INTERFACE_SPEC = "interface_spec"
    BOM = "bom"
    COMPONENT_SELECTION = "component_selection"
    CALCULATIONS = "calculations"
    PCB_SPEC = "pcb_spec"
    STACKUP = "stackup"
    MANUFACTURING_GUIDE = "manufacturing_guide"
    FIRMWARE_SPEC = "firmware_spec"
    AI_INTEGRATION = "ai_integration"
    TEST_PLAN = "test_plan"
    RESEARCH_PAPER = "research_paper"
    PATENT = "patent"
    COMPLIANCE_DOC = "compliance_doc"
    CUSTOM = "custom"


class ArtifactCategory(str, Enum):
    """Categories of ideation artifacts."""
    ARCHITECTURE = "architecture"
    SCHEMATIC = "schematic"
    COMPONENT = "component"
    PCB = "pcb"
    FIRMWARE = "firmware"
    VALIDATION = "validation"
    RESEARCH = "research"


@dataclass
class GeneratedArtifact:
    """Result of artifact generation."""
    artifact_type: ArtifactType
    category: ArtifactCategory
    name: str
    content: str
    content_format: str  # 'markdown', 'text', 'json', 'mermaid', 'csv'
    generation_prompt: str
    generation_model: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    tokens_used: int = 0
    generation_time_ms: int = 0


@dataclass
class GenerationConfig:
    """Configuration for artifact generation."""
    model: str = "anthropic/claude-opus-4.6"
    max_tokens: int = 8192
    temperature: float = 0.3
    timeout_seconds: int = 120
    include_thinking: bool = True


# Pre-defined generation prompt templates
ARTIFACT_PROMPTS: Dict[str, str] = {
    "system_overview": """Generate a comprehensive system overview document for the following hardware project.

PROJECT INFORMATION:
- Project Name: {project_name}
- Project Type: {project_type}
- Target Specifications: {specifications}
- Key Requirements: {requirements}
- Subsystems: {subsystems}

Generate a detailed markdown document that includes:

1. **Executive Summary** (2-3 paragraphs)
   - Problem statement and solution approach
   - Key differentiators and innovations
   - Target applications and use cases

2. **System Architecture Overview**
   - High-level block diagram description
   - Major functional blocks and their interactions
   - Data/signal flow between subsystems

3. **Key Specifications Table**
   | Parameter | Target | Units | Notes |
   |-----------|--------|-------|-------|
   (Include all relevant electrical, mechanical, environmental specs)

4. **Design Constraints**
   - Physical constraints (size, weight, form factor)
   - Electrical constraints (voltage rails, current limits)
   - Environmental constraints (temperature, humidity)
   - Cost targets and volume considerations

5. **Trade-off Analysis**
   - Key design decisions with alternatives considered
   - Rationale for selected approaches
   - Risks and mitigations

6. **Preliminary Risk Assessment**
   | Risk | Probability | Impact | Mitigation |
   |------|-------------|--------|------------|

7. **Cost Estimate Summary**
   - BOM cost target
   - Manufacturing cost estimate
   - Development effort estimate

Format the output as clean markdown suitable for technical documentation.""",

    "schematic_spec": """Generate a detailed schematic specification for a hardware subsystem.

SUBSYSTEM INFORMATION:
- Subsystem Name: {subsystem_name}
- Function: {function}
- Input Requirements: {inputs}
- Output Requirements: {outputs}
- Performance Targets: {targets}
- Interface Constraints: {interfaces}
- Related Subsystems: {related_subsystems}

Generate a comprehensive specification document that includes:

1. **Subsystem Overview**
   - Purpose and functionality
   - Key performance parameters
   - Interface summary

2. **Topology Selection**
   - Chosen topology/architecture
   - Alternative topologies considered
   - Rationale for selection

3. **Component Selection**
   For each major component:
   - Part number (manufacturer PN)
   - Key specifications
   - Selection rationale
   - Alternatives considered

4. **Pin-by-Pin Connections**
   Detailed connection table:
   | Component | Pin | Signal | Connected To | Notes |
   |-----------|-----|--------|--------------|-------|

5. **Power Requirements**
   - Input voltage range
   - Current consumption (typical, max)
   - Power dissipation
   - Thermal considerations

6. **Signal Routing Considerations**
   - Critical signals requiring special attention
   - Impedance requirements
   - Crosstalk/noise considerations

7. **Protection Circuits**
   - Overvoltage/undervoltage protection
   - Overcurrent protection
   - ESD protection
   - Reverse polarity protection

8. **Decoupling Strategy**
   - Capacitor placement guidelines
   - Value selection rationale
   - High-frequency considerations

9. **Design Calculations**
   Include relevant calculations for:
   - Feedback networks
   - Filter cutoff frequencies
   - Current limits
   - Thermal margins

10. **Testing & Verification Checklist**
    [ ] Item to verify
    [ ] Expected result
    [ ] Pass criteria

Format as markdown with code blocks for calculations.""",

    "bom": """Generate a detailed Bill of Materials for a hardware project.

PROJECT INFORMATION:
- Project Name: {project_name}
- Project Type: {project_type}
- Subsystems: {subsystems}
- Target Cost: {cost_target}
- Production Volume: {production_volume}
- Quality Level: {quality_level}

Generate a comprehensive BOM that includes:

1. **BOM Summary**
   - Total component count
   - Total BOM cost (at target volume)
   - Cost breakdown by subsystem

2. **Component Table**
   | Ref Des | Description | Manufacturer | MPN | Package | Qty | Unit Cost | Ext Cost | Vendor |
   |---------|-------------|--------------|-----|---------|-----|-----------|----------|--------|

3. **By Subsystem**
   Group components by subsystem with subtotals.

4. **Critical Components**
   List components with:
   - Long lead times (>8 weeks)
   - Single source parts
   - End-of-life risk
   - Include alternatives where available

5. **Alternative Components**
   For critical/expensive parts, provide drop-in alternatives:
   | Original | Alternative | Notes | Cost Delta |
   |----------|-------------|-------|------------|

6. **Vendor Distribution**
   - Primary vendors (DigiKey, Mouser, LCSC)
   - Percentage by vendor
   - Minimum order considerations

7. **Assembly Considerations**
   - SMD vs through-hole ratio
   - BGA/QFN packages requiring special handling
   - Component height constraints
   - Special handling requirements

Format as markdown with tables. Include actual part numbers where possible.""",

    "architecture_diagram": """Generate a Mermaid architecture diagram for a hardware system.

DIAGRAM INFORMATION:
- Diagram Type: {diagram_type}
- System Context: {context}
- Focus Area: {focus}
- Level of Detail: {detail_level}

Generate a Mermaid diagram that shows:

1. **Major Functional Blocks**
   - Use descriptive names
   - Group related components
   - Show hierarchy where appropriate

2. **Signal/Data Flow**
   - Direction of data flow
   - Signal types (power, analog, digital, communication)
   - Bus connections

3. **Power Distribution** (if applicable)
   - Power rails
   - Voltage levels
   - Current paths

4. **Communication Buses**
   - Protocol labels (SPI, I2C, UART, CAN, etc.)
   - Bus topology

5. **External Interfaces**
   - Connectors
   - User interfaces
   - Debug/programming interfaces

Output the diagram in Mermaid format suitable for rendering.
Use appropriate diagram type (flowchart, block diagram, sequence diagram) based on the content.

Example format:
```mermaid
graph TB
    subgraph "Power Supply"
        VIN[Battery Input] --> BUCK[Buck Converter]
        BUCK --> LDO[LDO Regulator]
    end
    subgraph "Processing"
        MCU[Microcontroller]
    end
    LDO -->|3.3V| MCU
```

Include styling for clarity.""",

    "calculations": """Generate detailed design calculations for a hardware subsystem.

CALCULATION CONTEXT:
- Subsystem: {subsystem}
- Calculation Type: {calculation_type}
- Input Parameters: {parameters}
- Target Specifications: {targets}

Generate comprehensive calculations including:

1. **Problem Statement**
   - What we're calculating
   - Why it's important
   - Target values

2. **Input Parameters**
   List all input values with units and sources.

3. **Equations Used**
   Show each equation with:
   - Equation number
   - Variable definitions
   - Source/reference

4. **Step-by-Step Calculations**
   Work through each calculation showing:
   - Substituted values
   - Intermediate results
   - Final answers with units

5. **Results Summary**
   | Parameter | Calculated | Target | Margin | Status |
   |-----------|------------|--------|--------|--------|

6. **Sensitivity Analysis**
   How results change with component tolerances.

7. **Safety Margins**
   - Operating margins
   - Absolute maximum ratings
   - Derating applied

8. **Component Selection**
   Based on calculations, specify:
   - Exact component values needed
   - Standard value selections
   - Tolerance requirements

Format with LaTeX-style equations where appropriate.
Use code blocks for clarity.""",

    "test_plan": """Generate a comprehensive test plan for hardware verification.

PROJECT INFORMATION:
- Project Name: {project_name}
- Project Type: {project_type}
- Test Scope: {scope}
- Test Environment: {environment}

Generate a detailed test plan including:

1. **Test Overview**
   - Objectives
   - Scope and limitations
   - Pass/fail criteria

2. **Test Equipment Required**
   | Equipment | Model/Spec | Purpose | Calibration Due |
   |-----------|------------|---------|-----------------|

3. **Functional Tests**
   For each function:
   - Test ID
   - Description
   - Setup procedure
   - Steps
   - Expected results
   - Pass criteria

4. **Parametric Tests**
   | Test ID | Parameter | Condition | Min | Typ | Max | Unit |
   |---------|-----------|-----------|-----|-----|-----|------|

5. **Environmental Tests**
   - Temperature range testing
   - Humidity testing
   - Vibration/shock (if applicable)

6. **EMC Pre-compliance**
   - Radiated emissions
   - Conducted emissions
   - Susceptibility

7. **Safety Tests**
   - Dielectric withstand
   - Ground continuity
   - Fault conditions

8. **Test Sequence**
   Recommended order of testing with dependencies.

9. **Data Recording**
   - Parameters to record
   - Format requirements
   - Storage/retention

10. **Failure Analysis Procedure**
    Steps when a test fails.

Format as markdown suitable for a test engineering team.""",
}


class IdeationArtifactGenerator:
    """
    LLM-powered generator for ideation artifacts.

    Uses OpenRouter API to access various LLM models for generating
    design documentation, specifications, and analysis.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        config: Optional[GenerationConfig] = None,
    ):
        """
        Initialize the artifact generator.

        Args:
            api_key: OpenRouter API key. If not provided, reads from OPENROUTER_API_KEY env var.
            config: Generation configuration. Uses defaults if not provided.
        """
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        if not self.api_key:
            raise ValueError("OpenRouter API key is required. Set OPENROUTER_API_KEY environment variable.")

        self.config = config or GenerationConfig()
        self.base_url = "https://openrouter.ai/api/v1"

        logger.info(f"IdeationArtifactGenerator initialized with model: {self.config.model}")

    async def generate(
        self,
        artifact_type: ArtifactType,
        category: ArtifactCategory,
        name: str,
        context: Dict[str, Any],
        custom_prompt: Optional[str] = None,
    ) -> GeneratedArtifact:
        """
        Generate an ideation artifact using LLM.

        Args:
            artifact_type: Type of artifact to generate
            category: Category of the artifact
            name: Name for the generated artifact
            context: Dictionary of context variables to fill into the prompt template
            custom_prompt: Optional custom prompt (overrides template)

        Returns:
            GeneratedArtifact with the generated content
        """
        start_time = datetime.now()

        # Get prompt template
        if custom_prompt:
            prompt = custom_prompt
        else:
            template_key = artifact_type.value
            if template_key not in ARTIFACT_PROMPTS:
                # Use schematic_spec as fallback for spec types
                if artifact_type.value.endswith("_spec"):
                    template_key = "schematic_spec"
                else:
                    raise ValueError(f"No prompt template for artifact type: {artifact_type}")
            prompt = ARTIFACT_PROMPTS[template_key]

        # Fill in template variables
        try:
            filled_prompt = prompt.format(**context)
        except KeyError as e:
            logger.warning(f"Missing context variable: {e}. Using partial fill.")
            # Partial fill - replace what we can
            for key, value in context.items():
                filled_prompt = prompt.replace(f"{{{key}}}", str(value))

        logger.debug(f"Generating artifact: {artifact_type.value} / {name}")

        # Call LLM
        content, tokens_used = await self._call_llm(filled_prompt)

        # Determine content format
        content_format = self._detect_content_format(content, artifact_type)

        generation_time_ms = int((datetime.now() - start_time).total_seconds() * 1000)

        return GeneratedArtifact(
            artifact_type=artifact_type,
            category=category,
            name=name,
            content=content,
            content_format=content_format,
            generation_prompt=filled_prompt,
            generation_model=self.config.model,
            metadata={
                "context_keys": list(context.keys()),
                "template_used": artifact_type.value,
            },
            tokens_used=tokens_used,
            generation_time_ms=generation_time_ms,
        )

    async def _call_llm(self, prompt: str) -> tuple[str, int]:
        """
        Call the LLM API to generate content.

        Args:
            prompt: The prompt to send to the LLM

        Returns:
            Tuple of (generated_content, tokens_used)
        """
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://adverant.ai",
            "X-Title": "Nexus EE Design Partner",
        }

        messages = [
            {
                "role": "system",
                "content": "You are an expert hardware design engineer creating detailed technical documentation for electronic design projects. Generate comprehensive, accurate, and professionally formatted content suitable for engineering teams."
            },
            {
                "role": "user",
                "content": prompt
            }
        ]

        payload = {
            "model": self.config.model,
            "messages": messages,
            "max_tokens": self.config.max_tokens,
            "temperature": self.config.temperature,
        }

        async with httpx.AsyncClient(timeout=self.config.timeout_seconds) as client:
            try:
                response = await client.post(
                    f"{self.base_url}/chat/completions",
                    headers=headers,
                    json=payload,
                )
                response.raise_for_status()

                data = response.json()
                content = data["choices"][0]["message"]["content"]
                tokens = data.get("usage", {}).get("total_tokens", 0)

                logger.debug(f"LLM response received: {tokens} tokens")
                return content, tokens

            except httpx.HTTPStatusError as e:
                logger.error(f"LLM API error: {e.response.status_code} - {e.response.text}")
                raise RuntimeError(f"LLM API error: {e.response.status_code}")
            except Exception as e:
                logger.error(f"LLM call failed: {e}")
                raise

    def _detect_content_format(self, content: str, artifact_type: ArtifactType) -> str:
        """Detect the format of generated content."""
        if artifact_type == ArtifactType.ARCHITECTURE_DIAGRAM:
            if "```mermaid" in content or content.strip().startswith("graph ") or content.strip().startswith("flowchart "):
                return "mermaid"

        if artifact_type == ArtifactType.BOM:
            if content.count("|") > 10:  # Likely a markdown table
                return "markdown"
            if "," in content and "\n" in content:
                return "csv"

        if content.strip().startswith("{") and content.strip().endswith("}"):
            try:
                json.loads(content)
                return "json"
            except json.JSONDecodeError:
                pass

        if "#" in content or "**" in content or "|" in content:
            return "markdown"

        return "text"

    async def generate_batch(
        self,
        artifacts: List[Dict[str, Any]],
    ) -> List[GeneratedArtifact]:
        """
        Generate multiple artifacts in sequence.

        Args:
            artifacts: List of artifact specifications, each containing:
                - artifact_type: ArtifactType
                - category: ArtifactCategory
                - name: str
                - context: Dict[str, Any]
                - custom_prompt: Optional[str]

        Returns:
            List of GeneratedArtifact objects
        """
        results = []
        for spec in artifacts:
            try:
                artifact = await self.generate(
                    artifact_type=spec["artifact_type"],
                    category=spec["category"],
                    name=spec["name"],
                    context=spec.get("context", {}),
                    custom_prompt=spec.get("custom_prompt"),
                )
                results.append(artifact)
            except Exception as e:
                logger.error(f"Failed to generate artifact {spec.get('name')}: {e}")
                # Continue with other artifacts

        return results

    def get_prompt_template(self, artifact_type: ArtifactType) -> Optional[str]:
        """Get the prompt template for an artifact type."""
        return ARTIFACT_PROMPTS.get(artifact_type.value)

    def list_available_templates(self) -> List[str]:
        """List all available prompt template names."""
        return list(ARTIFACT_PROMPTS.keys())
