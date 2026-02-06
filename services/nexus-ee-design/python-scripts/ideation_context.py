"""
Structured Ideation Context for MAPO Pipeline Integration.

Defines typed dataclasses that represent structured data extracted from
raw ideation artifacts. These dataclasses drive every stage of the MAPO
schematic generation pipeline: symbol resolution, connection inference,
component placement, and validation.

The master ``IdeationContext`` aggregates all domain-specific context
objects and exposes convenience properties so pipeline stages can
quickly determine what structured data is available.

Author: Nexus EE Design Team
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# ArtifactType enum
# ---------------------------------------------------------------------------
# Re-export the canonical ArtifactType defined in the ideation agent so that
# every module in the pipeline can import it from a single location without
# pulling in the full artifact generator and its ``httpx`` dependency.


class ArtifactType(str, Enum):
    """
    Enumeration of all ideation artifact types recognised by the pipeline.

    Each member corresponds to a distinct category of design documentation
    that can be generated during the ideation phase and later consumed by
    the MAPO pipeline stages.

    The string values match the ``artifact_type`` column stored in the
    ideation artifact repository so that look-ups are consistent across
    the Python and TypeScript layers.
    """

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


# ---------------------------------------------------------------------------
# Symbol Resolution dataclasses
# ---------------------------------------------------------------------------


@dataclass
class BOMEntry:
    """
    A single line-item from the Bill of Materials.

    Captures all information needed by the symbol resolution stage to
    locate or generate the correct KiCad symbol, select the right
    footprint, and propagate manufacturer data into the schematic.

    Attributes:
        part_number: Manufacturer or distributor part number (e.g. ``STM32G431CBT6``).
        manufacturer: Component manufacturer name (e.g. ``ST Microelectronics``).
        description: Human-readable description of the component function.
        reference_designator: Schematic reference designator prefix or full
            designator (e.g. ``U1``, ``R``).
        package: Physical package type (e.g. ``LQFP-48``, ``0603``).
        quantity: Number of instances required in the design.
        category: Component category such as ``MCU``, ``Resistor``, ``Capacitor``.
        value: Nominal value string (e.g. ``100nF``, ``10k``, ``3.3V LDO``).
        subsystem: Name of the parent subsystem this component belongs to.
        alternatives: List of alternative part numbers that can substitute
            this component.
        priority: Placement priority where lower numbers indicate higher
            priority during symbol resolution.  Defaults to ``0``.
    """

    part_number: str = ""
    manufacturer: str = ""
    description: str = ""
    reference_designator: str = ""
    package: str = ""
    quantity: int = 1
    category: str = ""
    value: str = ""
    subsystem: str = ""
    alternatives: List[str] = field(default_factory=list)
    priority: int = 0


@dataclass
class SymbolResolutionContext:
    """
    Aggregated hints for the symbol resolution pipeline stage.

    Provides the resolver with a prioritised list of preferred parts,
    manufacturer ordering, package-level preferences, and an exclusion
    list so that the correct KiCad symbols are chosen without manual
    intervention.

    Attributes:
        preferred_parts: Ordered list of ``BOMEntry`` objects extracted
            from BOM and component-selection artifacts.
        manufacturer_priority: Ordered list of manufacturer names.
            The resolver should prefer symbols from earlier manufacturers
            when multiple options exist.
        package_preferences: Mapping from component category to the
            preferred package type (e.g. ``{"Resistor": "0603"}``).
        avoid_parts: List of part numbers that should be excluded from
            symbol resolution (e.g. end-of-life or unavailable parts).
    """

    preferred_parts: List[BOMEntry] = field(default_factory=list)
    manufacturer_priority: List[str] = field(default_factory=list)
    package_preferences: Dict[str, str] = field(default_factory=dict)
    avoid_parts: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Connection Inference dataclasses
# ---------------------------------------------------------------------------


@dataclass
class PinConnection:
    """
    An explicit pin-to-pin connection extracted from a schematic specification.

    Each instance represents a single wire or net segment that the
    connection inference stage should treat as a hard constraint when
    generating the netlist.

    Attributes:
        from_component: Reference designator or name of the source component.
        from_pin: Pin name or number on the source component.
        to_component: Reference designator or name of the destination component.
        to_pin: Pin name or number on the destination component.
        signal_name: Logical net or signal name (e.g. ``SPI1_MOSI``).
        signal_type: Classification of the signal.  Common values include
            ``"signal"``, ``"power"``, ``"ground"``, ``"clock"``, ``"analog"``.
            Defaults to ``"signal"``.
        notes: Free-form notes about this connection (impedance constraints,
            routing considerations, etc.).
    """

    from_component: str = ""
    from_pin: str = ""
    to_component: str = ""
    to_pin: str = ""
    signal_name: str = ""
    signal_type: str = "signal"
    notes: str = ""


@dataclass
class InterfaceDefinition:
    """
    A structured communication interface definition.

    Represents a multi-signal interface bus (SPI, I2C, UART, etc.) with
    its master/slave topology, pin mappings, speed, and protocol notes.
    The connection inference stage uses this to generate all constituent
    connections in a single pass.

    Attributes:
        interface_type: Protocol identifier.  Expected values: ``"SPI"``,
            ``"I2C"``, ``"UART"``, ``"CAN"``, ``"USB"``, ``"PWM"``,
            ``"ADC"``, or any other standard bus name.
        master_component: Reference designator or name of the bus master.
        slave_components: List of reference designators or names of all
            slave devices on this bus.
        pin_mappings: Mapping from signal names (e.g. ``"MOSI"``) to
            physical pin identifiers.  The exact structure depends on the
            protocol; for SPI it might be
            ``{"MOSI": "PA7", "MISO": "PA6", "SCK": "PA5", "CS": "PA4"}``.
        speed: Human-readable speed or baud rate (e.g. ``"10 MHz"``,
            ``"115200 baud"``).
        protocol_notes: Additional protocol-specific notes such as
            addressing mode, clock polarity, or termination requirements.
    """

    interface_type: str = ""
    master_component: str = ""
    slave_components: List[str] = field(default_factory=list)
    pin_mappings: Dict[str, str] = field(default_factory=dict)
    speed: str = ""
    protocol_notes: str = ""


@dataclass
class PowerRail:
    """
    Definition of a single power rail in the design.

    Captures the net name, voltage, maximum current, regulator topology,
    and the source/consumer relationship.  Used by both the connection
    inference stage (to wire power nets) and the validation stage (to
    check voltage/current limits).

    Attributes:
        net_name: Net label for this rail (e.g. ``"VCC_3V3"``, ``"+5V"``).
        voltage: Nominal rail voltage in volts.
        current_max: Maximum expected current draw in amperes.
        regulator_type: Topology of the regulator producing this rail
            (e.g. ``"LDO"``, ``"Buck"``, ``"Boost"``, ``"SEPIC"``).
        source_component: Reference designator or name of the component
            that sources this rail.
        consumer_components: List of reference designators or names of
            components that draw from this rail.
    """

    net_name: str = ""
    voltage: float = 0.0
    current_max: float = 0.0
    regulator_type: str = ""
    source_component: str = ""
    consumer_components: List[str] = field(default_factory=list)


@dataclass
class ConnectionInferenceContext:
    """
    Aggregated hints for the connection inference pipeline stage.

    Combines explicit pin connections, high-level interface definitions,
    power rail definitions, ground net names, and critical signal lists
    so that the connection generator can produce a complete and correct
    netlist.

    Attributes:
        explicit_connections: List of hard-constraint pin-to-pin
            connections extracted from schematic specifications.
        interfaces: List of communication bus definitions (SPI, I2C, etc.).
        power_rails: List of power rail definitions with voltage, current,
            and regulator information.
        ground_nets: List of ground net names (e.g. ``["GND", "AGND"]``).
        critical_signals: List of signal names that require special
            routing attention (matched impedance, guard traces, etc.).
        design_intent_text: Free-form design intent narrative that
            provides additional context for the LLM connection generator.
    """

    explicit_connections: List[PinConnection] = field(default_factory=list)
    interfaces: List[InterfaceDefinition] = field(default_factory=list)
    power_rails: List[PowerRail] = field(default_factory=list)
    ground_nets: List[str] = field(default_factory=list)
    critical_signals: List[str] = field(default_factory=list)
    design_intent_text: str = ""


# ---------------------------------------------------------------------------
# Placement dataclasses
# ---------------------------------------------------------------------------


@dataclass
class SubsystemBlock:
    """
    A logical grouping of components forming a functional subsystem.

    Used by the layout / placement optimizer to cluster related
    components and enforce proximity or isolation constraints.

    Attributes:
        name: Human-readable subsystem name (e.g. ``"Power Stage"``).
        components: List of reference designators that belong to this block.
        position_hint: Optional placement hint such as ``"top-left"``,
            ``"center"``, or ``"near:U1"``.
        connections_to: List of other ``SubsystemBlock`` names that this
            block has connections to.  Used for adjacency optimization.
    """

    name: str = ""
    components: List[str] = field(default_factory=list)
    position_hint: str = ""
    connections_to: List[str] = field(default_factory=list)


@dataclass
class PlacementContext:
    """
    Aggregated hints for the component placement pipeline stage.

    Provides subsystem block definitions, global signal flow direction,
    critical proximity pairs, and isolation zones.

    Attributes:
        subsystem_blocks: List of ``SubsystemBlock`` objects describing
            logical component groupings.
        signal_flow_direction: Overall signal flow direction for the
            schematic layout.  Defaults to ``"left_to_right"``.  Other
            accepted values: ``"right_to_left"``, ``"top_to_bottom"``,
            ``"bottom_to_top"``.
        critical_proximity: List of ``(component_a, component_b)`` tuples
            where the two components must be placed in close proximity
            (e.g. decoupling caps near ICs).
        isolation_zones: List of ``(zone_name, reason)`` tuples describing
            areas that must be physically isolated (e.g. high-voltage vs.
            low-voltage regions).
    """

    subsystem_blocks: List[SubsystemBlock] = field(default_factory=list)
    signal_flow_direction: str = "left_to_right"
    critical_proximity: List[Tuple[str, str]] = field(default_factory=list)
    isolation_zones: List[Tuple[str, str]] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Validation dataclasses
# ---------------------------------------------------------------------------


@dataclass
class TestCriterion:
    """
    A single testable criterion extracted from a test plan artifact.

    Used by the smoke-test and validation stages to programmatically
    verify schematic correctness.

    Attributes:
        test_name: Short identifier for the test (e.g. ``"VCC_3V3_rail_voltage"``).
        expected_result: Human-readable description of the expected outcome.
        pass_criteria: Machine-evaluable pass condition (e.g.
            ``"voltage >= 3.2 and voltage <= 3.4"``).
        severity: Impact level if this test fails.  Defaults to ``"error"``.
            Accepted values: ``"critical"``, ``"error"``, ``"warning"``,
            ``"info"``.
        category: Grouping category (e.g. ``"power"``, ``"signal_integrity"``,
            ``"thermal"``).
    """

    test_name: str = ""
    expected_result: str = ""
    pass_criteria: str = ""
    severity: str = "error"
    category: str = ""


@dataclass
class ComplianceRequirement:
    """
    A regulatory or standards-compliance requirement.

    Extracted from compliance documentation artifacts.  Used by the
    validation stage to ensure the generated schematic meets applicable
    standards.

    Attributes:
        standard: Name of the standard (e.g. ``"IEC 61000"``,
            ``"NASA-STD-8739.4"``, ``"MIL-STD-883"``).
        requirement: Specific requirement text from the standard.
        verification_method: How this requirement should be verified
            (e.g. ``"analysis"``, ``"inspection"``, ``"test"``).
        applicable_components: List of component reference designators
            or categories that this requirement applies to.
    """

    standard: str = ""
    requirement: str = ""
    verification_method: str = ""
    applicable_components: List[str] = field(default_factory=list)


@dataclass
class ValidationContext:
    """
    Aggregated hints for the validation and smoke-test pipeline stages.

    Combines test criteria, compliance requirements, and limit
    dictionaries so that validators can apply design-specific checks
    beyond the generic rule set.

    Attributes:
        test_criteria: List of ``TestCriterion`` objects extracted from
            test-plan artifacts.
        compliance_requirements: List of ``ComplianceRequirement`` objects
            extracted from compliance documentation.
        voltage_limits: Mapping from rail/net name to ``(min, max)`` voltage
            tuples in volts (e.g. ``{"VCC_3V3": (3.13, 3.47)}``).
        current_limits: Mapping from rail/net name to maximum current in
            amperes (e.g. ``{"VCC_3V3": 0.5}``).
        thermal_limits: Mapping from component reference designator to
            maximum junction temperature in degrees Celsius
            (e.g. ``{"U1": 125.0}``).
        emc_requirements: List of free-form EMC requirement strings
            extracted from compliance or spec artifacts.
    """

    test_criteria: List[TestCriterion] = field(default_factory=list)
    compliance_requirements: List[ComplianceRequirement] = field(default_factory=list)
    voltage_limits: Dict[str, Any] = field(default_factory=dict)
    current_limits: Dict[str, Any] = field(default_factory=dict)
    thermal_limits: Dict[str, Any] = field(default_factory=dict)
    emc_requirements: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Master IdeationContext
# ---------------------------------------------------------------------------


@dataclass
class IdeationContext:
    """
    Master aggregation of all structured ideation data for the MAPO pipeline.

    An ``IdeationContext`` is built by the extraction layer
    (``ideation_extractors.py``) from raw ideation artifacts and then
    threaded through every MAPO pipeline stage.  Each stage inspects the
    relevant sub-context (``symbol_resolution``, ``connection_inference``,
    ``placement``, ``validation``) and uses the structured data to produce
    a higher-quality schematic.

    When no ideation artifacts are available, a default (empty)
    ``IdeationContext`` is created so that the pipeline can operate in
    backward-compatible mode without conditional checks.

    Attributes:
        symbol_resolution: Hints for the symbol resolution stage.
        connection_inference: Hints for the connection inference stage.
        placement: Hints for the component placement stage.
        validation: Hints for the validation / smoke-test stage.
        raw_artifacts: Dictionary mapping artifact type strings to their
            raw content.  Preserved for debugging and for the legacy
            ``design_intent_text`` property.
        artifact_count: Total number of raw artifacts that were processed
            to build this context.
        extraction_errors: List of human-readable error messages
            encountered during extraction.  If non-empty, the pipeline
            should log these prominently and may choose to abort.
    """

    symbol_resolution: SymbolResolutionContext = field(
        default_factory=SymbolResolutionContext
    )
    connection_inference: ConnectionInferenceContext = field(
        default_factory=ConnectionInferenceContext
    )
    placement: PlacementContext = field(default_factory=PlacementContext)
    validation: ValidationContext = field(default_factory=ValidationContext)
    raw_artifacts: Dict[str, Any] = field(default_factory=dict)
    artifact_count: int = 0
    extraction_errors: List[str] = field(default_factory=list)

    # -- Convenience properties ---------------------------------------------

    @property
    def has_bom_artifacts(self) -> bool:
        """Return ``True`` if BOM / component-selection data was extracted."""
        return len(self.symbol_resolution.preferred_parts) > 0

    @property
    def has_connection_hints(self) -> bool:
        """Return ``True`` if any explicit connections or interfaces were extracted."""
        ci = self.connection_inference
        return (
            len(ci.explicit_connections) > 0
            or len(ci.interfaces) > 0
            or len(ci.power_rails) > 0
        )

    @property
    def has_placement_hints(self) -> bool:
        """Return ``True`` if subsystem-block placement data was extracted."""
        return len(self.placement.subsystem_blocks) > 0

    @property
    def has_validation_criteria(self) -> bool:
        """Return ``True`` if test criteria or compliance requirements were extracted."""
        v = self.validation
        return (
            len(v.test_criteria) > 0
            or len(v.compliance_requirements) > 0
        )

    @property
    def design_intent_text(self) -> str:
        """
        Backward-compatible plain-text representation of the ideation context.

        Generates a flattened text blob that mirrors the output of the legacy
        ``create_design_intent()`` function.  Pipeline stages that have not yet
        been updated to consume structured data can use this property as a
        drop-in replacement.
        """
        parts: List[str] = []

        # BOM / component summary
        if self.has_bom_artifacts:
            parts.append("=== COMPONENT SELECTION (from ideation) ===")
            for entry in self.symbol_resolution.preferred_parts:
                line = f"- {entry.part_number}"
                if entry.manufacturer:
                    line += f" ({entry.manufacturer})"
                if entry.value:
                    line += f" [{entry.value}]"
                if entry.description:
                    line += f" - {entry.description}"
                if entry.subsystem:
                    line += f" (subsystem: {entry.subsystem})"
                parts.append(line)
            parts.append("")

        # Connection hints
        if self.has_connection_hints:
            ci = self.connection_inference
            if ci.explicit_connections:
                parts.append("=== EXPLICIT CONNECTIONS (from ideation) ===")
                for conn in ci.explicit_connections:
                    parts.append(
                        f"- {conn.from_component}.{conn.from_pin} -> "
                        f"{conn.to_component}.{conn.to_pin}"
                        f" [{conn.signal_name}] ({conn.signal_type})"
                    )
                parts.append("")

            if ci.interfaces:
                parts.append("=== INTERFACE DEFINITIONS (from ideation) ===")
                for iface in ci.interfaces:
                    slaves = ", ".join(iface.slave_components) if iface.slave_components else "N/A"
                    parts.append(
                        f"- {iface.interface_type}: "
                        f"{iface.master_component} -> [{slaves}]"
                    )
                    if iface.speed:
                        parts.append(f"  Speed: {iface.speed}")
                    if iface.pin_mappings:
                        for sig, pin in iface.pin_mappings.items():
                            parts.append(f"  {sig}: {pin}")
                parts.append("")

            if ci.power_rails:
                parts.append("=== POWER RAILS (from ideation) ===")
                for rail in ci.power_rails:
                    consumers = ", ".join(rail.consumer_components) if rail.consumer_components else "N/A"
                    parts.append(
                        f"- {rail.net_name}: {rail.voltage}V "
                        f"(max {rail.current_max}A) "
                        f"[{rail.regulator_type}] "
                        f"from {rail.source_component} -> [{consumers}]"
                    )
                parts.append("")

            if ci.design_intent_text:
                parts.append("=== DESIGN INTENT ===")
                parts.append(ci.design_intent_text)
                parts.append("")

        # Placement hints
        if self.has_placement_hints:
            parts.append("=== PLACEMENT HINTS (from ideation) ===")
            parts.append(f"Signal flow: {self.placement.signal_flow_direction}")
            for block in self.placement.subsystem_blocks:
                comps = ", ".join(block.components) if block.components else "N/A"
                parts.append(f"- {block.name}: [{comps}]")
                if block.position_hint:
                    parts.append(f"  Position: {block.position_hint}")
                if block.connections_to:
                    parts.append(f"  Connects to: {', '.join(block.connections_to)}")
            parts.append("")

        # Validation criteria
        if self.has_validation_criteria:
            v = self.validation
            if v.test_criteria:
                parts.append("=== TEST CRITERIA (from ideation) ===")
                for tc in v.test_criteria:
                    parts.append(
                        f"- [{tc.severity.upper()}] {tc.test_name}: "
                        f"{tc.expected_result} (pass: {tc.pass_criteria})"
                    )
                parts.append("")

            if v.compliance_requirements:
                parts.append("=== COMPLIANCE REQUIREMENTS (from ideation) ===")
                for cr in v.compliance_requirements:
                    components = ", ".join(cr.applicable_components) if cr.applicable_components else "all"
                    parts.append(
                        f"- [{cr.standard}] {cr.requirement} "
                        f"(verify: {cr.verification_method}, applies to: {components})"
                    )
                parts.append("")

        # Raw artifact content as fallback
        if not parts and self.raw_artifacts:
            parts.append("=== RAW IDEATION ARTIFACTS ===")
            for art_type, content in self.raw_artifacts.items():
                if isinstance(content, str) and content.strip():
                    parts.append(f"\n--- {art_type} ---")
                    parts.append(content[:3000])
                    if len(content) > 3000:
                        parts.append("... [truncated]")
            parts.append("")

        return "\n".join(parts)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = [
    # Enum
    "ArtifactType",
    # Symbol resolution
    "BOMEntry",
    "SymbolResolutionContext",
    # Connection inference
    "PinConnection",
    "InterfaceDefinition",
    "PowerRail",
    "ConnectionInferenceContext",
    # Placement
    "SubsystemBlock",
    "PlacementContext",
    # Validation
    "TestCriterion",
    "ComplianceRequirement",
    "ValidationContext",
    # Master context
    "IdeationContext",
]
