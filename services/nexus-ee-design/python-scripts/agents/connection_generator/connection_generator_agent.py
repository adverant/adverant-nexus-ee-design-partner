"""
Connection Generator Agent v3.2 - Logical Connection Generation

MAPO v3.2 - Generates LOGICAL circuit connections (pin-to-pin mappings):
- Component reference and pin identification (from_ref/from_pin -> to_ref/to_pin)
- Net naming (VCC, GND, SPI_MOSI, etc.)
- Signal type classification (power, signal, digital, analog, clock, high_speed, differential)
- Current and voltage specifications for each net
- Differential pair validation
- Functional connectivity verification

ARCHITECTURE SEPARATION:
- Connection Generator → Logical connections (THIS MODULE)
- Wire Router → Physical routing (coordinates, waypoints, IPC-2221 compliance)

This version includes:
1. LLM prompts for logical pin-to-pin connections (NOT physical routing)
2. Logical connection validation (component/pin existence, differential pairs)
3. Retry logic (up to 5 attempts) if validation fails
4. Structured JSON output enforcement

Author: Nexus EE Design Team
Version: 3.2 (MAPO v3.2 - Logical Connection Generation)
"""

import asyncio
import json
import logging
import pathlib
import re
import os
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple

import httpx
import yaml

# Note: WireValidator is NOT imported - physical wire routing validation
# is done by the separate WireRouter agent. This module only handles
# logical connection generation.

# Import ideation context types for seed connections
try:
    from ideation_context import ConnectionInferenceContext, PinConnection, InterfaceDefinition, PowerRail
except ImportError:
    ConnectionInferenceContext = None
    PinConnection = None
    InterfaceDefinition = None
    PowerRail = None

logger = logging.getLogger(__name__)

# LLM provider configuration
# Supports both Claude Code Max proxy pod (default) and OpenRouter
AI_PROVIDER = os.environ.get("AI_PROVIDER", "claude_code_max")
CLAUDE_CODE_PROXY_URL = os.environ.get(
    "CLAUDE_CODE_PROXY_URL",
    "http://claude-code-proxy.nexus.svc.cluster.local:3100"
)

if AI_PROVIDER == "claude_code_max":
    LLM_BASE_URL = f"{CLAUDE_CODE_PROXY_URL}/v1"
else:
    LLM_BASE_URL = "https://openrouter.ai/api/v1"

LLM_MODEL = "anthropic/claude-opus-4.6"

# Validation configuration
MAX_WIRE_GENERATION_RETRIES = 3  # Retry on proxy CLI failures (exit code 1 → empty response)
TARGET_WIRE_CROSSINGS = 10


class ConnectionType(Enum):
    """Type of electrical connection."""
    POWER = "power"        # VCC, GND power connections
    SIGNAL = "signal"      # General signal connections
    ANALOG = "analog"      # Analog signals (ADC, DAC, etc.)
    DIGITAL = "digital"    # Digital signals (GPIO, SPI, I2C, etc.)
    HIGH_SPEED = "high_speed"  # High-speed differential (USB, Ethernet)
    DIFFERENTIAL = "differential"  # Differential pairs
    CLOCK = "clock"        # Clock signals


@dataclass
class GeneratedConnection:
    """A generated connection between two component pins."""
    from_ref: str          # Component reference (e.g., "U1")
    from_pin: str          # Pin name or number
    to_ref: str            # Target component reference
    to_pin: str            # Target pin
    net_name: str          # Net name (e.g., "VCC", "SPI_MOSI")
    connection_type: ConnectionType = ConnectionType.SIGNAL
    priority: int = 0      # Higher = more important
    notes: str = ""        # Reasoning/notes


@dataclass
class ComponentInfo:
    """Extracted component information for connection inference."""
    reference: str
    part_number: str
    category: str
    value: str
    pins: List[Dict[str, Any]] = field(default_factory=list)
    power_pins: List[str] = field(default_factory=list)
    ground_pins: List[str] = field(default_factory=list)
    signal_pins: List[str] = field(default_factory=list)


@dataclass
class WireGenerationContext:
    """Context for wire generation with IPC-2221 parameters."""
    voltage_map: Dict[str, float]      # Net name -> voltage (V)
    current_map: Dict[str, float]      # Net name -> current (A)
    signal_types: Dict[str, str]       # Net name -> signal type
    symbol_positions: Dict[str, Tuple[float, float]]  # Ref -> (x, y)
    existing_wires: List[Dict]         # Previously generated wires


class ValidationError(Exception):
    """Raised when wire validation fails after all retries."""
    pass


class ConnectionGeneratorAgent:
    """
    Generates LOGICAL circuit connections (pin-to-pin mappings) with validation.

    v3.2 Features:
    - Logical connection generation (from_ref/from_pin -> to_ref/to_pin)
    - Net naming and signal type classification
    - Current and voltage specification for IPC-2221 compliance downstream
    - Logical validation (component/pin existence, differential pairs)
    - Retry logic with feedback to LLM

    ARCHITECTURE NOTE:
    This agent generates LOGICAL connections only. Physical wire routing
    (coordinates, waypoints, bend angles, spacing) is handled by the
    separate WireRouter agent.
    """

    # Standard power pin patterns
    POWER_PIN_PATTERNS = [
        r"^V[CDS]{2}$",         # VCC, VDD, VSS
        r"^AVCC$",              # Analog VCC
        r"^DVCC$",              # Digital VCC
        r"^V[+-]?\d*$",         # V+, V-, V12, etc.
        r"^VIN$",               # Input voltage
        r"^VOUT$",              # Output voltage
        r"^VB[AUT]*$",          # VBAT, VBA
        r"^PWR$",               # Power
        r"^\+\d+V$",            # +5V, +3.3V, etc.
    ]

    GROUND_PIN_PATTERNS = [
        r"^GND$",               # Ground
        r"^AGND$",              # Analog ground
        r"^DGND$",              # Digital ground
        r"^PGND$",              # Power ground
        r"^VSS$",               # VSS (often ground)
        r"^EP$",                # Exposed pad (usually ground)
        r"^PAD$",               # Thermal pad
        r"^0V$",                # 0V reference
    ]

    def __init__(self, api_key: Optional[str] = None):
        """Initialize the connection generator for logical connection generation."""
        self._ai_provider = AI_PROVIDER
        self._llm_base_url = LLM_BASE_URL

        # API key: not needed for claude_code_max proxy, required for openrouter
        if self._ai_provider == "claude_code_max":
            self._api_key = "internal-proxy"
        else:
            self._api_key = api_key or os.environ.get("OPENROUTER_API_KEY", "")

        self._http_client: Optional[httpx.AsyncClient] = None

        # Load IPC-2221 rules for current/voltage specifications
        rules_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "ipc_2221_rules.yaml"
        )
        with open(rules_path) as f:
            self.ipc_rules = yaml.safe_load(f)

        if self._ai_provider == "claude_code_max":
            logger.info(f"ConnectionGeneratorAgent v3.2: Using Claude Code Max proxy at {self._llm_base_url}")
        elif self._api_key:
            logger.info("ConnectionGeneratorAgent v3.2: Logical connection generation mode (OpenRouter)")
        else:
            logger.error(
                "CRITICAL: No LLM provider configured! "
                "Set OPENROUTER_API_KEY or AI_PROVIDER=claude_code_max"
            )

    async def generate_connections(
        self,
        bom: List[Dict[str, Any]],
        design_intent: str,
        component_pins: Optional[Dict[str, List[Dict]]] = None,
        seed_connections: Optional[Any] = None,
    ) -> List[GeneratedConnection]:
        """
        Generate logical connections (pin-to-pin mappings) for a circuit.

        Args:
            bom: List of BOM items with part_number, category, reference, etc.
            design_intent: Natural language description of circuit function
            component_pins: Optional dict of reference -> list of pins
            seed_connections: Optional ConnectionInferenceContext from ideation

        Returns:
            List of GeneratedConnection objects (logical connections with
            IPC-2221 metadata for downstream wire routing)
        """
        connections: List[GeneratedConnection] = []

        # Step 1: Extract component information
        components = self._extract_component_info(bom, component_pins)
        logger.info(f"Extracted info for {len(components)} components")

        # Step 1.5: Convert ideation seed connections to GeneratedConnection objects
        seed_generated: List[GeneratedConnection] = []
        if seed_connections is not None:
            seed_generated = self._convert_seed_connections(seed_connections, components, component_pins)
            logger.info(f"Converted {len(seed_generated)} seed connections from ideation data")

        # Step 2: Build connection generation context (voltage, current, signal types)
        context = self._build_wire_context(components, design_intent)
        logger.info(f"Built connection context: {len(context.voltage_map)} nets with voltage/current info")

        # Step 3: Generate power connections (rule-based)
        power_connections = self._generate_power_connections(components)
        connections.extend(power_connections)
        logger.info(f"Generated {len(power_connections)} power connections")

        # Step 4: Generate bypass capacitor connections
        bypass_connections = self._generate_bypass_cap_connections(components)
        connections.extend(bypass_connections)
        logger.info(f"Generated {len(bypass_connections)} bypass cap connections")

        # Step 5: Generate signal connections with logical validation (LLM)
        signal_connections: List[GeneratedConnection] = []

        try:
            signal_connections = await self._generate_signal_connections_with_validation(
                components,
                design_intent,
                context,
                seed_connections=seed_connections,
            )
            logger.info(f"Generated {len(signal_connections)} logically validated signal connections")
        except Exception as e:
            error_msg = (
                f"CRITICAL: LLM connection generation FAILED with exception. "
                f"Error: {str(e)}. "
                f"Components: {[c.reference for c in components][:15]}... "
                f"NO FALLBACK - Schematic generation cannot proceed."
            )
            logger.error(error_msg)
            raise RuntimeError(error_msg)

        # Validation: Check that signal connections were generated
        if not signal_connections and not seed_generated:
            error_msg = (
                f"CRITICAL: LLM returned ZERO signal connections AND no seed connections "
                f"for {len(components)} components. This is FATAL - schematic would have no signal routing."
            )
            logger.error(error_msg)
            raise RuntimeError(error_msg)

        connections.extend(signal_connections)

        # Step 5.5: Merge seed connections (higher priority, deduplicated)
        if seed_generated:
            connections = self._merge_seed_with_generated(seed_generated, connections)
            logger.info(f"After merging seed connections: {len(connections)} total")

        # Step 6: Deduplicate and prioritize
        connections = self._deduplicate_connections(connections)

        # Step 7: Power completeness check — ensure every IC has VCC + GND
        ic_categories = {"mcu", "ic", "gate_driver", "can_transceiver", "regulator",
                         "amplifier", "current_sense", "opamp"}
        ic_components = [c for c in components if c.category.lower() in ic_categories]

        # Build lookup: ref -> set of connected net_names
        ic_power_nets: Dict[str, Set[str]] = {c.reference: set() for c in ic_components}
        for conn in connections:
            if conn.connection_type == ConnectionType.POWER:
                if conn.from_ref in ic_power_nets:
                    ic_power_nets[conn.from_ref].add(conn.net_name)
                if conn.to_ref in ic_power_nets:
                    ic_power_nets[conn.to_ref].add(conn.net_name)

        missing_power_count = 0
        added_power = []
        for ic in ic_components:
            nets = ic_power_nets.get(ic.reference, set())
            has_vcc = any(n.upper() in {"VCC", "VDD", "3V3", "5V", "VDDA"} for n in nets)
            has_gnd = any(n.upper() in {"GND", "VSS", "VSSA", "AGND"} for n in nets)

            if not has_vcc and ic.power_pins:
                logger.error(
                    f"POWER CHECK: IC {ic.reference} ({ic.part_number}) has NO VCC connection. "
                    f"Auto-adding VCC on pin {ic.power_pins[0]}."
                )
                added_power.append(GeneratedConnection(
                    from_ref=ic.reference,
                    from_pin=ic.power_pins[0],
                    to_ref="PWR",
                    to_pin="VCC",
                    net_name="VCC",
                    connection_type=ConnectionType.POWER,
                    priority=10,
                    notes="Auto-added: IC was missing VCC connection"
                ))
                missing_power_count += 1

            if not has_gnd and ic.ground_pins:
                logger.error(
                    f"POWER CHECK: IC {ic.reference} ({ic.part_number}) has NO GND connection. "
                    f"Auto-adding GND on pin {ic.ground_pins[0]}."
                )
                added_power.append(GeneratedConnection(
                    from_ref=ic.reference,
                    from_pin=ic.ground_pins[0],
                    to_ref="PWR",
                    to_pin="GND",
                    net_name="GND",
                    connection_type=ConnectionType.POWER,
                    priority=10,
                    notes="Auto-added: IC was missing GND connection"
                ))
                missing_power_count += 1

        if added_power:
            connections.extend(added_power)
            connections = self._deduplicate_connections(connections)
            logger.error(
                f"POWER COMPLETENESS: {missing_power_count} missing power connections auto-added "
                f"for {len([ic for ic in ic_components if not any(n.upper() in {'VCC','VDD','3V3','5V','VDDA'} for n in ic_power_nets.get(ic.reference, set())) or not any(n.upper() in {'GND','VSS','VSSA','AGND'} for n in ic_power_nets.get(ic.reference, set()))])} ICs. "
                f"LLM or seed connections should have provided these."
            )
        else:
            logger.info(
                f"POWER COMPLETENESS: All {len(ic_components)} ICs have VCC+GND connections."
            )

        logger.info(f"Total {len(connections)} logical connections generated (with IPC-2221 metadata)")

        return connections

    def _extract_component_info(
        self,
        bom: List[Dict[str, Any]],
        component_pins: Optional[Dict[str, List[Dict]]] = None
    ) -> List[ComponentInfo]:
        """Extract component information from BOM."""
        components = []
        ref_counters: Dict[str, int] = {}

        for item in bom:
            # Get or generate reference
            reference = item.get("reference")
            if not reference:
                category = item.get("category", "Other")
                prefix = self._get_ref_prefix(category)
                ref_counters[prefix] = ref_counters.get(prefix, 0) + 1
                reference = f"{prefix}{ref_counters[prefix]}"

            comp = ComponentInfo(
                reference=reference,
                part_number=item.get("part_number", ""),
                category=item.get("category", "Other"),
                value=item.get("value", ""),
            )

            # Get pins if available
            if component_pins and reference in component_pins:
                comp.pins = component_pins[reference]

                # Classify pins — check name patterns FIRST (more specific than pin_type)
                # Bug fix: GND with pin_type="power_in" was matching the power branch
                for pin in comp.pins:
                    pin_name = pin.get("name", "").upper()
                    pin_type = pin.get("pin_type", "")

                    if self._is_ground_pin(pin_name):
                        comp.ground_pins.append(pin_name)
                    elif self._is_power_pin(pin_name):
                        comp.power_pins.append(pin_name)
                    elif pin_type == "power_in":
                        # Ambiguous pin_type — classify by name heuristic
                        if any(g in pin_name for g in ("GND", "VSS", "AGND", "DGND")):
                            comp.ground_pins.append(pin_name)
                        else:
                            comp.power_pins.append(pin_name)
                    else:
                        comp.signal_pins.append(pin_name)
            else:
                # Infer pins from category
                comp.power_pins, comp.ground_pins, comp.signal_pins = \
                    self._infer_pins_from_category(comp.category, comp.value)

            components.append(comp)

        return components

    def _build_wire_context(
        self,
        components: List[ComponentInfo],
        design_intent: str
    ) -> WireGenerationContext:
        """
        Build context for wire generation with IPC-2221 parameters.

        Infers voltage, current, and signal types for each net.
        """
        voltage_map: Dict[str, float] = {}
        current_map: Dict[str, float] = {}
        signal_types: Dict[str, str] = {}
        symbol_positions: Dict[str, Tuple[float, float]] = {}

        # Infer voltage rails from design intent and component categories
        if "3.3v" in design_intent.lower() or "3v3" in design_intent.lower():
            voltage_map["VCC"] = 3.3
            voltage_map["+3V3"] = 3.3
        elif "5v" in design_intent.lower():
            voltage_map["VCC"] = 5.0
            voltage_map["+5V"] = 5.0
        else:
            voltage_map["VCC"] = 5.0  # Default

        voltage_map["GND"] = 0.0
        voltage_map["AGND"] = 0.0
        voltage_map["DGND"] = 0.0

        # Infer current requirements
        current_map["VCC"] = 2.0  # Default 2A for power rails
        current_map["GND"] = 3.0  # Ground can carry more current

        # Infer signal types from net names (will be populated by LLM)
        signal_types["VCC"] = "power"
        signal_types["GND"] = "ground"

        # Clock signal detection from design intent
        if "clock" in design_intent.lower() or "clk" in design_intent.lower():
            signal_types["CLK"] = "clock"
            signal_types["XTAL"] = "clock"

        # High-speed signal detection
        if "spi" in design_intent.lower():
            signal_types["SPI_MOSI"] = "high_speed"
            signal_types["SPI_MISO"] = "high_speed"
            signal_types["SPI_SCK"] = "clock"

        if "can" in design_intent.lower():
            signal_types["CANH"] = "differential"
            signal_types["CANL"] = "differential"

        if "usb" in design_intent.lower():
            signal_types["USB_DP"] = "differential"
            signal_types["USB_DN"] = "differential"

        # Symbol positions (placeholder - would be populated from layout)
        for i, comp in enumerate(components):
            symbol_positions[comp.reference] = (float(i * 50), 0.0)

        return WireGenerationContext(
            voltage_map=voltage_map,
            current_map=current_map,
            signal_types=signal_types,
            symbol_positions=symbol_positions,
            existing_wires=[]
        )

    async def _generate_signal_connections_with_validation(
        self,
        components: List[ComponentInfo],
        design_intent: str,
        context: WireGenerationContext,
        seed_connections: Optional[Any] = None,
    ) -> List[GeneratedConnection]:
        """
        Generate signal connections with logical validation.

        Uses a filter-based approach: invalid connections are removed rather
        than rejecting the entire batch. Only retries if the LLM call itself
        fails (network error, rate limit, etc.).
        """
        best_connections: List[Dict] = []

        for attempt in range(1, MAX_WIRE_GENERATION_RETRIES + 1):
            logger.info(f"Connection generation attempt {attempt}/{MAX_WIRE_GENERATION_RETRIES}")

            # Build logical connection prompt
            prompt = self._build_ipc_2221_prompt(
                components,
                design_intent,
                context,
                attempt_number=attempt,
                seed_connections=seed_connections,
            )

            # Call LLM (Opus 4.6)
            try:
                connections_data = await self._call_llm_for_wires(prompt)
            except Exception as e:
                error_msg = str(e).lower()
                # Don't retry auth errors — token needs manual refresh
                # Be specific to avoid false matches (e.g. "output token" != auth token)
                is_auth_error = any(
                    kw in error_msg
                    for kw in ["auth expired", "401", "unauthorized", "token needs", "proxy auth"]
                )
                if is_auth_error:
                    logger.error(f"Auth error on attempt {attempt} — not retrying: {e}")
                    raise
                logger.error(f"LLM call failed on attempt {attempt}: {e}")
                if attempt == MAX_WIRE_GENERATION_RETRIES:
                    raise
                backoff = min(30 * (2 ** (attempt - 1)), 120)
                logger.info(f"Waiting {backoff}s before retry (exponential backoff)...")
                await asyncio.sleep(backoff)
                continue

            # Diagnostic: save raw LLM connection data to file for debugging
            try:
                diag_path = pathlib.Path("/data/output/connection_diagnostics.json")
                diag_path.parent.mkdir(parents=True, exist_ok=True)
                comp_refs = [c.reference for c in components]
                diag_data = {
                    "timestamp": datetime.now().isoformat(),
                    "attempt": attempt,
                    "raw_connection_count": len(connections_data),
                    "component_refs": comp_refs,
                    "sample_connections": connections_data[:5] if connections_data else [],
                    "sample_connection_refs": [
                        {"from": c.get("from_ref"), "to": c.get("to_ref")}
                        for c in connections_data[:10]
                    ] if connections_data else [],
                }
                diag_path.write_text(json.dumps(diag_data, indent=2, default=str))
                logger.info(f"Connection diagnostics saved to {diag_path}")
            except Exception as diag_err:
                logger.warning(f"Failed to save diagnostics: {diag_err}")

            # Remap any LLM-invented references back to actual BOM references
            connections_data = self._remap_references(connections_data, components)

            # Fill in default current_amps/voltage_volts where missing
            connections_data = self._fill_ipc_defaults(connections_data, context)

            # Filter connections: keep valid ones, discard invalid ones
            valid_connections, invalid_count = self._filter_valid_connections(
                connections_data, components, context
            )

            total = len(connections_data)
            valid = len(valid_connections)
            logger.info(
                f"Connection validation: {valid}/{total} connections valid "
                f"({invalid_count} filtered out)"
            )

            # Accept if we have valid connections
            if valid > 0:
                if valid > len(best_connections):
                    best_connections = valid_connections

                pass_rate = valid / total if total > 0 else 0
                if pass_rate < 0.5:
                    logger.error(
                        f"LOW PASS RATE: Only {valid}/{total} ({pass_rate:.0%}) connections passed validation. "
                        f"{invalid_count} rejected. This indicates pin name mismatches between "
                        f"LLM output and actual symbol pins."
                    )
                else:
                    logger.info(
                        f"Accepting {valid} valid connections "
                        f"({pass_rate:.0%} pass rate, {invalid_count} filtered)"
                    )
                return self._convert_wires_to_connections(valid_connections)

            pct = f"{valid/total:.0%}" if total > 0 else "0%"
            if attempt < MAX_WIRE_GENERATION_RETRIES:
                logger.warning(
                    f"Low validation pass rate: {valid}/{total} ({pct}). Retrying..."
                )
            else:
                logger.warning(
                    f"Low validation pass rate: {valid}/{total} ({pct}). "
                    f"Using best result with {len(best_connections)} connections."
                )

        # Use whatever valid connections we collected
        if best_connections:
            logger.warning(
                f"Using best attempt with {len(best_connections)} valid connections "
                f"(some validation errors were filtered out)"
            )
            return self._convert_wires_to_connections(best_connections)

        # Truly no valid connections from any attempt
        raise ValidationError(
            f"Failed to generate any valid connections after "
            f"{MAX_WIRE_GENERATION_RETRIES} attempts."
        )

    def _build_ipc_2221_prompt(
        self,
        components: List[ComponentInfo],
        design_intent: str,
        context: WireGenerationContext,
        attempt_number: int = 1,
        seed_connections: Optional[Any] = None,
    ) -> str:
        """
        Build LLM prompt with strict IPC-2221 rules embedded.

        This is the core prompt that enforces standards compliance.
        """
        # Get minimum spacing for this design
        max_voltage = max(context.voltage_map.values()) if context.voltage_map else 5.0
        min_spacing = self._get_min_spacing(max_voltage)

        # Build component summary
        comp_summary = []
        for comp in components:
            comp_summary.append({
                "reference": comp.reference,
                "part_number": comp.part_number,
                "category": comp.category,
                "value": comp.value,
                "signal_pins": comp.signal_pins[:15],  # Limit for prompt size
            })

        # Build symbol positions
        positions_str = json.dumps(context.symbol_positions, indent=2)

        # Build voltage/current info
        voltage_str = json.dumps(context.voltage_map, indent=2)
        current_str = json.dumps(context.current_map, indent=2)

        # Build explicit reference table for LLM clarity
        ref_table = "\n".join(
            f"  {comp.reference} = {comp.part_number} ({comp.category}, {comp.value})"
            for comp in components
        )
        valid_refs = [comp.reference for comp in components]

        # Build seed connections section for prompt (ground truth from ideation)
        seed_section = ""
        if seed_connections is not None:
            seed_lines = []
            for pc in getattr(seed_connections, "explicit_connections", []):
                seed_lines.append(
                    f"  {pc.from_component}.{pc.from_pin} → {pc.to_component}.{pc.to_pin} "
                    f"({pc.signal_name}, {pc.signal_type})"
                )
            for iface in getattr(seed_connections, "interfaces", []):
                slaves = ", ".join(iface.slave_components)
                pins = ", ".join(f"{k}={v}" for k, v in iface.pin_mappings.items())
                seed_lines.append(
                    f"  {iface.interface_type}: {iface.master_component} → [{slaves}] "
                    f"pins=[{pins}] speed={iface.speed}"
                )
            for rail in getattr(seed_connections, "power_rails", []):
                consumers = ", ".join(rail.consumer_components)
                seed_lines.append(
                    f"  POWER: {rail.net_name} ({rail.voltage}V/{rail.current_max}A) "
                    f"source={rail.source_component} → [{consumers}]"
                )
            if seed_lines:
                num_components = len(components)
                # Provide explicit connection count guidance to prevent LLM self-limiting
                min_conns = max(num_components, 40)
                typical_conns = int(num_components * 1.5)
                seed_section = (
                    "\n───────────────────────────────────────────────────────────────────\n"
                    "REFERENCE CONNECTIONS (extracted from design specification)\n"
                    "───────────────────────────────────────────────────────────────────\n"
                    "The following connections were extracted from the design documents.\n"
                    "Use these as a STARTING POINT — include them in your output, and then\n"
                    "generate ALL ADDITIONAL connections needed for a complete, functional circuit.\n"
                    f"These {len(seed_lines)} reference connections are only a partial list.\n"
                    f"A complete schematic with {num_components} components typically requires "
                    f"{min_conns}-{typical_conns} total signal connections (excluding power/ground).\n"
                    "Generate every connection needed — do NOT stop at just the reference set.\n\n"
                    + "\n".join(seed_lines) + "\n"
                )

        # Truncate design_intent to avoid bloating the prompt (was 242KB in production!)
        # Connection generation only needs component list + brief design overview,
        # not full ideation artifact text.
        max_intent_chars = 3000
        truncated_intent = design_intent[:max_intent_chars]
        if len(design_intent) > max_intent_chars:
            truncated_intent += f"\n... [truncated from {len(design_intent)} chars to {max_intent_chars}]"

        prompt = f"""You are an expert electronics engineer specializing in circuit connectivity.

Generate LOGICAL CONNECTIONS for a KiCad schematic based on the design intent and components.

IMPORTANT: You are generating LOGICAL connections (which pins connect to which pins),
NOT physical wire routing. Physical routing (coordinates, waypoints, angles) will be
handled by a separate wire router agent.

DESIGN INTENT:
{truncated_intent}

═══════════════════════════════════════════════════════════════════
CRITICAL: VALID COMPONENT REFERENCES — USE THESE EXACT REFERENCES
═══════════════════════════════════════════════════════════════════
{ref_table}

Valid references: {json.dumps(valid_refs)}

You MUST use ONLY these exact reference designators (e.g., "U1", "C4", "R7").
DO NOT invent references like "U_STM32", "C_BYPASS1", "R_PULL". Use the EXACT
references from the table above. Any connection using a reference NOT in this
list will be rejected as invalid.

COMPONENTS (full detail):
{json.dumps(comp_summary, indent=2)}

VOLTAGE MAP (V):
{voltage_str}
{seed_section}

CURRENT MAP (A):
{current_str}

═══════════════════════════════════════════════════════════════════
YOUR TASK: GENERATE LOGICAL PIN-TO-PIN CONNECTIONS
═══════════════════════════════════════════════════════════════════

For each connection, specify:
1. Source component reference and pin name (from_ref, from_pin)
2. Destination component reference and pin name (to_ref, to_pin)
3. Net name (e.g., "VCC", "SPI_MOSI", "GND")
4. Signal type (power, signal, digital, analog, clock, high_speed, differential)
5. Current requirement for the net (amps)
6. Voltage for the net (volts)

CONNECTION RULES:
- Signal connections must make functional sense (e.g., MCU SPI MOSI → peripheral SPI MOSI)
- Power connections: All VCC/VDD pins connect to VCC net, all GND/VSS pins to GND net
- Differential pairs must be paired (e.g., USB_DP with USB_DN, CANH with CANL)
- High-speed signals (SPI, I2C, CAN, USB) should be marked as such
- Clock signals should be identified (CLK, XTAL, etc.)

═══════════════════════════════════════════════════════════════════
OUTPUT FORMAT (JSON ONLY - NO OTHER TEXT):
═══════════════════════════════════════════════════════════════════

{{
  "connections": [
    {{
      "from_ref": "U1",
      "from_pin": "VCC",
      "to_ref": "C1",
      "to_pin": "1",
      "net_name": "VCC",
      "signal_type": "power",
      "current_amps": 2.0,
      "voltage_volts": 3.3
    }},
    {{
      "from_ref": "U1",
      "from_pin": "PA5",
      "to_ref": "U2",
      "to_pin": "MOSI",
      "net_name": "SPI_MOSI",
      "signal_type": "high_speed",
      "current_amps": 0.1,
      "voltage_volts": 3.3
    }},
    {{
      "from_ref": "U3",
      "from_pin": "CANH",
      "to_ref": "J1",
      "to_pin": "1",
      "net_name": "CAN_H",
      "signal_type": "differential",
      "current_amps": 0.05,
      "voltage_volts": 3.3
    }}
  ],
  "validation_notes": "All signal connections verified for functional correctness. Differential pairs matched."
}}

VALIDATION CHECKLIST (Your output MUST pass all checks):
□ All connections have valid component references (from_ref, to_ref exist in component list)
□ All connections have valid pin names for those components
□ Power nets (VCC, GND) have appropriate current ratings
□ Differential pairs are complete (both _P and _N or both H and L)
□ High-speed signals are properly identified
□ Net names are descriptive and follow conventions

COMPLETENESS REQUIREMENT:
You are generating connections for {len(components)} components. A complete, functional circuit
of this size typically needs {max(len(components), 40)}-{int(len(components) * 1.5)} signal connections
(not counting power/ground, which are handled separately). Generate ALL connections needed:
- Every IC communication bus (SPI, I2C, UART, CAN, USB) with all signal lines
- Every analog signal path (ADC inputs, DAC outputs, sensor signals)
- Every feedback/control loop (enable pins, chip selects, reset lines)
- Pull-up/pull-down resistors on open-drain/open-collector signals
- Crystal/oscillator connections
- LED indicator connections
- Connector pinouts (every functional pin on every connector)
Do NOT stop at a minimal set. Generate the COMPLETE netlist.

Attempt: {attempt_number}/{MAX_WIRE_GENERATION_RETRIES}

CRITICAL: Generate ONLY a single JSON object. Do NOT split into parts.
Do NOT wrap in markdown code fences. Do NOT add explanations.
Output EXACTLY one JSON object: {{"connections": [...]}}"""

        return prompt

    async def _call_llm_for_wires(self, prompt: str) -> List[Dict]:
        """
        Call LLM (Opus 4.6) to generate logical connections.

        Routes through Claude Code Max proxy or OpenRouter based on AI_PROVIDER.
        Returns raw connection data (not yet converted to GeneratedConnection).
        """
        # Validate provider config
        if self._ai_provider != "claude_code_max" and not self._api_key:
            raise ValueError(
                "CRITICAL: Cannot generate wires - no LLM provider configured!\n"
                "Set OPENROUTER_API_KEY or AI_PROVIDER=claude_code_max"
            )

        # Initialize HTTP client if needed
        if self._http_client is None:
            self._http_client = httpx.AsyncClient(timeout=httpx.Timeout(7200.0))

        # Pre-flight: check proxy auth before starting a 30-80min LLM call
        if self._ai_provider == "claude_code_max":
            try:
                auth_resp = await self._http_client.get(
                    f"{CLAUDE_CODE_PROXY_URL}/auth/status",
                    timeout=10.0
                )
                if auth_resp.status_code == 200:
                    auth_data = auth_resp.json()
                    if not auth_data.get("authenticated") or auth_data.get("expired"):
                        raise ValueError(
                            "Claude Code proxy auth expired! Token needs manual refresh. "
                            f"Auth status: {auth_data}"
                        )
                    # Warn if token expires within 2 hours (connection gen takes 30-80 min)
                    expires_at = auth_data.get("expiresAt", "")
                    if expires_at:
                        from datetime import datetime, timezone
                        try:
                            exp_dt = datetime.fromisoformat(expires_at.replace("Z", "+00:00"))
                            remaining = (exp_dt - datetime.now(timezone.utc)).total_seconds()
                            if remaining < 7200:
                                logger.warning(
                                    f"Proxy token expires in {remaining/60:.0f}min — "
                                    f"connection gen may fail mid-request"
                                )
                        except (ValueError, TypeError):
                            pass
            except httpx.ConnectError:
                raise ValueError(
                    f"Cannot reach Claude Code proxy at {CLAUDE_CODE_PROXY_URL}. "
                    "Ensure proxy pod is running."
                )

        # Build request payload (OpenAI-compatible format works with both providers)
        # Use streaming to avoid Claude CLI proxy hanging on large non-streaming requests
        use_streaming = self._ai_provider == "claude_code_max"
        request_payload = {
            "model": LLM_MODEL,
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "max_tokens": 16384,  # Large enough for complex circuits
            "temperature": 0.1,   # Low temperature for structured output
            "stream": use_streaming,
        }

        # Build headers based on provider
        headers = {"Content-Type": "application/json"}
        if self._ai_provider == "claude_code_max":
            # No auth or OpenRouter headers needed for internal proxy
            pass
        else:
            headers["Authorization"] = f"Bearer {self._api_key}"
            headers["HTTP-Referer"] = "https://adverant.ai"
            headers["X-Title"] = "Nexus EE Design Connection Generator v3.2"

        provider_label = "Claude Code Max proxy" if self._ai_provider == "claude_code_max" else "OpenRouter"
        logger.info(f"Calling {provider_label} ({LLM_MODEL}) for logical connection generation (stream={use_streaming})...")

        if use_streaming:
            # Streaming mode for Claude Code Max proxy to avoid hung CLI processes
            response_text = ""
            async with self._http_client.stream(
                "POST",
                f"{self._llm_base_url}/chat/completions",
                json=request_payload,
                headers=headers,
            ) as stream_response:
                if stream_response.status_code != 200:
                    error_body = await stream_response.aread()
                    raise ValueError(
                        f"LLM API Error ({provider_label}): Status {stream_response.status_code}, "
                        f"Response: {error_body.decode()[:500]}"
                    )
                async for line in stream_response.aiter_lines():
                    if not line or not line.startswith("data: "):
                        continue
                    data_str = line[6:]  # strip "data: " prefix
                    if data_str.strip() == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data_str)
                        # Check for SSE error events from proxy
                        if "error" in chunk and "choices" not in chunk:
                            err_msg = chunk["error"].get("message", str(chunk["error"]))
                            err_type = chunk["error"].get("type", "unknown")
                            logger.error(
                                f"Proxy SSE error event: type={err_type}, "
                                f"message={err_msg[:500]}"
                            )
                            raise ValueError(
                                f"Proxy returned error: [{err_type}] {err_msg[:500]}"
                            )
                        delta = chunk.get("choices", [{}])[0].get("delta", {})
                        content = delta.get("content", "")
                        if content:
                            response_text += content
                    except ValueError:
                        raise  # Re-raise proxy error
                    except (json.JSONDecodeError, IndexError, KeyError):
                        continue
            logger.info(f"Received {len(response_text)} chars from {provider_label} (streaming)")
            if len(response_text) < 10:
                raise ValueError(
                    f"Empty or near-empty response from {provider_label} "
                    f"(streaming): {len(response_text)} chars. "
                    f"Proxy CLI may have crashed (exit code 1)."
                )
        else:
            # Non-streaming mode for OpenRouter
            response = await self._http_client.post(
                f"{self._llm_base_url}/chat/completions",
                json=request_payload,
                headers=headers,
            )

            # Check for HTTP errors
            if response.status_code != 200:
                raise ValueError(
                    f"LLM API Error ({provider_label}): Status {response.status_code}, "
                    f"Response: {response.text[:500]}"
                )

            # Parse response
            response_data = response.json()

            if "choices" not in response_data or len(response_data["choices"]) == 0:
                raise ValueError(f"OpenRouter returned empty response: {response_data}")

            response_text = response_data["choices"][0]["message"]["content"].strip()
            logger.info(f"Received {len(response_text)} chars from {provider_label}")

        # Parse JSON (with cleanup)
        connections_json = self._parse_llm_json(response_text)

        # Extract connections array
        if isinstance(connections_json, dict) and "connections" in connections_json:
            return connections_json["connections"]
        elif isinstance(connections_json, list):
            return connections_json
        else:
            raise ValueError(f"Unexpected connection format: {type(connections_json)}")

    def _parse_llm_json(self, text: str) -> Any:
        """
        Parse JSON from LLM response with robust error handling.

        Handles markdown code fences, truncation, conversational preambles, etc.
        """
        # Strip markdown code fences
        clean_text = text.strip()
        if clean_text.startswith("```"):
            first_newline = clean_text.find('\n')
            if first_newline > 0:
                clean_text = clean_text[first_newline + 1:]
        if clean_text.rstrip().endswith("```"):
            clean_text = clean_text.rstrip()[:-3].rstrip()

        # Try direct parse
        try:
            return json.loads(clean_text)
        except json.JSONDecodeError:
            pass

        # Try regex extraction: find outermost { ... } with "connections" key
        json_match = re.search(r'\{[^{}]*"connections"\s*:\s*\[[\s\S]*\]\s*[^{}]*\}', clean_text)
        if json_match:
            try:
                return json.loads(json_match.group())
            except json.JSONDecodeError:
                pass

        # Last resort: find ANY JSON object in response
        json_match = re.search(r'\{[\s\S]*\}', clean_text)
        if json_match:
            try:
                parsed = json.loads(json_match.group())
                logger.warning(
                    f"JSON extraction: 'connections' key not found, "
                    f"parsed generic JSON object (keys: {list(parsed.keys()) if isinstance(parsed, dict) else 'not a dict'})"
                )
                return parsed
            except json.JSONDecodeError:
                pass

        # Handle multi-part responses: LLM sometimes splits JSON into
        # "PART 1" and "PART 2" code blocks. Extract all code blocks and merge.
        code_blocks = re.findall(r'```(?:json)?\s*\n?([\s\S]*?)```', text)
        if len(code_blocks) >= 2:
            # Try to merge connection arrays from multiple parts
            all_connections = []
            for block in code_blocks:
                block = block.strip()
                # Try parsing the block as a complete JSON or as a partial array
                for candidate in [block, '{"connections":' + block + '}']:
                    try:
                        parsed = json.loads(candidate)
                        if isinstance(parsed, dict) and "connections" in parsed:
                            all_connections.extend(parsed["connections"])
                            break
                        elif isinstance(parsed, list):
                            all_connections.extend(parsed)
                            break
                    except json.JSONDecodeError:
                        continue
                else:
                    # Try extracting array portion from partial block
                    arr_match = re.search(r'\[[\s\S]*\]', block)
                    if arr_match:
                        try:
                            items = json.loads(arr_match.group())
                            if isinstance(items, list):
                                all_connections.extend(items)
                        except json.JSONDecodeError:
                            pass
            if all_connections:
                logger.info(
                    f"Merged {len(all_connections)} connections from "
                    f"{len(code_blocks)} code blocks"
                )
                return {"connections": all_connections}

        # Last resort: try to find a JSON array of connections
        array_match = re.search(r'\[[\s\S]*\]', clean_text)
        if array_match:
            try:
                parsed = json.loads(array_match.group())
                if isinstance(parsed, list) and len(parsed) > 0:
                    # Check if it looks like connection data
                    if isinstance(parsed[0], dict) and "from_ref" in parsed[0]:
                        return {"connections": parsed}
            except json.JSONDecodeError:
                pass

        raise ValueError(f"Failed to parse LLM JSON. First 300 chars: {clean_text[:300]}")

    def _convert_wires_to_connections(self, connections_data: List[Dict]) -> List[GeneratedConnection]:
        """
        Convert validated logical connection data to GeneratedConnection format.

        Connection format from LLM includes logical pin-to-pin connections
        (from_ref, from_pin, to_ref, to_pin, net_name, signal_type, etc).
        """
        connections = []

        for conn_data in connections_data:
            net_name = conn_data.get("net_name", "NET")
            signal_type_str = conn_data.get("signal_type", "signal").lower()

            # Map signal type string to ConnectionType enum
            conn_type = ConnectionType.SIGNAL
            if signal_type_str == "power":
                conn_type = ConnectionType.POWER
            elif signal_type_str == "digital":
                conn_type = ConnectionType.DIGITAL
            elif signal_type_str == "analog":
                conn_type = ConnectionType.ANALOG
            elif signal_type_str == "clock":
                conn_type = ConnectionType.CLOCK
            elif signal_type_str == "high_speed":
                conn_type = ConnectionType.HIGH_SPEED
            elif signal_type_str == "differential":
                conn_type = ConnectionType.DIFFERENTIAL

            # Extract logical connection from LLM output
            from_ref = conn_data.get("from_ref")
            from_pin = conn_data.get("from_pin")
            to_ref = conn_data.get("to_ref")
            to_pin = conn_data.get("to_pin")

            # Validate required fields
            if not all([from_ref, from_pin, to_ref, to_pin]):
                logger.warning(
                    f"Skipping connection with missing fields: "
                    f"from_ref={from_ref}, from_pin={from_pin}, to_ref={to_ref}, to_pin={to_pin}"
                )
                continue

            # Get current and voltage for notes
            current = conn_data.get("current_amps", 0.1)
            voltage = conn_data.get("voltage_volts", 3.3)

            connections.append(GeneratedConnection(
                from_ref=from_ref,
                from_pin=from_pin,
                to_ref=to_ref,
                to_pin=to_pin,
                net_name=net_name,
                connection_type=conn_type,
                priority=10 if conn_type == ConnectionType.POWER else 5,
                notes=f"Logical connection: {current:.1f}A @ {voltage:.1f}V, {signal_type_str}"
            ))

        return connections

    def _fill_ipc_defaults(
        self,
        connections_data: List[Dict],
        context: 'WireGenerationContext',
    ) -> List[Dict]:
        """Fill in missing current_amps/voltage_volts with sensible defaults.

        The LLM sometimes omits these IPC-2221 fields on some connections.
        Rather than failing validation, infer defaults from the net name and
        signal type so the pipeline can proceed.
        """
        filled = 0
        for conn in connections_data:
            net_name = conn.get("net_name", "")
            signal_type = conn.get("signal_type", "signal")

            if "current_amps" not in conn or conn["current_amps"] is None:
                # Infer current from context voltage map or signal type
                if signal_type in ("power",):
                    conn["current_amps"] = context.current_map.get(net_name, 2.0)
                elif signal_type in ("ground",):
                    conn["current_amps"] = context.current_map.get(net_name, 3.0)
                else:
                    conn["current_amps"] = 0.1  # Signal-level default
                filled += 1

            if "voltage_volts" not in conn or conn["voltage_volts"] is None:
                if net_name in context.voltage_map:
                    conn["voltage_volts"] = context.voltage_map[net_name]
                elif signal_type in ("power",):
                    conn["voltage_volts"] = 3.3
                elif signal_type in ("ground",):
                    conn["voltage_volts"] = 0.0
                else:
                    conn["voltage_volts"] = 3.3  # Default logic level
                filled += 1

        if filled:
            logger.info(f"Filled {filled} missing IPC-2221 fields with defaults")

        return connections_data

    def _validate_logical_connections(
        self,
        connections_data: List[Dict],
        components: List[ComponentInfo],
        context: WireGenerationContext
    ) -> List[str]:
        """
        Validate logical connections for correctness.

        Returns list of error messages (empty list = valid).
        """
        errors = []

        # Build component reference lookup
        comp_refs = {c.reference for c in components}
        comp_pins = {
            c.reference: set(p.get("name", "") for p in c.pins) |
                        set(c.power_pins) | set(c.ground_pins) | set(c.signal_pins)
            for c in components
        }

        # Validate each connection
        for i, conn in enumerate(connections_data):
            # Check required fields
            required_fields = ["from_ref", "from_pin", "to_ref", "to_pin", "net_name", "signal_type"]
            missing_fields = [f for f in required_fields if f not in conn or not conn[f]]
            if missing_fields:
                errors.append(
                    f"Connection {i}: Missing required fields: {', '.join(missing_fields)}"
                )
                continue

            from_ref = conn["from_ref"]
            from_pin = conn["from_pin"]
            to_ref = conn["to_ref"]
            to_pin = conn["to_pin"]
            net_name = conn["net_name"]

            # Validate component references exist
            if from_ref not in comp_refs:
                errors.append(
                    f"Connection {i}: from_ref '{from_ref}' not found in component list"
                )

            if to_ref not in comp_refs:
                errors.append(
                    f"Connection {i}: to_ref '{to_ref}' not found in component list"
                )

            # Validate pins exist on components (if pin data available)
            # NOTE: Downgraded to warnings — LLM often uses datasheet pin names
            # that differ from cached symbol pin names. The assembler will attempt
            # to match by name or fall back to positional assignment.
            if from_ref in comp_pins and comp_pins[from_ref]:
                if from_pin not in comp_pins[from_ref]:
                    logger.debug(
                        f"Connection {i}: from_pin '{from_pin}' not in cached pins for {from_ref} "
                        f"(available: {list(comp_pins[from_ref])[:10]})"
                    )

            if to_ref in comp_pins and comp_pins[to_ref]:
                if to_pin not in comp_pins[to_ref]:
                    logger.debug(
                        f"Connection {i}: to_pin '{to_pin}' not in cached pins for {to_ref} "
                        f"(available: {list(comp_pins[to_ref])[:10]})"
                    )

            # Validate current and voltage are present for IPC-2221 compliance
            if "current_amps" not in conn:
                errors.append(
                    f"Connection {i} ({net_name}): Missing current_amps specification"
                )

            if "voltage_volts" not in conn:
                errors.append(
                    f"Connection {i} ({net_name}): Missing voltage_volts specification"
                )

        # Check differential pairs are complete
        diff_nets = {}
        for conn in connections_data:
            signal_type = conn.get("signal_type", "")
            if signal_type == "differential":
                net_name = conn.get("net_name", "")
                # Extract base name (remove _P/_N or H/L suffix)
                if net_name.endswith("_P") or net_name.endswith("_N"):
                    base = net_name[:-2]
                elif net_name.endswith("H") or net_name.endswith("L"):
                    base = net_name[:-1]
                else:
                    base = net_name

                if base not in diff_nets:
                    diff_nets[base] = []
                diff_nets[base].append(net_name)

        for base, nets in diff_nets.items():
            if len(nets) < 2:
                logger.warning(
                    f"Differential pair '{base}' incomplete: only found {nets}. "
                    f"Expected both positive and negative signals."
                )

        return errors

    # Power pin equivalents for fuzzy matching
    _POWER_PIN_EQUIVALENTS = {
        "VCC": {"VCC", "VDD", "VCCIO", "V+", "AVCC", "AVDD", "DVCC", "DVDD"},
        "GND": {"GND", "VSS", "GROUND", "AVSS", "DVSS", "AGND", "DGND", "V-", "PGND", "EPAD", "EP"},
        "3V3": {"3V3", "3.3V", "+3V3", "+3.3V", "VCC_3V3"},
        "5V": {"5V", "+5V", "VCC_5V"},
    }

    def _fuzzy_match_pin(self, pin_name: str, available_pins: set) -> Optional[str]:
        """
        Fuzzy match a pin name against available pins on a component.

        Tries multiple strategies in order:
        1. Exact match
        2. Case-insensitive match
        3. Strip suffix (before - or _) match
        4. Power pin equivalents (VCC↔VDD, GND↔VSS)
        5. Prefix match (pin starts with or available pin starts with)
        6. Number-only match (strip alpha prefix)

        Returns the matched pin name or None if no match found.
        """
        if not available_pins:
            return None

        # 1. Exact match
        if pin_name in available_pins:
            return pin_name

        # 2. Case-insensitive
        pin_upper = pin_name.upper()
        for ap in available_pins:
            if ap.upper() == pin_upper:
                return ap

        # 3. Strip suffix (before last - or _)
        base_pin = pin_name.rsplit("-", 1)[0].rsplit("_", 1)[0]
        if base_pin != pin_name:
            for ap in available_pins:
                if ap.upper() == base_pin.upper():
                    return ap

        # 4. Power pin equivalents
        for group_name, equivalents in self._POWER_PIN_EQUIVALENTS.items():
            if pin_upper in equivalents:
                for ap in available_pins:
                    if ap.upper() in equivalents:
                        return ap

        # 5. Prefix match
        if len(pin_upper) >= 2:
            for ap in available_pins:
                ap_upper = ap.upper()
                if ap_upper.startswith(pin_upper) or pin_upper.startswith(ap_upper):
                    return ap

        # 6. Number-only match (e.g., "1" matches "Pin_1" or "P1")
        pin_digits = ''.join(c for c in pin_name if c.isdigit())
        if pin_digits:
            for ap in available_pins:
                ap_digits = ''.join(c for c in ap if c.isdigit())
                if ap_digits == pin_digits and len(pin_digits) >= 1:
                    return ap

        return None

    def _filter_valid_connections(
        self,
        connections_data: List[Dict],
        components: List[ComponentInfo],
        context: WireGenerationContext
    ) -> Tuple[List[Dict], int]:
        """
        Filter connections: keep valid ones, discard invalid ones.

        Instead of rejecting the entire batch on any error, this evaluates
        each connection independently and returns only the valid subset.
        Performs fuzzy pin name matching and remaps pin names when possible.

        Returns:
            Tuple of (valid_connections, invalid_count)
        """
        valid = []
        invalid_count = 0
        pin_remapped_count = 0
        pin_unmatched_count = 0

        # Build component reference lookup
        comp_refs = {c.reference for c in components}

        # Build pin lookup per component (from all known pin sources)
        comp_pins: Dict[str, set] = {}
        for c in components:
            all_pins = set()
            for p in c.pins:
                name = p.get("name", "")
                num = p.get("number", "")
                if name:
                    all_pins.add(name)
                if num:
                    all_pins.add(num)
            all_pins.update(c.power_pins)
            all_pins.update(c.ground_pins)
            all_pins.update(c.signal_pins)
            comp_pins[c.reference] = all_pins

        required_fields = [
            "from_ref", "from_pin", "to_ref", "to_pin", "net_name", "signal_type"
        ]

        for i, conn in enumerate(connections_data):
            # Check required fields
            missing_fields = [
                f for f in required_fields if f not in conn or not conn[f]
            ]
            if missing_fields:
                logger.debug(
                    f"Filtering connection {i}: missing fields: "
                    f"{', '.join(missing_fields)}"
                )
                invalid_count += 1
                continue

            from_ref = conn["from_ref"]
            to_ref = conn["to_ref"]

            # Both component references must exist in BOM
            if from_ref not in comp_refs:
                logger.debug(
                    f"Filtering connection {i}: from_ref '{from_ref}' "
                    f"not in component list"
                )
                invalid_count += 1
                continue

            if to_ref not in comp_refs:
                logger.debug(
                    f"Filtering connection {i}: to_ref '{to_ref}' "
                    f"not in component list"
                )
                invalid_count += 1
                continue

            # Self-connections are invalid
            if from_ref == to_ref and conn.get("from_pin") == conn.get("to_pin"):
                logger.debug(
                    f"Filtering connection {i}: self-connection on "
                    f"{from_ref} pin {conn.get('from_pin')}"
                )
                invalid_count += 1
                continue

            # Pin validation with fuzzy matching (remap if possible, REJECT if not)
            pin_valid = True
            for ref_field, pin_field in [("from_ref", "from_pin"), ("to_ref", "to_pin")]:
                ref = conn[ref_field]
                pin = conn[pin_field]
                available = comp_pins.get(ref, set())

                if available and pin not in available:
                    matched = self._fuzzy_match_pin(pin, available)
                    if matched:
                        logger.debug(
                            f"Connection {i}: Remapped {ref}.{pin} → {ref}.{matched} "
                            f"(fuzzy pin match)"
                        )
                        conn[pin_field] = matched
                        pin_remapped_count += 1
                    else:
                        logger.error(
                            f"PIN MISMATCH: Connection {i}: Pin '{pin}' NOT FOUND on {ref}. "
                            f"Available pins: {sorted(list(available))[:10]}. "
                            f"Connection REJECTED — unresolvable pin name."
                        )
                        pin_unmatched_count += 1
                        pin_valid = False

            if not pin_valid:
                invalid_count += 1
                continue

            valid.append(conn)

        if invalid_count > 0:
            logger.info(
                f"Filtered {invalid_count}/{len(connections_data)} invalid connections. "
                f"Keeping {len(valid)} valid connections."
            )
        if pin_remapped_count > 0:
            logger.info(f"Fuzzy-matched and remapped {pin_remapped_count} pin names")
        if pin_unmatched_count > 0:
            logger.error(
                f"PIN VALIDATION: {pin_unmatched_count} pin names could NOT be matched "
                f"to known symbol pins. These connections may fail during wire routing."
            )

        return valid, invalid_count

    def _remap_references(
        self,
        connections_data: List[Dict],
        components: List[ComponentInfo]
    ) -> List[Dict]:
        """
        Remap LLM-invented references back to actual BOM references.

        The LLM sometimes invents descriptive references like "U_STM32", "C_BYPASS1"
        instead of using the actual BOM references "U1", "C4". This method attempts
        to fuzzy-match and remap them.
        """
        valid_refs = {c.reference for c in components}

        # Build lookup maps for fuzzy matching
        # Map: part_number -> reference (e.g., "STM32G431CBTxZ" -> "U1")
        pn_to_ref: Dict[str, str] = {}
        # Map: category prefix + index -> reference
        cat_to_refs: Dict[str, List[str]] = {}

        for comp in components:
            pn_to_ref[comp.part_number.upper()] = comp.reference
            # Also index by partial part number (first word)
            first_word = comp.part_number.split("-")[0].split("_")[0].upper()
            if first_word not in pn_to_ref:
                pn_to_ref[first_word] = comp.reference

            # Group by prefix letter
            prefix = ''.join(c for c in comp.reference if c.isalpha())
            if prefix not in cat_to_refs:
                cat_to_refs[prefix] = []
            cat_to_refs[prefix].append(comp.reference)

        remapped_count = 0

        for conn in connections_data:
            for ref_field in ["from_ref", "to_ref"]:
                ref = conn.get(ref_field, "")
                if ref in valid_refs:
                    continue  # Already valid

                # Try fuzzy matching strategies
                matched_ref = None

                # Strategy 1: Check if ref contains a part number
                ref_upper = ref.upper()
                for pn, actual_ref in pn_to_ref.items():
                    if pn in ref_upper or ref_upper in pn:
                        matched_ref = actual_ref
                        break

                # Strategy 2: Match by prefix letter (e.g., "U_STM32" -> prefix "U")
                if not matched_ref:
                    prefix = ''.join(c for c in ref if c.isalpha() and c.isupper())
                    if not prefix:
                        prefix = ''.join(c for c in ref if c.isalpha())[:1].upper()
                    if prefix in cat_to_refs and cat_to_refs[prefix]:
                        # Use the first available ref with this prefix
                        matched_ref = cat_to_refs[prefix][0]

                if matched_ref:
                    logger.warning(
                        f"Remapped LLM reference '{ref}' -> '{matched_ref}' "
                        f"(field: {ref_field})"
                    )
                    conn[ref_field] = matched_ref
                    remapped_count += 1
                else:
                    logger.error(
                        f"UNMATCHABLE reference '{ref}' (field: {ref_field}) - "
                        f"not in BOM. Valid refs: {sorted(valid_refs)}"
                    )

        if remapped_count > 0:
            logger.info(f"Remapped {remapped_count} LLM-invented references to actual BOM references")

        return connections_data

    def _get_min_spacing(self, voltage: float) -> float:
        """Get minimum IPC-2221 spacing for voltage."""
        for rule in self.ipc_rules['conductor_spacing']['voltage_classes']:
            if voltage <= rule['max_voltage']:
                return rule['min_spacing']
        return 6.4  # > 500V

    def _generate_power_connections(
        self,
        components: List[ComponentInfo]
    ) -> List[GeneratedConnection]:
        """Generate power rail connections (rule-based, not validated)."""
        connections = []

        # Connect all power pins to VCC rail
        for comp in components:
            for power_pin in comp.power_pins:
                connections.append(GeneratedConnection(
                    from_ref=comp.reference,
                    from_pin=power_pin,
                    to_ref="PWR",
                    to_pin="VCC",
                    net_name="VCC",
                    connection_type=ConnectionType.POWER,
                    priority=10,
                    notes="Auto power connection"
                ))

        # Connect all ground pins to GND rail
        for comp in components:
            for gnd_pin in comp.ground_pins:
                connections.append(GeneratedConnection(
                    from_ref=comp.reference,
                    from_pin=gnd_pin,
                    to_ref="PWR",
                    to_pin="GND",
                    net_name="GND",
                    connection_type=ConnectionType.POWER,
                    priority=10,
                    notes="Auto ground connection"
                ))

        return connections

    def _generate_bypass_cap_connections(
        self,
        components: List[ComponentInfo]
    ) -> List[GeneratedConnection]:
        """Generate bypass capacitor connections near ICs."""
        connections = []

        capacitors = [c for c in components if c.category.lower() == "capacitor"]
        ics = [c for c in components if c.category.lower() in ["mcu", "ic", "gate_driver"]]
        bypass_caps = [c for c in capacitors if self._is_bypass_cap(c.value)]

        for i, ic in enumerate(ics):
            if i < len(bypass_caps):
                cap = bypass_caps[i]

                if ic.power_pins:
                    connections.append(GeneratedConnection(
                        from_ref=cap.reference,
                        from_pin="1",
                        to_ref=ic.reference,
                        to_pin=ic.power_pins[0],
                        net_name="VCC",
                        connection_type=ConnectionType.POWER,
                        priority=8,
                        notes=f"Bypass cap for {ic.reference}"
                    ))
                else:
                    logger.error(
                        f"IC {ic.reference} ({ic.part_number}) has NO power pins detected — "
                        f"bypass cap {cap.reference} NOT connected to VCC. "
                        f"Available pins: {[p.get('name', '') for p in (ic.pins or [])[:10]]}"
                    )

                if ic.ground_pins:
                    connections.append(GeneratedConnection(
                        from_ref=cap.reference,
                        from_pin="2",
                        to_ref=ic.reference,
                        to_pin=ic.ground_pins[0],
                        net_name="GND",
                        connection_type=ConnectionType.POWER,
                        priority=8,
                        notes=f"Bypass cap for {ic.reference}"
                    ))
                else:
                    logger.error(
                        f"IC {ic.reference} ({ic.part_number}) has NO ground pins detected — "
                        f"bypass cap {cap.reference} NOT connected to GND. "
                        f"Available pins: {[p.get('name', '') for p in (ic.pins or [])[:10]]}"
                    )

        return connections

    def _deduplicate_connections(
        self,
        connections: List[GeneratedConnection]
    ) -> List[GeneratedConnection]:
        """Remove duplicate connections, keeping highest priority."""
        seen: Dict[str, GeneratedConnection] = {}

        for conn in connections:
            endpoints = sorted([
                f"{conn.from_ref}.{conn.from_pin}",
                f"{conn.to_ref}.{conn.to_pin}"
            ])
            key = "|".join(endpoints)

            if key not in seen or conn.priority > seen[key].priority:
                seen[key] = conn

        return list(seen.values())

    def _convert_seed_connections(
        self,
        seed: Any,
        components: List[ComponentInfo],
        component_pins: Optional[Dict[str, List[Dict]]] = None,
    ) -> List[GeneratedConnection]:
        """
        Convert ConnectionInferenceContext from ideation into GeneratedConnection objects.

        Processes:
        1. explicit_connections (PinConnection) → direct GeneratedConnection
        2. interfaces (InterfaceDefinition) → expanded per-signal connections
        3. power_rails (PowerRail) → power topology connections

        All seed connections get priority=15 (higher than LLM-generated priority=0-10).
        """
        result: List[GeneratedConnection] = []
        comp_refs = {c.reference for c in components}

        # --- 1. Explicit pin-to-pin connections ---
        for pc in getattr(seed, "explicit_connections", []):
            from_ref = pc.from_component
            to_ref = pc.to_component
            # Skip if component refs don't exist in BOM
            if from_ref not in comp_refs:
                logger.warning(f"SEED: Skipping explicit connection — from_ref '{from_ref}' not in BOM. Available: {sorted(comp_refs)[:10]}")
                continue
            if to_ref not in comp_refs:
                logger.warning(f"SEED: Skipping explicit connection — to_ref '{to_ref}' not in BOM. Available: {sorted(comp_refs)[:10]}")
                continue
            conn = GeneratedConnection(
                from_ref=from_ref,
                from_pin=pc.from_pin,
                to_ref=to_ref,
                to_pin=pc.to_pin,
                net_name=pc.signal_name or f"Net-({from_ref}-{pc.from_pin})",
                connection_type=self._map_signal_type(pc.signal_type),
                priority=15,
                notes=f"SEED:explicit — {pc.notes}" if pc.notes else "SEED:explicit",
            )
            result.append(conn)
            logger.debug(f"SEED explicit: {from_ref}.{pc.from_pin} → {to_ref}.{pc.to_pin} ({conn.net_name})")

        # --- 2. Interface definitions (SPI, I2C, UART, etc.) ---
        for iface in getattr(seed, "interfaces", []):
            master = iface.master_component
            if master not in comp_refs:
                logger.warning(f"SEED: Interface '{iface.interface_type}' master '{master}' not in BOM")
                continue
            for slave in iface.slave_components:
                if slave not in comp_refs:
                    logger.warning(f"SEED: Interface '{iface.interface_type}' slave '{slave}' not in BOM")
                    continue
                # Expand pin_mappings into connections
                for signal_name, pin_name in iface.pin_mappings.items():
                    # For interfaces, signal_name is e.g. "MOSI", "SCK", "SDA"
                    # pin_name is the physical pin on the master e.g. "PA7"
                    # The slave typically has a matching signal pin name
                    conn = GeneratedConnection(
                        from_ref=master,
                        from_pin=pin_name,
                        to_ref=slave,
                        to_pin=signal_name,  # Slave pin uses signal name (e.g., MOSI, SCK)
                        net_name=f"{iface.interface_type}_{signal_name}",
                        connection_type=ConnectionType.HIGH_SPEED if iface.interface_type in {"SPI", "USB"} else ConnectionType.DIGITAL,
                        priority=15,
                        notes=f"SEED:interface:{iface.interface_type} speed={iface.speed}",
                    )
                    result.append(conn)
                    logger.debug(
                        f"SEED interface {iface.interface_type}: {master}.{pin_name} → {slave}.{signal_name}"
                    )

        # --- 3. Power rails ---
        for rail in getattr(seed, "power_rails", []):
            source = rail.source_component
            if source and source not in comp_refs:
                logger.warning(f"SEED: Power rail '{rail.net_name}' source '{source}' not in BOM")
                continue
            for consumer in rail.consumer_components:
                if consumer not in comp_refs:
                    logger.warning(f"SEED: Power rail '{rail.net_name}' consumer '{consumer}' not in BOM")
                    continue
                if source:
                    conn = GeneratedConnection(
                        from_ref=source,
                        from_pin=rail.net_name,  # Use net name as pin (e.g., VOUT)
                        to_ref=consumer,
                        to_pin=rail.net_name,  # Consumer power pin
                        net_name=rail.net_name,
                        connection_type=ConnectionType.POWER,
                        priority=15,
                        notes=f"SEED:power_rail {rail.voltage}V/{rail.current_max}A ({rail.regulator_type})",
                    )
                    result.append(conn)

        # Validate seed connection pins against actual component pins
        seed_pin_errors = 0
        if component_pins:
            for conn in result:
                for attr, ref_attr in [("from_pin", "from_ref"), ("to_pin", "to_ref")]:
                    pin = getattr(conn, attr)
                    ref = getattr(conn, ref_attr)
                    if ref in component_pins:
                        available = {p.get("name", "") for p in component_pins[ref]}
                        if pin not in available:
                            matched = self._fuzzy_match_pin(pin, available)
                            if matched:
                                logger.debug(f"Seed connection: fuzzy-matched {ref}.{pin} -> {matched}")
                                setattr(conn, attr, matched)
                            else:
                                logger.error(
                                    f"SEED PIN MISMATCH: Pin '{pin}' NOT FOUND on {ref}. "
                                    f"Available pins: {sorted(list(available))[:10]}. "
                                    f"Seed connection kept but wire routing may fail."
                                )
                                seed_pin_errors += 1

        if seed_pin_errors > 0:
            logger.error(
                f"SEED VALIDATION: {seed_pin_errors} seed connection pins could not be "
                f"matched to actual symbol pins. Check ideation context pin names."
            )

        logger.info(
            f"SEED CONVERSION COMPLETE: {len(result)} connections from ideation "
            f"({len(getattr(seed, 'explicit_connections', []))} explicit, "
            f"{len(getattr(seed, 'interfaces', []))} interfaces, "
            f"{len(getattr(seed, 'power_rails', []))} power rails)"
        )
        return result

    def _merge_seed_with_generated(
        self,
        seed_connections: List[GeneratedConnection],
        generated_connections: List[GeneratedConnection],
    ) -> List[GeneratedConnection]:
        """
        Merge seed connections (priority 15) with LLM-generated connections.

        Seed connections win on conflict (same from_ref.from_pin → to_ref.to_pin).
        """
        # Build lookup of seed connections (include signal_type to allow
        # different signal types on the same endpoint pair)
        seed_keys = set()
        for conn in seed_connections:
            key = "|".join(sorted([
                f"{conn.from_ref}.{conn.from_pin}",
                f"{conn.to_ref}.{conn.to_pin}"
            ])) + f"|{conn.connection_type.value}"
            seed_keys.add(key)

        # Filter out LLM connections that conflict with seeds
        filtered = []
        conflicts = 0
        for conn in generated_connections:
            key = "|".join(sorted([
                f"{conn.from_ref}.{conn.from_pin}",
                f"{conn.to_ref}.{conn.to_pin}"
            ])) + f"|{conn.connection_type.value}"
            if key in seed_keys:
                conflicts += 1
                logger.debug(f"MERGE: LLM connection {key} overridden by seed connection")
            else:
                filtered.append(conn)

        if conflicts > 0:
            logger.info(f"MERGE: {conflicts} LLM connections overridden by seed connections")

        # Seeds first, then remaining LLM connections
        merged = seed_connections + filtered
        logger.info(f"MERGE RESULT: {len(seed_connections)} seed + {len(filtered)} LLM = {len(merged)} total")
        return merged

    def _map_signal_type(self, signal_type: str) -> ConnectionType:
        """Map ideation signal_type string to ConnectionType enum."""
        mapping = {
            "signal": ConnectionType.SIGNAL,
            "power": ConnectionType.POWER,
            "ground": ConnectionType.POWER,
            "clock": ConnectionType.CLOCK,
            "analog": ConnectionType.ANALOG,
            "digital": ConnectionType.DIGITAL,
            "high_speed": ConnectionType.HIGH_SPEED,
            "differential": ConnectionType.DIFFERENTIAL,
        }
        return mapping.get(signal_type.lower(), ConnectionType.SIGNAL)

    # Utility methods (inherited from v3.0)

    def _get_ref_prefix(self, category: str) -> str:
        """Get reference designator prefix for category.

        MUST stay synchronized with assembler_agent.REF_PREFIXES to ensure
        consistent reference numbering across resolve_symbols_only,
        _extract_component_info, and _assign_components.
        """
        prefixes = {
            "MCU": "U", "IC": "U", "MOSFET": "Q", "BJT": "Q",
            "Transistor": "Q", "Gate_Driver": "U", "OpAmp": "U",
            "Amplifier": "U", "CAN_Transceiver": "U", "Capacitor": "C",
            "Resistor": "R", "Inductor": "L", "Thermistor": "R",
            "Diode": "D", "TVS": "D", "LED": "D", "Connector": "J",
            "Power": "U", "Regulator": "U", "Crystal": "Y",
            "Fuse": "F", "Relay": "K", "Transformer": "T",
        }
        return prefixes.get(category, "U")

    def _is_power_pin(self, pin_name: str) -> bool:
        """Check if pin name matches power patterns."""
        return any(re.match(p, pin_name.upper()) for p in self.POWER_PIN_PATTERNS)

    def _is_ground_pin(self, pin_name: str) -> bool:
        """Check if pin name matches ground patterns."""
        return any(re.match(p, pin_name.upper()) for p in self.GROUND_PIN_PATTERNS)

    def _is_bypass_cap(self, value: str) -> bool:
        """Check if capacitor value suggests bypass cap."""
        value_lower = value.lower()
        bypass_values = ["100n", "0.1u", "100nf", "10n", "1u", "10uf"]
        return any(v in value_lower for v in bypass_values)

    def _infer_pins_from_category(
        self,
        category: str,
        value: str
    ) -> Tuple[List[str], List[str], List[str]]:
        """Infer typical pins from component category."""
        power_pins, ground_pins, signal_pins = [], [], []
        cat = category.lower()

        if cat in ["resistor", "capacitor", "inductor"]:
            signal_pins = ["1", "2"]
        elif cat in ["diode", "led"]:
            signal_pins = ["A", "K"]
        elif cat == "mosfet":
            signal_pins = ["G", "D", "S"]
        elif cat in ["mcu", "ic"]:
            power_pins, ground_pins = ["VCC"], ["GND"]
            signal_pins = ["PA0", "PB0"]
        elif cat == "gate_driver":
            power_pins, ground_pins = ["VCC"], ["GND"]
            signal_pins = ["INH", "INL", "HO", "LO"]
        elif cat == "can_transceiver":
            power_pins, ground_pins = ["VCC"], ["GND"]
            signal_pins = ["TXD", "RXD", "CANH", "CANL"]

        return power_pins, ground_pins, signal_pins

    def to_assembler_format(
        self,
        connections: List[GeneratedConnection]
    ) -> List[Dict[str, str]]:
        """Convert connections to format expected by SchematicAssembler."""
        return [
            {
                "from_ref": conn.from_ref,
                "from_pin": conn.from_pin,
                "to_ref": conn.to_ref,
                "to_pin": conn.to_pin,
                "net_name": conn.net_name,
            }
            for conn in connections
        ]


# CLI test
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    async def test():
        generator = ConnectionGeneratorAgent()

        bom = [
            {"part_number": "STM32G431CBT6", "category": "MCU", "reference": "U1"},
            {"part_number": "UCC21520DW", "category": "Gate_Driver", "reference": "U2"},
            {"part_number": "TJA1051T/3", "category": "CAN_Transceiver", "reference": "U3"},
            {"part_number": "100nF", "category": "Capacitor", "reference": "C1", "value": "100nF"},
        ]

        design_intent = "FOC ESC with 3-phase gate drivers, CAN communication, 3.3V logic"

        connections = await generator.generate_connections(bom, design_intent)

        print(f"\n✅ Generated {len(connections)} logical connections\n")
        for conn in connections:
            print(f"  {conn.from_ref}.{conn.from_pin} -> {conn.to_ref}.{conn.to_pin} "
                  f"({conn.net_name}) [{conn.connection_type.value}]")

    asyncio.run(test())
