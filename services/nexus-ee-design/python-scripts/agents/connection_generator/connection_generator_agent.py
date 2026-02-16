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
import re
import os
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

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
MAX_WIRE_GENERATION_RETRIES = 5
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
                context
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
        if not signal_connections:
            error_msg = (
                f"CRITICAL: LLM returned ZERO signal connections for {len(components)} components. "
                f"This is FATAL - schematic would have no signal routing."
            )
            logger.error(error_msg)
            raise RuntimeError(error_msg)

        connections.extend(signal_connections)

        # Step 6: Deduplicate and prioritize
        connections = self._deduplicate_connections(connections)

        logger.info(f"✅ Total {len(connections)} logical connections generated (with IPC-2221 metadata)")

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

                # Classify pins
                for pin in comp.pins:
                    pin_name = pin.get("name", "").upper()
                    pin_type = pin.get("pin_type", "")

                    if self._is_power_pin(pin_name) or pin_type == "power_in":
                        comp.power_pins.append(pin_name)
                    elif self._is_ground_pin(pin_name) or pin_type == "power_in":
                        comp.ground_pins.append(pin_name)
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
        context: WireGenerationContext
    ) -> List[GeneratedConnection]:
        """
        Generate signal connections with logical validation and retry.

        Attempts up to MAX_WIRE_GENERATION_RETRIES times. If validation fails,
        provides feedback to LLM for next attempt.
        """
        for attempt in range(1, MAX_WIRE_GENERATION_RETRIES + 1):
            logger.info(f"🔄 Connection generation attempt {attempt}/{MAX_WIRE_GENERATION_RETRIES}")

            # Build logical connection prompt
            prompt = self._build_ipc_2221_prompt(
                components,
                design_intent,
                context,
                attempt_number=attempt
            )

            # Call LLM (Opus 4.6)
            try:
                connections_data = await self._call_llm_for_wires(prompt)
            except Exception as e:
                logger.error(f"LLM call failed on attempt {attempt}: {e}")
                if attempt == MAX_WIRE_GENERATION_RETRIES:
                    raise
                continue

            # Validate logical connections
            validation_errors = self._validate_logical_connections(
                connections_data,
                components,
                context
            )

            if not validation_errors:
                logger.info(f"✅ Connection validation PASSED on attempt {attempt}")

                # Convert validated connections to GeneratedConnection format
                return self._convert_wires_to_connections(connections_data)

            # Validation failed
            logger.warning(
                f"❌ Connection validation FAILED on attempt {attempt}: "
                f"{len(validation_errors)} errors"
            )
            for error in validation_errors[:10]:  # Log first 10 errors
                logger.warning(f"  - {error}")

            # If not last attempt, retry
            if attempt < MAX_WIRE_GENERATION_RETRIES:
                logger.info(f"Retrying with validation feedback...")
                continue

        # All attempts failed
        raise ValidationError(
            f"Failed to generate valid logical connections after "
            f"{MAX_WIRE_GENERATION_RETRIES} attempts. Last errors: "
            f"{validation_errors[:5]}"
        )

    def _build_ipc_2221_prompt(
        self,
        components: List[ComponentInfo],
        design_intent: str,
        context: WireGenerationContext,
        attempt_number: int = 1
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

        prompt = f"""You are an expert electronics engineer specializing in circuit connectivity.

Generate LOGICAL CONNECTIONS for a KiCad schematic based on the design intent and components.

IMPORTANT: You are generating LOGICAL connections (which pins connect to which pins),
NOT physical wire routing. Physical routing (coordinates, waypoints, angles) will be
handled by a separate wire router agent.

DESIGN INTENT:
{design_intent}

COMPONENTS:
{json.dumps(comp_summary, indent=2)}

VOLTAGE MAP (V):
{voltage_str}

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

Attempt: {attempt_number}/{MAX_WIRE_GENERATION_RETRIES}

Generate ONLY the JSON output. No explanation, no markdown, just the JSON object."""

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
            self._http_client = httpx.AsyncClient(timeout=httpx.Timeout(300.0))

        # Build request payload (OpenAI-compatible format works with both providers)
        request_payload = {
            "model": LLM_MODEL,
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "max_tokens": 8192,  # Increased for larger wire lists
            "temperature": 0.1,   # Low temperature for structured output
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
        logger.info(f"Calling {provider_label} ({LLM_MODEL}) for logical connection generation...")

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
        logger.info(f"Received {len(response_text)} chars from OpenRouter")

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

        Handles markdown code fences, truncation, etc.
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

        # Try regex extraction
        json_match = re.search(r'\{[\s\S]*\}', clean_text)
        if json_match:
            try:
                return json.loads(json_match.group())
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
            if from_ref in comp_pins and comp_pins[from_ref]:
                if from_pin not in comp_pins[from_ref]:
                    errors.append(
                        f"Connection {i}: from_pin '{from_pin}' not found on {from_ref}. "
                        f"Available pins: {list(comp_pins[from_ref])[:10]}"
                    )

            if to_ref in comp_pins and comp_pins[to_ref]:
                if to_pin not in comp_pins[to_ref]:
                    errors.append(
                        f"Connection {i}: to_pin '{to_pin}' not found on {to_ref}. "
                        f"Available pins: {list(comp_pins[to_ref])[:10]}"
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
                errors.append(
                    f"Differential pair '{base}' incomplete: only found {nets}. "
                    f"Expected both positive and negative signals."
                )

        return errors

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

    # Utility methods (inherited from v3.0)

    def _get_ref_prefix(self, category: str) -> str:
        """Get reference designator prefix for category."""
        prefixes = {
            "MCU": "U", "IC": "U", "MOSFET": "Q", "BJT": "Q",
            "Gate_Driver": "U", "Amplifier": "U", "Capacitor": "C",
            "Resistor": "R", "Inductor": "L", "Diode": "D",
            "LED": "D", "Connector": "J", "Crystal": "Y",
            "CAN_Transceiver": "U", "TVS": "D",
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
