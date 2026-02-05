"""
Connection Generator Agent - Infers circuit connections from BOM and design intent.

This agent analyzes:
1. Component categories and pin types (power, signal, analog, digital)
2. Design intent to understand circuit function
3. Standard conventions (bypass caps, power rails, pull-ups)
4. Reference designs for similar circuits

Author: Nexus EE Design Team
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

logger = logging.getLogger(__name__)

# OpenRouter configuration (following mageagent pattern)
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
OPENROUTER_MODEL = "anthropic/claude-opus-4.5"


class ConnectionType(Enum):
    """Type of electrical connection."""
    POWER = "power"        # VCC, GND power connections
    SIGNAL = "signal"      # General signal connections
    ANALOG = "analog"      # Analog signals (ADC, DAC, etc.)
    DIGITAL = "digital"    # Digital signals (GPIO, SPI, I2C, etc.)
    HIGH_SPEED = "high_speed"  # High-speed differential (USB, Ethernet)
    DIFFERENTIAL = "differential"  # Differential pairs


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


class ConnectionGeneratorAgent:
    """
    Generates circuit connections from BOM and design intent.

    Uses a combination of:
    1. Rule-based power routing (VCC/GND)
    2. Standard conventions (bypass caps)
    3. LLM inference for signal connections
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

    # Standard interface pin patterns for MCUs
    MCU_INTERFACES = {
        "SPI": {
            "master": ["MOSI", "MISO", "SCK", "SS", "CS", "NSS"],
            "slave": ["MOSI", "MISO", "SCK", "SS", "CS", "NSS"],
        },
        "I2C": {
            "master": ["SDA", "SCL"],
            "slave": ["SDA", "SCL"],
        },
        "UART": {
            "tx": ["TX", "TXD", "UART_TX"],
            "rx": ["RX", "RXD", "UART_RX"],
        },
        "CAN": {
            "controller": ["CAN_TX", "CAN_RX", "CANH", "CANL"],
            "transceiver": ["TXD", "RXD", "CANH", "CANL"],
        },
        "PWM": ["PWM", "OC", "TIM", "PHASE"],
        "ADC": ["ADC", "AIN", "ANALOG"],
    }

    def __init__(self, api_key: Optional[str] = None):
        """Initialize the connection generator with OpenRouter support (mageagent pattern)."""
        # OpenRouter API key (required for LLM operations)
        self._openrouter_api_key = os.environ.get("OPENROUTER_API_KEY", "")
        self._http_client: Optional[httpx.AsyncClient] = None

        if self._openrouter_api_key:
            logger.info("ConnectionGeneratorAgent: OpenRouter API key configured")
        else:
            logger.error(
                "CRITICAL: OPENROUTER_API_KEY not set! LLM connection generation will fail. "
                "Set OPENROUTER_API_KEY environment variable with a valid key from https://openrouter.ai/keys"
            )

    async def generate_connections(
        self,
        bom: List[Dict[str, Any]],
        design_intent: str,
        component_pins: Optional[Dict[str, List[Dict]]] = None,
    ) -> List[GeneratedConnection]:
        """
        Generate connections for a circuit based on BOM and design intent.

        Args:
            bom: List of BOM items with part_number, category, reference, etc.
            design_intent: Natural language description of circuit function
            component_pins: Optional dict of reference -> list of pins

        Returns:
            List of GeneratedConnection objects
        """
        connections: List[GeneratedConnection] = []

        # Step 1: Extract component information
        components = self._extract_component_info(bom, component_pins)
        logger.info(f"Extracted info for {len(components)} components")

        # Step 2: Generate power connections (rule-based)
        power_connections = self._generate_power_connections(components)
        connections.extend(power_connections)
        logger.info(f"Generated {len(power_connections)} power connections")

        # Step 3: Generate bypass capacitor connections
        bypass_connections = self._generate_bypass_cap_connections(components)
        connections.extend(bypass_connections)
        logger.info(f"Generated {len(bypass_connections)} bypass cap connections")

        # Step 4: Use LLM to infer signal connections (NO FALLBACK - FAIL FAST)
        # LLM is REQUIRED for intelligent connection generation. No silent degradation.
        signal_connections: List[GeneratedConnection] = []
        llm_error: Optional[str] = None

        try:
            signal_connections = await self._generate_signal_connections_llm(
                components, design_intent
            )
            if signal_connections:
                logger.info(f"Generated {len(signal_connections)} signal connections via LLM (Opus 4.5)")
        except Exception as e:
            llm_error = str(e)
            # FAIL FAST - Do not attempt fallback, provide verbose diagnostic
            error_msg = (
                f"CRITICAL: LLM connection generation FAILED with exception. "
                f"Error: {llm_error}. "
                f"Components attempted ({len(components)}): {[c.reference for c in components][:15]}... "
                f"Component categories: {list(set(c.category for c in components))}. "
                f"Design intent provided: '{design_intent[:300] if design_intent else 'NONE - this may be the cause'}...'. "
                f"DIAGNOSTIC CHECKLIST: "
                f"1) OPENROUTER_API_KEY env var: {'SET' if os.environ.get('OPENROUTER_API_KEY') else 'MISSING - THIS IS LIKELY THE CAUSE'}. "
                f"2) Network connectivity: Check if pod can reach openrouter.ai. "
                f"3) API quota: Check OpenRouter dashboard for rate limits. "
                f"4) Model availability: anthropic/claude-opus-4.5 may be unavailable. "
                f"NO FALLBACK - Schematic generation cannot proceed without LLM connections."
            )
            logger.error(error_msg)
            raise RuntimeError(error_msg)

        # VALIDATION: Check that signal connections were actually generated
        # NO FALLBACK - Empty result is FATAL
        if not signal_connections:
            error_msg = (
                f"CRITICAL: LLM returned ZERO signal connections for {len(components)} components. "
                f"This is a FATAL error - schematic would have no signal routing. "
                f"Components: {[c.reference for c in components][:15]}... "
                f"Component categories: {list(set(c.category for c in components))}. "
                f"Design intent: '{design_intent[:300] if design_intent else 'NONE PROVIDED - THIS IS LIKELY THE CAUSE'}...'. "
                f"DIAGNOSTIC CHECKLIST: "
                f"1) Design intent quality: Is it specific enough for LLM to infer connections? "
                f"2) Component categories: Are they recognized categories (MCU, sensor, driver, etc.)? "
                f"3) BOM completeness: Are all required components included? "
                f"4) LLM response: Check backend logs for raw LLM output - may contain parsing error. "
                f"NO FALLBACK - User must provide clearer design intent or check component definitions."
            )
            logger.error(error_msg)
            raise RuntimeError(error_msg)

        connections.extend(signal_connections)

        # Warn if fewer connections than components (may indicate incomplete inference)
        if len(signal_connections) < len(components):
            logger.warning(
                f"Signal connections ({len(signal_connections)}) fewer than components ({len(components)}). "
                f"Some components may not be connected. Review design intent for completeness."
            )

        # Step 5: Deduplicate and prioritize
        connections = self._deduplicate_connections(connections)

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

    def _get_ref_prefix(self, category: str) -> str:
        """Get reference designator prefix for category."""
        prefixes = {
            "MCU": "U",
            "IC": "U",
            "MOSFET": "Q",
            "BJT": "Q",
            "Transistor": "Q",
            "Gate_Driver": "U",
            "Amplifier": "U",
            "Capacitor": "C",
            "Resistor": "R",
            "Inductor": "L",
            "Diode": "D",
            "LED": "D",
            "Connector": "J",
            "Crystal": "Y",
            "Thermistor": "RT",
            "CAN_Transceiver": "U",
            "TVS": "D",
        }
        return prefixes.get(category, "U")

    def _is_power_pin(self, pin_name: str) -> bool:
        """Check if pin name matches power patterns."""
        for pattern in self.POWER_PIN_PATTERNS:
            if re.match(pattern, pin_name.upper()):
                return True
        return False

    def _is_ground_pin(self, pin_name: str) -> bool:
        """Check if pin name matches ground patterns."""
        for pattern in self.GROUND_PIN_PATTERNS:
            if re.match(pattern, pin_name.upper()):
                return True
        return False

    def _infer_pins_from_category(
        self,
        category: str,
        value: str
    ) -> Tuple[List[str], List[str], List[str]]:
        """Infer typical pins from component category."""
        power_pins = []
        ground_pins = []
        signal_pins = []

        category_lower = category.lower()

        if category_lower in ["resistor", "capacitor", "inductor"]:
            # Two-terminal passive
            signal_pins = ["1", "2"]
        elif category_lower in ["diode", "led"]:
            signal_pins = ["A", "K"]  # Anode, Kathode
        elif category_lower in ["mosfet"]:
            signal_pins = ["G", "D", "S"]  # Gate, Drain, Source
        elif category_lower in ["bjt", "transistor"]:
            signal_pins = ["B", "C", "E"]  # Base, Collector, Emitter
        elif category_lower in ["mcu", "ic"]:
            power_pins = ["VCC", "VDD"]
            ground_pins = ["GND", "VSS"]
            signal_pins = ["PA0", "PA1", "PB0", "PB1"]  # Generic GPIO
        elif category_lower == "gate_driver":
            power_pins = ["VCC", "VDD", "VBOOT"]
            ground_pins = ["GND", "VSS"]
            signal_pins = ["INH", "INL", "HO", "LO", "HS", "LS"]
        elif category_lower == "can_transceiver":
            power_pins = ["VCC"]
            ground_pins = ["GND"]
            signal_pins = ["TXD", "RXD", "CANH", "CANL"]
        elif category_lower == "crystal":
            signal_pins = ["XI", "XO"]
        elif category_lower in ["power", "regulator"]:
            power_pins = ["VIN"]
            ground_pins = ["GND"]
            signal_pins = ["VOUT"]
        elif category_lower == "thermistor":
            signal_pins = ["1", "2"]
        elif category_lower == "tvs":
            signal_pins = ["1", "2"]

        return power_pins, ground_pins, signal_pins

    def _generate_power_connections(
        self,
        components: List[ComponentInfo]
    ) -> List[GeneratedConnection]:
        """Generate power rail connections for all components."""
        connections = []

        # Find power rail sources (regulators, connectors with power)
        power_sources = []
        ground_sources = []

        for comp in components:
            if comp.category.lower() in ["power", "regulator", "connector"]:
                if "VOUT" in comp.signal_pins or "VCC" in comp.power_pins:
                    power_sources.append(comp)
                if comp.ground_pins:
                    ground_sources.append(comp)

        # Connect all power pins to VCC rail
        for comp in components:
            for power_pin in comp.power_pins:
                connections.append(GeneratedConnection(
                    from_ref=comp.reference,
                    from_pin=power_pin,
                    to_ref="PWR",  # Virtual power symbol
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
                    to_ref="PWR",  # Virtual ground symbol
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

        # Find capacitors and ICs
        capacitors = [c for c in components if c.category.lower() == "capacitor"]
        ics = [c for c in components if c.category.lower() in ["mcu", "ic", "gate_driver"]]

        # Assign bypass caps to ICs (simple 1:1 for now)
        bypass_caps = [c for c in capacitors if self._is_bypass_cap(c.value)]

        for i, ic in enumerate(ics):
            if i < len(bypass_caps):
                cap = bypass_caps[i]

                # Cap pin 1 -> IC VCC
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

                # Cap pin 2 -> IC GND
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

    def _is_bypass_cap(self, value: str) -> bool:
        """Check if capacitor value suggests bypass cap."""
        value_lower = value.lower()
        bypass_values = ["100n", "0.1u", "100nf", "10n", "1u", "10uf", "100uf"]
        return any(v in value_lower for v in bypass_values)

    def _generate_signal_connections_rules(
        self,
        components: List[ComponentInfo]
    ) -> List[GeneratedConnection]:
        """Generate signal connections using rule-based matching."""
        connections = []

        # Find MCU
        mcus = [c for c in components if c.category.lower() == "mcu"]
        if not mcus:
            return connections

        mcu = mcus[0]

        # Find gate drivers and connect to MCU PWM outputs
        gate_drivers = [c for c in components if c.category.lower() == "gate_driver"]
        for i, driver in enumerate(gate_drivers):
            # Connect PWM high to driver high input
            if "INH" in driver.signal_pins or "HI" in driver.signal_pins:
                input_pin = "INH" if "INH" in driver.signal_pins else "HI"
                connections.append(GeneratedConnection(
                    from_ref=mcu.reference,
                    from_pin=f"PWM{i*2}",
                    to_ref=driver.reference,
                    to_pin=input_pin,
                    net_name=f"PWM_H{i+1}",
                    connection_type=ConnectionType.DIGITAL,
                    priority=5,
                    notes="MCU PWM to gate driver high"
                ))

            # Connect PWM low to driver low input
            if "INL" in driver.signal_pins or "LI" in driver.signal_pins:
                input_pin = "INL" if "INL" in driver.signal_pins else "LI"
                connections.append(GeneratedConnection(
                    from_ref=mcu.reference,
                    from_pin=f"PWM{i*2+1}",
                    to_ref=driver.reference,
                    to_pin=input_pin,
                    net_name=f"PWM_L{i+1}",
                    connection_type=ConnectionType.DIGITAL,
                    priority=5,
                    notes="MCU PWM to gate driver low"
                ))

        # Find CAN transceiver and connect to MCU
        can_transceivers = [c for c in components if c.category.lower() == "can_transceiver"]
        for can in can_transceivers:
            if "TXD" in can.signal_pins:
                connections.append(GeneratedConnection(
                    from_ref=mcu.reference,
                    from_pin="CAN_TX",
                    to_ref=can.reference,
                    to_pin="TXD",
                    net_name="CAN_TX",
                    connection_type=ConnectionType.DIGITAL,
                    priority=5,
                    notes="MCU CAN TX to transceiver"
                ))
            if "RXD" in can.signal_pins:
                connections.append(GeneratedConnection(
                    from_ref=can.reference,
                    from_pin="RXD",
                    to_ref=mcu.reference,
                    to_pin="CAN_RX",
                    net_name="CAN_RX",
                    connection_type=ConnectionType.DIGITAL,
                    priority=5,
                    notes="Transceiver RX to MCU CAN"
                ))

        # Find current sense amplifiers and connect to MCU ADC
        amplifiers = [c for c in components if c.category.lower() == "amplifier"]
        for i, amp in enumerate(amplifiers):
            if "OUT" in amp.signal_pins or "VOUT" in amp.signal_pins:
                out_pin = "OUT" if "OUT" in amp.signal_pins else "VOUT"
                connections.append(GeneratedConnection(
                    from_ref=amp.reference,
                    from_pin=out_pin,
                    to_ref=mcu.reference,
                    to_pin=f"ADC{i}",
                    net_name=f"ISENSE{i+1}",
                    connection_type=ConnectionType.ANALOG,
                    priority=5,
                    notes="Current sense to MCU ADC"
                ))

        return connections

    async def _generate_signal_connections_llm(
        self,
        components: List[ComponentInfo],
        design_intent: str
    ) -> List[GeneratedConnection]:
        """Use LLM (Opus 4.5 via OpenRouter) to infer signal connections from design intent.

        Following mageagent pattern: uses httpx with OpenRouter /chat/completions endpoint.
        """
        # Validate API key
        if not self._openrouter_api_key:
            error_msg = (
                "CRITICAL: Cannot generate signal connections - OPENROUTER_API_KEY not set!\n"
                "Action Required: Set OPENROUTER_API_KEY environment variable\n"
                "Documentation: https://openrouter.ai/keys"
            )
            logger.error(error_msg)
            raise ValueError(error_msg)

        # Build component summary for LLM
        comp_summary = []
        for comp in components:
            comp_summary.append({
                "reference": comp.reference,
                "part_number": comp.part_number,
                "category": comp.category,
                "value": comp.value,
                "signal_pins": comp.signal_pins[:10],  # Limit pins
            })

        prompt = f"""You are an expert electronics engineer. Given the following components and design intent, generate the signal connections needed to make a functional circuit.

DESIGN INTENT:
{design_intent}

COMPONENTS:
{json.dumps(comp_summary, indent=2)}

Generate connections in this exact JSON format:
[
  {{
    "from_ref": "U1",
    "from_pin": "PA0",
    "to_ref": "U2",
    "to_pin": "IN1",
    "net_name": "PWM_PHASE_A",
    "connection_type": "digital",
    "notes": "MCU PWM output to gate driver phase A"
  }}
]

Rules:
1. Connect MCU peripherals to appropriate ICs (SPI, I2C, UART, CAN, ADC, PWM)
2. Connect gate driver outputs to MOSFETs
3. Connect current sense amplifiers to MCU ADC inputs
4. Use meaningful net names
5. Only output valid JSON array, no other text

Generate the connections:"""

        # Initialize HTTP client if needed (following mageagent pattern)
        if self._http_client is None:
            self._http_client = httpx.AsyncClient(timeout=httpx.Timeout(300.0))

        # Build OpenRouter request (OpenAI-compatible format, NOT Anthropic format)
        request_payload = {
            "model": OPENROUTER_MODEL,
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "max_tokens": 4096,
            "temperature": 0.1,  # Low temperature for structured output
        }

        headers = {
            "Authorization": f"Bearer {self._openrouter_api_key}",
            "HTTP-Referer": "https://adverant.ai",
            "X-Title": "Nexus EE Design Connection Generator",
            "Content-Type": "application/json",
        }

        try:
            logger.info(f"Calling OpenRouter API ({OPENROUTER_MODEL}) for connection generation...")

            response = await self._http_client.post(
                f"{OPENROUTER_BASE_URL}/chat/completions",
                json=request_payload,
                headers=headers,
            )

            # Check for HTTP errors
            if response.status_code != 200:
                error_body = response.text
                error_msg = (
                    f"OpenRouter API Error:\n"
                    f"  Status: {response.status_code}\n"
                    f"  Response: {error_body[:500]}...\n"
                    f"  Model: {OPENROUTER_MODEL}\n"
                    f"  API Key: {self._openrouter_api_key[:15]}...\n"
                    f"  Action: Check API key validity at https://openrouter.ai/keys"
                )
                logger.error(error_msg)
                raise ValueError(error_msg)

            # Parse response
            response_data = response.json()

            # Extract content from OpenAI-format response
            if "choices" not in response_data or len(response_data["choices"]) == 0:
                error_msg = f"OpenRouter returned empty response: {json.dumps(response_data)}"
                logger.error(error_msg)
                raise ValueError(error_msg)

            response_text = response_data["choices"][0]["message"]["content"].strip()
            logger.info(f"Received {len(response_text)} chars from OpenRouter")

            # Strip markdown code fences if present (LLM often wraps JSON in ```json ... ```)
            clean_text = response_text
            if clean_text.startswith("```"):
                # Remove opening fence (```json or ```)
                first_newline = clean_text.find('\n')
                if first_newline > 0:
                    clean_text = clean_text[first_newline + 1:]
                else:
                    # No newline after opening fence, remove just the fence
                    clean_text = re.sub(r'^```\w*\s*', '', clean_text)
            if clean_text.rstrip().endswith("```"):
                # Remove closing fence
                clean_text = clean_text.rstrip()[:-3].rstrip()

            logger.debug(f"Cleaned response text (first 200 chars): {clean_text[:200]}")
            logger.debug(f"Cleaned response text (last 100 chars): ...{clean_text[-100:] if len(clean_text) > 100 else clean_text}")

            # Multi-strategy JSON parsing (handles truncated responses)
            connections_data = None
            parse_error = None

            # Strategy 1: Direct parse (if response is already valid JSON)
            clean_text = clean_text.strip()
            try:
                connections_data = json.loads(clean_text)
                logger.info("JSON parsed via Strategy 1 (direct parse)")
            except json.JSONDecodeError as e:
                parse_error = f"Direct parse failed: {e}"
                logger.debug(parse_error)

            # Strategy 2: Regex to extract [...]
            if connections_data is None:
                json_match = re.search(r'\[[\s\S]*\]', clean_text)
                if json_match:
                    try:
                        connections_data = json.loads(json_match.group())
                        logger.info("JSON parsed via Strategy 2 (regex extraction)")
                    except json.JSONDecodeError as e:
                        parse_error = f"Regex extraction failed: {e}"
                        logger.debug(parse_error)

            # Strategy 3: If starts with '[', try to repair truncated JSON
            if connections_data is None and clean_text.startswith('['):
                # Count bracket balance to find where to repair
                bracket_count = 0
                last_complete_obj = -1
                in_string = False
                escape_next = False

                for i, char in enumerate(clean_text):
                    if escape_next:
                        escape_next = False
                        continue
                    if char == '\\':
                        escape_next = True
                        continue
                    if char == '"' and not escape_next:
                        in_string = not in_string
                        continue
                    if in_string:
                        continue
                    if char == '{':
                        bracket_count += 1
                    elif char == '}':
                        bracket_count -= 1
                        if bracket_count == 0:
                            last_complete_obj = i

                # Try to salvage up to the last complete object
                if last_complete_obj > 0:
                    salvaged = clean_text[:last_complete_obj + 1]
                    # Close any open brackets
                    if not salvaged.rstrip().endswith(']'):
                        salvaged = salvaged.rstrip()
                        if salvaged.endswith(','):
                            salvaged = salvaged[:-1]
                        salvaged += ']'
                    try:
                        connections_data = json.loads(salvaged)
                        logger.warning(f"JSON repaired via Strategy 3 (truncation repair). Original: {len(clean_text)} chars, salvaged: {len(salvaged)} chars")
                    except json.JSONDecodeError as e:
                        parse_error = f"Truncation repair failed: {e}"
                        logger.debug(parse_error)

            # Final failure - raise with detailed diagnostics
            if connections_data is None:
                # Log full response for debugging (not truncated)
                logger.error(f"FULL cleaned response ({len(clean_text)} chars): {clean_text}")
                error_msg = (
                    f"LLM did not return valid JSON array after all parse strategies.\n"
                    f"  Original response (first 300 chars): {response_text[:300]}...\n"
                    f"  Cleaned response (first 300 chars): {clean_text[:300]}...\n"
                    f"  Cleaned response (last 200 chars): ...{clean_text[-200:] if len(clean_text) > 200 else clean_text}\n"
                    f"  Last parse error: {parse_error}\n"
                    f"  Expected format: [{'{'}\"from_ref\": \"U1\", \"from_pin\": \"PA0\", ...{'}'}]"
                )
                logger.error(error_msg)
                raise ValueError(error_msg)

            # Validate we got a list
            if not isinstance(connections_data, list):
                error_msg = f"Expected JSON array but got {type(connections_data).__name__}"
                logger.error(error_msg)
                raise ValueError(error_msg)
            logger.info(f"Parsed {len(connections_data)} connections from LLM response")

            connections = []
            for conn in connections_data:
                conn_type = ConnectionType.SIGNAL
                type_str = conn.get("connection_type", "signal").lower()
                if type_str == "digital":
                    conn_type = ConnectionType.DIGITAL
                elif type_str == "analog":
                    conn_type = ConnectionType.ANALOG
                elif type_str == "power":
                    conn_type = ConnectionType.POWER

                connections.append(GeneratedConnection(
                    from_ref=conn.get("from_ref", ""),
                    from_pin=conn.get("from_pin", ""),
                    to_ref=conn.get("to_ref", ""),
                    to_pin=conn.get("to_pin", ""),
                    net_name=conn.get("net_name", "NET"),
                    connection_type=conn_type,
                    priority=5,
                    notes=conn.get("notes", "LLM generated")
                ))

            return connections

        except httpx.RequestError as e:
            error_msg = (
                f"OpenRouter HTTP Request Failed:\n"
                f"  Error: {str(e)}\n"
                f"  URL: {OPENROUTER_BASE_URL}/chat/completions\n"
                f"  Action: Check network connectivity"
            )
            logger.error(error_msg)
            raise ValueError(error_msg)
        except json.JSONDecodeError as e:
            error_msg = f"Failed to parse LLM response as JSON: {str(e)}"
            logger.error(error_msg)
            raise ValueError(error_msg)
        except Exception as e:
            error_msg = f"LLM connection generation failed: {str(e)}"
            logger.error(error_msg)
            raise ValueError(error_msg)

    def _deduplicate_connections(
        self,
        connections: List[GeneratedConnection]
    ) -> List[GeneratedConnection]:
        """Remove duplicate connections, keeping highest priority."""
        seen: Dict[str, GeneratedConnection] = {}

        for conn in connections:
            # Create unique key (order-independent for the two endpoints)
            endpoints = sorted([
                f"{conn.from_ref}.{conn.from_pin}",
                f"{conn.to_ref}.{conn.to_pin}"
            ])
            key = "|".join(endpoints)

            if key not in seen or conn.priority > seen[key].priority:
                seen[key] = conn

        return list(seen.values())

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
    import asyncio

    logging.basicConfig(level=logging.INFO)

    async def test():
        generator = ConnectionGeneratorAgent()

        # Test BOM
        bom = [
            {"part_number": "STM32G431CBT6", "category": "MCU", "reference": "U1"},
            {"part_number": "UCC21520DW", "category": "Gate_Driver", "reference": "U2"},
            {"part_number": "UCC21520DW", "category": "Gate_Driver", "reference": "U3"},
            {"part_number": "TJA1051T/3", "category": "CAN_Transceiver", "reference": "U4"},
            {"part_number": "INA240A4PWR", "category": "Amplifier", "reference": "U5"},
            {"part_number": "100nF", "category": "Capacitor", "reference": "C1", "value": "100nF"},
            {"part_number": "100nF", "category": "Capacitor", "reference": "C2", "value": "100nF"},
            {"part_number": "10uF", "category": "Capacitor", "reference": "C3", "value": "10uF"},
        ]

        design_intent = """
        FOC (Field Oriented Control) ESC for brushless motor.
        - MCU generates 6 PWM signals for 3-phase gate drivers
        - Gate drivers control half-bridge MOSFETs
        - Current sense amplifiers measure phase currents
        - CAN bus for communication
        """

        connections = await generator.generate_connections(bom, design_intent)

        print(f"\nGenerated {len(connections)} connections:\n")
        for conn in connections:
            print(f"  {conn.from_ref}.{conn.from_pin} -> {conn.to_ref}.{conn.to_pin} ({conn.net_name}) [{conn.connection_type.value}]")
            if conn.notes:
                print(f"    Notes: {conn.notes}")

        # Convert to assembler format
        print("\n\nAssembler format:")
        print(json.dumps(generator.to_assembler_format(connections), indent=2))

    asyncio.run(test())
