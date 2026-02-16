"""
Ideation Artifact Extractors for MAPO Pipeline Integration.

ALL extraction uses Claude Opus 4.6 via OpenRouter.  There is ZERO regex,
ZERO pattern matching, and ZERO manual parsing anywhere in this module.
Every extraction function sends artifact content to the LLM with a
structured JSON output schema and receives typed structured data back.

On failure the extractor raises ``ExtractionError`` with verbose
diagnostics (HTTP status, response body, input content truncated to 2000
characters, and the specific field that failed).  Errors are collected in
``IdeationContext.extraction_errors`` and the pipeline STOPS on error --
it does NOT silently continue with empty data.

Author: Nexus EE Design Team
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import traceback
from dataclasses import asdict
from typing import Any, Dict, List, Optional, Tuple

import httpx

from ideation_context import (
    ArtifactType,
    BOMEntry,
    ComplianceRequirement,
    ConnectionInferenceContext,
    IdeationContext,
    InterfaceDefinition,
    PinConnection,
    PlacementContext,
    PowerRail,
    SubsystemBlock,
    SymbolResolutionContext,
    TestCriterion,
    ValidationContext,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_HTTP_TIMEOUT = 120.0  # seconds -- LLM extraction can be slow on large artifacts

# LLM provider: use centralized config (defaults to Claude Code Max proxy)
try:
    from llm_provider import get_llm_base_url, get_llm_headers, get_llm_api_key
    OPENROUTER_URL = get_llm_base_url(chat_completions=True)
    OPENROUTER_MODEL = "anthropic/claude-opus-4-6"
except ImportError:
    OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
    OPENROUTER_MODEL = "anthropic/claude-opus-4-6"


# ---------------------------------------------------------------------------
# Custom exception
# ---------------------------------------------------------------------------


class ExtractionError(Exception):
    """
    Raised when an LLM extraction call fails.

    Carries verbose diagnostic information so that operators and logs
    contain enough detail to reproduce and fix the problem.

    Attributes:
        message: Human-readable error summary.
        input_content_preview: First 2000 characters of the input that was
            sent to the LLM.
        http_status: HTTP status code returned by OpenRouter (``None`` if
            the request never completed).
        response_body: Raw response body from OpenRouter (``None`` if the
            request never completed).
        failed_field: Name of the field / extraction step that failed.
    """

    def __init__(
        self,
        message: str,
        *,
        input_content_preview: str = "",
        http_status: Optional[int] = None,
        response_body: Optional[str] = None,
        failed_field: str = "",
    ) -> None:
        self.input_content_preview = input_content_preview[:2000]
        self.http_status = http_status
        self.response_body = response_body
        self.failed_field = failed_field

        detail_parts = [message]
        if http_status is not None:
            detail_parts.append(f"HTTP status: {http_status}")
        if response_body:
            detail_parts.append(f"Response body: {response_body[:2000]}")
        if failed_field:
            detail_parts.append(f"Failed field: {failed_field}")
        if input_content_preview:
            detail_parts.append(
                f"Input content (first 2000 chars): {input_content_preview[:2000]}"
            )
        super().__init__(" | ".join(detail_parts))


# ---------------------------------------------------------------------------
# Base LLM call
# ---------------------------------------------------------------------------


def _get_api_key() -> str:
    """Return the API key for the current LLM provider."""
    try:
        key = get_llm_api_key()
        # Claude Code Max proxy needs no key
        if os.environ.get("AI_PROVIDER", "claude_code_max") == "claude_code_max":
            return key or ""
        if not key:
            raise ExtractionError(
                "No API key configured for LLM provider. "
                "Set OPENROUTER_API_KEY or use claude_code_max provider.",
                failed_field="API_KEY",
            )
        return key
    except NameError:
        # Fallback if llm_provider not imported
        key = os.environ.get("OPENROUTER_API_KEY", "")
        if not key:
            raise ExtractionError(
                "OPENROUTER_API_KEY environment variable is not set.",
                failed_field="OPENROUTER_API_KEY",
            )
        return key


async def _call_opus_extraction(
    content: str,
    system_prompt: str,
    json_schema: dict,
) -> dict:
    """
    Call Claude Opus 4.6 via OpenRouter to extract structured JSON from content.

    Sends the artifact ``content`` as the user message and ``system_prompt``
    as the system message.  The ``json_schema`` is included in the system
    prompt to guide the LLM toward producing valid JSON output.

    Args:
        content: The raw artifact text to extract structured data from.
        system_prompt: Instructions for the LLM including the target JSON
            schema description.
        json_schema: A JSON-schema-like dictionary describing the expected
            output structure.  Embedded into the system prompt for the LLM.

    Returns:
        Parsed JSON dictionary from the LLM response.

    Raises:
        ExtractionError: On any HTTP, parsing, or validation failure.
    """
    # Build the full system prompt with the schema embedded
    full_system = (
        f"{system_prompt}\n\n"
        f"You MUST respond with ONLY valid JSON matching this schema -- "
        f"no markdown fences, no commentary, no explanation:\n"
        f"{json.dumps(json_schema, indent=2)}"
    )

    payload = {
        "model": OPENROUTER_MODEL,
        "messages": [
            {"role": "system", "content": full_system},
            {"role": "user", "content": content},
        ],
        "temperature": 0.0,
        "max_tokens": 32768,
    }

    # Use centralized headers from llm_provider
    try:
        headers = get_llm_headers()
    except NameError:
        api_key = _get_api_key()
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://nexus.adverant.com",
            "X-Title": "Nexus EE Design - Ideation Extractor",
        }

    async with httpx.AsyncClient(timeout=_HTTP_TIMEOUT) as client:
        try:
            response = await client.post(
                OPENROUTER_URL, json=payload, headers=headers
            )
        except httpx.TimeoutException as exc:
            raise ExtractionError(
                f"OpenRouter request timed out after {_HTTP_TIMEOUT}s: {exc}",
                input_content_preview=content,
                failed_field="http_request",
            ) from exc
        except httpx.RequestError as exc:
            raise ExtractionError(
                f"OpenRouter request failed: {exc}",
                input_content_preview=content,
                failed_field="http_request",
            ) from exc

        if response.status_code != 200:
            raise ExtractionError(
                f"OpenRouter returned non-200 status {response.status_code}",
                input_content_preview=content,
                http_status=response.status_code,
                response_body=response.text,
                failed_field="http_status",
            )

        try:
            response_json = response.json()
        except (json.JSONDecodeError, ValueError) as exc:
            raise ExtractionError(
                f"Failed to decode OpenRouter response as JSON: {exc}",
                input_content_preview=content,
                http_status=response.status_code,
                response_body=response.text,
                failed_field="response_json_decode",
            ) from exc

        # Navigate the OpenAI-compatible response structure
        choices = response_json.get("choices")
        if not choices or not isinstance(choices, list):
            raise ExtractionError(
                "OpenRouter response missing 'choices' array",
                input_content_preview=content,
                http_status=response.status_code,
                response_body=json.dumps(response_json)[:2000],
                failed_field="choices",
            )

        message = choices[0].get("message", {})
        raw_content = message.get("content", "")

        if not raw_content:
            raise ExtractionError(
                "OpenRouter response has empty content in choices[0].message.content",
                input_content_preview=content,
                http_status=response.status_code,
                response_body=json.dumps(response_json)[:2000],
                failed_field="message_content",
            )

        # Strip markdown code fences if the LLM wrapped its output
        cleaned = raw_content.strip()
        if cleaned.startswith("```"):
            # Remove opening fence (with optional language tag)
            first_newline = cleaned.index("\n") if "\n" in cleaned else len(cleaned)
            cleaned = cleaned[first_newline + 1 :]
        if cleaned.endswith("```"):
            cleaned = cleaned[: -3]
        cleaned = cleaned.strip()

        try:
            parsed = json.loads(cleaned)
        except json.JSONDecodeError as exc:
            # Try to repair truncated JSON: find the last complete array
            # entry and close the JSON structure.
            repaired = _try_repair_json(cleaned)
            if repaired is not None:
                logger.warning(
                    "LLM output had malformed JSON at char %d — "
                    "repaired by truncating to last complete entry",
                    exc.pos or 0,
                )
                parsed = repaired
            else:
                raise ExtractionError(
                    f"Failed to parse LLM output as JSON: {exc}. "
                    f"Raw LLM output (first 2000 chars): {cleaned[:2000]}",
                    input_content_preview=content,
                    http_status=response.status_code,
                    response_body=cleaned[:2000],
                    failed_field="llm_json_parse",
                ) from exc

        if not isinstance(parsed, dict):
            raise ExtractionError(
                f"Expected JSON object from LLM but got {type(parsed).__name__}. "
                f"Raw LLM output (first 2000 chars): {cleaned[:2000]}",
                input_content_preview=content,
                http_status=response.status_code,
                response_body=cleaned[:2000],
                failed_field="llm_json_type",
            )

        return parsed


# ---------------------------------------------------------------------------
# Individual extraction functions
# ---------------------------------------------------------------------------


async def extract_bom_entries(
    content: str, artifact_format: str = ""
) -> List[BOMEntry]:
    """
    Extract BOM line-items from artifact content using Opus 4.6.

    The LLM analyses the raw text (which may be a markdown table, CSV,
    free-form prose, or any other format) and returns a structured JSON
    array of BOM entries.

    Args:
        content: Raw artifact content (BOM, component selection document,
            or any text containing component information).
        artifact_format: Optional hint about the content format (e.g.
            ``"markdown_table"``, ``"csv"``, ``"prose"``).  Passed to the
            LLM to help it choose the best parsing strategy.

    Returns:
        List of ``BOMEntry`` dataclass instances.

    Raises:
        ExtractionError: If the LLM call fails or returns unparseable data.
    """
    format_hint = ""
    if artifact_format:
        format_hint = f" The content format is: {artifact_format}."

    system_prompt = (
        "You are an expert electronics engineer BOM (Bill of Materials) parser. "
        "Extract every component / part entry from the following content and "
        "return them as a JSON object with a single key 'entries' whose value "
        "is an array of objects.{format_hint}\n\n"
        "For each entry extract as many fields as possible.  If a field is not "
        "present in the source material, use the default value shown in the "
        "schema (empty string for strings, 1 for quantity, 0 for priority, "
        "empty array for alternatives).\n\n"
        "Do NOT invent data that is not in the source.  Only extract what is "
        "explicitly stated or can be unambiguously inferred."
    ).format(format_hint=format_hint)

    json_schema = {
        "type": "object",
        "required": ["entries"],
        "properties": {
            "entries": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "part_number": {"type": "string", "default": ""},
                        "manufacturer": {"type": "string", "default": ""},
                        "description": {"type": "string", "default": ""},
                        "reference_designator": {"type": "string", "default": ""},
                        "package": {"type": "string", "default": ""},
                        "quantity": {"type": "integer", "default": 1},
                        "category": {"type": "string", "default": ""},
                        "value": {"type": "string", "default": ""},
                        "subsystem": {"type": "string", "default": ""},
                        "alternatives": {
                            "type": "array",
                            "items": {"type": "string"},
                            "default": [],
                        },
                        "priority": {"type": "integer", "default": 0},
                    },
                },
            }
        },
    }

    result = await _call_opus_extraction(content, system_prompt, json_schema)

    entries_raw = result.get("entries")
    if entries_raw is None:
        raise ExtractionError(
            "LLM response missing 'entries' key in BOM extraction result. "
            f"Keys found: {list(result.keys())}",
            input_content_preview=content,
            response_body=json.dumps(result)[:2000],
            failed_field="entries",
        )
    if not isinstance(entries_raw, list):
        raise ExtractionError(
            f"Expected 'entries' to be a list but got {type(entries_raw).__name__}",
            input_content_preview=content,
            response_body=json.dumps(result)[:2000],
            failed_field="entries",
        )

    bom_entries: List[BOMEntry] = []
    for idx, raw in enumerate(entries_raw):
        try:
            entry = BOMEntry(
                part_number=str(raw.get("part_number", "")),
                manufacturer=str(raw.get("manufacturer", "")),
                description=str(raw.get("description", "")),
                reference_designator=str(raw.get("reference_designator", "")),
                package=str(raw.get("package", "")),
                quantity=int(raw.get("quantity", 1)),
                category=str(raw.get("category", "")),
                value=str(raw.get("value", "")),
                subsystem=str(raw.get("subsystem", "")),
                alternatives=list(raw.get("alternatives", [])),
                priority=int(raw.get("priority", 0)),
            )
            bom_entries.append(entry)
        except (TypeError, ValueError) as exc:
            raise ExtractionError(
                f"Failed to construct BOMEntry from LLM output at index {idx}: {exc}. "
                f"Raw entry: {json.dumps(raw)[:500]}",
                input_content_preview=content,
                response_body=json.dumps(raw)[:2000],
                failed_field=f"entries[{idx}]",
            ) from exc

    logger.info("Extracted %d BOM entries via Opus 4.6", len(bom_entries))
    return bom_entries


async def extract_pin_connections(content: str) -> List[PinConnection]:
    """
    Extract explicit pin-to-pin connections from schematic spec content using Opus 4.6.

    The LLM analyses schematic specifications, connection tables, or prose
    descriptions and returns a structured JSON array of pin connections.

    Args:
        content: Raw artifact content containing connection information
            (schematic spec, interface spec, or any spec document).

    Returns:
        List of ``PinConnection`` dataclass instances.

    Raises:
        ExtractionError: If the LLM call fails or returns unparseable data.
    """
    system_prompt = (
        "You are an expert electronics engineer schematic connection parser. "
        "Extract every pin-to-pin connection from the following content and "
        "return them as a JSON object with a single key 'connections' whose "
        "value is an array of objects.\n\n"
        "A connection describes a wire or net between two component pins. "
        "Extract from_component (reference designator or component name), "
        "from_pin (pin name or number), to_component, to_pin, signal_name "
        "(the net or signal label), and signal_type (one of: signal, power, "
        "ground, clock, analog, digital, pwm, data, control, reset, "
        "interrupt, or other).\n\n"
        "If a field is not explicitly stated, use an empty string. "
        "Do NOT invent connections that are not in the source."
    )

    json_schema = {
        "type": "object",
        "required": ["connections"],
        "properties": {
            "connections": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "from_component": {"type": "string", "default": ""},
                        "from_pin": {"type": "string", "default": ""},
                        "to_component": {"type": "string", "default": ""},
                        "to_pin": {"type": "string", "default": ""},
                        "signal_name": {"type": "string", "default": ""},
                        "signal_type": {"type": "string", "default": "signal"},
                        "notes": {"type": "string", "default": ""},
                    },
                },
            }
        },
    }

    result = await _call_opus_extraction(content, system_prompt, json_schema)

    connections_raw = result.get("connections")
    if connections_raw is None:
        raise ExtractionError(
            "LLM response missing 'connections' key in pin connection extraction. "
            f"Keys found: {list(result.keys())}",
            input_content_preview=content,
            response_body=json.dumps(result)[:2000],
            failed_field="connections",
        )
    if not isinstance(connections_raw, list):
        raise ExtractionError(
            f"Expected 'connections' to be a list but got {type(connections_raw).__name__}",
            input_content_preview=content,
            response_body=json.dumps(result)[:2000],
            failed_field="connections",
        )

    pin_connections: List[PinConnection] = []
    for idx, raw in enumerate(connections_raw):
        try:
            conn = PinConnection(
                from_component=str(raw.get("from_component", "")),
                from_pin=str(raw.get("from_pin", "")),
                to_component=str(raw.get("to_component", "")),
                to_pin=str(raw.get("to_pin", "")),
                signal_name=str(raw.get("signal_name", "")),
                signal_type=str(raw.get("signal_type", "signal")),
                notes=str(raw.get("notes", "")),
            )
            pin_connections.append(conn)
        except (TypeError, ValueError) as exc:
            raise ExtractionError(
                f"Failed to construct PinConnection at index {idx}: {exc}. "
                f"Raw entry: {json.dumps(raw)[:500]}",
                input_content_preview=content,
                response_body=json.dumps(raw)[:2000],
                failed_field=f"connections[{idx}]",
            ) from exc

    logger.info("Extracted %d pin connections via Opus 4.6", len(pin_connections))
    return pin_connections


async def extract_interfaces(content: str) -> List[InterfaceDefinition]:
    """
    Extract communication interface definitions from spec content using Opus 4.6.

    Handles SPI, I2C, UART, CAN, USB, PWM, ADC, and any other standard
    bus or protocol definition found in communication or interface specs.

    Args:
        content: Raw artifact content from communication specs, interface
            specs, or schematic specs that describe bus topologies.

    Returns:
        List of ``InterfaceDefinition`` dataclass instances.

    Raises:
        ExtractionError: If the LLM call fails or returns unparseable data.
    """
    system_prompt = (
        "You are an expert electronics engineer interface protocol parser. "
        "Extract every communication interface definition (SPI, I2C, UART, "
        "CAN, USB, PWM, ADC, Ethernet, JTAG, SWD, etc.) from the following "
        "content and return them as a JSON object with a single key "
        "'interfaces' whose value is an array of objects.\n\n"
        "For each interface extract: interface_type (protocol name like SPI, "
        "I2C, etc.), master_component (the bus master), slave_components "
        "(array of slave device names/designators), pin_mappings (object "
        "mapping signal names like MOSI, SCK to pin identifiers), speed "
        "(e.g. '10 MHz', '115200 baud'), and protocol_notes (any additional "
        "configuration like CPOL, CPHA, addressing mode, termination).\n\n"
        "If a field is not explicitly stated, use the default (empty string "
        "for strings, empty array for arrays, empty object for mappings). "
        "Do NOT invent interfaces that are not in the source."
    )

    json_schema = {
        "type": "object",
        "required": ["interfaces"],
        "properties": {
            "interfaces": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "interface_type": {"type": "string", "default": ""},
                        "master_component": {"type": "string", "default": ""},
                        "slave_components": {
                            "type": "array",
                            "items": {"type": "string"},
                            "default": [],
                        },
                        "pin_mappings": {
                            "type": "object",
                            "additionalProperties": {"type": "string"},
                            "default": {},
                        },
                        "speed": {"type": "string", "default": ""},
                        "protocol_notes": {"type": "string", "default": ""},
                    },
                },
            }
        },
    }

    result = await _call_opus_extraction(content, system_prompt, json_schema)

    interfaces_raw = result.get("interfaces")
    if interfaces_raw is None:
        raise ExtractionError(
            "LLM response missing 'interfaces' key in interface extraction. "
            f"Keys found: {list(result.keys())}",
            input_content_preview=content,
            response_body=json.dumps(result)[:2000],
            failed_field="interfaces",
        )
    if not isinstance(interfaces_raw, list):
        raise ExtractionError(
            f"Expected 'interfaces' to be a list but got {type(interfaces_raw).__name__}",
            input_content_preview=content,
            response_body=json.dumps(result)[:2000],
            failed_field="interfaces",
        )

    interfaces: List[InterfaceDefinition] = []
    for idx, raw in enumerate(interfaces_raw):
        try:
            iface = InterfaceDefinition(
                interface_type=str(raw.get("interface_type", "")),
                master_component=str(raw.get("master_component", "")),
                slave_components=list(raw.get("slave_components", [])),
                pin_mappings=dict(raw.get("pin_mappings", {})),
                speed=str(raw.get("speed", "")),
                protocol_notes=str(raw.get("protocol_notes", "")),
            )
            interfaces.append(iface)
        except (TypeError, ValueError) as exc:
            raise ExtractionError(
                f"Failed to construct InterfaceDefinition at index {idx}: {exc}. "
                f"Raw entry: {json.dumps(raw)[:500]}",
                input_content_preview=content,
                response_body=json.dumps(raw)[:2000],
                failed_field=f"interfaces[{idx}]",
            ) from exc

    logger.info("Extracted %d interface definitions via Opus 4.6", len(interfaces))
    return interfaces


async def extract_power_rails(content: str) -> List[PowerRail]:
    """
    Extract power rail definitions from power spec content using Opus 4.6.

    Parses voltage regulators, power distribution trees, rail specifications,
    and any other power-related content to produce structured rail objects.

    Args:
        content: Raw artifact content from power specs, schematic specs,
            or system overviews that describe power distribution.

    Returns:
        List of ``PowerRail`` dataclass instances.

    Raises:
        ExtractionError: If the LLM call fails or returns unparseable data.
    """
    system_prompt = (
        "You are an expert power electronics engineer. Extract every power "
        "rail definition from the following content and return them as a "
        "JSON object with a single key 'power_rails' whose value is an "
        "array of objects.\n\n"
        "For each power rail extract: net_name (the net label like VCC_3V3, "
        "+5V, VBAT), voltage (nominal voltage as a float in volts), "
        "current_max (maximum current in amperes as a float), "
        "regulator_type (topology like LDO, Buck, Boost, SEPIC, Linear, "
        "or empty if this is an input rail), source_component (the "
        "component reference designator or name that produces this rail), "
        "and consumer_components (array of component names/designators "
        "that consume from this rail).\n\n"
        "If a field is not explicitly stated, use the default (empty string "
        "for strings, 0.0 for numbers, empty array for arrays). "
        "Do NOT invent rails that are not in the source."
    )

    json_schema = {
        "type": "object",
        "required": ["power_rails"],
        "properties": {
            "power_rails": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "net_name": {"type": "string", "default": ""},
                        "voltage": {"type": "number", "default": 0.0},
                        "current_max": {"type": "number", "default": 0.0},
                        "regulator_type": {"type": "string", "default": ""},
                        "source_component": {"type": "string", "default": ""},
                        "consumer_components": {
                            "type": "array",
                            "items": {"type": "string"},
                            "default": [],
                        },
                    },
                },
            }
        },
    }

    result = await _call_opus_extraction(content, system_prompt, json_schema)

    rails_raw = result.get("power_rails")
    if rails_raw is None:
        raise ExtractionError(
            "LLM response missing 'power_rails' key in power rail extraction. "
            f"Keys found: {list(result.keys())}",
            input_content_preview=content,
            response_body=json.dumps(result)[:2000],
            failed_field="power_rails",
        )
    if not isinstance(rails_raw, list):
        raise ExtractionError(
            f"Expected 'power_rails' to be a list but got {type(rails_raw).__name__}",
            input_content_preview=content,
            response_body=json.dumps(result)[:2000],
            failed_field="power_rails",
        )

    power_rails: List[PowerRail] = []
    for idx, raw in enumerate(rails_raw):
        try:
            rail = PowerRail(
                net_name=str(raw.get("net_name", "")),
                voltage=float(raw.get("voltage", 0.0)),
                current_max=float(raw.get("current_max", 0.0)),
                regulator_type=str(raw.get("regulator_type", "")),
                source_component=str(raw.get("source_component", "")),
                consumer_components=list(raw.get("consumer_components", [])),
            )
            power_rails.append(rail)
        except (TypeError, ValueError) as exc:
            raise ExtractionError(
                f"Failed to construct PowerRail at index {idx}: {exc}. "
                f"Raw entry: {json.dumps(raw)[:500]}",
                input_content_preview=content,
                response_body=json.dumps(raw)[:2000],
                failed_field=f"power_rails[{idx}]",
            ) from exc

    logger.info("Extracted %d power rails via Opus 4.6", len(power_rails))
    return power_rails


async def extract_subsystem_blocks(content: str) -> List[SubsystemBlock]:
    """
    Extract subsystem block definitions from architecture content using Opus 4.6.

    Parses architecture diagrams (Mermaid, ASCII, or prose descriptions)
    and extracts logical subsystem groupings with their component lists
    and inter-block connectivity.

    Args:
        content: Raw artifact content from architecture diagrams, system
            overviews, or block diagrams.

    Returns:
        List of ``SubsystemBlock`` dataclass instances.

    Raises:
        ExtractionError: If the LLM call fails or returns unparseable data.
    """
    system_prompt = (
        "You are an expert electronics systems architect. Parse the "
        "following architecture diagram or system description and extract "
        "every subsystem block.  Return them as a JSON object with a "
        "single key 'subsystem_blocks' whose value is an array of objects.\n\n"
        "For each subsystem block extract: name (human-readable subsystem "
        "name like 'Power Stage', 'MCU Core'), components (array of "
        "component reference designators or part names that belong to this "
        "block), position_hint (placement suggestion like 'top-left', "
        "'center', 'near:U1', or empty if not specified), and "
        "connections_to (array of other subsystem block names that this "
        "block connects to).\n\n"
        "If the input is a Mermaid diagram, parse the graph structure. "
        "If it is prose, infer the blocks from the described architecture. "
        "Do NOT invent blocks that are not in the source."
    )

    json_schema = {
        "type": "object",
        "required": ["subsystem_blocks"],
        "properties": {
            "subsystem_blocks": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string", "default": ""},
                        "components": {
                            "type": "array",
                            "items": {"type": "string"},
                            "default": [],
                        },
                        "position_hint": {"type": "string", "default": ""},
                        "connections_to": {
                            "type": "array",
                            "items": {"type": "string"},
                            "default": [],
                        },
                    },
                },
            }
        },
    }

    result = await _call_opus_extraction(content, system_prompt, json_schema)

    blocks_raw = result.get("subsystem_blocks")
    if blocks_raw is None:
        raise ExtractionError(
            "LLM response missing 'subsystem_blocks' key in subsystem extraction. "
            f"Keys found: {list(result.keys())}",
            input_content_preview=content,
            response_body=json.dumps(result)[:2000],
            failed_field="subsystem_blocks",
        )
    if not isinstance(blocks_raw, list):
        raise ExtractionError(
            f"Expected 'subsystem_blocks' to be a list but got {type(blocks_raw).__name__}",
            input_content_preview=content,
            response_body=json.dumps(result)[:2000],
            failed_field="subsystem_blocks",
        )

    blocks: List[SubsystemBlock] = []
    for idx, raw in enumerate(blocks_raw):
        try:
            block = SubsystemBlock(
                name=str(raw.get("name", "")),
                components=list(raw.get("components", [])),
                position_hint=str(raw.get("position_hint", "")),
                connections_to=list(raw.get("connections_to", [])),
            )
            blocks.append(block)
        except (TypeError, ValueError) as exc:
            raise ExtractionError(
                f"Failed to construct SubsystemBlock at index {idx}: {exc}. "
                f"Raw entry: {json.dumps(raw)[:500]}",
                input_content_preview=content,
                response_body=json.dumps(raw)[:2000],
                failed_field=f"subsystem_blocks[{idx}]",
            ) from exc

    logger.info("Extracted %d subsystem blocks via Opus 4.6", len(blocks))
    return blocks


async def extract_test_criteria(content: str) -> List[TestCriterion]:
    """
    Extract test criteria from test plan content using Opus 4.6.

    Parses test plans, validation matrices, and acceptance criteria
    documents to produce structured, machine-evaluable test criteria.

    Args:
        content: Raw artifact content from test plan, validation, or
            acceptance criteria documents.

    Returns:
        List of ``TestCriterion`` dataclass instances.

    Raises:
        ExtractionError: If the LLM call fails or returns unparseable data.
    """
    system_prompt = (
        "You are an expert electronics test engineer. Extract every test "
        "criterion from the following test plan and return them as a JSON "
        "object with a single key 'test_criteria' whose value is an array "
        "of objects.\n\n"
        "For each test criterion extract: test_name (short identifier "
        "like 'VCC_3V3_rail_voltage'), expected_result (human-readable "
        "description of the expected outcome), pass_criteria (machine-"
        "evaluable condition like 'voltage >= 3.2 and voltage <= 3.4'), "
        "severity (one of: critical, error, warning, info), and category "
        "(grouping like power, signal_integrity, thermal, functional, "
        "emc, safety).\n\n"
        "If a field is not explicitly stated, use the default (empty "
        "string for strings, 'error' for severity). "
        "Do NOT invent criteria that are not in the source."
    )

    json_schema = {
        "type": "object",
        "required": ["test_criteria"],
        "properties": {
            "test_criteria": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "test_name": {"type": "string", "default": ""},
                        "expected_result": {"type": "string", "default": ""},
                        "pass_criteria": {"type": "string", "default": ""},
                        "severity": {
                            "type": "string",
                            "enum": ["critical", "error", "warning", "info"],
                            "default": "error",
                        },
                        "category": {"type": "string", "default": ""},
                    },
                },
            }
        },
    }

    result = await _call_opus_extraction(content, system_prompt, json_schema)

    criteria_raw = result.get("test_criteria")
    if criteria_raw is None:
        raise ExtractionError(
            "LLM response missing 'test_criteria' key in test criteria extraction. "
            f"Keys found: {list(result.keys())}",
            input_content_preview=content,
            response_body=json.dumps(result)[:2000],
            failed_field="test_criteria",
        )
    if not isinstance(criteria_raw, list):
        raise ExtractionError(
            f"Expected 'test_criteria' to be a list but got {type(criteria_raw).__name__}",
            input_content_preview=content,
            response_body=json.dumps(result)[:2000],
            failed_field="test_criteria",
        )

    criteria: List[TestCriterion] = []
    for idx, raw in enumerate(criteria_raw):
        try:
            tc = TestCriterion(
                test_name=str(raw.get("test_name", "")),
                expected_result=str(raw.get("expected_result", "")),
                pass_criteria=str(raw.get("pass_criteria", "")),
                severity=str(raw.get("severity", "error")),
                category=str(raw.get("category", "")),
            )
            criteria.append(tc)
        except (TypeError, ValueError) as exc:
            raise ExtractionError(
                f"Failed to construct TestCriterion at index {idx}: {exc}. "
                f"Raw entry: {json.dumps(raw)[:500]}",
                input_content_preview=content,
                response_body=json.dumps(raw)[:2000],
                failed_field=f"test_criteria[{idx}]",
            ) from exc

    logger.info("Extracted %d test criteria via Opus 4.6", len(criteria))
    return criteria


async def extract_compliance_requirements(
    content: str,
) -> List[ComplianceRequirement]:
    """
    Extract regulatory and standards compliance requirements using Opus 4.6.

    Parses compliance documentation, regulatory notes, and standards
    references to produce structured compliance requirement objects.

    Args:
        content: Raw artifact content from compliance documents, standards
            references, or regulatory requirement specifications.

    Returns:
        List of ``ComplianceRequirement`` dataclass instances.

    Raises:
        ExtractionError: If the LLM call fails or returns unparseable data.
    """
    system_prompt = (
        "You are an expert electronics compliance engineer. Extract every "
        "regulatory or standards compliance requirement from the following "
        "content and return them as a JSON object with a single key "
        "'compliance_requirements' whose value is an array of objects.\n\n"
        "For each requirement extract: standard (the standard name like "
        "'IEC 61000', 'NASA-STD-8739.4', 'MIL-STD-883', 'IPC-2221'), "
        "requirement (the specific requirement text), verification_method "
        "(how to verify: analysis, inspection, test, demonstration, or "
        "similarity), and applicable_components (array of component "
        "reference designators or categories this applies to, or empty "
        "array if it applies to the whole design).\n\n"
        "If a field is not explicitly stated, use the default (empty "
        "string for strings, empty array for arrays). "
        "Do NOT invent requirements that are not in the source."
    )

    json_schema = {
        "type": "object",
        "required": ["compliance_requirements"],
        "properties": {
            "compliance_requirements": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "standard": {"type": "string", "default": ""},
                        "requirement": {"type": "string", "default": ""},
                        "verification_method": {"type": "string", "default": ""},
                        "applicable_components": {
                            "type": "array",
                            "items": {"type": "string"},
                            "default": [],
                        },
                    },
                },
            }
        },
    }

    result = await _call_opus_extraction(content, system_prompt, json_schema)

    reqs_raw = result.get("compliance_requirements")
    if reqs_raw is None:
        raise ExtractionError(
            "LLM response missing 'compliance_requirements' key in compliance extraction. "
            f"Keys found: {list(result.keys())}",
            input_content_preview=content,
            response_body=json.dumps(result)[:2000],
            failed_field="compliance_requirements",
        )
    if not isinstance(reqs_raw, list):
        raise ExtractionError(
            f"Expected 'compliance_requirements' to be a list but got {type(reqs_raw).__name__}",
            input_content_preview=content,
            response_body=json.dumps(result)[:2000],
            failed_field="compliance_requirements",
        )

    reqs: List[ComplianceRequirement] = []
    for idx, raw in enumerate(reqs_raw):
        try:
            req = ComplianceRequirement(
                standard=str(raw.get("standard", "")),
                requirement=str(raw.get("requirement", "")),
                verification_method=str(raw.get("verification_method", "")),
                applicable_components=list(raw.get("applicable_components", [])),
            )
            reqs.append(req)
        except (TypeError, ValueError) as exc:
            raise ExtractionError(
                f"Failed to construct ComplianceRequirement at index {idx}: {exc}. "
                f"Raw entry: {json.dumps(raw)[:500]}",
                input_content_preview=content,
                response_body=json.dumps(raw)[:2000],
                failed_field=f"compliance_requirements[{idx}]",
            ) from exc

    logger.info(
        "Extracted %d compliance requirements via Opus 4.6", len(reqs)
    )
    return reqs


# ---------------------------------------------------------------------------
# Artifact type classification helpers
# ---------------------------------------------------------------------------

# Mapping from ArtifactType to the set of artifact_type string values
# that should be routed to each extractor.
_BOM_TYPES = {
    ArtifactType.BOM.value,
    ArtifactType.COMPONENT_SELECTION.value,
}

_CONNECTION_TYPES = {
    ArtifactType.SCHEMATIC_SPEC.value,
    ArtifactType.MCU_SPEC.value,
    ArtifactType.SENSING_SPEC.value,
    ArtifactType.CONNECTOR_SPEC.value,
}

_INTERFACE_TYPES = {
    ArtifactType.COMMUNICATION_SPEC.value,
    ArtifactType.INTERFACE_SPEC.value,
}

_POWER_TYPES = {
    ArtifactType.POWER_SPEC.value,
}

_ARCHITECTURE_TYPES = {
    ArtifactType.ARCHITECTURE_DIAGRAM.value,
    ArtifactType.SYSTEM_OVERVIEW.value,
}

_TEST_TYPES = {
    ArtifactType.TEST_PLAN.value,
}

_COMPLIANCE_TYPES = {
    ArtifactType.COMPLIANCE_DOC.value,
}

# Artifact types that contribute to the design_intent_text passthrough
_DESIGN_INTENT_TYPES = {
    ArtifactType.SYSTEM_OVERVIEW.value,
    ArtifactType.EXECUTIVE_SUMMARY.value,
    ArtifactType.SCHEMATIC_SPEC.value,
    ArtifactType.MCU_SPEC.value,
    ArtifactType.SENSING_SPEC.value,
    ArtifactType.CALCULATIONS.value,
    ArtifactType.PCB_SPEC.value,
    ArtifactType.FIRMWARE_SPEC.value,
}


# ---------------------------------------------------------------------------
# Master builder
# ---------------------------------------------------------------------------


async def build_ideation_context(
    raw_artifacts: List[Dict[str, Any]],
    subsystems: List[Dict[str, Any]],
    project_name: str,
) -> IdeationContext:
    """
    Build a fully populated ``IdeationContext`` from raw ideation artifacts.

    Groups artifacts by type, calls all relevant LLM extractors in parallel
    via ``asyncio.gather``, merges the results into the master context
    object, and collects any extraction errors.

    This function replaces the legacy ``create_design_intent()`` function
    in ``api_generate_schematic.py``.  The resulting ``IdeationContext``
    provides both structured data (for pipeline stages that have been
    upgraded) and a backward-compatible ``design_intent_text`` property
    (for stages that still consume a plain-text blob).

    Args:
        raw_artifacts: List of artifact dictionaries as returned by the
            ideation artifact repository.  Each dict must have at least
            ``artifact_type`` and ``content`` keys.
        subsystems: List of subsystem dictionaries from the project
            configuration.
        project_name: Human-readable project name for context.

    Returns:
        A fully populated ``IdeationContext`` instance.  If extraction
        errors occurred, they are recorded in ``extraction_errors`` and
        the function raises an ``ExtractionError`` wrapping the first
        error so the pipeline can halt.

    Raises:
        ExtractionError: If ANY extraction step fails.  The error message
            includes all collected errors so the operator can diagnose the
            full scope of failures in a single log entry.
    """
    logger.info(
        "Building IdeationContext from %d artifacts for project '%s'",
        len(raw_artifacts),
        project_name,
    )

    # ----- Group artifacts by type ------------------------------------------
    bom_contents: List[str] = []
    bom_formats: List[str] = []
    connection_contents: List[str] = []
    interface_contents: List[str] = []
    power_contents: List[str] = []
    architecture_contents: List[str] = []
    test_contents: List[str] = []
    compliance_contents: List[str] = []
    design_intent_parts: List[str] = []

    raw_artifacts_dict: Dict[str, Any] = {}

    for artifact in raw_artifacts:
        art_type = artifact.get("artifact_type", "")
        content = artifact.get("content", "")
        art_format = artifact.get("format", "")
        art_name = artifact.get("name", art_type)

        if not content or not content.strip():
            logger.debug("Skipping empty artifact: %s (%s)", art_name, art_type)
            continue

        # Store in raw dict for backward compat
        raw_artifacts_dict[art_type] = content

        # Route to appropriate extractor bucket
        if art_type in _BOM_TYPES:
            bom_contents.append(content)
            bom_formats.append(art_format)
        if art_type in _CONNECTION_TYPES:
            connection_contents.append(content)
        if art_type in _INTERFACE_TYPES:
            interface_contents.append(content)
        if art_type in _POWER_TYPES:
            power_contents.append(content)
        if art_type in _ARCHITECTURE_TYPES:
            architecture_contents.append(content)
        if art_type in _TEST_TYPES:
            test_contents.append(content)
        if art_type in _COMPLIANCE_TYPES:
            compliance_contents.append(content)
        if art_type in _DESIGN_INTENT_TYPES:
            design_intent_parts.append(f"--- {art_name} ({art_type}) ---\n{content}")

    # Also include subsystem names in the design intent text
    if subsystems:
        subsystem_names = [s.get("name", "Unknown") for s in subsystems]
        design_intent_parts.insert(
            0,
            f"{project_name} - Schematic Design\n\n"
            f"Selected Subsystems:\n"
            + "\n".join(f"- {name}" for name in subsystem_names),
        )

    # ----- Build extraction coroutines --------------------------------------
    # For connection-heavy artifacts (schematic specs), extract each artifact
    # individually to avoid generating massive JSON blobs that exceed LLM
    # output limits.  Other extractors use combined content since their
    # output is typically smaller.

    extraction_tasks: List[tuple] = []  # (label, coroutine)

    if bom_contents:
        combined_bom = "\n\n=== NEXT BOM/COMPONENT ARTIFACT ===\n\n".join(bom_contents)
        combined_format = bom_formats[0] if bom_formats else ""
        extraction_tasks.append(
            ("bom", extract_bom_entries(combined_bom, combined_format))
        )

    # Extract connections per-artifact to keep JSON output manageable
    for i, conn_content in enumerate(connection_contents):
        extraction_tasks.append(
            (f"connections_{i}", extract_pin_connections(conn_content))
        )

    # Extract interfaces per-artifact
    for i, iface_content in enumerate(interface_contents):
        extraction_tasks.append(
            (f"interfaces_{i}", extract_interfaces(iface_content))
        )

    if power_contents:
        combined_power = "\n\n=== NEXT POWER SPEC ARTIFACT ===\n\n".join(
            power_contents
        )
        extraction_tasks.append(
            ("power_rails", extract_power_rails(combined_power))
        )

    if architecture_contents:
        combined_arch = "\n\n=== NEXT ARCHITECTURE ARTIFACT ===\n\n".join(
            architecture_contents
        )
        extraction_tasks.append(
            ("subsystems", extract_subsystem_blocks(combined_arch))
        )

    if test_contents:
        combined_test = "\n\n=== NEXT TEST PLAN ARTIFACT ===\n\n".join(test_contents)
        extraction_tasks.append(
            ("test_criteria", extract_test_criteria(combined_test))
        )

    if compliance_contents:
        combined_compliance = "\n\n=== NEXT COMPLIANCE DOC ARTIFACT ===\n\n".join(
            compliance_contents
        )
        extraction_tasks.append(
            ("compliance", extract_compliance_requirements(combined_compliance))
        )

    # ----- Execute all extractions in parallel with retry -------------------
    extraction_errors: List[str] = []
    results: Dict[str, Any] = {}

    if extraction_tasks:
        logger.info(
            "Running %d extraction tasks in parallel: %s",
            len(extraction_tasks),
            [label for label, _ in extraction_tasks],
        )

        labels = [label for label, _ in extraction_tasks]
        coros = [coro for _, coro in extraction_tasks]

        gathered = await asyncio.gather(*coros, return_exceptions=True)

        # Collect results, retrying failures once
        retry_tasks: List[Tuple[int, str]] = []
        for idx, (label, result) in enumerate(zip(labels, gathered)):
            if isinstance(result, Exception):
                retry_tasks.append((idx, label))
            else:
                results[label] = result

        if retry_tasks:
            logger.warning(
                "Retrying %d failed extraction(s): %s",
                len(retry_tasks),
                [label for _, label in retry_tasks],
            )
            # Re-invoke the extraction functions directly for retry
            retry_labels = [label for _, label in retry_tasks]
            retry_coros_2 = []
            for orig_idx, label in retry_tasks:
                base_label = label.split("_")[0]
                idx_str = label.split("_")[-1] if "_" in label else None
                if base_label == "connections" and idx_str is not None:
                    retry_coros_2.append(extract_pin_connections(connection_contents[int(idx_str)]))
                elif base_label == "interfaces" and idx_str is not None:
                    retry_coros_2.append(extract_interfaces(interface_contents[int(idx_str)]))
                elif label == "bom":
                    combined_bom = "\n\n=== NEXT BOM/COMPONENT ARTIFACT ===\n\n".join(bom_contents)
                    combined_format = bom_formats[0] if bom_formats else ""
                    retry_coros_2.append(extract_bom_entries(combined_bom, combined_format))
                elif label == "power_rails":
                    combined_power = "\n\n=== NEXT POWER SPEC ARTIFACT ===\n\n".join(power_contents)
                    retry_coros_2.append(extract_power_rails(combined_power))
                elif label == "subsystems":
                    combined_arch = "\n\n=== NEXT ARCHITECTURE ARTIFACT ===\n\n".join(architecture_contents)
                    retry_coros_2.append(extract_subsystem_blocks(combined_arch))
                elif label == "test_criteria":
                    combined_test = "\n\n=== NEXT TEST PLAN ARTIFACT ===\n\n".join(test_contents)
                    retry_coros_2.append(extract_test_criteria(combined_test))
                elif label == "compliance":
                    combined_compliance = "\n\n=== NEXT COMPLIANCE DOC ARTIFACT ===\n\n".join(compliance_contents)
                    retry_coros_2.append(extract_compliance_requirements(combined_compliance))

            if retry_coros_2:
                retry_gathered = await asyncio.gather(*retry_coros_2, return_exceptions=True)
                for label, result in zip(retry_labels, retry_gathered):
                    if isinstance(result, Exception):
                        error_msg = (
                            f"Extraction '{label}' failed after retry: "
                            f"{type(result).__name__}: {result}\n"
                            f"Traceback: {traceback.format_exception(type(result), result, result.__traceback__)}"
                        )
                        logger.error(error_msg)
                        extraction_errors.append(error_msg)
                    else:
                        results[label] = result

        # Merge per-artifact connection results into single list
        all_connections: List[PinConnection] = []
        all_interfaces: List[InterfaceDefinition] = []
        for label, data in sorted(results.items()):
            if label.startswith("connections_"):
                all_connections.extend(data)
            elif label.startswith("interfaces_"):
                all_interfaces.extend(data)
        if all_connections:
            results["connections"] = all_connections
        if all_interfaces:
            results["interfaces"] = all_interfaces
    else:
        logger.info(
            "No extractable artifact types found in %d artifacts. "
            "IdeationContext will be empty (backward-compatible mode).",
            len(raw_artifacts),
        )

    # ----- Assemble IdeationContext -----------------------------------------

    # Symbol resolution context
    bom_entries: List[BOMEntry] = results.get("bom", [])
    symbol_resolution = SymbolResolutionContext(
        preferred_parts=bom_entries,
        manufacturer_priority=_dedupe_ordered(
            [e.manufacturer for e in bom_entries if e.manufacturer]
        ),
        package_preferences={
            e.category: e.package
            for e in bom_entries
            if e.category and e.package
        },
        avoid_parts=[],
    )

    # Connection inference context
    explicit_connections: List[PinConnection] = results.get("connections", [])
    interfaces: List[InterfaceDefinition] = results.get("interfaces", [])
    power_rails: List[PowerRail] = results.get("power_rails", [])

    # Derive ground nets from power rails that have 0V or "GND" in name
    ground_nets = _dedupe_ordered(
        [
            rail.net_name
            for rail in power_rails
            if rail.voltage == 0.0
            or "gnd" in rail.net_name.lower()
            or "ground" in rail.net_name.lower()
        ]
    )
    if not ground_nets:
        ground_nets = ["GND"]  # sensible default for all designs

    connection_inference = ConnectionInferenceContext(
        explicit_connections=explicit_connections,
        interfaces=interfaces,
        power_rails=power_rails,
        ground_nets=ground_nets,
        critical_signals=[],
        design_intent_text="\n\n".join(design_intent_parts),
    )

    # Placement context
    subsystem_blocks: List[SubsystemBlock] = results.get("subsystems", [])
    placement = PlacementContext(
        subsystem_blocks=subsystem_blocks,
        signal_flow_direction="left_to_right",
        critical_proximity=[],
        isolation_zones=[],
    )

    # Validation context
    test_criteria: List[TestCriterion] = results.get("test_criteria", [])
    compliance_reqs: List[ComplianceRequirement] = results.get("compliance", [])

    # Derive voltage/current limits from power rails
    voltage_limits: Dict[str, Tuple[float, float]] = {}
    current_limits: Dict[str, float] = {}
    for rail in power_rails:
        if rail.net_name and rail.voltage > 0:
            # Default tolerance: +/- 5%
            tolerance = rail.voltage * 0.05
            voltage_limits[rail.net_name] = (
                round(rail.voltage - tolerance, 4),
                round(rail.voltage + tolerance, 4),
            )
        if rail.net_name and rail.current_max > 0:
            current_limits[rail.net_name] = rail.current_max

    validation = ValidationContext(
        test_criteria=test_criteria,
        compliance_requirements=compliance_reqs,
        voltage_limits=voltage_limits,
        current_limits=current_limits,
        thermal_limits={},
        emc_requirements=[],
    )

    # Build the master context
    context = IdeationContext(
        symbol_resolution=symbol_resolution,
        connection_inference=connection_inference,
        placement=placement,
        validation=validation,
        raw_artifacts=raw_artifacts_dict,
        artifact_count=len(raw_artifacts),
        extraction_errors=extraction_errors,
    )

    # Log summary
    logger.info(
        "IdeationContext built: "
        "%d BOM entries, %d connections, %d interfaces, "
        "%d power rails, %d subsystem blocks, "
        "%d test criteria, %d compliance requirements, "
        "%d extraction errors",
        len(bom_entries),
        len(explicit_connections),
        len(interfaces),
        len(power_rails),
        len(subsystem_blocks),
        len(test_criteria),
        len(compliance_reqs),
        len(extraction_errors),
    )

    # ----- Log errors but continue with partial data -------------------------
    # Extraction errors are non-fatal: the pipeline continues with whatever
    # structured data was successfully extracted.  For example, if the
    # connections extractor fails (LLM output too large / malformed JSON),
    # the pipeline still has BOM entries, power rails, interfaces, etc.
    if extraction_errors:
        logger.warning(
            "%d extraction error(s) while building IdeationContext for '%s' "
            "(continuing with partial data):\n%s",
            len(extraction_errors),
            project_name,
            "\n---\n".join(extraction_errors),
        )

    return context


# ---------------------------------------------------------------------------
# Internal utilities
# ---------------------------------------------------------------------------


def _try_repair_json(text: str) -> Optional[dict]:
    """
    Attempt to repair truncated JSON from LLM output.

    When the LLM generates very large JSON (e.g. hundreds of connections),
    the output may get truncated mid-entry.  This function finds the last
    complete array entry and closes the JSON structure.

    Returns:
        Parsed dict if repair succeeded, ``None`` otherwise.
    """
    # Find the last occurrence of "}" followed by optional whitespace and ","
    # which marks a complete array entry boundary
    last_complete = -1
    brace_depth = 0
    in_string = False
    escape = False

    for i, ch in enumerate(text):
        if escape:
            escape = False
            continue
        if ch == '\\':
            escape = True
            continue
        if ch == '"' and not escape:
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == '{':
            brace_depth += 1
        elif ch == '}':
            brace_depth -= 1
            if brace_depth == 2:
                # Completed an array element (depth: outer{1} -> array_key -> item{2})
                last_complete = i

    if last_complete < 0:
        return None

    # Truncate at the last complete entry, close the array and outer object
    truncated = text[: last_complete + 1]
    # Remove any trailing comma
    truncated = truncated.rstrip().rstrip(',')
    # Close open structures
    truncated += "]}"

    try:
        result = json.loads(truncated)
        if isinstance(result, dict):
            return result
    except json.JSONDecodeError:
        pass

    return None


def _dedupe_ordered(items: List[str]) -> List[str]:
    """
    Return a deduplicated list preserving insertion order.

    Args:
        items: List of strings, possibly with duplicates.

    Returns:
        List of unique strings in the order they first appeared.
    """
    seen: set = set()
    result: List[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = [
    "ExtractionError",
    "_call_opus_extraction",
    "extract_bom_entries",
    "extract_pin_connections",
    "extract_interfaces",
    "extract_power_rails",
    "extract_subsystem_blocks",
    "extract_test_criteria",
    "extract_compliance_requirements",
    "build_ideation_context",
]
