"""
Symbol Fetcher Agent - Multi-source KiCad symbol retrieval.

Fetches KiCad symbols from multiple sources with fallback chain:
1. Local cache (/opt/kicad-symbols/)
2. KiCad Official Libraries (GitLab)
3. SnapEDA API
4. Ultra Librarian API
5. Manufacturer libraries (TI, STM, Infineon, etc.)
6. LLM-generated symbols (fallback)

Author: Nexus EE Design Team
"""

import asyncio
import hashlib
import json
import logging
import os
import platform
import re
import sys
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import quote_plus

try:
    import httpx
except ImportError:
    httpx = None

try:
    import anthropic
except ImportError:
    anthropic = None

logger = logging.getLogger(__name__)


class SymbolResolutionError(Exception):
    """
    Raised when symbol resolution fails for a component.

    This exception provides detailed information about why resolution failed,
    which sources were tried, and what the user can do to fix it.
    """

    def __init__(
        self,
        part_number: str,
        category: str,
        sources_tried: List[str],
        errors: List[str],
        suggestion: Optional[str] = None
    ):
        self.part_number = part_number
        self.category = category
        self.sources_tried = sources_tried
        self.errors = errors
        self.suggestion = suggestion

        # Build detailed error message
        error_details = "\n".join([f"  - {e}" for e in errors]) if errors else "No specific errors logged"
        sources_list = ", ".join(sources_tried) if sources_tried else "none"

        message = (
            f"Failed to resolve symbol for '{part_number}' (category: {category}).\n"
            f"Sources tried: {sources_list}\n"
            f"Errors:\n{error_details}"
        )

        if suggestion:
            message += f"\nSuggestion: {suggestion}"

        super().__init__(message)


class SymbolSource(Enum):
    """Available symbol sources in priority order."""
    LOCAL_CACHE = "local_cache"
    KICAD_LOCAL_INSTALL = "kicad_local_install"  # Local KiCad installation
    KICAD_WORKER_INTERNAL = "kicad_worker_internal"  # Internal K8s KiCad worker
    KICAD_OFFICIAL = "kicad_official"
    SNAPEDA = "snapeda"
    ULTRA_LIBRARIAN = "ultralibrarian"
    TI_WEBENCH = "ti_webench"
    STM_CUBE = "stm_cube"
    INFINEON = "infineon"
    ANALOG_DEVICES = "analog_devices"
    LLM_GENERATED = "llm_generated"


# Internal KiCad worker URL (for K8s deployment)
KICAD_WORKER_URL = os.environ.get(
    'KICAD_WORKER_URL',
    'http://mapos-kicad-worker:8080'
)


# Platform-specific KiCad installation paths
KICAD_INSTALL_PATHS = {
    "darwin": [  # macOS
        Path("/Applications/KiCad/KiCad.app/Contents/SharedSupport/symbols"),
        Path.home() / "Applications/KiCad/KiCad.app/Contents/SharedSupport/symbols",
    ],
    "linux": [  # Linux
        Path("/usr/share/kicad/symbols"),
        Path("/usr/local/share/kicad/symbols"),
        Path.home() / ".local/share/kicad/symbols",
    ],
    "win32": [  # Windows
        Path("C:/Program Files/KiCad/share/kicad/symbols"),
        Path("C:/Program Files (x86)/KiCad/share/kicad/symbols"),
    ],
}


# Category-appropriate pin templates for placeholder symbols
# Each entry: (pin_name, pin_number, pin_type_str, side)
# side: 'L'=left(input), 'R'=right(output), 'T'=top(power), 'B'=bottom(ground)
CATEGORY_PIN_TEMPLATES = {
    "MCU": [
        ("VCC", "1", "power_in", "T"), ("VDD", "2", "power_in", "T"),
        ("GND", "3", "power_in", "B"), ("VSS", "4", "power_in", "B"),
        ("NRST", "5", "input", "L"), ("BOOT0", "6", "input", "L"),
        ("OSC_IN", "7", "input", "L"), ("OSC_OUT", "8", "output", "R"),
        ("PA0", "9", "bidirectional", "L"), ("PA1", "10", "bidirectional", "L"),
        ("PA2", "11", "bidirectional", "L"), ("PA3", "12", "bidirectional", "L"),
        ("PA4", "13", "bidirectional", "L"), ("PA5", "14", "bidirectional", "R"),
        ("PA6", "15", "bidirectional", "R"), ("PA7", "16", "bidirectional", "R"),
        ("PB0", "17", "bidirectional", "R"), ("PB1", "18", "bidirectional", "R"),
        ("PB2", "19", "bidirectional", "R"), ("PB3", "20", "bidirectional", "R"),
        ("PB4", "21", "bidirectional", "R"), ("PB5", "22", "bidirectional", "R"),
        ("PB6", "23", "bidirectional", "R"), ("PB7", "24", "bidirectional", "R"),
        ("SWD_IO", "25", "bidirectional", "L"), ("SWD_CLK", "26", "input", "L"),
        ("VBAT", "27", "power_in", "T"), ("VSSA", "28", "power_in", "B"),
        ("VDDA", "29", "power_in", "T"), ("VREF", "30", "passive", "T"),
    ],
    "Gate_Driver": [
        ("VCC", "1", "power_in", "T"), ("GND", "2", "power_in", "B"),
        ("INH", "3", "input", "L"), ("INL", "4", "input", "L"),
        ("EN", "5", "input", "L"), ("HO", "6", "output", "R"),
        ("LO", "7", "output", "R"), ("VB", "8", "power_in", "T"),
        ("VS", "9", "power_in", "R"), ("DT", "10", "input", "L"),
    ],
    "MOSFET": [
        ("G", "1", "input", "L"), ("D", "2", "passive", "T"),
        ("S", "3", "passive", "B"),
    ],
    "CAN_Transceiver": [
        ("VCC", "1", "power_in", "T"), ("GND", "2", "power_in", "B"),
        ("TXD", "3", "input", "L"), ("RXD", "4", "output", "R"),
        ("CANH", "5", "bidirectional", "R"), ("CANL", "6", "bidirectional", "R"),
        ("S", "7", "input", "L"), ("VREF", "8", "passive", "R"),
    ],
    "OpAmp": [
        ("V+", "1", "power_in", "T"), ("V-", "2", "power_in", "B"),
        ("IN+", "3", "input", "L"), ("IN-", "4", "input", "L"),
        ("OUT", "5", "output", "R"),
    ],
    "Regulator": [
        ("VIN", "1", "power_in", "L"), ("GND", "2", "power_in", "B"),
        ("VOUT", "3", "power_out", "R"), ("EN", "4", "input", "L"),
        ("FB", "5", "input", "R"),
    ],
    "Capacitor": [("1", "1", "passive", "L"), ("2", "2", "passive", "R")],
    "Resistor": [("1", "1", "passive", "L"), ("2", "2", "passive", "R")],
    "Inductor": [("1", "1", "passive", "L"), ("2", "2", "passive", "R")],
    "Diode": [("A", "1", "passive", "L"), ("K", "2", "passive", "R")],
    "LED": [("A", "1", "passive", "L"), ("K", "2", "passive", "R")],
    "Crystal": [("1", "1", "passive", "L"), ("2", "2", "passive", "R")],
    "Fuse": [("1", "1", "passive", "L"), ("2", "2", "passive", "R")],
    "Connector": [
        ("1", "1", "passive", "L"), ("2", "2", "passive", "L"),
        ("3", "3", "passive", "L"), ("4", "4", "passive", "L"),
    ],
    "BJT": [("B", "1", "input", "L"), ("C", "2", "passive", "T"), ("E", "3", "passive", "B")],
    "Current_Sense": [
        ("VCC", "1", "power_in", "T"), ("GND", "2", "power_in", "B"),
        ("IN+", "3", "input", "L"), ("IN-", "4", "input", "L"),
        ("OUT", "5", "output", "R"), ("REF", "6", "input", "L"),
    ],
    "ESD_Protection": [
        ("IO1", "1", "bidirectional", "L"), ("IO2", "2", "bidirectional", "L"),
        ("GND", "3", "power_in", "B"),
    ],
    "Amplifier": [
        ("VCC", "1", "power_in", "T"), ("GND", "2", "power_in", "B"),
        ("IN+", "3", "input", "L"), ("IN-", "4", "input", "L"),
        ("OUT", "5", "output", "R"), ("REF", "6", "input", "L"),
    ],
    "IC": [
        ("VIN", "1", "power_in", "L"), ("VOUT", "2", "power_out", "R"),
        ("GND", "3", "power_in", "B"), ("CTL", "4", "input", "L"),
        ("STAT", "5", "output", "R"), ("SENSE", "6", "input", "L"),
    ],
    "TVS": [
        ("IO1", "1", "bidirectional", "L"), ("IO2", "2", "bidirectional", "L"),
        ("GND", "3", "power_in", "B"),
    ],
    "Thermistor": [("1", "1", "passive", "L"), ("2", "2", "passive", "R")],
    "USB_Connector": [
        ("VBUS", "1", "power_in", "T"), ("GND", "2", "power_in", "B"),
        ("D+", "3", "bidirectional", "R"), ("D-", "4", "bidirectional", "R"),
        ("CC1", "5", "bidirectional", "L"), ("CC2", "6", "bidirectional", "L"),
        ("SBU1", "7", "bidirectional", "L"), ("SBU2", "8", "bidirectional", "L"),
        ("SHIELD", "9", "passive", "B"),
    ],
}


@dataclass
class SymbolSourceConfig:
    """Configuration for a symbol source."""
    source: SymbolSource
    priority: int
    api_url: Optional[str] = None
    local_path: Optional[Path] = None
    requires_auth: bool = False
    api_key_env: Optional[str] = None


@dataclass
class FetchedSymbol:
    """Result of symbol fetch operation."""
    part_number: str
    manufacturer: Optional[str]
    symbol_sexp: str  # KiCad S-expression format
    footprint_sexp: Optional[str] = None
    datasheet_url: Optional[str] = None
    source: SymbolSource = SymbolSource.LOCAL_CACHE
    metadata: Dict[str, Any] = field(default_factory=dict)
    fetched_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    needs_review: bool = False


@dataclass
class SymbolSearchResult:
    """Result from symbol search."""
    part_number: str
    manufacturer: str
    description: str
    category: str
    source: SymbolSource
    score: float = 1.0
    has_symbol: bool = True
    has_footprint: bool = False


# Detect platform and get KiCad installation paths
def _get_kicad_install_paths() -> List[Path]:
    """Get KiCad installation symbol paths for the current platform."""
    system = sys.platform
    paths = KICAD_INSTALL_PATHS.get(system, [])
    # Return only paths that exist
    return [p for p in paths if p.exists()]


# Default cache path - use project directory or user's home
DEFAULT_CACHE_PATH = Path(__file__).parent.parent.parent / "symbol_cache"

# Default source configuration with priorities
DEFAULT_SOURCES = [
    SymbolSourceConfig(
        source=SymbolSource.LOCAL_CACHE,
        priority=1,
        local_path=DEFAULT_CACHE_PATH
    ),
    SymbolSourceConfig(
        source=SymbolSource.KICAD_WORKER_INTERNAL,
        priority=2,  # Internal K8s KiCad worker (preferred in containerized environments)
        api_url=KICAD_WORKER_URL
    ),
    SymbolSourceConfig(
        source=SymbolSource.KICAD_LOCAL_INSTALL,
        priority=3,  # Local KiCad installation (macOS/Linux/Windows)
    ),
    SymbolSourceConfig(
        source=SymbolSource.KICAD_OFFICIAL,
        priority=4,
        api_url="https://gitlab.com/api/v4/projects/kicad%2Fkicad-symbols"
    ),
    SymbolSourceConfig(
        source=SymbolSource.SNAPEDA,
        priority=5,
        api_url="https://www.snapeda.com/api/v1",
        requires_auth=False,  # Basic search works without auth
        api_key_env="SNAPEDA_API_KEY"  # Optional: for higher rate limits
    ),
    SymbolSourceConfig(
        source=SymbolSource.ULTRA_LIBRARIAN,
        priority=6,
        api_url="https://app.ultralibrarian.com/api/v1",
        requires_auth=True,
        api_key_env="ULTRALIBRARIAN_API_KEY"
    ),
    SymbolSourceConfig(
        source=SymbolSource.TI_WEBENCH,
        priority=7,
        api_url="https://www.ti.com/lit/ds"
    ),
    SymbolSourceConfig(
        source=SymbolSource.STM_CUBE,
        priority=7,
        api_url="https://www.st.com"
    ),
    SymbolSourceConfig(
        source=SymbolSource.LLM_GENERATED,
        priority=99  # Last resort
    ),
]


class SymbolFetcherAgent:
    """
    Multi-source symbol fetcher with fallback chain.

    Fetches KiCad symbols from multiple sources, caches them locally,
    and indexes them in GraphRAG for semantic search.
    """

    def __init__(
        self,
        cache_path: Optional[Path] = None,
        graphrag_client: Optional[Any] = None,
        sources: Optional[List[SymbolSourceConfig]] = None,
        anthropic_api_key: Optional[str] = None
    ):
        """
        Initialize the symbol fetcher.

        Args:
            cache_path: Local directory for symbol cache (defaults to project symbol_cache)
            graphrag_client: Optional GraphRAG client for indexing
            sources: Custom source configuration (defaults to DEFAULT_SOURCES)
            anthropic_api_key: API key for LLM symbol generation
        """
        self.cache_path = cache_path or DEFAULT_CACHE_PATH
        self.graphrag = graphrag_client
        self.sources = sources or DEFAULT_SOURCES
        self.anthropic_api_key = anthropic_api_key

        # Initialize HTTP client
        if httpx is None:
            raise ImportError("httpx package required. Install with: pip install httpx")
        self.http_client = httpx.AsyncClient(timeout=60.0, follow_redirects=True)

        # LLM configuration: Claude Code Max proxy or OpenRouter
        self._openrouter_model = "anthropic/claude-opus-4.6"  # User directive: Opus 4.6 ONLY
        if os.environ.get("AI_PROVIDER") == "claude_code_max":
            proxy_url = os.environ.get(
                "CLAUDE_CODE_PROXY_URL",
                "http://claude-code-proxy.nexus.svc.cluster.local:3100"
            )
            self._openrouter_base_url = f"{proxy_url}/v1"
            self._openrouter_api_key = "internal-proxy"
            logger.info(f"SymbolFetcher: Using Claude Code Max proxy at {proxy_url}")
        else:
            # OpenRouter configuration (fallback)
            # IMPORTANT: OpenRouter uses OpenAI-compatible API, NOT Anthropic API format
            self._openrouter_api_key = os.environ.get("OPENROUTER_API_KEY")
            self._openrouter_base_url = "https://openrouter.ai/api/v1"

        # Direct Anthropic API (fallback)
        self.anthropic_client = None
        anthropic_key = anthropic_api_key or os.environ.get("ANTHROPIC_API_KEY")
        if anthropic and anthropic_key and not self._openrouter_api_key:
            self.anthropic_client = anthropic.Anthropic(api_key=anthropic_key)

        if not self._openrouter_api_key and not self.anthropic_client:
            logger.warning("No LLM API key found (set OPENROUTER_API_KEY or ANTHROPIC_API_KEY)")

        # Ensure cache directory exists
        self.cache_path.mkdir(parents=True, exist_ok=True)

        # Create category subdirectories
        for category in ['MCU', 'MOSFET', 'Gate_Driver', 'OpAmp', 'Capacitor',
                         'Resistor', 'Inductor', 'Connector', 'Power', 'Other']:
            (self.cache_path / category).mkdir(exist_ok=True)

    async def fetch_symbol(
        self,
        part_number: str,
        manufacturer: Optional[str] = None,
        category: str = "Other",
        allow_placeholder: bool = True
    ) -> FetchedSymbol:
        """
        Fetch symbol using priority fallback chain with nexus-memory learning.

        Args:
            part_number: Component part number (e.g., "STM32G431CBT6")
            manufacturer: Optional manufacturer name
            category: Component category for organization
            allow_placeholder: If False, raises SymbolResolutionError instead of returning placeholder

        Returns:
            FetchedSymbol with KiCad S-expression data

        Raises:
            SymbolResolutionError: If all sources fail and allow_placeholder is False
        """
        logger.info(f"Fetching symbol: {part_number} (manufacturer: {manufacturer}, category: {category})")

        # STEP 0: Check nexus-memory for previously learned resolution
        memory_result = await self._recall_from_nexus_memory(part_number, manufacturer)
        if memory_result:
            logger.info(f"Found {part_number} in nexus-memory (source: {memory_result.get('source', 'learned')})")
            # Try to use the learned resolution
            learned_symbol = await self._fetch_from_learned_resolution(memory_result, part_number, manufacturer, category)
            if learned_symbol:
                # Update success count in memory
                await self._update_memory_success_count(part_number, manufacturer, memory_result)
                return learned_symbol
            else:
                logger.warning(f"Learned resolution for {part_number} failed, falling back to standard chain")

        # STEP 0b: Hardcoded symbol overrides for parts that LLM consistently fails
        # Maps part_number -> (library, symbol_name).
        # UCC21530 is pin-compatible with UCC21520 (same TI isolated gate driver family).
        KNOWN_SYMBOL_OVERRIDES: Dict[str, Tuple[str, str]] = {
            "UCC21530ADWRR": ("Driver_FET", "UCC21520DW"),
            "UCC21530DWR":   ("Driver_FET", "UCC21520DW"),
            "USB4110-GF-A":  ("Connector", "USB_C_Receptacle_USB2.0_16P"),
            # SL22_10010: Murata NTC thermistor — LLM matches to 'L' (inductor), override to correct symbol
            "SL22_10010":    ("Device", "Thermistor_NTC"),
            # 0603WAF1200T5E: UniOhm 0603 single 120Ω resistor — LLM matches to R_Pack02 (dual), override to R
            "0603WAF1200T5E": ("Device", "R"),
            # LTC4412ES6: Linear Tech ideal diode controller — cache corrupts to 'L' in run 2, pin it to correct symbol
            "LTC4412ES6":    ("Power_Management", "LTC4412xS6"),
        }
        if part_number in KNOWN_SYMBOL_OVERRIDES:
            lib_name, sym_name = KNOWN_SYMBOL_OVERRIDES[part_number]
            logger.info(f"Using hardcoded override: {part_number} -> {sym_name} in {lib_name}")
            try:
                url = f"{KICAD_WORKER_URL}/v1/symbols/{lib_name}/symbol/{sym_name}"
                resp = await self.http_client.get(url, timeout=10.0)
                if resp.status_code == 200:
                    data = resp.json()
                    # Individual symbol endpoint returns 'sexp', library endpoint returns 'content'
                    kicad_sym = data.get("sexp", "") or data.get("content", "")
                    if kicad_sym:
                        result = FetchedSymbol(
                            part_number=part_number,
                            manufacturer=manufacturer,
                            symbol_sexp=kicad_sym,
                            source=SymbolSource.KICAD_WORKER_INTERNAL,
                            metadata={"override": True, "override_symbol": sym_name, "override_library": lib_name},
                        )
                        await self._cache_symbol(result, category)
                        return result
            except Exception as e:
                logger.warning(f"Override fetch failed for {part_number}: {e}")

        # Track sources tried and errors for verbose error reporting
        sources_tried: List[str] = []
        errors: List[str] = []

        # Sort sources by priority, excluding LLM_GENERATED (handled separately as last resort)
        sorted_sources = sorted(
            [s for s in self.sources if s.source != SymbolSource.LLM_GENERATED],
            key=lambda s: s.priority
        )

        for source_config in sorted_sources:
            source_name = source_config.source.value
            sources_tried.append(source_name)
            try:
                result = await self._fetch_from_source(
                    source_config, part_number, manufacturer, category
                )
                if result:
                    logger.info(f"Found symbol from {source_name}")

                    # Cache the symbol
                    await self._cache_symbol(result, category)

                    # Index in GraphRAG if available
                    if self.graphrag:
                        await self._index_in_graphrag(result, category)

                    # Store successful resolution in nexus-memory for future learning
                    await self._store_in_nexus_memory(
                        part_number, manufacturer, category, source_name, result
                    )

                    return result
                else:
                    errors.append(f"{source_name}: No matching symbol found")

            except Exception as e:
                error_msg = f"{source_name}: {str(e)}"
                errors.append(error_msg)
                logger.warning(f"Source {source_name} failed for {part_number}: {e}")
                continue

        # For passives, try generic symbol before LLM generation
        # Generic passive symbols (R, C, L) are acceptable quality
        generic_symbol = self.GENERIC_PASSIVE_SYMBOLS.get(category)
        if generic_symbol:
            sources_tried.append("generic_passive")
            try:
                result = await self._fetch_generic_passive(generic_symbol, part_number, category)
                if result:
                    logger.info(f"Using generic symbol {generic_symbol} for {part_number}")
                    await self._cache_symbol(result, category)
                    return result
                else:
                    errors.append("generic_passive: Device library not available")
            except Exception as e:
                errors.append(f"generic_passive: {str(e)}")

        # For active devices (MOSFETs, BJTs), try generic symbol before LLM generation
        generic_active = self.GENERIC_ACTIVE_SYMBOLS.get(category)
        if generic_active and not generic_symbol:
            sources_tried.append("generic_active")
            try:
                result = await self._fetch_generic_passive(generic_active, part_number, category)
                if result:
                    logger.info(f"Using generic active symbol {generic_active} for {part_number}")
                    await self._cache_symbol(result, category)
                    return result
                else:
                    errors.append("generic_active: Device library not available")
            except Exception as e:
                errors.append(f"generic_active: {str(e)}")

        # Search online for part information (Octopart, SnapEDA, manufacturer sites)
        sources_tried.append("online_search")
        try:
            online_result = await self._search_online_for_part(part_number, manufacturer)
            if online_result:
                logger.info(f"Found part info online: {online_result.get('source')}")
                if online_result.get('has_kicad_symbol'):
                    errors.append(f"online_search: Symbol available on {online_result.get('source')} - manual download required: {online_result.get('url', 'N/A')}")
                else:
                    errors.append(f"online_search: Part exists on {online_result.get('source')} but no KiCad symbol available")
            else:
                errors.append("online_search: Part not found on Octopart or SnapEDA")
        except Exception as e:
            errors.append(f"online_search: {str(e)}")

        # Try LLM-based symbol generation if API key available (OpenRouter)
        if self._openrouter_api_key:
            sources_tried.append("llm_generated")
            try:
                logger.warning(f"All sources exhausted for {part_number}, generating with LLM")
                result = await self._generate_symbol_with_llm(part_number, manufacturer, category)
                if result and not result.metadata.get("is_placeholder", False):
                    return result
                errors.append("llm_generated: Generation returned placeholder")
            except Exception as e:
                errors.append(f"llm_generated: {str(e)}")
        elif self.anthropic_client:
            sources_tried.append("llm_generated")
            try:
                logger.warning(f"All sources exhausted for {part_number}, generating with LLM")
                result = await self._generate_symbol_with_llm(part_number, manufacturer, category)
                if result and not result.metadata.get("is_placeholder", False):
                    return result
                errors.append("llm_generated: Generation returned placeholder")
            except Exception as e:
                errors.append(f"llm_generated: {str(e)}")
        else:
            errors.append("llm_generated: No API key (set OPENROUTER_API_KEY or ANTHROPIC_API_KEY)")

        # All sources exhausted - ALWAYS raise exception now, NO PLACEHOLDERS
        # User must resolve missing parts before schematic generation can proceed
        logger.error(
            f"CRITICAL: Symbol resolution FAILED for '{part_number}' (manufacturer: {manufacturer or 'Unknown'}). "
            f"Tried {len(sources_tried)} sources: {', '.join(sources_tried)}. "
            f"GENERATION STOPPED - this part must be resolved manually."
        )

        # Build helpful suggestion based on what we found
        suggestion = "Unable to find this part in any symbol source. Options:\n"
        suggestion += "1. Check the part number is correct (search DigiKey/Mouser to verify)\n"
        suggestion += "2. Download KiCad symbol from SnapEDA: https://www.snapeda.com/search/?q=" + part_number + "\n"
        suggestion += "3. Download from Ultra Librarian: https://www.ultralibrarian.com/search?q=" + part_number + "\n"
        suggestion += "4. Add symbol manually to project symbol_cache/ directory\n"
        suggestion += "5. Set OPENROUTER_API_KEY for LLM-based symbol generation"

        # ALWAYS raise exception - no more placeholder fallback
        raise SymbolResolutionError(
            part_number=part_number,
            category=category,
            sources_tried=sources_tried,
            errors=errors,
            suggestion=suggestion
        )

    async def _fetch_generic_passive(
        self,
        generic_symbol: str,
        original_part: str,
        category: str
    ) -> Optional[FetchedSymbol]:
        """
        Fetch a generic passive symbol (C, R, L) from Device library.

        Priority order:
        1. Internal KiCad worker (K8s - has all libraries)
        2. Local KiCad installation
        3. Local cache
        """
        lib_content = None
        source = SymbolSource.KICAD_OFFICIAL

        # PRIORITY 1: Try internal KiCad worker (K8s deployment)
        try:
            worker_url = f"{KICAD_WORKER_URL}/v1/symbols/Device"
            logger.debug(f"Trying KiCad worker: {worker_url}")
            response = await self.http_client.get(worker_url, timeout=10.0)
            if response.status_code == 200:
                data = response.json()
                lib_content = data.get('content')
                if lib_content:
                    logger.info("Using Device library from internal KiCad worker")
                    source = SymbolSource.KICAD_WORKER_INTERNAL
        except Exception as e:
            logger.debug(f"KiCad worker not available: {e}")

        # PRIORITY 2: Try local KiCad installation
        if not lib_content:
            kicad_paths = _get_kicad_install_paths()
            for kicad_path in kicad_paths:
                device_lib = kicad_path / "Device.kicad_sym"
                if device_lib.exists():
                    try:
                        lib_content = device_lib.read_text(encoding='utf-8')
                        logger.debug(f"Using local Device library: {device_lib}")
                        source = SymbolSource.KICAD_LOCAL_INSTALL
                        break
                    except Exception as e:
                        logger.debug(f"Error reading local Device library: {e}")

        # PRIORITY 3: Try local cache
        if not lib_content:
            cached_device_lib = self.cache_path / "_libraries" / "Device.kicad_sym"
            if cached_device_lib.exists():
                try:
                    lib_content = cached_device_lib.read_text(encoding='utf-8')
                    logger.debug("Using cached Device library")
                    source = SymbolSource.LOCAL_CACHE
                except Exception as e:
                    logger.debug(f"Error reading cached Device library: {e}")

        if not lib_content:
            return None

        # Try to find the generic symbol
        try:
            symbol_sexp = self._extract_symbol_from_library(lib_content, generic_symbol)
            if symbol_sexp:
                logger.info(f"Found generic symbol '{generic_symbol}' for {original_part} from {source.value}")
                return FetchedSymbol(
                    part_number=original_part,  # Keep original part number
                    manufacturer=None,
                    symbol_sexp=symbol_sexp,
                    source=source,
                    metadata={
                        'library': 'Device',
                        'match_type': 'generic_passive',
                        'generic_symbol': generic_symbol,
                    }
                )
        except Exception as e:
            logger.debug(f"Error extracting generic passive {generic_symbol}: {e}")

        return None

    async def _fetch_from_source(
        self,
        source_config: SymbolSourceConfig,
        part_number: str,
        manufacturer: Optional[str],
        category: str
    ) -> Optional[FetchedSymbol]:
        """Dispatch to appropriate source handler."""
        handlers = {
            SymbolSource.LOCAL_CACHE: self._fetch_from_local_cache,
            SymbolSource.KICAD_LOCAL_INSTALL: self._fetch_from_kicad_local_install,
            SymbolSource.KICAD_WORKER_INTERNAL: self._fetch_from_kicad_worker,
            SymbolSource.KICAD_OFFICIAL: self._fetch_from_kicad_official,
            SymbolSource.SNAPEDA: self._fetch_from_snapeda,
            SymbolSource.ULTRA_LIBRARIAN: self._fetch_from_ultralibrarian,
            SymbolSource.TI_WEBENCH: self._fetch_from_ti,
            SymbolSource.STM_CUBE: self._fetch_from_stm,
            SymbolSource.LLM_GENERATED: self._generate_symbol_with_llm,
        }

        handler = handlers.get(source_config.source)
        if handler:
            if source_config.source in (SymbolSource.LOCAL_CACHE, SymbolSource.KICAD_LOCAL_INSTALL, SymbolSource.KICAD_WORKER_INTERNAL):
                return await handler(part_number, manufacturer, category)
            elif source_config.source == SymbolSource.LLM_GENERATED:
                return await handler(part_number, manufacturer, category)
            else:
                return await handler(source_config, part_number, manufacturer)

        return None

    async def _fetch_from_local_cache(
        self,
        part_number: str,
        manufacturer: Optional[str],
        category: str
    ) -> Optional[FetchedSymbol]:
        """Check local symbol cache."""
        # Try exact match first
        for cat_dir in self.cache_path.iterdir():
            if cat_dir.is_dir():
                symbol_path = cat_dir / f"{part_number}.kicad_sym"
                if symbol_path.exists():
                    symbol_sexp = symbol_path.read_text()
                    footprint_path = cat_dir / f"{part_number}.kicad_mod"
                    footprint_sexp = footprint_path.read_text() if footprint_path.exists() else None

                    # Load metadata if exists
                    meta_path = cat_dir / f"{part_number}.meta.json"
                    metadata = {}
                    if meta_path.exists():
                        metadata = json.loads(meta_path.read_text())

                    return FetchedSymbol(
                        part_number=part_number,
                        manufacturer=metadata.get('manufacturer', manufacturer),
                        symbol_sexp=symbol_sexp,
                        footprint_sexp=footprint_sexp,
                        datasheet_url=metadata.get('datasheet_url'),
                        source=SymbolSource.LOCAL_CACHE,
                        metadata=metadata
                    )

        # Try fuzzy match using part number prefix
        # Skip fuzzy matching for power symbols (VCC, GND, etc.) — these must
        # match exactly, otherwise "GND" normalizes to "G" (stripping "ND" suffix)
        # and falsely matches unrelated parts like "FG28X7R1E222K".
        POWER_SYMBOLS = {"VCC", "GND", "VDD", "VSS", "VBUS", "3V3", "5V", "12V", "+3V3", "+5V", "+12V"}
        normalized = self._normalize_part_number(part_number)
        if part_number.upper() not in POWER_SYMBOLS and len(normalized) >= 4:
            for cat_dir in self.cache_path.iterdir():
                if cat_dir.is_dir():
                    for symbol_file in cat_dir.glob("*.kicad_sym"):
                        cached_normalized = self._normalize_part_number(symbol_file.stem)
                        if len(cached_normalized) < 4:
                            continue
                        # Check if either contains the other (handles suffix variants)
                        if normalized in cached_normalized or cached_normalized in normalized:
                            logger.info(
                                f"Fuzzy cache hit: '{part_number}' matched '{symbol_file.stem}' "
                                f"(normalized: '{normalized}' in '{cached_normalized}')"
                            )
                            symbol_sexp = symbol_file.read_text()
                            return FetchedSymbol(
                                part_number=symbol_file.stem,
                                manufacturer=manufacturer,
                                symbol_sexp=symbol_sexp,
                                source=SymbolSource.LOCAL_CACHE,
                                metadata={'fuzzy_match': True, 'original_query': part_number}
                            )

        # Try matching by base part family (strip trailing package/variant codes)
        # e.g., STM32G431CBT6 → STM32G431CB, TJA1051T/3 → TJA1051T
        base_pn = re.sub(r'[/].*$', '', part_number)  # Strip /3 suffix
        base_pn = re.sub(r'[-_\s]', '', base_pn).upper()
        if len(base_pn) > 6:
            # Try progressively shorter prefix matching
            for prefix_len in range(len(base_pn), max(len(base_pn) - 4, 5), -1):
                prefix = base_pn[:prefix_len]
                for cat_dir in self.cache_path.iterdir():
                    if cat_dir.is_dir():
                        for symbol_file in cat_dir.glob("*.kicad_sym"):
                            stem_norm = re.sub(r'[-_\s]', '', symbol_file.stem).upper()
                            # Require cached symbol name to be at least 4 chars to prevent
                            # single-letter symbols (L, R, C, D) from matching any part
                            # number starting with that letter (e.g. L matching LTC4412ES6)
                            if len(stem_norm) < 4:
                                continue
                            if stem_norm.startswith(prefix) or prefix.startswith(stem_norm):
                                logger.info(
                                    f"Base-family cache hit: '{part_number}' matched "
                                    f"'{symbol_file.stem}' (prefix: '{prefix}')"
                                )
                                symbol_sexp = symbol_file.read_text()
                                return FetchedSymbol(
                                    part_number=symbol_file.stem,
                                    manufacturer=manufacturer,
                                    symbol_sexp=symbol_sexp,
                                    source=SymbolSource.LOCAL_CACHE,
                                    metadata={'fuzzy_match': True, 'match_type': 'base_family', 'original_query': part_number}
                                )

        return None

    async def _fetch_from_kicad_worker(
        self,
        part_number: str,
        manufacturer: Optional[str],
        category: str
    ) -> Optional[FetchedSymbol]:
        """
        Fetch symbol from internal KiCad worker service (K8s deployment).

        The KiCad worker has all official KiCad libraries installed.
        This is the preferred source in Docker/K8s environments.
        """
        try:
            # Get list of likely library names for this part
            library_names = self._guess_kicad_library(part_number, manufacturer)

            for lib_name in library_names:
                # Try to get the specific symbol from the library
                url = f"{KICAD_WORKER_URL}/v1/symbols/{lib_name}/symbol/{part_number}"
                logger.debug(f"Trying KiCad worker: {url}")

                response = await self.http_client.get(url, timeout=10.0)
                if response.status_code == 200:
                    data = response.json()
                    symbol_sexp = data.get('sexp')
                    if symbol_sexp:
                        logger.info(f"Found symbol '{part_number}' in library '{lib_name}' from KiCad worker")

                        # Check if symbol uses extends - if so, fetch library and flatten
                        if '(extends ' in symbol_sexp:
                            logger.info(f"Symbol '{part_number}' uses extends, fetching library for flattening")
                            lib_url = f"{KICAD_WORKER_URL}/v1/symbols/{lib_name}"
                            try:
                                lib_response = await self.http_client.get(lib_url, timeout=15.0)
                                if lib_response.status_code == 200:
                                    lib_data = lib_response.json()
                                    lib_content = lib_data.get('content', '')
                                    if lib_content:
                                        symbol_sexp = self._flatten_inherited_symbol(lib_content, symbol_sexp, part_number)
                            except Exception as e:
                                logger.warning(f"Failed to flatten inherited symbol: {e}")

                        # Wrap in library format if not already
                        if not symbol_sexp.strip().startswith('(kicad_symbol_lib'):
                            symbol_sexp = f'(kicad_symbol_lib (version 20231120) (generator nexus_ee_design)\n  {symbol_sexp}\n)'

                        return FetchedSymbol(
                            part_number=part_number,
                            manufacturer=manufacturer,
                            symbol_sexp=symbol_sexp,
                            source=SymbolSource.KICAD_WORKER_INTERNAL,
                            metadata={
                                'library': lib_name,
                                'match_type': 'exact',
                                'source_url': url
                            }
                        )

            # If specific symbol not found, use LLM to find best match in guessed libraries
            # LLM understands KiCad naming conventions: wildcards (x), package variants (_6-8-B_), etc.
            for lib_name in library_names:
                try:
                    lib_url = f"{KICAD_WORKER_URL}/v1/symbols/{lib_name}"
                    lib_response = await self.http_client.get(lib_url, timeout=15.0)
                    if lib_response.status_code == 200:
                        lib_data = lib_response.json()
                        lib_content = lib_data.get('content', '')
                        if lib_content:
                            # Extract all symbol names from the library (filter out sub-symbols)
                            symbol_names = [
                                name for name in re.findall(r'\(symbol\s+"([^"]+)"', lib_content)
                                if not re.search(r'_\d+_\d+$', name)  # Skip sub-symbols like _0_1, _1_1
                            ]

                            # Use LLM to find the best matching symbol
                            matched_symbol = await self._llm_find_best_symbol_match(
                                part_number, manufacturer, symbol_names, lib_name
                            )

                            if matched_symbol:
                                symbol_sexp = self._extract_symbol_from_library(lib_content, matched_symbol)
                                if symbol_sexp:
                                    logger.info(f"LLM matched: '{matched_symbol}' for '{part_number}' in '{lib_name}'")
                                    return FetchedSymbol(
                                        part_number=matched_symbol,  # Use actual KiCad symbol name
                                        manufacturer=manufacturer,
                                        symbol_sexp=symbol_sexp,
                                        source=SymbolSource.KICAD_WORKER_INTERNAL,
                                        metadata={
                                            'library': lib_name,
                                            'match_type': 'llm_matched',
                                            'original_query': part_number,
                                            'matched_symbol': matched_symbol
                                        }
                                    )
                except Exception as e:
                    logger.debug(f"LLM match failed for {lib_name}: {e}")

            # If still not found, try searching all libraries (broad search with fuzzy)
            list_url = f"{KICAD_WORKER_URL}/v1/symbols"
            list_response = await self.http_client.get(list_url, timeout=10.0)
            if list_response.status_code == 200:
                libraries = list_response.json().get('libraries', [])

                # Search libraries that might contain this part (capped to avoid runaway 404s)
                broad_search_count = 0
                MAX_BROAD_SEARCH = 15
                for lib_name in libraries:
                    if broad_search_count >= MAX_BROAD_SEARCH:
                        logger.info(f"Broad search limit reached ({MAX_BROAD_SEARCH}) for '{part_number}', falling back to LLM generation")
                        break

                    if lib_name in self.SKIP_LIBRARIES:
                        continue

                    # Skip already-searched libraries
                    if lib_name in library_names:
                        continue

                    broad_search_count += 1

                    url = f"{KICAD_WORKER_URL}/v1/symbols/{lib_name}/symbol/{part_number}"
                    try:
                        response = await self.http_client.get(url, timeout=5.0)
                        if response.status_code == 200:
                            data = response.json()
                            symbol_sexp = data.get('sexp')
                            if symbol_sexp:
                                logger.info(f"Found symbol '{part_number}' in library '{lib_name}' from KiCad worker (broad search)")

                                # Check if symbol uses extends - if so, fetch library and flatten
                                if '(extends ' in symbol_sexp:
                                    logger.info(f"Symbol '{part_number}' uses extends, fetching library for flattening")
                                    lib_url = f"{KICAD_WORKER_URL}/v1/symbols/{lib_name}"
                                    try:
                                        lib_response = await self.http_client.get(lib_url, timeout=15.0)
                                        if lib_response.status_code == 200:
                                            lib_data = lib_response.json()
                                            lib_content = lib_data.get('content', '')
                                            if lib_content:
                                                symbol_sexp = self._flatten_inherited_symbol(lib_content, symbol_sexp, part_number)
                                    except Exception as e:
                                        logger.warning(f"Failed to flatten inherited symbol: {e}")

                                # Wrap in library format if not already
                                if not symbol_sexp.strip().startswith('(kicad_symbol_lib'):
                                    symbol_sexp = f'(kicad_symbol_lib (version 20231120) (generator nexus_ee_design)\n  {symbol_sexp}\n)'

                                return FetchedSymbol(
                                    part_number=part_number,
                                    manufacturer=manufacturer,
                                    symbol_sexp=symbol_sexp,
                                    source=SymbolSource.KICAD_WORKER_INTERNAL,
                                    metadata={
                                        'library': lib_name,
                                        'match_type': 'broad_search',
                                        'source_url': url
                                    }
                                )
                    except Exception:
                        # Symbol not in this library, continue
                        pass

        except Exception as e:
            logger.debug(f"KiCad worker fetch failed for {part_number}: {e}")

        return None

    # Libraries to skip during broad search (simulation, power symbols, non-component libraries)
    SKIP_LIBRARIES = {
        'Simulation_SPICE', 'power', 'Graphic', 'Mechanical',
        '4xxx', '4xxx_IEEE', '74xGxx', '74xx', '74xx_IEEE',
        'Connector', 'Connector_Audio', 'Connector_Generic',
        'Connector_Generic_MountingPin', 'Connector_Generic_Shielded',
        'Display_Character', 'Display_Graphic',
        'Jumper', 'LED', 'Motor', 'Relay', 'Relay_SolidState',
        'Fiber_Optic', 'RF_AM_FM', 'RF_GPS', 'RF_GSM', 'RF_NFC', 'RF_RFID', 'RF_ZigBee',
        'Memory_EPROM', 'Memory_ROM', 'Memory_UniqueID',
    }

    # Generic symbol mappings for active devices (MOSFETs, BJTs, etc.)
    GENERIC_ACTIVE_SYMBOLS = {
        'MOSFET': 'Q_NMOS_GDS',
        'MOSFET_P': 'Q_PMOS_GDS',
        'BJT_NPN': 'Q_NPN_BEC',
        'BJT_PNP': 'Q_PNP_BEC',
        'Diode': 'D',
        'Zener': 'D_Zener',
    }

    async def _fetch_from_kicad_local_install(
        self,
        part_number: str,
        manufacturer: Optional[str],
        category: str
    ) -> Optional[FetchedSymbol]:
        """
        Fetch symbol from local KiCad installation libraries.

        Searches the official KiCad symbol libraries installed on the local system.
        These libraries contain multiple symbols per file (e.g., MCU_ST_STM32G4.kicad_sym).
        Falls back to GitHub download when no local installation is found.
        """
        kicad_paths = _get_kicad_install_paths()

        # Get list of likely library names for this part
        library_names = self._guess_kicad_library(part_number, manufacturer)

        # Try local KiCad installation first
        if kicad_paths:
            for kicad_path in kicad_paths:
                logger.debug(f"Searching KiCad libraries in: {kicad_path}")

                # First try the guessed library names
                for lib_name in library_names:
                    lib_path = kicad_path / f"{lib_name}.kicad_sym"
                    if lib_path.exists():
                        result = await self._search_symbol_in_library_file(
                            lib_path, part_number, manufacturer
                        )
                        if result:
                            return result

                # If not found, scan all library files (skip problematic libraries)
                for lib_file in kicad_path.glob("*.kicad_sym"):
                    # Skip already checked libraries
                    if lib_file.stem in library_names:
                        continue

                    # Skip simulation and other non-component libraries
                    if lib_file.stem in self.SKIP_LIBRARIES:
                        continue

                    result = await self._search_symbol_in_library_file(
                        lib_file, part_number, manufacturer
                    )
                    if result:
                        return result
        else:
            # No local KiCad - try fetching libraries from GitHub
            logger.debug("No local KiCad installation, trying GitHub mirror")
            cached_libs_dir = self.cache_path / "_libraries"
            cached_libs_dir.mkdir(parents=True, exist_ok=True)

            for lib_name in library_names:
                # Check cache first
                cached_lib = cached_libs_dir / f"{lib_name}.kicad_sym"
                lib_content = None

                if cached_lib.exists():
                    try:
                        lib_content = cached_lib.read_text(encoding='utf-8')
                    except Exception:
                        pass

                # Fetch from GitHub if not cached
                if not lib_content:
                    try:
                        url = f"https://raw.githubusercontent.com/KiCad/kicad-symbols/master/{lib_name}.kicad_sym"
                        response = await self.http_client.get(url, timeout=30.0)
                        if response.status_code == 200:
                            lib_content = response.text
                            # Cache for future use
                            cached_lib.write_text(lib_content)
                            logger.info(f"Cached {lib_name} library from GitHub")
                    except Exception as e:
                        logger.debug(f"Failed to fetch {lib_name} from GitHub: {e}")
                        continue

                if lib_content:
                    # Search for the symbol in the fetched library
                    symbol_sexp = self._extract_symbol_from_library(lib_content, part_number)
                    if symbol_sexp:
                        logger.info(f"Found {part_number} in GitHub {lib_name}")
                        return FetchedSymbol(
                            part_number=part_number,
                            manufacturer=manufacturer,
                            symbol_sexp=symbol_sexp,
                            source=SymbolSource.KICAD_OFFICIAL,
                            metadata={
                                'library': lib_name,
                                'match_type': 'github_fetch',
                                'cached': True
                            }
                        )

                    # Try fuzzy match with all symbols in library
                    symbol_names = [
                        name for name in re.findall(r'\(symbol\s+"([^"]+)"', lib_content)
                        if not re.search(r'_\d+_\d+$', name)
                    ]

                    for symbol_name in symbol_names:
                        if self._match_kicad_symbol_name(part_number, symbol_name):
                            symbol_sexp = self._extract_symbol_from_library(lib_content, symbol_name)
                            if symbol_sexp:
                                logger.info(f"Found fuzzy match: {symbol_name} for {part_number} in GitHub {lib_name}")
                                return FetchedSymbol(
                                    part_number=symbol_name,
                                    manufacturer=manufacturer,
                                    symbol_sexp=symbol_sexp,
                                    source=SymbolSource.KICAD_OFFICIAL,
                                    metadata={
                                        'library': lib_name,
                                        'match_type': 'fuzzy_github_fetch',
                                        'original_query': part_number,
                                        'cached': True
                                    }
                                )

        return None

    async def _llm_find_best_symbol_match(
        self,
        part_number: str,
        manufacturer: Optional[str],
        available_symbols: List[str],
        library_name: str
    ) -> Optional[str]:
        """
        Use LLM to find the best matching KiCad symbol for a part number.

        NO REGEX MATCHING - Uses Claude to intelligently match part numbers to
        KiCad's naming conventions (wildcards, package suffixes, etc.).

        Args:
            part_number: The actual part number (e.g., "STM32G431CBT6")
            manufacturer: Manufacturer name (e.g., "STMicroelectronics")
            available_symbols: List of symbol names in the KiCad library
            library_name: Name of the library being searched

        Returns:
            The best matching symbol name, or None if no match found
        """
        if not self._openrouter_api_key:
            logger.warning("No OpenRouter API key for LLM symbol matching")
            return None

        # Filter to symbols that have some textual similarity (basic pre-filter)
        # This reduces the list size for the LLM to process
        part_upper = part_number.upper()
        part_prefix = part_upper[:6] if len(part_upper) >= 6 else part_upper[:3]
        relevant_symbols = [
            s for s in available_symbols
            if part_prefix[:3] in s.upper() or s.upper()[:3] in part_prefix
        ]

        # If no relevant symbols found with prefix, include all (small list)
        if not relevant_symbols and len(available_symbols) <= 50:
            relevant_symbols = available_symbols
        elif not relevant_symbols:
            # Try broader search
            relevant_symbols = [
                s for s in available_symbols
                if any(c in s.upper() for c in part_upper[:4])
            ][:100]  # Limit to 100

        if not relevant_symbols:
            logger.debug(f"No potentially matching symbols found for {part_number}")
            return None

        prompt = f"""You are an electronic component expert matching part numbers to KiCad symbols.

TASK: Find the EXACT KiCad symbol that matches this part number.

Part Number: {part_number}
Manufacturer: {manufacturer or 'Unknown'}
KiCad Library: {library_name}

Available symbols in this library:
{chr(10).join(relevant_symbols[:80])}

MATCHING RULES (KiCad naming conventions):
1. 'x' in KiCad symbols is a wildcard: "STM32G431C_6-8-B_Tx" matches STM32G431CBT6
2. "_6-8-B_" means "6, 8, or B package variant" - matches C6, C8, or CB
3. Package suffixes (T6, T7, TX) may differ - T6 part matches Tx symbol
4. Temperature grades may be omitted in symbols
5. Some symbols use underscores/dashes where parts use none

RESPOND WITH ONLY:
- The EXACT symbol name from the list above that matches the part
- "NO_MATCH" if no symbol matches this part

DO NOT explain. Just output the symbol name or NO_MATCH."""

        try:
            response = await self.http_client.post(
                f"{self._openrouter_base_url}/chat/completions",
                json={
                    "model": self._openrouter_model,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 100,
                    "temperature": 0,
                },
                headers={
                    "Authorization": f"Bearer {self._openrouter_api_key}",
                    "HTTP-Referer": "https://adverant.ai",
                    "X-Title": "Nexus EE Design Symbol Matcher",
                    "Content-Type": "application/json",
                },
                timeout=30.0
            )

            if response.status_code != 200:
                logger.error(f"OpenRouter API error: {response.status_code} - {response.text}")
                return None

            result = response.json()
            matched_symbol = result['choices'][0]['message']['content'].strip()
            # Strip markdown formatting (bold, italic, backticks) from LLM response
            matched_symbol = re.sub(r'^[\*`]+|[\*`]+$', '', matched_symbol).strip()

            if matched_symbol == "NO_MATCH":
                logger.debug(f"LLM found no match for {part_number} in {library_name}")
                return None

            # Verify the matched symbol is actually in the list
            if matched_symbol in available_symbols:
                logger.info(f"LLM matched '{part_number}' to '{matched_symbol}' in {library_name}")
                return matched_symbol
            else:
                # Sometimes LLM returns slightly modified name, find closest
                for sym in available_symbols:
                    if matched_symbol.lower() == sym.lower():
                        logger.info(f"LLM matched '{part_number}' to '{sym}' (case-insensitive)")
                        return sym

                logger.warning(f"LLM returned '{matched_symbol}' but it's not in the symbol list")
                return None

        except Exception as e:
            logger.error(f"LLM symbol matching failed: {e}")
            return None

    async def _search_online_for_part(
        self,
        part_number: str,
        manufacturer: Optional[str]
    ) -> Optional[Dict[str, Any]]:
        """
        Search online for part information when not found in KiCad libraries.

        Searches:
        1. Octopart API (aggregates DigiKey, Mouser, etc.)
        2. SnapEDA for KiCad symbols
        3. Manufacturer websites

        Returns part info including potential symbol sources.
        """
        logger.info(f"Searching online for part: {part_number}")

        # Try Octopart search (free tier, no API key needed for basic search)
        try:
            search_url = f"https://octopart.com/api/v4/rest/search"
            params = {"q": part_number, "limit": 5}
            response = await self.http_client.get(
                search_url,
                params=params,
                headers={"Accept": "application/json"},
                timeout=15.0
            )
            if response.status_code == 200:
                data = response.json()
                if data.get("results"):
                    logger.info(f"Found {part_number} on Octopart")
                    return {
                        "source": "octopart",
                        "data": data["results"][0],
                        "part_number": part_number
                    }
        except Exception as e:
            logger.debug(f"Octopart search failed: {e}")

        # Try SnapEDA search (public search, no API key needed)
        try:
            snapeda_url = f"https://www.snapeda.com/search/?q={part_number}"
            response = await self.http_client.get(snapeda_url, timeout=15.0)
            if response.status_code == 200 and part_number.lower() in response.text.lower():
                logger.info(f"Found {part_number} on SnapEDA")
                return {
                    "source": "snapeda",
                    "url": snapeda_url,
                    "part_number": part_number,
                    "has_kicad_symbol": True
                }
        except Exception as e:
            logger.debug(f"SnapEDA search failed: {e}")

        return None

    def _match_kicad_symbol_name(self, query: str, symbol_name: str) -> bool:
        """
        DEPRECATED: Basic regex matching - Use _llm_find_best_symbol_match instead.

        This method is kept for backwards compatibility but LLM matching is preferred.
        """
        # Exact match only - no fuzzy matching
        query_norm = query.upper().replace('-', '').replace('_', '').replace(' ', '')
        symbol_norm = symbol_name.upper().replace('-', '').replace('_', '').replace(' ', '')
        return query_norm == symbol_norm

    async def _search_symbol_in_library_file(
        self,
        lib_path: Path,
        part_number: str,
        manufacturer: Optional[str]
    ) -> Optional[FetchedSymbol]:
        """
        Search for a specific symbol within a KiCad library file.

        KiCad library files contain multiple symbol definitions. This method
        parses the file and extracts the matching symbol.
        """
        try:
            lib_content = lib_path.read_text(encoding='utf-8')

            # Try exact match first
            symbol_sexp = self._extract_symbol_from_library(lib_content, part_number)
            if symbol_sexp:
                logger.info(f"Found exact match for {part_number} in {lib_path.name}")
                return FetchedSymbol(
                    part_number=part_number,
                    manufacturer=manufacturer,
                    symbol_sexp=symbol_sexp,
                    source=SymbolSource.KICAD_LOCAL_INSTALL,
                    metadata={
                        'library': lib_path.stem,
                        'library_path': str(lib_path),
                        'match_type': 'exact'
                    }
                )

            # Extract all symbol names from the library (skip sub-symbols with _0_1, _1_1 suffixes)
            symbol_names = [
                name for name in re.findall(r'\(symbol\s+"([^"]+)"', lib_content)
                if not re.search(r'_\d+_\d+$', name)  # Skip unit/style variants
            ]

            # Find best match using the improved matching function
            for symbol_name in symbol_names:
                if self._match_kicad_symbol_name(part_number, symbol_name):
                    symbol_sexp = self._extract_symbol_from_library(lib_content, symbol_name)
                    if symbol_sexp:
                        logger.info(f"Found fuzzy match: {symbol_name} for query {part_number} in {lib_path.name}")
                        return FetchedSymbol(
                            part_number=symbol_name,  # Use actual symbol name
                            manufacturer=manufacturer,
                            symbol_sexp=symbol_sexp,
                            source=SymbolSource.KICAD_LOCAL_INSTALL,
                            metadata={
                                'library': lib_path.stem,
                                'library_path': str(lib_path),
                                'match_type': 'fuzzy',
                                'original_query': part_number
                            }
                        )

        except Exception as e:
            logger.debug(f"Error searching library {lib_path}: {e}")

        return None

    async def _fetch_from_kicad_official(
        self,
        source_config: SymbolSourceConfig,
        part_number: str,
        manufacturer: Optional[str]
    ) -> Optional[FetchedSymbol]:
        """
        Fetch from KiCad official symbol libraries on GitHub mirror.

        The official library is organized by component category.
        Note: GitLab requires auth, so we use the GitHub mirror instead.
        """
        # Map part numbers to likely library files
        library_mapping = self._guess_kicad_library(part_number, manufacturer)

        for library_name in library_mapping:
            try:
                # Use GitHub mirror (no auth required) instead of GitLab
                url = (
                    f"https://raw.githubusercontent.com/KiCad/kicad-symbols/master/"
                    f"{library_name}.kicad_sym"
                )

                response = await self.http_client.get(url, follow_redirects=True)
                if response.status_code == 200:
                    library_content = response.text

                    # Search for the specific symbol in the library file
                    symbol_sexp = self._extract_symbol_from_library(
                        library_content, part_number
                    )
                    if symbol_sexp:
                        return FetchedSymbol(
                            part_number=part_number,
                            manufacturer=manufacturer,
                            symbol_sexp=symbol_sexp,
                            source=SymbolSource.KICAD_OFFICIAL,
                            metadata={
                                'library': library_name,
                                'github_url': url
                            }
                        )

            except Exception as e:
                logger.debug(f"KiCad official lookup failed for {library_name}: {e}")
                continue

        return None

    async def _fetch_from_snapeda(
        self,
        source_config: SymbolSourceConfig,
        part_number: str,
        manufacturer: Optional[str]
    ) -> Optional[FetchedSymbol]:
        """
        Fetch from SnapEDA API.

        SnapEDA provides free access for basic part searches.
        API Documentation: https://www.snapeda.com/api/
        """
        api_key = os.getenv(source_config.api_key_env or "SNAPEDA_API_KEY")

        # Try free search first (doesn't require API key)
        try:
            # SnapEDA public search endpoint
            search_url = "https://www.snapeda.com/api/v1/parts"
            params = {"q": part_number, "limit": 5}
            if manufacturer:
                params["manufacturer"] = manufacturer

            headers = {}
            if api_key:
                headers["Authorization"] = f"Bearer {api_key}"

            logger.info(f"Searching SnapEDA for: {part_number}")
            response = await self.http_client.get(
                search_url, params=params, headers=headers, timeout=10.0
            )

            if response.status_code == 200:
                data = response.json()
                results = data.get("results", data.get("parts", []))

                if results:
                    # Find best match
                    part = self._find_best_snapeda_match(results, part_number, manufacturer)
                    if part:
                        part_id = part.get("id") or part.get("part_id")
                        logger.info(f"Found SnapEDA part: {part_id}")

                        # Try to download KiCad format
                        if part_id:
                            download_url = f"https://www.snapeda.com/api/v1/parts/{part_id}/kicad"
                            download_response = await self.http_client.get(
                                download_url,
                                headers=headers,
                                timeout=15.0
                            )

                            if download_response.status_code == 200:
                                return FetchedSymbol(
                                    part_number=part_number,
                                    manufacturer=part.get("manufacturer", manufacturer),
                                    symbol_sexp=download_response.text,
                                    datasheet_url=part.get("datasheet_url"),
                                    source=SymbolSource.SNAPEDA,
                                    metadata={
                                        "snapeda_id": part_id,
                                        "description": part.get("description"),
                                        "package": part.get("package"),
                                        "has_3d_model": part.get("has_3d_model", False)
                                    }
                                )

            elif response.status_code == 401:
                logger.debug("SnapEDA requires authentication for this request")
            elif response.status_code == 429:
                logger.warning("SnapEDA rate limit exceeded")
            else:
                logger.debug(f"SnapEDA search returned status {response.status_code}")

        except Exception as e:
            logger.debug(f"SnapEDA fetch failed: {e}")

        return None

    def _find_best_snapeda_match(
        self,
        results: List[Dict],
        part_number: str,
        manufacturer: Optional[str]
    ) -> Optional[Dict]:
        """Find the best matching part from SnapEDA results."""
        part_lower = part_number.lower().replace("-", "").replace("_", "")

        for result in results:
            result_pn = (result.get("part_number") or result.get("mpn") or "").lower()
            result_pn = result_pn.replace("-", "").replace("_", "")

            # Exact match
            if result_pn == part_lower:
                return result

            # Check if part number is contained
            if part_lower in result_pn or result_pn in part_lower:
                return result

        # Return first result as fallback
        return results[0] if results else None

    async def _fetch_from_ultralibrarian(
        self,
        source_config: SymbolSourceConfig,
        part_number: str,
        manufacturer: Optional[str]
    ) -> Optional[FetchedSymbol]:
        """
        Fetch from Ultra Librarian API.

        API Documentation: https://www.ultralibrarian.com/developers
        """
        api_key = os.getenv(source_config.api_key_env or "ULTRALIBRARIAN_API_KEY")
        if not api_key and source_config.requires_auth:
            logger.debug("Ultra Librarian API key not configured")
            return None

        search_url = f"{source_config.api_url}/search"
        params = {
            "query": part_number,
            "format": "kicad"
        }

        headers = {"X-Api-Key": api_key} if api_key else {}

        try:
            response = await self.http_client.get(
                search_url, params=params, headers=headers
            )

            if response.status_code != 200:
                return None

            data = response.json()
            results = data.get("parts", [])

            if not results:
                return None

            part = results[0]

            # Download symbol
            download_url = part.get("download_url")
            if download_url:
                download_response = await self.http_client.get(
                    download_url, headers=headers
                )

                if download_response.status_code == 200:
                    return FetchedSymbol(
                        part_number=part_number,
                        manufacturer=part.get("manufacturer", manufacturer),
                        symbol_sexp=download_response.text,
                        datasheet_url=part.get("datasheet"),
                        source=SymbolSource.ULTRA_LIBRARIAN,
                        metadata={
                            "ul_id": part.get("id"),
                            "description": part.get("description"),
                            "package": part.get("package")
                        }
                    )

        except Exception as e:
            logger.debug(f"Ultra Librarian fetch failed: {e}")

        return None

    async def _fetch_from_ti(
        self,
        source_config: SymbolSourceConfig,
        part_number: str,
        manufacturer: Optional[str]
    ) -> Optional[FetchedSymbol]:
        """
        Fetch from Texas Instruments.

        TI provides symbols through their reference design downloads.
        """
        # TI symbol library URL pattern
        # This is a simplified implementation - real TI API integration would be more complex
        logger.debug(f"TI source not fully implemented for {part_number}")
        return None

    async def _fetch_from_stm(
        self,
        source_config: SymbolSourceConfig,
        part_number: str,
        manufacturer: Optional[str]
    ) -> Optional[FetchedSymbol]:
        """
        Fetch from STMicroelectronics.

        STM provides symbols through STM32CubeMX exports.
        """
        # STM symbol library URL pattern
        logger.debug(f"STM source not fully implemented for {part_number}")
        return None

    async def _generate_symbol_with_llm(
        self,
        part_number: str,
        manufacturer: Optional[str],
        category: str
    ) -> FetchedSymbol:
        """
        Generate symbol using LLM when all other sources fail.

        Uses OpenRouter API (OpenAI-compatible) for Claude access.
        IMPORTANT: OpenRouter uses /chat/completions endpoint, NOT /messages endpoint.
        """
        if not self._openrouter_api_key and not self.anthropic_client:
            raise RuntimeError(
                "No LLM API key available for symbol generation. "
                "Set OPENROUTER_API_KEY or ANTHROPIC_API_KEY environment variable."
            )

        logger.info(f"Generating symbol with LLM for {part_number}")

        # Try to get datasheet info for context
        datasheet_context = await self._fetch_datasheet_info(part_number, manufacturer)

        prompt = f"""Generate a KiCad symbol for the following electronic component:

Part Number: {part_number}
Manufacturer: {manufacturer or 'Unknown'}
Category: {category}

{f'Datasheet Information:{chr(10)}{datasheet_context}' if datasheet_context else ''}

Generate a valid KiCad 7.0+ S-expression symbol (.kicad_sym format) with:
1. Correct pin names and numbers based on the component type
2. Appropriate graphical representation (rectangle for ICs, standard symbols for passives)
3. Power pins (VCC, VDD, GND, VSS) marked with correct pin types
4. Proper pin types: input, output, bidirectional, tri_state, passive, power_in, power_out, open_collector, open_emitter, no_connect
5. Pin positions following standard conventions (inputs left, outputs right, power top/bottom)
6. Reference designator field (U, R, C, L, D, Q based on category)
7. Value field with part number

The symbol should be immediately usable in KiCad without modification.

Output ONLY the KiCad S-expression, starting with (kicad_symbol_lib and ending with the closing parenthesis.
No explanation or markdown formatting."""

        try:
            # Use OpenRouter API (preferred) or fallback to direct Anthropic
            if self._openrouter_api_key:
                response = await self.http_client.post(
                    f"{self._openrouter_base_url}/chat/completions",
                    json={
                        "model": "anthropic/claude-opus-4.6",
                        "messages": [{"role": "user", "content": prompt}],
                        "max_tokens": 8192,
                        "temperature": 0.2,
                    },
                    headers={
                        "Authorization": f"Bearer {self._openrouter_api_key}",
                        "HTTP-Referer": "https://adverant.ai",
                        "X-Title": "Nexus EE Design Symbol Generator",
                        "Content-Type": "application/json",
                    },
                    timeout=120.0
                )

                if response.status_code != 200:
                    raise RuntimeError(f"OpenRouter API error: {response.status_code} - {response.text}")

                result = response.json()
                generated_symbol = result['choices'][0]['message']['content']
            else:
                # Fallback to direct Anthropic API
                response = self.anthropic_client.messages.create(
                    model="claude-opus-4-6-20260206",  # Direct API requires full model ID
                    max_tokens=8192,
                    messages=[{"role": "user", "content": prompt}]
                )
                generated_symbol = response.content[0].text

            # Validate the generated symbol has basic structure
            if not generated_symbol.strip().startswith("(kicad_symbol_lib"):
                # Try to extract S-expression if wrapped in markdown
                match = re.search(r'\(kicad_symbol_lib[\s\S]+\)', generated_symbol)
                if match:
                    generated_symbol = match.group()
                else:
                    raise ValueError("Generated symbol does not have valid S-expression structure")

            return FetchedSymbol(
                part_number=part_number,
                manufacturer=manufacturer,
                symbol_sexp=generated_symbol,
                source=SymbolSource.LLM_GENERATED,
                needs_review=True,
                metadata={
                    "generated_by": "anthropic/claude-opus-4.6",
                    "category": category,
                    "datasheet_context_used": bool(datasheet_context)
                }
            )

        except Exception as e:
            logger.error(f"LLM symbol generation failed: {e}")
            # Return a minimal placeholder symbol
            return self._create_placeholder_symbol(part_number, manufacturer, category)

    async def _fetch_datasheet_info(
        self,
        part_number: str,
        manufacturer: Optional[str]
    ) -> str:
        """Fetch datasheet information for context."""
        # Try to get datasheet info from web search or database
        # This is a simplified implementation
        try:
            # Search for datasheet using part number
            search_url = f"https://www.google.com/search?q={quote_plus(part_number + ' datasheet')}"
            # Note: In production, use a proper datasheet API or cached datasheet database
            return ""
        except Exception:
            return ""

    async def _cache_symbol(self, symbol: FetchedSymbol, category: str):
        """Cache symbol to local filesystem."""
        category_path = self.cache_path / category
        category_path.mkdir(exist_ok=True)

        # Save symbol
        symbol_path = category_path / f"{symbol.part_number}.kicad_sym"
        symbol_path.write_text(symbol.symbol_sexp)

        # Save footprint if available
        if symbol.footprint_sexp:
            footprint_path = category_path / f"{symbol.part_number}.kicad_mod"
            footprint_path.write_text(symbol.footprint_sexp)

        # Save metadata
        meta_path = category_path / f"{symbol.part_number}.meta.json"
        metadata = {
            "part_number": symbol.part_number,
            "manufacturer": symbol.manufacturer,
            "source": symbol.source.value,
            "datasheet_url": symbol.datasheet_url,
            "fetched_at": symbol.fetched_at,
            "needs_review": symbol.needs_review,
            **symbol.metadata
        }
        meta_path.write_text(json.dumps(metadata, indent=2))

        logger.info(f"Cached symbol: {symbol_path}")

    async def _index_in_graphrag(self, symbol: FetchedSymbol, category: str):
        """Index symbol in GraphRAG for semantic search."""
        if not self.graphrag:
            return

        document = {
            "type": "kicad_symbol",
            "part_number": symbol.part_number,
            "manufacturer": symbol.manufacturer,
            "category": category,
            "source": symbol.source.value,
            "symbol_preview": symbol.symbol_sexp[:1000],  # Truncate for embedding
            "datasheet_url": symbol.datasheet_url,
            "metadata": symbol.metadata,
            "needs_review": symbol.needs_review,
            "indexed_at": datetime.utcnow().isoformat()
        }

        embedding_text = (
            f"{symbol.part_number} {symbol.manufacturer or ''} "
            f"{category} {symbol.metadata.get('description', '')}"
        )

        try:
            await self.graphrag.add_document(
                collection="kicad_symbols",
                document=document,
                embedding_text=embedding_text
            )
            logger.info(f"Indexed symbol in GraphRAG: {symbol.part_number}")
        except Exception as e:
            logger.warning(f"Failed to index symbol in GraphRAG: {e}")

    # =========================================================================
    # NEXUS-MEMORY INTEGRATION: Self-Improving Part Discovery
    # =========================================================================

    async def _recall_from_nexus_memory(
        self,
        part_number: str,
        manufacturer: Optional[str]
    ) -> Optional[Dict[str, Any]]:
        """
        Query nexus-memory for previously learned symbol resolution.

        Returns:
            Dict with resolution info if found, None otherwise
        """
        try:
            query = f"component resolution {part_number}"
            if manufacturer:
                query += f" {manufacturer}"
            query += " KiCad symbol"

            result = await self._call_nexus_memory_api(
                operation="recall",
                query=query,
                limit=5
            )

            if result and result.get("memories"):
                # Find best match for this exact part number
                for memory in result["memories"]:
                    content = memory.get("content", "")
                    if isinstance(content, str):
                        try:
                            content = json.loads(content)
                        except json.JSONDecodeError:
                            continue

                    if isinstance(content, dict):
                        # Check if this is a component_resolution for our part
                        if content.get("type") == "component_resolution":
                            mem_pn = content.get("part_number", "").upper()
                            if mem_pn == part_number.upper():
                                resolution = content.get("resolution", {})
                                if resolution.get("verified"):
                                    logger.info(
                                        f"Nexus-memory hit: {part_number} → "
                                        f"{resolution.get('source')} (success_count: {resolution.get('success_count', 0)})"
                                    )
                                    return content

            logger.debug(f"No nexus-memory hit for {part_number}")
            return None

        except Exception as e:
            logger.warning(f"Nexus-memory recall failed: {e}")
            return None

    async def _fetch_from_learned_resolution(
        self,
        memory_result: Dict[str, Any],
        part_number: str,
        manufacturer: Optional[str],
        category: str
    ) -> Optional[FetchedSymbol]:
        """
        Use a learned resolution from nexus-memory to fetch a symbol.

        Args:
            memory_result: The stored resolution from nexus-memory
            part_number: Component part number
            manufacturer: Manufacturer name
            category: Component category

        Returns:
            FetchedSymbol if successful, None otherwise
        """
        try:
            resolution = memory_result.get("resolution", {})
            source = resolution.get("source", "")

            # Map source string to SymbolSource enum
            source_mapping = {
                "local_cache": SymbolSource.LOCAL_CACHE,
                "kicad_worker_internal": SymbolSource.KICAD_WORKER_INTERNAL,
                "kicad_local_install": SymbolSource.KICAD_LOCAL_INSTALL,
                "kicad_official": SymbolSource.KICAD_OFFICIAL,
                "snapeda": SymbolSource.SNAPEDA,
                "ultralibrarian": SymbolSource.ULTRA_LIBRARIAN,
                "learned_online": SymbolSource.SNAPEDA,  # Treat as SnapEDA
            }

            symbol_source = source_mapping.get(source.lower())
            if not symbol_source:
                logger.warning(f"Unknown learned source: {source}")
                return None

            # Find matching source config
            for source_config in self.sources:
                if source_config.source == symbol_source:
                    result = await self._fetch_from_source(
                        source_config, part_number, manufacturer, category
                    )
                    if result:
                        # Mark as from memory
                        result.metadata["from_nexus_memory"] = True
                        result.metadata["learned_source"] = source
                        return result

            return None

        except Exception as e:
            logger.warning(f"Failed to use learned resolution: {e}")
            return None

    async def _store_in_nexus_memory(
        self,
        part_number: str,
        manufacturer: Optional[str],
        category: str,
        source: str,
        result: FetchedSymbol
    ):
        """
        Store successful symbol resolution in nexus-memory for future learning.

        This creates a self-improving feedback loop where successful resolutions
        are remembered and used to speed up future symbol fetching.
        """
        try:
            memory_content = {
                "type": "component_resolution",
                "part_number": part_number,
                "manufacturer": manufacturer or "Unknown",
                "category": category,
                "package": result.metadata.get("package", "unknown"),
                "resolution": {
                    "source": source,
                    "url": result.metadata.get("source_url", ""),
                    "format": "kicad_sym",
                    "verified": True,
                    "pin_count": result.metadata.get("pin_count", 0),
                    "last_verified": datetime.utcnow().isoformat(),
                    "success_count": 1,
                    "failure_count": 0
                },
                "tags": ["component", category, source, part_number]
            }

            if manufacturer:
                memory_content["tags"].append(manufacturer)

            await self._call_nexus_memory_api(
                operation="store",
                content=json.dumps(memory_content),
                event_type="component_resolution"
            )

            logger.info(f"Stored {part_number} resolution in nexus-memory (source: {source})")

        except Exception as e:
            # Non-fatal - don't block symbol fetching if memory storage fails
            logger.warning(f"Failed to store in nexus-memory: {e}")

    async def _update_memory_success_count(
        self,
        part_number: str,
        manufacturer: Optional[str],
        memory_result: Dict[str, Any]
    ):
        """
        Update success count for a resolution that worked again.

        This reinforces successful resolutions in the learning system.
        """
        try:
            # Get current success count
            resolution = memory_result.get("resolution", {})
            current_count = resolution.get("success_count", 0)

            # Update the memory with incremented count
            memory_content = {
                **memory_result,
                "resolution": {
                    **resolution,
                    "success_count": current_count + 1,
                    "last_verified": datetime.utcnow().isoformat()
                }
            }

            await self._call_nexus_memory_api(
                operation="store",
                content=json.dumps(memory_content),
                event_type="component_resolution"
            )

            logger.debug(f"Updated success count for {part_number}: {current_count + 1}")

        except Exception as e:
            logger.debug(f"Failed to update success count: {e}")

    async def _call_nexus_memory_api(
        self,
        operation: str,
        query: Optional[str] = None,
        content: Optional[str] = None,
        event_type: Optional[str] = None,
        limit: int = 10
    ) -> Optional[Dict[str, Any]]:
        """
        Call the Nexus Memory API for recall or store operations.

        Args:
            operation: "recall" or "store"
            query: Search query for recall
            content: Content to store
            event_type: Type of event for storage
            limit: Max results for recall

        Returns:
            API response dict or None on failure
        """
        api_url = os.environ.get("NEXUS_API_URL", "https://api.adverant.ai")
        api_key = os.environ.get("NEXUS_API_KEY")

        if not api_key:
            logger.debug("NEXUS_API_KEY not set, skipping nexus-memory operation")
            return None

        try:
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "X-Company-ID": os.environ.get("NEXUS_COMPANY_ID", "adverant"),
                "X-App-ID": os.environ.get("NEXUS_APP_ID", "nexus-ee-design")
            }

            if operation == "recall":
                payload = {"query": query, "limit": limit}
            elif operation == "store":
                payload = {
                    "content": content,
                    "event_type": event_type or "context",
                    "tags": ["component", "symbol", "ee-design"]
                }
            else:
                logger.warning(f"Unknown memory operation: {operation}")
                return None

            response = await self.http_client.post(
                f"{api_url}/api/memory",
                json=payload,
                headers=headers,
                timeout=10.0
            )

            if response.status_code == 200:
                return response.json()
            else:
                logger.warning(f"Nexus memory API error: {response.status_code} - {response.text[:200]}")
                return None

        except Exception as e:
            logger.warning(f"Nexus memory API call failed: {e}")
            return None

    # =========================================================================
    # END NEXUS-MEMORY INTEGRATION
    # =========================================================================

    def _normalize_part_number(self, part_number: str) -> str:
        """Normalize part number for fuzzy matching."""
        # Remove common suffixes and normalize
        normalized = part_number.upper()
        normalized = re.sub(r'[-_\s/]', '', normalized)
        # Remove packaging/ordering suffixes — only if the remaining part is
        # long enough (>= 4 chars) to prevent stripping "ND" from "GND" etc.
        if len(normalized) > 5:
            normalized = re.sub(r'(TR|CT|ND|REEL|TAPE|CUT|BULK)$', '', normalized)
        return normalized

    # Generic symbol mappings for passives
    GENERIC_PASSIVE_SYMBOLS = {
        'Capacitor': 'C',
        'Resistor': 'R',
        'Inductor': 'L',
        'Diode': 'D',
        'LED': 'LED',
        'Fuse': 'Fuse',
        'Crystal': 'Crystal',
        'Ferrite_Bead': 'Ferrite_Bead',
    }

    def _guess_kicad_library(
        self,
        part_number: str,
        manufacturer: Optional[str]
    ) -> List[str]:
        """Guess which KiCad library might contain a part."""
        libraries = []

        pn_upper = part_number.upper()
        mfr_upper = (manufacturer or '').upper()

        # MCU patterns - check specific families first
        if 'STM32G4' in pn_upper:
            libraries.append('MCU_ST_STM32G4')
        elif 'STM32H7' in pn_upper:
            libraries.append('MCU_ST_STM32H7')
        elif 'STM32F4' in pn_upper:
            libraries.append('MCU_ST_STM32F4')
        elif 'STM32F1' in pn_upper:
            libraries.append('MCU_ST_STM32F1')
        elif 'STM32' in pn_upper or 'STM8' in pn_upper:
            libraries.extend(['MCU_ST_STM32G4', 'MCU_ST_STM32H7', 'MCU_ST_STM32F4', 'MCU_ST_STM32', 'MCU_ST_STM8'])

        if any(p in pn_upper for p in ['ATMEGA', 'ATTINY', 'AVR']):
            libraries.append('MCU_Microchip_ATmega')
        if 'ESP' in pn_upper:
            libraries.append('RF_Module')
        if any(p in pn_upper for p in ['PIC', 'DSPIC']):
            libraries.extend(['MCU_Microchip_PIC', 'MCU_Microchip_dsPIC'])
        if 'TMS320' in pn_upper or 'C2000' in pn_upper:
            libraries.append('MCU_Texas_MSP430')  # TI DSP
        if 'TC37' in pn_upper or 'AURIX' in pn_upper:
            libraries.append('MCU_Infineon')

        # Current sensing amplifiers (INA series from TI)
        if pn_upper.startswith('INA') or 'INA240' in pn_upper or 'INA219' in pn_upper:
            libraries.extend(['Sensor_Current', 'Amplifier_Current'])

        # Gate drivers - FOC ESC specific
        if any(p in pn_upper for p in ['UCC21', 'UCC27', 'IR21', 'DRV', 'FAN73']):
            libraries.extend(['Driver_FET', 'Driver_Motor'])
        if 'HIP41' in pn_upper or 'IRS21' in pn_upper:
            libraries.append('Driver_FET')

        # Power management - Linear regulators
        if any(p in pn_upper for p in ['LM78', 'LM79', 'LM317', 'LM1117', 'AMS1117']):
            libraries.append('Regulator_Linear')
        if any(p in pn_upper for p in ['TPS', 'LDO']):
            libraries.append('Regulator_Linear')

        # Power management - Switching regulators
        if any(p in pn_upper for p in ['LTC', 'MAX', 'ADP', 'MP1', 'MP2', 'LM25', 'TPS54', 'TPS56']):
            libraries.append('Regulator_Switching')

        # Ideal diode controllers
        if 'LTC44' in pn_upper:
            libraries.append('Power_Management')

        # Op-amps and amplifiers
        if any(p in pn_upper for p in ['OPA', 'LM358', 'TL0', 'NE5', 'OP0', 'AD8', 'MCP60']):
            libraries.append('Amplifier_Operational')

        # MOSFET and power transistors - FOC ESC specific
        if any(p in pn_upper for p in ['IRF', 'IRFZ', 'IPB', 'BSC', 'EPC', 'GAN', 'IMZA', 'SIC']):
            libraries.append('Transistor_FET')
        if 'IPB200' in pn_upper or 'IMZA65' in pn_upper:
            libraries.append('Transistor_FET')

        # CAN transceivers - FOC ESC specific
        if any(p in pn_upper for p in ['TJA1', 'MCP25', 'SN65HVD', 'ISO1050']):
            libraries.append('Interface_CAN_LIN')

        # Ethernet PHY
        if any(p in pn_upper for p in ['DP83', 'KSZ', 'LAN87', 'RTL8']):
            libraries.append('Interface_Ethernet')

        # Thermistors and NTCs
        if any(p in pn_upper for p in ['NTC', 'NTCLE', 'SL22']):
            libraries.append('Device')

        # TVS and ESD protection
        if any(p in pn_upper for p in ['PRTR', 'PESD', 'TVS', 'ESD', 'SMBJ']):
            libraries.append('Diode')

        # Crystals and oscillators
        if any(p in pn_upper for p in ['ABM', 'XTAL', 'NX3225', 'FA238']):
            libraries.append('Device')

        # Inductors - EMI filters and power
        if any(p in pn_upper for p in ['7448', 'SRP', 'XAL', 'IHLP', 'WE-']):
            libraries.extend(['Inductor_SMD', 'Device'])

        # Passives - Capacitors
        if pn_upper.startswith(('C', 'GRM', 'CL', 'CC', 'EMK', 'C5750', 'C3216', 'C2012', 'C1608')):
            libraries.append('Device')

        # Passives - Resistors (check prefixes that indicate resistors)
        if pn_upper.startswith(('R', 'RC', 'ERJ', 'CRCW', 'WSL', '0603W', '0402W', '0805W')):
            libraries.append('Device')

        # Passives - Inductors
        if pn_upper.startswith(('L', 'SRR', 'SRP', 'XAL', 'IHLP')):
            libraries.append('Inductor_SMD')

        # Diodes
        if pn_upper.startswith(('1N', 'BAS', 'BAT', 'SS', 'ES1', 'MBR')):
            libraries.append('Diode')

        # Connectors
        if any(p in pn_upper for p in ['CONN', 'PIN', 'HDR', 'JST', 'MOLEX', 'USB']):
            libraries.extend(['Connector_Generic', 'Connector_USB'])

        # Power symbols
        if pn_upper in ['VCC', 'GND', 'VDD', 'VSS', 'VBAT', '+3V3', '+5V', '+12V']:
            libraries.append('power')

        # Manufacturer-based guessing
        if 'INFINEON' in mfr_upper:
            libraries.extend(['Transistor_FET', 'MCU_Infineon', 'Driver_FET'])
        if 'TI' in mfr_upper or 'TEXAS' in mfr_upper:
            libraries.extend(['Amplifier_Operational', 'Regulator_Linear', 'Driver_FET', 'Interface_CAN_LIN'])
        if 'ST' in mfr_upper or 'STMICRO' in mfr_upper:
            libraries.extend(['MCU_ST_STM32G4', 'MCU_ST_STM32H7', 'MCU_ST_STM32F4'])
        if 'NXP' in mfr_upper:
            libraries.extend(['Interface_CAN_LIN', 'MCU_NXP'])
        if 'ANALOG' in mfr_upper or 'ADI' in mfr_upper:
            libraries.extend(['Amplifier_Operational', 'Regulator_Switching'])

        # Default fallback with most common libraries
        if not libraries:
            libraries = [
                'Device',
                'Transistor_FET',
                'MCU_ST_STM32G4',
                'Driver_FET',
                'Amplifier_Operational',
                'Sensor_Current',
                'Interface_CAN_LIN'
            ]

        # Remove duplicates while preserving order
        seen = set()
        unique_libraries = []
        for lib in libraries:
            if lib not in seen:
                seen.add(lib)
                unique_libraries.append(lib)

        return unique_libraries

    def _flatten_inherited_symbol(
        self,
        library_content: str,
        symbol_sexp: str,
        symbol_name: str
    ) -> str:
        """
        Flatten a symbol that uses (extends "parent") inheritance.

        KiCanvas doesn't support the extends keyword, so we need to inline
        the parent symbol's graphics and pins into the child symbol.
        """
        # Check if symbol uses extends
        extends_match = re.search(r'\(extends\s+"([^"]+)"\)', symbol_sexp)
        if not extends_match:
            return symbol_sexp  # No inheritance, return as-is

        parent_name = extends_match.group(1)
        logger.info(f"Symbol '{symbol_name}' extends '{parent_name}', flattening...")

        # Extract parent symbol from library
        parent_pattern = rf'\(symbol\s+"{re.escape(parent_name)}"\s+'
        parent_match = re.search(parent_pattern, library_content)

        if not parent_match:
            logger.warning(f"Parent symbol '{parent_name}' not found, returning child as-is")
            # Remove extends clause since KiCanvas can't handle it
            return re.sub(r'\s*\(extends\s+"[^"]+"\)', '', symbol_sexp)

        # Extract parent symbol S-expression
        start_pos = parent_match.start()
        depth = 0
        end_pos = start_pos
        for i, char in enumerate(library_content[start_pos:]):
            if char == '(':
                depth += 1
            elif char == ')':
                depth -= 1
                if depth == 0:
                    end_pos = start_pos + i + 1
                    break
        parent_sexp = library_content[start_pos:end_pos]

        # Extract child's properties (Reference, Value, Footprint, etc.)
        child_properties = []
        for prop_match in re.finditer(r'\(property\s+"[^"]+"\s+[^)]+\)(?:\s*\([^)]+\))*', symbol_sexp):
            child_properties.append(prop_match.group(0))

        # Extract parent's sub-symbols (graphics and pins like _0_1, _1_1)
        parent_subsymbols = []
        for subsym_match in re.finditer(r'\(symbol\s+"' + re.escape(parent_name) + r'_\d+_\d+"[^)]*\)(?:\s*[^)]+)*\)', parent_sexp):
            # Need to extract the full sub-symbol with balanced parens
            sub_start = subsym_match.start()
            sub_depth = 0
            sub_end = sub_start
            for i, char in enumerate(parent_sexp[sub_start:]):
                if char == '(':
                    sub_depth += 1
                elif char == ')':
                    sub_depth -= 1
                    if sub_depth == 0:
                        sub_end = sub_start + i + 1
                        break
            subsym = parent_sexp[sub_start:sub_end]
            # Rename subsymbol from parent name to child name
            renamed = subsym.replace(f'"{parent_name}_', f'"{symbol_name}_')
            parent_subsymbols.append(renamed)

        # If no subsymbols found with regex, try a different approach
        if not parent_subsymbols:
            # Find all (symbol "ParentName_X_Y" ...) blocks
            pattern = rf'\(symbol\s+"{re.escape(parent_name)}_(\d+)_(\d+)"'
            for m in re.finditer(pattern, parent_sexp):
                sub_start = m.start()
                sub_depth = 0
                sub_end = sub_start
                for i, char in enumerate(parent_sexp[sub_start:]):
                    if char == '(':
                        sub_depth += 1
                    elif char == ')':
                        sub_depth -= 1
                        if sub_depth == 0:
                            sub_end = sub_start + i + 1
                            break
                subsym = parent_sexp[sub_start:sub_end]
                renamed = subsym.replace(f'"{parent_name}_', f'"{symbol_name}_')
                parent_subsymbols.append(renamed)

        # Build the flattened symbol
        # Get other attributes from child (in_bom, on_board, etc.)
        attrs_match = re.search(r'\(symbol\s+"[^"]+"\s*((?:\([^)]+\)\s*)*)', symbol_sexp)
        attrs = ''
        if attrs_match:
            attrs_raw = attrs_match.group(1)
            # Remove extends from attrs
            attrs = re.sub(r'\s*\(extends\s+"[^"]+"\)', '', attrs_raw)

        # Build final flattened symbol
        flattened = f'(symbol "{symbol_name}" {attrs}\n'

        # Add properties
        for prop in child_properties:
            flattened += f'    {prop}\n'

        # Add parent's graphics and pins
        for subsym in parent_subsymbols:
            flattened += f'    {subsym}\n'

        flattened += '  )'

        logger.info(f"Flattened '{symbol_name}' with {len(parent_subsymbols)} sub-symbols from parent '{parent_name}'")
        return flattened

    def _extract_symbol_from_library(
        self,
        library_content: str,
        part_number: str
    ) -> Optional[str]:
        """Extract a specific symbol from a library file, flattening inheritance."""
        # Try exact match first
        exact_pattern = rf'\(symbol\s+"{re.escape(part_number)}"\s+'
        match = re.search(exact_pattern, library_content)

        # If no exact match, try partial match (symbol name contains part_number)
        if not match:
            partial_pattern = rf'\(symbol\s+"([^"]*{re.escape(part_number)}[^"]*)"\s+'
            match = re.search(partial_pattern, library_content, re.IGNORECASE)

        if not match:
            return None

        # Extract the full symbol S-expression
        start_pos = match.start()
        depth = 0
        end_pos = start_pos

        for i, char in enumerate(library_content[start_pos:]):
            if char == '(':
                depth += 1
            elif char == ')':
                depth -= 1
                if depth == 0:
                    end_pos = start_pos + i + 1
                    break

        symbol_sexp = library_content[start_pos:end_pos]

        # Flatten any inheritance (extends "parent") for KiCanvas compatibility
        symbol_sexp = self._flatten_inherited_symbol(library_content, symbol_sexp, part_number)

        # Wrap in library format
        return f'(kicad_symbol_lib (version 20231120) (generator nexus_ee_design)\n  {symbol_sexp}\n)'

    def _create_placeholder_symbol(
        self,
        part_number: str,
        manufacturer: Optional[str],
        category: str
    ) -> FetchedSymbol:
        """
        Create a category-appropriate placeholder symbol when all else fails.

        IMPORTANT: Placeholder symbols are NOT production-ready. The schematic
        validation pipeline should flag or reject schematics containing placeholders.

        Unlike the old generic 4-pin box, this creates symbols with correct pin
        names, types, and positions for each component category, which allows
        downstream agents (connection generator, wire router) to make meaningful
        connections.

        The placeholder is visually marked with "[PLACEHOLDER]" in the Value field
        to make it obvious in schematic editors that it needs replacement.
        """
        ref_prefix = {
            'MCU': 'U', 'MOSFET': 'Q', 'Gate_Driver': 'U', 'OpAmp': 'U',
            'Capacitor': 'C', 'Resistor': 'R', 'Inductor': 'L',
            'Connector': 'J', 'Power': 'U', 'Other': 'U',
            'CAN_Transceiver': 'U', 'Regulator': 'U', 'Diode': 'D',
            'LED': 'D', 'Crystal': 'Y', 'Fuse': 'F', 'BJT': 'Q',
            'Current_Sense': 'U', 'ESD_Protection': 'U',
        }.get(category, 'U')

        # Refine category for subcategories with specialized pin templates
        effective_category = category
        if category == "Connector" and "USB" in part_number.upper():
            effective_category = "USB_Connector"
            logger.info(
                f"Refined category for {part_number}: "
                f"'{category}' → '{effective_category}' (USB connector detected)"
            )

        # Get category-appropriate pin template
        default_pins = [
            ("1", "1", "passive", "L"), ("2", "2", "passive", "L"),
            ("3", "3", "passive", "R"), ("4", "4", "passive", "R"),
        ]
        pin_template = CATEGORY_PIN_TEMPLATES.get(effective_category, default_pins)
        if effective_category not in CATEGORY_PIN_TEMPLATES:
            logger.error(
                f"NO PIN TEMPLATE for category '{effective_category}' "
                f"(part: {part_number}). Using generic 4-pin placeholder. "
                f"Available categories: {sorted(CATEGORY_PIN_TEMPLATES.keys())}"
            )

        # Group pins by side for sizing and positioning
        sides: Dict[str, List[Tuple[str, str, str, str]]] = {
            "L": [], "R": [], "T": [], "B": []
        }
        for pin_def in pin_template:
            side = pin_def[3]
            sides[side].append(pin_def)

        # Calculate rectangle dimensions to fit all pins
        max_lr = max(len(sides["L"]), len(sides["R"]), 1)
        max_tb = max(len(sides["T"]), len(sides["B"]), 1)
        spacing = 2.54

        rect_half_h = max(max_lr * spacing / 2 + spacing, 5.08)
        rect_half_w = max(max_tb * spacing / 2 + spacing, 5.08)

        # Generate pin S-expressions with correct positions based on side
        pins_sexp_lines = []
        for pin_def in pin_template:
            pin_name, pin_number, pin_type_str, side = pin_def

            # Find this pin's index within its side group
            side_pins = sides[side]
            idx = next(j for j, p in enumerate(side_pins) if p[1] == pin_number)
            count = len(side_pins)

            if side == "L":
                x = -(rect_half_w + 2.54)
                y = (count - 1) * spacing / 2 - idx * spacing
                angle = 0
            elif side == "R":
                x = (rect_half_w + 2.54)
                y = (count - 1) * spacing / 2 - idx * spacing
                angle = 180
            elif side == "T":
                x = -(count - 1) * spacing / 2 + idx * spacing
                y = (rect_half_h + 2.54)
                angle = 270
            else:  # B
                x = -(count - 1) * spacing / 2 + idx * spacing
                y = -(rect_half_h + 2.54)
                angle = 90

            pins_sexp_lines.append(
                f'      (pin {pin_type_str} line (at {x:.2f} {y:.2f} {angle}) (length 2.54)\n'
                f'        (name "{pin_name}" (effects (font (size 1.27 1.27))))\n'
                f'        (number "{pin_number}" (effects (font (size 1.27 1.27))))\n'
                f'      )'
            )

        pins_str = "\n".join(pins_sexp_lines)

        # Mark the value clearly as a placeholder so it's visible in schematic editors
        placeholder_value = f"{part_number} [PLACEHOLDER]"

        placeholder_sexp = f'''(kicad_symbol_lib (version 20231120) (generator nexus_ee_design)
  (symbol "{part_number}" (in_bom yes) (on_board yes)
    (property "Reference" "{ref_prefix}" (at 0 {rect_half_h + 3.81:.2f} 0)
      (effects (font (size 1.27 1.27)))
    )
    (property "Value" "{placeholder_value}" (at 0 {-rect_half_h - 3.81:.2f} 0)
      (effects (font (size 1.27 1.27)))
    )
    (property "Footprint" "" (at 0 0 0)
      (effects (font (size 1.27 1.27)) hide)
    )
    (property "Datasheet" "~" (at 0 0 0)
      (effects (font (size 1.27 1.27)) hide)
    )
    (property "PLACEHOLDER_WARNING" "This symbol is a placeholder - replace with real component" (at 0 0 0)
      (effects (font (size 1.27 1.27)) hide)
    )
    (symbol "{part_number}_0_1"
      (rectangle (start {-rect_half_w:.2f} {rect_half_h:.2f}) (end {rect_half_w:.2f} {-rect_half_h:.2f})
        (stroke (width 0.254) (type default))
        (fill (type background))
      )
      (text "?" (at 0 0 0)
        (effects (font (size 3.81 3.81)))
      )
    )
    (symbol "{part_number}_1_1"
{pins_str}
    )
  )
)'''

        logger.warning(
            f"Created PLACEHOLDER symbol for {part_number} ({category}) "
            f"with {len(pin_template)} category-appropriate pins. "
            f"This symbol is NOT production-ready and should be replaced."
        )

        return FetchedSymbol(
            part_number=part_number,
            manufacturer=manufacturer,
            symbol_sexp=placeholder_sexp,
            source=SymbolSource.LLM_GENERATED,  # Use LLM_GENERATED for backwards compat
            needs_review=True,
            metadata={
                "is_placeholder": True,
                "is_generic": True,  # Flag for quality detection
                "pin_template_used": True,
                "category": category,
                "pin_count": len(pin_template),
                "reason": "All sources failed, created category-appropriate placeholder",
                "warning": "This placeholder must be replaced with a real symbol before production"
            }
        )

    async def search_symbols(
        self,
        query: str,
        category: Optional[str] = None,
        limit: int = 10
    ) -> List[SymbolSearchResult]:
        """
        Search for symbols across all sources.

        Args:
            query: Search query (part number, description, etc.)
            category: Optional category filter
            limit: Maximum results to return

        Returns:
            List of search results
        """
        results = []

        # Search local cache first
        for cat_dir in self.cache_path.iterdir():
            if cat_dir.is_dir():
                if category and cat_dir.name != category:
                    continue

                for meta_file in cat_dir.glob("*.meta.json"):
                    try:
                        metadata = json.loads(meta_file.read_text())
                        if (query.lower() in metadata.get('part_number', '').lower() or
                            query.lower() in metadata.get('description', '').lower()):
                            results.append(SymbolSearchResult(
                                part_number=metadata['part_number'],
                                manufacturer=metadata.get('manufacturer', ''),
                                description=metadata.get('description', ''),
                                category=cat_dir.name,
                                source=SymbolSource.LOCAL_CACHE,
                                has_footprint=metadata.get('has_footprint', False)
                            ))
                    except Exception:
                        continue

        # Search GraphRAG if available
        if self.graphrag and len(results) < limit:
            try:
                graphrag_results = await self.graphrag.search(
                    query=query,
                    collection="kicad_symbols",
                    filter={"category": category} if category else None,
                    limit=limit - len(results)
                )

                for r in graphrag_results:
                    results.append(SymbolSearchResult(
                        part_number=r.get('part_number', ''),
                        manufacturer=r.get('manufacturer', ''),
                        description=r.get('description', ''),
                        category=r.get('category', 'Other'),
                        source=SymbolSource(r.get('source', 'local_cache')),
                        score=r.get('score', 1.0)
                    ))
            except Exception as e:
                logger.warning(f"GraphRAG search failed: {e}")

        return results[:limit]

    async def close(self):
        """Close HTTP client."""
        await self.http_client.aclose()


# CLI entry point
if __name__ == "__main__":
    import sys

    async def main():
        if len(sys.argv) < 2:
            print("Usage: python symbol_fetcher_agent.py <part_number> [manufacturer] [category]")
            sys.exit(1)

        part_number = sys.argv[1]
        manufacturer = sys.argv[2] if len(sys.argv) > 2 else None
        category = sys.argv[3] if len(sys.argv) > 3 else "Other"

        fetcher = SymbolFetcherAgent()

        try:
            print(f"\nFetching symbol for: {part_number}")
            print(f"Manufacturer: {manufacturer or 'Unknown'}")
            print(f"Category: {category}")
            print("-" * 50)

            result = await fetcher.fetch_symbol(part_number, manufacturer, category)

            print(f"\nResult:")
            print(f"  Source: {result.source.value}")
            print(f"  Part Number: {result.part_number}")
            print(f"  Manufacturer: {result.manufacturer or 'Unknown'}")
            print(f"  Datasheet: {result.datasheet_url or 'N/A'}")
            print(f"  Needs Review: {result.needs_review}")
            print(f"\nSymbol S-expression (first 500 chars):")
            print(result.symbol_sexp[:500])

            if result.footprint_sexp:
                print(f"\nFootprint available: Yes")

        finally:
            await fetcher.close()

    asyncio.run(main())
