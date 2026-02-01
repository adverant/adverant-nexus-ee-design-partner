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


class SymbolSource(Enum):
    """Available symbol sources in priority order."""
    LOCAL_CACHE = "local_cache"
    KICAD_LOCAL_INSTALL = "kicad_local_install"  # Local KiCad installation
    KICAD_OFFICIAL = "kicad_official"
    SNAPEDA = "snapeda"
    ULTRA_LIBRARIAN = "ultralibrarian"
    TI_WEBENCH = "ti_webench"
    STM_CUBE = "stm_cube"
    INFINEON = "infineon"
    ANALOG_DEVICES = "analog_devices"
    LLM_GENERATED = "llm_generated"


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
        source=SymbolSource.KICAD_LOCAL_INSTALL,
        priority=2,  # After local cache, before GitLab
    ),
    SymbolSourceConfig(
        source=SymbolSource.KICAD_OFFICIAL,
        priority=3,
        api_url="https://gitlab.com/api/v4/projects/kicad%2Fkicad-symbols"
    ),
    SymbolSourceConfig(
        source=SymbolSource.SNAPEDA,
        priority=4,
        api_url="https://www.snapeda.com/api/v1",
        requires_auth=False,  # Basic search works without auth
        api_key_env="SNAPEDA_API_KEY"  # Optional: for higher rate limits
    ),
    SymbolSourceConfig(
        source=SymbolSource.ULTRA_LIBRARIAN,
        priority=5,
        api_url="https://app.ultralibrarian.com/api/v1",
        requires_auth=True,
        api_key_env="ULTRALIBRARIAN_API_KEY"
    ),
    SymbolSourceConfig(
        source=SymbolSource.TI_WEBENCH,
        priority=6,
        api_url="https://www.ti.com/lit/ds"
    ),
    SymbolSourceConfig(
        source=SymbolSource.STM_CUBE,
        priority=6,
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
        self.http_client = httpx.AsyncClient(timeout=30.0, follow_redirects=True)

        # Initialize Anthropic client for LLM generation via OpenRouter
        self.anthropic_client = None
        if anthropic:
            # Prefer OpenRouter API key, fallback to direct Anthropic API key
            openrouter_key = os.environ.get("OPENROUTER_API_KEY")
            anthropic_key = anthropic_api_key or os.environ.get("ANTHROPIC_API_KEY")

            if openrouter_key:
                # Use OpenRouter for LLM (supports Claude via their API)
                self.anthropic_client = anthropic.Anthropic(
                    api_key=openrouter_key,
                    base_url="https://openrouter.ai/api/v1"
                )
            elif anthropic_key:
                # Fallback to direct Anthropic API
                self.anthropic_client = anthropic.Anthropic(api_key=anthropic_key)
            else:
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
        category: str = "Other"
    ) -> FetchedSymbol:
        """
        Fetch symbol using priority fallback chain.

        Args:
            part_number: Component part number (e.g., "STM32G431CBT6")
            manufacturer: Optional manufacturer name
            category: Component category for organization

        Returns:
            FetchedSymbol with KiCad S-expression data
        """
        logger.info(f"Fetching symbol: {part_number} (manufacturer: {manufacturer})")

        # Sort sources by priority, excluding LLM_GENERATED (handled separately as last resort)
        sorted_sources = sorted(
            [s for s in self.sources if s.source != SymbolSource.LLM_GENERATED],
            key=lambda s: s.priority
        )

        for source_config in sorted_sources:
            try:
                result = await self._fetch_from_source(
                    source_config, part_number, manufacturer, category
                )
                if result:
                    logger.info(f"Found symbol from {source_config.source.value}")

                    # Cache the symbol
                    await self._cache_symbol(result, category)

                    # Index in GraphRAG if available
                    if self.graphrag:
                        await self._index_in_graphrag(result, category)

                    return result

            except Exception as e:
                logger.warning(
                    f"Source {source_config.source.value} failed for {part_number}: {e}"
                )
                continue

        # For passives, try generic symbol before LLM generation
        generic_symbol = self.GENERIC_PASSIVE_SYMBOLS.get(category)
        if generic_symbol:
            result = await self._fetch_generic_passive(generic_symbol, part_number, category)
            if result:
                logger.info(f"Using generic symbol {generic_symbol} for {part_number}")
                await self._cache_symbol(result, category)
                return result

        # All sources exhausted - generate with LLM as last resort
        logger.warning(f"All sources exhausted for {part_number}, generating with LLM")
        return await self._generate_symbol_with_llm(part_number, manufacturer, category)

    async def _fetch_generic_passive(
        self,
        generic_symbol: str,
        original_part: str,
        category: str
    ) -> Optional[FetchedSymbol]:
        """
        Fetch a generic passive symbol (C, R, L) from Device library.

        These are standard KiCad symbols for passives that are always available.
        Falls back to GitHub if no local KiCad installation.
        """
        # First try local KiCad installation
        kicad_paths = _get_kicad_install_paths()
        lib_content = None

        for kicad_path in kicad_paths:
            device_lib = kicad_path / "Device.kicad_sym"
            if device_lib.exists():
                try:
                    lib_content = device_lib.read_text(encoding='utf-8')
                    logger.debug(f"Using local Device library: {device_lib}")
                    break
                except Exception as e:
                    logger.debug(f"Error reading local Device library: {e}")

        # If no local library, try fetching from GitHub and caching
        if not lib_content:
            cached_device_lib = self.cache_path / "_libraries" / "Device.kicad_sym"

            # Check cache first
            if cached_device_lib.exists():
                try:
                    lib_content = cached_device_lib.read_text(encoding='utf-8')
                    logger.debug("Using cached Device library from GitHub")
                except Exception as e:
                    logger.debug(f"Error reading cached Device library: {e}")

            # Fetch from GitHub if not in cache
            if not lib_content:
                try:
                    logger.info("Fetching Device library from GitHub (no local KiCad)")
                    url = "https://raw.githubusercontent.com/KiCad/kicad-symbols/master/Device.kicad_sym"
                    response = await self.http_client.get(url, timeout=30.0)
                    if response.status_code == 200:
                        lib_content = response.text
                        # Cache for future use
                        cached_device_lib.parent.mkdir(parents=True, exist_ok=True)
                        cached_device_lib.write_text(lib_content)
                        logger.info("Cached Device library from GitHub")
                except Exception as e:
                    logger.warning(f"Failed to fetch Device library from GitHub: {e}")

        if not lib_content:
            return None

        # Try to find the generic symbol
        try:
            symbol_sexp = self._extract_symbol_from_library(lib_content, generic_symbol)
            if symbol_sexp:
                logger.info(f"Found generic symbol '{generic_symbol}' for {original_part}")
                return FetchedSymbol(
                    part_number=original_part,  # Keep original part number
                    manufacturer=None,
                    symbol_sexp=symbol_sexp,
                    source=SymbolSource.KICAD_OFFICIAL,
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
            SymbolSource.KICAD_OFFICIAL: self._fetch_from_kicad_official,
            SymbolSource.SNAPEDA: self._fetch_from_snapeda,
            SymbolSource.ULTRA_LIBRARIAN: self._fetch_from_ultralibrarian,
            SymbolSource.TI_WEBENCH: self._fetch_from_ti,
            SymbolSource.STM_CUBE: self._fetch_from_stm,
            SymbolSource.LLM_GENERATED: self._generate_symbol_with_llm,
        }

        handler = handlers.get(source_config.source)
        if handler:
            if source_config.source in (SymbolSource.LOCAL_CACHE, SymbolSource.KICAD_LOCAL_INSTALL):
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
        normalized = self._normalize_part_number(part_number)
        for cat_dir in self.cache_path.iterdir():
            if cat_dir.is_dir():
                for symbol_file in cat_dir.glob("*.kicad_sym"):
                    if normalized in self._normalize_part_number(symbol_file.stem):
                        symbol_sexp = symbol_file.read_text()
                        return FetchedSymbol(
                            part_number=symbol_file.stem,
                            manufacturer=manufacturer,
                            symbol_sexp=symbol_sexp,
                            source=SymbolSource.LOCAL_CACHE,
                            metadata={'fuzzy_match': True, 'original_query': part_number}
                        )

        return None

    # Libraries to skip during broad search (simulation, power symbols, etc.)
    SKIP_LIBRARIES = {
        'Simulation_SPICE',
        'power',
        'Graphic',
        'Mechanical',
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

    def _match_kicad_symbol_name(self, query: str, symbol_name: str) -> bool:
        """
        Check if a query matches a KiCad symbol name.

        KiCad uses 'x' as a wildcard for package variants in symbol names.
        For example: STM32G431CBTxZ matches STM32G431CBT6, STM32G431CBT7, etc.

        Also handles common suffixes like temperature grades.
        """
        # Reject very short symbol names (avoids false positives like "0", "1", "R", etc.)
        if len(symbol_name) < 4:
            return False

        # Normalize both strings
        query_norm = self._normalize_part_number(query)
        symbol_norm = self._normalize_part_number(symbol_name)

        # Reject if normalized symbol is too short
        if len(symbol_norm) < 4:
            return False

        # Exact match
        if query_norm == symbol_norm:
            return True

        # Simple containment - but require minimum overlap
        if len(query_norm) >= 6 and query_norm in symbol_norm:
            return True
        if len(symbol_norm) >= 6 and symbol_norm in query_norm:
            return True

        # Convert KiCad 'x' wildcards to regex pattern
        # 'x' in symbol name can match any character in query
        pattern = symbol_norm.replace('X', '.')  # X was uppercased during normalization
        try:
            if re.fullmatch(pattern, query_norm):
                return True
        except re.error:
            pass

        # Handle common variations: T6, T7, TX, TxZ, TXZ all refer to package variants
        # Strip trailing package variant suffixes and compare
        query_base = re.sub(r'(T[0-9]|TX|TXZ)$', 'T', query_norm)
        symbol_base = re.sub(r'(T[0-9]|TX|TXZ)$', 'T', symbol_norm)
        if query_base == symbol_base:
            return True

        # Strip more aggressively - remove all trailing variant indicators
        # STM32G431CB matches STM32G431CBTxZ
        query_core = re.sub(r'[A-Z]?[0-9X]+$', '', query_norm)
        symbol_core = re.sub(r'[A-Z]?[0-9X]+$', '', symbol_norm)
        if len(query_core) >= 8 and query_core == symbol_core:
            return True

        # Levenshtein-like simple distance check for very similar names
        # This helps with minor typos or suffix differences
        if len(query_norm) >= 8 and len(symbol_norm) >= 8:
            # Check if they share the same base (first 8 chars)
            if query_norm[:8] == symbol_norm[:8]:
                # And differ only in the suffix
                diff_len = abs(len(query_norm) - len(symbol_norm))
                if diff_len <= 3:
                    return True

        # Handle similar part numbers with slight variations
        # UCC21530 should match UCC21520 (same family, different model)
        if len(query_norm) >= 6 and len(symbol_norm) >= 6:
            # Check if first 6 chars match and last char is a digit
            if query_norm[:6] == symbol_norm[:6]:
                return True

        return False

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

        Uses datasheet context if available.
        """
        if not self.anthropic_client:
            raise RuntimeError(
                "Anthropic client required for LLM symbol generation. "
                "Install anthropic package and provide API key."
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
            response = self.anthropic_client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=4096,
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
                    "generated_by": "claude-sonnet-4-20250514",
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

    def _normalize_part_number(self, part_number: str) -> str:
        """Normalize part number for fuzzy matching."""
        # Remove common suffixes and normalize
        normalized = part_number.upper()
        normalized = re.sub(r'[-_\s]', '', normalized)
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

    def _extract_symbol_from_library(
        self,
        library_content: str,
        part_number: str
    ) -> Optional[str]:
        """Extract a specific symbol from a library file."""
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

        # Wrap in library format
        return f'(kicad_symbol_lib (version 20231120) (generator nexus_ee_design)\n  {symbol_sexp}\n)'

    def _create_placeholder_symbol(
        self,
        part_number: str,
        manufacturer: Optional[str],
        category: str
    ) -> FetchedSymbol:
        """Create a minimal placeholder symbol when all else fails."""
        ref_prefix = {
            'MCU': 'U', 'MOSFET': 'Q', 'Gate_Driver': 'U', 'OpAmp': 'U',
            'Capacitor': 'C', 'Resistor': 'R', 'Inductor': 'L',
            'Connector': 'J', 'Power': 'U', 'Other': 'U'
        }.get(category, 'U')

        placeholder_sexp = f'''(kicad_symbol_lib (version 20231120) (generator nexus_ee_design)
  (symbol "{part_number}" (in_bom yes) (on_board yes)
    (property "Reference" "{ref_prefix}" (at 0 1.27 0)
      (effects (font (size 1.27 1.27)))
    )
    (property "Value" "{part_number}" (at 0 -1.27 0)
      (effects (font (size 1.27 1.27)))
    )
    (property "Footprint" "" (at 0 0 0)
      (effects (font (size 1.27 1.27)) hide)
    )
    (property "Datasheet" "~" (at 0 0 0)
      (effects (font (size 1.27 1.27)) hide)
    )
    (symbol "{part_number}_0_1"
      (rectangle (start -5.08 5.08) (end 5.08 -5.08)
        (stroke (width 0.254) (type default))
        (fill (type background))
      )
    )
    (symbol "{part_number}_1_1"
      (pin passive line (at -7.62 2.54 0) (length 2.54)
        (name "1" (effects (font (size 1.27 1.27))))
        (number "1" (effects (font (size 1.27 1.27))))
      )
      (pin passive line (at -7.62 0 0) (length 2.54)
        (name "2" (effects (font (size 1.27 1.27))))
        (number "2" (effects (font (size 1.27 1.27))))
      )
      (pin passive line (at 7.62 2.54 180) (length 2.54)
        (name "3" (effects (font (size 1.27 1.27))))
        (number "3" (effects (font (size 1.27 1.27))))
      )
      (pin passive line (at 7.62 0 180) (length 2.54)
        (name "4" (effects (font (size 1.27 1.27))))
        (number "4" (effects (font (size 1.27 1.27))))
      )
    )
  )
)'''

        return FetchedSymbol(
            part_number=part_number,
            manufacturer=manufacturer,
            symbol_sexp=placeholder_sexp,
            source=SymbolSource.LLM_GENERATED,
            needs_review=True,
            metadata={
                "is_placeholder": True,
                "category": category,
                "reason": "All sources failed, created minimal placeholder"
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
