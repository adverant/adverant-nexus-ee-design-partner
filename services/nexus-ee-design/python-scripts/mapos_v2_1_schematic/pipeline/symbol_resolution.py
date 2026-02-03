"""
MAPO v2.1 Schematic - Memory-Enhanced Symbol Resolution

Wraps SymbolFetcherAgent with nexus-memory learning.
Learns which sources successfully resolve which components,
enabling faster and more accurate symbol fetching over time.

Author: Nexus EE Design Team
"""

import asyncio
import logging
import sys
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Add parent paths for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from agents.symbol_fetcher import SymbolFetcherAgent, FetchedSymbol

from ..core.schematic_state import (
    ComponentInstance,
    Pin,
    PinType,
    SymbolQuality,
)
from ..core.config import SchematicMAPOConfig, get_config
from ..nexus_memory.symbol_memory import SymbolMemoryClient, SymbolResolution

logger = logging.getLogger(__name__)


@dataclass
class ResolvedSymbol:
    """Result of symbol resolution including quality metadata."""
    component: ComponentInstance
    symbol_content: str
    quality: SymbolQuality
    source: str
    from_memory: bool = False
    pins: List[Pin] = None
    
    def __post_init__(self):
        if self.pins is None:
            self.pins = []


class MemoryEnhancedSymbolResolver:
    """
    Memory-enhanced symbol resolution for MAPO v2.1.
    
    Resolution priority:
    1. Nexus-memory (learned resolutions)
    2. KiCad Worker API (official libraries)
    3. SnapEDA API
    4. UltraLibrarian API
    5. Local cache
    6. LLM-generated (last resort)
    """
    
    def __init__(self, config: Optional[SchematicMAPOConfig] = None):
        """Initialize with configuration."""
        self.config = config or get_config()
        self._fetcher: Optional[SymbolFetcherAgent] = None
        self._memory: Optional[SymbolMemoryClient] = None
    
    def _ensure_fetcher(self) -> SymbolFetcherAgent:
        """Get or create the symbol fetcher agent."""
        if self._fetcher is None:
            self._fetcher = SymbolFetcherAgent(
                cache_dir=self.config.symbol_cache_path
            )
        return self._fetcher
    
    def _ensure_memory(self) -> SymbolMemoryClient:
        """Get or create the memory client."""
        if self._memory is None:
            self._memory = SymbolMemoryClient(
                api_key=self.config.nexus_api_key,
                api_url=self.config.nexus_api_url,
            )
        return self._memory
    
    async def close(self):
        """Close clients."""
        if self._memory:
            await self._memory.close()
            self._memory = None
    
    async def resolve_symbol(
        self,
        part_number: str,
        manufacturer: str,
        category: str,
        value: str = "",
        description: str = "",
    ) -> ResolvedSymbol:
        """
        Resolve a single symbol with memory-enhanced lookup.
        
        Args:
            part_number: Component part number
            manufacturer: Component manufacturer
            category: Component category
            value: Component value (e.g., "10uF")
            description: Component description
        
        Returns:
            ResolvedSymbol with quality metadata
        """
        fetcher = self._ensure_fetcher()
        
        # Step 1: Check nexus-memory for learned resolution
        if self.config.enable_memory:
            memory = self._ensure_memory()
            memory_result = await memory.recall_symbol(
                part_number=part_number,
                manufacturer=manufacturer,
                category=category,
            )
            
            if memory_result and memory_result.confidence > 0.7:
                logger.info(f"Using learned resolution for {part_number}: {memory_result.source}")
                
                # Fetch from learned source
                fetched = await fetcher.fetch_symbol(
                    part_number=part_number,
                    manufacturer=manufacturer,
                    category=category,
                    preferred_source=memory_result.source,
                )
                
                if fetched and fetched.quality.value not in ("placeholder", "generated"):
                    # Update success count in memory
                    await memory.update_success_count(part_number, manufacturer)
                    
                    return self._create_resolved_symbol(
                        fetched, part_number, manufacturer, category, value,
                        description, from_memory=True
                    )
        
        # Step 2: Use symbol fetcher agent fallback chain
        fetched = await fetcher.fetch_symbol(
            part_number=part_number,
            manufacturer=manufacturer,
            category=category,
        )
        
        if fetched is None:
            # Create placeholder
            logger.warning(f"No symbol found for {part_number}, using placeholder")
            return self._create_placeholder(
                part_number, manufacturer, category, value, description
            )
        
        # Step 3: Store successful resolution in memory
        if self.config.enable_memory and self.config.memory_store_successful:
            if fetched.quality.value not in ("placeholder", "generated"):
                memory = self._ensure_memory()
                await memory.store_symbol(
                    part_number=part_number,
                    manufacturer=manufacturer,
                    category=category,
                    source=fetched.source,
                    footprint=fetched.footprint,
                    verified=fetched.quality.value == "verified",
                )
        
        return self._create_resolved_symbol(
            fetched, part_number, manufacturer, category, value, description
        )
    
    def _create_resolved_symbol(
        self,
        fetched: FetchedSymbol,
        part_number: str,
        manufacturer: str,
        category: str,
        value: str,
        description: str,
        from_memory: bool = False,
    ) -> ResolvedSymbol:
        """Create ResolvedSymbol from FetchedSymbol."""
        # Map quality
        quality_map = {
            "verified": SymbolQuality.VERIFIED,
            "fetched": SymbolQuality.FETCHED,
            "cached": SymbolQuality.CACHED,
            "generated": SymbolQuality.LLM_GENERATED,
            "placeholder": SymbolQuality.PLACEHOLDER,
        }
        quality = quality_map.get(fetched.quality.value, SymbolQuality.FETCHED)
        
        # Parse pins from symbol content
        pins = self._parse_pins(fetched.symbol_content)
        
        # Generate reference based on category
        ref_prefix = self._get_ref_prefix(category)
        
        # Create component instance
        component = ComponentInstance(
            uuid=str(uuid.uuid4()),
            part_number=part_number,
            manufacturer=manufacturer,
            category=category,
            reference=f"{ref_prefix}?",  # Will be assigned later
            value=value or part_number,
            position=(0, 0),  # Will be set during placement
            rotation=0,
            symbol_id=fetched.lib_id,
            footprint=fetched.footprint,
            quality=quality,
            resolution_source=fetched.source,
            pins=tuple(pins),
            description=description,
        )
        
        return ResolvedSymbol(
            component=component,
            symbol_content=fetched.symbol_content,
            quality=quality,
            source=fetched.source,
            from_memory=from_memory,
            pins=pins,
        )
    
    def _create_placeholder(
        self,
        part_number: str,
        manufacturer: str,
        category: str,
        value: str,
        description: str,
    ) -> ResolvedSymbol:
        """Create placeholder when no symbol found."""
        ref_prefix = self._get_ref_prefix(category)
        
        # Create minimal component
        component = ComponentInstance(
            uuid=str(uuid.uuid4()),
            part_number=part_number,
            manufacturer=manufacturer,
            category=category,
            reference=f"{ref_prefix}?",
            value=value or part_number,
            position=(0, 0),
            rotation=0,
            symbol_id=f"placeholder:{category}",
            footprint="",
            quality=SymbolQuality.PLACEHOLDER,
            resolution_source="none",
            pins=tuple(),
            description=description,
        )
        
        # Generate minimal symbol content
        symbol_content = self._generate_placeholder_symbol(part_number, category)
        
        return ResolvedSymbol(
            component=component,
            symbol_content=symbol_content,
            quality=SymbolQuality.PLACEHOLDER,
            source="placeholder",
            from_memory=False,
            pins=[],
        )
    
    def _get_ref_prefix(self, category: str) -> str:
        """Get reference designator prefix for category."""
        prefix_map = {
            "MCU": "U",
            "IC": "U",
            "MOSFET": "Q",
            "BJT": "Q",
            "Transistor": "Q",
            "Gate_Driver": "U",
            "OpAmp": "U",
            "Amplifier": "U",
            "Capacitor": "C",
            "Resistor": "R",
            "Inductor": "L",
            "Diode": "D",
            "LED": "D",
            "Connector": "J",
            "Power": "U",
            "Regulator": "U",
            "Crystal": "Y",
            "Thermistor": "TH",
            "CAN_Transceiver": "U",
            "TVS": "D",
        }
        return prefix_map.get(category, "U")
    
    def _parse_pins(self, symbol_content: str) -> List[Pin]:
        """Parse pins from KiCad symbol S-expression."""
        pins = []
        
        # Simple parsing - find (pin ...) blocks
        import re
        pin_pattern = r'\(pin\s+(\w+)\s+\w+\s+\(at\s+([\d.-]+)\s+([\d.-]+)\s+(\d+)\)'
        matches = re.findall(pin_pattern, symbol_content)
        
        for i, match in enumerate(matches):
            pin_type_str, x, y, angle = match
            
            # Map KiCad pin type to our enum
            type_map = {
                "input": PinType.INPUT,
                "output": PinType.OUTPUT,
                "bidirectional": PinType.BIDIRECTIONAL,
                "passive": PinType.PASSIVE,
                "power_in": PinType.POWER_IN,
                "power_out": PinType.POWER_OUT,
            }
            pin_type = type_map.get(pin_type_str, PinType.UNSPECIFIED)
            
            # Find pin name and number (simplified)
            name_pattern = rf'\(pin\s+{pin_type_str}.*?\(name\s+"([^"]+)"'
            name_match = re.search(name_pattern, symbol_content)
            name = name_match.group(1) if name_match else f"PIN{i+1}"
            
            number_pattern = rf'\(pin\s+{pin_type_str}.*?\(number\s+"([^"]+)"'
            number_match = re.search(number_pattern, symbol_content)
            number = number_match.group(1) if number_match else str(i+1)
            
            pins.append(Pin(
                name=name,
                number=number,
                pin_type=pin_type,
                position=(float(x), float(y)),
                orientation=int(angle),
            ))
        
        return pins
    
    def _generate_placeholder_symbol(self, part_number: str, category: str) -> str:
        """Generate minimal placeholder symbol."""
        # Basic rectangle with a few generic pins
        return f'''
(symbol "{part_number}_placeholder"
  (property "Reference" "U" (at 0 5.08 0))
  (property "Value" "{part_number}" (at 0 -5.08 0))
  (symbol "{part_number}_placeholder_1_1"
    (rectangle (start -5.08 2.54) (end 5.08 -2.54) (stroke (width 0.254)))
    (pin input line (at -7.62 1.27 0) (length 2.54) (name "IN") (number "1"))
    (pin output line (at 7.62 1.27 180) (length 2.54) (name "OUT") (number "2"))
    (pin power_in line (at 0 5.08 270) (length 2.54) (name "VCC") (number "3"))
    (pin power_in line (at 0 -5.08 90) (length 2.54) (name "GND") (number "4"))
  )
)
'''
    
    async def resolve_symbols(
        self,
        bom_items: List[Dict[str, Any]],
    ) -> Tuple[List[ResolvedSymbol], Dict[str, str]]:
        """
        Resolve symbols for all BOM items.
        
        Args:
            bom_items: List of BOM item dicts
        
        Returns:
            Tuple of (resolved_symbols, lib_symbols_dict)
        """
        # Batch recall from memory first
        if self.config.enable_memory:
            memory = self._ensure_memory()
            memory_results = await memory.batch_recall([
                {
                    "part_number": item.get("part_number", ""),
                    "manufacturer": item.get("manufacturer", ""),
                    "category": item.get("category", ""),
                }
                for item in bom_items
            ])
            logger.info(f"Memory batch recall: {len(memory_results)} found")
        else:
            memory_results = {}
        
        # Resolve symbols with concurrency limit
        semaphore = asyncio.Semaphore(self.config.max_concurrent_symbol_fetches)
        
        async def resolve_one(item: Dict[str, Any]) -> ResolvedSymbol:
            async with semaphore:
                return await self.resolve_symbol(
                    part_number=item.get("part_number", ""),
                    manufacturer=item.get("manufacturer", ""),
                    category=item.get("category", ""),
                    value=item.get("value", ""),
                    description=item.get("description", ""),
                )
        
        tasks = [resolve_one(item) for item in bom_items]
        resolved = await asyncio.gather(*tasks)
        
        # Build lib_symbols dict
        lib_symbols = {}
        for r in resolved:
            if r.symbol_content and r.component.symbol_id not in lib_symbols:
                lib_symbols[r.component.symbol_id] = r.symbol_content
        
        # Assign unique references
        ref_counts = {}
        for r in resolved:
            prefix = self._get_ref_prefix(r.component.category)
            count = ref_counts.get(prefix, 0) + 1
            ref_counts[prefix] = count
            
            # Create new component with assigned reference
            r.component = ComponentInstance(
                uuid=r.component.uuid,
                part_number=r.component.part_number,
                manufacturer=r.component.manufacturer,
                category=r.component.category,
                reference=f"{prefix}{count}",
                value=r.component.value,
                position=r.component.position,
                rotation=r.component.rotation,
                symbol_id=r.component.symbol_id,
                footprint=r.component.footprint,
                quality=r.component.quality,
                resolution_source=r.component.resolution_source,
                pins=r.component.pins,
                description=r.component.description,
            )
        
        # Log summary
        quality_counts = {}
        for r in resolved:
            q = r.quality.value
            quality_counts[q] = quality_counts.get(q, 0) + 1
        logger.info(f"Symbol resolution complete: {quality_counts}")
        
        return resolved, lib_symbols
