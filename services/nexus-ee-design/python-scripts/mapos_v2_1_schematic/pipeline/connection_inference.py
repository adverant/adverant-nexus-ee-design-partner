"""
MAPO v2.1 Schematic - Memory-Enhanced Connection Generator

Wraps ConnectionGeneratorAgent with wiring pattern learning.
Learns which connection patterns work for which component combinations,
enabling smarter connection generation over time.

Author: Nexus EE Design Team
"""

import asyncio
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Add parent paths for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from agents.connection_generator import ConnectionGeneratorAgent

from ..core.schematic_state import ComponentInstance, Connection
from ..core.config import SchematicMAPOConfig, get_config
from ..nexus_memory.wiring_memory import WiringMemoryClient, ConnectionPattern

logger = logging.getLogger(__name__)


class MemoryEnhancedConnectionGenerator:
    """
    Memory-enhanced connection generation for MAPO v2.1.
    
    Connection generation workflow:
    1. Recall learned wiring patterns from nexus-memory
    2. Use patterns to seed connection generator
    3. Generate connections using LLM
    4. Store successful patterns after smoke test validation
    """
    
    def __init__(self, config: Optional[SchematicMAPOConfig] = None):
        """Initialize with configuration."""
        self.config = config or get_config()
        self._generator: Optional[ConnectionGeneratorAgent] = None
        self._memory: Optional[WiringMemoryClient] = None
    
    def _ensure_generator(self) -> ConnectionGeneratorAgent:
        """Get or create the connection generator agent."""
        if self._generator is None:
            self._generator = ConnectionGeneratorAgent()
        return self._generator
    
    def _ensure_memory(self) -> WiringMemoryClient:
        """Get or create the memory client."""
        if self._memory is None:
            self._memory = WiringMemoryClient(
                api_key=self.config.nexus_api_key,
                api_url=self.config.nexus_api_url,
            )
        return self._memory
    
    async def close(self):
        """Close clients."""
        if self._memory:
            await self._memory.close()
            self._memory = None
    
    async def generate_connections(
        self,
        components: List[ComponentInstance],
        design_intent: str,
        design_type: str = "foc_esc",
    ) -> List[Connection]:
        """
        Generate connections using memory-enhanced approach.
        
        Args:
            components: Resolved component instances
            design_intent: Natural language design description
            design_type: Type of design for pattern matching
        
        Returns:
            List of Connection objects
        """
        generator = self._ensure_generator()
        
        # Extract component categories
        categories = list(set(c.category for c in components))
        
        # Step 1: Recall learned wiring patterns
        seed_patterns: List[ConnectionPattern] = []
        if self.config.enable_memory:
            memory = self._ensure_memory()
            seed_patterns = await memory.recall_connection_patterns(
                component_categories=categories,
                design_type=design_type,
            )
            if seed_patterns:
                logger.info(f"Recalled {len(seed_patterns)} wiring patterns from memory")
        
        # Build pattern hints for the generator
        pattern_hints = self._extract_pattern_hints(seed_patterns, components)
        
        # Step 2: Convert components to BOM format for generator
        bom_items = [
            {
                "reference": c.reference,
                "part_number": c.part_number,
                "manufacturer": c.manufacturer,
                "category": c.category,
                "value": c.value,
                "pins": [{"name": p.name, "number": p.number, "type": p.pin_type.value} for p in c.pins],
            }
            for c in components
        ]
        
        # Step 3: Generate connections using agent
        # Enhance design intent with pattern hints
        enhanced_intent = design_intent
        if pattern_hints:
            enhanced_intent += f"\n\nLearned connection patterns to follow:\n{pattern_hints}"
        
        raw_connections = await generator.generate_connections(
            bom_items=bom_items,
            design_intent=enhanced_intent,
        )
        
        # Step 4: Convert to Connection objects
        connections = []
        for conn in raw_connections:
            connection = Connection(
                from_ref=conn.get("from_ref", ""),
                from_pin=conn.get("from_pin", ""),
                to_ref=conn.get("to_ref", ""),
                to_pin=conn.get("to_pin", ""),
                net_name=conn.get("net_name"),
                net_class=conn.get("net_class", "Default"),
                connection_type=self._infer_connection_type(conn),
                inferred=conn.get("inferred", True),
            )
            connections.append(connection)
        
        logger.info(f"Generated {len(connections)} connections")
        return connections
    
    def _extract_pattern_hints(
        self,
        patterns: List[ConnectionPattern],
        components: List[ComponentInstance],
    ) -> str:
        """Extract relevant hints from patterns for the generator."""
        if not patterns:
            return ""
        
        # Build component reference map by category
        ref_by_category = {}
        for comp in components:
            if comp.category not in ref_by_category:
                ref_by_category[comp.category] = []
            ref_by_category[comp.category].append(comp.reference)
        
        hints = []
        for pattern in patterns[:3]:  # Limit to top 3 patterns
            if not pattern.smoke_test_passed:
                continue
            
            hints.append(f"Pattern ({pattern.design_type}):")
            for conn in pattern.connections[:10]:  # Limit connections
                # Try to map generic references to actual components
                from_cat = conn.get("from_category", "")
                to_cat = conn.get("to_category", "")
                
                if from_cat and from_cat in ref_by_category:
                    from_ref = ref_by_category[from_cat][0] if ref_by_category[from_cat] else conn.get("from_ref", "?")
                else:
                    from_ref = conn.get("from_ref", "?")
                
                if to_cat and to_cat in ref_by_category:
                    to_ref = ref_by_category[to_cat][0] if ref_by_category[to_cat] else conn.get("to_ref", "?")
                else:
                    to_ref = conn.get("to_ref", "?")
                
                hints.append(
                    f"  - {from_ref}.{conn.get('from_pin', '?')} -> {to_ref}.{conn.get('to_pin', '?')}"
                )
        
        return "\n".join(hints) if hints else ""
    
    def _infer_connection_type(self, conn: Dict[str, Any]) -> str:
        """Infer connection type from connection properties."""
        net_name = (conn.get("net_name") or "").upper()
        
        # Power connections
        if any(p in net_name for p in ["VCC", "VDD", "VIN", "V+", "5V", "3V3", "12V", "48V"]):
            return "power"
        
        # Ground connections
        if any(g in net_name for g in ["GND", "VSS", "AGND", "DGND", "PGND"]):
            return "ground"
        
        # Default to signal
        return "signal"
    
    async def store_successful_connections(
        self,
        connections: List[Connection],
        components: List[ComponentInstance],
        design_type: str,
        design_intent: str,
        smoke_test_passed: bool,
        validation_score: float = 0.0,
    ) -> bool:
        """
        Store successful connections in memory.
        
        Only stores if smoke_test_passed=True.
        
        Args:
            connections: List of connections
            components: List of components
            design_type: Type of design
            design_intent: Design description
            smoke_test_passed: Whether smoke test passed
            validation_score: Overall validation score
        
        Returns:
            True if stored successfully
        """
        if not self.config.enable_memory or not smoke_test_passed:
            return False
        
        memory = self._ensure_memory()
        
        # Convert connections to dict format
        conn_dicts = [
            {
                "from_ref": c.from_ref,
                "from_pin": c.from_pin,
                "to_ref": c.to_ref,
                "to_pin": c.to_pin,
                "net_name": c.net_name,
                "connection_type": c.connection_type,
            }
            for c in connections
        ]
        
        # Extract categories
        categories = list(set(c.category for c in components))
        
        return await memory.store_successful_wiring(
            connections=conn_dicts,
            component_categories=categories,
            design_type=design_type,
            design_intent=design_intent,
            smoke_test_passed=smoke_test_passed,
            validation_score=validation_score,
        )
    
    async def generate_power_connections(
        self,
        components: List[ComponentInstance],
    ) -> List[Connection]:
        """
        Generate power and ground connections.
        
        Ensures all components have proper power rail connections.
        """
        connections = []
        
        # Find power symbols or create implicit power nets
        power_ref = None
        gnd_ref = None
        
        for comp in components:
            if comp.category == "Power":
                if "VCC" in comp.value.upper() or "VDD" in comp.value.upper():
                    power_ref = comp.reference
                elif "GND" in comp.value.upper():
                    gnd_ref = comp.reference
        
        # Connect power pins of all ICs
        for comp in components:
            if comp.category in ("MCU", "IC", "Gate_Driver", "OpAmp", "Amplifier"):
                for pin in comp.pins:
                    if pin.pin_type.value == "power_in":
                        # Determine which power rail
                        if "VCC" in pin.name.upper() or "VDD" in pin.name.upper():
                            connections.append(Connection(
                                from_ref=comp.reference,
                                from_pin=pin.name,
                                to_ref=power_ref or "PWR1",
                                to_pin="VCC",
                                net_name="VCC",
                                connection_type="power",
                                inferred=True,
                            ))
                        elif "GND" in pin.name.upper() or "VSS" in pin.name.upper():
                            connections.append(Connection(
                                from_ref=comp.reference,
                                from_pin=pin.name,
                                to_ref=gnd_ref or "PWR2",
                                to_pin="GND",
                                net_name="GND",
                                connection_type="ground",
                                inferred=True,
                            ))
        
        return connections
    
    def validate_connections(
        self,
        connections: List[Connection],
        components: List[ComponentInstance],
    ) -> Tuple[bool, List[str]]:
        """
        Validate connections against components.
        
        Returns:
            Tuple of (valid, issues)
        """
        issues = []
        
        # Build component map
        comp_map = {c.reference: c for c in components}
        
        for conn in connections:
            # Check from_ref exists
            if conn.from_ref not in comp_map:
                issues.append(f"Unknown component: {conn.from_ref}")
                continue
            
            # Check to_ref exists
            if conn.to_ref not in comp_map:
                issues.append(f"Unknown component: {conn.to_ref}")
                continue
            
            # Optionally check pins exist (if pins were resolved)
            from_comp = comp_map[conn.from_ref]
            if from_comp.pins:
                from_pins = [p.name for p in from_comp.pins] + [p.number for p in from_comp.pins]
                if conn.from_pin not in from_pins:
                    issues.append(f"Unknown pin {conn.from_pin} on {conn.from_ref}")
        
        return len(issues) == 0, issues
