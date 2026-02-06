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

from ideation_context import ConnectionInferenceContext, PinConnection, InterfaceDefinition, PowerRail

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
        connection_hints: Optional[ConnectionInferenceContext] = None,
    ) -> List[Connection]:
        """
        Generate connections using memory-enhanced approach.

        Args:
            components: Resolved component instances.
            design_intent: Natural language design description.
            design_type: Type of design for pattern matching.
            connection_hints: Optional connection inference hints from ideation
                context.  When provided, explicit_connections are converted to
                seed connections that anchor the generated netlist,
                interfaces drive structured bus generation, power_rails
                provide power net wiring data, and critical_signals are
                flagged for special routing attention.

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

        # Step 1.5: Build seed connections from ideation hints
        seed_connections: List[Connection] = []
        if connection_hints is not None:
            seed_connections = self._build_seed_connections(connection_hints, components)
            if seed_connections:
                logger.info(
                    f"Built {len(seed_connections)} seed connections from ideation hints"
                )

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
        # Enhance design intent with pattern hints and ideation context
        enhanced_intent = design_intent
        if pattern_hints:
            enhanced_intent += f"\n\nLearned connection patterns to follow:\n{pattern_hints}"

        if connection_hints is not None:
            ideation_supplement = self._format_connection_hints_for_llm(connection_hints)
            if ideation_supplement:
                enhanced_intent += ideation_supplement

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

        # Step 5: Merge seed connections (ideation explicit connections take
        # priority - they are hard constraints from the designer)
        if seed_connections:
            connections = self._merge_seed_connections(seed_connections, connections)

        logger.info(f"Generated {len(connections)} connections")
        return connections

    def _build_seed_connections(
        self,
        hints: ConnectionInferenceContext,
        components: List[ComponentInstance],
    ) -> List[Connection]:
        """Build seed Connection objects from ideation explicit connections,
        interface definitions, and power rail data.

        Seed connections are treated as hard constraints from the designer.
        They are included in the final connection list regardless of what
        the LLM generates, and duplicate connections are de-duplicated
        during merging.

        Args:
            hints: ConnectionInferenceContext from ideation.
            components: Resolved component instances (used to validate
                references and resolve component names to ref designators).

        Returns:
            List of seed Connection objects.
        """
        seeds: List[Connection] = []

        # Build reference lookup maps
        ref_set = {c.reference for c in components}
        # Also map by part_number and lowercase name for fuzzy matching
        pn_to_ref: Dict[str, str] = {}
        name_to_ref: Dict[str, str] = {}
        for c in components:
            if c.part_number:
                pn_to_ref[c.part_number.upper()] = c.reference
            if c.description:
                name_to_ref[c.description.lower()] = c.reference

        def resolve_ref(component_name: str) -> str:
            """Resolve a component name/part_number to a reference designator."""
            if component_name in ref_set:
                return component_name
            upper = component_name.upper()
            if upper in pn_to_ref:
                return pn_to_ref[upper]
            lower = component_name.lower()
            if lower in name_to_ref:
                return name_to_ref[lower]
            # Return as-is; the connection generator will handle unresolved refs
            return component_name

        # 1. Explicit pin-to-pin connections
        for pc in hints.explicit_connections:
            from_ref = resolve_ref(pc.from_component)
            to_ref = resolve_ref(pc.to_component)
            seeds.append(Connection(
                from_ref=from_ref,
                from_pin=pc.from_pin,
                to_ref=to_ref,
                to_pin=pc.to_pin,
                net_name=pc.signal_name or f"Net_{from_ref}_{pc.from_pin}",
                connection_type=pc.signal_type,
                inferred=False,
            ))

        # 2. Interface definitions -> expand into constituent connections
        for iface in hints.interfaces:
            master_ref = resolve_ref(iface.master_component)
            for slave_name in iface.slave_components:
                slave_ref = resolve_ref(slave_name)
                for signal_name, pin_id in iface.pin_mappings.items():
                    net_name = f"{iface.interface_type}_{signal_name}"
                    seeds.append(Connection(
                        from_ref=master_ref,
                        from_pin=pin_id,
                        to_ref=slave_ref,
                        to_pin=signal_name,
                        net_name=net_name,
                        connection_type="signal",
                        inferred=False,
                    ))

        # 3. Power rails -> connect source to each consumer
        for rail in hints.power_rails:
            source_ref = resolve_ref(rail.source_component) if rail.source_component else ""
            for consumer_name in rail.consumer_components:
                consumer_ref = resolve_ref(consumer_name)
                if source_ref:
                    seeds.append(Connection(
                        from_ref=source_ref,
                        from_pin=rail.net_name,
                        to_ref=consumer_ref,
                        to_pin=rail.net_name,
                        net_name=rail.net_name,
                        connection_type="power",
                        inferred=False,
                    ))

        # 4. Ground net connections for all components with ground pins
        for gnd_net in hints.ground_nets:
            for comp in components:
                for pin in comp.pins:
                    pin_name_upper = pin.name.upper()
                    if pin_name_upper in ("GND", "VSS", "AGND", "DGND", "PGND"):
                        seeds.append(Connection(
                            from_ref=comp.reference,
                            from_pin=pin.name,
                            to_ref="GND_SYMBOL",
                            to_pin=gnd_net,
                            net_name=gnd_net,
                            connection_type="ground",
                            inferred=False,
                        ))
                        break  # One ground connection per component per net

        logger.info(
            f"Seed connections built: {len(hints.explicit_connections)} explicit, "
            f"{len(hints.interfaces)} interfaces expanded, "
            f"{len(hints.power_rails)} power rails, "
            f"{len(hints.ground_nets)} ground nets"
        )
        return seeds

    def _format_connection_hints_for_llm(
        self,
        hints: ConnectionInferenceContext,
    ) -> str:
        """Format ideation connection hints as supplementary LLM prompt text.

        This structured text is appended to the design intent so that the
        LLM connection generator has full visibility into the designer's
        explicit connectivity intentions.

        Args:
            hints: ConnectionInferenceContext from ideation.

        Returns:
            Formatted string to append to design_intent, or empty string
            if no meaningful hints exist.
        """
        parts: List[str] = []

        if hints.explicit_connections:
            parts.append("\n\n=== EXPLICIT CONNECTIONS FROM IDEATION ===")
            parts.append("These connections are REQUIRED by the designer:")
            for pc in hints.explicit_connections:
                line = f"  - {pc.from_component}.{pc.from_pin} -> {pc.to_component}.{pc.to_pin}"
                if pc.signal_name:
                    line += f" [{pc.signal_name}]"
                if pc.signal_type != "signal":
                    line += f" ({pc.signal_type})"
                if pc.notes:
                    line += f" -- {pc.notes}"
                parts.append(line)

        if hints.interfaces:
            parts.append("\n=== INTERFACE DEFINITIONS FROM IDEATION ===")
            for iface in hints.interfaces:
                slaves = ", ".join(iface.slave_components) if iface.slave_components else "N/A"
                parts.append(
                    f"  - {iface.interface_type}: {iface.master_component} -> [{slaves}]"
                )
                if iface.speed:
                    parts.append(f"    Speed: {iface.speed}")
                if iface.pin_mappings:
                    for sig, pin in iface.pin_mappings.items():
                        parts.append(f"    {sig}: {pin}")
                if iface.protocol_notes:
                    parts.append(f"    Notes: {iface.protocol_notes}")

        if hints.power_rails:
            parts.append("\n=== POWER RAILS FROM IDEATION ===")
            for rail in hints.power_rails:
                consumers = ", ".join(rail.consumer_components) if rail.consumer_components else "N/A"
                parts.append(
                    f"  - {rail.net_name}: {rail.voltage}V (max {rail.current_max}A) "
                    f"[{rail.regulator_type}] from {rail.source_component} -> [{consumers}]"
                )

        if hints.ground_nets:
            parts.append(f"\n=== GROUND NETS: {', '.join(hints.ground_nets)} ===")

        if hints.critical_signals:
            parts.append(
                f"\n=== CRITICAL SIGNALS (require special routing): "
                f"{', '.join(hints.critical_signals)} ==="
            )

        if hints.design_intent_text:
            parts.append(f"\n=== ADDITIONAL DESIGN INTENT ===\n{hints.design_intent_text}")

        return "\n".join(parts) if parts else ""

    def _merge_seed_connections(
        self,
        seeds: List[Connection],
        generated: List[Connection],
    ) -> List[Connection]:
        """Merge seed connections with LLM-generated connections.

        Seed connections (from ideation) take priority.  If a generated
        connection has the same from_ref+from_pin+to_ref+to_pin tuple as
        a seed, the seed version is kept and the generated one is dropped.

        Args:
            seeds: Seed connections from ideation (hard constraints).
            generated: LLM-generated connections.

        Returns:
            Merged and de-duplicated list of connections.
        """
        # Build a set of seed connection keys for dedup
        seed_keys: set = set()
        for s in seeds:
            key = (s.from_ref, s.from_pin, s.to_ref, s.to_pin)
            seed_keys.add(key)
            # Also add the reverse direction
            seed_keys.add((s.to_ref, s.to_pin, s.from_ref, s.from_pin))

        # Start with seeds, then add non-duplicate generated connections
        merged = list(seeds)
        duplicates_skipped = 0
        for conn in generated:
            key = (conn.from_ref, conn.from_pin, conn.to_ref, conn.to_pin)
            if key not in seed_keys:
                merged.append(conn)
                seed_keys.add(key)
            else:
                duplicates_skipped += 1

        if duplicates_skipped > 0:
            logger.info(
                f"Connection merge: {len(seeds)} seeds + "
                f"{len(generated) - duplicates_skipped} generated "
                f"({duplicates_skipped} duplicates skipped)"
            )

        return merged
    
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
