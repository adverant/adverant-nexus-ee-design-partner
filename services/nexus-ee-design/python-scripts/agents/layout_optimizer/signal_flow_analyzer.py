"""
Signal Flow Analyzer - Graph-based circuit connectivity analysis.

Analyzes netlist connectivity to determine signal flow paths, component layers,
functional groupings, and placement constraints. Replaces simplistic zone-based
placement with professional signal flow analysis.

Key capabilities:
1. Build directed connectivity graph from netlist
2. Identify signal paths (sources → sinks) with criticality scoring
3. Determine component layers via topological sort
4. Group components by function (power, MCU core, drivers, etc.)
5. Find critical proximity pairs (bypass caps near ICs, etc.)
6. Define separation zones (analog/digital, power/signal)

Author: Nexus EE Design Team
"""

from __future__ import annotations

import logging
import re
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

try:
    import networkx as nx
except ImportError:
    nx = None
    logging.warning("networkx not installed - signal flow analysis will use fallback mode")

try:
    from ideation_context import PlacementContext, SubsystemBlock
except ImportError:
    PlacementContext = None
    SubsystemBlock = None

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class SignalPath:
    """A critical signal path through the schematic."""
    path_id: str
    source_component: str
    sink_components: List[str]
    intermediate_components: List[str]
    net_names: List[str]
    path_type: str  # 'power' | 'clock' | 'high_speed' | 'signal'
    criticality: float  # 0.0-1.0


@dataclass
class ComponentLayer:
    """Components grouped by signal flow layer."""
    layer_id: int
    layer_name: str  # 'input' | 'processing_1' | 'processing_2' | 'output'
    components: List[str]
    x_position_hint: float  # Normalized 0.0-1.0 (left to right)


@dataclass
class FunctionalGroup:
    """Components grouped by function."""
    group_id: str
    group_name: str  # e.g., "Power Supply", "MCU Core", "Motor Driver"
    components: List[str]
    position_hint: Optional[Tuple[float, float]] = None  # (x, y) from ideation


@dataclass
class SignalFlowAnalysis:
    """Complete signal flow analysis results."""
    connectivity_graph: Any  # nx.DiGraph or fallback dict
    signal_paths: List[SignalPath]
    component_layers: List[ComponentLayer]
    functional_groups: List[FunctionalGroup]
    critical_proximity_pairs: List[Tuple[str, str]]  # (component1, component2)
    separation_zones: Dict[str, List[str]]  # zone_type -> component list


# ---------------------------------------------------------------------------
# Signal Flow Analyzer
# ---------------------------------------------------------------------------


class SignalFlowAnalyzer:
    """
    Analyzes netlist connectivity to determine signal flow.

    Uses graph theory (topological sort, path finding) when networkx is
    available, falls back to heuristic analysis otherwise.
    """

    # Component category classifications
    SOURCE_CATEGORIES = {
        "Connector", "Power", "Regulator", "LDO", "Sensor",
        "Crystal", "Oscillator"
    }

    SINK_CATEGORIES = {
        "Connector", "LED", "Display", "Motor", "Speaker"
    }

    IC_CATEGORIES = {
        "MCU", "IC", "Gate_Driver", "Amplifier", "ADC", "DAC",
        "Regulator", "LDO", "Power"
    }

    PASSIVE_CATEGORIES = {
        "Resistor", "Capacitor", "Inductor", "Ferrite"
    }

    # Signal type patterns
    POWER_NET_PATTERNS = [
        r"^V[CDS]{2}$", r"^AVCC$", r"^DVCC$", r"^V[+-]?\d*$",
        r"^VIN$", r"^VOUT$", r"^VBAT$", r"^\+\d+V$", r"^-\d+V$"
    ]

    GROUND_NET_PATTERNS = [
        r"^GND$", r"^AGND$", r"^DGND$", r"^PGND$", r"^GND\d*$"
    ]

    CLOCK_NET_PATTERNS = [
        r".*CLK.*", r".*CLOCK.*", r".*OSC.*", r".*XTAL.*"
    ]

    HIGH_SPEED_NET_PATTERNS = [
        r".*SPI.*", r".*I2C.*", r".*UART.*", r".*USB.*",
        r".*CAN.*", r".*ETH.*", r".*MISO.*", r".*MOSI.*",
        r".*SCK.*", r".*SCL.*", r".*SDA.*"
    ]

    def __init__(self):
        """Initialize the signal flow analyzer."""
        self.use_networkx = nx is not None

    def analyze(
        self,
        netlist: List[Dict],
        bom: List[Dict],
        ideation_context: Optional[Any] = None
    ) -> SignalFlowAnalysis:
        """
        Perform complete signal flow analysis.

        Args:
            netlist: List of nets with connected pins
                [{"net_name": "VCC", "pins": [{"component": "U1", "pin": "1"}, ...]}, ...]
            bom: Bill of materials with component types
                [{"reference": "U1", "category": "MCU", "value": "STM32", ...}, ...]
            ideation_context: Optional PlacementContext from ideation

        Returns:
            SignalFlowAnalysis with placement guidance
        """
        logger.info(f"Analyzing signal flow for {len(bom)} components, {len(netlist)} nets")

        # Build component lookup
        component_info = self._build_component_lookup(bom)

        # Step 1: Build connectivity graph
        if self.use_networkx:
            graph = self._build_connectivity_graph_nx(netlist, component_info)
        else:
            graph = self._build_connectivity_graph_dict(netlist, component_info)

        # Step 2: Identify signal paths
        signal_paths = self._identify_signal_paths(graph, component_info)

        # Step 3: Determine component layers (topological sort)
        component_layers = self._determine_layers(graph, component_info, signal_paths)

        # Step 4: Identify functional groups
        functional_groups = self._identify_functional_groups(
            graph,
            component_info,
            ideation_context
        )

        # Step 5: Find critical proximity pairs
        proximity_pairs = self._find_proximity_pairs(graph, component_info, netlist)

        # Step 6: Define separation zones
        separation_zones = self._define_separation_zones(component_info)

        logger.info(
            f"Signal flow analysis complete: {len(signal_paths)} paths, "
            f"{len(component_layers)} layers, {len(functional_groups)} groups, "
            f"{len(proximity_pairs)} proximity pairs"
        )

        return SignalFlowAnalysis(
            connectivity_graph=graph,
            signal_paths=signal_paths,
            component_layers=component_layers,
            functional_groups=functional_groups,
            critical_proximity_pairs=proximity_pairs,
            separation_zones=separation_zones
        )

    # -------------------------------------------------------------------------
    # Component lookup
    # -------------------------------------------------------------------------

    def _build_component_lookup(self, bom: List[Dict]) -> Dict[str, Dict]:
        """Build fast lookup from reference to component info."""
        lookup = {}
        for item in bom:
            ref = item.get('reference') or item.get('ref_des', '')
            if ref:
                lookup[ref] = {
                    'reference': ref,
                    'category': item.get('category', 'Other'),
                    'value': item.get('value', ''),
                    'part_number': item.get('part_number', ''),
                    'package': item.get('package', ''),
                }
        return lookup

    # -------------------------------------------------------------------------
    # Graph construction (networkx version)
    # -------------------------------------------------------------------------

    def _build_connectivity_graph_nx(
        self,
        netlist: List[Dict],
        component_info: Dict[str, Dict]
    ) -> nx.DiGraph:
        """
        Build directed graph from netlist using networkx.

        Nodes = components (with attributes: category, value, etc.)
        Edges = nets (with attributes: net_name, signal_type, criticality)
        """
        graph = nx.DiGraph()

        # Add component nodes
        for ref, info in component_info.items():
            graph.add_node(
                ref,
                category=info['category'],
                value=info['value'],
                part_number=info['part_number'],
                package=info['package']
            )

        # Add net edges
        for net in netlist:
            net_name = net.get('net_name', '')
            connected_pins = net.get('pins', [])

            if len(connected_pins) < 2:
                continue

            # Determine signal type and criticality
            signal_type = self._classify_signal(net_name)
            criticality = self._calculate_criticality(signal_type)

            # Add edges for each pair of connected components
            components_in_net = set()
            for pin in connected_pins:
                comp_ref = pin.get('component', '')
                if comp_ref in component_info:
                    components_in_net.add(comp_ref)

            # Convert to list for iteration
            comp_list = list(components_in_net)

            # Add directed edges based on source/sink heuristics
            for i, comp1 in enumerate(comp_list):
                for comp2 in comp_list[i+1:]:
                    # Determine direction
                    cat1 = component_info[comp1]['category']
                    cat2 = component_info[comp2]['category']

                    if self._is_source_category(cat1) and not self._is_source_category(cat2):
                        # comp1 -> comp2
                        graph.add_edge(
                            comp1,
                            comp2,
                            net_name=net_name,
                            signal_type=signal_type,
                            criticality=criticality
                        )
                    elif self._is_source_category(cat2) and not self._is_source_category(cat1):
                        # comp2 -> comp1
                        graph.add_edge(
                            comp2,
                            comp1,
                            net_name=net_name,
                            signal_type=signal_type,
                            criticality=criticality
                        )
                    else:
                        # Bidirectional or unknown - add both edges
                        graph.add_edge(comp1, comp2, net_name=net_name, signal_type=signal_type, criticality=criticality)
                        graph.add_edge(comp2, comp1, net_name=net_name, signal_type=signal_type, criticality=criticality)

        return graph

    # -------------------------------------------------------------------------
    # Graph construction (fallback dict version)
    # -------------------------------------------------------------------------

    def _build_connectivity_graph_dict(
        self,
        netlist: List[Dict],
        component_info: Dict[str, Dict]
    ) -> Dict[str, Dict]:
        """
        Build connectivity graph using dict (fallback when networkx unavailable).

        Structure:
        {
            "nodes": {"U1": {...node_attrs...}, ...},
            "edges": [{"from": "U1", "to": "U2", ...edge_attrs...}, ...]
        }
        """
        graph = {
            "nodes": {},
            "edges": [],
            "adjacency": defaultdict(set)  # ref -> set of connected refs
        }

        # Add nodes
        for ref, info in component_info.items():
            graph["nodes"][ref] = info.copy()

        # Add edges
        for net in netlist:
            net_name = net.get('net_name', '')
            connected_pins = net.get('pins', [])

            if len(connected_pins) < 2:
                continue

            signal_type = self._classify_signal(net_name)
            criticality = self._calculate_criticality(signal_type)

            components_in_net = set()
            for pin in connected_pins:
                comp_ref = pin.get('component', '')
                if comp_ref in component_info:
                    components_in_net.add(comp_ref)

            comp_list = list(components_in_net)

            for i, comp1 in enumerate(comp_list):
                for comp2 in comp_list[i+1:]:
                    graph["edges"].append({
                        "from": comp1,
                        "to": comp2,
                        "net_name": net_name,
                        "signal_type": signal_type,
                        "criticality": criticality
                    })
                    graph["adjacency"][comp1].add(comp2)
                    graph["adjacency"][comp2].add(comp1)

        return graph

    # -------------------------------------------------------------------------
    # Signal path identification
    # -------------------------------------------------------------------------

    def _identify_signal_paths(
        self,
        graph: Any,
        component_info: Dict[str, Dict]
    ) -> List[SignalPath]:
        """Identify critical signal paths through the circuit."""
        paths = []

        if self.use_networkx and isinstance(graph, nx.DiGraph):
            paths = self._identify_paths_nx(graph, component_info)
        else:
            paths = self._identify_paths_dict(graph, component_info)

        # Sort by criticality (most critical first)
        paths.sort(key=lambda p: p.criticality, reverse=True)

        return paths

    def _identify_paths_nx(
        self,
        graph: nx.DiGraph,
        component_info: Dict[str, Dict]
    ) -> List[SignalPath]:
        """Identify paths using networkx."""
        paths = []

        # Find source components
        sources = [
            node for node, data in graph.nodes(data=True)
            if self._is_source_category(data.get('category', ''))
        ]

        # For each source, find paths to sinks
        for source in sources:
            reachable = set()
            try:
                # BFS to find all reachable nodes
                for node in nx.descendants(graph, source):
                    reachable.add(node)
            except:
                # If descendants fails, skip
                continue

            # Find sinks
            sinks = [
                node for node in reachable
                if graph.out_degree(node) == 0 or
                self._is_sink_category(graph.nodes[node].get('category', ''))
            ]

            # Create paths
            for sink in sinks:
                try:
                    path_nodes = nx.shortest_path(graph, source, sink)
                    if len(path_nodes) < 2:
                        continue

                    # Extract net names
                    net_names = []
                    for i in range(len(path_nodes) - 1):
                        if graph.has_edge(path_nodes[i], path_nodes[i+1]):
                            edge_data = graph[path_nodes[i]][path_nodes[i+1]]
                            net_names.append(edge_data.get('net_name', ''))

                    # Classify path
                    path_type = self._classify_path_type(path_nodes, net_names, component_info)
                    criticality = self._calculate_path_criticality(path_type, net_names)

                    paths.append(SignalPath(
                        path_id=f"{source}_to_{sink}",
                        source_component=source,
                        sink_components=[sink],
                        intermediate_components=path_nodes[1:-1],
                        net_names=net_names,
                        path_type=path_type,
                        criticality=criticality
                    ))
                except:
                    pass

        return paths

    def _identify_paths_dict(
        self,
        graph: Dict,
        component_info: Dict[str, Dict]
    ) -> List[SignalPath]:
        """Identify paths using dict-based graph (fallback)."""
        paths = []

        # Find sources
        sources = [
            ref for ref, info in graph["nodes"].items()
            if self._is_source_category(info.get('category', ''))
        ]

        # Simple heuristic: one path per source
        for source in sources:
            # BFS to find connected components
            visited = {source}
            queue = deque([source])
            connected = []

            while queue:
                current = queue.popleft()
                for neighbor in graph["adjacency"].get(current, []):
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append(neighbor)
                        connected.append(neighbor)

            if connected:
                # Find a sink
                sink = None
                for comp in connected:
                    if self._is_sink_category(graph["nodes"][comp].get('category', '')):
                        sink = comp
                        break

                if not sink and connected:
                    sink = connected[-1]

                if sink:
                    paths.append(SignalPath(
                        path_id=f"{source}_to_{sink}",
                        source_component=source,
                        sink_components=[sink],
                        intermediate_components=[c for c in connected if c != sink],
                        net_names=[],
                        path_type='signal',
                        criticality=0.5
                    ))

        return paths

    # -------------------------------------------------------------------------
    # Layer determination
    # -------------------------------------------------------------------------

    def _determine_layers(
        self,
        graph: Any,
        component_info: Dict[str, Dict],
        signal_paths: List[SignalPath]
    ) -> List[ComponentLayer]:
        """Assign components to layers using topological sort."""
        if self.use_networkx and isinstance(graph, nx.DiGraph):
            return self._determine_layers_nx(graph, component_info)
        else:
            return self._determine_layers_dict(graph, component_info, signal_paths)

    def _determine_layers_nx(
        self,
        graph: nx.DiGraph,
        component_info: Dict[str, Dict]
    ) -> List[ComponentLayer]:
        """Determine layers using networkx topological sort."""
        # Use longest path layering
        layers_dict = {}

        try:
            # Try topological sort
            topo_order = list(nx.topological_sort(graph))
            # Assign layers based on longest path from sources
            for node in graph.nodes():
                try:
                    # Find longest path to this node
                    max_depth = 0
                    for predecessor in graph.predecessors(node):
                        if predecessor in layers_dict:
                            max_depth = max(max_depth, layers_dict[predecessor] + 1)
                    layers_dict[node] = max_depth
                except:
                    layers_dict[node] = 0
        except:
            # Graph has cycles - use heuristic
            for node in graph.nodes():
                category = graph.nodes[node].get('category', '')
                if self._is_source_category(category):
                    layers_dict[node] = 0
                elif self._is_sink_category(category):
                    layers_dict[node] = 99
                else:
                    layers_dict[node] = 50

        return self._build_layer_objects(layers_dict)

    def _determine_layers_dict(
        self,
        graph: Dict,
        component_info: Dict[str, Dict],
        signal_paths: List[SignalPath]
    ) -> List[ComponentLayer]:
        """Determine layers using heuristic (fallback)."""
        layers_dict = {}

        # Assign layers based on category
        for ref, info in component_info.items():
            category = info.get('category', '')
            if self._is_source_category(category):
                layers_dict[ref] = 0
            elif self._is_sink_category(category):
                layers_dict[ref] = 3
            elif category in self.IC_CATEGORIES:
                layers_dict[ref] = 1
            elif category in self.PASSIVE_CATEGORIES:
                layers_dict[ref] = 2
            else:
                layers_dict[ref] = 2

        return self._build_layer_objects(layers_dict)

    def _build_layer_objects(self, layers_dict: Dict[str, int]) -> List[ComponentLayer]:
        """Convert layer assignments to ComponentLayer objects."""
        layer_groups: Dict[int, List[str]] = defaultdict(list)
        for node, layer in layers_dict.items():
            layer_groups[layer].append(node)

        component_layers = []
        if not layer_groups:
            return component_layers

        max_layer = max(layer_groups.keys())

        for layer_num in sorted(layer_groups.keys()):
            layer_name = self._get_layer_name(layer_num, max_layer)
            x_hint = layer_num / max(max_layer, 1) if max_layer > 0 else 0.5

            component_layers.append(ComponentLayer(
                layer_id=layer_num,
                layer_name=layer_name,
                components=layer_groups[layer_num],
                x_position_hint=x_hint
            ))

        return component_layers

    # -------------------------------------------------------------------------
    # Functional grouping
    # -------------------------------------------------------------------------

    def _identify_functional_groups(
        self,
        graph: Any,
        component_info: Dict[str, Dict],
        ideation_context: Optional[Any] = None
    ) -> List[FunctionalGroup]:
        """Group components by function."""
        groups = []

        # If ideation context has subsystem blocks, use those
        if ideation_context and hasattr(ideation_context, 'subsystem_blocks'):
            for block in ideation_context.subsystem_blocks:
                groups.append(FunctionalGroup(
                    group_id=block.name.lower().replace(' ', '_'),
                    group_name=block.name,
                    components=block.components,
                    position_hint=None  # Parse position_hint if needed
                ))
            if groups:
                return groups

        # Otherwise, use heuristic grouping
        power_comps = []
        mcu_comps = []
        driver_comps = []
        passive_comps = []

        for ref, info in component_info.items():
            category = info.get('category', '')
            if category in ['Power', 'Regulator', 'LDO']:
                power_comps.append(ref)
            elif category in ['MCU', 'IC']:
                mcu_comps.append(ref)
            elif 'Driver' in category or 'MOSFET' in category:
                driver_comps.append(ref)
            elif category in self.PASSIVE_CATEGORIES:
                passive_comps.append(ref)

        if power_comps:
            groups.append(FunctionalGroup(
                group_id="power_supply",
                group_name="Power Supply",
                components=power_comps
            ))

        if mcu_comps:
            groups.append(FunctionalGroup(
                group_id="mcu_core",
                group_name="MCU Core",
                components=mcu_comps
            ))

        if driver_comps:
            groups.append(FunctionalGroup(
                group_id="motor_driver",
                group_name="Motor Driver",
                components=driver_comps
            ))

        if passive_comps:
            groups.append(FunctionalGroup(
                group_id="passive_components",
                group_name="Passive Components",
                components=passive_comps
            ))

        return groups

    # -------------------------------------------------------------------------
    # Proximity pairs
    # -------------------------------------------------------------------------

    def _find_proximity_pairs(
        self,
        graph: Any,
        component_info: Dict[str, Dict],
        netlist: List[Dict]
    ) -> List[Tuple[str, str]]:
        """Identify component pairs that must be placed close together."""
        pairs = []

        # Get adjacency info
        if self.use_networkx and isinstance(graph, nx.DiGraph):
            adjacency = {node: set(graph.neighbors(node)) for node in graph.nodes()}
        else:
            adjacency = graph.get("adjacency", {})

        # Rule 1: Bypass caps near ICs
        for ref, info in component_info.items():
            if info.get('category') in self.IC_CATEGORIES:
                # Find bypass caps connected to this IC
                neighbors = adjacency.get(ref, set())
                for neighbor in neighbors:
                    neighbor_info = component_info.get(neighbor, {})
                    if neighbor_info.get('category') == 'Capacitor':
                        # Check if it's a bypass cap (look for small value)
                        value = neighbor_info.get('value', '')
                        if self._is_bypass_cap(value):
                            pairs.append((ref, neighbor))

        # Rule 2: Crystal near MCU
        for ref, info in component_info.items():
            if info.get('category') == 'MCU':
                neighbors = adjacency.get(ref, set())
                for neighbor in neighbors:
                    neighbor_info = component_info.get(neighbor, {})
                    if neighbor_info.get('category') in ['Crystal', 'Oscillator']:
                        pairs.append((ref, neighbor))

        return pairs

    # -------------------------------------------------------------------------
    # Separation zones
    # -------------------------------------------------------------------------

    def _define_separation_zones(
        self,
        component_info: Dict[str, Dict]
    ) -> Dict[str, List[str]]:
        """Define zones that should be spatially separated."""
        zones = defaultdict(list)

        for ref, info in component_info.items():
            category = info.get('category', '')

            # Analog zone
            if 'ADC' in category or 'DAC' in category or category == 'Analog':
                zones['analog'].append(ref)

            # Digital zone
            elif category in ['MCU', 'IC', 'Gate_Driver']:
                zones['digital'].append(ref)

            # Power zone
            elif category in ['Power', 'Regulator', 'LDO']:
                zones['power'].append(ref)

        return dict(zones)

    # -------------------------------------------------------------------------
    # Helper methods
    # -------------------------------------------------------------------------

    def _classify_signal(self, net_name: str) -> str:
        """Classify signal type from net name."""
        net_upper = net_name.upper()

        for pattern in self.POWER_NET_PATTERNS:
            if re.match(pattern, net_upper):
                return 'power'

        for pattern in self.GROUND_NET_PATTERNS:
            if re.match(pattern, net_upper):
                return 'ground'

        for pattern in self.CLOCK_NET_PATTERNS:
            if re.match(pattern, net_upper, re.IGNORECASE):
                return 'clock'

        for pattern in self.HIGH_SPEED_NET_PATTERNS:
            if re.match(pattern, net_upper, re.IGNORECASE):
                return 'high_speed'

        return 'signal'

    def _calculate_criticality(self, signal_type: str) -> float:
        """Assign criticality score 0.0-1.0."""
        if signal_type == 'clock':
            return 1.0
        elif signal_type == 'power':
            return 0.9
        elif signal_type == 'high_speed':
            return 0.8
        elif signal_type == 'ground':
            return 0.9
        else:
            return 0.5

    def _is_source_category(self, category: str) -> bool:
        """Check if category is a signal source."""
        return category in self.SOURCE_CATEGORIES

    def _is_sink_category(self, category: str) -> bool:
        """Check if category is a signal sink."""
        return category in self.SINK_CATEGORIES

    def _classify_path_type(
        self,
        path_nodes: List[str],
        net_names: List[str],
        component_info: Dict[str, Dict]
    ) -> str:
        """Classify signal path type."""
        for net in net_names:
            sig_type = self._classify_signal(net)
            if sig_type in ['clock', 'power']:
                return sig_type
            elif sig_type == 'high_speed':
                return sig_type
        return 'signal'

    def _calculate_path_criticality(
        self,
        path_type: str,
        net_names: List[str]
    ) -> float:
        """Calculate path criticality."""
        return self._calculate_criticality(path_type)

    def _get_layer_name(self, layer_num: int, max_layer: int) -> str:
        """Generate human-readable layer name."""
        if layer_num == 0:
            return 'input'
        elif layer_num == max_layer:
            return 'output'
        else:
            return f'processing_{layer_num}'

    def _is_bypass_cap(self, value: str) -> bool:
        """Check if capacitor value indicates a bypass cap."""
        value_lower = value.lower()
        # Bypass caps are typically 0.1uF, 1uF, 10uF, or nF range
        if 'nf' in value_lower:
            return True
        if 'uf' in value_lower or 'µf' in value_lower:
            try:
                # Extract numeric value
                num_str = re.findall(r'[\d.]+', value)[0]
                num = float(num_str)
                return num <= 10.0  # <= 10uF
            except:
                return False
        return False


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = [
    'SignalPath',
    'ComponentLayer',
    'FunctionalGroup',
    'SignalFlowAnalysis',
    'SignalFlowAnalyzer',
]
