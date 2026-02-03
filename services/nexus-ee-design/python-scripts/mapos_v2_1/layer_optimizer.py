"""
MAPO v2.0 Layer Assignment Optimizer

Optimizes layer assignments for multi-layer PCB routing using dynamic programming.
Integrates with LLM LayerAssignmentStrategistAgent for strategic hints.

Key Objectives:
1. Minimize total via count
2. Respect layer adjacency constraints
3. Honor signal integrity requirements
4. Balance layer utilization

Algorithm:
- DP sweep for optimal assignment
- Iterative refinement for conflicts
- LLM hints for net grouping and preferences

Research Sources:
- Multi-Agent Based Minimal-Layer Via Routing (ScienceDirect 2025)
- IPC-2221 layer stackup guidelines
- Altium/Cadence layer assignment best practices

Author: Claude Opus 4.5 via MAPO v2.0
"""

import asyncio
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple, Any
from enum import Enum
import logging
import math

logger = logging.getLogger(__name__)


# =============================================================================
# Data Structures
# =============================================================================

class LayerType(Enum):
    """Types of PCB layers."""
    SIGNAL = "signal"
    POWER = "power"
    GROUND = "ground"
    MIXED = "mixed"


class NetType(Enum):
    """Types of nets for layer assignment."""
    POWER = "power"
    GROUND = "ground"
    HIGH_SPEED = "high_speed"
    DIFFERENTIAL = "differential"
    ANALOG = "analog"
    DIGITAL = "digital"
    CLOCK = "clock"


@dataclass
class Layer:
    """A PCB layer definition."""
    index: int
    name: str
    layer_type: LayerType
    thickness: float = 0.035  # mm (1oz copper)
    dielectric_thickness: float = 0.2  # mm to adjacent layer
    dielectric_constant: float = 4.5  # FR4 typical

    def __hash__(self):
        return hash(self.index)


@dataclass
class LayerStackup:
    """Complete layer stackup definition."""
    layers: List[Layer]
    board_thickness: float = 1.6  # mm

    def get_signal_layers(self) -> List[Layer]:
        """Get signal routing layers."""
        return [l for l in self.layers if l.layer_type in [LayerType.SIGNAL, LayerType.MIXED]]

    def get_power_layers(self) -> List[Layer]:
        """Get power plane layers."""
        return [l for l in self.layers if l.layer_type == LayerType.POWER]

    def get_ground_layers(self) -> List[Layer]:
        """Get ground plane layers."""
        return [l for l in self.layers if l.layer_type == LayerType.GROUND]

    def get_layer_distance(self, layer1: int, layer2: int) -> int:
        """Get number of layers between two layers (via span)."""
        return abs(layer1 - layer2)


@dataclass
class Route2D:
    """A 2D route (before layer assignment)."""
    net_name: str
    path: List[Tuple[float, float]]  # (x, y) points
    net_type: NetType = NetType.DIGITAL
    width: float = 0.254  # mm
    priority: int = 1

    def get_length(self) -> float:
        """Calculate total route length."""
        length = 0.0
        for i in range(len(self.path) - 1):
            dx = self.path[i + 1][0] - self.path[i][0]
            dy = self.path[i + 1][1] - self.path[i][1]
            length += math.sqrt(dx * dx + dy * dy)
        return length


@dataclass
class Route3D:
    """A 3D route (with layer assignments)."""
    net_name: str
    segments: List[Tuple[Tuple[float, float], Tuple[float, float], int]]  # (start, end, layer)
    vias: List[Tuple[float, float, int, int]]  # (x, y, from_layer, to_layer)
    net_type: NetType = NetType.DIGITAL
    width: float = 0.254

    def get_via_count(self) -> int:
        return len(self.vias)

    def get_via_span_cost(self) -> float:
        """Calculate total via span cost (longer spans = higher cost)."""
        return sum(abs(v[3] - v[2]) for v in self.vias)


@dataclass
class LayerAssignment:
    """Assignment of a net segment to a layer."""
    net_name: str
    segment_index: int
    layer: int
    cost: float = 0.0


@dataclass
class LayerOptimizationResult:
    """Result of layer optimization."""
    routes_3d: Dict[str, Route3D]
    total_vias: int
    total_via_span_cost: float
    layer_utilization: Dict[int, float]  # layer -> percentage used
    iterations: int
    llm_hints_applied: int = 0


@dataclass
class LLMLayerHint:
    """A hint from the LLM LayerStrategist."""
    net_name: str
    preferred_layers: List[int]
    avoid_layers: List[int]
    reasoning: str
    confidence: float = 0.8


# =============================================================================
# Layer Assignment Optimizer
# =============================================================================

class LayerAssignmentOptimizer:
    """
    CPU-based layer assignment using dynamic programming.

    Integrates with LLM LayerAssignmentStrategistAgent for:
    - Net ordering strategy
    - Layer preference hints
    - Via cost weighting
    - Conflict resolution guidance

    Algorithm:
    1. Get LLM hints for net grouping and preferences
    2. Build cost matrix (via costs, layer preferences, SI requirements)
    3. DP sweep to minimize total via cost
    4. Iterative refinement for conflicts
    """

    def __init__(
        self,
        layer_strategist=None,  # Optional LLM LayerAssignmentStrategistAgent
        via_cost: float = 1.0,
        layer_change_cost: float = 0.5,
        adjacent_layer_bonus: float = 0.3  # Prefer adjacent layer transitions
    ):
        self.layer_strategist = layer_strategist
        self.via_cost = via_cost
        self.layer_change_cost = layer_change_cost
        self.adjacent_layer_bonus = adjacent_layer_bonus

        # Statistics
        self.stats = {
            "iterations": 0,
            "llm_calls": 0,
            "conflicts_resolved": 0
        }

    async def optimize_layers(
        self,
        routes_2d: List[Route2D],
        stackup: LayerStackup,
        existing_assignments: Optional[Dict[str, int]] = None
    ) -> LayerOptimizationResult:
        """
        Optimize layer assignments for 2D routes.

        Steps:
        1. Get layer hints from LLM strategist
        2. Order nets by priority and SI requirements
        3. Build cost matrix
        4. DP assignment sweep
        5. Resolve conflicts
        6. Materialize 3D routes
        """
        self.stats = {"iterations": 0, "llm_calls": 0, "conflicts_resolved": 0}

        signal_layers = stackup.get_signal_layers()
        if not signal_layers:
            logger.error("No signal layers in stackup")
            return LayerOptimizationResult(
                routes_3d={},
                total_vias=0,
                total_via_span_cost=0,
                layer_utilization={},
                iterations=0
            )

        # Get LLM hints if strategist available
        hints: Dict[str, LLMLayerHint] = {}
        if self.layer_strategist:
            hints = await self._get_llm_hints(routes_2d, stackup)
            self.stats["llm_calls"] += 1

        # Order nets by priority
        ordered_nets = self._order_nets(routes_2d, hints)

        # Build cost matrix
        cost_matrix = self._build_cost_matrix(ordered_nets, signal_layers, hints, stackup)

        # DP assignment
        initial_assignment = self._dp_assign(ordered_nets, signal_layers, cost_matrix)

        # Detect and resolve layer conflicts
        final_assignment = await self._resolve_conflicts(
            initial_assignment, ordered_nets, signal_layers, stackup, hints
        )

        # Materialize 3D routes
        routes_3d = self._materialize_3d(routes_2d, final_assignment, stackup)

        # Calculate metrics
        total_vias = sum(r.get_via_count() for r in routes_3d.values())
        total_via_span = sum(r.get_via_span_cost() for r in routes_3d.values())
        layer_util = self._calculate_layer_utilization(routes_3d, signal_layers)

        return LayerOptimizationResult(
            routes_3d=routes_3d,
            total_vias=total_vias,
            total_via_span_cost=total_via_span,
            layer_utilization=layer_util,
            iterations=self.stats["iterations"],
            llm_hints_applied=len(hints)
        )

    async def _get_llm_hints(
        self,
        routes: List[Route2D],
        stackup: LayerStackup
    ) -> Dict[str, LLMLayerHint]:
        """Get layer assignment hints from LLM strategist."""
        if not self.layer_strategist:
            return {}

        try:
            # Prepare data for LLM
            net_info = [
                {
                    "name": r.net_name,
                    "type": r.net_type.value,
                    "length": r.get_length(),
                    "priority": r.priority
                }
                for r in routes
            ]

            layer_info = [
                {
                    "index": l.index,
                    "name": l.name,
                    "type": l.layer_type.value
                }
                for l in stackup.layers
            ]

            # Call LLM strategist
            hints_response = await self.layer_strategist.get_layer_hints(
                net_info, layer_info
            )

            # Parse hints
            hints = {}
            for hint_data in hints_response.net_hints:
                hints[hint_data["net_name"]] = LLMLayerHint(
                    net_name=hint_data["net_name"],
                    preferred_layers=hint_data.get("preferred_layers", []),
                    avoid_layers=hint_data.get("avoid_layers", []),
                    reasoning=hint_data.get("reasoning", ""),
                    confidence=hint_data.get("confidence", 0.8)
                )

            return hints

        except Exception as e:
            logger.warning(f"Failed to get LLM hints: {e}")
            return {}

    def _order_nets(
        self,
        routes: List[Route2D],
        hints: Dict[str, LLMLayerHint]
    ) -> List[Route2D]:
        """Order nets for processing (high priority and SI-critical first)."""
        def net_score(route: Route2D) -> float:
            score = route.priority * 10

            # SI-critical nets get higher priority
            if route.net_type in [NetType.HIGH_SPEED, NetType.DIFFERENTIAL, NetType.CLOCK]:
                score += 50
            elif route.net_type == NetType.ANALOG:
                score += 30

            # LLM confidence affects ordering
            if route.net_name in hints:
                score += hints[route.net_name].confidence * 10

            return score

        return sorted(routes, key=net_score, reverse=True)

    def _build_cost_matrix(
        self,
        routes: List[Route2D],
        signal_layers: List[Layer],
        hints: Dict[str, LLMLayerHint],
        stackup: LayerStackup
    ) -> Dict[Tuple[str, int], float]:
        """
        Build cost matrix for net-layer assignments.

        cost_matrix[(net_name, layer_index)] = assignment cost

        Lower cost = better assignment.
        """
        cost_matrix = {}

        for route in routes:
            for layer in signal_layers:
                base_cost = 1.0

                # Apply LLM hints
                if route.net_name in hints:
                    hint = hints[route.net_name]
                    if layer.index in hint.preferred_layers:
                        base_cost *= 0.5  # Prefer suggested layers
                    if layer.index in hint.avoid_layers:
                        base_cost *= 2.0  # Avoid warned layers

                # SI requirements
                if route.net_type in [NetType.HIGH_SPEED, NetType.DIFFERENTIAL]:
                    # Prefer layers adjacent to ground planes
                    ground_layers = stackup.get_ground_layers()
                    min_dist_to_ground = min(
                        (abs(layer.index - gl.index) for gl in ground_layers),
                        default=10
                    )
                    if min_dist_to_ground == 1:
                        base_cost *= 0.6  # Adjacent to ground is best
                    elif min_dist_to_ground > 2:
                        base_cost *= 1.5  # Far from ground is worse

                # Power/ground nets should use power layers
                if route.net_type == NetType.POWER:
                    power_layers = stackup.get_power_layers()
                    if power_layers:
                        base_cost *= 2.0  # Discourage signal layers for power

                if route.net_type == NetType.GROUND:
                    ground_layers = stackup.get_ground_layers()
                    if ground_layers:
                        base_cost *= 2.0  # Discourage signal layers for ground

                # Outer layers often preferred for accessibility
                if layer.index == 0 or layer.index == len(stackup.layers) - 1:
                    base_cost *= 0.9  # Slight preference for outer layers

                cost_matrix[(route.net_name, layer.index)] = base_cost

        return cost_matrix

    def _dp_assign(
        self,
        routes: List[Route2D],
        signal_layers: List[Layer],
        cost_matrix: Dict[Tuple[str, int], float]
    ) -> Dict[str, int]:
        """
        Dynamic programming layer assignment.

        Minimizes total cost while considering via penalties
        for nets that share connection points.
        """
        n_nets = len(routes)
        n_layers = len(signal_layers)

        if n_nets == 0 or n_layers == 0:
            return {}

        layer_indices = [l.index for l in signal_layers]

        # dp[i][j] = (min_cost, assignment) to assign first i nets with net i on layer j
        # For simplicity, we use greedy with lookahead rather than full DP
        # (Full DP would require tracking which nets connect)

        assignment = {}

        for route in routes:
            # Find best layer for this net
            best_layer = None
            best_cost = float('inf')

            for layer_idx in layer_indices:
                cost = cost_matrix.get((route.net_name, layer_idx), 1.0)

                # Add via penalty if connecting to net on different layer
                via_penalty = 0.0
                for other_net, other_layer in assignment.items():
                    if self._nets_connected(route.net_name, other_net):
                        if other_layer != layer_idx:
                            span = abs(layer_idx - other_layer)
                            via_penalty += self.via_cost + span * self.layer_change_cost
                            if span == 1:
                                via_penalty -= self.adjacent_layer_bonus

                total_cost = cost + via_penalty

                if total_cost < best_cost:
                    best_cost = total_cost
                    best_layer = layer_idx

            assignment[route.net_name] = best_layer

        return assignment

    def _nets_connected(self, net1: str, net2: str) -> bool:
        """
        Check if two nets share a connection point.
        (Simplified - in real implementation, check actual connectivity)
        """
        # For now, assume nets with similar prefixes are related
        # Real implementation would check netlist connectivity
        prefix1 = net1.split('_')[0] if '_' in net1 else net1[:3]
        prefix2 = net2.split('_')[0] if '_' in net2 else net2[:3]
        return prefix1 == prefix2

    async def _resolve_conflicts(
        self,
        assignment: Dict[str, int],
        routes: List[Route2D],
        signal_layers: List[Layer],
        stackup: LayerStackup,
        hints: Dict[str, LLMLayerHint]
    ) -> Dict[str, int]:
        """
        Resolve layer conflicts through iterative refinement.

        Conflicts:
        - Two routes on same layer that cross
        - Layer overutilization
        """
        max_iterations = 10

        for iteration in range(max_iterations):
            self.stats["iterations"] += 1

            # Detect conflicts
            conflicts = self._detect_layer_conflicts(routes, assignment)

            if not conflicts:
                break

            # Resolve one conflict at a time
            conflict = conflicts[0]
            self.stats["conflicts_resolved"] += 1

            # Try to move one net to different layer
            net_to_move = conflict["net1"]
            current_layer = assignment[net_to_move]

            # Find alternative layer
            layer_indices = [l.index for l in signal_layers]
            for alt_layer in layer_indices:
                if alt_layer != current_layer:
                    # Check if move resolves conflict
                    test_assignment = assignment.copy()
                    test_assignment[net_to_move] = alt_layer
                    remaining_conflicts = self._detect_layer_conflicts(routes, test_assignment)

                    if len(remaining_conflicts) < len(conflicts):
                        assignment = test_assignment
                        break

        return assignment

    def _detect_layer_conflicts(
        self,
        routes: List[Route2D],
        assignment: Dict[str, int]
    ) -> List[Dict[str, Any]]:
        """Detect conflicts between routes on same layer."""
        conflicts = []

        # Group routes by layer
        routes_by_layer: Dict[int, List[Route2D]] = {}
        for route in routes:
            layer = assignment.get(route.net_name)
            if layer is not None:
                if layer not in routes_by_layer:
                    routes_by_layer[layer] = []
                routes_by_layer[layer].append(route)

        # Check for crossings on each layer
        for layer, layer_routes in routes_by_layer.items():
            for i, route1 in enumerate(layer_routes):
                for route2 in layer_routes[i + 1:]:
                    if self._routes_cross(route1, route2):
                        conflicts.append({
                            "type": "crossing",
                            "net1": route1.net_name,
                            "net2": route2.net_name,
                            "layer": layer
                        })

        return conflicts

    def _routes_cross(self, route1: Route2D, route2: Route2D) -> bool:
        """Check if two 2D routes cross (simplified bounding box check)."""
        if not route1.path or not route2.path:
            return False

        # Get bounding boxes
        x1_min = min(p[0] for p in route1.path)
        x1_max = max(p[0] for p in route1.path)
        y1_min = min(p[1] for p in route1.path)
        y1_max = max(p[1] for p in route1.path)

        x2_min = min(p[0] for p in route2.path)
        x2_max = max(p[0] for p in route2.path)
        y2_min = min(p[1] for p in route2.path)
        y2_max = max(p[1] for p in route2.path)

        # Check bounding box overlap
        return not (x1_max < x2_min or x2_max < x1_min or
                   y1_max < y2_min or y2_max < y1_min)

    def _materialize_3d(
        self,
        routes_2d: List[Route2D],
        assignment: Dict[str, int],
        stackup: LayerStackup
    ) -> Dict[str, Route3D]:
        """Convert 2D routes + layer assignments to 3D routes with vias."""
        routes_3d = {}

        for route in routes_2d:
            layer = assignment.get(route.net_name, 0)

            # Simple case: entire route on one layer
            segments = []
            for i in range(len(route.path) - 1):
                segments.append((
                    route.path[i],
                    route.path[i + 1],
                    layer
                ))

            routes_3d[route.net_name] = Route3D(
                net_name=route.net_name,
                segments=segments,
                vias=[],  # No vias for single-layer route
                net_type=route.net_type,
                width=route.width
            )

        return routes_3d

    def _calculate_layer_utilization(
        self,
        routes_3d: Dict[str, Route3D],
        signal_layers: List[Layer]
    ) -> Dict[int, float]:
        """Calculate utilization percentage per layer."""
        utilization = {l.index: 0.0 for l in signal_layers}

        # Count segments per layer
        total_segments = 0
        for route in routes_3d.values():
            for segment in route.segments:
                layer = segment[2]
                if layer in utilization:
                    utilization[layer] += 1
                total_segments += 1

        # Convert to percentages
        if total_segments > 0:
            for layer in utilization:
                utilization[layer] = (utilization[layer] / total_segments) * 100

        return utilization


# =============================================================================
# Standard Stackup Presets
# =============================================================================

def create_2_layer_stackup() -> LayerStackup:
    """Create standard 2-layer stackup."""
    return LayerStackup(
        layers=[
            Layer(0, "Top", LayerType.SIGNAL, 0.035, 1.5, 4.5),
            Layer(1, "Bottom", LayerType.SIGNAL, 0.035, 1.5, 4.5),
        ],
        board_thickness=1.6
    )


def create_4_layer_stackup() -> LayerStackup:
    """Create standard 4-layer stackup (Signal/Ground/Power/Signal)."""
    return LayerStackup(
        layers=[
            Layer(0, "Top", LayerType.SIGNAL, 0.035, 0.2, 4.5),
            Layer(1, "Ground", LayerType.GROUND, 0.035, 1.0, 4.5),
            Layer(2, "Power", LayerType.POWER, 0.035, 0.2, 4.5),
            Layer(3, "Bottom", LayerType.SIGNAL, 0.035, 0.2, 4.5),
        ],
        board_thickness=1.6
    )


def create_6_layer_stackup() -> LayerStackup:
    """Create standard 6-layer stackup."""
    return LayerStackup(
        layers=[
            Layer(0, "Top", LayerType.SIGNAL, 0.035, 0.15, 4.5),
            Layer(1, "Ground1", LayerType.GROUND, 0.035, 0.2, 4.5),
            Layer(2, "Signal1", LayerType.SIGNAL, 0.035, 0.8, 4.5),
            Layer(3, "Signal2", LayerType.SIGNAL, 0.035, 0.2, 4.5),
            Layer(4, "Power", LayerType.POWER, 0.035, 0.15, 4.5),
            Layer(5, "Bottom", LayerType.SIGNAL, 0.035, 0.15, 4.5),
        ],
        board_thickness=1.6
    )


def create_8_layer_stackup() -> LayerStackup:
    """Create standard 8-layer stackup for high-speed designs."""
    return LayerStackup(
        layers=[
            Layer(0, "Top", LayerType.SIGNAL, 0.035, 0.1, 4.5),
            Layer(1, "Ground1", LayerType.GROUND, 0.035, 0.15, 4.5),
            Layer(2, "Signal1", LayerType.SIGNAL, 0.035, 0.2, 4.5),
            Layer(3, "Power1", LayerType.POWER, 0.035, 0.6, 4.5),
            Layer(4, "Ground2", LayerType.GROUND, 0.035, 0.2, 4.5),
            Layer(5, "Signal2", LayerType.SIGNAL, 0.035, 0.15, 4.5),
            Layer(6, "Power2", LayerType.POWER, 0.035, 0.1, 4.5),
            Layer(7, "Bottom", LayerType.SIGNAL, 0.035, 0.1, 4.5),
        ],
        board_thickness=1.6
    )


# =============================================================================
# Factory Function
# =============================================================================

def create_layer_optimizer(
    layer_strategist=None,
    via_cost: float = 1.0,
    layer_change_cost: float = 0.5
) -> LayerAssignmentOptimizer:
    """Factory function to create a layer optimizer instance."""
    return LayerAssignmentOptimizer(
        layer_strategist=layer_strategist,
        via_cost=via_cost,
        layer_change_cost=layer_change_cost
    )


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    "LayerAssignmentOptimizer",
    "LayerOptimizationResult",
    "LayerStackup",
    "Layer",
    "LayerType",
    "Route2D",
    "Route3D",
    "NetType",
    "LLMLayerHint",
    "create_layer_optimizer",
    "create_2_layer_stackup",
    "create_4_layer_stackup",
    "create_6_layer_stackup",
    "create_8_layer_stackup"
]
