#!/usr/bin/env python3
"""
PathFinder Negotiation-Based Router - CPU-based iterative routing.

Adapted from OrthoRoute's PathFinder algorithm for MAPO v2.0.
This is a negotiation-based router that iteratively resolves conflicts
through cost updates, integrated with LLM agents for strategic guidance.

Algorithm:
1. Initial greedy routing (ignore congestion)
2. Identify overused resources
3. Increase costs iteratively
4. Rip-up worst offenders
5. Re-route with updated costs
6. Converge when no over-subscription

Part of the MAPO v2.0 Enhancement: "Opus 4.5 Thinks, Algorithms Execute"
"""

import json
import sys
import asyncio
import heapq
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any, Tuple, Set
from enum import Enum, auto
from pathlib import Path
from collections import defaultdict
import math

# Add parent directory to path for local imports
SCRIPT_DIR = Path(__file__).parent.resolve()
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

try:
    from agents.routing_strategist import RoutingStrategistAgent, RoutingStrategy
    from agents.congestion_predictor import CongestionPredictorAgent, CongestionPrediction
    from agents.conflict_resolver import ConflictResolverAgent, RoutingConflict, ConflictType
except ImportError:
    pass


class RouteStatus(Enum):
    """Status of a route."""
    UNROUTED = auto()
    ROUTED = auto()
    CONFLICT = auto()
    FAILED = auto()


@dataclass
class GridPoint:
    """A point on the routing grid."""
    x: float
    y: float
    layer: str = "F.Cu"

    def __hash__(self):
        return hash((round(self.x, 2), round(self.y, 2), self.layer))

    def __eq__(self, other):
        return (
            round(self.x, 2) == round(other.x, 2) and
            round(self.y, 2) == round(other.y, 2) and
            self.layer == other.layer
        )


@dataclass
class RouteSegment:
    """A segment of a route."""
    start: GridPoint
    end: GridPoint
    net_name: str
    layer: str
    width_mm: float = 0.25

    @property
    def length(self) -> float:
        """Calculate segment length."""
        dx = self.end.x - self.start.x
        dy = self.end.y - self.start.y
        return math.sqrt(dx*dx + dy*dy)


@dataclass
class Route:
    """Complete route for a net."""
    net_name: str
    segments: List[RouteSegment]
    vias: List[GridPoint]
    status: RouteStatus = RouteStatus.ROUTED
    total_length: float = 0.0
    via_count: int = 0

    def calculate_metrics(self):
        """Calculate route metrics."""
        self.total_length = sum(s.length for s in self.segments)
        self.via_count = len(self.vias)


@dataclass
class RoutingSolution:
    """Complete routing solution for all nets."""
    routes: Dict[str, Route]
    conflicts: List[Tuple[str, str, GridPoint]]  # (net_a, net_b, location)
    total_wire_length: float = 0.0
    total_via_count: int = 0
    unrouted_nets: List[str] = field(default_factory=list)

    def calculate_metrics(self):
        """Calculate solution metrics."""
        self.total_wire_length = sum(r.total_length for r in self.routes.values())
        self.total_via_count = sum(r.via_count for r in self.routes.values())


@dataclass
class CostMap:
    """Cost map for routing resources."""
    base_costs: Dict[GridPoint, float] = field(default_factory=dict)
    history_costs: Dict[GridPoint, float] = field(default_factory=dict)
    congestion_multiplier: float = 1.5

    def get_cost(self, point: GridPoint) -> float:
        """Get total cost for a point."""
        base = self.base_costs.get(point, 1.0)
        history = self.history_costs.get(point, 0.0)
        return base + history

    def add_congestion_penalty(self, point: GridPoint):
        """Add congestion penalty to a point."""
        current = self.history_costs.get(point, 0.0)
        self.history_costs[point] = current * self.congestion_multiplier + 0.5

    def set_base_cost(self, point: GridPoint, cost: float):
        """Set base cost for a point."""
        self.base_costs[point] = cost


class PathFinderRouter:
    """
    CPU-based negotiation routing (no GPU required).

    LLM Integration:
    - Opus 4.5 Strategist orders nets
    - Opus 4.5 Predictor provides congestion hints
    - Opus 4.5 Resolver handles persistent conflicts
    """

    def __init__(
        self,
        grid_size_mm: float = 2.54,
        max_iterations: int = 50,
        convergence_threshold: int = 3,
        strategist: Optional[RoutingStrategistAgent] = None,
        predictor: Optional[CongestionPredictorAgent] = None,
        resolver: Optional[ConflictResolverAgent] = None
    ):
        """
        Initialize the PathFinder router.

        Args:
            grid_size_mm: Routing grid size in mm (default 2.54 = 100 mil)
            max_iterations: Maximum negotiation iterations
            convergence_threshold: Iterations without improvement before stopping
            strategist: Optional LLM strategist for net ordering
            predictor: Optional LLM predictor for congestion
            resolver: Optional LLM resolver for conflicts
        """
        self.grid_size = grid_size_mm
        self.max_iterations = max_iterations
        self.convergence_threshold = convergence_threshold
        self.strategist = strategist
        self.predictor = predictor
        self.resolver = resolver

        self.cost_map = CostMap()
        self.occupancy: Dict[GridPoint, str] = {}  # point -> net_name
        self.congestion_history: List[int] = []

    async def route_with_negotiation(
        self,
        nets: List[Dict[str, Any]],
        placement: Dict[str, Tuple[float, float]],
        constraints: Optional[Dict[str, Any]] = None
    ) -> RoutingSolution:
        """
        Route all nets using negotiation-based algorithm.

        Args:
            nets: List of nets to route [{name, pins: [{x, y}]}]
            placement: Component positions (for reference)
            constraints: Optional routing constraints

        Returns:
            RoutingSolution with all routes
        """
        # Phase 1: Get strategic net ordering from LLM
        if self.strategist:
            ordered_nets = await self.strategist.order_nets(nets)
        else:
            ordered_nets = self._heuristic_order(nets)

        # Phase 2: Get congestion prediction for cost initialization
        if self.predictor:
            prediction = await self.predictor.predict_congestion(placement, nets)
            self._initialize_costs_from_prediction(prediction)
        else:
            self._initialize_default_costs()

        # Phase 3: Initial greedy routing
        print(f"  Initial routing of {len(ordered_nets)} nets...")
        solution = self._route_greedy(ordered_nets)
        initial_conflicts = len(solution.conflicts)
        print(f"  Initial conflicts: {initial_conflicts}")

        if initial_conflicts == 0:
            return solution

        # Phase 4: Negotiation loop
        best_solution = solution
        best_conflicts = initial_conflicts
        no_improvement_count = 0

        for iteration in range(self.max_iterations):
            # Identify conflicts
            conflicts = self._detect_conflicts(solution)
            self.congestion_history.append(len(conflicts))

            if not conflicts:
                print(f"  Converged at iteration {iteration}")
                break

            # Update costs based on congestion
            self._update_congestion_costs(conflicts)

            # Identify ripup candidates
            ripup_nets = self._identify_ripup_candidates(conflicts, solution)

            # Re-route with updated costs
            for net_name in ripup_nets:
                if net_name in solution.routes:
                    self._remove_route(solution, net_name)
                    new_route = self._route_single_net(
                        self._get_net_by_name(net_name, ordered_nets)
                    )
                    if new_route:
                        solution.routes[net_name] = new_route

            # Check improvement
            current_conflicts = len(self._detect_conflicts(solution))

            if current_conflicts < best_conflicts:
                best_solution = solution
                best_conflicts = current_conflicts
                no_improvement_count = 0
            else:
                no_improvement_count += 1

            # Check convergence
            if no_improvement_count >= self.convergence_threshold:
                print(f"  No improvement for {self.convergence_threshold} iterations, stopping")
                break

            if iteration % 10 == 0:
                print(f"  Iteration {iteration}: {current_conflicts} conflicts")

        # Phase 5: Handle persistent conflicts with LLM resolver
        final_conflicts = self._detect_conflicts(best_solution)
        if final_conflicts and self.resolver:
            print(f"  {len(final_conflicts)} persistent conflicts, consulting resolver...")
            # Convert to RoutingConflict format
            routing_conflicts = [
                RoutingConflict(
                    conflict_id=f"conflict_{i}",
                    conflict_type=ConflictType.SPATIAL,
                    net_a=c[0],
                    net_b=c[1],
                    location=(c[2].x, c[2].y),
                    severity="HIGH",
                    resources=[(c[2].x, c[2].y)],
                    description=f"Nets {c[0]} and {c[1]} conflict at ({c[2].x}, {c[2].y})"
                )
                for i, c in enumerate(final_conflicts[:10])
            ]

            plan = await self.resolver.resolve_conflicts(
                routing_conflicts,
                {'routes': {k: v.segments for k, v in best_solution.routes.items()}}
            )

            # Apply resolutions (simplified - would need more logic)
            print(f"  Resolver provided {len(plan.resolutions)} resolutions")

        # Calculate final metrics
        best_solution.calculate_metrics()
        best_solution.conflicts = self._detect_conflicts(best_solution)

        return best_solution

    def _heuristic_order(self, nets: List[Dict]) -> List[Dict]:
        """Order nets using heuristics when no LLM available."""
        def priority_key(net: Dict) -> Tuple[int, int]:
            name = net.get('name', '').upper()
            pins = net.get('pins', [])

            # Power nets first
            if any(p in name for p in ['VCC', 'VDD', 'PWR', '5V', '3V3', '12V']):
                return (0, -len(pins))
            # Ground next
            if any(g in name for g in ['GND', 'VSS', 'GROUND']):
                return (1, -len(pins))
            # Clock signals
            if 'CLK' in name or 'CLOCK' in name:
                return (2, -len(pins))
            # Everything else by pin count
            return (3, -len(pins))

        return sorted(nets, key=priority_key)

    def _initialize_costs_from_prediction(self, prediction: CongestionPrediction):
        """Initialize cost map from congestion prediction."""
        for region in prediction.regions:
            # Add cost penalty for congested regions
            for x in self._grid_range(region.x_min, region.x_max):
                for y in self._grid_range(region.y_min, region.y_max):
                    point = GridPoint(x, y)
                    if region.severity.name == 'CRITICAL':
                        self.cost_map.set_base_cost(point, 5.0)
                    elif region.severity.name == 'HIGH':
                        self.cost_map.set_base_cost(point, 3.0)
                    elif region.severity.name == 'MEDIUM':
                        self.cost_map.set_base_cost(point, 2.0)

    def _initialize_default_costs(self):
        """Initialize default cost map."""
        # All costs default to 1.0 in CostMap
        pass

    def _grid_range(self, start: float, end: float) -> List[float]:
        """Generate grid points in range."""
        points = []
        current = math.floor(start / self.grid_size) * self.grid_size
        while current <= end:
            points.append(current)
            current += self.grid_size
        return points

    def _route_greedy(self, nets: List[Dict]) -> RoutingSolution:
        """Initial greedy routing of all nets."""
        solution = RoutingSolution(routes={}, conflicts=[])

        for net in nets:
            route = self._route_single_net(net)
            if route:
                solution.routes[net['name']] = route
            else:
                solution.unrouted_nets.append(net['name'])

        solution.conflicts = self._detect_conflicts(solution)
        return solution

    def _route_single_net(self, net: Dict) -> Optional[Route]:
        """Route a single net using A* with cost map."""
        pins = net.get('pins', [])
        if len(pins) < 2:
            return None

        net_name = net.get('name', 'unnamed')
        segments = []
        vias = []

        # Route between consecutive pins
        for i in range(len(pins) - 1):
            start = GridPoint(
                self._snap_to_grid(pins[i].get('x', 0)),
                self._snap_to_grid(pins[i].get('y', 0))
            )
            end = GridPoint(
                self._snap_to_grid(pins[i+1].get('x', 0)),
                self._snap_to_grid(pins[i+1].get('y', 0))
            )

            path = self._astar_route(start, end, net_name)

            if path:
                # Convert path to segments
                for j in range(len(path) - 1):
                    segments.append(RouteSegment(
                        start=path[j],
                        end=path[j+1],
                        net_name=net_name,
                        layer=path[j].layer
                    ))

                # Register occupancy
                for point in path:
                    self.occupancy[point] = net_name

        if not segments:
            return None

        route = Route(
            net_name=net_name,
            segments=segments,
            vias=vias
        )
        route.calculate_metrics()
        return route

    def _astar_route(
        self,
        start: GridPoint,
        end: GridPoint,
        net_name: str
    ) -> Optional[List[GridPoint]]:
        """A* routing between two points."""
        # Priority queue: (f_score, g_score, point, path)
        open_set = [(0, 0, start, [start])]
        closed_set: Set[GridPoint] = set()

        while open_set:
            _, g_score, current, path = heapq.heappop(open_set)

            if current == end:
                return path

            if current in closed_set:
                continue

            closed_set.add(current)

            # Explore neighbors (Manhattan routing)
            for neighbor in self._get_neighbors(current):
                if neighbor in closed_set:
                    continue

                # Check if occupied by another net
                occupant = self.occupancy.get(neighbor)
                if occupant and occupant != net_name:
                    continue  # Skip occupied points

                # Calculate costs
                move_cost = self._get_move_cost(current, neighbor)
                new_g = g_score + move_cost
                h = self._heuristic(neighbor, end)
                f = new_g + h

                heapq.heappush(open_set, (f, new_g, neighbor, path + [neighbor]))

        return None  # No path found

    def _get_neighbors(self, point: GridPoint) -> List[GridPoint]:
        """Get valid neighbors of a point (Manhattan directions)."""
        neighbors = []
        for dx, dy in [(self.grid_size, 0), (-self.grid_size, 0),
                       (0, self.grid_size), (0, -self.grid_size)]:
            neighbors.append(GridPoint(
                round(point.x + dx, 2),
                round(point.y + dy, 2),
                point.layer
            ))
        return neighbors

    def _get_move_cost(self, from_point: GridPoint, to_point: GridPoint) -> float:
        """Get cost to move between points."""
        base_cost = self.cost_map.get_cost(to_point)
        distance = math.sqrt(
            (to_point.x - from_point.x)**2 +
            (to_point.y - from_point.y)**2
        )
        return base_cost * distance

    def _heuristic(self, point: GridPoint, goal: GridPoint) -> float:
        """Manhattan distance heuristic."""
        return abs(point.x - goal.x) + abs(point.y - goal.y)

    def _snap_to_grid(self, value: float) -> float:
        """Snap value to routing grid."""
        return round(value / self.grid_size) * self.grid_size

    def _detect_conflicts(
        self,
        solution: RoutingSolution
    ) -> List[Tuple[str, str, GridPoint]]:
        """Detect conflicts between routes."""
        conflicts = []
        point_to_nets: Dict[GridPoint, List[str]] = defaultdict(list)

        # Build point occupancy map
        for net_name, route in solution.routes.items():
            for segment in route.segments:
                # Sample points along segment
                points = self._sample_segment(segment)
                for point in points:
                    point_to_nets[point].append(net_name)

        # Find conflicts (points with multiple nets)
        for point, nets in point_to_nets.items():
            if len(nets) > 1:
                for i in range(len(nets)):
                    for j in range(i + 1, len(nets)):
                        conflicts.append((nets[i], nets[j], point))

        return conflicts

    def _sample_segment(self, segment: RouteSegment) -> List[GridPoint]:
        """Sample points along a segment."""
        points = [GridPoint(segment.start.x, segment.start.y, segment.layer)]

        dx = segment.end.x - segment.start.x
        dy = segment.end.y - segment.start.y
        length = math.sqrt(dx*dx + dy*dy)

        if length > 0:
            steps = max(1, int(length / self.grid_size))
            for i in range(1, steps + 1):
                t = i / steps
                x = segment.start.x + dx * t
                y = segment.start.y + dy * t
                points.append(GridPoint(
                    self._snap_to_grid(x),
                    self._snap_to_grid(y),
                    segment.layer
                ))

        return points

    def _update_congestion_costs(
        self,
        conflicts: List[Tuple[str, str, GridPoint]]
    ):
        """Update costs for congested resources."""
        for _, _, point in conflicts:
            self.cost_map.add_congestion_penalty(point)

    def _identify_ripup_candidates(
        self,
        conflicts: List[Tuple[str, str, GridPoint]],
        solution: RoutingSolution
    ) -> List[str]:
        """Identify nets to rip-up and re-route."""
        # Count conflicts per net
        net_conflicts: Dict[str, int] = defaultdict(int)
        for net_a, net_b, _ in conflicts:
            net_conflicts[net_a] += 1
            net_conflicts[net_b] += 1

        # Sort by conflict count (most conflicts first)
        sorted_nets = sorted(net_conflicts.items(), key=lambda x: -x[1])

        # Return top offenders (up to 5)
        return [net for net, _ in sorted_nets[:5]]

    def _remove_route(self, solution: RoutingSolution, net_name: str):
        """Remove a route from the solution."""
        if net_name in solution.routes:
            route = solution.routes[net_name]
            # Clear occupancy
            for segment in route.segments:
                for point in self._sample_segment(segment):
                    if self.occupancy.get(point) == net_name:
                        del self.occupancy[point]
            # Remove route
            del solution.routes[net_name]

    def _get_net_by_name(self, name: str, nets: List[Dict]) -> Optional[Dict]:
        """Get net by name from list."""
        for net in nets:
            if net.get('name') == name:
                return net
        return None

    def get_statistics(self) -> Dict[str, Any]:
        """Get routing statistics."""
        return {
            'iterations': len(self.congestion_history),
            'congestion_history': self.congestion_history,
            'final_congestion': self.congestion_history[-1] if self.congestion_history else 0,
            'cost_entries': len(self.cost_map.history_costs),
        }


# Convenience function
def create_pathfinder_router(
    grid_size_mm: float = 2.54,
    max_iterations: int = 50
) -> PathFinderRouter:
    """Create a PathFinder router."""
    return PathFinderRouter(
        grid_size_mm=grid_size_mm,
        max_iterations=max_iterations
    )


# Main entry point for testing
if __name__ == '__main__':
    async def test_pathfinder():
        """Test the PathFinder router."""
        print("\n" + "="*60)
        print("PATHFINDER ROUTER - Test")
        print("="*60)

        router = create_pathfinder_router(grid_size_mm=2.54, max_iterations=20)

        # Sample nets
        nets = [
            {
                'name': 'VCC',
                'pins': [
                    {'x': 10, 'y': 20},
                    {'x': 50, 'y': 20},
                    {'x': 90, 'y': 20}
                ]
            },
            {
                'name': 'GND',
                'pins': [
                    {'x': 10, 'y': 80},
                    {'x': 50, 'y': 80},
                    {'x': 90, 'y': 80}
                ]
            },
            {
                'name': 'DATA0',
                'pins': [
                    {'x': 20, 'y': 30},
                    {'x': 80, 'y': 70}
                ]
            },
            {
                'name': 'DATA1',
                'pins': [
                    {'x': 20, 'y': 70},
                    {'x': 80, 'y': 30}
                ]
            },
        ]

        placement = {
            'U1': (50, 50),
            'J1': (10, 50),
            'J2': (90, 50),
        }

        print(f"\nRouting {len(nets)} nets...")
        solution = await router.route_with_negotiation(nets, placement)

        print(f"\n--- Routing Results ---")
        print(f"  Routes completed: {len(solution.routes)}")
        print(f"  Total wire length: {solution.total_wire_length:.1f} mm")
        print(f"  Total vias: {solution.total_via_count}")
        print(f"  Conflicts remaining: {len(solution.conflicts)}")
        print(f"  Unrouted nets: {solution.unrouted_nets}")

        # Show route details
        for net_name, route in solution.routes.items():
            print(f"\n  {net_name}:")
            print(f"    Segments: {len(route.segments)}")
            print(f"    Length: {route.total_length:.1f} mm")
            print(f"    Status: {route.status.name}")

        # Show statistics
        stats = router.get_statistics()
        print(f"\n--- Statistics ---")
        print(f"  Iterations: {stats['iterations']}")
        print(f"  Final congestion: {stats['final_congestion']}")

    asyncio.run(test_pathfinder())
