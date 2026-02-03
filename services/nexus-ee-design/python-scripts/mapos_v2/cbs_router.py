"""
MAPO v2.0 CBS (Conflict-Based Search) Router

Adapts multi-agent pathfinding (MAPF) algorithms for PCB multi-net routing.
Integrates with LLM ConflictResolverAgent for complex conflict resolution.

Key Algorithm:
1. Low-level: Route each net independently (A* or Manhattan)
2. High-level: Build constraint tree from detected conflicts
3. Branch on conflicts: add constraints and re-route
4. LLM integration for complex/persistent conflicts

Research Sources:
- Sharon et al., "Conflict-Based Search for Optimal Multi-Agent Path Finding"
- Multi-Agent Based Minimal-Layer Via Routing (ScienceDirect 2025)
- Adapted from MAPF to PCB routing domain

Author: Claude Opus 4.5 via MAPO v2.0
"""

import asyncio
import heapq
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple, Any
from enum import Enum
import logging

logger = logging.getLogger(__name__)


# =============================================================================
# Data Structures
# =============================================================================

class ConflictType(Enum):
    """Types of routing conflicts."""
    VERTEX = "vertex"          # Two nets use same grid point
    EDGE = "edge"              # Two nets use same edge (crossing)
    SPACING = "spacing"        # Nets too close (clearance violation)
    LAYER = "layer"            # Layer resource contention


@dataclass
class GridPoint:
    """A point on the routing grid."""
    x: float
    y: float
    layer: int = 0

    def __hash__(self):
        return hash((self.x, self.y, self.layer))

    def __eq__(self, other):
        if not isinstance(other, GridPoint):
            return False
        return self.x == other.x and self.y == other.y and self.layer == other.layer


@dataclass
class Edge:
    """An edge between two grid points."""
    start: GridPoint
    end: GridPoint

    def __hash__(self):
        # Order-independent hash for undirected edges
        p1 = (self.start.x, self.start.y, self.start.layer)
        p2 = (self.end.x, self.end.y, self.end.layer)
        return hash(tuple(sorted([p1, p2])))

    def __eq__(self, other):
        if not isinstance(other, Edge):
            return False
        return {(self.start.x, self.start.y, self.start.layer),
                (self.end.x, self.end.y, self.end.layer)} == \
               {(other.start.x, other.start.y, other.start.layer),
                (other.end.x, other.end.y, other.end.layer)}


@dataclass
class Constraint:
    """A constraint preventing a net from using a resource."""
    net_name: str
    resource: GridPoint | Edge
    constraint_type: ConflictType

    def __hash__(self):
        return hash((self.net_name, hash(self.resource), self.constraint_type))


@dataclass
class Conflict:
    """A conflict between two nets."""
    net1: str
    net2: str
    conflict_type: ConflictType
    location: GridPoint | Edge
    severity: float = 1.0  # Higher = more severe

    def __hash__(self):
        return hash((frozenset([self.net1, self.net2]),
                     self.conflict_type,
                     hash(self.location)))


@dataclass
class Route:
    """A route for a single net."""
    net_name: str
    path: List[GridPoint]
    cost: float = 0.0
    vias: int = 0

    def get_edges(self) -> List[Edge]:
        """Get all edges in this route."""
        edges = []
        for i in range(len(self.path) - 1):
            edges.append(Edge(self.path[i], self.path[i + 1]))
        return edges

    def get_vertices(self) -> Set[GridPoint]:
        """Get all vertices in this route."""
        return set(self.path)


@dataclass
class CBSNode:
    """A node in the CBS constraint tree."""
    routes: Dict[str, Route]
    constraints: Set[Constraint]
    cost: float = 0.0
    conflicts: List[Conflict] = field(default_factory=list)
    parent: Optional['CBSNode'] = None
    depth: int = 0

    def __lt__(self, other):
        # For priority queue ordering
        return self.cost < other.cost

    def get_total_cost(self) -> float:
        """Calculate total cost of all routes."""
        return sum(r.cost for r in self.routes.values())


@dataclass
class CBSSolution:
    """Final CBS routing solution."""
    routes: Dict[str, Route]
    total_cost: float
    total_vias: int
    conflicts_resolved: int
    iterations: int
    llm_interventions: int = 0


@dataclass
class Net:
    """A net to be routed."""
    name: str
    pins: List[GridPoint]
    priority: int = 1  # Higher = route first
    width: float = 0.254  # mm
    clearance: float = 0.254  # mm
    net_class: str = "default"


# =============================================================================
# CBS Router
# =============================================================================

class CBSRouter:
    """
    Conflict-Based Search Router for multi-net PCB routing.

    Adapts MAPF (Multi-Agent Path Finding) CBS algorithm for PCB:
    - Nets are "agents"
    - Grid points and edges are shared resources
    - Conflicts resolved via constraint tree branching
    - LLM ConflictResolver handles complex cases

    Key differences from MAPF:
    - Continuous space discretized to routing grid
    - Layer dimension (3D routing)
    - Variable trace widths and clearances
    - Via costs and constraints
    """

    def __init__(
        self,
        conflict_resolver=None,  # Optional LLM ConflictResolverAgent
        grid_resolution: float = 0.254,  # mm (10 mil)
        max_iterations: int = 1000,
        llm_threshold: int = 5,  # Use LLM after this many conflicts
        via_cost: float = 10.0,
        layer_change_cost: float = 5.0
    ):
        self.conflict_resolver = conflict_resolver
        self.grid_resolution = grid_resolution
        self.max_iterations = max_iterations
        self.llm_threshold = llm_threshold
        self.via_cost = via_cost
        self.layer_change_cost = layer_change_cost

        # Routing grid
        self.grid_width: int = 0
        self.grid_height: int = 0
        self.num_layers: int = 2

        # Obstacles and blockages
        self.obstacles: Set[GridPoint] = set()
        self.blocked_edges: Set[Edge] = set()

        # Statistics
        self.stats = {
            "iterations": 0,
            "nodes_expanded": 0,
            "conflicts_detected": 0,
            "llm_calls": 0
        }

    def set_grid_bounds(
        self,
        width: float,
        height: float,
        num_layers: int = 2
    ):
        """Set the routing grid dimensions."""
        self.grid_width = int(width / self.grid_resolution)
        self.grid_height = int(height / self.grid_resolution)
        self.num_layers = num_layers
        logger.info(f"Grid set to {self.grid_width}x{self.grid_height}x{num_layers}")

    def add_obstacle(self, point: GridPoint):
        """Add an obstacle (blocked grid point)."""
        self.obstacles.add(point)

    def add_obstacles_from_components(self, components: List[Dict[str, Any]]):
        """Add obstacles from component placements."""
        for comp in components:
            x, y = comp.get("x", 0), comp.get("y", 0)
            w, h = comp.get("width", 5), comp.get("height", 5)

            # Block grid points covered by component
            for gx in range(int(x / self.grid_resolution),
                           int((x + w) / self.grid_resolution) + 1):
                for gy in range(int(y / self.grid_resolution),
                               int((y + h) / self.grid_resolution) + 1):
                    for layer in range(self.num_layers):
                        self.obstacles.add(GridPoint(
                            gx * self.grid_resolution,
                            gy * self.grid_resolution,
                            layer
                        ))

    # -------------------------------------------------------------------------
    # Main CBS Algorithm
    # -------------------------------------------------------------------------

    async def route_nets(
        self,
        nets: List[Net],
        constraints: Optional[Set[Constraint]] = None
    ) -> CBSSolution:
        """
        Route all nets using Conflict-Based Search.

        Algorithm:
        1. Create root node with independent A* routes
        2. While open list not empty:
           a. Pop lowest-cost node
           b. If no conflicts, return solution
           c. Choose conflict to resolve
           d. For each net in conflict:
              - Create child node with constraint
              - Re-route constrained net
              - Add to open list if valid
        """
        self.stats = {
            "iterations": 0,
            "nodes_expanded": 0,
            "conflicts_detected": 0,
            "llm_calls": 0
        }

        # Initialize root node with independent routes
        root = await self._create_root_node(nets, constraints or set())

        if root is None:
            logger.error("Failed to create initial routes")
            return CBSSolution(
                routes={},
                total_cost=float('inf'),
                total_vias=0,
                conflicts_resolved=0,
                iterations=0
            )

        # Priority queue (min-heap by cost)
        open_list: List[CBSNode] = []
        heapq.heappush(open_list, root)

        best_solution: Optional[CBSNode] = None

        while open_list and self.stats["iterations"] < self.max_iterations:
            self.stats["iterations"] += 1

            # Pop lowest-cost node
            current = heapq.heappop(open_list)
            self.stats["nodes_expanded"] += 1

            # Detect conflicts
            conflicts = self._detect_conflicts(current.routes)
            current.conflicts = conflicts
            self.stats["conflicts_detected"] += len(conflicts)

            # No conflicts = solution found
            if not conflicts:
                logger.info(f"CBS found solution in {self.stats['iterations']} iterations")
                return self._create_solution(current)

            # Track best (fewest conflicts)
            if best_solution is None or len(conflicts) < len(best_solution.conflicts):
                best_solution = current

            # Choose conflict to resolve
            conflict = self._choose_conflict(conflicts)

            # For complex conflicts, consult LLM
            if len(conflicts) > self.llm_threshold and self.conflict_resolver:
                await self._apply_llm_resolution(current, conflict, open_list, nets)
                continue

            # Standard CBS branching: create child for each net in conflict
            for net_name in [conflict.net1, conflict.net2]:
                child = await self._create_child_node(
                    current, net_name, conflict, nets
                )
                if child:
                    heapq.heappush(open_list, child)

        # Return best found (may have conflicts)
        logger.warning(f"CBS reached iteration limit, returning best with {len(best_solution.conflicts) if best_solution else 'no'} conflicts")
        return self._create_solution(best_solution) if best_solution else CBSSolution(
            routes={},
            total_cost=float('inf'),
            total_vias=0,
            conflicts_resolved=0,
            iterations=self.stats["iterations"]
        )

    async def _create_root_node(
        self,
        nets: List[Net],
        constraints: Set[Constraint]
    ) -> Optional[CBSNode]:
        """Create root node with independent routes for all nets."""
        routes = {}

        # Sort by priority (higher priority first)
        sorted_nets = sorted(nets, key=lambda n: -n.priority)

        for net in sorted_nets:
            route = self._route_single_net(net, constraints)
            if route:
                routes[net.name] = route
            else:
                logger.warning(f"Failed to route net {net.name} in root node")
                # Continue with partial solution

        if not routes:
            return None

        return CBSNode(
            routes=routes,
            constraints=constraints,
            cost=sum(r.cost for r in routes.values())
        )

    async def _create_child_node(
        self,
        parent: CBSNode,
        constrained_net: str,
        conflict: Conflict,
        nets: List[Net]
    ) -> Optional[CBSNode]:
        """Create child node with new constraint and re-routed net."""
        # Create new constraint
        new_constraint = Constraint(
            net_name=constrained_net,
            resource=conflict.location,
            constraint_type=conflict.conflict_type
        )

        # Copy parent's constraints and add new one
        new_constraints = parent.constraints.copy()
        new_constraints.add(new_constraint)

        # Copy routes
        new_routes = parent.routes.copy()

        # Find the net to re-route
        net = next((n for n in nets if n.name == constrained_net), None)
        if not net:
            return None

        # Re-route with new constraints
        new_route = self._route_single_net(net, new_constraints)
        if not new_route:
            # Can't satisfy constraints
            return None

        new_routes[constrained_net] = new_route

        return CBSNode(
            routes=new_routes,
            constraints=new_constraints,
            cost=sum(r.cost for r in new_routes.values()),
            parent=parent,
            depth=parent.depth + 1
        )

    async def _apply_llm_resolution(
        self,
        node: CBSNode,
        conflict: Conflict,
        open_list: List[CBSNode],
        nets: List[Net]
    ):
        """Use LLM ConflictResolver for complex conflicts."""
        if not self.conflict_resolver:
            return

        self.stats["llm_calls"] += 1

        try:
            # Prepare conflict info for LLM
            conflict_info = {
                "type": conflict.conflict_type.value,
                "nets": [conflict.net1, conflict.net2],
                "location": {
                    "x": conflict.location.x if isinstance(conflict.location, GridPoint) else conflict.location.start.x,
                    "y": conflict.location.y if isinstance(conflict.location, GridPoint) else conflict.location.start.y,
                    "layer": conflict.location.layer if isinstance(conflict.location, GridPoint) else conflict.location.start.layer
                },
                "severity": conflict.severity,
                "total_conflicts": len(node.conflicts)
            }

            # Get LLM resolution strategy
            resolution = await self.conflict_resolver.resolve_conflict(
                conflict_info,
                node.routes,
                {}  # constraints
            )

            # Apply resolution strategy
            if resolution.strategy == "ripup_reroute":
                # Rip up the recommended net
                net_to_ripup = resolution.affected_nets[0] if resolution.affected_nets else conflict.net1
                net = next((n for n in nets if n.name == net_to_ripup), None)

                if net:
                    # Add constraints from LLM recommendation
                    new_constraints = node.constraints.copy()
                    for detour in resolution.detour_points:
                        new_constraints.add(Constraint(
                            net_name=net_to_ripup,
                            resource=GridPoint(detour["x"], detour["y"], detour.get("layer", 0)),
                            constraint_type=ConflictType.VERTEX
                        ))

                    new_routes = node.routes.copy()
                    new_route = self._route_single_net(net, new_constraints)
                    if new_route:
                        new_routes[net_to_ripup] = new_route
                        child = CBSNode(
                            routes=new_routes,
                            constraints=new_constraints,
                            cost=sum(r.cost for r in new_routes.values()),
                            parent=node,
                            depth=node.depth + 1
                        )
                        heapq.heappush(open_list, child)

            elif resolution.strategy == "layer_change":
                # Route one net on different layer
                net_to_move = resolution.affected_nets[0] if resolution.affected_nets else conflict.net1
                net = next((n for n in nets if n.name == net_to_move), None)

                if net:
                    # Force to different layer via constraint
                    current_layer = conflict.location.layer if isinstance(conflict.location, GridPoint) else 0
                    target_layer = (current_layer + 1) % self.num_layers

                    # This is a simplified approach - full implementation would
                    # modify the net's layer preference and re-route
                    new_constraints = node.constraints.copy()
                    new_constraints.add(Constraint(
                        net_name=net_to_move,
                        resource=conflict.location,
                        constraint_type=ConflictType.LAYER
                    ))

                    new_routes = node.routes.copy()
                    new_route = self._route_single_net(net, new_constraints,
                                                       preferred_layer=target_layer)
                    if new_route:
                        new_routes[net_to_move] = new_route
                        child = CBSNode(
                            routes=new_routes,
                            constraints=new_constraints,
                            cost=sum(r.cost for r in new_routes.values()),
                            parent=node,
                            depth=node.depth + 1
                        )
                        heapq.heappush(open_list, child)

            elif resolution.strategy == "detour":
                # Both nets get detour constraints
                for net_name in [conflict.net1, conflict.net2]:
                    net = next((n for n in nets if n.name == net_name), None)
                    if net:
                        new_constraints = node.constraints.copy()
                        new_constraints.add(Constraint(
                            net_name=net_name,
                            resource=conflict.location,
                            constraint_type=conflict.conflict_type
                        ))

                        new_routes = node.routes.copy()
                        new_route = self._route_single_net(net, new_constraints)
                        if new_route:
                            new_routes[net_name] = new_route
                            child = CBSNode(
                                routes=new_routes,
                                constraints=new_constraints,
                                cost=sum(r.cost for r in new_routes.values()),
                                parent=node,
                                depth=node.depth + 1
                            )
                            heapq.heappush(open_list, child)
                        break  # Only need one successful detour

        except Exception as e:
            logger.error(f"LLM resolution failed: {e}")
            # Fall back to standard branching
            for net_name in [conflict.net1, conflict.net2]:
                child = await self._create_child_node(node, net_name, conflict, nets)
                if child:
                    heapq.heappush(open_list, child)

    # -------------------------------------------------------------------------
    # Single-Net Routing (A* or Manhattan)
    # -------------------------------------------------------------------------

    def _route_single_net(
        self,
        net: Net,
        constraints: Set[Constraint],
        preferred_layer: Optional[int] = None
    ) -> Optional[Route]:
        """
        Route a single net using A* with Manhattan distance heuristic.
        Respects constraints from CBS.
        """
        if len(net.pins) < 2:
            return Route(net_name=net.name, path=net.pins, cost=0)

        # Get constraints for this net
        net_constraints = {c for c in constraints if c.net_name == net.name}
        blocked_vertices = {c.resource for c in net_constraints
                          if isinstance(c.resource, GridPoint)}
        blocked_edges_local = {c.resource for c in net_constraints
                              if isinstance(c.resource, Edge)}

        # Combine with global obstacles
        all_blocked = self.obstacles | blocked_vertices
        all_blocked_edges = self.blocked_edges | blocked_edges_local

        # Route between each pair of pins (simple MST approach)
        # For multi-pin nets, we use Steiner tree approximation
        full_path: List[GridPoint] = []
        total_cost = 0.0
        total_vias = 0

        # Start with first pin
        current_point = net.pins[0]
        remaining_pins = list(net.pins[1:])
        full_path.append(current_point)

        while remaining_pins:
            # Find nearest unconnected pin
            nearest_pin = min(remaining_pins,
                            key=lambda p: self._manhattan_distance(current_point, p))

            # A* from current to nearest
            segment, cost, vias = self._astar_route(
                current_point,
                nearest_pin,
                all_blocked,
                all_blocked_edges,
                preferred_layer
            )

            if segment is None:
                logger.warning(f"Failed to route {net.name} segment")
                return None

            # Add segment (skip first point as it's already in path)
            full_path.extend(segment[1:])
            total_cost += cost
            total_vias += vias

            # Update current and remaining
            current_point = nearest_pin
            remaining_pins.remove(nearest_pin)

        return Route(
            net_name=net.name,
            path=full_path,
            cost=total_cost,
            vias=total_vias
        )

    def _astar_route(
        self,
        start: GridPoint,
        goal: GridPoint,
        blocked: Set[GridPoint],
        blocked_edges: Set[Edge],
        preferred_layer: Optional[int] = None
    ) -> Tuple[Optional[List[GridPoint]], float, int]:
        """A* routing between two points."""
        # Priority queue: (f_score, g_score, point, path, vias)
        open_set: List[Tuple[float, float, GridPoint, List[GridPoint], int]] = []
        heapq.heappush(open_set, (
            self._manhattan_distance(start, goal),
            0.0,
            start,
            [start],
            0
        ))

        # Closed set
        closed: Set[GridPoint] = set()

        # Best g_score for each point
        g_scores: Dict[GridPoint, float] = {start: 0.0}

        while open_set:
            f, g, current, path, vias = heapq.heappop(open_set)

            if current in closed:
                continue

            closed.add(current)

            # Goal reached
            if current.x == goal.x and current.y == goal.y:
                # May need layer change at end
                if current.layer != goal.layer:
                    path.append(goal)
                    vias += 1
                return path, g, vias

            # Expand neighbors
            for neighbor, edge_cost, is_via in self._get_neighbors(current, preferred_layer):
                if neighbor in closed or neighbor in blocked:
                    continue

                edge = Edge(current, neighbor)
                if edge in blocked_edges:
                    continue

                tentative_g = g + edge_cost

                if neighbor not in g_scores or tentative_g < g_scores[neighbor]:
                    g_scores[neighbor] = tentative_g
                    f_score = tentative_g + self._manhattan_distance(neighbor, goal)
                    new_path = path + [neighbor]
                    new_vias = vias + (1 if is_via else 0)

                    heapq.heappush(open_set, (
                        f_score,
                        tentative_g,
                        neighbor,
                        new_path,
                        new_vias
                    ))

        # No path found
        return None, float('inf'), 0

    def _get_neighbors(
        self,
        point: GridPoint,
        preferred_layer: Optional[int] = None
    ) -> List[Tuple[GridPoint, float, bool]]:
        """Get neighboring grid points with costs."""
        neighbors = []

        # 4-directional movement on same layer
        for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            nx = point.x + dx * self.grid_resolution
            ny = point.y + dy * self.grid_resolution

            # Check bounds
            if 0 <= nx <= self.grid_width * self.grid_resolution and \
               0 <= ny <= self.grid_height * self.grid_resolution:
                neighbor = GridPoint(nx, ny, point.layer)
                cost = self.grid_resolution

                # Prefer staying on preferred layer
                if preferred_layer is not None and point.layer != preferred_layer:
                    cost *= 1.5

                neighbors.append((neighbor, cost, False))

        # Layer transitions (vias)
        for layer in range(self.num_layers):
            if layer != point.layer:
                neighbor = GridPoint(point.x, point.y, layer)
                cost = self.via_cost + abs(layer - point.layer) * self.layer_change_cost

                # Prefer moving to preferred layer
                if preferred_layer is not None and layer == preferred_layer:
                    cost *= 0.5

                neighbors.append((neighbor, cost, True))

        return neighbors

    def _manhattan_distance(self, a: GridPoint, b: GridPoint) -> float:
        """Manhattan distance heuristic."""
        return abs(a.x - b.x) + abs(a.y - b.y) + abs(a.layer - b.layer) * self.via_cost

    # -------------------------------------------------------------------------
    # Conflict Detection
    # -------------------------------------------------------------------------

    def _detect_conflicts(self, routes: Dict[str, Route]) -> List[Conflict]:
        """Detect all conflicts between routes."""
        conflicts = []
        route_list = list(routes.items())

        for i, (name1, route1) in enumerate(route_list):
            for name2, route2 in route_list[i + 1:]:
                # Vertex conflicts
                vertices1 = route1.get_vertices()
                vertices2 = route2.get_vertices()
                shared_vertices = vertices1 & vertices2

                for vertex in shared_vertices:
                    conflicts.append(Conflict(
                        net1=name1,
                        net2=name2,
                        conflict_type=ConflictType.VERTEX,
                        location=vertex,
                        severity=1.0
                    ))

                # Edge conflicts
                edges1 = set(route1.get_edges())
                edges2 = set(route2.get_edges())
                shared_edges = edges1 & edges2

                for edge in shared_edges:
                    conflicts.append(Conflict(
                        net1=name1,
                        net2=name2,
                        conflict_type=ConflictType.EDGE,
                        location=edge,
                        severity=1.5  # Edge conflicts more severe
                    ))

        return conflicts

    def _choose_conflict(self, conflicts: List[Conflict]) -> Conflict:
        """Choose which conflict to resolve first (highest severity)."""
        return max(conflicts, key=lambda c: c.severity)

    # -------------------------------------------------------------------------
    # Solution Creation
    # -------------------------------------------------------------------------

    def _create_solution(self, node: CBSNode) -> CBSSolution:
        """Create final solution from CBS node."""
        total_vias = sum(r.vias for r in node.routes.values())

        return CBSSolution(
            routes=node.routes,
            total_cost=node.cost,
            total_vias=total_vias,
            conflicts_resolved=len(node.constraints),
            iterations=self.stats["iterations"],
            llm_interventions=self.stats["llm_calls"]
        )


# =============================================================================
# Factory Function
# =============================================================================

def create_cbs_router(
    conflict_resolver=None,
    grid_resolution: float = 0.254,
    max_iterations: int = 1000
) -> CBSRouter:
    """Factory function to create a CBS router instance."""
    return CBSRouter(
        conflict_resolver=conflict_resolver,
        grid_resolution=grid_resolution,
        max_iterations=max_iterations
    )


# =============================================================================
# Integration with PathFinder Router
# =============================================================================

class HybridRouter:
    """
    Hybrid router combining PathFinder and CBS approaches.

    Strategy:
    1. Use PathFinder for initial greedy routing
    2. Switch to CBS for conflict-heavy regions
    3. LLM orchestrates which approach to use where
    """

    def __init__(
        self,
        pathfinder,  # PathFinderRouter instance
        cbs_router: CBSRouter,
        conflict_threshold: int = 10
    ):
        self.pathfinder = pathfinder
        self.cbs_router = cbs_router
        self.conflict_threshold = conflict_threshold

    async def route_adaptively(
        self,
        nets: List[Net],
        components: List[Dict[str, Any]]
    ) -> Dict[str, Route]:
        """
        Adaptively route using best approach per region.

        1. Group nets by region
        2. For sparse regions: use PathFinder
        3. For dense regions: use CBS
        4. Merge solutions
        """
        # Set up CBS grid from components
        if components:
            # Find bounding box
            max_x = max(c.get("x", 0) + c.get("width", 0) for c in components)
            max_y = max(c.get("y", 0) + c.get("height", 0) for c in components)
            self.cbs_router.set_grid_bounds(max_x + 50, max_y + 50)
            self.cbs_router.add_obstacles_from_components(components)

        # Try PathFinder first (faster)
        pathfinder_solution = await self.pathfinder.route_with_negotiation(nets)

        # Check conflict level
        if pathfinder_solution.total_conflicts <= self.conflict_threshold:
            # PathFinder solution acceptable
            return pathfinder_solution.routes

        # Too many conflicts - switch to CBS for precision
        logger.info(f"PathFinder had {pathfinder_solution.total_conflicts} conflicts, switching to CBS")

        cbs_solution = await self.cbs_router.route_nets(nets)

        # Return better solution
        if len(cbs_solution.routes) > len(pathfinder_solution.routes):
            return cbs_solution.routes

        return pathfinder_solution.routes


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    "CBSRouter",
    "CBSSolution",
    "CBSNode",
    "Conflict",
    "ConflictType",
    "Constraint",
    "GridPoint",
    "Edge",
    "Route",
    "Net",
    "HybridRouter",
    "create_cbs_router"
]
