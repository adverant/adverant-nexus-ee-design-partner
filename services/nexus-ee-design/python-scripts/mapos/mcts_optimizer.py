#!/usr/bin/env python3
"""
MCTS Optimizer - Monte Carlo Tree Search for PCB optimization.

This module implements MCTS with LLM-guided expansion and value estimation:
1. Selection: UCB1 formula to balance exploration/exploitation
2. Expansion: LLM agents generate candidate modifications
3. Simulation: Quick DRC + heuristic evaluation
4. Backpropagation: Update statistics up the tree

Inspired by AlphaGo's MCTS and recent LLM reasoning work (Tree of Thoughts).
"""

import math
import random
import asyncio
import sys
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Set
from pathlib import Path
import json
import time

# Add script directory to path for local imports
SCRIPT_DIR = Path(__file__).parent.resolve()
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from pcb_state import PCBState, PCBModification, DRCResult, create_random_modification
from generator_agents import AgentPool, GeneratorAgent, GenerationResult


@dataclass
class MCTSStats:
    """Statistics for an MCTS node."""
    visits: int = 0
    total_reward: float = 0.0
    best_reward: float = 0.0

    @property
    def average_reward(self) -> float:
        """Get average reward across visits."""
        if self.visits == 0:
            return 0.0
        return self.total_reward / self.visits

    def update(self, reward: float) -> None:
        """Update stats with new reward."""
        self.visits += 1
        self.total_reward += reward
        self.best_reward = max(self.best_reward, reward)


@dataclass
class MCTSNode:
    """
    A node in the MCTS tree.

    Each node represents a PCB state that can be explored further.
    """
    state: PCBState
    parent: Optional['MCTSNode'] = None
    modification: Optional[PCBModification] = None  # Modification that led to this state
    children: List['MCTSNode'] = field(default_factory=list)
    stats: MCTSStats = field(default_factory=MCTSStats)
    untried_modifications: List[PCBModification] = field(default_factory=list)
    depth: int = 0

    def is_leaf(self) -> bool:
        """Check if this is a leaf node (no children)."""
        return len(self.children) == 0

    def is_fully_expanded(self) -> bool:
        """Check if all modifications have been tried."""
        return len(self.untried_modifications) == 0

    def is_terminal(self, target_violations: int = 100) -> bool:
        """Check if this is a terminal state (goal reached or max depth)."""
        if self.depth >= 20:  # Max depth
            return True
        if hasattr(self.state, '_drc_result') and self.state._drc_result:
            return self.state._drc_result.total_violations <= target_violations
        return False

    def get_ucb1_score(self, exploration_constant: float = 1.414) -> float:
        """
        Calculate UCB1 score for node selection.

        UCB1 = average_reward + c * sqrt(ln(parent_visits) / visits)

        Args:
            exploration_constant: c parameter (sqrt(2) is theoretical optimal)

        Returns:
            UCB1 score (higher = more promising/less explored)
        """
        if self.stats.visits == 0:
            return float('inf')  # Unexplored nodes have infinite priority

        if self.parent is None:
            return self.stats.average_reward

        exploitation = self.stats.average_reward
        exploration = exploration_constant * math.sqrt(
            math.log(self.parent.stats.visits) / self.stats.visits
        )

        return exploitation + exploration

    def best_child(self, exploration_constant: float = 1.414) -> 'MCTSNode':
        """Get best child according to UCB1."""
        return max(self.children, key=lambda c: c.get_ucb1_score(exploration_constant))

    def add_child(self, modification: PCBModification) -> 'MCTSNode':
        """Add a child node from applying a modification."""
        new_state = self.state.apply_modification(modification)
        child = MCTSNode(
            state=new_state,
            parent=self,
            modification=modification,
            depth=self.depth + 1
        )
        self.children.append(child)
        return child

    def __repr__(self) -> str:
        return f"MCTSNode(depth={self.depth}, visits={self.stats.visits}, reward={self.stats.average_reward:.3f})"


class MCTSOptimizer:
    """
    Monte Carlo Tree Search optimizer for PCB layouts.

    Uses MCTS with:
    - UCB1 selection
    - LLM-guided expansion (via AgentPool)
    - DRC-based simulation
    - Full backpropagation
    """

    def __init__(
        self,
        pcb_path: str,
        agent_pool: Optional[AgentPool] = None,
        exploration_constant: float = 1.414,
        target_violations: int = 100,
        max_iterations: int = 100,
        simulation_rollout_depth: int = 3
    ):
        """
        Initialize MCTS optimizer.

        Args:
            pcb_path: Path to PCB file
            agent_pool: Pool of generator agents (creates default if None)
            exploration_constant: UCB1 exploration parameter
            target_violations: Target violation count (terminal condition)
            max_iterations: Maximum MCTS iterations
            simulation_rollout_depth: Depth of random rollout in simulation
        """
        self.pcb_path = Path(pcb_path)
        self.agent_pool = agent_pool or AgentPool()
        self.exploration_constant = exploration_constant
        self.target_violations = target_violations
        self.max_iterations = max_iterations
        self.simulation_rollout_depth = simulation_rollout_depth

        # Initialize root node
        initial_state = PCBState.from_file(str(pcb_path))
        self.root = MCTSNode(state=initial_state)

        # Statistics
        self.iteration_count = 0
        self.best_state: Optional[PCBState] = None
        self.best_violations = float('inf')
        self.history: List[Dict] = []

        # Seen states for deduplication
        self.seen_states: Set[str] = set()

    async def search(self) -> PCBState:
        """
        Run MCTS search to find optimal PCB configuration.

        Returns:
            Best PCBState found
        """
        print(f"\nStarting MCTS search (max {self.max_iterations} iterations)")
        print(f"Target: {self.target_violations} violations")

        # Run initial DRC
        initial_drc = self.root.state.run_drc()
        print(f"Initial violations: {initial_drc.total_violations}")
        self.best_violations = initial_drc.total_violations
        self.best_state = self.root.state

        start_time = time.time()

        for iteration in range(self.max_iterations):
            self.iteration_count = iteration + 1

            # 1. Selection: Traverse tree using UCB1
            node = self._select(self.root)

            # 2. Expansion: Add new children using LLM agents
            if not node.is_terminal(self.target_violations) and node.is_fully_expanded():
                await self._expand(node)
                if node.children:
                    node = random.choice(node.children)

            # 3. Simulation: Quick evaluation
            reward = await self._simulate(node)

            # 4. Backpropagation: Update statistics
            self._backpropagate(node, reward)

            # Track best state
            if node.state._drc_result:
                violations = node.state._drc_result.total_violations
                if violations < self.best_violations:
                    self.best_violations = violations
                    self.best_state = node.state
                    print(f"  Iteration {iteration+1}: New best = {violations} violations")

            # Record history
            self.history.append({
                'iteration': iteration + 1,
                'best_violations': self.best_violations,
                'tree_size': self._count_nodes(self.root),
                'depth': node.depth
            })

            # Early termination
            if self.best_violations <= self.target_violations:
                print(f"\n  Target reached at iteration {iteration+1}!")
                break

            # Progress update every 10 iterations
            if (iteration + 1) % 10 == 0:
                elapsed = time.time() - start_time
                print(f"  Iteration {iteration+1}: best={self.best_violations}, "
                      f"tree_size={self._count_nodes(self.root)}, "
                      f"time={elapsed:.1f}s")

        total_time = time.time() - start_time
        print(f"\nMCTS complete: {self.iteration_count} iterations in {total_time:.1f}s")
        print(f"Best result: {self.best_violations} violations")

        return self.best_state

    def _select(self, node: MCTSNode) -> MCTSNode:
        """
        Select a node to expand using UCB1.

        Traverses tree from root, selecting best child at each level.
        """
        while not node.is_leaf():
            if not node.is_fully_expanded():
                return node
            node = node.best_child(self.exploration_constant)
        return node

    async def _expand(self, node: MCTSNode) -> None:
        """
        Expand a node by generating new modifications.

        Uses LLM agents to generate promising modifications.
        """
        # Get DRC for current state
        drc = node.state.run_drc()

        # Generate modifications from all agents
        results = await self.agent_pool.generate_all(node.state, drc, max_modifications_per_agent=3)

        # Collect all unique modifications
        all_mods = []
        for result in results:
            for mod in result.modifications:
                # Check for duplicate states
                test_state = node.state.apply_modification(mod)
                state_hash = test_state.get_hash()
                if state_hash not in self.seen_states:
                    all_mods.append(mod)
                    self.seen_states.add(state_hash)

        # Also add some random modifications for exploration
        for _ in range(2):
            random_mod = create_random_modification(node.state)
            test_state = node.state.apply_modification(random_mod)
            state_hash = test_state.get_hash()
            if state_hash not in self.seen_states:
                all_mods.append(random_mod)
                self.seen_states.add(state_hash)

        # Store untried modifications
        node.untried_modifications = all_mods

        # Create children for each modification
        for mod in all_mods[:5]:  # Limit branching factor
            node.add_child(mod)

    async def _simulate(self, node: MCTSNode) -> float:
        """
        Simulate from node to estimate value.

        Uses quick DRC + heuristic evaluation.
        """
        state = node.state

        # Run DRC
        drc = state.run_drc()

        # Base reward: inverse of violations (normalized)
        base_reward = 1.0 / (1.0 + drc.total_violations / self.best_violations)

        # Bonus for improvement over current best
        improvement_bonus = 0.0
        if drc.total_violations < self.best_violations:
            improvement = (self.best_violations - drc.total_violations) / self.best_violations
            improvement_bonus = improvement * 0.3

        # Penalty for too many errors
        error_penalty = 0.0
        if drc.errors > 500:
            error_penalty = 0.1

        # Bonus for low unconnected count
        connection_bonus = 0.0
        if drc.unconnected < 5:
            connection_bonus = 0.1

        total_reward = base_reward + improvement_bonus - error_penalty + connection_bonus
        return max(0.0, min(1.0, total_reward))

    def _backpropagate(self, node: MCTSNode, reward: float) -> None:
        """
        Backpropagate reward up the tree.

        Updates statistics for all nodes from leaf to root.
        """
        while node is not None:
            node.stats.update(reward)
            node = node.parent

    def _count_nodes(self, node: MCTSNode) -> int:
        """Count total nodes in subtree."""
        count = 1
        for child in node.children:
            count += self._count_nodes(child)
        return count

    def get_best_path(self) -> List[PCBModification]:
        """
        Get the path of modifications to the best state.

        Returns:
            List of modifications from root to best state
        """
        if self.best_state is None:
            return []
        return self.best_state.modifications

    def get_statistics(self) -> Dict:
        """Get search statistics."""
        return {
            'iterations': self.iteration_count,
            'tree_size': self._count_nodes(self.root),
            'best_violations': self.best_violations,
            'target_violations': self.target_violations,
            'states_explored': len(self.seen_states),
            'best_path_length': len(self.get_best_path()),
            'agent_stats': self.agent_pool.get_statistics()
        }

    def save_results(self, output_path: str) -> None:
        """Save search results to file."""
        results = {
            'pcb_path': str(self.pcb_path),
            'statistics': self.get_statistics(),
            'best_modifications': [m.to_dict() for m in self.get_best_path()],
            'history': self.history
        }

        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"Saved results to: {output_path}")


class ProgressiveMCTS(MCTSOptimizer):
    """
    Progressive MCTS with increasing iteration budget.

    Starts with short searches and progressively increases depth,
    allowing early termination if good solutions are found quickly.
    """

    def __init__(
        self,
        pcb_path: str,
        agent_pool: Optional[AgentPool] = None,
        initial_iterations: int = 20,
        max_iterations: int = 200,
        iteration_multiplier: float = 1.5,
        target_violations: int = 100
    ):
        super().__init__(
            pcb_path=pcb_path,
            agent_pool=agent_pool,
            max_iterations=max_iterations,
            target_violations=target_violations
        )
        self.initial_iterations = initial_iterations
        self.iteration_multiplier = iteration_multiplier

    async def progressive_search(self) -> PCBState:
        """
        Run progressive MCTS with increasing budgets.

        Returns:
            Best PCBState found
        """
        current_budget = self.initial_iterations
        total_iterations = 0

        while total_iterations < self.max_iterations:
            print(f"\n--- Progressive phase: {current_budget} iterations ---")

            self.max_iterations = min(current_budget, self.max_iterations - total_iterations)
            await self.search()

            total_iterations += self.iteration_count

            # Check if target reached
            if self.best_violations <= self.target_violations:
                print(f"Target reached after {total_iterations} total iterations")
                break

            # Increase budget for next phase
            current_budget = int(current_budget * self.iteration_multiplier)

        return self.best_state


async def run_mcts_optimization(
    pcb_path: str,
    max_iterations: int = 100,
    target_violations: int = 100,
    output_json: Optional[str] = None
) -> PCBState:
    """
    Run MCTS optimization on a PCB file.

    Args:
        pcb_path: Path to PCB file
        max_iterations: Maximum MCTS iterations
        target_violations: Target violation count
        output_json: Optional path to save results

    Returns:
        Best PCBState found
    """
    print("\n" + "="*60)
    print("MCTS PCB OPTIMIZER")
    print("="*60)

    optimizer = MCTSOptimizer(
        pcb_path=pcb_path,
        max_iterations=max_iterations,
        target_violations=target_violations
    )

    best_state = await optimizer.search()

    print(f"\nFinal Statistics:")
    stats = optimizer.get_statistics()
    print(f"  Iterations: {stats['iterations']}")
    print(f"  Tree size: {stats['tree_size']}")
    print(f"  States explored: {stats['states_explored']}")
    print(f"  Best violations: {stats['best_violations']}")
    print(f"  Best path length: {stats['best_path_length']}")

    if output_json:
        optimizer.save_results(output_json)

    return best_state


if __name__ == '__main__':
    import sys

    if len(sys.argv) < 2:
        print("Usage: python mcts_optimizer.py <path_to.kicad_pcb> [max_iterations] [target_violations]")
        sys.exit(1)

    pcb_path = sys.argv[1]
    max_iterations = int(sys.argv[2]) if len(sys.argv) > 2 else 50
    target_violations = int(sys.argv[3]) if len(sys.argv) > 3 else 100

    asyncio.run(run_mcts_optimization(
        pcb_path,
        max_iterations=max_iterations,
        target_violations=target_violations,
        output_json=f"{pcb_path.rsplit('.', 1)[0]}_mcts_results.json"
    ))
