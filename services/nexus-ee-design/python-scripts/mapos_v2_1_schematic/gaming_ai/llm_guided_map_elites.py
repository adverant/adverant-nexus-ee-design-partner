"""
MAPO v2.1 Schematic - LLM-Guided MAP-Elites

Quality-diversity optimization with LLM guidance for schematic generation.
Combines MAP-Elites archive exploration with Opus 4.5 for intelligent
mutation selection and guidance.

Author: Nexus EE Design Team
"""

import asyncio
import json
import logging
import os
import random
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import httpx
import numpy as np

from ..core.schematic_state import SchematicState, SchematicSolution, FitnessScores
from ..core.config import SchematicMAPOConfig, get_config

logger = logging.getLogger(__name__)


@dataclass
class ArchiveCell:
    """
    Cell in the MAP-Elites archive.
    
    Stores the best solution found for a particular behavioral region.
    """
    solution: Optional[SchematicSolution] = None
    cell_index: Tuple[int, ...] = field(default_factory=tuple)
    visits: int = 0
    improvements: int = 0
    last_updated: str = ""
    
    @property
    def fitness(self) -> float:
        if self.solution:
            return self.solution.fitness
        return 0.0
    
    @property
    def is_empty(self) -> bool:
        return self.solution is None


@dataclass
class MutationGuidance:
    """LLM-generated guidance for mutation."""
    mutation_type: str  # e.g., "connection_fix", "component_swap", "topology_change"
    target_component: Optional[str] = None
    target_connection: Optional[Tuple[str, str, str, str]] = None
    description: str = ""
    confidence: float = 0.5
    reasoning: str = ""


class MAPElitesArchive:
    """
    N-dimensional archive for MAP-Elites quality-diversity optimization.
    
    Each cell stores the best solution found for a particular region
    of the behavioral descriptor space.
    """
    
    def __init__(
        self,
        dims: Tuple[int, ...] = (5, 5, 5, 5, 5),
        behavior_dims: int = 10,
    ):
        """
        Initialize the archive.
        
        Args:
            dims: Archive dimensions (resolution in each behavior axis)
            behavior_dims: Total dimensions of behavioral descriptor
        """
        self.dims = dims
        self.behavior_dims = behavior_dims
        self.archive: Dict[Tuple[int, ...], ArchiveCell] = {}
        self.total_evaluations = 0
        self.improvements = 0
        self.best_fitness = 0.0
        self.best_solution: Optional[SchematicSolution] = None
    
    def _behavior_to_cell(self, behavior: np.ndarray) -> Tuple[int, ...]:
        """Convert behavioral descriptor to cell indices."""
        # Map behavior values [0, 1] to cell indices
        # Use first len(dims) dimensions of behavior
        cell = []
        for i, dim_size in enumerate(self.dims):
            if i < len(behavior):
                idx = int(behavior[i] * (dim_size - 1))
                idx = max(0, min(idx, dim_size - 1))  # Clamp
            else:
                idx = 0
            cell.append(idx)
        return tuple(cell)
    
    def add(self, solution: SchematicSolution) -> bool:
        """
        Try to add a solution to the archive.
        
        Returns True if solution was added (new cell or improvement).
        """
        if solution.behavior_descriptor is None:
            logger.warning("Cannot add solution without behavior descriptor")
            return False
        
        cell_idx = self._behavior_to_cell(solution.behavior_descriptor)
        self.total_evaluations += 1
        
        # Get or create cell
        if cell_idx not in self.archive:
            self.archive[cell_idx] = ArchiveCell(cell_index=cell_idx)
        
        cell = self.archive[cell_idx]
        cell.visits += 1
        
        # Check if improvement
        if cell.is_empty or solution.fitness > cell.fitness:
            cell.solution = solution
            cell.improvements += 1
            cell.last_updated = datetime.now().isoformat()
            solution.is_elite = True
            solution.elite_cell = cell_idx
            self.improvements += 1
            
            # Update global best
            if solution.fitness > self.best_fitness:
                self.best_fitness = solution.fitness
                self.best_solution = solution
            
            logger.debug(f"Added to archive cell {cell_idx}: fitness={solution.fitness:.3f}")
            return True
        
        return False
    
    def get_best_solution(self) -> Optional[SchematicSolution]:
        """Get the highest-fitness solution in the archive."""
        return self.best_solution
    
    def get_random_elite(self) -> Optional[SchematicSolution]:
        """Get a random non-empty cell's solution."""
        non_empty = [c for c in self.archive.values() if not c.is_empty]
        if not non_empty:
            return None
        return random.choice(non_empty).solution
    
    def get_cells_below_threshold(self, threshold: float) -> List[ArchiveCell]:
        """Get cells with fitness below threshold (for improvement)."""
        return [
            c for c in self.archive.values()
            if not c.is_empty and c.fitness < threshold
        ]
    
    def coverage_stats(self) -> Dict[str, Any]:
        """Get archive coverage statistics."""
        total_cells = 1
        for d in self.dims:
            total_cells *= d
        
        filled_cells = len([c for c in self.archive.values() if not c.is_empty])
        
        fitness_values = [c.fitness for c in self.archive.values() if not c.is_empty]
        
        return {
            "total_cells": total_cells,
            "filled_cells": filled_cells,
            "coverage": filled_cells / total_cells,
            "total_evaluations": self.total_evaluations,
            "improvements": self.improvements,
            "best_fitness": self.best_fitness,
            "mean_fitness": np.mean(fitness_values) if fitness_values else 0.0,
            "median_fitness": np.median(fitness_values) if fitness_values else 0.0,
        }


class LLMGuidedSchematicMAPElites:
    """
    LLM-Guided MAP-Elites for schematic optimization.
    
    Combines MAP-Elites quality-diversity with LLM intelligence:
    - LLM analyzes archive and recommends cells to explore
    - LLM guides mutations based on smoke test failures
    - LLM provides reasoning for design decisions
    """
    
    def __init__(self, config: Optional[SchematicMAPOConfig] = None):
        """Initialize with configuration."""
        self.config = config or get_config()
        self.archive = MAPElitesArchive(
            dims=self.config.map_elites_archive_dims,
            behavior_dims=self.config.map_elites_behavior_dims,
        )
        self._http_client: Optional[httpx.AsyncClient] = None
    
    async def _ensure_http_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client."""
        if self._http_client is None or self._http_client.is_closed:
            self._http_client = httpx.AsyncClient(
                timeout=httpx.Timeout(self.config.llm_timeout)
            )
        return self._http_client
    
    async def close(self):
        """Close HTTP client."""
        if self._http_client and not self._http_client.is_closed:
            await self._http_client.aclose()
            self._http_client = None
    
    async def _call_llm(self, prompt: str, system: str = "") -> str:
        """Call LLM via OpenRouter."""
        if not self.config.openrouter_api_key:
            logger.warning("No OpenRouter API key, returning empty response")
            return ""
        
        client = await self._ensure_http_client()
        
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        
        try:
            response = await client.post(
                self.config.openrouter_base_url,
                headers={
                    "Authorization": f"Bearer {self.config.openrouter_api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": self.config.openrouter_model,
                    "messages": messages,
                    "temperature": self.config.llm_temperature,
                    "max_tokens": self.config.llm_max_tokens,
                }
            )
            
            if response.status_code == 200:
                data = response.json()
                return data["choices"][0]["message"]["content"]
            else:
                logger.warning(f"LLM call failed: {response.status_code}")
                return ""
                
        except Exception as e:
            logger.warning(f"LLM call error: {e}")
            return ""
    
    async def select_cells_for_exploration(
        self,
        validation_issues: List[str],
        num_cells: int = 3,
    ) -> List[ArchiveCell]:
        """
        LLM analyzes archive and recommends cells to explore.
        
        Uses the archive's coverage stats and current validation issues
        to intelligently select where to explore next.
        """
        stats = self.archive.coverage_stats()
        
        # Get cells with room for improvement
        low_fitness_cells = self.archive.get_cells_below_threshold(
            stats["mean_fitness"] + 0.1
        )
        
        if not low_fitness_cells and not validation_issues:
            # Archive doing well, explore randomly
            cells = []
            for _ in range(num_cells):
                solution = self.archive.get_random_elite()
                if solution and solution.elite_cell:
                    cell = self.archive.archive.get(solution.elite_cell)
                    if cell:
                        cells.append(cell)
            return cells[:num_cells]
        
        # Build prompt for LLM
        prompt = f"""
Analyze this MAP-Elites archive for schematic optimization:

Archive Statistics:
- Coverage: {stats['coverage']:.1%}
- Best fitness: {stats['best_fitness']:.3f}
- Mean fitness: {stats['mean_fitness']:.3f}
- Total evaluations: {stats['total_evaluations']}

Current Issues (from smoke test):
{chr(10).join(f"- {issue}" for issue in validation_issues[:10])}

Cells with low fitness:
{chr(10).join(f"- Cell {c.cell_index}: fitness={c.fitness:.3f}, visits={c.visits}" for c in low_fitness_cells[:10])}

Which behavioral regions should we explore to improve the schematic?
Recommend {num_cells} cells to focus on and explain why.

Output as JSON:
{{"cells": [{{"index": [0,1,2,3,4], "reason": "..."}}], "strategy": "..."}}
"""
        
        response = await self._call_llm(
            prompt,
            system="You are an expert in electronics design optimization. Help guide MAP-Elites exploration."
        )
        
        # Parse response
        try:
            # Try to extract JSON from response
            if "```json" in response:
                json_str = response.split("```json")[1].split("```")[0]
            elif "{" in response:
                start = response.index("{")
                end = response.rindex("}") + 1
                json_str = response[start:end]
            else:
                json_str = "{}"
            
            data = json.loads(json_str)
            recommended_indices = [tuple(c["index"]) for c in data.get("cells", [])]
            
            # Get cells for recommended indices
            cells = []
            for idx in recommended_indices:
                if idx in self.archive.archive:
                    cells.append(self.archive.archive[idx])
                elif low_fitness_cells:
                    # Use low fitness cell as fallback
                    cells.append(low_fitness_cells.pop(0))
            
            return cells[:num_cells]
            
        except Exception as e:
            logger.debug(f"Failed to parse LLM response: {e}")
            # Fall back to random selection from low fitness cells
            return low_fitness_cells[:num_cells]
    
    async def guide_mutation(
        self,
        solution: SchematicSolution,
        validation_issues: List[str],
    ) -> MutationGuidance:
        """
        LLM recommends mutation based on current issues.
        
        Analyzes the solution's problems and suggests specific
        mutations to fix them.
        """
        state = solution.state
        
        # Build state summary
        components_summary = []
        for comp in state.components[:20]:
            components_summary.append(f"  - {comp.reference}: {comp.part_number} ({comp.category})")
        
        connections_summary = []
        for conn in state.connections[:20]:
            connections_summary.append(
                f"  - {conn.from_ref}.{conn.from_pin} -> {conn.to_ref}.{conn.to_pin}"
            )
        
        prompt = f"""
Analyze this schematic and recommend a mutation to improve it:

Current Fitness: {solution.fitness:.3f}

Components ({len(state.components)} total):
{chr(10).join(components_summary)}

Connections ({len(state.connections)} total):
{chr(10).join(connections_summary)}

Wiring Completeness: {state.wiring_completeness:.1%}

Validation Issues:
{chr(10).join(f"- {issue}" for issue in validation_issues[:10])}

Recommend ONE specific mutation to fix the most critical issue.

Output as JSON:
{{
  "mutation_type": "connection_fix|component_swap|topology_change|wire_route",
  "target_component": "U1" or null,
  "target_connection": ["from_ref", "from_pin", "to_ref", "to_pin"] or null,
  "description": "What to change",
  "confidence": 0.0-1.0,
  "reasoning": "Why this mutation will help"
}}
"""
        
        response = await self._call_llm(
            prompt,
            system="You are an expert electronics engineer. Suggest precise mutations to fix schematic issues."
        )
        
        # Parse response
        try:
            if "```json" in response:
                json_str = response.split("```json")[1].split("```")[0]
            elif "{" in response:
                start = response.index("{")
                end = response.rindex("}") + 1
                json_str = response[start:end]
            else:
                json_str = "{}"
            
            data = json.loads(json_str)
            
            target_conn = data.get("target_connection")
            if target_conn and len(target_conn) == 4:
                target_conn = tuple(target_conn)
            else:
                target_conn = None
            
            return MutationGuidance(
                mutation_type=data.get("mutation_type", "connection_fix"),
                target_component=data.get("target_component"),
                target_connection=target_conn,
                description=data.get("description", ""),
                confidence=data.get("confidence", 0.5),
                reasoning=data.get("reasoning", ""),
            )
            
        except Exception as e:
            logger.debug(f"Failed to parse mutation guidance: {e}")
            # Return default guidance
            return MutationGuidance(
                mutation_type="connection_fix",
                description="Fix missing connections",
                confidence=0.3,
            )
    
    def compute_behavior_descriptor(self, state: SchematicState) -> np.ndarray:
        """
        Compute 10D behavioral descriptor for a schematic state.
        
        Dimensions:
        1. Component count (normalized)
        2. Connection count (normalized)
        3. Wire count (normalized)
        4. Wiring completeness
        5. Power component ratio
        6. MCU/IC ratio
        7. Passive component ratio
        8. Symbol quality score
        9. Connection density
        10. Average wire length (normalized)
        """
        descriptor = np.zeros(10, dtype=np.float32)
        
        # Normalization constants
        MAX_COMPONENTS = 100
        MAX_CONNECTIONS = 200
        MAX_WIRES = 300
        MAX_WIRE_LENGTH = 500  # mm
        
        # 1. Component count
        descriptor[0] = min(len(state.components) / MAX_COMPONENTS, 1.0)
        
        # 2. Connection count
        descriptor[1] = min(len(state.connections) / MAX_CONNECTIONS, 1.0)
        
        # 3. Wire count
        descriptor[2] = min(len(state.wires) / MAX_WIRES, 1.0)
        
        # 4. Wiring completeness
        descriptor[3] = state.wiring_completeness
        
        # Category counts
        total = len(state.components) or 1
        power_count = sum(1 for c in state.components if c.category in ("Power", "Regulator"))
        mcu_ic_count = sum(1 for c in state.components if c.category in ("MCU", "IC", "Gate_Driver"))
        passive_count = sum(1 for c in state.components if c.category in ("Resistor", "Capacitor", "Inductor"))
        
        # 5. Power component ratio
        descriptor[4] = power_count / total
        
        # 6. MCU/IC ratio
        descriptor[5] = mcu_ic_count / total
        
        # 7. Passive component ratio
        descriptor[6] = passive_count / total
        
        # 8. Symbol quality score
        from ..core.schematic_state import SymbolQuality
        verified = sum(1 for c in state.components if c.quality == SymbolQuality.VERIFIED)
        fetched = sum(1 for c in state.components if c.quality == SymbolQuality.FETCHED)
        descriptor[7] = (verified + fetched * 0.8) / total
        
        # 9. Connection density (connections per component)
        descriptor[8] = min(len(state.connections) / (total * 2), 1.0)
        
        # 10. Average wire length
        if state.wires:
            avg_length = state.total_wire_length / len(state.wires)
            descriptor[9] = min(avg_length / (MAX_WIRE_LENGTH / len(state.wires)), 1.0)
        
        return descriptor
    
    def add_solution(self, solution: SchematicSolution) -> bool:
        """
        Add solution to archive with computed behavior descriptor.
        """
        if solution.behavior_descriptor is None:
            solution.behavior_descriptor = self.compute_behavior_descriptor(solution.state)
        return self.archive.add(solution)
