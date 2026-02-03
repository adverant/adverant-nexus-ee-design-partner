# MAPO v1.0 - Multi-Agent Prompt Optimization System

Original MAPO implementation for PCB and schematic optimization.

## Architecture

```
Phase 0: Load & Analyze → Initial DRC evaluation
Phase 1: Pre-DRC Fixes → Deterministic structural corrections (40-50% improvement)
Phase 2: MCTS Exploration → Monte Carlo Tree Search with LLM expansion
Phase 3: Evolutionary Optimization → Genetic algorithm with tournament selection
Phase 4: Tournament Selection → Elo-based ranking and candidate filtering
Phase 5: AlphaFold-style Refinement → Iterative recycling for convergence
```

## Key Components

- `multi_agent_optimizer.py` - Main orchestration
- `mcts_optimizer.py` - UCB1-based tree search
- `evolutionary_optimizer.py` - Genetic algorithm
- `tournament_judge.py` - Elo-based ranking
- `refinement_loop.py` - AlphaFold-style iteration
- `generator_agents.py` - Domain-specific LLM agents
- `pcb_state.py` - Immutable state representation

## Agents

1. **SignalIntegrityAgent** - Trace routing, impedance, crosstalk
2. **ThermalPowerAgent** - Heat dissipation, current paths
3. **ManufacturingAgent** - DFM, cost optimization, yield

## Usage

```python
from mapos_v1.0.multi_agent_optimizer import MultiAgentOptimizer

optimizer = MultiAgentOptimizer(pcb_path="board.kicad_pcb")
result = await optimizer.optimize()
```

## Performance Benchmarks

- Pre-DRC fixes: 40-50% violation reduction
- Full pipeline: 30-40% total improvement
- Runtime: 10-30 minutes (configurable)
