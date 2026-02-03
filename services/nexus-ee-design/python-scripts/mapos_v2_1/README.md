# MAPO v2.1 - LLM-Orchestrated Gaming AI for PCB Optimization

**Core Philosophy: "Opus 4.5 Thinks, Gaming AI Explores, Algorithms Execute"**

MAPO v2.1 merges the LLM orchestration layer from v2.0 with the Gaming AI algorithms from v1.0, creating a synergistic system where:

- **LLM agents** provide strategic reasoning and semantic understanding
- **Gaming AI** (MAP-Elites, Red Queen) provides quality-diversity exploration
- **Classical algorithms** (PathFinder, CBS, DP) execute routing decisions

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                    OPUS 4.5 THINKING LAYER                           │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐    │
│  │ Routing     │ │ Congestion  │ │ Conflict    │ │ SI          │    │
│  │ Strategist  │ │ Predictor   │ │ Resolver    │ │ Advisor     │    │
│  └──────┬──────┘ └──────┬──────┘ └──────┬──────┘ └──────┬──────┘    │
│         │               │               │               │            │
│         └───────────────┴───────┬───────┴───────────────┘            │
│                                 │                                    │
│                    ┌────────────┴────────────┐                       │
│                    │ Debate & Critique       │                       │
│                    │ Coordinator             │                       │
│                    └────────────┬────────────┘                       │
└──────────────────────────────────┼───────────────────────────────────┘
                                   │
┌──────────────────────────────────┼───────────────────────────────────┐
│             LLM-GAMING INTEGRATION LAYER (v2.1 NEW)                  │
│  ┌───────────────────────┐  ┌───────────────────────┐               │
│  │ LLM-Guided MAP-Elites │  │ LLM-Guided Red Queen  │               │
│  │ • LLM cell selection  │  │ • LLM mutation guide  │               │
│  │ • LLM mutation hints  │  │ • Semantic generality │               │
│  │ • Debate validation   │  │ • Champion analysis   │               │
│  └───────────┬───────────┘  └───────────┬───────────┘               │
└──────────────┼───────────────────────────┼──────────────────────────┘
               │                           │
┌──────────────┼───────────────────────────┼──────────────────────────┐
│              │    GAMING AI LAYER        │                          │
│  ┌───────────┴───────────┐  ┌───────────┴───────────┐              │
│  │ MAP-Elites Archive    │  │ Red Queen Evolver     │              │
│  │ • 10D behavioral desc │  │ • Champion pool       │              │
│  │ • Quality-diversity   │  │ • Adversarial evolve  │              │
│  │ • Stepping stones     │  │ • Generality scoring  │              │
│  └───────────────────────┘  └───────────────────────┘              │
│                                                                     │
│  ┌───────────────────────┐  ┌───────────────────────┐              │
│  │ Elo Tournament Judge  │  │ Ralph Wiggum Loop     │              │
│  │ • Multi-criteria rank │  │ • Self-referential    │              │
│  │ • Fair comparison     │  │ • Iterative refine    │              │
│  └───────────────────────┘  └───────────────────────┘              │
└─────────────────────────────────────────────────────────────────────┘
                                   │
┌──────────────────────────────────┼───────────────────────────────────┐
│                 ALGORITHM EXECUTION LAYER                            │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────────────┐            │
│  │ PathFinder  │ │ CBS Router  │ │ Layer Assignment    │            │
│  │ Negotiation │ │ (fallback)  │ │ (DP + LLM hints)    │            │
│  └─────────────┘ └─────────────┘ └─────────────────────┘            │
└─────────────────────────────────────────────────────────────────────┘
```

## Key Components

### LLM Agents (from v2.0)

| Agent | Purpose |
|-------|---------|
| **RoutingStrategistAgent** | Pre-routing strategy, net ordering |
| **CongestionPredictorAgent** | Congestion zone prediction |
| **ConflictResolverAgent** | Routing conflict resolution |
| **SignalIntegrityAdvisorAgent** | SI guidance with IPC-2141 formulas |
| **LayerAssignmentStrategistAgent** | Layer assignment hints |

### Gaming AI Components (from v1.0)

| Component | Purpose |
|-----------|---------|
| **MAP-Elites Archive** | Quality-diversity, 10D behavioral descriptors |
| **Red Queen Evolver** | Adversarial co-evolution against champions |
| **Elo Tournament** | Fair multi-criteria solution ranking |
| **Ralph Wiggum Loop** | Self-referential optimization |

### LLM-Gaming Integration (v2.1 NEW)

| Component | Purpose |
|-----------|---------|
| **LLMGuidedMAPElites** | LLM guides cell selection & mutation |
| **LLMGuidedRedQueen** | LLM analyzes champion weaknesses |
| **EnhancedBehavioralDescriptor** | 10D + semantic features |
| **IntegratedGamingLLMOptimizer** | Full pipeline orchestration |

## Usage

```python
from mapos_v2_1 import create_integrated_optimizer

# Create the integrated optimizer
optimizer = create_integrated_optimizer(
    openrouter_api_key="your-key",  # Or set OPENROUTER_API_KEY env var
    enable_debate=True
)

# Run optimization
best_solution, metrics = await optimizer.optimize(
    initial_solution=my_pcb_state,
    max_iterations=100
)

print(f"Final fitness: {metrics['final_fitness']}")
print(f"Archive coverage: {metrics['archive_coverage']:.1%}")
```

## Enhanced Behavioral Descriptors (18D)

v2.1 enhances the 10D behavioral descriptors with LLM-extracted semantic features:

### Base Descriptor (10D)
1. `routing_density` - Total trace length / board area
2. `via_count` - Number of vias
3. `layer_utilization` - Fraction of layers used
4. `zone_coverage` - Power/ground zone coverage
5. `thermal_spread` - Heat distribution variance
6. `signal_length_variance` - Variance in critical signal lengths
7. `component_clustering` - How clustered components are
8. `power_path_directness` - Efficiency of power distribution
9. `min_clearance_ratio` - Actual / required clearance
10. `silk_density` - Silkscreen area density

### Semantic Features (8D, LLM-extracted)
11. `routing_strategy` - "dense", "sparse", "mixed"
12. `thermal_strategy` - "distributed", "concentrated", "passive"
13. `power_distribution` - "star", "tree", "grid"
14. `signal_integrity_focus` - "impedance", "length", "shielding"
15. `manufacturing_complexity` - "low", "medium", "high"
16. `similarity_to_reference` - LLM-computed similarity
17. `novelty_score` - LLM-computed novelty
18. `robustness_estimate` - LLM-estimated robustness

## Version Comparison

| Feature | v1.0 | v2.0 | v2.1 |
|---------|------|------|------|
| MAP-Elites QD | ✅ | ❌ | ✅ |
| Red Queen Evolution | ✅ | ❌ | ✅ |
| Elo Tournament | ✅ | ❌ | ✅ |
| LLM Strategist | ❌ | ✅ | ✅ |
| LLM Conflict Resolver | ❌ | ✅ | ✅ |
| Debate Mechanism | ❌ | ✅ | ✅ |
| PathFinder Router | ❌ | ✅ | ✅ |
| CBS Router | ❌ | ✅ | ✅ |
| Layer Optimizer | ❌ | ✅ | ✅ |
| LLM-Guided Gaming | ❌ | ❌ | ✅ |

## Research Sources

- **CircuitLM**: Multi-agent debate for EDA (arxiv 2601.04505)
- **Digital Red Queen**: Adversarial co-evolution (Sakana AI)
- **MAP-Elites**: Quality-diversity optimization (arxiv 1504.04909)
- **OrthoRoute**: PathFinder negotiation algorithm
- **AlphaFold2**: Iterative refinement with confidence
- **IPC-2141**: Signal integrity impedance formulas

## Directory Structure

```
mapos_v2_1/
├── __init__.py
├── README.md
├── agents/                      # LLM agents (from v2.0)
│   ├── routing_strategist.py
│   ├── congestion_predictor.py
│   ├── conflict_resolver.py
│   ├── si_advisor.py
│   └── layer_strategist.py
├── gaming_ai/                   # Gaming AI (from v1.0)
│   ├── map_elites.py
│   ├── red_queen_evolver.py
│   ├── ralph_wiggum_optimizer.py
│   └── ...
├── llm_gaming_integration.py    # NEW: LLM-Gaming bridge
├── debate_coordinator.py
├── pathfinder_router.py
├── cbs_router.py
├── layer_optimizer.py
├── multi_agent_optimizer.py
└── tournament_judge.py
```

## Requirements

- Python 3.10+
- numpy
- httpx (for OpenRouter API)
- OPENROUTER_API_KEY environment variable (for LLM features)
