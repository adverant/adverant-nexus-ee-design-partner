# MAPO v2.0 - Enhanced Multi-Agent Prompt Optimization System

Enhanced MAPO with LLM-first architecture using Claude Opus 4.5 via OpenRouter as primary orchestrator.

## Core Philosophy

**"Opus 4.5 Thinks, Algorithms Execute"**

LLM agents reason about routing strategies, predict conflicts, and orchestrate classical CPU-friendly algorithms.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    OPUS 4.5 THINKING LAYER                   │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────────────┐    │
│  │ Strategist  │ │ Predictor   │ │ Conflict Resolver   │    │
│  │   Agent     │ │   Agent     │ │      Agent          │    │
│  └──────┬──────┘ └──────┬──────┘ └──────────┬──────────┘    │
│         │               │                    │               │
│         ▼               ▼                    ▼               │
│  ┌─────────────────────────────────────────────────────┐    │
│  │           DEBATE & CRITIQUE COORDINATOR              │    │
│  └─────────────────────────┬───────────────────────────┘    │
└────────────────────────────┼────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────┐
│                 ALGORITHM EXECUTION LAYER                    │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────────────┐    │
│  │ PathFinder  │ │   Layer     │ │   Manhattan/        │    │
│  │ Negotiation │ │ Assignment  │ │   Topological       │    │
│  └─────────────┘ └─────────────┘ └─────────────────────┘    │
└─────────────────────────────────────────────────────────────┘
```

## Enhanced Pipeline (9 Phases)

```
Phase 0: Load & Analyze
    └─ Initial DRC evaluation

Phase 1: LLM Strategic Planning (NEW)
    ├─ Routing Strategist: Net ordering, method selection
    ├─ Congestion Predictor: Identify problem zones
    └─ SI Advisor: High-speed routing guidance

Phase 2: Pre-DRC Fixes (Enhanced)
    ├─ Deterministic structural corrections
    └─ LLM-guided power net optimization

Phase 3: Negotiation-Based Routing (NEW)
    ├─ PathFinder iterative routing
    ├─ CBS conflict resolution
    └─ LLM Resolver for complex cases

Phase 4: Layer Assignment (NEW for PCB)
    ├─ DP-based layer optimization
    ├─ Via minimization
    └─ LLM layer strategist hints

Phase 5: MCTS Exploration (Enhanced)
    ├─ UCB1 selection
    ├─ LLM-guided expansion with debate
    └─ Congestion-aware evaluation

Phase 6: Evolutionary Optimization
    ├─ GA with tournament selection
    └─ Debate-validated mutations

Phase 7: Tournament Selection
    ├─ Elo-based ranking
    └─ Multi-agent judging

Phase 8: AlphaFold-style Refinement
    ├─ Iterative recycling
    ├─ Convergence detection
    └─ Final LLM validation
```

## New Opus 4.5 Agents

| Agent | Purpose |
|-------|---------|
| **RoutingStrategistAgent** | Pre-routing analysis, net ordering, method selection |
| **CongestionPredictorAgent** | Semantic congestion prediction (no GPU) |
| **ConflictResolverAgent** | Multi-net conflict resolution with trade-off analysis |
| **SignalIntegrityAdvisorAgent** | Impedance calculations, diff pair guidance |
| **LayerAssignmentStrategistAgent** | Layer optimization hints, via minimization |

## CPU-Friendly Algorithms

| Algorithm | Source | Purpose |
|-----------|--------|---------|
| **PathFinder Negotiation** | OrthoRoute (adapted) | Iterative rip-up/reroute |
| **CBS Conflict-Based Search** | MAPF research | Multi-net conflict resolution |
| **DP Layer Assignment** | Academic papers | Via minimization |

## Key Innovation: Debate-and-Critique

From CircuitLM research - multiple agents debate proposals before execution:

1. Proposer Agent suggests modification
2. Critic Agents identify potential issues
3. Refinement round addresses concerns
4. Consensus reached or escalated

## Usage

```python
from mapos_v2.0.multi_agent_optimizer_v2 import MultiAgentOptimizerV2

optimizer = MultiAgentOptimizerV2(pcb_path="board.kicad_pcb")
result = await optimizer.optimize()
```

## Requirements

- **OpenRouter API Key**: `OPENROUTER_API_KEY` environment variable
- **No GPU Required**: All algorithms are CPU-friendly

## Target Metrics

| Metric | v1.0 Baseline | v2.0 Target |
|--------|---------------|-------------|
| DRC Violations | Baseline | -50% |
| Via Count | Baseline | -30% |
| 4-Way Junctions | 0 | 0 |
| Diff Pair Skew | N/A | < 5% |
| Runtime | 30 min | < 45 min |

## Research Sources

- [CircuitLM Multi-Agent Framework](https://arxiv.org/html/2601.04505)
- [Multi-Agent Based Minimal-Layer Via Routing](https://www.sciencedirect.com/science/article/abs/pii/S0167926025001907)
- [OrthoRoute PathFinder](https://bbenchoff.github.io/pages/OrthoRoute.html)
- [Altium Situs Topological Autorouter](https://www.altium.com/documentation/altium-designer/pcb/routing/situs-topological-autorouter)
