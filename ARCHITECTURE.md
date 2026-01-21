# EE Design Partner - Architecture Overview

System architecture documentation for the EE Design Partner plugin.

---

## System Overview

EE Design Partner is a Claude Code-driven electronic design automation (EDA) platform. The architecture centers on a terminal-first user experience with Claude Code as the central orchestrator.

```mermaid
graph TB
    subgraph "User Interface Layer"
        Terminal[Claude Code Terminal]
        ProjectBrowser[Project Browser]
        KiCadViewer[KiCad WASM Viewer]
        SimPanel[Simulation Panel]
        ValPanel[Validation Panel]
        Viewer3D[3D Viewer]
    end

    subgraph "Orchestration Layer"
        MCP[MCP Tools Layer]
        Skills[40+ Skills Engine]
        WebSocket[WebSocket Events]
    end

    subgraph "Processing Layer"
        SchematicGen[Schematic Generation]
        PCBLayout[PCB Layout Engine]
        SimEngine[Simulation Suite]
        FirmwareGen[Firmware Generator]
        GamingAI[Gaming AI Engine]
    end

    subgraph "Integration Layer"
        GraphRAG[Nexus GraphRAG]
        MageAgent[Nexus MageAgent]
        Sandbox[Nexus Sandbox]
    end

    subgraph "External Services"
        Anthropic[Anthropic Claude API]
        OpenRouter[OpenRouter API]
    end

    Terminal --> MCP
    ProjectBrowser --> MCP
    KiCadViewer --> MCP

    MCP --> Skills
    Skills --> SchematicGen
    Skills --> PCBLayout
    Skills --> SimEngine
    Skills --> FirmwareGen
    Skills --> GamingAI

    WebSocket --> SimPanel
    WebSocket --> ValPanel

    SchematicGen --> GraphRAG
    PCBLayout --> MageAgent
    SimEngine --> Sandbox

    GamingAI --> Anthropic
    GamingAI --> OpenRouter

    PCBLayout --> GamingAI
    SchematicGen --> GamingAI
```

---

## 10-Phase Pipeline

The EE Design Partner implements a comprehensive 10-phase pipeline for hardware development:

```mermaid
graph LR
    subgraph "Phase 1-3: Design"
        P1[1. Ideation & Research]
        P2[2. Architecture]
        P3[3. Schematic Capture]
    end

    subgraph "Phase 4-6: Implementation"
        P4[4. Simulation]
        P5[5. PCB Layout]
        P6[6. Manufacturing]
    end

    subgraph "Phase 7-10: Delivery"
        P7[7. Firmware]
        P8[8. Testing]
        P9[9. Production]
        P10[10. Field Support]
    end

    P1 --> P2 --> P3 --> P4 --> P5 --> P6 --> P7 --> P8 --> P9 --> P10
```

### Phase Details

| Phase | Name | Description | Associated Skills |
|-------|------|-------------|-------------------|
| 1 | Ideation & Research | Requirements gathering, patent search | `/research-paper`, `/patent-search` |
| 2 | Architecture | System design, component selection | `/ee-architecture`, `/component-select` |
| 3 | Schematic Capture | AI-generated schematics with Gaming AI | `/schematic-gen`, `/schematic-review` |
| 4 | Simulation | SPICE, Thermal, SI, RF/EMC | `/simulate-spice`, `/simulate-thermal`, `/simulate-si`, `/simulate-rf`, `/simulate-emc` |
| 5 | PCB Layout | Multi-agent tournament with Gaming AI | `/pcb-layout`, `/mapos` |
| 6 | Manufacturing | Gerber export, DFM, vendor quotes | `/gerber-gen`, `/dfm-check` |
| 7 | Firmware | HAL generation, RTOS config | `/firmware-gen`, `/hal-gen` |
| 8 | Testing | Test generation, HIL setup | `/test-gen`, `/hil-setup` |
| 9 | Production | Assembly guides, quality checks | `/manufacture`, `/quality-check` |
| 10 | Field Support | Debug assistance, service manuals | `/debug-assist`, `/service-manual` |

---

## Gaming AI Engine

The Gaming AI Engine brings techniques from game-playing AI to electronic design optimization:

```mermaid
graph TB
    subgraph "Gaming AI Engine"
        subgraph "MAP-Elites Archive"
            ME[Quality-Diversity Optimization]
            Archive[Elite Solutions Archive]
        end

        subgraph "Red Queen Evolver"
            RQ[Adversarial Co-Evolution]
            Compete[Competing Solutions]
        end

        subgraph "Ralph Wiggum Optimizer"
            RW[Persistent Iteration]
            Loop[Iterative Refinement]
        end
    end

    Input[Design Input] --> ME
    ME <--> RQ
    RQ <--> RW
    ME --> Archive
    Archive --> Output[Optimized Design]

    LLM[OpenRouter LLM] --> ME
    LLM --> RQ
    LLM --> RW
```

### Algorithm Components

| Algorithm | Inspiration | Purpose | Key Feature |
|-----------|-------------|---------|-------------|
| **MAP-Elites** | Quality-Diversity | Maintain diverse elite solutions | Preserves high-quality solutions across behavior dimensions |
| **Red Queen** | Digital Red Queen | Adversarial co-evolution | Solutions compete and evolve against each other |
| **Ralph Wiggum** | Persistent iteration | Continuous optimization | File-based state with git integration |

### Gaming AI Features

- **LLM-First Mode**: OpenRouter Claude as primary intelligence (PyTorch optional)
- **Optional GPU Offloading**: RunPod, Modal, Replicate, Together AI
- **Multi-Objective Fitness**: 8-domain PCB scoring, 4-domain schematic scoring
- **Quality-Diversity Archive**: Preserves diverse high-quality solutions
- **Persistent Optimization**: File-based state, git integration

---

## MAPOS Pipeline

Multi-Agent PCB Optimization System (MAPOS) implements a 7-phase pipeline for DRC violation reduction:

```mermaid
graph TD
    subgraph "MAPOS 7-Phase Pipeline"
        Phase1[Phase 1: Design Rules<br/>IPC-2221 compliant .kicad_dru]
        Phase2[Phase 2: pcbnew Fixes<br/>Zone nets, dangling vias]
        Phase3[Phase 3: Zone Fill<br/>ZONE_FILLER API]
        Phase4[Phase 4: Net Assignment<br/>Orphan pad nets]
        Phase5[Phase 5: Solder Mask<br/>Via tenting, bridges]
        Phase6[Phase 6: Silkscreen<br/>Graphics to Fab layer]
        Phase7[Phase 7: Gaming AI<br/>MAP-Elites + Red Queen + Ralph Wiggum]
    end

    Input[PCB Design] --> Phase1
    Phase1 --> Phase2
    Phase2 --> Phase3
    Phase3 --> Phase4
    Phase4 --> Phase5
    Phase5 --> Phase6
    Phase6 --> Phase7
    Phase7 --> Output[Optimized PCB]
```

---

## UI Components

The plugin provides 6 UI components for interactive design:

```mermaid
graph TB
    subgraph "UI Layer"
        Terminal[Terminal<br/>Claude Code integration<br/>Type: iframe]
        Browser[Project Browser<br/>VSCode-like repo browser<br/>Type: panel]
        KiCad[KiCad Viewer<br/>WASM schematic/PCB editor<br/>Type: tab]
        SimPanel[Simulation Panel<br/>Waveforms and results<br/>Type: panel]
        ValPanel[Validation Panel<br/>DRC/ERC/Multi-LLM results<br/>Type: panel]
        Viewer3D[3D Viewer<br/>Three.js PCB visualization<br/>Type: tab]
    end

    Terminal --> API[Plugin API]
    Browser --> API
    KiCad --> API
    SimPanel --> WebSocket[WebSocket]
    ValPanel --> WebSocket
    Viewer3D --> API
```

### Component Details

| Component | Description | Type |
|-----------|-------------|------|
| `terminal` | Claude Code terminal integration | iframe |
| `project-browser` | VSCode-like GitHub repo browser | panel |
| `kicad-viewer` | KiCad WASM schematic and PCB editor | tab |
| `simulation-panel` | Simulation waveforms and results | panel |
| `validation-panel` | DRC/ERC/Multi-LLM validation results | panel |
| `3d-viewer` | Three.js PCB 3D visualization | tab |

---

## Integration Points

### Nexus Services Integration

```mermaid
graph LR
    subgraph "EE Design Partner"
        Plugin[Plugin Core]
    end

    subgraph "Nexus Platform"
        GraphRAG[GraphRAG<br/>Memory & Knowledge Graph<br/>Required]
        MageAgent[MageAgent<br/>Multi-agent Orchestration<br/>Optional]
        Sandbox[Sandbox<br/>Code Execution<br/>Optional]
    end

    Plugin --> GraphRAG
    Plugin --> MageAgent
    Plugin --> Sandbox
```

### External API Integration

```mermaid
graph LR
    subgraph "EE Design Partner"
        Plugin[Plugin Core]
        GamingAI[Gaming AI Engine]
        Validation[Multi-LLM Validation]
    end

    subgraph "External APIs"
        Anthropic[Anthropic API<br/>Claude Opus 4<br/>Primary LLM]
        OpenRouter[OpenRouter API<br/>Gemini 2.5 Pro<br/>Validation & Gaming AI]
    end

    Plugin --> Anthropic
    GamingAI --> OpenRouter
    Validation --> Anthropic
    Validation --> OpenRouter
```

---

## Multi-LLM Validation Architecture

```mermaid
graph TB
    Artifact[Design Artifact]

    subgraph "Validation Ensemble"
        Claude[Claude Opus 4<br/>Primary Generation<br/>Weight: 0.4]
        Gemini[Gemini 2.5 Pro<br/>Cross-validation<br/>Weight: 0.3]
        Domain[Domain Expert Validators<br/>Weight: 0.3]
    end

    subgraph "Consensus Engine"
        Aggregator[Score Aggregator]
        FinalScore[Final Score]
    end

    Artifact --> Claude
    Artifact --> Gemini
    Artifact --> Domain

    Claude --> Aggregator
    Gemini --> Aggregator
    Domain --> Aggregator

    Aggregator --> FinalScore
```

---

## Simulation Architecture

```mermaid
graph TB
    subgraph "Simulation Suite"
        subgraph "SPICE"
            ngspice[ngspice]
            ltspice[LTspice]
        end

        subgraph "Thermal"
            openfoam[OpenFOAM]
            elmer[Elmer]
        end

        subgraph "Signal Integrity"
            scikitrf[scikit-rf]
        end

        subgraph "RF/EMC"
            openems[openEMS]
            meep[MEEP]
        end
    end

    Orchestrator[Simulation Orchestrator] --> ngspice
    Orchestrator --> ltspice
    Orchestrator --> openfoam
    Orchestrator --> elmer
    Orchestrator --> scikitrf
    Orchestrator --> openems
    Orchestrator --> meep

    Orchestrator --> WebSocket[WebSocket Events]
    WebSocket --> UI[Simulation Panel]
```

### Simulation Capabilities

| Type | Engine | Capabilities |
|------|--------|--------------|
| SPICE | ngspice/LTspice | DC, AC, Transient, Monte Carlo, Noise |
| Thermal | OpenFOAM/Elmer | FEA, CFD, Steady-state, Transient |
| Signal Integrity | scikit-rf | Impedance, Crosstalk, Eye diagram |
| RF/EMC | openEMS/MEEP | Field patterns, S-parameters, Emissions |

---

## Project Structure

```
adverant-nexus-ee-design-partner/
├── services/nexus-ee-design/
│   ├── src/                           # TypeScript backend
│   │   ├── api/                       # REST API routes
│   │   ├── services/                  # Business logic
│   │   │   ├── schematic/             # Schematic generation
│   │   │   ├── pcb/                   # PCB layout, Ralph Loop
│   │   │   ├── simulation/            # Simulation orchestration
│   │   │   └── validation/            # Multi-LLM validation
│   │   └── types/                     # TypeScript definitions
│   └── python-scripts/
│       └── mapos/
│           ├── gaming_ai/             # Gaming AI for PCB
│           │   ├── config.py          # Centralized config
│           │   ├── llm_backends.py    # LLM replacements
│           │   ├── map_elites.py      # Quality-diversity
│           │   ├── red_queen_evolver.py
│           │   └── ralph_wiggum_optimizer.py
│           └── schematic_gaming_ai/   # Gaming AI for Schematics
│               ├── behavior_descriptor.py
│               ├── fitness_function.py
│               ├── mutation_operators.py
│               └── integration.py
├── ui/                                # Next.js dashboard
├── skills/                            # 40+ skill definitions
├── docs/                              # Documentation
│   ├── architecture.md
│   ├── gaming-ai.md                   # Gaming AI deep dive
│   └── use-cases.md                   # Use case examples
└── k8s/                               # Kubernetes manifests
```

---

## Additional Resources

- [QUICKSTART.md](QUICKSTART.md) - Installation and basic setup
- [USE-CASES.md](USE-CASES.md) - Real-world application examples
- [TECHNICAL.md](TECHNICAL.md) - API reference and configuration
