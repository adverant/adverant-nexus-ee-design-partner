# EE Design Partner - Claude Code Instructions

## Project Overview

**EE Design Partner** is an end-to-end hardware/software development automation platform that integrates:
- **Claude Code** as the central orchestrator (terminal-first UX)
- **Gemini 2.5 Pro** via OpenRouter for multi-LLM validation
- **KiCad WASM** for browser-native schematic and PCB editing
- **Comprehensive simulation suite** (SPICE, Thermal, SI, RF, EMC)
- **Skills Engine** for dynamic workflow automation
- **Ralph Loop** for multi-agent PCB optimization

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    NEXUS DASHBOARD - EE DESIGN PARTNER                       │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │              CLAUDE CODE TERMINAL (Central Element)                   │  │
│  │  $ /ee-design analyze-requirements "200A FOC ESC"                    │  │
│  │  $ /schematic-gen power-stage --mosfets=18 --topology=3phase         │  │
│  │  $ /pcb-layout generate --strategy=thermal --layers=10               │  │
│  │  $ /simulate all --gemini-validate                                   │  │
│  │  $ /firmware-gen stm32h755 --foc --triple-redundant                  │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
│                                                                              │
│  ┌─────────────────────┐  ┌─────────────────────┐  ┌────────────────────┐  │
│  │   PROJECT REPO      │  │   DESIGN TABS       │  │   VALIDATION       │  │
│  │   (VSCode-like)     │  │   (KiCad WASM)      │  │   PANEL            │  │
│  └─────────────────────┘  └─────────────────────┘  └────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────────┘
```

## 10-Phase Pipeline

| Phase | Description | Key Skills |
|-------|-------------|------------|
| 1 | **Ideation & Research** | `/research-paper`, `/patent-search`, `/requirements-gen` |
| 2 | **Architecture** | `/ee-architecture`, `/component-select`, `/bom-optimize` |
| 3 | **Schematic Capture** | `/schematic-gen`, `/schematic-review`, `/netlist-gen` |
| 4 | **Simulation** | `/simulate-spice`, `/simulate-thermal`, `/simulate-si`, `/simulate-rf` |
| 5 | **PCB Layout** | `/pcb-layout`, `/pcb-review`, `/stackup-design` |
| 6 | **Manufacturing** | `/gerber-gen`, `/dfm-check`, `/vendor-quote` |
| 7 | **Firmware** | `/firmware-gen`, `/hal-gen`, `/driver-gen` |
| 8 | **Testing** | `/test-gen`, `/hil-setup`, `/test-procedure` |
| 9 | **Production** | `/manufacture`, `/assembly-guide`, `/quality-check` |
| 10 | **Field Support** | `/debug-assist`, `/service-manual`, `/firmware-update` |

## Development Commands

```bash
# Install dependencies
npm install

# Start development server
npm run dev

# Build for production
npm run build

# Run tests
npm test

# Type checking
npm run typecheck

# Lint code
npm run lint
```

## Docker & Deployment

**🚫 CRITICAL: NEVER BUILD DOCKER IMAGES LOCALLY!**

All Docker builds MUST happen on the remote server (157.173.102.118).

```bash
# ❌ WRONG - Never run locally
docker build .

# ✅ CORRECT - Use the deploy skill
/build-deploy
```

## Project Structure

```
adverant-nexus-ee-design-partner/
├── services/nexus-ee-design/    # Main service
│   ├── src/
│   │   ├── api/                 # REST API routes
│   │   ├── services/            # Business logic
│   │   │   ├── schematic/       # Schematic generation
│   │   │   ├── pcb/             # PCB layout (Ralph Loop)
│   │   │   ├── simulation/      # All simulation types
│   │   │   ├── firmware/        # Code generation
│   │   │   ├── manufacturing/   # Vendor integration
│   │   │   └── validation/      # Multi-LLM validation
│   │   ├── types/               # TypeScript types
│   │   └── utils/               # Logger, errors, helpers
│   └── python-scripts/          # KiCad automation
├── ui/                          # Dashboard UI components
├── skills/                      # Skill definitions (SKILL.md)
├── k8s/                         # Kubernetes manifests
├── docs/                        # Documentation
└── tests/                       # Test suites
```

## Key Services

### PCB Layout Generation
- **5 Competing Agents**: Conservative, Aggressive Compact, Thermal, EMI, DFM
- **Ralph Loop**: 100-iteration tournament with convergence detection
- **8 Validation Domains**: DRC, ERC, IPC-2221, SI, Thermal, DFM, Best Practices, Testing

### Simulation Suite
| Type | Engine | Analysis |
|------|--------|----------|
| SPICE | ngspice/LTspice | DC, AC, Transient, Noise, Monte Carlo |
| Thermal | OpenFOAM/Elmer | Steady-state, Transient, CFD |
| Signal Integrity | scikit-rf | Impedance, Crosstalk, Eye diagram |
| RF/EMC | openEMS/MEEP | Field patterns, S11, Emissions |

### Multi-LLM Validation
```
Design Artifact → Claude Opus 4 (Primary)
                → Gemini 2.5 Pro (Validator)
                → Domain Expert Validators
                → Consensus Engine → Final Score
```

### Firmware Generation
Supported MCU families:
- STM32 (all families) - FreeRTOS, Zephyr
- ESP32/ESP32-S3/C3 - ESP-IDF, FreeRTOS
- TI TMS320 - TI-RTOS
- Infineon AURIX - AUTOSAR
- Nordic nRF - Zephyr
- Raspberry Pi Pico - FreeRTOS
- NXP i.MX RT - FreeRTOS, Zephyr

## Environment Variables

```bash
# Required
NEXUS_API_KEY=your_key
ANTHROPIC_API_KEY=your_key
OPENROUTER_API_KEY=your_key

# Service URLs (K8s internal)
NEXUS_GRAPHRAG_URL=http://nexus-graphrag.nexus.svc.cluster.local:9000
NEXUS_MAGEAGENT_URL=http://nexus-mageagent.nexus.svc.cluster.local:9010

# Simulation Tools
NGSPICE_PATH=/usr/bin/ngspice
OPENFOAM_PATH=/opt/openfoam
OPENEMS_PATH=/opt/openems
```

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/v1/projects` | Create new project |
| POST | `/api/v1/projects/:id/schematic/generate` | Generate schematic |
| POST | `/api/v1/projects/:id/pcb-layout/generate` | Start PCB layout |
| POST | `/api/v1/projects/:id/simulation/spice` | Run SPICE simulation |
| POST | `/api/v1/projects/:id/simulation/thermal` | Run thermal analysis |
| POST | `/api/v1/projects/:id/firmware/generate` | Generate firmware |
| POST | `/api/v1/projects/:id/manufacturing/gerbers` | Export Gerbers |
| POST | `/api/v1/projects/:id/validate/multi-llm` | Multi-LLM validation |

## WebSocket Events

| Event | Direction | Description |
|-------|-----------|-------------|
| `simulation:started` | Server→Client | Simulation job started |
| `simulation:progress` | Server→Client | Progress update |
| `simulation:completed` | Server→Client | Results available |
| `layout:started` | Server→Client | Layout generation started |
| `layout:iteration` | Server→Client | Ralph Loop iteration |
| `layout:completed` | Server→Client | Final layout ready |
| `validation:started` | Server→Client | Multi-LLM validation started |
| `validation:result` | Server→Client | Validator result |

## Reference Project

The **foc-esc-heavy-lift** project serves as the reference implementation:
- 10-layer PCB, 164 components
- 200A continuous, 400A peak
- Triple-redundant MCUs (AURIX, TMS320, STM32H755)
- 18× SiC MOSFETs
- 15,000+ lines firmware

## Memory Integration

This project uses Nexus GraphRAG for persistent memory:

```bash
# Store important context
echo '{"content": "Design decision: chose thermal via array for MOSFET cooling", "event_type": "decision"}' | ~/.claude/hooks/store-memory.sh

# Recall relevant memories
echo '{"query": "PCB thermal management patterns"}' | ~/.claude/hooks/recall-memory.sh
```

## Contributing

1. Create feature branch: `git checkout -b feature/my-feature`
2. Make changes, add tests
3. Run full test suite: `npm test && npm run typecheck`
4. Commit with conventional commits: `feat: add new validator`
5. Push and create PR

## License

MIT - Adverant Inc. 2024-2026