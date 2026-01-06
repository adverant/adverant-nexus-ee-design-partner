# Nexus EE Design Partner

> **End-to-end hardware/software development automation platform**

[![TypeScript](https://img.shields.io/badge/TypeScript-5.3-blue.svg)](https://www.typescriptlang.org/)
[![Node.js](https://img.shields.io/badge/Node.js-20%2B-green.svg)](https://nodejs.org/)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

## Overview

EE Design Partner is a revolutionary **Claude Code-driven** electronic design automation (EDA) platform that integrates:

- **Terminal-first UX** with Claude Code as the central orchestrator
- **Skills Engine** for dynamic workflow automation
- **Multi-LLM validation** (Claude Opus 4 + Gemini 2.5 Pro via OpenRouter)
- **Full EDA toolchain** embedded via KiCad WASM
- **Comprehensive simulation suite** (SPICE, Thermal, SI, RF, EMC)
- **GitHub-centric** project management with VSCode-like file browser

## 10-Phase Pipeline

| Phase | Name | Description |
|-------|------|-------------|
| 1 | **Ideation & Research** | Requirements gathering, patent search, research papers |
| 2 | **Architecture** | System design, component selection, BOM optimization |
| 3 | **Schematic Capture** | AI-generated schematics, KiCad WASM editing |
| 4 | **Simulation** | SPICE, Thermal, Signal Integrity, RF/EMC |
| 5 | **PCB Layout** | Multi-agent tournament, Ralph Loop optimization |
| 6 | **Manufacturing** | Gerber export, DFM check, vendor quotes |
| 7 | **Firmware** | HAL generation, driver scaffolding, RTOS config |
| 8 | **Testing** | Test generation, HIL setup, coverage analysis |
| 9 | **Production** | Assembly guides, quality checks, traceability |
| 10 | **Field Support** | Debug assistance, service manuals, RMA |

## Quick Start

### Prerequisites

- Node.js 20+
- Python 3.11+ (for KiCad automation)
- Docker (for simulation containers)

### Installation

```bash
# Clone repository
git clone https://github.com/adverant/adverant-nexus-ee-design-partner.git
cd adverant-nexus-ee-design-partner

# Install dependencies
npm install

# Configure environment
cp .env.example .env
# Edit .env with your API keys

# Start development server
npm run dev
```

### Configuration

Create `.env` in the project root:

```env
# Required API Keys
NEXUS_API_KEY=your_nexus_key
ANTHROPIC_API_KEY=your_anthropic_key
OPENROUTER_API_KEY=your_openrouter_key

# Service URLs (use defaults for local dev)
NEXUS_GRAPHRAG_URL=http://localhost:9000
NEXUS_MAGEAGENT_URL=http://localhost:9010
```

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    NEXUS DASHBOARD - EE DESIGN PARTNER                       │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │              CLAUDE CODE TERMINAL (Central Element)                   │  │
│  │  $ /ee-design analyze-requirements "200A FOC ESC"                    │  │
│  │  $ /schematic-gen power-stage --mosfets=18                           │  │
│  │  $ /pcb-layout generate --strategy=thermal --layers=10               │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
│                                                                              │
│  ┌─────────────────────┐  ┌─────────────────────┐  ┌────────────────────┐  │
│  │   PROJECT REPO      │  │   DESIGN TABS       │  │   VALIDATION       │  │
│  │   (VSCode-like)     │  │   (KiCad WASM)      │  │   PANEL            │  │
│  └─────────────────────┘  └─────────────────────┘  └────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Key Features

### Multi-Agent PCB Layout Tournament

Five specialized AI agents compete to produce optimal layouts:

| Agent | Strategy | Best For |
|-------|----------|----------|
| **Conservative** | Reliability first | High-power, industrial |
| **Aggressive Compact** | Size optimization | Consumer electronics |
| **Thermal** | Heat management | Power electronics |
| **EMI** | Signal integrity | High-speed digital, RF |
| **DFM** | Manufacturing yield | High-volume production |

### Simulation Suite

| Type | Engine | Capabilities |
|------|--------|--------------|
| **SPICE** | ngspice/LTspice | DC, AC, Transient, Monte Carlo |
| **Thermal** | OpenFOAM/Elmer | FEA, CFD, Transient |
| **Signal Integrity** | scikit-rf | Impedance, Crosstalk, Eye diagram |
| **RF/EMC** | openEMS | Field patterns, S-parameters |

### Multi-LLM Validation

```
Design Artifact
     │
     ├──► Claude Opus 4 (Primary generation)
     ├──► Gemini 2.5 Pro (Cross-validation)
     └──► Domain Expert Validators
              │
              ▼
        Consensus Engine → Final Score
```

### Firmware Generation

Supported MCU families:
- **STM32** (all families) - FreeRTOS, Zephyr
- **ESP32** - ESP-IDF, FreeRTOS
- **TI TMS320** - TI-RTOS
- **Infineon AURIX** - AUTOSAR
- **Nordic nRF** - Zephyr
- **NXP i.MX RT** - FreeRTOS, Zephyr

## API Reference

### Projects

```bash
# Create project
POST /api/v1/projects
{
  "name": "FOC ESC Heavy-Lift",
  "description": "200A triple-redundant motor controller"
}

# Generate schematic
POST /api/v1/projects/:id/schematic/generate
{
  "architecture": {...},
  "components": [...]
}

# Start PCB layout
POST /api/v1/projects/:id/pcb-layout/generate
{
  "schematicId": "uuid",
  "boardConstraints": {
    "width": 100,
    "height": 80,
    "layers": 10
  },
  "agents": ["thermal_optimized", "emi_optimized"]
}
```

### Simulations

```bash
# SPICE simulation
POST /api/v1/projects/:id/simulation/spice
{
  "schematicId": "uuid",
  "analysisType": "transient",
  "parameters": {...}
}

# Thermal simulation
POST /api/v1/projects/:id/simulation/thermal
{
  "pcbLayoutId": "uuid",
  "analysisType": "steady_state"
}
```

## Development

```bash
# Build
npm run build

# Test
npm test

# Type check
npm run typecheck

# Lint
npm run lint
```

## Deployment

**⚠️ Docker builds must be done on the remote server!**

```bash
# Use the deploy skill
/build-deploy
```

## Reference Implementation

The **foc-esc-heavy-lift** project serves as reference:
- 10-layer PCB, 164 components
- 200A continuous, 400A peak current
- Triple-redundant MCUs (AURIX, TMS320, STM32H755)
- 18× SiC MOSFETs with thermal management
- 15,000+ lines of firmware

## Success Metrics

| Metric | Target |
|--------|--------|
| Time to First PCB | < 2 hours |
| DRC/ERC First-pass | 95%+ |
| Simulation Coverage | All 8 types |
| Firmware Auto-gen | 80%+ |
| Manufacturing Yield | 98%+ |

## License

MIT - Adverant Inc. 2024-2026

## Support

- Documentation: `/docs`
- Issues: [GitHub Issues](https://github.com/adverant/adverant-nexus-ee-design-partner/issues)
- Slack: `#ee-design-partner`