---
name: ee-design-partner
displayName: "EE Design Partner"
description: "End-to-end hardware/software development automation platform with Claude Code orchestration, multi-LLM validation, and comprehensive simulation suite"
version: 1.0.0
status: published
visibility: organization

allowed-tools:
  - Read
  - Write
  - Edit
  - Glob
  - Grep
  - Bash
  - WebFetch
  - WebSearch
  - Task

triggers:
  - /ee-design
  - /eda
  - /hardware
  - /electronics

capabilities:
  - name: full_pipeline
    description: Execute complete 10-phase design pipeline
    parameters:
      - name: project_name
        type: string
        required: true
        description: Name of the project
      - name: requirements
        type: string
        required: true
        description: Natural language requirements description

  - name: phase_execute
    description: Execute specific pipeline phase
    parameters:
      - name: phase
        type: string
        required: true
        description: "Phase to execute: ideation, architecture, schematic, simulation, pcb_layout, manufacturing, firmware, testing, production, field_support"
      - name: project_id
        type: string
        required: true
        description: Project identifier

  - name: generate_schematic
    description: Generate schematic from architecture and requirements
    parameters:
      - name: architecture
        type: object
        required: true
        description: System architecture specification
      - name: components
        type: array
        required: true
        description: Component list with specifications

  - name: generate_pcb_layout
    description: Generate PCB layout using multi-agent tournament
    parameters:
      - name: schematic_id
        type: string
        required: true
        description: Schematic identifier
      - name: board_constraints
        type: object
        required: true
        description: Board dimensions, layer count, etc.
      - name: agents
        type: array
        required: false
        description: Agent strategies to use

  - name: run_simulation
    description: Run simulation of specified type
    parameters:
      - name: type
        type: string
        required: true
        description: "Simulation type: spice, thermal, si, rf, emc"
      - name: artifact_id
        type: string
        required: true
        description: Schematic or PCB layout ID

  - name: generate_firmware
    description: Generate firmware scaffolding for target MCU
    parameters:
      - name: mcu_family
        type: string
        required: true
        description: "MCU family: stm32, esp32, ti_tms320, aurix, nrf, rpi_pico, imxrt"
      - name: mcu_part
        type: string
        required: true
        description: Specific MCU part number
      - name: rtos
        type: string
        required: false
        description: "RTOS to use: freertos, zephyr, tirtos, autosar"

  - name: validate_design
    description: Run multi-LLM validation on design artifacts
    parameters:
      - name: artifact_type
        type: string
        required: true
        description: "Type: schematic, pcb, firmware, simulation"
      - name: artifact_id
        type: string
        required: true
        description: Artifact identifier

  - name: export_manufacturing
    description: Export manufacturing files (Gerbers, BOM, pick-and-place)
    parameters:
      - name: layout_id
        type: string
        required: true
        description: PCB layout identifier
      - name: vendor
        type: string
        required: false
        description: "Target vendor: pcbway, jlcpcb, oshpark"

sub-skills:
  # Phase 1: Ideation & Research
  - research-paper
  - patent-search
  - market-analysis
  - requirements-gen

  # Phase 2: Architecture
  - ee-architecture
  - component-select
  - bom-optimize
  - power-budget

  # Phase 3: Schematic
  - schematic-gen
  - schematic-review
  - netlist-gen

  # Phase 4: Simulation
  - simulate-spice
  - simulate-thermal
  - simulate-si
  - simulate-rf
  - simulate-emc
  - simulate-stress
  - simulate-reliability

  # Phase 5: PCB Layout
  - pcb-layout
  - pcb-review
  - stackup-design
  - via-optimize

  # Phase 6: Manufacturing
  - gerber-gen
  - dfm-check
  - vendor-quote
  - panelize

  # Phase 7: Firmware
  - firmware-gen
  - hal-gen
  - driver-gen
  - rtos-config
  - build-setup

  # Phase 8: Testing
  - test-gen
  - hil-setup
  - test-procedure
  - coverage-analysis

  # Phase 9: Production
  - manufacture
  - assembly-guide
  - quality-check
  - traceability

  # Phase 10: Field Support
  - debug-assist
  - service-manual
  - rma-process
  - firmware-update
---

# EE Design Partner

## Overview

EE Design Partner is an end-to-end hardware/software development automation platform that guides you through all 10 phases of electronic product development, from initial concept to field support.

## Usage

### Start a New Project

```
/ee-design new "200A FOC ESC for heavy-lift drone"
```

This will:
1. Analyze requirements and extract specifications
2. Search for prior art and existing solutions
3. Generate initial architecture proposal
4. Create project structure in GitHub repository

### Execute Specific Phase

```
/ee-design phase schematic --project=foc-esc-v2
```

### Generate Schematic

```
/schematic-gen power-stage --mosfets=18 --topology=3phase --current=200A
```

### Generate PCB Layout

```
/pcb-layout generate --schematic=sch-001 --layers=10 --strategy=thermal
```

This starts the Ralph Loop tournament with 5 competing agents, iterating up to 100 times until convergence.

### Run Simulations

```
/simulate spice --schematic=sch-001 --analysis=transient
/simulate thermal --layout=pcb-001 --ambient=25
/simulate si --layout=pcb-001 --nets=PHASE_A,PHASE_B,PHASE_C
/simulate emc --layout=pcb-001 --frequency=1M-1G
```

### Generate Firmware

```
/firmware-gen stm32h755 --foc --triple-redundant --rtos=freertos
```

### Multi-LLM Validation

```
/validate --type=pcb --id=pcb-001 --validators=claude,gemini,expert
```

### Export for Manufacturing

```
/gerber-gen --layout=pcb-001 --format=x2
/vendor-quote --vendor=pcbway --qty=10
```

## Workflow Example

For a complete FOC ESC development:

```bash
# Phase 1: Research & Requirements
/ee-design new "200A triple-redundant FOC ESC for heavy-lift drone"

# Phase 2: Architecture
/ee-architecture generate --project=foc-esc
/component-select mosfet --current=200A --voltage=60V --package=TO-247

# Phase 3: Schematic
/schematic-gen all --project=foc-esc

# Phase 4: Simulation
/simulate all --project=foc-esc

# Phase 5: PCB Layout
/pcb-layout generate --project=foc-esc --strategy=thermal --layers=10

# Phase 6: Manufacturing
/gerber-gen --project=foc-esc
/vendor-quote --project=foc-esc --vendor=pcbway

# Phase 7: Firmware
/firmware-gen --project=foc-esc --mcu=stm32h755 --rtos=freertos

# Phase 8: Testing
/test-gen --project=foc-esc

# Phase 9-10: Production & Support
/assembly-guide --project=foc-esc
/service-manual --project=foc-esc
```

## Multi-Agent Tournament

The PCB layout phase uses 5 competing agent strategies:

| Agent | Priority | Best For |
|-------|----------|----------|
| **Conservative** | Reliability > DFM > Cost | High-power, industrial |
| **Aggressive Compact** | Size > Cost > DFM | Consumer, space-constrained |
| **Thermal Optimized** | Thermal > Reliability > Size | Power electronics |
| **EMI Optimized** | SI > EMI > Size | High-speed, RF |
| **DFM Optimized** | DFM > Cost > Size | High-volume production |

## Validation Framework

8 validation domains (all must pass):

1. **DRC** - Design Rule Check (KiCad engine)
2. **ERC** - Electrical Rule Check
3. **IPC-2221** - High-current trace compliance
4. **Signal Integrity** - Impedance, crosstalk, length matching
5. **Thermal** - Junction temps, thermal via density
6. **DFM** - Aspect ratios, copper balance
7. **Best Practices** - 30+ industry rules
8. **Automated Testing** - Regression tests

## Multi-LLM Validation

Design artifacts are validated by multiple AI models:

- **Claude Opus 4** - Primary analysis and suggestions
- **Gemini 2.5 Pro** - Cross-validation and alternatives
- **Domain Expert Validators** - DRC, SPICE, Thermal FEA engines

Consensus engine combines results and provides final score with audit trail.

## API Integration

All commands call the EE Design Partner API:

```
Base URL: https://api.adverant.ai/ee-design/api/v1
WebSocket: wss://api.adverant.ai/ee-design/ws
```

Real-time updates via WebSocket events:
- `simulation:progress`
- `layout:iteration`
- `validation:result`

## Environment Requirements

- NEXUS_API_KEY - Nexus API authentication
- ANTHROPIC_API_KEY - Claude API access
- OPENROUTER_API_KEY - Gemini 2.5 Pro access

## Reference Project

The `foc-esc-heavy-lift` project demonstrates the complete pipeline:
- 10-layer PCB, 164 components
- 200A continuous, 400A peak
- Triple-redundant MCUs
- 18× SiC MOSFETs
- 15,000+ lines firmware