---
name: ee-architecture
displayName: "EE Architecture Designer"
description: "Generate system architecture and block diagrams from requirements"
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
  - /ee-architecture
  - /architecture
  - /block-diagram

capabilities:
  - name: generate
    description: Generate system architecture from requirements
    parameters:
      - name: requirements
        type: string
        required: true
        description: Natural language requirements
      - name: constraints
        type: object
        required: false
        description: Design constraints (cost, size, power)

  - name: block_diagram
    description: Generate block diagram
    parameters:
      - name: project_id
        type: string
        required: true
        description: Project identifier
      - name: format
        type: string
        required: false
        description: "Format: mermaid, svg, pdf"

  - name: power_budget
    description: Generate power budget analysis
    parameters:
      - name: project_id
        type: string
        required: true
        description: Project identifier
---

# EE Architecture Designer

## Overview

Generate comprehensive system architectures from natural language requirements. Produces block diagrams, power budgets, and interface specifications.

## Usage

### Generate Architecture

```bash
# From requirements
/ee-architecture "200A FOC ESC with triple redundant MCUs and CAN bus"

# With constraints
/ee-architecture "Battery-powered sensor node" --budget=$50 --size=50x50mm --power=100mW
```

### Generate Block Diagram

```bash
# Mermaid format
/ee-architecture diagram --project=proj-001 --format=mermaid

# Visual diagram
/ee-architecture diagram --project=proj-001 --format=svg
```

### Power Budget

```bash
# Generate power analysis
/ee-architecture power-budget --project=proj-001

# With operating modes
/ee-architecture power-budget --project=proj-001 --modes=active,sleep,standby
```

## Output Example

### Block Diagram (Mermaid)

```mermaid
graph TB
    subgraph Power
        VIN[12V Input] --> FUSE[Fuse]
        FUSE --> REG5V[5V Regulator]
        FUSE --> REG3V3[3.3V Regulator]
    end

    subgraph Control
        REG3V3 --> MCU1[STM32H755<br>Main Controller]
        REG3V3 --> MCU2[AURIX TC377<br>Safety Monitor]
        REG3V3 --> MCU3[STM32G4<br>Gate Driver]
    end

    subgraph Power Stage
        VIN --> GATE[Gate Drivers]
        GATE --> MOSFET[18x SiC MOSFETs]
        MOSFET --> MOTOR[Motor Output]
    end

    subgraph Communication
        MCU1 --> CAN[CAN Bus]
        MCU2 --> CAN
        CAN --> EXT[External Interface]
    end
```

### Power Budget

| Subsystem | Active | Sleep | Notes |
|-----------|--------|-------|-------|
| MCU Main | 150mA | 5µA | STM32H755 |
| MCU Safety | 80mA | 10µA | AURIX TC377 |
| Gate Drivers | 50mA | 0 | Per phase |
| CAN Transceivers | 60mA | 0 | 2x |
| Sensors | 20mA | 0 | Current sensors |
| **Total** | **360mA** | **15µA** | At 3.3V |

## API Endpoint

```
POST /ee-design/api/v1/architecture/generate
```

## Integration

Part of EE Design Partner Phase 2 (Architecture). Output feeds into:
- Component selection
- Schematic generation
- Power supply design
