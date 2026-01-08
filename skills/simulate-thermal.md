---
name: simulate-thermal
displayName: "Thermal Simulation"
description: "Thermal analysis including steady-state, transient, and CFD airflow simulation"
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
  - Task

triggers:
  - /simulate-thermal
  - /thermal
  - /sim-thermal

capabilities:
  - name: steady_state
    description: Steady-state thermal analysis
    parameters:
      - name: layout_id
        type: string
        required: true
        description: PCB layout identifier
      - name: ambient_temp
        type: number
        required: true
        description: Ambient temperature (°C)
      - name: heat_sources
        type: array
        required: false
        description: Component power dissipation overrides

  - name: transient
    description: Transient thermal analysis
    parameters:
      - name: layout_id
        type: string
        required: true
        description: PCB layout identifier
      - name: duration
        type: number
        required: true
        description: Simulation duration (seconds)
      - name: time_step
        type: number
        required: false
        description: Time step for transient (seconds)
      - name: power_profile
        type: array
        required: false
        description: Time-varying power profile

  - name: cfd
    description: Computational Fluid Dynamics airflow analysis
    parameters:
      - name: layout_id
        type: string
        required: true
        description: PCB layout identifier
      - name: enclosure
        type: object
        required: true
        description: Enclosure dimensions and vents
      - name: airflow
        type: object
        required: false
        description: Forced airflow specifications
---

# Thermal Simulation

## Overview

Perform comprehensive thermal analysis of PCB designs using OpenFOAM and Elmer FEA solvers. Identify hot spots, optimize thermal via placement, and verify designs meet thermal requirements.

## Usage

### Steady-State Analysis

```bash
# Basic thermal analysis
/thermal steady --layout=pcb-001 --ambient=25

# With custom power dissipation
/thermal steady --layout=pcb-001 --ambient=40 \
  --power="U1:2.5W,Q1:5W,Q2:5W"

# High ambient conditions
/thermal steady --layout=pcb-001 --ambient=85 --derating
```

### Transient Analysis

```bash
# Power-on thermal transient
/thermal transient --layout=pcb-001 --duration=60s --startup

# Pulsed load analysis
/thermal transient --layout=pcb-001 --duration=10s \
  --pulse="Q1:10W@50%,1Hz"

# Thermal cycling stress
/thermal transient --layout=pcb-001 --cycle="-40,85" --cycles=100
```

### CFD Airflow Analysis

```bash
# Natural convection in enclosure
/thermal cfd --layout=pcb-001 --enclosure="100x80x30mm" --vents="bottom,top"

# Forced air cooling
/thermal cfd --layout=pcb-001 --fan="40mm,2CFM" --position="inlet"

# Heatsink optimization
/thermal cfd --layout=pcb-001 --heatsink="U1:25x25x10mm"
```

## Analysis Results

### Thermal Maps

Generated outputs:
- **Temperature distribution** - Color-coded thermal image
- **Heat flux vectors** - Heat flow visualization
- **Isothermal contours** - Temperature gradient lines
- **Hot spot identification** - Flagged high-temp regions

### Key Metrics

| Metric | Description | Typical Limit |
|--------|-------------|---------------|
| T_max | Maximum temperature | <85°C |
| T_junction | Component junction temp | <125°C |
| ΔT | Temperature rise above ambient | <40°C |
| θ_ja | Junction-to-ambient resistance | Component-specific |
| Airflow | Required CFM | Design-specific |

### Recommendations

The AI generates thermal improvement suggestions:
- Additional thermal vias needed
- Copper pour optimization
- Component placement changes
- Heatsink requirements
- Airflow requirements

## Material Properties

Built-in material database:
| Material | k (W/m·K) | Cp (J/kg·K) |
|----------|-----------|-------------|
| FR4 | 0.3 | 1100 |
| Copper | 385 | 385 |
| Aluminum | 205 | 900 |
| Solder | 50 | 180 |
| Silicon | 150 | 700 |
| Thermal paste | 4-8 | 1000 |

## Heat Source Extraction

Automatic power extraction from:
- Schematic annotations
- Component datasheets
- SPICE simulation results
- Manual overrides

## Solver Backends

| Solver | Type | Use Case |
|--------|------|----------|
| Elmer FEA | Finite Element | PCB/component analysis |
| OpenFOAM | CFD | Airflow, enclosure |
| Analytical | Fast estimate | Quick thermal checks |

## API Endpoint

```
POST /ee-design/api/v1/simulation/thermal
```

Request body:
```json
{
  "layoutId": "pcb-uuid",
  "analysis": "steady_state",
  "parameters": {
    "ambientTemp": 25,
    "heatSources": [
      {"component": "U1", "power": 2.5},
      {"component": "Q1", "power": 5.0}
    ],
    "boundaryConditions": [
      {"surface": "bottom", "type": "convection", "coefficient": 10}
    ]
  },
  "outputs": ["thermal_map", "max_temp", "recommendations"]
}
```

## Integration

Part of EE Design Partner Phase 4 (Simulation). Results influence:
- PCB layout thermal via placement
- Component selection (power rating)
- Enclosure design requirements
- Reliability estimation (MTBF)
