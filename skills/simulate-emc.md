---
name: simulate-emc
displayName: "EMC Simulation"
description: "Electromagnetic compatibility simulation for radiated and conducted emissions"
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
  - /simulate-emc
  - /emc
  - /emi

capabilities:
  - name: radiated
    description: Radiated emissions analysis
    parameters:
      - name: layout_id
        type: string
        required: true
        description: PCB layout identifier
      - name: freq_range
        type: object
        required: true
        description: "{start, stop} frequency range"
      - name: standard
        type: string
        required: false
        description: "Compliance standard: CISPR32, FCC_Part15"

  - name: conducted
    description: Conducted emissions analysis
    parameters:
      - name: layout_id
        type: string
        required: true
        description: PCB layout identifier
      - name: power_port
        type: string
        required: true
        description: Power input port
      - name: standard
        type: string
        required: false
        description: "Compliance standard: CISPR32, DO-160"

  - name: susceptibility
    description: EMI susceptibility analysis
    parameters:
      - name: layout_id
        type: string
        required: true
        description: PCB layout identifier
      - name: immunity_level
        type: number
        required: true
        description: Test level (V/m)
---

# EMC Simulation

## Overview

Pre-compliance EMC analysis using openEMS and MEEP solvers. Predict radiated and conducted emissions, identify coupling paths, and optimize filtering before lab testing.

## Usage

### Radiated Emissions

```bash
# Full radiated scan
/emc radiated --layout=pcb-001 --freq="30M-1G" --standard=CISPR32

# Near-field scan
/emc radiated --layout=pcb-001 --near-field --height=5mm

# Specific frequency investigation
/emc radiated --layout=pcb-001 --freq="125MHz" --harmonics=10
```

### Conducted Emissions

```bash
# Power line conducted
/emc conducted --layout=pcb-001 --port=VIN --standard=CISPR32

# Differential/common mode separation
/emc conducted --layout=pcb-001 --port=VIN --mode=both

# Line impedance stabilization
/emc conducted --layout=pcb-001 --lisn=50uH
```

### Susceptibility Analysis

```bash
# Radiated immunity
/emc susceptibility --layout=pcb-001 --level=10 --freq="80M-1G"

# ESD susceptibility
/emc susceptibility --layout=pcb-001 --esd --level=8kV

# Surge immunity
/emc susceptibility --layout=pcb-001 --surge --level=2kV
```

## EMC Standards Support

| Standard | Application | Limits |
|----------|-------------|--------|
| CISPR 32 | Multimedia equipment | Class A/B |
| FCC Part 15 | IT equipment (US) | Class A/B |
| EN 55032 | European EMC | Class A/B |
| DO-160 | Aerospace | Category A-Z |
| MIL-STD-461 | Military | RE102, CE102 |
| CISPR 25 | Automotive | Class 1-5 |

## Analysis Results

### Emissions Plot

Frequency spectrum with:
- Measured emissions (dBµV/m)
- Limit lines (standard-specific)
- Margin to limit
- Harmonic markers

### Emission Sources

Identified sources:
- Clock harmonics
- Switching noise
- Cable radiation
- Slot antenna effects
- Common-mode currents

### Recommendations

AI-generated mitigation strategies:
- Filter component values
- Shielding requirements
- Layout modifications
- Cable treatment
- Grounding improvements

## Coupling Path Analysis

```bash
# Identify coupling paths
/emc coupling --layout=pcb-001 --source=CLK --victim=ADC_IN

# Current return path
/emc current-path --layout=pcb-001 --net=VCC --freq=100MHz
```

## Filter Design

```bash
# Design EMI filter
/emc filter --type=pi --attenuation=40dB --freq=100MHz --impedance=50

# Optimize existing filter
/emc filter-optimize --layout=pcb-001 --target-margin=6dB
```

## API Endpoint

```
POST /ee-design/api/v1/simulation/emc
```

Request body:
```json
{
  "layoutId": "pcb-uuid",
  "analysis": "radiated",
  "parameters": {
    "frequencyRange": {"start": 30e6, "stop": 1e9},
    "standard": "CISPR32",
    "class": "B",
    "distance": 10
  },
  "outputs": ["spectrum", "margin", "sources", "recommendations"]
}
```

## Integration

Part of EE Design Partner Phase 4 (Simulation). Essential for:
- Pre-compliance testing
- Design for EMC
- Certification preparation
- Troubleshooting EMC failures
