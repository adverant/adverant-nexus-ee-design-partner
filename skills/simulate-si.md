---
name: simulate-si
displayName: "Signal Integrity Simulation"
description: "Signal integrity analysis including impedance, crosstalk, eye diagrams, and S-parameters"
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
  - /simulate-si
  - /signal-integrity
  - /si

capabilities:
  - name: impedance
    description: Trace impedance calculation and analysis
    parameters:
      - name: layout_id
        type: string
        required: true
        description: PCB layout identifier
      - name: nets
        type: array
        required: false
        description: Specific nets to analyze
      - name: target_impedance
        type: number
        required: false
        description: Target impedance (ohms)

  - name: crosstalk
    description: Near-end and far-end crosstalk analysis
    parameters:
      - name: layout_id
        type: string
        required: true
        description: PCB layout identifier
      - name: aggressor_net
        type: string
        required: true
        description: Aggressor signal net
      - name: victim_nets
        type: array
        required: true
        description: Victim signal nets

  - name: eye_diagram
    description: Eye diagram analysis for high-speed signals
    parameters:
      - name: layout_id
        type: string
        required: true
        description: PCB layout identifier
      - name: net
        type: string
        required: true
        description: Signal net to analyze
      - name: data_rate
        type: number
        required: true
        description: Data rate (bps)

  - name: s_parameters
    description: S-parameter extraction and analysis
    parameters:
      - name: layout_id
        type: string
        required: true
        description: PCB layout identifier
      - name: ports
        type: array
        required: true
        description: Port definitions for S-param extraction
      - name: freq_range
        type: object
        required: true
        description: "{start, stop} in Hz"
---

# Signal Integrity Simulation

## Overview

Comprehensive signal integrity analysis for high-speed digital and mixed-signal PCB designs. Calculate impedances, analyze crosstalk, generate eye diagrams, and extract S-parameters.

## Usage

### Impedance Analysis

```bash
# Calculate all trace impedances
/si impedance --layout=pcb-001 --all

# Target impedance matching
/si impedance --layout=pcb-001 --nets=CLK,DATA --target=50

# Differential pair impedance
/si impedance --layout=pcb-001 --diff-pairs=USB_DP_DM --target=90
```

### Crosstalk Analysis

```bash
# Crosstalk between adjacent traces
/si crosstalk --layout=pcb-001 --aggressor=CLK --victims=DATA0,DATA1,DATA2

# Crosstalk matrix for bus
/si crosstalk --layout=pcb-001 --bus=DATA[0:7] --matrix

# Time-domain crosstalk
/si crosstalk --layout=pcb-001 --aggressor=CLK --time-domain --edge=1ns
```

### Eye Diagram Analysis

```bash
# Generate eye diagram
/si eye --layout=pcb-001 --net=LVDS_CLK --rate=1.25Gbps

# Eye with jitter analysis
/si eye --layout=pcb-001 --net=SERDES_TX --rate=5Gbps --jitter

# Multi-gigabit with equalization
/si eye --layout=pcb-001 --net=PCIE_TX --rate=8Gbps --ctle=auto
```

### S-Parameter Analysis

```bash
# Extract S-parameters for transmission line
/si s-param --layout=pcb-001 --ports="U1.TX,J1.1" --freq="1M,10G"

# Insertion/return loss
/si s-param --layout=pcb-001 --channels=all --loss-budget

# TDR analysis
/si tdr --layout=pcb-001 --net=DDR_CLK --impedance-profile
```

## Analysis Results

### Impedance Report

| Net | Layer | Width | Gap | Z_single | Z_diff | Target | Status |
|-----|-------|-------|-----|----------|--------|--------|--------|
| USB_DP | L1 | 0.15mm | - | 45Ω | - | 50Ω | ⚠️ |
| USB_DM | L1 | 0.15mm | - | 45Ω | - | 50Ω | ⚠️ |
| USB | L1 | 0.15mm | 0.2mm | - | 88Ω | 90Ω | ✓ |

### Eye Diagram Metrics

| Metric | Value | Spec | Status |
|--------|-------|------|--------|
| Eye Height | 320 mV | >200 mV | ✓ |
| Eye Width | 0.78 UI | >0.6 UI | ✓ |
| Jitter (RMS) | 12 ps | <25 ps | ✓ |
| Rise Time | 85 ps | - | - |
| ISI | 45 mV | - | - |

### Crosstalk Limits

| Standard | NEXT | FEXT |
|----------|------|------|
| USB 2.0 | -25 dB | -30 dB |
| USB 3.0 | -30 dB | -35 dB |
| PCIe Gen3 | -25 dB | -28 dB |
| DDR4 | -20 dB | -25 dB |

## Stackup Analysis

Automatic stackup extraction and analysis:
- Layer thicknesses
- Dielectric constants (Er)
- Loss tangent (tan δ)
- Copper weight/roughness

## Length Matching

```bash
# DDR length matching
/si length-match --layout=pcb-001 --group=DDR_DQ --reference=DDR_CLK

# Differential skew
/si length-match --layout=pcb-001 --diff-pairs=LVDS --max-skew=5mil
```

## Recommendations

AI-generated improvements:
- Trace width adjustments for impedance
- Spacing increases for crosstalk
- Via placement optimization
- Reference plane recommendations
- Guard trace suggestions

## API Endpoint

```
POST /ee-design/api/v1/simulation/signal-integrity
```

Request body:
```json
{
  "layoutId": "pcb-uuid",
  "analysis": "eye_diagram",
  "parameters": {
    "net": "SERDES_TX",
    "dataRate": 5e9,
    "numBits": 10000,
    "jitterAnalysis": true,
    "channelModel": {
      "txOutput": "component:U1.TX",
      "rxInput": "component:U2.RX"
    }
  }
}
```

## Integration

Part of EE Design Partner Phase 4 (Simulation). Critical for:
- High-speed PCB layout validation
- DDR memory interface design
- SerDes and multi-gigabit links
- USB, PCIe, Ethernet compliance
