---
name: simulate-spice
displayName: "SPICE Simulation"
description: "Run SPICE circuit simulations including DC, AC, transient, noise, and Monte Carlo analysis"
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
  - /simulate-spice
  - /spice
  - /sim-spice

capabilities:
  - name: dc_analysis
    description: DC operating point and sweep analysis
    parameters:
      - name: schematic_id
        type: string
        required: true
        description: Schematic identifier
      - name: sweep_source
        type: string
        required: false
        description: Source to sweep (V1, I1, etc.)
      - name: sweep_range
        type: object
        required: false
        description: "{start, stop, step}"

  - name: ac_analysis
    description: AC small-signal frequency response
    parameters:
      - name: schematic_id
        type: string
        required: true
        description: Schematic identifier
      - name: freq_start
        type: number
        required: true
        description: Start frequency (Hz)
      - name: freq_stop
        type: number
        required: true
        description: Stop frequency (Hz)
      - name: points_per_decade
        type: number
        required: false
        description: Points per decade (default 20)

  - name: transient_analysis
    description: Time-domain transient simulation
    parameters:
      - name: schematic_id
        type: string
        required: true
        description: Schematic identifier
      - name: stop_time
        type: number
        required: true
        description: Simulation end time (seconds)
      - name: step
        type: number
        required: false
        description: Time step (seconds)
      - name: start_time
        type: number
        required: false
        description: Start recording time

  - name: noise_analysis
    description: Noise spectral density analysis
    parameters:
      - name: schematic_id
        type: string
        required: true
        description: Schematic identifier
      - name: output_node
        type: string
        required: true
        description: Output node for noise measurement
      - name: input_source
        type: string
        required: true
        description: Input source reference
      - name: freq_range
        type: object
        required: true
        description: "{start, stop}"

  - name: monte_carlo
    description: Monte Carlo statistical analysis
    parameters:
      - name: schematic_id
        type: string
        required: true
        description: Schematic identifier
      - name: runs
        type: number
        required: true
        description: Number of Monte Carlo runs
      - name: parameters
        type: array
        required: true
        description: Parameters to vary with tolerances
---

# SPICE Simulation

## Overview

Run comprehensive SPICE circuit simulations using ngspice or LTspice backends. Supports DC, AC, transient, noise, and Monte Carlo analysis with automatic netlist generation from schematics.

## Usage

### DC Analysis

```bash
# Operating point analysis
/spice dc --schematic=sch-001

# DC sweep
/spice dc --schematic=sch-001 --sweep=VIN --range="0,12,0.1"

# Temperature sweep
/spice dc --schematic=sch-001 --temp-sweep="-40,125,5"
```

### AC Analysis

```bash
# Frequency response
/spice ac --schematic=sch-001 --freq="1,1e9" --ppd=20

# Bode plot with phase
/spice ac --schematic=sch-001 --freq="1,1e6" --plot=bode

# Stability analysis
/spice ac --schematic=sch-001 --stability --feedback-break=U1.OUT
```

### Transient Analysis

```bash
# Step response
/spice transient --schematic=sch-001 --stop=1ms --step=10ns

# Startup analysis
/spice transient --schematic=sch-001 --stop=10ms --startup

# PWM simulation
/spice transient --schematic=sch-001 --stop=100us --pwm-freq=50kHz
```

### Noise Analysis

```bash
# Input-referred noise
/spice noise --schematic=sch-001 --output=VOUT --input=VIN --freq="1,1M"

# Total integrated noise
/spice noise --schematic=sch-001 --output=VOUT --integrate="10,100k"
```

### Monte Carlo Analysis

```bash
# Statistical analysis
/spice monte-carlo --schematic=sch-001 --runs=1000 \
  --vary="R1:5%,C1:10%,U1.VOS:3mV"

# Worst-case analysis
/spice worst-case --schematic=sch-001 --parameters="all:5%"
```

## Simulation Results

Results include:
- **Waveforms** - Interactive plots with zoom/pan
- **Metrics** - Key measurements (rise time, bandwidth, etc.)
- **Pass/Fail** - Against specified limits
- **Recommendations** - AI-generated improvement suggestions

### Example Metrics

| Metric | Value | Limit | Status |
|--------|-------|-------|--------|
| Bandwidth | 10.5 MHz | >10 MHz | ✓ |
| Phase Margin | 62° | >45° | ✓ |
| Rise Time | 45 ns | <100 ns | ✓ |
| Overshoot | 8% | <10% | ✓ |
| PSRR @ 100Hz | -75 dB | <-60 dB | ✓ |

## Supported Simulators

| Simulator | Backend | Features |
|-----------|---------|----------|
| ngspice | Docker container | Open source, fast |
| LTspice | Wine container | Advanced models |
| Custom | Python/NumPy | Special analyses |

## Model Libraries

Built-in libraries for:
- OpAmps (TI, Analog Devices, Maxim)
- MOSFETs (Infineon, ON Semi)
- Power regulators (TI, Linear Tech)
- MCU I/O models (STM32, ESP32)

## API Endpoint

```
POST /ee-design/api/v1/simulation/spice
```

Request body:
```json
{
  "schematicId": "sch-uuid",
  "analysis": "transient",
  "parameters": {
    "stopTime": 0.001,
    "step": 1e-8,
    "signals": ["VOUT", "IL"]
  },
  "metrics": [
    {"name": "rise_time", "signal": "VOUT", "threshold": [0.1, 0.9]},
    {"name": "overshoot", "signal": "VOUT", "limit": 10}
  ]
}
```

## Integration

Part of EE Design Partner Phase 4 (Simulation). Results feed into:
- Design validation scoring
- Component value optimization
- PCB layout thermal requirements
