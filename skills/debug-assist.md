---
name: debug-assist
displayName: "Debug Assistant"
description: "AI-assisted hardware and firmware debugging with fault diagnosis"
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
  - /debug-assist
  - /debug
  - /troubleshoot

capabilities:
  - name: diagnose
    description: Diagnose issue from symptoms
    parameters:
      - name: project_id
        type: string
        required: true
        description: Project identifier
      - name: symptoms
        type: string
        required: true
        description: Description of observed symptoms
      - name: context
        type: object
        required: false
        description: Additional context (measurements, logs)

  - name: analyze_fault
    description: Analyze fault codes and errors
    parameters:
      - name: fault_code
        type: string
        required: true
        description: Fault code or error message
      - name: logs
        type: string
        required: false
        description: Relevant log data

  - name: generate_procedure
    description: Generate debugging procedure
    parameters:
      - name: issue_type
        type: string
        required: true
        description: "Type: power, communication, thermal, software"
      - name: project_id
        type: string
        required: true
        description: Project identifier
---

# Debug Assistant

## Overview

AI-powered debugging assistant for hardware and firmware issues. Analyzes symptoms, fault codes, and measurements to identify root causes and suggest fixes.

## Usage

### Diagnose Issues

```bash
# Describe symptoms
/debug "Board not powering up, LED1 not lighting"

# With measurements
/debug "Output voltage low (2.8V instead of 3.3V)" --measure="VIN=12V, VOUT=2.8V, Current=450mA"

# Communication issue
/debug "CAN bus not receiving messages" --logs="can_trace.log"
```

### Analyze Fault Codes

```bash
# MCU fault
/debug fault 0xDEADBEEF --mcu=STM32H755

# Application error
/debug fault ERR_COMM_TIMEOUT --module=motor_driver

# System crash
/debug crash --coredump=crash.bin --symbols=firmware.elf
```

### Generate Debug Procedure

```bash
# Power issue
/debug procedure --type=power --project=proj-001

# Communication debug
/debug procedure --type=can --interface=CAN1

# Thermal issue
/debug procedure --type=thermal --component=U1
```

## Diagnostic Categories

### Power Issues

| Symptom | Possible Causes | Checks |
|---------|-----------------|--------|
| No power | Fuse blown, reverse polarity | Check F1, D1 |
| Low voltage | Overload, bad regulator | Measure load, swap U_REG |
| Noise on rail | Bad capacitor, layout | Scope power, check ESR |
| Hot component | Short, over-current | Thermal camera, current |

### Communication Issues

| Symptom | Possible Causes | Checks |
|---------|-----------------|--------|
| No response | Wrong address, no power | Scope bus, check address |
| Corrupted data | Noise, timing | Check termination, timing |
| Intermittent | Loose connection, EMI | Reseat connectors, shielding |
| Timeout | Bus locked, wrong speed | Reset bus, verify baud |

### Thermal Issues

| Symptom | Possible Causes | Checks |
|---------|-----------------|--------|
| Overheating | Overload, bad thermal | Measure power, check TIM |
| Shutdown | Thermal protection | Check Tj vs max |
| Drift | Temperature coefficient | Characterize vs temp |

## Diagnostic Tools Integration

### Supported Equipment

| Tool | Protocol | Use |
|------|----------|-----|
| Oscilloscope | SCPI | Signal analysis |
| Logic Analyzer | SALEAE | Digital protocols |
| DMM | USB/SERIAL | Voltage/current |
| Thermal Camera | USB | Temperature mapping |
| JTAG Debugger | GDB/SWD | Firmware debug |

### Automated Measurements

```bash
# Automated power rail check
/debug measure power-rails --scope=rigol --project=proj-001

# Protocol analysis
/debug capture spi --analyzer=saleae --duration=1s

# Thermal scan
/debug thermal-scan --camera=flir --area="pcb-001"
```

## Decision Trees

### Power-Up Failure

```
Board doesn't power up
├── Check input voltage
│   ├── <10V → Check power supply
│   └── >10V → Continue
├── Check fuse
│   ├── Open → Replace fuse, find root cause
│   └── OK → Continue
├── Check reverse polarity diode
│   ├── Shorted → Replace D1
│   └── OK → Continue
├── Check regulator output
│   ├── 0V → Check enable, PGOOD
│   ├── Low → Check load, thermal shutdown
│   └── OK → Check MCU
└── Check MCU
    ├── No clock → Check crystal, caps
    └── Clock OK → Debug firmware
```

## Report Generation

```bash
# Generate debug report
/debug report --session=debug-001 --format=pdf

# Export measurements
/debug export --session=debug-001 --format=csv
```

## Debug Report Example

```markdown
# Debug Report - Session debug-001

## Issue
Board powers up but MCU not running

## Analysis
1. Power rails verified OK (3.3V, 5V)
2. Clock signal present (25MHz)
3. NRST held low by external circuit

## Root Cause
R_RST (10k pull-up) was DNP, leaving NRST floating

## Fix
Populated R_RST with 10kΩ resistor

## Verification
- MCU now boots correctly
- All POST tests pass
```

## API Endpoint

```
POST /ee-design/api/v1/debug/diagnose
```

Request body:
```json
{
  "projectId": "proj-uuid",
  "symptoms": "Board not powering up, LED1 not lighting",
  "context": {
    "measurements": {
      "VIN": 12.1,
      "VOUT_3V3": 0,
      "Current": 0.02
    },
    "observations": ["F1 appears intact", "No burning smell"]
  }
}
```

## Integration

Part of EE Design Partner Phase 10 (Field Support). Integrates with:
- Schematic for circuit analysis
- Firmware for error codes
- Test procedures
- Service manual
