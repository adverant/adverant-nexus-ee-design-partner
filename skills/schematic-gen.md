---
name: schematic-gen
displayName: "Schematic Generator"
description: "AI-assisted schematic generation from natural language requirements, specifications, or reference designs"
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
  - Task

triggers:
  - /schematic-gen
  - /schematic
  - /sch-gen

capabilities:
  - name: generate
    description: Generate complete schematic from requirements
    parameters:
      - name: requirements
        type: string
        required: true
        description: Natural language description of circuit requirements
      - name: mcu
        type: string
        required: false
        description: Target MCU part number
      - name: power
        type: object
        required: false
        description: Power requirements (input voltage, rails)
      - name: interfaces
        type: array
        required: false
        description: Required interfaces (UART, SPI, I2C, etc.)

  - name: generate_block
    description: Generate specific schematic block
    parameters:
      - name: block_type
        type: string
        required: true
        description: "Block type: power_supply, mcu, interface, protection, sensor"
      - name: specifications
        type: object
        required: true
        description: Block-specific specifications

  - name: add_component
    description: Add component to existing schematic
    parameters:
      - name: schematic_id
        type: string
        required: true
        description: Schematic to modify
      - name: component
        type: object
        required: true
        description: Component specifications

  - name: export
    description: Export schematic to different formats
    parameters:
      - name: schematic_id
        type: string
        required: true
        description: Schematic identifier
      - name: format
        type: string
        required: true
        description: "Export format: kicad, eagle, altium, pdf"
---

# Schematic Generator

## Overview

Generate professional schematics from natural language requirements or specifications. The generator understands circuit topologies, component selection, and automatically applies best practices.

## Usage

### Generate Complete Schematic

```bash
/schematic-gen "12V to 3.3V/5V power supply with USB-C input, STM32G4 MCU, CAN interface, and 4 analog inputs"
```

### Generate Power Supply Block

```bash
/schematic-gen power --input=12V --outputs="3.3V@1A,5V@2A" --type=switching
```

### Generate MCU Block

```bash
/schematic-gen mcu --part=STM32H755ZIT6 --crystal=25MHz --debug=swd
```

### Generate Interface Block

```bash
/schematic-gen interface can --instance=1 --termination=120R --esd=yes
/schematic-gen interface usb-c --power-delivery=65W
/schematic-gen interface ethernet --phy=DP83825I --isolation=yes
```

### Generate Protection Circuit

```bash
/schematic-gen protection --type=reverse-polarity --input=12-48V --current=10A
/schematic-gen protection --type=esd --interfaces=usb,can,gpio
```

## Capabilities

### Supported Block Types

| Block Type | Description |
|------------|-------------|
| `power_input` | Power input with protection |
| `voltage_regulator` | Linear/switching regulators |
| `mcu` | Microcontroller with support circuitry |
| `crystal_oscillator` | Crystal/oscillator circuits |
| `reset_circuit` | Reset with button and watchdog |
| `debug_interface` | SWD/JTAG debug connectors |
| `communication_interface` | UART, SPI, I2C, CAN, USB, Ethernet |
| `analog_input` | ADC input conditioning |
| `digital_io` | GPIO buffers, level shifters |
| `power_output` | High-side/low-side drivers |
| `protection` | ESD, reverse polarity, overcurrent |
| `filtering` | EMI filters, signal conditioning |

### Supported MCU Families

- STM32 (all series)
- ESP32, ESP32-S3, ESP32-C3
- TI TMS320 (C2000 series)
- Infineon AURIX
- Nordic nRF
- Raspberry Pi Pico (RP2040)
- NXP i.MX RT

### Component Selection

The generator automatically selects components based on:
- Required specifications (voltage, current, frequency)
- Availability from major distributors
- Cost optimization
- Preferred manufacturers
- Grade requirements (commercial, industrial, automotive)

## Output

Generated schematics include:
- KiCad schematic files (.kicad_sch)
- Netlist (for layout)
- BOM (Bill of Materials)
- ERC report (Electrical Rule Check)

## API Endpoint

```
POST /ee-design/api/v1/schematic/generate
```

Request body:
```json
{
  "projectId": "project-uuid",
  "requirements": {
    "description": "12V power supply with STM32",
    "targetMcu": "STM32G474RET6",
    "powerRequirements": {
      "inputVoltage": 12,
      "outputVoltages": [
        {"name": "3V3", "voltage": 3.3, "current": 0.5},
        {"name": "5V", "voltage": 5.0, "current": 1.0}
      ]
    },
    "interfaces": [
      {"type": "uart", "count": 2},
      {"type": "can", "count": 1},
      {"type": "spi", "count": 1}
    ]
  }
}
```

## Integration

Part of the EE Design Partner Phase 3 (Schematic Capture). Output feeds directly into:
- Phase 4: Simulation (SPICE analysis)
- Phase 5: PCB Layout (netlist import)
