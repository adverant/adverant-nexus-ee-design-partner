---
name: component-select
displayName: "Component Selector"
description: "AI-assisted component selection with real-time pricing and availability"
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
  - /component-select
  - /select-component
  - /part-search

capabilities:
  - name: search
    description: Search for components by specifications
    parameters:
      - name: type
        type: string
        required: true
        description: "Component type: mcu, mosfet, regulator, capacitor, etc."
      - name: specs
        type: object
        required: true
        description: Required specifications
      - name: constraints
        type: object
        required: false
        description: "Constraints: manufacturers, packages, grade"

  - name: compare
    description: Compare alternative components
    parameters:
      - name: parts
        type: array
        required: true
        description: Part numbers to compare
      - name: criteria
        type: array
        required: false
        description: Comparison criteria

  - name: availability
    description: Check stock and pricing
    parameters:
      - name: parts
        type: array
        required: true
        description: Part numbers to check
      - name: quantity
        type: number
        required: true
        description: Required quantity
---

# Component Selector

## Overview

AI-powered component selection with live pricing from distributors (Digi-Key, Mouser, LCSC). Find optimal parts based on specifications, availability, and cost.

## Usage

### Search by Specifications

```bash
# Find MCU
/component-select mcu --core="cortex-m7" --flash=">1MB" --ram=">512KB" --can --usb

# Find MOSFET
/component-select mosfet --vds=">60V" --id=">30A" --rds="<5m" --package=TO-247

# Find voltage regulator
/component-select regulator --type=switching --vin="5-24V" --vout=3.3V --iout=">2A"

# Find capacitor
/component-select capacitor --value=10uF --voltage=">16V" --package=0805 --dielectric=X5R
```

### Compare Parts

```bash
# Compare MOSFETs
/component-select compare IPT015N10N5 IRFB7430 FDP3632

# Detailed comparison
/component-select compare STM32H755ZIT6 STM32H743ZIT6 --criteria=price,flash,peripherals
```

### Check Availability

```bash
# Stock check
/component-select availability STM32G474RET6 --qty=100

# Multi-distributor check
/component-select availability TPS62840 --distributors=digikey,mouser,lcsc
```

## Component Categories

| Category | Types |
|----------|-------|
| **MCU** | ARM Cortex, AVR, PIC, RISC-V |
| **Power** | LDO, Buck, Boost, SEPIC, Flyback |
| **Semiconductors** | MOSFET, BJT, IGBT, Diode, LED |
| **Passives** | Resistor, Capacitor, Inductor, Ferrite |
| **Connectors** | USB, Ethernet, Power, Header |
| **Sensors** | Temperature, IMU, Current, Voltage |
| **Memory** | Flash, EEPROM, FRAM, SRAM |
| **Interface** | CAN, RS-485, Ethernet PHY |

## Search Filters

### Grade Requirements

```bash
# Automotive grade
/component-select --grade=automotive --aec-q100

# Industrial temperature
/component-select --temp-range="-40,125"

# Military grade
/component-select --grade=military --mil-std
```

### Manufacturer Preferences

```bash
# Preferred manufacturers
/component-select --prefer="TI,STMicroelectronics,Infineon"

# Avoid manufacturers
/component-select --avoid="Microchip"

# Single source only
/component-select --single-source
```

### Package Constraints

```bash
# SMD only
/component-select --package-type=smd

# Specific packages
/component-select --package="QFN,LQFP,BGA"

# Max height
/component-select --max-height=2mm
```

## Pricing & Availability

### Live Pricing

```
Part: STM32G474RET6
==============================
Distributor   | Stock    | Price (1)  | Price (100)
--------------+----------+------------+------------
Digi-Key      | 2,450    | $7.85      | $6.12
Mouser        | 1,890    | $7.92      | $6.18
LCSC          | 5,000    | $5.20      | $4.85
Arrow         | 3,200    | $7.50      | $5.95
```

### Lead Time Alerts

- ⚠️ Extended lead time (>12 weeks)
- 🔴 EOL (End of Life) warning
- 🟢 In stock, ready to ship

## BOM Optimization

```bash
# Optimize BOM for cost
/component-select optimize-bom --project=proj-001 --target=cost

# Consolidate parts
/component-select consolidate --project=proj-001 --tolerance=10%
```

## API Endpoint

```
POST /ee-design/api/v1/components/search
```

Request body:
```json
{
  "type": "mcu",
  "specifications": {
    "core": "cortex-m7",
    "flash": {"min": 1048576},
    "ram": {"min": 524288},
    "peripherals": ["can", "usb", "ethernet"]
  },
  "constraints": {
    "manufacturers": ["STMicroelectronics"],
    "packages": ["LQFP-144", "LQFP-176"],
    "grade": "industrial"
  },
  "quantity": 100
}
```

## Integration

Part of EE Design Partner Phase 2 (Architecture). Feeds into:
- Schematic component library
- BOM generation
- Cost estimation
- Procurement planning
