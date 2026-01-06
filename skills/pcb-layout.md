---
name: pcb-layout
displayName: "PCB Layout Generator"
description: "Generate PCB layouts using multi-agent competing strategies with Ralph Loop optimization"
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
  - /pcb-layout
  - /layout

capabilities:
  - name: generate
    description: Generate PCB layout from schematic using multi-agent tournament
    parameters:
      - name: schematic_id
        type: string
        required: true
        description: Schematic identifier or file path
      - name: layers
        type: number
        required: false
        default: 4
        description: Number of PCB layers (2, 4, 6, 8, 10, etc.)
      - name: width
        type: number
        required: false
        description: Board width in mm
      - name: height
        type: number
        required: false
        description: Board height in mm
      - name: strategy
        type: string
        required: false
        description: "Agent strategy: conservative, compact, thermal, emi, dfm, or 'all' for tournament"
      - name: max_iterations
        type: number
        required: false
        default: 100
        description: Maximum Ralph Loop iterations
      - name: target_score
        type: number
        required: false
        default: 95
        description: Target validation score (0-100)

  - name: validate
    description: Run validation on existing PCB layout
    parameters:
      - name: layout_id
        type: string
        required: true
        description: Layout identifier or file path

  - name: render
    description: Render PCB layer images
    parameters:
      - name: layout_id
        type: string
        required: true
        description: Layout identifier
      - name: layers
        type: array
        required: false
        description: Specific layers to render
      - name: format
        type: string
        required: false
        default: "png"
        description: "Output format: png, svg, pdf"

  - name: refine
    description: Refine existing layout based on validation feedback
    parameters:
      - name: layout_id
        type: string
        required: true
        description: Layout to refine
      - name: focus
        type: string
        required: false
        description: "Area to focus: thermal, si, dfm, routing"
---

# PCB Layout Generator

## Overview

Generate production-ready PCB layouts using a multi-agent competing tournament system with the Ralph Loop optimization algorithm.

## Usage

### Basic Layout Generation

```bash
/pcb-layout generate --schematic=power-stage.kicad_sch --layers=10
```

### Specify Strategy

```bash
# Use thermal-optimized strategy
/pcb-layout generate --schematic=sch-001 --strategy=thermal

# Run full tournament with all agents
/pcb-layout generate --schematic=sch-001 --strategy=all
```

### Custom Board Dimensions

```bash
/pcb-layout generate --schematic=sch-001 --width=100 --height=80 --layers=6
```

### Validate Existing Layout

```bash
/pcb-layout validate --layout=layout.kicad_pcb
```

### Render Layer Images

```bash
/pcb-layout render --layout=layout-001 --format=png --layers=F.Cu,B.Cu,F.SilkS
```

## Agent Strategies

### Conservative Agent
- **Priority**: Reliability > DFM > Cost
- **Best For**: High-power electronics, industrial, automotive
- **Characteristics**:
  - Wide trace spacing (150% of minimum)
  - Large via sizes
  - Conservative thermal management
  - Extra ground plane stitching

### Aggressive Compact Agent
- **Priority**: Size > Cost > DFM
- **Best For**: Consumer electronics, space-constrained designs
- **Characteristics**:
  - Minimum feature sizes where safe
  - Dense component placement
  - Optimized routing density
  - Smaller via sizes

### Thermal Optimized Agent
- **Priority**: Thermal > Reliability > Size
- **Best For**: Power electronics, motor controllers, high-current
- **Characteristics**:
  - Thermal via arrays under hot components
  - Wide copper pours for heat spreading
  - Strategic component placement for airflow
  - Copper balance optimization

### EMI Optimized Agent
- **Priority**: Signal Integrity > EMI > Size
- **Best For**: High-speed digital, RF, mixed-signal
- **Characteristics**:
  - Controlled impedance routing
  - Differential pair management
  - Ground plane integrity
  - Shield placement

### DFM Optimized Agent
- **Priority**: Manufacturability > Cost > Size
- **Best For**: High-volume production
- **Characteristics**:
  - Assembly-friendly component placement
  - Testpoint accessibility
  - Fiducial placement
  - Copper balance 40-60%

## Ralph Loop Tournament

The Ralph Loop runs a tournament with these phases:

1. **Rounds 1-30**: All 5 agents compete → top 3 advance
2. **Rounds 31-60**: Top 3 compete → top 2 advance
3. **Rounds 61-90**: Top 2 compete → winner advances
4. **Rounds 91-100**: Winner refines with expert feedback

### Convergence Criteria

Stop when ANY condition is met:
- Perfect score (≥100) AND zero critical violations
- Good enough (≥95) AND plateau (no improvement in 10 iterations)
- Plateau (<0.1 point improvement over 10 iterations)
- Timeout (100 iterations OR 4 hours)

## Validation Domains

Each layout is validated across 8 domains:

| Domain | Weight | Criteria |
|--------|--------|----------|
| **DRC** | 30% | Zero critical violations |
| **ERC** | 20% | All nets connected |
| **IPC-2221** | 15% | Trace widths for current |
| **Signal Integrity** | 10% | Impedance ±10% |
| **Thermal** | 10% | Junction temps in spec |
| **DFM** | 10% | Aspect ratios <10:1 |
| **Best Practices** | 5% | 30+ industry rules |

## Output Files

- `layout.kicad_pcb` - KiCad PCB file
- `layers/*.png` - Layer images
- `3d-view.png` - 3D board render
- `validation-report.json` - Detailed validation results
- `tournament-log.json` - Agent competition history

## Performance

For a typical 10-layer, 164-component board:
- **Average convergence**: 72 iterations
- **Runtime**: 2.5-3.5 hours
- **Success rate**: 94% (score ≥95/100)
- **Perfect layouts**: 23% (score = 100/100)

## Examples

### FOC ESC Layout

```bash
/pcb-layout generate \
  --schematic=foc-esc.kicad_sch \
  --layers=10 \
  --width=120 \
  --height=100 \
  --strategy=thermal \
  --target_score=98
```

### High-Speed Digital

```bash
/pcb-layout generate \
  --schematic=high-speed-digital.kicad_sch \
  --layers=6 \
  --strategy=emi \
  --max_iterations=150
```