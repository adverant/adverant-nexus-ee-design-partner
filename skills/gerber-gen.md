---
name: gerber-gen
displayName: "Gerber Generator"
description: "Generate manufacturing files including Gerbers, drill files, and fabrication outputs"
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
  - /gerber-gen
  - /gerber
  - /manufacturing-files

capabilities:
  - name: generate
    description: Generate complete manufacturing package
    parameters:
      - name: layout_id
        type: string
        required: true
        description: PCB layout identifier
      - name: format
        type: string
        required: false
        description: "Format: gerber_x2, rs274x"
      - name: vendor
        type: string
        required: false
        description: "Vendor preset: pcbway, jlcpcb, oshpark"

  - name: export_bom
    description: Export Bill of Materials
    parameters:
      - name: project_id
        type: string
        required: true
        description: Project identifier
      - name: format
        type: string
        required: false
        description: "Format: csv, xlsx, jlcpcb"

  - name: export_pnp
    description: Export Pick and Place file
    parameters:
      - name: layout_id
        type: string
        required: true
        description: PCB layout identifier
      - name: format
        type: string
        required: false
        description: "Format: csv, kicad, altium"

  - name: generate_assembly
    description: Generate assembly drawings
    parameters:
      - name: layout_id
        type: string
        required: true
        description: PCB layout identifier
      - name: include
        type: array
        required: false
        description: "Include: top, bottom, stackup, dimensions"
---

# Gerber Generator

## Overview

Generate production-ready manufacturing files from PCB layouts. Supports all major PCB vendors and includes comprehensive fabrication documentation.

## Usage

### Generate Complete Package

```bash
# Full manufacturing package
/gerber-gen --layout=pcb-001

# Vendor-specific package
/gerber-gen --layout=pcb-001 --vendor=jlcpcb

# With assembly files
/gerber-gen --layout=pcb-001 --assembly --bom
```

### Generate Specific Files

```bash
# Gerbers only
/gerber-gen gerbers --layout=pcb-001 --format=x2

# Drill files
/gerber-gen drill --layout=pcb-001 --format=excellon2

# Pick and place
/gerber-gen pnp --layout=pcb-001 --format=jlcpcb

# BOM export
/gerber-gen bom --project=proj-001 --format=xlsx
```

## Output Files

### Gerber Files (X2 Format)

| File | Layer | Extension |
|------|-------|-----------|
| Top Copper | F.Cu | .gtl |
| Bottom Copper | B.Cu | .gbl |
| Top Solder Mask | F.Mask | .gts |
| Bottom Solder Mask | B.Mask | .gbs |
| Top Silkscreen | F.SilkS | .gto |
| Bottom Silkscreen | B.SilkS | .gbo |
| Board Outline | Edge.Cuts | .gm1 |
| Top Paste | F.Paste | .gtp |
| Bottom Paste | B.Paste | .gbp |
| Inner Layers | In*.Cu | .g2, .g3... |

### Drill Files

| File | Contents |
|------|----------|
| PTH.drl | Plated through holes |
| NPTH.drl | Non-plated holes |
| PTH.drl_map | Drill map |

### Assembly Files

| File | Contents |
|------|----------|
| BOM.csv | Bill of Materials |
| CPL.csv | Component placement (PnP) |
| Assembly_Top.pdf | Top assembly drawing |
| Assembly_Bottom.pdf | Bottom assembly drawing |

## Vendor Presets

### JLCPCB

```bash
/gerber-gen --layout=pcb-001 --vendor=jlcpcb
```

Generates:
- Gerber X2 format
- Excellon drill files
- JLCPCB BOM format (Designator, Footprint, Quantity, MPN)
- JLCPCB CPL format (Designator, Mid X, Mid Y, Layer, Rotation)

### PCBWay

```bash
/gerber-gen --layout=pcb-001 --vendor=pcbway
```

Generates:
- RS-274X Gerbers
- PCBWay-compatible naming
- Assembly drawings included

### OSHPark

```bash
/gerber-gen --layout=pcb-001 --vendor=oshpark
```

Generates:
- OSHPark-compatible package
- Includes .kicad_pcb for preview

## Verification

Built-in verification checks:
- Gerber integrity validation
- Layer alignment verification
- Drill-to-copper clearance
- Outline closure check
- Minimum feature size check

## API Endpoint

```
POST /ee-design/api/v1/manufacturing/gerbers
```

Request body:
```json
{
  "layoutId": "pcb-uuid",
  "format": "gerber_x2",
  "vendor": "jlcpcb",
  "options": {
    "includeAssembly": true,
    "includeBom": true,
    "bomFormat": "jlcpcb",
    "pnpFormat": "jlcpcb"
  }
}
```

Response:
```json
{
  "success": true,
  "package": {
    "gerbers": "https://storage/gerbers.zip",
    "drill": "https://storage/drill.zip",
    "bom": "https://storage/bom.csv",
    "pnp": "https://storage/pnp.csv",
    "assembly": "https://storage/assembly.pdf"
  },
  "verification": {
    "passed": true,
    "warnings": []
  }
}
```

## Integration

Part of EE Design Partner Phase 6 (Manufacturing Prep). Output feeds into:
- Vendor quoting
- DFM analysis
- Production ordering
