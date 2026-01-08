---
name: dfm-check
displayName: "DFM Check"
description: "Design for Manufacturability analysis against vendor capabilities"
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
  - /dfm-check
  - /dfm
  - /manufacturability

capabilities:
  - name: analyze
    description: Run complete DFM analysis
    parameters:
      - name: layout_id
        type: string
        required: true
        description: PCB layout identifier
      - name: vendor
        type: string
        required: false
        description: "Target vendor for capability check"
      - name: technology
        type: string
        required: false
        description: "Technology: standard, hdi, flex, rigid-flex"

  - name: check_vendor
    description: Check design against vendor capabilities
    parameters:
      - name: layout_id
        type: string
        required: true
        description: PCB layout identifier
      - name: vendor
        type: string
        required: true
        description: "Vendor: pcbway, jlcpcb, eurocircuits"

  - name: optimize
    description: Suggest DFM optimizations
    parameters:
      - name: layout_id
        type: string
        required: true
        description: PCB layout identifier
      - name: target
        type: string
        required: false
        description: "Optimize for: cost, yield, reliability"
---

# DFM Check

## Overview

Analyze PCB designs for manufacturability against specific vendor capabilities. Identify potential fabrication issues before ordering to maximize yield and minimize cost.

## Usage

### Basic DFM Analysis

```bash
# Standard DFM check
/dfm-check --layout=pcb-001

# Vendor-specific check
/dfm-check --layout=pcb-001 --vendor=jlcpcb

# HDI technology check
/dfm-check --layout=pcb-001 --technology=hdi
```

### Vendor Capability Check

```bash
# Compare against JLCPCB capabilities
/dfm-check vendor --layout=pcb-001 --vendor=jlcpcb

# Multi-vendor comparison
/dfm-check compare --layout=pcb-001 --vendors=jlcpcb,pcbway,oshpark
```

### Optimization

```bash
# Cost optimization suggestions
/dfm-check optimize --layout=pcb-001 --target=cost

# Yield optimization
/dfm-check optimize --layout=pcb-001 --target=yield
```

## DFM Checks

### Minimum Features

| Feature | Standard | HDI | Check |
|---------|----------|-----|-------|
| Trace Width | 0.15mm | 0.075mm | ✓ |
| Trace Spacing | 0.15mm | 0.075mm | ✓ |
| Via Drill | 0.3mm | 0.15mm | ✓ |
| Via Pad | 0.6mm | 0.35mm | ✓ |
| Annular Ring | 0.15mm | 0.075mm | ✓ |
| SMD Pad | 0.25mm | 0.2mm | ✓ |

### Board-Level Checks

| Check | Description |
|-------|-------------|
| Copper Balance | Layer copper distribution |
| Aspect Ratio | Via depth to diameter ratio |
| Solder Mask | Dam, opening sizes |
| Silkscreen | Min line width, spacing |
| Panel Utilization | Efficient panelization |
| Edge Clearance | Components to board edge |

### Assembly Checks

| Check | Description |
|-------|-------------|
| Component Spacing | Pick-and-place clearance |
| Fiducial Presence | Global/local fiducials |
| Orientation Marks | Pin 1 indicators |
| Tombstoning Risk | Pad symmetry for passives |
| Solder Bridging | Fine-pitch spacing |
| BGA Escape | Via-in-pad, dogbone |

## Vendor Capabilities

### JLCPCB Standard

| Parameter | Capability |
|-----------|------------|
| Layers | 1-20 |
| Min Trace | 0.127mm (5mil) |
| Min Space | 0.127mm (5mil) |
| Min Drill | 0.2mm |
| Min Via | 0.45mm pad |
| Aspect Ratio | 10:1 |

### PCBWay Standard

| Parameter | Capability |
|-----------|------------|
| Layers | 1-32 |
| Min Trace | 0.1mm (4mil) |
| Min Space | 0.1mm (4mil) |
| Min Drill | 0.15mm |
| Min Via | 0.4mm pad |
| Aspect Ratio | 12:1 |

## Analysis Report

### Summary

```
DFM Analysis Report - pcb-001
==============================
Overall Score: 94/100
Vendor Compatibility: JLCPCB ✓, PCBWay ✓, OSHPark ⚠️

Critical Issues: 0
Warnings: 3
Suggestions: 5
```

### Issue Categories

| Category | Count | Impact |
|----------|-------|--------|
| Spacing Violations | 0 | - |
| Drill Issues | 1 | Low |
| Copper Balance | 2 | Medium |
| Assembly Risk | 3 | Low |

### Cost Optimization

```
Current Design: $8.50/board (10 qty)
Optimized Design: $6.20/board (10 qty)
Savings: 27%

Suggestions:
- Increase via drill 0.25→0.3mm (standard process)
- Reduce to 4 layers (sufficient for routing)
- Remove redundant test points
```

## API Endpoint

```
POST /ee-design/api/v1/manufacturing/dfm
```

Request body:
```json
{
  "layoutId": "pcb-uuid",
  "vendor": "jlcpcb",
  "technology": "standard",
  "options": {
    "checkAssembly": true,
    "optimizeFor": "cost"
  }
}
```

## Integration

Part of EE Design Partner Phase 6 (Manufacturing Prep). Results influence:
- Vendor selection
- Design modifications
- Cost estimation
- Lead time planning
