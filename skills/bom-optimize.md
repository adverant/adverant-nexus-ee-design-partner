---
name: bom-optimize
displayName: "BOM Optimizer"
description: "Optimize Bill of Materials for cost, availability, and consolidation"
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
  - /bom-optimize
  - /bom
  - /optimize-bom

capabilities:
  - name: optimize
    description: Optimize BOM for specified criteria
    parameters:
      - name: project_id
        type: string
        required: true
        description: Project identifier
      - name: target
        type: string
        required: true
        description: "Target: cost, availability, consolidation, all"

  - name: alternatives
    description: Find alternative parts
    parameters:
      - name: part_number
        type: string
        required: true
        description: Part to find alternatives for
      - name: criteria
        type: object
        required: false
        description: Match criteria

  - name: export
    description: Export BOM to various formats
    parameters:
      - name: project_id
        type: string
        required: true
        description: Project identifier
      - name: format
        type: string
        required: true
        description: "Format: csv, xlsx, jlcpcb, octopart"
---

# BOM Optimizer

## Overview

Optimize Bill of Materials for cost reduction, improved availability, and part consolidation while maintaining design integrity.

## Usage

### Optimize BOM

```bash
# Cost optimization
/bom-optimize --project=proj-001 --target=cost

# Availability optimization
/bom-optimize --project=proj-001 --target=availability --min-stock=1000

# Part consolidation
/bom-optimize --project=proj-001 --target=consolidation

# Full optimization
/bom-optimize --project=proj-001 --target=all
```

### Find Alternatives

```bash
# Find alternatives for single part
/bom-optimize alternatives STM32G474RET6 --prefer-in-stock

# With specific criteria
/bom-optimize alternatives TPS62840 --criteria="iq<100nA,package=QFN"
```

### Export BOM

```bash
# CSV export
/bom-optimize export --project=proj-001 --format=csv

# JLCPCB format
/bom-optimize export --project=proj-001 --format=jlcpcb

# Octopart import
/bom-optimize export --project=proj-001 --format=octopart
```

## Optimization Results

### Cost Optimization

```
BOM Optimization Report - proj-001
===================================

Original Cost: $125.40 (100 units)
Optimized Cost: $98.20 (100 units)
Savings: $27.20 (21.7%)

Changes:
1. C1-C20: GRM188R71E104KA01D → CL10B104KB8NNNC
   Savings: $2.80 (equivalent X7R 0603)

2. R1-R50: RC0603FR-0710KL → 0603WAF1002T5E
   Savings: $5.50 (same specs, different vendor)

3. U_REG1: TPS62840 → TPS62842
   Savings: $1.20 (newer part, better price)
```

### Consolidation Report

```
Part Consolidation Opportunities
================================

Current unique part numbers: 87
After consolidation: 62 (-29%)

Consolidations:
- Resistors: 15 values → 10 values (using E24 series)
- Capacitors: 12 values → 8 values (tolerance overlap)
- Connectors: 5 types → 3 types (header unification)
```

## API Endpoint

```
POST /ee-design/api/v1/bom/optimize
```

## Integration

Part of EE Design Partner Phase 2 (Architecture). Works with:
- Component selection
- Vendor quoting
- DFM analysis
