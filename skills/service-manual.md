---
name: service-manual
displayName: "Service Manual Generator"
description: "Generate comprehensive service and maintenance documentation"
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
  - /service-manual
  - /manual
  - /documentation

capabilities:
  - name: generate
    description: Generate complete service manual
    parameters:
      - name: project_id
        type: string
        required: true
        description: Project identifier
      - name: format
        type: string
        required: false
        description: "Format: pdf, html, markdown"
      - name: sections
        type: array
        required: false
        description: Specific sections to generate

  - name: troubleshooting
    description: Generate troubleshooting guide
    parameters:
      - name: project_id
        type: string
        required: true
        description: Project identifier
      - name: symptoms
        type: array
        required: false
        description: Specific symptoms to cover

  - name: maintenance
    description: Generate maintenance schedule
    parameters:
      - name: project_id
        type: string
        required: true
        description: Project identifier
      - name: environment
        type: string
        required: false
        description: "Environment: indoor, outdoor, industrial, marine"
---

# Service Manual Generator

## Overview

Generate professional service documentation including installation guides, troubleshooting procedures, maintenance schedules, and repair instructions.

## Usage

### Generate Complete Manual

```bash
# Full service manual
/service-manual --project=proj-001

# PDF format
/service-manual --project=proj-001 --format=pdf

# Specific sections
/service-manual --project=proj-001 --sections=installation,troubleshooting,maintenance
```

### Generate Troubleshooting Guide

```bash
# All known issues
/service-manual troubleshooting --project=proj-001

# Specific symptoms
/service-manual troubleshooting --project=proj-001 --symptoms="no power,communication error"
```

### Generate Maintenance Schedule

```bash
# Standard maintenance
/service-manual maintenance --project=proj-001

# Environmental considerations
/service-manual maintenance --project=proj-001 --environment=industrial
```

## Document Structure

### Complete Service Manual

```
Service Manual - [Product Name]
================================

1. Introduction
   1.1 Product Overview
   1.2 Specifications
   1.3 Safety Warnings

2. Installation
   2.1 Site Requirements
   2.2 Unpacking
   2.3 Mounting
   2.4 Wiring
   2.5 Initial Setup

3. Operation
   3.1 Controls and Indicators
   3.2 Normal Operation
   3.3 Operating Modes

4. Configuration
   4.1 Parameter Settings
   4.2 Communication Setup
   4.3 Calibration

5. Troubleshooting
   5.1 LED Indicators
   5.2 Error Codes
   5.3 Symptom-Based Diagnosis
   5.4 Decision Trees

6. Maintenance
   6.1 Preventive Maintenance
   6.2 Inspection Checklist
   6.3 Cleaning Procedures
   6.4 Calibration Schedule

7. Repair
   7.1 Field-Replaceable Units
   7.2 Disassembly Procedures
   7.3 Component Replacement
   7.4 Reassembly

8. Technical Reference
   8.1 Schematics
   8.2 PCB Layout
   8.3 Wiring Diagrams
   8.4 Bill of Materials

9. Appendices
   A. Specifications
   B. Warranty Information
   C. Support Contacts
```

## Troubleshooting Section Example

### Error Code Reference

| Code | Description | Cause | Action |
|------|-------------|-------|--------|
| E001 | Over-temperature | Blocked airflow, ambient temp | Check cooling, reduce load |
| E002 | Under-voltage | Power supply, cable | Check input, connections |
| E003 | Communication Error | Cable, termination | Check wiring, termination |
| E004 | Sensor Fault | Sensor, wiring | Replace sensor |

### Symptom-Based Diagnosis

```
Symptom: No Power
├── Check power LED
│   ├── Off → Check input power
│   │   ├── Input OK → Check fuse
│   │   └── No input → Check supply/cable
│   └── On → Check MCU LED
│       ├── Off → MCU not running, see E005
│       └── Blinking → System booting
```

## Maintenance Schedule

### Standard Schedule

| Interval | Task | Procedure |
|----------|------|-----------|
| Weekly | Visual inspection | Check LEDs, listen for noise |
| Monthly | Clean enclosure | Wipe down, check vents |
| Quarterly | Check connections | Tighten terminals, inspect cables |
| Annually | Full inspection | Complete system test |
| 2 Years | Replace fans | Preemptive replacement |

### Environmental Factors

| Environment | Additional Requirements |
|-------------|------------------------|
| Dusty | Monthly filter cleaning |
| High humidity | Quarterly desiccant check |
| High vibration | Monthly fastener check |
| Corrosive | Monthly coating inspection |

## Auto-Generated Content

From project data, automatically generates:
- Technical specifications
- Wiring diagrams
- Error code reference
- LED state table
- Part numbers for FRUs
- Safety warnings from design

## Output Formats

| Format | Use Case |
|--------|----------|
| PDF | Print, distribution |
| HTML | Online hosting |
| Markdown | Version control |
| DITA | Enterprise CMS |

## API Endpoint

```
POST /ee-design/api/v1/documentation/service-manual
```

Request body:
```json
{
  "projectId": "proj-uuid",
  "format": "pdf",
  "options": {
    "includeSchematic": true,
    "includePCBImages": true,
    "language": "en",
    "companyName": "Acme Corp",
    "logo": "https://example.com/logo.png"
  }
}
```

## Integration

Part of EE Design Partner Phase 10 (Field Support). Generates from:
- Schematic and layout data
- Firmware error definitions
- Test procedures
- Design specifications

Output supports:
- Field service technicians
- End-user documentation
- Regulatory compliance
