---
name: vendor-quote
displayName: "Vendor Quote"
description: "Get instant PCB manufacturing quotes from major vendors"
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
  - /vendor-quote
  - /quote
  - /pcb-quote

capabilities:
  - name: quote
    description: Get quotes from vendors
    parameters:
      - name: layout_id
        type: string
        required: true
        description: PCB layout identifier
      - name: vendors
        type: array
        required: false
        description: "Vendors: pcbway, jlcpcb, oshpark, eurocircuits"
      - name: quantity
        type: array
        required: true
        description: Quantities to quote

  - name: compare
    description: Compare quotes across vendors
    parameters:
      - name: quotes
        type: array
        required: true
        description: Quote IDs to compare

  - name: order
    description: Place PCB order
    parameters:
      - name: quote_id
        type: string
        required: true
        description: Quote to order
      - name: shipping
        type: string
        required: false
        description: "Shipping method: standard, express, dhl"
---

# Vendor Quote

## Overview

Get instant PCB fabrication and assembly quotes from major manufacturers. Compare pricing, lead times, and capabilities across vendors.

## Usage

### Get Quotes

```bash
# Single vendor quote
/vendor-quote --layout=pcb-001 --vendor=jlcpcb --qty="5,10,50"

# Multi-vendor comparison
/vendor-quote --layout=pcb-001 --vendors=jlcpcb,pcbway,oshpark --qty=10

# With assembly
/vendor-quote --layout=pcb-001 --assembly --vendor=jlcpcb --qty=10
```

### Compare Quotes

```bash
# Compare received quotes
/vendor-quote compare --quotes=quote-001,quote-002,quote-003

# Factor in shipping
/vendor-quote compare --quotes=all --include-shipping --destination=US
```

### Place Order

```bash
# Order from quote
/vendor-quote order --quote=quote-001 --shipping=dhl

# With payment method
/vendor-quote order --quote=quote-001 --payment=corporate-card
```

## Supported Vendors

| Vendor | Region | Capabilities |
|--------|--------|--------------|
| **JLCPCB** | China | PCB, PCBA, stencil |
| **PCBWay** | China | PCB, PCBA, 3D print |
| **OSHPark** | USA | PCB (purple!) |
| **Eurocircuits** | Europe | PCB, PCBA |
| **Advanced Circuits** | USA | PCB, quick-turn |

## Quote Factors

### PCB Fabrication

| Factor | Impact |
|--------|--------|
| Layer Count | +$$ per layer |
| Board Size | Area-based pricing |
| Thickness | Special = +$$ |
| Min Feature | HDI = +$$$ |
| Material | Rogers = +$$$ |
| Surface Finish | ENIG > HASL |
| Color | Non-green = +$ |
| Lead Time | Rush = +$$$ |

### Assembly

| Factor | Impact |
|--------|--------|
| Component Count | Per placement |
| Package Types | BGA/QFN = +$ |
| Sides | 2-sided = +$$ |
| Special Parts | Hand solder = +$$ |
| Testing | +$$ |

## Quote Example

```
Quote Comparison - pcb-001 (10 qty)
=====================================

Vendor      | PCB     | Assembly | Parts   | Ship    | Total   | Lead
------------|---------|----------|---------|---------|---------|------
JLCPCB      | $8.50   | $35.00   | $125.40 | $18.00  | $186.90 | 7 days
PCBWay      | $12.30  | $28.50   | $125.40 | $15.00  | $181.20 | 8 days
OSHPark     | $48.00  | N/A      | -       | FREE    | $48.00  | 12 days

Recommended: PCBWay (best total cost with assembly)
```

## PCB Specifications Auto-Detection

From layout, automatically extracts:
- Board dimensions
- Layer count
- Minimum trace/space
- Minimum drill size
- Via types (blind/buried)
- Copper weight requirements
- Impedance control needs

## Assembly Detection

From BOM and placement:
- Component count
- Unique part count
- Package types
- Component sides
- Fine-pitch parts
- BGA count

## API Endpoint

```
POST /ee-design/api/v1/manufacturing/quote
```

Request body:
```json
{
  "layoutId": "pcb-uuid",
  "vendors": ["jlcpcb", "pcbway"],
  "quantities": [5, 10, 50],
  "options": {
    "assembly": true,
    "bomId": "bom-uuid",
    "pnpId": "pnp-uuid",
    "shippingDestination": "US"
  }
}
```

Response:
```json
{
  "quotes": [
    {
      "id": "quote-001",
      "vendor": "jlcpcb",
      "quantity": 10,
      "pcbCost": 8.50,
      "assemblyCost": 35.00,
      "partsCost": 125.40,
      "shippingCost": 18.00,
      "totalCost": 186.90,
      "leadTime": 7,
      "validUntil": "2025-02-01"
    }
  ]
}
```

## Integration

Part of EE Design Partner Phase 6 (Manufacturing). Requires:
- Gerber files (auto-generated)
- BOM with MPN
- Pick and place file
- DFM validation pass
