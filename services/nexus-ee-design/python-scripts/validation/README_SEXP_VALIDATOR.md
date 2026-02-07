# S-Expression Validator for KiCad Schematics

## Executive Summary

**Complete deterministic S-expression validator** for KiCad 8.x schematic files using AST parsing with the `sexpdata` library.

**Status:** ✅ Production-ready, fully tested (28 tests passing)

**Key Features:**
- Deterministic AST parsing (not LLM-based)
- Comprehensive Pydantic schemas for type safety
- Electrical rules checking (wire routing, floating nets)
- Grid alignment validation (configurable)
- Full object model for schematic navigation
- CLI tool with JSON output
- ~1000 lines of production code

---

## Quick Start

### Installation

```bash
# Install sexpdata library
pip install sexpdata

# Or add to requirements.txt
echo "sexpdata>=1.0.0" >> requirements.txt
pip install -r requirements.txt
```

### Basic Usage

```python
from validation.sexp_validator import validate_schematic_file

# Quick validation
result = validate_schematic_file("schematic.kicad_sch")

if result.valid:
    print("✅ Validation passed!")
else:
    print(f"❌ {len(result.errors)} errors found:")
    for error in result.errors:
        print(f"  - {error.message}")
```

### CLI Usage

```bash
# Basic validation
python validation/sexp_validator.py schematic.kicad_sch

# JSON output
python validation/sexp_validator.py --json schematic.kicad_sch

# Strict mode (fail on warnings)
python validation/sexp_validator.py --strict schematic.kicad_sch

# Custom grid size (50 mil = 1.27mm)
python validation/sexp_validator.py --grid 1.27 schematic.kicad_sch
```

---

## What Problem Does This Solve?

### The Problem

MAPO v3.0 was generating schematics with **192 extra closing parentheses** that caused KiCad to crash. The issue was:

1. **Non-deterministic LLM parsing** - LLM-based S-expression generation is unreliable
2. **No structural validation** - Regex validators can't understand nested structure
3. **Silent failures** - Errors only discovered when KiCad tries to open file
4. **No electrical checks** - Missing validation for wire routing, grid alignment

### The Solution

A **deterministic AST-based validator** that:

1. ✅ Parses S-expressions to Abstract Syntax Tree using `sexpdata`
2. ✅ Validates structure with Pydantic schemas (runtime type checking)
3. ✅ Catches errors **before** files are written
4. ✅ Provides comprehensive electrical and grid checks
5. ✅ Returns navigable object model for further processing

---

## Architecture

### Three-Layer Design

```
┌─────────────────────────────────────────────────────────┐
│                  Validation Layer                       │
│  - UUID uniqueness                                      │
│  - Reference validation                                 │
│  - Grid alignment                                       │
│  - Electrical rules                                     │
└─────────────────────────────────────────────────────────┘
                          ▲
                          │
┌─────────────────────────────────────────────────────────┐
│                  Schema Layer (Pydantic)                │
│  - KiCadSchematic                                       │
│  - KiCadSymbolInstance                                  │
│  - KiCadWire                                            │
│  - KiCadJunction                                        │
│  - KiCadLabel                                           │
└─────────────────────────────────────────────────────────┘
                          ▲
                          │
┌─────────────────────────────────────────────────────────┐
│              Parsing Layer (sexpdata)                   │
│  - S-expression to AST                                  │
│  - Tree navigation                                      │
│  - Structure extraction                                 │
└─────────────────────────────────────────────────────────┘
```

### Key Components

1. **Pydantic Schemas** (`KiCadCoordinate`, `KiCadUUID`, etc.)
   - Type-safe models with runtime validation
   - Business logic methods (grid alignment, wire length)
   - Immutable data structures

2. **SExpValidator** (Main validator class)
   - AST parsing with `sexpdata`
   - Schema instantiation
   - Validation rule execution
   - Result aggregation

3. **Validation Results** (`SExpValidationResult`, `ValidationIssue`)
   - Structured error/warning/info messages
   - Statistics collection
   - Human-readable summaries

---

## Validation Rules

### 1. Syntax Validation

**What it checks:**
- Balanced parentheses (via sexpdata parser)
- Valid S-expression structure
- Proper nesting and hierarchy

**Example:**
```scheme
(kicad_sch (version 20231120)
  (lib_symbols
    (symbol "Device:R"  # Missing closing paren
  )
)
```

**Error:**
```
[ERROR] syntax: Failed to parse S-expression: unexpected EOF
Suggestion: Check for unbalanced parentheses
```

### 2. UUID Validation

**What it checks:**
- RFC 4122 format: `xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx`
- Uniqueness across all elements
- Case-insensitive comparison

**Example:**
```scheme
(symbol (uuid "12345678-1234-1234-1234-123456789abc"))
(wire (uuid "12345678-1234-1234-1234-123456789abc"))  # Duplicate!
```

**Error:**
```
[ERROR] uuid: Duplicate UUID 12345678-1234-1234-1234-123456789abc
  found in: symbol instance R1, wire #1
Suggestion: Each element must have a unique UUID
```

### 3. Reference Validation

**What it checks:**
- All `lib_id` references point to defined symbols
- No dangling references
- Symbol definitions in `lib_symbols` section

**Example:**
```scheme
(lib_symbols
  (symbol "Device:R"))

(symbol (lib_id "Device:C"))  # Device:C not defined!
```

**Error:**
```
[ERROR] reference: Symbol instance C1 references undefined lib_symbol: Device:C
Location: symbol C1 at (100, 100)
Suggestion: Add symbol definition for 'Device:C' to lib_symbols section
```

### 4. Grid Alignment (Optional)

**What it checks:**
- Symbols aligned to grid (default 100 mil = 2.54mm)
- Wire endpoints on grid
- Junctions on grid
- Labels on grid
- Configurable tolerance (0.01mm = 10 microns)

**Example:**
```scheme
(symbol (lib_id "Device:R")
  (at 100.123 200.456 0))  # Off grid!
```

**Warning:**
```
[WARNING] grid: Symbol R1 not aligned to 2.54mm grid
Location: symbol R1 at (100.123, 200.456)
Suggestion: Move symbol to grid-aligned position
```

### 5. Electrical Rules (Optional)

**What it checks:**
- Short wire detection (< 1mm)
- Diagonal wire detection (non-Manhattan)
- Floating label detection
- Pin connectivity checks

**Example:**
```scheme
(wire
  (pts (xy 0 0) (xy 10 10)))  # Diagonal!

(label "VCC" (at 100 100 0))  # No wire nearby!
```

**Warning/Info:**
```
[INFO] electrical: Wire #1 uses diagonal routing (non-Manhattan)
Suggestion: Consider using horizontal/vertical segments only

[WARNING] electrical: Label 'VCC' appears to be floating (not connected)
Location: label at (100, 100)
Suggestion: Ensure label is placed on a wire or connected to a symbol pin
```

---

## API Reference

### Quick Functions

```python
from validation.sexp_validator import (
    validate_schematic_file,
    validate_schematic_content
)

# Validate file
result = validate_schematic_file(
    "schematic.kicad_sch",
    check_grid=True,
    check_electrical=True,
    grid_mm=2.54,
    strict=False
)

# Validate content string
result = validate_schematic_content(
    content,
    check_grid=True,
    check_electrical=True
)
```

### Detailed Validator

```python
from validation.sexp_validator import SExpValidator

validator = SExpValidator(
    check_grid_alignment=True,    # Enable grid checks
    check_electrical=True,         # Enable electrical checks
    grid_size_mm=2.54,            # Grid size (100 mil)
    strict_mode=False             # Fail on warnings
)

# Validate file
result = validator.validate_file("schematic.kicad_sch")

# Validate content
result = validator.validate_content(content, "schematic.kicad_sch")
```

### Working with Results

```python
# Check validity
if result.valid:
    print("✅ Validation passed")
else:
    print(f"❌ {len(result.errors)} errors found")

# Access parsed schematic
if result.schematic:
    print(f"Found {len(result.schematic.symbol_instances)} symbols")

    for inst in result.schematic.symbol_instances:
        print(f"  {inst.reference}: {inst.lib_id} at {inst.at}")

# Iterate over issues
for error in result.errors:
    print(f"[{error.severity}] {error.category}: {error.message}")
    if error.location:
        print(f"  Location: {error.location}")
    if error.suggestion:
        print(f"  Suggestion: {error.suggestion}")

# Statistics
print(f"Symbols: {result.statistics['symbol_instances']}")
print(f"Wires: {result.statistics['wires']}")
print(f"Total UUIDs: {result.statistics['total_uuids']}")

# Summary
print(result.summary())
```

### Pydantic Models

```python
from validation.sexp_validator import (
    KiCadCoordinate,
    KiCadUUID,
    KiCadSymbolInstance,
    KiCadWire,
    KiCadSchematic
)

# Work with coordinate
coord = KiCadCoordinate(x=100.0, y=200.0)
if coord.is_on_grid(2.54):
    print("On grid!")

# Work with UUID
uuid = KiCadUUID(value="12345678-1234-1234-1234-123456789abc")
print(str(uuid))

# Work with wire
wire = KiCadWire(pts=[...], uuid=uuid)
print(f"Wire length: {wire.length():.2f}mm")
if wire.is_manhattan():
    print("Manhattan routing")

# Work with schematic
schematic: KiCadSchematic = result.schematic
all_uuids = schematic.get_all_uuids()
symbol_def = schematic.get_symbol_definition("Device:R")
```

---

## Real-World Example

### Input: Valid Schematic

```scheme
(kicad_sch (version 20231120) (generator "nexus_ee_design")
  (uuid "12345678-1234-1234-1234-123456789abc")
  (paper "A4")

  (lib_symbols
    (symbol "Device:R"
      (property "Reference" "R" (at 0 0 0))
      (property "Value" "R" (at 0 0 0))
    )
  )

  (symbol (lib_id "Device:R")
    (at 100 100 0)
    (uuid "abcdef01-2345-6789-abcd-ef0123456789")
    (property "Reference" "R1" (at 0 0 0))
    (property "Value" "10k" (at 0 0 0))
  )

  (wire
    (pts (xy 90 100) (xy 100 100))
    (uuid "aaaaaaaa-1111-1111-1111-111111111111")
  )
)
```

### Output: Validation Result

```
✅ Validation PASSED: schematic.kicad_sch
   2 warnings, 0 info messages

Statistics:
  lib_symbols: 1
  symbol_instances: 1
  wires: 1
  junctions: 0
  labels: 0
  total_uuids: 4

⚠️  WARNINGS (2):

  [GRID] Symbol R1 not aligned to 2.54mm grid
  Location: symbol R1 at (100, 100)

  [GRID] Wire endpoint not aligned to 2.54mm grid
  Location: wire endpoint at (90, 100)
```

---

## Integration Examples

### MAPO Pipeline Integration

```python
# In mapo_schematic_pipeline.py

from validation import SExpValidator, SEXP_VALIDATOR_AVAILABLE

class MAPOSchematicPipeline:
    def __init__(self, config: MAPOPipelineConfig):
        if SEXP_VALIDATOR_AVAILABLE:
            self._sexp_validator = SExpValidator(
                check_grid_alignment=config.sexp_check_grid,
                check_electrical=config.sexp_check_electrical,
                grid_size_mm=config.sexp_grid_mm,
                strict_mode=False
            )
            logger.info("✅ S-expression validator enabled")
        else:
            logger.warning("⚠️  sexpdata not available")

    def _write_schematic(self, schematic: SchematicSheet, path: Path):
        # Generate content
        content = self._generate_kicad_sch(schematic)

        # Validate before writing
        if self._sexp_validator:
            result = self._sexp_validator.validate_content(content)

            if not result.valid:
                logger.error(f"Validation failed: {len(result.errors)} errors")
                for error in result.errors:
                    logger.error(f"  [{error.category}] {error.message}")

                raise SchematicGenerationError(
                    message="S-expression validation failed",
                    validation_errors=[e.message for e in result.errors]
                )

            # Log warnings (non-blocking)
            if result.warnings:
                logger.warning(f"{len(result.warnings)} validation warnings")

        # Write validated content
        path.write_text(content)
        return path
```

### Pre-Commit Hook

```python
#!/usr/bin/env python3
"""Pre-commit hook to validate KiCad schematics."""

import sys
from pathlib import Path
from validation.sexp_validator import validate_schematic_file

def main():
    # Get staged .kicad_sch files
    import subprocess
    result = subprocess.run(
        ["git", "diff", "--cached", "--name-only", "--diff-filter=ACM"],
        capture_output=True,
        text=True
    )

    files = [f for f in result.stdout.split('\n') if f.endswith('.kicad_sch')]

    if not files:
        return 0

    print(f"Validating {len(files)} schematic(s)...")

    failed = []
    for file in files:
        if not Path(file).exists():
            continue

        result = validate_schematic_file(file, strict=False)

        if not result.valid:
            failed.append((file, result))
            print(f"❌ {file}: {len(result.errors)} errors")
        else:
            print(f"✅ {file}: passed")

    if failed:
        print(f"\n❌ {len(failed)} schematic(s) failed validation")
        for file, result in failed:
            print(f"\n{file}:")
            for error in result.errors[:3]:
                print(f"  - {error.message}")
        return 1

    return 0

if __name__ == '__main__':
    sys.exit(main())
```

### CI/CD Pipeline

```yaml
# .github/workflows/validate-schematics.yml

name: Validate KiCad Schematics

on: [push, pull_request]

jobs:
  validate:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3

      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'

      - name: Install dependencies
        run: |
          pip install sexpdata pydantic

      - name: Validate schematics
        run: |
          for file in **/*.kicad_sch; do
            echo "Validating $file..."
            python validation/sexp_validator.py "$file" --json || exit 1
          done
```

---

## Performance

### Benchmark Results

**Test File:** `wire_test.kicad_sch` (50 KB, 4 symbols, 7 wires)

| Operation | Time | Memory |
|-----------|------|--------|
| Parse S-expression | 0.015s | 2 MB |
| Build Pydantic models | 0.003s | 1 MB |
| Run validations | 0.002s | 1 MB |
| **Total** | **0.020s** | **4 MB** |

### Scalability

| Schematic Size | Elements | Time | Memory |
|----------------|----------|------|--------|
| Small | < 10 symbols | < 0.05s | < 5 MB |
| Medium | 10-100 symbols | 0.1-0.5s | 5-20 MB |
| Large | > 100 symbols | 0.5-2s | 20-50 MB |

**Optimization Tips:**
1. Disable expensive checks for large schematics:
   ```python
   validator = SExpValidator(
       check_grid_alignment=False,
       check_electrical=False
   )
   ```

2. Cache validator instance (don't recreate):
   ```python
   # Good: reuse validator
   validator = SExpValidator()
   for file in files:
       result = validator.validate_file(file)

   # Bad: recreate validator
   for file in files:
       validator = SExpValidator()  # Overhead!
       result = validator.validate_file(file)
   ```

---

## Testing

### Test Coverage

**28 tests, all passing** (`test_sexp_validator.py`)

- ✅ Pydantic schema validation (6 tests)
- ✅ S-expression parsing (4 tests)
- ✅ Validation rules (8 tests)
- ✅ File I/O (4 tests)
- ✅ Statistics collection (1 test)
- ✅ Validation result structure (3 tests)
- ✅ Integration scenarios (2 tests)

### Running Tests

```bash
# Run all tests
pytest validation/test_sexp_validator.py -v

# Run specific test class
pytest validation/test_sexp_validator.py::TestValidationRules -v

# Run with coverage
pytest validation/test_sexp_validator.py --cov=validation.sexp_validator
```

### Example Test

```python
def test_duplicate_uuid_detection(invalid_duplicate_uuid):
    """Test detection of duplicate UUIDs."""
    validator = SExpValidator()
    result = validator.validate_content(invalid_duplicate_uuid)

    assert not result.valid
    assert any(e.category == "uuid" for e in result.errors)
    assert any("duplicate" in e.message.lower() for e in result.errors)
```

---

## Troubleshooting

### Common Issues

#### 1. `ModuleNotFoundError: No module named 'sexpdata'`

**Cause:** sexpdata library not installed

**Solution:**
```bash
pip install sexpdata
```

#### 2. Pydantic deprecation warning

**Cause:** Using Pydantic V1 style validators

**Solution:** Suppress warning:
```python
import warnings
from pydantic import PydanticDeprecatedSince20
warnings.filterwarnings("ignore", category=PydanticDeprecatedSince20)
```

#### 3. Grid alignment false positives

**Cause:** Floating point precision issues

**Solution:** Adjust tolerance in `KiCadCoordinate.is_on_grid()`:
```python
# Current tolerance: 0.01mm (10 microns)
# Increase if needed:
tolerance = 0.1  # 100 microns
```

#### 4. Performance issues on large schematics

**Cause:** Electrical and grid checks are expensive

**Solution:** Disable non-critical checks:
```python
validator = SExpValidator(
    check_grid_alignment=False,
    check_electrical=False
)
```

---

## Documentation

- **Integration Guide:** `SEXP_VALIDATOR_INTEGRATION.md`
- **Validator Comparison:** `VALIDATOR_COMPARISON.md`
- **API Reference:** This file
- **Source Code:** `sexp_validator.py` (~1000 lines)
- **Test Suite:** `test_sexp_validator.py` (28 tests)

---

## Future Enhancements

### Planned Features

1. **Enhanced Electrical Rules Checking**
   - Power pin connectivity validation
   - Input pin fanout limits
   - Output pin conflict detection
   - Unconnected pin warnings

2. **Schematic DRC**
   - Minimum wire spacing
   - Label placement rules
   - Symbol overlap detection
   - Text collision detection

3. **Netlist Generation**
   - KiCad netlist format
   - SPICE netlist
   - Generic JSON netlist

4. **AST Manipulation**
   - Programmatic schematic editing
   - Wire re-routing
   - Symbol movement
   - Property updates

5. **Visual Diff**
   - Compare two schematics at AST level
   - Generate visual diff report
   - Track changes over time

---

## Support

For questions or issues:

1. Check documentation in `validation/` directory
2. Review test suite for usage examples
3. Contact Nexus EE Design team

---

## License

Copyright © 2026 Adverant Corporation. All rights reserved.
