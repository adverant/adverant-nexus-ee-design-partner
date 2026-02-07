# S-Expression Validator Quick Start

**5-Minute Guide to Using the New Validator**

---

## Installation

```bash
pip install sexpdata
```

---

## Quick Validation

### From Python

```python
from validation.sexp_validator import validate_schematic_file

# Validate a file
result = validate_schematic_file("schematic.kicad_sch")

if result.valid:
    print("✅ Valid!")
else:
    print(f"❌ {len(result.errors)} errors:")
    for error in result.errors:
        print(f"  - {error.message}")
```

### From Command Line

```bash
# Basic validation
python validation/sexp_validator.py schematic.kicad_sch

# Get JSON output
python validation/sexp_validator.py --json schematic.kicad_sch
```

---

## Integration into MAPO Pipeline

**Add to `mapo_schematic_pipeline.py`:**

```python
from validation.sexp_validator import validate_schematic_content

def _write_schematic(self, schematic, output_path):
    # Generate content
    content = self._generate_kicad_sch(schematic)

    # VALIDATE BEFORE WRITING
    result = validate_schematic_content(content)

    if not result.valid:
        raise SchematicGenerationError(
            validation_errors=[e.message for e in result.errors]
        )

    # Write if valid
    output_path.write_text(content)
    return output_path
```

---

## Common Patterns

### 1. Quick Check (No Extras)

```python
result = validate_schematic_file(
    "schematic.kicad_sch",
    check_grid=False,        # Skip grid checks
    check_electrical=False   # Skip electrical checks
)
```

### 2. Comprehensive Check

```python
result = validate_schematic_file(
    "schematic.kicad_sch",
    check_grid=True,         # Check grid alignment
    check_electrical=True,   # Check electrical rules
    grid_mm=2.54,           # 100 mil grid
    strict=False            # Don't fail on warnings
)
```

### 3. Strict Mode (Fail on Warnings)

```python
result = validate_schematic_file(
    "schematic.kicad_sch",
    strict=True  # Treat warnings as errors
)
```

---

## Accessing Parsed Schematic

```python
result = validate_schematic_file("schematic.kicad_sch")

if result.schematic:
    # Get statistics
    print(f"Symbols: {len(result.schematic.symbol_instances)}")
    print(f"Wires: {len(result.schematic.wires)}")

    # Navigate structure
    for inst in result.schematic.symbol_instances:
        print(f"{inst.reference}: {inst.lib_id} at {inst.at}")
```

---

## What Gets Checked?

### Always Checked (Errors)
- ✅ Balanced parentheses
- ✅ Valid S-expression syntax
- ✅ UUID format and uniqueness
- ✅ Symbol references (lib_id exists)

### Optional Checks (Warnings)
- ⚠️ Grid alignment (off by default)
- ⚠️ Wire routing (diagonal detection)
- ⚠️ Floating labels
- ⚠️ Short wires

---

## Configuration

```python
from validation.sexp_validator import SExpValidator

validator = SExpValidator(
    check_grid_alignment=True,   # Enable grid checks
    check_electrical=True,        # Enable electrical checks
    grid_size_mm=2.54,           # 100 mil grid
    strict_mode=False            # Don't fail on warnings
)

result = validator.validate_file("schematic.kicad_sch")
```

---

## Output Format

### Human-Readable

```
✅ Validation PASSED: schematic.kicad_sch
   2 warnings, 0 info messages

Statistics:
  symbol_instances: 4
  wires: 7

⚠️  WARNINGS (2):
  [GRID] Symbol R1 not aligned to grid
  [ELECTRICAL] Wire uses diagonal routing
```

### JSON

```json
{
  "valid": true,
  "file": "schematic.kicad_sch",
  "errors": [],
  "statistics": {
    "symbol_instances": 4,
    "wires": 7
  }
}
```

---

## Troubleshooting

### Error: `ModuleNotFoundError: No module named 'sexpdata'`

**Solution:**
```bash
pip install sexpdata
```

### Warning: Pydantic deprecation

**Solution:** Ignore it (validator works fine) or:
```python
import warnings
from pydantic import PydanticDeprecatedSince20
warnings.filterwarnings("ignore", category=PydanticDeprecatedSince20)
```

### Performance issues on large files

**Solution:** Disable expensive checks:
```python
result = validate_schematic_file(
    "large.kicad_sch",
    check_grid=False,
    check_electrical=False
)
```

---

## When to Use?

### Use AST Validator When:
- ✅ Generating schematics (validate before writing)
- ✅ Need comprehensive checks (grid, electrical)
- ✅ Want object model (navigate structure)
- ✅ Building on top (netlist generation, editing)

### Use Regex Validator When:
- ✅ Need maximum speed
- ✅ Can't install sexpdata
- ✅ Only need basic syntax checks
- ✅ Already integrated and working

---

## Documentation

- **Full Guide:** `README_SEXP_VALIDATOR.md`
- **Integration:** `SEXP_VALIDATOR_INTEGRATION.md`
- **Comparison:** `VALIDATOR_COMPARISON.md`
- **Implementation:** `IMPLEMENTATION_REPORT.md`

---

## Support

Questions? Check:
1. Documentation in `validation/` directory
2. Test suite for usage examples
3. Source code comments

---

**That's it!** You're ready to use the S-expression validator.

For detailed usage, see `README_SEXP_VALIDATOR.md`.
