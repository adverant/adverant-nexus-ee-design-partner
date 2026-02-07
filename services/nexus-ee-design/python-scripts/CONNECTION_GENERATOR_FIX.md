# Connection Generator Architecture Fix - v3.2

## Problem Summary

The Connection Generator Agent v3.1 had a fundamental layer confusion bug that violated the separation of concerns between logical connections and physical routing.

### The Bug

**Lines 474-562 (Old Prompt)**: Asked LLM for physical wire routing data including:
- `start_point`: Physical coordinates (x, y)
- `end_point`: Physical coordinates (x, y)
- `waypoints`: List of physical waypoint coordinates
- Bend angles and wire crossing constraints
- IPC-2221 spacing validation

**Lines 666-707 (Old Conversion)**: Attempted to convert physical wire data back to logical connections:
```python
connections.append(GeneratedConnection(
    from_ref=wire.get("from_ref", "U1"),  # ❌ PLACEHOLDER - not from LLM
    from_pin=wire.get("from_pin", "1"),   # ❌ PLACEHOLDER - not from LLM
    to_ref=wire.get("to_ref", "U2"),      # ❌ PLACEHOLDER - not from LLM
    to_pin=wire.get("to_pin", "1"),       # ❌ PLACEHOLDER - not from LLM
    ...
))
```

The code was using placeholder values (`"U1"`, `"1"`, etc.) because the LLM was never asked for component references or pin names - only physical coordinates.

## Root Cause

### Incorrect Architecture (v3.1)
```
Connection Generator → Physical Routing (start/end coordinates, waypoints, angles)
                       ❌ WRONG LAYER
```

### Correct Architecture (v3.2)
```
Connection Generator → Logical Connections (from_ref, from_pin, to_ref, to_pin, net_name)
Wire Router         → Physical Routing (coordinates, waypoints, IPC-2221 compliance)
```

## The Fix

### 1. Updated LLM Prompt (Lines 474-562)

**Old Prompt** (v3.1):
```json
{
  "wires": [
    {
      "net_name": "VCC",
      "start_point": {"x": 100.0, "y": 50.0},  // ❌ Physical routing
      "end_point": {"x": 150.0, "y": 50.0},    // ❌ Physical routing
      "waypoints": [{"x": 125.0, "y": 60.0}],  // ❌ Physical routing
      "width": 0.8,
      "signal_type": "power"
    }
  ]
}
```

**New Prompt** (v3.2):
```json
{
  "connections": [
    {
      "from_ref": "U1",                  // ✅ Logical connection
      "from_pin": "VCC",                 // ✅ Logical connection
      "to_ref": "C1",                    // ✅ Logical connection
      "to_pin": "1",                     // ✅ Logical connection
      "net_name": "VCC",
      "signal_type": "power",
      "current_amps": 2.0,               // ✅ For downstream IPC-2221 compliance
      "voltage_volts": 3.3               // ✅ For downstream IPC-2221 compliance
    }
  ]
}
```

### 2. Updated Conversion Function (Lines 666-707)

**Old Code** (v3.1):
```python
def _convert_wires_to_connections(self, wires: List[Dict]) -> List[GeneratedConnection]:
    """Convert validated wire data to GeneratedConnection format."""
    # ❌ Tries to extract logical connections from physical wire data
    connections.append(GeneratedConnection(
        from_ref=wire.get("from_ref", "U1"),  # PLACEHOLDER
        from_pin=wire.get("from_pin", "1"),   # PLACEHOLDER
        ...
    ))
```

**New Code** (v3.2):
```python
def _convert_wires_to_connections(self, connections_data: List[Dict]) -> List[GeneratedConnection]:
    """Convert validated logical connection data to GeneratedConnection format."""
    # ✅ Extracts actual values from LLM response
    from_ref = conn_data.get("from_ref")  # ACTUAL VALUE
    from_pin = conn_data.get("from_pin")  # ACTUAL VALUE
    to_ref = conn_data.get("to_ref")      # ACTUAL VALUE
    to_pin = conn_data.get("to_pin")      # ACTUAL VALUE

    # ✅ Validates all required fields are present
    if not all([from_ref, from_pin, to_ref, to_pin]):
        logger.warning(f"Skipping connection with missing fields...")
        continue
```

### 3. Replaced Wire Validation with Logical Validation

**Old Validation** (v3.1):
```python
# ❌ Validated physical routing (spacing, angles, crossings)
report = self.validator.validate(
    wires_data,
    context.voltage_map,
    context.current_map,
    context.signal_types
)
```

**New Validation** (v3.2):
```python
# ✅ Validates logical connections
validation_errors = self._validate_logical_connections(
    connections_data,
    components,
    context
)
```

**Logical Validation Checks**:
1. All required fields present (from_ref, from_pin, to_ref, to_pin, net_name, signal_type)
2. Component references exist in BOM
3. Pin names exist on specified components
4. Current and voltage specifications present
5. Differential pairs are complete (both positive and negative signals)

### 4. Removed WireValidator Dependency

**Old Code** (v3.1):
```python
# ❌ Imported physical routing validator
from .wire_validator import WireValidator, format_validation_report

self.validator = WireValidator(rules_path)
```

**New Code** (v3.2):
```python
# ✅ No wire validator import
# Physical routing validation is done by WireRouter agent
```

## Impact

### Before Fix (v3.1)
- ❌ LLM generated physical wire routing data
- ❌ Placeholder values used for logical connections
- ❌ Runtime failures when downstream components expected real component references
- ❌ Layer confusion: Connection Generator doing Wire Router's job

### After Fix (v3.2)
- ✅ LLM generates logical pin-to-pin connections
- ✅ Real component references and pin names from LLM
- ✅ Proper separation of concerns
- ✅ Downstream components receive valid connection data
- ✅ IPC-2221 metadata (current, voltage) included for wire router

## Testing Recommendations

1. **Unit Test**: Verify `_validate_logical_connections()` catches:
   - Missing component references
   - Invalid pin names
   - Missing current/voltage specifications
   - Incomplete differential pairs

2. **Integration Test**: Verify generated connections have:
   - Non-placeholder component references
   - Valid pin names from component pin lists
   - Correct net names
   - Appropriate current/voltage values

3. **End-to-End Test**: Verify connection data flows to SchematicAssembler:
   - `Connection` dataclass receives real values
   - Wire router can resolve pin positions from component references
   - No runtime errors from placeholder values

## Files Modified

1. **connection_generator_agent.py**:
   - Updated file header docstring (v3.1 → v3.2)
   - Updated class docstring
   - Updated `__init__()` docstring
   - Updated `generate_connections()` docstring
   - Updated `_build_ipc_2221_prompt()` to request logical connections
   - Updated `_call_llm_for_wires()` to extract "connections" array
   - Updated `_convert_wires_to_connections()` to handle logical connection data
   - Replaced `_validate_wires()` with `_validate_logical_connections()`
   - Updated all log messages
   - Removed WireValidator import

## Verification Checklist

- [x] LLM prompt asks for logical connections (from_ref, from_pin, to_ref, to_pin)
- [x] LLM prompt does NOT ask for physical routing (start_point, end_point, waypoints)
- [x] Conversion function extracts actual values (not placeholders)
- [x] Validation checks component/pin existence
- [x] Validation checks differential pairs
- [x] WireValidator removed (physical routing is separate layer)
- [x] Docstrings updated to reflect architecture
- [x] Log messages updated to reflect architecture

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────┐
│ Connection Generator Agent (v3.2)                           │
│ ─────────────────────────────────────────────────────────── │
│ Input:  BOM, Design Intent, Component Pins                  │
│ Output: Logical Connections (from_ref/pin → to_ref/pin)     │
│                                                              │
│ Generates:                                                   │
│ - Component references (U1, U2, C1, etc.)                   │
│ - Pin names (VCC, PA5, MOSI, etc.)                          │
│ - Net names (VCC, GND, SPI_MOSI, etc.)                      │
│ - Signal types (power, digital, high_speed, etc.)           │
│ - Current/voltage specs for IPC-2221 compliance             │
└─────────────────────────────────────────────────────────────┘
                            ↓
                  Logical Connections
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ Wire Router Agent (separate module)                         │
│ ─────────────────────────────────────────────────────────── │
│ Input:  Logical Connections, Component Positions            │
│ Output: Physical Wire Routes (coordinates, waypoints)       │
│                                                              │
│ Generates:                                                   │
│ - Start/end coordinates (x, y)                              │
│ - Waypoints for routing                                     │
│ - Bend angles (IPC-2221 compliant)                          │
│ - Wire spacing (voltage-based)                              │
│ - Wire width (current-based)                                │
└─────────────────────────────────────────────────────────────┘
```

## Summary

The fix corrects a fundamental architecture violation where the Connection Generator was attempting to generate physical wire routing instead of logical connections. With v3.2, the Connection Generator now properly:

1. Generates **logical connections** (component references + pin names)
2. Includes **IPC-2221 metadata** (current, voltage) for downstream routing
3. Validates **logical correctness** (not physical routing)
4. Leaves **physical routing** to the Wire Router agent

This ensures proper separation of concerns and eliminates placeholder values that would cause runtime failures.
