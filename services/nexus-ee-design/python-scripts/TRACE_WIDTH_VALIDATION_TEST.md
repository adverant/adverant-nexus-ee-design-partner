# IPC-2221 Trace Width Validation Test Cases

## Overview

This document provides comprehensive test cases for the IPC-2221 trace width validation implemented in `enhanced_wire_router.py`. The validation ensures that PCB traces can safely carry their specified current without exceeding a 10°C temperature rise.

## IPC-2221 Formula

The implementation uses the standard IPC-2221 formula for external layers with 1oz copper:

```
A = (I / (k * ΔT^b))^(1/c)
```

Where:
- **I** = Current in amperes
- **k** = 0.048 (constant for external layers)
- **ΔT** = 10.0°C (temperature rise)
- **b** = 0.44 (empirical constant)
- **c** = 0.725 (empirical constant)
- **A** = Cross-sectional area in square mils

Then convert to width:
```
Width (mils) = A / Copper_Thickness
Width (mm) = Width (mils) * 0.0254
```

For 1oz copper: Thickness = 1.378 mils

## Test Cases

### Test Case 1: Low Current Signal (0.1A)

**Input:**
- Net Name: `DATA_BUS[0]`
- Current: 0.1A
- Connection Type: `signal`

**Expected Calculation:**
```
A = (0.1 / (0.048 * 10^0.44))^(1/0.725)
A ≈ 3.94 sq mils
Width = 3.94 / 1.378 ≈ 2.86 mils
Width ≈ 0.073 mm
```

**Expected Result:**
✅ **PASS** - Default signal trace (0.25mm) exceeds minimum requirement (0.073mm)

---

### Test Case 2: Moderate Current (0.5A)

**Input:**
- Net Name: `LED_DRIVER`
- Current: 0.5A
- Connection Type: `signal`

**Expected Calculation:**
```
A = (0.5 / (0.048 * 10^0.44))^(1/0.725)
A ≈ 6.27 sq mils
Width = 6.27 / 1.378 ≈ 4.55 mils
Width ≈ 0.115 mm
```

**Expected Result:**
✅ **PASS** - Default signal trace (0.25mm) exceeds minimum requirement (0.115mm)

**Note:** For 0.5A, the formula gives 0.115mm but the conservative table recommends 0.20mm. The default 0.25mm trace provides ample margin.

---

### Test Case 3: Power Rail - 2A (VCC)

**Input:**
- Net Name: `VCC` or `+5V`
- Current: 2.0A
- Connection Type: `power`

**Expected Calculation:**
```
A = (2.0 / (0.048 * 10^0.44))^(1/0.725)
A ≈ 42.42 sq mils
Width = 42.42 / 1.378 ≈ 30.78 mils
Width ≈ 0.781 mm
```

**Expected Result:**
✅ **PASS** - Default power trace (0.80mm) meets minimum requirement (0.781mm)
**Note:** The 0.80mm default trace width is appropriate for 2A power rails, providing a small safety margin over the calculated 0.781mm requirement.

---

### Test Case 4: Ground Rail - 3A (GND)

**Input:**
- Net Name: `GND`
- Current: 3.0A
- Connection Type: `ground`

**Expected Calculation:**
```
A = (3.0 / (0.048 * 10^0.44))^(1/0.725)
A ≈ 58.88 sq mils
Width = 58.88 / 1.378 ≈ 42.73 mils
Width ≈ 1.367 mm
```

**Expected Result:**
❌ **FAIL** - Default ground trace (0.80mm) is below minimum (1.367mm)
**Violation Message:**
```
TRACE WIDTH VIOLATION: Net 'GND' carries 3.00A but trace width 0.800mm
is less than required 1.367mm per IPC-2221 (1oz copper, 10°C rise).
Increase trace width to 1.367mm or use heavier copper.
```

**Recommended Fix:** Use 1.4mm trace or 2oz copper (would require ~0.68mm)

---

### Test Case 5: High Current Motor Driver - 5A

**Input:**
- Net Name: `MOTOR_POWER`
- Current: 5.0A
- Connection Type: `power`

**Expected Calculation:**
```
A = (5.0 / (0.048 * 10^0.44))^(1/0.725)
A ≈ 108.88 sq mils
Width = 108.88 / 1.378 ≈ 79.02 mils
Width ≈ 2.765 mm
```

**Expected Result:**
❌ **FAIL** - Default power trace (0.80mm) is far below minimum (2.765mm)
**Violation Message:**
```
TRACE WIDTH VIOLATION: Net 'MOTOR_POWER' carries 5.00A but trace width 0.800mm
is less than required 2.765mm per IPC-2221 (1oz copper, 10°C rise).
Increase trace width to 2.765mm or use heavier copper.
```

**Recommended Fix:** Use 2.8mm trace width or 2oz/3oz copper

---

### Test Case 6: Extreme Current - 10A

**Input:**
- Net Name: `BATTERY_MAIN`
- Current: 10.0A
- Connection Type: `power`

**Expected Calculation:**
```
A = (10.0 / (0.048 * 10^0.44))^(1/0.725)
A ≈ 283.23 sq mils
Width = 283.23 / 1.378 ≈ 205.58 mils
Width ≈ 7.194 mm
```

**Expected Result:**
❌ **CRITICAL FAIL** - Default power trace (0.80mm) is dangerously undersized
**Violation Message:**
```
TRACE WIDTH VIOLATION: Net 'BATTERY_MAIN' carries 10.00A but trace width 0.800mm
is less than required 7.194mm per IPC-2221 (1oz copper, 10°C rise).
Increase trace width to 7.194mm or use heavier copper.
```

**Recommended Fix:** Use 7.2mm trace or 4oz copper (would require ~1.80mm width)

**Note:** At very high currents (>5A), consider using heavier copper (2oz, 3oz, 4oz) rather than extremely wide traces.

---

### Test Case 7: Net Name Pattern Detection

**Inputs:**
Test that net name patterns correctly infer current:

| Net Name | Inferred Type | Expected Current |
|----------|--------------|------------------|
| `VCC_3V3` | Power | 2.0A |
| `VDD_5V` | Power | 2.0A |
| `+12V` | Power | 2.0A |
| `GND` | Ground | 3.0A |
| `AGND` | Ground | 3.0A |
| `VSS` | Ground | 3.0A |
| `DATA[0]` | Signal | 0.1A |
| `SPI_MOSI` | Signal | 0.1A |

**Expected Result:** All nets are correctly classified and assigned appropriate currents

---

### Test Case 8: Very Low Current (< 0.05A) - Should Skip

**Input:**
- Net Name: `SENSOR_INPUT`
- Current: 0.01A
- Connection Type: `signal`

**Expected Result:**
✅ **SKIP** - Validation skipped for currents below 50mA (no violation reported)

---

## IPC-2221 Reference Table

For quick verification, here's the expected minimum width for common currents:

| Current | Formula (1oz Cu) | Conservative Table | Notes |
|---------|------------------|-------------------|-------|
| 0.1A | 0.013mm | 0.07mm | Table adds safety margin |
| 0.5A | 0.115mm | 0.20mm | Table adds ~75% margin |
| 1.0A | 0.300mm | 0.40mm | Table adds ~33% margin |
| 2.0A | 0.781mm | 0.80mm | Very close match |
| 3.0A | 1.367mm | 1.50mm | Table adds ~10% margin |
| 5.0A | 2.765mm | 2.50mm | Formula exceeds table |
| 10.0A | 7.194mm | 5.00mm | Formula exceeds table |

**Note:** The implementation uses the IPC-2221 formula which gives mathematically precise results. The conservative table values from `ipc_2221_rules.yaml` include safety margins that vary by current level. For 2A (common power rail current), the formula (0.781mm) closely matches the table (0.80mm).

## Implementation Validation

To validate the implementation, run the wire router with these test connections and verify:

1. ✅ Formula produces correct calculations (match reference table within 5%)
2. ✅ Violations are reported for undersized traces
3. ✅ Net name patterns correctly infer power/ground currents
4. ✅ Very low current nets are skipped
5. ✅ Violation messages include all required information:
   - Net name
   - Actual current
   - Actual width
   - Required width
   - IPC-2221 reference
   - Remediation guidance

## Python Test Code

```python
def test_trace_width_calculations():
    """Test IPC-2221 trace width calculations."""
    from enhanced_wire_router import EnhancedWireRouter

    router = EnhancedWireRouter()

    # Test Case: 2A power rail
    test_connections = [
        {
            "net_name": "VCC",
            "connection_type": "power",
            "from_ref": "U1",
            "from_pin": "VCC",
            "to_ref": "C1",
            "to_pin": "1"
        }
    ]

    test_wires = [
        WireSegment(
            start=(0, 0),
            end=(10, 0),
            net_name="VCC",
            route_type=RouteType.POWER
        )
    ]

    violations = router._check_trace_widths(test_wires, test_connections)

    # Should report violation: 2A requires 0.89mm, default is 0.80mm
    assert len(violations) == 1
    assert "VCC" in violations[0]
    assert "2.00A" in violations[0]
    assert "0.890mm" in violations[0]

    print("✅ Test passed: 2A power rail correctly flagged as violation")

if __name__ == "__main__":
    test_trace_width_calculations()
```

## Known Limitations

1. **Schematic-level validation**: This implementation validates at the schematic level where actual PCB trace widths are not yet defined. It uses assumed defaults (0.25mm signal, 0.80mm power).

2. **No heavier copper calculation**: The implementation assumes 1oz copper. For 2oz/3oz/4oz copper, the required width would be proportionally smaller, but this is not yet implemented.

3. **No temperature rise options**: Fixed at 10°C rise. Some designs may allow 20°C or require only 5°C.

4. **Internal layer adjustment**: The formula uses external layer constants (k=0.048). Internal layers require different constants (k=0.024) and would need wider traces.

5. **No net-specific current metadata**: Currently infers current from net name patterns and connection types. Future enhancement would use explicit current specifications from power rail definitions in the ideation context.

## Future Enhancements

1. **Read power rail current from ideation context**: Extract `current_max` from `PowerRail` objects
2. **Support multiple copper weights**: Allow user to specify 2oz/3oz copper
3. **Internal layer support**: Different formula constants for internal layers
4. **Temperature rise options**: Allow configurable ΔT (5°C, 10°C, 20°C)
5. **Per-segment width**: When PCB trace data is available, validate actual widths
6. **Current density warnings**: Flag high current density even if width is technically acceptable

---

**Author:** Nexus EE Design Team
**Last Updated:** 2026-02-07
**IPC-2221 Standard Reference:** IPC-2221 Generic Standard on Printed Board Design, Table 6-3
