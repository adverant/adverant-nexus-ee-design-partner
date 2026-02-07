# Connection Generator Agent v3.1 - IPC-2221 Compliance

**MAPO v3.1 Enhancement**: Strict IPC-2221 standards compliance with pre-validation.

## Overview

This version introduces comprehensive IPC-2221 validation for wire routing, ensuring:
- ✅ Voltage-based conductor spacing
- ✅ Current-based conductor width
- ✅ Bend angle constraints (>= 45°)
- ✅ Wire crossing minimization (< 10 target)
- ✅ High-speed signal isolation
- ✅ Differential pair matching

## Architecture

```
connection_generator_agent.py (v3.1)
    │
    ├─► ipc_2221_rules.yaml          # Standards database
    │
    ├─► wire_validator.py             # Pre-acceptance validation
    │   └─► 6 validation checks
    │
    └─► LLM (Claude Opus 4.6)
        └─► Enhanced prompts with IPC-2221 rules
        └─► Retry loop (max 5 attempts)
```

## Files

### 1. `ipc_2221_rules.yaml` (97 lines)

Complete IPC-2221 standards database:

```yaml
conductor_spacing:
  voltage_classes:
    - max_voltage: 15
      min_spacing: 0.13  # mm
    - max_voltage: 50
      min_spacing: 0.50
    # ... up to 500V+

conductor_width:
  current_capacity:
    - current_amps: 1.0
      min_width: 0.40  # mm
    # ... up to 10A

routing_rules:
  - id: no_acute_angles
    severity: error
  - id: minimize_crossings
    target: "< 10"
  # ... 6 total rules
```

### 2. `wire_validator.py` (484 lines)

IPC-2221 validation engine:

```python
class WireValidator:
    def validate(
        self,
        wires: List[Dict],
        voltage_map: Dict[str, float],
        current_map: Dict[str, float],
        signal_types: Dict[str, str]
    ) -> WireValidationReport:
        """
        6 validation checks:
        1. Conductor spacing (voltage-dependent)
        2. Bend angles (>= 45°)
        3. Wire crossings (target < 10)
        4. High-speed signal isolation
        5. Conductor width (current-dependent)
        6. Differential pair matching
        """
```

### 3. `connection_generator_agent.py` (907 lines)

Main agent with IPC-2221 compliance:

```python
class ConnectionGeneratorAgent:
    async def generate_connections(
        self,
        bom: List[Dict],
        design_intent: str,
        component_pins: Optional[Dict] = None,
    ) -> List[GeneratedConnection]:
        """
        Generate IPC-2221 compliant connections.

        Flow:
        1. Extract component info
        2. Build wire context (voltage/current/signal-types)
        3. Generate power connections (rule-based)
        4. Generate signal connections with validation (LLM)
           - Retry up to 5 times if validation fails
        5. Deduplicate and return
        """
```

### 4. `test_connection_generator.py` (351 lines)

Comprehensive test suite:

```python
class TestWireValidator:
    - test_spacing_violation_detection()
    - test_acute_angle_detection()
    - test_crossing_count()
    - test_high_speed_crossing_violation()
    - test_conductor_width_warning()

class TestConnectionGeneratorAgent:
    - test_voltage_inference_from_design_intent()
    - test_power_connection_generation()
    - test_ipc_2221_prompt_generation()

class TestLLMIntegration:
    - test_full_connection_generation()  # Requires API key
```

## Usage

### Basic Usage

```python
from connection_generator_agent import ConnectionGeneratorAgent

# Initialize agent
agent = ConnectionGeneratorAgent()

# Define BOM and design intent
bom = [
    {"part_number": "STM32G431", "category": "MCU", "reference": "U1"},
    {"part_number": "UCC21520", "category": "Gate_Driver", "reference": "U2"},
    {"part_number": "100nF", "category": "Capacitor", "reference": "C1", "value": "100nF"},
]

design_intent = "FOC ESC with 3-phase gate drivers, CAN communication, 3.3V logic"

# Generate IPC-2221 compliant connections
connections = await agent.generate_connections(bom, design_intent)

print(f"✅ Generated {len(connections)} IPC-2221 compliant connections")
```

### Running Tests

```bash
# Install dependencies
pip install -r requirements.txt

# Run tests (unit tests only)
cd services/nexus-ee-design/python-scripts/agents/connection_generator
pytest test_connection_generator.py -v

# Run integration tests (requires OPENROUTER_API_KEY)
export OPENROUTER_API_KEY="your-key-here"
pytest test_connection_generator.py -v -s
```

## Enhanced LLM Prompt

The core innovation is the IPC-2221 enhanced prompt:

```python
prompt = f"""
IPC-2221 MANDATORY REQUIREMENTS (MUST PASS VALIDATION):

1. CONDUCTOR SPACING:
   - Minimum {min_spacing}mm between ALL conductors (based on {max_voltage}V)

2. NO ACUTE ANGLES:
   - All bends MUST be >= 45 degrees

3. MINIMIZE WIRE CROSSINGS:
   - Target: < 10 total crossings
   - Current attempt: {attempt_number}/5

4. HIGH-SPEED SIGNAL ISOLATION:
   - Clock signals: NO crossings allowed
   - Differential pairs: Route together, equal length

5. CONDUCTOR WIDTH:
   - Power nets: >= 0.8mm for 2A
   - Signal nets: >= 0.25mm

6. DIFFERENTIAL PAIRS:
   - Maximum length mismatch: 5mm

OUTPUT FORMAT (JSON ONLY):
{{
  "wires": [
    {{
      "net_name": "VCC",
      "start_point": {{"x": 100.0, "y": 50.0}},
      "end_point": {{"x": 150.0, "y": 50.0}},
      "waypoints": [],
      "width": 0.8,
      "signal_type": "power"
    }}
  ],
  "crossings_count": 8,
  "validation_notes": "All spacing requirements met."
}}
"""
```

## Validation Retry Loop

```python
for attempt in range(1, MAX_WIRE_GENERATION_RETRIES + 1):
    logger.info(f"🔄 Wire generation attempt {attempt}/5")

    # Generate wires with LLM
    wires = await self._call_llm_for_wires(prompt)

    # Validate against IPC-2221
    report = self.validator.validate(wires, voltage_map, current_map, signal_types)

    if report.passed:
        logger.info(f"✅ Wire validation PASSED on attempt {attempt}")
        return self._convert_wires_to_connections(wires)

    # Retry with feedback
    logger.warning(f"❌ Wire validation FAILED: {len(report.violations)} violations")
    # (Add violation feedback to next prompt)

# All attempts failed
raise ValidationError(f"Failed after {MAX_WIRE_GENERATION_RETRIES} attempts")
```

## Configuration

Key constants in `connection_generator_agent.py`:

```python
# OpenRouter LLM
OPENROUTER_MODEL = "anthropic/claude-opus-4.6"

# Validation
MAX_WIRE_GENERATION_RETRIES = 5
TARGET_WIRE_CROSSINGS = 10
```

## Expected Improvements Over v3.0

| Metric                  | v3.0 (Baseline) | v3.1 (Target) | Improvement |
|-------------------------|-----------------|---------------|-------------|
| Wire crossings          | 30-50           | < 10          | 70-80% ↓    |
| IPC-2221 violations     | Unknown         | 0             | 100% ↓      |
| Acute angles            | Present         | 0             | 100% ↓      |
| Spacing violations      | Present         | 0             | 100% ↓      |
| High-speed crossings    | Present         | 0             | 100% ↓      |

## Known Limitations

### 1. Wire-to-Connection Mapping
- **Issue**: `_convert_wires_to_connections()` uses placeholder refs
- **Impact**: Real pin locations not mapped
- **Fix**: Requires symbol position → pin location resolution

### 2. LLM Performance
- **Issue**: LLM may struggle with complex routing
- **Impact**: May not achieve < 10 crossings
- **Fix**: Tune prompt, increase retries, add feedback

### 3. Crossing Detection
- **Issue**: Uses simplified line segment intersection
- **Impact**: May miss some crossings
- **Fix**: Implement full path intersection checking

### 4. Differential Pair Matching
- **Issue**: Basic length calculation
- **Impact**: Does not account for serpentine routing
- **Fix**: Implement sophisticated length matching

## Integration with MAPO Pipeline

This agent integrates into the MAPO v3.1 schematic pipeline:

```
1. BOM Generation
2. Component Sourcing
3. Symbol Resolution
   ↓
4. CONNECTION GENERATION ← v3.1 IPC-2221 VALIDATION
   ↓
5. Layout Optimization
6. Wire Routing
7. KiCad Output
8. Validation
```

## Dependencies

From `requirements.txt`:

```
httpx>=0.27.0          # OpenRouter API
PyYAML>=6.0.1          # IPC rules database
pytest>=7.0.0          # Testing
```

## Version History

- **v3.0**: LLM-based connection generation, no validation
- **v3.1**: IPC-2221 compliance with pre-validation (this version)

## Support

For issues or questions:
- Check test suite: `pytest test_connection_generator.py -v`
- Review validation reports: Look for `WireValidationReport` in logs
- Adjust retry limit: Increase `MAX_WIRE_GENERATION_RETRIES` if needed

## License

Copyright © 2025 Adverant (Nexus EE Design Team)
