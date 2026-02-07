# Smoke Test Agent

**Version:** MAPO v3.1
**Status:** ✅ Production-ready (LLM-based implementation)
**Model:** Claude Opus 4.6 via OpenRouter

## Overview

The Smoke Test Agent validates that generated schematics will not "smoke" (burn/fail) when power is applied. It performs comprehensive electrical analysis to catch critical issues before fabrication.

## Implementation Approach

**Current: LLM-First Analysis (Option C)**

The agent uses Claude Opus 4.6 to semantically analyze circuit topology and identify electrical issues. This approach is more flexible than rule-based or SPICE simulation because the LLM understands:

- Circuit intent and topology
- Component relationships
- Standard design practices
- KiCad schematic semantics (global labels, implicit connections)

**Alternatives considered:**
- Option A: SPICE simulation (using PySpice/ngspice) - More precise but brittle
- Option B: Rule-based analysis (using electrical_rules.yaml) - Faster but less intelligent

## Validation Checks

The agent performs these electrical validations:

1. **Power Rail Check**: VCC/VDD/3V3/5V connected to all ICs that need power
2. **Ground Check**: GND/VSS connected to all components
3. **Short Circuit Detection**: No direct power-to-ground paths
4. **Floating Node Detection**: Critical pins (power, enable, reset) connected
5. **Current Path Validation**: Current can flow from power through loads to ground
6. **Bypass Capacitor Check**: ICs have decoupling caps near power pins
7. **Polarity Check**: Polarized components oriented correctly

## Usage

### Basic Usage

```python
from agents.smoke_test import SmokeTestAgent, SmokeTestResult

agent = SmokeTestAgent()

# Prepare inputs
kicad_sch_content = Path("circuit.kicad_sch").read_text()
bom_items = [
    {"reference": "U1", "part_number": "STM32G431CBT6", "category": "MCU", "value": ""},
    {"reference": "C1", "part_number": "CL21B104KBCNNNC", "category": "Capacitor", "value": "100nF"}
]
power_sources = [
    {"net": "VCC", "voltage": 3.3, "current_limit": 1.0}
]

# Run smoke test
result: SmokeTestResult = await agent.run_smoke_test(
    kicad_sch_content=kicad_sch_content,
    bom_items=bom_items,
    power_sources=power_sources
)

# Check results
if result.passed:
    print("✅ Smoke test PASSED - circuit is safe to power")
else:
    print(f"❌ Smoke test FAILED with {len(result.issues)} issues")
    for issue in result.issues:
        print(f"  [{issue.severity.value}] {issue.message}")

await agent.close()
```

### Integration with MAPO Pipeline

The agent is integrated via `SmokeTestValidator` wrapper:

```python
from mapos_v2_1_schematic.validation.smoke_test_validator import SmokeTestValidator

validator = SmokeTestValidator()
validation_result = await validator.validate(
    state=schematic_state,
    validation_context=ideation_context  # Optional
)

if validation_result.passed:
    print(f"Fitness score: {validation_result.combined_fitness:.2f}")
else:
    print(f"Fatal issues: {validation_result.fatal_issues}")
```

## Data Structures

### SmokeTestResult

```python
@dataclass
class SmokeTestResult:
    passed: bool                      # Overall pass/fail
    power_rails_ok: bool              # Power connectivity OK
    ground_ok: bool                   # Ground connectivity OK
    no_shorts: bool                   # No short circuits
    no_floating_nodes: bool           # No floating critical pins
    power_dissipation_ok: bool        # Power dissipation within limits
    current_paths_valid: bool         # Current can flow properly
    issues: List[SmokeTestIssue]      # Detailed issues
    llm_analysis: Dict[str, Any]      # Raw LLM response
```

### SmokeTestIssue

```python
@dataclass
class SmokeTestIssue:
    severity: SmokeTestSeverity       # FATAL | ERROR | WARNING | INFO
    test_name: str                    # Name of test that failed
    message: str                      # Human-readable description
    component: Optional[str]          # Affected component reference
    net: Optional[str]                # Affected net name
    recommendation: Optional[str]     # How to fix
```

### SmokeTestSeverity

```python
class SmokeTestSeverity(Enum):
    FATAL = "fatal"        # Circuit will definitely smoke/fail
    ERROR = "error"        # Circuit likely won't work correctly
    WARNING = "warning"    # Potential issue, needs review
    INFO = "info"          # Advisory information
```

## Configuration

### Environment Variables

```bash
# Required
OPENROUTER_API_KEY=your_openrouter_api_key_here

# Optional (defaults shown)
OPENROUTER_MODEL=anthropic/claude-opus-4-6
```

Get an API key from: https://openrouter.ai/keys

### Error Handling

The agent fails gracefully when:

- **API key missing**: Returns `SmokeTestResult` with FATAL issue explaining configuration error
- **LLM returns invalid JSON**: Returns failure result with diagnostic information
- **Network error**: Returns failure result with connectivity error
- **Timeout**: Configurable via `httpx.AsyncClient(timeout=120.0)`

## Testing

### Run Test Suite

```bash
# Install dependencies
pip install pytest pytest-asyncio

# Run all tests
cd services/nexus-ee-design/python-scripts/agents/smoke_test
pytest test_smoke_test.py -v

# Run specific test
pytest test_smoke_test.py::test_power_short_detection -v -s
```

### Test Coverage

The test suite includes 9 comprehensive test cases:

1. **Power Short Detection** - VCC-to-GND short → FAIL
2. **Open Critical Signal** - Floating clock → FAIL
3. **Voltage Mismatch** - 5V driving 3.3V input → FAIL/WARN
4. **Missing Decoupling** - IC without bypass cap → WARNING
5. **Valid LED Circuit** - Proper design → PASS
6. **Missing Power** - IC without VCC connection → FAIL
7. **Reverse Polarity** - Backwards diode → FAIL/WARN
8. **Connectivity Validation** - Quick structural check
9. **Missing API Key** - Graceful error handling

## Performance

- **Average analysis time**: 5-15 seconds per schematic (depends on complexity)
- **Token usage**: ~2,000-4,000 tokens per analysis
- **Cost**: ~$0.02-$0.05 per validation (Opus 4.6 pricing)

## KiCad Semantic Understanding

The LLM understands these KiCad-specific semantics:

1. **Global Labels**: `(global_label "VCC" ...)` elements with same text are electrically connected
2. **Power Net Equivalents**: VCC=VDD=3V3=5V, GND=VSS=GROUND, VDDA=analog power
3. **Component Rotation**: `(at x y rotation)` affects pin positions
4. **Wire Connectivity**: `(wire (pts (xy x1 y1) (xy x2 y2)))` connects two points
5. **Symbol Instances**: `(symbol (lib_id "..."))` references library symbols

**Critical**: The agent does NOT require explicit wire connections if global labels are used. This prevents false positives for properly labeled power/ground connections.

## Future Enhancements

Potential improvements for v3.2:

1. **Hybrid SPICE Validation**: Run ngspice simulation for numerical verification of LLM analysis
2. **Component Database**: Load actual voltage/current ratings from part datasheets
3. **Thermal Analysis**: Calculate junction temperatures for power components
4. **EMC Checks**: Validate decoupling cap placement distances (<5mm from IC)
5. **Current Budget**: Sum all component currents vs. power supply capacity
6. **Inrush Current**: Check startup surge current
7. **Protection Circuits**: Verify ESD, overvoltage, reverse polarity protection

## Files

```
agents/smoke_test/
├── __init__.py              # Package exports
├── smoke_test_agent.py      # Main agent implementation (523 lines)
├── test_smoke_test.py       # Test suite (9 test cases)
└── README.md               # This file
```

## Integration Points

The agent is called from:

1. **`mapo_schematic_pipeline.py`** (line ~665): After wiring phase in main pipeline
2. **`smoke_test_validator.py`** (line ~154): MAPO v2.1 validation wrapper with fitness scoring
3. **`schematic_mapo_optimizer.py`**: Gaming AI mutation validation (future)
4. **`ralph_loop_orchestrator.py`**: RALPH quality scoring (future)

## Dependencies

```
httpx>=0.25.0        # Async HTTP client for OpenRouter API
```

No additional dependencies required (uses stdlib: `asyncio`, `json`, `logging`, `dataclasses`, `enum`, `pathlib`).

## License

Copyright © 2025 Adverant Technologies Inc.
Proprietary - All Rights Reserved

## Support

For issues or questions:
- Internal: Contact Nexus EE Design Team
- External: support@adverant.com
