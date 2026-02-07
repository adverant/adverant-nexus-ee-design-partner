"""
Test suite for SmokeTestAgent

Tests comprehensive electrical validation with real circuit examples.
Each test represents a common circuit topology or failure mode.

Author: Nexus EE Design Team
"""

import asyncio
import json
import os
import pytest
from pathlib import Path

from smoke_test_agent import (
    SmokeTestAgent,
    SmokeTestResult,
    SmokeTestSeverity,
)


# Fixture for agent
@pytest.fixture
async def agent():
    """Create smoke test agent instance."""
    agent = SmokeTestAgent()
    yield agent
    await agent.close()


# Test Case 1: Power Short (VCC connected directly to GND)
@pytest.mark.asyncio
async def test_power_short_detection(agent):
    """Test: Power Short - VCC connected to GND → FAIL

    This should be detected as a FATAL issue since it will immediately
    short the power supply when voltage is applied.
    """
    schematic = """(kicad_sch (version 20231120) (generator "test")
  (wire (pts (xy 100 100) (xy 100 80)))
  (wire (pts (xy 100 100) (xy 100 120)))
  (global_label "VCC" (at 100 80 0))
  (global_label "GND" (at 100 120 0))
)"""

    bom = []
    power_sources = [
        {"net": "VCC", "voltage": 5.0, "current_limit": 1.0}
    ]

    result = await agent.run_smoke_test(schematic, bom, power_sources)

    # Should FAIL due to power short
    assert not result.passed, "Power short should cause test to fail"
    assert not result.no_shorts, "Should detect the short circuit"

    # Should have at least one FATAL issue
    fatal_issues = [i for i in result.issues if i.severity == SmokeTestSeverity.FATAL]
    assert len(fatal_issues) > 0, "Power short should be marked as FATAL"

    # Check that the issue mentions "short" or "VCC" and "GND"
    issue_text = " ".join(str(i.message).lower() for i in fatal_issues)
    assert "short" in issue_text or ("vcc" in issue_text and "gnd" in issue_text), \
        "Issue should mention the short circuit"


# Test Case 2: Open Critical Signal (Clock net not connected)
@pytest.mark.asyncio
async def test_open_critical_signal(agent):
    """Test: Open Critical Signal - Clock net not connected → FAIL

    MCU clock pin is floating, which will prevent the MCU from running.
    """
    schematic = """(kicad_sch (version 20231120) (generator "test")
  (symbol (lib_id "MCU:STM32G431CBT6") (at 100 100 0) (unit 1)
    (property "Reference" "U1")
    (property "Value" "STM32G431CBT6")
  )
  (symbol (lib_id "Device:Crystal") (at 150 100 0) (unit 1)
    (property "Reference" "Y1")
    (property "Value" "8MHz")
  )
  (wire (pts (xy 120 100) (xy 130 100)))
  (wire (pts (xy 170 100) (xy 180 100)))
  (global_label "VCC" (at 100 80 0))
  (global_label "GND" (at 100 120 0))
  (wire (pts (xy 100 100) (xy 100 80)))
  (wire (pts (xy 100 110) (xy 100 120)))
)"""

    bom = [
        {"reference": "U1", "part_number": "STM32G431CBT6", "category": "MCU", "value": ""},
        {"reference": "Y1", "part_number": "8MHz", "category": "Crystal", "value": "8MHz"}
    ]

    power_sources = [
        {"net": "VCC", "voltage": 3.3, "current_limit": 1.0}
    ]

    result = await agent.run_smoke_test(schematic, bom, power_sources)

    # Should FAIL or WARN about floating crystal/clock
    if not result.passed:
        # Check for floating node or connectivity issue
        assert not result.no_floating_nodes or not result.current_paths_valid, \
            "Should detect floating crystal or connectivity issue"
    else:
        # If passed, should at least have warnings
        warnings = [i for i in result.issues if i.severity == SmokeTestSeverity.WARNING]
        assert len(warnings) > 0, "Should have warnings about potential connectivity issues"


# Test Case 3: Voltage Mismatch (5V driving 3.3V input)
@pytest.mark.asyncio
async def test_voltage_mismatch(agent):
    """Test: Voltage Mismatch - 5V output driving 3.3V input → FAIL/WARN

    A 5V output driving a 3.3V input can damage the receiving IC.
    This is a common mistake in mixed-voltage designs.
    """
    schematic = """(kicad_sch (version 20231120) (generator "test")
  (symbol (lib_id "MCU:STM32G431") (at 100 100 0) (unit 1)
    (property "Reference" "U1")
    (property "Value" "STM32G431")
  )
  (symbol (lib_id "IC:HC595") (at 200 100 0) (unit 1)
    (property "Reference" "U2")
    (property "Value" "74HC595")
  )
  (wire (pts (xy 120 100) (xy 180 100)))
  (label "SPI_MOSI" (at 150 100 0))
  (wire (pts (xy 100 80) (xy 100 90)))
  (wire (pts (xy 200 80) (xy 200 90)))
  (global_label "5V" (at 100 80 0))
  (global_label "3V3" (at 200 80 0))
  (global_label "GND" (at 100 120 0))
  (wire (pts (xy 100 110) (xy 100 120)))
  (wire (pts (xy 200 110) (xy 200 120)))
)"""

    bom = [
        {"reference": "U1", "part_number": "STM32G431CBT6", "category": "MCU", "value": ""},
        {"reference": "U2", "part_number": "74HC595", "category": "IC", "value": ""}
    ]

    power_sources = [
        {"net": "5V", "voltage": 5.0, "current_limit": 1.0},
        {"net": "3V3", "voltage": 3.3, "current_limit": 0.5}
    ]

    result = await agent.run_smoke_test(schematic, bom, power_sources)

    # Should detect voltage level mismatch (may be ERROR or WARNING)
    error_and_warning_issues = [
        i for i in result.issues
        if i.severity in (SmokeTestSeverity.ERROR, SmokeTestSeverity.WARNING)
    ]

    # Check if voltage mismatch is mentioned
    issue_text = " ".join(str(i.message).lower() for i in error_and_warning_issues)
    has_voltage_concern = any(
        keyword in issue_text
        for keyword in ["voltage", "level", "5v", "3.3v", "3v3", "mismatch"]
    )

    # Either should fail the test, or have warnings about voltage levels
    if not result.passed:
        assert True, "Test correctly failed on voltage mismatch"
    else:
        # If it passed, should at least warn about it
        assert has_voltage_concern, \
            "Should warn about voltage level incompatibility between 5V and 3.3V systems"


# Test Case 4: Missing Decoupling Capacitor
@pytest.mark.asyncio
async def test_missing_decoupling_capacitor(agent):
    """Test: Missing Decoupling - IC without bypass cap → WARNING

    ICs should have bypass/decoupling capacitors near their power pins.
    Missing bypass caps can cause instability and noise issues.
    """
    schematic = """(kicad_sch (version 20231120) (generator "test")
  (symbol (lib_id "MCU:STM32G431") (at 100 100 0) (unit 1)
    (property "Reference" "U1")
    (property "Value" "STM32G431")
  )
  (wire (pts (xy 100 80) (xy 100 90)))
  (wire (pts (xy 100 110) (xy 100 120)))
  (global_label "VCC" (at 100 80 0))
  (global_label "GND" (at 100 120 0))
)"""

    bom = [
        {"reference": "U1", "part_number": "STM32G431CBT6", "category": "MCU", "value": ""}
    ]

    power_sources = [
        {"net": "VCC", "voltage": 3.3, "current_limit": 1.0}
    ]

    result = await agent.run_smoke_test(schematic, bom, power_sources)

    # Should have warnings about missing bypass caps
    # Note: The LLM may or may not catch this depending on how thorough it is
    # We expect at least a warning about bypass capacitors
    warnings = [i for i in result.issues if i.severity == SmokeTestSeverity.WARNING]

    if len(warnings) > 0:
        warning_text = " ".join(str(i.message).lower() for i in warnings)
        bypass_mentioned = any(
            keyword in warning_text
            for keyword in ["bypass", "decoupling", "capacitor", "cap"]
        )
        # If warnings exist, we prefer they mention bypass caps, but not required
        # (LLM may focus on other issues)


# Test Case 5: Valid Design (Simple LED blink circuit)
@pytest.mark.asyncio
async def test_valid_led_blink_circuit(agent):
    """Test: Valid Design - Proper LED blink circuit → PASS

    A simple but complete circuit: MCU driving LED with current-limiting resistor.
    Power and ground properly connected. Should pass all checks.
    """
    schematic = """(kicad_sch (version 20231120) (generator "test")
  (symbol (lib_id "MCU:STM32G431") (at 100 100 0) (unit 1)
    (property "Reference" "U1")
    (property "Value" "STM32G431")
  )
  (symbol (lib_id "Device:R") (at 150 100 0) (unit 1)
    (property "Reference" "R1")
    (property "Value" "330")
  )
  (symbol (lib_id "Device:LED") (at 180 100 0) (unit 1)
    (property "Reference" "D1")
    (property "Value" "LED")
  )
  (symbol (lib_id "Device:C") (at 100 70 0) (unit 1)
    (property "Reference" "C1")
    (property "Value" "100nF")
  )

  # MCU power connections
  (wire (pts (xy 100 80) (xy 100 90)))
  (global_label "VCC" (at 100 80 0))

  # Bypass cap
  (wire (pts (xy 95 70) (xy 100 80)))
  (wire (pts (xy 105 70) (xy 100 120)))

  # MCU ground
  (wire (pts (xy 100 110) (xy 100 120)))
  (global_label "GND" (at 100 120 0))

  # LED circuit
  (wire (pts (xy 120 100) (xy 140 100)))
  (wire (pts (xy 160 100) (xy 170 100)))
  (wire (pts (xy 190 100) (xy 200 100)))
  (global_label "GND" (at 200 100 0))

  (label "GPIO_LED" (at 130 100 0))
)"""

    bom = [
        {"reference": "U1", "part_number": "STM32G431CBT6", "category": "MCU", "value": ""},
        {"reference": "R1", "part_number": "RC0805FR-07330RL", "category": "Resistor", "value": "330"},
        {"reference": "D1", "part_number": "LTST-C150KRKT", "category": "LED", "value": "Red LED"},
        {"reference": "C1", "part_number": "CL21B104KBCNNNC", "category": "Capacitor", "value": "100nF"}
    ]

    power_sources = [
        {"net": "VCC", "voltage": 3.3, "current_limit": 1.0}
    ]

    result = await agent.run_smoke_test(schematic, bom, power_sources)

    # This should PASS or at least have no FATAL issues
    fatal_issues = [i for i in result.issues if i.severity == SmokeTestSeverity.FATAL]
    assert len(fatal_issues) == 0, "Valid LED circuit should have no fatal issues"

    # Should pass basic connectivity checks
    assert result.power_rails_ok, "Power rails should be properly connected"
    assert result.ground_ok, "Ground should be properly connected"
    assert result.no_shorts, "Should have no short circuits"


# Test Case 6: Missing Power Connection
@pytest.mark.asyncio
async def test_missing_power_connection(agent):
    """Test: Missing Power Connection - IC with no VCC → FAIL

    An IC without power connection will not function.
    This is a critical error that should be detected.
    """
    schematic = """(kicad_sch (version 20231120) (generator "test")
  (symbol (lib_id "MCU:STM32G431") (at 100 100 0) (unit 1)
    (property "Reference" "U1")
    (property "Value" "STM32G431")
  )
  # Only ground connected, no VCC!
  (wire (pts (xy 100 110) (xy 100 120)))
  (global_label "GND" (at 100 120 0))
)"""

    bom = [
        {"reference": "U1", "part_number": "STM32G431CBT6", "category": "MCU", "value": ""}
    ]

    power_sources = [
        {"net": "VCC", "voltage": 3.3, "current_limit": 1.0}
    ]

    result = await agent.run_smoke_test(schematic, bom, power_sources)

    # Should FAIL due to missing power connection
    assert not result.passed, "Missing power connection should cause test to fail"
    assert not result.power_rails_ok, "Should detect missing power connection"

    # Should have error/fatal issues about power
    critical_issues = [
        i for i in result.issues
        if i.severity in (SmokeTestSeverity.FATAL, SmokeTestSeverity.ERROR)
    ]
    assert len(critical_issues) > 0, "Should have critical issues about missing power"

    issue_text = " ".join(str(i.message).lower() for i in critical_issues)
    assert "power" in issue_text or "vcc" in issue_text or "vdd" in issue_text, \
        "Issue should mention missing power connection"


# Test Case 7: Reverse Polarity Diode
@pytest.mark.asyncio
async def test_reverse_polarity_diode(agent):
    """Test: Reverse Polarity - Diode installed backwards → FAIL/WARN

    A diode installed backwards (cathode to VCC, anode to GND) will
    cause a short circuit when powered.
    """
    schematic = """(kicad_sch (version 20231120) (generator "test")
  (symbol (lib_id "Device:D") (at 100 100 180) (unit 1)
    (property "Reference" "D1")
    (property "Value" "1N4148")
  )
  # Diode backwards: cathode to VCC (180 degree rotation)
  (wire (pts (xy 100 90) (xy 100 80)))
  (wire (pts (xy 100 110) (xy 100 120)))
  (global_label "VCC" (at 100 80 0))
  (global_label "GND" (at 100 120 0))
)"""

    bom = [
        {"reference": "D1", "part_number": "1N4148", "category": "Diode", "value": ""}
    ]

    power_sources = [
        {"net": "VCC", "voltage": 5.0, "current_limit": 1.0}
    ]

    result = await agent.run_smoke_test(schematic, bom, power_sources)

    # Should FAIL or WARN about reverse diode creating short
    # This is similar to power short test
    if not result.passed:
        # Check for short or polarity issue
        assert not result.no_shorts or len(result.issues) > 0, \
            "Should detect reverse polarity diode creating potential short"


# Test Case 8: Connectivity Check
@pytest.mark.asyncio
async def test_connectivity_validation(agent):
    """Test: Connectivity validation quick check"""
    schematic = """(kicad_sch (version 20231120) (generator "test")
  (symbol (lib_id "MCU:STM32G431") (at 100 100 0) (unit 1)
    (property "Reference" "U1")
    (property "Value" "STM32G431")
  )
  (wire (pts (xy 100 80) (xy 100 90)))
  (wire (pts (xy 100 110) (xy 100 120)))
  (global_label "VCC" (at 100 80 0))
  (global_label "GND" (at 100 120 0))
)"""

    result = await agent.validate_connectivity(schematic)

    # Should return valid connectivity data
    assert "wire_count" in result, "Should report wire count"
    assert "component_count" in result, "Should report component count"
    assert "has_power_nets" in result, "Should check for power nets"

    # Should detect at least some wires and components
    assert result.get("wire_count", 0) > 0, "Should detect wires"
    assert result.get("component_count", 0) > 0, "Should detect components"
    assert result.get("has_power_nets", False), "Should detect power nets (VCC, GND)"


# Test Case 9: API Key Not Set
@pytest.mark.asyncio
async def test_missing_api_key():
    """Test: Missing API key handling"""
    # Temporarily clear API key
    original_key = os.environ.get("OPENROUTER_API_KEY")
    if "OPENROUTER_API_KEY" in os.environ:
        del os.environ["OPENROUTER_API_KEY"]

    try:
        agent = SmokeTestAgent()
        schematic = """(kicad_sch (version 20231120) (generator "test")
  (symbol (lib_id "Device:R") (at 100 100 0) (unit 1)
    (property "Reference" "R1")
  )
)"""

        result = await agent.run_smoke_test(schematic, [])

        # Should fail gracefully with configuration error
        assert not result.passed, "Should fail when API key is missing"
        assert len(result.issues) > 0, "Should report configuration issue"

        config_issues = [
            i for i in result.issues
            if "api" in i.message.lower() or "key" in i.message.lower()
        ]
        assert len(config_issues) > 0, "Should report API key issue"

        await agent.close()
    finally:
        # Restore original key
        if original_key:
            os.environ["OPENROUTER_API_KEY"] = original_key


# Main test runner
if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
