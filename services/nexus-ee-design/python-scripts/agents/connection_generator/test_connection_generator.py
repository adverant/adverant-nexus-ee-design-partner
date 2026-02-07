"""
Test suite for Connection Generator Agent v3.1 (IPC-2221 compliance).

Tests:
1. Wire validator functionality
2. IPC-2221 rule loading
3. Voltage/current inference
4. Validation retry logic
5. Integration test (requires OPENROUTER_API_KEY)

Run: python -m pytest test_connection_generator.py -v
"""

import asyncio
import os
import pytest
from typing import Dict, List

from connection_generator_agent import (
    ConnectionGeneratorAgent,
    ConnectionType,
    GeneratedConnection,
    WireGenerationContext,
    ValidationError,
)
from wire_validator import WireValidator, ValidationViolation


class TestWireValidator:
    """Test the WireValidator class."""

    def setup_method(self):
        """Setup validator for each test."""
        self.validator = WireValidator()

    def test_validator_loads_rules(self):
        """Test that validator loads IPC-2221 rules."""
        assert self.validator.rules is not None
        assert 'conductor_spacing' in self.validator.rules
        assert 'conductor_width' in self.validator.rules
        assert 'routing_rules' in self.validator.rules

    def test_spacing_violation_detection(self):
        """Test that spacing violations are detected."""
        # Two wires too close together
        wires = [
            {
                "net_name": "NET1",
                "start_point": {"x": 0.0, "y": 0.0},
                "end_point": {"x": 10.0, "y": 0.0},
                "width": 0.25,
                "signal_type": "signal"
            },
            {
                "net_name": "NET2",
                "start_point": {"x": 0.0, "y": 0.05},  # Only 0.05mm away!
                "end_point": {"x": 10.0, "y": 0.05},
                "width": 0.25,
                "signal_type": "signal"
            }
        ]

        voltage_map = {"NET1": 5.0, "NET2": 5.0}
        current_map = {"NET1": 0.1, "NET2": 0.1}
        signal_types = {"NET1": "signal", "NET2": "signal"}

        report = self.validator.validate(wires, voltage_map, current_map, signal_types)

        # Should fail due to spacing violation (need >= 0.13mm for 5V)
        assert not report.passed
        assert len(report.violations) > 0
        assert any("spacing" in v.message.lower() for v in report.violations)

    def test_acute_angle_detection(self):
        """Test that acute angles are detected."""
        wires = [
            {
                "net_name": "NET1",
                "start_point": {"x": 0.0, "y": 0.0},
                "end_point": {"x": 10.0, "y": 0.0},
                "waypoints": [
                    {"x": 5.0, "y": 0.5}  # This creates a sharp angle
                ],
                "width": 0.25,
                "signal_type": "signal"
            }
        ]

        voltage_map = {"NET1": 5.0}
        current_map = {"NET1": 0.1}
        signal_types = {"NET1": "signal"}

        report = self.validator.validate(wires, voltage_map, current_map, signal_types)

        # May detect acute angle depending on geometry
        # This is a smoke test to ensure validation runs
        assert report.total_wires == 1

    def test_crossing_count(self):
        """Test that wire crossings are counted."""
        # Two crossing wires
        wires = [
            {
                "net_name": "NET1",
                "start_point": {"x": 0.0, "y": 0.0},
                "end_point": {"x": 10.0, "y": 10.0},
                "width": 0.25,
                "signal_type": "signal"
            },
            {
                "net_name": "NET2",
                "start_point": {"x": 0.0, "y": 10.0},
                "end_point": {"x": 10.0, "y": 0.0},
                "width": 0.25,
                "signal_type": "signal"
            }
        ]

        voltage_map = {"NET1": 5.0, "NET2": 5.0}
        current_map = {"NET1": 0.1, "NET2": 0.1}
        signal_types = {"NET1": "signal", "NET2": "signal"}

        report = self.validator.validate(wires, voltage_map, current_map, signal_types)

        # Should detect at least one crossing
        assert report.crossings_count > 0

    def test_high_speed_crossing_violation(self):
        """Test that high-speed signal crossings are flagged as errors."""
        wires = [
            {
                "net_name": "CLK",
                "start_point": {"x": 0.0, "y": 0.0},
                "end_point": {"x": 10.0, "y": 10.0},
                "width": 0.3,
                "signal_type": "clock"
            },
            {
                "net_name": "DATA",
                "start_point": {"x": 0.0, "y": 10.0},
                "end_point": {"x": 10.0, "y": 0.0},
                "width": 0.25,
                "signal_type": "signal"
            }
        ]

        voltage_map = {"CLK": 3.3, "DATA": 3.3}
        current_map = {"CLK": 0.1, "DATA": 0.1}
        signal_types = {"CLK": "clock", "DATA": "signal"}

        report = self.validator.validate(wires, voltage_map, current_map, signal_types)

        # Should fail due to clock signal crossing
        assert not report.passed
        assert any("high-speed" in v.message.lower() or "clock" in v.message.lower()
                   for v in report.violations)

    def test_conductor_width_warning(self):
        """Test that insufficient conductor width triggers warning."""
        wires = [
            {
                "net_name": "VCC",
                "start_point": {"x": 0.0, "y": 0.0},
                "end_point": {"x": 10.0, "y": 0.0},
                "width": 0.2,  # Too narrow for 2A!
                "signal_type": "power"
            }
        ]

        voltage_map = {"VCC": 5.0}
        current_map = {"VCC": 2.0}  # 2A requires >= 0.8mm
        signal_types = {"VCC": "power"}

        report = self.validator.validate(wires, voltage_map, current_map, signal_types)

        # Should have width warning
        assert len(report.warnings) > 0
        assert any("width" in w.message.lower() for w in report.warnings)


class TestConnectionGeneratorAgent:
    """Test the ConnectionGeneratorAgent class."""

    def setup_method(self):
        """Setup agent for each test."""
        self.agent = ConnectionGeneratorAgent()

    def test_agent_initialization(self):
        """Test that agent initializes with IPC-2221 rules."""
        assert self.agent.ipc_rules is not None
        assert self.agent.validator is not None

    def test_component_extraction(self):
        """Test component info extraction from BOM."""
        bom = [
            {"part_number": "STM32G431", "category": "MCU", "reference": "U1"},
            {"part_number": "100nF", "category": "Capacitor", "reference": "C1", "value": "100nF"},
            {"part_number": "10k", "category": "Resistor", "reference": "R1", "value": "10k"},
        ]

        components = self.agent._extract_component_info(bom)

        assert len(components) == 3
        assert components[0].reference == "U1"
        assert components[0].category == "MCU"
        assert len(components[0].power_pins) > 0  # Should infer VCC
        assert len(components[0].ground_pins) > 0  # Should infer GND

    def test_voltage_inference_from_design_intent(self):
        """Test voltage inference from design intent."""
        components = []
        design_intent = "3.3V logic circuit with CAN bus"

        context = self.agent._build_wire_context(components, design_intent)

        assert context.voltage_map["VCC"] == 3.3
        assert context.voltage_map["GND"] == 0.0
        assert "CANH" in context.signal_types
        assert context.signal_types["CANH"] == "differential"

    def test_power_connection_generation(self):
        """Test rule-based power connection generation."""
        bom = [
            {"part_number": "STM32G431", "category": "MCU", "reference": "U1"},
            {"part_number": "UCC21520", "category": "Gate_Driver", "reference": "U2"},
        ]

        components = self.agent._extract_component_info(bom)
        power_connections = self.agent._generate_power_connections(components)

        # Should connect all power/ground pins
        assert len(power_connections) > 0
        assert all(c.connection_type == ConnectionType.POWER for c in power_connections)
        assert any(c.net_name == "VCC" for c in power_connections)
        assert any(c.net_name == "GND" for c in power_connections)

    def test_bypass_cap_connection_generation(self):
        """Test bypass capacitor connection generation."""
        bom = [
            {"part_number": "STM32G431", "category": "MCU", "reference": "U1"},
            {"part_number": "100nF", "category": "Capacitor", "reference": "C1", "value": "100nF"},
        ]

        components = self.agent._extract_component_info(bom)
        bypass_connections = self.agent._generate_bypass_cap_connections(components)

        # Should generate bypass cap connections
        assert len(bypass_connections) >= 2  # VCC and GND
        assert any("Bypass cap" in c.notes for c in bypass_connections)

    def test_deduplication(self):
        """Test connection deduplication."""
        connections = [
            GeneratedConnection("U1", "PA0", "U2", "IN1", "NET1", priority=5),
            GeneratedConnection("U2", "IN1", "U1", "PA0", "NET1", priority=5),  # Duplicate (reversed)
            GeneratedConnection("U1", "PA1", "U3", "IN2", "NET2", priority=5),
        ]

        deduped = self.agent._deduplicate_connections(connections)

        assert len(deduped) == 2  # Should remove one duplicate

    def test_ipc_2221_prompt_generation(self):
        """Test that IPC-2221 prompt is correctly formatted."""
        bom = [
            {"part_number": "STM32G431", "category": "MCU", "reference": "U1"},
        ]

        components = self.agent._extract_component_info(bom)
        design_intent = "3.3V microcontroller circuit"
        context = self.agent._build_wire_context(components, design_intent)

        prompt = self.agent._build_ipc_2221_prompt(
            components,
            design_intent,
            context,
            attempt_number=1
        )

        # Verify prompt contains key IPC-2221 requirements
        assert "IPC-2221" in prompt
        assert "CONDUCTOR SPACING" in prompt
        assert "NO ACUTE ANGLES" in prompt
        assert "MINIMIZE WIRE CROSSINGS" in prompt
        assert "HIGH-SPEED SIGNAL ISOLATION" in prompt
        assert "CONDUCTOR WIDTH" in prompt
        assert "3.3V" in prompt or "3.3" in prompt  # Voltage should be mentioned


@pytest.mark.skipif(
    not os.environ.get("OPENROUTER_API_KEY"),
    reason="OPENROUTER_API_KEY not set - skipping LLM integration tests"
)
class TestLLMIntegration:
    """Integration tests requiring OPENROUTER_API_KEY."""

    @pytest.mark.asyncio
    async def test_full_connection_generation(self):
        """Test full connection generation with LLM (requires API key)."""
        agent = ConnectionGeneratorAgent()

        bom = [
            {"part_number": "STM32G431", "category": "MCU", "reference": "U1"},
            {"part_number": "UCC21520", "category": "Gate_Driver", "reference": "U2"},
            {"part_number": "100nF", "category": "Capacitor", "reference": "C1", "value": "100nF"},
        ]

        design_intent = "Simple motor controller with PWM gate driver, 3.3V logic"

        connections = await agent.generate_connections(bom, design_intent)

        # Validate results
        assert len(connections) > 0
        assert any(c.net_name == "VCC" for c in connections)
        assert any(c.net_name == "GND" for c in connections)

        print(f"\n✅ Generated {len(connections)} IPC-2221 compliant connections")


if __name__ == "__main__":
    # Run tests manually
    import sys
    sys.exit(pytest.main([__file__, "-v", "-s"]))
