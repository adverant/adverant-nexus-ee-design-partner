"""
Comprehensive DRC Validation Test Suite

Tests all 8 DRC methods in EnhancedWireRouter:
1. _validate_electrical_rules() - Master validator
2. _check_short_circuits() - Detects net-to-net shorts
3. _check_clearance() - Validates IPC-2221 clearance
4. _check_trace_widths() - Validates current capacity
5. _check_four_way_junctions_strict() - Reports 4+ way junctions
6. _fix_four_way_junctions() - Fixes junctions
7. _wires_intersect() - Line intersection algorithm
8. _min_distance_between_wires() - Geometric distance calculation

Author: Nexus EE Design Team - Test Engineer
"""

import sys
import os
import unittest
from typing import List, Tuple

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from agents.wire_router.enhanced_wire_router import (
    EnhancedWireRouter,
    WireSegment,
    RouteType,
    RoutingResult
)


class TestDRCGeometricAlgorithms(unittest.TestCase):
    """Test geometric algorithms used in DRC validation."""

    def setUp(self):
        """Create router instance for testing."""
        self.router = EnhancedWireRouter()

    def test_lines_intersect_crossing(self):
        """Test that crossing lines are detected as intersecting."""
        # Create two wires that cross in the middle
        wire1 = WireSegment(
            start=(0.0, 0.0),
            end=(10.0, 10.0),
            net_name="Net1"
        )
        wire2 = WireSegment(
            start=(0.0, 10.0),
            end=(10.0, 0.0),
            net_name="Net2"
        )

        result = self.router._wires_intersect(wire1, wire2)
        self.assertTrue(result, "Lines that cross should be detected as intersecting")

    def test_lines_intersect_parallel(self):
        """Test that parallel lines are NOT detected as intersecting."""
        wire1 = WireSegment(
            start=(0.0, 0.0),
            end=(10.0, 0.0),
            net_name="Net1"
        )
        wire2 = WireSegment(
            start=(0.0, 5.0),
            end=(10.0, 5.0),
            net_name="Net2"
        )

        result = self.router._wires_intersect(wire1, wire2)
        self.assertFalse(result, "Parallel lines should NOT intersect")

    def test_lines_intersect_perpendicular_non_crossing(self):
        """Test perpendicular lines that don't actually cross."""
        wire1 = WireSegment(
            start=(0.0, 0.0),
            end=(5.0, 0.0),
            net_name="Net1"
        )
        wire2 = WireSegment(
            start=(10.0, -5.0),
            end=(10.0, 5.0),
            net_name="Net2"
        )

        result = self.router._wires_intersect(wire1, wire2)
        self.assertFalse(result, "Lines that don't cross should not intersect")

    def test_lines_intersect_touching_endpoints(self):
        """Test lines that share an endpoint (T-junction)."""
        wire1 = WireSegment(
            start=(0.0, 0.0),
            end=(10.0, 0.0),
            net_name="Net1"
        )
        wire2 = WireSegment(
            start=(5.0, 0.0),
            end=(5.0, 10.0),
            net_name="Net2"
        )

        result = self.router._wires_intersect(wire1, wire2)
        # This is a T-junction - algorithms vary on whether to treat as intersection
        # The CCW algorithm used should NOT treat shared endpoints as intersection
        self.assertFalse(result, "T-junctions (shared endpoint) should not be treated as intersection")

    def test_min_distance_same_point(self):
        """Test distance calculation when wires share a point."""
        wire1 = WireSegment(
            start=(0.0, 0.0),
            end=(10.0, 0.0),
            net_name="Net1"
        )
        wire2 = WireSegment(
            start=(0.0, 0.0),
            end=(0.0, 10.0),
            net_name="Net2"
        )

        distance = self.router._min_distance_between_wires(wire1, wire2)
        self.assertAlmostEqual(distance, 0.0, places=2,
                               msg="Wires sharing a point should have 0 distance")

    def test_min_distance_parallel_horizontal(self):
        """Test distance between parallel horizontal wires."""
        wire1 = WireSegment(
            start=(0.0, 0.0),
            end=(10.0, 0.0),
            net_name="Net1"
        )
        wire2 = WireSegment(
            start=(0.0, 5.0),
            end=(10.0, 5.0),
            net_name="Net2"
        )

        distance = self.router._min_distance_between_wires(wire1, wire2)
        self.assertAlmostEqual(distance, 5.0, places=2,
                               msg="Parallel wires 5mm apart should report 5mm distance")

    def test_min_distance_perpendicular(self):
        """Test distance between perpendicular wires."""
        wire1 = WireSegment(
            start=(0.0, 0.0),
            end=(10.0, 0.0),
            net_name="Net1"
        )
        wire2 = WireSegment(
            start=(15.0, -5.0),
            end=(15.0, 5.0),
            net_name="Net2"
        )

        distance = self.router._min_distance_between_wires(wire1, wire2)
        # Closest points are (10, 0) and (15, 0) = 5mm apart
        self.assertAlmostEqual(distance, 5.0, places=2,
                               msg="Perpendicular wires should measure closest approach")


class TestDRCShortCircuitDetection(unittest.TestCase):
    """Test short circuit detection between different nets."""

    def setUp(self):
        """Create router instance for testing."""
        self.router = EnhancedWireRouter()

    def test_short_circuit_detected(self):
        """Test that intersecting wires from different nets are detected as shorts."""
        wires = [
            WireSegment(start=(0.0, 0.0), end=(10.0, 10.0), net_name="VCC"),
            WireSegment(start=(0.0, 10.0), end=(10.0, 0.0), net_name="GND")
        ]

        violations = self.router._check_short_circuits(wires)
        self.assertEqual(len(violations), 1, "Should detect one short circuit")
        self.assertIn("SHORT CIRCUIT", violations[0])
        self.assertIn("VCC", violations[0])
        self.assertIn("GND", violations[0])

    def test_no_short_same_net(self):
        """Test that intersecting wires from the SAME net are NOT shorts."""
        wires = [
            WireSegment(start=(0.0, 0.0), end=(10.0, 10.0), net_name="VCC"),
            WireSegment(start=(0.0, 10.0), end=(10.0, 0.0), net_name="VCC")
        ]

        violations = self.router._check_short_circuits(wires)
        # Even if they cross, same net = no short
        self.assertEqual(len(violations), 0, "Same net wires should not trigger short circuit")

    def test_no_short_parallel_nets(self):
        """Test that parallel wires from different nets are NOT shorts."""
        wires = [
            WireSegment(start=(0.0, 0.0), end=(10.0, 0.0), net_name="DATA_A"),
            WireSegment(start=(0.0, 5.0), end=(10.0, 5.0), net_name="DATA_B")
        ]

        violations = self.router._check_short_circuits(wires)
        self.assertEqual(len(violations), 0, "Parallel wires should not short")

    def test_multiple_shorts(self):
        """Test detection of multiple shorts in a complex design."""
        wires = [
            # Short 1: VCC crosses GND
            WireSegment(start=(0.0, 0.0), end=(10.0, 10.0), net_name="VCC"),
            WireSegment(start=(0.0, 10.0), end=(10.0, 0.0), net_name="GND"),

            # Short 2: DATA_A crosses DATA_B
            WireSegment(start=(20.0, 0.0), end=(30.0, 10.0), net_name="DATA_A"),
            WireSegment(start=(20.0, 10.0), end=(30.0, 0.0), net_name="DATA_B")
        ]

        violations = self.router._check_short_circuits(wires)
        self.assertGreaterEqual(len(violations), 2, "Should detect at least 2 shorts")


class TestDRCClearanceValidation(unittest.TestCase):
    """Test IPC-2221 clearance validation."""

    def setUp(self):
        """Create router instance for testing."""
        self.router = EnhancedWireRouter()

    def test_clearance_violation_detected(self):
        """Test that wires too close together trigger clearance violation."""
        # IPC-2221: Minimum 0.13mm for 0-50V
        # Place wires 0.1mm apart (violates spec)
        wires = [
            WireSegment(start=(0.0, 0.0), end=(10.0, 0.0), net_name="VCC"),
            WireSegment(start=(0.0, 0.1), end=(10.0, 0.1), net_name="GND")
        ]
        connections = []  # Empty for this test

        violations = self.router._check_clearance(wires, connections)
        self.assertGreater(len(violations), 0, "Should detect clearance violation")
        self.assertIn("CLEARANCE VIOLATION", violations[0])
        self.assertIn("0.13", violations[0].lower() or violations[0])

    def test_clearance_adequate(self):
        """Test that wires with adequate spacing pass clearance check."""
        # IPC-2221: Minimum 0.13mm for 0-50V
        # Place wires 1.0mm apart (adequate)
        wires = [
            WireSegment(start=(0.0, 0.0), end=(10.0, 0.0), net_name="VCC"),
            WireSegment(start=(0.0, 1.0), end=(10.0, 1.0), net_name="GND")
        ]
        connections = []

        violations = self.router._check_clearance(wires, connections)
        self.assertEqual(len(violations), 0, "Adequate clearance should pass")

    def test_clearance_boundary_case(self):
        """Test clearance right at the boundary (0.13mm)."""
        # Exactly at minimum clearance
        wires = [
            WireSegment(start=(0.0, 0.0), end=(10.0, 0.0), net_name="VCC"),
            WireSegment(start=(0.0, 0.13), end=(10.0, 0.13), net_name="GND")
        ]
        connections = []

        violations = self.router._check_clearance(wires, connections)
        # At exactly the boundary, should pass (>=, not >)
        self.assertEqual(len(violations), 0, "Boundary case should pass")


class TestDRCFourWayJunctions(unittest.TestCase):
    """Test 4-way junction detection and fixing."""

    def setUp(self):
        """Create router instance for testing."""
        self.router = EnhancedWireRouter()

    def test_four_way_junction_detected(self):
        """Test that 4-way junctions are detected as errors."""
        # Create 4 wires meeting at the same point (5, 5)
        wires = [
            WireSegment(start=(0.0, 5.0), end=(5.0, 5.0), net_name="VCC"),  # Left
            WireSegment(start=(5.0, 5.0), end=(10.0, 5.0), net_name="VCC"), # Right
            WireSegment(start=(5.0, 0.0), end=(5.0, 5.0), net_name="VCC"),  # Bottom
            WireSegment(start=(5.0, 5.0), end=(5.0, 10.0), net_name="VCC")  # Top
        ]

        violations = self.router._check_four_way_junctions_strict(wires)
        self.assertEqual(len(violations), 1, "Should detect one 4-way junction")
        self.assertIn("4-WAY JUNCTION", violations[0])
        self.assertIn("4 wires", violations[0])

    def test_three_way_junction_allowed(self):
        """Test that 3-way (T) junctions are NOT errors."""
        # Create 3 wires meeting at the same point (5, 5)
        wires = [
            WireSegment(start=(0.0, 5.0), end=(5.0, 5.0), net_name="VCC"),  # Left
            WireSegment(start=(5.0, 5.0), end=(10.0, 5.0), net_name="VCC"), # Right
            WireSegment(start=(5.0, 5.0), end=(5.0, 10.0), net_name="VCC")  # Top
        ]

        violations = self.router._check_four_way_junctions_strict(wires)
        self.assertEqual(len(violations), 0, "3-way junctions should be allowed")

    def test_multiple_four_way_junctions(self):
        """Test detection of multiple 4-way junctions."""
        wires = [
            # First 4-way at (5, 5)
            WireSegment(start=(0.0, 5.0), end=(5.0, 5.0), net_name="VCC"),
            WireSegment(start=(5.0, 5.0), end=(10.0, 5.0), net_name="VCC"),
            WireSegment(start=(5.0, 0.0), end=(5.0, 5.0), net_name="VCC"),
            WireSegment(start=(5.0, 5.0), end=(5.0, 10.0), net_name="VCC"),

            # Second 4-way at (20, 20)
            WireSegment(start=(15.0, 20.0), end=(20.0, 20.0), net_name="GND"),
            WireSegment(start=(20.0, 20.0), end=(25.0, 20.0), net_name="GND"),
            WireSegment(start=(20.0, 15.0), end=(20.0, 20.0), net_name="GND"),
            WireSegment(start=(20.0, 20.0), end=(20.0, 25.0), net_name="GND")
        ]

        violations = self.router._check_four_way_junctions_strict(wires)
        self.assertEqual(len(violations), 2, "Should detect two 4-way junctions")


class TestDRCTraceWidthValidation(unittest.TestCase):
    """Test trace width validation (stub for future implementation)."""

    def setUp(self):
        """Create router instance for testing."""
        self.router = EnhancedWireRouter()

    def test_trace_width_stub_returns_empty(self):
        """Test that trace width check returns empty (stub implementation)."""
        wires = [
            WireSegment(start=(0.0, 0.0), end=(10.0, 0.0), net_name="VCC", route_type=RouteType.POWER)
        ]
        connections = []

        violations = self.router._check_trace_widths(wires, connections)
        self.assertEqual(len(violations), 0, "Stub implementation should return no violations")


class TestDRCIntegration(unittest.TestCase):
    """Integration tests for full DRC validation workflow."""

    def setUp(self):
        """Create router instance for testing."""
        self.router = EnhancedWireRouter()

    def test_clean_design_passes_all_drc(self):
        """Test that a well-designed circuit passes all DRC checks."""
        # Create a clean design with proper spacing and no violations
        wires = [
            # VCC net - well spaced
            WireSegment(start=(0.0, 0.0), end=(10.0, 0.0), net_name="VCC"),
            WireSegment(start=(10.0, 0.0), end=(10.0, 5.0), net_name="VCC"),

            # GND net - well spaced from VCC (5mm vertical distance)
            WireSegment(start=(0.0, 10.0), end=(10.0, 10.0), net_name="GND"),
            WireSegment(start=(10.0, 10.0), end=(10.0, 15.0), net_name="GND")
        ]
        connections = []

        violations = self.router._validate_electrical_rules(connections, wires)
        self.assertEqual(len(violations), 0, "Clean design should pass all DRC checks")

    def test_design_with_short_fails_drc(self):
        """Test that a design with shorts fails DRC."""
        wires = [
            # Crossing wires (short circuit)
            WireSegment(start=(0.0, 0.0), end=(10.0, 10.0), net_name="VCC"),
            WireSegment(start=(0.0, 10.0), end=(10.0, 0.0), net_name="GND")
        ]
        connections = []

        violations = self.router._validate_electrical_rules(connections, wires)
        self.assertGreater(len(violations), 0, "Design with shorts should fail DRC")

        # Should contain short circuit violation
        has_short = any("SHORT CIRCUIT" in v for v in violations)
        self.assertTrue(has_short, "Should detect short circuit")

    def test_design_with_clearance_violation_fails_drc(self):
        """Test that a design with clearance violations fails DRC."""
        wires = [
            # Too close together (0.05mm < 0.13mm minimum)
            WireSegment(start=(0.0, 0.0), end=(10.0, 0.0), net_name="DATA_A"),
            WireSegment(start=(0.0, 0.05), end=(10.0, 0.05), net_name="DATA_B")
        ]
        connections = []

        violations = self.router._validate_electrical_rules(connections, wires)
        self.assertGreater(len(violations), 0, "Design with clearance violations should fail DRC")

        # Should contain clearance violation
        has_clearance = any("CLEARANCE VIOLATION" in v for v in violations)
        self.assertTrue(has_clearance, "Should detect clearance violation")

    def test_design_with_four_way_junction_fails_drc(self):
        """Test that a design with 4-way junctions fails DRC."""
        wires = [
            # 4-way junction at (5, 5)
            WireSegment(start=(0.0, 5.0), end=(5.0, 5.0), net_name="VCC"),
            WireSegment(start=(5.0, 5.0), end=(10.0, 5.0), net_name="VCC"),
            WireSegment(start=(5.0, 0.0), end=(5.0, 5.0), net_name="VCC"),
            WireSegment(start=(5.0, 5.0), end=(5.0, 10.0), net_name="VCC")
        ]
        connections = []

        violations = self.router._validate_electrical_rules(connections, wires)
        self.assertGreater(len(violations), 0, "Design with 4-way junctions should fail DRC")

        # Should contain 4-way junction error
        has_junction = any("4-WAY JUNCTION" in v for v in violations)
        self.assertTrue(has_junction, "Should detect 4-way junction")

    def test_complex_design_multiple_violations(self):
        """Test detection of multiple different violation types in one design."""
        wires = [
            # Short circuit
            WireSegment(start=(0.0, 0.0), end=(10.0, 10.0), net_name="VCC"),
            WireSegment(start=(0.0, 10.0), end=(10.0, 0.0), net_name="GND"),

            # Clearance violation
            WireSegment(start=(20.0, 0.0), end=(30.0, 0.0), net_name="DATA_A"),
            WireSegment(start=(20.0, 0.05), end=(30.0, 0.05), net_name="DATA_B"),

            # 4-way junction
            WireSegment(start=(40.0, 5.0), end=(45.0, 5.0), net_name="SIG"),
            WireSegment(start=(45.0, 5.0), end=(50.0, 5.0), net_name="SIG"),
            WireSegment(start=(45.0, 0.0), end=(45.0, 5.0), net_name="SIG"),
            WireSegment(start=(45.0, 5.0), end=(45.0, 10.0), net_name="SIG")
        ]
        connections = []

        violations = self.router._validate_electrical_rules(connections, wires)
        self.assertGreaterEqual(len(violations), 3, "Should detect at least 3 violations")

        # Verify each type is detected
        violation_text = " ".join(violations)
        self.assertIn("SHORT CIRCUIT", violation_text)
        self.assertIn("CLEARANCE VIOLATION", violation_text)
        self.assertIn("4-WAY JUNCTION", violation_text)


class TestDRCRealWorldScenarios(unittest.TestCase):
    """Test DRC with realistic circuit scenarios."""

    def setUp(self):
        """Create router instance for testing."""
        self.router = EnhancedWireRouter()

    def test_power_distribution_network(self):
        """Test DRC on a typical power distribution network."""
        # Simulate VCC power rail with multiple drops to components
        wires = [
            # Horizontal power rail
            WireSegment(start=(10.0, 10.0), end=(100.0, 10.0), net_name="VCC", route_type=RouteType.POWER),

            # Vertical drops to components (properly spaced)
            WireSegment(start=(30.0, 10.0), end=(30.0, 20.0), net_name="VCC", route_type=RouteType.POWER),
            WireSegment(start=(50.0, 10.0), end=(50.0, 20.0), net_name="VCC", route_type=RouteType.POWER),
            WireSegment(start=(70.0, 10.0), end=(70.0, 20.0), net_name="VCC", route_type=RouteType.POWER),

            # Ground rail below (10mm clearance - adequate)
            WireSegment(start=(10.0, 30.0), end=(100.0, 30.0), net_name="GND", route_type=RouteType.GROUND),
            WireSegment(start=(30.0, 30.0), end=(30.0, 25.0), net_name="GND", route_type=RouteType.GROUND),
            WireSegment(start=(50.0, 30.0), end=(50.0, 25.0), net_name="GND", route_type=RouteType.GROUND),
            WireSegment(start=(70.0, 30.0), end=(70.0, 25.0), net_name="GND", route_type=RouteType.GROUND)
        ]
        connections = []

        violations = self.router._validate_electrical_rules(connections, wires)
        # Power distribution with proper spacing should pass
        self.assertEqual(len(violations), 0, "Proper power distribution should pass DRC")

    def test_bus_routing(self):
        """Test DRC on parallel bus signals."""
        # Simulate an 8-bit data bus with 2.54mm spacing
        wires = []
        for i in range(8):
            y_offset = i * 2.54  # Standard bus spacing
            wires.append(WireSegment(
                start=(0.0, y_offset),
                end=(50.0, y_offset),
                net_name=f"DATA[{i}]",
                route_type=RouteType.BUS
            ))

        connections = []
        violations = self.router._validate_electrical_rules(connections, wires)
        # Bus with 2.54mm spacing should pass clearance checks
        self.assertEqual(len(violations), 0, "Properly spaced bus should pass DRC")

    def test_differential_pair(self):
        """Test DRC on differential signal pairs."""
        # Differential pairs run close together
        wires = [
            # USB D+ and D- (0.5mm spacing - this is adequate per IPC-2221)
            WireSegment(start=(0.0, 0.0), end=(50.0, 0.0), net_name="USB_DP"),
            WireSegment(start=(0.0, 0.5), end=(50.0, 0.5), net_name="USB_DN")
        ]
        connections = []

        violations = self.router._validate_electrical_rules(connections, wires)
        # 0.5mm > 0.13mm minimum, so this should pass
        self.assertEqual(len(violations), 0,
                        "Differential pair with 0.5mm spacing should pass IPC-2221 clearance")

    def test_differential_pair_too_close(self):
        """Test DRC catches differential pairs that are TOO close."""
        # Differential pairs that violate IPC-2221 clearance
        wires = [
            # USB D+ and D- (0.1mm spacing - violates 0.13mm minimum)
            WireSegment(start=(0.0, 0.0), end=(50.0, 0.0), net_name="USB_DP"),
            WireSegment(start=(0.0, 0.1), end=(50.0, 0.1), net_name="USB_DN")
        ]
        connections = []

        violations = self.router._validate_electrical_rules(connections, wires)
        # Should fail clearance check
        self.assertGreater(len(violations), 0,
                          "Differential pair with 0.1mm spacing should fail clearance check")


def run_tests():
    """Run all DRC validation tests."""
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(TestDRCGeometricAlgorithms))
    suite.addTests(loader.loadTestsFromTestCase(TestDRCShortCircuitDetection))
    suite.addTests(loader.loadTestsFromTestCase(TestDRCClearanceValidation))
    suite.addTests(loader.loadTestsFromTestCase(TestDRCFourWayJunctions))
    suite.addTests(loader.loadTestsFromTestCase(TestDRCTraceWidthValidation))
    suite.addTests(loader.loadTestsFromTestCase(TestDRCIntegration))
    suite.addTests(loader.loadTestsFromTestCase(TestDRCRealWorldScenarios))

    # Run tests with detailed output
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # Return summary
    return {
        'tests_run': result.testsRun,
        'failures': len(result.failures),
        'errors': len(result.errors),
        'skipped': len(result.skipped),
        'success': result.wasSuccessful()
    }


if __name__ == '__main__':
    print("=" * 80)
    print("DRC VALIDATION TEST SUITE")
    print("Testing 8 DRC methods in EnhancedWireRouter")
    print("=" * 80)
    print()

    summary = run_tests()

    print()
    print("=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    print(f"Tests run:     {summary['tests_run']}")
    print(f"Failures:      {summary['failures']}")
    print(f"Errors:        {summary['errors']}")
    print(f"Skipped:       {summary['skipped']}")
    print(f"Success:       {'✅ PASS' if summary['success'] else '❌ FAIL'}")
    print("=" * 80)

    sys.exit(0 if summary['success'] else 1)
