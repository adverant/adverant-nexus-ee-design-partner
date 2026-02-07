"""
Test suite for Layout Optimizer Agent (MAPO v3.1).

Tests the signal flow-based layout optimizer against zone-based v3.0 baseline.
Measures improvements in wire length, crossings, and signal flow clarity.

Author: Nexus EE Design Team
"""

import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

# Add parent directories to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from layout_optimizer_agent import LayoutOptimizerAgent, OptimizationResult
from signal_flow_analyzer import SignalFlowAnalyzer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Mock data structures (simulating pipeline objects)
# ---------------------------------------------------------------------------


@dataclass
class MockSymbol:
    """Mock SymbolInstance for testing."""
    reference: str
    position: Tuple[float, float]
    value: str = ""


@dataclass
class MockConnection:
    """Mock Connection for testing."""
    from_ref: str
    from_pin: str
    to_ref: str
    to_pin: str
    net_name: str


# ---------------------------------------------------------------------------
# Test scenarios
# ---------------------------------------------------------------------------


def create_simple_power_circuit():
    """
    Simple power circuit: Regulator → MCU → LED

    Components:
    - J1: Input connector
    - U1: LDO regulator
    - U2: MCU
    - C1, C2: Bypass caps
    - R1: Current limiting resistor
    - D1: LED
    """
    symbols = [
        MockSymbol('J1', (50.0, 50.0)),     # Input connector
        MockSymbol('U1', (100.0, 50.0)),    # LDO regulator
        MockSymbol('U2', (150.0, 50.0)),    # MCU
        MockSymbol('C1', (110.0, 60.0)),    # Bypass cap for U1
        MockSymbol('C2', (160.0, 60.0)),    # Bypass cap for U2
        MockSymbol('R1', (180.0, 50.0)),    # Current limiting resistor
        MockSymbol('D1', (200.0, 50.0)),    # LED
    ]

    connections = [
        # Power from J1 to U1
        MockConnection('J1', '1', 'U1', 'VIN', 'VIN'),
        # Ground
        MockConnection('J1', '2', 'U1', 'GND', 'GND'),
        MockConnection('U1', 'GND', 'U2', 'GND', 'GND'),
        # Regulated power to MCU
        MockConnection('U1', 'VOUT', 'U2', 'VDD', 'VCC_3V3'),
        # Bypass caps
        MockConnection('U1', 'VOUT', 'C1', '1', 'VCC_3V3'),
        MockConnection('C1', '2', 'U1', 'GND', 'GND'),
        MockConnection('U2', 'VDD', 'C2', '1', 'VCC_3V3'),
        MockConnection('C2', '2', 'U2', 'GND', 'GND'),
        # MCU to LED
        MockConnection('U2', 'PA5', 'R1', '1', 'LED_DRIVE'),
        MockConnection('R1', '2', 'D1', 'A', 'LED_DRIVE'),
        MockConnection('D1', 'K', 'U2', 'GND', 'GND'),
    ]

    bom = [
        {'reference': 'J1', 'category': 'Connector', 'value': '2-pin', 'part_number': ''},
        {'reference': 'U1', 'category': 'Regulator', 'value': '3.3V LDO', 'part_number': 'MIC5504'},
        {'reference': 'U2', 'category': 'MCU', 'value': 'STM32', 'part_number': 'STM32G431'},
        {'reference': 'C1', 'category': 'Capacitor', 'value': '10uF', 'part_number': ''},
        {'reference': 'C2', 'category': 'Capacitor', 'value': '100nF', 'part_number': ''},
        {'reference': 'R1', 'category': 'Resistor', 'value': '330R', 'part_number': ''},
        {'reference': 'D1', 'category': 'LED', 'value': 'Red', 'part_number': ''},
    ]

    return symbols, connections, bom


def create_motor_driver_circuit():
    """
    Motor driver circuit: MCU → Gate Driver → MOSFET → Motor

    More complex with multiple signal paths and power domains.
    """
    symbols = [
        # Power input
        MockSymbol('J1', (50.0, 50.0)),     # 12V input

        # Power regulation
        MockSymbol('U1', (100.0, 30.0)),    # 12V to 5V buck converter
        MockSymbol('U2', (100.0, 70.0)),    # 5V to 3.3V LDO

        # MCU
        MockSymbol('U3', (150.0, 70.0)),    # MCU

        # Gate driver
        MockSymbol('U4', (200.0, 50.0)),    # Gate driver IC

        # Power stage
        MockSymbol('Q1', (250.0, 30.0)),    # High-side MOSFET
        MockSymbol('Q2', (250.0, 60.0)),    # Low-side MOSFET

        # Motor
        MockSymbol('M1', (300.0, 45.0)),    # Motor

        # Passives
        MockSymbol('C1', (110.0, 40.0)),    # Buck output cap
        MockSymbol('C2', (110.0, 80.0)),    # LDO output cap
        MockSymbol('C3', (160.0, 80.0)),    # MCU bypass cap
        MockSymbol('C4', (210.0, 60.0)),    # Gate driver bypass cap
        MockSymbol('R1', (190.0, 70.0)),    # Gate resistor 1
        MockSymbol('R2', (190.0, 80.0)),    # Gate resistor 2
        MockSymbol('D1', (260.0, 45.0)),    # Freewheeling diode
    ]

    connections = [
        # Power distribution
        MockConnection('J1', '1', 'U1', 'VIN', 'VIN_12V'),
        MockConnection('U1', 'VOUT', 'U2', 'VIN', 'V5V'),
        MockConnection('U2', 'VOUT', 'U3', 'VDD', 'VCC_3V3'),
        MockConnection('U1', 'VOUT', 'U4', 'VDD', 'V5V'),

        # Ground
        MockConnection('J1', '2', 'U1', 'GND', 'GND'),
        MockConnection('U1', 'GND', 'U2', 'GND', 'GND'),
        MockConnection('U2', 'GND', 'U3', 'GND', 'GND'),
        MockConnection('U3', 'GND', 'U4', 'GND', 'GND'),

        # Bypass caps
        MockConnection('U1', 'VOUT', 'C1', '1', 'V5V'),
        MockConnection('U2', 'VOUT', 'C2', '1', 'VCC_3V3'),
        MockConnection('U3', 'VDD', 'C3', '1', 'VCC_3V3'),
        MockConnection('U4', 'VDD', 'C4', '1', 'V5V'),

        # MCU to gate driver
        MockConnection('U3', 'PA8', 'U4', 'IN1', 'PWM_H'),
        MockConnection('U3', 'PA9', 'U4', 'IN2', 'PWM_L'),

        # Gate driver to MOSFETs
        MockConnection('U4', 'OUT1', 'R1', '1', 'GATE_H'),
        MockConnection('R1', '2', 'Q1', 'G', 'GATE_H'),
        MockConnection('U4', 'OUT2', 'R2', '1', 'GATE_L'),
        MockConnection('R2', '2', 'Q2', 'G', 'GATE_L'),

        # H-bridge
        MockConnection('J1', '1', 'Q1', 'D', 'VIN_12V'),
        MockConnection('Q1', 'S', 'M1', '1', 'MOTOR_A'),
        MockConnection('M1', '1', 'Q2', 'D', 'MOTOR_A'),
        MockConnection('Q2', 'S', 'U1', 'GND', 'GND'),

        # Freewheeling diode
        MockConnection('M1', '1', 'D1', 'A', 'MOTOR_A'),
        MockConnection('D1', 'K', 'J1', '1', 'VIN_12V'),
    ]

    bom = [
        {'reference': 'J1', 'category': 'Connector', 'value': '12V', 'part_number': ''},
        {'reference': 'U1', 'category': 'Regulator', 'value': 'Buck 5V', 'part_number': 'TPS54331'},
        {'reference': 'U2', 'category': 'LDO', 'value': 'LDO 3.3V', 'part_number': 'AMS1117'},
        {'reference': 'U3', 'category': 'MCU', 'value': 'STM32', 'part_number': 'STM32G431'},
        {'reference': 'U4', 'category': 'Gate_Driver', 'value': 'Gate Driver', 'part_number': 'IR2110'},
        {'reference': 'Q1', 'category': 'MOSFET', 'value': 'N-CH', 'part_number': 'IRF540'},
        {'reference': 'Q2', 'category': 'MOSFET', 'value': 'N-CH', 'part_number': 'IRF540'},
        {'reference': 'M1', 'category': 'Motor', 'value': 'DC Motor', 'part_number': ''},
        {'reference': 'C1', 'category': 'Capacitor', 'value': '100uF', 'part_number': ''},
        {'reference': 'C2', 'category': 'Capacitor', 'value': '10uF', 'part_number': ''},
        {'reference': 'C3', 'category': 'Capacitor', 'value': '100nF', 'part_number': ''},
        {'reference': 'C4', 'category': 'Capacitor', 'value': '100nF', 'part_number': ''},
        {'reference': 'R1', 'category': 'Resistor', 'value': '10R', 'part_number': ''},
        {'reference': 'R2', 'category': 'Resistor', 'value': '10R', 'part_number': ''},
        {'reference': 'D1', 'category': 'Diode', 'value': 'Schottky', 'part_number': 'SS34'},
    ]

    return symbols, connections, bom


# ---------------------------------------------------------------------------
# Test execution
# ---------------------------------------------------------------------------


def run_test_scenario(
    name: str,
    symbols: List[MockSymbol],
    connections: List[MockConnection],
    bom: List[Dict]
):
    """Run optimization test on a scenario."""
    logger.info(f"\n{'='*70}")
    logger.info(f"Test Scenario: {name}")
    logger.info(f"{'='*70}")
    logger.info(f"Components: {len(symbols)}, Connections: {len(connections)}")

    # Store original positions
    original_positions = {s.reference: s.position for s in symbols}

    # Create optimizer
    optimizer = LayoutOptimizerAgent()

    # Run optimization
    result = optimizer.optimize_layout(
        symbols=symbols,
        connections=connections,
        bom=bom,
        placement_hints=None
    )

    # Print results
    logger.info(f"\n--- Optimization Results ---")
    logger.info(f"Success: {result.success}")
    logger.info(f"Grid corrections: {result.grid_corrections}")
    logger.info(f"Spacing corrections: {result.spacing_corrections}")
    logger.info(f"Improvements: {len(result.improvements)}")
    logger.info(f"Violations: {len(result.violations)}")

    if result.violations:
        logger.warning(f"\nViolations:")
        for violation in result.violations[:5]:  # Show first 5
            logger.warning(f"  - {violation}")

    # Print metrics
    if result.metrics:
        logger.info(f"\n--- Quality Metrics ---")
        logger.info(f"Total wire length: {result.metrics.get('total_wire_length', 0):.1f} mm")
        logger.info(f"Wire crossings: {result.metrics.get('wire_crossings', 0)}")
        logger.info(f"Signal flow score: {result.metrics.get('signal_flow_score', 0):.2f} / 1.0")

        improvement_pct = result.metrics.get('wire_length_improvement_pct', 0)
        logger.info(f"Wire length improvement: {improvement_pct:.1f}%")

    # Print signal flow analysis summary
    if result.analysis:
        logger.info(f"\n--- Signal Flow Analysis ---")
        logger.info(f"Signal paths identified: {len(result.analysis.signal_paths)}")
        logger.info(f"Component layers: {len(result.analysis.component_layers)}")
        logger.info(f"Functional groups: {len(result.analysis.functional_groups)}")
        logger.info(f"Proximity pairs: {len(result.analysis.critical_proximity_pairs)}")
        logger.info(f"Separation zones: {len(result.analysis.separation_zones)}")

        # Show critical paths
        if result.analysis.signal_paths:
            logger.info(f"\nTop 3 Critical Signal Paths:")
            for i, path in enumerate(result.analysis.signal_paths[:3], 1):
                logger.info(
                    f"  {i}. {path.source_component} → {', '.join(path.sink_components)} "
                    f"({path.path_type}, criticality={path.criticality:.2f})"
                )

        # Show layers
        if result.analysis.component_layers:
            logger.info(f"\nComponent Layers (left → right):")
            for layer in result.analysis.component_layers:
                logger.info(
                    f"  Layer {layer.layer_id} ({layer.layer_name}): "
                    f"{len(layer.components)} components @ x={layer.x_position_hint:.2f}"
                )

        # Show functional groups
        if result.analysis.functional_groups:
            logger.info(f"\nFunctional Groups:")
            for group in result.analysis.functional_groups:
                logger.info(
                    f"  {group.group_name}: {len(group.components)} components"
                )

    # Show position changes
    logger.info(f"\n--- Position Changes (sample) ---")
    for i, improvement in enumerate(result.improvements[:5], 1):
        logger.info(f"  {i}. {improvement}")

    return result


def calculate_comparison_metrics(results: List[OptimizationResult]) -> Dict[str, float]:
    """Calculate aggregate metrics across all test scenarios."""
    if not results:
        return {}

    metrics = {
        'avg_wire_length': sum(r.metrics.get('total_wire_length', 0) for r in results if r.metrics) / len(results),
        'avg_crossings': sum(r.metrics.get('wire_crossings', 0) for r in results if r.metrics) / len(results),
        'avg_signal_flow_score': sum(r.metrics.get('signal_flow_score', 0) for r in results if r.metrics) / len(results),
        'avg_improvement_pct': sum(r.metrics.get('wire_length_improvement_pct', 0) for r in results if r.metrics) / len(results),
        'success_rate': sum(1 for r in results if r.success) / len(results) * 100,
    }

    return metrics


# ---------------------------------------------------------------------------
# Main test runner
# ---------------------------------------------------------------------------


def main():
    """Run all test scenarios."""
    logger.info("="*70)
    logger.info("MAPO v3.1 Layout Optimizer Test Suite")
    logger.info("Signal Flow Analysis vs. Zone-Based Baseline")
    logger.info("="*70)

    results = []

    # Test 1: Simple power circuit
    symbols, connections, bom = create_simple_power_circuit()
    result = run_test_scenario(
        "Simple Power Circuit (Regulator → MCU → LED)",
        symbols,
        connections,
        bom
    )
    results.append(result)

    # Test 2: Motor driver circuit
    symbols, connections, bom = create_motor_driver_circuit()
    result = run_test_scenario(
        "Motor Driver Circuit (MCU → Gate Driver → H-Bridge)",
        symbols,
        connections,
        bom
    )
    results.append(result)

    # Aggregate metrics
    logger.info(f"\n{'='*70}")
    logger.info("AGGREGATE METRICS (all scenarios)")
    logger.info(f"{'='*70}")

    comparison = calculate_comparison_metrics(results)
    logger.info(f"Average wire length: {comparison.get('avg_wire_length', 0):.1f} mm")
    logger.info(f"Average crossings: {comparison.get('avg_crossings', 0):.1f}")
    logger.info(f"Average signal flow score: {comparison.get('avg_signal_flow_score', 0):.2f} / 1.0")
    logger.info(f"Average improvement: {comparison.get('avg_improvement_pct', 0):.1f}%")
    logger.info(f"Success rate: {comparison.get('success_rate', 0):.1f}%")

    # Final verdict
    logger.info(f"\n{'='*70}")
    logger.info("ASSESSMENT")
    logger.info(f"{'='*70}")

    signal_flow_score = comparison.get('avg_signal_flow_score', 0)
    improvement = comparison.get('avg_improvement_pct', 0)
    success_rate = comparison.get('success_rate', 0)

    if signal_flow_score >= 0.7 and improvement >= 30 and success_rate >= 80:
        logger.info("✓ PASS: Signal flow analysis produces professional-quality layouts")
        logger.info(f"  - Signal flow clarity: {signal_flow_score:.2f} / 1.0")
        logger.info(f"  - Wire length improvement: {improvement:.1f}%")
        logger.info(f"  - Success rate: {success_rate:.1f}%")
    elif signal_flow_score >= 0.5 and improvement >= 20:
        logger.warning("⚠ PARTIAL: Signal flow analysis shows improvement but needs refinement")
        logger.warning(f"  - Signal flow clarity: {signal_flow_score:.2f} / 1.0 (target: 0.7)")
        logger.warning(f"  - Wire length improvement: {improvement:.1f}% (target: 30%)")
        logger.warning(f"  - Success rate: {success_rate:.1f}% (target: 80%)")
    else:
        logger.error("✗ FAIL: Signal flow analysis does not meet quality standards")
        logger.error(f"  - Signal flow clarity: {signal_flow_score:.2f} / 1.0 (target: 0.7)")
        logger.error(f"  - Wire length improvement: {improvement:.1f}% (target: 30%)")
        logger.error(f"  - Success rate: {success_rate:.1f}% (target: 80%)")

    logger.info(f"\n{'='*70}")
    logger.info("Test suite complete")
    logger.info(f"{'='*70}\n")


if __name__ == "__main__":
    main()
