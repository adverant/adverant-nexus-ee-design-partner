#!/usr/bin/env python3
"""
Test script for MAPO Schematic Pipeline.

Runs a quick test of the schematic generation pipeline without full validation.
"""

import asyncio
import sys
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent))

from mapo_schematic_pipeline import MAPOSchematicPipeline, PipelineConfig


async def test_pipeline():
    """Test the MAPO pipeline with a simple FOC ESC BOM."""

    print("\n" + "=" * 60)
    print("MAPO Schematic Pipeline Test")
    print("=" * 60)

    # Create test BOM for FOC ESC
    test_bom = [
        {
            "part_number": "STM32G431CBT6",
            "category": "MCU",
            "manufacturer": "STMicroelectronics",
            "value": "STM32G431",
            "description": "ARM Cortex-M4 MCU with FPU",
        },
        {
            "part_number": "DRV8323RS",
            "category": "Gate_Driver",
            "manufacturer": "Texas Instruments",
            "value": "DRV8323",
            "description": "3-Phase Gate Driver with Buck Regulator",
        },
        {
            "part_number": "CSD19505KCS",
            "category": "MOSFET",
            "manufacturer": "Texas Instruments",
            "value": "80V/150A",
            "description": "N-Channel Power MOSFET",
        },
        {
            "part_number": "100uF/50V",
            "category": "Capacitor",
            "value": "100uF",
            "description": "Electrolytic bulk capacitor",
        },
        {
            "part_number": "10uF/25V",
            "category": "Capacitor",
            "value": "10uF",
            "description": "Ceramic bypass capacitor",
        },
        {
            "part_number": "0.1uF",
            "category": "Capacitor",
            "value": "0.1uF",
            "description": "Decoupling capacitor",
        },
        {
            "part_number": "10k_0603",
            "category": "Resistor",
            "value": "10k",
            "description": "Pull-up/pull-down resistor",
        },
        {
            "part_number": "1k_0603",
            "category": "Resistor",
            "value": "1k",
            "description": "Current limiting resistor",
        },
        {
            "part_number": "0.001R_2512",
            "category": "Resistor",
            "value": "1mOhm",
            "description": "Current sense shunt resistor",
        },
    ]

    # Test connections
    test_connections = [
        {"from_ref": "U1", "from_pin": "VDD", "to_ref": "C1", "to_pin": "1", "net_name": "VCC_3V3"},
        {"from_ref": "U1", "from_pin": "VSS", "to_ref": "C1", "to_pin": "2", "net_name": "GND"},
        {"from_ref": "U2", "from_pin": "PVDD", "to_ref": "C2", "to_pin": "1", "net_name": "PVDD"},
        {"from_ref": "U2", "from_pin": "GND", "to_ref": "C2", "to_pin": "2", "net_name": "GND"},
    ]

    design_intent = """
    200A FOC (Field-Oriented Control) Electronic Speed Controller for brushless motors.
    Features:
    - STM32G4 MCU for motor control algorithms
    - DRV8323 3-phase gate driver
    - 6 N-channel MOSFETs in half-bridge configuration
    - Current sensing via shunt resistors
    - 12-48V input voltage range
    """

    # Configure pipeline (skip validation for quick test)
    config = PipelineConfig(
        symbol_cache_path=Path("/tmp/kicad-symbols-test"),
        use_graphrag=False,  # Skip GraphRAG for quick test
        output_dir=Path("/tmp/mapo-test-output"),
    )

    pipeline = MAPOSchematicPipeline(config)

    try:
        # Initialize pipeline
        print("\n[1/3] Initializing pipeline...")
        await pipeline.initialize()
        print("      Pipeline initialized successfully")

        # Run generation (skip validation for quick test)
        print("\n[2/3] Generating schematic...")
        result = await pipeline.generate(
            bom=test_bom,
            design_intent=design_intent,
            connections=test_connections,
            design_name="foc_esc_test",
            skip_validation=True,  # Skip for quick test
        )

        # Show results
        print("\n[3/3] Results:")
        print("-" * 40)
        print(f"  Success: {result.success}")
        print(f"  Output: {result.schematic_path}")
        print(f"  Sheets: {len(result.sheets)}")
        print(f"  Symbols fetched: {result.symbols_fetched}")
        print(f"    - From cache: {result.symbols_from_cache}")
        print(f"    - Generated: {result.symbols_generated}")
        print(f"  Time: {result.total_time_seconds:.2f}s")

        if result.errors:
            print(f"\n  Errors:")
            for error in result.errors:
                print(f"    - {error}")

        # Show first 50 lines of output
        if result.schematic_path and result.schematic_path.exists():
            print(f"\n  Schematic preview (first 30 lines):")
            print("-" * 40)
            content = result.schematic_path.read_text()
            lines = content.split("\n")[:30]
            for line in lines:
                print(f"  {line}")
            print("  ...")

        print("\n" + "=" * 60)
        print("TEST COMPLETE")
        print("=" * 60 + "\n")

        return result.success

    except Exception as e:
        print(f"\n[ERROR] Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    finally:
        await pipeline.close()


if __name__ == "__main__":
    success = asyncio.run(test_pipeline())
    sys.exit(0 if success else 1)
