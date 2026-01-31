#!/usr/bin/env python3
"""
Test wire connections in schematic generation.
Verifies that wires are properly generated between connected components.
"""

import asyncio
import sys
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from mapo_schematic_pipeline import MAPOSchematicPipeline, PipelineConfig


async def test_wire_connections():
    """Test schematic generation with explicit connections."""
    print("=" * 60)
    print("TESTING WIRE CONNECTIONS")
    print("=" * 60)

    # Define a simple test BOM - MCU with bypass caps
    test_bom = [
        {
            "part_number": "STM32G431CBT6",
            "manufacturer": "ST",
            "category": "MCU",
            "value": "STM32G4",
            "reference": "U1",
        },
        {
            "part_number": "100nF",
            "manufacturer": "Generic",
            "category": "Capacitor",
            "value": "100nF",
            "reference": "C1",
        },
        {
            "part_number": "10uF",
            "manufacturer": "Generic",
            "category": "Capacitor",
            "value": "10uF",
            "reference": "C2",
        },
        {
            "part_number": "10k",
            "manufacturer": "Generic",
            "category": "Resistor",
            "value": "10k",
            "reference": "R1",
        },
    ]

    # Define connections between components
    # These should generate wires in the schematic
    test_connections = [
        # MCU VDD to bypass cap
        {"from_ref": "U1", "from_pin": "VDD", "to_ref": "C1", "to_pin": "1", "net_name": "VCC_3V3"},
        # Bypass cap GND
        {"from_ref": "C1", "from_pin": "2", "to_ref": "C2", "to_pin": "2", "net_name": "GND"},
        # MCU NRST to resistor
        {"from_ref": "U1", "from_pin": "NRST", "to_ref": "R1", "to_pin": "1", "net_name": "NRST"},
        # Resistor to VCC
        {"from_ref": "R1", "from_pin": "2", "to_ref": "C2", "to_pin": "1", "net_name": "VCC_3V3"},
    ]

    design_intent = "Test schematic with wire connections"

    # Configure pipeline
    config = PipelineConfig(
        symbol_cache_path=Path(__file__).parent / "symbol_cache",
        use_graphrag=False,
        output_dir=Path(__file__).parent / "output",
    )

    pipeline = MAPOSchematicPipeline(config)

    try:
        print("\n[1/3] Initializing pipeline...")
        await pipeline.initialize()

        print("\n[2/3] Generating schematic with connections...")
        result = await pipeline.generate(
            bom=test_bom,
            design_intent=design_intent,
            connections=test_connections,
            design_name="wire_test",
            skip_validation=True,
        )

        print("\n[3/3] Results:")
        print("-" * 40)
        print(f"  Success: {result.success}")
        print(f"  Output: {result.schematic_path}")

        # Check for wires in output
        if result.schematic_path and result.schematic_path.exists():
            content = result.schematic_path.read_text()

            # Count wires and labels
            wire_count = content.count("(wire ")
            label_count = content.count("(global_label ") + content.count("(label ")
            junction_count = content.count("(junction ")

            print(f"\n  Wire segments: {wire_count}")
            print(f"  Labels: {label_count}")
            print(f"  Junctions: {junction_count}")

            # Look for specific net labels
            if "VCC_3V3" in content:
                print("  ✓ VCC_3V3 net label found")
            else:
                print("  ✗ VCC_3V3 net label NOT found")

            if "GND" in content:
                print("  ✓ GND net label found")
            else:
                print("  ✗ GND net label NOT found")

            # Show sample wires
            if wire_count > 0:
                print(f"\n  Sample wire definitions:")
                import re
                wires = re.findall(r'\(wire \(pts.*?\)\)', content, re.DOTALL)
                for i, wire in enumerate(wires[:3]):
                    print(f"    {i+1}. {wire[:80]}...")

            # Verdict
            print("\n" + "=" * 40)
            if wire_count > 0:
                print("✓ WIRE CONNECTIONS WORKING!")
            else:
                print("✗ NO WIRES GENERATED - connections may not be routed")
            print("=" * 40)

            return wire_count > 0
        else:
            print("  ERROR: No schematic file generated")
            return False

    except Exception as e:
        print(f"\n  ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = asyncio.run(test_wire_connections())
    sys.exit(0 if success else 1)
