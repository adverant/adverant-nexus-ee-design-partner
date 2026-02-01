#!/usr/bin/env python3
"""
API Wrapper for MAPO Schematic Generation Pipeline.

This script provides a JSON-in, JSON-out interface for the TypeScript API
to call the MAPO schematic generation pipeline.

Usage:
    python api_generate_schematic.py --json '{
        "bom": [...],
        "design_intent": "...",
        "design_name": "...",
        "skip_validation": true
    }'

Or via stdin:
    echo '{"bom": [...], ...}' | python api_generate_schematic.py --stdin
"""

import asyncio
import argparse
import json
import logging
import sys
import tempfile
from pathlib import Path
from typing import Dict, Any, List, Optional

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from mapo_schematic_pipeline import (
    MAPOSchematicPipeline,
    PipelineConfig,
    PipelineResult,
    generate_schematic
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    stream=sys.stderr  # Log to stderr, output JSON to stdout
)
logger = logging.getLogger(__name__)


def create_foc_esc_bom(subsystems: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Create a comprehensive BOM for FOC ESC based on selected subsystems.

    This maps subsystem names to actual component definitions from the
    foc-esc-heavy-lift reference design.
    """
    # Component definitions by subsystem
    SUBSYSTEM_COMPONENTS = {
        "Power Input Stage": [
            {"part_number": "IPB200N25N3 G", "category": "MOSFET", "manufacturer": "Infineon", "value": "250V/66A N-FET", "description": "Reverse polarity protection"},
            {"part_number": "LTC4412ES6", "category": "IC", "manufacturer": "Analog Devices", "value": "Ideal Diode Controller", "description": "OR-ing controller"},
            {"part_number": "SL22_10010", "category": "Thermistor", "manufacturer": "Ametherm", "value": "10R NTC", "description": "Inrush limiter"},
            {"part_number": "744823347", "category": "Inductor", "manufacturer": "Wurth", "value": "470uH CM Choke", "description": "EMI filter"},
            {"part_number": "C5750X7R2A226M", "category": "Capacitor", "manufacturer": "TDK", "value": "22uF/100V", "quantity": 10, "description": "Bulk decoupling"},
            {"part_number": "FG28X7R1E222K", "category": "Capacitor", "manufacturer": "TDK", "value": "2.2nF Y1", "quantity": 2, "description": "Y-cap EMI"},
        ],
        "Gate Driver": [
            {"part_number": "UCC21530ADWRR", "category": "Gate_Driver", "manufacturer": "TI", "value": "6A/9A Isolated", "quantity": 6, "description": "Half-bridge driver"},
            {"part_number": "0603WAF4700T5E", "category": "Resistor", "manufacturer": "UniOhm", "value": "4.7R", "quantity": 12, "description": "Gate resistor"},
            {"part_number": "C2012X7R1E104K", "category": "Capacitor", "manufacturer": "TDK", "value": "100nF/25V", "quantity": 6, "description": "Bootstrap cap"},
            {"part_number": "C3216X7R1E106K", "category": "Capacitor", "manufacturer": "TDK", "value": "10uF/25V", "quantity": 6, "description": "VCC bypass"},
        ],
        "Power Stage": [
            {"part_number": "IMZA65R027M1H", "category": "MOSFET", "manufacturer": "Infineon", "value": "650V/31A SiC", "quantity": 6, "description": "Half-bridge switches"},
            {"part_number": "C5750X7R2A476M", "category": "Capacitor", "manufacturer": "TDK", "value": "47uF/100V", "quantity": 3, "description": "DC bus cap"},
            {"part_number": "1N4148WS", "category": "Diode", "manufacturer": "ON Semi", "value": "100V Fast", "quantity": 6, "description": "Flyback protection"},
        ],
        "MCU Core": [
            {"part_number": "STM32G431CBT6", "category": "MCU", "manufacturer": "ST", "value": "Cortex-M4 170MHz", "description": "Main controller"},
            {"part_number": "ABM8-8.000MHZ", "category": "Crystal", "manufacturer": "Abracon", "value": "8MHz", "description": "HSE crystal"},
            {"part_number": "C2012X7R1E105K", "category": "Capacitor", "manufacturer": "TDK", "value": "1uF/25V", "quantity": 4, "description": "VDD bypass"},
            {"part_number": "C2012X7R1E104K", "category": "Capacitor", "manufacturer": "TDK", "value": "100nF/25V", "quantity": 8, "description": "Decoupling"},
            {"part_number": "0603WAF1002T5E", "category": "Resistor", "manufacturer": "UniOhm", "value": "10k", "quantity": 4, "description": "Pullup/pulldown"},
        ],
        "Current Sensing": [
            {"part_number": "INA240A4PWR", "category": "Amplifier", "manufacturer": "TI", "value": "200V/V Gain", "quantity": 3, "description": "Current sense amp"},
            {"part_number": "WSL3637R0005FEA", "category": "Resistor", "manufacturer": "Vishay", "value": "0.5mR 3W", "quantity": 3, "description": "Shunt resistor"},
            {"part_number": "0603WAF1000T5E", "category": "Resistor", "manufacturer": "UniOhm", "value": "100R", "quantity": 6, "description": "RC filter"},
            {"part_number": "C2012X7R1E103K", "category": "Capacitor", "manufacturer": "TDK", "value": "10nF/25V", "quantity": 6, "description": "RC filter"},
        ],
        "Communication": [
            {"part_number": "TJA1051T/3", "category": "CAN_Transceiver", "manufacturer": "NXP", "value": "CAN FD", "description": "CAN transceiver"},
            {"part_number": "PRTR5V0U2X", "category": "TVS", "manufacturer": "Nexperia", "value": "ESD Protection", "description": "CAN ESD"},
            {"part_number": "0603WAF1200T5E", "category": "Resistor", "manufacturer": "UniOhm", "value": "120R", "description": "CAN termination"},
            {"part_number": "C2012X7R1E104K", "category": "Capacitor", "manufacturer": "TDK", "value": "100nF/25V", "quantity": 2, "description": "CAN bypass"},
            {"part_number": "USB4110-GF-A", "category": "Connector", "manufacturer": "GCT", "value": "USB-C", "description": "Debug connector"},
        ],
    }

    # Build BOM from selected subsystems
    bom = []
    for subsystem in subsystems:
        name = subsystem.get("name", "")
        if name in SUBSYSTEM_COMPONENTS:
            components = SUBSYSTEM_COMPONENTS[name]
            for comp in components:
                bom_item = comp.copy()
                bom_item["subsystem"] = name
                bom.append(bom_item)

    # Add power symbols if any subsystem selected
    if bom:
        bom.extend([
            {"part_number": "VCC", "category": "Power", "value": "VCC", "description": "Power rail"},
            {"part_number": "GND", "category": "Power", "value": "GND", "description": "Ground"},
        ])

    return bom


def create_design_intent(subsystems: List[Dict[str, Any]], project_name: str) -> str:
    """Create a detailed design intent from subsystem selection."""
    subsystem_names = [s.get("name", "Unknown") for s in subsystems]

    intent = f"""
{project_name} - FOC ESC Schematic

This is a Field Oriented Control (FOC) Electronic Speed Controller for brushless motors.

Selected Subsystems:
{chr(10).join(f"- {name}" for name in subsystem_names)}

Design Requirements:
1. Power Input: 48-60V battery input with reverse polarity protection
2. Gate Drivers: Isolated half-bridge drivers with bootstrap supply
3. Power Stage: SiC MOSFETs for high-efficiency switching
4. MCU Core: STM32G4 microcontroller with FOC algorithm
5. Current Sensing: Low-side shunt measurement with differential amplifiers
6. Communication: CAN bus interface for control commands

Key Connections:
- VCC power rail to all IC power pins
- GND reference to all ground pins
- Phase outputs from power stage to motor connector
- Current sense outputs to MCU ADC inputs
- PWM signals from MCU to gate drivers
- CAN bus to external connector

Layout Considerations:
- Keep power stage and gate drivers close together
- Minimize current sense loop area
- Proper decoupling for MCU
- EMI filtering on input power
"""
    return intent.strip()


async def run_generation(
    bom: Optional[List[Dict[str, Any]]] = None,
    subsystems: Optional[List[Dict[str, Any]]] = None,
    design_intent: Optional[str] = None,
    design_name: str = "schematic",
    project_name: str = "FOC ESC",
    skip_validation: bool = True,  # Default skip for faster generation
    output_dir: Optional[str] = None,
    project_id: Optional[str] = None,  # Project ID for NFS export organization
    auto_export: bool = True,  # Enable auto-export to PDF/image and NFS
) -> Dict[str, Any]:
    """
    Run the MAPO schematic generation pipeline.

    Args:
        bom: Explicit BOM list (optional, will be generated from subsystems if not provided)
        subsystems: List of subsystem definitions to include
        design_intent: Natural language design description
        design_name: Name for output file
        project_name: Project name for title block
        skip_validation: Skip MAPO validation loop
        output_dir: Custom output directory

    Returns:
        Dictionary with generation results
    """
    try:
        # Generate BOM from subsystems if not provided
        if bom is None and subsystems:
            bom = create_foc_esc_bom(subsystems)
            logger.info(f"Generated BOM with {len(bom)} components from {len(subsystems)} subsystems")
        elif bom is None:
            bom = []
            logger.warning("No BOM or subsystems provided, generating minimal schematic")

        # Generate design intent if not provided
        if design_intent is None:
            design_intent = create_design_intent(subsystems or [], project_name)

        # Configure output directory
        if output_dir:
            output_path = Path(output_dir)
        else:
            output_path = Path(tempfile.mkdtemp(prefix="mapo_schematic_"))

        output_path.mkdir(parents=True, exist_ok=True)

        logger.info(f"Starting schematic generation: {len(bom)} components")
        logger.info(f"Output directory: {output_path}")

        # Configure pipeline with auto-export to PDF/image and NFS
        config = PipelineConfig(
            output_dir=output_path,
            validation_threshold=0.85,
            max_iterations=3 if not skip_validation else 1,
            # Enable auto-export to PDF/image and NFS sync
            auto_export=auto_export,
            export_pdf=True,
            export_svg=True,
            export_png=True,
            nfs_base_path="/Volumes/Nexus/plugins/ee-design-plugin/artifacts",
            project_id=project_id,
        )

        # Run generation
        result = await generate_schematic(
            bom=bom,
            design_intent=design_intent,
            design_name=design_name,
            skip_validation=skip_validation
        )

        # Read generated schematic content
        schematic_content = ""
        if result.schematic_path and result.schematic_path.exists():
            schematic_content = result.schematic_path.read_text()

        # Build response
        response = {
            "success": result.success,
            "schematic_path": str(result.schematic_path) if result.schematic_path else None,
            "schematic_content": schematic_content,
            "sheets": [
                {"name": s.name, "uuid": s.uuid, "component_count": len(s.symbols)}
                for s in result.sheets
            ],
            "component_count": sum(len(s.symbols) for s in result.sheets),
            "symbols_fetched": result.symbols_fetched,
            "symbols_from_cache": result.symbols_from_cache,
            "symbols_generated": result.symbols_generated,
            "iterations": result.iterations,
            "total_time_seconds": result.total_time_seconds,
            "errors": result.errors,
        }

        # Include validation results if available
        if result.validation_report:
            response["validation"] = {
                "overall_score": result.validation_report.overall_score,
                "passed": result.validation_report.passed,
                "critical_issues": len(result.validation_report.critical_issues),
            }

        # Include export results if auto-export was enabled
        if result.export_result:
            response["export"] = {
                "success": result.export_result.success,
                "pdf_path": str(result.pdf_path) if result.pdf_path else None,
                "svg_path": str(result.svg_path) if result.svg_path else None,
                "png_path": str(result.png_path) if result.png_path else None,
                "nfs_synced": result.nfs_synced,
                "nfs_paths": result.nfs_paths,
                "errors": result.export_result.errors if result.export_result.errors else [],
            }
            if result.nfs_synced:
                logger.info(f"Artifacts synced to NFS: {result.nfs_paths}")

        return response

    except Exception as e:
        logger.error(f"Generation failed: {e}", exc_info=True)
        return {
            "success": False,
            "error": str(e),
            "schematic_content": "",
            "component_count": 0,
            "errors": [str(e)],
        }


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="API Wrapper for MAPO Schematic Generation"
    )
    parser.add_argument(
        "--json",
        type=str,
        help="JSON input string with bom, design_intent, design_name, skip_validation"
    )
    parser.add_argument(
        "--stdin",
        action="store_true",
        help="Read JSON input from stdin"
    )
    parser.add_argument(
        "--pretty",
        action="store_true",
        help="Pretty print JSON output"
    )

    args = parser.parse_args()

    # Get input JSON
    if args.stdin:
        input_json = sys.stdin.read()
    elif args.json:
        input_json = args.json
    else:
        # Default test input
        input_json = json.dumps({
            "subsystems": [
                {"id": "1", "name": "MCU Core", "category": "Control"},
                {"id": "2", "name": "Current Sensing", "category": "Sensing"},
            ],
            "project_name": "Test FOC ESC",
            "design_name": "test_schematic",
            "skip_validation": True,
        })

    try:
        params = json.loads(input_json)
    except json.JSONDecodeError as e:
        result = {"success": False, "error": f"Invalid JSON input: {e}"}
        print(json.dumps(result))
        sys.exit(1)

    # Run async generation
    result = asyncio.run(run_generation(
        bom=params.get("bom"),
        subsystems=params.get("subsystems"),
        design_intent=params.get("design_intent"),
        design_name=params.get("design_name", "schematic"),
        project_name=params.get("project_name", "FOC ESC"),
        skip_validation=params.get("skip_validation", True),
        output_dir=params.get("output_dir"),
        project_id=params.get("project_id"),
        auto_export=params.get("auto_export", True),
    ))

    # Output JSON result to stdout
    indent = 2 if args.pretty else None
    print(json.dumps(result, indent=indent))

    # Exit with appropriate code
    sys.exit(0 if result.get("success", False) else 1)


if __name__ == "__main__":
    main()
