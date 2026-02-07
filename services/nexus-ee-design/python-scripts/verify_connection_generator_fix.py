#!/usr/bin/env python3
"""
Simple verification script for Connection Generator v3.2 architecture fix.

This script performs static analysis of the source code to verify:
1. Prompt asks for logical connections (not physical routing)
2. Response parsing extracts "connections" array (not "wires")
3. Conversion function uses actual values (not placeholders)
4. Validation checks component/pin existence
"""

import re
import sys
from pathlib import Path


def verify_file_changes(file_path: str) -> dict:
    """Verify the connection generator file has correct changes."""
    with open(file_path, 'r') as f:
        content = f.read()

    checks = {}

    # Check 1: File version is v3.2
    checks['version_updated'] = 'Version: 3.2' in content

    # Check 2: Docstring mentions logical connections
    checks['docstring_correct'] = 'LOGICAL circuit connections' in content

    # Check 3: Architecture separation mentioned
    checks['architecture_separation'] = 'Connection Generator → Logical connections' in content

    # Check 4: WireValidator NOT imported
    checks['wire_validator_removed'] = 'from .wire_validator import' not in content

    # Check 5: Prompt asks for logical connections
    prompt_start = content.find('Generate LOGICAL CONNECTIONS')
    prompt_end = content.find('Generate ONLY the JSON output')
    if prompt_start > 0 and prompt_end > prompt_start:
        prompt_section = content[prompt_start:prompt_end]
    else:
        prompt_section = ""

    checks['prompt_logical'] = all([
        'from_ref' in prompt_section,
        'from_pin' in prompt_section,
        'to_ref' in prompt_section,
        'to_pin' in prompt_section,
        'current_amps' in prompt_section,
        'voltage_volts' in prompt_section,
    ])

    # Check 6: Prompt does NOT ask for physical routing in OUTPUT FORMAT
    # Look specifically in the JSON example output format
    json_example_start = prompt_section.find('"connections": [')
    json_example_end = prompt_section.find('}},\n  ],', json_example_start)
    if json_example_start > 0 and json_example_end > json_example_start:
        json_example = prompt_section[json_example_start:json_example_end]
    else:
        json_example = prompt_section

    checks['prompt_no_routing'] = all([
        '"start_point"' not in json_example,
        '"end_point"' not in json_example,
        '"waypoints"' not in json_example,
    ])

    # Check 7: Response parsing looks for "connections" not "wires"
    parse_section = content[content.find('def _call_llm_for_wires'):content.find('def _parse_llm_json')]
    checks['parse_connections'] = '"connections" in connections_json' in parse_section

    # Check 8: Conversion function uses actual values (not placeholders)
    conversion_section = content[content.find('def _convert_wires_to_connections'):content.find('def _validate_logical_connections')]
    checks['no_placeholders'] = all([
        'conn_data.get("from_ref")' in conversion_section,
        'conn_data.get("from_pin")' in conversion_section,
        'conn_data.get("to_ref")' in conversion_section,
        'conn_data.get("to_pin")' in conversion_section,
        'wire.get("from_ref", "U1")' not in conversion_section,  # Old placeholder pattern
    ])

    # Check 9: Validation function exists
    checks['validation_function_exists'] = 'def _validate_logical_connections(' in content

    # Check 10: Validation checks component references
    if 'def _validate_logical_connections(' in content:
        validation_section = content[content.find('def _validate_logical_connections'):content.find('def _get_min_spacing')]
        checks['validation_checks_components'] = all([
            'comp_refs' in validation_section,
            'not found in component list' in validation_section,
        ])
    else:
        checks['validation_checks_components'] = False

    return checks


def main():
    """Run verification."""
    print("\n" + "="*80)
    print("CONNECTION GENERATOR v3.2 ARCHITECTURE FIX VERIFICATION")
    print("="*80 + "\n")

    file_path = Path(__file__).parent / 'agents' / 'connection_generator' / 'connection_generator_agent.py'

    if not file_path.exists():
        print(f"❌ ERROR: File not found: {file_path}")
        return 1

    print(f"Analyzing: {file_path.name}\n")

    checks = verify_file_changes(str(file_path))

    check_descriptions = {
        'version_updated': 'File version updated to v3.2',
        'docstring_correct': 'Docstring mentions logical connections',
        'architecture_separation': 'Architecture separation documented',
        'wire_validator_removed': 'WireValidator import removed',
        'prompt_logical': 'Prompt asks for logical connections (from_ref, from_pin, etc.)',
        'prompt_no_routing': 'Prompt does NOT ask for physical routing (start_point, waypoints)',
        'parse_connections': 'Response parsing looks for "connections" array',
        'no_placeholders': 'Conversion uses actual values (not placeholders)',
        'validation_function_exists': 'Logical validation function exists',
        'validation_checks_components': 'Validation checks component/pin existence',
    }

    all_passed = True
    for check_name, description in check_descriptions.items():
        passed = checks.get(check_name, False)
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status}: {description}")
        if not passed:
            all_passed = False

    print("\n" + "="*80)
    if all_passed:
        print("✅ ALL CHECKS PASSED")
        print("\nThe Connection Generator has been successfully updated to v3.2.")
        print("It now generates LOGICAL connections (pin-to-pin mappings) instead")
        print("of physical wire routing data.")
        print("\nKey Changes:")
        print("  • LLM prompt asks for from_ref/from_pin → to_ref/to_pin")
        print("  • Response parsing extracts 'connections' array")
        print("  • Conversion uses actual LLM values (no placeholders)")
        print("  • Validation checks component/pin existence")
        print("  • Physical routing delegated to Wire Router agent")
    else:
        print("❌ SOME CHECKS FAILED")
        print("\nReview the implementation to ensure all architecture")
        print("changes have been applied correctly.")

    print("="*80 + "\n")

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
