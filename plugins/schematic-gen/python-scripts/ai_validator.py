#!/usr/bin/env python3
"""
AI-First Schematic & PCB Validation

Uses Claude Opus 4.6 for intelligent analysis and decision-making.
Instead of regex parsing, sends design files directly to Claude for:
1. Connectivity analysis
2. Design rule checking
3. Best practice recommendations
4. Error detection and fix suggestions

This is the AI-FIRST approach - Claude makes all decisions.

Usage:
    python ai_validator.py --schematic path/to/file.kicad_sch
    python ai_validator.py --pcb path/to/file.kicad_pcb
    python ai_validator.py --both --schematic sch.kicad_sch --pcb pcb.kicad_pcb
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, Any, Optional
import subprocess

# Try to import anthropic SDK
try:
    import anthropic
    HAS_ANTHROPIC = True
except ImportError:
    HAS_ANTHROPIC = False


def get_claude_client():
    """Get Anthropic client from environment."""
    api_key = os.environ.get('ANTHROPIC_API_KEY')
    if not api_key:
        # Try to read from .env or config
        env_path = Path.home() / '.anthropic' / 'api_key'
        if env_path.exists():
            api_key = env_path.read_text().strip()

    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY not found in environment")

    return anthropic.Anthropic(api_key=api_key)


def read_file_content(path: str, max_lines: int = 2000) -> str:
    """Read file content with size limit."""
    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        lines = f.readlines()

    if len(lines) > max_lines:
        # Truncate large files
        content = ''.join(lines[:max_lines])
        content += f"\n... [TRUNCATED - {len(lines) - max_lines} lines omitted] ..."
    else:
        content = ''.join(lines)

    return content


def validate_with_claude(file_path: str, file_type: str, client) -> Dict[str, Any]:
    """
    Send file to Claude for AI-powered validation.

    Args:
        file_path: Path to schematic or PCB file
        file_type: "schematic" or "pcb"
        client: Anthropic client

    Returns:
        Validation results from Claude
    """
    content = read_file_content(file_path)
    file_name = Path(file_path).name

    if file_type == "schematic":
        system_prompt = """You are an expert electronics engineer and KiCad schematic validator.
Analyze the provided KiCad schematic (.kicad_sch) S-expression file and provide:

1. COMPONENT ANALYSIS:
   - List all components found (reference, value, footprint)
   - Identify component types (resistors, capacitors, ICs, etc.)

2. CONNECTIVITY ANALYSIS:
   - Identify all nets and their connections
   - Find unconnected pins
   - Verify power and ground connections

3. DESIGN ISSUES:
   - Missing decoupling capacitors
   - Missing pull-up/pull-down resistors
   - ESD protection concerns
   - Power sequencing issues

4. BEST PRACTICES:
   - Component value recommendations
   - Layout suggestions
   - Reliability concerns

Return your analysis as structured JSON with these sections.
Be specific - cite actual component references from the schematic."""

    else:  # pcb
        system_prompt = """You are an expert PCB designer and KiCad PCB validator.
Analyze the provided KiCad PCB (.kicad_pcb) S-expression file and provide:

1. BOARD ANALYSIS:
   - Board dimensions and layer count
   - Component count and placement

2. DRC CHECKS:
   - Trace width analysis (especially for power nets)
   - Clearance verification
   - Via sizes
   - Thermal relief

3. ROUTING QUALITY:
   - Identify potential signal integrity issues
   - Check for acute angles (should be 45° or 90°)
   - Ground return paths
   - Power plane connectivity

4. MANUFACTURING:
   - Minimum feature sizes
   - Silkscreen legibility
   - Fiducial markers
   - Assembly concerns

Return your analysis as structured JSON with these sections.
Cite specific coordinates and net names from the PCB."""

    user_message = f"""Analyze this KiCad {file_type} file: {file_name}

FILE CONTENT:
```
{content}
```

Provide comprehensive validation results as JSON."""

    # Call Claude Opus 4.6
    response = client.messages.create(
        model="claude-opus-4-6-20260206",
        max_tokens=8192,
        messages=[
            {
                "role": "user",
                "content": user_message
            }
        ],
        system=system_prompt
    )

    # Extract response text
    response_text = response.content[0].text

    # Try to parse JSON from response
    try:
        # Find JSON block in response
        json_start = response_text.find('{')
        json_end = response_text.rfind('}') + 1
        if json_start >= 0 and json_end > json_start:
            result = json.loads(response_text[json_start:json_end])
        else:
            result = {"raw_analysis": response_text}
    except json.JSONDecodeError:
        result = {"raw_analysis": response_text}

    return {
        "file": file_path,
        "type": file_type,
        "model": "claude-opus-4-6-20260206",
        "analysis": result
    }


def validate_schematic_pcb_pair(schematic_path: str, pcb_path: str, client) -> Dict[str, Any]:
    """
    Validate schematic and PCB together for consistency.

    This is where AI shines - cross-referencing both files.
    """
    sch_content = read_file_content(schematic_path, max_lines=1500)
    pcb_content = read_file_content(pcb_path, max_lines=1500)

    system_prompt = """You are an expert electronics engineer validating a schematic and PCB pair.
Compare both files and verify:

1. CONSISTENCY:
   - All schematic components exist on PCB
   - All nets match between schematic and PCB
   - Footprints are appropriate for components

2. CONNECTIVITY:
   - All schematic connections are routed on PCB
   - No missing traces or unconnected pads
   - Power and ground planes properly connected

3. DESIGN QUALITY:
   - Critical signal integrity (differential pairs, high-speed)
   - Thermal considerations (power components, heat sinks)
   - EMI/EMC concerns

4. ISSUES AND FIXES:
   - List any discrepancies
   - Provide specific fix recommendations

Return comprehensive JSON analysis."""

    user_message = f"""Analyze this schematic and PCB pair for consistency:

SCHEMATIC ({Path(schematic_path).name}):
```
{sch_content}
```

PCB ({Path(pcb_path).name}):
```
{pcb_content}
```

Verify they are consistent and provide validation results as JSON."""

    response = client.messages.create(
        model="claude-opus-4-6-20260206",
        max_tokens=8192,
        messages=[{"role": "user", "content": user_message}],
        system=system_prompt
    )

    response_text = response.content[0].text

    try:
        json_start = response_text.find('{')
        json_end = response_text.rfind('}') + 1
        if json_start >= 0 and json_end > json_start:
            result = json.loads(response_text[json_start:json_end])
        else:
            result = {"raw_analysis": response_text}
    except json.JSONDecodeError:
        result = {"raw_analysis": response_text}

    return {
        "schematic": schematic_path,
        "pcb": pcb_path,
        "model": "claude-opus-4-6-20260206",
        "analysis": result
    }


def validate_without_api(file_path: str, file_type: str) -> Dict[str, Any]:
    """
    Fallback validation when API key is not available.
    Uses Claude Code CLI through subprocess.
    """
    content = read_file_content(file_path, max_lines=500)

    prompt = f"""Analyze this KiCad {file_type} and provide validation:

{content[:5000]}

List:
1. Components found
2. Connectivity issues
3. Design rule violations
4. Recommendations"""

    # This is a placeholder - in practice would use Claude Code MCP or other integration
    return {
        "file": file_path,
        "type": file_type,
        "status": "FALLBACK_MODE",
        "message": "API key not available. Set ANTHROPIC_API_KEY for full AI validation.",
        "basic_stats": {
            "file_size": os.path.getsize(file_path),
            "symbol_count": content.count('(symbol'),
            "wire_count": content.count('(wire'),
            "footprint_count": content.count('(footprint'),
            "segment_count": content.count('(segment')
        }
    }


def main():
    parser = argparse.ArgumentParser(
        description='AI-First validation using Claude Opus 4.6'
    )
    parser.add_argument(
        '--schematic', '-s',
        type=str,
        help='Path to .kicad_sch file'
    )
    parser.add_argument(
        '--pcb', '-p',
        type=str,
        help='Path to .kicad_pcb file'
    )
    parser.add_argument(
        '--both', '-b',
        action='store_true',
        help='Validate schematic and PCB together for consistency'
    )
    parser.add_argument(
        '--json',
        action='store_true',
        help='Output as JSON'
    )
    parser.add_argument(
        '--model',
        type=str,
        default='claude-opus-4-5-20251101',
        help='Claude model to use (default: claude-opus-4-5-20251101)'
    )

    args = parser.parse_args()

    if not args.schematic and not args.pcb:
        parser.error("At least one of --schematic or --pcb is required")

    results = []

    # Try to get Anthropic client
    client = None
    if HAS_ANTHROPIC:
        try:
            client = get_claude_client()
        except Exception as e:
            print(f"Warning: Could not initialize Anthropic client: {e}", file=sys.stderr)

    if args.both and args.schematic and args.pcb:
        # Cross-validate both files
        if client:
            result = validate_schematic_pcb_pair(args.schematic, args.pcb, client)
        else:
            result = {
                "status": "API_UNAVAILABLE",
                "message": "Install anthropic SDK and set ANTHROPIC_API_KEY for AI validation"
            }
        results.append(result)

    else:
        # Validate individual files
        if args.schematic:
            if client:
                result = validate_with_claude(args.schematic, "schematic", client)
            else:
                result = validate_without_api(args.schematic, "schematic")
            results.append(result)

        if args.pcb:
            if client:
                result = validate_with_claude(args.pcb, "pcb", client)
            else:
                result = validate_without_api(args.pcb, "pcb")
            results.append(result)

    # Output results
    if args.json:
        print(json.dumps(results, indent=2))
    else:
        for result in results:
            print("\n" + "="*60)
            print(f"AI Validation: {result.get('file', result.get('schematic', 'N/A'))}")
            print("="*60)

            if 'analysis' in result:
                analysis = result['analysis']
                if isinstance(analysis, dict):
                    for key, value in analysis.items():
                        print(f"\n{key.upper()}:")
                        if isinstance(value, list):
                            for item in value:
                                print(f"  - {item}")
                        elif isinstance(value, dict):
                            for k, v in value.items():
                                print(f"  {k}: {v}")
                        else:
                            print(f"  {value}")
                else:
                    print(analysis)
            else:
                print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
