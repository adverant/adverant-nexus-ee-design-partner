#!/usr/bin/env python3
"""
KiCad API Server - FastAPI endpoint for MAPOS pcbnew operations.

This server provides REST API endpoints for:
- Zone filling
- Net assignment
- DRC validation
- Trace width adjustment
- Full MAPOS optimization

Designed to run in K8s with Xvfb sidecar for headless KiCad operations.

Part of MAPOS (Multi-Agent PCB Optimization System) for the Nexus EE Design Partner plugin.
"""

import os
import sys
import json
import shutil
import subprocess
import tempfile
import asyncio
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime
from enum import Enum

# FastAPI imports
try:
    from fastapi import FastAPI, HTTPException, BackgroundTasks, UploadFile, File
    from fastapi.responses import JSONResponse, FileResponse
    from pydantic import BaseModel, Field
    import uvicorn
except ImportError:
    print("ERROR: FastAPI not installed. Run: pip install fastapi uvicorn python-multipart")
    sys.exit(1)

# ============================================================================
# Configuration
# ============================================================================

# Support both MAPOS_DATA_DIR (legacy) and PCB_DATA_DIR (K8s deployment)
DATA_DIR = Path(os.environ.get('PCB_DATA_DIR', os.environ.get('MAPOS_DATA_DIR', '/data')))
OUTPUT_DIR = Path(os.environ.get('OUTPUT_DIR', os.environ.get('MAPOS_OUTPUT_DIR', '/output')))
OPENROUTER_API_KEY = os.environ.get('OPENROUTER_API_KEY', '')

# Schematic export directory (for PDF/SVG/PNG export via kicad-cli)
SCHEMATIC_DATA_DIR = Path(os.environ.get('SCHEMATIC_DATA_DIR', '/schematic-data'))

# Ensure directories exist
DATA_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
SCHEMATIC_DATA_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================================
# API Models
# ============================================================================

class OperationType(str, Enum):
    ZONE_FILL = "zone_fill"
    FIX_ZONE_NETS = "fix_zone_nets"
    FIX_CLEARANCES = "fix_clearances"
    REMOVE_DANGLING_VIAS = "remove_dangling_vias"
    FIX_DANGLING_TRACKS = "fix_dangling_tracks"  # NEW: Fix dangling track endpoints
    ASSIGN_ORPHAN_NETS = "assign_orphan_nets"
    ADJUST_TRACE_WIDTH = "adjust_trace_width"
    ADJUST_POWER_TRACES = "adjust_power_traces"
    RUN_DRC = "run_drc"
    FULL_OPTIMIZE = "full_optimize"
    LLM_OPTIMIZE = "llm_optimize"
    # New DRC fixers
    FIX_SOLDER_MASK = "fix_solder_mask"
    FIX_SILKSCREEN = "fix_silkscreen"
    FIX_DFM = "fix_dfm"
    # Gaming AI operations (MAP-Elites + Red Queen + Ralph Wiggum)
    GAMING_AI_OPTIMIZE = "gaming_ai_optimize"
    GAMING_AI_QUICK = "gaming_ai_quick"
    UNIFIED_OPTIMIZE = "unified_optimize"


class OperationRequest(BaseModel):
    """Request to run a pcbnew operation."""
    pcb_filename: str = Field(..., description="PCB filename in /data directory")
    operation: OperationType = Field(..., description="Operation to perform")
    params: Dict[str, Any] = Field(default_factory=dict, description="Operation parameters")
    api_key: Optional[str] = Field(default=None, description="User's OpenRouter API key for LLM operations (multi-tenant)")


class OperationResponse(BaseModel):
    """Response from a pcbnew operation."""
    success: bool
    operation: str
    result: Dict[str, Any]
    duration_seconds: float
    timestamp: str


class DRCResult(BaseModel):
    """DRC result summary."""
    total_violations: int
    errors: int
    warnings: int
    unconnected: int
    violations_by_type: Dict[str, int]


class OptimizationRequest(BaseModel):
    """Request for full MAPOS optimization."""
    pcb_filename: str = Field(..., description="PCB filename in /data directory")
    target_violations: int = Field(default=100, description="Target violation count")
    max_iterations: int = Field(default=5, description="Maximum optimization iterations")
    use_llm: bool = Field(default=False, description="Use LLM-guided optimization")
    api_key: Optional[str] = Field(default=None, description="User's OpenRouter API key for multi-tenant LLM access")


class JobStatus(BaseModel):
    """Background job status."""
    job_id: str
    status: str  # pending, running, completed, failed
    progress: int
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


class GamingAIRequest(BaseModel):
    """Request for Gaming AI optimization (MAP-Elites + Red Queen + Ralph Wiggum)."""
    pcb_filename: str = Field(..., description="PCB filename in /data directory")
    target_violations: int = Field(default=100, description="Target DRC violations to reach")
    rq_rounds: int = Field(default=10, description="Red Queen evolution rounds")
    max_stagnation: int = Field(default=15, description="Max rounds without improvement before stopping")
    use_llm: bool = Field(default=True, description="Use LLM for intelligent guidance")
    api_key: Optional[str] = Field(default=None, description="OpenRouter API key for LLM calls")


class UnifiedOptimizerRequest(BaseModel):
    """Request for unified optimization (Base + LLM + Gaming AI)."""
    pcb_filename: str = Field(..., description="PCB filename in /data directory")
    mode: str = Field(default="hybrid", description="Mode: base, llm, gaming_ai, or hybrid")
    target_violations: int = Field(default=100, description="Target DRC violations")
    api_key: Optional[str] = Field(default=None, description="OpenRouter API key")


# ============================================================================
# Schematic Export Models (for PDF/SVG/PNG export via kicad-cli)
# ============================================================================

class SchematicExportFormat(str, Enum):
    """Supported schematic export formats."""
    PDF = "pdf"
    SVG = "svg"
    PNG = "png"


class SchematicExportRequest(BaseModel):
    """Request to export a schematic to PDF/SVG/PNG."""
    schematic_content: str = Field(..., description="Full .kicad_sch file content")
    export_format: SchematicExportFormat = Field(..., description="Export format: pdf, svg, or png")
    design_name: str = Field(default="schematic", description="Base name for output file")
    # PNG-specific options
    dpi: int = Field(default=300, description="DPI for PNG export (default: 300)")
    # PDF/SVG options
    black_and_white: bool = Field(default=False, description="Export in black and white")
    no_background: bool = Field(default=False, description="Export with transparent background (SVG only)")


class SchematicExportResponse(BaseModel):
    """Response from schematic export."""
    success: bool
    export_id: str = Field(..., description="Unique export ID for download")
    format: str
    filename: str
    size_bytes: int
    download_url: str
    errors: List[str] = Field(default_factory=list)
    duration_seconds: float


# ============================================================================
# Application
# ============================================================================

app = FastAPI(
    title="MAPOS KiCad API",
    description="PCB optimization API for Nexus EE Design Partner plugin",
    version="1.0.0"
)

# Background jobs storage
jobs: Dict[str, JobStatus] = {}


# ============================================================================
# Helper Functions
# ============================================================================

def run_python_script(script_name: str, args: List[str], timeout: int = 300, env: Dict[str, str] = None) -> Dict[str, Any]:
    """Run a Python script and return the result.

    Args:
        script_name: Name of the Python script to run
        args: Command line arguments for the script
        timeout: Execution timeout in seconds
        env: Custom environment variables (for multi-tenant API keys)
    """
    script_path = Path(__file__).parent / script_name

    if not script_path.exists():
        # Try /app directory (K8s deployment)
        script_path = Path('/app') / script_name

    if not script_path.exists():
        return {'success': False, 'error': f'Script not found: {script_name}'}

    # Use custom env if provided, otherwise default
    run_env = env if env else {**os.environ, 'PYTHONUNBUFFERED': '1'}
    if env and 'PYTHONUNBUFFERED' not in run_env:
        run_env['PYTHONUNBUFFERED'] = '1'

    try:
        result = subprocess.run(
            ['python3', str(script_path)] + args,
            capture_output=True,
            text=True,
            timeout=timeout,
            env=run_env
        )

        # Try to parse JSON output
        output = result.stdout.strip()
        try:
            # Find JSON in output
            if output.startswith('{'):
                return json.loads(output)
            else:
                json_start = output.rfind('\n{')
                if json_start >= 0:
                    return json.loads(output[json_start + 1:])
        except json.JSONDecodeError:
            pass

        return {
            'success': result.returncode == 0,
            'output': output,
            'error': result.stderr if result.returncode != 0 else None
        }

    except subprocess.TimeoutExpired:
        return {'success': False, 'error': 'Operation timed out'}
    except Exception as e:
        return {'success': False, 'error': str(e)}


def run_kicad_cli(command: List[str], timeout: int = 120) -> Dict[str, Any]:
    """Run kicad-cli command."""
    try:
        result = subprocess.run(
            ['kicad-cli'] + command,
            capture_output=True,
            text=True,
            timeout=timeout
        )
        return {
            'success': result.returncode == 0,
            'output': result.stdout,
            'error': result.stderr if result.returncode != 0 else None
        }
    except Exception as e:
        return {'success': False, 'error': str(e)}


# ============================================================================
# API Endpoints
# ============================================================================

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "timestamp": datetime.utcnow().isoformat()}


# ============================================================================
# Symbol Library Endpoints (for schematic generation without local KiCad)
# ============================================================================

KICAD_SYMBOLS_PATH = Path("/usr/share/kicad/symbols")

@app.get("/v1/symbols")
async def list_symbol_libraries():
    """List available KiCad symbol libraries."""
    if not KICAD_SYMBOLS_PATH.exists():
        raise HTTPException(status_code=404, detail="KiCad symbols directory not found")

    libraries = [f.stem for f in KICAD_SYMBOLS_PATH.glob("*.kicad_sym")]
    return {
        "libraries": sorted(libraries),
        "count": len(libraries),
        "path": str(KICAD_SYMBOLS_PATH)
    }


@app.get("/v1/symbols/{library_name}")
async def get_symbol_library(library_name: str):
    """Get a full KiCad symbol library file content."""
    # Remove .kicad_sym extension if provided
    lib_name = library_name.replace(".kicad_sym", "")
    lib_path = KICAD_SYMBOLS_PATH / f"{lib_name}.kicad_sym"

    if not lib_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Library '{lib_name}' not found. Use /v1/symbols to list available libraries."
        )

    try:
        content = lib_path.read_text(encoding='utf-8')
        return {
            "library": lib_name,
            "content": content,
            "size": len(content)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading library: {str(e)}")


@app.get("/v1/symbols/{library_name}/symbol/{symbol_name}")
async def get_symbol(library_name: str, symbol_name: str):
    """Extract a specific symbol from a KiCad library."""
    lib_name = library_name.replace(".kicad_sym", "")
    lib_path = KICAD_SYMBOLS_PATH / f"{lib_name}.kicad_sym"

    if not lib_path.exists():
        raise HTTPException(status_code=404, detail=f"Library '{lib_name}' not found")

    try:
        content = lib_path.read_text(encoding='utf-8')

        # Find the symbol in the library
        # Symbols start with (symbol "name" and end with matching parenthesis
        import re

        # Escape the symbol name for regex
        escaped_name = re.escape(symbol_name)

        # Match the symbol definition - handles nested parentheses
        pattern = rf'\(symbol\s+"{escaped_name}"'
        match = re.search(pattern, content)

        if not match:
            # Try without quotes
            pattern = rf'\(symbol\s+{escaped_name}\s'
            match = re.search(pattern, content)

        if not match:
            raise HTTPException(
                status_code=404,
                detail=f"Symbol '{symbol_name}' not found in library '{lib_name}'"
            )

        # Extract the full symbol S-expression
        start = match.start()
        depth = 0
        end = start

        for i, char in enumerate(content[start:], start):
            if char == '(':
                depth += 1
            elif char == ')':
                depth -= 1
                if depth == 0:
                    end = i + 1
                    break

        symbol_sexp = content[start:end]

        return {
            "library": lib_name,
            "symbol": symbol_name,
            "sexp": symbol_sexp,
            "size": len(symbol_sexp)
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error extracting symbol: {str(e)}")


# ============================================================================
# Schematic Export Endpoints (PDF/SVG/PNG via kicad-cli)
# ============================================================================

@app.post("/v1/schematic/export", response_model=SchematicExportResponse)
async def export_schematic(request: SchematicExportRequest):
    """
    Export a KiCad schematic to PDF, SVG, or PNG format.

    This endpoint:
    1. Receives schematic content as a string
    2. Saves it to a temporary .kicad_sch file
    3. Runs kicad-cli sch export {format}
    4. Returns download URL for the exported file

    For PNG export, uses ImageMagick to convert from SVG if direct PNG export fails.
    """
    import uuid
    start_time = datetime.utcnow()
    export_id = str(uuid.uuid4())[:8]
    errors = []

    # Create export directory
    export_dir = SCHEMATIC_DATA_DIR / export_id
    export_dir.mkdir(parents=True, exist_ok=True)

    # Save schematic content to file
    sch_filename = f"{request.design_name}.kicad_sch"
    sch_path = export_dir / sch_filename

    try:
        sch_path.write_text(request.schematic_content, encoding='utf-8')
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save schematic: {str(e)}")

    # Determine output filename
    format_ext = request.export_format.value
    output_filename = f"{request.design_name}.{format_ext}"
    output_path = export_dir / output_filename

    # Build kicad-cli command
    if request.export_format == SchematicExportFormat.PDF:
        cmd = ['sch', 'export', 'pdf', '-o', str(output_path)]
        if request.black_and_white:
            cmd.append('--black-and-white')
        cmd.append(str(sch_path))

    elif request.export_format == SchematicExportFormat.SVG:
        cmd = ['sch', 'export', 'svg', '-o', str(export_dir)]  # SVG outputs to directory
        if request.black_and_white:
            cmd.append('--black-and-white')
        if request.no_background:
            cmd.append('--no-background-color')
        cmd.append(str(sch_path))
        # SVG export creates file with sheet name, we'll rename it

    elif request.export_format == SchematicExportFormat.PNG:
        # KiCad doesn't have direct PNG export for schematics
        # We'll export SVG first, then convert with ImageMagick
        svg_path = export_dir / f"{request.design_name}.svg"
        cmd = ['sch', 'export', 'svg', '-o', str(export_dir)]
        if request.black_and_white:
            cmd.append('--black-and-white')
        cmd.append(str(sch_path))

    # Run kicad-cli export
    cli_result = run_kicad_cli(cmd, timeout=60)

    if not cli_result.get('success'):
        error_msg = cli_result.get('error', 'Unknown export error')
        errors.append(f"kicad-cli error: {error_msg}")
        # Don't fail yet for PNG - we might still convert
        if request.export_format != SchematicExportFormat.PNG:
            raise HTTPException(status_code=500, detail=f"Export failed: {error_msg}")

    # Handle SVG renaming (kicad-cli uses sheet name)
    if request.export_format == SchematicExportFormat.SVG:
        svg_files = list(export_dir.glob("*.svg"))
        if svg_files:
            # Rename first SVG to our desired name
            svg_files[0].rename(output_path)

    # Handle PNG conversion from SVG
    if request.export_format == SchematicExportFormat.PNG:
        svg_files = list(export_dir.glob("*.svg"))
        if svg_files:
            svg_source = svg_files[0]
            # Convert SVG to PNG using ImageMagick
            try:
                convert_result = subprocess.run(
                    ['convert', '-density', str(request.dpi), str(svg_source), str(output_path)],
                    capture_output=True,
                    text=True,
                    timeout=60
                )
                if convert_result.returncode != 0:
                    errors.append(f"ImageMagick conversion error: {convert_result.stderr}")
                # Clean up SVG
                svg_source.unlink()
            except FileNotFoundError:
                errors.append("ImageMagick not installed - PNG conversion unavailable")
            except Exception as e:
                errors.append(f"PNG conversion failed: {str(e)}")

    # Check if output file exists
    if not output_path.exists():
        raise HTTPException(
            status_code=500,
            detail=f"Export failed - output file not created. Errors: {'; '.join(errors)}"
        )

    duration = (datetime.utcnow() - start_time).total_seconds()

    return SchematicExportResponse(
        success=len(errors) == 0,
        export_id=export_id,
        format=request.export_format.value,
        filename=output_filename,
        size_bytes=output_path.stat().st_size,
        download_url=f"/v1/schematic/download/{export_id}/{output_filename}",
        errors=errors,
        duration_seconds=duration
    )


@app.get("/v1/schematic/download/{export_id}/{filename}")
async def download_schematic_export(export_id: str, filename: str):
    """
    Download an exported schematic file.

    Args:
        export_id: The export ID returned from /v1/schematic/export
        filename: The filename to download (e.g., "schematic.pdf")

    Returns:
        The exported file as a download
    """
    export_dir = SCHEMATIC_DATA_DIR / export_id
    file_path = export_dir / filename

    if not export_dir.exists():
        raise HTTPException(status_code=404, detail=f"Export not found: {export_id}")

    if not file_path.exists():
        raise HTTPException(status_code=404, detail=f"File not found: {filename}")

    # Determine media type
    media_types = {
        '.pdf': 'application/pdf',
        '.svg': 'image/svg+xml',
        '.png': 'image/png',
    }
    suffix = Path(filename).suffix.lower()
    media_type = media_types.get(suffix, 'application/octet-stream')

    return FileResponse(
        path=str(file_path),
        filename=filename,
        media_type=media_type
    )


@app.delete("/v1/schematic/export/{export_id}")
async def cleanup_schematic_export(export_id: str):
    """
    Clean up an export directory to free disk space.

    Args:
        export_id: The export ID to clean up
    """
    export_dir = SCHEMATIC_DATA_DIR / export_id

    if not export_dir.exists():
        raise HTTPException(status_code=404, detail=f"Export not found: {export_id}")

    try:
        shutil.rmtree(export_dir)
        return {"success": True, "message": f"Cleaned up export {export_id}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Cleanup failed: {str(e)}")


@app.get("/v1/operations")
async def list_operations():
    """List available pcbnew operations."""
    return {
        "operations": [
            {
                "name": op.value,
                "description": {
                    "zone_fill": "Fill all copper zones using ZONE_FILLER API",
                    "fix_zone_nets": "Correct zone net assignments",
                    "fix_clearances": "Update design settings clearances",
                    "remove_dangling_vias": "Remove vias not connected to tracks",
                    "fix_dangling_tracks": "Fix tracks with dangling endpoints (extend or remove)",
                    "assign_orphan_nets": "Assign nets to unconnected pads",
                    "adjust_trace_width": "Modify trace widths for a specific net",
                    "adjust_power_traces": "Widen all power traces",
                    "run_drc": "Run DRC and return violations",
                    "full_optimize": "Run full MAPOS optimization",
                    "llm_optimize": "Run LLM-guided optimization",
                    "fix_solder_mask": "Fix solder mask bridge violations (via tenting + bridge allowance)",
                    "fix_silkscreen": "Fix silk over copper violations (move graphics to Fab layer)",
                    "fix_dfm": "Run full DFM fix pipeline (mask + silk + footprint fixes)"
                }.get(op.value, "")
            }
            for op in OperationType
        ]
    }


@app.post("/v1/operation", response_model=OperationResponse)
async def run_operation(request: OperationRequest):
    """Run a single pcbnew operation."""
    start_time = datetime.utcnow()

    pcb_path = DATA_DIR / request.pcb_filename
    if not pcb_path.exists():
        raise HTTPException(status_code=404, detail=f"PCB file not found: {request.pcb_filename}")

    result = {}

    try:
        if request.operation == OperationType.ZONE_FILL:
            result = run_python_script('kicad_zone_filler.py', [str(pcb_path), '--json'])

        elif request.operation == OperationType.FIX_ZONE_NETS:
            result = run_python_script('kicad_pcb_fixer.py', ['zone-nets', str(pcb_path), '--json'])

        elif request.operation == OperationType.FIX_CLEARANCES:
            result = run_python_script('kicad_pcb_fixer.py', ['design-settings', str(pcb_path), '--json'])

        elif request.operation == OperationType.REMOVE_DANGLING_VIAS:
            result = run_python_script('kicad_pcb_fixer.py', ['dangling-vias', str(pcb_path), '--json'])

        elif request.operation == OperationType.FIX_DANGLING_TRACKS:
            # Fix dangling track endpoints - extend to nearest connection or remove
            strategy = request.params.get('strategy', 'smart')  # smart, extend, remove, trim
            max_extension_mm = request.params.get('max_extension_mm', 5.0)

            if strategy == 'extend':
                result = run_python_script('kicad_dangling_track_fixer.py',
                                           [str(pcb_path), '--extend',
                                            '--max-extension-mm', str(max_extension_mm),
                                            '--json'])
            elif strategy == 'remove':
                result = run_python_script('kicad_dangling_track_fixer.py',
                                           [str(pcb_path), '--remove', '--json'])
            elif strategy == 'trim':
                result = run_python_script('kicad_dangling_track_fixer.py',
                                           [str(pcb_path), '--trim', '--json'])
            else:  # smart: extend first, then trim remaining
                # First try to extend
                extend_result = run_python_script('kicad_dangling_track_fixer.py',
                                                  [str(pcb_path), '--extend',
                                                   '--max-extension-mm', str(max_extension_mm),
                                                   '--json'])
                # Then trim what couldn't be extended
                trim_result = run_python_script('kicad_dangling_track_fixer.py',
                                                [str(pcb_path), '--trim', '--json'])
                result = {
                    'success': extend_result.get('success', False) or trim_result.get('success', False),
                    'strategy': 'smart',
                    'extend': extend_result,
                    'trim': trim_result,
                    'total_fixed': extend_result.get('tracks_extended', 0) + trim_result.get('fully_removed', 0),
                }

        elif request.operation == OperationType.ASSIGN_ORPHAN_NETS:
            result = run_python_script('kicad_net_assigner.py', [str(pcb_path), '--json'])

        elif request.operation == OperationType.ADJUST_TRACE_WIDTH:
            net_name = request.params.get('net_name', 'GND')
            width_mm = request.params.get('width_mm', 2.0)
            result = run_python_script('kicad_trace_adjuster.py',
                                       ['net', str(pcb_path), net_name, str(width_mm), '--json'])

        elif request.operation == OperationType.ADJUST_POWER_TRACES:
            width_mm = request.params.get('width_mm', 2.0)
            result = run_python_script('kicad_trace_adjuster.py',
                                       ['power', str(pcb_path), '--width', str(width_mm), '--json'])

        elif request.operation == OperationType.RUN_DRC:
            with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
                drc_output = f.name

            cli_result = run_kicad_cli(['pcb', 'drc', '--format', 'json', '--output', drc_output, str(pcb_path)])

            if Path(drc_output).exists():
                with open(drc_output) as f:
                    drc_data = json.load(f)
                Path(drc_output).unlink()

                violations = drc_data.get('violations', [])
                unconnected = drc_data.get('unconnected_items', [])

                by_type = {}
                errors = warnings = 0
                for v in violations:
                    vtype = v.get('type', 'unknown')
                    by_type[vtype] = by_type.get(vtype, 0) + 1
                    if v.get('severity') == 'error':
                        errors += 1
                    else:
                        warnings += 1

                result = {
                    'success': True,
                    'total_violations': len(violations) + len(unconnected),
                    'errors': errors,
                    'warnings': warnings,
                    'unconnected': len(unconnected),
                    'violations_by_type': by_type
                }
            else:
                result = {'success': False, 'error': cli_result.get('error', 'DRC failed')}

        elif request.operation == OperationType.FULL_OPTIMIZE:
            target = request.params.get('target_violations', 100)
            iterations = request.params.get('max_iterations', 5)
            result = run_python_script('mapos_pcb_optimizer.py',
                                       [str(pcb_path), '--target', str(target), '--iterations', str(iterations)])

        elif request.operation == OperationType.LLM_OPTIMIZE:
            # Multi-tenant: Use user's API key if provided, otherwise fall back to server key
            user_api_key = request.api_key or OPENROUTER_API_KEY
            if not user_api_key:
                result = {'success': False, 'error': 'No API key provided. Set OPENROUTER_API_KEY or pass api_key in request.'}
            else:
                iterations = request.params.get('max_iterations', 3)
                # Pass API key via environment for subprocess
                env = {**os.environ, 'OPENROUTER_API_KEY': user_api_key}
                result = run_python_script('llm_pcb_fixer.py',
                                           [str(pcb_path), '--max-iterations', str(iterations)],
                                           env=env)

        elif request.operation == OperationType.FIX_SOLDER_MASK:
            # Fix solder mask bridge violations via tenting and bridge allowance
            pitch_threshold = request.params.get('pitch_threshold', 0.5)
            mask_expansion = request.params.get('mask_expansion', 0.0508)
            result = run_python_script('kicad_mask_fixer.py',
                                       [str(pcb_path), '--no-backup',
                                        '--pitch-threshold', str(pitch_threshold),
                                        '--mask-expansion', str(mask_expansion),
                                        '--json'])

        elif request.operation == OperationType.FIX_SILKSCREEN:
            # Fix silk over copper by moving graphics to Fab layer
            offset = request.params.get('offset', 1.5)
            result = run_python_script('kicad_silk_fixer.py',
                                       [str(pcb_path), '--no-backup',
                                        '--offset', str(offset),
                                        '--json'])

        elif request.operation == OperationType.FIX_DFM:
            # Run full DFM fix pipeline (mask + silk + footprint fixes)
            mask_margin = request.params.get('mask_margin', -0.03)
            silk_offset = request.params.get('silk_offset', 1.5)
            result = run_python_script('footprint_dfm_fixer.py',
                                       [str(pcb_path),
                                        '--mask-margin', str(mask_margin),
                                        '--silk-offset', str(silk_offset)])

        elif request.operation == OperationType.GAMING_AI_OPTIMIZE:
            # Full Gaming AI optimization (MAP-Elites + Red Queen + Ralph Wiggum)
            user_api_key = request.api_key or OPENROUTER_API_KEY
            if not user_api_key:
                result = {'success': False, 'error': 'OpenRouter API key required for Gaming AI'}
            else:
                try:
                    # Import Gaming AI integration
                    sys.path.insert(0, str(Path(__file__).parent))
                    from gaming_ai.integration import MAPOSRQOptimizer, MAPOSRQConfig

                    config = MAPOSRQConfig(
                        target_violations=request.params.get('target_violations', 100),
                        rq_rounds=request.params.get('rq_rounds', 10),
                        max_stagnation=request.params.get('max_stagnation', 15),
                        use_llm=True,
                        use_neural_networks=False,
                        openrouter_api_key=user_api_key,
                    )

                    optimizer = MAPOSRQOptimizer(str(pcb_path), config=config)

                    # Run async optimizer (we're already in an async context)
                    opt_result = await optimizer.optimize()
                    # Calculate improvement percentage
                    improvement_pct = (opt_result.improvement / opt_result.initial_violations * 100) if opt_result.initial_violations > 0 else 0.0
                    result = {
                        'success': opt_result.status.name in ['SUCCESS', 'PARTIAL'],
                        'status': opt_result.status.name,
                        'initial_violations': opt_result.initial_violations,
                        'final_violations': opt_result.final_violations,
                        'improvement': opt_result.improvement,
                        'improvement_pct': improvement_pct,
                        'red_queen_rounds': opt_result.red_queen_rounds,
                        'champions_found': len(opt_result.champions) if opt_result.champions else 0,
                    }
                except ImportError as e:
                    result = {'success': False, 'error': f'Gaming AI module not available: {e}'}
                except Exception as e:
                    result = {'success': False, 'error': f'Gaming AI error: {e}'}

        elif request.operation == OperationType.GAMING_AI_QUICK:
            # Quick Gaming AI (fewer rounds for faster results)
            user_api_key = request.api_key or OPENROUTER_API_KEY
            if not user_api_key:
                result = {'success': False, 'error': 'OpenRouter API key required for Gaming AI'}
            else:
                try:
                    sys.path.insert(0, str(Path(__file__).parent))
                    from gaming_ai.integration import MAPOSRQOptimizer, MAPOSRQConfig

                    config = MAPOSRQConfig(
                        target_violations=request.params.get('target_violations', 100),
                        rq_rounds=3,  # Quick mode - fewer rounds
                        max_stagnation=5,
                        use_llm=True,
                        use_neural_networks=False,
                        openrouter_api_key=user_api_key,
                    )

                    optimizer = MAPOSRQOptimizer(str(pcb_path), config=config)

                    # Run async optimizer (we're already in an async context)
                    opt_result = await optimizer.optimize()
                    # Calculate improvement percentage
                    improvement_pct = (opt_result.improvement / opt_result.initial_violations * 100) if opt_result.initial_violations > 0 else 0.0
                    result = {
                        'success': opt_result.status.name in ['SUCCESS', 'PARTIAL'],
                        'status': opt_result.status.name,
                        'initial_violations': opt_result.initial_violations,
                        'final_violations': opt_result.final_violations,
                        'improvement': opt_result.improvement,
                        'improvement_pct': improvement_pct,
                    }
                except ImportError as e:
                    result = {'success': False, 'error': f'Gaming AI module not available: {e}'}
                except Exception as e:
                    result = {'success': False, 'error': f'Gaming AI error: {e}'}

        elif request.operation == OperationType.UNIFIED_OPTIMIZE:
            # Unified optimizer (Base + LLM + Gaming AI)
            user_api_key = request.api_key or OPENROUTER_API_KEY
            mode = request.params.get('mode', 'hybrid')
            try:
                sys.path.insert(0, str(Path(__file__).parent))
                from unified_optimizer import UnifiedMAPOSOptimizer, OptimizationMode

                mode_map = {
                    'base': OptimizationMode.BASE,
                    'llm': OptimizationMode.LLM,
                    'gaming_ai': OptimizationMode.GAMING_AI,
                    'hybrid': OptimizationMode.HYBRID,
                }
                opt_mode = mode_map.get(mode, OptimizationMode.HYBRID)

                optimizer = UnifiedMAPOSOptimizer(
                    str(pcb_path),
                    mode=opt_mode,
                    target_violations=request.params.get('target_violations', 100),
                    api_key=user_api_key,
                )

                # Run async optimizer (we're already in an async context)
                opt_result = await optimizer.optimize()
                result = {
                    'success': opt_result.success,
                    'mode': opt_result.mode,
                    'initial_violations': opt_result.initial_violations,
                    'final_violations': opt_result.final_violations,
                    'improvement': opt_result.improvement,
                    'improvement_pct': opt_result.improvement_pct,
                    'phases': opt_result.phases,
                    'duration_seconds': opt_result.duration_seconds,
                }
            except ImportError as e:
                result = {'success': False, 'error': f'Unified optimizer not available: {e}'}
            except Exception as e:
                result = {'success': False, 'error': f'Unified optimizer error: {e}'}

        else:
            raise HTTPException(status_code=400, detail=f"Unknown operation: {request.operation}")

    except Exception as e:
        result = {'success': False, 'error': str(e)}

    duration = (datetime.utcnow() - start_time).total_seconds()

    return OperationResponse(
        success=result.get('success', False),
        operation=request.operation.value,
        result=result,
        duration_seconds=duration,
        timestamp=start_time.isoformat()
    )


@app.post("/v1/optimize")
async def optimize_pcb(request: OptimizationRequest, background_tasks: BackgroundTasks):
    """Start a full MAPOS optimization job."""
    import uuid

    job_id = str(uuid.uuid4())[:8]

    jobs[job_id] = JobStatus(
        job_id=job_id,
        status="pending",
        progress=0
    )

    async def run_optimization():
        jobs[job_id].status = "running"

        try:
            pcb_path = DATA_DIR / request.pcb_filename
            if not pcb_path.exists():
                jobs[job_id].status = "failed"
                jobs[job_id].error = f"PCB file not found: {request.pcb_filename}"
                return

            # Multi-tenant: Use user's API key if provided, otherwise fall back to server key
            user_api_key = request.api_key or OPENROUTER_API_KEY
            if request.use_llm and user_api_key:
                env = {**os.environ, 'OPENROUTER_API_KEY': user_api_key}
                result = run_python_script('llm_pcb_fixer.py', [
                    str(pcb_path),
                    '--max-iterations', str(request.max_iterations)
                ], timeout=600, env=env)
            elif request.use_llm and not user_api_key:
                jobs[job_id].status = "failed"
                jobs[job_id].error = "No API key provided for LLM optimization. Pass api_key in request or set OPENROUTER_API_KEY."
                return
            else:
                result = run_python_script('mapos_pcb_optimizer.py', [
                    str(pcb_path),
                    '--target', str(request.target_violations),
                    '--iterations', str(request.max_iterations)
                ], timeout=600)

            jobs[job_id].status = "completed" if result.get('success') else "failed"
            jobs[job_id].result = result
            jobs[job_id].progress = 100

        except Exception as e:
            jobs[job_id].status = "failed"
            jobs[job_id].error = str(e)

    background_tasks.add_task(run_optimization)

    return {"job_id": job_id, "status": "pending"}


@app.get("/v1/job/{job_id}", response_model=JobStatus)
async def get_job_status(job_id: str):
    """Get status of a background optimization job."""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")
    return jobs[job_id]


@app.post("/v1/gaming-ai/optimize")
async def gaming_ai_optimize(request: GamingAIRequest, background_tasks: BackgroundTasks):
    """
    Start Gaming AI optimization with MAP-Elites + Red Queen + Ralph Wiggum.

    This is the full Gaming AI pipeline for maximum optimization quality.
    Uses evolutionary algorithms and LLM guidance to optimize PCB layout.
    """
    import uuid
    import hashlib

    pcb_path = DATA_DIR / request.pcb_filename
    if not pcb_path.exists():
        raise HTTPException(status_code=404, detail=f"PCB not found: {request.pcb_filename}")

    # Generate job ID
    job_id = hashlib.md5(f"{request.pcb_filename}:{datetime.utcnow()}".encode()).hexdigest()[:8]
    jobs[job_id] = JobStatus(job_id=job_id, status="pending", progress=0)

    async def run_gaming_ai():
        try:
            jobs[job_id].status = "running"
            jobs[job_id].progress = 5

            user_api_key = request.api_key or OPENROUTER_API_KEY
            if request.use_llm and not user_api_key:
                jobs[job_id].status = "failed"
                jobs[job_id].error = "OpenRouter API key required for LLM-guided Gaming AI"
                return

            # Import Gaming AI
            sys.path.insert(0, str(Path(__file__).parent))
            from gaming_ai.integration import MAPOSRQOptimizer, MAPOSRQConfig

            config = MAPOSRQConfig(
                target_violations=request.target_violations,
                rq_rounds=request.rq_rounds,
                max_stagnation=request.max_stagnation,
                use_llm=request.use_llm,
                use_neural_networks=False,
                openrouter_api_key=user_api_key,
            )

            jobs[job_id].progress = 10

            optimizer = MAPOSRQOptimizer(str(pcb_path), config=config)
            result = await optimizer.optimize()

            jobs[job_id].status = "completed" if result.status.name in ['SUCCESS', 'PARTIAL'] else "failed"
            jobs[job_id].progress = 100
            jobs[job_id].result = {
                'status': result.status.name,
                'initial_violations': result.initial_violations,
                'final_violations': result.final_violations,
                'improvement': result.improvement,
                'improvement_pct': result.improvement_pct,
                'red_queen_rounds': result.red_queen_rounds,
                'champions_found': len(result.champions) if result.champions else 0,
                'best_champion': result.champions[0] if result.champions else None,
            }

        except ImportError as e:
            jobs[job_id].status = "failed"
            jobs[job_id].error = f"Gaming AI module not available: {e}"
        except Exception as e:
            jobs[job_id].status = "failed"
            jobs[job_id].error = str(e)

    background_tasks.add_task(run_gaming_ai)
    return {"job_id": job_id, "status": "pending", "message": "Gaming AI optimization started"}


@app.post("/v1/unified/optimize")
async def unified_optimize(request: UnifiedOptimizerRequest, background_tasks: BackgroundTasks):
    """
    Start unified optimization (Base + LLM + Gaming AI).

    Modes:
    - base: Basic MAPOS (zone fill, net assignment, etc.)
    - llm: LLM-guided fixing via Claude Opus 4.6
    - gaming_ai: MAP-Elites + Red Queen + Ralph Wiggum
    - hybrid: Base -> LLM -> Gaming AI (most thorough)
    """
    import hashlib

    pcb_path = DATA_DIR / request.pcb_filename
    if not pcb_path.exists():
        raise HTTPException(status_code=404, detail=f"PCB not found: {request.pcb_filename}")

    job_id = hashlib.md5(f"{request.pcb_filename}:{datetime.utcnow()}".encode()).hexdigest()[:8]
    jobs[job_id] = JobStatus(job_id=job_id, status="pending", progress=0)

    async def run_unified():
        try:
            jobs[job_id].status = "running"
            jobs[job_id].progress = 5

            user_api_key = request.api_key or OPENROUTER_API_KEY

            sys.path.insert(0, str(Path(__file__).parent))
            from unified_optimizer import UnifiedMAPOSOptimizer, OptimizationMode

            mode_map = {
                'base': OptimizationMode.BASE,
                'llm': OptimizationMode.LLM,
                'gaming_ai': OptimizationMode.GAMING_AI,
                'hybrid': OptimizationMode.HYBRID,
            }
            opt_mode = mode_map.get(request.mode, OptimizationMode.HYBRID)

            optimizer = UnifiedMAPOSOptimizer(
                str(pcb_path),
                mode=opt_mode,
                target_violations=request.target_violations,
                api_key=user_api_key,
            )

            jobs[job_id].progress = 10
            result = await optimizer.optimize()

            jobs[job_id].status = "completed" if result.success else "failed"
            jobs[job_id].progress = 100
            jobs[job_id].result = {
                'mode': result.mode,
                'initial_violations': result.initial_violations,
                'final_violations': result.final_violations,
                'improvement': result.improvement,
                'improvement_pct': result.improvement_pct,
                'success': result.success,
                'phases': result.phases,
                'duration_seconds': result.duration_seconds,
            }

        except ImportError as e:
            jobs[job_id].status = "failed"
            jobs[job_id].error = f"Unified optimizer not available: {e}"
        except Exception as e:
            jobs[job_id].status = "failed"
            jobs[job_id].error = str(e)

    background_tasks.add_task(run_unified)
    return {"job_id": job_id, "status": "pending", "message": f"Unified optimization ({request.mode}) started"}


@app.post("/v1/upload")
async def upload_pcb(file: UploadFile = File(...)):
    """Upload a PCB file for processing."""
    if not file.filename.endswith('.kicad_pcb'):
        raise HTTPException(status_code=400, detail="File must be a .kicad_pcb file")

    dest_path = DATA_DIR / file.filename

    with open(dest_path, 'wb') as f:
        content = await file.read()
        f.write(content)

    return {
        "success": True,
        "filename": file.filename,
        "path": str(dest_path),
        "size_bytes": dest_path.stat().st_size
    }


@app.get("/v1/download/{filename}")
async def download_pcb(filename: str):
    """Download a processed PCB file."""
    file_path = DATA_DIR / filename

    if not file_path.exists():
        raise HTTPException(status_code=404, detail=f"File not found: {filename}")

    return FileResponse(
        path=str(file_path),
        filename=filename,
        media_type="application/octet-stream"
    )


@app.get("/v1/files")
async def list_files():
    """List PCB files in the data directory."""
    files = []
    for f in DATA_DIR.glob('*.kicad_pcb'):
        files.append({
            "filename": f.name,
            "size_bytes": f.stat().st_size,
            "modified": datetime.fromtimestamp(f.stat().st_mtime).isoformat()
        })
    return {"files": files}


# ============================================================================
# Main
# ============================================================================

def main():
    """Run the API server."""
    import argparse

    parser = argparse.ArgumentParser(description='MAPOS KiCad API Server')
    parser.add_argument('--host', default='0.0.0.0', help='Host to bind to')
    parser.add_argument('--port', type=int, default=8080, help='Port to bind to')
    parser.add_argument('--reload', action='store_true', help='Enable auto-reload')

    args = parser.parse_args()

    print(f"Starting MAPOS KiCad API Server on {args.host}:{args.port}")
    print(f"Data directory: {DATA_DIR}")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"OpenRouter API: {'configured' if OPENROUTER_API_KEY else 'not configured'}")

    uvicorn.run(
        "kicad_api_server:app",
        host=args.host,
        port=args.port,
        reload=args.reload
    )


if __name__ == '__main__':
    main()
