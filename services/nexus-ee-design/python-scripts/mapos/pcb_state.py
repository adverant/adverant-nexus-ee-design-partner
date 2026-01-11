#!/usr/bin/env python3
"""
PCB State Representation - Foundation for Multi-Agent PCB Optimization System (MAPOS).

This module provides:
1. PCBState: Immutable snapshot of a PCB configuration
2. PCBModification: Atomic changes to PCB state
3. ParameterSpace: Definition of tunable parameters and their bounds
4. Fitness evaluation and DRC integration

Inspired by AlphaFold's state representation and MCTS node structures.
"""

import re
import json
import hashlib
import subprocess
from copy import deepcopy
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Set, Any
from enum import Enum, auto
import tempfile
import shutil


def find_kicad_cli() -> Optional[str]:
    """
    Auto-detect kicad-cli executable across platforms.

    Searches common installation paths for macOS, Linux, and containers.

    Returns:
        Path to kicad-cli if found, None otherwise
    """
    candidates = [
        # macOS KiCad 8.x
        '/Applications/KiCad/KiCad.app/Contents/MacOS/kicad-cli',
        # Linux system paths
        '/usr/bin/kicad-cli',
        '/usr/local/bin/kicad-cli',
        # Linux PPA/snap installs
        '/snap/kicad/current/bin/kicad-cli',
        '/opt/kicad/bin/kicad-cli',
        # Container paths
        '/usr/lib/kicad/bin/kicad-cli',
        # System PATH (last resort)
        shutil.which('kicad-cli'),
    ]

    for path in candidates:
        if path and Path(path).exists():
            return path

    return None


class ModificationType(Enum):
    """Types of modifications that can be applied to a PCB."""
    # Component positioning
    MOVE_COMPONENT = auto()       # Change component X, Y, rotation
    ROTATE_COMPONENT = auto()     # Rotate component only

    # Trace modifications
    ADJUST_TRACE_WIDTH = auto()   # Modify trace width
    ADD_TRACE = auto()            # Add trace segment
    DELETE_TRACE = auto()         # Remove trace segment

    # Via modifications
    ADJUST_VIA_SIZE = auto()      # Modify via diameter/drill
    MOVE_VIA = auto()             # Reposition a via
    ADD_VIA = auto()              # Add thermal or stitching via
    DELETE_VIA = auto()           # Remove unnecessary via

    # Zone and clearance
    ADJUST_CLEARANCE = auto()     # Modify clearance settings
    ADJUST_ZONE = auto()          # Modify zone parameters
    ADJUST_ZONE_CLEARANCE = auto()# Zone-specific clearance
    FIX_ZONE_NET = auto()         # Fix zone net assignment

    # Silkscreen/designator
    MOVE_SILKSCREEN = auto()      # Reposition silkscreen text
    ADJUST_SILKSCREEN = auto()    # Modify silkscreen properties
    ADD_DESIGNATOR = auto()       # Add reference designator
    MOVE_DESIGNATOR = auto()      # Move reference designator

    # Footprint-level DFM (MAPOS Phase 2)
    ADJUST_SOLDER_MASK = auto()   # Modify solder mask expansion
    ADJUST_PASTE_MASK = auto()    # Modify solder paste aperture
    ADJUST_PAD_SIZE = auto()      # Modify pad dimensions
    ADJUST_COURTYARD = auto()     # Modify courtyard boundary

    # Copper pour
    ADD_COPPER_POUR = auto()      # Add copper zone/pour
    ADD_KEEPOUT = auto()          # Add keepout region
    ADD_THERMAL_VIA = auto()      # Add thermal via array


@dataclass
class ComponentPosition:
    """Position and orientation of a component."""
    reference: str
    x: float
    y: float
    rotation: float
    layer: str
    footprint: str


@dataclass
class TraceSegment:
    """A PCB trace segment."""
    net_name: str
    layer: str
    width: float
    start_x: float
    start_y: float
    end_x: float
    end_y: float


@dataclass
class Via:
    """A PCB via."""
    x: float
    y: float
    diameter: float
    drill: float
    net_name: str
    layers: Tuple[str, str]


@dataclass
class Zone:
    """A copper zone/pour."""
    name: str
    net_name: str
    layer: str
    clearance: float
    min_thickness: float
    thermal_gap: float
    bounds: Tuple[float, float, float, float]  # left, top, right, bottom


@dataclass
class PCBModification:
    """
    An atomic modification to apply to a PCB state.

    Modifications are designed to be:
    - Reversible (can undo)
    - Composable (can combine multiple)
    - Serializable (can save/load)
    """
    mod_type: ModificationType
    target: str  # Component ref, net name, or zone name
    parameters: Dict[str, Any]
    description: str = ""
    source_agent: str = ""
    confidence: float = 0.5

    def to_dict(self) -> Dict:
        """Serialize modification to dictionary."""
        return {
            'mod_type': self.mod_type.name,
            'target': self.target,
            'parameters': self.parameters,
            'description': self.description,
            'source_agent': self.source_agent,
            'confidence': self.confidence
        }

    @classmethod
    def from_dict(cls, data: Dict) -> 'PCBModification':
        """Deserialize modification from dictionary."""
        return cls(
            mod_type=ModificationType[data['mod_type']],
            target=data['target'],
            parameters=data['parameters'],
            description=data.get('description', ''),
            source_agent=data.get('source_agent', ''),
            confidence=data.get('confidence', 0.5)
        )


@dataclass
class DRCResult:
    """Results from running DRC on a PCB."""
    total_violations: int
    errors: int
    warnings: int
    unconnected: int
    violations_by_type: Dict[str, int]
    top_violations: List[Dict]

    @property
    def fitness_score(self) -> float:
        """Calculate fitness score (higher is better, 0-1 range)."""
        # Inverse of violations, normalized
        base_score = 1.0 / (1.0 + self.total_violations / 100)

        # Bonus for low critical errors
        error_bonus = 0.1 if self.errors < 50 else 0

        # Bonus for resolved connections
        connection_bonus = 0.1 if self.unconnected < 5 else 0

        return min(1.0, base_score + error_bonus + connection_bonus)


@dataclass
class ParameterBounds:
    """Bounds for a tunable parameter."""
    name: str
    min_value: float
    max_value: float
    step: float
    default: float
    description: str = ""

    def clamp(self, value: float) -> float:
        """Clamp value to bounds."""
        return max(self.min_value, min(self.max_value, value))

    def is_valid(self, value: float) -> bool:
        """Check if value is within bounds."""
        return self.min_value <= value <= self.max_value


class ParameterSpace:
    """
    Definition of the PCB parameter space for optimization.

    Based on analysis of the foc-esc-heavy-lift PCB:
    - Board size: 250mm x 85mm
    - 10-layer stackup
    - High voltage (58V) and high current (100A) design
    """

    # Component position bounds (relative to board)
    COMPONENT_X = ParameterBounds("component_x", 5.0, 245.0, 0.1, 125.0, "Component X position (mm)")
    COMPONENT_Y = ParameterBounds("component_y", 5.0, 80.0, 0.1, 42.5, "Component Y position (mm)")
    COMPONENT_ROTATION = ParameterBounds("component_rotation", 0.0, 360.0, 45.0, 0.0, "Component rotation (degrees)")

    # Trace parameters
    SIGNAL_TRACE_WIDTH = ParameterBounds("signal_trace_width", 0.15, 0.5, 0.05, 0.25, "Signal trace width (mm)")
    POWER_TRACE_WIDTH = ParameterBounds("power_trace_width", 1.0, 4.0, 0.5, 2.0, "Power trace width (mm)")
    HV_TRACE_WIDTH = ParameterBounds("hv_trace_width", 1.5, 5.0, 0.5, 2.5, "High voltage trace width (mm)")

    # Via parameters
    VIA_DIAMETER = ParameterBounds("via_diameter", 0.6, 1.2, 0.1, 0.8, "Via outer diameter (mm)")
    VIA_DRILL = ParameterBounds("via_drill", 0.3, 0.6, 0.05, 0.4, "Via drill diameter (mm)")
    THERMAL_VIA_DIAMETER = ParameterBounds("thermal_via_diameter", 0.8, 1.5, 0.1, 1.0, "Thermal via diameter (mm)")

    # Clearance parameters (IPC-2221 based)
    SIGNAL_CLEARANCE = ParameterBounds("signal_clearance", 0.1, 0.3, 0.02, 0.15, "Signal-signal clearance (mm)")
    POWER_CLEARANCE = ParameterBounds("power_clearance", 0.2, 0.5, 0.05, 0.25, "Power net clearance (mm)")
    HV_CLEARANCE = ParameterBounds("hv_clearance", 0.5, 1.0, 0.1, 0.6, "High voltage clearance (mm)")

    # Zone parameters
    ZONE_CLEARANCE = ParameterBounds("zone_clearance", 0.2, 0.5, 0.05, 0.25, "Zone clearance (mm)")
    ZONE_MIN_WIDTH = ParameterBounds("zone_min_width", 0.2, 0.5, 0.05, 0.25, "Zone minimum width (mm)")
    ZONE_THERMAL_GAP = ParameterBounds("zone_thermal_gap", 0.3, 0.8, 0.1, 0.5, "Zone thermal gap (mm)")
    ZONE_INSET = ParameterBounds("zone_inset", 0.5, 2.0, 0.25, 1.0, "Zone edge inset (mm)")

    # Silkscreen parameters
    SILK_OFFSET = ParameterBounds("silk_offset", -2.0, 2.0, 0.5, 0.0, "Silkscreen offset (mm)")

    # Solder mask parameters
    MASK_EXPANSION = ParameterBounds("mask_expansion", 0.02, 0.1, 0.01, 0.05, "Solder mask expansion (mm)")

    @classmethod
    def get_all_bounds(cls) -> Dict[str, ParameterBounds]:
        """Get all parameter bounds as a dictionary."""
        return {
            name: value for name, value in vars(cls).items()
            if isinstance(value, ParameterBounds)
        }

    @classmethod
    def get_parameter_count(cls) -> int:
        """Get total number of tunable parameters."""
        return len(cls.get_all_bounds())


class PCBState:
    """
    Immutable snapshot of a PCB configuration.

    This is the core data structure for the optimization system.
    Each state represents a complete PCB configuration that can be:
    - Evaluated (DRC)
    - Modified (apply PCBModification)
    - Compared (fitness score)
    - Serialized (save/load)

    Inspired by AlphaFold's MSA representation and game state in AlphaGo.
    """

    def __init__(
        self,
        pcb_path: Optional[str] = None,
        state_id: Optional[str] = None,
        parent_id: Optional[str] = None
    ):
        """
        Initialize PCB state.

        Args:
            pcb_path: Path to .kicad_pcb file (loads from file)
            state_id: Unique identifier for this state
            parent_id: ID of parent state (for lineage tracking)
        """
        self.pcb_path = Path(pcb_path) if pcb_path else None
        self.state_id = state_id or self._generate_id()
        self.parent_id = parent_id
        self.generation = 0

        # Extracted PCB data
        self.components: Dict[str, ComponentPosition] = {}
        self.traces: List[TraceSegment] = []
        self.vias: List[Via] = []
        self.zones: Dict[str, Zone] = {}
        self.nets: Dict[str, int] = {}  # name -> id

        # Parameters (current values)
        self.parameters: Dict[str, float] = {}

        # Cached evaluation results
        self._drc_result: Optional[DRCResult] = None
        self._fitness: Optional[float] = None

        # Modification history
        self.modifications: List[PCBModification] = []

        # Load from file if path provided
        if self.pcb_path and self.pcb_path.exists():
            self._load_from_file()

    def _generate_id(self) -> str:
        """Generate unique state ID."""
        import uuid
        return str(uuid.uuid4())[:8]

    def _load_from_file(self) -> None:
        """Load PCB data from .kicad_pcb file."""
        with open(self.pcb_path, 'r', encoding='utf-8') as f:
            content = f.read()

        self._parse_nets(content)
        self._parse_components(content)
        self._parse_traces(content)
        self._parse_vias(content)
        self._parse_zones(content)
        self._extract_parameters()

    def _parse_nets(self, content: str) -> None:
        """Parse net declarations."""
        pattern = r'^\s*\(net\s+(\d+)\s+"([^"]+)"\)\s*$'
        for match in re.finditer(pattern, content, re.MULTILINE):
            net_id = int(match.group(1))
            net_name = match.group(2)
            self.nets[net_name] = net_id

    def _parse_components(self, content: str) -> None:
        """Parse footprint/component placements."""
        # Pattern for footprint blocks
        fp_pattern = r'\(footprint\s+"([^"]+)".*?\(layer\s+"([^"]+)"\).*?\(at\s+([\d.-]+)\s+([\d.-]+)(?:\s+([\d.-]+))?\).*?\(property\s+"Reference"\s+"([^"]+)"'

        for match in re.finditer(fp_pattern, content, re.DOTALL):
            footprint = match.group(1)
            layer = match.group(2)
            x = float(match.group(3))
            y = float(match.group(4))
            rotation = float(match.group(5)) if match.group(5) else 0.0
            reference = match.group(6)

            self.components[reference] = ComponentPosition(
                reference=reference,
                x=x,
                y=y,
                rotation=rotation,
                layer=layer,
                footprint=footprint
            )

    def _parse_traces(self, content: str) -> None:
        """Parse trace segments."""
        pattern = r'\(segment\s+\(start\s+([\d.-]+)\s+([\d.-]+)\)\s+\(end\s+([\d.-]+)\s+([\d.-]+)\)\s+\(width\s+([\d.-]+)\)\s+\(layer\s+"([^"]+)"\)\s+\(net\s+(\d+)\)'

        # Build reverse net map
        id_to_name = {v: k for k, v in self.nets.items()}

        for match in re.finditer(pattern, content):
            net_id = int(match.group(7))
            net_name = id_to_name.get(net_id, f"net_{net_id}")

            self.traces.append(TraceSegment(
                net_name=net_name,
                layer=match.group(6),
                width=float(match.group(5)),
                start_x=float(match.group(1)),
                start_y=float(match.group(2)),
                end_x=float(match.group(3)),
                end_y=float(match.group(4))
            ))

    def _parse_vias(self, content: str) -> None:
        """Parse via definitions."""
        pattern = r'\(via\s+\(at\s+([\d.-]+)\s+([\d.-]+)\)\s+\(size\s+([\d.-]+)\)\s+\(drill\s+([\d.-]+)\)\s+\(layers\s+"([^"]+)"\s+"([^"]+)"\)\s+\(net\s+(\d+)\)'

        id_to_name = {v: k for k, v in self.nets.items()}

        for match in re.finditer(pattern, content):
            net_id = int(match.group(7))
            net_name = id_to_name.get(net_id, f"net_{net_id}")

            self.vias.append(Via(
                x=float(match.group(1)),
                y=float(match.group(2)),
                diameter=float(match.group(3)),
                drill=float(match.group(4)),
                net_name=net_name,
                layers=(match.group(5), match.group(6))
            ))

    def _parse_zones(self, content: str) -> None:
        """Parse zone definitions."""
        # Simplified zone parsing
        zone_pattern = r'\(zone\s*\n?\s*\(net\s+(\d+)\)\s*\n?\s*\(net_name\s+"([^"]+)"\)\s*\n?\s*\(layer\s+"([^"]+)"\)'

        for match in re.finditer(zone_pattern, content):
            net_name = match.group(2)
            layer = match.group(3)

            # Extract clearance if present
            clearance_match = re.search(
                rf'\(zone.*?\(net_name\s+"{re.escape(net_name)}"\).*?\(clearance\s+([\d.-]+)\)',
                content, re.DOTALL
            )
            clearance = float(clearance_match.group(1)) if clearance_match else 0.2

            zone_key = f"{net_name}_{layer}"
            self.zones[zone_key] = Zone(
                name=net_name,
                net_name=net_name,
                layer=layer,
                clearance=clearance,
                min_thickness=0.25,
                thermal_gap=0.5,
                bounds=(0, 0, 250, 85)  # Default board size
            )

    def _extract_parameters(self) -> None:
        """Extract current parameter values from parsed data."""
        # Average trace widths by type
        signal_widths = [t.width for t in self.traces if t.width < 0.5]
        power_widths = [t.width for t in self.traces if 0.5 <= t.width < 1.5]
        hv_widths = [t.width for t in self.traces if t.width >= 1.5]

        self.parameters['signal_trace_width'] = sum(signal_widths) / len(signal_widths) if signal_widths else 0.25
        self.parameters['power_trace_width'] = sum(power_widths) / len(power_widths) if power_widths else 1.0
        self.parameters['hv_trace_width'] = sum(hv_widths) / len(hv_widths) if hv_widths else 2.0

        # Via parameters
        if self.vias:
            self.parameters['via_diameter'] = sum(v.diameter for v in self.vias) / len(self.vias)
            self.parameters['via_drill'] = sum(v.drill for v in self.vias) / len(self.vias)

        # Zone parameters
        if self.zones:
            clearances = [z.clearance for z in self.zones.values()]
            self.parameters['zone_clearance'] = sum(clearances) / len(clearances)

    @classmethod
    def from_file(cls, pcb_path: str) -> 'PCBState':
        """Create PCBState from a KiCad PCB file."""
        return cls(pcb_path=pcb_path)

    def copy(self) -> 'PCBState':
        """Create a deep copy of this state."""
        new_state = PCBState(state_id=self._generate_id(), parent_id=self.state_id)
        new_state.pcb_path = self.pcb_path
        new_state.generation = self.generation + 1
        new_state.components = deepcopy(self.components)
        new_state.traces = deepcopy(self.traces)
        new_state.vias = deepcopy(self.vias)
        new_state.zones = deepcopy(self.zones)
        new_state.nets = deepcopy(self.nets)
        new_state.parameters = deepcopy(self.parameters)
        new_state.modifications = list(self.modifications)
        return new_state

    def apply_modification(self, mod: PCBModification) -> 'PCBState':
        """
        Apply a modification and return new state.

        This is the core state transition function. The original state
        is not modified (immutability).

        Args:
            mod: Modification to apply

        Returns:
            New PCBState with modification applied
        """
        new_state = self.copy()
        new_state.modifications.append(mod)

        if mod.mod_type == ModificationType.MOVE_COMPONENT:
            ref = mod.target
            if ref in new_state.components:
                comp = new_state.components[ref]
                if 'x' in mod.parameters:
                    new_state.components[ref] = ComponentPosition(
                        reference=comp.reference,
                        x=mod.parameters.get('x', comp.x),
                        y=mod.parameters.get('y', comp.y),
                        rotation=mod.parameters.get('rotation', comp.rotation),
                        layer=comp.layer,
                        footprint=comp.footprint
                    )

        elif mod.mod_type == ModificationType.ADJUST_TRACE_WIDTH:
            width = mod.parameters.get('width', 0.25)
            layer = mod.parameters.get('layer')
            net = mod.target

            new_traces = []
            for trace in new_state.traces:
                if trace.net_name == net and (layer is None or trace.layer == layer):
                    new_traces.append(TraceSegment(
                        net_name=trace.net_name,
                        layer=trace.layer,
                        width=width,
                        start_x=trace.start_x,
                        start_y=trace.start_y,
                        end_x=trace.end_x,
                        end_y=trace.end_y
                    ))
                else:
                    new_traces.append(trace)
            new_state.traces = new_traces

        elif mod.mod_type == ModificationType.ADJUST_VIA_SIZE:
            diameter = mod.parameters.get('diameter', 0.8)
            drill = mod.parameters.get('drill', 0.4)

            new_vias = []
            for via in new_state.vias:
                if via.net_name == mod.target:
                    new_vias.append(Via(
                        x=via.x,
                        y=via.y,
                        diameter=diameter,
                        drill=drill,
                        net_name=via.net_name,
                        layers=via.layers
                    ))
                else:
                    new_vias.append(via)
            new_state.vias = new_vias

        elif mod.mod_type == ModificationType.ADJUST_CLEARANCE:
            param_name = mod.parameters.get('param_name', 'signal_clearance')
            value = mod.parameters.get('value', 0.15)
            new_state.parameters[param_name] = value

        elif mod.mod_type == ModificationType.DELETE_VIA:
            x, y = mod.parameters.get('x'), mod.parameters.get('y')
            tolerance = 0.1
            new_state.vias = [
                v for v in new_state.vias
                if not (abs(v.x - x) < tolerance and abs(v.y - y) < tolerance)
            ]

        elif mod.mod_type == ModificationType.ADJUST_ZONE:
            zone_key = mod.target
            if zone_key in new_state.zones:
                zone = new_state.zones[zone_key]
                new_state.zones[zone_key] = Zone(
                    name=zone.name,
                    net_name=zone.net_name,
                    layer=zone.layer,
                    clearance=mod.parameters.get('clearance', zone.clearance),
                    min_thickness=mod.parameters.get('min_thickness', zone.min_thickness),
                    thermal_gap=mod.parameters.get('thermal_gap', zone.thermal_gap),
                    bounds=zone.bounds
                )

        # Invalidate cached results
        new_state._drc_result = None
        new_state._fitness = None

        return new_state

    def save_to_file(self, output_path: str) -> Path:
        """
        Save state to a new PCB file.

        This applies all modifications to the original file and saves
        the result to a new location.

        Args:
            output_path: Path for output file

        Returns:
            Path to saved file
        """
        if not self.pcb_path or not self.pcb_path.exists():
            raise RuntimeError("No source PCB file available")

        output = Path(output_path)

        # Load original content
        with open(self.pcb_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # Apply component position changes
        for ref, comp in self.components.items():
            # Find and update component position
            pattern = rf'(\(footprint\s+"[^"]+"\s*\(layer\s+"[^"]+"\)\s*)\(at\s+[\d.-]+\s+[\d.-]+(?:\s+[\d.-]+)?\)'
            # This is simplified - full implementation would need more robust parsing

        # Write modified content
        with open(output, 'w', encoding='utf-8') as f:
            f.write(content)

        return output

    def run_drc(self, kicad_cli: Optional[str] = None) -> DRCResult:
        """
        Run KiCad DRC on this state.

        Args:
            kicad_cli: Path to kicad-cli executable. If None, auto-detects.

        Returns:
            DRCResult with violation counts
        """
        if self._drc_result is not None:
            return self._drc_result

        if not self.pcb_path or not self.pcb_path.exists():
            raise RuntimeError("No PCB file available for DRC")

        # Auto-detect kicad-cli if not provided
        if kicad_cli is None:
            kicad_cli = find_kicad_cli()
        if not kicad_cli:
            raise RuntimeError("kicad-cli not found. Please install KiCad or provide explicit path.")

        # Create temp file if modifications need to be applied
        if self.modifications:
            with tempfile.NamedTemporaryFile(suffix='.kicad_pcb', delete=False) as tmp:
                tmp_path = tmp.name
            self.save_to_file(tmp_path)
            pcb_to_check = tmp_path
        else:
            pcb_to_check = str(self.pcb_path)
            tmp_path = None

        try:
            with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as drc_out:
                drc_output = drc_out.name

            result = subprocess.run([
                kicad_cli, "pcb", "drc",
                "--output", drc_output,
                "--format", "json",
                pcb_to_check
            ], capture_output=True, text=True, timeout=120)

            if Path(drc_output).exists():
                with open(drc_output) as f:
                    drc_data = json.load(f)

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

                self._drc_result = DRCResult(
                    total_violations=len(violations),
                    errors=errors,
                    warnings=warnings,
                    unconnected=len(unconnected),
                    violations_by_type=by_type,
                    top_violations=violations[:10]
                )
            else:
                # DRC failed
                self._drc_result = DRCResult(
                    total_violations=9999,
                    errors=9999,
                    warnings=0,
                    unconnected=0,
                    violations_by_type={},
                    top_violations=[]
                )
        finally:
            if tmp_path and Path(tmp_path).exists():
                Path(tmp_path).unlink()
            if Path(drc_output).exists():
                Path(drc_output).unlink()

        return self._drc_result

    @property
    def fitness(self) -> float:
        """Get fitness score for this state."""
        if self._fitness is not None:
            return self._fitness

        drc = self.run_drc()
        self._fitness = drc.fitness_score
        return self._fitness

    @property
    def drc_summary(self) -> str:
        """Get human-readable DRC summary."""
        drc = self.run_drc()
        return f"Total: {drc.total_violations}, Errors: {drc.errors}, Unconnected: {drc.unconnected}"

    @property
    def component_summary(self) -> str:
        """Get summary of component positions."""
        return f"{len(self.components)} components, {len(self.vias)} vias, {len(self.traces)} traces"

    @property
    def via_summary(self) -> str:
        """Get summary of via configuration."""
        if not self.vias:
            return "No vias"
        avg_dia = sum(v.diameter for v in self.vias) / len(self.vias)
        return f"{len(self.vias)} vias, avg diameter {avg_dia:.2f}mm"

    def get_hash(self) -> str:
        """Get hash of current state for deduplication."""
        state_str = json.dumps({
            'parameters': self.parameters,
            'component_count': len(self.components),
            'via_count': len(self.vias),
            'modifications': [m.to_dict() for m in self.modifications]
        }, sort_keys=True)
        return hashlib.md5(state_str.encode()).hexdigest()[:12]

    def to_dict(self) -> Dict:
        """Serialize state to dictionary."""
        return {
            'state_id': self.state_id,
            'parent_id': self.parent_id,
            'generation': self.generation,
            'pcb_path': str(self.pcb_path) if self.pcb_path else None,
            'parameters': self.parameters,
            'component_count': len(self.components),
            'via_count': len(self.vias),
            'trace_count': len(self.traces),
            'zone_count': len(self.zones),
            'modifications': [m.to_dict() for m in self.modifications],
            'fitness': self._fitness,
            'drc_result': {
                'total_violations': self._drc_result.total_violations,
                'errors': self._drc_result.errors,
                'unconnected': self._drc_result.unconnected
            } if self._drc_result else None
        }

    @classmethod
    def from_dict(cls, data: Dict) -> 'PCBState':
        """Deserialize state from dictionary."""
        state = cls(
            pcb_path=data.get('pcb_path'),
            state_id=data.get('state_id'),
            parent_id=data.get('parent_id')
        )
        state.generation = data.get('generation', 0)
        state.parameters = data.get('parameters', {})
        state.modifications = [
            PCBModification.from_dict(m)
            for m in data.get('modifications', [])
        ]
        return state

    def __repr__(self) -> str:
        return f"PCBState(id={self.state_id}, gen={self.generation}, fitness={self._fitness or 'N/A'})"


# Convenience functions

def load_pcb_state(pcb_path: str) -> PCBState:
    """Load a PCB state from file."""
    return PCBState.from_file(pcb_path)


def create_random_modification(state: PCBState, rng=None) -> PCBModification:
    """Create a random valid modification for the given state."""
    import random
    rng = rng or random

    mod_types = [
        ModificationType.MOVE_COMPONENT,
        ModificationType.ADJUST_CLEARANCE,
        ModificationType.ADJUST_VIA_SIZE,
        ModificationType.ADJUST_ZONE
    ]

    mod_type = rng.choice(mod_types)

    if mod_type == ModificationType.MOVE_COMPONENT and state.components:
        ref = rng.choice(list(state.components.keys()))
        comp = state.components[ref]
        return PCBModification(
            mod_type=mod_type,
            target=ref,
            parameters={
                'x': comp.x + rng.uniform(-2.0, 2.0),
                'y': comp.y + rng.uniform(-2.0, 2.0),
                'rotation': (comp.rotation + rng.choice([0, 45, 90, 180, 270])) % 360
            },
            description=f"Move {ref} randomly"
        )

    elif mod_type == ModificationType.ADJUST_CLEARANCE:
        params = ['signal_clearance', 'power_clearance', 'zone_clearance']
        param = rng.choice(params)
        bounds = getattr(ParameterSpace, param.upper(), ParameterSpace.SIGNAL_CLEARANCE)
        return PCBModification(
            mod_type=mod_type,
            target=param,
            parameters={
                'param_name': param,
                'value': rng.uniform(bounds.min_value, bounds.max_value)
            },
            description=f"Adjust {param}"
        )

    elif mod_type == ModificationType.ADJUST_VIA_SIZE and state.nets:
        net = rng.choice(list(state.nets.keys()))
        return PCBModification(
            mod_type=mod_type,
            target=net,
            parameters={
                'diameter': rng.uniform(0.6, 1.2),
                'drill': rng.uniform(0.3, 0.6)
            },
            description=f"Adjust vias for {net}"
        )

    else:  # ADJUST_ZONE
        if state.zones:
            zone = rng.choice(list(state.zones.keys()))
            return PCBModification(
                mod_type=mod_type,
                target=zone,
                parameters={
                    'clearance': rng.uniform(0.2, 0.5),
                    'thermal_gap': rng.uniform(0.3, 0.8)
                },
                description=f"Adjust zone {zone}"
            )

    # Fallback
    return PCBModification(
        mod_type=ModificationType.ADJUST_CLEARANCE,
        target='signal_clearance',
        parameters={'param_name': 'signal_clearance', 'value': 0.15},
        description="Default modification"
    )


if __name__ == '__main__':
    import sys

    if len(sys.argv) < 2:
        print("Usage: python pcb_state.py <path_to.kicad_pcb>")
        sys.exit(1)

    pcb_path = sys.argv[1]

    print("\n" + "="*60)
    print("PCB STATE ANALYSIS")
    print("="*60)

    try:
        state = PCBState.from_file(pcb_path)

        print(f"\nState ID: {state.state_id}")
        print(f"PCB File: {state.pcb_path.name}")
        print(f"\nComponents: {len(state.components)}")
        print(f"Traces: {len(state.traces)}")
        print(f"Vias: {len(state.vias)}")
        print(f"Zones: {len(state.zones)}")
        print(f"Nets: {len(state.nets)}")

        print(f"\nExtracted Parameters:")
        for name, value in state.parameters.items():
            print(f"  {name}: {value:.3f}")

        print(f"\nParameter Space ({ParameterSpace.get_parameter_count()} parameters):")
        for name, bounds in list(ParameterSpace.get_all_bounds().items())[:5]:
            print(f"  {name}: [{bounds.min_value}, {bounds.max_value}] (default: {bounds.default})")
        print(f"  ... and {ParameterSpace.get_parameter_count() - 5} more")

        # Run DRC
        print(f"\nRunning DRC...")
        drc = state.run_drc()
        print(f"  Total violations: {drc.total_violations}")
        print(f"  Errors: {drc.errors}")
        print(f"  Warnings: {drc.warnings}")
        print(f"  Unconnected: {drc.unconnected}")
        print(f"  Fitness score: {drc.fitness_score:.4f}")

        # Show top violation types
        print(f"\n  Top violation types:")
        for vtype, count in sorted(drc.violations_by_type.items(), key=lambda x: -x[1])[:5]:
            print(f"    {vtype}: {count}")

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
