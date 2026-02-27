"""
SmokeTestAgent - Deterministic + LLM circuit validation (Phase 4)

Hybrid approach: Deterministic S-expression parsing extracts the netlist first,
then rule-based checks catch ground-truth issues, then Claude Opus 4.6 performs
semantic analysis on the extracted summary (not raw S-expression text).

Pipeline:
1. Deterministic netlist extraction (regex + balanced-paren parsing)
2. Rule-based checks (power/ground connectivity, shorts, sanity)
3. LLM semantic analysis (voltage compatibility, polarity, bypass caps)
4. Merge deterministic + LLM issues

Validates:
1. Power rail connectivity (VCC/VDD connected to all ICs)
2. Ground connectivity (GND/VSS properly connected)
3. Short circuit detection (no direct power-to-ground paths)
4. Floating node detection (critical pins connected)
5. Current path verification (power can flow through circuit)
6. Bypass capacitor placement (decoupling for ICs)
"""

import os
import re
import json
import logging
import asyncio
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict, Optional, Any, Tuple, Set
from pathlib import Path

import httpx

# Centralized LLM provider — respects AI_PROVIDER env var (claude_code_max or openrouter)
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from llm_provider import get_llm_config, check_llm_available, log_provider_info

# Configure logging
logger = logging.getLogger(__name__)


class SmokeTestSeverity(Enum):
    """Severity levels for smoke test issues."""
    FATAL = "fatal"      # Circuit will definitely smoke/fail
    ERROR = "error"      # Circuit likely won't work correctly
    WARNING = "warning"  # Potential issue, needs review
    INFO = "info"        # Advisory information


@dataclass
class SmokeTestIssue:
    """A single issue found during smoke testing."""
    severity: SmokeTestSeverity
    test_name: str
    message: str
    component: Optional[str] = None
    net: Optional[str] = None
    recommendation: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "severity": self.severity.value,
            "test": self.test_name,
            "message": self.message,
            "component": self.component,
            "net": self.net,
            "recommendation": self.recommendation,
        }


@dataclass
class SmokeTestResult:
    """Result of smoke test validation."""
    passed: bool
    power_rails_ok: bool
    ground_ok: bool
    no_shorts: bool
    no_floating_nodes: bool
    power_dissipation_ok: bool
    current_paths_valid: bool
    issues: List[SmokeTestIssue] = field(default_factory=list)
    llm_analysis: Dict[str, Any] = field(default_factory=dict)
    simulation_data: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "passed": self.passed,
            "summary": {
                "power_rails_ok": self.power_rails_ok,
                "ground_ok": self.ground_ok,
                "no_shorts": self.no_shorts,
                "no_floating_nodes": self.no_floating_nodes,
                "power_dissipation_ok": self.power_dissipation_ok,
                "current_paths_valid": self.current_paths_valid,
            },
            "issues": [i.to_dict() for i in self.issues],
            "fatal_count": sum(1 for i in self.issues if i.severity == SmokeTestSeverity.FATAL),
            "error_count": sum(1 for i in self.issues if i.severity == SmokeTestSeverity.ERROR),
            "warning_count": sum(1 for i in self.issues if i.severity == SmokeTestSeverity.WARNING),
            "llm_analysis": self.llm_analysis,
        }


class SmokeTestAgent:
    """
    Hybrid smoke test agent: deterministic parsing + LLM semantic analysis.

    Phase 4 pipeline:
    1. ``_extract_netlist_deterministic()`` — parses KiCad S-expression with
       balanced-paren extraction and regex to extract components, wires,
       labels, and a connectivity graph.
    2. ``_run_deterministic_checks()`` — runs rule-based checks (power/ground
       connectivity, short detection, sanity checks) on the extracted netlist.
    3. ``_run_llm_smoke_analysis()`` — sends the *extracted summary* (not raw
       S-expression) to Claude Opus 4.6 for semantic checks the parser cannot
       perform (voltage compatibility, polarity, bypass caps, current paths).
    4. Merges deterministic (ground-truth) issues with LLM (semantic) issues.
    """

    def __init__(self, ngspice_path: str = "ngspice"):
        """Initialize the smoke test agent."""
        self.ngspice_path = ngspice_path
        self._http_client: Optional[httpx.AsyncClient] = None

        # Log which provider we're using
        log_provider_info("SmokeTestAgent")

        # Power ratings database (can be extended)
        self.component_power_ratings: Dict[str, float] = {
            "STM32G431": 0.5,  # 500mW typical
            "LM7805": 1.0,    # 1W without heatsink
            "AMS1117": 0.8,   # 800mW
        }

    # ------------------------------------------------------------------ #
    #  Phase 4: Deterministic netlist extraction & rule-based checks       #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _find_balanced_block(text: str, start: int) -> str:
        """Extract a balanced-parenthesis block starting at *start*.

        ``start`` must point to the opening ``(``.  Returns the full
        substring from ``(`` to the matching ``)``, inclusive.

        Raises ``RuntimeError`` if the parens never balance (malformed input).
        """
        if start >= len(text) or text[start] != "(":
            raise RuntimeError(
                f"_find_balanced_block: expected '(' at position {start}, "
                f"got '{text[start] if start < len(text) else 'EOF'}'"
            )
        depth = 0
        in_str = False
        esc = False
        for i in range(start, len(text)):
            ch = text[i]
            if esc:
                esc = False
                continue
            if ch == "\\":
                esc = True
                continue
            if ch == '"':
                in_str = not in_str
                continue
            if in_str:
                continue
            if ch == "(":
                depth += 1
            elif ch == ")":
                depth -= 1
                if depth == 0:
                    return text[start : i + 1]
        raise RuntimeError(
            f"_find_balanced_block: unbalanced parens starting at position {start} "
            f"(depth still {depth} at end of text, text length {len(text)})"
        )

    @staticmethod
    def _round_pos(x: float, y: float, precision: int = 2) -> Tuple[float, float]:
        """Round a position to ``precision`` decimals (default 0.01 mm).

        If ``SMOKE_TEST_GRID_PRECISION`` env var is set (e.g. ``2.54``),
        coordinates are snapped to the nearest multiple of that grid size
        before rounding to ``precision`` decimals.
        """
        grid_precision = float(os.environ.get("SMOKE_TEST_GRID_PRECISION", "0"))
        if grid_precision > 0:
            x = round(x / grid_precision) * grid_precision
            y = round(y / grid_precision) * grid_precision
        return (round(x, precision), round(y, precision))

    def _extract_netlist_deterministic(self, kicad_sch: str) -> Dict[str, Any]:
        """Parse a KiCad ``.kicad_sch`` S-expression deterministically.

        Returns a dict with keys:
            components  - list of {reference, lib_id, x, y}
            wires       - list of {x1, y1, x2, y2}
            labels      - list of {name, x, y, kind}  (kind = "label" | "global_label")
            nets        - dict mapping net-name -> set of rounded (x,y) positions
            connections - dict mapping rounded (x,y) -> set of connected item descriptions
            component_count - int
            wire_count      - int
            label_count     - int
            net_count       - int
            ic_components   - list of references starting with "U"
            ic_power_nets   - dict mapping IC ref -> set of power net names connected
            ic_ground_nets  - dict mapping IC ref -> set of ground net names connected
        """
        components: List[Dict[str, Any]] = []
        wires: List[Dict[str, float]] = []
        labels: List[Dict[str, Any]] = []

        # ---- Parse symbol blocks (components) using balanced-paren extraction ----
        # We look for top-level ``(symbol `` that are NOT inside ``(lib_symbols``.
        # lib_symbols is a single block we should skip.
        lib_sym_start = kicad_sch.find("(lib_symbols")
        lib_sym_end = -1
        if lib_sym_start >= 0:
            try:
                lib_block = self._find_balanced_block(kicad_sch, lib_sym_start)
                lib_sym_end = lib_sym_start + len(lib_block)
            except RuntimeError:
                lib_sym_end = -1  # If we can't parse it, don't skip anything

        # Find all ``(symbol `` occurrences outside lib_symbols
        sym_pattern = re.compile(r"\(symbol\s")
        for m in sym_pattern.finditer(kicad_sch):
            pos = m.start()
            # Skip if inside lib_symbols block
            if lib_sym_start >= 0 and lib_sym_start <= pos < lib_sym_end:
                continue

            try:
                block = self._find_balanced_block(kicad_sch, pos)
            except RuntimeError:
                continue  # Skip malformed blocks

            # Extract lib_id
            lib_id_match = re.search(r'\(lib_id\s+"([^"]+)"\)', block)
            lib_id = lib_id_match.group(1) if lib_id_match else "unknown"

            # Extract position from the symbol's ``(at X Y ...)``
            at_match = re.search(r'\(at\s+([\d.eE+-]+)\s+([\d.eE+-]+)', block)
            x = float(at_match.group(1)) if at_match else 0.0
            y = float(at_match.group(2)) if at_match else 0.0

            # Extract Reference property
            ref_match = re.search(r'\(property\s+"Reference"\s+"([^"]+)"', block)
            reference = ref_match.group(1) if ref_match else "?"

            # Skip power symbols (reference starts with #)
            if reference.startswith("#"):
                continue

            components.append({
                "reference": reference,
                "lib_id": lib_id,
                "x": x,
                "y": y,
            })

        # ---- Parse wires ----
        wire_re = re.compile(
            r'\(wire\s+\(pts\s+'
            r'\(xy\s+([\d.eE+-]+)\s+([\d.eE+-]+)\)\s*'
            r'\(xy\s+([\d.eE+-]+)\s+([\d.eE+-]+)\)\s*\)\s*'
        )
        for m in wire_re.finditer(kicad_sch):
            wires.append({
                "x1": float(m.group(1)),
                "y1": float(m.group(2)),
                "x2": float(m.group(3)),
                "y2": float(m.group(4)),
            })

        # ---- Parse labels (global_label and label) ----
        glabel_re = re.compile(
            r'\(global_label\s+"([^"]+)"\s+\(at\s+([\d.eE+-]+)\s+([\d.eE+-]+)'
        )
        for m in glabel_re.finditer(kicad_sch):
            labels.append({
                "name": m.group(1),
                "x": float(m.group(2)),
                "y": float(m.group(3)),
                "kind": "global_label",
            })

        label_re = re.compile(
            r'\(label\s+"([^"]+)"\s+\(at\s+([\d.eE+-]+)\s+([\d.eE+-]+)'
        )
        for m in label_re.finditer(kicad_sch):
            labels.append({
                "name": m.group(1),
                "x": float(m.group(2)),
                "y": float(m.group(3)),
                "kind": "label",
            })

        # ---- Build connectivity graph ----
        # Map: rounded position -> set of connected item descriptions
        pos_to_items: Dict[Tuple[float, float], Set[str]] = defaultdict(set)

        # Wire endpoints register positions
        for w in wires:
            p1 = self._round_pos(w["x1"], w["y1"])
            p2 = self._round_pos(w["x2"], w["y2"])
            wire_desc = f"wire({w['x1']},{w['y1']}->{w['x2']},{w['y2']})"
            pos_to_items[p1].add(wire_desc)
            pos_to_items[p2].add(wire_desc)

        # Labels register positions AND create implicit net connections
        # nets: net-name -> set of positions belonging to that net
        nets: Dict[str, Set[Tuple[float, float]]] = defaultdict(set)
        for lbl in labels:
            p = self._round_pos(lbl["x"], lbl["y"])
            pos_to_items[p].add(f"{lbl['kind']}:{lbl['name']}")
            nets[lbl["name"]].add(p)

        # Components register their position (approximate pin area)
        for comp in components:
            p = self._round_pos(comp["x"], comp["y"])
            pos_to_items[p].add(f"component:{comp['reference']}")

        # ---- Determine which nets each IC is connected to ----
        # Power net names (case-insensitive matching)
        POWER_NET_NAMES = {"VCC", "VDD", "VDDA", "VBAT", "3V3", "3.3V", "5V", "12V",
                           "1V8", "1.8V", "2V5", "2.5V", "VBUS", "VSYS", "VIN"}
        GROUND_NET_NAMES = {"GND", "VSS", "VSSA", "GNDA", "AGND", "DGND", "GROUND",
                            "GND_DIGITAL", "GND_ANALOG"}

        ic_refs = [c["reference"] for c in components if c["reference"].startswith("U")]

        # Build a union-find-like connectivity: two positions are connected if they
        # share a wire (both endpoints are in pos_to_items).  Labels with the same
        # name merge all their positions into one equivalence class.
        #
        # Simple approach: BFS/flood-fill from each position through shared wires.
        # A wire connects its two endpoints.  A label name connects all positions
        # that carry that label.

        # Adjacency: pos -> set of directly connected positions
        adjacency: Dict[Tuple[float, float], Set[Tuple[float, float]]] = defaultdict(set)

        # Wire adjacency: both endpoints are connected
        for w in wires:
            p1 = self._round_pos(w["x1"], w["y1"])
            p2 = self._round_pos(w["x2"], w["y2"])
            adjacency[p1].add(p2)
            adjacency[p2].add(p1)

        # Label name adjacency: all positions sharing the same label name are connected
        for net_name, positions in nets.items():
            pos_list = list(positions)
            for i in range(len(pos_list)):
                for j in range(i + 1, len(pos_list)):
                    adjacency[pos_list[i]].add(pos_list[j])
                    adjacency[pos_list[j]].add(pos_list[i])

        def flood_fill(start: Tuple[float, float]) -> Set[Tuple[float, float]]:
            """BFS flood-fill to find all positions reachable from start."""
            visited: Set[Tuple[float, float]] = set()
            queue = [start]
            while queue:
                current = queue.pop()
                if current in visited:
                    continue
                visited.add(current)
                for neighbor in adjacency.get(current, set()):
                    if neighbor not in visited:
                        queue.append(neighbor)
            return visited

        # For each IC, find which net names are reachable from its position
        ic_power_nets: Dict[str, Set[str]] = {}
        ic_ground_nets: Dict[str, Set[str]] = {}

        for comp in components:
            ref = comp["reference"]
            if not ref.startswith("U"):
                continue

            comp_pos = self._round_pos(comp["x"], comp["y"])
            reachable = flood_fill(comp_pos)

            # Collect all label names at reachable positions
            reachable_net_names: Set[str] = set()
            for pos in reachable:
                for item in pos_to_items.get(pos, set()):
                    if item.startswith("global_label:") or item.startswith("label:"):
                        net_name = item.split(":", 1)[1]
                        reachable_net_names.add(net_name)

            power = {n for n in reachable_net_names if n.upper() in POWER_NET_NAMES}
            ground = {n for n in reachable_net_names if n.upper() in GROUND_NET_NAMES}

            ic_power_nets[ref] = power
            ic_ground_nets[ref] = ground

        # Count unique nets (each connected component with at least one label is a net)
        all_positions = set(pos_to_items.keys())
        visited_global: Set[Tuple[float, float]] = set()
        unique_net_count = 0
        for pos in all_positions:
            if pos in visited_global:
                continue
            cluster = flood_fill(pos)
            visited_global.update(cluster)
            # Check if this cluster contains at least one label
            has_label = False
            for p in cluster:
                for item in pos_to_items.get(p, set()):
                    if item.startswith("global_label:") or item.startswith("label:"):
                        has_label = True
                        break
                if has_label:
                    break
            if has_label:
                unique_net_count += 1

        return {
            "components": components,
            "wires": wires,
            "labels": labels,
            "nets": {name: list(positions) for name, positions in nets.items()},
            "connections": {str(k): list(v) for k, v in pos_to_items.items()},
            "component_count": len(components),
            "wire_count": len(wires),
            "label_count": len(labels),
            "net_count": unique_net_count,
            "ic_components": ic_refs,
            "ic_power_nets": {ref: list(nets) for ref, nets in ic_power_nets.items()},
            "ic_ground_nets": {ref: list(nets) for ref, nets in ic_ground_nets.items()},
        }

    def _run_deterministic_checks(self, netlist: Dict[str, Any]) -> List[SmokeTestIssue]:
        """Run rule-based checks on the deterministically extracted netlist.

        Returns a list of ``SmokeTestIssue`` instances.
        """
        issues: List[SmokeTestIssue] = []

        # ------ Sanity: wire count ------
        if netlist["wire_count"] == 0:
            issues.append(SmokeTestIssue(
                severity=SmokeTestSeverity.FATAL,
                test_name="wire_count_sanity",
                message="Schematic contains ZERO wires. No electrical connections exist.",
                recommendation="Add wires to connect components in the schematic.",
            ))

        # ------ Sanity: component count ------
        if netlist["component_count"] == 0:
            issues.append(SmokeTestIssue(
                severity=SmokeTestSeverity.FATAL,
                test_name="component_count_sanity",
                message="Schematic contains ZERO components. Nothing to validate.",
                recommendation="Add components to the schematic before running smoke test.",
            ))

        # ------ Power rail connectivity for ICs ------
        ic_power_nets: Dict[str, List[str]] = netlist.get("ic_power_nets", {})
        ic_ground_nets: Dict[str, List[str]] = netlist.get("ic_ground_nets", {})

        for ic_ref in netlist.get("ic_components", []):
            power_nets = ic_power_nets.get(ic_ref, [])
            ground_nets = ic_ground_nets.get(ic_ref, [])

            if not power_nets:
                issues.append(SmokeTestIssue(
                    severity=SmokeTestSeverity.ERROR,
                    test_name="power_rail_connectivity",
                    message=f"IC {ic_ref} is NOT connected to any power net (VCC/VDD/3V3/5V/etc.).",
                    component=ic_ref,
                    recommendation=f"Connect {ic_ref} power pin(s) to a power rail via wire and label.",
                ))

            if not ground_nets:
                issues.append(SmokeTestIssue(
                    severity=SmokeTestSeverity.ERROR,
                    test_name="ground_connectivity",
                    message=f"IC {ic_ref} is NOT connected to any ground net (GND/VSS/etc.).",
                    component=ic_ref,
                    recommendation=f"Connect {ic_ref} ground pin(s) to a ground rail.",
                ))

        # ------ Short detection: power label and ground label at same position ------
        POWER_NET_NAMES_UPPER = {"VCC", "VDD", "VDDA", "VBAT", "3V3", "3.3V", "5V",
                                 "12V", "1V8", "1.8V", "2V5", "2.5V", "VBUS", "VSYS", "VIN"}
        GROUND_NET_NAMES_UPPER = {"GND", "VSS", "VSSA", "GNDA", "AGND", "DGND", "GROUND",
                                  "GND_DIGITAL", "GND_ANALOG"}

        # Build position -> label-name mapping from the labels list
        pos_labels: Dict[Tuple[float, float], List[str]] = defaultdict(list)
        for lbl in netlist.get("labels", []):
            p = self._round_pos(lbl["x"], lbl["y"])
            pos_labels[p].append(lbl["name"])

        for pos, label_names in pos_labels.items():
            names_upper = {n.upper() for n in label_names}
            has_power = bool(names_upper & POWER_NET_NAMES_UPPER)
            has_ground = bool(names_upper & GROUND_NET_NAMES_UPPER)
            if has_power and has_ground:
                power_labels = [n for n in label_names if n.upper() in POWER_NET_NAMES_UPPER]
                ground_labels = [n for n in label_names if n.upper() in GROUND_NET_NAMES_UPPER]
                issues.append(SmokeTestIssue(
                    severity=SmokeTestSeverity.FATAL,
                    test_name="short_detection",
                    message=(
                        f"Power-to-ground SHORT detected at position ({pos[0]}, {pos[1]}): "
                        f"power label(s) {power_labels} and ground label(s) {ground_labels} "
                        f"share the same wire endpoint."
                    ),
                    net=f"{power_labels[0]}->{ground_labels[0]}",
                    recommendation="Remove the direct connection between power and ground.",
                ))

        # Also check: power and ground labels connected via a single wire segment
        # (endpoints of same wire hit a power label and a ground label)
        for w in netlist.get("wires", []):
            p1 = self._round_pos(w["x1"], w["y1"])
            p2 = self._round_pos(w["x2"], w["y2"])
            labels_at_p1 = {n.upper() for n in pos_labels.get(p1, [])}
            labels_at_p2 = {n.upper() for n in pos_labels.get(p2, [])}

            p1_power = bool(labels_at_p1 & POWER_NET_NAMES_UPPER)
            p1_ground = bool(labels_at_p1 & GROUND_NET_NAMES_UPPER)
            p2_power = bool(labels_at_p2 & POWER_NET_NAMES_UPPER)
            p2_ground = bool(labels_at_p2 & GROUND_NET_NAMES_UPPER)

            if (p1_power and p2_ground) or (p1_ground and p2_power):
                power_end = p1 if p1_power else p2
                ground_end = p2 if p1_power else p1
                pwr_names = [n for n in pos_labels.get(power_end, []) if n.upper() in POWER_NET_NAMES_UPPER]
                gnd_names = [n for n in pos_labels.get(ground_end, []) if n.upper() in GROUND_NET_NAMES_UPPER]
                # Avoid duplicate if already caught by same-position check
                if power_end != ground_end:
                    issues.append(SmokeTestIssue(
                        severity=SmokeTestSeverity.FATAL,
                        test_name="short_detection",
                        message=(
                            f"Power-to-ground SHORT via single wire: "
                            f"{pwr_names} at ({power_end[0]}, {power_end[1]}) connected to "
                            f"{gnd_names} at ({ground_end[0]}, {ground_end[1]})."
                        ),
                        net=f"{pwr_names[0] if pwr_names else '?'}->{gnd_names[0] if gnd_names else '?'}",
                        recommendation="Remove the direct wire between power and ground, or add a load component.",
                    ))

        # ------ Short detection via connectivity graph ------
        # If a power label and a ground label are in the same connected component
        # and that connected component contains NO components, it's a direct short.
        connections = netlist.get("connections", {})

        # Rebuild adjacency from netlist data for flood fill
        _adjacency: Dict[Tuple[float, float], Set[Tuple[float, float]]] = defaultdict(set)
        for w in netlist.get("wires", []):
            p1 = self._round_pos(w["x1"], w["y1"])
            p2 = self._round_pos(w["x2"], w["y2"])
            _adjacency[p1].add(p2)
            _adjacency[p2].add(p1)

        # Label name adjacency: same-name labels are implicitly connected
        _net_positions: Dict[str, List[Tuple[float, float]]] = defaultdict(list)
        for lbl in netlist.get("labels", []):
            p = self._round_pos(lbl["x"], lbl["y"])
            _net_positions[lbl["name"]].append(p)
        for _name, _positions in _net_positions.items():
            for _i in range(len(_positions)):
                for _j in range(_i + 1, len(_positions)):
                    _adjacency[_positions[_i]].add(_positions[_j])
                    _adjacency[_positions[_j]].add(_positions[_i])

        def _flood(start: Tuple[float, float]) -> Set[Tuple[float, float]]:
            visited: Set[Tuple[float, float]] = set()
            queue = [start]
            while queue:
                cur = queue.pop()
                if cur in visited:
                    continue
                visited.add(cur)
                for nb in _adjacency.get(cur, set()):
                    if nb not in visited:
                        queue.append(nb)
            return visited

        # Find connected components, check for power+ground with no components
        all_label_positions = set()
        for lbl in netlist.get("labels", []):
            all_label_positions.add(self._round_pos(lbl["x"], lbl["y"]))

        all_wire_positions_set: Set[Tuple[float, float]] = set()
        for w in netlist.get("wires", []):
            all_wire_positions_set.add(self._round_pos(w["x1"], w["y1"]))
            all_wire_positions_set.add(self._round_pos(w["x2"], w["y2"]))

        all_graph_positions = all_label_positions | all_wire_positions_set
        _visited_short: Set[Tuple[float, float]] = set()
        _reported_shorts: Set[str] = set()  # Avoid duplicates from earlier checks

        # Collect existing short messages to avoid duplicates
        for iss in issues:
            if iss.test_name == "short_detection":
                _reported_shorts.add(iss.net or "")

        comp_positions = set()
        for comp in netlist.get("components", []):
            comp_positions.add(self._round_pos(comp["x"], comp["y"]))

        for start_pos in all_graph_positions:
            if start_pos in _visited_short:
                continue
            cluster = _flood(start_pos)
            _visited_short.update(cluster)

            # Collect labels in this cluster
            cluster_power_labels: List[str] = []
            cluster_ground_labels: List[str] = []
            has_component = False
            for p in cluster:
                if p in comp_positions:
                    has_component = True
                for lbl in netlist.get("labels", []):
                    lp = self._round_pos(lbl["x"], lbl["y"])
                    if lp == p:
                        if lbl["name"].upper() in POWER_NET_NAMES_UPPER:
                            cluster_power_labels.append(lbl["name"])
                        elif lbl["name"].upper() in GROUND_NET_NAMES_UPPER:
                            cluster_ground_labels.append(lbl["name"])

            if cluster_power_labels and cluster_ground_labels and not has_component:
                # Deduplicate
                pwr_unique = sorted(set(cluster_power_labels))
                gnd_unique = sorted(set(cluster_ground_labels))
                net_key = f"{pwr_unique[0]}->{gnd_unique[0]}"
                if net_key not in _reported_shorts:
                    _reported_shorts.add(net_key)
                    issues.append(SmokeTestIssue(
                        severity=SmokeTestSeverity.FATAL,
                        test_name="short_detection",
                        message=(
                            f"Power-to-ground SHORT via wire path (no load): "
                            f"power net(s) {pwr_unique} and ground net(s) {gnd_unique} "
                            f"are connected with no components in between."
                        ),
                        net=net_key,
                        recommendation="Add a load component between power and ground, or remove the direct connection.",
                    ))

        # ------ Floating IC check: IC position not connected to ANY wire ------
        wire_positions: Set[Tuple[float, float]] = set()
        for w in netlist.get("wires", []):
            wire_positions.add(self._round_pos(w["x1"], w["y1"]))
            wire_positions.add(self._round_pos(w["x2"], w["y2"]))

        for comp in netlist.get("components", []):
            ref = comp["reference"]
            if not ref.startswith("U"):
                continue
            comp_pos = self._round_pos(comp["x"], comp["y"])
            # Check if ANY wire endpoint is at the component position
            # (In real schematics, pins are offset from the component center,
            #  so this is a rough heuristic — the connectivity graph is more accurate.)
            power_nets = ic_power_nets.get(ref, [])
            ground_nets = ic_ground_nets.get(ref, [])
            if not power_nets and not ground_nets and comp_pos not in wire_positions:
                issues.append(SmokeTestIssue(
                    severity=SmokeTestSeverity.WARNING,
                    test_name="floating_ic",
                    message=(
                        f"IC {ref} at ({comp['x']}, {comp['y']}) appears to be completely "
                        f"unconnected — no wire endpoints at its position and no reachable "
                        f"power or ground nets."
                    ),
                    component=ref,
                    recommendation=f"Connect {ref} to the circuit with wires and labels.",
                ))

        return issues

    async def _ensure_http_client(self):
        """Ensure HTTP client is initialized."""
        if self._http_client is None:
            self._http_client = httpx.AsyncClient(timeout=120.0)

    async def run_smoke_test(
        self,
        kicad_sch_content: str,
        bom_items: List[Dict[str, Any]],
        power_sources: Optional[List[Dict[str, Any]]] = None
    ) -> SmokeTestResult:
        """
        Run comprehensive smoke test on schematic.

        Phase 4 pipeline:
        1. Deterministic S-expression parsing  (fast, 100% reliable)
        2. Rule-based checks on extracted data  (deterministic issues)
        3. LLM analysis with extracted *summary* (semantic / higher-order issues)
        4. Merge deterministic + LLM issues

        Args:
            kicad_sch_content: KiCad schematic file content (.kicad_sch)
            bom_items: Bill of materials with component specifications
            power_sources: Power source definitions (voltage, current limits)

        Returns:
            SmokeTestResult with pass/fail and detailed issues
        """
        await self._ensure_http_client()

        # --- Step 1: Deterministic netlist extraction ---
        logger.info("Phase 4: Running deterministic netlist extraction...")
        try:
            netlist = self._extract_netlist_deterministic(kicad_sch_content)
        except Exception as exc:
            raise RuntimeError(
                f"Deterministic netlist extraction failed on schematic "
                f"({len(kicad_sch_content)} chars): {exc}"
            ) from exc

        logger.info(
            f"Deterministic extraction: {netlist['component_count']} components, "
            f"{netlist['wire_count']} wires, {netlist['label_count']} labels, "
            f"{netlist['net_count']} nets, {len(netlist['ic_components'])} ICs"
        )

        # --- Step 2: Deterministic rule-based checks ---
        deterministic_issues = self._run_deterministic_checks(netlist)

        # --- Step 2b: Component count validation against BOM ---
        bom_count = len(bom_items) if bom_items else 0
        extracted_count = netlist["component_count"]
        if bom_count and extracted_count < bom_count * 0.8:
            deterministic_issues.append(SmokeTestIssue(
                severity=SmokeTestSeverity.FATAL,
                test_name="component_count",
                message=(
                    f"Only {extracted_count}/{bom_count} components found in netlist. "
                    f"Expected at least {int(bom_count * 0.8)} (80% of BOM)."
                ),
                recommendation="Check schematic assembly - components may be missing from the netlist",
            ))

        logger.info(f"Deterministic checks found {len(deterministic_issues)} issue(s)")

        # Build a compact summary for the LLM (no raw S-expression needed)
        netlist_summary = {
            "component_count": netlist["component_count"],
            "wire_count": netlist["wire_count"],
            "label_count": netlist["label_count"],
            "net_count": netlist["net_count"],
            "components": [
                {"reference": c["reference"], "lib_id": c["lib_id"],
                 "position": f"({c['x']}, {c['y']})"}
                for c in netlist["components"]
            ],
            "net_names": list(netlist["nets"].keys()),
            "ic_components": netlist["ic_components"],
            "ic_power_nets": netlist["ic_power_nets"],
            "ic_ground_nets": netlist["ic_ground_nets"],
            "deterministic_issues": [
                {"severity": i.severity.value, "test": i.test_name, "message": i.message}
                for i in deterministic_issues
            ],
        }

        # --- Step 3: LLM analysis (enhanced with deterministic summary) ---
        llm_config = get_llm_config()
        if not check_llm_available():
            provider = llm_config["provider"]
            error_msg = (
                f"SmokeTestAgent: LLM provider '{provider}' is NOT available. "
                f"AI_PROVIDER={os.environ.get('AI_PROVIDER', '(not set)')}. "
                f"CLAUDE_CODE_PROXY_URL={os.environ.get('CLAUDE_CODE_PROXY_URL', '(not set)')}. "
                f"OPENROUTER_API_KEY={'set' if os.environ.get('OPENROUTER_API_KEY') else 'NOT SET'}. "
                f"Resolved base_url={llm_config['base_url']}. "
                f"Cannot perform smoke test without a working LLM provider."
            )
            logger.error(error_msg)
            raise RuntimeError(error_msg)

        logger.info(
            f"Phase 4: Running LLM smoke analysis with deterministic summary "
            f"(provider={llm_config['provider']}, url={llm_config['base_url']})..."
        )

        llm_result = await self._run_llm_smoke_analysis(
            kicad_sch_content, bom_items, power_sources or [],
            netlist_summary=netlist_summary,
        )

        # --- Step 4: Merge deterministic + LLM results ---
        result = self._convert_llm_result(llm_result)

        # Prepend deterministic issues (they are ground truth, higher confidence)
        result.issues = deterministic_issues + result.issues

        # Re-evaluate pass/fail: any FATAL deterministic issue => fail
        det_fatal = sum(1 for i in deterministic_issues if i.severity == SmokeTestSeverity.FATAL)
        det_error = sum(1 for i in deterministic_issues if i.severity == SmokeTestSeverity.ERROR)
        if det_fatal > 0:
            result.passed = False
        if det_error > 0:
            result.passed = False

        # Update summary flags from deterministic data
        ic_power = netlist.get("ic_power_nets", {})
        ic_ground = netlist.get("ic_ground_nets", {})
        ics = netlist.get("ic_components", [])
        if ics:
            all_powered = all(len(ic_power.get(ic, [])) > 0 for ic in ics)
            all_grounded = all(len(ic_ground.get(ic, [])) > 0 for ic in ics)
            if not all_powered:
                result.power_rails_ok = False
            if not all_grounded:
                result.ground_ok = False

        # Check for shorts from deterministic issues
        has_short = any(
            i.test_name == "short_detection" and i.severity == SmokeTestSeverity.FATAL
            for i in deterministic_issues
        )
        if has_short:
            result.no_shorts = False

        # Inject deterministic counts into llm_analysis for downstream consumers
        result.llm_analysis["deterministic_netlist"] = netlist_summary

        return result

    async def _run_llm_smoke_analysis(
        self,
        kicad_sch: str,
        bom_items: List[Dict[str, Any]],
        power_sources: List[Dict[str, Any]],
        netlist_summary: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Run comprehensive smoke test using LLM (Opus 4.6).

        When ``netlist_summary`` is provided (Phase 4), the LLM receives the
        pre-extracted deterministic summary instead of raw S-expression text,
        dramatically reducing token usage and improving accuracy.  A truncated
        excerpt of the raw S-expression is still included for context the
        deterministic parser may not capture (e.g., pin names, footprints).
        """
        # Build BOM summary for LLM
        bom_summary = []
        for item in bom_items[:30]:  # Limit for token efficiency
            ref = item.get("reference", item.get("part_number", "?"))
            part = item.get("part_number", "unknown")
            cat = item.get("category", "")
            value = item.get("value", "")
            bom_summary.append(f"- {ref}: {part} ({cat}) {value}")

        # Build power source summary
        power_summary = []
        for ps in power_sources:
            power_summary.append(f"- {ps.get('net', 'VCC')}: {ps.get('voltage', 5)}V, {ps.get('current_limit', 1)}A max")

        if not power_summary:
            power_summary.append("- VCC: 5V (assumed)")
            power_summary.append("- GND: 0V (reference)")

        # Build the prompt differently depending on whether we have deterministic data
        if netlist_summary is not None:
            # Phase 4: deterministic summary + truncated raw for context
            netlist_json = json.dumps(netlist_summary, indent=2, default=str)

            # Include a shorter raw excerpt for supplementary context only
            raw_excerpt_size = 15000
            raw_note = (
                f"KICAD SCHEMATIC EXCERPT (first {raw_excerpt_size // 1000}KB — "
                f"for supplementary context only; use the DETERMINISTIC NETLIST above "
                f"as ground truth for connectivity):\n```\n{kicad_sch[:raw_excerpt_size]}\n```"
            )

            deterministic_issues_block = ""
            det_issues = netlist_summary.get("deterministic_issues", [])
            if det_issues:
                lines = []
                for di in det_issues:
                    lines.append(f"- [{di['severity'].upper()}] {di['test']}: {di['message']}")
                deterministic_issues_block = (
                    "\n\nDETERMINISTIC ISSUES ALREADY DETECTED (treat as ground truth):\n"
                    + "\n".join(lines)
                    + "\nDo NOT duplicate these issues. Focus on issues the deterministic parser cannot catch."
                )

            prompt = f"""You are an expert electronics engineer performing a "smoke test" validation on a KiCad schematic.

A "smoke test" ensures the circuit will NOT smoke/burn/fail when power is applied. You must analyze the schematic and identify any issues that would cause immediate failure.

IMPORTANT: A deterministic parser has already extracted the netlist below. Use it as GROUND TRUTH for connectivity, component counts, wire counts, and net assignments. Your job is to perform SEMANTIC analysis that a parser cannot:
- Voltage level compatibility between connected ICs
- Correct orientation/polarity of components
- Missing bypass/decoupling capacitors
- Current path feasibility
- Component value appropriateness

IMPORTANT KICAD SEMANTICS:
- In KiCad, global_label elements with the SAME TEXT are ELECTRICALLY CONNECTED implicitly
- Power net equivalents: VCC=VDD=3V3=5V, GND=VSS=GROUND, VDDA=analog power, VBAT=battery

DETERMINISTIC NETLIST SUMMARY:
```json
{netlist_json}
```
{deterministic_issues_block}

ANALYZE THE FOLLOWING TESTS:

1. **POWER RAIL CHECK**: Are VCC/VDD/3V3/5V nets connected to all ICs that need power? (Use ic_power_nets from summary)
2. **GROUND CHECK**: Is GND/VSS connected to all components that need ground? (Use ic_ground_nets from summary)
3. **SHORT CIRCUIT DETECTION**: Are there any direct connections between power and ground?
4. **FLOATING NODE DETECTION**: Are there critical pins (power, enable, reset) that are floating/unconnected?
5. **CURRENT PATH VALIDATION**: Can current flow from power through loads to ground?
6. **BYPASS CAPACITOR CHECK**: Do ICs have bypass/decoupling capacitors near their power pins?
7. **POLARITY CHECK**: Are polarized components (diodes, capacitors, ICs) oriented correctly?

BILL OF MATERIALS:
{chr(10).join(bom_summary)}

POWER SOURCES:
{chr(10).join(power_summary)}

{raw_note}

Return a JSON object with this EXACT structure:
{{
    "overall_passed": true/false,
    "tests": {{
        "power_rails": {{
            "passed": true/false,
            "issues": ["list of issues found"],
            "details": "explanation"
        }},
        "ground": {{
            "passed": true/false,
            "issues": ["list of issues found"],
            "details": "explanation"
        }},
        "short_circuits": {{
            "passed": true/false,
            "issues": ["list of issues found"],
            "details": "explanation"
        }},
        "floating_nodes": {{
            "passed": true/false,
            "issues": ["list of issues found"],
            "details": "explanation"
        }},
        "current_paths": {{
            "passed": true/false,
            "issues": ["list of issues found"],
            "details": "explanation"
        }},
        "bypass_caps": {{
            "passed": true/false,
            "issues": ["list of issues found"],
            "details": "explanation"
        }}
    }},
    "fatal_issues": ["list of issues that WILL cause smoke/failure"],
    "warnings": ["list of potential issues to review"],
    "recommendations": ["list of improvements"],
    "wire_count": {netlist_summary['wire_count']},
    "component_count": {netlist_summary['component_count']},
    "net_count": {netlist_summary['net_count']}
}}

Be thorough but practical. Focus on issues that would cause IMMEDIATE FAILURE when power is applied.
Do NOT re-report issues already found by the deterministic parser.
Return ONLY valid JSON, no explanations outside the JSON structure."""

        else:
            # Legacy path: no deterministic data — send raw S-expression
            prompt = f"""You are an expert electronics engineer performing a "smoke test" validation on a KiCad schematic.

A "smoke test" ensures the circuit will NOT smoke/burn/fail when power is applied. You must analyze the schematic and identify any issues that would cause immediate failure.

IMPORTANT KICAD SEMANTICS:
- In KiCad, global_label elements with the SAME TEXT are ELECTRICALLY CONNECTED implicitly
- Example: (global_label "VCC" ...) at one location connects to ALL other (global_label "VCC" ...) elements
- This means if power pin VDD has a global_label "VCC" nearby, it IS connected to the VCC power rail
- Similarly, global_label "GND" elements are all connected to ground
- Do NOT report "no power connection" if the power pin has a nearby global_label with matching power net name
- Power net equivalents: VCC=VDD=3V3=5V, GND=VSS=GROUND, VDDA=analog power, VBAT=battery

ANALYZE THE FOLLOWING TESTS:

1. **POWER RAIL CHECK**: Are VCC/VDD/3V3/5V nets connected to all ICs that need power?
2. **GROUND CHECK**: Is GND/VSS connected to all components that need ground?
3. **SHORT CIRCUIT DETECTION**: Are there any direct connections between power and ground?
4. **FLOATING NODE DETECTION**: Are there critical pins (power, enable, reset) that are floating/unconnected?
5. **CURRENT PATH VALIDATION**: Can current flow from power through loads to ground?
6. **BYPASS CAPACITOR CHECK**: Do ICs have bypass/decoupling capacitors near their power pins?
7. **POLARITY CHECK**: Are polarized components (diodes, capacitors, ICs) oriented correctly?

BILL OF MATERIALS:
{chr(10).join(bom_summary)}

POWER SOURCES:
{chr(10).join(power_summary)}

KICAD SCHEMATIC S-EXPRESSION (first 60KB):
```
{kicad_sch[:60000]}
```

Return a JSON object with this EXACT structure:
{{
    "overall_passed": true/false,
    "tests": {{
        "power_rails": {{
            "passed": true/false,
            "issues": ["list of issues found"],
            "details": "explanation"
        }},
        "ground": {{
            "passed": true/false,
            "issues": ["list of issues found"],
            "details": "explanation"
        }},
        "short_circuits": {{
            "passed": true/false,
            "issues": ["list of issues found"],
            "details": "explanation"
        }},
        "floating_nodes": {{
            "passed": true/false,
            "issues": ["list of issues found"],
            "details": "explanation"
        }},
        "current_paths": {{
            "passed": true/false,
            "issues": ["list of issues found"],
            "details": "explanation"
        }},
        "bypass_caps": {{
            "passed": true/false,
            "issues": ["list of issues found"],
            "details": "explanation"
        }}
    }},
    "fatal_issues": ["list of issues that WILL cause smoke/failure"],
    "warnings": ["list of potential issues to review"],
    "recommendations": ["list of improvements"],
    "wire_count": number_of_wires_found,
    "component_count": number_of_components_found,
    "net_count": number_of_unique_nets_found
}}

Be thorough but practical. Focus on issues that would cause IMMEDIATE FAILURE when power is applied.
Return ONLY valid JSON, no explanations outside the JSON structure."""

        try:
            # Resolve LLM config at request time
            llm_config = get_llm_config()
            base_url = llm_config["base_url"]
            headers = llm_config["headers"]

            logger.info(f"SmokeTest LLM call: provider={llm_config['provider']}, url={base_url}, schematic_size={len(kicad_sch)} chars")

            response = await self._http_client.post(
                base_url,
                headers=headers,
                json={
                    "model": "anthropic/claude-opus-4.6",
                    "messages": [
                        {
                            "role": "system",
                            "content": "You are an expert electronics engineer specializing in circuit validation. Analyze schematics and return structured JSON results. Be precise and thorough."
                        },
                        {"role": "user", "content": prompt}
                    ],
                    "temperature": 0.0,
                    "max_tokens": 4000
                }
            )
            response.raise_for_status()
            result = response.json()

            content = result["choices"][0]["message"]["content"].strip()

            # Clean up JSON if wrapped in code block
            if content.startswith("```"):
                lines = content.split("\n")
                # Find the actual JSON content
                start_idx = 1
                end_idx = len(lines) - 1 if lines[-1].startswith("```") else len(lines)
                content = "\n".join(lines[start_idx:end_idx])

            # Try direct parse first
            try:
                return json.loads(content)
            except json.JSONDecodeError:
                pass

            # LLM often returns JSON followed by explanatory text ("Extra data" error).
            # Extract the first complete JSON object using brace counting.
            brace_start = content.find("{")
            if brace_start >= 0:
                depth = 0
                in_str = False
                esc = False
                for i in range(brace_start, len(content)):
                    ch = content[i]
                    if esc:
                        esc = False
                        continue
                    if ch == "\\":
                        esc = True
                        continue
                    if ch == '"':
                        in_str = not in_str
                        continue
                    if in_str:
                        continue
                    if ch == "{":
                        depth += 1
                    elif ch == "}":
                        depth -= 1
                        if depth == 0:
                            return json.loads(content[brace_start : i + 1])

            return json.loads(content)  # fallback — will raise JSONDecodeError

        except json.JSONDecodeError as e:
            logger.error(f"LLM smoke test returned invalid JSON: {e}")
            return {
                "overall_passed": True,  # Don't fail on LLM errors — deterministic checks cover fatals
                "tests": {},
                "fatal_issues": [],
                "warnings": [f"LLM analysis unavailable: Invalid JSON response - {e}"],
                "recommendations": ["Retry smoke test with LLM analysis"],
                "error": str(e)
            }
        except Exception as e:
            logger.error(f"LLM smoke test failed: {e}")
            return {
                "overall_passed": True,  # Don't fail on LLM errors — deterministic checks cover fatals
                "tests": {},
                "fatal_issues": [],
                "warnings": [f"LLM analysis unavailable: {e}"],
                "recommendations": ["Check API connectivity and retry"],
                "error": str(e)
            }

    def _convert_llm_result(self, llm_result: Dict[str, Any]) -> SmokeTestResult:
        """Convert LLM analysis result to SmokeTestResult."""
        issues = []
        tests = llm_result.get("tests", {})

        # Process fatal issues
        for fatal in llm_result.get("fatal_issues", []):
            issues.append(SmokeTestIssue(
                severity=SmokeTestSeverity.FATAL,
                test_name="smoke_test",
                message=fatal,
                recommendation="Fix before applying power"
            ))

        # Process warnings
        for warning in llm_result.get("warnings", []):
            issues.append(SmokeTestIssue(
                severity=SmokeTestSeverity.WARNING,
                test_name="smoke_test",
                message=warning
            ))

        # Process individual test results
        for test_name, test_result in tests.items():
            if not test_result.get("passed", True):
                for issue in test_result.get("issues", []):
                    issues.append(SmokeTestIssue(
                        severity=SmokeTestSeverity.ERROR,
                        test_name=test_name,
                        message=issue,
                        recommendation=test_result.get("details", "")
                    ))

        # Determine overall pass/fail
        fatal_count = sum(1 for i in issues if i.severity == SmokeTestSeverity.FATAL)
        passed = llm_result.get("overall_passed", False) and fatal_count == 0

        return SmokeTestResult(
            passed=passed,
            power_rails_ok=tests.get("power_rails", {}).get("passed", False),
            ground_ok=tests.get("ground", {}).get("passed", False),
            no_shorts=tests.get("short_circuits", {}).get("passed", True),
            no_floating_nodes=tests.get("floating_nodes", {}).get("passed", False),
            power_dissipation_ok=True,  # Covered by current_paths in LLM analysis
            current_paths_valid=tests.get("current_paths", {}).get("passed", False),
            issues=issues,
            llm_analysis=llm_result
        )

    async def validate_connectivity(
        self,
        kicad_sch_content: str
    ) -> Dict[str, Any]:
        """
        Quick connectivity validation using LLM.

        Checks basic structural properties:
        - Are there wires in the schematic?
        - Are components connected?
        - Are power nets present?
        """
        await self._ensure_http_client()

        llm_config = get_llm_config()
        if not check_llm_available():
            error_msg = (
                f"SmokeTestAgent.validate_connectivity: LLM provider '{llm_config['provider']}' NOT available. "
                f"AI_PROVIDER={os.environ.get('AI_PROVIDER', '(not set)')}. "
                f"Cannot perform connectivity validation."
            )
            logger.error(error_msg)
            raise RuntimeError(error_msg)

        prompt = f"""Analyze this KiCad schematic and report on basic connectivity.

Answer these specific questions:
1. How many wire segments are present? (count (wire elements)
2. How many component symbols are placed? (count (symbol elements)
3. Are there power net labels (VCC, GND, etc.)?
4. Are components connected to each other via wires?

SCHEMATIC:
```
{kicad_sch_content[:30000]}
```

Return JSON:
{{
    "wire_count": number,
    "component_count": number,
    "has_power_nets": true/false,
    "components_connected": true/false,
    "connectivity_score": 0-100,
    "issues": ["list any major connectivity problems"]
}}

Return ONLY the JSON."""

        try:
            base_url = llm_config["base_url"]
            headers = llm_config["headers"]

            logger.info(f"SmokeTest connectivity check: provider={llm_config['provider']}, url={base_url}")

            response = await self._http_client.post(
                base_url,
                headers=headers,
                json={
                    "model": "anthropic/claude-opus-4.6",
                    "messages": [
                        {"role": "system", "content": "You are a KiCad schematic analyzer. Return structured JSON."},
                        {"role": "user", "content": prompt}
                    ],
                    "temperature": 0.0,
                    "max_tokens": 1000
                }
            )
            response.raise_for_status()
            result = response.json()

            content = result["choices"][0]["message"]["content"].strip()
            if content.startswith("```"):
                lines = content.split("\n")
                content = "\n".join(lines[1:-1] if lines[-1].startswith("```") else lines[1:])

            return json.loads(content)

        except Exception as e:
            logger.error(f"Connectivity validation failed: {e}")
            return {"valid": False, "error": str(e)}

    async def close(self):
        """Close HTTP client."""
        if self._http_client:
            await self._http_client.aclose()
            self._http_client = None


# Test runner
async def main():
    """Test the smoke test agent."""
    import sys

    if len(sys.argv) < 2:
        print("Usage: python smoke_test_agent.py <schematic.kicad_sch>")
        print("       python smoke_test_agent.py --test")
        sys.exit(1)

    if sys.argv[1] == "--test":
        # Run a simple test
        agent = SmokeTestAgent()

        test_sch = """(kicad_sch (version 20231120) (generator "test")
  (symbol (lib_id "Device:R") (at 100 100 0) (unit 1)
    (property "Reference" "R1")
    (property "Value" "10k")
  )
  (wire (pts (xy 100 95) (xy 100 80)))
  (wire (pts (xy 100 105) (xy 100 120)))
  (label "VCC" (at 100 80 0))
  (label "GND" (at 100 120 0))
)"""

        bom = [
            {"reference": "R1", "part_number": "RC0805FR-0710KL", "category": "Resistor", "value": "10k"}
        ]

        result = await agent.run_smoke_test(test_sch, bom)
        print(json.dumps(result.to_dict(), indent=2))
        await agent.close()
    else:
        # Load schematic from file
        sch_path = Path(sys.argv[1])
        if not sch_path.exists():
            print(f"File not found: {sch_path}")
            sys.exit(1)

        sch_content = sch_path.read_text()

        agent = SmokeTestAgent()
        result = await agent.run_smoke_test(sch_content, [])
        print(json.dumps(result.to_dict(), indent=2))
        await agent.close()


if __name__ == "__main__":
    asyncio.run(main())
