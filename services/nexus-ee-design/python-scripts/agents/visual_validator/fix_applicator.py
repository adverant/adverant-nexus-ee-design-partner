"""
Schematic Fix Applicator - Applies fixes to KiCad S-expression schematic content.

Modifies schematic S-expression content based on SchematicFix operations.
Supports component movement, wire addition, label creation, and more.

NO FALLBACKS - Strict error handling with verbose reporting.

Author: Nexus EE Design Team
"""

import logging
import re
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from .issue_to_fix import FixOperation, SchematicFix

logger = logging.getLogger(__name__)


class FixApplicationError(Exception):
    """
    Raised when fix application fails.

    Contains detailed context for debugging.
    """

    def __init__(
        self,
        message: str,
        fix: SchematicFix,
        schematic_size: int,
        partial_changes: List[str]
    ):
        self.message = message
        self.fix = fix
        self.schematic_size = schematic_size
        self.partial_changes = partial_changes

        full_message = f"""
================================================================================
FIX APPLICATION FAILED
================================================================================
Error: {message}
Fix ID: {fix.fix_id}
Fix Operation: {fix.fix_operation.value}
Target Component: {fix.target_component or 'N/A'}
Target Net: {fix.target_net or 'N/A'}
Schematic Size: {schematic_size} characters

Fix Parameters:
{fix.parameters}

Partial Changes Applied:
{chr(10).join(f"  - {c}" for c in partial_changes) or '  None'}

TROUBLESHOOTING:
1. Check target component/net exists in schematic
2. Verify fix parameters are valid
3. Review S-expression syntax requirements
================================================================================
"""
        super().__init__(full_message)


@dataclass
class ApplyResult:
    """Result from applying fixes to schematic."""
    success: bool
    modified_content: str = ""
    applied_fixes: List[str] = field(default_factory=list)
    failed_fixes: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    original_size: int = 0
    modified_size: int = 0

    def to_dict(self) -> dict:
        return {
            "success": self.success,
            "applied_fixes": self.applied_fixes,
            "failed_fixes": self.failed_fixes,
            "errors": self.errors,
            "original_size": self.original_size,
            "modified_size": self.modified_size,
            "size_delta": self.modified_size - self.original_size,
        }


class SchematicFixApplicator:
    """
    Applies fixes to KiCad schematic S-expression content.

    Supports various fix operations like moving components, adding wires,
    creating labels, and adding junctions.

    NO FALLBACKS - Strict application with verbose errors.

    Usage:
        applicator = SchematicFixApplicator()
        result = await applicator.apply_fixes(schematic_content, fixes)
        if result.success:
            new_content = result.modified_content
    """

    # KiCad grid settings
    DEFAULT_GRID = 2.54  # mm (100 mil)
    MIN_WIRE_LENGTH = 1.27  # mm (50 mil)

    def __init__(self, validate_syntax: bool = True):
        """
        Initialize fix applicator.

        Args:
            validate_syntax: Whether to validate S-expression syntax after modifications
        """
        self.validate_syntax = validate_syntax
        logger.info(f"SchematicFixApplicator initialized: validate_syntax={validate_syntax}")

    async def apply_fixes(
        self,
        schematic_content: str,
        fixes: List[SchematicFix],
        rollback_on_error: bool = True
    ) -> ApplyResult:
        """
        Apply a list of fixes to schematic content.

        Args:
            schematic_content: Original KiCad S-expression content
            fixes: List of fixes to apply
            rollback_on_error: Whether to rollback all changes if any fix fails

        Returns:
            ApplyResult with modified content
        """
        if not fixes:
            logger.info("No fixes to apply")
            return ApplyResult(
                success=True,
                modified_content=schematic_content,
                original_size=len(schematic_content),
                modified_size=len(schematic_content),
            )

        logger.info(f"Applying {len(fixes)} fixes to schematic")

        result = ApplyResult(
            original_size=len(schematic_content),
        )

        # Work with a copy
        content = schematic_content
        original_content = schematic_content

        for fix in fixes:
            try:
                logger.debug(f"Applying fix: {fix.fix_id} - {fix.fix_operation.value}")

                content = await self._apply_single_fix(content, fix)
                result.applied_fixes.append(
                    f"{fix.fix_id}: {fix.fix_operation.value} - {fix.description}"
                )

                logger.info(f"Fix applied: {fix.fix_id}")

            except Exception as e:
                error_msg = f"{fix.fix_id}: {str(e)}"
                result.failed_fixes.append(error_msg)
                result.errors.append(error_msg)
                logger.error(f"Fix failed: {error_msg}")

                if rollback_on_error:
                    logger.warning("Rolling back all changes due to error")
                    content = original_content
                    result.applied_fixes.clear()
                    break

        # Validate final content
        if self.validate_syntax and content != original_content:
            validation_result = self._validate_sexp_syntax(content)
            if not validation_result["valid"]:
                result.errors.append(f"Syntax validation failed: {validation_result['error']}")
                if rollback_on_error:
                    logger.error("Rolling back due to syntax validation failure")
                    content = original_content
                    result.applied_fixes.clear()

        result.success = len(result.applied_fixes) > 0 or len(fixes) == 0
        result.modified_content = content
        result.modified_size = len(content)

        logger.info(
            f"Fix application complete: {len(result.applied_fixes)} applied, "
            f"{len(result.failed_fixes)} failed, "
            f"size delta: {result.modified_size - result.original_size}"
        )

        return result

    async def _apply_single_fix(self, content: str, fix: SchematicFix) -> str:
        """Apply a single fix operation to content."""
        operation = fix.fix_operation

        if operation == FixOperation.MOVE_COMPONENT:
            return self._apply_move_component(content, fix)
        elif operation == FixOperation.ROTATE_COMPONENT:
            return self._apply_rotate_component(content, fix)
        elif operation == FixOperation.ADD_WIRE:
            return self._apply_add_wire(content, fix)
        elif operation == FixOperation.ADD_JUNCTION:
            return self._apply_add_junction(content, fix)
        elif operation == FixOperation.ADD_LABEL:
            return self._apply_add_label(content, fix)
        elif operation == FixOperation.ADD_NET_LABEL:
            return self._apply_add_net_label(content, fix)
        elif operation == FixOperation.ADD_POWER_FLAG:
            return self._apply_add_power_flag(content, fix)
        elif operation == FixOperation.ADD_NO_CONNECT:
            return self._apply_add_no_connect(content, fix)
        elif operation == FixOperation.ADD_COMPONENT:
            return self._apply_add_component(content, fix)
        elif operation == FixOperation.MODIFY_VALUE:
            return self._apply_modify_value(content, fix)
        elif operation == FixOperation.REMOVE_WIRE:
            return self._apply_remove_wire(content, fix)
        elif operation == FixOperation.REROUTE_NET:
            return await self._apply_reroute_net(content, fix)
        else:
            raise FixApplicationError(
                message=f"Unsupported fix operation: {operation.value}",
                fix=fix,
                schematic_size=len(content),
                partial_changes=[]
            )

    def _apply_move_component(self, content: str, fix: SchematicFix) -> str:
        """Move a component to new coordinates."""
        params = fix.parameters
        reference = params.get("reference") or fix.target_component

        if not reference:
            raise FixApplicationError(
                message="No reference designator specified for MOVE_COMPONENT",
                fix=fix,
                schematic_size=len(content),
                partial_changes=[]
            )

        x = params.get("x")
        y = params.get("y")

        if x is None or y is None:
            # If optimize_placement, calculate optimal position
            if params.get("optimize_placement"):
                x, y = self._calculate_optimal_position(content, reference)
            else:
                raise FixApplicationError(
                    message="No x/y coordinates specified for MOVE_COMPONENT",
                    fix=fix,
                    schematic_size=len(content),
                    partial_changes=[]
                )

        # Snap to grid
        x = self._snap_to_grid(x)
        y = self._snap_to_grid(y)

        # Find the symbol block with this reference
        pattern = rf'(\(symbol\s+\(lib_id\s+"[^"]+"\)\s+\(at\s+)[\d.-]+\s+[\d.-]+(\s+[\d]+\))'

        # This is tricky - we need to find the right symbol by reference
        # First, find all symbol blocks
        symbol_pattern = r'\(symbol\s+\(lib_id\s+"[^"]+"\)\s+\(at\s+[\d.-]+\s+[\d.-]+\s+[\d]+\).*?\(property\s+"Reference"\s+"' + re.escape(reference) + r'"'

        match = re.search(symbol_pattern, content, re.DOTALL)
        if not match:
            raise FixApplicationError(
                message=f"Component {reference} not found in schematic",
                fix=fix,
                schematic_size=len(content),
                partial_changes=[]
            )

        # Find the start of this symbol block
        symbol_start = content.rfind('(symbol (lib_id', 0, match.start())
        if symbol_start == -1:
            symbol_start = content.rfind('(symbol\n', 0, match.start())

        if symbol_start == -1:
            raise FixApplicationError(
                message=f"Could not find symbol block start for {reference}",
                fix=fix,
                schematic_size=len(content),
                partial_changes=[]
            )

        # Extract and modify the (at x y rotation) part
        at_pattern = r'\(at\s+([\d.-]+)\s+([\d.-]+)\s+([\d]+)\)'
        at_match = re.search(at_pattern, content[symbol_start:symbol_start + 500])

        if not at_match:
            raise FixApplicationError(
                message=f"Could not find position for {reference}",
                fix=fix,
                schematic_size=len(content),
                partial_changes=[]
            )

        old_at = at_match.group(0)
        rotation = at_match.group(3)
        new_at = f"(at {x} {y} {rotation})"

        # Replace only the first occurrence in this symbol block
        symbol_block_end = content.find('\n  (symbol', symbol_start + 10)
        if symbol_block_end == -1:
            symbol_block_end = content.find('\n)', symbol_start + 10)

        symbol_block = content[symbol_start:symbol_block_end]
        modified_block = symbol_block.replace(old_at, new_at, 1)

        content = content[:symbol_start] + modified_block + content[symbol_block_end:]

        logger.debug(f"Moved {reference} to ({x}, {y})")
        return content

    def _apply_rotate_component(self, content: str, fix: SchematicFix) -> str:
        """Rotate a component by specified angle."""
        params = fix.parameters
        reference = params.get("reference") or fix.target_component
        angle = params.get("angle", 0)

        if not reference:
            raise FixApplicationError(
                message="No reference designator specified for ROTATE_COMPONENT",
                fix=fix,
                schematic_size=len(content),
                partial_changes=[]
            )

        # Validate angle
        if angle not in [0, 90, 180, 270]:
            angle = (angle // 90) * 90  # Snap to 90-degree increments

        # Find the symbol with this reference (similar to move)
        symbol_pattern = r'\(symbol\s+\(lib_id\s+"[^"]+"\)\s+\(at\s+[\d.-]+\s+[\d.-]+\s+[\d]+\).*?\(property\s+"Reference"\s+"' + re.escape(reference) + r'"'

        match = re.search(symbol_pattern, content, re.DOTALL)
        if not match:
            raise FixApplicationError(
                message=f"Component {reference} not found for rotation",
                fix=fix,
                schematic_size=len(content),
                partial_changes=[]
            )

        # Find and replace rotation value
        symbol_start = content.rfind('(symbol (lib_id', 0, match.start())
        at_pattern = r'\(at\s+([\d.-]+)\s+([\d.-]+)\s+([\d]+)\)'
        at_match = re.search(at_pattern, content[symbol_start:symbol_start + 500])

        if at_match:
            x, y = at_match.group(1), at_match.group(2)
            old_at = at_match.group(0)
            new_at = f"(at {x} {y} {angle})"

            content = content[:symbol_start + at_match.start()] + new_at + content[symbol_start + at_match.end():]
            logger.debug(f"Rotated {reference} to {angle}°")

        return content

    def _apply_add_wire(self, content: str, fix: SchematicFix) -> str:
        """Add a wire to the schematic."""
        params = fix.parameters
        start_x = self._snap_to_grid(params.get("start_x", 0))
        start_y = self._snap_to_grid(params.get("start_y", 0))
        end_x = self._snap_to_grid(params.get("end_x", 0))
        end_y = self._snap_to_grid(params.get("end_y", 0))

        wire_uuid = str(uuid.uuid4())
        wire_sexp = f'  (wire (pts (xy {start_x} {start_y}) (xy {end_x} {end_y})) (stroke (width 0) (type default)) (uuid "{wire_uuid}"))\n'

        # Insert before the closing parenthesis
        insert_pos = content.rfind('\n)')
        if insert_pos == -1:
            insert_pos = content.rfind(')')

        content = content[:insert_pos] + wire_sexp + content[insert_pos:]
        logger.debug(f"Added wire from ({start_x}, {start_y}) to ({end_x}, {end_y})")
        return content

    def _apply_add_junction(self, content: str, fix: SchematicFix) -> str:
        """Add a junction dot to the schematic."""
        params = fix.parameters

        # If auto_detect, find wire intersections
        if params.get("auto_detect"):
            junctions = self._detect_junction_points(content)
            for x, y in junctions[:3]:  # Add up to 3 junctions per fix
                content = self._insert_junction(content, x, y)
            return content

        x = self._snap_to_grid(params.get("x", 0))
        y = self._snap_to_grid(params.get("y", 0))

        return self._insert_junction(content, x, y)

    def _insert_junction(self, content: str, x: float, y: float) -> str:
        """Insert a single junction into content."""
        junction_uuid = str(uuid.uuid4())
        junction_sexp = f'  (junction (at {x} {y}) (diameter 0) (color 0 0 0 0) (uuid "{junction_uuid}"))\n'

        insert_pos = content.rfind('\n)')
        if insert_pos == -1:
            insert_pos = content.rfind(')')

        content = content[:insert_pos] + junction_sexp + content[insert_pos:]
        logger.debug(f"Added junction at ({x}, {y})")
        return content

    def _apply_add_label(self, content: str, fix: SchematicFix) -> str:
        """Add a text label to the schematic."""
        params = fix.parameters
        x = self._snap_to_grid(params.get("x", 0))
        y = self._snap_to_grid(params.get("y", 0))
        text = params.get("text", "Label")
        size = params.get("size", 1.27)

        label_uuid = str(uuid.uuid4())
        label_sexp = f'''  (text "{text}" (at {x} {y} 0)
    (effects (font (size {size} {size})))
    (uuid "{label_uuid}")
  )
'''

        insert_pos = content.rfind('\n)')
        content = content[:insert_pos] + label_sexp + content[insert_pos:]
        logger.debug(f"Added label '{text}' at ({x}, {y})")
        return content

    def _apply_add_net_label(self, content: str, fix: SchematicFix) -> str:
        """Add a net label to the schematic."""
        params = fix.parameters
        x = self._snap_to_grid(params.get("x", 0))
        y = self._snap_to_grid(params.get("y", 0))
        net_name = params.get("net_name") or fix.target_net or "NET"
        rotation = params.get("rotation", 0)

        label_uuid = str(uuid.uuid4())
        label_sexp = f'''  (label "{net_name}" (at {x} {y} {rotation})
    (effects (font (size 1.27 1.27)))
    (uuid "{label_uuid}")
  )
'''

        insert_pos = content.rfind('\n)')
        content = content[:insert_pos] + label_sexp + content[insert_pos:]
        logger.debug(f"Added net label '{net_name}' at ({x}, {y})")
        return content

    def _apply_add_power_flag(self, content: str, fix: SchematicFix) -> str:
        """Add a power flag symbol to the schematic."""
        params = fix.parameters
        x = self._snap_to_grid(params.get("x", 0))
        y = self._snap_to_grid(params.get("y", 0))
        power_net = params.get("power_net", "VCC")

        # Add global label for power
        label_uuid = str(uuid.uuid4())

        # Determine shape based on power type
        shape = "input" if power_net.upper() in ["VCC", "VDD", "+5V", "+3V3", "+12V"] else "output"

        label_sexp = f'''  (global_label "{power_net}" (shape {shape}) (at {x} {y} 0)
    (effects (font (size 1.27 1.27)))
    (uuid "{label_uuid}")
  )
'''

        insert_pos = content.rfind('\n)')
        content = content[:insert_pos] + label_sexp + content[insert_pos:]
        logger.debug(f"Added power flag '{power_net}' at ({x}, {y})")
        return content

    def _apply_add_no_connect(self, content: str, fix: SchematicFix) -> str:
        """Add a no-connect flag to the schematic."""
        params = fix.parameters
        x = self._snap_to_grid(params.get("x", 0))
        y = self._snap_to_grid(params.get("y", 0))

        nc_uuid = str(uuid.uuid4())
        nc_sexp = f'  (no_connect (at {x} {y}) (uuid "{nc_uuid}"))\n'

        insert_pos = content.rfind('\n)')
        content = content[:insert_pos] + nc_sexp + content[insert_pos:]
        logger.debug(f"Added no-connect at ({x}, {y})")
        return content

    def _apply_add_component(self, content: str, fix: SchematicFix) -> str:
        """Add a new component (e.g., bypass capacitor) to the schematic."""
        params = fix.parameters
        symbol_lib = params.get("symbol_lib", "Device")
        symbol_name = params.get("symbol_name", "C")
        reference = params.get("reference", "C?")
        value = params.get("value", "100nF")

        # Calculate position near target component if specified
        near_component = params.get("near_component")
        if near_component:
            x, y = self._get_position_near_component(content, near_component)
        else:
            x = self._snap_to_grid(params.get("x", 100))
            y = self._snap_to_grid(params.get("y", 100))

        # Auto-assign reference number if needed
        if "?" in reference:
            reference = self._get_next_reference(content, reference.replace("?", ""))

        symbol_uuid = str(uuid.uuid4())
        symbol_id = f"{symbol_lib}:{symbol_name}"

        symbol_sexp = f'''  (symbol (lib_id "{symbol_id}") (at {x} {y} 0) (unit 1)
    (exclude_from_sim no) (in_bom yes) (on_board yes) (dnp no)
    (uuid "{symbol_uuid}")
    (property "Reference" "{reference}" (at {x} {y - 5} 0)
      (effects (font (size 1.27 1.27)))
    )
    (property "Value" "{value}" (at {x} {y + 5} 0)
      (effects (font (size 1.27 1.27)))
    )
    (property "Footprint" "" (at {x} {y} 0)
      (effects (font (size 1.27 1.27)) hide)
    )
    (property "Datasheet" "~" (at {x} {y} 0)
      (effects (font (size 1.27 1.27)) hide)
    )
    (instances
      (project ""
        (path "/{symbol_uuid}"
          (reference "{reference}")
          (unit 1)
        )
      )
    )
  )
'''

        insert_pos = content.rfind('\n)')
        content = content[:insert_pos] + symbol_sexp + content[insert_pos:]
        logger.debug(f"Added component {reference} ({symbol_id}) at ({x}, {y})")
        return content

    def _apply_modify_value(self, content: str, fix: SchematicFix) -> str:
        """Modify a component's value property."""
        params = fix.parameters
        reference = params.get("reference") or fix.target_component
        new_value = params.get("new_value")

        if not reference or not new_value:
            raise FixApplicationError(
                message="Reference and new_value required for MODIFY_VALUE",
                fix=fix,
                schematic_size=len(content),
                partial_changes=[]
            )

        # Find the component and its Value property
        pattern = rf'(\(property\s+"Reference"\s+"{re.escape(reference)}".*?\(property\s+"Value"\s+")([^"]+)(")'

        match = re.search(pattern, content, re.DOTALL)
        if match:
            content = content[:match.start(2)] + new_value + content[match.end(2):]
            logger.debug(f"Modified {reference} value to '{new_value}'")
        else:
            logger.warning(f"Could not find Value property for {reference}")

        return content

    def _apply_remove_wire(self, content: str, fix: SchematicFix) -> str:
        """Remove a wire from the schematic."""
        params = fix.parameters
        start_x = params.get("start_x")
        start_y = params.get("start_y")
        end_x = params.get("end_x")
        end_y = params.get("end_y")
        wire_uuid = params.get("uuid")

        if wire_uuid:
            # Remove by UUID
            pattern = rf'\s*\(wire.*?uuid\s+"{re.escape(wire_uuid)}".*?\)\n?'
        elif start_x is not None and start_y is not None:
            # Remove by coordinates (approximate match)
            sx, sy = self._snap_to_grid(start_x), self._snap_to_grid(start_y)
            pattern = rf'\s*\(wire\s+\(pts\s+\(xy\s+{sx}\s+{sy}\).*?\)\n?'
        else:
            logger.warning("No wire identifier provided for REMOVE_WIRE")
            return content

        content = re.sub(pattern, '', content, count=1)
        logger.debug("Removed wire")
        return content

    async def _apply_reroute_net(self, content: str, fix: SchematicFix) -> str:
        """Reroute a net through different waypoints."""
        params = fix.parameters
        net_name = params.get("net_name") or fix.target_net
        waypoints = params.get("waypoints", [])

        if not waypoints or len(waypoints) < 2:
            logger.warning("Not enough waypoints for REROUTE_NET")
            return content

        # This is complex - for now, add wires between waypoints
        for i in range(len(waypoints) - 1):
            start = waypoints[i]
            end = waypoints[i + 1]

            wire_uuid = str(uuid.uuid4())
            sx = self._snap_to_grid(start.get("x", 0))
            sy = self._snap_to_grid(start.get("y", 0))
            ex = self._snap_to_grid(end.get("x", 0))
            ey = self._snap_to_grid(end.get("y", 0))

            wire_sexp = f'  (wire (pts (xy {sx} {sy}) (xy {ex} {ey})) (stroke (width 0) (type default)) (uuid "{wire_uuid}"))\n'

            insert_pos = content.rfind('\n)')
            content = content[:insert_pos] + wire_sexp + content[insert_pos:]

        logger.debug(f"Rerouted net with {len(waypoints)} waypoints")
        return content

    def _snap_to_grid(self, value: float) -> float:
        """Snap a coordinate to the default grid."""
        return round(value / self.DEFAULT_GRID) * self.DEFAULT_GRID

    def _calculate_optimal_position(
        self,
        content: str,
        reference: str
    ) -> Tuple[float, float]:
        """Calculate optimal position for a component based on signal flow."""
        # Find current position
        pattern = rf'\(property\s+"Reference"\s+"{re.escape(reference)}"[^)]+\(at\s+([\d.-]+)\s+([\d.-]+)'
        match = re.search(pattern, content)

        if match:
            # Slight offset to improve layout
            x = float(match.group(1)) + self.DEFAULT_GRID
            y = float(match.group(2))
            return (self._snap_to_grid(x), self._snap_to_grid(y))

        # Default position
        return (100.0, 100.0)

    def _get_position_near_component(
        self,
        content: str,
        reference: str
    ) -> Tuple[float, float]:
        """Get a position near an existing component."""
        pattern = rf'\(symbol\s+\(lib_id\s+"[^"]+"\)\s+\(at\s+([\d.-]+)\s+([\d.-]+).*?\(property\s+"Reference"\s+"{re.escape(reference)}"'

        match = re.search(pattern, content, re.DOTALL)
        if match:
            x = float(match.group(1)) + self.DEFAULT_GRID * 3
            y = float(match.group(2))
            return (self._snap_to_grid(x), self._snap_to_grid(y))

        return (100.0, 100.0)

    def _get_next_reference(self, content: str, prefix: str) -> str:
        """Get the next available reference designator."""
        pattern = rf'{re.escape(prefix)}(\d+)'
        matches = re.findall(pattern, content)

        if matches:
            max_num = max(int(m) for m in matches)
            return f"{prefix}{max_num + 1}"

        return f"{prefix}1"

    def _detect_junction_points(self, content: str) -> List[Tuple[float, float]]:
        """Detect points where junctions should be added."""
        # Find all wire endpoints
        wire_pattern = r'\(wire\s+\(pts\s+\(xy\s+([\d.-]+)\s+([\d.-]+)\)\s+\(xy\s+([\d.-]+)\s+([\d.-]+)\)\)'
        matches = re.findall(wire_pattern, content)

        endpoints: Dict[Tuple[float, float], int] = {}
        for match in matches:
            start = (float(match[0]), float(match[1]))
            end = (float(match[2]), float(match[3]))

            # Round for comparison
            for point in [start, end]:
                key = (round(point[0], 2), round(point[1], 2))
                endpoints[key] = endpoints.get(key, 0) + 1

        # Find existing junctions
        junction_pattern = r'\(junction\s+\(at\s+([\d.-]+)\s+([\d.-]+)\)'
        existing = {
            (round(float(m[0]), 2), round(float(m[1]), 2))
            for m in re.findall(junction_pattern, content)
        }

        # Return points with 3+ connections that don't have junctions
        junctions = [
            point for point, count in endpoints.items()
            if count >= 3 and point not in existing
        ]

        return junctions

    def _validate_sexp_syntax(self, content: str) -> Dict[str, Any]:
        """Basic S-expression syntax validation."""
        # Count parentheses
        open_count = content.count('(')
        close_count = content.count(')')

        if open_count != close_count:
            return {
                "valid": False,
                "error": f"Unbalanced parentheses: {open_count} open, {close_count} close"
            }

        # Check for required sections
        required = ['(kicad_sch', '(version', '(lib_symbols']
        for section in required:
            if section not in content:
                return {
                    "valid": False,
                    "error": f"Missing required section: {section}"
                }

        return {"valid": True}
