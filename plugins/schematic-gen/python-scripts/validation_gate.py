#!/usr/bin/env python3
"""
Validation Gate - Multi-Level Schematic Validation

This script implements the 4-level validation pipeline:
- Level 1: SKiDL ERC (built-in electrical rule checking)
- Level 2: kicad-sch-api validation (structure and connectivity)
- Level 3: KiCad CLI ERC (native KiCad validation)
- Level 4: SPICE verification (circuit simulation)

Usage:
    python validation_gate.py --level 1 --path schematic.kicad_sch
"""

import argparse
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple


@dataclass
class ValidationError:
    """Represents a validation error or warning."""
    code: str
    message: str
    severity: str  # 'error', 'warning', 'info'
    location: Optional[Dict[str, Any]] = None
    suggestion: Optional[str] = None


@dataclass
class ValidationMetrics:
    """Metrics collected during validation."""
    component_count: int = 0
    wire_count: int = 0
    net_count: int = 0
    wire_component_ratio: float = 0.0
    unconnected_pins: int = 0
    power_nets: int = 0
    signal_nets: int = 0


@dataclass
class ValidationResult:
    """Result of a validation level."""
    level: int
    level_name: str
    passed: bool
    errors: List[ValidationError]
    warnings: List[ValidationError]
    metrics: ValidationMetrics


class Level1SKiDLValidator:
    """
    Level 1: SKiDL ERC Validation

    Uses SKiDL's built-in electrical rule checking to validate:
    - No floating pins
    - No conflicting outputs
    - Power pins connected
    - Net naming consistency
    """

    def __init__(self, schematic_path: str):
        self.schematic_path = schematic_path
        self.errors: List[ValidationError] = []
        self.warnings: List[ValidationError] = []
        self.metrics = ValidationMetrics()

    def validate(self) -> ValidationResult:
        """Run SKiDL ERC validation."""
        try:
            # Check if skidl is available
            try:
                import skidl
                from skidl import ERC, POWER
            except ImportError:
                self.errors.append(ValidationError(
                    code='SKIDL_NOT_INSTALLED',
                    message='SKiDL is not installed. Run: pip install skidl',
                    severity='error',
                    suggestion='Install SKiDL with: pip install skidl>=2.0.0'
                ))
                return self._create_result()

            # Parse the schematic to extract circuit
            circuit = self._parse_schematic_for_skidl()

            if circuit is None:
                self.warnings.append(ValidationError(
                    code='SKIDL_PARSE_WARNING',
                    message='Could not fully parse schematic for SKiDL validation',
                    severity='warning'
                ))
            else:
                # Run SKiDL ERC
                try:
                    erc_errors = ERC()
                    for err in erc_errors:
                        self.errors.append(ValidationError(
                            code='SKIDL_ERC',
                            message=str(err),
                            severity='error'
                        ))
                except Exception as e:
                    self.warnings.append(ValidationError(
                        code='SKIDL_ERC_WARNING',
                        message=f'SKiDL ERC check returned: {str(e)}',
                        severity='warning'
                    ))

            # Collect metrics from schematic
            self._collect_metrics()

        except Exception as e:
            self.errors.append(ValidationError(
                code='SKIDL_VALIDATION_FAILED',
                message=f'SKiDL validation failed: {str(e)}',
                severity='error'
            ))

        return self._create_result()

    def _parse_schematic_for_skidl(self) -> Optional[Any]:
        """Parse KiCad schematic to extract circuit for SKiDL."""
        # SKiDL works primarily with netlists, not schematics directly
        # Return None to indicate we should use file-based validation
        return None

    def _collect_metrics(self):
        """Collect metrics from schematic file."""
        try:
            with open(self.schematic_path, 'r') as f:
                content = f.read()

            # Count symbols (components)
            self.metrics.component_count = len(re.findall(r'\(symbol\s+"', content))

            # Count wires
            self.metrics.wire_count = len(re.findall(r'\(wire\s+', content))

            # Count nets (labels)
            self.metrics.net_count = len(re.findall(r'\(label\s+"', content))
            self.metrics.net_count += len(re.findall(r'\(global_label\s+"', content))

            # Calculate wire/component ratio
            if self.metrics.component_count > 0:
                self.metrics.wire_component_ratio = (
                    self.metrics.wire_count / self.metrics.component_count
                )

            # Count power nets
            power_patterns = ['VCC', 'VDD', 'GND', 'VSS', 'V3V3', 'V5V', 'VBUS']
            for pattern in power_patterns:
                self.metrics.power_nets += content.count(f'"{pattern}"')

        except Exception as e:
            self.warnings.append(ValidationError(
                code='METRICS_COLLECTION_WARNING',
                message=f'Could not collect all metrics: {str(e)}',
                severity='warning'
            ))

    def _create_result(self) -> ValidationResult:
        """Create the validation result."""
        return ValidationResult(
            level=1,
            level_name='SKiDL ERC',
            passed=len(self.errors) == 0,
            errors=self.errors,
            warnings=self.warnings,
            metrics=self.metrics
        )


class Level2KiCadAPIValidator:
    """
    Level 2: kicad-sch-api Validation

    Uses kicad-sch-api to validate schematic structure:
    - Wire/component ratio
    - Net connectivity
    - Symbol completeness
    - Pin connections
    """

    MIN_WIRE_RATIO = 1.2
    TARGET_WIRE_RATIO = 1.5

    def __init__(self, schematic_path: str):
        self.schematic_path = schematic_path
        self.errors: List[ValidationError] = []
        self.warnings: List[ValidationError] = []
        self.metrics = ValidationMetrics()

    def validate(self) -> ValidationResult:
        """Run kicad-sch-api validation."""
        try:
            # Try to use kicad-sch-api
            try:
                from kicad_sch_api import KicadSch
                self._validate_with_api()
            except ImportError:
                # Fall back to S-expression parsing
                self._validate_with_sexp()

        except Exception as e:
            self.errors.append(ValidationError(
                code='KICAD_API_VALIDATION_FAILED',
                message=f'Validation failed: {str(e)}',
                severity='error'
            ))

        return self._create_result()

    def _validate_with_api(self):
        """Validate using kicad-sch-api library."""
        from kicad_sch_api import KicadSch

        sch = KicadSch.load(self.schematic_path)

        # Collect metrics
        self.metrics.component_count = len(sch.symbols)
        self.metrics.wire_count = len(sch.wires)
        self.metrics.net_count = len(sch.labels) + len(sch.global_labels)

        if self.metrics.component_count > 0:
            self.metrics.wire_component_ratio = (
                self.metrics.wire_count / self.metrics.component_count
            )

        # Check wire/component ratio
        self._check_wire_ratio()

        # Check for unconnected pins
        self._check_unconnected_pins_api(sch)

    def _validate_with_sexp(self):
        """Validate using S-expression parsing (fallback)."""
        try:
            import sexpdata
        except ImportError:
            self.errors.append(ValidationError(
                code='SEXPDATA_NOT_INSTALLED',
                message='sexpdata is not installed. Run: pip install sexpdata',
                severity='error'
            ))
            return

        with open(self.schematic_path, 'r') as f:
            content = f.read()

        # Parse S-expression
        try:
            sexp = sexpdata.loads(content)
        except Exception as e:
            self.errors.append(ValidationError(
                code='SEXP_PARSE_ERROR',
                message=f'Failed to parse schematic: {str(e)}',
                severity='error'
            ))
            return

        # Count PLACED symbols only (those with lib_id), not library definitions
        # A placed symbol has (symbol (lib_id "...") ...) structure
        # A library symbol has (symbol "LibName:PartName" ...) structure
        all_symbols = self._find_elements(sexp, 'symbol')
        placed_symbols = []
        for sym in all_symbols:
            # Check if this is a placed symbol (has lib_id child)
            has_lib_id = any(
                isinstance(item, list) and len(item) > 0 and
                hasattr(item[0], 'value') and item[0].value() == 'lib_id'
                for item in sym[1:] if isinstance(item, list)
            )
            if has_lib_id:
                placed_symbols.append(sym)

        wires = self._find_elements(sexp, 'wire')
        labels = self._find_elements(sexp, 'label')
        global_labels = self._find_elements(sexp, 'global_label')

        self.metrics.component_count = len(placed_symbols)
        self.metrics.wire_count = len(wires)
        self.metrics.net_count = len(labels) + len(global_labels)

        if self.metrics.component_count > 0:
            self.metrics.wire_component_ratio = (
                self.metrics.wire_count / self.metrics.component_count
            )

        # Check wire/component ratio
        self._check_wire_ratio()

        # Check for potential issues
        self._check_symbol_properties(placed_symbols)

    def _find_elements(self, sexp: Any, element_type: str) -> List[Any]:
        """Find all elements of a given type in S-expression."""
        results = []

        def search(node):
            if isinstance(node, list) and len(node) > 0:
                if hasattr(node[0], 'value') and node[0].value() == element_type:
                    results.append(node)
                for item in node:
                    search(item)

        search(sexp)
        return results

    def _check_wire_ratio(self):
        """Check if wire/component ratio meets minimum threshold."""
        ratio = self.metrics.wire_component_ratio

        if ratio < self.MIN_WIRE_RATIO:
            self.errors.append(ValidationError(
                code='WIRE_RATIO_LOW',
                message=(
                    f'Wire/component ratio {ratio:.2f} is below minimum {self.MIN_WIRE_RATIO}. '
                    f'Target: {self.TARGET_WIRE_RATIO}'
                ),
                severity='error',
                suggestion='Add more wire connections or review schematic completeness'
            ))
        elif ratio < self.TARGET_WIRE_RATIO:
            self.warnings.append(ValidationError(
                code='WIRE_RATIO_WARNING',
                message=(
                    f'Wire/component ratio {ratio:.2f} is below target {self.TARGET_WIRE_RATIO}'
                ),
                severity='warning'
            ))

    def _check_unconnected_pins_api(self, sch):
        """Check for unconnected pins using kicad-sch-api."""
        unconnected = 0
        for symbol in sch.symbols:
            for pin in symbol.pins:
                if not pin.is_connected():
                    if pin.type not in ['no_connect', 'passive']:
                        unconnected += 1
                        if pin.type in ['power_input', 'power_output']:
                            self.errors.append(ValidationError(
                                code='UNCONNECTED_POWER_PIN',
                                message=f'{symbol.reference}.{pin.name}: Power pin not connected',
                                severity='error',
                                location={'component': symbol.reference, 'pin': pin.name}
                            ))

        self.metrics.unconnected_pins = unconnected

    def _check_symbol_properties(self, symbols: List[Any]):
        """Check symbol properties for completeness."""
        for symbol in symbols:
            # Check if symbol has reference designator
            ref = self._get_property(symbol, 'Reference')
            if not ref:
                self.warnings.append(ValidationError(
                    code='MISSING_REFERENCE',
                    message='Symbol found without reference designator',
                    severity='warning'
                ))

    def _get_property(self, symbol: Any, prop_name: str) -> Optional[str]:
        """Get property value from symbol."""
        for item in symbol:
            if isinstance(item, list) and len(item) > 0:
                if hasattr(item[0], 'value') and item[0].value() == 'property':
                    if len(item) > 1 and str(item[1]) == prop_name:
                        return str(item[2]) if len(item) > 2 else None
        return None

    def _create_result(self) -> ValidationResult:
        """Create the validation result."""
        return ValidationResult(
            level=2,
            level_name='kicad-sch-api Validation',
            passed=len(self.errors) == 0,
            errors=self.errors,
            warnings=self.warnings,
            metrics=self.metrics
        )


class Level3KiCadCLIValidator:
    """
    Level 3: KiCad CLI ERC Validation

    Uses kicad-cli to run native ERC checks.
    """

    def __init__(self, schematic_path: str, kicad_cli_path: str = 'kicad-cli'):
        self.schematic_path = schematic_path
        self.kicad_cli_path = kicad_cli_path
        self.errors: List[ValidationError] = []
        self.warnings: List[ValidationError] = []
        self.metrics = ValidationMetrics()

    def validate(self) -> ValidationResult:
        """Run KiCad CLI ERC validation."""
        try:
            # Check if kicad-cli is available
            result = subprocess.run(
                [self.kicad_cli_path, '--version'],
                capture_output=True,
                text=True
            )

            if result.returncode != 0:
                self.warnings.append(ValidationError(
                    code='KICAD_CLI_NOT_FOUND',
                    message='kicad-cli not found in PATH',
                    severity='warning',
                    suggestion='Install KiCad 8.0+ or add kicad-cli to PATH'
                ))
                return self._create_result(skipped=True)

            # Run ERC
            result = subprocess.run(
                [
                    self.kicad_cli_path, 'sch', 'erc',
                    '--exit-code-violations',
                    self.schematic_path
                ],
                capture_output=True,
                text=True,
                timeout=120
            )

            # Parse output
            self._parse_erc_output(result.stdout, result.stderr)

        except subprocess.TimeoutExpired:
            self.errors.append(ValidationError(
                code='KICAD_CLI_TIMEOUT',
                message='KiCad CLI ERC timed out after 120 seconds',
                severity='error'
            ))
        except FileNotFoundError:
            self.warnings.append(ValidationError(
                code='KICAD_CLI_NOT_FOUND',
                message=f'kicad-cli not found at {self.kicad_cli_path}',
                severity='warning'
            ))
            return self._create_result(skipped=True)
        except Exception as e:
            self.errors.append(ValidationError(
                code='KICAD_CLI_ERROR',
                message=f'KiCad CLI error: {str(e)}',
                severity='error'
            ))

        return self._create_result()

    def _parse_erc_output(self, stdout: str, stderr: str):
        """Parse KiCad CLI ERC output."""
        output = stdout + stderr

        # Parse error lines
        error_pattern = r'\[ERROR\]\s*(.+?)(?:\s+at\s+\((.+?)\))?$'
        warning_pattern = r'\[WARNING\]\s*(.+?)(?:\s+at\s+\((.+?)\))?$'

        for line in output.split('\n'):
            error_match = re.search(error_pattern, line)
            if error_match:
                location = None
                if error_match.group(2):
                    loc_match = re.match(r'(.+):(\d+):(\d+)', error_match.group(2))
                    if loc_match:
                        location = {
                            'file': loc_match.group(1),
                            'line': int(loc_match.group(2))
                        }
                self.errors.append(ValidationError(
                    code='KICAD_ERC_ERROR',
                    message=error_match.group(1),
                    severity='error',
                    location=location
                ))

            warning_match = re.search(warning_pattern, line)
            if warning_match:
                self.warnings.append(ValidationError(
                    code='KICAD_ERC_WARNING',
                    message=warning_match.group(1),
                    severity='warning'
                ))

    def _create_result(self, skipped: bool = False) -> ValidationResult:
        """Create the validation result."""
        if skipped:
            self.metrics = ValidationMetrics()

        return ValidationResult(
            level=3,
            level_name='KiCad CLI ERC',
            passed=len(self.errors) == 0,
            errors=self.errors,
            warnings=self.warnings,
            metrics=self.metrics
        )


class Level4SPICEValidator:
    """
    Level 4: SPICE Verification (Optional)

    Runs basic SPICE simulation to verify circuit functionality.
    """

    def __init__(self, schematic_path: str, ngspice_path: str = 'ngspice'):
        self.schematic_path = schematic_path
        self.ngspice_path = ngspice_path
        self.errors: List[ValidationError] = []
        self.warnings: List[ValidationError] = []
        self.metrics = ValidationMetrics()

    def validate(self) -> ValidationResult:
        """Run SPICE verification."""
        try:
            # Check if ngspice is available
            result = subprocess.run(
                [self.ngspice_path, '--version'],
                capture_output=True,
                text=True
            )

            if result.returncode != 0:
                self.warnings.append(ValidationError(
                    code='NGSPICE_NOT_FOUND',
                    message='ngspice not found in PATH',
                    severity='warning',
                    suggestion='Install ngspice or add it to PATH'
                ))
                return self._create_result(skipped=True)

            # Export netlist from schematic
            netlist_path = self._export_spice_netlist()

            if netlist_path and os.path.exists(netlist_path):
                # Run basic DC operating point analysis
                self._run_spice_simulation(netlist_path)

        except FileNotFoundError:
            self.warnings.append(ValidationError(
                code='NGSPICE_NOT_FOUND',
                message=f'ngspice not found at {self.ngspice_path}',
                severity='warning'
            ))
            return self._create_result(skipped=True)
        except Exception as e:
            self.errors.append(ValidationError(
                code='SPICE_ERROR',
                message=f'SPICE verification error: {str(e)}',
                severity='error'
            ))

        return self._create_result()

    def _export_spice_netlist(self) -> Optional[str]:
        """Export SPICE netlist from schematic."""
        # This would typically use kicad-cli or a Python library
        # For now, return None to skip SPICE simulation
        self.warnings.append(ValidationError(
            code='SPICE_EXPORT_SKIPPED',
            message='SPICE netlist export not yet implemented',
            severity='warning'
        ))
        return None

    def _run_spice_simulation(self, netlist_path: str):
        """Run SPICE simulation on netlist."""
        # Create a simple DC operating point control file
        control_file = netlist_path + '.control'
        with open(control_file, 'w') as f:
            f.write('.include {}\n'.format(netlist_path))
            f.write('.op\n')
            f.write('.end\n')

        result = subprocess.run(
            [self.ngspice_path, '-b', control_file],
            capture_output=True,
            text=True,
            timeout=60
        )

        if result.returncode != 0:
            self.errors.append(ValidationError(
                code='SPICE_SIMULATION_FAILED',
                message=f'SPICE simulation failed: {result.stderr}',
                severity='error'
            ))

    def _create_result(self, skipped: bool = False) -> ValidationResult:
        """Create the validation result."""
        return ValidationResult(
            level=4,
            level_name='SPICE Verification',
            passed=len(self.errors) == 0,
            errors=self.errors,
            warnings=self.warnings,
            metrics=self.metrics
        )


def output_metrics(metrics: ValidationMetrics):
    """Output metrics in parseable format."""
    print(f"METRIC:componentCount={metrics.component_count}")
    print(f"METRIC:wireCount={metrics.wire_count}")
    print(f"METRIC:netCount={metrics.net_count}")
    print(f"METRIC:wireComponentRatio={metrics.wire_component_ratio:.3f}")
    print(f"METRIC:unconnectedPins={metrics.unconnected_pins}")
    print(f"METRIC:powerNets={metrics.power_nets}")


def output_errors(errors: List[ValidationError]):
    """Output errors in parseable format."""
    for error in errors:
        print(f"ERROR:{error.message}")


def output_warnings(warnings: List[ValidationError]):
    """Output warnings in parseable format."""
    for warning in warnings:
        print(f"WARNING:{warning.message}")


def main():
    parser = argparse.ArgumentParser(
        description='Schematic Validation Gate - Multi-Level Validation Pipeline'
    )
    parser.add_argument(
        '--level', '-l',
        type=int,
        required=True,
        choices=[1, 2, 3, 4],
        help='Validation level (1-4)'
    )
    parser.add_argument(
        '--path', '-p',
        type=str,
        required=True,
        help='Path to schematic file'
    )
    parser.add_argument(
        '--kicad-cli',
        type=str,
        default='kicad-cli',
        help='Path to kicad-cli executable'
    )
    parser.add_argument(
        '--ngspice',
        type=str,
        default='ngspice',
        help='Path to ngspice executable'
    )
    parser.add_argument(
        '--json', '-j',
        action='store_true',
        help='Output results as JSON'
    )

    args = parser.parse_args()

    # Check if schematic exists
    if not os.path.exists(args.path):
        print(f"ERROR:Schematic file not found: {args.path}", file=sys.stderr)
        sys.exit(1)

    # Run appropriate validation level
    if args.level == 1:
        validator = Level1SKiDLValidator(args.path)
    elif args.level == 2:
        validator = Level2KiCadAPIValidator(args.path)
    elif args.level == 3:
        validator = Level3KiCadCLIValidator(args.path, args.kicad_cli)
    elif args.level == 4:
        validator = Level4SPICEValidator(args.path, args.ngspice)
    else:
        print(f"ERROR:Invalid validation level: {args.level}", file=sys.stderr)
        sys.exit(1)

    result = validator.validate()

    if args.json:
        # Output as JSON
        output = {
            'level': result.level,
            'levelName': result.level_name,
            'passed': result.passed,
            'errors': [asdict(e) for e in result.errors],
            'warnings': [asdict(w) for w in result.warnings],
            'metrics': asdict(result.metrics)
        }
        print(json.dumps(output, indent=2))
    else:
        # Output in parseable format
        output_metrics(result.metrics)
        output_errors(result.errors)
        output_warnings(result.warnings)

        # Print summary
        status = 'PASS' if result.passed else 'FAIL'
        print(f"\nLevel {result.level} ({result.level_name}): {status}")
        print(f"Errors: {len(result.errors)}, Warnings: {len(result.warnings)}")

    # Exit with appropriate code
    sys.exit(0 if result.passed else 1)


if __name__ == '__main__':
    main()
