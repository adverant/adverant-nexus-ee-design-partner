#!/usr/bin/env python3
"""
Master Schematic Generation Script

This script orchestrates the complete schematic generation pipeline:
1. Execute SKiDL circuit descriptions to generate netlists
2. Convert netlists to KiCad schematics
3. Run multi-level validation
4. Apply fixes if needed
5. Generate final output

Usage:
    python generate_all.py --project-dir /path/to/project --output-dir /path/to/output
"""

import argparse
import json
import os
import subprocess
import sys
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import importlib.util
import shutil


@dataclass
class SheetConfig:
    """Configuration for a single schematic sheet."""
    sheet_number: int
    name: str
    skidl_module: str
    description: str = ''
    output_netlist: str = ''
    output_schematic: str = ''


@dataclass
class GenerationConfig:
    """Complete generation configuration."""
    project_name: str
    project_dir: str
    output_dir: str
    sheets: List[SheetConfig]
    validation_levels: List[int]
    max_iterations: int = 10
    target_score: float = 90.0


@dataclass
class GenerationResult:
    """Result of generation process."""
    success: bool
    sheets_generated: int
    validation_passed: bool
    validation_score: float
    errors: List[str]
    warnings: List[str]
    duration_ms: int
    output_files: List[str]


class SchematicGenerationPipeline:
    """
    Orchestrates the complete schematic generation pipeline.

    This pipeline is designed to work with any circuit type:
    - Power electronics
    - Digital circuits
    - Analog circuits
    - RF/microwave
    - Mixed-signal designs
    """

    def __init__(self, config: GenerationConfig):
        self.config = config
        self.errors: List[str] = []
        self.warnings: List[str] = []
        self.output_files: List[str] = []

        # Get script directory
        self.script_dir = Path(__file__).parent

    def run(self) -> GenerationResult:
        """Execute the complete generation pipeline."""
        start_time = datetime.now()

        try:
            # Step 1: Setup output directories
            self._setup_directories()

            # Step 2: Generate netlists from SKiDL modules
            netlists = self._generate_netlists()

            if not netlists:
                return self._create_result(
                    success=False,
                    duration_ms=self._elapsed_ms(start_time)
                )

            # Step 3: Convert netlists to schematics
            schematic_path = self._convert_to_schematic(netlists)

            if not schematic_path:
                return self._create_result(
                    success=False,
                    duration_ms=self._elapsed_ms(start_time)
                )

            # Step 4: Run validation
            validation_result = self._run_validation(schematic_path)

            # Step 5: Iterative improvement if needed
            iteration = 0
            while (
                not validation_result['passed'] and
                iteration < self.config.max_iterations and
                validation_result.get('score', 0) < self.config.target_score
            ):
                iteration += 1
                print(f"\nIteration {iteration}: Applying fixes...")

                # Apply fixes based on validation errors
                self._apply_fixes(schematic_path, validation_result.get('errors', []))

                # Re-validate
                validation_result = self._run_validation(schematic_path)

            return self._create_result(
                success=validation_result['passed'],
                validation_passed=validation_result['passed'],
                validation_score=validation_result.get('score', 0),
                duration_ms=self._elapsed_ms(start_time)
            )

        except Exception as e:
            self.errors.append(f"Pipeline error: {str(e)}")
            return self._create_result(
                success=False,
                duration_ms=self._elapsed_ms(start_time)
            )

    def _setup_directories(self):
        """Create output directories."""
        dirs = [
            self.config.output_dir,
            os.path.join(self.config.output_dir, 'netlists'),
            os.path.join(self.config.output_dir, 'schematics'),
            os.path.join(self.config.output_dir, 'reports')
        ]

        for dir_path in dirs:
            os.makedirs(dir_path, exist_ok=True)

    def _generate_netlists(self) -> List[str]:
        """Generate netlists from SKiDL modules."""
        netlists = []

        for sheet in self.config.sheets:
            print(f"Generating netlist for {sheet.name}...")

            # Determine module path
            module_path = self._find_skidl_module(sheet.skidl_module)

            if not module_path:
                self.errors.append(f"SKiDL module not found: {sheet.skidl_module}")
                continue

            # Output netlist path
            netlist_path = os.path.join(
                self.config.output_dir,
                'netlists',
                f"{sheet.name}.net"
            )

            # Execute SKiDL module
            try:
                result = subprocess.run(
                    [sys.executable, module_path, '--output', netlist_path],
                    capture_output=True,
                    text=True,
                    timeout=120
                )

                if result.returncode == 0:
                    netlists.append(netlist_path)
                    sheet.output_netlist = netlist_path
                    print(f"  Generated: {netlist_path}")
                else:
                    self.errors.append(
                        f"Failed to generate {sheet.name}: {result.stderr}"
                    )
            except subprocess.TimeoutExpired:
                self.errors.append(f"Timeout generating {sheet.name}")
            except Exception as e:
                self.errors.append(f"Error generating {sheet.name}: {str(e)}")

        return netlists

    def _find_skidl_module(self, module_name: str) -> Optional[str]:
        """Find SKiDL module file."""
        # Check in skidl_circuits directory
        search_paths = [
            self.script_dir / 'skidl_circuits' / f"{module_name}.py",
            self.script_dir / 'skidl_circuits' / module_name / '__init__.py',
            Path(self.config.project_dir) / 'skidl_circuits' / f"{module_name}.py",
        ]

        for path in search_paths:
            if path.exists():
                return str(path)

        return None

    def _convert_to_schematic(self, netlists: List[str]) -> Optional[str]:
        """Convert netlists to KiCad schematic."""
        if not netlists:
            return None

        schematic_path = os.path.join(
            self.config.output_dir,
            'schematics',
            f"{self.config.project_name}.kicad_sch"
        )

        converter_script = self.script_dir / 'netlist_to_schematic.py'

        if not converter_script.exists():
            self.errors.append(f"Converter script not found: {converter_script}")
            return None

        try:
            result = subprocess.run(
                [
                    sys.executable,
                    str(converter_script),
                    '--netlists', ','.join(netlists),
                    '--output', schematic_path
                ],
                capture_output=True,
                text=True,
                timeout=300
            )

            if result.returncode == 0:
                self.output_files.append(schematic_path)
                print(f"Generated schematic: {schematic_path}")
                return schematic_path
            else:
                self.errors.append(f"Conversion failed: {result.stderr}")
                return None

        except Exception as e:
            self.errors.append(f"Conversion error: {str(e)}")
            return None

    def _run_validation(self, schematic_path: str) -> Dict[str, Any]:
        """Run multi-level validation on schematic."""
        validation_script = self.script_dir / 'validation_gate.py'

        if not validation_script.exists():
            self.warnings.append("Validation script not found, skipping validation")
            return {'passed': True, 'score': 100}

        results = {
            'passed': True,
            'score': 0,
            'errors': [],
            'warnings': [],
            'levels': []
        }

        total_score = 0
        level_count = 0

        for level in self.config.validation_levels:
            print(f"Running validation level {level}...")

            try:
                result = subprocess.run(
                    [
                        sys.executable,
                        str(validation_script),
                        '--level', str(level),
                        '--path', schematic_path,
                        '--json'
                    ],
                    capture_output=True,
                    text=True,
                    timeout=120
                )

                level_result = json.loads(result.stdout) if result.stdout else {}

                results['levels'].append(level_result)

                if not level_result.get('passed', True):
                    results['passed'] = False
                    results['errors'].extend(
                        [e.get('message', str(e)) for e in level_result.get('errors', [])]
                    )

                results['warnings'].extend(
                    [w.get('message', str(w)) for w in level_result.get('warnings', [])]
                )

                # Calculate score
                metrics = level_result.get('metrics', {})
                ratio = metrics.get('wire_component_ratio', 0)
                if ratio >= 1.2:
                    total_score += 100
                elif ratio >= 1.0:
                    total_score += 50
                level_count += 1

            except json.JSONDecodeError:
                self.warnings.append(f"Could not parse level {level} output")
            except subprocess.TimeoutExpired:
                self.errors.append(f"Validation level {level} timed out")
            except Exception as e:
                self.errors.append(f"Validation level {level} error: {str(e)}")

        results['score'] = total_score / max(level_count, 1)
        return results

    def _apply_fixes(self, schematic_path: str, errors: List[str]):
        """Apply fixes based on validation errors."""
        fix_script = self.script_dir / 'apply_fixes.py'

        if not fix_script.exists():
            self.warnings.append("Fix script not found, skipping auto-fix")
            return

        try:
            subprocess.run(
                [
                    sys.executable,
                    str(fix_script),
                    '--path', schematic_path,
                    '--errors', json.dumps(errors)
                ],
                capture_output=True,
                text=True,
                timeout=60
            )
        except Exception as e:
            self.warnings.append(f"Auto-fix error: {str(e)}")

    def _elapsed_ms(self, start_time: datetime) -> int:
        """Calculate elapsed time in milliseconds."""
        return int((datetime.now() - start_time).total_seconds() * 1000)

    def _create_result(
        self,
        success: bool,
        duration_ms: int,
        validation_passed: bool = False,
        validation_score: float = 0
    ) -> GenerationResult:
        """Create generation result."""
        return GenerationResult(
            success=success,
            sheets_generated=len([s for s in self.config.sheets if s.output_netlist]),
            validation_passed=validation_passed,
            validation_score=validation_score,
            errors=self.errors,
            warnings=self.warnings,
            duration_ms=duration_ms,
            output_files=self.output_files
        )


def main():
    parser = argparse.ArgumentParser(
        description='Master schematic generation script'
    )
    parser.add_argument(
        '--project-dir', '-p',
        type=str,
        required=True,
        help='Project directory containing SKiDL modules'
    )
    parser.add_argument(
        '--output-dir', '-o',
        type=str,
        required=True,
        help='Output directory for generated files'
    )
    parser.add_argument(
        '--config', '-c',
        type=str,
        help='JSON configuration file'
    )
    parser.add_argument(
        '--sheets', '-s',
        type=str,
        help='Comma-separated list of sheet modules to generate'
    )
    parser.add_argument(
        '--max-iterations', '-m',
        type=int,
        default=10,
        help='Maximum validation fix iterations'
    )
    parser.add_argument(
        '--json',
        action='store_true',
        help='Output result as JSON'
    )

    args = parser.parse_args()

    # Load or create configuration
    if args.config and os.path.exists(args.config):
        with open(args.config, 'r') as f:
            config_data = json.load(f)
        sheets = [SheetConfig(**s) for s in config_data.get('sheets', [])]
    elif args.sheets:
        sheets = []
        for i, module in enumerate(args.sheets.split(','), 1):
            module = module.strip()
            sheets.append(SheetConfig(
                sheet_number=i,
                name=module,
                skidl_module=module
            ))
    else:
        # Default: try to find all SKiDL modules
        sheets = []
        skidl_dir = Path(__file__).parent / 'skidl_circuits'
        if skidl_dir.exists():
            for i, py_file in enumerate(sorted(skidl_dir.glob('sheet*.py')), 1):
                sheets.append(SheetConfig(
                    sheet_number=i,
                    name=py_file.stem,
                    skidl_module=py_file.stem
                ))

    config = GenerationConfig(
        project_name=Path(args.project_dir).name,
        project_dir=args.project_dir,
        output_dir=args.output_dir,
        sheets=sheets,
        validation_levels=[1, 2, 3],
        max_iterations=args.max_iterations
    )

    # Run pipeline
    pipeline = SchematicGenerationPipeline(config)
    result = pipeline.run()

    # Output result
    if args.json:
        print(json.dumps(asdict(result), indent=2))
    else:
        print("\n" + "=" * 60)
        print("GENERATION RESULT")
        print("=" * 60)
        print(f"Success: {result.success}")
        print(f"Sheets Generated: {result.sheets_generated}")
        print(f"Validation: {'PASS' if result.validation_passed else 'FAIL'}")
        print(f"Score: {result.validation_score:.1f}%")
        print(f"Duration: {result.duration_ms}ms")

        if result.errors:
            print(f"\nErrors ({len(result.errors)}):")
            for err in result.errors:
                print(f"  - {err}")

        if result.warnings:
            print(f"\nWarnings ({len(result.warnings)}):")
            for warn in result.warnings:
                print(f"  - {warn}")

        if result.output_files:
            print(f"\nOutput Files:")
            for f in result.output_files:
                print(f"  - {f}")

    sys.exit(0 if result.success else 1)


if __name__ == '__main__':
    main()
