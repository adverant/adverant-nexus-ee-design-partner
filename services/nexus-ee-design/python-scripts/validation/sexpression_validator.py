"""
S-expression Validator for KiCad 8.x Schematic Files

Strict parser that catches malformed KiCad S-expression syntax BEFORE files are written,
preventing issues like:
- Unbalanced parentheses (e.g., 192 extra closing parentheses)
- Invalid UUID formats
- Incorrect indentation (spaces instead of tabs)
- Missing required sections
- Dangling references (references to non-existent symbols/nets)

This validator is designed to prevent KiCad from failing to open or crashing on generated files.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Set, Tuple
import re


@dataclass
class ValidationError:
    """A single validation error found."""
    line_number: int
    column: int
    error_type: str  # 'unbalanced_parens' | 'invalid_uuid' | 'invalid_indent' | ...
    severity: str  # 'error' | 'warning'
    message: str
    suggestion: str = ""
    context: str = ""  # Surrounding lines for context


@dataclass
class SExpressionValidationReport:
    """Complete validation report."""
    valid: bool
    file_path: str
    total_lines: int
    errors: List[ValidationError] = field(default_factory=list)
    warnings: List[ValidationError] = field(default_factory=list)
    statistics: Dict[str, int] = field(default_factory=dict)


class SExpressionValidator:
    """
    Strict validator for KiCad 8.x S-expression schematic files.

    Validates:
    - Balanced parentheses
    - Pure tab indentation (no spaces)
    - Required sections
    - UUID format and uniqueness
    - Coordinate validity
    - Symbol/net references
    """

    # UUID regex: 8-4-4-4-12 hex digits
    UUID_PATTERN = re.compile(
        r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$',
        re.IGNORECASE
    )

    # Required top-level sections
    REQUIRED_SECTIONS = ['kicad_sch', 'lib_symbols']

    def validate_file(self, file_path: str) -> SExpressionValidationReport:
        """
        Validate a .kicad_sch file.

        Args:
            file_path: Path to .kicad_sch file

        Returns:
            SExpressionValidationReport with detailed errors
        """
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        return self.validate_content(content, file_path)

    def validate_content(
        self,
        content: str,
        file_path: str = "<string>"
    ) -> SExpressionValidationReport:
        """
        Validate S-expression content string.

        Args:
            content: S-expression content to validate
            file_path: Path for reporting (optional)

        Returns:
            SExpressionValidationReport with all errors and warnings
        """
        lines = content.split('\n')
        errors: List[ValidationError] = []
        warnings: List[ValidationError] = []

        # Check 1: Balanced parentheses
        paren_errors = self._check_balanced_parentheses(lines)
        errors.extend(paren_errors)

        # Check 2: Indentation (pure tabs)
        indent_errors = self._check_indentation(lines)
        errors.extend(indent_errors)

        # Check 3: Required sections
        section_errors = self._check_required_sections(content)
        errors.extend(section_errors)

        # Check 4: UUID validation
        uuid_errors = self._check_uuids(lines)
        errors.extend(uuid_errors)

        # Check 5: Coordinate validation
        coord_errors = self._check_coordinates(lines)
        warnings.extend(coord_errors)  # Warnings, not blocking

        # Check 6: Symbol references
        ref_errors = self._check_references(content, lines)
        errors.extend(ref_errors)

        # Collect statistics
        statistics = self._collect_statistics(content)

        valid = len(errors) == 0

        return SExpressionValidationReport(
            valid=valid,
            file_path=file_path,
            total_lines=len(lines),
            errors=errors,
            warnings=warnings,
            statistics=statistics
        )

    def _check_balanced_parentheses(
        self,
        lines: List[str]
    ) -> List[ValidationError]:
        """
        Check for balanced parentheses throughout the file.

        Track depth at every character, detect unbalanced.
        Reports exact line and column of mismatch.
        """
        errors = []
        depth = 0
        in_string = False
        escape_next = False

        for line_num, line in enumerate(lines, start=1):
            for col, char in enumerate(line):
                # Handle string escaping
                if escape_next:
                    escape_next = False
                    continue

                if char == '\\' and in_string:
                    escape_next = True
                    continue

                # Toggle string state
                if char == '"':
                    in_string = not in_string
                    continue

                # Only count parentheses outside of strings
                if not in_string:
                    if char == '(':
                        depth += 1
                    elif char == ')':
                        depth -= 1

                        if depth < 0:
                            errors.append(ValidationError(
                                line_number=line_num,
                                column=col,
                                error_type='unbalanced_parens',
                                severity='error',
                                message=f"Extra closing parenthesis (depth went negative)",
                                suggestion="Remove this closing parenthesis",
                                context=self._get_context(lines, line_num)
                            ))
                            depth = 0  # Reset to continue checking

        # Check final depth
        if depth > 0:
            errors.append(ValidationError(
                line_number=len(lines),
                column=0,
                error_type='unbalanced_parens',
                severity='error',
                message=f"{depth} unclosed opening parentheses",
                suggestion=f"Add {depth} closing parentheses at end of file",
                context=""
            ))

        return errors

    def _check_indentation(
        self,
        lines: List[str]
    ) -> List[ValidationError]:
        """
        Check that indentation uses ONLY tabs, no spaces.

        KiCad 8.x requires pure tab indentation.
        """
        errors = []

        for line_num, line in enumerate(lines, start=1):
            # Skip empty lines
            if not line.strip():
                continue

            # Get indentation (count leading tabs)
            stripped = line.lstrip('\t')
            indent_tabs = len(line) - len(stripped)

            # Check for spaces in indentation
            if line.startswith(' '):
                errors.append(ValidationError(
                    line_number=line_num,
                    column=0,
                    error_type='invalid_indent',
                    severity='error',
                    message="Indentation uses spaces instead of tabs",
                    suggestion="Replace leading spaces with tabs",
                    context=self._get_context(lines, line_num)
                ))

            # Check for mixed tabs and spaces in leading whitespace
            leading_whitespace = line[:len(line) - len(stripped)]
            if '\t' in leading_whitespace and ' ' in leading_whitespace:
                errors.append(ValidationError(
                    line_number=line_num,
                    column=0,
                    error_type='invalid_indent',
                    severity='error',
                    message="Mixed tabs and spaces in indentation",
                    suggestion="Use only tabs for indentation",
                    context=self._get_context(lines, line_num)
                ))

        return errors

    def _check_required_sections(
        self,
        content: str
    ) -> List[ValidationError]:
        """
        Check that required top-level sections exist.
        """
        errors = []

        for section in self.REQUIRED_SECTIONS:
            if f'({section}' not in content:
                errors.append(ValidationError(
                    line_number=1,
                    column=0,
                    error_type='missing_section',
                    severity='error',
                    message=f"Required section '({section} ...)' not found",
                    suggestion=f"Add ({section} ...) section to schematic"
                ))

        return errors

    def _check_uuids(
        self,
        lines: List[str]
    ) -> List[ValidationError]:
        """
        Validate all UUIDs in the file.

        Check format and uniqueness.
        """
        errors = []
        seen_uuids: Set[str] = set()

        for line_num, line in enumerate(lines, start=1):
            # Find UUID declarations: (uuid "xxx-xxx-xxx")
            if '(uuid' in line:
                match = re.search(r'\(uuid\s+"([^"]+)"\)', line)
                if match:
                    uuid_str = match.group(1)

                    # Validate format
                    if not self.UUID_PATTERN.match(uuid_str):
                        errors.append(ValidationError(
                            line_number=line_num,
                            column=line.index('(uuid'),
                            error_type='invalid_uuid',
                            severity='error',
                            message=f"Invalid UUID format: {uuid_str}",
                            suggestion="Use format: xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx",
                            context=self._get_context(lines, line_num)
                        ))

                    # Check uniqueness
                    if uuid_str in seen_uuids:
                        errors.append(ValidationError(
                            line_number=line_num,
                            column=line.index('(uuid'),
                            error_type='duplicate_uuid',
                            severity='error',
                            message=f"Duplicate UUID: {uuid_str}",
                            suggestion="Generate a new unique UUID",
                            context=self._get_context(lines, line_num)
                        ))
                    else:
                        seen_uuids.add(uuid_str)

        return errors

    def _check_coordinates(
        self,
        lines: List[str]
    ) -> List[ValidationError]:
        """
        Validate coordinate values are numeric and reasonable.

        Returns warnings for out-of-range coordinates.
        """
        errors = []

        for line_num, line in enumerate(lines, start=1):
            # Find (at x y) or (xy x y) patterns
            at_matches = re.findall(r'\(at\s+([\d.-]+)\s+([\d.-]+)', line)
            xy_matches = re.findall(r'\(xy\s+([\d.-]+)\s+([\d.-]+)', line)

            for x_str, y_str in at_matches + xy_matches:
                try:
                    x = float(x_str)
                    y = float(y_str)

                    # Check reasonable ranges (-1000mm to +1000mm)
                    if abs(x) > 1000 or abs(y) > 1000:
                        errors.append(ValidationError(
                            line_number=line_num,
                            column=0,
                            error_type='invalid_coordinate',
                            severity='warning',
                            message=f"Coordinate out of reasonable range: ({x}, {y})",
                            suggestion="Typical schematics use -500mm to +500mm range"
                        ))
                except ValueError:
                    errors.append(ValidationError(
                        line_number=line_num,
                        column=0,
                        error_type='invalid_coordinate',
                        severity='error',
                        message=f"Non-numeric coordinate: ({x_str}, {y_str})",
                        suggestion="Coordinates must be numeric values"
                    ))

        return errors

    def _check_references(
        self,
        content: str,
        lines: List[str]
    ) -> List[ValidationError]:
        """
        Validate all symbol/net references point to valid targets.

        Checks:
        - Symbol instances reference defined lib_symbols
        - No dangling references
        """
        errors = []

        # Extract all lib_symbol definitions (in lib_symbols section)
        lib_symbols: Set[str] = set()

        # Find lib_symbols section
        lib_symbols_match = re.search(r'\(lib_symbols(.*?)\n\)', content, re.DOTALL)
        if lib_symbols_match:
            lib_symbols_content = lib_symbols_match.group(1)
            # Find all (symbol "lib_id:name" ...) definitions
            for match in re.finditer(r'\(symbol\s+"([^"]+)"', lib_symbols_content):
                lib_symbols.add(match.group(1))

        # Extract all symbol instances and check their lib_id references
        # Pattern: (symbol (lib_id "Device:R") ...)
        for match in re.finditer(r'\(symbol\s+\(lib_id\s+"([^"]+)"\)', content):
            lib_id = match.group(1)

            # Check if lib_id exists in lib_symbols
            if lib_id not in lib_symbols:
                # Find line number
                line_num = content[:match.start()].count('\n') + 1
                errors.append(ValidationError(
                    line_number=line_num,
                    column=0,
                    error_type='dangling_reference',
                    severity='error',
                    message=f"Symbol instance references undefined lib_symbol: {lib_id}",
                    suggestion=f"Add (symbol \"{lib_id}\" ...) definition in lib_symbols section",
                    context=self._get_context(lines, line_num) if line_num <= len(lines) else ""
                ))

        return errors

    def _collect_statistics(self, content: str) -> Dict[str, int]:
        """
        Collect counts of various elements.

        Returns dictionary with element counts for reporting.
        """
        return {
            'symbol_instances': len(re.findall(r'\(symbol\s+\(lib_id', content)),
            'lib_symbols': len(re.findall(r'\(lib_symbols', content)),
            'wire_count': content.count('(wire '),
            'junction_count': content.count('(junction '),
            'label_count': content.count('(label '),
            'net_count': content.count('(net '),
            'uuid_count': content.count('(uuid '),
        }

    def _get_context(
        self,
        lines: List[str],
        line_num: int,
        context_lines: int = 2
    ) -> str:
        """
        Get surrounding lines for error context.

        Args:
            lines: All file lines
            line_num: Line number with error (1-indexed)
            context_lines: Number of lines before/after to show

        Returns:
            Formatted context string with line numbers
        """
        start = max(0, line_num - context_lines - 1)
        end = min(len(lines), line_num + context_lines)

        context_lines_list = []
        for i in range(start, end):
            marker = '>>>' if i == line_num - 1 else '   '
            context_lines_list.append(f"{marker} {i+1}: {lines[i]}")

        return '\n'.join(context_lines_list)


# CLI Tool
if __name__ == '__main__':
    import sys
    import argparse

    parser = argparse.ArgumentParser(
        description='Validate KiCad 8.x S-expression schematic files'
    )
    parser.add_argument('file', help='Path to .kicad_sch file')
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Show warnings and statistics'
    )
    parser.add_argument(
        '--max-errors',
        type=int,
        default=10,
        help='Maximum errors to display (default: 10)'
    )

    args = parser.parse_args()

    # Validate file
    validator = SExpressionValidator()
    try:
        report = validator.validate_file(args.file)
    except FileNotFoundError:
        print(f"❌ ERROR: File not found: {args.file}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"❌ ERROR: Failed to validate file: {e}", file=sys.stderr)
        sys.exit(1)

    # Print header
    print(f"Validating: {report.file_path}")
    print(f"Total lines: {report.total_lines}")

    if args.verbose:
        print(f"\nStatistics:")
        for key, value in report.statistics.items():
            print(f"  {key}: {value}")

    print()

    # Print errors
    if report.errors:
        print(f"❌ VALIDATION FAILED: {len(report.errors)} error(s) found")
        print()

        errors_to_show = report.errors[:args.max_errors]
        for i, error in enumerate(errors_to_show, 1):
            print(f"Error {i}/{len(report.errors)}:")
            print(f"  [{error.severity.upper()}] Line {error.line_number}:{error.column}")
            print(f"  Type: {error.error_type}")
            print(f"  {error.message}")
            if error.suggestion:
                print(f"  💡 Suggestion: {error.suggestion}")
            if error.context:
                print(f"  Context:")
                for line in error.context.split('\n'):
                    print(f"    {line}")
            print()

        if len(report.errors) > args.max_errors:
            remaining = len(report.errors) - args.max_errors
            print(f"... and {remaining} more error(s)")
            print(f"Use --max-errors={len(report.errors)} to see all errors")
            print()
    else:
        print("✅ VALIDATION PASSED - No errors found")

    # Print warnings
    if args.verbose and report.warnings:
        print(f"\n⚠️  {len(report.warnings)} warning(s):")
        for warning in report.warnings:
            print(f"  Line {warning.line_number}: {warning.message}")

    # Exit with appropriate code
    sys.exit(0 if report.valid else 1)
