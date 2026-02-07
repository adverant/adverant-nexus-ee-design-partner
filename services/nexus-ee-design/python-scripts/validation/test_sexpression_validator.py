"""
Test Suite for S-expression Validator

Tests all validation checks with both valid and malformed examples.
"""

import pytest
from validation.sexpression_validator import (
    SExpressionValidator,
    ValidationError,
    SExpressionValidationReport
)


class TestBalancedParentheses:
    """Test parentheses balancing detection."""

    def test_extra_closing_parens(self):
        """Test detection of extra closing parentheses."""
        content = """(kicad_sch
\t(lib_symbols)
\t(symbol (lib_id "Device:R"))
)))"""  # 2 extra closing parens

        validator = SExpressionValidator()
        report = validator.validate_content(content)

        assert not report.valid
        errors = [e for e in report.errors if e.error_type == 'unbalanced_parens']
        assert len(errors) >= 1
        assert any('extra closing parenthesis' in e.message.lower() for e in errors)

    def test_unclosed_opening_parens(self):
        """Test detection of unclosed opening parentheses."""
        content = """(kicad_sch
\t(lib_symbols
\t(symbol (lib_id "Device:R"))"""  # Missing 2 closing parens

        validator = SExpressionValidator()
        report = validator.validate_content(content)

        assert not report.valid
        errors = [e for e in report.errors if e.error_type == 'unbalanced_parens']
        assert len(errors) == 1
        assert 'unclosed opening parentheses' in errors[0].message.lower()
        assert '2' in errors[0].message  # Should report 2 unclosed

    def test_balanced_parens_valid(self):
        """Test that balanced parentheses pass validation."""
        content = """(kicad_sch
\t(lib_symbols)
\t(symbol (lib_id "Device:R")))"""

        validator = SExpressionValidator()
        report = validator.validate_content(content)

        errors = [e for e in report.errors if e.error_type == 'unbalanced_parens']
        assert len(errors) == 0

    def test_parens_in_strings_ignored(self):
        """Test that parentheses inside strings are ignored."""
        content = """(kicad_sch
\t(lib_symbols)
\t(property "Value" "R (10k)"))"""  # Parens in string should be ignored

        validator = SExpressionValidator()
        report = validator.validate_content(content)

        errors = [e for e in report.errors if e.error_type == 'unbalanced_parens']
        assert len(errors) == 0


class TestIndentation:
    """Test indentation validation."""

    def test_space_indentation(self):
        """Test detection of space indentation."""
        content = """(kicad_sch
  (lib_symbols)
  (symbol (lib_id "R")))"""  # Uses spaces, not tabs

        validator = SExpressionValidator()
        report = validator.validate_content(content)

        assert not report.valid
        errors = [e for e in report.errors if e.error_type == 'invalid_indent']
        assert len(errors) >= 1
        assert any('spaces instead of tabs' in e.message.lower() for e in errors)

    def test_mixed_tabs_spaces(self):
        """Test detection of mixed tabs and spaces."""
        content = """(kicad_sch
\t (lib_symbols))"""  # Tab followed by space

        validator = SExpressionValidator()
        report = validator.validate_content(content)

        assert not report.valid
        errors = [e for e in report.errors if e.error_type == 'invalid_indent']
        assert len(errors) >= 1

    def test_pure_tab_indentation_valid(self):
        """Test that pure tab indentation passes."""
        content = """(kicad_sch
\t(lib_symbols)
\t\t(symbol "Device:R"))"""

        validator = SExpressionValidator()
        report = validator.validate_content(content)

        errors = [e for e in report.errors if e.error_type == 'invalid_indent']
        assert len(errors) == 0


class TestUUID:
    """Test UUID validation."""

    def test_invalid_uuid_format(self):
        """Test detection of invalid UUID format."""
        content = """(kicad_sch
\t(lib_symbols)
\t(uuid "invalid-uuid-format"))"""

        validator = SExpressionValidator()
        report = validator.validate_content(content)

        assert not report.valid
        errors = [e for e in report.errors if e.error_type == 'invalid_uuid']
        assert len(errors) == 1
        assert 'invalid-uuid-format' in errors[0].message

    def test_duplicate_uuid(self):
        """Test detection of duplicate UUIDs."""
        content = """(kicad_sch
\t(lib_symbols)
\t(symbol (uuid "12345678-1234-1234-1234-123456789abc"))
\t(symbol (uuid "12345678-1234-1234-1234-123456789abc")))"""  # Duplicate UUID

        validator = SExpressionValidator()
        report = validator.validate_content(content)

        assert not report.valid
        errors = [e for e in report.errors if e.error_type == 'duplicate_uuid']
        assert len(errors) == 1
        assert 'duplicate' in errors[0].message.lower()

    def test_valid_uuid(self):
        """Test that valid UUIDs pass."""
        content = """(kicad_sch
\t(lib_symbols)
\t(symbol (uuid "12345678-1234-1234-1234-123456789abc"))
\t(symbol (uuid "abcdef01-2345-6789-abcd-ef0123456789")))"""

        validator = SExpressionValidator()
        report = validator.validate_content(content)

        errors = [e for e in report.errors if e.error_type in ['invalid_uuid', 'duplicate_uuid']]
        assert len(errors) == 0

    def test_uuid_case_insensitive(self):
        """Test that UUID validation is case-insensitive."""
        content = """(kicad_sch
\t(lib_symbols)
\t(symbol (uuid "ABCDEF01-2345-6789-ABCD-EF0123456789")))"""

        validator = SExpressionValidator()
        report = validator.validate_content(content)

        errors = [e for e in report.errors if e.error_type == 'invalid_uuid']
        assert len(errors) == 0


class TestRequiredSections:
    """Test required section validation."""

    def test_missing_kicad_sch(self):
        """Test detection of missing kicad_sch section."""
        content = """(some_other_section
\t(data))"""

        validator = SExpressionValidator()
        report = validator.validate_content(content)

        assert not report.valid
        errors = [e for e in report.errors if e.error_type == 'missing_section']
        assert any('kicad_sch' in e.message for e in errors)

    def test_missing_lib_symbols(self):
        """Test detection of missing lib_symbols section."""
        content = """(kicad_sch
\t(version 20))"""

        validator = SExpressionValidator()
        report = validator.validate_content(content)

        assert not report.valid
        errors = [e for e in report.errors if e.error_type == 'missing_section']
        assert any('lib_symbols' in e.message for e in errors)

    def test_required_sections_present(self):
        """Test that required sections pass when present."""
        content = """(kicad_sch
\t(lib_symbols))"""

        validator = SExpressionValidator()
        report = validator.validate_content(content)

        errors = [e for e in report.errors if e.error_type == 'missing_section']
        assert len(errors) == 0


class TestCoordinates:
    """Test coordinate validation."""

    def test_non_numeric_coordinates(self):
        """Test detection of non-numeric coordinates."""
        content = """(kicad_sch
\t(lib_symbols)
\t(at abc def))"""

        validator = SExpressionValidator()
        report = validator.validate_content(content)

        # Should have warnings or errors for non-numeric coords
        coord_issues = [e for e in report.errors + report.warnings
                       if e.error_type == 'invalid_coordinate']
        assert len(coord_issues) >= 1

    def test_out_of_range_coordinates_warning(self):
        """Test that out-of-range coordinates generate warnings."""
        content = """(kicad_sch
\t(lib_symbols)
\t(at 2000 3000))"""  # Way out of range

        validator = SExpressionValidator()
        report = validator.validate_content(content)

        warnings = [w for w in report.warnings if w.error_type == 'invalid_coordinate']
        assert len(warnings) >= 1
        assert 'out of reasonable range' in warnings[0].message.lower()

    def test_valid_coordinates(self):
        """Test that valid coordinates pass."""
        content = """(kicad_sch
\t(lib_symbols)
\t(at 100.5 200.3)
\t(xy 50 75))"""

        validator = SExpressionValidator()
        report = validator.validate_content(content)

        coord_errors = [e for e in report.errors if e.error_type == 'invalid_coordinate']
        assert len(coord_errors) == 0


class TestReferences:
    """Test symbol reference validation."""

    def test_dangling_symbol_reference(self):
        """Test detection of symbol instances referencing undefined lib_symbols."""
        content = """(kicad_sch
\t(lib_symbols
\t\t(symbol "Device:R"
\t\t\t(property "Reference" "R"))
\t)
\t(symbol (lib_id "Device:C")))"""  # References Device:C which doesn't exist

        validator = SExpressionValidator()
        report = validator.validate_content(content)

        assert not report.valid
        errors = [e for e in report.errors if e.error_type == 'dangling_reference']
        assert len(errors) == 1
        assert 'Device:C' in errors[0].message

    def test_valid_symbol_references(self):
        """Test that valid symbol references pass."""
        content = """(kicad_sch
\t(lib_symbols
\t\t(symbol "Device:R"
\t\t\t(property "Reference" "R"))
\t\t(symbol "Device:C"
\t\t\t(property "Reference" "C"))
\t)
\t(symbol (lib_id "Device:R"))
\t(symbol (lib_id "Device:C")))"""

        validator = SExpressionValidator()
        report = validator.validate_content(content)

        errors = [e for e in report.errors if e.error_type == 'dangling_reference']
        assert len(errors) == 0


class TestStatistics:
    """Test statistics collection."""

    def test_statistics_collection(self):
        """Test that statistics are collected correctly."""
        content = """(kicad_sch
\t(lib_symbols
\t\t(symbol "Device:R"))
\t(symbol (lib_id "Device:R")
\t\t(uuid "12345678-1234-1234-1234-123456789abc"))
\t(wire (pts (xy 0 0) (xy 10 10)))
\t(junction (at 5 5))
\t(label "VCC"))"""

        validator = SExpressionValidator()
        report = validator.validate_content(content)

        stats = report.statistics
        assert stats['lib_symbols'] >= 1
        assert stats['symbol_instances'] >= 1
        assert stats['wire_count'] >= 1
        assert stats['junction_count'] >= 1
        assert stats['label_count'] >= 1
        assert stats['uuid_count'] >= 1


class TestCompleteValidation:
    """Test complete validation scenarios."""

    def test_completely_valid_schematic(self):
        """Test that a completely valid schematic passes all checks."""
        content = """(kicad_sch
\t(version 20230121)
\t(generator eeschema)
\t(lib_symbols
\t\t(symbol "Device:R"
\t\t\t(property "Reference" "R")
\t\t\t(property "Value" "R"))
\t)
\t(symbol (lib_id "Device:R")
\t\t(at 100 100 0)
\t\t(uuid "12345678-1234-1234-1234-123456789abc"))
\t(wire
\t\t(pts (xy 90 100) (xy 100 100))
\t\t(uuid "abcdef01-2345-6789-abcd-ef0123456789")))"""

        validator = SExpressionValidator()
        report = validator.validate_content(content)

        assert report.valid
        assert len(report.errors) == 0
        assert report.statistics['symbol_instances'] >= 1

    def test_multiple_errors_detected(self):
        """Test that multiple different errors are all detected."""
        content = """(kicad_sch
  (lib_symbols)
  (symbol (lib_id "Device:C"))
  (uuid "invalid-uuid")
)))"""  # Space indent, dangling ref, invalid uuid, extra parens

        validator = SExpressionValidator()
        report = validator.validate_content(content)

        assert not report.valid
        assert len(report.errors) >= 3

        # Should have at least one of each type
        error_types = {e.error_type for e in report.errors}
        assert 'invalid_indent' in error_types or 'unbalanced_parens' in error_types
        assert 'invalid_uuid' in error_types or 'unbalanced_parens' in error_types


class TestContextReporting:
    """Test error context reporting."""

    def test_context_includes_surrounding_lines(self):
        """Test that error context includes surrounding lines."""
        content = """(kicad_sch
\t(lib_symbols)
\t(symbol (lib_id "R"))
\t(uuid "invalid")
\t(more stuff))"""

        validator = SExpressionValidator()
        report = validator.validate_content(content)

        # Find an error with context
        errors_with_context = [e for e in report.errors if e.context]
        assert len(errors_with_context) > 0

        # Check that context contains line numbers and surrounding lines
        context = errors_with_context[0].context
        assert '>>>' in context  # Marker for error line
        assert any(char.isdigit() for char in context)  # Line numbers


# Integration test with file I/O
class TestFileValidation:
    """Test file-based validation."""

    def test_validate_file(self, tmp_path):
        """Test validation from file path."""
        # Create a test file
        test_file = tmp_path / "test_schematic.kicad_sch"
        content = """(kicad_sch
\t(lib_symbols))"""

        test_file.write_text(content)

        validator = SExpressionValidator()
        report = validator.validate_file(str(test_file))

        assert report.file_path == str(test_file)
        assert report.total_lines > 0

    def test_file_not_found(self):
        """Test handling of non-existent file."""
        validator = SExpressionValidator()

        with pytest.raises(FileNotFoundError):
            validator.validate_file("/nonexistent/file.kicad_sch")


if __name__ == '__main__':
    # Run tests with pytest
    pytest.main([__file__, '-v', '--tb=short'])
