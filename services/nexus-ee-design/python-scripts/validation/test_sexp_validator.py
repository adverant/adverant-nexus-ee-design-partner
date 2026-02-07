"""
Comprehensive Test Suite for Enhanced S-Expression Validator

Tests deterministic AST parsing, Pydantic schemas, and all validation rules.
"""

import pytest
import tempfile
from pathlib import Path

try:
    from validation.sexp_validator import (
        SExpValidator,
        validate_schematic_file,
        validate_schematic_content,
        KiCadSchematic,
        KiCadCoordinate,
        KiCadUUID,
        KiCadSymbolInstance,
        KiCadWire,
        KiCadJunction,
        KiCadLabel,
        ValidationSeverity,
        ValidationIssue,
        SExpValidationResult,
        SEXPDATA_AVAILABLE
    )
except ImportError as e:
    pytest.skip(f"Could not import sexp_validator: {e}", allow_module_level=True)


# Skip all tests if sexpdata is not available
pytestmark = pytest.mark.skipif(
    not SEXPDATA_AVAILABLE,
    reason="sexpdata library not installed"
)


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def valid_minimal_schematic():
    """Minimal valid schematic."""
    return '''(kicad_sch (version 20231120) (generator "test")
  (uuid "12345678-1234-1234-1234-123456789abc")
  (paper "A4")
  (lib_symbols
    (symbol "Device:R"
      (property "Reference" "R" (at 0 0 0))
      (property "Value" "R" (at 0 0 0))
    )
  )
)'''


@pytest.fixture
def valid_schematic_with_symbols():
    """Schematic with placed symbols."""
    return '''(kicad_sch (version 20231120) (generator "test")
  (uuid "12345678-1234-1234-1234-123456789abc")
  (paper "A4")
  (lib_symbols
    (symbol "Device:R"
      (property "Reference" "R" (at 0 0 0))
      (property "Value" "R" (at 0 0 0))
    )
  )
  (symbol (lib_id "Device:R")
    (at 100 100 0)
    (uuid "abcdef01-2345-6789-abcd-ef0123456789")
    (property "Reference" "R1" (at 0 0 0))
    (property "Value" "10k" (at 0 0 0))
  )
)'''


@pytest.fixture
def valid_schematic_with_wires():
    """Schematic with wires and junctions."""
    return '''(kicad_sch (version 20231120) (generator "test")
  (uuid "12345678-1234-1234-1234-123456789abc")
  (paper "A4")
  (lib_symbols)
  (wire
    (pts (xy 0 0) (xy 10 0) (xy 10 10))
    (stroke (width 0) (type default))
    (uuid "aaaaaaaa-1111-1111-1111-111111111111")
  )
  (junction
    (at 10 0)
    (diameter 0)
    (uuid "bbbbbbbb-2222-2222-2222-222222222222")
  )
  (label "VCC"
    (at 10 10 0)
    (uuid "cccccccc-3333-3333-3333-333333333333")
  )
)'''


@pytest.fixture
def invalid_duplicate_uuid():
    """Schematic with duplicate UUIDs."""
    return '''(kicad_sch (version 20231120) (generator "test")
  (uuid "12345678-1234-1234-1234-123456789abc")
  (paper "A4")
  (lib_symbols
    (symbol "Device:R"
      (uuid "aaaaaaaa-1111-1111-1111-111111111111")
    )
  )
  (symbol (lib_id "Device:R")
    (at 100 100 0)
    (uuid "aaaaaaaa-1111-1111-1111-111111111111")
    (property "Reference" "R1" (at 0 0 0))
  )
)'''


@pytest.fixture
def invalid_dangling_reference():
    """Schematic with symbol instance referencing non-existent lib_symbol."""
    return '''(kicad_sch (version 20231120) (generator "test")
  (uuid "12345678-1234-1234-1234-123456789abc")
  (paper "A4")
  (lib_symbols
    (symbol "Device:R"
      (property "Reference" "R" (at 0 0 0))
    )
  )
  (symbol (lib_id "Device:C")
    (at 100 100 0)
    (uuid "abcdef01-2345-6789-abcd-ef0123456789")
    (property "Reference" "C1" (at 0 0 0))
  )
)'''


@pytest.fixture
def schematic_off_grid():
    """Schematic with coordinates not aligned to grid."""
    return '''(kicad_sch (version 20231120) (generator "test")
  (uuid "12345678-1234-1234-1234-123456789abc")
  (paper "A4")
  (lib_symbols
    (symbol "Device:R"
      (property "Reference" "R" (at 0 0 0))
    )
  )
  (symbol (lib_id "Device:R")
    (at 100.123 200.456 0)
    (uuid "abcdef01-2345-6789-abcd-ef0123456789")
    (property "Reference" "R1" (at 0 0 0))
  )
  (wire
    (pts (xy 1.23 4.56) (xy 7.89 10.11))
    (uuid "aaaaaaaa-1111-1111-1111-111111111111")
  )
)'''


@pytest.fixture
def schematic_with_diagonal_wire():
    """Schematic with diagonal (non-Manhattan) wire."""
    return '''(kicad_sch (version 20231120) (generator "test")
  (uuid "12345678-1234-1234-1234-123456789abc")
  (paper "A4")
  (lib_symbols)
  (wire
    (pts (xy 0 0) (xy 10 10))
    (uuid "aaaaaaaa-1111-1111-1111-111111111111")
  )
)'''


# ============================================================================
# Test Pydantic Schemas
# ============================================================================

class TestPydanticSchemas:
    """Test Pydantic schema validation."""

    def test_kicad_coordinate_creation(self):
        """Test KiCadCoordinate creation."""
        coord = KiCadCoordinate(x=100.0, y=200.0)
        assert coord.x == 100.0
        assert coord.y == 200.0
        assert str(coord) == "(100.0, 200.0)"

    def test_kicad_coordinate_grid_alignment(self):
        """Test grid alignment checking."""
        # On grid (2.54mm = 100 mil)
        coord_on_grid = KiCadCoordinate(x=2.54, y=5.08)
        assert coord_on_grid.is_on_grid(2.54)

        # Off grid
        coord_off_grid = KiCadCoordinate(x=2.5, y=5.1)
        assert not coord_off_grid.is_on_grid(2.54)

        # Origin is always on grid
        coord_origin = KiCadCoordinate(x=0, y=0)
        assert coord_origin.is_on_grid(2.54)

    def test_kicad_uuid_validation(self):
        """Test UUID format validation."""
        # Valid UUID
        valid_uuid = KiCadUUID(value="12345678-1234-1234-1234-123456789abc")
        assert str(valid_uuid) == "12345678-1234-1234-1234-123456789abc"

        # Invalid UUID format
        with pytest.raises(ValueError):
            KiCadUUID(value="invalid-uuid")

        # UUID comparison
        uuid1 = KiCadUUID(value="12345678-1234-1234-1234-123456789abc")
        uuid2 = KiCadUUID(value="12345678-1234-1234-1234-123456789abc")
        uuid3 = KiCadUUID(value="aaaaaaaa-1111-1111-1111-111111111111")

        assert uuid1 == uuid2
        assert uuid1 != uuid3

    def test_kicad_wire_manhattan_check(self):
        """Test Manhattan routing detection."""
        # Manhattan wire (horizontal then vertical)
        wire_manhattan = KiCadWire(
            pts=[
                KiCadCoordinate(x=0, y=0),
                KiCadCoordinate(x=10, y=0),
                KiCadCoordinate(x=10, y=10)
            ],
            uuid=KiCadUUID(value="12345678-1234-1234-1234-123456789abc")
        )
        assert wire_manhattan.is_manhattan()

        # Diagonal wire
        wire_diagonal = KiCadWire(
            pts=[
                KiCadCoordinate(x=0, y=0),
                KiCadCoordinate(x=10, y=10)
            ],
            uuid=KiCadUUID(value="12345678-1234-1234-1234-123456789abc")
        )
        assert not wire_diagonal.is_manhattan()

    def test_kicad_wire_length(self):
        """Test wire length calculation."""
        wire = KiCadWire(
            pts=[
                KiCadCoordinate(x=0, y=0),
                KiCadCoordinate(x=3, y=0),
                KiCadCoordinate(x=3, y=4)
            ],
            uuid=KiCadUUID(value="12345678-1234-1234-1234-123456789abc")
        )
        # 3 + 4 = 7
        assert abs(wire.length() - 7.0) < 0.01

    def test_kicad_schematic_uuid_collection(self):
        """Test collecting all UUIDs from schematic."""
        schematic = KiCadSchematic(
            version="20231120",
            generator="test",
            uuid=KiCadUUID(value="12345678-1234-1234-1234-123456789abc"),
            wires=[
                KiCadWire(
                    pts=[KiCadCoordinate(x=0, y=0), KiCadCoordinate(x=10, y=0)],
                    uuid=KiCadUUID(value="aaaaaaaa-1111-1111-1111-111111111111")
                )
            ]
        )

        all_uuids = schematic.get_all_uuids()
        assert len(all_uuids) == 2
        assert KiCadUUID(value="12345678-1234-1234-1234-123456789abc") in all_uuids
        assert KiCadUUID(value="aaaaaaaa-1111-1111-1111-111111111111") in all_uuids


# ============================================================================
# Test S-Expression Parsing
# ============================================================================

class TestSExpressionParsing:
    """Test S-expression parsing to AST."""

    def test_parse_minimal_schematic(self, valid_minimal_schematic):
        """Test parsing minimal valid schematic."""
        validator = SExpValidator()
        result = validator.validate_content(valid_minimal_schematic)

        assert result.valid
        assert result.schematic is not None
        assert result.schematic.version == "20231120"
        assert result.schematic.generator == "test"
        assert len(result.schematic.lib_symbols) == 1
        assert "Device:R" in result.schematic.lib_symbols

    def test_parse_schematic_with_symbols(self, valid_schematic_with_symbols):
        """Test parsing schematic with placed symbols."""
        validator = SExpValidator()
        result = validator.validate_content(valid_schematic_with_symbols)

        assert result.valid
        assert result.schematic is not None
        assert len(result.schematic.symbol_instances) == 1

        inst = result.schematic.symbol_instances[0]
        assert inst.lib_id == "Device:R"
        assert inst.reference == "R1"
        assert inst.value == "10k"
        assert inst.at.x == 100
        assert inst.at.y == 100

    def test_parse_schematic_with_wires(self, valid_schematic_with_wires):
        """Test parsing schematic with wires and junctions."""
        validator = SExpValidator()
        result = validator.validate_content(valid_schematic_with_wires)

        assert result.valid
        assert result.schematic is not None
        assert len(result.schematic.wires) == 1
        assert len(result.schematic.junctions) == 1
        assert len(result.schematic.labels) == 1

        wire = result.schematic.wires[0]
        assert len(wire.pts) == 3

        label = result.schematic.labels[0]
        assert label.text == "VCC"

    def test_parse_invalid_sexpression(self):
        """Test parsing invalid S-expression syntax."""
        invalid_content = "(kicad_sch (version 20231120) (unclosed"

        validator = SExpValidator()
        result = validator.validate_content(invalid_content)

        assert not result.valid
        assert len(result.errors) > 0
        assert result.errors[0].category == "syntax"


# ============================================================================
# Test Validation Rules
# ============================================================================

class TestValidationRules:
    """Test all validation rules."""

    def test_valid_schematic_passes(self, valid_schematic_with_symbols):
        """Test that valid schematic passes all checks."""
        validator = SExpValidator()
        result = validator.validate_content(valid_schematic_with_symbols)

        assert result.valid
        assert len(result.errors) == 0

    def test_duplicate_uuid_detection(self, invalid_duplicate_uuid):
        """Test detection of duplicate UUIDs."""
        validator = SExpValidator()
        result = validator.validate_content(invalid_duplicate_uuid)

        assert not result.valid
        assert any(e.category == "uuid" for e in result.errors)
        assert any("duplicate" in e.message.lower() for e in result.errors)

    def test_dangling_reference_detection(self, invalid_dangling_reference):
        """Test detection of dangling symbol references."""
        validator = SExpValidator()
        result = validator.validate_content(invalid_dangling_reference)

        assert not result.valid
        assert any(e.category == "reference" for e in result.errors)
        assert any("Device:C" in e.message for e in result.errors)

    def test_grid_alignment_warning(self, schematic_off_grid):
        """Test grid alignment warnings."""
        validator = SExpValidator(check_grid_alignment=True, grid_size_mm=2.54)
        result = validator.validate_content(schematic_off_grid)

        # Should have warnings about off-grid elements
        assert len(result.warnings) > 0
        assert any(w.category == "grid" for w in result.warnings)

    def test_grid_alignment_can_be_disabled(self, schematic_off_grid):
        """Test that grid alignment checks can be disabled."""
        validator = SExpValidator(check_grid_alignment=False)
        result = validator.validate_content(schematic_off_grid)

        # Should have no grid warnings
        grid_warnings = [w for w in result.warnings if w.category == "grid"]
        assert len(grid_warnings) == 0

    def test_diagonal_wire_detection(self, schematic_with_diagonal_wire):
        """Test detection of diagonal (non-Manhattan) wires."""
        validator = SExpValidator(check_electrical=True)
        result = validator.validate_content(schematic_with_diagonal_wire)

        # Should have info message about diagonal routing
        assert len(result.info) > 0
        assert any("diagonal" in i.message.lower() for i in result.info)

    def test_electrical_checks_can_be_disabled(self, schematic_with_diagonal_wire):
        """Test that electrical checks can be disabled."""
        validator = SExpValidator(check_electrical=False)
        result = validator.validate_content(schematic_with_diagonal_wire)

        # Should have no electrical info messages
        electrical_info = [i for i in result.info if i.category == "electrical"]
        assert len(electrical_info) == 0

    def test_strict_mode_fails_on_warnings(self, schematic_off_grid):
        """Test that strict mode treats warnings as errors."""
        validator = SExpValidator(strict_mode=True, check_grid_alignment=True)
        result = validator.validate_content(schematic_off_grid)

        # Strict mode should fail if there are warnings
        if len(result.warnings) > 0:
            assert not result.valid


# ============================================================================
# Test File I/O
# ============================================================================

class TestFileIO:
    """Test file-based validation."""

    def test_validate_file(self, valid_schematic_with_symbols, tmp_path):
        """Test validation from file."""
        # Write schematic to temp file
        test_file = tmp_path / "test.kicad_sch"
        test_file.write_text(valid_schematic_with_symbols)

        validator = SExpValidator()
        result = validator.validate_file(test_file)

        assert result.valid
        assert result.file_path == str(test_file)

    def test_validate_nonexistent_file(self):
        """Test validation of non-existent file."""
        validator = SExpValidator()
        result = validator.validate_file("/nonexistent/file.kicad_sch")

        assert not result.valid
        assert len(result.errors) > 0
        assert "not found" in result.errors[0].message.lower()

    def test_validate_file_convenience_function(self, valid_schematic_with_symbols, tmp_path):
        """Test convenience function for file validation."""
        test_file = tmp_path / "test.kicad_sch"
        test_file.write_text(valid_schematic_with_symbols)

        result = validate_schematic_file(test_file)

        assert result.valid

    def test_validate_content_convenience_function(self, valid_schematic_with_symbols):
        """Test convenience function for content validation."""
        result = validate_schematic_content(valid_schematic_with_symbols)

        assert result.valid


# ============================================================================
# Test Statistics Collection
# ============================================================================

class TestStatistics:
    """Test statistics collection."""

    def test_statistics_collection(self, valid_schematic_with_wires):
        """Test that statistics are collected correctly."""
        validator = SExpValidator()
        result = validator.validate_content(valid_schematic_with_wires)

        assert result.statistics is not None
        assert "wires" in result.statistics
        assert "junctions" in result.statistics
        assert "labels" in result.statistics
        assert result.statistics["wires"] == 1
        assert result.statistics["junctions"] == 1
        assert result.statistics["labels"] == 1


# ============================================================================
# Test Validation Result
# ============================================================================

class TestValidationResult:
    """Test validation result structure."""

    def test_validation_result_summary(self, valid_minimal_schematic):
        """Test validation result summary generation."""
        validator = SExpValidator()
        result = validator.validate_content(valid_minimal_schematic)

        summary = result.summary()
        assert "✅" in summary or "PASSED" in summary

    def test_validation_result_all_issues(self, invalid_duplicate_uuid):
        """Test getting all issues from result."""
        validator = SExpValidator()
        result = validator.validate_content(invalid_duplicate_uuid)

        all_issues = result.all_issues()
        assert len(all_issues) >= len(result.errors)

    def test_validation_result_critical_issues(self, invalid_duplicate_uuid):
        """Test getting only critical issues."""
        validator = SExpValidator()
        result = validator.validate_content(invalid_duplicate_uuid)

        critical = result.critical_issues()
        assert len(critical) == len(result.errors)
        assert all(i.severity == ValidationSeverity.ERROR for i in critical)


# ============================================================================
# Integration Tests
# ============================================================================

class TestIntegration:
    """Integration tests with real-world scenarios."""

    def test_complex_schematic(self):
        """Test validation of complex schematic with multiple elements."""
        complex_schematic = '''(kicad_sch (version 20231120) (generator "test")
  (uuid "12345678-1234-1234-1234-123456789abc")
  (paper "A4")
  (lib_symbols
    (symbol "Device:R"
      (property "Reference" "R" (at 0 0 0))
      (property "Value" "R" (at 0 0 0))
    )
    (symbol "Device:C"
      (property "Reference" "C" (at 0 0 0))
      (property "Value" "C" (at 0 0 0))
    )
  )
  (symbol (lib_id "Device:R")
    (at 100 100 0)
    (uuid "aaaaaaaa-1111-1111-1111-111111111111")
    (property "Reference" "R1" (at 0 0 0))
    (property "Value" "10k" (at 0 0 0))
  )
  (symbol (lib_id "Device:C")
    (at 150 100 0)
    (uuid "bbbbbbbb-2222-2222-2222-222222222222")
    (property "Reference" "C1" (at 0 0 0))
    (property "Value" "100nF" (at 0 0 0))
  )
  (wire
    (pts (xy 100 100) (xy 150 100))
    (uuid "cccccccc-3333-3333-3333-333333333333")
  )
  (junction
    (at 125 100)
    (uuid "dddddddd-4444-4444-4444-444444444444")
  )
  (label "GND"
    (at 150 100 0)
    (uuid "eeeeeeee-5555-5555-5555-555555555555")
  )
)'''

        validator = SExpValidator()
        result = validator.validate_content(complex_schematic)

        assert result.valid
        assert result.schematic is not None
        assert len(result.schematic.lib_symbols) == 2
        assert len(result.schematic.symbol_instances) == 2
        assert len(result.schematic.wires) == 1
        assert len(result.schematic.junctions) == 1
        assert len(result.schematic.labels) == 1

    def test_multiple_validation_issues(self):
        """Test schematic with multiple validation issues."""
        problematic_schematic = '''(kicad_sch (version 20231120) (generator "test")
  (uuid "12345678-1234-1234-1234-123456789abc")
  (paper "A4")
  (lib_symbols
    (symbol "Device:R"
      (uuid "aaaaaaaa-1111-1111-1111-111111111111")
    )
  )
  (symbol (lib_id "Device:R")
    (at 100.123 200.456 0)
    (uuid "aaaaaaaa-1111-1111-1111-111111111111")
    (property "Reference" "R1" (at 0 0 0))
  )
  (symbol (lib_id "Device:C")
    (at 150 250 0)
    (uuid "bbbbbbbb-2222-2222-2222-222222222222")
    (property "Reference" "C1" (at 0 0 0))
  )
  (wire
    (pts (xy 1.23 4.56) (xy 7.89 10.11))
    (uuid "cccccccc-3333-3333-3333-333333333333")
  )
)'''

        validator = SExpValidator(check_grid_alignment=True, check_electrical=True)
        result = validator.validate_content(problematic_schematic)

        assert not result.valid

        # Should have UUID duplicate error
        assert any(e.category == "uuid" for e in result.errors)

        # Should have dangling reference error (Device:C not defined)
        assert any(e.category == "reference" for e in result.errors)

        # Should have grid alignment warnings
        assert any(w.category == "grid" for w in result.warnings)

        # Should have diagonal wire info
        assert any(i.category == "electrical" for i in result.info)


# ============================================================================
# Run Tests
# ============================================================================

if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
