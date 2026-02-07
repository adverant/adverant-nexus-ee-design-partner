"""
Test suite for Visual Verification Agent

Tests image generation, Opus 4.5 analysis, and quality scoring.
"""

import asyncio
import json
import os
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

import pytest

from visual_verifier import (
    VisualVerifier,
    VisualQualityReport,
    QualityIssue,
    VerificationError,
    verify_schematic
)


# Test data directory
SCRIPT_DIR = Path(__file__).parent
TEST_DATA_DIR = SCRIPT_DIR / "test_data"
RUBRIC_PATH = SCRIPT_DIR / "quality_rubric.yaml"


@pytest.fixture
def temp_output_dir():
    """Create temporary directory for test outputs."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def verifier():
    """Create VisualVerifier instance with test rubric."""
    # Mock API key for tests
    os.environ['OPENROUTER_API_KEY'] = 'test_key_12345'
    return VisualVerifier(rubric_path=RUBRIC_PATH)


@pytest.fixture
def mock_opus_response_excellent():
    """Mock response for excellent schematic."""
    return {
        "criterion_scores": {
            "symbol_overlap": {"score": 100, "issues": "None", "suggestion": ""},
            "wire_crossings": {"score": 100, "issues": "None", "suggestion": ""},
            "signal_flow": {"score": 95, "issues": "Excellent left-to-right flow", "suggestion": ""},
            "power_flow": {"score": 100, "issues": "None", "suggestion": ""},
            "functional_grouping": {"score": 95, "issues": "Very clear grouping", "suggestion": ""},
            "net_labels": {"score": 100, "issues": "None", "suggestion": ""},
            "spacing": {"score": 100, "issues": "None", "suggestion": ""},
            "professional_appearance": {"score": 100, "issues": "None", "suggestion": ""}
        },
        "overall_assessment": "Exceptional schematic. Production-ready with textbook-quality layout.",
        "production_ready": True
    }


@pytest.fixture
def mock_opus_response_poor():
    """Mock response for poor quality schematic."""
    return {
        "criterion_scores": {
            "symbol_overlap": {"score": 40, "issues": "U1, U2, and U3 overlap", "suggestion": "Reposition U2 and U3"},
            "wire_crossings": {"score": 30, "issues": "25 wire crossings throughout", "suggestion": "Complete layout redesign needed"},
            "signal_flow": {"score": 20, "issues": "No clear signal flow direction", "suggestion": "Reorganize left-to-right"},
            "power_flow": {"score": 50, "issues": "Power rails inconsistent", "suggestion": "Move power symbols to top/bottom"},
            "functional_grouping": {"score": 30, "issues": "No functional grouping visible", "suggestion": "Group related components"},
            "net_labels": {"score": 40, "issues": "Many labels overlap with wires and symbols", "suggestion": "Reposition all labels"},
            "spacing": {"score": 20, "issues": "Components extremely cramped", "suggestion": "Use larger canvas, increase spacing"},
            "professional_appearance": {"score": 25, "issues": "Unprofessional appearance, not production-ready", "suggestion": "Complete redesign recommended"}
        },
        "overall_assessment": "Poor quality schematic with major issues. Not production-ready. Requires significant rework.",
        "production_ready": False
    }


class TestVisualVerifier:
    """Test VisualVerifier class."""

    def test_init_success(self, verifier):
        """Test successful initialization."""
        assert verifier.model == "anthropic/claude-opus-4-5"
        assert verifier.api_key == "test_key_12345"
        assert len(verifier.rubric) == 8
        assert verifier.thresholds['pass'] == 90

    def test_init_no_api_key(self):
        """Test initialization fails without API key."""
        os.environ.pop('OPENROUTER_API_KEY', None)
        with pytest.raises(VerificationError, match="API key not found"):
            VisualVerifier(rubric_path=RUBRIC_PATH)

    def test_init_missing_rubric(self):
        """Test initialization fails with missing rubric."""
        with pytest.raises(VerificationError, match="Quality rubric not found"):
            VisualVerifier(rubric_path="/nonexistent/rubric.yaml")

    def test_build_analysis_prompt(self, verifier):
        """Test prompt construction includes all rubric items."""
        prompt = verifier._build_analysis_prompt()

        assert "symbol_overlap" in prompt
        assert "wire_crossings" in prompt
        assert "professional_appearance" in prompt
        assert "JSON" in prompt
        assert "criterion_scores" in prompt

    def test_build_report_excellent(self, verifier, mock_opus_response_excellent):
        """Test report generation for excellent schematic."""
        report = verifier._build_report(
            mock_opus_response_excellent,
            "/tmp/test.png"
        )

        assert report.passed is True
        assert report.overall_score >= 95.0
        assert report.production_ready is True
        assert len(report.issues) <= 2  # Minor issues only

    def test_build_report_poor(self, verifier, mock_opus_response_poor):
        """Test report generation for poor schematic."""
        report = verifier._build_report(
            mock_opus_response_poor,
            "/tmp/test.png"
        )

        assert report.passed is False
        assert report.overall_score < 50.0
        assert report.production_ready is False
        assert len(report.issues) == 8  # All criteria have issues

    def test_report_to_dict(self, verifier, mock_opus_response_excellent):
        """Test report serialization."""
        report = verifier._build_report(
            mock_opus_response_excellent,
            "/tmp/test.png"
        )
        report_dict = report.to_dict()

        assert isinstance(report_dict, dict)
        assert 'passed' in report_dict
        assert 'overall_score' in report_dict
        assert 'issues' in report_dict
        assert isinstance(report_dict['issues'], list)

    def test_report_str_formatting(self, verifier, mock_opus_response_excellent):
        """Test human-readable report formatting."""
        report = verifier._build_report(
            mock_opus_response_excellent,
            "/tmp/test.png"
        )
        report_str = str(report)

        assert "VISUAL QUALITY REPORT" in report_str
        assert "PASS" in report_str
        assert "Overall Score" in report_str
        assert "CRITERION SCORES" in report_str


class TestImageGeneration:
    """Test schematic image generation."""

    @pytest.mark.asyncio
    async def test_generate_image_missing_kicad(self, verifier, temp_output_dir):
        """Test error handling when KiCad CLI not found."""
        # Create dummy schematic file
        test_sch = temp_output_dir / "test.kicad_sch"
        test_sch.write_text("(kicad_sch)")

        with patch('subprocess.run', side_effect=FileNotFoundError):
            with pytest.raises(VerificationError, match="KiCad CLI not found"):
                await verifier._generate_image(test_sch, temp_output_dir)

    @pytest.mark.asyncio
    async def test_generate_image_kicad_error(self, verifier, temp_output_dir):
        """Test error handling when KiCad export fails."""
        test_sch = temp_output_dir / "test.kicad_sch"
        test_sch.write_text("(kicad_sch)")

        # Mock failed KiCad execution
        mock_result = Mock()
        mock_result.returncode = 1
        mock_result.stderr = "Error: Invalid schematic format"
        mock_result.stdout = ""

        with patch('subprocess.run', return_value=mock_result):
            with pytest.raises(VerificationError, match="KiCad export to SVG failed"):
                await verifier._generate_image(test_sch, temp_output_dir)

    @pytest.mark.asyncio
    async def test_generate_image_timeout(self, verifier, temp_output_dir):
        """Test timeout handling for KiCad export."""
        test_sch = temp_output_dir / "test.kicad_sch"
        test_sch.write_text("(kicad_sch)")

        import subprocess
        with patch('subprocess.run', side_effect=subprocess.TimeoutExpired('kicad-cli', 30)):
            with pytest.raises(VerificationError, match="timed out"):
                await verifier._generate_image(test_sch, temp_output_dir)


class TestOpusAnalysis:
    """Test Claude Opus 4.5 API integration."""

    @pytest.mark.asyncio
    async def test_analyze_with_opus_success(
        self,
        verifier,
        temp_output_dir,
        mock_opus_response_excellent
    ):
        """Test successful Opus 4.5 analysis."""
        # Create test image
        test_image = temp_output_dir / "test.png"
        test_image.write_bytes(b'\x89PNG\r\n\x1a\n')  # PNG header

        # Mock httpx response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            'choices': [{
                'message': {
                    'content': json.dumps(mock_opus_response_excellent)
                }
            }]
        }

        mock_client = AsyncMock()
        mock_client.__aenter__.return_value.post = AsyncMock(return_value=mock_response)

        with patch('httpx.AsyncClient', return_value=mock_client):
            analysis = await verifier._analyze_with_opus(test_image)

        assert 'criterion_scores' in analysis
        assert analysis['production_ready'] is True

    @pytest.mark.asyncio
    async def test_analyze_with_opus_api_error(self, verifier, temp_output_dir):
        """Test error handling for API failures."""
        test_image = temp_output_dir / "test.png"
        test_image.write_bytes(b'\x89PNG\r\n\x1a\n')

        # Mock failed API call
        mock_response = Mock()
        mock_response.status_code = 401
        mock_response.text = "Invalid API key"

        mock_client = AsyncMock()
        mock_client.__aenter__.return_value.post = AsyncMock(return_value=mock_response)

        with patch('httpx.AsyncClient', return_value=mock_client):
            with pytest.raises(VerificationError, match="OpenRouter API failed"):
                await verifier._analyze_with_opus(test_image)

    @pytest.mark.asyncio
    async def test_analyze_with_opus_timeout(self, verifier, temp_output_dir):
        """Test timeout handling for API calls."""
        test_image = temp_output_dir / "test.png"
        test_image.write_bytes(b'\x89PNG\r\n\x1a\n')

        import httpx
        mock_client = AsyncMock()
        mock_client.__aenter__.return_value.post = AsyncMock(
            side_effect=httpx.TimeoutException("Timeout")
        )

        with patch('httpx.AsyncClient', return_value=mock_client):
            with pytest.raises(VerificationError, match="timed out"):
                await verifier._analyze_with_opus(test_image)

    @pytest.mark.asyncio
    async def test_analyze_with_opus_invalid_json(self, verifier, temp_output_dir):
        """Test error handling for invalid JSON response."""
        test_image = temp_output_dir / "test.png"
        test_image.write_bytes(b'\x89PNG\r\n\x1a\n')

        # Mock response with invalid JSON
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            'choices': [{
                'message': {
                    'content': "This is not valid JSON"
                }
            }]
        }

        mock_client = AsyncMock()
        mock_client.__aenter__.return_value.post = AsyncMock(return_value=mock_response)

        with patch('httpx.AsyncClient', return_value=mock_client):
            with pytest.raises(VerificationError, match="Failed to parse JSON"):
                await verifier._analyze_with_opus(test_image)


class TestEndToEnd:
    """End-to-end integration tests."""

    @pytest.mark.asyncio
    async def test_verify_invalid_path(self, verifier):
        """Test verification with invalid schematic path."""
        with pytest.raises(VerificationError, match="not found"):
            await verifier.verify("/nonexistent/schematic.kicad_sch")

    @pytest.mark.asyncio
    async def test_verify_invalid_extension(self, verifier, temp_output_dir):
        """Test verification with wrong file type."""
        test_file = temp_output_dir / "test.txt"
        test_file.write_text("not a schematic")

        with pytest.raises(VerificationError, match="Invalid file type"):
            await verifier.verify(str(test_file))

    @pytest.mark.asyncio
    @patch('visual_verifier.VisualVerifier._generate_image')
    @patch('visual_verifier.VisualVerifier._analyze_with_opus')
    async def test_verify_full_flow_excellent(
        self,
        mock_analyze,
        mock_generate,
        verifier,
        temp_output_dir,
        mock_opus_response_excellent
    ):
        """Test full verification flow with excellent schematic."""
        # Setup
        test_sch = temp_output_dir / "excellent.kicad_sch"
        test_sch.write_text("(kicad_sch)")

        mock_generate.return_value = Path("/tmp/test.png")
        mock_analyze.return_value = mock_opus_response_excellent

        # Execute
        report = await verifier.verify(str(test_sch), str(temp_output_dir))

        # Verify
        assert report.passed is True
        assert report.overall_score >= 90.0
        assert report.production_ready is True
        assert mock_generate.called
        assert mock_analyze.called

    @pytest.mark.asyncio
    @patch('visual_verifier.VisualVerifier._generate_image')
    @patch('visual_verifier.VisualVerifier._analyze_with_opus')
    async def test_verify_full_flow_poor(
        self,
        mock_analyze,
        mock_generate,
        verifier,
        temp_output_dir,
        mock_opus_response_poor
    ):
        """Test full verification flow with poor schematic."""
        # Setup
        test_sch = temp_output_dir / "poor.kicad_sch"
        test_sch.write_text("(kicad_sch)")

        mock_generate.return_value = Path("/tmp/test.png")
        mock_analyze.return_value = mock_opus_response_poor

        # Execute
        report = await verifier.verify(str(test_sch), str(temp_output_dir))

        # Verify
        assert report.passed is False
        assert report.overall_score < 50.0
        assert report.production_ready is False
        assert len(report.issues) == 8


class TestConvenienceFunction:
    """Test convenience function."""

    @pytest.mark.asyncio
    @patch('visual_verifier.VisualVerifier.verify')
    async def test_verify_schematic_function(self, mock_verify, temp_output_dir):
        """Test verify_schematic convenience function."""
        test_sch = temp_output_dir / "test.kicad_sch"
        test_sch.write_text("(kicad_sch)")

        mock_report = Mock()
        mock_verify.return_value = mock_report

        result = await verify_schematic(str(test_sch))

        assert result == mock_report
        assert mock_verify.called


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
