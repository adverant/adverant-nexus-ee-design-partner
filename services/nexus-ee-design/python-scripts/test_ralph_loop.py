"""
Test Suite for Ralph Loop Orchestrator

Tests the iterative improvement system with various scenarios.

Author: Nexus EE Design Team
"""

import asyncio
import json
import logging
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Dict, Any, List
from unittest.mock import AsyncMock, MagicMock, patch

# Add parent paths for imports
sys.path.insert(0, str(Path(__file__).parent))

from ralph_loop_orchestrator import (
    RalphLoopOrchestrator,
    RalphLoopResult,
    IterationResult,
)

from agents.smoke_test.smoke_test_agent import SmokeTestResult, SmokeTestIssue, SmokeTestSeverity
from agents.visual_validator.dual_llm_validator import ComparisonResult, VisualIssue, VisualIssueCategory, IssueSeverity

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


class TestRalphLoopOrchestrator(unittest.IsolatedAsyncioTestCase):
    """Test Ralph loop orchestrator functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.test_bom = [
            {
                "part_number": "STM32G431CBT6",
                "category": "MCU",
                "manufacturer": "ST",
                "value": "Cortex-M4",
            },
            {
                "part_number": "VCC",
                "category": "Power",
                "value": "5V",
            },
            {
                "part_number": "GND",
                "category": "Power",
                "value": "GND",
            },
        ]

        self.test_design_intent = "Simple MCU circuit with power and decoupling"

    async def test_initialization(self):
        """Test orchestrator initialization."""
        orchestrator = RalphLoopOrchestrator(
            max_iterations=100,
            target_score=95.0,
            plateau_threshold=10,
        )

        self.assertEqual(orchestrator.max_iterations, 100)
        self.assertEqual(orchestrator.target_score, 95.0)
        self.assertEqual(orchestrator.plateau_threshold, 10)

    async def test_convergence_scenario(self):
        """Test scenario where design converges to target."""
        orchestrator = RalphLoopOrchestrator(
            max_iterations=10,
            target_score=95.0,
            plateau_threshold=5,
        )

        # Mock MAPO optimizer to return improving schematics
        mock_optimizer = AsyncMock()

        iteration_count = [0]  # Use list to allow mutation in closure

        async def mock_optimize(*args, **kwargs):
            iteration_count[0] += 1
            iter_num = iteration_count[0]

            # Create temp schematic file
            temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.kicad_sch', delete=False)
            temp_file.write(f"(kicad_sch (version 20211123) ; iteration {iter_num})")
            temp_file.close()

            from mapos_v2_1_schematic.orchestrator.schematic_mapo_optimizer import OptimizationResult
            return OptimizationResult(
                success=True,
                schematic_path=Path(temp_file.name),
                schematic_content=f"(kicad_sch ; iteration {iter_num})",
            )

        mock_optimizer.optimize = mock_optimize
        orchestrator._mapo_optimizer = mock_optimizer

        # Mock smoke test to improve each iteration
        mock_smoke_tester = AsyncMock()

        async def mock_smoke_test(*args, **kwargs):
            iter_num = iteration_count[0]
            # Score improves: 70% -> 80% -> 90% -> 100%
            score = min(100.0, 60.0 + (iter_num * 15.0))
            passed = score >= 95.0

            return SmokeTestResult(
                passed=passed,
                power_rails_ok=True,
                ground_ok=True,
                no_shorts=True,
                no_floating_nodes=score >= 80.0,
                power_dissipation_ok=True,
                current_paths_valid=score >= 90.0,
                issues=[] if passed else [
                    SmokeTestIssue(
                        severity=SmokeTestSeverity.WARNING,
                        test_name="connectivity",
                        message=f"Minor issue (iter {iter_num})",
                    )
                ]
            )

        mock_smoke_tester.run_smoke_test = mock_smoke_test
        orchestrator._smoke_tester = mock_smoke_tester

        # Mock visual validator to also improve
        mock_visual_verifier = AsyncMock()

        async def mock_visual_validate(*args, **kwargs):
            iter_num = iteration_count[0]
            # Score improves: 0.75 -> 0.85 -> 0.95 -> 1.0
            score = min(1.0, 0.65 + (iter_num * 0.12))
            passed = score >= 0.95

            return ComparisonResult(
                agreement_score=0.9,
                combined_score=score,
                agreed_issues=[] if passed else [
                    VisualIssue(
                        category=VisualIssueCategory.READABILITY,
                        severity=IssueSeverity.WARNING,
                        description=f"Minor visual issue (iter {iter_num})",
                    )
                ]
            )

        mock_visual_verifier.validate_schematic = mock_visual_validate
        orchestrator._visual_verifier = mock_visual_verifier

        # Mock LLM feedback
        with patch.object(orchestrator, '_generate_feedback', return_value="- Improve connections"):
            # Run loop
            result = await orchestrator.run(
                bom=self.test_bom,
                design_intent=self.test_design_intent,
                project_name="test_convergence",
            )

        # Verify convergence
        self.assertTrue(result.success, "Should converge successfully")
        self.assertTrue(result.converged, "Should mark as converged")
        self.assertIsNotNone(result.convergence_iteration, "Should have convergence iteration")
        self.assertGreaterEqual(result.final_score, 95.0, "Should reach target score")
        self.assertLessEqual(result.total_iterations, 5, "Should converge quickly")

        # Clean up temp files
        for iter_result in result.iterations:
            try:
                Path(iter_result.schematic_path).unlink()
            except:
                pass

        await orchestrator.close()

    async def test_plateau_detection(self):
        """Test scenario where design plateaus without reaching target."""
        orchestrator = RalphLoopOrchestrator(
            max_iterations=30,
            target_score=100.0,
            plateau_threshold=5,
        )

        # Mock MAPO optimizer
        mock_optimizer = AsyncMock()

        iteration_count = [0]

        async def mock_optimize(*args, **kwargs):
            iteration_count[0] += 1
            iter_num = iteration_count[0]

            temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.kicad_sch', delete=False)
            temp_file.write(f"(kicad_sch ; iteration {iter_num})")
            temp_file.close()

            from mapos_v2_1_schematic.orchestrator.schematic_mapo_optimizer import OptimizationResult
            return OptimizationResult(
                success=True,
                schematic_path=Path(temp_file.name),
            )

        mock_optimizer.optimize = mock_optimize
        orchestrator._mapo_optimizer = mock_optimizer

        # Mock smoke test - plateaus at 85%
        mock_smoke_tester = AsyncMock()

        async def mock_smoke_test(*args, **kwargs):
            # Plateaus at 85%
            return SmokeTestResult(
                passed=False,
                power_rails_ok=True,
                ground_ok=True,
                no_shorts=True,
                no_floating_nodes=True,
                power_dissipation_ok=False,  # Stuck issue
                current_paths_valid=True,
                issues=[
                    SmokeTestIssue(
                        severity=SmokeTestSeverity.ERROR,
                        test_name="power_dissipation",
                        message="Power dissipation exceeds limits",
                    )
                ]
            )

        mock_smoke_tester.run_smoke_test = mock_smoke_test
        orchestrator._smoke_tester = mock_smoke_tester

        # Mock visual validator - plateaus at 0.85
        mock_visual_verifier = AsyncMock()

        async def mock_visual_validate(*args, **kwargs):
            return ComparisonResult(
                agreement_score=0.85,
                combined_score=0.85,
                agreed_issues=[
                    VisualIssue(
                        category=VisualIssueCategory.WIRE_ROUTING,
                        severity=IssueSeverity.WARNING,
                        description="Persistent routing issue",
                    )
                ]
            )

        mock_visual_verifier.validate_schematic = mock_visual_validate
        orchestrator._visual_verifier = mock_visual_verifier

        # Mock feedback
        with patch.object(orchestrator, '_generate_feedback', return_value="- Fix power dissipation"):
            # Run loop
            result = await orchestrator.run(
                bom=self.test_bom,
                design_intent=self.test_design_intent,
                project_name="test_plateau",
            )

        # Verify plateau detection
        self.assertTrue(result.plateau_detected, "Should detect plateau")
        self.assertIsNotNone(result.plateau_iteration, "Should record plateau iteration")
        self.assertLess(result.total_iterations, 15, "Should stop early due to plateau")
        self.assertGreater(result.final_score, 80.0, "Should have reasonable final score")

        # Clean up
        for iter_result in result.iterations:
            try:
                Path(iter_result.schematic_path).unlink()
            except:
                pass

        await orchestrator.close()

    async def test_max_iterations_reached(self):
        """Test scenario where max iterations is reached."""
        orchestrator = RalphLoopOrchestrator(
            max_iterations=3,  # Very low for testing
            target_score=100.0,
            plateau_threshold=10,
        )

        # Mock components
        mock_optimizer = AsyncMock()

        iteration_count = [0]

        async def mock_optimize(*args, **kwargs):
            iteration_count[0] += 1
            temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.kicad_sch', delete=False)
            temp_file.write("(kicad_sch)")
            temp_file.close()

            from mapos_v2_1_schematic.orchestrator.schematic_mapo_optimizer import OptimizationResult
            return OptimizationResult(
                success=True,
                schematic_path=Path(temp_file.name),
            )

        mock_optimizer.optimize = mock_optimize
        orchestrator._mapo_optimizer = mock_optimizer

        mock_smoke_tester = AsyncMock()
        mock_smoke_tester.run_smoke_test = AsyncMock(return_value=SmokeTestResult(
            passed=False,
            power_rails_ok=True,
            ground_ok=True,
            no_shorts=True,
            no_floating_nodes=False,
            power_dissipation_ok=True,
            current_paths_valid=True,
            issues=[SmokeTestIssue(
                severity=SmokeTestSeverity.ERROR,
                test_name="connectivity",
                message="Floating nodes detected",
            )]
        ))
        orchestrator._smoke_tester = mock_smoke_tester

        mock_visual_verifier = AsyncMock()
        mock_visual_verifier.validate_schematic = AsyncMock(return_value=ComparisonResult(
            agreement_score=0.7,
            combined_score=0.7,
        ))
        orchestrator._visual_verifier = mock_visual_verifier

        with patch.object(orchestrator, '_generate_feedback', return_value="- Fix floating nodes"):
            result = await orchestrator.run(
                bom=self.test_bom,
                design_intent=self.test_design_intent,
                project_name="test_max_iter",
            )

        # Verify max iterations behavior
        self.assertEqual(result.total_iterations, 3, "Should run exactly max iterations")
        self.assertFalse(result.converged, "Should not be marked as converged")
        self.assertIsNone(result.convergence_iteration, "Should have no convergence iteration")

        # Clean up
        for iter_result in result.iterations:
            try:
                Path(iter_result.schematic_path).unlink()
            except:
                pass

        await orchestrator.close()

    async def test_power_source_detection(self):
        """Test power source detection from BOM."""
        orchestrator = RalphLoopOrchestrator()

        bom = [
            {"part_number": "VCC", "value": "5V"},
            {"part_number": "VDD", "value": "3.3V"},
            {"part_number": "GND", "value": "GND"},
            {"part_number": "STM32F4", "value": "MCU"},
        ]

        power_sources = orchestrator._detect_power_sources(bom)

        # Should detect VCC, VDD, and GND
        self.assertGreaterEqual(len(power_sources), 2, "Should detect at least 2 power sources")

        net_names = {ps["net"] for ps in power_sources}
        self.assertIn("GND", net_names, "Should always include GND")

        # Check voltage inference
        vcc_source = next((ps for ps in power_sources if ps["net"] == "VCC"), None)
        if vcc_source:
            self.assertEqual(vcc_source["voltage"], 5.0, "Should infer 5V for VCC")

    async def test_smoke_score_calculation(self):
        """Test smoke test score calculation."""
        orchestrator = RalphLoopOrchestrator()

        # Perfect result
        perfect_result = SmokeTestResult(
            passed=True,
            power_rails_ok=True,
            ground_ok=True,
            no_shorts=True,
            no_floating_nodes=True,
            power_dissipation_ok=True,
            current_paths_valid=True,
            issues=[],
        )
        score = orchestrator._calculate_smoke_score(perfect_result)
        self.assertEqual(score, 100.0, "Perfect result should score 100")

        # Result with fatal issue
        fatal_result = SmokeTestResult(
            passed=False,
            power_rails_ok=False,
            ground_ok=True,
            no_shorts=True,
            no_floating_nodes=True,
            power_dissipation_ok=True,
            current_paths_valid=True,
            issues=[
                SmokeTestIssue(
                    severity=SmokeTestSeverity.FATAL,
                    test_name="power",
                    message="Power rail disconnected",
                )
            ],
        )
        score = orchestrator._calculate_smoke_score(fatal_result)
        self.assertLess(score, 80.0, "Fatal issue should significantly reduce score")

    async def test_heuristic_feedback_generation(self):
        """Test heuristic feedback generation."""
        orchestrator = RalphLoopOrchestrator()

        violations = [
            {"severity": "error", "message": "Power rail VCC not connected"},
            {"severity": "warning", "message": "Ground plane discontinuity"},
        ]

        issues = [
            {"category": "wire_routing", "description": "Excessive wire crossings"},
            {"category": "component_placement", "description": "Poor signal flow"},
        ]

        feedback = orchestrator._generate_heuristic_feedback(violations, issues)

        self.assertIsInstance(feedback, str, "Should return string")
        self.assertGreater(len(feedback), 0, "Should generate non-empty feedback")
        self.assertIn("-", feedback, "Should contain bullet points")

    async def test_iteration_result_serialization(self):
        """Test IterationResult serialization."""
        iter_result = IterationResult(
            iteration=1,
            schematic_path="/tmp/test.kicad_sch",
            smoke_test_passed=True,
            smoke_test_score=95.0,
            visual_test_passed=False,
            visual_test_score=85.0,
            overall_score=91.0,
            duration_seconds=12.5,
        )

        result_dict = iter_result.to_dict()

        self.assertIsInstance(result_dict, dict, "Should serialize to dict")
        self.assertEqual(result_dict["iteration"], 1)
        self.assertEqual(result_dict["overall_score"], 91.0)
        self.assertIn("smoke_test", result_dict)
        self.assertIn("visual_test", result_dict)

    async def test_ralph_loop_result_serialization(self):
        """Test RalphLoopResult serialization."""
        result = RalphLoopResult(
            success=True,
            final_schematic_path="/tmp/final.kicad_sch",
            final_score=98.5,
            total_iterations=5,
            converged=True,
            convergence_iteration=4,
            iterations=[],
            total_duration_seconds=45.2,
            improvement_graph={1: 70.0, 2: 85.0, 3: 95.0, 4: 98.5},
            failure_analysis=["Initial power issues resolved"],
        )

        result_dict = result.to_dict()

        self.assertIsInstance(result_dict, dict, "Should serialize to dict")
        self.assertTrue(result_dict["success"])
        self.assertEqual(result_dict["total_iterations"], 5)
        self.assertIn("improvement_graph", result_dict)


class TestRalphLoopIntegration(unittest.IsolatedAsyncioTestCase):
    """Integration tests with real components (if available)."""

    async def test_real_bom_validation(self):
        """Test with a real BOM structure."""
        # This test would run with actual MAPO components
        # Skip if components not available
        try:
            from mapos_v2_1_schematic.orchestrator.schematic_mapo_optimizer import SchematicMAPOOptimizer
        except ImportError:
            self.skipTest("MAPO components not available")

        # Would run a short real loop here
        # For now, just verify imports work
        self.assertTrue(True)


def run_tests():
    """Run all tests."""
    # Run unittest
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromModule(sys.modules[__name__])
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # Exit with appropriate code
    sys.exit(0 if result.wasSuccessful() else 1)


if __name__ == "__main__":
    run_tests()
