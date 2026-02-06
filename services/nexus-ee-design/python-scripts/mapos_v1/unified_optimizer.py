#!/usr/bin/env python3
"""
Unified MAPOS Optimizer - Single entry point for all PCB optimization modes.

This module coordinates all MAPOS optimization components into a single,
coherent pipeline. It supports multiple optimization modes:

- BASE: Basic MAPOS (zone fill, net assignment, design rules, etc.)
- LLM: LLM-guided fixing via Claude Opus 4.6 through OpenRouter
- GAMING_AI: MAP-Elites + Red Queen + Ralph Wiggum evolutionary optimization
- HYBRID: Base -> LLM -> Gaming AI (most thorough, highest quality)

The hybrid mode is recommended for production use as it combines the
deterministic fixes of Base mode with the intelligent guidance of LLM
and the evolutionary exploration of Gaming AI.

Part of MAPOS (Multi-Agent PCB Optimization System) for the Nexus EE Design Partner plugin.
"""

import os
import sys
import time
import asyncio
import subprocess
import tempfile
import json
from enum import Enum
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class OptimizationMode(str, Enum):
    """Available optimization modes."""
    BASE = "base"           # Basic pcbnew fixes only
    LLM = "llm"             # LLM-guided optimization
    GAMING_AI = "gaming_ai" # MAP-Elites + Red Queen + Ralph Wiggum
    HYBRID = "hybrid"       # All three: Base -> LLM -> Gaming AI


@dataclass
class PhaseResult:
    """Result from a single optimization phase."""
    name: str
    initial_violations: int
    final_violations: int
    improvement: int
    success: bool
    duration_seconds: float
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class UnifiedResult:
    """Complete result from unified optimization."""
    mode: str
    initial_violations: int
    final_violations: int
    improvement: int
    improvement_pct: float
    success: bool
    phases: Dict[str, Any]
    duration_seconds: float


class UnifiedMAPOSOptimizer:
    """
    Unified optimizer coordinating all MAPOS components.

    This class provides a single entry point for PCB optimization,
    intelligently coordinating between base fixes, LLM guidance,
    and Gaming AI evolution based on the selected mode.
    """

    def __init__(
        self,
        pcb_path: str,
        mode: OptimizationMode = OptimizationMode.HYBRID,
        target_violations: int = 100,
        api_key: Optional[str] = None,
        output_dir: Optional[str] = None,
    ):
        """
        Initialize the unified optimizer.

        Args:
            pcb_path: Path to the .kicad_pcb file
            mode: Optimization mode (base, llm, gaming_ai, or hybrid)
            target_violations: Target DRC violation count
            api_key: OpenRouter API key for LLM operations
        """
        self.pcb_path = Path(pcb_path).resolve()
        if not self.pcb_path.exists():
            raise FileNotFoundError(f"PCB file not found: {self.pcb_path}")

        self.mode = mode
        self.target_violations = target_violations
        self.api_key = api_key or os.environ.get('OPENROUTER_API_KEY', '')
        self.output_dir = Path(output_dir) if output_dir else self.pcb_path.parent / 'mapos_output'
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Script directory for subprocess calls
        self.script_dir = Path(__file__).parent

    async def optimize(self) -> UnifiedResult:
        """
        Run optimization based on the selected mode.

        Returns:
            UnifiedResult with complete optimization statistics
        """
        start_time = time.time()
        phases: Dict[str, Any] = {}

        logger.info(f"Starting unified optimization (mode={self.mode.value})")
        logger.info(f"PCB: {self.pcb_path}")
        logger.info(f"Target violations: {self.target_violations}")

        # Get initial violation count
        initial = await self._get_violations()
        current = initial
        logger.info(f"Initial violations: {initial}")

        # Phase 1: Base MAPOS (always run for BASE or HYBRID)
        if self.mode in [OptimizationMode.BASE, OptimizationMode.HYBRID]:
            logger.info("\n=== Phase 1: Base MAPOS ===")
            phase_start = time.time()
            result = await self._run_base()
            phase_duration = time.time() - phase_start

            phases['base'] = {
                'initial_violations': current,
                'final_violations': result.get('final_violations', current),
                'improvement': current - result.get('final_violations', current),
                'duration_seconds': phase_duration,
                'details': result,
            }
            current = result.get('final_violations', current)
            logger.info(f"Base phase complete: {phases['base']['improvement']} violations fixed")

            # Early exit if target reached
            if current <= self.target_violations:
                logger.info(f"Target reached after base phase! ({current} <= {self.target_violations})")

        # Phase 2: LLM-guided (skip if target reached)
        if self.mode in [OptimizationMode.LLM, OptimizationMode.HYBRID]:
            if current > self.target_violations:
                logger.info("\n=== Phase 2: LLM-Guided Optimization ===")
                phase_start = time.time()
                result = await self._run_llm()
                phase_duration = time.time() - phase_start

                phases['llm'] = {
                    'initial_violations': current,
                    'final_violations': result.get('final_violations', current),
                    'improvement': current - result.get('final_violations', current),
                    'duration_seconds': phase_duration,
                    'details': result,
                }
                current = result.get('final_violations', current)
                logger.info(f"LLM phase complete: {phases['llm']['improvement']} violations fixed")

                if current <= self.target_violations:
                    logger.info(f"Target reached after LLM phase! ({current} <= {self.target_violations})")

        # Phase 3: Gaming AI (skip if target reached)
        if self.mode in [OptimizationMode.GAMING_AI, OptimizationMode.HYBRID]:
            if current > self.target_violations:
                logger.info("\n=== Phase 3: Gaming AI (MAP-Elites + Red Queen + Ralph Wiggum) ===")
                phase_start = time.time()
                result = await self._run_gaming_ai()
                phase_duration = time.time() - phase_start

                phases['gaming_ai'] = {
                    'initial_violations': current,
                    'final_violations': result.get('final_violations', current),
                    'improvement': current - result.get('final_violations', current),
                    'duration_seconds': phase_duration,
                    'status': result.get('status', 'UNKNOWN'),
                    'red_queen_rounds': result.get('red_queen_rounds', 0),
                    'champions_found': result.get('champions_found', 0),
                }
                current = result.get('final_violations', current)
                logger.info(f"Gaming AI phase complete: {phases['gaming_ai']['improvement']} violations fixed")

        # Calculate final stats
        duration = time.time() - start_time
        improvement = initial - current
        improvement_pct = (improvement / initial * 100) if initial > 0 else 0.0
        success = current <= self.target_violations

        logger.info(f"\n=== Optimization Complete ===")
        logger.info(f"Initial: {initial} -> Final: {current}")
        logger.info(f"Improvement: {improvement} ({improvement_pct:.1f}%)")
        logger.info(f"Target reached: {success}")
        logger.info(f"Duration: {duration:.1f}s")

        return UnifiedResult(
            mode=self.mode.value,
            initial_violations=initial,
            final_violations=current,
            improvement=improvement,
            improvement_pct=improvement_pct,
            success=success,
            phases=phases,
            duration_seconds=duration,
        )

    async def _get_violations(self) -> int:
        """Get current DRC violation count."""
        try:
            # Use kicad-cli for DRC
            with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
                drc_output = f.name

            result = subprocess.run(
                ['kicad-cli', 'pcb', 'drc', '--format', 'json', '--output', drc_output, str(self.pcb_path)],
                capture_output=True,
                text=True,
                timeout=120
            )

            if Path(drc_output).exists():
                with open(drc_output) as f:
                    drc_data = json.load(f)
                Path(drc_output).unlink()

                violations = len(drc_data.get('violations', []))
                unconnected = len(drc_data.get('unconnected_items', []))
                return violations + unconnected

            return 9999  # Error fallback

        except Exception as e:
            logger.warning(f"DRC error: {e}")
            return 9999

    async def _run_base(self) -> Dict[str, Any]:
        """Run base MAPOS optimization (zone fill, net assignment, etc.)."""
        try:
            result = subprocess.run(
                ['python3', str(self.script_dir / 'mapos_pcb_optimizer.py'),
                 str(self.pcb_path),
                 '--target', str(self.target_violations),
                 '--iterations', '3'],
                capture_output=True,
                text=True,
                timeout=300,
                cwd=str(self.script_dir)
            )

            # Parse output for final violations
            final_violations = await self._get_violations()

            return {
                'success': result.returncode == 0,
                'final_violations': final_violations,
                'output': result.stdout[-500:] if result.stdout else '',
                'error': result.stderr[-200:] if result.returncode != 0 else None,
            }

        except subprocess.TimeoutExpired:
            return {'success': False, 'error': 'Base optimization timed out', 'final_violations': await self._get_violations()}
        except Exception as e:
            return {'success': False, 'error': str(e), 'final_violations': await self._get_violations()}

    async def _run_llm(self) -> Dict[str, Any]:
        """Run LLM-guided optimization."""
        if not self.api_key:
            logger.warning("No API key for LLM optimization - skipping")
            return {'success': False, 'error': 'No OpenRouter API key', 'final_violations': await self._get_violations()}

        try:
            env = {**os.environ, 'OPENROUTER_API_KEY': self.api_key}

            result = subprocess.run(
                ['python3', str(self.script_dir / 'llm_pcb_fixer.py'),
                 str(self.pcb_path),
                 '--max-iterations', '3'],
                capture_output=True,
                text=True,
                timeout=600,
                env=env,
                cwd=str(self.script_dir)
            )

            final_violations = await self._get_violations()

            return {
                'success': result.returncode == 0,
                'final_violations': final_violations,
                'output': result.stdout[-500:] if result.stdout else '',
                'error': result.stderr[-200:] if result.returncode != 0 else None,
            }

        except subprocess.TimeoutExpired:
            return {'success': False, 'error': 'LLM optimization timed out', 'final_violations': await self._get_violations()}
        except Exception as e:
            return {'success': False, 'error': str(e), 'final_violations': await self._get_violations()}

    async def _run_gaming_ai(self) -> Dict[str, Any]:
        """Run Gaming AI optimization (MAP-Elites + Red Queen + Ralph Wiggum)."""
        if not self.api_key:
            logger.warning("No API key for Gaming AI - using without LLM guidance")

        try:
            # Import Gaming AI directly for better control
            sys.path.insert(0, str(self.script_dir))
            from gaming_ai.integration import MAPOSRQOptimizer, MAPOSRQConfig

            config = MAPOSRQConfig(
                target_violations=self.target_violations,
                rq_rounds=10,
                max_stagnation=15,
                use_llm=bool(self.api_key),
                use_neural_networks=False,
                openrouter_api_key=self.api_key,
            )

            optimizer = MAPOSRQOptimizer(str(self.pcb_path), config=config)
            result = await optimizer.optimize()

            return {
                'success': result.status.name in ['SUCCESS', 'PARTIAL'],
                'status': result.status.name,
                'final_violations': result.final_violations,
                'red_queen_rounds': result.red_queen_rounds,
                'champions_found': len(result.champions) if result.champions else 0,
            }

        except ImportError as e:
            logger.warning(f"Gaming AI not available: {e}")
            return {'success': False, 'error': f'Gaming AI not available: {e}', 'final_violations': await self._get_violations()}
        except Exception as e:
            logger.error(f"Gaming AI error: {e}")
            return {'success': False, 'error': str(e), 'final_violations': await self._get_violations()}


async def optimize_pcb(
    pcb_path: str,
    mode: str = "hybrid",
    target_violations: int = 100,
    api_key: Optional[str] = None,
) -> UnifiedResult:
    """
    Convenience function for PCB optimization.

    Args:
        pcb_path: Path to .kicad_pcb file
        mode: Optimization mode (base, llm, gaming_ai, hybrid)
        target_violations: Target DRC violations
        api_key: OpenRouter API key

    Returns:
        UnifiedResult with optimization statistics
    """
    mode_map = {
        'base': OptimizationMode.BASE,
        'llm': OptimizationMode.LLM,
        'gaming_ai': OptimizationMode.GAMING_AI,
        'hybrid': OptimizationMode.HYBRID,
    }
    opt_mode = mode_map.get(mode.lower(), OptimizationMode.HYBRID)

    optimizer = UnifiedMAPOSOptimizer(
        pcb_path,
        mode=opt_mode,
        target_violations=target_violations,
        api_key=api_key,
    )

    return await optimizer.optimize()


def main():
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description='Unified MAPOS PCB Optimizer',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Optimization Modes:
  base       Basic MAPOS (zone fill, net assignment, design rules)
  llm        LLM-guided optimization via Claude Opus 4.6
  gaming_ai  MAP-Elites + Red Queen + Ralph Wiggum evolution
  hybrid     All three phases (most thorough, recommended)

Examples:
  %(prog)s board.kicad_pcb --mode hybrid --target 50
  %(prog)s board.kicad_pcb --mode llm --target 100
  %(prog)s board.kicad_pcb --mode gaming_ai --target 0

Part of MAPOS (Multi-Agent PCB Optimization System)
        '''
    )

    parser.add_argument('pcb_path', help='Path to KiCad PCB file')
    parser.add_argument('--mode', choices=['base', 'llm', 'gaming_ai', 'hybrid'],
                        default='hybrid', help='Optimization mode (default: hybrid)')
    parser.add_argument('--target', type=int, default=100,
                        help='Target DRC violations (default: 100)')
    parser.add_argument('--api-key', help='OpenRouter API key (or set OPENROUTER_API_KEY env)')
    parser.add_argument('--json', action='store_true', help='Output result as JSON')

    args = parser.parse_args()

    # Run async optimization
    result = asyncio.run(optimize_pcb(
        args.pcb_path,
        mode=args.mode,
        target_violations=args.target,
        api_key=args.api_key,
    ))

    if args.json:
        output = {
            'mode': result.mode,
            'initial_violations': result.initial_violations,
            'final_violations': result.final_violations,
            'improvement': result.improvement,
            'improvement_pct': result.improvement_pct,
            'success': result.success,
            'phases': result.phases,
            'duration_seconds': result.duration_seconds,
        }
        print(json.dumps(output, indent=2, default=str))
    else:
        print(f"\n=== Unified MAPOS Optimization Result ===")
        print(f"Mode:         {result.mode}")
        print(f"Initial:      {result.initial_violations} violations")
        print(f"Final:        {result.final_violations} violations")
        print(f"Improvement:  {result.improvement} ({result.improvement_pct:.1f}%)")
        print(f"Target met:   {'Yes' if result.success else 'No'}")
        print(f"Duration:     {result.duration_seconds:.1f}s")
        print(f"\nPhases completed: {', '.join(result.phases.keys())}")

    return 0 if result.success else 1


if __name__ == '__main__':
    sys.exit(main())
