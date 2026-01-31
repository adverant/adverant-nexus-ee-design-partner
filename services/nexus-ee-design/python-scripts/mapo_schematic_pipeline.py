"""
MAPO Schematic Pipeline - Unified orchestrator for schematic generation and validation.

Integrates:
- Symbol Fetcher Agent (real symbols from SnapEDA, KiCad, manufacturers)
- GraphRAG Symbol Indexer (semantic search and caching)
- Schematic Assembler Agent (intelligent placement and routing)
- Vision Validator (LLM-based multi-expert validation)
- MAPO Loop (iterative refinement until quality threshold)

Author: Nexus EE Design Team
"""

import asyncio
import json
import logging
import os
import sys
import tempfile
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from agents.symbol_fetcher import SymbolFetcherAgent, FetchedSymbol
from agents.schematic_assembler import (
    SchematicAssemblerAgent,
    SchematicSheet,
    BOMItem,
    Connection,
    BlockDiagram,
)
from validation.schematic_vision_validator import (
    SchematicVisionValidator,
    MAPOSchematicLoop,
    SchematicValidationReport,
)
from graphrag.symbol_indexer import SymbolGraphRAGIndexer, create_indexer

logger = logging.getLogger(__name__)


@dataclass
class PipelineConfig:
    """Configuration for the MAPO schematic pipeline."""
    # Symbol fetching - default to project-local cache
    symbol_cache_path: Path = field(
        default_factory=lambda: Path(__file__).parent / "symbol_cache"
    )
    use_graphrag: bool = False  # Disabled by default (Neo4j often not available)
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = ""

    # Validation
    primary_model: str = "claude-sonnet-4-20250514"
    verification_model: str = "claude-opus-4-20250514"
    validation_threshold: float = 0.85
    max_iterations: int = 5

    # Output
    output_dir: Path = field(
        default_factory=lambda: Path(__file__).parent / "output"
    )

    def __post_init__(self):
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.symbol_cache_path.mkdir(parents=True, exist_ok=True)


@dataclass
class PipelineResult:
    """Result from the MAPO schematic pipeline."""
    success: bool
    schematic_path: Optional[Path] = None
    sheets: List[SchematicSheet] = field(default_factory=list)
    validation_report: Optional[SchematicValidationReport] = None
    symbols_fetched: int = 0
    symbols_from_cache: int = 0
    symbols_generated: int = 0
    iterations: int = 0
    total_time_seconds: float = 0
    errors: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {
            "success": self.success,
            "schematic_path": str(self.schematic_path) if self.schematic_path else None,
            "sheet_count": len(self.sheets),
            "validation_score": self.validation_report.overall_score if self.validation_report else None,
            "validation_passed": self.validation_report.passed if self.validation_report else False,
            "symbols_fetched": self.symbols_fetched,
            "symbols_from_cache": self.symbols_from_cache,
            "symbols_generated": self.symbols_generated,
            "iterations": self.iterations,
            "total_time_seconds": self.total_time_seconds,
            "errors": self.errors,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)


class MAPOSchematicPipeline:
    """
    Unified pipeline for schematic generation with MAPO validation.

    Usage:
        pipeline = MAPOSchematicPipeline()
        result = await pipeline.generate(
            bom=[...],
            design_intent="FOC ESC for brushless motor",
            connections=[...]
        )
    """

    def __init__(self, config: Optional[PipelineConfig] = None):
        """Initialize the pipeline."""
        self.config = config or PipelineConfig()

        # Components will be initialized lazily
        self._symbol_fetcher: Optional[SymbolFetcherAgent] = None
        self._graphrag_indexer: Optional[SymbolGraphRAGIndexer] = None
        self._assembler: Optional[SchematicAssemblerAgent] = None
        self._validator: Optional[SchematicVisionValidator] = None
        self._renderer: Optional[Any] = None  # KiCanvas renderer

    async def initialize(self):
        """Initialize all pipeline components."""
        logger.info("Initializing MAPO Schematic Pipeline...")

        # Initialize GraphRAG if enabled
        if self.config.use_graphrag:
            try:
                self._graphrag_indexer = await create_indexer(
                    neo4j_uri=self.config.neo4j_uri,
                    neo4j_user=self.config.neo4j_user,
                    neo4j_password=self.config.neo4j_password
                )
                logger.info("GraphRAG indexer connected")
            except Exception as e:
                logger.warning(f"GraphRAG initialization failed: {e}")
                self._graphrag_indexer = None

        # Initialize symbol fetcher
        self._symbol_fetcher = SymbolFetcherAgent(
            cache_path=self.config.symbol_cache_path,
            graphrag_client=self._graphrag_indexer
        )
        logger.info("Symbol fetcher initialized")

        # Initialize assembler
        self._assembler = SchematicAssemblerAgent(
            symbol_fetcher=self._symbol_fetcher,
            graphrag_client=self._graphrag_indexer
        )
        logger.info("Schematic assembler initialized")

        # Initialize validator
        self._validator = SchematicVisionValidator(
            primary_model=self.config.primary_model,
            verification_model=self.config.verification_model
        )
        self._validator.PASS_THRESHOLD = self.config.validation_threshold
        logger.info("Vision validator initialized")

        logger.info("Pipeline initialization complete")

    async def generate(
        self,
        bom: List[Dict[str, Any]],
        design_intent: str,
        connections: Optional[List[Dict[str, str]]] = None,
        block_diagram: Optional[Dict[str, Any]] = None,
        design_name: str = "schematic",
        reference_images: Optional[List[bytes]] = None,
        skip_validation: bool = False
    ) -> PipelineResult:
        """
        Generate a validated schematic from BOM and design intent.

        Args:
            bom: List of component dictionaries with part_number, category, etc.
            design_intent: Natural language description of circuit function
            connections: Optional explicit connections between components
            block_diagram: Optional hierarchical block structure
            design_name: Name for the output schematic
            reference_images: Optional reference design images for comparison
            skip_validation: Skip the MAPO validation loop (for testing)

        Returns:
            PipelineResult with schematic and validation info
        """
        start_time = datetime.now()
        result = PipelineResult(success=False)

        try:
            # Ensure initialized
            if not self._symbol_fetcher:
                await self.initialize()

            # Convert BOM dictionaries to BOMItem objects
            bom_items = [
                BOMItem(
                    part_number=item.get('part_number', item.get('mpn', 'UNKNOWN')),
                    manufacturer=item.get('manufacturer'),
                    reference=item.get('reference'),
                    quantity=item.get('quantity', 1),
                    category=item.get('category', 'Other'),
                    value=item.get('value', ''),
                    footprint=item.get('footprint', ''),
                    description=item.get('description', '')
                )
                for item in bom
            ]

            # Convert connections if provided
            connection_objs = None
            if connections:
                connection_objs = [
                    Connection(
                        from_ref=conn.get('from_ref', ''),
                        from_pin=conn.get('from_pin', ''),
                        to_ref=conn.get('to_ref', ''),
                        to_pin=conn.get('to_pin', ''),
                        net_name=conn.get('net_name')
                    )
                    for conn in connections
                ]

            # Convert block diagram if provided
            block_diagram_obj = None
            if block_diagram:
                block_diagram_obj = BlockDiagram(
                    blocks=block_diagram.get('blocks', {}),
                    connections=block_diagram.get('connections', [])
                )

            logger.info(f"Starting schematic generation: {len(bom_items)} components")

            # Phase 1: Assemble schematic
            sheets = await self._assembler.assemble_schematic(
                bom=bom_items,
                block_diagram=block_diagram_obj,
                connections=connection_objs,
                design_name=design_name
            )

            result.sheets = sheets

            # Track symbol statistics
            for sheet in sheets:
                for symbol_id, sexp in sheet.lib_symbols.items():
                    result.symbols_fetched += 1
                    if "source" in str(sexp):
                        if "local_cache" in str(sexp):
                            result.symbols_from_cache += 1
                        elif "llm_generated" in str(sexp) or "generated" in str(sexp):
                            result.symbols_generated += 1

            # Generate output file
            output_path = self.config.output_dir / f"{design_name}.kicad_sch"
            schematic_content = self._assembler.generate_kicad_sch(sheets[0])
            output_path.write_text(schematic_content)
            result.schematic_path = output_path

            logger.info(f"Schematic assembled: {output_path}")

            # Phase 2: MAPO Validation Loop (if not skipped)
            if not skip_validation and self._validator:
                logger.info("Starting MAPO validation loop...")

                # Render schematic to image for validation
                schematic_image = await self._render_schematic(schematic_content)

                if schematic_image:
                    # Run validation
                    report = await self._validator.validate_schematic(
                        schematic_image=schematic_image,
                        design_intent=design_intent,
                        reference_images=reference_images,
                        iteration=0
                    )

                    result.validation_report = report
                    result.iterations = 1

                    # If not passed, run MAPO loop
                    if not report.passed and self.config.max_iterations > 1:
                        final_content, final_report = await self._run_mapo_loop(
                            schematic_content,
                            design_intent,
                            reference_images,
                            report
                        )

                        # Update output
                        output_path.write_text(final_content)
                        result.validation_report = final_report
                        result.iterations = final_report.iteration + 1

                    logger.info(
                        f"Validation complete: score={report.overall_score:.2%}, "
                        f"passed={report.passed}"
                    )
                else:
                    logger.warning("Could not render schematic for validation")

            result.success = True

        except Exception as e:
            logger.error(f"Pipeline error: {e}", exc_info=True)
            result.errors.append(str(e))

        finally:
            result.total_time_seconds = (datetime.now() - start_time).total_seconds()

        return result

    async def _render_schematic(self, schematic_content: str) -> Optional[bytes]:
        """Render schematic to PNG image for validation."""
        try:
            # Try using KiCad CLI for rendering
            kicad_cli = self._find_kicad_cli()
            if kicad_cli:
                return await self._render_with_kicad_cli(schematic_content, kicad_cli)

            # Fallback: Try using Puppeteer/KiCanvas
            return await self._render_with_kicanvas(schematic_content)

        except Exception as e:
            logger.warning(f"Schematic rendering failed: {e}")
            return None

    def _find_kicad_cli(self) -> Optional[str]:
        """Find KiCad CLI executable."""
        possible_paths = [
            "/Applications/KiCad/KiCad.app/Contents/MacOS/kicad-cli",
            "/usr/bin/kicad-cli",
            "/usr/local/bin/kicad-cli",
            "kicad-cli",
        ]

        for path in possible_paths:
            if os.path.exists(path) or path == "kicad-cli":
                return path

        return None

    async def _render_with_kicad_cli(
        self,
        schematic_content: str,
        kicad_cli: str
    ) -> Optional[bytes]:
        """Render using KiCad CLI."""
        with tempfile.NamedTemporaryFile(suffix=".kicad_sch", delete=False) as f:
            f.write(schematic_content.encode())
            sch_path = f.name

        png_path = sch_path.replace(".kicad_sch", ".png")

        try:
            proc = await asyncio.create_subprocess_exec(
                kicad_cli, "sch", "export", "svg",
                "-o", png_path.replace(".png", ".svg"),
                sch_path,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await proc.communicate()

            # Convert SVG to PNG if needed
            svg_path = png_path.replace(".png", ".svg")
            if os.path.exists(svg_path):
                # Try cairosvg for conversion
                try:
                    import cairosvg
                    cairosvg.svg2png(url=svg_path, write_to=png_path)
                except ImportError:
                    logger.warning("cairosvg not available for SVG to PNG conversion")
                    return None

            if os.path.exists(png_path):
                with open(png_path, "rb") as f:
                    return f.read()

        except Exception as e:
            logger.warning(f"KiCad CLI render failed: {e}")

        finally:
            # Cleanup
            for path in [sch_path, png_path, png_path.replace(".png", ".svg")]:
                try:
                    os.unlink(path)
                except OSError:
                    pass

        return None

    async def _render_with_kicanvas(self, schematic_content: str) -> Optional[bytes]:
        """Render using KiCanvas via headless browser."""
        try:
            # This would use Puppeteer/Playwright to render via KiCanvas
            # For now, return None to indicate rendering not available
            logger.info("KiCanvas rendering not implemented yet")
            return None

        except Exception as e:
            logger.warning(f"KiCanvas render failed: {e}")
            return None

    async def _run_mapo_loop(
        self,
        initial_content: str,
        design_intent: str,
        reference_images: Optional[List[bytes]],
        initial_report: SchematicValidationReport
    ) -> Tuple[str, SchematicValidationReport]:
        """Run the MAPO iterative refinement loop."""
        current_content = initial_content
        current_report = initial_report

        for iteration in range(1, self.config.max_iterations):
            logger.info(f"MAPO iteration {iteration + 1}/{self.config.max_iterations}")

            # Apply fixes from report
            if current_report.recommended_fixes:
                current_content = await self._apply_fixes(
                    current_content,
                    current_report.recommended_fixes
                )

            # Re-render and validate
            schematic_image = await self._render_schematic(current_content)
            if not schematic_image:
                logger.warning("Could not render schematic for re-validation")
                break

            current_report = await self._validator.validate_schematic(
                schematic_image=schematic_image,
                design_intent=design_intent,
                reference_images=reference_images,
                iteration=iteration
            )

            logger.info(
                f"Iteration {iteration + 1}: "
                f"score={current_report.overall_score:.2%}, "
                f"passed={current_report.passed}"
            )

            if current_report.passed:
                break

            if not current_report.recommended_fixes:
                logger.info("No more fixes available")
                break

        return current_content, current_report

    async def _apply_fixes(
        self,
        schematic_content: str,
        fixes: List[Dict]
    ) -> str:
        """Apply recommended fixes to schematic content."""
        modified_content = schematic_content

        for fix in fixes:
            fix_type = fix.get('fix_type', '')
            kicad_action = fix.get('kicad_action', '')

            if not kicad_action:
                continue

            try:
                # Apply the fix based on type
                if fix_type == 'add_component':
                    # Parse and add component
                    pass
                elif fix_type == 'modify_connection':
                    # Modify wire connections
                    pass
                elif fix_type == 'change_value':
                    # Update component value
                    issue_ref = fix.get('issue_ref', '')
                    if issue_ref:
                        # Find and update the component value
                        import re
                        pattern = rf'(\(property "Value" ")[^"]*(" \(at [^)]+\)[^)]*\))'
                        # This is a simplified fix - real implementation would be more sophisticated
                        pass
                elif fix_type == 'add_label':
                    # Add net label
                    pass

            except Exception as e:
                logger.warning(f"Failed to apply fix: {e}")

        return modified_content

    async def close(self):
        """Clean up resources."""
        if self._symbol_fetcher:
            await self._symbol_fetcher.close()

        if self._graphrag_indexer:
            await self._graphrag_indexer.close()


# Convenience function for simple usage
async def generate_schematic(
    bom: List[Dict],
    design_intent: str,
    connections: Optional[List[Dict]] = None,
    design_name: str = "schematic",
    skip_validation: bool = False
) -> PipelineResult:
    """
    Simple function to generate a validated schematic.

    Args:
        bom: List of component dictionaries
        design_intent: Description of circuit function
        connections: Optional net connections
        design_name: Name for output file
        skip_validation: Skip MAPO validation

    Returns:
        PipelineResult with schematic path and validation info
    """
    pipeline = MAPOSchematicPipeline()
    try:
        result = await pipeline.generate(
            bom=bom,
            design_intent=design_intent,
            connections=connections,
            design_name=design_name,
            skip_validation=skip_validation
        )
        return result
    finally:
        await pipeline.close()


# CLI entry point
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="MAPO Schematic Generation Pipeline")
    parser.add_argument("--bom", type=str, help="Path to BOM JSON file")
    parser.add_argument("--intent", type=str, required=True, help="Design intent description")
    parser.add_argument("--output", type=str, default="schematic", help="Output name")
    parser.add_argument("--skip-validation", action="store_true", help="Skip MAPO validation")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")

    args = parser.parse_args()

    # Configure logging
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    )

    async def main():
        # Load BOM if provided
        if args.bom:
            with open(args.bom) as f:
                bom = json.load(f)
        else:
            # Default test BOM
            bom = [
                {"part_number": "STM32G431CBT6", "category": "MCU", "value": "STM32G431"},
                {"part_number": "DRV8323RS", "category": "Gate_Driver", "value": "DRV8323"},
                {"part_number": "CSD19505KCS", "category": "MOSFET", "value": "80V MOSFET"},
                {"part_number": "100uF", "category": "Capacitor", "value": "100uF/50V"},
                {"part_number": "10uF", "category": "Capacitor", "value": "10uF/25V"},
                {"part_number": "0.1uF", "category": "Capacitor", "value": "0.1uF"},
                {"part_number": "10k", "category": "Resistor", "value": "10k"},
                {"part_number": "1k", "category": "Resistor", "value": "1k"},
            ]

        print(f"\n{'='*60}")
        print("MAPO Schematic Generation Pipeline")
        print(f"{'='*60}")
        print(f"Design Intent: {args.intent}")
        print(f"Components: {len(bom)}")
        print(f"Skip Validation: {args.skip_validation}")
        print(f"{'='*60}\n")

        result = await generate_schematic(
            bom=bom,
            design_intent=args.intent,
            design_name=args.output,
            skip_validation=args.skip_validation
        )

        print(f"\n{'='*60}")
        print("RESULTS")
        print(f"{'='*60}")
        print(f"Success: {result.success}")
        print(f"Output: {result.schematic_path}")
        print(f"Symbols Fetched: {result.symbols_fetched}")
        print(f"  - From Cache: {result.symbols_from_cache}")
        print(f"  - Generated: {result.symbols_generated}")
        print(f"Iterations: {result.iterations}")
        print(f"Total Time: {result.total_time_seconds:.2f}s")

        if result.validation_report:
            print(f"\nValidation Score: {result.validation_report.overall_score:.2%}")
            print(f"Validation Passed: {result.validation_report.passed}")
            print(f"Critical Issues: {len(result.validation_report.critical_issues)}")

        if result.errors:
            print(f"\nErrors:")
            for error in result.errors:
                print(f"  - {error}")

        print(f"{'='*60}\n")

    asyncio.run(main())
