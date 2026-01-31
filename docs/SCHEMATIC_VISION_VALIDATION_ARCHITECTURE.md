# Schematic Vision Validation Architecture (MAPO-Schematic)

## Executive Summary

This document defines the architecture for an LLM vision-based schematic validation and generation system. The system uses a multi-agent pipeline to:
1. Fetch real component symbols from manufacturer/vendor repositories
2. Store symbols in GraphRAG for semantic retrieval
3. Assemble schematics using actual KiCad symbols
4. Validate schematics using LLM vision with expert personas

---

## System Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    MAPO-Schematic Pipeline                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐    ┌───────────┐ │
│  │   Symbol     │───▶│   GraphRAG   │───▶│  Schematic   │───▶│  Vision   │ │
│  │   Fetcher    │    │   Storage    │    │   Assembler  │    │ Validator │ │
│  │   Agent      │    │              │    │   Agent      │    │   Loop    │ │
│  └──────────────┘    └──────────────┘    └──────────────┘    └───────────┘ │
│         │                   │                   │                   │       │
│         ▼                   ▼                   ▼                   ▼       │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │                     Shared Symbol Repository                          │  │
│  │  /opt/kicad-symbols/  on Terminal Server (157.173.102.118)           │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Phase 1: Symbol Fetcher Agent System

### 1.1 Symbol Source Priority Chain

```
Priority 1: Local Cache (/opt/kicad-symbols/)
    ↓ miss
Priority 2: KiCad Official Libraries (kicad-symbols GitHub)
    ↓ miss
Priority 3: SnapEDA/SnapMagic API
    ↓ miss
Priority 4: Ultra Librarian API
    ↓ miss
Priority 5: Manufacturer Libraries (TI, STM, Analog, Infineon)
    ↓ miss
Priority 6: LLM-Generated Symbol (fallback)
```

### 1.2 Symbol Fetcher Implementation

```python
# services/nexus-ee-design/src/agents/symbol-fetcher/symbol_fetcher_agent.py

from dataclasses import dataclass
from typing import Optional, List, Dict, Any
from pathlib import Path
import httpx
import asyncio

@dataclass
class SymbolSource:
    name: str
    priority: int
    api_url: Optional[str]
    local_path: Optional[Path]
    requires_auth: bool = False

SYMBOL_SOURCES = [
    SymbolSource("local_cache", 1, None, Path("/opt/kicad-symbols"), False),
    SymbolSource("kicad_official", 2, "https://gitlab.com/api/v4/projects/kicad%2Fkicad-symbols", None, False),
    SymbolSource("snapeda", 3, "https://www.snapeda.com/api/v1/", None, True),
    SymbolSource("ultralibrarian", 4, "https://app.ultralibrarian.com/api/v1/", None, True),
    SymbolSource("ti_webench", 5, "https://webench.ti.com/api/", None, True),
    SymbolSource("stm_cube", 5, "https://www.st.com/content/ccc/resource/", None, False),
]

class SymbolFetcherAgent:
    """
    Multi-source symbol fetcher with fallback chain.
    Stores fetched symbols in both local cache and GraphRAG.
    """

    def __init__(self, graphrag_client, cache_path: Path = Path("/opt/kicad-symbols")):
        self.graphrag = graphrag_client
        self.cache_path = cache_path
        self.http_client = httpx.AsyncClient(timeout=30.0)

    async def fetch_symbol(self,
                          part_number: str,
                          manufacturer: Optional[str] = None,
                          category: str = "generic") -> Dict[str, Any]:
        """
        Fetch symbol using priority chain.

        Returns:
            {
                "symbol": KiCad S-expression symbol data,
                "footprint": Footprint data if available,
                "datasheet_url": Link to datasheet,
                "source": Which source provided the symbol,
                "metadata": Additional part info
            }
        """
        for source in sorted(SYMBOL_SOURCES, key=lambda s: s.priority):
            try:
                result = await self._fetch_from_source(source, part_number, manufacturer)
                if result:
                    # Store in local cache
                    await self._cache_symbol(part_number, result)
                    # Index in GraphRAG
                    await self._index_in_graphrag(part_number, result, category)
                    return result
            except Exception as e:
                print(f"[SymbolFetcher] {source.name} failed: {e}")
                continue

        # Fallback: Generate symbol using LLM
        return await self._generate_symbol_with_llm(part_number, manufacturer, category)

    async def _fetch_from_local_cache(self, part_number: str) -> Optional[Dict]:
        """Check local symbol cache first."""
        # Search by part number in cached symbols
        symbol_path = self.cache_path / f"{part_number}.kicad_sym"
        if symbol_path.exists():
            return {
                "symbol": symbol_path.read_text(),
                "source": "local_cache",
                "footprint": self._find_matching_footprint(part_number)
            }

        # Try fuzzy match using GraphRAG
        similar = await self.graphrag.search(
            f"KiCad symbol part_number:{part_number}",
            top_k=1,
            threshold=0.9
        )
        if similar:
            return similar[0]
        return None

    async def _fetch_from_snapeda(self, part_number: str, manufacturer: str) -> Optional[Dict]:
        """
        SnapEDA API integration.
        API Docs: https://www.snapeda.com/api/
        """
        search_url = f"https://www.snapeda.com/api/v1/parts/search"
        params = {"q": part_number}
        if manufacturer:
            params["manufacturer"] = manufacturer

        response = await self.http_client.get(search_url, params=params)
        if response.status_code != 200:
            return None

        data = response.json()
        if not data.get("results"):
            return None

        # Get first matching part
        part = data["results"][0]
        part_id = part["id"]

        # Download KiCad format
        download_url = f"https://www.snapeda.com/api/v1/parts/{part_id}/download"
        download_response = await self.http_client.get(
            download_url,
            params={"format": "kicad"}
        )

        if download_response.status_code == 200:
            return {
                "symbol": download_response.text,
                "source": "snapeda",
                "datasheet_url": part.get("datasheet_url"),
                "metadata": {
                    "manufacturer": part.get("manufacturer"),
                    "description": part.get("description"),
                    "package": part.get("package")
                }
            }
        return None

    async def _fetch_from_manufacturer(self,
                                       source: SymbolSource,
                                       part_number: str) -> Optional[Dict]:
        """
        Fetch from manufacturer-specific libraries.
        Each manufacturer has different API/download patterns.
        """
        manufacturer_handlers = {
            "ti_webench": self._fetch_ti_symbol,
            "stm_cube": self._fetch_stm_symbol,
            "infineon_designer": self._fetch_infineon_symbol,
            "analog_ltspice": self._fetch_analog_symbol,
        }

        handler = manufacturer_handlers.get(source.name)
        if handler:
            return await handler(part_number)
        return None

    async def _generate_symbol_with_llm(self,
                                        part_number: str,
                                        manufacturer: str,
                                        category: str) -> Dict:
        """
        Last resort: Generate symbol using LLM with datasheet context.
        """
        # Search for datasheet info
        datasheet_context = await self._fetch_datasheet_info(part_number, manufacturer)

        prompt = f"""Generate a KiCad symbol for:
Part Number: {part_number}
Manufacturer: {manufacturer or 'Unknown'}
Category: {category}

Datasheet Information:
{datasheet_context}

Generate a valid KiCad S-expression symbol with:
1. Correct pin names and numbers from datasheet
2. Appropriate graphical representation
3. Power pins (VCC, GND) if applicable
4. Proper pin types (input, output, bidirectional, power_in, etc.)

Output ONLY the KiCad S-expression, no explanation."""

        # Call LLM (Claude Sonnet for efficiency)
        from anthropic import Anthropic
        client = Anthropic()
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=4096,
            messages=[{"role": "user", "content": prompt}]
        )

        generated_symbol = response.content[0].text

        return {
            "symbol": generated_symbol,
            "source": "llm_generated",
            "metadata": {
                "part_number": part_number,
                "manufacturer": manufacturer,
                "generated": True,
                "needs_review": True
            }
        }

    async def _index_in_graphrag(self, part_number: str, symbol_data: Dict, category: str):
        """Index symbol metadata in GraphRAG for semantic search."""
        document = {
            "type": "kicad_symbol",
            "part_number": part_number,
            "category": category,
            "source": symbol_data.get("source"),
            "symbol_data": symbol_data.get("symbol")[:1000],  # Truncate for embedding
            "metadata": symbol_data.get("metadata", {}),
            "datasheet_url": symbol_data.get("datasheet_url"),
            "indexed_at": datetime.utcnow().isoformat()
        }

        await self.graphrag.add_document(
            collection="kicad_symbols",
            document=document,
            embedding_text=f"{part_number} {category} {symbol_data.get('metadata', {}).get('description', '')}"
        )
```

### 1.3 KiCad HTTP Libraries Integration

For dynamic symbol loading in KiCanvas, implement KiCad HTTP Libraries API:

```python
# services/nexus-ee-design/src/api/kicad-http-library.ts

import { Hono } from 'hono';
import { symbolFetcher } from './symbol-fetcher';

const app = new Hono();

/**
 * KiCad HTTP Libraries API Implementation
 * Spec: https://dev-docs.kicad.org/en/apis-and-binding/http-libraries/
 */

// GET /api/v1/libraries - List available libraries
app.get('/api/v1/libraries', async (c) => {
  return c.json([
    { name: "project_symbols", description: "Project-specific symbols" },
    { name: "cached_symbols", description: "Locally cached vendor symbols" },
    { name: "vendor_symbols", description: "Dynamic vendor symbol lookup" }
  ]);
});

// GET /api/v1/libraries/:name/parts - List parts in library
app.get('/api/v1/libraries/:name/parts', async (c) => {
  const libraryName = c.req.param('name');
  const category = c.req.query('category');

  // Query GraphRAG for indexed symbols
  const symbols = await graphragClient.search({
    collection: "kicad_symbols",
    filter: { category },
    limit: 100
  });

  return c.json({
    parts: symbols.map(s => ({
      id: s.part_number,
      name: s.part_number,
      description: s.metadata?.description || '',
      category: s.category
    }))
  });
});

// GET /api/v1/libraries/:name/parts/:id - Get specific part symbol
app.get('/api/v1/libraries/:name/parts/:id', async (c) => {
  const partId = c.req.param('id');

  // Fetch from cache or external source
  const symbolData = await symbolFetcher.fetch_symbol(partId);

  return c.json({
    id: partId,
    symbol: symbolData.symbol,
    footprints: symbolData.footprint ? [symbolData.footprint] : [],
    datasheet: symbolData.datasheet_url
  });
});

export default app;
```

---

## Phase 2: GraphRAG Symbol Storage

### 2.1 Symbol Schema

```typescript
// services/nexus-ee-design/src/types/symbol-schema.ts

interface KiCadSymbolDocument {
  // Identity
  id: string;                    // UUID
  part_number: string;           // e.g., "STM32G431CBT6"
  manufacturer: string;          // e.g., "STMicroelectronics"

  // Classification
  category: SymbolCategory;      // MCU, MOSFET, Capacitor, etc.
  subcategory?: string;          // ARM Cortex-M4, N-Channel, MLCC
  package: string;               // LQFP48, TO-220, 0603

  // Symbol Data
  symbol_sexp: string;           // Full KiCad S-expression
  footprint_sexp?: string;       // Matching footprint if available

  // Metadata for search
  description: string;
  keywords: string[];
  specifications: Record<string, string | number>;

  // Provenance
  source: SymbolSource;          // snapeda, kicad_official, manufacturer, llm_generated
  datasheet_url?: string;
  source_url?: string;
  fetched_at: Date;

  // Quality indicators
  verified: boolean;             // Human or vision-verified
  usage_count: number;           // How often used in projects
  last_used_at?: Date;
}

type SymbolCategory =
  | 'MCU'
  | 'MOSFET'
  | 'Gate_Driver'
  | 'OpAmp'
  | 'Capacitor'
  | 'Resistor'
  | 'Inductor'
  | 'Connector'
  | 'Power_Management'
  | 'Sensor'
  | 'Communication'
  | 'Memory'
  | 'Other';
```

### 2.2 GraphRAG Integration

```python
# services/nexus-ee-design/src/graphrag/symbol_indexer.py

class SymbolGraphRAGIndexer:
    """
    Index KiCad symbols in GraphRAG for semantic search and relationship discovery.
    """

    def __init__(self, neo4j_driver, embedding_model):
        self.neo4j = neo4j_driver
        self.embedder = embedding_model

    async def index_symbol(self, symbol: KiCadSymbolDocument):
        """
        Index symbol in both vector store and knowledge graph.
        """
        # Generate embedding from searchable text
        embedding_text = f"""
        {symbol.part_number} {symbol.manufacturer}
        {symbol.category} {symbol.subcategory or ''}
        {symbol.description}
        {' '.join(symbol.keywords)}
        Package: {symbol.package}
        """
        embedding = await self.embedder.embed(embedding_text)

        # Create knowledge graph nodes and relationships
        cypher = """
        MERGE (s:Symbol {part_number: $part_number})
        SET s += {
            manufacturer: $manufacturer,
            category: $category,
            package: $package,
            description: $description,
            embedding: $embedding
        }

        MERGE (m:Manufacturer {name: $manufacturer})
        MERGE (c:Category {name: $category})
        MERGE (p:Package {name: $package})

        MERGE (s)-[:MADE_BY]->(m)
        MERGE (s)-[:IN_CATEGORY]->(c)
        MERGE (s)-[:HAS_PACKAGE]->(p)

        // Link to related symbols (same category/package)
        WITH s
        MATCH (related:Symbol)
        WHERE related.category = s.category
          AND related.part_number <> s.part_number
        MERGE (s)-[:SIMILAR_TO {type: 'category'}]->(related)
        """

        await self.neo4j.run(cypher, {
            "part_number": symbol.part_number,
            "manufacturer": symbol.manufacturer,
            "category": symbol.category,
            "package": symbol.package,
            "description": symbol.description,
            "embedding": embedding.tolist()
        })

    async def search_symbols(self,
                            query: str,
                            category: Optional[str] = None,
                            top_k: int = 10) -> List[KiCadSymbolDocument]:
        """
        Semantic search for symbols.
        """
        query_embedding = await self.embedder.embed(query)

        cypher = """
        CALL db.index.vector.queryNodes('symbol_embeddings', $top_k, $embedding)
        YIELD node, score
        WHERE score > 0.7
        AND ($category IS NULL OR node.category = $category)
        RETURN node, score
        ORDER BY score DESC
        """

        results = await self.neo4j.run(cypher, {
            "embedding": query_embedding.tolist(),
            "top_k": top_k,
            "category": category
        })

        return [self._node_to_document(r["node"]) for r in results]

    async def find_compatible_symbols(self, symbol_id: str) -> List[Dict]:
        """
        Find symbols that are pin-compatible or functional alternatives.
        """
        cypher = """
        MATCH (s:Symbol {part_number: $symbol_id})
        MATCH (s)-[:SIMILAR_TO]-(alt:Symbol)
        WHERE alt.package = s.package  // Pin-compatible
        RETURN alt,
               gds.similarity.cosine(s.embedding, alt.embedding) as similarity
        ORDER BY similarity DESC
        LIMIT 5
        """

        results = await self.neo4j.run(cypher, {"symbol_id": symbol_id})
        return results
```

---

## Phase 3: Schematic Assembler Agent

### 3.1 Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                  Schematic Assembler Agent                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Input: BOM + Block Diagram + Component Selection                │
│                     │                                            │
│                     ▼                                            │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │  1. Symbol Resolution                                        ││
│  │     - Query GraphRAG for each component                      ││
│  │     - Fetch missing symbols via SymbolFetcher                ││
│  │     - Validate symbol quality                                ││
│  └─────────────────────────────────────────────────────────────┘│
│                     │                                            │
│                     ▼                                            │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │  2. Hierarchical Sheet Planning                              ││
│  │     - Group components by functional block                   ││
│  │     - Create sheet hierarchy (Power, MCU, Driver, etc.)      ││
│  │     - Define inter-sheet connections                         ││
│  └─────────────────────────────────────────────────────────────┘│
│                     │                                            │
│                     ▼                                            │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │  3. Component Placement                                      ││
│  │     - Use reference design patterns                          ││
│  │     - Apply EE placement heuristics                          ││
│  │     - Optimize for signal flow (left→right, top→bottom)      ││
│  └─────────────────────────────────────────────────────────────┘│
│                     │                                            │
│                     ▼                                            │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │  4. Net Routing                                              ││
│  │     - Connect pins based on netlist                          ││
│  │     - Apply Manhattan routing                                ││
│  │     - Add junction dots, labels                              ││
│  └─────────────────────────────────────────────────────────────┘│
│                     │                                            │
│                     ▼                                            │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │  5. Annotation & Labels                                      ││
│  │     - Reference designators (U1, R1, C1...)                  ││
│  │     - Net labels (VCC, GND, SIG_A...)                        ││
│  │     - Add power flags, no-connect markers                    ││
│  └─────────────────────────────────────────────────────────────┘│
│                     │                                            │
│                     ▼                                            │
│  Output: .kicad_sch file with real symbols                       │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 3.2 Implementation

```python
# services/nexus-ee-design/src/agents/schematic-assembler/assembler_agent.py

from dataclasses import dataclass
from typing import List, Dict, Tuple
import math

@dataclass
class PlacedComponent:
    symbol_id: str
    reference: str          # U1, R1, etc.
    position: Tuple[float, float]
    rotation: int           # 0, 90, 180, 270
    mirror: bool
    unit: int = 1          # For multi-unit symbols
    properties: Dict[str, str] = None

@dataclass
class Wire:
    start: Tuple[float, float]
    end: Tuple[float, float]

@dataclass
class SchematicSheet:
    name: str
    filename: str
    components: List[PlacedComponent]
    wires: List[Wire]
    labels: List[Dict]
    hierarchical_labels: List[Dict]

class SchematicAssemblerAgent:
    """
    Assembles KiCad schematics using real symbols from GraphRAG.
    """

    GRID_UNIT = 2.54  # mm (100 mil KiCad grid)

    def __init__(self, symbol_fetcher, graphrag_client):
        self.symbol_fetcher = symbol_fetcher
        self.graphrag = graphrag_client
        self.ref_counter = {}

    async def assemble_schematic(self,
                                 bom: List[Dict],
                                 block_diagram: Dict,
                                 connections: List[Dict]) -> List[SchematicSheet]:
        """
        Main entry point for schematic assembly.

        Args:
            bom: Bill of materials with part numbers
            block_diagram: Functional block structure
            connections: Net connections between components

        Returns:
            List of SchematicSheet objects
        """
        # Step 1: Resolve all symbols
        resolved_symbols = await self._resolve_symbols(bom)

        # Step 2: Plan hierarchical sheets
        sheets = self._plan_hierarchy(block_diagram, resolved_symbols)

        # Step 3: Place components on each sheet
        for sheet in sheets:
            self._place_components(sheet, resolved_symbols)

        # Step 4: Route wires
        for sheet in sheets:
            self._route_wires(sheet, connections)

        # Step 5: Add annotations
        for sheet in sheets:
            self._add_annotations(sheet)

        return sheets

    async def _resolve_symbols(self, bom: List[Dict]) -> Dict[str, Dict]:
        """Fetch all required symbols."""
        resolved = {}

        for item in bom:
            part_number = item.get('part_number') or item.get('mpn')
            manufacturer = item.get('manufacturer')
            category = item.get('category', 'generic')

            if part_number in resolved:
                continue

            # Try GraphRAG first
            cached = await self.graphrag.search(
                f"part_number:{part_number}",
                collection="kicad_symbols",
                top_k=1
            )

            if cached and cached[0].get('verified'):
                resolved[part_number] = cached[0]
            else:
                # Fetch from external sources
                symbol_data = await self.symbol_fetcher.fetch_symbol(
                    part_number, manufacturer, category
                )
                resolved[part_number] = symbol_data

        return resolved

    def _plan_hierarchy(self,
                       block_diagram: Dict,
                       symbols: Dict) -> List[SchematicSheet]:
        """Create hierarchical sheet structure."""
        sheets = []

        # Root sheet with hierarchical sheet symbols
        root = SchematicSheet(
            name="Root",
            filename="root.kicad_sch",
            components=[],
            wires=[],
            labels=[],
            hierarchical_labels=[]
        )
        sheets.append(root)

        # Create sheet for each functional block
        for block_name, block_data in block_diagram.get('blocks', {}).items():
            sheet = SchematicSheet(
                name=block_name,
                filename=f"{block_name.lower().replace(' ', '_')}.kicad_sch",
                components=[],
                wires=[],
                labels=[],
                hierarchical_labels=block_data.get('external_pins', [])
            )
            sheets.append(sheet)

            # Add components belonging to this block
            for comp_ref in block_data.get('components', []):
                # Will be filled in placement phase
                pass

        return sheets

    def _place_components(self, sheet: SchematicSheet, symbols: Dict):
        """
        Place components using signal-flow heuristics.

        Placement strategy:
        - Power components (regulators) at top
        - Signal flow left-to-right
        - Inputs on left, outputs on right
        - Bypass capacitors near their ICs
        - Ground symbols at bottom
        """
        # Group by component type
        ics = []
        passives = []
        connectors = []
        power = []

        for comp in sheet.components:
            symbol = symbols.get(comp.symbol_id, {})
            category = symbol.get('category', 'Other')

            if category in ['MCU', 'Gate_Driver', 'OpAmp', 'Communication']:
                ics.append(comp)
            elif category in ['Resistor', 'Capacitor', 'Inductor']:
                passives.append(comp)
            elif category == 'Connector':
                connectors.append(comp)
            elif category == 'Power_Management':
                power.append(comp)

        # Place ICs in center, spaced appropriately
        y_offset = 50.0  # mm from top
        for i, ic in enumerate(ics):
            ic.position = (100.0 + i * 80.0, y_offset + 50.0)

        # Place passives near their associated ICs
        # (determined by netlist analysis)
        passive_y = y_offset + 100.0
        for i, passive in enumerate(passives):
            passive.position = (50.0 + (i % 10) * 15.0, passive_y + (i // 10) * 15.0)

        # Connectors on edges
        for i, conn in enumerate(connectors):
            conn.position = (20.0, 50.0 + i * 30.0)  # Left edge

    def _route_wires(self, sheet: SchematicSheet, connections: List[Dict]):
        """
        Route wires between component pins using Manhattan routing.
        """
        for conn in connections:
            from_ref = conn.get('from_ref')
            from_pin = conn.get('from_pin')
            to_ref = conn.get('to_ref')
            to_pin = conn.get('to_pin')
            net_name = conn.get('net')

            # Find component positions
            from_comp = next((c for c in sheet.components if c.reference == from_ref), None)
            to_comp = next((c for c in sheet.components if c.reference == to_ref), None)

            if not from_comp or not to_comp:
                continue

            # Get pin positions (would need symbol geometry)
            from_pos = self._get_pin_position(from_comp, from_pin)
            to_pos = self._get_pin_position(to_comp, to_pin)

            # Manhattan routing (L-shaped or Z-shaped)
            wires = self._manhattan_route(from_pos, to_pos)
            sheet.wires.extend(wires)

            # Add net label if named net
            if net_name and not net_name.startswith('Net-'):
                mid_point = ((from_pos[0] + to_pos[0]) / 2, (from_pos[1] + to_pos[1]) / 2)
                sheet.labels.append({
                    'text': net_name,
                    'position': mid_point,
                    'type': 'net_label'
                })

    def _manhattan_route(self,
                        start: Tuple[float, float],
                        end: Tuple[float, float]) -> List[Wire]:
        """
        Create Manhattan (orthogonal) routing between two points.
        """
        wires = []

        # Simple L-route: horizontal then vertical
        mid_x = end[0]
        mid_y = start[1]

        # Horizontal segment
        if abs(start[0] - mid_x) > 0.1:
            wires.append(Wire(start, (mid_x, mid_y)))

        # Vertical segment
        if abs(mid_y - end[1]) > 0.1:
            wires.append(Wire((mid_x, mid_y), end))

        return wires

    def generate_kicad_sch(self, sheet: SchematicSheet, symbols: Dict) -> str:
        """
        Generate KiCad S-expression schematic file.
        """
        sexp = ['kicad_sch', ['version', '20231120'], ['generator', 'nexus_ee_design']]

        # Add library symbols
        lib_symbols = ['lib_symbols']
        for comp in sheet.components:
            symbol_data = symbols.get(comp.symbol_id, {})
            if symbol_data.get('symbol'):
                lib_symbols.append(self._parse_symbol_sexp(symbol_data['symbol']))
        sexp.append(lib_symbols)

        # Add component instances
        for comp in sheet.components:
            symbol_instance = [
                'symbol',
                ['lib_id', comp.symbol_id],
                ['at', comp.position[0], comp.position[1], comp.rotation],
                ['unit', comp.unit],
                ['exclude_from_sim', 'no'],
                ['in_bom', 'yes'],
                ['on_board', 'yes'],
                ['uuid', self._generate_uuid()],
                ['property', 'Reference', comp.reference, ['at', comp.position[0], comp.position[1] - 5, 0]],
            ]
            sexp.append(symbol_instance)

        # Add wires
        for wire in sheet.wires:
            wire_sexp = [
                'wire',
                ['pts', ['xy', wire.start[0], wire.start[1]], ['xy', wire.end[0], wire.end[1]]],
                ['stroke', ['width', 0], ['type', 'default']],
                ['uuid', self._generate_uuid()]
            ]
            sexp.append(wire_sexp)

        # Add labels
        for label in sheet.labels:
            label_sexp = [
                'label', label['text'],
                ['at', label['position'][0], label['position'][1], 0],
                ['uuid', self._generate_uuid()]
            ]
            sexp.append(label_sexp)

        return self._sexp_to_string(sexp)
```

---

## Phase 4: LLM Vision Validation Loop

### 4.1 Multi-Expert Vision Validation

```python
# services/nexus-ee-design/src/validation/schematic_vision_validator.py

from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
from enum import Enum
import base64
import anthropic
import asyncio

class ExpertPersona(Enum):
    CIRCUIT_ANALYST = "circuit_analyst"
    COMPONENT_VALIDATOR = "component_validator"
    CONNECTION_VERIFIER = "connection_verifier"
    MANUFACTURABILITY_REVIEWER = "manufacturability_reviewer"
    REFERENCE_COMPARATOR = "reference_comparator"

@dataclass
class ValidationResult:
    expert: ExpertPersona
    score: float           # 0.0 - 1.0
    passed: bool
    issues: List[Dict]
    suggestions: List[str]
    confidence: float

@dataclass
class SchematicValidationReport:
    overall_score: float
    passed: bool
    expert_results: List[ValidationResult]
    critical_issues: List[Dict]
    recommended_fixes: List[Dict]
    iteration: int

class SchematicVisionValidator:
    """
    LLM Vision-based schematic validation using multiple expert personas.
    Implements MAPO (Multi-Agent Pattern Orchestration) for quality assurance.
    """

    EXPERT_PROMPTS = {
        ExpertPersona.CIRCUIT_ANALYST: """You are an expert circuit analyst reviewing a KiCad schematic.

Analyze this schematic image for:
1. **Circuit Topology**: Is the overall circuit topology correct for the intended function?
2. **Signal Flow**: Does signal flow logically (inputs → processing → outputs)?
3. **Power Distribution**: Are power rails properly distributed to all components?
4. **Feedback Loops**: Are feedback paths correctly implemented (if applicable)?
5. **Protection Circuits**: Are necessary protection elements present (ESD, overcurrent)?

Rate each aspect 0-100 and identify specific issues with component references.""",

        ExpertPersona.COMPONENT_VALIDATOR: """You are a component specialist validating KiCad schematic symbols.

Check each component for:
1. **Symbol Accuracy**: Do symbols match their actual component representation?
2. **Pin Mapping**: Are pins correctly labeled and positioned?
3. **Values/Ratings**: Are component values visible and appropriate?
4. **Reference Designators**: Are designators unique and follow conventions (R1, C1, U1)?
5. **Footprint Association**: Do components have appropriate footprints assigned?

List any components with incorrect or missing symbol data.""",

        ExpertPersona.CONNECTION_VERIFIER: """You are a connectivity specialist checking schematic wiring.

Verify:
1. **Wire Continuity**: Are all wires properly connected (no floating endpoints)?
2. **Junction Dots**: Are wire junctions clearly marked with dots?
3. **Net Labels**: Are important nets labeled consistently?
4. **Power Connections**: Are VCC and GND properly connected to all ICs?
5. **No-Connect Pins**: Are unused pins marked with no-connect flags?
6. **Bus Connections**: Are buses properly labeled and connected?

Identify any disconnected nets or ambiguous connections.""",

        ExpertPersona.MANUFACTURABILITY_REVIEWER: """You are a DFM specialist reviewing schematic for manufacturability.

Evaluate:
1. **BOM Completeness**: Are all components properly specified for ordering?
2. **Standard Values**: Are resistors/capacitors using standard E-series values?
3. **Package Selection**: Are packages appropriate for the design (SMD vs THT)?
4. **Second Sourcing**: Can critical components be second-sourced?
5. **Assembly Clarity**: Is the schematic clear enough for assembly?

Flag any components that may cause manufacturing issues.""",

        ExpertPersona.REFERENCE_COMPARATOR: """You are comparing this schematic against reference designs.

Given the intended circuit function, compare against known-good implementations:
1. **Best Practices**: Does the design follow manufacturer recommendations?
2. **Application Notes**: Are reference design patterns correctly applied?
3. **Critical Component Placement**: Are key components (decoupling caps, etc.) placed correctly?
4. **Known Issues**: Does the design avoid known problematic patterns?

Identify deviations from reference designs that may cause issues."""
    }

    def __init__(self,
                 primary_model: str = "claude-sonnet-4-20250514",
                 verification_model: str = "claude-opus-4-20250514"):
        self.primary_model = primary_model
        self.verification_model = verification_model
        self.client = anthropic.Anthropic()

    async def validate_schematic(self,
                                schematic_image: bytes,
                                design_intent: str,
                                reference_designs: Optional[List[bytes]] = None,
                                iteration: int = 0) -> SchematicValidationReport:
        """
        Run full validation pipeline with all expert personas.

        Args:
            schematic_image: PNG/JPEG of rendered schematic
            design_intent: Description of what the circuit should do
            reference_designs: Optional reference schematic images
            iteration: Current iteration number in MAPO loop

        Returns:
            Comprehensive validation report
        """
        # Run all experts in parallel
        expert_tasks = []
        for expert in ExpertPersona:
            task = self._run_expert_validation(
                expert, schematic_image, design_intent, reference_designs
            )
            expert_tasks.append(task)

        expert_results = await asyncio.gather(*expert_tasks)

        # Calculate overall score (weighted average)
        weights = {
            ExpertPersona.CIRCUIT_ANALYST: 0.30,
            ExpertPersona.COMPONENT_VALIDATOR: 0.25,
            ExpertPersona.CONNECTION_VERIFIER: 0.25,
            ExpertPersona.MANUFACTURABILITY_REVIEWER: 0.10,
            ExpertPersona.REFERENCE_COMPARATOR: 0.10
        }

        overall_score = sum(
            result.score * weights[result.expert]
            for result in expert_results
        )

        # Aggregate critical issues
        critical_issues = []
        for result in expert_results:
            for issue in result.issues:
                if issue.get('severity') == 'critical':
                    critical_issues.append({
                        **issue,
                        'expert': result.expert.value
                    })

        # Generate fix recommendations
        recommended_fixes = await self._generate_fix_recommendations(
            expert_results, critical_issues
        )

        return SchematicValidationReport(
            overall_score=overall_score,
            passed=overall_score >= 0.8 and len(critical_issues) == 0,
            expert_results=expert_results,
            critical_issues=critical_issues,
            recommended_fixes=recommended_fixes,
            iteration=iteration
        )

    async def _run_expert_validation(self,
                                     expert: ExpertPersona,
                                     schematic_image: bytes,
                                     design_intent: str,
                                     reference_designs: Optional[List[bytes]]) -> ValidationResult:
        """Run single expert validation."""

        # Build message with image
        content = [
            {
                "type": "text",
                "text": f"""Design Intent: {design_intent}

{self.EXPERT_PROMPTS[expert]}

Provide your analysis in the following JSON format:
{{
    "score": <0-100>,
    "passed": <true/false>,
    "issues": [
        {{
            "component": "<ref designator or 'general'>",
            "description": "<issue description>",
            "severity": "critical|major|minor",
            "location": "<description of location in schematic>"
        }}
    ],
    "suggestions": ["<improvement suggestion>", ...],
    "confidence": <0.0-1.0>
}}"""
            },
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/png",
                    "data": base64.b64encode(schematic_image).decode()
                }
            }
        ]

        # Add reference designs if provided
        if reference_designs and expert == ExpertPersona.REFERENCE_COMPARATOR:
            for i, ref_img in enumerate(reference_designs[:2]):  # Max 2 references
                content.append({
                    "type": "text",
                    "text": f"\nReference Design {i+1}:"
                })
                content.append({
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/png",
                        "data": base64.b64encode(ref_img).decode()
                    }
                })

        response = await self.client.messages.create(
            model=self.primary_model,
            max_tokens=2048,
            messages=[{"role": "user", "content": content}]
        )

        # Parse JSON response
        import json
        result_text = response.content[0].text

        # Extract JSON from response
        json_match = re.search(r'\{[\s\S]*\}', result_text)
        if json_match:
            result_data = json.loads(json_match.group())
        else:
            result_data = {"score": 50, "passed": False, "issues": [], "suggestions": [], "confidence": 0.5}

        return ValidationResult(
            expert=expert,
            score=result_data.get('score', 50) / 100.0,
            passed=result_data.get('passed', False),
            issues=result_data.get('issues', []),
            suggestions=result_data.get('suggestions', []),
            confidence=result_data.get('confidence', 0.5)
        )

    async def _generate_fix_recommendations(self,
                                           expert_results: List[ValidationResult],
                                           critical_issues: List[Dict]) -> List[Dict]:
        """Generate actionable fix recommendations from validation results."""

        # Consolidate all issues
        all_issues = []
        for result in expert_results:
            all_issues.extend(result.issues)

        if not all_issues:
            return []

        # Use verification model for fix generation
        prompt = f"""Based on these schematic validation issues, generate specific fix recommendations:

Issues:
{json.dumps(all_issues, indent=2)}

For each issue, provide a fix recommendation in this format:
{{
    "issue_ref": "<component or general>",
    "fix_type": "add_component|remove_component|modify_connection|change_value|add_label|other",
    "description": "<what to change>",
    "kicad_action": "<specific KiCad operation or S-expression modification>",
    "priority": "high|medium|low"
}}

Return a JSON array of recommendations."""

        response = await self.client.messages.create(
            model=self.verification_model,
            max_tokens=2048,
            messages=[{"role": "user", "content": prompt}]
        )

        # Parse recommendations
        result_text = response.content[0].text
        json_match = re.search(r'\[[\s\S]*\]', result_text)
        if json_match:
            return json.loads(json_match.group())
        return []
```

### 4.2 MAPO Validation Loop

```python
# services/nexus-ee-design/src/validation/mapo_schematic_loop.py

class MAPOSchematicLoop:
    """
    Multi-Agent Pattern Orchestration loop for schematic generation and validation.

    Loop:
    1. Generate schematic using Assembler Agent
    2. Render to image
    3. Validate with Vision Validator
    4. If failed, apply fixes and repeat
    5. Continue until passed or max iterations
    """

    MAX_ITERATIONS = 5
    PASS_THRESHOLD = 0.85

    def __init__(self,
                 assembler: SchematicAssemblerAgent,
                 validator: SchematicVisionValidator,
                 renderer: KiCanvasRenderer):
        self.assembler = assembler
        self.validator = validator
        self.renderer = renderer

    async def generate_and_validate(self,
                                    bom: List[Dict],
                                    block_diagram: Dict,
                                    connections: List[Dict],
                                    design_intent: str) -> Tuple[List[SchematicSheet], SchematicValidationReport]:
        """
        Main MAPO loop entry point.

        Returns:
            Tuple of (final schematic sheets, final validation report)
        """
        best_schematic = None
        best_score = 0.0
        history = []

        for iteration in range(self.MAX_ITERATIONS):
            print(f"\n[MAPO] Iteration {iteration + 1}/{self.MAX_ITERATIONS}")

            # Step 1: Generate/modify schematic
            if iteration == 0:
                sheets = await self.assembler.assemble_schematic(
                    bom, block_diagram, connections
                )
            else:
                # Apply fixes from previous validation
                sheets = await self._apply_fixes(
                    sheets,
                    history[-1].recommended_fixes
                )

            # Step 2: Render to image
            schematic_image = await self.renderer.render_to_png(sheets[0])  # Root sheet

            # Step 3: Validate with vision
            report = await self.validator.validate_schematic(
                schematic_image,
                design_intent,
                iteration=iteration
            )

            history.append(report)
            print(f"[MAPO] Score: {report.overall_score:.2f}, Passed: {report.passed}")
            print(f"[MAPO] Critical issues: {len(report.critical_issues)}")

            # Track best
            if report.overall_score > best_score:
                best_score = report.overall_score
                best_schematic = sheets

            # Step 4: Check termination
            if report.passed:
                print(f"[MAPO] Validation passed at iteration {iteration + 1}")
                return sheets, report

            if not report.recommended_fixes:
                print("[MAPO] No more fixes to apply, terminating")
                break

        # Return best attempt if never passed
        print(f"[MAPO] Max iterations reached, returning best (score={best_score:.2f})")
        return best_schematic, history[-1]

    async def _apply_fixes(self,
                          sheets: List[SchematicSheet],
                          fixes: List[Dict]) -> List[SchematicSheet]:
        """Apply recommended fixes to schematic."""

        for fix in fixes:
            fix_type = fix.get('fix_type')

            if fix_type == 'add_component':
                # Add missing component
                await self._add_component(sheets, fix)
            elif fix_type == 'remove_component':
                # Remove problematic component
                await self._remove_component(sheets, fix)
            elif fix_type == 'modify_connection':
                # Fix wire connections
                await self._modify_connection(sheets, fix)
            elif fix_type == 'change_value':
                # Update component value
                await self._change_value(sheets, fix)
            elif fix_type == 'add_label':
                # Add net label
                await self._add_label(sheets, fix)

        return sheets
```

### 4.3 KiCanvas Image Renderer

```typescript
// services/nexus-ee-design/src/utils/kicanvas-renderer.ts

import puppeteer from 'puppeteer';

export class KiCanvasRenderer {
  private browser: puppeteer.Browser | null = null;

  async initialize() {
    this.browser = await puppeteer.launch({
      headless: true,
      args: ['--no-sandbox', '--disable-setuid-sandbox']
    });
  }

  async renderToPng(schematicContent: string): Promise<Buffer> {
    if (!this.browser) {
      await this.initialize();
    }

    const page = await this.browser!.newPage();
    await page.setViewport({ width: 1920, height: 1080 });

    // Create HTML page with KiCanvas
    const html = `
      <!DOCTYPE html>
      <html>
      <head>
        <script type="module" src="https://kicanvas.org/kicanvas/kicanvas.js"></script>
        <style>
          body { margin: 0; background: white; }
          kicanvas-embed { width: 100vw; height: 100vh; }
        </style>
      </head>
      <body>
        <kicanvas-embed id="viewer" controls></kicanvas-embed>
        <script type="module">
          const viewer = document.getElementById('viewer');
          const schematicBlob = new Blob([${JSON.stringify(schematicContent)}], { type: 'text/plain' });
          const url = URL.createObjectURL(schematicBlob);
          viewer.src = url + '#schematic.kicad_sch';
        </script>
      </body>
      </html>
    `;

    await page.setContent(html);

    // Wait for KiCanvas to render
    await page.waitForSelector('kicanvas-embed', { timeout: 10000 });
    await page.waitForFunction(() => {
      const viewer = document.querySelector('kicanvas-embed');
      return viewer?.shadowRoot?.querySelector('canvas') !== null;
    }, { timeout: 10000 });

    // Additional wait for rendering to complete
    await new Promise(resolve => setTimeout(resolve, 2000));

    // Take screenshot
    const screenshot = await page.screenshot({
      type: 'png',
      fullPage: false
    });

    await page.close();
    return screenshot;
  }

  async close() {
    if (this.browser) {
      await this.browser.close();
      this.browser = null;
    }
  }
}
```

---

## Phase 5: Integration Architecture

### 5.1 Full Pipeline Flow

```
User Request: "Generate FOC ESC schematic"
                    │
                    ▼
┌─────────────────────────────────────────────────────────────────┐
│  1. Design Intent Analysis (LLM)                                │
│     Input: User request + project context                       │
│     Output: Structured requirements, block diagram              │
└─────────────────────────────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────────────────────────┐
│  2. Component Selection (LLM + GraphRAG)                        │
│     Input: Requirements, available symbols                      │
│     Output: BOM with part numbers, categories                   │
└─────────────────────────────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────────────────────────┐
│  3. Symbol Fetching (Symbol Fetcher Agent)                      │
│     Input: BOM part numbers                                     │
│     Output: Resolved KiCad symbols for all parts                │
│     Sources: Local cache → KiCad official → SnapEDA → Mfr      │
└─────────────────────────────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────────────────────────┐
│  4. Schematic Assembly (Assembler Agent)                        │
│     Input: BOM, symbols, block diagram, connections             │
│     Output: .kicad_sch files with real symbols                  │
└─────────────────────────────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────────────────────────┐
│  5. MAPO Vision Validation Loop                                 │
│     ┌──────────────────────────────────────────────────────┐   │
│     │  Render → Validate → Fix → Repeat until passed       │   │
│     │  5 Expert Personas, Multi-model verification         │   │
│     └──────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────────────────────────┐
│  6. Output                                                      │
│     - Validated .kicad_sch files                               │
│     - Validation report with scores                             │
│     - Symbol library for project                                │
└─────────────────────────────────────────────────────────────────┘
```

### 5.2 API Endpoints

```typescript
// services/nexus-ee-design/src/api/routes.ts

// POST /api/v1/schematic/generate
// Generate schematic with full MAPO validation
interface GenerateSchematicRequest {
  project_id: string;
  design_intent: string;
  bom?: BOMItem[];              // Optional pre-defined BOM
  reference_design_id?: string; // Optional reference to base from
}

// POST /api/v1/schematic/validate
// Validate existing schematic with vision
interface ValidateSchematicRequest {
  project_id: string;
  schematic_id: string;
  design_intent: string;
}

// POST /api/v1/symbols/fetch
// Fetch symbols for given part numbers
interface FetchSymbolsRequest {
  part_numbers: string[];
  manufacturer_hints?: Record<string, string>;
}

// GET /api/v1/symbols/search
// Search GraphRAG for symbols
interface SearchSymbolsRequest {
  query: string;
  category?: string;
  limit?: number;
}
```

---

## Implementation Priority

| Phase | Component | Priority | Effort | Dependencies |
|-------|-----------|----------|--------|--------------|
| 1 | Symbol Fetcher Agent | HIGH | 2 days | GraphRAG client |
| 2 | GraphRAG Symbol Indexer | HIGH | 1 day | Neo4j setup |
| 3 | Schematic Assembler | HIGH | 3 days | Symbol Fetcher |
| 4 | Vision Validator | HIGH | 2 days | Anthropic API |
| 5 | MAPO Loop | MEDIUM | 1 day | All above |
| 6 | KiCanvas Renderer | MEDIUM | 1 day | Puppeteer |
| 7 | API Integration | MEDIUM | 1 day | All above |

**Total Estimated Effort**: ~11 days

---

## Quality Metrics

### Validation Pass Criteria

| Metric | Threshold | Weight |
|--------|-----------|--------|
| Circuit Topology | ≥ 80% | 30% |
| Component Validation | ≥ 85% | 25% |
| Connection Verification | ≥ 90% | 25% |
| Manufacturability | ≥ 70% | 10% |
| Reference Comparison | ≥ 75% | 10% |
| **Overall** | **≥ 85%** | - |
| **Critical Issues** | **= 0** | - |

### Symbol Quality Criteria

| Source | Trust Level | Auto-Approve |
|--------|-------------|--------------|
| KiCad Official | HIGH | Yes |
| SnapEDA Verified | HIGH | Yes |
| Manufacturer | MEDIUM | After validation |
| LLM Generated | LOW | Never (manual review) |

---

## References

1. [KiCad HTTP Libraries Specification](https://dev-docs.kicad.org/en/apis-and-binding/http-libraries/)
2. [SnapEDA API Documentation](https://www.snapeda.com/api/)
3. [Ultra Librarian Integration](https://www.ultralibrarian.com/developers)
4. [KiCad S-Expression Format](https://dev-docs.kicad.org/en/file-formats/sexpr-intro/)
5. [MAPO Research Paper](https://arxiv.org/abs/2408.10566) - Gaming AI for PCB
6. [Claude Vision API](https://docs.anthropic.com/en/docs/build-with-claude/vision)
