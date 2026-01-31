"""
GraphRAG Symbol Indexer - Index KiCad symbols in knowledge graph.

Provides semantic search and relationship discovery for electronic component symbols.
Integrates with Neo4j for knowledge graph and vector similarity search.

Author: Nexus EE Design Team
"""

import asyncio
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path

logger = logging.getLogger(__name__)

# Type definitions for symbol documents
@dataclass
class KiCadSymbolDocument:
    """Document schema for indexed KiCad symbols."""
    id: str
    part_number: str
    manufacturer: Optional[str] = None
    category: str = "Other"
    subcategory: Optional[str] = None
    package: Optional[str] = None
    symbol_sexp: str = ""
    footprint_sexp: Optional[str] = None
    description: str = ""
    keywords: List[str] = field(default_factory=list)
    specifications: Dict[str, Any] = field(default_factory=dict)
    source: str = "local_cache"
    datasheet_url: Optional[str] = None
    source_url: Optional[str] = None
    fetched_at: Optional[str] = None
    verified: bool = False
    usage_count: int = 0
    last_used_at: Optional[str] = None
    embedding: Optional[List[float]] = None

    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "part_number": self.part_number,
            "manufacturer": self.manufacturer,
            "category": self.category,
            "subcategory": self.subcategory,
            "package": self.package,
            "symbol_preview": self.symbol_sexp[:1000] if self.symbol_sexp else "",
            "has_footprint": bool(self.footprint_sexp),
            "description": self.description,
            "keywords": self.keywords,
            "specifications": self.specifications,
            "source": self.source,
            "datasheet_url": self.datasheet_url,
            "verified": self.verified,
            "usage_count": self.usage_count,
        }


class SymbolGraphRAGIndexer:
    """
    Index KiCad symbols in GraphRAG for semantic search and relationship discovery.

    Uses Neo4j for knowledge graph storage with vector embeddings for semantic search.
    Creates relationships between:
    - Symbols and Manufacturers
    - Symbols and Categories
    - Symbols and Packages
    - Similar symbols (by category, function, or embedding similarity)
    """

    SYMBOL_EMBEDDING_INDEX = "symbol_embeddings"
    SYMBOL_NODE_LABEL = "Symbol"

    def __init__(
        self,
        neo4j_uri: str = "bolt://localhost:7687",
        neo4j_user: str = "neo4j",
        neo4j_password: str = "",
        embedding_model: Optional[Any] = None,
        embedding_dimension: int = 1536
    ):
        """
        Initialize the indexer.

        Args:
            neo4j_uri: Neo4j connection URI
            neo4j_user: Neo4j username
            neo4j_password: Neo4j password
            embedding_model: Model for generating embeddings (OpenAI, Anthropic, etc.)
            embedding_dimension: Dimension of embedding vectors
        """
        self.neo4j_uri = neo4j_uri
        self.neo4j_user = neo4j_user
        self.neo4j_password = neo4j_password
        self.embedding_model = embedding_model
        self.embedding_dimension = embedding_dimension

        self._driver = None

    async def connect(self):
        """Establish connection to Neo4j."""
        try:
            from neo4j import AsyncGraphDatabase
            self._driver = AsyncGraphDatabase.driver(
                self.neo4j_uri,
                auth=(self.neo4j_user, self.neo4j_password)
            )
            # Verify connection
            async with self._driver.session() as session:
                await session.run("RETURN 1")
            logger.info("Connected to Neo4j")

            # Ensure indexes exist
            await self._ensure_indexes()

        except ImportError:
            logger.warning("neo4j package not installed, using mock driver")
            self._driver = MockNeo4jDriver()
        except Exception as e:
            logger.error(f"Failed to connect to Neo4j: {e}")
            self._driver = MockNeo4jDriver()

    async def _ensure_indexes(self):
        """Create necessary indexes and constraints."""
        if isinstance(self._driver, MockNeo4jDriver):
            return

        async with self._driver.session() as session:
            # Create constraints
            constraints = [
                "CREATE CONSTRAINT symbol_part_number IF NOT EXISTS FOR (s:Symbol) REQUIRE s.part_number IS UNIQUE",
                "CREATE CONSTRAINT manufacturer_name IF NOT EXISTS FOR (m:Manufacturer) REQUIRE m.name IS UNIQUE",
                "CREATE CONSTRAINT category_name IF NOT EXISTS FOR (c:Category) REQUIRE c.name IS UNIQUE",
                "CREATE CONSTRAINT package_name IF NOT EXISTS FOR (p:Package) REQUIRE p.name IS UNIQUE",
            ]

            for constraint in constraints:
                try:
                    await session.run(constraint)
                except Exception as e:
                    logger.debug(f"Constraint may already exist: {e}")

            # Create vector index for semantic search
            try:
                await session.run(f"""
                    CREATE VECTOR INDEX {self.SYMBOL_EMBEDDING_INDEX} IF NOT EXISTS
                    FOR (s:Symbol)
                    ON s.embedding
                    OPTIONS {{
                        indexConfig: {{
                            `vector.dimensions`: {self.embedding_dimension},
                            `vector.similarity_function`: 'cosine'
                        }}
                    }}
                """)
            except Exception as e:
                logger.debug(f"Vector index creation: {e}")

    async def index_symbol(self, symbol: KiCadSymbolDocument) -> bool:
        """
        Index a symbol in the knowledge graph.

        Args:
            symbol: Symbol document to index

        Returns:
            True if indexed successfully
        """
        # Generate embedding
        embedding = await self._generate_embedding(symbol)
        symbol.embedding = embedding

        # Insert into Neo4j
        return await self._insert_symbol_node(symbol)

    async def _generate_embedding(self, symbol: KiCadSymbolDocument) -> List[float]:
        """Generate embedding vector for symbol."""
        embedding_text = self._build_embedding_text(symbol)

        if self.embedding_model:
            try:
                # Use provided embedding model
                embedding = await self.embedding_model.embed(embedding_text)
                return embedding.tolist() if hasattr(embedding, 'tolist') else list(embedding)
            except Exception as e:
                logger.warning(f"Embedding generation failed: {e}")

        # Fallback: try OpenAI
        try:
            import openai
            client = openai.OpenAI()
            response = client.embeddings.create(
                model="text-embedding-3-small",
                input=embedding_text
            )
            return response.data[0].embedding
        except Exception as e:
            logger.warning(f"OpenAI embedding failed: {e}")

        # Last resort: zero vector (will not work for similarity search)
        return [0.0] * self.embedding_dimension

    def _build_embedding_text(self, symbol: KiCadSymbolDocument) -> str:
        """Build text for embedding generation."""
        parts = [
            symbol.part_number,
            symbol.manufacturer or "",
            symbol.category,
            symbol.subcategory or "",
            symbol.description,
            " ".join(symbol.keywords),
            f"Package: {symbol.package}" if symbol.package else "",
        ]

        # Add specifications
        for key, value in symbol.specifications.items():
            parts.append(f"{key}: {value}")

        return " ".join(filter(None, parts))

    async def _insert_symbol_node(self, symbol: KiCadSymbolDocument) -> bool:
        """Insert symbol and relationships into Neo4j."""
        if isinstance(self._driver, MockNeo4jDriver):
            logger.info(f"Mock: Would index symbol {symbol.part_number}")
            return True

        async with self._driver.session() as session:
            try:
                # Merge symbol node with all properties
                result = await session.run("""
                    MERGE (s:Symbol {part_number: $part_number})
                    SET s += {
                        id: $id,
                        manufacturer: $manufacturer,
                        category: $category,
                        subcategory: $subcategory,
                        package: $package,
                        description: $description,
                        keywords: $keywords,
                        source: $source,
                        datasheet_url: $datasheet_url,
                        verified: $verified,
                        usage_count: $usage_count,
                        embedding: $embedding,
                        indexed_at: $indexed_at
                    }

                    // Create manufacturer relationship
                    WITH s
                    WHERE $manufacturer IS NOT NULL
                    MERGE (m:Manufacturer {name: $manufacturer})
                    MERGE (s)-[:MADE_BY]->(m)

                    // Create category relationship
                    WITH s
                    MERGE (c:Category {name: $category})
                    MERGE (s)-[:IN_CATEGORY]->(c)

                    // Create package relationship
                    WITH s
                    WHERE $package IS NOT NULL
                    MERGE (p:Package {name: $package})
                    MERGE (s)-[:HAS_PACKAGE]->(p)

                    RETURN s.part_number as part_number
                """, {
                    "id": symbol.id,
                    "part_number": symbol.part_number,
                    "manufacturer": symbol.manufacturer,
                    "category": symbol.category,
                    "subcategory": symbol.subcategory,
                    "package": symbol.package,
                    "description": symbol.description,
                    "keywords": symbol.keywords,
                    "source": symbol.source,
                    "datasheet_url": symbol.datasheet_url,
                    "verified": symbol.verified,
                    "usage_count": symbol.usage_count,
                    "embedding": symbol.embedding,
                    "indexed_at": datetime.utcnow().isoformat()
                })

                record = await result.single()
                logger.info(f"Indexed symbol: {record['part_number']}")
                return True

            except Exception as e:
                logger.error(f"Failed to index symbol {symbol.part_number}: {e}")
                return False

    async def search_symbols(
        self,
        query: str,
        category: Optional[str] = None,
        manufacturer: Optional[str] = None,
        top_k: int = 10,
        similarity_threshold: float = 0.7
    ) -> List[KiCadSymbolDocument]:
        """
        Semantic search for symbols.

        Args:
            query: Search query
            category: Optional category filter
            manufacturer: Optional manufacturer filter
            top_k: Maximum results to return
            similarity_threshold: Minimum similarity score

        Returns:
            List of matching symbol documents
        """
        # Generate query embedding
        query_embedding = await self._generate_embedding(
            KiCadSymbolDocument(
                id="query",
                part_number=query,
                description=query
            )
        )

        if isinstance(self._driver, MockNeo4jDriver):
            logger.info(f"Mock: Would search for '{query}'")
            return []

        async with self._driver.session() as session:
            # Build Cypher query with optional filters
            where_clauses = []
            params = {
                "embedding": query_embedding,
                "top_k": top_k,
                "threshold": similarity_threshold
            }

            if category:
                where_clauses.append("node.category = $category")
                params["category"] = category

            if manufacturer:
                where_clauses.append("node.manufacturer = $manufacturer")
                params["manufacturer"] = manufacturer

            where_clause = f"WHERE {' AND '.join(where_clauses)}" if where_clauses else ""

            cypher = f"""
                CALL db.index.vector.queryNodes('{self.SYMBOL_EMBEDDING_INDEX}', $top_k, $embedding)
                YIELD node, score
                {where_clause}
                {"AND " if where_clause else "WHERE "}score > $threshold
                RETURN node, score
                ORDER BY score DESC
            """

            try:
                result = await session.run(cypher, params)
                records = await result.data()

                symbols = []
                for record in records:
                    node = record["node"]
                    symbols.append(KiCadSymbolDocument(
                        id=node.get("id", ""),
                        part_number=node.get("part_number", ""),
                        manufacturer=node.get("manufacturer"),
                        category=node.get("category", "Other"),
                        subcategory=node.get("subcategory"),
                        package=node.get("package"),
                        description=node.get("description", ""),
                        keywords=node.get("keywords", []),
                        source=node.get("source", "local_cache"),
                        datasheet_url=node.get("datasheet_url"),
                        verified=node.get("verified", False),
                        usage_count=node.get("usage_count", 0)
                    ))

                return symbols

            except Exception as e:
                logger.error(f"Search failed: {e}")
                return []

    async def find_similar_symbols(
        self,
        part_number: str,
        limit: int = 5
    ) -> List[Tuple[KiCadSymbolDocument, float]]:
        """
        Find symbols similar to a given part.

        Args:
            part_number: Reference part number
            limit: Maximum results

        Returns:
            List of (symbol, similarity_score) tuples
        """
        if isinstance(self._driver, MockNeo4jDriver):
            return []

        async with self._driver.session() as session:
            try:
                # Get the reference symbol's embedding
                result = await session.run("""
                    MATCH (s:Symbol {part_number: $part_number})
                    RETURN s.embedding as embedding, s.category as category, s.package as package
                """, {"part_number": part_number})

                record = await result.single()
                if not record or not record["embedding"]:
                    return []

                # Find similar symbols
                result = await session.run(f"""
                    MATCH (ref:Symbol {{part_number: $part_number}})
                    CALL db.index.vector.queryNodes('{self.SYMBOL_EMBEDDING_INDEX}', $limit + 1, ref.embedding)
                    YIELD node, score
                    WHERE node.part_number <> $part_number
                    RETURN node, score
                    ORDER BY score DESC
                    LIMIT $limit
                """, {"part_number": part_number, "limit": limit})

                records = await result.data()

                symbols = []
                for r in records:
                    node = r["node"]
                    score = r["score"]
                    symbols.append((
                        KiCadSymbolDocument(
                            id=node.get("id", ""),
                            part_number=node.get("part_number", ""),
                            manufacturer=node.get("manufacturer"),
                            category=node.get("category", "Other"),
                            package=node.get("package"),
                            description=node.get("description", ""),
                        ),
                        score
                    ))

                return symbols

            except Exception as e:
                logger.error(f"Find similar failed: {e}")
                return []

    async def find_pin_compatible(
        self,
        part_number: str,
        limit: int = 5
    ) -> List[KiCadSymbolDocument]:
        """
        Find pin-compatible alternatives for a part.

        Considers same package and similar function.

        Args:
            part_number: Reference part number
            limit: Maximum results

        Returns:
            List of pin-compatible symbol documents
        """
        if isinstance(self._driver, MockNeo4jDriver):
            return []

        async with self._driver.session() as session:
            try:
                result = await session.run("""
                    MATCH (s:Symbol {part_number: $part_number})
                    MATCH (alt:Symbol)-[:HAS_PACKAGE]->(p:Package)<-[:HAS_PACKAGE]-(s)
                    WHERE alt.part_number <> $part_number
                      AND alt.category = s.category
                    WITH alt, s,
                         gds.similarity.cosine(s.embedding, alt.embedding) as similarity
                    ORDER BY similarity DESC
                    LIMIT $limit
                    RETURN alt, similarity
                """, {"part_number": part_number, "limit": limit})

                records = await result.data()

                symbols = []
                for r in records:
                    node = r["alt"]
                    symbols.append(KiCadSymbolDocument(
                        id=node.get("id", ""),
                        part_number=node.get("part_number", ""),
                        manufacturer=node.get("manufacturer"),
                        category=node.get("category", "Other"),
                        package=node.get("package"),
                        description=node.get("description", ""),
                    ))

                return symbols

            except Exception as e:
                logger.error(f"Find pin compatible failed: {e}")
                return []

    async def get_category_statistics(self) -> Dict[str, int]:
        """Get count of symbols per category."""
        if isinstance(self._driver, MockNeo4jDriver):
            return {}

        async with self._driver.session() as session:
            try:
                result = await session.run("""
                    MATCH (s:Symbol)-[:IN_CATEGORY]->(c:Category)
                    RETURN c.name as category, count(s) as count
                    ORDER BY count DESC
                """)

                records = await result.data()
                return {r["category"]: r["count"] for r in records}

            except Exception as e:
                logger.error(f"Get statistics failed: {e}")
                return {}

    async def update_usage_count(self, part_number: str):
        """Increment usage count for a symbol."""
        if isinstance(self._driver, MockNeo4jDriver):
            return

        async with self._driver.session() as session:
            try:
                await session.run("""
                    MATCH (s:Symbol {part_number: $part_number})
                    SET s.usage_count = coalesce(s.usage_count, 0) + 1,
                        s.last_used_at = $timestamp
                """, {
                    "part_number": part_number,
                    "timestamp": datetime.utcnow().isoformat()
                })
            except Exception as e:
                logger.warning(f"Update usage count failed: {e}")

    async def mark_verified(self, part_number: str, verified: bool = True):
        """Mark a symbol as verified/unverified."""
        if isinstance(self._driver, MockNeo4jDriver):
            return

        async with self._driver.session() as session:
            try:
                await session.run("""
                    MATCH (s:Symbol {part_number: $part_number})
                    SET s.verified = $verified,
                        s.verified_at = $timestamp
                """, {
                    "part_number": part_number,
                    "verified": verified,
                    "timestamp": datetime.utcnow().isoformat()
                })
            except Exception as e:
                logger.warning(f"Mark verified failed: {e}")

    async def close(self):
        """Close Neo4j connection."""
        if self._driver and not isinstance(self._driver, MockNeo4jDriver):
            await self._driver.close()


class MockNeo4jDriver:
    """Mock driver for testing without Neo4j."""

    async def close(self):
        pass

    def session(self):
        return MockSession()


class MockSession:
    """Mock session for testing."""

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        pass

    async def run(self, query, params=None):
        return MockResult()


class MockResult:
    """Mock result for testing."""

    async def single(self):
        return None

    async def data(self):
        return []


# Convenience functions for integration
async def create_indexer(
    neo4j_uri: Optional[str] = None,
    neo4j_user: Optional[str] = None,
    neo4j_password: Optional[str] = None
) -> SymbolGraphRAGIndexer:
    """
    Create and connect an indexer with default settings.

    Reads connection details from environment variables if not provided.
    """
    import os

    indexer = SymbolGraphRAGIndexer(
        neo4j_uri=neo4j_uri or os.getenv("NEO4J_URI", "bolt://localhost:7687"),
        neo4j_user=neo4j_user or os.getenv("NEO4J_USER", "neo4j"),
        neo4j_password=neo4j_password or os.getenv("NEO4J_PASSWORD", "")
    )

    await indexer.connect()
    return indexer


# CLI entry point
if __name__ == "__main__":
    import sys
    import uuid

    async def main():
        indexer = await create_indexer()

        if len(sys.argv) < 2:
            print("Usage:")
            print("  python symbol_indexer.py index <part_number> <category>")
            print("  python symbol_indexer.py search <query>")
            print("  python symbol_indexer.py stats")
            sys.exit(1)

        command = sys.argv[1]

        try:
            if command == "index":
                if len(sys.argv) < 4:
                    print("Usage: python symbol_indexer.py index <part_number> <category>")
                    sys.exit(1)

                part_number = sys.argv[2]
                category = sys.argv[3]

                symbol = KiCadSymbolDocument(
                    id=str(uuid.uuid4()),
                    part_number=part_number,
                    category=category,
                    description=f"Test symbol for {part_number}"
                )

                success = await indexer.index_symbol(symbol)
                print(f"Indexed: {success}")

            elif command == "search":
                if len(sys.argv) < 3:
                    print("Usage: python symbol_indexer.py search <query>")
                    sys.exit(1)

                query = sys.argv[2]
                results = await indexer.search_symbols(query)

                print(f"\nFound {len(results)} results for '{query}':")
                for symbol in results:
                    print(f"  - {symbol.part_number} ({symbol.category})")
                    if symbol.manufacturer:
                        print(f"    Manufacturer: {symbol.manufacturer}")
                    if symbol.description:
                        print(f"    Description: {symbol.description}")

            elif command == "stats":
                stats = await indexer.get_category_statistics()
                print("\nSymbol Statistics by Category:")
                for category, count in stats.items():
                    print(f"  {category}: {count}")

            else:
                print(f"Unknown command: {command}")
                sys.exit(1)

        finally:
            await indexer.close()

    asyncio.run(main())
