"""
MAPO v2.1 Schematic - Wiring Memory Client

Provides GraphRAG-powered memory for connection pattern learning.
Learns which wiring patterns work for which component combinations,
enabling smarter connection generation over time.

Author: Nexus EE Design Team
"""

import asyncio
import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import httpx

logger = logging.getLogger(__name__)


@dataclass
class ConnectionPattern:
    """
    Learned connection pattern from nexus-memory.
    
    Represents a successful wiring pattern between component
    categories, verified by smoke test simulation.
    """
    # Pattern identity
    design_type: str  # e.g., "foc_esc", "power_supply", "mcu_board"
    component_categories: Tuple[str, ...]  # e.g., ("MCU", "Gate_Driver", "MOSFET")
    
    # Connection template
    connections: List[Dict[str, str]] = field(default_factory=list)
    # Each connection: {from_category, from_pin_type, to_category, to_pin_type, net_name}
    
    # Quality metrics
    smoke_test_passed: bool = False
    validation_score: float = 0.0
    success_count: int = 1
    last_success: str = ""
    
    # Design context
    design_intent: str = ""
    notes: str = ""
    
    @classmethod
    def from_memory(cls, memory_data: Dict[str, Any]) -> Optional["ConnectionPattern"]:
        """Parse ConnectionPattern from nexus-memory recall result."""
        try:
            content = memory_data.get("content", "")
            if isinstance(content, str):
                try:
                    data = json.loads(content)
                except json.JSONDecodeError:
                    return None
            else:
                data = content
            
            if data.get("type") != "wiring_pattern":
                return None
            
            return cls(
                design_type=data.get("design_type", ""),
                component_categories=tuple(data.get("component_categories", [])),
                connections=data.get("connections", []),
                smoke_test_passed=data.get("validated", False),
                validation_score=data.get("validation_score", 0.0),
                success_count=data.get("success_count", 1),
                last_success=data.get("last_success", ""),
                design_intent=data.get("design_intent", ""),
                notes=data.get("notes", ""),
            )
        except Exception as e:
            logger.debug(f"Failed to parse wiring pattern: {e}")
            return None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "type": "wiring_pattern",
            "design_type": self.design_type,
            "component_categories": list(self.component_categories),
            "connections": self.connections,
            "validated": self.smoke_test_passed,
            "validation_score": self.validation_score,
            "success_count": self.success_count,
            "last_success": self.last_success,
            "design_intent": self.design_intent,
            "notes": self.notes,
        }
    
    def matches_categories(self, categories: List[str]) -> float:
        """
        Score how well this pattern matches given categories.
        
        Returns:
            Overlap score (0-1)
        """
        pattern_cats = set(self.component_categories)
        query_cats = set(categories)
        
        if not pattern_cats:
            return 0.0
        
        intersection = pattern_cats & query_cats
        union = pattern_cats | query_cats
        
        return len(intersection) / len(union) if union else 0.0


class WiringMemoryClient:
    """
    Client for nexus-memory wiring pattern learning.
    
    Provides:
    - Recall: Query for successful wiring patterns
    - Store: Save new successful patterns after smoke test
    - Learn: Build up patterns from successful designs
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        api_url: Optional[str] = None,
        timeout: float = 30.0,
    ):
        self.api_key = api_key or os.environ.get("NEXUS_API_KEY", "")
        self.api_url = api_url or os.environ.get("NEXUS_API_URL", "https://api.adverant.ai")
        self.timeout = timeout
        self._client: Optional[httpx.AsyncClient] = None
    
    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client."""
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(self.timeout),
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                }
            )
        return self._client
    
    async def close(self):
        """Close the HTTP client."""
        if self._client and not self._client.is_closed:
            await self._client.aclose()
            self._client = None
    
    async def recall_connection_patterns(
        self,
        component_categories: List[str],
        design_type: Optional[str] = None,
        limit: int = 5,
    ) -> List[ConnectionPattern]:
        """
        Query nexus-memory for typical wiring patterns.
        
        Args:
            component_categories: Categories to find patterns for
            design_type: Optional design type filter
            limit: Maximum patterns to return
        
        Returns:
            List of ConnectionPattern sorted by relevance
        """
        if not self.api_key:
            logger.debug("No API key, skipping pattern recall")
            return []
        
        try:
            client = await self._get_client()
            
            # Build semantic search query
            query_parts = [
                "wiring_pattern",
                "connection_pattern",
                "schematic wiring",
            ]
            if design_type:
                query_parts.append(design_type)
            query_parts.extend(component_categories)
            
            query = " ".join(query_parts)
            
            response = await client.post(
                f"{self.api_url}/api/memory/recall",
                json={
                    "query": query,
                    "limit": limit * 2,  # Over-fetch for filtering
                    "filters": {
                        "event_type": "wiring_pattern",
                    }
                }
            )
            
            if response.status_code != 200:
                logger.warning(f"Pattern recall failed: {response.status_code}")
                return []
            
            result = response.json()
            memories = result.get("memories", [])
            
            # Parse and filter patterns
            patterns = []
            for memory in memories:
                pattern = ConnectionPattern.from_memory(memory)
                if pattern:
                    # Score relevance
                    score = pattern.matches_categories(component_categories)
                    if score > 0.3:  # Minimum relevance threshold
                        patterns.append((score, pattern))
            
            # Sort by relevance and smoke test status
            patterns.sort(key=lambda x: (x[1].smoke_test_passed, x[0]), reverse=True)
            
            result_patterns = [p for _, p in patterns[:limit]]
            logger.info(
                f"Recalled {len(result_patterns)} wiring patterns for {component_categories}"
            )
            return result_patterns
            
        except Exception as e:
            logger.warning(f"Pattern recall error: {e}")
            return []
    
    async def store_successful_wiring(
        self,
        connections: List[Dict[str, Any]],
        component_categories: List[str],
        design_type: str,
        design_intent: str = "",
        smoke_test_passed: bool = True,
        validation_score: float = 0.0,
    ) -> bool:
        """
        Store connections that passed smoke test.
        
        Only stores patterns where smoke_test_passed=True,
        ensuring we only learn from validated designs.
        
        Args:
            connections: List of connection dicts
            component_categories: Categories involved
            design_type: Type of design
            design_intent: Design description
            smoke_test_passed: Whether smoke test passed
            validation_score: Overall validation score
        
        Returns:
            True if stored successfully
        """
        if not self.api_key:
            logger.debug("No API key, skipping pattern store")
            return False
        
        if not smoke_test_passed:
            logger.debug("Smoke test failed, not storing pattern")
            return False
        
        try:
            client = await self._get_client()
            
            # Convert connections to pattern format
            pattern_connections = []
            for conn in connections:
                pattern_connections.append({
                    "from_ref": conn.get("from_ref", ""),
                    "from_pin": conn.get("from_pin", ""),
                    "to_ref": conn.get("to_ref", ""),
                    "to_pin": conn.get("to_pin", ""),
                    "net_name": conn.get("net_name", ""),
                    "connection_type": conn.get("connection_type", "signal"),
                })
            
            pattern = ConnectionPattern(
                design_type=design_type,
                component_categories=tuple(sorted(set(component_categories))),
                connections=pattern_connections,
                smoke_test_passed=True,
                validation_score=validation_score,
                success_count=1,
                last_success=datetime.now().isoformat(),
                design_intent=design_intent,
            )
            
            memory_content = json.dumps(pattern.to_dict())
            
            response = await client.post(
                f"{self.api_url}/api/memory/store",
                json={
                    "content": memory_content,
                    "event_type": "wiring_pattern",
                    "metadata": {
                        "design_type": design_type,
                        "component_count": len(component_categories),
                        "connection_count": len(pattern_connections),
                        "validated": smoke_test_passed,
                    }
                }
            )
            
            if response.status_code == 200:
                logger.info(
                    f"Stored wiring pattern: {design_type} with {len(pattern_connections)} connections"
                )
                return True
            else:
                logger.warning(f"Pattern store failed: {response.status_code}")
                return False
                
        except Exception as e:
            logger.warning(f"Pattern store error: {e}")
            return False
    
    async def recall_pin_mapping(
        self,
        from_category: str,
        to_category: str,
    ) -> List[Dict[str, str]]:
        """
        Recall common pin mappings between two component categories.
        
        Args:
            from_category: Source component category
            to_category: Target component category
        
        Returns:
            List of pin mapping dicts: {from_pin_type, to_pin_type, description}
        """
        if not self.api_key:
            return []
        
        try:
            client = await self._get_client()
            
            query = f"pin mapping {from_category} {to_category} connection"
            
            response = await client.post(
                f"{self.api_url}/api/memory/recall",
                json={
                    "query": query,
                    "limit": 10,
                }
            )
            
            if response.status_code != 200:
                return []
            
            result = response.json()
            memories = result.get("memories", [])
            
            # Extract pin mappings from patterns
            mappings = []
            for memory in memories:
                pattern = ConnectionPattern.from_memory(memory)
                if pattern and pattern.smoke_test_passed:
                    for conn in pattern.connections:
                        mappings.append({
                            "from_pin": conn.get("from_pin", ""),
                            "to_pin": conn.get("to_pin", ""),
                            "net_name": conn.get("net_name", ""),
                        })
            
            return mappings
            
        except Exception as e:
            logger.warning(f"Pin mapping recall error: {e}")
            return []
    
    async def get_design_type_patterns(
        self,
        design_type: str,
    ) -> List[ConnectionPattern]:
        """
        Get all patterns for a specific design type.
        
        Useful for understanding common wiring for design types
        like "foc_esc", "power_supply", "sensor_board".
        """
        if not self.api_key:
            return []
        
        try:
            client = await self._get_client()
            
            response = await client.post(
                f"{self.api_url}/api/memory/recall",
                json={
                    "query": f"wiring_pattern {design_type} schematic connections",
                    "limit": 20,
                    "filters": {
                        "event_type": "wiring_pattern",
                    }
                }
            )
            
            if response.status_code != 200:
                return []
            
            result = response.json()
            memories = result.get("memories", [])
            
            patterns = []
            for memory in memories:
                pattern = ConnectionPattern.from_memory(memory)
                if pattern and pattern.design_type == design_type:
                    patterns.append(pattern)
            
            return patterns
            
        except Exception as e:
            logger.warning(f"Design type patterns error: {e}")
            return []
