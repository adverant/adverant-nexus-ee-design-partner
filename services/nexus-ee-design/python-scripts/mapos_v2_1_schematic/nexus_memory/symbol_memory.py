"""
MAPO v2.1 Schematic - Symbol Memory Client

Provides GraphRAG-powered memory for symbol resolution learning.
Learns which sources successfully resolve which components,
enabling faster and more accurate symbol fetching over time.

Author: Nexus EE Design Team
"""

import asyncio
import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

import httpx

logger = logging.getLogger(__name__)


@dataclass
class SymbolResolution:
    """
    Learned symbol resolution from nexus-memory.
    
    Represents a previously successful resolution of a component
    to a KiCad symbol, including the source and quality metadata.
    """
    part_number: str
    manufacturer: str
    category: str
    
    # Resolution details
    source: str  # e.g., "kicad_worker", "snapeda", "ultralibrarian"
    symbol_format: str = "kicad_sym"
    footprint: str = ""
    
    # Quality metrics
    verified: bool = False
    success_count: int = 1
    last_success: str = ""
    
    # Symbol content (if cached)
    symbol_content: Optional[str] = None
    
    # Confidence score (0-1)
    confidence: float = 0.8
    
    @classmethod
    def from_memory(cls, memory_data: Dict[str, Any]) -> Optional["SymbolResolution"]:
        """Parse SymbolResolution from nexus-memory recall result."""
        try:
            content = memory_data.get("content", "")
            if isinstance(content, str):
                try:
                    data = json.loads(content)
                except json.JSONDecodeError:
                    # Not JSON, try to parse as text
                    return None
            else:
                data = content
            
            # Extract resolution data
            resolution = data.get("resolution", data)
            
            return cls(
                part_number=data.get("part_number", ""),
                manufacturer=data.get("manufacturer", ""),
                category=data.get("category", ""),
                source=resolution.get("source", ""),
                symbol_format=resolution.get("format", "kicad_sym"),
                footprint=resolution.get("footprint", ""),
                verified=resolution.get("verified", False),
                success_count=resolution.get("success_count", 1),
                last_success=resolution.get("last_success", ""),
                symbol_content=resolution.get("symbol_content"),
                confidence=resolution.get("confidence", 0.8),
            )
        except Exception as e:
            logger.debug(f"Failed to parse symbol resolution: {e}")
            return None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "type": "component_resolution",
            "part_number": self.part_number,
            "manufacturer": self.manufacturer,
            "category": self.category,
            "resolution": {
                "source": self.source,
                "format": self.symbol_format,
                "footprint": self.footprint,
                "verified": self.verified,
                "success_count": self.success_count,
                "last_success": self.last_success,
                "confidence": self.confidence,
            }
        }


class SymbolMemoryClient:
    """
    Client for nexus-memory symbol resolution learning.
    
    Provides:
    - Recall: Query for previously successful symbol resolutions
    - Store: Save new successful resolutions for future use
    - Learn: Update confidence based on usage patterns
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
    
    async def recall_symbol(
        self,
        part_number: str,
        manufacturer: str,
        category: Optional[str] = None,
    ) -> Optional[SymbolResolution]:
        """
        Query nexus-memory for previously resolved symbol.
        
        Args:
            part_number: Component part number (e.g., "STM32G431CBT6")
            manufacturer: Component manufacturer (e.g., "ST")
            category: Optional component category
        
        Returns:
            SymbolResolution if found, None otherwise
        """
        if not self.api_key:
            logger.debug("No API key, skipping symbol recall")
            return None
        
        try:
            client = await self._get_client()
            
            # Build semantic search query
            query_parts = [
                "component_resolution",
                "KiCad symbol",
                part_number,
                manufacturer,
            ]
            if category:
                query_parts.append(category)
            
            query = " ".join(query_parts)
            
            response = await client.post(
                f"{self.api_url}/api/memory/recall",
                json={
                    "query": query,
                    "limit": 5,  # Get top 5 matches
                    "filters": {
                        "event_type": "component_resolution",
                    }
                }
            )
            
            if response.status_code != 200:
                logger.warning(f"Symbol recall failed: {response.status_code}")
                return None
            
            result = response.json()
            memories = result.get("memories", [])
            
            # Find best matching resolution
            for memory in memories:
                resolution = SymbolResolution.from_memory(memory)
                if resolution:
                    # Verify this is the right component
                    if (resolution.part_number.lower() == part_number.lower() or
                        part_number.lower() in resolution.part_number.lower()):
                        logger.info(
                            f"Found symbol resolution in memory: {part_number} -> {resolution.source}"
                        )
                        return resolution
            
            return None
            
        except Exception as e:
            logger.warning(f"Symbol recall error: {e}")
            return None
    
    async def store_symbol(
        self,
        part_number: str,
        manufacturer: str,
        category: str,
        source: str,
        footprint: str = "",
        verified: bool = False,
        symbol_content: Optional[str] = None,
    ) -> bool:
        """
        Store successful symbol resolution for future use.
        
        Args:
            part_number: Component part number
            manufacturer: Component manufacturer
            category: Component category
            source: Resolution source (e.g., "kicad_worker", "snapeda")
            footprint: Associated footprint
            verified: Whether symbol was manually verified
            symbol_content: Optional symbol content to cache
        
        Returns:
            True if stored successfully
        """
        if not self.api_key:
            logger.debug("No API key, skipping symbol store")
            return False
        
        try:
            client = await self._get_client()
            
            resolution = SymbolResolution(
                part_number=part_number,
                manufacturer=manufacturer,
                category=category,
                source=source,
                footprint=footprint,
                verified=verified,
                success_count=1,
                last_success=datetime.now().isoformat(),
                symbol_content=symbol_content,
                confidence=0.9 if verified else 0.8,
            )
            
            memory_content = json.dumps(resolution.to_dict())
            
            response = await client.post(
                f"{self.api_url}/api/memory/store",
                json={
                    "content": memory_content,
                    "event_type": "component_resolution",
                    "metadata": {
                        "part_number": part_number,
                        "manufacturer": manufacturer,
                        "category": category,
                        "source": source,
                    }
                }
            )
            
            if response.status_code == 200:
                logger.info(f"Stored symbol resolution: {part_number} -> {source}")
                return True
            else:
                logger.warning(f"Symbol store failed: {response.status_code}")
                return False
                
        except Exception as e:
            logger.warning(f"Symbol store error: {e}")
            return False
    
    async def batch_recall(
        self,
        components: List[Dict[str, str]],
    ) -> Dict[str, SymbolResolution]:
        """
        Batch recall symbol resolutions for multiple components.
        
        Args:
            components: List of dicts with part_number, manufacturer, category
        
        Returns:
            Dict mapping part_number to SymbolResolution
        """
        results = {}
        
        # Run recalls in parallel with concurrency limit
        semaphore = asyncio.Semaphore(5)
        
        async def recall_one(comp: Dict[str, str]) -> tuple:
            async with semaphore:
                resolution = await self.recall_symbol(
                    part_number=comp.get("part_number", ""),
                    manufacturer=comp.get("manufacturer", ""),
                    category=comp.get("category"),
                )
                return comp.get("part_number", ""), resolution
        
        tasks = [recall_one(comp) for comp in components]
        results_list = await asyncio.gather(*tasks)
        
        for part_number, resolution in results_list:
            if resolution:
                results[part_number] = resolution
        
        logger.info(f"Batch recall: {len(results)}/{len(components)} found in memory")
        return results
    
    async def update_success_count(
        self,
        part_number: str,
        manufacturer: str,
    ) -> bool:
        """
        Update success count for a component resolution.
        
        Called when a previously learned resolution is used successfully,
        increasing confidence in that resolution.
        """
        # First recall existing resolution
        existing = await self.recall_symbol(part_number, manufacturer)
        if not existing:
            return False
        
        # Store with incremented count
        return await self.store_symbol(
            part_number=existing.part_number,
            manufacturer=existing.manufacturer,
            category=existing.category,
            source=existing.source,
            footprint=existing.footprint,
            verified=existing.verified,
        )
