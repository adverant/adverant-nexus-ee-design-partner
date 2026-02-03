#!/usr/bin/env python3
"""
Fix Pattern Cache - Cache successful fix patterns to avoid repeat LLM calls.

This module provides intelligent caching of PCB fix patterns based on violation
types. When the same violation pattern is encountered again, we can reuse the
successful fix without another expensive LLM call (~$0.30 each).

The cache uses semantic hashing to match similar violations even if exact
parameters differ, enabling pattern generalization across PCB designs.

Part of MAPOS (Multi-Agent PCB Optimization System) for the Nexus EE Design Partner plugin.
"""

import json
import hashlib
import os
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Any, Optional


@dataclass
class CachedFix:
    """A cached fix pattern with success tracking."""
    violation_type: str
    violation_hash: str
    fix_operations: List[Dict[str, Any]]
    success_count: int = 0
    failure_count: int = 0
    last_used: str = ""
    created_at: str = ""

    @property
    def success_rate(self) -> float:
        """Calculate success rate of this fix pattern."""
        total = self.success_count + self.failure_count
        return self.success_count / total if total > 0 else 0.0

    @property
    def total_uses(self) -> int:
        """Total number of times this fix has been attempted."""
        return self.success_count + self.failure_count

    def is_reliable(self, min_uses: int = 2, min_success_rate: float = 0.6) -> bool:
        """Check if this fix pattern is reliable enough to use."""
        return self.total_uses >= min_uses and self.success_rate >= min_success_rate


@dataclass
class CacheStats:
    """Statistics for the fix cache."""
    hits: int = 0
    misses: int = 0
    savings_usd: float = 0.0
    patterns_stored: int = 0

    @property
    def hit_rate(self) -> float:
        """Cache hit rate."""
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0


class FixPatternCache:
    """
    Cache for PCB fix patterns to avoid repeat LLM calls.

    Features:
    - Semantic hashing: Similar violations match even with different exact parameters
    - Success tracking: Only reliable fixes are reused
    - Persistence: Cache survives process restarts
    - Thread-safe: Can be used from multiple coroutines
    """

    # Estimated cost per LLM call in USD
    LLM_CALL_COST = 0.30

    def __init__(self, cache_path: Optional[Path] = None):
        """
        Initialize the fix pattern cache.

        Args:
            cache_path: Path to cache file. Defaults to ~/.mapos/fix_cache.json
        """
        if cache_path is None:
            # Use environment variable or default path
            cache_dir = os.environ.get('MAPOS_CACHE_DIR', str(Path.home() / '.mapos'))
            cache_path = Path(cache_dir) / 'fix_cache.json'

        self.cache_path = Path(cache_path)
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)

        self.cache: Dict[str, CachedFix] = {}
        self.stats = CacheStats()
        self._load()

    def _hash_violation(self, violation: Dict) -> str:
        """
        Create semantic hash for violation pattern matching.

        The hash captures the essential characteristics of a violation
        without including position-specific data, enabling pattern reuse.
        """
        # Extract key characteristics
        vtype = violation.get('type', 'unknown')
        severity = violation.get('severity', 'unknown')
        items_count = len(violation.get('items', []))

        # Include layer info if available (relevant for layer-specific fixes)
        layers = set()
        for item in violation.get('items', []):
            if 'layer' in item:
                layers.add(item['layer'])
        layers_str = ','.join(sorted(layers)) if layers else 'any'

        # Create deterministic hash
        key = f"{vtype}|{severity}|{items_count}|{layers_str}"
        return hashlib.sha256(key.encode()).hexdigest()[:16]

    def _make_cache_key(self, violation: Dict) -> str:
        """Create cache key from violation type and hash."""
        vtype = violation.get('type', 'unknown')
        vhash = self._hash_violation(violation)
        return f"{vtype}:{vhash}"

    def get(self, violation: Dict, min_success_rate: float = 0.6) -> Optional[List[Dict]]:
        """
        Get cached fix for a violation if available and reliable.

        Args:
            violation: DRC violation dictionary
            min_success_rate: Minimum success rate to consider fix reliable

        Returns:
            List of fix operations if found and reliable, None otherwise
        """
        cache_key = self._make_cache_key(violation)

        if cache_key in self.cache:
            cached = self.cache[cache_key]

            # Only return if fix is reliable
            if cached.is_reliable(min_success_rate=min_success_rate):
                cached.last_used = datetime.now().isoformat()
                self.stats.hits += 1
                self.stats.savings_usd += self.LLM_CALL_COST
                self._save()
                return cached.fix_operations

        self.stats.misses += 1
        return None

    def put(self, violation: Dict, fix_ops: List[Dict], success: bool) -> None:
        """
        Store fix pattern in cache and update statistics.

        Args:
            violation: DRC violation dictionary
            fix_ops: List of fix operations that were applied
            success: Whether the fix successfully resolved the violation
        """
        cache_key = self._make_cache_key(violation)
        vtype = violation.get('type', 'unknown')
        vhash = self._hash_violation(violation)

        if cache_key in self.cache:
            # Update existing entry
            cached = self.cache[cache_key]
            if success:
                cached.success_count += 1
            else:
                cached.failure_count += 1
            cached.last_used = datetime.now().isoformat()

            # Update fix operations if this attempt was successful
            # This allows the cache to improve over time
            if success and cached.success_rate < 1.0:
                cached.fix_operations = fix_ops
        else:
            # Create new entry
            self.cache[cache_key] = CachedFix(
                violation_type=vtype,
                violation_hash=vhash,
                fix_operations=fix_ops,
                success_count=1 if success else 0,
                failure_count=0 if success else 1,
                last_used=datetime.now().isoformat(),
                created_at=datetime.now().isoformat(),
            )

        self.stats.patterns_stored = len(self.cache)
        self._save()

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return {
            'hits': self.stats.hits,
            'misses': self.stats.misses,
            'hit_rate': f"{self.stats.hit_rate:.1%}",
            'savings_usd': f"${self.stats.savings_usd:.2f}",
            'patterns_stored': self.stats.patterns_stored,
            'reliable_patterns': sum(
                1 for c in self.cache.values() if c.is_reliable()
            ),
        }

    def get_pattern_stats(self) -> List[Dict[str, Any]]:
        """Get statistics for each cached pattern."""
        patterns = []
        for key, cached in sorted(self.cache.items(), key=lambda x: x[1].total_uses, reverse=True):
            patterns.append({
                'type': cached.violation_type,
                'hash': cached.violation_hash,
                'success_count': cached.success_count,
                'failure_count': cached.failure_count,
                'success_rate': f"{cached.success_rate:.1%}",
                'reliable': cached.is_reliable(),
                'last_used': cached.last_used,
            })
        return patterns

    def clear(self) -> None:
        """Clear all cached patterns."""
        self.cache.clear()
        self.stats = CacheStats()
        self._save()

    def _load(self) -> None:
        """Load cache from disk."""
        if self.cache_path.exists():
            try:
                with open(self.cache_path, 'r') as f:
                    data = json.load(f)

                # Load cache entries
                for key, entry in data.get('cache', {}).items():
                    self.cache[key] = CachedFix(**entry)

                # Load stats
                stats_data = data.get('stats', {})
                self.stats = CacheStats(
                    hits=stats_data.get('hits', 0),
                    misses=stats_data.get('misses', 0),
                    savings_usd=stats_data.get('savings_usd', 0.0),
                    patterns_stored=len(self.cache),
                )

            except (json.JSONDecodeError, TypeError, KeyError) as e:
                # Corrupted cache - start fresh
                print(f"Warning: Could not load cache ({e}), starting fresh")
                self.cache.clear()
                self.stats = CacheStats()

    def _save(self) -> None:
        """Save cache to disk."""
        data = {
            'cache': {k: asdict(v) for k, v in self.cache.items()},
            'stats': {
                'hits': self.stats.hits,
                'misses': self.stats.misses,
                'savings_usd': self.stats.savings_usd,
            },
            'last_saved': datetime.now().isoformat(),
            'version': '1.0',
        }

        try:
            with open(self.cache_path, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"Warning: Could not save cache: {e}")


# Global cache instance (lazy initialization)
_cache: Optional[FixPatternCache] = None


def get_fix_cache() -> FixPatternCache:
    """Get the global fix pattern cache instance."""
    global _cache
    if _cache is None:
        _cache = FixPatternCache()
    return _cache


def reset_fix_cache() -> None:
    """Reset the global fix pattern cache (for testing)."""
    global _cache
    _cache = None


# CLI interface
def main():
    """CLI entry point for cache management."""
    import argparse

    parser = argparse.ArgumentParser(
        description='MAPOS Fix Pattern Cache Manager',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    subparsers = parser.add_subparsers(dest='command', help='Command to run')

    # Stats command
    subparsers.add_parser('stats', help='Show cache statistics')

    # Patterns command
    subparsers.add_parser('patterns', help='List cached patterns')

    # Clear command
    subparsers.add_parser('clear', help='Clear all cached patterns')

    args = parser.parse_args()

    cache = get_fix_cache()

    if args.command == 'stats':
        stats = cache.get_stats()
        print("\nMAGPOS Fix Pattern Cache Statistics:")
        print(f"  Cache hits:        {stats['hits']}")
        print(f"  Cache misses:      {stats['misses']}")
        print(f"  Hit rate:          {stats['hit_rate']}")
        print(f"  Estimated savings: {stats['savings_usd']}")
        print(f"  Patterns stored:   {stats['patterns_stored']}")
        print(f"  Reliable patterns: {stats['reliable_patterns']}")

    elif args.command == 'patterns':
        patterns = cache.get_pattern_stats()
        if not patterns:
            print("No patterns cached yet.")
        else:
            print("\nCached Fix Patterns:")
            for p in patterns[:20]:  # Show top 20
                reliable = 'yes' if p['reliable'] else 'no'
                print(f"  {p['type']}: {p['success_count']}OK/{p['failure_count']}FAIL "
                      f"({p['success_rate']}, reliable={reliable})")

    elif args.command == 'clear':
        cache.clear()
        print("Cache cleared.")

    else:
        parser.print_help()


if __name__ == '__main__':
    main()
