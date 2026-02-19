"""
Retry utilities for MAPO Schematic Pipeline.

Provides an async retry wrapper with exponential back-off for phases that
call external LLM services (connections, smoke test).

Deterministic phases (assembly, layout, export) should NOT be wrapped.
"""

import asyncio
import logging
from typing import Any, Awaitable, Callable, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


async def retry_phase(
    func: Callable[..., Awaitable[T]],
    *args: Any,
    max_retries: int = 3,
    base_delay: float = 10.0,
    phase_name: str = "unknown",
    **kwargs: Any,
) -> T:
    """
    Retry an async callable with exponential back-off.

    Args:
        func: Async function to call.
        *args: Positional args forwarded to func.
        max_retries: Maximum number of attempts (including the first).
        base_delay: Initial delay in seconds before first retry.
        phase_name: Human-readable name for logging.
        **kwargs: Keyword args forwarded to func.

    Returns:
        The return value of func on success.

    Raises:
        The last exception if all retries are exhausted.
    """
    last_exc: BaseException | None = None

    for attempt in range(1, max_retries + 1):
        try:
            return await func(*args, **kwargs)
        except Exception as exc:
            last_exc = exc
            if attempt < max_retries:
                delay = base_delay * (2 ** (attempt - 1))
                logger.warning(
                    f"[{phase_name}] Attempt {attempt}/{max_retries} failed: {exc}. "
                    f"Retrying in {delay:.0f}s..."
                )
                await asyncio.sleep(delay)
            else:
                logger.error(
                    f"[{phase_name}] All {max_retries} attempts failed. Last error: {exc}"
                )

    # Should not reach here, but satisfy type checker
    raise last_exc  # type: ignore[misc]
