"""
Hardware-in-the-Loop (HIL) Testing Package

This package provides comprehensive HIL testing capabilities for EE Design Partner,
including instrument drivers, test execution, and data analysis.
"""

from .base_instrument import (
    BaseInstrument,
    InstrumentInfo,
    InstrumentError,
    ConnectionError,
    CommunicationError,
    TimeoutError,
)
from .progress_reporter import ProgressReporter, emit_progress
from .instrument_registry import InstrumentRegistry, get_registry

__version__ = "1.0.0"

__all__ = [
    "BaseInstrument",
    "InstrumentInfo",
    "InstrumentError",
    "ConnectionError",
    "CommunicationError",
    "TimeoutError",
    "ProgressReporter",
    "emit_progress",
    "InstrumentRegistry",
    "get_registry",
]
