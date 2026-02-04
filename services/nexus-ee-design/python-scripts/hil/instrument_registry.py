"""
Instrument Registry for HIL Testing

This module provides a registry for discovering and managing HIL instruments.
It supports automatic discovery, factory creation, and instrument pooling.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Type
from threading import Lock

from .base_instrument import (
    BaseInstrument,
    InstrumentInfo,
    InstrumentType,
    ConnectionType,
    InstrumentStatus,
    InstrumentError,
    ConnectionError,
)
from .progress_reporter import ProgressReporter

logger = logging.getLogger(__name__)


@dataclass
class RegisteredInstrument:
    """Information about a registered instrument in the registry."""
    instrument_id: str
    instrument: BaseInstrument
    info: InstrumentInfo
    registered_at: datetime
    last_used_at: Optional[datetime] = None
    in_use: bool = False
    use_count: int = 0
    error_count: int = 0


@dataclass
class DriverRegistration:
    """Registration for an instrument driver class."""
    driver_class: Type[BaseInstrument]
    instrument_type: InstrumentType
    manufacturers: List[str]  # Manufacturer names this driver supports
    models: List[str]  # Model patterns (can use wildcards)
    connection_types: List[ConnectionType]
    discovery_function: Optional[Callable[[], List[InstrumentInfo]]] = None


class InstrumentRegistry:
    """
    Central registry for HIL instruments.

    The registry provides:
    - Driver registration for different instrument types
    - Automatic instrument discovery
    - Instrument factory for creating driver instances
    - Instrument pooling for resource management
    """

    _instance: Optional["InstrumentRegistry"] = None
    _lock = Lock()

    def __new__(cls) -> "InstrumentRegistry":
        """Singleton pattern."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        """Initialize the registry."""
        if self._initialized:
            return

        self._drivers: Dict[str, DriverRegistration] = {}
        self._instruments: Dict[str, RegisteredInstrument] = {}
        self._discovery_handlers: Dict[InstrumentType, List[Callable]] = {}
        self._progress_reporter: Optional[ProgressReporter] = None
        self._initialized = True

        logger.info("Instrument registry initialized")

    def set_progress_reporter(self, reporter: ProgressReporter) -> None:
        """Set the progress reporter for emitting discovery events."""
        self._progress_reporter = reporter

    # ==========================================================================
    # Driver Registration
    # ==========================================================================

    def register_driver(
        self,
        driver_class: Type[BaseInstrument],
        instrument_type: InstrumentType,
        manufacturers: List[str],
        models: List[str],
        connection_types: List[ConnectionType],
        discovery_function: Optional[Callable[[], List[InstrumentInfo]]] = None,
    ) -> None:
        """
        Register a driver class for a specific instrument type.

        Args:
            driver_class: The driver class to register
            instrument_type: Type of instrument this driver handles
            manufacturers: List of supported manufacturer names
            models: List of supported model patterns
            connection_types: Supported connection types
            discovery_function: Optional function to discover instruments
        """
        driver_key = f"{instrument_type.value}:{driver_class.__name__}"

        self._drivers[driver_key] = DriverRegistration(
            driver_class=driver_class,
            instrument_type=instrument_type,
            manufacturers=manufacturers,
            models=models,
            connection_types=connection_types,
            discovery_function=discovery_function,
        )

        # Register discovery handler
        if discovery_function:
            if instrument_type not in self._discovery_handlers:
                self._discovery_handlers[instrument_type] = []
            self._discovery_handlers[instrument_type].append(discovery_function)

        logger.info(
            f"Registered driver {driver_class.__name__} for "
            f"{instrument_type.value}: {manufacturers}"
        )

    def get_driver(
        self,
        instrument_type: InstrumentType,
        manufacturer: str,
        model: str,
    ) -> Optional[Type[BaseInstrument]]:
        """
        Get the appropriate driver class for an instrument.

        Args:
            instrument_type: Type of instrument
            manufacturer: Manufacturer name
            model: Model name

        Returns:
            Driver class or None if not found
        """
        for registration in self._drivers.values():
            if registration.instrument_type != instrument_type:
                continue

            # Check manufacturer match (case-insensitive)
            manufacturer_match = any(
                m.lower() in manufacturer.lower() or manufacturer.lower() in m.lower()
                for m in registration.manufacturers
            )
            if not manufacturer_match:
                continue

            # Check model match (supports wildcards with *)
            model_match = False
            for pattern in registration.models:
                if pattern == "*":
                    model_match = True
                    break
                if pattern.endswith("*"):
                    if model.lower().startswith(pattern[:-1].lower()):
                        model_match = True
                        break
                elif pattern.lower() in model.lower():
                    model_match = True
                    break

            if model_match:
                return registration.driver_class

        return None

    # ==========================================================================
    # Instrument Discovery
    # ==========================================================================

    def discover_instruments(
        self,
        instrument_types: Optional[List[InstrumentType]] = None,
    ) -> List[InstrumentInfo]:
        """
        Discover available instruments.

        Args:
            instrument_types: Types to discover (None for all)

        Returns:
            List of discovered instrument info
        """
        discovered: List[InstrumentInfo] = []
        types_to_scan = instrument_types or list(InstrumentType)

        logger.info(f"Starting instrument discovery for types: {types_to_scan}")

        for inst_type in types_to_scan:
            handlers = self._discovery_handlers.get(inst_type, [])
            for handler in handlers:
                try:
                    instruments = handler()
                    for info in instruments:
                        discovered.append(info)
                        if self._progress_reporter:
                            self._progress_reporter.instrument_discovered(
                                instrument_id=info.serial_number or f"{info.manufacturer}_{info.model}",
                                manufacturer=info.manufacturer,
                                model=info.model,
                                instrument_type=info.instrument_type.value,
                            )
                        logger.info(
                            f"Discovered {info.manufacturer} {info.model} "
                            f"(SN: {info.serial_number})"
                        )
                except Exception as e:
                    logger.error(f"Discovery handler failed: {e}")

        logger.info(f"Discovery complete: {len(discovered)} instruments found")
        return discovered

    # ==========================================================================
    # Instrument Factory
    # ==========================================================================

    def create_instrument(
        self,
        info: InstrumentInfo,
        instrument_id: Optional[str] = None,
    ) -> BaseInstrument:
        """
        Create an instrument driver instance.

        Args:
            info: Instrument info from discovery
            instrument_id: Optional custom ID

        Returns:
            Configured driver instance

        Raises:
            InstrumentError: If no suitable driver found
        """
        driver_class = self.get_driver(
            info.instrument_type,
            info.manufacturer,
            info.model,
        )

        if driver_class is None:
            raise InstrumentError(
                f"No driver found for {info.manufacturer} {info.model}",
                instrument_id=instrument_id,
            )

        instrument = driver_class(
            instrument_id=instrument_id or info.serial_number,
            connection_params=info.connection_params,
        )

        logger.info(
            f"Created instrument driver {driver_class.__name__} "
            f"for {info.manufacturer} {info.model}"
        )

        return instrument

    def connect_instrument(
        self,
        info: InstrumentInfo,
        instrument_id: Optional[str] = None,
    ) -> BaseInstrument:
        """
        Create and connect an instrument.

        Args:
            info: Instrument info from discovery
            instrument_id: Optional custom ID

        Returns:
            Connected driver instance

        Raises:
            ConnectionError: If connection fails
        """
        instrument = self.create_instrument(info, instrument_id)

        try:
            if instrument.connect():
                actual_info = instrument.get_info()
                self._register_instrument(instrument_id or info.serial_number, instrument, actual_info)

                if self._progress_reporter:
                    self._progress_reporter.instrument_connected(
                        instrument_id=instrument.instrument_id,
                        name=f"{actual_info.manufacturer} {actual_info.model}",
                    )

                return instrument
            else:
                raise ConnectionError(
                    f"Failed to connect to {info.manufacturer} {info.model}",
                    instrument_id=instrument_id,
                )
        except Exception as e:
            if self._progress_reporter:
                self._progress_reporter.instrument_error(
                    instrument_id=instrument_id or "unknown",
                    error_message=str(e),
                )
            raise

    # ==========================================================================
    # Instrument Management
    # ==========================================================================

    def _register_instrument(
        self,
        instrument_id: str,
        instrument: BaseInstrument,
        info: InstrumentInfo,
    ) -> None:
        """Register an instrument in the registry."""
        self._instruments[instrument_id] = RegisteredInstrument(
            instrument_id=instrument_id,
            instrument=instrument,
            info=info,
            registered_at=datetime.now(),
        )

    def get_instrument(self, instrument_id: str) -> Optional[BaseInstrument]:
        """
        Get a registered instrument by ID.

        Args:
            instrument_id: Instrument ID

        Returns:
            Instrument instance or None
        """
        registered = self._instruments.get(instrument_id)
        if registered:
            registered.last_used_at = datetime.now()
            registered.use_count += 1
            return registered.instrument
        return None

    def release_instrument(self, instrument_id: str) -> None:
        """
        Release an instrument back to the pool.

        Args:
            instrument_id: Instrument ID
        """
        registered = self._instruments.get(instrument_id)
        if registered:
            registered.in_use = False
            logger.debug(f"Released instrument {instrument_id}")

    def disconnect_instrument(self, instrument_id: str) -> None:
        """
        Disconnect and unregister an instrument.

        Args:
            instrument_id: Instrument ID
        """
        registered = self._instruments.get(instrument_id)
        if registered:
            try:
                registered.instrument.disconnect()
            except Exception as e:
                logger.error(f"Error disconnecting instrument: {e}")

            del self._instruments[instrument_id]

            if self._progress_reporter:
                self._progress_reporter.instrument_disconnected(
                    instrument_id=instrument_id,
                    reason="Disconnected by registry",
                )

            logger.info(f"Disconnected and unregistered instrument {instrument_id}")

    def disconnect_all(self) -> None:
        """Disconnect all registered instruments."""
        instrument_ids = list(self._instruments.keys())
        for instrument_id in instrument_ids:
            self.disconnect_instrument(instrument_id)

    def get_connected_instruments(
        self,
        instrument_type: Optional[InstrumentType] = None,
    ) -> List[RegisteredInstrument]:
        """
        Get list of connected instruments.

        Args:
            instrument_type: Optional type filter

        Returns:
            List of registered instruments
        """
        instruments = list(self._instruments.values())

        if instrument_type:
            instruments = [
                i for i in instruments
                if i.info.instrument_type == instrument_type
            ]

        return instruments

    def get_available_instruments(
        self,
        instrument_type: Optional[InstrumentType] = None,
    ) -> List[RegisteredInstrument]:
        """
        Get list of available (not in use) instruments.

        Args:
            instrument_type: Optional type filter

        Returns:
            List of available instruments
        """
        instruments = self.get_connected_instruments(instrument_type)
        return [i for i in instruments if not i.in_use]

    def acquire_instrument(
        self,
        instrument_type: InstrumentType,
        capabilities: Optional[List[str]] = None,
    ) -> Optional[BaseInstrument]:
        """
        Acquire an available instrument of the specified type.

        Args:
            instrument_type: Required instrument type
            capabilities: Required capabilities

        Returns:
            Acquired instrument or None if none available
        """
        available = self.get_available_instruments(instrument_type)

        for registered in available:
            # Check capabilities if specified
            if capabilities:
                inst_caps = {c.name for c in registered.info.capabilities}
                if not all(c in inst_caps for c in capabilities):
                    continue

            # Mark as in use
            registered.in_use = True
            registered.last_used_at = datetime.now()
            registered.use_count += 1

            logger.debug(
                f"Acquired instrument {registered.instrument_id} "
                f"(type: {instrument_type.value})"
            )

            return registered.instrument

        logger.warning(f"No available instrument of type {instrument_type.value}")
        return None

    # ==========================================================================
    # Statistics
    # ==========================================================================

    def get_statistics(self) -> Dict[str, Any]:
        """Get registry statistics."""
        by_type: Dict[str, int] = {}
        for registered in self._instruments.values():
            type_name = registered.info.instrument_type.value
            by_type[type_name] = by_type.get(type_name, 0) + 1

        return {
            "total_instruments": len(self._instruments),
            "in_use": sum(1 for i in self._instruments.values() if i.in_use),
            "available": sum(1 for i in self._instruments.values() if not i.in_use),
            "by_type": by_type,
            "registered_drivers": len(self._drivers),
        }


# Singleton accessor
def get_registry() -> InstrumentRegistry:
    """Get the global instrument registry instance."""
    return InstrumentRegistry()
