"""
Base Instrument Classes for HIL Testing

This module provides abstract base classes for all HIL instrument types,
defining common interfaces for connection, control, and measurement.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Callable
import logging
import time

logger = logging.getLogger(__name__)


# ============================================================================
# Exceptions
# ============================================================================


class InstrumentError(Exception):
    """Base exception for all instrument errors."""

    def __init__(self, message: str, instrument_id: Optional[str] = None):
        self.instrument_id = instrument_id
        super().__init__(message)


class ConnectionError(InstrumentError):
    """Failed to connect to instrument."""
    pass


class CommunicationError(InstrumentError):
    """Communication with instrument failed."""
    pass


class TimeoutError(InstrumentError):
    """Operation timed out."""
    pass


class ConfigurationError(InstrumentError):
    """Invalid instrument configuration."""
    pass


class CalibrationError(InstrumentError):
    """Instrument calibration issue."""
    pass


# ============================================================================
# Data Types
# ============================================================================


class InstrumentType(Enum):
    """Supported instrument types."""
    LOGIC_ANALYZER = "logic_analyzer"
    OSCILLOSCOPE = "oscilloscope"
    POWER_SUPPLY = "power_supply"
    MOTOR_EMULATOR = "motor_emulator"
    DAQ = "daq"
    CAN_ANALYZER = "can_analyzer"
    FUNCTION_GEN = "function_gen"
    THERMAL_CAMERA = "thermal_camera"
    ELECTRONIC_LOAD = "electronic_load"


class ConnectionType(Enum):
    """Connection types for instruments."""
    USB = "usb"
    ETHERNET = "ethernet"
    GPIB = "gpib"
    SERIAL = "serial"
    GRPC = "grpc"
    MODBUS_TCP = "modbus_tcp"
    MODBUS_RTU = "modbus_rtu"


class InstrumentStatus(Enum):
    """Instrument connection status."""
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    BUSY = "busy"
    ERROR = "error"
    INITIALIZING = "initializing"


@dataclass
class InstrumentCapability:
    """Describes a capability of an instrument."""
    name: str
    type: str  # 'protocol', 'feature', 'range', 'measurement'
    parameters: Dict[str, Any] = field(default_factory=dict)


@dataclass
class InstrumentInfo:
    """Information about a discovered or connected instrument."""
    instrument_type: InstrumentType
    manufacturer: str
    model: str
    serial_number: Optional[str] = None
    firmware_version: Optional[str] = None
    connection_type: Optional[ConnectionType] = None
    connection_params: Dict[str, Any] = field(default_factory=dict)
    capabilities: List[InstrumentCapability] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "instrument_type": self.instrument_type.value,
            "manufacturer": self.manufacturer,
            "model": self.model,
            "serial_number": self.serial_number,
            "firmware_version": self.firmware_version,
            "connection_type": self.connection_type.value if self.connection_type else None,
            "connection_params": self.connection_params,
            "capabilities": [
                {"name": c.name, "type": c.type, "parameters": c.parameters}
                for c in self.capabilities
            ],
            "metadata": self.metadata,
        }


@dataclass
class MeasurementResult:
    """Result of a single measurement."""
    measurement_type: str
    value: float
    unit: str
    channel: Optional[str] = None
    timestamp: Optional[datetime] = None
    passed: Optional[bool] = None
    min_limit: Optional[float] = None
    max_limit: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "measurement_type": self.measurement_type,
            "value": self.value,
            "unit": self.unit,
            "channel": self.channel,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "passed": self.passed,
            "min_limit": self.min_limit,
            "max_limit": self.max_limit,
            "metadata": self.metadata,
        }


# ============================================================================
# Base Instrument Class
# ============================================================================


class BaseInstrument(ABC):
    """
    Abstract base class for all HIL test instruments.

    All instrument drivers must inherit from this class and implement
    the required abstract methods.
    """

    def __init__(
        self,
        instrument_id: Optional[str] = None,
        connection_params: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize the instrument.

        Args:
            instrument_id: Unique identifier for this instrument instance
            connection_params: Connection parameters (host, port, serial, etc.)
        """
        self.instrument_id = instrument_id
        self.connection_params = connection_params or {}
        self._status = InstrumentStatus.DISCONNECTED
        self._info: Optional[InstrumentInfo] = None
        self._last_error: Optional[str] = None
        self._connected_at: Optional[datetime] = None
        self._error_count = 0

        # Callbacks
        self._on_status_change: Optional[Callable[[InstrumentStatus], None]] = None
        self._on_error: Optional[Callable[[Exception], None]] = None

    @property
    def status(self) -> InstrumentStatus:
        """Get current instrument status."""
        return self._status

    @status.setter
    def status(self, new_status: InstrumentStatus) -> None:
        """Set instrument status and notify listeners."""
        if new_status != self._status:
            old_status = self._status
            self._status = new_status
            logger.debug(
                f"Instrument {self.instrument_id} status changed: "
                f"{old_status.value} -> {new_status.value}"
            )
            if self._on_status_change:
                self._on_status_change(new_status)

    @property
    def info(self) -> Optional[InstrumentInfo]:
        """Get instrument information (populated after connection)."""
        return self._info

    @property
    def is_connected(self) -> bool:
        """Check if instrument is connected."""
        return self._status in (InstrumentStatus.CONNECTED, InstrumentStatus.BUSY)

    @property
    def last_error(self) -> Optional[str]:
        """Get last error message."""
        return self._last_error

    def set_status_callback(
        self, callback: Callable[[InstrumentStatus], None]
    ) -> None:
        """Set callback for status changes."""
        self._on_status_change = callback

    def set_error_callback(self, callback: Callable[[Exception], None]) -> None:
        """Set callback for errors."""
        self._on_error = callback

    def _handle_error(self, error: Exception) -> None:
        """Handle an error - update state and notify listeners."""
        self._last_error = str(error)
        self._error_count += 1
        self.status = InstrumentStatus.ERROR
        logger.error(f"Instrument {self.instrument_id} error: {error}")
        if self._on_error:
            self._on_error(error)

    # ==========================================================================
    # Abstract Methods - Must be implemented by all instruments
    # ==========================================================================

    @abstractmethod
    def connect(self) -> bool:
        """
        Establish connection to the instrument.

        Returns:
            True if connection successful, False otherwise

        Raises:
            ConnectionError: If connection fails
        """
        pass

    @abstractmethod
    def disconnect(self) -> None:
        """
        Disconnect from the instrument.

        Should clean up resources and set status to DISCONNECTED.
        """
        pass

    @abstractmethod
    def get_info(self) -> InstrumentInfo:
        """
        Get instrument information.

        Should query the instrument for manufacturer, model, serial, etc.

        Returns:
            InstrumentInfo object with device details
        """
        pass

    @abstractmethod
    def reset(self) -> None:
        """
        Reset the instrument to default state.

        Raises:
            CommunicationError: If reset fails
        """
        pass

    @abstractmethod
    def self_test(self) -> Tuple[bool, Optional[str]]:
        """
        Run instrument self-test.

        Returns:
            Tuple of (passed, error_message)
        """
        pass


# ============================================================================
# Oscilloscope Base Class
# ============================================================================


class Oscilloscope(BaseInstrument):
    """
    Abstract base class for oscilloscopes.

    Provides interface for channel configuration, triggering, and waveform capture.
    """

    @abstractmethod
    def configure_channel(
        self,
        channel: str,
        enabled: bool = True,
        coupling: str = "dc",
        voltage_scale: float = 1.0,
        voltage_offset: float = 0.0,
        probe_attenuation: float = 1.0,
        bandwidth_limit: Optional[float] = None,
    ) -> None:
        """
        Configure an oscilloscope channel.

        Args:
            channel: Channel identifier (e.g., 'CH1', 'CH2')
            enabled: Whether channel is enabled
            coupling: Coupling mode ('dc', 'ac', 'gnd')
            voltage_scale: Volts per division
            voltage_offset: Vertical offset in volts
            probe_attenuation: Probe attenuation factor (1x, 10x, etc.)
            bandwidth_limit: Bandwidth limit in Hz (None for full bandwidth)
        """
        pass

    @abstractmethod
    def configure_timebase(
        self,
        time_scale: float,
        time_offset: float = 0.0,
        sample_rate: Optional[float] = None,
    ) -> None:
        """
        Configure timebase settings.

        Args:
            time_scale: Time per division in seconds
            time_offset: Horizontal offset in seconds
            sample_rate: Optional specific sample rate in Hz
        """
        pass

    @abstractmethod
    def configure_trigger(
        self,
        source: str,
        level: float,
        edge: str = "rising",
        mode: str = "auto",
        holdoff: float = 0.0,
        position: float = 0.5,
    ) -> None:
        """
        Configure trigger settings.

        Args:
            source: Trigger source channel
            level: Trigger level in volts
            edge: Trigger edge ('rising', 'falling', 'either')
            mode: Trigger mode ('auto', 'normal', 'single')
            holdoff: Holdoff time in seconds
            position: Trigger position (0-1, fraction of screen)
        """
        pass

    @abstractmethod
    def arm_trigger(self) -> None:
        """Arm the trigger and prepare for acquisition."""
        pass

    @abstractmethod
    def wait_for_trigger(self, timeout_s: float = 10.0) -> bool:
        """
        Wait for trigger event.

        Args:
            timeout_s: Timeout in seconds

        Returns:
            True if triggered, False if timed out
        """
        pass

    @abstractmethod
    def get_waveform(
        self,
        channels: List[str],
        num_points: Optional[int] = None,
    ) -> Dict[str, Tuple[List[float], List[float]]]:
        """
        Get waveform data from specified channels.

        Args:
            channels: List of channel identifiers
            num_points: Optional number of points to retrieve

        Returns:
            Dict mapping channel to (time_array, voltage_array)
        """
        pass

    @abstractmethod
    def measure(
        self,
        channel: str,
        measurement_type: str,
    ) -> MeasurementResult:
        """
        Perform automatic measurement on a channel.

        Args:
            channel: Channel to measure
            measurement_type: Type of measurement (e.g., 'frequency', 'vrms', 'vpp')

        Returns:
            MeasurementResult with measurement value
        """
        pass

    @abstractmethod
    def get_screenshot(self, format: str = "png") -> bytes:
        """
        Capture screenshot from oscilloscope display.

        Args:
            format: Image format ('png', 'bmp', etc.)

        Returns:
            Image data as bytes
        """
        pass


# ============================================================================
# Logic Analyzer Base Class
# ============================================================================


class LogicAnalyzer(BaseInstrument):
    """
    Abstract base class for logic analyzers.

    Provides interface for digital capture and protocol decoding.
    """

    @abstractmethod
    def configure_channels(
        self,
        channels: List[int],
        sample_rate: float,
        voltage_threshold: float = 1.5,
    ) -> None:
        """
        Configure digital channels for capture.

        Args:
            channels: List of channel numbers to capture
            sample_rate: Sample rate in Hz
            voltage_threshold: Logic threshold voltage
        """
        pass

    @abstractmethod
    def configure_trigger(
        self,
        channel: int,
        edge: str = "rising",
        position: float = 0.1,
    ) -> None:
        """
        Configure digital trigger.

        Args:
            channel: Trigger channel number
            edge: Trigger edge ('rising', 'falling', 'either')
            position: Pre-trigger buffer position (0-1)
        """
        pass

    @abstractmethod
    def start_capture(
        self,
        duration_s: Optional[float] = None,
        num_samples: Optional[int] = None,
    ) -> None:
        """
        Start logic capture.

        Args:
            duration_s: Capture duration in seconds
            num_samples: Number of samples to capture
        """
        pass

    @abstractmethod
    def wait_for_capture(self, timeout_s: float = 30.0) -> bool:
        """
        Wait for capture to complete.

        Args:
            timeout_s: Timeout in seconds

        Returns:
            True if capture complete, False if timed out
        """
        pass

    @abstractmethod
    def get_capture_data(
        self,
        channels: Optional[List[int]] = None,
    ) -> Dict[int, List[int]]:
        """
        Get captured digital data.

        Args:
            channels: Optional list of channels to retrieve (None for all)

        Returns:
            Dict mapping channel number to sample values
        """
        pass

    @abstractmethod
    def decode_protocol(
        self,
        protocol: str,
        channel_mapping: Dict[str, int],
        options: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Decode a protocol from captured data.

        Args:
            protocol: Protocol name (e.g., 'spi', 'i2c', 'uart', 'can')
            channel_mapping: Maps signal names to channel numbers
            options: Protocol-specific options

        Returns:
            List of decoded frames/messages
        """
        pass

    @abstractmethod
    def export_capture(
        self,
        filepath: str,
        format: str = "csv",
    ) -> None:
        """
        Export captured data to file.

        Args:
            filepath: Output file path
            format: Export format ('csv', 'vcd', 'salae', 'sigrok')
        """
        pass


# ============================================================================
# Power Supply Base Class
# ============================================================================


class PowerSupply(BaseInstrument):
    """
    Abstract base class for programmable power supplies.

    Provides interface for voltage/current control and monitoring.
    """

    @abstractmethod
    def get_num_channels(self) -> int:
        """Get number of output channels."""
        pass

    @abstractmethod
    def set_voltage(self, channel: int, voltage: float) -> None:
        """
        Set output voltage.

        Args:
            channel: Channel number (1-based)
            voltage: Target voltage in volts
        """
        pass

    @abstractmethod
    def set_current_limit(self, channel: int, current: float) -> None:
        """
        Set current limit.

        Args:
            channel: Channel number (1-based)
            current: Current limit in amps
        """
        pass

    @abstractmethod
    def enable_output(self, channel: int, enabled: bool = True) -> None:
        """
        Enable or disable channel output.

        Args:
            channel: Channel number (1-based)
            enabled: Whether to enable output
        """
        pass

    @abstractmethod
    def get_voltage(self, channel: int) -> float:
        """
        Read actual output voltage.

        Args:
            channel: Channel number (1-based)

        Returns:
            Measured voltage in volts
        """
        pass

    @abstractmethod
    def get_current(self, channel: int) -> float:
        """
        Read actual output current.

        Args:
            channel: Channel number (1-based)

        Returns:
            Measured current in amps
        """
        pass

    @abstractmethod
    def get_power(self, channel: int) -> float:
        """
        Get output power.

        Args:
            channel: Channel number (1-based)

        Returns:
            Output power in watts
        """
        pass

    @abstractmethod
    def get_mode(self, channel: int) -> str:
        """
        Get current operating mode.

        Args:
            channel: Channel number (1-based)

        Returns:
            Mode string ('cv', 'cc', 'off', 'unregulated')
        """
        pass

    @abstractmethod
    def is_output_enabled(self, channel: int) -> bool:
        """
        Check if output is enabled.

        Args:
            channel: Channel number (1-based)

        Returns:
            True if output is enabled
        """
        pass

    def set_voltage_current(
        self,
        channel: int,
        voltage: float,
        current_limit: float,
    ) -> None:
        """
        Set both voltage and current limit.

        Args:
            channel: Channel number (1-based)
            voltage: Target voltage in volts
            current_limit: Current limit in amps
        """
        self.set_voltage(channel, voltage)
        self.set_current_limit(channel, current_limit)

    def enable_all(self, enabled: bool = True) -> None:
        """Enable or disable all channels."""
        for ch in range(1, self.get_num_channels() + 1):
            self.enable_output(ch, enabled)


# ============================================================================
# Data Acquisition Base Class
# ============================================================================


class DataAcquisition(BaseInstrument):
    """
    Abstract base class for DAQ devices.

    Provides interface for analog/digital I/O.
    """

    @abstractmethod
    def configure_analog_input(
        self,
        channel: str,
        voltage_range: Tuple[float, float],
        sample_rate: float,
        samples_per_channel: int,
        coupling: str = "dc",
    ) -> None:
        """
        Configure analog input channel.

        Args:
            channel: Channel name (e.g., 'ai0', 'ai1')
            voltage_range: (min_voltage, max_voltage) tuple
            sample_rate: Sample rate in Hz
            samples_per_channel: Number of samples to acquire
            coupling: Coupling mode ('dc', 'ac')
        """
        pass

    @abstractmethod
    def configure_analog_output(
        self,
        channel: str,
        voltage_range: Tuple[float, float],
    ) -> None:
        """
        Configure analog output channel.

        Args:
            channel: Channel name (e.g., 'ao0', 'ao1')
            voltage_range: (min_voltage, max_voltage) tuple
        """
        pass

    @abstractmethod
    def read_analog(
        self,
        channels: List[str],
        num_samples: int,
    ) -> Dict[str, List[float]]:
        """
        Read analog input values.

        Args:
            channels: List of channel names
            num_samples: Number of samples to read

        Returns:
            Dict mapping channel to list of voltage values
        """
        pass

    @abstractmethod
    def write_analog(
        self,
        channel: str,
        values: List[float],
    ) -> None:
        """
        Write analog output values.

        Args:
            channel: Channel name
            values: List of voltage values to output
        """
        pass

    @abstractmethod
    def configure_digital_input(
        self,
        lines: List[str],
    ) -> None:
        """
        Configure digital input lines.

        Args:
            lines: List of line names (e.g., 'port0/line0')
        """
        pass

    @abstractmethod
    def configure_digital_output(
        self,
        lines: List[str],
    ) -> None:
        """
        Configure digital output lines.

        Args:
            lines: List of line names
        """
        pass

    @abstractmethod
    def read_digital(
        self,
        lines: List[str],
    ) -> Dict[str, bool]:
        """
        Read digital input values.

        Args:
            lines: List of line names

        Returns:
            Dict mapping line to boolean value
        """
        pass

    @abstractmethod
    def write_digital(
        self,
        line: str,
        value: bool,
    ) -> None:
        """
        Write digital output value.

        Args:
            line: Line name
            value: Boolean value to output
        """
        pass


# ============================================================================
# Motor Emulator Base Class
# ============================================================================


class MotorEmulator(BaseInstrument):
    """
    Abstract base class for motor load emulators/dynamometers.

    Provides interface for controlling motor load and measuring performance.
    """

    @abstractmethod
    def set_load_torque(self, torque: float) -> None:
        """
        Set load torque.

        Args:
            torque: Load torque in Nm
        """
        pass

    @abstractmethod
    def set_speed(self, speed: float) -> None:
        """
        Set target speed (for speed-controlled mode).

        Args:
            speed: Target speed in RPM
        """
        pass

    @abstractmethod
    def set_inertia(self, inertia: float) -> None:
        """
        Set emulated inertia.

        Args:
            inertia: Inertia in kg*m^2
        """
        pass

    @abstractmethod
    def get_speed(self) -> float:
        """
        Get current motor speed.

        Returns:
            Speed in RPM
        """
        pass

    @abstractmethod
    def get_torque(self) -> float:
        """
        Get current torque.

        Returns:
            Torque in Nm
        """
        pass

    @abstractmethod
    def get_power(self) -> float:
        """
        Get mechanical power.

        Returns:
            Power in watts
        """
        pass

    @abstractmethod
    def get_position(self) -> float:
        """
        Get rotor position.

        Returns:
            Position in degrees (0-360)
        """
        pass

    @abstractmethod
    def enable(self, enabled: bool = True) -> None:
        """
        Enable or disable the emulator.

        Args:
            enabled: Whether to enable
        """
        pass

    @abstractmethod
    def emergency_stop(self) -> None:
        """Trigger emergency stop."""
        pass


# ============================================================================
# CAN Analyzer Base Class
# ============================================================================


class CANAnalyzer(BaseInstrument):
    """
    Abstract base class for CAN bus analyzers.

    Provides interface for CAN message monitoring and transmission.
    """

    @abstractmethod
    def configure(
        self,
        bitrate: int = 500000,
        sample_point: float = 0.75,
        fd_enabled: bool = False,
        fd_data_bitrate: Optional[int] = None,
    ) -> None:
        """
        Configure CAN interface.

        Args:
            bitrate: CAN bitrate in bps
            sample_point: Sample point (0-1)
            fd_enabled: Enable CAN FD
            fd_data_bitrate: CAN FD data phase bitrate
        """
        pass

    @abstractmethod
    def start(self) -> None:
        """Start CAN bus monitoring."""
        pass

    @abstractmethod
    def stop(self) -> None:
        """Stop CAN bus monitoring."""
        pass

    @abstractmethod
    def send_message(
        self,
        arbitration_id: int,
        data: bytes,
        extended_id: bool = False,
        fd: bool = False,
    ) -> None:
        """
        Send a CAN message.

        Args:
            arbitration_id: CAN ID
            data: Message data (up to 8 bytes, or 64 for CAN FD)
            extended_id: Use extended (29-bit) ID
            fd: Send as CAN FD frame
        """
        pass

    @abstractmethod
    def receive_message(
        self,
        timeout_s: float = 1.0,
    ) -> Optional[Dict[str, Any]]:
        """
        Receive a CAN message.

        Args:
            timeout_s: Receive timeout in seconds

        Returns:
            Message dict or None if timeout
        """
        pass

    @abstractmethod
    def set_filter(
        self,
        can_id: int,
        mask: int = 0x7FF,
        extended: bool = False,
    ) -> None:
        """
        Set receive filter.

        Args:
            can_id: CAN ID to accept
            mask: Filter mask
            extended: Use extended ID
        """
        pass

    @abstractmethod
    def get_bus_statistics(self) -> Dict[str, Any]:
        """
        Get CAN bus statistics.

        Returns:
            Dict with message counts, error counts, bus load, etc.
        """
        pass
