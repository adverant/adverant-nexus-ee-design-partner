"""
Saleae Logic Analyzer Driver

This driver provides integration with Saleae Logic 2 software via the
Automation API (gRPC). Supports Logic Pro 8, Logic Pro 16, and compatible devices.

Requirements:
- Saleae Logic 2 software running with automation server enabled
- saleae-automation Python package: pip install saleae-automation

The automation API must be enabled in Logic 2 via:
Preferences -> Enable automation server on port 10430
"""

import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ...base_instrument import (
    LogicAnalyzer,
    InstrumentInfo,
    InstrumentType,
    ConnectionType,
    InstrumentStatus,
    InstrumentCapability,
    ConnectionError,
    CommunicationError,
    TimeoutError as InstrumentTimeoutError,
)

logger = logging.getLogger(__name__)

# Try to import saleae automation library
try:
    from saleae import automation
    from saleae.automation import (
        DeviceConfiguration,
        DeviceType,
        LogicDeviceConfiguration,
        GlitchFilterEntry,
        RadixType,
        GraphTimeDelta,
        GraphTimeMarker,
        CaptureConfiguration,
        CaptureMode,
        TimingCaptureModeSettings,
        DigitalTriggerType,
        DigitalTriggerLinkedChannelState,
        DigitalTriggerCaptureMode,
    )
    SALEAE_AVAILABLE = True
except ImportError:
    SALEAE_AVAILABLE = False
    logger.warning("saleae-automation package not installed. Saleae features will be unavailable.")


class SaleaeLogicAnalyzer(LogicAnalyzer):
    """
    Driver for Saleae Logic analyzers via Logic 2 Automation API.

    Supports:
    - Logic Pro 8 (8 digital/analog channels)
    - Logic Pro 16 (16 digital/analog channels)
    - Logic 8 (8 digital channels)

    Features:
    - High-speed digital capture up to 500 MS/s
    - Analog capture up to 50 MS/s
    - Protocol analysis (SPI, I2C, UART, CAN, etc.)
    - Automated capture with triggers
    - Export to multiple formats
    """

    # Supported protocols
    SUPPORTED_PROTOCOLS = [
        "spi", "i2c", "uart", "can", "1-wire", "i2s",
        "jtag", "swd", "manchester", "modbus", "dmx512",
        "parallel", "hdlc", "midi", "lin", "usb_ls_fs",
    ]

    def __init__(
        self,
        instrument_id: Optional[str] = None,
        connection_params: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize the Saleae Logic analyzer driver.

        Args:
            instrument_id: Unique identifier for this instance
            connection_params: Connection parameters:
                - host: Automation server host (default: localhost)
                - port: Automation server port (default: 10430)
                - device_id: Specific device ID to connect to
        """
        super().__init__(instrument_id, connection_params)

        # Connection settings
        self._host = connection_params.get("host", "localhost") if connection_params else "localhost"
        self._port = connection_params.get("port", 10430) if connection_params else 10430
        self._device_id = connection_params.get("device_id") if connection_params else None

        # Saleae objects
        self._manager: Optional[Any] = None  # automation.Manager
        self._device: Optional[Any] = None  # automation.Device
        self._capture: Optional[Any] = None  # Current capture

        # Configuration state
        self._configured_channels: List[int] = []
        self._sample_rate: float = 10_000_000  # 10 MHz default
        self._voltage_threshold: float = 1.5
        self._trigger_channel: Optional[int] = None
        self._trigger_edge: str = "rising"
        self._pre_trigger_buffer: float = 0.1  # 10%

        # Capture data
        self._capture_complete = False
        self._captured_data: Dict[int, List[int]] = {}

    def connect(self) -> bool:
        """
        Connect to the Saleae Logic 2 automation server.

        Returns:
            True if connection successful

        Raises:
            ConnectionError: If connection fails
        """
        if not SALEAE_AVAILABLE:
            raise ConnectionError(
                "saleae-automation package not installed. "
                "Install with: pip install saleae-automation",
                self.instrument_id,
            )

        self.status = InstrumentStatus.CONNECTING
        logger.info(f"Connecting to Saleae automation server at {self._host}:{self._port}")

        try:
            # Connect to automation server
            self._manager = automation.Manager.connect(
                address=self._host,
                port=self._port,
            )

            # Get list of connected devices
            devices = self._manager.get_devices()

            if not devices:
                raise ConnectionError(
                    "No Saleae devices found. Ensure device is connected and Logic 2 is running.",
                    self.instrument_id,
                )

            # Select device
            if self._device_id:
                # Find specific device
                self._device = None
                for dev in devices:
                    if dev.device_id == self._device_id:
                        self._device = dev
                        break
                if not self._device:
                    raise ConnectionError(
                        f"Device {self._device_id} not found",
                        self.instrument_id,
                    )
            else:
                # Use first available device
                self._device = devices[0]

            # Get device info
            self._info = self.get_info()

            self.status = InstrumentStatus.CONNECTED
            self._connected_at = datetime.now()

            logger.info(
                f"Connected to {self._info.manufacturer} {self._info.model} "
                f"(SN: {self._info.serial_number})"
            )

            return True

        except automation.errors.AutomationError as e:
            self._handle_error(ConnectionError(f"Automation API error: {e}", self.instrument_id))
            raise
        except Exception as e:
            self._handle_error(ConnectionError(str(e), self.instrument_id))
            raise

    def disconnect(self) -> None:
        """Disconnect from the automation server."""
        logger.info("Disconnecting from Saleae automation server")

        try:
            if self._capture:
                try:
                    self._capture.close()
                except Exception:
                    pass
                self._capture = None

            if self._manager:
                try:
                    self._manager.close()
                except Exception:
                    pass
                self._manager = None

            self._device = None
        except Exception as e:
            logger.error(f"Error during disconnect: {e}")
        finally:
            self.status = InstrumentStatus.DISCONNECTED

    def get_info(self) -> InstrumentInfo:
        """Get device information."""
        if not self._device:
            raise CommunicationError("Not connected to device", self.instrument_id)

        # Map device type to model name
        device_type = self._device.device_type
        if device_type == DeviceType.LOGIC_PRO_16:
            model = "Logic Pro 16"
            digital_channels = 16
            analog_channels = 16
            max_digital_rate = 500_000_000
            max_analog_rate = 50_000_000
        elif device_type == DeviceType.LOGIC_PRO_8:
            model = "Logic Pro 8"
            digital_channels = 8
            analog_channels = 8
            max_digital_rate = 500_000_000
            max_analog_rate = 50_000_000
        elif device_type == DeviceType.LOGIC_8:
            model = "Logic 8"
            digital_channels = 8
            analog_channels = 0
            max_digital_rate = 100_000_000
            max_analog_rate = 0
        else:
            model = f"Logic ({device_type.name})"
            digital_channels = 8
            analog_channels = 0
            max_digital_rate = 24_000_000
            max_analog_rate = 0

        capabilities = [
            InstrumentCapability(
                name="digital_channels",
                type="feature",
                parameters={"count": digital_channels, "max_sample_rate": max_digital_rate},
            ),
            InstrumentCapability(
                name="analog_channels",
                type="feature",
                parameters={"count": analog_channels, "max_sample_rate": max_analog_rate},
            ),
        ]

        # Add protocol capabilities
        for protocol in self.SUPPORTED_PROTOCOLS:
            capabilities.append(
                InstrumentCapability(
                    name=protocol,
                    type="protocol",
                    parameters={},
                )
            )

        return InstrumentInfo(
            instrument_type=InstrumentType.LOGIC_ANALYZER,
            manufacturer="Saleae",
            model=model,
            serial_number=self._device.device_id,
            firmware_version=None,  # Not available via API
            connection_type=ConnectionType.GRPC,
            connection_params={
                "host": self._host,
                "port": self._port,
                "device_id": self._device.device_id,
            },
            capabilities=capabilities,
            metadata={
                "device_type": device_type.name,
                "digital_channels": digital_channels,
                "analog_channels": analog_channels,
            },
        )

    def reset(self) -> None:
        """Reset the analyzer to default state."""
        if self._capture:
            try:
                self._capture.close()
            except Exception:
                pass
            self._capture = None

        self._configured_channels = []
        self._capture_complete = False
        self._captured_data = {}

        logger.info("Saleae analyzer reset")

    def self_test(self) -> Tuple[bool, Optional[str]]:
        """Run self-test."""
        try:
            if not self._manager or not self._device:
                return False, "Not connected"

            # Try to get device info as a connectivity test
            devices = self._manager.get_devices()
            device_found = any(d.device_id == self._device.device_id for d in devices)

            if device_found:
                return True, None
            else:
                return False, "Device no longer connected"

        except Exception as e:
            return False, str(e)

    def configure_channels(
        self,
        channels: List[int],
        sample_rate: float,
        voltage_threshold: float = 1.5,
    ) -> None:
        """
        Configure digital channels for capture.

        Args:
            channels: List of channel numbers (0-15 for Pro 16)
            sample_rate: Sample rate in Hz
            voltage_threshold: Logic threshold voltage (1.2, 1.5, 1.8, 3.3)
        """
        if not self._device:
            raise CommunicationError("Not connected", self.instrument_id)

        # Validate channels
        device_info = self._info.metadata if self._info else {}
        max_channels = device_info.get("digital_channels", 8)

        for ch in channels:
            if ch < 0 or ch >= max_channels:
                raise ValueError(f"Channel {ch} out of range (0-{max_channels - 1})")

        # Validate sample rate
        max_rate = device_info.get("digital_channels", 100_000_000)
        if sample_rate > max_rate:
            logger.warning(
                f"Requested sample rate {sample_rate} exceeds maximum {max_rate}, "
                f"using maximum"
            )
            sample_rate = max_rate

        self._configured_channels = channels
        self._sample_rate = sample_rate
        self._voltage_threshold = voltage_threshold

        logger.info(
            f"Configured channels {channels} at {sample_rate / 1e6:.1f} MS/s, "
            f"threshold {voltage_threshold}V"
        )

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
        if channel not in self._configured_channels:
            logger.warning(f"Trigger channel {channel} not in configured channels")

        self._trigger_channel = channel
        self._trigger_edge = edge
        self._pre_trigger_buffer = position

        logger.info(f"Configured trigger on channel {channel}, {edge} edge, {position:.0%} pre-trigger")

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
        if not self._manager or not self._device:
            raise CommunicationError("Not connected", self.instrument_id)

        if not self._configured_channels:
            raise ValueError("No channels configured")

        self.status = InstrumentStatus.BUSY
        self._capture_complete = False
        self._captured_data = {}

        try:
            # Build device configuration
            enabled_digital = self._configured_channels
            enabled_analog: List[int] = []  # Could add analog support

            device_config = LogicDeviceConfiguration(
                enabled_digital_channels=enabled_digital,
                enabled_analog_channels=enabled_analog,
                digital_sample_rate=int(self._sample_rate),
                digital_threshold_volts=self._voltage_threshold,
            )

            # Build capture mode settings
            if duration_s:
                capture_settings = TimingCaptureModeSettings(
                    duration_seconds=duration_s,
                    trim_data_seconds=0,
                )
            else:
                # Default 1 second capture
                capture_settings = TimingCaptureModeSettings(
                    duration_seconds=1.0,
                    trim_data_seconds=0,
                )

            # Build trigger if configured
            trigger = None
            if self._trigger_channel is not None:
                if self._trigger_edge == "rising":
                    trigger_type = DigitalTriggerType.RISING
                elif self._trigger_edge == "falling":
                    trigger_type = DigitalTriggerType.FALLING
                else:
                    trigger_type = DigitalTriggerType.PULSE_LOW  # 'either' approximation

                trigger = automation.Trigger(
                    type=DigitalTriggerCaptureMode.DIGITAL,
                    digital_channels=[
                        automation.DigitalTriggerChannel(
                            channel=self._trigger_channel,
                            edge=trigger_type,
                        )
                    ],
                    pre_trigger_fill_percentage=int(self._pre_trigger_buffer * 100),
                )

            # Start capture
            capture_config = CaptureConfiguration(
                capture_mode=CaptureMode.TIMING,
                timing_capture_mode_settings=capture_settings,
                trigger=trigger,
            )

            self._capture = self._manager.start_capture(
                device_id=self._device.device_id,
                device_configuration=device_config,
                capture_configuration=capture_config,
            )

            logger.info(f"Started capture: {duration_s}s, {len(enabled_digital)} channels")

        except Exception as e:
            self.status = InstrumentStatus.ERROR
            raise CommunicationError(f"Failed to start capture: {e}", self.instrument_id)

    def wait_for_capture(self, timeout_s: float = 30.0) -> bool:
        """
        Wait for capture to complete.

        Args:
            timeout_s: Timeout in seconds

        Returns:
            True if capture complete, False if timed out
        """
        if not self._capture:
            return False

        start_time = time.time()

        try:
            # Wait for capture to complete
            self._capture.wait()
            self._capture_complete = True
            self.status = InstrumentStatus.CONNECTED

            logger.info("Capture complete")
            return True

        except automation.errors.CaptureAbortedError:
            logger.warning("Capture was aborted")
            self._capture_complete = False
            self.status = InstrumentStatus.CONNECTED
            return False

        except Exception as e:
            if time.time() - start_time >= timeout_s:
                logger.warning("Capture timed out")
                return False
            raise CommunicationError(f"Capture failed: {e}", self.instrument_id)

    def get_capture_data(
        self,
        channels: Optional[List[int]] = None,
    ) -> Dict[int, List[int]]:
        """
        Get captured digital data.

        Args:
            channels: Optional list of channels to retrieve (None for all)

        Returns:
            Dict mapping channel number to sample values (0 or 1)
        """
        if not self._capture:
            raise CommunicationError("No capture available", self.instrument_id)

        if not self._capture_complete:
            raise CommunicationError("Capture not complete", self.instrument_id)

        target_channels = channels or self._configured_channels
        result: Dict[int, List[int]] = {}

        try:
            # Export to temporary CSV and parse
            # Note: The automation API doesn't provide direct data access,
            # so we export to CSV and parse it

            import tempfile
            import csv

            with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
                temp_path = f.name

            # Export raw digital data
            self._capture.export_raw_data_csv(
                directory=str(Path(temp_path).parent),
                digital_channels=target_channels,
            )

            # Parse CSV
            csv_path = Path(temp_path).parent / "digital.csv"
            if csv_path.exists():
                with open(csv_path, 'r') as f:
                    reader = csv.reader(f)
                    header = next(reader)  # Skip header

                    # Initialize result arrays
                    for ch in target_channels:
                        result[ch] = []

                    # Read data rows
                    for row in reader:
                        for i, ch in enumerate(target_channels):
                            if i + 1 < len(row):
                                result[ch].append(int(row[i + 1]))

                # Cleanup
                csv_path.unlink()

            logger.info(f"Retrieved data for {len(result)} channels")
            return result

        except Exception as e:
            raise CommunicationError(f"Failed to get capture data: {e}", self.instrument_id)

    def decode_protocol(
        self,
        protocol: str,
        channel_mapping: Dict[str, int],
        options: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Decode a protocol from captured data.

        Args:
            protocol: Protocol name (e.g., 'spi', 'i2c', 'uart')
            channel_mapping: Maps signal names to channel numbers
                SPI: {'clk': 0, 'mosi': 1, 'miso': 2, 'cs': 3}
                I2C: {'scl': 0, 'sda': 1}
                UART: {'tx': 0, 'rx': 1}
            options: Protocol-specific options

        Returns:
            List of decoded frames/messages
        """
        if not self._capture:
            raise CommunicationError("No capture available", self.instrument_id)

        protocol = protocol.lower()
        if protocol not in self.SUPPORTED_PROTOCOLS:
            raise ValueError(f"Unsupported protocol: {protocol}")

        options = options or {}
        decoded_frames: List[Dict[str, Any]] = []

        try:
            # Add protocol analyzer based on type
            if protocol == "spi":
                analyzer = self._capture.add_analyzer(
                    "SPI",
                    label="SPI",
                    settings={
                        "MOSI": channel_mapping.get("mosi", 0),
                        "MISO": channel_mapping.get("miso", 1),
                        "Clock": channel_mapping.get("clk", 2),
                        "Enable": channel_mapping.get("cs", 3),
                        "Bits per Transfer": options.get("bits", "8 Bits per Transfer"),
                        "Significant Bit": options.get("msb_first", "Most Significant Bit First"),
                        "Clock State": options.get("cpol", "Clock is Low when inactive"),
                        "Clock Phase": options.get("cpha", "Data is Valid on Clock Leading Edge"),
                        "Enable Line": options.get("cs_active", "Enable line is Active Low"),
                    },
                )

            elif protocol == "i2c":
                analyzer = self._capture.add_analyzer(
                    "I2C",
                    label="I2C",
                    settings={
                        "SDA": channel_mapping.get("sda", 0),
                        "SCL": channel_mapping.get("scl", 1),
                    },
                )

            elif protocol == "uart":
                analyzer = self._capture.add_analyzer(
                    "Async Serial",
                    label="UART",
                    settings={
                        "Input Channel": channel_mapping.get("tx", 0),
                        "Bit Rate (Bits/s)": options.get("baudrate", 115200),
                        "Bits per Frame": options.get("databits", "8 Bits per Frame"),
                        "Stop Bits": options.get("stopbits", "1 Stop Bit"),
                        "Parity Bit": options.get("parity", "No Parity Bit"),
                        "Significant Bit": options.get("msb_first", "Least Significant Bit Sent First"),
                        "Signal inversion": options.get("inverted", "Non Inverted (Idle High)"),
                    },
                )

            elif protocol == "can":
                analyzer = self._capture.add_analyzer(
                    "CAN",
                    label="CAN",
                    settings={
                        "CAN": channel_mapping.get("can", 0),
                        "Bit Rate (Bits/s)": options.get("bitrate", 500000),
                    },
                )

            else:
                raise ValueError(f"Protocol {protocol} not yet implemented")

            # Export decoded data
            import tempfile
            import csv

            with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
                temp_path = f.name

            # Export analyzer data
            self._capture.export_analyzer(
                analyzer=analyzer,
                output_path=temp_path,
                radix=RadixType.HEXADECIMAL,
            )

            # Parse exported frames
            with open(temp_path, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    decoded_frames.append(dict(row))

            # Cleanup
            Path(temp_path).unlink()

            logger.info(f"Decoded {len(decoded_frames)} {protocol.upper()} frames")
            return decoded_frames

        except Exception as e:
            raise CommunicationError(f"Failed to decode protocol: {e}", self.instrument_id)

    def export_capture(
        self,
        filepath: str,
        format: str = "csv",
    ) -> None:
        """
        Export captured data to file.

        Args:
            filepath: Output file path
            format: Export format ('csv', 'vcd', 'salae')
        """
        if not self._capture:
            raise CommunicationError("No capture available", self.instrument_id)

        try:
            output_path = Path(filepath)
            output_dir = output_path.parent
            output_dir.mkdir(parents=True, exist_ok=True)

            if format == "csv":
                self._capture.export_raw_data_csv(
                    directory=str(output_dir),
                    digital_channels=self._configured_channels,
                )
            elif format == "salae":
                # Save as Saleae capture file
                self._capture.save_capture(str(output_path))
            else:
                raise ValueError(f"Unsupported export format: {format}")

            logger.info(f"Exported capture to {filepath}")

        except Exception as e:
            raise CommunicationError(f"Failed to export: {e}", self.instrument_id)


# Discovery function for the registry
def discover_saleae_devices() -> List[InstrumentInfo]:
    """
    Discover connected Saleae devices.

    Returns:
        List of discovered device info
    """
    if not SALEAE_AVAILABLE:
        return []

    discovered: List[InstrumentInfo] = []

    try:
        # Try to connect to automation server
        manager = automation.Manager.connect()

        try:
            devices = manager.get_devices()

            for device in devices:
                # Map device type
                device_type = device.device_type
                if device_type == DeviceType.LOGIC_PRO_16:
                    model = "Logic Pro 16"
                elif device_type == DeviceType.LOGIC_PRO_8:
                    model = "Logic Pro 8"
                elif device_type == DeviceType.LOGIC_8:
                    model = "Logic 8"
                else:
                    model = f"Logic ({device_type.name})"

                discovered.append(InstrumentInfo(
                    instrument_type=InstrumentType.LOGIC_ANALYZER,
                    manufacturer="Saleae",
                    model=model,
                    serial_number=device.device_id,
                    connection_type=ConnectionType.GRPC,
                    connection_params={
                        "host": "localhost",
                        "port": 10430,
                        "device_id": device.device_id,
                    },
                ))

        finally:
            manager.close()

    except Exception as e:
        logger.debug(f"Saleae discovery failed: {e}")

    return discovered
