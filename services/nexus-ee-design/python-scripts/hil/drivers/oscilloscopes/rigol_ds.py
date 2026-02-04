"""
Rigol Oscilloscope Driver

This driver provides integration with Rigol DS/MSO series oscilloscopes
via PyVISA using SCPI commands over USB-TMC or Ethernet.

Supported Models:
- DS1000Z series (DS1054Z, DS1074Z, DS1104Z)
- DS2000A series (DS2072A, DS2102A, DS2202A, DS2302A)
- MSO5000 series
- DS4000E series

Requirements:
- pyvisa: pip install pyvisa
- pyvisa-py or NI-VISA backend

Connection:
- USB: Connect via USB-TMC
- Ethernet: Connect via LXI/VXI-11 or raw socket
"""

import logging
import time
import struct
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from ...base_instrument import (
    Oscilloscope,
    InstrumentInfo,
    InstrumentType,
    ConnectionType,
    InstrumentStatus,
    InstrumentCapability,
    MeasurementResult,
    ConnectionError,
    CommunicationError,
    TimeoutError as InstrumentTimeoutError,
)

logger = logging.getLogger(__name__)

# Try to import pyvisa
try:
    import pyvisa
    PYVISA_AVAILABLE = True
except ImportError:
    PYVISA_AVAILABLE = False
    logger.warning("pyvisa package not installed. Rigol oscilloscope features will be unavailable.")


class RigolOscilloscope(Oscilloscope):
    """
    Driver for Rigol DS/MSO series oscilloscopes via SCPI/VISA.

    Supports:
    - Channel configuration (coupling, scale, offset, probe)
    - Timebase and trigger configuration
    - Waveform capture and measurements
    - Screenshot capture
    - Multiple acquisition modes
    """

    # Measurement type mappings
    MEASUREMENT_TYPES = {
        "frequency": "FREQ",
        "period": "PER",
        "vrms": "VRMS",
        "vpp": "VPP",
        "vmax": "VMAX",
        "vmin": "VMIN",
        "vavg": "VAV",
        "vtop": "VTOP",
        "vbase": "VBAS",
        "vamplitude": "VAMP",
        "rise_time": "RTIM",
        "fall_time": "FTIM",
        "positive_width": "PWID",
        "negative_width": "NWID",
        "duty_cycle": "PDUT",
        "overshoot": "OVER",
        "preshoot": "PRES",
    }

    def __init__(
        self,
        instrument_id: Optional[str] = None,
        connection_params: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize the Rigol oscilloscope driver.

        Args:
            instrument_id: Unique identifier for this instance
            connection_params: Connection parameters:
                - resource_name: VISA resource name (e.g., USB0::0x1AB1::...)
                - host: IP address for Ethernet connection
                - port: Port for raw socket (default 5555)
        """
        super().__init__(instrument_id, connection_params)

        self._resource_name = connection_params.get("resource_name") if connection_params else None
        self._host = connection_params.get("host") if connection_params else None
        self._port = connection_params.get("port", 5555) if connection_params else 5555

        # VISA objects
        self._rm: Optional[Any] = None  # pyvisa.ResourceManager
        self._inst: Optional[Any] = None  # VISA instrument

        # State tracking
        self._num_channels = 4
        self._channel_config: Dict[str, Dict[str, Any]] = {}

    def connect(self) -> bool:
        """
        Connect to the oscilloscope.

        Returns:
            True if connection successful

        Raises:
            ConnectionError: If connection fails
        """
        if not PYVISA_AVAILABLE:
            raise ConnectionError(
                "pyvisa package not installed. Install with: pip install pyvisa pyvisa-py",
                self.instrument_id,
            )

        self.status = InstrumentStatus.CONNECTING
        logger.info("Connecting to Rigol oscilloscope")

        try:
            # Create resource manager
            self._rm = pyvisa.ResourceManager()

            # Build resource name if needed
            if not self._resource_name and self._host:
                # Use VXI-11 for Ethernet
                self._resource_name = f"TCPIP::{self._host}::INSTR"

            if not self._resource_name:
                # Try to find Rigol device
                resources = self._rm.list_resources()
                rigol_resources = [r for r in resources if "1AB1" in r]  # Rigol vendor ID

                if not rigol_resources:
                    raise ConnectionError(
                        "No Rigol oscilloscope found. Check connection.",
                        self.instrument_id,
                    )

                self._resource_name = rigol_resources[0]
                logger.info(f"Auto-detected Rigol at {self._resource_name}")

            # Open connection
            self._inst = self._rm.open_resource(
                self._resource_name,
                timeout=5000,  # 5 second timeout
            )

            # Set termination characters
            self._inst.read_termination = "\n"
            self._inst.write_termination = "\n"

            # Test connection with *IDN?
            idn = self._inst.query("*IDN?")
            if not idn or "RIGOL" not in idn.upper():
                raise ConnectionError(
                    f"Unexpected device response: {idn}",
                    self.instrument_id,
                )

            # Get device info
            self._info = self.get_info()

            self.status = InstrumentStatus.CONNECTED
            self._connected_at = datetime.now()

            logger.info(
                f"Connected to {self._info.manufacturer} {self._info.model} "
                f"(SN: {self._info.serial_number})"
            )

            return True

        except pyvisa.VisaIOError as e:
            self._handle_error(ConnectionError(f"VISA error: {e}", self.instrument_id))
            raise ConnectionError(str(e), self.instrument_id)
        except Exception as e:
            self._handle_error(ConnectionError(str(e), self.instrument_id))
            raise

    def disconnect(self) -> None:
        """Disconnect from the oscilloscope."""
        logger.info("Disconnecting from Rigol oscilloscope")

        try:
            if self._inst:
                try:
                    self._inst.close()
                except Exception:
                    pass
                self._inst = None

            if self._rm:
                try:
                    self._rm.close()
                except Exception:
                    pass
                self._rm = None

        except Exception as e:
            logger.error(f"Error during disconnect: {e}")
        finally:
            self.status = InstrumentStatus.DISCONNECTED

    def _write(self, cmd: str) -> None:
        """Send command to oscilloscope."""
        if not self._inst:
            raise CommunicationError("Not connected", self.instrument_id)
        self._inst.write(cmd)

    def _query(self, cmd: str) -> str:
        """Query oscilloscope and return response."""
        if not self._inst:
            raise CommunicationError("Not connected", self.instrument_id)
        return self._inst.query(cmd).strip()

    def _query_binary(self, cmd: str) -> bytes:
        """Query binary data."""
        if not self._inst:
            raise CommunicationError("Not connected", self.instrument_id)
        return self._inst.query_binary_values(cmd, datatype="B", container=bytes)

    def get_info(self) -> InstrumentInfo:
        """Get oscilloscope information."""
        idn = self._query("*IDN?")
        parts = idn.split(",")

        manufacturer = parts[0].strip() if len(parts) > 0 else "Rigol"
        model = parts[1].strip() if len(parts) > 1 else "Unknown"
        serial = parts[2].strip() if len(parts) > 2 else None
        firmware = parts[3].strip() if len(parts) > 3 else None

        # Determine number of channels
        if "1054" in model or "DS1" in model:
            self._num_channels = 4
            bandwidth = 50e6 if "54" in model else 100e6
        elif "DS2" in model:
            self._num_channels = 2 if "2072" in model or "2102" in model else 4
            bandwidth = 200e6
        elif "MSO5" in model or "DS4" in model:
            self._num_channels = 4
            bandwidth = 500e6
        else:
            self._num_channels = 4
            bandwidth = 100e6

        capabilities = [
            InstrumentCapability(
                name="analog_channels",
                type="feature",
                parameters={"count": self._num_channels},
            ),
            InstrumentCapability(
                name="bandwidth",
                type="range",
                parameters={"max": bandwidth, "unit": "Hz"},
            ),
            InstrumentCapability(
                name="sample_rate",
                type="range",
                parameters={"max": 1e9, "unit": "Sa/s"},
            ),
            InstrumentCapability(
                name="memory_depth",
                type="range",
                parameters={"max": 24e6, "unit": "pts"},
            ),
        ]

        return InstrumentInfo(
            instrument_type=InstrumentType.OSCILLOSCOPE,
            manufacturer=manufacturer,
            model=model,
            serial_number=serial,
            firmware_version=firmware,
            connection_type=ConnectionType.USB if "USB" in (self._resource_name or "") else ConnectionType.ETHERNET,
            connection_params={
                "resource_name": self._resource_name,
                "host": self._host,
            },
            capabilities=capabilities,
            metadata={
                "num_channels": self._num_channels,
                "bandwidth_hz": bandwidth,
            },
        )

    def reset(self) -> None:
        """Reset oscilloscope to default state."""
        self._write("*RST")
        time.sleep(1)  # Wait for reset
        self._write("*CLS")  # Clear status
        logger.info("Oscilloscope reset to defaults")

    def self_test(self) -> Tuple[bool, Optional[str]]:
        """Run self-test."""
        try:
            result = self._query("*TST?")
            passed = result == "0"
            return passed, None if passed else f"Self-test failed: {result}"
        except Exception as e:
            return False, str(e)

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
        """Configure an oscilloscope channel."""
        # Normalize channel name
        ch = channel.upper().replace("CH", "CHAN")
        if not ch.startswith("CHAN"):
            ch = f"CHAN{channel}"

        # Enable/disable
        self._write(f":{ch}:DISP {'ON' if enabled else 'OFF'}")

        if enabled:
            # Coupling
            coupling_mode = coupling.upper()
            if coupling_mode in ("DC", "AC", "GND"):
                self._write(f":{ch}:COUP {coupling_mode}")

            # Voltage scale (V/div)
            self._write(f":{ch}:SCAL {voltage_scale}")

            # Offset
            self._write(f":{ch}:OFFS {voltage_offset}")

            # Probe attenuation
            self._write(f":{ch}:PROB {probe_attenuation}")

            # Bandwidth limit
            if bandwidth_limit:
                if bandwidth_limit < 20e6:
                    self._write(f":{ch}:BWL 20M")
                else:
                    self._write(f":{ch}:BWL OFF")

        # Store configuration
        self._channel_config[channel] = {
            "enabled": enabled,
            "coupling": coupling,
            "voltage_scale": voltage_scale,
            "voltage_offset": voltage_offset,
            "probe_attenuation": probe_attenuation,
            "bandwidth_limit": bandwidth_limit,
        }

        logger.info(f"Configured {channel}: scale={voltage_scale}V/div, offset={voltage_offset}V")

    def configure_timebase(
        self,
        time_scale: float,
        time_offset: float = 0.0,
        sample_rate: Optional[float] = None,
    ) -> None:
        """Configure timebase settings."""
        # Time scale (s/div)
        self._write(f":TIM:SCAL {time_scale}")

        # Time offset
        self._write(f":TIM:OFFS {time_offset}")

        # Sample rate is typically auto, but can be set via memory depth
        if sample_rate:
            # Calculate memory depth for desired sample rate
            # memory_depth = sample_rate * (time_scale * 12)
            pass

        logger.info(f"Configured timebase: {time_scale}s/div, offset={time_offset}s")

    def configure_trigger(
        self,
        source: str,
        level: float,
        edge: str = "rising",
        mode: str = "auto",
        holdoff: float = 0.0,
        position: float = 0.5,
    ) -> None:
        """Configure trigger settings."""
        # Trigger source
        src = source.upper().replace("CH", "CHAN")
        if not src.startswith("CHAN") and src not in ("EXT", "LINE", "WGEN"):
            src = f"CHAN{source}"
        self._write(f":TRIG:EDG:SOUR {src}")

        # Trigger level
        self._write(f":TRIG:EDG:LEV {level}")

        # Trigger edge
        edge_mode = "POS" if edge.lower() in ("rising", "pos", "positive") else "NEG"
        self._write(f":TRIG:EDG:SLOP {edge_mode}")

        # Trigger mode
        mode_map = {"auto": "AUTO", "normal": "NORM", "single": "SING"}
        self._write(f":TRIG:SWE {mode_map.get(mode.lower(), 'AUTO')}")

        # Trigger holdoff
        if holdoff > 0:
            self._write(f":TRIG:HOLD {holdoff}")

        # Trigger position (horizontal)
        # Position 0.5 means trigger at center of screen
        # Rigol uses time offset relative to trigger point
        # This is handled via timebase offset

        logger.info(f"Configured trigger: {source} {edge} @ {level}V, mode={mode}")

    def arm_trigger(self) -> None:
        """Arm the trigger and prepare for acquisition."""
        self._write(":SING")  # Single acquisition mode
        logger.debug("Trigger armed")

    def wait_for_trigger(self, timeout_s: float = 10.0) -> bool:
        """Wait for trigger event."""
        start_time = time.time()

        while time.time() - start_time < timeout_s:
            status = self._query(":TRIG:STAT?")
            if status in ("STOP", "TD"):  # Triggered
                return True
            elif status == "AUTO":  # Auto triggered
                return True
            time.sleep(0.1)

        logger.warning("Trigger timeout")
        return False

    def get_waveform(
        self,
        channels: List[str],
        num_points: Optional[int] = None,
    ) -> Dict[str, Tuple[List[float], List[float]]]:
        """
        Get waveform data from specified channels.

        Returns:
            Dict mapping channel to (time_array, voltage_array)
        """
        self.status = InstrumentStatus.BUSY
        result: Dict[str, Tuple[List[float], List[float]]] = {}

        try:
            # Stop acquisition to read data
            self._write(":STOP")
            time.sleep(0.2)

            for channel in channels:
                ch = channel.upper().replace("CH", "CHAN")
                if not ch.startswith("CHAN"):
                    ch = f"CHAN{channel}"

                # Set waveform source
                self._write(f":WAV:SOUR {ch}")

                # Set waveform mode to RAW for full data
                self._write(":WAV:MODE RAW")

                # Set format to BYTE for efficiency
                self._write(":WAV:FORM BYTE")

                # Set read points
                if num_points:
                    self._write(f":WAV:POIN {num_points}")
                else:
                    self._write(":WAV:POIN MAX")

                # Get waveform preamble
                preamble = self._query(":WAV:PRE?")
                params = preamble.split(",")

                # Parse preamble
                # format, type, points, count, xincrement, xorigin, xreference, yincrement, yorigin, yreference
                points = int(params[2])
                x_increment = float(params[4])
                x_origin = float(params[5])
                x_reference = float(params[6])
                y_increment = float(params[7])
                y_origin = float(params[8])
                y_reference = float(params[9])

                # Get waveform data
                self._write(":WAV:DATA?")
                raw_data = self._inst.read_raw()

                # Parse TMC header
                # Format: #NXXXXXXXX<data>
                if raw_data[0:1] == b"#":
                    header_len = int(raw_data[1:2])
                    data_len = int(raw_data[2 : 2 + header_len])
                    data_start = 2 + header_len
                    raw_values = raw_data[data_start : data_start + data_len]
                else:
                    raw_values = raw_data

                # Convert to voltage values
                time_data = []
                voltage_data = []

                for i, byte_val in enumerate(raw_values):
                    t = (i - x_reference) * x_increment + x_origin
                    v = (byte_val - y_reference - y_origin) * y_increment
                    time_data.append(t)
                    voltage_data.append(v)

                result[channel] = (time_data, voltage_data)
                logger.debug(f"Retrieved {len(voltage_data)} samples from {channel}")

            return result

        except Exception as e:
            raise CommunicationError(f"Failed to get waveform: {e}", self.instrument_id)
        finally:
            self.status = InstrumentStatus.CONNECTED

    def measure(
        self,
        channel: str,
        measurement_type: str,
    ) -> MeasurementResult:
        """Perform automatic measurement on a channel."""
        ch = channel.upper().replace("CH", "CHAN")
        if not ch.startswith("CHAN"):
            ch = f"CHAN{channel}"

        # Map measurement type to SCPI command
        meas_type = self.MEASUREMENT_TYPES.get(
            measurement_type.lower(),
            measurement_type.upper(),
        )

        # Perform measurement
        value_str = self._query(f":MEAS:ITEM? {meas_type},{ch}")

        try:
            value = float(value_str)
        except ValueError:
            value = float("nan")

        # Determine unit based on measurement type
        unit_map = {
            "frequency": "Hz",
            "period": "s",
            "vrms": "V",
            "vpp": "V",
            "vmax": "V",
            "vmin": "V",
            "vavg": "V",
            "vtop": "V",
            "vbase": "V",
            "vamplitude": "V",
            "rise_time": "s",
            "fall_time": "s",
            "positive_width": "s",
            "negative_width": "s",
            "duty_cycle": "%",
            "overshoot": "%",
            "preshoot": "%",
        }
        unit = unit_map.get(measurement_type.lower(), "")

        return MeasurementResult(
            measurement_type=measurement_type,
            value=value,
            unit=unit,
            channel=channel,
            timestamp=datetime.now(),
        )

    def get_screenshot(self, format: str = "png") -> bytes:
        """Capture screenshot from oscilloscope display."""
        # Set format
        if format.lower() == "bmp":
            self._write(":DISP:DATA? ON,OFF,BMP")
        else:
            self._write(":DISP:DATA? ON,OFF,PNG")

        # Read binary data
        raw_data = self._inst.read_raw()

        # Parse TMC header
        if raw_data[0:1] == b"#":
            header_len = int(raw_data[1:2])
            data_len = int(raw_data[2 : 2 + header_len])
            data_start = 2 + header_len
            image_data = raw_data[data_start : data_start + data_len]
        else:
            image_data = raw_data

        logger.info(f"Captured screenshot ({len(image_data)} bytes)")
        return image_data


# Discovery function
def discover_rigol_oscilloscopes() -> List[InstrumentInfo]:
    """Discover connected Rigol oscilloscopes."""
    if not PYVISA_AVAILABLE:
        return []

    discovered: List[InstrumentInfo] = []

    try:
        rm = pyvisa.ResourceManager()
        resources = rm.list_resources()

        for resource in resources:
            if "1AB1" not in resource:  # Rigol vendor ID
                continue

            try:
                inst = rm.open_resource(resource, timeout=2000)
                idn = inst.query("*IDN?").strip()
                inst.close()

                parts = idn.split(",")
                if len(parts) >= 3 and "RIGOL" in parts[0].upper():
                    model = parts[1].strip()
                    if "DS" in model or "MSO" in model:
                        discovered.append(InstrumentInfo(
                            instrument_type=InstrumentType.OSCILLOSCOPE,
                            manufacturer="Rigol",
                            model=model,
                            serial_number=parts[2].strip() if len(parts) > 2 else None,
                            firmware_version=parts[3].strip() if len(parts) > 3 else None,
                            connection_type=ConnectionType.USB if "USB" in resource else ConnectionType.ETHERNET,
                            connection_params={"resource_name": resource},
                        ))

            except Exception as e:
                logger.debug(f"Failed to query {resource}: {e}")

        rm.close()

    except Exception as e:
        logger.debug(f"Rigol oscilloscope discovery failed: {e}")

    return discovered
