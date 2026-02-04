"""
Rigol Power Supply Driver

This driver provides integration with Rigol DP series programmable power supplies
via PyVISA using SCPI commands.

Supported Models:
- DP832 (3 channels: 30V/3A, 30V/3A, 5V/3A)
- DP832A (3 channels with better accuracy)
- DP831 (3 channels: 8V/5A, 30V/2A, -30V/2A)
- DP821 (2 channels: 60V/1A, 8V/10A)
- DP811 (1 channel: 20V/10A or 40V/5A)

Requirements:
- pyvisa: pip install pyvisa
- pyvisa-py or NI-VISA backend

Connection:
- USB: Connect via USB-TMC
- Ethernet: Connect via LXI/VXI-11
"""

import logging
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from ...base_instrument import (
    PowerSupply,
    InstrumentInfo,
    InstrumentType,
    ConnectionType,
    InstrumentStatus,
    InstrumentCapability,
    ConnectionError,
    CommunicationError,
)

logger = logging.getLogger(__name__)

# Try to import pyvisa
try:
    import pyvisa
    PYVISA_AVAILABLE = True
except ImportError:
    PYVISA_AVAILABLE = False
    logger.warning("pyvisa package not installed. Rigol power supply features will be unavailable.")


class RigolPowerSupply(PowerSupply):
    """
    Driver for Rigol DP series programmable power supplies.

    Provides:
    - Multi-channel voltage and current control
    - Output enable/disable per channel
    - Real-time voltage, current, and power monitoring
    - OVP/OCP protection settings
    - Timed output control
    """

    # Channel specifications by model
    MODEL_SPECS = {
        "DP832": {
            "channels": 3,
            "ch1": {"v_max": 30, "i_max": 3},
            "ch2": {"v_max": 30, "i_max": 3},
            "ch3": {"v_max": 5, "i_max": 3},
        },
        "DP832A": {
            "channels": 3,
            "ch1": {"v_max": 30, "i_max": 3},
            "ch2": {"v_max": 30, "i_max": 3},
            "ch3": {"v_max": 5, "i_max": 3},
        },
        "DP831": {
            "channels": 3,
            "ch1": {"v_max": 8, "i_max": 5},
            "ch2": {"v_max": 30, "i_max": 2},
            "ch3": {"v_max": -30, "i_max": 2},
        },
        "DP821": {
            "channels": 2,
            "ch1": {"v_max": 60, "i_max": 1},
            "ch2": {"v_max": 8, "i_max": 10},
        },
        "DP811": {
            "channels": 1,
            "ch1": {"v_max": 40, "i_max": 10},
        },
    }

    def __init__(
        self,
        instrument_id: Optional[str] = None,
        connection_params: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize the Rigol power supply driver.

        Args:
            instrument_id: Unique identifier for this instance
            connection_params: Connection parameters:
                - resource_name: VISA resource name
                - host: IP address for Ethernet connection
        """
        super().__init__(instrument_id, connection_params)

        self._resource_name = connection_params.get("resource_name") if connection_params else None
        self._host = connection_params.get("host") if connection_params else None

        # VISA objects
        self._rm: Optional[Any] = None
        self._inst: Optional[Any] = None

        # Device info
        self._model: str = ""
        self._num_channels: int = 3
        self._specs: Dict[str, Any] = {}

    def connect(self) -> bool:
        """
        Connect to the power supply.

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
        logger.info("Connecting to Rigol power supply")

        try:
            self._rm = pyvisa.ResourceManager()

            # Build resource name if needed
            if not self._resource_name and self._host:
                self._resource_name = f"TCPIP::{self._host}::INSTR"

            if not self._resource_name:
                # Auto-detect Rigol PSU
                resources = self._rm.list_resources()
                for r in resources:
                    if "1AB1" in r:  # Rigol vendor ID
                        try:
                            test_inst = self._rm.open_resource(r, timeout=2000)
                            idn = test_inst.query("*IDN?")
                            test_inst.close()
                            if "DP8" in idn:
                                self._resource_name = r
                                break
                        except Exception:
                            continue

                if not self._resource_name:
                    raise ConnectionError(
                        "No Rigol power supply found",
                        self.instrument_id,
                    )

            # Open connection
            self._inst = self._rm.open_resource(
                self._resource_name,
                timeout=5000,
            )
            self._inst.read_termination = "\n"
            self._inst.write_termination = "\n"

            # Verify device
            idn = self._inst.query("*IDN?")
            if not idn or "RIGOL" not in idn.upper() or "DP" not in idn.upper():
                raise ConnectionError(
                    f"Unexpected device: {idn}",
                    self.instrument_id,
                )

            # Parse model
            parts = idn.split(",")
            self._model = parts[1].strip() if len(parts) > 1 else "DP832"

            # Get specifications
            for model_key, specs in self.MODEL_SPECS.items():
                if model_key in self._model:
                    self._specs = specs
                    self._num_channels = specs["channels"]
                    break
            else:
                # Default to DP832 specs
                self._specs = self.MODEL_SPECS["DP832"]
                self._num_channels = 3

            self._info = self.get_info()

            self.status = InstrumentStatus.CONNECTED
            self._connected_at = datetime.now()

            logger.info(
                f"Connected to {self._info.manufacturer} {self._info.model} "
                f"({self._num_channels} channels)"
            )

            return True

        except pyvisa.VisaIOError as e:
            self._handle_error(ConnectionError(f"VISA error: {e}", self.instrument_id))
            raise ConnectionError(str(e), self.instrument_id)
        except Exception as e:
            self._handle_error(ConnectionError(str(e), self.instrument_id))
            raise

    def disconnect(self) -> None:
        """Disconnect from the power supply."""
        logger.info("Disconnecting from Rigol power supply")

        try:
            # Optionally disable outputs before disconnect
            # self.enable_all(False)

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
        """Send command to power supply."""
        if not self._inst:
            raise CommunicationError("Not connected", self.instrument_id)
        self._inst.write(cmd)

    def _query(self, cmd: str) -> str:
        """Query power supply and return response."""
        if not self._inst:
            raise CommunicationError("Not connected", self.instrument_id)
        return self._inst.query(cmd).strip()

    def get_info(self) -> InstrumentInfo:
        """Get power supply information."""
        idn = self._query("*IDN?")
        parts = idn.split(",")

        manufacturer = parts[0].strip() if len(parts) > 0 else "Rigol"
        model = parts[1].strip() if len(parts) > 1 else "DP832"
        serial = parts[2].strip() if len(parts) > 2 else None
        firmware = parts[3].strip() if len(parts) > 3 else None

        capabilities = []
        for ch in range(1, self._num_channels + 1):
            ch_specs = self._specs.get(f"ch{ch}", {"v_max": 30, "i_max": 3})
            capabilities.append(
                InstrumentCapability(
                    name=f"channel_{ch}",
                    type="range",
                    parameters={
                        "v_max": ch_specs["v_max"],
                        "i_max": ch_specs["i_max"],
                    },
                )
            )

        return InstrumentInfo(
            instrument_type=InstrumentType.POWER_SUPPLY,
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
                "specs": self._specs,
            },
        )

    def reset(self) -> None:
        """Reset power supply to default state."""
        self._write("*RST")
        time.sleep(1)
        self._write("*CLS")
        logger.info("Power supply reset")

    def self_test(self) -> Tuple[bool, Optional[str]]:
        """Run self-test."""
        try:
            result = self._query("*TST?")
            passed = result == "0"
            return passed, None if passed else f"Self-test failed: {result}"
        except Exception as e:
            return False, str(e)

    def get_num_channels(self) -> int:
        """Get number of output channels."""
        return self._num_channels

    def _validate_channel(self, channel: int) -> None:
        """Validate channel number."""
        if channel < 1 or channel > self._num_channels:
            raise ValueError(f"Invalid channel {channel}. Must be 1-{self._num_channels}")

    def set_voltage(self, channel: int, voltage: float) -> None:
        """
        Set output voltage.

        Args:
            channel: Channel number (1-based)
            voltage: Target voltage in volts
        """
        self._validate_channel(channel)

        ch_specs = self._specs.get(f"ch{channel}", {})
        v_max = ch_specs.get("v_max", 30)

        if abs(voltage) > abs(v_max):
            raise ValueError(f"Voltage {voltage}V exceeds maximum {v_max}V for channel {channel}")

        self._write(f":SOUR{channel}:VOLT {voltage}")
        logger.debug(f"Set CH{channel} voltage to {voltage}V")

    def set_current_limit(self, channel: int, current: float) -> None:
        """
        Set current limit.

        Args:
            channel: Channel number (1-based)
            current: Current limit in amps
        """
        self._validate_channel(channel)

        ch_specs = self._specs.get(f"ch{channel}", {})
        i_max = ch_specs.get("i_max", 3)

        if current > i_max:
            raise ValueError(f"Current {current}A exceeds maximum {i_max}A for channel {channel}")

        self._write(f":SOUR{channel}:CURR {current}")
        logger.debug(f"Set CH{channel} current limit to {current}A")

    def enable_output(self, channel: int, enabled: bool = True) -> None:
        """
        Enable or disable channel output.

        Args:
            channel: Channel number (1-based)
            enabled: Whether to enable output
        """
        self._validate_channel(channel)

        self._write(f":OUTP CH{channel},{'ON' if enabled else 'OFF'}")
        logger.info(f"CH{channel} output {'enabled' if enabled else 'disabled'}")

    def get_voltage(self, channel: int) -> float:
        """
        Read actual output voltage.

        Args:
            channel: Channel number (1-based)

        Returns:
            Measured voltage in volts
        """
        self._validate_channel(channel)
        value = self._query(f":MEAS:VOLT? CH{channel}")
        return float(value)

    def get_current(self, channel: int) -> float:
        """
        Read actual output current.

        Args:
            channel: Channel number (1-based)

        Returns:
            Measured current in amps
        """
        self._validate_channel(channel)
        value = self._query(f":MEAS:CURR? CH{channel}")
        return float(value)

    def get_power(self, channel: int) -> float:
        """
        Get output power.

        Args:
            channel: Channel number (1-based)

        Returns:
            Output power in watts
        """
        self._validate_channel(channel)
        value = self._query(f":MEAS:POWE? CH{channel}")
        return float(value)

    def get_mode(self, channel: int) -> str:
        """
        Get current operating mode.

        Args:
            channel: Channel number (1-based)

        Returns:
            Mode string ('cv', 'cc', 'off', 'unregulated')
        """
        self._validate_channel(channel)

        # Check if output is enabled
        if not self.is_output_enabled(channel):
            return "off"

        # Query CV/CC status
        try:
            mode = self._query(f":SOUR{channel}:MODE?")
            return mode.lower()
        except Exception:
            # Fallback: compare set vs actual values
            voltage_set = float(self._query(f":SOUR{channel}:VOLT?"))
            voltage_actual = self.get_voltage(channel)
            current_set = float(self._query(f":SOUR{channel}:CURR?"))
            current_actual = self.get_current(channel)

            if abs(current_actual - current_set) < 0.01:
                return "cc"
            elif abs(voltage_actual - voltage_set) < 0.1:
                return "cv"
            else:
                return "unregulated"

    def is_output_enabled(self, channel: int) -> bool:
        """
        Check if output is enabled.

        Args:
            channel: Channel number (1-based)

        Returns:
            True if output is enabled
        """
        self._validate_channel(channel)
        result = self._query(f":OUTP? CH{channel}")
        return result.upper() in ("ON", "1")

    def get_voltage_setpoint(self, channel: int) -> float:
        """Get the voltage setpoint."""
        self._validate_channel(channel)
        value = self._query(f":SOUR{channel}:VOLT?")
        return float(value)

    def get_current_setpoint(self, channel: int) -> float:
        """Get the current limit setpoint."""
        self._validate_channel(channel)
        value = self._query(f":SOUR{channel}:CURR?")
        return float(value)

    def set_ovp(self, channel: int, voltage: float, enabled: bool = True) -> None:
        """
        Set over-voltage protection.

        Args:
            channel: Channel number
            voltage: OVP threshold voltage
            enabled: Enable OVP
        """
        self._validate_channel(channel)
        self._write(f":SOUR{channel}:VOLT:PROT {voltage}")
        self._write(f":SOUR{channel}:VOLT:PROT:STAT {'ON' if enabled else 'OFF'}")
        logger.info(f"CH{channel} OVP set to {voltage}V, {'enabled' if enabled else 'disabled'}")

    def set_ocp(self, channel: int, current: float, enabled: bool = True) -> None:
        """
        Set over-current protection.

        Args:
            channel: Channel number
            current: OCP threshold current
            enabled: Enable OCP
        """
        self._validate_channel(channel)
        self._write(f":SOUR{channel}:CURR:PROT {current}")
        self._write(f":SOUR{channel}:CURR:PROT:STAT {'ON' if enabled else 'OFF'}")
        logger.info(f"CH{channel} OCP set to {current}A, {'enabled' if enabled else 'disabled'}")

    def clear_protection(self, channel: int) -> None:
        """Clear protection status (tripped OVP/OCP)."""
        self._validate_channel(channel)
        self._write(f":OUTP:PROT:CLE CH{channel}")
        logger.info(f"CH{channel} protection cleared")

    def get_all_measurements(self) -> Dict[int, Dict[str, float]]:
        """
        Get all measurements from all channels.

        Returns:
            Dict mapping channel number to measurements
        """
        result = {}
        for ch in range(1, self._num_channels + 1):
            result[ch] = {
                "voltage": self.get_voltage(ch),
                "current": self.get_current(ch),
                "power": self.get_power(ch),
                "enabled": self.is_output_enabled(ch),
            }
        return result


# Discovery function
def discover_rigol_power_supplies() -> List[InstrumentInfo]:
    """Discover connected Rigol power supplies."""
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
                if len(parts) >= 2 and "RIGOL" in parts[0].upper():
                    model = parts[1].strip()
                    if "DP8" in model:
                        discovered.append(InstrumentInfo(
                            instrument_type=InstrumentType.POWER_SUPPLY,
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
        logger.debug(f"Rigol power supply discovery failed: {e}")

    return discovered
