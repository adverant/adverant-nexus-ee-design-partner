"""
Power Symbol Definitions for SKiDL

Provides standardized power symbols that match KiCad's power library.
These are used throughout the circuit for consistent power connections.
"""

from typing import Dict, Optional
try:
    from skidl import Part, Net, POWER, TEMPLATE
except ImportError:
    # Provide stub for when skidl is not installed
    class Part:
        def __init__(self, *args, **kwargs):
            pass
    class Net:
        def __init__(self, *args, **kwargs):
            self.drive = None
        def __iadd__(self, other):
            return self
    POWER = 'POWER'
    TEMPLATE = 'TEMPLATE'


class PowerSymbols:
    """
    Factory class for creating standardized power symbols.

    Usage:
        ps = PowerSymbols()
        vdd = ps.VDD()        # 3.3V digital supply
        gnd = ps.GND()        # Ground
        vbus = ps.VBUS()      # DC bus voltage
    """

    # Standard voltage rails for FOC ESC
    RAILS: Dict[str, Dict[str, float]] = {
        'VDD': {'voltage': 3.3, 'description': '3.3V Digital Supply'},
        'VDDA': {'voltage': 3.3, 'description': '3.3V Analog Supply'},
        'V5V0': {'voltage': 5.0, 'description': '5V Supply'},
        'V12V': {'voltage': 12.0, 'description': '12V Supply'},
        'VBUS': {'voltage': 48.0, 'description': 'DC Bus (48V nominal)'},
        'GND': {'voltage': 0.0, 'description': 'Ground'},
        'AGND': {'voltage': 0.0, 'description': 'Analog Ground'},
        'PGND': {'voltage': 0.0, 'description': 'Power Ground'},
    }

    def __init__(self, lib_path: str = 'power'):
        """
        Initialize power symbols factory.

        Args:
            lib_path: Path to KiCad power library
        """
        self.lib_path = lib_path
        self._nets: Dict[str, Net] = {}

    def _get_or_create_net(self, name: str) -> Net:
        """Get existing net or create new one."""
        if name not in self._nets:
            net = Net(name)
            net.drive = POWER
            self._nets[name] = net
        return self._nets[name]

    def VDD(self) -> Net:
        """3.3V digital supply net."""
        return self._get_or_create_net('VDD')

    def VDDA(self) -> Net:
        """3.3V analog supply net."""
        return self._get_or_create_net('VDDA')

    def V5V0(self) -> Net:
        """5V supply net."""
        return self._get_or_create_net('V5V0')

    def V12V(self) -> Net:
        """12V supply net."""
        return self._get_or_create_net('V12V')

    def VBUS(self) -> Net:
        """DC bus voltage net (48V nominal)."""
        return self._get_or_create_net('DC_BUS_P')

    def GND(self) -> Net:
        """Ground net."""
        return self._get_or_create_net('GND')

    def AGND(self) -> Net:
        """Analog ground net."""
        return self._get_or_create_net('AGND')

    def PGND(self) -> Net:
        """Power ground net."""
        return self._get_or_create_net('PGND')

    def DC_BUS_P(self) -> Net:
        """Positive DC bus net."""
        return self._get_or_create_net('DC_BUS_P')

    def DC_BUS_N(self) -> Net:
        """Negative DC bus net (connected to PGND)."""
        return self._get_or_create_net('DC_BUS_N')

    def create_power_symbol(self, name: str) -> Part:
        """
        Create a KiCad power symbol part.

        Args:
            name: Symbol name (e.g., 'VDD', 'GND')

        Returns:
            SKiDL Part representing the power symbol
        """
        return Part(
            self.lib_path,
            name,
            footprint='',  # Power symbols have no footprint
            dest=TEMPLATE
        )

    def get_all_nets(self) -> Dict[str, Net]:
        """Get all created power nets."""
        return self._nets.copy()


# Global instance for convenience
_power_symbols: Optional[PowerSymbols] = None


def get_power_symbols() -> PowerSymbols:
    """Get global PowerSymbols instance."""
    global _power_symbols
    if _power_symbols is None:
        _power_symbols = PowerSymbols()
    return _power_symbols
