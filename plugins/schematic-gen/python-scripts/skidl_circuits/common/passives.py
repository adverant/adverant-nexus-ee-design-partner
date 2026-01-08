"""
Passive Component Definitions for SKiDL

Provides factory functions for creating resistors, capacitors, inductors,
and other passive components with proper footprints and tolerances.
"""

from typing import Optional, Tuple
from enum import Enum

try:
    from skidl import Part, Net, TEMPLATE
except ImportError:
    class Part:
        def __init__(self, *args, **kwargs):
            self.ref = kwargs.get('ref', 'X?')
            self.value = kwargs.get('value', '')
        def __getitem__(self, key):
            return Net()
        def __setitem__(self, key, value):
            pass
    class Net:
        def __iadd__(self, other):
            return self
    TEMPLATE = 'TEMPLATE'


class SMDSize(Enum):
    """Standard SMD package sizes."""
    R0201 = ('0201', 'Resistor_SMD:R_0201_0603Metric')
    R0402 = ('0402', 'Resistor_SMD:R_0402_1005Metric')
    R0603 = ('0603', 'Resistor_SMD:R_0603_1608Metric')
    R0805 = ('0805', 'Resistor_SMD:R_0805_2012Metric')
    R1206 = ('1206', 'Resistor_SMD:R_1206_3216Metric')
    R2010 = ('2010', 'Resistor_SMD:R_2010_5025Metric')
    R2512 = ('2512', 'Resistor_SMD:R_2512_6332Metric')


class CapSize(Enum):
    """Standard capacitor package sizes."""
    C0201 = ('0201', 'Capacitor_SMD:C_0201_0603Metric')
    C0402 = ('0402', 'Capacitor_SMD:C_0402_1005Metric')
    C0603 = ('0603', 'Capacitor_SMD:C_0603_1608Metric')
    C0805 = ('0805', 'Capacitor_SMD:C_0805_2012Metric')
    C1206 = ('1206', 'Capacitor_SMD:C_1206_3216Metric')
    C1210 = ('1210', 'Capacitor_SMD:C_1210_3225Metric')
    C1812 = ('1812', 'Capacitor_SMD:C_1812_4532Metric')


class Passives:
    """
    Factory class for creating passive components.

    Provides methods for creating resistors, capacitors, inductors,
    and ferrite beads with appropriate footprints and specifications.

    Design Notes (MIT EE perspective):
    - Resistors: Use 0603 for general purpose, 0805+ for power
    - Capacitors: Use X7R/X5R for decoupling, C0G for timing circuits
    - Gate resistors: Matched values (±1%) for parallel MOSFETs
    - Current sense: 4-terminal shunts for Kelvin connection
    """

    def __init__(self):
        self._ref_counter = {
            'R': 0,
            'C': 0,
            'L': 0,
            'FB': 0
        }

    def _next_ref(self, prefix: str) -> str:
        """Generate next reference designator."""
        self._ref_counter[prefix] = self._ref_counter.get(prefix, 0) + 1
        return f"{prefix}{self._ref_counter[prefix]}"

    def resistor(
        self,
        value: str,
        ref: Optional[str] = None,
        size: SMDSize = SMDSize.R0603,
        power_rating: str = '0.1W',
        tolerance: str = '1%'
    ) -> Part:
        """
        Create a resistor.

        Args:
            value: Resistance value (e.g., '10k', '4.7', '100R')
            ref: Reference designator (auto-generated if None)
            size: SMD package size
            power_rating: Power rating
            tolerance: Tolerance

        Returns:
            SKiDL Part for the resistor
        """
        if ref is None:
            ref = self._next_ref('R')

        return Part(
            'Device',
            'R',
            ref=ref,
            value=value,
            footprint=size.value[1],
            power_rating=power_rating,
            tolerance=tolerance,
            dest=TEMPLATE
        )

    def gate_resistor(
        self,
        value: str = '4.7',
        ref: Optional[str] = None,
        size: SMDSize = SMDSize.R0805
    ) -> Part:
        """
        Create a gate resistor for MOSFET drive.

        Design Note: Gate resistors for parallel MOSFETs MUST be matched
        to within ±1% to ensure equal current sharing during switching.
        Use 0805 for better thermal handling of gate drive dissipation.

        Args:
            value: Resistance in ohms (4.7-10 ohms typical for SiC)
            ref: Reference designator
            size: Package size (0805 recommended)

        Returns:
            SKiDL Part for gate resistor
        """
        if ref is None:
            ref = self._next_ref('R')

        return Part(
            'Device',
            'R',
            ref=ref,
            value=value,
            footprint=size.value[1],
            power_rating='0.25W',
            tolerance='1%',
            dest=TEMPLATE
        )

    def shunt_resistor(
        self,
        value: str,
        ref: Optional[str] = None,
        four_terminal: bool = True
    ) -> Part:
        """
        Create a current sense shunt resistor.

        Design Note: For high-accuracy current sensing, use 4-terminal
        (Kelvin) shunts. The sense pins (3,4) are separate from the
        power path (1,2), eliminating contact resistance errors.

        Args:
            value: Resistance value (e.g., '0.5m', '1m')
            ref: Reference designator
            four_terminal: Use 4-terminal Kelvin package

        Returns:
            SKiDL Part for shunt resistor
        """
        if ref is None:
            ref = self._next_ref('R')

        footprint = (
            'Resistor_SMD:R_Shunt_Vishay_WSK2512_6332Metric'
            if four_terminal else
            'Resistor_SMD:R_2512_6332Metric'
        )

        part = Part(
            'Device',
            'R_Shunt' if four_terminal else 'R',
            ref=ref,
            value=value,
            footprint=footprint,
            power_rating='3W',
            tolerance='1%',
            dest=TEMPLATE
        )

        return part

    def capacitor(
        self,
        value: str,
        ref: Optional[str] = None,
        size: CapSize = CapSize.C0603,
        voltage_rating: str = '25V',
        dielectric: str = 'X7R'
    ) -> Part:
        """
        Create a ceramic capacitor.

        Args:
            value: Capacitance value (e.g., '100nF', '10uF')
            ref: Reference designator
            size: SMD package size
            voltage_rating: Voltage rating
            dielectric: Dielectric type (X7R, X5R, C0G)

        Returns:
            SKiDL Part for the capacitor
        """
        if ref is None:
            ref = self._next_ref('C')

        return Part(
            'Device',
            'C',
            ref=ref,
            value=value,
            footprint=size.value[1],
            voltage_rating=voltage_rating,
            dielectric=dielectric,
            dest=TEMPLATE
        )

    def decoupling_cap(
        self,
        value: str = '100nF',
        ref: Optional[str] = None,
        size: CapSize = CapSize.C0603
    ) -> Part:
        """
        Create a decoupling capacitor.

        Design Note: 100nF X7R is the standard decoupling value.
        Place as close as possible to IC power pins.
        For ICs with multiple VDD pins, use one cap per pin.

        Args:
            value: Capacitance (100nF default)
            ref: Reference designator
            size: Package size

        Returns:
            SKiDL Part for decoupling cap
        """
        return self.capacitor(
            value=value,
            ref=ref,
            size=size,
            voltage_rating='25V',
            dielectric='X7R'
        )

    def bulk_capacitor(
        self,
        value: str,
        ref: Optional[str] = None,
        voltage_rating: str = '100V'
    ) -> Part:
        """
        Create an electrolytic bulk capacitor.

        Design Note: For DC bus capacitors, use low-ESR electrolytics
        rated for 2x the expected voltage. ESR and ESL are critical
        for minimizing voltage ripple during switching.

        Args:
            value: Capacitance (e.g., '350uF')
            ref: Reference designator
            voltage_rating: Voltage rating

        Returns:
            SKiDL Part for bulk capacitor
        """
        if ref is None:
            ref = self._next_ref('C')

        return Part(
            'Device',
            'CP',
            ref=ref,
            value=value,
            footprint='Capacitor_THT:CP_Radial_D35.0mm_P10mm',
            voltage_rating=voltage_rating,
            dest=TEMPLATE
        )

    def bootstrap_cap(
        self,
        value: str = '1uF',
        ref: Optional[str] = None
    ) -> Part:
        """
        Create a bootstrap capacitor for high-side gate drivers.

        Design Note: Bootstrap cap must be sized for at least 10x
        the total gate charge: C_boot >= (Q_gate × 10) / ΔV
        For IMZA65R027M1H (150nC), minimum is 3µF with 0.5V drop.

        Args:
            value: Capacitance (1µF minimum recommended)
            ref: Reference designator

        Returns:
            SKiDL Part for bootstrap cap
        """
        return self.capacitor(
            value=value,
            ref=ref,
            size=CapSize.C0805,
            voltage_rating='25V',
            dielectric='X7R'
        )

    def inductor(
        self,
        value: str,
        ref: Optional[str] = None,
        current_rating: str = '1A',
        footprint: str = 'Inductor_SMD:L_1210_3225Metric'
    ) -> Part:
        """
        Create an inductor.

        Args:
            value: Inductance value (e.g., '4.7uH', '10uH')
            ref: Reference designator
            current_rating: DC current rating
            footprint: Package footprint

        Returns:
            SKiDL Part for the inductor
        """
        if ref is None:
            ref = self._next_ref('L')

        return Part(
            'Device',
            'L',
            ref=ref,
            value=value,
            footprint=footprint,
            current_rating=current_rating,
            dest=TEMPLATE
        )

    def ferrite_bead(
        self,
        value: str = '600R@100MHz',
        ref: Optional[str] = None,
        size: SMDSize = SMDSize.R0805
    ) -> Part:
        """
        Create a ferrite bead for noise filtering.

        Args:
            value: Impedance at frequency
            ref: Reference designator
            size: Package size

        Returns:
            SKiDL Part for ferrite bead
        """
        if ref is None:
            ref = self._next_ref('FB')

        return Part(
            'Device',
            'Ferrite_Bead',
            ref=ref,
            value=value,
            footprint=size.value[1].replace('Resistor', 'Inductor'),
            dest=TEMPLATE
        )


# Global instance
_passives: Optional[Passives] = None


def get_passives() -> Passives:
    """Get global Passives factory instance."""
    global _passives
    if _passives is None:
        _passives = Passives()
    return _passives
