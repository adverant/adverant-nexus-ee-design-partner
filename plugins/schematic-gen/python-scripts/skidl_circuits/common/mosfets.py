"""
MOSFET and Power Transistor Definitions for SKiDL

Provides factory functions for creating MOSFETs, IGBTs, and associated
gate driver circuits. Supports both discrete and integrated driver solutions.

This module is designed for general power electronics applications including:
- Motor drives (BLDC, PMSM, stepper)
- DC-DC converters
- Inverters and rectifiers
- High-side/low-side switches
"""

from typing import Optional, Tuple, List
from enum import Enum
from dataclasses import dataclass

try:
    from skidl import Part, Net, SubCircuit, Interface, TEMPLATE
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
    def SubCircuit(func):
        return func
    class Interface:
        def __init__(self, **kwargs):
            pass
    TEMPLATE = 'TEMPLATE'


class MOSFETType(Enum):
    """MOSFET technology types."""
    SILICON = 'Si'
    SILICON_CARBIDE = 'SiC'
    GALLIUM_NITRIDE = 'GaN'


class PackageType(Enum):
    """Common MOSFET package types."""
    TO247_3 = ('TO-247-3', 'Package_TO_SOT_THT:TO-247-3_Horizontal')
    TO247_4 = ('TO-247-4', 'Package_TO_SOT_THT:TO-247-4')  # Kelvin source
    TO220 = ('TO-220', 'Package_TO_SOT_THT:TO-220-3_Horizontal')
    D2PAK = ('D2PAK', 'Package_TO_SOT_SMD:D2PAK')
    LFPAK = ('LFPAK', 'Package_TO_SOT_SMD:LFPAK33')
    SOIC8 = ('SOIC-8', 'Package_SO:SOIC-8_3.9x4.9mm_P1.27mm')


@dataclass
class MOSFETSpecs:
    """MOSFET specifications for design calculations."""
    part_number: str
    technology: MOSFETType
    vds_max: float  # Maximum drain-source voltage
    ids_max: float  # Maximum continuous drain current
    rds_on: float   # On-resistance at 25°C (mΩ)
    qg_total: float # Total gate charge (nC)
    qgd: float      # Gate-drain charge (nC)
    vgs_th: float   # Gate threshold voltage
    package: PackageType
    has_kelvin_source: bool = False


# Common MOSFET part definitions
MOSFET_LIBRARY = {
    # SiC MOSFETs (high voltage, high performance)
    'IMZA65R027M1H': MOSFETSpecs(
        part_number='IMZA65R027M1H',
        technology=MOSFETType.SILICON_CARBIDE,
        vds_max=650,
        ids_max=100,
        rds_on=27,
        qg_total=150,
        qgd=35,
        vgs_th=3.5,
        package=PackageType.TO247_4,
        has_kelvin_source=True
    ),
    'C3M0065100K': MOSFETSpecs(
        part_number='C3M0065100K',
        technology=MOSFETType.SILICON_CARBIDE,
        vds_max=1000,
        ids_max=35,
        rds_on=65,
        qg_total=95,
        qgd=19,
        vgs_th=2.5,
        package=PackageType.TO247_4,
        has_kelvin_source=True
    ),
    # GaN FETs (high frequency)
    'EPC2034C': MOSFETSpecs(
        part_number='EPC2034C',
        technology=MOSFETType.GALLIUM_NITRIDE,
        vds_max=200,
        ids_max=48,
        rds_on=2.2,
        qg_total=15,
        qgd=4,
        vgs_th=1.4,
        package=PackageType.SOIC8,
        has_kelvin_source=False
    ),
    # Silicon MOSFETs (general purpose)
    'IRF3205': MOSFETSpecs(
        part_number='IRF3205',
        technology=MOSFETType.SILICON,
        vds_max=55,
        ids_max=110,
        rds_on=8,
        qg_total=146,
        qgd=50,
        vgs_th=4.0,
        package=PackageType.TO220,
        has_kelvin_source=False
    ),
    'BSC0500NSI': MOSFETSpecs(
        part_number='BSC0500NSI',
        technology=MOSFETType.SILICON,
        vds_max=30,
        ids_max=100,
        rds_on=0.5,
        qg_total=100,
        qgd=20,
        vgs_th=2.5,
        package=PackageType.LFPAK,
        has_kelvin_source=False
    )
}


class MOSFETs:
    """
    Factory class for creating MOSFET and driver circuits.

    Design Philosophy (MIT EE perspective):
    - Always use Kelvin source for high dV/dt applications
    - Match gate resistors for parallel devices
    - Size bootstrap caps for 10x gate charge minimum
    - Include dead-time control for half-bridges
    """

    def __init__(self):
        self._ref_counter = {
            'Q': 0,
            'U': 0
        }

    def _next_ref(self, prefix: str) -> str:
        """Generate next reference designator."""
        self._ref_counter[prefix] = self._ref_counter.get(prefix, 0) + 1
        return f"{prefix}{self._ref_counter[prefix]}"

    def mosfet(
        self,
        part_number: str = 'IMZA65R027M1H',
        ref: Optional[str] = None
    ) -> Part:
        """
        Create a MOSFET component.

        Args:
            part_number: MOSFET part number from library
            ref: Reference designator

        Returns:
            SKiDL Part for the MOSFET
        """
        if ref is None:
            ref = self._next_ref('Q')

        specs = MOSFET_LIBRARY.get(part_number)
        if specs is None:
            # Generic MOSFET if not in library
            return Part(
                'Transistor_FET',
                'NMOS',
                ref=ref,
                value=part_number,
                footprint='Package_TO_SOT_THT:TO-247-3_Horizontal',
                dest=TEMPLATE
            )

        return Part(
            'Transistor_FET',
            specs.part_number,
            ref=ref,
            value=specs.part_number,
            footprint=specs.package.value[1],
            dest=TEMPLATE
        )

    def gate_driver(
        self,
        driver_type: str = 'UCC21530',
        ref: Optional[str] = None,
        isolated: bool = True
    ) -> Part:
        """
        Create a gate driver IC.

        Common drivers:
        - UCC21530: Isolated dual-channel, 4A sink/source
        - IR2110: High/low side driver, bootstrap
        - FAN7390: Half-bridge driver with dead-time

        Args:
            driver_type: Driver IC part number
            ref: Reference designator
            isolated: Whether driver is isolated

        Returns:
            SKiDL Part for the gate driver
        """
        if ref is None:
            ref = self._next_ref('U')

        footprints = {
            'UCC21530': 'Package_SO:SOIC-16W_7.5x10.3mm_P1.27mm',
            'IR2110': 'Package_DIP:DIP-14_W7.62mm',
            'FAN7390': 'Package_SO:SOIC-8_3.9x4.9mm_P1.27mm',
            'SI8261': 'Package_SO:SOIC-8_3.9x4.9mm_P1.27mm'
        }

        return Part(
            'Driver_FET',
            driver_type,
            ref=ref,
            value=driver_type,
            footprint=footprints.get(driver_type, 'Package_SO:SOIC-16W_7.5x10.3mm_P1.27mm'),
            dest=TEMPLATE
        )

    def calculate_bootstrap_cap(self, mosfet_part: str, voltage_drop: float = 0.5) -> float:
        """
        Calculate minimum bootstrap capacitor value.

        Formula: C_boot >= (Q_gate × 10) / ΔV

        Args:
            mosfet_part: MOSFET part number
            voltage_drop: Allowable voltage drop during on-time

        Returns:
            Minimum bootstrap capacitance in µF
        """
        specs = MOSFET_LIBRARY.get(mosfet_part)
        if specs is None:
            return 1.0  # Default 1µF

        qg_nc = specs.qg_total
        c_boot_pf = (qg_nc * 10) / voltage_drop * 1000  # Convert to pF
        c_boot_uf = c_boot_pf / 1e6

        # Round up to next standard value
        standard_values = [0.1, 0.22, 0.47, 1.0, 2.2, 4.7, 10]
        for val in standard_values:
            if val >= c_boot_uf:
                return val

        return 10.0

    def calculate_gate_resistor(
        self,
        mosfet_part: str,
        drive_voltage: float = 15.0,
        peak_current: float = 4.0
    ) -> float:
        """
        Calculate recommended gate resistor value.

        Formula: R_g = (V_drv - V_th) / I_peak

        Args:
            mosfet_part: MOSFET part number
            drive_voltage: Gate drive voltage
            peak_current: Peak gate drive current

        Returns:
            Recommended gate resistance in ohms
        """
        specs = MOSFET_LIBRARY.get(mosfet_part)
        if specs is None:
            return 10.0  # Default 10Ω

        vgs_th = specs.vgs_th
        r_gate = (drive_voltage - vgs_th) / peak_current

        # Round to standard value
        standard_values = [2.2, 3.3, 4.7, 6.8, 10, 15, 22]
        for val in standard_values:
            if val >= r_gate:
                return val

        return 22.0

    def calculate_dead_time(
        self,
        mosfet_part: str,
        gate_resistor: float,
        safety_margin: float = 1.5
    ) -> float:
        """
        Calculate minimum dead-time for half-bridge.

        Dead-time must exceed MOSFET turn-off time to prevent shoot-through.

        Args:
            mosfet_part: MOSFET part number
            gate_resistor: Gate resistor value in ohms
            safety_margin: Safety factor (typically 1.5-2x)

        Returns:
            Minimum dead-time in nanoseconds
        """
        specs = MOSFET_LIBRARY.get(mosfet_part)
        if specs is None:
            return 500  # Default 500ns

        # Turn-off time approximation: t_off ≈ R_g × Q_gd / (V_drv - V_plateau)
        # V_plateau typically ~6V for SiC, ~4V for Si
        v_plateau = 6.0 if specs.technology == MOSFETType.SILICON_CARBIDE else 4.0
        v_drv = 15.0

        t_off_ns = gate_resistor * specs.qgd / (v_drv - v_plateau)
        dead_time = t_off_ns * safety_margin

        return max(dead_time, 100)  # Minimum 100ns


@SubCircuit
def half_bridge_subcircuit(
    dc_plus: Net,
    dc_minus: Net,
    phase_out: Net,
    pwm_h: Net,
    pwm_l: Net,
    vcc_drv: Net,
    gnd_drv: Net,
    phase_name: str = 'A',
    parallel_count: int = 3,
    mosfet_part: str = 'IMZA65R027M1H'
) -> Interface:
    """
    Half-bridge subcircuit with parallel MOSFETs and gate drivers.

    MIT EE Design Notes:
    - 3 parallel MOSFETs per switch for high current capacity
    - Kelvin source connection (pin 4) prevents gate bounce at high dV/dt
    - Bootstrap supply for floating high-side driver
    - Dead-time control via DT pin resistor

    Args:
        dc_plus: Positive DC bus
        dc_minus: Negative DC bus (ground)
        phase_out: Phase output
        pwm_h: High-side PWM input
        pwm_l: Low-side PWM input
        vcc_drv: Driver supply voltage
        gnd_drv: Driver ground
        phase_name: Phase identifier (A, B, C)
        parallel_count: Number of parallel MOSFETs
        mosfet_part: MOSFET part number

    Returns:
        Interface with all connections
    """
    mosfets = MOSFETs()

    # Create high-side MOSFETs (parallel)
    for i in range(parallel_count):
        mos_h = mosfets.mosfet(mosfet_part, ref=f'Q{phase_name}H{i+1}')

        # Connect drain to DC+, source to phase output
        mos_h['D'] += dc_plus
        mos_h['S'] += phase_out

        # Kelvin source for gate driver ground (if available)
        if MOSFET_LIBRARY.get(mosfet_part, {}).has_kelvin_source:
            ks_net = Net(f'KS_{phase_name}H{i+1}')
            mos_h['KS'] += ks_net

    # Create low-side MOSFETs (parallel)
    for i in range(parallel_count):
        mos_l = mosfets.mosfet(mosfet_part, ref=f'Q{phase_name}L{i+1}')

        # Connect drain to phase output, source to DC-
        mos_l['D'] += phase_out
        mos_l['S'] += dc_minus

    return Interface(
        dc_plus=dc_plus,
        dc_minus=dc_minus,
        phase_out=phase_out,
        pwm_h=pwm_h,
        pwm_l=pwm_l
    )


# Global instance
_mosfets: Optional[MOSFETs] = None


def get_mosfets() -> MOSFETs:
    """Get global MOSFETs factory instance."""
    global _mosfets
    if _mosfets is None:
        _mosfets = MOSFETs()
    return _mosfets
