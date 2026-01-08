"""
SKiDL Circuit Descriptions for FOC ESC Heavy-Lift

This package contains SKiDL-based circuit descriptions for generating
production-ready schematics for the triple-redundant FOC ESC.

Sheets:
- sheet1_power_supply: Power input, filtering, and regulation
- sheet2_triple_mcu: Triple-redundant STM32H755 MCUs
- sheet3_power_stage: 3-phase inverter with SiC MOSFETs
- sheet4_current_sensing: Inline shunt current measurement
- sheet5_ethernet_phy: Ethernet PHY interface
- sheet6_connectors: External connectors
- sheet7_motor_interface: Motor and encoder connections
"""

from .common.power_symbols import PowerSymbols
from .common.passives import Passives
from .common.mosfets import MOSFETs

__all__ = [
    'PowerSymbols',
    'Passives',
    'MOSFETs',
    'sheet1_power_supply',
    'sheet2_triple_mcu',
    'sheet3_power_stage',
    'sheet4_current_sensing',
    'sheet5_ethernet_phy',
    'sheet6_connectors',
    'sheet7_motor_interface'
]
