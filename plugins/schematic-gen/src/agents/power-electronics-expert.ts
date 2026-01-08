/**
 * Power Electronics Expert Agent - Dr. Elena Vasquez
 *
 * Specialized validation for power electronics circuits including:
 * - Gate driver design
 * - Bootstrap supply dimensioning
 * - Dead-time calculation
 * - SiC/GaN MOSFET best practices
 * - Current sharing in parallel configurations
 */

import {
  ExpertAgentConfig,
  ExpertCheckDefinition,
  ExpertCheckResult,
  ExpertReviewResult,
  ExpertCheck,
  SchematicData,
  ComponentData,
  NetData
} from '../types';

export const POWER_ELECTRONICS_EXPERT: ExpertAgentConfig = {
  id: 'power-electronics-expert',
  name: 'Dr. Elena Vasquez',
  role: 'Power Electronics Specialist',
  expertise: [
    'High-frequency switching converters',
    'Gate driver design',
    'SiC MOSFET applications',
    'Bootstrap supply dimensioning',
    'Dead-time calculation',
    'Parallel MOSFET current sharing'
  ],
  validationFocus: {
    gateDriveIntegrity: [
      'Bootstrap supply dimensioning (C = I_gate × t_on / ΔV)',
      'Dead-time calculation (prevent shoot-through)',
      'Miller clamp effectiveness',
      'Kelvin source connection (CRITICAL for high dV/dt)'
    ],
    powerStageDesign: [
      'Parallel MOSFET current sharing (matched Rds_on and gate timing)',
      'DC bus capacitor ESR/ESL impact on voltage ripple',
      'Thermal considerations (junction-to-case thermal resistance)'
    ],
    failureModeAnalysis: [
      'Desaturation detection requirements',
      'Fault propagation paths',
      'Safe operating area (SOA) margins'
    ]
  },
  checks: []
};

/**
 * Check that every MOSFET has a Kelvin source (pin 4) connected to driver GND2
 * This is CRITICAL for high dV/dt SiC MOSFETs to prevent gate bounce
 */
function checkKelvinSourceConnections(schematic: SchematicData): ExpertCheckResult {
  const mosfets = findComponentsByPattern(schematic, /^Q[A-Z]+[HL]\d+$/);
  const drivers = findComponentsByPattern(schematic, /^U[A-Z]+[HL]\d+$/);

  const errors: string[] = [];
  const details: string[] = [];

  for (const mosfet of mosfets) {
    // Check if MOSFET is a 4-pin device (Kelvin source)
    const kelvinPin = mosfet.pins.find(p =>
      p.name === 'KS' || p.name === 'S2' || p.number === '4'
    );

    if (!kelvinPin) {
      errors.push(`${mosfet.reference}: No Kelvin source pin found - using 3-pin package is not recommended for high dV/dt`);
      continue;
    }

    if (!kelvinPin.connected || !kelvinPin.netName) {
      errors.push(`${mosfet.reference}: Kelvin source (pin 4) is not connected`);
      continue;
    }

    // Verify Kelvin source is connected to driver GND2
    const ksNet = findNetByName(schematic, kelvinPin.netName);
    if (!ksNet) {
      errors.push(`${mosfet.reference}: Kelvin source net '${kelvinPin.netName}' not found`);
      continue;
    }

    const connectedToDriverGnd = ksNet.connections.some(conn => {
      const driver = findComponentByRef(schematic, conn.componentRef);
      return driver && (conn.pinName === 'GND2' || conn.pinName === 'VSS2');
    });

    if (!connectedToDriverGnd) {
      errors.push(`${mosfet.reference}: Kelvin source not connected to gate driver GND2`);
    } else {
      details.push(`${mosfet.reference}: Kelvin source properly connected`);
    }
  }

  return {
    passed: errors.length === 0,
    details: errors.length > 0
      ? `Issues found:\n${errors.join('\n')}`
      : `All ${mosfets.length} MOSFETs have proper Kelvin source connections:\n${details.join('\n')}`,
    evidence: { mosfetCount: mosfets.length, errors }
  };
}

/**
 * Check bootstrap capacitor sizing for high-side gate drivers
 * C_boot >= (Q_gate × 10) / ΔV where ΔV is typically 0.5V max drop
 */
function checkBootstrapCapacitors(schematic: SchematicData): ExpertCheckResult {
  const drivers = findComponentsByPattern(schematic, /^U[A-Z]+H\d+|UCC21|IR2|FAN73/);
  const errors: string[] = [];
  const details: string[] = [];

  // Typical gate charge for IMZA65R027M1H is ~150nC
  const TYPICAL_GATE_CHARGE_NC = 150;
  const MIN_MULTIPLIER = 10;
  const MAX_VOLTAGE_DROP = 0.5;
  const MIN_BOOT_CAP_UF = (TYPICAL_GATE_CHARGE_NC * MIN_MULTIPLIER) / (MAX_VOLTAGE_DROP * 1000);

  for (const driver of drivers) {
    // Find VCC2/VBOOT pin
    const vcc2Pin = driver.pins.find(p =>
      p.name === 'VCC2' || p.name === 'VB' || p.name === 'VBOOT' || p.name === 'HO_VCC'
    );

    if (!vcc2Pin || !vcc2Pin.netName) {
      errors.push(`${driver.reference}: No bootstrap supply pin found`);
      continue;
    }

    // Find bootstrap capacitor on this net
    const bootNet = findNetByName(schematic, vcc2Pin.netName);
    if (!bootNet) {
      errors.push(`${driver.reference}: Bootstrap net not found`);
      continue;
    }

    const bootCaps = bootNet.connections
      .map(conn => findComponentByRef(schematic, conn.componentRef))
      .filter((comp): comp is ComponentData =>
        comp !== undefined && comp.reference.startsWith('C')
      );

    if (bootCaps.length === 0) {
      errors.push(`${driver.reference}: No bootstrap capacitor found on ${vcc2Pin.netName}`);
      continue;
    }

    // Parse capacitor value and check sizing
    for (const cap of bootCaps) {
      const valueUf = parseCapacitorValue(cap.value);
      if (valueUf < MIN_BOOT_CAP_UF) {
        errors.push(
          `${cap.reference} on ${driver.reference}: Bootstrap cap ${cap.value} (${valueUf.toFixed(2)}µF) ` +
          `is undersized. Minimum: ${MIN_BOOT_CAP_UF.toFixed(2)}µF for 150nC gate charge`
        );
      } else {
        details.push(`${cap.reference}: ${cap.value} properly sized for ${driver.reference}`);
      }
    }
  }

  return {
    passed: errors.length === 0,
    details: errors.length > 0
      ? `Bootstrap capacitor issues:\n${errors.join('\n')}`
      : `All bootstrap capacitors properly sized:\n${details.join('\n')}`,
    evidence: { driverCount: drivers.length, minBootCapUf: MIN_BOOT_CAP_UF, errors }
  };
}

/**
 * Check that gate resistors are properly matched for parallel MOSFETs
 * Mismatch should be ±1% to ensure equal current sharing during switching
 */
function checkGateResistorMatching(schematic: SchematicData): ExpertCheckResult {
  // Group MOSFETs by phase (A, B, C) and position (H, L)
  const mosfetGroups = new Map<string, ComponentData[]>();

  const mosfets = findComponentsByPattern(schematic, /^Q([ABC])([HL])(\d+)$/);
  for (const mosfet of mosfets) {
    const match = mosfet.reference.match(/^Q([ABC])([HL])/);
    if (match) {
      const key = `${match[1]}${match[2]}`;
      if (!mosfetGroups.has(key)) {
        mosfetGroups.set(key, []);
      }
      mosfetGroups.get(key)!.push(mosfet);
    }
  }

  const errors: string[] = [];
  const details: string[] = [];

  for (const [group, fets] of mosfetGroups) {
    if (fets.length < 2) continue; // Not parallel

    // Find gate resistors for each MOSFET
    const gateResistors: Array<{ mosfet: string; resistor: string; value: number }> = [];

    for (const mosfet of fets) {
      const gatePin = mosfet.pins.find(p => p.name === 'G' || p.number === '1');
      if (!gatePin || !gatePin.netName) continue;

      const gateNet = findNetByName(schematic, gatePin.netName);
      if (!gateNet) continue;

      const resistors = gateNet.connections
        .map(conn => findComponentByRef(schematic, conn.componentRef))
        .filter((comp): comp is ComponentData =>
          comp !== undefined && comp.reference.startsWith('R')
        );

      for (const resistor of resistors) {
        const value = parseResistorValue(resistor.value);
        gateResistors.push({
          mosfet: mosfet.reference,
          resistor: resistor.reference,
          value
        });
      }
    }

    // Check for matching values
    if (gateResistors.length >= 2) {
      const values = gateResistors.map(r => r.value);
      const avgValue = values.reduce((a, b) => a + b, 0) / values.length;
      const maxDeviation = Math.max(...values.map(v => Math.abs(v - avgValue) / avgValue * 100));

      if (maxDeviation > 1) {
        errors.push(
          `Phase ${group}: Gate resistors have ${maxDeviation.toFixed(1)}% mismatch. ` +
          `Values: ${gateResistors.map(r => `${r.resistor}=${r.value}Ω`).join(', ')}. ` +
          `Max allowed: ±1%`
        );
      } else {
        details.push(`Phase ${group}: Gate resistors matched (${avgValue}Ω ±${maxDeviation.toFixed(1)}%)`);
      }
    }
  }

  return {
    passed: errors.length === 0,
    details: errors.length > 0
      ? `Gate resistor matching issues:\n${errors.join('\n')}`
      : `Gate resistors properly matched for parallel MOSFETs:\n${details.join('\n')}`,
    evidence: { groupCount: mosfetGroups.size, errors }
  };
}

/**
 * Check dead-time resistor presence on gate drivers
 */
function checkDeadTimeResistor(schematic: SchematicData): ExpertCheckResult {
  const drivers = findComponentsByPattern(schematic, /^U.*DRV|UCC21|IR2|FAN73/);
  const errors: string[] = [];
  const details: string[] = [];

  for (const driver of drivers) {
    const dtPin = driver.pins.find(p =>
      p.name === 'DT' || p.name === 'DEAD' || p.name === 'DTC'
    );

    if (!dtPin) {
      // Driver may not have programmable dead-time
      continue;
    }

    if (!dtPin.connected || !dtPin.netName) {
      errors.push(`${driver.reference}: Dead-time pin (DT) is not connected`);
      continue;
    }

    const dtNet = findNetByName(schematic, dtPin.netName);
    if (!dtNet) {
      errors.push(`${driver.reference}: Dead-time net not found`);
      continue;
    }

    const dtResistors = dtNet.connections
      .map(conn => findComponentByRef(schematic, conn.componentRef))
      .filter((comp): comp is ComponentData =>
        comp !== undefined && comp.reference.startsWith('R')
      );

    if (dtResistors.length === 0) {
      errors.push(`${driver.reference}: No dead-time resistor found on DT pin`);
    } else {
      details.push(`${driver.reference}: Dead-time set by ${dtResistors[0].reference} (${dtResistors[0].value})`);
    }
  }

  return {
    passed: errors.length === 0,
    details: errors.length > 0
      ? `Dead-time configuration issues:\n${errors.join('\n')}`
      : `Dead-time properly configured:\n${details.join('\n')}`,
    evidence: { driverCount: drivers.length, errors }
  };
}

/**
 * Check DC bus capacitor ESR/quantity for voltage ripple
 */
function checkDcBusCapacitors(schematic: SchematicData): ExpertCheckResult {
  const dcBusNets = ['DC_BUS_P', 'DC_BUS+', 'VBUS', 'V_DC'];
  const errors: string[] = [];
  const details: string[] = [];

  let totalCapacitance = 0;
  let capCount = 0;

  for (const netName of dcBusNets) {
    const net = findNetByName(schematic, netName);
    if (!net) continue;

    const caps = net.connections
      .map(conn => findComponentByRef(schematic, conn.componentRef))
      .filter((comp): comp is ComponentData =>
        comp !== undefined && comp.reference.startsWith('C')
      );

    for (const cap of caps) {
      const valueUf = parseCapacitorValue(cap.value);
      totalCapacitance += valueUf;
      capCount++;
    }
  }

  // For 300A motor controller, minimum DC bus capacitance is ~3500µF
  const MIN_DC_BUS_CAP_UF = 3500;

  if (totalCapacitance < MIN_DC_BUS_CAP_UF) {
    errors.push(
      `DC bus capacitance ${totalCapacitance.toFixed(0)}µF is insufficient. ` +
      `Minimum ${MIN_DC_BUS_CAP_UF}µF required for 300A switching`
    );
  } else {
    details.push(
      `DC bus capacitance: ${totalCapacitance.toFixed(0)}µF across ${capCount} capacitors - ADEQUATE`
    );
  }

  return {
    passed: errors.length === 0,
    details: errors.length > 0
      ? `DC bus issues:\n${errors.join('\n')}`
      : details.join('\n'),
    evidence: { totalCapacitance, capCount, minRequired: MIN_DC_BUS_CAP_UF }
  };
}

// ============================================================================
// Expert Check Definitions
// ============================================================================

const POWER_ELECTRONICS_CHECKS: ExpertCheckDefinition[] = [
  {
    id: 'kelvin_source',
    description: 'Every MOSFET has Kelvin source (pin 4) connected to driver GND2',
    category: 'gateDriveIntegrity',
    severity: 'critical',
    validator: checkKelvinSourceConnections
  },
  {
    id: 'bootstrap_cap',
    description: 'Bootstrap capacitor sized for 10x gate charge minimum',
    category: 'gateDriveIntegrity',
    severity: 'critical',
    validator: checkBootstrapCapacitors
  },
  {
    id: 'matched_rg',
    description: 'Parallel MOSFETs have matched gate resistors (±1%)',
    category: 'powerStageDesign',
    severity: 'major',
    validator: checkGateResistorMatching
  },
  {
    id: 'dead_time',
    description: 'Dead-time resistor present on each driver DT pin',
    category: 'gateDriveIntegrity',
    severity: 'major',
    validator: checkDeadTimeResistor
  },
  {
    id: 'dc_bus_cap',
    description: 'DC bus capacitance adequate for switching current',
    category: 'powerStageDesign',
    severity: 'major',
    validator: checkDcBusCapacitors
  }
];

// Add checks to config
POWER_ELECTRONICS_EXPERT.checks = POWER_ELECTRONICS_CHECKS;

// ============================================================================
// Helper Functions
// ============================================================================

function findComponentsByPattern(schematic: SchematicData, pattern: RegExp): ComponentData[] {
  const results: ComponentData[] = [];
  for (const sheet of schematic.sheets) {
    for (const component of sheet.components) {
      if (pattern.test(component.reference)) {
        results.push(component);
      }
    }
  }
  return results;
}

function findComponentByRef(schematic: SchematicData, ref: string): ComponentData | undefined {
  for (const sheet of schematic.sheets) {
    const component = sheet.components.find(c => c.reference === ref);
    if (component) return component;
  }
  return undefined;
}

function findNetByName(schematic: SchematicData, netName: string): NetData | undefined {
  for (const sheet of schematic.sheets) {
    const net = sheet.nets.find(n => n.name === netName);
    if (net) return net;
  }
  return undefined;
}

function parseCapacitorValue(value: string): number {
  // Parse values like "1uF", "100nF", "10pF", "350uF/100V"
  const match = value.match(/(\d+(?:\.\d+)?)\s*(p|n|u|µ|m)?F?/i);
  if (!match) return 0;

  const num = parseFloat(match[1]);
  const unit = (match[2] || 'u').toLowerCase();

  switch (unit) {
    case 'p': return num / 1000000;
    case 'n': return num / 1000;
    case 'u':
    case 'µ': return num;
    case 'm': return num * 1000;
    default: return num;
  }
}

function parseResistorValue(value: string): number {
  // Parse values like "4.7", "10k", "1M", "100R"
  const match = value.match(/(\d+(?:\.\d+)?)\s*(k|M|R|Ω)?/i);
  if (!match) return 0;

  const num = parseFloat(match[1]);
  const unit = (match[2] || '').toUpperCase();

  switch (unit) {
    case 'K': return num * 1000;
    case 'M': return num * 1000000;
    default: return num;
  }
}

// ============================================================================
// Expert Review Runner
// ============================================================================

export function runPowerElectronicsReview(schematic: SchematicData): ExpertReviewResult {
  const checks: ExpertCheck[] = [];
  let totalScore = 0;
  let maxScore = 0;
  const recommendations: string[] = [];

  for (const checkDef of POWER_ELECTRONICS_CHECKS) {
    const result = checkDef.validator(schematic);

    const weight = checkDef.severity === 'critical' ? 3 :
      checkDef.severity === 'major' ? 2 : 1;
    maxScore += weight * 10;

    if (result.passed) {
      totalScore += weight * 10;
    } else {
      // Add recommendation based on failure
      recommendations.push(
        `[${checkDef.severity.toUpperCase()}] ${checkDef.description}: ${result.details}`
      );
    }

    checks.push({
      id: checkDef.id,
      description: checkDef.description,
      category: checkDef.category,
      passed: result.passed,
      details: result.details,
      severity: checkDef.severity
    });
  }

  const passed = checks.every(c => c.severity !== 'critical' || c.passed);
  const score = maxScore > 0 ? (totalScore / maxScore) * 100 : 0;

  return {
    expertId: POWER_ELECTRONICS_EXPERT.id,
    expertName: POWER_ELECTRONICS_EXPERT.name,
    role: POWER_ELECTRONICS_EXPERT.role,
    checks,
    passed,
    score,
    recommendations,
    timestamp: new Date().toISOString()
  };
}

export default {
  config: POWER_ELECTRONICS_EXPERT,
  runReview: runPowerElectronicsReview
};
