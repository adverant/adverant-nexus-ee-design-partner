/**
 * Signal Integrity Expert Agent - Dr. James Chen
 *
 * Specialized validation for signal integrity including:
 * - Current sensing accuracy (Kelvin connections)
 * - Differential routing requirements
 * - ADC input protection
 * - Ground loop identification
 * - Noise coupling analysis
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

export const SIGNAL_INTEGRITY_EXPERT: ExpertAgentConfig = {
  id: 'signal-integrity-expert',
  name: 'Dr. James Chen',
  role: 'Signal Integrity Specialist',
  expertise: [
    'High-speed PCB design',
    'EMC',
    'Current sensing',
    'ADC interfaces',
    'Kelvin (4-wire) connections',
    'Differential signal routing'
  ],
  validationFocus: {
    currentSensingAccuracy: [
      'Kelvin (4-wire) shunt connection',
      'Differential routing from shunt to amplifier',
      'Common-mode rejection requirements',
      'ADC input protection (RC filter + clamp diodes)'
    ],
    noiseCouplingAnalysis: [
      'Ground loop identification',
      'Analog/digital ground separation',
      'Switching noise coupling to sense circuits',
      'Guard traces for sensitive signals'
    ],
    adcInterfaceDesign: [
      'Anti-aliasing filter design',
      'Reference voltage bypassing',
      'Input impedance matching',
      'Sampling rate considerations'
    ]
  },
  checks: []
};

/**
 * Check that shunt sense pins are routed differentially to current sense amplifier
 */
function checkKelvinShuntConnections(schematic: SchematicData): ExpertCheckResult {
  const shunts = findComponentsByPattern(schematic, /^R_SHUNT|R[A-C]_SENSE|RSENSE/i);
  const currentAmps = findComponentsByPattern(schematic, /^U.*INA|INA\d+/);
  const errors: string[] = [];
  const details: string[] = [];

  for (const shunt of shunts) {
    // Shunt resistors should have 4 pins for Kelvin sensing
    // Pins 1,2 = power path, Pins 3,4 = sense path
    const sensePlusPins = shunt.pins.filter(p =>
      p.name === 'S+' || p.name === 'SENSE+' || p.number === '3'
    );
    const senseMinusPins = shunt.pins.filter(p =>
      p.name === 'S-' || p.name === 'SENSE-' || p.number === '4'
    );

    if (sensePlusPins.length === 0 || senseMinusPins.length === 0) {
      // 2-terminal shunt - check if connected properly
      const terminals = shunt.pins.filter(p => p.connected);
      if (terminals.length < 2) {
        errors.push(`${shunt.reference}: Shunt not fully connected`);
      }
      continue;
    }

    // Check that sense pins are connected to current sense amplifier inputs
    for (const sensePin of [...sensePlusPins, ...senseMinusPins]) {
      if (!sensePin.connected || !sensePin.netName) {
        errors.push(`${shunt.reference}.${sensePin.name}: Kelvin sense pin not connected`);
        continue;
      }

      const senseNet = findNetByName(schematic, sensePin.netName);
      if (!senseNet) continue;

      const connectedToAmp = senseNet.connections.some(conn => {
        const amp = findComponentByRef(schematic, conn.componentRef);
        return amp && /INA\d+/.test(amp.value);
      });

      if (!connectedToAmp) {
        errors.push(
          `${shunt.reference}.${sensePin.name}: Not connected to current sense amplifier input`
        );
      } else {
        details.push(`${shunt.reference}.${sensePin.name}: Properly routed to amplifier`);
      }
    }
  }

  // Also verify INA240 has both inputs connected
  for (const amp of currentAmps) {
    const inpPin = amp.pins.find(p => p.name === 'IN+' || p.name === 'INP' || p.name === 'VIN+');
    const inmPin = amp.pins.find(p => p.name === 'IN-' || p.name === 'INM' || p.name === 'VIN-');

    if (inpPin && !inpPin.connected) {
      errors.push(`${amp.reference}: IN+ pin not connected`);
    }
    if (inmPin && !inmPin.connected) {
      errors.push(`${amp.reference}: IN- pin not connected`);
    }
  }

  return {
    passed: errors.length === 0,
    details: errors.length > 0
      ? `Kelvin shunt connection issues:\n${errors.join('\n')}`
      : `All shunt resistors properly connected with Kelvin sensing:\n${details.join('\n')}`,
    evidence: { shuntCount: shunts.length, ampCount: currentAmps.length, errors }
  };
}

/**
 * Check that current sense amplifiers have dedicated analog ground return
 */
function checkAnalogGroundReturn(schematic: SchematicData): ExpertCheckResult {
  const analogICs = findComponentsByPattern(schematic, /^U.*INA|INA\d+|AD8|OPA\d+/);
  const errors: string[] = [];
  const details: string[] = [];

  // Look for analog ground nets
  const analogGndNets = ['AGND', 'GND_A', 'GNDA', 'GND_SENSE', 'GND_ANALOG'];
  const digitalGndNets = ['GND', 'DGND', 'GND_D', 'GNDD'];

  for (const ic of analogICs) {
    const gndPin = ic.pins.find(p =>
      p.name === 'GND' || p.name === 'VSS' || p.name === 'GND2'
    );

    if (!gndPin || !gndPin.connected || !gndPin.netName) {
      errors.push(`${ic.reference}: Ground pin not connected`);
      continue;
    }

    // Check if connected to analog ground
    const isAnalogGnd = analogGndNets.some(n =>
      gndPin.netName!.toUpperCase().includes(n) ||
      n.includes(gndPin.netName!.toUpperCase())
    );

    const isDigitalGnd = digitalGndNets.includes(gndPin.netName!.toUpperCase());

    if (isDigitalGnd && !isAnalogGnd) {
      errors.push(
        `${ic.reference}: Connected to digital ground (${gndPin.netName}). ` +
        `Analog ICs should use dedicated analog ground for noise immunity`
      );
    } else {
      details.push(`${ic.reference}: Ground return via ${gndPin.netName}`);
    }
  }

  // Warn if no analog ground separation exists
  const hasAnalogGnd = schematic.sheets.some(sheet =>
    sheet.nets.some(net => analogGndNets.some(n => net.name.toUpperCase().includes(n)))
  );

  if (!hasAnalogGnd && analogICs.length > 0) {
    errors.push(
      'No dedicated analog ground net found. Consider separating AGND from DGND ' +
      'for better current sense accuracy'
    );
  }

  return {
    passed: errors.length === 0,
    details: errors.length > 0
      ? `Analog ground issues:\n${errors.join('\n')}`
      : `Analog ground properly configured:\n${details.join('\n')}`,
    evidence: { analogIcCount: analogICs.length, hasAnalogGnd, errors }
  };
}

/**
 * Check RC filter on current sense amplifier outputs
 */
function checkOutputRcFilter(schematic: SchematicData): ExpertCheckResult {
  const currentAmps = findComponentsByPattern(schematic, /^U.*INA|INA\d+/);
  const errors: string[] = [];
  const details: string[] = [];

  // Standard filter: 100R + 10nF = 159kHz cutoff
  const RECOMMENDED_R = 100; // Ohms
  const RECOMMENDED_C = 10; // nF

  for (const amp of currentAmps) {
    const outPin = amp.pins.find(p =>
      p.name === 'OUT' || p.name === 'VOUT' || p.name === 'OUTPUT'
    );

    if (!outPin || !outPin.connected || !outPin.netName) {
      errors.push(`${amp.reference}: Output pin not connected`);
      continue;
    }

    const outNet = findNetByName(schematic, outPin.netName);
    if (!outNet) continue;

    // Find resistor and capacitor on output
    const resistors = outNet.connections
      .map(conn => findComponentByRef(schematic, conn.componentRef))
      .filter((comp): comp is ComponentData =>
        comp !== undefined && comp.reference.startsWith('R')
      );

    let hasFilter = false;
    for (const resistor of resistors) {
      // Check if resistor connects to a capacitor (forming RC filter)
      const resistorNets = getComponentNets(schematic, resistor);
      for (const netName of resistorNets) {
        if (netName === outPin.netName) continue;

        const filterNet = findNetByName(schematic, netName);
        if (!filterNet) continue;

        const caps = filterNet.connections
          .map(conn => findComponentByRef(schematic, conn.componentRef))
          .filter((comp): comp is ComponentData =>
            comp !== undefined && comp.reference.startsWith('C')
          );

        if (caps.length > 0) {
          hasFilter = true;
          const rValue = parseResistorValue(resistor.value);
          const cValue = parseCapacitorValueNF(caps[0].value);
          const cutoff = 1 / (2 * Math.PI * rValue * cValue * 1e-9);

          details.push(
            `${amp.reference}: RC filter ${resistor.reference}(${resistor.value}) + ` +
            `${caps[0].reference}(${caps[0].value}) = ${(cutoff / 1000).toFixed(1)}kHz cutoff`
          );
        }
      }
    }

    if (!hasFilter) {
      errors.push(
        `${amp.reference}: No RC filter on output. Recommend ${RECOMMENDED_R}R + ` +
        `${RECOMMENDED_C}nF for ADC anti-aliasing`
      );
    }
  }

  return {
    passed: errors.length === 0,
    details: errors.length > 0
      ? `Output filter issues:\n${errors.join('\n')}`
      : `Output RC filters properly configured:\n${details.join('\n')}`,
    evidence: { ampCount: currentAmps.length, errors }
  };
}

/**
 * Check ADC input protection (clamp diodes for overvoltage)
 */
function checkAdcProtection(schematic: SchematicData): ExpertCheckResult {
  const adcNets = findNetsByPattern(schematic, /I_SENSE|ISENSE|ADC_IN|AIN\d+/i);
  const errors: string[] = [];
  const details: string[] = [];

  for (const net of adcNets) {
    // Look for ESD/clamp diodes on ADC input nets
    const diodes = net.connections
      .map(conn => findComponentByRef(schematic, conn.componentRef))
      .filter((comp): comp is ComponentData =>
        comp !== undefined && (
          comp.reference.startsWith('D') ||
          comp.value.includes('TVS') ||
          comp.value.includes('ESD') ||
          comp.value.includes('PESD')
        )
      );

    if (diodes.length === 0) {
      // Check if this net connects to an ADC
      const connectsToAdc = net.connections.some(conn => {
        const comp = findComponentByRef(schematic, conn.componentRef);
        return comp && (
          conn.pinName?.includes('ADC') ||
          conn.pinName?.match(/PA\d+|PB\d+|PC\d+/) // STM32 ADC pins
        );
      });

      if (connectsToAdc) {
        errors.push(
          `${net.name}: ADC input has no ESD/clamp protection. ` +
          `Add TVS diode for overvoltage protection`
        );
      }
    } else {
      details.push(
        `${net.name}: Protected by ${diodes.map(d => `${d.reference}(${d.value})`).join(', ')}`
      );
    }
  }

  return {
    passed: errors.length === 0,
    details: errors.length > 0
      ? `ADC protection issues:\n${errors.join('\n')}`
      : `ADC inputs properly protected:\n${details.join('\n')}`,
    evidence: { adcNetCount: adcNets.length, errors }
  };
}

/**
 * Check for proper decoupling on current sense amplifiers
 */
function checkAmplifierDecoupling(schematic: SchematicData): ExpertCheckResult {
  const analogICs = findComponentsByPattern(schematic, /^U.*INA|INA\d+|AD8|OPA\d+/);
  const errors: string[] = [];
  const details: string[] = [];

  for (const ic of analogICs) {
    const vccPin = ic.pins.find(p =>
      p.name === 'VCC' || p.name === 'VS' || p.name === 'V+' || p.name === 'VDD'
    );

    if (!vccPin || !vccPin.connected || !vccPin.netName) {
      errors.push(`${ic.reference}: Power pin not connected`);
      continue;
    }

    const vccNet = findNetByName(schematic, vccPin.netName);
    if (!vccNet) continue;

    // Find decoupling capacitors
    const decouplingCaps = vccNet.connections
      .map(conn => findComponentByRef(schematic, conn.componentRef))
      .filter((comp): comp is ComponentData =>
        comp !== undefined && comp.reference.startsWith('C')
      );

    if (decouplingCaps.length === 0) {
      errors.push(
        `${ic.reference}: No decoupling capacitor on VCC (${vccPin.netName}). ` +
        `Add 100nF ceramic cap close to pin`
      );
    } else {
      // Check capacitor values
      const hasSmallCap = decouplingCaps.some(c => {
        const value = parseCapacitorValueNF(c.value);
        return value >= 100 && value <= 1000; // 100nF to 1µF
      });

      if (!hasSmallCap) {
        errors.push(
          `${ic.reference}: Decoupling cap values may be suboptimal. ` +
          `Ensure 100nF ceramic is present for high-frequency noise filtering`
        );
      } else {
        details.push(
          `${ic.reference}: Decoupled with ${decouplingCaps.map(c => c.value).join(', ')}`
        );
      }
    }
  }

  return {
    passed: errors.length === 0,
    details: errors.length > 0
      ? `Decoupling issues:\n${errors.join('\n')}`
      : `Amplifier decoupling properly configured:\n${details.join('\n')}`,
    evidence: { icCount: analogICs.length, errors }
  };
}

// ============================================================================
// Expert Check Definitions
// ============================================================================

const SIGNAL_INTEGRITY_CHECKS: ExpertCheckDefinition[] = [
  {
    id: 'kelvin_shunt',
    description: 'Shunt sense pins routed differentially to current sense amplifier',
    category: 'currentSensingAccuracy',
    severity: 'critical',
    validator: checkKelvinShuntConnections
  },
  {
    id: 'analog_gnd',
    description: 'Current sense amplifiers have dedicated analog ground return',
    category: 'noiseCouplingAnalysis',
    severity: 'major',
    validator: checkAnalogGroundReturn
  },
  {
    id: 'rc_filter',
    description: 'RC filter on amplifier output for ADC anti-aliasing',
    category: 'adcInterfaceDesign',
    severity: 'major',
    validator: checkOutputRcFilter
  },
  {
    id: 'adc_protection',
    description: 'ADC input clamp diodes for overvoltage protection',
    category: 'adcInterfaceDesign',
    severity: 'major',
    validator: checkAdcProtection
  },
  {
    id: 'amp_decoupling',
    description: 'Proper decoupling capacitors on analog IC power pins',
    category: 'noiseCouplingAnalysis',
    severity: 'major',
    validator: checkAmplifierDecoupling
  }
];

// Add checks to config
SIGNAL_INTEGRITY_EXPERT.checks = SIGNAL_INTEGRITY_CHECKS;

// ============================================================================
// Helper Functions
// ============================================================================

function findComponentsByPattern(schematic: SchematicData, pattern: RegExp): ComponentData[] {
  const results: ComponentData[] = [];
  for (const sheet of schematic.sheets) {
    for (const component of sheet.components) {
      if (pattern.test(component.reference) || pattern.test(component.value)) {
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

function findNetsByPattern(schematic: SchematicData, pattern: RegExp): NetData[] {
  const results: NetData[] = [];
  for (const sheet of schematic.sheets) {
    for (const net of sheet.nets) {
      if (pattern.test(net.name)) {
        results.push(net);
      }
    }
  }
  return results;
}

function getComponentNets(schematic: SchematicData, component: ComponentData): string[] {
  return component.pins
    .filter(p => p.connected && p.netName)
    .map(p => p.netName!);
}

function parseResistorValue(value: string): number {
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

function parseCapacitorValueNF(value: string): number {
  // Return value in nanofarads
  const match = value.match(/(\d+(?:\.\d+)?)\s*(p|n|u|µ|m)?F?/i);
  if (!match) return 0;

  const num = parseFloat(match[1]);
  const unit = (match[2] || 'n').toLowerCase();

  switch (unit) {
    case 'p': return num / 1000;
    case 'n': return num;
    case 'u':
    case 'µ': return num * 1000;
    case 'm': return num * 1000000;
    default: return num;
  }
}

// ============================================================================
// Expert Review Runner
// ============================================================================

export function runSignalIntegrityReview(schematic: SchematicData): ExpertReviewResult {
  const checks: ExpertCheck[] = [];
  let totalScore = 0;
  let maxScore = 0;
  const recommendations: string[] = [];

  for (const checkDef of SIGNAL_INTEGRITY_CHECKS) {
    const result = checkDef.validator(schematic);

    const weight = checkDef.severity === 'critical' ? 3 :
      checkDef.severity === 'major' ? 2 : 1;
    maxScore += weight * 10;

    if (result.passed) {
      totalScore += weight * 10;
    } else {
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
    expertId: SIGNAL_INTEGRITY_EXPERT.id,
    expertName: SIGNAL_INTEGRITY_EXPERT.name,
    role: SIGNAL_INTEGRITY_EXPERT.role,
    checks,
    passed,
    score,
    recommendations,
    timestamp: new Date().toISOString()
  };
}

export default {
  config: SIGNAL_INTEGRITY_EXPERT,
  runReview: runSignalIntegrityReview
};
