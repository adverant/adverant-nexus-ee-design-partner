/**
 * Validation & Verification Expert Agent - Dr. Sarah Kim
 *
 * Specialized validation for schematic quality including:
 * - Wire/component ratio analysis
 * - Pin conflict detection
 * - Unconnected pin detection
 * - Net driver analysis
 * - Reference uniqueness
 * - Specification coverage
 */

import {
  ExpertAgentConfig,
  ExpertCheckDefinition,
  ExpertCheckResult,
  ExpertReviewResult,
  ExpertCheck,
  SchematicData,
  ComponentData,
  NetData,
  VALIDATION_THRESHOLDS
} from '../types';

export const VALIDATION_EXPERT: ExpertAgentConfig = {
  id: 'validation-expert',
  name: 'Dr. Sarah Kim',
  role: 'Validation & Verification Specialist',
  expertise: [
    'ERC automation',
    'Design rule checking',
    'Formal circuit verification',
    'Wire/component ratio analysis',
    'Connectivity validation'
  ],
  validationFocus: {
    structuralIntegrity: [
      'Wire/component ratio (minimum 1.2, target 1.5)',
      'No orphaned components',
      'Complete net connectivity'
    ],
    electricalRuleCheck: [
      'No output-output conflicts',
      'No power net conflicts',
      'Every net has exactly one driver'
    ],
    designConsistency: [
      'Unique component references',
      'Consistent naming conventions',
      'Proper hierarchical structure'
    ]
  },
  checks: []
};

/**
 * Check wire/component ratio meets minimum threshold
 */
function checkWireComponentRatio(schematic: SchematicData): ExpertCheckResult {
  let totalComponents = 0;
  let totalWires = 0;
  const sheetRatios: Array<{ sheet: string; ratio: number; components: number; wires: number }> = [];

  for (const sheet of schematic.sheets) {
    const componentCount = sheet.components.length;
    const wireCount = sheet.wires.length;
    totalComponents += componentCount;
    totalWires += wireCount;

    if (componentCount > 0) {
      const ratio = wireCount / componentCount;
      sheetRatios.push({
        sheet: sheet.name,
        ratio,
        components: componentCount,
        wires: wireCount
      });
    }
  }

  const overallRatio = totalComponents > 0 ? totalWires / totalComponents : 0;
  const errors: string[] = [];
  const details: string[] = [];

  // Check overall ratio
  if (overallRatio < VALIDATION_THRESHOLDS.WIRE_COMPONENT_RATIO_CRITICAL) {
    errors.push(
      `CRITICAL: Overall wire/component ratio ${overallRatio.toFixed(2)} is below ` +
      `critical threshold (${VALIDATION_THRESHOLDS.WIRE_COMPONENT_RATIO_CRITICAL}). ` +
      `Schematic may have significant connectivity issues`
    );
  } else if (overallRatio < VALIDATION_THRESHOLDS.WIRE_COMPONENT_RATIO_MIN) {
    errors.push(
      `Overall wire/component ratio ${overallRatio.toFixed(2)} is below minimum ` +
      `threshold (${VALIDATION_THRESHOLDS.WIRE_COMPONENT_RATIO_MIN}). ` +
      `Target: ${VALIDATION_THRESHOLDS.WIRE_COMPONENT_RATIO_TARGET}`
    );
  } else {
    details.push(
      `Overall wire/component ratio: ${overallRatio.toFixed(2)} ` +
      `(${totalWires} wires / ${totalComponents} components) - PASS`
    );
  }

  // Check per-sheet ratios
  for (const { sheet, ratio, components, wires } of sheetRatios) {
    if (ratio < VALIDATION_THRESHOLDS.WIRE_COMPONENT_RATIO_MIN) {
      errors.push(
        `${sheet}: Ratio ${ratio.toFixed(2)} (${wires}/${components}) below minimum`
      );
    } else {
      details.push(`${sheet}: Ratio ${ratio.toFixed(2)} (${wires}/${components})`);
    }
  }

  return {
    passed: errors.length === 0,
    details: errors.length > 0
      ? `Wire/component ratio issues:\n${errors.join('\n')}`
      : `Wire/component ratios:\n${details.join('\n')}`,
    evidence: {
      overallRatio,
      totalComponents,
      totalWires,
      sheetRatios,
      threshold: VALIDATION_THRESHOLDS.WIRE_COMPONENT_RATIO_MIN
    }
  };
}

/**
 * Check for pin conflicts (output-output, power conflicts)
 */
function checkPinConflicts(schematic: SchematicData): ExpertCheckResult {
  const errors: string[] = [];
  const details: string[] = [];
  let netsChecked = 0;

  for (const sheet of schematic.sheets) {
    for (const net of sheet.nets) {
      netsChecked++;
      const connections = net.connections;
      if (connections.length < 2) continue;

      // Collect pin types for this net
      const pinTypes: Array<{ comp: string; pin: string; type: string }> = [];

      for (const conn of connections) {
        const component = sheet.components.find(c => c.reference === conn.componentRef);
        if (!component) continue;

        const pin = component.pins.find(p =>
          p.number === conn.pinNumber || p.name === conn.pinName
        );
        if (!pin) continue;

        pinTypes.push({
          comp: conn.componentRef,
          pin: conn.pinName || conn.pinNumber,
          type: pin.type
        });
      }

      // Check for output-output conflicts
      const outputs = pinTypes.filter(p =>
        p.type === 'output' || p.type === 'power_output'
      );
      if (outputs.length > 1) {
        // Allow power outputs on power nets
        const isPowerNet = net.name.includes('VCC') || net.name.includes('VDD') ||
          net.name.includes('GND') || net.name.includes('V3V3') ||
          net.properties.netType === 'power';

        if (!isPowerNet) {
          errors.push(
            `${net.name}: Multiple outputs connected - ` +
            outputs.map(o => `${o.comp}.${o.pin}`).join(', ')
          );
        }
      }

      // Check for driver on signal nets
      if (!net.name.includes('GND') && !net.name.includes('VCC') && !net.name.includes('VDD')) {
        const drivers = pinTypes.filter(p =>
          p.type === 'output' || p.type === 'power_output' ||
          p.type === 'bidirectional' || p.type === 'tri_state'
        );

        if (drivers.length === 0) {
          const inputs = pinTypes.filter(p => p.type === 'input' || p.type === 'power_input');
          if (inputs.length > 0 && pinTypes.filter(p => p.type === 'passive').length === 0) {
            // Net has inputs but no drivers and no passive components
            errors.push(
              `${net.name}: No driver found for net with ${inputs.length} inputs`
            );
          }
        }
      }
    }
  }

  details.push(`Checked ${netsChecked} nets for pin conflicts`);

  return {
    passed: errors.length === 0,
    details: errors.length > 0
      ? `Pin conflict issues:\n${errors.join('\n')}`
      : `No pin conflicts found. ${details.join('\n')}`,
    evidence: { netsChecked, conflicts: errors.length }
  };
}

/**
 * Check for unconnected pins
 */
function checkUnconnectedPins(schematic: SchematicData): ExpertCheckResult {
  const errors: string[] = [];
  const details: string[] = [];
  let totalPins = 0;
  let connectedPins = 0;
  let ncPins = 0;

  for (const sheet of schematic.sheets) {
    for (const component of sheet.components) {
      for (const pin of component.pins) {
        totalPins++;

        // Skip no-connect pins
        if (pin.type === 'no_connect' || pin.name === 'NC' || pin.name.startsWith('NC')) {
          ncPins++;
          continue;
        }

        if (pin.connected) {
          connectedPins++;
        } else {
          // Check if this is a critical pin
          const isCritical = pin.type === 'power_input' || pin.type === 'power_output' ||
            pin.name === 'VCC' || pin.name === 'VDD' || pin.name === 'GND' ||
            pin.name === 'VSS';

          if (isCritical) {
            errors.push(
              `${component.reference}.${pin.name}: Critical ${pin.type} pin not connected`
            );
          } else if (pin.type !== 'passive') {
            // Warning for non-passive unconnected pins
            errors.push(
              `${component.reference}.${pin.name}: ${pin.type} pin not connected`
            );
          }
        }
      }
    }
  }

  const unconnectedCount = totalPins - connectedPins - ncPins;
  details.push(
    `Total pins: ${totalPins}, Connected: ${connectedPins}, NC: ${ncPins}, ` +
    `Unconnected: ${unconnectedCount}`
  );

  if (unconnectedCount > VALIDATION_THRESHOLDS.MAX_UNCONNECTED_PINS) {
    details.push(`Unconnected pins exceed threshold of ${VALIDATION_THRESHOLDS.MAX_UNCONNECTED_PINS}`);
  }

  return {
    passed: errors.length === 0,
    details: errors.length > 0
      ? `Unconnected pin issues:\n${errors.join('\n')}\n${details.join('\n')}`
      : `All critical pins connected. ${details.join('\n')}`,
    evidence: { totalPins, connectedPins, ncPins, unconnectedCount }
  };
}

/**
 * Check that component references are unique
 */
function checkReferenceUniqueness(schematic: SchematicData): ExpertCheckResult {
  const references = new Map<string, string[]>();
  const errors: string[] = [];

  for (const sheet of schematic.sheets) {
    for (const component of sheet.components) {
      if (!references.has(component.reference)) {
        references.set(component.reference, []);
      }
      references.get(component.reference)!.push(sheet.name);
    }
  }

  let duplicates = 0;
  for (const [ref, sheets] of references) {
    if (sheets.length > 1) {
      duplicates++;
      errors.push(
        `${ref}: Duplicate reference found in sheets: ${sheets.join(', ')}`
      );
    }
  }

  return {
    passed: errors.length === 0,
    details: errors.length > 0
      ? `Reference uniqueness issues:\n${errors.join('\n')}`
      : `All ${references.size} component references are unique`,
    evidence: { totalReferences: references.size, duplicates }
  };
}

/**
 * Check that every net has at least one connection (no floating nets)
 */
function checkNetDrivers(schematic: SchematicData): ExpertCheckResult {
  const errors: string[] = [];
  const details: string[] = [];
  let floatingNets = 0;
  let singleConnectionNets = 0;

  for (const sheet of schematic.sheets) {
    for (const net of sheet.nets) {
      if (net.connections.length === 0) {
        floatingNets++;
        errors.push(`${net.name}: Net has no connections (floating)`);
      } else if (net.connections.length === 1) {
        singleConnectionNets++;
        // Single connection is usually a problem unless it's a test point
        const conn = net.connections[0];
        if (!conn.pinName?.includes('TP') && !conn.componentRef?.includes('TP')) {
          errors.push(
            `${net.name}: Net has only one connection (${conn.componentRef}.${conn.pinName})`
          );
        }
      }
    }
  }

  const totalNets = schematic.sheets.reduce((sum, s) => sum + s.nets.length, 0);
  details.push(
    `Total nets: ${totalNets}, Floating: ${floatingNets}, Single connection: ${singleConnectionNets}`
  );

  return {
    passed: errors.length === 0,
    details: errors.length > 0
      ? `Net connectivity issues:\n${errors.join('\n')}`
      : `All nets properly connected. ${details.join('\n')}`,
    evidence: { totalNets, floatingNets, singleConnectionNets }
  };
}

/**
 * Check power net consistency (VCC, VDD, GND naming)
 */
function checkPowerNetConsistency(schematic: SchematicData): ExpertCheckResult {
  const powerNets = new Map<string, { voltage?: string; connections: number }>();
  const errors: string[] = [];
  const details: string[] = [];

  for (const sheet of schematic.sheets) {
    for (const net of sheet.nets) {
      const isPower = net.name.includes('VCC') || net.name.includes('VDD') ||
        net.name.includes('GND') || net.name.includes('VSS') ||
        net.name.includes('V3V3') || net.name.includes('V5V') ||
        net.name.includes('VIN') || net.name.includes('VBUS') ||
        net.properties.netType === 'power' || net.properties.netType === 'ground';

      if (isPower) {
        if (!powerNets.has(net.name)) {
          powerNets.set(net.name, {
            voltage: net.properties.maxVoltage?.toString(),
            connections: 0
          });
        }
        powerNets.get(net.name)!.connections += net.connections.length;
      }
    }
  }

  // Check for power net naming consistency
  const vccNets = Array.from(powerNets.keys()).filter(n =>
    n.includes('VCC') || n.includes('VDD')
  );
  const gndNets = Array.from(powerNets.keys()).filter(n =>
    n.includes('GND') || n.includes('VSS')
  );

  // Verify GND exists
  if (gndNets.length === 0) {
    errors.push('No ground net (GND/VSS) found in schematic');
  }

  // Verify at least one power rail
  if (vccNets.length === 0) {
    errors.push('No power rail (VCC/VDD) found in schematic');
  }

  // Report power nets
  for (const [name, info] of powerNets) {
    details.push(`${name}: ${info.connections} connections`);
  }

  return {
    passed: errors.length === 0,
    details: errors.length > 0
      ? `Power net issues:\n${errors.join('\n')}`
      : `Power nets configured:\n${details.join('\n')}`,
    evidence: {
      powerNetCount: powerNets.size,
      vccNets: vccNets.length,
      gndNets: gndNets.length
    }
  };
}

// ============================================================================
// Expert Check Definitions
// ============================================================================

const VALIDATION_CHECKS: ExpertCheckDefinition[] = [
  {
    id: 'wire_ratio',
    description: `Wire/component ratio >= ${VALIDATION_THRESHOLDS.WIRE_COMPONENT_RATIO_MIN}`,
    category: 'structuralIntegrity',
    severity: 'critical',
    validator: checkWireComponentRatio
  },
  {
    id: 'pin_conflicts',
    description: 'No output-output or power conflicts on nets',
    category: 'electricalRuleCheck',
    severity: 'critical',
    validator: checkPinConflicts
  },
  {
    id: 'unconnected',
    description: 'No unconnected critical pins',
    category: 'electricalRuleCheck',
    severity: 'critical',
    validator: checkUnconnectedPins
  },
  {
    id: 'ref_unique',
    description: 'Component references are unique across all sheets',
    category: 'designConsistency',
    severity: 'critical',
    validator: checkReferenceUniqueness
  },
  {
    id: 'net_drivers',
    description: 'Every net has at least one driver',
    category: 'electricalRuleCheck',
    severity: 'major',
    validator: checkNetDrivers
  },
  {
    id: 'power_nets',
    description: 'Power net consistency and presence',
    category: 'designConsistency',
    severity: 'major',
    validator: checkPowerNetConsistency
  }
];

// Add checks to config
VALIDATION_EXPERT.checks = VALIDATION_CHECKS;

// ============================================================================
// Expert Review Runner
// ============================================================================

export function runValidationReview(schematic: SchematicData): ExpertReviewResult {
  const checks: ExpertCheck[] = [];
  let totalScore = 0;
  let maxScore = 0;
  const recommendations: string[] = [];

  for (const checkDef of VALIDATION_CHECKS) {
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
    expertId: VALIDATION_EXPERT.id,
    expertName: VALIDATION_EXPERT.name,
    role: VALIDATION_EXPERT.role,
    checks,
    passed,
    score,
    recommendations,
    timestamp: new Date().toISOString()
  };
}

export default {
  config: VALIDATION_EXPERT,
  runReview: runValidationReview
};
