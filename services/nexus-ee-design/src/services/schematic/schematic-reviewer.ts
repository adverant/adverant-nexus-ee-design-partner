/**
 * Schematic Reviewer Service
 *
 * Multi-LLM validation for schematic designs.
 * Performs ERC, best practices checks, and component validation.
 */

import { EventEmitter } from 'events';
import { v4 as uuidv4 } from 'uuid';
import { logger } from '../../utils/logger';
import {
  Schematic,
  Component,
  Net,
  ValidationResults,
  ValidationDomain,
  Violation,
  Warning
} from '../../types';

// ============================================================================
// Types
// ============================================================================

export interface ReviewConfig {
  enableERC: boolean;
  enableBestPractices: boolean;
  enableComponentValidation: boolean;
  enableNetClassValidation: boolean;
  enablePowerValidation: boolean;
  enableSignalValidation: boolean;
  strictMode: boolean;
  customRules?: CustomRule[];
}

export interface CustomRule {
  id: string;
  name: string;
  description: string;
  severity: 'critical' | 'error' | 'warning';
  check: (schematic: Schematic) => RuleViolation[];
}

export interface RuleViolation {
  ruleId: string;
  componentId?: string;
  netId?: string;
  message: string;
  suggestion?: string;
}

export interface ReviewResult {
  schematicId: string;
  passed: boolean;
  score: number;
  validationResults: ValidationResults;
  recommendations: string[];
  estimatedIssueCount: {
    critical: number;
    error: number;
    warning: number;
  };
  reviewedAt: string;
  reviewDuration: number;
}

export interface ERCRule {
  id: string;
  name: string;
  description: string;
  check: (schematic: Schematic) => Violation[];
}

// ============================================================================
// ERC Rules
// ============================================================================

const ERC_RULES: ERCRule[] = [
  {
    id: 'ERC001',
    name: 'Floating Input',
    description: 'Input pins must be connected to a net',
    check: (schematic: Schematic): Violation[] => {
      const violations: Violation[] = [];

      for (const component of schematic.components) {
        for (const pin of component.pins) {
          if (pin.type === 'input' && !pin.connectedNet) {
            violations.push({
              id: uuidv4(),
              severity: 'error',
              code: 'ERC001',
              message: `Floating input pin: ${component.reference}.${pin.name}`,
              location: {
                componentId: component.id,
                position: component.position
              },
              suggestion: `Connect ${component.reference}.${pin.name} to a valid net or tie to ground/VDD`
            });
          }
        }
      }

      return violations;
    }
  },
  {
    id: 'ERC002',
    name: 'Output Conflict',
    description: 'Multiple outputs should not be connected to the same net',
    check: (schematic: Schematic): Violation[] => {
      const violations: Violation[] = [];
      const netOutputs = new Map<string, Array<{ component: Component; pin: string }>>();

      for (const component of schematic.components) {
        for (const pin of component.pins) {
          if ((pin.type === 'output' || pin.type === 'power_output') && pin.connectedNet) {
            if (!netOutputs.has(pin.connectedNet)) {
              netOutputs.set(pin.connectedNet, []);
            }
            netOutputs.get(pin.connectedNet)!.push({
              component,
              pin: pin.name
            });
          }
        }
      }

      for (const [netName, outputs] of netOutputs) {
        if (outputs.length > 1) {
          // Allow multiple power outputs on power nets
          const isPowerNet = netName.includes('VDD') || netName.includes('VCC') ||
            netName.includes('GND') || netName.includes('VSS');
          const allPowerOutputs = outputs.every(o =>
            schematic.components.find(c => c.id === o.component.id)?.pins
              .find(p => p.name === o.pin)?.type === 'power_output'
          );

          if (!isPowerNet || !allPowerOutputs) {
            violations.push({
              id: uuidv4(),
              severity: 'critical',
              code: 'ERC002',
              message: `Output conflict on net ${netName}: ${outputs.map(o => `${o.component.reference}.${o.pin}`).join(', ')}`,
              location: { netId: netName },
              suggestion: 'Use tri-state buffers or mux to resolve output conflicts'
            });
          }
        }
      }

      return violations;
    }
  },
  {
    id: 'ERC003',
    name: 'Power Pin Connection',
    description: 'Power pins must be connected to power nets',
    check: (schematic: Schematic): Violation[] => {
      const violations: Violation[] = [];

      for (const component of schematic.components) {
        for (const pin of component.pins) {
          if (pin.type === 'power_input') {
            if (!pin.connectedNet) {
              violations.push({
                id: uuidv4(),
                severity: 'critical',
                code: 'ERC003',
                message: `Unconnected power pin: ${component.reference}.${pin.name}`,
                location: {
                  componentId: component.id,
                  position: component.position
                },
                suggestion: `Connect ${component.reference}.${pin.name} to appropriate power rail`
              });
            } else {
              // Verify connection to appropriate power net
              const netName = pin.connectedNet.toLowerCase();
              const pinName = pin.name.toLowerCase();

              if (pinName.includes('vdd') || pinName.includes('vcc') || pinName.includes('3v3') || pinName.includes('5v')) {
                if (netName.includes('gnd') || netName.includes('vss')) {
                  violations.push({
                    id: uuidv4(),
                    severity: 'critical',
                    code: 'ERC003',
                    message: `Power pin ${component.reference}.${pin.name} connected to wrong polarity net ${pin.connectedNet}`,
                    location: {
                      componentId: component.id,
                      netId: pin.connectedNet
                    },
                    suggestion: `Connect ${component.reference}.${pin.name} to a positive supply rail`
                  });
                }
              }
            }
          }
        }
      }

      return violations;
    }
  },
  {
    id: 'ERC004',
    name: 'No-Connect Pin',
    description: 'Unconnected pins should be marked as no-connect',
    check: (schematic: Schematic): Violation[] => {
      const violations: Violation[] = [];

      for (const component of schematic.components) {
        for (const pin of component.pins) {
          if (pin.type !== 'unconnected' && !pin.connectedNet) {
            // Allow unconnected passive pins on some components
            const isDecouplingCap = component.reference.startsWith('C') &&
              component.description?.toLowerCase().includes('decoupling');

            if (!isDecouplingCap && pin.type !== 'passive') {
              violations.push({
                id: uuidv4(),
                severity: 'warning',
                code: 'ERC004',
                message: `Unconnected pin: ${component.reference}.${pin.name}`,
                location: {
                  componentId: component.id,
                  position: component.position
                },
                suggestion: `Add no-connect flag to ${component.reference}.${pin.name} if intentionally unconnected`
              });
            }
          }
        }
      }

      return violations;
    }
  },
  {
    id: 'ERC005',
    name: 'Single Pin Net',
    description: 'Nets should have at least two connections',
    check: (schematic: Schematic): Violation[] => {
      const violations: Violation[] = [];

      for (const net of schematic.nets) {
        if (net.connections.length === 1) {
          violations.push({
            id: uuidv4(),
            severity: 'warning',
            code: 'ERC005',
            message: `Net ${net.name} has only one connection`,
            location: { netId: net.id },
            suggestion: `Connect net ${net.name} to another component or remove if unused`
          });
        }
      }

      return violations;
    }
  }
];

// ============================================================================
// Best Practices Rules
// ============================================================================

const BEST_PRACTICES_RULES: ERCRule[] = [
  {
    id: 'BP001',
    name: 'Decoupling Capacitors',
    description: 'ICs should have proper decoupling capacitors',
    check: (schematic: Schematic): Violation[] => {
      const violations: Violation[] = [];
      const ics = schematic.components.filter(c =>
        c.reference.startsWith('U') && !c.reference.includes('_REG')
      );

      for (const ic of ics) {
        // Check for nearby decoupling capacitors
        const powerPins = ic.pins.filter(p => p.type === 'power_input');
        const nearbyDecoupling = schematic.components.filter(c => {
          if (!c.reference.startsWith('C')) return false;
          const distance = Math.sqrt(
            Math.pow(c.position.x - ic.position.x, 2) +
            Math.pow(c.position.y - ic.position.y, 2)
          );
          return distance < 500; // Within 500 mils
        });

        if (nearbyDecoupling.length < powerPins.length) {
          violations.push({
            id: uuidv4(),
            severity: 'warning',
            code: 'BP001',
            message: `IC ${ic.reference} may need more decoupling capacitors (has ${nearbyDecoupling.length}, power pins: ${powerPins.length})`,
            location: {
              componentId: ic.id,
              position: ic.position
            },
            suggestion: `Add 100nF decoupling capacitors close to each power pin of ${ic.reference}`
          });
        }
      }

      return violations;
    }
  },
  {
    id: 'BP002',
    name: 'Pull-up/Pull-down Resistors',
    description: 'Open-drain outputs should have pull-up resistors',
    check: (schematic: Schematic): Violation[] => {
      const violations: Violation[] = [];

      for (const component of schematic.components) {
        for (const pin of component.pins) {
          if (pin.type === 'open_collector' || pin.type === 'open_emitter') {
            if (pin.connectedNet) {
              // Check if net has a resistor connected to power
              const net = schematic.nets.find(n => n.id === pin.connectedNet);
              if (net) {
                const hasResistor = net.connections.some(conn => {
                  const connComp = schematic.components.find(c => c.id === conn.componentId);
                  return connComp?.reference.startsWith('R');
                });

                if (!hasResistor) {
                  violations.push({
                    id: uuidv4(),
                    severity: 'warning',
                    code: 'BP002',
                    message: `Open-drain pin ${component.reference}.${pin.name} may need pull-up resistor`,
                    location: {
                      componentId: component.id,
                      netId: pin.connectedNet
                    },
                    suggestion: `Add a pull-up resistor (typically 4.7k-10k) to ${net.name}`
                  });
                }
              }
            }
          }
        }
      }

      return violations;
    }
  },
  {
    id: 'BP003',
    name: 'Reference Designator Sequence',
    description: 'Reference designators should be sequential',
    check: (schematic: Schematic): Violation[] => {
      const violations: Violation[] = [];
      const refGroups = new Map<string, number[]>();

      for (const component of schematic.components) {
        const match = component.reference.match(/^([A-Z]+)(\d+)$/);
        if (match) {
          const prefix = match[1];
          const number = parseInt(match[2]);
          if (!refGroups.has(prefix)) {
            refGroups.set(prefix, []);
          }
          refGroups.get(prefix)!.push(number);
        }
      }

      for (const [prefix, numbers] of refGroups) {
        numbers.sort((a, b) => a - b);
        const gaps: number[] = [];
        for (let i = 1; i < numbers.length; i++) {
          if (numbers[i] - numbers[i - 1] > 1) {
            for (let j = numbers[i - 1] + 1; j < numbers[i]; j++) {
              gaps.push(j);
            }
          }
        }

        if (gaps.length > 0) {
          violations.push({
            id: uuidv4(),
            severity: 'warning',
            code: 'BP003',
            message: `Missing ${prefix} designators: ${gaps.join(', ')}`,
            suggestion: 'Run annotation to fix reference designator sequence'
          });
        }
      }

      return violations;
    }
  },
  {
    id: 'BP004',
    name: 'Crystal Load Capacitors',
    description: 'Crystals should have proper load capacitors',
    check: (schematic: Schematic): Violation[] => {
      const violations: Violation[] = [];
      const crystals = schematic.components.filter(c => c.reference.startsWith('Y'));

      for (const crystal of crystals) {
        const crystalNets = crystal.pins
          .filter(p => p.connectedNet)
          .map(p => p.connectedNet!);

        const loadCaps = schematic.components.filter(c => {
          if (!c.reference.startsWith('C')) return false;
          return c.pins.some(p => p.connectedNet && crystalNets.includes(p.connectedNet));
        });

        if (loadCaps.length < 2) {
          violations.push({
            id: uuidv4(),
            severity: 'warning',
            code: 'BP004',
            message: `Crystal ${crystal.reference} may be missing load capacitors (found ${loadCaps.length}, expected 2)`,
            location: {
              componentId: crystal.id,
              position: crystal.position
            },
            suggestion: `Add two load capacitors (typically 10-22pF) to ${crystal.reference}`
          });
        }
      }

      return violations;
    }
  },
  {
    id: 'BP005',
    name: 'ESD Protection',
    description: 'External interfaces should have ESD protection',
    check: (schematic: Schematic): Violation[] => {
      const violations: Violation[] = [];
      const externalConnectors = schematic.components.filter(c =>
        c.reference.startsWith('J') &&
        (c.description?.toLowerCase().includes('usb') ||
          c.description?.toLowerCase().includes('ethernet') ||
          c.description?.toLowerCase().includes('hdmi') ||
          c.description?.toLowerCase().includes('external'))
      );

      for (const connector of externalConnectors) {
        const connectorNets = connector.pins
          .filter(p => p.connectedNet && p.type !== 'power_input')
          .map(p => p.connectedNet!);

        const hasESD = schematic.components.some(c => {
          if (!c.description?.toLowerCase().includes('esd')) return false;
          return c.pins.some(p => p.connectedNet && connectorNets.includes(p.connectedNet));
        });

        if (!hasESD) {
          violations.push({
            id: uuidv4(),
            severity: 'warning',
            code: 'BP005',
            message: `External connector ${connector.reference} may need ESD protection`,
            location: {
              componentId: connector.id,
              position: connector.position
            },
            suggestion: `Add TVS diode or ESD protection IC near ${connector.reference}`
          });
        }
      }

      return violations;
    }
  }
];

// ============================================================================
// Schematic Reviewer
// ============================================================================

export class SchematicReviewer extends EventEmitter {
  private config: ReviewConfig;
  private ercRules: ERCRule[];
  private bestPracticesRules: ERCRule[];

  constructor(config: Partial<ReviewConfig> = {}) {
    super();
    this.config = {
      enableERC: config.enableERC !== false,
      enableBestPractices: config.enableBestPractices !== false,
      enableComponentValidation: config.enableComponentValidation !== false,
      enableNetClassValidation: config.enableNetClassValidation !== false,
      enablePowerValidation: config.enablePowerValidation !== false,
      enableSignalValidation: config.enableSignalValidation !== false,
      strictMode: config.strictMode || false,
      customRules: config.customRules || []
    };

    this.ercRules = [...ERC_RULES];
    this.bestPracticesRules = [...BEST_PRACTICES_RULES];
  }

  /**
   * Review a schematic and generate validation results
   */
  async review(schematic: Schematic): Promise<ReviewResult> {
    const startTime = Date.now();
    const domains: ValidationDomain[] = [];
    const recommendations: string[] = [];
    let totalScore = 0;
    let maxScore = 0;

    this.emit('review:start', { schematicId: schematic.id });
    logger.info('Starting schematic review', { schematicId: schematic.id });

    // Run ERC checks
    if (this.config.enableERC) {
      this.emit('review:progress', { phase: 'erc', progress: 20 });
      const ercDomain = await this.runERCChecks(schematic);
      domains.push(ercDomain);
      totalScore += ercDomain.score * ercDomain.weight;
      maxScore += 100 * ercDomain.weight;
    }

    // Run best practices checks
    if (this.config.enableBestPractices) {
      this.emit('review:progress', { phase: 'best_practices', progress: 40 });
      const bpDomain = await this.runBestPracticesChecks(schematic);
      domains.push(bpDomain);
      totalScore += bpDomain.score * bpDomain.weight;
      maxScore += 100 * bpDomain.weight;
    }

    // Run component validation
    if (this.config.enableComponentValidation) {
      this.emit('review:progress', { phase: 'component_validation', progress: 60 });
      const compDomain = await this.runComponentValidation(schematic);
      domains.push(compDomain);
      totalScore += compDomain.score * compDomain.weight;
      maxScore += 100 * compDomain.weight;
    }

    // Run power validation
    if (this.config.enablePowerValidation) {
      this.emit('review:progress', { phase: 'power_validation', progress: 80 });
      const powerDomain = await this.runPowerValidation(schematic);
      domains.push(powerDomain);
      totalScore += powerDomain.score * powerDomain.weight;
      maxScore += 100 * powerDomain.weight;
    }

    // Run custom rules
    if (this.config.customRules && this.config.customRules.length > 0) {
      this.emit('review:progress', { phase: 'custom_rules', progress: 90 });
      const customDomain = await this.runCustomRules(schematic);
      domains.push(customDomain);
      totalScore += customDomain.score * customDomain.weight;
      maxScore += 100 * customDomain.weight;
    }

    // Calculate final score
    const finalScore = maxScore > 0 ? Math.round((totalScore / maxScore) * 100) : 100;
    const passed = this.config.strictMode
      ? domains.every(d => d.passed)
      : finalScore >= 70 && !domains.some(d => d.violations.some(v => v.severity === 'critical'));

    // Generate recommendations
    recommendations.push(...this.generateRecommendations(domains));

    // Count issues by severity
    const estimatedIssueCount = {
      critical: domains.flatMap(d => d.violations).filter(v => v.severity === 'critical').length,
      error: domains.flatMap(d => d.violations).filter(v => v.severity === 'error').length,
      warning: domains.flatMap(d => d.violations).filter(v => v.severity === 'warning').length
    };

    const result: ReviewResult = {
      schematicId: schematic.id,
      passed,
      score: finalScore,
      validationResults: {
        passed,
        score: finalScore,
        timestamp: new Date().toISOString(),
        domains
      },
      recommendations,
      estimatedIssueCount,
      reviewedAt: new Date().toISOString(),
      reviewDuration: Date.now() - startTime
    };

    this.emit('review:progress', { phase: 'complete', progress: 100 });
    this.emit('review:complete', { result });

    logger.info('Schematic review complete', {
      schematicId: schematic.id,
      passed,
      score: finalScore,
      duration: Date.now() - startTime
    });

    return result;
  }

  /**
   * Run ERC checks
   */
  private async runERCChecks(schematic: Schematic): Promise<ValidationDomain> {
    const violations: Violation[] = [];
    const warnings: Warning[] = [];

    for (const rule of this.ercRules) {
      const ruleViolations = rule.check(schematic);
      violations.push(...ruleViolations);
    }

    // Calculate score (deduct points for violations)
    let score = 100;
    for (const violation of violations) {
      switch (violation.severity) {
        case 'critical':
          score -= 25;
          break;
        case 'error':
          score -= 10;
          break;
        case 'warning':
          score -= 3;
          break;
      }
    }

    return {
      name: 'Electrical Rules Check (ERC)',
      type: 'erc',
      passed: score >= 70 && !violations.some(v => v.severity === 'critical'),
      score: Math.max(0, score),
      weight: 0.3,
      violations,
      warnings
    };
  }

  /**
   * Run best practices checks
   */
  private async runBestPracticesChecks(schematic: Schematic): Promise<ValidationDomain> {
    const violations: Violation[] = [];
    const warnings: Warning[] = [];

    for (const rule of this.bestPracticesRules) {
      const ruleViolations = rule.check(schematic);
      violations.push(...ruleViolations);
    }

    let score = 100;
    for (const violation of violations) {
      switch (violation.severity) {
        case 'critical':
          score -= 20;
          break;
        case 'error':
          score -= 8;
          break;
        case 'warning':
          score -= 2;
          break;
      }
    }

    return {
      name: 'Best Practices',
      type: 'best_practices',
      passed: score >= 60,
      score: Math.max(0, score),
      weight: 0.2,
      violations,
      warnings
    };
  }

  /**
   * Run component validation
   */
  private async runComponentValidation(schematic: Schematic): Promise<ValidationDomain> {
    const violations: Violation[] = [];
    const warnings: Warning[] = [];

    // Check for missing values
    for (const component of schematic.components) {
      if (!component.value || component.value === '?') {
        violations.push({
          id: uuidv4(),
          severity: 'error',
          code: 'COMP001',
          message: `Component ${component.reference} has no value`,
          location: { componentId: component.id },
          suggestion: 'Add a value for this component'
        });
      }

      // Check for missing footprints
      if (!component.footprint || component.footprint === '?') {
        violations.push({
          id: uuidv4(),
          severity: 'error',
          code: 'COMP002',
          message: `Component ${component.reference} has no footprint`,
          location: { componentId: component.id },
          suggestion: 'Assign a footprint to this component'
        });
      }

      // Check for obsolete components
      if (component.properties.status === 'obsolete') {
        violations.push({
          id: uuidv4(),
          severity: 'warning',
          code: 'COMP003',
          message: `Component ${component.reference} (${component.partNumber}) is obsolete`,
          location: { componentId: component.id },
          suggestion: 'Consider using an alternative part'
        });
      }
    }

    let score = 100;
    score -= violations.filter(v => v.severity === 'error').length * 10;
    score -= violations.filter(v => v.severity === 'warning').length * 3;

    return {
      name: 'Component Validation',
      type: 'best_practices',
      passed: score >= 70,
      score: Math.max(0, score),
      weight: 0.2,
      violations,
      warnings
    };
  }

  /**
   * Run power validation
   */
  private async runPowerValidation(schematic: Schematic): Promise<ValidationDomain> {
    const violations: Violation[] = [];
    const warnings: Warning[] = [];

    // Check for power net naming
    const powerNets = schematic.nets.filter(n =>
      n.name.includes('VDD') || n.name.includes('VCC') ||
      n.name.includes('GND') || n.name.includes('VSS') ||
      n.name.includes('3V3') || n.name.includes('5V')
    );

    // Check each power net has multiple connections
    for (const net of powerNets) {
      if (net.connections.length < 2) {
        violations.push({
          id: uuidv4(),
          severity: 'warning',
          code: 'PWR001',
          message: `Power net ${net.name} has only ${net.connections.length} connection(s)`,
          location: { netId: net.id },
          suggestion: 'Verify power net connections'
        });
      }
    }

    // Check for bulk capacitors on power rails
    const hasBulkCap = schematic.components.some(c =>
      c.reference.startsWith('C') &&
      c.value.includes('uF') &&
      parseFloat(c.value) >= 10
    );

    if (!hasBulkCap) {
      warnings.push({
        code: 'PWR002',
        message: 'No bulk capacitors found on power rails'
      });
    }

    let score = 100;
    score -= violations.filter(v => v.severity === 'critical').length * 25;
    score -= violations.filter(v => v.severity === 'error').length * 10;
    score -= violations.filter(v => v.severity === 'warning').length * 3;
    score -= warnings.length * 2;

    return {
      name: 'Power Validation',
      type: 'best_practices',
      passed: score >= 70,
      score: Math.max(0, score),
      weight: 0.2,
      violations,
      warnings
    };
  }

  /**
   * Run custom rules
   */
  private async runCustomRules(schematic: Schematic): Promise<ValidationDomain> {
    const violations: Violation[] = [];
    const warnings: Warning[] = [];

    for (const rule of this.config.customRules || []) {
      const ruleViolations = rule.check(schematic);
      for (const rv of ruleViolations) {
        violations.push({
          id: uuidv4(),
          severity: rule.severity,
          code: rule.id,
          message: rv.message,
          location: {
            componentId: rv.componentId,
            netId: rv.netId
          },
          suggestion: rv.suggestion
        });
      }
    }

    let score = 100;
    score -= violations.filter(v => v.severity === 'critical').length * 20;
    score -= violations.filter(v => v.severity === 'error').length * 8;
    score -= violations.filter(v => v.severity === 'warning').length * 2;

    return {
      name: 'Custom Rules',
      type: 'best_practices',
      passed: score >= 70,
      score: Math.max(0, score),
      weight: 0.1,
      violations,
      warnings
    };
  }

  /**
   * Generate recommendations based on review results
   */
  private generateRecommendations(domains: ValidationDomain[]): string[] {
    const recommendations: string[] = [];

    for (const domain of domains) {
      if (domain.score < 70) {
        recommendations.push(`Focus on improving ${domain.name} (current score: ${domain.score}%)`);
      }

      const criticalCount = domain.violations.filter(v => v.severity === 'critical').length;
      if (criticalCount > 0) {
        recommendations.push(`Address ${criticalCount} critical issue(s) in ${domain.name} immediately`);
      }
    }

    // Add general recommendations
    const totalViolations = domains.flatMap(d => d.violations).length;
    if (totalViolations > 20) {
      recommendations.push('Consider breaking the schematic into multiple sheets for better organization');
    }

    return recommendations;
  }

  /**
   * Add a custom rule
   */
  addCustomRule(rule: CustomRule): void {
    if (!this.config.customRules) {
      this.config.customRules = [];
    }
    this.config.customRules.push(rule);
  }

  /**
   * Add an ERC rule
   */
  addERCRule(rule: ERCRule): void {
    this.ercRules.push(rule);
  }

  /**
   * Add a best practices rule
   */
  addBestPracticesRule(rule: ERCRule): void {
    this.bestPracticesRules.push(rule);
  }
}

export default SchematicReviewer;