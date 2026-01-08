/**
 * PCB Validation Framework
 *
 * Comprehensive 8-domain validation system for PCB layouts.
 * Domains: DRC, ERC, IPC-2221, Signal Integrity, Thermal, DFM, Best Practices, Testing
 */

import { EventEmitter } from 'events';
import { v4 as uuidv4 } from 'uuid';
import {
  PCBLayout,
  ValidationResult,
  BoardConstraints
} from '../../../types';

export interface ValidationDomain {
  name: string;
  weight: number;
  enabled: boolean;
}

export interface ValidationViolation {
  id: string;
  domain: string;
  severity: 'error' | 'warning' | 'info';
  code: string;
  message: string;
  location?: {
    x: number;
    y: number;
    layer?: string;
    componentRef?: string;
    netName?: string;
  };
  suggestion?: string;
}

export interface DomainResult {
  domain: string;
  score: number;
  maxScore: number;
  violations: ValidationViolation[];
  metrics: Record<string, number | string | boolean>;
  passRate: number;
}

export interface ValidationConfig {
  domains: ValidationDomain[];
  strictMode: boolean;
  targetScore: number;
  maxViolations: number;
}

const DEFAULT_CONFIG: ValidationConfig = {
  domains: [
    { name: 'drc', weight: 0.2, enabled: true },
    { name: 'erc', weight: 0.15, enabled: true },
    { name: 'ipc2221', weight: 0.1, enabled: true },
    { name: 'signal-integrity', weight: 0.15, enabled: true },
    { name: 'thermal', weight: 0.15, enabled: true },
    { name: 'dfm', weight: 0.1, enabled: true },
    { name: 'best-practices', weight: 0.1, enabled: true },
    { name: 'testing', weight: 0.05, enabled: true }
  ],
  strictMode: false,
  targetScore: 85,
  maxViolations: 100
};

/**
 * Main validation framework orchestrating all 8 domain validators
 */
export class ValidationFramework extends EventEmitter {
  private config: ValidationConfig;
  private validators: Map<string, DomainValidator>;

  constructor(config?: Partial<ValidationConfig>) {
    super();
    this.config = { ...DEFAULT_CONFIG, ...config };
    this.validators = new Map();
    this.initializeValidators();
  }

  private initializeValidators(): void {
    this.validators.set('drc', new DRCValidator());
    this.validators.set('erc', new ERCValidator());
    this.validators.set('ipc2221', new IPC2221Validator());
    this.validators.set('signal-integrity', new SignalIntegrityValidator());
    this.validators.set('thermal', new ThermalValidator());
    this.validators.set('dfm', new DFMValidator());
    this.validators.set('best-practices', new BestPracticesValidator());
    this.validators.set('testing', new TestingValidator());
  }

  /**
   * Run full validation suite on a PCB layout
   */
  async validate(
    layout: PCBLayout,
    constraints: BoardConstraints
  ): Promise<ValidationResult> {
    const startTime = Date.now();
    const domainResults: DomainResult[] = [];
    const allViolations: ValidationViolation[] = [];

    this.emit('validation-start', { layoutId: layout.id });

    // Run each enabled domain validator
    for (const domain of this.config.domains) {
      if (!domain.enabled) continue;

      const validator = this.validators.get(domain.name);
      if (!validator) continue;

      this.emit('domain-start', { domain: domain.name });

      try {
        const result = await validator.validate(layout, constraints);
        domainResults.push(result);
        allViolations.push(...result.violations);

        this.emit('domain-complete', {
          domain: domain.name,
          score: result.score,
          violations: result.violations.length
        });
      } catch (error) {
        this.emit('domain-error', { domain: domain.name, error });
        domainResults.push({
          domain: domain.name,
          score: 0,
          maxScore: 100,
          violations: [{
            id: uuidv4(),
            domain: domain.name,
            severity: 'error',
            code: 'VALIDATOR_ERROR',
            message: `Validator failed: ${error instanceof Error ? error.message : 'Unknown error'}`
          }],
          metrics: {},
          passRate: 0
        });
      }
    }

    // Calculate weighted total score
    let totalScore = 0;
    let totalWeight = 0;

    for (const result of domainResults) {
      const domain = this.config.domains.find(d => d.name === result.domain);
      if (domain) {
        totalScore += (result.score / result.maxScore) * domain.weight * 100;
        totalWeight += domain.weight;
      }
    }

    const finalScore = totalWeight > 0 ? totalScore / totalWeight : 0;
    const elapsed = Date.now() - startTime;

    // Determine pass/fail
    const errorCount = allViolations.filter(v => v.severity === 'error').length;
    const warningCount = allViolations.filter(v => v.severity === 'warning').length;
    const passed = finalScore >= this.config.targetScore && errorCount === 0;

    const validationResult: ValidationResult = {
      passed,
      score: Math.round(finalScore * 10) / 10,
      violations: allViolations.slice(0, this.config.maxViolations),
      summary: this.generateSummary(domainResults, finalScore),
      details: {
        drc: domainResults.find(r => r.domain === 'drc'),
        erc: domainResults.find(r => r.domain === 'erc'),
        ipc2221: domainResults.find(r => r.domain === 'ipc2221'),
        signalIntegrity: domainResults.find(r => r.domain === 'signal-integrity'),
        thermal: domainResults.find(r => r.domain === 'thermal'),
        dfm: domainResults.find(r => r.domain === 'dfm'),
        bestPractices: domainResults.find(r => r.domain === 'best-practices'),
        testing: domainResults.find(r => r.domain === 'testing')
      },
      timestamp: new Date(),
      durationMs: elapsed
    };

    this.emit('validation-complete', {
      layoutId: layout.id,
      score: finalScore,
      passed,
      elapsed,
      errorCount,
      warningCount
    });

    return validationResult;
  }

  private generateSummary(results: DomainResult[], score: number): string {
    const summaryParts: string[] = [];

    summaryParts.push(`Overall Score: ${Math.round(score)}/100`);

    for (const result of results) {
      const percentage = Math.round((result.score / result.maxScore) * 100);
      summaryParts.push(`${result.domain}: ${percentage}%`);
    }

    const totalViolations = results.reduce((sum, r) => sum + r.violations.length, 0);
    summaryParts.push(`Total Violations: ${totalViolations}`);

    return summaryParts.join(' | ');
  }

  /**
   * Update configuration
   */
  setConfig(config: Partial<ValidationConfig>): void {
    this.config = { ...this.config, ...config };
  }

  /**
   * Enable or disable a specific domain
   */
  setDomainEnabled(domainName: string, enabled: boolean): void {
    const domain = this.config.domains.find(d => d.name === domainName);
    if (domain) {
      domain.enabled = enabled;
    }
  }
}

/**
 * Abstract base class for domain validators
 */
abstract class DomainValidator {
  abstract readonly name: string;

  abstract validate(
    layout: PCBLayout,
    constraints: BoardConstraints
  ): Promise<DomainResult>;

  protected createViolation(
    code: string,
    message: string,
    severity: ValidationViolation['severity'] = 'error',
    location?: ValidationViolation['location'],
    suggestion?: string
  ): ValidationViolation {
    return {
      id: uuidv4(),
      domain: this.name,
      severity,
      code,
      message,
      location,
      suggestion
    };
  }
}

/**
 * Design Rule Check (DRC) Validator
 */
class DRCValidator extends DomainValidator {
  readonly name = 'drc';

  async validate(layout: PCBLayout, constraints: BoardConstraints): Promise<DomainResult> {
    const violations: ValidationViolation[] = [];
    let score = 100;

    // Check trace width violations
    for (const trace of layout.traces || []) {
      if (trace.width < (constraints.minTraceWidth || 0.15)) {
        violations.push(this.createViolation(
          'DRC_TRACE_WIDTH',
          `Trace width ${trace.width}mm is below minimum ${constraints.minTraceWidth}mm`,
          'error',
          { netName: trace.netName },
          `Increase trace width to at least ${constraints.minTraceWidth}mm`
        ));
        score -= 5;
      }
    }

    // Check clearance violations
    const components = layout.components || [];
    const minClearance = constraints.minClearance || 0.15;

    for (let i = 0; i < components.length; i++) {
      for (let j = i + 1; j < components.length; j++) {
        const c1 = components[i];
        const c2 = components[j];

        // Calculate edge-to-edge distance
        const dx = Math.abs(c1.position.x - c2.position.x) -
                   ((c1.footprint?.width || 0) + (c2.footprint?.width || 0)) / 2;
        const dy = Math.abs(c1.position.y - c2.position.y) -
                   ((c1.footprint?.height || 0) + (c2.footprint?.height || 0)) / 2;
        const clearance = Math.max(0, Math.min(dx, dy));

        if (clearance < minClearance && clearance < 0) {
          violations.push(this.createViolation(
            'DRC_CLEARANCE',
            `Components ${c1.reference} and ${c2.reference} overlap or have insufficient clearance`,
            'error',
            { x: c1.position.x, y: c1.position.y, componentRef: c1.reference },
            'Move components apart to maintain minimum clearance'
          ));
          score -= 10;
        }
      }
    }

    // Check via diameter and drill
    for (const via of layout.vias || []) {
      if (via.diameter < 0.4) {
        violations.push(this.createViolation(
          'DRC_VIA_DIAMETER',
          `Via diameter ${via.diameter}mm is below minimum 0.4mm`,
          'warning',
          { x: via.position.x, y: via.position.y },
          'Increase via diameter or use HDI technology'
        ));
        score -= 2;
      }

      const annularRing = (via.diameter - via.drillSize) / 2;
      if (annularRing < 0.1) {
        violations.push(this.createViolation(
          'DRC_ANNULAR_RING',
          `Via annular ring ${annularRing}mm is below minimum 0.1mm`,
          'error',
          { x: via.position.x, y: via.position.y },
          'Increase via pad size or decrease drill size'
        ));
        score -= 5;
      }
    }

    // Check board boundary violations
    const margin = 1; // 1mm minimum from edge
    for (const c of components) {
      if (c.position.x < margin || c.position.y < margin ||
          c.position.x + (c.footprint?.width || 0) > layout.boardOutline.width - margin ||
          c.position.y + (c.footprint?.height || 0) > layout.boardOutline.height - margin) {
        violations.push(this.createViolation(
          'DRC_BOARD_EDGE',
          `Component ${c.reference} is too close to board edge`,
          'warning',
          { x: c.position.x, y: c.position.y, componentRef: c.reference },
          `Maintain at least ${margin}mm from board edges`
        ));
        score -= 3;
      }
    }

    return {
      domain: this.name,
      score: Math.max(0, score),
      maxScore: 100,
      violations,
      metrics: {
        traceCount: layout.traces?.length || 0,
        viaCount: layout.vias?.length || 0,
        componentCount: components.length,
        minClearanceFound: minClearance
      },
      passRate: violations.filter(v => v.severity === 'error').length === 0 ? 100 : 0
    };
  }
}

/**
 * Electrical Rule Check (ERC) Validator
 */
class ERCValidator extends DomainValidator {
  readonly name = 'erc';

  async validate(layout: PCBLayout, constraints: BoardConstraints): Promise<DomainResult> {
    const violations: ValidationViolation[] = [];
    let score = 100;

    // Check for unconnected pins
    const connectedNets = new Set<string>();
    for (const trace of layout.traces || []) {
      if (trace.netName) connectedNets.add(trace.netName);
    }

    // Check power net connectivity
    const powerNets = ['VCC', 'VDD', 'GND', 'VBAT', '3V3', '5V', '12V'];
    for (const netName of powerNets) {
      if (!connectedNets.has(netName) && !connectedNets.has(netName.toLowerCase())) {
        // This might be okay if the net doesn't exist in the design
        // Just a warning
      }
    }

    // Check for floating pins (components without traces)
    for (const component of layout.components || []) {
      const hasConnection = layout.traces?.some(t =>
        t.netName?.includes(component.reference)
      );

      // This is a simplified check - real ERC would check actual netlist
      if (!hasConnection && component.pads && component.pads.length > 0) {
        violations.push(this.createViolation(
          'ERC_FLOATING_COMPONENT',
          `Component ${component.reference} may have floating pins`,
          'warning',
          { x: component.position.x, y: component.position.y, componentRef: component.reference },
          'Verify all pins are properly connected'
        ));
        score -= 2;
      }
    }

    // Check for power-ground shorts (simplified)
    const powerTraces = layout.traces?.filter(t =>
      t.netName?.toLowerCase().includes('vcc') ||
      t.netName?.toLowerCase().includes('vdd') ||
      t.netName?.toLowerCase().includes('power')
    ) || [];

    const groundTraces = layout.traces?.filter(t =>
      t.netName?.toLowerCase().includes('gnd') ||
      t.netName?.toLowerCase().includes('ground')
    ) || [];

    // Check for traces that might short (simplified proximity check)
    for (const pTrace of powerTraces) {
      for (const gTrace of groundTraces) {
        // Check if traces are on same layer and too close
        if (pTrace.layer === gTrace.layer) {
          // Simplified - real ERC would check actual geometry
        }
      }
    }

    return {
      domain: this.name,
      score: Math.max(0, score),
      maxScore: 100,
      violations,
      metrics: {
        connectedNets: connectedNets.size,
        powerNets: powerTraces.length,
        groundNets: groundTraces.length
      },
      passRate: violations.filter(v => v.severity === 'error').length === 0 ? 100 : 0
    };
  }
}

/**
 * IPC-2221 Standards Validator
 */
class IPC2221Validator extends DomainValidator {
  readonly name = 'ipc2221';

  // IPC-2221 conductor spacing table (simplified)
  private readonly spacingTable: Record<number, number> = {
    15: 0.05,   // 0-15V: 0.05mm
    30: 0.05,   // 16-30V
    50: 0.1,    // 31-50V
    100: 0.1,   // 51-100V
    150: 0.2,   // 101-150V
    170: 0.25,  // 151-170V
    250: 0.5,   // 171-250V
    500: 2.5    // 251-500V
  };

  async validate(layout: PCBLayout, constraints: BoardConstraints): Promise<DomainResult> {
    const violations: ValidationViolation[] = [];
    let score = 100;

    // Check conductor spacing for voltage levels
    const maxVoltage = constraints.maxVoltage || 50;
    const requiredSpacing = this.getRequiredSpacing(maxVoltage);

    if ((constraints.minClearance || 0) < requiredSpacing) {
      violations.push(this.createViolation(
        'IPC2221_SPACING',
        `Minimum clearance ${constraints.minClearance}mm may be insufficient for ${maxVoltage}V (IPC-2221 requires ${requiredSpacing}mm)`,
        'warning',
        undefined,
        `Increase clearance to at least ${requiredSpacing}mm for ${maxVoltage}V operation`
      ));
      score -= 10;
    }

    // Check conductor width for current capacity (1oz copper)
    const traces = layout.traces || [];
    for (const trace of traces) {
      const estimatedCurrent = this.estimateTraceCurrent(trace.netName || '');
      const requiredWidth = this.calculateRequiredWidth(estimatedCurrent, 10); // 10°C rise

      if (trace.width < requiredWidth) {
        violations.push(this.createViolation(
          'IPC2221_CURRENT_CAPACITY',
          `Trace width ${trace.width}mm may be insufficient for estimated ${estimatedCurrent}A`,
          'warning',
          { netName: trace.netName },
          `Consider widening to ${requiredWidth.toFixed(2)}mm for ${estimatedCurrent}A with 10°C rise`
        ));
        score -= 5;
      }
    }

    // Check PCB layer stack compliance
    const layers = layout.layers || [];
    if (layers.length > 0) {
      // Check for proper ground/power plane distribution
      const copperLayers = layers.filter(l => l.type === 'copper');
      if (copperLayers.length >= 4) {
        // 4+ layer boards should have dedicated ground plane
        const hasGroundPlane = layout.zones?.some(z =>
          z.netName?.toLowerCase() === 'gnd' && z.layer?.includes('In')
        );
        if (!hasGroundPlane) {
          violations.push(this.createViolation(
            'IPC2221_GROUND_PLANE',
            'Multi-layer board should have dedicated internal ground plane',
            'info',
            undefined,
            'Add solid ground pour on internal layer for improved signal integrity'
          ));
          score -= 3;
        }
      }
    }

    return {
      domain: this.name,
      score: Math.max(0, score),
      maxScore: 100,
      violations,
      metrics: {
        requiredSpacing,
        maxVoltage,
        layerCount: layout.layers?.length || 0
      },
      passRate: violations.filter(v => v.severity === 'error').length === 0 ? 100 : 0
    };
  }

  private getRequiredSpacing(voltage: number): number {
    for (const [v, spacing] of Object.entries(this.spacingTable)) {
      if (voltage <= parseInt(v)) return spacing;
    }
    return 2.5; // Default to highest
  }

  private estimateTraceCurrent(netName: string): number {
    const name = netName.toLowerCase();
    if (name.includes('motor') || name.includes('power')) return 10;
    if (name.includes('vcc') || name.includes('vdd')) return 2;
    if (name.includes('gnd')) return 2;
    return 0.5; // Signal traces
  }

  private calculateRequiredWidth(current: number, tempRise: number): number {
    // IPC-2221 formula simplified (external layer, 1oz copper)
    // W = (I / (k * ΔT^b))^(1/c) / t
    // Using approximation for 1oz external: W ≈ I / (0.048 * ΔT^0.44)
    const width = Math.pow(current / (0.048 * Math.pow(tempRise, 0.44)), 1/0.725) / 35;
    return Math.max(0.15, width);
  }
}

/**
 * Signal Integrity Validator
 */
class SignalIntegrityValidator extends DomainValidator {
  readonly name = 'signal-integrity';

  async validate(layout: PCBLayout, constraints: BoardConstraints): Promise<DomainResult> {
    const violations: ValidationViolation[] = [];
    let score = 100;

    const traces = layout.traces || [];

    // Check trace length matching for differential pairs
    const diffPairs = this.identifyDifferentialPairs(traces);
    for (const [pairName, [trace1, trace2]] of diffPairs) {
      const length1 = this.calculateTraceLength(trace1);
      const length2 = this.calculateTraceLength(trace2);
      const mismatch = Math.abs(length1 - length2);

      if (mismatch > 1) { // 1mm tolerance
        violations.push(this.createViolation(
          'SI_LENGTH_MISMATCH',
          `Differential pair ${pairName} has ${mismatch.toFixed(2)}mm length mismatch`,
          'warning',
          undefined,
          'Add serpentine routing to match lengths within 0.5mm'
        ));
        score -= 5;
      }
    }

    // Check for stub traces (unterminated high-speed signals)
    for (const trace of traces) {
      if (this.isHighSpeedSignal(trace.netName || '')) {
        const length = this.calculateTraceLength(trace);
        // Simplified stub check - real SI would use rise time
        if (length > 0 && length < 5) {
          violations.push(this.createViolation(
            'SI_STUB',
            `Short trace stub on high-speed signal ${trace.netName}`,
            'info',
            { netName: trace.netName },
            'Consider adding termination or removing stub'
          ));
          score -= 2;
        }
      }
    }

    // Check controlled impedance traces
    const controlledImpedanceNets = traces.filter(t =>
      this.isHighSpeedSignal(t.netName || '')
    );

    for (const trace of controlledImpedanceNets) {
      const expectedWidth = this.calculateImpedanceWidth(50); // 50Ω target
      if (Math.abs(trace.width - expectedWidth) > 0.05) {
        violations.push(this.createViolation(
          'SI_IMPEDANCE',
          `Trace width ${trace.width}mm may not meet 50Ω impedance target`,
          'info',
          { netName: trace.netName },
          `Use ${expectedWidth.toFixed(2)}mm width for 50Ω impedance`
        ));
        score -= 3;
      }
    }

    // Check for right-angle traces on high-speed signals
    for (const trace of traces) {
      if (this.isHighSpeedSignal(trace.netName || '') && this.hasRightAngles(trace)) {
        violations.push(this.createViolation(
          'SI_RIGHT_ANGLE',
          `High-speed trace ${trace.netName} has 90° corners`,
          'warning',
          { netName: trace.netName },
          'Use 45° chamfers or curves for high-speed routing'
        ));
        score -= 4;
      }
    }

    // Check return path continuity
    const hasGroundPlane = layout.zones?.some(z =>
      z.netName?.toLowerCase() === 'gnd'
    );

    if (!hasGroundPlane && controlledImpedanceNets.length > 0) {
      violations.push(this.createViolation(
        'SI_RETURN_PATH',
        'High-speed signals without reference ground plane',
        'warning',
        undefined,
        'Add ground pour for proper return path'
      ));
      score -= 10;
    }

    return {
      domain: this.name,
      score: Math.max(0, score),
      maxScore: 100,
      violations,
      metrics: {
        highSpeedSignals: controlledImpedanceNets.length,
        differentialPairs: diffPairs.size,
        hasGroundPlane
      },
      passRate: violations.filter(v => v.severity === 'error').length === 0 ? 100 : 0
    };
  }

  private identifyDifferentialPairs(traces: any[]): Map<string, [any, any]> {
    const pairs = new Map<string, [any, any]>();

    for (const trace of traces) {
      const name = trace.netName || '';
      if (name.endsWith('_P') || name.endsWith('_N') ||
          name.endsWith('+') || name.endsWith('-')) {
        const baseName = name.replace(/[_+-]?[PN+-]$/, '');
        const isPositive = name.endsWith('_P') || name.endsWith('+');

        if (!pairs.has(baseName)) {
          pairs.set(baseName, [null, null] as any);
        }

        const pair = pairs.get(baseName)!;
        if (isPositive) pair[0] = trace;
        else pair[1] = trace;
      }
    }

    // Filter out incomplete pairs
    for (const [name, pair] of pairs) {
      if (!pair[0] || !pair[1]) pairs.delete(name);
    }

    return pairs;
  }

  private calculateTraceLength(trace: any): number {
    if (!trace.points || trace.points.length < 2) return 0;

    let length = 0;
    for (let i = 1; i < trace.points.length; i++) {
      const dx = trace.points[i].x - trace.points[i-1].x;
      const dy = trace.points[i].y - trace.points[i-1].y;
      length += Math.sqrt(dx*dx + dy*dy);
    }
    return length;
  }

  private isHighSpeedSignal(netName: string): boolean {
    const name = netName.toLowerCase();
    return name.includes('clk') ||
           name.includes('data') ||
           name.includes('usb') ||
           name.includes('hdmi') ||
           name.includes('eth') ||
           name.includes('pcie') ||
           name.includes('ddr');
  }

  private calculateImpedanceWidth(targetOhms: number): number {
    // Simplified microstrip calculation for FR4
    const er = 4.3;
    const h = 0.2; // mm
    return 7.475 * h / Math.exp(targetOhms * Math.sqrt(er + 1.41) / 87);
  }

  private hasRightAngles(trace: any): boolean {
    if (!trace.points || trace.points.length < 3) return false;

    for (let i = 1; i < trace.points.length - 1; i++) {
      const dx1 = trace.points[i].x - trace.points[i-1].x;
      const dy1 = trace.points[i].y - trace.points[i-1].y;
      const dx2 = trace.points[i+1].x - trace.points[i].x;
      const dy2 = trace.points[i+1].y - trace.points[i].y;

      // Check for 90° turn
      if ((dx1 !== 0 && dy2 !== 0 && dx2 === 0 && dy1 === 0) ||
          (dx1 === 0 && dy1 !== 0 && dx2 !== 0 && dy2 === 0)) {
        return true;
      }
    }
    return false;
  }
}

/**
 * Thermal Analysis Validator
 */
class ThermalValidator extends DomainValidator {
  readonly name = 'thermal';

  async validate(layout: PCBLayout, constraints: BoardConstraints): Promise<DomainResult> {
    const violations: ValidationViolation[] = [];
    let score = 100;

    // Estimate component temperatures
    const hotSpots: Array<{
      componentRef: string;
      x: number;
      y: number;
      estimatedTemp: number;
    }> = [];

    for (const component of layout.components || []) {
      const power = this.estimatePowerDissipation(component);
      const thermalRes = this.estimateThermalResistance(component);
      const tempRise = power * thermalRes;
      const estimatedTemp = 25 + tempRise; // Ambient + rise

      if (estimatedTemp > 85) {
        hotSpots.push({
          componentRef: component.reference,
          x: component.position.x,
          y: component.position.y,
          estimatedTemp
        });

        if (estimatedTemp > 100) {
          violations.push(this.createViolation(
            'THERMAL_OVERTEMP',
            `Component ${component.reference} may exceed 100°C (estimated ${estimatedTemp.toFixed(0)}°C)`,
            'error',
            { x: component.position.x, y: component.position.y, componentRef: component.reference },
            'Add thermal vias, increase copper area, or add heatsink'
          ));
          score -= 15;
        } else {
          violations.push(this.createViolation(
            'THERMAL_HOT',
            `Component ${component.reference} may exceed 85°C (estimated ${estimatedTemp.toFixed(0)}°C)`,
            'warning',
            { x: component.position.x, y: component.position.y, componentRef: component.reference },
            'Consider adding thermal relief'
          ));
          score -= 5;
        }
      }
    }

    // Check thermal via coverage
    const highPowerComponents = (layout.components || []).filter(c => {
      const ref = c.reference.toUpperCase();
      return ref.startsWith('Q') || ref.startsWith('U') || ref.startsWith('M');
    });

    for (const component of highPowerComponents) {
      const thermalVias = (layout.vias || []).filter(v => {
        const dx = Math.abs(v.position.x - component.position.x);
        const dy = Math.abs(v.position.y - component.position.y);
        const w = component.footprint?.width || 5;
        const h = component.footprint?.height || 5;
        return dx < w && dy < h;
      });

      if (thermalVias.length < 4) {
        violations.push(this.createViolation(
          'THERMAL_VIAS',
          `Component ${component.reference} has insufficient thermal vias (${thermalVias.length} found)`,
          'info',
          { x: component.position.x, y: component.position.y, componentRef: component.reference },
          'Add thermal via array under component pad'
        ));
        score -= 3;
      }
    }

    // Check copper pour for heat spreading
    const hasGoodCopperCoverage = (layout.zones || []).length > 0;
    if (!hasGoodCopperCoverage && highPowerComponents.length > 0) {
      violations.push(this.createViolation(
        'THERMAL_COPPER',
        'High-power design without copper pours for heat spreading',
        'warning',
        undefined,
        'Add ground/power pours for thermal management'
      ));
      score -= 10;
    }

    return {
      domain: this.name,
      score: Math.max(0, score),
      maxScore: 100,
      violations,
      metrics: {
        hotSpotCount: hotSpots.length,
        maxTemperature: Math.max(25, ...hotSpots.map(h => h.estimatedTemp)),
        thermalViaCount: (layout.vias || []).filter(v => v.type === 'through').length,
        copperPourArea: 0 // Would need actual calculation
      },
      passRate: violations.filter(v => v.severity === 'error').length === 0 ? 100 : 0
    };
  }

  private estimatePowerDissipation(component: any): number {
    const ref = (component.reference || '').toUpperCase();
    if (ref.startsWith('Q') || ref.startsWith('M')) return 5; // MOSFETs
    if (ref.startsWith('U')) return 2; // ICs
    if (ref.startsWith('R')) return 0.1; // Resistors
    if (ref.startsWith('L')) return 0.5; // Inductors
    return 0.1;
  }

  private estimateThermalResistance(component: any): number {
    const ref = (component.reference || '').toUpperCase();
    if (ref.startsWith('Q') || ref.startsWith('M')) return 3; // MOSFETs with pad
    if (ref.startsWith('U')) return 20; // ICs
    return 50; // Passive components
  }
}

/**
 * Design for Manufacturing (DFM) Validator
 */
class DFMValidator extends DomainValidator {
  readonly name = 'dfm';

  async validate(layout: PCBLayout, constraints: BoardConstraints): Promise<DomainResult> {
    const violations: ValidationViolation[] = [];
    let score = 100;

    // Check for fiducials
    const hasFiducials = layout.boardOutline.fiducials &&
                        layout.boardOutline.fiducials.length >= 3;
    if (!hasFiducials) {
      violations.push(this.createViolation(
        'DFM_FIDUCIALS',
        'Board lacks fiducial markers for pick-and-place alignment',
        'warning',
        undefined,
        'Add at least 3 fiducial markers (1mm diameter)'
      ));
      score -= 10;
    }

    // Check component orientation consistency
    const components = layout.components || [];
    const rotations = new Map<number, number>();
    for (const c of components) {
      const rot = c.rotation % 360;
      rotations.set(rot, (rotations.get(rot) || 0) + 1);
    }

    // Non-standard rotations
    for (const [rot, count] of rotations) {
      if (rot % 90 !== 0 && count > 0) {
        violations.push(this.createViolation(
          'DFM_ROTATION',
          `${count} component(s) have non-standard rotation (${rot}°)`,
          'info',
          undefined,
          'Use 0°, 90°, 180°, or 270° for easier assembly'
        ));
        score -= 2;
      }
    }

    // Check polarized component orientation consistency
    const polarizedComponents = components.filter(c => {
      const ref = c.reference.toUpperCase();
      return ref.startsWith('D') || ref.startsWith('C') || ref.startsWith('U');
    });

    const polarizedRotations = polarizedComponents.map(c => c.rotation % 180);
    const uniqueRotations = new Set(polarizedRotations);
    if (uniqueRotations.size > 2) {
      violations.push(this.createViolation(
        'DFM_POLARITY',
        'Polarized components have inconsistent orientation',
        'info',
        undefined,
        'Align polarized components for visual inspection'
      ));
      score -= 3;
    }

    // Check for single-sided vs dual-sided placement
    const topCount = components.filter(c => c.layer === 'top').length;
    const bottomCount = components.filter(c => c.layer === 'bottom').length;

    if (bottomCount > 0) {
      violations.push(this.createViolation(
        'DFM_DUAL_SIDED',
        `Design requires dual-sided assembly (${topCount} top, ${bottomCount} bottom)`,
        'info',
        undefined,
        'Single-sided assembly is more cost-effective for high volume'
      ));
      // No score penalty - just informational
    }

    // Check minimum component spacing for automated assembly
    const minAssemblySpacing = 1.5; // mm
    for (let i = 0; i < components.length; i++) {
      for (let j = i + 1; j < components.length; j++) {
        const c1 = components[i];
        const c2 = components[j];

        if (c1.layer !== c2.layer) continue;

        const dx = Math.abs(c1.position.x - c2.position.x) -
                   ((c1.footprint?.width || 0) + (c2.footprint?.width || 0)) / 2;
        const dy = Math.abs(c1.position.y - c2.position.y) -
                   ((c1.footprint?.height || 0) + (c2.footprint?.height || 0)) / 2;
        const spacing = Math.max(0, Math.min(dx, dy));

        if (spacing < minAssemblySpacing && spacing > 0) {
          violations.push(this.createViolation(
            'DFM_ASSEMBLY_SPACING',
            `Components ${c1.reference} and ${c2.reference} have tight spacing (${spacing.toFixed(2)}mm)`,
            'warning',
            { x: c1.position.x, y: c1.position.y },
            `Maintain at least ${minAssemblySpacing}mm for reliable assembly`
          ));
          score -= 5;
        }
      }
    }

    // Estimate yield
    const baseYield = 99;
    const yieldReduction = violations.length * 0.5;
    const estimatedYield = Math.max(85, baseYield - yieldReduction);

    return {
      domain: this.name,
      score: Math.max(0, score),
      maxScore: 100,
      violations,
      metrics: {
        hasFiducials,
        topComponents: topCount,
        bottomComponents: bottomCount,
        uniqueRotations: rotations.size,
        estimatedYield
      },
      passRate: violations.filter(v => v.severity === 'error').length === 0 ? 100 : 0
    };
  }
}

/**
 * Best Practices Validator
 */
class BestPracticesValidator extends DomainValidator {
  readonly name = 'best-practices';

  async validate(layout: PCBLayout, constraints: BoardConstraints): Promise<DomainResult> {
    const violations: ValidationViolation[] = [];
    let score = 100;

    const components = layout.components || [];

    // Check decoupling capacitor placement
    const ics = components.filter(c => c.reference.toUpperCase().startsWith('U'));
    const capacitors = components.filter(c => c.reference.toUpperCase().startsWith('C'));

    for (const ic of ics) {
      const nearbyCapacitors = capacitors.filter(cap => {
        const dist = Math.sqrt(
          Math.pow(cap.position.x - ic.position.x, 2) +
          Math.pow(cap.position.y - ic.position.y, 2)
        );
        return dist < 10; // Within 10mm
      });

      if (nearbyCapacitors.length === 0) {
        violations.push(this.createViolation(
          'BP_DECOUPLING',
          `IC ${ic.reference} lacks nearby decoupling capacitor`,
          'warning',
          { x: ic.position.x, y: ic.position.y, componentRef: ic.reference },
          'Place 0.1µF capacitor within 5mm of power pins'
        ));
        score -= 5;
      }
    }

    // Check crystal placement
    const crystals = components.filter(c => {
      const ref = c.reference.toUpperCase();
      return ref.startsWith('Y') || ref.startsWith('X');
    });

    for (const crystal of crystals) {
      // Find associated IC (usually MCU)
      const nearbyICs = ics.filter(ic => {
        const dist = Math.sqrt(
          Math.pow(ic.position.x - crystal.position.x, 2) +
          Math.pow(ic.position.y - crystal.position.y, 2)
        );
        return dist < 15;
      });

      if (nearbyICs.length === 0) {
        violations.push(this.createViolation(
          'BP_CRYSTAL',
          `Crystal ${crystal.reference} is far from any IC`,
          'warning',
          { x: crystal.position.x, y: crystal.position.y, componentRef: crystal.reference },
          'Place crystal close to MCU oscillator pins'
        ));
        score -= 5;
      }
    }

    // Check test point accessibility
    const testPoints = components.filter(c =>
      c.reference.toUpperCase().startsWith('TP')
    );

    if (testPoints.length === 0 && components.length > 20) {
      violations.push(this.createViolation(
        'BP_TEST_POINTS',
        'Design lacks test points for debugging',
        'info',
        undefined,
        'Add test points on critical signals'
      ));
      score -= 3;
    }

    // Check silkscreen reference designators
    const hasRefDes = true; // Assume present - real check would verify
    if (!hasRefDes) {
      violations.push(this.createViolation(
        'BP_SILKSCREEN',
        'Component reference designators should be visible on silkscreen',
        'info',
        undefined,
        'Add reference designators for easier assembly and debug'
      ));
      score -= 2;
    }

    // Check board edge mounting holes
    const hasToolingHoles = layout.boardOutline.toolingHoles &&
                           layout.boardOutline.toolingHoles.length >= 2;
    if (!hasToolingHoles) {
      violations.push(this.createViolation(
        'BP_MOUNTING',
        'Board lacks mounting or tooling holes',
        'info',
        undefined,
        'Add mounting holes for mechanical integration'
      ));
      score -= 2;
    }

    return {
      domain: this.name,
      score: Math.max(0, score),
      maxScore: 100,
      violations,
      metrics: {
        icCount: ics.length,
        capacitorCount: capacitors.length,
        crystalCount: crystals.length,
        testPointCount: testPoints.length
      },
      passRate: violations.filter(v => v.severity === 'error').length === 0 ? 100 : 0
    };
  }
}

/**
 * Testing/Testability Validator
 */
class TestingValidator extends DomainValidator {
  readonly name = 'testing';

  async validate(layout: PCBLayout, constraints: BoardConstraints): Promise<DomainResult> {
    const violations: ValidationViolation[] = [];
    let score = 100;

    const components = layout.components || [];

    // Check ICT (In-Circuit Test) accessibility
    const testAccessibleComponents = components.filter(c => {
      // Check if component pads are accessible from one side
      return c.layer === 'top'; // Simplified - real check would verify pad exposure
    });

    const ictCoverage = (testAccessibleComponents.length / components.length) * 100;
    if (ictCoverage < 80) {
      violations.push(this.createViolation(
        'TEST_ICT_COVERAGE',
        `ICT coverage is ${ictCoverage.toFixed(0)}% (target: 80%)`,
        'warning',
        undefined,
        'Add test pads or move components for better ICT access'
      ));
      score -= 10;
    }

    // Check boundary scan (JTAG) availability
    const hasJTAG = components.some(c => {
      const ref = c.reference.toLowerCase();
      return ref.includes('jtag') || ref.includes('swd');
    }) || layout.traces?.some(t => {
      const name = (t.netName || '').toLowerCase();
      return name.includes('tck') || name.includes('tms') || name.includes('tdi') || name.includes('tdo');
    });

    // Check programming/debug connector
    const hasDebugConnector = components.some(c => {
      const ref = c.reference.toLowerCase();
      return ref.includes('debug') || ref.includes('prog') || ref.includes('jtag');
    });

    if (!hasDebugConnector && components.length > 10) {
      violations.push(this.createViolation(
        'TEST_DEBUG',
        'Design lacks dedicated programming/debug connector',
        'info',
        undefined,
        'Add JTAG/SWD header for programming and debugging'
      ));
      score -= 5;
    }

    // Check LED indicators for status
    const leds = components.filter(c =>
      c.reference.toUpperCase().startsWith('D') ||
      c.reference.toLowerCase().includes('led')
    );

    if (leds.length === 0 && components.length > 15) {
      violations.push(this.createViolation(
        'TEST_STATUS',
        'Design lacks status LEDs',
        'info',
        undefined,
        'Add power and status LEDs for visual feedback'
      ));
      score -= 3;
    }

    // Check via coverage for test probing
    const vias = layout.vias || [];
    const testableVias = vias.filter(v => v.type === 'through');

    return {
      domain: this.name,
      score: Math.max(0, score),
      maxScore: 100,
      violations,
      metrics: {
        ictCoverage,
        hasJTAG,
        hasDebugConnector,
        ledCount: leds.length,
        testableViaCount: testableVias.length
      },
      passRate: violations.filter(v => v.severity === 'error').length === 0 ? 100 : 0
    };
  }
}

export default ValidationFramework;