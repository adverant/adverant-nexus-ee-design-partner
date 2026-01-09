/**
 * EE Design Partner - Project Type Configuration Registry
 *
 * Complete configurations for each ProjectType defining phase pipelines,
 * simulation requirements, validation rules, and layout strategies.
 */

import type { ProjectPhase, SimulationType, ValidationType, LayoutStrategy } from '../types/index.js';
import {
  ProjectType,
  ProjectComplexity,
  type ProjectTypeConfiguration,
  type PhaseConfig,
  type SimulationConfig,
  type ValidationConfig,
  type LayoutAgentConfig,
} from '../types/project-types.js';

// ============================================================================
// Common Phase Definitions
// ============================================================================

const createPhaseConfig = (
  phase: ProjectPhase,
  displayName: string,
  required: boolean,
  enabledByDefault: boolean,
  estimatedDuration: number,
  dependencies: ProjectPhase[],
  availableSkills: string[],
  validationDomains: ValidationType[],
  deliverables: string[]
): PhaseConfig => ({
  phase,
  displayName,
  required,
  enabledByDefault,
  estimatedDuration,
  dependencies,
  availableSkills,
  validationDomains,
  deliverables,
});

// Standard phase definitions used across project types
const PHASE_IDEATION: PhaseConfig = createPhaseConfig(
  'ideation',
  'Ideation & Research',
  true,
  true,
  4,
  [],
  ['/research-paper', '/patent-search', '/requirements-gen', '/market-analysis'],
  [],
  ['requirements.md', 'market-research.md', 'feasibility-report.md']
);

const PHASE_ARCHITECTURE: PhaseConfig = createPhaseConfig(
  'architecture',
  'System Architecture',
  true,
  true,
  8,
  ['ideation'],
  ['/ee-architecture', '/component-select', '/bom-optimize', '/block-diagram'],
  [],
  ['architecture.md', 'block-diagram.svg', 'preliminary-bom.csv']
);

const PHASE_SCHEMATIC: PhaseConfig = createPhaseConfig(
  'schematic',
  'Schematic Capture',
  true,
  true,
  16,
  ['architecture'],
  ['/schematic-gen', '/schematic-review', '/netlist-gen', '/erc-check'],
  ['erc'],
  ['schematic.kicad_sch', 'netlist.net', 'schematic-review.md']
);

const PHASE_SIMULATION: PhaseConfig = createPhaseConfig(
  'simulation',
  'Circuit Simulation',
  false,
  true,
  12,
  ['schematic'],
  ['/simulate-spice', '/simulate-thermal', '/simulate-si', '/simulate-rf'],
  [],
  ['simulation-results/', 'simulation-report.md']
);

const PHASE_PCB_LAYOUT: PhaseConfig = createPhaseConfig(
  'pcb_layout',
  'PCB Layout',
  true,
  true,
  24,
  ['schematic'],
  ['/pcb-layout', '/mapos', '/stackup-design', '/impedance-calc'],
  ['drc', 'ipc_2221', 'dfm'],
  ['board.kicad_pcb', 'stackup.md', 'layout-report.md']
);

const PHASE_MANUFACTURING: PhaseConfig = createPhaseConfig(
  'manufacturing',
  'Manufacturing Output',
  true,
  true,
  4,
  ['pcb_layout'],
  ['/gerber-gen', '/dfm-check', '/vendor-quote', '/bom-export'],
  ['dfm'],
  ['gerbers/', 'drill-files/', 'bom-final.csv', 'pick-place.csv']
);

const PHASE_FIRMWARE: PhaseConfig = createPhaseConfig(
  'firmware',
  'Firmware Development',
  false,
  true,
  40,
  ['schematic'],
  ['/firmware-gen', '/hal-gen', '/driver-gen', '/rtos-config'],
  [],
  ['firmware/', 'hal/', 'drivers/', 'firmware-docs.md']
);

const PHASE_TESTING: PhaseConfig = createPhaseConfig(
  'testing',
  'Testing & Validation',
  true,
  true,
  16,
  ['manufacturing'],
  ['/test-gen', '/hil-setup', '/test-procedure', '/test-report'],
  ['automated_testing'],
  ['test-procedures/', 'test-results/', 'test-report.md']
);

const PHASE_PRODUCTION: PhaseConfig = createPhaseConfig(
  'production',
  'Production Preparation',
  false,
  false,
  8,
  ['testing'],
  ['/manufacture', '/assembly-guide', '/quality-check', '/production-bom'],
  [],
  ['production-files/', 'assembly-guide.md', 'qc-checklist.md']
);

const PHASE_FIELD_SUPPORT: PhaseConfig = createPhaseConfig(
  'field_support',
  'Field Support',
  false,
  false,
  4,
  ['production'],
  ['/debug-assist', '/service-manual', '/firmware-update', '/rma-process'],
  [],
  ['service-manual.md', 'troubleshooting-guide.md', 'firmware-releases/']
);

// ============================================================================
// Common Simulation Definitions
// ============================================================================

const createSimulationConfig = (
  type: SimulationType,
  displayName: string,
  required: boolean,
  enabledByDefault: boolean,
  priority: number,
  estimatedDuration: number,
  dependencies: SimulationType[],
  minScore: number,
  criticalMetrics: Array<{ name: string; unit: string; min?: number; max?: number }>,
  defaultParameters: Record<string, unknown>
): SimulationConfig => ({
  type,
  displayName,
  required,
  enabledByDefault,
  priority,
  estimatedDuration,
  dependencies,
  criteria: {
    minScore,
    criticalMetrics,
    warningThresholds: {},
  },
  defaultParameters,
});

const SIM_SPICE_DC: SimulationConfig = createSimulationConfig(
  'spice_dc',
  'DC Operating Point',
  false,
  true,
  1,
  5,
  [],
  80,
  [{ name: 'convergence', unit: 'boolean', min: 1 }],
  { maxIterations: 1000, relTol: 0.001 }
);

const SIM_SPICE_AC: SimulationConfig = createSimulationConfig(
  'spice_ac',
  'AC Frequency Analysis',
  false,
  true,
  2,
  10,
  ['spice_dc'],
  80,
  [{ name: 'bandwidth', unit: 'Hz' }],
  { startFreq: 1, endFreq: 1e9, pointsPerDecade: 100 }
);

const SIM_SPICE_TRANSIENT: SimulationConfig = createSimulationConfig(
  'spice_transient',
  'Transient Analysis',
  false,
  true,
  3,
  30,
  ['spice_dc'],
  80,
  [{ name: 'settlingTime', unit: 's' }],
  { stepTime: 1e-9, endTime: 1e-3 }
);

const SIM_SPICE_MONTE_CARLO: SimulationConfig = createSimulationConfig(
  'spice_monte_carlo',
  'Monte Carlo Analysis',
  false,
  false,
  4,
  60,
  ['spice_transient'],
  75,
  [{ name: 'yieldEstimate', unit: '%', min: 95 }],
  { runs: 1000, seed: 42 }
);

const SIM_THERMAL_STEADY: SimulationConfig = createSimulationConfig(
  'thermal_steady_state',
  'Thermal Steady State',
  false,
  true,
  5,
  20,
  [],
  85,
  [{ name: 'maxJunctionTemp', unit: 'C', max: 125 }],
  { ambientTemp: 25, airflow: 0 }
);

const SIM_THERMAL_TRANSIENT: SimulationConfig = createSimulationConfig(
  'thermal_transient',
  'Thermal Transient',
  false,
  false,
  6,
  45,
  ['thermal_steady_state'],
  85,
  [{ name: 'thermalTimeConstant', unit: 's' }],
  { pulseDuration: 1, dutyCycle: 0.5 }
);

const SIM_SIGNAL_INTEGRITY: SimulationConfig = createSimulationConfig(
  'signal_integrity',
  'Signal Integrity',
  false,
  true,
  7,
  30,
  [],
  85,
  [
    { name: 'eyeHeight', unit: 'V', min: 0.3 },
    { name: 'eyeWidth', unit: 's', min: 0.7 },
  ],
  { dataRate: 1e9, traceImpedance: 50 }
);

const SIM_POWER_INTEGRITY: SimulationConfig = createSimulationConfig(
  'power_integrity',
  'Power Integrity',
  false,
  true,
  8,
  25,
  [],
  85,
  [
    { name: 'pdnImpedance', unit: 'Ohm', max: 0.1 },
    { name: 'ripple', unit: 'mV', max: 50 },
  ],
  { frequency: 1e6, targetImpedance: 0.05 }
);

const SIM_RF_SPARAMETERS: SimulationConfig = createSimulationConfig(
  'rf_sparameters',
  'S-Parameter Analysis',
  false,
  false,
  9,
  40,
  [],
  80,
  [
    { name: 'returnLoss', unit: 'dB', max: -10 },
    { name: 'insertionLoss', unit: 'dB', max: -3 },
  ],
  { startFreq: 1e6, endFreq: 6e9, points: 1001 }
);

const SIM_EMC_RADIATED: SimulationConfig = createSimulationConfig(
  'emc_radiated',
  'Radiated Emissions',
  false,
  false,
  10,
  60,
  [],
  80,
  [{ name: 'maxEmission', unit: 'dBuV/m', max: 40 }],
  { frequency: 30e6, distance: 3 }
);

// ============================================================================
// Common Validation Definitions
// ============================================================================

const createValidationConfig = (
  type: ValidationType,
  displayName: string,
  required: boolean,
  enabledByDefault: boolean,
  weight: number,
  failureSeverity: 'blocking' | 'warning' | 'info'
): ValidationConfig => ({
  type,
  displayName,
  required,
  enabledByDefault,
  weight,
  failureSeverity,
  rules: [],
});

const VAL_DRC: ValidationConfig = createValidationConfig(
  'drc',
  'Design Rule Check',
  true,
  true,
  0.25,
  'blocking'
);

const VAL_ERC: ValidationConfig = createValidationConfig(
  'erc',
  'Electrical Rule Check',
  true,
  true,
  0.2,
  'blocking'
);

const VAL_IPC2221: ValidationConfig = createValidationConfig(
  'ipc_2221',
  'IPC-2221 Compliance',
  true,
  true,
  0.15,
  'blocking'
);

const VAL_SIGNAL_INTEGRITY: ValidationConfig = createValidationConfig(
  'signal_integrity',
  'Signal Integrity Validation',
  false,
  true,
  0.1,
  'warning'
);

const VAL_THERMAL: ValidationConfig = createValidationConfig(
  'thermal',
  'Thermal Validation',
  false,
  true,
  0.1,
  'warning'
);

const VAL_DFM: ValidationConfig = createValidationConfig(
  'dfm',
  'Design for Manufacturing',
  true,
  true,
  0.1,
  'warning'
);

const VAL_BEST_PRACTICES: ValidationConfig = createValidationConfig(
  'best_practices',
  'Best Practices Review',
  false,
  true,
  0.05,
  'info'
);

const VAL_TESTING: ValidationConfig = createValidationConfig(
  'automated_testing',
  'Automated Testing',
  false,
  true,
  0.05,
  'info'
);

// ============================================================================
// Common Layout Agent Definitions
// ============================================================================

const createLayoutAgentConfig = (
  strategy: LayoutStrategy,
  displayName: string,
  description: string,
  enabled: boolean,
  priorityWeight: number,
  densityTarget: number,
  thermalMargin: number,
  emiClearanceMultiplier: number,
  dfmStrictness: number
): LayoutAgentConfig => ({
  strategy,
  displayName,
  description,
  enabled,
  priorityWeight,
  parameters: {
    densityTarget,
    thermalMargin,
    emiClearanceMultiplier,
    dfmStrictness,
    groupingAggressiveness: 0.5,
    viaMinimization: 0.5,
    traceLengthOptimization: 0.5,
  },
});

const AGENT_CONSERVATIVE: LayoutAgentConfig = createLayoutAgentConfig(
  'conservative',
  'Conservative Layout',
  'Prioritizes reliability and manufacturability with generous spacing',
  true,
  0.8,
  0.4,
  1.5,
  1.5,
  0.9
);

const AGENT_AGGRESSIVE_COMPACT: LayoutAgentConfig = createLayoutAgentConfig(
  'aggressive_compact',
  'Aggressive Compact',
  'Maximizes component density for space-constrained designs',
  true,
  0.6,
  0.85,
  1.0,
  1.0,
  0.6
);

const AGENT_THERMAL_OPTIMIZED: LayoutAgentConfig = createLayoutAgentConfig(
  'thermal_optimized',
  'Thermal Optimized',
  'Prioritizes thermal performance with strategic heat spreading',
  true,
  0.9,
  0.5,
  2.0,
  1.2,
  0.8
);

const AGENT_EMI_OPTIMIZED: LayoutAgentConfig = createLayoutAgentConfig(
  'emi_optimized',
  'EMI Optimized',
  'Minimizes electromagnetic interference with careful routing',
  true,
  0.7,
  0.5,
  1.2,
  2.0,
  0.8
);

const AGENT_DFM_OPTIMIZED: LayoutAgentConfig = createLayoutAgentConfig(
  'dfm_optimized',
  'DFM Optimized',
  'Prioritizes ease of manufacturing and assembly yield',
  true,
  0.75,
  0.5,
  1.3,
  1.3,
  1.0
);

// ============================================================================
// POWER_ELECTRONICS Configuration
// ============================================================================

const POWER_ELECTRONICS_CONFIG: ProjectTypeConfiguration = {
  type: ProjectType.POWER_ELECTRONICS,
  displayName: 'Power Electronics',
  description: 'High-power designs including motor drivers, power supplies, battery management systems, and power converters. Requires thermal analysis and conservative layout.',
  typicalComplexity: [ProjectComplexity.HIGH, ProjectComplexity.EXTREME],

  phases: [
    PHASE_IDEATION,
    PHASE_ARCHITECTURE,
    PHASE_SCHEMATIC,
    {
      ...PHASE_SIMULATION,
      required: true,
      options: { multiLlmValidation: true },
    },
    {
      ...PHASE_PCB_LAYOUT,
      estimatedDuration: 40,
      options: { maxIterations: 100, targetScore: 95, multiLlmValidation: true },
    },
    PHASE_MANUFACTURING,
    {
      ...PHASE_FIRMWARE,
      required: true,
      enabledByDefault: true,
    },
    {
      ...PHASE_TESTING,
      estimatedDuration: 24,
    },
    PHASE_PRODUCTION,
    PHASE_FIELD_SUPPORT,
  ],

  simulations: [
    SIM_SPICE_DC,
    SIM_SPICE_TRANSIENT,
    {
      ...SIM_THERMAL_STEADY,
      required: true,
      criteria: {
        minScore: 90,
        criticalMetrics: [
          { name: 'maxJunctionTemp', unit: 'C', max: 125 },
          { name: 'maxCaseTemp', unit: 'C', max: 100 },
        ],
        warningThresholds: { maxJunctionTemp: 100, maxCaseTemp: 85 },
      },
    },
    {
      ...SIM_THERMAL_TRANSIENT,
      enabledByDefault: true,
    },
    SIM_POWER_INTEGRITY,
    SIM_SPICE_MONTE_CARLO,
    SIM_EMC_RADIATED,
  ],

  validations: [
    VAL_DRC,
    VAL_ERC,
    VAL_IPC2221,
    {
      ...VAL_THERMAL,
      required: true,
      weight: 0.2,
      failureSeverity: 'blocking',
    },
    VAL_DFM,
    VAL_BEST_PRACTICES,
    VAL_TESTING,
  ],

  layoutAgents: [
    {
      ...AGENT_CONSERVATIVE,
      priorityWeight: 0.95,
    },
    {
      ...AGENT_THERMAL_OPTIMIZED,
      priorityWeight: 1.0,
    },
    {
      ...AGENT_DFM_OPTIMIZED,
      priorityWeight: 0.85,
    },
    {
      ...AGENT_AGGRESSIVE_COMPACT,
      enabled: false,
    },
    AGENT_EMI_OPTIMIZED,
  ],

  subsystemTemplates: [
    'power-mosfet-halfbridge',
    'power-mosfet-fullbridge',
    'gate-driver-isolated',
    'gate-driver-bootstrap',
    'current-sense-shunt',
    'current-sense-hall',
    'dc-dc-buck',
    'dc-dc-boost',
    'dc-dc-buck-boost',
    'input-protection-tvs',
    'soft-start-circuit',
    'thermal-protection',
    'mcu-stm32h7',
    'mcu-aurix',
  ],

  defaultBoardConstraints: {
    minLayers: 4,
    maxLayers: 16,
    defaultLayers: 6,
    minTraceWidth: 0.2,
    minClearance: 0.25,
    minViaDiameter: 0.6,
    minViaDrill: 0.3,
    defaultCopperWeight: 2,
    impedanceControlDefault: false,
  },

  manufacturingRecommendations: {
    recommendedMaterial: 'aluminum',
    recommendedFinish: 'enig',
    viaFillingRecommended: true,
    assemblyType: 'mixed',
    qualityLevel: 'automotive',
  },

  options: {
    firmwareApplicable: true,
    thermalCritical: true,
    signalIntegrityCritical: false,
    rfApplicable: false,
    emcTestingRequired: true,
    safetyCertifications: ['UL', 'CE', 'IEC 61800'],
    targetApplications: ['Motor Control', 'Power Supply', 'Battery Management', 'Solar Inverter'],
  },
};

// ============================================================================
// ANALOG_CIRCUIT Configuration
// ============================================================================

const ANALOG_CIRCUIT_CONFIG: ProjectTypeConfiguration = {
  type: ProjectType.ANALOG_CIRCUIT,
  displayName: 'Analog Circuit',
  description: 'Pure analog designs including amplifiers, filters, sensors, and precision measurement circuits. Requires extensive SPICE simulation.',
  typicalComplexity: [ProjectComplexity.MEDIUM, ProjectComplexity.HIGH],

  phases: [
    PHASE_IDEATION,
    PHASE_ARCHITECTURE,
    PHASE_SCHEMATIC,
    {
      ...PHASE_SIMULATION,
      required: true,
      estimatedDuration: 20,
    },
    PHASE_PCB_LAYOUT,
    PHASE_MANUFACTURING,
    // No firmware phase for pure analog
    PHASE_TESTING,
    PHASE_PRODUCTION,
    PHASE_FIELD_SUPPORT,
  ],

  simulations: [
    {
      ...SIM_SPICE_DC,
      required: true,
    },
    {
      ...SIM_SPICE_AC,
      required: true,
    },
    {
      ...SIM_SPICE_TRANSIENT,
      required: true,
    },
    {
      ...SIM_SPICE_MONTE_CARLO,
      enabledByDefault: true,
    },
    {
      type: 'spice_noise',
      displayName: 'Noise Analysis',
      required: true,
      enabledByDefault: true,
      priority: 5,
      estimatedDuration: 15,
      dependencies: ['spice_ac'],
      criteria: {
        minScore: 85,
        criticalMetrics: [
          { name: 'inputReferredNoise', unit: 'nV/rtHz', max: 10 },
          { name: 'snr', unit: 'dB', min: 80 },
        ],
        warningThresholds: { inputReferredNoise: 5, snr: 90 },
      },
      defaultParameters: { startFreq: 1, endFreq: 1e6 },
    },
    // Thermal not typically required for low-power analog
  ],

  validations: [
    VAL_DRC,
    VAL_ERC,
    VAL_IPC2221,
    VAL_DFM,
    VAL_BEST_PRACTICES,
    VAL_TESTING,
  ],

  layoutAgents: [
    {
      ...AGENT_EMI_OPTIMIZED,
      priorityWeight: 1.0,
      description: 'Minimizes noise coupling in sensitive analog circuits',
    },
    {
      ...AGENT_CONSERVATIVE,
      priorityWeight: 0.9,
    },
    AGENT_DFM_OPTIMIZED,
    {
      ...AGENT_AGGRESSIVE_COMPACT,
      enabled: false,
    },
    {
      ...AGENT_THERMAL_OPTIMIZED,
      enabled: false,
    },
  ],

  subsystemTemplates: [
    'opamp-inverting',
    'opamp-noninverting',
    'opamp-differential',
    'instrumentation-amp',
    'active-filter-lowpass',
    'active-filter-highpass',
    'active-filter-bandpass',
    'voltage-reference-precision',
    'adc-sar',
    'adc-sigma-delta',
    'dac-precision',
    'power-supply-ldo',
    'power-supply-dual',
  ],

  defaultBoardConstraints: {
    minLayers: 2,
    maxLayers: 6,
    defaultLayers: 4,
    minTraceWidth: 0.15,
    minClearance: 0.15,
    minViaDiameter: 0.5,
    minViaDrill: 0.25,
    defaultCopperWeight: 1,
    impedanceControlDefault: false,
  },

  manufacturingRecommendations: {
    recommendedMaterial: 'fr4',
    recommendedFinish: 'enig',
    viaFillingRecommended: false,
    assemblyType: 'smt_only',
    qualityLevel: 'production',
  },

  options: {
    firmwareApplicable: false,
    thermalCritical: false,
    signalIntegrityCritical: false,
    rfApplicable: false,
    emcTestingRequired: false,
    safetyCertifications: [],
    targetApplications: ['Audio Equipment', 'Instrumentation', 'Sensors', 'Test Equipment'],
  },
};

// ============================================================================
// DIGITAL_LOGIC Configuration
// ============================================================================

const DIGITAL_LOGIC_CONFIG: ProjectTypeConfiguration = {
  type: ProjectType.DIGITAL_LOGIC,
  displayName: 'Digital Logic',
  description: 'Digital-only designs including FPGAs, microcontroller boards, and digital interfaces. Requires signal integrity analysis.',
  typicalComplexity: [ProjectComplexity.MEDIUM, ProjectComplexity.HIGH],

  phases: [
    PHASE_IDEATION,
    PHASE_ARCHITECTURE,
    PHASE_SCHEMATIC,
    {
      ...PHASE_SIMULATION,
      availableSkills: ['/simulate-si', '/simulate-spice'],
    },
    {
      ...PHASE_PCB_LAYOUT,
      options: { maxIterations: 100, targetScore: 90 },
    },
    PHASE_MANUFACTURING,
    {
      ...PHASE_FIRMWARE,
      required: true,
      estimatedDuration: 60,
    },
    PHASE_TESTING,
    PHASE_PRODUCTION,
    PHASE_FIELD_SUPPORT,
  ],

  simulations: [
    SIM_SPICE_DC,
    {
      ...SIM_SIGNAL_INTEGRITY,
      required: true,
      criteria: {
        minScore: 90,
        criticalMetrics: [
          { name: 'eyeHeight', unit: 'V', min: 0.4 },
          { name: 'eyeWidth', unit: 'UI', min: 0.7 },
          { name: 'jitter', unit: 'ps', max: 100 },
        ],
        warningThresholds: { eyeHeight: 0.5, eyeWidth: 0.8, jitter: 50 },
      },
    },
    {
      ...SIM_POWER_INTEGRITY,
      required: true,
    },
    SIM_THERMAL_STEADY,
  ],

  validations: [
    VAL_DRC,
    VAL_ERC,
    VAL_IPC2221,
    {
      ...VAL_SIGNAL_INTEGRITY,
      required: true,
      weight: 0.2,
      failureSeverity: 'blocking',
    },
    VAL_DFM,
    VAL_BEST_PRACTICES,
    VAL_TESTING,
  ],

  layoutAgents: [
    AGENT_CONSERVATIVE,
    {
      ...AGENT_EMI_OPTIMIZED,
      priorityWeight: 0.9,
    },
    AGENT_DFM_OPTIMIZED,
    AGENT_AGGRESSIVE_COMPACT,
    AGENT_THERMAL_OPTIMIZED,
  ],

  subsystemTemplates: [
    'mcu-stm32h7',
    'mcu-esp32',
    'fpga-artix7',
    'fpga-ice40',
    'memory-sdram',
    'memory-flash-qspi',
    'usb-type-c',
    'ethernet-phy',
    'can-transceiver',
    'power-supply-switching',
    'clock-oscillator',
    'clock-pll',
    'debug-jtag',
    'debug-swd',
  ],

  defaultBoardConstraints: {
    minLayers: 4,
    maxLayers: 12,
    defaultLayers: 6,
    minTraceWidth: 0.1,
    minClearance: 0.1,
    minViaDiameter: 0.4,
    minViaDrill: 0.2,
    defaultCopperWeight: 1,
    impedanceControlDefault: true,
  },

  manufacturingRecommendations: {
    recommendedMaterial: 'fr4',
    recommendedFinish: 'enig',
    viaFillingRecommended: false,
    assemblyType: 'smt_only',
    qualityLevel: 'production',
  },

  options: {
    firmwareApplicable: true,
    thermalCritical: false,
    signalIntegrityCritical: true,
    rfApplicable: false,
    emcTestingRequired: true,
    safetyCertifications: ['CE', 'FCC'],
    targetApplications: ['Embedded Systems', 'FPGA Development', 'MCU Boards', 'Digital Interfaces'],
  },
};

// ============================================================================
// MIXED_SIGNAL Configuration
// ============================================================================

const MIXED_SIGNAL_CONFIG: ProjectTypeConfiguration = {
  type: ProjectType.MIXED_SIGNAL,
  displayName: 'Mixed Signal',
  description: 'Combined analog and digital designs including data acquisition, audio processing, and mixed-signal ASICs. Requires both SPICE and SI analysis.',
  typicalComplexity: [ProjectComplexity.HIGH, ProjectComplexity.EXTREME],

  phases: [
    PHASE_IDEATION,
    PHASE_ARCHITECTURE,
    PHASE_SCHEMATIC,
    {
      ...PHASE_SIMULATION,
      required: true,
      estimatedDuration: 24,
    },
    {
      ...PHASE_PCB_LAYOUT,
      estimatedDuration: 32,
      options: { maxIterations: 100, targetScore: 95, multiLlmValidation: true },
    },
    PHASE_MANUFACTURING,
    {
      ...PHASE_FIRMWARE,
      required: true,
    },
    {
      ...PHASE_TESTING,
      estimatedDuration: 20,
    },
    PHASE_PRODUCTION,
    PHASE_FIELD_SUPPORT,
  ],

  simulations: [
    {
      ...SIM_SPICE_DC,
      required: true,
    },
    {
      ...SIM_SPICE_AC,
      required: true,
    },
    SIM_SPICE_TRANSIENT,
    {
      type: 'spice_noise',
      displayName: 'Noise Analysis',
      required: true,
      enabledByDefault: true,
      priority: 5,
      estimatedDuration: 15,
      dependencies: ['spice_ac'],
      criteria: {
        minScore: 85,
        criticalMetrics: [{ name: 'enob', unit: 'bits', min: 12 }],
        warningThresholds: { enob: 14 },
      },
      defaultParameters: { startFreq: 1, endFreq: 1e6 },
    },
    {
      ...SIM_SIGNAL_INTEGRITY,
      required: true,
    },
    SIM_POWER_INTEGRITY,
    SIM_THERMAL_STEADY,
    SIM_SPICE_MONTE_CARLO,
  ],

  validations: [
    VAL_DRC,
    VAL_ERC,
    VAL_IPC2221,
    {
      ...VAL_SIGNAL_INTEGRITY,
      required: true,
      weight: 0.15,
    },
    VAL_THERMAL,
    VAL_DFM,
    VAL_BEST_PRACTICES,
    VAL_TESTING,
  ],

  layoutAgents: [
    {
      ...AGENT_EMI_OPTIMIZED,
      priorityWeight: 1.0,
    },
    {
      ...AGENT_CONSERVATIVE,
      priorityWeight: 0.95,
    },
    AGENT_DFM_OPTIMIZED,
    AGENT_THERMAL_OPTIMIZED,
    {
      ...AGENT_AGGRESSIVE_COMPACT,
      enabled: false,
    },
  ],

  subsystemTemplates: [
    'adc-high-speed',
    'adc-precision',
    'dac-high-speed',
    'dac-precision',
    'analog-frontend',
    'antialiasing-filter',
    'digital-isolator',
    'mcu-stm32h7',
    'fpga-artix7',
    'clock-low-jitter',
    'power-supply-split',
    'power-supply-ldo-precision',
    'ground-star',
  ],

  defaultBoardConstraints: {
    minLayers: 4,
    maxLayers: 12,
    defaultLayers: 6,
    minTraceWidth: 0.1,
    minClearance: 0.15,
    minViaDiameter: 0.45,
    minViaDrill: 0.2,
    defaultCopperWeight: 1,
    impedanceControlDefault: true,
  },

  manufacturingRecommendations: {
    recommendedMaterial: 'fr4',
    recommendedFinish: 'enig',
    viaFillingRecommended: false,
    assemblyType: 'smt_only',
    qualityLevel: 'production',
  },

  options: {
    firmwareApplicable: true,
    thermalCritical: false,
    signalIntegrityCritical: true,
    rfApplicable: false,
    emcTestingRequired: true,
    safetyCertifications: ['CE', 'FCC'],
    targetApplications: ['Data Acquisition', 'Audio Processing', 'Measurement Systems', 'Signal Processing'],
  },
};

// ============================================================================
// RF_DESIGN Configuration
// ============================================================================

const RF_DESIGN_CONFIG: ProjectTypeConfiguration = {
  type: ProjectType.RF_DESIGN,
  displayName: 'RF Design',
  description: 'Radio frequency designs including wireless modules, antennas, RF front-ends, and radar systems. Requires RF and EMC simulation.',
  typicalComplexity: [ProjectComplexity.HIGH, ProjectComplexity.EXTREME],

  phases: [
    PHASE_IDEATION,
    {
      ...PHASE_ARCHITECTURE,
      estimatedDuration: 12,
      availableSkills: ['/ee-architecture', '/rf-link-budget', '/antenna-design', '/component-select'],
    },
    PHASE_SCHEMATIC,
    {
      ...PHASE_SIMULATION,
      required: true,
      estimatedDuration: 30,
      availableSkills: ['/simulate-rf', '/simulate-emc', '/simulate-si'],
    },
    {
      ...PHASE_PCB_LAYOUT,
      estimatedDuration: 40,
      options: { maxIterations: 150, targetScore: 95, multiLlmValidation: true },
    },
    PHASE_MANUFACTURING,
    {
      ...PHASE_FIRMWARE,
      enabledByDefault: true,
    },
    {
      ...PHASE_TESTING,
      estimatedDuration: 24,
      availableSkills: ['/test-gen', '/rf-test-setup', '/antenna-test', '/test-report'],
    },
    PHASE_PRODUCTION,
    PHASE_FIELD_SUPPORT,
  ],

  simulations: [
    SIM_SPICE_DC,
    SIM_SPICE_AC,
    {
      ...SIM_RF_SPARAMETERS,
      required: true,
      enabledByDefault: true,
      criteria: {
        minScore: 90,
        criticalMetrics: [
          { name: 'returnLoss', unit: 'dB', max: -15 },
          { name: 'insertionLoss', unit: 'dB', max: -1 },
          { name: 'vswr', unit: 'ratio', max: 1.5 },
        ],
        warningThresholds: { returnLoss: -20, insertionLoss: -0.5 },
      },
    },
    {
      type: 'rf_field_pattern',
      displayName: 'Antenna Field Pattern',
      required: false,
      enabledByDefault: true,
      priority: 10,
      estimatedDuration: 60,
      dependencies: [],
      criteria: {
        minScore: 85,
        criticalMetrics: [
          { name: 'gain', unit: 'dBi', min: 0 },
          { name: 'efficiency', unit: '%', min: 70 },
        ],
        warningThresholds: { gain: 2, efficiency: 80 },
      },
      defaultParameters: { frequency: 2.4e9, meshSize: 0.1 },
    },
    {
      ...SIM_EMC_RADIATED,
      required: true,
      enabledByDefault: true,
    },
    SIM_SIGNAL_INTEGRITY,
    SIM_THERMAL_STEADY,
  ],

  validations: [
    VAL_DRC,
    VAL_ERC,
    VAL_IPC2221,
    {
      ...VAL_SIGNAL_INTEGRITY,
      required: true,
      weight: 0.2,
      failureSeverity: 'blocking',
    },
    VAL_DFM,
    VAL_BEST_PRACTICES,
    VAL_TESTING,
  ],

  layoutAgents: [
    {
      ...AGENT_EMI_OPTIMIZED,
      priorityWeight: 1.0,
      parameters: {
        densityTarget: 0.5,
        thermalMargin: 1.2,
        emiClearanceMultiplier: 2.5,
        dfmStrictness: 0.8,
        groupingAggressiveness: 0.8,
        viaMinimization: 0.9,
        traceLengthOptimization: 0.9,
      },
    },
    AGENT_CONSERVATIVE,
    AGENT_DFM_OPTIMIZED,
    {
      ...AGENT_AGGRESSIVE_COMPACT,
      enabled: false,
    },
    AGENT_THERMAL_OPTIMIZED,
  ],

  subsystemTemplates: [
    'rf-lna',
    'rf-pa',
    'rf-mixer',
    'rf-filter-saw',
    'rf-filter-ceramic',
    'antenna-chip',
    'antenna-pcb-trace',
    'balun',
    'rf-switch',
    'pll-synthesizer',
    'clock-tcxo',
    'power-supply-ldo-quiet',
    'esd-protection-rf',
  ],

  defaultBoardConstraints: {
    minLayers: 4,
    maxLayers: 10,
    defaultLayers: 4,
    minTraceWidth: 0.1,
    minClearance: 0.15,
    minViaDiameter: 0.4,
    minViaDrill: 0.2,
    defaultCopperWeight: 1,
    impedanceControlDefault: true,
  },

  manufacturingRecommendations: {
    recommendedMaterial: 'rogers',
    recommendedFinish: 'enig',
    viaFillingRecommended: false,
    assemblyType: 'smt_only',
    qualityLevel: 'production',
  },

  options: {
    firmwareApplicable: true,
    thermalCritical: false,
    signalIntegrityCritical: true,
    rfApplicable: true,
    emcTestingRequired: true,
    safetyCertifications: ['CE', 'FCC', 'IC', 'MIC'],
    targetApplications: ['Wireless Modules', 'Radar Systems', 'RF Front-ends', 'Antenna Design'],
  },
};

// ============================================================================
// IOT_DEVICE Configuration
// ============================================================================

const IOT_DEVICE_CONFIG: ProjectTypeConfiguration = {
  type: ProjectType.IOT_DEVICE,
  displayName: 'IoT Device',
  description: 'Connected devices including sensors with wireless, edge computing platforms, and wearables. Compact layout with power optimization.',
  typicalComplexity: [ProjectComplexity.MEDIUM, ProjectComplexity.HIGH],

  phases: [
    PHASE_IDEATION,
    PHASE_ARCHITECTURE,
    PHASE_SCHEMATIC,
    {
      ...PHASE_SIMULATION,
      enabledByDefault: true,
    },
    {
      ...PHASE_PCB_LAYOUT,
      options: { maxIterations: 80, targetScore: 90 },
    },
    PHASE_MANUFACTURING,
    {
      ...PHASE_FIRMWARE,
      required: true,
      estimatedDuration: 48,
      availableSkills: ['/firmware-gen', '/hal-gen', '/driver-gen', '/ble-stack', '/wifi-stack', '/power-management'],
    },
    PHASE_TESTING,
    PHASE_PRODUCTION,
    PHASE_FIELD_SUPPORT,
  ],

  simulations: [
    SIM_SPICE_DC,
    SIM_SPICE_TRANSIENT,
    {
      ...SIM_RF_SPARAMETERS,
      enabledByDefault: true,
    },
    SIM_SIGNAL_INTEGRITY,
    {
      ...SIM_THERMAL_STEADY,
      enabledByDefault: true,
    },
    {
      type: 'reliability_mtbf',
      displayName: 'MTBF Analysis',
      required: false,
      enabledByDefault: true,
      priority: 12,
      estimatedDuration: 20,
      dependencies: [],
      criteria: {
        minScore: 80,
        criticalMetrics: [{ name: 'mtbf', unit: 'hours', min: 50000 }],
        warningThresholds: { mtbf: 100000 },
      },
      defaultParameters: { ambientTemp: 40, dutyCycle: 0.5 },
    },
  ],

  validations: [
    VAL_DRC,
    VAL_ERC,
    VAL_IPC2221,
    VAL_SIGNAL_INTEGRITY,
    VAL_DFM,
    VAL_BEST_PRACTICES,
    VAL_TESTING,
  ],

  layoutAgents: [
    {
      ...AGENT_AGGRESSIVE_COMPACT,
      priorityWeight: 1.0,
    },
    AGENT_EMI_OPTIMIZED,
    AGENT_DFM_OPTIMIZED,
    AGENT_CONSERVATIVE,
    {
      ...AGENT_THERMAL_OPTIMIZED,
      priorityWeight: 0.6,
    },
  ],

  subsystemTemplates: [
    'mcu-esp32',
    'mcu-nrf52',
    'mcu-stm32l4',
    'wifi-module',
    'ble-module',
    'lora-module',
    'antenna-chip-24ghz',
    'antenna-pcb-24ghz',
    'sensor-temperature',
    'sensor-humidity',
    'sensor-accelerometer',
    'sensor-light',
    'power-battery-charger',
    'power-battery-gauge',
    'power-supply-ultra-low',
    'usb-type-c-pd',
  ],

  defaultBoardConstraints: {
    minLayers: 2,
    maxLayers: 6,
    defaultLayers: 4,
    minTraceWidth: 0.1,
    minClearance: 0.1,
    minViaDiameter: 0.35,
    minViaDrill: 0.15,
    defaultCopperWeight: 1,
    impedanceControlDefault: true,
  },

  manufacturingRecommendations: {
    recommendedMaterial: 'fr4',
    recommendedFinish: 'enig',
    viaFillingRecommended: false,
    assemblyType: 'smt_only',
    qualityLevel: 'production',
  },

  options: {
    firmwareApplicable: true,
    thermalCritical: false,
    signalIntegrityCritical: false,
    rfApplicable: true,
    emcTestingRequired: true,
    safetyCertifications: ['CE', 'FCC', 'Bluetooth SIG'],
    targetApplications: ['Wearables', 'Smart Sensors', 'Edge Computing', 'Connected Devices'],
  },
};

// ============================================================================
// PASSIVE_BOARD Configuration
// ============================================================================

const PASSIVE_BOARD_CONFIG: ProjectTypeConfiguration = {
  type: ProjectType.PASSIVE_BOARD,
  displayName: 'Passive Board',
  description: 'Simple boards including breakouts, adapters, passive networks, and connector boards. Minimal phases - schematic to manufacturing.',
  typicalComplexity: [ProjectComplexity.LOW, ProjectComplexity.MEDIUM],

  phases: [
    {
      ...PHASE_IDEATION,
      estimatedDuration: 1,
      required: false,
    },
    {
      ...PHASE_ARCHITECTURE,
      estimatedDuration: 2,
      required: false,
    },
    {
      ...PHASE_SCHEMATIC,
      estimatedDuration: 4,
    },
    // No simulation phase
    {
      ...PHASE_PCB_LAYOUT,
      estimatedDuration: 8,
      dependencies: ['schematic'],
      options: { maxIterations: 30, targetScore: 85 },
    },
    {
      ...PHASE_MANUFACTURING,
      estimatedDuration: 2,
    },
    // No firmware phase
    {
      ...PHASE_TESTING,
      required: false,
      estimatedDuration: 2,
    },
    // No production or field support for simple boards
  ],

  simulations: [
    // Minimal simulations for passive boards
    {
      ...SIM_SPICE_DC,
      required: false,
      enabledByDefault: false,
    },
  ],

  validations: [
    VAL_DRC,
    VAL_ERC,
    VAL_IPC2221,
    VAL_DFM,
    {
      ...VAL_BEST_PRACTICES,
      weight: 0.1,
    },
  ],

  layoutAgents: [
    AGENT_DFM_OPTIMIZED,
    AGENT_CONSERVATIVE,
    {
      ...AGENT_AGGRESSIVE_COMPACT,
      priorityWeight: 0.7,
    },
    {
      ...AGENT_EMI_OPTIMIZED,
      enabled: false,
    },
    {
      ...AGENT_THERMAL_OPTIMIZED,
      enabled: false,
    },
  ],

  subsystemTemplates: [
    'breakout-qfn',
    'breakout-sop',
    'breakout-bga',
    'adapter-voltage-level',
    'adapter-interface',
    'connector-header',
    'connector-terminal-block',
    'filter-passive-rc',
    'filter-passive-lc',
    'resistor-network',
    'capacitor-bank',
  ],

  defaultBoardConstraints: {
    minLayers: 2,
    maxLayers: 4,
    defaultLayers: 2,
    minTraceWidth: 0.2,
    minClearance: 0.2,
    minViaDiameter: 0.6,
    minViaDrill: 0.3,
    defaultCopperWeight: 1,
    impedanceControlDefault: false,
  },

  manufacturingRecommendations: {
    recommendedMaterial: 'fr4',
    recommendedFinish: 'hasl',
    viaFillingRecommended: false,
    assemblyType: 'through_hole',
    qualityLevel: 'prototype',
  },

  options: {
    firmwareApplicable: false,
    thermalCritical: false,
    signalIntegrityCritical: false,
    rfApplicable: false,
    emcTestingRequired: false,
    safetyCertifications: [],
    targetApplications: ['Breakout Boards', 'Adapters', 'Test Fixtures', 'Passive Networks'],
  },
};

// ============================================================================
// FIRMWARE_ONLY Configuration
// ============================================================================

const FIRMWARE_ONLY_CONFIG: ProjectTypeConfiguration = {
  type: ProjectType.FIRMWARE_ONLY,
  displayName: 'Firmware Only',
  description: 'Software-only projects for existing hardware. Requirements to firmware to testing - no PCB phases.',
  typicalComplexity: [ProjectComplexity.MEDIUM, ProjectComplexity.HIGH],

  phases: [
    PHASE_IDEATION,
    {
      ...PHASE_ARCHITECTURE,
      availableSkills: ['/ee-architecture', '/software-architecture', '/api-design'],
      deliverables: ['software-architecture.md', 'api-spec.yaml', 'data-model.md'],
    },
    // No schematic phase
    // No PCB layout phase
    // No manufacturing phase
    {
      ...PHASE_FIRMWARE,
      required: true,
      estimatedDuration: 80,
      dependencies: ['architecture'],
    },
    {
      ...PHASE_TESTING,
      estimatedDuration: 40,
      dependencies: ['firmware'],
      availableSkills: ['/test-gen', '/unit-test', '/integration-test', '/ci-setup'],
    },
    {
      ...PHASE_PRODUCTION,
      required: false,
      displayName: 'Release Preparation',
      deliverables: ['release-notes.md', 'deployment-guide.md', 'firmware-binary/'],
    },
    {
      ...PHASE_FIELD_SUPPORT,
      availableSkills: ['/debug-assist', '/firmware-update', '/ota-setup'],
    },
  ],

  simulations: [
    // No hardware simulations for firmware-only
  ],

  validations: [
    // Software-focused validations
    {
      type: 'automated_testing',
      displayName: 'Unit Test Coverage',
      required: true,
      enabledByDefault: true,
      weight: 0.3,
      failureSeverity: 'blocking',
      rules: [
        {
          id: 'min-coverage',
          name: 'Minimum Test Coverage',
          description: 'Code must have at least 80% test coverage',
          severity: 'error',
          enabled: true,
          parameters: { minCoverage: 80 },
        },
      ],
    },
    {
      type: 'best_practices',
      displayName: 'Code Quality',
      required: true,
      enabledByDefault: true,
      weight: 0.3,
      failureSeverity: 'warning',
      rules: [
        {
          id: 'static-analysis',
          name: 'Static Analysis',
          description: 'Code must pass static analysis with no critical issues',
          severity: 'warning',
          enabled: true,
        },
      ],
    },
  ],

  layoutAgents: [
    // No layout agents for firmware-only projects
  ],

  subsystemTemplates: [
    // Software component templates
    'firmware-rtos-task',
    'firmware-driver-template',
    'firmware-hal-template',
    'firmware-protocol-stack',
    'firmware-bootloader',
    'firmware-ota-update',
  ],

  defaultBoardConstraints: {
    minLayers: 0,
    maxLayers: 0,
    defaultLayers: 0,
    minTraceWidth: 0,
    minClearance: 0,
    minViaDiameter: 0,
    minViaDrill: 0,
    defaultCopperWeight: 0,
    impedanceControlDefault: false,
  },

  manufacturingRecommendations: {
    recommendedMaterial: 'fr4',
    recommendedFinish: 'hasl',
    viaFillingRecommended: false,
    assemblyType: 'manual',
    qualityLevel: 'prototype',
  },

  options: {
    firmwareApplicable: true,
    thermalCritical: false,
    signalIntegrityCritical: false,
    rfApplicable: false,
    emcTestingRequired: false,
    safetyCertifications: [],
    targetApplications: ['Embedded Software', 'Firmware Updates', 'Driver Development', 'RTOS Applications'],
  },
};

// ============================================================================
// CUSTOM Configuration (Template)
// ============================================================================

const CUSTOM_CONFIG: ProjectTypeConfiguration = {
  type: ProjectType.CUSTOM,
  displayName: 'Custom Project',
  description: 'User-defined configuration for specialized requirements. All phases and simulations available for customization.',
  typicalComplexity: [ProjectComplexity.LOW, ProjectComplexity.MEDIUM, ProjectComplexity.HIGH, ProjectComplexity.EXTREME],

  phases: [
    { ...PHASE_IDEATION, required: false },
    { ...PHASE_ARCHITECTURE, required: false },
    { ...PHASE_SCHEMATIC, required: false },
    { ...PHASE_SIMULATION, required: false },
    { ...PHASE_PCB_LAYOUT, required: false },
    { ...PHASE_MANUFACTURING, required: false },
    { ...PHASE_FIRMWARE, required: false },
    { ...PHASE_TESTING, required: false },
    { ...PHASE_PRODUCTION, required: false },
    { ...PHASE_FIELD_SUPPORT, required: false },
  ],

  simulations: [
    { ...SIM_SPICE_DC, required: false },
    { ...SIM_SPICE_AC, required: false },
    { ...SIM_SPICE_TRANSIENT, required: false },
    { ...SIM_SPICE_MONTE_CARLO, required: false },
    { ...SIM_THERMAL_STEADY, required: false },
    { ...SIM_THERMAL_TRANSIENT, required: false },
    { ...SIM_SIGNAL_INTEGRITY, required: false },
    { ...SIM_POWER_INTEGRITY, required: false },
    { ...SIM_RF_SPARAMETERS, required: false },
    { ...SIM_EMC_RADIATED, required: false },
  ],

  validations: [
    { ...VAL_DRC, required: false },
    { ...VAL_ERC, required: false },
    { ...VAL_IPC2221, required: false },
    { ...VAL_SIGNAL_INTEGRITY, required: false },
    { ...VAL_THERMAL, required: false },
    { ...VAL_DFM, required: false },
    { ...VAL_BEST_PRACTICES, required: false },
    { ...VAL_TESTING, required: false },
  ],

  layoutAgents: [
    AGENT_CONSERVATIVE,
    AGENT_AGGRESSIVE_COMPACT,
    AGENT_THERMAL_OPTIMIZED,
    AGENT_EMI_OPTIMIZED,
    AGENT_DFM_OPTIMIZED,
  ],

  subsystemTemplates: [],

  defaultBoardConstraints: {
    minLayers: 2,
    maxLayers: 16,
    defaultLayers: 4,
    minTraceWidth: 0.1,
    minClearance: 0.1,
    minViaDiameter: 0.4,
    minViaDrill: 0.2,
    defaultCopperWeight: 1,
    impedanceControlDefault: false,
  },

  manufacturingRecommendations: {
    recommendedMaterial: 'fr4',
    recommendedFinish: 'enig',
    viaFillingRecommended: false,
    assemblyType: 'mixed',
    qualityLevel: 'production',
  },

  options: {
    firmwareApplicable: true,
    thermalCritical: false,
    signalIntegrityCritical: false,
    rfApplicable: false,
    emcTestingRequired: false,
    safetyCertifications: [],
    targetApplications: ['Custom Applications'],
  },
};

// ============================================================================
// Configuration Registry
// ============================================================================

/**
 * Complete registry of all project type configurations.
 */
export const PROJECT_TYPE_CONFIGURATIONS: Record<ProjectType, ProjectTypeConfiguration> = {
  [ProjectType.POWER_ELECTRONICS]: POWER_ELECTRONICS_CONFIG,
  [ProjectType.ANALOG_CIRCUIT]: ANALOG_CIRCUIT_CONFIG,
  [ProjectType.DIGITAL_LOGIC]: DIGITAL_LOGIC_CONFIG,
  [ProjectType.MIXED_SIGNAL]: MIXED_SIGNAL_CONFIG,
  [ProjectType.RF_DESIGN]: RF_DESIGN_CONFIG,
  [ProjectType.IOT_DEVICE]: IOT_DEVICE_CONFIG,
  [ProjectType.PASSIVE_BOARD]: PASSIVE_BOARD_CONFIG,
  [ProjectType.FIRMWARE_ONLY]: FIRMWARE_ONLY_CONFIG,
  [ProjectType.CUSTOM]: CUSTOM_CONFIG,
};

// ============================================================================
// Helper Functions
// ============================================================================

/**
 * Get the configuration for a specific project type.
 *
 * @param type - The project type to get configuration for
 * @returns The complete configuration for the project type
 */
export function getProjectTypeConfiguration(type: ProjectType): ProjectTypeConfiguration {
  const config = PROJECT_TYPE_CONFIGURATIONS[type];
  if (!config) {
    throw new Error(`Unknown project type: ${type}`);
  }
  return config;
}

/**
 * Get the default phase configuration for a project type.
 * Returns only enabled phases in execution order.
 *
 * @param type - The project type
 * @returns Array of enabled phase configurations
 */
export function getDefaultPhaseConfig(type: ProjectType): PhaseConfig[] {
  const config = getProjectTypeConfiguration(type);
  return config.phases.filter(phase => phase.enabledByDefault);
}

/**
 * Get required phases for a project type.
 *
 * @param type - The project type
 * @returns Array of required phase configurations
 */
export function getRequiredPhases(type: ProjectType): PhaseConfig[] {
  const config = getProjectTypeConfiguration(type);
  return config.phases.filter(phase => phase.required);
}

/**
 * Get all enabled simulations for a project type.
 *
 * @param type - The project type
 * @returns Array of enabled simulation configurations
 */
export function getEnabledSimulations(type: ProjectType): SimulationConfig[] {
  const config = getProjectTypeConfiguration(type);
  return config.simulations.filter(sim => sim.enabledByDefault);
}

/**
 * Get required simulations for a project type.
 *
 * @param type - The project type
 * @returns Array of required simulation configurations
 */
export function getRequiredSimulations(type: ProjectType): SimulationConfig[] {
  const config = getProjectTypeConfiguration(type);
  return config.simulations.filter(sim => sim.required);
}

/**
 * Get enabled layout agents for a project type, sorted by priority.
 *
 * @param type - The project type
 * @returns Array of enabled layout agent configurations sorted by priority weight (descending)
 */
export function getEnabledLayoutAgents(type: ProjectType): LayoutAgentConfig[] {
  const config = getProjectTypeConfiguration(type);
  return config.layoutAgents
    .filter(agent => agent.enabled)
    .sort((a, b) => b.priorityWeight - a.priorityWeight);
}

/**
 * Get a summary of all project types for UI display.
 *
 * @returns Array of project type summaries
 */
export function getAllProjectTypeSummaries(): Array<{
  type: ProjectType;
  displayName: string;
  description: string;
  phaseCount: number;
  requiredPhases: ProjectPhase[];
  hasFirmware: boolean;
  hasThermal: boolean;
  hasRF: boolean;
  typicalComplexity: ProjectComplexity[];
}> {
  return Object.values(PROJECT_TYPE_CONFIGURATIONS).map(config => ({
    type: config.type,
    displayName: config.displayName,
    description: config.description,
    phaseCount: config.phases.filter(p => p.enabledByDefault).length,
    requiredPhases: config.phases.filter(p => p.required).map(p => p.phase),
    hasFirmware: config.options.firmwareApplicable,
    hasThermal: config.options.thermalCritical,
    hasRF: config.options.rfApplicable,
    typicalComplexity: config.typicalComplexity,
  }));
}

/**
 * Validate that a project configuration is valid for its type.
 *
 * @param type - The project type
 * @param activePhases - Phases the user wants to enable
 * @returns Validation result with any issues found
 */
export function validateProjectConfiguration(
  type: ProjectType,
  activePhases: ProjectPhase[]
): { valid: boolean; errors: string[]; warnings: string[] } {
  const config = getProjectTypeConfiguration(type);
  const errors: string[] = [];
  const warnings: string[] = [];

  // Check required phases are included
  const requiredPhases = config.phases.filter(p => p.required).map(p => p.phase);
  for (const required of requiredPhases) {
    if (!activePhases.includes(required)) {
      errors.push(`Required phase '${required}' is not included`);
    }
  }

  // Check dependencies are satisfied
  for (const phaseId of activePhases) {
    const phaseConfig = config.phases.find(p => p.phase === phaseId);
    if (phaseConfig) {
      for (const dep of phaseConfig.dependencies) {
        if (!activePhases.includes(dep)) {
          errors.push(`Phase '${phaseId}' requires '${dep}' which is not included`);
        }
      }
    }
  }

  // Check for recommended phases
  const recommendedPhases = config.phases.filter(p => p.enabledByDefault && !p.required);
  for (const recommended of recommendedPhases) {
    if (!activePhases.includes(recommended.phase)) {
      warnings.push(`Recommended phase '${recommended.phase}' is not included`);
    }
  }

  return {
    valid: errors.length === 0,
    errors,
    warnings,
  };
}

/**
 * Estimate total project duration based on selected phases.
 *
 * @param type - The project type
 * @param activePhases - Phases to include in estimate
 * @returns Estimated duration in hours
 */
export function estimateProjectDuration(type: ProjectType, activePhases: ProjectPhase[]): number {
  const config = getProjectTypeConfiguration(type);
  let totalHours = 0;

  for (const phaseId of activePhases) {
    const phaseConfig = config.phases.find(p => p.phase === phaseId);
    if (phaseConfig) {
      totalHours += phaseConfig.estimatedDuration;
    }
  }

  return totalHours;
}

/**
 * Get the recommended project type based on project characteristics.
 *
 * @param characteristics - Project characteristics
 * @returns Recommended project type
 */
export function recommendProjectType(characteristics: {
  hasPowerComponents: boolean;
  hasAnalogCircuits: boolean;
  hasDigitalLogic: boolean;
  hasRFComponents: boolean;
  hasWirelessConnectivity: boolean;
  hasFirmware: boolean;
  hasPassiveOnly: boolean;
  maxPower?: number;
  componentCount?: number;
}): ProjectType {
  // Firmware only
  if (characteristics.hasFirmware && !characteristics.hasAnalogCircuits &&
      !characteristics.hasDigitalLogic && !characteristics.hasPowerComponents) {
    return ProjectType.FIRMWARE_ONLY;
  }

  // Passive board
  if (characteristics.hasPassiveOnly) {
    return ProjectType.PASSIVE_BOARD;
  }

  // RF design
  if (characteristics.hasRFComponents) {
    return ProjectType.RF_DESIGN;
  }

  // IoT device
  if (characteristics.hasWirelessConnectivity && characteristics.hasFirmware) {
    return ProjectType.IOT_DEVICE;
  }

  // Power electronics
  if (characteristics.hasPowerComponents && (characteristics.maxPower ?? 0) > 10) {
    return ProjectType.POWER_ELECTRONICS;
  }

  // Mixed signal
  if (characteristics.hasAnalogCircuits && characteristics.hasDigitalLogic) {
    return ProjectType.MIXED_SIGNAL;
  }

  // Digital logic
  if (characteristics.hasDigitalLogic && !characteristics.hasAnalogCircuits) {
    return ProjectType.DIGITAL_LOGIC;
  }

  // Analog circuit
  if (characteristics.hasAnalogCircuits && !characteristics.hasDigitalLogic) {
    return ProjectType.ANALOG_CIRCUIT;
  }

  // Default to custom
  return ProjectType.CUSTOM;
}
