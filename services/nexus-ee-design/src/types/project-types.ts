/**
 * EE Design Partner - Project Type System
 *
 * Comprehensive type definitions for project classification and phase configuration.
 * Each project type has tailored phase pipelines, simulation requirements, and layout strategies.
 */

import type {
  ProjectPhase,
  SimulationType,
  ValidationType,
  LayoutStrategy,
} from './index.js';

// ============================================================================
// Project Type Enums
// ============================================================================

/**
 * Classification of electronic design projects.
 * Determines which phases, simulations, and validations are applicable.
 */
export enum ProjectType {
  /** High-power designs: motor drivers, power supplies, battery management */
  POWER_ELECTRONICS = 'power_electronics',

  /** Pure analog circuits: amplifiers, filters, sensors, precision measurement */
  ANALOG_CIRCUIT = 'analog_circuit',

  /** Digital-only designs: FPGAs, microcontroller boards, digital logic */
  DIGITAL_LOGIC = 'digital_logic',

  /** Combined analog/digital: data acquisition, mixed-signal processing */
  MIXED_SIGNAL = 'mixed_signal',

  /** Radio frequency designs: wireless modules, antennas, RF front-ends */
  RF_DESIGN = 'rf_design',

  /** Connected devices: sensors with wireless, edge computing, wearables */
  IOT_DEVICE = 'iot_device',

  /** Simple boards: breakouts, adapters, passive networks, connectors */
  PASSIVE_BOARD = 'passive_board',

  /** Software-only projects: embedded firmware without new hardware */
  FIRMWARE_ONLY = 'firmware_only',

  /** User-defined configuration for specialized requirements */
  CUSTOM = 'custom',
}

/**
 * Project complexity levels affecting resource allocation and validation depth.
 */
export enum ProjectComplexity {
  /** Simple designs: <20 components, 2 layers, minimal constraints */
  LOW = 'low',

  /** Standard designs: 20-100 components, 2-4 layers */
  MEDIUM = 'medium',

  /** Complex designs: 100-500 components, 4-8 layers, multi-domain */
  HIGH = 'high',

  /** Extreme complexity: >500 components, 8+ layers, safety-critical */
  EXTREME = 'extreme',
}

/**
 * Categories for organizing subsystem templates.
 */
export enum SubsystemCategory {
  /** Power management: regulators, converters, protection */
  POWER = 'power',

  /** Signal conditioning: amplifiers, filters, ADC/DAC */
  SIGNAL_CONDITIONING = 'signal_conditioning',

  /** Processing: MCUs, FPGAs, DSPs */
  PROCESSING = 'processing',

  /** Communication: USB, Ethernet, CAN, wireless */
  COMMUNICATION = 'communication',

  /** Sensing: temperature, current, position, IMU */
  SENSING = 'sensing',

  /** Actuation: motor drivers, solenoids, relays */
  ACTUATION = 'actuation',

  /** User interface: displays, LEDs, buttons, touch */
  USER_INTERFACE = 'user_interface',

  /** Protection: ESD, overvoltage, overcurrent */
  PROTECTION = 'protection',

  /** Connectivity: connectors, headers, test points */
  CONNECTIVITY = 'connectivity',

  /** Thermal: heatsinks, thermal vias, cooling */
  THERMAL = 'thermal',
}

// ============================================================================
// Phase Configuration
// ============================================================================

/**
 * Configuration for a single project phase.
 */
export interface PhaseConfig {
  /** Phase identifier matching ProjectPhase type */
  phase: ProjectPhase;

  /** Human-readable phase name */
  displayName: string;

  /** Whether this phase is required for the project type */
  required: boolean;

  /** Whether this phase is enabled by default */
  enabledByDefault: boolean;

  /** Estimated duration in hours for this phase */
  estimatedDuration: number;

  /** Phase dependencies - must complete before this phase */
  dependencies: ProjectPhase[];

  /** Skills available during this phase */
  availableSkills: string[];

  /** Validation domains to run after this phase completes */
  validationDomains: ValidationType[];

  /** Deliverables produced by this phase */
  deliverables: string[];

  /** Phase-specific configuration options */
  options?: PhaseOptions;
}

/**
 * Phase-specific configuration options.
 */
export interface PhaseOptions {
  /** Maximum iterations for iterative phases (e.g., PCB layout) */
  maxIterations?: number;

  /** Target quality score (0-100) */
  targetScore?: number;

  /** Required approval level */
  approvalLevel?: 'auto' | 'reviewer' | 'owner';

  /** Whether multi-LLM validation is required */
  multiLlmValidation?: boolean;

  /** Custom phase parameters */
  parameters?: Record<string, unknown>;
}

// ============================================================================
// Simulation Configuration
// ============================================================================

/**
 * Configuration for simulation requirements.
 */
export interface SimulationConfig {
  /** Simulation type from SimulationType */
  type: SimulationType;

  /** Human-readable name */
  displayName: string;

  /** Whether this simulation is required */
  required: boolean;

  /** Whether enabled by default */
  enabledByDefault: boolean;

  /** Priority order (lower = higher priority) */
  priority: number;

  /** Estimated simulation time in minutes */
  estimatedDuration: number;

  /** Dependencies on other simulations */
  dependencies: SimulationType[];

  /** Pass/fail criteria */
  criteria: SimulationCriteria;

  /** Default parameters for this simulation */
  defaultParameters: Record<string, unknown>;
}

/**
 * Pass/fail criteria for simulations.
 */
export interface SimulationCriteria {
  /** Minimum passing score (0-100) */
  minScore: number;

  /** Critical metrics that must pass */
  criticalMetrics: CriticalMetric[];

  /** Warning thresholds */
  warningThresholds: Record<string, number>;
}

/**
 * A critical metric that must pass validation.
 */
export interface CriticalMetric {
  /** Metric name */
  name: string;

  /** Metric unit */
  unit: string;

  /** Minimum acceptable value (undefined = no minimum) */
  min?: number;

  /** Maximum acceptable value (undefined = no maximum) */
  max?: number;
}

// ============================================================================
// Validation Configuration
// ============================================================================

/**
 * Configuration for validation requirements.
 */
export interface ValidationConfig {
  /** Validation domain type */
  type: ValidationType;

  /** Human-readable name */
  displayName: string;

  /** Whether this validation is required */
  required: boolean;

  /** Whether enabled by default */
  enabledByDefault: boolean;

  /** Weight in overall score calculation (0-1) */
  weight: number;

  /** Severity level for failures */
  failureSeverity: 'blocking' | 'warning' | 'info';

  /** Rules to apply for this validation */
  rules: ValidationRule[];
}

/**
 * A single validation rule.
 */
export interface ValidationRule {
  /** Rule identifier */
  id: string;

  /** Rule name */
  name: string;

  /** Rule description */
  description: string;

  /** Severity if rule fails */
  severity: 'critical' | 'error' | 'warning' | 'info';

  /** Whether rule is enabled */
  enabled: boolean;

  /** Rule parameters */
  parameters?: Record<string, unknown>;
}

// ============================================================================
// Subsystem Templates
// ============================================================================

/**
 * Pre-defined subsystem template for rapid design assembly.
 */
export interface SubsystemTemplate {
  /** Unique template identifier */
  id: string;

  /** Template name */
  name: string;

  /** Template description */
  description: string;

  /** Category for organization */
  category: SubsystemCategory;

  /** Component list */
  components: SubsystemComponent[];

  /** Default schematic snippet (KiCad S-expression or reference) */
  schematicTemplate?: string;

  /** Layout constraints and recommendations */
  layoutHints: LayoutHints;

  /** Applicable project types */
  applicableTypes: ProjectType[];

  /** Tags for search */
  tags: string[];

  /** Version of the template */
  version: string;
}

/**
 * Component within a subsystem template.
 */
export interface SubsystemComponent {
  /** Component reference designator pattern (e.g., "U*", "R*") */
  referencePattern: string;

  /** Component type/function */
  type: string;

  /** Value or part number pattern */
  valuePattern: string;

  /** Whether component is required or optional */
  required: boolean;

  /** Quantity in template */
  quantity: number;

  /** Design notes */
  notes?: string;
}

/**
 * Layout hints for subsystem placement and routing.
 */
export interface LayoutHints {
  /** Preferred placement region */
  preferredRegion?: 'center' | 'edge' | 'corner' | 'near_mcu' | 'near_connector';

  /** Minimum clearance from other subsystems (mm) */
  minClearance: number;

  /** Whether components should be grouped together */
  keepTogether: boolean;

  /** Thermal considerations */
  thermalRequirements?: ThermalRequirements;

  /** EMI/EMC considerations */
  emcRequirements?: EMCRequirements;

  /** Routing priority (lower = higher priority) */
  routingPriority: number;
}

/**
 * Thermal requirements for subsystem layout.
 */
export interface ThermalRequirements {
  /** Maximum junction temperature (Celsius) */
  maxJunctionTemp: number;

  /** Required thermal vias */
  thermalVias: boolean;

  /** Heatsink required */
  heatsinkRequired: boolean;

  /** Minimum copper pour area (mm^2) */
  minCopperArea?: number;
}

/**
 * EMC requirements for subsystem layout.
 */
export interface EMCRequirements {
  /** Shield required */
  shieldRequired: boolean;

  /** Guard ring required */
  guardRing: boolean;

  /** Maximum trace length (mm) */
  maxTraceLength?: number;

  /** Sensitive to noise */
  noiseSensitive: boolean;

  /** Noise generator */
  noiseGenerator: boolean;
}

// ============================================================================
// Layout Agent Configuration
// ============================================================================

/**
 * Configuration for PCB layout agents.
 */
export interface LayoutAgentConfig {
  /** Agent strategy type */
  strategy: LayoutStrategy;

  /** Human-readable name */
  displayName: string;

  /** Agent description */
  description: string;

  /** Whether this agent is enabled for the project type */
  enabled: boolean;

  /** Priority weight for tournament selection (higher = more likely to be selected) */
  priorityWeight: number;

  /** Strategy-specific parameters */
  parameters: LayoutAgentParameters;
}

/**
 * Parameters for layout agent strategies.
 */
export interface LayoutAgentParameters {
  /** Placement density target (0-1) */
  densityTarget: number;

  /** Thermal margin safety factor */
  thermalMargin: number;

  /** EMI clearance multiplier */
  emiClearanceMultiplier: number;

  /** DFM strictness (0-1) */
  dfmStrictness: number;

  /** Component grouping aggressiveness (0-1) */
  groupingAggressiveness: number;

  /** Via minimization priority (0-1) */
  viaMinimization: number;

  /** Trace length optimization priority (0-1) */
  traceLengthOptimization: number;
}

// ============================================================================
// Main Project Type Configuration
// ============================================================================

/**
 * Complete configuration for a project type.
 * Defines all phases, simulations, validations, and layout strategies.
 */
export interface ProjectTypeConfiguration {
  /** Project type identifier */
  type: ProjectType;

  /** Human-readable name */
  displayName: string;

  /** Project type description */
  description: string;

  /** Typical complexity range */
  typicalComplexity: ProjectComplexity[];

  /** Configured phases for this project type */
  phases: PhaseConfig[];

  /** Required and optional simulations */
  simulations: SimulationConfig[];

  /** Validation configurations */
  validations: ValidationConfig[];

  /** Layout agent configurations */
  layoutAgents: LayoutAgentConfig[];

  /** Applicable subsystem templates */
  subsystemTemplates: string[]; // Template IDs

  /** Default board constraints */
  defaultBoardConstraints: DefaultBoardConstraints;

  /** Manufacturing recommendations */
  manufacturingRecommendations: ManufacturingRecommendations;

  /** Project type-specific options */
  options: ProjectTypeOptions;
}

/**
 * Default board constraints for a project type.
 */
export interface DefaultBoardConstraints {
  /** Recommended minimum layers */
  minLayers: number;

  /** Recommended maximum layers */
  maxLayers: number;

  /** Default layer count */
  defaultLayers: number;

  /** Minimum trace width (mm) */
  minTraceWidth: number;

  /** Minimum clearance (mm) */
  minClearance: number;

  /** Minimum via diameter (mm) */
  minViaDiameter: number;

  /** Minimum via drill (mm) */
  minViaDrill: number;

  /** Default copper weight (oz) */
  defaultCopperWeight: number;

  /** Whether impedance control is typically needed */
  impedanceControlDefault: boolean;
}

/**
 * Manufacturing recommendations for a project type.
 */
export interface ManufacturingRecommendations {
  /** Recommended PCB material */
  recommendedMaterial: 'fr4' | 'aluminum' | 'rogers' | 'polyimide';

  /** Recommended surface finish */
  recommendedFinish: 'hasl' | 'enig' | 'osp' | 'immersion_silver' | 'immersion_tin';

  /** Whether via filling is typically recommended */
  viaFillingRecommended: boolean;

  /** Typical assembly type */
  assemblyType: 'manual' | 'smt_only' | 'mixed' | 'through_hole';

  /** Quality level recommendation */
  qualityLevel: 'prototype' | 'production' | 'automotive' | 'aerospace';
}

/**
 * Project type-specific options.
 */
export interface ProjectTypeOptions {
  /** Whether firmware generation is applicable */
  firmwareApplicable: boolean;

  /** Whether thermal analysis is critical */
  thermalCritical: boolean;

  /** Whether SI analysis is critical */
  signalIntegrityCritical: boolean;

  /** Whether RF analysis is applicable */
  rfApplicable: boolean;

  /** Whether EMC testing is required */
  emcTestingRequired: boolean;

  /** Safety certification requirements */
  safetyCertifications: string[];

  /** Typical target applications */
  targetApplications: string[];
}

// ============================================================================
// Helper Types
// ============================================================================

/**
 * Phase with status information for project tracking.
 */
export interface PhaseStatus extends PhaseConfig {
  /** Current status */
  status: 'pending' | 'in_progress' | 'completed' | 'skipped' | 'blocked';

  /** Start timestamp */
  startedAt?: string;

  /** Completion timestamp */
  completedAt?: string;

  /** Blocking issues */
  blockingIssues?: string[];

  /** Phase output/deliverable references */
  outputs?: Record<string, string>;
}

/**
 * Project configuration instance with selected options.
 */
export interface ProjectConfiguration {
  /** Selected project type */
  projectType: ProjectType;

  /** Project complexity level */
  complexity: ProjectComplexity;

  /** Active phases (from type configuration) */
  activePhases: ProjectPhase[];

  /** Enabled simulations */
  enabledSimulations: SimulationType[];

  /** Enabled validations */
  enabledValidations: ValidationType[];

  /** Selected layout agents */
  selectedAgents: LayoutStrategy[];

  /** Custom overrides */
  overrides?: Partial<ProjectTypeConfiguration>;
}

/**
 * Summary of a project type for UI display.
 */
export interface ProjectTypeSummary {
  /** Project type */
  type: ProjectType;

  /** Display name */
  displayName: string;

  /** Short description */
  description: string;

  /** Number of phases */
  phaseCount: number;

  /** Required phases */
  requiredPhases: ProjectPhase[];

  /** Has firmware phase */
  hasFirmware: boolean;

  /** Has thermal simulation */
  hasThermal: boolean;

  /** Has RF simulation */
  hasRF: boolean;

  /** Typical complexity */
  typicalComplexity: ProjectComplexity[];
}
