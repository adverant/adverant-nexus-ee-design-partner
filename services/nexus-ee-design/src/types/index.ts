/**
 * EE Design Partner - Core Type Definitions
 *
 * Comprehensive type system for end-to-end hardware/software development automation
 */

// ============================================================================
// Common Types
// ============================================================================

export interface BuildMetadata {
  buildId: string;
  buildTimestamp: string;
  gitCommit: string;
  gitBranch: string;
  version: string;
}

export interface PaginatedResponse<T> {
  data: T[];
  total: number;
  page: number;
  pageSize: number;
  hasMore: boolean;
}

export interface ApiResponse<T> {
  success: boolean;
  data?: T;
  error?: {
    code: string;
    message: string;
    details?: Record<string, unknown>;
  };
  metadata?: {
    requestId: string;
    timestamp: string;
    duration: number;
  };
}

// ============================================================================
// Project Types
// ============================================================================

export interface EEProject {
  id: string;
  name: string;
  description: string;
  repositoryUrl: string;
  phase: ProjectPhase;
  status: ProjectStatus;
  owner: string;
  collaborators: string[];
  createdAt: string;
  updatedAt: string;
  metadata: ProjectMetadata;
}

export type ProjectPhase =
  | 'ideation'
  | 'architecture'
  | 'schematic'
  | 'simulation'
  | 'pcb_layout'
  | 'manufacturing'
  | 'firmware'
  | 'testing'
  | 'production'
  | 'field_support';

export type ProjectStatus =
  | 'draft'
  | 'in_progress'
  | 'review'
  | 'approved'
  | 'completed'
  | 'on_hold'
  | 'cancelled';

export interface ProjectMetadata {
  targetMcu?: string;
  layerCount?: number;
  componentCount?: number;
  estimatedCost?: number;
  complexity?: 'low' | 'medium' | 'high' | 'extreme';
  tags: string[];
}

// ============================================================================
// Schematic Types
// ============================================================================

export interface Schematic {
  id: string;
  projectId: string;
  name: string;
  version: number;
  filePath: string;
  format: 'kicad_sch' | 'eagle' | 'altium' | 'orcad';
  sheets: SchematicSheet[];
  components: Component[];
  nets: Net[];
  validationResults?: ValidationResults;
  createdAt: string;
  updatedAt: string;
}

export interface SchematicSheet {
  id: string;
  name: string;
  pageNumber: number;
  components: string[]; // Component IDs
  nets: string[]; // Net IDs
}

export interface Component {
  id: string;
  reference: string; // e.g., "U1", "R1", "C1"
  value: string;
  footprint: string;
  partNumber?: string;
  manufacturer?: string;
  description?: string;
  position: Position2D;
  rotation: number;
  properties: Record<string, string>;
  pins: Pin[];
}

export interface Pin {
  id: string;
  name: string;
  number: string;
  type: PinType;
  position: Position2D;
  connectedNet?: string;
}

export type PinType =
  | 'input'
  | 'output'
  | 'bidirectional'
  | 'power_input'
  | 'power_output'
  | 'passive'
  | 'open_collector'
  | 'open_emitter'
  | 'unconnected';

export interface Net {
  id: string;
  name: string;
  class?: string; // Net class for routing rules
  connections: NetConnection[];
  properties: NetProperties;
}

export interface NetConnection {
  componentId: string;
  pinId: string;
}

export interface NetProperties {
  impedance?: number;
  differentialPair?: string;
  maxCurrent?: number;
  maxVoltage?: number;
  lengthMatching?: number;
}

export interface Position2D {
  x: number;
  y: number;
}

export interface Position3D extends Position2D {
  z: number;
}

// ============================================================================
// PCB Layout Types
// ============================================================================

export interface PCBLayout {
  id: string;
  schematicId: string;
  projectId: string;
  version: number;
  filePath: string;
  boardOutline: BoardOutline;
  stackup: LayerStackup;
  components: PlacedComponent[];
  traces: Trace[];
  vias: Via[];
  zones: CopperZone[];
  validationResults?: ValidationResults;
  score: number;
  createdAt: string;
  updatedAt: string;
}

export interface BoardOutline {
  width: number;
  height: number;
  shape: 'rectangular' | 'circular' | 'custom';
  polygon?: Position2D[];
  keepoutZones: KeepoutZone[];
}

export interface KeepoutZone {
  id: string;
  type: 'component' | 'routing' | 'via' | 'copper';
  polygon: Position2D[];
  layers: string[];
}

export interface LayerStackup {
  totalThickness: number;
  layers: Layer[];
}

export interface Layer {
  id: string;
  name: string;
  type: LayerType;
  thickness: number;
  material: string;
  copperWeight?: string; // e.g., "1oz", "2oz"
  dielectricConstant?: number;
}

export type LayerType =
  | 'signal'
  | 'plane'
  | 'mixed'
  | 'dielectric'
  | 'soldermask'
  | 'silkscreen'
  | 'paste';

export interface PlacedComponent extends Component {
  layer: 'top' | 'bottom';
  locked: boolean;
  placementScore?: number;
}

export interface Trace {
  id: string;
  netId: string;
  layer: string;
  width: number;
  points: Position2D[];
  impedance?: number;
}

export interface Via {
  id: string;
  position: Position2D;
  type: ViaType;
  drill: number;
  pad: number;
  startLayer: string;
  endLayer: string;
  netId?: string;
  thermalRelief?: boolean;
}

export type ViaType = 'through' | 'blind' | 'buried' | 'microvia';

export interface CopperZone {
  id: string;
  netId: string;
  layer: string;
  polygon: Position2D[];
  fillType: 'solid' | 'hatched' | 'none';
  thermalRelief: ThermalRelief;
  priority: number;
}

export interface ThermalRelief {
  enabled: boolean;
  gap: number;
  spokeWidth: number;
  spokeCount: number;
}

// ============================================================================
// Simulation Types
// ============================================================================

export interface Simulation {
  id: string;
  projectId: string;
  type: SimulationType;
  name: string;
  status: SimulationStatus;
  input: SimulationInput;
  results?: SimulationResults;
  startedAt?: string;
  completedAt?: string;
  error?: string;
}

export type SimulationType =
  | 'spice_dc'
  | 'spice_ac'
  | 'spice_transient'
  | 'spice_noise'
  | 'spice_monte_carlo'
  | 'thermal_steady_state'
  | 'thermal_transient'
  | 'thermal_cfd'
  | 'signal_integrity'
  | 'power_integrity'
  | 'rf_sparameters'
  | 'rf_field_pattern'
  | 'emc_radiated'
  | 'emc_conducted'
  | 'stress_thermal_cycling'
  | 'stress_vibration'
  | 'reliability_mtbf';

export type SimulationStatus =
  | 'pending'
  | 'running'
  | 'completed'
  | 'failed'
  | 'cancelled';

export interface SimulationInput {
  schematicId?: string;
  pcbLayoutId?: string;
  parameters: Record<string, unknown>;
  testBench?: string;
}

export interface SimulationResults {
  passed: boolean;
  score: number;
  waveforms?: Waveform[];
  images?: SimulationImage[];
  metrics: Record<string, SimulationMetric>;
  warnings: string[];
  recommendations: string[];
}

export interface Waveform {
  name: string;
  xLabel: string;
  yLabel: string;
  xUnit: string;
  yUnit: string;
  data: WaveformPoint[];
}

export interface WaveformPoint {
  x: number;
  y: number;
}

export interface SimulationImage {
  name: string;
  type: 'thermal_map' | 'field_pattern' | 'eye_diagram' | 'impedance_plot';
  url: string;
  metadata?: Record<string, unknown>;
}

export interface SimulationMetric {
  value: number;
  unit: string;
  min?: number;
  max?: number;
  passed: boolean;
}

// ============================================================================
// Firmware Types
// ============================================================================

export interface FirmwareProject {
  id: string;
  projectId: string;
  name: string;
  targetMcu: MCUTarget;
  rtos?: RTOSConfig;
  hal: HALConfig;
  drivers: Driver[];
  tasks: FirmwareTask[];
  buildConfig: BuildConfig;
  generatedFiles: GeneratedFile[];
  createdAt: string;
  updatedAt: string;
}

export interface MCUTarget {
  family: MCUFamily;
  part: string;
  core: string;
  flashSize: number;
  ramSize: number;
  clockSpeed: number;
  peripherals: string[];
}

export type MCUFamily =
  | 'stm32'
  | 'esp32'
  | 'ti_tms320'
  | 'infineon_aurix'
  | 'nordic_nrf'
  | 'rpi_pico'
  | 'nxp_imxrt';

export interface RTOSConfig {
  type: 'freertos' | 'zephyr' | 'tirtos' | 'autosar';
  version: string;
  tickRate: number;
  heapSize: number;
  maxTasks: number;
}

export interface HALConfig {
  type: 'vendor' | 'custom';
  peripherals: PeripheralConfig[];
}

export interface PeripheralConfig {
  type: PeripheralType;
  instance: string;
  config: Record<string, unknown>;
  pinMapping: PinMapping[];
}

export type PeripheralType =
  | 'gpio'
  | 'uart'
  | 'spi'
  | 'i2c'
  | 'adc'
  | 'dac'
  | 'pwm'
  | 'timer'
  | 'can'
  | 'usb'
  | 'ethernet';

export interface PinMapping {
  signal: string;
  pin: string;
  alternate?: number;
}

export interface Driver {
  id: string;
  name: string;
  type: string;
  component: string;
  interface: PeripheralType;
  functions: DriverFunction[];
}

export interface DriverFunction {
  name: string;
  description: string;
  parameters: FunctionParameter[];
  returnType: string;
}

export interface FunctionParameter {
  name: string;
  type: string;
  description: string;
}

export interface FirmwareTask {
  name: string;
  priority: number;
  stackSize: number;
  period?: number;
  description: string;
}

export interface BuildConfig {
  toolchain: 'gcc-arm' | 'clang' | 'iar' | 'keil';
  buildSystem: 'cmake' | 'make' | 'ninja';
  optimizationLevel: 'O0' | 'O1' | 'O2' | 'O3' | 'Os' | 'Og';
  debugSymbols: boolean;
  defines: Record<string, string>;
}

export interface GeneratedFile {
  path: string;
  type: 'source' | 'header' | 'config' | 'linker' | 'build';
  content: string;
  generatedAt: string;
}

// ============================================================================
// Manufacturing Types
// ============================================================================

export interface ManufacturingOrder {
  id: string;
  projectId: string;
  pcbLayoutId: string;
  vendor: PCBVendor;
  quantity: number;
  options: PCBOptions;
  gerberFiles: GerberFile[];
  bomFile?: string;
  pickAndPlaceFile?: string;
  quote?: ManufacturingQuote;
  status: OrderStatus;
  trackingNumber?: string;
  createdAt: string;
  updatedAt: string;
}

export type PCBVendor = 'pcbway' | 'jlcpcb' | 'oshpark' | 'eurocircuits' | 'advanced_circuits';

export interface PCBOptions {
  layers: number;
  thickness: number;
  material: 'fr4' | 'aluminum' | 'rogers' | 'polyimide';
  surfaceFinish: 'hasl' | 'enig' | 'osp' | 'immersion_silver' | 'immersion_tin';
  soldermaskColor: string;
  silkscreenColor: string;
  copperWeight: string;
  viaFilling?: boolean;
  impedanceControl?: boolean;
  panelization?: PanelizationConfig;
}

export interface PanelizationConfig {
  columns: number;
  rows: number;
  railWidth: number;
  vScore: boolean;
  mouseBites: boolean;
}

export interface GerberFile {
  layer: string;
  filename: string;
  format: 'gerber_x2' | 'rs274x';
  url: string;
}

export interface ManufacturingQuote {
  vendor: PCBVendor;
  pcbCost: number;
  assemblyCost?: number;
  componentsCost?: number;
  shippingCost: number;
  totalCost: number;
  leadTime: number; // days
  currency: string;
  validUntil: string;
}

export type OrderStatus =
  | 'draft'
  | 'quoted'
  | 'ordered'
  | 'in_production'
  | 'shipped'
  | 'delivered'
  | 'cancelled';

// ============================================================================
// Validation Types
// ============================================================================

export interface ValidationResults {
  passed: boolean;
  score: number;
  timestamp: string;
  domains: ValidationDomain[];
}

export interface ValidationDomain {
  name: string;
  type: ValidationType;
  passed: boolean;
  score: number;
  weight: number;
  violations: Violation[];
  warnings: Warning[];
}

export type ValidationType =
  | 'drc'
  | 'erc'
  | 'ipc_2221'
  | 'signal_integrity'
  | 'thermal'
  | 'dfm'
  | 'best_practices'
  | 'automated_testing';

export interface Violation {
  id: string;
  severity: 'critical' | 'error' | 'warning';
  code: string;
  message: string;
  location?: ViolationLocation;
  suggestion?: string;
}

export interface ViolationLocation {
  componentId?: string;
  netId?: string;
  layer?: string;
  position?: Position2D;
}

export interface Warning {
  code: string;
  message: string;
  location?: ViolationLocation;
}

// ============================================================================
// Multi-LLM Validation Types
// ============================================================================

export interface LLMValidation {
  id: string;
  projectId: string;
  artifactType: 'schematic' | 'pcb' | 'firmware' | 'simulation';
  artifactId: string;
  validators: ValidatorResult[];
  consensus: ConsensusResult;
  createdAt: string;
}

export interface ValidatorResult {
  validator: 'claude_opus' | 'gemini_25_pro' | 'domain_expert';
  score: number;
  passed: boolean;
  issues: LLMIssue[];
  suggestions: string[];
  confidence: number;
  reasoning: string;
  timestamp: string;
}

export interface LLMIssue {
  severity: 'critical' | 'major' | 'minor' | 'info';
  category: string;
  description: string;
  location?: string;
  recommendation: string;
}

export interface LegacyConsensusResult {
  finalScore: number;
  passed: boolean;
  agreementLevel: number; // 0-1, how much validators agreed
  conflicts: ConflictResolution[];
  auditTrail: string;
}

export interface ConflictResolution {
  issue: string;
  validatorOpinions: Record<string, string>;
  resolution: string;
  resolutionMethod: 'majority_vote' | 'weighted_average' | 'expert_override';
}

// ============================================================================
// Agent Types
// ============================================================================

export interface LayoutAgent {
  id: string;
  name: string;
  strategy: LayoutStrategy;
  priority: string[];
  bestFor: string[];
}

export type LayoutStrategy =
  | 'conservative'
  | 'aggressive_compact'
  | 'thermal_optimized'
  | 'emi_optimized'
  | 'dfm_optimized';

export interface AgentResult {
  agentId: string;
  strategy: LayoutStrategy;
  score: number;
  layout?: PCBLayout;
  violations: Violation[];
  metadata: Record<string, unknown>;
  iteration: number;
  timestamp: string;
}

export interface TournamentResult {
  id: string;
  projectId: string;
  iterations: number;
  maxIterations: number;
  converged: boolean;
  convergenceReason?: string;
  winningAgent: AgentResult;
  allResults: AgentResult[];
  finalScore: number;
  startedAt: string;
  completedAt: string;
}

// ============================================================================
// Skill Types
// ============================================================================

export interface Skill {
  name: string;
  displayName: string;
  description: string;
  version: string;
  status: 'draft' | 'testing' | 'published' | 'deprecated';
  visibility: 'private' | 'organization' | 'public';
  allowedTools: string[];
  triggers: string[];
  capabilities: SkillCapability[];
  subSkills?: string[];
}

export interface SkillCapability {
  name: string;
  description: string;
  parameters?: SkillParameter[];
}

export interface SkillParameter {
  name: string;
  type: 'string' | 'number' | 'boolean' | 'object' | 'array';
  required: boolean;
  description: string;
  default?: unknown;
}

export interface SkillExecution {
  id: string;
  skillName: string;
  status: 'pending' | 'running' | 'completed' | 'failed';
  input: Record<string, unknown>;
  output?: Record<string, unknown>;
  error?: string;
  startedAt: string;
  completedAt?: string;
  duration?: number;
}

// ============================================================================
// Extended PCB Layout Types (for Ralph Loop & Validation)
// ============================================================================

export interface BoardConstraints {
  maxWidth: number;
  maxHeight: number;
  layerCount: number;
  minTraceWidth?: number;
  minClearance?: number;
  minViaDiameter?: number;
  minViaDrill?: number;
  copperWeight?: number;
  gridSize?: number;
  minComponentSpacing?: number;
  maxVoltage?: number;
  impedanceControl?: boolean;
  targetImpedance?: number;
}

export interface ComponentPlacement {
  id: string;
  componentId: string;
  reference: string;
  position: { x: number; y: number };
  rotation: number;
  layer: 'top' | 'bottom';
  footprint?: {
    name: string;
    width: number;
    height: number;
  };
  pads?: Pad[];
}

export interface Pad {
  id: string;
  name: string;
  position: { x: number; y: number };
  size: { width: number; height: number };
  shape: 'circular' | 'rectangular' | 'oval';
  drillSize?: number;
}

export interface ValidationResult {
  passed: boolean;
  score: number;
  violations: ValidationViolation[];
  summary: string;
  details?: {
    drc?: DomainValidationResult;
    erc?: DomainValidationResult;
    ipc2221?: DomainValidationResult;
    signalIntegrity?: DomainValidationResult;
    thermal?: DomainValidationResult;
    dfm?: DomainValidationResult;
    bestPractices?: DomainValidationResult;
    testing?: DomainValidationResult;
    [key: string]: DomainValidationResult | undefined;
  };
  timestamp: Date;
  durationMs?: number;
}

export interface DomainValidationResult {
  domain: string;
  score: number;
  maxScore: number;
  violations: ValidationViolation[];
  metrics: Record<string, number | string | boolean>;
  passRate: number;
}

export interface ValidationViolation {
  id: string;
  domain: string;
  severity: 'error' | 'warning' | 'info';
  code: string;
  message: string;
  location?: {
    x?: number;
    y?: number;
    layer?: string;
    componentRef?: string;
    netName?: string;
  };
  suggestion?: string;
}

// ============================================================================
// Multi-LLM Validation Types (Enhanced)
// ============================================================================

export interface MultiLLMValidation {
  id: string;
  requestType: string;
  validators: LLMValidatorResult[];
  consensus: {
    score: number;
    confidence: number;
    passed: boolean;
    agreement: number;
    conflicts: string[];
  };
  summary: string;
  timestamp: Date;
  durationMs: number;
}

export interface LLMValidatorResult {
  name: string;
  type: 'llm' | 'domain-expert';
  score: number;
  confidence: number;
  issues: Array<{
    severity: 'error' | 'warning' | 'info';
    category: string;
    description: string;
    location?: string;
    suggestion?: string;
  }>;
  recommendations: string[];
  latencyMs: number;
}

export interface ConsensusResult {
  score: number;
  confidence: number;
  agreement: number;
  conflicts: string[];
  validatorCount: number;
  passed: boolean;
}