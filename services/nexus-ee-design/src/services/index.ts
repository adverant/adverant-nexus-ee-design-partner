/**
 * Services Index
 *
 * Exports all services for the Nexus EE Design Partner plugin.
 */

// PCB Layout Services
export {
  RalphLoopOrchestrator,
  ValidationFramework,
  PythonExecutor,
  pythonExecutor,
  createAllAgents,
  createAgent,
  AgentRegistry
} from './pcb';

export type {
  RalphLoopConfig,
  RalphLoopResult,
  TournamentPhase,
  AgentScore,
  ValidationDomain,
  ValidationViolation,
  DomainResult,
  ValidationConfig,
  PythonExecutorConfig,
  ScriptResult,
  ScriptJob,
  AgentConfig,
  AgentStrategy,
  StrategyWeights,
  AgentParameters,
  PlacementContext,
  AgentFeedback,
  PlacementResult,
  LayoutMetrics
} from './pcb';

// Validation Services
export { ConsensusEngine } from './validation';

export type {
  ValidatorConfig,
  ConsensusConfig,
  ValidationRequest,
  ValidatorResponse
} from './validation';

// Schematic Services
export { SchematicGenerator, SchematicReviewer } from './schematic';

export type {
  SchematicGeneratorConfig,
  SchematicRequirements,
  PowerRequirements,
  VoltageRail,
  InterfaceRequirement,
  SchematicConstraints,
  GenerationResult as SchematicGenerationResult,
  BOMEntry,
  NetlistEntry,
  ComponentTemplate,
  ComponentCategory,
  PinDefinition,
  SchematicBlock,
  BlockType,
  BlockConnection,
  ReviewConfig,
  CustomRule,
  RuleViolation,
  ReviewResult,
  ERCRule
} from './schematic';

// Simulation Services
export { SimulationOrchestrator } from './simulation';

export type {
  SimulationOrchestratorConfig,
  SimulationContainerConfig,
  ContainerSpec,
  SimulationRequest,
  SimulationOptions,
  SimulationJob,
  SPICEConfig,
  SPICEParameters,
  ThermalConfig,
  ThermalParameters,
  ThermalGeometry,
  MaterialDefinition,
  HeatSource,
  BoundaryCondition,
  SignalIntegrityConfig,
  SIParameters,
  TraceDefinition,
  StackupDefinition,
  LayerDefinition,
  RFEMCConfig,
  RFParameters,
  RFGeometry,
  Excitation,
  PortDefinition
} from './simulation';

// Firmware Services
export { FirmwareGenerator } from './firmware';

export type {
  FirmwareGeneratorConfig,
  FirmwareRequirements,
  PeripheralRequirement,
  FeatureRequirement,
  GenerationResult as FirmwareGenerationResult,
  HALTemplate,
  PeripheralTemplate,
  DriverTemplate,
  DriverFunctionTemplate
} from './firmware';

// Skills Engine Services
export { SkillsEngineClient } from './skills';

export type {
  SkillsEngineConfig,
  SkillDefinition,
  SkillSearchResult,
  SkillExecutionRequest,
  SkillExecutionResponse,
  RegistrationResult
} from './skills';