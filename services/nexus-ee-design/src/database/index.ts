/**
 * EE Design Partner - Database Module
 *
 * Exports all database functionality for the EE Design Partner backend.
 */

// Connection and query functions
export {
  getPool,
  query,
  withTransaction,
  clientQuery,
  healthCheck,
  getPoolStats,
  closePool,
  rawQuery,
  buildInsert,
  buildUpdate,
  buildSelect,
  DatabaseError,
} from './connection.js';

// Types
export type {
  QueryOptions,
  TransactionOptions,
  TransactionCallback,
} from './connection.js';

// ============================================================================
// Repositories
// ============================================================================

// Project Repository
export {
  createProject,
  findProjectById,
  findAllProjects,
  updateProject,
  deleteProject,
  updateProjectStatus,
  updateProjectPhase,
  getProjectPhases,
  countProjects,
  batchUpdateProjects,
  projectRepository,
} from './repositories/index.js';

export type {
  CreateProjectInput,
  UpdateProjectInput,
  ProjectFilters,
} from './repositories/index.js';

// Schematic Repository
export {
  createSchematic,
  findSchematicById,
  findSchematicsByProject,
  updateSchematic,
  updateSchematicKicadContent,
  updateSchematicValidation,
  getSchematicKicadContent,
  lockSchematic,
  unlockSchematic,
  deleteSchematic,
  getLatestSchematicVersion,
  createNewSchematicVersion,
  schematicRepository,
} from './repositories/index.js';

export type {
  CreateSchematicInput,
  UpdateSchematicInput,
  SchematicFilters,
} from './repositories/index.js';

// PCB Layout Repository
export {
  createPCBLayout,
  findPCBLayoutById,
  findPCBLayoutsByProject,
  updatePCBLayout,
  updatePCBKicadContent,
  updateDrcResults,
  addMaposIteration,
  getMaposIterations,
  updateMaposConfig,
  updatePCBScores,
  getPCBKicadContent,
  deletePCBLayout,
  getLatestPCBVersion,
  pcbRepository,
} from './repositories/index.js';

export type {
  CreatePCBLayoutInput,
  UpdatePCBLayoutInput,
  MaposIterationInput,
  DRCResults,
  PCBLayoutFilters,
} from './repositories/index.js';

// Simulation Repository
export {
  createSimulation,
  findSimulationById,
  findSimulationsByProject,
  updateSimulationStatus,
  completeSimulation,
  failSimulation,
  updateSimulation,
  cancelSimulation,
  deleteSimulation,
  getPendingSimulations,
  retrySimulation,
  simulationRepository,
} from './repositories/index.js';

export type {
  CreateSimulationInput,
  UpdateSimulationInput,
  SimulationFilters,
} from './repositories/index.js';

// Firmware Repository
export {
  createFirmware,
  findFirmwareById,
  findFirmwareByProject,
  updateFirmware,
  updateFirmwareSourceFiles,
  updateFirmwareBuildStatus,
  updateFirmwarePeripheralConfigs,
  updateFirmwarePinMappings,
  setFirmwareSourceTreePath,
  deleteFirmwareProject,
  getLatestFirmware,
  firmwareRepository,
} from './repositories/index.js';

export type {
  CreateFirmwareInput,
  UpdateFirmwareInput,
  FirmwareFilters,
} from './repositories/index.js';

// Repository collection
export { repositories } from './repositories/index.js';

// Default export for convenience
export { default } from './connection.js';
