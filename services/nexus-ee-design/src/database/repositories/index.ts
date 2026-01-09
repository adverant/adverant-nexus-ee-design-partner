/**
 * EE Design Partner - Database Repositories Index
 *
 * Central export for all database repository modules.
 * Each repository provides type-safe CRUD operations and domain-specific
 * database operations for its respective entity.
 */

// ============================================================================
// Project Repository
// ============================================================================

export {
  create as createProject,
  findById as findProjectById,
  findAll as findAllProjects,
  update as updateProject,
  deleteProject,
  updateStatus as updateProjectStatus,
  updatePhase as updateProjectPhase,
  getPhases as getProjectPhases,
  count as countProjects,
  batchUpdate as batchUpdateProjects,
  default as projectRepository,
} from './project-repository.js';

export type {
  CreateProjectInput,
  UpdateProjectInput,
  ProjectFilters,
} from './project-repository.js';

// ============================================================================
// Schematic Repository
// ============================================================================

export {
  create as createSchematic,
  findById as findSchematicById,
  findByProject as findSchematicsByProject,
  update as updateSchematic,
  updateKicadContent as updateSchematicKicadContent,
  updateValidation as updateSchematicValidation,
  getKicadContent as getSchematicKicadContent,
  lock as lockSchematic,
  unlock as unlockSchematic,
  deleteSchematic,
  getLatestVersion as getLatestSchematicVersion,
  createNewVersion as createNewSchematicVersion,
  default as schematicRepository,
} from './schematic-repository.js';

export type {
  CreateSchematicInput,
  UpdateSchematicInput,
  SchematicFilters,
} from './schematic-repository.js';

// ============================================================================
// PCB Layout Repository
// ============================================================================

export {
  create as createPCBLayout,
  findById as findPCBLayoutById,
  findByProject as findPCBLayoutsByProject,
  update as updatePCBLayout,
  updateKicadContent as updatePCBKicadContent,
  updateDrcResults,
  addMaposIteration,
  getMaposIterations,
  updateMaposConfig,
  updateScores as updatePCBScores,
  getKicadContent as getPCBKicadContent,
  deletePCBLayout,
  getLatestVersion as getLatestPCBVersion,
  default as pcbRepository,
} from './pcb-repository.js';

export type {
  CreatePCBLayoutInput,
  UpdatePCBLayoutInput,
  MaposIterationInput,
  DRCResults,
  PCBLayoutFilters,
} from './pcb-repository.js';

// ============================================================================
// Simulation Repository
// ============================================================================

export {
  create as createSimulation,
  findById as findSimulationById,
  findByProject as findSimulationsByProject,
  updateStatus as updateSimulationStatus,
  complete as completeSimulation,
  fail as failSimulation,
  update as updateSimulation,
  cancel as cancelSimulation,
  deleteSimulation,
  getPendingSimulations,
  retry as retrySimulation,
  default as simulationRepository,
} from './simulation-repository.js';

export type {
  CreateSimulationInput,
  UpdateSimulationInput,
  SimulationFilters,
} from './simulation-repository.js';

// ============================================================================
// Firmware Repository
// ============================================================================

export {
  create as createFirmware,
  findById as findFirmwareById,
  findByProject as findFirmwareByProject,
  update as updateFirmware,
  updateSourceFiles as updateFirmwareSourceFiles,
  updateBuildStatus as updateFirmwareBuildStatus,
  updatePeripheralConfigs as updateFirmwarePeripheralConfigs,
  updatePinMappings as updateFirmwarePinMappings,
  setSourceTreePath as setFirmwareSourceTreePath,
  deleteFirmwareProject,
  getLatest as getLatestFirmware,
  default as firmwareRepository,
} from './firmware-repository.js';

export type {
  CreateFirmwareInput,
  UpdateFirmwareInput,
  FirmwareFilters,
} from './firmware-repository.js';

// ============================================================================
// Repository Collection
// ============================================================================

/**
 * All repositories bundled together for convenient access.
 */
export const repositories = {
  project: {
    create: async (input: import('./project-repository.js').CreateProjectInput) =>
      (await import('./project-repository.js')).create(input),
    findById: async (id: string) =>
      (await import('./project-repository.js')).findById(id),
    findAll: async (filters?: import('./project-repository.js').ProjectFilters) =>
      (await import('./project-repository.js')).findAll(filters),
    update: async (id: string, data: import('./project-repository.js').UpdateProjectInput) =>
      (await import('./project-repository.js')).update(id, data),
    delete: async (id: string) =>
      (await import('./project-repository.js')).deleteProject(id),
    updateStatus: async (id: string, status: string) =>
      (await import('./project-repository.js')).updateStatus(id, status as import('../../types/index.js').ProjectStatus),
    updatePhase: async (id: string, phase: string) =>
      (await import('./project-repository.js')).updatePhase(id, phase as import('../../types/index.js').ProjectPhase),
    getPhases: async (projectId: string) =>
      (await import('./project-repository.js')).getPhases(projectId),
  },
  schematic: {
    create: async (input: import('./schematic-repository.js').CreateSchematicInput) =>
      (await import('./schematic-repository.js')).create(input),
    findById: async (id: string) =>
      (await import('./schematic-repository.js')).findById(id),
    findByProject: async (projectId: string) =>
      (await import('./schematic-repository.js')).findByProject(projectId),
    update: async (id: string, data: import('./schematic-repository.js').UpdateSchematicInput) =>
      (await import('./schematic-repository.js')).update(id, data),
    updateKicadContent: async (id: string, kicadSch: string) =>
      (await import('./schematic-repository.js')).updateKicadContent(id, kicadSch),
    updateValidation: async (id: string, results: object) =>
      (await import('./schematic-repository.js')).updateValidation(id, results as import('../../types/index.js').ValidationResults),
    delete: async (id: string) =>
      (await import('./schematic-repository.js')).deleteSchematic(id),
  },
  pcbLayout: {
    create: async (input: import('./pcb-repository.js').CreatePCBLayoutInput) =>
      (await import('./pcb-repository.js')).create(input),
    findById: async (id: string) =>
      (await import('./pcb-repository.js')).findById(id),
    findByProject: async (projectId: string) =>
      (await import('./pcb-repository.js')).findByProject(projectId),
    update: async (id: string, data: import('./pcb-repository.js').UpdatePCBLayoutInput) =>
      (await import('./pcb-repository.js')).update(id, data),
    updateKicadContent: async (id: string, kicadPcb: string) =>
      (await import('./pcb-repository.js')).updateKicadContent(id, kicadPcb),
    updateDrcResults: async (id: string, drcResults: import('./pcb-repository.js').DRCResults) =>
      (await import('./pcb-repository.js')).updateDrcResults(id, drcResults),
    addMaposIteration: async (layoutId: string, iteration: import('./pcb-repository.js').MaposIterationInput) =>
      (await import('./pcb-repository.js')).addMaposIteration(layoutId, iteration),
    delete: async (id: string) =>
      (await import('./pcb-repository.js')).deletePCBLayout(id),
  },
  simulation: {
    create: async (input: import('./simulation-repository.js').CreateSimulationInput) =>
      (await import('./simulation-repository.js')).create(input),
    findById: async (id: string) =>
      (await import('./simulation-repository.js')).findById(id),
    findByProject: async (projectId: string, filters?: import('./simulation-repository.js').SimulationFilters) =>
      (await import('./simulation-repository.js')).findByProject(projectId, filters),
    updateStatus: async (id: string, status: string, workerId?: string) =>
      (await import('./simulation-repository.js')).updateStatus(id, status as import('../../types/index.js').SimulationStatus, workerId),
    complete: async (id: string, results: object) =>
      (await import('./simulation-repository.js')).complete(id, results as import('../../types/index.js').SimulationResults),
    fail: async (id: string, errorMessage: string) =>
      (await import('./simulation-repository.js')).fail(id, errorMessage),
    delete: async (id: string) =>
      (await import('./simulation-repository.js')).deleteSimulation(id),
  },
  firmware: {
    create: async (input: import('./firmware-repository.js').CreateFirmwareInput) =>
      (await import('./firmware-repository.js')).create(input),
    findById: async (id: string) =>
      (await import('./firmware-repository.js')).findById(id),
    findByProject: async (projectId: string) =>
      (await import('./firmware-repository.js')).findByProject(projectId),
    update: async (id: string, data: import('./firmware-repository.js').UpdateFirmwareInput) =>
      (await import('./firmware-repository.js')).update(id, data),
    updateSourceFiles: async (id: string, files: object) =>
      (await import('./firmware-repository.js')).updateSourceFiles(id, files as import('../../types/index.js').GeneratedFile[]),
    updateBuildStatus: async (id: string, status: string) =>
      (await import('./firmware-repository.js')).updateBuildStatus(id, status as 'not_built' | 'building' | 'success' | 'failed' | 'warnings'),
    delete: async (id: string) =>
      (await import('./firmware-repository.js')).deleteFirmwareProject(id),
  },
};

export default repositories;
