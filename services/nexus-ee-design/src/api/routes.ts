/**
 * EE Design Partner - API Routes
 *
 * REST API endpoints for all phases of hardware/software development.
 * Uses real repository implementations for database operations.
 */

import { Router, Request, Response, NextFunction } from 'express';
import { Server as SocketIOServer } from 'socket.io';
import multer from 'multer';
import { z } from 'zod';
import fs from 'fs/promises';
import path from 'path';

import { ValidationError, NotFoundError } from '../utils/errors.js';
import { log } from '../utils/logger.js';
import { config } from '../config.js';
import { generateMinimalSchematic, parseKicadSchematic } from '../utils/kicad-generator.js';
import { PythonExecutor, ProgressEvent } from '../services/pcb/python-executor.js';
import { getSkillsEngineClient } from '../state.js';
import { createSkillsRoutes } from './skills-routes.js';
import { getSchematicWsManager } from './schematic-ws.js';

// Repository imports
import {
  create as createProject,
  findById as findProjectById,
  findAll as findAllProjects,
  update as updateProject,
  deleteProject,
  updateStatus as updateProjectStatus,
  updatePhase as updateProjectPhase,
  getPhases as getProjectPhases,
  count as countProjects,
  type CreateProjectInput,
  type UpdateProjectInput,
  type ProjectFilters,
} from '../database/repositories/project-repository.js';

import {
  create as createSchematic,
  findById as findSchematicById,
  findByProject as findSchematicsByProject,
  update as updateSchematic,
  updateKicadContent as updateSchematicKicadContent,
  updateValidation as updateSchematicValidation,
  deleteSchematic,
  type CreateSchematicInput,
} from '../database/repositories/schematic-repository.js';

import {
  create as createPCBLayout,
  findById as findPCBLayoutById,
  findByProject as findPCBLayoutsByProject,
  update as updatePCBLayout,
  updateDrcResults,
  updateMaposConfig,
  getMaposIterations,
  updateScores as updatePCBScores,
  getKicadContent,
  type CreatePCBLayoutInput,
  type DRCResults,
} from '../database/repositories/pcb-repository.js';

import {
  create as createSimulation,
  findById as findSimulationById,
  findByProject as findSimulationsByProject,
  updateStatus as updateSimulationStatus,
  complete as completeSimulation,
  fail as failSimulation,
  type CreateSimulationInput,
  type SimulationFilters,
} from '../database/repositories/simulation-repository.js';

import {
  create as createFirmware,
  findById as findFirmwareById,
  findByProject as findFirmwareByProject,
  update as updateFirmware,
  updateSourceFiles as updateFirmwareSourceFiles,
  type CreateFirmwareInput,
} from '../database/repositories/firmware-repository.js';

// File browser imports
import {
  getFileTreeHandler,
  getFileContentHandler,
} from './file-browser.js';

// Configuration imports
import {
  PROJECT_TYPE_CONFIGURATIONS,
  getAllProjectTypeSummaries,
  getProjectTypeConfiguration,
  getEnabledSimulations,
  getEnabledLayoutAgents,
} from '../config/project-type-configs.js';

import { ProjectType } from '../types/project-types.js';
import type { MCUFamily, SimulationType, ProjectPhase } from '../types/index.js';

// File upload configuration
const upload = multer({
  storage: multer.diskStorage({
    destination: config.storage.tempDir,
    filename: (_req, file, cb) => {
      const uniqueSuffix = `${Date.now()}-${Math.round(Math.random() * 1e9)}`;
      cb(null, `${uniqueSuffix}-${file.originalname}`);
    },
  }),
  limits: {
    fileSize: config.storage.maxUploadSize,
  },
});

// MCU Families configuration
const MCU_FAMILIES: Array<{
  family: MCUFamily;
  displayName: string;
  description: string;
  supportedRtos: string[];
  toolchains: string[];
}> = [
  {
    family: 'stm32',
    displayName: 'STM32',
    description: 'STMicroelectronics ARM Cortex-M and Cortex-A microcontrollers',
    supportedRtos: ['freertos', 'zephyr', 'tirtos', 'none'],
    toolchains: ['gcc-arm', 'armcc', 'iar'],
  },
  {
    family: 'esp32',
    displayName: 'ESP32',
    description: 'Espressif ESP32/ESP32-S/ESP32-C series with WiFi and Bluetooth',
    supportedRtos: ['freertos', 'none'],
    toolchains: ['gcc-xtensa', 'gcc-riscv'],
  },
  {
    family: 'ti_tms320',
    displayName: 'TI TMS320',
    description: 'Texas Instruments DSPs for motor control and signal processing',
    supportedRtos: ['tirtos', 'freertos', 'none'],
    toolchains: ['ti-cgt'],
  },
  {
    family: 'infineon_aurix',
    displayName: 'Infineon AURIX',
    description: 'Infineon AURIX TriCore safety MCUs for automotive applications',
    supportedRtos: ['autosar', 'freertos', 'none'],
    toolchains: ['tasking', 'gcc-tricore', 'hightec'],
  },
  {
    family: 'nordic_nrf',
    displayName: 'Nordic nRF',
    description: 'Nordic Semiconductor BLE and wireless SoCs',
    supportedRtos: ['zephyr', 'freertos', 'none'],
    toolchains: ['gcc-arm'],
  },
  {
    family: 'rpi_pico',
    displayName: 'Raspberry Pi Pico',
    description: 'RP2040-based microcontroller boards',
    supportedRtos: ['freertos', 'none'],
    toolchains: ['gcc-arm'],
  },
  {
    family: 'nxp_imxrt',
    displayName: 'NXP i.MX RT',
    description: 'NXP crossover processors with real-time capabilities',
    supportedRtos: ['freertos', 'zephyr', 'none'],
    toolchains: ['gcc-arm', 'mcuxpresso'],
  },
];

// Layout agents configuration
const LAYOUT_AGENTS = [
  {
    id: 'conservative',
    name: 'Conservative',
    strategy: 'conservative',
    description: 'Prioritizes reliability and manufacturability with generous spacing',
    priority: ['reliability', 'dfm', 'cost'],
    bestFor: ['High-power', 'Industrial', 'Automotive'],
  },
  {
    id: 'aggressive_compact',
    name: 'Aggressive Compact',
    strategy: 'aggressive_compact',
    description: 'Maximizes component density for space-constrained designs',
    priority: ['size', 'cost', 'manufacturability'],
    bestFor: ['Consumer electronics', 'Space-constrained', 'Wearables'],
  },
  {
    id: 'thermal_optimized',
    name: 'Thermal Optimized',
    strategy: 'thermal_optimized',
    description: 'Prioritizes thermal performance with strategic heat spreading',
    priority: ['thermal', 'reliability', 'size'],
    bestFor: ['Power electronics', 'Motor controllers', 'High-power LEDs'],
  },
  {
    id: 'emi_optimized',
    name: 'EMI Optimized',
    strategy: 'emi_optimized',
    description: 'Minimizes electromagnetic interference with careful routing',
    priority: ['signal_integrity', 'emi', 'size'],
    bestFor: ['High-speed digital', 'RF', 'Mixed-signal', 'Sensitive analog'],
  },
  {
    id: 'dfm_optimized',
    name: 'DFM Optimized',
    strategy: 'dfm_optimized',
    description: 'Prioritizes ease of manufacturing and assembly yield',
    priority: ['manufacturability', 'cost', 'size'],
    bestFor: ['High-volume production', 'Cost-sensitive'],
  },
];

// Validation middleware factory
function validate<T>(schema: z.ZodSchema<T>) {
  return (req: Request, _res: Response, next: NextFunction) => {
    try {
      req.body = schema.parse(req.body);
      next();
    } catch (error) {
      if (error instanceof z.ZodError) {
        next(new ValidationError('Invalid request body', {
          operation: 'validation',
          errors: error.errors,
        }));
      } else {
        next(error);
      }
    }
  };
}

// Helper to get default owner ID (in production, this would come from auth)
function getDefaultOwnerId(req: Request): string {
  return (req.headers['x-user-id'] as string) || 'system';
}

// UUID validation regex
const UUID_REGEX = /^[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/i;

// Validate UUID parameter middleware
function validateUUIDParam(paramName: string) {
  return (req: Request, _res: Response, next: NextFunction) => {
    const value = req.params[paramName];
    if (!value || !UUID_REGEX.test(value)) {
      return next(new ValidationError(`Invalid ${paramName}: must be a valid UUID`, {
        operation: 'uuid_validation',
        param: paramName,
        value: value,
      }));
    }
    next();
  };
}

export function createApiRoutes(io: SocketIOServer): Router {
  const router = Router();

  // ============================================================================
  // Configuration Endpoints
  // These are served at both /config/* and /* for backward compatibility
  // ============================================================================

  // Config endpoints at /config/* path (frontend API client expects these)
  router.get('/config/project-types', async (_req: Request, res: Response, next: NextFunction) => {
    try {
      log.debug('Getting all project types (config endpoint)');
      const summaries = getAllProjectTypeSummaries();

      res.json({
        success: true,
        data: summaries,
        metadata: { count: summaries.length },
      });
    } catch (error) {
      next(error);
    }
  });

  router.get('/config/mcu-families', async (_req: Request, res: Response, next: NextFunction) => {
    try {
      log.debug('Getting MCU families (config endpoint)');

      res.json({
        success: true,
        data: MCU_FAMILIES,
        metadata: { count: MCU_FAMILIES.length },
      });
    } catch (error) {
      next(error);
    }
  });

  router.get('/config/simulation-types', async (_req: Request, res: Response, next: NextFunction) => {
    try {
      log.debug('Getting all simulation types (config endpoint)');

      // Collect all unique simulation types from all project configurations
      const allSimulations = new Map<string, { type: SimulationType; displayName: string; description?: string }>();

      for (const configItem of Object.values(PROJECT_TYPE_CONFIGURATIONS)) {
        for (const sim of configItem.simulations) {
          if (!allSimulations.has(sim.type)) {
            allSimulations.set(sim.type, {
              type: sim.type,
              displayName: sim.displayName,
            });
          }
        }
      }

      res.json({
        success: true,
        data: Array.from(allSimulations.values()),
        metadata: { count: allSimulations.size },
      });
    } catch (error) {
      next(error);
    }
  });

  router.get('/config/validation-domains', async (_req: Request, res: Response, next: NextFunction) => {
    try {
      log.debug('Getting all validation domains (config endpoint)');

      // Collect all unique validation types from all project configurations
      const allValidations = new Map<string, { type: string; displayName: string; required: boolean }>();

      for (const configItem of Object.values(PROJECT_TYPE_CONFIGURATIONS)) {
        for (const val of configItem.validations) {
          if (!allValidations.has(val.type)) {
            allValidations.set(val.type, {
              type: val.type,
              displayName: val.displayName,
              required: val.required,
            });
          }
        }
      }

      res.json({
        success: true,
        data: Array.from(allValidations.values()),
        metadata: { count: allValidations.size },
      });
    } catch (error) {
      next(error);
    }
  });

  // Legacy endpoints without /config prefix (keep for backward compatibility)
  router.get('/project-types', async (_req: Request, res: Response, next: NextFunction) => {
    try {
      log.debug('Getting all project types');
      const summaries = getAllProjectTypeSummaries();

      res.json({
        success: true,
        data: summaries,
        metadata: { count: summaries.length },
      });
    } catch (error) {
      next(error);
    }
  });

  router.get('/project-types/:type', async (req: Request, res: Response, next: NextFunction) => {
    try {
      const projectType = req.params.type as ProjectType;
      log.debug('Getting project type configuration', { type: projectType });

      const configuration = getProjectTypeConfiguration(projectType);

      res.json({
        success: true,
        data: configuration,
      });
    } catch (error) {
      if (error instanceof Error && error.message.includes('Unknown project type')) {
        next(new NotFoundError('ProjectType', req.params.type, { operation: 'getProjectTypeConfiguration' }));
      } else {
        next(error);
      }
    }
  });

  router.get('/mcu-families', async (_req: Request, res: Response, next: NextFunction) => {
    try {
      log.debug('Getting MCU families');

      res.json({
        success: true,
        data: MCU_FAMILIES,
        metadata: { count: MCU_FAMILIES.length },
      });
    } catch (error) {
      next(error);
    }
  });

  router.get('/simulation-types', async (_req: Request, res: Response, next: NextFunction) => {
    try {
      log.debug('Getting all simulation types');

      // Collect all unique simulation types from all project configurations
      const allSimulations = new Map<string, { type: SimulationType; displayName: string; description?: string }>();

      for (const config of Object.values(PROJECT_TYPE_CONFIGURATIONS)) {
        for (const sim of config.simulations) {
          if (!allSimulations.has(sim.type)) {
            allSimulations.set(sim.type, {
              type: sim.type,
              displayName: sim.displayName,
            });
          }
        }
      }

      res.json({
        success: true,
        data: Array.from(allSimulations.values()),
        metadata: { count: allSimulations.size },
      });
    } catch (error) {
      next(error);
    }
  });

  router.get('/simulation-types/:projectType', async (req: Request, res: Response, next: NextFunction) => {
    try {
      const projectType = req.params.projectType as ProjectType;
      log.debug('Getting simulation types for project type', { type: projectType });

      const simulations = getEnabledSimulations(projectType);

      res.json({
        success: true,
        data: simulations,
        metadata: { count: simulations.length, projectType },
      });
    } catch (error) {
      if (error instanceof Error && error.message.includes('Unknown project type')) {
        next(new NotFoundError('ProjectType', req.params.projectType, { operation: 'getSimulationTypes' }));
      } else {
        next(error);
      }
    }
  });

  router.get('/validation-domains', async (_req: Request, res: Response, next: NextFunction) => {
    try {
      log.debug('Getting all validation domains');

      // Collect all unique validation types from all project configurations
      const allValidations = new Map<string, { type: string; displayName: string; required: boolean }>();

      for (const config of Object.values(PROJECT_TYPE_CONFIGURATIONS)) {
        for (const val of config.validations) {
          if (!allValidations.has(val.type)) {
            allValidations.set(val.type, {
              type: val.type,
              displayName: val.displayName,
              required: val.required,
            });
          }
        }
      }

      res.json({
        success: true,
        data: Array.from(allValidations.values()),
        metadata: { count: allValidations.size },
      });
    } catch (error) {
      next(error);
    }
  });

  router.get('/validation-domains/:projectType', async (req: Request, res: Response, next: NextFunction) => {
    try {
      const projectType = req.params.projectType as ProjectType;
      log.debug('Getting validation domains for project type', { type: projectType });

      const configuration = getProjectTypeConfiguration(projectType);

      res.json({
        success: true,
        data: configuration.validations,
        metadata: { count: configuration.validations.length, projectType },
      });
    } catch (error) {
      if (error instanceof Error && error.message.includes('Unknown project type')) {
        next(new NotFoundError('ProjectType', req.params.projectType, { operation: 'getValidationDomains' }));
      } else {
        next(error);
      }
    }
  });

  router.get('/layout-agents', async (_req: Request, res: Response, next: NextFunction) => {
    try {
      log.debug('Getting all layout agents');

      res.json({
        success: true,
        data: LAYOUT_AGENTS,
        metadata: { count: LAYOUT_AGENTS.length },
      });
    } catch (error) {
      next(error);
    }
  });

  router.get('/layout-agents/:projectType', async (req: Request, res: Response, next: NextFunction) => {
    try {
      const projectType = req.params.projectType as ProjectType;
      log.debug('Getting layout agents for project type', { type: projectType });

      const agents = getEnabledLayoutAgents(projectType);

      res.json({
        success: true,
        data: agents.map(agent => ({
          id: agent.strategy,
          name: agent.displayName,
          strategy: agent.strategy,
          description: agent.description,
          enabled: agent.enabled,
          priorityWeight: agent.priorityWeight,
          parameters: agent.parameters,
        })),
        metadata: { count: agents.length, projectType },
      });
    } catch (error) {
      if (error instanceof Error && error.message.includes('Unknown project type')) {
        next(new NotFoundError('ProjectType', req.params.projectType, { operation: 'getLayoutAgents' }));
      } else {
        next(error);
      }
    }
  });

  // ============================================================================
  // Project Management
  // ============================================================================

  // Apply UUID validation to all routes with :projectId parameter
  router.param('projectId', (req: Request, _res: Response, next: NextFunction, value: string) => {
    if (!UUID_REGEX.test(value)) {
      return next(new ValidationError(`Invalid projectId: must be a valid UUID (received: ${value})`, {
        operation: 'uuid_validation',
        param: 'projectId',
        value: value,
      }));
    }
    next();
  });

  // Apply UUID validation to schematicId parameter
  router.param('schematicId', (req: Request, _res: Response, next: NextFunction, value: string) => {
    if (!UUID_REGEX.test(value)) {
      return next(new ValidationError(`Invalid schematicId: must be a valid UUID (received: ${value})`, {
        operation: 'uuid_validation',
        param: 'schematicId',
        value: value,
      }));
    }
    next();
  });

  // Apply UUID validation to layoutId parameter
  router.param('layoutId', (req: Request, _res: Response, next: NextFunction, value: string) => {
    if (!UUID_REGEX.test(value)) {
      return next(new ValidationError(`Invalid layoutId: must be a valid UUID (received: ${value})`, {
        operation: 'uuid_validation',
        param: 'layoutId',
        value: value,
      }));
    }
    next();
  });

  // Apply UUID validation to simulationId parameter
  router.param('simulationId', (req: Request, _res: Response, next: NextFunction, value: string) => {
    if (!UUID_REGEX.test(value)) {
      return next(new ValidationError(`Invalid simulationId: must be a valid UUID (received: ${value})`, {
        operation: 'uuid_validation',
        param: 'simulationId',
        value: value,
      }));
    }
    next();
  });

  // Apply UUID validation to firmwareId parameter
  router.param('firmwareId', (req: Request, _res: Response, next: NextFunction, value: string) => {
    if (!UUID_REGEX.test(value)) {
      return next(new ValidationError(`Invalid firmwareId: must be a valid UUID (received: ${value})`, {
        operation: 'uuid_validation',
        param: 'firmwareId',
        value: value,
      }));
    }
    next();
  });

  router.get('/projects', async (req: Request, res: Response, next: NextFunction) => {
    try {
      const filters: ProjectFilters = {};

      // Parse query parameters
      if (req.query.type) filters.type = req.query.type as string;
      if (req.query.status) filters.status = req.query.status as ProjectFilters['status'];
      if (req.query.phase) filters.phase = req.query.phase as ProjectPhase;
      if (req.query.ownerId) filters.ownerId = req.query.ownerId as string;
      if (req.query.organizationId) filters.organizationId = req.query.organizationId as string;

      // Pagination
      const page = parseInt(req.query.page as string, 10) || 1;
      const pageSize = Math.min(parseInt(req.query.pageSize as string, 10) || 20, 100);
      filters.limit = pageSize;
      filters.offset = (page - 1) * pageSize;

      log.debug('Listing projects', { filters, page, pageSize });

      // Create count filters without limit/offset
      const countFilters: ProjectFilters = {};
      if (filters.type) countFilters.type = filters.type;
      if (filters.status) countFilters.status = filters.status;
      if (filters.phase) countFilters.phase = filters.phase;
      if (filters.ownerId) countFilters.ownerId = filters.ownerId;
      if (filters.organizationId) countFilters.organizationId = filters.organizationId;

      const [projects, total] = await Promise.all([
        findAllProjects(filters),
        countProjects(countFilters),
      ]);

      res.json({
        success: true,
        data: {
          data: projects,
          total,
          page,
          pageSize,
          totalPages: Math.ceil(total / pageSize),
        },
      });
    } catch (error) {
      next(error);
    }
  });

  router.post('/projects', validate(z.object({
    name: z.string().min(1).max(255),
    description: z.string().optional(),
    repositoryUrl: z.string().url().optional(),
    projectType: z.nativeEnum(ProjectType).optional(),
    organizationId: z.string().uuid().optional(),
    collaborators: z.array(z.string()).optional(),
    metadata: z.object({
      tags: z.array(z.string()).optional(),
      priority: z.enum(['low', 'medium', 'high', 'critical']).optional(),
    }).optional(),
  })), async (req: Request, res: Response, next: NextFunction) => {
    try {
      const ownerId = getDefaultOwnerId(req);

      log.info('Creating project', { name: req.body.name, ownerId, projectType: req.body.projectType });

      // Get default configuration for project type if specified
      let phaseConfig: Record<string, unknown> | undefined;
      if (req.body.projectType) {
        try {
          const typeConfig = getProjectTypeConfiguration(req.body.projectType);
          phaseConfig = {
            enabledPhases: typeConfig.phases.filter(p => p.enabledByDefault).map(p => p.phase),
            enabledSimulations: typeConfig.simulations.filter(s => s.enabledByDefault).map(s => s.type),
            enabledValidations: typeConfig.validations.filter(v => v.enabledByDefault).map(v => v.type),
          };
        } catch {
          // Use default if project type is invalid
          log.warn('Invalid project type, using defaults', { projectType: req.body.projectType });
        }
      }

      const input: CreateProjectInput = {
        name: req.body.name,
        description: req.body.description,
        repositoryUrl: req.body.repositoryUrl,
        projectType: req.body.projectType || ProjectType.POWER_ELECTRONICS,
        ownerId,
        organizationId: req.body.organizationId,
        collaborators: req.body.collaborators,
        metadata: req.body.metadata,
        phaseConfig,
      };

      const project = await createProject(input);

      // Emit project created event
      io.emit('project:created', { project });

      res.status(201).json({
        success: true,
        data: project,
      });
    } catch (error) {
      next(error);
    }
  });

  router.get('/projects/:projectId', async (req: Request, res: Response, next: NextFunction) => {
    try {
      const projectId = req.params.projectId;
      log.debug('Getting project', { projectId });

      const project = await findProjectById(projectId);

      if (!project) {
        throw new NotFoundError('Project', projectId, { operation: 'getProject' });
      }

      res.json({
        success: true,
        data: project,
      });
    } catch (error) {
      next(error);
    }
  });

  router.put('/projects/:projectId', validate(z.object({
    name: z.string().min(1).max(255).optional(),
    description: z.string().optional(),
    repositoryUrl: z.string().url().optional().nullable(),
    projectType: z.string().optional(),
    collaborators: z.array(z.string()).optional(),
    metadata: z.object({
      tags: z.array(z.string()).optional(),
      priority: z.enum(['low', 'medium', 'high', 'critical']).optional(),
    }).optional(),
  })), async (req: Request, res: Response, next: NextFunction) => {
    try {
      const projectId = req.params.projectId;
      log.info('Updating project', { projectId, fields: Object.keys(req.body) });

      const updateData: UpdateProjectInput = {};

      if (req.body.name !== undefined) updateData.name = req.body.name;
      if (req.body.description !== undefined) updateData.description = req.body.description;
      if (req.body.repositoryUrl !== undefined) updateData.repositoryUrl = req.body.repositoryUrl;
      if (req.body.projectType !== undefined) updateData.projectType = req.body.projectType;
      if (req.body.collaborators !== undefined) updateData.collaborators = req.body.collaborators;
      if (req.body.metadata !== undefined) updateData.metadata = req.body.metadata;

      const project = await updateProject(projectId, updateData);

      // Emit project updated event
      io.to(`project:${projectId}`).emit('project:updated', { project });

      res.json({
        success: true,
        data: project,
      });
    } catch (error) {
      next(error);
    }
  });

  router.delete('/projects/:projectId', async (req: Request, res: Response, next: NextFunction) => {
    try {
      const projectId = req.params.projectId;
      log.info('Deleting project', { projectId });

      await deleteProject(projectId);

      // Emit project deleted event
      io.emit('project:deleted', { projectId });

      res.json({
        success: true,
        data: { id: projectId, deleted: true },
      });
    } catch (error) {
      next(error);
    }
  });

  // ============================================================================
  // File Browser Routes (Virtual File System)
  // ============================================================================

  /**
   * GET /projects/:projectId/files/tree
   * Get virtual file tree for project artifacts
   */
  router.get('/projects/:projectId/files/tree', getFileTreeHandler);

  /**
   * GET /projects/:projectId/files/content?path=<file-path>
   * Get file content from virtual file system
   */
  router.get('/projects/:projectId/files/content', getFileContentHandler);

  router.patch('/projects/:projectId/status', validate(z.object({
    status: z.enum(['draft', 'in_progress', 'review', 'approved', 'completed', 'on_hold', 'cancelled']),
  })), async (req: Request, res: Response, next: NextFunction) => {
    try {
      const projectId = req.params.projectId;
      log.info('Updating project status', { projectId, status: req.body.status });

      const project = await updateProjectStatus(projectId, req.body.status);

      io.to(`project:${projectId}`).emit('project:status-changed', {
        projectId,
        status: req.body.status,
        project,
      });

      res.json({
        success: true,
        data: project,
      });
    } catch (error) {
      next(error);
    }
  });

  router.patch('/projects/:projectId/phase', validate(z.object({
    phase: z.enum([
      'ideation', 'architecture', 'schematic', 'simulation',
      'pcb_layout', 'manufacturing', 'firmware', 'testing',
      'production', 'field_support'
    ]),
  })), async (req: Request, res: Response, next: NextFunction) => {
    try {
      const projectId = req.params.projectId;
      log.info('Updating project phase', { projectId, phase: req.body.phase });

      const project = await updateProjectPhase(projectId, req.body.phase);

      io.to(`project:${projectId}`).emit('project:phase-changed', {
        projectId,
        phase: req.body.phase,
        project,
      });

      res.json({
        success: true,
        data: project,
      });
    } catch (error) {
      next(error);
    }
  });

  router.get('/projects/:projectId/phases', async (req: Request, res: Response, next: NextFunction) => {
    try {
      const projectId = req.params.projectId;
      log.debug('Getting project phases', { projectId });

      const phases = await getProjectPhases(projectId);

      res.json({
        success: true,
        data: phases,
        metadata: { count: phases.length, projectId },
      });
    } catch (error) {
      next(error);
    }
  });

  // ============================================================================
  // Phase 1: Ideation & Research
  // ============================================================================

  router.post('/projects/:projectId/requirements', validate(z.object({
    description: z.string().min(10),
    constraints: z.array(z.string()).optional(),
    targetSpecs: z.record(z.unknown()).optional(),
  })), async (req: Request, res: Response, next: NextFunction) => {
    try {
      const projectId = req.params.projectId;
      log.info('Generating requirements', { projectId });

      // Verify project exists
      const project = await findProjectById(projectId);
      if (!project) {
        throw new NotFoundError('Project', projectId, { operation: 'generateRequirements' });
      }

      // Queue requirements generation job
      const jobId = crypto.randomUUID();

      io.to(`project:${projectId}`).emit('requirements:started', {
        jobId,
        projectId,
        description: req.body.description,
      });

      res.status(202).json({
        success: true,
        data: {
          jobId,
          projectId,
          status: 'pending',
          description: req.body.description,
        },
      });
    } catch (error) {
      next(error);
    }
  });

  router.post('/projects/:projectId/patent-search', validate(z.object({
    query: z.string().min(5),
    jurisdictions: z.array(z.enum(['USPTO', 'EPO', 'WIPO', 'JPO'])).optional(),
  })), async (req: Request, res: Response, next: NextFunction) => {
    try {
      const projectId = req.params.projectId;
      log.info('Searching patents', { projectId, query: req.body.query });

      // Verify project exists
      const project = await findProjectById(projectId);
      if (!project) {
        throw new NotFoundError('Project', projectId, { operation: 'patentSearch' });
      }

      // Queue patent search job
      const jobId = crypto.randomUUID();

      io.to(`project:${projectId}`).emit('patent-search:started', {
        jobId,
        projectId,
        query: req.body.query,
        jurisdictions: req.body.jurisdictions || ['USPTO', 'EPO', 'WIPO'],
      });

      res.status(202).json({
        success: true,
        data: {
          jobId,
          projectId,
          status: 'pending',
          query: req.body.query,
        },
      });
    } catch (error) {
      next(error);
    }
  });

  // ============================================================================
  // Phase 2: Architecture & Specification
  // ============================================================================

  router.post('/projects/:projectId/architecture', validate(z.object({
    requirements: z.array(z.string()),
    targetMcu: z.string().optional(),
    powerBudget: z.number().optional(),
  })), async (req: Request, res: Response, next: NextFunction) => {
    try {
      const projectId = req.params.projectId;
      log.info('Generating architecture', { projectId });

      // Verify project exists
      const project = await findProjectById(projectId);
      if (!project) {
        throw new NotFoundError('Project', projectId, { operation: 'generateArchitecture' });
      }

      const jobId = crypto.randomUUID();

      io.to(`project:${projectId}`).emit('architecture:started', {
        jobId,
        projectId,
        requirements: req.body.requirements,
      });

      res.status(202).json({
        success: true,
        data: {
          jobId,
          projectId,
          status: 'pending',
        },
      });
    } catch (error) {
      next(error);
    }
  });

  router.post('/projects/:projectId/component-selection', validate(z.object({
    category: z.string(),
    specifications: z.record(z.unknown()),
    preferredVendors: z.array(z.string()).optional(),
    maxCost: z.number().optional(),
  })), async (req: Request, res: Response, next: NextFunction) => {
    try {
      const projectId = req.params.projectId;
      log.info('Selecting components', { projectId, category: req.body.category });

      // Verify project exists
      const project = await findProjectById(projectId);
      if (!project) {
        throw new NotFoundError('Project', projectId, { operation: 'componentSelection' });
      }

      const jobId = crypto.randomUUID();

      io.to(`project:${projectId}`).emit('component-selection:started', {
        jobId,
        projectId,
        category: req.body.category,
      });

      res.status(202).json({
        success: true,
        data: {
          jobId,
          projectId,
          status: 'pending',
          category: req.body.category,
        },
      });
    } catch (error) {
      next(error);
    }
  });

  router.get('/projects/:projectId/bom', async (req: Request, res: Response, next: NextFunction) => {
    try {
      const projectId = req.params.projectId;
      log.debug('Getting BOM', { projectId });

      // Verify project exists
      const project = await findProjectById(projectId);
      if (!project) {
        throw new NotFoundError('Project', projectId, { operation: 'getBom' });
      }

      // Get latest schematic to extract BOM
      const schematics = await findSchematicsByProject(projectId);

      if (schematics.length === 0) {
        res.json({
          success: true,
          data: {
            components: [],
            totalCost: 0,
            currency: 'USD',
            message: 'No schematics found for this project',
          },
        });
        return;
      }

      // Extract components from latest schematic
      const latestSchematic = schematics[0];
      const components = latestSchematic.components || [];

      res.json({
        success: true,
        data: {
          components,
          totalCost: 0, // Would be calculated from pricing data
          currency: 'USD',
          schematicId: latestSchematic.id,
          schematicVersion: latestSchematic.version,
        },
      });
    } catch (error) {
      next(error);
    }
  });

  // ============================================================================
  // Phase 3: Schematic Capture
  // ============================================================================

  router.post('/projects/:projectId/schematic/generate', validate(z.object({
    architecture: z.record(z.unknown()),
    components: z.array(z.unknown()),
    name: z.string().optional(),
  })), async (req: Request, res: Response, next: NextFunction) => {
    try {
      const projectId = req.params.projectId;
      log.info('Generating schematic', { projectId });

      // Verify project exists
      const project = await findProjectById(projectId);
      if (!project) {
        throw new NotFoundError('Project', projectId, { operation: 'generateSchematic' });
      }

      // Emit generation started event
      io.to(`project:${projectId}`).emit('schematic:generation-started', {
        projectId,
        status: 'generating',
      });

      // Extract architecture data from request
      const architectureData = req.body.architecture || {};
      const subsystems = Array.isArray(architectureData.subsystems)
        ? architectureData.subsystems.map((s: Record<string, unknown>) => ({
            id: String(s.id || ''),
            name: String(s.name || 'Unnamed'),
            category: String(s.category || 'Default'),
            description: s.description ? String(s.description) : undefined,
          }))
        : [];

      // Generate KiCad schematic content using MAPO pipeline (NO PLACEHOLDER FALLBACK)
      interface GeneratedSchematic {
        content: string;
        sheets: Array<{ name: string; uuid: string; page: number }>;
        components: Array<{ reference: string; value: string; uuid: string }>;
        nets: Array<{ name: string; uuid: string; code: number }>;
      }
      let generatedSchematic: GeneratedSchematic;
      let operationId: string | undefined;

      if (subsystems.length > 0) {
        // Use Python MAPO pipeline for real schematic generation
        log.info('Using MAPO pipeline for schematic generation', {
          projectId,
          subsystemCount: subsystems.length,
        });

        const pythonExecutor = new PythonExecutor();
        await pythonExecutor.initialize();

        // Create operation ID for WebSocket streaming
        const schematicWsManager = getSchematicWsManager();
        operationId = schematicWsManager.createOperation(projectId);
        log.info('Created schematic operation for streaming', { operationId, projectId });

        // Prepare input for MAPO pipeline with operation ID for progress streaming
        const mapoInput = {
          subsystems,
          project_name: req.body.name || project.name,
          design_name: `schematic_${Date.now()}`,
          skip_validation: true, // Skip validation for faster initial generation
          operation_id: operationId, // For WebSocket progress streaming
          project_id: projectId,
        };

        // Emit progress event (legacy)
        io.to(`project:${projectId}`).emit('schematic:progress', {
          projectId,
          status: 'fetching_symbols',
          message: 'Fetching real KiCad symbols from libraries...',
          operationId, // Include operation ID so frontend can subscribe
        });

        try {
          // Use executeWithProgress to relay PROGRESS: events via WebSocket
          const result = await pythonExecutor.executeWithProgress(
            'api_generate_schematic.py',
            ['--json', JSON.stringify(mapoInput)],
            {
              timeout: 600000, // 10 minute timeout - schematic gen is LLM-heavy
              onProgress: (event: ProgressEvent) => {
                // Relay progress event to WebSocket subscribers
                // Cast to SchematicProgressEvent (types are compatible at runtime)
                schematicWsManager.emitProgress(operationId, event as import('./websocket-schematic.js').SchematicProgressEvent);

                // Also emit to legacy project room for backwards compatibility
                io.to(`project:${projectId}`).emit('schematic:progress', {
                  projectId,
                  operationId,
                  ...event,
                });
              },
            }
          );

          if (!result.success) {
            // NO FALLBACK - Return verbose error message
            log.error(
              'MAPO pipeline failed - NO FALLBACK (placeholder generator removed)',
              new Error(result.stderr || 'Unknown pipeline error'),
              { projectId, stdout: result.stdout }
            );

            // Emit detailed error to client
            io.to(`project:${projectId}`).emit('schematic:error', {
              projectId,
              status: 'failed',
              message: 'Schematic generation failed',
              details: {
                error: result.stderr || 'Unknown error',
                action: 'Check server logs for full error details. Ensure OPENROUTER_API_KEY is set.',
              },
            });

            // Return detailed error response
            return res.status(500).json({
              error: 'Schematic generation failed',
              message: 'MAPO pipeline failed. Placeholder generation has been removed.',
              details: {
                stderr: result.stderr,
                stdout: result.stdout,
                action: 'Check that OPENROUTER_API_KEY is configured correctly in k8s/secrets.yaml',
                documentation: 'https://openrouter.ai/keys',
              },
            });
          }

          // MAPO succeeded - process the result
          {
            // Parse MAPO pipeline output
            const mapoResult = result.output as {
              success: boolean;
              schematic_content: string;
              sheets?: Array<{ name: string; uuid: string; component_count: number }>;
              component_count: number;
              errors: string[];
            };

            if (mapoResult.success && mapoResult.schematic_content) {
              // Parse the generated KiCad schematic to extract component/net metadata
              const parsed = parseKicadSchematic(mapoResult.schematic_content);

              generatedSchematic = {
                content: mapoResult.schematic_content,
                sheets: parsed.sheets.length > 0
                  ? parsed.sheets
                  : (mapoResult.sheets || []).map((s, i) => ({
                      name: s.name,
                      uuid: s.uuid,
                      page: i + 1,
                    })),
                components: parsed.components,
                nets: parsed.nets,
              };

              // Emit success with real component count
              io.to(`project:${projectId}`).emit('schematic:progress', {
                projectId,
                status: 'assembled',
                message: `Assembled schematic with ${mapoResult.component_count} real components`,
                componentCount: mapoResult.component_count,
              });

              log.info('MAPO pipeline generated schematic successfully', {
                projectId,
                componentCount: mapoResult.component_count,
                parsedComponents: parsed.components.length,
                parsedNets: parsed.nets.length,
                contentLength: mapoResult.schematic_content.length,
              });
            } else {
              // MAPO returned success=false, use fallback
              throw new Error(mapoResult.errors?.join(', ') || 'MAPO generation failed');
            }
          }
        } catch (mapoError) {
          // NO FALLBACK - Return verbose error to client
          const errorMessage = mapoError instanceof Error ? mapoError.message : 'Unknown error';

          log.error(
            'MAPO pipeline error - NO FALLBACK (placeholder generator removed)',
            mapoError instanceof Error ? mapoError : new Error(String(mapoError)),
            { projectId }
          );

          // Emit detailed error to client
          io.to(`project:${projectId}`).emit('schematic:error', {
            projectId,
            status: 'failed',
            message: 'Schematic generation failed',
            details: {
              error: errorMessage,
              action: 'Check server logs for full error details. Ensure OPENROUTER_API_KEY is set.',
            },
          });

          // Return detailed error response
          return res.status(500).json({
            error: 'Schematic generation failed',
            message: `MAPO pipeline error: ${errorMessage}`,
            details: {
              error: errorMessage,
              action: 'Check that OPENROUTER_API_KEY is configured correctly in k8s/secrets.yaml',
              documentation: 'https://openrouter.ai/keys',
            },
          });
        }
      } else {
        // Generate minimal schematic if no subsystems specified
        generatedSchematic = generateMinimalSchematic(req.body.name || project.name);
      }

      log.info('KiCad schematic content generated', {
        projectId,
        contentLength: generatedSchematic.content.length,
        componentCount: generatedSchematic.components?.length || 0,
        netCount: generatedSchematic.nets?.length || 0,
      });

      // Create schematic record with generated content
      // Transform parsed metadata to match database schema types
      const dbSheets: import('../types/index.js').SchematicSheet[] = generatedSchematic.sheets.map((s) => ({
        id: s.uuid,
        name: s.name,
        pageNumber: s.page,
        components: generatedSchematic.components.map((c) => c.uuid),
        nets: generatedSchematic.nets.map((n) => n.uuid),
      }));

      const dbComponents: import('../types/index.js').Component[] = generatedSchematic.components.map((c) => ({
        id: c.uuid,
        reference: c.reference,
        value: c.value,
        footprint: '',
        position: { x: 0, y: 0 },
        rotation: 0,
        properties: { library: (c as any).library || '', symbol: (c as any).symbol || '' },
        pins: [],
      }));

      const dbNets: import('../types/index.js').Net[] = generatedSchematic.nets.map((n) => ({
        id: n.uuid,
        name: n.name,
        connections: [],
        properties: {},
      }));

      const schematicInput: CreateSchematicInput = {
        projectId,
        name: req.body.name || `${project.name} Schematic`,
        format: 'kicad_sch',
        kicadSch: generatedSchematic.content,
        sheets: dbSheets,
        components: dbComponents,
        nets: dbNets,
      };

      const schematic = await createSchematic(schematicInput);

      // Emit generation completed event
      io.to(`project:${projectId}`).emit('schematic:generation-completed', {
        schematicId: schematic.id,
        projectId,
        componentCount: generatedSchematic.components.length,
        netCount: generatedSchematic.nets.length,
      });

      log.info('Schematic created successfully', {
        schematicId: schematic.id,
        projectId,
      });

      res.status(201).json({
        success: true,
        data: {
          schematicId: schematic.id,
          status: 'completed',
          projectId,
          operationId, // WebSocket operation ID for streaming (undefined if no streaming)
          componentCount: generatedSchematic.components.length,
          netCount: generatedSchematic.nets.length,
          sheetCount: generatedSchematic.sheets.length,
        },
      });
    } catch (error) {
      log.error(
        'Schematic generation failed',
        error instanceof Error ? error : new Error(String(error)),
        { projectId: req.params.projectId }
      );
      next(error);
    }
  });

  router.post('/projects/:projectId/schematic/upload',
    upload.single('schematic'),
    async (req: Request, res: Response, next: NextFunction) => {
      try {
        const projectId = req.params.projectId;

        if (!req.file) {
          throw new ValidationError('No schematic file provided', { operation: 'schematic-upload' });
        }

        log.info('Uploading schematic', {
          projectId,
          filename: req.file.originalname,
          size: req.file.size,
        });

        // Verify project exists
        const project = await findProjectById(projectId);
        if (!project) {
          throw new NotFoundError('Project', projectId, { operation: 'uploadSchematic' });
        }

        // Read file content
        const filePath = req.file.path;
        const kicadContent = await fs.readFile(filePath, 'utf-8');

        // Create schematic record
        const schematicInput: CreateSchematicInput = {
          projectId,
          name: path.basename(req.file.originalname, path.extname(req.file.originalname)),
          format: 'kicad_sch',
          filePath: req.file.path,
          kicadSch: kicadContent,
        };

        const schematic = await createSchematic(schematicInput);

        // Clean up temp file
        await fs.unlink(filePath).catch(() => {});

        io.to(`project:${projectId}`).emit('schematic:uploaded', {
          schematicId: schematic.id,
          projectId,
          filename: req.file.originalname,
        });

        res.json({
          success: true,
          data: {
            schematicId: schematic.id,
            filename: req.file.originalname,
            status: 'processing',
            projectId,
          },
        });
      } catch (error) {
        next(error);
      }
    }
  );

  router.get('/projects/:projectId/schematics', async (req: Request, res: Response, next: NextFunction) => {
    try {
      const projectId = req.params.projectId;
      log.debug('Getting project schematics', { projectId });

      // Verify project exists
      const project = await findProjectById(projectId);
      if (!project) {
        throw new NotFoundError('Project', projectId, { operation: 'getSchematics' });
      }

      const schematics = await findSchematicsByProject(projectId);

      res.json({
        success: true,
        data: schematics,
        metadata: { count: schematics.length, projectId },
      });
    } catch (error) {
      next(error);
    }
  });

  /**
   * Get the latest schematic for a project
   * Returns the schematic URL for KiCanvas rendering
   * IMPORTANT: This route MUST come before /schematic/:schematicId to avoid "latest" being matched as an ID
   */
  router.get('/projects/:projectId/schematic/latest', async (req: Request, res: Response, next: NextFunction) => {
    try {
      const projectId = req.params.projectId;
      log.debug('Getting latest schematic', { projectId });

      // Verify project exists
      const project = await findProjectById(projectId);
      if (!project) {
        throw new NotFoundError('Project', projectId, { operation: 'getLatestSchematic' });
      }

      const schematics = await findSchematicsByProject(projectId);

      if (!schematics.length) {
        // No schematic yet - return null URL (frontend will show "Generate" message)
        return res.json({
          success: true,
          data: { schematicUrl: null },
          metadata: { projectId, message: 'No schematic generated yet' },
        });
      }

      // Sort by updated_at descending to get the most recent
      const sorted = schematics.sort((a, b) =>
        new Date(b.updatedAt || b.createdAt).getTime() - new Date(a.updatedAt || a.createdAt).getTime()
      );
      const latest = sorted[0];

      // Build the URL for KiCanvas to fetch the raw schematic file
      // Using the .kicad_sch extension route for proper file type detection
      // Use X-Forwarded headers if behind a proxy, otherwise fall back to request values
      const protocol = req.get('X-Forwarded-Proto') || req.protocol || 'https';
      const host = req.get('X-Forwarded-Host') || req.get('host') || 'api.adverant.ai';
      // The service is mounted at /ee-design prefix on the ingress
      const basePath = '/ee-design';
      const schematicUrl = `${protocol}://${host}${basePath}/api/v1/projects/${projectId}/schematic/${latest.id}/schematic.kicad_sch`;

      res.json({
        success: true,
        data: {
          schematicId: latest.id,
          schematicUrl,
          name: latest.name,
          version: latest.version,
          componentCount: latest.components?.length || 0,
          netCount: latest.nets?.length || 0,
          updatedAt: latest.updatedAt || latest.createdAt,
        },
        metadata: { projectId },
      });
    } catch (error) {
      next(error);
    }
  });

  router.get('/projects/:projectId/schematic/:schematicId', async (req: Request, res: Response, next: NextFunction) => {
    try {
      const { projectId, schematicId } = req.params;
      log.debug('Getting schematic', { projectId, schematicId });

      const schematic = await findSchematicById(schematicId);

      if (!schematic) {
        throw new NotFoundError('Schematic', schematicId, { operation: 'getSchematic' });
      }

      if (schematic.projectId !== projectId) {
        throw new ValidationError('Schematic does not belong to this project', {
          operation: 'getSchematic',
          schematicId,
          projectId,
        });
      }

      res.json({
        success: true,
        data: schematic,
      });
    } catch (error) {
      next(error);
    }
  });

  /**
   * Get schematic file content for KiCanvas viewer
   * Returns the raw KiCad schematic file with proper content-type
   *
   * Two routes available:
   * - /projects/:projectId/schematic/:schematicId/file (original)
   * - /projects/:projectId/schematic/:schematicId/schematic.kicad_sch (for KiCanvas file type detection)
   */
  const schematicFileHandler = async (req: Request, res: Response, next: NextFunction) => {
    try {
      const { projectId, schematicId } = req.params;
      log.debug('Getting schematic file', { projectId, schematicId });

      const schematic = await findSchematicById(schematicId);

      if (!schematic) {
        throw new NotFoundError('Schematic', schematicId, { operation: 'getSchematicFile' });
      }

      if (schematic.projectId !== projectId) {
        throw new ValidationError('Schematic does not belong to this project', {
          operation: 'getSchematicFile',
          schematicId,
          projectId,
        });
      }

      // Get the kicadSch content from the schematic record
      // Note: kicadSch is stored in the database, not on disk
      if (!schematic.kicadSch) {
        throw new NotFoundError('Schematic file content', schematicId, {
          operation: 'getSchematicFile',
          message: 'Schematic file content not available. The schematic may still be generating.',
        });
      }

      // Set appropriate headers for KiCad schematic file
      res.setHeader('Content-Type', 'application/x-kicad-schematic');
      res.setHeader('Content-Disposition', `inline; filename="${schematic.name}.kicad_sch"`);
      res.setHeader('Cache-Control', 'no-cache, no-store, must-revalidate');

      // Send the schematic content
      res.send(schematic.kicadSch);
    } catch (error) {
      next(error);
    }
  };

  // Original route (for backwards compatibility)
  router.get('/projects/:projectId/schematic/:schematicId/file', schematicFileHandler);

  // New route with .kicad_sch extension for KiCanvas file type detection
  // KiCanvas uses the URL basename to detect file type, so including .kicad_sch in the URL is essential
  router.get('/projects/:projectId/schematic/:schematicId/schematic.kicad_sch', schematicFileHandler);

  router.post('/projects/:projectId/schematic/:schematicId/validate', async (req: Request, res: Response, next: NextFunction) => {
    try {
      const { projectId, schematicId } = req.params;
      log.info('Validating schematic', { projectId, schematicId });

      const schematic = await findSchematicById(schematicId);

      if (!schematic) {
        throw new NotFoundError('Schematic', schematicId, { operation: 'validateSchematic' });
      }

      if (schematic.projectId !== projectId) {
        throw new ValidationError('Schematic does not belong to this project', {
          operation: 'validateSchematic',
          schematicId,
          projectId,
        });
      }

      // Queue ERC validation job
      const jobId = crypto.randomUUID();

      io.to(`project:${projectId}`).emit('schematic:validation-started', {
        jobId,
        schematicId,
        projectId,
      });

      res.status(202).json({
        success: true,
        data: {
          jobId,
          schematicId,
          projectId,
          status: 'pending',
        },
      });
    } catch (error) {
      next(error);
    }
  });

  router.delete('/projects/:projectId/schematic/:schematicId', async (req: Request, res: Response, next: NextFunction) => {
    try {
      const { projectId, schematicId } = req.params;
      log.info('Deleting schematic', { projectId, schematicId });

      const schematic = await findSchematicById(schematicId);

      if (!schematic) {
        throw new NotFoundError('Schematic', schematicId, { operation: 'deleteSchematic' });
      }

      if (schematic.projectId !== projectId) {
        throw new ValidationError('Schematic does not belong to this project', {
          operation: 'deleteSchematic',
          schematicId,
          projectId,
        });
      }

      await deleteSchematic(schematicId);

      io.to(`project:${projectId}`).emit('schematic:deleted', {
        schematicId,
        projectId,
      });

      res.json({
        success: true,
        data: { id: schematicId, deleted: true },
      });
    } catch (error) {
      next(error);
    }
  });

  // ============================================================================
  // Phase 4: Simulation Suite
  // ============================================================================

  router.post('/projects/:projectId/simulation/:type', validate(z.object({
    name: z.string().optional(),
    schematicId: z.string().uuid().optional(),
    pcbLayoutId: z.string().uuid().optional(),
    analysisType: z.string().optional(),
    parameters: z.record(z.unknown()).optional(),
    priority: z.number().min(1).max(10).optional(),
    timeoutMs: z.number().optional(),
  })), async (req: Request, res: Response, next: NextFunction) => {
    try {
      const { projectId, type } = req.params;
      const simulationType = type as SimulationType;

      log.info('Creating simulation', { projectId, type: simulationType });

      // Verify project exists
      const project = await findProjectById(projectId);
      if (!project) {
        throw new NotFoundError('Project', projectId, { operation: 'createSimulation' });
      }

      // Create simulation record
      const simulationInput: CreateSimulationInput = {
        projectId,
        name: req.body.name || `${simulationType} Simulation`,
        simulationType,
        schematicId: req.body.schematicId,
        pcbLayoutId: req.body.pcbLayoutId,
        parameters: req.body.parameters,
        priority: req.body.priority,
        timeoutMs: req.body.timeoutMs,
      };

      const simulation = await createSimulation(simulationInput);

      // Emit simulation started event
      io.to(`project:${projectId}`).emit('simulation:started', {
        simulationId: simulation.id,
        type: simulationType,
        projectId,
      });

      res.status(202).json({
        success: true,
        data: {
          simulationId: simulation.id,
          status: 'pending',
          type: simulationType,
          projectId,
        },
      });
    } catch (error) {
      next(error);
    }
  });

  router.get('/projects/:projectId/simulations', async (req: Request, res: Response, next: NextFunction) => {
    try {
      const projectId = req.params.projectId;
      log.debug('Getting project simulations', { projectId });

      // Verify project exists
      const project = await findProjectById(projectId);
      if (!project) {
        throw new NotFoundError('Project', projectId, { operation: 'getSimulations' });
      }

      // Build filters from query params
      const filters: SimulationFilters = {};
      if (req.query.type) filters.type = req.query.type as SimulationType;
      if (req.query.status) filters.status = req.query.status as SimulationFilters['status'];
      if (req.query.schematicId) filters.schematicId = req.query.schematicId as string;
      if (req.query.pcbLayoutId) filters.pcbLayoutId = req.query.pcbLayoutId as string;

      const simulations = await findSimulationsByProject(projectId, filters);

      res.json({
        success: true,
        data: simulations,
        metadata: { count: simulations.length, projectId },
      });
    } catch (error) {
      next(error);
    }
  });

  router.get('/projects/:projectId/simulation/:simulationId', async (req: Request, res: Response, next: NextFunction) => {
    try {
      const { projectId, simulationId } = req.params;
      log.debug('Getting simulation', { projectId, simulationId });

      const simulation = await findSimulationById(simulationId);

      if (!simulation) {
        throw new NotFoundError('Simulation', simulationId, { operation: 'getSimulation' });
      }

      if (simulation.projectId !== projectId) {
        throw new ValidationError('Simulation does not belong to this project', {
          operation: 'getSimulation',
          simulationId,
          projectId,
        });
      }

      res.json({
        success: true,
        data: simulation,
      });
    } catch (error) {
      next(error);
    }
  });

  router.post('/projects/:projectId/simulation/:simulationId/cancel', async (req: Request, res: Response, next: NextFunction) => {
    try {
      const { projectId, simulationId } = req.params;
      log.info('Cancelling simulation', { projectId, simulationId });

      const simulation = await findSimulationById(simulationId);

      if (!simulation) {
        throw new NotFoundError('Simulation', simulationId, { operation: 'cancelSimulation' });
      }

      if (simulation.projectId !== projectId) {
        throw new ValidationError('Simulation does not belong to this project', {
          operation: 'cancelSimulation',
          simulationId,
          projectId,
        });
      }

      const updated = await updateSimulationStatus(simulationId, 'cancelled');

      io.to(`project:${projectId}`).emit('simulation:cancelled', {
        simulationId,
        projectId,
      });

      res.json({
        success: true,
        data: updated,
      });
    } catch (error) {
      next(error);
    }
  });

  // ============================================================================
  // Phase 5: PCB Layout
  // ============================================================================

  router.post('/projects/:projectId/pcb-layout/generate', validate(z.object({
    schematicId: z.string().uuid(),
    name: z.string().optional(),
    boardConstraints: z.object({
      width: z.number().positive(),
      height: z.number().positive(),
      layers: z.number().min(2).max(32),
    }),
    agents: z.array(z.enum([
      'conservative',
      'aggressive_compact',
      'thermal_optimized',
      'emi_optimized',
      'dfm_optimized',
    ])).optional(),
    maxIterations: z.number().optional(),
    targetScore: z.number().optional(),
  })), async (req: Request, res: Response, next: NextFunction) => {
    try {
      const projectId = req.params.projectId;
      log.info('Starting PCB layout generation', { projectId, schematicId: req.body.schematicId });

      // Verify project exists
      const project = await findProjectById(projectId);
      if (!project) {
        throw new NotFoundError('Project', projectId, { operation: 'generatePcbLayout' });
      }

      // Verify schematic exists
      const schematic = await findSchematicById(req.body.schematicId);
      if (!schematic) {
        throw new NotFoundError('Schematic', req.body.schematicId, { operation: 'generatePcbLayout' });
      }

      // Create PCB layout record
      const layoutInput: CreatePCBLayoutInput = {
        projectId,
        schematicId: req.body.schematicId,
        name: req.body.name || `${project.name} PCB`,
        layerCount: req.body.boardConstraints.layers,
        boardOutline: {
          width: req.body.boardConstraints.width,
          height: req.body.boardConstraints.height,
          shape: 'rectangular',
          keepoutZones: [],
        },
      };

      const layout = await createPCBLayout(layoutInput);

      // Store MAPOS configuration
      const maposConfig = {
        agents: req.body.agents || config.layout.enabledAgents,
        maxIterations: req.body.maxIterations || config.layout.maxIterations,
        targetScore: req.body.targetScore || config.layout.targetScore,
        boardConstraints: req.body.boardConstraints,
      };

      await updateMaposConfig(layout.id, maposConfig);

      // Emit layout generation started
      io.to(`project:${projectId}`).emit('layout:started', {
        layoutId: layout.id,
        projectId,
        ...maposConfig,
      });

      res.status(202).json({
        success: true,
        data: {
          layoutId: layout.id,
          status: 'pending',
          projectId,
          maxIterations: maposConfig.maxIterations,
          targetScore: maposConfig.targetScore,
        },
      });
    } catch (error) {
      next(error);
    }
  });

  router.get('/projects/:projectId/pcb-layouts', async (req: Request, res: Response, next: NextFunction) => {
    try {
      const projectId = req.params.projectId;
      log.debug('Getting project PCB layouts', { projectId });

      // Verify project exists
      const project = await findProjectById(projectId);
      if (!project) {
        throw new NotFoundError('Project', projectId, { operation: 'getPcbLayouts' });
      }

      const layouts = await findPCBLayoutsByProject(projectId);

      res.json({
        success: true,
        data: layouts,
        metadata: { count: layouts.length, projectId },
      });
    } catch (error) {
      next(error);
    }
  });

  router.get('/projects/:projectId/pcb-layout/:layoutId', async (req: Request, res: Response, next: NextFunction) => {
    try {
      const { projectId, layoutId } = req.params;
      log.debug('Getting PCB layout', { projectId, layoutId });

      const layout = await findPCBLayoutById(layoutId);

      if (!layout) {
        throw new NotFoundError('PCB Layout', layoutId, { operation: 'getPcbLayout' });
      }

      if (layout.projectId !== projectId) {
        throw new ValidationError('PCB Layout does not belong to this project', {
          operation: 'getPcbLayout',
          layoutId,
          projectId,
        });
      }

      // Get MAPOS iteration history
      const iterations = await getMaposIterations(layoutId);

      res.json({
        success: true,
        data: {
          ...layout,
          maposIterations: iterations,
        },
      });
    } catch (error) {
      next(error);
    }
  });

  /**
   * Get PCB file content for KiCanvas viewer
   * Returns the raw KiCad PCB file with proper content-type
   *
   * Two routes available:
   * - /projects/:projectId/pcb-layout/:layoutId/file (original)
   * - /projects/:projectId/pcb-layout/:layoutId/board.kicad_pcb (for KiCanvas file type detection)
   */
  const pcbFileHandler = async (req: Request, res: Response, next: NextFunction) => {
    try {
      const { projectId, layoutId } = req.params;
      log.debug('Getting PCB file', { projectId, layoutId });

      const layout = await findPCBLayoutById(layoutId);

      if (!layout) {
        throw new NotFoundError('PCB Layout', layoutId, { operation: 'getPcbFile' });
      }

      if (layout.projectId !== projectId) {
        throw new ValidationError('PCB Layout does not belong to this project', {
          operation: 'getPcbFile',
          layoutId,
          projectId,
        });
      }

      // Get the kicadPcb content from the layout record
      if (!layout.kicadPcb) {
        throw new NotFoundError('PCB file content', layoutId, {
          operation: 'getPcbFile',
          message: 'PCB file content not available. The PCB may still be generating.',
        });
      }

      // Set appropriate headers for KiCad PCB file
      res.setHeader('Content-Type', 'application/x-kicad-pcb');
      res.setHeader('Content-Disposition', `inline; filename="${layout.name}.kicad_pcb"`);
      res.setHeader('Cache-Control', 'no-cache, no-store, must-revalidate');

      // Send the PCB content
      res.send(layout.kicadPcb);
    } catch (error) {
      next(error);
    }
  };

  // Original route (for backwards compatibility)
  router.get('/projects/:projectId/pcb-layout/:layoutId/file', pcbFileHandler);

  // New route with .kicad_pcb extension for KiCanvas file type detection
  router.get('/projects/:projectId/pcb-layout/:layoutId/board.kicad_pcb', pcbFileHandler);

  /**
   * Export PCB as 3D STEP model
   * Uses KiCad CLI to generate STEP file for CAD/3D viewing
   */
  router.get('/projects/:projectId/pcb-layout/:layoutId/board.step', async (req: Request, res: Response, next: NextFunction) => {
    try {
      const { projectId, layoutId } = req.params;
      log.info('Exporting PCB as STEP', { projectId, layoutId });

      const layout = await findPCBLayoutById(layoutId);
      if (!layout) {
        throw new NotFoundError('PCB Layout', layoutId, { operation: 'exportStep' });
      }
      if (layout.projectId !== projectId) {
        throw new ValidationError('PCB Layout does not belong to this project', {
          operation: 'exportStep', layoutId, projectId,
        });
      }

      const pcbContent = await getKicadContent(layoutId);
      if (!pcbContent) {
        throw new NotFoundError('PCB content', layoutId, { operation: 'exportStep' });
      }

      // Write PCB to temp file
      const fs = await import('fs/promises');
      const path = await import('path');
      const os = await import('os');
      const tempDir = await fs.mkdtemp(path.join(os.tmpdir(), 'step-export-'));
      const pcbPath = path.join(tempDir, 'board.kicad_pcb');
      const stepPath = path.join(tempDir, 'board.step');
      await fs.writeFile(pcbPath, pcbContent);

      // Run export script
      const { spawn } = await import('child_process');
      const scriptsDir = config.kicad.scriptsDir;
      const pythonPath = config.kicad.pythonPath;
      const scriptPath = path.join(scriptsDir, 'export_3d_model.py');

      const result = await new Promise<{ success: boolean; filePath?: string; message?: string }>((resolve) => {
        const proc = spawn(pythonPath, [scriptPath, pcbPath, '--format', 'step', '--output', stepPath]);
        let stdout = '';
        proc.stdout.on('data', (data) => { stdout += data.toString(); });
        proc.on('close', () => {
          try {
            resolve(JSON.parse(stdout));
          } catch {
            resolve({ success: false, message: stdout });
          }
        });
        proc.on('error', () => resolve({ success: false, message: 'Process error' }));
      });

      if (!result.success || !result.filePath) {
        await fs.rm(tempDir, { recursive: true, force: true }).catch(() => {});
        throw new Error(`STEP export failed: ${result.message || 'Unknown error'}`);
      }

      // Send STEP file
      const stepContent = await fs.readFile(result.filePath);
      await fs.rm(tempDir, { recursive: true, force: true }).catch(() => {});

      res.setHeader('Content-Type', 'application/step');
      res.setHeader('Content-Disposition', `attachment; filename="board-${layoutId.substring(0,8)}.step"`);
      res.setHeader('Cache-Control', 'no-cache, no-store, must-revalidate');
      res.send(stepContent);
    } catch (error) {
      next(error);
    }
  });

  /**
   * Export PCB as 3D VRML model
   * Uses KiCad CLI to generate VRML file for web 3D viewing
   */
  router.get('/projects/:projectId/pcb-layout/:layoutId/board.wrl', async (req: Request, res: Response, next: NextFunction) => {
    try {
      const { projectId, layoutId } = req.params;
      log.info('Exporting PCB as VRML', { projectId, layoutId });

      const layout = await findPCBLayoutById(layoutId);
      if (!layout) {
        throw new NotFoundError('PCB Layout', layoutId, { operation: 'exportVrml' });
      }
      if (layout.projectId !== projectId) {
        throw new ValidationError('PCB Layout does not belong to this project', {
          operation: 'exportVrml', layoutId, projectId,
        });
      }

      const pcbContent = await getKicadContent(layoutId);
      if (!pcbContent) {
        throw new NotFoundError('PCB content', layoutId, { operation: 'exportVrml' });
      }

      // Write PCB to temp file
      const fs = await import('fs/promises');
      const path = await import('path');
      const os = await import('os');
      const tempDir = await fs.mkdtemp(path.join(os.tmpdir(), 'vrml-export-'));
      const pcbPath = path.join(tempDir, 'board.kicad_pcb');
      const vrmlPath = path.join(tempDir, 'board.wrl');
      await fs.writeFile(pcbPath, pcbContent);

      // Run export script
      const { spawn } = await import('child_process');
      const scriptsDir = config.kicad.scriptsDir;
      const pythonPath = config.kicad.pythonPath;
      const scriptPath = path.join(scriptsDir, 'export_3d_model.py');

      const result = await new Promise<{ success: boolean; filePath?: string; message?: string }>((resolve) => {
        const proc = spawn(pythonPath, [scriptPath, pcbPath, '--format', 'vrml', '--output', vrmlPath]);
        let stdout = '';
        proc.stdout.on('data', (data) => { stdout += data.toString(); });
        proc.on('close', () => {
          try {
            resolve(JSON.parse(stdout));
          } catch {
            resolve({ success: false, message: stdout });
          }
        });
        proc.on('error', () => resolve({ success: false, message: 'Process error' }));
      });

      if (!result.success || !result.filePath) {
        await fs.rm(tempDir, { recursive: true, force: true }).catch(() => {});
        throw new Error(`VRML export failed: ${result.message || 'Unknown error'}`);
      }

      // Send VRML file
      const vrmlContent = await fs.readFile(result.filePath);
      await fs.rm(tempDir, { recursive: true, force: true }).catch(() => {});

      res.setHeader('Content-Type', 'model/vrml');
      res.setHeader('Content-Disposition', `attachment; filename="board-${layoutId.substring(0,8)}.wrl"`);
      res.setHeader('Cache-Control', 'no-cache, no-store, must-revalidate');
      res.send(vrmlContent);
    } catch (error) {
      next(error);
    }
  });

  router.post('/projects/:projectId/pcb-layout/:layoutId/validate', async (req: Request, res: Response, next: NextFunction) => {
    try {
      const { projectId, layoutId } = req.params;
      log.info('Validating PCB layout', { projectId, layoutId });

      const layout = await findPCBLayoutById(layoutId);

      if (!layout) {
        throw new NotFoundError('PCB Layout', layoutId, { operation: 'validatePcbLayout' });
      }

      if (layout.projectId !== projectId) {
        throw new ValidationError('PCB Layout does not belong to this project', {
          operation: 'validatePcbLayout',
          layoutId,
          projectId,
        });
      }

      // Queue DRC validation job
      const jobId = crypto.randomUUID();

      io.to(`project:${projectId}`).emit('layout:validation-started', {
        jobId,
        layoutId,
        projectId,
      });

      // Return current validation status if available
      res.status(202).json({
        success: true,
        data: {
          jobId,
          layoutId,
          projectId,
          status: 'pending',
          currentScore: layout.score,
          validationResults: layout.validationResults,
        },
      });
    } catch (error) {
      next(error);
    }
  });

  router.post('/projects/:projectId/pcb-layout/:layoutId/render', validate(z.object({
    layers: z.array(z.string()).optional(),
    format: z.enum(['png', 'svg', 'pdf']).default('png'),
    resolution: z.number().default(300),
  })), async (req: Request, res: Response, next: NextFunction) => {
    try {
      const { projectId, layoutId } = req.params;
      log.info('Rendering PCB layers', { projectId, layoutId, format: req.body.format });

      const layout = await findPCBLayoutById(layoutId);

      if (!layout) {
        throw new NotFoundError('PCB Layout', layoutId, { operation: 'renderPcbLayout' });
      }

      if (layout.projectId !== projectId) {
        throw new ValidationError('PCB Layout does not belong to this project', {
          operation: 'renderPcbLayout',
          layoutId,
          projectId,
        });
      }

      // Queue render job
      const jobId = crypto.randomUUID();

      io.to(`project:${projectId}`).emit('layout:render-started', {
        jobId,
        layoutId,
        projectId,
        format: req.body.format,
        layers: req.body.layers,
      });

      res.status(202).json({
        success: true,
        data: {
          jobId,
          layoutId,
          projectId,
          status: 'pending',
          format: req.body.format,
        },
      });
    } catch (error) {
      next(error);
    }
  });

  // ============================================================================
  // Phase 6: Manufacturing
  // ============================================================================

  router.post('/projects/:projectId/manufacturing/gerbers', validate(z.object({
    layoutId: z.string().uuid(),
    format: z.enum(['gerber_x2', 'rs274x']).default('gerber_x2'),
  })), async (req: Request, res: Response, next: NextFunction) => {
    try {
      const projectId = req.params.projectId;
      log.info('Generating Gerber files', { projectId, layoutId: req.body.layoutId, format: req.body.format });

      // Verify project and layout exist
      const project = await findProjectById(projectId);
      if (!project) {
        throw new NotFoundError('Project', projectId, { operation: 'generateGerbers' });
      }

      const layout = await findPCBLayoutById(req.body.layoutId);
      if (!layout) {
        throw new NotFoundError('PCB Layout', req.body.layoutId, { operation: 'generateGerbers' });
      }

      if (layout.projectId !== projectId) {
        throw new ValidationError('PCB Layout does not belong to this project', {
          operation: 'generateGerbers',
          layoutId: req.body.layoutId,
          projectId,
        });
      }

      // Queue Gerber generation job
      const jobId = crypto.randomUUID();

      io.to(`project:${projectId}`).emit('gerbers:generation-started', {
        jobId,
        layoutId: req.body.layoutId,
        projectId,
        format: req.body.format,
      });

      res.status(202).json({
        success: true,
        data: {
          jobId,
          layoutId: req.body.layoutId,
          projectId,
          status: 'pending',
          format: req.body.format,
        },
      });
    } catch (error) {
      next(error);
    }
  });

  router.post('/projects/:projectId/manufacturing/quote', validate(z.object({
    layoutId: z.string().uuid(),
    vendor: z.enum(['pcbway', 'jlcpcb', 'oshpark', 'eurocircuits']),
    quantity: z.number().min(1),
    options: z.object({
      layers: z.number(),
      thickness: z.number(),
      surfaceFinish: z.string(),
      soldermaskColor: z.string(),
    }),
  })), async (req: Request, res: Response, next: NextFunction) => {
    try {
      const projectId = req.params.projectId;
      log.info('Getting manufacturing quote', { projectId, vendor: req.body.vendor, quantity: req.body.quantity });

      // Verify project and layout exist
      const project = await findProjectById(projectId);
      if (!project) {
        throw new NotFoundError('Project', projectId, { operation: 'getQuote' });
      }

      const layout = await findPCBLayoutById(req.body.layoutId);
      if (!layout) {
        throw new NotFoundError('PCB Layout', req.body.layoutId, { operation: 'getQuote' });
      }

      // Queue quote request job
      const jobId = crypto.randomUUID();

      io.to(`project:${projectId}`).emit('quote:request-started', {
        jobId,
        layoutId: req.body.layoutId,
        projectId,
        vendor: req.body.vendor,
        quantity: req.body.quantity,
      });

      res.status(202).json({
        success: true,
        data: {
          jobId,
          layoutId: req.body.layoutId,
          projectId,
          status: 'pending',
          vendor: req.body.vendor,
          quantity: req.body.quantity,
        },
      });
    } catch (error) {
      next(error);
    }
  });

  router.post('/projects/:projectId/manufacturing/order', validate(z.object({
    quoteId: z.string(),
    vendor: z.string(),
    shippingAddress: z.object({
      name: z.string(),
      street: z.string(),
      city: z.string(),
      state: z.string(),
      zip: z.string(),
      country: z.string(),
    }),
  })), async (req: Request, res: Response, next: NextFunction) => {
    try {
      const projectId = req.params.projectId;
      log.info('Placing manufacturing order', { projectId, vendor: req.body.vendor });

      // Verify project exists
      const project = await findProjectById(projectId);
      if (!project) {
        throw new NotFoundError('Project', projectId, { operation: 'placeOrder' });
      }

      // Queue order placement job
      const jobId = crypto.randomUUID();

      io.to(`project:${projectId}`).emit('order:placement-started', {
        jobId,
        projectId,
        vendor: req.body.vendor,
        quoteId: req.body.quoteId,
      });

      res.status(202).json({
        success: true,
        data: {
          jobId,
          projectId,
          status: 'pending',
          vendor: req.body.vendor,
          quoteId: req.body.quoteId,
        },
      });
    } catch (error) {
      next(error);
    }
  });

  // ============================================================================
  // Phase 7: Firmware Generation
  // ============================================================================

  router.post('/projects/:projectId/firmware/generate', validate(z.object({
    schematicId: z.string().uuid().optional(),
    pcbLayoutId: z.string().uuid().optional(),
    name: z.string().optional(),
    targetMcu: z.object({
      family: z.enum(['stm32', 'esp32', 'ti_tms320', 'infineon_aurix', 'nordic_nrf', 'rpi_pico', 'nxp_imxrt']),
      part: z.string(),
      core: z.string().optional(),
      flashSize: z.number().optional(),
      ramSize: z.number().optional(),
      peripherals: z.array(z.string()).optional(),
    }),
    rtos: z.enum(['freertos', 'zephyr', 'tirtos', 'autosar', 'none']).optional(),
    features: z.array(z.string()).optional(),
  })), async (req: Request, res: Response, next: NextFunction) => {
    try {
      const projectId = req.params.projectId;
      log.info('Generating firmware', { projectId, mcuFamily: req.body.targetMcu.family });

      // Verify project exists
      const project = await findProjectById(projectId);
      if (!project) {
        throw new NotFoundError('Project', projectId, { operation: 'generateFirmware' });
      }

      // Create firmware project record
      const firmwareInput: CreateFirmwareInput = {
        projectId,
        pcbLayoutId: req.body.pcbLayoutId,
        name: req.body.name || `${project.name} Firmware`,
        targetMcu: {
          family: req.body.targetMcu.family,
          part: req.body.targetMcu.part,
          core: req.body.targetMcu.core || 'cortex-m4',
          flashSize: req.body.targetMcu.flashSize || 512,
          ramSize: req.body.targetMcu.ramSize || 128,
          clockSpeed: req.body.targetMcu.clockSpeed || 168,
          peripherals: req.body.targetMcu.peripherals || [],
        },
        rtosConfig: req.body.rtos && req.body.rtos !== 'none' ? {
          type: req.body.rtos,
          version: '10.4.3',
          tickRate: 1000,
          heapSize: 32768,
          maxTasks: 16,
        } : undefined,
      };

      const firmware = await createFirmware(firmwareInput);

      io.to(`project:${projectId}`).emit('firmware:generation-started', {
        firmwareId: firmware.id,
        projectId,
        mcuFamily: req.body.targetMcu.family,
        mcuPart: req.body.targetMcu.part,
      });

      res.status(202).json({
        success: true,
        data: {
          firmwareId: firmware.id,
          status: 'pending',
          targetMcu: req.body.targetMcu,
          projectId,
        },
      });
    } catch (error) {
      next(error);
    }
  });

  router.get('/projects/:projectId/firmware', async (req: Request, res: Response, next: NextFunction) => {
    try {
      const projectId = req.params.projectId;
      log.debug('Getting project firmware', { projectId });

      // Verify project exists
      const project = await findProjectById(projectId);
      if (!project) {
        throw new NotFoundError('Project', projectId, { operation: 'getFirmware' });
      }

      const firmwareProjects = await findFirmwareByProject(projectId);

      res.json({
        success: true,
        data: firmwareProjects,
        metadata: { count: firmwareProjects.length, projectId },
      });
    } catch (error) {
      next(error);
    }
  });

  router.get('/projects/:projectId/firmware/:firmwareId', async (req: Request, res: Response, next: NextFunction) => {
    try {
      const { projectId, firmwareId } = req.params;
      log.debug('Getting firmware', { projectId, firmwareId });

      const firmware = await findFirmwareById(firmwareId);

      if (!firmware) {
        throw new NotFoundError('Firmware', firmwareId, { operation: 'getFirmware' });
      }

      if (firmware.projectId !== projectId) {
        throw new ValidationError('Firmware does not belong to this project', {
          operation: 'getFirmware',
          firmwareId,
          projectId,
        });
      }

      res.json({
        success: true,
        data: firmware,
      });
    } catch (error) {
      next(error);
    }
  });

  router.post('/projects/:projectId/firmware/:firmwareId/hal', validate(z.object({
    peripherals: z.array(z.object({
      type: z.string(),
      instance: z.string(),
      config: z.record(z.unknown()),
    })),
  })), async (req: Request, res: Response, next: NextFunction) => {
    try {
      const { projectId, firmwareId } = req.params;
      log.info('Generating HAL code', { projectId, firmwareId, peripheralCount: req.body.peripherals.length });

      const firmware = await findFirmwareById(firmwareId);

      if (!firmware) {
        throw new NotFoundError('Firmware', firmwareId, { operation: 'generateHal' });
      }

      if (firmware.projectId !== projectId) {
        throw new ValidationError('Firmware does not belong to this project', {
          operation: 'generateHal',
          firmwareId,
          projectId,
        });
      }

      // Queue HAL generation job
      const jobId = crypto.randomUUID();

      io.to(`project:${projectId}`).emit('firmware:hal-generation-started', {
        jobId,
        firmwareId,
        projectId,
        peripherals: req.body.peripherals,
      });

      res.status(202).json({
        success: true,
        data: {
          jobId,
          firmwareId,
          projectId,
          status: 'pending',
          peripheralCount: req.body.peripherals.length,
        },
      });
    } catch (error) {
      next(error);
    }
  });

  router.post('/projects/:projectId/firmware/:firmwareId/driver', validate(z.object({
    component: z.string(),
    datasheet: z.string().url().optional(),
    interface: z.enum(['gpio', 'uart', 'spi', 'i2c', 'adc', 'pwm', 'can']),
  })), async (req: Request, res: Response, next: NextFunction) => {
    try {
      const { projectId, firmwareId } = req.params;
      log.info('Generating driver code', { projectId, firmwareId, component: req.body.component });

      const firmware = await findFirmwareById(firmwareId);

      if (!firmware) {
        throw new NotFoundError('Firmware', firmwareId, { operation: 'generateDriver' });
      }

      if (firmware.projectId !== projectId) {
        throw new ValidationError('Firmware does not belong to this project', {
          operation: 'generateDriver',
          firmwareId,
          projectId,
        });
      }

      // Queue driver generation job
      const jobId = crypto.randomUUID();

      io.to(`project:${projectId}`).emit('firmware:driver-generation-started', {
        jobId,
        firmwareId,
        projectId,
        component: req.body.component,
        interface: req.body.interface,
      });

      res.status(202).json({
        success: true,
        data: {
          jobId,
          firmwareId,
          projectId,
          status: 'pending',
          component: req.body.component,
        },
      });
    } catch (error) {
      next(error);
    }
  });

  // ============================================================================
  // Phase 8: Testing
  // ============================================================================

  router.post('/projects/:projectId/testing/generate-tests', validate(z.object({
    firmwareId: z.string().uuid(),
    testFramework: z.enum(['unity', 'cppuTest', 'gtest']).default('unity'),
    coverage: z.enum(['unit', 'integration', 'system']).default('unit'),
  })), async (req: Request, res: Response, next: NextFunction) => {
    try {
      const projectId = req.params.projectId;
      log.info('Generating tests', { projectId, firmwareId: req.body.firmwareId, framework: req.body.testFramework });

      // Verify project and firmware exist
      const project = await findProjectById(projectId);
      if (!project) {
        throw new NotFoundError('Project', projectId, { operation: 'generateTests' });
      }

      const firmware = await findFirmwareById(req.body.firmwareId);
      if (!firmware) {
        throw new NotFoundError('Firmware', req.body.firmwareId, { operation: 'generateTests' });
      }

      // Queue test generation job
      const jobId = crypto.randomUUID();

      io.to(`project:${projectId}`).emit('testing:generation-started', {
        jobId,
        firmwareId: req.body.firmwareId,
        projectId,
        framework: req.body.testFramework,
        coverage: req.body.coverage,
      });

      res.status(202).json({
        success: true,
        data: {
          jobId,
          firmwareId: req.body.firmwareId,
          projectId,
          status: 'pending',
          framework: req.body.testFramework,
          coverage: req.body.coverage,
        },
      });
    } catch (error) {
      next(error);
    }
  });

  router.post('/projects/:projectId/testing/hil-setup', validate(z.object({
    targetBoard: z.string(),
    testEquipment: z.array(z.string()),
  })), async (req: Request, res: Response, next: NextFunction) => {
    try {
      const projectId = req.params.projectId;
      log.info('Generating HIL setup', { projectId, targetBoard: req.body.targetBoard });

      // Verify project exists
      const project = await findProjectById(projectId);
      if (!project) {
        throw new NotFoundError('Project', projectId, { operation: 'generateHilSetup' });
      }

      // Queue HIL setup generation job
      const jobId = crypto.randomUUID();

      io.to(`project:${projectId}`).emit('testing:hil-setup-started', {
        jobId,
        projectId,
        targetBoard: req.body.targetBoard,
        testEquipment: req.body.testEquipment,
      });

      res.status(202).json({
        success: true,
        data: {
          jobId,
          projectId,
          status: 'pending',
          targetBoard: req.body.targetBoard,
        },
      });
    } catch (error) {
      next(error);
    }
  });

  // ============================================================================
  // Multi-LLM Validation
  // ============================================================================

  router.post('/projects/:projectId/validate/multi-llm', validate(z.object({
    artifactType: z.enum(['schematic', 'pcb', 'firmware', 'simulation']),
    artifactId: z.string().uuid(),
    validators: z.array(z.enum(['claude_opus', 'gemini_25_pro', 'domain_expert'])).optional(),
  })), async (req: Request, res: Response, next: NextFunction) => {
    try {
      const projectId = req.params.projectId;
      const validationId = crypto.randomUUID();

      log.info('Starting multi-LLM validation', {
        projectId,
        validationId,
        artifactType: req.body.artifactType,
      });

      // Verify project exists
      const project = await findProjectById(projectId);
      if (!project) {
        throw new NotFoundError('Project', projectId, { operation: 'multiLlmValidation' });
      }

      // Verify artifact exists based on type
      let artifact;
      switch (req.body.artifactType) {
        case 'schematic':
          artifact = await findSchematicById(req.body.artifactId);
          if (!artifact) {
            throw new NotFoundError('Schematic', req.body.artifactId, { operation: 'multiLlmValidation' });
          }
          break;
        case 'pcb':
          artifact = await findPCBLayoutById(req.body.artifactId);
          if (!artifact) {
            throw new NotFoundError('PCB Layout', req.body.artifactId, { operation: 'multiLlmValidation' });
          }
          break;
        case 'firmware':
          artifact = await findFirmwareById(req.body.artifactId);
          if (!artifact) {
            throw new NotFoundError('Firmware', req.body.artifactId, { operation: 'multiLlmValidation' });
          }
          break;
        case 'simulation':
          artifact = await findSimulationById(req.body.artifactId);
          if (!artifact) {
            throw new NotFoundError('Simulation', req.body.artifactId, { operation: 'multiLlmValidation' });
          }
          break;
      }

      const validators = req.body.validators || ['claude_opus', 'gemini_25_pro', 'domain_expert'];

      // Emit validation started
      io.to(`project:${projectId}`).emit('validation:started', {
        validationId,
        artifactType: req.body.artifactType,
        artifactId: req.body.artifactId,
        validators,
      });

      res.status(202).json({
        success: true,
        data: {
          validationId,
          status: 'pending',
          artifactType: req.body.artifactType,
          artifactId: req.body.artifactId,
          validators,
          projectId,
        },
      });
    } catch (error) {
      next(error);
    }
  });

  // ============================================================================
  // Legacy Agents endpoint (for backward compatibility)
  // ============================================================================

  router.get('/agents', async (_req: Request, res: Response, next: NextFunction) => {
    try {
      log.debug('Getting layout agents (legacy endpoint)');

      res.json({
        success: true,
        data: LAYOUT_AGENTS,
      });
    } catch (error) {
      next(error);
    }
  });

  // ============================================================================
  // Skills Routes (mounted from skills-routes.ts)
  // ============================================================================

  // Mount skills routes if client is available
  const skillsClient = getSkillsEngineClient();
  if (skillsClient) {
    router.use('/skills', createSkillsRoutes(skillsClient));
    log.info('Skills routes mounted');
  } else {
    // Fallback skills routes when client not initialized
    router.get('/skills', async (_req: Request, res: Response, next: NextFunction) => {
      try {
        res.json({
          success: true,
          data: [],
          metadata: {
            message: 'Skills Engine not initialized',
          },
        });
      } catch (error) {
        next(error);
      }
    });

    router.post('/skills/execute', async (_req: Request, res: Response, next: NextFunction) => {
      try {
        res.status(503).json({
          success: false,
          error: {
            code: 'SKILLS_ENGINE_UNAVAILABLE',
            message: 'Skills Engine is not initialized',
          },
        });
      } catch (error) {
        next(error);
      }
    });
  }

  return router;
}
