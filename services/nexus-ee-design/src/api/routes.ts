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
import { generateSchematic as generateKicadSchematic, generateMinimalSchematic } from '../utils/kicad-generator.js';
import { getSkillsEngineClient } from '../state.js';
import { createSkillsRoutes } from './skills-routes.js';

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

      // Extract component definitions if provided
      const componentDefs = Array.isArray(req.body.components)
        ? req.body.components.map((c: Record<string, unknown>) => ({
            reference: String(c.reference || 'U1'),
            value: String(c.value || 'Component'),
            library: String(c.library || 'Device'),
            symbol: String(c.symbol || 'R'),
            footprint: c.footprint ? String(c.footprint) : undefined,
          }))
        : [];

      // Generate KiCad schematic content
      // Import types from generator for proper typing
      type GeneratedSchematic = ReturnType<typeof generateKicadSchematic>;
      let generatedSchematic: GeneratedSchematic;
      if (subsystems.length > 0) {
        generatedSchematic = generateKicadSchematic({
          architecture: {
            subsystems,
            projectType: architectureData.projectType ? String(architectureData.projectType) : project.type,
            title: req.body.name || project.name,
            company: 'Adverant EE Design',
          },
          components: componentDefs.length > 0 ? componentDefs : undefined,
          projectName: req.body.name || project.name,
          paperSize: 'A4',
        });
      } else {
        // Generate minimal schematic if no subsystems specified
        generatedSchematic = generateMinimalSchematic(req.body.name || project.name);
      }

      log.info('KiCad schematic content generated', {
        projectId,
        contentLength: generatedSchematic.content.length,
        componentCount: generatedSchematic.components.length,
        netCount: generatedSchematic.nets.length,
      });

      // Create schematic record with generated content
      // The CreateSchematicInput type allows simplified sheet/component/net structures
      // that get stored as JSON in the database
      const schematicInput: CreateSchematicInput = {
        projectId,
        name: req.body.name || `${project.name} Schematic`,
        format: 'kicad_sch',
        kicadSch: generatedSchematic.content,
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
          componentCount: generatedSchematic.components.length,
          netCount: generatedSchematic.nets.length,
          sheetCount: generatedSchematic.sheets.length,
        },
      });
    } catch (error) {
      log.error('Schematic generation failed', {
        projectId: req.params.projectId,
        error: error instanceof Error ? error.message : 'Unknown error',
      });
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
   */
  router.get('/projects/:projectId/schematic/:schematicId/file', async (req: Request, res: Response, next: NextFunction) => {
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
  });

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
