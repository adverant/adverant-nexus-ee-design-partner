/**
 * HIL (Hardware-in-the-Loop) Testing API Routes
 *
 * REST API endpoints for instrument management, test sequences, test runs,
 * captured data, and measurements. Production-ready with comprehensive
 * Zod validation and proper error handling.
 */

import { Router, Request, Response, NextFunction } from 'express';
import { z } from 'zod';
import path from 'path';
import fs from 'fs/promises';

import { ValidationError, NotFoundError, ConflictError } from '../utils/errors.js';
import { log } from '../utils/logger.js';
import { config } from '../config.js';
import { queueManager } from '../queue/queue-manager.js';

// Repository imports
import * as HILInstrumentRepository from '../database/repositories/hil-instrument-repository.js';
import * as HILTestSequenceRepository from '../database/repositories/hil-test-sequence-repository.js';
import * as HILTestRunRepository from '../database/repositories/hil-test-run-repository.js';
import * as HILCapturedDataRepository from '../database/repositories/hil-captured-data-repository.js';
import * as HILMeasurementRepository from '../database/repositories/hil-measurement-repository.js';

// Type imports
import type {
  HILInstrumentType,
  HILConnectionType,
  HILInstrumentStatus,
  HILTestType,
  HILTestRunStatus,
  HILSequenceConfig,
  HILPassCriteria,
  HILTestConditions,
  HILConnectionParams,
  HILInstrumentCapability,
  HILTemplateVariable,
  HILInstrumentFilters,
  HILTestSequenceFilters,
  HILTestRunFilters,
  HILCaptureType,
  FOCMotorParameters,
  FOCControllerParameters,
} from '../types/hil-types.js';

// Local measurement filters type (not exported from hil-types)
interface HILMeasurementFilters {
  passed?: boolean;
  isCritical?: boolean;
}

// ============================================================================
// Validation Middleware
// ============================================================================

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

function validateQuery<T>(schema: z.ZodSchema<T>) {
  return (req: Request, _res: Response, next: NextFunction) => {
    try {
      req.query = schema.parse(req.query) as typeof req.query;
      next();
    } catch (error) {
      if (error instanceof z.ZodError) {
        next(new ValidationError('Invalid query parameters', {
          operation: 'validation',
          errors: error.errors,
        }));
      } else {
        next(error);
      }
    }
  };
}

// ============================================================================
// Validation Schemas
// ============================================================================

// Instrument types
const HILInstrumentTypeSchema = z.enum([
  'logic_analyzer',
  'oscilloscope',
  'power_supply',
  'motor_emulator',
  'daq',
  'can_analyzer',
  'function_gen',
  'thermal_camera',
  'electronic_load',
]);

const HILConnectionTypeSchema = z.enum([
  'usb',
  'ethernet',
  'gpib',
  'serial',
  'grpc',
  'modbus_tcp',
  'modbus_rtu',
]);

const HILInstrumentStatusSchema = z.enum([
  'connected',
  'disconnected',
  'error',
  'busy',
  'initializing',
]);

const HILTestTypeSchema = z.enum([
  'foc_startup',
  'foc_steady_state',
  'foc_transient',
  'foc_speed_reversal',
  'pwm_analysis',
  'phase_current',
  'hall_sensor',
  'thermal_profile',
  'overcurrent_protection',
  'efficiency_sweep',
  'load_step',
  'no_load',
  'locked_rotor',
  'custom',
]);

// Connection parameters
const HILConnectionParamsSchema = z.object({
  // USB
  serial_number: z.string().optional(),
  vendor_id: z.string().optional(),
  product_id: z.string().optional(),
  // Ethernet
  host: z.string().optional(),
  port: z.number().int().positive().optional(),
  // GPIB
  address: z.number().int().min(0).max(30).optional(),
  board: z.number().int().min(0).optional(),
  // Serial
  serial_port: z.string().optional(),
  baud_rate: z.number().int().positive().optional(),
  data_bits: z.number().int().min(5).max(8).optional(),
  stop_bits: z.number().int().min(1).max(2).optional(),
  parity: z.enum(['none', 'odd', 'even']).optional(),
  // VISA
  resource_name: z.string().optional(),
}).passthrough();

// Instrument capability
const HILInstrumentCapabilitySchema = z.object({
  name: z.string().min(1),
  type: z.enum(['protocol', 'feature', 'range', 'measurement']),
  parameters: z.record(z.unknown()).optional(),
});

// Create instrument
const CreateInstrumentSchema = z.object({
  projectId: z.string().uuid(),
  name: z.string().min(1).max(255),
  instrumentType: HILInstrumentTypeSchema,
  manufacturer: z.string().min(1).max(255),
  model: z.string().min(1).max(255),
  connectionType: HILConnectionTypeSchema,
  connectionParams: HILConnectionParamsSchema,
  capabilities: z.array(HILInstrumentCapabilitySchema).optional(),
  serialNumber: z.string().max(255).optional(),
  firmwareVersion: z.string().max(50).optional(),
  notes: z.string().max(5000).optional(),
  tags: z.array(z.string().max(50)).max(20).optional(),
  metadata: z.record(z.unknown()).optional(),
});

// Update instrument
const UpdateInstrumentSchema = z.object({
  name: z.string().min(1).max(255).optional(),
  connectionParams: HILConnectionParamsSchema.optional(),
  capabilities: z.array(HILInstrumentCapabilitySchema).optional(),
  status: HILInstrumentStatusSchema.optional(),
  firmwareVersion: z.string().max(50).optional(),
  calibrationDate: z.string().datetime().optional(),
  calibrationDueDate: z.string().datetime().optional(),
  notes: z.string().max(5000).optional(),
  tags: z.array(z.string().max(50)).max(20).optional(),
  presets: z.record(z.object({
    name: z.string(),
    description: z.string().optional(),
    config: z.record(z.unknown()),
  })).optional(),
  defaultPreset: z.string().optional(),
  metadata: z.record(z.unknown()).optional(),
});

// Test step schema
const HILTestStepSchema = z.object({
  id: z.string().min(1),
  name: z.string().min(1),
  type: z.enum(['configure', 'measure', 'capture', 'wait', 'control', 'validate', 'loop', 'branch', 'script']),
  instrumentId: z.string().optional(),
  instrumentType: HILInstrumentTypeSchema.optional(),
  parameters: z.record(z.unknown()),
  expectedResults: z.array(z.object({
    measurement: z.string(),
    channel: z.string().optional(),
    operator: z.enum(['eq', 'ne', 'gt', 'gte', 'lt', 'lte', 'range', 'tolerance', 'contains', 'regex']),
    value: z.union([z.number(), z.string(), z.tuple([z.number(), z.number()])]),
    unit: z.string(),
    tolerancePercent: z.number().optional(),
    toleranceAbsolute: z.number().optional(),
    isCritical: z.boolean().optional(),
    description: z.string().optional(),
  })).optional(),
  timeout_ms: z.number().int().positive().optional(),
  retryOnFail: z.boolean().optional(),
  maxRetries: z.number().int().min(0).max(10).optional(),
  continueOnFail: z.boolean().optional(),
  description: z.string().optional(),
  delayBefore_ms: z.number().int().min(0).optional(),
  delayAfter_ms: z.number().int().min(0).optional(),
});

// Instrument requirement schema
const HILInstrumentRequirementSchema = z.object({
  instrumentType: HILInstrumentTypeSchema,
  capabilities: z.array(z.string()).optional(),
  minChannels: z.number().int().positive().optional(),
  minSampleRate: z.number().positive().optional(),
  optional: z.boolean().optional(),
  description: z.string().optional(),
});

// Sequence config schema
const HILSequenceConfigSchema = z.object({
  steps: z.array(HILTestStepSchema),
  globalParameters: z.record(z.unknown()).optional(),
  instrumentRequirements: z.array(HILInstrumentRequirementSchema),
  setupSteps: z.array(HILTestStepSchema).optional(),
  teardownSteps: z.array(HILTestStepSchema).optional(),
});

// Pass criteria schema
const HILPassCriteriaSchema = z.object({
  minPassPercentage: z.number().min(0).max(100),
  criticalMeasurements: z.array(z.string()),
  failFast: z.boolean(),
  allowedWarnings: z.number().int().min(0).optional(),
  customCondition: z.string().optional(),
});

// Template variable schema
const HILTemplateVariableSchema = z.object({
  name: z.string(),
  type: z.enum(['number', 'string', 'boolean', 'select']),
  label: z.string(),
  description: z.string().optional(),
  defaultValue: z.unknown().optional(),
  required: z.boolean().optional(),
  options: z.array(z.object({
    value: z.unknown(),
    label: z.string(),
  })).optional(),
  min: z.number().optional(),
  max: z.number().optional(),
  unit: z.string().optional(),
});

// Create test sequence
const CreateTestSequenceSchema = z.object({
  projectId: z.string().uuid(),
  name: z.string().min(1).max(255),
  description: z.string().max(5000).optional(),
  testType: HILTestTypeSchema,
  sequenceConfig: HILSequenceConfigSchema,
  passCriteria: HILPassCriteriaSchema,
  schematicId: z.string().uuid().optional(),
  pcbLayoutId: z.string().uuid().optional(),
  estimatedDurationMs: z.number().int().positive().optional(),
  timeoutMs: z.number().int().positive().optional(),
  priority: z.number().int().min(0).max(10).optional(),
  tags: z.array(z.string().max(50)).max(20).optional(),
  category: z.string().max(100).optional(),
  isTemplate: z.boolean().optional(),
  templateVariables: z.record(HILTemplateVariableSchema).optional(),
  metadata: z.record(z.unknown()).optional(),
});

// Update test sequence
const UpdateTestSequenceSchema = z.object({
  name: z.string().min(1).max(255).optional(),
  description: z.string().max(5000).optional(),
  sequenceConfig: HILSequenceConfigSchema.optional(),
  passCriteria: HILPassCriteriaSchema.optional(),
  estimatedDurationMs: z.number().int().positive().optional(),
  timeoutMs: z.number().int().positive().optional(),
  priority: z.number().int().min(0).max(10).optional(),
  tags: z.array(z.string().max(50)).max(20).optional(),
  category: z.string().max(100).optional(),
  metadata: z.record(z.unknown()).optional(),
});

// Start test run
const StartTestRunSchema = z.object({
  testConditions: z.object({
    ambientTemperature: z.number().optional(),
    humidity: z.number().min(0).max(100).optional(),
    supplyVoltage: z.number().positive().optional(),
    pressure: z.number().positive().optional(),
    notes: z.string().optional(),
  }).passthrough().optional(),
  parameterOverrides: z.record(z.unknown()).optional(),
  baselineRunId: z.string().uuid().optional(),
  tags: z.array(z.string().max(50)).max(20).optional(),
});

// Instantiate template
const InstantiateTemplateSchema = z.object({
  projectId: z.string().uuid(),
  name: z.string().min(1).max(255),
  variableValues: z.record(z.unknown()),
  description: z.string().max(5000).optional(),
  tags: z.array(z.string().max(50)).max(20).optional(),
});

// Discovery request
const DiscoverInstrumentsSchema = z.object({
  projectId: z.string().uuid(),
  connectionTypes: z.array(HILConnectionTypeSchema).optional(),
  instrumentTypes: z.array(HILInstrumentTypeSchema).optional(),
  scanTimeout: z.number().int().positive().max(30000).optional(),
});

// FOC test parameters
const FOCMotorParametersSchema = z.object({
  polePairs: z.number().int().positive(),
  ratedCurrent: z.number().positive(),
  maxCurrent: z.number().positive(),
  ratedSpeed: z.number().positive(),
  maxSpeed: z.number().positive(),
  ratedTorque: z.number().positive().optional(),
  resistance: z.number().positive().optional(),
  inductance: z.number().positive().optional(),
  backEmfConstant: z.number().positive().optional(),
  rotorInertia: z.number().positive().optional(),
  encoderResolution: z.number().int().positive().optional(),
  hallSensorType: z.enum(['digital', 'analog', 'none']).optional(),
  hallConfig: z.number().int().optional(),
});

const FOCControllerParametersSchema = z.object({
  pwmFrequency: z.number().positive(),
  deadTime: z.number().nonnegative(),
  currentKp: z.number(),
  currentKi: z.number(),
  speedKp: z.number(),
  speedKi: z.number(),
  positionKp: z.number().optional(),
  fieldWeakeningEnabled: z.boolean().optional(),
  maxFieldWeakeningCurrent: z.number().positive().optional(),
  currentLimit: z.number().positive(),
  accelerationLimit: z.number().positive().optional(),
});

const FOCTestParametersSchema = z.object({
  motor: FOCMotorParametersSchema,
  controller: FOCControllerParametersSchema,
  dcBusVoltage: z.number().positive(),
  targetSpeed: z.number(),
  targetTorque: z.number().optional(),
  loadProfile: z.array(z.object({
    time_ms: z.number().nonnegative(),
    torque: z.number(),
  })).optional(),
  testDuration: z.number().int().positive(),
});

// ============================================================================
// Route Factory
// ============================================================================

export function createHILRoutes(): Router {
  const router = Router();

  // ============================================================================
  // Instrument Discovery
  // ============================================================================

  /**
   * Trigger hardware instrument discovery
   * POST /instruments/discover
   */
  router.post('/instruments/discover',
    validate(DiscoverInstrumentsSchema),
    async (req: Request, res: Response, next: NextFunction) => {
      try {
        const { projectId, connectionTypes, instrumentTypes, scanTimeout } = req.body;

        log.info('Starting instrument discovery', {
          projectId,
          connectionTypes,
          instrumentTypes,
        });

        // Queue discovery job - addJob returns job ID as string
        const jobId = await queueManager.addJob('hil', {
          type: 'discover',
          projectId,
          config: {
            connectionTypes: connectionTypes || ['usb', 'ethernet'],
            instrumentTypes: instrumentTypes || [
              'logic_analyzer', 'oscilloscope', 'power_supply',
              'motor_emulator', 'daq', 'can_analyzer',
            ],
            scanTimeout: scanTimeout || 10000,
          },
        } as any);

        res.status(202).json({
          success: true,
          data: {
            jobId,
            status: 'queued',
            message: 'Instrument discovery started',
          },
          metadata: {
            projectId,
            timestamp: new Date().toISOString(),
          },
        });
      } catch (error) {
        next(error);
      }
    }
  );

  // ============================================================================
  // Instrument Management
  // ============================================================================

  /**
   * List instruments for a project
   * GET /projects/:projectId/instruments
   */
  router.get('/projects/:projectId/instruments',
    async (req: Request, res: Response, next: NextFunction) => {
      try {
        const { projectId } = req.params;
        const { type, status, connection_type, limit, offset } = req.query;

        const filters: HILInstrumentFilters = {};
        if (type) filters.instrumentType = type as HILInstrumentType;
        if (status) filters.status = status as HILInstrumentStatus;
        if (connection_type) filters.connectionType = connection_type as HILConnectionType;

        // Note: limit/offset not yet supported in repository
        const instruments = await HILInstrumentRepository.findByProject(projectId, filters);

        const stats = await HILInstrumentRepository.getProjectStats(projectId);

        res.json({
          success: true,
          data: instruments,
          metadata: {
            projectId,
            total: stats.total,
            stats,
            timestamp: new Date().toISOString(),
          },
        });
      } catch (error) {
        next(error);
      }
    }
  );

  /**
   * Create a new instrument
   * POST /instruments
   */
  router.post('/instruments',
    validate(CreateInstrumentSchema),
    async (req: Request, res: Response, next: NextFunction) => {
      try {
        log.info('Creating instrument', {
          projectId: req.body.projectId,
          name: req.body.name,
          type: req.body.instrumentType,
        });

        const instrument = await HILInstrumentRepository.create(req.body);

        res.status(201).json({
          success: true,
          data: instrument,
        });
      } catch (error) {
        next(error);
      }
    }
  );

  /**
   * Get instrument by ID
   * GET /instruments/:id
   */
  router.get('/instruments/:id',
    async (req: Request, res: Response, next: NextFunction) => {
      try {
        const instrument = await HILInstrumentRepository.findById(req.params.id);

        if (!instrument) {
          return res.status(404).json({
            success: false,
            error: {
              code: 'INSTRUMENT_NOT_FOUND',
              message: `Instrument '${req.params.id}' not found`,
            },
          });
        }

        res.json({
          success: true,
          data: instrument,
        });
      } catch (error) {
        next(error);
      }
    }
  );

  /**
   * Update instrument
   * PATCH /instruments/:id
   */
  router.patch('/instruments/:id',
    validate(UpdateInstrumentSchema),
    async (req: Request, res: Response, next: NextFunction) => {
      try {
        const instrument = await HILInstrumentRepository.update(req.params.id, req.body);

        res.json({
          success: true,
          data: instrument,
        });
      } catch (error) {
        next(error);
      }
    }
  );

  /**
   * Delete instrument
   * DELETE /instruments/:id
   */
  router.delete('/instruments/:id',
    async (req: Request, res: Response, next: NextFunction) => {
      try {
        await HILInstrumentRepository.deleteInstrument(req.params.id);

        res.json({
          success: true,
          message: 'Instrument deleted successfully',
        });
      } catch (error) {
        next(error);
      }
    }
  );

  /**
   * Test instrument connection
   * POST /instruments/:id/connect
   */
  router.post('/instruments/:id/connect',
    async (req: Request, res: Response, next: NextFunction) => {
      try {
        const instrument = await HILInstrumentRepository.findById(req.params.id);

        if (!instrument) {
          return res.status(404).json({
            success: false,
            error: {
              code: 'INSTRUMENT_NOT_FOUND',
              message: `Instrument '${req.params.id}' not found`,
            },
          });
        }

        // Queue connection test - addJob returns job ID as string
        const jobId = await queueManager.addJob('hil', {
          type: 'connect',
          instrumentId: req.params.id,
          projectId: instrument.projectId,
          connectionParams: instrument.connectionParams,
        } as any);

        // Update status to initializing
        await HILInstrumentRepository.updateStatus(req.params.id, 'initializing');

        res.status(202).json({
          success: true,
          data: {
            jobId,
            instrumentId: req.params.id,
            status: 'initializing',
            message: 'Connection test started',
          },
        });
      } catch (error) {
        next(error);
      }
    }
  );

  /**
   * Disconnect all instruments for a project
   * POST /projects/:projectId/instruments/disconnect-all
   */
  router.post('/projects/:projectId/instruments/disconnect-all',
    async (req: Request, res: Response, next: NextFunction) => {
      try {
        const { projectId } = req.params;

        const disconnectedCount = await HILInstrumentRepository.disconnectAll(projectId);

        res.json({
          success: true,
          data: {
            disconnectedCount,
            message: `Disconnected ${disconnectedCount} instruments`,
          },
        });
      } catch (error) {
        next(error);
      }
    }
  );

  /**
   * Get connected instruments for a project
   * GET /projects/:projectId/instruments/connected
   */
  router.get('/projects/:projectId/instruments/connected',
    async (req: Request, res: Response, next: NextFunction) => {
      try {
        const instruments = await HILInstrumentRepository.findConnected(req.params.projectId);

        res.json({
          success: true,
          data: instruments,
          metadata: {
            count: instruments.length,
          },
        });
      } catch (error) {
        next(error);
      }
    }
  );

  // ============================================================================
  // Test Sequence Management
  // ============================================================================

  /**
   * List test sequences for a project
   * GET /projects/:projectId/test-sequences
   */
  router.get('/projects/:projectId/test-sequences',
    async (req: Request, res: Response, next: NextFunction) => {
      try {
        const { projectId } = req.params;
        const { type, is_template, category, limit, offset } = req.query;

        const filters: HILTestSequenceFilters = {};
        if (type) filters.testType = type as HILTestType;
        if (is_template !== undefined) filters.isTemplate = is_template === 'true';
        if (category) filters.category = category as string;

        // Note: limit/offset not yet supported in repository
        const sequences = await HILTestSequenceRepository.findByProject(projectId, filters);

        const stats = await HILTestSequenceRepository.getProjectStats(projectId);

        res.json({
          success: true,
          data: sequences,
          metadata: {
            projectId,
            total: stats.total,
            stats,
            timestamp: new Date().toISOString(),
          },
        });
      } catch (error) {
        next(error);
      }
    }
  );

  /**
   * Create a test sequence
   * POST /test-sequences
   */
  router.post('/test-sequences',
    validate(CreateTestSequenceSchema),
    async (req: Request, res: Response, next: NextFunction) => {
      try {
        log.info('Creating test sequence', {
          projectId: req.body.projectId,
          name: req.body.name,
          testType: req.body.testType,
        });

        const sequence = await HILTestSequenceRepository.create(req.body);

        res.status(201).json({
          success: true,
          data: sequence,
        });
      } catch (error) {
        next(error);
      }
    }
  );

  /**
   * Get test sequence by ID
   * GET /test-sequences/:id
   */
  router.get('/test-sequences/:id',
    async (req: Request, res: Response, next: NextFunction) => {
      try {
        const sequence = await HILTestSequenceRepository.findById(req.params.id);

        if (!sequence) {
          return res.status(404).json({
            success: false,
            error: {
              code: 'SEQUENCE_NOT_FOUND',
              message: `Test sequence '${req.params.id}' not found`,
            },
          });
        }

        res.json({
          success: true,
          data: sequence,
        });
      } catch (error) {
        next(error);
      }
    }
  );

  /**
   * Update test sequence
   * PATCH /test-sequences/:id
   */
  router.patch('/test-sequences/:id',
    validate(UpdateTestSequenceSchema),
    async (req: Request, res: Response, next: NextFunction) => {
      try {
        const sequence = await HILTestSequenceRepository.update(req.params.id, req.body);

        res.json({
          success: true,
          data: sequence,
        });
      } catch (error) {
        next(error);
      }
    }
  );

  /**
   * Delete test sequence
   * DELETE /test-sequences/:id
   */
  router.delete('/test-sequences/:id',
    async (req: Request, res: Response, next: NextFunction) => {
      try {
        await HILTestSequenceRepository.deleteSequence(req.params.id);

        res.json({
          success: true,
          message: 'Test sequence deleted successfully',
        });
      } catch (error) {
        next(error);
      }
    }
  );

  /**
   * Clone a test sequence
   * POST /test-sequences/:id/clone
   */
  router.post('/test-sequences/:id/clone',
    validate(z.object({
      newName: z.string().min(1).max(255),
      targetProjectId: z.string().uuid().optional(),
    })),
    async (req: Request, res: Response, next: NextFunction) => {
      try {
        const { newName, targetProjectId } = req.body;

        const clonedSequence = await HILTestSequenceRepository.clone(
          req.params.id,
          newName,
          targetProjectId
        );

        res.status(201).json({
          success: true,
          data: clonedSequence,
        });
      } catch (error) {
        next(error);
      }
    }
  );

  /**
   * Get version history for a test sequence
   * GET /test-sequences/:id/versions
   */
  router.get('/test-sequences/:id/versions',
    async (req: Request, res: Response, next: NextFunction) => {
      try {
        const versions = await HILTestSequenceRepository.getVersionHistory(req.params.id);

        res.json({
          success: true,
          data: versions,
          metadata: {
            count: versions.length,
          },
        });
      } catch (error) {
        next(error);
      }
    }
  );

  /**
   * Start a test run from a sequence
   * POST /test-sequences/:id/run
   */
  router.post('/test-sequences/:id/run',
    validate(StartTestRunSchema),
    async (req: Request, res: Response, next: NextFunction) => {
      try {
        const sequenceId = req.params.id;
        const { testConditions, parameterOverrides, baselineRunId, tags } = req.body;

        // Get the sequence
        const sequence = await HILTestSequenceRepository.findById(sequenceId);
        if (!sequence) {
          return res.status(404).json({
            success: false,
            error: {
              code: 'SEQUENCE_NOT_FOUND',
              message: `Test sequence '${sequenceId}' not found`,
            },
          });
        }

        // Check for active test runs
        const activeRuns = await HILTestRunRepository.findActive(sequence.projectId);
        if (activeRuns.length > 0) {
          return res.status(409).json({
            success: false,
            error: {
              code: 'TEST_RUN_IN_PROGRESS',
              message: 'Another test run is already in progress',
              activeRunId: activeRuns[0].id,
            },
          });
        }

        log.info('Starting test run', {
          sequenceId,
          projectId: sequence.projectId,
          testType: sequence.testType,
        });

        // Create test run record (requires sequence name for run naming)
        const testRun = await HILTestRunRepository.create({
          sequenceId,
          testConditions: testConditions || {},
          baselineRunId,
          tags,
        }, sequence.name);

        // Get connected instruments
        const connectedInstruments = await HILInstrumentRepository.findConnected(sequence.projectId);

        // Queue the test run job - addJob returns job ID as string
        const jobId = await queueManager.addJob('hil', {
          type: 'test_run',
          testRunId: testRun.id,
          sequenceId,
          projectId: sequence.projectId,
          sequenceConfig: {
            ...sequence.sequenceConfig,
            globalParameters: {
              ...sequence.sequenceConfig.globalParameters,
              ...parameterOverrides,
            },
          },
          passCriteria: sequence.passCriteria,
          connectedInstruments: connectedInstruments.map(i => ({
            id: i.id,
            type: i.instrumentType,
            manufacturer: i.manufacturer,
            model: i.model,
            connectionParams: i.connectionParams,
          })),
          timeoutMs: sequence.timeoutMs || 600000,
        } as any, {
          priority: sequence.priority || 1,
          removeOnComplete: { count: 100 },
          removeOnFail: { count: 50 },
        });

        // Update test run status (worker will set job ID via assignWorker when it starts)
        await HILTestRunRepository.updateStatus(testRun.id, 'queued');

        res.status(202).json({
          success: true,
          data: {
            testRunId: testRun.id,
            jobId,
            sequenceId,
            status: 'queued',
            message: 'Test run queued successfully',
          },
        });
      } catch (error) {
        next(error);
      }
    }
  );

  // ============================================================================
  // Test Run Management
  // ============================================================================

  /**
   * List test runs for a project
   * GET /projects/:projectId/test-runs
   */
  router.get('/projects/:projectId/test-runs',
    async (req: Request, res: Response, next: NextFunction) => {
      try {
        const { projectId } = req.params;
        const { status, result, sequence_id, limit, offset } = req.query;

        const filters: HILTestRunFilters = {};
        if (status) filters.status = status as HILTestRunStatus;
        if (result) filters.result = result as any;
        if (sequence_id) filters.sequenceId = sequence_id as string;

        // Note: limit/offset not yet supported in repository
        const runs = await HILTestRunRepository.findByProject(projectId, filters);

        const stats = await HILTestRunRepository.getProjectStats(projectId);

        res.json({
          success: true,
          data: runs,
          metadata: {
            projectId,
            total: stats.total,
            stats,
            timestamp: new Date().toISOString(),
          },
        });
      } catch (error) {
        next(error);
      }
    }
  );

  /**
   * Get test run by ID
   * GET /test-runs/:id
   */
  router.get('/test-runs/:id',
    async (req: Request, res: Response, next: NextFunction) => {
      try {
        const testRun = await HILTestRunRepository.findById(req.params.id);

        if (!testRun) {
          return res.status(404).json({
            success: false,
            error: {
              code: 'TEST_RUN_NOT_FOUND',
              message: `Test run '${req.params.id}' not found`,
            },
          });
        }

        res.json({
          success: true,
          data: testRun,
        });
      } catch (error) {
        next(error);
      }
    }
  );

  /**
   * Abort a test run
   * POST /test-runs/:id/abort
   */
  router.post('/test-runs/:id/abort',
    validate(z.object({
      reason: z.string().max(1000).optional(),
    })),
    async (req: Request, res: Response, next: NextFunction) => {
      try {
        const { reason } = req.body;

        const testRun = await HILTestRunRepository.findById(req.params.id);
        if (!testRun) {
          return res.status(404).json({
            success: false,
            error: {
              code: 'TEST_RUN_NOT_FOUND',
              message: `Test run '${req.params.id}' not found`,
            },
          });
        }

        // Check if run can be aborted
        if (!['pending', 'queued', 'running'].includes(testRun.status)) {
          return res.status(400).json({
            success: false,
            error: {
              code: 'TEST_RUN_NOT_ABORTABLE',
              message: `Test run is in '${testRun.status}' state and cannot be aborted`,
            },
          });
        }

        log.info('Aborting test run', {
          testRunId: req.params.id,
          reason,
        });

        // If there's an active job, try to cancel it
        if (testRun.jobId) {
          try {
            const queue = queueManager.getQueue('hil' as any);
            const job = await queue.getJob(testRun.jobId);
            if (job && ['waiting', 'delayed', 'active'].includes(await job.getState())) {
              await job.moveToFailed(new Error(`Aborted: ${reason || 'User requested abort'}`), 'abort');
            }
          } catch (err) {
            log.warn('Failed to cancel job', { jobId: testRun.jobId, error: err });
          }
        }

        await HILTestRunRepository.abort(req.params.id, undefined, reason);

        res.json({
          success: true,
          data: {
            testRunId: req.params.id,
            status: 'aborted',
            message: 'Test run aborted successfully',
          },
        });
      } catch (error) {
        next(error);
      }
    }
  );

  /**
   * Compare test run to baseline
   * GET /test-runs/:id/compare/:baselineId
   */
  router.get('/test-runs/:id/compare/:baselineId',
    async (req: Request, res: Response, next: NextFunction) => {
      try {
        const comparison = await HILTestRunRepository.compareToBaseline(
          req.params.id,
          req.params.baselineId
        );

        res.json({
          success: true,
          data: comparison,
        });
      } catch (error) {
        next(error);
      }
    }
  );

  // ============================================================================
  // Measurements
  // ============================================================================

  /**
   * Get measurements for a test run
   * GET /test-runs/:id/measurements
   */
  router.get('/test-runs/:id/measurements',
    async (req: Request, res: Response, next: NextFunction) => {
      try {
        const { limit, offset, passed, critical } = req.query;

        const filters: HILMeasurementFilters = {};
        if (passed !== undefined) filters.passed = passed === 'true';
        if (critical !== undefined) filters.isCritical = critical === 'true';

        // Note: limit/offset not yet supported in repository
        const measurements = await HILMeasurementRepository.findByTestRun(req.params.id, filters);

        const summary = await HILMeasurementRepository.getSummary(req.params.id);

        res.json({
          success: true,
          data: measurements,
          metadata: {
            testRunId: req.params.id,
            summary,
            timestamp: new Date().toISOString(),
          },
        });
      } catch (error) {
        next(error);
      }
    }
  );

  /**
   * Get measurement summary by type
   * GET /test-runs/:id/measurements/by-type
   */
  router.get('/test-runs/:id/measurements/by-type',
    async (req: Request, res: Response, next: NextFunction) => {
      try {
        const byType = await HILMeasurementRepository.getByType(req.params.id);

        res.json({
          success: true,
          data: byType,
        });
      } catch (error) {
        next(error);
      }
    }
  );

  /**
   * Get key metrics for a test run
   * GET /test-runs/:id/measurements/key-metrics
   */
  router.get('/test-runs/:id/measurements/key-metrics',
    async (req: Request, res: Response, next: NextFunction) => {
      try {
        const keyMetrics = await HILMeasurementRepository.getKeyMetrics(req.params.id);

        res.json({
          success: true,
          data: keyMetrics,
        });
      } catch (error) {
        next(error);
      }
    }
  );

  /**
   * Get failed measurements for a test run
   * GET /test-runs/:id/measurements/failed
   */
  router.get('/test-runs/:id/measurements/failed',
    async (req: Request, res: Response, next: NextFunction) => {
      try {
        const { critical_only } = req.query;

        const failed = await HILMeasurementRepository.findFailed(
          req.params.id,
          critical_only === 'true'
        );

        res.json({
          success: true,
          data: failed,
          metadata: {
            count: failed.length,
            criticalOnly: critical_only === 'true',
          },
        });
      } catch (error) {
        next(error);
      }
    }
  );

  /**
   * Compare measurements between two test runs
   * GET /test-runs/:id/measurements/compare/:otherRunId
   */
  router.get('/test-runs/:id/measurements/compare/:otherRunId',
    async (req: Request, res: Response, next: NextFunction) => {
      try {
        const comparison = await HILMeasurementRepository.compareMeasurements(
          req.params.id,
          req.params.otherRunId
        );

        res.json({
          success: true,
          data: comparison,
        });
      } catch (error) {
        next(error);
      }
    }
  );

  // ============================================================================
  // Captured Data
  // ============================================================================

  /**
   * Get captured data for a test run
   * GET /test-runs/:id/captures
   */
  router.get('/test-runs/:id/captures',
    async (req: Request, res: Response, next: NextFunction) => {
      try {
        const { type, limit, offset } = req.query;

        // Note: limit/offset not yet supported in repository
        const captureType = type as HILCaptureType | undefined;
        const captures = await HILCapturedDataRepository.findByTestRun(req.params.id, captureType);

        const stats = await HILCapturedDataRepository.getTestRunStats(req.params.id);

        res.json({
          success: true,
          data: captures,
          metadata: {
            testRunId: req.params.id,
            stats,
            timestamp: new Date().toISOString(),
          },
        });
      } catch (error) {
        next(error);
      }
    }
  );

  /**
   * Get waveforms for a test run
   * GET /test-runs/:id/captures/waveforms
   */
  router.get('/test-runs/:id/captures/waveforms',
    async (req: Request, res: Response, next: NextFunction) => {
      try {
        const waveforms = await HILCapturedDataRepository.findWaveforms(req.params.id);

        res.json({
          success: true,
          data: waveforms,
          metadata: {
            count: waveforms.length,
          },
        });
      } catch (error) {
        next(error);
      }
    }
  );

  /**
   * Get logic traces for a test run
   * GET /test-runs/:id/captures/logic-traces
   */
  router.get('/test-runs/:id/captures/logic-traces',
    async (req: Request, res: Response, next: NextFunction) => {
      try {
        const traces = await HILCapturedDataRepository.findLogicTraces(req.params.id);

        res.json({
          success: true,
          data: traces,
          metadata: {
            count: traces.length,
          },
        });
      } catch (error) {
        next(error);
      }
    }
  );

  /**
   * Get capture by ID
   * GET /captures/:id
   */
  router.get('/captures/:id',
    async (req: Request, res: Response, next: NextFunction) => {
      try {
        const capture = await HILCapturedDataRepository.findById(req.params.id);

        if (!capture) {
          return res.status(404).json({
            success: false,
            error: {
              code: 'CAPTURE_NOT_FOUND',
              message: `Captured data '${req.params.id}' not found`,
            },
          });
        }

        res.json({
          success: true,
          data: capture,
        });
      } catch (error) {
        next(error);
      }
    }
  );

  /**
   * Download capture data file
   * GET /captures/:id/data
   */
  router.get('/captures/:id/data',
    async (req: Request, res: Response, next: NextFunction) => {
      try {
        const capture = await HILCapturedDataRepository.findById(req.params.id);

        if (!capture) {
          return res.status(404).json({
            success: false,
            error: {
              code: 'CAPTURE_NOT_FOUND',
              message: `Captured data '${req.params.id}' not found`,
            },
          });
        }

        // If data is inline, return it directly
        if (capture.dataInline) {
          res.json({
            success: true,
            data: capture.dataInline,
          });
          return;
        }

        // If data is in a file, stream it
        if (capture.dataPath) {
          const filePath = path.isAbsolute(capture.dataPath)
            ? capture.dataPath
            : path.join(config.hil.storage.capturesDir, capture.dataPath);

          try {
            await fs.access(filePath);
          } catch {
            return res.status(404).json({
              success: false,
              error: {
                code: 'DATA_FILE_NOT_FOUND',
                message: 'Capture data file not found',
              },
            });
          }

          // Determine content type based on format
          const contentTypes: Record<string, string> = {
            json: 'application/json',
            csv: 'text/csv',
            binary: 'application/octet-stream',
            vcd: 'application/x-vcd',
            salae: 'application/octet-stream',
            sigrok: 'application/octet-stream',
          };

          const contentType = contentTypes[capture.dataFormat] || 'application/octet-stream';
          const ext = capture.dataFormat === 'json' ? 'json' : capture.dataFormat;

          res.setHeader('Content-Type', contentType);
          res.setHeader(
            'Content-Disposition',
            `attachment; filename="${capture.id}.${ext}"`
          );

          const fileData = await fs.readFile(filePath);
          res.send(fileData);
          return;
        }

        res.status(404).json({
          success: false,
          error: {
            code: 'NO_DATA',
            message: 'No capture data available',
          },
        });
      } catch (error) {
        next(error);
      }
    }
  );

  /**
   * Add annotation to capture
   * POST /captures/:id/annotations
   */
  router.post('/captures/:id/annotations',
    validate(z.object({
      timestamp_ms: z.number().nonnegative(),
      text: z.string().min(1).max(500),
      color: z.string().optional(),
      type: z.enum(['marker', 'region', 'note']).optional(),
      endTimestamp_ms: z.number().nonnegative().optional(),
    })),
    async (req: Request, res: Response, next: NextFunction) => {
      try {
        const capture = await HILCapturedDataRepository.addAnnotation(
          req.params.id,
          req.body
        );

        res.status(201).json({
          success: true,
          data: capture,
        });
      } catch (error) {
        next(error);
      }
    }
  );

  // ============================================================================
  // FOC Test Templates
  // ============================================================================

  /**
   * Get FOC test templates
   * GET /templates/foc
   */
  router.get('/templates/foc',
    async (_req: Request, res: Response, next: NextFunction) => {
      try {
        // Get all templates and filter by FOC test types
        const allTemplates = await HILTestSequenceRepository.findTemplates();
        const templates = allTemplates.filter(t =>
          t.category === 'foc' ||
          t.testType?.startsWith('foc_') ||
          t.testType === 'pwm_analysis' ||
          t.testType === 'phase_current' ||
          t.testType === 'efficiency_sweep'
        );

        // If no templates in DB, return built-in templates
        const builtInTemplates = getFOCTestTemplates();

        res.json({
          success: true,
          data: {
            custom: templates,
            builtin: builtInTemplates,
          },
          metadata: {
            customCount: templates.length,
            builtinCount: builtInTemplates.length,
          },
        });
      } catch (error) {
        next(error);
      }
    }
  );

  /**
   * Get specific FOC template
   * GET /templates/foc/:templateId
   */
  router.get('/templates/foc/:templateId',
    async (req: Request, res: Response, next: NextFunction) => {
      try {
        // Check built-in templates first
        const builtInTemplates = getFOCTestTemplates();
        const builtIn = builtInTemplates.find(t => t.id === req.params.templateId);

        if (builtIn) {
          return res.json({
            success: true,
            data: builtIn,
            metadata: {
              type: 'builtin',
            },
          });
        }

        // Check custom templates
        const template = await HILTestSequenceRepository.findById(req.params.templateId);

        if (!template || !template.isTemplate) {
          return res.status(404).json({
            success: false,
            error: {
              code: 'TEMPLATE_NOT_FOUND',
              message: `FOC template '${req.params.templateId}' not found`,
            },
          });
        }

        res.json({
          success: true,
          data: template,
          metadata: {
            type: 'custom',
          },
        });
      } catch (error) {
        next(error);
      }
    }
  );

  /**
   * Instantiate FOC test template
   * POST /templates/foc/:templateId/instantiate
   */
  router.post('/templates/foc/:templateId/instantiate',
    validate(InstantiateTemplateSchema),
    async (req: Request, res: Response, next: NextFunction) => {
      try {
        const { templateId } = req.params;
        const { projectId, name, variableValues, description, tags } = req.body;

        // Check if it's a built-in template
        const builtInTemplates = getFOCTestTemplates();
        const builtIn = builtInTemplates.find(t => t.id === templateId);

        if (builtIn) {
          // Create from built-in template
          const sequence = await HILTestSequenceRepository.create({
            projectId,
            name,
            description: description || builtIn.description,
            testType: builtIn.testType as HILTestType,
            sequenceConfig: applyTemplateVariables(builtIn.sequenceConfig, variableValues),
            passCriteria: builtIn.passCriteria,
            estimatedDurationMs: builtIn.estimatedDurationMs,
            timeoutMs: builtIn.timeoutMs,
            tags: tags || builtIn.tags,
            category: 'foc',
            isTemplate: false,
            parentTemplateId: templateId,
            metadata: {
              instantiatedFrom: templateId,
              instantiatedAt: new Date().toISOString(),
              variableValues,
            },
          });

          return res.status(201).json({
            success: true,
            data: sequence,
          });
        }

        // Use custom template
        const sequence = await HILTestSequenceRepository.instantiateTemplate(
          templateId,
          projectId,
          name,
          variableValues
        );

        res.status(201).json({
          success: true,
          data: sequence,
        });
      } catch (error) {
        next(error);
      }
    }
  );

  // ============================================================================
  // Active Test Runs
  // ============================================================================

  /**
   * Get active test runs (optionally filtered by project)
   * GET /test-runs/active
   */
  router.get('/test-runs/active',
    async (req: Request, res: Response, next: NextFunction) => {
      try {
        const { project_id } = req.query;

        const activeRuns = await HILTestRunRepository.findActive(
          project_id as string | undefined
        );

        res.json({
          success: true,
          data: activeRuns,
          metadata: {
            count: activeRuns.length,
          },
        });
      } catch (error) {
        next(error);
      }
    }
  );

  return router;
}

// ============================================================================
// Built-in FOC Test Templates
// ============================================================================

interface FOCTestTemplate {
  id: string;
  name: string;
  description: string;
  testType: string;
  category: string;
  estimatedDurationMs: number;
  timeoutMs: number;
  tags: string[];
  templateVariables: Record<string, any>;
  sequenceConfig: any;
  passCriteria: any;
}

function getFOCTestTemplates(): FOCTestTemplate[] {
  return [
    {
      id: 'foc_startup_test',
      name: 'FOC Startup Test',
      description: 'Measures motor startup characteristics including time-to-speed, current overshoot, and initial alignment',
      testType: 'foc_startup',
      category: 'foc',
      estimatedDurationMs: 30000,
      timeoutMs: 60000,
      tags: ['foc', 'startup', 'motor', 'bldc'],
      templateVariables: {
        dcBusVoltage: { name: 'dcBusVoltage', type: 'number', label: 'DC Bus Voltage', unit: 'V', defaultValue: 24, min: 6, max: 60 },
        targetSpeed: { name: 'targetSpeed', type: 'number', label: 'Target Speed', unit: 'RPM', defaultValue: 1000, min: 100, max: 10000 },
        polePairs: { name: 'polePairs', type: 'number', label: 'Pole Pairs', defaultValue: 7, min: 1, max: 20 },
        ratedCurrent: { name: 'ratedCurrent', type: 'number', label: 'Rated Current', unit: 'A', defaultValue: 10, min: 0.1, max: 100 },
        pwmFrequency: { name: 'pwmFrequency', type: 'number', label: 'PWM Frequency', unit: 'Hz', defaultValue: 20000, min: 1000, max: 100000 },
      },
      sequenceConfig: {
        instrumentRequirements: [
          { instrumentType: 'oscilloscope', capabilities: ['4_channels'], minChannels: 4, description: 'For phase current capture' },
          { instrumentType: 'power_supply', capabilities: ['voltage_control'], description: 'DC bus power' },
          { instrumentType: 'logic_analyzer', capabilities: ['digital_channels'], optional: true, description: 'PWM capture' },
        ],
        setupSteps: [
          {
            id: 'setup_psu',
            name: 'Configure Power Supply',
            type: 'configure',
            instrumentType: 'power_supply',
            parameters: { voltage: '{{dcBusVoltage}}', currentLimit: '{{ratedCurrent * 1.5}}', enabled: false },
          },
          {
            id: 'setup_scope',
            name: 'Configure Oscilloscope',
            type: 'configure',
            instrumentType: 'oscilloscope',
            parameters: {
              channels: [
                { channel: 1, label: 'Phase A Current', scale: 'auto', coupling: 'dc' },
                { channel: 2, label: 'Phase B Current', scale: 'auto', coupling: 'dc' },
                { channel: 3, label: 'Phase C Current', scale: 'auto', coupling: 'dc' },
                { channel: 4, label: 'DC Bus Voltage', scale: 'auto', coupling: 'dc' },
              ],
              timebase: { scale: '1ms', position: 0 },
              trigger: { source: 'CH4', level: '{{dcBusVoltage * 0.1}}', edge: 'rising' },
            },
          },
        ],
        steps: [
          {
            id: 'enable_psu',
            name: 'Enable Power Supply',
            type: 'control',
            instrumentType: 'power_supply',
            parameters: { action: 'enable_output' },
          },
          {
            id: 'wait_stabilize',
            name: 'Wait for Bus Stabilization',
            type: 'wait',
            parameters: { duration_ms: 500 },
          },
          {
            id: 'measure_bus_voltage',
            name: 'Measure DC Bus Voltage',
            type: 'measure',
            instrumentType: 'oscilloscope',
            parameters: { channel: 4, measurement: 'mean' },
            expectedResults: [
              { measurement: 'mean', channel: '4', operator: 'tolerance', value: '{{dcBusVoltage}}', unit: 'V', tolerancePercent: 5, isCritical: true },
            ],
          },
          {
            id: 'arm_capture',
            name: 'Arm Oscilloscope',
            type: 'configure',
            instrumentType: 'oscilloscope',
            parameters: { trigger: { mode: 'single' } },
          },
          {
            id: 'start_motor',
            name: 'Send Motor Start Command',
            type: 'script',
            parameters: {
              script: 'motor_start',
              args: { targetSpeed: '{{targetSpeed}}', rampTime_ms: 1000 },
            },
          },
          {
            id: 'capture_startup',
            name: 'Capture Startup Waveform',
            type: 'capture',
            instrumentType: 'oscilloscope',
            parameters: { duration_ms: 5000, channels: [1, 2, 3, 4] },
          },
          {
            id: 'measure_startup_time',
            name: 'Measure Time to Speed',
            type: 'measure',
            instrumentType: 'oscilloscope',
            parameters: { measurement: 'time_to_stable', threshold_percent: 95 },
            expectedResults: [
              { measurement: 'startup_time', operator: 'lt', value: 2000, unit: 'ms', isCritical: true, description: 'Motor should reach target speed within 2 seconds' },
            ],
          },
          {
            id: 'measure_current_overshoot',
            name: 'Measure Current Overshoot',
            type: 'measure',
            instrumentType: 'oscilloscope',
            parameters: { channels: [1, 2, 3], measurement: 'peak' },
            expectedResults: [
              { measurement: 'peak_current', operator: 'lt', value: '{{ratedCurrent * 2}}', unit: 'A', isCritical: true, description: 'Peak current should not exceed 2x rated' },
            ],
          },
        ],
        teardownSteps: [
          {
            id: 'stop_motor',
            name: 'Stop Motor',
            type: 'script',
            parameters: { script: 'motor_stop' },
          },
          {
            id: 'disable_psu',
            name: 'Disable Power Supply',
            type: 'control',
            instrumentType: 'power_supply',
            parameters: { action: 'disable_output' },
          },
        ],
        globalParameters: {},
      },
      passCriteria: {
        minPassPercentage: 100,
        criticalMeasurements: ['startup_time', 'peak_current', 'bus_voltage'],
        failFast: false,
        allowedWarnings: 2,
      },
    },
    {
      id: 'foc_steady_state_test',
      name: 'FOC Steady State Test',
      description: 'Measures steady-state FOC performance including phase current RMS, THD, speed stability, and efficiency',
      testType: 'foc_steady_state',
      category: 'foc',
      estimatedDurationMs: 60000,
      timeoutMs: 120000,
      tags: ['foc', 'steady-state', 'efficiency', 'thd'],
      templateVariables: {
        dcBusVoltage: { name: 'dcBusVoltage', type: 'number', label: 'DC Bus Voltage', unit: 'V', defaultValue: 24, min: 6, max: 60 },
        targetSpeed: { name: 'targetSpeed', type: 'number', label: 'Target Speed', unit: 'RPM', defaultValue: 3000, min: 100, max: 10000 },
        loadTorque: { name: 'loadTorque', type: 'number', label: 'Load Torque', unit: 'Nm', defaultValue: 0.5, min: 0, max: 10 },
        polePairs: { name: 'polePairs', type: 'number', label: 'Pole Pairs', defaultValue: 7, min: 1, max: 20 },
        ratedCurrent: { name: 'ratedCurrent', type: 'number', label: 'Rated Current', unit: 'A', defaultValue: 10, min: 0.1, max: 100 },
        measurementDuration: { name: 'measurementDuration', type: 'number', label: 'Measurement Duration', unit: 'ms', defaultValue: 10000, min: 1000, max: 60000 },
      },
      sequenceConfig: {
        instrumentRequirements: [
          { instrumentType: 'oscilloscope', capabilities: ['4_channels', 'fft'], minChannels: 4, description: 'Phase current + FFT' },
          { instrumentType: 'power_supply', capabilities: ['voltage_control', 'measurement'], description: 'DC bus with measurement' },
          { instrumentType: 'motor_emulator', optional: true, description: 'For load control' },
        ],
        setupSteps: [
          {
            id: 'setup_psu',
            name: 'Configure Power Supply',
            type: 'configure',
            instrumentType: 'power_supply',
            parameters: { voltage: '{{dcBusVoltage}}', currentLimit: '{{ratedCurrent * 1.5}}', enabled: false },
          },
          {
            id: 'setup_scope',
            name: 'Configure Oscilloscope',
            type: 'configure',
            instrumentType: 'oscilloscope',
            parameters: {
              channels: [
                { channel: 1, label: 'Phase A', scale: 'auto', coupling: 'dc' },
                { channel: 2, label: 'Phase B', scale: 'auto', coupling: 'dc' },
                { channel: 3, label: 'Phase C', scale: 'auto', coupling: 'dc' },
                { channel: 4, label: 'DC Bus', scale: 'auto', coupling: 'dc' },
              ],
              timebase: { scale: '500us', position: 0 },
              acquisition: { mode: 'average', count: 16 },
            },
          },
        ],
        steps: [
          {
            id: 'enable_psu',
            name: 'Enable Power Supply',
            type: 'control',
            instrumentType: 'power_supply',
            parameters: { action: 'enable_output' },
          },
          {
            id: 'start_motor',
            name: 'Start Motor to Target Speed',
            type: 'script',
            parameters: { script: 'motor_start', args: { targetSpeed: '{{targetSpeed}}', rampTime_ms: 2000 } },
          },
          {
            id: 'wait_steady_state',
            name: 'Wait for Steady State',
            type: 'wait',
            parameters: { duration_ms: 5000 },
          },
          {
            id: 'apply_load',
            name: 'Apply Load Torque',
            type: 'control',
            instrumentType: 'motor_emulator',
            parameters: { action: 'set_torque', torque: '{{loadTorque}}' },
            continueOnFail: true,
          },
          {
            id: 'wait_load_settle',
            name: 'Wait for Load Settlement',
            type: 'wait',
            parameters: { duration_ms: 2000 },
          },
          {
            id: 'capture_waveforms',
            name: 'Capture Phase Currents',
            type: 'capture',
            instrumentType: 'oscilloscope',
            parameters: { duration_ms: '{{measurementDuration}}', channels: [1, 2, 3, 4], highRes: true },
          },
          {
            id: 'measure_rms_currents',
            name: 'Measure RMS Phase Currents',
            type: 'measure',
            instrumentType: 'oscilloscope',
            parameters: { channels: [1, 2, 3], measurement: 'rms' },
            expectedResults: [
              { measurement: 'rms', channel: '1', operator: 'lt', value: '{{ratedCurrent}}', unit: 'A', isCritical: true },
              { measurement: 'rms', channel: '2', operator: 'lt', value: '{{ratedCurrent}}', unit: 'A', isCritical: true },
              { measurement: 'rms', channel: '3', operator: 'lt', value: '{{ratedCurrent}}', unit: 'A', isCritical: true },
            ],
          },
          {
            id: 'measure_thd',
            name: 'Measure Current THD',
            type: 'measure',
            instrumentType: 'oscilloscope',
            parameters: { channels: [1, 2, 3], measurement: 'thd', fundamentalFreq: '{{targetSpeed * polePairs / 60}}' },
            expectedResults: [
              { measurement: 'thd', operator: 'lt', value: 10, unit: '%', description: 'THD should be less than 10%' },
            ],
          },
          {
            id: 'measure_input_power',
            name: 'Measure Input Power',
            type: 'measure',
            instrumentType: 'power_supply',
            parameters: { measurement: 'power' },
          },
          {
            id: 'calculate_efficiency',
            name: 'Calculate Efficiency',
            type: 'script',
            parameters: {
              script: 'calculate_efficiency',
              args: { inputPowerMeasurement: 'input_power', mechanicalPower: '{{targetSpeed * loadTorque * 2 * 3.14159 / 60}}' },
            },
            expectedResults: [
              { measurement: 'efficiency', operator: 'gt', value: 80, unit: '%', description: 'Efficiency should be greater than 80%' },
            ],
          },
        ],
        teardownSteps: [
          {
            id: 'remove_load',
            name: 'Remove Load',
            type: 'control',
            instrumentType: 'motor_emulator',
            parameters: { action: 'set_torque', torque: 0 },
            continueOnFail: true,
          },
          {
            id: 'stop_motor',
            name: 'Stop Motor',
            type: 'script',
            parameters: { script: 'motor_stop' },
          },
          {
            id: 'disable_psu',
            name: 'Disable Power Supply',
            type: 'control',
            instrumentType: 'power_supply',
            parameters: { action: 'disable_output' },
          },
        ],
        globalParameters: {},
      },
      passCriteria: {
        minPassPercentage: 90,
        criticalMeasurements: ['rms_current', 'thd', 'efficiency'],
        failFast: false,
        allowedWarnings: 3,
      },
    },
    {
      id: 'pwm_analysis_test',
      name: 'PWM Analysis Test',
      description: 'Captures and analyzes PWM signals including duty cycle, dead time, and switching characteristics',
      testType: 'pwm_analysis',
      category: 'foc',
      estimatedDurationMs: 20000,
      timeoutMs: 45000,
      tags: ['pwm', 'dead-time', 'switching', 'gate-drive'],
      templateVariables: {
        dcBusVoltage: { name: 'dcBusVoltage', type: 'number', label: 'DC Bus Voltage', unit: 'V', defaultValue: 24, min: 6, max: 60 },
        targetSpeed: { name: 'targetSpeed', type: 'number', label: 'Target Speed', unit: 'RPM', defaultValue: 1000, min: 100, max: 10000 },
        pwmFrequency: { name: 'pwmFrequency', type: 'number', label: 'Expected PWM Frequency', unit: 'Hz', defaultValue: 20000, min: 1000, max: 100000 },
        expectedDeadTime: { name: 'expectedDeadTime', type: 'number', label: 'Expected Dead Time', unit: 'ns', defaultValue: 500, min: 50, max: 5000 },
      },
      sequenceConfig: {
        instrumentRequirements: [
          { instrumentType: 'logic_analyzer', capabilities: ['digital_channels', 'protocol_decode'], minChannels: 6, description: 'For PWM capture' },
          { instrumentType: 'oscilloscope', capabilities: ['4_channels', 'high_bandwidth'], optional: true, description: 'For analog verification' },
          { instrumentType: 'power_supply', capabilities: ['voltage_control'], description: 'DC bus power' },
        ],
        setupSteps: [
          {
            id: 'setup_psu',
            name: 'Configure Power Supply',
            type: 'configure',
            instrumentType: 'power_supply',
            parameters: { voltage: '{{dcBusVoltage}}', currentLimit: 2, enabled: false },
          },
          {
            id: 'setup_logic',
            name: 'Configure Logic Analyzer',
            type: 'configure',
            instrumentType: 'logic_analyzer',
            parameters: {
              sampleRate: 100000000,
              channels: [
                { channel: 0, label: 'PWM_AH', trigger: true },
                { channel: 1, label: 'PWM_AL' },
                { channel: 2, label: 'PWM_BH' },
                { channel: 3, label: 'PWM_BL' },
                { channel: 4, label: 'PWM_CH' },
                { channel: 5, label: 'PWM_CL' },
              ],
              trigger: { channel: 0, edge: 'rising' },
            },
          },
        ],
        steps: [
          {
            id: 'enable_psu',
            name: 'Enable Power Supply',
            type: 'control',
            instrumentType: 'power_supply',
            parameters: { action: 'enable_output' },
          },
          {
            id: 'start_motor',
            name: 'Start Motor',
            type: 'script',
            parameters: { script: 'motor_start', args: { targetSpeed: '{{targetSpeed}}', rampTime_ms: 1000 } },
          },
          {
            id: 'wait_stable',
            name: 'Wait for Stable PWM',
            type: 'wait',
            parameters: { duration_ms: 2000 },
          },
          {
            id: 'capture_pwm',
            name: 'Capture PWM Signals',
            type: 'capture',
            instrumentType: 'logic_analyzer',
            parameters: { duration_ms: 100, preTrigger_percent: 10 },
          },
          {
            id: 'measure_frequency',
            name: 'Measure PWM Frequency',
            type: 'measure',
            instrumentType: 'logic_analyzer',
            parameters: { channel: 0, measurement: 'frequency' },
            expectedResults: [
              { measurement: 'frequency', operator: 'tolerance', value: '{{pwmFrequency}}', unit: 'Hz', tolerancePercent: 1, isCritical: true, description: 'PWM frequency should match expected' },
            ],
          },
          {
            id: 'measure_dead_time',
            name: 'Measure Dead Time',
            type: 'measure',
            instrumentType: 'logic_analyzer',
            parameters: { channels: [0, 1], measurement: 'dead_time' },
            expectedResults: [
              { measurement: 'dead_time', operator: 'tolerance', value: '{{expectedDeadTime}}', unit: 'ns', tolerancePercent: 20, isCritical: true, description: 'Dead time should match expected' },
              { measurement: 'dead_time', operator: 'gt', value: 100, unit: 'ns', isCritical: true, description: 'Dead time must be positive to prevent shoot-through' },
            ],
          },
          {
            id: 'verify_complementary',
            name: 'Verify Complementary Switching',
            type: 'validate',
            instrumentType: 'logic_analyzer',
            parameters: { validation: 'complementary', channelPairs: [[0, 1], [2, 3], [4, 5]] },
            expectedResults: [
              { measurement: 'complementary', operator: 'eq', value: true, unit: 'bool', isCritical: true, description: 'High/low side must be complementary' },
              { measurement: 'overlap', operator: 'eq', value: 0, unit: 'ns', isCritical: true, description: 'No overlap allowed between high and low side' },
            ],
          },
        ],
        teardownSteps: [
          {
            id: 'stop_motor',
            name: 'Stop Motor',
            type: 'script',
            parameters: { script: 'motor_stop' },
          },
          {
            id: 'disable_psu',
            name: 'Disable Power Supply',
            type: 'control',
            instrumentType: 'power_supply',
            parameters: { action: 'disable_output' },
          },
        ],
        globalParameters: {},
      },
      passCriteria: {
        minPassPercentage: 100,
        criticalMeasurements: ['frequency', 'dead_time', 'complementary', 'overlap'],
        failFast: true,
        allowedWarnings: 0,
      },
    },
    {
      id: 'efficiency_sweep_test',
      name: 'Efficiency Sweep Test',
      description: 'Sweeps motor speed from minimum to maximum while measuring efficiency at each point',
      testType: 'efficiency_sweep',
      category: 'foc',
      estimatedDurationMs: 300000,
      timeoutMs: 600000,
      tags: ['efficiency', 'sweep', 'characterization'],
      templateVariables: {
        dcBusVoltage: { name: 'dcBusVoltage', type: 'number', label: 'DC Bus Voltage', unit: 'V', defaultValue: 24, min: 6, max: 60 },
        minSpeed: { name: 'minSpeed', type: 'number', label: 'Minimum Speed', unit: 'RPM', defaultValue: 500, min: 100, max: 5000 },
        maxSpeed: { name: 'maxSpeed', type: 'number', label: 'Maximum Speed', unit: 'RPM', defaultValue: 5000, min: 500, max: 20000 },
        speedSteps: { name: 'speedSteps', type: 'number', label: 'Number of Speed Steps', defaultValue: 10, min: 3, max: 50 },
        loadTorque: { name: 'loadTorque', type: 'number', label: 'Load Torque', unit: 'Nm', defaultValue: 0.5, min: 0, max: 10 },
        settleTime: { name: 'settleTime', type: 'number', label: 'Settle Time per Point', unit: 'ms', defaultValue: 3000, min: 1000, max: 10000 },
      },
      sequenceConfig: {
        instrumentRequirements: [
          { instrumentType: 'power_supply', capabilities: ['voltage_control', 'measurement'], description: 'DC bus with measurement' },
          { instrumentType: 'oscilloscope', capabilities: ['4_channels'], optional: true, description: 'Current verification' },
          { instrumentType: 'motor_emulator', description: 'For load control' },
        ],
        setupSteps: [
          {
            id: 'setup_psu',
            name: 'Configure Power Supply',
            type: 'configure',
            instrumentType: 'power_supply',
            parameters: { voltage: '{{dcBusVoltage}}', currentLimit: 15, enabled: false },
          },
          {
            id: 'setup_emulator',
            name: 'Configure Motor Emulator',
            type: 'configure',
            instrumentType: 'motor_emulator',
            parameters: { mode: 'torque', torqueLimit: 10 },
          },
        ],
        steps: [
          {
            id: 'enable_psu',
            name: 'Enable Power Supply',
            type: 'control',
            instrumentType: 'power_supply',
            parameters: { action: 'enable_output' },
          },
          {
            id: 'apply_load',
            name: 'Apply Load Torque',
            type: 'control',
            instrumentType: 'motor_emulator',
            parameters: { action: 'set_torque', torque: '{{loadTorque}}' },
          },
          {
            id: 'speed_sweep',
            name: 'Speed Sweep Loop',
            type: 'loop',
            parameters: {
              variable: 'speed',
              start: '{{minSpeed}}',
              end: '{{maxSpeed}}',
              steps: '{{speedSteps}}',
              innerSteps: [
                {
                  id: 'set_speed',
                  name: 'Set Target Speed',
                  type: 'script',
                  parameters: { script: 'motor_set_speed', args: { speed: '{{speed}}' } },
                },
                {
                  id: 'wait_settle',
                  name: 'Wait for Speed Settlement',
                  type: 'wait',
                  parameters: { duration_ms: '{{settleTime}}' },
                },
                {
                  id: 'measure_power',
                  name: 'Measure Input Power',
                  type: 'measure',
                  instrumentType: 'power_supply',
                  parameters: { measurement: 'power' },
                },
                {
                  id: 'measure_speed',
                  name: 'Measure Actual Speed',
                  type: 'script',
                  parameters: { script: 'motor_get_speed' },
                },
                {
                  id: 'calculate_point',
                  name: 'Calculate Efficiency Point',
                  type: 'script',
                  parameters: {
                    script: 'calculate_efficiency_point',
                    args: {
                      targetSpeed: '{{speed}}',
                      torque: '{{loadTorque}}',
                    },
                  },
                },
              ],
            },
          },
          {
            id: 'generate_curve',
            name: 'Generate Efficiency Curve',
            type: 'script',
            parameters: { script: 'generate_efficiency_curve' },
          },
        ],
        teardownSteps: [
          {
            id: 'remove_load',
            name: 'Remove Load',
            type: 'control',
            instrumentType: 'motor_emulator',
            parameters: { action: 'set_torque', torque: 0 },
          },
          {
            id: 'stop_motor',
            name: 'Stop Motor',
            type: 'script',
            parameters: { script: 'motor_stop' },
          },
          {
            id: 'disable_psu',
            name: 'Disable Power Supply',
            type: 'control',
            instrumentType: 'power_supply',
            parameters: { action: 'disable_output' },
          },
        ],
        globalParameters: {},
      },
      passCriteria: {
        minPassPercentage: 80,
        criticalMeasurements: ['min_efficiency'],
        failFast: false,
        allowedWarnings: 5,
      },
    },
  ];
}

/**
 * Apply template variable values to sequence config
 */
function applyTemplateVariables(
  config: any,
  values: Record<string, unknown>
): any {
  const configStr = JSON.stringify(config);
  let result = configStr;

  // Replace {{variable}} patterns with actual values
  for (const [key, value] of Object.entries(values)) {
    const pattern = new RegExp(`{{${key}}}`, 'g');
    result = result.replace(pattern, String(value));

    // Also handle expressions like {{variable * 1.5}}
    const exprPattern = new RegExp(`{{${key}\\s*([*+\\-/])\\s*([\\d.]+)}}`, 'g');
    result = result.replace(exprPattern, (_match, op, num) => {
      const val = Number(value);
      const multiplier = parseFloat(num);
      switch (op) {
        case '*': return String(val * multiplier);
        case '/': return String(val / multiplier);
        case '+': return String(val + multiplier);
        case '-': return String(val - multiplier);
        default: return String(val);
      }
    });
  }

  return JSON.parse(result);
}

export default createHILRoutes;
