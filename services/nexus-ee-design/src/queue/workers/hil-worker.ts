/**
 * EE Design Partner - HIL (Hardware-in-the-Loop) Worker
 *
 * BullMQ worker for processing HIL test jobs including instrument control,
 * waveform capture, and automated test sequence execution.
 */

import { Worker, Job, ConnectionOptions } from 'bullmq';
import { spawn, ChildProcess } from 'child_process';
import * as fs from 'fs/promises';
import * as path from 'path';
import { config } from '../../config.js';
import { log, Logger } from '../../utils/logger.js';
import * as HILTestRunRepository from '../../database/repositories/hil-test-run-repository.js';
import * as HILTestSequenceRepository from '../../database/repositories/hil-test-sequence-repository.js';
import * as HILInstrumentRepository from '../../database/repositories/hil-instrument-repository.js';
import * as HILCapturedDataRepository from '../../database/repositories/hil-captured-data-repository.js';
import * as HILMeasurementRepository from '../../database/repositories/hil-measurement-repository.js';
import {
  getHILWebSocketManager,
  parseHILProgressLine,
} from '../../api/websocket-hil.js';
import type {
  HILTestRun,
  HILTestSequence,
  HILTestStep,
  HILInstrument,
  HILPassCriteria,
  HILTestRunSummary,
  HILTestResult,
  HILMeasurement,
  HILCaptureType,
  HILDataFormat,
} from '../../types/hil-types.js';

// ============================================================================
// Types
// ============================================================================

export interface HILJobData {
  /** Test run ID in database */
  testRunId: string;
  /** Test sequence ID */
  sequenceId: string;
  /** Project ID */
  projectId: string;
  /** WebSocket operation ID */
  operationId: string;
  /** Sequence configuration override */
  sequenceConfig?: {
    steps: HILTestStep[];
    instrumentRequirements: unknown[];
  };
  /** Pass criteria override */
  passCriteria?: HILPassCriteria;
  /** Test conditions */
  testConditions?: Record<string, unknown>;
  /** Parameter overrides */
  parameterOverrides?: Record<string, unknown>;
  /** Timeout in milliseconds */
  timeoutMs?: number;
}

interface HILWorkerContext {
  logger: Logger;
  workDir: string;
  outputDir: string;
  pythonPath: string;
  scriptsDir: string;
}

interface StepResult {
  stepId: string;
  stepName: string;
  passed: boolean;
  measurements: HILMeasurement[];
  captureIds: string[];
  durationMs: number;
  error?: string;
}

// ============================================================================
// Configuration
// ============================================================================

const getRedisConnection = (): ConnectionOptions => ({
  host: config.redis.host,
  port: config.redis.port,
  password: config.redis.password,
  maxRetriesPerRequest: null,
  enableReadyCheck: false,
});

const WORK_DIR = config.storage.tempDir;
const OUTPUT_DIR = config.storage.outputDir;
const HIL_TIMEOUT_MS = 600000; // 10 minutes default

// ============================================================================
// Worker Creation
// ============================================================================

/**
 * Create the HIL worker.
 *
 * @param concurrency - Number of concurrent jobs (default 1 for exclusive hardware)
 */
export function createHILWorker(concurrency: number = 1): Worker<HILJobData> {
  const workerLogger = log.child({ service: 'hil-worker' });

  const context: HILWorkerContext = {
    logger: workerLogger,
    workDir: WORK_DIR,
    outputDir: OUTPUT_DIR,
    pythonPath: config.kicad.pythonPath || 'python3',
    scriptsDir: config.hil.pythonScriptsDir || path.join(process.cwd(), 'python-scripts', 'hil'),
  };

  const worker = new Worker<HILJobData>(
    'hil',
    async (job) => {
      return processHILJob(job, context);
    },
    {
      connection: getRedisConnection(),
      concurrency,
      lockDuration: HIL_TIMEOUT_MS,
      limiter: {
        max: concurrency,
        duration: 1000,
      },
      settings: {
        backoffStrategy: (attemptsMade: number) => {
          // Exponential backoff: 1s, 2s, 4s, 8s, ...
          return Math.min(1000 * Math.pow(2, attemptsMade - 1), 30000);
        },
      },
    }
  );

  // Event handlers
  worker.on('completed', (job) => {
    workerLogger.info('HIL job completed', {
      jobId: job.id,
      testRunId: job.data.testRunId,
    });
  });

  worker.on('failed', (job, err) => {
    workerLogger.error('HIL job failed', err, {
      jobId: job?.id,
      testRunId: job?.data?.testRunId,
      attemptsMade: job?.attemptsMade,
    });
  });

  worker.on('stalled', (jobId) => {
    workerLogger.warn('HIL job stalled', { jobId });
  });

  worker.on('error', (err) => {
    workerLogger.error('Worker error', err);
  });

  workerLogger.info('HIL worker created', { concurrency });

  return worker;
}

// ============================================================================
// Job Processor
// ============================================================================

async function processHILJob(
  job: Job<HILJobData>,
  context: HILWorkerContext
): Promise<HILTestRunSummary> {
  const { logger } = context;
  const { testRunId, sequenceId, projectId, operationId } = job.data;

  logger.info('Processing HIL job', {
    jobId: job.id,
    testRunId,
    sequenceId,
    projectId,
  });

  // Get WebSocket manager for progress reporting
  let wsManager;
  try {
    wsManager = getHILWebSocketManager();
  } catch (e) {
    logger.warn('WebSocket manager not available, progress will not be streamed');
  }

  // Update test run status
  await HILTestRunRepository.updateStatus(testRunId, 'running');
  await HILTestRunRepository.assignWorker(
    testRunId,
    `worker-${process.pid}`,
    process.env.HOSTNAME || 'localhost',
    job.id || 'unknown'
  );

  // Notify start
  if (wsManager) {
    wsManager.startTestRun(operationId, testRunId, projectId);
  }

  try {
    // Get sequence configuration
    const sequence = await HILTestSequenceRepository.findById(sequenceId);
    if (!sequence) {
      throw new Error(`Test sequence ${sequenceId} not found`);
    }

    // Get connected instruments
    const instruments = await HILInstrumentRepository.findConnected(projectId);
    const instrumentMap: Record<string, HILInstrument> = {};
    for (const inst of instruments) {
      instrumentMap[inst.id] = inst;
    }

    // Store instrument snapshot
    await HILTestRunRepository.storeInstrumentSnapshot(testRunId, instrumentMap);

    // Verify instrument requirements
    await verifyInstrumentRequirements(
      sequence.sequenceConfig.instrumentRequirements,
      instruments,
      logger
    );

    // Get configuration
    const steps = job.data.sequenceConfig?.steps || sequence.sequenceConfig.steps;
    const passCriteria = job.data.passCriteria || sequence.passCriteria;
    const totalSteps = steps.length;

    // Create working directory
    const runWorkDir = path.join(context.workDir, `hil_${testRunId}`);
    const runOutputDir = path.join(context.outputDir, 'hil-captures', testRunId);
    await fs.mkdir(runWorkDir, { recursive: true });
    await fs.mkdir(runOutputDir, { recursive: true });

    // Execute test steps
    const stepResults: StepResult[] = [];
    let overallPassed = true;
    let criticalFailure = false;

    for (let i = 0; i < steps.length; i++) {
      const step = steps[i];

      // Check for abort
      if (await job.isFailed()) {
        throw new Error('Job was aborted');
      }

      // Notify step started
      if (wsManager) {
        wsManager.emitTestStepStarted(
          operationId,
          testRunId,
          i,
          totalSteps,
          step.id,
          step.name
        );
      }

      // Update progress
      await HILTestRunRepository.updateProgress(testRunId, {
        progressPercentage: (i / totalSteps) * 100,
        currentStep: step.name,
        currentStepIndex: i,
        totalSteps,
      });

      try {
        // Execute the step
        const result = await executeStep(
          step,
          testRunId,
          instrumentMap,
          {
            ...context,
            workDir: runWorkDir,
            outputDir: runOutputDir,
            operationId,
          },
          job.data.parameterOverrides
        );

        stepResults.push(result);

        // Check pass/fail
        if (!result.passed) {
          overallPassed = false;

          // Check if critical
          const isCritical = result.measurements.some((m) => m.isCritical && !m.passed);
          if (isCritical) {
            criticalFailure = true;
          }

          // Fail fast check
          if (passCriteria.failFast && (criticalFailure || !result.passed)) {
            logger.warn('Fail fast triggered', { stepId: step.id, stepName: step.name });
            if (wsManager) {
              wsManager.emitTestStepCompleted(
                operationId,
                testRunId,
                i,
                totalSteps,
                step.id,
                step.name,
                false,
                result.measurements.length
              );
            }
            break;
          }
        }

        // Notify step completed
        if (wsManager) {
          wsManager.emitTestStepCompleted(
            operationId,
            testRunId,
            i,
            totalSteps,
            step.id,
            step.name,
            result.passed,
            result.measurements.length
          );
        }
      } catch (stepError) {
        const errorMessage =
          stepError instanceof Error ? stepError.message : 'Unknown step error';

        logger.error('Step execution failed', stepError as Error, {
          stepId: step.id,
          stepName: step.name,
        });

        stepResults.push({
          stepId: step.id,
          stepName: step.name,
          passed: false,
          measurements: [],
          captureIds: [],
          durationMs: 0,
          error: errorMessage,
        });

        overallPassed = false;

        if (passCriteria.failFast || !step.continueOnFail) {
          break;
        }
      }
    }

    // Calculate summary
    const allMeasurements = stepResults.flatMap((r) => r.measurements);
    const passedMeasurements = allMeasurements.filter((m) => m.passed);
    const failedMeasurements = allMeasurements.filter((m) => m.passed === false);
    const criticalFailures = failedMeasurements.filter((m) => m.isCritical);

    const passPercentage =
      allMeasurements.length > 0
        ? (passedMeasurements.length / allMeasurements.length) * 100
        : 0;

    // Determine final result
    let result: HILTestResult;
    if (criticalFailures.length > 0) {
      result = 'fail';
    } else if (passPercentage >= passCriteria.minPassPercentage) {
      result = overallPassed ? 'pass' : 'partial';
    } else {
      result = 'fail';
    }

    // Build summary
    const summary: HILTestRunSummary = {
      totalMeasurements: allMeasurements.length,
      passedMeasurements: passedMeasurements.length,
      failedMeasurements: failedMeasurements.length,
      warningMeasurements: allMeasurements.filter((m) => m.isWarning).length,
      criticalFailures: criticalFailures.map(
        (m) => `${m.measurementType}: ${m.failureReason || 'Failed'}`
      ),
      keyMetrics: await HILMeasurementRepository.getKeyMetrics(testRunId),
      score: passPercentage,
    };

    // Calculate duration
    const testRun = await HILTestRunRepository.findById(testRunId);
    const durationMs = testRun?.startedAt
      ? Date.now() - new Date(testRun.startedAt).getTime()
      : 0;

    // Complete the test run
    await HILTestRunRepository.complete(testRunId, result, summary);

    // Notify completion
    if (wsManager) {
      wsManager.emitTestRunCompleted(operationId, testRunId, result, {
        totalMeasurements: summary.totalMeasurements,
        passedMeasurements: summary.passedMeasurements,
        failedMeasurements: summary.failedMeasurements,
        durationMs,
      });
    }

    // Cleanup working directory
    await fs.rm(runWorkDir, { recursive: true, force: true }).catch(() => {});

    logger.info('HIL test run completed', {
      testRunId,
      result,
      passPercentage: passPercentage.toFixed(1),
      durationMs,
    });

    return summary;
  } catch (error) {
    const errorMessage = error instanceof Error ? error.message : 'Unknown error';

    // Get step that failed
    const testRun = await HILTestRunRepository.findById(testRunId);
    const errorStepId = testRun?.currentStep || undefined;

    // Fail the test run
    await HILTestRunRepository.fail(testRunId, errorMessage, { stack: (error as Error)?.stack }, errorStepId);

    // Notify failure
    if (wsManager) {
      wsManager.emitTestRunFailed(operationId, testRunId, errorMessage, errorStepId);
    }

    logger.error('HIL test run failed', error as Error, {
      testRunId,
      sequenceId,
    });

    throw error;
  }
}

// ============================================================================
// Step Execution
// ============================================================================

async function executeStep(
  step: HILTestStep,
  testRunId: string,
  instruments: Record<string, HILInstrument>,
  context: HILWorkerContext & { operationId: string },
  parameterOverrides?: Record<string, unknown>
): Promise<StepResult> {
  const { logger } = context;
  const startTime = Date.now();

  logger.debug('Executing step', {
    stepId: step.id,
    stepName: step.name,
    stepType: step.type,
  });

  // Merge parameters
  const parameters = {
    ...step.parameters,
    ...parameterOverrides,
  };

  // Get target instrument if specified
  let targetInstrument: HILInstrument | undefined;
  if (step.instrumentId) {
    targetInstrument = instruments[step.instrumentId];
  } else if (step.instrumentType) {
    targetInstrument = Object.values(instruments).find(
      (i) => i.instrumentType === step.instrumentType
    );
  }

  const measurements: HILMeasurement[] = [];
  const captureIds: string[] = [];
  let passed = true;
  let error: string | undefined;

  try {
    // Execute based on step type
    switch (step.type) {
      case 'configure':
        await executeConfigureStep(step, targetInstrument, parameters, context);
        break;

      case 'measure':
        const measureResults = await executeMeasureStep(
          step,
          testRunId,
          targetInstrument,
          parameters,
          context
        );
        measurements.push(...measureResults);
        passed = measureResults.every((m) => m.passed !== false);
        break;

      case 'capture':
        const captureResult = await executeCaptureStep(
          step,
          testRunId,
          targetInstrument,
          parameters,
          context
        );
        captureIds.push(captureResult.captureId);
        if (captureResult.measurements) {
          measurements.push(...captureResult.measurements);
        }
        break;

      case 'wait':
        await executeWaitStep(step, parameters);
        break;

      case 'control':
        await executeControlStep(step, targetInstrument, parameters, context);
        break;

      case 'validate':
        const validateResult = await executeValidateStep(
          step,
          testRunId,
          parameters,
          context
        );
        passed = validateResult.passed;
        if (validateResult.measurements) {
          measurements.push(...validateResult.measurements);
        }
        break;

      case 'script':
        const scriptResult = await executeScriptStep(
          step,
          testRunId,
          instruments,
          parameters,
          context
        );
        measurements.push(...scriptResult.measurements);
        passed = scriptResult.passed;
        break;

      default:
        logger.warn('Unknown step type', { stepType: step.type });
    }

    // Validate expected results if specified
    if (step.expectedResults && step.expectedResults.length > 0) {
      const validationResults = await validateExpectedResults(
        step.expectedResults,
        testRunId,
        step.id,
        context
      );
      measurements.push(...validationResults);
      passed = passed && validationResults.every((m) => m.passed !== false);
    }
  } catch (err) {
    passed = false;
    error = err instanceof Error ? err.message : 'Step execution error';
    logger.error('Step execution error', err as Error, { stepId: step.id });
  }

  const durationMs = Date.now() - startTime;

  return {
    stepId: step.id,
    stepName: step.name,
    passed,
    measurements,
    captureIds,
    durationMs,
    error,
  };
}

// ============================================================================
// Step Type Implementations
// ============================================================================

async function executeConfigureStep(
  step: HILTestStep,
  instrument: HILInstrument | undefined,
  parameters: Record<string, unknown>,
  context: HILWorkerContext
): Promise<void> {
  const { logger, pythonPath, scriptsDir } = context;

  if (!instrument) {
    throw new Error('No instrument specified for configure step');
  }

  logger.debug('Configuring instrument', {
    instrumentId: instrument.id,
    instrumentType: instrument.instrumentType,
  });

  // Execute Python configuration script
  await executePythonScript(
    pythonPath,
    path.join(scriptsDir, 'cli.py'),
    [
      'configure',
      '--instrument-type',
      instrument.instrumentType,
      '--connection-params',
      JSON.stringify(instrument.connectionParams),
      '--config',
      JSON.stringify(parameters),
    ],
    context.workDir,
    logger
  );
}

async function executeMeasureStep(
  step: HILTestStep,
  testRunId: string,
  instrument: HILInstrument | undefined,
  parameters: Record<string, unknown>,
  context: HILWorkerContext
): Promise<HILMeasurement[]> {
  const { logger, pythonPath, scriptsDir } = context;

  if (!instrument) {
    throw new Error('No instrument specified for measure step');
  }

  // Execute measurement via Python
  const result = await executePythonScript(
    pythonPath,
    path.join(scriptsDir, 'cli.py'),
    [
      'measure',
      '--instrument-type',
      instrument.instrumentType,
      '--connection-params',
      JSON.stringify(instrument.connectionParams),
      '--measurement-config',
      JSON.stringify(parameters),
    ],
    context.workDir,
    logger
  );

  // Parse measurements from output
  const measurements: HILMeasurement[] = [];

  try {
    const outputLines = result.stdout.split('\n');
    for (const line of outputLines) {
      if (line.startsWith('MEASUREMENT:')) {
        const measurementData = JSON.parse(line.substring(12));

        // Create measurement in database
        const measurement = await HILMeasurementRepository.create({
          testRunId,
          stepId: step.id,
          measurementType: measurementData.type,
          measurementName: measurementData.name,
          channel: measurementData.channel,
          value: measurementData.value,
          unit: measurementData.unit,
          minLimit: measurementData.min_limit,
          maxLimit: measurementData.max_limit,
          nominalValue: measurementData.nominal,
          tolerancePercent: measurementData.tolerance_percent,
          isCritical: measurementData.is_critical,
        });

        measurements.push(measurement);
      }
    }
  } catch (err) {
    logger.warn('Failed to parse measurement output', { error: err });
  }

  return measurements;
}

async function executeCaptureStep(
  step: HILTestStep,
  testRunId: string,
  instrument: HILInstrument | undefined,
  parameters: Record<string, unknown>,
  context: HILWorkerContext & { operationId: string }
): Promise<{ captureId: string; measurements?: HILMeasurement[] }> {
  const { logger, pythonPath, scriptsDir, outputDir, operationId } = context;

  if (!instrument) {
    throw new Error('No instrument specified for capture step');
  }

  // Generate capture filename
  const captureFilename = `capture_${step.id}_${Date.now()}`;
  const capturePath = path.join(outputDir, captureFilename);

  // Execute capture via Python
  const result = await executePythonScript(
    pythonPath,
    path.join(scriptsDir, 'cli.py'),
    [
      'capture',
      '--instrument-type',
      instrument.instrumentType,
      '--connection-params',
      JSON.stringify(instrument.connectionParams),
      '--capture-config',
      JSON.stringify(parameters),
      '--output-path',
      capturePath,
    ],
    context.workDir,
    logger,
    (line) => {
      // Stream waveform chunks via WebSocket
      try {
        const wsManager = getHILWebSocketManager();
        parseHILProgressLine(line, operationId, wsManager);
      } catch (e) {
        // WebSocket not available
      }
    }
  );

  // Create captured data record
  const channelConfig = (parameters.channels as unknown[]) || [];
  const captureType = (parameters.capture_type as HILCaptureType) || 'waveform';
  const capturedData = await HILCapturedDataRepository.create({
    testRunId,
    instrumentId: instrument.id,
    name: step.name,
    captureType,
    stepId: step.id,
    channelConfig: channelConfig.map((ch: unknown) => ({
      name: String((ch as Record<string, unknown>).name || 'CH1'),
      scale: Number((ch as Record<string, unknown>).scale) || 1,
      offset: Number((ch as Record<string, unknown>).offset) || 0,
      unit: String((ch as Record<string, unknown>).unit) || 'V',
    })),
    sampleRateHz: parameters.sample_rate as number,
    durationMs: parameters.duration_ms as number,
    dataFormat: 'json' as HILDataFormat,
    dataPath: capturePath,
  });

  return {
    captureId: capturedData.id,
  };
}

async function executeWaitStep(
  step: HILTestStep,
  parameters: Record<string, unknown>
): Promise<void> {
  const waitMs = (parameters.duration_ms as number) || 1000;
  await new Promise((resolve) => setTimeout(resolve, waitMs));
}

async function executeControlStep(
  step: HILTestStep,
  instrument: HILInstrument | undefined,
  parameters: Record<string, unknown>,
  context: HILWorkerContext
): Promise<void> {
  const { logger, pythonPath, scriptsDir } = context;

  if (!instrument) {
    throw new Error('No instrument specified for control step');
  }

  await executePythonScript(
    pythonPath,
    path.join(scriptsDir, 'cli.py'),
    [
      'control',
      '--instrument-type',
      instrument.instrumentType,
      '--connection-params',
      JSON.stringify(instrument.connectionParams),
      '--control-config',
      JSON.stringify(parameters),
    ],
    context.workDir,
    logger
  );
}

async function executeValidateStep(
  step: HILTestStep,
  testRunId: string,
  parameters: Record<string, unknown>,
  context: HILWorkerContext
): Promise<{ passed: boolean; measurements?: HILMeasurement[] }> {
  const { logger } = context;

  // Get measurements from this test run
  const measurements = await HILMeasurementRepository.findByTestRun(testRunId);

  // Apply validation rules
  const validationRules = parameters.rules as Array<{
    measurement_type: string;
    operator: string;
    value: unknown;
  }>;

  if (!validationRules) {
    return { passed: true };
  }

  let passed = true;

  for (const rule of validationRules) {
    const matchingMeasurements = measurements.filter(
      (m) => m.measurementType === rule.measurement_type
    );

    for (const m of matchingMeasurements) {
      const ruleValue = Number(rule.value);
      let rulePassed = true;

      switch (rule.operator) {
        case 'eq':
          rulePassed = m.value === ruleValue;
          break;
        case 'ne':
          rulePassed = m.value !== ruleValue;
          break;
        case 'gt':
          rulePassed = m.value > ruleValue;
          break;
        case 'gte':
          rulePassed = m.value >= ruleValue;
          break;
        case 'lt':
          rulePassed = m.value < ruleValue;
          break;
        case 'lte':
          rulePassed = m.value <= ruleValue;
          break;
        default:
          logger.warn('Unknown validation operator', { operator: rule.operator });
      }

      if (!rulePassed) {
        passed = false;
      }
    }
  }

  return { passed };
}

async function executeScriptStep(
  step: HILTestStep,
  testRunId: string,
  instruments: Record<string, HILInstrument>,
  parameters: Record<string, unknown>,
  context: HILWorkerContext & { operationId: string }
): Promise<{ passed: boolean; measurements: HILMeasurement[] }> {
  const { logger, pythonPath, scriptsDir, operationId } = context;

  const scriptName = parameters.script as string;
  if (!scriptName) {
    throw new Error('No script specified for script step');
  }

  const scriptPath = path.join(scriptsDir, 'foc_tests', scriptName);

  const result = await executePythonScript(
    pythonPath,
    scriptPath,
    [
      '--test-run-id',
      testRunId,
      '--instruments',
      JSON.stringify(instruments),
      '--parameters',
      JSON.stringify(parameters),
    ],
    context.workDir,
    logger,
    (line) => {
      try {
        const wsManager = getHILWebSocketManager();
        parseHILProgressLine(line, operationId, wsManager);
      } catch (e) {
        // WebSocket not available
      }
    }
  );

  // Parse measurements and result
  const measurements: HILMeasurement[] = [];
  let passed = true;

  const outputLines = result.stdout.split('\n');
  for (const line of outputLines) {
    if (line.startsWith('MEASUREMENT:')) {
      try {
        const data = JSON.parse(line.substring(12));
        const measurement = await HILMeasurementRepository.create({
          testRunId,
          stepId: step.id,
          measurementType: data.type,
          value: data.value,
          unit: data.unit,
          isCritical: data.is_critical,
        });
        measurements.push(measurement);
      } catch (err) {
        logger.warn('Failed to parse measurement', { line });
      }
    } else if (line.startsWith('RESULT:')) {
      try {
        const data = JSON.parse(line.substring(7));
        passed = data.passed;
      } catch (err) {
        logger.warn('Failed to parse result', { line });
      }
    }
  }

  return { passed, measurements };
}

// ============================================================================
// Validation Helpers
// ============================================================================

async function validateExpectedResults(
  expectedResults: unknown[],
  testRunId: string,
  stepId: string,
  context: HILWorkerContext
): Promise<HILMeasurement[]> {
  const measurements: HILMeasurement[] = [];

  // Get measurements from this step
  const stepMeasurements = await HILMeasurementRepository.findByStep(testRunId, stepId);

  for (const expected of expectedResults as Array<{
    measurement: string;
    operator: string;
    value: unknown;
    unit: string;
    is_critical?: boolean;
  }>) {
    const matching = stepMeasurements.find(
      (m) => m.measurementType === expected.measurement
    );

    if (matching) {
      // Already evaluated in create
      measurements.push(matching);
    }
  }

  return measurements;
}

async function verifyInstrumentRequirements(
  requirements: unknown[],
  instruments: HILInstrument[],
  logger: Logger
): Promise<void> {
  for (const req of requirements as Array<{
    instrumentType: string;
    optional?: boolean;
  }>) {
    const found = instruments.find(
      (i) => i.instrumentType === req.instrumentType && i.status === 'connected'
    );

    if (!found && !req.optional) {
      throw new Error(
        `Required instrument type ${req.instrumentType} not connected`
      );
    }

    if (!found && req.optional) {
      logger.warn('Optional instrument not available', {
        instrumentType: req.instrumentType,
      });
    }
  }
}

// ============================================================================
// Python Execution
// ============================================================================

interface ScriptResult {
  stdout: string;
  stderr: string;
  exitCode: number;
}

async function executePythonScript(
  pythonPath: string,
  scriptPath: string,
  args: string[],
  workDir: string,
  logger: Logger,
  onStdoutLine?: (line: string) => void
): Promise<ScriptResult> {
  return new Promise((resolve, reject) => {
    const proc = spawn(pythonPath, [scriptPath, ...args], {
      cwd: workDir,
      env: {
        ...process.env,
        PYTHONUNBUFFERED: '1',
      },
    });

    let stdout = '';
    let stderr = '';

    proc.stdout.on('data', (data: Buffer) => {
      const text = data.toString();
      stdout += text;

      if (onStdoutLine) {
        const lines = text.split('\n');
        for (const line of lines) {
          if (line.trim()) {
            onStdoutLine(line);
          }
        }
      }
    });

    proc.stderr.on('data', (data: Buffer) => {
      stderr += data.toString();
    });

    proc.on('error', (err) => {
      reject(err);
    });

    proc.on('close', (code) => {
      if (code !== 0) {
        logger.error('Python script failed', new Error(stderr || `Exit code ${code}`), {
          script: scriptPath,
          exitCode: code,
        });
        reject(new Error(`Script exited with code ${code}: ${stderr}`));
      } else {
        resolve({
          stdout,
          stderr,
          exitCode: code || 0,
        });
      }
    });
  });
}

export default createHILWorker;
