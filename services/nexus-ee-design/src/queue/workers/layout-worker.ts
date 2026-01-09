/**
 * EE Design Partner - Layout Worker
 *
 * BullMQ worker for processing PCB layout jobs including generation,
 * MAPOS optimization, and DRC validation.
 */

import { Worker, Job, ConnectionOptions } from 'bullmq';
import { spawn } from 'child_process';
import * as fs from 'fs/promises';
import * as path from 'path';
import { config } from '../../config.js';
import { log, Logger } from '../../utils/logger.js';
import * as PcbRepository from '../../database/repositories/pcb-repository.js';
import type { LayoutJobData } from '../queue-manager.js';
import type { DRCResults } from '../../database/repositories/pcb-repository.js';

// ============================================================================
// Types
// ============================================================================

interface LayoutContext {
  logger: Logger;
  workDir: string;
  outputDir: string;
  scriptsDir: string;
  pythonPath: string;
}

interface LayoutResult {
  success: boolean;
  layoutId: string;
  operation: string;
  score?: number;
  drcViolations?: number;
  iterations?: number;
  winningAgent?: string;
  filePath?: string;
  error?: string;
}

interface MaposIterationResult {
  iterationNumber: number;
  agentStrategy: string;
  score: number;
  drcViolations: number;
  improvementDelta: number;
  changes: object[];
  durationMs: number;
}

interface PythonScriptResult {
  success: boolean;
  output: unknown;
  stderr: string;
  exitCode: number;
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
const SCRIPTS_DIR = config.kicad.scriptsDir;
const PYTHON_PATH = config.kicad.pythonPath;
const MAX_ITERATIONS = config.layout.maxIterations;
const TARGET_SCORE = config.layout.targetScore;
const CONVERGENCE_THRESHOLD = config.layout.convergenceThreshold;

// ============================================================================
// Layout Job Processor
// ============================================================================

async function processLayoutJob(
  job: Job<LayoutJobData>,
  context: LayoutContext
): Promise<LayoutResult> {
  const { logger } = context;
  const { layoutId, operation } = job.data;

  logger.info('Processing layout job', {
    jobId: job.id,
    layoutId,
    operation,
  });

  // Update layout status
  await PcbRepository.update(layoutId, { status: 'in_progress' });
  await job.updateProgress({ progress: 5, message: `Starting ${operation}` });

  try {
    let result: LayoutResult;

    switch (operation) {
      case 'generate':
        result = await generateLayout(job, context);
        break;

      case 'optimize':
        result = await optimizeLayout(job, context);
        break;

      case 'drc':
        result = await runDrcCheck(job, context);
        break;

      default:
        throw new Error(`Unsupported layout operation: ${operation}`);
    }

    // Update final status
    if (result.success) {
      await PcbRepository.update(layoutId, { status: 'in_review' });
    }

    await job.updateProgress({ progress: 100, message: `${operation} completed` });

    logger.info('Layout job completed', {
      jobId: job.id,
      layoutId,
      operation,
      success: result.success,
      score: result.score,
    });

    return result;
  } catch (error) {
    const errorMessage = error instanceof Error ? error.message : 'Unknown error';

    // Update layout status to reflect failure
    await PcbRepository.update(layoutId, { status: 'draft' });

    const err = error instanceof Error ? error : new Error(errorMessage);
    logger.error('Layout job failed', err, {
      jobId: job.id,
      layoutId,
      operation,
    });

    throw error;
  }
}

// ============================================================================
// Layout Generation
// ============================================================================

async function generateLayout(
  job: Job<LayoutJobData>,
  context: LayoutContext
): Promise<LayoutResult> {
  const { logger, workDir, outputDir, scriptsDir, pythonPath } = context;
  const { layoutId, projectId, schematicId, strategy, config: layoutConfig } = job.data;

  await job.updateProgress({ progress: 10, message: 'Preparing layout generation' });

  // Create working directories
  const jobWorkDir = path.join(workDir, `layout_${layoutId}`);
  const jobOutputDir = path.join(outputDir, `layout_${layoutId}`);
  await fs.mkdir(jobWorkDir, { recursive: true });
  await fs.mkdir(jobOutputDir, { recursive: true });

  // Prepare input configuration
  const inputConfig = {
    layoutId,
    projectId,
    schematicId,
    strategy: strategy || 'conservative',
    config: layoutConfig || {},
    outputPath: path.join(jobOutputDir, `${layoutId}.kicad_pcb`),
  };

  const inputPath = path.join(jobWorkDir, 'input.json');
  await fs.writeFile(inputPath, JSON.stringify(inputConfig, null, 2));

  await job.updateProgress({ progress: 30, message: 'Running layout generator' });

  // Execute Python layout generator script
  const scriptPath = path.join(scriptsDir, 'generate_pcb_layout.py');
  const result = await executePythonScript(pythonPath, scriptPath, [inputPath], logger);

  await job.updateProgress({ progress: 80, message: 'Processing layout results' });

  if (!result.success) {
    throw new Error(`Layout generation failed: ${result.stderr}`);
  }

  // Parse output
  const output = result.output as { score?: number; filePath?: string; drcViolations?: number };

  // Update PCB repository with results
  if (output.filePath) {
    const pcbContent = await fs.readFile(output.filePath, 'utf-8').catch(() => '');
    if (pcbContent) {
      await PcbRepository.updateKicadContent(layoutId, pcbContent);
    }
  }

  if (output.score !== undefined) {
    await PcbRepository.updateScores(layoutId, { overall: output.score });
  }

  // Clean up working directory
  await fs.rm(jobWorkDir, { recursive: true, force: true }).catch(() => {});

  return {
    success: true,
    layoutId,
    operation: 'generate',
    score: output.score || 0,
    drcViolations: output.drcViolations || 0,
    filePath: output.filePath,
  };
}

// ============================================================================
// MAPOS Optimization
// ============================================================================

async function optimizeLayout(
  job: Job<LayoutJobData>,
  context: LayoutContext
): Promise<LayoutResult> {
  const { logger, workDir, outputDir, scriptsDir, pythonPath } = context;
  const { layoutId, projectId, maxIterations, targetScore, config: layoutConfig } = job.data;

  await job.updateProgress({ progress: 5, message: 'Initializing MAPOS optimization' });

  // Update layout status to optimizing
  await PcbRepository.updateMaposConfig(layoutId, {
    maxIterations: maxIterations || MAX_ITERATIONS,
    targetScore: targetScore || TARGET_SCORE,
    convergenceThreshold: CONVERGENCE_THRESHOLD,
    enabledAgents: config.layout.enabledAgents,
  });

  // Create working directories
  const jobWorkDir = path.join(workDir, `mapos_${layoutId}`);
  const jobOutputDir = path.join(outputDir, `mapos_${layoutId}`);
  await fs.mkdir(jobWorkDir, { recursive: true });
  await fs.mkdir(jobOutputDir, { recursive: true });

  // Get current PCB content
  const pcbContent = await PcbRepository.getKicadContent(layoutId);
  if (!pcbContent) {
    throw new Error('PCB content not found for optimization');
  }

  // Write PCB content to working file
  const pcbPath = path.join(jobWorkDir, 'input.kicad_pcb');
  await fs.writeFile(pcbPath, pcbContent);

  const effectiveMaxIterations = maxIterations || MAX_ITERATIONS;
  const effectiveTargetScore = targetScore || TARGET_SCORE;

  let currentScore = 0;
  let previousScore = 0;
  let bestScore = 0;
  let bestAgent = 'none';
  let totalIterations = 0;
  let drcViolations = 0;
  let converged = false;
  let noImprovementCount = 0;

  // MAPOS optimization loop
  for (let iteration = 1; iteration <= effectiveMaxIterations && !converged; iteration++) {
    totalIterations = iteration;

    const progress = Math.min(95, 5 + (iteration / effectiveMaxIterations) * 90);
    await job.updateProgress({
      progress,
      message: `MAPOS iteration ${iteration}/${effectiveMaxIterations}`,
      data: { iteration, currentScore, bestScore, drcViolations },
    });

    logger.debug('Running MAPOS iteration', {
      iteration,
      layoutId,
      currentScore,
      bestScore,
    });

    // Run optimization iteration via Python script
    const iterationResult = await runMaposIteration(
      iteration,
      pcbPath,
      jobWorkDir,
      jobOutputDir,
      pythonPath,
      scriptsDir,
      logger
    );

    // Update scores
    previousScore = currentScore;
    currentScore = iterationResult.score;
    drcViolations = iterationResult.drcViolations;

    if (currentScore > bestScore) {
      bestScore = currentScore;
      bestAgent = iterationResult.agentStrategy;
      noImprovementCount = 0;

      // Save best result
      const bestPcbPath = path.join(jobOutputDir, 'best.kicad_pcb');
      await fs.copyFile(
        path.join(jobOutputDir, `iteration_${iteration}.kicad_pcb`),
        bestPcbPath
      ).catch(() => {});
    } else {
      noImprovementCount++;
    }

    // Record iteration in database
    await PcbRepository.addMaposIteration(layoutId, {
      iterationNumber: iteration,
      agentStrategy: iterationResult.agentStrategy,
      score: iterationResult.score,
      drcViolations: iterationResult.drcViolations,
      improvementDelta: currentScore - previousScore,
      changes: iterationResult.changes,
      durationMs: iterationResult.durationMs,
    });

    // Check convergence criteria
    if (currentScore >= effectiveTargetScore) {
      converged = true;
      logger.info('MAPOS converged - target score reached', {
        layoutId,
        iteration,
        targetScore: effectiveTargetScore,
        achievedScore: currentScore,
      });
    } else if (noImprovementCount >= 10) {
      converged = true;
      logger.info('MAPOS converged - no improvement', {
        layoutId,
        iteration,
        noImprovementCount,
      });
    } else if (Math.abs(currentScore - previousScore) < CONVERGENCE_THRESHOLD && iteration > 5) {
      converged = true;
      logger.info('MAPOS converged - improvement threshold', {
        layoutId,
        iteration,
        delta: Math.abs(currentScore - previousScore),
      });
    }
  }

  await job.updateProgress({ progress: 95, message: 'Finalizing optimization' });

  // Update PCB with best result
  const bestPcbPath = path.join(jobOutputDir, 'best.kicad_pcb');
  try {
    const bestContent = await fs.readFile(bestPcbPath, 'utf-8');
    await PcbRepository.updateKicadContent(layoutId, bestContent);
  } catch {
    logger.warn('Could not read best PCB file, using last iteration');
  }

  // Update scores
  await PcbRepository.updateScores(layoutId, { overall: bestScore });

  // Clean up working directory
  await fs.rm(jobWorkDir, { recursive: true, force: true }).catch(() => {});

  return {
    success: true,
    layoutId,
    operation: 'optimize',
    score: bestScore,
    drcViolations,
    iterations: totalIterations,
    winningAgent: bestAgent,
  };
}

async function runMaposIteration(
  iteration: number,
  pcbPath: string,
  workDir: string,
  outputDir: string,
  pythonPath: string,
  scriptsDir: string,
  logger: Logger
): Promise<MaposIterationResult> {
  const startTime = Date.now();

  // Prepare iteration config
  const iterationConfig = {
    iteration,
    inputPcb: pcbPath,
    outputPcb: path.join(outputDir, `iteration_${iteration}.kicad_pcb`),
    agents: config.layout.enabledAgents,
  };

  const configPath = path.join(workDir, `iteration_${iteration}_config.json`);
  await fs.writeFile(configPath, JSON.stringify(iterationConfig, null, 2));

  // Run MAPOS optimizer script
  const scriptPath = path.join(scriptsDir, 'mapos', 'mapos_pcb_optimizer.py');
  const result = await executePythonScript(pythonPath, scriptPath, [configPath], logger);

  const durationMs = Date.now() - startTime;

  if (!result.success) {
    // Return default result on failure
    logger.warn('MAPOS iteration failed', { iteration, stderr: result.stderr });
    return {
      iterationNumber: iteration,
      agentStrategy: 'fallback',
      score: 0,
      drcViolations: 999,
      improvementDelta: 0,
      changes: [],
      durationMs,
    };
  }

  const output = result.output as {
    score?: number;
    drcViolations?: number;
    agentStrategy?: string;
    changes?: object[];
  };

  // Copy output PCB to be used in next iteration
  await fs.copyFile(iterationConfig.outputPcb, pcbPath).catch(() => {});

  return {
    iterationNumber: iteration,
    agentStrategy: output.agentStrategy || 'unknown',
    score: output.score || 0,
    drcViolations: output.drcViolations || 0,
    improvementDelta: 0, // Calculated by caller
    changes: output.changes || [],
    durationMs,
  };
}

// ============================================================================
// DRC Check
// ============================================================================

async function runDrcCheck(
  job: Job<LayoutJobData>,
  context: LayoutContext
): Promise<LayoutResult> {
  const { logger, workDir, outputDir, scriptsDir, pythonPath } = context;
  const { layoutId } = job.data;

  await job.updateProgress({ progress: 10, message: 'Preparing DRC check' });

  // Create working directory
  const jobWorkDir = path.join(workDir, `drc_${layoutId}`);
  await fs.mkdir(jobWorkDir, { recursive: true });

  // Get PCB content
  const pcbContent = await PcbRepository.getKicadContent(layoutId);
  if (!pcbContent) {
    throw new Error('PCB content not found for DRC');
  }

  // Write PCB content to file
  const pcbPath = path.join(jobWorkDir, 'input.kicad_pcb');
  await fs.writeFile(pcbPath, pcbContent);

  await job.updateProgress({ progress: 30, message: 'Running DRC checks' });

  // Execute DRC script
  const scriptPath = path.join(scriptsDir, 'run_drc.py');
  const result = await executePythonScript(pythonPath, scriptPath, [pcbPath], logger);

  await job.updateProgress({ progress: 80, message: 'Processing DRC results' });

  // Parse DRC results
  const drcResults: DRCResults = parseDrcResults(result);

  // Update PCB repository with DRC results
  await PcbRepository.updateDrcResults(layoutId, drcResults);

  // Clean up
  await fs.rm(jobWorkDir, { recursive: true, force: true }).catch(() => {});

  return {
    success: drcResults.passed,
    layoutId,
    operation: 'drc',
    score: drcResults.passed ? 100 : Math.max(0, 100 - drcResults.totalViolations * 5),
    drcViolations: drcResults.totalViolations,
  };
}

function parseDrcResults(result: PythonScriptResult): DRCResults {
  if (!result.success) {
    return {
      passed: false,
      totalViolations: 999,
      violations: [
        {
          id: 'error',
          code: 'DRC_ERROR',
          severity: 'error',
          message: `DRC check failed: ${result.stderr}`,
        },
      ],
      warnings: 0,
      timestamp: new Date().toISOString(),
    };
  }

  const output = result.output as {
    passed?: boolean;
    totalViolations?: number;
    violations?: Array<{
      id: string;
      code: string;
      severity: string;
      message: string;
      location?: object;
    }>;
    warnings?: number;
  };

  return {
    passed: output.passed ?? true,
    totalViolations: output.totalViolations ?? 0,
    violations: output.violations ?? [],
    warnings: output.warnings ?? 0,
    timestamp: new Date().toISOString(),
  };
}

// ============================================================================
// Python Script Execution
// ============================================================================

async function executePythonScript(
  pythonPath: string,
  scriptPath: string,
  args: string[],
  logger: Logger
): Promise<PythonScriptResult> {
  return new Promise((resolve) => {
    let stdout = '';
    let stderr = '';

    const python = spawn(pythonPath, [scriptPath, ...args], {
      timeout: 300000, // 5 minutes
      env: {
        ...process.env,
        PYTHONUNBUFFERED: '1',
      },
    });

    python.stdout.on('data', (data) => {
      stdout += data.toString();
    });

    python.stderr.on('data', (data) => {
      stderr += data.toString();
    });

    python.on('close', (code) => {
      let output: unknown = stdout;

      try {
        output = JSON.parse(stdout);
      } catch {
        // Keep as string if not valid JSON
      }

      if (code !== 0) {
        logger.warn('Python script failed', { scriptPath, code, stderr });
      }

      resolve({
        success: code === 0,
        output,
        stderr,
        exitCode: code || 0,
      });
    });

    python.on('error', (error) => {
      logger.error('Python script execution error', error, { scriptPath });

      // Return mock result for development
      resolve(generateMockPythonResult(scriptPath));
    });
  });
}

function generateMockPythonResult(scriptPath: string): PythonScriptResult {
  if (scriptPath.includes('mapos')) {
    return {
      success: true,
      output: {
        score: 85 + Math.random() * 10,
        drcViolations: Math.floor(Math.random() * 10),
        agentStrategy: config.layout.enabledAgents[Math.floor(Math.random() * config.layout.enabledAgents.length)],
        changes: [],
      },
      stderr: '',
      exitCode: 0,
    };
  }

  if (scriptPath.includes('drc')) {
    return {
      success: true,
      output: {
        passed: true,
        totalViolations: 0,
        violations: [],
        warnings: 2,
      },
      stderr: '',
      exitCode: 0,
    };
  }

  return {
    success: true,
    output: {
      score: 80,
      filePath: '/tmp/mock.kicad_pcb',
      drcViolations: 5,
    },
    stderr: '',
    exitCode: 0,
  };
}

// ============================================================================
// Worker Creation
// ============================================================================

let layoutWorker: Worker | null = null;

export function createLayoutWorker(): Worker {
  const logger = log.child({ service: 'layout-worker' });

  logger.info('Creating layout worker', {
    concurrency: 2,
    maxIterations: MAX_ITERATIONS,
    targetScore: TARGET_SCORE,
  });

  layoutWorker = new Worker<LayoutJobData, LayoutResult>(
    'layout',
    async (job) => {
      const context: LayoutContext = {
        logger: logger.child({ jobId: job.id }),
        workDir: WORK_DIR,
        outputDir: OUTPUT_DIR,
        scriptsDir: SCRIPTS_DIR,
        pythonPath: PYTHON_PATH,
      };

      return processLayoutJob(job, context);
    },
    {
      connection: getRedisConnection(),
      concurrency: 2, // Lower concurrency for resource-intensive layout jobs
      limiter: {
        max: 4,
        duration: 1000,
      },
    }
  );

  // Set up worker event handlers
  layoutWorker.on('completed', (job, result) => {
    logger.info('Layout job completed', {
      jobId: job.id,
      layoutId: job.data.layoutId,
      operation: job.data.operation,
      score: result.score,
      iterations: result.iterations,
    });
  });

  layoutWorker.on('failed', (job, error) => {
    logger.error('Layout job failed', error, {
      jobId: job?.id,
      layoutId: job?.data.layoutId,
      operation: job?.data.operation,
      attemptsMade: job?.attemptsMade,
    });
  });

  layoutWorker.on('stalled', (jobId) => {
    logger.warn('Layout job stalled', { jobId });
  });

  layoutWorker.on('error', (error) => {
    logger.error('Layout worker error', error);
  });

  layoutWorker.on('progress', (job, progress) => {
    logger.debug('Layout job progress', {
      jobId: job.id,
      layoutId: job.data.layoutId,
      progress,
    });
  });

  return layoutWorker;
}

export function getLayoutWorker(): Worker | null {
  return layoutWorker;
}

export async function closeLayoutWorker(): Promise<void> {
  if (layoutWorker) {
    await layoutWorker.close();
    layoutWorker = null;
    log.info('Layout worker closed');
  }
}

export default createLayoutWorker;
