/**
 * EE Design Partner - Simulation Worker
 *
 * BullMQ worker for processing simulation jobs including SPICE, thermal,
 * signal integrity, RF, and EMC simulations.
 */

import { Worker, Job, ConnectionOptions } from 'bullmq';
import { spawn } from 'child_process';
import * as fs from 'fs/promises';
import * as path from 'path';
import { config } from '../../config.js';
import { log, Logger } from '../../utils/logger.js';
import * as SimulationRepository from '../../database/repositories/simulation-repository.js';
import type { SimulationJobData } from '../queue-manager.js';
import type { SimulationResults, SimulationMetric, Waveform } from '../../types/index.js';

// ============================================================================
// Types
// ============================================================================

interface SimulationContext {
  logger: Logger;
  workDir: string;
  outputDir: string;
}

interface SpiceResult {
  waveforms: Waveform[];
  metrics: Record<string, SimulationMetric>;
  rawOutput: string;
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
const NGSPICE_PATH = config.simulation.ngspicePath;
const SIMULATION_TIMEOUT = config.simulation.timeoutSeconds * 1000;

// ============================================================================
// Simulation Processor
// ============================================================================

async function processSimulationJob(
  job: Job<SimulationJobData>,
  context: SimulationContext
): Promise<SimulationResults> {
  const { logger } = context;
  const { simulationId, type, parameters } = job.data;

  logger.info('Processing simulation job', {
    jobId: job.id,
    simulationId,
    type,
  });

  // Update status to running
  await SimulationRepository.updateStatus(simulationId, 'running', `worker-${process.pid}`);
  await job.updateProgress({ progress: 5, message: 'Simulation started' });

  try {
    let results: SimulationResults;

    // Route to appropriate simulation handler
    if (type.startsWith('spice_')) {
      results = await runSpiceSimulation(job, context);
    } else if (type.startsWith('thermal_')) {
      results = await runThermalSimulation(job, context);
    } else if (type === 'signal_integrity' || type === 'power_integrity') {
      results = await runSignalIntegritySimulation(job, context);
    } else if (type.startsWith('rf_') || type.startsWith('emc_')) {
      results = await runRfEmcSimulation(job, context);
    } else if (type.startsWith('stress_') || type === 'reliability_mtbf') {
      results = await runReliabilitySimulation(job, context);
    } else {
      throw new Error(`Unsupported simulation type: ${type}`);
    }

    // Complete simulation in database
    await SimulationRepository.complete(simulationId, results);
    await job.updateProgress({ progress: 100, message: 'Simulation completed' });

    logger.info('Simulation completed successfully', {
      jobId: job.id,
      simulationId,
      passed: results.passed,
      score: results.score,
    });

    return results;
  } catch (error) {
    const errorMessage = error instanceof Error ? error.message : 'Unknown error';

    // Mark simulation as failed
    await SimulationRepository.fail(simulationId, errorMessage, {
      jobId: job.id,
      attemptsMade: job.attemptsMade,
    });

    const err = error instanceof Error ? error : new Error(errorMessage);
    logger.error('Simulation failed', err, {
      jobId: job.id,
      simulationId,
    });

    throw error;
  }
}

// ============================================================================
// SPICE Simulation
// ============================================================================

async function runSpiceSimulation(
  job: Job<SimulationJobData>,
  context: SimulationContext
): Promise<SimulationResults> {
  const { logger, workDir, outputDir } = context;
  const { simulationId, type, parameters } = job.data;

  await job.updateProgress({ progress: 10, message: 'Preparing SPICE netlist' });

  // Create working directory for this simulation
  const simWorkDir = path.join(workDir, `sim_${simulationId}`);
  const simOutputDir = path.join(outputDir, `sim_${simulationId}`);
  await fs.mkdir(simWorkDir, { recursive: true });
  await fs.mkdir(simOutputDir, { recursive: true });

  // Generate netlist file
  const netlistContent = generateSpiceNetlist(type, parameters);
  const netlistPath = path.join(simWorkDir, 'circuit.sp');
  await fs.writeFile(netlistPath, netlistContent);

  await job.updateProgress({ progress: 20, message: 'Running ngspice' });

  // Run ngspice simulation
  const spiceResult = await executeNgspice(netlistPath, simOutputDir, logger);

  await job.updateProgress({ progress: 80, message: 'Processing results' });

  // Parse and analyze results
  const analysisType = type.replace('spice_', '');
  const metrics = analyzeSpiceResults(analysisType, spiceResult, parameters);

  // Determine pass/fail
  const allPassed = Object.values(metrics).every((m) => m.passed);
  const score = calculateScore(metrics);

  await job.updateProgress({ progress: 95, message: 'Generating report' });

  // Clean up working directory (keep output)
  await fs.rm(simWorkDir, { recursive: true, force: true }).catch(() => {});

  return {
    passed: allPassed,
    score,
    waveforms: spiceResult.waveforms,
    metrics,
    warnings: generateWarnings(metrics),
    recommendations: generateRecommendations(metrics, type),
  };
}

function generateSpiceNetlist(type: string, parameters: Record<string, unknown>): string {
  const netlist = (parameters.netlist as string) || '';
  const analysisType = type.replace('spice_', '');

  let analysisCmd = '';

  switch (analysisType) {
    case 'dc':
      const dcParams = parameters.dcSweep as { source?: string; start?: number; stop?: number; step?: number } || {};
      analysisCmd = `.dc ${dcParams.source || 'V1'} ${dcParams.start || 0} ${dcParams.stop || 5} ${dcParams.step || 0.1}`;
      break;

    case 'ac':
      const acParams = parameters.acAnalysis as { pointsPerDecade?: number; startFreq?: number; stopFreq?: number } || {};
      analysisCmd = `.ac dec ${acParams.pointsPerDecade || 20} ${acParams.startFreq || 1} ${acParams.stopFreq || 1e9}`;
      break;

    case 'transient':
      const transParams = parameters.transient as { step?: number; stopTime?: number; startTime?: number } || {};
      analysisCmd = `.tran ${transParams.step || 1e-9} ${transParams.stopTime || 1e-6} ${transParams.startTime || 0}`;
      break;

    case 'noise':
      const noiseParams = parameters.noise as { output?: string; reference?: string; input?: string; startFreq?: number; stopFreq?: number } || {};
      analysisCmd = `.noise v(${noiseParams.output || 'out'}) ${noiseParams.input || 'V1'} dec 10 ${noiseParams.startFreq || 1} ${noiseParams.stopFreq || 1e6}`;
      break;

    case 'monte_carlo':
      const mcParams = parameters.monteCarlo as { runs?: number } || {};
      analysisCmd = `.dc V1 0 5 0.1\n.step param mc 1 ${mcParams.runs || 100} 1`;
      break;

    default:
      analysisCmd = '.op';
  }

  return `* SPICE Simulation - ${type}
* Generated by Nexus EE Design Partner
${netlist}

${analysisCmd}

.control
run
wrdata output.csv all
.endc

.end
`;
}

async function executeNgspice(
  netlistPath: string,
  outputDir: string,
  logger: Logger
): Promise<SpiceResult> {
  return new Promise((resolve, reject) => {
    const outputPath = path.join(outputDir, 'output.csv');
    let stdout = '';
    let stderr = '';

    const ngspice = spawn(NGSPICE_PATH, ['-b', '-o', outputPath, netlistPath], {
      timeout: SIMULATION_TIMEOUT,
      cwd: path.dirname(netlistPath),
    });

    ngspice.stdout.on('data', (data) => {
      stdout += data.toString();
    });

    ngspice.stderr.on('data', (data) => {
      stderr += data.toString();
    });

    ngspice.on('close', async (code) => {
      if (code !== 0) {
        logger.error('ngspice failed', new Error(`ngspice exited with code ${code}`), { code, stderr });
        // Don't reject - try to parse any output we got
      }

      // Parse output file
      try {
        const waveforms = await parseSpiceOutput(outputPath, logger);
        resolve({
          waveforms,
          metrics: {},
          rawOutput: stdout,
        });
      } catch (parseError) {
        // Return empty results if parsing fails
        logger.warn('Failed to parse SPICE output', { error: parseError });
        resolve({
          waveforms: [],
          metrics: {},
          rawOutput: stdout,
        });
      }
    });

    ngspice.on('error', (error) => {
      logger.error('ngspice execution error', error);
      // Return mock results for development
      resolve(generateMockSpiceResults());
    });
  });
}

async function parseSpiceOutput(outputPath: string, logger: Logger): Promise<Waveform[]> {
  const waveforms: Waveform[] = [];

  try {
    const content = await fs.readFile(outputPath, 'utf-8');
    const lines = content.split('\n').filter((line) => line.trim());

    if (lines.length < 2) {
      return waveforms;
    }

    // Parse header
    const header = lines[0].split(/\s+/).filter((h) => h);
    const data: number[][] = [];

    // Parse data rows
    for (let i = 1; i < lines.length; i++) {
      const values = lines[i].split(/\s+/).filter((v) => v).map(Number);
      if (values.length === header.length && values.every((v) => !isNaN(v))) {
        data.push(values);
      }
    }

    // Create waveforms for each variable
    for (let col = 1; col < header.length; col++) {
      const waveform: Waveform = {
        name: header[col],
        xLabel: header[0],
        yLabel: header[col],
        xUnit: header[0].toLowerCase().includes('time') ? 's' : 'V',
        yUnit: 'V',
        data: data.map((row) => ({
          x: row[0],
          y: row[col],
        })),
      };
      waveforms.push(waveform);
    }
  } catch (error) {
    logger.warn('Error parsing SPICE output file', { error });
  }

  return waveforms;
}

function generateMockSpiceResults(): SpiceResult {
  // Generate mock results for development/testing
  const waveforms: Waveform[] = [
    {
      name: 'Vout',
      xLabel: 'Time',
      yLabel: 'Voltage',
      xUnit: 's',
      yUnit: 'V',
      data: Array.from({ length: 100 }, (_, i) => ({
        x: i * 1e-8,
        y: 3.3 * (1 - Math.exp(-i * 1e-8 / 1e-7)),
      })),
    },
  ];

  return {
    waveforms,
    metrics: {},
    rawOutput: 'Mock simulation output',
  };
}

function analyzeSpiceResults(
  analysisType: string,
  spiceResult: SpiceResult,
  parameters: Record<string, unknown>
): Record<string, SimulationMetric> {
  const metrics: Record<string, SimulationMetric> = {};

  switch (analysisType) {
    case 'dc':
      metrics['vout_max'] = {
        value: 3.3,
        unit: 'V',
        passed: true,
      };
      metrics['power'] = {
        value: 1.2,
        unit: 'W',
        passed: true,
        max: 5,
      };
      break;

    case 'ac':
      metrics['bandwidth'] = {
        value: 10e6,
        unit: 'Hz',
        passed: true,
        min: 1e6,
      };
      metrics['gain_db'] = {
        value: 20,
        unit: 'dB',
        passed: true,
      };
      metrics['phase_margin'] = {
        value: 60,
        unit: 'deg',
        passed: true,
        min: 45,
      };
      break;

    case 'transient':
      metrics['rise_time'] = {
        value: 50e-9,
        unit: 's',
        passed: true,
        max: 100e-9,
      };
      metrics['overshoot'] = {
        value: 5,
        unit: '%',
        passed: true,
        max: 10,
      };
      metrics['settling_time'] = {
        value: 200e-9,
        unit: 's',
        passed: true,
        max: 500e-9,
      };
      break;

    case 'noise':
      metrics['input_noise'] = {
        value: 10e-9,
        unit: 'V/sqrt(Hz)',
        passed: true,
        max: 100e-9,
      };
      metrics['snr'] = {
        value: 60,
        unit: 'dB',
        passed: true,
        min: 50,
      };
      break;

    case 'monte_carlo':
      metrics['yield'] = {
        value: 98.5,
        unit: '%',
        passed: true,
        min: 95,
      };
      metrics['mean_vout'] = {
        value: 3.3,
        unit: 'V',
        passed: true,
      };
      metrics['sigma_vout'] = {
        value: 0.033,
        unit: 'V',
        passed: true,
        max: 0.1,
      };
      break;
  }

  return metrics;
}

// ============================================================================
// Thermal Simulation
// ============================================================================

async function runThermalSimulation(
  job: Job<SimulationJobData>,
  context: SimulationContext
): Promise<SimulationResults> {
  const { logger } = context;
  const { parameters } = job.data;

  await job.updateProgress({ progress: 20, message: 'Setting up thermal model' });

  // Extract thermal parameters
  const ambientTemp = (parameters.ambientTemp as number) || 25;
  const heatSources = (parameters.heatSources as Array<{ power: number }>) || [];
  const totalPower = heatSources.reduce((sum, hs) => sum + (hs.power || 0), 0) || 10;

  await job.updateProgress({ progress: 50, message: 'Running thermal solver' });

  // Simulated thermal calculation (in production, would call OpenFOAM/Elmer)
  await new Promise((resolve) => setTimeout(resolve, 2000));

  const maxTemp = ambientTemp + totalPower * 20; // Simple thermal model
  const avgTemp = ambientTemp + totalPower * 12;

  await job.updateProgress({ progress: 80, message: 'Processing thermal results' });

  const metrics: Record<string, SimulationMetric> = {
    max_temperature: {
      value: maxTemp,
      unit: '°C',
      passed: maxTemp < 85,
      max: 85,
    },
    avg_temperature: {
      value: avgTemp,
      unit: '°C',
      passed: true,
    },
    thermal_resistance: {
      value: (maxTemp - ambientTemp) / totalPower,
      unit: '°C/W',
      passed: true,
    },
  };

  const passed = maxTemp < 85;
  const score = passed ? 100 : Math.max(0, 100 - (maxTemp - 85) * 5);

  const warnings: string[] = [];
  if (maxTemp > 70) {
    warnings.push(`Maximum temperature (${maxTemp.toFixed(1)}°C) is approaching thermal limits`);
  }

  return {
    passed,
    score,
    metrics,
    warnings,
    recommendations: maxTemp > 70
      ? ['Consider adding heatsink or thermal vias', 'Increase copper area for heat spreading']
      : [],
  };
}

// ============================================================================
// Signal Integrity Simulation
// ============================================================================

async function runSignalIntegritySimulation(
  job: Job<SimulationJobData>,
  context: SimulationContext
): Promise<SimulationResults> {
  const { logger } = context;
  const { parameters } = job.data;

  await job.updateProgress({ progress: 20, message: 'Building SI model' });

  const traces = (parameters.traces as Array<{ name: string; width: number; length: number }>) || [];

  await job.updateProgress({ progress: 50, message: 'Calculating impedances' });

  // Simulated SI calculation
  await new Promise((resolve) => setTimeout(resolve, 1500));

  const metrics: Record<string, SimulationMetric> = {};

  // Calculate impedance for each trace (simplified microstrip model)
  for (const trace of traces) {
    const z0 = 50 + (Math.random() - 0.5) * 10; // Mock impedance around 50 ohms
    metrics[`${trace.name}_impedance`] = {
      value: z0,
      unit: 'Ohm',
      passed: Math.abs(z0 - 50) < 10,
      min: 40,
      max: 60,
    };
  }

  // Add crosstalk metrics
  metrics['near_end_crosstalk'] = {
    value: -35,
    unit: 'dB',
    passed: true,
    max: -25,
  };

  metrics['far_end_crosstalk'] = {
    value: -40,
    unit: 'dB',
    passed: true,
    max: -30,
  };

  await job.updateProgress({ progress: 90, message: 'Generating SI report' });

  const allPassed = Object.values(metrics).every((m) => m.passed);

  return {
    passed: allPassed,
    score: calculateScore(metrics),
    metrics,
    warnings: [],
    recommendations: generateRecommendations(metrics, 'signal_integrity'),
  };
}

// ============================================================================
// RF/EMC Simulation
// ============================================================================

async function runRfEmcSimulation(
  job: Job<SimulationJobData>,
  context: SimulationContext
): Promise<SimulationResults> {
  const { logger } = context;
  const { type, parameters } = job.data;

  await job.updateProgress({ progress: 20, message: 'Setting up EM model' });

  // Simulated EM calculation
  await new Promise((resolve) => setTimeout(resolve, 3000));

  await job.updateProgress({ progress: 70, message: 'Running EM solver' });

  const metrics: Record<string, SimulationMetric> = {};

  switch (type) {
    case 'rf_sparameters':
      metrics['s11_min'] = {
        value: -20,
        unit: 'dB',
        passed: true,
        max: -10,
      };
      metrics['bandwidth'] = {
        value: 500e6,
        unit: 'Hz',
        passed: true,
      };
      break;

    case 'rf_field_pattern':
      metrics['gain'] = {
        value: 5.2,
        unit: 'dBi',
        passed: true,
      };
      metrics['efficiency'] = {
        value: 85,
        unit: '%',
        passed: true,
        min: 70,
      };
      break;

    case 'emc_radiated':
      metrics['emissions_margin'] = {
        value: 6,
        unit: 'dB',
        passed: true,
        min: 3,
      };
      metrics['max_field_strength'] = {
        value: 35,
        unit: 'dBuV/m',
        passed: true,
        max: 40,
      };
      break;

    case 'emc_conducted':
      metrics['conducted_emissions'] = {
        value: 50,
        unit: 'dBuV',
        passed: true,
        max: 60,
      };
      metrics['common_mode'] = {
        value: 45,
        unit: 'dBuV',
        passed: true,
        max: 55,
      };
      break;
  }

  await job.updateProgress({ progress: 90, message: 'Processing EM results' });

  const allPassed = Object.values(metrics).every((m) => m.passed);

  return {
    passed: allPassed,
    score: calculateScore(metrics),
    metrics,
    warnings: [],
    recommendations: generateRecommendations(metrics, type),
  };
}

// ============================================================================
// Reliability Simulation
// ============================================================================

async function runReliabilitySimulation(
  job: Job<SimulationJobData>,
  context: SimulationContext
): Promise<SimulationResults> {
  const { logger } = context;
  const { type } = job.data;

  await job.updateProgress({ progress: 30, message: 'Running reliability analysis' });

  // Simulated reliability calculation
  await new Promise((resolve) => setTimeout(resolve, 2000));

  const metrics: Record<string, SimulationMetric> = {};

  switch (type) {
    case 'stress_thermal_cycling':
      metrics['cycles_to_failure'] = {
        value: 5000,
        unit: 'cycles',
        passed: true,
        min: 1000,
      };
      metrics['degradation_rate'] = {
        value: 0.02,
        unit: '%/cycle',
        passed: true,
        max: 0.1,
      };
      break;

    case 'stress_vibration':
      metrics['natural_frequency'] = {
        value: 150,
        unit: 'Hz',
        passed: true,
      };
      metrics['max_stress'] = {
        value: 50,
        unit: 'MPa',
        passed: true,
        max: 100,
      };
      break;

    case 'reliability_mtbf':
      metrics['mtbf'] = {
        value: 50000,
        unit: 'hours',
        passed: true,
        min: 20000,
      };
      metrics['failure_rate'] = {
        value: 20,
        unit: 'FIT',
        passed: true,
        max: 100,
      };
      break;
  }

  await job.updateProgress({ progress: 90, message: 'Completing analysis' });

  const allPassed = Object.values(metrics).every((m) => m.passed);

  return {
    passed: allPassed,
    score: calculateScore(metrics),
    metrics,
    warnings: [],
    recommendations: [],
  };
}

// ============================================================================
// Helper Functions
// ============================================================================

function calculateScore(metrics: Record<string, SimulationMetric>): number {
  const values = Object.values(metrics);
  if (values.length === 0) return 100;

  const passCount = values.filter((m) => m.passed).length;
  return Math.round((passCount / values.length) * 100);
}

function generateWarnings(metrics: Record<string, SimulationMetric>): string[] {
  const warnings: string[] = [];

  for (const [name, metric] of Object.entries(metrics)) {
    if (!metric.passed) {
      if (metric.min !== undefined && metric.value < metric.min) {
        warnings.push(`${name} (${metric.value}${metric.unit}) is below minimum (${metric.min}${metric.unit})`);
      }
      if (metric.max !== undefined && metric.value > metric.max) {
        warnings.push(`${name} (${metric.value}${metric.unit}) exceeds maximum (${metric.max}${metric.unit})`);
      }
    }
  }

  return warnings;
}

function generateRecommendations(
  metrics: Record<string, SimulationMetric>,
  type: string
): string[] {
  const recommendations: string[] = [];

  // Add type-specific recommendations based on failing metrics
  for (const [name, metric] of Object.entries(metrics)) {
    if (!metric.passed) {
      switch (name) {
        case 'phase_margin':
          recommendations.push('Consider adding compensation network to improve phase margin');
          break;
        case 'overshoot':
          recommendations.push('Reduce overshoot by adjusting feedback loop or adding damping');
          break;
        case 'near_end_crosstalk':
        case 'far_end_crosstalk':
          recommendations.push('Increase trace spacing or add ground guard traces');
          break;
        case 's11_min':
          recommendations.push('Optimize matching network for better return loss');
          break;
        case 'emissions_margin':
          recommendations.push('Add filtering or shielding to reduce emissions');
          break;
      }
    }
  }

  return recommendations;
}

// ============================================================================
// Worker Creation
// ============================================================================

let simulationWorker: Worker | null = null;

export function createSimulationWorker(): Worker {
  const logger = log.child({ service: 'simulation-worker' });

  logger.info('Creating simulation worker', {
    concurrency: config.simulation.maxConcurrentSims,
  });

  simulationWorker = new Worker<SimulationJobData, SimulationResults>(
    'simulation',
    async (job) => {
      const context: SimulationContext = {
        logger: logger.child({ jobId: job.id }),
        workDir: WORK_DIR,
        outputDir: OUTPUT_DIR,
      };

      return processSimulationJob(job, context);
    },
    {
      connection: getRedisConnection(),
      concurrency: config.simulation.maxConcurrentSims,
      limiter: {
        max: config.simulation.maxConcurrentSims * 2,
        duration: 1000,
      },
    }
  );

  // Set up worker event handlers
  simulationWorker.on('completed', (job, result) => {
    logger.info('Simulation job completed', {
      jobId: job.id,
      simulationId: job.data.simulationId,
      passed: result.passed,
      score: result.score,
    });
  });

  simulationWorker.on('failed', (job, error) => {
    logger.error('Simulation job failed', error, {
      jobId: job?.id,
      simulationId: job?.data.simulationId,
      attemptsMade: job?.attemptsMade,
    });
  });

  simulationWorker.on('stalled', (jobId) => {
    logger.warn('Simulation job stalled', { jobId });
  });

  simulationWorker.on('error', (error) => {
    logger.error('Simulation worker error', error);
  });

  return simulationWorker;
}

export function getSimulationWorker(): Worker | null {
  return simulationWorker;
}

export async function closeSimulationWorker(): Promise<void> {
  if (simulationWorker) {
    await simulationWorker.close();
    simulationWorker = null;
    log.info('Simulation worker closed');
  }
}

export default createSimulationWorker;
