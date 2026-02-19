/**
 * Python Executor Service
 *
 * Executes KiCad automation Python scripts for schematic and PCB operations.
 * Manages Python virtual environment and script execution with proper error handling.
 */

import { spawn, ChildProcess, exec } from 'child_process';
import { promisify } from 'util';
import * as path from 'path';
import * as fs from 'fs/promises';
import { existsSync } from 'fs';
import { EventEmitter } from 'events';
import { v4 as uuidv4 } from 'uuid';
import { fileURLToPath } from 'url';
import { log as logger } from '../../utils/logger.js';

// ESM-compatible __dirname
const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const execAsync = promisify(exec);

export interface PythonExecutorConfig {
  pythonPath?: string;
  scriptsDir: string;
  workDir: string;
  timeout: number;
  maxConcurrent: number;
}

/**
 * SIGTERM→SIGKILL grace period (ms).
 * When killing a process, send SIGTERM first, then SIGKILL after this delay.
 */
const SIGKILL_GRACE_MS = 30000;

export interface ScriptResult {
  success: boolean;
  stdout: string;
  stderr: string;
  exitCode: number;
  duration: number;
  output?: any;
}

/**
 * Progress event from Python PROGRESS: lines
 */
export interface ProgressEvent {
  type: string;
  operationId: string;
  timestamp: string;
  progress_percentage: number;
  current_step: string;
  phase?: string;
  phase_progress?: number;
  data?: Record<string, unknown>;
  error_message?: string;
  error_code?: string;
  [key: string]: unknown;
}

/**
 * Callback for progress events
 */
export type ProgressCallback = (event: ProgressEvent) => void;

export interface ScriptJob {
  id: string;
  script: string;
  args: string[];
  status: 'pending' | 'running' | 'completed' | 'failed' | 'timeout';
  result?: ScriptResult;
  startTime?: Date;
  endTime?: Date;
}

// Get the scripts directory relative to the project root
const getScriptsDir = (): string => {
  // In development, we're in src/services/pcb
  // In production, we're in dist/services/pcb
  const possiblePaths = [
    path.join(__dirname, '../../../../python-scripts'),
    path.join(__dirname, '../../../python-scripts'),
    path.resolve(process.cwd(), 'python-scripts'),
    path.resolve(process.cwd(), 'services/nexus-ee-design/python-scripts'),
  ];

  for (const p of possiblePaths) {
    if (existsSync(p)) {
      return p;
    }
  }
  return possiblePaths[0]; // Default to first option
};

const DEFAULT_CONFIG: PythonExecutorConfig = {
  scriptsDir: getScriptsDir(),
  workDir: '/tmp/nexus-ee-design',
  timeout: 120000, // 2 minutes default
  maxConcurrent: 4
};

/**
 * Python script executor for KiCad automation
 */
export class PythonExecutor extends EventEmitter {
  private config: PythonExecutorConfig;
  private jobs: Map<string, ScriptJob> = new Map();
  private runningJobs: number = 0;
  private pythonVersion: string = '';
  private initialized: boolean = false;

  constructor(config?: Partial<PythonExecutorConfig>) {
    super();
    this.config = { ...DEFAULT_CONFIG, ...config };
  }

  /**
   * Initialize the executor - verify Python and dependencies
   */
  async initialize(): Promise<void> {
    if (this.initialized) return;

    try {
      // Log configuration
      logger.info(`Python executor scriptsDir: ${this.config.scriptsDir}`);

      // Check Python availability
      const pythonPath = await this.findPython();
      this.config.pythonPath = pythonPath;

      // Get Python version
      const { stdout } = await execAsync(`${pythonPath} --version`);
      this.pythonVersion = stdout.trim();
      logger.info(`Python executor initialized: ${this.pythonVersion}, scriptsDir: ${this.config.scriptsDir}`);

      // Ensure work directory exists
      await fs.mkdir(this.config.workDir, { recursive: true });

      // Verify scripts directory
      const scriptsExist = await this.checkScriptsDirectory();
      if (!scriptsExist) {
        logger.warn('Scripts directory not found or empty, creating default scripts');
        await this.createDefaultScripts();
      }

      // Check for required packages
      await this.checkDependencies();

      this.initialized = true;
      this.emit('initialized', { pythonPath, version: this.pythonVersion });
    } catch (error) {
      logger.error('Failed to initialize Python executor:', error);
      throw new Error(`Python executor initialization failed: ${error instanceof Error ? error.message : 'Unknown error'}`);
    }
  }

  /**
   * Find Python executable - prioritize venv in scripts directory
   */
  private async findPython(): Promise<string> {
    // First, check for venv in scripts directory (this has all the dependencies)
    const venvPaths = [
      path.join(this.config.scriptsDir, 'venv', 'bin', 'python'),
      path.join(this.config.scriptsDir, 'venv', 'bin', 'python3'),
      path.join(this.config.scriptsDir, '.venv', 'bin', 'python'),
      path.join(this.config.scriptsDir, '.venv', 'bin', 'python3'),
    ];

    for (const venvPath of venvPaths) {
      try {
        await fs.access(venvPath);
        await execAsync(`${venvPath} --version`);
        logger.info(`Using venv Python: ${venvPath}`);
        return venvPath;
      } catch {
        continue;
      }
    }

    // Fall back to system Python
    const candidates = ['python3', 'python', '/usr/bin/python3', '/usr/local/bin/python3'];

    for (const candidate of candidates) {
      try {
        await execAsync(`${candidate} --version`);
        logger.warn(`Using system Python (no venv found): ${candidate}`);
        return candidate;
      } catch {
        continue;
      }
    }

    throw new Error('Python 3 not found. Please install Python 3.8 or later.');
  }

  /**
   * Check scripts directory
   */
  private async checkScriptsDirectory(): Promise<boolean> {
    try {
      const files = await fs.readdir(this.config.scriptsDir);
      return files.some(f => f.endsWith('.py'));
    } catch {
      return false;
    }
  }

  /**
   * Create default script stubs if not present
   */
  private async createDefaultScripts(): Promise<void> {
    await fs.mkdir(this.config.scriptsDir, { recursive: true });

    // Create a simple health check script
    const healthCheck = `#!/usr/bin/env python3
"""Health check script for Python executor."""
import sys
import json

def main():
    result = {
        "status": "healthy",
        "python_version": sys.version,
        "platform": sys.platform
    }
    print(json.dumps(result))
    return 0

if __name__ == "__main__":
    sys.exit(main())
`;
    await fs.writeFile(path.join(this.config.scriptsDir, 'health_check.py'), healthCheck);
  }

  /**
   * Check for required Python packages
   */
  private async checkDependencies(): Promise<void> {
    const requiredPackages = ['sexpdata', 'json', 'pathlib'];
    const { pythonPath } = this.config;

    for (const pkg of requiredPackages) {
      try {
        if (pkg === 'json' || pkg === 'pathlib') continue; // Built-in modules

        await execAsync(`${pythonPath} -c "import ${pkg}"`);
      } catch {
        logger.warn(`Optional package '${pkg}' not installed. Some features may be limited.`);
      }
    }
  }

  /**
   * Execute a Python script with progress streaming support.
   * Parses PROGRESS:{json} lines from stdout and calls the progress callback.
   */
  async executeWithProgress(
    script: string,
    args: string[] = [],
    options?: {
      timeout?: number;
      inactivityTimeout?: number; // Kill only after N ms of silence (for streaming pipelines)
      cwd?: string;
      env?: Record<string, string>;
      onProgress?: ProgressCallback;
      stdin?: string; // Optional stdin data (e.g., JSON payload)
    }
  ): Promise<ScriptResult> {
    if (!this.initialized) {
      await this.initialize();
    }

    const jobId = uuidv4();
    const job: ScriptJob = {
      id: jobId,
      script,
      args,
      status: 'pending'
    };
    this.jobs.set(jobId, job);

    while (this.runningJobs >= this.config.maxConcurrent) {
      await new Promise(resolve => setTimeout(resolve, 100));
    }

    this.runningJobs++;
    job.status = 'running';
    job.startTime = new Date();
    this.emit('job-start', { jobId, script, args });

    try {
      const result = await this.runScriptWithProgress(script, args, {
        timeout: options?.timeout || this.config.timeout,
        inactivityTimeout: options?.inactivityTimeout,
        cwd: options?.cwd || this.config.workDir,
        env: options?.env,
        onProgress: options?.onProgress,
        stdin: options?.stdin
      });

      job.status = result.success ? 'completed' : 'failed';
      job.result = result;
      job.endTime = new Date();

      this.emit('job-complete', { jobId, result });
      return result;
    } catch (error) {
      job.status = 'failed';
      job.endTime = new Date();

      const result: ScriptResult = {
        success: false,
        stdout: '',
        stderr: error instanceof Error ? error.message : 'Unknown error',
        exitCode: -1,
        duration: job.startTime ? Date.now() - job.startTime.getTime() : 0
      };
      job.result = result;

      this.emit('job-error', { jobId, error });
      return result;
    } finally {
      this.runningJobs--;
    }
  }

  /**
   * Run a Python script with progress streaming
   */
  private runScriptWithProgress(
    script: string,
    args: string[],
    options: {
      timeout: number;
      inactivityTimeout?: number;
      cwd: string;
      env?: Record<string, string>;
      onProgress?: ProgressCallback;
      stdin?: string;
    }
  ): Promise<ScriptResult> {
    return new Promise((resolve, reject) => {
      const startTime = Date.now();
      const scriptPath = path.isAbsolute(script)
        ? script
        : path.join(this.config.scriptsDir, script);

      const pythonArgs = [scriptPath, ...args];
      let stdout = '';
      let stderr = '';
      let lineBuffer = '';
      let settled = false;

      logger.info(`Executing Python script with progress: ${this.config.pythonPath} ${scriptPath}`);

      const env = {
        ...process.env,
        ...options.env,
        PYTHONUNBUFFERED: '1'
      };

      const proc: ChildProcess = spawn(this.config.pythonPath!, pythonArgs, {
        cwd: options.cwd,
        env,
        stdio: ['pipe', 'pipe', 'pipe']
      });

      // Write stdin data if provided (e.g., for --stdin flag to avoid E2BIG errors)
      if (options.stdin && proc.stdin) {
        proc.stdin.write(options.stdin);
        proc.stdin.end();
      }

      // --- Graceful kill helper: SIGTERM first, SIGKILL after grace period ---
      const gracefulKill = (reason: string): void => {
        if (settled) return;
        settled = true;
        logger.warn(`Killing Python process: ${reason}`);
        proc.kill('SIGTERM');
        const escalation = setTimeout(() => {
          try { proc.kill('SIGKILL'); } catch { /* already dead */ }
        }, SIGKILL_GRACE_MS);
        // Don't let the escalation timer keep Node alive
        escalation.unref();
        reject(new Error(reason));
      };

      // --- Timeout strategy ---
      // When inactivityTimeout is set, use a watchdog that resets on stdout activity.
      // Otherwise, use the traditional fixed timeout (for short scripts like DRC/Gerber).
      let watchdogHandle: ReturnType<typeof setTimeout> | null = null;
      let fixedTimeoutHandle: ReturnType<typeof setTimeout> | null = null;

      const useWatchdog = typeof options.inactivityTimeout === 'number' && options.inactivityTimeout > 0;

      const resetWatchdog = (): void => {
        if (!useWatchdog) return;
        if (watchdogHandle) clearTimeout(watchdogHandle);
        watchdogHandle = setTimeout(() => {
          gracefulKill(
            `Script killed: no stdout activity for ${options.inactivityTimeout}ms (inactivity watchdog)`
          );
        }, options.inactivityTimeout!);
      };

      if (useWatchdog) {
        // Start initial watchdog
        resetWatchdog();
        logger.info(`Using activity watchdog: inactivityTimeout=${options.inactivityTimeout}ms`);
      } else {
        // Fixed timeout (legacy behavior for short scripts)
        fixedTimeoutHandle = setTimeout(() => {
          gracefulKill(`Script execution timed out after ${options.timeout}ms`);
        }, options.timeout);
      }

      const clearAllTimers = (): void => {
        if (watchdogHandle) clearTimeout(watchdogHandle);
        if (fixedTimeoutHandle) clearTimeout(fixedTimeoutHandle);
      };

      proc.stdout?.on('data', (data: Buffer) => {
        const chunk = data.toString();
        stdout += chunk;

        // Parse line by line for PROGRESS: events
        lineBuffer += chunk;
        const lines = lineBuffer.split('\n');

        // Keep the last incomplete line in buffer
        lineBuffer = lines.pop() || '';

        for (const line of lines) {
          // Reset watchdog on every non-empty line (activity detected)
          if (line.trim()) {
            resetWatchdog();
          }

          if (line.startsWith('PROGRESS:')) {
            try {
              const jsonStr = line.slice(9).trim();
              const event = JSON.parse(jsonStr) as ProgressEvent;
              if (options.onProgress) {
                options.onProgress(event);
              }
              this.emit('progress', { event });
            } catch (parseError) {
              logger.warn('Failed to parse progress line', { line: line.substring(0, 100) });
            }
          }
        }

        this.emit('stdout', { data: chunk });
      });

      proc.stderr?.on('data', (data: Buffer) => {
        stderr += data.toString();
        // stderr activity also resets the watchdog (Python logs go to stderr)
        resetWatchdog();
        this.emit('stderr', { data: data.toString() });
      });

      proc.on('close', (code: number | null) => {
        clearAllTimers();
        settled = true;
        const duration = Date.now() - startTime;

        // Process any remaining buffered line
        if (lineBuffer.startsWith('PROGRESS:')) {
          try {
            const jsonStr = lineBuffer.slice(9).trim();
            const event = JSON.parse(jsonStr) as ProgressEvent;
            if (options.onProgress) {
              options.onProgress(event);
            }
          } catch {
            // Ignore parse errors on final buffer
          }
        }

        let output: any;
        try {
          // Filter out PROGRESS: lines for JSON parsing
          const jsonLines = stdout.split('\n')
            .filter(line => !line.startsWith('PROGRESS:'))
            .join('\n')
            .trim();
          if (jsonLines) {
            output = JSON.parse(jsonLines);
          }
        } catch {
          output = stdout.trim();
        }

        resolve({
          success: code === 0,
          stdout: stdout.trim(),
          stderr: stderr.trim(),
          exitCode: code || 0,
          duration,
          output
        });
      });

      proc.on('error', (error: Error) => {
        clearAllTimers();
        settled = true;
        reject(error);
      });
    });
  }

  /**
   * Execute a Python script
   */
  async execute(
    script: string,
    args: string[] = [],
    options?: {
      timeout?: number;
      cwd?: string;
      env?: Record<string, string>;
    }
  ): Promise<ScriptResult> {
    if (!this.initialized) {
      await this.initialize();
    }

    const jobId = uuidv4();
    const job: ScriptJob = {
      id: jobId,
      script,
      args,
      status: 'pending'
    };
    this.jobs.set(jobId, job);

    // Wait if too many concurrent jobs
    while (this.runningJobs >= this.config.maxConcurrent) {
      await new Promise(resolve => setTimeout(resolve, 100));
    }

    this.runningJobs++;
    job.status = 'running';
    job.startTime = new Date();
    this.emit('job-start', { jobId, script, args });

    try {
      const result = await this.runScript(script, args, {
        timeout: options?.timeout || this.config.timeout,
        cwd: options?.cwd || this.config.workDir,
        env: options?.env
      });

      job.status = result.success ? 'completed' : 'failed';
      job.result = result;
      job.endTime = new Date();

      this.emit('job-complete', { jobId, result });
      return result;
    } catch (error) {
      job.status = 'failed';
      job.endTime = new Date();

      const result: ScriptResult = {
        success: false,
        stdout: '',
        stderr: error instanceof Error ? error.message : 'Unknown error',
        exitCode: -1,
        duration: job.startTime ? Date.now() - job.startTime.getTime() : 0
      };
      job.result = result;

      this.emit('job-error', { jobId, error });
      return result;
    } finally {
      this.runningJobs--;
    }
  }

  /**
   * Run a Python script
   */
  private runScript(
    script: string,
    args: string[],
    options: { timeout: number; cwd: string; env?: Record<string, string> }
  ): Promise<ScriptResult> {
    return new Promise((resolve, reject) => {
      const startTime = Date.now();
      const scriptPath = path.isAbsolute(script)
        ? script
        : path.join(this.config.scriptsDir, script);

      const pythonArgs = [scriptPath, ...args];
      let stdout = '';
      let stderr = '';

      logger.info(`Executing Python script: ${this.config.pythonPath} ${scriptPath}`);

      const env = {
        ...process.env,
        ...options.env,
        PYTHONUNBUFFERED: '1'
      };

      const proc: ChildProcess = spawn(this.config.pythonPath!, pythonArgs, {
        cwd: options.cwd,
        env,
        stdio: ['pipe', 'pipe', 'pipe']
      });

      // Set timeout
      const timeoutHandle = setTimeout(() => {
        proc.kill('SIGKILL');
        reject(new Error(`Script execution timed out after ${options.timeout}ms`));
      }, options.timeout);

      proc.stdout?.on('data', (data: Buffer) => {
        stdout += data.toString();
        this.emit('stdout', { data: data.toString() });
      });

      proc.stderr?.on('data', (data: Buffer) => {
        stderr += data.toString();
        this.emit('stderr', { data: data.toString() });
      });

      proc.on('close', (code: number | null) => {
        clearTimeout(timeoutHandle);
        const duration = Date.now() - startTime;

        let output: any;
        try {
          // Try to parse JSON output
          output = JSON.parse(stdout.trim());
        } catch {
          output = stdout.trim();
        }

        resolve({
          success: code === 0,
          stdout: stdout.trim(),
          stderr: stderr.trim(),
          exitCode: code || 0,
          duration,
          output
        });
      });

      proc.on('error', (error: Error) => {
        clearTimeout(timeoutHandle);
        reject(error);
      });
    });
  }

  /**
   * Parse a KiCad schematic file
   */
  async parseSchematic(filePath: string): Promise<any> {
    const result = await this.execute('parse_schematic.py', [filePath]);
    if (!result.success) {
      throw new Error(`Failed to parse schematic: ${result.stderr}`);
    }
    return result.output;
  }

  /**
   * Generate a KiCad schematic
   */
  async generateSchematic(
    components: any[],
    connections: any[],
    outputPath: string
  ): Promise<string> {
    const inputData = JSON.stringify({ components, connections });
    const inputFile = path.join(this.config.workDir, `schematic_input_${Date.now()}.json`);

    await fs.writeFile(inputFile, inputData);

    const result = await this.execute('generate_schematic.py', [inputFile, outputPath]);
    if (!result.success) {
      throw new Error(`Failed to generate schematic: ${result.stderr}`);
    }

    // Clean up temp file
    await fs.unlink(inputFile).catch(() => {});

    return outputPath;
  }

  /**
   * Generate PCB layout from netlist
   */
  async generatePCBLayout(
    netlistPath: string,
    constraints: any,
    outputPath: string
  ): Promise<string> {
    const constraintsFile = path.join(this.config.workDir, `constraints_${Date.now()}.json`);
    await fs.writeFile(constraintsFile, JSON.stringify(constraints));

    const result = await this.execute('generate_pcb_layout.py', [
      netlistPath,
      constraintsFile,
      outputPath
    ]);

    if (!result.success) {
      throw new Error(`Failed to generate PCB layout: ${result.stderr}`);
    }

    // Clean up temp file
    await fs.unlink(constraintsFile).catch(() => {});

    return outputPath;
  }

  /**
   * Run DRC on a PCB file
   */
  async runDRC(pcbPath: string): Promise<any> {
    const result = await this.execute('run_drc.py', [pcbPath], { timeout: 60000 });
    if (!result.success) {
      throw new Error(`DRC failed: ${result.stderr}`);
    }
    return result.output;
  }

  /**
   * Export Gerber files
   */
  async exportGerbers(pcbPath: string, outputDir: string): Promise<string[]> {
    const result = await this.execute('export_gerbers.py', [pcbPath, outputDir], {
      timeout: 180000 // 3 minutes for gerber export
    });

    if (!result.success) {
      throw new Error(`Gerber export failed: ${result.stderr}`);
    }

    return result.output?.files || [];
  }

  /**
   * Export BOM (Bill of Materials)
   */
  async exportBOM(schematicPath: string, format: 'csv' | 'json' | 'xlsx' = 'csv'): Promise<string> {
    const outputFile = path.join(
      this.config.workDir,
      `bom_${Date.now()}.${format}`
    );

    const result = await this.execute('export_bom.py', [schematicPath, outputFile, format]);
    if (!result.success) {
      throw new Error(`BOM export failed: ${result.stderr}`);
    }

    return outputFile;
  }

  /**
   * Run SPICE simulation
   */
  async runSPICE(netlistPath: string, analysisType: string): Promise<any> {
    const result = await this.execute('run_spice.py', [netlistPath, analysisType], {
      timeout: 300000 // 5 minutes for simulation
    });

    if (!result.success) {
      throw new Error(`SPICE simulation failed: ${result.stderr}`);
    }

    return result.output;
  }

  /**
   * Get job status
   */
  getJob(jobId: string): ScriptJob | undefined {
    return this.jobs.get(jobId);
  }

  /**
   * Get all jobs
   */
  getAllJobs(): ScriptJob[] {
    return Array.from(this.jobs.values());
  }

  /**
   * Clear completed jobs
   */
  clearCompletedJobs(): void {
    for (const [id, job] of this.jobs) {
      if (job.status === 'completed' || job.status === 'failed') {
        this.jobs.delete(id);
      }
    }
  }

  /**
   * Get executor statistics
   */
  getStats(): {
    pythonVersion: string;
    initialized: boolean;
    runningJobs: number;
    totalJobs: number;
    completedJobs: number;
    failedJobs: number;
  } {
    const jobs = Array.from(this.jobs.values());
    return {
      pythonVersion: this.pythonVersion,
      initialized: this.initialized,
      runningJobs: this.runningJobs,
      totalJobs: jobs.length,
      completedJobs: jobs.filter(j => j.status === 'completed').length,
      failedJobs: jobs.filter(j => j.status === 'failed').length
    };
  }
}

// Export singleton instance
export const pythonExecutor = new PythonExecutor();

export default PythonExecutor;