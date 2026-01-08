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
import { EventEmitter } from 'events';
import { v4 as uuidv4 } from 'uuid';
import { logger } from '../../utils/logger';

const execAsync = promisify(exec);

export interface PythonExecutorConfig {
  pythonPath?: string;
  scriptsDir: string;
  workDir: string;
  timeout: number;
  maxConcurrent: number;
}

export interface ScriptResult {
  success: boolean;
  stdout: string;
  stderr: string;
  exitCode: number;
  duration: number;
  output?: any;
}

export interface ScriptJob {
  id: string;
  script: string;
  args: string[];
  status: 'pending' | 'running' | 'completed' | 'failed' | 'timeout';
  result?: ScriptResult;
  startTime?: Date;
  endTime?: Date;
}

const DEFAULT_CONFIG: PythonExecutorConfig = {
  scriptsDir: path.join(__dirname, '../../../../python-scripts'),
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
      // Check Python availability
      const pythonPath = await this.findPython();
      this.config.pythonPath = pythonPath;

      // Get Python version
      const { stdout } = await execAsync(`${pythonPath} --version`);
      this.pythonVersion = stdout.trim();
      logger.info(`Python executor initialized: ${this.pythonVersion}`);

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
   * Find Python executable
   */
  private async findPython(): Promise<string> {
    const candidates = ['python3', 'python', '/usr/bin/python3', '/usr/local/bin/python3'];

    for (const candidate of candidates) {
      try {
        await execAsync(`${candidate} --version`);
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