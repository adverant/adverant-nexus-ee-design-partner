/**
 * Multi-Level Validation Pipeline
 *
 * Implements a 4-level validation pipeline for schematic verification:
 * Level 1: SKiDL ERC - Built-in electrical rule checking
 * Level 2: kicad-sch-api - Schematic structure and connectivity
 * Level 3: KiCad CLI ERC - Native KiCad electrical rule check
 * Level 4: SPICE Verification - Circuit simulation (optional)
 */

import { EventEmitter } from 'events';
import { spawn } from 'child_process';
import * as path from 'path';
import * as fs from 'fs/promises';
import {
  ValidationLevel,
  LevelValidationResult,
  ValidationPipelineResult,
  ValidationError,
  ValidationMetrics,
  SchematicData,
  VALIDATION_THRESHOLDS
} from '../types';
import { runAllExpertReviews } from '../agents';

export interface ValidationPipelineConfig {
  pythonPath: string;
  pythonScriptsDir: string;
  kicadCliPath: string;
  ngspicePath: string;
  timeout: number;
  levels: ValidationLevel[];
  enableExpertReview: boolean;
}

const DEFAULT_CONFIG: ValidationPipelineConfig = {
  pythonPath: 'python3',
  pythonScriptsDir: path.join(__dirname, '../../python-scripts'),
  kicadCliPath: 'kicad-cli',
  ngspicePath: 'ngspice',
  timeout: 300000, // 5 minutes
  levels: [
    ValidationLevel.SKIDL_ERC,
    ValidationLevel.KICAD_SCH_API,
    ValidationLevel.KICAD_CLI
  ],
  enableExpertReview: true
};

export class ValidationPipeline extends EventEmitter {
  private config: ValidationPipelineConfig;

  constructor(config: Partial<ValidationPipelineConfig> = {}) {
    super();
    this.config = { ...DEFAULT_CONFIG, ...config };
  }

  /**
   * Run the complete validation pipeline
   */
  async runAll(schematicPath: string, schematicData?: SchematicData): Promise<ValidationPipelineResult> {
    const startTime = Date.now();
    const levels: LevelValidationResult[] = [];
    let success = true;

    this.emit('pipeline:start', { schematicPath, levels: this.config.levels });

    // Run each validation level
    for (const level of this.config.levels) {
      if (!success && level > ValidationLevel.SKIDL_ERC) {
        // Skip subsequent levels if a blocking level failed
        break;
      }

      let result: LevelValidationResult;
      try {
        switch (level) {
          case ValidationLevel.SKIDL_ERC:
            result = await this.runLevel1SKiDL(schematicPath);
            break;
          case ValidationLevel.KICAD_SCH_API:
            result = await this.runLevel2KiCadAPI(schematicPath);
            break;
          case ValidationLevel.KICAD_CLI:
            result = await this.runLevel3KiCadCLI(schematicPath);
            break;
          case ValidationLevel.SPICE:
            result = await this.runLevel4SPICE(schematicPath);
            break;
          default:
            throw new Error(`Unknown validation level: ${level}`);
        }
      } catch (error) {
        result = this.createErrorResult(level, error);
      }

      levels.push(result);
      this.emit('validation:level:complete', { level, result });

      if (!result.passed) {
        success = false;
        this.emit('validation:level:failed', { level, result });
      }
    }

    // Run expert reviews if enabled and we have schematic data
    let expertReviews: ReturnType<typeof runAllExpertReviews> | null = null;
    let overallScore = 0;

    if (this.config.enableExpertReview && schematicData) {
      expertReviews = runAllExpertReviews(schematicData);
      this.emit('validation:expert:complete', expertReviews);

      if (!expertReviews.passed) {
        success = false;
      }
      overallScore = expertReviews.overallScore;
    } else {
      // Calculate score from level results
      const passedLevels = levels.filter(l => l.passed).length;
      overallScore = (passedLevels / levels.length) * 100;
    }

    const pipelineResult: ValidationPipelineResult = {
      success,
      levels,
      overallScore,
      expertReviews: expertReviews?.reviews || [],
      summary: this.generateSummary(levels, expertReviews)
    };

    this.emit('pipeline:complete', {
      result: pipelineResult,
      duration: Date.now() - startTime
    });

    return pipelineResult;
  }

  /**
   * Level 1: SKiDL ERC - Built-in electrical rule checking
   */
  async runLevel1SKiDL(schematicPath: string): Promise<LevelValidationResult> {
    const startTime = Date.now();
    this.emit('validation:start', { level: ValidationLevel.SKIDL_ERC, schematicPath });

    const errors: ValidationError[] = [];
    const warnings: ValidationError[] = [];
    const metrics: Partial<ValidationMetrics> = {};

    try {
      const result = await this.executePython('validation_gate.py', {
        level: 1,
        path: schematicPath
      });

      // Parse SKiDL output
      if (result.exitCode !== 0) {
        errors.push({
          code: 'SKIDL_ERC_FAILED',
          message: result.stderr || 'SKiDL ERC check failed',
          severity: 'error'
        });
      }

      // Parse metrics from stdout
      if (result.stdout) {
        const metricsMatch = result.stdout.match(/METRICS:(.+)/);
        if (metricsMatch) {
          try {
            const parsedMetrics = JSON.parse(metricsMatch[1]);
            Object.assign(metrics, parsedMetrics);
          } catch {
            // Ignore parse errors
          }
        }

        // Parse warnings
        const warningLines = result.stdout.split('\n').filter(l => l.includes('WARNING'));
        for (const line of warningLines) {
          warnings.push({
            code: 'SKIDL_WARNING',
            message: line.replace('WARNING:', '').trim(),
            severity: 'warning'
          });
        }
      }

      return {
        level: ValidationLevel.SKIDL_ERC,
        levelName: 'SKiDL ERC',
        passed: result.exitCode === 0 && errors.length === 0,
        errors,
        warnings,
        metrics,
        duration: Date.now() - startTime,
        timestamp: new Date().toISOString()
      };

    } catch (error) {
      return this.createErrorResult(ValidationLevel.SKIDL_ERC, error, Date.now() - startTime);
    }
  }

  /**
   * Level 2: kicad-sch-api Validation
   */
  async runLevel2KiCadAPI(schematicPath: string): Promise<LevelValidationResult> {
    const startTime = Date.now();
    this.emit('validation:start', { level: ValidationLevel.KICAD_SCH_API, schematicPath });

    const errors: ValidationError[] = [];
    const warnings: ValidationError[] = [];
    const metrics: Partial<ValidationMetrics> = {};

    try {
      const result = await this.executePython('validation_gate.py', {
        level: 2,
        path: schematicPath
      });

      // Parse kicad-sch-api output
      if (result.stdout) {
        const lines = result.stdout.split('\n');

        for (const line of lines) {
          if (line.startsWith('METRIC:')) {
            const [, name, value] = line.match(/METRIC:(\w+)=(.+)/) || [];
            if (name && value) {
              (metrics as Record<string, unknown>)[name] = parseFloat(value) || value;
            }
          } else if (line.startsWith('ERROR:')) {
            errors.push({
              code: 'KICAD_API_ERROR',
              message: line.replace('ERROR:', '').trim(),
              severity: 'error'
            });
          } else if (line.startsWith('WARNING:')) {
            warnings.push({
              code: 'KICAD_API_WARNING',
              message: line.replace('WARNING:', '').trim(),
              severity: 'warning'
            });
          }
        }
      }

      // Check wire/component ratio
      const ratio = metrics.wireComponentRatio;
      if (typeof ratio === 'number') {
        if (ratio < VALIDATION_THRESHOLDS.WIRE_COMPONENT_RATIO_CRITICAL) {
          errors.push({
            code: 'WIRE_RATIO_CRITICAL',
            message: `Wire/component ratio ${ratio.toFixed(2)} is critically low (minimum: ${VALIDATION_THRESHOLDS.WIRE_COMPONENT_RATIO_CRITICAL})`,
            severity: 'error'
          });
        } else if (ratio < VALIDATION_THRESHOLDS.WIRE_COMPONENT_RATIO_MIN) {
          errors.push({
            code: 'WIRE_RATIO_LOW',
            message: `Wire/component ratio ${ratio.toFixed(2)} is below minimum (${VALIDATION_THRESHOLDS.WIRE_COMPONENT_RATIO_MIN})`,
            severity: 'error'
          });
        }
      }

      return {
        level: ValidationLevel.KICAD_SCH_API,
        levelName: 'kicad-sch-api Validation',
        passed: result.exitCode === 0 && errors.length === 0,
        errors,
        warnings,
        metrics,
        duration: Date.now() - startTime,
        timestamp: new Date().toISOString()
      };

    } catch (error) {
      return this.createErrorResult(ValidationLevel.KICAD_SCH_API, error, Date.now() - startTime);
    }
  }

  /**
   * Level 3: KiCad CLI ERC
   */
  async runLevel3KiCadCLI(schematicPath: string): Promise<LevelValidationResult> {
    const startTime = Date.now();
    this.emit('validation:start', { level: ValidationLevel.KICAD_CLI, schematicPath });

    const errors: ValidationError[] = [];
    const warnings: ValidationError[] = [];
    const metrics: Partial<ValidationMetrics> = {};

    try {
      // Check if kicad-cli exists
      const kicadCliExists = await this.checkCommandExists(this.config.kicadCliPath);
      if (!kicadCliExists) {
        warnings.push({
          code: 'KICAD_CLI_NOT_FOUND',
          message: `KiCad CLI not found at ${this.config.kicadCliPath}. Skipping native ERC.`,
          severity: 'warning'
        });

        return {
          level: ValidationLevel.KICAD_CLI,
          levelName: 'KiCad CLI ERC',
          passed: true, // Pass with warning if CLI not available
          errors,
          warnings,
          metrics: { erc_status: 'SKIPPED' },
          duration: Date.now() - startTime,
          timestamp: new Date().toISOString()
        };
      }

      // Run kicad-cli sch erc
      const result = await this.executeCommand(
        this.config.kicadCliPath,
        ['sch', 'erc', '--exit-code-violations', schematicPath]
      );

      // Parse ERC output
      if (result.stdout) {
        const lines = result.stdout.split('\n');

        for (const line of lines) {
          // KiCad ERC format: [ERROR] message at (file:line:col)
          const errorMatch = line.match(/\[ERROR\]\s*(.+?)(?:\s+at\s+\((.+)\))?$/);
          if (errorMatch) {
            const locationMatch = errorMatch[2]?.match(/(.+):(\d+):(\d+)/);
            errors.push({
              code: 'KICAD_ERC_ERROR',
              message: errorMatch[1],
              severity: 'error',
              location: locationMatch ? {
                file: locationMatch[1],
                line: parseInt(locationMatch[2], 10)
              } : undefined
            });
          }

          const warningMatch = line.match(/\[WARNING\]\s*(.+?)(?:\s+at\s+\((.+)\))?$/);
          if (warningMatch) {
            warnings.push({
              code: 'KICAD_ERC_WARNING',
              message: warningMatch[1],
              severity: 'warning'
            });
          }
        }
      }

      metrics.erc_status = result.exitCode === 0 ? 'PASS' : 'FAIL';

      return {
        level: ValidationLevel.KICAD_CLI,
        levelName: 'KiCad CLI ERC',
        passed: result.exitCode === 0,
        errors,
        warnings,
        metrics,
        duration: Date.now() - startTime,
        timestamp: new Date().toISOString()
      };

    } catch (error) {
      return this.createErrorResult(ValidationLevel.KICAD_CLI, error, Date.now() - startTime);
    }
  }

  /**
   * Level 4: SPICE Verification (optional)
   */
  async runLevel4SPICE(schematicPath: string): Promise<LevelValidationResult> {
    const startTime = Date.now();
    this.emit('validation:start', { level: ValidationLevel.SPICE, schematicPath });

    const errors: ValidationError[] = [];
    const warnings: ValidationError[] = [];
    const metrics: Partial<ValidationMetrics> = {};

    try {
      // Check if ngspice exists
      const ngspiceExists = await this.checkCommandExists(this.config.ngspicePath);
      if (!ngspiceExists) {
        warnings.push({
          code: 'NGSPICE_NOT_FOUND',
          message: `ngspice not found at ${this.config.ngspicePath}. Skipping SPICE verification.`,
          severity: 'warning'
        });

        return {
          level: ValidationLevel.SPICE,
          levelName: 'SPICE Verification',
          passed: true, // Pass with warning if ngspice not available
          errors,
          warnings,
          metrics: { spice_status: 'SKIPPED' },
          duration: Date.now() - startTime,
          timestamp: new Date().toISOString()
        };
      }

      // Export netlist for SPICE
      const result = await this.executePython('validation_gate.py', {
        level: 4,
        path: schematicPath
      });

      if (result.exitCode !== 0) {
        errors.push({
          code: 'SPICE_FAILED',
          message: result.stderr || 'SPICE simulation failed',
          severity: 'error'
        });
      }

      metrics.spice_status = result.exitCode === 0 ? 'PASS' : 'FAIL';

      return {
        level: ValidationLevel.SPICE,
        levelName: 'SPICE Verification',
        passed: result.exitCode === 0,
        errors,
        warnings,
        metrics,
        duration: Date.now() - startTime,
        timestamp: new Date().toISOString()
      };

    } catch (error) {
      return this.createErrorResult(ValidationLevel.SPICE, error, Date.now() - startTime);
    }
  }

  /**
   * Execute a Python script
   */
  private async executePython(
    scriptName: string,
    args: Record<string, unknown>
  ): Promise<{ exitCode: number; stdout: string; stderr: string }> {
    const scriptPath = path.join(this.config.pythonScriptsDir, scriptName);

    // Check if script exists
    try {
      await fs.access(scriptPath);
    } catch {
      throw new Error(`Python script not found: ${scriptPath}`);
    }

    const argList = Object.entries(args).flatMap(([key, value]) => [
      `--${key}`,
      String(value)
    ]);

    return this.executeCommand(this.config.pythonPath, [scriptPath, ...argList]);
  }

  /**
   * Execute a shell command
   */
  private executeCommand(
    command: string,
    args: string[]
  ): Promise<{ exitCode: number; stdout: string; stderr: string }> {
    return new Promise((resolve, reject) => {
      let stdout = '';
      let stderr = '';

      const process = spawn(command, args, {
        timeout: this.config.timeout
      });

      process.stdout?.on('data', (data) => {
        stdout += data.toString();
      });

      process.stderr?.on('data', (data) => {
        stderr += data.toString();
      });

      process.on('close', (exitCode) => {
        resolve({ exitCode: exitCode || 0, stdout, stderr });
      });

      process.on('error', (error) => {
        reject(error);
      });
    });
  }

  /**
   * Check if a command exists
   */
  private async checkCommandExists(command: string): Promise<boolean> {
    try {
      const result = await this.executeCommand('which', [command]);
      return result.exitCode === 0;
    } catch {
      return false;
    }
  }

  /**
   * Create error result for failed validation
   */
  private createErrorResult(
    level: ValidationLevel,
    error: unknown,
    duration?: number
  ): LevelValidationResult {
    const levelNames: Record<ValidationLevel, string> = {
      [ValidationLevel.SKIDL_ERC]: 'SKiDL ERC',
      [ValidationLevel.KICAD_SCH_API]: 'kicad-sch-api Validation',
      [ValidationLevel.KICAD_CLI]: 'KiCad CLI ERC',
      [ValidationLevel.SPICE]: 'SPICE Verification'
    };

    return {
      level,
      levelName: levelNames[level],
      passed: false,
      errors: [{
        code: 'VALIDATION_ERROR',
        message: error instanceof Error ? error.message : String(error),
        severity: 'error'
      }],
      warnings: [],
      metrics: {},
      duration: duration || 0,
      timestamp: new Date().toISOString()
    };
  }

  /**
   * Generate summary of validation results
   */
  private generateSummary(
    levels: LevelValidationResult[],
    expertReviews: ReturnType<typeof runAllExpertReviews> | null
  ): string {
    const lines: string[] = [];
    lines.push('═══════════════════════════════════════════════════');
    lines.push('           SCHEMATIC VALIDATION REPORT             ');
    lines.push('═══════════════════════════════════════════════════');
    lines.push('');

    // Level results
    lines.push('VALIDATION LEVELS:');
    for (const level of levels) {
      const status = level.passed ? '✓ PASS' : '✗ FAIL';
      lines.push(`  Level ${level.level} (${level.levelName}): ${status}`);
      if (level.errors.length > 0) {
        lines.push(`    Errors: ${level.errors.length}`);
      }
      if (level.warnings.length > 0) {
        lines.push(`    Warnings: ${level.warnings.length}`);
      }
    }
    lines.push('');

    // Expert reviews
    if (expertReviews) {
      lines.push('EXPERT REVIEWS:');
      for (const review of expertReviews.reviews) {
        const status = review.passed ? '✓ PASS' : '✗ FAIL';
        lines.push(`  ${review.expertName} (${review.role}): ${status} (${review.score.toFixed(1)}%)`);
      }
      lines.push('');
      lines.push(`OVERALL SCORE: ${expertReviews.overallScore.toFixed(1)}%`);
    }

    lines.push('');
    lines.push('═══════════════════════════════════════════════════');

    return lines.join('\n');
  }
}

export default ValidationPipeline;
