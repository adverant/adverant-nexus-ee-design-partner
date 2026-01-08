/**
 * SKiDL Schematic Generator Plugin - Main Entry Point
 *
 * Production-ready schematic generation using SKiDL circuit description
 * language with 4-level validation pipeline and expert agent review.
 */

import { EventEmitter } from 'events';
import { v4 as uuidv4 } from 'uuid';
import * as path from 'path';
import * as fs from 'fs/promises';
import { spawn } from 'child_process';

import {
  GenerationConfig,
  GenerationResult,
  GenerationEvent,
  SchematicData,
  ValidationPipelineResult,
  GenerationConfigSchema
} from './types';
import { ValidationPipeline, ValidationPipelineConfig } from './validators';
import { runAllExpertReviews } from './agents';

export interface SchematicGenPluginConfig {
  pythonPath: string;
  pythonScriptsDir: string;
  kicadCliPath: string;
  ngspicePath: string;
  defaultOutputDir: string;
  maxIterations: number;
  targetScore: number;
}

const DEFAULT_PLUGIN_CONFIG: SchematicGenPluginConfig = {
  pythonPath: 'python3',
  pythonScriptsDir: path.join(__dirname, '../python-scripts'),
  kicadCliPath: 'kicad-cli',
  ngspicePath: 'ngspice',
  defaultOutputDir: './output/schematic',
  maxIterations: 100,
  targetScore: 90
};

/**
 * SKiDL Schematic Generator Plugin
 *
 * Provides complete schematic generation workflow:
 * 1. SKiDL circuit description → netlist
 * 2. Netlist → KiCad schematic conversion
 * 3. Multi-level validation pipeline
 * 4. Expert agent review
 */
export class SchematicGenPlugin extends EventEmitter {
  private config: SchematicGenPluginConfig;
  private validator: ValidationPipeline;

  constructor(config: Partial<SchematicGenPluginConfig> = {}) {
    super();
    this.config = { ...DEFAULT_PLUGIN_CONFIG, ...config };

    const validatorConfig: Partial<ValidationPipelineConfig> = {
      pythonPath: this.config.pythonPath,
      pythonScriptsDir: this.config.pythonScriptsDir,
      kicadCliPath: this.config.kicadCliPath,
      ngspicePath: this.config.ngspicePath
    };
    this.validator = new ValidationPipeline(validatorConfig);

    // Forward validator events
    this.validator.on('validation:start', (data) => this.emit('validation:start', data));
    this.validator.on('validation:level:complete', (data) => this.emit('validation:level:complete', data));
    this.validator.on('validation:level:failed', (data) => this.emit('validation:level:failed', data));
    this.validator.on('validation:expert:complete', (data) => this.emit('validation:expert:complete', data));
    this.validator.on('pipeline:complete', (data) => this.emit('pipeline:complete', data));
  }

  /**
   * Generate schematic from configuration
   */
  async generate(generationConfig: GenerationConfig): Promise<GenerationResult> {
    const startTime = Date.now();
    const generationId = uuidv4();

    // Validate configuration
    const configValidation = GenerationConfigSchema.safeParse(generationConfig);
    if (!configValidation.success) {
      const error = {
        code: 'INVALID_CONFIG',
        message: `Invalid generation configuration: ${configValidation.error.message}`,
        phase: 'initialization',
        context: { errors: configValidation.error.errors }
      };
      this.emitEvent({ type: 'error', error });
      return {
        success: false,
        validation: this.createFailedValidationResult(error.message),
        duration: Date.now() - startTime,
        timestamp: new Date().toISOString()
      };
    }

    this.emitEvent({ type: 'start', config: generationConfig });

    try {
      // Phase 1: Generate SKiDL netlists
      this.emitEvent({ type: 'phase', phase: 'skidl_generation', progress: 10 });
      const netlists = await this.generateSKiDLNetlists(generationConfig);

      // Phase 2: Convert netlists to KiCad schematics
      this.emitEvent({ type: 'phase', phase: 'schematic_conversion', progress: 40 });
      const schematicPath = await this.convertNetlistsToSchematic(
        netlists,
        generationConfig.outputDir
      );

      // Phase 3: Load schematic data for validation
      this.emitEvent({ type: 'phase', phase: 'loading_schematic', progress: 50 });
      const schematicData = await this.loadSchematicData(schematicPath);

      // Phase 4: Run validation pipeline
      this.emitEvent({ type: 'phase', phase: 'validation', progress: 60 });
      const validationResult = await this.validator.runAll(schematicPath, schematicData);

      // Phase 5: Iterative improvement if needed
      let finalSchematicData = schematicData;
      let finalValidationResult = validationResult;
      let iteration = 1;

      while (
        !finalValidationResult.success &&
        iteration < generationConfig.maxIterations &&
        finalValidationResult.overallScore < generationConfig.targetScore
      ) {
        this.emitEvent({
          type: 'phase',
          phase: `iteration_${iteration}`,
          progress: 60 + (iteration / generationConfig.maxIterations) * 30
        });

        // Apply fixes based on validation feedback
        const fixedSchematicPath = await this.applyValidationFixes(
          schematicPath,
          finalValidationResult
        );

        finalSchematicData = await this.loadSchematicData(fixedSchematicPath);
        finalValidationResult = await this.validator.runAll(fixedSchematicPath, finalSchematicData);
        iteration++;
      }

      this.emitEvent({ type: 'phase', phase: 'complete', progress: 100 });

      const result: GenerationResult = {
        success: finalValidationResult.success,
        schematic: finalSchematicData,
        filePath: schematicPath,
        validation: finalValidationResult,
        duration: Date.now() - startTime,
        timestamp: new Date().toISOString()
      };

      this.emitEvent({ type: 'complete', result });
      return result;

    } catch (error) {
      const generationError = {
        code: 'GENERATION_FAILED',
        message: error instanceof Error ? error.message : String(error),
        phase: 'unknown',
        stack: error instanceof Error ? error.stack : undefined
      };

      this.emitEvent({ type: 'error', error: generationError });

      return {
        success: false,
        validation: this.createFailedValidationResult(generationError.message),
        duration: Date.now() - startTime,
        timestamp: new Date().toISOString()
      };
    }
  }

  /**
   * Generate SKiDL netlists from circuit modules
   */
  private async generateSKiDLNetlists(config: GenerationConfig): Promise<string[]> {
    const netlists: string[] = [];

    for (const sheet of config.sheets) {
      const scriptPath = path.join(
        this.config.pythonScriptsDir,
        'skidl_circuits',
        `${sheet.skidlModule}.py`
      );

      // Check if module exists
      try {
        await fs.access(scriptPath);
      } catch {
        throw new Error(
          `SKiDL module not found: ${scriptPath}. ` +
          `Ensure the circuit description exists at this path.`
        );
      }

      const outputPath = path.join(config.outputDir, 'netlists', `${sheet.name}.net`);

      // Ensure output directory exists
      await fs.mkdir(path.dirname(outputPath), { recursive: true });

      // Execute SKiDL script
      const result = await this.executePython(scriptPath, {
        output: outputPath
      });

      if (result.exitCode !== 0) {
        throw new Error(
          `SKiDL generation failed for ${sheet.name}: ${result.stderr}`
        );
      }

      netlists.push(outputPath);

      this.emitEvent({
        type: 'sheet_generated',
        sheetNumber: sheet.sheetNumber,
        name: sheet.name
      });
    }

    return netlists;
  }

  /**
   * Convert netlists to KiCad schematic format
   */
  private async convertNetlistsToSchematic(
    netlists: string[],
    outputDir: string
  ): Promise<string> {
    const schematicPath = path.join(outputDir, 'schematic.kicad_sch');

    // Execute conversion script
    const result = await this.executePython(
      path.join(this.config.pythonScriptsDir, 'netlist_to_schematic.py'),
      {
        netlists: netlists.join(','),
        output: schematicPath
      }
    );

    if (result.exitCode !== 0) {
      throw new Error(`Netlist to schematic conversion failed: ${result.stderr}`);
    }

    return schematicPath;
  }

  /**
   * Load and parse schematic data
   */
  private async loadSchematicData(schematicPath: string): Promise<SchematicData> {
    const result = await this.executePython(
      path.join(this.config.pythonScriptsDir, 'parse_schematic.py'),
      { path: schematicPath, format: 'json' }
    );

    if (result.exitCode !== 0) {
      throw new Error(`Failed to parse schematic: ${result.stderr}`);
    }

    try {
      return JSON.parse(result.stdout) as SchematicData;
    } catch (error) {
      throw new Error(`Failed to parse schematic JSON: ${error}`);
    }
  }

  /**
   * Apply fixes based on validation feedback
   */
  private async applyValidationFixes(
    schematicPath: string,
    validationResult: ValidationPipelineResult
  ): Promise<string> {
    // Collect all errors from all levels and expert reviews
    const allErrors: string[] = [];

    for (const level of validationResult.levels) {
      for (const error of level.errors) {
        allErrors.push(`[${level.levelName}] ${error.message}`);
      }
    }

    for (const review of validationResult.expertReviews) {
      for (const recommendation of review.recommendations) {
        allErrors.push(`[${review.expertName}] ${recommendation}`);
      }
    }

    // Execute fix script
    const result = await this.executePython(
      path.join(this.config.pythonScriptsDir, 'apply_fixes.py'),
      {
        path: schematicPath,
        errors: JSON.stringify(allErrors)
      }
    );

    if (result.exitCode !== 0) {
      // Return original path if fixes fail
      return schematicPath;
    }

    return schematicPath;
  }

  /**
   * Execute a Python script
   */
  private executePython(
    scriptPath: string,
    args: Record<string, string>
  ): Promise<{ exitCode: number; stdout: string; stderr: string }> {
    return new Promise((resolve, reject) => {
      const argList = Object.entries(args).flatMap(([key, value]) => [
        `--${key}`,
        value
      ]);

      let stdout = '';
      let stderr = '';

      const process = spawn(this.config.pythonPath, [scriptPath, ...argList]);

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
   * Create a failed validation result
   */
  private createFailedValidationResult(message: string): ValidationPipelineResult {
    return {
      success: false,
      levels: [{
        level: 0 as any,
        levelName: 'Pre-validation',
        passed: false,
        errors: [{
          code: 'PRE_VALIDATION_ERROR',
          message,
          severity: 'error'
        }],
        warnings: [],
        metrics: {},
        duration: 0,
        timestamp: new Date().toISOString()
      }],
      overallScore: 0,
      expertReviews: [],
      summary: `Validation failed: ${message}`
    };
  }

  /**
   * Emit a generation event
   */
  private emitEvent(event: GenerationEvent): void {
    this.emit(event.type, event);
    this.emit('event', event);
  }

  /**
   * Validate an existing schematic without regeneration
   */
  async validate(schematicPath: string): Promise<ValidationPipelineResult> {
    const schematicData = await this.loadSchematicData(schematicPath);
    return this.validator.runAll(schematicPath, schematicData);
  }

  /**
   * Run expert review on schematic data
   */
  runExpertReview(schematicData: SchematicData) {
    return runAllExpertReviews(schematicData);
  }

  /**
   * Get plugin configuration
   */
  getConfig(): SchematicGenPluginConfig {
    return { ...this.config };
  }
}

// Export types
export * from './types';

// Export agents
export * from './agents';

// Export validators
export * from './validators';

// Default export
export default SchematicGenPlugin;
