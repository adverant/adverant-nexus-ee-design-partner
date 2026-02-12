/**
 * Symbol Assembly API Routes (MAPO v3.0)
 *
 * Handles symbol gathering, datasheet download, and characterization
 * before schematic generation. Searches GraphRAG first, then external sources.
 *
 * Endpoints:
 * - POST /api/v1/projects/:projectId/symbol-assembly/gather - Trigger symbol assembly
 * - GET /api/v1/projects/:projectId/symbol-assembly/:operationId - Get assembly report
 * - GET /api/v1/projects/:projectId/symbol-assembly/:operationId/symbols - List gathered symbols
 * - GET /api/v1/projects/:projectId/symbol-assembly/:operationId/datasheets - List datasheets
 * - GET /api/v1/projects/:projectId/symbol-assembly/:operationId/download/:filename - Download file
 */

import { Router, Request, Response, NextFunction } from 'express';
import { spawn } from 'child_process';
import { z } from 'zod';
import path from 'path';
import fs from 'fs/promises';
import { ValidationError, NotFoundError } from '../../utils/errors.js';
import { log } from '../../utils/logger.js';
import { config } from '../../config.js';
import { getSchematicWsManager } from '../schematic-ws.js';
import { SchematicEventType } from '../websocket-schematic.js';

/**
 * Validation schemas
 */
const GatherSymbolsSchema = z.object({
  body: z.object({
    ideationArtifacts: z.array(z.object({
      type: z.string(),
      content: z.string(),
    })),
    operationId: z.string().optional(),
  }),
});

/**
 * Assembly report structure
 */
interface AssemblyReport {
  operationId: string;
  projectId: string;
  startedAt: string;
  completedAt?: string;
  status: 'running' | 'complete' | 'error';
  totalComponents: number;
  symbolsFound: number;
  symbolsGenerated: number;
  datasheetsDownloaded: number;
  characterizationsCreated: number;
  errors: Array<{
    component: string;
    manufacturer?: string;
    partNumber?: string;
    message: string;
    sources: string[];
  }>;
  components: Array<{
    partNumber: string;
    manufacturer: string;
    package: string;
    category: string;
    symbolSource: 'graphrag' | 'kicad' | 'snapeda' | 'ultralibrarian' | 'llm-generated' | 'not-found';
    symbolPath?: string;
    datasheetSource?: string;
    datasheetPath?: string;
    characterizationPath?: string;
  }>;
}

/**
 * Create symbol assembly routes
 */
export function createSymbolAssemblyRoutes(): Router {
  const router = Router({ mergeParams: true });

  /**
   * Trigger symbol assembly
   * POST /api/v1/projects/:projectId/symbol-assembly/gather
   */
  router.post('/gather', async (req: Request, res: Response, next: NextFunction) => {
    try {
      const { projectId } = req.params;
      const validation = GatherSymbolsSchema.safeParse({ body: req.body });

      if (!validation.success) {
        throw new ValidationError('Invalid request body', {
          operation: 'gatherSymbols',
          errors: validation.error.errors,
        });
      }

      const { ideationArtifacts, operationId: providedOperationId } = validation.data.body;

      // Generate operation ID if not provided
      const { v4: uuidv4 } = await import('uuid');
      const operationId = providedOperationId || uuidv4();

      log.info('Starting symbol assembly', { projectId, operationId, artifactCount: ideationArtifacts.length });

      // Get WebSocket manager for streaming progress
      const wsManager = getSchematicWsManager();

      // Register operation in WebSocket manager so events aren't dropped
      wsManager.createOperation(projectId, operationId);

      // Create output directory
      const outputDir = path.join(config.artifacts.basePath, projectId, 'symbol-assembly');
      await fs.mkdir(outputDir, { recursive: true });

      // Write artifacts to temp file for Python script
      const artifactsPath = path.join(outputDir, `artifacts-${operationId}.json`);
      await fs.writeFile(artifactsPath, JSON.stringify(ideationArtifacts, null, 2));

      // Spawn Python symbol assembler
      const pythonPath = config.kicad.pythonPath || 'python3';
      const scriptPath = path.join(
        config.kicad.scriptsDir,
        'agents/symbol_assembly/symbol_assembler.py'
      );

      // Fail fast: verify script exists before spawning async process
      try {
        await fs.access(scriptPath);
      } catch {
        throw new Error(
          `Symbol assembler script not found at ${scriptPath}. ` +
          `config.kicad.scriptsDir=${config.kicad.scriptsDir}`
        );
      }

      const python = spawn(pythonPath, [
        scriptPath,
        '--project-id', projectId,
        '--operation-id', operationId,
        '--artifacts-json', artifactsPath,
        '--output-dir', outputDir,
      ]);

      // Emit initial start event so frontend leaves "Initializing..."
      wsManager.emitProgress(operationId, {
        type: SchematicEventType.SYMBOL_ASSEMBLY_START,
        operationId,
        timestamp: new Date().toISOString(),
        progress_percentage: 0,
        current_step: `Starting symbol assembly for ${ideationArtifacts.length} artifacts...`,
      });

      // Stream progress to WebSocket
      python.stdout.on('data', (data) => {
        const lines = data.toString().split('\n');
        for (const line of lines) {
          if (!line.trim()) continue;

          // Parse PROGRESS: prefix for WebSocket events
          if (line.startsWith('PROGRESS:')) {
            try {
              const event = JSON.parse(line.substring(9));
              wsManager.emitProgress(operationId, event);
            } catch (err) {
              log.warn('Failed to parse progress event', { line, error: err });
            }
          } else {
            // Log other output
            log.debug('Symbol assembler output', { operationId, line });
          }
        }
      });

      python.stderr.on('data', (data) => {
        const stderr = data.toString().trim();
        if (!stderr) return;

        // Python logging writes all levels to stderr by default.
        // Only forward WARNING/ERROR/CRITICAL as error events to the UI.
        const isActualError = /\[(ERROR|CRITICAL|WARNING)\]/.test(stderr)
          || /Traceback \(most recent call last\)/.test(stderr)
          || /^(Error|Exception|RuntimeError|ConnectionError|TimeoutError)/i.test(stderr);

        if (isActualError) {
          log.error('Symbol assembler error', { operation: operationId, stderr } as any);
          wsManager.emitProgress(operationId, {
            type: SchematicEventType.ERROR,
            operationId,
            timestamp: new Date().toISOString(),
            progress_percentage: 0,
            current_step: `Python error: ${stderr.slice(0, 200)}`,
            error_message: stderr,
          });
        } else {
          // INFO/DEBUG logs - just log server-side, don't spam the UI
          log.debug('Symbol assembler log', { operation: operationId, stderr } as any);
        }
      });

      python.on('close', (code) => {
        if (code !== 0) {
          log.error('Symbol assembler process failed', { operation: operationId, exitCode: code } as any);
          wsManager.emitProgress(operationId, {
            type: SchematicEventType.ERROR,
            operationId,
            timestamp: new Date().toISOString(),
            progress_percentage: 0,
            current_step: `Symbol assembly failed with exit code ${code}`,
            error_message: `Symbol assembly failed with exit code ${code}`,
          });
        } else {
          log.info('Symbol assembly complete', { operationId });
          wsManager.emitProgress(operationId, {
            type: SchematicEventType.SYMBOL_ASSEMBLY_COMPLETE,
            operationId,
            timestamp: new Date().toISOString(),
            progress_percentage: 100,
            current_step: 'Symbol assembly completed successfully',
          });
        }
      });

      python.on('error', (err) => {
        log.error('Failed to start symbol assembler', { operation: operationId, errorMessage: err.message } as any);
        wsManager.emitProgress(operationId, {
          type: SchematicEventType.ERROR,
          operationId,
          timestamp: new Date().toISOString(),
          progress_percentage: 0,
          current_step: `Failed to start symbol assembler: ${err.message}`,
          error_message: `Failed to start symbol assembler: ${err.message}`,
        });
      });

      // Return immediately with operation ID
      res.json({
        success: true,
        data: {
          operationId,
          status: 'running',
          projectId,
        },
      });
    } catch (error) {
      next(error);
    }
  });

  /**
   * Get assembly report
   * GET /api/v1/projects/:projectId/symbol-assembly/:operationId
   */
  router.get('/:operationId', async (req: Request, res: Response, next: NextFunction) => {
    try {
      const { projectId, operationId } = req.params;

      log.debug('Fetching assembly report', { projectId, operationId });

      const reportPath = path.join(
        config.artifacts.basePath,
        projectId,
        'symbol-assembly',
        'assembly_report.json'
      );

      try {
        const reportContent = await fs.readFile(reportPath, 'utf-8');
        const report: AssemblyReport = JSON.parse(reportContent);

        // Verify operation ID matches
        if (report.operationId !== operationId) {
          throw new NotFoundError('AssemblyReport', operationId, {
            operation: 'getAssemblyReport',
            message: 'Operation ID mismatch',
          } as any);
        }

        res.json({
          success: true,
          data: report,
        });
      } catch (err: any) {
        if (err.code === 'ENOENT') {
          throw new NotFoundError('AssemblyReport', operationId, {
            operation: 'getAssemblyReport',
            message: 'Assembly report not found. Operation may still be running.',
          } as any);
        }
        throw err;
      }
    } catch (error) {
      next(error);
    }
  });

  /**
   * List gathered symbols
   * GET /api/v1/projects/:projectId/symbol-assembly/:operationId/symbols
   */
  router.get('/:operationId/symbols', async (req: Request, res: Response, next: NextFunction) => {
    try {
      const { projectId, operationId } = req.params;

      log.debug('Listing gathered symbols', { projectId, operationId });

      const symbolsDir = path.join(
        config.artifacts.basePath,
        projectId,
        'symbol-assembly',
        'symbols'
      );

      try {
        const files = await fs.readdir(symbolsDir);
        const symbols = files
          .filter((f) => f.endsWith('.kicad_sym'))
          .map((filename) => ({
            filename,
            path: path.join(symbolsDir, filename),
            relativePath: `symbol-assembly/symbols/${filename}`,
          }));

        res.json({
          success: true,
          data: {
            operationId,
            count: symbols.length,
            symbols,
          },
        });
      } catch (err: any) {
        if (err.code === 'ENOENT') {
          // Directory doesn't exist yet - no symbols gathered
          res.json({
            success: true,
            data: {
              operationId,
              count: 0,
              symbols: [],
            },
          });
          return;
        }
        throw err;
      }
    } catch (error) {
      next(error);
    }
  });

  /**
   * List downloaded datasheets
   * GET /api/v1/projects/:projectId/symbol-assembly/:operationId/datasheets
   */
  router.get('/:operationId/datasheets', async (req: Request, res: Response, next: NextFunction) => {
    try {
      const { projectId, operationId } = req.params;

      log.debug('Listing datasheets', { projectId, operationId });

      const datasheetsDir = path.join(
        config.artifacts.basePath,
        projectId,
        'symbol-assembly',
        'datasheets'
      );

      try {
        const files = await fs.readdir(datasheetsDir);
        const datasheets = files
          .filter((f) => f.endsWith('.pdf'))
          .map((filename) => ({
            filename,
            path: path.join(datasheetsDir, filename),
            relativePath: `symbol-assembly/datasheets/${filename}`,
          }));

        res.json({
          success: true,
          data: {
            operationId,
            count: datasheets.length,
            datasheets,
          },
        });
      } catch (err: any) {
        if (err.code === 'ENOENT') {
          // Directory doesn't exist yet - no datasheets
          res.json({
            success: true,
            data: {
              operationId,
              count: 0,
              datasheets: [],
            },
          });
          return;
        }
        throw err;
      }
    } catch (error) {
      next(error);
    }
  });

  /**
   * Download specific file (symbol, datasheet, or characterization)
   * GET /api/v1/projects/:projectId/symbol-assembly/:operationId/download/:filename
   */
  router.get('/:operationId/download/:filename', async (req: Request, res: Response, next: NextFunction) => {
    try {
      const { projectId, operationId, filename } = req.params;

      log.debug('Downloading symbol assembly file', { projectId, operationId, filename });

      // Validate filename to prevent directory traversal
      if (filename.includes('..') || filename.includes('/') || filename.includes('\\')) {
        throw new ValidationError('Invalid filename', {
          operation: 'downloadFile',
          filename,
        });
      }

      const baseDir = path.join(
        config.artifacts.basePath,
        projectId,
        'symbol-assembly'
      );

      // Determine file type and subdirectory
      let subdirectory: string;
      let contentType: string;
      let extension: string;

      if (filename.endsWith('.kicad_sym')) {
        subdirectory = 'symbols';
        contentType = 'text/plain';
        extension = '.kicad_sym';
      } else if (filename.endsWith('.pdf')) {
        subdirectory = 'datasheets';
        contentType = 'application/pdf';
        extension = '.pdf';
      } else if (filename.endsWith('.md')) {
        subdirectory = 'characterizations';
        contentType = 'text/markdown';
        extension = '.md';
      } else {
        throw new ValidationError('Unsupported file type', {
          operation: 'downloadFile',
          filename,
          supportedTypes: ['.kicad_sym', '.pdf', '.md'],
        });
      }

      const filePath = path.join(baseDir, subdirectory, filename);

      // Verify file exists and is within allowed directory
      const realPath = await fs.realpath(filePath).catch(() => null);
      if (!realPath || !realPath.startsWith(path.join(baseDir, subdirectory))) {
        throw new NotFoundError('File', filename, {
          operation: 'downloadFile',
          message: 'File not found or access denied',
        } as any);
      }

      // Read and send file
      const fileContent = await fs.readFile(realPath);

      res.setHeader('Content-Type', contentType);
      res.setHeader('Content-Disposition', `attachment; filename="${filename}"`);
      res.setHeader('Cache-Control', 'no-cache, no-store, must-revalidate');
      res.send(fileContent);
    } catch (error) {
      next(error);
    }
  });

  return router;
}
