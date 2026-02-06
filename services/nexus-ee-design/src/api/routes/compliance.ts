/**
 * Compliance Validation API Routes (MAPO v3.0)
 *
 * Handles schematic compliance validation against NASA/MIL-SPEC/IPC standards.
 * Provides real-time compliance checking with WebSocket streaming, waiver management,
 * and report export functionality.
 *
 * Endpoints:
 * - POST /api/v1/projects/:projectId/compliance/validate - Run compliance validation
 * - GET /api/v1/projects/:projectId/compliance/:reportId - Get compliance report
 * - POST /api/v1/projects/:projectId/compliance/:reportId/export - Export report
 * - POST /api/v1/projects/:projectId/compliance/waivers - Create waiver
 * - GET /api/v1/projects/:projectId/compliance/waivers - List waivers
 * - DELETE /api/v1/projects/:projectId/compliance/waivers/:waiverId - Delete waiver
 */

import { Router, Request, Response, NextFunction } from 'express';
import { z } from 'zod';
import path from 'path';
import fs from 'fs/promises';
import { ValidationError, NotFoundError } from '../../utils/errors.js';
import { log } from '../../utils/logger.js';
import { config } from '../../config.js';
import { getSchematicWsManager } from '../schematic-ws.js';
// Type imports - these types should be defined in the backend or imported from a shared package
// For now, we'll define them inline or import from the schematic-quality types
type ComplianceStandard = string;
type CheckCategory = string;

interface ComplianceWaiver {
  id: string;
  checkId: string;
  componentRef?: string;
  netName?: string;
  justification: string;
  submittedBy: string;
  submittedAt: string;
  expiresAt?: string;
  status: 'pending' | 'approved' | 'rejected' | 'expired';
  approvedBy?: string;
  approvedAt?: string;
}

interface ComplianceReport {
  reportId: string;
  operationId: string;
  projectId: string;
  generatedAt: string;
  score: number;
  passed: boolean;
  totalChecks: number;
  passedChecks: number;
  failedChecks: number;
  warningsChecks: number;
  skippedChecks: number;
  totalViolations: number;
  violationsBySeverity: Record<string, number>;
  checkResults: any[];
  standardsCoverage: string[];
  autoFixEnabled: boolean;
  autoFixedCount: number;
  waivers: ComplianceWaiver[];
  exportMetadata?: {
    exportedAt: string;
    exportedBy: string;
    format: 'json' | 'pdf' | 'html';
    filePath: string;
  };
}

/**
 * Validation schemas
 */
const ValidateComplianceSchema = z.object({
  body: z.object({
    schematicPath: z.string(),
    autoFixEnabled: z.boolean().optional().default(false),
    standards: z.array(z.string()).optional(),
    categories: z.array(z.string()).optional(),
  }),
});

const ExportReportSchema = z.object({
  body: z.object({
    format: z.enum(['json', 'pdf', 'html']),
  }),
});

const CreateWaiverSchema = z.object({
  body: z.object({
    checkId: z.string(),
    componentRef: z.string().optional(),
    netName: z.string().optional(),
    justification: z.string().min(10),
  }),
});

/**
 * In-memory storage for waivers (should be replaced with database in production)
 */
const waiversStore = new Map<string, ComplianceWaiver[]>();

/**
 * In-memory storage for reports (should be replaced with database in production)
 */
const reportsStore = new Map<string, ComplianceReport>();

/**
 * Create compliance routes
 */
export function createComplianceRoutes(): Router {
  const router = Router({ mergeParams: true });

  /**
   * Run compliance validation
   * POST /api/v1/projects/:projectId/compliance/validate
   */
  router.post('/validate', async (req: Request, res: Response, next: NextFunction) => {
    try {
      const { projectId } = req.params;
      const validation = ValidateComplianceSchema.safeParse({ body: req.body });

      if (!validation.success) {
        throw new ValidationError('Invalid request body', {
          operation: 'validateCompliance',
          errors: validation.error.errors,
        });
      }

      const { schematicPath, autoFixEnabled, standards, categories } = validation.data.body;

      log.info('Starting compliance validation', {
        projectId,
        schematicPath,
        autoFixEnabled,
        standardsCount: standards?.length || 0,
        categoriesCount: categories?.length || 0,
      });

      // Generate report ID
      const { v4: uuidv4 } = await import('uuid');
      const reportId = uuidv4();
      const operationId = uuidv4();

      // Get WebSocket manager for streaming progress
      const wsManager = getSchematicWsManager();

      // Verify schematic file exists
      const fullSchematicPath = path.join(config.artifacts.basePath, projectId, schematicPath);
      try {
        await fs.access(fullSchematicPath);
      } catch {
        throw new NotFoundError('Schematic', schematicPath, {
          operation: 'validateCompliance',
          message: 'Schematic file not found',
        });
      }

      // Import compliance validator service
      // NOTE: This service needs to be implemented separately
      // For now, we'll create a mock implementation
      const report = await runComplianceValidation(
        reportId,
        operationId,
        projectId,
        fullSchematicPath,
        autoFixEnabled,
        standards,
        categories,
        wsManager
      );

      // Store report
      reportsStore.set(reportId, report);

      // Also save to file system
      const reportDir = path.join(config.artifacts.basePath, projectId, 'compliance-reports');
      await fs.mkdir(reportDir, { recursive: true });
      const reportPath = path.join(reportDir, `${reportId}.json`);
      await fs.writeFile(reportPath, JSON.stringify(report, null, 2));

      log.info('Compliance validation complete', { projectId, reportId, score: report.score });

      res.json({
        success: true,
        data: report,
      });
    } catch (error) {
      next(error);
    }
  });

  /**
   * Get compliance report
   * GET /api/v1/projects/:projectId/compliance/:reportId
   */
  router.get('/:reportId', async (req: Request, res: Response, next: NextFunction) => {
    try {
      const { projectId, reportId } = req.params;

      log.debug('Fetching compliance report', { projectId, reportId });

      // Try in-memory store first
      let report = reportsStore.get(reportId);

      // If not in memory, try file system
      if (!report) {
        const reportPath = path.join(
          config.artifacts.basePath,
          projectId,
          'compliance-reports',
          `${reportId}.json`
        );

        try {
          const reportContent = await fs.readFile(reportPath, 'utf-8');
          report = JSON.parse(reportContent);
          // Cache in memory
          reportsStore.set(reportId, report);
        } catch (err: any) {
          if (err.code === 'ENOENT') {
            throw new NotFoundError('ComplianceReport', reportId, {
              operation: 'getComplianceReport',
            });
          }
          throw err;
        }
      }

      res.json({
        success: true,
        data: report,
      });
    } catch (error) {
      next(error);
    }
  });

  /**
   * Export compliance report
   * POST /api/v1/projects/:projectId/compliance/:reportId/export
   */
  router.post('/:reportId/export', async (req: Request, res: Response, next: NextFunction) => {
    try {
      const { projectId, reportId } = req.params;
      const validation = ExportReportSchema.safeParse({ body: req.body });

      if (!validation.success) {
        throw new ValidationError('Invalid request body', {
          operation: 'exportComplianceReport',
          errors: validation.error.errors,
        });
      }

      const { format } = validation.data.body;

      log.info('Exporting compliance report', { projectId, reportId, format });

      // Get report
      let report = reportsStore.get(reportId);
      if (!report) {
        const reportPath = path.join(
          config.artifacts.basePath,
          projectId,
          'compliance-reports',
          `${reportId}.json`
        );
        const reportContent = await fs.readFile(reportPath, 'utf-8');
        report = JSON.parse(reportContent);
      }

      // Create exports directory
      const exportsDir = path.join(config.artifacts.basePath, projectId, 'compliance-reports', 'exports');
      await fs.mkdir(exportsDir, { recursive: true });

      let exportPath: string;
      let contentType: string;
      let fileContent: Buffer | string;

      switch (format) {
        case 'json':
          exportPath = path.join(exportsDir, `${reportId}.json`);
          fileContent = JSON.stringify(report, null, 2);
          contentType = 'application/json';
          await fs.writeFile(exportPath, fileContent);
          break;

        case 'html':
          exportPath = path.join(exportsDir, `${reportId}.html`);
          fileContent = generateHTMLReport(report);
          contentType = 'text/html';
          await fs.writeFile(exportPath, fileContent);
          break;

        case 'pdf':
          // PDF generation would require additional dependencies (e.g., puppeteer)
          // For now, we'll throw an error
          throw new ValidationError('PDF export not yet implemented', {
            operation: 'exportComplianceReport',
            format,
          });

        default:
          throw new ValidationError('Unsupported export format', {
            operation: 'exportComplianceReport',
            format,
          });
      }

      // Send file
      const fileBuffer = typeof fileContent === 'string' ? Buffer.from(fileContent) : fileContent;
      res.setHeader('Content-Type', contentType);
      res.setHeader('Content-Disposition', `attachment; filename="compliance-report-${reportId}.${format}"`);
      res.setHeader('Cache-Control', 'no-cache, no-store, must-revalidate');
      res.send(fileBuffer);
    } catch (error) {
      next(error);
    }
  });

  /**
   * Create compliance waiver
   * POST /api/v1/projects/:projectId/compliance/waivers
   */
  router.post('/waivers', async (req: Request, res: Response, next: NextFunction) => {
    try {
      const { projectId } = req.params;
      const validation = CreateWaiverSchema.safeParse({ body: req.body });

      if (!validation.success) {
        throw new ValidationError('Invalid request body', {
          operation: 'createWaiver',
          errors: validation.error.errors,
        });
      }

      const { checkId, componentRef, netName, justification } = validation.data.body;

      log.info('Creating compliance waiver', { projectId, checkId, componentRef, netName });

      // Generate waiver
      const { v4: uuidv4 } = await import('uuid');
      const waiver: ComplianceWaiver = {
        id: uuidv4(),
        checkId,
        componentRef,
        netName,
        justification,
        submittedBy: 'current-user', // TODO: Get from auth context
        submittedAt: new Date().toISOString(),
        status: 'pending',
      };

      // Store waiver
      const projectWaivers = waiversStore.get(projectId) || [];
      projectWaivers.push(waiver);
      waiversStore.set(projectId, projectWaivers);

      // Also save to file system
      const waiversDir = path.join(config.artifacts.basePath, projectId, 'compliance-waivers');
      await fs.mkdir(waiversDir, { recursive: true });
      const waiversPath = path.join(waiversDir, 'waivers.json');
      await fs.writeFile(waiversPath, JSON.stringify(projectWaivers, null, 2));

      log.info('Compliance waiver created', { projectId, waiverId: waiver.id });

      res.json({
        success: true,
        data: waiver,
      });
    } catch (error) {
      next(error);
    }
  });

  /**
   * List compliance waivers
   * GET /api/v1/projects/:projectId/compliance/waivers
   */
  router.get('/waivers', async (req: Request, res: Response, next: NextFunction) => {
    try {
      const { projectId } = req.params;

      log.debug('Listing compliance waivers', { projectId });

      // Try in-memory store first
      let waivers = waiversStore.get(projectId);

      // If not in memory, try file system
      if (!waivers) {
        const waiversPath = path.join(
          config.artifacts.basePath,
          projectId,
          'compliance-waivers',
          'waivers.json'
        );

        try {
          const waiversContent = await fs.readFile(waiversPath, 'utf-8');
          waivers = JSON.parse(waiversContent);
          waiversStore.set(projectId, waivers);
        } catch (err: any) {
          if (err.code === 'ENOENT') {
            // No waivers file yet
            waivers = [];
          } else {
            throw err;
          }
        }
      }

      res.json({
        success: true,
        data: {
          projectId,
          count: waivers?.length || 0,
          waivers: waivers || [],
        },
      });
    } catch (error) {
      next(error);
    }
  });

  /**
   * Delete compliance waiver
   * DELETE /api/v1/projects/:projectId/compliance/waivers/:waiverId
   */
  router.delete('/waivers/:waiverId', async (req: Request, res: Response, next: NextFunction) => {
    try {
      const { projectId, waiverId } = req.params;

      log.info('Deleting compliance waiver', { projectId, waiverId });

      // Get waivers
      let waivers = waiversStore.get(projectId) || [];

      // Find and remove waiver
      const waiverIndex = waivers.findIndex((w) => w.id === waiverId);
      if (waiverIndex === -1) {
        throw new NotFoundError('ComplianceWaiver', waiverId, {
          operation: 'deleteWaiver',
        });
      }

      waivers.splice(waiverIndex, 1);
      waiversStore.set(projectId, waivers);

      // Update file system
      const waiversPath = path.join(
        config.artifacts.basePath,
        projectId,
        'compliance-waivers',
        'waivers.json'
      );
      await fs.writeFile(waiversPath, JSON.stringify(waivers, null, 2));

      log.info('Compliance waiver deleted', { projectId, waiverId });

      res.json({
        success: true,
        data: {
          deleted: true,
          waiverId,
        },
      });
    } catch (error) {
      next(error);
    }
  });

  return router;
}

/**
 * Run compliance validation (mock implementation)
 * TODO: Replace with actual compliance validator service
 */
async function runComplianceValidation(
  reportId: string,
  operationId: string,
  projectId: string,
  schematicPath: string,
  autoFixEnabled: boolean,
  standards?: ComplianceStandard[],
  categories?: CheckCategory[],
  wsManager?: any
): Promise<ComplianceReport> {
  // Emit start event
  if (wsManager) {
    wsManager.emitProgress(operationId, {
      type: 'validation_start',
      message: 'Starting compliance validation',
      timestamp: new Date().toISOString(),
    });
  }

  // TODO: Implement actual compliance validation
  // This should:
  // 1. Parse the schematic file
  // 2. Run all applicable checks
  // 3. Stream progress via WebSocket
  // 4. Generate detailed report

  // For now, return a mock report
  const report: ComplianceReport = {
    reportId,
    operationId,
    projectId,
    generatedAt: new Date().toISOString(),
    score: 85,
    passed: true,
    totalChecks: 51,
    passedChecks: 43,
    failedChecks: 5,
    warningsChecks: 3,
    skippedChecks: 0,
    totalViolations: 8,
    violationsBySeverity: {
      critical: 0,
      high: 1,
      medium: 4,
      low: 3,
      info: 0,
    },
    checkResults: [],
    standardsCoverage: standards || [
      'NASA-STD-8739.4',
      'IPC-2221',
      'Professional Best Practices',
    ],
    autoFixEnabled,
    autoFixedCount: autoFixEnabled ? 2 : 0,
    waivers: [],
  };

  // Emit complete event
  if (wsManager) {
    wsManager.emitProgress(operationId, {
      type: 'validation_complete',
      message: `Compliance validation complete. Score: ${report.score}`,
      timestamp: new Date().toISOString(),
    });
  }

  return report;
}

/**
 * Generate HTML report from compliance data
 */
function generateHTMLReport(report: ComplianceReport): string {
  const scoreColor = report.score >= 90 ? 'green' : report.score >= 70 ? 'yellow' : 'red';

  return `
<!DOCTYPE html>
<html>
<head>
  <meta charset="UTF-8">
  <title>Compliance Report ${report.reportId}</title>
  <style>
    body { font-family: Arial, sans-serif; margin: 40px; background: #f5f5f5; }
    .header { background: white; padding: 20px; border-radius: 8px; margin-bottom: 20px; }
    .score { font-size: 48px; font-weight: bold; color: ${scoreColor}; }
    .summary { background: white; padding: 20px; border-radius: 8px; margin-bottom: 20px; }
    .checks { background: white; padding: 20px; border-radius: 8px; }
    table { width: 100%; border-collapse: collapse; }
    th, td { padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }
    th { background: #f0f0f0; font-weight: bold; }
    .passed { color: green; }
    .failed { color: red; }
    .warning { color: orange; }
  </style>
</head>
<body>
  <div class="header">
    <h1>Schematic Compliance Report</h1>
    <div class="score">${report.score}/100</div>
    <p>Report ID: ${report.reportId}</p>
    <p>Generated: ${new Date(report.generatedAt).toLocaleString()}</p>
    <p>Project ID: ${report.projectId}</p>
  </div>

  <div class="summary">
    <h2>Summary</h2>
    <table>
      <tr>
        <th>Metric</th>
        <th>Value</th>
      </tr>
      <tr>
        <td>Total Checks</td>
        <td>${report.totalChecks}</td>
      </tr>
      <tr>
        <td class="passed">Passed</td>
        <td>${report.passedChecks}</td>
      </tr>
      <tr>
        <td class="failed">Failed</td>
        <td>${report.failedChecks}</td>
      </tr>
      <tr>
        <td class="warning">Warnings</td>
        <td>${report.warningsChecks}</td>
      </tr>
      <tr>
        <td>Total Violations</td>
        <td>${report.totalViolations}</td>
      </tr>
    </table>
  </div>

  <div class="checks">
    <h2>Standards Coverage</h2>
    <ul>
      ${report.standardsCoverage.map((std) => `<li>${std}</li>`).join('')}
    </ul>
  </div>

  <div class="checks">
    <h2>Violations by Severity</h2>
    <table>
      <tr>
        <th>Severity</th>
        <th>Count</th>
      </tr>
      <tr>
        <td>Critical</td>
        <td>${report.violationsBySeverity.critical}</td>
      </tr>
      <tr>
        <td>High</td>
        <td>${report.violationsBySeverity.high}</td>
      </tr>
      <tr>
        <td>Medium</td>
        <td>${report.violationsBySeverity.medium}</td>
      </tr>
      <tr>
        <td>Low</td>
        <td>${report.violationsBySeverity.low}</td>
      </tr>
      <tr>
        <td>Info</td>
        <td>${report.violationsBySeverity.info}</td>
      </tr>
    </table>
  </div>
</body>
</html>
  `.trim();
}
