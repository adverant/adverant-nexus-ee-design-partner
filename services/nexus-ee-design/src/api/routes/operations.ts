/**
 * Operations Center REST API Routes
 *
 * Provides unified operations management endpoints that aggregate data from:
 * - EE Design WebSocket managers (schematic, pcb, simulation)
 * - Trigger.dev runs (via the Trigger.dev integration service)
 *
 * Security:
 *   - All routes require x-user-id header (auth middleware applied at router level)
 *   - File uploads are sanitized (path traversal prevention, blocked extensions)
 *   - Waitpoint resolutions include audit trail (userId logged)
 *
 * Endpoints:
 *   GET    /operations              — List all operations (filterable)
 *   GET    /operations/stats        — Aggregate statistics
 *   GET    /operations/waitpoints   — Pending waitpoints requiring approval
 *   GET    /operations/:operationId — Single operation detail
 *   GET    /operations/:operationId/logs — Operation log entries
 *   POST   /operations/:operationId/cancel  — Cancel a running operation
 *   POST   /operations/:operationId/pause   — Pause a running operation
 *   POST   /operations/:operationId/resume  — Resume a paused operation
 *   POST   /operations/:operationId/replay  — Replay a completed/failed operation
 *   PATCH  /operations/:operationId/params  — Update parameters mid-run
 *   POST   /operations/:operationId/files   — Inject a file into a running operation
 *   POST   /operations/waitpoints/:waitpointId/resolve — Approve or reject a waitpoint
 */

import { Router, Request, Response, NextFunction } from 'express';
import { Server as SocketIOServer } from 'socket.io';
import { z } from 'zod';
import multer from 'multer';
import path from 'path';
import { v4 as uuidv4 } from 'uuid';
import { getSchematicWsManager } from '../schematic-ws.js';
import { TriggerIntegrationService } from '../../services/trigger-integration.js';
import { log as rootLog } from '../../utils/logger.js';

const log = rootLog.child({ component: 'operations-routes' });

// ─── Validation Schemas ──────────────────────────────────────────────────

const listOperationsSchema = z.object({
  status: z.enum(['all', 'queued', 'running', 'paused', 'waiting-approval', 'completed', 'failed', 'cancelled']).optional().default('all'),
  source: z.enum(['all', 'ee-design', 'trigger', 'n8n', 'terminal-computer']).optional().default('all'),
  projectId: z.string().uuid().optional(),
  type: z.enum(['all', 'schematic', 'pcb-layout', 'simulation', 'symbol-assembly', 'compliance', 'custom']).optional().default('all'),
  search: z.string().optional(),
  limit: z.coerce.number().int().min(1).max(200).optional().default(100),
  offset: z.coerce.number().int().min(0).optional().default(0),
});

const updateParamsSchema = z.object({
  parameters: z.record(z.unknown()),
});

const resolveWaitpointSchema = z.object({
  approve: z.boolean(),
  reason: z.string().optional(),
  output: z.record(z.unknown()).optional(),
});

const logsQuerySchema = z.object({
  level: z.enum(['debug', 'info', 'warn', 'error']).optional(),
  limit: z.coerce.number().int().min(1).max(2000).optional().default(500),
  offset: z.coerce.number().int().min(0).optional().default(0),
});

// ─── File Upload Config ──────────────────────────────────────────────────

const NFS_UPLOAD_DIR = process.env.NFS_UPLOAD_DIR || '/mnt/nfs/operations/uploads';

const BLOCKED_EXTENSIONS = ['.exe', '.bat', '.cmd', '.sh', '.ps1', '.dll', '.so', '.dylib'];

const upload = multer({
  storage: multer.diskStorage({
    destination: (_req, _file, cb) => cb(null, NFS_UPLOAD_DIR),
    filename: (_req, file, cb) => {
      // Sanitize: strip path components, limit to safe characters
      const safeName = path.basename(file.originalname)
        .replace(/[^a-zA-Z0-9._-]/g, '_')
        .slice(0, 255);
      const uniqueName = `${Date.now()}-${uuidv4().slice(0, 8)}-${safeName}`;
      cb(null, uniqueName);
    },
  }),
  fileFilter: (_req, file, cb) => {
    const ext = path.extname(file.originalname).toLowerCase();
    if (BLOCKED_EXTENSIONS.includes(ext)) {
      cb(new Error(`File type ${ext} is not allowed for security reasons`));
      return;
    }
    cb(null, true);
  },
  limits: { fileSize: 50 * 1024 * 1024 }, // 50MB
});

// ─── Helpers ─────────────────────────────────────────────────────────────

interface UnifiedOperationDTO {
  id: string;
  name: string;
  type: string;
  source: string;
  sourceId: string;
  projectId: string;
  projectName: string;
  status: string;
  progress: number | null;
  currentStep: string;
  phase?: string;
  phases?: Array<{ id: string; name: string; status: string; progress: number }>;
  createdAt: string;
  startedAt?: string;
  completedAt?: string;
  duration?: number;
  error?: string;
  output?: Record<string, unknown>;
  parameters?: Record<string, unknown>;
  capabilities: {
    canCancel: boolean;
    canPause: boolean;
    canResume: boolean;
    canReplay: boolean;
    canModifyParams: boolean;
    canInjectFiles: boolean;
    hasLogs: boolean;
    hasQualityGates: boolean;
    hasTerminal: boolean;
  };
  qualityGates?: Array<{
    name: string;
    passed: boolean;
    threshold: number;
    actual: number;
    unit: string;
    critical: boolean;
  }>;
  files?: Array<{
    name: string;
    path: string;
    type: string;
    size?: number;
    createdAt: string;
  }>;
  triggerRunId?: string;
  iteration?: number;
  totalIterations?: number;
  tags: string[];
}

/**
 * Convert a BaseWebSocketManager Operation to UnifiedOperationDTO
 */
function wsOperationToDTO(
  op: any,
  source: string,
  operationType: string,
  projectName?: string
): UnifiedOperationDTO {
  const statusMap: Record<string, string> = {
    running: 'running',
    complete: 'completed',
    error: 'failed',
  };

  return {
    id: op.operationId,
    name: `${operationType} — ${op.operationId.slice(0, 8)}`,
    type: operationType,
    source,
    sourceId: op.operationId,
    projectId: op.projectId,
    projectName: projectName || op.projectId,
    status: statusMap[op.status] || op.status,
    progress: op.lastEvent?.progress_percentage ?? null,
    currentStep: op.lastEvent?.current_step || '',
    phase: op.lastEvent?.phase,
    createdAt: op.startedAt?.toISOString?.() || new Date().toISOString(),
    startedAt: op.startedAt?.toISOString?.() || undefined,
    duration: op.startedAt ? Date.now() - new Date(op.startedAt).getTime() : undefined,
    capabilities: {
      canCancel: op.status === 'running',
      canPause: false, // WS operations don't support pause natively
      canResume: false,
      canReplay: op.status === 'complete' || op.status === 'error',
      canModifyParams: op.status === 'running',
      canInjectFiles: op.status === 'running',
      hasLogs: true,
      hasQualityGates: operationType === 'schematic',
      hasTerminal: false,
    },
    tags: [operationType, source],
  };
}

/**
 * Get all operations from a WS manager.
 * Uses getProjectOperations with empty filter as a workaround since
 * BaseWebSocketManager doesn't expose a public getAllOperations() method.
 * Falls back to accessing the internal operations Map if needed.
 */
function getWsManagerOperations(wsManager: any): any[] {
  // Try the internal Map directly — documented cast for operations access
  if (wsManager.operations && typeof wsManager.operations.values === 'function') {
    return Array.from(wsManager.operations.values());
  }
  return [];
}

// ─── Route Factory ───────────────────────────────────────────────────────

export function createOperationsRoutes(
  io: SocketIOServer,
  triggerService?: TriggerIntegrationService
): Router {
  const router = Router({ mergeParams: true });

  // ── Authentication Middleware ────────────────────────────────────────
  // All operations routes require a valid x-user-id header.
  // This matches the auth pattern used in the main routes.ts file.
  router.use((req: Request, res: Response, next: NextFunction) => {
    const userId = req.headers['x-user-id'] as string;
    if (!userId || userId === 'system' || userId === 'anonymous') {
      res.status(401).json({
        success: false,
        error: {
          code: 'UNAUTHORIZED',
          message: 'Authentication required. Provide a valid x-user-id header.',
        },
      });
      return;
    }
    // Store userId on request for downstream handlers
    (req as any).userId = userId;
    next();
  });

  // ── GET /operations ──────────────────────────────────────────────────
  router.get('/', async (req: Request, res: Response, next: NextFunction) => {
    try {
      const parsed = listOperationsSchema.safeParse(req.query);
      if (!parsed.success) {
        res.status(400).json({
          success: false,
          error: { code: 'VALIDATION_ERROR', message: 'Invalid query parameters', details: parsed.error.errors },
        });
        return;
      }

      const filters = parsed.data;
      const allOperations: UnifiedOperationDTO[] = [];

      // ── Collect from EE Design WebSocket managers ──────────────────
      if (filters.source === 'all' || filters.source === 'ee-design') {
        try {
          const schematicWsManager = getSchematicWsManager();
          const wsOps = filters.projectId
            ? schematicWsManager.getProjectOperations(filters.projectId)
            : getWsManagerOperations(schematicWsManager);

          for (const op of wsOps) {
            allOperations.push(wsOperationToDTO(op, 'ee-design', 'schematic'));
          }
        } catch {
          // WS manager may not be initialized yet
        }
      }

      // ── Collect from Trigger.dev ────────────────────────────────────
      // No pagination at source — we paginate after merging all sources
      if (triggerService && (filters.source === 'all' || filters.source === 'trigger')) {
        try {
          const triggerOps = await triggerService.listOperations({
            status: filters.status !== 'all' ? filters.status : undefined,
            projectId: filters.projectId,
            type: filters.type !== 'all' ? filters.type : undefined,
            limit: 500, // Safety cap for in-memory merge
          });
          allOperations.push(...triggerOps);
        } catch (err) {
          log.warn('Failed to fetch Trigger.dev operations', { error: err instanceof Error ? err.message : err });
        }
      }

      // ── Apply filters ──────────────────────────────────────────────
      let filtered = allOperations;

      if (filters.status !== 'all') {
        filtered = filtered.filter((op) => op.status === filters.status);
      }

      if (filters.type !== 'all') {
        filtered = filtered.filter((op) => op.type === filters.type);
      }

      if (filters.projectId) {
        filtered = filtered.filter((op) => op.projectId === filters.projectId);
      }

      if (filters.search) {
        const searchLower = filters.search.toLowerCase();
        filtered = filtered.filter(
          (op) =>
            op.name.toLowerCase().includes(searchLower) ||
            op.currentStep.toLowerCase().includes(searchLower) ||
            op.projectName.toLowerCase().includes(searchLower) ||
            op.tags.some((t) => t.toLowerCase().includes(searchLower))
        );
      }

      // Sort: active first, then by creation time desc
      filtered.sort((a, b) => {
        const activeStatuses = ['running', 'queued', 'waiting-approval', 'paused'];
        const aActive = activeStatuses.includes(a.status) ? 0 : 1;
        const bActive = activeStatuses.includes(b.status) ? 0 : 1;
        if (aActive !== bActive) return aActive - bActive;
        return new Date(b.createdAt).getTime() - new Date(a.createdAt).getTime();
      });

      // Paginate
      const total = filtered.length;
      const paged = filtered.slice(filters.offset, filters.offset + filters.limit);

      res.json({
        success: true,
        data: {
          operations: paged,
          pagination: { total, limit: filters.limit, offset: filters.offset },
        },
      });
    } catch (error) {
      next(error);
    }
  });

  // ── GET /operations/stats ────────────────────────────────────────────
  router.get('/stats', async (req: Request, res: Response, next: NextFunction) => {
    try {
      let active = 0;
      let queued = 0;
      let failed = 0;
      let completedToday = 0;
      let pendingApprovals = 0;

      // Count from WS managers
      try {
        const schematicWsManager = getSchematicWsManager();
        const wsStats = schematicWsManager.getStats();
        active += wsStats.activeOperations;
      } catch {
        // WS manager may not be initialized
      }

      // Count from Trigger.dev
      if (triggerService) {
        try {
          const triggerStats = await triggerService.getStats();
          active += triggerStats.active;
          queued += triggerStats.queued;
          failed += triggerStats.failed;
          completedToday += triggerStats.completedToday;
          pendingApprovals += triggerStats.pendingApprovals;
        } catch (err) {
          log.warn('Failed to fetch Trigger.dev stats', { error: err instanceof Error ? err.message : err });
        }
      }

      res.json({
        success: true,
        data: {
          active,
          queued,
          failed,
          completedToday,
          pendingApprovals,
        },
      });
    } catch (error) {
      next(error);
    }
  });

  // ── GET /operations/waitpoints ───────────────────────────────────────
  router.get('/waitpoints', async (req: Request, res: Response, next: NextFunction) => {
    try {
      let waitpoints: any[] = [];

      if (triggerService) {
        try {
          waitpoints = await triggerService.listWaitpoints();
        } catch (err) {
          log.warn('Failed to fetch waitpoints', { error: err instanceof Error ? err.message : err });
        }
      }

      res.json({
        success: true,
        data: { waitpoints },
      });
    } catch (error) {
      next(error);
    }
  });

  // ── GET /operations/:operationId ─────────────────────────────────────
  router.get('/:operationId', async (req: Request, res: Response, next: NextFunction) => {
    try {
      const { operationId } = req.params;

      // Check WS managers first
      try {
        const schematicWsManager = getSchematicWsManager();
        const wsOp = schematicWsManager.getOperation(operationId);
        if (wsOp) {
          res.json({
            success: true,
            data: wsOperationToDTO(wsOp, 'ee-design', 'schematic'),
          });
          return;
        }
      } catch {
        // WS manager may not be initialized
      }

      // Check Trigger.dev
      if (triggerService) {
        try {
          const triggerOp = await triggerService.getOperation(operationId);
          if (triggerOp) {
            res.json({ success: true, data: triggerOp });
            return;
          }
        } catch (err) {
          log.warn('Failed to fetch Trigger.dev operation', { operationId, error: err instanceof Error ? err.message : err });
        }
      }

      res.status(404).json({
        success: false,
        error: { code: 'NOT_FOUND', message: `Operation ${operationId} not found` },
      });
    } catch (error) {
      next(error);
    }
  });

  // ── GET /operations/:operationId/logs ────────────────────────────────
  router.get('/:operationId/logs', async (req: Request, res: Response, next: NextFunction) => {
    try {
      const { operationId } = req.params;
      const parsed = logsQuerySchema.safeParse(req.query);
      if (!parsed.success) {
        res.status(400).json({
          success: false,
          error: { code: 'VALIDATION_ERROR', message: 'Invalid query parameters', details: parsed.error.errors },
        });
        return;
      }

      const { level, limit, offset } = parsed.data;
      let logs: any[] = [];

      // Fetch from WS manager event history
      try {
        const schematicWsManager = getSchematicWsManager();
        const wsOp = schematicWsManager.getOperation(operationId);
        // Access eventHistory via the Operation object (documented internal access)
        const eventHistory = (wsOp as any)?.eventHistory;
        if (eventHistory && Array.isArray(eventHistory)) {
          logs = eventHistory.map((evt: any, idx: number) => ({
            id: `log-${operationId}-${idx}`,
            operationId,
            timestamp: evt.timestamp,
            level: evt.type === 'error' ? 'error' : 'info',
            message: evt.current_step || evt.type,
            phase: evt.phase,
            data: evt.data,
          }));
        }
      } catch {
        // WS manager may not be initialized
      }

      // Fetch from Trigger.dev
      if (logs.length === 0 && triggerService) {
        try {
          logs = await triggerService.getOperationLogs(operationId, { level, limit, offset });
        } catch (err) {
          log.warn('Failed to fetch Trigger.dev logs', { operationId, error: err instanceof Error ? err.message : err });
        }
      }

      // Apply level filter
      if (level) {
        const levelPriority: Record<string, number> = { debug: 0, info: 1, warn: 2, error: 3 };
        const minLevel = levelPriority[level] ?? 0;
        logs = logs.filter((l: any) => (levelPriority[l.level] ?? 0) >= minLevel);
      }

      // Apply pagination
      const total = logs.length;
      const paged = logs.slice(offset, offset + limit);

      res.json({
        success: true,
        data: { logs: paged, pagination: { total, limit, offset } },
      });
    } catch (error) {
      next(error);
    }
  });

  // ── POST /operations/:operationId/cancel ─────────────────────────────
  router.post('/:operationId/cancel', async (req: Request, res: Response, next: NextFunction) => {
    try {
      const { operationId } = req.params;
      const userId = (req as any).userId;
      log.info('Cancel operation requested', { operationId, userId });

      // Try WS manager (kills Python process)
      try {
        const schematicWsManager = getSchematicWsManager();
        const wsOp = schematicWsManager.getOperation(operationId);
        if (wsOp && wsOp.status === 'running') {
          schematicWsManager.failOperation(operationId, `Cancelled by user ${userId}`);
          res.json({ success: true, data: { operationId, status: 'cancelled' } });
          return;
        }
      } catch {
        // WS manager not initialized
      }

      // Try Trigger.dev
      if (triggerService) {
        await triggerService.cancelOperation(operationId);
        res.json({ success: true, data: { operationId, status: 'cancelled' } });
        return;
      }

      res.status(404).json({
        success: false,
        error: { code: 'NOT_FOUND', message: `Operation ${operationId} not found or not cancellable` },
      });
    } catch (error) {
      next(error);
    }
  });

  // ── POST /operations/:operationId/pause ──────────────────────────────
  router.post('/:operationId/pause', async (req: Request, res: Response, next: NextFunction) => {
    try {
      const { operationId } = req.params;
      const userId = (req as any).userId;
      log.info('Pause operation requested', { operationId, userId });

      if (triggerService) {
        await triggerService.pauseOperation(operationId);
        res.json({ success: true, data: { operationId, status: 'paused' } });
        return;
      }

      res.status(404).json({
        success: false,
        error: { code: 'NOT_FOUND', message: `Operation ${operationId} not found or does not support pause` },
      });
    } catch (error) {
      next(error);
    }
  });

  // ── POST /operations/:operationId/resume ─────────────────────────────
  router.post('/:operationId/resume', async (req: Request, res: Response, next: NextFunction) => {
    try {
      const { operationId } = req.params;
      const userId = (req as any).userId;
      log.info('Resume operation requested', { operationId, userId });

      if (triggerService) {
        await triggerService.resumeOperation(operationId);
        res.json({ success: true, data: { operationId, status: 'running' } });
        return;
      }

      res.status(404).json({
        success: false,
        error: { code: 'NOT_FOUND', message: `Operation ${operationId} not found or does not support resume` },
      });
    } catch (error) {
      next(error);
    }
  });

  // ── POST /operations/:operationId/replay ─────────────────────────────
  router.post('/:operationId/replay', async (req: Request, res: Response, next: NextFunction) => {
    try {
      const { operationId } = req.params;
      const userId = (req as any).userId;
      log.info('Replay operation requested', { operationId, userId });

      if (triggerService) {
        const newOperationId = await triggerService.replayOperation(operationId);
        res.json({ success: true, data: { operationId, newOperationId, status: 'queued' } });
        return;
      }

      res.status(404).json({
        success: false,
        error: { code: 'NOT_FOUND', message: `Operation ${operationId} not found or does not support replay` },
      });
    } catch (error) {
      next(error);
    }
  });

  // ── PATCH /operations/:operationId/params ────────────────────────────
  router.patch('/:operationId/params', async (req: Request, res: Response, next: NextFunction) => {
    try {
      const { operationId } = req.params;
      const userId = (req as any).userId;
      const parsed = updateParamsSchema.safeParse(req.body);
      if (!parsed.success) {
        res.status(400).json({
          success: false,
          error: { code: 'VALIDATION_ERROR', message: 'Invalid request body', details: parsed.error.errors },
        });
        return;
      }

      log.info('Update operation params requested', {
        operationId,
        userId,
        paramKeys: Object.keys(parsed.data.parameters),
      });

      // Emit params:update event to running operation via WebSocket
      try {
        const schematicWsManager = getSchematicWsManager();
        const wsOp = schematicWsManager.getOperation(operationId);
        if (wsOp && wsOp.status === 'running') {
          // Access namespace for room-level broadcast (documented internal access)
          const namespace = (schematicWsManager as any).namespace;
          if (namespace) {
            namespace.to(`operation:${operationId}`).emit('params:update', {
              operationId,
              parameters: parsed.data.parameters,
              timestamp: new Date().toISOString(),
            });
          }
          res.json({ success: true, data: { operationId, parameters: parsed.data.parameters } });
          return;
        }
      } catch {
        // WS manager not initialized
      }

      // Try Trigger.dev
      if (triggerService) {
        await triggerService.updateOperationParams(operationId, parsed.data.parameters);
        res.json({ success: true, data: { operationId, parameters: parsed.data.parameters } });
        return;
      }

      res.status(404).json({
        success: false,
        error: { code: 'NOT_FOUND', message: `Operation ${operationId} not found` },
      });
    } catch (error) {
      next(error);
    }
  });

  // ── POST /operations/:operationId/files ──────────────────────────────
  router.post('/:operationId/files', upload.single('file'), async (req: Request, res: Response, next: NextFunction) => {
    try {
      const { operationId } = req.params;
      const userId = (req as any).userId;
      const file = req.file;

      if (!file) {
        res.status(400).json({
          success: false,
          error: { code: 'VALIDATION_ERROR', message: 'No file provided' },
        });
        return;
      }

      log.info('File injection requested', {
        operationId,
        userId,
        fileName: file.originalname,
        fileSize: file.size,
        savedAs: file.filename,
      });

      // Notify running operation about injected file via WebSocket
      try {
        const schematicWsManager = getSchematicWsManager();
        const wsOp = schematicWsManager.getOperation(operationId);
        if (wsOp && wsOp.status === 'running') {
          // Access namespace for room-level broadcast (documented internal access)
          const namespace = (schematicWsManager as any).namespace;
          if (namespace) {
            namespace.to(`operation:${operationId}`).emit('file:injected', {
              operationId,
              file: {
                name: file.originalname,
                path: file.path,
                size: file.size,
                type: 'injected',
                createdAt: new Date().toISOString(),
              },
            });
          }
        }
      } catch {
        // WS manager not initialized
      }

      // Also notify Trigger.dev if applicable
      if (triggerService) {
        try {
          await triggerService.injectFile(operationId, {
            name: file.originalname,
            path: file.path,
            size: file.size,
          });
        } catch (err) {
          log.warn('Failed to notify Trigger.dev of file injection', { operationId, error: err instanceof Error ? err.message : err });
        }
      }

      res.json({
        success: true,
        data: {
          operationId,
          file: {
            name: file.originalname,
            path: file.path,
            size: file.size,
            type: 'injected',
            createdAt: new Date().toISOString(),
          },
        },
      });
    } catch (error) {
      next(error);
    }
  });

  // ── POST /operations/waitpoints/:waitpointId/resolve ─────────────────
  router.post('/waitpoints/:waitpointId/resolve', async (req: Request, res: Response, next: NextFunction) => {
    try {
      const { waitpointId } = req.params;
      const userId = (req as any).userId;
      const parsed = resolveWaitpointSchema.safeParse(req.body);
      if (!parsed.success) {
        res.status(400).json({
          success: false,
          error: { code: 'VALIDATION_ERROR', message: 'Invalid request body', details: parsed.error.errors },
        });
        return;
      }

      const { approve, reason, output } = parsed.data;
      log.info('Waitpoint resolution requested', { waitpointId, approve, reason, userId });

      if (triggerService) {
        if (approve) {
          await triggerService.approveWaitpoint(waitpointId, output);
        } else {
          await triggerService.rejectWaitpoint(waitpointId, reason);
        }

        res.json({
          success: true,
          data: { waitpointId, resolved: true, approved: approve },
        });
        return;
      }

      res.status(404).json({
        success: false,
        error: { code: 'NOT_FOUND', message: `Waitpoint ${waitpointId} not found` },
      });
    } catch (error) {
      next(error);
    }
  });

  return router;
}
