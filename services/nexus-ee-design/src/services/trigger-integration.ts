/**
 * Trigger.dev Integration Service
 *
 * Provides a typed client for communicating with the Nexus Trigger.dev plugin
 * from the EE Design Partner backend. This service:
 *
 * - Creates and manages Trigger.dev runs for MAPO pipeline phases
 * - Translates between Trigger.dev run model and UnifiedOperation model
 * - Handles waitpoint lifecycle (create, approve, reject)
 * - Provides operation stats aggregated from Trigger.dev
 *
 * Configuration:
 *   TRIGGER_API_URL  — Base URL of the Trigger.dev plugin API (default: http://nexus-trigger:3500)
 *   TRIGGER_API_KEY  — API key for auth with the Trigger.dev plugin
 */

import log from '../utils/logger.js';

const TRIGGER_API_URL = process.env.TRIGGER_API_URL || 'http://nexus-trigger.nexus.svc.cluster.local:3500';
const TRIGGER_API_KEY = process.env.TRIGGER_API_KEY || '';

// ─── Types ───────────────────────────────────────────────────────────────

export interface TriggerRunDTO {
  id: string;
  taskId: string;
  taskSlug: string;
  status: string;
  payload?: Record<string, unknown>;
  output?: Record<string, unknown>;
  error?: string;
  startedAt?: string;
  completedAt?: string;
  duration?: number;
  tags?: string[];
  createdAt: string;
}

export interface UnifiedOperationDTO {
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
  phases?: Array<{ id: string; label: string; status: string; progress: number }>;
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
    canOverrideGates: boolean;
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
  triggerRunId: string;
  iteration?: number;
  totalIterations?: number;
  tags: string[];
}

export interface TriggerWaitpointDTO {
  id: string;
  runId: string;
  type: string;
  description?: string;
  operationName?: string;
  projectName?: string;
  qualityGates?: Array<{
    name: string;
    passed: boolean;
    threshold: number;
    actual: number;
  }>;
  createdAt: string;
  status: string;
}

export interface OperationStatsDTO {
  active: number;
  queued: number;
  failed: number;
  completedToday: number;
  pendingApprovals: number;
}

export interface ListOperationsOptions {
  status?: string;
  projectId?: string;
  type?: string;
  limit?: number;
  offset?: number;
}

export interface LogEntry {
  id: string;
  operationId: string;
  timestamp: string;
  level: string;
  message: string;
  phase?: string;
  data?: Record<string, unknown>;
}

interface FileInfo {
  name: string;
  path: string;
  size: number;
}

// ─── Status Mapping ──────────────────────────────────────────────────────

const TRIGGER_STATUS_MAP: Record<string, string> = {
  PENDING: 'queued',
  QUEUED: 'queued',
  EXECUTING: 'running',
  REATTEMPTING: 'running',
  WAITING: 'waiting-approval',
  WAITING_FOR_DEPLOY: 'queued',
  COMPLETED: 'completed',
  SYSTEM_FAILURE: 'failed',
  INTERRUPTED: 'failed',
  CRASHED: 'failed',
  CANCELED: 'cancelled',
  CANCELLED: 'cancelled',
  FROZEN: 'paused',
  PAUSED: 'paused',
  DELAYED: 'queued',
  STARTING: 'queued',
  EXPIRED: 'failed',
  TIMED_OUT: 'failed',
};

/**
 * Infer operation type from Trigger.dev task identifier
 */
function inferOperationType(taskSlug: string): string {
  if (taskSlug.includes('resolve-symbols') || taskSlug.includes('symbol-assembly')) return 'symbol-assembly';
  if (taskSlug.includes('generate-connections')) return 'schematic';
  if (taskSlug.includes('optimize-layout')) return 'schematic';
  if (taskSlug.includes('route-wires')) return 'schematic';
  if (taskSlug.includes('assemble-schematic')) return 'schematic';
  if (taskSlug.includes('smoke-test')) return 'schematic';
  if (taskSlug.includes('visual-validate')) return 'schematic';
  if (taskSlug.includes('export-artifacts')) return 'schematic';
  if (taskSlug.includes('mapo-pipeline')) return 'schematic';
  if (taskSlug.includes('ralph-loop')) return 'schematic';
  if (taskSlug.includes('pcb')) return 'pcb-layout';
  if (taskSlug.includes('simulation')) return 'simulation';
  return 'custom';
}

/**
 * Infer human-readable operation name from task identifier and payload
 */
function inferOperationName(taskSlug: string, payload?: Record<string, unknown>): string {
  const projectName = (payload?.project_name as string) || '';
  const nameMap: Record<string, string> = {
    'ee-design/resolve-symbols': 'Symbol Resolution',
    'ee-design/generate-connections': 'Connection Generation',
    'ee-design/optimize-layout': 'Layout Optimization',
    'ee-design/route-wires': 'Wire Routing',
    'ee-design/assemble-schematic': 'Schematic Assembly',
    'ee-design/smoke-test': 'Smoke Test',
    'ee-design/visual-validate': 'Visual Validation',
    'ee-design/export-artifacts': 'Export Artifacts',
    'ee-design/mapo-pipeline': 'MAPO Pipeline',
    'ee-design/ralph-loop': 'Continuous Loop',
  };

  const baseName = nameMap[taskSlug] || taskSlug;
  return projectName ? `${baseName} — ${projectName}` : baseName;
}

// ─── Service ─────────────────────────────────────────────────────────────

export class TriggerIntegrationService {
  private baseUrl: string;
  private apiKey: string;

  constructor(baseUrl?: string, apiKey?: string) {
    this.baseUrl = baseUrl || TRIGGER_API_URL;
    this.apiKey = apiKey || TRIGGER_API_KEY;
  }

  private async fetch(path: string, options: RequestInit = {}): Promise<any> {
    const url = `${this.baseUrl}/api/v1${path}`;
    const headers: Record<string, string> = {
      'Content-Type': 'application/json',
      ...(this.apiKey ? { Authorization: `Bearer ${this.apiKey}` } : {}),
      ...(options.headers as Record<string, string> || {}),
    };

    let res: Response;
    try {
      res = await fetch(url, { ...options, headers });
    } catch (err) {
      throw new Error(
        `Trigger.dev API network error: ${err instanceof Error ? err.message : String(err)} — URL: ${url}`
      );
    }

    if (!res.ok) {
      const body = await res.text().catch(() => '(unable to read response body)');
      throw new Error(
        `Trigger.dev API error: ${res.status} ${res.statusText} — ${body.slice(0, 500)}`
      );
    }

    const contentType = res.headers.get('content-type') || '';
    if (!contentType.includes('application/json')) {
      const body = await res.text().catch(() => '');
      throw new Error(
        `Trigger.dev API returned non-JSON response (${contentType}): ${body.slice(0, 200)}`
      );
    }

    try {
      return await res.json();
    } catch (err) {
      throw new Error(
        `Trigger.dev API returned invalid JSON: ${err instanceof Error ? err.message : String(err)}`
      );
    }
  }

  // ── Run → UnifiedOperation Conversion ────────────────────────────────

  private runToOperation(run: TriggerRunDTO): UnifiedOperationDTO {
    const mappedStatus = TRIGGER_STATUS_MAP[run.status];
    if (!mappedStatus) {
      log.warn('Unmapped Trigger.dev status encountered', {
        runId: run.id,
        status: run.status,
        message: 'Defaulting to "running" — update TRIGGER_STATUS_MAP to handle this status',
      });
    }
    const status = mappedStatus || 'running';
    const isActive = status === 'running' || status === 'queued';
    const isTerminal = status === 'completed' || status === 'failed' || status === 'cancelled';
    const type = inferOperationType(run.taskSlug);

    return {
      id: run.id,
      name: inferOperationName(run.taskSlug, run.payload),
      type,
      source: 'trigger',
      sourceId: run.id,
      projectId: (run.payload?.project_id as string) || (run.payload?.projectId as string) || '',
      projectName: (run.payload?.project_name as string) || (run.payload?.projectName as string) || '',
      status,
      progress: (run.payload?.progress as number) ?? null,
      currentStep: (run.payload?.current_step as string) || '',
      createdAt: run.createdAt,
      startedAt: run.startedAt,
      completedAt: run.completedAt,
      duration: run.duration,
      error: run.error,
      output: run.output,
      parameters: run.payload,
      capabilities: {
        canCancel: isActive,
        canPause: status === 'running',
        canResume: status === 'paused',
        canReplay: isTerminal,
        canModifyParams: status === 'running',
        canInjectFiles: status === 'running',
        hasLogs: true,
        hasQualityGates: type === 'schematic',
        hasTerminal: false,
        canOverrideGates: status === 'running',
      },
      triggerRunId: run.id,
      iteration: (run.payload?.iteration as number) ?? (run.output?.finalIteration as number) ?? undefined,
      totalIterations: (run.payload?.maxIterations as number) ?? (run.output?.totalIterations as number) ?? undefined,
      tags: run.tags || [type, 'trigger'],
    };
  }

  // ── Operations CRUD ──────────────────────────────────────────────────

  async listOperations(options: ListOperationsOptions = {}): Promise<UnifiedOperationDTO[]> {
    const params = new URLSearchParams();
    if (options.status) params.set('status', options.status);
    if (options.projectId) params.set('projectId', options.projectId);
    if (options.limit) params.set('limit', String(options.limit));
    if (options.offset) params.set('offset', String(options.offset));

    // Filter to EE Design tasks only
    params.set('taskSlug', 'ee-design/*');

    const queryStr = params.toString();
    const data = await this.fetch(`/runs${queryStr ? `?${queryStr}` : ''}`);
    const runs: TriggerRunDTO[] = data.data || [];

    return runs.map((run) => this.runToOperation(run));
  }

  async getOperation(operationId: string): Promise<UnifiedOperationDTO | null> {
    try {
      const data = await this.fetch(`/runs/${operationId}`);
      if (!data.data) return null;
      return this.runToOperation(data.data);
    } catch (err) {
      if (err instanceof Error && err.message.includes('404')) return null;
      throw err;
    }
  }

  async getOperationLogs(
    operationId: string,
    options?: { level?: string; limit?: number; offset?: number }
  ): Promise<LogEntry[]> {
    const params = new URLSearchParams();
    if (options?.level) params.set('level', options.level);
    if (options?.limit) params.set('limit', String(options.limit));
    if (options?.offset) params.set('offset', String(options.offset));

    const queryStr = params.toString();

    try {
      const data = await this.fetch(`/runs/${operationId}/logs${queryStr ? `?${queryStr}` : ''}`);
      const rawLogs: any[] = data.data?.logs || data.data || [];

      return rawLogs.map((l: any, idx: number) => ({
        id: l.id || `log-${operationId}-${idx}`,
        operationId,
        timestamp: l.timestamp || l.createdAt || new Date().toISOString(),
        level: l.level || 'info',
        message: l.message || l.text || '',
        phase: l.phase,
        data: l.data,
      }));
    } catch {
      return [];
    }
  }

  // ── Actions ──────────────────────────────────────────────────────────

  async cancelOperation(operationId: string): Promise<void> {
    await this.fetch(`/runs/${operationId}/cancel`, { method: 'POST' });
    log.info('Operation cancelled via Trigger.dev', { operationId });
  }

  async pauseOperation(operationId: string): Promise<void> {
    // Trigger.dev uses waitpoints for pausing — we create a manual waitpoint
    await this.fetch(`/runs/${operationId}/pause`, { method: 'POST' });
    log.info('Operation paused via Trigger.dev', { operationId });
  }

  async resumeOperation(operationId: string): Promise<void> {
    await this.fetch(`/runs/${operationId}/resume`, { method: 'POST' });
    log.info('Operation resumed via Trigger.dev', { operationId });
  }

  async replayOperation(operationId: string): Promise<string> {
    const data = await this.fetch(`/runs/${operationId}/replay`, { method: 'POST' });
    const newId = data.data?.newRunId || data.data?.id || '';
    log.info('Operation replayed via Trigger.dev', { operationId, newOperationId: newId });
    return newId;
  }

  async updateOperationParams(operationId: string, params: Record<string, unknown>): Promise<void> {
    await this.fetch(`/runs/${operationId}/metadata`, {
      method: 'PATCH',
      body: JSON.stringify({ metadata: params }),
    });
    log.info('Operation params updated via Trigger.dev', { operationId, paramKeys: Object.keys(params) });
  }

  async injectFile(operationId: string, file: FileInfo): Promise<void> {
    await this.fetch(`/runs/${operationId}/metadata`, {
      method: 'PATCH',
      body: JSON.stringify({
        metadata: {
          injectedFile: {
            name: file.name,
            path: file.path,
            size: file.size,
            injectedAt: new Date().toISOString(),
          },
        },
      }),
    });
    log.info('File injection recorded via Trigger.dev', { operationId, fileName: file.name });
  }

  // ── Waitpoints ───────────────────────────────────────────────────────

  async listWaitpoints(): Promise<TriggerWaitpointDTO[]> {
    const data = await this.fetch('/waitpoints');
    return data.data || [];
  }

  async approveWaitpoint(waitpointId: string, output?: Record<string, unknown>): Promise<void> {
    await this.fetch(`/waitpoints/${waitpointId}/complete`, {
      method: 'POST',
      body: JSON.stringify({
        output: { approved: true, ...(output || {}) },
        completedBy: 'operations-center',
      }),
    });
    log.info('Waitpoint approved via Trigger.dev', { waitpointId });
  }

  async rejectWaitpoint(waitpointId: string, reason?: string): Promise<void> {
    await this.fetch(`/waitpoints/${waitpointId}/complete`, {
      method: 'POST',
      body: JSON.stringify({
        output: { approved: false, reason: reason || 'Rejected by user' },
        completedBy: 'operations-center',
      }),
    });
    log.info('Waitpoint rejected via Trigger.dev', { waitpointId, reason });
  }

  // ── Stats ────────────────────────────────────────────────────────────

  async getStats(): Promise<OperationStatsDTO> {
    try {
      // Fetch recent EE Design runs to compute stats
      const params = new URLSearchParams();
      params.set('taskSlug', 'ee-design/*');
      params.set('limit', '200');

      const data = await this.fetch(`/runs?${params}`);
      const runs: TriggerRunDTO[] = data.data || [];

      const todayStart = new Date();
      todayStart.setHours(0, 0, 0, 0);

      let active = 0;
      let queued = 0;
      let failed = 0;
      let completedToday = 0;

      for (const run of runs) {
        const status = TRIGGER_STATUS_MAP[run.status] || run.status;
        if (status === 'running') active++;
        if (status === 'queued') queued++;
        if (status === 'failed') failed++;
        if (status === 'completed' && run.completedAt) {
          const completedAt = new Date(run.completedAt);
          if (completedAt >= todayStart) completedToday++;
        }
      }

      // Count pending waitpoints
      let pendingApprovals = 0;
      try {
        const wpData = await this.fetch('/waitpoints');
        pendingApprovals = (wpData.data || []).length;
      } catch {
        // Waitpoints endpoint may not be available
      }

      return { active, queued, failed, completedToday, pendingApprovals };
    } catch (err) {
      log.warn('Failed to compute Trigger.dev stats', { error: err instanceof Error ? err.message : err });
      return { active: 0, queued: 0, failed: 0, completedToday: 0, pendingApprovals: 0 };
    }
  }

  // ── Create Run ───────────────────────────────────────────────────────

  /**
   * Create a new Trigger.dev run for an EE Design task
   */
  async createRun(taskIdentifier: string, payload: Record<string, unknown>): Promise<string> {
    const data = await this.fetch('/runs', {
      method: 'POST',
      body: JSON.stringify({
        taskIdentifier,
        payload,
        tags: ['ee-design', inferOperationType(taskIdentifier)],
      }),
    });

    const runId = data.data?.id || data.data?.runId || '';
    log.info('Created Trigger.dev run', { taskIdentifier, runId });
    return runId;
  }

  // ── Health Check ─────────────────────────────────────────────────────

  async healthCheck(): Promise<{ healthy: boolean; latencyMs: number }> {
    const start = Date.now();
    try {
      await this.fetch('/health');
      return { healthy: true, latencyMs: Date.now() - start };
    } catch {
      return { healthy: false, latencyMs: Date.now() - start };
    }
  }
}
