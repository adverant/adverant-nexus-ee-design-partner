/**
 * Operations Repository
 *
 * Persists operation records to PostgreSQL so completed/failed operations
 * survive pod restarts and appear in the Operations Center history.
 */

import { query, buildInsert, buildUpdate } from '../connection.js';
import { log } from '../../utils/logger.js';

const logger = log.child({ service: 'operations-repository' });

// ============================================================================
// Types
// ============================================================================

export interface OperationRow {
  id: string;
  project_id: string;
  project_name: string | null;
  type: string;
  source: string;
  status: string;
  progress: number;
  current_step: string;
  phase: string | null;
  started_at: string;
  completed_at: string | null;
  duration_ms: number | null;
  completed_phases: string[];
  quality_gates: unknown | null;
  result_data: unknown | null;
  error_message: string | null;
  interrupt_reason: string | null;
  parameters: unknown;
  subsystems: unknown;
  event_history: unknown[];
  owner: string | null;
  organization_id: string | null;
  created_at: string;
  updated_at: string;
}

export interface CreateOperationInput {
  id: string;
  project_id: string;
  project_name?: string;
  type?: string;
  source?: string;
  status?: string;
  owner?: string;
  organization_id?: string;
  started_at?: Date;
  parameters?: unknown;
  subsystems?: unknown;
}

export interface CompleteOperationInput {
  status: string;
  progress: number;
  current_step: string;
  phase?: string;
  completed_at: Date;
  duration_ms: number;
  completed_phases: string[];
  quality_gates?: unknown;
  result_data?: unknown;
  error_message?: string;
  event_history?: unknown[];
}

// ============================================================================
// CRUD
// ============================================================================

/**
 * Create a new operation record.
 */
export async function create(input: CreateOperationInput): Promise<OperationRow | null> {
  try {
    const { text, values } = buildInsert('operations', {
      id: input.id,
      project_id: input.project_id,
      project_name: input.project_name || null,
      type: input.type || 'schematic',
      source: input.source || 'ee-design',
      status: input.status || 'running',
      owner: input.owner || null,
      organization_id: input.organization_id || null,
      started_at: input.started_at || new Date(),
      parameters: input.parameters ? JSON.stringify(input.parameters) : '{}',
      subsystems: input.subsystems ? JSON.stringify(input.subsystems) : '[]',
    });

    const result = await query<OperationRow>(text, values, {
      operation: 'operations.create',
    });
    return result.rows[0] || null;
  } catch (error) {
    logger.error('Failed to create operation', error as Error, { operationId: input.id });
    return null;
  }
}

/**
 * Complete (or fail) an operation — update with final state.
 */
export async function complete(id: string, input: CompleteOperationInput): Promise<OperationRow | null> {
  try {
    const data: Record<string, unknown> = {
      status: input.status,
      progress: input.progress,
      current_step: input.current_step,
      completed_at: input.completed_at,
      duration_ms: input.duration_ms,
      completed_phases: input.completed_phases,
    };

    if (input.phase) data.phase = input.phase;
    if (input.quality_gates) data.quality_gates = JSON.stringify(input.quality_gates);
    if (input.result_data) data.result_data = JSON.stringify(input.result_data);
    if (input.error_message) data.error_message = input.error_message;
    if (input.event_history) data.event_history = JSON.stringify(input.event_history);

    const { text, values } = buildUpdate('operations', data, 'id', id);

    const result = await query<OperationRow>(text, values, {
      operation: 'operations.complete',
    });
    return result.rows[0] || null;
  } catch (error) {
    logger.error('Failed to complete operation', error as Error, { operationId: id });
    return null;
  }
}

/**
 * Find an operation by ID.
 */
export async function findById(id: string): Promise<OperationRow | null> {
  try {
    const result = await query<OperationRow>(
      'SELECT * FROM operations WHERE id = $1',
      [id],
      { operation: 'operations.findById' }
    );
    return result.rows[0] || null;
  } catch (error) {
    logger.error('Failed to find operation', error as Error, { operationId: id });
    return null;
  }
}

/**
 * Find all operations, optionally filtered.
 */
export async function findAll(filters?: {
  status?: string;
  source?: string;
  projectId?: string;
  type?: string;
  owner?: string;
  limit?: number;
  offset?: number;
}): Promise<{ operations: OperationRow[]; total: number }> {
  try {
    const conditions: string[] = [];
    const values: unknown[] = [];
    let paramIdx = 1;

    if (filters?.status && filters.status !== 'all') {
      conditions.push(`status = $${paramIdx++}`);
      values.push(filters.status);
    }
    if (filters?.source && filters.source !== 'all') {
      conditions.push(`source = $${paramIdx++}`);
      values.push(filters.source);
    }
    if (filters?.projectId) {
      conditions.push(`project_id = $${paramIdx++}`);
      values.push(filters.projectId);
    }
    if (filters?.type && filters.type !== 'all') {
      conditions.push(`type = $${paramIdx++}`);
      values.push(filters.type);
    }
    if (filters?.owner) {
      conditions.push(`owner = $${paramIdx++}`);
      values.push(filters.owner);
    }

    const where = conditions.length > 0 ? `WHERE ${conditions.join(' AND ')}` : '';

    // Count total
    const countResult = await query<{ count: string }>(
      `SELECT COUNT(*) as count FROM operations ${where}`,
      values,
      { operation: 'operations.count' }
    );
    const total = parseInt(countResult.rows[0]?.count || '0', 10);

    // Fetch rows
    const limit = filters?.limit || 100;
    const offset = filters?.offset || 0;
    values.push(limit);
    values.push(offset);

    const result = await query<OperationRow>(
      `SELECT * FROM operations ${where} ORDER BY created_at DESC LIMIT $${paramIdx++} OFFSET $${paramIdx++}`,
      values,
      { operation: 'operations.findAll' }
    );

    return { operations: result.rows, total };
  } catch (error) {
    logger.error('Failed to find operations', error as Error);
    return { operations: [], total: 0 };
  }
}

/**
 * Get stats for the dashboard summary cards.
 */
export async function getStats(owner?: string): Promise<{
  active: number;
  queued: number;
  failed: number;
  completedToday: number;
  pendingApprovals: number;
}> {
  try {
    const ownerFilter = owner ? `AND owner = $1` : '';
    const values = owner ? [owner] : [];

    const result = await query<{
      active: string;
      queued: string;
      failed: string;
      completed_today: string;
      pending_approvals: string;
    }>(
      `SELECT
        COUNT(*) FILTER (WHERE status = 'running') as active,
        COUNT(*) FILTER (WHERE status = 'queued') as queued,
        COUNT(*) FILTER (WHERE status = 'failed') as failed,
        COUNT(*) FILTER (WHERE status = 'completed' AND completed_at >= CURRENT_DATE) as completed_today,
        COUNT(*) FILTER (WHERE status = 'waiting-approval') as pending_approvals
      FROM operations
      WHERE 1=1 ${ownerFilter}`,
      values,
      { operation: 'operations.getStats' }
    );

    const row = result.rows[0];
    return {
      active: parseInt(row?.active || '0', 10),
      queued: parseInt(row?.queued || '0', 10),
      failed: parseInt(row?.failed || '0', 10),
      completedToday: parseInt(row?.completed_today || '0', 10),
      pendingApprovals: parseInt(row?.pending_approvals || '0', 10),
    };
  } catch (error) {
    logger.error('Failed to get operation stats', error as Error);
    return { active: 0, queued: 0, failed: 0, completedToday: 0, pendingApprovals: 0 };
  }
}

/**
 * Update operation progress (lightweight update, no event history).
 */
export async function updateProgress(id: string, progress: number, currentStep: string, phase?: string): Promise<void> {
  try {
    const data: Record<string, unknown> = { progress, current_step: currentStep };
    if (phase) data.phase = phase;
    const { text, values } = buildUpdate('operations', data, 'id', id);
    await query(text, values, { operation: 'operations.updateProgress' });
  } catch (error) {
    logger.warn('Failed to update operation progress', { operationId: id, error: (error as Error).message });
  }
}

/**
 * Mark an operation as interrupted (called on server startup for orphaned running ops).
 */
export async function markInterrupted(id: string, reason: string): Promise<void> {
  try {
    const { text, values } = buildUpdate(
      'operations',
      { status: 'interrupted', interrupt_reason: reason, completed_at: new Date() },
      'id',
      id
    );
    await query(text, values, { operation: 'operations.markInterrupted' });
  } catch (error) {
    logger.warn('Failed to mark operation interrupted', { operationId: id });
  }
}

/**
 * Find all operations still in 'running' status that are older than maxAgeMs.
 * Used on startup to detect ops orphaned by a pod restart.
 */
export async function findOrphanedRunning(maxAgeMs = 300_000): Promise<OperationRow[]> {
  try {
    const cutoff = new Date(Date.now() - maxAgeMs);
    const result = await query<OperationRow>(
      `SELECT * FROM operations WHERE status = 'running' AND started_at < $1`,
      [cutoff],
      { operation: 'operations.findOrphanedRunning' }
    );
    return result.rows;
  } catch (error) {
    logger.error('Failed to find orphaned operations', error as Error);
    return [];
  }
}

export default {
  create,
  complete,
  updateProgress,
  markInterrupted,
  findOrphanedRunning,
  findById,
  findAll,
  getStats,
};
