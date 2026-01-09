/**
 * EE Design Partner - Simulation Repository
 *
 * Database operations for simulation jobs. Handles SPICE, thermal, signal integrity,
 * RF, and EMC simulation jobs with proper status tracking and result storage.
 */

import {
  query,
  withTransaction,
  clientQuery,
  buildInsert,
  buildUpdate,
  DatabaseError,
} from '../connection.js';
import { NotFoundError, ValidationError } from '../../utils/errors.js';
import { log, Logger } from '../../utils/logger.js';
import type {
  Simulation,
  SimulationType,
  SimulationStatus,
  SimulationInput,
  SimulationResults,
} from '../../types/index.js';

// ============================================================================
// Types
// ============================================================================

export interface CreateSimulationInput {
  projectId: string;
  schematicId?: string;
  pcbLayoutId?: string;
  name: string;
  simulationType: SimulationType;
  config?: Record<string, unknown>;
  testBench?: string;
  parameters?: Record<string, unknown>;
  priority?: number;
  timeoutMs?: number;
}

export interface UpdateSimulationInput {
  name?: string;
  config?: Record<string, unknown>;
  testBench?: string;
  parameters?: Record<string, unknown>;
  priority?: number;
  timeoutMs?: number;
}

export interface SimulationFilters {
  type?: SimulationType;
  status?: SimulationStatus;
  schematicId?: string;
  pcbLayoutId?: string;
  passed?: boolean;
}

interface SimulationRow {
  id: string;
  project_id: string;
  schematic_id: string | null;
  pcb_layout_id: string | null;
  name: string;
  simulation_type: SimulationType;
  status: SimulationStatus;
  priority: number;
  config: Record<string, unknown>;
  test_bench: string | null;
  parameters: Record<string, unknown>;
  results: SimulationResults | null;
  waveforms: object | null;
  images: object | null;
  metrics: Record<string, unknown> | null;
  passed: boolean | null;
  score: number | null;
  error_message: string | null;
  error_details: object | null;
  retry_count: number;
  max_retries: number;
  started_at: Date | null;
  completed_at: Date | null;
  duration_ms: number | null;
  timeout_ms: number;
  worker_id: string | null;
  worker_host: string | null;
  created_at: Date;
  updated_at: Date;
}

// ============================================================================
// Repository Implementation
// ============================================================================

const repoLogger: Logger = log.child({ service: 'simulation-repository' });

/**
 * Map a database row to a Simulation object.
 */
function mapRowToSimulation(row: SimulationRow): Simulation {
  return {
    id: row.id,
    projectId: row.project_id,
    type: row.simulation_type,
    name: row.name,
    status: row.status,
    input: {
      schematicId: row.schematic_id || undefined,
      pcbLayoutId: row.pcb_layout_id || undefined,
      parameters: row.parameters || {},
      testBench: row.test_bench || undefined,
    },
    results: row.results || undefined,
    startedAt: row.started_at?.toISOString(),
    completedAt: row.completed_at?.toISOString(),
    error: row.error_message || undefined,
  };
}

/**
 * Create a new simulation job.
 *
 * @param input - Simulation creation data
 * @returns The created simulation
 */
export async function create(input: CreateSimulationInput): Promise<Simulation> {
  const logger = repoLogger.child({ operation: 'create' });

  if (!input.projectId || input.projectId.trim().length === 0) {
    throw new ValidationError('Project ID is required', {
      operation: 'create',
      input,
    });
  }

  if (!input.name || input.name.trim().length === 0) {
    throw new ValidationError('Simulation name is required', {
      operation: 'create',
      input,
    });
  }

  if (!input.simulationType) {
    throw new ValidationError('Simulation type is required', {
      operation: 'create',
      input,
    });
  }

  const validTypes: SimulationType[] = [
    'spice_dc',
    'spice_ac',
    'spice_transient',
    'spice_noise',
    'spice_monte_carlo',
    'thermal_steady_state',
    'thermal_transient',
    'thermal_cfd',
    'signal_integrity',
    'power_integrity',
    'rf_sparameters',
    'rf_field_pattern',
    'emc_radiated',
    'emc_conducted',
    'stress_thermal_cycling',
    'stress_vibration',
    'reliability_mtbf',
  ];

  if (!validTypes.includes(input.simulationType)) {
    throw new ValidationError(`Invalid simulation type: ${input.simulationType}`, {
      operation: 'create',
      validTypes,
    });
  }

  const data: Record<string, unknown> = {
    project_id: input.projectId,
    schematic_id: input.schematicId || null,
    pcb_layout_id: input.pcbLayoutId || null,
    name: input.name.trim(),
    simulation_type: input.simulationType,
    config: JSON.stringify(input.config || {}),
    test_bench: input.testBench || null,
    parameters: JSON.stringify(input.parameters || {}),
    priority: input.priority || 5,
    timeout_ms: input.timeoutMs || 3600000,
  };

  logger.debug('Creating simulation', {
    name: input.name,
    projectId: input.projectId,
    type: input.simulationType,
  });

  const { text, values } = buildInsert('simulations', data);
  const result = await query<SimulationRow>(text, values, { operation: 'create_simulation' });

  if (result.rows.length === 0) {
    throw new DatabaseError(
      'Failed to create simulation - no row returned',
      new Error('Insert returned no rows'),
      { operation: 'create_simulation' }
    );
  }

  const simulation = mapRowToSimulation(result.rows[0]);
  logger.info('Simulation created', { simulationId: simulation.id, name: simulation.name });

  return simulation;
}

/**
 * Find a simulation by its ID.
 *
 * @param id - Simulation UUID
 * @returns The simulation or null if not found
 */
export async function findById(id: string): Promise<Simulation | null> {
  const logger = repoLogger.child({ operation: 'findById', simulationId: id });

  if (!id || id.trim().length === 0) {
    throw new ValidationError('Simulation ID is required', {
      operation: 'findById',
    });
  }

  logger.debug('Finding simulation by ID');

  const result = await query<SimulationRow>(
    'SELECT * FROM simulations WHERE id = $1',
    [id],
    { operation: 'find_simulation_by_id' }
  );

  if (result.rows.length === 0) {
    logger.debug('Simulation not found');
    return null;
  }

  return mapRowToSimulation(result.rows[0]);
}

/**
 * Find all simulations for a project with optional filters.
 *
 * @param projectId - Project UUID
 * @param filters - Optional filters
 * @returns Array of simulations for the project
 */
export async function findByProject(
  projectId: string,
  filters?: SimulationFilters
): Promise<Simulation[]> {
  const logger = repoLogger.child({ operation: 'findByProject', projectId });

  if (!projectId || projectId.trim().length === 0) {
    throw new ValidationError('Project ID is required', {
      operation: 'findByProject',
    });
  }

  const conditions: string[] = ['project_id = $1'];
  const values: unknown[] = [projectId];
  let paramIndex = 2;

  if (filters?.type) {
    conditions.push(`simulation_type = $${paramIndex++}`);
    values.push(filters.type);
  }

  if (filters?.status) {
    conditions.push(`status = $${paramIndex++}`);
    values.push(filters.status);
  }

  if (filters?.schematicId) {
    conditions.push(`schematic_id = $${paramIndex++}`);
    values.push(filters.schematicId);
  }

  if (filters?.pcbLayoutId) {
    conditions.push(`pcb_layout_id = $${paramIndex++}`);
    values.push(filters.pcbLayoutId);
  }

  if (filters?.passed !== undefined) {
    conditions.push(`passed = $${paramIndex++}`);
    values.push(filters.passed);
  }

  const sql = `SELECT * FROM simulations WHERE ${conditions.join(' AND ')} ORDER BY created_at DESC`;

  logger.debug('Finding simulations by project', { filters });

  const result = await query<SimulationRow>(sql, values, { operation: 'find_simulations_by_project' });

  logger.debug('Found simulations', { count: result.rows.length });

  return result.rows.map(mapRowToSimulation);
}

/**
 * Update simulation status.
 *
 * @param id - Simulation UUID
 * @param status - New status
 * @param workerId - Optional worker ID claiming the job
 * @returns The updated simulation
 */
export async function updateStatus(
  id: string,
  status: SimulationStatus,
  workerId?: string
): Promise<Simulation> {
  const logger = repoLogger.child({ operation: 'updateStatus', simulationId: id });

  if (!id || id.trim().length === 0) {
    throw new ValidationError('Simulation ID is required', {
      operation: 'updateStatus',
    });
  }

  const validStatuses: SimulationStatus[] = [
    'pending',
    'running',
    'completed',
    'failed',
    'cancelled',
  ];

  // Also allow database-specific statuses
  const extendedStatuses = [...validStatuses, 'queued', 'timeout'];

  if (!extendedStatuses.includes(status)) {
    throw new ValidationError(`Invalid status: ${status}`, {
      operation: 'updateStatus',
      validStatuses: extendedStatuses,
    });
  }

  // Check if simulation exists
  const existing = await findById(id);
  if (!existing) {
    throw new NotFoundError('Simulation', id, { operation: 'updateStatus' });
  }

  const updateData: Record<string, unknown> = { status };

  // Set started_at when transitioning to running
  if (status === 'running' && existing.status !== 'running') {
    updateData.started_at = new Date();
    if (workerId) {
      updateData.worker_id = workerId;
    }
  }

  // Set completed_at for terminal states
  if (['completed', 'failed', 'cancelled', 'timeout'].includes(status) && !existing.completedAt) {
    updateData.completed_at = new Date();
    if (existing.startedAt) {
      updateData.duration_ms = Date.now() - new Date(existing.startedAt).getTime();
    }
  }

  logger.debug('Updating simulation status', {
    currentStatus: existing.status,
    newStatus: status,
    workerId,
  });

  const setClauses = Object.keys(updateData).map((key, i) => `${key} = $${i + 1}`);
  const values = [...Object.values(updateData), id];

  const result = await query<SimulationRow>(
    `UPDATE simulations SET ${setClauses.join(', ')} WHERE id = $${values.length} RETURNING *`,
    values,
    { operation: 'update_simulation_status' }
  );

  if (result.rows.length === 0) {
    throw new DatabaseError(
      'Failed to update simulation status - no row returned',
      new Error('Update returned no rows'),
      { operation: 'update_simulation_status' }
    );
  }

  const simulation = mapRowToSimulation(result.rows[0]);
  logger.info('Simulation status updated', {
    simulationId: simulation.id,
    previousStatus: existing.status,
    newStatus: status,
  });

  return simulation;
}

/**
 * Complete a simulation with results.
 *
 * @param id - Simulation UUID
 * @param results - Simulation results
 * @returns The completed simulation
 */
export async function complete(id: string, results: SimulationResults): Promise<Simulation> {
  const logger = repoLogger.child({ operation: 'complete', simulationId: id });

  if (!id || id.trim().length === 0) {
    throw new ValidationError('Simulation ID is required', {
      operation: 'complete',
    });
  }

  // Check if simulation exists
  const existing = await findById(id);
  if (!existing) {
    throw new NotFoundError('Simulation', id, { operation: 'complete' });
  }

  const completedAt = new Date();
  let durationMs: number | null = null;

  if (existing.startedAt) {
    durationMs = completedAt.getTime() - new Date(existing.startedAt).getTime();
  }

  logger.debug('Completing simulation', {
    passed: results.passed,
    score: results.score,
    durationMs,
  });

  const result = await query<SimulationRow>(
    `UPDATE simulations
     SET status = 'completed',
         results = $1,
         waveforms = $2,
         images = $3,
         metrics = $4,
         passed = $5,
         score = $6,
         completed_at = $7,
         duration_ms = $8
     WHERE id = $9
     RETURNING *`,
    [
      JSON.stringify(results),
      results.waveforms ? JSON.stringify(results.waveforms) : null,
      results.images ? JSON.stringify(results.images) : null,
      JSON.stringify(results.metrics || {}),
      results.passed,
      results.score,
      completedAt,
      durationMs,
      id,
    ],
    { operation: 'complete_simulation' }
  );

  if (result.rows.length === 0) {
    throw new DatabaseError(
      'Failed to complete simulation - no row returned',
      new Error('Update returned no rows'),
      { operation: 'complete_simulation' }
    );
  }

  const simulation = mapRowToSimulation(result.rows[0]);
  logger.info('Simulation completed', {
    simulationId: simulation.id,
    passed: results.passed,
    score: results.score,
    durationMs,
  });

  return simulation;
}

/**
 * Mark a simulation as failed.
 *
 * @param id - Simulation UUID
 * @param errorMessage - Error message
 * @param errorDetails - Optional error details
 * @returns The failed simulation
 */
export async function fail(
  id: string,
  errorMessage: string,
  errorDetails?: object
): Promise<Simulation> {
  const logger = repoLogger.child({ operation: 'fail', simulationId: id });

  if (!id || id.trim().length === 0) {
    throw new ValidationError('Simulation ID is required', {
      operation: 'fail',
    });
  }

  // Check if simulation exists
  const existing = await findById(id);
  if (!existing) {
    throw new NotFoundError('Simulation', id, { operation: 'fail' });
  }

  const completedAt = new Date();
  let durationMs: number | null = null;

  if (existing.startedAt) {
    durationMs = completedAt.getTime() - new Date(existing.startedAt).getTime();
  }

  logger.debug('Failing simulation', {
    errorMessage,
    durationMs,
  });

  const result = await query<SimulationRow>(
    `UPDATE simulations
     SET status = 'failed',
         error_message = $1,
         error_details = $2,
         passed = false,
         completed_at = $3,
         duration_ms = $4,
         retry_count = retry_count + 1
     WHERE id = $5
     RETURNING *`,
    [
      errorMessage,
      errorDetails ? JSON.stringify(errorDetails) : null,
      completedAt,
      durationMs,
      id,
    ],
    { operation: 'fail_simulation' }
  );

  if (result.rows.length === 0) {
    throw new DatabaseError(
      'Failed to mark simulation as failed - no row returned',
      new Error('Update returned no rows'),
      { operation: 'fail_simulation' }
    );
  }

  const simulation = mapRowToSimulation(result.rows[0]);
  logger.info('Simulation failed', {
    simulationId: simulation.id,
    errorMessage,
    durationMs,
  });

  return simulation;
}

/**
 * Update simulation parameters.
 *
 * @param id - Simulation UUID
 * @param data - Fields to update
 * @returns The updated simulation
 */
export async function update(id: string, data: UpdateSimulationInput): Promise<Simulation> {
  const logger = repoLogger.child({ operation: 'update', simulationId: id });

  if (!id || id.trim().length === 0) {
    throw new ValidationError('Simulation ID is required', {
      operation: 'update',
    });
  }

  // Check if simulation exists
  const existing = await findById(id);
  if (!existing) {
    throw new NotFoundError('Simulation', id, { operation: 'update' });
  }

  // Don't allow updates to completed/failed simulations
  if (['completed', 'failed', 'cancelled'].includes(existing.status)) {
    throw new ValidationError(`Cannot update simulation in ${existing.status} status`, {
      operation: 'update',
      simulationId: id,
      status: existing.status,
    });
  }

  const updateData: Record<string, unknown> = {};

  if (data.name !== undefined) {
    if (data.name.trim().length === 0) {
      throw new ValidationError('Simulation name cannot be empty', {
        operation: 'update',
        simulationId: id,
      });
    }
    updateData.name = data.name.trim();
  }

  if (data.config !== undefined) {
    updateData.config = JSON.stringify(data.config);
  }

  if (data.testBench !== undefined) {
    updateData.test_bench = data.testBench || null;
  }

  if (data.parameters !== undefined) {
    updateData.parameters = JSON.stringify(data.parameters);
  }

  if (data.priority !== undefined) {
    if (data.priority < 1 || data.priority > 10) {
      throw new ValidationError('Priority must be between 1 and 10', {
        operation: 'update',
        simulationId: id,
      });
    }
    updateData.priority = data.priority;
  }

  if (data.timeoutMs !== undefined) {
    updateData.timeout_ms = data.timeoutMs;
  }

  if (Object.keys(updateData).length === 0) {
    logger.debug('No fields to update');
    return existing;
  }

  logger.debug('Updating simulation', { fields: Object.keys(updateData) });

  const { text, values } = buildUpdate('simulations', updateData, 'id', id);
  const result = await query<SimulationRow>(text, values, { operation: 'update_simulation' });

  if (result.rows.length === 0) {
    throw new DatabaseError(
      'Failed to update simulation - no row returned',
      new Error('Update returned no rows'),
      { operation: 'update_simulation' }
    );
  }

  const simulation = mapRowToSimulation(result.rows[0]);
  logger.info('Simulation updated', { simulationId: simulation.id });

  return simulation;
}

/**
 * Cancel a simulation.
 *
 * @param id - Simulation UUID
 * @returns The cancelled simulation
 */
export async function cancel(id: string): Promise<Simulation> {
  const logger = repoLogger.child({ operation: 'cancel', simulationId: id });

  if (!id || id.trim().length === 0) {
    throw new ValidationError('Simulation ID is required', {
      operation: 'cancel',
    });
  }

  // Check if simulation exists
  const existing = await findById(id);
  if (!existing) {
    throw new NotFoundError('Simulation', id, { operation: 'cancel' });
  }

  // Only allow cancelling pending or running simulations
  if (!['pending', 'queued', 'running'].includes(existing.status)) {
    throw new ValidationError(`Cannot cancel simulation in ${existing.status} status`, {
      operation: 'cancel',
      simulationId: id,
      status: existing.status,
    });
  }

  logger.debug('Cancelling simulation');

  const completedAt = new Date();
  let durationMs: number | null = null;

  if (existing.startedAt) {
    durationMs = completedAt.getTime() - new Date(existing.startedAt).getTime();
  }

  const result = await query<SimulationRow>(
    `UPDATE simulations
     SET status = 'cancelled', completed_at = $1, duration_ms = $2
     WHERE id = $3
     RETURNING *`,
    [completedAt, durationMs, id],
    { operation: 'cancel_simulation' }
  );

  if (result.rows.length === 0) {
    throw new DatabaseError(
      'Failed to cancel simulation - no row returned',
      new Error('Update returned no rows'),
      { operation: 'cancel_simulation' }
    );
  }

  const simulation = mapRowToSimulation(result.rows[0]);
  logger.info('Simulation cancelled', { simulationId: simulation.id });

  return simulation;
}

/**
 * Delete a simulation.
 *
 * @param id - Simulation UUID
 */
export async function deleteSimulation(id: string): Promise<void> {
  const logger = repoLogger.child({ operation: 'delete', simulationId: id });

  if (!id || id.trim().length === 0) {
    throw new ValidationError('Simulation ID is required', {
      operation: 'delete',
    });
  }

  // Check if simulation exists
  const existing = await findById(id);
  if (!existing) {
    throw new NotFoundError('Simulation', id, { operation: 'delete' });
  }

  logger.debug('Deleting simulation');

  await query(
    'DELETE FROM simulations WHERE id = $1',
    [id],
    { operation: 'delete_simulation' }
  );

  logger.info('Simulation deleted', { simulationId: id });
}

/**
 * Get pending simulations ordered by priority for processing.
 *
 * @param limit - Maximum number of simulations to return
 * @returns Array of pending simulations
 */
export async function getPendingSimulations(limit: number = 10): Promise<Simulation[]> {
  const logger = repoLogger.child({ operation: 'getPendingSimulations' });

  logger.debug('Getting pending simulations', { limit });

  const result = await query<SimulationRow>(
    `SELECT * FROM simulations
     WHERE status IN ('pending', 'queued')
     ORDER BY priority DESC, created_at ASC
     LIMIT $1`,
    [limit],
    { operation: 'get_pending_simulations' }
  );

  return result.rows.map(mapRowToSimulation);
}

/**
 * Retry a failed simulation.
 *
 * @param id - Simulation UUID
 * @returns The reset simulation ready for retry
 */
export async function retry(id: string): Promise<Simulation> {
  const logger = repoLogger.child({ operation: 'retry', simulationId: id });

  if (!id || id.trim().length === 0) {
    throw new ValidationError('Simulation ID is required', {
      operation: 'retry',
    });
  }

  // Check if simulation exists
  const existing = await findById(id);
  if (!existing) {
    throw new NotFoundError('Simulation', id, { operation: 'retry' });
  }

  // Only allow retrying failed simulations
  if (existing.status !== 'failed') {
    throw new ValidationError('Can only retry failed simulations', {
      operation: 'retry',
      simulationId: id,
      status: existing.status,
    });
  }

  logger.debug('Retrying simulation');

  const result = await query<SimulationRow>(
    `UPDATE simulations
     SET status = 'pending',
         error_message = NULL,
         error_details = NULL,
         started_at = NULL,
         completed_at = NULL,
         duration_ms = NULL,
         worker_id = NULL
     WHERE id = $1
     RETURNING *`,
    [id],
    { operation: 'retry_simulation' }
  );

  if (result.rows.length === 0) {
    throw new DatabaseError(
      'Failed to retry simulation - no row returned',
      new Error('Update returned no rows'),
      { operation: 'retry_simulation' }
    );
  }

  const simulation = mapRowToSimulation(result.rows[0]);
  logger.info('Simulation queued for retry', { simulationId: simulation.id });

  return simulation;
}

// Default export
export default {
  create,
  findById,
  findByProject,
  updateStatus,
  complete,
  fail,
  update,
  cancel,
  delete: deleteSimulation,
  getPendingSimulations,
  retry,
};
