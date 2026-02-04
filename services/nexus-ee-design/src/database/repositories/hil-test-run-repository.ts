/**
 * HIL Test Run Repository
 *
 * Database operations for HIL test runs. Handles CRUD operations,
 * status updates, progress tracking, and result management.
 */

import {
  query,
  withTransaction,
  buildInsert,
  buildUpdate,
  DatabaseError,
} from '../connection.js';
import { NotFoundError, ValidationError } from '../../utils/errors.js';
import { log, Logger } from '../../utils/logger.js';
import type {
  HILTestRun,
  HILTestRunStatus,
  HILTestResult,
  HILTestConditions,
  HILTestRunSummary,
  HILInstrument,
  StartHILTestRunInput,
  HILTestRunFilters,
} from '../../types/hil-types.js';

// ============================================================================
// Types
// ============================================================================

interface TestRunRow {
  id: string;
  sequence_id: string;
  project_id: string;
  name: string;
  run_number: number;
  status: HILTestRunStatus;
  result: HILTestResult | null;
  progress_percentage: number;
  current_step: string | null;
  current_step_index: number | null;
  total_steps: number | null;
  queued_at: Date | null;
  started_at: Date | null;
  completed_at: Date | null;
  duration_ms: number | null;
  error_message: string | null;
  error_details: Record<string, unknown> | null;
  error_step_id: string | null;
  test_conditions: HILTestConditions | null;
  instrument_snapshot: Record<string, HILInstrument> | null;
  summary: HILTestRunSummary | null;
  worker_id: string | null;
  worker_host: string | null;
  job_id: string | null;
  retry_count: number;
  max_retries: number;
  last_retry_at: Date | null;
  baseline_run_id: string | null;
  comparison_results: Record<string, unknown> | null;
  started_by: string | null;
  aborted_by: string | null;
  abort_reason: string | null;
  tags: string[] | null;
  metadata: Record<string, unknown>;
  created_at: Date;
  updated_at: Date;
}

// ============================================================================
// Repository Implementation
// ============================================================================

const repoLogger: Logger = log.child({ service: 'hil-test-run-repository' });

/**
 * Validate test run status
 */
function validateStatus(status: string): asserts status is HILTestRunStatus {
  const validStatuses: HILTestRunStatus[] = [
    'pending',
    'queued',
    'running',
    'completed',
    'failed',
    'aborted',
    'timeout',
  ];

  if (!validStatuses.includes(status as HILTestRunStatus)) {
    throw new ValidationError(`Invalid test run status: ${status}`, {
      operation: 'validate_status',
      validStatuses,
    });
  }
}

/**
 * Validate test result
 */
function validateResult(result: string): asserts result is HILTestResult {
  const validResults: HILTestResult[] = ['pass', 'fail', 'partial', 'inconclusive'];

  if (!validResults.includes(result as HILTestResult)) {
    throw new ValidationError(`Invalid test result: ${result}`, {
      operation: 'validate_result',
      validResults,
    });
  }
}

/**
 * Map a database row to an HILTestRun object.
 */
function mapRowToRun(row: TestRunRow): HILTestRun {
  return {
    id: row.id,
    sequenceId: row.sequence_id,
    projectId: row.project_id,
    name: row.name,
    runNumber: row.run_number,
    status: row.status,
    result: row.result || undefined,
    progressPercentage: row.progress_percentage,
    currentStep: row.current_step || undefined,
    currentStepIndex: row.current_step_index ?? undefined,
    totalSteps: row.total_steps ?? undefined,
    queuedAt: row.queued_at?.toISOString(),
    startedAt: row.started_at?.toISOString(),
    completedAt: row.completed_at?.toISOString(),
    durationMs: row.duration_ms ?? undefined,
    errorMessage: row.error_message || undefined,
    errorDetails: row.error_details || undefined,
    errorStepId: row.error_step_id || undefined,
    testConditions: row.test_conditions || undefined,
    instrumentSnapshot: row.instrument_snapshot || undefined,
    summary: row.summary || undefined,
    workerId: row.worker_id || undefined,
    workerHost: row.worker_host || undefined,
    jobId: row.job_id || undefined,
    retryCount: row.retry_count,
    maxRetries: row.max_retries,
    lastRetryAt: row.last_retry_at?.toISOString(),
    baselineRunId: row.baseline_run_id || undefined,
    comparisonResults: row.comparison_results || undefined,
    startedBy: row.started_by || undefined,
    abortedBy: row.aborted_by || undefined,
    abortReason: row.abort_reason || undefined,
    tags: row.tags || undefined,
    metadata: row.metadata || {},
    createdAt: row.created_at.toISOString(),
    updatedAt: row.updated_at.toISOString(),
  };
}

/**
 * Create a new test run.
 *
 * @param input - Test run creation data
 * @param sequenceName - Name of the sequence for run naming
 * @returns The created test run
 */
export async function create(
  input: StartHILTestRunInput,
  sequenceName: string,
  userId?: string
): Promise<HILTestRun> {
  const logger = repoLogger.child({ operation: 'create' });

  if (!input.sequenceId || input.sequenceId.trim().length === 0) {
    throw new ValidationError('Sequence ID is required', {
      operation: 'create',
      input,
    });
  }

  // Get the project ID from the sequence
  const sequenceResult = await query<{ project_id: string }>(
    'SELECT project_id FROM hil_test_sequences WHERE id = $1',
    [input.sequenceId],
    { operation: 'get_sequence_project' }
  );

  if (sequenceResult.rows.length === 0) {
    throw new NotFoundError('HILTestSequence', input.sequenceId, {
      operation: 'create',
    });
  }

  const projectId = sequenceResult.rows[0].project_id;

  // Get next run number for this sequence (auto-handled by trigger but we read it)
  const data: Record<string, unknown> = {
    sequence_id: input.sequenceId,
    project_id: projectId,
    name: sequenceName,
    status: 'pending',
    test_conditions: input.testConditions
      ? JSON.stringify(input.testConditions)
      : null,
    baseline_run_id: input.baselineRunId || null,
    tags: input.tags || [],
    started_by: userId || null,
    metadata: JSON.stringify({
      ...input.parameterOverrides,
    }),
  };

  logger.debug('Creating HIL test run', {
    sequenceId: input.sequenceId,
    projectId,
    hasConditions: !!input.testConditions,
  });

  const { text, values } = buildInsert('hil_test_runs', data);
  const result = await query<TestRunRow>(text, values, {
    operation: 'create_hil_test_run',
  });

  if (result.rows.length === 0) {
    throw new DatabaseError(
      'Failed to create HIL test run - no row returned',
      new Error('Insert returned no rows'),
      { operation: 'create_hil_test_run' }
    );
  }

  const run = mapRowToRun(result.rows[0]);
  logger.info('HIL test run created', {
    runId: run.id,
    runNumber: run.runNumber,
    sequenceId: run.sequenceId,
  });

  return run;
}

/**
 * Find a test run by its ID.
 *
 * @param id - Run UUID
 * @returns The run or null if not found
 */
export async function findById(id: string): Promise<HILTestRun | null> {
  const logger = repoLogger.child({ operation: 'findById', runId: id });

  if (!id || id.trim().length === 0) {
    throw new ValidationError('Run ID is required', {
      operation: 'findById',
    });
  }

  logger.debug('Finding test run by ID');

  const result = await query<TestRunRow>(
    'SELECT * FROM hil_test_runs WHERE id = $1',
    [id],
    { operation: 'find_test_run_by_id' }
  );

  if (result.rows.length === 0) {
    logger.debug('Test run not found');
    return null;
  }

  return mapRowToRun(result.rows[0]);
}

/**
 * Find all test runs for a project with optional filters.
 *
 * @param projectId - Project UUID
 * @param filters - Optional filters
 * @param limit - Max results (default 50)
 * @param offset - Result offset (default 0)
 * @returns Array of test runs
 */
export async function findByProject(
  projectId: string,
  filters?: HILTestRunFilters,
  limit: number = 50,
  offset: number = 0
): Promise<HILTestRun[]> {
  const logger = repoLogger.child({ operation: 'findByProject', projectId });

  if (!projectId || projectId.trim().length === 0) {
    throw new ValidationError('Project ID is required', {
      operation: 'findByProject',
    });
  }

  const conditions: string[] = ['project_id = $1'];
  const values: unknown[] = [projectId];
  let paramIndex = 2;

  if (filters?.status) {
    validateStatus(filters.status);
    conditions.push(`status = $${paramIndex++}`);
    values.push(filters.status);
  }

  if (filters?.result) {
    validateResult(filters.result);
    conditions.push(`result = $${paramIndex++}`);
    values.push(filters.result);
  }

  if (filters?.sequenceId) {
    conditions.push(`sequence_id = $${paramIndex++}`);
    values.push(filters.sequenceId);
  }

  if (filters?.startedAfter) {
    conditions.push(`started_at >= $${paramIndex++}`);
    values.push(filters.startedAfter);
  }

  if (filters?.startedBefore) {
    conditions.push(`started_at <= $${paramIndex++}`);
    values.push(filters.startedBefore);
  }

  values.push(limit, offset);

  const sql = `SELECT * FROM hil_test_runs
               WHERE ${conditions.join(' AND ')}
               ORDER BY created_at DESC
               LIMIT $${paramIndex++} OFFSET $${paramIndex++}`;

  logger.debug('Finding test runs by project', { filters, limit, offset });

  const result = await query<TestRunRow>(sql, values, {
    operation: 'find_test_runs_by_project',
  });

  logger.debug('Found test runs', { count: result.rows.length });

  return result.rows.map(mapRowToRun);
}

/**
 * Find test runs for a sequence.
 *
 * @param sequenceId - Sequence UUID
 * @param limit - Max results (default 20)
 * @returns Array of test runs
 */
export async function findBySequence(
  sequenceId: string,
  limit: number = 20
): Promise<HILTestRun[]> {
  const logger = repoLogger.child({ operation: 'findBySequence', sequenceId });

  if (!sequenceId || sequenceId.trim().length === 0) {
    throw new ValidationError('Sequence ID is required', {
      operation: 'findBySequence',
    });
  }

  logger.debug('Finding test runs by sequence');

  const result = await query<TestRunRow>(
    `SELECT * FROM hil_test_runs
     WHERE sequence_id = $1
     ORDER BY run_number DESC
     LIMIT $2`,
    [sequenceId, limit],
    { operation: 'find_test_runs_by_sequence' }
  );

  return result.rows.map(mapRowToRun);
}

/**
 * Update test run status.
 *
 * @param id - Run UUID
 * @param status - New status
 * @returns The updated run
 */
export async function updateStatus(
  id: string,
  status: HILTestRunStatus
): Promise<HILTestRun> {
  const logger = repoLogger.child({ operation: 'updateStatus', runId: id });

  if (!id || id.trim().length === 0) {
    throw new ValidationError('Run ID is required', {
      operation: 'updateStatus',
    });
  }

  validateStatus(status);

  // Check if run exists
  const existing = await findById(id);
  if (!existing) {
    throw new NotFoundError('HILTestRun', id, { operation: 'updateStatus' });
  }

  const updateData: Record<string, unknown> = { status };

  // Set timestamp based on status
  if (status === 'queued' && !existing.queuedAt) {
    updateData.queued_at = new Date();
  } else if (status === 'running' && !existing.startedAt) {
    updateData.started_at = new Date();
  } else if (['completed', 'failed', 'aborted', 'timeout'].includes(status)) {
    updateData.completed_at = new Date();
    if (existing.startedAt) {
      updateData.duration_ms =
        Date.now() - new Date(existing.startedAt).getTime();
    }
  }

  logger.debug('Updating run status', {
    currentStatus: existing.status,
    newStatus: status,
  });

  const setClauses = Object.keys(updateData).map((key, i) => `${key} = $${i + 1}`);
  const values = [...Object.values(updateData), id];

  const result = await query<TestRunRow>(
    `UPDATE hil_test_runs SET ${setClauses.join(', ')}, updated_at = NOW()
     WHERE id = $${values.length} RETURNING *`,
    values,
    { operation: 'update_run_status' }
  );

  if (result.rows.length === 0) {
    throw new DatabaseError(
      'Failed to update run status - no row returned',
      new Error('Update returned no rows'),
      { operation: 'update_run_status' }
    );
  }

  const run = mapRowToRun(result.rows[0]);
  logger.info('Run status updated', {
    runId: run.id,
    previousStatus: existing.status,
    newStatus: status,
  });

  return run;
}

/**
 * Update test run progress.
 *
 * @param id - Run UUID
 * @param progress - Progress update data
 * @returns The updated run
 */
export async function updateProgress(
  id: string,
  progress: {
    progressPercentage: number;
    currentStep?: string;
    currentStepIndex?: number;
    totalSteps?: number;
  }
): Promise<HILTestRun> {
  const logger = repoLogger.child({ operation: 'updateProgress', runId: id });

  if (!id || id.trim().length === 0) {
    throw new ValidationError('Run ID is required', {
      operation: 'updateProgress',
    });
  }

  if (
    progress.progressPercentage < 0 ||
    progress.progressPercentage > 100
  ) {
    throw new ValidationError('Progress percentage must be between 0 and 100', {
      operation: 'updateProgress',
      value: progress.progressPercentage,
    });
  }

  const updateData: Record<string, unknown> = {
    progress_percentage: progress.progressPercentage,
  };

  if (progress.currentStep !== undefined) {
    updateData.current_step = progress.currentStep;
  }
  if (progress.currentStepIndex !== undefined) {
    updateData.current_step_index = progress.currentStepIndex;
  }
  if (progress.totalSteps !== undefined) {
    updateData.total_steps = progress.totalSteps;
  }

  const setClauses = Object.keys(updateData).map((key, i) => `${key} = $${i + 1}`);
  const values = [...Object.values(updateData), id];

  const result = await query<TestRunRow>(
    `UPDATE hil_test_runs SET ${setClauses.join(', ')}, updated_at = NOW()
     WHERE id = $${values.length} RETURNING *`,
    values,
    { operation: 'update_run_progress' }
  );

  if (result.rows.length === 0) {
    throw new DatabaseError(
      'Failed to update run progress - no row returned',
      new Error('Update returned no rows'),
      { operation: 'update_run_progress' }
    );
  }

  logger.debug('Run progress updated', {
    runId: id,
    progress: progress.progressPercentage,
    step: progress.currentStep,
  });

  return mapRowToRun(result.rows[0]);
}

/**
 * Complete a test run with results.
 *
 * @param id - Run UUID
 * @param result - Test result
 * @param summary - Test summary
 * @returns The updated run
 */
export async function complete(
  id: string,
  result: HILTestResult,
  summary: HILTestRunSummary
): Promise<HILTestRun> {
  const logger = repoLogger.child({ operation: 'complete', runId: id });

  if (!id || id.trim().length === 0) {
    throw new ValidationError('Run ID is required', {
      operation: 'complete',
    });
  }

  validateResult(result);

  // Get existing run
  const existing = await findById(id);
  if (!existing) {
    throw new NotFoundError('HILTestRun', id, { operation: 'complete' });
  }

  const durationMs = existing.startedAt
    ? Date.now() - new Date(existing.startedAt).getTime()
    : null;

  logger.debug('Completing test run', {
    result,
    passedCount: summary.passedMeasurements,
    totalCount: summary.totalMeasurements,
  });

  const updateResult = await query<TestRunRow>(
    `UPDATE hil_test_runs SET
       status = 'completed',
       result = $1,
       summary = $2,
       progress_percentage = 100,
       completed_at = NOW(),
       duration_ms = $3,
       updated_at = NOW()
     WHERE id = $4 RETURNING *`,
    [result, JSON.stringify(summary), durationMs, id],
    { operation: 'complete_test_run' }
  );

  if (updateResult.rows.length === 0) {
    throw new DatabaseError(
      'Failed to complete test run - no row returned',
      new Error('Update returned no rows'),
      { operation: 'complete_test_run' }
    );
  }

  const run = mapRowToRun(updateResult.rows[0]);
  logger.info('Test run completed', {
    runId: run.id,
    result: run.result,
    durationMs: run.durationMs,
  });

  return run;
}

/**
 * Fail a test run with error details.
 *
 * @param id - Run UUID
 * @param errorMessage - Error message
 * @param errorDetails - Additional error details
 * @param errorStepId - ID of the step that caused the error
 * @returns The updated run
 */
export async function fail(
  id: string,
  errorMessage: string,
  errorDetails?: Record<string, unknown>,
  errorStepId?: string
): Promise<HILTestRun> {
  const logger = repoLogger.child({ operation: 'fail', runId: id });

  if (!id || id.trim().length === 0) {
    throw new ValidationError('Run ID is required', {
      operation: 'fail',
    });
  }

  if (!errorMessage || errorMessage.trim().length === 0) {
    throw new ValidationError('Error message is required', {
      operation: 'fail',
    });
  }

  // Get existing run
  const existing = await findById(id);
  if (!existing) {
    throw new NotFoundError('HILTestRun', id, { operation: 'fail' });
  }

  const durationMs = existing.startedAt
    ? Date.now() - new Date(existing.startedAt).getTime()
    : null;

  logger.debug('Failing test run', {
    errorMessage,
    errorStepId,
  });

  const result = await query<TestRunRow>(
    `UPDATE hil_test_runs SET
       status = 'failed',
       result = 'fail',
       error_message = $1,
       error_details = $2,
       error_step_id = $3,
       completed_at = NOW(),
       duration_ms = $4,
       updated_at = NOW()
     WHERE id = $5 RETURNING *`,
    [
      errorMessage,
      errorDetails ? JSON.stringify(errorDetails) : null,
      errorStepId || null,
      durationMs,
      id,
    ],
    { operation: 'fail_test_run' }
  );

  if (result.rows.length === 0) {
    throw new DatabaseError(
      'Failed to fail test run - no row returned',
      new Error('Update returned no rows'),
      { operation: 'fail_test_run' }
    );
  }

  const run = mapRowToRun(result.rows[0]);
  logger.info('Test run failed', {
    runId: run.id,
    errorMessage,
    durationMs: run.durationMs,
  });

  return run;
}

/**
 * Abort a test run.
 *
 * @param id - Run UUID
 * @param userId - User who aborted
 * @param reason - Abort reason
 * @returns The updated run
 */
export async function abort(
  id: string,
  userId?: string,
  reason?: string
): Promise<HILTestRun> {
  const logger = repoLogger.child({ operation: 'abort', runId: id });

  if (!id || id.trim().length === 0) {
    throw new ValidationError('Run ID is required', {
      operation: 'abort',
    });
  }

  // Get existing run
  const existing = await findById(id);
  if (!existing) {
    throw new NotFoundError('HILTestRun', id, { operation: 'abort' });
  }

  if (['completed', 'failed', 'aborted', 'timeout'].includes(existing.status)) {
    throw new ValidationError('Cannot abort a run that has already finished', {
      operation: 'abort',
      runId: id,
      currentStatus: existing.status,
    });
  }

  const durationMs = existing.startedAt
    ? Date.now() - new Date(existing.startedAt).getTime()
    : null;

  logger.debug('Aborting test run', { userId, reason });

  const result = await query<TestRunRow>(
    `UPDATE hil_test_runs SET
       status = 'aborted',
       aborted_by = $1,
       abort_reason = $2,
       completed_at = NOW(),
       duration_ms = $3,
       updated_at = NOW()
     WHERE id = $4 RETURNING *`,
    [userId || null, reason || null, durationMs, id],
    { operation: 'abort_test_run' }
  );

  if (result.rows.length === 0) {
    throw new DatabaseError(
      'Failed to abort test run - no row returned',
      new Error('Update returned no rows'),
      { operation: 'abort_test_run' }
    );
  }

  const run = mapRowToRun(result.rows[0]);
  logger.info('Test run aborted', {
    runId: run.id,
    abortedBy: userId,
    reason,
  });

  return run;
}

/**
 * Update worker assignment for a run.
 *
 * @param id - Run UUID
 * @param workerId - Worker ID
 * @param workerHost - Worker host
 * @param jobId - BullMQ job ID
 * @returns The updated run
 */
export async function assignWorker(
  id: string,
  workerId: string,
  workerHost: string,
  jobId: string
): Promise<HILTestRun> {
  const logger = repoLogger.child({ operation: 'assignWorker', runId: id });

  if (!id || id.trim().length === 0) {
    throw new ValidationError('Run ID is required', {
      operation: 'assignWorker',
    });
  }

  logger.debug('Assigning worker', { workerId, workerHost, jobId });

  const result = await query<TestRunRow>(
    `UPDATE hil_test_runs SET
       worker_id = $1,
       worker_host = $2,
       job_id = $3,
       updated_at = NOW()
     WHERE id = $4 RETURNING *`,
    [workerId, workerHost, jobId, id],
    { operation: 'assign_worker' }
  );

  if (result.rows.length === 0) {
    throw new DatabaseError(
      'Failed to assign worker - no row returned',
      new Error('Update returned no rows'),
      { operation: 'assign_worker' }
    );
  }

  return mapRowToRun(result.rows[0]);
}

/**
 * Store instrument snapshot for a run.
 *
 * @param id - Run UUID
 * @param instruments - Instrument snapshot
 * @returns The updated run
 */
export async function storeInstrumentSnapshot(
  id: string,
  instruments: Record<string, HILInstrument>
): Promise<HILTestRun> {
  const logger = repoLogger.child({
    operation: 'storeInstrumentSnapshot',
    runId: id,
  });

  if (!id || id.trim().length === 0) {
    throw new ValidationError('Run ID is required', {
      operation: 'storeInstrumentSnapshot',
    });
  }

  logger.debug('Storing instrument snapshot', {
    instrumentCount: Object.keys(instruments).length,
  });

  const result = await query<TestRunRow>(
    `UPDATE hil_test_runs SET
       instrument_snapshot = $1,
       updated_at = NOW()
     WHERE id = $2 RETURNING *`,
    [JSON.stringify(instruments), id],
    { operation: 'store_instrument_snapshot' }
  );

  if (result.rows.length === 0) {
    throw new DatabaseError(
      'Failed to store instrument snapshot - no row returned',
      new Error('Update returned no rows'),
      { operation: 'store_instrument_snapshot' }
    );
  }

  return mapRowToRun(result.rows[0]);
}

/**
 * Increment retry count for a run.
 *
 * @param id - Run UUID
 * @returns The updated run
 */
export async function incrementRetry(id: string): Promise<HILTestRun> {
  const logger = repoLogger.child({ operation: 'incrementRetry', runId: id });

  if (!id || id.trim().length === 0) {
    throw new ValidationError('Run ID is required', {
      operation: 'incrementRetry',
    });
  }

  const result = await query<TestRunRow>(
    `UPDATE hil_test_runs SET
       retry_count = retry_count + 1,
       last_retry_at = NOW(),
       status = 'pending',
       error_message = NULL,
       error_details = NULL,
       error_step_id = NULL,
       updated_at = NOW()
     WHERE id = $1 RETURNING *`,
    [id],
    { operation: 'increment_retry' }
  );

  if (result.rows.length === 0) {
    throw new DatabaseError(
      'Failed to increment retry - no row returned',
      new Error('Update returned no rows'),
      { operation: 'increment_retry' }
    );
  }

  const run = mapRowToRun(result.rows[0]);
  logger.info('Retry count incremented', {
    runId: run.id,
    retryCount: run.retryCount,
  });

  return run;
}

/**
 * Find active runs (pending, queued, or running).
 *
 * @param projectId - Optional project filter
 * @returns Array of active runs
 */
export async function findActive(projectId?: string): Promise<HILTestRun[]> {
  const logger = repoLogger.child({ operation: 'findActive', projectId });

  let sql = `SELECT * FROM hil_test_runs
             WHERE status IN ('pending', 'queued', 'running')`;
  const values: unknown[] = [];

  if (projectId) {
    sql += ' AND project_id = $1';
    values.push(projectId);
  }

  sql += ' ORDER BY created_at ASC';

  logger.debug('Finding active runs');

  const result = await query<TestRunRow>(sql, values, {
    operation: 'find_active_runs',
  });

  return result.rows.map(mapRowToRun);
}

/**
 * Get run statistics for a project.
 *
 * @param projectId - Project UUID
 * @returns Statistics object
 */
export async function getProjectStats(projectId: string): Promise<{
  total: number;
  passed: number;
  failed: number;
  partial: number;
  inconclusive: number;
  active: number;
  avgDurationMs: number | null;
  successRate: number | null;
}> {
  const logger = repoLogger.child({ operation: 'getProjectStats', projectId });

  if (!projectId || projectId.trim().length === 0) {
    throw new ValidationError('Project ID is required', {
      operation: 'getProjectStats',
    });
  }

  logger.debug('Getting project run stats');

  const result = await query<{
    total: string;
    passed: string;
    failed: string;
    partial: string;
    inconclusive: string;
    active: string;
    avg_duration: string | null;
  }>(
    `SELECT
       COUNT(*) as total,
       COUNT(*) FILTER (WHERE result = 'pass') as passed,
       COUNT(*) FILTER (WHERE result = 'fail') as failed,
       COUNT(*) FILTER (WHERE result = 'partial') as partial,
       COUNT(*) FILTER (WHERE result = 'inconclusive') as inconclusive,
       COUNT(*) FILTER (WHERE status IN ('pending', 'queued', 'running')) as active,
       AVG(duration_ms) FILTER (WHERE status = 'completed') as avg_duration
     FROM hil_test_runs
     WHERE project_id = $1`,
    [projectId],
    { operation: 'get_run_stats' }
  );

  const stats = result.rows[0];
  const total = parseInt(stats.total, 10);
  const passed = parseInt(stats.passed, 10);
  const completed = passed + parseInt(stats.failed, 10);

  return {
    total,
    passed,
    failed: parseInt(stats.failed, 10),
    partial: parseInt(stats.partial, 10),
    inconclusive: parseInt(stats.inconclusive, 10),
    active: parseInt(stats.active, 10),
    avgDurationMs: stats.avg_duration ? parseFloat(stats.avg_duration) : null,
    successRate: completed > 0 ? (passed / completed) * 100 : null,
  };
}

/**
 * Get comparison with baseline run.
 *
 * @param id - Run UUID
 * @param baselineId - Baseline run UUID
 * @returns Comparison results
 */
export async function compareToBaseline(
  id: string,
  baselineId: string
): Promise<{
  improvement: Record<string, number>;
  regression: Record<string, number>;
  unchanged: string[];
}> {
  const logger = repoLogger.child({
    operation: 'compareToBaseline',
    runId: id,
    baselineId,
  });

  const [current, baseline] = await Promise.all([findById(id), findById(baselineId)]);

  if (!current) {
    throw new NotFoundError('HILTestRun (current)', id, {
      operation: 'compareToBaseline',
    });
  }

  if (!baseline) {
    throw new NotFoundError('HILTestRun (baseline)', baselineId, {
      operation: 'compareToBaseline',
    });
  }

  if (!current.summary || !baseline.summary) {
    throw new ValidationError('Both runs must have summary data', {
      operation: 'compareToBaseline',
      currentHasSummary: !!current.summary,
      baselineHasSummary: !!baseline.summary,
    });
  }

  logger.debug('Comparing runs');

  const improvement: Record<string, number> = {};
  const regression: Record<string, number> = {};
  const unchanged: string[] = [];

  // Compare key metrics
  const currentMetrics = current.summary.keyMetrics || {};
  const baselineMetrics = baseline.summary.keyMetrics || {};

  const allMetrics = new Set([
    ...Object.keys(currentMetrics),
    ...Object.keys(baselineMetrics),
  ]);

  for (const metric of allMetrics) {
    const currentValue = currentMetrics[metric];
    const baselineValue = baselineMetrics[metric];

    if (currentValue === undefined || baselineValue === undefined) {
      continue;
    }

    const delta = currentValue - baselineValue;
    const percentChange =
      baselineValue !== 0 ? (delta / baselineValue) * 100 : 0;

    if (Math.abs(percentChange) < 1) {
      unchanged.push(metric);
    } else if (percentChange > 0) {
      improvement[metric] = percentChange;
    } else {
      regression[metric] = percentChange;
    }
  }

  // Store comparison results
  await query(
    `UPDATE hil_test_runs SET
       comparison_results = $1,
       updated_at = NOW()
     WHERE id = $2`,
    [JSON.stringify({ improvement, regression, unchanged }), id],
    { operation: 'store_comparison' }
  );

  return { improvement, regression, unchanged };
}

/**
 * Delete a test run and all associated data.
 *
 * @param id - Run UUID
 */
export async function deleteRun(id: string): Promise<void> {
  const logger = repoLogger.child({ operation: 'delete', runId: id });

  if (!id || id.trim().length === 0) {
    throw new ValidationError('Run ID is required', {
      operation: 'delete',
    });
  }

  const existing = await findById(id);
  if (!existing) {
    throw new NotFoundError('HILTestRun', id, { operation: 'delete' });
  }

  if (['running', 'pending', 'queued'].includes(existing.status)) {
    throw new ValidationError('Cannot delete an active run', {
      operation: 'delete',
      runId: id,
      status: existing.status,
    });
  }

  logger.debug('Deleting test run');

  // Delete in transaction (cascade will handle related records)
  await withTransaction(async (client) => {
    await client.query('DELETE FROM hil_measurements WHERE test_run_id = $1', [id]);
    await client.query('DELETE FROM hil_captured_data WHERE test_run_id = $1', [id]);
    await client.query('DELETE FROM hil_test_runs WHERE id = $1', [id]);
  });

  logger.info('Test run deleted', { runId: id, name: existing.name });
}

// Default export
export default {
  create,
  findById,
  findByProject,
  findBySequence,
  updateStatus,
  updateProgress,
  complete,
  fail,
  abort,
  assignWorker,
  storeInstrumentSnapshot,
  incrementRetry,
  findActive,
  getProjectStats,
  compareToBaseline,
  delete: deleteRun,
};
