/**
 * HIL Measurement Repository
 *
 * Database operations for HIL measurements. Handles CRUD operations,
 * pass/fail evaluation, and measurement aggregation.
 */

import {
  query,
  buildInsert,
  buildUpdate,
  DatabaseError,
} from '../connection.js';
import { NotFoundError, ValidationError } from '../../utils/errors.js';
import { log, Logger } from '../../utils/logger.js';
import type {
  HILMeasurement,
  HILMeasurementType,
  CreateHILMeasurementInput,
} from '../../types/hil-types.js';

// ============================================================================
// Types
// ============================================================================

interface MeasurementRow {
  id: string;
  test_run_id: string;
  captured_data_id: string | null;
  step_id: string | null;
  measurement_type: HILMeasurementType;
  measurement_name: string | null;
  channel: string | null;
  value: number;
  unit: string;
  min_limit: number | null;
  max_limit: number | null;
  nominal_value: number | null;
  tolerance_percent: number | null;
  tolerance_absolute: number | null;
  passed: boolean | null;
  is_critical: boolean;
  is_warning: boolean;
  failure_reason: string | null;
  timestamp_offset_ms: number | null;
  measured_at: Date;
  sample_count: number | null;
  mean_value: number | null;
  std_deviation: number | null;
  min_observed: number | null;
  max_observed: number | null;
  metadata: Record<string, unknown>;
  created_at: Date;
}

interface UpdateMeasurementInput {
  value?: number;
  passed?: boolean;
  failureReason?: string;
  isWarning?: boolean;
  sampleCount?: number;
  meanValue?: number;
  stdDeviation?: number;
  minObserved?: number;
  maxObserved?: number;
  metadata?: Record<string, unknown>;
}

interface MeasurementFilters {
  measurementType?: HILMeasurementType;
  passed?: boolean;
  isCritical?: boolean;
  isWarning?: boolean;
  stepId?: string;
  capturedDataId?: string;
  channel?: string;
}

// ============================================================================
// Repository Implementation
// ============================================================================

const repoLogger: Logger = log.child({ service: 'hil-measurement-repository' });

/**
 * Validate measurement type
 */
function validateMeasurementType(type: string): asserts type is HILMeasurementType {
  const validTypes: HILMeasurementType[] = [
    'rms_current',
    'peak_current',
    'avg_current',
    'rms_voltage',
    'peak_voltage',
    'avg_voltage',
    'frequency',
    'duty_cycle',
    'dead_time',
    'rise_time',
    'fall_time',
    'temperature',
    'efficiency',
    'thd',
    'phase_angle',
    'power',
    'power_factor',
    'speed',
    'torque',
    'position',
    'startup_time',
    'settling_time',
    'overshoot',
    'ripple',
    'custom',
  ];

  if (!validTypes.includes(type as HILMeasurementType)) {
    throw new ValidationError(`Invalid measurement type: ${type}`, {
      operation: 'validate_measurement_type',
      validTypes,
    });
  }
}

/**
 * Map a database row to an HILMeasurement object.
 */
function mapRowToMeasurement(row: MeasurementRow): HILMeasurement {
  return {
    id: row.id,
    testRunId: row.test_run_id,
    capturedDataId: row.captured_data_id || undefined,
    stepId: row.step_id || undefined,
    measurementType: row.measurement_type,
    measurementName: row.measurement_name || undefined,
    channel: row.channel || undefined,
    value: row.value,
    unit: row.unit,
    minLimit: row.min_limit ?? undefined,
    maxLimit: row.max_limit ?? undefined,
    nominalValue: row.nominal_value ?? undefined,
    tolerancePercent: row.tolerance_percent ?? undefined,
    toleranceAbsolute: row.tolerance_absolute ?? undefined,
    passed: row.passed ?? undefined,
    isCritical: row.is_critical,
    isWarning: row.is_warning,
    failureReason: row.failure_reason || undefined,
    timestampOffsetMs: row.timestamp_offset_ms ?? undefined,
    measuredAt: row.measured_at.toISOString(),
    sampleCount: row.sample_count ?? undefined,
    meanValue: row.mean_value ?? undefined,
    stdDeviation: row.std_deviation ?? undefined,
    minObserved: row.min_observed ?? undefined,
    maxObserved: row.max_observed ?? undefined,
    metadata: row.metadata || {},
    createdAt: row.created_at.toISOString(),
  };
}

/**
 * Evaluate pass/fail status for a measurement.
 */
function evaluatePassFail(
  value: number,
  minLimit?: number,
  maxLimit?: number,
  nominalValue?: number,
  tolerancePercent?: number,
  toleranceAbsolute?: number
): { passed: boolean; failureReason?: string } {
  // If nominal value with tolerance
  if (nominalValue !== undefined) {
    let allowedDeviation = 0;

    if (tolerancePercent !== undefined) {
      allowedDeviation = Math.abs(nominalValue * tolerancePercent / 100);
    }

    if (toleranceAbsolute !== undefined) {
      allowedDeviation = Math.max(allowedDeviation, toleranceAbsolute);
    }

    if (allowedDeviation > 0) {
      const deviation = Math.abs(value - nominalValue);
      if (deviation > allowedDeviation) {
        return {
          passed: false,
          failureReason: `Value ${value} deviates ${deviation.toFixed(4)} from nominal ${nominalValue} (allowed: ${allowedDeviation.toFixed(4)})`,
        };
      }
      return { passed: true };
    }
  }

  // If min/max limits
  if (minLimit !== undefined && value < minLimit) {
    return {
      passed: false,
      failureReason: `Value ${value} is below minimum limit ${minLimit}`,
    };
  }

  if (maxLimit !== undefined && value > maxLimit) {
    return {
      passed: false,
      failureReason: `Value ${value} is above maximum limit ${maxLimit}`,
    };
  }

  // If limits were provided and we got here, it passed
  if (minLimit !== undefined || maxLimit !== undefined || nominalValue !== undefined) {
    return { passed: true };
  }

  // No limits provided - cannot determine pass/fail
  return { passed: true };
}

/**
 * Create a new measurement.
 *
 * @param input - Measurement creation data
 * @returns The created measurement
 */
export async function create(input: CreateHILMeasurementInput): Promise<HILMeasurement> {
  const logger = repoLogger.child({ operation: 'create' });

  // Validate required fields
  if (!input.testRunId || input.testRunId.trim().length === 0) {
    throw new ValidationError('Test run ID is required', {
      operation: 'create',
      input,
    });
  }

  if (!input.measurementType) {
    throw new ValidationError('Measurement type is required', {
      operation: 'create',
      input,
    });
  }

  if (typeof input.value !== 'number' || !isFinite(input.value)) {
    throw new ValidationError('Value must be a finite number', {
      operation: 'create',
      value: input.value,
    });
  }

  if (!input.unit || input.unit.trim().length === 0) {
    throw new ValidationError('Unit is required', {
      operation: 'create',
      input,
    });
  }

  // Validate measurement type
  validateMeasurementType(input.measurementType);

  // Evaluate pass/fail
  const { passed, failureReason } = evaluatePassFail(
    input.value,
    input.minLimit,
    input.maxLimit,
    input.nominalValue,
    input.tolerancePercent
  );

  const data: Record<string, unknown> = {
    test_run_id: input.testRunId,
    captured_data_id: input.capturedDataId || null,
    step_id: input.stepId || null,
    measurement_type: input.measurementType,
    measurement_name: input.measurementName || null,
    channel: input.channel || null,
    value: input.value,
    unit: input.unit.trim(),
    min_limit: input.minLimit ?? null,
    max_limit: input.maxLimit ?? null,
    nominal_value: input.nominalValue ?? null,
    tolerance_percent: input.tolerancePercent ?? null,
    is_critical: input.isCritical ?? false,
    passed,
    failure_reason: failureReason || null,
    timestamp_offset_ms: input.timestampOffsetMs ?? null,
    measured_at: new Date(),
    metadata: JSON.stringify(input.metadata || {}),
  };

  logger.debug('Creating measurement', {
    testRunId: input.testRunId,
    measurementType: input.measurementType,
    value: input.value,
    unit: input.unit,
    passed,
  });

  const { text, values } = buildInsert('hil_measurements', data);
  const result = await query<MeasurementRow>(text, values, {
    operation: 'create_measurement',
  });

  if (result.rows.length === 0) {
    throw new DatabaseError(
      'Failed to create measurement - no row returned',
      new Error('Insert returned no rows'),
      { operation: 'create_measurement' }
    );
  }

  const measurement = mapRowToMeasurement(result.rows[0]);
  logger.info('Measurement created', {
    measurementId: measurement.id,
    type: measurement.measurementType,
    passed: measurement.passed,
    isCritical: measurement.isCritical,
  });

  return measurement;
}

/**
 * Create multiple measurements in bulk.
 *
 * @param measurements - Array of measurement creation data
 * @returns Array of created measurements
 */
export async function createBulk(
  measurements: CreateHILMeasurementInput[]
): Promise<HILMeasurement[]> {
  const logger = repoLogger.child({ operation: 'createBulk' });

  if (!measurements || measurements.length === 0) {
    return [];
  }

  logger.debug('Creating measurements in bulk', { count: measurements.length });

  // Validate all measurements first
  for (const input of measurements) {
    if (!input.testRunId || input.testRunId.trim().length === 0) {
      throw new ValidationError('Test run ID is required for all measurements', {
        operation: 'createBulk',
      });
    }
    if (!input.measurementType) {
      throw new ValidationError('Measurement type is required for all measurements', {
        operation: 'createBulk',
      });
    }
    validateMeasurementType(input.measurementType);
  }

  // Build bulk insert
  const rows: string[] = [];
  const values: unknown[] = [];
  let paramIndex = 1;

  for (const input of measurements) {
    const { passed, failureReason } = evaluatePassFail(
      input.value,
      input.minLimit,
      input.maxLimit,
      input.nominalValue,
      input.tolerancePercent
    );

    const rowParams: string[] = [];

    // test_run_id
    rowParams.push(`$${paramIndex++}`);
    values.push(input.testRunId);

    // captured_data_id
    rowParams.push(`$${paramIndex++}`);
    values.push(input.capturedDataId || null);

    // step_id
    rowParams.push(`$${paramIndex++}`);
    values.push(input.stepId || null);

    // measurement_type
    rowParams.push(`$${paramIndex++}`);
    values.push(input.measurementType);

    // measurement_name
    rowParams.push(`$${paramIndex++}`);
    values.push(input.measurementName || null);

    // channel
    rowParams.push(`$${paramIndex++}`);
    values.push(input.channel || null);

    // value
    rowParams.push(`$${paramIndex++}`);
    values.push(input.value);

    // unit
    rowParams.push(`$${paramIndex++}`);
    values.push(input.unit);

    // min_limit
    rowParams.push(`$${paramIndex++}`);
    values.push(input.minLimit ?? null);

    // max_limit
    rowParams.push(`$${paramIndex++}`);
    values.push(input.maxLimit ?? null);

    // nominal_value
    rowParams.push(`$${paramIndex++}`);
    values.push(input.nominalValue ?? null);

    // tolerance_percent
    rowParams.push(`$${paramIndex++}`);
    values.push(input.tolerancePercent ?? null);

    // is_critical
    rowParams.push(`$${paramIndex++}`);
    values.push(input.isCritical ?? false);

    // passed
    rowParams.push(`$${paramIndex++}`);
    values.push(passed);

    // failure_reason
    rowParams.push(`$${paramIndex++}`);
    values.push(failureReason || null);

    // timestamp_offset_ms
    rowParams.push(`$${paramIndex++}`);
    values.push(input.timestampOffsetMs ?? null);

    // measured_at
    rowParams.push('NOW()');

    // metadata
    rowParams.push(`$${paramIndex++}`);
    values.push(JSON.stringify(input.metadata || {}));

    rows.push(`(${rowParams.join(', ')})`);
  }

  const sql = `INSERT INTO hil_measurements (
    test_run_id, captured_data_id, step_id, measurement_type, measurement_name,
    channel, value, unit, min_limit, max_limit, nominal_value, tolerance_percent,
    is_critical, passed, failure_reason, timestamp_offset_ms, measured_at, metadata
  ) VALUES ${rows.join(', ')} RETURNING *`;

  const result = await query<MeasurementRow>(sql, values, {
    operation: 'create_measurements_bulk',
  });

  logger.info('Measurements created in bulk', { count: result.rows.length });

  return result.rows.map(mapRowToMeasurement);
}

/**
 * Find a measurement by its ID.
 *
 * @param id - Measurement UUID
 * @returns The measurement or null if not found
 */
export async function findById(id: string): Promise<HILMeasurement | null> {
  const logger = repoLogger.child({ operation: 'findById', measurementId: id });

  if (!id || id.trim().length === 0) {
    throw new ValidationError('Measurement ID is required', {
      operation: 'findById',
    });
  }

  logger.debug('Finding measurement by ID');

  const result = await query<MeasurementRow>(
    'SELECT * FROM hil_measurements WHERE id = $1',
    [id],
    { operation: 'find_measurement_by_id' }
  );

  if (result.rows.length === 0) {
    logger.debug('Measurement not found');
    return null;
  }

  return mapRowToMeasurement(result.rows[0]);
}

/**
 * Find all measurements for a test run with optional filters.
 *
 * @param testRunId - Test run UUID
 * @param filters - Optional filters
 * @returns Array of measurements
 */
export async function findByTestRun(
  testRunId: string,
  filters?: MeasurementFilters
): Promise<HILMeasurement[]> {
  const logger = repoLogger.child({ operation: 'findByTestRun', testRunId });

  if (!testRunId || testRunId.trim().length === 0) {
    throw new ValidationError('Test run ID is required', {
      operation: 'findByTestRun',
    });
  }

  const conditions: string[] = ['test_run_id = $1'];
  const values: unknown[] = [testRunId];
  let paramIndex = 2;

  if (filters?.measurementType) {
    validateMeasurementType(filters.measurementType);
    conditions.push(`measurement_type = $${paramIndex++}`);
    values.push(filters.measurementType);
  }

  if (filters?.passed !== undefined) {
    conditions.push(`passed = $${paramIndex++}`);
    values.push(filters.passed);
  }

  if (filters?.isCritical !== undefined) {
    conditions.push(`is_critical = $${paramIndex++}`);
    values.push(filters.isCritical);
  }

  if (filters?.isWarning !== undefined) {
    conditions.push(`is_warning = $${paramIndex++}`);
    values.push(filters.isWarning);
  }

  if (filters?.stepId) {
    conditions.push(`step_id = $${paramIndex++}`);
    values.push(filters.stepId);
  }

  if (filters?.capturedDataId) {
    conditions.push(`captured_data_id = $${paramIndex++}`);
    values.push(filters.capturedDataId);
  }

  if (filters?.channel) {
    conditions.push(`channel = $${paramIndex++}`);
    values.push(filters.channel);
  }

  const sql = `SELECT * FROM hil_measurements
               WHERE ${conditions.join(' AND ')}
               ORDER BY measured_at ASC`;

  logger.debug('Finding measurements by test run', { filters });

  const result = await query<MeasurementRow>(sql, values, {
    operation: 'find_measurements_by_test_run',
  });

  logger.debug('Found measurements', { count: result.rows.length });

  return result.rows.map(mapRowToMeasurement);
}

/**
 * Find failed measurements for a test run.
 *
 * @param testRunId - Test run UUID
 * @param criticalOnly - Only return critical failures
 * @returns Array of failed measurements
 */
export async function findFailed(
  testRunId: string,
  criticalOnly: boolean = false
): Promise<HILMeasurement[]> {
  const logger = repoLogger.child({ operation: 'findFailed', testRunId });

  if (!testRunId || testRunId.trim().length === 0) {
    throw new ValidationError('Test run ID is required', {
      operation: 'findFailed',
    });
  }

  let sql = `SELECT * FROM hil_measurements
             WHERE test_run_id = $1 AND passed = false`;

  if (criticalOnly) {
    sql += ' AND is_critical = true';
  }

  sql += ' ORDER BY measured_at ASC';

  logger.debug('Finding failed measurements', { criticalOnly });

  const result = await query<MeasurementRow>(sql, [testRunId], {
    operation: 'find_failed_measurements',
  });

  return result.rows.map(mapRowToMeasurement);
}

/**
 * Find measurements by step.
 *
 * @param testRunId - Test run UUID
 * @param stepId - Step ID
 * @returns Array of measurements
 */
export async function findByStep(
  testRunId: string,
  stepId: string
): Promise<HILMeasurement[]> {
  const logger = repoLogger.child({ operation: 'findByStep', testRunId, stepId });

  if (!testRunId || testRunId.trim().length === 0) {
    throw new ValidationError('Test run ID is required', {
      operation: 'findByStep',
    });
  }

  if (!stepId || stepId.trim().length === 0) {
    throw new ValidationError('Step ID is required', {
      operation: 'findByStep',
    });
  }

  logger.debug('Finding measurements by step');

  const result = await query<MeasurementRow>(
    `SELECT * FROM hil_measurements
     WHERE test_run_id = $1 AND step_id = $2
     ORDER BY measured_at ASC`,
    [testRunId, stepId],
    { operation: 'find_measurements_by_step' }
  );

  return result.rows.map(mapRowToMeasurement);
}

/**
 * Update a measurement.
 *
 * @param id - Measurement UUID
 * @param data - Fields to update
 * @returns The updated measurement
 */
export async function update(
  id: string,
  data: UpdateMeasurementInput
): Promise<HILMeasurement> {
  const logger = repoLogger.child({ operation: 'update', measurementId: id });

  if (!id || id.trim().length === 0) {
    throw new ValidationError('Measurement ID is required', {
      operation: 'update',
    });
  }

  // Check if measurement exists
  const existing = await findById(id);
  if (!existing) {
    throw new NotFoundError('HILMeasurement', id, { operation: 'update' });
  }

  const updateData: Record<string, unknown> = {};

  if (data.value !== undefined) {
    updateData.value = data.value;
  }

  if (data.passed !== undefined) {
    updateData.passed = data.passed;
  }

  if (data.failureReason !== undefined) {
    updateData.failure_reason = data.failureReason || null;
  }

  if (data.isWarning !== undefined) {
    updateData.is_warning = data.isWarning;
  }

  if (data.sampleCount !== undefined) {
    updateData.sample_count = data.sampleCount;
  }

  if (data.meanValue !== undefined) {
    updateData.mean_value = data.meanValue;
  }

  if (data.stdDeviation !== undefined) {
    updateData.std_deviation = data.stdDeviation;
  }

  if (data.minObserved !== undefined) {
    updateData.min_observed = data.minObserved;
  }

  if (data.maxObserved !== undefined) {
    updateData.max_observed = data.maxObserved;
  }

  if (data.metadata !== undefined) {
    updateData.metadata = JSON.stringify(data.metadata);
  }

  if (Object.keys(updateData).length === 0) {
    logger.debug('No fields to update');
    return existing;
  }

  logger.debug('Updating measurement', { fields: Object.keys(updateData) });

  const { text, values } = buildUpdate('hil_measurements', updateData, 'id', id);
  const result = await query<MeasurementRow>(text, values, {
    operation: 'update_measurement',
  });

  if (result.rows.length === 0) {
    throw new DatabaseError(
      'Failed to update measurement - no row returned',
      new Error('Update returned no rows'),
      { operation: 'update_measurement' }
    );
  }

  return mapRowToMeasurement(result.rows[0]);
}

/**
 * Add statistics to a measurement.
 *
 * @param id - Measurement UUID
 * @param stats - Statistics data
 * @returns The updated measurement
 */
export async function addStatistics(
  id: string,
  stats: {
    sampleCount: number;
    meanValue: number;
    stdDeviation: number;
    minObserved: number;
    maxObserved: number;
  }
): Promise<HILMeasurement> {
  const logger = repoLogger.child({ operation: 'addStatistics', measurementId: id });

  if (!id || id.trim().length === 0) {
    throw new ValidationError('Measurement ID is required', {
      operation: 'addStatistics',
    });
  }

  logger.debug('Adding statistics', stats);

  const result = await query<MeasurementRow>(
    `UPDATE hil_measurements SET
       sample_count = $1,
       mean_value = $2,
       std_deviation = $3,
       min_observed = $4,
       max_observed = $5
     WHERE id = $6 RETURNING *`,
    [
      stats.sampleCount,
      stats.meanValue,
      stats.stdDeviation,
      stats.minObserved,
      stats.maxObserved,
      id,
    ],
    { operation: 'add_statistics' }
  );

  if (result.rows.length === 0) {
    throw new DatabaseError(
      'Failed to add statistics - no row returned',
      new Error('Update returned no rows'),
      { operation: 'add_statistics' }
    );
  }

  return mapRowToMeasurement(result.rows[0]);
}

/**
 * Delete a measurement.
 *
 * @param id - Measurement UUID
 */
export async function deleteMeasurement(id: string): Promise<void> {
  const logger = repoLogger.child({ operation: 'delete', measurementId: id });

  if (!id || id.trim().length === 0) {
    throw new ValidationError('Measurement ID is required', {
      operation: 'delete',
    });
  }

  const existing = await findById(id);
  if (!existing) {
    throw new NotFoundError('HILMeasurement', id, { operation: 'delete' });
  }

  logger.debug('Deleting measurement');

  await query('DELETE FROM hil_measurements WHERE id = $1', [id], {
    operation: 'delete_measurement',
  });

  logger.info('Measurement deleted', { measurementId: id });
}

/**
 * Get measurement summary for a test run.
 *
 * @param testRunId - Test run UUID
 * @returns Summary statistics
 */
export async function getSummary(testRunId: string): Promise<{
  total: number;
  passed: number;
  failed: number;
  warnings: number;
  criticalTotal: number;
  criticalPassed: number;
  criticalFailed: number;
  passRate: number;
  criticalPassRate: number | null;
}> {
  const logger = repoLogger.child({ operation: 'getSummary', testRunId });

  if (!testRunId || testRunId.trim().length === 0) {
    throw new ValidationError('Test run ID is required', {
      operation: 'getSummary',
    });
  }

  logger.debug('Getting measurement summary');

  const result = await query<{
    total: string;
    passed: string;
    failed: string;
    warnings: string;
    critical_total: string;
    critical_passed: string;
    critical_failed: string;
  }>(
    `SELECT
       COUNT(*) as total,
       COUNT(*) FILTER (WHERE passed = true) as passed,
       COUNT(*) FILTER (WHERE passed = false) as failed,
       COUNT(*) FILTER (WHERE is_warning = true) as warnings,
       COUNT(*) FILTER (WHERE is_critical = true) as critical_total,
       COUNT(*) FILTER (WHERE is_critical = true AND passed = true) as critical_passed,
       COUNT(*) FILTER (WHERE is_critical = true AND passed = false) as critical_failed
     FROM hil_measurements
     WHERE test_run_id = $1`,
    [testRunId],
    { operation: 'get_measurement_summary' }
  );

  const stats = result.rows[0];
  const total = parseInt(stats.total, 10);
  const passed = parseInt(stats.passed, 10);
  const criticalTotal = parseInt(stats.critical_total, 10);
  const criticalPassed = parseInt(stats.critical_passed, 10);

  return {
    total,
    passed,
    failed: parseInt(stats.failed, 10),
    warnings: parseInt(stats.warnings, 10),
    criticalTotal,
    criticalPassed,
    criticalFailed: parseInt(stats.critical_failed, 10),
    passRate: total > 0 ? (passed / total) * 100 : 0,
    criticalPassRate: criticalTotal > 0 ? (criticalPassed / criticalTotal) * 100 : null,
  };
}

/**
 * Get measurements grouped by type for a test run.
 *
 * @param testRunId - Test run UUID
 * @returns Measurements grouped by type with aggregations
 */
export async function getByType(testRunId: string): Promise<
  Record<
    HILMeasurementType,
    {
      count: number;
      passed: number;
      failed: number;
      avgValue: number | null;
      minValue: number | null;
      maxValue: number | null;
      unit: string | null;
    }
  >
> {
  const logger = repoLogger.child({ operation: 'getByType', testRunId });

  if (!testRunId || testRunId.trim().length === 0) {
    throw new ValidationError('Test run ID is required', {
      operation: 'getByType',
    });
  }

  logger.debug('Getting measurements by type');

  const result = await query<{
    measurement_type: HILMeasurementType;
    count: string;
    passed: string;
    failed: string;
    avg_value: string | null;
    min_value: string | null;
    max_value: string | null;
    unit: string | null;
  }>(
    `SELECT
       measurement_type,
       COUNT(*) as count,
       COUNT(*) FILTER (WHERE passed = true) as passed,
       COUNT(*) FILTER (WHERE passed = false) as failed,
       AVG(value) as avg_value,
       MIN(value) as min_value,
       MAX(value) as max_value,
       (SELECT unit FROM hil_measurements WHERE test_run_id = $1 AND measurement_type = m.measurement_type LIMIT 1) as unit
     FROM hil_measurements m
     WHERE test_run_id = $1
     GROUP BY measurement_type`,
    [testRunId],
    { operation: 'get_measurements_by_type' }
  );

  const byType: Record<string, unknown> = {};
  for (const row of result.rows) {
    byType[row.measurement_type] = {
      count: parseInt(row.count, 10),
      passed: parseInt(row.passed, 10),
      failed: parseInt(row.failed, 10),
      avgValue: row.avg_value ? parseFloat(row.avg_value) : null,
      minValue: row.min_value ? parseFloat(row.min_value) : null,
      maxValue: row.max_value ? parseFloat(row.max_value) : null,
      unit: row.unit,
    };
  }

  return byType as Record<
    HILMeasurementType,
    {
      count: number;
      passed: number;
      failed: number;
      avgValue: number | null;
      minValue: number | null;
      maxValue: number | null;
      unit: string | null;
    }
  >;
}

/**
 * Get key metrics for a test run (for summary display).
 *
 * @param testRunId - Test run UUID
 * @returns Key metrics object
 */
export async function getKeyMetrics(
  testRunId: string
): Promise<Record<string, number>> {
  const logger = repoLogger.child({ operation: 'getKeyMetrics', testRunId });

  if (!testRunId || testRunId.trim().length === 0) {
    throw new ValidationError('Test run ID is required', {
      operation: 'getKeyMetrics',
    });
  }

  logger.debug('Getting key metrics');

  // Get latest measurement for each important type
  const keyTypes: HILMeasurementType[] = [
    'efficiency',
    'rms_current',
    'speed',
    'temperature',
    'thd',
    'power',
    'startup_time',
  ];

  const result = await query<{
    measurement_type: HILMeasurementType;
    value: number;
  }>(
    `SELECT DISTINCT ON (measurement_type)
       measurement_type,
       value
     FROM hil_measurements
     WHERE test_run_id = $1
       AND measurement_type = ANY($2)
     ORDER BY measurement_type, measured_at DESC`,
    [testRunId, keyTypes],
    { operation: 'get_key_metrics' }
  );

  const metrics: Record<string, number> = {};
  for (const row of result.rows) {
    metrics[row.measurement_type] = row.value;
  }

  return metrics;
}

/**
 * Compare measurements between two test runs.
 *
 * @param runId1 - First test run UUID
 * @param runId2 - Second test run UUID
 * @returns Comparison of measurements
 */
export async function compareMeasurements(
  runId1: string,
  runId2: string
): Promise<{
  improvements: Array<{
    type: HILMeasurementType;
    run1Value: number;
    run2Value: number;
    changePercent: number;
    unit: string;
  }>;
  regressions: Array<{
    type: HILMeasurementType;
    run1Value: number;
    run2Value: number;
    changePercent: number;
    unit: string;
  }>;
}> {
  const logger = repoLogger.child({
    operation: 'compareMeasurements',
    runId1,
    runId2,
  });

  if (!runId1 || !runId2) {
    throw new ValidationError('Both run IDs are required', {
      operation: 'compareMeasurements',
    });
  }

  logger.debug('Comparing measurements between runs');

  const result = await query<{
    measurement_type: HILMeasurementType;
    run1_value: number;
    run2_value: number;
    unit: string;
  }>(
    `SELECT
       r1.measurement_type,
       r1.value as run1_value,
       r2.value as run2_value,
       r1.unit
     FROM (
       SELECT DISTINCT ON (measurement_type)
         measurement_type, value, unit
       FROM hil_measurements
       WHERE test_run_id = $1
       ORDER BY measurement_type, measured_at DESC
     ) r1
     INNER JOIN (
       SELECT DISTINCT ON (measurement_type)
         measurement_type, value
       FROM hil_measurements
       WHERE test_run_id = $2
       ORDER BY measurement_type, measured_at DESC
     ) r2 ON r1.measurement_type = r2.measurement_type`,
    [runId1, runId2],
    { operation: 'compare_measurements' }
  );

  const improvements: Array<{
    type: HILMeasurementType;
    run1Value: number;
    run2Value: number;
    changePercent: number;
    unit: string;
  }> = [];

  const regressions: Array<{
    type: HILMeasurementType;
    run1Value: number;
    run2Value: number;
    changePercent: number;
    unit: string;
  }> = [];

  for (const row of result.rows) {
    if (row.run1_value === 0) continue;

    const changePercent =
      ((row.run2_value - row.run1_value) / Math.abs(row.run1_value)) * 100;

    // Skip insignificant changes (< 1%)
    if (Math.abs(changePercent) < 1) continue;

    const entry = {
      type: row.measurement_type,
      run1Value: row.run1_value,
      run2Value: row.run2_value,
      changePercent,
      unit: row.unit,
    };

    // Determine if change is improvement or regression
    // For most metrics, higher is better (efficiency, speed)
    // For some, lower is better (thd, ripple, temperature)
    const lowerIsBetter = [
      'thd',
      'ripple',
      'overshoot',
      'settling_time',
      'startup_time',
      'dead_time',
    ];

    if (lowerIsBetter.includes(row.measurement_type)) {
      if (changePercent < 0) {
        improvements.push(entry);
      } else {
        regressions.push(entry);
      }
    } else {
      if (changePercent > 0) {
        improvements.push(entry);
      } else {
        regressions.push(entry);
      }
    }
  }

  return { improvements, regressions };
}

// Default export
export default {
  create,
  createBulk,
  findById,
  findByTestRun,
  findFailed,
  findByStep,
  update,
  addStatistics,
  delete: deleteMeasurement,
  getSummary,
  getByType,
  getKeyMetrics,
  compareMeasurements,
};
