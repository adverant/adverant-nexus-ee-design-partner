/**
 * HIL Captured Data Repository
 *
 * Database operations for HIL captured data (waveforms, logic traces, etc.).
 * Handles CRUD operations, data storage, and analysis results.
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
  HILCapturedData,
  HILCaptureType,
  HILDataFormat,
  HILChannelConfig,
  HILTriggerConfig,
  HILAnalysisResults,
  HILDataAnnotation,
} from '../../types/hil-types.js';

// ============================================================================
// Types
// ============================================================================

interface CapturedDataRow {
  id: string;
  test_run_id: string;
  instrument_id: string | null;
  name: string | null;
  capture_type: HILCaptureType;
  step_id: string | null;
  channel_config: HILChannelConfig[];
  sample_rate_hz: number | null;
  sample_count: number | null;
  duration_ms: number | null;
  trigger_config: HILTriggerConfig | null;
  data_format: HILDataFormat;
  data_path: string | null;
  data_inline: unknown | null;
  data_size_bytes: number | null;
  data_checksum: string | null;
  compression: string | null;
  analysis_results: HILAnalysisResults | null;
  annotations: HILDataAnnotation[] | null;
  captured_at: Date;
  processing_completed_at: Date | null;
  metadata: Record<string, unknown>;
  created_at: Date;
}

interface CreateCapturedDataInput {
  testRunId: string;
  instrumentId?: string;
  name?: string;
  captureType: HILCaptureType;
  stepId?: string;
  channelConfig: HILChannelConfig[];
  sampleRateHz?: number;
  sampleCount?: number;
  durationMs?: number;
  triggerConfig?: HILTriggerConfig;
  dataFormat: HILDataFormat;
  dataPath?: string;
  dataInline?: unknown;
  dataSizeBytes?: number;
  dataChecksum?: string;
  compression?: string;
  metadata?: Record<string, unknown>;
}

interface UpdateCapturedDataInput {
  name?: string;
  analysisResults?: HILAnalysisResults;
  annotations?: HILDataAnnotation[];
  processingCompletedAt?: string;
  metadata?: Record<string, unknown>;
}

// ============================================================================
// Repository Implementation
// ============================================================================

const repoLogger: Logger = log.child({ service: 'hil-captured-data-repository' });

/**
 * Validate capture type
 */
function validateCaptureType(type: string): asserts type is HILCaptureType {
  const validTypes: HILCaptureType[] = [
    'waveform',
    'logic_trace',
    'spectrum',
    'thermal_image',
    'measurement',
    'protocol_decode',
    'can_log',
  ];

  if (!validTypes.includes(type as HILCaptureType)) {
    throw new ValidationError(`Invalid capture type: ${type}`, {
      operation: 'validate_capture_type',
      validTypes,
    });
  }
}

/**
 * Validate data format
 */
function validateDataFormat(format: string): asserts format is HILDataFormat {
  const validFormats: HILDataFormat[] = [
    'binary',
    'csv',
    'json',
    'vcd',
    'salae',
    'sigrok',
  ];

  if (!validFormats.includes(format as HILDataFormat)) {
    throw new ValidationError(`Invalid data format: ${format}`, {
      operation: 'validate_data_format',
      validFormats,
    });
  }
}

/**
 * Map a database row to an HILCapturedData object.
 */
function mapRowToData(row: CapturedDataRow): HILCapturedData {
  return {
    id: row.id,
    testRunId: row.test_run_id,
    instrumentId: row.instrument_id || undefined,
    name: row.name || undefined,
    captureType: row.capture_type,
    stepId: row.step_id || undefined,
    channelConfig: row.channel_config || [],
    sampleRateHz: row.sample_rate_hz ?? undefined,
    sampleCount: row.sample_count ?? undefined,
    durationMs: row.duration_ms ?? undefined,
    triggerConfig: row.trigger_config || undefined,
    dataFormat: row.data_format,
    dataPath: row.data_path || undefined,
    dataInline: row.data_inline || undefined,
    dataSizeBytes: row.data_size_bytes ?? undefined,
    dataChecksum: row.data_checksum || undefined,
    compression: row.compression || undefined,
    analysisResults: row.analysis_results || undefined,
    annotations: row.annotations || undefined,
    capturedAt: row.captured_at.toISOString(),
    processingCompletedAt: row.processing_completed_at?.toISOString(),
    metadata: row.metadata || {},
    createdAt: row.created_at.toISOString(),
  };
}

/**
 * Validate channel configuration
 */
function validateChannelConfig(config: HILChannelConfig[]): void {
  if (!config || !Array.isArray(config)) {
    throw new ValidationError('Channel configuration must be an array', {
      operation: 'validate_channel_config',
    });
  }

  for (let i = 0; i < config.length; i++) {
    const channel = config[i];
    if (!channel.name || channel.name.trim().length === 0) {
      throw new ValidationError(`Channel ${i} must have a name`, {
        operation: 'validate_channel_config',
        channelIndex: i,
      });
    }
    if (typeof channel.scale !== 'number') {
      throw new ValidationError(`Channel ${i} must have a numeric scale`, {
        operation: 'validate_channel_config',
        channelIndex: i,
      });
    }
    if (typeof channel.offset !== 'number') {
      throw new ValidationError(`Channel ${i} must have a numeric offset`, {
        operation: 'validate_channel_config',
        channelIndex: i,
      });
    }
    if (!channel.unit || channel.unit.trim().length === 0) {
      throw new ValidationError(`Channel ${i} must have a unit`, {
        operation: 'validate_channel_config',
        channelIndex: i,
      });
    }
  }
}

/**
 * Create a new captured data record.
 *
 * @param input - Captured data creation data
 * @returns The created captured data record
 */
export async function create(input: CreateCapturedDataInput): Promise<HILCapturedData> {
  const logger = repoLogger.child({ operation: 'create' });

  // Validate required fields
  if (!input.testRunId || input.testRunId.trim().length === 0) {
    throw new ValidationError('Test run ID is required', {
      operation: 'create',
      input,
    });
  }

  if (!input.captureType) {
    throw new ValidationError('Capture type is required', {
      operation: 'create',
      input,
    });
  }

  if (!input.channelConfig || input.channelConfig.length === 0) {
    throw new ValidationError('At least one channel configuration is required', {
      operation: 'create',
      input,
    });
  }

  if (!input.dataFormat) {
    throw new ValidationError('Data format is required', {
      operation: 'create',
      input,
    });
  }

  // Either dataPath or dataInline must be provided
  if (!input.dataPath && input.dataInline === undefined) {
    throw new ValidationError('Either dataPath or dataInline must be provided', {
      operation: 'create',
      input,
    });
  }

  // Validate types
  validateCaptureType(input.captureType);
  validateDataFormat(input.dataFormat);
  validateChannelConfig(input.channelConfig);

  const data: Record<string, unknown> = {
    test_run_id: input.testRunId,
    instrument_id: input.instrumentId || null,
    name: input.name || null,
    capture_type: input.captureType,
    step_id: input.stepId || null,
    channel_config: JSON.stringify(input.channelConfig),
    sample_rate_hz: input.sampleRateHz || null,
    sample_count: input.sampleCount || null,
    duration_ms: input.durationMs || null,
    trigger_config: input.triggerConfig ? JSON.stringify(input.triggerConfig) : null,
    data_format: input.dataFormat,
    data_path: input.dataPath || null,
    data_inline: input.dataInline !== undefined ? JSON.stringify(input.dataInline) : null,
    data_size_bytes: input.dataSizeBytes || null,
    data_checksum: input.dataChecksum || null,
    compression: input.compression || null,
    captured_at: new Date(),
    metadata: JSON.stringify(input.metadata || {}),
  };

  logger.debug('Creating captured data', {
    testRunId: input.testRunId,
    captureType: input.captureType,
    channelCount: input.channelConfig.length,
    sampleRateHz: input.sampleRateHz,
  });

  const { text, values } = buildInsert('hil_captured_data', data);
  const result = await query<CapturedDataRow>(text, values, {
    operation: 'create_captured_data',
  });

  if (result.rows.length === 0) {
    throw new DatabaseError(
      'Failed to create captured data - no row returned',
      new Error('Insert returned no rows'),
      { operation: 'create_captured_data' }
    );
  }

  const capturedData = mapRowToData(result.rows[0]);
  logger.info('Captured data created', {
    capturedDataId: capturedData.id,
    testRunId: capturedData.testRunId,
    captureType: capturedData.captureType,
  });

  return capturedData;
}

/**
 * Find captured data by its ID.
 *
 * @param id - Captured data UUID
 * @returns The captured data or null if not found
 */
export async function findById(id: string): Promise<HILCapturedData | null> {
  const logger = repoLogger.child({ operation: 'findById', capturedDataId: id });

  if (!id || id.trim().length === 0) {
    throw new ValidationError('Captured data ID is required', {
      operation: 'findById',
    });
  }

  logger.debug('Finding captured data by ID');

  const result = await query<CapturedDataRow>(
    'SELECT * FROM hil_captured_data WHERE id = $1',
    [id],
    { operation: 'find_captured_data_by_id' }
  );

  if (result.rows.length === 0) {
    logger.debug('Captured data not found');
    return null;
  }

  return mapRowToData(result.rows[0]);
}

/**
 * Find all captured data for a test run.
 *
 * @param testRunId - Test run UUID
 * @param captureType - Optional capture type filter
 * @returns Array of captured data
 */
export async function findByTestRun(
  testRunId: string,
  captureType?: HILCaptureType
): Promise<HILCapturedData[]> {
  const logger = repoLogger.child({ operation: 'findByTestRun', testRunId });

  if (!testRunId || testRunId.trim().length === 0) {
    throw new ValidationError('Test run ID is required', {
      operation: 'findByTestRun',
    });
  }

  let sql = 'SELECT * FROM hil_captured_data WHERE test_run_id = $1';
  const values: unknown[] = [testRunId];

  if (captureType) {
    validateCaptureType(captureType);
    sql += ' AND capture_type = $2';
    values.push(captureType);
  }

  sql += ' ORDER BY captured_at ASC';

  logger.debug('Finding captured data by test run', { captureType });

  const result = await query<CapturedDataRow>(sql, values, {
    operation: 'find_captured_data_by_test_run',
  });

  logger.debug('Found captured data', { count: result.rows.length });

  return result.rows.map(mapRowToData);
}

/**
 * Find captured data by instrument.
 *
 * @param instrumentId - Instrument UUID
 * @param limit - Max results (default 50)
 * @returns Array of captured data
 */
export async function findByInstrument(
  instrumentId: string,
  limit: number = 50
): Promise<HILCapturedData[]> {
  const logger = repoLogger.child({ operation: 'findByInstrument', instrumentId });

  if (!instrumentId || instrumentId.trim().length === 0) {
    throw new ValidationError('Instrument ID is required', {
      operation: 'findByInstrument',
    });
  }

  logger.debug('Finding captured data by instrument');

  const result = await query<CapturedDataRow>(
    `SELECT * FROM hil_captured_data
     WHERE instrument_id = $1
     ORDER BY captured_at DESC
     LIMIT $2`,
    [instrumentId, limit],
    { operation: 'find_captured_data_by_instrument' }
  );

  return result.rows.map(mapRowToData);
}

/**
 * Find captured data by step ID within a test run.
 *
 * @param testRunId - Test run UUID
 * @param stepId - Step ID
 * @returns Array of captured data
 */
export async function findByStep(
  testRunId: string,
  stepId: string
): Promise<HILCapturedData[]> {
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

  logger.debug('Finding captured data by step');

  const result = await query<CapturedDataRow>(
    `SELECT * FROM hil_captured_data
     WHERE test_run_id = $1 AND step_id = $2
     ORDER BY captured_at ASC`,
    [testRunId, stepId],
    { operation: 'find_captured_data_by_step' }
  );

  return result.rows.map(mapRowToData);
}

/**
 * Update captured data.
 *
 * @param id - Captured data UUID
 * @param data - Fields to update
 * @returns The updated captured data
 */
export async function update(
  id: string,
  data: UpdateCapturedDataInput
): Promise<HILCapturedData> {
  const logger = repoLogger.child({ operation: 'update', capturedDataId: id });

  if (!id || id.trim().length === 0) {
    throw new ValidationError('Captured data ID is required', {
      operation: 'update',
    });
  }

  // Check if captured data exists
  const existing = await findById(id);
  if (!existing) {
    throw new NotFoundError('HILCapturedData', id, { operation: 'update' });
  }

  const updateData: Record<string, unknown> = {};

  if (data.name !== undefined) {
    updateData.name = data.name || null;
  }

  if (data.analysisResults !== undefined) {
    updateData.analysis_results = JSON.stringify(data.analysisResults);
    updateData.processing_completed_at = new Date();
  }

  if (data.annotations !== undefined) {
    updateData.annotations = JSON.stringify(data.annotations);
  }

  if (data.processingCompletedAt !== undefined) {
    updateData.processing_completed_at = data.processingCompletedAt;
  }

  if (data.metadata !== undefined) {
    updateData.metadata = JSON.stringify(data.metadata);
  }

  if (Object.keys(updateData).length === 0) {
    logger.debug('No fields to update');
    return existing;
  }

  logger.debug('Updating captured data', { fields: Object.keys(updateData) });

  const { text, values } = buildUpdate('hil_captured_data', updateData, 'id', id);
  const result = await query<CapturedDataRow>(text, values, {
    operation: 'update_captured_data',
  });

  if (result.rows.length === 0) {
    throw new DatabaseError(
      'Failed to update captured data - no row returned',
      new Error('Update returned no rows'),
      { operation: 'update_captured_data' }
    );
  }

  const capturedData = mapRowToData(result.rows[0]);
  logger.info('Captured data updated', { capturedDataId: capturedData.id });

  return capturedData;
}

/**
 * Store analysis results for captured data.
 *
 * @param id - Captured data UUID
 * @param analysisResults - Analysis results
 * @returns The updated captured data
 */
export async function storeAnalysisResults(
  id: string,
  analysisResults: HILAnalysisResults
): Promise<HILCapturedData> {
  const logger = repoLogger.child({
    operation: 'storeAnalysisResults',
    capturedDataId: id,
  });

  if (!id || id.trim().length === 0) {
    throw new ValidationError('Captured data ID is required', {
      operation: 'storeAnalysisResults',
    });
  }

  logger.debug('Storing analysis results', {
    hasFft: !!analysisResults.fft,
    hasMeasurements: !!analysisResults.measurements,
    hasProtocolDecode: !!analysisResults.protocolDecode,
  });

  const result = await query<CapturedDataRow>(
    `UPDATE hil_captured_data SET
       analysis_results = $1,
       processing_completed_at = NOW()
     WHERE id = $2 RETURNING *`,
    [JSON.stringify(analysisResults), id],
    { operation: 'store_analysis_results' }
  );

  if (result.rows.length === 0) {
    throw new DatabaseError(
      'Failed to store analysis results - no row returned',
      new Error('Update returned no rows'),
      { operation: 'store_analysis_results' }
    );
  }

  const capturedData = mapRowToData(result.rows[0]);
  logger.info('Analysis results stored', { capturedDataId: capturedData.id });

  return capturedData;
}

/**
 * Add annotation to captured data.
 *
 * @param id - Captured data UUID
 * @param annotation - Annotation to add
 * @returns The updated captured data
 */
export async function addAnnotation(
  id: string,
  annotation: HILDataAnnotation
): Promise<HILCapturedData> {
  const logger = repoLogger.child({ operation: 'addAnnotation', capturedDataId: id });

  if (!id || id.trim().length === 0) {
    throw new ValidationError('Captured data ID is required', {
      operation: 'addAnnotation',
    });
  }

  if (typeof annotation.timestamp_ms !== 'number') {
    throw new ValidationError('Annotation timestamp_ms must be a number', {
      operation: 'addAnnotation',
    });
  }

  if (!annotation.text || annotation.text.trim().length === 0) {
    throw new ValidationError('Annotation text is required', {
      operation: 'addAnnotation',
    });
  }

  logger.debug('Adding annotation', {
    timestamp_ms: annotation.timestamp_ms,
    type: annotation.type,
  });

  const result = await query<CapturedDataRow>(
    `UPDATE hil_captured_data SET
       annotations = COALESCE(annotations, '[]'::jsonb) || $1::jsonb
     WHERE id = $2 RETURNING *`,
    [JSON.stringify([annotation]), id],
    { operation: 'add_annotation' }
  );

  if (result.rows.length === 0) {
    throw new DatabaseError(
      'Failed to add annotation - no row returned',
      new Error('Update returned no rows'),
      { operation: 'add_annotation' }
    );
  }

  return mapRowToData(result.rows[0]);
}

/**
 * Delete captured data.
 *
 * @param id - Captured data UUID
 */
export async function deleteData(id: string): Promise<void> {
  const logger = repoLogger.child({ operation: 'delete', capturedDataId: id });

  if (!id || id.trim().length === 0) {
    throw new ValidationError('Captured data ID is required', {
      operation: 'delete',
    });
  }

  // Check if captured data exists
  const existing = await findById(id);
  if (!existing) {
    throw new NotFoundError('HILCapturedData', id, { operation: 'delete' });
  }

  logger.debug('Deleting captured data');

  // Delete associated measurements first
  await query(
    'DELETE FROM hil_measurements WHERE captured_data_id = $1',
    [id],
    { operation: 'delete_captured_data_measurements' }
  );

  // Delete the captured data
  await query('DELETE FROM hil_captured_data WHERE id = $1', [id], {
    operation: 'delete_captured_data',
  });

  logger.info('Captured data deleted', { capturedDataId: id });
}

/**
 * Get total data size for a test run.
 *
 * @param testRunId - Test run UUID
 * @returns Total size in bytes
 */
export async function getTotalSize(testRunId: string): Promise<number> {
  const logger = repoLogger.child({ operation: 'getTotalSize', testRunId });

  if (!testRunId || testRunId.trim().length === 0) {
    throw new ValidationError('Test run ID is required', {
      operation: 'getTotalSize',
    });
  }

  logger.debug('Getting total data size');

  const result = await query<{ total_size: string | null }>(
    `SELECT COALESCE(SUM(data_size_bytes), 0) as total_size
     FROM hil_captured_data
     WHERE test_run_id = $1`,
    [testRunId],
    { operation: 'get_total_size' }
  );

  return parseInt(result.rows[0].total_size || '0', 10);
}

/**
 * Find captured data with pending analysis.
 *
 * @param limit - Max results (default 10)
 * @returns Array of captured data awaiting analysis
 */
export async function findPendingAnalysis(limit: number = 10): Promise<HILCapturedData[]> {
  const logger = repoLogger.child({ operation: 'findPendingAnalysis' });

  logger.debug('Finding captured data pending analysis');

  const result = await query<CapturedDataRow>(
    `SELECT * FROM hil_captured_data
     WHERE analysis_results IS NULL
       AND processing_completed_at IS NULL
       AND capture_type IN ('waveform', 'logic_trace', 'spectrum')
     ORDER BY captured_at ASC
     LIMIT $1`,
    [limit],
    { operation: 'find_pending_analysis' }
  );

  return result.rows.map(mapRowToData);
}

/**
 * Get capture statistics for a test run.
 *
 * @param testRunId - Test run UUID
 * @returns Statistics object
 */
export async function getTestRunStats(testRunId: string): Promise<{
  totalCaptures: number;
  totalSizeBytes: number;
  byCaptureType: Record<HILCaptureType, number>;
  analyzedCount: number;
  pendingAnalysisCount: number;
}> {
  const logger = repoLogger.child({ operation: 'getTestRunStats', testRunId });

  if (!testRunId || testRunId.trim().length === 0) {
    throw new ValidationError('Test run ID is required', {
      operation: 'getTestRunStats',
    });
  }

  logger.debug('Getting test run capture stats');

  const result = await query<{
    total_captures: string;
    total_size: string;
    analyzed_count: string;
    pending_count: string;
  }>(
    `SELECT
       COUNT(*) as total_captures,
       COALESCE(SUM(data_size_bytes), 0) as total_size,
       COUNT(*) FILTER (WHERE analysis_results IS NOT NULL) as analyzed_count,
       COUNT(*) FILTER (WHERE analysis_results IS NULL AND capture_type IN ('waveform', 'logic_trace', 'spectrum')) as pending_count
     FROM hil_captured_data
     WHERE test_run_id = $1`,
    [testRunId],
    { operation: 'get_capture_stats' }
  );

  const byTypeResult = await query<{ capture_type: HILCaptureType; count: string }>(
    `SELECT capture_type, COUNT(*) as count
     FROM hil_captured_data
     WHERE test_run_id = $1
     GROUP BY capture_type`,
    [testRunId],
    { operation: 'get_capture_stats_by_type' }
  );

  const byCaptureType: Record<string, number> = {};
  for (const row of byTypeResult.rows) {
    byCaptureType[row.capture_type] = parseInt(row.count, 10);
  }

  const stats = result.rows[0];
  return {
    totalCaptures: parseInt(stats.total_captures, 10),
    totalSizeBytes: parseInt(stats.total_size, 10),
    byCaptureType: byCaptureType as Record<HILCaptureType, number>,
    analyzedCount: parseInt(stats.analyzed_count, 10),
    pendingAnalysisCount: parseInt(stats.pending_count, 10),
  };
}

/**
 * Find waveform captures for display.
 *
 * @param testRunId - Test run UUID
 * @returns Array of waveform captures with channel info
 */
export async function findWaveforms(testRunId: string): Promise<HILCapturedData[]> {
  const logger = repoLogger.child({ operation: 'findWaveforms', testRunId });

  if (!testRunId || testRunId.trim().length === 0) {
    throw new ValidationError('Test run ID is required', {
      operation: 'findWaveforms',
    });
  }

  logger.debug('Finding waveform captures');

  const result = await query<CapturedDataRow>(
    `SELECT * FROM hil_captured_data
     WHERE test_run_id = $1
       AND capture_type IN ('waveform', 'spectrum')
     ORDER BY captured_at ASC`,
    [testRunId],
    { operation: 'find_waveforms' }
  );

  return result.rows.map(mapRowToData);
}

/**
 * Find logic traces for display.
 *
 * @param testRunId - Test run UUID
 * @returns Array of logic trace captures
 */
export async function findLogicTraces(testRunId: string): Promise<HILCapturedData[]> {
  const logger = repoLogger.child({ operation: 'findLogicTraces', testRunId });

  if (!testRunId || testRunId.trim().length === 0) {
    throw new ValidationError('Test run ID is required', {
      operation: 'findLogicTraces',
    });
  }

  logger.debug('Finding logic trace captures');

  const result = await query<CapturedDataRow>(
    `SELECT * FROM hil_captured_data
     WHERE test_run_id = $1
       AND capture_type IN ('logic_trace', 'protocol_decode')
     ORDER BY captured_at ASC`,
    [testRunId],
    { operation: 'find_logic_traces' }
  );

  return result.rows.map(mapRowToData);
}

// Default export
export default {
  create,
  findById,
  findByTestRun,
  findByInstrument,
  findByStep,
  update,
  storeAnalysisResults,
  addAnnotation,
  delete: deleteData,
  getTotalSize,
  findPendingAnalysis,
  getTestRunStats,
  findWaveforms,
  findLogicTraces,
};
