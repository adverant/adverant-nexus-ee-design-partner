/**
 * HIL Instrument Repository
 *
 * Database operations for HIL test instruments. Handles CRUD operations,
 * connection status tracking, and instrument discovery.
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
  HILInstrument,
  HILInstrumentType,
  HILInstrumentStatus,
  HILConnectionType,
  HILInstrumentCapability,
  HILConnectionParams,
  HILInstrumentPreset,
  CreateHILInstrumentInput,
  UpdateHILInstrumentInput,
  HILInstrumentFilters,
} from '../../types/hil-types.js';

// ============================================================================
// Types
// ============================================================================

interface InstrumentRow {
  id: string;
  project_id: string;
  name: string;
  instrument_type: HILInstrumentType;
  manufacturer: string;
  model: string;
  serial_number: string | null;
  firmware_version: string | null;
  connection_type: HILConnectionType;
  connection_params: HILConnectionParams;
  capabilities: HILInstrumentCapability[];
  status: HILInstrumentStatus;
  last_seen_at: Date | null;
  last_error: string | null;
  error_count: number;
  calibration_date: Date | null;
  calibration_due_date: Date | null;
  calibration_certificate: string | null;
  presets: Record<string, HILInstrumentPreset>;
  default_preset: string | null;
  notes: string | null;
  tags: string[];
  metadata: Record<string, unknown>;
  created_at: Date;
  updated_at: Date;
}

// ============================================================================
// Repository Implementation
// ============================================================================

const repoLogger: Logger = log.child({ service: 'hil-instrument-repository' });

/**
 * Map a database row to an HILInstrument object.
 */
function mapRowToInstrument(row: InstrumentRow): HILInstrument {
  return {
    id: row.id,
    projectId: row.project_id,
    name: row.name,
    instrumentType: row.instrument_type,
    manufacturer: row.manufacturer,
    model: row.model,
    serialNumber: row.serial_number || undefined,
    firmwareVersion: row.firmware_version || undefined,
    connectionType: row.connection_type,
    connectionParams: row.connection_params || {},
    capabilities: row.capabilities || [],
    status: row.status,
    lastSeenAt: row.last_seen_at?.toISOString(),
    lastError: row.last_error || undefined,
    errorCount: row.error_count,
    calibrationDate: row.calibration_date?.toISOString().split('T')[0],
    calibrationDueDate: row.calibration_due_date?.toISOString().split('T')[0],
    calibrationCertificate: row.calibration_certificate || undefined,
    presets: row.presets || {},
    defaultPreset: row.default_preset || undefined,
    notes: row.notes || undefined,
    tags: row.tags || [],
    metadata: row.metadata || {},
    createdAt: row.created_at.toISOString(),
    updatedAt: row.updated_at.toISOString(),
  };
}

/**
 * Validate instrument type
 */
function validateInstrumentType(type: string): asserts type is HILInstrumentType {
  const validTypes: HILInstrumentType[] = [
    'logic_analyzer',
    'oscilloscope',
    'power_supply',
    'motor_emulator',
    'daq',
    'can_analyzer',
    'function_gen',
    'thermal_camera',
    'electronic_load',
  ];

  if (!validTypes.includes(type as HILInstrumentType)) {
    throw new ValidationError(`Invalid instrument type: ${type}`, {
      operation: 'validate_instrument_type',
      validTypes,
    });
  }
}

/**
 * Validate connection type
 */
function validateConnectionType(type: string): asserts type is HILConnectionType {
  const validTypes: HILConnectionType[] = [
    'usb',
    'ethernet',
    'gpib',
    'serial',
    'grpc',
    'modbus_tcp',
    'modbus_rtu',
  ];

  if (!validTypes.includes(type as HILConnectionType)) {
    throw new ValidationError(`Invalid connection type: ${type}`, {
      operation: 'validate_connection_type',
      validTypes,
    });
  }
}

/**
 * Create a new HIL instrument.
 *
 * @param input - Instrument creation data
 * @returns The created instrument
 */
export async function create(input: CreateHILInstrumentInput): Promise<HILInstrument> {
  const logger = repoLogger.child({ operation: 'create' });

  // Validate required fields
  if (!input.projectId || input.projectId.trim().length === 0) {
    throw new ValidationError('Project ID is required', {
      operation: 'create',
      input,
    });
  }

  if (!input.name || input.name.trim().length === 0) {
    throw new ValidationError('Instrument name is required', {
      operation: 'create',
      input,
    });
  }

  if (!input.instrumentType) {
    throw new ValidationError('Instrument type is required', {
      operation: 'create',
      input,
    });
  }

  if (!input.manufacturer || input.manufacturer.trim().length === 0) {
    throw new ValidationError('Manufacturer is required', {
      operation: 'create',
      input,
    });
  }

  if (!input.model || input.model.trim().length === 0) {
    throw new ValidationError('Model is required', {
      operation: 'create',
      input,
    });
  }

  if (!input.connectionType) {
    throw new ValidationError('Connection type is required', {
      operation: 'create',
      input,
    });
  }

  if (!input.connectionParams || Object.keys(input.connectionParams).length === 0) {
    throw new ValidationError('Connection parameters are required', {
      operation: 'create',
      input,
    });
  }

  // Validate enum values
  validateInstrumentType(input.instrumentType);
  validateConnectionType(input.connectionType);

  const data: Record<string, unknown> = {
    project_id: input.projectId,
    name: input.name.trim(),
    instrument_type: input.instrumentType,
    manufacturer: input.manufacturer.trim(),
    model: input.model.trim(),
    connection_type: input.connectionType,
    connection_params: JSON.stringify(input.connectionParams),
    capabilities: JSON.stringify(input.capabilities || []),
    serial_number: input.serialNumber || null,
    firmware_version: input.firmwareVersion || null,
    notes: input.notes || null,
    tags: input.tags || [],
    metadata: JSON.stringify(input.metadata || {}),
  };

  logger.debug('Creating HIL instrument', {
    name: input.name,
    projectId: input.projectId,
    type: input.instrumentType,
    manufacturer: input.manufacturer,
    model: input.model,
  });

  const { text, values } = buildInsert('hil_instruments', data);
  const result = await query<InstrumentRow>(text, values, { operation: 'create_hil_instrument' });

  if (result.rows.length === 0) {
    throw new DatabaseError(
      'Failed to create HIL instrument - no row returned',
      new Error('Insert returned no rows'),
      { operation: 'create_hil_instrument' }
    );
  }

  const instrument = mapRowToInstrument(result.rows[0]);
  logger.info('HIL instrument created', {
    instrumentId: instrument.id,
    name: instrument.name,
    type: instrument.instrumentType,
  });

  return instrument;
}

/**
 * Find an instrument by its ID.
 *
 * @param id - Instrument UUID
 * @returns The instrument or null if not found
 */
export async function findById(id: string): Promise<HILInstrument | null> {
  const logger = repoLogger.child({ operation: 'findById', instrumentId: id });

  if (!id || id.trim().length === 0) {
    throw new ValidationError('Instrument ID is required', {
      operation: 'findById',
    });
  }

  logger.debug('Finding instrument by ID');

  const result = await query<InstrumentRow>(
    'SELECT * FROM hil_instruments WHERE id = $1',
    [id],
    { operation: 'find_instrument_by_id' }
  );

  if (result.rows.length === 0) {
    logger.debug('Instrument not found');
    return null;
  }

  return mapRowToInstrument(result.rows[0]);
}

/**
 * Find all instruments for a project with optional filters.
 *
 * @param projectId - Project UUID
 * @param filters - Optional filters
 * @returns Array of instruments for the project
 */
export async function findByProject(
  projectId: string,
  filters?: HILInstrumentFilters
): Promise<HILInstrument[]> {
  const logger = repoLogger.child({ operation: 'findByProject', projectId });

  if (!projectId || projectId.trim().length === 0) {
    throw new ValidationError('Project ID is required', {
      operation: 'findByProject',
    });
  }

  const conditions: string[] = ['project_id = $1'];
  const values: unknown[] = [projectId];
  let paramIndex = 2;

  if (filters?.instrumentType) {
    validateInstrumentType(filters.instrumentType);
    conditions.push(`instrument_type = $${paramIndex++}`);
    values.push(filters.instrumentType);
  }

  if (filters?.status) {
    conditions.push(`status = $${paramIndex++}`);
    values.push(filters.status);
  }

  if (filters?.connectionType) {
    validateConnectionType(filters.connectionType);
    conditions.push(`connection_type = $${paramIndex++}`);
    values.push(filters.connectionType);
  }

  if (filters?.manufacturer) {
    conditions.push(`manufacturer ILIKE $${paramIndex++}`);
    values.push(`%${filters.manufacturer}%`);
  }

  if (filters?.tags && filters.tags.length > 0) {
    conditions.push(`tags && $${paramIndex++}`);
    values.push(filters.tags);
  }

  const sql = `SELECT * FROM hil_instruments WHERE ${conditions.join(' AND ')} ORDER BY name ASC`;

  logger.debug('Finding instruments by project', { filters });

  const result = await query<InstrumentRow>(sql, values, { operation: 'find_instruments_by_project' });

  logger.debug('Found instruments', { count: result.rows.length });

  return result.rows.map(mapRowToInstrument);
}

/**
 * Update instrument connection status.
 *
 * @param id - Instrument UUID
 * @param status - New status
 * @param error - Error message (for error status)
 * @returns The updated instrument
 */
export async function updateStatus(
  id: string,
  status: HILInstrumentStatus,
  error?: string
): Promise<HILInstrument> {
  const logger = repoLogger.child({ operation: 'updateStatus', instrumentId: id });

  if (!id || id.trim().length === 0) {
    throw new ValidationError('Instrument ID is required', {
      operation: 'updateStatus',
    });
  }

  const validStatuses: HILInstrumentStatus[] = [
    'connected',
    'disconnected',
    'error',
    'busy',
    'initializing',
  ];

  if (!validStatuses.includes(status)) {
    throw new ValidationError(`Invalid status: ${status}`, {
      operation: 'updateStatus',
      validStatuses,
    });
  }

  // Check if instrument exists
  const existing = await findById(id);
  if (!existing) {
    throw new NotFoundError('HILInstrument', id, { operation: 'updateStatus' });
  }

  const updateData: Record<string, unknown> = {
    status,
    last_seen_at: status === 'connected' ? new Date() : existing.lastSeenAt,
  };

  if (error) {
    updateData.last_error = error;
    updateData.error_count = (existing.errorCount || 0) + 1;
  } else if (status === 'connected') {
    updateData.last_error = null;
  }

  logger.debug('Updating instrument status', {
    currentStatus: existing.status,
    newStatus: status,
    hasError: !!error,
  });

  const setClauses = Object.keys(updateData).map((key, i) => `${key} = $${i + 1}`);
  const values = [...Object.values(updateData), id];

  const result = await query<InstrumentRow>(
    `UPDATE hil_instruments SET ${setClauses.join(', ')}, updated_at = NOW() WHERE id = $${values.length} RETURNING *`,
    values,
    { operation: 'update_instrument_status' }
  );

  if (result.rows.length === 0) {
    throw new DatabaseError(
      'Failed to update instrument status - no row returned',
      new Error('Update returned no rows'),
      { operation: 'update_instrument_status' }
    );
  }

  const instrument = mapRowToInstrument(result.rows[0]);
  logger.info('Instrument status updated', {
    instrumentId: instrument.id,
    previousStatus: existing.status,
    newStatus: status,
  });

  return instrument;
}

/**
 * Update instrument properties.
 *
 * @param id - Instrument UUID
 * @param data - Fields to update
 * @returns The updated instrument
 */
export async function update(id: string, data: UpdateHILInstrumentInput): Promise<HILInstrument> {
  const logger = repoLogger.child({ operation: 'update', instrumentId: id });

  if (!id || id.trim().length === 0) {
    throw new ValidationError('Instrument ID is required', {
      operation: 'update',
    });
  }

  // Check if instrument exists
  const existing = await findById(id);
  if (!existing) {
    throw new NotFoundError('HILInstrument', id, { operation: 'update' });
  }

  const updateData: Record<string, unknown> = {};

  if (data.name !== undefined) {
    if (data.name.trim().length === 0) {
      throw new ValidationError('Instrument name cannot be empty', {
        operation: 'update',
        instrumentId: id,
      });
    }
    updateData.name = data.name.trim();
  }

  if (data.connectionParams !== undefined) {
    updateData.connection_params = JSON.stringify(data.connectionParams);
  }

  if (data.capabilities !== undefined) {
    updateData.capabilities = JSON.stringify(data.capabilities);
  }

  if (data.status !== undefined) {
    updateData.status = data.status;
    if (data.status === 'connected') {
      updateData.last_seen_at = new Date();
    }
  }

  if (data.firmwareVersion !== undefined) {
    updateData.firmware_version = data.firmwareVersion || null;
  }

  if (data.calibrationDate !== undefined) {
    updateData.calibration_date = data.calibrationDate || null;
  }

  if (data.calibrationDueDate !== undefined) {
    updateData.calibration_due_date = data.calibrationDueDate || null;
  }

  if (data.notes !== undefined) {
    updateData.notes = data.notes || null;
  }

  if (data.tags !== undefined) {
    updateData.tags = data.tags;
  }

  if (data.presets !== undefined) {
    updateData.presets = JSON.stringify(data.presets);
  }

  if (data.defaultPreset !== undefined) {
    updateData.default_preset = data.defaultPreset || null;
  }

  if (data.metadata !== undefined) {
    updateData.metadata = JSON.stringify(data.metadata);
  }

  if (Object.keys(updateData).length === 0) {
    logger.debug('No fields to update');
    return existing;
  }

  logger.debug('Updating instrument', { fields: Object.keys(updateData) });

  const { text, values } = buildUpdate('hil_instruments', updateData, 'id', id);
  const result = await query<InstrumentRow>(text, values, { operation: 'update_instrument' });

  if (result.rows.length === 0) {
    throw new DatabaseError(
      'Failed to update instrument - no row returned',
      new Error('Update returned no rows'),
      { operation: 'update_instrument' }
    );
  }

  const instrument = mapRowToInstrument(result.rows[0]);
  logger.info('Instrument updated', { instrumentId: instrument.id });

  return instrument;
}

/**
 * Delete an instrument.
 *
 * @param id - Instrument UUID
 */
export async function deleteInstrument(id: string): Promise<void> {
  const logger = repoLogger.child({ operation: 'delete', instrumentId: id });

  if (!id || id.trim().length === 0) {
    throw new ValidationError('Instrument ID is required', {
      operation: 'delete',
    });
  }

  // Check if instrument exists
  const existing = await findById(id);
  if (!existing) {
    throw new NotFoundError('HILInstrument', id, { operation: 'delete' });
  }

  logger.debug('Deleting instrument');

  await query(
    'DELETE FROM hil_instruments WHERE id = $1',
    [id],
    { operation: 'delete_instrument' }
  );

  logger.info('Instrument deleted', { instrumentId: id, name: existing.name });
}

/**
 * Find instruments by type across all projects (for discovery).
 *
 * @param instrumentType - Instrument type to find
 * @returns Array of instruments
 */
export async function findByType(instrumentType: HILInstrumentType): Promise<HILInstrument[]> {
  const logger = repoLogger.child({ operation: 'findByType', instrumentType });

  validateInstrumentType(instrumentType);

  logger.debug('Finding instruments by type');

  const result = await query<InstrumentRow>(
    'SELECT * FROM hil_instruments WHERE instrument_type = $1 ORDER BY manufacturer, model, name',
    [instrumentType],
    { operation: 'find_instruments_by_type' }
  );

  return result.rows.map(mapRowToInstrument);
}

/**
 * Find connected instruments for a project.
 *
 * @param projectId - Project UUID
 * @returns Array of connected instruments
 */
export async function findConnected(projectId: string): Promise<HILInstrument[]> {
  const logger = repoLogger.child({ operation: 'findConnected', projectId });

  if (!projectId || projectId.trim().length === 0) {
    throw new ValidationError('Project ID is required', {
      operation: 'findConnected',
    });
  }

  logger.debug('Finding connected instruments');

  const result = await query<InstrumentRow>(
    `SELECT * FROM hil_instruments
     WHERE project_id = $1 AND status = 'connected'
     ORDER BY instrument_type, name`,
    [projectId],
    { operation: 'find_connected_instruments' }
  );

  return result.rows.map(mapRowToInstrument);
}

/**
 * Check if an instrument with the same connection params already exists.
 *
 * @param projectId - Project UUID
 * @param connectionType - Connection type
 * @param connectionParams - Connection parameters
 * @returns Existing instrument or null
 */
export async function findByConnection(
  projectId: string,
  connectionType: HILConnectionType,
  connectionParams: HILConnectionParams
): Promise<HILInstrument | null> {
  const logger = repoLogger.child({ operation: 'findByConnection', projectId });

  if (!projectId || projectId.trim().length === 0) {
    throw new ValidationError('Project ID is required', {
      operation: 'findByConnection',
    });
  }

  validateConnectionType(connectionType);

  logger.debug('Finding instrument by connection params');

  // Build a query based on connection type
  let paramKey: string;
  let paramValue: unknown;

  switch (connectionType) {
    case 'usb':
      paramKey = 'serial_number';
      paramValue = connectionParams.serial_number;
      break;
    case 'ethernet':
    case 'grpc':
    case 'modbus_tcp':
      paramKey = 'host';
      paramValue = connectionParams.host;
      break;
    case 'serial':
    case 'modbus_rtu':
      paramKey = 'serial_port';
      paramValue = connectionParams.serial_port;
      break;
    case 'gpib':
      paramKey = 'address';
      paramValue = connectionParams.address;
      break;
    default:
      return null;
  }

  if (!paramValue) {
    return null;
  }

  const result = await query<InstrumentRow>(
    `SELECT * FROM hil_instruments
     WHERE project_id = $1
       AND connection_type = $2
       AND connection_params->>'${paramKey}' = $3`,
    [projectId, connectionType, String(paramValue)],
    { operation: 'find_instrument_by_connection' }
  );

  if (result.rows.length === 0) {
    return null;
  }

  return mapRowToInstrument(result.rows[0]);
}

/**
 * Update last seen timestamp for an instrument.
 *
 * @param id - Instrument UUID
 */
export async function updateLastSeen(id: string): Promise<void> {
  const logger = repoLogger.child({ operation: 'updateLastSeen', instrumentId: id });

  if (!id || id.trim().length === 0) {
    throw new ValidationError('Instrument ID is required', {
      operation: 'updateLastSeen',
    });
  }

  await query(
    'UPDATE hil_instruments SET last_seen_at = NOW(), updated_at = NOW() WHERE id = $1',
    [id],
    { operation: 'update_instrument_last_seen' }
  );

  logger.debug('Updated instrument last seen timestamp');
}

/**
 * Get instrument statistics for a project.
 *
 * @param projectId - Project UUID
 * @returns Statistics object
 */
export async function getProjectStats(projectId: string): Promise<{
  total: number;
  connected: number;
  disconnected: number;
  error: number;
  byType: Record<HILInstrumentType, number>;
}> {
  const logger = repoLogger.child({ operation: 'getProjectStats', projectId });

  if (!projectId || projectId.trim().length === 0) {
    throw new ValidationError('Project ID is required', {
      operation: 'getProjectStats',
    });
  }

  logger.debug('Getting project instrument stats');

  const result = await query<{
    total: string;
    connected: string;
    disconnected: string;
    error: string;
  }>(
    `SELECT
       COUNT(*) as total,
       COUNT(*) FILTER (WHERE status = 'connected') as connected,
       COUNT(*) FILTER (WHERE status = 'disconnected') as disconnected,
       COUNT(*) FILTER (WHERE status = 'error') as error
     FROM hil_instruments
     WHERE project_id = $1`,
    [projectId],
    { operation: 'get_instrument_stats' }
  );

  const byTypeResult = await query<{ instrument_type: HILInstrumentType; count: string }>(
    `SELECT instrument_type, COUNT(*) as count
     FROM hil_instruments
     WHERE project_id = $1
     GROUP BY instrument_type`,
    [projectId],
    { operation: 'get_instrument_stats_by_type' }
  );

  const byType: Record<string, number> = {};
  for (const row of byTypeResult.rows) {
    byType[row.instrument_type] = parseInt(row.count, 10);
  }

  const stats = result.rows[0];
  return {
    total: parseInt(stats.total, 10),
    connected: parseInt(stats.connected, 10),
    disconnected: parseInt(stats.disconnected, 10),
    error: parseInt(stats.error, 10),
    byType: byType as Record<HILInstrumentType, number>,
  };
}

/**
 * Mark all instruments as disconnected for a project.
 * Useful when starting up or handling connection loss.
 *
 * @param projectId - Project UUID
 * @returns Number of instruments updated
 */
export async function disconnectAll(projectId: string): Promise<number> {
  const logger = repoLogger.child({ operation: 'disconnectAll', projectId });

  if (!projectId || projectId.trim().length === 0) {
    throw new ValidationError('Project ID is required', {
      operation: 'disconnectAll',
    });
  }

  logger.debug('Disconnecting all instruments');

  const result = await query(
    `UPDATE hil_instruments
     SET status = 'disconnected', updated_at = NOW()
     WHERE project_id = $1 AND status != 'disconnected'`,
    [projectId],
    { operation: 'disconnect_all_instruments' }
  );

  const count = result.rowCount || 0;
  logger.info('Disconnected all instruments', { count });

  return count;
}

// Default export
export default {
  create,
  findById,
  findByProject,
  findByType,
  findConnected,
  findByConnection,
  updateStatus,
  update,
  delete: deleteInstrument,
  updateLastSeen,
  getProjectStats,
  disconnectAll,
};
