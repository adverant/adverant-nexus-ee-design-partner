/**
 * EE Design Partner - Firmware Repository
 *
 * Database operations for firmware projects. Handles MCU configuration,
 * RTOS setup, HAL generation, driver management, and build status tracking.
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
  FirmwareProject,
  MCUTarget,
  MCUFamily,
  RTOSConfig,
  HALConfig,
  Driver,
  FirmwareTask,
  BuildConfig,
  GeneratedFile,
} from '../../types/index.js';

// ============================================================================
// Types
// ============================================================================

export interface CreateFirmwareInput {
  projectId: string;
  pcbLayoutId?: string;
  name: string;
  version?: string;
  targetMcu: MCUTarget;
  rtosConfig?: RTOSConfig;
  halConfig?: HALConfig;
  drivers?: Driver[];
  tasks?: FirmwareTask[];
  buildConfig?: BuildConfig;
}

export interface UpdateFirmwareInput {
  name?: string;
  version?: string;
  rtosConfig?: RTOSConfig;
  halConfig?: HALConfig;
  drivers?: Driver[];
  tasks?: FirmwareTask[];
  buildConfig?: BuildConfig;
  status?: 'draft' | 'generating' | 'generated' | 'building' | 'built' | 'testing' | 'tested' | 'released' | 'obsolete';
}

export interface FirmwareFilters {
  status?: string;
  mcuFamily?: MCUFamily;
  rtosType?: string;
  pcbLayoutId?: string;
}

type BuildStatus = 'not_built' | 'building' | 'success' | 'failed' | 'warnings';

interface FirmwareRow {
  id: string;
  project_id: string;
  pcb_layout_id: string | null;
  name: string;
  version: string;
  target_mcu: MCUTarget;
  mcu_family: MCUFamily;
  mcu_part: string;
  rtos_config: RTOSConfig | null;
  rtos_type: string | null;
  hal_config: HALConfig;
  peripheral_configs: object[];
  pin_mappings: object[];
  drivers: Driver[];
  tasks: FirmwareTask[];
  build_config: BuildConfig;
  toolchain: string;
  build_system: string;
  generated_files: GeneratedFile[];
  source_tree_path: string | null;
  status: string;
  build_status: BuildStatus;
  last_build_at: Date | null;
  last_build_output: string | null;
  created_at: Date;
  updated_at: Date;
}

// ============================================================================
// Repository Implementation
// ============================================================================

const repoLogger: Logger = log.child({ service: 'firmware-repository' });

/**
 * Map a database row to a FirmwareProject object.
 */
function mapRowToFirmwareProject(row: FirmwareRow): FirmwareProject {
  return {
    id: row.id,
    projectId: row.project_id,
    name: row.name,
    targetMcu: row.target_mcu,
    rtos: row.rtos_config || undefined,
    hal: row.hal_config || { type: 'vendor', peripherals: [] },
    drivers: row.drivers || [],
    tasks: row.tasks || [],
    buildConfig: row.build_config || {
      toolchain: 'gcc-arm',
      buildSystem: 'cmake',
      optimizationLevel: 'O2',
      debugSymbols: true,
      defines: {},
    },
    generatedFiles: row.generated_files || [],
    createdAt: row.created_at.toISOString(),
    updatedAt: row.updated_at.toISOString(),
  };
}

/**
 * Create a new firmware project.
 *
 * @param input - Firmware project creation data
 * @returns The created firmware project
 */
export async function create(input: CreateFirmwareInput): Promise<FirmwareProject> {
  const logger = repoLogger.child({ operation: 'create' });

  if (!input.projectId || input.projectId.trim().length === 0) {
    throw new ValidationError('Project ID is required', {
      operation: 'create',
      input,
    });
  }

  if (!input.name || input.name.trim().length === 0) {
    throw new ValidationError('Firmware project name is required', {
      operation: 'create',
      input,
    });
  }

  if (!input.targetMcu) {
    throw new ValidationError('Target MCU configuration is required', {
      operation: 'create',
      input,
    });
  }

  if (!input.targetMcu.family || !input.targetMcu.part) {
    throw new ValidationError('MCU family and part number are required', {
      operation: 'create',
      input,
    });
  }

  const validFamilies: MCUFamily[] = [
    'stm32',
    'esp32',
    'ti_tms320',
    'infineon_aurix',
    'nordic_nrf',
    'rpi_pico',
    'nxp_imxrt',
  ];

  if (!validFamilies.includes(input.targetMcu.family)) {
    throw new ValidationError(`Invalid MCU family: ${input.targetMcu.family}`, {
      operation: 'create',
      validFamilies,
    });
  }

  const defaultBuildConfig: BuildConfig = {
    toolchain: 'gcc-arm',
    buildSystem: 'cmake',
    optimizationLevel: 'O2',
    debugSymbols: true,
    defines: {},
  };

  const data: Record<string, unknown> = {
    project_id: input.projectId,
    pcb_layout_id: input.pcbLayoutId || null,
    name: input.name.trim(),
    version: input.version || '0.1.0',
    target_mcu: JSON.stringify(input.targetMcu),
    mcu_family: input.targetMcu.family,
    mcu_part: input.targetMcu.part,
    rtos_config: input.rtosConfig ? JSON.stringify(input.rtosConfig) : null,
    rtos_type: input.rtosConfig?.type || null,
    hal_config: JSON.stringify(input.halConfig || { type: 'vendor', peripherals: [] }),
    peripheral_configs: JSON.stringify([]),
    pin_mappings: JSON.stringify([]),
    drivers: JSON.stringify(input.drivers || []),
    tasks: JSON.stringify(input.tasks || []),
    build_config: JSON.stringify(input.buildConfig || defaultBuildConfig),
    toolchain: input.buildConfig?.toolchain || 'gcc-arm',
    build_system: input.buildConfig?.buildSystem || 'cmake',
    generated_files: JSON.stringify([]),
  };

  logger.debug('Creating firmware project', {
    name: input.name,
    projectId: input.projectId,
    mcuFamily: input.targetMcu.family,
    mcuPart: input.targetMcu.part,
  });

  const { text, values } = buildInsert('firmware_projects', data);
  const result = await query<FirmwareRow>(text, values, { operation: 'create_firmware' });

  if (result.rows.length === 0) {
    throw new DatabaseError(
      'Failed to create firmware project - no row returned',
      new Error('Insert returned no rows'),
      { operation: 'create_firmware' }
    );
  }

  const firmware = mapRowToFirmwareProject(result.rows[0]);
  logger.info('Firmware project created', { firmwareId: firmware.id, name: firmware.name });

  return firmware;
}

/**
 * Find a firmware project by its ID.
 *
 * @param id - Firmware project UUID
 * @returns The firmware project or null if not found
 */
export async function findById(id: string): Promise<FirmwareProject | null> {
  const logger = repoLogger.child({ operation: 'findById', firmwareId: id });

  if (!id || id.trim().length === 0) {
    throw new ValidationError('Firmware project ID is required', {
      operation: 'findById',
    });
  }

  logger.debug('Finding firmware project by ID');

  const result = await query<FirmwareRow>(
    'SELECT * FROM firmware_projects WHERE id = $1',
    [id],
    { operation: 'find_firmware_by_id' }
  );

  if (result.rows.length === 0) {
    logger.debug('Firmware project not found');
    return null;
  }

  return mapRowToFirmwareProject(result.rows[0]);
}

/**
 * Find all firmware projects for a project.
 *
 * @param projectId - Project UUID
 * @param filters - Optional filters
 * @returns Array of firmware projects
 */
export async function findByProject(
  projectId: string,
  filters?: FirmwareFilters
): Promise<FirmwareProject[]> {
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
    conditions.push(`status = $${paramIndex++}`);
    values.push(filters.status);
  }

  if (filters?.mcuFamily) {
    conditions.push(`mcu_family = $${paramIndex++}`);
    values.push(filters.mcuFamily);
  }

  if (filters?.rtosType) {
    conditions.push(`rtos_type = $${paramIndex++}`);
    values.push(filters.rtosType);
  }

  if (filters?.pcbLayoutId) {
    conditions.push(`pcb_layout_id = $${paramIndex++}`);
    values.push(filters.pcbLayoutId);
  }

  const sql = `SELECT * FROM firmware_projects WHERE ${conditions.join(' AND ')} ORDER BY created_at DESC`;

  logger.debug('Finding firmware projects by project', { filters });

  const result = await query<FirmwareRow>(sql, values, { operation: 'find_firmware_by_project' });

  logger.debug('Found firmware projects', { count: result.rows.length });

  return result.rows.map(mapRowToFirmwareProject);
}

/**
 * Update a firmware project.
 *
 * @param id - Firmware project UUID
 * @param data - Fields to update
 * @returns The updated firmware project
 */
export async function update(id: string, data: UpdateFirmwareInput): Promise<FirmwareProject> {
  const logger = repoLogger.child({ operation: 'update', firmwareId: id });

  if (!id || id.trim().length === 0) {
    throw new ValidationError('Firmware project ID is required', {
      operation: 'update',
    });
  }

  // Check if firmware exists
  const existing = await findById(id);
  if (!existing) {
    throw new NotFoundError('Firmware Project', id, { operation: 'update' });
  }

  const updateData: Record<string, unknown> = {};

  if (data.name !== undefined) {
    if (data.name.trim().length === 0) {
      throw new ValidationError('Firmware project name cannot be empty', {
        operation: 'update',
        firmwareId: id,
      });
    }
    updateData.name = data.name.trim();
  }

  if (data.version !== undefined) {
    updateData.version = data.version;
  }

  if (data.rtosConfig !== undefined) {
    updateData.rtos_config = data.rtosConfig ? JSON.stringify(data.rtosConfig) : null;
    updateData.rtos_type = data.rtosConfig?.type || null;
  }

  if (data.halConfig !== undefined) {
    updateData.hal_config = JSON.stringify(data.halConfig);
  }

  if (data.drivers !== undefined) {
    updateData.drivers = JSON.stringify(data.drivers);
  }

  if (data.tasks !== undefined) {
    updateData.tasks = JSON.stringify(data.tasks);
  }

  if (data.buildConfig !== undefined) {
    updateData.build_config = JSON.stringify(data.buildConfig);
    if (data.buildConfig.toolchain) {
      updateData.toolchain = data.buildConfig.toolchain;
    }
    if (data.buildConfig.buildSystem) {
      updateData.build_system = data.buildConfig.buildSystem;
    }
  }

  if (data.status !== undefined) {
    updateData.status = data.status;
  }

  if (Object.keys(updateData).length === 0) {
    logger.debug('No fields to update');
    return existing;
  }

  logger.debug('Updating firmware project', { fields: Object.keys(updateData) });

  const { text, values } = buildUpdate('firmware_projects', updateData, 'id', id);
  const result = await query<FirmwareRow>(text, values, { operation: 'update_firmware' });

  if (result.rows.length === 0) {
    throw new DatabaseError(
      'Failed to update firmware project - no row returned',
      new Error('Update returned no rows'),
      { operation: 'update_firmware' }
    );
  }

  const firmware = mapRowToFirmwareProject(result.rows[0]);
  logger.info('Firmware project updated', { firmwareId: firmware.id });

  return firmware;
}

/**
 * Update the source files for a firmware project.
 *
 * @param id - Firmware project UUID
 * @param files - Array of generated files
 * @returns The updated firmware project
 */
export async function updateSourceFiles(id: string, files: GeneratedFile[]): Promise<FirmwareProject> {
  const logger = repoLogger.child({ operation: 'updateSourceFiles', firmwareId: id });

  if (!id || id.trim().length === 0) {
    throw new ValidationError('Firmware project ID is required', {
      operation: 'updateSourceFiles',
    });
  }

  // Check if firmware exists
  const existing = await findById(id);
  if (!existing) {
    throw new NotFoundError('Firmware Project', id, { operation: 'updateSourceFiles' });
  }

  logger.debug('Updating source files', { fileCount: files.length });

  // Update generated files with timestamp
  const filesWithTimestamp = files.map((file) => ({
    ...file,
    generatedAt: file.generatedAt || new Date().toISOString(),
  }));

  const result = await query<FirmwareRow>(
    `UPDATE firmware_projects
     SET generated_files = $1, status = 'generated'
     WHERE id = $2
     RETURNING *`,
    [JSON.stringify(filesWithTimestamp), id],
    { operation: 'update_firmware_source_files' }
  );

  if (result.rows.length === 0) {
    throw new DatabaseError(
      'Failed to update firmware source files - no row returned',
      new Error('Update returned no rows'),
      { operation: 'update_firmware_source_files' }
    );
  }

  const firmware = mapRowToFirmwareProject(result.rows[0]);
  logger.info('Firmware source files updated', {
    firmwareId: firmware.id,
    fileCount: files.length,
  });

  return firmware;
}

/**
 * Update the build status for a firmware project.
 *
 * @param id - Firmware project UUID
 * @param status - Build status
 * @param output - Optional build output
 * @returns The updated firmware project
 */
export async function updateBuildStatus(
  id: string,
  status: BuildStatus,
  output?: string
): Promise<FirmwareProject> {
  const logger = repoLogger.child({ operation: 'updateBuildStatus', firmwareId: id });

  if (!id || id.trim().length === 0) {
    throw new ValidationError('Firmware project ID is required', {
      operation: 'updateBuildStatus',
    });
  }

  const validStatuses: BuildStatus[] = ['not_built', 'building', 'success', 'failed', 'warnings'];

  if (!validStatuses.includes(status)) {
    throw new ValidationError(`Invalid build status: ${status}`, {
      operation: 'updateBuildStatus',
      validStatuses,
    });
  }

  // Check if firmware exists
  const existing = await findById(id);
  if (!existing) {
    throw new NotFoundError('Firmware Project', id, { operation: 'updateBuildStatus' });
  }

  logger.debug('Updating build status', { status });

  // Determine new project status based on build status
  let projectStatus: string | null = null;
  if (status === 'building') {
    projectStatus = 'building';
  } else if (status === 'success' || status === 'warnings') {
    projectStatus = 'built';
  }

  const setClauses = ['build_status = $1', 'last_build_output = $2'];
  const values: unknown[] = [status, output || null];
  let paramIndex = 3;

  // Set last_build_at for terminal build states
  if (['success', 'failed', 'warnings'].includes(status)) {
    setClauses.push(`last_build_at = $${paramIndex++}`);
    values.push(new Date());
  }

  if (projectStatus) {
    setClauses.push(`status = $${paramIndex++}`);
    values.push(projectStatus);
  }

  values.push(id);

  const result = await query<FirmwareRow>(
    `UPDATE firmware_projects SET ${setClauses.join(', ')} WHERE id = $${paramIndex} RETURNING *`,
    values,
    { operation: 'update_firmware_build_status' }
  );

  if (result.rows.length === 0) {
    throw new DatabaseError(
      'Failed to update firmware build status - no row returned',
      new Error('Update returned no rows'),
      { operation: 'update_firmware_build_status' }
    );
  }

  const firmware = mapRowToFirmwareProject(result.rows[0]);
  logger.info('Firmware build status updated', {
    firmwareId: firmware.id,
    buildStatus: status,
  });

  return firmware;
}

/**
 * Update peripheral configurations for a firmware project.
 *
 * @param id - Firmware project UUID
 * @param peripherals - Array of peripheral configurations
 * @returns The updated firmware project
 */
export async function updatePeripheralConfigs(
  id: string,
  peripherals: object[]
): Promise<FirmwareProject> {
  const logger = repoLogger.child({ operation: 'updatePeripheralConfigs', firmwareId: id });

  if (!id || id.trim().length === 0) {
    throw new ValidationError('Firmware project ID is required', {
      operation: 'updatePeripheralConfigs',
    });
  }

  // Check if firmware exists
  const existing = await findById(id);
  if (!existing) {
    throw new NotFoundError('Firmware Project', id, { operation: 'updatePeripheralConfigs' });
  }

  logger.debug('Updating peripheral configs', { peripheralCount: peripherals.length });

  const result = await query<FirmwareRow>(
    'UPDATE firmware_projects SET peripheral_configs = $1 WHERE id = $2 RETURNING *',
    [JSON.stringify(peripherals), id],
    { operation: 'update_firmware_peripheral_configs' }
  );

  if (result.rows.length === 0) {
    throw new DatabaseError(
      'Failed to update firmware peripheral configs - no row returned',
      new Error('Update returned no rows'),
      { operation: 'update_firmware_peripheral_configs' }
    );
  }

  const firmware = mapRowToFirmwareProject(result.rows[0]);
  logger.info('Firmware peripheral configs updated', {
    firmwareId: firmware.id,
    peripheralCount: peripherals.length,
  });

  return firmware;
}

/**
 * Update pin mappings for a firmware project.
 *
 * @param id - Firmware project UUID
 * @param pinMappings - Array of pin mappings
 * @returns The updated firmware project
 */
export async function updatePinMappings(
  id: string,
  pinMappings: object[]
): Promise<FirmwareProject> {
  const logger = repoLogger.child({ operation: 'updatePinMappings', firmwareId: id });

  if (!id || id.trim().length === 0) {
    throw new ValidationError('Firmware project ID is required', {
      operation: 'updatePinMappings',
    });
  }

  // Check if firmware exists
  const existing = await findById(id);
  if (!existing) {
    throw new NotFoundError('Firmware Project', id, { operation: 'updatePinMappings' });
  }

  logger.debug('Updating pin mappings', { mappingCount: pinMappings.length });

  const result = await query<FirmwareRow>(
    'UPDATE firmware_projects SET pin_mappings = $1 WHERE id = $2 RETURNING *',
    [JSON.stringify(pinMappings), id],
    { operation: 'update_firmware_pin_mappings' }
  );

  if (result.rows.length === 0) {
    throw new DatabaseError(
      'Failed to update firmware pin mappings - no row returned',
      new Error('Update returned no rows'),
      { operation: 'update_firmware_pin_mappings' }
    );
  }

  const firmware = mapRowToFirmwareProject(result.rows[0]);
  logger.info('Firmware pin mappings updated', {
    firmwareId: firmware.id,
    mappingCount: pinMappings.length,
  });

  return firmware;
}

/**
 * Set the source tree path for a firmware project.
 *
 * @param id - Firmware project UUID
 * @param path - Path to the generated source tree
 * @returns The updated firmware project
 */
export async function setSourceTreePath(id: string, path: string): Promise<FirmwareProject> {
  const logger = repoLogger.child({ operation: 'setSourceTreePath', firmwareId: id });

  if (!id || id.trim().length === 0) {
    throw new ValidationError('Firmware project ID is required', {
      operation: 'setSourceTreePath',
    });
  }

  // Check if firmware exists
  const existing = await findById(id);
  if (!existing) {
    throw new NotFoundError('Firmware Project', id, { operation: 'setSourceTreePath' });
  }

  logger.debug('Setting source tree path', { path });

  const result = await query<FirmwareRow>(
    'UPDATE firmware_projects SET source_tree_path = $1 WHERE id = $2 RETURNING *',
    [path, id],
    { operation: 'set_firmware_source_tree_path' }
  );

  if (result.rows.length === 0) {
    throw new DatabaseError(
      'Failed to set firmware source tree path - no row returned',
      new Error('Update returned no rows'),
      { operation: 'set_firmware_source_tree_path' }
    );
  }

  const firmware = mapRowToFirmwareProject(result.rows[0]);
  logger.info('Firmware source tree path set', { firmwareId: firmware.id, path });

  return firmware;
}

/**
 * Delete a firmware project.
 *
 * @param id - Firmware project UUID
 */
export async function deleteFirmwareProject(id: string): Promise<void> {
  const logger = repoLogger.child({ operation: 'delete', firmwareId: id });

  if (!id || id.trim().length === 0) {
    throw new ValidationError('Firmware project ID is required', {
      operation: 'delete',
    });
  }

  // Check if firmware exists
  const existing = await findById(id);
  if (!existing) {
    throw new NotFoundError('Firmware Project', id, { operation: 'delete' });
  }

  logger.debug('Deleting firmware project');

  await query(
    'DELETE FROM firmware_projects WHERE id = $1',
    [id],
    { operation: 'delete_firmware' }
  );

  logger.info('Firmware project deleted', { firmwareId: id });
}

/**
 * Get the latest firmware project for a project.
 *
 * @param projectId - Project UUID
 * @returns The latest firmware project or null
 */
export async function getLatest(projectId: string): Promise<FirmwareProject | null> {
  const logger = repoLogger.child({ operation: 'getLatest', projectId });

  if (!projectId || projectId.trim().length === 0) {
    throw new ValidationError('Project ID is required', {
      operation: 'getLatest',
    });
  }

  logger.debug('Getting latest firmware project');

  const result = await query<FirmwareRow>(
    'SELECT * FROM firmware_projects WHERE project_id = $1 ORDER BY created_at DESC LIMIT 1',
    [projectId],
    { operation: 'get_latest_firmware' }
  );

  if (result.rows.length === 0) {
    return null;
  }

  return mapRowToFirmwareProject(result.rows[0]);
}

// Default export
export default {
  create,
  findById,
  findByProject,
  update,
  updateSourceFiles,
  updateBuildStatus,
  updatePeripheralConfigs,
  updatePinMappings,
  setSourceTreePath,
  delete: deleteFirmwareProject,
  getLatest,
};
