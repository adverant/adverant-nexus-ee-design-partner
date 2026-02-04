/**
 * EE Design Partner - Schematic Repository
 *
 * Database operations for schematic designs. Handles storage of KiCad schematics,
 * netlists, BOMs, and validation results.
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
  Schematic,
  SchematicSheet,
  Component,
  Net,
  ValidationResults,
} from '../../types/index.js';

// ============================================================================
// Types
// ============================================================================

export interface CreateSchematicInput {
  projectId: string;
  name: string;
  version?: number;
  format?: 'kicad_sch' | 'eagle' | 'altium' | 'orcad';
  filePath?: string;
  kicadSch?: string;
  sheets?: SchematicSheet[];
  components?: Component[];
  nets?: Net[];
}

export interface UpdateSchematicInput {
  name?: string;
  filePath?: string;
  sheets?: SchematicSheet[];
  components?: Component[];
  nets?: Net[];
  netlist?: object;
  bom?: object;
  status?: 'draft' | 'in_review' | 'approved' | 'released' | 'obsolete';
}

export interface SchematicFilters {
  status?: string;
  version?: number;
}

interface SchematicRow {
  id: string;
  project_id: string;
  name: string;
  version: number;
  kicad_sch: string | null;
  format: 'kicad_sch' | 'eagle' | 'altium' | 'orcad';
  file_path: string | null;
  netlist: object | null;
  bom: object | null;
  sheets: SchematicSheet[];
  components: Component[];
  nets: Net[];
  validation_results: ValidationResults | null;
  erc_violations: number;
  erc_warnings: number;
  status: string;
  locked: boolean;
  locked_by: string | null;
  locked_at: Date | null;
  created_at: Date;
  updated_at: Date;
}

// ============================================================================
// Repository Implementation
// ============================================================================

const repoLogger: Logger = log.child({ service: 'schematic-repository' });

/**
 * Map a database row to a Schematic object.
 */
function mapRowToSchematic(row: SchematicRow): Schematic {
  // Calculate wire count from KiCad content (dynamic, not stored in DB)
  let wireCount = 0;
  if (row.kicad_sch) {
    const wireMatches = row.kicad_sch.match(/\(wire\s+\(pts/g);
    wireCount = wireMatches ? wireMatches.length : 0;
  }

  return {
    id: row.id,
    projectId: row.project_id,
    name: row.name,
    version: row.version,
    filePath: row.file_path || '',
    format: row.format,
    sheets: row.sheets || [],
    components: row.components || [],
    nets: row.nets || [],
    validationResults: row.validation_results || undefined,
    createdAt: row.created_at.toISOString(),
    updatedAt: row.updated_at.toISOString(),
    kicadSch: row.kicad_sch || undefined,
    wireCount: wireCount,
  };
}

/**
 * Create a new schematic.
 *
 * @param input - Schematic creation data
 * @returns The created schematic
 */
export async function create(input: CreateSchematicInput): Promise<Schematic> {
  const logger = repoLogger.child({ operation: 'create' });

  if (!input.projectId || input.projectId.trim().length === 0) {
    throw new ValidationError('Project ID is required', {
      operation: 'create',
      input,
    });
  }

  if (!input.name || input.name.trim().length === 0) {
    throw new ValidationError('Schematic name is required', {
      operation: 'create',
      input,
    });
  }

  // Determine version - get highest version for project and increment
  let version = input.version || 1;
  if (!input.version) {
    const versionResult = await query<{ max_version: number | null }>(
      'SELECT MAX(version) as max_version FROM schematics WHERE project_id = $1',
      [input.projectId],
      { operation: 'get_max_schematic_version' }
    );
    if (versionResult.rows[0].max_version !== null) {
      version = versionResult.rows[0].max_version + 1;
    }
  }

  const data: Record<string, unknown> = {
    project_id: input.projectId,
    name: input.name.trim(),
    version,
    format: input.format || 'kicad_sch',
    file_path: input.filePath?.trim() || null,
    kicad_sch: input.kicadSch || null,
    sheets: JSON.stringify(input.sheets || []),
    components: JSON.stringify(input.components || []),
    nets: JSON.stringify(input.nets || []),
  };

  logger.debug('Creating schematic', {
    name: input.name,
    projectId: input.projectId,
    version,
  });

  const { text, values } = buildInsert('schematics', data);
  const result = await query<SchematicRow>(text, values, { operation: 'create_schematic' });

  if (result.rows.length === 0) {
    throw new DatabaseError(
      'Failed to create schematic - no row returned',
      new Error('Insert returned no rows'),
      { operation: 'create_schematic' }
    );
  }

  const schematic = mapRowToSchematic(result.rows[0]);
  logger.info('Schematic created', { schematicId: schematic.id, name: schematic.name });

  return schematic;
}

/**
 * Find a schematic by its ID.
 *
 * @param id - Schematic UUID
 * @returns The schematic or null if not found
 */
export async function findById(id: string): Promise<Schematic | null> {
  const logger = repoLogger.child({ operation: 'findById', schematicId: id });

  if (!id || id.trim().length === 0) {
    throw new ValidationError('Schematic ID is required', {
      operation: 'findById',
    });
  }

  logger.debug('Finding schematic by ID');

  const result = await query<SchematicRow>(
    'SELECT * FROM schematics WHERE id = $1',
    [id],
    { operation: 'find_schematic_by_id' }
  );

  if (result.rows.length === 0) {
    logger.debug('Schematic not found');
    return null;
  }

  return mapRowToSchematic(result.rows[0]);
}

/**
 * Find all schematics for a project.
 *
 * @param projectId - Project UUID
 * @param filters - Optional filters
 * @returns Array of schematics for the project
 */
export async function findByProject(
  projectId: string,
  filters?: SchematicFilters
): Promise<Schematic[]> {
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

  if (filters?.version !== undefined) {
    conditions.push(`version = $${paramIndex++}`);
    values.push(filters.version);
  }

  const sql = `SELECT * FROM schematics WHERE ${conditions.join(' AND ')} ORDER BY version DESC`;

  logger.debug('Finding schematics by project', { filters });

  const result = await query<SchematicRow>(sql, values, { operation: 'find_schematics_by_project' });

  logger.debug('Found schematics', { count: result.rows.length });

  return result.rows.map(mapRowToSchematic);
}

/**
 * Update a schematic.
 *
 * @param id - Schematic UUID
 * @param data - Fields to update
 * @returns The updated schematic
 */
export async function update(id: string, data: UpdateSchematicInput): Promise<Schematic> {
  const logger = repoLogger.child({ operation: 'update', schematicId: id });

  if (!id || id.trim().length === 0) {
    throw new ValidationError('Schematic ID is required', {
      operation: 'update',
    });
  }

  // Check if schematic exists
  const existing = await findById(id);
  if (!existing) {
    throw new NotFoundError('Schematic', id, { operation: 'update' });
  }

  const updateData: Record<string, unknown> = {};

  if (data.name !== undefined) {
    if (data.name.trim().length === 0) {
      throw new ValidationError('Schematic name cannot be empty', {
        operation: 'update',
        schematicId: id,
      });
    }
    updateData.name = data.name.trim();
  }

  if (data.filePath !== undefined) {
    updateData.file_path = data.filePath?.trim() || null;
  }

  if (data.sheets !== undefined) {
    updateData.sheets = JSON.stringify(data.sheets);
  }

  if (data.components !== undefined) {
    updateData.components = JSON.stringify(data.components);
  }

  if (data.nets !== undefined) {
    updateData.nets = JSON.stringify(data.nets);
  }

  if (data.netlist !== undefined) {
    updateData.netlist = JSON.stringify(data.netlist);
  }

  if (data.bom !== undefined) {
    updateData.bom = JSON.stringify(data.bom);
  }

  if (data.status !== undefined) {
    updateData.status = data.status;
  }

  if (Object.keys(updateData).length === 0) {
    logger.debug('No fields to update');
    return existing;
  }

  logger.debug('Updating schematic', { fields: Object.keys(updateData) });

  const { text, values } = buildUpdate('schematics', updateData, 'id', id);
  const result = await query<SchematicRow>(text, values, { operation: 'update_schematic' });

  if (result.rows.length === 0) {
    throw new DatabaseError(
      'Failed to update schematic - no row returned',
      new Error('Update returned no rows'),
      { operation: 'update_schematic' }
    );
  }

  const schematic = mapRowToSchematic(result.rows[0]);
  logger.info('Schematic updated', { schematicId: schematic.id });

  return schematic;
}

/**
 * Update the KiCad schematic content.
 *
 * @param id - Schematic UUID
 * @param kicadSch - KiCad schematic content (S-expression format)
 * @returns The updated schematic
 */
export async function updateKicadContent(id: string, kicadSch: string): Promise<Schematic> {
  const logger = repoLogger.child({ operation: 'updateKicadContent', schematicId: id });

  if (!id || id.trim().length === 0) {
    throw new ValidationError('Schematic ID is required', {
      operation: 'updateKicadContent',
    });
  }

  // Check if schematic exists
  const existing = await findById(id);
  if (!existing) {
    throw new NotFoundError('Schematic', id, { operation: 'updateKicadContent' });
  }

  logger.debug('Updating KiCad content', { contentLength: kicadSch.length });

  const result = await query<SchematicRow>(
    'UPDATE schematics SET kicad_sch = $1 WHERE id = $2 RETURNING *',
    [kicadSch, id],
    { operation: 'update_schematic_kicad_content' }
  );

  if (result.rows.length === 0) {
    throw new DatabaseError(
      'Failed to update schematic KiCad content - no row returned',
      new Error('Update returned no rows'),
      { operation: 'update_schematic_kicad_content' }
    );
  }

  const schematic = mapRowToSchematic(result.rows[0]);
  logger.info('Schematic KiCad content updated', { schematicId: schematic.id });

  return schematic;
}

/**
 * Update validation results for a schematic.
 *
 * @param id - Schematic UUID
 * @param results - Validation results object
 * @returns The updated schematic
 */
export async function updateValidation(id: string, results: ValidationResults): Promise<Schematic> {
  const logger = repoLogger.child({ operation: 'updateValidation', schematicId: id });

  if (!id || id.trim().length === 0) {
    throw new ValidationError('Schematic ID is required', {
      operation: 'updateValidation',
    });
  }

  // Check if schematic exists
  const existing = await findById(id);
  if (!existing) {
    throw new NotFoundError('Schematic', id, { operation: 'updateValidation' });
  }

  // Count ERC violations and warnings
  let ercViolations = 0;
  let ercWarnings = 0;

  if (results.domains) {
    for (const domain of results.domains) {
      if (domain.type === 'erc') {
        for (const violation of domain.violations || []) {
          if (violation.severity === 'critical' || violation.severity === 'error') {
            ercViolations++;
          } else if (violation.severity === 'warning') {
            ercWarnings++;
          }
        }
      }
    }
  }

  logger.debug('Updating validation results', {
    passed: results.passed,
    score: results.score,
    ercViolations,
    ercWarnings,
  });

  const result = await query<SchematicRow>(
    `UPDATE schematics
     SET validation_results = $1, erc_violations = $2, erc_warnings = $3
     WHERE id = $4
     RETURNING *`,
    [JSON.stringify(results), ercViolations, ercWarnings, id],
    { operation: 'update_schematic_validation' }
  );

  if (result.rows.length === 0) {
    throw new DatabaseError(
      'Failed to update schematic validation - no row returned',
      new Error('Update returned no rows'),
      { operation: 'update_schematic_validation' }
    );
  }

  const schematic = mapRowToSchematic(result.rows[0]);
  logger.info('Schematic validation updated', {
    schematicId: schematic.id,
    passed: results.passed,
    score: results.score,
  });

  return schematic;
}

/**
 * Get the KiCad schematic content.
 *
 * @param id - Schematic UUID
 * @returns KiCad schematic content or null
 */
export async function getKicadContent(id: string): Promise<string | null> {
  const logger = repoLogger.child({ operation: 'getKicadContent', schematicId: id });

  if (!id || id.trim().length === 0) {
    throw new ValidationError('Schematic ID is required', {
      operation: 'getKicadContent',
    });
  }

  logger.debug('Getting KiCad content');

  const result = await query<{ kicad_sch: string | null }>(
    'SELECT kicad_sch FROM schematics WHERE id = $1',
    [id],
    { operation: 'get_schematic_kicad_content' }
  );

  if (result.rows.length === 0) {
    throw new NotFoundError('Schematic', id, { operation: 'getKicadContent' });
  }

  return result.rows[0].kicad_sch;
}

/**
 * Lock a schematic for editing.
 *
 * @param id - Schematic UUID
 * @param userId - User ID acquiring the lock
 * @returns The locked schematic
 */
export async function lock(id: string, userId: string): Promise<Schematic> {
  const logger = repoLogger.child({ operation: 'lock', schematicId: id });

  if (!id || id.trim().length === 0) {
    throw new ValidationError('Schematic ID is required', { operation: 'lock' });
  }

  if (!userId || userId.trim().length === 0) {
    throw new ValidationError('User ID is required', { operation: 'lock' });
  }

  // Check if schematic exists and is not already locked
  const existing = await query<SchematicRow>(
    'SELECT * FROM schematics WHERE id = $1',
    [id],
    { operation: 'check_schematic_lock' }
  );

  if (existing.rows.length === 0) {
    throw new NotFoundError('Schematic', id, { operation: 'lock' });
  }

  const row = existing.rows[0];
  if (row.locked && row.locked_by !== userId) {
    throw new ValidationError(`Schematic is already locked by ${row.locked_by}`, {
      operation: 'lock',
      schematicId: id,
      lockedBy: row.locked_by,
    });
  }

  logger.debug('Locking schematic', { userId });

  const result = await query<SchematicRow>(
    'UPDATE schematics SET locked = true, locked_by = $1, locked_at = NOW() WHERE id = $2 RETURNING *',
    [userId, id],
    { operation: 'lock_schematic' }
  );

  const schematic = mapRowToSchematic(result.rows[0]);
  logger.info('Schematic locked', { schematicId: schematic.id, lockedBy: userId });

  return schematic;
}

/**
 * Unlock a schematic.
 *
 * @param id - Schematic UUID
 * @param userId - User ID releasing the lock (must match locker)
 * @returns The unlocked schematic
 */
export async function unlock(id: string, userId: string): Promise<Schematic> {
  const logger = repoLogger.child({ operation: 'unlock', schematicId: id });

  if (!id || id.trim().length === 0) {
    throw new ValidationError('Schematic ID is required', { operation: 'unlock' });
  }

  // Check if schematic exists
  const existing = await query<SchematicRow>(
    'SELECT * FROM schematics WHERE id = $1',
    [id],
    { operation: 'check_schematic_unlock' }
  );

  if (existing.rows.length === 0) {
    throw new NotFoundError('Schematic', id, { operation: 'unlock' });
  }

  const row = existing.rows[0];
  if (row.locked && row.locked_by !== userId) {
    throw new ValidationError('Cannot unlock schematic - locked by different user', {
      operation: 'unlock',
      schematicId: id,
      lockedBy: row.locked_by,
    });
  }

  logger.debug('Unlocking schematic');

  const result = await query<SchematicRow>(
    'UPDATE schematics SET locked = false, locked_by = NULL, locked_at = NULL WHERE id = $1 RETURNING *',
    [id],
    { operation: 'unlock_schematic' }
  );

  const schematic = mapRowToSchematic(result.rows[0]);
  logger.info('Schematic unlocked', { schematicId: schematic.id });

  return schematic;
}

/**
 * Delete a schematic.
 *
 * @param id - Schematic UUID
 */
export async function deleteSchematic(id: string): Promise<void> {
  const logger = repoLogger.child({ operation: 'delete', schematicId: id });

  if (!id || id.trim().length === 0) {
    throw new ValidationError('Schematic ID is required', {
      operation: 'delete',
    });
  }

  // Check if schematic exists
  const existing = await findById(id);
  if (!existing) {
    throw new NotFoundError('Schematic', id, { operation: 'delete' });
  }

  logger.debug('Deleting schematic');

  await query(
    'DELETE FROM schematics WHERE id = $1',
    [id],
    { operation: 'delete_schematic' }
  );

  logger.info('Schematic deleted', { schematicId: id });
}

/**
 * Get the latest version of a schematic for a project.
 *
 * @param projectId - Project UUID
 * @returns The latest schematic or null
 */
export async function getLatestVersion(projectId: string): Promise<Schematic | null> {
  const logger = repoLogger.child({ operation: 'getLatestVersion', projectId });

  if (!projectId || projectId.trim().length === 0) {
    throw new ValidationError('Project ID is required', {
      operation: 'getLatestVersion',
    });
  }

  logger.debug('Getting latest schematic version');

  const result = await query<SchematicRow>(
    'SELECT * FROM schematics WHERE project_id = $1 ORDER BY version DESC LIMIT 1',
    [projectId],
    { operation: 'get_latest_schematic_version' }
  );

  if (result.rows.length === 0) {
    return null;
  }

  return mapRowToSchematic(result.rows[0]);
}

/**
 * Create a new version of a schematic by copying an existing one.
 *
 * @param id - Schematic UUID to copy from
 * @param name - Optional new name for the version
 * @returns The new schematic version
 */
export async function createNewVersion(id: string, name?: string): Promise<Schematic> {
  const logger = repoLogger.child({ operation: 'createNewVersion', schematicId: id });

  // Get existing schematic
  const existing = await findById(id);
  if (!existing) {
    throw new NotFoundError('Schematic', id, { operation: 'createNewVersion' });
  }

  // Get KiCad content
  const kicadSch = await getKicadContent(id);

  // Get next version number
  const versionResult = await query<{ max_version: number | null }>(
    'SELECT MAX(version) as max_version FROM schematics WHERE project_id = $1',
    [existing.projectId],
    { operation: 'get_max_schematic_version' }
  );

  const newVersion = (versionResult.rows[0].max_version || 0) + 1;

  logger.debug('Creating new schematic version', { fromVersion: existing.version, newVersion });

  // Create new schematic with incremented version
  return create({
    projectId: existing.projectId,
    name: name || existing.name,
    version: newVersion,
    format: existing.format,
    filePath: existing.filePath,
    kicadSch: kicadSch || undefined,
    sheets: existing.sheets,
    components: existing.components,
    nets: existing.nets,
  });
}

// Default export
export default {
  create,
  findById,
  findByProject,
  update,
  updateKicadContent,
  updateValidation,
  getKicadContent,
  lock,
  unlock,
  delete: deleteSchematic,
  getLatestVersion,
  createNewVersion,
};
