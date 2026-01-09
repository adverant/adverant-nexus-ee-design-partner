/**
 * EE Design Partner - PCB Layout Repository
 *
 * Database operations for PCB layouts. Handles storage of KiCad PCB files,
 * DRC results, MAPOS optimization data, and layout scoring.
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
  PCBLayout,
  BoardOutline,
  LayerStackup,
  PlacedComponent,
  Trace,
  Via,
  CopperZone,
  ValidationResults,
} from '../../types/index.js';

// ============================================================================
// Types
// ============================================================================

export interface CreatePCBLayoutInput {
  projectId: string;
  schematicId?: string;
  name: string;
  version?: number;
  filePath?: string;
  kicadPcb?: string;
  layerCount?: number;
  boardOutline?: BoardOutline;
  stackup?: LayerStackup;
  components?: PlacedComponent[];
  traces?: Trace[];
  vias?: Via[];
  zones?: CopperZone[];
}

export interface UpdatePCBLayoutInput {
  name?: string;
  filePath?: string;
  layerCount?: number;
  boardOutline?: BoardOutline;
  stackup?: LayerStackup;
  components?: PlacedComponent[];
  traces?: Trace[];
  vias?: Via[];
  zones?: CopperZone[];
  status?: 'draft' | 'in_progress' | 'optimizing' | 'in_review' | 'approved' | 'released' | 'obsolete';
}

export interface MaposIterationInput {
  iterationNumber: number;
  agentStrategy: string;
  score: number;
  drcViolations: number;
  improvementDelta?: number;
  changes?: object[];
  validationSnapshot?: object;
  durationMs?: number;
}

export interface DRCResults {
  passed: boolean;
  totalViolations: number;
  violations: Array<{
    id: string;
    code: string;
    severity: string;
    message: string;
    location?: object;
  }>;
  warnings: number;
  timestamp: string;
}

export interface PCBLayoutFilters {
  status?: string;
  version?: number;
  schematicId?: string;
}

interface PCBLayoutRow {
  id: string;
  project_id: string;
  schematic_id: string | null;
  name: string;
  version: number;
  kicad_pcb: string | null;
  file_path: string | null;
  board_outline: BoardOutline | null;
  stackup: LayerStackup | null;
  layer_count: number;
  components: PlacedComponent[];
  traces: Trace[];
  vias: Via[];
  zones: CopperZone[];
  drc_results: DRCResults | null;
  drc_violations: number;
  drc_warnings: number;
  mapos_score: number | null;
  mapos_iterations: number;
  mapos_config: object | null;
  winning_agent: string | null;
  overall_score: number | null;
  thermal_score: number | null;
  emi_score: number | null;
  dfm_score: number | null;
  si_score: number | null;
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

const repoLogger: Logger = log.child({ service: 'pcb-repository' });

/**
 * Map a database row to a PCBLayout object.
 */
function mapRowToPCBLayout(row: PCBLayoutRow): PCBLayout {
  return {
    id: row.id,
    projectId: row.project_id,
    schematicId: row.schematic_id || '',
    version: row.version,
    filePath: row.file_path || '',
    boardOutline: row.board_outline || {
      width: 0,
      height: 0,
      shape: 'rectangular',
      keepoutZones: [],
    },
    stackup: row.stackup || {
      totalThickness: 1.6,
      layers: [],
    },
    components: row.components || [],
    traces: row.traces || [],
    vias: row.vias || [],
    zones: row.zones || [],
    validationResults: row.drc_results
      ? {
          passed: row.drc_results.passed,
          score: row.overall_score || 0,
          timestamp: row.drc_results.timestamp,
          domains: [],
        }
      : undefined,
    score: row.overall_score || 0,
    createdAt: row.created_at.toISOString(),
    updatedAt: row.updated_at.toISOString(),
  };
}

/**
 * Create a new PCB layout.
 *
 * @param input - PCB layout creation data
 * @returns The created PCB layout
 */
export async function create(input: CreatePCBLayoutInput): Promise<PCBLayout> {
  const logger = repoLogger.child({ operation: 'create' });

  if (!input.projectId || input.projectId.trim().length === 0) {
    throw new ValidationError('Project ID is required', {
      operation: 'create',
      input,
    });
  }

  if (!input.name || input.name.trim().length === 0) {
    throw new ValidationError('PCB layout name is required', {
      operation: 'create',
      input,
    });
  }

  // Determine version - get highest version for project and increment
  let version = input.version || 1;
  if (!input.version) {
    const versionResult = await query<{ max_version: number | null }>(
      'SELECT MAX(version) as max_version FROM pcb_layouts WHERE project_id = $1',
      [input.projectId],
      { operation: 'get_max_pcb_version' }
    );
    if (versionResult.rows[0].max_version !== null) {
      version = versionResult.rows[0].max_version + 1;
    }
  }

  const data: Record<string, unknown> = {
    project_id: input.projectId,
    schematic_id: input.schematicId || null,
    name: input.name.trim(),
    version,
    file_path: input.filePath?.trim() || null,
    kicad_pcb: input.kicadPcb || null,
    layer_count: input.layerCount || 2,
    board_outline: input.boardOutline ? JSON.stringify(input.boardOutline) : null,
    stackup: input.stackup ? JSON.stringify(input.stackup) : null,
    components: JSON.stringify(input.components || []),
    traces: JSON.stringify(input.traces || []),
    vias: JSON.stringify(input.vias || []),
    zones: JSON.stringify(input.zones || []),
  };

  logger.debug('Creating PCB layout', {
    name: input.name,
    projectId: input.projectId,
    version,
  });

  const { text, values } = buildInsert('pcb_layouts', data);
  const result = await query<PCBLayoutRow>(text, values, { operation: 'create_pcb_layout' });

  if (result.rows.length === 0) {
    throw new DatabaseError(
      'Failed to create PCB layout - no row returned',
      new Error('Insert returned no rows'),
      { operation: 'create_pcb_layout' }
    );
  }

  const layout = mapRowToPCBLayout(result.rows[0]);
  logger.info('PCB layout created', { layoutId: layout.id, name: input.name });

  return layout;
}

/**
 * Find a PCB layout by its ID.
 *
 * @param id - PCB layout UUID
 * @returns The PCB layout or null if not found
 */
export async function findById(id: string): Promise<PCBLayout | null> {
  const logger = repoLogger.child({ operation: 'findById', layoutId: id });

  if (!id || id.trim().length === 0) {
    throw new ValidationError('PCB layout ID is required', {
      operation: 'findById',
    });
  }

  logger.debug('Finding PCB layout by ID');

  const result = await query<PCBLayoutRow>(
    'SELECT * FROM pcb_layouts WHERE id = $1',
    [id],
    { operation: 'find_pcb_layout_by_id' }
  );

  if (result.rows.length === 0) {
    logger.debug('PCB layout not found');
    return null;
  }

  return mapRowToPCBLayout(result.rows[0]);
}

/**
 * Find all PCB layouts for a project.
 *
 * @param projectId - Project UUID
 * @param filters - Optional filters
 * @returns Array of PCB layouts for the project
 */
export async function findByProject(
  projectId: string,
  filters?: PCBLayoutFilters
): Promise<PCBLayout[]> {
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

  if (filters?.schematicId) {
    conditions.push(`schematic_id = $${paramIndex++}`);
    values.push(filters.schematicId);
  }

  const sql = `SELECT * FROM pcb_layouts WHERE ${conditions.join(' AND ')} ORDER BY version DESC`;

  logger.debug('Finding PCB layouts by project', { filters });

  const result = await query<PCBLayoutRow>(sql, values, { operation: 'find_pcb_layouts_by_project' });

  logger.debug('Found PCB layouts', { count: result.rows.length });

  return result.rows.map(mapRowToPCBLayout);
}

/**
 * Update a PCB layout.
 *
 * @param id - PCB layout UUID
 * @param data - Fields to update
 * @returns The updated PCB layout
 */
export async function update(id: string, data: UpdatePCBLayoutInput): Promise<PCBLayout> {
  const logger = repoLogger.child({ operation: 'update', layoutId: id });

  if (!id || id.trim().length === 0) {
    throw new ValidationError('PCB layout ID is required', {
      operation: 'update',
    });
  }

  // Check if layout exists
  const existing = await findById(id);
  if (!existing) {
    throw new NotFoundError('PCB Layout', id, { operation: 'update' });
  }

  const updateData: Record<string, unknown> = {};

  if (data.name !== undefined) {
    if (data.name.trim().length === 0) {
      throw new ValidationError('PCB layout name cannot be empty', {
        operation: 'update',
        layoutId: id,
      });
    }
    updateData.name = data.name.trim();
  }

  if (data.filePath !== undefined) {
    updateData.file_path = data.filePath?.trim() || null;
  }

  if (data.layerCount !== undefined) {
    updateData.layer_count = data.layerCount;
  }

  if (data.boardOutline !== undefined) {
    updateData.board_outline = JSON.stringify(data.boardOutline);
  }

  if (data.stackup !== undefined) {
    updateData.stackup = JSON.stringify(data.stackup);
  }

  if (data.components !== undefined) {
    updateData.components = JSON.stringify(data.components);
  }

  if (data.traces !== undefined) {
    updateData.traces = JSON.stringify(data.traces);
  }

  if (data.vias !== undefined) {
    updateData.vias = JSON.stringify(data.vias);
  }

  if (data.zones !== undefined) {
    updateData.zones = JSON.stringify(data.zones);
  }

  if (data.status !== undefined) {
    updateData.status = data.status;
  }

  if (Object.keys(updateData).length === 0) {
    logger.debug('No fields to update');
    return existing;
  }

  logger.debug('Updating PCB layout', { fields: Object.keys(updateData) });

  const { text, values } = buildUpdate('pcb_layouts', updateData, 'id', id);
  const result = await query<PCBLayoutRow>(text, values, { operation: 'update_pcb_layout' });

  if (result.rows.length === 0) {
    throw new DatabaseError(
      'Failed to update PCB layout - no row returned',
      new Error('Update returned no rows'),
      { operation: 'update_pcb_layout' }
    );
  }

  const layout = mapRowToPCBLayout(result.rows[0]);
  logger.info('PCB layout updated', { layoutId: layout.id });

  return layout;
}

/**
 * Update the KiCad PCB content.
 *
 * @param id - PCB layout UUID
 * @param kicadPcb - KiCad PCB content (S-expression format)
 * @returns The updated PCB layout
 */
export async function updateKicadContent(id: string, kicadPcb: string): Promise<PCBLayout> {
  const logger = repoLogger.child({ operation: 'updateKicadContent', layoutId: id });

  if (!id || id.trim().length === 0) {
    throw new ValidationError('PCB layout ID is required', {
      operation: 'updateKicadContent',
    });
  }

  // Check if layout exists
  const existing = await findById(id);
  if (!existing) {
    throw new NotFoundError('PCB Layout', id, { operation: 'updateKicadContent' });
  }

  logger.debug('Updating KiCad content', { contentLength: kicadPcb.length });

  const result = await query<PCBLayoutRow>(
    'UPDATE pcb_layouts SET kicad_pcb = $1 WHERE id = $2 RETURNING *',
    [kicadPcb, id],
    { operation: 'update_pcb_kicad_content' }
  );

  if (result.rows.length === 0) {
    throw new DatabaseError(
      'Failed to update PCB KiCad content - no row returned',
      new Error('Update returned no rows'),
      { operation: 'update_pcb_kicad_content' }
    );
  }

  const layout = mapRowToPCBLayout(result.rows[0]);
  logger.info('PCB layout KiCad content updated', { layoutId: layout.id });

  return layout;
}

/**
 * Update DRC results for a PCB layout.
 *
 * @param id - PCB layout UUID
 * @param drcResults - DRC results object
 * @returns The updated PCB layout
 */
export async function updateDrcResults(id: string, drcResults: DRCResults): Promise<PCBLayout> {
  const logger = repoLogger.child({ operation: 'updateDrcResults', layoutId: id });

  if (!id || id.trim().length === 0) {
    throw new ValidationError('PCB layout ID is required', {
      operation: 'updateDrcResults',
    });
  }

  // Check if layout exists
  const existing = await findById(id);
  if (!existing) {
    throw new NotFoundError('PCB Layout', id, { operation: 'updateDrcResults' });
  }

  const drcViolations = drcResults.totalViolations || 0;
  const drcWarnings = drcResults.warnings || 0;

  logger.debug('Updating DRC results', {
    passed: drcResults.passed,
    violations: drcViolations,
    warnings: drcWarnings,
  });

  const result = await query<PCBLayoutRow>(
    `UPDATE pcb_layouts
     SET drc_results = $1, drc_violations = $2, drc_warnings = $3
     WHERE id = $4
     RETURNING *`,
    [JSON.stringify(drcResults), drcViolations, drcWarnings, id],
    { operation: 'update_pcb_drc_results' }
  );

  if (result.rows.length === 0) {
    throw new DatabaseError(
      'Failed to update PCB DRC results - no row returned',
      new Error('Update returned no rows'),
      { operation: 'update_pcb_drc_results' }
    );
  }

  const layout = mapRowToPCBLayout(result.rows[0]);
  logger.info('PCB layout DRC results updated', {
    layoutId: layout.id,
    passed: drcResults.passed,
    violations: drcViolations,
  });

  return layout;
}

/**
 * Add a MAPOS optimization iteration record.
 *
 * @param layoutId - PCB layout UUID
 * @param iteration - MAPOS iteration data
 */
export async function addMaposIteration(layoutId: string, iteration: MaposIterationInput): Promise<void> {
  const logger = repoLogger.child({ operation: 'addMaposIteration', layoutId });

  if (!layoutId || layoutId.trim().length === 0) {
    throw new ValidationError('PCB layout ID is required', {
      operation: 'addMaposIteration',
    });
  }

  // Check if layout exists
  const existing = await findById(layoutId);
  if (!existing) {
    throw new NotFoundError('PCB Layout', layoutId, { operation: 'addMaposIteration' });
  }

  logger.debug('Adding MAPOS iteration', {
    iterationNumber: iteration.iterationNumber,
    agentStrategy: iteration.agentStrategy,
    score: iteration.score,
  });

  // Use a transaction to update both tables
  await withTransaction(async (client) => {
    // Insert iteration record
    const iterationData: Record<string, unknown> = {
      pcb_layout_id: layoutId,
      iteration_number: iteration.iterationNumber,
      agent_strategy: iteration.agentStrategy,
      score: iteration.score,
      drc_violations: iteration.drcViolations,
      improvement_delta: iteration.improvementDelta || null,
      changes: JSON.stringify(iteration.changes || []),
      validation_snapshot: iteration.validationSnapshot ? JSON.stringify(iteration.validationSnapshot) : null,
      duration_ms: iteration.durationMs || null,
    };

    const { text: insertText, values: insertValues } = buildInsert('mapos_iterations', iterationData);
    await clientQuery(client, insertText, insertValues);

    // Update layout with latest MAPOS stats
    await clientQuery(
      client,
      `UPDATE pcb_layouts
       SET mapos_iterations = mapos_iterations + 1,
           mapos_score = $1,
           winning_agent = $2
       WHERE id = $3`,
      [iteration.score, iteration.agentStrategy, layoutId]
    );
  }, { operation: 'add_mapos_iteration' });

  logger.info('MAPOS iteration added', {
    layoutId,
    iterationNumber: iteration.iterationNumber,
    score: iteration.score,
  });
}

/**
 * Get MAPOS iteration history for a PCB layout.
 *
 * @param layoutId - PCB layout UUID
 * @returns Array of MAPOS iterations
 */
export async function getMaposIterations(layoutId: string): Promise<MaposIterationInput[]> {
  const logger = repoLogger.child({ operation: 'getMaposIterations', layoutId });

  if (!layoutId || layoutId.trim().length === 0) {
    throw new ValidationError('PCB layout ID is required', {
      operation: 'getMaposIterations',
    });
  }

  logger.debug('Getting MAPOS iterations');

  interface MaposIterationRow {
    iteration_number: number;
    agent_strategy: string;
    score: number;
    drc_violations: number;
    improvement_delta: number | null;
    changes: object[];
    validation_snapshot: object | null;
    duration_ms: number | null;
  }

  const result = await query<MaposIterationRow>(
    `SELECT * FROM mapos_iterations
     WHERE pcb_layout_id = $1
     ORDER BY iteration_number ASC`,
    [layoutId],
    { operation: 'get_mapos_iterations' }
  );

  return result.rows.map((row) => ({
    iterationNumber: row.iteration_number,
    agentStrategy: row.agent_strategy,
    score: row.score,
    drcViolations: row.drc_violations,
    improvementDelta: row.improvement_delta || undefined,
    changes: row.changes || [],
    validationSnapshot: row.validation_snapshot || undefined,
    durationMs: row.duration_ms || undefined,
  }));
}

/**
 * Update MAPOS configuration for a layout.
 *
 * @param id - PCB layout UUID
 * @param config - MAPOS configuration object
 * @returns The updated PCB layout
 */
export async function updateMaposConfig(id: string, config: object): Promise<PCBLayout> {
  const logger = repoLogger.child({ operation: 'updateMaposConfig', layoutId: id });

  if (!id || id.trim().length === 0) {
    throw new ValidationError('PCB layout ID is required', {
      operation: 'updateMaposConfig',
    });
  }

  // Check if layout exists
  const existing = await findById(id);
  if (!existing) {
    throw new NotFoundError('PCB Layout', id, { operation: 'updateMaposConfig' });
  }

  logger.debug('Updating MAPOS config');

  const result = await query<PCBLayoutRow>(
    'UPDATE pcb_layouts SET mapos_config = $1, status = $2 WHERE id = $3 RETURNING *',
    [JSON.stringify(config), 'optimizing', id],
    { operation: 'update_mapos_config' }
  );

  if (result.rows.length === 0) {
    throw new DatabaseError(
      'Failed to update MAPOS config - no row returned',
      new Error('Update returned no rows'),
      { operation: 'update_mapos_config' }
    );
  }

  const layout = mapRowToPCBLayout(result.rows[0]);
  logger.info('MAPOS config updated', { layoutId: layout.id });

  return layout;
}

/**
 * Update layout scores (thermal, EMI, DFM, SI, overall).
 *
 * @param id - PCB layout UUID
 * @param scores - Score values to update
 * @returns The updated PCB layout
 */
export async function updateScores(
  id: string,
  scores: {
    overall?: number;
    thermal?: number;
    emi?: number;
    dfm?: number;
    si?: number;
  }
): Promise<PCBLayout> {
  const logger = repoLogger.child({ operation: 'updateScores', layoutId: id });

  if (!id || id.trim().length === 0) {
    throw new ValidationError('PCB layout ID is required', {
      operation: 'updateScores',
    });
  }

  // Check if layout exists
  const existing = await findById(id);
  if (!existing) {
    throw new NotFoundError('PCB Layout', id, { operation: 'updateScores' });
  }

  const setClauses: string[] = [];
  const values: unknown[] = [];
  let paramIndex = 1;

  if (scores.overall !== undefined) {
    setClauses.push(`overall_score = $${paramIndex++}`);
    values.push(scores.overall);
  }
  if (scores.thermal !== undefined) {
    setClauses.push(`thermal_score = $${paramIndex++}`);
    values.push(scores.thermal);
  }
  if (scores.emi !== undefined) {
    setClauses.push(`emi_score = $${paramIndex++}`);
    values.push(scores.emi);
  }
  if (scores.dfm !== undefined) {
    setClauses.push(`dfm_score = $${paramIndex++}`);
    values.push(scores.dfm);
  }
  if (scores.si !== undefined) {
    setClauses.push(`si_score = $${paramIndex++}`);
    values.push(scores.si);
  }

  if (setClauses.length === 0) {
    return existing;
  }

  values.push(id);
  const sql = `UPDATE pcb_layouts SET ${setClauses.join(', ')} WHERE id = $${paramIndex} RETURNING *`;

  logger.debug('Updating scores', { scores });

  const result = await query<PCBLayoutRow>(sql, values, { operation: 'update_pcb_scores' });

  if (result.rows.length === 0) {
    throw new DatabaseError(
      'Failed to update PCB scores - no row returned',
      new Error('Update returned no rows'),
      { operation: 'update_pcb_scores' }
    );
  }

  const layout = mapRowToPCBLayout(result.rows[0]);
  logger.info('PCB layout scores updated', { layoutId: layout.id, scores });

  return layout;
}

/**
 * Get the KiCad PCB content.
 *
 * @param id - PCB layout UUID
 * @returns KiCad PCB content or null
 */
export async function getKicadContent(id: string): Promise<string | null> {
  const logger = repoLogger.child({ operation: 'getKicadContent', layoutId: id });

  if (!id || id.trim().length === 0) {
    throw new ValidationError('PCB layout ID is required', {
      operation: 'getKicadContent',
    });
  }

  logger.debug('Getting KiCad content');

  const result = await query<{ kicad_pcb: string | null }>(
    'SELECT kicad_pcb FROM pcb_layouts WHERE id = $1',
    [id],
    { operation: 'get_pcb_kicad_content' }
  );

  if (result.rows.length === 0) {
    throw new NotFoundError('PCB Layout', id, { operation: 'getKicadContent' });
  }

  return result.rows[0].kicad_pcb;
}

/**
 * Delete a PCB layout.
 *
 * @param id - PCB layout UUID
 */
export async function deletePCBLayout(id: string): Promise<void> {
  const logger = repoLogger.child({ operation: 'delete', layoutId: id });

  if (!id || id.trim().length === 0) {
    throw new ValidationError('PCB layout ID is required', {
      operation: 'delete',
    });
  }

  // Check if layout exists
  const existing = await findById(id);
  if (!existing) {
    throw new NotFoundError('PCB Layout', id, { operation: 'delete' });
  }

  logger.debug('Deleting PCB layout');

  await query(
    'DELETE FROM pcb_layouts WHERE id = $1',
    [id],
    { operation: 'delete_pcb_layout' }
  );

  logger.info('PCB layout deleted', { layoutId: id });
}

/**
 * Get the latest version of a PCB layout for a project.
 *
 * @param projectId - Project UUID
 * @returns The latest PCB layout or null
 */
export async function getLatestVersion(projectId: string): Promise<PCBLayout | null> {
  const logger = repoLogger.child({ operation: 'getLatestVersion', projectId });

  if (!projectId || projectId.trim().length === 0) {
    throw new ValidationError('Project ID is required', {
      operation: 'getLatestVersion',
    });
  }

  logger.debug('Getting latest PCB layout version');

  const result = await query<PCBLayoutRow>(
    'SELECT * FROM pcb_layouts WHERE project_id = $1 ORDER BY version DESC LIMIT 1',
    [projectId],
    { operation: 'get_latest_pcb_version' }
  );

  if (result.rows.length === 0) {
    return null;
  }

  return mapRowToPCBLayout(result.rows[0]);
}

// Default export
export default {
  create,
  findById,
  findByProject,
  update,
  updateKicadContent,
  updateDrcResults,
  addMaposIteration,
  getMaposIterations,
  updateMaposConfig,
  updateScores,
  getKicadContent,
  delete: deletePCBLayout,
  getLatestVersion,
};
