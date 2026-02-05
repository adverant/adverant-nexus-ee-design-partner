/**
 * EE Design Partner - Ideation Artifact Repository
 *
 * Database operations for ideation artifacts. Handles storage of design documentation,
 * specifications, architecture diagrams, and other pre-schematic design decisions.
 * These artifacts provide context for LLM-based schematic generation.
 */

import {
  query,
  buildInsert,
  buildUpdate,
  DatabaseError,
} from '../connection.js';
import { NotFoundError, ValidationError } from '../../utils/errors.js';
import { log, Logger } from '../../utils/logger.js';

// ============================================================================
// Types
// ============================================================================

export type ArtifactType =
  | 'system_overview'
  | 'executive_summary'
  | 'architecture_diagram'
  | 'schematic_spec'
  | 'power_spec'
  | 'mcu_spec'
  | 'sensing_spec'
  | 'communication_spec'
  | 'connector_spec'
  | 'interface_spec'
  | 'bom'
  | 'component_selection'
  | 'calculations'
  | 'pcb_spec'
  | 'stackup'
  | 'manufacturing_guide'
  | 'firmware_spec'
  | 'ai_integration'
  | 'test_plan'
  | 'research_paper'
  | 'patent'
  | 'compliance_doc'
  | 'custom';

export type ArtifactCategory =
  | 'architecture'
  | 'schematic'
  | 'component'
  | 'pcb'
  | 'firmware'
  | 'validation'
  | 'research';

export type ContentFormat =
  | 'markdown'
  | 'text'
  | 'csv'
  | 'json'
  | 'mermaid'
  | 'pdf'
  | 'image';

export interface IdeationArtifact {
  id: string;
  projectId: string;
  artifactType: ArtifactType;
  category: ArtifactCategory;
  name: string;
  description?: string;
  content?: string;
  contentFormat?: ContentFormat;
  filePath?: string;
  generationPrompt?: string;
  generationModel?: string;
  isGenerated: boolean;
  version: number;
  parentArtifactId?: string;
  subsystemIds: string[];
  componentRefs: string[];
  metadata: Record<string, unknown>;
  tags: string[];
  createdAt: string;
  updatedAt: string;
}

export interface CreateArtifactInput {
  projectId: string;
  artifactType: ArtifactType;
  category: ArtifactCategory;
  name: string;
  description?: string;
  content?: string;
  contentFormat?: ContentFormat;
  filePath?: string;
  generationPrompt?: string;
  generationModel?: string;
  isGenerated?: boolean;
  subsystemIds?: string[];
  componentRefs?: string[];
  metadata?: Record<string, unknown>;
  tags?: string[];
}

export interface UpdateArtifactInput {
  name?: string;
  description?: string;
  content?: string;
  contentFormat?: ContentFormat;
  filePath?: string;
  generationPrompt?: string;
  generationModel?: string;
  subsystemIds?: string[];
  componentRefs?: string[];
  metadata?: Record<string, unknown>;
  tags?: string[];
}

export interface ArtifactFilters {
  artifactType?: ArtifactType;
  category?: ArtifactCategory;
  isGenerated?: boolean;
  tags?: string[];
  subsystemId?: string;
}

interface ArtifactRow {
  id: string;
  project_id: string;
  artifact_type: ArtifactType;
  category: ArtifactCategory;
  name: string;
  description: string | null;
  content: string | null;
  content_format: ContentFormat | null;
  file_path: string | null;
  generation_prompt: string | null;
  generation_model: string | null;
  is_generated: boolean;
  version: number;
  parent_artifact_id: string | null;
  subsystem_ids: string[];
  component_refs: string[];
  metadata: Record<string, unknown>;
  tags: string[];
  created_at: Date;
  updated_at: Date;
}

// ============================================================================
// Repository Implementation
// ============================================================================

const repoLogger: Logger = log.child({ service: 'ideation-artifact-repository' });

/**
 * Map a database row to an IdeationArtifact object.
 */
function mapRowToArtifact(row: ArtifactRow): IdeationArtifact {
  return {
    id: row.id,
    projectId: row.project_id,
    artifactType: row.artifact_type,
    category: row.category,
    name: row.name,
    description: row.description || undefined,
    content: row.content || undefined,
    contentFormat: row.content_format || undefined,
    filePath: row.file_path || undefined,
    generationPrompt: row.generation_prompt || undefined,
    generationModel: row.generation_model || undefined,
    isGenerated: row.is_generated,
    version: row.version,
    parentArtifactId: row.parent_artifact_id || undefined,
    subsystemIds: row.subsystem_ids || [],
    componentRefs: row.component_refs || [],
    metadata: row.metadata || {},
    tags: row.tags || [],
    createdAt: row.created_at.toISOString(),
    updatedAt: row.updated_at.toISOString(),
  };
}

/**
 * Create a new ideation artifact.
 *
 * @param input - Artifact creation data
 * @returns The created artifact
 */
export async function create(input: CreateArtifactInput): Promise<IdeationArtifact> {
  const logger = repoLogger.child({ operation: 'create' });

  if (!input.projectId || input.projectId.trim().length === 0) {
    throw new ValidationError('Project ID is required', {
      operation: 'create',
      input,
    });
  }

  if (!input.name || input.name.trim().length === 0) {
    throw new ValidationError('Artifact name is required', {
      operation: 'create',
      input,
    });
  }

  if (!input.artifactType) {
    throw new ValidationError('Artifact type is required', {
      operation: 'create',
      input,
    });
  }

  if (!input.category) {
    throw new ValidationError('Artifact category is required', {
      operation: 'create',
      input,
    });
  }

  const data: Record<string, unknown> = {
    project_id: input.projectId,
    artifact_type: input.artifactType,
    category: input.category,
    name: input.name.trim(),
    description: input.description?.trim() || null,
    content: input.content || null,
    content_format: input.contentFormat || null,
    file_path: input.filePath?.trim() || null,
    generation_prompt: input.generationPrompt || null,
    generation_model: input.generationModel || null,
    is_generated: input.isGenerated || false,
    subsystem_ids: input.subsystemIds || [],
    component_refs: input.componentRefs || [],
    metadata: JSON.stringify(input.metadata || {}),
    tags: input.tags || [],
  };

  logger.debug('Creating ideation artifact', {
    name: input.name,
    projectId: input.projectId,
    artifactType: input.artifactType,
    category: input.category,
  });

  const { text, values } = buildInsert('ideation_artifacts', data);
  const result = await query<ArtifactRow>(text, values, { operation: 'create_ideation_artifact' });

  if (result.rows.length === 0) {
    throw new DatabaseError(
      'Failed to create ideation artifact - no row returned',
      new Error('Insert returned no rows'),
      { operation: 'create_ideation_artifact' }
    );
  }

  const artifact = mapRowToArtifact(result.rows[0]);
  logger.info('Ideation artifact created', { artifactId: artifact.id, name: artifact.name });

  return artifact;
}

/**
 * Find an artifact by its ID.
 *
 * @param id - Artifact UUID
 * @returns The artifact or null if not found
 */
export async function findById(id: string): Promise<IdeationArtifact | null> {
  const logger = repoLogger.child({ operation: 'findById', artifactId: id });

  if (!id || id.trim().length === 0) {
    throw new ValidationError('Artifact ID is required', {
      operation: 'findById',
    });
  }

  logger.debug('Finding artifact by ID');

  const result = await query<ArtifactRow>(
    'SELECT * FROM ideation_artifacts WHERE id = $1',
    [id],
    { operation: 'find_ideation_artifact_by_id' }
  );

  if (result.rows.length === 0) {
    logger.debug('Artifact not found');
    return null;
  }

  return mapRowToArtifact(result.rows[0]);
}

/**
 * Find all artifacts for a project.
 *
 * @param projectId - Project UUID
 * @param filters - Optional filters
 * @returns Array of artifacts for the project
 */
export async function findByProject(
  projectId: string,
  filters?: ArtifactFilters
): Promise<IdeationArtifact[]> {
  const logger = repoLogger.child({ operation: 'findByProject', projectId });

  if (!projectId || projectId.trim().length === 0) {
    throw new ValidationError('Project ID is required', {
      operation: 'findByProject',
    });
  }

  const conditions: string[] = ['project_id = $1'];
  const values: unknown[] = [projectId];
  let paramIndex = 2;

  if (filters?.artifactType) {
    conditions.push(`artifact_type = $${paramIndex++}`);
    values.push(filters.artifactType);
  }

  if (filters?.category) {
    conditions.push(`category = $${paramIndex++}`);
    values.push(filters.category);
  }

  if (filters?.isGenerated !== undefined) {
    conditions.push(`is_generated = $${paramIndex++}`);
    values.push(filters.isGenerated);
  }

  if (filters?.subsystemId) {
    conditions.push(`$${paramIndex++} = ANY(subsystem_ids)`);
    values.push(filters.subsystemId);
  }

  if (filters?.tags && filters.tags.length > 0) {
    conditions.push(`tags && $${paramIndex++}`);
    values.push(filters.tags);
  }

  const sql = `SELECT * FROM ideation_artifacts WHERE ${conditions.join(' AND ')} ORDER BY category, artifact_type, created_at DESC`;

  logger.debug('Finding artifacts by project', { filters });

  const result = await query<ArtifactRow>(sql, values, { operation: 'find_ideation_artifacts_by_project' });

  logger.debug('Found artifacts', { count: result.rows.length });

  return result.rows.map(mapRowToArtifact);
}

/**
 * Find artifacts by category.
 *
 * @param projectId - Project UUID
 * @param category - Artifact category
 * @returns Array of artifacts in the category
 */
export async function findByCategory(
  projectId: string,
  category: ArtifactCategory
): Promise<IdeationArtifact[]> {
  return findByProject(projectId, { category });
}

/**
 * Find artifacts by type.
 *
 * @param projectId - Project UUID
 * @param artifactType - Artifact type
 * @returns Array of artifacts of the type
 */
export async function findByType(
  projectId: string,
  artifactType: ArtifactType
): Promise<IdeationArtifact[]> {
  return findByProject(projectId, { artifactType });
}

/**
 * Update an artifact.
 *
 * @param id - Artifact UUID
 * @param data - Fields to update
 * @returns The updated artifact
 */
export async function update(id: string, data: UpdateArtifactInput): Promise<IdeationArtifact> {
  const logger = repoLogger.child({ operation: 'update', artifactId: id });

  if (!id || id.trim().length === 0) {
    throw new ValidationError('Artifact ID is required', {
      operation: 'update',
    });
  }

  // Check if artifact exists
  const existing = await findById(id);
  if (!existing) {
    throw new NotFoundError('IdeationArtifact', id, { operation: 'update' });
  }

  const updateData: Record<string, unknown> = {};

  if (data.name !== undefined) {
    if (data.name.trim().length === 0) {
      throw new ValidationError('Artifact name cannot be empty', {
        operation: 'update',
        artifactId: id,
      });
    }
    updateData.name = data.name.trim();
  }

  if (data.description !== undefined) {
    updateData.description = data.description?.trim() || null;
  }

  if (data.content !== undefined) {
    updateData.content = data.content || null;
  }

  if (data.contentFormat !== undefined) {
    updateData.content_format = data.contentFormat || null;
  }

  if (data.filePath !== undefined) {
    updateData.file_path = data.filePath?.trim() || null;
  }

  if (data.generationPrompt !== undefined) {
    updateData.generation_prompt = data.generationPrompt || null;
  }

  if (data.generationModel !== undefined) {
    updateData.generation_model = data.generationModel || null;
  }

  if (data.subsystemIds !== undefined) {
    updateData.subsystem_ids = data.subsystemIds;
  }

  if (data.componentRefs !== undefined) {
    updateData.component_refs = data.componentRefs;
  }

  if (data.metadata !== undefined) {
    updateData.metadata = JSON.stringify(data.metadata);
  }

  if (data.tags !== undefined) {
    updateData.tags = data.tags;
  }

  if (Object.keys(updateData).length === 0) {
    logger.debug('No fields to update');
    return existing;
  }

  logger.debug('Updating artifact', { fields: Object.keys(updateData) });

  const { text, values } = buildUpdate('ideation_artifacts', updateData, 'id', id);
  const result = await query<ArtifactRow>(text, values, { operation: 'update_ideation_artifact' });

  if (result.rows.length === 0) {
    throw new DatabaseError(
      'Failed to update ideation artifact - no row returned',
      new Error('Update returned no rows'),
      { operation: 'update_ideation_artifact' }
    );
  }

  const artifact = mapRowToArtifact(result.rows[0]);
  logger.info('Ideation artifact updated', { artifactId: artifact.id });

  return artifact;
}

/**
 * Update artifact content.
 *
 * @param id - Artifact UUID
 * @param content - New content
 * @param contentFormat - Content format
 * @returns The updated artifact
 */
export async function updateContent(
  id: string,
  content: string,
  contentFormat?: ContentFormat
): Promise<IdeationArtifact> {
  const logger = repoLogger.child({ operation: 'updateContent', artifactId: id });

  if (!id || id.trim().length === 0) {
    throw new ValidationError('Artifact ID is required', {
      operation: 'updateContent',
    });
  }

  // Check if artifact exists
  const existing = await findById(id);
  if (!existing) {
    throw new NotFoundError('IdeationArtifact', id, { operation: 'updateContent' });
  }

  logger.debug('Updating artifact content', { contentLength: content.length });

  const result = await query<ArtifactRow>(
    `UPDATE ideation_artifacts SET content = $1, content_format = COALESCE($2, content_format) WHERE id = $3 RETURNING *`,
    [content, contentFormat || null, id],
    { operation: 'update_ideation_artifact_content' }
  );

  if (result.rows.length === 0) {
    throw new DatabaseError(
      'Failed to update ideation artifact content - no row returned',
      new Error('Update returned no rows'),
      { operation: 'update_ideation_artifact_content' }
    );
  }

  const artifact = mapRowToArtifact(result.rows[0]);
  logger.info('Ideation artifact content updated', { artifactId: artifact.id });

  return artifact;
}

/**
 * Create a new version of an artifact.
 *
 * @param id - Artifact UUID to copy from
 * @param name - Optional new name
 * @returns The new artifact version
 */
export async function createNewVersion(id: string, name?: string): Promise<IdeationArtifact> {
  const logger = repoLogger.child({ operation: 'createNewVersion', artifactId: id });

  // Get existing artifact
  const existing = await findById(id);
  if (!existing) {
    throw new NotFoundError('IdeationArtifact', id, { operation: 'createNewVersion' });
  }

  // Get next version number for this artifact type in the project
  const versionResult = await query<{ max_version: number | null }>(
    `SELECT MAX(version) as max_version FROM ideation_artifacts
     WHERE project_id = $1 AND artifact_type = $2`,
    [existing.projectId, existing.artifactType],
    { operation: 'get_max_artifact_version' }
  );

  const newVersion = (versionResult.rows[0].max_version || 0) + 1;

  logger.debug('Creating new artifact version', { fromVersion: existing.version, newVersion });

  // Create new artifact with incremented version
  const data: Record<string, unknown> = {
    project_id: existing.projectId,
    artifact_type: existing.artifactType,
    category: existing.category,
    name: name || existing.name,
    description: existing.description || null,
    content: existing.content || null,
    content_format: existing.contentFormat || null,
    file_path: existing.filePath || null,
    generation_prompt: existing.generationPrompt || null,
    generation_model: existing.generationModel || null,
    is_generated: existing.isGenerated,
    version: newVersion,
    parent_artifact_id: existing.id,
    subsystem_ids: existing.subsystemIds,
    component_refs: existing.componentRefs,
    metadata: JSON.stringify(existing.metadata),
    tags: existing.tags,
  };

  const { text, values } = buildInsert('ideation_artifacts', data);
  const result = await query<ArtifactRow>(text, values, { operation: 'create_artifact_version' });

  if (result.rows.length === 0) {
    throw new DatabaseError(
      'Failed to create artifact version - no row returned',
      new Error('Insert returned no rows'),
      { operation: 'create_artifact_version' }
    );
  }

  const artifact = mapRowToArtifact(result.rows[0]);
  logger.info('Artifact version created', { artifactId: artifact.id, version: artifact.version });

  return artifact;
}

/**
 * Delete an artifact.
 *
 * @param id - Artifact UUID
 */
export async function deleteArtifact(id: string): Promise<void> {
  const logger = repoLogger.child({ operation: 'delete', artifactId: id });

  if (!id || id.trim().length === 0) {
    throw new ValidationError('Artifact ID is required', {
      operation: 'delete',
    });
  }

  // Check if artifact exists
  const existing = await findById(id);
  if (!existing) {
    throw new NotFoundError('IdeationArtifact', id, { operation: 'delete' });
  }

  logger.debug('Deleting artifact');

  await query(
    'DELETE FROM ideation_artifacts WHERE id = $1',
    [id],
    { operation: 'delete_ideation_artifact' }
  );

  logger.info('Ideation artifact deleted', { artifactId: id });
}

/**
 * Get artifact completeness summary for a project.
 *
 * @param projectId - Project UUID
 * @returns Completeness summary with counts by category
 */
export async function getCompleteness(projectId: string): Promise<{
  total: number;
  byCategory: Record<ArtifactCategory, number>;
  required: { present: number; total: number };
  percentComplete: number;
}> {
  const logger = repoLogger.child({ operation: 'getCompleteness', projectId });

  if (!projectId || projectId.trim().length === 0) {
    throw new ValidationError('Project ID is required', {
      operation: 'getCompleteness',
    });
  }

  logger.debug('Getting artifact completeness');

  // Get counts by category
  const result = await query<{ category: ArtifactCategory; count: string }>(
    `SELECT category, COUNT(*) as count FROM ideation_artifacts
     WHERE project_id = $1 GROUP BY category`,
    [projectId],
    { operation: 'get_artifact_completeness' }
  );

  const byCategory: Record<ArtifactCategory, number> = {
    architecture: 0,
    schematic: 0,
    component: 0,
    pcb: 0,
    firmware: 0,
    validation: 0,
    research: 0,
  };

  let total = 0;
  for (const row of result.rows) {
    byCategory[row.category] = parseInt(row.count, 10);
    total += parseInt(row.count, 10);
  }

  // Check required artifacts
  const requiredTypes: ArtifactType[] = ['system_overview', 'bom'];
  const requiredResult = await query<{ count: string }>(
    `SELECT COUNT(*) as count FROM ideation_artifacts
     WHERE project_id = $1 AND artifact_type = ANY($2)`,
    [projectId, requiredTypes],
    { operation: 'get_required_artifact_count' }
  );

  const requiredPresent = parseInt(requiredResult.rows[0].count, 10);
  const requiredTotal = requiredTypes.length;

  // Calculate percentage (weighted: required artifacts count for 50%, others for 50%)
  const requiredPercent = (requiredPresent / requiredTotal) * 50;
  const otherPercent = total > requiredPresent ? Math.min(50, ((total - requiredPresent) / 5) * 50) : 0;
  const percentComplete = Math.round(requiredPercent + otherPercent);

  return {
    total,
    byCategory,
    required: { present: requiredPresent, total: requiredTotal },
    percentComplete: Math.min(100, percentComplete),
  };
}

/**
 * Get artifacts related to a specific subsystem.
 *
 * @param projectId - Project UUID
 * @param subsystemId - Subsystem ID
 * @returns Array of related artifacts
 */
export async function findBySubsystem(
  projectId: string,
  subsystemId: string
): Promise<IdeationArtifact[]> {
  return findByProject(projectId, { subsystemId });
}

/**
 * Search artifacts by content or name.
 *
 * @param projectId - Project UUID
 * @param searchTerm - Search term
 * @returns Array of matching artifacts
 */
export async function search(
  projectId: string,
  searchTerm: string
): Promise<IdeationArtifact[]> {
  const logger = repoLogger.child({ operation: 'search', projectId });

  if (!projectId || projectId.trim().length === 0) {
    throw new ValidationError('Project ID is required', {
      operation: 'search',
    });
  }

  if (!searchTerm || searchTerm.trim().length === 0) {
    return findByProject(projectId);
  }

  const term = `%${searchTerm.trim().toLowerCase()}%`;

  logger.debug('Searching artifacts', { searchTerm });

  const result = await query<ArtifactRow>(
    `SELECT * FROM ideation_artifacts
     WHERE project_id = $1 AND (
       LOWER(name) LIKE $2 OR
       LOWER(description) LIKE $2 OR
       LOWER(content) LIKE $2
     )
     ORDER BY created_at DESC`,
    [projectId, term],
    { operation: 'search_ideation_artifacts' }
  );

  logger.debug('Search results', { count: result.rows.length });

  return result.rows.map(mapRowToArtifact);
}

// Default export
export default {
  create,
  findById,
  findByProject,
  findByCategory,
  findByType,
  findBySubsystem,
  update,
  updateContent,
  createNewVersion,
  delete: deleteArtifact,
  getCompleteness,
  search,
};
