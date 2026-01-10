/**
 * EE Design Partner - Project Repository
 *
 * Database operations for EE Design projects. Handles CRUD operations,
 * status management, and phase transitions.
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
  EEProject,
  ProjectPhase,
  ProjectStatus,
  ProjectMetadata,
} from '../../types/index.js';
import type { PhaseStatus } from '../../types/project-types.js';

// ============================================================================
// Types
// ============================================================================

export interface CreateProjectInput {
  name: string;
  description?: string;
  repositoryUrl?: string;
  projectType?: string;
  ownerId: string;
  organizationId?: string;
  collaborators?: string[];
  metadata?: Partial<ProjectMetadata>;
  phaseConfig?: Record<string, unknown>;
}

export interface UpdateProjectInput {
  name?: string;
  description?: string;
  repositoryUrl?: string;
  projectType?: string;
  collaborators?: string[];
  metadata?: Partial<ProjectMetadata>;
  phaseConfig?: Record<string, unknown>;
}

export interface ProjectFilters {
  type?: string;
  status?: ProjectStatus;
  ownerId?: string;
  organizationId?: string;
  phase?: ProjectPhase;
  limit?: number;
  offset?: number;
}

interface ProjectRow {
  id: string;
  name: string;
  description: string | null;
  repository_url: string | null;
  project_type: string;
  phase: ProjectPhase;
  status: ProjectStatus;
  phase_config: Record<string, unknown>;
  metadata: ProjectMetadata;
  owner_id: string;
  organization_id: string | null;
  collaborators: string[];
  created_at: Date;
  updated_at: Date;
  archived_at: Date | null;
}

// ============================================================================
// Repository Implementation
// ============================================================================

const repoLogger: Logger = log.child({ service: 'project-repository' });

/**
 * Calculate completion percentage based on project phase.
 * Each phase represents ~10% of total project completion.
 */
function calculateCompletion(phase: ProjectPhase, status: ProjectStatus): number {
  if (status === 'completed') return 100;
  if (status === 'cancelled') return 0;

  const phaseOrder: ProjectPhase[] = [
    'ideation',
    'architecture',
    'schematic',
    'simulation',
    'pcb_layout',
    'manufacturing',
    'firmware',
    'testing',
    'production',
    'field_support',
  ];

  const phaseIndex = phaseOrder.indexOf(phase);
  if (phaseIndex === -1) return 0;

  // Each completed phase is 10%, current phase counts as half
  return phaseIndex * 10 + 5;
}

/**
 * Map a database row to an EEProject object.
 */
function mapRowToProject(row: ProjectRow): EEProject {
  const completion = calculateCompletion(row.phase, row.status);

  return {
    id: row.id,
    name: row.name,
    description: row.description || '',
    repositoryUrl: row.repository_url || '',
    type: row.project_type || 'power_electronics',
    phase: row.phase,
    status: row.status,
    owner: row.owner_id,
    collaborators: row.collaborators || [],
    completion,
    createdAt: row.created_at.toISOString(),
    updatedAt: row.updated_at.toISOString(),
    metadata: {
      ...row.metadata,
      tags: row.metadata?.tags || [],
    },
  };
}

/**
 * Create a new EE Design project.
 *
 * @param input - Project creation data
 * @returns The created project
 */
export async function create(input: CreateProjectInput): Promise<EEProject> {
  const logger = repoLogger.child({ operation: 'create' });

  if (!input.name || input.name.trim().length === 0) {
    throw new ValidationError('Project name is required', {
      operation: 'create',
      input,
    });
  }

  if (!input.ownerId || input.ownerId.trim().length === 0) {
    throw new ValidationError('Owner ID is required', {
      operation: 'create',
      input,
    });
  }

  const data: Record<string, unknown> = {
    name: input.name.trim(),
    description: input.description?.trim() || null,
    repository_url: input.repositoryUrl?.trim() || null,
    project_type: input.projectType || 'hardware',
    owner_id: input.ownerId,
    organization_id: input.organizationId || null,
    collaborators: JSON.stringify(input.collaborators || []),
    metadata: JSON.stringify(input.metadata || { tags: [] }),
    phase_config: JSON.stringify(input.phaseConfig || {}),
  };

  logger.debug('Creating project', { name: input.name, ownerId: input.ownerId });

  const { text, values } = buildInsert('projects', data);
  const result = await query<ProjectRow>(text, values, { operation: 'create_project' });

  if (result.rows.length === 0) {
    throw new DatabaseError(
      'Failed to create project - no row returned',
      new Error('Insert returned no rows'),
      { operation: 'create_project' }
    );
  }

  const project = mapRowToProject(result.rows[0]);
  logger.info('Project created', { projectId: project.id, name: project.name });

  return project;
}

/**
 * Find a project by its ID.
 *
 * @param id - Project UUID
 * @returns The project or null if not found
 */
export async function findById(id: string): Promise<EEProject | null> {
  const logger = repoLogger.child({ operation: 'findById', projectId: id });

  if (!id || id.trim().length === 0) {
    throw new ValidationError('Project ID is required', {
      operation: 'findById',
    });
  }

  logger.debug('Finding project by ID');

  const result = await query<ProjectRow>(
    'SELECT * FROM projects WHERE id = $1 AND archived_at IS NULL',
    [id],
    { operation: 'find_project_by_id' }
  );

  if (result.rows.length === 0) {
    logger.debug('Project not found');
    return null;
  }

  return mapRowToProject(result.rows[0]);
}

/**
 * Find all projects matching the given filters.
 *
 * @param filters - Optional filters for querying projects
 * @returns Array of projects matching the filters
 */
export async function findAll(filters?: ProjectFilters): Promise<EEProject[]> {
  const logger = repoLogger.child({ operation: 'findAll' });
  const conditions: string[] = ['archived_at IS NULL'];
  const values: unknown[] = [];
  let paramIndex = 1;

  if (filters?.type) {
    conditions.push(`project_type = $${paramIndex++}`);
    values.push(filters.type);
  }

  if (filters?.status) {
    conditions.push(`status = $${paramIndex++}`);
    values.push(filters.status);
  }

  if (filters?.ownerId) {
    conditions.push(`owner_id = $${paramIndex++}`);
    values.push(filters.ownerId);
  }

  if (filters?.organizationId) {
    conditions.push(`organization_id = $${paramIndex++}`);
    values.push(filters.organizationId);
  }

  if (filters?.phase) {
    conditions.push(`phase = $${paramIndex++}`);
    values.push(filters.phase);
  }

  let sql = `SELECT * FROM projects WHERE ${conditions.join(' AND ')} ORDER BY created_at DESC`;

  if (filters?.limit !== undefined) {
    sql += ` LIMIT $${paramIndex++}`;
    values.push(filters.limit);
  }

  if (filters?.offset !== undefined) {
    sql += ` OFFSET $${paramIndex++}`;
    values.push(filters.offset);
  }

  logger.debug('Finding projects', { filters, conditionCount: conditions.length });

  const result = await query<ProjectRow>(sql, values, { operation: 'find_all_projects' });

  logger.debug('Found projects', { count: result.rows.length });

  return result.rows.map(mapRowToProject);
}

/**
 * Update an existing project.
 *
 * @param id - Project UUID
 * @param data - Fields to update
 * @returns The updated project
 */
export async function update(id: string, data: UpdateProjectInput): Promise<EEProject> {
  const logger = repoLogger.child({ operation: 'update', projectId: id });

  if (!id || id.trim().length === 0) {
    throw new ValidationError('Project ID is required', {
      operation: 'update',
    });
  }

  // Check if project exists
  const existing = await findById(id);
  if (!existing) {
    throw new NotFoundError('Project', id, { operation: 'update' });
  }

  const updateData: Record<string, unknown> = {};

  if (data.name !== undefined) {
    if (data.name.trim().length === 0) {
      throw new ValidationError('Project name cannot be empty', {
        operation: 'update',
        projectId: id,
      });
    }
    updateData.name = data.name.trim();
  }

  if (data.description !== undefined) {
    updateData.description = data.description?.trim() || null;
  }

  if (data.repositoryUrl !== undefined) {
    updateData.repository_url = data.repositoryUrl?.trim() || null;
  }

  if (data.projectType !== undefined) {
    updateData.project_type = data.projectType;
  }

  if (data.collaborators !== undefined) {
    updateData.collaborators = JSON.stringify(data.collaborators);
  }

  if (data.metadata !== undefined) {
    // Merge with existing metadata
    const mergedMetadata = {
      ...existing.metadata,
      ...data.metadata,
      tags: data.metadata.tags || existing.metadata.tags,
    };
    updateData.metadata = JSON.stringify(mergedMetadata);
  }

  if (data.phaseConfig !== undefined) {
    updateData.phase_config = JSON.stringify(data.phaseConfig);
  }

  if (Object.keys(updateData).length === 0) {
    logger.debug('No fields to update');
    return existing;
  }

  logger.debug('Updating project', { fields: Object.keys(updateData) });

  const { text, values } = buildUpdate('projects', updateData, 'id', id);
  const result = await query<ProjectRow>(text, values, { operation: 'update_project' });

  if (result.rows.length === 0) {
    throw new DatabaseError(
      'Failed to update project - no row returned',
      new Error('Update returned no rows'),
      { operation: 'update_project' }
    );
  }

  const project = mapRowToProject(result.rows[0]);
  logger.info('Project updated', { projectId: project.id });

  return project;
}

/**
 * Soft delete a project by setting archived_at timestamp.
 *
 * @param id - Project UUID
 */
export async function deleteProject(id: string): Promise<void> {
  const logger = repoLogger.child({ operation: 'delete', projectId: id });

  if (!id || id.trim().length === 0) {
    throw new ValidationError('Project ID is required', {
      operation: 'delete',
    });
  }

  // Check if project exists
  const existing = await findById(id);
  if (!existing) {
    throw new NotFoundError('Project', id, { operation: 'delete' });
  }

  logger.debug('Soft deleting project');

  await query(
    'UPDATE projects SET archived_at = NOW() WHERE id = $1',
    [id],
    { operation: 'delete_project' }
  );

  logger.info('Project deleted', { projectId: id });
}

/**
 * Update the status of a project.
 *
 * @param id - Project UUID
 * @param status - New project status
 * @returns The updated project
 */
export async function updateStatus(id: string, status: ProjectStatus): Promise<EEProject> {
  const logger = repoLogger.child({ operation: 'updateStatus', projectId: id });

  if (!id || id.trim().length === 0) {
    throw new ValidationError('Project ID is required', {
      operation: 'updateStatus',
    });
  }

  const validStatuses: ProjectStatus[] = [
    'draft',
    'in_progress',
    'review',
    'approved',
    'completed',
    'on_hold',
    'cancelled',
  ];

  if (!validStatuses.includes(status)) {
    throw new ValidationError(`Invalid status: ${status}. Valid values: ${validStatuses.join(', ')}`, {
      operation: 'updateStatus',
      projectId: id,
      status,
    });
  }

  // Check if project exists
  const existing = await findById(id);
  if (!existing) {
    throw new NotFoundError('Project', id, { operation: 'updateStatus' });
  }

  logger.debug('Updating project status', { currentStatus: existing.status, newStatus: status });

  const result = await query<ProjectRow>(
    'UPDATE projects SET status = $1 WHERE id = $2 RETURNING *',
    [status, id],
    { operation: 'update_project_status' }
  );

  if (result.rows.length === 0) {
    throw new DatabaseError(
      'Failed to update project status - no row returned',
      new Error('Update returned no rows'),
      { operation: 'update_project_status' }
    );
  }

  const project = mapRowToProject(result.rows[0]);
  logger.info('Project status updated', {
    projectId: project.id,
    previousStatus: existing.status,
    newStatus: status,
  });

  return project;
}

/**
 * Update the current phase of a project.
 *
 * @param id - Project UUID
 * @param phase - New project phase
 * @returns The updated project
 */
export async function updatePhase(id: string, phase: ProjectPhase): Promise<EEProject> {
  const logger = repoLogger.child({ operation: 'updatePhase', projectId: id });

  if (!id || id.trim().length === 0) {
    throw new ValidationError('Project ID is required', {
      operation: 'updatePhase',
    });
  }

  const validPhases: ProjectPhase[] = [
    'ideation',
    'architecture',
    'schematic',
    'simulation',
    'pcb_layout',
    'manufacturing',
    'firmware',
    'testing',
    'production',
    'field_support',
  ];

  if (!validPhases.includes(phase)) {
    throw new ValidationError(`Invalid phase: ${phase}. Valid values: ${validPhases.join(', ')}`, {
      operation: 'updatePhase',
      projectId: id,
      phase,
    });
  }

  // Check if project exists
  const existing = await findById(id);
  if (!existing) {
    throw new NotFoundError('Project', id, { operation: 'updatePhase' });
  }

  logger.debug('Updating project phase', { currentPhase: existing.phase, newPhase: phase });

  const result = await query<ProjectRow>(
    'UPDATE projects SET phase = $1 WHERE id = $2 RETURNING *',
    [phase, id],
    { operation: 'update_project_phase' }
  );

  if (result.rows.length === 0) {
    throw new DatabaseError(
      'Failed to update project phase - no row returned',
      new Error('Update returned no rows'),
      { operation: 'update_project_phase' }
    );
  }

  const project = mapRowToProject(result.rows[0]);
  logger.info('Project phase updated', {
    projectId: project.id,
    previousPhase: existing.phase,
    newPhase: phase,
  });

  return project;
}

/**
 * Get the phase status for all phases of a project.
 *
 * @param projectId - Project UUID
 * @returns Array of phase statuses
 */
export async function getPhases(projectId: string): Promise<PhaseStatus[]> {
  const logger = repoLogger.child({ operation: 'getPhases', projectId });

  if (!projectId || projectId.trim().length === 0) {
    throw new ValidationError('Project ID is required', {
      operation: 'getPhases',
    });
  }

  // Get project with phase config
  const result = await query<ProjectRow>(
    'SELECT * FROM projects WHERE id = $1 AND archived_at IS NULL',
    [projectId],
    { operation: 'get_project_phases' }
  );

  if (result.rows.length === 0) {
    throw new NotFoundError('Project', projectId, { operation: 'getPhases', projectId });
  }

  const project = result.rows[0];
  const currentPhase = project.phase;
  const phaseConfig = project.phase_config || {};

  // Define the ordered phases
  const orderedPhases: ProjectPhase[] = [
    'ideation',
    'architecture',
    'schematic',
    'simulation',
    'pcb_layout',
    'manufacturing',
    'firmware',
    'testing',
    'production',
    'field_support',
  ];

  const currentPhaseIndex = orderedPhases.indexOf(currentPhase);

  logger.debug('Building phase statuses', { currentPhase, currentPhaseIndex });

  // Build phase status for each phase
  const phaseStatuses: PhaseStatus[] = orderedPhases.map((phase, index) => {
    const config = (phaseConfig[phase] as Record<string, unknown>) || {};

    let status: PhaseStatus['status'];
    if (index < currentPhaseIndex) {
      status = 'completed';
    } else if (index === currentPhaseIndex) {
      status = 'in_progress';
    } else {
      status = 'pending';
    }

    // Check if phase is blocked
    if (config.blocked === true) {
      status = 'blocked';
    }

    // Check if phase is skipped
    if (config.skipped === true) {
      status = 'skipped';
    }

    return {
      phase,
      displayName: getPhaseDisplayName(phase),
      required: config.required !== false,
      enabledByDefault: config.enabledByDefault !== false,
      estimatedDuration: (config.estimatedDuration as number) || getDefaultDuration(phase),
      dependencies: (config.dependencies as ProjectPhase[]) || getDefaultDependencies(phase),
      availableSkills: (config.availableSkills as string[]) || getDefaultSkills(phase),
      validationDomains: [],
      deliverables: (config.deliverables as string[]) || [],
      status,
      startedAt: config.startedAt as string | undefined,
      completedAt: config.completedAt as string | undefined,
      blockingIssues: config.blockingIssues as string[] | undefined,
      outputs: config.outputs as Record<string, string> | undefined,
    };
  });

  return phaseStatuses;
}

/**
 * Get display name for a phase.
 */
function getPhaseDisplayName(phase: ProjectPhase): string {
  const displayNames: Record<ProjectPhase, string> = {
    ideation: 'Ideation & Research',
    architecture: 'Architecture',
    schematic: 'Schematic Capture',
    simulation: 'Simulation',
    pcb_layout: 'PCB Layout',
    manufacturing: 'Manufacturing',
    firmware: 'Firmware',
    testing: 'Testing',
    production: 'Production',
    field_support: 'Field Support',
  };
  return displayNames[phase];
}

/**
 * Get default estimated duration for a phase (in hours).
 */
function getDefaultDuration(phase: ProjectPhase): number {
  const durations: Record<ProjectPhase, number> = {
    ideation: 8,
    architecture: 16,
    schematic: 24,
    simulation: 16,
    pcb_layout: 40,
    manufacturing: 80,
    firmware: 80,
    testing: 40,
    production: 160,
    field_support: 40,
  };
  return durations[phase];
}

/**
 * Get default dependencies for a phase.
 */
function getDefaultDependencies(phase: ProjectPhase): ProjectPhase[] {
  const dependencies: Record<ProjectPhase, ProjectPhase[]> = {
    ideation: [],
    architecture: ['ideation'],
    schematic: ['architecture'],
    simulation: ['schematic'],
    pcb_layout: ['simulation'],
    manufacturing: ['pcb_layout'],
    firmware: ['schematic'],
    testing: ['manufacturing', 'firmware'],
    production: ['testing'],
    field_support: ['production'],
  };
  return dependencies[phase];
}

/**
 * Get default skills for a phase.
 */
function getDefaultSkills(phase: ProjectPhase): string[] {
  const skills: Record<ProjectPhase, string[]> = {
    ideation: ['research-paper', 'patent-search', 'requirements-gen'],
    architecture: ['ee-architecture', 'component-select', 'bom-optimize'],
    schematic: ['schematic-gen', 'schematic-review', 'netlist-gen'],
    simulation: ['simulate-spice', 'simulate-thermal', 'simulate-si', 'simulate-rf'],
    pcb_layout: ['pcb-layout', 'mapos', 'stackup-design'],
    manufacturing: ['gerber-gen', 'dfm-check', 'vendor-quote'],
    firmware: ['firmware-gen', 'hal-gen', 'driver-gen'],
    testing: ['test-gen', 'hil-setup', 'test-procedure'],
    production: ['manufacture', 'assembly-guide', 'quality-check'],
    field_support: ['debug-assist', 'service-manual', 'firmware-update'],
  };
  return skills[phase];
}

/**
 * Count projects matching filters.
 *
 * @param filters - Optional filters for counting
 * @returns Count of matching projects
 */
export async function count(filters?: Omit<ProjectFilters, 'limit' | 'offset'>): Promise<number> {
  const conditions: string[] = ['archived_at IS NULL'];
  const values: unknown[] = [];
  let paramIndex = 1;

  if (filters?.type) {
    conditions.push(`project_type = $${paramIndex++}`);
    values.push(filters.type);
  }

  if (filters?.status) {
    conditions.push(`status = $${paramIndex++}`);
    values.push(filters.status);
  }

  if (filters?.ownerId) {
    conditions.push(`owner_id = $${paramIndex++}`);
    values.push(filters.ownerId);
  }

  if (filters?.organizationId) {
    conditions.push(`organization_id = $${paramIndex++}`);
    values.push(filters.organizationId);
  }

  if (filters?.phase) {
    conditions.push(`phase = $${paramIndex++}`);
    values.push(filters.phase);
  }

  const sql = `SELECT COUNT(*) as count FROM projects WHERE ${conditions.join(' AND ')}`;
  const result = await query<{ count: string }>(sql, values, { operation: 'count_projects' });

  return parseInt(result.rows[0].count, 10);
}

/**
 * Batch update multiple projects within a transaction.
 *
 * @param updates - Array of {id, data} pairs
 * @returns Array of updated projects
 */
export async function batchUpdate(
  updates: Array<{ id: string; data: UpdateProjectInput }>
): Promise<EEProject[]> {
  const logger = repoLogger.child({ operation: 'batchUpdate' });

  if (updates.length === 0) {
    return [];
  }

  logger.debug('Batch updating projects', { count: updates.length });

  return withTransaction(async (client) => {
    const results: EEProject[] = [];

    for (const { id, data } of updates) {
      const updateData: Record<string, unknown> = {};

      if (data.name !== undefined) {
        updateData.name = data.name.trim();
      }
      if (data.description !== undefined) {
        updateData.description = data.description?.trim() || null;
      }
      if (data.repositoryUrl !== undefined) {
        updateData.repository_url = data.repositoryUrl?.trim() || null;
      }
      if (data.collaborators !== undefined) {
        updateData.collaborators = JSON.stringify(data.collaborators);
      }
      if (data.metadata !== undefined) {
        updateData.metadata = JSON.stringify(data.metadata);
      }

      if (Object.keys(updateData).length > 0) {
        const { text, values } = buildUpdate('projects', updateData, 'id', id);
        const result = await clientQuery<ProjectRow>(client, text, values);

        if (result.rows.length > 0) {
          results.push(mapRowToProject(result.rows[0]));
        }
      }
    }

    logger.info('Batch update completed', { updatedCount: results.length });
    return results;
  }, { operation: 'batch_update_projects' });
}

// Default export
export default {
  create,
  findById,
  findAll,
  update,
  delete: deleteProject,
  updateStatus,
  updatePhase,
  getPhases,
  count,
  batchUpdate,
};
