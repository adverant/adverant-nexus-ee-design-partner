/**
 * HIL Test Sequence Repository
 *
 * Database operations for HIL test sequences. Handles CRUD operations,
 * template management, versioning, and sequence configuration.
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
  HILTestSequence,
  HILTestType,
  HILSequenceConfig,
  HILPassCriteria,
  HILTemplateVariable,
  CreateHILTestSequenceInput,
  UpdateHILTestSequenceInput,
  HILTestSequenceFilters,
} from '../../types/hil-types.js';

// ============================================================================
// Types
// ============================================================================

interface TestSequenceRow {
  id: string;
  project_id: string;
  schematic_id: string | null;
  pcb_layout_id: string | null;
  name: string;
  description: string | null;
  test_type: HILTestType;
  sequence_config: HILSequenceConfig;
  pass_criteria: HILPassCriteria;
  estimated_duration_ms: number | null;
  timeout_ms: number | null;
  priority: number;
  tags: string[];
  category: string | null;
  version: number;
  parent_version_id: string | null;
  is_template: boolean;
  template_variables: Record<string, HILTemplateVariable> | null;
  parent_template_id: string | null;
  created_by: string | null;
  last_modified_by: string | null;
  metadata: Record<string, unknown>;
  created_at: Date;
  updated_at: Date;
}

// ============================================================================
// Repository Implementation
// ============================================================================

const repoLogger: Logger = log.child({ service: 'hil-test-sequence-repository' });

/**
 * Validate test type
 */
function validateTestType(type: string): asserts type is HILTestType {
  const validTypes: HILTestType[] = [
    'foc_startup',
    'foc_steady_state',
    'foc_transient',
    'foc_speed_reversal',
    'pwm_analysis',
    'phase_current',
    'hall_sensor',
    'thermal_profile',
    'overcurrent_protection',
    'efficiency_sweep',
    'load_step',
    'no_load',
    'locked_rotor',
    'custom',
  ];

  if (!validTypes.includes(type as HILTestType)) {
    throw new ValidationError(`Invalid test type: ${type}`, {
      operation: 'validate_test_type',
      validTypes,
    });
  }
}

/**
 * Map a database row to an HILTestSequence object.
 */
function mapRowToSequence(row: TestSequenceRow): HILTestSequence {
  return {
    id: row.id,
    projectId: row.project_id,
    schematicId: row.schematic_id || undefined,
    pcbLayoutId: row.pcb_layout_id || undefined,
    name: row.name,
    description: row.description || undefined,
    testType: row.test_type,
    sequenceConfig: row.sequence_config || { steps: [], instrumentRequirements: [] },
    passCriteria: row.pass_criteria || {
      minPassPercentage: 100,
      criticalMeasurements: [],
      failFast: false,
    },
    estimatedDurationMs: row.estimated_duration_ms || undefined,
    timeoutMs: row.timeout_ms || undefined,
    priority: row.priority,
    tags: row.tags || [],
    category: row.category || undefined,
    version: row.version,
    parentVersionId: row.parent_version_id || undefined,
    isTemplate: row.is_template,
    templateVariables: row.template_variables || undefined,
    parentTemplateId: row.parent_template_id || undefined,
    createdBy: row.created_by || undefined,
    lastModifiedBy: row.last_modified_by || undefined,
    metadata: row.metadata || {},
    createdAt: row.created_at.toISOString(),
    updatedAt: row.updated_at.toISOString(),
  };
}

/**
 * Validate sequence configuration
 */
function validateSequenceConfig(config: HILSequenceConfig): void {
  if (!config) {
    throw new ValidationError('Sequence configuration is required', {
      operation: 'validate_sequence_config',
    });
  }

  if (!config.steps || !Array.isArray(config.steps)) {
    throw new ValidationError('Sequence steps must be an array', {
      operation: 'validate_sequence_config',
    });
  }

  if (!config.instrumentRequirements || !Array.isArray(config.instrumentRequirements)) {
    throw new ValidationError('Instrument requirements must be an array', {
      operation: 'validate_sequence_config',
    });
  }

  // Validate each step
  for (let i = 0; i < config.steps.length; i++) {
    const step = config.steps[i];
    if (!step.id || step.id.trim().length === 0) {
      throw new ValidationError(`Step ${i} must have an ID`, {
        operation: 'validate_sequence_config',
        stepIndex: i,
      });
    }
    if (!step.name || step.name.trim().length === 0) {
      throw new ValidationError(`Step ${i} must have a name`, {
        operation: 'validate_sequence_config',
        stepIndex: i,
      });
    }
    if (!step.type) {
      throw new ValidationError(`Step ${i} must have a type`, {
        operation: 'validate_sequence_config',
        stepIndex: i,
      });
    }
  }
}

/**
 * Validate pass criteria
 */
function validatePassCriteria(criteria: HILPassCriteria): void {
  if (!criteria) {
    throw new ValidationError('Pass criteria is required', {
      operation: 'validate_pass_criteria',
    });
  }

  if (
    typeof criteria.minPassPercentage !== 'number' ||
    criteria.minPassPercentage < 0 ||
    criteria.minPassPercentage > 100
  ) {
    throw new ValidationError('minPassPercentage must be a number between 0 and 100', {
      operation: 'validate_pass_criteria',
      value: criteria.minPassPercentage,
    });
  }

  if (!Array.isArray(criteria.criticalMeasurements)) {
    throw new ValidationError('criticalMeasurements must be an array', {
      operation: 'validate_pass_criteria',
    });
  }

  if (typeof criteria.failFast !== 'boolean') {
    throw new ValidationError('failFast must be a boolean', {
      operation: 'validate_pass_criteria',
    });
  }
}

/**
 * Create a new HIL test sequence.
 *
 * @param input - Test sequence creation data
 * @returns The created test sequence
 */
export async function create(input: CreateHILTestSequenceInput): Promise<HILTestSequence> {
  const logger = repoLogger.child({ operation: 'create' });

  // Validate required fields
  if (!input.projectId || input.projectId.trim().length === 0) {
    throw new ValidationError('Project ID is required', {
      operation: 'create',
      input,
    });
  }

  if (!input.name || input.name.trim().length === 0) {
    throw new ValidationError('Sequence name is required', {
      operation: 'create',
      input,
    });
  }

  if (!input.testType) {
    throw new ValidationError('Test type is required', {
      operation: 'create',
      input,
    });
  }

  if (!input.sequenceConfig) {
    throw new ValidationError('Sequence configuration is required', {
      operation: 'create',
      input,
    });
  }

  if (!input.passCriteria) {
    throw new ValidationError('Pass criteria is required', {
      operation: 'create',
      input,
    });
  }

  // Validate enum and complex types
  validateTestType(input.testType);
  validateSequenceConfig(input.sequenceConfig);
  validatePassCriteria(input.passCriteria);

  const data: Record<string, unknown> = {
    project_id: input.projectId,
    name: input.name.trim(),
    description: input.description || null,
    test_type: input.testType,
    sequence_config: JSON.stringify(input.sequenceConfig),
    pass_criteria: JSON.stringify(input.passCriteria),
    schematic_id: input.schematicId || null,
    pcb_layout_id: input.pcbLayoutId || null,
    estimated_duration_ms: input.estimatedDurationMs || null,
    timeout_ms: input.timeoutMs || null,
    priority: input.priority ?? 0,
    tags: input.tags || [],
    category: input.category || null,
    is_template: input.isTemplate ?? false,
    template_variables: input.templateVariables
      ? JSON.stringify(input.templateVariables)
      : null,
    metadata: JSON.stringify(input.metadata || {}),
  };

  logger.debug('Creating HIL test sequence', {
    name: input.name,
    projectId: input.projectId,
    testType: input.testType,
    isTemplate: input.isTemplate,
  });

  const { text, values } = buildInsert('hil_test_sequences', data);
  const result = await query<TestSequenceRow>(text, values, {
    operation: 'create_hil_test_sequence',
  });

  if (result.rows.length === 0) {
    throw new DatabaseError(
      'Failed to create HIL test sequence - no row returned',
      new Error('Insert returned no rows'),
      { operation: 'create_hil_test_sequence' }
    );
  }

  const sequence = mapRowToSequence(result.rows[0]);
  logger.info('HIL test sequence created', {
    sequenceId: sequence.id,
    name: sequence.name,
    testType: sequence.testType,
    isTemplate: sequence.isTemplate,
  });

  return sequence;
}

/**
 * Find a test sequence by its ID.
 *
 * @param id - Sequence UUID
 * @returns The sequence or null if not found
 */
export async function findById(id: string): Promise<HILTestSequence | null> {
  const logger = repoLogger.child({ operation: 'findById', sequenceId: id });

  if (!id || id.trim().length === 0) {
    throw new ValidationError('Sequence ID is required', {
      operation: 'findById',
    });
  }

  logger.debug('Finding test sequence by ID');

  const result = await query<TestSequenceRow>(
    'SELECT * FROM hil_test_sequences WHERE id = $1',
    [id],
    { operation: 'find_test_sequence_by_id' }
  );

  if (result.rows.length === 0) {
    logger.debug('Test sequence not found');
    return null;
  }

  return mapRowToSequence(result.rows[0]);
}

/**
 * Find all test sequences for a project with optional filters.
 *
 * @param projectId - Project UUID
 * @param filters - Optional filters
 * @returns Array of test sequences for the project
 */
export async function findByProject(
  projectId: string,
  filters?: HILTestSequenceFilters
): Promise<HILTestSequence[]> {
  const logger = repoLogger.child({ operation: 'findByProject', projectId });

  if (!projectId || projectId.trim().length === 0) {
    throw new ValidationError('Project ID is required', {
      operation: 'findByProject',
    });
  }

  const conditions: string[] = ['project_id = $1'];
  const values: unknown[] = [projectId];
  let paramIndex = 2;

  if (filters?.testType) {
    validateTestType(filters.testType);
    conditions.push(`test_type = $${paramIndex++}`);
    values.push(filters.testType);
  }

  if (filters?.isTemplate !== undefined) {
    conditions.push(`is_template = $${paramIndex++}`);
    values.push(filters.isTemplate);
  }

  if (filters?.category) {
    conditions.push(`category = $${paramIndex++}`);
    values.push(filters.category);
  }

  if (filters?.schematicId) {
    conditions.push(`schematic_id = $${paramIndex++}`);
    values.push(filters.schematicId);
  }

  if (filters?.pcbLayoutId) {
    conditions.push(`pcb_layout_id = $${paramIndex++}`);
    values.push(filters.pcbLayoutId);
  }

  if (filters?.tags && filters.tags.length > 0) {
    conditions.push(`tags && $${paramIndex++}`);
    values.push(filters.tags);
  }

  const sql = `SELECT * FROM hil_test_sequences WHERE ${conditions.join(' AND ')} ORDER BY priority DESC, name ASC`;

  logger.debug('Finding test sequences by project', { filters });

  const result = await query<TestSequenceRow>(sql, values, {
    operation: 'find_test_sequences_by_project',
  });

  logger.debug('Found test sequences', { count: result.rows.length });

  return result.rows.map(mapRowToSequence);
}

/**
 * Find templates (global or project-specific).
 *
 * @param projectId - Optional project ID for project-specific templates
 * @param testType - Optional test type filter
 * @returns Array of template sequences
 */
export async function findTemplates(
  projectId?: string,
  testType?: HILTestType
): Promise<HILTestSequence[]> {
  const logger = repoLogger.child({ operation: 'findTemplates' });

  const conditions: string[] = ['is_template = true'];
  const values: unknown[] = [];
  let paramIndex = 1;

  if (projectId) {
    conditions.push(`project_id = $${paramIndex++}`);
    values.push(projectId);
  }

  if (testType) {
    validateTestType(testType);
    conditions.push(`test_type = $${paramIndex++}`);
    values.push(testType);
  }

  const sql = `SELECT * FROM hil_test_sequences WHERE ${conditions.join(' AND ')} ORDER BY test_type, name`;

  logger.debug('Finding templates', { projectId, testType });

  const result = await query<TestSequenceRow>(sql, values, {
    operation: 'find_templates',
  });

  return result.rows.map(mapRowToSequence);
}

/**
 * Instantiate a template into a new test sequence.
 *
 * @param templateId - Template sequence ID
 * @param projectId - Target project ID
 * @param name - Name for the new sequence
 * @param variables - Template variable values
 * @returns The created sequence
 */
export async function instantiateTemplate(
  templateId: string,
  projectId: string,
  name: string,
  variables?: Record<string, unknown>
): Promise<HILTestSequence> {
  const logger = repoLogger.child({
    operation: 'instantiateTemplate',
    templateId,
    projectId,
  });

  if (!templateId || templateId.trim().length === 0) {
    throw new ValidationError('Template ID is required', {
      operation: 'instantiateTemplate',
    });
  }

  if (!projectId || projectId.trim().length === 0) {
    throw new ValidationError('Project ID is required', {
      operation: 'instantiateTemplate',
    });
  }

  if (!name || name.trim().length === 0) {
    throw new ValidationError('Sequence name is required', {
      operation: 'instantiateTemplate',
    });
  }

  // Find the template
  const template = await findById(templateId);
  if (!template) {
    throw new NotFoundError('HILTestSequence (template)', templateId, {
      operation: 'instantiateTemplate',
    });
  }

  if (!template.isTemplate) {
    throw new ValidationError('The specified sequence is not a template', {
      operation: 'instantiateTemplate',
      sequenceId: templateId,
    });
  }

  logger.debug('Instantiating template', {
    templateName: template.name,
    targetName: name,
    hasVariables: !!variables,
  });

  // Apply variable substitutions to sequence config
  let sequenceConfig = template.sequenceConfig;
  if (variables && template.templateVariables) {
    sequenceConfig = applyTemplateVariables(
      template.sequenceConfig,
      template.templateVariables,
      variables
    );
  }

  // Create the new sequence
  return create({
    projectId,
    name: name.trim(),
    description: template.description,
    testType: template.testType,
    sequenceConfig,
    passCriteria: template.passCriteria,
    estimatedDurationMs: template.estimatedDurationMs,
    timeoutMs: template.timeoutMs,
    priority: template.priority,
    tags: template.tags,
    category: template.category,
    isTemplate: false,
    metadata: {
      ...template.metadata,
      instantiatedFromTemplate: templateId,
      templateVariables: variables,
    },
  });
}

/**
 * Apply template variables to sequence configuration.
 */
function applyTemplateVariables(
  config: HILSequenceConfig,
  templateVariables: Record<string, HILTemplateVariable>,
  values: Record<string, unknown>
): HILSequenceConfig {
  // Deep clone the config
  const newConfig = JSON.parse(JSON.stringify(config)) as HILSequenceConfig;

  // Recursively replace variable references
  const replaceVariables = (obj: unknown): unknown => {
    if (typeof obj === 'string') {
      // Replace ${variableName} patterns
      return obj.replace(/\$\{(\w+)\}/g, (match, varName) => {
        if (varName in values) {
          return String(values[varName]);
        }
        if (varName in templateVariables && templateVariables[varName].defaultValue !== undefined) {
          return String(templateVariables[varName].defaultValue);
        }
        return match;
      });
    }

    if (Array.isArray(obj)) {
      return obj.map(replaceVariables);
    }

    if (obj !== null && typeof obj === 'object') {
      const result: Record<string, unknown> = {};
      for (const [key, value] of Object.entries(obj)) {
        result[key] = replaceVariables(value);
      }
      return result;
    }

    return obj;
  };

  return replaceVariables(newConfig) as HILSequenceConfig;
}

/**
 * Update a test sequence.
 *
 * @param id - Sequence UUID
 * @param data - Fields to update
 * @returns The updated sequence
 */
export async function update(
  id: string,
  data: UpdateHILTestSequenceInput
): Promise<HILTestSequence> {
  const logger = repoLogger.child({ operation: 'update', sequenceId: id });

  if (!id || id.trim().length === 0) {
    throw new ValidationError('Sequence ID is required', {
      operation: 'update',
    });
  }

  // Check if sequence exists
  const existing = await findById(id);
  if (!existing) {
    throw new NotFoundError('HILTestSequence', id, { operation: 'update' });
  }

  const updateData: Record<string, unknown> = {};

  if (data.name !== undefined) {
    if (data.name.trim().length === 0) {
      throw new ValidationError('Sequence name cannot be empty', {
        operation: 'update',
        sequenceId: id,
      });
    }
    updateData.name = data.name.trim();
  }

  if (data.description !== undefined) {
    updateData.description = data.description || null;
  }

  if (data.sequenceConfig !== undefined) {
    validateSequenceConfig(data.sequenceConfig);
    updateData.sequence_config = JSON.stringify(data.sequenceConfig);
  }

  if (data.passCriteria !== undefined) {
    validatePassCriteria(data.passCriteria);
    updateData.pass_criteria = JSON.stringify(data.passCriteria);
  }

  if (data.estimatedDurationMs !== undefined) {
    updateData.estimated_duration_ms = data.estimatedDurationMs || null;
  }

  if (data.timeoutMs !== undefined) {
    updateData.timeout_ms = data.timeoutMs || null;
  }

  if (data.priority !== undefined) {
    updateData.priority = data.priority;
  }

  if (data.tags !== undefined) {
    updateData.tags = data.tags;
  }

  if (data.category !== undefined) {
    updateData.category = data.category || null;
  }

  if (data.metadata !== undefined) {
    updateData.metadata = JSON.stringify(data.metadata);
  }

  if (Object.keys(updateData).length === 0) {
    logger.debug('No fields to update');
    return existing;
  }

  // Increment version on config/criteria changes
  if (updateData.sequence_config || updateData.pass_criteria) {
    updateData.version = existing.version + 1;
    updateData.parent_version_id = existing.id;
  }

  logger.debug('Updating test sequence', { fields: Object.keys(updateData) });

  const { text, values } = buildUpdate('hil_test_sequences', updateData, 'id', id);
  const result = await query<TestSequenceRow>(text, values, {
    operation: 'update_test_sequence',
  });

  if (result.rows.length === 0) {
    throw new DatabaseError(
      'Failed to update test sequence - no row returned',
      new Error('Update returned no rows'),
      { operation: 'update_test_sequence' }
    );
  }

  const sequence = mapRowToSequence(result.rows[0]);
  logger.info('Test sequence updated', {
    sequenceId: sequence.id,
    version: sequence.version,
  });

  return sequence;
}

/**
 * Delete a test sequence.
 *
 * @param id - Sequence UUID
 */
export async function deleteSequence(id: string): Promise<void> {
  const logger = repoLogger.child({ operation: 'delete', sequenceId: id });

  if (!id || id.trim().length === 0) {
    throw new ValidationError('Sequence ID is required', {
      operation: 'delete',
    });
  }

  // Check if sequence exists
  const existing = await findById(id);
  if (!existing) {
    throw new NotFoundError('HILTestSequence', id, { operation: 'delete' });
  }

  // Check if any test runs reference this sequence
  const runsResult = await query<{ count: string }>(
    'SELECT COUNT(*) as count FROM hil_test_runs WHERE sequence_id = $1',
    [id],
    { operation: 'check_sequence_runs' }
  );

  const runCount = parseInt(runsResult.rows[0].count, 10);
  if (runCount > 0) {
    throw new ValidationError(
      `Cannot delete sequence with ${runCount} associated test runs`,
      {
        operation: 'delete',
        sequenceId: id,
        runCount,
      }
    );
  }

  logger.debug('Deleting test sequence');

  await query('DELETE FROM hil_test_sequences WHERE id = $1', [id], {
    operation: 'delete_test_sequence',
  });

  logger.info('Test sequence deleted', { sequenceId: id, name: existing.name });
}

/**
 * Find sequences by test type.
 *
 * @param testType - Test type to filter by
 * @returns Array of sequences
 */
export async function findByTestType(testType: HILTestType): Promise<HILTestSequence[]> {
  const logger = repoLogger.child({ operation: 'findByTestType', testType });

  validateTestType(testType);

  logger.debug('Finding sequences by test type');

  const result = await query<TestSequenceRow>(
    'SELECT * FROM hil_test_sequences WHERE test_type = $1 ORDER BY priority DESC, name',
    [testType],
    { operation: 'find_sequences_by_test_type' }
  );

  return result.rows.map(mapRowToSequence);
}

/**
 * Find sequences linked to a schematic.
 *
 * @param schematicId - Schematic UUID
 * @returns Array of sequences
 */
export async function findBySchematic(schematicId: string): Promise<HILTestSequence[]> {
  const logger = repoLogger.child({ operation: 'findBySchematic', schematicId });

  if (!schematicId || schematicId.trim().length === 0) {
    throw new ValidationError('Schematic ID is required', {
      operation: 'findBySchematic',
    });
  }

  logger.debug('Finding sequences by schematic');

  const result = await query<TestSequenceRow>(
    'SELECT * FROM hil_test_sequences WHERE schematic_id = $1 ORDER BY priority DESC, name',
    [schematicId],
    { operation: 'find_sequences_by_schematic' }
  );

  return result.rows.map(mapRowToSequence);
}

/**
 * Clone a sequence within the same or different project.
 *
 * @param id - Source sequence UUID
 * @param projectId - Target project ID
 * @param newName - Name for the cloned sequence
 * @returns The cloned sequence
 */
export async function clone(
  id: string,
  projectId: string,
  newName: string
): Promise<HILTestSequence> {
  const logger = repoLogger.child({ operation: 'clone', sequenceId: id });

  if (!id || id.trim().length === 0) {
    throw new ValidationError('Sequence ID is required', {
      operation: 'clone',
    });
  }

  if (!projectId || projectId.trim().length === 0) {
    throw new ValidationError('Project ID is required', {
      operation: 'clone',
    });
  }

  if (!newName || newName.trim().length === 0) {
    throw new ValidationError('New sequence name is required', {
      operation: 'clone',
    });
  }

  // Find the source sequence
  const source = await findById(id);
  if (!source) {
    throw new NotFoundError('HILTestSequence', id, { operation: 'clone' });
  }

  logger.debug('Cloning sequence', {
    sourceName: source.name,
    targetName: newName,
    targetProjectId: projectId,
  });

  // Create the cloned sequence
  return create({
    projectId,
    name: newName.trim(),
    description: source.description,
    testType: source.testType,
    sequenceConfig: source.sequenceConfig,
    passCriteria: source.passCriteria,
    schematicId: source.schematicId,
    pcbLayoutId: source.pcbLayoutId,
    estimatedDurationMs: source.estimatedDurationMs,
    timeoutMs: source.timeoutMs,
    priority: source.priority,
    tags: source.tags,
    category: source.category,
    isTemplate: false,
    metadata: {
      ...source.metadata,
      clonedFrom: id,
    },
  });
}

/**
 * Get sequence statistics for a project.
 *
 * @param projectId - Project UUID
 * @returns Statistics object
 */
export async function getProjectStats(projectId: string): Promise<{
  total: number;
  byTestType: Record<HILTestType, number>;
  templates: number;
  avgDurationMs: number | null;
}> {
  const logger = repoLogger.child({ operation: 'getProjectStats', projectId });

  if (!projectId || projectId.trim().length === 0) {
    throw new ValidationError('Project ID is required', {
      operation: 'getProjectStats',
    });
  }

  logger.debug('Getting project sequence stats');

  const result = await query<{
    total: string;
    templates: string;
    avg_duration: string | null;
  }>(
    `SELECT
       COUNT(*) as total,
       COUNT(*) FILTER (WHERE is_template = true) as templates,
       AVG(estimated_duration_ms) as avg_duration
     FROM hil_test_sequences
     WHERE project_id = $1`,
    [projectId],
    { operation: 'get_sequence_stats' }
  );

  const byTypeResult = await query<{ test_type: HILTestType; count: string }>(
    `SELECT test_type, COUNT(*) as count
     FROM hil_test_sequences
     WHERE project_id = $1
     GROUP BY test_type`,
    [projectId],
    { operation: 'get_sequence_stats_by_type' }
  );

  const byTestType: Record<string, number> = {};
  for (const row of byTypeResult.rows) {
    byTestType[row.test_type] = parseInt(row.count, 10);
  }

  const stats = result.rows[0];
  return {
    total: parseInt(stats.total, 10),
    templates: parseInt(stats.templates, 10),
    avgDurationMs: stats.avg_duration ? parseFloat(stats.avg_duration) : null,
    byTestType: byTestType as Record<HILTestType, number>,
  };
}

/**
 * Get version history for a sequence.
 *
 * @param id - Sequence UUID
 * @returns Array of sequence versions
 */
export async function getVersionHistory(id: string): Promise<HILTestSequence[]> {
  const logger = repoLogger.child({ operation: 'getVersionHistory', sequenceId: id });

  if (!id || id.trim().length === 0) {
    throw new ValidationError('Sequence ID is required', {
      operation: 'getVersionHistory',
    });
  }

  logger.debug('Getting version history');

  // Recursive CTE to get all versions
  const result = await query<TestSequenceRow>(
    `WITH RECURSIVE version_tree AS (
       SELECT * FROM hil_test_sequences WHERE id = $1
       UNION ALL
       SELECT s.* FROM hil_test_sequences s
       INNER JOIN version_tree v ON s.id = v.parent_version_id
     )
     SELECT * FROM version_tree ORDER BY version DESC`,
    [id],
    { operation: 'get_version_history' }
  );

  return result.rows.map(mapRowToSequence);
}

// Default export
export default {
  create,
  findById,
  findByProject,
  findTemplates,
  instantiateTemplate,
  update,
  delete: deleteSequence,
  findByTestType,
  findBySchematic,
  clone,
  getProjectStats,
  getVersionHistory,
};
