/**
 * Skills Engine API Routes
 *
 * REST API endpoints for skill management and execution.
 * All skills are stored in the Skills Engine system and are searchable.
 */

import { Router, Request, Response, NextFunction } from 'express';
import { z } from 'zod';
import { SkillsEngineClient } from '../services/skills/skills-engine-client';
import { ValidationError } from '../utils/errors';
import { log } from '../utils/logger';

// Validation middleware factory
function validate<T>(schema: z.ZodSchema<T>) {
  return (req: Request, _res: Response, next: NextFunction) => {
    try {
      req.body = schema.parse(req.body);
      next();
    } catch (error) {
      if (error instanceof z.ZodError) {
        next(new ValidationError('Invalid request body', {
          operation: 'validation',
          errors: error.errors,
        }));
      } else {
        next(error);
      }
    }
  };
}

export function createSkillsRoutes(skillsClient: SkillsEngineClient): Router {
  const router = Router();

  // ============================================================================
  // Skill Discovery & Search
  // ============================================================================

  /**
   * List all registered skills
   * GET /skills
   */
  router.get('/', async (_req: Request, res: Response, next: NextFunction) => {
    try {
      const skills = skillsClient.getRegisteredSkills();

      res.json({
        success: true,
        data: skills.map(skill => ({
          name: skill.name,
          displayName: skill.displayName,
          description: skill.description,
          version: skill.version,
          status: skill.status,
          triggers: skill.triggers,
          capabilities: skill.capabilities?.map(cap => ({
            name: cap.name,
            description: cap.description,
          })),
        })),
        metadata: {
          total: skills.length,
          timestamp: new Date().toISOString(),
        },
      });
    } catch (error) {
      next(error);
    }
  });

  /**
   * Search skills by query (semantic search)
   * POST /skills/search
   */
  router.post('/search', validate(z.object({
    query: z.string().min(2).max(500),
    limit: z.number().min(1).max(50).optional().default(10),
    category: z.string().optional(),
    phase: z.enum([
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
    ]).optional(),
  })), async (req: Request, res: Response, next: NextFunction) => {
    try {
      log.info('Searching skills', {
        query: req.body.query,
        limit: req.body.limit,
        phase: req.body.phase,
      });

      const results = await skillsClient.searchSkills(req.body.query, {
        limit: req.body.limit,
        category: req.body.category,
        phase: req.body.phase,
      });

      res.json({
        success: true,
        data: results,
        metadata: {
          query: req.body.query,
          resultCount: results.length,
          timestamp: new Date().toISOString(),
        },
      });
    } catch (error) {
      next(error);
    }
  });

  /**
   * Get skill by name
   * GET /skills/:skillName
   */
  router.get('/:skillName', async (req: Request, res: Response, next: NextFunction) => {
    try {
      const skill = await skillsClient.getSkill(req.params.skillName);

      if (!skill) {
        return res.status(404).json({
          success: false,
          error: {
            code: 'SKILL_NOT_FOUND',
            message: `Skill '${req.params.skillName}' not found`,
          },
        });
      }

      res.json({
        success: true,
        data: skill,
      });
    } catch (error) {
      next(error);
    }
  });

  /**
   * Get skills by pipeline phase
   * GET /skills/phase/:phase
   */
  router.get('/phase/:phase', async (req: Request, res: Response, next: NextFunction) => {
    try {
      const validPhases = [
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

      if (!validPhases.includes(req.params.phase)) {
        return res.status(400).json({
          success: false,
          error: {
            code: 'INVALID_PHASE',
            message: `Invalid phase '${req.params.phase}'. Valid phases: ${validPhases.join(', ')}`,
          },
        });
      }

      const skills = skillsClient.getSkillsByPhase(req.params.phase);

      res.json({
        success: true,
        data: skills,
        metadata: {
          phase: req.params.phase,
          count: skills.length,
        },
      });
    } catch (error) {
      next(error);
    }
  });

  // ============================================================================
  // Skill Execution
  // ============================================================================

  /**
   * Execute a skill
   * POST /skills/execute
   */
  router.post('/execute', validate(z.object({
    skillName: z.string(),
    capability: z.string().optional(),
    parameters: z.record(z.unknown()),
    context: z.object({
      projectId: z.string().uuid().optional(),
      userId: z.string().optional(),
      sessionId: z.string().optional(),
    }).optional(),
  })), async (req: Request, res: Response, next: NextFunction) => {
    try {
      log.info('Executing skill', {
        skillName: req.body.skillName,
        capability: req.body.capability,
        projectId: req.body.context?.projectId,
      });

      const response = await skillsClient.executeSkill({
        skillName: req.body.skillName,
        capability: req.body.capability,
        parameters: req.body.parameters,
        context: req.body.context,
      });

      if (response.status === 'failed') {
        return res.status(500).json({
          success: false,
          error: {
            code: 'SKILL_EXECUTION_FAILED',
            message: response.error || 'Skill execution failed',
          },
          data: {
            executionId: response.executionId,
          },
        });
      }

      res.status(response.status === 'queued' ? 202 : 200).json({
        success: true,
        data: response,
      });
    } catch (error) {
      next(error);
    }
  });

  // ============================================================================
  // Skill Registration (Admin)
  // ============================================================================

  /**
   * Register a new skill
   * POST /skills/register
   */
  router.post('/register', validate(z.object({
    name: z.string().min(1).max(100),
    displayName: z.string().min(1).max(255),
    description: z.string().min(10).max(2000),
    version: z.string().regex(/^\d+\.\d+\.\d+$/),
    status: z.enum(['draft', 'testing', 'published', 'deprecated']).default('draft'),
    visibility: z.enum(['private', 'organization', 'public']).default('private'),
    allowedTools: z.array(z.string()),
    triggers: z.array(z.string()),
    capabilities: z.array(z.object({
      name: z.string(),
      description: z.string(),
      parameters: z.array(z.object({
        name: z.string(),
        type: z.string(),
        required: z.boolean(),
        description: z.string(),
      })).optional(),
    })),
  })), async (req: Request, res: Response, next: NextFunction) => {
    try {
      log.info('Registering skill', {
        name: req.body.name,
        version: req.body.version,
      });

      const result = await skillsClient.registerSkill(req.body);

      if (!result.success) {
        return res.status(400).json({
          success: false,
          error: {
            code: 'REGISTRATION_FAILED',
            message: result.error || 'Skill registration failed',
          },
        });
      }

      res.status(201).json({
        success: true,
        data: {
          skillName: result.skillName,
          skillId: result.skillId,
        },
      });
    } catch (error) {
      next(error);
    }
  });

  /**
   * Re-register all skills from filesystem
   * POST /skills/register-all
   */
  router.post('/register-all', async (_req: Request, res: Response, next: NextFunction) => {
    try {
      log.info('Re-registering all skills');

      const results = await skillsClient.registerAllSkills();

      const successCount = results.filter(r => r.success).length;
      const failedCount = results.filter(r => !r.success).length;

      res.json({
        success: true,
        data: {
          total: results.length,
          success: successCount,
          failed: failedCount,
          results: results.map(r => ({
            skillName: r.skillName,
            success: r.success,
            skillId: r.skillId,
            error: r.error,
          })),
        },
      });
    } catch (error) {
      next(error);
    }
  });

  // ============================================================================
  // Pipeline Phase Mapping
  // ============================================================================

  /**
   * Get all pipeline phases with their associated skills
   * GET /skills/phases
   */
  router.get('/pipeline/phases', async (_req: Request, res: Response, next: NextFunction) => {
    try {
      const phases = [
        {
          id: 'ideation',
          name: 'Ideation & Research',
          order: 1,
          skills: skillsClient.getSkillsByPhase('ideation'),
        },
        {
          id: 'architecture',
          name: 'Architecture & Specification',
          order: 2,
          skills: skillsClient.getSkillsByPhase('architecture'),
        },
        {
          id: 'schematic',
          name: 'Schematic Capture',
          order: 3,
          skills: skillsClient.getSkillsByPhase('schematic'),
        },
        {
          id: 'simulation',
          name: 'Simulation Suite',
          order: 4,
          skills: skillsClient.getSkillsByPhase('simulation'),
        },
        {
          id: 'pcb_layout',
          name: 'PCB Layout',
          order: 5,
          skills: skillsClient.getSkillsByPhase('pcb_layout'),
        },
        {
          id: 'manufacturing',
          name: 'Manufacturing Prep',
          order: 6,
          skills: skillsClient.getSkillsByPhase('manufacturing'),
        },
        {
          id: 'firmware',
          name: 'Firmware Development',
          order: 7,
          skills: skillsClient.getSkillsByPhase('firmware'),
        },
        {
          id: 'testing',
          name: 'Test Development',
          order: 8,
          skills: skillsClient.getSkillsByPhase('testing'),
        },
        {
          id: 'production',
          name: 'Production & Assembly',
          order: 9,
          skills: skillsClient.getSkillsByPhase('production'),
        },
        {
          id: 'field_support',
          name: 'Field Support',
          order: 10,
          skills: skillsClient.getSkillsByPhase('field_support'),
        },
      ];

      res.json({
        success: true,
        data: phases,
        metadata: {
          totalPhases: phases.length,
          totalSkills: phases.reduce((sum, p) => sum + p.skills.length, 0),
        },
      });
    } catch (error) {
      next(error);
    }
  });

  return router;
}

export default createSkillsRoutes;