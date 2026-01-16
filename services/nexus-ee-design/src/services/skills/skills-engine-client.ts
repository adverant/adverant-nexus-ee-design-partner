/**
 * Skills Engine Client
 *
 * Integrates EE Design Partner skills with the Nexus Skills Engine.
 * Handles skill registration, search, and execution.
 */

import { EventEmitter } from 'events';
import * as fs from 'fs/promises';
import * as path from 'path';
import * as yaml from 'js-yaml';
import { v4 as uuidv4 } from 'uuid';
import { log as logger } from '../../utils/logger.js';
import { Skill, SkillCapability, SkillParameter, SkillExecution } from '../../types';

// ============================================================================
// Types
// ============================================================================

export interface SkillsEngineConfig {
  apiUrl: string;
  apiKey: string;
  skillsDirectory: string;
  autoRegister: boolean;
  syncInterval: number; // ms
}

export interface SkillDefinition {
  name: string;
  displayName: string;
  description: string;
  version: string;
  status: 'draft' | 'testing' | 'published' | 'deprecated';
  visibility: 'private' | 'organization' | 'public';
  allowedTools: string[];
  triggers: string[];
  capabilities: SkillCapability[];
  subSkills?: string[];
  content?: string; // Full markdown content
}

export interface SkillSearchResult {
  skill: Skill;
  relevance: number;
  matchedOn: string[];
}

export interface SkillExecutionRequest {
  skillName: string;
  capability?: string;
  parameters: Record<string, unknown>;
  context?: {
    projectId?: string;
    userId?: string;
    sessionId?: string;
  };
}

export interface SkillExecutionResponse {
  executionId: string;
  status: 'queued' | 'running' | 'completed' | 'failed';
  result?: unknown;
  error?: string;
  startedAt?: string;
  completedAt?: string;
}

export interface RegistrationResult {
  skillName: string;
  success: boolean;
  skillId?: string;
  error?: string;
}

// ============================================================================
// Skills Engine Client
// ============================================================================

export class SkillsEngineClient extends EventEmitter {
  private config: SkillsEngineConfig;
  private registeredSkills: Map<string, Skill>;
  private syncTimer?: NodeJS.Timeout;

  constructor(config: Partial<SkillsEngineConfig> = {}) {
    super();
    this.config = {
      apiUrl: config.apiUrl || process.env.NEXUS_API_URL || 'https://api.adverant.ai',
      apiKey: config.apiKey || process.env.NEXUS_API_KEY || '',
      skillsDirectory: config.skillsDirectory || './skills',
      autoRegister: config.autoRegister !== false,
      syncInterval: config.syncInterval || 300000 // 5 minutes
    };
    this.registeredSkills = new Map();
  }

  /**
   * Initialize the client and register all skills
   */
  async initialize(): Promise<void> {
    logger.info('Initializing Skills Engine client');

    // Load and register all skills from directory
    if (this.config.autoRegister) {
      await this.registerAllSkills();
    }

    // Start periodic sync
    this.startSync();

    this.emit('initialized');
    logger.info('Skills Engine client initialized', {
      skillCount: this.registeredSkills.size
    });
  }

  /**
   * Register all skills from the skills directory
   */
  async registerAllSkills(): Promise<RegistrationResult[]> {
    const results: RegistrationResult[] = [];

    try {
      // Read all .md files from skills directory
      const skillsPath = path.resolve(this.config.skillsDirectory);
      const files = await fs.readdir(skillsPath);
      const skillFiles = files.filter(f => f.endsWith('.md'));

      logger.info('Found skill files', { count: skillFiles.length });

      for (const file of skillFiles) {
        const filePath = path.join(skillsPath, file);
        const result = await this.registerSkillFromFile(filePath);
        results.push(result);
      }

      // Log summary
      const successCount = results.filter(r => r.success).length;
      logger.info('Skill registration complete', {
        total: results.length,
        success: successCount,
        failed: results.length - successCount
      });

    } catch (error) {
      logger.error('Failed to register skills', error instanceof Error ? error : undefined);
    }

    return results;
  }

  /**
   * Register a skill from a markdown file
   */
  async registerSkillFromFile(filePath: string): Promise<RegistrationResult> {
    const fileName = path.basename(filePath);

    try {
      const content = await fs.readFile(filePath, 'utf-8');
      const definition = this.parseSkillDefinition(content);

      if (!definition) {
        return {
          skillName: fileName,
          success: false,
          error: 'Failed to parse skill definition'
        };
      }

      return await this.registerSkill(definition);

    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Unknown error';
      logger.error('Failed to register skill from file', error instanceof Error ? error : undefined, { filePath });

      return {
        skillName: fileName,
        success: false,
        error: errorMessage
      };
    }
  }

  /**
   * Parse skill definition from markdown content
   */
  private parseSkillDefinition(content: string): SkillDefinition | null {
    try {
      // Extract YAML frontmatter
      const frontmatterMatch = content.match(/^---\n([\s\S]*?)\n---/);
      if (!frontmatterMatch) {
        return null;
      }

      const frontmatter = yaml.load(frontmatterMatch[1]) as Record<string, unknown>;

      // Extract markdown content after frontmatter
      const markdownContent = content.slice(frontmatterMatch[0].length).trim();

      return {
        name: frontmatter.name as string,
        displayName: frontmatter.displayName as string,
        description: frontmatter.description as string,
        version: frontmatter.version as string,
        status: frontmatter.status as SkillDefinition['status'],
        visibility: frontmatter.visibility as SkillDefinition['visibility'],
        allowedTools: frontmatter['allowed-tools'] as string[] || [],
        triggers: frontmatter.triggers as string[] || [],
        capabilities: (frontmatter.capabilities as SkillCapability[]) || [],
        subSkills: frontmatter['sub-skills'] as string[] || undefined,
        content: markdownContent
      };

    } catch (error) {
      logger.error('Failed to parse skill definition', error instanceof Error ? error : undefined);
      return null;
    }
  }

  /**
   * Register a skill with the Skills Engine
   */
  async registerSkill(definition: SkillDefinition): Promise<RegistrationResult> {
    try {
      // Build skill object
      const skill: Skill = {
        name: definition.name,
        displayName: definition.displayName,
        description: definition.description,
        version: definition.version,
        status: definition.status,
        visibility: definition.visibility,
        allowedTools: definition.allowedTools,
        triggers: definition.triggers,
        capabilities: definition.capabilities,
        subSkills: definition.subSkills
      };

      // Register with API
      const response = await this.callApi('POST', '/api/skills/register', {
        skill,
        content: definition.content,
        embeddings: await this.generateEmbeddings(definition)
      });

      if (response.success) {
        // Store locally
        this.registeredSkills.set(skill.name, skill);

        this.emit('skill:registered', { skill });
        logger.info('Skill registered', { name: skill.name });

        return {
          skillName: skill.name,
          success: true,
          skillId: (response.data as { skillId: string }).skillId
        };
      } else {
        return {
          skillName: skill.name,
          success: false,
          error: response.error?.message || 'Registration failed'
        };
      }

    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Unknown error';
      logger.error('Failed to register skill', error instanceof Error ? error : undefined, {
        name: definition.name
      });

      return {
        skillName: definition.name,
        success: false,
        error: errorMessage
      };
    }
  }

  /**
   * Generate embeddings for skill searchability
   *
   * Uses OpenRouter API for high-quality embeddings when available,
   * falls back to deterministic hash-based embeddings for offline/test use.
   */
  private async generateEmbeddings(definition: SkillDefinition): Promise<{
    nameEmbedding: number[];
    descriptionEmbedding: number[];
    contentEmbedding: number[];
  }> {
    const openRouterApiKey = process.env.OPENROUTER_API_KEY;

    if (openRouterApiKey) {
      try {
        // Use OpenRouter for real embeddings
        const embeddings = await this.fetchOpenRouterEmbeddings(
          openRouterApiKey,
          [
            definition.name,
            definition.description,
            definition.content?.slice(0, 8000) || definition.description,
          ]
        );

        return {
          nameEmbedding: embeddings[0] || this.generateHashEmbedding(definition.name),
          descriptionEmbedding: embeddings[1] || this.generateHashEmbedding(definition.description),
          contentEmbedding: embeddings[2] || this.generateHashEmbedding(definition.content || ''),
        };
      } catch (error) {
        logger.warn('OpenRouter embeddings failed, using hash fallback', {
          error: error instanceof Error ? error.message : String(error),
          skillName: definition.name,
        });
      }
    }

    // Fallback: Generate deterministic hash-based embeddings
    // These provide consistent but lower quality semantic matching
    return {
      nameEmbedding: this.generateHashEmbedding(definition.name),
      descriptionEmbedding: this.generateHashEmbedding(definition.description),
      contentEmbedding: this.generateHashEmbedding(definition.content || definition.description),
    };
  }

  /**
   * Fetch embeddings from OpenRouter API
   */
  private async fetchOpenRouterEmbeddings(
    apiKey: string,
    texts: string[]
  ): Promise<number[][]> {
    const response = await fetch('https://openrouter.ai/api/v1/embeddings', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${apiKey}`,
        'HTTP-Referer': 'https://adverant.ai',
        'X-Title': 'EE Design Partner',
      },
      body: JSON.stringify({
        model: 'text-embedding-3-small',
        input: texts,
      }),
    });

    if (!response.ok) {
      const errorText = await response.text();
      throw new Error(`OpenRouter API error: ${response.status} - ${errorText}`);
    }

    const data = await response.json() as {
      data: Array<{ embedding: number[]; index: number }>;
    };

    // Sort by index to maintain order
    const sorted = data.data.sort((a, b) => a.index - b.index);
    return sorted.map((d) => d.embedding);
  }

  /**
   * Generate deterministic hash-based embedding for a text
   *
   * Creates a 384-dimensional vector based on text content.
   * While not as semantically meaningful as neural embeddings,
   * this provides consistent, reproducible vectors for basic matching.
   */
  private generateHashEmbedding(text: string, dimensions: number = 384): number[] {
    const embedding: number[] = new Array(dimensions).fill(0);
    const normalizedText = text.toLowerCase().trim();

    if (!normalizedText) {
      return embedding;
    }

    // Use multiple hash functions for better distribution
    const words = normalizedText.split(/\s+/);
    const chars = normalizedText.split('');

    // Word-level features
    for (let i = 0; i < words.length; i++) {
      const word = words[i];
      const wordHash = this.hashString(word);
      const idx = Math.abs(wordHash) % dimensions;
      embedding[idx] += 1.0 / (1 + Math.log(i + 1));

      // Add bigram features
      if (i < words.length - 1) {
        const bigram = word + ' ' + words[i + 1];
        const bigramHash = this.hashString(bigram);
        const bigramIdx = Math.abs(bigramHash) % dimensions;
        embedding[bigramIdx] += 0.5 / (1 + Math.log(i + 1));
      }
    }

    // Character n-gram features
    for (let i = 0; i < chars.length - 2; i++) {
      const trigram = chars.slice(i, i + 3).join('');
      const trigramHash = this.hashString(trigram);
      const idx = (Math.abs(trigramHash) % (dimensions / 2)) + dimensions / 2;
      embedding[idx] += 0.1;
    }

    // Normalize to unit length
    const magnitude = Math.sqrt(embedding.reduce((sum, val) => sum + val * val, 0));
    if (magnitude > 0) {
      for (let i = 0; i < dimensions; i++) {
        embedding[i] /= magnitude;
      }
    }

    return embedding;
  }

  /**
   * Simple string hash function (djb2 algorithm)
   */
  private hashString(str: string): number {
    let hash = 5381;
    for (let i = 0; i < str.length; i++) {
      hash = ((hash << 5) + hash) + str.charCodeAt(i);
      hash = hash & hash; // Convert to 32-bit integer
    }
    return hash;
  }

  /**
   * Search for skills by query
   */
  async searchSkills(query: string, options?: {
    limit?: number;
    category?: string;
    phase?: string;
  }): Promise<SkillSearchResult[]> {
    try {
      const response = await this.callApi('POST', '/api/skills/search', {
        query,
        limit: options?.limit || 10,
        filters: {
          category: options?.category,
          phase: options?.phase,
          plugin: 'ee-design-partner'
        }
      });

      if (response.success) {
        return (response.data as { results: SkillSearchResult[] }).results;
      }

      return [];

    } catch (error) {
      logger.error('Failed to search skills', error instanceof Error ? error : undefined);
      return [];
    }
  }

  /**
   * Get a specific skill by name
   */
  async getSkill(name: string): Promise<Skill | null> {
    // Check local cache first
    if (this.registeredSkills.has(name)) {
      return this.registeredSkills.get(name)!;
    }

    try {
      const response = await this.callApi('GET', `/api/skills/${name}`);

      if (response.success) {
        const skill = response.data as Skill;
        this.registeredSkills.set(skill.name, skill);
        return skill;
      }

      return null;

    } catch (error) {
      logger.error('Failed to get skill', error instanceof Error ? error : undefined, { name });
      return null;
    }
  }

  /**
   * Execute a skill
   */
  async executeSkill(request: SkillExecutionRequest): Promise<SkillExecutionResponse> {
    const executionId = uuidv4();

    try {
      this.emit('skill:execution:start', { executionId, request });

      const response = await this.callApi('POST', '/api/skills/execute', {
        executionId,
        skillName: request.skillName,
        capability: request.capability,
        parameters: request.parameters,
        context: request.context
      });

      if (response.success) {
        const data = response.data as {
          status: SkillExecutionResponse['status'];
          result?: unknown;
          startedAt?: string;
          completedAt?: string;
        };
        const execution: SkillExecutionResponse = {
          executionId,
          status: data.status,
          result: data.result,
          startedAt: data.startedAt,
          completedAt: data.completedAt
        };

        this.emit('skill:execution:complete', { execution });
        return execution;
      }

      return {
        executionId,
        status: 'failed',
        error: response.error?.message || 'Execution failed'
      };

    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Unknown error';

      this.emit('skill:execution:error', { executionId, error: errorMessage });

      return {
        executionId,
        status: 'failed',
        error: errorMessage
      };
    }
  }

  /**
   * Get all registered EE Design Partner skills
   */
  getRegisteredSkills(): Skill[] {
    return Array.from(this.registeredSkills.values());
  }

  /**
   * Get skills by category/phase
   */
  getSkillsByPhase(phase: string): Skill[] {
    const phaseSkillMap: Record<string, string[]> = {
      'ideation': ['research-paper', 'patent-search', 'market-analysis', 'requirements-gen'],
      'architecture': ['ee-architecture', 'component-select', 'bom-optimize', 'power-budget'],
      'schematic': ['schematic-gen', 'schematic-review', 'netlist-gen'],
      'simulation': ['simulate-spice', 'simulate-thermal', 'simulate-si', 'simulate-rf', 'simulate-emc', 'simulate-stress', 'simulate-reliability'],
      'pcb_layout': ['pcb-layout', 'pcb-review', 'stackup-design', 'via-optimize'],
      'manufacturing': ['gerber-gen', 'dfm-check', 'vendor-quote', 'panelize'],
      'firmware': ['firmware-gen', 'hal-gen', 'driver-gen', 'rtos-config', 'build-setup'],
      'testing': ['test-gen', 'hil-setup', 'test-procedure', 'coverage-analysis'],
      'production': ['manufacture', 'assembly-guide', 'quality-check', 'traceability'],
      'field_support': ['debug-assist', 'service-manual', 'rma-process', 'firmware-update']
    };

    const skillNames = phaseSkillMap[phase] || [];
    return skillNames
      .map(name => this.registeredSkills.get(name))
      .filter((s): s is Skill => s !== undefined);
  }

  /**
   * Sync skills with the Skills Engine
   */
  private async syncSkills(): Promise<void> {
    try {
      const response = await this.callApi('GET', '/api/skills/sync', {
        plugin: 'ee-design-partner'
      });

      if (response.success && response.data) {
        // Update local cache
        const skills = (response.data as { skills?: Skill[] }).skills || [];
        for (const skill of skills) {
          this.registeredSkills.set(skill.name, skill);
        }

        this.emit('skills:synced', { count: this.registeredSkills.size });
      }

    } catch (error) {
      logger.error('Failed to sync skills', error instanceof Error ? error : undefined);
    }
  }

  /**
   * Start periodic sync
   */
  private startSync(): void {
    if (this.syncTimer) {
      clearInterval(this.syncTimer);
    }

    this.syncTimer = setInterval(() => {
      this.syncSkills();
    }, this.config.syncInterval);
  }

  /**
   * Stop periodic sync
   */
  stopSync(): void {
    if (this.syncTimer) {
      clearInterval(this.syncTimer);
      this.syncTimer = undefined;
    }
  }

  /**
   * Make API call to Skills Engine
   */
  private async callApi(
    method: 'GET' | 'POST' | 'PUT' | 'DELETE',
    endpoint: string,
    body?: unknown
  ): Promise<{
    success: boolean;
    data?: unknown;
    error?: { code: string; message: string };
  }> {
    try {
      const url = `${this.config.apiUrl}${endpoint}`;
      const headers: Record<string, string> = {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${this.config.apiKey}`
      };

      const options: RequestInit = {
        method,
        headers
      };

      if (body && method !== 'GET') {
        options.body = JSON.stringify(body);
      }

      const response = await fetch(url, options);
      const data = await response.json() as { error?: { code?: string; message?: string } };

      if (!response.ok) {
        return {
          success: false,
          error: {
            code: data.error?.code || 'API_ERROR',
            message: data.error?.message || `HTTP ${response.status}`
          }
        };
      }

      return {
        success: true,
        data
      };

    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Unknown error';
      return {
        success: false,
        error: {
          code: 'NETWORK_ERROR',
          message: errorMessage
        }
      };
    }
  }

  /**
   * Cleanup
   */
  async shutdown(): Promise<void> {
    this.stopSync();
    this.registeredSkills.clear();
    this.emit('shutdown');
  }
}

export default SkillsEngineClient;
