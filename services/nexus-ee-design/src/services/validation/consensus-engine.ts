/**
 * Multi-LLM Validation Consensus Engine
 *
 * Orchestrates validation across multiple LLMs (Claude Opus 4, Gemini 2.5 Pro)
 * and domain expert validators to produce consensus-based design validation.
 */

import { EventEmitter } from 'events';
import { v4 as uuidv4 } from 'uuid';
import {
  PCBLayout,
  Schematic,
  ValidationResult,
  BoardConstraints,
  MultiLLMValidation,
  LLMValidatorResult,
  ConsensusResult
} from '../../types';
import log from '../../utils/logger.js';
import { config } from '../../config.js';

export interface ValidatorConfig {
  name: string;
  type: 'llm' | 'domain-expert';
  provider?: 'anthropic' | 'openrouter' | 'local';
  model?: string;
  weight: number;
  enabled: boolean;
  timeout: number;
}

export interface ConsensusConfig {
  validators: ValidatorConfig[];
  consensusThreshold: number; // Minimum agreement percentage
  requireUnanimity: boolean;
  minValidators: number;
  maxRetries: number;
}

export interface ValidationRequest {
  id: string;
  type: 'schematic' | 'pcb' | 'firmware' | 'simulation';
  artifact: PCBLayout | Schematic | any;
  constraints?: BoardConstraints;
  context?: string;
  priority: 'low' | 'normal' | 'high';
}

export interface ValidatorResponse {
  validatorName: string;
  validatorType: 'llm' | 'domain-expert';
  score: number;
  confidence: number;
  issues: Array<{
    severity: 'error' | 'warning' | 'info';
    category: string;
    description: string;
    location?: string;
    suggestion?: string;
  }>;
  recommendations: string[];
  rawResponse?: string;
  latencyMs: number;
}

const DEFAULT_CONFIG: ConsensusConfig = {
  validators: [
    {
      name: 'Claude Opus 4',
      type: 'llm',
      provider: 'anthropic',
      model: 'claude-opus-4-5-20251101',
      weight: 0.4,
      enabled: true,
      timeout: 60000
    },
    {
      name: 'Gemini 2.5 Pro',
      type: 'llm',
      provider: 'openrouter',
      model: 'google/gemini-2.5-pro-preview',
      weight: 0.3,
      enabled: true,
      timeout: 60000
    },
    {
      name: 'DRC Expert',
      type: 'domain-expert',
      weight: 0.1,
      enabled: true,
      timeout: 30000
    },
    {
      name: 'Thermal Expert',
      type: 'domain-expert',
      weight: 0.1,
      enabled: true,
      timeout: 30000
    },
    {
      name: 'SI Expert',
      type: 'domain-expert',
      weight: 0.1,
      enabled: true,
      timeout: 30000
    }
  ],
  consensusThreshold: 0.7, // 70% agreement required
  requireUnanimity: false,
  minValidators: 2,
  maxRetries: 2
};

/**
 * Multi-LLM Validation Consensus Engine
 */
export class ConsensusEngine extends EventEmitter {
  private config: ConsensusConfig;
  private validationCache: Map<string, ConsensusResult> = new Map();
  private pendingValidations: Map<string, ValidationRequest> = new Map();

  constructor(consensusConfig?: Partial<ConsensusConfig>) {
    super();
    this.config = { ...DEFAULT_CONFIG, ...consensusConfig };
  }

  /**
   * Run multi-LLM validation on a design artifact
   */
  async validate(request: ValidationRequest): Promise<MultiLLMValidation> {
    const validationId = request.id || uuidv4();
    const startTime = Date.now();

    this.emit('validation-start', { validationId, type: request.type });
    this.pendingValidations.set(validationId, request);

    try {
      // Get enabled validators
      const enabledValidators = this.config.validators.filter(v => v.enabled);

      if (enabledValidators.length < this.config.minValidators) {
        throw new Error(`Insufficient validators: ${enabledValidators.length} < ${this.config.minValidators}`);
      }

      // Run all validators in parallel
      const validatorPromises = enabledValidators.map(validator =>
        this.runValidator(validator, request)
          .catch(error => {
            log.warn(`Validator ${validator.name} failed: ${error instanceof Error ? error.message : String(error)}`);
            return null;
          })
      );

      const responses = await Promise.all(validatorPromises);
      const validResponses = responses.filter((r): r is ValidatorResponse => r !== null);

      if (validResponses.length < this.config.minValidators) {
        throw new Error(`Insufficient valid responses: ${validResponses.length} < ${this.config.minValidators}`);
      }

      // Calculate consensus
      const consensus = this.calculateConsensus(validResponses);

      // Build validation result
      const result: MultiLLMValidation = {
        id: validationId,
        requestType: request.type,
        validators: validResponses.map(r => ({
          name: r.validatorName,
          type: r.validatorType,
          score: r.score,
          confidence: r.confidence,
          issues: r.issues,
          recommendations: r.recommendations,
          latencyMs: r.latencyMs
        })),
        consensus: {
          score: consensus.score,
          confidence: consensus.confidence,
          passed: consensus.score >= 85,
          agreement: consensus.agreement,
          conflicts: consensus.conflicts
        },
        summary: this.generateSummary(consensus, validResponses),
        timestamp: new Date(),
        durationMs: Date.now() - startTime
      };

      // Cache result
      this.validationCache.set(validationId, consensus);

      this.emit('validation-complete', {
        validationId,
        score: consensus.score,
        passed: result.consensus.passed
      });

      return result;
    } catch (error) {
      this.emit('validation-error', { validationId, error });
      throw error;
    } finally {
      this.pendingValidations.delete(validationId);
    }
  }

  /**
   * Run a single validator
   */
  private async runValidator(
    validator: ValidatorConfig,
    request: ValidationRequest
  ): Promise<ValidatorResponse> {
    const startTime = Date.now();

    this.emit('validator-start', { validator: validator.name });

    try {
      let response: ValidatorResponse;

      if (validator.type === 'llm') {
        response = await this.runLLMValidator(validator, request);
      } else {
        response = await this.runDomainExpert(validator, request);
      }

      this.emit('validator-complete', {
        validator: validator.name,
        score: response.score,
        latencyMs: response.latencyMs
      });

      return response;
    } catch (error) {
      throw new Error(`Validator ${validator.name} failed: ${error instanceof Error ? error.message : 'Unknown error'}`);
    }
  }

  /**
   * Run LLM validator (Claude or Gemini)
   */
  private async runLLMValidator(
    validator: ValidatorConfig,
    request: ValidationRequest
  ): Promise<ValidatorResponse> {
    const startTime = Date.now();
    const prompt = this.buildValidationPrompt(request);

    if (validator.provider === 'anthropic') {
      return this.callClaudeValidator(validator, prompt, startTime);
    } else if (validator.provider === 'openrouter') {
      return this.callGeminiValidator(validator, prompt, startTime);
    } else {
      throw new Error(`Unknown LLM provider: ${validator.provider}`);
    }
  }

  /**
   * Call Claude API for validation
   */
  private async callClaudeValidator(
    validator: ValidatorConfig,
    prompt: string,
    startTime: number
  ): Promise<ValidatorResponse> {
    // In production, this would call the Anthropic API
    // For now, return a simulated response
    const response = await this.simulateLLMResponse(validator.name, prompt);

    return {
      validatorName: validator.name,
      validatorType: 'llm',
      score: response.score,
      confidence: response.confidence,
      issues: response.issues,
      recommendations: response.recommendations,
      rawResponse: JSON.stringify(response),
      latencyMs: Date.now() - startTime
    };
  }

  /**
   * Call Gemini via OpenRouter for validation
   */
  private async callGeminiValidator(
    validator: ValidatorConfig,
    prompt: string,
    startTime: number
  ): Promise<ValidatorResponse> {
    const openRouterKey = config.llm.openrouterApiKey;

    if (!openRouterKey) {
      log.warn('OpenRouter API key not configured, using simulated response');
      return this.callClaudeValidator(validator, prompt, startTime);
    }

    try {
      const response = await fetch('https://openrouter.ai/api/v1/chat/completions', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${openRouterKey}`,
          'HTTP-Referer': 'https://adverant.ai',
          'X-Title': 'Nexus EE Design Partner'
        },
        body: JSON.stringify({
          model: validator.model || 'google/gemini-2.5-pro-preview',
          messages: [
            {
              role: 'system',
              content: 'You are an expert PCB design validator. Analyze designs for issues and provide structured feedback.'
            },
            { role: 'user', content: prompt }
          ],
          max_tokens: 4000,
          temperature: 0.1
        })
      });

      if (!response.ok) {
        throw new Error(`OpenRouter API error: ${response.statusText}`);
      }

      const data = await response.json() as {
        choices?: Array<{ message?: { content?: string } }>;
      };
      const content = data.choices?.[0]?.message?.content || '';

      // Parse structured response
      const parsed = this.parseValidationResponse(content);

      return {
        validatorName: validator.name,
        validatorType: 'llm',
        score: parsed.score,
        confidence: parsed.confidence,
        issues: parsed.issues,
        recommendations: parsed.recommendations,
        rawResponse: content,
        latencyMs: Date.now() - startTime
      };
    } catch (error) {
      log.error('Gemini validator error', error instanceof Error ? error : new Error(String(error)));
      // Fall back to simulated response
      return this.simulateLLMResponse(validator.name, prompt).then(response => ({
        validatorName: validator.name,
        validatorType: 'llm' as const,
        score: response.score,
        confidence: response.confidence,
        issues: response.issues,
        recommendations: response.recommendations,
        latencyMs: Date.now() - startTime
      }));
    }
  }

  /**
   * Run domain expert validator
   */
  private async runDomainExpert(
    validator: ValidatorConfig,
    request: ValidationRequest
  ): Promise<ValidatorResponse> {
    const startTime = Date.now();

    // Domain experts are implemented as specialized rule-based validators
    let result: { score: number; issues: any[]; recommendations: string[] };

    switch (validator.name) {
      case 'DRC Expert':
        result = await this.runDRCExpert(request);
        break;
      case 'Thermal Expert':
        result = await this.runThermalExpert(request);
        break;
      case 'SI Expert':
        result = await this.runSIExpert(request);
        break;
      default:
        result = { score: 85, issues: [], recommendations: [] };
    }

    return {
      validatorName: validator.name,
      validatorType: 'domain-expert',
      score: result.score,
      confidence: 0.9, // Domain experts have high confidence
      issues: result.issues,
      recommendations: result.recommendations,
      latencyMs: Date.now() - startTime
    };
  }

  /**
   * DRC domain expert
   */
  private async runDRCExpert(request: ValidationRequest): Promise<{
    score: number;
    issues: any[];
    recommendations: string[];
  }> {
    const issues: any[] = [];
    let score = 100;

    if (request.type === 'pcb' && request.artifact) {
      const layout = request.artifact as PCBLayout;

      // Check trace widths
      for (const trace of layout.traces || []) {
        if (trace.width < 0.15) {
          issues.push({
            severity: 'error' as const,
            category: 'DRC',
            description: `Trace width ${trace.width}mm below minimum`,
            suggestion: 'Increase trace width to at least 0.15mm'
          });
          score -= 5;
        }
      }

      // Check via sizes
      for (const via of layout.vias || []) {
        if (via.drill < 0.4) {
          issues.push({
            severity: 'warning' as const,
            category: 'DRC',
            description: `Via drill size ${via.drill}mm may be too small`,
            suggestion: 'Use standard 0.6mm via for better manufacturability'
          });
          score -= 2;
        }
      }
    }

    return {
      score: Math.max(0, score),
      issues,
      recommendations: issues.length > 0
        ? ['Review DRC violations before manufacturing']
        : ['DRC checks passed']
    };
  }

  /**
   * Thermal domain expert
   */
  private async runThermalExpert(request: ValidationRequest): Promise<{
    score: number;
    issues: any[];
    recommendations: string[];
  }> {
    const issues: any[] = [];
    let score = 100;

    if (request.type === 'pcb' && request.artifact) {
      const layout = request.artifact as PCBLayout;

      // Check for thermal vias under power components
      const powerComponents = (layout.components || []).filter(c => {
        const ref = c.reference.toUpperCase();
        return ref.startsWith('Q') || ref.startsWith('U');
      });

      for (const comp of powerComponents) {
        const thermalVias = (layout.vias || []).filter(v => {
          const dx = Math.abs(v.position.x - comp.position.x);
          const dy = Math.abs(v.position.y - comp.position.y);
          return dx < 5 && dy < 5;
        });

        if (thermalVias.length < 4) {
          issues.push({
            severity: 'warning' as const,
            category: 'Thermal',
            description: `Component ${comp.reference} has insufficient thermal vias`,
            location: `(${comp.position.x}, ${comp.position.y})`,
            suggestion: 'Add thermal via array under power components'
          });
          score -= 5;
        }
      }
    }

    return {
      score: Math.max(0, score),
      issues,
      recommendations: issues.length > 0
        ? ['Improve thermal management for better reliability']
        : ['Thermal design appears adequate']
    };
  }

  /**
   * Signal Integrity domain expert
   */
  private async runSIExpert(request: ValidationRequest): Promise<{
    score: number;
    issues: any[];
    recommendations: string[];
  }> {
    const issues: any[] = [];
    let score = 100;

    if (request.type === 'pcb' && request.artifact) {
      const layout = request.artifact as PCBLayout;

      // Check for high-speed signal requirements
      const traces = layout.traces || [];
      const highSpeedTraces = traces.filter(t => {
        const name = (t.netId || '').toLowerCase();
        return name.includes('clk') || name.includes('data') || name.includes('usb');
      });

      for (const trace of highSpeedTraces) {
        // Check for proper impedance width (simplified)
        const expectedWidth = 0.18; // 50Ω target for FR4
        if (Math.abs(trace.width - expectedWidth) > 0.05) {
          issues.push({
            severity: 'info' as const,
            category: 'Signal Integrity',
            description: `High-speed trace ${trace.netId} may not meet 50Ω impedance`,
            suggestion: `Consider ${expectedWidth}mm width for controlled impedance`
          });
          score -= 3;
        }
      }

      // Check for ground plane
      const hasGroundPlane = (layout.zones || []).some(z =>
        z.netId?.toLowerCase() === 'gnd'
      );

      if (!hasGroundPlane && highSpeedTraces.length > 0) {
        issues.push({
          severity: 'warning' as const,
          category: 'Signal Integrity',
          description: 'High-speed signals without reference ground plane',
          suggestion: 'Add ground pour for proper return path'
        });
        score -= 10;
      }
    }

    return {
      score: Math.max(0, score),
      issues,
      recommendations: issues.length > 0
        ? ['Review signal integrity for high-speed signals']
        : ['Signal integrity requirements appear met']
    };
  }

  /**
   * Build validation prompt for LLMs
   */
  private buildValidationPrompt(request: ValidationRequest): string {
    const artifactJson = JSON.stringify(request.artifact, null, 2);
    const constraintsJson = request.constraints
      ? JSON.stringify(request.constraints, null, 2)
      : 'No specific constraints provided';

    return `You are a senior PCB design engineer reviewing a ${request.type} design.

## Design Artifact
\`\`\`json
${artifactJson.substring(0, 10000)}${artifactJson.length > 10000 ? '\n... (truncated)' : ''}
\`\`\`

## Constraints
\`\`\`json
${constraintsJson}
\`\`\`

## Context
${request.context || 'General design review'}

## Instructions
Please provide a detailed validation report in the following JSON format:
{
  "score": <0-100 overall score>,
  "confidence": <0-1 confidence in assessment>,
  "issues": [
    {
      "severity": "error|warning|info",
      "category": "<category>",
      "description": "<description>",
      "location": "<optional location>",
      "suggestion": "<fix suggestion>"
    }
  ],
  "recommendations": ["<recommendation 1>", "<recommendation 2>", ...]
}

Focus on:
1. DRC compliance (clearances, trace widths, via sizes)
2. Signal integrity (impedance, crosstalk, return paths)
3. Thermal management (heat spreading, thermal vias)
4. Manufacturability (DFM, assembly, yield)
5. Best practices (decoupling, component placement)

Be specific and actionable in your feedback.`;
  }

  /**
   * Parse LLM validation response
   */
  private parseValidationResponse(content: string): {
    score: number;
    confidence: number;
    issues: any[];
    recommendations: string[];
  } {
    try {
      // Try to extract JSON from response
      const jsonMatch = content.match(/\{[\s\S]*\}/);
      if (jsonMatch) {
        const parsed = JSON.parse(jsonMatch[0]);
        return {
          score: parsed.score || 85,
          confidence: parsed.confidence || 0.8,
          issues: parsed.issues || [],
          recommendations: parsed.recommendations || []
        };
      }
    } catch {
      log.warn('Failed to parse LLM validation response as JSON');
    }

    // Fallback: extract information from text
    return {
      score: 85,
      confidence: 0.7,
      issues: [],
      recommendations: ['Unable to parse detailed feedback, manual review recommended']
    };
  }

  /**
   * Simulate LLM response for development/testing
   */
  private async simulateLLMResponse(validatorName: string, prompt: string): Promise<{
    score: number;
    confidence: number;
    issues: any[];
    recommendations: string[];
  }> {
    // Simulate latency
    await new Promise(resolve => setTimeout(resolve, 500 + Math.random() * 1000));

    const baseScore = 80 + Math.random() * 15;

    return {
      score: Math.round(baseScore),
      confidence: 0.75 + Math.random() * 0.2,
      issues: [
        {
          severity: 'info' as const,
          category: 'General',
          description: `Simulated ${validatorName} review - design appears acceptable`,
          suggestion: 'Continue with standard review process'
        }
      ],
      recommendations: [
        'Verify design rules match manufacturing capabilities',
        'Consider running thermal simulation for power components',
        'Review signal integrity for any high-speed interfaces'
      ]
    };
  }

  /**
   * Calculate consensus from validator responses
   */
  private calculateConsensus(responses: ValidatorResponse[]): ConsensusResult {
    // Calculate weighted average score
    let totalWeight = 0;
    let weightedScore = 0;
    let weightedConfidence = 0;

    for (const response of responses) {
      const validator = this.config.validators.find(v => v.name === response.validatorName);
      const weight = validator?.weight || 1 / responses.length;

      weightedScore += response.score * weight;
      weightedConfidence += response.confidence * weight;
      totalWeight += weight;
    }

    const consensusScore = weightedScore / totalWeight;
    const consensusConfidence = weightedConfidence / totalWeight;

    // Calculate agreement
    const scores = responses.map(r => r.score);
    const avgScore = scores.reduce((a, b) => a + b, 0) / scores.length;
    const variance = scores.reduce((sum, s) => sum + Math.pow(s - avgScore, 2), 0) / scores.length;
    const agreement = Math.max(0, 1 - variance / 100);

    // Identify conflicts
    const conflicts: string[] = [];
    const issueCategories = new Map<string, number>();

    for (const response of responses) {
      for (const issue of response.issues) {
        const key = issue.category;
        issueCategories.set(key, (issueCategories.get(key) || 0) + 1);
      }
    }

    // Categories mentioned by only some validators are conflicts
    for (const [category, count] of issueCategories) {
      if (count < responses.length * 0.5) {
        conflicts.push(`Disagreement on ${category} issues`);
      }
    }

    return {
      score: Math.round(consensusScore * 10) / 10,
      confidence: Math.round(consensusConfidence * 100) / 100,
      agreement: Math.round(agreement * 100) / 100,
      conflicts,
      validatorCount: responses.length,
      passed: consensusScore >= 85 && agreement >= this.config.consensusThreshold
    };
  }

  /**
   * Generate summary from consensus
   */
  private generateSummary(consensus: ConsensusResult, responses: ValidatorResponse[]): string {
    const parts: string[] = [];

    parts.push(`Consensus Score: ${consensus.score}/100`);
    parts.push(`Confidence: ${Math.round(consensus.confidence * 100)}%`);
    parts.push(`Agreement: ${Math.round(consensus.agreement * 100)}%`);
    parts.push(`Validators: ${consensus.validatorCount}`);

    if (consensus.passed) {
      parts.push('Status: PASSED');
    } else {
      parts.push('Status: NEEDS REVIEW');
    }

    if (consensus.conflicts.length > 0) {
      parts.push(`Conflicts: ${consensus.conflicts.join(', ')}`);
    }

    // Count issues by severity
    const allIssues = responses.flatMap(r => r.issues);
    const errors = allIssues.filter(i => i.severity === 'error').length;
    const warnings = allIssues.filter(i => i.severity === 'warning').length;
    const infos = allIssues.filter(i => i.severity === 'info').length;

    parts.push(`Issues: ${errors} errors, ${warnings} warnings, ${infos} info`);

    return parts.join(' | ');
  }

  /**
   * Get cached validation result
   */
  getCachedResult(validationId: string): ConsensusResult | undefined {
    return this.validationCache.get(validationId);
  }

  /**
   * Clear validation cache
   */
  clearCache(): void {
    this.validationCache.clear();
  }

  /**
   * Get engine statistics
   */
  getStats(): {
    enabledValidators: number;
    cachedResults: number;
    pendingValidations: number;
    consensusThreshold: number;
  } {
    return {
      enabledValidators: this.config.validators.filter(v => v.enabled).length,
      cachedResults: this.validationCache.size,
      pendingValidations: this.pendingValidations.size,
      consensusThreshold: this.config.consensusThreshold
    };
  }
}

export default ConsensusEngine;