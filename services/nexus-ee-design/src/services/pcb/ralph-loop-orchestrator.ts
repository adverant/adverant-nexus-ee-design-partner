/**
 * Ralph Loop Orchestrator
 *
 * Multi-agent PCB layout competition and iterative refinement system.
 *
 * Tournament Structure:
 * - Rounds 1-30: All 5 agents compete → top 3 advance
 * - Rounds 31-60: Top 3 compete → top 2 advance
 * - Rounds 61-90: Top 2 compete → top 1 advances
 * - Rounds 91-100: Winner refines with expert feedback
 */

import { EventEmitter } from 'events';
import type { Server as SocketIOServer } from 'socket.io';
import { v4 as uuidv4 } from 'uuid';
import { log } from '../../utils/logger.js';
import { config } from '../../config.js';
import type {
  PCBLayout,
  Schematic,
  ValidationResults,
  LayoutStrategy,
  AgentResult,
  TournamentResult,
} from '../../types/index.js';

// ============================================================================
// Types
// ============================================================================

export interface RalphLoopConfig {
  maxIterations: number;
  targetScore: number;
  convergenceThreshold: number;
  timeout: number;
  agents: LayoutStrategy[];
  verbose: boolean;
  outputDir: string;
}

export interface LayoutContext {
  schematic: Schematic;
  constraints: BoardConstraints;
  iteration: number;
  previousBestScore: number;
  expertFeedback: ExpertFeedback[];
}

export interface BoardConstraints {
  width: number;
  height: number;
  layers: number;
  minTraceWidth: number;
  minClearance: number;
  minViaDiameter: number;
  minViaDrill: number;
  copperWeight: string;
  material: string;
}

export interface ExpertFeedback {
  expert: string;
  message: string;
  severity: 'critical' | 'warning' | 'suggestion';
  suggestion?: string;
  target?: string;
  location?: { x: number; y: number };
}

export interface AgentRanking {
  agent: LayoutStrategy;
  score: number;
  layout: PCBLayout | null;
  violations: number;
}

export interface RalphLoopState {
  id: string;
  iteration: number;
  phase: 1 | 2 | 3 | 4;
  rankings: AgentRanking[];
  bestLayout: PCBLayout | null;
  bestScore: number;
  validationHistory: ValidationResults[];
  feedbackLog: ExpertFeedback[];
  startTime: Date;
  lastUpdate: Date;
  converged: boolean;
  convergenceReason?: 'perfect_score' | 'good_enough' | 'plateau' | 'timeout';
}

export interface RalphLoopResult {
  success: boolean;
  layout: PCBLayout | null;
  score: number;
  winningAgent: LayoutStrategy;
  iterations: number;
  validation: ValidationResults | null;
  runtime: number;
  convergence: {
    converged: boolean;
    reason: string;
    iteration: number;
  };
}

// ============================================================================
// Layout Agent Interface
// ============================================================================

export interface ILayoutAgent {
  readonly strategy: LayoutStrategy;
  readonly priority: string[];

  generateLayout(context: LayoutContext): Promise<AgentResult>;
  refine(context: LayoutContext, feedback: ExpertFeedback[]): Promise<AgentResult>;
}

// ============================================================================
// Validation Agent Interface
// ============================================================================

export interface IValidationAgent {
  readonly domain: string;
  readonly weight: number;

  validate(layout: PCBLayout, constraints: BoardConstraints): Promise<DomainValidationResult>;
  generateFeedback(result: DomainValidationResult): ExpertFeedback[];
}

export interface DomainValidationResult {
  domain: string;
  passed: boolean;
  score: number;
  violations: ValidationViolation[];
  executionTime: number;
}

export interface ValidationViolation {
  type: string;
  severity: 'critical' | 'error' | 'warning' | 'info';
  message: string;
  location?: { x: number; y: number; layer?: string };
  target?: string;
  suggestion?: string;
}

// ============================================================================
// Ralph Loop Orchestrator
// ============================================================================

export class RalphLoopOrchestrator extends EventEmitter {
  private readonly config: RalphLoopConfig;
  private state: RalphLoopState;
  private agents: Map<LayoutStrategy, ILayoutAgent> = new Map();
  private validators: IValidationAgent[] = [];
  private io?: SocketIOServer;
  private projectId?: string;

  constructor(configOverrides: Partial<RalphLoopConfig> = {}) {
    super();

    this.config = {
      maxIterations: configOverrides.maxIterations ?? config.layout.maxIterations,
      targetScore: configOverrides.targetScore ?? config.layout.targetScore,
      convergenceThreshold: configOverrides.convergenceThreshold ?? config.layout.convergenceThreshold,
      timeout: configOverrides.timeout ?? 4 * 60 * 60 * 1000, // 4 hours
      agents: (configOverrides.agents ?? config.layout.enabledAgents) as LayoutStrategy[],
      verbose: configOverrides.verbose ?? false,
      outputDir: configOverrides.outputDir ?? config.storage.outputDir,
    };

    this.state = this.initializeState();
  }

  /**
   * Set Socket.IO server for real-time updates
   */
  setSocketIO(io: SocketIOServer, projectId: string): void {
    this.io = io;
    this.projectId = projectId;
  }

  /**
   * Register a layout agent
   */
  registerAgent(agent: ILayoutAgent): void {
    this.agents.set(agent.strategy, agent);
  }

  /**
   * Register a validation agent
   */
  registerValidator(validator: IValidationAgent): void {
    this.validators.push(validator);
  }

  /**
   * Run the Ralph Loop tournament
   */
  async run(schematic: Schematic, constraints: BoardConstraints): Promise<RalphLoopResult> {
    this.state = this.initializeState();
    const startTime = Date.now();

    this.log('🚀 Ralph Loop starting...');
    this.log(`   Max iterations: ${this.config.maxIterations}`);
    this.log(`   Target score: ${this.config.targetScore}`);
    this.log(`   Competing agents: ${this.agents.size}`);
    this.log(`   Validation domains: ${this.validators.length}`);

    this.emitProgress('started', {
      maxIterations: this.config.maxIterations,
      agents: Array.from(this.agents.keys()),
    });

    try {
      const context: LayoutContext = {
        schematic,
        constraints,
        iteration: 0,
        previousBestScore: 0,
        expertFeedback: [],
      };

      // Phase 1: Rounds 1-30 (All agents)
      await this.runPhase(1, context, 1, 30, Array.from(this.agents.values()));

      if (this.checkConvergence()) {
        return this.finalizeResult(startTime);
      }

      // Get top 3 agents
      const top3 = this.getTopAgents(3);
      this.log(`\n✅ Top 3 advancing: ${top3.map((a) => a.strategy).join(', ')}`);

      // Phase 2: Rounds 31-60 (Top 3)
      await this.runPhase(2, context, 31, 60, top3);

      if (this.checkConvergence()) {
        return this.finalizeResult(startTime);
      }

      // Get top 2 agents
      const top2 = this.getTopAgents(2);
      this.log(`\n✅ Top 2 advancing: ${top2.map((a) => a.strategy).join(', ')}`);

      // Phase 3: Rounds 61-90 (Top 2)
      await this.runPhase(3, context, 61, 90, top2);

      if (this.checkConvergence()) {
        return this.finalizeResult(startTime);
      }

      // Get winner
      const winner = this.getTopAgents(1)[0];
      if (!winner) {
        throw new Error('No winner found');
      }

      this.log(`\n🏆 Winner: ${winner.strategy}`);

      // Phase 4: Rounds 91-100 (Winner refinement)
      await this.runPhase(4, context, 91, 100, [winner]);

      return this.finalizeResult(startTime);
    } catch (error) {
      this.log(`\n❌ Error: ${error instanceof Error ? error.message : String(error)}`);
      throw error;
    }
  }

  /**
   * Run a phase of the tournament
   */
  private async runPhase(
    phase: 1 | 2 | 3 | 4,
    context: LayoutContext,
    startRound: number,
    endRound: number,
    agents: ILayoutAgent[]
  ): Promise<void> {
    this.state.phase = phase;
    const phaseNames = ['', 'All agents', 'Top 3', 'Top 2', 'Winner refinement'];

    this.log(`\n📊 Phase ${phase}: ${phaseNames[phase]} (Rounds ${startRound}-${endRound})`);

    this.emitProgress('phase_started', {
      phase,
      name: phaseNames[phase],
      startRound,
      endRound,
      agents: agents.map((a) => a.strategy),
    });

    for (let iteration = startRound; iteration <= endRound; iteration++) {
      if (this.checkTimeout()) {
        this.log('\n⏰ Timeout reached');
        break;
      }

      this.state.iteration = iteration;
      context.iteration = iteration;
      context.previousBestScore = this.state.bestScore;
      context.expertFeedback = this.generateExpertFeedback();

      this.log(`\nIteration ${iteration}/${this.config.maxIterations}:`);

      // Run agents (generate or refine based on phase)
      const results = phase === 1
        ? await this.runAgentsGenerate(agents, context)
        : await this.runAgentsRefine(agents, context);

      // Validate and score
      await this.validateAndScore(results);

      // Update rankings
      this.updateRankings(results);

      // Emit progress
      this.emitProgress('iteration', {
        iteration,
        phase,
        bestScore: this.state.bestScore,
        rankings: this.state.rankings.slice(0, 3),
      });

      // Check convergence
      if (this.checkConvergence()) {
        this.log(`\n✅ Converged at iteration ${iteration}`);
        break;
      }
    }
  }

  /**
   * Run agents to generate initial layouts
   */
  private async runAgentsGenerate(
    agents: ILayoutAgent[],
    context: LayoutContext
  ): Promise<AgentResult[]> {
    const results: AgentResult[] = [];

    for (const agent of agents) {
      try {
        this.log(`  Running ${agent.strategy}...`);
        const result = await agent.generateLayout(context);
        results.push(result);
        this.log(`    ✓ Score: ${result.score.toFixed(1)}/100`);
      } catch (error) {
        this.log(`    ✗ Failed: ${error instanceof Error ? error.message : String(error)}`);
        results.push(this.createFailedResult(agent.strategy));
      }
    }

    return results;
  }

  /**
   * Run agents to refine layouts
   */
  private async runAgentsRefine(
    agents: ILayoutAgent[],
    context: LayoutContext
  ): Promise<AgentResult[]> {
    const results: AgentResult[] = [];

    for (const agent of agents) {
      try {
        this.log(`  Refining ${agent.strategy}...`);
        const result = await agent.refine(context, context.expertFeedback);
        results.push(result);
        this.log(`    ✓ Score: ${result.score.toFixed(1)}/100`);
      } catch (error) {
        this.log(`    ✗ Failed: ${error instanceof Error ? error.message : String(error)}`);
        results.push(this.createFailedResult(agent.strategy));
      }
    }

    return results;
  }

  /**
   * Validate layouts and calculate scores
   */
  private async validateAndScore(results: AgentResult[]): Promise<void> {
    for (const result of results) {
      if (!result.layout) continue;

      // Run all validators
      const domainResults = await Promise.all(
        this.validators.map((v) =>
          v.validate(result.layout!, {} as BoardConstraints).catch(() => ({
            domain: v.domain,
            passed: false,
            score: 0,
            violations: [],
            executionTime: 0,
          }))
        )
      );

      // Calculate weighted score
      let totalScore = 0;
      let totalWeight = 0;

      for (const dr of domainResults) {
        const validator = this.validators.find((v) => v.domain === dr.domain);
        if (validator) {
          totalScore += dr.score * validator.weight;
          totalWeight += validator.weight;
        }
      }

      result.score = totalWeight > 0 ? totalScore / totalWeight : 0;

      // Count violations
      result.violations = domainResults.flatMap((dr) => dr.violations);
    }
  }

  /**
   * Update agent rankings
   */
  private updateRankings(results: AgentResult[]): void {
    for (const result of results) {
      const existingIdx = this.state.rankings.findIndex(
        (r) => r.agent === result.strategy
      );

      const ranking: AgentRanking = {
        agent: result.strategy,
        score: result.score,
        layout: result.layout,
        violations: result.violations?.length ?? 0,
      };

      if (existingIdx >= 0) {
        this.state.rankings[existingIdx] = ranking;
      } else {
        this.state.rankings.push(ranking);
      }
    }

    // Sort by score descending
    this.state.rankings.sort((a, b) => b.score - a.score);

    // Update best if improved
    const best = this.state.rankings[0];
    if (best && best.score > this.state.bestScore) {
      this.state.bestLayout = best.layout;
      this.state.bestScore = best.score;
      this.log(`  🎯 New best: ${best.score.toFixed(1)}/100 (${best.agent})`);
    }

    this.state.lastUpdate = new Date();
  }

  /**
   * Generate expert feedback from validation results
   */
  private generateExpertFeedback(): ExpertFeedback[] {
    const feedback: ExpertFeedback[] = [];

    // Generate feedback from each validator
    for (const validator of this.validators) {
      if (this.state.validationHistory.length > 0) {
        const latest = this.state.validationHistory[this.state.validationHistory.length - 1];
        const domainResult = latest?.domains?.find((d) => d.name === validator.domain);

        if (domainResult) {
          const domainFeedback = validator.generateFeedback({
            domain: validator.domain,
            passed: domainResult.passed,
            score: domainResult.score,
            violations: domainResult.violations.map((v) => ({
              type: v.code,
              severity: v.severity,
              message: v.message,
              suggestion: v.suggestion,
            })),
            executionTime: 0,
          });
          feedback.push(...domainFeedback);
        }
      }
    }

    this.state.feedbackLog.push(...feedback);
    return feedback;
  }

  /**
   * Get top N agents by score
   */
  private getTopAgents(count: number): ILayoutAgent[] {
    return this.state.rankings
      .slice(0, count)
      .map((r) => this.agents.get(r.agent))
      .filter((a): a is ILayoutAgent => a !== undefined);
  }

  /**
   * Check if convergence criteria met
   */
  private checkConvergence(): boolean {
    // Perfect score
    if (this.state.bestScore >= 100) {
      this.state.converged = true;
      this.state.convergenceReason = 'perfect_score';
      return true;
    }

    // Good enough with plateau
    if (this.state.bestScore >= this.config.targetScore) {
      if (this.state.validationHistory.length >= 10) {
        const recentScores = this.state.validationHistory
          .slice(-10)
          .map((v) => v.score);
        const maxRecent = Math.max(...recentScores);
        const minRecent = Math.min(...recentScores);

        if (maxRecent - minRecent < this.config.convergenceThreshold) {
          this.state.converged = true;
          this.state.convergenceReason = 'good_enough';
          return true;
        }
      }
    }

    // Plateau detection
    if (this.state.validationHistory.length >= 10) {
      const recentScores = this.state.validationHistory
        .slice(-10)
        .map((v) => v.score);
      const maxRecent = Math.max(...recentScores);
      const minRecent = Math.min(...recentScores);

      if (maxRecent - minRecent < this.config.convergenceThreshold) {
        this.state.converged = true;
        this.state.convergenceReason = 'plateau';
        return true;
      }
    }

    return false;
  }

  /**
   * Check if timeout reached
   */
  private checkTimeout(): boolean {
    const elapsed = Date.now() - this.state.startTime.getTime();
    if (elapsed >= this.config.timeout) {
      this.state.converged = true;
      this.state.convergenceReason = 'timeout';
      return true;
    }
    return false;
  }

  /**
   * Finalize and return result
   */
  private finalizeResult(startTime: number): RalphLoopResult {
    const runtime = Date.now() - startTime;
    const winner = this.state.rankings[0];

    this.log('\n' + '='.repeat(60));
    this.log('🏁 Ralph Loop Complete!');
    this.log('='.repeat(60));
    this.log(`   Final Score: ${this.state.bestScore.toFixed(1)}/100`);
    this.log(`   Winning Agent: ${winner?.agent ?? 'none'}`);
    this.log(`   Iterations: ${this.state.iteration}`);
    this.log(`   Runtime: ${this.formatRuntime(runtime)}`);
    this.log('='.repeat(60));

    const result: RalphLoopResult = {
      success: this.state.bestScore >= this.config.targetScore,
      layout: this.state.bestLayout,
      score: this.state.bestScore,
      winningAgent: winner?.agent ?? 'conservative',
      iterations: this.state.iteration,
      validation: this.state.validationHistory[this.state.validationHistory.length - 1] ?? null,
      runtime,
      convergence: {
        converged: this.state.converged,
        reason: this.state.convergenceReason ?? 'max_iterations',
        iteration: this.state.iteration,
      },
    };

    this.emitProgress('completed', result);
    return result;
  }

  /**
   * Initialize state
   */
  private initializeState(): RalphLoopState {
    return {
      id: uuidv4(),
      iteration: 0,
      phase: 1,
      rankings: [],
      bestLayout: null,
      bestScore: 0,
      validationHistory: [],
      feedbackLog: [],
      startTime: new Date(),
      lastUpdate: new Date(),
      converged: false,
    };
  }

  /**
   * Create a failed result for an agent
   */
  private createFailedResult(strategy: LayoutStrategy): AgentResult {
    return {
      agentId: strategy,
      strategy,
      score: 0,
      violations: [],
      metadata: {},
      iteration: this.state.iteration,
      timestamp: new Date().toISOString(),
    };
  }

  /**
   * Format runtime for display
   */
  private formatRuntime(ms: number): string {
    const hours = Math.floor(ms / (1000 * 60 * 60));
    const minutes = Math.floor((ms % (1000 * 60 * 60)) / (1000 * 60));
    const seconds = Math.floor((ms % (1000 * 60)) / 1000);

    if (hours > 0) return `${hours}h ${minutes}m ${seconds}s`;
    if (minutes > 0) return `${minutes}m ${seconds}s`;
    return `${seconds}s`;
  }

  /**
   * Log message
   */
  private log(message: string): void {
    if (this.config.verbose) {
      console.log(message);
    }
    log.debug(message, { component: 'ralph-loop' });
  }

  /**
   * Emit progress via Socket.IO
   */
  private emitProgress(event: string, data: unknown): void {
    if (this.io && this.projectId) {
      this.io.to(`project:${this.projectId}`).emit(`layout:${event}`, {
        loopId: this.state.id,
        ...data,
      });
    }
    this.emit(event, data);
  }

  /**
   * Get current state (for monitoring)
   */
  getState(): RalphLoopState {
    return { ...this.state };
  }
}

export default RalphLoopOrchestrator;