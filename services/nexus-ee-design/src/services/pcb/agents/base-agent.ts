/**
 * Base Layout Agent
 *
 * Abstract base class for all PCB layout agents in the Ralph Loop tournament.
 * Each agent implements a specific layout strategy and competes against others.
 */

import { EventEmitter } from 'events';
import { v4 as uuidv4 } from 'uuid';
import {
  Schematic,
  PCBLayout,
  BoardConstraints,
  ValidationResult,
  ComponentPlacement,
  Trace,
  Via
} from '../../../types';

export interface AgentConfig {
  name: string;
  strategy: AgentStrategy;
  weights: StrategyWeights;
  parameters: AgentParameters;
}

export type AgentStrategy =
  | 'conservative'
  | 'aggressive-compact'
  | 'thermal-optimized'
  | 'emi-optimized'
  | 'dfm-optimized';

export interface StrategyWeights {
  area: number;           // Board area minimization (0-1)
  thermal: number;        // Thermal performance (0-1)
  signal: number;         // Signal integrity (0-1)
  emi: number;            // EMI compliance (0-1)
  dfm: number;            // Design for manufacturing (0-1)
  routing: number;        // Routing efficiency (0-1)
}

export interface AgentParameters {
  maxIterations: number;
  convergenceThreshold: number;
  placementAlgorithm: 'force-directed' | 'simulated-annealing' | 'genetic' | 'hybrid';
  routingAlgorithm: 'maze' | 'channel' | 'area' | 'manhattan';
  optimizationPasses: number;
  coolingSchedule?: 'linear' | 'exponential' | 'adaptive';
  populationSize?: number;
  mutationRate?: number;
}

export interface PlacementContext {
  schematic: Schematic;
  constraints: BoardConstraints;
  previousLayout?: PCBLayout;
  feedbackHistory: AgentFeedback[];
  roundNumber: number;
  phaseNumber: number;
}

export interface AgentFeedback {
  round: number;
  validation: ValidationResult;
  ranking: number;
  competitorScores: Map<string, number>;
  suggestions: string[];
}

export interface PlacementResult {
  layout: PCBLayout;
  metrics: LayoutMetrics;
  iterations: number;
  converged: boolean;
  improvementHistory: number[];
}

export interface LayoutMetrics {
  boardArea: number;
  componentDensity: number;
  routingCompletion: number;
  averageTraceLength: number;
  viaCount: number;
  layerUtilization: Map<string, number>;
  thermalScore: number;
  signalIntegrityScore: number;
  drcViolations: number;
  estimatedYield: number;
}

export abstract class BaseLayoutAgent extends EventEmitter {
  public readonly id: string;
  public readonly name: string;
  public readonly strategy: AgentStrategy;
  protected readonly weights: StrategyWeights;
  protected readonly parameters: AgentParameters;

  protected currentLayout: PCBLayout | null = null;
  protected bestLayout: PCBLayout | null = null;
  protected bestScore: number = 0;
  protected iterationCount: number = 0;
  protected feedbackHistory: AgentFeedback[] = [];

  constructor(config: AgentConfig) {
    super();
    this.id = uuidv4();
    this.name = config.name;
    this.strategy = config.strategy;
    this.weights = config.weights;
    this.parameters = config.parameters;
  }

  /**
   * Generate a PCB layout from schematic and constraints
   */
  abstract generateLayout(context: PlacementContext): Promise<PlacementResult>;

  /**
   * Refine an existing layout based on validation feedback
   */
  abstract refineLayout(
    layout: PCBLayout,
    feedback: AgentFeedback,
    context: PlacementContext
  ): Promise<PlacementResult>;

  /**
   * Calculate placement score for a component position
   */
  protected abstract calculatePlacementScore(
    component: ComponentPlacement,
    neighbors: ComponentPlacement[],
    constraints: BoardConstraints
  ): number;

  /**
   * Optimize routing for a set of traces
   */
  protected abstract optimizeRouting(
    traces: Trace[],
    layout: PCBLayout,
    constraints: BoardConstraints
  ): Trace[];

  /**
   * Process feedback from validators and adapt strategy
   */
  public processFeedback(feedback: AgentFeedback): void {
    this.feedbackHistory.push(feedback);
    this.adaptStrategy(feedback);
    this.emit('feedback-processed', { agentId: this.id, feedback });
  }

  /**
   * Adapt strategy based on feedback (override in subclasses for specific behavior)
   */
  protected adaptStrategy(feedback: AgentFeedback): void {
    // Default implementation - subclasses can override for learning behavior
    if (feedback.validation.score > this.bestScore) {
      this.bestScore = feedback.validation.score;
      this.bestLayout = this.currentLayout;
    }
  }

  /**
   * Get the agent's current best layout
   */
  public getBestLayout(): PCBLayout | null {
    return this.bestLayout;
  }

  /**
   * Get performance metrics for this agent
   */
  public getMetrics(): {
    bestScore: number;
    totalIterations: number;
    roundsParticipated: number;
    averageScore: number;
    improvementRate: number;
  } {
    const scores = this.feedbackHistory.map(f => f.validation.score);
    const avgScore = scores.length > 0
      ? scores.reduce((a, b) => a + b, 0) / scores.length
      : 0;

    const improvementRate = scores.length > 1
      ? (scores[scores.length - 1] - scores[0]) / scores.length
      : 0;

    return {
      bestScore: this.bestScore,
      totalIterations: this.iterationCount,
      roundsParticipated: this.feedbackHistory.length,
      averageScore: avgScore,
      improvementRate
    };
  }

  /**
   * Reset the agent's state for a new tournament
   */
  public reset(): void {
    this.currentLayout = null;
    this.bestLayout = null;
    this.bestScore = 0;
    this.iterationCount = 0;
    this.feedbackHistory = [];
    this.emit('reset', { agentId: this.id });
  }

  /**
   * Serialize agent state for persistence
   */
  public serialize(): object {
    return {
      id: this.id,
      name: this.name,
      strategy: this.strategy,
      weights: this.weights,
      parameters: this.parameters,
      bestScore: this.bestScore,
      iterationCount: this.iterationCount,
      feedbackHistory: this.feedbackHistory,
      bestLayout: this.bestLayout
    };
  }

  /**
   * Helper: Calculate distance between two components
   */
  protected calculateDistance(
    a: { x: number; y: number },
    b: { x: number; y: number }
  ): number {
    return Math.sqrt(Math.pow(b.x - a.x, 2) + Math.pow(b.y - a.y, 2));
  }

  /**
   * Helper: Check if two rectangles overlap
   */
  protected checkOverlap(
    a: { x: number; y: number; width: number; height: number },
    b: { x: number; y: number; width: number; height: number },
    margin: number = 0
  ): boolean {
    return !(
      a.x + a.width + margin < b.x ||
      b.x + b.width + margin < a.x ||
      a.y + a.height + margin < b.y ||
      b.y + b.height + margin < a.y
    );
  }

  /**
   * Helper: Calculate Manhattan distance for routing estimation
   */
  protected manhattanDistance(
    a: { x: number; y: number },
    b: { x: number; y: number }
  ): number {
    return Math.abs(b.x - a.x) + Math.abs(b.y - a.y);
  }

  /**
   * Helper: Generate via positions for layer transitions
   */
  protected generateVias(
    trace: Trace,
    startLayer: string,
    endLayer: string
  ): Via[] {
    const vias: Via[] = [];

    // Simple via generation - place at trace midpoint
    if (trace.points.length >= 2) {
      const midIndex = Math.floor(trace.points.length / 2);
      const midPoint = trace.points[midIndex];

      vias.push({
        id: uuidv4(),
        position: { x: midPoint.x, y: midPoint.y },
        diameter: 0.6, // Standard via diameter in mm
        drillSize: 0.3,
        layers: [startLayer, endLayer],
        type: 'through'
      });
    }

    return vias;
  }

  /**
   * Helper: Calculate board utilization percentage
   */
  protected calculateUtilization(
    components: ComponentPlacement[],
    boardWidth: number,
    boardHeight: number
  ): number {
    const totalComponentArea = components.reduce((sum, c) => {
      return sum + (c.footprint?.width || 0) * (c.footprint?.height || 0);
    }, 0);

    const boardArea = boardWidth * boardHeight;
    return boardArea > 0 ? (totalComponentArea / boardArea) * 100 : 0;
  }
}

export default BaseLayoutAgent;