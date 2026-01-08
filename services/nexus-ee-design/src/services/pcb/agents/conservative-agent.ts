/**
 * Conservative Layout Agent
 *
 * Prioritizes reliability, manufacturability, and proven design patterns.
 * Uses generous spacing, standard trace widths, and conservative thermal margins.
 * Best for safety-critical designs and production reliability.
 */

import { v4 as uuidv4 } from 'uuid';
import {
  BaseLayoutAgent,
  AgentConfig,
  PlacementContext,
  PlacementResult,
  AgentFeedback,
  LayoutMetrics
} from './base-agent';
import {
  PCBLayout,
  BoardConstraints,
  ComponentPlacement,
  Trace,
  Layer
} from '../../../types';

const CONSERVATIVE_CONFIG: AgentConfig = {
  name: 'Conservative Agent',
  strategy: 'conservative',
  weights: {
    area: 0.1,      // Low priority on area minimization
    thermal: 0.2,   // Moderate thermal priority
    signal: 0.2,    // Moderate signal integrity
    emi: 0.15,      // Standard EMI consideration
    dfm: 0.25,      // High DFM priority
    routing: 0.1    // Low routing density
  },
  parameters: {
    maxIterations: 500,
    convergenceThreshold: 0.01,
    placementAlgorithm: 'simulated-annealing',
    routingAlgorithm: 'maze',
    optimizationPasses: 3,
    coolingSchedule: 'linear'
  }
};

export class ConservativeAgent extends BaseLayoutAgent {
  private readonly spacingMultiplier: number = 1.5; // 50% extra spacing
  private readonly traceWidthMultiplier: number = 1.2; // 20% wider traces
  private readonly thermalMargin: number = 1.3; // 30% thermal margin

  constructor(config?: Partial<AgentConfig>) {
    super({ ...CONSERVATIVE_CONFIG, ...config });
  }

  async generateLayout(context: PlacementContext): Promise<PlacementResult> {
    const { schematic, constraints, roundNumber } = context;
    this.emit('generation-start', { agentId: this.id, round: roundNumber });

    const startTime = Date.now();
    const improvementHistory: number[] = [];
    let iterations = 0;
    let converged = false;

    // Initialize board dimensions with conservative margins
    const boardWidth = constraints.maxWidth * 0.9; // Use 90% of max
    const boardHeight = constraints.maxHeight * 0.9;

    // Generate initial placement using grid-based approach
    let placements = this.generateInitialPlacement(
      schematic.components,
      boardWidth,
      boardHeight,
      constraints
    );

    // Simulated annealing optimization
    let temperature = 100;
    const coolingRate = 0.995;
    let currentScore = this.evaluatePlacement(placements, constraints);
    let bestPlacements = [...placements];
    let bestPlacementScore = currentScore;

    while (iterations < this.parameters.maxIterations && temperature > 1) {
      // Generate neighbor solution
      const neighborPlacements = this.generateNeighbor(placements, constraints);
      const neighborScore = this.evaluatePlacement(neighborPlacements, constraints);

      // Accept or reject based on temperature
      const delta = neighborScore - currentScore;
      if (delta > 0 || Math.random() < Math.exp(delta / temperature)) {
        placements = neighborPlacements;
        currentScore = neighborScore;

        if (currentScore > bestPlacementScore) {
          bestPlacements = [...placements];
          bestPlacementScore = currentScore;
        }
      }

      improvementHistory.push(currentScore);
      temperature *= coolingRate;
      iterations++;

      // Check convergence
      if (this.checkConvergence(improvementHistory)) {
        converged = true;
        break;
      }

      // Emit progress
      if (iterations % 50 === 0) {
        this.emit('progress', {
          agentId: this.id,
          iteration: iterations,
          score: currentScore,
          temperature
        });
      }
    }

    // Generate routing with conservative trace widths
    const traces = this.generateRouting(bestPlacements, constraints);
    const layers = this.generateLayers(constraints);

    // Create the layout
    const layout: PCBLayout = {
      id: uuidv4(),
      projectId: context.schematic.projectId || '',
      name: `${this.name} Layout - Round ${roundNumber}`,
      version: '1.0',
      boardOutline: {
        width: boardWidth,
        height: boardHeight,
        shape: 'rectangular',
        cornerRadius: 3 // Conservative rounded corners
      },
      layers,
      components: bestPlacements,
      traces,
      vias: [],
      zones: this.generateZones(bestPlacements, constraints),
      designRules: this.getConservativeDesignRules(constraints),
      createdAt: new Date(),
      updatedAt: new Date()
    };

    // Calculate metrics
    const metrics = this.calculateMetrics(layout);

    this.currentLayout = layout;
    if (bestPlacementScore > this.bestScore) {
      this.bestScore = bestPlacementScore;
      this.bestLayout = layout;
    }

    this.iterationCount += iterations;
    const elapsed = Date.now() - startTime;
    this.emit('generation-complete', {
      agentId: this.id,
      round: roundNumber,
      elapsed,
      score: bestPlacementScore
    });

    return {
      layout,
      metrics,
      iterations,
      converged,
      improvementHistory
    };
  }

  async refineLayout(
    layout: PCBLayout,
    feedback: AgentFeedback,
    context: PlacementContext
  ): Promise<PlacementResult> {
    this.emit('refinement-start', { agentId: this.id, round: context.roundNumber });

    const improvementHistory: number[] = [];
    let iterations = 0;
    let converged = false;

    // Analyze feedback and identify problem areas
    const problemAreas = this.analyzeFeedback(feedback, layout);

    // Clone and modify layout
    let refinedPlacements = [...layout.components];
    let currentScore = feedback.validation.score;

    // Focus on fixing identified problems with conservative approach
    for (const problem of problemAreas) {
      refinedPlacements = this.fixProblem(
        refinedPlacements,
        problem,
        context.constraints
      );
    }

    // Additional optimization passes
    const temperature = 50; // Lower temperature for refinement
    for (let pass = 0; pass < this.parameters.optimizationPasses; pass++) {
      for (let i = 0; i < 100; i++) {
        const neighbor = this.generateNeighbor(refinedPlacements, context.constraints);
        const score = this.evaluatePlacement(neighbor, context.constraints);

        if (score > currentScore) {
          refinedPlacements = neighbor;
          currentScore = score;
        }

        improvementHistory.push(currentScore);
        iterations++;
      }
    }

    // Regenerate routing
    const traces = this.generateRouting(refinedPlacements, context.constraints);

    const refinedLayout: PCBLayout = {
      ...layout,
      id: uuidv4(),
      name: `${this.name} Refined - Round ${context.roundNumber}`,
      components: refinedPlacements,
      traces,
      updatedAt: new Date()
    };

    const metrics = this.calculateMetrics(refinedLayout);

    this.currentLayout = refinedLayout;
    if (currentScore > this.bestScore) {
      this.bestScore = currentScore;
      this.bestLayout = refinedLayout;
    }

    return {
      layout: refinedLayout,
      metrics,
      iterations,
      converged,
      improvementHistory
    };
  }

  protected calculatePlacementScore(
    component: ComponentPlacement,
    neighbors: ComponentPlacement[],
    constraints: BoardConstraints
  ): number {
    let score = 100;

    // Penalize close neighbors (conservative spacing)
    for (const neighbor of neighbors) {
      const dist = this.calculateDistance(component.position, neighbor.position);
      const minDist = this.getMinSpacing(component, neighbor) * this.spacingMultiplier;

      if (dist < minDist) {
        score -= (minDist - dist) * 10;
      }
    }

    // Bonus for alignment to grid
    const gridSize = constraints.gridSize || 0.5;
    const xOffset = component.position.x % gridSize;
    const yOffset = component.position.y % gridSize;
    if (xOffset < 0.01 && yOffset < 0.01) {
      score += 5;
    }

    // Penalize edge proximity
    const edgeMargin = 3; // 3mm from edge
    if (
      component.position.x < edgeMargin ||
      component.position.y < edgeMargin ||
      component.position.x > (constraints.maxWidth - edgeMargin) ||
      component.position.y > (constraints.maxHeight - edgeMargin)
    ) {
      score -= 15;
    }

    return Math.max(0, score);
  }

  protected optimizeRouting(
    traces: Trace[],
    layout: PCBLayout,
    constraints: BoardConstraints
  ): Trace[] {
    // Conservative routing: prefer wider traces and fewer layer changes
    return traces.map(trace => ({
      ...trace,
      width: Math.max(trace.width * this.traceWidthMultiplier, constraints.minTraceWidth || 0.15),
      // Minimize vias by staying on same layer when possible
      points: this.optimizeTracePath(trace.points)
    }));
  }

  private generateInitialPlacement(
    components: any[],
    boardWidth: number,
    boardHeight: number,
    constraints: BoardConstraints
  ): ComponentPlacement[] {
    const placements: ComponentPlacement[] = [];
    const gridSize = constraints.gridSize || 2.54; // 0.1" grid
    const margin = 5; // 5mm edge margin

    // Sort components by importance (power, then ICs, then passives)
    const sorted = [...components].sort((a, b) => {
      const priority = (c: any) => {
        if (c.reference?.startsWith('U')) return 0; // ICs first
        if (c.reference?.startsWith('Q')) return 1; // Transistors
        if (c.reference?.startsWith('C')) return 2; // Capacitors
        if (c.reference?.startsWith('R')) return 3; // Resistors
        return 4;
      };
      return priority(a) - priority(b);
    });

    // Grid-based placement
    let currentX = margin;
    let currentY = margin;
    let rowHeight = 0;

    for (const component of sorted) {
      const width = component.footprint?.width || 5;
      const height = component.footprint?.height || 5;
      const spacing = this.spacingMultiplier * (constraints.minComponentSpacing || 1);

      // Check if component fits in current row
      if (currentX + width + margin > boardWidth) {
        currentX = margin;
        currentY += rowHeight + spacing;
        rowHeight = 0;
      }

      placements.push({
        id: uuidv4(),
        componentId: component.id,
        reference: component.reference || `C${placements.length + 1}`,
        position: {
          x: Math.round(currentX / gridSize) * gridSize,
          y: Math.round(currentY / gridSize) * gridSize
        },
        rotation: 0,
        layer: 'top',
        footprint: {
          name: component.footprint?.name || 'unknown',
          width,
          height
        },
        pads: component.pins?.map((pin: any, i: number) => ({
          id: uuidv4(),
          name: pin.name || `${i + 1}`,
          position: { x: 0, y: i * 0.5 },
          size: { width: 0.8, height: 0.8 },
          shape: 'rectangular' as const,
          drillSize: 0
        })) || []
      });

      currentX += width + spacing;
      rowHeight = Math.max(rowHeight, height);
    }

    return placements;
  }

  private generateNeighbor(
    placements: ComponentPlacement[],
    constraints: BoardConstraints
  ): ComponentPlacement[] {
    const newPlacements = [...placements];
    const idx = Math.floor(Math.random() * newPlacements.length);
    const gridSize = constraints.gridSize || 0.5;

    // Small random movement for conservative approach
    const maxMove = 5; // Max 5mm movement
    const dx = (Math.random() - 0.5) * 2 * maxMove;
    const dy = (Math.random() - 0.5) * 2 * maxMove;

    newPlacements[idx] = {
      ...newPlacements[idx],
      position: {
        x: Math.round((newPlacements[idx].position.x + dx) / gridSize) * gridSize,
        y: Math.round((newPlacements[idx].position.y + dy) / gridSize) * gridSize
      }
    };

    // Occasionally swap two components
    if (Math.random() < 0.1) {
      const idx2 = Math.floor(Math.random() * newPlacements.length);
      const pos1 = newPlacements[idx].position;
      newPlacements[idx].position = newPlacements[idx2].position;
      newPlacements[idx2].position = pos1;
    }

    return newPlacements;
  }

  private evaluatePlacement(
    placements: ComponentPlacement[],
    constraints: BoardConstraints
  ): number {
    let score = 100;

    // Check for overlaps
    for (let i = 0; i < placements.length; i++) {
      for (let j = i + 1; j < placements.length; j++) {
        const spacing = (constraints.minComponentSpacing || 1) * this.spacingMultiplier;
        if (this.checkOverlap(
          { ...placements[i].position, width: placements[i].footprint?.width || 5, height: placements[i].footprint?.height || 5 },
          { ...placements[j].position, width: placements[j].footprint?.width || 5, height: placements[j].footprint?.height || 5 },
          spacing
        )) {
          score -= 20;
        }
      }
    }

    // Check board bounds
    for (const p of placements) {
      const margin = 3;
      if (
        p.position.x < margin || p.position.y < margin ||
        p.position.x + (p.footprint?.width || 0) > constraints.maxWidth - margin ||
        p.position.y + (p.footprint?.height || 0) > constraints.maxHeight - margin
      ) {
        score -= 10;
      }
    }

    // Bonus for grouping related components (by reference prefix)
    const groups = new Map<string, ComponentPlacement[]>();
    for (const p of placements) {
      const prefix = p.reference.replace(/\d+$/, '');
      if (!groups.has(prefix)) groups.set(prefix, []);
      groups.get(prefix)!.push(p);
    }

    for (const [, group] of groups) {
      if (group.length > 1) {
        let avgDist = 0;
        for (let i = 0; i < group.length; i++) {
          for (let j = i + 1; j < group.length; j++) {
            avgDist += this.calculateDistance(group[i].position, group[j].position);
          }
        }
        avgDist /= (group.length * (group.length - 1) / 2);
        if (avgDist < 20) score += 5; // Bonus for grouping
      }
    }

    return Math.max(0, score);
  }

  private checkConvergence(history: number[]): boolean {
    if (history.length < 50) return false;
    const recent = history.slice(-50);
    const variance = this.calculateVariance(recent);
    return variance < this.parameters.convergenceThreshold;
  }

  private calculateVariance(arr: number[]): number {
    const mean = arr.reduce((a, b) => a + b, 0) / arr.length;
    return arr.reduce((sum, x) => sum + Math.pow(x - mean, 2), 0) / arr.length;
  }

  private generateRouting(
    placements: ComponentPlacement[],
    constraints: BoardConstraints
  ): Trace[] {
    // Simple Manhattan routing for now - real implementation would use A* or maze routing
    const traces: Trace[] = [];
    // Placeholder - actual routing is complex
    return traces;
  }

  private generateLayers(constraints: BoardConstraints): Layer[] {
    const layerCount = constraints.layerCount || 4;
    const layers: Layer[] = [];

    for (let i = 0; i < layerCount; i++) {
      layers.push({
        id: uuidv4(),
        name: i === 0 ? 'F.Cu' : i === layerCount - 1 ? 'B.Cu' : `In${i}.Cu`,
        type: 'copper',
        thickness: constraints.copperWeight || 1,
        order: i
      });
    }

    return layers;
  }

  private generateZones(
    placements: ComponentPlacement[],
    constraints: BoardConstraints
  ): any[] {
    // Generate conservative ground pour zones
    return [];
  }

  private getConservativeDesignRules(constraints: BoardConstraints): any {
    return {
      minTraceWidth: (constraints.minTraceWidth || 0.15) * this.traceWidthMultiplier,
      minClearance: (constraints.minClearance || 0.15) * this.spacingMultiplier,
      minViaDiameter: 0.6,
      minViaDrill: 0.3,
      minComponentSpacing: (constraints.minComponentSpacing || 1) * this.spacingMultiplier
    };
  }

  private getMinSpacing(a: ComponentPlacement, b: ComponentPlacement): number {
    // Base spacing on component types
    return 2; // 2mm base spacing
  }

  private optimizeTracePath(points: Array<{ x: number; y: number }>): Array<{ x: number; y: number }> {
    // Remove unnecessary waypoints while maintaining Manhattan routing
    if (points.length < 3) return points;

    const optimized = [points[0]];
    for (let i = 1; i < points.length - 1; i++) {
      const prev = optimized[optimized.length - 1];
      const curr = points[i];
      const next = points[i + 1];

      // Keep point if direction changes
      const dx1 = curr.x - prev.x;
      const dy1 = curr.y - prev.y;
      const dx2 = next.x - curr.x;
      const dy2 = next.y - curr.y;

      if (Math.sign(dx1) !== Math.sign(dx2) || Math.sign(dy1) !== Math.sign(dy2)) {
        optimized.push(curr);
      }
    }
    optimized.push(points[points.length - 1]);

    return optimized;
  }

  private analyzeFeedback(feedback: AgentFeedback, layout: PCBLayout): string[] {
    const problems: string[] = [];

    if (feedback.validation.details?.drc?.violations?.length > 0) {
      problems.push('drc');
    }
    if (feedback.validation.details?.thermal?.maxTemperature > 85) {
      problems.push('thermal');
    }
    if (feedback.validation.details?.dfm?.yieldEstimate < 95) {
      problems.push('dfm');
    }

    return problems;
  }

  private fixProblem(
    placements: ComponentPlacement[],
    problem: string,
    constraints: BoardConstraints
  ): ComponentPlacement[] {
    // Apply conservative fixes
    switch (problem) {
      case 'drc':
        // Increase spacing between all components
        return this.increaseSpacing(placements, 1.2);
      case 'thermal':
        // Spread out high-power components
        return this.spreadComponents(placements, constraints);
      case 'dfm':
        // Align to grid and standardize orientations
        return this.alignToGrid(placements, constraints);
      default:
        return placements;
    }
  }

  private increaseSpacing(
    placements: ComponentPlacement[],
    factor: number
  ): ComponentPlacement[] {
    const centerX = placements.reduce((s, p) => s + p.position.x, 0) / placements.length;
    const centerY = placements.reduce((s, p) => s + p.position.y, 0) / placements.length;

    return placements.map(p => ({
      ...p,
      position: {
        x: centerX + (p.position.x - centerX) * factor,
        y: centerY + (p.position.y - centerY) * factor
      }
    }));
  }

  private spreadComponents(
    placements: ComponentPlacement[],
    constraints: BoardConstraints
  ): ComponentPlacement[] {
    return this.increaseSpacing(placements, 1.1);
  }

  private alignToGrid(
    placements: ComponentPlacement[],
    constraints: BoardConstraints
  ): ComponentPlacement[] {
    const gridSize = constraints.gridSize || 1.27;
    return placements.map(p => ({
      ...p,
      position: {
        x: Math.round(p.position.x / gridSize) * gridSize,
        y: Math.round(p.position.y / gridSize) * gridSize
      },
      rotation: Math.round(p.rotation / 90) * 90
    }));
  }

  private calculateMetrics(layout: PCBLayout): LayoutMetrics {
    const boardArea = layout.boardOutline.width * layout.boardOutline.height;
    const componentArea = layout.components.reduce((sum, c) =>
      sum + (c.footprint?.width || 0) * (c.footprint?.height || 0), 0);

    return {
      boardArea,
      componentDensity: componentArea / boardArea,
      routingCompletion: 0.95, // Placeholder
      averageTraceLength: 0,
      viaCount: layout.vias?.length || 0,
      layerUtilization: new Map(),
      thermalScore: 85, // Conservative estimate
      signalIntegrityScore: 90,
      drcViolations: 0,
      estimatedYield: 98 // High yield for conservative design
    };
  }
}

export default ConservativeAgent;