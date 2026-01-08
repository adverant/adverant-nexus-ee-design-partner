/**
 * Aggressive Compact Layout Agent
 *
 * Prioritizes minimal board area and high component density.
 * Uses tight spacing, aggressive optimization, and multi-layer routing.
 * Best for size-constrained designs and cost optimization.
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

const AGGRESSIVE_CONFIG: AgentConfig = {
  name: 'Aggressive Compact Agent',
  strategy: 'aggressive-compact',
  weights: {
    area: 0.35,     // Highest priority on area minimization
    thermal: 0.1,   // Lower thermal priority (trade-off)
    signal: 0.15,   // Moderate signal integrity
    emi: 0.1,       // Lower EMI consideration
    dfm: 0.15,      // Moderate DFM
    routing: 0.15   // High routing density
  },
  parameters: {
    maxIterations: 1000,
    convergenceThreshold: 0.005,
    placementAlgorithm: 'genetic',
    routingAlgorithm: 'area',
    optimizationPasses: 5,
    populationSize: 50,
    mutationRate: 0.15
  }
};

interface Individual {
  placements: ComponentPlacement[];
  fitness: number;
}

export class AggressiveCompactAgent extends BaseLayoutAgent {
  private readonly spacingMultiplier: number = 0.9; // 10% tighter spacing
  private readonly densityTarget: number = 0.7; // 70% board utilization target
  private population: Individual[] = [];

  constructor(config?: Partial<AgentConfig>) {
    super({ ...AGGRESSIVE_CONFIG, ...config });
  }

  async generateLayout(context: PlacementContext): Promise<PlacementResult> {
    const { schematic, constraints, roundNumber } = context;
    this.emit('generation-start', { agentId: this.id, round: roundNumber });

    const startTime = Date.now();
    const improvementHistory: number[] = [];
    let iterations = 0;
    let converged = false;

    // Calculate optimal board size based on component area
    const componentArea = this.calculateTotalComponentArea(schematic.components);
    const targetArea = componentArea / this.densityTarget;
    const aspectRatio = constraints.maxWidth / constraints.maxHeight;
    let boardWidth = Math.sqrt(targetArea * aspectRatio);
    let boardHeight = Math.sqrt(targetArea / aspectRatio);

    // Ensure within constraints
    boardWidth = Math.min(boardWidth, constraints.maxWidth);
    boardHeight = Math.min(boardHeight, constraints.maxHeight);

    // Initialize population
    this.population = this.initializePopulation(
      schematic.components,
      boardWidth,
      boardHeight,
      constraints,
      this.parameters.populationSize || 50
    );

    // Genetic algorithm optimization
    let generation = 0;
    let bestIndividual = this.population[0];

    while (iterations < this.parameters.maxIterations) {
      // Evaluate fitness
      for (const individual of this.population) {
        individual.fitness = this.evaluateFitness(individual.placements, constraints, boardWidth, boardHeight);
      }

      // Sort by fitness
      this.population.sort((a, b) => b.fitness - a.fitness);

      // Update best
      if (this.population[0].fitness > bestIndividual.fitness) {
        bestIndividual = {
          placements: JSON.parse(JSON.stringify(this.population[0].placements)),
          fitness: this.population[0].fitness
        };
      }

      improvementHistory.push(bestIndividual.fitness);

      // Check convergence
      if (this.checkConvergence(improvementHistory)) {
        converged = true;
        break;
      }

      // Selection, crossover, mutation
      this.population = this.evolvePopulation(constraints, boardWidth, boardHeight);

      generation++;
      iterations += this.population.length;

      // Emit progress
      if (generation % 10 === 0) {
        this.emit('progress', {
          agentId: this.id,
          generation,
          bestFitness: bestIndividual.fitness,
          avgFitness: this.population.reduce((s, i) => s + i.fitness, 0) / this.population.length
        });
      }
    }

    // Try to shrink board if possible
    const finalSize = this.optimizeBoardSize(bestIndividual.placements, constraints);
    boardWidth = finalSize.width;
    boardHeight = finalSize.height;

    // Generate routing
    const traces = this.generateCompactRouting(bestIndividual.placements, constraints);
    const layers = this.generateLayers(constraints);

    const layout: PCBLayout = {
      id: uuidv4(),
      projectId: context.schematic.projectId || '',
      name: `${this.name} Layout - Round ${roundNumber}`,
      version: '1.0',
      boardOutline: {
        width: boardWidth,
        height: boardHeight,
        shape: 'rectangular',
        cornerRadius: 1 // Minimal corner radius for space
      },
      layers,
      components: bestIndividual.placements,
      traces,
      vias: [],
      zones: [],
      designRules: this.getAggressiveDesignRules(constraints),
      createdAt: new Date(),
      updatedAt: new Date()
    };

    const metrics = this.calculateMetrics(layout);

    this.currentLayout = layout;
    if (bestIndividual.fitness > this.bestScore) {
      this.bestScore = bestIndividual.fitness;
      this.bestLayout = layout;
    }

    this.iterationCount += iterations;
    const elapsed = Date.now() - startTime;
    this.emit('generation-complete', {
      agentId: this.id,
      round: roundNumber,
      elapsed,
      score: bestIndividual.fitness,
      boardArea: boardWidth * boardHeight
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

    // Analyze constraints from feedback
    const violationTypes = this.analyzeViolations(feedback);
    let placements = [...layout.components];

    // If too compact, slightly expand
    if (violationTypes.includes('clearance') || violationTypes.includes('drc')) {
      placements = this.relaxPlacement(placements, 1.05);
    }

    // Reinitialize population with current best as seed
    this.population = [];
    for (let i = 0; i < (this.parameters.populationSize || 50); i++) {
      if (i === 0) {
        this.population.push({ placements: [...placements], fitness: 0 });
      } else {
        // Mutate from best
        this.population.push({
          placements: this.mutatePlacements(placements, context.constraints, 0.2),
          fitness: 0
        });
      }
    }

    // Run shorter GA for refinement
    const refinementIterations = Math.floor(this.parameters.maxIterations / 3);
    let bestFitness = 0;
    let bestPlacements = placements;

    for (let gen = 0; gen < refinementIterations / (this.parameters.populationSize || 50); gen++) {
      for (const individual of this.population) {
        individual.fitness = this.evaluateFitness(
          individual.placements,
          context.constraints,
          layout.boardOutline.width,
          layout.boardOutline.height
        );
        if (individual.fitness > bestFitness) {
          bestFitness = individual.fitness;
          bestPlacements = [...individual.placements];
        }
      }

      improvementHistory.push(bestFitness);
      this.population = this.evolvePopulation(
        context.constraints,
        layout.boardOutline.width,
        layout.boardOutline.height
      );
      iterations += this.population.length;
    }

    const traces = this.generateCompactRouting(bestPlacements, context.constraints);

    const refinedLayout: PCBLayout = {
      ...layout,
      id: uuidv4(),
      name: `${this.name} Refined - Round ${context.roundNumber}`,
      components: bestPlacements,
      traces,
      updatedAt: new Date()
    };

    const metrics = this.calculateMetrics(refinedLayout);

    this.currentLayout = refinedLayout;
    if (bestFitness > this.bestScore) {
      this.bestScore = bestFitness;
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

    // Reward for being close to neighbors (compact)
    const minSpacing = (constraints.minComponentSpacing || 1) * this.spacingMultiplier;
    for (const neighbor of neighbors) {
      const dist = this.calculateDistance(component.position, neighbor.position);
      if (dist < minSpacing) {
        score -= 30; // Penalize violations
      } else if (dist < minSpacing * 2) {
        score += 5; // Reward close but valid
      }
    }

    // Penalize wasted edge space
    const edgeDist = Math.min(
      component.position.x,
      component.position.y,
      constraints.maxWidth - component.position.x,
      constraints.maxHeight - component.position.y
    );
    if (edgeDist > 5) score -= 2; // Slight penalty for unused edge area

    return Math.max(0, score);
  }

  protected optimizeRouting(
    traces: Trace[],
    layout: PCBLayout,
    constraints: BoardConstraints
  ): Trace[] {
    // Minimize trace width while maintaining signal integrity
    return traces.map(trace => ({
      ...trace,
      width: constraints.minTraceWidth || 0.15,
      // Use more vias for layer changes to allow denser routing
    }));
  }

  private calculateTotalComponentArea(components: any[]): number {
    return components.reduce((sum, c) => {
      const w = c.footprint?.width || 5;
      const h = c.footprint?.height || 5;
      return sum + w * h;
    }, 0);
  }

  private initializePopulation(
    components: any[],
    boardWidth: number,
    boardHeight: number,
    constraints: BoardConstraints,
    size: number
  ): Individual[] {
    const population: Individual[] = [];

    for (let i = 0; i < size; i++) {
      const placements = this.generateRandomPlacement(
        components,
        boardWidth,
        boardHeight,
        constraints,
        i === 0 // First one is grid-based
      );
      population.push({ placements, fitness: 0 });
    }

    return population;
  }

  private generateRandomPlacement(
    components: any[],
    boardWidth: number,
    boardHeight: number,
    constraints: BoardConstraints,
    useGrid: boolean = false
  ): ComponentPlacement[] {
    const placements: ComponentPlacement[] = [];
    const margin = 1; // Tight margin

    if (useGrid) {
      // Dense grid placement
      let x = margin;
      let y = margin;
      let rowHeight = 0;

      for (const component of components) {
        const w = component.footprint?.width || 3;
        const h = component.footprint?.height || 3;
        const spacing = (constraints.minComponentSpacing || 0.5) * this.spacingMultiplier;

        if (x + w + margin > boardWidth) {
          x = margin;
          y += rowHeight + spacing;
          rowHeight = 0;
        }

        placements.push({
          id: uuidv4(),
          componentId: component.id,
          reference: component.reference || `C${placements.length + 1}`,
          position: { x, y },
          rotation: 0,
          layer: 'top',
          footprint: { name: component.footprint?.name || 'unknown', width: w, height: h },
          pads: []
        });

        x += w + spacing;
        rowHeight = Math.max(rowHeight, h);
      }
    } else {
      // Random placement
      for (const component of components) {
        const w = component.footprint?.width || 3;
        const h = component.footprint?.height || 3;

        placements.push({
          id: uuidv4(),
          componentId: component.id,
          reference: component.reference || `C${placements.length + 1}`,
          position: {
            x: margin + Math.random() * (boardWidth - w - 2 * margin),
            y: margin + Math.random() * (boardHeight - h - 2 * margin)
          },
          rotation: Math.floor(Math.random() * 4) * 90,
          layer: Math.random() > 0.8 ? 'bottom' : 'top', // 20% chance bottom
          footprint: { name: component.footprint?.name || 'unknown', width: w, height: h },
          pads: []
        });
      }
    }

    return placements;
  }

  private evaluateFitness(
    placements: ComponentPlacement[],
    constraints: BoardConstraints,
    boardWidth: number,
    boardHeight: number
  ): number {
    let fitness = 100;

    // Calculate bounding box
    let minX = Infinity, minY = Infinity, maxX = 0, maxY = 0;
    for (const p of placements) {
      minX = Math.min(minX, p.position.x);
      minY = Math.min(minY, p.position.y);
      maxX = Math.max(maxX, p.position.x + (p.footprint?.width || 0));
      maxY = Math.max(maxY, p.position.y + (p.footprint?.height || 0));
    }

    const usedArea = (maxX - minX) * (maxY - minY);
    const boardArea = boardWidth * boardHeight;
    const efficiency = 1 - (usedArea / boardArea);

    // Reward compact placement
    fitness += efficiency * 30;

    // Penalize overlaps heavily
    const minSpacing = (constraints.minComponentSpacing || 0.5) * this.spacingMultiplier;
    for (let i = 0; i < placements.length; i++) {
      for (let j = i + 1; j < placements.length; j++) {
        if (this.checkOverlap(
          { ...placements[i].position, width: placements[i].footprint?.width || 3, height: placements[i].footprint?.height || 3 },
          { ...placements[j].position, width: placements[j].footprint?.width || 3, height: placements[j].footprint?.height || 3 },
          minSpacing
        )) {
          fitness -= 25;
        }
      }
    }

    // Penalize out of bounds
    for (const p of placements) {
      if (p.position.x < 0 || p.position.y < 0 ||
          p.position.x + (p.footprint?.width || 0) > boardWidth ||
          p.position.y + (p.footprint?.height || 0) > boardHeight) {
        fitness -= 50;
      }
    }

    // Reward layer usage (use both sides)
    const topCount = placements.filter(p => p.layer === 'top').length;
    const bottomCount = placements.filter(p => p.layer === 'bottom').length;
    if (bottomCount > 0 && topCount > 0) {
      const balance = Math.min(topCount, bottomCount) / Math.max(topCount, bottomCount);
      fitness += balance * 10;
    }

    return Math.max(0, fitness);
  }

  private evolvePopulation(
    constraints: BoardConstraints,
    boardWidth: number,
    boardHeight: number
  ): Individual[] {
    const newPopulation: Individual[] = [];
    const eliteCount = 2;
    const popSize = this.parameters.populationSize || 50;

    // Keep elite
    for (let i = 0; i < eliteCount; i++) {
      newPopulation.push({
        placements: JSON.parse(JSON.stringify(this.population[i].placements)),
        fitness: this.population[i].fitness
      });
    }

    // Generate rest through crossover and mutation
    while (newPopulation.length < popSize) {
      // Tournament selection
      const parent1 = this.tournamentSelect(3);
      const parent2 = this.tournamentSelect(3);

      // Crossover
      const child = this.crossover(parent1, parent2);

      // Mutation
      const mutated = this.mutatePlacements(
        child,
        constraints,
        this.parameters.mutationRate || 0.15
      );

      newPopulation.push({ placements: mutated, fitness: 0 });
    }

    return newPopulation;
  }

  private tournamentSelect(tournamentSize: number): ComponentPlacement[] {
    let best = this.population[Math.floor(Math.random() * this.population.length)];
    for (let i = 1; i < tournamentSize; i++) {
      const candidate = this.population[Math.floor(Math.random() * this.population.length)];
      if (candidate.fitness > best.fitness) {
        best = candidate;
      }
    }
    return best.placements;
  }

  private crossover(
    parent1: ComponentPlacement[],
    parent2: ComponentPlacement[]
  ): ComponentPlacement[] {
    // Two-point crossover by position
    const child: ComponentPlacement[] = [];
    const crossPoint1 = Math.floor(Math.random() * parent1.length);
    const crossPoint2 = crossPoint1 + Math.floor(Math.random() * (parent1.length - crossPoint1));

    for (let i = 0; i < parent1.length; i++) {
      if (i >= crossPoint1 && i <= crossPoint2) {
        child.push({ ...parent2[i] });
      } else {
        child.push({ ...parent1[i] });
      }
    }

    return child;
  }

  private mutatePlacements(
    placements: ComponentPlacement[],
    constraints: BoardConstraints,
    rate: number
  ): ComponentPlacement[] {
    return placements.map(p => {
      if (Math.random() < rate) {
        const moveRange = 3;
        return {
          ...p,
          position: {
            x: Math.max(0.5, p.position.x + (Math.random() - 0.5) * 2 * moveRange),
            y: Math.max(0.5, p.position.y + (Math.random() - 0.5) * 2 * moveRange)
          },
          rotation: Math.random() < 0.1 ? (p.rotation + 90) % 360 : p.rotation,
          layer: Math.random() < 0.05 ? (p.layer === 'top' ? 'bottom' : 'top') : p.layer
        };
      }
      return { ...p };
    });
  }

  private checkConvergence(history: number[]): boolean {
    if (history.length < 100) return false;
    const recent = history.slice(-100);
    const early = recent.slice(0, 50);
    const late = recent.slice(-50);
    const earlyAvg = early.reduce((a, b) => a + b, 0) / early.length;
    const lateAvg = late.reduce((a, b) => a + b, 0) / late.length;
    return Math.abs(lateAvg - earlyAvg) < this.parameters.convergenceThreshold;
  }

  private optimizeBoardSize(
    placements: ComponentPlacement[],
    constraints: BoardConstraints
  ): { width: number; height: number } {
    let maxX = 0, maxY = 0;
    for (const p of placements) {
      maxX = Math.max(maxX, p.position.x + (p.footprint?.width || 0));
      maxY = Math.max(maxY, p.position.y + (p.footprint?.height || 0));
    }

    return {
      width: Math.min(Math.max(maxX + 2, 20), constraints.maxWidth), // 2mm margin
      height: Math.min(Math.max(maxY + 2, 20), constraints.maxHeight)
    };
  }

  private generateCompactRouting(
    placements: ComponentPlacement[],
    constraints: BoardConstraints
  ): Trace[] {
    // Placeholder - actual routing is complex
    return [];
  }

  private generateLayers(constraints: BoardConstraints): Layer[] {
    const layerCount = Math.max(constraints.layerCount || 4, 4); // Minimum 4 layers
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

  private getAggressiveDesignRules(constraints: BoardConstraints): any {
    return {
      minTraceWidth: constraints.minTraceWidth || 0.1,
      minClearance: constraints.minClearance || 0.1,
      minViaDiameter: 0.4,
      minViaDrill: 0.2,
      minComponentSpacing: (constraints.minComponentSpacing || 0.5) * this.spacingMultiplier
    };
  }

  private analyzeViolations(feedback: AgentFeedback): string[] {
    const violations: string[] = [];
    if (feedback.validation.details?.drc?.violations?.length > 0) {
      violations.push('drc');
      if (feedback.validation.details.drc.violations.some((v: any) => v.type === 'clearance')) {
        violations.push('clearance');
      }
    }
    return violations;
  }

  private relaxPlacement(
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

  private calculateMetrics(layout: PCBLayout): LayoutMetrics {
    const boardArea = layout.boardOutline.width * layout.boardOutline.height;
    const componentArea = layout.components.reduce((sum, c) =>
      sum + (c.footprint?.width || 0) * (c.footprint?.height || 0), 0);

    return {
      boardArea,
      componentDensity: componentArea / boardArea,
      routingCompletion: 0.9, // Aggressive routing may have lower completion
      averageTraceLength: 0,
      viaCount: layout.vias?.length || 0,
      layerUtilization: new Map(),
      thermalScore: 70, // May have thermal challenges
      signalIntegrityScore: 80,
      drcViolations: 0,
      estimatedYield: 92 // Slightly lower yield due to tight tolerances
    };
  }
}

export default AggressiveCompactAgent;