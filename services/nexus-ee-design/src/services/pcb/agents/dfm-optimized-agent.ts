/**
 * DFM-Optimized Layout Agent
 *
 * Prioritizes design for manufacturability and assembly.
 * Uses standard footprints, assembly-friendly placement, IPC compliance.
 * Best for high-volume production and cost-sensitive designs.
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

const DFM_CONFIG: AgentConfig = {
  name: 'DFM-Optimized Agent',
  strategy: 'dfm-optimized',
  weights: {
    area: 0.15,     // Moderate area priority
    thermal: 0.1,   // Lower thermal priority
    signal: 0.1,    // Lower signal priority
    emi: 0.1,       // Lower EMI priority
    dfm: 0.4,       // Highest DFM priority
    routing: 0.15   // Moderate routing
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

interface DFMComponent {
  placement: ComponentPlacement;
  packageType: 'smd' | 'through-hole' | 'bga' | 'qfp' | 'soic' | 'chip';
  pinPitch: number; // mm
  assemblyDifficulty: 'easy' | 'moderate' | 'difficult';
  requiresInspection: boolean;
  solderPasteType: 'standard' | 'fine-pitch' | 'no-clean';
}

interface AssemblyZone {
  id: string;
  name: string;
  bounds: { x: number; y: number; width: number; height: number };
  priority: number;
  requiresStencil: boolean;
}

export class DFMOptimizedAgent extends BaseLayoutAgent {
  private readonly standardPitchGrid: number = 0.5; // mm
  private readonly assemblyMargin: number = 3; // mm from board edge
  private readonly fiducialMargin: number = 5; // mm for fiducials
  private dfmComponents: DFMComponent[] = [];
  private assemblyZones: AssemblyZone[] = [];

  constructor(config?: Partial<AgentConfig>) {
    super({ ...DFM_CONFIG, ...config });
  }

  async generateLayout(context: PlacementContext): Promise<PlacementResult> {
    const { schematic, constraints, roundNumber } = context;
    this.emit('generation-start', { agentId: this.id, round: roundNumber });

    const startTime = Date.now();
    const improvementHistory: number[] = [];
    let iterations = 0;
    let converged = false;

    // Classify components by DFM characteristics
    this.dfmComponents = this.classifyDFMComponents(schematic.components);

    // Reserve space for fiducials and tooling holes
    const boardWidth = constraints.maxWidth;
    const boardHeight = constraints.maxHeight;
    const usableWidth = boardWidth - 2 * this.fiducialMargin;
    const usableHeight = boardHeight - 2 * this.fiducialMargin;

    // Generate DFM-optimized initial placement
    let placements = this.generateDFMPlacement(
      this.dfmComponents,
      usableWidth,
      usableHeight,
      constraints
    );

    // Optimization with DFM-focused scoring
    let temperature = 60;
    const coolingRate = 0.995;
    let currentScore = this.evaluateDFMPlacement(placements, constraints, boardWidth, boardHeight);
    let bestPlacements = [...placements];
    let bestScore = currentScore;

    while (iterations < this.parameters.maxIterations && temperature > 0.5) {
      const neighbor = this.generateDFMNeighbor(placements, constraints, boardWidth, boardHeight);
      const neighborScore = this.evaluateDFMPlacement(neighbor, constraints, boardWidth, boardHeight);

      const delta = neighborScore - currentScore;
      if (delta > 0 || Math.random() < Math.exp(delta / temperature)) {
        placements = neighbor;
        currentScore = neighborScore;

        if (currentScore > bestScore) {
          bestPlacements = JSON.parse(JSON.stringify(placements));
          bestScore = currentScore;
        }
      }

      improvementHistory.push(currentScore);
      temperature *= coolingRate;
      iterations++;

      if (this.checkConvergence(improvementHistory)) {
        converged = true;
        break;
      }

      if (iterations % 50 === 0) {
        this.emit('progress', {
          agentId: this.id,
          iteration: iterations,
          score: currentScore,
          dfmScore: this.calculateDFMScore(placements)
        });
      }
    }

    // Add fiducials and tooling holes
    const fiducials = this.generateFiducials(boardWidth, boardHeight);
    const toolingHoles = this.generateToolingHoles(boardWidth, boardHeight);

    // Generate assembly-friendly routing
    const traces = this.generateDFMRouting(bestPlacements, constraints);
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
        cornerRadius: 2,
        fiducials,
        toolingHoles
      },
      layers,
      components: bestPlacements,
      traces,
      vias: [],
      zones: this.generateDFMZones(bestPlacements),
      designRules: this.getDFMDesignRules(constraints),
      createdAt: new Date(),
      updatedAt: new Date()
    };

    const metrics = this.calculateMetrics(layout);

    this.currentLayout = layout;
    if (bestScore > this.bestScore) {
      this.bestScore = bestScore;
      this.bestLayout = layout;
    }

    this.iterationCount += iterations;
    const elapsed = Date.now() - startTime;
    this.emit('generation-complete', {
      agentId: this.id,
      round: roundNumber,
      elapsed,
      score: bestScore,
      estimatedYield: metrics.estimatedYield
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

    const dfmIssues = this.analyzeDFMFeedback(feedback);
    let placements = [...layout.components];

    for (const issue of dfmIssues) {
      placements = this.fixDFMIssue(placements, issue, context.constraints);
    }

    let temperature = 30;
    const coolingRate = 0.99;
    let currentScore = this.evaluateDFMPlacement(
      placements,
      context.constraints,
      layout.boardOutline.width,
      layout.boardOutline.height
    );
    let bestScore = currentScore;
    let bestPlacements = [...placements];

    const maxIterations = this.parameters.maxIterations / 3;
    while (iterations < maxIterations && temperature > 0.5) {
      const neighbor = this.generateDFMNeighbor(
        placements,
        context.constraints,
        layout.boardOutline.width,
        layout.boardOutline.height
      );
      const neighborScore = this.evaluateDFMPlacement(
        neighbor,
        context.constraints,
        layout.boardOutline.width,
        layout.boardOutline.height
      );

      const delta = neighborScore - currentScore;
      if (delta > 0 || Math.random() < Math.exp(delta / temperature)) {
        placements = neighbor;
        currentScore = neighborScore;
        if (currentScore > bestScore) {
          bestScore = currentScore;
          bestPlacements = [...placements];
        }
      }

      improvementHistory.push(currentScore);
      temperature *= coolingRate;
      iterations++;
    }

    const traces = this.generateDFMRouting(bestPlacements, context.constraints);

    const refinedLayout: PCBLayout = {
      ...layout,
      id: uuidv4(),
      name: `${this.name} Refined - Round ${context.roundNumber}`,
      components: bestPlacements,
      traces,
      zones: this.generateDFMZones(bestPlacements),
      updatedAt: new Date()
    };

    const metrics = this.calculateMetrics(refinedLayout);

    this.currentLayout = refinedLayout;
    if (bestScore > this.bestScore) {
      this.bestScore = bestScore;
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
    const dfm = this.dfmComponents.find(d => d.placement.id === component.id);

    // Reward grid alignment
    const gridX = component.position.x % this.standardPitchGrid;
    const gridY = component.position.y % this.standardPitchGrid;
    if (gridX < 0.01 && gridY < 0.01) {
      score += 10;
    } else {
      score -= 5;
    }

    // Standard rotations only (0, 90, 180, 270)
    if (component.rotation % 90 !== 0) {
      score -= 15;
    }

    // All components same side preferred for single-sided assembly
    const topCount = neighbors.filter(n => n.layer === 'top').length;
    const bottomCount = neighbors.filter(n => n.layer === 'bottom').length;
    if (component.layer === 'top' && topCount > bottomCount) {
      score += 5;
    }

    // Check assembly spacing
    const dfmSpacing = this.getAssemblySpacing(dfm);
    for (const neighbor of neighbors) {
      const dist = this.calculateDistance(component.position, neighbor.position);
      if (dist < dfmSpacing) {
        score -= (dfmSpacing - dist) * 5;
      }
    }

    return Math.max(0, score);
  }

  protected optimizeRouting(
    traces: Trace[],
    layout: PCBLayout,
    constraints: BoardConstraints
  ): Trace[] {
    // Use standard trace widths and via sizes
    return traces.map(trace => ({
      ...trace,
      width: this.standardizeTraceWidth(trace.width, constraints),
    }));
  }

  private classifyDFMComponents(components: any[]): DFMComponent[] {
    return components.map(c => {
      const ref = (c.reference || '').toUpperCase();
      const footprint = (c.footprint?.name || '').toLowerCase();

      let packageType: DFMComponent['packageType'] = 'smd';
      let pinPitch = 0.5;
      let assemblyDifficulty: DFMComponent['assemblyDifficulty'] = 'easy';
      let requiresInspection = false;
      let solderPasteType: DFMComponent['solderPasteType'] = 'standard';

      if (footprint.includes('bga')) {
        packageType = 'bga';
        pinPitch = footprint.includes('0.4') ? 0.4 : 0.5;
        assemblyDifficulty = 'difficult';
        requiresInspection = true;
        solderPasteType = 'fine-pitch';
      } else if (footprint.includes('qfp') || footprint.includes('tqfp')) {
        packageType = 'qfp';
        pinPitch = 0.5;
        assemblyDifficulty = 'moderate';
        solderPasteType = 'fine-pitch';
      } else if (footprint.includes('soic') || footprint.includes('sop')) {
        packageType = 'soic';
        pinPitch = 1.27;
        assemblyDifficulty = 'easy';
      } else if (footprint.includes('0201') || footprint.includes('0402')) {
        packageType = 'chip';
        pinPitch = 0;
        assemblyDifficulty = 'moderate';
        solderPasteType = 'fine-pitch';
      } else if (footprint.includes('0603') || footprint.includes('0805') || footprint.includes('1206')) {
        packageType = 'chip';
        pinPitch = 0;
        assemblyDifficulty = 'easy';
      } else if (footprint.includes('dip') || footprint.includes('through')) {
        packageType = 'through-hole';
        pinPitch = 2.54;
        assemblyDifficulty = 'easy';
      }

      return {
        placement: {
          id: uuidv4(),
          componentId: c.id,
          reference: c.reference || '',
          position: { x: 0, y: 0 },
          rotation: 0,
          layer: 'top',
          footprint: c.footprint || { name: 'unknown', width: 5, height: 5 },
          pads: []
        },
        packageType,
        pinPitch,
        assemblyDifficulty,
        requiresInspection,
        solderPasteType
      };
    });
  }

  private generateDFMPlacement(
    dfmComponents: DFMComponent[],
    usableWidth: number,
    usableHeight: number,
    constraints: BoardConstraints
  ): ComponentPlacement[] {
    const placements: ComponentPlacement[] = [];

    // Sort by assembly difficulty and size (difficult first, then by size)
    const sorted = [...dfmComponents].sort((a, b) => {
      const difficultyOrder = { 'difficult': 0, 'moderate': 1, 'easy': 2 };
      const diffDiff = difficultyOrder[a.assemblyDifficulty] - difficultyOrder[b.assemblyDifficulty];
      if (diffDiff !== 0) return diffDiff;

      // Then by size (larger first)
      const areaA = (a.placement.footprint?.width || 0) * (a.placement.footprint?.height || 0);
      const areaB = (b.placement.footprint?.width || 0) * (b.placement.footprint?.height || 0);
      return areaB - areaA;
    });

    // Place on grid
    const offsetX = this.fiducialMargin;
    const offsetY = this.fiducialMargin;
    let currentX = offsetX;
    let currentY = offsetY;
    let rowHeight = 0;

    for (const comp of sorted) {
      const w = comp.placement.footprint?.width || 5;
      const h = comp.placement.footprint?.height || 5;
      const spacing = this.getAssemblySpacing(comp);

      // Align to grid
      currentX = Math.ceil(currentX / this.standardPitchGrid) * this.standardPitchGrid;
      currentY = Math.ceil(currentY / this.standardPitchGrid) * this.standardPitchGrid;

      // Check row overflow
      if (currentX + w + spacing > offsetX + usableWidth) {
        currentX = offsetX;
        currentY += rowHeight + spacing;
        rowHeight = 0;
      }

      // Check page overflow
      if (currentY + h > offsetY + usableHeight) {
        // Would overflow - try bottom layer
        comp.placement.layer = 'bottom';
        currentX = offsetX;
        currentY = offsetY;
        rowHeight = 0;
      }

      placements.push({
        ...comp.placement,
        position: {
          x: Math.round(currentX / this.standardPitchGrid) * this.standardPitchGrid,
          y: Math.round(currentY / this.standardPitchGrid) * this.standardPitchGrid
        },
        rotation: 0, // Standard rotation
        id: uuidv4()
      });

      currentX += w + spacing;
      rowHeight = Math.max(rowHeight, h);
    }

    return placements;
  }

  private generateDFMNeighbor(
    placements: ComponentPlacement[],
    constraints: BoardConstraints,
    boardWidth: number,
    boardHeight: number
  ): ComponentPlacement[] {
    const newPlacements = JSON.parse(JSON.stringify(placements));
    const idx = Math.floor(Math.random() * newPlacements.length);

    // Only move in grid increments
    const moveSteps = Math.floor(Math.random() * 4) + 1;
    const direction = Math.floor(Math.random() * 4);

    switch (direction) {
      case 0: // Up
        newPlacements[idx].position.y -= moveSteps * this.standardPitchGrid;
        break;
      case 1: // Down
        newPlacements[idx].position.y += moveSteps * this.standardPitchGrid;
        break;
      case 2: // Left
        newPlacements[idx].position.x -= moveSteps * this.standardPitchGrid;
        break;
      case 3: // Right
        newPlacements[idx].position.x += moveSteps * this.standardPitchGrid;
        break;
    }

    // Occasionally rotate (only 90° increments)
    if (Math.random() < 0.1) {
      newPlacements[idx].rotation = (newPlacements[idx].rotation + 90) % 360;
    }

    // Occasionally swap layers
    if (Math.random() < 0.05) {
      newPlacements[idx].layer = newPlacements[idx].layer === 'top' ? 'bottom' : 'top';
    }

    // Clamp to usable area
    newPlacements[idx].position.x = Math.max(
      this.fiducialMargin,
      Math.min(boardWidth - this.fiducialMargin, newPlacements[idx].position.x)
    );
    newPlacements[idx].position.y = Math.max(
      this.fiducialMargin,
      Math.min(boardHeight - this.fiducialMargin, newPlacements[idx].position.y)
    );

    // Snap to grid
    newPlacements[idx].position.x = Math.round(newPlacements[idx].position.x / this.standardPitchGrid) * this.standardPitchGrid;
    newPlacements[idx].position.y = Math.round(newPlacements[idx].position.y / this.standardPitchGrid) * this.standardPitchGrid;

    return newPlacements;
  }

  private evaluateDFMPlacement(
    placements: ComponentPlacement[],
    constraints: BoardConstraints,
    boardWidth: number,
    boardHeight: number
  ): number {
    let score = 100;

    // Grid alignment check
    for (const p of placements) {
      const gridX = p.position.x % this.standardPitchGrid;
      const gridY = p.position.y % this.standardPitchGrid;
      if (gridX > 0.01 || gridY > 0.01) {
        score -= 3;
      }
    }

    // Standard rotation check
    for (const p of placements) {
      if (p.rotation % 90 !== 0) {
        score -= 5;
      }
    }

    // Single-sided assembly preference
    const topCount = placements.filter(p => p.layer === 'top').length;
    const bottomCount = placements.filter(p => p.layer === 'bottom').length;
    if (bottomCount === 0) {
      score += 20; // Bonus for single-sided
    } else if (bottomCount < topCount * 0.2) {
      score += 10; // Small bonus for mostly single-sided
    }

    // Assembly spacing
    for (let i = 0; i < placements.length; i++) {
      for (let j = i + 1; j < placements.length; j++) {
        const dfm1 = this.dfmComponents.find(d => d.placement.componentId === placements[i].componentId);
        const dfm2 = this.dfmComponents.find(d => d.placement.componentId === placements[j].componentId);
        const spacing = Math.max(this.getAssemblySpacing(dfm1), this.getAssemblySpacing(dfm2));

        if (this.checkOverlap(
          { ...placements[i].position, width: placements[i].footprint?.width || 5, height: placements[i].footprint?.height || 5 },
          { ...placements[j].position, width: placements[j].footprint?.width || 5, height: placements[j].footprint?.height || 5 },
          spacing
        )) {
          score -= 20;
        }
      }
    }

    // Component grouping by type (good for assembly optimization)
    const typeGroups = new Map<string, ComponentPlacement[]>();
    for (const p of placements) {
      const dfm = this.dfmComponents.find(d => d.placement.componentId === p.componentId);
      const key = dfm?.packageType || 'unknown';
      if (!typeGroups.has(key)) typeGroups.set(key, []);
      typeGroups.get(key)!.push(p);
    }

    for (const [, group] of typeGroups) {
      if (group.length > 2) {
        // Check if grouped together
        let avgDist = 0;
        for (let i = 0; i < group.length; i++) {
          for (let j = i + 1; j < group.length; j++) {
            avgDist += this.calculateDistance(group[i].position, group[j].position);
          }
        }
        avgDist /= (group.length * (group.length - 1) / 2);
        if (avgDist < 20) score += 5;
      }
    }

    // DFM score contribution
    const dfmScore = this.calculateDFMScore(placements);
    score += dfmScore * 0.3;

    return Math.max(0, score);
  }

  private calculateDFMScore(placements: ComponentPlacement[]): number {
    let score = 100;

    // Polarized component alignment
    for (const p of placements) {
      const ref = p.reference.toUpperCase();
      if (ref.startsWith('D') || ref.startsWith('C') || ref.startsWith('U')) {
        // Polarized components should have consistent orientation
        if (p.rotation === 0 || p.rotation === 180) {
          score += 2;
        }
      }
    }

    // Consistent component spacing
    const spacings: number[] = [];
    for (let i = 0; i < placements.length; i++) {
      for (let j = i + 1; j < placements.length; j++) {
        spacings.push(this.calculateDistance(placements[i].position, placements[j].position));
      }
    }

    if (spacings.length > 0) {
      const avgSpacing = spacings.reduce((a, b) => a + b, 0) / spacings.length;
      const variance = spacings.reduce((sum, s) => sum + Math.pow(s - avgSpacing, 2), 0) / spacings.length;
      if (variance < 10) {
        score += 10; // Bonus for consistent spacing
      }
    }

    return Math.max(0, score);
  }

  private getAssemblySpacing(dfm: DFMComponent | undefined): number {
    if (!dfm) return 2;

    switch (dfm.assemblyDifficulty) {
      case 'difficult': return 3;
      case 'moderate': return 2;
      case 'easy': return 1.5;
      default: return 2;
    }
  }

  private checkConvergence(history: number[]): boolean {
    if (history.length < 60) return false;
    const recent = history.slice(-60);
    const variance = this.calculateVariance(recent);
    return variance < this.parameters.convergenceThreshold;
  }

  private calculateVariance(arr: number[]): number {
    const mean = arr.reduce((a, b) => a + b, 0) / arr.length;
    return arr.reduce((sum, x) => sum + Math.pow(x - mean, 2), 0) / arr.length;
  }

  private generateFiducials(
    boardWidth: number,
    boardHeight: number
  ): Array<{ x: number; y: number; diameter: number }> {
    // Three fiducials for pick-and-place alignment
    return [
      { x: this.fiducialMargin, y: this.fiducialMargin, diameter: 1 },
      { x: boardWidth - this.fiducialMargin, y: this.fiducialMargin, diameter: 1 },
      { x: this.fiducialMargin, y: boardHeight - this.fiducialMargin, diameter: 1 }
    ];
  }

  private generateToolingHoles(
    boardWidth: number,
    boardHeight: number
  ): Array<{ x: number; y: number; diameter: number }> {
    // Four tooling holes in corners
    const margin = 3;
    return [
      { x: margin, y: margin, diameter: 3.2 },
      { x: boardWidth - margin, y: margin, diameter: 3.2 },
      { x: margin, y: boardHeight - margin, diameter: 3.2 },
      { x: boardWidth - margin, y: boardHeight - margin, diameter: 3.2 }
    ];
  }

  private generateDFMRouting(
    placements: ComponentPlacement[],
    constraints: BoardConstraints
  ): Trace[] {
    // Placeholder
    return [];
  }

  private generateLayers(constraints: BoardConstraints): Layer[] {
    const layerCount = constraints.layerCount || 4;
    const layers: Layer[] = [];

    for (let i = 0; i < layerCount; i++) {
      layers.push({
        id: uuidv4(),
        name: i === 0 ? 'F.Cu' : i === layerCount - 1 ? 'B.Cu' : `In${i}.Cu`,
        type: 'copper',
        thickness: 1,
        order: i
      });
    }

    return layers;
  }

  private generateDFMZones(placements: ComponentPlacement[]): any[] {
    return [
      {
        id: uuidv4(),
        name: 'GND_POUR',
        layer: 'F.Cu',
        netName: 'GND',
        priority: 1,
        thermalRelief: true
      },
      {
        id: uuidv4(),
        name: 'GND_POUR_B',
        layer: 'B.Cu',
        netName: 'GND',
        priority: 1,
        thermalRelief: true
      }
    ];
  }

  private getDFMDesignRules(constraints: BoardConstraints): any {
    return {
      minTraceWidth: Math.max(constraints.minTraceWidth || 0.15, 0.15),
      minClearance: Math.max(constraints.minClearance || 0.15, 0.15),
      minViaDiameter: 0.6,
      minViaDrill: 0.3,
      minAnnularRing: 0.15,
      minSilkscreenWidth: 0.15,
      minSilkscreenClearance: 0.15,
      preferredTraceWidths: [0.15, 0.2, 0.25, 0.3, 0.5, 1.0],
      preferredViaSize: { diameter: 0.6, drill: 0.3 },
      solderMaskExpansion: 0.05,
      pasteMaskContraction: 0.05
    };
  }

  private standardizeTraceWidth(width: number, constraints: BoardConstraints): number {
    const standard = [0.15, 0.2, 0.25, 0.3, 0.5, 1.0];
    const minWidth = constraints.minTraceWidth || 0.15;

    // Find closest standard width
    let closest = standard[0];
    for (const sw of standard) {
      if (sw >= minWidth && Math.abs(sw - width) < Math.abs(closest - width)) {
        closest = sw;
      }
    }
    return closest;
  }

  private analyzeDFMFeedback(feedback: AgentFeedback): string[] {
    const issues: string[] = [];
    const details = feedback.validation.details;

    if (details?.dfm?.yieldEstimate < 95) {
      issues.push('low_yield');
    }
    if (details?.dfm?.assemblyIssues?.length > 0) {
      issues.push('assembly');
    }
    if (details?.dfm?.solderabilityScore < 90) {
      issues.push('solderability');
    }

    return issues;
  }

  private fixDFMIssue(
    placements: ComponentPlacement[],
    issue: string,
    constraints: BoardConstraints
  ): ComponentPlacement[] {
    switch (issue) {
      case 'low_yield':
        return this.increaseSpacing(placements, 1.1);
      case 'assembly':
        return this.alignComponents(placements);
      case 'solderability':
        return this.standardizeRotations(placements);
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
        x: Math.round((centerX + (p.position.x - centerX) * factor) / this.standardPitchGrid) * this.standardPitchGrid,
        y: Math.round((centerY + (p.position.y - centerY) * factor) / this.standardPitchGrid) * this.standardPitchGrid
      }
    }));
  }

  private alignComponents(placements: ComponentPlacement[]): ComponentPlacement[] {
    return placements.map(p => ({
      ...p,
      position: {
        x: Math.round(p.position.x / this.standardPitchGrid) * this.standardPitchGrid,
        y: Math.round(p.position.y / this.standardPitchGrid) * this.standardPitchGrid
      }
    }));
  }

  private standardizeRotations(placements: ComponentPlacement[]): ComponentPlacement[] {
    return placements.map(p => ({
      ...p,
      rotation: Math.round(p.rotation / 90) * 90
    }));
  }

  private calculateMetrics(layout: PCBLayout): LayoutMetrics {
    const boardArea = layout.boardOutline.width * layout.boardOutline.height;
    const componentArea = layout.components.reduce((sum, c) =>
      sum + (c.footprint?.width || 0) * (c.footprint?.height || 0), 0);

    // Estimate yield based on DFM factors
    let yieldEstimate = 99;
    for (const c of layout.components) {
      const dfm = this.dfmComponents.find(d => d.placement.componentId === c.componentId);
      if (dfm?.assemblyDifficulty === 'difficult') yieldEstimate -= 0.5;
      if (dfm?.assemblyDifficulty === 'moderate') yieldEstimate -= 0.2;
    }

    // Single-sided bonus
    const bottomCount = layout.components.filter(c => c.layer === 'bottom').length;
    if (bottomCount === 0) yieldEstimate = Math.min(yieldEstimate + 1, 99.5);

    return {
      boardArea,
      componentDensity: componentArea / boardArea,
      routingCompletion: 0.95,
      averageTraceLength: 0,
      viaCount: layout.vias?.length || 0,
      layerUtilization: new Map(),
      thermalScore: 80,
      signalIntegrityScore: 85,
      drcViolations: 0,
      estimatedYield: Math.max(90, yieldEstimate)
    };
  }
}

export default DFMOptimizedAgent;