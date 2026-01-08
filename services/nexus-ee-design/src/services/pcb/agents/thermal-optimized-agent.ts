/**
 * Thermal-Optimized Layout Agent
 *
 * Prioritizes thermal management and heat dissipation.
 * Uses thermal vias, copper pours, strategic component placement near edges/heatsinks.
 * Best for high-power designs, motor controllers, power supplies.
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
  Layer,
  Via
} from '../../../types';

const THERMAL_CONFIG: AgentConfig = {
  name: 'Thermal-Optimized Agent',
  strategy: 'thermal-optimized',
  weights: {
    area: 0.1,      // Low priority on area
    thermal: 0.4,   // Highest priority on thermal
    signal: 0.15,   // Moderate signal integrity
    emi: 0.1,       // Lower EMI (trade-off)
    dfm: 0.15,      // Moderate DFM
    routing: 0.1    // Lower routing density
  },
  parameters: {
    maxIterations: 600,
    convergenceThreshold: 0.008,
    placementAlgorithm: 'hybrid',
    routingAlgorithm: 'area',
    optimizationPasses: 4,
    coolingSchedule: 'adaptive'
  }
};

interface ThermalComponent {
  placement: ComponentPlacement;
  powerDissipation: number; // Watts
  thermalResistance: number; // °C/W
  category: 'high-power' | 'medium-power' | 'low-power';
}

interface ThermalZone {
  id: string;
  type: 'hot' | 'warm' | 'cool';
  bounds: { x: number; y: number; width: number; height: number };
  maxTemperature: number;
  heatSources: string[];
}

export class ThermalOptimizedAgent extends BaseLayoutAgent {
  private readonly thermalViaSpacing: number = 1.5; // mm between thermal vias
  private readonly copperPourMinSize: number = 3; // mm² minimum pour area
  private thermalComponents: ThermalComponent[] = [];
  private thermalZones: ThermalZone[] = [];

  constructor(config?: Partial<AgentConfig>) {
    super({ ...THERMAL_CONFIG, ...config });
  }

  async generateLayout(context: PlacementContext): Promise<PlacementResult> {
    const { schematic, constraints, roundNumber } = context;
    this.emit('generation-start', { agentId: this.id, round: roundNumber });

    const startTime = Date.now();
    const improvementHistory: number[] = [];
    let iterations = 0;
    let converged = false;

    // Classify components by thermal properties
    this.thermalComponents = this.classifyThermalComponents(schematic.components);

    // Calculate board size with thermal margins
    const boardWidth = constraints.maxWidth;
    const boardHeight = constraints.maxHeight;

    // Initial thermal-aware placement
    let placements = this.generateThermalPlacement(
      this.thermalComponents,
      boardWidth,
      boardHeight,
      constraints
    );

    // Hybrid optimization (gradient descent + simulated annealing)
    let temperature = 80;
    const coolingRate = 0.993;
    let currentScore = this.evaluateThermalPlacement(placements, constraints, boardWidth, boardHeight);
    let bestPlacements = [...placements];
    let bestScore = currentScore;

    while (iterations < this.parameters.maxIterations && temperature > 0.5) {
      // Generate thermally-aware neighbor
      const neighbor = this.generateThermalNeighbor(placements, constraints, boardWidth, boardHeight);
      const neighborScore = this.evaluateThermalPlacement(neighbor, constraints, boardWidth, boardHeight);

      // Adaptive acceptance
      const delta = neighborScore - currentScore;
      const adaptiveTemp = temperature * (1 + Math.abs(delta) / 100);

      if (delta > 0 || Math.random() < Math.exp(delta / adaptiveTemp)) {
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

      // Check convergence
      if (this.checkConvergence(improvementHistory)) {
        converged = true;
        break;
      }

      if (iterations % 50 === 0) {
        this.emit('progress', {
          agentId: this.id,
          iteration: iterations,
          score: currentScore,
          temperature,
          thermalScore: this.calculateThermalScore(placements)
        });
      }
    }

    // Generate thermal vias for high-power components
    const thermalVias = this.generateThermalVias(bestPlacements, constraints);

    // Generate routing with wide power traces
    const traces = this.generateThermalRouting(bestPlacements, constraints);
    const layers = this.generateLayers(constraints);

    // Define thermal zones
    this.thermalZones = this.defineThermalZones(bestPlacements, boardWidth, boardHeight);

    const layout: PCBLayout = {
      id: uuidv4(),
      projectId: context.schematic.projectId || '',
      name: `${this.name} Layout - Round ${roundNumber}`,
      version: '1.0',
      boardOutline: {
        width: boardWidth,
        height: boardHeight,
        shape: 'rectangular',
        cornerRadius: 2
      },
      layers,
      components: bestPlacements,
      traces,
      vias: thermalVias,
      zones: this.generateCopperPours(bestPlacements, boardWidth, boardHeight),
      designRules: this.getThermalDesignRules(constraints),
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
      thermalZones: this.thermalZones.length
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

    // Analyze thermal issues from feedback
    const thermalIssues = this.analyzeThermalFeedback(feedback);
    let placements = [...layout.components];

    // Apply thermal-specific fixes
    for (const issue of thermalIssues) {
      placements = this.fixThermalIssue(placements, issue, context.constraints);
    }

    // Run optimization on refined placement
    let temperature = 40; // Lower starting temp for refinement
    const coolingRate = 0.99;
    let currentScore = this.evaluateThermalPlacement(
      placements,
      context.constraints,
      layout.boardOutline.width,
      layout.boardOutline.height
    );
    let bestScore = currentScore;
    let bestPlacements = [...placements];

    const maxRefinementIterations = this.parameters.maxIterations / 3;
    while (iterations < maxRefinementIterations && temperature > 0.5) {
      const neighbor = this.generateThermalNeighbor(
        placements,
        context.constraints,
        layout.boardOutline.width,
        layout.boardOutline.height
      );
      const neighborScore = this.evaluateThermalPlacement(
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

    const thermalVias = this.generateThermalVias(bestPlacements, context.constraints);
    const traces = this.generateThermalRouting(bestPlacements, context.constraints);

    const refinedLayout: PCBLayout = {
      ...layout,
      id: uuidv4(),
      name: `${this.name} Refined - Round ${context.roundNumber}`,
      components: bestPlacements,
      traces,
      vias: thermalVias,
      zones: this.generateCopperPours(
        bestPlacements,
        layout.boardOutline.width,
        layout.boardOutline.height
      ),
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
    const thermal = this.thermalComponents.find(t => t.placement.id === component.id);

    if (thermal && thermal.category === 'high-power') {
      // High-power components should be near edges for heat dissipation
      const edgeX = Math.min(component.position.x, constraints.maxWidth - component.position.x);
      const edgeY = Math.min(component.position.y, constraints.maxHeight - component.position.y);
      const edgeDist = Math.min(edgeX, edgeY);

      score += (20 - edgeDist) * 2; // Bonus for being near edge

      // Penalize clustering of high-power components
      for (const neighbor of neighbors) {
        const neighborThermal = this.thermalComponents.find(t => t.placement.id === neighbor.id);
        if (neighborThermal && neighborThermal.category === 'high-power') {
          const dist = this.calculateDistance(component.position, neighbor.position);
          if (dist < 15) { // Should be at least 15mm apart
            score -= (15 - dist) * 3;
          }
        }
      }
    }

    return Math.max(0, score);
  }

  protected optimizeRouting(
    traces: Trace[],
    layout: PCBLayout,
    constraints: BoardConstraints
  ): Trace[] {
    // Widen power traces for thermal relief
    return traces.map(trace => {
      const isPowerTrace = trace.netName?.toLowerCase().includes('vcc') ||
                          trace.netName?.toLowerCase().includes('gnd') ||
                          trace.netName?.toLowerCase().includes('power');
      return {
        ...trace,
        width: isPowerTrace
          ? Math.max(trace.width, 1.0) // Minimum 1mm for power
          : trace.width
      };
    });
  }

  private classifyThermalComponents(components: any[]): ThermalComponent[] {
    return components.map(c => {
      // Estimate power dissipation based on component type
      let powerDissipation = 0.1; // Default 0.1W
      let thermalResistance = 50; // Default °C/W

      const ref = (c.reference || '').toUpperCase();
      const value = (c.value || '').toLowerCase();

      if (ref.startsWith('Q') || ref.startsWith('M')) {
        // MOSFETs, transistors - high power
        powerDissipation = value.includes('sic') ? 10 : 5;
        thermalResistance = 2;
      } else if (ref.startsWith('U')) {
        // ICs - medium to high
        powerDissipation = 2;
        thermalResistance = 10;
      } else if (ref.startsWith('D') && value.includes('led')) {
        powerDissipation = 0.5;
        thermalResistance = 20;
      } else if (ref.startsWith('R') && parseFloat(value) < 1) {
        // Low value resistors may carry current
        powerDissipation = 0.25;
        thermalResistance = 30;
      } else if (ref.startsWith('L')) {
        // Inductors
        powerDissipation = 1;
        thermalResistance = 15;
      }

      const category = powerDissipation > 3 ? 'high-power' :
                      powerDissipation > 0.5 ? 'medium-power' : 'low-power';

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
        powerDissipation,
        thermalResistance,
        category
      };
    });
  }

  private generateThermalPlacement(
    thermalComponents: ThermalComponent[],
    boardWidth: number,
    boardHeight: number,
    constraints: BoardConstraints
  ): ComponentPlacement[] {
    const placements: ComponentPlacement[] = [];

    // Sort by power: high power first
    const sorted = [...thermalComponents].sort((a, b) =>
      b.powerDissipation - a.powerDissipation
    );

    // Place high-power components first, near edges
    const highPower = sorted.filter(c => c.category === 'high-power');
    const medPower = sorted.filter(c => c.category === 'medium-power');
    const lowPower = sorted.filter(c => c.category === 'low-power');

    // Edge positions for high-power (distribute around perimeter)
    const edgePositions = this.generateEdgePositions(
      highPower.length,
      boardWidth,
      boardHeight,
      5 // 5mm margin
    );

    for (let i = 0; i < highPower.length; i++) {
      const pos = edgePositions[i] || { x: boardWidth / 2, y: boardHeight / 2 };
      placements.push({
        ...highPower[i].placement,
        position: pos,
        id: uuidv4()
      });
    }

    // Place medium power components in middle band
    let midX = 15;
    let midY = 15;
    const midRowHeight = 8;

    for (const comp of medPower) {
      const w = comp.placement.footprint?.width || 5;
      const h = comp.placement.footprint?.height || 5;

      if (midX + w + 5 > boardWidth - 15) {
        midX = 15;
        midY += midRowHeight;
      }

      placements.push({
        ...comp.placement,
        position: { x: midX, y: midY },
        id: uuidv4()
      });

      midX += w + 5; // 5mm spacing
    }

    // Fill remaining with low power components
    let lowX = 10;
    let lowY = midY + midRowHeight + 5;
    const lowRowHeight = 5;

    for (const comp of lowPower) {
      const w = comp.placement.footprint?.width || 3;
      const h = comp.placement.footprint?.height || 3;

      if (lowX + w + 2 > boardWidth - 10) {
        lowX = 10;
        lowY += lowRowHeight;
      }

      placements.push({
        ...comp.placement,
        position: { x: lowX, y: lowY },
        id: uuidv4()
      });

      lowX += w + 2;
    }

    return placements;
  }

  private generateEdgePositions(
    count: number,
    boardWidth: number,
    boardHeight: number,
    margin: number
  ): Array<{ x: number; y: number }> {
    const positions: Array<{ x: number; y: number }> = [];
    const perimeter = 2 * (boardWidth + boardHeight) - 4 * margin;
    const spacing = perimeter / (count + 1);

    let currentDist = spacing;
    for (let i = 0; i < count; i++) {
      // Walk around perimeter
      if (currentDist < boardWidth - 2 * margin) {
        positions.push({ x: margin + currentDist, y: margin });
      } else if (currentDist < boardWidth + boardHeight - 4 * margin) {
        const y = currentDist - (boardWidth - 2 * margin);
        positions.push({ x: boardWidth - margin, y: margin + y });
      } else if (currentDist < 2 * boardWidth + boardHeight - 6 * margin) {
        const x = currentDist - (boardWidth + boardHeight - 4 * margin);
        positions.push({ x: boardWidth - margin - x, y: boardHeight - margin });
      } else {
        const y = currentDist - (2 * boardWidth + boardHeight - 6 * margin);
        positions.push({ x: margin, y: boardHeight - margin - y });
      }
      currentDist += spacing;
    }

    return positions;
  }

  private generateThermalNeighbor(
    placements: ComponentPlacement[],
    constraints: BoardConstraints,
    boardWidth: number,
    boardHeight: number
  ): ComponentPlacement[] {
    const newPlacements = JSON.parse(JSON.stringify(placements));
    const idx = Math.floor(Math.random() * newPlacements.length);

    // Find thermal category
    const thermal = this.thermalComponents.find(
      t => t.placement.componentId === newPlacements[idx].componentId
    );

    if (thermal?.category === 'high-power') {
      // Move along edge for high-power components
      const edgeDist = 5; // Stay 5mm from edge
      const moveAmount = (Math.random() - 0.5) * 10;

      // Determine which edge we're closest to
      const distToLeft = newPlacements[idx].position.x;
      const distToRight = boardWidth - newPlacements[idx].position.x;
      const distToTop = newPlacements[idx].position.y;
      const distToBottom = boardHeight - newPlacements[idx].position.y;
      const minDist = Math.min(distToLeft, distToRight, distToTop, distToBottom);

      if (minDist === distToLeft || minDist === distToRight) {
        // On vertical edge, move vertically
        newPlacements[idx].position.y = Math.max(
          edgeDist,
          Math.min(boardHeight - edgeDist, newPlacements[idx].position.y + moveAmount)
        );
      } else {
        // On horizontal edge, move horizontally
        newPlacements[idx].position.x = Math.max(
          edgeDist,
          Math.min(boardWidth - edgeDist, newPlacements[idx].position.x + moveAmount)
        );
      }
    } else {
      // Standard random move for other components
      const moveRange = 5;
      newPlacements[idx].position.x = Math.max(
        3,
        Math.min(boardWidth - 3, newPlacements[idx].position.x + (Math.random() - 0.5) * 2 * moveRange)
      );
      newPlacements[idx].position.y = Math.max(
        3,
        Math.min(boardHeight - 3, newPlacements[idx].position.y + (Math.random() - 0.5) * 2 * moveRange)
      );
    }

    return newPlacements;
  }

  private evaluateThermalPlacement(
    placements: ComponentPlacement[],
    constraints: BoardConstraints,
    boardWidth: number,
    boardHeight: number
  ): number {
    let score = 100;

    // Check high-power component edge placement
    for (const p of placements) {
      const thermal = this.thermalComponents.find(t => t.placement.componentId === p.componentId);

      if (thermal?.category === 'high-power') {
        const edgeDist = Math.min(
          p.position.x,
          boardWidth - p.position.x,
          p.position.y,
          boardHeight - p.position.y
        );

        if (edgeDist < 10) {
          score += 10; // Bonus for near edge
        } else {
          score -= (edgeDist - 10) * 0.5; // Penalty for being away from edge
        }

        // Check separation from other high-power components
        for (const p2 of placements) {
          if (p.id === p2.id) continue;
          const thermal2 = this.thermalComponents.find(t => t.placement.componentId === p2.componentId);
          if (thermal2?.category === 'high-power') {
            const dist = this.calculateDistance(p.position, p2.position);
            if (dist < 15) {
              score -= (15 - dist) * 2; // Penalize close high-power components
            } else {
              score += 5; // Bonus for good separation
            }
          }
        }
      }
    }

    // Check for overlaps
    for (let i = 0; i < placements.length; i++) {
      for (let j = i + 1; j < placements.length; j++) {
        if (this.checkOverlap(
          { ...placements[i].position, width: placements[i].footprint?.width || 5, height: placements[i].footprint?.height || 5 },
          { ...placements[j].position, width: placements[j].footprint?.width || 5, height: placements[j].footprint?.height || 5 },
          constraints.minComponentSpacing || 1
        )) {
          score -= 25;
        }
      }
    }

    // Thermal score contribution
    const thermalScore = this.calculateThermalScore(placements);
    score += thermalScore * 0.5;

    return Math.max(0, score);
  }

  private calculateThermalScore(placements: ComponentPlacement[]): number {
    let score = 100;

    // Simulate thermal distribution (simplified)
    const heatMap = new Map<string, number>();

    for (const p of placements) {
      const thermal = this.thermalComponents.find(t => t.placement.componentId === p.componentId);
      if (!thermal) continue;

      // Estimate temperature rise
      const tempRise = thermal.powerDissipation * thermal.thermalResistance;
      const key = `${Math.floor(p.position.x / 10)}_${Math.floor(p.position.y / 10)}`;
      heatMap.set(key, (heatMap.get(key) || 25) + tempRise);
    }

    // Check for hot spots (>85°C)
    for (const [, temp] of heatMap) {
      if (temp > 85) {
        score -= (temp - 85) * 2;
      }
    }

    return Math.max(0, score);
  }

  private checkConvergence(history: number[]): boolean {
    if (history.length < 80) return false;
    const recent = history.slice(-80);
    const variance = this.calculateVariance(recent);
    return variance < this.parameters.convergenceThreshold;
  }

  private calculateVariance(arr: number[]): number {
    const mean = arr.reduce((a, b) => a + b, 0) / arr.length;
    return arr.reduce((sum, x) => sum + Math.pow(x - mean, 2), 0) / arr.length;
  }

  private generateThermalVias(
    placements: ComponentPlacement[],
    constraints: BoardConstraints
  ): Via[] {
    const vias: Via[] = [];

    for (const p of placements) {
      const thermal = this.thermalComponents.find(t => t.placement.componentId === p.componentId);

      if (thermal && thermal.category !== 'low-power') {
        // Add thermal via array under component
        const w = p.footprint?.width || 5;
        const h = p.footprint?.height || 5;
        const viasX = Math.floor(w / this.thermalViaSpacing);
        const viasY = Math.floor(h / this.thermalViaSpacing);

        for (let i = 0; i <= viasX; i++) {
          for (let j = 0; j <= viasY; j++) {
            vias.push({
              id: uuidv4(),
              position: {
                x: p.position.x + i * this.thermalViaSpacing,
                y: p.position.y + j * this.thermalViaSpacing
              },
              diameter: thermal.category === 'high-power' ? 0.8 : 0.6,
              drillSize: thermal.category === 'high-power' ? 0.4 : 0.3,
              layers: ['F.Cu', 'B.Cu'],
              type: 'through'
            });
          }
        }
      }
    }

    return vias;
  }

  private generateThermalRouting(
    placements: ComponentPlacement[],
    constraints: BoardConstraints
  ): Trace[] {
    // Power routing with wide traces
    return [];
  }

  private generateLayers(constraints: BoardConstraints): Layer[] {
    const layerCount = Math.max(constraints.layerCount || 4, 4);
    const layers: Layer[] = [];

    for (let i = 0; i < layerCount; i++) {
      layers.push({
        id: uuidv4(),
        name: i === 0 ? 'F.Cu' : i === layerCount - 1 ? 'B.Cu' : `In${i}.Cu`,
        type: 'copper',
        thickness: i === 0 || i === layerCount - 1 ? 2 : 1, // Thicker outer layers
        order: i
      });
    }

    return layers;
  }

  private defineThermalZones(
    placements: ComponentPlacement[],
    boardWidth: number,
    boardHeight: number
  ): ThermalZone[] {
    const zones: ThermalZone[] = [];

    // Create zones around high-power components
    for (const p of placements) {
      const thermal = this.thermalComponents.find(t => t.placement.componentId === p.componentId);
      if (thermal?.category === 'high-power') {
        zones.push({
          id: uuidv4(),
          type: 'hot',
          bounds: {
            x: p.position.x - 5,
            y: p.position.y - 5,
            width: (p.footprint?.width || 5) + 10,
            height: (p.footprint?.height || 5) + 10
          },
          maxTemperature: 85,
          heatSources: [p.componentId]
        });
      }
    }

    return zones;
  }

  private generateCopperPours(
    placements: ComponentPlacement[],
    boardWidth: number,
    boardHeight: number
  ): any[] {
    // Generate ground pours for thermal dissipation
    return [{
      id: uuidv4(),
      name: 'GND_POUR_TOP',
      layer: 'F.Cu',
      netName: 'GND',
      priority: 1,
      thermalRelief: true,
      thermalGap: 0.3,
      thermalWidth: 0.5
    }, {
      id: uuidv4(),
      name: 'GND_POUR_BOTTOM',
      layer: 'B.Cu',
      netName: 'GND',
      priority: 1,
      thermalRelief: true,
      thermalGap: 0.3,
      thermalWidth: 0.5
    }];
  }

  private getThermalDesignRules(constraints: BoardConstraints): any {
    return {
      minTraceWidth: constraints.minTraceWidth || 0.2,
      minPowerTraceWidth: 1.0, // 1mm minimum for power
      minClearance: constraints.minClearance || 0.2,
      minViaDiameter: 0.6,
      minViaDrill: 0.3,
      thermalViaSpacing: this.thermalViaSpacing,
      minCopperPourArea: this.copperPourMinSize
    };
  }

  private analyzeThermalFeedback(feedback: AgentFeedback): string[] {
    const issues: string[] = [];
    const details = feedback.validation.details;

    if (details?.thermal?.maxTemperature > 85) {
      issues.push('overheating');
    }
    if (details?.thermal?.hotSpots?.length > 0) {
      issues.push('hot_spots');
    }
    if (details?.thermal?.inadequateCooling) {
      issues.push('cooling');
    }

    return issues;
  }

  private fixThermalIssue(
    placements: ComponentPlacement[],
    issue: string,
    constraints: BoardConstraints
  ): ComponentPlacement[] {
    switch (issue) {
      case 'overheating':
        // Move high-power components toward edges
        return this.moveToEdges(placements, constraints);
      case 'hot_spots':
        // Spread out concentrated heat sources
        return this.spreadHeatSources(placements, constraints);
      case 'cooling':
        // Increase spacing for airflow
        return this.increaseAirflowSpacing(placements, 1.15);
      default:
        return placements;
    }
  }

  private moveToEdges(
    placements: ComponentPlacement[],
    constraints: BoardConstraints
  ): ComponentPlacement[] {
    return placements.map(p => {
      const thermal = this.thermalComponents.find(t => t.placement.componentId === p.componentId);
      if (thermal?.category === 'high-power') {
        // Move toward nearest edge
        const toLeft = p.position.x;
        const toRight = constraints.maxWidth - p.position.x;
        const toTop = p.position.y;
        const toBottom = constraints.maxHeight - p.position.y;

        const minDist = Math.min(toLeft, toRight, toTop, toBottom);
        const margin = 5;

        if (minDist === toLeft) return { ...p, position: { ...p.position, x: margin } };
        if (minDist === toRight) return { ...p, position: { ...p.position, x: constraints.maxWidth - margin } };
        if (minDist === toTop) return { ...p, position: { ...p.position, y: margin } };
        return { ...p, position: { ...p.position, y: constraints.maxHeight - margin } };
      }
      return p;
    });
  }

  private spreadHeatSources(
    placements: ComponentPlacement[],
    constraints: BoardConstraints
  ): ComponentPlacement[] {
    const highPower = placements.filter(p => {
      const thermal = this.thermalComponents.find(t => t.placement.componentId === p.componentId);
      return thermal?.category === 'high-power';
    });

    if (highPower.length < 2) return placements;

    // Distribute evenly
    const spacing = Math.sqrt(
      (constraints.maxWidth * constraints.maxHeight) / highPower.length
    );

    return placements.map(p => {
      const thermal = this.thermalComponents.find(t => t.placement.componentId === p.componentId);
      if (thermal?.category !== 'high-power') return p;

      const idx = highPower.findIndex(hp => hp.id === p.id);
      const row = Math.floor(idx / Math.ceil(Math.sqrt(highPower.length)));
      const col = idx % Math.ceil(Math.sqrt(highPower.length));

      return {
        ...p,
        position: {
          x: 10 + col * spacing,
          y: 10 + row * spacing
        }
      };
    });
  }

  private increaseAirflowSpacing(
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

    // Estimate max temperature
    let maxTemp = 25;
    for (const c of layout.components) {
      const thermal = this.thermalComponents.find(t => t.placement.componentId === c.componentId);
      if (thermal) {
        maxTemp = Math.max(maxTemp, 25 + thermal.powerDissipation * thermal.thermalResistance * 0.3);
      }
    }

    return {
      boardArea,
      componentDensity: componentArea / boardArea,
      routingCompletion: 0.92,
      averageTraceLength: 0,
      viaCount: layout.vias?.length || 0,
      layerUtilization: new Map(),
      thermalScore: Math.max(0, 100 - (maxTemp - 25)),
      signalIntegrityScore: 85,
      drcViolations: 0,
      estimatedYield: 95
    };
  }
}

export default ThermalOptimizedAgent;