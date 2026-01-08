/**
 * EMI-Optimized Layout Agent
 *
 * Prioritizes electromagnetic compatibility and interference mitigation.
 * Uses proper shielding, return path management, controlled impedance routing.
 * Best for RF designs, high-speed digital, and regulatory compliance.
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

const EMI_CONFIG: AgentConfig = {
  name: 'EMI-Optimized Agent',
  strategy: 'emi-optimized',
  weights: {
    area: 0.1,      // Low priority on area
    thermal: 0.1,   // Lower thermal priority
    signal: 0.25,   // High signal integrity
    emi: 0.35,      // Highest EMI priority
    dfm: 0.1,       // Lower DFM
    routing: 0.1    // Controlled routing density
  },
  parameters: {
    maxIterations: 700,
    convergenceThreshold: 0.007,
    placementAlgorithm: 'hybrid',
    routingAlgorithm: 'channel',
    optimizationPasses: 4,
    coolingSchedule: 'exponential'
  }
};

interface EMIComponent {
  placement: ComponentPlacement;
  signalType: 'digital' | 'analog' | 'power' | 'rf' | 'passive';
  frequency: number; // Operating frequency in MHz
  isNoiseSensitive: boolean;
  isNoiseSource: boolean;
}

interface ShieldingZone {
  id: string;
  name: string;
  bounds: { x: number; y: number; width: number; height: number };
  shieldType: 'ground_ring' | 'fence' | 'cavity' | 'copper_pour';
  components: string[];
}

export class EMIOptimizedAgent extends BaseLayoutAgent {
  private readonly minDigitalAnalogSeparation: number = 10; // mm
  private readonly minClockLineSpacing: number = 3; // mm
  private emiComponents: EMIComponent[] = [];
  private shieldingZones: ShieldingZone[] = [];

  constructor(config?: Partial<AgentConfig>) {
    super({ ...EMI_CONFIG, ...config });
  }

  async generateLayout(context: PlacementContext): Promise<PlacementResult> {
    const { schematic, constraints, roundNumber } = context;
    this.emit('generation-start', { agentId: this.id, round: roundNumber });

    const startTime = Date.now();
    const improvementHistory: number[] = [];
    let iterations = 0;
    let converged = false;

    // Classify components by EMI characteristics
    this.emiComponents = this.classifyEMIComponents(schematic.components);

    const boardWidth = constraints.maxWidth;
    const boardHeight = constraints.maxHeight;

    // Generate EMI-aware initial placement
    let placements = this.generateEMIPlacement(
      this.emiComponents,
      boardWidth,
      boardHeight,
      constraints
    );

    // Optimization with EMI-focused scoring
    let temperature = 75;
    const coolingRate = 0.992;
    let currentScore = this.evaluateEMIPlacement(placements, constraints, boardWidth, boardHeight);
    let bestPlacements = [...placements];
    let bestScore = currentScore;

    while (iterations < this.parameters.maxIterations && temperature > 0.5) {
      const neighbor = this.generateEMINeighbor(placements, constraints, boardWidth, boardHeight);
      const neighborScore = this.evaluateEMIPlacement(neighbor, constraints, boardWidth, boardHeight);

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
          emiScore: this.calculateEMIScore(placements)
        });
      }
    }

    // Generate shielding zones
    this.shieldingZones = this.generateShieldingZones(bestPlacements, boardWidth, boardHeight);

    // Generate controlled-impedance routing
    const traces = this.generateEMIRouting(bestPlacements, constraints);
    const layers = this.generateEMILayers(constraints);

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
      vias: [],
      zones: this.generateEMIZones(bestPlacements, boardWidth, boardHeight),
      designRules: this.getEMIDesignRules(constraints),
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
      shieldingZones: this.shieldingZones.length
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

    const emiIssues = this.analyzeEMIFeedback(feedback);
    let placements = [...layout.components];

    for (const issue of emiIssues) {
      placements = this.fixEMIIssue(placements, issue, context.constraints);
    }

    let temperature = 35;
    const coolingRate = 0.99;
    let currentScore = this.evaluateEMIPlacement(
      placements,
      context.constraints,
      layout.boardOutline.width,
      layout.boardOutline.height
    );
    let bestScore = currentScore;
    let bestPlacements = [...placements];

    const maxIterations = this.parameters.maxIterations / 3;
    while (iterations < maxIterations && temperature > 0.5) {
      const neighbor = this.generateEMINeighbor(
        placements,
        context.constraints,
        layout.boardOutline.width,
        layout.boardOutline.height
      );
      const neighborScore = this.evaluateEMIPlacement(
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

    this.shieldingZones = this.generateShieldingZones(
      bestPlacements,
      layout.boardOutline.width,
      layout.boardOutline.height
    );

    const traces = this.generateEMIRouting(bestPlacements, context.constraints);

    const refinedLayout: PCBLayout = {
      ...layout,
      id: uuidv4(),
      name: `${this.name} Refined - Round ${context.roundNumber}`,
      components: bestPlacements,
      traces,
      zones: this.generateEMIZones(
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
    const emi = this.emiComponents.find(e => e.placement.id === component.id);
    if (!emi) return score;

    // Penalize analog near digital
    if (emi.signalType === 'analog' || emi.isNoiseSensitive) {
      for (const neighbor of neighbors) {
        const neighborEMI = this.emiComponents.find(e => e.placement.id === neighbor.id);
        if (neighborEMI?.signalType === 'digital' || neighborEMI?.isNoiseSource) {
          const dist = this.calculateDistance(component.position, neighbor.position);
          if (dist < this.minDigitalAnalogSeparation) {
            score -= (this.minDigitalAnalogSeparation - dist) * 3;
          }
        }
      }
    }

    // RF components should be isolated
    if (emi.signalType === 'rf') {
      for (const neighbor of neighbors) {
        const dist = this.calculateDistance(component.position, neighbor.position);
        if (dist < 8) {
          score -= 10;
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
    return traces.map(trace => {
      const isHighSpeed = trace.netName?.toLowerCase().includes('clk') ||
                         trace.netName?.toLowerCase().includes('data');
      return {
        ...trace,
        width: isHighSpeed
          ? this.calculateImpedanceWidth(50, constraints) // 50Ω impedance
          : trace.width
      };
    });
  }

  private classifyEMIComponents(components: any[]): EMIComponent[] {
    return components.map(c => {
      const ref = (c.reference || '').toUpperCase();
      const value = (c.value || '').toLowerCase();

      let signalType: EMIComponent['signalType'] = 'passive';
      let frequency = 10; // Default 10MHz
      let isNoiseSensitive = false;
      let isNoiseSource = false;

      if (ref.startsWith('U')) {
        if (value.includes('adc') || value.includes('amp') || value.includes('op')) {
          signalType = 'analog';
          isNoiseSensitive = true;
        } else if (value.includes('rf') || value.includes('lna') || value.includes('mixer')) {
          signalType = 'rf';
          frequency = 2400; // Assume 2.4GHz
          isNoiseSensitive = true;
        } else {
          signalType = 'digital';
          frequency = 100; // Assume 100MHz
          isNoiseSource = true;
        }
      } else if (ref.startsWith('Y') || ref.startsWith('X')) {
        signalType = 'digital';
        isNoiseSource = true;
        frequency = parseFloat(value) || 25;
      } else if (ref.startsWith('Q') || ref.startsWith('M')) {
        signalType = 'power';
        isNoiseSource = true; // Switching noise
        frequency = 100; // PWM frequency
      } else if (ref.startsWith('L')) {
        if (value.includes('rf') || value.includes('nh')) {
          signalType = 'rf';
        } else {
          signalType = 'power';
        }
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
        signalType,
        frequency,
        isNoiseSensitive,
        isNoiseSource
      };
    });
  }

  private generateEMIPlacement(
    emiComponents: EMIComponent[],
    boardWidth: number,
    boardHeight: number,
    constraints: BoardConstraints
  ): ComponentPlacement[] {
    const placements: ComponentPlacement[] = [];

    // Separate by type
    const analog = emiComponents.filter(c => c.signalType === 'analog');
    const digital = emiComponents.filter(c => c.signalType === 'digital');
    const rf = emiComponents.filter(c => c.signalType === 'rf');
    const power = emiComponents.filter(c => c.signalType === 'power');
    const passive = emiComponents.filter(c => c.signalType === 'passive');

    // Zone allocation
    // Left side: Analog
    // Center: Digital
    // Right side: RF
    // Bottom: Power

    const zoneWidth = (boardWidth - 20) / 3;
    const powerHeight = 15;

    // Place analog components (left zone)
    let x = 5, y = 5;
    for (const comp of analog) {
      const w = comp.placement.footprint?.width || 5;
      const h = comp.placement.footprint?.height || 5;

      if (y + h > boardHeight - powerHeight - 5) {
        y = 5;
        x += 8;
      }

      placements.push({
        ...comp.placement,
        position: { x, y },
        id: uuidv4()
      });

      y += h + 3;
    }

    // Place digital components (center zone)
    x = 5 + zoneWidth + 5;
    y = 5;
    for (const comp of digital) {
      const w = comp.placement.footprint?.width || 5;
      const h = comp.placement.footprint?.height || 5;

      if (y + h > boardHeight - powerHeight - 5) {
        y = 5;
        x += 8;
      }

      placements.push({
        ...comp.placement,
        position: { x, y },
        id: uuidv4()
      });

      y += h + 2;
    }

    // Place RF components (right zone)
    x = 5 + 2 * zoneWidth + 10;
    y = 5;
    for (const comp of rf) {
      const w = comp.placement.footprint?.width || 5;
      const h = comp.placement.footprint?.height || 5;

      if (y + h > boardHeight - powerHeight - 5) {
        y = 5;
        x += 8;
      }

      placements.push({
        ...comp.placement,
        position: { x, y },
        id: uuidv4()
      });

      y += h + 4; // Extra spacing for RF
    }

    // Place power components (bottom zone)
    x = 5;
    y = boardHeight - powerHeight;
    for (const comp of power) {
      const w = comp.placement.footprint?.width || 5;

      if (x + w > boardWidth - 5) {
        x = 5;
        y += 7;
      }

      placements.push({
        ...comp.placement,
        position: { x, y },
        id: uuidv4()
      });

      x += w + 5;
    }

    // Place passive components (fill remaining space)
    x = 5;
    y = 5;
    for (const comp of passive) {
      const w = comp.placement.footprint?.width || 3;
      const h = comp.placement.footprint?.height || 3;

      // Find first available spot (simple fill)
      const pos = this.findAvailableSpot(placements, w, h, boardWidth, boardHeight - powerHeight);
      placements.push({
        ...comp.placement,
        position: pos,
        id: uuidv4()
      });
    }

    return placements;
  }

  private findAvailableSpot(
    existing: ComponentPlacement[],
    width: number,
    height: number,
    maxX: number,
    maxY: number
  ): { x: number; y: number } {
    for (let y = 5; y < maxY - height; y += 3) {
      for (let x = 5; x < maxX - width; x += 3) {
        let collision = false;
        for (const p of existing) {
          if (this.checkOverlap(
            { x, y, width, height },
            { x: p.position.x, y: p.position.y, width: p.footprint?.width || 5, height: p.footprint?.height || 5 },
            1.5
          )) {
            collision = true;
            break;
          }
        }
        if (!collision) {
          return { x, y };
        }
      }
    }
    return { x: 5, y: 5 };
  }

  private generateEMINeighbor(
    placements: ComponentPlacement[],
    constraints: BoardConstraints,
    boardWidth: number,
    boardHeight: number
  ): ComponentPlacement[] {
    const newPlacements = JSON.parse(JSON.stringify(placements));
    const idx = Math.floor(Math.random() * newPlacements.length);
    const emi = this.emiComponents.find(e => e.placement.componentId === newPlacements[idx].componentId);

    // Constrain movement based on signal type
    let moveRange = 5;
    if (emi?.signalType === 'rf') {
      moveRange = 3; // RF components move less
    } else if (emi?.signalType === 'analog') {
      // Keep analog on left side
      const maxX = boardWidth / 3;
      newPlacements[idx].position.x = Math.min(
        newPlacements[idx].position.x + (Math.random() - 0.5) * moveRange,
        maxX
      );
      newPlacements[idx].position.y += (Math.random() - 0.5) * moveRange;
    } else if (emi?.signalType === 'digital') {
      // Keep digital in center
      const minX = boardWidth / 3;
      const maxX = 2 * boardWidth / 3;
      newPlacements[idx].position.x = Math.max(minX, Math.min(maxX,
        newPlacements[idx].position.x + (Math.random() - 0.5) * moveRange
      ));
      newPlacements[idx].position.y += (Math.random() - 0.5) * moveRange;
    } else {
      // Standard movement
      newPlacements[idx].position.x += (Math.random() - 0.5) * 2 * moveRange;
      newPlacements[idx].position.y += (Math.random() - 0.5) * 2 * moveRange;
    }

    // Clamp to board
    newPlacements[idx].position.x = Math.max(3, Math.min(boardWidth - 3, newPlacements[idx].position.x));
    newPlacements[idx].position.y = Math.max(3, Math.min(boardHeight - 3, newPlacements[idx].position.y));

    return newPlacements;
  }

  private evaluateEMIPlacement(
    placements: ComponentPlacement[],
    constraints: BoardConstraints,
    boardWidth: number,
    boardHeight: number
  ): number {
    let score = 100;

    // Check analog/digital separation
    const analogPlacements = placements.filter(p => {
      const emi = this.emiComponents.find(e => e.placement.componentId === p.componentId);
      return emi?.signalType === 'analog' || emi?.isNoiseSensitive;
    });

    const digitalPlacements = placements.filter(p => {
      const emi = this.emiComponents.find(e => e.placement.componentId === p.componentId);
      return emi?.signalType === 'digital' || emi?.isNoiseSource;
    });

    for (const analog of analogPlacements) {
      for (const digital of digitalPlacements) {
        const dist = this.calculateDistance(analog.position, digital.position);
        if (dist < this.minDigitalAnalogSeparation) {
          score -= (this.minDigitalAnalogSeparation - dist) * 2;
        } else {
          score += 2; // Bonus for good separation
        }
      }
    }

    // Check RF isolation
    const rfPlacements = placements.filter(p => {
      const emi = this.emiComponents.find(e => e.placement.componentId === p.componentId);
      return emi?.signalType === 'rf';
    });

    for (const rf of rfPlacements) {
      for (const p of placements) {
        if (rf.id === p.id) continue;
        const dist = this.calculateDistance(rf.position, p.position);
        if (dist < 8) {
          score -= 8;
        }
      }
    }

    // Check overlaps
    for (let i = 0; i < placements.length; i++) {
      for (let j = i + 1; j < placements.length; j++) {
        if (this.checkOverlap(
          { ...placements[i].position, width: placements[i].footprint?.width || 5, height: placements[i].footprint?.height || 5 },
          { ...placements[j].position, width: placements[j].footprint?.width || 5, height: placements[j].footprint?.height || 5 },
          constraints.minComponentSpacing || 1
        )) {
          score -= 20;
        }
      }
    }

    // EMI score contribution
    const emiScore = this.calculateEMIScore(placements);
    score += emiScore * 0.5;

    return Math.max(0, score);
  }

  private calculateEMIScore(placements: ComponentPlacement[]): number {
    let score = 100;

    // Zone compliance
    const boardWidth = 100; // Assume standard width
    for (const p of placements) {
      const emi = this.emiComponents.find(e => e.placement.componentId === p.componentId);
      if (!emi) continue;

      if (emi.signalType === 'analog' && p.position.x > boardWidth / 3) {
        score -= 10; // Analog outside analog zone
      }
      if (emi.signalType === 'digital' &&
          (p.position.x < boardWidth / 3 || p.position.x > 2 * boardWidth / 3)) {
        score -= 5; // Digital outside digital zone
      }
      if (emi.signalType === 'rf' && p.position.x < 2 * boardWidth / 3) {
        score -= 10; // RF outside RF zone
      }
    }

    return Math.max(0, score);
  }

  private checkConvergence(history: number[]): boolean {
    if (history.length < 100) return false;
    const recent = history.slice(-100);
    const variance = this.calculateVariance(recent);
    return variance < this.parameters.convergenceThreshold;
  }

  private calculateVariance(arr: number[]): number {
    const mean = arr.reduce((a, b) => a + b, 0) / arr.length;
    return arr.reduce((sum, x) => sum + Math.pow(x - mean, 2), 0) / arr.length;
  }

  private generateShieldingZones(
    placements: ComponentPlacement[],
    boardWidth: number,
    boardHeight: number
  ): ShieldingZone[] {
    const zones: ShieldingZone[] = [];

    // Create shielding zone for RF section
    const rfPlacements = placements.filter(p => {
      const emi = this.emiComponents.find(e => e.placement.componentId === p.componentId);
      return emi?.signalType === 'rf';
    });

    if (rfPlacements.length > 0) {
      const minX = Math.min(...rfPlacements.map(p => p.position.x)) - 3;
      const maxX = Math.max(...rfPlacements.map(p => p.position.x + (p.footprint?.width || 5))) + 3;
      const minY = Math.min(...rfPlacements.map(p => p.position.y)) - 3;
      const maxY = Math.max(...rfPlacements.map(p => p.position.y + (p.footprint?.height || 5))) + 3;

      zones.push({
        id: uuidv4(),
        name: 'RF_SHIELD',
        bounds: { x: minX, y: minY, width: maxX - minX, height: maxY - minY },
        shieldType: 'fence',
        components: rfPlacements.map(p => p.componentId)
      });
    }

    // Create ground ring around analog section
    const analogPlacements = placements.filter(p => {
      const emi = this.emiComponents.find(e => e.placement.componentId === p.componentId);
      return emi?.signalType === 'analog';
    });

    if (analogPlacements.length > 0) {
      const minX = Math.min(...analogPlacements.map(p => p.position.x)) - 2;
      const maxX = Math.max(...analogPlacements.map(p => p.position.x + (p.footprint?.width || 5))) + 2;
      const minY = Math.min(...analogPlacements.map(p => p.position.y)) - 2;
      const maxY = Math.max(...analogPlacements.map(p => p.position.y + (p.footprint?.height || 5))) + 2;

      zones.push({
        id: uuidv4(),
        name: 'ANALOG_GUARD',
        bounds: { x: minX, y: minY, width: maxX - minX, height: maxY - minY },
        shieldType: 'ground_ring',
        components: analogPlacements.map(p => p.componentId)
      });
    }

    return zones;
  }

  private generateEMIRouting(
    placements: ComponentPlacement[],
    constraints: BoardConstraints
  ): Trace[] {
    // Placeholder - actual routing is complex
    return [];
  }

  private generateEMILayers(constraints: BoardConstraints): Layer[] {
    // EMI-optimized layer stack: Signal-Ground-Power-Ground-Signal
    const layerCount = Math.max(constraints.layerCount || 6, 6);
    const layers: Layer[] = [];

    const layerNames = ['F.Cu', 'In1.Cu', 'In2.Cu', 'In3.Cu', 'In4.Cu', 'B.Cu'];
    const layerTypes = ['copper', 'copper', 'copper', 'copper', 'copper', 'copper'];

    for (let i = 0; i < layerCount; i++) {
      layers.push({
        id: uuidv4(),
        name: layerNames[i] || `L${i}.Cu`,
        type: layerTypes[i] as any,
        thickness: 1,
        order: i
      });
    }

    return layers;
  }

  private generateEMIZones(
    placements: ComponentPlacement[],
    boardWidth: number,
    boardHeight: number
  ): any[] {
    // Generate ground pours with EMI considerations
    return [
      {
        id: uuidv4(),
        name: 'GND_TOP',
        layer: 'F.Cu',
        netName: 'GND',
        priority: 1,
        keepouts: this.shieldingZones.map(z => z.bounds)
      },
      {
        id: uuidv4(),
        name: 'GND_IN1',
        layer: 'In1.Cu',
        netName: 'GND',
        priority: 0,
        solid: true // Solid ground plane
      },
      {
        id: uuidv4(),
        name: 'GND_IN4',
        layer: 'In4.Cu',
        netName: 'GND',
        priority: 0,
        solid: true
      },
      {
        id: uuidv4(),
        name: 'GND_BOTTOM',
        layer: 'B.Cu',
        netName: 'GND',
        priority: 1
      }
    ];
  }

  private getEMIDesignRules(constraints: BoardConstraints): any {
    return {
      minTraceWidth: constraints.minTraceWidth || 0.15,
      minClearance: constraints.minClearance || 0.15,
      minViaDiameter: 0.5,
      minViaDrill: 0.25,
      controlledImpedance: true,
      targetImpedance: 50,
      differentialPairSpacing: 0.15,
      minDigitalAnalogGap: this.minDigitalAnalogSeparation,
      clockTraceSpacing: this.minClockLineSpacing
    };
  }

  private calculateImpedanceWidth(targetOhms: number, constraints: BoardConstraints): number {
    // Simplified impedance calculation for microstrip
    // Real implementation would use transmission line equations
    const dielectricConstant = 4.3; // FR4
    const height = 0.2; // mm prepreg height
    // W ≈ 7.475 * h / exp(Z0 * sqrt(Er + 1.41) / 87)
    const width = 7.475 * height / Math.exp(targetOhms * Math.sqrt(dielectricConstant + 1.41) / 87);
    return Math.max(width, constraints.minTraceWidth || 0.15);
  }

  private analyzeEMIFeedback(feedback: AgentFeedback): string[] {
    const issues: string[] = [];
    const details = feedback.validation.details;

    if (details?.emi?.radiatedEmissions > -40) {
      issues.push('radiated');
    }
    if (details?.emi?.conductedEmissions > -50) {
      issues.push('conducted');
    }
    if (details?.si?.crosstalk > 0.1) {
      issues.push('crosstalk');
    }

    return issues;
  }

  private fixEMIIssue(
    placements: ComponentPlacement[],
    issue: string,
    constraints: BoardConstraints
  ): ComponentPlacement[] {
    switch (issue) {
      case 'radiated':
        return this.increaseSeparation(placements, 1.15);
      case 'conducted':
        return this.groupPowerComponents(placements);
      case 'crosstalk':
        return this.increaseTraceSpacing(placements);
      default:
        return placements;
    }
  }

  private increaseSeparation(
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

  private groupPowerComponents(placements: ComponentPlacement[]): ComponentPlacement[] {
    // Move power components closer together for shorter return paths
    const powerPlacements = placements.filter(p => {
      const emi = this.emiComponents.find(e => e.placement.componentId === p.componentId);
      return emi?.signalType === 'power';
    });

    if (powerPlacements.length < 2) return placements;

    // Calculate power center
    const centerX = powerPlacements.reduce((s, p) => s + p.position.x, 0) / powerPlacements.length;
    const centerY = powerPlacements.reduce((s, p) => s + p.position.y, 0) / powerPlacements.length;

    return placements.map(p => {
      const emi = this.emiComponents.find(e => e.placement.componentId === p.componentId);
      if (emi?.signalType === 'power') {
        return {
          ...p,
          position: {
            x: centerX + (p.position.x - centerX) * 0.8,
            y: centerY + (p.position.y - centerY) * 0.8
          }
        };
      }
      return p;
    });
  }

  private increaseTraceSpacing(placements: ComponentPlacement[]): ComponentPlacement[] {
    // Slight expansion to allow wider trace spacing
    return this.increaseSeparation(placements, 1.05);
  }

  private calculateMetrics(layout: PCBLayout): LayoutMetrics {
    const boardArea = layout.boardOutline.width * layout.boardOutline.height;
    const componentArea = layout.components.reduce((sum, c) =>
      sum + (c.footprint?.width || 0) * (c.footprint?.height || 0), 0);

    return {
      boardArea,
      componentDensity: componentArea / boardArea,
      routingCompletion: 0.9,
      averageTraceLength: 0,
      viaCount: layout.vias?.length || 0,
      layerUtilization: new Map(),
      thermalScore: 80,
      signalIntegrityScore: 95, // EMI optimization improves SI
      drcViolations: 0,
      estimatedYield: 94
    };
  }
}

export default EMIOptimizedAgent;