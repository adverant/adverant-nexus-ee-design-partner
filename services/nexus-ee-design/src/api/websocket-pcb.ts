/**
 * WebSocket Event Types for PCB Layout Generation Streaming
 *
 * Enables real-time progress updates from PCB layout generation to browser.
 * Uses the standard BaseWebSocketManager pattern.
 */

import { Server as SocketIOServer } from 'socket.io';
import { BaseWebSocketManager, BaseProgressEvent } from './base-ws-manager.js';

/**
 * Event types emitted during PCB layout generation.
 * Maps to phases in the MAPOS layout pipeline.
 */
export enum PCBEventType {
  // Phase lifecycle
  PHASE_START = 'phase_start',
  PHASE_COMPLETE = 'phase_complete',

  // Agent-based layout generation
  AGENT_START = 'agent_start',
  AGENT_ITERATION = 'agent_iteration',
  AGENT_SCORE = 'agent_score',
  AGENT_COMPLETE = 'agent_complete',

  // MAPOS iteration loop
  ITERATION_START = 'iteration_start',
  ITERATION_COMPLETE = 'iteration_complete',
  IMPROVEMENT_DETECTED = 'improvement_detected',
  CONVERGENCE_CHECK = 'convergence_check',

  // DRC validation
  DRC_START = 'drc_start',
  DRC_RUNNING = 'drc_running',
  DRC_COMPLETE = 'drc_complete',
  DRC_VIOLATION = 'drc_violation',

  // Component placement
  PLACEMENT_START = 'placement_start',
  COMPONENT_PLACED = 'component_placed',
  PLACEMENT_COMPLETE = 'placement_complete',

  // Routing
  ROUTING_START = 'routing_start',
  TRACE_ROUTED = 'trace_routed',
  ROUTING_PROGRESS = 'routing_progress',
  ROUTING_COMPLETE = 'routing_complete',

  // Rendering
  RENDERING_START = 'rendering_start',
  RENDERING_LAYER = 'rendering_layer',
  RENDERING_COMPLETE = 'rendering_complete',

  // Export
  EXPORT_START = 'export_start',
  GERBER_GENERATED = 'gerber_generated',
  EXPORT_COMPLETE = 'export_complete',

  // Final states
  COMPLETE = 'complete',
  ERROR = 'error',
}

/**
 * PCB layout phases for UI display
 */
export type PCBPhase =
  | 'initialization'
  | 'placement'
  | 'routing'
  | 'optimization'
  | 'drc'
  | 'rendering'
  | 'export';

/**
 * PCB-specific progress event
 */
export interface PCBProgressEvent extends BaseProgressEvent {
  /** Event type */
  type: PCBEventType | string;

  /** Current phase */
  phase?: PCBPhase;

  /** Agent information (for multi-agent layout) */
  agent_name?: string;
  agent_strategy?: 'conservative' | 'aggressive_compact' | 'thermal_optimized' | 'emi_optimized' | 'dfm_optimized';

  /** Iteration tracking */
  iteration?: number;
  max_iterations?: number;

  /** Score/quality metrics */
  score?: number;
  previous_score?: number;
  improvement_delta?: number;

  /** DRC information */
  drc_violations?: number;
  drc_errors?: number;
  drc_warnings?: number;
  violation_types?: string[];

  /** Placement info */
  components_placed?: number;
  total_components?: number;
  placement_density?: number;

  /** Routing info */
  traces_routed?: number;
  total_traces?: number;
  routing_completion?: number;
  unrouted_nets?: number;

  /** Layer info */
  layer_name?: string;
  layer_index?: number;
  total_layers?: number;

  /** Export info */
  export_format?: string;
  export_path?: string;
  gerber_files?: string[];

  /** Error details */
  error_message?: string;
  error_code?: string;
}

/**
 * Phase progress ranges for overall progress calculation
 */
export const PCB_PHASE_PROGRESS_RANGES: Record<PCBPhase, [number, number]> = {
  initialization: [0, 5],
  placement: [5, 30],
  routing: [30, 60],
  optimization: [60, 80],
  drc: [80, 90],
  rendering: [90, 95],
  export: [95, 100],
};

/**
 * Calculate overall progress from phase and phase progress
 */
export function calculatePCBProgress(phase: PCBPhase, phaseProgress: number): number {
  const [start, end] = PCB_PHASE_PROGRESS_RANGES[phase];
  const range = end - start;
  return Math.round(start + (range * phaseProgress) / 100);
}

/**
 * PCB WebSocket Manager
 */
export class PCBWebSocketManager extends BaseWebSocketManager<PCBProgressEvent> {
  constructor(io: SocketIOServer) {
    super(io, '/pcb-layout', 'pcb_layout');
  }

  /**
   * Emit agent iteration progress
   */
  emitAgentIteration(
    operationId: string,
    agentName: string,
    iteration: number,
    maxIterations: number,
    score: number,
    drcViolations: number
  ): void {
    const phaseProgress = (iteration / maxIterations) * 100;
    const overallProgress = calculatePCBProgress('optimization', phaseProgress);

    this.emitProgress(operationId, {
      type: PCBEventType.AGENT_ITERATION,
      operationId,
      timestamp: new Date().toISOString(),
      progress_percentage: overallProgress,
      current_step: `${agentName}: Iteration ${iteration}/${maxIterations} - Score: ${score.toFixed(2)}, DRC: ${drcViolations}`,
      phase: 'optimization',
      agent_name: agentName,
      iteration,
      max_iterations: maxIterations,
      score,
      drc_violations: drcViolations,
    });
  }

  /**
   * Emit DRC result
   */
  emitDRCResult(
    operationId: string,
    violations: number,
    errors: number,
    warnings: number
  ): void {
    this.emitProgress(operationId, {
      type: PCBEventType.DRC_COMPLETE,
      operationId,
      timestamp: new Date().toISOString(),
      progress_percentage: calculatePCBProgress('drc', 100),
      current_step: `DRC Complete: ${errors} errors, ${warnings} warnings`,
      phase: 'drc',
      drc_violations: violations,
      drc_errors: errors,
      drc_warnings: warnings,
    });
  }

  /**
   * Emit routing progress
   */
  emitRoutingProgress(
    operationId: string,
    tracesRouted: number,
    totalTraces: number
  ): void {
    const completion = (tracesRouted / totalTraces) * 100;
    const overallProgress = calculatePCBProgress('routing', completion);

    this.emitProgress(operationId, {
      type: PCBEventType.ROUTING_PROGRESS,
      operationId,
      timestamp: new Date().toISOString(),
      progress_percentage: overallProgress,
      current_step: `Routing traces: ${tracesRouted}/${totalTraces} (${completion.toFixed(1)}%)`,
      phase: 'routing',
      traces_routed: tracesRouted,
      total_traces: totalTraces,
      routing_completion: completion,
    });
  }
}

// Singleton instance
let pcbWsManager: PCBWebSocketManager | null = null;

/**
 * Initialize the PCB WebSocket manager.
 * Call this from index.ts after creating the SocketIO server.
 */
export function initPCBWebSocket(io: SocketIOServer): PCBWebSocketManager {
  if (!pcbWsManager) {
    pcbWsManager = new PCBWebSocketManager(io);
  }
  return pcbWsManager;
}

/**
 * Get the PCB WebSocket manager instance.
 */
export function getPCBWsManager(): PCBWebSocketManager {
  if (!pcbWsManager) {
    throw new Error('PCBWebSocketManager not initialized. Call initPCBWebSocket first.');
  }
  return pcbWsManager;
}
