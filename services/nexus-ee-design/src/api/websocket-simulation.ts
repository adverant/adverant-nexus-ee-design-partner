/**
 * WebSocket Event Types for Simulation Streaming
 *
 * Enables real-time progress updates from all simulation types:
 * - SPICE (circuit simulation)
 * - Thermal (heat analysis)
 * - Signal Integrity (SI/PI)
 * - RF/EMC (electromagnetic)
 */

import { Server as SocketIOServer } from 'socket.io';
import { BaseWebSocketManager, BaseProgressEvent } from './base-ws-manager.js';

/**
 * Event types for all simulation workflows
 */
export enum SimulationEventType {
  // Phase lifecycle
  PHASE_START = 'phase_start',
  PHASE_COMPLETE = 'phase_complete',

  // Setup phase
  SETUP_START = 'setup_start',
  NETLIST_GENERATING = 'netlist_generating',
  NETLIST_COMPLETE = 'netlist_complete',
  GEOMETRY_LOADING = 'geometry_loading',
  GEOMETRY_LOADED = 'geometry_loaded',

  // Solver execution
  SOLVER_START = 'solver_start',
  SOLVER_RUNNING = 'solver_running',
  SOLVER_PROGRESS = 'solver_progress',
  SOLVER_ITERATION = 'solver_iteration',
  SOLVER_CONVERGING = 'solver_converging',
  SOLVER_COMPLETE = 'solver_complete',

  // Post-processing
  WAVEFORM_EXTRACTING = 'waveform_extracting',
  WAVEFORM_COMPLETE = 'waveform_complete',
  METRICS_CALCULATING = 'metrics_calculating',
  METRICS_COMPLETE = 'metrics_complete',
  REPORT_GENERATING = 'report_generating',
  REPORT_COMPLETE = 'report_complete',

  // SPICE-specific
  SPICE_DC_OPERATING = 'spice_dc_operating',
  SPICE_TRANSIENT = 'spice_transient',
  SPICE_AC_ANALYSIS = 'spice_ac_analysis',

  // Thermal-specific
  THERMAL_MESH_GENERATING = 'thermal_mesh_generating',
  THERMAL_BOUNDARY_SETTING = 'thermal_boundary_setting',
  THERMAL_SOLVING = 'thermal_solving',

  // SI/PI-specific
  SI_STACKUP_ANALYZING = 'si_stackup_analyzing',
  SI_IMPEDANCE_CALCULATING = 'si_impedance_calculating',
  SI_CROSSTALK_ANALYZING = 'si_crosstalk_analyzing',

  // RF/EMC-specific
  RF_PORT_CONFIGURING = 'rf_port_configuring',
  RF_FIELD_SOLVING = 'rf_field_solving',
  RF_S_PARAMETERS = 'rf_s_parameters',

  // Final states
  COMPLETE = 'complete',
  ERROR = 'error',
}

/**
 * Simulation types
 */
export type SimulationType = 'spice' | 'thermal' | 'signal_integrity' | 'rf_emc';

/**
 * Simulation phases
 */
export type SimulationPhase = 'setup' | 'preprocessing' | 'solving' | 'postprocessing' | 'reporting';

/**
 * Simulation-specific progress event
 */
export interface SimulationProgressEvent extends BaseProgressEvent {
  /** Event type */
  type: SimulationEventType | string;

  /** Simulation type */
  simulation_type?: SimulationType;

  /** Current phase */
  phase?: SimulationPhase;

  /** Solver information */
  solver_name?: string;  // e.g., 'ngspice', 'openems', 'elmer'

  /** Iteration/time tracking */
  time_step?: number;
  total_steps?: number;
  current_time?: number;
  end_time?: number;
  iteration?: number;
  max_iterations?: number;

  /** Convergence info */
  convergence?: number;
  target_convergence?: number;
  is_converging?: boolean;

  /** SPICE-specific */
  analysis_type?: 'dc' | 'ac' | 'transient' | 'noise';
  frequency?: number;
  voltage_node?: string;
  current_value?: number;

  /** Thermal-specific */
  max_temperature?: number;
  min_temperature?: number;
  mesh_elements?: number;

  /** SI/PI-specific */
  impedance?: number;
  target_impedance?: number;
  crosstalk_db?: number;

  /** RF-specific */
  s11?: number;
  s21?: number;
  frequency_ghz?: number;

  /** Metrics/results */
  metrics?: Record<string, number>;
  waveform_count?: number;

  /** Error details */
  error_message?: string;
  error_code?: string;
}

/**
 * Phase progress ranges
 */
export const SIMULATION_PHASE_PROGRESS_RANGES: Record<SimulationPhase, [number, number]> = {
  setup: [0, 10],
  preprocessing: [10, 20],
  solving: [20, 80],
  postprocessing: [80, 95],
  reporting: [95, 100],
};

/**
 * Calculate overall progress from phase and phase progress
 */
export function calculateSimulationProgress(phase: SimulationPhase, phaseProgress: number): number {
  const [start, end] = SIMULATION_PHASE_PROGRESS_RANGES[phase];
  const range = end - start;
  return Math.round(start + (range * phaseProgress) / 100);
}

/**
 * Simulation WebSocket Manager
 */
export class SimulationWebSocketManager extends BaseWebSocketManager<SimulationProgressEvent> {
  constructor(io: SocketIOServer) {
    super(io, '/simulation', 'simulation');
  }

  /**
   * Emit solver progress
   */
  emitSolverProgress(
    operationId: string,
    simulationType: SimulationType,
    solverName: string,
    currentStep: number,
    totalSteps: number,
    message?: string
  ): void {
    const phaseProgress = (currentStep / totalSteps) * 100;
    const overallProgress = calculateSimulationProgress('solving', phaseProgress);

    this.emitProgress(operationId, {
      type: SimulationEventType.SOLVER_PROGRESS,
      operationId,
      timestamp: new Date().toISOString(),
      progress_percentage: overallProgress,
      current_step: message || `${solverName}: Step ${currentStep}/${totalSteps}`,
      phase: 'solving',
      simulation_type: simulationType,
      solver_name: solverName,
      time_step: currentStep,
      total_steps: totalSteps,
    });
  }

  /**
   * Emit SPICE transient progress
   */
  emitSpiceTransient(
    operationId: string,
    currentTime: number,
    endTime: number
  ): void {
    const phaseProgress = (currentTime / endTime) * 100;
    const overallProgress = calculateSimulationProgress('solving', phaseProgress);

    this.emitProgress(operationId, {
      type: SimulationEventType.SPICE_TRANSIENT,
      operationId,
      timestamp: new Date().toISOString(),
      progress_percentage: overallProgress,
      current_step: `Transient analysis: ${(currentTime * 1e6).toFixed(2)}us / ${(endTime * 1e6).toFixed(2)}us`,
      phase: 'solving',
      simulation_type: 'spice',
      solver_name: 'ngspice',
      analysis_type: 'transient',
      current_time: currentTime,
      end_time: endTime,
    });
  }

  /**
   * Emit thermal solving progress
   */
  emitThermalProgress(
    operationId: string,
    iteration: number,
    maxIterations: number,
    maxTemp: number,
    convergence: number
  ): void {
    const phaseProgress = (iteration / maxIterations) * 100;
    const overallProgress = calculateSimulationProgress('solving', phaseProgress);

    this.emitProgress(operationId, {
      type: SimulationEventType.THERMAL_SOLVING,
      operationId,
      timestamp: new Date().toISOString(),
      progress_percentage: overallProgress,
      current_step: `Thermal iteration ${iteration}/${maxIterations}: Max temp ${maxTemp.toFixed(1)}°C`,
      phase: 'solving',
      simulation_type: 'thermal',
      iteration,
      max_iterations: maxIterations,
      max_temperature: maxTemp,
      convergence,
    });
  }

  /**
   * Emit metrics/results
   */
  emitMetricsComplete(
    operationId: string,
    simulationType: SimulationType,
    metrics: Record<string, number>
  ): void {
    this.emitProgress(operationId, {
      type: SimulationEventType.METRICS_COMPLETE,
      operationId,
      timestamp: new Date().toISOString(),
      progress_percentage: calculateSimulationProgress('postprocessing', 100),
      current_step: `Analysis complete: ${Object.keys(metrics).length} metrics calculated`,
      phase: 'postprocessing',
      simulation_type: simulationType,
      metrics,
    });
  }
}

// Singleton instance
let simulationWsManager: SimulationWebSocketManager | null = null;

/**
 * Initialize the Simulation WebSocket manager.
 * Call this from index.ts after creating the SocketIO server.
 */
export function initSimulationWebSocket(io: SocketIOServer): SimulationWebSocketManager {
  if (!simulationWsManager) {
    simulationWsManager = new SimulationWebSocketManager(io);
  }
  return simulationWsManager;
}

/**
 * Get the Simulation WebSocket manager instance.
 */
export function getSimulationWsManager(): SimulationWebSocketManager {
  if (!simulationWsManager) {
    throw new Error('SimulationWebSocketManager not initialized. Call initSimulationWebSocket first.');
  }
  return simulationWsManager;
}
