/**
 * HIL (Hardware-in-the-Loop) WebSocket Manager
 *
 * Provides real-time streaming for HIL testing operations including:
 * - Instrument discovery and connection status
 * - Test run progress and measurements
 * - Live waveform and logic trace data
 * - Motor state and power state changes
 * - Fault detection and emergency stops
 */

import { Server as SocketIOServer, Socket } from 'socket.io';
import {
  BaseWebSocketManager,
  BaseProgressEvent,
  parsePythonProgressLine,
} from './base-ws-manager.js';
import { log, Logger } from '../utils/logger.js';
import type {
  HILEventType,
  HILPhase,
  HILInstrument,
  HILInstrumentStatus,
  HILTestRun,
  HILTestRunStatus,
  HILMeasurement,
  HILWaveformChunk,
  HILLiveMeasurement,
  HILMeasurementType,
} from '../types/hil-types.js';

// ============================================================================
// Event Types
// ============================================================================

/**
 * HIL progress event interface
 */
export interface HILProgressEvent extends BaseProgressEvent {
  /** HIL-specific event type */
  hilEventType?: string;

  /** Current phase of HIL operation */
  hilPhase?: HILPhase;

  /** Instrument ID (for instrument events) */
  instrumentId?: string;

  /** Instrument details (for discovery/connection) */
  instrument?: Partial<HILInstrument>;

  /** Instrument status (for status changes) */
  instrumentStatus?: HILInstrumentStatus;

  /** Test run ID (for test events) */
  testRunId?: string;

  /** Test run details (for status updates) */
  testRun?: Partial<HILTestRun>;

  /** Current step index */
  stepIndex?: number;

  /** Total steps in sequence */
  totalSteps?: number;

  /** Step ID (for step events) */
  stepId?: string;

  /** Step name (for step events) */
  stepName?: string;

  /** Measurement data (for measurement events) */
  measurement?: HILLiveMeasurement;

  /** Waveform chunk (for streaming) */
  waveformChunk?: HILWaveformChunk;

  /** Logic trace chunk (for streaming) */
  logicTraceChunk?: {
    captureId: string;
    channels: string[];
    startIndex: number;
    data: number[][];
    isLast?: boolean;
  };

  /** Capture ID (for capture events) */
  captureId?: string;

  /** Motor state (for motor events) */
  motorState?: {
    speed: number;
    torque: number;
    position: number;
    direction: 'forward' | 'reverse' | 'stopped';
    faultCode?: number;
  };

  /** Power state (for power events) */
  powerState?: {
    channel: string;
    voltage: number;
    current: number;
    enabled: boolean;
    mode: 'cv' | 'cc' | 'off';
  };

  /** Fault details (for fault events) */
  fault?: {
    code: string;
    message: string;
    severity: 'warning' | 'error' | 'critical';
    instrumentId?: string;
    timestamp: string;
  };

  /** Error details (for error events) */
  error?: {
    code: string;
    message: string;
    details?: Record<string, unknown>;
  };
}

/**
 * Subscription options for HIL events
 */
interface HILSubscriptionOptions {
  /** Subscribe to instrument events */
  instruments?: boolean;
  /** Subscribe to test run events */
  testRuns?: boolean;
  /** Subscribe to waveform streaming */
  waveforms?: boolean;
  /** Subscribe to live measurements */
  measurements?: boolean;
  /** Subscribe to power/motor control events */
  control?: boolean;
  /** Subscribe to fault events */
  faults?: boolean;
}

// ============================================================================
// HIL WebSocket Manager Implementation
// ============================================================================

const wsLogger: Logger = log.child({ service: 'hil-websocket' });

/**
 * HIL WebSocket Manager
 *
 * Extends BaseWebSocketManager with HIL-specific functionality:
 * - Instrument status broadcasting
 * - Multi-channel waveform streaming
 * - Live measurement updates
 * - Motor and power state tracking
 * - Fault/emergency handling
 */
export class HILWebSocketManager extends BaseWebSocketManager<HILProgressEvent> {
  /** Track active waveform streams by capture ID */
  private activeWaveformStreams: Map<string, {
    operationId: string;
    channels: string[];
    samplesStreamed: number;
    startedAt: Date;
  }> = new Map();

  /** Track connected instruments */
  private connectedInstruments: Map<string, HILInstrument> = new Map();

  /** Track active test runs */
  private activeTestRuns: Map<string, {
    testRunId: string;
    operationId: string;
    status: HILTestRunStatus;
    startedAt: Date;
  }> = new Map();

  constructor(io: SocketIOServer) {
    super(io, '/hil', 'hil');
  }

  /**
   * Override setup to add HIL-specific event handlers
   */
  protected setupNamespace(): void {
    super.setupNamespace();

    this.namespace.on('connection', (socket: Socket) => {
      wsLogger.info('HIL client connected', { socketId: socket.id });

      // Subscribe to project instruments
      socket.on('subscribe:instruments', (projectId: string) => {
        socket.join(`project:${projectId}:instruments`);
        wsLogger.debug('Client subscribed to project instruments', {
          socketId: socket.id,
          projectId,
        });

        // Send current instrument states
        const projectInstruments = Array.from(this.connectedInstruments.values())
          .filter(i => i.projectId === projectId);
        if (projectInstruments.length > 0) {
          socket.emit('instruments:list', projectInstruments);
        }
      });

      // Subscribe to test run by ID
      socket.on('subscribe:test-run', (testRunId: string) => {
        socket.join(`test-run:${testRunId}`);
        wsLogger.debug('Client subscribed to test run', {
          socketId: socket.id,
          testRunId,
        });

        // Send current test run status
        const testRun = this.activeTestRuns.get(testRunId);
        if (testRun) {
          socket.emit('test-run:status', {
            testRunId: testRun.testRunId,
            status: testRun.status,
            operationId: testRun.operationId,
          });
        }
      });

      // Subscribe to waveform streaming
      socket.on('subscribe:waveform', (captureId: string) => {
        socket.join(`waveform:${captureId}`);
        wsLogger.debug('Client subscribed to waveform stream', {
          socketId: socket.id,
          captureId,
        });
      });

      // Subscribe to live measurements
      socket.on('subscribe:measurements', (testRunId: string) => {
        socket.join(`measurements:${testRunId}`);
        wsLogger.debug('Client subscribed to live measurements', {
          socketId: socket.id,
          testRunId,
        });
      });

      // Subscribe to control events (motor/power)
      socket.on('subscribe:control', (projectId: string) => {
        socket.join(`control:${projectId}`);
        wsLogger.debug('Client subscribed to control events', {
          socketId: socket.id,
          projectId,
        });
      });

      // Subscribe to faults
      socket.on('subscribe:faults', (projectId: string) => {
        socket.join(`faults:${projectId}`);
        wsLogger.debug('Client subscribed to fault events', {
          socketId: socket.id,
          projectId,
        });
      });

      // Unsubscribe handlers
      socket.on('unsubscribe:instruments', (projectId: string) => {
        socket.leave(`project:${projectId}:instruments`);
      });

      socket.on('unsubscribe:test-run', (testRunId: string) => {
        socket.leave(`test-run:${testRunId}`);
      });

      socket.on('unsubscribe:waveform', (captureId: string) => {
        socket.leave(`waveform:${captureId}`);
      });

      socket.on('unsubscribe:measurements', (testRunId: string) => {
        socket.leave(`measurements:${testRunId}`);
      });

      // Request current instrument list
      socket.on('get:instruments', (projectId: string) => {
        const instruments = Array.from(this.connectedInstruments.values())
          .filter(i => i.projectId === projectId);
        socket.emit('instruments:list', instruments);
      });

      // Request active test runs
      socket.on('get:active-runs', (projectId: string) => {
        const runs = Array.from(this.activeTestRuns.values())
          .filter(r => {
            const operation = this.getOperation(r.operationId);
            return operation?.projectId === projectId;
          });
        socket.emit('test-runs:active', runs);
      });

      // Emergency stop handler
      socket.on('emergency:stop', async (projectId: string) => {
        wsLogger.warn('Emergency stop requested', { socketId: socket.id, projectId });
        this.emitEmergencyStop(projectId, 'User requested emergency stop');
      });
    });
  }

  // ===========================================================================
  // Instrument Events
  // ===========================================================================

  /**
   * Emit instrument discovered event
   */
  emitInstrumentDiscovered(
    operationId: string,
    projectId: string,
    instrument: HILInstrument
  ): void {
    this.connectedInstruments.set(instrument.id, instrument);

    const event: HILProgressEvent = {
      type: 'instrument_discovered',
      hilEventType: 'INSTRUMENT_DISCOVERED',
      operationId,
      timestamp: new Date().toISOString(),
      progress_percentage: 0,
      current_step: `Discovered ${instrument.manufacturer} ${instrument.model}`,
      instrumentId: instrument.id,
      instrument,
      hilPhase: 'discovery',
    };

    this.emitProgress(operationId, event);
    this.namespace.to(`project:${projectId}:instruments`).emit('instrument:discovered', instrument);

    wsLogger.info('Instrument discovered', {
      instrumentId: instrument.id,
      type: instrument.instrumentType,
      manufacturer: instrument.manufacturer,
      model: instrument.model,
    });
  }

  /**
   * Emit instrument connected event
   */
  emitInstrumentConnected(
    operationId: string,
    projectId: string,
    instrument: HILInstrument
  ): void {
    instrument.status = 'connected';
    this.connectedInstruments.set(instrument.id, instrument);

    const event: HILProgressEvent = {
      type: 'instrument_connected',
      hilEventType: 'INSTRUMENT_CONNECTED',
      operationId,
      timestamp: new Date().toISOString(),
      progress_percentage: 0,
      current_step: `Connected to ${instrument.name}`,
      instrumentId: instrument.id,
      instrument,
      instrumentStatus: 'connected',
    };

    this.emitProgress(operationId, event);
    this.namespace.to(`project:${projectId}:instruments`).emit('instrument:connected', instrument);

    wsLogger.info('Instrument connected', {
      instrumentId: instrument.id,
      name: instrument.name,
    });
  }

  /**
   * Emit instrument disconnected event
   */
  emitInstrumentDisconnected(
    operationId: string,
    projectId: string,
    instrumentId: string,
    reason?: string
  ): void {
    const instrument = this.connectedInstruments.get(instrumentId);
    if (instrument) {
      instrument.status = 'disconnected';
    }

    const event: HILProgressEvent = {
      type: 'instrument_disconnected',
      hilEventType: 'INSTRUMENT_DISCONNECTED',
      operationId,
      timestamp: new Date().toISOString(),
      progress_percentage: 0,
      current_step: reason || 'Instrument disconnected',
      instrumentId,
      instrumentStatus: 'disconnected',
    };

    this.emitProgress(operationId, event);
    this.namespace.to(`project:${projectId}:instruments`).emit('instrument:disconnected', {
      instrumentId,
      reason,
    });

    wsLogger.info('Instrument disconnected', { instrumentId, reason });
  }

  /**
   * Emit instrument status change
   */
  emitInstrumentStatusChanged(
    projectId: string,
    instrumentId: string,
    status: HILInstrumentStatus,
    error?: string
  ): void {
    const instrument = this.connectedInstruments.get(instrumentId);
    if (instrument) {
      instrument.status = status;
      if (error) {
        instrument.lastError = error;
      }
    }

    this.namespace.to(`project:${projectId}:instruments`).emit('instrument:status', {
      instrumentId,
      status,
      error,
      timestamp: new Date().toISOString(),
    });

    wsLogger.debug('Instrument status changed', { instrumentId, status, error });
  }

  // ===========================================================================
  // Test Run Events
  // ===========================================================================

  /**
   * Start tracking a test run
   */
  startTestRun(
    operationId: string,
    testRunId: string,
    projectId: string
  ): void {
    this.activeTestRuns.set(testRunId, {
      testRunId,
      operationId,
      status: 'running',
      startedAt: new Date(),
    });

    const event: HILProgressEvent = {
      type: 'test_run_started',
      hilEventType: 'TEST_RUN_STARTED',
      operationId,
      timestamp: new Date().toISOString(),
      progress_percentage: 0,
      current_step: 'Test run started',
      testRunId,
      hilPhase: 'setup',
    };

    this.emitProgress(operationId, event);
    this.namespace.to(`test-run:${testRunId}`).emit('test-run:started', { testRunId, operationId });

    wsLogger.info('Test run started', { testRunId, operationId, projectId });
  }

  /**
   * Emit test step started
   */
  emitTestStepStarted(
    operationId: string,
    testRunId: string,
    stepIndex: number,
    totalSteps: number,
    stepId: string,
    stepName: string
  ): void {
    const progress = (stepIndex / totalSteps) * 100;

    const event: HILProgressEvent = {
      type: 'test_step_started',
      hilEventType: 'TEST_STEP_STARTED',
      operationId,
      timestamp: new Date().toISOString(),
      progress_percentage: progress,
      current_step: stepName,
      testRunId,
      stepIndex,
      totalSteps,
      stepId,
      stepName,
      hilPhase: 'measurement',
    };

    this.emitProgress(operationId, event);
    this.namespace.to(`test-run:${testRunId}`).emit('test-run:step-started', {
      stepIndex,
      totalSteps,
      stepId,
      stepName,
      progress,
    });

    wsLogger.debug('Test step started', { testRunId, stepIndex, totalSteps, stepName });
  }

  /**
   * Emit test step completed
   */
  emitTestStepCompleted(
    operationId: string,
    testRunId: string,
    stepIndex: number,
    totalSteps: number,
    stepId: string,
    stepName: string,
    passed: boolean,
    measurementCount: number
  ): void {
    const progress = ((stepIndex + 1) / totalSteps) * 100;

    const event: HILProgressEvent = {
      type: 'test_step_completed',
      hilEventType: 'TEST_STEP_COMPLETED',
      operationId,
      timestamp: new Date().toISOString(),
      progress_percentage: progress,
      current_step: `Step "${stepName}" ${passed ? 'passed' : 'failed'}`,
      testRunId,
      stepIndex,
      totalSteps,
      stepId,
      stepName,
      data: { passed, measurementCount },
    };

    this.emitProgress(operationId, event);
    this.namespace.to(`test-run:${testRunId}`).emit('test-run:step-completed', {
      stepIndex,
      totalSteps,
      stepId,
      stepName,
      passed,
      measurementCount,
      progress,
    });

    wsLogger.debug('Test step completed', { testRunId, stepId, passed, measurementCount });
  }

  /**
   * Emit test run progress
   */
  emitTestRunProgress(
    operationId: string,
    testRunId: string,
    progress: number,
    currentStep: string,
    phase?: HILPhase
  ): void {
    const event: HILProgressEvent = {
      type: 'test_run_progress',
      hilEventType: 'TEST_RUN_PROGRESS',
      operationId,
      timestamp: new Date().toISOString(),
      progress_percentage: progress,
      current_step: currentStep,
      testRunId,
      hilPhase: phase,
    };

    this.emitProgress(operationId, event);
    this.namespace.to(`test-run:${testRunId}`).emit('test-run:progress', {
      testRunId,
      progress,
      currentStep,
      phase,
    });
  }

  /**
   * Emit test run completed
   */
  emitTestRunCompleted(
    operationId: string,
    testRunId: string,
    result: 'pass' | 'fail' | 'partial' | 'inconclusive',
    summary: {
      totalMeasurements: number;
      passedMeasurements: number;
      failedMeasurements: number;
      durationMs: number;
    }
  ): void {
    const testRun = this.activeTestRuns.get(testRunId);
    if (testRun) {
      testRun.status = 'completed';
    }

    const event: HILProgressEvent = {
      type: 'test_run_completed',
      hilEventType: 'TEST_RUN_COMPLETED',
      operationId,
      timestamp: new Date().toISOString(),
      progress_percentage: 100,
      current_step: `Test run ${result}`,
      testRunId,
      hilPhase: 'teardown',
      data: { result, summary },
    };

    this.emitProgress(operationId, event);
    this.namespace.to(`test-run:${testRunId}`).emit('test-run:completed', {
      testRunId,
      result,
      summary,
    });

    // Clean up after delay
    setTimeout(() => {
      this.activeTestRuns.delete(testRunId);
    }, 60000);

    wsLogger.info('Test run completed', { testRunId, result, summary });
  }

  /**
   * Emit test run failed
   */
  emitTestRunFailed(
    operationId: string,
    testRunId: string,
    error: string,
    stepId?: string
  ): void {
    const testRun = this.activeTestRuns.get(testRunId);
    if (testRun) {
      testRun.status = 'failed';
    }

    const event: HILProgressEvent = {
      type: 'test_run_failed',
      hilEventType: 'TEST_RUN_FAILED',
      operationId,
      timestamp: new Date().toISOString(),
      progress_percentage: this.getOperation(operationId)?.lastEvent?.progress_percentage || 0,
      current_step: `Test failed: ${error}`,
      testRunId,
      stepId,
      error: {
        code: 'TEST_FAILED',
        message: error,
      },
    };

    this.emitProgress(operationId, event);
    this.namespace.to(`test-run:${testRunId}`).emit('test-run:failed', {
      testRunId,
      error,
      stepId,
    });

    setTimeout(() => {
      this.activeTestRuns.delete(testRunId);
    }, 60000);

    wsLogger.error('Test run failed', new Error(error), { testRunId, stepId });
  }

  // ===========================================================================
  // Measurement & Capture Events
  // ===========================================================================

  /**
   * Emit live measurement
   */
  emitLiveMeasurement(
    operationId: string,
    testRunId: string,
    measurement: HILLiveMeasurement
  ): void {
    const event: HILProgressEvent = {
      type: 'live_measurement',
      hilEventType: 'LIVE_MEASUREMENT',
      operationId,
      timestamp: new Date().toISOString(),
      progress_percentage: 0,
      current_step: `${measurement.type}: ${measurement.value}${measurement.unit}`,
      testRunId,
      measurement,
    };

    // Emit to operation subscribers
    this.emitProgress(operationId, event);

    // Also emit to measurement subscribers
    this.namespace.to(`measurements:${testRunId}`).emit('measurement:live', {
      testRunId,
      measurement,
      timestamp: new Date().toISOString(),
    });
  }

  /**
   * Emit batch of measurements
   */
  emitMeasurementBatch(
    operationId: string,
    testRunId: string,
    measurements: HILLiveMeasurement[]
  ): void {
    this.namespace.to(`measurements:${testRunId}`).emit('measurement:batch', {
      testRunId,
      measurements,
      count: measurements.length,
      timestamp: new Date().toISOString(),
    });

    wsLogger.debug('Emitted measurement batch', {
      testRunId,
      count: measurements.length,
    });
  }

  /**
   * Start waveform streaming
   */
  startWaveformStream(
    operationId: string,
    captureId: string,
    channels: string[]
  ): void {
    this.activeWaveformStreams.set(captureId, {
      operationId,
      channels,
      samplesStreamed: 0,
      startedAt: new Date(),
    });

    const event: HILProgressEvent = {
      type: 'capture_started',
      hilEventType: 'CAPTURE_STARTED',
      operationId,
      timestamp: new Date().toISOString(),
      progress_percentage: 0,
      current_step: `Starting waveform capture (${channels.length} channels)`,
      captureId,
      hilPhase: 'capture',
      data: { channels },
    };

    this.emitProgress(operationId, event);

    wsLogger.info('Waveform stream started', { captureId, channels });
  }

  /**
   * Emit waveform chunk
   */
  emitWaveformChunk(
    captureId: string,
    chunk: HILWaveformChunk
  ): void {
    const stream = this.activeWaveformStreams.get(captureId);
    if (!stream) {
      wsLogger.warn('Emitting to unknown waveform stream', { captureId });
      return;
    }

    stream.samplesStreamed += chunk.sampleCount;

    // Emit to waveform subscribers
    this.namespace.to(`waveform:${captureId}`).emit('waveform:chunk', {
      captureId,
      chunk,
      totalSamples: stream.samplesStreamed,
    });

    // Also emit to operation subscribers
    if (stream.operationId) {
      const event: HILProgressEvent = {
        type: 'waveform_chunk',
        hilEventType: 'WAVEFORM_CHUNK',
        operationId: stream.operationId,
        timestamp: new Date().toISOString(),
        progress_percentage: 0,
        current_step: `Streaming waveform data (${stream.samplesStreamed} samples)`,
        captureId,
        waveformChunk: chunk,
      };

      this.emitProgress(stream.operationId, event);
    }

    if (chunk.isLast) {
      wsLogger.info('Waveform stream complete', {
        captureId,
        totalSamples: stream.samplesStreamed,
      });
    }
  }

  /**
   * Complete waveform capture
   */
  completeWaveformCapture(
    operationId: string,
    captureId: string,
    analysisResults?: Record<string, unknown>
  ): void {
    const stream = this.activeWaveformStreams.get(captureId);

    const event: HILProgressEvent = {
      type: 'capture_complete',
      hilEventType: 'CAPTURE_COMPLETE',
      operationId,
      timestamp: new Date().toISOString(),
      progress_percentage: 100,
      current_step: 'Waveform capture complete',
      captureId,
      hilPhase: 'analysis',
      data: {
        totalSamples: stream?.samplesStreamed || 0,
        analysisResults,
      },
    };

    this.emitProgress(operationId, event);
    this.namespace.to(`waveform:${captureId}`).emit('waveform:complete', {
      captureId,
      totalSamples: stream?.samplesStreamed || 0,
      analysisResults,
    });

    // Clean up
    this.activeWaveformStreams.delete(captureId);

    wsLogger.info('Waveform capture complete', { captureId });
  }

  /**
   * Emit logic trace chunk
   */
  emitLogicTraceChunk(
    operationId: string,
    captureId: string,
    chunk: {
      channels: string[];
      startIndex: number;
      data: number[][];
      isLast?: boolean;
    }
  ): void {
    const event: HILProgressEvent = {
      type: 'logic_trace_chunk',
      hilEventType: 'LOGIC_TRACE_CHUNK',
      operationId,
      timestamp: new Date().toISOString(),
      progress_percentage: 0,
      current_step: 'Streaming logic trace data',
      captureId,
      logicTraceChunk: { captureId, ...chunk },
    };

    this.emitProgress(operationId, event);
    this.namespace.to(`waveform:${captureId}`).emit('logic-trace:chunk', {
      captureId,
      chunk,
    });
  }

  // ===========================================================================
  // Control Events (Motor/Power)
  // ===========================================================================

  /**
   * Emit motor state change
   */
  emitMotorStateChanged(
    operationId: string,
    projectId: string,
    motorState: {
      speed: number;
      torque: number;
      position: number;
      direction: 'forward' | 'reverse' | 'stopped';
      faultCode?: number;
    }
  ): void {
    const event: HILProgressEvent = {
      type: 'motor_state_changed',
      hilEventType: 'MOTOR_STATE_CHANGED',
      operationId,
      timestamp: new Date().toISOString(),
      progress_percentage: 0,
      current_step: `Motor: ${motorState.speed} RPM, ${motorState.direction}`,
      motorState,
    };

    this.emitProgress(operationId, event);
    this.namespace.to(`control:${projectId}`).emit('motor:state', {
      motorState,
      timestamp: new Date().toISOString(),
    });
  }

  /**
   * Emit power state change
   */
  emitPowerStateChanged(
    operationId: string,
    projectId: string,
    powerState: {
      channel: string;
      voltage: number;
      current: number;
      enabled: boolean;
      mode: 'cv' | 'cc' | 'off';
    }
  ): void {
    const event: HILProgressEvent = {
      type: 'power_state_changed',
      hilEventType: 'POWER_STATE_CHANGED',
      operationId,
      timestamp: new Date().toISOString(),
      progress_percentage: 0,
      current_step: `Power ${powerState.channel}: ${powerState.voltage}V / ${powerState.current}A`,
      powerState,
    };

    this.emitProgress(operationId, event);
    this.namespace.to(`control:${projectId}`).emit('power:state', {
      powerState,
      timestamp: new Date().toISOString(),
    });
  }

  // ===========================================================================
  // Fault & Emergency Events
  // ===========================================================================

  /**
   * Emit fault detected
   */
  emitFaultDetected(
    operationId: string,
    projectId: string,
    fault: {
      code: string;
      message: string;
      severity: 'warning' | 'error' | 'critical';
      instrumentId?: string;
    }
  ): void {
    const event: HILProgressEvent = {
      type: 'fault_detected',
      hilEventType: 'FAULT_DETECTED',
      operationId,
      timestamp: new Date().toISOString(),
      progress_percentage: 0,
      current_step: `Fault: ${fault.message}`,
      fault: {
        ...fault,
        timestamp: new Date().toISOString(),
      },
    };

    this.emitProgress(operationId, event);
    this.namespace.to(`faults:${projectId}`).emit('fault:detected', {
      fault: event.fault,
      operationId,
    });

    wsLogger.warn('Fault detected', { fault, operationId, projectId });
  }

  /**
   * Emit emergency stop
   */
  emitEmergencyStop(
    projectId: string,
    reason: string
  ): void {
    const timestamp = new Date().toISOString();

    // Notify all control subscribers
    this.namespace.to(`control:${projectId}`).emit('emergency:stop', {
      reason,
      timestamp,
    });

    // Notify all fault subscribers
    this.namespace.to(`faults:${projectId}`).emit('emergency:stop', {
      reason,
      timestamp,
    });

    // Abort all active test runs for this project
    for (const [testRunId, testRun] of this.activeTestRuns.entries()) {
      const operation = this.getOperation(testRun.operationId);
      if (operation?.projectId === projectId && testRun.status === 'running') {
        this.emitTestRunFailed(testRun.operationId, testRunId, `Emergency stop: ${reason}`);
      }
    }

    wsLogger.error('Emergency stop triggered', new Error(reason), { projectId });
  }

  // ===========================================================================
  // Utility Methods
  // ===========================================================================

  /**
   * Get connected instruments for a project
   */
  getConnectedInstruments(projectId: string): HILInstrument[] {
    return Array.from(this.connectedInstruments.values())
      .filter(i => i.projectId === projectId && i.status === 'connected');
  }

  /**
   * Get active test runs for a project
   */
  getActiveTestRuns(projectId: string): Array<{
    testRunId: string;
    operationId: string;
    status: HILTestRunStatus;
  }> {
    return Array.from(this.activeTestRuns.values())
      .filter(r => {
        const operation = this.getOperation(r.operationId);
        return operation?.projectId === projectId;
      });
  }

  /**
   * Get HIL-specific statistics
   */
  getHILStats(): {
    activeOperations: number;
    connectedClients: number;
    connectedInstruments: number;
    activeTestRuns: number;
    activeWaveformStreams: number;
  } {
    const baseStats = this.getStats();
    return {
      ...baseStats,
      connectedInstruments: this.connectedInstruments.size,
      activeTestRuns: this.activeTestRuns.size,
      activeWaveformStreams: this.activeWaveformStreams.size,
    };
  }
}

// ============================================================================
// Singleton Export
// ============================================================================

let hilWsManager: HILWebSocketManager | null = null;

/**
 * Initialize the HIL WebSocket manager.
 * Call this once during server startup.
 */
export function initHILWebSocket(io: SocketIOServer): HILWebSocketManager {
  if (!hilWsManager) {
    hilWsManager = new HILWebSocketManager(io);
    wsLogger.info('HIL WebSocket manager initialized');
  }
  return hilWsManager;
}

/**
 * Get the HIL WebSocket manager instance.
 * Throws if not initialized.
 */
export function getHILWebSocketManager(): HILWebSocketManager {
  if (!hilWsManager) {
    throw new Error('HIL WebSocket manager not initialized. Call initHILWebSocket first.');
  }
  return hilWsManager;
}

/**
 * Parse HIL-specific Python progress lines.
 */
export function parseHILProgressLine(
  line: string,
  operationId: string,
  manager: HILWebSocketManager
): boolean {
  return parsePythonProgressLine(line, operationId, manager);
}

export default HILWebSocketManager;
