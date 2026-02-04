/**
 * HIL (Hardware-in-the-Loop) Testing - State Management Store
 *
 * Centralized state management for HIL testing using Zustand with immer.
 * Handles instruments, test sequences, test runs, measurements, and captured data.
 */

import { create } from "zustand";
import { immer } from "zustand/middleware/immer";

// ============================================================================
// Types - Mirroring backend types
// ============================================================================

export type HILInstrumentType =
  | "logic_analyzer"
  | "oscilloscope"
  | "power_supply"
  | "motor_emulator"
  | "daq"
  | "can_analyzer"
  | "function_gen"
  | "thermal_camera"
  | "electronic_load";

export type HILConnectionType =
  | "usb"
  | "ethernet"
  | "gpib"
  | "serial"
  | "grpc"
  | "modbus_tcp"
  | "modbus_rtu";

export type HILInstrumentStatus =
  | "connected"
  | "disconnected"
  | "error"
  | "busy"
  | "initializing";

export type HILTestType =
  | "foc_startup"
  | "foc_steady_state"
  | "foc_transient"
  | "foc_speed_reversal"
  | "pwm_analysis"
  | "phase_current"
  | "hall_sensor"
  | "thermal_profile"
  | "overcurrent_protection"
  | "efficiency_sweep"
  | "load_step"
  | "no_load"
  | "locked_rotor"
  | "custom";

export type HILTestRunStatus =
  | "pending"
  | "queued"
  | "running"
  | "completed"
  | "failed"
  | "aborted"
  | "timeout";

export type HILTestResult = "pass" | "fail" | "partial" | "inconclusive";

export type HILCaptureType =
  | "waveform"
  | "logic_trace"
  | "spectrum"
  | "thermal_image"
  | "measurement"
  | "protocol_decode"
  | "can_log";

export type HILMeasurementType =
  | "rms_current"
  | "peak_current"
  | "avg_current"
  | "rms_voltage"
  | "peak_voltage"
  | "avg_voltage"
  | "frequency"
  | "duty_cycle"
  | "dead_time"
  | "rise_time"
  | "fall_time"
  | "temperature"
  | "efficiency"
  | "thd"
  | "phase_angle"
  | "power"
  | "power_factor"
  | "speed"
  | "torque"
  | "position"
  | "startup_time"
  | "settling_time"
  | "overshoot"
  | "ripple"
  | "custom";

export interface HILInstrumentCapability {
  name: string;
  type: "protocol" | "feature" | "range" | "measurement";
  parameters?: Record<string, unknown>;
}

export interface HILInstrument {
  id: string;
  projectId: string;
  name: string;
  instrumentType: HILInstrumentType;
  manufacturer: string;
  model: string;
  serialNumber?: string;
  firmwareVersion?: string;
  connectionType: HILConnectionType;
  connectionParams: Record<string, unknown>;
  capabilities: HILInstrumentCapability[];
  status: HILInstrumentStatus;
  lastSeenAt?: string;
  lastError?: string;
  errorCount?: number;
  calibrationDate?: string;
  calibrationDueDate?: string;
  notes?: string;
  tags?: string[];
  createdAt: string;
  updatedAt: string;
}

export interface HILTestStep {
  id: string;
  name: string;
  type: string;
  instrumentId?: string;
  instrumentType?: HILInstrumentType;
  parameters: Record<string, unknown>;
  expectedResults?: Array<{
    measurement: string;
    channel?: string;
    operator: string;
    value: unknown;
    unit: string;
    isCritical?: boolean;
    description?: string;
  }>;
  timeout_ms?: number;
  description?: string;
}

export interface HILTestSequence {
  id: string;
  projectId: string;
  name: string;
  description?: string;
  testType: HILTestType;
  sequenceConfig: {
    steps: HILTestStep[];
    globalParameters?: Record<string, unknown>;
    instrumentRequirements: Array<{
      instrumentType: HILInstrumentType;
      capabilities?: string[];
      minChannels?: number;
      optional?: boolean;
      description?: string;
    }>;
    setupSteps?: HILTestStep[];
    teardownSteps?: HILTestStep[];
  };
  passCriteria: {
    minPassPercentage: number;
    criticalMeasurements: string[];
    failFast: boolean;
    allowedWarnings?: number;
  };
  estimatedDurationMs?: number;
  timeoutMs?: number;
  priority: number;
  tags: string[];
  category?: string;
  isTemplate: boolean;
  version: number;
  createdAt: string;
  updatedAt: string;
}

export interface HILTestRunSummary {
  totalMeasurements: number;
  passedMeasurements: number;
  failedMeasurements: number;
  warningMeasurements: number;
  criticalFailures: string[];
  keyMetrics: Record<string, number>;
  score?: number;
}

export interface HILTestRun {
  id: string;
  sequenceId: string;
  projectId: string;
  name: string;
  runNumber: number;
  status: HILTestRunStatus;
  result?: HILTestResult;
  progressPercentage: number;
  currentStep?: string;
  currentStepIndex?: number;
  totalSteps?: number;
  queuedAt?: string;
  startedAt?: string;
  completedAt?: string;
  durationMs?: number;
  errorMessage?: string;
  errorDetails?: Record<string, unknown>;
  testConditions?: Record<string, unknown>;
  instrumentSnapshot?: Record<string, HILInstrument>;
  summary?: HILTestRunSummary;
  workerId?: string;
  jobId?: string;
  retryCount: number;
  maxRetries: number;
  tags?: string[];
  createdAt: string;
  updatedAt: string;
}

export interface HILChannelConfig {
  name: string;
  label?: string;
  scale: number;
  offset: number;
  unit: string;
  coupling?: "ac" | "dc" | "gnd";
  probeAttenuation?: number;
  color?: string;
}

export interface HILCapturedData {
  id: string;
  testRunId: string;
  instrumentId?: string;
  name?: string;
  captureType: HILCaptureType;
  stepId?: string;
  channelConfig: HILChannelConfig[];
  sampleRateHz?: number;
  sampleCount?: number;
  durationMs?: number;
  dataFormat: string;
  dataPath?: string;
  dataSizeBytes?: number;
  analysisResults?: {
    fft?: {
      fundamental: number;
      harmonics: number[];
      thd: number;
    };
    measurements?: {
      rms: number[];
      peak: number[];
      peakToPeak: number[];
      mean: number[];
      frequency: number[];
    };
    statistics?: {
      min: number;
      max: number;
      mean: number;
      stddev: number;
      rms: number;
    };
  };
  annotations?: Array<{
    timestamp_ms: number;
    text: string;
    color?: string;
    type?: string;
  }>;
  capturedAt: string;
  createdAt: string;
}

export interface HILMeasurement {
  id: string;
  testRunId: string;
  capturedDataId?: string;
  stepId?: string;
  measurementType: HILMeasurementType;
  measurementName?: string;
  channel?: string;
  value: number;
  unit: string;
  minLimit?: number;
  maxLimit?: number;
  nominalValue?: number;
  tolerancePercent?: number;
  passed?: boolean;
  isCritical?: boolean;
  isWarning?: boolean;
  failureReason?: string;
  timestampOffsetMs?: number;
  measuredAt: string;
  sampleCount?: number;
  meanValue?: number;
  stdDeviation?: number;
  minObserved?: number;
  maxObserved?: number;
  createdAt: string;
}

export interface WaveformData {
  captureId: string;
  channels: string[];
  sampleRate: number;
  totalSamples: number;
  data: number[][]; // [channel][samples]
  timebase: {
    startTime: number;
    endTime: number;
    unit: string;
  };
  channelConfig: HILChannelConfig[];
}

export interface LiveMeasurement {
  type: HILMeasurementType;
  channel?: string;
  value: number;
  unit: string;
  passed?: boolean;
  timestamp: Date;
}

// ============================================================================
// Loading & Error States
// ============================================================================

interface HILLoadingStates {
  instruments: boolean;
  sequences: boolean;
  testRuns: boolean;
  measurements: boolean;
  captures: boolean;
  templates: boolean;
  discovery: boolean;
  connecting: boolean;
}

interface HILErrorStates {
  instruments: string | null;
  sequences: string | null;
  testRuns: string | null;
  measurements: string | null;
  captures: string | null;
  templates: string | null;
  discovery: string | null;
  connecting: string | null;
}

// ============================================================================
// Store Interface
// ============================================================================

interface HILState {
  // Instruments
  instruments: Record<string, HILInstrument>;
  selectedInstrumentId: string | null;
  discoveryInProgress: boolean;

  // Test Sequences
  testSequences: HILTestSequence[];
  selectedSequenceId: string | null;
  templates: HILTestSequence[];

  // Test Runs
  testRuns: HILTestRun[];
  activeTestRun: HILTestRun | null;
  activeTestRunId: string | null;

  // Measurements
  measurements: HILMeasurement[];
  liveMeasurements: LiveMeasurement[];
  measurementSummary: HILTestRunSummary | null;

  // Captured Data
  captures: HILCapturedData[];
  waveformData: WaveformData | null;
  selectedCaptureId: string | null;

  // UI State
  activePanel: "instruments" | "sequences" | "runs" | "results" | "waveform";
  isLiveViewEnabled: boolean;
  waveformZoom: { start: number; end: number };
  selectedChannels: string[];

  // WebSocket State
  wsConnected: boolean;
  wsSubscribedOperations: string[];

  // Loading & Error States
  loading: HILLoadingStates;
  errors: HILErrorStates;

  // ========================================================================
  // Actions - Instruments
  // ========================================================================
  setInstruments: (instruments: HILInstrument[]) => void;
  addInstrument: (instrument: HILInstrument) => void;
  updateInstrument: (id: string, update: Partial<HILInstrument>) => void;
  updateInstrumentStatus: (id: string, status: HILInstrumentStatus, error?: string) => void;
  removeInstrument: (id: string) => void;
  setSelectedInstrument: (id: string | null) => void;
  setDiscoveryInProgress: (inProgress: boolean) => void;

  // ========================================================================
  // Actions - Test Sequences
  // ========================================================================
  setTestSequences: (sequences: HILTestSequence[]) => void;
  addTestSequence: (sequence: HILTestSequence) => void;
  updateTestSequence: (id: string, update: Partial<HILTestSequence>) => void;
  removeTestSequence: (id: string) => void;
  setSelectedSequence: (id: string | null) => void;
  setTemplates: (templates: HILTestSequence[]) => void;

  // ========================================================================
  // Actions - Test Runs
  // ========================================================================
  setTestRuns: (runs: HILTestRun[]) => void;
  addTestRun: (run: HILTestRun) => void;
  updateTestRun: (id: string, update: Partial<HILTestRun>) => void;
  setActiveTestRun: (run: HILTestRun | null) => void;
  updateTestRunProgress: (id: string, progress: number, currentStep?: string, stepIndex?: number) => void;
  completeTestRun: (id: string, result: HILTestResult, summary: HILTestRunSummary) => void;
  failTestRun: (id: string, error: string, details?: Record<string, unknown>) => void;
  removeTestRun: (id: string) => void;

  // ========================================================================
  // Actions - Measurements
  // ========================================================================
  setMeasurements: (measurements: HILMeasurement[]) => void;
  addMeasurement: (measurement: HILMeasurement) => void;
  addLiveMeasurement: (measurement: LiveMeasurement) => void;
  clearLiveMeasurements: () => void;
  setMeasurementSummary: (summary: HILTestRunSummary | null) => void;

  // ========================================================================
  // Actions - Captured Data
  // ========================================================================
  setCaptures: (captures: HILCapturedData[]) => void;
  addCapture: (capture: HILCapturedData) => void;
  updateCapture: (id: string, update: Partial<HILCapturedData>) => void;
  setWaveformData: (data: WaveformData | null) => void;
  appendWaveformChunk: (captureId: string, channels: string[], startIndex: number, data: number[][]) => void;
  setSelectedCapture: (id: string | null) => void;

  // ========================================================================
  // Actions - UI State
  // ========================================================================
  setActivePanel: (panel: HILState["activePanel"]) => void;
  setLiveViewEnabled: (enabled: boolean) => void;
  setWaveformZoom: (zoom: { start: number; end: number }) => void;
  toggleChannel: (channel: string) => void;
  setSelectedChannels: (channels: string[]) => void;

  // ========================================================================
  // Actions - WebSocket
  // ========================================================================
  setWsConnected: (connected: boolean) => void;
  addWsSubscription: (operationId: string) => void;
  removeWsSubscription: (operationId: string) => void;
  clearWsSubscriptions: () => void;

  // ========================================================================
  // Actions - Loading & Error
  // ========================================================================
  setLoading: (key: keyof HILLoadingStates, loading: boolean) => void;
  setError: (key: keyof HILErrorStates, error: string | null) => void;
  clearErrors: () => void;

  // ========================================================================
  // Actions - Cleanup
  // ========================================================================
  reset: () => void;
  resetForProject: () => void;
}

// ============================================================================
// Initial State
// ============================================================================

const initialLoadingState: HILLoadingStates = {
  instruments: false,
  sequences: false,
  testRuns: false,
  measurements: false,
  captures: false,
  templates: false,
  discovery: false,
  connecting: false,
};

const initialErrorState: HILErrorStates = {
  instruments: null,
  sequences: null,
  testRuns: null,
  measurements: null,
  captures: null,
  templates: null,
  discovery: null,
  connecting: null,
};

// ============================================================================
// Store Implementation
// ============================================================================

export const useHILStore = create<HILState>()(
  immer((set, get) => ({
    // Initial State
    instruments: {},
    selectedInstrumentId: null,
    discoveryInProgress: false,

    testSequences: [],
    selectedSequenceId: null,
    templates: [],

    testRuns: [],
    activeTestRun: null,
    activeTestRunId: null,

    measurements: [],
    liveMeasurements: [],
    measurementSummary: null,

    captures: [],
    waveformData: null,
    selectedCaptureId: null,

    activePanel: "instruments",
    isLiveViewEnabled: false,
    waveformZoom: { start: 0, end: 1 },
    selectedChannels: [],

    wsConnected: false,
    wsSubscribedOperations: [],

    loading: initialLoadingState,
    errors: initialErrorState,

    // ========================================================================
    // Instrument Actions
    // ========================================================================

    setInstruments: (instruments: HILInstrument[]) => {
      set((state) => {
        state.instruments = {};
        for (const inst of instruments) {
          state.instruments[inst.id] = inst;
        }
      });
    },

    addInstrument: (instrument: HILInstrument) => {
      set((state) => {
        state.instruments[instrument.id] = instrument;
      });
    },

    updateInstrument: (id: string, update: Partial<HILInstrument>) => {
      set((state) => {
        if (state.instruments[id]) {
          Object.assign(state.instruments[id], update);
          state.instruments[id].updatedAt = new Date().toISOString();
        }
      });
    },

    updateInstrumentStatus: (id: string, status: HILInstrumentStatus, error?: string) => {
      set((state) => {
        if (state.instruments[id]) {
          state.instruments[id].status = status;
          if (error) {
            state.instruments[id].lastError = error;
            state.instruments[id].errorCount = (state.instruments[id].errorCount || 0) + 1;
          }
          if (status === "connected") {
            state.instruments[id].lastSeenAt = new Date().toISOString();
          }
        }
      });
    },

    removeInstrument: (id: string) => {
      set((state) => {
        delete state.instruments[id];
        if (state.selectedInstrumentId === id) {
          state.selectedInstrumentId = null;
        }
      });
    },

    setSelectedInstrument: (id: string | null) => {
      set((state) => {
        state.selectedInstrumentId = id;
      });
    },

    setDiscoveryInProgress: (inProgress: boolean) => {
      set((state) => {
        state.discoveryInProgress = inProgress;
        state.loading.discovery = inProgress;
      });
    },

    // ========================================================================
    // Test Sequence Actions
    // ========================================================================

    setTestSequences: (sequences: HILTestSequence[]) => {
      set((state) => {
        state.testSequences = sequences;
      });
    },

    addTestSequence: (sequence: HILTestSequence) => {
      set((state) => {
        const index = state.testSequences.findIndex((s) => s.id === sequence.id);
        if (index >= 0) {
          state.testSequences[index] = sequence;
        } else {
          state.testSequences.push(sequence);
        }
      });
    },

    updateTestSequence: (id: string, update: Partial<HILTestSequence>) => {
      set((state) => {
        const index = state.testSequences.findIndex((s) => s.id === id);
        if (index >= 0) {
          Object.assign(state.testSequences[index], update);
        }
      });
    },

    removeTestSequence: (id: string) => {
      set((state) => {
        state.testSequences = state.testSequences.filter((s) => s.id !== id);
        if (state.selectedSequenceId === id) {
          state.selectedSequenceId = null;
        }
      });
    },

    setSelectedSequence: (id: string | null) => {
      set((state) => {
        state.selectedSequenceId = id;
      });
    },

    setTemplates: (templates: HILTestSequence[]) => {
      set((state) => {
        state.templates = templates;
      });
    },

    // ========================================================================
    // Test Run Actions
    // ========================================================================

    setTestRuns: (runs: HILTestRun[]) => {
      set((state) => {
        state.testRuns = runs;
      });
    },

    addTestRun: (run: HILTestRun) => {
      set((state) => {
        const index = state.testRuns.findIndex((r) => r.id === run.id);
        if (index >= 0) {
          state.testRuns[index] = run;
        } else {
          state.testRuns.unshift(run); // Add to beginning (newest first)
        }
      });
    },

    updateTestRun: (id: string, update: Partial<HILTestRun>) => {
      set((state) => {
        const index = state.testRuns.findIndex((r) => r.id === id);
        if (index >= 0) {
          Object.assign(state.testRuns[index], update);
        }
        if (state.activeTestRun?.id === id) {
          Object.assign(state.activeTestRun, update);
        }
      });
    },

    setActiveTestRun: (run: HILTestRun | null) => {
      set((state) => {
        state.activeTestRun = run;
        state.activeTestRunId = run?.id || null;
        // Clear live measurements when starting new run
        if (run && run.status === "running") {
          state.liveMeasurements = [];
        }
      });
    },

    updateTestRunProgress: (id: string, progress: number, currentStep?: string, stepIndex?: number) => {
      set((state) => {
        const index = state.testRuns.findIndex((r) => r.id === id);
        if (index >= 0) {
          state.testRuns[index].progressPercentage = progress;
          if (currentStep !== undefined) {
            state.testRuns[index].currentStep = currentStep;
          }
          if (stepIndex !== undefined) {
            state.testRuns[index].currentStepIndex = stepIndex;
          }
        }
        if (state.activeTestRun?.id === id) {
          state.activeTestRun.progressPercentage = progress;
          if (currentStep !== undefined) {
            state.activeTestRun.currentStep = currentStep;
          }
          if (stepIndex !== undefined) {
            state.activeTestRun.currentStepIndex = stepIndex;
          }
        }
      });
    },

    completeTestRun: (id: string, result: HILTestResult, summary: HILTestRunSummary) => {
      set((state) => {
        const index = state.testRuns.findIndex((r) => r.id === id);
        if (index >= 0) {
          state.testRuns[index].status = "completed";
          state.testRuns[index].result = result;
          state.testRuns[index].summary = summary;
          state.testRuns[index].progressPercentage = 100;
          state.testRuns[index].completedAt = new Date().toISOString();
        }
        if (state.activeTestRun?.id === id) {
          state.activeTestRun.status = "completed";
          state.activeTestRun.result = result;
          state.activeTestRun.summary = summary;
          state.activeTestRun.progressPercentage = 100;
          state.activeTestRun.completedAt = new Date().toISOString();
        }
        state.measurementSummary = summary;
      });
    },

    failTestRun: (id: string, error: string, details?: Record<string, unknown>) => {
      set((state) => {
        const index = state.testRuns.findIndex((r) => r.id === id);
        if (index >= 0) {
          state.testRuns[index].status = "failed";
          state.testRuns[index].result = "fail";
          state.testRuns[index].errorMessage = error;
          state.testRuns[index].errorDetails = details;
          state.testRuns[index].completedAt = new Date().toISOString();
        }
        if (state.activeTestRun?.id === id) {
          state.activeTestRun.status = "failed";
          state.activeTestRun.result = "fail";
          state.activeTestRun.errorMessage = error;
          state.activeTestRun.errorDetails = details;
          state.activeTestRun.completedAt = new Date().toISOString();
        }
      });
    },

    removeTestRun: (id: string) => {
      set((state) => {
        state.testRuns = state.testRuns.filter((r) => r.id !== id);
        if (state.activeTestRunId === id) {
          state.activeTestRun = null;
          state.activeTestRunId = null;
        }
      });
    },

    // ========================================================================
    // Measurement Actions
    // ========================================================================

    setMeasurements: (measurements: HILMeasurement[]) => {
      set((state) => {
        state.measurements = measurements;
      });
    },

    addMeasurement: (measurement: HILMeasurement) => {
      set((state) => {
        state.measurements.push(measurement);
      });
    },

    addLiveMeasurement: (measurement: LiveMeasurement) => {
      set((state) => {
        // Keep only last 100 live measurements
        if (state.liveMeasurements.length >= 100) {
          state.liveMeasurements.shift();
        }
        state.liveMeasurements.push(measurement);
      });
    },

    clearLiveMeasurements: () => {
      set((state) => {
        state.liveMeasurements = [];
      });
    },

    setMeasurementSummary: (summary: HILTestRunSummary | null) => {
      set((state) => {
        state.measurementSummary = summary;
      });
    },

    // ========================================================================
    // Captured Data Actions
    // ========================================================================

    setCaptures: (captures: HILCapturedData[]) => {
      set((state) => {
        state.captures = captures;
      });
    },

    addCapture: (capture: HILCapturedData) => {
      set((state) => {
        const index = state.captures.findIndex((c) => c.id === capture.id);
        if (index >= 0) {
          state.captures[index] = capture;
        } else {
          state.captures.push(capture);
        }
      });
    },

    updateCapture: (id: string, update: Partial<HILCapturedData>) => {
      set((state) => {
        const index = state.captures.findIndex((c) => c.id === id);
        if (index >= 0) {
          Object.assign(state.captures[index], update);
        }
      });
    },

    setWaveformData: (data: WaveformData | null) => {
      set((state) => {
        state.waveformData = data;
        if (data) {
          state.selectedChannels = data.channels;
          state.waveformZoom = { start: 0, end: 1 };
        }
      });
    },

    appendWaveformChunk: (captureId: string, channels: string[], startIndex: number, data: number[][]) => {
      set((state) => {
        if (!state.waveformData || state.waveformData.captureId !== captureId) {
          // Initialize new waveform data
          state.waveformData = {
            captureId,
            channels,
            sampleRate: 0, // Will be updated later
            totalSamples: startIndex + (data[0]?.length || 0),
            data,
            timebase: {
              startTime: 0,
              endTime: 0,
              unit: "s",
            },
            channelConfig: channels.map((ch) => ({
              name: ch,
              scale: 1,
              offset: 0,
              unit: "V",
            })),
          };
        } else {
          // Append to existing data
          for (let i = 0; i < channels.length; i++) {
            if (!state.waveformData.data[i]) {
              state.waveformData.data[i] = [];
            }
            state.waveformData.data[i].push(...data[i]);
          }
          state.waveformData.totalSamples = state.waveformData.data[0]?.length || 0;
        }
      });
    },

    setSelectedCapture: (id: string | null) => {
      set((state) => {
        state.selectedCaptureId = id;
      });
    },

    // ========================================================================
    // UI State Actions
    // ========================================================================

    setActivePanel: (panel: HILState["activePanel"]) => {
      set((state) => {
        state.activePanel = panel;
      });
    },

    setLiveViewEnabled: (enabled: boolean) => {
      set((state) => {
        state.isLiveViewEnabled = enabled;
      });
    },

    setWaveformZoom: (zoom: { start: number; end: number }) => {
      set((state) => {
        state.waveformZoom = zoom;
      });
    },

    toggleChannel: (channel: string) => {
      set((state) => {
        const index = state.selectedChannels.indexOf(channel);
        if (index >= 0) {
          state.selectedChannels.splice(index, 1);
        } else {
          state.selectedChannels.push(channel);
        }
      });
    },

    setSelectedChannels: (channels: string[]) => {
      set((state) => {
        state.selectedChannels = channels;
      });
    },

    // ========================================================================
    // WebSocket Actions
    // ========================================================================

    setWsConnected: (connected: boolean) => {
      set((state) => {
        state.wsConnected = connected;
      });
    },

    addWsSubscription: (operationId: string) => {
      set((state) => {
        if (!state.wsSubscribedOperations.includes(operationId)) {
          state.wsSubscribedOperations.push(operationId);
        }
      });
    },

    removeWsSubscription: (operationId: string) => {
      set((state) => {
        state.wsSubscribedOperations = state.wsSubscribedOperations.filter(
          (id) => id !== operationId
        );
      });
    },

    clearWsSubscriptions: () => {
      set((state) => {
        state.wsSubscribedOperations = [];
      });
    },

    // ========================================================================
    // Loading & Error Actions
    // ========================================================================

    setLoading: (key: keyof HILLoadingStates, loading: boolean) => {
      set((state) => {
        state.loading[key] = loading;
      });
    },

    setError: (key: keyof HILErrorStates, error: string | null) => {
      set((state) => {
        state.errors[key] = error;
      });
    },

    clearErrors: () => {
      set((state) => {
        state.errors = { ...initialErrorState };
      });
    },

    // ========================================================================
    // Cleanup Actions
    // ========================================================================

    reset: () => {
      set((state) => {
        state.instruments = {};
        state.selectedInstrumentId = null;
        state.discoveryInProgress = false;
        state.testSequences = [];
        state.selectedSequenceId = null;
        state.templates = [];
        state.testRuns = [];
        state.activeTestRun = null;
        state.activeTestRunId = null;
        state.measurements = [];
        state.liveMeasurements = [];
        state.measurementSummary = null;
        state.captures = [];
        state.waveformData = null;
        state.selectedCaptureId = null;
        state.activePanel = "instruments";
        state.isLiveViewEnabled = false;
        state.waveformZoom = { start: 0, end: 1 };
        state.selectedChannels = [];
        state.wsConnected = false;
        state.wsSubscribedOperations = [];
        state.loading = { ...initialLoadingState };
        state.errors = { ...initialErrorState };
      });
    },

    resetForProject: () => {
      set((state) => {
        // Keep templates and WebSocket connection, reset project-specific data
        state.instruments = {};
        state.selectedInstrumentId = null;
        state.discoveryInProgress = false;
        state.testSequences = [];
        state.selectedSequenceId = null;
        state.testRuns = [];
        state.activeTestRun = null;
        state.activeTestRunId = null;
        state.measurements = [];
        state.liveMeasurements = [];
        state.measurementSummary = null;
        state.captures = [];
        state.waveformData = null;
        state.selectedCaptureId = null;
        state.loading = { ...initialLoadingState };
        state.errors = { ...initialErrorState };
      });
    },
  }))
);

// ============================================================================
// Selectors for common queries
// ============================================================================

export const selectConnectedInstruments = (state: HILState): HILInstrument[] =>
  Object.values(state.instruments).filter((i) => i.status === "connected");

export const selectInstrumentsByType = (
  state: HILState,
  type: HILInstrumentType
): HILInstrument[] =>
  Object.values(state.instruments).filter((i) => i.instrumentType === type);

export const selectActiveSequence = (state: HILState): HILTestSequence | null =>
  state.testSequences.find((s) => s.id === state.selectedSequenceId) || null;

export const selectRunningTestRuns = (state: HILState): HILTestRun[] =>
  state.testRuns.filter((r) => r.status === "running" || r.status === "queued");

export const selectCompletedTestRuns = (state: HILState): HILTestRun[] =>
  state.testRuns.filter(
    (r) => r.status === "completed" || r.status === "failed" || r.status === "aborted"
  );

export const selectFailedMeasurements = (state: HILState): HILMeasurement[] =>
  state.measurements.filter((m) => m.passed === false);

export const selectCriticalFailures = (state: HILState): HILMeasurement[] =>
  state.measurements.filter((m) => m.passed === false && m.isCritical);

export default useHILStore;
