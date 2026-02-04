/**
 * HIL (Hardware-in-the-Loop) Testing Type Definitions
 *
 * Complete type definitions for instruments, test sequences, test runs,
 * captured data, measurements, and FOC ESC specific parameters.
 */

// ============================================================================
// INSTRUMENT TYPES
// ============================================================================

/**
 * Supported instrument types for HIL testing
 */
export type HILInstrumentType =
  | 'logic_analyzer'
  | 'oscilloscope'
  | 'power_supply'
  | 'motor_emulator'
  | 'daq'
  | 'can_analyzer'
  | 'function_gen'
  | 'thermal_camera'
  | 'electronic_load';

/**
 * Connection types for instruments
 */
export type HILConnectionType =
  | 'usb'
  | 'ethernet'
  | 'gpib'
  | 'serial'
  | 'grpc'
  | 'modbus_tcp'
  | 'modbus_rtu';

/**
 * Instrument connection status
 */
export type HILInstrumentStatus =
  | 'connected'
  | 'disconnected'
  | 'error'
  | 'busy'
  | 'initializing';

/**
 * Instrument capability definition
 */
export interface HILInstrumentCapability {
  /** Capability name (e.g., 'digital_channels', 'spi', 'voltage_range') */
  name: string;
  /** Capability type */
  type: 'protocol' | 'feature' | 'range' | 'measurement';
  /** Capability parameters (varies by type) */
  parameters?: {
    count?: number;
    max_sample_rate?: number;
    min?: number;
    max?: number;
    unit?: string;
    resolution?: number;
    [key: string]: unknown;
  };
}

/**
 * Connection parameters for different connection types
 */
export interface HILConnectionParams {
  // USB
  serial_number?: string;
  vendor_id?: string;
  product_id?: string;

  // Ethernet
  host?: string;
  port?: number;

  // GPIB
  address?: number;
  board?: number;

  // Serial
  serial_port?: string;
  baud_rate?: number;
  data_bits?: number;
  stop_bits?: number;
  parity?: 'none' | 'odd' | 'even';

  // VISA
  resource_name?: string;

  // Custom
  [key: string]: unknown;
}

/**
 * Instrument preset configuration
 */
export interface HILInstrumentPreset {
  name: string;
  description?: string;
  config: Record<string, unknown>;
  createdAt?: string;
}

/**
 * HIL Instrument definition
 */
export interface HILInstrument {
  id: string;
  projectId: string;

  // Identification
  name: string;
  instrumentType: HILInstrumentType;
  manufacturer: string;
  model: string;
  serialNumber?: string;
  firmwareVersion?: string;

  // Connection
  connectionType: HILConnectionType;
  connectionParams: HILConnectionParams;

  // Capabilities
  capabilities: HILInstrumentCapability[];

  // Status
  status: HILInstrumentStatus;
  lastSeenAt?: string;
  lastError?: string;
  errorCount?: number;

  // Calibration
  calibrationDate?: string;
  calibrationDueDate?: string;
  calibrationCertificate?: string;

  // Configuration
  presets?: Record<string, HILInstrumentPreset>;
  defaultPreset?: string;

  // Metadata
  notes?: string;
  tags?: string[];
  metadata?: Record<string, unknown>;

  // Timestamps
  createdAt: string;
  updatedAt: string;
}

// ============================================================================
// TEST SEQUENCE TYPES
// ============================================================================

/**
 * Test types for FOC ESC and general HIL testing
 */
export type HILTestType =
  | 'foc_startup'
  | 'foc_steady_state'
  | 'foc_transient'
  | 'foc_speed_reversal'
  | 'pwm_analysis'
  | 'phase_current'
  | 'hall_sensor'
  | 'thermal_profile'
  | 'overcurrent_protection'
  | 'efficiency_sweep'
  | 'load_step'
  | 'no_load'
  | 'locked_rotor'
  | 'custom';

/**
 * Test step types
 */
export type HILTestStepType =
  | 'configure'      // Configure an instrument
  | 'measure'        // Take a measurement
  | 'capture'        // Capture waveform/logic data
  | 'wait'           // Wait for time or condition
  | 'control'        // Control output (PSU, load, etc.)
  | 'validate'       // Validate a condition
  | 'loop'           // Loop/iterate
  | 'branch'         // Conditional branch
  | 'script';        // Run custom script

/**
 * Comparison operators for expected results
 */
export type HILComparisonOperator =
  | 'eq'        // Equal
  | 'ne'        // Not equal
  | 'gt'        // Greater than
  | 'gte'       // Greater than or equal
  | 'lt'        // Less than
  | 'lte'       // Less than or equal
  | 'range'     // Within range [min, max]
  | 'tolerance' // Within tolerance of nominal
  | 'contains'  // Contains value
  | 'regex';    // Matches regex

/**
 * Expected result definition for a test step
 */
export interface HILExpectedResult {
  /** Measurement type to check */
  measurement: string;
  /** Channel (if applicable) */
  channel?: string;
  /** Comparison operator */
  operator: HILComparisonOperator;
  /** Expected value or range */
  value: number | string | [number, number];
  /** Unit of measurement */
  unit: string;
  /** Tolerance percentage (for 'tolerance' operator) */
  tolerancePercent?: number;
  /** Tolerance absolute (for 'tolerance' operator) */
  toleranceAbsolute?: number;
  /** Is this a critical measurement? */
  isCritical?: boolean;
  /** Description of expected result */
  description?: string;
}

/**
 * Test step definition
 */
export interface HILTestStep {
  /** Unique step ID */
  id: string;
  /** Step name */
  name: string;
  /** Step type */
  type: HILTestStepType;
  /** Target instrument ID */
  instrumentId?: string;
  /** Instrument type (for auto-selection) */
  instrumentType?: HILInstrumentType;
  /** Step parameters (varies by type) */
  parameters: Record<string, unknown>;
  /** Expected results to validate */
  expectedResults?: HILExpectedResult[];
  /** Step timeout in ms */
  timeout_ms?: number;
  /** Retry on failure */
  retryOnFail?: boolean;
  /** Maximum retries */
  maxRetries?: number;
  /** Continue sequence on failure */
  continueOnFail?: boolean;
  /** Step description */
  description?: string;
  /** Delay before step (ms) */
  delayBefore_ms?: number;
  /** Delay after step (ms) */
  delayAfter_ms?: number;
}

/**
 * Instrument requirement for a test sequence
 */
export interface HILInstrumentRequirement {
  /** Required instrument type */
  instrumentType: HILInstrumentType;
  /** Required capabilities */
  capabilities?: string[];
  /** Minimum channels */
  minChannels?: number;
  /** Minimum sample rate (Hz) */
  minSampleRate?: number;
  /** Is this instrument optional? */
  optional?: boolean;
  /** Description of requirement */
  description?: string;
}

/**
 * Sequence configuration
 */
export interface HILSequenceConfig {
  /** Test steps */
  steps: HILTestStep[];
  /** Global parameters accessible to all steps */
  globalParameters?: Record<string, unknown>;
  /** Required instruments */
  instrumentRequirements: HILInstrumentRequirement[];
  /** Setup steps (run before main steps) */
  setupSteps?: HILTestStep[];
  /** Teardown steps (run after main steps, even on failure) */
  teardownSteps?: HILTestStep[];
}

/**
 * Pass criteria for a test sequence
 */
export interface HILPassCriteria {
  /** Minimum percentage of measurements that must pass */
  minPassPercentage: number;
  /** Measurements that must pass for test to pass */
  criticalMeasurements: string[];
  /** Stop test on first failure */
  failFast: boolean;
  /** Maximum allowed warnings */
  allowedWarnings?: number;
  /** Custom pass condition expression */
  customCondition?: string;
}

/**
 * Template variable definition
 */
export interface HILTemplateVariable {
  name: string;
  type: 'number' | 'string' | 'boolean' | 'select';
  label: string;
  description?: string;
  defaultValue?: unknown;
  required?: boolean;
  options?: Array<{ value: unknown; label: string }>;
  min?: number;
  max?: number;
  unit?: string;
}

/**
 * HIL Test Sequence definition
 */
export interface HILTestSequence {
  id: string;
  projectId: string;

  // Links
  schematicId?: string;
  pcbLayoutId?: string;

  // Identification
  name: string;
  description?: string;
  testType: HILTestType;

  // Configuration
  sequenceConfig: HILSequenceConfig;
  passCriteria: HILPassCriteria;

  // Execution
  estimatedDurationMs?: number;
  timeoutMs?: number;
  priority: number;

  // Organization
  tags: string[];
  category?: string;

  // Versioning
  version: number;
  parentVersionId?: string;

  // Template
  isTemplate: boolean;
  templateVariables?: Record<string, HILTemplateVariable>;
  parentTemplateId?: string;

  // Audit
  createdBy?: string;
  lastModifiedBy?: string;

  // Metadata
  metadata?: Record<string, unknown>;

  // Timestamps
  createdAt: string;
  updatedAt: string;
}

// ============================================================================
// TEST RUN TYPES
// ============================================================================

/**
 * Test run status
 */
export type HILTestRunStatus =
  | 'pending'
  | 'queued'
  | 'running'
  | 'completed'
  | 'failed'
  | 'aborted'
  | 'timeout';

/**
 * Test result
 */
export type HILTestResult = 'pass' | 'fail' | 'partial' | 'inconclusive';

/**
 * Test conditions at start of run
 */
export interface HILTestConditions {
  /** Ambient temperature (C) */
  ambientTemperature?: number;
  /** Relative humidity (%) */
  humidity?: number;
  /** Supply voltage (V) */
  supplyVoltage?: number;
  /** Atmospheric pressure (hPa) */
  pressure?: number;
  /** Operator notes */
  notes?: string;
  /** Custom conditions */
  [key: string]: unknown;
}

/**
 * Test run summary
 */
export interface HILTestRunSummary {
  /** Total number of measurements */
  totalMeasurements: number;
  /** Number of passed measurements */
  passedMeasurements: number;
  /** Number of failed measurements */
  failedMeasurements: number;
  /** Number of warning measurements */
  warningMeasurements: number;
  /** List of critical failures */
  criticalFailures: string[];
  /** Key metrics from the test */
  keyMetrics: Record<string, number>;
  /** Overall score (0-100) */
  score?: number;
}

/**
 * HIL Test Run definition
 */
export interface HILTestRun {
  id: string;
  sequenceId: string;
  projectId: string;

  // Identification
  name: string;
  runNumber: number;

  // Status
  status: HILTestRunStatus;
  result?: HILTestResult;

  // Progress
  progressPercentage: number;
  currentStep?: string;
  currentStepIndex?: number;
  totalSteps?: number;

  // Timing
  queuedAt?: string;
  startedAt?: string;
  completedAt?: string;
  durationMs?: number;

  // Error
  errorMessage?: string;
  errorDetails?: Record<string, unknown>;
  errorStepId?: string;

  // Conditions
  testConditions?: HILTestConditions;
  instrumentSnapshot?: Record<string, HILInstrument>;

  // Results
  summary?: HILTestRunSummary;

  // Worker
  workerId?: string;
  workerHost?: string;
  jobId?: string;

  // Retry
  retryCount: number;
  maxRetries: number;
  lastRetryAt?: string;

  // Comparison
  baselineRunId?: string;
  comparisonResults?: Record<string, unknown>;

  // Audit
  startedBy?: string;
  abortedBy?: string;
  abortReason?: string;

  // Metadata
  tags?: string[];
  metadata?: Record<string, unknown>;

  // Timestamps
  createdAt: string;
  updatedAt: string;
}

// ============================================================================
// CAPTURED DATA TYPES
// ============================================================================

/**
 * Capture data types
 */
export type HILCaptureType =
  | 'waveform'
  | 'logic_trace'
  | 'spectrum'
  | 'thermal_image'
  | 'measurement'
  | 'protocol_decode'
  | 'can_log';

/**
 * Data formats for captured data
 */
export type HILDataFormat =
  | 'binary'
  | 'csv'
  | 'json'
  | 'vcd'
  | 'salae'
  | 'sigrok';

/**
 * Channel configuration for captured data
 */
export interface HILChannelConfig {
  /** Channel identifier (e.g., 'CH1', 'D0') */
  name: string;
  /** Display label */
  label?: string;
  /** Vertical scale */
  scale: number;
  /** Vertical offset */
  offset: number;
  /** Unit of measurement */
  unit: string;
  /** Coupling mode */
  coupling?: 'ac' | 'dc' | 'gnd';
  /** Input impedance (ohms) */
  impedance?: number;
  /** Probe attenuation */
  probeAttenuation?: number;
  /** Bandwidth limit (Hz) */
  bandwidthLimit?: number;
  /** Color for display */
  color?: string;
}

/**
 * Trigger configuration
 */
export interface HILTriggerConfig {
  /** Trigger source */
  source: string;
  /** Trigger level */
  level: number;
  /** Trigger edge */
  edge: 'rising' | 'falling' | 'either';
  /** Trigger position (0-1) */
  position: number;
  /** Trigger mode */
  mode?: 'auto' | 'normal' | 'single';
  /** Holdoff time (s) */
  holdoff?: number;
}

/**
 * Data annotation
 */
export interface HILDataAnnotation {
  /** Timestamp in ms from start */
  timestamp_ms: number;
  /** Annotation text */
  text: string;
  /** Display color */
  color?: string;
  /** Annotation type */
  type?: 'marker' | 'region' | 'note';
  /** End timestamp for regions */
  endTimestamp_ms?: number;
}

/**
 * Analysis results for captured data
 */
export interface HILAnalysisResults {
  /** FFT results */
  fft?: {
    fundamental: number;
    harmonics: number[];
    thd: number;
    magnitudes: number[];
    frequencies: number[];
  };
  /** Per-channel measurements */
  measurements?: {
    rms: number[];
    peak: number[];
    peakToPeak: number[];
    mean: number[];
    frequency: number[];
  };
  /** Statistics */
  statistics?: {
    min: number;
    max: number;
    mean: number;
    stddev: number;
    rms: number;
  };
  /** Protocol decode results */
  protocolDecode?: Array<{
    timestamp: number;
    type: string;
    data: unknown;
    error?: string;
  }>;
}

/**
 * HIL Captured Data definition
 */
export interface HILCapturedData {
  id: string;
  testRunId: string;
  instrumentId?: string;

  // Identification
  name?: string;
  captureType: HILCaptureType;
  stepId?: string;

  // Configuration
  channelConfig: HILChannelConfig[];

  // Sampling
  sampleRateHz?: number;
  sampleCount?: number;
  durationMs?: number;

  // Trigger
  triggerConfig?: HILTriggerConfig;

  // Data storage
  dataFormat: HILDataFormat;
  dataPath?: string;
  dataInline?: unknown;
  dataSizeBytes?: number;
  dataChecksum?: string;
  compression?: string;

  // Analysis
  analysisResults?: HILAnalysisResults;

  // Annotations
  annotations?: HILDataAnnotation[];

  // Timing
  capturedAt: string;
  processingCompletedAt?: string;

  // Metadata
  metadata?: Record<string, unknown>;

  // Timestamps
  createdAt: string;
}

// ============================================================================
// MEASUREMENT TYPES
// ============================================================================

/**
 * Measurement types
 */
export type HILMeasurementType =
  | 'rms_current'
  | 'peak_current'
  | 'avg_current'
  | 'rms_voltage'
  | 'peak_voltage'
  | 'avg_voltage'
  | 'frequency'
  | 'duty_cycle'
  | 'dead_time'
  | 'rise_time'
  | 'fall_time'
  | 'temperature'
  | 'efficiency'
  | 'thd'
  | 'phase_angle'
  | 'power'
  | 'power_factor'
  | 'speed'
  | 'torque'
  | 'position'
  | 'startup_time'
  | 'settling_time'
  | 'overshoot'
  | 'ripple'
  | 'custom';

/**
 * HIL Measurement definition
 */
export interface HILMeasurement {
  id: string;
  testRunId: string;
  capturedDataId?: string;
  stepId?: string;

  // Identification
  measurementType: HILMeasurementType;
  measurementName?: string;
  channel?: string;

  // Value
  value: number;
  unit: string;

  // Limits
  minLimit?: number;
  maxLimit?: number;
  nominalValue?: number;
  tolerancePercent?: number;
  toleranceAbsolute?: number;

  // Result
  passed?: boolean;
  isCritical?: boolean;
  isWarning?: boolean;
  failureReason?: string;

  // Timing
  timestampOffsetMs?: number;
  measuredAt: string;

  // Statistics
  sampleCount?: number;
  meanValue?: number;
  stdDeviation?: number;
  minObserved?: number;
  maxObserved?: number;

  // Metadata
  metadata?: Record<string, unknown>;

  // Timestamps
  createdAt: string;
}

// ============================================================================
// FOC ESC SPECIFIC TYPES
// ============================================================================

/**
 * Motor parameters for FOC testing
 */
export interface FOCMotorParameters {
  /** Number of pole pairs */
  polePairs: number;
  /** Rated current (A) */
  ratedCurrent: number;
  /** Maximum current (A) */
  maxCurrent: number;
  /** Rated speed (RPM) */
  ratedSpeed: number;
  /** Maximum speed (RPM) */
  maxSpeed: number;
  /** Rated torque (Nm) */
  ratedTorque?: number;
  /** Motor resistance (ohms) */
  resistance?: number;
  /** Motor inductance (H) */
  inductance?: number;
  /** Back-EMF constant (V/rad/s) */
  backEmfConstant?: number;
  /** Rotor inertia (kg*m^2) */
  rotorInertia?: number;
  /** Encoder resolution (PPR) */
  encoderResolution?: number;
  /** Hall sensor type */
  hallSensorType?: 'digital' | 'analog' | 'none';
  /** Hall sensor configuration */
  hallConfig?: number;
}

/**
 * FOC controller parameters
 */
export interface FOCControllerParameters {
  /** PWM frequency (Hz) */
  pwmFrequency: number;
  /** Dead time (ns) */
  deadTime: number;
  /** Current loop Kp */
  currentKp: number;
  /** Current loop Ki */
  currentKi: number;
  /** Speed loop Kp */
  speedKp: number;
  /** Speed loop Ki */
  speedKi: number;
  /** Position loop Kp */
  positionKp?: number;
  /** Field weakening enabled */
  fieldWeakeningEnabled?: boolean;
  /** Maximum field weakening current (A) */
  maxFieldWeakeningCurrent?: number;
  /** Current limit (A) */
  currentLimit: number;
  /** Acceleration limit (RPM/s) */
  accelerationLimit?: number;
}

/**
 * FOC test parameters
 */
export interface FOCTestParameters {
  /** Motor parameters */
  motor: FOCMotorParameters;
  /** Controller parameters */
  controller: FOCControllerParameters;
  /** DC bus voltage (V) */
  dcBusVoltage: number;
  /** Target speed (RPM) */
  targetSpeed: number;
  /** Target torque (Nm) - for torque mode */
  targetTorque?: number;
  /** Load torque profile */
  loadProfile?: Array<{
    time_ms: number;
    torque: number;
  }>;
  /** Test duration (ms) */
  testDuration: number;
}

/**
 * FOC measurement results
 */
export interface FOCMeasurementResult {
  /** Phase currents */
  phaseCurrents: {
    ia: { rms: number; peak: number; thd: number };
    ib: { rms: number; peak: number; thd: number };
    ic: { rms: number; peak: number; thd: number };
  };
  /** DC bus current */
  dcBusCurrent: { average: number; ripple: number };
  /** DC bus voltage */
  dcBusVoltage: { average: number; ripple: number };
  /** Speed */
  speed: { actual: number; error: number; ripple: number };
  /** Torque */
  torque?: { actual: number; ripple: number };
  /** Efficiency */
  efficiency: number;
  /** Input power (W) */
  inputPower: number;
  /** Output power (W) */
  outputPower?: number;
  /** Power losses (W) */
  losses?: number;
  /** Temperatures */
  temperatures?: {
    mosfets?: number[];
    motor?: number;
    ambient?: number;
  };
  /** PWM analysis */
  pwmDutyCycles?: {
    high: number[];
    low: number[];
    deadTime: number;
  };
  /** Hall sensor states */
  hallStates?: number[];
  /** Encoder position */
  encoderPosition?: number;
  /** Faults detected */
  faults?: string[];
  /** Startup metrics */
  startup?: {
    timeToSpeed_ms: number;
    currentOvershoot: number;
    speedOvershoot: number;
  };
}

// ============================================================================
// WEBSOCKET EVENT TYPES
// ============================================================================

/**
 * HIL WebSocket event types
 */
export enum HILEventType {
  // Instrument events
  INSTRUMENT_DISCOVERED = 'instrument_discovered',
  INSTRUMENT_CONNECTED = 'instrument_connected',
  INSTRUMENT_DISCONNECTED = 'instrument_disconnected',
  INSTRUMENT_ERROR = 'instrument_error',
  INSTRUMENT_STATUS_CHANGED = 'instrument_status_changed',

  // Test run events
  TEST_RUN_QUEUED = 'test_run_queued',
  TEST_RUN_STARTED = 'test_run_started',
  TEST_STEP_STARTED = 'test_step_started',
  TEST_STEP_COMPLETED = 'test_step_completed',
  TEST_MEASUREMENT = 'test_measurement',
  TEST_RUN_PROGRESS = 'test_run_progress',
  TEST_RUN_COMPLETED = 'test_run_completed',
  TEST_RUN_FAILED = 'test_run_failed',
  TEST_RUN_ABORTED = 'test_run_aborted',

  // Data streaming events
  WAVEFORM_CHUNK = 'waveform_chunk',
  LOGIC_TRACE_CHUNK = 'logic_trace_chunk',
  LIVE_MEASUREMENT = 'live_measurement',
  CAPTURE_STARTED = 'capture_started',
  CAPTURE_COMPLETE = 'capture_complete',

  // Control events
  MOTOR_STATE_CHANGED = 'motor_state_changed',
  POWER_STATE_CHANGED = 'power_state_changed',
  FAULT_DETECTED = 'fault_detected',
  EMERGENCY_STOP = 'emergency_stop',

  // Final states
  COMPLETE = 'complete',
  ERROR = 'error',
}

/**
 * HIL operation phase
 */
export type HILPhase =
  | 'discovery'
  | 'configuration'
  | 'setup'
  | 'measurement'
  | 'capture'
  | 'analysis'
  | 'validation'
  | 'teardown';

/**
 * Waveform chunk for streaming
 */
export interface HILWaveformChunk {
  /** Capture ID */
  captureId: string;
  /** Channel names */
  channels: string[];
  /** Start sample index */
  startIndex: number;
  /** Number of samples in this chunk */
  sampleCount: number;
  /** Sample data [channel][samples] */
  data: number[][];
  /** Is this the last chunk? */
  isLast?: boolean;
}

/**
 * Live measurement for streaming
 */
export interface HILLiveMeasurement {
  /** Measurement type */
  type: HILMeasurementType;
  /** Channel */
  channel?: string;
  /** Value */
  value: number;
  /** Unit */
  unit: string;
  /** Pass/fail status */
  passed?: boolean;
  /** Timestamp offset from start (ms) */
  timestampOffset?: number;
}

// ============================================================================
// API INPUT/OUTPUT TYPES
// ============================================================================

/**
 * Input for creating an instrument
 */
export interface CreateHILInstrumentInput {
  projectId: string;
  name: string;
  instrumentType: HILInstrumentType;
  manufacturer: string;
  model: string;
  connectionType: HILConnectionType;
  connectionParams: HILConnectionParams;
  capabilities?: HILInstrumentCapability[];
  serialNumber?: string;
  firmwareVersion?: string;
  notes?: string;
  tags?: string[];
  metadata?: Record<string, unknown>;
}

/**
 * Input for updating an instrument
 */
export interface UpdateHILInstrumentInput {
  name?: string;
  connectionParams?: HILConnectionParams;
  capabilities?: HILInstrumentCapability[];
  status?: HILInstrumentStatus;
  firmwareVersion?: string;
  calibrationDate?: string;
  calibrationDueDate?: string;
  notes?: string;
  tags?: string[];
  presets?: Record<string, HILInstrumentPreset>;
  defaultPreset?: string;
  metadata?: Record<string, unknown>;
}

/**
 * Input for creating a test sequence
 */
export interface CreateHILTestSequenceInput {
  projectId: string;
  name: string;
  description?: string;
  testType: HILTestType;
  sequenceConfig: HILSequenceConfig;
  passCriteria: HILPassCriteria;
  schematicId?: string;
  pcbLayoutId?: string;
  estimatedDurationMs?: number;
  timeoutMs?: number;
  priority?: number;
  tags?: string[];
  category?: string;
  isTemplate?: boolean;
  templateVariables?: Record<string, HILTemplateVariable>;
  parentTemplateId?: string;
  metadata?: Record<string, unknown>;
}

/**
 * Input for updating a test sequence
 */
export interface UpdateHILTestSequenceInput {
  name?: string;
  description?: string;
  sequenceConfig?: HILSequenceConfig;
  passCriteria?: HILPassCriteria;
  estimatedDurationMs?: number;
  timeoutMs?: number;
  priority?: number;
  tags?: string[];
  category?: string;
  metadata?: Record<string, unknown>;
}

/**
 * Input for starting a test run
 */
export interface StartHILTestRunInput {
  sequenceId: string;
  testConditions?: HILTestConditions;
  parameterOverrides?: Record<string, unknown>;
  baselineRunId?: string;
  tags?: string[];
}

/**
 * Input for creating a measurement
 */
export interface CreateHILMeasurementInput {
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
  isCritical?: boolean;
  timestampOffsetMs?: number;
  metadata?: Record<string, unknown>;
}

/**
 * Filters for querying instruments
 */
export interface HILInstrumentFilters {
  instrumentType?: HILInstrumentType;
  status?: HILInstrumentStatus;
  connectionType?: HILConnectionType;
  manufacturer?: string;
  tags?: string[];
}

/**
 * Filters for querying test sequences
 */
export interface HILTestSequenceFilters {
  testType?: HILTestType;
  isTemplate?: boolean;
  category?: string;
  tags?: string[];
  schematicId?: string;
  pcbLayoutId?: string;
}

/**
 * Filters for querying test runs
 */
export interface HILTestRunFilters {
  status?: HILTestRunStatus;
  result?: HILTestResult;
  sequenceId?: string;
  startedAfter?: string;
  startedBefore?: string;
}

// ============================================================================
// EXPORT ALL
// ============================================================================

export type {
  HILInstrumentType as InstrumentType,
  HILConnectionType as ConnectionType,
  HILInstrumentStatus as InstrumentStatus,
  HILTestType as TestType,
  HILTestRunStatus as TestRunStatus,
  HILTestResult as TestResult,
  HILCaptureType as CaptureType,
  HILMeasurementType as MeasurementType,
};
