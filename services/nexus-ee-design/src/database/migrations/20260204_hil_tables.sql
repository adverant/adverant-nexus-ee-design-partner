-- HIL (Hardware-in-the-Loop) Testing Tables Migration
-- Created: 2026-02-04
-- Description: Complete database schema for HIL testing integration

-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pgcrypto";

-- ============================================================================
-- ENUM TYPES
-- ============================================================================

-- Instrument types supported by the HIL system
CREATE TYPE hil_instrument_type AS ENUM (
    'logic_analyzer',
    'oscilloscope',
    'power_supply',
    'motor_emulator',
    'daq',
    'can_analyzer',
    'function_gen',
    'thermal_camera',
    'electronic_load'
);

-- Connection types for instruments
CREATE TYPE hil_connection_type AS ENUM (
    'usb',
    'ethernet',
    'gpib',
    'serial',
    'grpc',
    'modbus_tcp',
    'modbus_rtu'
);

-- Instrument connection status
CREATE TYPE hil_instrument_status AS ENUM (
    'connected',
    'disconnected',
    'error',
    'busy',
    'initializing'
);

-- Test types for FOC ESC and general HIL testing
CREATE TYPE hil_test_type AS ENUM (
    'foc_startup',
    'foc_steady_state',
    'foc_transient',
    'foc_speed_reversal',
    'pwm_analysis',
    'phase_current',
    'hall_sensor',
    'thermal_profile',
    'overcurrent_protection',
    'efficiency_sweep',
    'load_step',
    'no_load',
    'locked_rotor',
    'custom'
);

-- Test run status
CREATE TYPE hil_test_run_status AS ENUM (
    'pending',
    'queued',
    'running',
    'completed',
    'failed',
    'aborted',
    'timeout'
);

-- Test result
CREATE TYPE hil_test_result AS ENUM (
    'pass',
    'fail',
    'partial',
    'inconclusive'
);

-- Capture data types
CREATE TYPE hil_capture_type AS ENUM (
    'waveform',
    'logic_trace',
    'spectrum',
    'thermal_image',
    'measurement',
    'protocol_decode',
    'can_log'
);

-- Measurement types
CREATE TYPE hil_measurement_type AS ENUM (
    'rms_current',
    'peak_current',
    'avg_current',
    'rms_voltage',
    'peak_voltage',
    'avg_voltage',
    'frequency',
    'duty_cycle',
    'dead_time',
    'rise_time',
    'fall_time',
    'temperature',
    'efficiency',
    'thd',
    'phase_angle',
    'power',
    'power_factor',
    'speed',
    'torque',
    'position',
    'startup_time',
    'settling_time',
    'overshoot',
    'ripple',
    'custom'
);

-- ============================================================================
-- TABLE: hil_instruments
-- Description: Connected test instruments with connection params and capabilities
-- ============================================================================

CREATE TABLE hil_instruments (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    project_id UUID NOT NULL REFERENCES projects(id) ON DELETE CASCADE,

    -- Instrument identification
    name VARCHAR(255) NOT NULL,
    instrument_type hil_instrument_type NOT NULL,
    manufacturer VARCHAR(100) NOT NULL,
    model VARCHAR(100) NOT NULL,
    serial_number VARCHAR(100),
    firmware_version VARCHAR(50),

    -- Connection configuration
    connection_type hil_connection_type NOT NULL,
    connection_params JSONB NOT NULL DEFAULT '{}'::jsonb,
    -- Example connection_params:
    -- USB: {"serial_number": "DS5A244800001"}
    -- Ethernet: {"host": "192.168.1.50", "port": 5555}
    -- GPIB: {"address": 7, "board": 0}
    -- Serial: {"port": "/dev/ttyUSB0", "baud_rate": 115200}
    -- gRPC: {"host": "localhost", "port": 10430}

    -- Instrument capabilities
    capabilities JSONB NOT NULL DEFAULT '[]'::jsonb,
    -- Example capabilities:
    -- [{"name": "digital_channels", "type": "feature", "parameters": {"count": 16, "max_sample_rate": 500000000}}]
    -- [{"name": "spi", "type": "protocol"}]
    -- [{"name": "voltage_range", "type": "range", "parameters": {"min": 0, "max": 30, "unit": "V"}}]

    -- Status and health
    status hil_instrument_status NOT NULL DEFAULT 'disconnected',
    last_seen_at TIMESTAMP WITH TIME ZONE,
    last_error VARCHAR(1000),
    error_count INTEGER DEFAULT 0,

    -- Calibration tracking
    calibration_date DATE,
    calibration_due_date DATE,
    calibration_certificate VARCHAR(100),

    -- Configuration presets
    presets JSONB DEFAULT '{}'::jsonb,
    default_preset VARCHAR(50),

    -- Metadata
    notes TEXT,
    tags TEXT[] DEFAULT ARRAY[]::TEXT[],
    metadata JSONB DEFAULT '{}'::jsonb,

    -- Timestamps
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),

    -- Constraints
    CONSTRAINT unique_instrument_per_project UNIQUE (project_id, name)
);

-- Indexes for hil_instruments
CREATE INDEX idx_hil_instruments_project ON hil_instruments(project_id);
CREATE INDEX idx_hil_instruments_type ON hil_instruments(instrument_type);
CREATE INDEX idx_hil_instruments_status ON hil_instruments(status);
CREATE INDEX idx_hil_instruments_manufacturer ON hil_instruments(manufacturer);
CREATE INDEX idx_hil_instruments_tags ON hil_instruments USING GIN(tags);

-- ============================================================================
-- TABLE: hil_test_sequences
-- Description: Reusable test procedures with steps and pass criteria
-- ============================================================================

CREATE TABLE hil_test_sequences (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    project_id UUID NOT NULL REFERENCES projects(id) ON DELETE CASCADE,

    -- Link to design artifacts
    schematic_id UUID REFERENCES schematics(id) ON DELETE SET NULL,
    pcb_layout_id UUID REFERENCES pcb_layouts(id) ON DELETE SET NULL,

    -- Sequence identification
    name VARCHAR(255) NOT NULL,
    description TEXT,
    test_type hil_test_type NOT NULL,

    -- Sequence configuration
    sequence_config JSONB NOT NULL,
    -- Structure:
    -- {
    --   "steps": [
    --     {
    --       "id": "step-1",
    --       "name": "Configure Power Supply",
    --       "type": "configure",
    --       "instrumentId": "uuid",
    --       "parameters": {"channel": 1, "voltage": 48, "current_limit": 15},
    --       "timeout_ms": 5000,
    --       "retryOnFail": true,
    --       "continueOnFail": false
    --     },
    --     {
    --       "id": "step-2",
    --       "name": "Capture Phase Currents",
    --       "type": "measure",
    --       "instrumentId": "uuid",
    --       "parameters": {"channels": [1,2,3], "duration_ms": 1000, "sample_rate": 1000000},
    --       "expectedResults": [
    --         {"measurement": "rms_current", "operator": "lte", "value": 15, "unit": "A"}
    --       ]
    --     }
    --   ],
    --   "globalParameters": {"motor_pole_pairs": 7, "rated_speed": 3000},
    --   "instrumentRequirements": [
    --     {"instrumentType": "oscilloscope", "capabilities": ["4_channel"], "optional": false},
    --     {"instrumentType": "power_supply", "capabilities": ["programmable"], "optional": false}
    --   ]
    -- }

    -- Pass/fail criteria
    pass_criteria JSONB NOT NULL,
    -- Structure:
    -- {
    --   "minPassPercentage": 90,
    --   "criticalMeasurements": ["startup_time", "overcurrent"],
    --   "failFast": true,
    --   "allowedWarnings": 2
    -- }

    -- Execution settings
    estimated_duration_ms INTEGER,
    timeout_ms INTEGER DEFAULT 600000,
    priority INTEGER DEFAULT 5 CHECK (priority >= 1 AND priority <= 10),

    -- Organization
    tags TEXT[] DEFAULT ARRAY[]::TEXT[],
    category VARCHAR(50),

    -- Versioning
    version INTEGER DEFAULT 1,
    parent_version_id UUID REFERENCES hil_test_sequences(id) ON DELETE SET NULL,

    -- Template support
    is_template BOOLEAN DEFAULT FALSE,
    template_variables JSONB DEFAULT '{}'::jsonb,
    parent_template_id UUID REFERENCES hil_test_sequences(id) ON DELETE SET NULL,

    -- Audit
    created_by VARCHAR(255),
    last_modified_by VARCHAR(255),

    -- Metadata
    metadata JSONB DEFAULT '{}'::jsonb,

    -- Timestamps
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),

    -- Constraints
    CONSTRAINT unique_sequence_name_per_project UNIQUE (project_id, name, version)
);

-- Indexes for hil_test_sequences
CREATE INDEX idx_hil_test_sequences_project ON hil_test_sequences(project_id);
CREATE INDEX idx_hil_test_sequences_type ON hil_test_sequences(test_type);
CREATE INDEX idx_hil_test_sequences_template ON hil_test_sequences(is_template) WHERE is_template = TRUE;
CREATE INDEX idx_hil_test_sequences_schematic ON hil_test_sequences(schematic_id);
CREATE INDEX idx_hil_test_sequences_pcb ON hil_test_sequences(pcb_layout_id);
CREATE INDEX idx_hil_test_sequences_tags ON hil_test_sequences USING GIN(tags);
CREATE INDEX idx_hil_test_sequences_category ON hil_test_sequences(category);

-- ============================================================================
-- TABLE: hil_test_runs
-- Description: Individual test executions with progress and results
-- ============================================================================

CREATE TABLE hil_test_runs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    sequence_id UUID NOT NULL REFERENCES hil_test_sequences(id) ON DELETE CASCADE,
    project_id UUID NOT NULL REFERENCES projects(id) ON DELETE CASCADE,

    -- Run identification
    name VARCHAR(255) NOT NULL,
    run_number INTEGER NOT NULL DEFAULT 1,

    -- Execution status
    status hil_test_run_status NOT NULL DEFAULT 'pending',
    result hil_test_result,

    -- Progress tracking
    progress_percentage INTEGER DEFAULT 0 CHECK (progress_percentage >= 0 AND progress_percentage <= 100),
    current_step VARCHAR(255),
    current_step_index INTEGER DEFAULT 0,
    total_steps INTEGER,

    -- Timing
    queued_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    started_at TIMESTAMP WITH TIME ZONE,
    completed_at TIMESTAMP WITH TIME ZONE,
    duration_ms INTEGER,

    -- Error handling
    error_message TEXT,
    error_details JSONB,
    error_step_id VARCHAR(100),

    -- Test conditions at start
    test_conditions JSONB DEFAULT '{}'::jsonb,
    -- Structure:
    -- {
    --   "ambientTemperature": 25.5,
    --   "humidity": 45,
    --   "supplyVoltage": 48.0,
    --   "notes": "Initial test run"
    -- }

    -- Instrument snapshot at test start
    instrument_snapshot JSONB DEFAULT '{}'::jsonb,

    -- Results summary
    summary JSONB DEFAULT '{}'::jsonb,
    -- Structure:
    -- {
    --   "totalMeasurements": 20,
    --   "passedMeasurements": 18,
    --   "failedMeasurements": 0,
    --   "warningMeasurements": 2,
    --   "criticalFailures": [],
    --   "keyMetrics": {
    --     "startupTime": 423,
    --     "efficiency": 89.2,
    --     "maxCurrent": 14.8
    --   }
    -- }

    -- Worker assignment
    worker_id VARCHAR(100),
    worker_host VARCHAR(100),
    job_id VARCHAR(100),

    -- Retry tracking
    retry_count INTEGER DEFAULT 0,
    max_retries INTEGER DEFAULT 3,
    last_retry_at TIMESTAMP WITH TIME ZONE,

    -- Comparison
    baseline_run_id UUID REFERENCES hil_test_runs(id) ON DELETE SET NULL,
    comparison_results JSONB,

    -- Audit
    started_by VARCHAR(255),
    aborted_by VARCHAR(255),
    abort_reason TEXT,

    -- Metadata
    tags TEXT[] DEFAULT ARRAY[]::TEXT[],
    metadata JSONB DEFAULT '{}'::jsonb,

    -- Timestamps
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Indexes for hil_test_runs
CREATE INDEX idx_hil_test_runs_sequence ON hil_test_runs(sequence_id);
CREATE INDEX idx_hil_test_runs_project ON hil_test_runs(project_id);
CREATE INDEX idx_hil_test_runs_status ON hil_test_runs(status);
CREATE INDEX idx_hil_test_runs_result ON hil_test_runs(result);
CREATE INDEX idx_hil_test_runs_started_at ON hil_test_runs(started_at DESC);
CREATE INDEX idx_hil_test_runs_worker ON hil_test_runs(worker_id);
CREATE INDEX idx_hil_test_runs_pending ON hil_test_runs(status, priority) WHERE status IN ('pending', 'queued');

-- Auto-increment run_number per sequence
CREATE OR REPLACE FUNCTION set_test_run_number()
RETURNS TRIGGER AS $$
BEGIN
    NEW.run_number := COALESCE(
        (SELECT MAX(run_number) + 1 FROM hil_test_runs WHERE sequence_id = NEW.sequence_id),
        1
    );
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trigger_set_test_run_number
    BEFORE INSERT ON hil_test_runs
    FOR EACH ROW
    EXECUTE FUNCTION set_test_run_number();

-- ============================================================================
-- TABLE: hil_captured_data
-- Description: Waveforms, logic traces, measurements (file paths + inline JSON)
-- ============================================================================

CREATE TABLE hil_captured_data (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    test_run_id UUID NOT NULL REFERENCES hil_test_runs(id) ON DELETE CASCADE,
    instrument_id UUID REFERENCES hil_instruments(id) ON DELETE SET NULL,

    -- Capture identification
    name VARCHAR(255),
    capture_type hil_capture_type NOT NULL,
    step_id VARCHAR(100),

    -- Channel configuration
    channel_config JSONB NOT NULL DEFAULT '[]'::jsonb,
    -- Structure:
    -- [
    --   {"name": "CH1", "label": "Phase A", "scale": 5.0, "offset": 0, "unit": "V", "coupling": "dc", "probe_attenuation": 10},
    --   {"name": "CH2", "label": "Phase B", "scale": 5.0, "offset": 0, "unit": "V", "coupling": "dc", "probe_attenuation": 10}
    -- ]

    -- Sampling parameters
    sample_rate_hz DOUBLE PRECISION,
    sample_count BIGINT,
    duration_ms DOUBLE PRECISION,

    -- Trigger configuration
    trigger_config JSONB,
    -- Structure:
    -- {"source": "CH1", "level": 2.5, "edge": "rising", "position": 0.5, "mode": "normal"}

    -- Data storage
    data_format VARCHAR(20) NOT NULL DEFAULT 'binary',
    -- Formats: 'binary', 'csv', 'json', 'vcd', 'salae', 'sigrok'

    data_path VARCHAR(500),
    -- File path for large datasets: /plugins/ee-design-plugin/artifacts/hil-captures/{project_id}/{test_run_id}/{id}.{ext}

    data_inline JSONB,
    -- For small datasets stored inline (< 1MB)

    data_size_bytes BIGINT,
    data_checksum VARCHAR(64),
    compression VARCHAR(20),

    -- Analysis results
    analysis_results JSONB DEFAULT '{}'::jsonb,
    -- Structure:
    -- {
    --   "fft": {"fundamental": 200, "harmonics": [400, 600, 800], "thd": 3.2},
    --   "measurements": {"rms": [12.4, 12.3, 12.5], "peak": [14.8, 14.6, 14.9]},
    --   "statistics": {"min": -14.3, "max": 14.8, "mean": 0.01, "stddev": 8.7}
    -- }

    -- Annotations
    annotations JSONB DEFAULT '[]'::jsonb,
    -- Structure:
    -- [{"timestamp_ms": 23.5, "text": "Motor started", "color": "#22c55e"}]

    -- Timing
    captured_at TIMESTAMP WITH TIME ZONE NOT NULL,
    processing_completed_at TIMESTAMP WITH TIME ZONE,

    -- Metadata
    metadata JSONB DEFAULT '{}'::jsonb,

    -- Timestamps
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Indexes for hil_captured_data
CREATE INDEX idx_hil_captured_data_test_run ON hil_captured_data(test_run_id);
CREATE INDEX idx_hil_captured_data_instrument ON hil_captured_data(instrument_id);
CREATE INDEX idx_hil_captured_data_type ON hil_captured_data(capture_type);
CREATE INDEX idx_hil_captured_data_captured_at ON hil_captured_data(captured_at DESC);
CREATE INDEX idx_hil_captured_data_step ON hil_captured_data(step_id);

-- ============================================================================
-- TABLE: hil_measurements
-- Description: Individual measurement points from test runs
-- ============================================================================

CREATE TABLE hil_measurements (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    test_run_id UUID NOT NULL REFERENCES hil_test_runs(id) ON DELETE CASCADE,
    captured_data_id UUID REFERENCES hil_captured_data(id) ON DELETE SET NULL,
    step_id VARCHAR(100),

    -- Measurement identification
    measurement_type hil_measurement_type NOT NULL,
    measurement_name VARCHAR(100),
    channel VARCHAR(50),

    -- Value
    value DOUBLE PRECISION NOT NULL,
    unit VARCHAR(20) NOT NULL,

    -- Limits and validation
    min_limit DOUBLE PRECISION,
    max_limit DOUBLE PRECISION,
    nominal_value DOUBLE PRECISION,
    tolerance_percent DOUBLE PRECISION,
    tolerance_absolute DOUBLE PRECISION,

    -- Result
    passed BOOLEAN,
    is_critical BOOLEAN DEFAULT FALSE,
    is_warning BOOLEAN DEFAULT FALSE,
    failure_reason VARCHAR(500),

    -- Timing
    timestamp_offset_ms DOUBLE PRECISION,
    measured_at TIMESTAMP WITH TIME ZONE NOT NULL,

    -- Statistics (for repeated measurements)
    sample_count INTEGER DEFAULT 1,
    mean_value DOUBLE PRECISION,
    std_deviation DOUBLE PRECISION,
    min_observed DOUBLE PRECISION,
    max_observed DOUBLE PRECISION,

    -- Metadata
    metadata JSONB DEFAULT '{}'::jsonb,

    -- Timestamps
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Indexes for hil_measurements
CREATE INDEX idx_hil_measurements_test_run ON hil_measurements(test_run_id);
CREATE INDEX idx_hil_measurements_type ON hil_measurements(measurement_type);
CREATE INDEX idx_hil_measurements_passed ON hil_measurements(passed);
CREATE INDEX idx_hil_measurements_critical ON hil_measurements(is_critical) WHERE is_critical = TRUE;
CREATE INDEX idx_hil_measurements_captured_data ON hil_measurements(captured_data_id);
CREATE INDEX idx_hil_measurements_step ON hil_measurements(step_id);
CREATE INDEX idx_hil_measurements_channel ON hil_measurements(channel);

-- ============================================================================
-- HELPER FUNCTIONS
-- ============================================================================

-- Function to update updated_at timestamp
CREATE OR REPLACE FUNCTION update_hil_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Apply updated_at triggers
CREATE TRIGGER trigger_hil_instruments_updated_at
    BEFORE UPDATE ON hil_instruments
    FOR EACH ROW
    EXECUTE FUNCTION update_hil_updated_at();

CREATE TRIGGER trigger_hil_test_sequences_updated_at
    BEFORE UPDATE ON hil_test_sequences
    FOR EACH ROW
    EXECUTE FUNCTION update_hil_updated_at();

CREATE TRIGGER trigger_hil_test_runs_updated_at
    BEFORE UPDATE ON hil_test_runs
    FOR EACH ROW
    EXECUTE FUNCTION update_hil_updated_at();

-- Function to get test run statistics
CREATE OR REPLACE FUNCTION get_hil_test_run_stats(p_test_run_id UUID)
RETURNS JSONB AS $$
DECLARE
    result JSONB;
BEGIN
    SELECT jsonb_build_object(
        'totalMeasurements', COUNT(*),
        'passedMeasurements', COUNT(*) FILTER (WHERE passed = TRUE),
        'failedMeasurements', COUNT(*) FILTER (WHERE passed = FALSE),
        'warningMeasurements', COUNT(*) FILTER (WHERE is_warning = TRUE),
        'criticalFailures', COALESCE(
            jsonb_agg(measurement_name) FILTER (WHERE is_critical = TRUE AND passed = FALSE),
            '[]'::jsonb
        )
    ) INTO result
    FROM hil_measurements
    WHERE test_run_id = p_test_run_id;

    RETURN result;
END;
$$ LANGUAGE plpgsql;

-- ============================================================================
-- SEED DATA: FOC ESC Test Templates
-- ============================================================================

-- Note: This will be inserted after projects table has data
-- Templates are created with is_template = TRUE and project_id = NULL initially
-- Then cloned to specific projects when used

-- Insert statement for templates will be in a separate seed file
-- to avoid foreign key issues with projects table

-- ============================================================================
-- GRANTS (adjust schema/role names as needed)
-- ============================================================================

-- Grant permissions to application role
-- GRANT ALL ON ALL TABLES IN SCHEMA public TO nexus_ee_design;
-- GRANT ALL ON ALL SEQUENCES IN SCHEMA public TO nexus_ee_design;
-- GRANT EXECUTE ON ALL FUNCTIONS IN SCHEMA public TO nexus_ee_design;

-- ============================================================================
-- COMMENTS
-- ============================================================================

COMMENT ON TABLE hil_instruments IS 'Connected test instruments for HIL testing (oscilloscopes, logic analyzers, power supplies, etc.)';
COMMENT ON TABLE hil_test_sequences IS 'Reusable test procedures with steps, expected results, and pass criteria';
COMMENT ON TABLE hil_test_runs IS 'Individual executions of test sequences with progress tracking and results';
COMMENT ON TABLE hil_captured_data IS 'Waveforms, logic traces, and other captured data from test runs';
COMMENT ON TABLE hil_measurements IS 'Individual measurement points extracted from test runs with pass/fail status';

COMMENT ON COLUMN hil_instruments.connection_params IS 'Connection parameters as JSONB: host, port, serial_number, etc.';
COMMENT ON COLUMN hil_instruments.capabilities IS 'Instrument capabilities: channels, sample rates, protocols supported';
COMMENT ON COLUMN hil_test_sequences.sequence_config IS 'Test steps configuration with instruments, parameters, and expected results';
COMMENT ON COLUMN hil_test_sequences.pass_criteria IS 'Criteria for determining test pass/fail status';
COMMENT ON COLUMN hil_test_runs.summary IS 'Aggregated results summary with key metrics';
COMMENT ON COLUMN hil_captured_data.data_path IS 'File path for large datasets, NULL if stored inline';
COMMENT ON COLUMN hil_captured_data.data_inline IS 'Inline data storage for small datasets (< 1MB)';
COMMENT ON COLUMN hil_measurements.is_critical IS 'If TRUE, failure of this measurement fails the entire test';
