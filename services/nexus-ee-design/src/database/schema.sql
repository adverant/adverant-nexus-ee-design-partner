-- ============================================================================
-- EE Design Partner - PostgreSQL Database Schema
-- ============================================================================
-- Complete database schema for the EE Design Partner backend service.
-- Supports the full 10-phase hardware/software development pipeline.
-- ============================================================================

-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS "pgcrypto";

-- ============================================================================
-- PROJECTS TABLE
-- ============================================================================
-- Core table for EE Design projects, supporting all 10 phases of development.

CREATE TABLE IF NOT EXISTS projects (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name VARCHAR(255) NOT NULL,
    description TEXT,
    repository_url VARCHAR(512),

    -- Project lifecycle
    project_type VARCHAR(50) NOT NULL DEFAULT 'hardware',
    phase VARCHAR(50) NOT NULL DEFAULT 'ideation',
    status VARCHAR(50) NOT NULL DEFAULT 'draft',

    -- Configuration
    phase_config JSONB NOT NULL DEFAULT '{}',
    metadata JSONB NOT NULL DEFAULT '{}',

    -- Ownership
    owner_id VARCHAR(255) NOT NULL,
    organization_id VARCHAR(255),
    collaborators JSONB NOT NULL DEFAULT '[]',

    -- Timestamps
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    archived_at TIMESTAMPTZ,

    -- Constraints
    CONSTRAINT projects_phase_check CHECK (phase IN (
        'ideation', 'architecture', 'schematic', 'simulation',
        'pcb_layout', 'manufacturing', 'firmware', 'testing',
        'production', 'field_support'
    )),
    CONSTRAINT projects_status_check CHECK (status IN (
        'draft', 'in_progress', 'review', 'approved',
        'completed', 'on_hold', 'cancelled'
    )),
    CONSTRAINT projects_type_check CHECK (project_type IN (
        'power_electronics', 'analog_circuit', 'digital_logic', 'mixed_signal',
        'rf_design', 'iot_device', 'passive_board', 'firmware_only', 'custom'
    ))
);

-- Indexes for projects
CREATE INDEX IF NOT EXISTS idx_projects_owner_id ON projects(owner_id);
CREATE INDEX IF NOT EXISTS idx_projects_organization_id ON projects(organization_id);
CREATE INDEX IF NOT EXISTS idx_projects_phase ON projects(phase);
CREATE INDEX IF NOT EXISTS idx_projects_status ON projects(status);
CREATE INDEX IF NOT EXISTS idx_projects_created_at ON projects(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_projects_metadata ON projects USING GIN(metadata);

-- ============================================================================
-- SCHEMATICS TABLE
-- ============================================================================
-- Stores schematic designs including KiCad source, netlists, and BOMs.

CREATE TABLE IF NOT EXISTS schematics (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    project_id UUID NOT NULL REFERENCES projects(id) ON DELETE CASCADE,

    -- Identification
    name VARCHAR(255) NOT NULL,
    version INTEGER NOT NULL DEFAULT 1,

    -- Schematic content
    kicad_sch TEXT,
    format VARCHAR(50) NOT NULL DEFAULT 'kicad_sch',
    file_path VARCHAR(512),

    -- Derived data
    netlist JSONB,
    bom JSONB,
    sheets JSONB NOT NULL DEFAULT '[]',
    components JSONB NOT NULL DEFAULT '[]',
    nets JSONB NOT NULL DEFAULT '[]',

    -- Validation
    validation_results JSONB,
    erc_violations INTEGER DEFAULT 0,
    erc_warnings INTEGER DEFAULT 0,

    -- Status
    status VARCHAR(50) NOT NULL DEFAULT 'draft',
    locked BOOLEAN NOT NULL DEFAULT FALSE,
    locked_by VARCHAR(255),
    locked_at TIMESTAMPTZ,

    -- Timestamps
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    -- Constraints
    CONSTRAINT schematics_format_check CHECK (format IN (
        'kicad_sch', 'eagle', 'altium', 'orcad'
    )),
    CONSTRAINT schematics_status_check CHECK (status IN (
        'draft', 'in_review', 'approved', 'released', 'obsolete'
    )),
    CONSTRAINT schematics_version_positive CHECK (version > 0)
);

-- Indexes for schematics
CREATE INDEX IF NOT EXISTS idx_schematics_project_id ON schematics(project_id);
CREATE INDEX IF NOT EXISTS idx_schematics_status ON schematics(status);
CREATE INDEX IF NOT EXISTS idx_schematics_version ON schematics(project_id, version DESC);
CREATE INDEX IF NOT EXISTS idx_schematics_components ON schematics USING GIN(components);
CREATE INDEX IF NOT EXISTS idx_schematics_nets ON schematics USING GIN(nets);

-- ============================================================================
-- PCB_LAYOUTS TABLE
-- ============================================================================
-- Stores PCB layout designs including KiCad PCB files and DRC results.

CREATE TABLE IF NOT EXISTS pcb_layouts (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    project_id UUID NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    schematic_id UUID REFERENCES schematics(id) ON DELETE SET NULL,

    -- Identification
    name VARCHAR(255) NOT NULL,
    version INTEGER NOT NULL DEFAULT 1,

    -- PCB content
    kicad_pcb TEXT,
    file_path VARCHAR(512),

    -- Board specifications
    board_outline JSONB,
    stackup JSONB,
    layer_count INTEGER NOT NULL DEFAULT 2,

    -- Layout data
    components JSONB NOT NULL DEFAULT '[]',
    traces JSONB NOT NULL DEFAULT '[]',
    vias JSONB NOT NULL DEFAULT '[]',
    zones JSONB NOT NULL DEFAULT '[]',

    -- Validation results
    drc_results JSONB,
    drc_violations INTEGER DEFAULT 0,
    drc_warnings INTEGER DEFAULT 0,

    -- MAPOS optimization
    mapos_score DECIMAL(5,2),
    mapos_iterations INTEGER DEFAULT 0,
    mapos_config JSONB,
    winning_agent VARCHAR(50),

    -- Scoring
    overall_score DECIMAL(5,2),
    thermal_score DECIMAL(5,2),
    emi_score DECIMAL(5,2),
    dfm_score DECIMAL(5,2),
    si_score DECIMAL(5,2),

    -- Status
    status VARCHAR(50) NOT NULL DEFAULT 'draft',
    locked BOOLEAN NOT NULL DEFAULT FALSE,
    locked_by VARCHAR(255),
    locked_at TIMESTAMPTZ,

    -- Timestamps
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    -- Constraints
    CONSTRAINT pcb_layouts_status_check CHECK (status IN (
        'draft', 'in_progress', 'optimizing', 'in_review',
        'approved', 'released', 'obsolete'
    )),
    CONSTRAINT pcb_layouts_version_positive CHECK (version > 0),
    CONSTRAINT pcb_layouts_layer_count_check CHECK (layer_count >= 1 AND layer_count <= 64)
);

-- Indexes for pcb_layouts
CREATE INDEX IF NOT EXISTS idx_pcb_layouts_project_id ON pcb_layouts(project_id);
CREATE INDEX IF NOT EXISTS idx_pcb_layouts_schematic_id ON pcb_layouts(schematic_id);
CREATE INDEX IF NOT EXISTS idx_pcb_layouts_status ON pcb_layouts(status);
CREATE INDEX IF NOT EXISTS idx_pcb_layouts_version ON pcb_layouts(project_id, version DESC);
CREATE INDEX IF NOT EXISTS idx_pcb_layouts_mapos_score ON pcb_layouts(mapos_score DESC);
CREATE INDEX IF NOT EXISTS idx_pcb_layouts_overall_score ON pcb_layouts(overall_score DESC);
CREATE INDEX IF NOT EXISTS idx_pcb_layouts_drc_results ON pcb_layouts USING GIN(drc_results);

-- ============================================================================
-- SIMULATIONS TABLE
-- ============================================================================
-- Tracks all simulation jobs and their results (SPICE, thermal, SI, RF, EMC).

CREATE TABLE IF NOT EXISTS simulations (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    project_id UUID NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    schematic_id UUID REFERENCES schematics(id) ON DELETE SET NULL,
    pcb_layout_id UUID REFERENCES pcb_layouts(id) ON DELETE SET NULL,

    -- Identification
    name VARCHAR(255) NOT NULL,
    simulation_type VARCHAR(50) NOT NULL,

    -- Execution
    status VARCHAR(50) NOT NULL DEFAULT 'pending',
    priority INTEGER NOT NULL DEFAULT 5,

    -- Configuration
    config JSONB NOT NULL DEFAULT '{}',
    test_bench TEXT,
    parameters JSONB NOT NULL DEFAULT '{}',

    -- Results
    results JSONB,
    waveforms JSONB,
    images JSONB,
    metrics JSONB,

    -- Scoring
    passed BOOLEAN,
    score DECIMAL(5,2),

    -- Error handling
    error_message TEXT,
    error_details JSONB,
    retry_count INTEGER NOT NULL DEFAULT 0,
    max_retries INTEGER NOT NULL DEFAULT 3,

    -- Timing
    started_at TIMESTAMPTZ,
    completed_at TIMESTAMPTZ,
    duration_ms INTEGER,
    timeout_ms INTEGER DEFAULT 3600000,

    -- Worker info
    worker_id VARCHAR(255),
    worker_host VARCHAR(255),

    -- Timestamps
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    -- Constraints
    CONSTRAINT simulations_type_check CHECK (simulation_type IN (
        'spice_dc', 'spice_ac', 'spice_transient', 'spice_noise', 'spice_monte_carlo',
        'thermal_steady_state', 'thermal_transient', 'thermal_cfd',
        'signal_integrity', 'power_integrity',
        'rf_sparameters', 'rf_field_pattern',
        'emc_radiated', 'emc_conducted',
        'stress_thermal_cycling', 'stress_vibration', 'reliability_mtbf'
    )),
    CONSTRAINT simulations_status_check CHECK (status IN (
        'pending', 'queued', 'running', 'completed', 'failed', 'cancelled', 'timeout'
    )),
    CONSTRAINT simulations_priority_check CHECK (priority >= 1 AND priority <= 10)
);

-- Indexes for simulations
CREATE INDEX IF NOT EXISTS idx_simulations_project_id ON simulations(project_id);
CREATE INDEX IF NOT EXISTS idx_simulations_schematic_id ON simulations(schematic_id);
CREATE INDEX IF NOT EXISTS idx_simulations_pcb_layout_id ON simulations(pcb_layout_id);
CREATE INDEX IF NOT EXISTS idx_simulations_type ON simulations(simulation_type);
CREATE INDEX IF NOT EXISTS idx_simulations_status ON simulations(status);
CREATE INDEX IF NOT EXISTS idx_simulations_status_priority ON simulations(status, priority DESC) WHERE status IN ('pending', 'queued');
CREATE INDEX IF NOT EXISTS idx_simulations_created_at ON simulations(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_simulations_results ON simulations USING GIN(results);

-- ============================================================================
-- FIRMWARE_PROJECTS TABLE
-- ============================================================================
-- Stores firmware projects including MCU configuration, RTOS, and HAL settings.

CREATE TABLE IF NOT EXISTS firmware_projects (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    project_id UUID NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    pcb_layout_id UUID REFERENCES pcb_layouts(id) ON DELETE SET NULL,

    -- Identification
    name VARCHAR(255) NOT NULL,
    version VARCHAR(50) NOT NULL DEFAULT '0.1.0',

    -- MCU Configuration
    target_mcu JSONB NOT NULL,
    mcu_family VARCHAR(50) NOT NULL,
    mcu_part VARCHAR(100) NOT NULL,

    -- RTOS Configuration
    rtos_config JSONB,
    rtos_type VARCHAR(50),

    -- HAL Configuration
    hal_config JSONB NOT NULL DEFAULT '{}',
    peripheral_configs JSONB NOT NULL DEFAULT '[]',
    pin_mappings JSONB NOT NULL DEFAULT '[]',

    -- Drivers
    drivers JSONB NOT NULL DEFAULT '[]',

    -- Tasks
    tasks JSONB NOT NULL DEFAULT '[]',

    -- Build Configuration
    build_config JSONB NOT NULL DEFAULT '{}',
    toolchain VARCHAR(50) NOT NULL DEFAULT 'gcc-arm',
    build_system VARCHAR(50) NOT NULL DEFAULT 'cmake',

    -- Generated Files
    generated_files JSONB NOT NULL DEFAULT '[]',
    source_tree_path VARCHAR(512),

    -- Status
    status VARCHAR(50) NOT NULL DEFAULT 'draft',
    build_status VARCHAR(50) DEFAULT 'not_built',
    last_build_at TIMESTAMPTZ,
    last_build_output TEXT,

    -- Timestamps
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    -- Constraints
    CONSTRAINT firmware_mcu_family_check CHECK (mcu_family IN (
        'stm32', 'esp32', 'ti_tms320', 'infineon_aurix',
        'nordic_nrf', 'rpi_pico', 'nxp_imxrt'
    )),
    CONSTRAINT firmware_rtos_type_check CHECK (rtos_type IS NULL OR rtos_type IN (
        'freertos', 'zephyr', 'tirtos', 'autosar', 'bare_metal'
    )),
    CONSTRAINT firmware_status_check CHECK (status IN (
        'draft', 'generating', 'generated', 'building', 'built',
        'testing', 'tested', 'released', 'obsolete'
    )),
    CONSTRAINT firmware_build_status_check CHECK (build_status IN (
        'not_built', 'building', 'success', 'failed', 'warnings'
    ))
);

-- Indexes for firmware_projects
CREATE INDEX IF NOT EXISTS idx_firmware_project_id ON firmware_projects(project_id);
CREATE INDEX IF NOT EXISTS idx_firmware_pcb_layout_id ON firmware_projects(pcb_layout_id);
CREATE INDEX IF NOT EXISTS idx_firmware_mcu_family ON firmware_projects(mcu_family);
CREATE INDEX IF NOT EXISTS idx_firmware_status ON firmware_projects(status);
CREATE INDEX IF NOT EXISTS idx_firmware_target_mcu ON firmware_projects USING GIN(target_mcu);

-- ============================================================================
-- JOB_HISTORY TABLE
-- ============================================================================
-- Provides visibility into BullMQ job execution for debugging and analytics.

CREATE TABLE IF NOT EXISTS job_history (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),

    -- Job identification
    job_id VARCHAR(255) NOT NULL,
    queue_name VARCHAR(100) NOT NULL,
    job_name VARCHAR(255) NOT NULL,

    -- Related entities
    project_id UUID REFERENCES projects(id) ON DELETE SET NULL,
    entity_type VARCHAR(50),
    entity_id UUID,

    -- Job data
    job_data JSONB NOT NULL DEFAULT '{}',
    job_options JSONB,

    -- Execution
    status VARCHAR(50) NOT NULL DEFAULT 'pending',
    priority INTEGER NOT NULL DEFAULT 5,
    attempts INTEGER NOT NULL DEFAULT 0,
    max_attempts INTEGER NOT NULL DEFAULT 3,

    -- Progress
    progress INTEGER DEFAULT 0,
    progress_data JSONB,

    -- Results
    result JSONB,
    return_value JSONB,

    -- Error handling
    failed_reason TEXT,
    error_stack TEXT,
    error_code VARCHAR(100),

    -- Worker info
    worker_id VARCHAR(255),
    worker_host VARCHAR(255),
    processed_on VARCHAR(255),

    -- Timing
    delay_ms INTEGER DEFAULT 0,
    timeout_ms INTEGER,
    started_at TIMESTAMPTZ,
    completed_at TIMESTAMPTZ,
    failed_at TIMESTAMPTZ,
    duration_ms INTEGER,
    wait_time_ms INTEGER,

    -- Timestamps
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    -- Constraints
    CONSTRAINT job_history_status_check CHECK (status IN (
        'pending', 'waiting', 'delayed', 'active', 'completed',
        'failed', 'cancelled', 'stalled'
    )),
    CONSTRAINT job_history_priority_check CHECK (priority >= 1 AND priority <= 10),
    CONSTRAINT job_history_progress_check CHECK (progress >= 0 AND progress <= 100)
);

-- Indexes for job_history
CREATE INDEX IF NOT EXISTS idx_job_history_job_id ON job_history(job_id);
CREATE INDEX IF NOT EXISTS idx_job_history_queue_name ON job_history(queue_name);
CREATE INDEX IF NOT EXISTS idx_job_history_project_id ON job_history(project_id);
CREATE INDEX IF NOT EXISTS idx_job_history_status ON job_history(status);
CREATE INDEX IF NOT EXISTS idx_job_history_entity ON job_history(entity_type, entity_id);
CREATE INDEX IF NOT EXISTS idx_job_history_created_at ON job_history(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_job_history_status_queue ON job_history(queue_name, status);
CREATE INDEX IF NOT EXISTS idx_job_history_job_data ON job_history USING GIN(job_data);

-- ============================================================================
-- MANUFACTURING_ORDERS TABLE
-- ============================================================================
-- Tracks PCB manufacturing orders and vendor quotes.

CREATE TABLE IF NOT EXISTS manufacturing_orders (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    project_id UUID NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    pcb_layout_id UUID NOT NULL REFERENCES pcb_layouts(id) ON DELETE CASCADE,

    -- Vendor info
    vendor VARCHAR(50) NOT NULL,
    vendor_order_id VARCHAR(255),

    -- Order details
    quantity INTEGER NOT NULL,
    pcb_options JSONB NOT NULL,

    -- Files
    gerber_files JSONB NOT NULL DEFAULT '[]',
    bom_file VARCHAR(512),
    pick_and_place_file VARCHAR(512),

    -- Quote
    quote JSONB,
    quoted_at TIMESTAMPTZ,
    quote_valid_until TIMESTAMPTZ,

    -- Status
    status VARCHAR(50) NOT NULL DEFAULT 'draft',
    tracking_number VARCHAR(255),
    shipped_at TIMESTAMPTZ,
    delivered_at TIMESTAMPTZ,

    -- Timestamps
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    ordered_at TIMESTAMPTZ,

    -- Constraints
    CONSTRAINT manufacturing_vendor_check CHECK (vendor IN (
        'pcbway', 'jlcpcb', 'oshpark', 'eurocircuits', 'advanced_circuits'
    )),
    CONSTRAINT manufacturing_status_check CHECK (status IN (
        'draft', 'quoted', 'ordered', 'in_production',
        'shipped', 'delivered', 'cancelled'
    )),
    CONSTRAINT manufacturing_quantity_check CHECK (quantity > 0)
);

-- Indexes for manufacturing_orders
CREATE INDEX IF NOT EXISTS idx_manufacturing_project_id ON manufacturing_orders(project_id);
CREATE INDEX IF NOT EXISTS idx_manufacturing_pcb_layout_id ON manufacturing_orders(pcb_layout_id);
CREATE INDEX IF NOT EXISTS idx_manufacturing_vendor ON manufacturing_orders(vendor);
CREATE INDEX IF NOT EXISTS idx_manufacturing_status ON manufacturing_orders(status);
CREATE INDEX IF NOT EXISTS idx_manufacturing_created_at ON manufacturing_orders(created_at DESC);

-- ============================================================================
-- VALIDATION_RESULTS TABLE
-- ============================================================================
-- Stores detailed validation results from multi-LLM and domain validators.

CREATE TABLE IF NOT EXISTS validation_results (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    project_id UUID NOT NULL REFERENCES projects(id) ON DELETE CASCADE,

    -- Target artifact
    artifact_type VARCHAR(50) NOT NULL,
    artifact_id UUID NOT NULL,

    -- Validation info
    validation_type VARCHAR(50) NOT NULL,
    validator_name VARCHAR(100) NOT NULL,

    -- Results
    passed BOOLEAN NOT NULL,
    score DECIMAL(5,2) NOT NULL,
    confidence DECIMAL(5,4),

    -- Details
    violations JSONB NOT NULL DEFAULT '[]',
    warnings JSONB NOT NULL DEFAULT '[]',
    recommendations JSONB NOT NULL DEFAULT '[]',
    metrics JSONB NOT NULL DEFAULT '{}',

    -- For multi-LLM validation
    llm_model VARCHAR(100),
    llm_reasoning TEXT,
    consensus_data JSONB,

    -- Timing
    duration_ms INTEGER,

    -- Timestamps
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    -- Constraints
    CONSTRAINT validation_artifact_type_check CHECK (artifact_type IN (
        'schematic', 'pcb_layout', 'firmware', 'simulation'
    )),
    CONSTRAINT validation_type_check CHECK (validation_type IN (
        'drc', 'erc', 'ipc_2221', 'signal_integrity', 'thermal',
        'dfm', 'best_practices', 'automated_testing', 'multi_llm'
    ))
);

-- Indexes for validation_results
CREATE INDEX IF NOT EXISTS idx_validation_project_id ON validation_results(project_id);
CREATE INDEX IF NOT EXISTS idx_validation_artifact ON validation_results(artifact_type, artifact_id);
CREATE INDEX IF NOT EXISTS idx_validation_type ON validation_results(validation_type);
CREATE INDEX IF NOT EXISTS idx_validation_passed ON validation_results(passed);
CREATE INDEX IF NOT EXISTS idx_validation_created_at ON validation_results(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_validation_violations ON validation_results USING GIN(violations);

-- ============================================================================
-- MAPOS_ITERATIONS TABLE
-- ============================================================================
-- Tracks individual MAPOS optimization iterations for analysis.

CREATE TABLE IF NOT EXISTS mapos_iterations (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    pcb_layout_id UUID NOT NULL REFERENCES pcb_layouts(id) ON DELETE CASCADE,

    -- Iteration info
    iteration_number INTEGER NOT NULL,
    agent_strategy VARCHAR(50) NOT NULL,

    -- Scores
    score DECIMAL(5,2) NOT NULL,
    drc_violations INTEGER NOT NULL DEFAULT 0,
    improvement_delta DECIMAL(5,2),

    -- Changes made
    changes JSONB NOT NULL DEFAULT '[]',

    -- Validation snapshot
    validation_snapshot JSONB,

    -- Timing
    duration_ms INTEGER,

    -- Timestamps
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    -- Constraints
    CONSTRAINT mapos_agent_check CHECK (agent_strategy IN (
        'conservative', 'aggressive_compact', 'thermal_optimized',
        'emi_optimized', 'dfm_optimized'
    )),
    CONSTRAINT mapos_iteration_positive CHECK (iteration_number >= 0)
);

-- Indexes for mapos_iterations
CREATE INDEX IF NOT EXISTS idx_mapos_pcb_layout_id ON mapos_iterations(pcb_layout_id);
CREATE INDEX IF NOT EXISTS idx_mapos_iteration ON mapos_iterations(pcb_layout_id, iteration_number);
CREATE INDEX IF NOT EXISTS idx_mapos_agent ON mapos_iterations(agent_strategy);
CREATE INDEX IF NOT EXISTS idx_mapos_score ON mapos_iterations(score DESC);

-- ============================================================================
-- UPDATED_AT TRIGGER FUNCTION
-- ============================================================================
-- Automatically updates the updated_at timestamp on row modifications.

CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Apply triggers to all tables with updated_at column
DROP TRIGGER IF EXISTS update_projects_updated_at ON projects;
CREATE TRIGGER update_projects_updated_at
    BEFORE UPDATE ON projects
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

DROP TRIGGER IF EXISTS update_schematics_updated_at ON schematics;
CREATE TRIGGER update_schematics_updated_at
    BEFORE UPDATE ON schematics
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

DROP TRIGGER IF EXISTS update_pcb_layouts_updated_at ON pcb_layouts;
CREATE TRIGGER update_pcb_layouts_updated_at
    BEFORE UPDATE ON pcb_layouts
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

DROP TRIGGER IF EXISTS update_simulations_updated_at ON simulations;
CREATE TRIGGER update_simulations_updated_at
    BEFORE UPDATE ON simulations
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

DROP TRIGGER IF EXISTS update_firmware_projects_updated_at ON firmware_projects;
CREATE TRIGGER update_firmware_projects_updated_at
    BEFORE UPDATE ON firmware_projects
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

DROP TRIGGER IF EXISTS update_job_history_updated_at ON job_history;
CREATE TRIGGER update_job_history_updated_at
    BEFORE UPDATE ON job_history
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

DROP TRIGGER IF EXISTS update_manufacturing_orders_updated_at ON manufacturing_orders;
CREATE TRIGGER update_manufacturing_orders_updated_at
    BEFORE UPDATE ON manufacturing_orders
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- ============================================================================
-- COMMENTS FOR DOCUMENTATION
-- ============================================================================

COMMENT ON TABLE projects IS 'Core EE Design projects supporting all 10 development phases';
COMMENT ON TABLE schematics IS 'Schematic designs with KiCad source, netlists, and BOMs';
COMMENT ON TABLE pcb_layouts IS 'PCB layouts with KiCad PCB files and DRC/MAPOS results';
COMMENT ON TABLE simulations IS 'Simulation jobs for SPICE, thermal, SI, RF, and EMC analysis';
COMMENT ON TABLE firmware_projects IS 'Firmware projects with MCU config, RTOS, and HAL settings';
COMMENT ON TABLE job_history IS 'BullMQ job execution history for debugging and analytics';
COMMENT ON TABLE manufacturing_orders IS 'PCB manufacturing orders and vendor quotes';
COMMENT ON TABLE validation_results IS 'Multi-LLM and domain validation results';
COMMENT ON TABLE mapos_iterations IS 'MAPOS optimization iteration history for analysis';
