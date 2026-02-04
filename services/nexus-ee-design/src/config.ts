/**
 * EE Design Partner - Configuration
 *
 * Centralized configuration management with environment variable support
 */

import { z } from 'zod';

const ConfigSchema = z.object({
  // Server Configuration
  port: z.number().default(8080),
  nodeEnv: z.enum(['development', 'production', 'test']).default('development'),
  logLevel: z.enum(['debug', 'info', 'warn', 'error']).default('info'),

  // Build Metadata
  buildId: z.string().optional(),
  buildTimestamp: z.string().optional(),
  gitCommit: z.string().optional(),
  gitBranch: z.string().optional(),
  version: z.string().default('1.0.0'),

  // Nexus Plugin Configuration
  pluginId: z.string().default('nexus-ee-design'),
  pluginName: z.string().default('EE Design Partner'),
  executionMode: z.enum(['mcp_container', 'standalone']).default('mcp_container'),

  // Database Configuration
  postgres: z.object({
    host: z.string().default('localhost'),
    port: z.number().default(5432),
    database: z.string().default('nexus'),
    user: z.string().default('nexus'),
    password: z.string().default(''),
    ssl: z.boolean().default(false),
  }),

  // Redis Configuration
  redis: z.object({
    host: z.string().default('localhost'),
    port: z.number().default(6379),
    password: z.string().optional(),
  }),

  // Service URLs
  services: z.object({
    graphragUrl: z.string().default('http://localhost:9000'),
    mageagentUrl: z.string().default('http://localhost:9010'),
    sandboxUrl: z.string().default('http://localhost:9020'),
    fileprocessUrl: z.string().default('http://localhost:9030'),
    skillsEngineUrl: z.string().default('http://localhost:9040'),
  }),

  // LLM Configuration
  llm: z.object({
    openrouterApiKey: z.string().default(''),
    anthropicApiKey: z.string().default(''),
    primaryModel: z.string().default('anthropic/claude-opus-4'),
    validationModel: z.string().default('google/gemini-2.5-pro'),
    fastModel: z.string().default('anthropic/claude-3-5-haiku'),
  }),

  // KiCad Configuration
  kicad: z.object({
    pythonPath: z.string().default('/opt/venv/bin/python3'),
    scriptsDir: z.string().default('/app/python-scripts'),
    libraryPath: z.string().optional(),
  }),

  // Simulation Configuration
  simulation: z.object({
    ngspicePath: z.string().default('/usr/bin/ngspice'),
    openfoamPath: z.string().optional(),
    openEMSPath: z.string().optional(),
    maxConcurrentSims: z.number().default(4),
    timeoutSeconds: z.number().default(3600),
  }),

  // Storage Configuration
  storage: z.object({
    tempDir: z.string().default('/tmp/ee-design'),
    projectsDir: z.string().default('/data/projects'),
    outputDir: z.string().default('/data/output'),
    maxUploadSize: z.number().default(100 * 1024 * 1024), // 100MB
  }),

  // NFS Share Artifacts Configuration (Terminal Computer)
  artifacts: z.object({
    enabled: z.boolean().default(true),
    basePath: z.string().default('/plugins/ee-design-plugin/artifacts'),
    paths: z.object({
      schematics: z.string().default('/plugins/ee-design-plugin/artifacts/schematics'),
      pcbLayouts: z.string().default('/plugins/ee-design-plugin/artifacts/pcb-layouts'),
      threeDRenders: z.string().default('/plugins/ee-design-plugin/artifacts/3d-renders'),
      gerbers: z.string().default('/plugins/ee-design-plugin/artifacts/gerbers'),
      bom: z.string().default('/plugins/ee-design-plugin/artifacts/bom'),
      benchTesting: z.string().default('/plugins/ee-design-plugin/artifacts/bench-testing'),
      drcReports: z.string().default('/plugins/ee-design-plugin/artifacts/drc-reports'),
      simulations: z.string().default('/plugins/ee-design-plugin/artifacts/simulations'),
      validationReports: z.string().default('/plugins/ee-design-plugin/artifacts/validation-reports'),
    }),
    retentionDays: z.number().default(90), // Keep artifacts for 90 days
    syncEnabled: z.boolean().default(true), // Sync to GraphRAG
  }),

  // Layout Generation Configuration
  layout: z.object({
    maxIterations: z.number().default(100),
    convergenceThreshold: z.number().default(0.1),
    targetScore: z.number().default(95),
    enabledAgents: z.array(z.string()).default([
      'conservative',
      'aggressive_compact',
      'thermal_optimized',
      'emi_optimized',
      'dfm_optimized',
    ]),
  }),

  // Validation Configuration
  validation: z.object({
    drcEnabled: z.boolean().default(true),
    ercEnabled: z.boolean().default(true),
    ipc2221Enabled: z.boolean().default(true),
    siEnabled: z.boolean().default(true),
    thermalEnabled: z.boolean().default(true),
    dfmEnabled: z.boolean().default(true),
    multiLlmEnabled: z.boolean().default(true),
  }),

  // HIL (Hardware-in-the-Loop) Testing Configuration
  hil: z.object({
    enabled: z.boolean().default(true),
    pythonScriptsDir: z.string().default('/app/python-scripts/hil'),

    // Discovery Configuration
    discovery: z.object({
      autoDiscover: z.boolean().default(true),
      scanIntervalSeconds: z.number().default(30),
      scanTimeout: z.number().default(10000),
      enabledTypes: z.array(z.string()).default([
        'logic_analyzer',
        'oscilloscope',
        'power_supply',
        'motor_emulator',
        'daq',
        'can_analyzer',
      ]),
    }),

    // Saleae Logic Analyzer Configuration
    saleae: z.object({
      host: z.string().default('localhost'),
      port: z.number().default(10430),
      automationApiEnabled: z.boolean().default(true),
    }),

    // VISA/PyVISA Configuration
    visa: z.object({
      resourceManager: z.string().optional(),
      scanTimeout: z.number().default(5000),
      defaultTimeout: z.number().default(10000),
    }),

    // Capture Storage Configuration
    storage: z.object({
      capturesDir: z.string().default('/plugins/ee-design-plugin/artifacts/hil-captures'),
      maxCaptureSizeMb: z.number().default(500),
      retentionDays: z.number().default(30),
      compressionEnabled: z.boolean().default(true),
    }),

    // Execution Configuration
    execution: z.object({
      maxConcurrentTests: z.number().default(1),
      defaultTimeoutMs: z.number().default(600000),
      retryOnHardwareError: z.boolean().default(true),
      maxRetries: z.number().default(3),
      retryDelayMs: z.number().default(5000),
    }),

    // Safety Configuration
    safety: z.object({
      emergencyStopEnabled: z.boolean().default(true),
      maxVoltage: z.number().default(60),
      maxCurrent: z.number().default(30),
      thermalShutdownTemp: z.number().default(85),
      requireCalibration: z.boolean().default(false),
    }),
  }),

  // CLI/Terminal Access Configuration
  cliAccess: z.object({
    enabled: z.boolean().default(true),
    allowedUserAgents: z.array(z.string()).default([
      'curl',
      'wget',
      'httpie',
      'claude-code',
      'nexus-cli',
      'python-requests',
      'node-fetch',
      'axios',
    ]),
    trustedIPs: z.array(z.string()).default([
      '127.0.0.1',
      '::1',
      'localhost',
    ]),
    systemUserId: z.string().default('claude-code-cli'),
    permissions: z.object({
      read: z.boolean().default(true),
      write: z.boolean().default(true),
      execute: z.boolean().default(true),
      admin: z.boolean().default(true),
    }),
  }),
});

export type Config = z.infer<typeof ConfigSchema>;

function loadConfig(): Config {
  const rawConfig = {
    port: parseInt(process.env.PORT || '8080', 10),
    nodeEnv: process.env.NODE_ENV || 'development',
    logLevel: process.env.LOG_LEVEL || 'info',

    buildId: process.env.NEXUS_BUILD_ID,
    buildTimestamp: process.env.NEXUS_BUILD_TIMESTAMP,
    gitCommit: process.env.NEXUS_GIT_COMMIT,
    gitBranch: process.env.NEXUS_GIT_BRANCH,
    version: process.env.NEXUS_VERSION || '1.0.0',

    pluginId: process.env.NEXUS_PLUGIN_ID || 'nexus-ee-design',
    pluginName: process.env.NEXUS_PLUGIN_NAME || 'EE Design Partner',
    executionMode: process.env.NEXUS_EXECUTION_MODE || 'mcp_container',

    postgres: {
      host: process.env.POSTGRES_HOST || 'localhost',
      port: parseInt(process.env.POSTGRES_PORT || '5432', 10),
      database: process.env.POSTGRES_DATABASE || 'nexus',
      user: process.env.POSTGRES_USER || 'nexus',
      password: process.env.POSTGRES_PASSWORD || '',
      ssl: process.env.POSTGRES_SSL === 'true',
    },

    redis: {
      host: process.env.REDIS_HOST || 'localhost',
      port: parseInt(process.env.REDIS_PORT || '6379', 10),
      password: process.env.REDIS_PASSWORD,
    },

    services: {
      graphragUrl: process.env.NEXUS_GRAPHRAG_URL || 'http://localhost:9000',
      mageagentUrl: process.env.NEXUS_MAGEAGENT_URL || 'http://localhost:9010',
      sandboxUrl: process.env.NEXUS_SANDBOX_URL || 'http://localhost:9020',
      fileprocessUrl: process.env.NEXUS_FILEPROCESS_URL || 'http://localhost:9030',
      skillsEngineUrl: process.env.NEXUS_SKILLS_ENGINE_URL || 'http://localhost:9040',
    },

    llm: {
      openrouterApiKey: process.env.OPENROUTER_API_KEY || '',
      anthropicApiKey: process.env.ANTHROPIC_API_KEY || '',
      primaryModel: process.env.PRIMARY_LLM_MODEL || 'anthropic/claude-opus-4',
      validationModel: process.env.VALIDATION_LLM_MODEL || 'google/gemini-2.5-pro',
      fastModel: process.env.FAST_LLM_MODEL || 'anthropic/claude-3-5-haiku',
    },

    kicad: {
      pythonPath: process.env.PYTHON_PATH || '/opt/venv/bin/python3',
      scriptsDir: process.env.PYTHON_SCRIPTS_DIR || '/app/python-scripts',
      libraryPath: process.env.KICAD_LIBRARY_PATH,
    },

    simulation: {
      ngspicePath: process.env.NGSPICE_PATH || '/usr/bin/ngspice',
      openfoamPath: process.env.OPENFOAM_PATH,
      openEMSPath: process.env.OPENEMS_PATH,
      maxConcurrentSims: parseInt(process.env.MAX_CONCURRENT_SIMS || '4', 10),
      timeoutSeconds: parseInt(process.env.SIM_TIMEOUT_SECONDS || '3600', 10),
    },

    storage: {
      tempDir: process.env.TEMP_DIR || '/tmp/ee-design',
      projectsDir: process.env.PROJECTS_DIR || '/data/projects',
      outputDir: process.env.OUTPUT_DIR || '/data/output',
      maxUploadSize: parseInt(process.env.MAX_UPLOAD_SIZE || String(100 * 1024 * 1024), 10),
    },

    layout: {
      maxIterations: parseInt(process.env.LAYOUT_MAX_ITERATIONS || '100', 10),
      convergenceThreshold: parseFloat(process.env.LAYOUT_CONVERGENCE_THRESHOLD || '0.1'),
      targetScore: parseFloat(process.env.LAYOUT_TARGET_SCORE || '95'),
      enabledAgents: process.env.ENABLED_AGENTS?.split(',') || [
        'conservative',
        'aggressive_compact',
        'thermal_optimized',
        'emi_optimized',
        'dfm_optimized',
      ],
    },

    validation: {
      drcEnabled: process.env.DRC_ENABLED !== 'false',
      ercEnabled: process.env.ERC_ENABLED !== 'false',
      ipc2221Enabled: process.env.IPC2221_ENABLED !== 'false',
      siEnabled: process.env.SI_ENABLED !== 'false',
      thermalEnabled: process.env.THERMAL_ENABLED !== 'false',
      dfmEnabled: process.env.DFM_ENABLED !== 'false',
      multiLlmEnabled: process.env.MULTI_LLM_ENABLED !== 'false',
    },

    cliAccess: {
      enabled: process.env.CLI_ACCESS_ENABLED !== 'false',
      allowedUserAgents: process.env.CLI_ALLOWED_USER_AGENTS?.split(',') || [
        'curl',
        'wget',
        'httpie',
        'claude-code',
        'nexus-cli',
        'python-requests',
        'node-fetch',
        'axios',
      ],
      trustedIPs: process.env.CLI_TRUSTED_IPS?.split(',') || [
        '127.0.0.1',
        '::1',
        'localhost',
      ],
      systemUserId: process.env.CLI_SYSTEM_USER_ID || 'claude-code-cli',
      permissions: {
        read: process.env.CLI_PERMISSION_READ !== 'false',
        write: process.env.CLI_PERMISSION_WRITE !== 'false',
        execute: process.env.CLI_PERMISSION_EXECUTE !== 'false',
        admin: process.env.CLI_PERMISSION_ADMIN !== 'false',
      },
    },

    artifacts: {
      enabled: process.env.ARTIFACTS_ENABLED !== 'false',
      basePath: process.env.ARTIFACTS_BASE_PATH || '/plugins/ee-design-plugin/artifacts',
      paths: {
        schematics: process.env.ARTIFACTS_SCHEMATICS_PATH || '/plugins/ee-design-plugin/artifacts/schematics',
        pcbLayouts: process.env.ARTIFACTS_PCB_LAYOUTS_PATH || '/plugins/ee-design-plugin/artifacts/pcb-layouts',
        threeDRenders: process.env.ARTIFACTS_3D_RENDERS_PATH || '/plugins/ee-design-plugin/artifacts/3d-renders',
        gerbers: process.env.ARTIFACTS_GERBERS_PATH || '/plugins/ee-design-plugin/artifacts/gerbers',
        bom: process.env.ARTIFACTS_BOM_PATH || '/plugins/ee-design-plugin/artifacts/bom',
        benchTesting: process.env.ARTIFACTS_BENCH_TESTING_PATH || '/plugins/ee-design-plugin/artifacts/bench-testing',
        drcReports: process.env.ARTIFACTS_DRC_REPORTS_PATH || '/plugins/ee-design-plugin/artifacts/drc-reports',
        simulations: process.env.ARTIFACTS_SIMULATIONS_PATH || '/plugins/ee-design-plugin/artifacts/simulations',
        validationReports: process.env.ARTIFACTS_VALIDATION_REPORTS_PATH || '/plugins/ee-design-plugin/artifacts/validation-reports',
      },
      retentionDays: parseInt(process.env.ARTIFACTS_RETENTION_DAYS || '90', 10),
      syncEnabled: process.env.ARTIFACTS_SYNC_ENABLED !== 'false',
    },

    hil: {
      enabled: process.env.HIL_ENABLED !== 'false',
      pythonScriptsDir: process.env.HIL_PYTHON_SCRIPTS_DIR || '/app/python-scripts/hil',

      discovery: {
        autoDiscover: process.env.HIL_AUTO_DISCOVER !== 'false',
        scanIntervalSeconds: parseInt(process.env.HIL_SCAN_INTERVAL || '30', 10),
        scanTimeout: parseInt(process.env.HIL_SCAN_TIMEOUT || '10000', 10),
        enabledTypes: process.env.HIL_ENABLED_TYPES?.split(',') || [
          'logic_analyzer',
          'oscilloscope',
          'power_supply',
          'motor_emulator',
          'daq',
          'can_analyzer',
        ],
      },

      saleae: {
        host: process.env.HIL_SALEAE_HOST || 'localhost',
        port: parseInt(process.env.HIL_SALEAE_PORT || '10430', 10),
        automationApiEnabled: process.env.HIL_SALEAE_API_ENABLED !== 'false',
      },

      visa: {
        resourceManager: process.env.HIL_VISA_RESOURCE_MANAGER,
        scanTimeout: parseInt(process.env.HIL_VISA_SCAN_TIMEOUT || '5000', 10),
        defaultTimeout: parseInt(process.env.HIL_VISA_DEFAULT_TIMEOUT || '10000', 10),
      },

      storage: {
        capturesDir: process.env.HIL_CAPTURES_DIR || '/plugins/ee-design-plugin/artifacts/hil-captures',
        maxCaptureSizeMb: parseInt(process.env.HIL_MAX_CAPTURE_SIZE_MB || '500', 10),
        retentionDays: parseInt(process.env.HIL_RETENTION_DAYS || '30', 10),
        compressionEnabled: process.env.HIL_COMPRESSION_ENABLED !== 'false',
      },

      execution: {
        maxConcurrentTests: parseInt(process.env.HIL_MAX_CONCURRENT_TESTS || '1', 10),
        defaultTimeoutMs: parseInt(process.env.HIL_DEFAULT_TIMEOUT_MS || '600000', 10),
        retryOnHardwareError: process.env.HIL_RETRY_ON_ERROR !== 'false',
        maxRetries: parseInt(process.env.HIL_MAX_RETRIES || '3', 10),
        retryDelayMs: parseInt(process.env.HIL_RETRY_DELAY_MS || '5000', 10),
      },

      safety: {
        emergencyStopEnabled: process.env.HIL_EMERGENCY_STOP !== 'false',
        maxVoltage: parseFloat(process.env.HIL_MAX_VOLTAGE || '60'),
        maxCurrent: parseFloat(process.env.HIL_MAX_CURRENT || '30'),
        thermalShutdownTemp: parseFloat(process.env.HIL_THERMAL_SHUTDOWN_TEMP || '85'),
        requireCalibration: process.env.HIL_REQUIRE_CALIBRATION === 'true',
      },
    },
  };

  return ConfigSchema.parse(rawConfig);
}

export const config = loadConfig();

export function getConfig(): Config {
  return config;
}