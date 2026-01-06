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
  };

  return ConfigSchema.parse(rawConfig);
}

export const config = loadConfig();

export function getConfig(): Config {
  return config;
}