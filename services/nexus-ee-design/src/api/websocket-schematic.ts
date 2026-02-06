/**
 * WebSocket Event Types for Schematic Generation Streaming
 *
 * Enables real-time progress updates from Python MAPO pipeline to browser.
 * Following the patent plugin streaming pattern.
 */

/**
 * Event types emitted during schematic generation.
 * Maps to phases in the Python MAPO pipeline.
 */
export enum SchematicEventType {
  // Phase lifecycle events
  PHASE_START = 'phase_start',
  PHASE_COMPLETE = 'phase_complete',

  // Symbol resolution phase (0-30%)
  SYMBOL_RESOLVING = 'symbol_resolving',
  SYMBOL_RESOLVED = 'symbol_resolved',
  SYMBOL_FROM_MEMORY = 'symbol_from_memory',
  SYMBOL_FROM_CACHE = 'symbol_from_cache',
  SYMBOL_PLACEHOLDER = 'symbol_placeholder',

  // Connection generation phase (30-50%)
  CONNECTIONS_GENERATING = 'connections_generating',
  CONNECTIONS_LLM_CALL = 'connections_llm_call',
  CONNECTIONS_GENERATED = 'connections_generated',

  // Layout optimization phase (50-60%)
  LAYOUT_START = 'layout_start',
  LAYOUT_OPTIMIZING = 'layout_optimizing',
  LAYOUT_COMPLETE = 'layout_complete',

  // Wire routing phase (60-70%)
  WIRING_START = 'wiring_start',
  WIRING_PROGRESS = 'wiring_progress',
  WIRING_COMPLETE = 'wiring_complete',

  // Assembly phase (70-80%)
  ASSEMBLY_START = 'assembly_start',
  ASSEMBLY_PROGRESS = 'assembly_progress',
  ASSEMBLY_COMPLETE = 'assembly_complete',

  // Smoke test phase (80-90%)
  SMOKE_TEST_START = 'smoke_test_start',
  SMOKE_TEST_RUNNING = 'smoke_test_running',
  SMOKE_TEST_RESULT = 'smoke_test_result',

  // Gaming AI optimization phase (if smoke test fails)
  GAMING_AI_START = 'gaming_ai_start',
  GAMING_AI_ITERATION = 'gaming_ai_iteration',
  GAMING_AI_MUTATION = 'gaming_ai_mutation',
  GAMING_AI_IMPROVEMENT = 'gaming_ai_improvement',
  GAMING_AI_COMPLETE = 'gaming_ai_complete',

  // Validation phase (90-95%)
  VALIDATION_START = 'validation_start',
  VALIDATION_RUNNING = 'validation_running',
  VALIDATION_COMPLETE = 'validation_complete',

  // Export phase (95-100%)
  EXPORT_START = 'export_start',
  EXPORT_PROGRESS = 'export_progress',
  EXPORT_COMPLETE = 'export_complete',

  // Compliance Checklist events (MAPO v3.0)
  CHECKLIST_ITEM_START = 'checklist_item_start',
  CHECKLIST_ITEM_PASS = 'checklist_item_pass',
  CHECKLIST_ITEM_FAIL = 'checklist_item_fail',
  CHECKLIST_BATCH_UPDATE = 'checklist_batch_update',
  COMPLIANCE_SCORE_UPDATE = 'compliance_score_update',

  // Symbol Assembly events (MAPO v3.0)
  SYMBOL_ASSEMBLY_START = 'symbol_assembly_start',
  SYMBOL_ASSEMBLY_SEARCH = 'symbol_assembly_search',
  SYMBOL_ASSEMBLY_FOUND = 'symbol_assembly_found',
  SYMBOL_ASSEMBLY_GENERATED = 'symbol_assembly_generated',
  SYMBOL_ASSEMBLY_DATASHEET = 'symbol_assembly_datasheet',
  SYMBOL_ASSEMBLY_COMPLETE = 'symbol_assembly_complete',

  // Final states
  COMPLETE = 'complete',
  ERROR = 'error',
}

/**
 * Phase names for UI display
 */
export type SchematicPhase =
  | 'symbols'
  | 'connections'
  | 'layout'
  | 'wiring'
  | 'assembly'
  | 'smoke_test'
  | 'gaming_ai'
  | 'standards'
  | 'validation'
  | 'export';

/**
 * Progress event emitted from Python pipeline through WebSocket
 */
export interface SchematicProgressEvent {
  /** Event type */
  type: SchematicEventType;

  /** Unique operation ID for this generation request */
  operationId: string;

  /** ISO timestamp */
  timestamp: string;

  /** Overall progress 0-100 */
  progress_percentage: number;

  /** Human-readable message for current step */
  current_step: string;

  /** Current phase */
  phase?: SchematicPhase;

  /** Progress within current phase (0-100) */
  phase_progress?: number;

  /** Component details (for symbol resolution) */
  component_name?: string;
  component_index?: number;
  total_components?: number;

  /** Symbol source (for symbol resolution) */
  symbol_source?: string;
  symbol_quality?: 'verified' | 'cached' | 'generated' | 'placeholder';

  /** Connection details (for connection generation) */
  connection_count?: number;

  /** Gaming AI details */
  iteration?: number;
  max_iterations?: number;
  fitness?: number;
  mutation_type?: string;

  /** Smoke test details */
  smoke_test_passed?: boolean;
  smoke_test_issues?: string[];

  /** Validation details */
  validation_score?: number;
  validation_passed?: boolean;

  /** Export details */
  export_format?: string;
  export_path?: string;

  /** Error details (for ERROR type) */
  error_message?: string;
  error_code?: string;

  /** Compliance checklist update (MAPO v3.0) */
  checklist_update?: {
    checkId: string;
    status: 'checking' | 'passed' | 'failed';
    title?: string;
    result?: {
      passed: boolean;
      score: number;
      message: string;
      violations?: Array<{
        id: string;
        componentRef?: string;
        netName?: string;
        message: string;
        suggestion?: string;
      }>;
    };
  };

  /** Compliance batch update (MAPO v3.0) */
  checklist_batch?: {
    updates: Array<{
      checkId: string;
      status: 'passed' | 'failed' | 'skipped';
      result?: { passed: boolean; score: number; message: string };
    }>;
    overallScore: number;
  };

  /** Compliance score (MAPO v3.0) */
  compliance_score?: {
    overallScore: number;
    passRate: number;
    criticalFailures: number;
  };

  /** Symbol assembly progress (MAPO v3.0) */
  symbol_assembly?: {
    componentName?: string;
    source?: string;
    status?: 'searching' | 'found' | 'generated' | 'error';
    totalComponents?: number;
    completedComponents?: number;
  };

  /** Additional data */
  data?: Record<string, unknown>;
}

/**
 * Final result event (type === 'complete')
 */
export interface SchematicCompleteEvent extends SchematicProgressEvent {
  type: SchematicEventType.COMPLETE;

  /** Result summary */
  result: {
    success: boolean;
    schematicId?: string;
    schematicPath?: string;
    componentCount: number;
    connectionCount: number;
    wireCount: number;
    validationScore?: number;
    smokeTestPassed?: boolean;
    exportPaths?: {
      kicad?: string;
      pdf?: string;
      svg?: string;
      png?: string;
    };
    errors?: string[];
  };
}

/**
 * Error event (type === 'error')
 */
export interface SchematicErrorEvent extends SchematicProgressEvent {
  type: SchematicEventType.ERROR;

  error_message: string;
  error_code: string;

  /** Stack trace (dev only) */
  stack?: string;

  /** Partial result if any */
  partial_result?: {
    symbolsResolved?: number;
    connectionsGenerated?: number;
    lastSuccessfulPhase?: SchematicPhase;
  };
}

/**
 * Helper to create progress events
 */
export function createProgressEvent(
  operationId: string,
  type: SchematicEventType,
  progress: number,
  message: string,
  extra?: Partial<SchematicProgressEvent>
): SchematicProgressEvent {
  return {
    type,
    operationId,
    timestamp: new Date().toISOString(),
    progress_percentage: Math.min(100, Math.max(0, progress)),
    current_step: message,
    ...extra,
  };
}

/**
 * Phase progress ranges
 * Used to calculate overall progress from phase progress
 */
export const PHASE_PROGRESS_RANGES: Record<SchematicPhase, [number, number]> = {
  symbols: [0, 25],
  connections: [25, 45],
  layout: [45, 55],
  wiring: [55, 65],
  assembly: [65, 72],
  smoke_test: [72, 80],
  gaming_ai: [80, 87], // Only if smoke test fails
  standards: [80, 87], // Compliance checks (overlaps with gaming_ai)
  validation: [87, 93],
  export: [93, 100],
};

/**
 * Calculate overall progress from phase and phase progress
 */
export function calculateOverallProgress(
  phase: SchematicPhase,
  phaseProgress: number
): number {
  const [start, end] = PHASE_PROGRESS_RANGES[phase];
  const range = end - start;
  return Math.round(start + (range * phaseProgress) / 100);
}
