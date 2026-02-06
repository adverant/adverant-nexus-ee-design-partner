/**
 * EE Design Partner - LLM Type Definitions
 *
 * Types for OpenRouter LLM client and prompt management
 */

// ============================================================================
// Message Types
// ============================================================================

export type LLMRole = 'system' | 'user' | 'assistant';

export interface LLMMessage {
  role: LLMRole;
  content: string;
  name?: string;
}

export interface LLMMessageWithImages extends LLMMessage {
  images?: LLMImage[];
}

export interface LLMImage {
  type: 'base64' | 'url';
  data: string;
  mediaType?: 'image/png' | 'image/jpeg' | 'image/gif' | 'image/webp';
}

// ============================================================================
// Model Configuration
// ============================================================================

export type SupportedModel =
  | 'anthropic/claude-opus-4.6'
  | 'anthropic/claude-opus-4.6'
  | 'google/gemini-2.0-flash'
  | 'google/gemini-2.5-pro'
  | 'anthropic/claude-3-5-haiku';

export interface ModelConfig {
  id: SupportedModel;
  displayName: string;
  contextWindow: number;
  maxOutputTokens: number;
  inputCostPer1k: number;
  outputCostPer1k: number;
  supportsImages: boolean;
  supportsStreaming: boolean;
  bestFor: string[];
}

export const MODEL_CONFIGS: Record<SupportedModel, ModelConfig> = {
  'anthropic/claude-opus-4.6': {
    id: 'anthropic/claude-opus-4.6',
    displayName: 'Claude Opus 4',
    contextWindow: 200000,
    maxOutputTokens: 32000,
    inputCostPer1k: 0.015,
    outputCostPer1k: 0.075,
    supportsImages: true,
    supportsStreaming: true,
    bestFor: ['complex reasoning', 'design review', 'architecture decisions'],
  },
  'anthropic/claude-opus-4.6': {
    id: 'anthropic/claude-opus-4.6',
    displayName: 'Claude Opus 4.6',
    contextWindow: 200000,
    maxOutputTokens: 16000,
    inputCostPer1k: 0.003,
    outputCostPer1k: 0.015,
    supportsImages: true,
    supportsStreaming: true,
    bestFor: ['code generation', 'general tasks', 'balanced performance'],
  },
  'anthropic/claude-3-5-haiku': {
    id: 'anthropic/claude-3-5-haiku',
    displayName: 'Claude 3.5 Haiku',
    contextWindow: 200000,
    maxOutputTokens: 8192,
    inputCostPer1k: 0.0008,
    outputCostPer1k: 0.004,
    supportsImages: true,
    supportsStreaming: true,
    bestFor: ['fast responses', 'simple tasks', 'high throughput'],
  },
  'google/gemini-2.0-flash': {
    id: 'google/gemini-2.0-flash',
    displayName: 'Gemini 2.0 Flash',
    contextWindow: 1000000,
    maxOutputTokens: 8192,
    inputCostPer1k: 0.00015,
    outputCostPer1k: 0.0006,
    supportsImages: true,
    supportsStreaming: true,
    bestFor: ['fast validation', 'large context', 'cost efficiency'],
  },
  'google/gemini-2.5-pro': {
    id: 'google/gemini-2.5-pro',
    displayName: 'Gemini 2.5 Pro',
    contextWindow: 1000000,
    maxOutputTokens: 65536,
    inputCostPer1k: 0.00125,
    outputCostPer1k: 0.005,
    supportsImages: true,
    supportsStreaming: true,
    bestFor: ['validation', 'second opinion', 'large document analysis'],
  },
};

// ============================================================================
// Request/Response Types
// ============================================================================

export interface LLMOptions {
  model?: SupportedModel;
  temperature?: number;
  maxTokens?: number;
  topP?: number;
  topK?: number;
  frequencyPenalty?: number;
  presencePenalty?: number;
  stopSequences?: string[];
  responseFormat?: 'text' | 'json';
  timeout?: number;
}

export interface LLMResponse {
  id: string;
  model: string;
  content: string;
  finishReason: FinishReason;
  usage: TokenUsage;
  latencyMs: number;
}

export type FinishReason = 'stop' | 'length' | 'content_filter' | 'error';

export interface TokenUsage {
  promptTokens: number;
  completionTokens: number;
  totalTokens: number;
  estimatedCost: number;
}

// ============================================================================
// Streaming Types
// ============================================================================

export interface StreamChunk {
  id: string;
  content: string;
  finishReason?: FinishReason;
  usage?: TokenUsage;
}

export type StreamCallback = (chunk: StreamChunk) => void;

// ============================================================================
// Validation Types
// ============================================================================

export interface ValidationOptions<T> {
  validator: (response: string) => T | null;
  maxRetries?: number;
  retryDelay?: number;
  onRetry?: (attempt: number, error: string) => void;
}

export interface ValidatedResponse<T> {
  success: boolean;
  data?: T;
  raw: string;
  attempts: number;
  error?: string;
}

// ============================================================================
// Rate Limiting Types
// ============================================================================

export interface RateLimitConfig {
  requestsPerMinute: number;
  tokensPerMinute: number;
  burstAllowance: number;
}

export interface RateLimitState {
  requestCount: number;
  tokenCount: number;
  windowStart: number;
  retryAfter?: number;
}

// ============================================================================
// Error Types
// ============================================================================

export interface LLMErrorDetails {
  code: LLMErrorCode;
  message: string;
  retryable: boolean;
  retryAfterMs?: number;
  model?: string;
  requestId?: string;
}

export type LLMErrorCode =
  | 'RATE_LIMIT'
  | 'CONTEXT_LENGTH'
  | 'INVALID_REQUEST'
  | 'AUTHENTICATION'
  | 'MODEL_UNAVAILABLE'
  | 'CONTENT_FILTER'
  | 'TIMEOUT'
  | 'NETWORK_ERROR'
  | 'PARSE_ERROR'
  | 'UNKNOWN';

// ============================================================================
// Prompt Template Types
// ============================================================================

export interface PromptTemplate<TInput, TOutput> {
  name: string;
  description: string;
  version: string;
  inputSchema: Record<keyof TInput, string>;
  outputSchema: Record<keyof TOutput, string>;
  build: (input: TInput) => LLMMessage[];
  parse: (response: string) => TOutput | null;
}

// ============================================================================
// EE Design Specific Types
// ============================================================================

export interface SchematicBlockDiagram {
  blocks: SchematicBlock[];
  connections: BlockConnection[];
  powerRails: PowerRail[];
  notes: string[];
}

export interface SchematicBlock {
  id: string;
  name: string;
  type: string;
  description: string;
  inputs: string[];
  outputs: string[];
  components: string[];
}

export interface BlockConnection {
  from: { block: string; port: string };
  to: { block: string; port: string };
  signalType: 'power' | 'ground' | 'analog' | 'digital' | 'differential';
}

export interface PowerRail {
  name: string;
  voltage: number;
  current: number;
  source: string;
}

export interface ComponentSelection {
  reference: string;
  partNumber: string;
  manufacturer: string;
  value: string;
  footprint: string;
  description: string;
  alternatives: string[];
  reasoning: string;
}

export interface NetlistNode {
  net: string;
  connections: Array<{ component: string; pin: string }>;
  netClass: string;
  properties: Record<string, unknown>;
}

export interface FirmwareArchitecture {
  layers: FirmwareLayer[];
  modules: FirmwareModule[];
  tasks: FirmwareTaskSpec[];
  interfaces: FirmwareInterface[];
  dependencies: string[];
}

export interface FirmwareLayer {
  name: string;
  description: string;
  modules: string[];
}

export interface FirmwareModule {
  name: string;
  layer: string;
  description: string;
  files: string[];
  dependencies: string[];
  publicApi: string[];
}

export interface FirmwareTaskSpec {
  name: string;
  priority: number;
  stackSize: number;
  periodMs: number;
  description: string;
  modules: string[];
}

export interface FirmwareInterface {
  name: string;
  type: 'uart' | 'spi' | 'i2c' | 'can' | 'usb' | 'ethernet' | 'gpio';
  peripheral: string;
  config: Record<string, unknown>;
}

export interface CodeReviewResult {
  score: number;
  passed: boolean;
  issues: CodeIssue[];
  suggestions: string[];
  securityConcerns: string[];
  performanceNotes: string[];
}

export interface CodeIssue {
  severity: 'critical' | 'error' | 'warning' | 'info';
  line?: number;
  file?: string;
  message: string;
  suggestion: string;
}

export interface DesignReview {
  domain: string;
  score: number;
  passed: boolean;
  issues: DesignIssue[];
  recommendations: string[];
  confidence: number;
  reasoning: string;
}

export interface DesignIssue {
  severity: 'critical' | 'major' | 'minor' | 'info';
  category: string;
  description: string;
  location?: string;
  recommendation: string;
  reference?: string;
}

export interface ConsensusInput {
  reviews: DesignReview[];
  artifact: {
    type: string;
    id: string;
    summary: string;
  };
}

export interface ConsensusSummary {
  finalScore: number;
  passed: boolean;
  confidence: number;
  agreementLevel: number;
  resolvedIssues: ResolvedIssue[];
  keyFindings: string[];
  recommendations: string[];
}

export interface ResolvedIssue {
  issue: string;
  opinions: Array<{ reviewer: string; opinion: string }>;
  resolution: string;
  method: 'unanimous' | 'majority' | 'weighted' | 'expert';
}
