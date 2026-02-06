/**
 * EE Design Partner - LLM Module
 *
 * OpenRouter LLM client with specialized prompts for EE design automation.
 *
 * Usage:
 * ```typescript
 * import {
 *   callLLM,
 *   callLLMWithValidation,
 *   callLLMStreaming,
 *   callClaudeOpus,
 *   callGeminiPro,
 *   schematicPrompts,
 *   firmwarePrompts,
 *   validationPrompts,
 * } from './llm';
 *
 * // Basic LLM call
 * const response = await callLLM([
 *   { role: 'system', content: 'You are an expert...' },
 *   { role: 'user', content: 'Design a power supply...' }
 * ], { model: 'anthropic/claude-opus-4.6' });
 *
 * // With JSON validation
 * const result = await callLLMWithValidation(
 *   schematicPrompts.generateBlockDiagramPrompt(requirements, 'motor_controller'),
 *   schematicPrompts.validateBlockDiagramResponse,
 *   { temperature: 0.3 }
 * );
 *
 * // Streaming
 * await callLLMStreaming(messages, {}, chunk => {
 *   process.stdout.write(chunk.content);
 * });
 * ```
 */

// ============================================================================
// Core Client Exports
// ============================================================================

export {
  callLLM,
  callLLMWithValidation,
  callLLMStreaming,
  callClaudeOpus,
  callGeminiFlash,
  callGeminiPro,
  estimateTokens,
  calculateCost,
  getAvailableModels,
  getModelConfig,
} from './openrouter-client.js';

// ============================================================================
// Type Exports
// ============================================================================

export type {
  // Message Types
  LLMRole,
  LLMMessage,
  LLMMessageWithImages,
  LLMImage,

  // Model Types
  SupportedModel,
  ModelConfig,

  // Request/Response Types
  LLMOptions,
  LLMResponse,
  TokenUsage,
  FinishReason,

  // Streaming Types
  StreamChunk,
  StreamCallback,

  // Validation Types
  ValidationOptions,
  ValidatedResponse,

  // Rate Limiting Types
  RateLimitConfig,
  RateLimitState,

  // Error Types
  LLMErrorDetails,
  LLMErrorCode,

  // Prompt Template Types
  PromptTemplate,

  // EE Design Types
  SchematicBlockDiagram,
  SchematicBlock,
  BlockConnection,
  PowerRail,
  ComponentSelection,
  NetlistNode,
  FirmwareArchitecture,
  FirmwareLayer,
  FirmwareModule,
  FirmwareTaskSpec,
  FirmwareInterface,
  CodeReviewResult,
  CodeIssue,
  DesignReview,
  DesignIssue,
  ConsensusInput,
  ConsensusSummary,
  ResolvedIssue,
} from './types.js';

export { MODEL_CONFIGS } from './types.js';

// ============================================================================
// Schematic Prompts
// ============================================================================

import schematicPromptsModule from './prompts/schematic-prompts.js';

export const schematicPrompts = schematicPromptsModule;

export {
  generateBlockDiagramPrompt,
  validateBlockDiagramResponse,
  generateComponentSelectionPrompt,
  validateComponentSelectionResponse,
  generateNetlistPrompt,
  validateNetlistResponse,
  validateSchematicPrompt,
} from './prompts/schematic-prompts.js';

export type {
  ProjectRequirements,
  ProjectType,
  BlockRequirements,
  SchematicForValidation,
} from './prompts/schematic-prompts.js';

// ============================================================================
// Firmware Prompts
// ============================================================================

import firmwarePromptsModule from './prompts/firmware-prompts.js';

export const firmwarePrompts = firmwarePromptsModule;

export {
  generateFirmwareArchitecturePrompt,
  validateFirmwareArchitectureResponse,
  generateHALCodePrompt,
  generateDriverCodePrompt,
  validateCodePrompt,
  validateCodeReviewResponse,
  generateISRPrompt,
  generateStateMachinePrompt,
} from './prompts/firmware-prompts.js';

export type {
  MCUFamily,
  RTOSType,
  FirmwareRequirements,
  PeripheralRequirement,
  PeripheralSpec,
  ComponentDatasheet,
  CodeLanguage,
  CodeForReview,
} from './prompts/firmware-prompts.js';

// ============================================================================
// Validation Prompts
// ============================================================================

import validationPromptsModule from './prompts/validation-prompts.js';

export const validationPrompts = validationPromptsModule;

export {
  reviewDesignPrompt,
  validateDesignReviewResponse,
  consensusPrompt,
  validateConsensusResponse,
  reviewSafetyDesignPrompt,
  reviewDFMPrompt,
} from './prompts/validation-prompts.js';

export type {
  ArtifactType,
  ReviewDomain,
  DesignArtifact,
  SchematicArtifact,
  PCBArtifact,
  FirmwareArtifact,
  SimulationArtifact,
} from './prompts/validation-prompts.js';

// ============================================================================
// Default Export
// ============================================================================

import openrouterClient from './openrouter-client.js';

export default {
  ...openrouterClient,
  prompts: {
    schematic: schematicPromptsModule,
    firmware: firmwarePromptsModule,
    validation: validationPromptsModule,
  },
};
