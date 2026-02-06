/**
 * EE Design Partner - OpenRouter LLM Client
 *
 * Production-ready OpenRouter client with:
 * - Multiple model support (Claude, Gemini)
 * - JSON validation with retries
 * - Streaming responses
 * - Rate limiting handling
 * - Token counting/cost estimation
 * - Comprehensive error handling
 */

import { config } from '../config.js';
import log from '../utils/logger.js';
import { LLMServiceError, RateLimitError, TimeoutError } from '../utils/errors.js';
import {
  LLMMessage,
  LLMOptions,
  LLMResponse,
  StreamChunk,
  StreamCallback,
  TokenUsage,
  FinishReason,
  SupportedModel,
  MODEL_CONFIGS,
  ValidatedResponse,
  LLMErrorCode,
  LLMErrorDetails,
  RateLimitState,
} from './types.js';

// ============================================================================
// Constants
// ============================================================================

const OPENROUTER_API_URL = 'https://openrouter.ai/api/v1/chat/completions';
const DEFAULT_TIMEOUT = 120000; // 2 minutes
const MAX_RETRIES = 3;
const RETRY_DELAY_BASE = 1000; // 1 second
const TOKENS_PER_CHAR_ESTIMATE = 0.25; // Conservative estimate

// ============================================================================
// OpenRouter API Response Types
// ============================================================================

interface OpenRouterResponse {
  id?: string;
  model?: string;
  choices?: Array<{
    index: number;
    message?: {
      role: string;
      content: string;
    };
    delta?: {
      content?: string;
    };
    finish_reason?: string;
  }>;
  usage?: {
    prompt_tokens: number;
    completion_tokens: number;
    total_tokens: number;
  };
}

// ============================================================================
// Rate Limiting
// ============================================================================

const rateLimitState: RateLimitState = {
  requestCount: 0,
  tokenCount: 0,
  windowStart: Date.now(),
};

function checkRateLimit(estimatedTokens: number): void {
  const now = Date.now();
  const windowMs = 60000; // 1 minute window

  // Reset window if expired
  if (now - rateLimitState.windowStart > windowMs) {
    rateLimitState.requestCount = 0;
    rateLimitState.tokenCount = 0;
    rateLimitState.windowStart = now;
  }

  // Check if we're rate limited
  if (rateLimitState.retryAfter && now < rateLimitState.retryAfter) {
    const waitTime = Math.ceil((rateLimitState.retryAfter - now) / 1000);
    throw new RateLimitError(waitTime, { operation: 'checkRateLimit' });
  }

  // Update counters
  rateLimitState.requestCount++;
  rateLimitState.tokenCount += estimatedTokens;
}

function handleRateLimitResponse(retryAfterHeader: string | null): void {
  const retryAfterSeconds = retryAfterHeader ? parseInt(retryAfterHeader, 10) : 60;
  rateLimitState.retryAfter = Date.now() + retryAfterSeconds * 1000;
}

// ============================================================================
// Token Estimation
// ============================================================================

/**
 * Estimate token count for messages
 * Uses conservative approximation: ~4 characters per token for English text
 */
export function estimateTokens(messages: LLMMessage[]): number {
  let totalChars = 0;

  for (const message of messages) {
    totalChars += message.content.length;
    totalChars += message.role.length + 10; // Role overhead
    if (message.name) {
      totalChars += message.name.length + 5;
    }
  }

  return Math.ceil(totalChars * TOKENS_PER_CHAR_ESTIMATE);
}

/**
 * Calculate cost based on token usage
 */
export function calculateCost(usage: TokenUsage, model: SupportedModel): number {
  const modelConfig = MODEL_CONFIGS[model];
  if (!modelConfig) return 0;

  const inputCost = (usage.promptTokens / 1000) * modelConfig.inputCostPer1k;
  const outputCost = (usage.completionTokens / 1000) * modelConfig.outputCostPer1k;

  return inputCost + outputCost;
}

// ============================================================================
// Error Handling
// ============================================================================

function parseError(error: unknown, statusCode?: number): LLMErrorDetails {
  if (error instanceof Error) {
    const message = error.message.toLowerCase();

    if (statusCode === 429 || message.includes('rate limit')) {
      return {
        code: 'RATE_LIMIT',
        message: 'Rate limit exceeded',
        retryable: true,
        retryAfterMs: 60000,
      };
    }

    if (statusCode === 401 || message.includes('unauthorized') || message.includes('api key')) {
      return {
        code: 'AUTHENTICATION',
        message: 'Invalid or missing API key',
        retryable: false,
      };
    }

    if (statusCode === 400 && message.includes('context length')) {
      return {
        code: 'CONTEXT_LENGTH',
        message: 'Input exceeds model context length',
        retryable: false,
      };
    }

    if (message.includes('timeout') || message.includes('timed out')) {
      return {
        code: 'TIMEOUT',
        message: 'Request timed out',
        retryable: true,
        retryAfterMs: 5000,
      };
    }

    if (message.includes('network') || message.includes('econnrefused') || message.includes('fetch')) {
      return {
        code: 'NETWORK_ERROR',
        message: 'Network error - unable to reach OpenRouter',
        retryable: true,
        retryAfterMs: 5000,
      };
    }

    return {
      code: 'UNKNOWN',
      message: error.message,
      retryable: false,
    };
  }

  return {
    code: 'UNKNOWN',
    message: String(error),
    retryable: false,
  };
}

// ============================================================================
// Main Client Functions
// ============================================================================

/**
 * Make a basic LLM call to OpenRouter
 */
export async function callLLM(
  messages: LLMMessage[],
  options: LLMOptions = {}
): Promise<LLMResponse> {
  const model = options.model || (config.llm.primaryModel as SupportedModel);
  const timeout = options.timeout || DEFAULT_TIMEOUT;
  const startTime = Date.now();

  // Validate API key
  if (!config.llm.openrouterApiKey) {
    throw new LLMServiceError('OpenRouter', 'API key not configured', {
      operation: 'callLLM',
      suggestion: 'Set OPENROUTER_API_KEY environment variable',
    });
  }

  // Estimate tokens and check rate limit
  const estimatedInputTokens = estimateTokens(messages);
  checkRateLimit(estimatedInputTokens);

  const requestBody = {
    model,
    messages: messages.map(m => ({
      role: m.role,
      content: m.content,
      ...(m.name && { name: m.name }),
    })),
    temperature: options.temperature ?? 0.7,
    max_tokens: options.maxTokens ?? MODEL_CONFIGS[model]?.maxOutputTokens ?? 4096,
    top_p: options.topP,
    top_k: options.topK,
    frequency_penalty: options.frequencyPenalty,
    presence_penalty: options.presencePenalty,
    stop: options.stopSequences,
    ...(options.responseFormat === 'json' && {
      response_format: { type: 'json_object' },
    }),
  };

  log.debug('LLM request', {
    model,
    messageCount: messages.length,
    estimatedTokens: estimatedInputTokens,
    operation: 'callLLM',
  });

  let lastError: Error | null = null;
  let attempts = 0;

  while (attempts < MAX_RETRIES) {
    attempts++;

    try {
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), timeout);

      const response = await fetch(OPENROUTER_API_URL, {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${config.llm.openrouterApiKey}`,
          'Content-Type': 'application/json',
          'HTTP-Referer': 'https://adverant.ai',
          'X-Title': 'EE Design Partner',
        },
        body: JSON.stringify(requestBody),
        signal: controller.signal,
      });

      clearTimeout(timeoutId);

      // Handle rate limiting
      if (response.status === 429) {
        handleRateLimitResponse(response.headers.get('Retry-After'));
        const retryAfter = rateLimitState.retryAfter
          ? Math.ceil((rateLimitState.retryAfter - Date.now()) / 1000)
          : 60;

        if (attempts < MAX_RETRIES) {
          log.warn('Rate limited, retrying', { attempt: attempts, retryAfter });
          await sleep(retryAfter * 1000);
          continue;
        }

        throw new RateLimitError(retryAfter, { operation: 'callLLM', model });
      }

      // Handle other errors
      if (!response.ok) {
        const errorBody = await response.text();
        const errorDetails = parseError(new Error(errorBody), response.status);

        if (errorDetails.retryable && attempts < MAX_RETRIES) {
          log.warn('Retryable error, retrying', {
            attempt: attempts,
            code: errorDetails.code,
            message: errorDetails.message,
          });
          await sleep(RETRY_DELAY_BASE * Math.pow(2, attempts - 1));
          continue;
        }

        throw new LLMServiceError('OpenRouter', errorDetails.message, {
          operation: 'callLLM',
          model,
          statusCode: response.status,
          errorCode: errorDetails.code,
        });
      }

      const data = await response.json() as OpenRouterResponse;
      const latencyMs = Date.now() - startTime;

      // Extract response
      const choice = data.choices?.[0];
      if (!choice) {
        throw new LLMServiceError('OpenRouter', 'No response choice returned', {
          operation: 'callLLM',
          model,
        });
      }

      // Build usage information
      const usage: TokenUsage = {
        promptTokens: data.usage?.prompt_tokens || estimatedInputTokens,
        completionTokens: data.usage?.completion_tokens || 0,
        totalTokens: data.usage?.total_tokens || estimatedInputTokens,
        estimatedCost: 0,
      };
      usage.estimatedCost = calculateCost(usage, model);

      const llmResponse: LLMResponse = {
        id: data.id || `or-${Date.now()}`,
        model: data.model || model,
        content: choice.message?.content || '',
        finishReason: mapFinishReason(choice.finish_reason),
        usage,
        latencyMs,
      };

      log.info('LLM response received', {
        model,
        latencyMs,
        promptTokens: usage.promptTokens,
        completionTokens: usage.completionTokens,
        cost: usage.estimatedCost.toFixed(6),
        operation: 'callLLM',
      });

      return llmResponse;

    } catch (error) {
      lastError = error instanceof Error ? error : new Error(String(error));

      if (error instanceof LLMServiceError || error instanceof RateLimitError) {
        throw error;
      }

      const errorDetails = parseError(error);
      if (errorDetails.retryable && attempts < MAX_RETRIES) {
        log.warn('Request failed, retrying', {
          attempt: attempts,
          error: lastError.message,
        });
        await sleep(RETRY_DELAY_BASE * Math.pow(2, attempts - 1));
        continue;
      }

      throw new LLMServiceError('OpenRouter', lastError.message, {
        operation: 'callLLM',
        model,
        attempts,
      });
    }
  }

  throw new LLMServiceError(
    'OpenRouter',
    lastError?.message || 'Max retries exceeded',
    { operation: 'callLLM', model, attempts }
  );
}

/**
 * Make an LLM call with JSON validation and automatic retries
 */
export async function callLLMWithValidation<T>(
  messages: LLMMessage[],
  validator: (response: string) => T | null,
  options: LLMOptions = {},
  maxRetries: number = 3
): Promise<ValidatedResponse<T>> {
  let attempts = 0;
  let lastRaw = '';
  let lastError = '';

  // Ensure JSON response format
  const llmOptions: LLMOptions = {
    ...options,
    responseFormat: 'json',
  };

  while (attempts < maxRetries) {
    attempts++;

    try {
      const response = await callLLM(messages, llmOptions);
      lastRaw = response.content;

      // Try to parse and validate
      const validated = validator(response.content);

      if (validated !== null) {
        log.debug('LLM validation succeeded', {
          attempts,
          operation: 'callLLMWithValidation',
        });

        return {
          success: true,
          data: validated,
          raw: response.content,
          attempts,
        };
      }

      lastError = 'Validation returned null - response did not match expected schema';
      log.warn('LLM response validation failed', {
        attempt: attempts,
        error: lastError,
        operation: 'callLLMWithValidation',
      });

      // Add a hint to the next attempt
      if (attempts < maxRetries) {
        messages = [
          ...messages,
          {
            role: 'assistant',
            content: response.content,
          },
          {
            role: 'user',
            content: `The previous response could not be parsed correctly. Please ensure your response is valid JSON that matches the requested schema. Error: ${lastError}`,
          },
        ];
      }

    } catch (error) {
      lastError = error instanceof Error ? error.message : String(error);
      const errorObj = error instanceof Error ? error : new Error(lastError);
      log.error('LLM call failed during validation', errorObj, {
        attempt: attempts,
        operation: 'callLLMWithValidation',
      });

      if (error instanceof RateLimitError) {
        // Wait for rate limit to clear
        await sleep(error.retryAfter * 1000);
      } else if (attempts < maxRetries) {
        await sleep(RETRY_DELAY_BASE * attempts);
      }
    }
  }

  return {
    success: false,
    raw: lastRaw,
    attempts,
    error: lastError,
  };
}

/**
 * Make a streaming LLM call
 */
export async function callLLMStreaming(
  messages: LLMMessage[],
  options: LLMOptions = {},
  onChunk: StreamCallback
): Promise<LLMResponse> {
  const model = options.model || (config.llm.primaryModel as SupportedModel);
  const timeout = options.timeout || DEFAULT_TIMEOUT;
  const startTime = Date.now();

  // Validate API key
  if (!config.llm.openrouterApiKey) {
    throw new LLMServiceError('OpenRouter', 'API key not configured', {
      operation: 'callLLMStreaming',
      suggestion: 'Set OPENROUTER_API_KEY environment variable',
    });
  }

  // Estimate tokens and check rate limit
  const estimatedInputTokens = estimateTokens(messages);
  checkRateLimit(estimatedInputTokens);

  const requestBody = {
    model,
    messages: messages.map(m => ({
      role: m.role,
      content: m.content,
      ...(m.name && { name: m.name }),
    })),
    temperature: options.temperature ?? 0.7,
    max_tokens: options.maxTokens ?? MODEL_CONFIGS[model]?.maxOutputTokens ?? 4096,
    top_p: options.topP,
    stream: true,
  };

  log.debug('LLM streaming request', {
    model,
    messageCount: messages.length,
    operation: 'callLLMStreaming',
  });

  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), timeout);

  try {
    const response = await fetch(OPENROUTER_API_URL, {
      method: 'POST',
      headers: {
        'Authorization': `Bearer ${config.llm.openrouterApiKey}`,
        'Content-Type': 'application/json',
        'HTTP-Referer': 'https://adverant.ai',
        'X-Title': 'EE Design Partner',
      },
      body: JSON.stringify(requestBody),
      signal: controller.signal,
    });

    clearTimeout(timeoutId);

    if (!response.ok) {
      const errorBody = await response.text();
      throw new LLMServiceError('OpenRouter', errorBody, {
        operation: 'callLLMStreaming',
        model,
        statusCode: response.status,
      });
    }

    if (!response.body) {
      throw new LLMServiceError('OpenRouter', 'No response body for streaming', {
        operation: 'callLLMStreaming',
        model,
      });
    }

    // Process the stream
    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let fullContent = '';
    let responseId = `or-stream-${Date.now()}`;
    let finishReason: FinishReason = 'stop';
    let buffer = '';

    while (true) {
      const { done, value } = await reader.read();

      if (done) break;

      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split('\n');
      buffer = lines.pop() || '';

      for (const line of lines) {
        const trimmed = line.trim();
        if (!trimmed || trimmed === 'data: [DONE]') continue;

        if (trimmed.startsWith('data: ')) {
          try {
            const data = JSON.parse(trimmed.slice(6));

            if (data.id) responseId = data.id;

            const delta = data.choices?.[0]?.delta;
            if (delta?.content) {
              fullContent += delta.content;
              onChunk({
                id: responseId,
                content: delta.content,
              });
            }

            if (data.choices?.[0]?.finish_reason) {
              finishReason = mapFinishReason(data.choices[0].finish_reason);
            }
          } catch {
            // Skip malformed JSON chunks
            log.debug('Skipping malformed SSE chunk', { chunk: trimmed });
          }
        }
      }
    }

    const latencyMs = Date.now() - startTime;

    // Estimate output tokens
    const outputTokens = Math.ceil(fullContent.length * TOKENS_PER_CHAR_ESTIMATE);
    const usage: TokenUsage = {
      promptTokens: estimatedInputTokens,
      completionTokens: outputTokens,
      totalTokens: estimatedInputTokens + outputTokens,
      estimatedCost: 0,
    };
    usage.estimatedCost = calculateCost(usage, model);

    // Send final chunk with usage
    onChunk({
      id: responseId,
      content: '',
      finishReason,
      usage,
    });

    log.info('LLM streaming complete', {
      model,
      latencyMs,
      contentLength: fullContent.length,
      operation: 'callLLMStreaming',
    });

    return {
      id: responseId,
      model,
      content: fullContent,
      finishReason,
      usage,
      latencyMs,
    };

  } catch (error) {
    clearTimeout(timeoutId);

    if (error instanceof Error && error.name === 'AbortError') {
      throw new TimeoutError('callLLMStreaming', timeout, {
        operation: 'callLLMStreaming',
        model,
      });
    }

    throw error;
  }
}

// ============================================================================
// Utility Functions
// ============================================================================

function mapFinishReason(reason: string | undefined): FinishReason {
  switch (reason) {
    case 'stop':
    case 'end_turn':
      return 'stop';
    case 'length':
    case 'max_tokens':
      return 'length';
    case 'content_filter':
      return 'content_filter';
    default:
      return 'stop';
  }
}

function sleep(ms: number): Promise<void> {
  return new Promise(resolve => setTimeout(resolve, ms));
}

// ============================================================================
// Convenience Functions
// ============================================================================

/**
 * Quick call with Claude Opus 4.6 (highest quality)
 */
export async function callClaudeOpus(
  messages: LLMMessage[],
  options: Omit<LLMOptions, 'model'> = {}
): Promise<LLMResponse> {
  return callLLM(messages, { ...options, model: 'anthropic/claude-opus-4.6' });
}

/**
 * Quick call with Gemini 2.0 Flash (fast/cheap)
 */
export async function callGeminiFlash(
  messages: LLMMessage[],
  options: Omit<LLMOptions, 'model'> = {}
): Promise<LLMResponse> {
  return callLLM(messages, { ...options, model: 'google/gemini-2.0-flash' });
}

/**
 * Quick call with Gemini 2.5 Pro (validation)
 */
export async function callGeminiPro(
  messages: LLMMessage[],
  options: Omit<LLMOptions, 'model'> = {}
): Promise<LLMResponse> {
  return callLLM(messages, { ...options, model: 'google/gemini-2.5-pro' });
}

/**
 * Get available models
 */
export function getAvailableModels(): SupportedModel[] {
  return Object.keys(MODEL_CONFIGS) as SupportedModel[];
}

/**
 * Get model config
 */
export function getModelConfig(model: SupportedModel) {
  return MODEL_CONFIGS[model];
}

export default {
  callLLM,
  callLLMWithValidation,
  callLLMStreaming,
  callClaudeOpus,
  callClaudeOpus,
  callGeminiFlash,
  callGeminiPro,
  estimateTokens,
  calculateCost,
  getAvailableModels,
  getModelConfig,
};
