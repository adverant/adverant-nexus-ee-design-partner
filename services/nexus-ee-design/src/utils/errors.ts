/**
 * EE Design Partner - Custom Error Classes
 *
 * Structured error handling with full context for debugging
 */

export interface ErrorContext {
  operation: string;
  input?: unknown;
  timestamp: Date;
  requestId?: string;
  suggestion?: string;
  [key: string]: unknown;
}

export class EEDesignError extends Error {
  public readonly code: string;
  public readonly statusCode: number;
  public readonly context: ErrorContext;
  public readonly isOperational: boolean;

  constructor(
    message: string,
    code: string,
    statusCode: number,
    context?: Partial<ErrorContext>,
    isOperational = true
  ) {
    super(message);
    this.name = this.constructor.name;
    this.code = code;
    this.statusCode = statusCode;
    this.context = {
      operation: context?.operation || 'unknown',
      timestamp: new Date(),
      ...(context || {}),
    };
    this.isOperational = isOperational;
    Error.captureStackTrace(this, this.constructor);
  }

  toJSON(): Record<string, unknown> {
    return {
      name: this.name,
      message: this.message,
      code: this.code,
      statusCode: this.statusCode,
      context: this.context,
    };
  }
}

// Validation Errors (400)
export class ValidationError extends EEDesignError {
  constructor(message: string, context?: Partial<ErrorContext>) {
    super(message, 'VALIDATION_ERROR', 400, context);
  }
}

export class SchematicValidationError extends EEDesignError {
  constructor(message: string, violations: unknown[], context: Partial<ErrorContext>) {
    super(message, 'SCHEMATIC_VALIDATION_ERROR', 400, { ...context, violations });
  }
}

export class PCBValidationError extends EEDesignError {
  public readonly drcViolations: unknown[];
  public readonly ercViolations: unknown[];

  constructor(
    message: string,
    drcViolations: unknown[],
    ercViolations: unknown[],
    context: Partial<ErrorContext>
  ) {
    super(message, 'PCB_VALIDATION_ERROR', 400, {
      ...context,
      drcViolations,
      ercViolations,
    });
    this.drcViolations = drcViolations;
    this.ercViolations = ercViolations;
  }
}

// Not Found Errors (404)
export class NotFoundError extends EEDesignError {
  constructor(resource: string, identifier: string, context: Partial<ErrorContext>) {
    super(
      `${resource} not found: ${identifier}`,
      'NOT_FOUND',
      404,
      { ...context, resource, identifier }
    );
  }
}

export class ProjectNotFoundError extends NotFoundError {
  constructor(projectId: string, context: Partial<ErrorContext>) {
    super('Project', projectId, context);
  }
}

export class SchematicNotFoundError extends NotFoundError {
  constructor(schematicId: string, context: Partial<ErrorContext>) {
    super('Schematic', schematicId, context);
  }
}

export class LayoutNotFoundError extends NotFoundError {
  constructor(layoutId: string, context: Partial<ErrorContext>) {
    super('PCB Layout', layoutId, context);
  }
}

// Authorization Errors (401, 403)
export class AuthenticationError extends EEDesignError {
  constructor(message: string, context: Partial<ErrorContext>) {
    super(message, 'AUTHENTICATION_ERROR', 401, context);
  }
}

export class AuthorizationError extends EEDesignError {
  constructor(message: string, context: Partial<ErrorContext>) {
    super(message, 'AUTHORIZATION_ERROR', 403, context);
  }
}

// Conflict Errors (409)
export class ConflictError extends EEDesignError {
  constructor(message: string, context: Partial<ErrorContext>) {
    super(message, 'CONFLICT_ERROR', 409, context);
  }
}

// Processing Errors (422)
export class ProcessingError extends EEDesignError {
  constructor(message: string, context: Partial<ErrorContext>) {
    super(message, 'PROCESSING_ERROR', 422, context);
  }
}

export class SimulationError extends EEDesignError {
  constructor(
    message: string,
    simulationType: string,
    context: Partial<ErrorContext>
  ) {
    super(message, 'SIMULATION_ERROR', 422, { ...context, simulationType });
  }
}

export class LayoutGenerationError extends EEDesignError {
  constructor(
    message: string,
    agentStrategy: string,
    context: Partial<ErrorContext>
  ) {
    super(message, 'LAYOUT_GENERATION_ERROR', 422, { ...context, agentStrategy });
  }
}

export class FirmwareGenerationError extends EEDesignError {
  constructor(
    message: string,
    targetMcu: string,
    context: Partial<ErrorContext>
  ) {
    super(message, 'FIRMWARE_GENERATION_ERROR', 422, { ...context, targetMcu });
  }
}

// External Service Errors (502, 503)
export class ExternalServiceError extends EEDesignError {
  constructor(
    serviceName: string,
    message: string,
    context: Partial<ErrorContext>
  ) {
    super(
      `External service error (${serviceName}): ${message}`,
      'EXTERNAL_SERVICE_ERROR',
      502,
      { ...context, serviceName }
    );
  }
}

export class LLMServiceError extends ExternalServiceError {
  constructor(provider: string, message: string, context: Partial<ErrorContext>) {
    super(`LLM/${provider}`, message, context);
  }
}

export class KiCadError extends ExternalServiceError {
  constructor(message: string, context: Partial<ErrorContext>) {
    super('KiCad', message, context);
  }
}

export class VendorApiError extends ExternalServiceError {
  constructor(vendor: string, message: string, context: Partial<ErrorContext>) {
    super(`Vendor/${vendor}`, message, context);
  }
}

// Rate Limit Errors (429)
export class RateLimitError extends EEDesignError {
  public readonly retryAfter: number;

  constructor(retryAfter: number, context: Partial<ErrorContext>) {
    super(
      `Rate limit exceeded. Retry after ${retryAfter} seconds`,
      'RATE_LIMIT_ERROR',
      429,
      { ...context, retryAfter }
    );
    this.retryAfter = retryAfter;
  }
}

// Timeout Errors (504)
export class TimeoutError extends EEDesignError {
  constructor(operation: string, timeoutMs: number, context: Partial<ErrorContext>) {
    super(
      `Operation timed out after ${timeoutMs}ms: ${operation}`,
      'TIMEOUT_ERROR',
      504,
      { ...context, operation, timeoutMs }
    );
  }
}

// Internal Server Errors (500)
export class InternalError extends EEDesignError {
  constructor(message: string, context: Partial<ErrorContext>) {
    super(message, 'INTERNAL_ERROR', 500, context, false);
  }
}

// Error type guard
export function isEEDesignError(error: unknown): error is EEDesignError {
  return error instanceof EEDesignError;
}

// Error handler helper
export function handleError(error: unknown): EEDesignError {
  if (isEEDesignError(error)) {
    return error;
  }

  if (error instanceof Error) {
    return new InternalError(error.message, {
      operation: 'unknown',
      originalError: error.name,
      stack: error.stack,
    });
  }

  return new InternalError('An unexpected error occurred', {
    operation: 'unknown',
    originalError: String(error),
  });
}