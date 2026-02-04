/**
 * Base WebSocket Manager - Standard Pattern for All EE Design Services
 *
 * This is the reusable base class for WebSocket streaming across all
 * EE design operations (schematic, PCB, simulation, firmware, etc.)
 *
 * Pattern Usage:
 *   1. Create event types enum (e.g., PCBEventType)
 *   2. Create progress event interface extending BaseProgressEvent
 *   3. Create manager class extending BaseWebSocketManager
 *   4. Register namespace in index.ts
 *   5. Create frontend hook using useEEDesignProgress pattern
 *
 * Example:
 *   class PCBWebSocketManager extends BaseWebSocketManager<PCBProgressEvent> {
 *     constructor(io: SocketIOServer) {
 *       super(io, '/pcb-layout', 'pcb_layout');
 *     }
 *   }
 */

import { Server as SocketIOServer, Namespace, Socket } from 'socket.io';
import { EventEmitter } from 'events';
import { v4 as uuidv4 } from 'uuid';
import { log } from '../utils/logger.js';

/**
 * Base progress event interface.
 * All service-specific events should extend this.
 */
export interface BaseProgressEvent {
  /** Event type (e.g., 'phase_start', 'complete', 'error') */
  type: string;

  /** Unique operation ID */
  operationId: string;

  /** ISO timestamp */
  timestamp: string;

  /** Overall progress 0-100 */
  progress_percentage: number;

  /** Human-readable message for current step */
  current_step: string;

  /** Current phase name */
  phase?: string;

  /** Progress within current phase (0-100) */
  phase_progress?: number;

  /** Additional data */
  data?: Record<string, unknown>;
}

/**
 * Operation tracking structure
 */
export interface Operation {
  /** Unique operation ID */
  operationId: string;

  /** Project ID this operation belongs to */
  projectId: string;

  /** Type of operation (e.g., 'schematic', 'pcb_layout') */
  operationType: string;

  /** When the operation started */
  startedAt: Date;

  /** Current status */
  status: 'running' | 'complete' | 'error';

  /** Most recent event */
  lastEvent?: BaseProgressEvent;

  /** Total events emitted */
  eventCount: number;

  /** Organization ID for multi-tenant support */
  organizationId?: string;
}

/**
 * Base WebSocket Manager
 *
 * Provides standard functionality for all EE design streaming:
 * - Operation tracking with unique IDs
 * - Room-based subscriptions
 * - Progress event broadcasting
 * - Heartbeat support
 * - Auto-cleanup of completed operations
 * - EventEmitter for internal listeners
 */
export abstract class BaseWebSocketManager<T extends BaseProgressEvent> extends EventEmitter {
  protected namespace: Namespace;
  protected operations: Map<string, Operation> = new Map();
  protected namespacePath: string;
  protected operationType: string;

  constructor(io: SocketIOServer, namespacePath: string, operationType: string) {
    super();
    this.namespacePath = namespacePath;
    this.operationType = operationType;
    this.namespace = io.of(namespacePath);
    this.setupNamespace();

    log.info(`${operationType} WebSocket manager initialized`, { namespace: namespacePath });
  }

  /**
   * Setup namespace event handlers
   */
  protected setupNamespace(): void {
    this.namespace.on('connection', (socket: Socket) => {
      log.info(`Client connected to ${this.namespacePath}`, { socketId: socket.id });

      // Subscribe to specific operation
      socket.on('subscribe:operation', (operationId: string) => {
        socket.join(`operation:${operationId}`);
        log.debug(`Client subscribed to operation`, {
          socketId: socket.id,
          operationId,
          namespace: this.namespacePath,
        });

        // Send current status if operation exists
        const operation = this.operations.get(operationId);
        if (operation?.lastEvent) {
          socket.emit('progress', operation.lastEvent);
        }
      });

      // Unsubscribe from operation
      socket.on('unsubscribe:operation', (operationId: string) => {
        socket.leave(`operation:${operationId}`);
        log.debug(`Client unsubscribed from operation`, {
          socketId: socket.id,
          operationId,
        });
      });

      // List active operations (for debugging/admin)
      socket.on('list:operations', () => {
        const ops = Array.from(this.operations.values()).map((op) => ({
          operationId: op.operationId,
          projectId: op.projectId,
          operationType: op.operationType,
          status: op.status,
          startedAt: op.startedAt,
          eventCount: op.eventCount,
          lastProgress: op.lastEvent?.progress_percentage,
          lastStep: op.lastEvent?.current_step,
        }));
        socket.emit('operations:list', ops);
      });

      // Heartbeat
      socket.on('heartbeat', () => {
        socket.emit('heartbeat:ack', { timestamp: new Date().toISOString() });
      });

      // Disconnect handler
      socket.on('disconnect', () => {
        log.info(`Client disconnected from ${this.namespacePath}`, { socketId: socket.id });
      });
    });
  }

  /**
   * Create a new operation and return its ID.
   * Call this when starting a long-running operation.
   */
  createOperation(projectId: string, organizationId?: string): string {
    const operationId = uuidv4();
    const operation: Operation = {
      operationId,
      projectId,
      organizationId,
      operationType: this.operationType,
      startedAt: new Date(),
      status: 'running',
      eventCount: 0,
    };

    this.operations.set(operationId, operation);
    log.info(`Created ${this.operationType} operation`, { operationId, projectId, organizationId });

    return operationId;
  }

  /**
   * Emit a progress event to all subscribers of an operation.
   */
  emitProgress(operationId: string, event: T): void {
    const operation = this.operations.get(operationId);
    if (!operation) {
      log.warn(`Emitting to unknown operation`, { operationId, namespace: this.namespacePath });
      return;
    }

    // Update operation tracking
    operation.lastEvent = event;
    operation.eventCount++;

    // Update status based on event type
    if (event.type === 'complete') {
      operation.status = 'complete';
    } else if (event.type === 'error') {
      operation.status = 'error';
    }

    // Emit to all subscribers
    this.namespace.to(`operation:${operationId}`).emit('progress', event);

    // Also emit on EventEmitter for internal listeners
    this.emit('progress', operationId, event);

    log.debug(`Emitted ${this.operationType} progress event`, {
      operationId,
      type: event.type,
      progress: event.progress_percentage,
      step: event.current_step?.substring(0, 50),
    });
  }

  /**
   * Emit a simple progress update (convenience method)
   */
  emitSimpleProgress(
    operationId: string,
    type: string,
    progress: number,
    message: string,
    phase?: string,
    extra?: Partial<T>
  ): void {
    const event = {
      type,
      operationId,
      timestamp: new Date().toISOString(),
      progress_percentage: Math.min(100, Math.max(0, progress)),
      current_step: message,
      phase,
      ...extra,
    } as T;

    this.emitProgress(operationId, event);
  }

  /**
   * Complete an operation with success
   */
  completeOperation(operationId: string, result: Record<string, unknown>): void {
    const event = {
      type: 'complete',
      operationId,
      timestamp: new Date().toISOString(),
      progress_percentage: 100,
      current_step: `${this.operationType} complete!`,
      data: { result },
    } as T;

    this.emitProgress(operationId, event);

    // Clean up after delay (keep for reconnection)
    setTimeout(() => {
      this.operations.delete(operationId);
      log.debug(`Cleaned up completed operation`, { operationId });
    }, 60000); // Keep for 1 minute
  }

  /**
   * Fail an operation with error
   */
  failOperation(
    operationId: string,
    error: Error | string,
    partialResult?: Record<string, unknown>
  ): void {
    const errorMessage = error instanceof Error ? error.message : error;
    const event = {
      type: 'error',
      operationId,
      timestamp: new Date().toISOString(),
      progress_percentage: this.operations.get(operationId)?.lastEvent?.progress_percentage || 0,
      current_step: `Error: ${errorMessage}`,
      data: {
        error_message: errorMessage,
        error_code: 'OPERATION_FAILED',
        partial_result: partialResult,
      },
    } as T;

    this.emitProgress(operationId, event);

    // Clean up after delay
    setTimeout(() => {
      this.operations.delete(operationId);
    }, 60000);
  }

  /**
   * Get operation status
   */
  getOperation(operationId: string): Operation | undefined {
    return this.operations.get(operationId);
  }

  /**
   * Get all active operations for a project
   */
  getProjectOperations(projectId: string): Operation[] {
    return Array.from(this.operations.values()).filter((op) => op.projectId === projectId);
  }

  /**
   * Get all active operations for an organization
   */
  getOrganizationOperations(organizationId: string): Operation[] {
    return Array.from(this.operations.values()).filter((op) => op.organizationId === organizationId);
  }

  /**
   * Get statistics
   */
  getStats(): {
    activeOperations: number;
    connectedClients: number;
    operationType: string;
    namespace: string;
  } {
    return {
      activeOperations: this.operations.size,
      connectedClients: this.namespace.sockets.size,
      operationType: this.operationType,
      namespace: this.namespacePath,
    };
  }

  /**
   * Get namespace path
   */
  getNamespacePath(): string {
    return this.namespacePath;
  }
}

/**
 * Parse Python PROGRESS: lines from stdout and emit as WebSocket events.
 *
 * Python format: PROGRESS:{"type":"symbol_resolving","progress_percentage":10,...}
 *
 * Usage:
 *   proc.stdout.on('data', (data) => {
 *     for (const line of data.toString().split('\n')) {
 *       parsePythonProgressLine(line, operationId, manager);
 *     }
 *   });
 */
export function parsePythonProgressLine<T extends BaseProgressEvent>(
  line: string,
  operationId: string,
  manager: BaseWebSocketManager<T>
): boolean {
  if (!line.startsWith('PROGRESS:')) {
    return false;
  }

  try {
    const jsonStr = line.slice(9).trim();
    const eventData = JSON.parse(jsonStr) as Partial<T>;

    // Ensure required fields
    const event = {
      type: eventData.type || 'progress_update',
      operationId,
      timestamp: eventData.timestamp || new Date().toISOString(),
      progress_percentage: eventData.progress_percentage || 0,
      current_step: eventData.current_step || 'Processing...',
      ...eventData,
    } as T;

    manager.emitProgress(operationId, event);
    return true;
  } catch (error) {
    log.warn('Failed to parse Python progress line', { line: line.substring(0, 100), error });
    return false;
  }
}

/**
 * Create a progress emitter function for use in Python stdout streaming.
 *
 * Usage:
 *   const emitter = createProgressEmitter(operationId, manager);
 *   proc.stdout.on('data', emitter);
 */
export function createProgressEmitter<T extends BaseProgressEvent>(
  operationId: string,
  manager: BaseWebSocketManager<T>
): (data: Buffer | string) => void {
  return (data: Buffer | string) => {
    const lines = data.toString().split('\n');
    for (const line of lines) {
      if (line.trim()) {
        parsePythonProgressLine(line, operationId, manager);
      }
    }
  };
}
