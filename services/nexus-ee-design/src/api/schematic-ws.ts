/**
 * Schematic Generation WebSocket Namespace
 *
 * Handles real-time streaming of schematic generation progress to clients.
 * This is the first implementation of the standard MAPO workflow streaming pattern.
 *
 * Pattern Usage (for other workflows like PCB, simulation):
 * 1. Create event types file (websocket-{workflow}.ts)
 * 2. Create WS namespace file ({workflow}-ws.ts)
 * 3. Register namespace in index.ts
 * 4. Add progress callbacks to Python pipeline
 * 5. Create frontend hook (use{Workflow}Progress.ts)
 */

import { Server as SocketIOServer, Namespace, Socket } from 'socket.io';
import { EventEmitter } from 'events';
import { v4 as uuidv4 } from 'uuid';
import {
  SchematicProgressEvent,
  SchematicEventType,
  SchematicPhase,
  createProgressEvent,
} from './websocket-schematic.js';
import { log } from '../utils/logger.js';

/**
 * Operation tracking for in-flight schematic generations
 */
interface SchematicOperation {
  operationId: string;
  projectId: string;
  startedAt: Date;
  status: 'running' | 'complete' | 'error';
  lastEvent?: SchematicProgressEvent;
  eventCount: number;
  completedPhases: string[];                    // Track which phases finished
  eventHistory: SchematicProgressEvent[];       // Last 50 events for reconnect replay
}

const MAX_EVENT_HISTORY = 50;

/**
 * Schematic WebSocket Manager
 *
 * Manages WebSocket connections and event streaming for schematic generation.
 * Can be used as a template for other MAPO workflow managers.
 */
export class SchematicWebSocketManager extends EventEmitter {
  private namespace: Namespace;
  private operations: Map<string, SchematicOperation> = new Map();

  constructor(io: SocketIOServer) {
    super();

    // Create /schematic namespace
    this.namespace = io.of('/schematic');

    this.setupNamespace();
    log.info('SchematicWebSocketManager initialized');
  }

  private setupNamespace(): void {
    this.namespace.on('connection', (socket: Socket) => {
      log.info('Client connected to /schematic namespace', { socketId: socket.id });

      // Subscribe to specific operation
      socket.on('subscribe:operation', (operationId: string) => {
        socket.join(`operation:${operationId}`);
        log.debug('Client subscribed to operation', { socketId: socket.id, operationId });

        // Send full operation state on reconnect (not just last event)
        const operation = this.operations.get(operationId);
        if (operation) {
          socket.emit('operation:state', {
            operationId: operation.operationId,
            status: operation.status,
            completedPhases: operation.completedPhases,
            eventHistory: operation.eventHistory,
            lastEvent: operation.lastEvent,
            eventCount: operation.eventCount,
            startedAt: operation.startedAt.toISOString(),
          });
        }
      });

      // Unsubscribe from operation
      socket.on('unsubscribe:operation', (operationId: string) => {
        socket.leave(`operation:${operationId}`);
        log.debug('Client unsubscribed from operation', { socketId: socket.id, operationId });
      });

      // List active operations (for debugging/admin)
      socket.on('list:operations', () => {
        const ops = Array.from(this.operations.values()).map((op) => ({
          operationId: op.operationId,
          projectId: op.projectId,
          status: op.status,
          startedAt: op.startedAt,
          eventCount: op.eventCount,
          lastProgress: op.lastEvent?.progress_percentage,
        }));
        socket.emit('operations:list', ops);
      });

      // Heartbeat
      socket.on('heartbeat', () => {
        socket.emit('heartbeat:ack', { timestamp: new Date().toISOString() });
      });

      socket.on('disconnect', () => {
        log.info('Client disconnected from /schematic namespace', { socketId: socket.id });
      });
    });
  }

  /**
   * Create a new operation and return its ID.
   * Call this when starting schematic generation.
   */
  createOperation(projectId: string, existingOperationId?: string): string {
    const operationId = existingOperationId || uuidv4();
    const operation: SchematicOperation = {
      operationId,
      projectId,
      startedAt: new Date(),
      status: 'running',
      eventCount: 0,
      completedPhases: [],
      eventHistory: [],
    };

    this.operations.set(operationId, operation);
    log.info('Created schematic operation', { operationId, projectId });

    return operationId;
  }

  /**
   * Emit a progress event to all subscribers of an operation.
   */
  emitProgress(operationId: string, event: SchematicProgressEvent): void {
    const operation = this.operations.get(operationId);
    if (!operation) {
      log.warn('Emitting to unknown operation', { operationId });
      return;
    }

    // Update operation tracking
    operation.lastEvent = event;
    operation.eventCount++;

    // Maintain capped event history for reconnection replay
    operation.eventHistory.push(event);
    if (operation.eventHistory.length > MAX_EVENT_HISTORY) {
      operation.eventHistory.shift();
    }

    // Track completed phases
    const eventTypeStr = String(event.type);
    if (eventTypeStr === SchematicEventType.PHASE_COMPLETE || eventTypeStr.includes('complete')) {
      const phase = (event as any).phase;
      if (phase && !operation.completedPhases.includes(phase)) {
        operation.completedPhases.push(phase);
      }
    }

    if (event.type === SchematicEventType.COMPLETE) {
      operation.status = 'complete';
    } else if (event.type === SchematicEventType.ERROR) {
      operation.status = 'error';
    }

    // Emit to all subscribers
    this.namespace.to(`operation:${operationId}`).emit('progress', event);

    // Also emit on EventEmitter for internal listeners
    this.emit('progress', operationId, event);

    log.debug('Emitted progress event', {
      operationId,
      type: event.type,
      progress: event.progress_percentage,
    });
  }

  /**
   * Emit a simple progress update (convenience method)
   */
  emitSimpleProgress(
    operationId: string,
    type: SchematicEventType,
    progress: number,
    message: string,
    phase?: SchematicPhase,
    extra?: Partial<SchematicProgressEvent>
  ): void {
    const event = createProgressEvent(operationId, type, progress, message, {
      phase,
      ...extra,
    });
    this.emitProgress(operationId, event);
  }

  /**
   * Complete an operation with success
   */
  completeOperation(
    operationId: string,
    result: {
      schematicId?: string;
      componentCount: number;
      connectionCount: number;
      wireCount: number;
      validationScore?: number;
      smokeTestPassed?: boolean;
    }
  ): void {
    const event: SchematicProgressEvent = {
      type: SchematicEventType.COMPLETE,
      operationId,
      timestamp: new Date().toISOString(),
      progress_percentage: 100,
      current_step: 'Schematic generation complete!',
      phase: 'export',
      // Flatten the result so frontend can access schematicId directly
      data: {
        schematicId: result.schematicId,
        componentCount: result.componentCount,
        connectionCount: result.connectionCount,
        wireCount: result.wireCount,
        validationScore: result.validationScore,
        smokeTestPassed: result.smokeTestPassed,
      },
    };

    this.emitProgress(operationId, event);

    // Clean up after delay (keep for reconnection)
    setTimeout(() => {
      this.operations.delete(operationId);
      log.debug('Cleaned up completed operation', { operationId });
    }, 60000); // Keep for 1 minute
  }

  /**
   * Fail an operation with error
   */
  failOperation(
    operationId: string,
    error: Error | string,
    partialResult?: {
      symbolsResolved?: number;
      connectionsGenerated?: number;
      lastSuccessfulPhase?: SchematicPhase;
    }
  ): void {
    const errorMessage = error instanceof Error ? error.message : error;
    const event: SchematicProgressEvent = {
      type: SchematicEventType.ERROR,
      operationId,
      timestamp: new Date().toISOString(),
      progress_percentage: this.operations.get(operationId)?.lastEvent?.progress_percentage || 0,
      current_step: `Error: ${errorMessage}`,
      error_message: errorMessage,
      error_code: 'GENERATION_FAILED',
      data: { partial_result: partialResult },
    };

    this.emitProgress(operationId, event);

    // Clean up after delay
    setTimeout(() => {
      this.operations.delete(operationId);
    }, 60000);
  }

  /**
   * Get operation status
   */
  getOperation(operationId: string): SchematicOperation | undefined {
    return this.operations.get(operationId);
  }

  /**
   * Get all active operations for a project
   */
  getProjectOperations(projectId: string): SchematicOperation[] {
    return Array.from(this.operations.values()).filter((op) => op.projectId === projectId);
  }

  /**
   * Get statistics
   */
  getStats(): { activeOperations: number; connectedClients: number } {
    return {
      activeOperations: this.operations.size,
      connectedClients: this.namespace.sockets.size,
    };
  }

  /**
   * Get namespace path (for logging and debugging)
   */
  getNamespacePath(): string {
    return '/schematic';
  }
}

// Singleton instance
let schematicWsManager: SchematicWebSocketManager | null = null;

/**
 * Initialize the schematic WebSocket manager.
 * Call this from index.ts after creating the SocketIO server.
 */
export function initSchematicWebSocket(io: SocketIOServer): SchematicWebSocketManager {
  if (!schematicWsManager) {
    schematicWsManager = new SchematicWebSocketManager(io);
  }
  return schematicWsManager;
}

/**
 * Get the schematic WebSocket manager instance.
 * Throws if not initialized.
 */
export function getSchematicWsManager(): SchematicWebSocketManager {
  if (!schematicWsManager) {
    throw new Error('SchematicWebSocketManager not initialized. Call initSchematicWebSocket first.');
  }
  return schematicWsManager;
}

/**
 * Parse Python PROGRESS: lines from stdout and emit as WebSocket events.
 *
 * Python format: PROGRESS:{"type":"symbol_resolving","progress_percentage":10,...}
 */
export function parsePythonProgressLine(
  line: string,
  operationId: string,
  manager: SchematicWebSocketManager
): boolean {
  if (!line.startsWith('PROGRESS:')) {
    return false;
  }

  try {
    const jsonStr = line.slice(9).trim();
    const eventData = JSON.parse(jsonStr) as Partial<SchematicProgressEvent>;

    // Add operationId and timestamp if not present
    const event: SchematicProgressEvent = {
      type: eventData.type || SchematicEventType.PHASE_START,
      operationId,
      timestamp: eventData.timestamp || new Date().toISOString(),
      progress_percentage: eventData.progress_percentage || 0,
      current_step: eventData.current_step || 'Processing...',
      ...eventData,
    };

    manager.emitProgress(operationId, event);
    return true;
  } catch (error) {
    log.warn('Failed to parse Python progress line', { line, error });
    return false;
  }
}
