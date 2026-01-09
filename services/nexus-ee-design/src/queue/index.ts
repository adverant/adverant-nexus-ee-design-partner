/**
 * EE Design Partner - Queue System
 *
 * Main entry point for the BullMQ queue system.
 * Exports queues, workers, and management functions.
 */

import { Worker } from 'bullmq';
import { log } from '../utils/logger.js';

// Queue Manager and types
export {
  queueManager,
  getSimulationQueue,
  getLayoutQueue,
  getFirmwareQueue,
  getValidationQueue,
  addSimulationJob,
  addLayoutJob,
  addFirmwareJob,
  addValidationJob,
  type QueueName,
  type SimulationJobData,
  type LayoutJobData,
  type FirmwareJobData,
  type ValidationJobData,
  type JobData,
  type QueueProgress,
  type QueueResult,
} from './queue-manager.js';

// Workers
export {
  createSimulationWorker,
  getSimulationWorker,
  closeSimulationWorker,
} from './workers/simulation-worker.js';

export {
  createLayoutWorker,
  getLayoutWorker,
  closeLayoutWorker,
} from './workers/layout-worker.js';

export {
  createFirmwareWorker,
  getFirmwareWorker,
  closeFirmwareWorker,
} from './workers/firmware-worker.js';

// Import for internal use
import { queueManager } from './queue-manager.js';
import { createSimulationWorker, closeSimulationWorker } from './workers/simulation-worker.js';
import { createLayoutWorker, closeLayoutWorker } from './workers/layout-worker.js';
import { createFirmwareWorker, closeFirmwareWorker } from './workers/firmware-worker.js';

// ============================================================================
// Worker Management
// ============================================================================

interface WorkerRegistry {
  simulation: Worker | null;
  layout: Worker | null;
  firmware: Worker | null;
  validation: Worker | null;
}

const workers: WorkerRegistry = {
  simulation: null,
  layout: null,
  firmware: null,
  validation: null,
};

/**
 * Initialize the queue system.
 * Sets up queues and queue events.
 */
export async function initializeQueues(): Promise<void> {
  const logger = log.child({ service: 'queue-init' });

  logger.info('Initializing queue system');

  try {
    await queueManager.initialize();
    logger.info('Queue system initialized successfully');
  } catch (error) {
    const err = error instanceof Error ? error : new Error(String(error));
    logger.error('Failed to initialize queue system', err);
    throw error;
  }
}

/**
 * Start all workers.
 * Creates and starts workers for all queue types.
 */
export async function startWorkers(options?: {
  simulation?: boolean;
  layout?: boolean;
  firmware?: boolean;
  validation?: boolean;
}): Promise<void> {
  const logger = log.child({ service: 'worker-start' });

  const opts = {
    simulation: true,
    layout: true,
    firmware: true,
    validation: false, // Validation worker not yet implemented
    ...options,
  };

  logger.info('Starting workers', opts);

  // Ensure queues are initialized
  if (!queueManager.isInitialized()) {
    await initializeQueues();
  }

  try {
    // Start simulation worker
    if (opts.simulation) {
      workers.simulation = createSimulationWorker();
      logger.info('Simulation worker started');
    }

    // Start layout worker
    if (opts.layout) {
      workers.layout = createLayoutWorker();
      logger.info('Layout worker started');
    }

    // Start firmware worker
    if (opts.firmware) {
      workers.firmware = createFirmwareWorker();
      logger.info('Firmware worker started');
    }

    // Validation worker would be started here when implemented
    if (opts.validation) {
      logger.warn('Validation worker not yet implemented');
    }

    logger.info('All requested workers started', {
      simulation: !!workers.simulation,
      layout: !!workers.layout,
      firmware: !!workers.firmware,
      validation: !!workers.validation,
    });
  } catch (error) {
    const err = error instanceof Error ? error : new Error(String(error));
    logger.error('Failed to start workers', err);
    throw error;
  }
}

/**
 * Stop all workers gracefully.
 */
export async function stopWorkers(): Promise<void> {
  const logger = log.child({ service: 'worker-stop' });

  logger.info('Stopping workers');

  const closeTasks: Promise<void>[] = [];

  if (workers.simulation) {
    closeTasks.push(closeSimulationWorker());
    workers.simulation = null;
  }

  if (workers.layout) {
    closeTasks.push(closeLayoutWorker());
    workers.layout = null;
  }

  if (workers.firmware) {
    closeTasks.push(closeFirmwareWorker());
    workers.firmware = null;
  }

  await Promise.all(closeTasks);

  logger.info('All workers stopped');
}

/**
 * Shutdown the entire queue system.
 * Stops workers and closes queue connections.
 */
export async function shutdownQueues(): Promise<void> {
  const logger = log.child({ service: 'queue-shutdown' });

  logger.info('Shutting down queue system');

  // Stop workers first
  await stopWorkers();

  // Close queue manager connections
  await queueManager.close();

  logger.info('Queue system shut down');
}

/**
 * Get the status of all workers.
 */
export function getWorkersStatus(): Record<keyof WorkerRegistry, boolean> {
  return {
    simulation: workers.simulation !== null,
    layout: workers.layout !== null,
    firmware: workers.firmware !== null,
    validation: workers.validation !== null,
  };
}

/**
 * Get health status of the queue system.
 */
export async function getQueueHealth(): Promise<{
  initialized: boolean;
  queues: Record<string, {
    waiting: number;
    active: number;
    completed: number;
    failed: number;
    delayed: number;
    paused: number;
  }>;
  workers: Record<string, boolean>;
}> {
  const initialized = queueManager.isInitialized();

  if (!initialized) {
    return {
      initialized: false,
      queues: {},
      workers: getWorkersStatus(),
    };
  }

  const queues = await queueManager.getAllQueueStats();

  return {
    initialized: true,
    queues,
    workers: getWorkersStatus(),
  };
}

// ============================================================================
// Event Forwarding for Socket.IO
// ============================================================================

/**
 * Subscribe to queue events for real-time updates.
 * Returns an unsubscribe function.
 */
export function subscribeToQueueEvents(
  handlers: {
    onJobCompleted?: (data: { queueName: string; jobId: string; result: unknown }) => void;
    onJobFailed?: (data: { queueName: string; jobId: string; error: string }) => void;
    onJobProgress?: (data: { queueName: string; jobId: string; progress: number; message?: string }) => void;
    onJobActive?: (data: { queueName: string; jobId: string }) => void;
    onJobWaiting?: (data: { queueName: string; jobId: string }) => void;
    onJobStalled?: (data: { queueName: string; jobId: string }) => void;
  }
): () => void {
  if (handlers.onJobCompleted) {
    queueManager.on('job:completed', handlers.onJobCompleted);
  }
  if (handlers.onJobFailed) {
    queueManager.on('job:failed', handlers.onJobFailed);
  }
  if (handlers.onJobProgress) {
    queueManager.on('job:progress', handlers.onJobProgress);
  }
  if (handlers.onJobActive) {
    queueManager.on('job:active', handlers.onJobActive);
  }
  if (handlers.onJobWaiting) {
    queueManager.on('job:waiting', handlers.onJobWaiting);
  }
  if (handlers.onJobStalled) {
    queueManager.on('job:stalled', handlers.onJobStalled);
  }

  // Return unsubscribe function
  return () => {
    if (handlers.onJobCompleted) {
      queueManager.off('job:completed', handlers.onJobCompleted);
    }
    if (handlers.onJobFailed) {
      queueManager.off('job:failed', handlers.onJobFailed);
    }
    if (handlers.onJobProgress) {
      queueManager.off('job:progress', handlers.onJobProgress);
    }
    if (handlers.onJobActive) {
      queueManager.off('job:active', handlers.onJobActive);
    }
    if (handlers.onJobWaiting) {
      queueManager.off('job:waiting', handlers.onJobWaiting);
    }
    if (handlers.onJobStalled) {
      queueManager.off('job:stalled', handlers.onJobStalled);
    }
  };
}

// ============================================================================
// Graceful Shutdown Handler
// ============================================================================

/**
 * Register signal handlers for graceful shutdown.
 * Call this in your main application entry point.
 */
export function registerShutdownHandlers(): void {
  const logger = log.child({ service: 'shutdown-handler' });

  const shutdown = async (signal: string) => {
    logger.info(`Received ${signal}, initiating graceful shutdown`);

    try {
      await shutdownQueues();
      logger.info('Graceful shutdown completed');
      process.exit(0);
    } catch (error) {
      const err = error instanceof Error ? error : new Error(String(error));
      logger.error('Error during shutdown', err);
      process.exit(1);
    }
  };

  process.on('SIGTERM', () => shutdown('SIGTERM'));
  process.on('SIGINT', () => shutdown('SIGINT'));

  logger.debug('Shutdown handlers registered');
}

// Default export
export default {
  initializeQueues,
  startWorkers,
  stopWorkers,
  shutdownQueues,
  getWorkersStatus,
  getQueueHealth,
  subscribeToQueueEvents,
  registerShutdownHandlers,
};
