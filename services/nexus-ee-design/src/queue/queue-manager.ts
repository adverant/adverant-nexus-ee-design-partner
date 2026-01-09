/**
 * EE Design Partner - Queue Manager
 *
 * BullMQ queue definitions and management for long-running tasks.
 * Provides queues for simulation, layout, firmware, and validation jobs.
 */

import { Queue, QueueEvents, ConnectionOptions, JobsOptions } from 'bullmq';
import { EventEmitter } from 'events';
import { config } from '../config.js';
import { log, Logger } from '../utils/logger.js';

// ============================================================================
// Types
// ============================================================================

export type QueueName = 'simulation' | 'layout' | 'firmware' | 'validation';

export interface SimulationJobData {
  simulationId: string;
  projectId: string;
  type: string;
  name: string;
  schematicId?: string;
  pcbLayoutId?: string;
  parameters: Record<string, unknown>;
  testBench?: string;
  priority?: number;
  timeoutMs?: number;
}

export interface LayoutJobData {
  layoutId: string;
  projectId: string;
  schematicId: string;
  operation: 'generate' | 'optimize' | 'drc';
  strategy?: string;
  maxIterations?: number;
  targetScore?: number;
  config?: Record<string, unknown>;
}

export interface FirmwareJobData {
  firmwareId: string;
  projectId: string;
  pcbLayoutId?: string;
  name: string;
  targetMcu: {
    family: string;
    part: string;
    core: string;
    flashSize: number;
    ramSize: number;
    clockSpeed: number;
    peripherals: string[];
  };
  rtosConfig?: {
    type: string;
    version: string;
    tickRate: number;
    heapSize: number;
    maxTasks: number;
  };
  peripherals: Array<{
    type: string;
    instance: string;
    config: Record<string, unknown>;
    connectedTo?: string;
  }>;
  buildConfig?: {
    toolchain: string;
    buildSystem: string;
    optimizationLevel: string;
    debugSymbols: boolean;
    defines: Record<string, string>;
  };
}

export interface ValidationJobData {
  validationId: string;
  projectId: string;
  artifactType: 'schematic' | 'pcb' | 'firmware' | 'simulation';
  artifactId: string;
  validators: string[];
  config?: Record<string, unknown>;
}

export type JobData = SimulationJobData | LayoutJobData | FirmwareJobData | ValidationJobData;

export interface QueueProgress {
  jobId: string;
  queueName: QueueName;
  progress: number;
  message?: string;
  data?: Record<string, unknown>;
}

export interface QueueResult {
  jobId: string;
  queueName: QueueName;
  success: boolean;
  data?: Record<string, unknown>;
  error?: string;
}

// ============================================================================
// Redis Connection Configuration
// ============================================================================

const getRedisConnection = (): ConnectionOptions => ({
  host: config.redis.host,
  port: config.redis.port,
  password: config.redis.password,
  maxRetriesPerRequest: null,
  enableReadyCheck: false,
});

// ============================================================================
// Default Job Options
// ============================================================================

const defaultJobOptions: JobsOptions = {
  attempts: 3,
  backoff: {
    type: 'exponential',
    delay: 1000,
  },
  removeOnComplete: {
    age: 24 * 3600, // 24 hours
    count: 1000,
  },
  removeOnFail: {
    age: 7 * 24 * 3600, // 7 days
  },
};

const queueSpecificOptions: Record<QueueName, Partial<JobsOptions>> = {
  simulation: {
    attempts: 3,
    backoff: {
      type: 'exponential',
      delay: 2000,
    },
  },
  layout: {
    attempts: 2,
    backoff: {
      type: 'exponential',
      delay: 5000,
    },
  },
  firmware: {
    attempts: 3,
    backoff: {
      type: 'exponential',
      delay: 3000,
    },
  },
  validation: {
    attempts: 3,
    backoff: {
      type: 'exponential',
      delay: 1000,
    },
  },
};

// ============================================================================
// Queue Manager Class
// ============================================================================

class QueueManagerImpl extends EventEmitter {
  private logger: Logger;
  private queues: Map<QueueName, Queue>;
  private queueEvents: Map<QueueName, QueueEvents>;
  private initialized: boolean = false;

  constructor() {
    super();
    this.logger = log.child({ service: 'queue-manager' });
    this.queues = new Map();
    this.queueEvents = new Map();
  }

  /**
   * Initialize all queues and queue events.
   */
  async initialize(): Promise<void> {
    if (this.initialized) {
      this.logger.warn('Queue manager already initialized');
      return;
    }

    this.logger.info('Initializing queue manager', {
      redisHost: config.redis.host,
      redisPort: config.redis.port,
    });

    const connection = getRedisConnection();
    const queueNames: QueueName[] = ['simulation', 'layout', 'firmware', 'validation'];

    for (const name of queueNames) {
      // Create queue
      const queue = new Queue(name, {
        connection,
        defaultJobOptions: {
          ...defaultJobOptions,
          ...queueSpecificOptions[name],
        },
      });

      this.queues.set(name, queue);

      // Create queue events listener
      const events = new QueueEvents(name, { connection });
      this.queueEvents.set(name, events);

      // Forward events to EventEmitter for Socket.IO integration
      this.setupEventForwarding(name, events);

      this.logger.debug('Queue initialized', { queueName: name });
    }

    this.initialized = true;
    this.logger.info('Queue manager initialized successfully', {
      queueCount: queueNames.length,
    });
  }

  /**
   * Set up event forwarding from QueueEvents to EventEmitter.
   */
  private setupEventForwarding(queueName: QueueName, events: QueueEvents): void {
    // Job completed
    events.on('completed', ({ jobId, returnvalue }) => {
      this.logger.debug('Job completed', { queueName, jobId });
      this.emit('job:completed', {
        queueName,
        jobId,
        result: returnvalue,
      });
    });

    // Job failed
    events.on('failed', ({ jobId, failedReason }) => {
      this.logger.warn('Job failed', { queueName, jobId, reason: failedReason });
      this.emit('job:failed', {
        queueName,
        jobId,
        error: failedReason,
      });
    });

    // Job progress
    events.on('progress', ({ jobId, data }) => {
      this.logger.debug('Job progress', { queueName, jobId, data });
      this.emit('job:progress', {
        queueName,
        jobId,
        ...(data as object),
      });
    });

    // Job active (started)
    events.on('active', ({ jobId }) => {
      this.logger.debug('Job active', { queueName, jobId });
      this.emit('job:active', {
        queueName,
        jobId,
      });
    });

    // Job waiting
    events.on('waiting', ({ jobId }) => {
      this.logger.debug('Job waiting', { queueName, jobId });
      this.emit('job:waiting', {
        queueName,
        jobId,
      });
    });

    // Job stalled
    events.on('stalled', ({ jobId }) => {
      this.logger.warn('Job stalled', { queueName, jobId });
      this.emit('job:stalled', {
        queueName,
        jobId,
      });
    });

    // Job retrying
    events.on('retries-exhausted', ({ jobId }) => {
      this.logger.error('Job retries exhausted', undefined, { queueName, jobId });
      this.emit('job:retries-exhausted', {
        queueName,
        jobId,
      });
    });
  }

  /**
   * Get a queue by name.
   */
  getQueue(name: QueueName): Queue {
    const queue = this.queues.get(name);
    if (!queue) {
      throw new Error(`Queue '${name}' not found. Did you call initialize()?`);
    }
    return queue;
  }

  /**
   * Get queue events by name.
   */
  getQueueEvents(name: QueueName): QueueEvents {
    const events = this.queueEvents.get(name);
    if (!events) {
      throw new Error(`QueueEvents for '${name}' not found. Did you call initialize()?`);
    }
    return events;
  }

  /**
   * Add a job to a queue.
   */
  async addJob<T extends JobData>(
    queueName: QueueName,
    data: T,
    options?: Partial<JobsOptions>
  ): Promise<string> {
    const queue = this.getQueue(queueName);

    // Determine priority (higher number = higher priority in BullMQ)
    let priority = 5; // Default medium priority
    if ('priority' in data && typeof data.priority === 'number') {
      priority = data.priority;
    }

    const job = await queue.add(queueName, data, {
      ...options,
      priority,
    });

    this.logger.info('Job added to queue', {
      queueName,
      jobId: job.id,
      priority,
    });

    return job.id || '';
  }

  /**
   * Get job status.
   */
  async getJobStatus(queueName: QueueName, jobId: string): Promise<{
    id: string;
    state: string;
    progress: number;
    attemptsMade: number;
    data: JobData | undefined;
    returnValue: unknown;
    failedReason: string | undefined;
  } | null> {
    const queue = this.getQueue(queueName);
    const job = await queue.getJob(jobId);

    if (!job) {
      return null;
    }

    const state = await job.getState();

    return {
      id: job.id || '',
      state,
      progress: job.progress as number || 0,
      attemptsMade: job.attemptsMade,
      data: job.data as JobData,
      returnValue: job.returnvalue,
      failedReason: job.failedReason,
    };
  }

  /**
   * Cancel a job.
   */
  async cancelJob(queueName: QueueName, jobId: string): Promise<boolean> {
    const queue = this.getQueue(queueName);
    const job = await queue.getJob(jobId);

    if (!job) {
      return false;
    }

    const state = await job.getState();

    // Can only cancel waiting or delayed jobs
    if (state === 'waiting' || state === 'delayed') {
      await job.remove();
      this.logger.info('Job cancelled', { queueName, jobId });
      return true;
    }

    // For active jobs, we can't cancel directly but we can set a flag
    // Workers should check for cancellation
    this.logger.warn('Cannot cancel job in state', { queueName, jobId, state });
    return false;
  }

  /**
   * Get queue statistics.
   */
  async getQueueStats(queueName: QueueName): Promise<{
    waiting: number;
    active: number;
    completed: number;
    failed: number;
    delayed: number;
    paused: number;
  }> {
    const queue = this.getQueue(queueName);

    const [waiting, active, completed, failed, delayed] = await Promise.all([
      queue.getWaitingCount(),
      queue.getActiveCount(),
      queue.getCompletedCount(),
      queue.getFailedCount(),
      queue.getDelayedCount(),
    ]);

    // Get paused count via getJobCountByTypes
    const pausedCount = await queue.getJobCountByTypes('paused');
    const paused = pausedCount;

    return { waiting, active, completed, failed, delayed, paused };
  }

  /**
   * Get all queue statistics.
   */
  async getAllQueueStats(): Promise<Record<QueueName, {
    waiting: number;
    active: number;
    completed: number;
    failed: number;
    delayed: number;
    paused: number;
  }>> {
    const stats: Record<string, {
      waiting: number;
      active: number;
      completed: number;
      failed: number;
      delayed: number;
      paused: number;
    }> = {};

    const queueEntries = Array.from(this.queues.entries());
    for (const [name] of queueEntries) {
      stats[name] = await this.getQueueStats(name);
    }

    return stats as Record<QueueName, {
      waiting: number;
      active: number;
      completed: number;
      failed: number;
      delayed: number;
      paused: number;
    }>;
  }

  /**
   * Pause a queue.
   */
  async pauseQueue(queueName: QueueName): Promise<void> {
    const queue = this.getQueue(queueName);
    await queue.pause();
    this.logger.info('Queue paused', { queueName });
  }

  /**
   * Resume a queue.
   */
  async resumeQueue(queueName: QueueName): Promise<void> {
    const queue = this.getQueue(queueName);
    await queue.resume();
    this.logger.info('Queue resumed', { queueName });
  }

  /**
   * Clean old jobs from a queue.
   */
  async cleanQueue(
    queueName: QueueName,
    grace: number = 24 * 3600 * 1000, // 24 hours
    limit: number = 1000,
    status: 'completed' | 'failed' = 'completed'
  ): Promise<string[]> {
    const queue = this.getQueue(queueName);
    const jobs = await queue.clean(grace, limit, status);
    this.logger.info('Queue cleaned', { queueName, removedCount: jobs.length, status });
    return jobs;
  }

  /**
   * Drain a queue (remove all jobs).
   */
  async drainQueue(queueName: QueueName): Promise<void> {
    const queue = this.getQueue(queueName);
    await queue.drain();
    this.logger.info('Queue drained', { queueName });
  }

  /**
   * Close all queues and connections.
   */
  async close(): Promise<void> {
    this.logger.info('Closing queue manager');

    const eventEntries = Array.from(this.queueEvents.entries());
    for (const [name, events] of eventEntries) {
      await events.close();
      this.logger.debug('QueueEvents closed', { queueName: name });
    }

    const queueEntries = Array.from(this.queues.entries());
    for (const [name, queue] of queueEntries) {
      await queue.close();
      this.logger.debug('Queue closed', { queueName: name });
    }

    this.queues.clear();
    this.queueEvents.clear();
    this.initialized = false;

    this.logger.info('Queue manager closed');
  }

  /**
   * Check if initialized.
   */
  isInitialized(): boolean {
    return this.initialized;
  }
}

// ============================================================================
// Singleton Export
// ============================================================================

export const queueManager = new QueueManagerImpl();

// ============================================================================
// Individual Queue Exports
// ============================================================================

export const getSimulationQueue = (): Queue => queueManager.getQueue('simulation');
export const getLayoutQueue = (): Queue => queueManager.getQueue('layout');
export const getFirmwareQueue = (): Queue => queueManager.getQueue('firmware');
export const getValidationQueue = (): Queue => queueManager.getQueue('validation');

// ============================================================================
// Helper Functions for Adding Jobs
// ============================================================================

export async function addSimulationJob(
  data: SimulationJobData,
  options?: Partial<JobsOptions>
): Promise<string> {
  return queueManager.addJob('simulation', data, options);
}

export async function addLayoutJob(
  data: LayoutJobData,
  options?: Partial<JobsOptions>
): Promise<string> {
  return queueManager.addJob('layout', data, options);
}

export async function addFirmwareJob(
  data: FirmwareJobData,
  options?: Partial<JobsOptions>
): Promise<string> {
  return queueManager.addJob('firmware', data, options);
}

export async function addValidationJob(
  data: ValidationJobData,
  options?: Partial<JobsOptions>
): Promise<string> {
  return queueManager.addJob('validation', data, options);
}

export default queueManager;
