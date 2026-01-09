/**
 * EE Design Partner - Main Entry Point
 *
 * End-to-end hardware/software development automation platform
 */

import 'dotenv/config';
import express, { Express, Request, Response, NextFunction } from 'express';
import cors from 'cors';
import helmet from 'helmet';
import compression from 'compression';
import { Server as SocketIOServer } from 'socket.io';
import { createServer } from 'http';
import path from 'path';
import { fileURLToPath } from 'url';

import { config } from './config.js';
import { log } from './utils/logger.js';
import { EEDesignError, handleError } from './utils/errors.js';
import { createApiRoutes } from './api/routes.js';
import { SkillsEngineClient } from './services/skills/skills-engine-client.js';
import { getFileWatcherService, clearFileWatcherService } from './services/file-watcher/index.js';
import { setSkillsEngineClient, getSkillsEngineClient, clearSkillsEngineClient } from './state.js';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

async function startServer(): Promise<void> {
  const app: Express = express();
  const httpServer = createServer(app);

  // Socket.IO for real-time updates
  const io = new SocketIOServer(httpServer, {
    cors: {
      origin: '*',
      methods: ['GET', 'POST'],
    },
    path: '/ws',
  });

  // Middleware
  app.use(helmet({
    contentSecurityPolicy: false, // Disable for embedded UI
  }));
  app.use(cors());
  app.use(compression());
  app.use(express.json({ limit: '50mb' }));
  app.use(express.urlencoded({ extended: true, limit: '50mb' }));

  // Request logging middleware
  app.use((req: Request, res: Response, next: NextFunction) => {
    const startTime = Date.now();
    const requestId = req.headers['x-request-id'] as string || crypto.randomUUID();

    res.setHeader('X-Request-ID', requestId);

    res.on('finish', () => {
      const duration = Date.now() - startTime;
      log.info(`${req.method} ${req.path}`, {
        requestId,
        method: req.method,
        path: req.path,
        statusCode: res.statusCode,
        duration,
        userAgent: req.headers['user-agent'],
      });
    });

    next();
  });

  // Health check endpoints
  app.get('/health', (_req: Request, res: Response) => {
    res.json({
      status: 'healthy',
      service: config.pluginId,
      version: config.version,
      buildId: config.buildId,
      timestamp: new Date().toISOString(),
    });
  });

  app.get('/ready', (_req: Request, res: Response) => {
    // TODO: Add database connectivity check
    res.json({ ready: true });
  });

  app.get('/live', (_req: Request, res: Response) => {
    res.json({ alive: true });
  });

  // Build metadata endpoint
  app.get('/build-info', (_req: Request, res: Response) => {
    res.json({
      buildId: config.buildId,
      buildTimestamp: config.buildTimestamp,
      gitCommit: config.gitCommit,
      gitBranch: config.gitBranch,
      version: config.version,
      nodeEnv: config.nodeEnv,
    });
  });

  // API routes
  app.use('/api/v1', createApiRoutes(io));

  // Static UI files
  app.use('/ui', express.static(path.join(__dirname, '../../ui')));

  // Metrics endpoint (Prometheus format)
  app.get('/metrics', (_req: Request, res: Response) => {
    // TODO: Implement Prometheus metrics
    res.type('text/plain').send('# No metrics yet\n');
  });

  // Socket.IO connection handling
  io.on('connection', (socket) => {
    log.info('Client connected', { socketId: socket.id });

    socket.on('subscribe:project', (projectId: string) => {
      socket.join(`project:${projectId}`);
      log.debug('Client subscribed to project', { socketId: socket.id, projectId });
    });

    socket.on('subscribe:simulation', (simulationId: string) => {
      socket.join(`simulation:${simulationId}`);
      log.debug('Client subscribed to simulation', { socketId: socket.id, simulationId });
    });

    socket.on('subscribe:layout', (layoutId: string) => {
      socket.join(`layout:${layoutId}`);
      log.debug('Client subscribed to layout', { socketId: socket.id, layoutId });
    });

    socket.on('unsubscribe:project', (projectId: string) => {
      socket.leave(`project:${projectId}`);
      log.debug('Client unsubscribed from project', { socketId: socket.id, projectId });
    });

    socket.on('heartbeat', () => {
      socket.emit('heartbeat:ack');
    });

    socket.on('disconnect', () => {
      log.info('Client disconnected', { socketId: socket.id });
    });
  });

  // Global error handler
  app.use((err: Error, req: Request, res: Response, _next: NextFunction) => {
    const error = handleError(err);
    const requestId = res.getHeader('X-Request-ID') as string;

    log.error('Request failed', err, {
      requestId,
      method: req.method,
      path: req.path,
      code: error.code,
    });

    res.status(error.statusCode).json({
      success: false,
      error: {
        code: error.code,
        message: error.message,
        ...(config.nodeEnv === 'development' && {
          context: error.context,
          stack: error.stack,
        }),
      },
      metadata: {
        requestId,
        timestamp: new Date().toISOString(),
      },
    });
  });

  // 404 handler
  app.use((_req: Request, res: Response) => {
    res.status(404).json({
      success: false,
      error: {
        code: 'NOT_FOUND',
        message: 'Endpoint not found',
      },
    });
  });

  // Initialize Skills Engine client
  try {
    const skillsClient = new SkillsEngineClient({
      apiUrl: config.services.graphragUrl,
      apiKey: process.env.NEXUS_API_KEY || '',
      skillsDirectory: path.join(__dirname, '../../../skills'),
      autoRegister: true,
      syncInterval: 300000, // 5 minutes
    });

    await skillsClient.initialize();
    setSkillsEngineClient(skillsClient);

    log.info('Skills Engine client initialized', {
      registeredSkills: skillsClient.getRegisteredSkills().length,
    });
  } catch (error) {
    log.warn('Failed to initialize Skills Engine client', { error });
    // Non-fatal - continue starting server
  }

  // Initialize FileWatcherService for real-time KiCad file monitoring
  try {
    const fileWatcher = getFileWatcherService();
    await fileWatcher.initialize(io);

    log.info('FileWatcherService initialized', {
      stats: fileWatcher.getStats(),
    });
  } catch (error) {
    log.warn('Failed to initialize FileWatcherService', { error });
    // Non-fatal - continue starting server
  }

  // Start server
  httpServer.listen(config.port, () => {
    log.info(`EE Design Partner started`, {
      port: config.port,
      nodeEnv: config.nodeEnv,
      version: config.version,
      buildId: config.buildId,
    });

    log.info('Service URLs configured', {
      graphrag: config.services.graphragUrl,
      mageagent: config.services.mageagentUrl,
      sandbox: config.services.sandboxUrl,
    });
  });

  // Graceful shutdown
  const shutdown = async (signal: string): Promise<void> => {
    log.info(`Received ${signal}, shutting down gracefully...`);

    // Shutdown FileWatcherService
    clearFileWatcherService();
    log.info('FileWatcherService shutdown');

    // Shutdown Skills Engine client
    const skillsClient = getSkillsEngineClient();
    if (skillsClient) {
      await skillsClient.shutdown();
      clearSkillsEngineClient();
      log.info('Skills Engine client shutdown');
    }

    io.close();

    httpServer.close(() => {
      log.info('HTTP server closed');
      process.exit(0);
    });

    // Force exit after 30 seconds
    setTimeout(() => {
      log.error('Forced shutdown after timeout');
      process.exit(1);
    }, 30000);
  };

  process.on('SIGTERM', () => shutdown('SIGTERM'));
  process.on('SIGINT', () => shutdown('SIGINT'));
}

startServer().catch((error) => {
  log.error('Failed to start server', error);
  process.exit(1);
});