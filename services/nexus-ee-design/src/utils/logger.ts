/**
 * EE Design Partner - Logger
 *
 * Winston-based structured logging with build metadata
 */

import winston from 'winston';
import { config } from '../config.js';

const { combine, timestamp, printf, colorize, errors } = winston.format;

interface LogMetadata {
  requestId?: string;
  userId?: string;
  projectId?: string;
  service?: string;
  operation?: string;
  duration?: number;
  [key: string]: unknown;
}

const logFormat = printf(({ level, message, timestamp, stack, ...metadata }) => {
  const meta = Object.keys(metadata).length > 0 ? ` ${JSON.stringify(metadata)}` : '';
  const stackTrace = stack ? `\n${stack}` : '';
  return `${timestamp} [${level}] ${message}${meta}${stackTrace}`;
});

const logger = winston.createLogger({
  level: config.logLevel,
  format: combine(
    errors({ stack: true }),
    timestamp({ format: 'YYYY-MM-DD HH:mm:ss.SSS' }),
    logFormat
  ),
  defaultMeta: {
    service: config.pluginId,
    buildId: config.buildId,
    version: config.version,
  },
  transports: [
    new winston.transports.Console({
      format: combine(
        colorize({ all: config.nodeEnv === 'development' }),
        logFormat
      ),
    }),
  ],
});

// Add file transport in production
if (config.nodeEnv === 'production') {
  logger.add(
    new winston.transports.File({
      filename: '/var/log/nexus-ee-design/error.log',
      level: 'error',
      maxsize: 10 * 1024 * 1024, // 10MB
      maxFiles: 5,
    })
  );
  logger.add(
    new winston.transports.File({
      filename: '/var/log/nexus-ee-design/combined.log',
      maxsize: 10 * 1024 * 1024, // 10MB
      maxFiles: 10,
    })
  );
}

export interface Logger {
  debug(message: string, metadata?: LogMetadata): void;
  info(message: string, metadata?: LogMetadata): void;
  warn(message: string, metadata?: LogMetadata): void;
  error(message: string, error?: Error, metadata?: LogMetadata): void;
  child(defaultMetadata: LogMetadata): Logger;
}

function createLogger(defaultMetadata: LogMetadata = {}): Logger {
  return {
    debug(message: string, metadata?: LogMetadata): void {
      logger.debug(message, { ...defaultMetadata, ...metadata });
    },

    info(message: string, metadata?: LogMetadata): void {
      logger.info(message, { ...defaultMetadata, ...metadata });
    },

    warn(message: string, metadata?: LogMetadata): void {
      logger.warn(message, { ...defaultMetadata, ...metadata });
    },

    error(message: string, error?: Error, metadata?: LogMetadata): void {
      logger.error(message, {
        ...defaultMetadata,
        ...metadata,
        error: error?.message,
        stack: error?.stack,
      });
    },

    child(childMetadata: LogMetadata): Logger {
      return createLogger({ ...defaultMetadata, ...childMetadata });
    },
  };
}

export const log = createLogger();

export default log;