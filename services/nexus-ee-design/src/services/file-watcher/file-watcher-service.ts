/**
 * FileWatcherService - Real-time KiCad File Monitoring
 *
 * Watches KiCad schematic and PCB files for changes, emitting Socket.IO
 * events to connected clients for real-time UI updates.
 *
 * Features:
 * - Watches /data/projects/**\/*.kicad_{sch,pcb} patterns
 * - Debounces rapid file changes (500ms)
 * - Computes MD5 checksums for change detection
 * - Tracks file versions for cache busting
 * - Extracts projectId from file paths
 * - Emits 'file:changed' and 'pcb:updated' events
 */

import { watch, FSWatcher } from 'chokidar';
import { createHash } from 'crypto';
import { readFile, stat } from 'fs/promises';
import path from 'path';
import { Server as SocketIOServer } from 'socket.io';
import { log, Logger } from '../../utils/logger.js';
import { config } from '../../config.js';

// ============================================================================
// Types
// ============================================================================

export interface FileWatcherConfig {
  /** Base directory to watch (default: /data/projects) */
  baseDir: string;

  /** Debounce delay in milliseconds (default: 500) */
  debounceMs: number;

  /** File patterns to watch */
  patterns: string[];

  /** Enable watching (can be disabled in test environments) */
  enabled: boolean;
}

export interface FileChangeEvent {
  /** Type of change */
  type: 'add' | 'change' | 'unlink';

  /** Full file path */
  filePath: string;

  /** Relative path from base directory */
  relativePath: string;

  /** Project ID extracted from path */
  projectId: string;

  /** File type (schematic or pcb) */
  fileType: 'schematic' | 'pcb';

  /** File extension */
  extension: string;

  /** MD5 checksum of file content */
  checksum: string;

  /** File version (increments on each change) */
  version: number;

  /** File size in bytes */
  size: number;

  /** Last modified timestamp */
  modifiedAt: string;

  /** Event timestamp */
  timestamp: string;
}

export interface PCBUpdateEvent extends FileChangeEvent {
  /** PCB-specific metadata */
  pcbMetadata?: {
    /** Number of layers detected */
    layerCount?: number;

    /** Board dimensions if extractable */
    dimensions?: {
      width: number;
      height: number;
      unit: 'mm' | 'inch';
    };
  };
}

interface FileVersionInfo {
  version: number;
  checksum: string;
  lastModified: number;
}

// ============================================================================
// FileWatcherService
// ============================================================================

export class FileWatcherService {
  private watcher: FSWatcher | null = null;
  private io: SocketIOServer | null = null;
  private config: FileWatcherConfig;
  private logger: Logger;
  private fileVersions: Map<string, FileVersionInfo> = new Map();
  private debounceTimers: Map<string, NodeJS.Timeout> = new Map();
  private isInitialized = false;

  constructor(configOverrides: Partial<FileWatcherConfig> = {}) {
    this.config = {
      baseDir: config.storage.projectsDir || '/data/projects',
      debounceMs: 500,
      patterns: [
        '**/*.kicad_sch',
        '**/*.kicad_pcb',
      ],
      enabled: config.nodeEnv !== 'test',
      ...configOverrides,
    };

    this.logger = log.child({ service: 'FileWatcherService' });
  }

  /**
   * Initialize the file watcher with Socket.IO server
   */
  async initialize(io: SocketIOServer): Promise<void> {
    if (this.isInitialized) {
      this.logger.warn('FileWatcherService already initialized');
      return;
    }

    if (!this.config.enabled) {
      this.logger.info('FileWatcherService disabled in current environment');
      return;
    }

    this.io = io;

    // Build glob patterns for chokidar
    const watchPatterns = this.config.patterns.map(
      (pattern) => path.join(this.config.baseDir, pattern)
    );

    this.logger.info('Initializing file watcher', {
      baseDir: this.config.baseDir,
      patterns: this.config.patterns,
      debounceMs: this.config.debounceMs,
    });

    try {
      this.watcher = watch(watchPatterns, {
        persistent: true,
        ignoreInitial: true,
        awaitWriteFinish: {
          stabilityThreshold: 200,
          pollInterval: 100,
        },
        usePolling: false, // Use native events when possible
        atomic: true, // Handle atomic writes from editors
      });

      this.watcher
        .on('add', (filePath) => this.handleFileEvent('add', filePath))
        .on('change', (filePath) => this.handleFileEvent('change', filePath))
        .on('unlink', (filePath) => this.handleFileEvent('unlink', filePath))
        .on('error', (error) => this.handleWatcherError(error))
        .on('ready', () => {
          this.isInitialized = true;
          this.logger.info('File watcher ready', {
            watchedPaths: watchPatterns,
          });
        });

    } catch (error) {
      this.logger.error('Failed to initialize file watcher', error as Error);
      throw error;
    }
  }

  /**
   * Handle file system events with debouncing
   */
  private handleFileEvent(type: 'add' | 'change' | 'unlink', filePath: string): void {
    // Clear existing debounce timer for this file
    const existingTimer = this.debounceTimers.get(filePath);
    if (existingTimer) {
      clearTimeout(existingTimer);
    }

    // Set new debounce timer
    const timer = setTimeout(async () => {
      this.debounceTimers.delete(filePath);
      await this.processFileChange(type, filePath);
    }, this.config.debounceMs);

    this.debounceTimers.set(filePath, timer);
  }

  /**
   * Process file change after debounce period
   */
  private async processFileChange(
    type: 'add' | 'change' | 'unlink',
    filePath: string
  ): Promise<void> {
    const relativePath = path.relative(this.config.baseDir, filePath);
    const projectId = this.extractProjectId(relativePath);
    const fileType = this.getFileType(filePath);
    const extension = path.extname(filePath);

    this.logger.debug('Processing file change', {
      type,
      filePath,
      relativePath,
      projectId,
      fileType,
    });

    try {
      let checksum = '';
      let size = 0;
      let modifiedAt = new Date().toISOString();
      let version = 1;

      if (type !== 'unlink') {
        // Compute checksum and get file stats
        const [content, stats] = await Promise.all([
          readFile(filePath),
          stat(filePath),
        ]);

        checksum = this.computeChecksum(content);
        size = stats.size;
        modifiedAt = stats.mtime.toISOString();

        // Update version tracking
        const versionInfo = this.fileVersions.get(filePath);
        if (versionInfo) {
          // Only increment version if checksum actually changed
          if (versionInfo.checksum !== checksum) {
            version = versionInfo.version + 1;
            this.fileVersions.set(filePath, {
              version,
              checksum,
              lastModified: stats.mtimeMs,
            });
          } else {
            version = versionInfo.version;
            this.logger.debug('File touched but content unchanged', { filePath });
            return; // Skip emitting event if content hasn't changed
          }
        } else {
          // First time seeing this file
          this.fileVersions.set(filePath, {
            version: 1,
            checksum,
            lastModified: stats.mtimeMs,
          });
        }
      } else {
        // File deleted - remove from version tracking
        this.fileVersions.delete(filePath);
      }

      const event: FileChangeEvent = {
        type,
        filePath,
        relativePath,
        projectId,
        fileType,
        extension,
        checksum,
        version,
        size,
        modifiedAt,
        timestamp: new Date().toISOString(),
      };

      // Emit events
      this.emitFileChangedEvent(event);

      // Emit PCB-specific event with additional metadata
      if (fileType === 'pcb') {
        await this.emitPCBUpdatedEvent(event);
      }

    } catch (error) {
      this.logger.error('Failed to process file change', error as Error, {
        type,
        filePath,
      });
    }
  }

  /**
   * Emit file:changed event to project room
   */
  private emitFileChangedEvent(event: FileChangeEvent): void {
    if (!this.io) {
      this.logger.warn('Socket.IO not initialized, skipping event emission');
      return;
    }

    const roomName = `project:${event.projectId}`;

    this.logger.debug('Emitting file:changed event', {
      room: roomName,
      fileType: event.fileType,
      type: event.type,
      version: event.version,
    });

    this.io.to(roomName).emit('file:changed', event);

    // Also emit to a global file changes channel
    this.io.emit('file:changed:global', event);
  }

  /**
   * Emit pcb:updated event with PCB-specific metadata
   */
  private async emitPCBUpdatedEvent(baseEvent: FileChangeEvent): Promise<void> {
    if (!this.io) {
      return;
    }

    const pcbEvent: PCBUpdateEvent = {
      ...baseEvent,
      pcbMetadata: undefined,
    };

    // Try to extract PCB metadata (non-blocking)
    if (baseEvent.type !== 'unlink') {
      try {
        pcbEvent.pcbMetadata = await this.extractPCBMetadata(baseEvent.filePath);
      } catch (error) {
        this.logger.debug('Could not extract PCB metadata', { error });
      }
    }

    const roomName = `project:${baseEvent.projectId}`;

    this.logger.info('Emitting pcb:updated event', {
      room: roomName,
      type: baseEvent.type,
      version: baseEvent.version,
      hasMetadata: !!pcbEvent.pcbMetadata,
    });

    this.io.to(roomName).emit('pcb:updated', pcbEvent);
  }

  /**
   * Extract project ID from relative file path
   * Assumes structure: {projectId}/... or {projectId}.kicad_*
   */
  private extractProjectId(relativePath: string): string {
    const parts = relativePath.split(path.sep);

    // If file is directly in a project folder: projectId/file.kicad_*
    if (parts.length >= 1) {
      return parts[0];
    }

    // Fallback: use filename without extension as projectId
    return path.basename(relativePath, path.extname(relativePath));
  }

  /**
   * Determine file type from path
   */
  private getFileType(filePath: string): 'schematic' | 'pcb' {
    const ext = path.extname(filePath).toLowerCase();
    return ext === '.kicad_pcb' ? 'pcb' : 'schematic';
  }

  /**
   * Compute MD5 checksum of file content
   */
  private computeChecksum(content: Buffer): string {
    return createHash('md5').update(content).digest('hex');
  }

  /**
   * Extract basic metadata from PCB file
   * This is a lightweight extraction - detailed parsing done elsewhere
   */
  private async extractPCBMetadata(
    filePath: string
  ): Promise<PCBUpdateEvent['pcbMetadata']> {
    try {
      const content = await readFile(filePath, 'utf-8');

      // Extract layer count from (layers ...) section
      const layersMatch = content.match(/\(layers\s+(\d+)/);
      const layerCount = layersMatch ? parseInt(layersMatch[1], 10) : undefined;

      // Extract board dimensions from (gr_rect) or (rect) on Edge.Cuts layer
      // This is a simplified extraction
      let dimensions: PCBUpdateEvent['pcbMetadata'] extends { dimensions?: infer D } ? D : undefined;

      const edgeCutsMatch = content.match(
        /\(gr_rect[^)]*\(start\s+([\d.]+)\s+([\d.]+)\)[^)]*\(end\s+([\d.]+)\s+([\d.]+)\)/
      );

      if (edgeCutsMatch) {
        const [, x1, y1, x2, y2] = edgeCutsMatch.map(parseFloat);
        dimensions = {
          width: Math.abs(x2 - x1),
          height: Math.abs(y2 - y1),
          unit: 'mm' as const,
        };
      }

      return {
        layerCount,
        dimensions,
      };
    } catch {
      return undefined;
    }
  }

  /**
   * Handle watcher errors
   */
  private handleWatcherError(error: Error): void {
    this.logger.error('File watcher error', error);
  }

  /**
   * Get version info for a file
   */
  getFileVersion(filePath: string): FileVersionInfo | undefined {
    return this.fileVersions.get(filePath);
  }

  /**
   * Get all tracked file versions
   */
  getAllFileVersions(): Map<string, FileVersionInfo> {
    return new Map(this.fileVersions);
  }

  /**
   * Manually trigger a file change event (useful for testing)
   */
  async triggerFileChange(filePath: string): Promise<void> {
    await this.processFileChange('change', filePath);
  }

  /**
   * Check if watcher is running
   */
  isRunning(): boolean {
    return this.isInitialized && this.watcher !== null;
  }

  /**
   * Get watcher statistics
   */
  getStats(): {
    isRunning: boolean;
    trackedFiles: number;
    pendingDebounces: number;
    baseDir: string;
    patterns: string[];
  } {
    return {
      isRunning: this.isRunning(),
      trackedFiles: this.fileVersions.size,
      pendingDebounces: this.debounceTimers.size,
      baseDir: this.config.baseDir,
      patterns: this.config.patterns,
    };
  }

  /**
   * Stop the file watcher and cleanup
   */
  async shutdown(): Promise<void> {
    this.logger.info('Shutting down FileWatcherService');

    // Clear all debounce timers
    for (const timer of this.debounceTimers.values()) {
      clearTimeout(timer);
    }
    this.debounceTimers.clear();

    // Close watcher
    if (this.watcher) {
      await this.watcher.close();
      this.watcher = null;
    }

    this.isInitialized = false;
    this.io = null;

    this.logger.info('FileWatcherService shutdown complete');
  }
}

// ============================================================================
// Singleton Instance
// ============================================================================

let fileWatcherInstance: FileWatcherService | null = null;

/**
 * Get or create the FileWatcherService singleton
 */
export function getFileWatcherService(
  configOverrides?: Partial<FileWatcherConfig>
): FileWatcherService {
  if (!fileWatcherInstance) {
    fileWatcherInstance = new FileWatcherService(configOverrides);
  }
  return fileWatcherInstance;
}

/**
 * Clear the singleton instance (for testing)
 */
export function clearFileWatcherService(): void {
  if (fileWatcherInstance) {
    fileWatcherInstance.shutdown();
    fileWatcherInstance = null;
  }
}

export default FileWatcherService;
