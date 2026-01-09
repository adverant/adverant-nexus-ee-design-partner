/**
 * FileWatcher Service Exports
 *
 * Real-time KiCad file monitoring with Socket.IO event emission.
 */

export {
  FileWatcherService,
  getFileWatcherService,
  clearFileWatcherService,
} from './file-watcher-service.js';

export type {
  FileWatcherConfig,
  FileChangeEvent,
  PCBUpdateEvent,
} from './file-watcher-service.js';
