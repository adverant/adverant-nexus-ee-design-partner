/**
 * NFS Storage Utility - Artifact Storage Pattern for EE Design
 *
 * Stores all generated artifacts to the terminal computer NFS share
 * for persistence and cross-device access.
 *
 * Pattern follows the patent plugin implementation.
 *
 * Directory Structure:
 *   /workspace/Exports/ee-design/
 *   ├── {organization_id}/
 *   │   └── {project_id}/
 *   │       ├── schematics/
 *   │       │   ├── {schematic_id}.kicad_sch
 *   │       │   ├── {schematic_id}.pdf
 *   │       │   └── {schematic_id}.svg
 *   │       ├── pcb-layouts/
 *   │       │   ├── {layout_id}.kicad_pcb
 *   │       │   ├── {layout_id}_layers.zip
 *   │       │   └── gerbers/
 *   │       ├── simulations/
 *   │       │   ├── {sim_id}_results.json
 *   │       │   └── {sim_id}_waveforms.csv
 *   │       └── firmware/
 *   │           └── {fw_id}/
 *   │               └── src/
 */

import * as fs from 'fs/promises';
import * as fsSync from 'fs';
import * as path from 'path';
import { log } from './logger.js';

/**
 * Default NFS base path (can be overridden via env)
 * Production: /workspace/Exports/ee-design
 * Development: Uses temp directory
 */
const getNFSBasePath = (): string => {
  const envPath = process.env.ARTIFACT_STORAGE_PATH;
  if (envPath) {
    return envPath;
  }

  // Check if production NFS mount exists
  const productionPath = '/workspace/Exports/ee-design';
  if (fsSync.existsSync('/workspace/Exports')) {
    return productionPath;
  }

  // Fallback to temp directory for development
  const tempPath = path.join(process.env.TMPDIR || '/tmp', 'ee-design-artifacts');
  return tempPath;
};

const NFS_BASE = getNFSBasePath();

/**
 * Artifact types supported
 */
export type ArtifactType = 'schematics' | 'pcb-layouts' | 'simulations' | 'firmware' | 'gerbers' | 'exports';

/**
 * Artifact path result
 */
export interface ArtifactPath {
  /** Local filesystem path (NFS) */
  localPath: string;

  /** API endpoint URL for retrieval */
  apiUrl: string;

  /** Whether the file was successfully created */
  success: boolean;

  /** File size in bytes */
  size?: number;
}

/**
 * Storage status information
 */
export interface StorageStatus {
  /** Base storage path */
  basePath: string;

  /** Whether the storage is writable */
  writable: boolean;

  /** Whether the storage path exists */
  exists: boolean;

  /** Available space in bytes (if determinable) */
  availableSpace?: number;

  /** Is this production NFS or development temp? */
  isProduction: boolean;
}

/**
 * NFS Storage class for EE design artifacts
 */
export class NFSStorage {
  /**
   * Get the storage status
   */
  static async getStorageStatus(): Promise<StorageStatus> {
    const basePath = NFS_BASE;
    const isProduction = basePath.startsWith('/workspace/Exports');

    let exists = false;
    let writable = false;

    try {
      // Check if base path exists
      await fs.access(basePath);
      exists = true;

      // Check if writable by creating a test file
      const testFile = path.join(basePath, '.write-test');
      try {
        await fs.writeFile(testFile, 'test');
        await fs.unlink(testFile);
        writable = true;
      } catch {
        // Not writable
      }
    } catch {
      // Path doesn't exist, try to create it
      try {
        await fs.mkdir(basePath, { recursive: true });
        exists = true;
        writable = true;
      } catch {
        // Can't create
      }
    }

    return {
      basePath,
      writable,
      exists,
      isProduction,
    };
  }

  /**
   * Get the artifact directory for a project
   */
  static async getArtifactDir(
    organizationId: string,
    projectId: string,
    artifactType: ArtifactType
  ): Promise<string> {
    const dir = path.join(NFS_BASE, organizationId, projectId, artifactType);
    await fs.mkdir(dir, { recursive: true });
    return dir;
  }

  /**
   * Store an artifact file
   */
  static async storeArtifact(
    organizationId: string,
    projectId: string,
    artifactType: ArtifactType,
    artifactId: string,
    content: Buffer | string,
    filename: string
  ): Promise<ArtifactPath> {
    try {
      const dir = await this.getArtifactDir(organizationId, projectId, artifactType);
      const localPath = path.join(dir, filename);

      await fs.writeFile(localPath, content);

      const stats = await fs.stat(localPath);

      log.info('Artifact stored to NFS', {
        localPath,
        artifactType,
        artifactId,
        size: stats.size,
        organization: organizationId,
        project: projectId,
      });

      return {
        localPath,
        apiUrl: `/api/v1/artifacts/${organizationId}/${projectId}/${artifactType}/${artifactId}/${filename}`,
        success: true,
        size: stats.size,
      };
    } catch (error) {
      log.error('Failed to store artifact to NFS', error as Error, {
        organizationId,
        projectId,
        artifactType,
        artifactId,
        filename,
      });

      return {
        localPath: '',
        apiUrl: '',
        success: false,
      };
    }
  }

  /**
   * Store multiple artifacts in a subdirectory
   */
  static async storeArtifacts(
    organizationId: string,
    projectId: string,
    artifactType: ArtifactType,
    artifactId: string,
    files: Array<{ filename: string; content: Buffer | string }>
  ): Promise<ArtifactPath[]> {
    const results: ArtifactPath[] = [];

    for (const file of files) {
      const result = await this.storeArtifact(
        organizationId,
        projectId,
        artifactType,
        artifactId,
        file.content,
        file.filename
      );
      results.push(result);
    }

    return results;
  }

  /**
   * Get an artifact file
   */
  static async getArtifact(localPath: string): Promise<Buffer> {
    return fs.readFile(localPath);
  }

  /**
   * Get artifact as stream (for large files)
   */
  static getArtifactStream(localPath: string): fsSync.ReadStream {
    return fsSync.createReadStream(localPath);
  }

  /**
   * Check if an artifact exists
   */
  static async artifactExists(localPath: string): Promise<boolean> {
    try {
      await fs.access(localPath);
      return true;
    } catch {
      return false;
    }
  }

  /**
   * Delete an artifact
   */
  static async deleteArtifact(localPath: string): Promise<boolean> {
    try {
      await fs.unlink(localPath);
      log.info('Artifact deleted from NFS', { localPath });
      return true;
    } catch (error) {
      log.error('Failed to delete artifact', error as Error, { localPath });
      return false;
    }
  }

  /**
   * List directory contents (for file browser)
   * Returns FileNode structure for the given path
   */
  static async listDirectory(
    fullPath: string
  ): Promise<{
    name: string;
    path: string;
    isDirectory: boolean;
    size: number;
    modifiedAt: string;
    extension?: string;
    children?: {
      name: string;
      path: string;
      isDirectory: boolean;
      size: number;
      modifiedAt: string;
      extension?: string;
    }[];
  }> {
    try {
      // Validate the path is within allowed directories
      const normalizedPath = path.normalize(fullPath);

      // Security: Ensure path doesn't escape NFS_BASE or /workspace
      if (!normalizedPath.startsWith(NFS_BASE) && !normalizedPath.startsWith('/workspace')) {
        throw new Error('Path traversal not allowed');
      }

      const stats = await fs.stat(normalizedPath);
      const name = path.basename(normalizedPath);

      const result = {
        name,
        path: normalizedPath,
        isDirectory: stats.isDirectory(),
        size: stats.size,
        modifiedAt: stats.mtime.toISOString(),
        extension: !stats.isDirectory() ? path.extname(name).slice(1) : undefined,
        children: undefined as {
          name: string;
          path: string;
          isDirectory: boolean;
          size: number;
          modifiedAt: string;
          extension?: string;
        }[] | undefined,
      };

      // If it's a directory, list its contents
      if (stats.isDirectory()) {
        const entries = await fs.readdir(normalizedPath, { withFileTypes: true });
        result.children = await Promise.all(
          entries.map(async (entry) => {
            const entryPath = path.join(normalizedPath, entry.name);
            try {
              const entryStats = await fs.stat(entryPath);
              return {
                name: entry.name,
                path: entryPath,
                isDirectory: entry.isDirectory(),
                size: entryStats.size,
                modifiedAt: entryStats.mtime.toISOString(),
                extension: !entry.isDirectory() ? path.extname(entry.name).slice(1) : undefined,
              };
            } catch {
              return {
                name: entry.name,
                path: entryPath,
                isDirectory: entry.isDirectory(),
                size: 0,
                modifiedAt: new Date().toISOString(),
                extension: !entry.isDirectory() ? path.extname(entry.name).slice(1) : undefined,
              };
            }
          })
        );
      }

      return result;
    } catch (error) {
      log.error('Failed to list directory', error as Error, { path: fullPath });
      throw error;
    }
  }

  /**
   * Create a directory
   */
  static async createDirectory(fullPath: string): Promise<void> {
    // Security: Ensure path doesn't escape allowed directories
    const normalizedPath = path.normalize(fullPath);
    if (!normalizedPath.startsWith(NFS_BASE) && !normalizedPath.startsWith('/workspace')) {
      throw new Error('Path traversal not allowed');
    }

    await fs.mkdir(normalizedPath, { recursive: true });
    log.info('Directory created on NFS', { path: normalizedPath });
  }

  /**
   * Write file content
   */
  static async writeFile(fullPath: string, content: Buffer | string): Promise<void> {
    // Security: Ensure path doesn't escape allowed directories
    const normalizedPath = path.normalize(fullPath);
    if (!normalizedPath.startsWith(NFS_BASE) && !normalizedPath.startsWith('/workspace')) {
      throw new Error('Path traversal not allowed');
    }

    // Ensure parent directory exists
    const dir = path.dirname(normalizedPath);
    await fs.mkdir(dir, { recursive: true });

    await fs.writeFile(normalizedPath, content);
    log.info('File written to NFS', { path: normalizedPath, size: Buffer.isBuffer(content) ? content.length : content.length });
  }

  /**
   * List all artifacts for a project
   */
  static async listProjectArtifacts(
    organizationId: string,
    projectId: string,
    artifactType?: ArtifactType
  ): Promise<string[]> {
    const results: string[] = [];

    try {
      const baseDir = artifactType
        ? path.join(NFS_BASE, organizationId, projectId, artifactType)
        : path.join(NFS_BASE, organizationId, projectId);

      const listDir = async (dir: string) => {
        try {
          const entries = await fs.readdir(dir, { withFileTypes: true });
          for (const entry of entries) {
            const fullPath = path.join(dir, entry.name);
            if (entry.isDirectory()) {
              await listDir(fullPath);
            } else {
              results.push(fullPath);
            }
          }
        } catch {
          // Directory doesn't exist
        }
      };

      await listDir(baseDir);
    } catch (error) {
      log.error('Failed to list project artifacts', error as Error, {
        organizationId,
        projectId,
        artifactType,
      });
    }

    return results;
  }

  /**
   * Get the schematic file path
   */
  static async getSchematicPath(
    organizationId: string,
    projectId: string,
    schematicId: string,
    extension: 'kicad_sch' | 'pdf' | 'svg' | 'png' = 'kicad_sch'
  ): Promise<string> {
    const dir = await this.getArtifactDir(organizationId, projectId, 'schematics');
    return path.join(dir, `${schematicId}.${extension}`);
  }

  /**
   * Get the PCB layout file path
   */
  static async getPCBLayoutPath(
    organizationId: string,
    projectId: string,
    layoutId: string,
    extension: 'kicad_pcb' | 'pdf' | 'svg' | 'png' = 'kicad_pcb'
  ): Promise<string> {
    const dir = await this.getArtifactDir(organizationId, projectId, 'pcb-layouts');
    return path.join(dir, `${layoutId}.${extension}`);
  }

  /**
   * Get the gerber directory path
   */
  static async getGerberDir(
    organizationId: string,
    projectId: string,
    layoutId: string
  ): Promise<string> {
    const dir = path.join(NFS_BASE, organizationId, projectId, 'pcb-layouts', layoutId, 'gerbers');
    await fs.mkdir(dir, { recursive: true });
    return dir;
  }

  /**
   * Get the simulation results path
   */
  static async getSimulationPath(
    organizationId: string,
    projectId: string,
    simulationId: string,
    extension: 'json' | 'csv' = 'json'
  ): Promise<string> {
    const dir = await this.getArtifactDir(organizationId, projectId, 'simulations');
    return path.join(dir, `${simulationId}_results.${extension}`);
  }

  /**
   * Get the firmware directory path
   */
  static async getFirmwareDir(
    organizationId: string,
    projectId: string,
    firmwareId: string
  ): Promise<string> {
    const dir = path.join(NFS_BASE, organizationId, projectId, 'firmware', firmwareId);
    await fs.mkdir(dir, { recursive: true });
    return dir;
  }

  /**
   * Copy a file to NFS storage
   */
  static async copyToNFS(
    sourcePath: string,
    organizationId: string,
    projectId: string,
    artifactType: ArtifactType,
    filename: string
  ): Promise<ArtifactPath> {
    try {
      const content = await fs.readFile(sourcePath);
      return this.storeArtifact(organizationId, projectId, artifactType, '', content, filename);
    } catch (error) {
      log.error('Failed to copy file to NFS', error as Error, {
        sourcePath,
        organizationId,
        projectId,
        artifactType,
        filename,
      });

      return {
        localPath: '',
        apiUrl: '',
        success: false,
      };
    }
  }
}

// Export convenience functions
export const getStorageStatus = NFSStorage.getStorageStatus.bind(NFSStorage);
export const storeArtifact = NFSStorage.storeArtifact.bind(NFSStorage);
export const getArtifact = NFSStorage.getArtifact.bind(NFSStorage);
export const artifactExists = NFSStorage.artifactExists.bind(NFSStorage);
export const deleteArtifact = NFSStorage.deleteArtifact.bind(NFSStorage);
