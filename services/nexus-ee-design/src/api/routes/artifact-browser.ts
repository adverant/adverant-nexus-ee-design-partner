/**
 * Artifact Browser Routes
 *
 * Unified NFS + DB artifact browsing endpoints for the ArtifactBrowser UI.
 * Reads symbol-assembly outputs from NFS disk and schematics from PostgreSQL,
 * presenting them as a single folder/file tree.
 */

import { Router, Request, Response, NextFunction } from 'express';
import fs from 'fs/promises';
import fsSync from 'fs';
import path from 'path';
import multer from 'multer';
import { log } from '../../utils/logger.js';
import { config } from '../../config.js';
import { NFSStorage } from '../../utils/nfs-storage.js';
import { ValidationError } from '../../utils/errors.js';

import {
  findByProject as findSchematicsByProject,
  update as updateSchematic,
  deleteSchematic,
} from '../../database/repositories/schematic-repository.js';

// Multer for file uploads (memory storage, 50MB limit)
const upload = multer({
  storage: multer.memoryStorage(),
  limits: { fileSize: 50 * 1024 * 1024 },
});

/**
 * Content-type mapping for common EE design file extensions
 */
const EXTENSION_CONTENT_TYPES: Record<string, string> = {
  '.kicad_sym': 'text/plain; charset=utf-8',
  '.kicad_sch': 'text/plain; charset=utf-8',
  '.kicad_pcb': 'text/plain; charset=utf-8',
  '.pdf': 'application/pdf',
  '.md': 'text/markdown; charset=utf-8',
  '.txt': 'text/plain; charset=utf-8',
  '.json': 'application/json; charset=utf-8',
  '.csv': 'text/csv; charset=utf-8',
  '.svg': 'image/svg+xml',
  '.png': 'image/png',
  '.jpg': 'image/jpeg',
  '.jpeg': 'image/jpeg',
  '.zip': 'application/zip',
};

/**
 * Inline-viewable content types (display in browser instead of download)
 */
const INLINE_CONTENT_TYPES = new Set([
  'application/pdf',
  'text/plain; charset=utf-8',
  'text/markdown; charset=utf-8',
  'application/json; charset=utf-8',
  'text/csv; charset=utf-8',
  'image/svg+xml',
  'image/png',
  'image/jpeg',
]);

/**
 * Get the NFS artifacts base path for a project.
 * Uses ARTIFACT_STORAGE_PATH (the NFS mount, /data/artifacts in production).
 */
function getProjectNfsPath(projectId: string): string {
  const basePath = config.artifacts.basePath;
  return path.join(basePath, projectId);
}

/**
 * Validate that a resolved path stays within the artifacts base directory.
 * Prevents path traversal attacks.
 */
function validatePath(resolvedPath: string): boolean {
  const basePath = path.resolve(config.artifacts.basePath);
  const normalized = path.resolve(resolvedPath);
  return normalized.startsWith(basePath);
}

/**
 * Infer a tag from the NFS subdirectory path
 */
function inferTag(relativePath: string): string {
  const parts = relativePath.split('/');
  // e.g. symbol-assembly/symbols → "symbol", symbol-assembly/datasheets → "datasheet"
  if (parts.length >= 2) {
    const subdir = parts[1];
    // Remove trailing 's' for singular tag
    return subdir.endsWith('s') ? subdir.slice(0, -1) : subdir;
  }
  return 'artifact';
}

export function createArtifactBrowserRoutes(): Router {
  const router = Router({ mergeParams: true });

  // ============================================================================
  // GET /folders — Scan NFS disk + query DB to build folder tree
  // ============================================================================
  router.get('/folders', async (req: Request, res: Response, next: NextFunction) => {
    try {
      const { projectId } = req.params;
      if (!projectId) {
        throw new ValidationError('projectId is required');
      }

      const projectPath = getProjectNfsPath(projectId);
      const folders: Array<{
        id: string;
        projectId: string;
        parentId: string | null;
        name: string;
        path: string;
        artifactType: string;
        childCount: number;
        createdAt: string;
        updatedAt: string;
        children?: Array<{
          id: string;
          projectId: string;
          parentId: string;
          name: string;
          path: string;
          artifactType: string;
          childCount: number;
          createdAt: string;
          updatedAt: string;
        }>;
      }> = [];

      const now = new Date().toISOString();

      // 1. Count DB schematics
      let dbSchematicCount = 0;
      try {
        const schematics = await findSchematicsByProject(projectId);
        dbSchematicCount = schematics.length;
      } catch (err) {
        log.warn('Failed to count DB schematics', { projectId, error: (err as Error).message });
      }

      // 2. Scan NFS directories
      interface NfsSubfolder {
        name: string;
        dirName: string;
        fileCount: number;
      }

      const nfsFolders: Array<{ name: string; dirName: string; subfolders: NfsSubfolder[]; totalFiles: number }> = [];

      try {
        const entries = await fs.readdir(projectPath, { withFileTypes: true });
        for (const entry of entries) {
          if (!entry.isDirectory()) continue;

          const dirPath = path.join(projectPath, entry.name);
          const subfolders: NfsSubfolder[] = [];
          let totalFiles = 0;

          try {
            const subEntries = await fs.readdir(dirPath, { withFileTypes: true });
            for (const sub of subEntries) {
              if (sub.isDirectory()) {
                const subPath = path.join(dirPath, sub.name);
                try {
                  const files = await fs.readdir(subPath);
                  subfolders.push({
                    name: sub.name.charAt(0).toUpperCase() + sub.name.slice(1),
                    dirName: sub.name,
                    fileCount: files.length,
                  });
                  totalFiles += files.length;
                } catch {
                  subfolders.push({ name: sub.name, dirName: sub.name, fileCount: 0 });
                }
              } else {
                totalFiles++;
              }
            }
          } catch {
            // Empty or unreadable directory
          }

          nfsFolders.push({
            name: entry.name.split('-').map(w => w.charAt(0).toUpperCase() + w.slice(1)).join(' '),
            dirName: entry.name,
            subfolders,
            totalFiles,
          });
        }
      } catch {
        // Project NFS directory doesn't exist yet - that's fine
        log.debug('NFS project directory not found (expected for new projects)', { projectPath });
      }

      // 3. Build folder tree

      // Root
      const rootChildCount = dbSchematicCount + nfsFolders.reduce((sum, f) => sum + f.totalFiles, 0);
      const rootFolder = {
        id: 'root',
        projectId,
        parentId: null,
        name: 'All Artifacts',
        path: '/',
        artifactType: 'schematic',
        childCount: rootChildCount,
        createdAt: now,
        updatedAt: now,
        children: [] as Array<{
          id: string;
          projectId: string;
          parentId: string;
          name: string;
          path: string;
          artifactType: string;
          childCount: number;
          createdAt: string;
          updatedAt: string;
          children?: Array<{
            id: string;
            projectId: string;
            parentId: string;
            name: string;
            path: string;
            artifactType: string;
            childCount: number;
            createdAt: string;
            updatedAt: string;
          }>;
        }>,
      };

      // DB Schematics folder
      if (dbSchematicCount > 0) {
        rootFolder.children.push({
          id: 'db:schematics',
          projectId,
          parentId: 'root',
          name: 'Schematics',
          path: '/schematics',
          artifactType: 'schematic',
          childCount: dbSchematicCount,
          createdAt: now,
          updatedAt: now,
        });
      }

      // NFS folders
      for (const nfsFolder of nfsFolders) {
        const folderId = `nfs:${nfsFolder.dirName}`;
        const folderNode = {
          id: folderId,
          projectId,
          parentId: 'root',
          name: nfsFolder.name,
          path: `/${nfsFolder.dirName}`,
          artifactType: 'schematic',
          childCount: nfsFolder.totalFiles,
          createdAt: now,
          updatedAt: now,
          children: nfsFolder.subfolders.map(sub => ({
            id: `nfs:${nfsFolder.dirName}/${sub.dirName}`,
            projectId,
            parentId: folderId,
            name: sub.name,
            path: `/${nfsFolder.dirName}/${sub.dirName}`,
            artifactType: 'schematic',
            childCount: sub.fileCount,
            createdAt: now,
            updatedAt: now,
          })),
        };
        rootFolder.children.push(folderNode);
      }

      folders.push(rootFolder);

      res.json({
        success: true,
        data: folders,
      });
    } catch (error) {
      next(error);
    }
  });

  // ============================================================================
  // GET /artifacts — Paginated, unified artifact list from NFS + DB
  // ============================================================================
  router.get('/artifacts', async (req: Request, res: Response, next: NextFunction) => {
    try {
      const { projectId } = req.params;
      if (!projectId) {
        throw new ValidationError('projectId is required');
      }

      const folderId = (req.query.folderId as string) || 'root';
      const page = Math.max(1, parseInt(req.query.page as string) || 1);
      const pageSize = Math.min(100, Math.max(1, parseInt(req.query.pageSize as string) || 20));
      const search = (req.query.search as string) || '';
      const sortBy = (req.query.sortBy as string) || 'name';
      const sortOrder = ((req.query.sortOrder as string) || 'asc').toLowerCase() as 'asc' | 'desc';

      interface ArtifactItem {
        id: string;
        projectId: string;
        folderId: string | null;
        name: string;
        version: number;
        artifactType: string;
        status: string;
        filePath: string;
        fileSize: number;
        tags: string[];
        createdBy: string;
        createdAt: string;
        updatedAt: string;
        metadata?: Record<string, unknown>;
      }

      let allArtifacts: ArtifactItem[] = [];

      // Gather DB schematics
      const includeDbSchematics = folderId === 'root' || folderId === 'db:schematics';
      if (includeDbSchematics) {
        try {
          const schematics = await findSchematicsByProject(projectId);
          for (const s of schematics) {
            allArtifacts.push({
              id: s.id,
              projectId,
              folderId: 'db:schematics',
              name: s.name,
              version: s.version || 1,
              artifactType: 'schematic',
              status: 'draft',
              filePath: `/api/v1/projects/${projectId}/artifact-browser/download/${s.id}`,
              fileSize: s.kicadSch ? Buffer.byteLength(s.kicadSch, 'utf-8') : 0,
              tags: ['schematic'],
              createdBy: 'system',
              createdAt: s.createdAt || new Date().toISOString(),
              updatedAt: s.updatedAt || new Date().toISOString(),
              metadata: { source: 'database' },
            });
          }
        } catch (err) {
          log.warn('Failed to fetch DB schematics', { projectId, error: (err as Error).message });
        }
      }

      // Gather NFS files
      const includeNfs = folderId === 'root' || folderId.startsWith('nfs:');
      if (includeNfs) {
        const projectPath = getProjectNfsPath(projectId);
        let nfsRelPath: string;

        if (folderId === 'root') {
          nfsRelPath = '';
        } else {
          // folderId = "nfs:symbol-assembly/symbols" → relative path = "symbol-assembly/symbols"
          nfsRelPath = folderId.replace(/^nfs:/, '');
        }

        const nfsDirPath = path.join(projectPath, nfsRelPath);

        if (validatePath(nfsDirPath)) {
          try {
            const collectFiles = async (dirPath: string, relBase: string) => {
              const entries = await fs.readdir(dirPath, { withFileTypes: true });
              for (const entry of entries) {
                const entryPath = path.join(dirPath, entry.name);
                const entryRel = relBase ? `${relBase}/${entry.name}` : entry.name;

                if (entry.isDirectory()) {
                  // For root folder, recurse into subdirectories
                  if (folderId === 'root') {
                    await collectFiles(entryPath, entryRel);
                  }
                } else {
                  try {
                    const stats = await fs.stat(entryPath);
                    const ext = path.extname(entry.name);
                    const artifactId = `nfs:${nfsRelPath ? nfsRelPath + '/' : ''}${entryRel}`;
                    const tag = inferTag(nfsRelPath || entryRel);

                    allArtifacts.push({
                      id: artifactId,
                      projectId,
                      folderId: folderId === 'root' ? null : folderId,
                      name: entry.name,
                      version: 1,
                      artifactType: 'schematic',
                      status: 'released',
                      filePath: `/api/v1/projects/${projectId}/artifact-browser/download/${encodeURIComponent(artifactId)}`,
                      fileSize: stats.size,
                      tags: [tag],
                      createdBy: 'symbol-assembly',
                      createdAt: stats.birthtime.toISOString(),
                      updatedAt: stats.mtime.toISOString(),
                      metadata: { source: 'nfs', nfsPath: entryPath },
                    });
                  } catch {
                    // Skip unreadable files
                  }
                }
              }
            };

            await collectFiles(nfsDirPath, '');
          } catch {
            // NFS directory doesn't exist - return empty
            log.debug('NFS directory not found for artifacts listing', { nfsDirPath });
          }
        }
      }

      // Apply search filter
      if (search) {
        const lowerSearch = search.toLowerCase();
        allArtifacts = allArtifacts.filter(a => a.name.toLowerCase().includes(lowerSearch));
      }

      // Sort
      allArtifacts.sort((a, b) => {
        let cmp = 0;
        switch (sortBy) {
          case 'name':
            cmp = a.name.localeCompare(b.name);
            break;
          case 'updatedAt':
            cmp = new Date(a.updatedAt).getTime() - new Date(b.updatedAt).getTime();
            break;
          case 'createdAt':
            cmp = new Date(a.createdAt).getTime() - new Date(b.createdAt).getTime();
            break;
          case 'fileSize':
            cmp = a.fileSize - b.fileSize;
            break;
          default:
            cmp = a.name.localeCompare(b.name);
        }
        return sortOrder === 'desc' ? -cmp : cmp;
      });

      // Paginate
      const total = allArtifacts.length;
      const offset = (page - 1) * pageSize;
      const data = allArtifacts.slice(offset, offset + pageSize);

      res.json({
        success: true,
        data,
        total,
        page,
        pageSize,
      });
    } catch (error) {
      next(error);
    }
  });

  // ============================================================================
  // GET /download/:artifactId — Serve file for download/preview
  // ============================================================================
  router.get('/download/:artifactId(*)', async (req: Request, res: Response, next: NextFunction) => {
    try {
      const { projectId, artifactId } = req.params;
      if (!projectId || !artifactId) {
        throw new ValidationError('projectId and artifactId are required');
      }

      const decodedId = decodeURIComponent(artifactId);

      if (decodedId.startsWith('nfs:')) {
        // NFS file: decode path and serve from disk
        const relativePath = decodedId.replace(/^nfs:/, '');
        const projectPath = getProjectNfsPath(projectId);
        const filePath = path.join(projectPath, relativePath);
        const resolvedPath = path.resolve(filePath);

        // Path traversal protection
        if (!validatePath(resolvedPath)) {
          throw new ValidationError('Invalid artifact path');
        }

        // Check file exists
        try {
          await fs.access(resolvedPath);
        } catch {
          res.status(404).json({ success: false, error: { code: 'NOT_FOUND', message: 'Artifact file not found' } });
          return;
        }

        const ext = path.extname(resolvedPath).toLowerCase();
        const contentType = EXTENSION_CONTENT_TYPES[ext] || 'application/octet-stream';
        const filename = path.basename(resolvedPath);
        const disposition = INLINE_CONTENT_TYPES.has(contentType) ? 'inline' : 'attachment';

        res.setHeader('Content-Type', contentType);
        res.setHeader('Content-Disposition', `${disposition}; filename="${filename}"`);

        const stream = fsSync.createReadStream(resolvedPath);
        stream.pipe(res);
      } else {
        // DB schematic: fetch kicad content
        const { findById: findSchematicById } = await import(
          '../../database/repositories/schematic-repository.js'
        );

        const schematic = await findSchematicById(decodedId);
        if (!schematic) {
          res.status(404).json({ success: false, error: { code: 'NOT_FOUND', message: 'Schematic not found' } });
          return;
        }

        if (schematic.projectId !== projectId) {
          throw new ValidationError('Schematic does not belong to this project');
        }

        const content = schematic.kicadSch || '';
        res.setHeader('Content-Type', 'text/plain; charset=utf-8');
        res.setHeader('Content-Disposition', `inline; filename="${schematic.name}.kicad_sch"`);
        res.send(content);
      }
    } catch (error) {
      next(error);
    }
  });

  // ============================================================================
  // POST /folders — Create folder on NFS
  // ============================================================================
  router.post('/folders', async (req: Request, res: Response, next: NextFunction) => {
    try {
      const { projectId } = req.params;
      if (!projectId) {
        throw new ValidationError('projectId is required');
      }

      const { name, parentPath } = req.body;
      if (!name) {
        throw new ValidationError('Folder name is required');
      }

      // Sanitize folder name
      const safeName = name.replace(/[^a-zA-Z0-9_\-. ]/g, '_');
      const projectPath = getProjectNfsPath(projectId);
      const parentDir = parentPath ? path.join(projectPath, parentPath) : projectPath;
      const folderPath = path.join(parentDir, safeName);

      if (!validatePath(folderPath)) {
        throw new ValidationError('Invalid folder path');
      }

      await NFSStorage.createDirectory(folderPath);

      const now = new Date().toISOString();
      const relativePath = path.relative(projectPath, folderPath);
      const folderId = `nfs:${relativePath}`;

      res.status(201).json({
        success: true,
        data: {
          id: folderId,
          projectId,
          parentId: parentPath ? `nfs:${parentPath}` : 'root',
          name: safeName,
          path: `/${relativePath}`,
          artifactType: 'schematic',
          childCount: 0,
          createdAt: now,
          updatedAt: now,
        },
      });
    } catch (error) {
      next(error);
    }
  });

  // ============================================================================
  // POST /upload — Upload file to NFS folder
  // ============================================================================
  router.post('/upload', upload.single('file'), async (req: Request, res: Response, next: NextFunction) => {
    try {
      const { projectId } = req.params;
      if (!projectId) {
        throw new ValidationError('projectId is required');
      }

      const file = req.file;
      if (!file) {
        throw new ValidationError('File is required');
      }

      const folderId = req.body.folderId || '';
      const projectPath = getProjectNfsPath(projectId);
      let targetDir: string;

      if (folderId.startsWith('nfs:')) {
        const relativePath = folderId.replace(/^nfs:/, '');
        targetDir = path.join(projectPath, relativePath);
      } else {
        targetDir = projectPath;
      }

      const filePath = path.join(targetDir, file.originalname);

      if (!validatePath(filePath)) {
        throw new ValidationError('Invalid upload path');
      }

      await NFSStorage.writeFile(filePath, file.buffer);

      const stats = await fs.stat(filePath);
      const relativePath = path.relative(projectPath, filePath);
      const artifactId = `nfs:${relativePath}`;
      const tag = inferTag(relativePath);

      res.status(201).json({
        success: true,
        data: {
          id: artifactId,
          projectId,
          folderId: folderId || null,
          name: file.originalname,
          version: 1,
          artifactType: 'schematic',
          status: 'draft',
          filePath: `/api/v1/projects/${projectId}/artifact-browser/download/${encodeURIComponent(artifactId)}`,
          fileSize: stats.size,
          tags: [tag],
          createdBy: 'upload',
          createdAt: stats.birthtime.toISOString(),
          updatedAt: stats.mtime.toISOString(),
          metadata: { source: 'nfs', nfsPath: filePath },
        },
      });
    } catch (error) {
      next(error);
    }
  });

  // ============================================================================
  // PATCH /artifacts/:artifactId — Rename artifact
  // ============================================================================
  router.patch('/artifacts/:artifactId(*)', async (req: Request, res: Response, next: NextFunction) => {
    try {
      const { projectId, artifactId } = req.params;
      if (!projectId || !artifactId) {
        throw new ValidationError('projectId and artifactId are required');
      }

      const { name } = req.body;
      if (!name) {
        throw new ValidationError('New name is required');
      }

      const decodedId = decodeURIComponent(artifactId);

      if (decodedId.startsWith('nfs:')) {
        const relativePath = decodedId.replace(/^nfs:/, '');
        const projectPath = getProjectNfsPath(projectId);
        const oldPath = path.join(projectPath, relativePath);
        const newPath = path.join(path.dirname(oldPath), name);

        if (!validatePath(oldPath) || !validatePath(newPath)) {
          throw new ValidationError('Invalid artifact path');
        }

        await fs.rename(oldPath, newPath);

        const newRelPath = path.relative(projectPath, newPath);
        res.json({
          success: true,
          data: { id: `nfs:${newRelPath}`, name },
        });
      } else {
        // DB schematic
        await updateSchematic(decodedId, { name });
        res.json({
          success: true,
          data: { id: decodedId, name },
        });
      }
    } catch (error) {
      next(error);
    }
  });

  // ============================================================================
  // DELETE /artifacts/:artifactId — Delete artifact
  // ============================================================================
  router.delete('/artifacts/:artifactId(*)', async (req: Request, res: Response, next: NextFunction) => {
    try {
      const { projectId, artifactId } = req.params;
      if (!projectId || !artifactId) {
        throw new ValidationError('projectId and artifactId are required');
      }

      const decodedId = decodeURIComponent(artifactId);

      if (decodedId.startsWith('nfs:')) {
        const relativePath = decodedId.replace(/^nfs:/, '');
        const projectPath = getProjectNfsPath(projectId);
        const filePath = path.join(projectPath, relativePath);

        if (!validatePath(filePath)) {
          throw new ValidationError('Invalid artifact path');
        }

        const deleted = await NFSStorage.deleteArtifact(filePath);
        if (!deleted) {
          res.status(404).json({ success: false, error: { code: 'NOT_FOUND', message: 'Artifact file not found or could not be deleted' } });
          return;
        }

        res.json({ success: true, data: { deleted: true, artifactId: decodedId } });
      } else {
        // DB schematic
        await deleteSchematic(decodedId);
        res.json({ success: true, data: { deleted: true, artifactId: decodedId } });
      }
    } catch (error) {
      next(error);
    }
  });

  // ============================================================================
  // DELETE /folders/:folderId — Delete NFS folder
  // ============================================================================
  router.delete('/folders/:folderId(*)', async (req: Request, res: Response, next: NextFunction) => {
    try {
      const { projectId, folderId } = req.params;
      if (!projectId || !folderId) {
        throw new ValidationError('projectId and folderId are required');
      }

      const force = req.query.force === 'true';
      const decodedId = decodeURIComponent(folderId);

      if (!decodedId.startsWith('nfs:')) {
        throw new ValidationError('Can only delete NFS folders');
      }

      const relativePath = decodedId.replace(/^nfs:/, '');
      const projectPath = getProjectNfsPath(projectId);
      const folderPath = path.join(projectPath, relativePath);

      if (!validatePath(folderPath)) {
        throw new ValidationError('Invalid folder path');
      }

      try {
        await fs.access(folderPath);
      } catch {
        res.status(404).json({ success: false, error: { code: 'NOT_FOUND', message: 'Folder not found' } });
          return;
      }

      if (force) {
        await fs.rm(folderPath, { recursive: true });
      } else {
        // Only delete empty folders
        const entries = await fs.readdir(folderPath);
        if (entries.length > 0) {
          throw new ValidationError('Folder is not empty. Use ?force=true to delete non-empty folders.');
        }
        await fs.rmdir(folderPath);
      }

      log.info('Folder deleted from NFS', { folderPath, force });

      res.json({ success: true, data: { deleted: true, folderId: decodedId } });
    } catch (error) {
      next(error);
    }
  });

  return router;
}
