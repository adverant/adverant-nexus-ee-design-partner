/**
 * EE Design Partner - Virtual File Browser
 */

import { Request, Response, NextFunction } from 'express';
import { NotFoundError, ValidationError } from '../utils/errors.js';
import { log, Logger } from '../utils/logger.js';
import {
  findById as findProjectById,
} from '../database/repositories/project-repository.js';
import {
  findByProject as findSchematicsByProject,
} from '../database/repositories/schematic-repository.js';
import {
  findByProject as findPCBLayoutsByProject,
} from '../database/repositories/pcb-repository.js';
import {
  findByProject as findFirmwareByProject,
} from '../database/repositories/firmware-repository.js';
import {
  findByProject as findSimulationsByProject,
} from '../database/repositories/simulation-repository.js';

export interface FileNode {
  id: string;
  name: string;
  path: string;
  type: 'file' | 'directory';
  size?: number;
  modified?: string;
  metadata?: Record<string, unknown>;
  children?: FileNode[];
}

const fileLogger: Logger = log.child({ service: 'file-browser' });

/**
 * Build virtual file tree from project artifacts stored in PostgreSQL
 */
export async function buildVirtualFileTree(
  projectId: string,
  ownerId: string
): Promise<FileNode> {
  // Fetch project and validate access
  const project = await findProjectById(projectId);

  if (!project) {
    throw new NotFoundError('Project', projectId, { operation: 'buildVirtualFileTree' });
  }

  // Multi-tenant access validation
  const hasAccess =
    project.owner === ownerId ||
    (project.collaborators && project.collaborators.includes(ownerId));

  if (!hasAccess) {
    throw new ValidationError('Access denied to this project', { projectId, ownerId });
  }

  // Fetch all artifacts in parallel
  const [schematics, pcbLayouts, firmwareProjects, simulations] = await Promise.all([
    findSchematicsByProject(projectId, undefined),
    findPCBLayoutsByProject(projectId, undefined),
    findFirmwareByProject(projectId, undefined),
    findSimulationsByProject(projectId, undefined),
  ]);

  // Build virtual root directory
  const root: FileNode = {
    id: projectId,
    name: project.name || 'Project',
    path: '/',
    type: 'directory',
    children: [],
  };

  // Add schematics directory
  if (schematics && schematics.length > 0) {
    const schematicsDir: FileNode = {
      id: `${projectId}-schematics`,
      name: 'schematics',
      path: '/schematics',
      type: 'directory',
      children: schematics.map((schematic) => ({
        id: schematic.id,
        name: schematic.name || `schematic-v${schematic.version}.kicad_sch`,
        path: `/schematics/${schematic.name || `schematic-v${schematic.version}.kicad_sch`}`,
        type: 'file' as const,
        size: 0, // Size will be fetched when content is requested
        modified: schematic.updatedAt,
        metadata: {
          version: schematic.version,
          format: schematic.format || 'kicad',
          validationScore: schematic.validationResults?.score,
        },
      })),
    };
    root.children!.push(schematicsDir);
  }

  // Add PCB layouts directory
  if (pcbLayouts && pcbLayouts.length > 0) {
    const pcbDir: FileNode = {
      id: `${projectId}-pcb`,
      name: 'pcb',
      path: '/pcb',
      type: 'directory',
      children: pcbLayouts.map((pcb) => ({
        id: pcb.id,
        name: `layout-v${pcb.version}.kicad_pcb`,
        path: `/pcb/layout-v${pcb.version}.kicad_pcb`,
        type: 'file' as const,
        size: 0, // Size will be fetched when content is requested
        modified: pcb.updatedAt,
        metadata: {
          version: pcb.version,
          format: 'kicad',
          score: pcb.score,
          layers: pcb.stackup?.layers?.length,
        },
      })),
    };
    root.children!.push(pcbDir);
  }

  // Add firmware directory
  if (firmwareProjects && firmwareProjects.length > 0) {
    const firmwareDir: FileNode = {
      id: `${projectId}-firmware`,
      name: 'firmware',
      path: '/firmware',
      type: 'directory',
      children: [],
    };

    for (const firmware of firmwareProjects) {
      const projectDir: FileNode = {
        id: firmware.id,
        name: firmware.name || `firmware-${firmware.id}`,
        path: `/firmware/${firmware.name || firmware.id}`,
        type: 'directory',
        children: [],
      };

      // Parse generated files
      if (firmware.generatedFiles && Array.isArray(firmware.generatedFiles)) {
        projectDir.children = firmware.generatedFiles.map((file) => ({
          id: `${firmware.id}-${file.path}`,
          name: file.path.split('/').pop() || file.path,
          path: `/firmware/${firmware.name || firmware.id}/${file.path}`,
          type: 'file' as const,
          size: file.content?.length || 0,
          modified: file.generatedAt || firmware.updatedAt,
          metadata: {
            language: detectLanguage(file.path),
            fileType: file.type,
          },
        }));
      }

      firmwareDir.children!.push(projectDir);
    }

    root.children!.push(firmwareDir);
  }

  // Add simulations directory
  if (simulations && simulations.length > 0) {
    const simulationsDir: FileNode = {
      id: `${projectId}-simulations`,
      name: 'simulations',
      path: '/simulations',
      type: 'directory',
      children: simulations.map((simulation) => ({
        id: simulation.id,
        name: simulation.name || `${simulation.type}-${simulation.id}.json`,
        path: `/simulations/${simulation.name || `${simulation.type}-${simulation.id}.json`}`,
        type: 'file' as const,
        size: simulation.results ? JSON.stringify(simulation.results).length : 0,
        modified: simulation.completedAt || simulation.startedAt,
        metadata: {
          type: simulation.type,
          status: simulation.status,
        },
      })),
    };
    root.children!.push(simulationsDir);
  }

  return root;
}

/**
 * Get file content from database by path
 */
async function getFileContent(
  projectId: string,
  ownerId: string,
  filePath: string
): Promise<{ content: string; metadata?: Record<string, unknown> }> {
  // Validate access
  const project = await findProjectById(projectId);

  if (!project) {
    throw new NotFoundError('Project', projectId, { operation: 'getFileContent' });
  }

  const hasAccess =
    project.owner === ownerId ||
    (project.collaborators && project.collaborators.includes(ownerId));

  if (!hasAccess) {
    throw new ValidationError('Access denied', { projectId, ownerId });
  }

  // Parse path to determine artifact type
  const pathParts = filePath.split('/').filter(Boolean);

  if (pathParts.length === 0) {
    throw new ValidationError('Invalid file path', { filePath });
  }

  const category = pathParts[0];

  // Fetch content based on category
  if (category === 'schematics') {
    const schematics = await findSchematicsByProject(projectId, undefined);
    const schematic = schematics?.find((s) =>
      filePath.includes(`schematic-v${s.version}`)
    );

    if (!schematic) {
      throw new NotFoundError('Schematic', filePath, { projectId, filePath });
    }

    // Fetch actual KiCad content from database
    const { getKicadContent } = await import('../database/repositories/schematic-repository.js');
    const content = await getKicadContent(schematic.id);

    return {
      content: content || '',
      metadata: {
        version: schematic.version,
        format: schematic.format,
        validationScore: schematic.validationResults?.score,
      },
    };
  } else if (category === 'pcb') {
    const pcbLayouts = await findPCBLayoutsByProject(projectId, undefined);
    const pcb = pcbLayouts?.find((p) =>
      filePath.includes(`layout-v${p.version}`)
    );

    if (!pcb) {
      throw new NotFoundError('PCBLayout', filePath, { projectId, filePath });
    }

    // Fetch actual KiCad PCB content from database
    const { getKicadContent } = await import('../database/repositories/pcb-repository.js');
    const content = await getKicadContent(pcb.id);

    return {
      content: content || '',
      metadata: {
        version: pcb.version,
        format: 'kicad',
        score: pcb.score,
        layers: pcb.stackup?.layers?.length,
      },
    };
  } else if (category === 'firmware') {
    const firmwareProjects = await findFirmwareByProject(projectId, undefined);
    const firmwareProjectName = pathParts[1];
    const fileName = pathParts[2];

    const firmware = firmwareProjects?.find((f) =>
      f.name === firmwareProjectName || f.id === firmwareProjectName
    );

    if (!firmware) {
      throw new NotFoundError('FirmwareProject', firmwareProjectName, { projectId, firmwareProjectName });
    }

    if (firmware.generatedFiles && Array.isArray(firmware.generatedFiles)) {
      const file = firmware.generatedFiles.find((f) =>
        f.path.split('/').pop() === fileName || f.path === fileName
      );

      if (!file) {
        throw new NotFoundError('FirmwareFile', fileName, { firmwareId: firmware.id, fileName });
      }

      return {
        content: file.content || '',
        metadata: {
          language: detectLanguage(fileName),
          fileType: file.type,
        },
      };
    }

    throw new NotFoundError('FirmwareFile', filePath, { projectId, filePath });
  } else if (category === 'simulations') {
    const simulations = await findSimulationsByProject(projectId, undefined);
    const simulation = simulations?.find((s) =>
      filePath.includes(s.name || '') || filePath.includes(s.id)
    );

    if (!simulation) {
      throw new NotFoundError('Simulation', filePath, { projectId, filePath });
    }

    return {
      content: JSON.stringify(simulation.results || {}, null, 2),
      metadata: {
        type: simulation.type,
        status: simulation.status,
      },
    };
  }

  throw new NotFoundError('File', filePath, { projectId, filePath, category });
}

/**
 * Detect programming language from file extension
 */
function detectLanguage(fileName: string): string {
  const ext = fileName.split('.').pop()?.toLowerCase();
  const langMap: Record<string, string> = {
    c: 'c',
    h: 'c',
    cpp: 'cpp',
    hpp: 'cpp',
    py: 'python',
    js: 'javascript',
    ts: 'typescript',
    json: 'json',
    yaml: 'yaml',
    yml: 'yaml',
  };
  return langMap[ext || ''] || 'plaintext';
}

/**
 * Express route handler for GET /projects/:projectId/files/tree
 */
export async function getFileTreeHandler(req: Request, res: Response, next: NextFunction): Promise<void> {
  try {
    const projectId = req.params.projectId;
    const ownerId = String(req.headers['x-user-id'] || 'system');

    fileLogger.info('Fetching file tree', { projectId, ownerId });

    const tree = await buildVirtualFileTree(projectId, ownerId);

    res.json({
      success: true,
      data: {
        projectId,
        tree,
      }
    });
  } catch (err) {
    fileLogger.error('Failed to get file tree', err instanceof Error ? err : new Error(String(err)));
    next(err);
  }
}

/**
 * Express route handler for GET /projects/:projectId/files/content
 */
export async function getFileContentHandler(req: Request, res: Response, next: NextFunction): Promise<void> {
  try {
    const projectId = req.params.projectId;
    const ownerId = String(req.headers['x-user-id'] || 'system');
    const filePath = String(req.query.path || '');

    if (!filePath) {
      throw new ValidationError('File path is required', { projectId });
    }

    fileLogger.info('Fetching file content', { projectId, ownerId, filePath });

    const result = await getFileContent(projectId, ownerId, filePath);

    res.json({
      success: true,
      data: {
        path: filePath,
        content: result.content,
        metadata: result.metadata,
      }
    });
  } catch (err) {
    fileLogger.error('Failed to get file content', err instanceof Error ? err : new Error(String(err)));
    next(err);
  }
}
