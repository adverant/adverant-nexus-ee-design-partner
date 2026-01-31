/**
 * EE Design Client - TypeScript client for terminal/CLI access
 *
 * Provides full read/write/execute/admin access to the EE Design API
 * for Claude Code CLI and other programmatic tools.
 */

import { config } from '../config.js';

export interface ClientConfig {
  baseUrl?: string;
  userId?: string;
  timeout?: number;
}

export interface Project {
  id: string;
  name: string;
  description?: string;
  status: string;
  ownerId: string;
  createdAt: string;
  updatedAt: string;
}

export interface Schematic {
  id: string;
  projectId: string;
  name: string;
  kicadSch?: string;
  status: string;
  createdAt: string;
}

export interface PCBLayout {
  id: string;
  projectId: string;
  schematicId: string;
  name: string;
  kicadPcb?: string;
  status: string;
  createdAt: string;
}

export interface DRCResult {
  passed: boolean;
  violations: Array<{
    code: string;
    severity: 'error' | 'warning';
    message: string;
    location?: { x: number; y: number };
  }>;
  totalViolations: number;
}

export interface BOMItem {
  id: string;
  reference: string;
  partNumber: string;
  manufacturer: string;
  description: string;
  quantity: number;
  footprint: string;
  value?: string;
}

/**
 * EE Design API Client with full CLI access
 */
export class EEDesignClient {
  private baseUrl: string;
  private userId: string;
  private timeout: number;

  constructor(options: ClientConfig = {}) {
    this.baseUrl = options.baseUrl || `http://localhost:${config.port}`;
    this.userId = options.userId || config.cliAccess.systemUserId;
    this.timeout = options.timeout || 30000;
  }

  private async request<T>(
    method: string,
    endpoint: string,
    body?: unknown
  ): Promise<T> {
    const url = `${this.baseUrl}/api/v1${endpoint}`;

    const headers: Record<string, string> = {
      'Content-Type': 'application/json',
      'X-User-Id': this.userId,
      'X-CLI-Access': 'true',
      'X-Claude-Code': 'true',
    };

    const options: RequestInit = {
      method,
      headers,
      signal: AbortSignal.timeout(this.timeout),
    };

    if (body && method !== 'GET') {
      options.body = JSON.stringify(body);
    }

    const response = await fetch(url, options);

    if (!response.ok) {
      const error = await response.json().catch(() => ({ message: response.statusText })) as { message?: string };
      throw new Error(`API Error ${response.status}: ${error.message || response.statusText}`);
    }

    return response.json() as Promise<T>;
  }

  // ============================================================================
  // Health & Status
  // ============================================================================

  async health(): Promise<{ status: string; version: string }> {
    const response = await fetch(`${this.baseUrl}/health`);
    return response.json() as Promise<{ status: string; version: string }>;
  }

  // ============================================================================
  // Projects
  // ============================================================================

  async listProjects(): Promise<{ projects: Project[] }> {
    return this.request('GET', '/projects');
  }

  async getProject(projectId: string): Promise<Project> {
    return this.request('GET', `/projects/${projectId}`);
  }

  async createProject(name: string, description?: string): Promise<Project> {
    return this.request('POST', '/projects', { name, description });
  }

  async updateProject(projectId: string, updates: Partial<Project>): Promise<Project> {
    return this.request('PUT', `/projects/${projectId}`, updates);
  }

  async deleteProject(projectId: string): Promise<void> {
    return this.request('DELETE', `/projects/${projectId}`);
  }

  // ============================================================================
  // Schematics
  // ============================================================================

  async generateSchematic(projectId: string, requirements: string): Promise<Schematic> {
    return this.request('POST', `/projects/${projectId}/schematic/generate`, { requirements });
  }

  async getSchematic(projectId: string, schematicId: string): Promise<Schematic> {
    return this.request('GET', `/projects/${projectId}/schematic/${schematicId}`);
  }

  async listSchematics(projectId: string): Promise<{ schematics: Schematic[] }> {
    return this.request('GET', `/projects/${projectId}/schematics`);
  }

  async downloadSchematicFile(projectId: string, schematicId: string): Promise<string> {
    const url = `${this.baseUrl}/api/v1/projects/${projectId}/schematic/${schematicId}/schematic.kicad_sch`;
    const response = await fetch(url, {
      headers: {
        'X-User-Id': this.userId,
        'X-CLI-Access': 'true',
      },
    });
    return response.text();
  }

  // ============================================================================
  // PCB Layouts
  // ============================================================================

  async generatePCBLayout(projectId: string, schematicId: string): Promise<PCBLayout> {
    return this.request('POST', `/projects/${projectId}/pcb-layout/generate`, { schematicId });
  }

  async getPCBLayout(projectId: string, layoutId: string): Promise<PCBLayout> {
    return this.request('GET', `/projects/${projectId}/pcb-layout/${layoutId}`);
  }

  async listPCBLayouts(projectId: string): Promise<{ layouts: PCBLayout[] }> {
    return this.request('GET', `/projects/${projectId}/pcb-layouts`);
  }

  async downloadPCBFile(projectId: string, layoutId: string): Promise<string> {
    const url = `${this.baseUrl}/api/v1/projects/${projectId}/pcb-layout/${layoutId}/board.kicad_pcb`;
    const response = await fetch(url, {
      headers: {
        'X-User-Id': this.userId,
        'X-CLI-Access': 'true',
      },
    });
    return response.text();
  }

  // ============================================================================
  // DRC & Validation
  // ============================================================================

  async runDRC(projectId: string, layoutId: string): Promise<DRCResult> {
    return this.request('POST', `/projects/${projectId}/pcb-layout/${layoutId}/drc`);
  }

  async runERC(projectId: string, schematicId: string): Promise<DRCResult> {
    return this.request('POST', `/projects/${projectId}/schematic/${schematicId}/erc`);
  }

  // ============================================================================
  // 3D Export
  // ============================================================================

  async export3DModel(
    projectId: string,
    layoutId: string,
    format: 'step' | 'wrl' = 'step'
  ): Promise<ArrayBuffer> {
    const ext = format === 'wrl' ? 'wrl' : 'step';
    const url = `${this.baseUrl}/api/v1/projects/${projectId}/pcb-layout/${layoutId}/board.${ext}`;
    const response = await fetch(url, {
      headers: {
        'X-User-Id': this.userId,
        'X-CLI-Access': 'true',
      },
    });
    return response.arrayBuffer();
  }

  // ============================================================================
  // BOM
  // ============================================================================

  async getBOM(projectId: string): Promise<{ items: BOMItem[] }> {
    return this.request('GET', `/projects/${projectId}/bom`);
  }

  async addBOMItem(projectId: string, item: Omit<BOMItem, 'id'>): Promise<BOMItem> {
    return this.request('POST', `/projects/${projectId}/bom`, item);
  }

  async updateBOMItem(projectId: string, itemId: string, updates: Partial<BOMItem>): Promise<BOMItem> {
    return this.request('PUT', `/projects/${projectId}/bom/${itemId}`, updates);
  }

  async deleteBOMItem(projectId: string, itemId: string): Promise<void> {
    return this.request('DELETE', `/projects/${projectId}/bom/${itemId}`);
  }

  // ============================================================================
  // Simulations
  // ============================================================================

  async runSimulation(
    projectId: string,
    type: 'spice' | 'thermal' | 'emc',
    params?: Record<string, unknown>
  ): Promise<unknown> {
    return this.request('POST', `/projects/${projectId}/simulation/${type}`, params || {});
  }

  async getSimulationResult(projectId: string, simulationId: string): Promise<unknown> {
    return this.request('GET', `/projects/${projectId}/simulation/${simulationId}`);
  }

  // ============================================================================
  // Firmware
  // ============================================================================

  async generateFirmware(projectId: string, options?: Record<string, unknown>): Promise<unknown> {
    return this.request('POST', `/projects/${projectId}/firmware/generate`, options || {});
  }
}

/**
 * Create a pre-configured client instance
 */
export function createEEDesignClient(options?: ClientConfig): EEDesignClient {
  return new EEDesignClient(options);
}

/**
 * Default client instance for quick access
 */
export const eeDesignClient = new EEDesignClient();
