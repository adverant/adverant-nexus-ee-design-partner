/**
 * EE Design Partner - API Client
 *
 * Centralized API client for all backend communication.
 * Implements proper error handling, request interceptors, and type safety.
 */

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:9080';

// ============================================================================
// Types
// ============================================================================

export interface ApiResponse<T = unknown> {
  success: boolean;
  data?: T;
  error?: {
    code: string;
    message: string;
    context?: Record<string, unknown>;
  };
  metadata?: {
    requestId: string;
    timestamp: string;
  };
}

export interface Project {
  id: string;
  name: string;
  description?: string;
  type: string;
  phase: string;
  status: 'draft' | 'in_progress' | 'completed';
  completedPhases: string[];
  createdAt: string;
  updatedAt: string;
}

export interface FileNode {
  name: string;
  path: string;
  type: 'file' | 'directory';
  size?: number;
  modifiedAt?: string;
  children?: FileNode[];
  extension?: string;
}

export interface ValidationDomain {
  id: string;
  name: string;
  status: 'passed' | 'warning' | 'failed' | 'pending' | 'running';
  score: number;
  issues: number;
  details?: string;
  validators?: ValidatorResult[];
}

export interface ValidatorResult {
  validatorId: string;
  validatorName: string;
  score: number;
  weight: number;
  feedback?: string;
}

export interface ValidationResult {
  projectId: string;
  domains: ValidationDomain[];
  overallScore: number;
  overallStatus: 'passed' | 'warning' | 'failed' | 'pending';
  timestamp: string;
}

export interface CommandExecutionResult {
  commandId: string;
  status: 'queued' | 'running' | 'completed' | 'failed';
  output?: string;
  error?: string;
  startedAt?: string;
  completedAt?: string;
}

export interface SkillExecutionRequest {
  skillName: string;
  capability?: string;
  parameters: Record<string, unknown>;
  projectId?: string;
}

// ============================================================================
// Error Classes
// ============================================================================

export class ApiError extends Error {
  constructor(
    public readonly code: string,
    message: string,
    public readonly statusCode?: number,
    public readonly context?: Record<string, unknown>
  ) {
    super(message);
    this.name = 'ApiError';
  }
}

export class NetworkError extends ApiError {
  constructor(message: string, originalError?: Error) {
    super('NETWORK_ERROR', message, undefined, {
      originalError: originalError?.message,
    });
    this.name = 'NetworkError';
  }
}

export class ValidationApiError extends ApiError {
  constructor(message: string, validationErrors?: unknown[]) {
    super('VALIDATION_ERROR', message, 400, { validationErrors });
    this.name = 'ValidationApiError';
  }
}

// ============================================================================
// API Client Implementation
// ============================================================================

class EEDesignApiClient {
  private baseUrl: string;
  private defaultHeaders: Record<string, string>;

  constructor() {
    this.baseUrl = `${API_BASE_URL}/api/v1`;
    this.defaultHeaders = {
      'Content-Type': 'application/json',
    };
  }

  /**
   * Set authentication token
   */
  setAuthToken(token: string): void {
    this.defaultHeaders['Authorization'] = `Bearer ${token}`;
  }

  /**
   * Core fetch wrapper with error handling
   */
  private async request<T>(
    method: 'GET' | 'POST' | 'PUT' | 'PATCH' | 'DELETE',
    endpoint: string,
    body?: unknown,
    options?: { timeout?: number }
  ): Promise<ApiResponse<T>> {
    const url = `${this.baseUrl}${endpoint}`;
    const timeout = options?.timeout || 30000;

    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), timeout);

    try {
      const requestOptions: RequestInit = {
        method,
        headers: this.defaultHeaders,
        signal: controller.signal,
      };

      if (body && method !== 'GET') {
        requestOptions.body = JSON.stringify(body);
      }

      const response = await fetch(url, requestOptions);
      clearTimeout(timeoutId);

      const data = await response.json();

      if (!response.ok) {
        throw new ApiError(
          data.error?.code || 'API_ERROR',
          data.error?.message || `HTTP ${response.status}: ${response.statusText}`,
          response.status,
          data.error?.context
        );
      }

      return data as ApiResponse<T>;
    } catch (error) {
      clearTimeout(timeoutId);

      if (error instanceof ApiError) {
        throw error;
      }

      if (error instanceof TypeError && error.message.includes('fetch')) {
        throw new NetworkError(
          'Unable to connect to EE Design Partner API. Please check your network connection.',
          error
        );
      }

      if (error instanceof DOMException && error.name === 'AbortError') {
        throw new NetworkError('Request timed out. The server took too long to respond.');
      }

      throw new NetworkError(
        `Unexpected error: ${error instanceof Error ? error.message : String(error)}`
      );
    }
  }

  // ==========================================================================
  // Project Endpoints
  // ==========================================================================

  async getProjects(filters?: {
    status?: string;
    type?: string;
    limit?: number;
    offset?: number;
  }): Promise<ApiResponse<{ projects: Project[]; total: number }>> {
    const params = new URLSearchParams();
    if (filters?.status) params.set('status', filters.status);
    if (filters?.type) params.set('type', filters.type);
    if (filters?.limit) params.set('limit', String(filters.limit));
    if (filters?.offset) params.set('offset', String(filters.offset));

    const query = params.toString();
    return this.request('GET', `/projects${query ? `?${query}` : ''}`);
  }

  async getProject(projectId: string): Promise<ApiResponse<Project>> {
    return this.request('GET', `/projects/${projectId}`);
  }

  async createProject(input: {
    name: string;
    description?: string;
    type: string;
  }): Promise<ApiResponse<Project>> {
    return this.request('POST', '/projects', input);
  }

  async updateProject(
    projectId: string,
    updates: Partial<Project>
  ): Promise<ApiResponse<Project>> {
    return this.request('PATCH', `/projects/${projectId}`, updates);
  }

  async deleteProject(projectId: string): Promise<ApiResponse<{ deleted: boolean }>> {
    return this.request('DELETE', `/projects/${projectId}`);
  }

  // ==========================================================================
  // File Browser Endpoints
  // ==========================================================================

  async getFileTree(projectId: string, dirPath?: string): Promise<ApiResponse<FileNode>> {
    const params = dirPath ? `?path=${encodeURIComponent(dirPath)}` : '';
    return this.request('GET', `/projects/${projectId}/files${params}`);
  }

  async getFileContent(
    projectId: string,
    filePath: string
  ): Promise<ApiResponse<{ content: string; mimeType: string }>> {
    return this.request(
      'GET',
      `/projects/${projectId}/files/content?path=${encodeURIComponent(filePath)}`
    );
  }

  // ==========================================================================
  // Validation Endpoints
  // ==========================================================================

  async getValidationResults(projectId: string): Promise<ApiResponse<ValidationResult>> {
    return this.request('GET', `/projects/${projectId}/validation`);
  }

  async runValidation(
    projectId: string,
    options?: { domains?: string[]; validators?: string[] }
  ): Promise<ApiResponse<{ validationId: string; status: string }>> {
    return this.request('POST', `/projects/${projectId}/validate/multi-llm`, options);
  }

  async getValidationStatus(
    projectId: string,
    validationId: string
  ): Promise<ApiResponse<ValidationResult>> {
    return this.request('GET', `/projects/${projectId}/validation/${validationId}`);
  }

  // ==========================================================================
  // Command Execution Endpoints
  // ==========================================================================

  async executeCommand(
    command: string,
    projectId?: string
  ): Promise<ApiResponse<CommandExecutionResult>> {
    return this.request('POST', '/commands/execute', { command, projectId });
  }

  async getCommandStatus(commandId: string): Promise<ApiResponse<CommandExecutionResult>> {
    return this.request('GET', `/commands/${commandId}`);
  }

  // ==========================================================================
  // Skills Endpoints
  // ==========================================================================

  async executeSkill(request: SkillExecutionRequest): Promise<ApiResponse<{
    executionId: string;
    status: string;
    result?: unknown;
  }>> {
    return this.request('POST', '/skills/execute', request);
  }

  async searchSkills(query: string, options?: {
    limit?: number;
    phase?: string;
  }): Promise<ApiResponse<{
    results: Array<{ skill: { name: string; description: string }; relevance: number }>;
  }>> {
    return this.request('POST', '/skills/search', { query, ...options });
  }

  async getAvailableSkills(): Promise<ApiResponse<{
    skills: Array<{
      name: string;
      displayName: string;
      description: string;
      phase: string;
    }>;
  }>> {
    return this.request('GET', '/skills');
  }

  // ==========================================================================
  // Simulation Endpoints
  // ==========================================================================

  async runSimulation(
    projectId: string,
    simulationType: string,
    config?: Record<string, unknown>
  ): Promise<ApiResponse<{ simulationId: string; status: string }>> {
    return this.request('POST', `/projects/${projectId}/simulation/${simulationType}`, config);
  }

  async getSimulationStatus(
    projectId: string,
    simulationId: string
  ): Promise<ApiResponse<{
    id: string;
    type: string;
    status: string;
    progress: number;
    results?: unknown;
  }>> {
    return this.request('GET', `/projects/${projectId}/simulations/${simulationId}`);
  }

  // ==========================================================================
  // PCB Layout Endpoints
  // ==========================================================================

  async startPCBLayout(
    projectId: string,
    config?: { agents?: string[]; strategy?: string }
  ): Promise<ApiResponse<{ layoutId: string; status: string }>> {
    return this.request('POST', `/projects/${projectId}/pcb-layout/generate`, config);
  }

  async runMAPOS(
    projectId: string,
    config?: { targetViolations?: number; maxIterations?: number }
  ): Promise<ApiResponse<{ optimizationId: string; status: string }>> {
    return this.request('POST', `/projects/${projectId}/mapos/optimize`, config);
  }

  // ==========================================================================
  // Health Check
  // ==========================================================================

  async checkHealth(): Promise<ApiResponse<{
    status: string;
    service: string;
    version: string;
    timestamp: string;
  }>> {
    try {
      const response = await fetch(`${API_BASE_URL}/health`);
      return response.json();
    } catch {
      throw new NetworkError('Unable to reach EE Design Partner API health endpoint');
    }
  }
}

// Export singleton instance
export const apiClient = new EEDesignApiClient();

// Export class for testing
export { EEDesignApiClient };
