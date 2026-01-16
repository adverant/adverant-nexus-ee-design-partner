/**
 * EE Design Partner - State Management Store
 *
 * Centralized state management using Zustand with proper API integration.
 * All state modifications go through this store for consistency.
 */

import { create } from "zustand";
import { immer } from "zustand/middleware/immer";
import {
  apiClient,
  ApiError,
  type Project,
  type FileNode,
  type ValidationResult,
  type ValidationDomain,
  type CommandExecutionResult,
} from "@/lib/api-client";

// ============================================================================
// Types
// ============================================================================

export interface TerminalEntry {
  id: string;
  type: "input" | "output" | "error" | "system";
  content: string;
  timestamp: Date;
  commandId?: string;
  status?: "pending" | "completed" | "failed";
}

export interface SimulationState {
  id: string;
  type: string;
  status: "pending" | "running" | "completed" | "failed";
  progress: number;
  results?: unknown;
  error?: string;
  startedAt?: string;
  completedAt?: string;
}

export interface LayoutState {
  id: string;
  status: "pending" | "running" | "completed" | "failed";
  iteration: number;
  maxIterations: number;
  score: number;
  agents: string[];
  startedAt?: string;
  completedAt?: string;
}

interface LoadingStates {
  projects: boolean;
  fileTree: boolean;
  validation: boolean;
  command: boolean;
  simulation: boolean;
  layout: boolean;
}

interface ErrorStates {
  projects: string | null;
  fileTree: string | null;
  validation: string | null;
  command: string | null;
  simulation: string | null;
  layout: string | null;
}

// ============================================================================
// Store Interface
// ============================================================================

// Use Record instead of Map for React compatibility (Map reference stays same on mutation)
type SimulationsRecord = Record<string, SimulationState>;
// Use Record instead of Set for React compatibility (Set reference stays same on mutation)
type ExpandedFoldersRecord = Record<string, boolean>;

interface EEDesignState {
  // Connection
  isConnected: boolean;
  apiError: string | null;

  // Projects
  projects: Project[];
  activeProjectId: string | null;
  activeProject: Project | null;

  // File Browser
  fileTree: FileNode | null;
  selectedFile: string | null;

  // Validation
  validationResult: ValidationResult | null;
  isValidationRunning: boolean;

  // Terminal
  terminalHistory: TerminalEntry[];
  currentCommandId: string | null;

  // Simulations - using Record for React compatibility
  simulations: SimulationsRecord;

  // Layout
  layoutState: LayoutState | null;

  // UI State
  activeTab: string;
  isTerminalMaximized: boolean;
  // Using Record<string, boolean> instead of Set<string> for React compatibility
  expandedFolders: ExpandedFoldersRecord;

  // Loading & Error States
  loading: LoadingStates;
  errors: ErrorStates;

  // Actions - Connection
  checkConnection: () => Promise<boolean>;
  clearApiError: () => void;

  // Actions - Projects
  fetchProjects: () => Promise<void>;
  fetchProject: (projectId: string) => Promise<void>;
  setActiveProject: (projectId: string | null) => void;
  createProject: (name: string, description: string, type: string) => Promise<Project | null>;

  // Actions - File Browser
  fetchFileTree: (projectId: string, path?: string) => Promise<void>;
  setSelectedFile: (path: string | null) => void;
  toggleFolder: (path: string) => void;

  // Actions - Validation
  fetchValidationResults: (projectId: string) => Promise<void>;
  runValidation: (projectId: string, domains?: string[]) => Promise<void>;
  updateValidationFromWebSocket: (data: Partial<ValidationResult>) => void;

  // Actions - Terminal
  executeCommand: (command: string) => Promise<void>;
  addTerminalEntry: (entry: Omit<TerminalEntry, "id" | "timestamp">) => void;
  updateCommandStatus: (commandId: string, result: CommandExecutionResult) => void;
  clearTerminal: () => void;

  // Actions - Simulations
  updateSimulation: (simulationId: string, update: Partial<SimulationState>) => void;
  removeSimulation: (simulationId: string) => void;

  // Actions - Layout
  updateLayoutState: (update: Partial<LayoutState>) => void;
  clearLayoutState: () => void;

  // Actions - UI
  setActiveTab: (tab: string) => void;
  setTerminalMaximized: (maximized: boolean) => void;

  // Actions - Cleanup
  reset: () => void;
}

// ============================================================================
// Initial State
// ============================================================================

const initialLoadingState: LoadingStates = {
  projects: false,
  fileTree: false,
  validation: false,
  command: false,
  simulation: false,
  layout: false,
};

const initialErrorState: ErrorStates = {
  projects: null,
  fileTree: null,
  validation: null,
  command: null,
  simulation: null,
  layout: null,
};

// ============================================================================
// Store Implementation
// ============================================================================

export const useEEDesignStore = create<EEDesignState>()(
  immer((set, get) => ({
    // Initial State
    isConnected: false,
    apiError: null,
    projects: [],
    activeProjectId: null,
    activeProject: null,
    fileTree: null,
    selectedFile: null,
    validationResult: null,
    isValidationRunning: false,
    terminalHistory: [
      {
        id: "welcome",
        type: "system",
        content: `Welcome to EE Design Partner - Claude Code Terminal
Type /help for available commands or describe your requirements in natural language.

Connected to backend API at ${process.env.NEXT_PUBLIC_API_URL || "http://localhost:9080"}`,
        timestamp: new Date(),
      },
    ],
    currentCommandId: null,
    simulations: {},
    layoutState: null,
    activeTab: "schematic",
    isTerminalMaximized: false,
    expandedFolders: {},
    loading: initialLoadingState,
    errors: initialErrorState,

    // ========================================================================
    // Connection Actions
    // ========================================================================

    checkConnection: async () => {
      try {
        const response = await apiClient.checkHealth();
        if (response.data?.status === "healthy") {
          set((state) => {
            state.isConnected = true;
            state.apiError = null;
          });
          return true;
        }
        set((state) => {
          state.isConnected = false;
          state.apiError = "API health check failed";
        });
        return false;
      } catch (error) {
        const message = error instanceof ApiError ? error.message : "Connection failed";
        set((state) => {
          state.isConnected = false;
          state.apiError = message;
        });
        return false;
      }
    },

    clearApiError: () => {
      set((state) => {
        state.apiError = null;
      });
    },

    // ========================================================================
    // Project Actions
    // ========================================================================

    fetchProjects: async () => {
      set((state) => {
        state.loading.projects = true;
        state.errors.projects = null;
      });

      try {
        const response = await apiClient.getProjects();
        if (response.success && response.data) {
          set((state) => {
            state.projects = response.data!.projects;
            state.loading.projects = false;
          });
        }
      } catch (error) {
        const message = error instanceof ApiError ? error.message : "Failed to fetch projects";
        set((state) => {
          state.errors.projects = message;
          state.loading.projects = false;
        });
      }
    },

    fetchProject: async (projectId: string) => {
      try {
        const response = await apiClient.getProject(projectId);
        if (response.success && response.data) {
          set((state) => {
            state.activeProject = response.data!;
            state.activeProjectId = projectId;
          });
        }
      } catch (error) {
        const message = error instanceof ApiError ? error.message : "Failed to fetch project";
        set((state) => {
          state.errors.projects = message;
        });
      }
    },

    setActiveProject: (projectId: string | null) => {
      set((state) => {
        state.activeProjectId = projectId;
        state.activeProject = projectId
          ? state.projects.find((p) => p.id === projectId) || null
          : null;
        // Reset related state when project changes
        state.fileTree = null;
        state.validationResult = null;
        state.selectedFile = null;
      });

      // Fetch related data if project is selected
      if (projectId) {
        get().fetchFileTree(projectId);
        get().fetchValidationResults(projectId);
      }
    },

    createProject: async (name: string, description: string, type: string) => {
      try {
        const response = await apiClient.createProject({ name, description, type });
        if (response.success && response.data) {
          set((state) => {
            state.projects.push(response.data!);
          });
          return response.data;
        }
        return null;
      } catch (error) {
        const message = error instanceof ApiError ? error.message : "Failed to create project";
        set((state) => {
          state.errors.projects = message;
        });
        return null;
      }
    },

    // ========================================================================
    // File Browser Actions
    // ========================================================================

    fetchFileTree: async (projectId: string, path?: string) => {
      set((state) => {
        state.loading.fileTree = true;
        state.errors.fileTree = null;
      });

      try {
        const response = await apiClient.getFileTree(projectId, path);
        if (response.success && response.data) {
          set((state) => {
            state.fileTree = response.data!;
            state.loading.fileTree = false;
          });
        }
      } catch (error) {
        const message = error instanceof ApiError ? error.message : "Failed to fetch file tree";
        set((state) => {
          state.errors.fileTree = message;
          state.loading.fileTree = false;
        });
      }
    },

    setSelectedFile: (path: string | null) => {
      set((state) => {
        state.selectedFile = path;
      });
    },

    toggleFolder: (path: string) => {
      set((state) => {
        if (state.expandedFolders[path]) {
          delete state.expandedFolders[path];
        } else {
          state.expandedFolders[path] = true;
        }
      });
    },

    // ========================================================================
    // Validation Actions
    // ========================================================================

    fetchValidationResults: async (projectId: string) => {
      set((state) => {
        state.loading.validation = true;
        state.errors.validation = null;
      });

      try {
        const response = await apiClient.getValidationResults(projectId);
        if (response.success && response.data) {
          set((state) => {
            state.validationResult = response.data!;
            state.loading.validation = false;
          });
        }
      } catch (error) {
        const message = error instanceof ApiError ? error.message : "Failed to fetch validation";
        set((state) => {
          state.errors.validation = message;
          state.loading.validation = false;
        });
      }
    },

    runValidation: async (projectId: string, domains?: string[]) => {
      set((state) => {
        state.isValidationRunning = true;
        state.errors.validation = null;
      });

      try {
        const response = await apiClient.runValidation(projectId, { domains });
        if (response.success) {
          get().addTerminalEntry({
            type: "system",
            content: `Validation started (ID: ${response.data?.validationId})`,
          });
        }
      } catch (error) {
        const message = error instanceof ApiError ? error.message : "Failed to start validation";
        set((state) => {
          state.errors.validation = message;
          state.isValidationRunning = false;
        });
        get().addTerminalEntry({
          type: "error",
          content: `Validation failed: ${message}`,
        });
      }
    },

    updateValidationFromWebSocket: (data: Partial<ValidationResult>) => {
      set((state) => {
        if (state.validationResult) {
          Object.assign(state.validationResult, data);
        } else {
          state.validationResult = data as ValidationResult;
        }
        if (data.overallStatus && data.overallStatus !== "pending") {
          state.isValidationRunning = false;
        }
      });
    },

    // ========================================================================
    // Terminal Actions
    // ========================================================================

    executeCommand: async (command: string) => {
      const trimmedCommand = command.trim();
      if (!trimmedCommand) return;

      const inputId = `cmd-${Date.now()}`;

      // Add input to history
      set((state) => {
        state.terminalHistory.push({
          id: inputId,
          type: "input",
          content: trimmedCommand,
          timestamp: new Date(),
        });
        state.loading.command = true;
        state.errors.command = null;
      });

      // Handle local commands
      if (trimmedCommand === "/help") {
        get().addTerminalEntry({
          type: "output",
          content: `Available Commands:

Phase 1 - Ideation:
  /research-paper, /patent-search, /market-analysis, /requirements-gen

Phase 2 - Architecture:
  /ee-architecture, /component-select, /bom-optimize, /power-budget

Phase 3 - Schematic:
  /schematic-gen, /schematic-review, /netlist-gen

Phase 4 - Simulation:
  /simulate-spice, /simulate-thermal, /simulate-si, /simulate-rf, /simulate-emc

Phase 5 - PCB Layout:
  /pcb-layout, /pcb-review, /stackup-design, /via-optimize, /mapos

Phase 6 - Manufacturing:
  /gerber-gen, /dfm-check, /vendor-quote, /panelize

Phase 7 - Firmware:
  /firmware-gen, /hal-gen, /driver-gen, /rtos-config

Phase 8 - Testing:
  /test-gen, /hil-setup, /test-procedure

Phase 9 - Production:
  /manufacture, /assembly-guide, /quality-check

Phase 10 - Field Support:
  /debug-assist, /service-manual, /firmware-update

Type any command with --help for detailed usage.`,
        });
        set((state) => {
          state.loading.command = false;
        });
        return;
      }

      if (trimmedCommand === "/clear") {
        get().clearTerminal();
        return;
      }

      // Execute command via API
      try {
        const projectId = get().activeProjectId;
        const response = await apiClient.executeCommand(trimmedCommand, projectId || undefined);

        if (response.success && response.data) {
          const result = response.data;
          set((state) => {
            state.currentCommandId = result.commandId;
          });

          if (result.status === "completed" && result.output) {
            get().addTerminalEntry({
              type: "output",
              content: result.output,
              commandId: result.commandId,
              status: "completed",
            });
          } else if (result.status === "failed" && result.error) {
            get().addTerminalEntry({
              type: "error",
              content: `Error: ${result.error}`,
              commandId: result.commandId,
              status: "failed",
            });
          } else {
            // Command is queued/running, will receive updates via WebSocket
            get().addTerminalEntry({
              type: "system",
              content: `Command queued (ID: ${result.commandId}). Waiting for execution...`,
              commandId: result.commandId,
              status: "pending",
            });
          }
        }
      } catch (error) {
        const message = error instanceof ApiError ? error.message : "Command execution failed";
        get().addTerminalEntry({
          type: "error",
          content: `Error: ${message}`,
        });
        set((state) => {
          state.errors.command = message;
        });
      } finally {
        set((state) => {
          state.loading.command = false;
        });
      }
    },

    addTerminalEntry: (entry: Omit<TerminalEntry, "id" | "timestamp">) => {
      set((state) => {
        state.terminalHistory.push({
          ...entry,
          id: `entry-${Date.now()}-${Math.random().toString(36).slice(2, 9)}`,
          timestamp: new Date(),
        });
      });
    },

    updateCommandStatus: (commandId: string, result: CommandExecutionResult) => {
      set((state) => {
        const index = state.terminalHistory.findIndex((e) => e.commandId === commandId);
        if (index !== -1) {
          state.terminalHistory[index].status =
            result.status === "completed" || result.status === "failed"
              ? result.status
              : "pending";
        }

        if (result.output) {
          state.terminalHistory.push({
            id: `output-${Date.now()}`,
            type: "output",
            content: result.output,
            timestamp: new Date(),
            commandId,
            status: "completed",
          });
        }

        if (result.error) {
          state.terminalHistory.push({
            id: `error-${Date.now()}`,
            type: "error",
            content: result.error,
            timestamp: new Date(),
            commandId,
            status: "failed",
          });
        }
      });
    },

    clearTerminal: () => {
      set((state) => {
        state.terminalHistory = [
          {
            id: "cleared",
            type: "system",
            content: "Terminal cleared. Type /help for available commands.",
            timestamp: new Date(),
          },
        ];
        state.loading.command = false;
      });
    },

    // ========================================================================
    // Simulation Actions
    // ========================================================================

    updateSimulation: (simulationId: string, update: Partial<SimulationState>) => {
      set((state) => {
        const existing = state.simulations[simulationId];
        if (existing) {
          state.simulations[simulationId] = { ...existing, ...update };
        } else {
          state.simulations[simulationId] = {
            id: simulationId,
            type: update.type || "unknown",
            status: update.status || "pending",
            progress: update.progress || 0,
            ...update,
          };
        }
      });
    },

    removeSimulation: (simulationId: string) => {
      set((state) => {
        delete state.simulations[simulationId];
      });
    },

    // ========================================================================
    // Layout Actions
    // ========================================================================

    updateLayoutState: (update: Partial<LayoutState>) => {
      set((state) => {
        if (state.layoutState) {
          Object.assign(state.layoutState, update);
        } else {
          state.layoutState = {
            id: update.id || "unknown",
            status: update.status || "pending",
            iteration: update.iteration || 0,
            maxIterations: update.maxIterations || 100,
            score: update.score || 0,
            agents: update.agents || [],
            ...update,
          };
        }
      });
    },

    clearLayoutState: () => {
      set((state) => {
        state.layoutState = null;
      });
    },

    // ========================================================================
    // UI Actions
    // ========================================================================

    setActiveTab: (tab: string) => {
      set((state) => {
        state.activeTab = tab;
      });
    },

    setTerminalMaximized: (maximized: boolean) => {
      set((state) => {
        state.isTerminalMaximized = maximized;
      });
    },

    // ========================================================================
    // Cleanup
    // ========================================================================

    reset: () => {
      set((state) => {
        state.projects = [];
        state.activeProjectId = null;
        state.activeProject = null;
        state.fileTree = null;
        state.selectedFile = null;
        state.validationResult = null;
        state.isValidationRunning = false;
        state.simulations = {};
        state.layoutState = null;
        state.loading = initialLoadingState;
        state.errors = initialErrorState;
      });
    },
  }))
);

// Re-export types for convenience
export type {
  Project,
  FileNode,
  ValidationResult,
  ValidationDomain,
} from "@/lib/api-client";
