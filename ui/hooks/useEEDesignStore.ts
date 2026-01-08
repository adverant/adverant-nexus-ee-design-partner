import { create } from "zustand";
import { immer } from "zustand/middleware/immer";

export interface Project {
  id: string;
  name: string;
  description?: string;
  phase: string;
  status: "draft" | "in_progress" | "completed";
  completedPhases: string[];
  createdAt: string;
  updatedAt: string;
}

export interface ValidationResult {
  domain: string;
  status: "passed" | "warning" | "failed" | "pending";
  score: number;
  issues: number;
  details?: string;
}

export interface SimulationResult {
  id: string;
  type: string;
  status: "pending" | "running" | "completed" | "failed";
  progress: number;
  results?: unknown;
  error?: string;
}

interface EEDesignState {
  // Projects
  projects: Project[];
  activeProjectId: string | null;

  // Validation
  validationResults: Record<string, ValidationResult[]>;
  overallScore: number;

  // Simulations
  simulations: SimulationResult[];

  // Terminal
  terminalHistory: Array<{ type: "input" | "output"; content: string; timestamp: Date }>;

  // UI State
  activeTab: string;
  isTerminalMaximized: boolean;

  // Actions
  setActiveProject: (id: string | null) => void;
  addProject: (project: Project) => void;
  updateProject: (id: string, updates: Partial<Project>) => void;
  setValidationResults: (projectId: string, results: ValidationResult[]) => void;
  addSimulation: (simulation: SimulationResult) => void;
  updateSimulation: (id: string, updates: Partial<SimulationResult>) => void;
  addTerminalEntry: (type: "input" | "output", content: string) => void;
  setActiveTab: (tab: string) => void;
  setTerminalMaximized: (maximized: boolean) => void;
}

export const useEEDesignStore = create<EEDesignState>()(
  immer((set) => ({
    // Initial State
    projects: [],
    activeProjectId: null,
    validationResults: {},
    overallScore: 0,
    simulations: [],
    terminalHistory: [
      {
        type: "output",
        content: `Welcome to EE Design Partner - Claude Code Terminal
Type /help for available commands or start with natural language requirements.`,
        timestamp: new Date(),
      },
    ],
    activeTab: "schematic",
    isTerminalMaximized: false,

    // Actions
    setActiveProject: (id) =>
      set((state) => {
        state.activeProjectId = id;
      }),

    addProject: (project) =>
      set((state) => {
        state.projects.push(project);
      }),

    updateProject: (id, updates) =>
      set((state) => {
        const index = state.projects.findIndex((p) => p.id === id);
        if (index !== -1) {
          state.projects[index] = { ...state.projects[index], ...updates };
        }
      }),

    setValidationResults: (projectId, results) =>
      set((state) => {
        state.validationResults[projectId] = results;
        // Calculate overall score
        if (results.length > 0) {
          state.overallScore =
            results.reduce((sum, r) => sum + r.score, 0) / results.length;
        }
      }),

    addSimulation: (simulation) =>
      set((state) => {
        state.simulations.push(simulation);
      }),

    updateSimulation: (id, updates) =>
      set((state) => {
        const index = state.simulations.findIndex((s) => s.id === id);
        if (index !== -1) {
          state.simulations[index] = { ...state.simulations[index], ...updates };
        }
      }),

    addTerminalEntry: (type, content) =>
      set((state) => {
        state.terminalHistory.push({
          type,
          content,
          timestamp: new Date(),
        });
      }),

    setActiveTab: (tab) =>
      set((state) => {
        state.activeTab = tab;
      }),

    setTerminalMaximized: (maximized) =>
      set((state) => {
        state.isTerminalMaximized = maximized;
      }),
  }))
);