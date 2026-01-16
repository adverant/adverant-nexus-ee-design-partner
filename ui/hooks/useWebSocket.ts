/**
 * EE Design Partner - WebSocket Hook
 *
 * Real-time event handling for simulations, layouts, and validation.
 * Integrates with Zustand store for state updates.
 */

import { useEffect, useRef, useCallback } from "react";
import { io, Socket } from "socket.io-client";
import { useEEDesignStore } from "./useEEDesignStore";

// ============================================================================
// Types
// ============================================================================

interface SimulationStartedEvent {
  simulationId: string;
  type: string;
  projectId: string;
}

interface SimulationProgressEvent {
  simulationId: string;
  progress: number;
  status: string;
  message?: string;
}

interface SimulationCompletedEvent {
  simulationId: string;
  results: unknown;
  duration: number;
}

interface SimulationFailedEvent {
  simulationId: string;
  error: string;
  code: string;
}

interface LayoutStartedEvent {
  layoutId: string;
  agents: string[];
  projectId: string;
}

interface LayoutIterationEvent {
  layoutId: string;
  iteration: number;
  maxIterations: number;
  score: number;
  improvements: string[];
}

interface LayoutCompletedEvent {
  layoutId: string;
  finalScore: number;
  duration: number;
  violations: number;
}

interface LayoutFailedEvent {
  layoutId: string;
  error: string;
  iteration: number;
}

interface ValidationStartedEvent {
  validationId: string;
  projectId: string;
  domains: string[];
}

interface ValidationProgressEvent {
  validationId: string;
  domain: string;
  status: "running" | "completed" | "failed";
  score?: number;
}

interface ValidationCompletedEvent {
  validationId: string;
  overallScore: number;
  overallStatus: "passed" | "warning" | "failed";
  domains: Array<{
    id: string;
    name: string;
    status: string;
    score: number;
    issues: number;
  }>;
}

interface CommandOutputEvent {
  commandId: string;
  output: string;
  status: "running" | "completed" | "failed";
  error?: string;
}

// ============================================================================
// Hook Options
// ============================================================================

interface UseWebSocketOptions {
  autoConnect?: boolean;
  onConnected?: () => void;
  onDisconnected?: () => void;
  onError?: (error: Error) => void;
}

// ============================================================================
// Hook Implementation
// ============================================================================

export function useWebSocket(options: UseWebSocketOptions = {}) {
  const { autoConnect = true, onConnected, onDisconnected, onError } = options;

  const socketRef = useRef<Socket | null>(null);
  const reconnectAttemptsRef = useRef(0);
  const maxReconnectAttempts = 5;

  // Get store actions
  const {
    activeProjectId,
    updateSimulation,
    updateLayoutState,
    updateValidationFromWebSocket,
    updateCommandStatus,
    addTerminalEntry,
  } = useEEDesignStore();

  // ========================================================================
  // Socket Connection
  // ========================================================================

  const connect = useCallback(() => {
    if (socketRef.current?.connected) {
      return;
    }

    const wsUrl = process.env.NEXT_PUBLIC_WS_URL || "http://localhost:9080";
    const wsPath = process.env.NEXT_PUBLIC_WS_PATH || "/ee-design/ws";

    socketRef.current = io(wsUrl, {
      path: wsPath,
      transports: ["websocket", "polling"],
      reconnection: true,
      reconnectionAttempts: maxReconnectAttempts,
      reconnectionDelay: 1000,
      reconnectionDelayMax: 5000,
      timeout: 20000,
    });

    const socket = socketRef.current;

    // Connection events
    socket.on("connect", () => {
      console.log("[WebSocket] Connected:", socket.id);
      reconnectAttemptsRef.current = 0;
      onConnected?.();

      // Subscribe to active project if set
      if (activeProjectId) {
        socket.emit("subscribe:project", activeProjectId);
      }
    });

    socket.on("disconnect", (reason) => {
      console.log("[WebSocket] Disconnected:", reason);
      onDisconnected?.();
    });

    socket.on("connect_error", (error) => {
      console.error("[WebSocket] Connection error:", error.message);
      reconnectAttemptsRef.current++;

      if (reconnectAttemptsRef.current >= maxReconnectAttempts) {
        addTerminalEntry({
          type: "error",
          content: `WebSocket connection failed after ${maxReconnectAttempts} attempts. Real-time updates unavailable.`,
        });
      }

      onError?.(error);
    });

    // ========================================================================
    // Simulation Events
    // ========================================================================

    socket.on("simulation:started", (data: SimulationStartedEvent) => {
      console.log("[WebSocket] Simulation started:", data.simulationId);
      updateSimulation(data.simulationId, {
        id: data.simulationId,
        type: data.type,
        status: "running",
        progress: 0,
        startedAt: new Date().toISOString(),
      });
      addTerminalEntry({
        type: "system",
        content: `Simulation started: ${data.type} (ID: ${data.simulationId})`,
      });
    });

    socket.on("simulation:progress", (data: SimulationProgressEvent) => {
      updateSimulation(data.simulationId, {
        progress: data.progress,
        status: data.status === "running" ? "running" : "pending",
      });
    });

    socket.on("simulation:completed", (data: SimulationCompletedEvent) => {
      console.log("[WebSocket] Simulation completed:", data.simulationId);
      updateSimulation(data.simulationId, {
        status: "completed",
        progress: 100,
        results: data.results,
        completedAt: new Date().toISOString(),
      });
      addTerminalEntry({
        type: "output",
        content: `Simulation completed (ID: ${data.simulationId}) in ${(data.duration / 1000).toFixed(1)}s`,
      });
    });

    socket.on("simulation:failed", (data: SimulationFailedEvent) => {
      console.error("[WebSocket] Simulation failed:", data.simulationId, data.error);
      updateSimulation(data.simulationId, {
        status: "failed",
        error: data.error,
        completedAt: new Date().toISOString(),
      });
      addTerminalEntry({
        type: "error",
        content: `Simulation failed (ID: ${data.simulationId}): ${data.error}`,
      });
    });

    // ========================================================================
    // Layout Events
    // ========================================================================

    socket.on("layout:started", (data: LayoutStartedEvent) => {
      console.log("[WebSocket] Layout started:", data.layoutId);
      updateLayoutState({
        id: data.layoutId,
        status: "running",
        iteration: 0,
        maxIterations: 100,
        score: 0,
        agents: data.agents,
        startedAt: new Date().toISOString(),
      });
      addTerminalEntry({
        type: "system",
        content: `PCB layout optimization started (ID: ${data.layoutId}) with agents: ${data.agents.join(", ")}`,
      });
    });

    socket.on("layout:iteration", (data: LayoutIterationEvent) => {
      updateLayoutState({
        iteration: data.iteration,
        maxIterations: data.maxIterations,
        score: data.score,
      });
      // Only log every 10th iteration to avoid terminal spam
      if (data.iteration % 10 === 0) {
        addTerminalEntry({
          type: "output",
          content: `Layout iteration ${data.iteration}/${data.maxIterations}: Score ${data.score.toFixed(1)}%`,
        });
      }
    });

    socket.on("layout:completed", (data: LayoutCompletedEvent) => {
      console.log("[WebSocket] Layout completed:", data.layoutId);
      updateLayoutState({
        status: "completed",
        score: data.finalScore,
        completedAt: new Date().toISOString(),
      });
      addTerminalEntry({
        type: "output",
        content: `Layout optimization completed (ID: ${data.layoutId})\n  Final Score: ${data.finalScore.toFixed(1)}%\n  DRC Violations: ${data.violations}\n  Duration: ${(data.duration / 1000).toFixed(1)}s`,
      });
    });

    socket.on("layout:failed", (data: LayoutFailedEvent) => {
      console.error("[WebSocket] Layout failed:", data.layoutId, data.error);
      updateLayoutState({
        status: "failed",
        completedAt: new Date().toISOString(),
      });
      addTerminalEntry({
        type: "error",
        content: `Layout optimization failed at iteration ${data.iteration}: ${data.error}`,
      });
    });

    // ========================================================================
    // Validation Events
    // ========================================================================

    socket.on("validation:started", (data: ValidationStartedEvent) => {
      console.log("[WebSocket] Validation started:", data.validationId);
      updateValidationFromWebSocket({
        projectId: data.projectId,
        overallStatus: "pending",
        domains: data.domains.map((d) => ({
          id: d,
          name: d,
          status: "pending",
          score: 0,
          issues: 0,
        })),
      });
    });

    socket.on("validation:progress", (data: ValidationProgressEvent) => {
      // This would need more complex state management to update individual domains
      console.log("[WebSocket] Validation progress:", data.domain, data.status);
    });

    socket.on("validation:completed", (data: ValidationCompletedEvent) => {
      console.log("[WebSocket] Validation completed:", data.validationId);
      updateValidationFromWebSocket({
        overallScore: data.overallScore,
        overallStatus: data.overallStatus,
        domains: data.domains.map((d) => ({
          id: d.id,
          name: d.name,
          status: d.status as "passed" | "warning" | "failed" | "pending",
          score: d.score,
          issues: d.issues,
        })),
        timestamp: new Date().toISOString(),
      });
      addTerminalEntry({
        type: "output",
        content: `Validation completed: ${data.overallScore.toFixed(1)}% (${data.overallStatus.toUpperCase()})`,
      });
    });

    // ========================================================================
    // Command Events
    // ========================================================================

    socket.on("command:output", (data: CommandOutputEvent) => {
      updateCommandStatus(data.commandId, {
        commandId: data.commandId,
        status: data.status,
        output: data.output,
        error: data.error,
      });
    });

    // Heartbeat
    socket.on("heartbeat:ack", () => {
      // Connection is alive
    });
  }, [
    activeProjectId,
    updateSimulation,
    updateLayoutState,
    updateValidationFromWebSocket,
    updateCommandStatus,
    addTerminalEntry,
    onConnected,
    onDisconnected,
    onError,
  ]);

  const disconnect = useCallback(() => {
    if (socketRef.current) {
      socketRef.current.disconnect();
      socketRef.current = null;
    }
  }, []);

  // ========================================================================
  // Subscription Methods
  // ========================================================================

  const subscribeToProject = useCallback((projectId: string) => {
    if (socketRef.current?.connected) {
      socketRef.current.emit("subscribe:project", projectId);
      console.log("[WebSocket] Subscribed to project:", projectId);
    }
  }, []);

  const unsubscribeFromProject = useCallback((projectId: string) => {
    if (socketRef.current?.connected) {
      socketRef.current.emit("unsubscribe:project", projectId);
      console.log("[WebSocket] Unsubscribed from project:", projectId);
    }
  }, []);

  const subscribeToSimulation = useCallback((simulationId: string) => {
    if (socketRef.current?.connected) {
      socketRef.current.emit("subscribe:simulation", simulationId);
    }
  }, []);

  const subscribeToLayout = useCallback((layoutId: string) => {
    if (socketRef.current?.connected) {
      socketRef.current.emit("subscribe:layout", layoutId);
    }
  }, []);

  // ========================================================================
  // Lifecycle
  // ========================================================================

  useEffect(() => {
    if (autoConnect) {
      connect();
    }

    return () => {
      disconnect();
    };
  }, [autoConnect, connect, disconnect]);

  // Subscribe to project changes
  useEffect(() => {
    if (activeProjectId && socketRef.current?.connected) {
      subscribeToProject(activeProjectId);
    }
  }, [activeProjectId, subscribeToProject]);

  // ========================================================================
  // Return Value
  // ========================================================================

  return {
    socket: socketRef.current,
    isConnected: socketRef.current?.connected ?? false,
    connect,
    disconnect,
    subscribeToProject,
    unsubscribeFromProject,
    subscribeToSimulation,
    subscribeToLayout,
  };
}

export default useWebSocket;
