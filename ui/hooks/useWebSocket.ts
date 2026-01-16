/**
 * EE Design Partner - WebSocket Hook
 *
 * Real-time event handling for simulations, layouts, and validation.
 * Integrates with Zustand store for state updates.
 *
 * IMPORTANT: Uses refs for callback functions to prevent infinite render loops.
 * Callbacks passed to this hook may change on every render, but we store them
 * in refs to avoid recreating the socket connection unnecessarily.
 */

import { useEffect, useRef, useCallback, useState } from "react";
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
// Constants
// ============================================================================

const MAX_RECONNECT_ATTEMPTS = 5;
const RECONNECTION_DELAY = 1000;
const RECONNECTION_DELAY_MAX = 5000;
const CONNECTION_TIMEOUT = 20000;

// ============================================================================
// Hook Implementation
// ============================================================================

export function useWebSocket(options: UseWebSocketOptions = {}) {
  const { autoConnect = true } = options;

  // Use state to track connection status reactively
  const [isConnected, setIsConnected] = useState(false);

  // Use refs for socket and connection state to avoid render loops
  const socketRef = useRef<Socket | null>(null);
  const reconnectAttemptsRef = useRef(0);
  const isConnectingRef = useRef(false);

  // CRITICAL: Store callbacks in refs to prevent infinite render loops
  // When caller passes inline functions, they change every render.
  // By storing them in refs, we don't need them in dependency arrays.
  const onConnectedRef = useRef(options.onConnected);
  const onDisconnectedRef = useRef(options.onDisconnected);
  const onErrorRef = useRef(options.onError);

  // Update callback refs when they change (without triggering re-renders)
  useEffect(() => {
    onConnectedRef.current = options.onConnected;
  }, [options.onConnected]);

  useEffect(() => {
    onDisconnectedRef.current = options.onDisconnected;
  }, [options.onDisconnected]);

  useEffect(() => {
    onErrorRef.current = options.onError;
  }, [options.onError]);

  // Get store actions - these are stable references from Zustand
  const activeProjectId = useEEDesignStore((state) => state.activeProjectId);
  const updateSimulation = useEEDesignStore((state) => state.updateSimulation);
  const updateLayoutState = useEEDesignStore((state) => state.updateLayoutState);
  const updateValidationFromWebSocket = useEEDesignStore((state) => state.updateValidationFromWebSocket);
  const updateCommandStatus = useEEDesignStore((state) => state.updateCommandStatus);
  const addTerminalEntry = useEEDesignStore((state) => state.addTerminalEntry);

  // Store activeProjectId in ref for use in socket handlers
  const activeProjectIdRef = useRef(activeProjectId);
  useEffect(() => {
    activeProjectIdRef.current = activeProjectId;
  }, [activeProjectId]);

  // ========================================================================
  // Socket Connection
  // ========================================================================

  const connect = useCallback(() => {
    // Prevent multiple simultaneous connection attempts
    if (socketRef.current?.connected || isConnectingRef.current) {
      return;
    }

    isConnectingRef.current = true;

    const wsUrl = process.env.NEXT_PUBLIC_WS_URL || "http://localhost:9080";
    const wsPath = process.env.NEXT_PUBLIC_WS_PATH || "/ee-design/ws";

    try {
      socketRef.current = io(wsUrl, {
        path: wsPath,
        transports: ["websocket", "polling"],
        reconnection: true,
        reconnectionAttempts: MAX_RECONNECT_ATTEMPTS,
        reconnectionDelay: RECONNECTION_DELAY,
        reconnectionDelayMax: RECONNECTION_DELAY_MAX,
        timeout: CONNECTION_TIMEOUT,
      });

      const socket = socketRef.current;

      // Connection events
      socket.on("connect", () => {
        console.log("[WebSocket] Connected:", socket.id);
        reconnectAttemptsRef.current = 0;
        isConnectingRef.current = false;
        setIsConnected(true);

        // Call the callback via ref (won't cause re-renders)
        onConnectedRef.current?.();

        // Subscribe to active project if set
        const projectId = activeProjectIdRef.current;
        if (projectId) {
          socket.emit("subscribe:project", projectId);
        }
      });

      socket.on("disconnect", (reason) => {
        console.log("[WebSocket] Disconnected:", reason);
        isConnectingRef.current = false;
        setIsConnected(false);

        // Call the callback via ref
        onDisconnectedRef.current?.();
      });

      socket.on("connect_error", (error) => {
        console.error("[WebSocket] Connection error:", error.message);
        reconnectAttemptsRef.current++;
        isConnectingRef.current = false;

        if (reconnectAttemptsRef.current >= MAX_RECONNECT_ATTEMPTS) {
          addTerminalEntry({
            type: "error",
            content: `WebSocket connection failed after ${MAX_RECONNECT_ATTEMPTS} attempts. Real-time updates unavailable.`,
          });
        }

        // Call the callback via ref
        onErrorRef.current?.(error);
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

    } catch (error) {
      console.error("[WebSocket] Failed to create socket:", error);
      isConnectingRef.current = false;
      onErrorRef.current?.(error instanceof Error ? error : new Error(String(error)));
    }
  }, [
    // Only stable references from Zustand - no inline callbacks!
    updateSimulation,
    updateLayoutState,
    updateValidationFromWebSocket,
    updateCommandStatus,
    addTerminalEntry,
  ]);

  const disconnect = useCallback(() => {
    if (socketRef.current) {
      socketRef.current.removeAllListeners();
      socketRef.current.disconnect();
      socketRef.current = null;
      isConnectingRef.current = false;
      setIsConnected(false);
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
  // Lifecycle - Auto Connect
  // ========================================================================

  useEffect(() => {
    if (autoConnect) {
      connect();
    }

    return () => {
      disconnect();
    };
    // Only depend on autoConnect, connect, and disconnect
    // connect/disconnect are stable due to useCallback with stable deps
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
    isConnected,
    connect,
    disconnect,
    subscribeToProject,
    unsubscribeFromProject,
    subscribeToSimulation,
    subscribeToLayout,
  };
}

export default useWebSocket;
