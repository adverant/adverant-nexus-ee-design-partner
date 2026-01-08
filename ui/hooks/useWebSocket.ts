import { useEffect, useRef, useCallback } from "react";
import { io, Socket } from "socket.io-client";

interface WebSocketEvents {
  "simulation:started": (data: { simulationId: string; type: string }) => void;
  "simulation:progress": (data: { simulationId: string; progress: number }) => void;
  "simulation:completed": (data: { simulationId: string; results: unknown }) => void;
  "layout:started": (data: { layoutId: string; agents: string[] }) => void;
  "layout:iteration": (data: { layoutId: string; iteration: number; score: number }) => void;
  "layout:completed": (data: { layoutId: string; finalScore: number }) => void;
  "validation:started": (data: { validationId: string }) => void;
  "validation:completed": (data: { validationId: string; score: number }) => void;
}

interface UseWebSocketOptions {
  projectId?: string | null;
  onEvent?: <K extends keyof WebSocketEvents>(
    event: K,
    handler: WebSocketEvents[K]
  ) => void;
}

export function useWebSocket({ projectId, onEvent }: UseWebSocketOptions = {}) {
  const socketRef = useRef<Socket | null>(null);
  const handlersRef = useRef<Map<string, Set<Function>>>(new Map());

  useEffect(() => {
    const wsUrl = process.env.NEXT_PUBLIC_WS_URL || "http://localhost:9080";

    socketRef.current = io(wsUrl, {
      path: "/ws",
      transports: ["websocket"],
      reconnection: true,
      reconnectionAttempts: 5,
      reconnectionDelay: 1000,
    });

    const socket = socketRef.current;

    socket.on("connect", () => {
      console.log("[WebSocket] Connected");

      // Subscribe to project updates if projectId is provided
      if (projectId) {
        socket.emit("subscribe:project", projectId);
      }
    });

    socket.on("disconnect", () => {
      console.log("[WebSocket] Disconnected");
    });

    socket.on("connect_error", (error) => {
      console.error("[WebSocket] Connection error:", error);
    });

    // Set up event handlers
    const events: (keyof WebSocketEvents)[] = [
      "simulation:started",
      "simulation:progress",
      "simulation:completed",
      "layout:started",
      "layout:iteration",
      "layout:completed",
      "validation:started",
      "validation:completed",
    ];

    events.forEach((event) => {
      socket.on(event, (data: unknown) => {
        const handlers = handlersRef.current.get(event);
        if (handlers) {
          handlers.forEach((handler) => handler(data));
        }
      });
    });

    return () => {
      socket.disconnect();
    };
  }, [projectId]);

  // Subscribe to specific project
  const subscribeToProject = useCallback((id: string) => {
    if (socketRef.current) {
      socketRef.current.emit("subscribe:project", id);
    }
  }, []);

  // Subscribe to simulation updates
  const subscribeToSimulation = useCallback((simulationId: string) => {
    if (socketRef.current) {
      socketRef.current.emit("subscribe:simulation", simulationId);
    }
  }, []);

  // Subscribe to layout updates
  const subscribeToLayout = useCallback((layoutId: string) => {
    if (socketRef.current) {
      socketRef.current.emit("subscribe:layout", layoutId);
    }
  }, []);

  // Add event handler
  const on = useCallback(
    <K extends keyof WebSocketEvents>(event: K, handler: WebSocketEvents[K]) => {
      if (!handlersRef.current.has(event)) {
        handlersRef.current.set(event, new Set());
      }
      handlersRef.current.get(event)!.add(handler as Function);

      return () => {
        handlersRef.current.get(event)?.delete(handler as Function);
      };
    },
    []
  );

  return {
    socket: socketRef.current,
    subscribeToProject,
    subscribeToSimulation,
    subscribeToLayout,
    on,
  };
}