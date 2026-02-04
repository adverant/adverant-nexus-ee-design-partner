/**
 * HIL (Hardware-in-the-Loop) Testing - WebSocket Hook
 *
 * Real-time event handling for HIL instruments, test runs, measurements,
 * and waveform streaming. Connects to the /hil namespace.
 *
 * IMPORTANT: Uses refs for callback functions to prevent infinite render loops.
 */

import { useEffect, useRef, useCallback, useState } from "react";
import { io, Socket } from "socket.io-client";
import {
  useHILStore,
  type HILInstrument,
  type HILTestRun,
  type HILMeasurement,
  type HILCapturedData,
  type HILTestRunSummary,
  type HILInstrumentStatus,
  type HILTestResult,
} from "./useHILStore";

// ============================================================================
// Event Types - Matching backend websocket-hil.ts
// ============================================================================

interface InstrumentDiscoveredEvent {
  instrument: HILInstrument;
  operationId?: string;
}

interface InstrumentConnectedEvent {
  instrumentId: string;
  status: HILInstrumentStatus;
  info?: {
    manufacturer: string;
    model: string;
    firmwareVersion: string;
  };
}

interface InstrumentDisconnectedEvent {
  instrumentId: string;
  reason?: string;
}

interface InstrumentStatusChangedEvent {
  instrumentId: string;
  status: HILInstrumentStatus;
  error?: string;
}

interface TestRunStartedEvent {
  testRunId: string;
  sequenceId: string;
  sequenceName: string;
  projectId: string;
  totalSteps: number;
  testRun: HILTestRun;
}

interface TestStepStartedEvent {
  testRunId: string;
  stepIndex: number;
  stepId: string;
  stepName: string;
  stepType: string;
  totalSteps: number;
}

interface TestStepCompletedEvent {
  testRunId: string;
  stepIndex: number;
  stepId: string;
  success: boolean;
  measurements?: HILMeasurement[];
  duration_ms: number;
}

interface TestMeasurementEvent {
  testRunId: string;
  measurement: HILMeasurement;
}

interface TestRunProgressEvent {
  testRunId: string;
  progress: number;
  currentStep: string;
  stepIndex: number;
  totalSteps: number;
  message?: string;
  phase?: string;
}

interface TestRunCompletedEvent {
  testRunId: string;
  result: HILTestResult;
  summary: HILTestRunSummary;
  duration_ms: number;
}

interface TestRunFailedEvent {
  testRunId: string;
  error: string;
  errorCode?: string;
  stepId?: string;
  details?: Record<string, unknown>;
}

interface TestRunAbortedEvent {
  testRunId: string;
  reason?: string;
  abortedBy?: string;
}

interface WaveformChunkEvent {
  testRunId: string;
  captureId: string;
  channels: string[];
  startIndex: number;
  sampleCount: number;
  data: number[][];
  isLast?: boolean;
}

interface LiveMeasurementEvent {
  testRunId: string;
  instrumentId: string;
  type: string;
  channel?: string;
  value: number;
  unit: string;
  passed?: boolean;
  timestampOffset?: number;
}

interface CaptureStartedEvent {
  testRunId: string;
  captureId: string;
  instrumentId: string;
  captureType: string;
  channelConfig: Array<{
    name: string;
    label?: string;
    scale: number;
    offset: number;
    unit: string;
  }>;
  sampleRateHz: number;
}

interface CaptureCompleteEvent {
  testRunId: string;
  capture: HILCapturedData;
}

interface MotorStateChangedEvent {
  testRunId: string;
  instrumentId: string;
  state: "stopped" | "running" | "starting" | "stopping" | "faulted";
  speed?: number;
  torque?: number;
  direction?: "forward" | "reverse";
}

interface PowerStateChangedEvent {
  testRunId: string;
  instrumentId: string;
  channel: number;
  enabled: boolean;
  voltage: number;
  current: number;
  power: number;
}

interface FaultDetectedEvent {
  testRunId: string;
  instrumentId?: string;
  faultType: string;
  severity: "warning" | "error" | "critical";
  message: string;
  details?: Record<string, unknown>;
}

interface EmergencyStopEvent {
  testRunId: string;
  triggeredBy: string;
  reason: string;
}

// ============================================================================
// Hook Options
// ============================================================================

interface UseHILWebSocketOptions {
  autoConnect?: boolean;
  projectId?: string;
  onConnected?: () => void;
  onDisconnected?: () => void;
  onError?: (error: Error) => void;
  onEmergencyStop?: (event: EmergencyStopEvent) => void;
  onFaultDetected?: (event: FaultDetectedEvent) => void;
}

// ============================================================================
// Constants
// ============================================================================

const MAX_RECONNECT_ATTEMPTS = 5;
const RECONNECTION_DELAY = 1000;
const RECONNECTION_DELAY_MAX = 5000;
const CONNECTION_TIMEOUT = 20000;
const HEARTBEAT_INTERVAL = 30000;

// ============================================================================
// Hook Implementation
// ============================================================================

export function useHILWebSocket(options: UseHILWebSocketOptions = {}) {
  const { autoConnect = true, projectId } = options;

  // Use state to track connection status reactively
  const [isConnected, setIsConnected] = useState(false);

  // Use refs for socket and connection state to avoid render loops
  const socketRef = useRef<Socket | null>(null);
  const reconnectAttemptsRef = useRef(0);
  const isConnectingRef = useRef(false);
  const heartbeatIntervalRef = useRef<NodeJS.Timeout | null>(null);
  const mountedRef = useRef(true);

  // CRITICAL: Store callbacks in refs to prevent infinite render loops
  const onConnectedRef = useRef(options.onConnected);
  const onDisconnectedRef = useRef(options.onDisconnected);
  const onErrorRef = useRef(options.onError);
  const onEmergencyStopRef = useRef(options.onEmergencyStop);
  const onFaultDetectedRef = useRef(options.onFaultDetected);

  // Update callback refs when they change
  useEffect(() => {
    onConnectedRef.current = options.onConnected;
  }, [options.onConnected]);

  useEffect(() => {
    onDisconnectedRef.current = options.onDisconnected;
  }, [options.onDisconnected]);

  useEffect(() => {
    onErrorRef.current = options.onError;
  }, [options.onError]);

  useEffect(() => {
    onEmergencyStopRef.current = options.onEmergencyStop;
  }, [options.onEmergencyStop]);

  useEffect(() => {
    onFaultDetectedRef.current = options.onFaultDetected;
  }, [options.onFaultDetected]);

  // Get store actions - these are stable references from Zustand
  const setWsConnected = useHILStore((state) => state.setWsConnected);
  const addWsSubscription = useHILStore((state) => state.addWsSubscription);
  const removeWsSubscription = useHILStore((state) => state.removeWsSubscription);

  // Instrument actions
  const addInstrument = useHILStore((state) => state.addInstrument);
  const updateInstrumentStatus = useHILStore((state) => state.updateInstrumentStatus);
  const setDiscoveryInProgress = useHILStore((state) => state.setDiscoveryInProgress);

  // Test run actions
  const addTestRun = useHILStore((state) => state.addTestRun);
  const updateTestRun = useHILStore((state) => state.updateTestRun);
  const setActiveTestRun = useHILStore((state) => state.setActiveTestRun);
  const updateTestRunProgress = useHILStore((state) => state.updateTestRunProgress);
  const completeTestRun = useHILStore((state) => state.completeTestRun);
  const failTestRun = useHILStore((state) => state.failTestRun);

  // Measurement actions
  const addMeasurement = useHILStore((state) => state.addMeasurement);
  const addLiveMeasurement = useHILStore((state) => state.addLiveMeasurement);
  const setMeasurementSummary = useHILStore((state) => state.setMeasurementSummary);

  // Capture actions
  const addCapture = useHILStore((state) => state.addCapture);
  const setWaveformData = useHILStore((state) => state.setWaveformData);
  const appendWaveformChunk = useHILStore((state) => state.appendWaveformChunk);

  // Store projectId in ref for use in socket handlers
  const projectIdRef = useRef(projectId);
  useEffect(() => {
    projectIdRef.current = projectId;
  }, [projectId]);

  // ========================================================================
  // Heartbeat
  // ========================================================================

  const startHeartbeat = useCallback(() => {
    if (heartbeatIntervalRef.current) {
      clearInterval(heartbeatIntervalRef.current);
    }
    heartbeatIntervalRef.current = setInterval(() => {
      if (socketRef.current?.connected) {
        socketRef.current.emit("heartbeat");
      }
    }, HEARTBEAT_INTERVAL);
  }, []);

  const stopHeartbeat = useCallback(() => {
    if (heartbeatIntervalRef.current) {
      clearInterval(heartbeatIntervalRef.current);
      heartbeatIntervalRef.current = null;
    }
  }, []);

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
      // Connect to the /hil namespace
      socketRef.current = io(`${wsUrl}/hil`, {
        path: wsPath,
        transports: ["polling", "websocket"], // Start with polling for Istio stability
        reconnection: true,
        reconnectionAttempts: MAX_RECONNECT_ATTEMPTS,
        reconnectionDelay: RECONNECTION_DELAY,
        reconnectionDelayMax: RECONNECTION_DELAY_MAX,
        timeout: CONNECTION_TIMEOUT,
        forceNew: true,
      });

      const socket = socketRef.current;

      // Connection events
      socket.on("connect", () => {
        console.log("[HIL WebSocket] Connected:", socket.id);
        reconnectAttemptsRef.current = 0;
        isConnectingRef.current = false;

        if (mountedRef.current) {
          setIsConnected(true);
          setWsConnected(true);
        }

        startHeartbeat();
        onConnectedRef.current?.();

        // Subscribe to project if set
        const pid = projectIdRef.current;
        if (pid) {
          socket.emit("subscribe:project", pid);
        }
      });

      socket.on("disconnect", (reason) => {
        console.log("[HIL WebSocket] Disconnected:", reason);
        isConnectingRef.current = false;

        if (mountedRef.current) {
          setIsConnected(false);
          setWsConnected(false);
        }

        stopHeartbeat();
        onDisconnectedRef.current?.();
      });

      socket.on("connect_error", (error) => {
        console.error("[HIL WebSocket] Connection error:", error.message);
        reconnectAttemptsRef.current++;
        isConnectingRef.current = false;
        onErrorRef.current?.(error);
      });

      // Heartbeat acknowledgment
      socket.on("heartbeat:ack", () => {
        // Connection is alive
      });

      // ========================================================================
      // Instrument Events
      // ========================================================================

      socket.on("instrument:discovered", (data: InstrumentDiscoveredEvent) => {
        console.log("[HIL WebSocket] Instrument discovered:", data.instrument.name);
        if (mountedRef.current) {
          addInstrument(data.instrument);
        }
      });

      socket.on("instrument:connected", (data: InstrumentConnectedEvent) => {
        console.log("[HIL WebSocket] Instrument connected:", data.instrumentId);
        if (mountedRef.current) {
          updateInstrumentStatus(data.instrumentId, data.status);
        }
      });

      socket.on("instrument:disconnected", (data: InstrumentDisconnectedEvent) => {
        console.log("[HIL WebSocket] Instrument disconnected:", data.instrumentId);
        if (mountedRef.current) {
          updateInstrumentStatus(data.instrumentId, "disconnected", data.reason);
        }
      });

      socket.on("instrument:status_changed", (data: InstrumentStatusChangedEvent) => {
        if (mountedRef.current) {
          updateInstrumentStatus(data.instrumentId, data.status, data.error);
        }
      });

      socket.on("instrument:error", (data: { instrumentId: string; error: string }) => {
        console.error("[HIL WebSocket] Instrument error:", data.instrumentId, data.error);
        if (mountedRef.current) {
          updateInstrumentStatus(data.instrumentId, "error", data.error);
        }
      });

      // Discovery complete
      socket.on("discovery:complete", (data: { count: number }) => {
        console.log("[HIL WebSocket] Discovery complete:", data.count, "instruments");
        if (mountedRef.current) {
          setDiscoveryInProgress(false);
        }
      });

      // ========================================================================
      // Test Run Events
      // ========================================================================

      socket.on("test_run:started", (data: TestRunStartedEvent) => {
        console.log("[HIL WebSocket] Test run started:", data.testRunId);
        if (mountedRef.current) {
          addTestRun(data.testRun);
          setActiveTestRun(data.testRun);
        }
      });

      socket.on("test_run:queued", (data: { testRunId: string; testRun: HILTestRun }) => {
        console.log("[HIL WebSocket] Test run queued:", data.testRunId);
        if (mountedRef.current) {
          addTestRun(data.testRun);
        }
      });

      socket.on("test_step:started", (data: TestStepStartedEvent) => {
        console.log("[HIL WebSocket] Test step started:", data.stepName);
        if (mountedRef.current) {
          updateTestRun(data.testRunId, {
            currentStep: data.stepName,
            currentStepIndex: data.stepIndex,
            totalSteps: data.totalSteps,
            status: "running",
          });
        }
      });

      socket.on("test_step:completed", (data: TestStepCompletedEvent) => {
        console.log("[HIL WebSocket] Test step completed:", data.stepId, data.success);
        if (mountedRef.current && data.measurements) {
          for (const measurement of data.measurements) {
            addMeasurement(measurement);
          }
        }
      });

      socket.on("test_run:measurement", (data: TestMeasurementEvent) => {
        if (mountedRef.current) {
          addMeasurement(data.measurement);
        }
      });

      socket.on("test_run:progress", (data: TestRunProgressEvent) => {
        if (mountedRef.current) {
          updateTestRunProgress(
            data.testRunId,
            data.progress,
            data.currentStep,
            data.stepIndex
          );
        }
      });

      socket.on("test_run:completed", (data: TestRunCompletedEvent) => {
        console.log("[HIL WebSocket] Test run completed:", data.testRunId, data.result);
        if (mountedRef.current) {
          completeTestRun(data.testRunId, data.result, data.summary);
          setMeasurementSummary(data.summary);
        }
      });

      socket.on("test_run:failed", (data: TestRunFailedEvent) => {
        console.error("[HIL WebSocket] Test run failed:", data.testRunId, data.error);
        if (mountedRef.current) {
          failTestRun(data.testRunId, data.error, data.details);
        }
      });

      socket.on("test_run:aborted", (data: TestRunAbortedEvent) => {
        console.log("[HIL WebSocket] Test run aborted:", data.testRunId);
        if (mountedRef.current) {
          updateTestRun(data.testRunId, {
            status: "aborted",
            result: "inconclusive",
            completedAt: new Date().toISOString(),
          });
        }
      });

      // ========================================================================
      // Data Streaming Events
      // ========================================================================

      socket.on("waveform:chunk", (data: WaveformChunkEvent) => {
        if (mountedRef.current) {
          appendWaveformChunk(data.captureId, data.channels, data.startIndex, data.data);
        }
      });

      socket.on("live_measurement", (data: LiveMeasurementEvent) => {
        if (mountedRef.current) {
          addLiveMeasurement({
            type: data.type as any,
            channel: data.channel,
            value: data.value,
            unit: data.unit,
            passed: data.passed,
            timestamp: new Date(),
          });
        }
      });

      socket.on("capture:started", (data: CaptureStartedEvent) => {
        console.log("[HIL WebSocket] Capture started:", data.captureId);
        if (mountedRef.current) {
          setWaveformData({
            captureId: data.captureId,
            channels: data.channelConfig.map((c) => c.name),
            sampleRate: data.sampleRateHz,
            totalSamples: 0,
            data: [],
            timebase: { startTime: 0, endTime: 0, unit: "s" },
            channelConfig: data.channelConfig.map((c) => ({
              name: c.name,
              label: c.label,
              scale: c.scale,
              offset: c.offset,
              unit: c.unit,
            })),
          });
        }
      });

      socket.on("capture:complete", (data: CaptureCompleteEvent) => {
        console.log("[HIL WebSocket] Capture complete:", data.capture.id);
        if (mountedRef.current) {
          addCapture(data.capture);
        }
      });

      // ========================================================================
      // Control & Safety Events
      // ========================================================================

      socket.on("motor:state_changed", (data: MotorStateChangedEvent) => {
        console.log("[HIL WebSocket] Motor state changed:", data.state);
        // Could emit to a motor state store if needed
      });

      socket.on("power:state_changed", (data: PowerStateChangedEvent) => {
        console.log("[HIL WebSocket] Power state changed:", data.channel, data.enabled);
        // Could emit to a power state store if needed
      });

      socket.on("fault:detected", (data: FaultDetectedEvent) => {
        console.warn("[HIL WebSocket] Fault detected:", data.faultType, data.message);
        onFaultDetectedRef.current?.(data);
      });

      socket.on("emergency_stop", (data: EmergencyStopEvent) => {
        console.error("[HIL WebSocket] EMERGENCY STOP:", data.reason);
        onEmergencyStopRef.current?.(data);
        if (mountedRef.current) {
          updateTestRun(data.testRunId, {
            status: "aborted",
            result: "fail",
            errorMessage: `Emergency stop: ${data.reason}`,
            completedAt: new Date().toISOString(),
          });
        }
      });

      // ========================================================================
      // Operation Events
      // ========================================================================

      socket.on("operation:subscribed", (data: { operationId: string }) => {
        if (mountedRef.current) {
          addWsSubscription(data.operationId);
        }
      });

      socket.on("operation:unsubscribed", (data: { operationId: string }) => {
        if (mountedRef.current) {
          removeWsSubscription(data.operationId);
        }
      });

    } catch (error) {
      console.error("[HIL WebSocket] Failed to create socket:", error);
      isConnectingRef.current = false;
      onErrorRef.current?.(error instanceof Error ? error : new Error(String(error)));
    }
  }, [
    // Only stable references from Zustand
    setWsConnected,
    addWsSubscription,
    removeWsSubscription,
    addInstrument,
    updateInstrumentStatus,
    setDiscoveryInProgress,
    addTestRun,
    updateTestRun,
    setActiveTestRun,
    updateTestRunProgress,
    completeTestRun,
    failTestRun,
    addMeasurement,
    addLiveMeasurement,
    setMeasurementSummary,
    addCapture,
    setWaveformData,
    appendWaveformChunk,
    startHeartbeat,
    stopHeartbeat,
  ]);

  const disconnect = useCallback(() => {
    stopHeartbeat();
    if (socketRef.current) {
      socketRef.current.removeAllListeners();
      socketRef.current.disconnect();
      socketRef.current = null;
      isConnectingRef.current = false;
      setIsConnected(false);
      setWsConnected(false);
    }
  }, [stopHeartbeat, setWsConnected]);

  // ========================================================================
  // Subscription Methods
  // ========================================================================

  const subscribeToProject = useCallback((pid: string) => {
    if (socketRef.current?.connected) {
      socketRef.current.emit("subscribe:project", pid);
      console.log("[HIL WebSocket] Subscribed to project:", pid);
    }
  }, []);

  const unsubscribeFromProject = useCallback((pid: string) => {
    if (socketRef.current?.connected) {
      socketRef.current.emit("unsubscribe:project", pid);
      console.log("[HIL WebSocket] Unsubscribed from project:", pid);
    }
  }, []);

  const subscribeToOperation = useCallback((operationId: string) => {
    if (socketRef.current?.connected) {
      socketRef.current.emit("subscribe:operation", operationId);
      console.log("[HIL WebSocket] Subscribed to operation:", operationId);
    }
  }, []);

  const unsubscribeFromOperation = useCallback((operationId: string) => {
    if (socketRef.current?.connected) {
      socketRef.current.emit("unsubscribe:operation", operationId);
      console.log("[HIL WebSocket] Unsubscribed from operation:", operationId);
    }
  }, []);

  const subscribeToTestRun = useCallback((testRunId: string) => {
    if (socketRef.current?.connected) {
      socketRef.current.emit("subscribe:test_run", testRunId);
      console.log("[HIL WebSocket] Subscribed to test run:", testRunId);
    }
  }, []);

  const subscribeToInstrument = useCallback((instrumentId: string) => {
    if (socketRef.current?.connected) {
      socketRef.current.emit("subscribe:instrument", instrumentId);
      console.log("[HIL WebSocket] Subscribed to instrument:", instrumentId);
    }
  }, []);

  // ========================================================================
  // Control Methods
  // ========================================================================

  const requestEmergencyStop = useCallback((testRunId: string, reason: string) => {
    if (socketRef.current?.connected) {
      socketRef.current.emit("emergency_stop:request", { testRunId, reason });
      console.warn("[HIL WebSocket] Emergency stop requested:", reason);
    }
  }, []);

  // ========================================================================
  // Lifecycle
  // ========================================================================

  useEffect(() => {
    mountedRef.current = true;

    if (autoConnect) {
      connect();
    }

    return () => {
      mountedRef.current = false;
      disconnect();
    };
  }, [autoConnect, connect, disconnect]);

  // Subscribe to project changes
  useEffect(() => {
    if (projectId && socketRef.current?.connected) {
      subscribeToProject(projectId);
    }
  }, [projectId, subscribeToProject]);

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
    subscribeToOperation,
    unsubscribeFromOperation,
    subscribeToTestRun,
    subscribeToInstrument,
    requestEmergencyStop,
  };
}

export default useHILWebSocket;
