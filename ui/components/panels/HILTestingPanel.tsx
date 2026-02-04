"use client";

/**
 * HIL Testing Panel - Hardware-in-the-Loop Test Management
 *
 * Comprehensive interface for managing HIL instruments, test sequences,
 * running tests, and viewing results with live waveform streaming.
 */

import { useState, useEffect, useMemo, useCallback } from "react";
import { cn } from "@/lib/utils";
import * as Progress from "@radix-ui/react-progress";
import * as Tabs from "@radix-ui/react-tabs";
import {
  Play,
  Square,
  RefreshCw,
  Settings,
  CheckCircle,
  XCircle,
  AlertTriangle,
  Zap,
  Activity,
  Radio,
  Cpu,
  Gauge,
  Wifi,
  WifiOff,
  ChevronRight,
  ChevronDown,
  Plus,
  Trash2,
  Download,
  Upload,
  Clock,
  AlertOctagon,
  FileText,
  BarChart3,
  Waves,
  Power,
  Thermometer,
} from "lucide-react";
import {
  useHILStore,
  type HILInstrument,
  type HILTestSequence,
  type HILTestRun,
  type HILMeasurement,
  type HILInstrumentType,
  type HILInstrumentStatus,
  type HILTestRunStatus,
} from "@/hooks/useHILStore";
import { useHILWebSocket } from "@/hooks/useHILWebSocket";

// ============================================================================
// Types
// ============================================================================

interface HILTestingPanelProps {
  projectId: string;
  className?: string;
  onTestComplete?: (testRunId: string, result: string) => void;
}

// ============================================================================
// Helper Functions
// ============================================================================

const getInstrumentIcon = (type: HILInstrumentType) => {
  switch (type) {
    case "logic_analyzer":
      return <Radio className="w-4 h-4 text-cyan-400" />;
    case "oscilloscope":
      return <Waves className="w-4 h-4 text-green-400" />;
    case "power_supply":
      return <Power className="w-4 h-4 text-yellow-400" />;
    case "motor_emulator":
      return <Cpu className="w-4 h-4 text-orange-400" />;
    case "daq":
      return <Activity className="w-4 h-4 text-purple-400" />;
    case "can_analyzer":
      return <Gauge className="w-4 h-4 text-blue-400" />;
    case "function_gen":
      return <Waves className="w-4 h-4 text-pink-400" />;
    case "thermal_camera":
      return <Thermometer className="w-4 h-4 text-red-400" />;
    case "electronic_load":
      return <Zap className="w-4 h-4 text-amber-400" />;
    default:
      return <Settings className="w-4 h-4 text-slate-400" />;
  }
};

const getStatusColor = (status: HILInstrumentStatus): string => {
  switch (status) {
    case "connected":
      return "text-green-400";
    case "disconnected":
      return "text-slate-500";
    case "error":
      return "text-red-400";
    case "busy":
      return "text-yellow-400";
    case "initializing":
      return "text-blue-400";
    default:
      return "text-slate-400";
  }
};

const getStatusIcon = (status: HILInstrumentStatus) => {
  switch (status) {
    case "connected":
      return <Wifi className="w-4 h-4 text-green-400" />;
    case "disconnected":
      return <WifiOff className="w-4 h-4 text-slate-500" />;
    case "error":
      return <AlertOctagon className="w-4 h-4 text-red-400" />;
    case "busy":
      return <Activity className="w-4 h-4 text-yellow-400 animate-pulse" />;
    case "initializing":
      return <RefreshCw className="w-4 h-4 text-blue-400 animate-spin" />;
    default:
      return <Settings className="w-4 h-4 text-slate-400" />;
  }
};

const getTestRunStatusColor = (status: HILTestRunStatus): string => {
  switch (status) {
    case "completed":
      return "text-green-400";
    case "failed":
      return "text-red-400";
    case "running":
      return "text-blue-400";
    case "queued":
      return "text-yellow-400";
    case "aborted":
      return "text-orange-400";
    case "timeout":
      return "text-red-400";
    default:
      return "text-slate-400";
  }
};

const getResultIcon = (result?: string) => {
  switch (result) {
    case "pass":
      return <CheckCircle className="w-5 h-5 text-green-500" />;
    case "fail":
      return <XCircle className="w-5 h-5 text-red-500" />;
    case "partial":
      return <AlertTriangle className="w-5 h-5 text-yellow-500" />;
    default:
      return null;
  }
};

// ============================================================================
// Instrument Card Component
// ============================================================================

interface InstrumentCardProps {
  instrument: HILInstrument;
  isSelected: boolean;
  onSelect: () => void;
  onConnect: () => void;
  onDisconnect: () => void;
}

function InstrumentCard({
  instrument,
  isSelected,
  onSelect,
  onConnect,
  onDisconnect,
}: InstrumentCardProps) {
  const [expanded, setExpanded] = useState(false);

  return (
    <div
      className={cn(
        "rounded-lg border transition-all cursor-pointer",
        isSelected
          ? "border-primary-500 bg-primary-600/10"
          : "border-slate-700 hover:border-slate-600",
        instrument.status === "error" && "border-red-500/50"
      )}
      onClick={onSelect}
    >
      <div className="flex items-center gap-3 p-3">
        {getInstrumentIcon(instrument.instrumentType)}
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2">
            <span className="text-white font-medium truncate">{instrument.name}</span>
            {getStatusIcon(instrument.status)}
          </div>
          <div className="text-xs text-slate-500 truncate">
            {instrument.manufacturer} {instrument.model}
          </div>
        </div>
        <button
          onClick={(e) => {
            e.stopPropagation();
            setExpanded(!expanded);
          }}
          className="p-1 hover:bg-slate-700 rounded"
        >
          {expanded ? (
            <ChevronDown className="w-4 h-4 text-slate-400" />
          ) : (
            <ChevronRight className="w-4 h-4 text-slate-400" />
          )}
        </button>
      </div>

      {expanded && (
        <div className="px-3 pb-3 border-t border-slate-700/50 pt-2 space-y-2">
          <div className="grid grid-cols-2 gap-2 text-xs">
            <div className="text-slate-500">Connection:</div>
            <div className="text-slate-300 capitalize">{instrument.connectionType}</div>
            {instrument.serialNumber && (
              <>
                <div className="text-slate-500">Serial:</div>
                <div className="text-slate-300 font-mono text-[10px]">
                  {instrument.serialNumber}
                </div>
              </>
            )}
            {instrument.firmwareVersion && (
              <>
                <div className="text-slate-500">Firmware:</div>
                <div className="text-slate-300">{instrument.firmwareVersion}</div>
              </>
            )}
            <div className="text-slate-500">Status:</div>
            <div className={cn("capitalize", getStatusColor(instrument.status))}>
              {instrument.status}
            </div>
          </div>

          {instrument.lastError && (
            <div className="text-xs text-red-400 bg-red-500/10 p-2 rounded">
              {instrument.lastError}
            </div>
          )}

          <div className="flex gap-2 pt-2">
            {instrument.status === "connected" ? (
              <button
                onClick={(e) => {
                  e.stopPropagation();
                  onDisconnect();
                }}
                className="flex-1 px-2 py-1 bg-red-600/20 text-red-400 rounded text-xs hover:bg-red-600/30"
              >
                Disconnect
              </button>
            ) : (
              <button
                onClick={(e) => {
                  e.stopPropagation();
                  onConnect();
                }}
                disabled={instrument.status === "initializing"}
                className="flex-1 px-2 py-1 bg-primary-600/20 text-primary-400 rounded text-xs hover:bg-primary-600/30 disabled:opacity-50"
              >
                Connect
              </button>
            )}
          </div>
        </div>
      )}
    </div>
  );
}

// ============================================================================
// Test Sequence Card Component
// ============================================================================

interface TestSequenceCardProps {
  sequence: HILTestSequence;
  isSelected: boolean;
  onSelect: () => void;
  onRun: () => void;
}

function TestSequenceCard({ sequence, isSelected, onSelect, onRun }: TestSequenceCardProps) {
  return (
    <div
      className={cn(
        "rounded-lg border p-3 cursor-pointer transition-all",
        isSelected
          ? "border-primary-500 bg-primary-600/10"
          : "border-slate-700 hover:border-slate-600"
      )}
      onClick={onSelect}
    >
      <div className="flex items-start justify-between mb-2">
        <div>
          <h4 className="text-white font-medium">{sequence.name}</h4>
          <span className="text-xs text-slate-500 capitalize">
            {sequence.testType.replace(/_/g, " ")}
          </span>
        </div>
        <span
          className={cn(
            "text-[10px] px-2 py-0.5 rounded",
            sequence.isTemplate
              ? "bg-purple-500/20 text-purple-400"
              : "bg-slate-700 text-slate-400"
          )}
        >
          {sequence.isTemplate ? "Template" : `v${sequence.version}`}
        </span>
      </div>

      {sequence.description && (
        <p className="text-xs text-slate-400 line-clamp-2 mb-2">{sequence.description}</p>
      )}

      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2 text-xs text-slate-500">
          <span>{sequence.sequenceConfig.steps.length} steps</span>
          {sequence.estimatedDurationMs && (
            <>
              <span>•</span>
              <span>{Math.round(sequence.estimatedDurationMs / 1000)}s</span>
            </>
          )}
        </div>
        <button
          onClick={(e) => {
            e.stopPropagation();
            onRun();
          }}
          className="flex items-center gap-1 px-2 py-1 bg-green-600 text-white rounded text-xs hover:bg-green-700"
        >
          <Play className="w-3 h-3" />
          Run
        </button>
      </div>
    </div>
  );
}

// ============================================================================
// Test Run Card Component
// ============================================================================

interface TestRunCardProps {
  run: HILTestRun;
  isActive: boolean;
  onSelect: () => void;
  onAbort: () => void;
  onViewResults: () => void;
}

function TestRunCard({ run, isActive, onSelect, onAbort, onViewResults }: TestRunCardProps) {
  return (
    <div
      className={cn(
        "rounded-lg border p-3 cursor-pointer transition-all",
        isActive ? "border-primary-500 bg-primary-600/10" : "border-slate-700 hover:border-slate-600"
      )}
      onClick={onSelect}
    >
      <div className="flex items-start justify-between mb-2">
        <div>
          <h4 className="text-white font-medium">{run.name}</h4>
          <span className="text-xs text-slate-500">Run #{run.runNumber}</span>
        </div>
        <div className="flex items-center gap-1">
          {getResultIcon(run.result)}
          <span className={cn("text-xs capitalize", getTestRunStatusColor(run.status))}>
            {run.status}
          </span>
        </div>
      </div>

      {(run.status === "running" || run.status === "queued") && (
        <div className="mb-2">
          <Progress.Root
            className="relative h-1.5 w-full overflow-hidden rounded-full bg-slate-700"
            value={run.progressPercentage}
          >
            <Progress.Indicator
              className="h-full bg-primary-500 transition-all duration-300"
              style={{ width: `${run.progressPercentage}%` }}
            />
          </Progress.Root>
          <div className="flex justify-between text-[10px] text-slate-500 mt-1">
            <span>{run.currentStep || "Waiting..."}</span>
            <span>{Math.round(run.progressPercentage)}%</span>
          </div>
        </div>
      )}

      {run.summary && (
        <div className="grid grid-cols-3 gap-2 text-xs mb-2">
          <div className="text-center">
            <div className="text-green-400 font-bold">{run.summary.passedMeasurements}</div>
            <div className="text-slate-500">Pass</div>
          </div>
          <div className="text-center">
            <div className="text-red-400 font-bold">{run.summary.failedMeasurements}</div>
            <div className="text-slate-500">Fail</div>
          </div>
          <div className="text-center">
            <div className="text-white font-bold">
              {run.summary.score !== undefined ? `${run.summary.score}%` : "-"}
            </div>
            <div className="text-slate-500">Score</div>
          </div>
        </div>
      )}

      <div className="flex gap-2">
        {run.status === "running" && (
          <button
            onClick={(e) => {
              e.stopPropagation();
              onAbort();
            }}
            className="flex-1 flex items-center justify-center gap-1 px-2 py-1 bg-red-600/20 text-red-400 rounded text-xs hover:bg-red-600/30"
          >
            <Square className="w-3 h-3" />
            Abort
          </button>
        )}
        {(run.status === "completed" || run.status === "failed") && (
          <button
            onClick={(e) => {
              e.stopPropagation();
              onViewResults();
            }}
            className="flex-1 flex items-center justify-center gap-1 px-2 py-1 bg-slate-700 text-slate-300 rounded text-xs hover:bg-slate-600"
          >
            <BarChart3 className="w-3 h-3" />
            Results
          </button>
        )}
      </div>
    </div>
  );
}

// ============================================================================
// Live Measurements Display
// ============================================================================

interface LiveMeasurementsProps {
  measurements: Array<{
    type: string;
    channel?: string;
    value: number;
    unit: string;
    passed?: boolean;
    timestamp: Date;
  }>;
}

function LiveMeasurements({ measurements }: LiveMeasurementsProps) {
  const recentMeasurements = measurements.slice(-20);

  return (
    <div className="space-y-1 max-h-64 overflow-y-auto">
      {recentMeasurements.map((m, i) => (
        <div
          key={i}
          className={cn(
            "flex items-center justify-between px-2 py-1 rounded text-xs",
            m.passed === true && "bg-green-500/10",
            m.passed === false && "bg-red-500/10"
          )}
        >
          <div className="flex items-center gap-2">
            <span className="text-slate-400 capitalize">
              {m.type.replace(/_/g, " ")}
              {m.channel && <span className="text-slate-500">({m.channel})</span>}
            </span>
          </div>
          <div className="flex items-center gap-2">
            <span className={cn("font-mono", m.passed === false ? "text-red-400" : "text-white")}>
              {m.value.toFixed(3)} {m.unit}
            </span>
            {m.passed !== undefined && (
              m.passed ? (
                <CheckCircle className="w-3 h-3 text-green-400" />
              ) : (
                <XCircle className="w-3 h-3 text-red-400" />
              )
            )}
          </div>
        </div>
      ))}
      {recentMeasurements.length === 0 && (
        <div className="text-center py-4 text-slate-500 text-xs">
          No live measurements
        </div>
      )}
    </div>
  );
}

// ============================================================================
// Main Component
// ============================================================================

export function HILTestingPanel({ projectId, className, onTestComplete }: HILTestingPanelProps) {
  const [activeTab, setActiveTab] = useState<"instruments" | "sequences" | "runs" | "results">(
    "instruments"
  );
  const [isDiscovering, setIsDiscovering] = useState(false);

  // Store state
  const instruments = useHILStore((state) => state.instruments);
  const selectedInstrumentId = useHILStore((state) => state.selectedInstrumentId);
  const testSequences = useHILStore((state) => state.testSequences);
  const selectedSequenceId = useHILStore((state) => state.selectedSequenceId);
  const testRuns = useHILStore((state) => state.testRuns);
  const activeTestRun = useHILStore((state) => state.activeTestRun);
  const liveMeasurements = useHILStore((state) => state.liveMeasurements);
  const measurementSummary = useHILStore((state) => state.measurementSummary);
  const loading = useHILStore((state) => state.loading);
  const errors = useHILStore((state) => state.errors);

  // Store actions
  const setSelectedInstrument = useHILStore((state) => state.setSelectedInstrument);
  const setSelectedSequence = useHILStore((state) => state.setSelectedSequence);
  const setActiveTestRun = useHILStore((state) => state.setActiveTestRun);
  const setInstruments = useHILStore((state) => state.setInstruments);
  const setTestSequences = useHILStore((state) => state.setTestSequences);
  const setTestRuns = useHILStore((state) => state.setTestRuns);
  const setLoading = useHILStore((state) => state.setLoading);
  const setError = useHILStore((state) => state.setError);

  // WebSocket connection
  const { isConnected: wsConnected, subscribeToTestRun, requestEmergencyStop } = useHILWebSocket({
    projectId,
    onEmergencyStop: (event) => {
      console.error("Emergency stop triggered:", event.reason);
    },
    onFaultDetected: (event) => {
      console.warn("Fault detected:", event.faultType, event.message);
    },
  });

  // API base URL
  const apiUrl = process.env.NEXT_PUBLIC_API_URL || "http://localhost:9080";

  // Computed values
  const instrumentList = useMemo(() => Object.values(instruments), [instruments]);
  const connectedInstruments = useMemo(
    () => instrumentList.filter((i) => i.status === "connected"),
    [instrumentList]
  );

  // ========================================================================
  // API Calls
  // ========================================================================

  const fetchInstruments = useCallback(async () => {
    setLoading("instruments", true);
    setError("instruments", null);
    try {
      const response = await fetch(`${apiUrl}/api/v1/hil/projects/${projectId}/instruments`);
      const data = await response.json();
      if (data.success) {
        setInstruments(data.data);
      } else {
        setError("instruments", data.error?.message || "Failed to fetch instruments");
      }
    } catch (err) {
      setError("instruments", err instanceof Error ? err.message : "Failed to fetch instruments");
    } finally {
      setLoading("instruments", false);
    }
  }, [apiUrl, projectId, setInstruments, setLoading, setError]);

  const discoverInstruments = useCallback(async () => {
    setIsDiscovering(true);
    setError("discovery", null);
    try {
      const response = await fetch(`${apiUrl}/api/v1/hil/instruments/discover`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ projectId }),
      });
      const data = await response.json();
      if (!data.success) {
        setError("discovery", data.error?.message || "Discovery failed");
      }
      // Results will come via WebSocket
    } catch (err) {
      setError("discovery", err instanceof Error ? err.message : "Discovery failed");
    } finally {
      // Keep discovering state until WebSocket signals completion
      setTimeout(() => setIsDiscovering(false), 30000);
    }
  }, [apiUrl, projectId, setError]);

  const connectInstrument = useCallback(
    async (instrumentId: string) => {
      try {
        const response = await fetch(`${apiUrl}/api/v1/hil/instruments/${instrumentId}/connect`, {
          method: "POST",
        });
        const data = await response.json();
        if (!data.success) {
          console.error("Failed to connect instrument:", data.error);
        }
      } catch (err) {
        console.error("Failed to connect instrument:", err);
      }
    },
    [apiUrl]
  );

  const fetchTestSequences = useCallback(async () => {
    setLoading("sequences", true);
    setError("sequences", null);
    try {
      const response = await fetch(`${apiUrl}/api/v1/hil/projects/${projectId}/test-sequences`);
      const data = await response.json();
      if (data.success) {
        setTestSequences(data.data);
      } else {
        setError("sequences", data.error?.message || "Failed to fetch sequences");
      }
    } catch (err) {
      setError("sequences", err instanceof Error ? err.message : "Failed to fetch sequences");
    } finally {
      setLoading("sequences", false);
    }
  }, [apiUrl, projectId, setTestSequences, setLoading, setError]);

  const fetchTestRuns = useCallback(async () => {
    setLoading("testRuns", true);
    setError("testRuns", null);
    try {
      const response = await fetch(`${apiUrl}/api/v1/hil/projects/${projectId}/test-runs?limit=20`);
      const data = await response.json();
      if (data.success) {
        setTestRuns(data.data);
      } else {
        setError("testRuns", data.error?.message || "Failed to fetch test runs");
      }
    } catch (err) {
      setError("testRuns", err instanceof Error ? err.message : "Failed to fetch test runs");
    } finally {
      setLoading("testRuns", false);
    }
  }, [apiUrl, projectId, setTestRuns, setLoading, setError]);

  const startTestRun = useCallback(
    async (sequenceId: string) => {
      try {
        const response = await fetch(`${apiUrl}/api/v1/hil/test-sequences/${sequenceId}/run`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({}),
        });
        const data = await response.json();
        if (data.success) {
          subscribeToTestRun(data.data.testRunId);
          setActiveTab("runs");
          fetchTestRuns();
        } else {
          console.error("Failed to start test run:", data.error);
        }
      } catch (err) {
        console.error("Failed to start test run:", err);
      }
    },
    [apiUrl, subscribeToTestRun, fetchTestRuns]
  );

  const abortTestRun = useCallback(
    async (testRunId: string) => {
      try {
        const response = await fetch(`${apiUrl}/api/v1/hil/test-runs/${testRunId}/abort`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ reason: "User requested abort" }),
        });
        const data = await response.json();
        if (!data.success) {
          console.error("Failed to abort test run:", data.error);
        }
      } catch (err) {
        console.error("Failed to abort test run:", err);
      }
    },
    [apiUrl]
  );

  // ========================================================================
  // Effects
  // ========================================================================

  useEffect(() => {
    if (projectId) {
      fetchInstruments();
      fetchTestSequences();
      fetchTestRuns();
    }
  }, [projectId, fetchInstruments, fetchTestSequences, fetchTestRuns]);

  // Notify parent when test completes
  useEffect(() => {
    if (activeTestRun && (activeTestRun.status === "completed" || activeTestRun.status === "failed")) {
      onTestComplete?.(activeTestRun.id, activeTestRun.result || activeTestRun.status);
    }
  }, [activeTestRun, onTestComplete]);

  // ========================================================================
  // Render
  // ========================================================================

  return (
    <div className={cn("flex flex-col h-full bg-surface-primary", className)}>
      {/* Header */}
      <div className="flex items-center justify-between p-4 border-b border-slate-700">
        <div className="flex items-center gap-3">
          <Activity className="w-5 h-5 text-primary-400" />
          <div>
            <h2 className="text-lg font-semibold text-white">HIL Testing</h2>
            <p className="text-xs text-slate-500">
              {connectedInstruments.length} of {instrumentList.length} instruments connected
            </p>
          </div>
        </div>

        <div className="flex items-center gap-2">
          {/* WebSocket Status */}
          <div
            className={cn(
              "flex items-center gap-1 px-2 py-1 rounded text-xs",
              wsConnected ? "bg-green-500/20 text-green-400" : "bg-red-500/20 text-red-400"
            )}
          >
            {wsConnected ? <Wifi className="w-3 h-3" /> : <WifiOff className="w-3 h-3" />}
            {wsConnected ? "Live" : "Offline"}
          </div>

          {/* Active Test Indicator */}
          {activeTestRun && activeTestRun.status === "running" && (
            <div className="flex items-center gap-2 px-3 py-1 bg-primary-600/20 rounded">
              <Activity className="w-4 h-4 text-primary-400 animate-pulse" />
              <span className="text-xs text-primary-400">{activeTestRun.currentStep}</span>
              <span className="text-xs text-slate-500">
                {Math.round(activeTestRun.progressPercentage)}%
              </span>
            </div>
          )}
        </div>
      </div>

      {/* Tabs */}
      <Tabs.Root value={activeTab} onValueChange={(v) => setActiveTab(v as any)}>
        <Tabs.List className="flex border-b border-slate-700">
          <Tabs.Trigger
            value="instruments"
            className={cn(
              "flex-1 px-4 py-2 text-sm font-medium transition-colors",
              activeTab === "instruments"
                ? "text-primary-400 border-b-2 border-primary-400"
                : "text-slate-400 hover:text-white"
            )}
          >
            <div className="flex items-center justify-center gap-2">
              <Settings className="w-4 h-4" />
              Instruments
              {connectedInstruments.length > 0 && (
                <span className="px-1.5 py-0.5 bg-green-500/20 text-green-400 text-[10px] rounded">
                  {connectedInstruments.length}
                </span>
              )}
            </div>
          </Tabs.Trigger>
          <Tabs.Trigger
            value="sequences"
            className={cn(
              "flex-1 px-4 py-2 text-sm font-medium transition-colors",
              activeTab === "sequences"
                ? "text-primary-400 border-b-2 border-primary-400"
                : "text-slate-400 hover:text-white"
            )}
          >
            <div className="flex items-center justify-center gap-2">
              <FileText className="w-4 h-4" />
              Sequences
            </div>
          </Tabs.Trigger>
          <Tabs.Trigger
            value="runs"
            className={cn(
              "flex-1 px-4 py-2 text-sm font-medium transition-colors",
              activeTab === "runs"
                ? "text-primary-400 border-b-2 border-primary-400"
                : "text-slate-400 hover:text-white"
            )}
          >
            <div className="flex items-center justify-center gap-2">
              <Play className="w-4 h-4" />
              Test Runs
              {activeTestRun?.status === "running" && (
                <span className="px-1.5 py-0.5 bg-primary-500/20 text-primary-400 text-[10px] rounded animate-pulse">
                  Running
                </span>
              )}
            </div>
          </Tabs.Trigger>
          <Tabs.Trigger
            value="results"
            className={cn(
              "flex-1 px-4 py-2 text-sm font-medium transition-colors",
              activeTab === "results"
                ? "text-primary-400 border-b-2 border-primary-400"
                : "text-slate-400 hover:text-white"
            )}
          >
            <div className="flex items-center justify-center gap-2">
              <BarChart3 className="w-4 h-4" />
              Results
            </div>
          </Tabs.Trigger>
        </Tabs.List>

        {/* Instruments Tab */}
        <Tabs.Content value="instruments" className="flex-1 overflow-hidden">
          <div className="h-full flex flex-col">
            {/* Toolbar */}
            <div className="flex items-center gap-2 p-3 border-b border-slate-700/50">
              <button
                onClick={discoverInstruments}
                disabled={isDiscovering}
                className="flex items-center gap-1 px-3 py-1.5 bg-primary-600 text-white rounded text-sm hover:bg-primary-700 disabled:opacity-50"
              >
                <RefreshCw className={cn("w-4 h-4", isDiscovering && "animate-spin")} />
                {isDiscovering ? "Discovering..." : "Discover"}
              </button>
              <button
                onClick={fetchInstruments}
                disabled={loading.instruments}
                className="flex items-center gap-1 px-3 py-1.5 bg-slate-700 text-slate-300 rounded text-sm hover:bg-slate-600"
              >
                <RefreshCw className={cn("w-4 h-4", loading.instruments && "animate-spin")} />
                Refresh
              </button>
            </div>

            {/* Instrument List */}
            <div className="flex-1 overflow-y-auto p-3 space-y-2">
              {loading.instruments ? (
                <div className="flex items-center justify-center py-8">
                  <RefreshCw className="w-6 h-6 text-slate-500 animate-spin" />
                </div>
              ) : instrumentList.length === 0 ? (
                <div className="text-center py-8 text-slate-500">
                  <Settings className="w-12 h-12 mx-auto mb-3" />
                  <p>No instruments detected</p>
                  <button
                    onClick={discoverInstruments}
                    className="mt-2 text-primary-400 hover:text-primary-300"
                  >
                    Start discovery
                  </button>
                </div>
              ) : (
                instrumentList.map((instrument) => (
                  <InstrumentCard
                    key={instrument.id}
                    instrument={instrument}
                    isSelected={selectedInstrumentId === instrument.id}
                    onSelect={() => setSelectedInstrument(instrument.id)}
                    onConnect={() => connectInstrument(instrument.id)}
                    onDisconnect={() => {}}
                  />
                ))
              )}
            </div>
          </div>
        </Tabs.Content>

        {/* Sequences Tab */}
        <Tabs.Content value="sequences" className="flex-1 overflow-hidden">
          <div className="h-full flex flex-col">
            {/* Toolbar */}
            <div className="flex items-center gap-2 p-3 border-b border-slate-700/50">
              <button className="flex items-center gap-1 px-3 py-1.5 bg-primary-600 text-white rounded text-sm hover:bg-primary-700">
                <Plus className="w-4 h-4" />
                Create
              </button>
              <button
                onClick={fetchTestSequences}
                disabled={loading.sequences}
                className="flex items-center gap-1 px-3 py-1.5 bg-slate-700 text-slate-300 rounded text-sm hover:bg-slate-600"
              >
                <RefreshCw className={cn("w-4 h-4", loading.sequences && "animate-spin")} />
              </button>
            </div>

            {/* Sequence List */}
            <div className="flex-1 overflow-y-auto p-3 space-y-2">
              {loading.sequences ? (
                <div className="flex items-center justify-center py-8">
                  <RefreshCw className="w-6 h-6 text-slate-500 animate-spin" />
                </div>
              ) : testSequences.length === 0 ? (
                <div className="text-center py-8 text-slate-500">
                  <FileText className="w-12 h-12 mx-auto mb-3" />
                  <p>No test sequences</p>
                  <p className="text-xs mt-1">Create a sequence or use a template</p>
                </div>
              ) : (
                testSequences.map((sequence) => (
                  <TestSequenceCard
                    key={sequence.id}
                    sequence={sequence}
                    isSelected={selectedSequenceId === sequence.id}
                    onSelect={() => setSelectedSequence(sequence.id)}
                    onRun={() => startTestRun(sequence.id)}
                  />
                ))
              )}
            </div>
          </div>
        </Tabs.Content>

        {/* Test Runs Tab */}
        <Tabs.Content value="runs" className="flex-1 overflow-hidden">
          <div className="h-full grid grid-cols-2 gap-4 p-3">
            {/* Test Runs List */}
            <div className="flex flex-col bg-slate-800/50 rounded-lg border border-slate-700 overflow-hidden">
              <div className="flex items-center justify-between p-3 border-b border-slate-700">
                <h3 className="text-sm font-medium text-white">Recent Runs</h3>
                <button
                  onClick={fetchTestRuns}
                  disabled={loading.testRuns}
                  className="p-1 hover:bg-slate-700 rounded"
                >
                  <RefreshCw
                    className={cn("w-4 h-4 text-slate-400", loading.testRuns && "animate-spin")}
                  />
                </button>
              </div>
              <div className="flex-1 overflow-y-auto p-2 space-y-2">
                {testRuns.length === 0 ? (
                  <div className="text-center py-8 text-slate-500 text-sm">No test runs yet</div>
                ) : (
                  testRuns.map((run) => (
                    <TestRunCard
                      key={run.id}
                      run={run}
                      isActive={activeTestRun?.id === run.id}
                      onSelect={() => setActiveTestRun(run)}
                      onAbort={() => abortTestRun(run.id)}
                      onViewResults={() => {
                        setActiveTestRun(run);
                        setActiveTab("results");
                      }}
                    />
                  ))
                )}
              </div>
            </div>

            {/* Live Measurements */}
            <div className="flex flex-col bg-slate-800/50 rounded-lg border border-slate-700 overflow-hidden">
              <div className="flex items-center justify-between p-3 border-b border-slate-700">
                <h3 className="text-sm font-medium text-white">Live Measurements</h3>
                {activeTestRun?.status === "running" && (
                  <Activity className="w-4 h-4 text-green-400 animate-pulse" />
                )}
              </div>
              <div className="flex-1 overflow-y-auto p-2">
                <LiveMeasurements measurements={liveMeasurements} />
              </div>
            </div>
          </div>
        </Tabs.Content>

        {/* Results Tab */}
        <Tabs.Content value="results" className="flex-1 overflow-hidden">
          <div className="h-full p-4">
            {activeTestRun ? (
              <div className="space-y-4">
                {/* Summary Header */}
                <div className="flex items-center justify-between p-4 bg-slate-800/50 rounded-lg border border-slate-700">
                  <div>
                    <h3 className="text-lg font-medium text-white">{activeTestRun.name}</h3>
                    <p className="text-sm text-slate-400">
                      Run #{activeTestRun.runNumber} •{" "}
                      {activeTestRun.durationMs
                        ? `${(activeTestRun.durationMs / 1000).toFixed(1)}s`
                        : "In progress"}
                    </p>
                  </div>
                  <div className="flex items-center gap-3">
                    {getResultIcon(activeTestRun.result)}
                    <span
                      className={cn(
                        "text-lg font-bold capitalize",
                        getTestRunStatusColor(activeTestRun.status)
                      )}
                    >
                      {activeTestRun.result || activeTestRun.status}
                    </span>
                  </div>
                </div>

                {/* Summary Stats */}
                {measurementSummary && (
                  <div className="grid grid-cols-4 gap-4">
                    <div className="p-4 bg-slate-800/50 rounded-lg border border-slate-700 text-center">
                      <div className="text-2xl font-bold text-white">
                        {measurementSummary.totalMeasurements}
                      </div>
                      <div className="text-sm text-slate-400">Total</div>
                    </div>
                    <div className="p-4 bg-green-500/10 rounded-lg border border-green-500/30 text-center">
                      <div className="text-2xl font-bold text-green-400">
                        {measurementSummary.passedMeasurements}
                      </div>
                      <div className="text-sm text-green-400/70">Passed</div>
                    </div>
                    <div className="p-4 bg-red-500/10 rounded-lg border border-red-500/30 text-center">
                      <div className="text-2xl font-bold text-red-400">
                        {measurementSummary.failedMeasurements}
                      </div>
                      <div className="text-sm text-red-400/70">Failed</div>
                    </div>
                    <div className="p-4 bg-primary-500/10 rounded-lg border border-primary-500/30 text-center">
                      <div className="text-2xl font-bold text-primary-400">
                        {measurementSummary.score !== undefined
                          ? `${measurementSummary.score}%`
                          : "-"}
                      </div>
                      <div className="text-sm text-primary-400/70">Score</div>
                    </div>
                  </div>
                )}

                {/* Key Metrics */}
                {measurementSummary?.keyMetrics &&
                  Object.keys(measurementSummary.keyMetrics).length > 0 && (
                    <div className="p-4 bg-slate-800/50 rounded-lg border border-slate-700">
                      <h4 className="text-sm font-medium text-white mb-3">Key Metrics</h4>
                      <div className="grid grid-cols-3 gap-4">
                        {Object.entries(measurementSummary.keyMetrics).map(([key, value]) => (
                          <div key={key} className="text-center">
                            <div className="text-lg font-mono text-white">
                              {typeof value === "number" ? value.toFixed(3) : value}
                            </div>
                            <div className="text-xs text-slate-500 capitalize">
                              {key.replace(/_/g, " ")}
                            </div>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}

                {/* Critical Failures */}
                {measurementSummary?.criticalFailures &&
                  measurementSummary.criticalFailures.length > 0 && (
                    <div className="p-4 bg-red-500/10 rounded-lg border border-red-500/30">
                      <h4 className="text-sm font-medium text-red-400 mb-2 flex items-center gap-2">
                        <AlertTriangle className="w-4 h-4" />
                        Critical Failures
                      </h4>
                      <ul className="space-y-1">
                        {measurementSummary.criticalFailures.map((failure, i) => (
                          <li key={i} className="text-sm text-red-300">
                            {failure}
                          </li>
                        ))}
                      </ul>
                    </div>
                  )}

                {/* Error Message */}
                {activeTestRun.errorMessage && (
                  <div className="p-4 bg-red-500/10 rounded-lg border border-red-500/30">
                    <h4 className="text-sm font-medium text-red-400 mb-2">Error</h4>
                    <p className="text-sm text-red-300">{activeTestRun.errorMessage}</p>
                  </div>
                )}
              </div>
            ) : (
              <div className="flex flex-col items-center justify-center h-full text-slate-500">
                <BarChart3 className="w-16 h-16 mb-4" />
                <p>Select a test run to view results</p>
              </div>
            )}
          </div>
        </Tabs.Content>
      </Tabs.Root>
    </div>
  );
}

export default HILTestingPanel;
