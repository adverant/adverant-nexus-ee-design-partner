"use client";

/**
 * Bench Testing Panel - Test Point Configuration and Results
 *
 * Interface for configuring test points, running bench tests,
 * and viewing/exporting test results.
 */

import { useState, useMemo } from "react";
import { cn } from "@/lib/utils";
import * as Progress from "@radix-ui/react-progress";
import {
  Plus,
  Trash2,
  Play,
  Square,
  Download,
  Save,
  CheckCircle,
  XCircle,
  AlertTriangle,
  Zap,
  Activity,
  Clock,
  Settings,
  ChevronDown,
  ChevronRight,
} from "lucide-react";

// ============================================================================
// Types
// ============================================================================

export interface TestPoint {
  id: string;
  name: string;
  type: "voltage" | "current" | "frequency" | "digital" | "resistance";
  reference: string; // Component reference (e.g., "U1.pin3")
  expectedValue: string;
  tolerance: string;
  unit: string;
  actualValue?: string;
  status: "pending" | "pass" | "fail" | "running" | "skipped";
  notes?: string;
}

export interface TestSession {
  id: string;
  name: string;
  startedAt: string;
  completedAt?: string;
  status: "pending" | "running" | "completed" | "aborted";
  progress: number;
  passCount: number;
  failCount: number;
  testPoints: TestPoint[];
}

interface BenchTestingProps {
  /** PCB layout ID */
  pcbLayoutId: string;
  /** Optional class name */
  className?: string;
  /** Initial test points */
  initialTestPoints?: TestPoint[];
  /** Callback when test points change */
  onTestPointsChange?: (points: TestPoint[]) => void;
  /** Callback to run tests */
  onRunTests?: (testPointIds: string[]) => Promise<void>;
}

// ============================================================================
// Helper Functions
// ============================================================================

const getTypeIcon = (type: TestPoint["type"]) => {
  switch (type) {
    case "voltage":
      return <Zap className="w-4 h-4 text-yellow-400" />;
    case "current":
      return <Activity className="w-4 h-4 text-orange-400" />;
    case "frequency":
      return <Clock className="w-4 h-4 text-purple-400" />;
    case "digital":
      return <CheckCircle className="w-4 h-4 text-cyan-400" />;
    case "resistance":
      return <Settings className="w-4 h-4 text-green-400" />;
  }
};

const getStatusIcon = (status: TestPoint["status"]) => {
  switch (status) {
    case "pass":
      return <CheckCircle className="w-5 h-5 text-green-500" />;
    case "fail":
      return <XCircle className="w-5 h-5 text-red-500" />;
    case "running":
      return <Activity className="w-5 h-5 text-primary-500 animate-pulse" />;
    case "skipped":
      return <AlertTriangle className="w-5 h-5 text-yellow-500" />;
    default:
      return <div className="w-5 h-5 rounded-full border-2 border-slate-600" />;
  }
};

const getDefaultUnit = (type: TestPoint["type"]): string => {
  switch (type) {
    case "voltage":
      return "V";
    case "current":
      return "mA";
    case "frequency":
      return "Hz";
    case "digital":
      return "";
    case "resistance":
      return "Ω";
  }
};

// ============================================================================
// Test Point Editor Component
// ============================================================================

interface TestPointEditorProps {
  testPoint: TestPoint;
  onSave: (tp: TestPoint) => void;
  onCancel: () => void;
}

function TestPointEditor({ testPoint, onSave, onCancel }: TestPointEditorProps) {
  const [formData, setFormData] = useState<TestPoint>(testPoint);

  const handleTypeChange = (type: TestPoint["type"]) => {
    setFormData({
      ...formData,
      type,
      unit: getDefaultUnit(type),
    });
  };

  return (
    <div className="p-4 bg-slate-800 rounded-lg border border-slate-700 space-y-4">
      <div className="grid grid-cols-2 gap-4">
        <div>
          <label className="block text-sm text-slate-400 mb-1">Test Name</label>
          <input
            type="text"
            value={formData.name}
            onChange={(e) => setFormData({ ...formData, name: e.target.value })}
            className="w-full px-3 py-2 bg-slate-900 border border-slate-700 rounded-lg text-white"
            placeholder="VCC Rail Test"
          />
        </div>
        <div>
          <label className="block text-sm text-slate-400 mb-1">Reference</label>
          <input
            type="text"
            value={formData.reference}
            onChange={(e) => setFormData({ ...formData, reference: e.target.value })}
            className="w-full px-3 py-2 bg-slate-900 border border-slate-700 rounded-lg text-white"
            placeholder="TP1, U1.VCC"
          />
        </div>
      </div>

      <div>
        <label className="block text-sm text-slate-400 mb-1">Test Type</label>
        <div className="flex gap-2">
          {(["voltage", "current", "frequency", "digital", "resistance"] as const).map((type) => (
            <button
              key={type}
              onClick={() => handleTypeChange(type)}
              className={cn(
                "flex items-center gap-1 px-3 py-1.5 rounded-lg text-sm capitalize transition-colors",
                formData.type === type
                  ? "bg-primary-600 text-white"
                  : "bg-slate-700 text-slate-400 hover:bg-slate-600"
              )}
            >
              {getTypeIcon(type)}
              {type}
            </button>
          ))}
        </div>
      </div>

      <div className="grid grid-cols-3 gap-4">
        <div>
          <label className="block text-sm text-slate-400 mb-1">Expected Value</label>
          <input
            type="text"
            value={formData.expectedValue}
            onChange={(e) => setFormData({ ...formData, expectedValue: e.target.value })}
            className="w-full px-3 py-2 bg-slate-900 border border-slate-700 rounded-lg text-white"
            placeholder="3.3"
          />
        </div>
        <div>
          <label className="block text-sm text-slate-400 mb-1">Unit</label>
          <input
            type="text"
            value={formData.unit}
            onChange={(e) => setFormData({ ...formData, unit: e.target.value })}
            className="w-full px-3 py-2 bg-slate-900 border border-slate-700 rounded-lg text-white"
            placeholder="V"
          />
        </div>
        <div>
          <label className="block text-sm text-slate-400 mb-1">Tolerance</label>
          <input
            type="text"
            value={formData.tolerance}
            onChange={(e) => setFormData({ ...formData, tolerance: e.target.value })}
            className="w-full px-3 py-2 bg-slate-900 border border-slate-700 rounded-lg text-white"
            placeholder="±5%"
          />
        </div>
      </div>

      <div>
        <label className="block text-sm text-slate-400 mb-1">Notes</label>
        <textarea
          value={formData.notes || ""}
          onChange={(e) => setFormData({ ...formData, notes: e.target.value })}
          className="w-full px-3 py-2 bg-slate-900 border border-slate-700 rounded-lg text-white resize-none"
          rows={2}
          placeholder="Additional test notes..."
        />
      </div>

      <div className="flex justify-end gap-2">
        <button
          onClick={onCancel}
          className="px-4 py-2 text-slate-400 hover:text-white"
        >
          Cancel
        </button>
        <button
          onClick={() => onSave(formData)}
          className="flex items-center gap-2 px-4 py-2 bg-primary-600 text-white rounded-lg hover:bg-primary-700"
        >
          <Save className="w-4 h-4" />
          Save
        </button>
      </div>
    </div>
  );
}

// ============================================================================
// Main Component
// ============================================================================

export function BenchTesting({
  pcbLayoutId,
  className,
  initialTestPoints = [],
  onTestPointsChange,
  onRunTests,
}: BenchTestingProps) {
  const [testPoints, setTestPoints] = useState<TestPoint[]>(initialTestPoints);
  const [currentSession, setCurrentSession] = useState<TestSession | null>(null);
  const [selectedIds, setSelectedIds] = useState<Set<string>>(new Set());
  const [editingId, setEditingId] = useState<string | null>(null);
  const [isRunning, setIsRunning] = useState(false);
  const [expandedResults, setExpandedResults] = useState<Set<string>>(new Set());

  // Summary stats
  const stats = useMemo(() => {
    const total = testPoints.length;
    const passed = testPoints.filter((tp) => tp.status === "pass").length;
    const failed = testPoints.filter((tp) => tp.status === "fail").length;
    const pending = testPoints.filter((tp) => tp.status === "pending").length;
    return { total, passed, failed, pending };
  }, [testPoints]);

  // Handlers
  const handleAddTestPoint = () => {
    const newTestPoint: TestPoint = {
      id: crypto.randomUUID(),
      name: `Test Point ${testPoints.length + 1}`,
      type: "voltage",
      reference: "",
      expectedValue: "",
      tolerance: "±5%",
      unit: "V",
      status: "pending",
    };
    setTestPoints((prev) => [...prev, newTestPoint]);
    setEditingId(newTestPoint.id);
  };

  const handleSaveTestPoint = (tp: TestPoint) => {
    setTestPoints((prev) => {
      const updated = prev.map((p) => (p.id === tp.id ? tp : p));
      onTestPointsChange?.(updated);
      return updated;
    });
    setEditingId(null);
  };

  const handleDeleteSelected = () => {
    if (selectedIds.size === 0) return;
    if (!confirm(`Delete ${selectedIds.size} test point(s)?`)) return;

    setTestPoints((prev) => {
      const updated = prev.filter((tp) => !selectedIds.has(tp.id));
      onTestPointsChange?.(updated);
      return updated;
    });
    setSelectedIds(new Set());
  };

  const handleRunTests = async () => {
    const idsToRun = selectedIds.size > 0 ? Array.from(selectedIds) : testPoints.map((tp) => tp.id);
    if (idsToRun.length === 0) return;

    setIsRunning(true);

    // Create new session
    const session: TestSession = {
      id: crypto.randomUUID(),
      name: `Test Session ${new Date().toLocaleTimeString()}`,
      startedAt: new Date().toISOString(),
      status: "running",
      progress: 0,
      passCount: 0,
      failCount: 0,
      testPoints: testPoints.filter((tp) => idsToRun.includes(tp.id)),
    };
    setCurrentSession(session);

    // Simulate running tests (in real implementation, call onRunTests)
    if (onRunTests) {
      await onRunTests(idsToRun);
    } else {
      // Simulate test execution
      for (let i = 0; i < idsToRun.length; i++) {
        const tpId = idsToRun[i];
        setTestPoints((prev) =>
          prev.map((tp) =>
            tp.id === tpId ? { ...tp, status: "running" as const } : tp
          )
        );

        // Simulate test delay
        await new Promise((resolve) => setTimeout(resolve, 500 + Math.random() * 500));

        // Simulate pass/fail (80% pass rate)
        const passed = Math.random() > 0.2;
        const actualValue = passed
          ? testPoints.find((tp) => tp.id === tpId)?.expectedValue
          : String(parseFloat(testPoints.find((tp) => tp.id === tpId)?.expectedValue || "0") * (0.8 + Math.random() * 0.4));

        setTestPoints((prev) =>
          prev.map((tp) =>
            tp.id === tpId
              ? {
                  ...tp,
                  status: passed ? "pass" : "fail",
                  actualValue,
                }
              : tp
          )
        );

        setCurrentSession((prev) =>
          prev
            ? {
                ...prev,
                progress: ((i + 1) / idsToRun.length) * 100,
                passCount: prev.passCount + (passed ? 1 : 0),
                failCount: prev.failCount + (passed ? 0 : 1),
              }
            : prev
        );
      }
    }

    setCurrentSession((prev) =>
      prev
        ? {
            ...prev,
            status: "completed",
            completedAt: new Date().toISOString(),
          }
        : prev
    );
    setIsRunning(false);
  };

  const handleAbortTests = () => {
    setIsRunning(false);
    setCurrentSession((prev) =>
      prev ? { ...prev, status: "aborted" } : prev
    );
    setTestPoints((prev) =>
      prev.map((tp) =>
        tp.status === "running" ? { ...tp, status: "pending" } : tp
      )
    );
  };

  const handleExportResults = () => {
    const results = testPoints.map((tp) => ({
      name: tp.name,
      type: tp.type,
      reference: tp.reference,
      expected: `${tp.expectedValue}${tp.unit}`,
      actual: tp.actualValue ? `${tp.actualValue}${tp.unit}` : "N/A",
      tolerance: tp.tolerance,
      status: tp.status,
      notes: tp.notes || "",
    }));

    const csv = [
      ["Name", "Type", "Reference", "Expected", "Actual", "Tolerance", "Status", "Notes"].join(","),
      ...results.map((r) => Object.values(r).join(",")),
    ].join("\n");

    const blob = new Blob([csv], { type: "text/csv" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `bench-test-results-${pcbLayoutId}.csv`;
    a.click();
    URL.revokeObjectURL(url);
  };

  return (
    <div className={cn("flex flex-col h-full", className)}>
      {/* Header */}
      <div className="flex items-center justify-between p-4 border-b border-slate-700">
        <div>
          <h2 className="text-lg font-semibold text-white">Bench Testing</h2>
          <p className="text-xs text-slate-500">PCB: {pcbLayoutId.substring(0, 8)}</p>
        </div>

        {/* Stats */}
        <div className="flex items-center gap-4 text-sm">
          <span className="text-slate-400">
            Total: <strong className="text-white">{stats.total}</strong>
          </span>
          <span className="text-green-400">
            Pass: <strong>{stats.passed}</strong>
          </span>
          <span className="text-red-400">
            Fail: <strong>{stats.failed}</strong>
          </span>
        </div>
      </div>

      {/* Main Content */}
      <div className="flex-1 grid grid-cols-2 gap-4 p-4 overflow-hidden">
        {/* Test Points Configuration */}
        <div className="flex flex-col bg-surface-primary rounded-lg border border-slate-700 overflow-hidden">
          <div className="flex items-center gap-2 p-3 border-b border-slate-700">
            <h3 className="font-medium text-white flex-1">Test Points</h3>
            <button
              onClick={handleAddTestPoint}
              className="flex items-center gap-1 px-2 py-1 bg-primary-600 text-white rounded text-sm hover:bg-primary-700"
            >
              <Plus className="w-4 h-4" />
              Add
            </button>
            <button
              onClick={handleDeleteSelected}
              disabled={selectedIds.size === 0}
              className="flex items-center gap-1 px-2 py-1 bg-red-600/20 text-red-400 rounded text-sm hover:bg-red-600/30 disabled:opacity-50"
            >
              <Trash2 className="w-4 h-4" />
            </button>
          </div>

          <div className="flex-1 overflow-y-auto p-3 space-y-2">
            {testPoints.map((tp) =>
              editingId === tp.id ? (
                <TestPointEditor
                  key={tp.id}
                  testPoint={tp}
                  onSave={handleSaveTestPoint}
                  onCancel={() => setEditingId(null)}
                />
              ) : (
                <div
                  key={tp.id}
                  className={cn(
                    "flex items-center gap-3 p-3 rounded-lg border cursor-pointer transition-colors",
                    selectedIds.has(tp.id)
                      ? "border-primary-500 bg-primary-600/10"
                      : "border-slate-700 hover:border-slate-600"
                  )}
                  onClick={() => {
                    setSelectedIds((prev) => {
                      const next = new Set(prev);
                      if (next.has(tp.id)) {
                        next.delete(tp.id);
                      } else {
                        next.add(tp.id);
                      }
                      return next;
                    });
                  }}
                  onDoubleClick={() => setEditingId(tp.id)}
                >
                  <input
                    type="checkbox"
                    checked={selectedIds.has(tp.id)}
                    onChange={() => {}}
                    className="rounded border-slate-600"
                    onClick={(e) => e.stopPropagation()}
                  />
                  {getStatusIcon(tp.status)}
                  <div className="flex-1 min-w-0">
                    <div className="flex items-center gap-2">
                      {getTypeIcon(tp.type)}
                      <span className="text-white font-medium">{tp.name}</span>
                      <span className="text-xs text-slate-500">{tp.reference}</span>
                    </div>
                    <div className="text-xs text-slate-400 mt-1">
                      Expected: {tp.expectedValue}{tp.unit} {tp.tolerance}
                      {tp.actualValue && (
                        <span className={cn("ml-2", tp.status === "pass" ? "text-green-400" : "text-red-400")}>
                          | Actual: {tp.actualValue}{tp.unit}
                        </span>
                      )}
                    </div>
                  </div>
                </div>
              )
            )}

            {testPoints.length === 0 && (
              <div className="text-center py-8 text-slate-500">
                <Settings className="w-12 h-12 mx-auto mb-3" />
                <p>No test points configured</p>
                <button
                  onClick={handleAddTestPoint}
                  className="mt-2 text-primary-400 hover:text-primary-300"
                >
                  Add your first test point
                </button>
              </div>
            )}
          </div>
        </div>

        {/* Test Results / Control Panel */}
        <div className="flex flex-col bg-surface-primary rounded-lg border border-slate-700 overflow-hidden">
          <div className="flex items-center gap-2 p-3 border-b border-slate-700">
            <h3 className="font-medium text-white flex-1">Test Control</h3>
            {!isRunning ? (
              <button
                onClick={handleRunTests}
                disabled={testPoints.length === 0}
                className="flex items-center gap-1 px-3 py-1.5 bg-green-600 text-white rounded text-sm hover:bg-green-700 disabled:opacity-50"
              >
                <Play className="w-4 h-4" />
                Run {selectedIds.size > 0 ? `(${selectedIds.size})` : "All"}
              </button>
            ) : (
              <button
                onClick={handleAbortTests}
                className="flex items-center gap-1 px-3 py-1.5 bg-red-600 text-white rounded text-sm hover:bg-red-700"
              >
                <Square className="w-4 h-4" />
                Abort
              </button>
            )}
            <button
              onClick={handleExportResults}
              disabled={stats.passed + stats.failed === 0}
              className="flex items-center gap-1 px-3 py-1.5 bg-slate-700 text-slate-300 rounded text-sm hover:bg-slate-600 disabled:opacity-50"
            >
              <Download className="w-4 h-4" />
              Export
            </button>
          </div>

          <div className="flex-1 p-4 overflow-y-auto">
            {/* Session Progress */}
            {currentSession && (
              <div className="mb-6 p-4 bg-slate-800 rounded-lg">
                <div className="flex items-center justify-between mb-2">
                  <span className="text-sm font-medium text-white">{currentSession.name}</span>
                  <span className={cn(
                    "text-xs px-2 py-0.5 rounded",
                    currentSession.status === "completed" ? "bg-green-500/20 text-green-400" :
                    currentSession.status === "aborted" ? "bg-red-500/20 text-red-400" :
                    "bg-primary-500/20 text-primary-400"
                  )}>
                    {currentSession.status}
                  </span>
                </div>

                <Progress.Root
                  className="relative h-2 w-full overflow-hidden rounded-full bg-slate-700 mb-2"
                  value={currentSession.progress}
                >
                  <Progress.Indicator
                    className="h-full bg-primary-500 transition-all duration-300"
                    style={{ width: `${currentSession.progress}%` }}
                  />
                </Progress.Root>

                <div className="flex items-center gap-4 text-xs text-slate-400">
                  <span>Pass: <strong className="text-green-400">{currentSession.passCount}</strong></span>
                  <span>Fail: <strong className="text-red-400">{currentSession.failCount}</strong></span>
                  <span className="ml-auto">{Math.round(currentSession.progress)}%</span>
                </div>
              </div>
            )}

            {/* Results Summary */}
            {(stats.passed > 0 || stats.failed > 0) && (
              <div className="space-y-2">
                <h4 className="text-sm font-medium text-slate-400 mb-3">Results</h4>

                {testPoints
                  .filter((tp) => tp.status !== "pending")
                  .map((tp) => (
                    <div
                      key={tp.id}
                      className={cn(
                        "flex items-center gap-3 p-2 rounded-lg border",
                        tp.status === "pass"
                          ? "border-green-500/30 bg-green-500/5"
                          : tp.status === "fail"
                          ? "border-red-500/30 bg-red-500/5"
                          : "border-slate-700"
                      )}
                    >
                      {getStatusIcon(tp.status)}
                      <div className="flex-1">
                        <span className="text-sm text-white">{tp.name}</span>
                        <span className="text-xs text-slate-500 ml-2">{tp.reference}</span>
                      </div>
                      <div className="text-right text-xs">
                        <div className="text-slate-400">
                          Expected: {tp.expectedValue}{tp.unit}
                        </div>
                        {tp.actualValue && (
                          <div className={tp.status === "pass" ? "text-green-400" : "text-red-400"}>
                            Actual: {tp.actualValue}{tp.unit}
                          </div>
                        )}
                      </div>
                    </div>
                  ))}
              </div>
            )}

            {stats.passed === 0 && stats.failed === 0 && !currentSession && (
              <div className="text-center py-12 text-slate-500">
                <Activity className="w-12 h-12 mx-auto mb-3" />
                <p>No test results yet</p>
                <p className="text-xs mt-1">Configure test points and run tests</p>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}

export default BenchTesting;
