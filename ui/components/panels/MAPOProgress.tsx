"use client";

/**
 * MAPO Progress Panel - Real-time generation progress display
 *
 * Shows the progress of the Multi-Agent Pattern Orchestration pipeline
 * during schematic or PCB generation.
 */

import { useEffect, useState } from "react";
import { cn } from "@/lib/utils";
import * as Progress from "@radix-ui/react-progress";
import {
  CheckCircle,
  Circle,
  Loader2,
  XCircle,
  Cpu,
  Layers,
  GitBranch,
  Eye,
  Zap,
} from "lucide-react";

// ============================================================================
// Types
// ============================================================================

export interface MAPOStage {
  id: string;
  name: string;
  status: "pending" | "running" | "completed" | "error";
  progress: number;
  message?: string;
  startedAt?: string;
  completedAt?: string;
  duration?: number;
}

interface MAPOProgressProps {
  /** Unique generation ID to track */
  generationId: string;
  /** Type of generation */
  generationType: "schematic" | "pcb";
  /** Optional class name */
  className?: string;
  /** Callback when generation completes */
  onComplete?: (success: boolean) => void;
  /** Callback when a stage updates */
  onStageUpdate?: (stage: MAPOStage) => void;
  /** External stages data (for controlled mode) */
  stages?: MAPOStage[];
  /** Overall progress (for controlled mode) */
  overallProgress?: number;
}

// ============================================================================
// Default Stages
// ============================================================================

const DEFAULT_SCHEMATIC_STAGES: MAPOStage[] = [
  { id: "symbol-fetch", name: "Symbol Fetching", status: "pending", progress: 0 },
  { id: "schematic-assembly", name: "Schematic Assembly", status: "pending", progress: 0 },
  { id: "wire-routing", name: "Wire Routing", status: "pending", progress: 0 },
  { id: "vision-validation", name: "Vision Validation", status: "pending", progress: 0 },
];

const DEFAULT_PCB_STAGES: MAPOStage[] = [
  { id: "component-placement", name: "Component Placement", status: "pending", progress: 0 },
  { id: "trace-routing", name: "Trace Routing", status: "pending", progress: 0 },
  { id: "drc-check", name: "DRC Check", status: "pending", progress: 0 },
  { id: "optimization", name: "Optimization", status: "pending", progress: 0 },
  { id: "3d-export", name: "3D Export", status: "pending", progress: 0 },
];

// ============================================================================
// Helper Components
// ============================================================================

const StageIcon = ({ stageId }: { stageId: string }) => {
  const iconClass = "w-4 h-4";

  switch (stageId) {
    case "symbol-fetch":
      return <Cpu className={iconClass} />;
    case "schematic-assembly":
      return <Layers className={iconClass} />;
    case "wire-routing":
      return <GitBranch className={iconClass} />;
    case "vision-validation":
      return <Eye className={iconClass} />;
    case "component-placement":
      return <Layers className={iconClass} />;
    case "trace-routing":
      return <GitBranch className={iconClass} />;
    case "drc-check":
      return <CheckCircle className={iconClass} />;
    case "optimization":
      return <Zap className={iconClass} />;
    default:
      return <Circle className={iconClass} />;
  }
};

const StatusIcon = ({ status }: { status: MAPOStage["status"] }) => {
  switch (status) {
    case "completed":
      return <CheckCircle className="w-5 h-5 text-green-500" />;
    case "running":
      return <Loader2 className="w-5 h-5 text-primary-500 animate-spin" />;
    case "error":
      return <XCircle className="w-5 h-5 text-red-500" />;
    default:
      return <Circle className="w-5 h-5 text-slate-500" />;
  }
};

// ============================================================================
// Main Component
// ============================================================================

export function MAPOProgress({
  generationId,
  generationType,
  className,
  onComplete,
  onStageUpdate,
  stages: externalStages,
  overallProgress: externalProgress,
}: MAPOProgressProps) {
  const [internalStages, setInternalStages] = useState<MAPOStage[]>(
    generationType === "schematic" ? DEFAULT_SCHEMATIC_STAGES : DEFAULT_PCB_STAGES
  );

  const stages = externalStages || internalStages;

  // Calculate overall progress
  const overallProgress =
    externalProgress ??
    stages.reduce((acc, stage) => acc + stage.progress, 0) / stages.length;

  // Check if all stages are complete
  const isComplete = stages.every(
    (s) => s.status === "completed" || s.status === "error"
  );
  const hasError = stages.some((s) => s.status === "error");

  // Notify on completion
  useEffect(() => {
    if (isComplete) {
      onComplete?.(!hasError);
    }
  }, [isComplete, hasError, onComplete]);

  // Get current stage
  const currentStage = stages.find((s) => s.status === "running");
  const currentStageIndex = currentStage
    ? stages.findIndex((s) => s.id === currentStage.id)
    : -1;

  // Format duration
  const formatDuration = (ms?: number): string => {
    if (!ms) return "";
    const seconds = Math.floor(ms / 1000);
    if (seconds < 60) return `${seconds}s`;
    const minutes = Math.floor(seconds / 60);
    const remainingSeconds = seconds % 60;
    return `${minutes}m ${remainingSeconds}s`;
  };

  return (
    <div className={cn("space-y-4 p-4 bg-surface-primary rounded-lg border border-slate-700", className)}>
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h3 className="font-semibold text-white">
            {generationType === "schematic" ? "Schematic" : "PCB"} Generation
          </h3>
          <p className="text-xs text-slate-500">ID: {generationId.substring(0, 8)}</p>
        </div>
        <div className="text-right">
          <div className="text-2xl font-bold text-primary-400">
            {Math.round(overallProgress)}%
          </div>
          {currentStage && (
            <p className="text-xs text-slate-500">{currentStage.name}</p>
          )}
        </div>
      </div>

      {/* Overall Progress Bar */}
      <Progress.Root
        className="relative h-3 w-full overflow-hidden rounded-full bg-slate-700"
        value={overallProgress}
      >
        <Progress.Indicator
          className={cn(
            "h-full transition-all duration-500",
            hasError
              ? "bg-red-500"
              : isComplete
              ? "bg-green-500"
              : "bg-primary-500"
          )}
          style={{ width: `${overallProgress}%` }}
        />
      </Progress.Root>

      {/* Stage List */}
      <div className="space-y-3">
        {stages.map((stage, index) => (
          <div
            key={stage.id}
            className={cn(
              "flex items-center gap-3 p-2 rounded-lg transition-colors",
              stage.status === "running" && "bg-primary-600/10 border border-primary-600/30"
            )}
          >
            {/* Status Icon */}
            <StatusIcon status={stage.status} />

            {/* Stage Info */}
            <div className="flex-1 min-w-0">
              <div className="flex items-center gap-2">
                <StageIcon stageId={stage.id} />
                <span
                  className={cn(
                    "text-sm font-medium",
                    stage.status === "running"
                      ? "text-primary-400"
                      : stage.status === "completed"
                      ? "text-green-400"
                      : stage.status === "error"
                      ? "text-red-400"
                      : "text-slate-400"
                  )}
                >
                  {stage.name}
                </span>
              </div>

              {/* Stage Message */}
              {stage.message && (
                <p className="text-xs text-slate-500 truncate mt-0.5">{stage.message}</p>
              )}

              {/* Stage Progress (when running) */}
              {stage.status === "running" && (
                <Progress.Root
                  className="relative h-1.5 w-full overflow-hidden rounded-full bg-slate-700 mt-2"
                  value={stage.progress}
                >
                  <Progress.Indicator
                    className="h-full bg-primary-500 transition-all duration-300"
                    style={{ width: `${stage.progress}%` }}
                  />
                </Progress.Root>
              )}
            </div>

            {/* Duration / Progress */}
            <div className="text-right text-xs text-slate-500">
              {stage.status === "completed" && stage.duration
                ? formatDuration(stage.duration)
                : stage.status === "running"
                ? `${stage.progress}%`
                : ""}
            </div>
          </div>
        ))}
      </div>

      {/* Completion Message */}
      {isComplete && (
        <div
          className={cn(
            "flex items-center gap-2 p-3 rounded-lg",
            hasError ? "bg-red-500/10 border border-red-500/30" : "bg-green-500/10 border border-green-500/30"
          )}
        >
          {hasError ? (
            <>
              <XCircle className="w-5 h-5 text-red-500" />
              <span className="text-sm text-red-400">
                Generation failed. Check the error stages above.
              </span>
            </>
          ) : (
            <>
              <CheckCircle className="w-5 h-5 text-green-500" />
              <span className="text-sm text-green-400">
                Generation completed successfully!
              </span>
            </>
          )}
        </div>
      )}
    </div>
  );
}

export default MAPOProgress;
