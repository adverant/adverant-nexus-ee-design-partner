"use client";

/**
 * EE Design Partner - Validation Panel Component
 *
 * Displays real-time validation results from the backend API.
 * No mock data - all data comes from Zustand store populated via API/WebSocket.
 */

import { useEffect, useState } from "react";
import { cn } from "@/lib/utils";
import {
  CheckCircle2,
  XCircle,
  AlertTriangle,
  Clock,
  ChevronDown,
  ChevronRight,
  Sparkles,
  Brain,
  Cpu,
  Loader2,
  RefreshCw,
  Download,
  AlertCircle,
} from "lucide-react";
import { useEEDesignStore, type ValidationDomain } from "@/hooks/useEEDesignStore";

interface ValidationPanelProps {
  projectId: string | null;
}

// Validator configuration - defines the multi-LLM validators
const VALIDATORS = [
  { id: "claude", name: "Claude Opus 4", icon: Brain, weight: 0.4 },
  { id: "gemini", name: "Gemini 2.5 Pro", icon: Sparkles, weight: 0.3 },
  { id: "domain", name: "Domain Experts", icon: Cpu, weight: 0.3 },
] as const;

// Status icon component
function StatusIcon({ status }: { status: ValidationDomain["status"] }) {
  switch (status) {
    case "passed":
      return <CheckCircle2 className="w-4 h-4 text-green-400" />;
    case "warning":
      return <AlertTriangle className="w-4 h-4 text-yellow-400" />;
    case "failed":
      return <XCircle className="w-4 h-4 text-red-400" />;
    case "running":
      return <Loader2 className="w-4 h-4 text-primary-400 animate-spin" />;
    case "pending":
    default:
      return <Clock className="w-4 h-4 text-slate-400" />;
  }
}

// Score color based on value
function getScoreColor(score: number): string {
  if (score >= 95) return "text-green-400";
  if (score >= 80) return "text-yellow-400";
  return "text-red-400";
}

// Status badge component
function StatusBadge({ status, score }: { status: string; score: number }) {
  let bgColor: string;
  let textColor: string;
  let label: string;

  if (status === "passed" || score >= 95) {
    bgColor = "bg-green-500/20";
    textColor = "text-green-400";
    label = "Approved";
  } else if (status === "warning" || score >= 80) {
    bgColor = "bg-yellow-500/20";
    textColor = "text-yellow-400";
    label = "Review Required";
  } else {
    bgColor = "bg-red-500/20";
    textColor = "text-red-400";
    label = "Not Ready";
  }

  return (
    <div className={cn("px-3 py-1.5 rounded-lg text-sm font-medium", bgColor, textColor)}>
      {label}
    </div>
  );
}

export function ValidationPanel({ projectId }: ValidationPanelProps) {
  const [expandedDomains, setExpandedDomains] = useState<Set<string>>(new Set());

  // Store state and actions
  const {
    validationResult,
    isValidationRunning,
    loading,
    errors,
    fetchValidationResults,
    runValidation,
  } = useEEDesignStore();

  // Fetch validation results when project changes
  useEffect(() => {
    if (projectId) {
      fetchValidationResults(projectId);
    }
  }, [projectId, fetchValidationResults]);

  // Toggle domain expansion
  const toggleDomain = (id: string) => {
    setExpandedDomains((prev) => {
      const next = new Set(prev);
      if (next.has(id)) {
        next.delete(id);
      } else {
        next.add(id);
      }
      return next;
    });
  };

  // Handle run validation
  const handleRunValidation = () => {
    if (projectId && !isValidationRunning) {
      runValidation(projectId);
    }
  };

  // Handle export report (placeholder for actual implementation)
  const handleExportReport = () => {
    if (!validationResult) return;

    const report = {
      projectId,
      timestamp: new Date().toISOString(),
      overallScore: validationResult.overallScore,
      status: validationResult.overallStatus,
      domains: validationResult.domains,
    };

    const blob = new Blob([JSON.stringify(report, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `validation-report-${projectId?.slice(0, 8)}-${Date.now()}.json`;
    a.click();
    URL.revokeObjectURL(url);
  };

  // Empty state when no project selected
  if (!projectId) {
    return (
      <div className="h-full flex items-center justify-center bg-surface-primary">
        <div className="text-center">
          <CheckCircle2 className="w-12 h-12 text-slate-600 mx-auto mb-3" />
          <p className="text-slate-500">Select a project to view validation results</p>
        </div>
      </div>
    );
  }

  // Loading state
  if (loading.validation && !validationResult) {
    return (
      <div className="h-full flex items-center justify-center bg-surface-primary">
        <div className="text-center">
          <Loader2 className="w-8 h-8 text-primary-400 mx-auto mb-3 animate-spin" />
          <p className="text-slate-400">Loading validation results...</p>
        </div>
      </div>
    );
  }

  // Error state
  if (errors.validation && !validationResult) {
    return (
      <div className="h-full flex items-center justify-center bg-surface-primary">
        <div className="text-center">
          <AlertCircle className="w-12 h-12 text-red-400 mx-auto mb-3" />
          <p className="text-red-400 mb-2">Failed to load validation results</p>
          <p className="text-sm text-slate-500 mb-4">{errors.validation}</p>
          <button
            onClick={() => fetchValidationResults(projectId)}
            className="px-4 py-2 bg-primary-600 hover:bg-primary-500 text-white text-sm rounded-lg transition-colors"
          >
            Retry
          </button>
        </div>
      </div>
    );
  }

  // No validation results yet
  if (!validationResult || !validationResult.domains || validationResult.domains.length === 0) {
    return (
      <div className="h-full flex flex-col bg-surface-primary">
        <div className="px-4 py-3 border-b border-slate-700">
          <h2 className="text-sm font-medium text-white mb-2">Multi-LLM Validation</h2>
          <div className="flex items-center gap-2 mb-3">
            {VALIDATORS.map((v) => {
              const Icon = v.icon;
              return (
                <div
                  key={v.id}
                  className="flex items-center gap-1.5 px-2 py-1 rounded bg-surface-secondary text-xs"
                >
                  <Icon className="w-3.5 h-3.5 text-slate-500" />
                  <span className="text-slate-400">{v.name}</span>
                  <span className="text-slate-600">({(v.weight * 100).toFixed(0)}%)</span>
                </div>
              );
            })}
          </div>
        </div>

        <div className="flex-1 flex items-center justify-center">
          <div className="text-center">
            <Clock className="w-12 h-12 text-slate-600 mx-auto mb-3" />
            <p className="text-slate-500 mb-4">No validation has been run yet</p>
            <button
              onClick={handleRunValidation}
              disabled={isValidationRunning}
              className="px-4 py-2 bg-primary-600 hover:bg-primary-500 disabled:bg-primary-600/50 text-white text-sm rounded-lg transition-colors flex items-center gap-2 mx-auto"
            >
              {isValidationRunning ? (
                <>
                  <Loader2 className="w-4 h-4 animate-spin" />
                  Running...
                </>
              ) : (
                "Run Validation"
              )}
            </button>
          </div>
        </div>
      </div>
    );
  }

  // Calculate totals from real data
  const overallScore = validationResult.overallScore;
  const totalIssues = validationResult.domains.reduce((sum, d) => sum + (d.issues || 0), 0);

  return (
    <div className="h-full flex flex-col bg-surface-primary">
      {/* Header */}
      <div className="px-4 py-3 border-b border-slate-700">
        <h2 className="text-sm font-medium text-white mb-2">Multi-LLM Validation</h2>

        {/* Validator Status */}
        <div className="flex items-center gap-2 mb-3 flex-wrap">
          {VALIDATORS.map((v) => {
            const Icon = v.icon;
            return (
              <div
                key={v.id}
                className="flex items-center gap-1.5 px-2 py-1 rounded bg-surface-secondary text-xs"
              >
                <Icon className="w-3.5 h-3.5 text-primary-400" />
                <span className="text-slate-300">{v.name}</span>
                <span className="text-slate-500">({(v.weight * 100).toFixed(0)}%)</span>
              </div>
            );
          })}
        </div>

        {/* Overall Score */}
        <div className="flex items-center justify-between">
          <div>
            <span className={cn("text-3xl font-bold", getScoreColor(overallScore))}>
              {overallScore.toFixed(1)}
            </span>
            <span className="text-slate-400 text-sm ml-1">/ 100</span>
          </div>
          <StatusBadge status={validationResult.overallStatus} score={overallScore} />
        </div>

        <p className="text-xs text-slate-500 mt-2">
          {totalIssues} issue{totalIssues !== 1 ? "s" : ""} across{" "}
          {validationResult.domains.length} validation domains
        </p>

        {/* Last updated */}
        {validationResult.timestamp && (
          <p className="text-xs text-slate-600 mt-1">
            Last updated: {new Date(validationResult.timestamp).toLocaleString()}
          </p>
        )}
      </div>

      {/* Validation Domains */}
      <div className="flex-1 overflow-auto">
        {validationResult.domains.map((domain) => {
          const isExpanded = expandedDomains.has(domain.id);

          return (
            <div key={domain.id} className="border-b border-slate-700/50">
              <button
                onClick={() => toggleDomain(domain.id)}
                className="w-full flex items-center gap-3 px-4 py-3 hover:bg-surface-secondary transition-colors"
              >
                {isExpanded ? (
                  <ChevronDown className="w-4 h-4 text-slate-500" />
                ) : (
                  <ChevronRight className="w-4 h-4 text-slate-500" />
                )}

                <StatusIcon status={domain.status} />

                <div className="flex-1 text-left">
                  <div className="flex items-center justify-between">
                    <span className="text-sm font-medium text-white">{domain.name}</span>
                    <span className={cn("text-sm font-mono", getScoreColor(domain.score))}>
                      {domain.score.toFixed(0)}%
                    </span>
                  </div>

                  {domain.issues > 0 && (
                    <span className="text-xs text-slate-500">
                      {domain.issues} {domain.issues === 1 ? "issue" : "issues"}
                    </span>
                  )}
                </div>
              </button>

              {isExpanded && (
                <div className="px-4 py-3 bg-background-primary mx-4 mb-3 rounded-lg">
                  {domain.details && (
                    <p className="text-sm text-slate-300 mb-3">{domain.details}</p>
                  )}

                  {/* Validator Score Breakdown (if available) */}
                  {domain.validators && domain.validators.length > 0 && (
                    <div className="space-y-2">
                      {domain.validators.map((v) => (
                        <div key={v.validatorId} className="flex items-center gap-2 text-xs">
                          <span className="text-slate-500 w-24">{v.validatorName}:</span>
                          <div className="flex-1 h-1.5 bg-slate-700 rounded-full overflow-hidden">
                            <div
                              className={cn(
                                "h-full rounded-full",
                                v.score >= 95
                                  ? "bg-green-500"
                                  : v.score >= 80
                                  ? "bg-yellow-500"
                                  : "bg-red-500"
                              )}
                              style={{ width: `${Math.min(100, v.score)}%` }}
                            />
                          </div>
                          <span className="text-slate-400 w-10 text-right">
                            {v.score.toFixed(0)}%
                          </span>
                        </div>
                      ))}
                    </div>
                  )}

                  {/* Fallback if no detailed validators */}
                  {(!domain.validators || domain.validators.length === 0) && (
                    <div className="space-y-2">
                      {VALIDATORS.map((v) => (
                        <div key={v.id} className="flex items-center gap-2 text-xs">
                          <span className="text-slate-500 w-24">{v.name}:</span>
                          <div className="flex-1 h-1.5 bg-slate-700 rounded-full overflow-hidden">
                            <div
                              className={cn(
                                "h-full rounded-full",
                                domain.score >= 95
                                  ? "bg-green-500"
                                  : domain.score >= 80
                                  ? "bg-yellow-500"
                                  : "bg-red-500"
                              )}
                              style={{ width: `${Math.min(100, domain.score)}%` }}
                            />
                          </div>
                          <span className="text-slate-400 w-10 text-right">
                            {domain.score.toFixed(0)}%
                          </span>
                        </div>
                      ))}
                    </div>
                  )}
                </div>
              )}
            </div>
          );
        })}
      </div>

      {/* Footer Actions */}
      <div className="px-4 py-3 border-t border-slate-700">
        <button
          onClick={handleRunValidation}
          disabled={isValidationRunning}
          className={cn(
            "w-full py-2 text-white text-sm font-medium rounded-lg transition-colors flex items-center justify-center gap-2",
            isValidationRunning
              ? "bg-primary-600/50 cursor-not-allowed"
              : "bg-primary-600 hover:bg-primary-500"
          )}
        >
          {isValidationRunning ? (
            <>
              <Loader2 className="w-4 h-4 animate-spin" />
              Running Validation...
            </>
          ) : (
            <>
              <RefreshCw className="w-4 h-4" />
              Run Full Validation
            </>
          )}
        </button>
        <button
          onClick={handleExportReport}
          disabled={!validationResult}
          className="w-full mt-2 py-2 bg-transparent hover:bg-surface-secondary text-slate-400 hover:text-white text-sm font-medium rounded-lg border border-slate-700 transition-colors flex items-center justify-center gap-2 disabled:opacity-50 disabled:cursor-not-allowed"
        >
          <Download className="w-4 h-4" />
          Export Report
        </button>
      </div>
    </div>
  );
}
