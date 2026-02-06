"use client";

/**
 * SchematicQualityChecklist - Main compliance checklist container
 *
 * Real-time compliance validation UI with WebSocket updates.
 * Shows overall score, filters, checklist items grouped by standard,
 * and action buttons for export/waiver submission.
 */

import { useState, useEffect, useMemo, useCallback } from "react";
import { cn } from "@/lib/utils";
import {
  FileCheck,
  Download,
  Shield,
  Filter as FilterIcon,
  RefreshCw,
  AlertCircle,
} from "lucide-react";
import {
  SchematicQualityChecklistProps,
  ChecklistFilter,
  ChecklistItemResult,
  ChecklistItemDefinition,
  CheckStatus,
  ViolationSeverity,
  ComplianceStandard,
  CheckCategory,
  ChecklistItemStartEvent,
  ChecklistItemPassEvent,
  ChecklistItemFailEvent,
  ChecklistBatchUpdateEvent,
  ComplianceScoreUpdateEvent,
} from "@/types/schematic-quality";
import { DEFAULT_CHECKLIST } from "@/data/default-checklist";
import { ComplianceScoreCard } from "./ComplianceScoreCard";
import { ComplianceFilters } from "./ComplianceFilters";
import { ChecklistItem } from "./ChecklistItem";
import { useWebSocket } from "@/hooks/useWebSocket";

export function SchematicQualityChecklist({
  operationId,
  projectId,
  showFailedOnly = false,
  autoFixEnabled = false,
  onExport,
  onWaiverSubmit,
  className,
}: SchematicQualityChecklistProps) {
  // State
  const [filter, setFilter] = useState<ChecklistFilter>({
    hasViolations: showFailedOnly,
  });
  const [checkResults, setCheckResults] = useState<Map<string, ChecklistItemResult>>(new Map());
  const [overallScore, setOverallScore] = useState(0);
  const [isValidating, setIsValidating] = useState(false);

  // WebSocket connection
  const { socket, isConnected } = useWebSocket({
    autoConnect: true,
  });

  // Initialize check results with pending status
  useEffect(() => {
    const initialResults = new Map<string, ChecklistItemResult>();
    DEFAULT_CHECKLIST.forEach((def) => {
      initialResults.set(def.id, {
        id: def.id,
        status: CheckStatus.PENDING,
        violationCount: 0,
        violations: [],
        warnings: [],
        info: [],
        autoFixed: false,
        fixedCount: 0,
      });
    });
    setCheckResults(initialResults);
  }, []);

  // Subscribe to compliance events via WebSocket
  useEffect(() => {
    if (!socket || !isConnected) return;

    // Subscribe to operation
    socket.emit("subscribe:operation", operationId);

    // Check started
    socket.on("checklist:item:start", (event: ChecklistItemStartEvent) => {
      setCheckResults((prev) => {
        const updated = new Map(prev);
        const existing = updated.get(event.checkId);
        if (existing) {
          updated.set(event.checkId, {
            ...existing,
            status: CheckStatus.RUNNING,
            startedAt: event.timestamp,
          });
        }
        return updated;
      });
      setIsValidating(true);
    });

    // Check passed
    socket.on("checklist:item:pass", (event: ChecklistItemPassEvent) => {
      setCheckResults((prev) => {
        const updated = new Map(prev);
        const existing = updated.get(event.checkId);
        if (existing) {
          updated.set(event.checkId, {
            ...existing,
            status: CheckStatus.PASSED,
            completedAt: event.timestamp,
            duration: event.duration,
            info: event.info,
          });
        }
        return updated;
      });
    });

    // Check failed
    socket.on("checklist:item:fail", (event: ChecklistItemFailEvent) => {
      setCheckResults((prev) => {
        const updated = new Map(prev);
        const existing = updated.get(event.checkId);
        if (existing) {
          updated.set(event.checkId, {
            ...existing,
            status: CheckStatus.FAILED,
            completedAt: event.timestamp,
            duration: event.duration,
            violationCount: event.violationCount,
            violations: event.violations,
            warnings: event.warnings,
            autoFixed: event.autoFixed,
            fixedCount: event.fixedCount,
          });
        }
        return updated;
      });
    });

    // Batch update
    socket.on("checklist:batch:update", (event: ChecklistBatchUpdateEvent) => {
      setCheckResults((prev) => {
        const updated = new Map(prev);
        event.updates.forEach((update) => {
          updated.set(update.id, update);
        });
        return updated;
      });
    });

    // Score update
    socket.on("compliance:score:update", (event: ComplianceScoreUpdateEvent) => {
      setOverallScore(event.score);

      // Check if validation is complete
      if (event.passedChecks + event.totalChecks - event.passedChecks === event.totalChecks) {
        setIsValidating(false);
      }
    });

    // Cleanup
    return () => {
      socket.off("checklist:item:start");
      socket.off("checklist:item:pass");
      socket.off("checklist:item:fail");
      socket.off("checklist:batch:update");
      socket.off("compliance:score:update");
    };
  }, [socket, isConnected, operationId]);

  // Compute summary statistics
  const summary = useMemo(() => {
    let passedChecks = 0;
    let failedChecks = 0;
    let warningsChecks = 0;
    let skippedChecks = 0;
    let totalViolations = 0;
    const violationsBySeverity: Record<ViolationSeverity, number> = {
      [ViolationSeverity.CRITICAL]: 0,
      [ViolationSeverity.HIGH]: 0,
      [ViolationSeverity.MEDIUM]: 0,
      [ViolationSeverity.LOW]: 0,
      [ViolationSeverity.INFO]: 0,
    };

    checkResults.forEach((result) => {
      switch (result.status) {
        case CheckStatus.PASSED:
          passedChecks++;
          break;
        case CheckStatus.FAILED:
          failedChecks++;
          break;
        case CheckStatus.WARNING:
          warningsChecks++;
          break;
        case CheckStatus.SKIPPED:
          skippedChecks++;
          break;
      }

      totalViolations += result.violationCount;
      result.violations.forEach((v) => {
        violationsBySeverity[v.severity]++;
      });
    });

    return {
      passedChecks,
      failedChecks,
      warningsChecks,
      skippedChecks,
      totalChecks: DEFAULT_CHECKLIST.length,
      totalViolations,
      violationsBySeverity,
    };
  }, [checkResults]);

  // Filter checks based on current filter settings
  const filteredChecks = useMemo(() => {
    let filtered = DEFAULT_CHECKLIST;

    // Filter by search text
    if (filter.searchText) {
      const searchLower = filter.searchText.toLowerCase();
      filtered = filtered.filter((def) => {
        const result = checkResults.get(def.id);
        return (
          def.name.toLowerCase().includes(searchLower) ||
          def.description.toLowerCase().includes(searchLower) ||
          def.rationale.toLowerCase().includes(searchLower) ||
          result?.violations.some((v) =>
            v.message.toLowerCase().includes(searchLower) ||
            v.componentRef?.toLowerCase().includes(searchLower) ||
            v.netName?.toLowerCase().includes(searchLower)
          )
        );
      });
    }

    // Filter by status
    if (filter.status && filter.status.length > 0) {
      filtered = filtered.filter((def) => {
        const result = checkResults.get(def.id);
        return result && filter.status!.includes(result.status);
      });
    }

    // Filter by severity
    if (filter.severity && filter.severity.length > 0) {
      filtered = filtered.filter((def) =>
        filter.severity!.includes(def.severity)
      );
    }

    // Filter by category
    if (filter.category && filter.category.length > 0) {
      filtered = filtered.filter((def) =>
        filter.category!.includes(def.category)
      );
    }

    // Filter by standard
    if (filter.standard && filter.standard.length > 0) {
      filtered = filtered.filter((def) =>
        def.standards.some((s) => filter.standard!.includes(s))
      );
    }

    // Filter by violations
    if (filter.hasViolations) {
      filtered = filtered.filter((def) => {
        const result = checkResults.get(def.id);
        return result && result.violationCount > 0;
      });
    }

    // Filter by auto-fixable
    if (filter.autoFixable) {
      filtered = filtered.filter((def) => def.autoFixable);
    }

    return filtered;
  }, [filter, checkResults]);

  // Group checks by standard
  const checksByStandard = useMemo(() => {
    const grouped = new Map<ComplianceStandard, ChecklistItemDefinition[]>();

    filteredChecks.forEach((def) => {
      def.standards.forEach((standard) => {
        if (!grouped.has(standard)) {
          grouped.set(standard, []);
        }
        grouped.get(standard)!.push(def);
      });
    });

    return grouped;
  }, [filteredChecks]);

  // Handlers
  const handleExport = useCallback(() => {
    if (onExport) {
      const report = {
        reportId: `report-${Date.now()}`,
        operationId,
        projectId,
        generatedAt: new Date().toISOString(),
        score: overallScore,
        passed: summary.failedChecks === 0,
        totalChecks: summary.totalChecks,
        passedChecks: summary.passedChecks,
        failedChecks: summary.failedChecks,
        warningsChecks: summary.warningsChecks,
        skippedChecks: summary.skippedChecks,
        totalViolations: summary.totalViolations,
        violationsBySeverity: summary.violationsBySeverity,
        checkResults: Array.from(checkResults.values()),
        standardsCoverage: Object.values(ComplianceStandard),
        autoFixEnabled,
        autoFixedCount: Array.from(checkResults.values()).reduce(
          (sum, r) => sum + r.fixedCount,
          0
        ),
        waivers: [],
      };
      onExport(report);
    }
  }, [onExport, operationId, projectId, overallScore, summary, checkResults, autoFixEnabled]);

  const handleWaiverSubmit = useCallback(
    (checkId: string) => {
      if (onWaiverSubmit) {
        // In a real implementation, this would open a modal for justification
        const justification = prompt("Enter waiver justification:");
        if (justification) {
          onWaiverSubmit(checkId, justification);
        }
      }
    },
    [onWaiverSubmit]
  );

  return (
    <div className={cn("flex flex-col gap-6", className)}>
      {/* Header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-3">
          <Shield className="h-8 w-8 text-blue-600" />
          <div>
            <h2 className="text-2xl font-bold text-gray-900">Schematic Quality Checklist</h2>
            <p className="text-sm text-gray-600">
              {summary.totalChecks} compliance checks across 6 industry standards
            </p>
          </div>
        </div>

        <div className="flex items-center gap-2">
          {isValidating && (
            <div className="flex items-center gap-2 text-sm text-blue-600">
              <RefreshCw className="h-4 w-4 animate-spin" />
              Validating...
            </div>
          )}

          <button
            onClick={handleExport}
            className="flex items-center gap-2 rounded-md bg-blue-600 px-4 py-2 text-sm font-medium text-white transition-colors hover:bg-blue-700"
          >
            <Download className="h-4 w-4" />
            Export Report
          </button>
        </div>
      </div>

      {/* Score Card */}
      <ComplianceScoreCard
        score={overallScore}
        passedChecks={summary.passedChecks}
        totalChecks={summary.totalChecks}
        violationCount={summary.totalViolations}
        violationsBySeverity={summary.violationsBySeverity}
        isLoading={isValidating}
      />

      {/* Filters */}
      <ComplianceFilters filter={filter} onFilterChange={setFilter} />

      {/* Checklist Items by Standard */}
      <div className="space-y-6">
        {Array.from(checksByStandard.entries()).map(([standard, checks]) => {
          const standardPassedCount = checks.filter((def) => {
            const result = checkResults.get(def.id);
            return result?.status === CheckStatus.PASSED;
          }).length;

          const standardViolationCount = checks.reduce((sum, def) => {
            const result = checkResults.get(def.id);
            return sum + (result?.violationCount || 0);
          }, 0);

          return (
            <div key={standard} className="rounded-lg border bg-white shadow-sm">
              {/* Standard Header */}
              <div className="flex items-center justify-between border-b bg-gray-50 p-4">
                <div>
                  <h3 className="text-lg font-semibold text-gray-900">{standard}</h3>
                  <p className="text-sm text-gray-600">
                    {standardPassedCount} / {checks.length} checks passed
                  </p>
                </div>
                {standardViolationCount > 0 && (
                  <div className="flex items-center gap-2 rounded-md bg-red-100 px-3 py-1.5 text-sm font-semibold text-red-800">
                    <AlertCircle className="h-4 w-4" />
                    {standardViolationCount} violation{standardViolationCount !== 1 ? "s" : ""}
                  </div>
                )}
              </div>

              {/* Checks List */}
              <div className="divide-y p-4">
                {checks.map((def) => {
                  const result = checkResults.get(def.id);
                  if (!result) return null;

                  return (
                    <div key={def.id} className="py-2 first:pt-0 last:pb-0">
                      <ChecklistItem
                        definition={def}
                        result={result}
                        autoFixEnabled={autoFixEnabled}
                        onWaiverRequest={handleWaiverSubmit}
                        onViolationClick={(violation) => {
                          // Future: Highlight component in schematic viewer
                          console.log("Violation clicked:", violation);
                        }}
                      />
                    </div>
                  );
                })}
              </div>
            </div>
          );
        })}
      </div>

      {/* Empty State */}
      {filteredChecks.length === 0 && (
        <div className="flex flex-col items-center justify-center rounded-lg border border-dashed bg-gray-50 py-12">
          <FilterIcon className="mb-3 h-12 w-12 text-gray-400" />
          <h3 className="mb-1 text-lg font-semibold text-gray-900">No checks match filters</h3>
          <p className="text-sm text-gray-600">Try adjusting your filter settings</p>
          <button
            onClick={() => setFilter({ searchText: "" })}
            className="mt-4 text-sm text-blue-600 hover:text-blue-700"
          >
            Clear all filters
          </button>
        </div>
      )}

      {/* Footer Stats */}
      <div className="flex items-center justify-between rounded-lg border bg-gray-50 p-4 text-sm">
        <div className="text-gray-600">
          Showing {filteredChecks.length} of {summary.totalChecks} checks
        </div>
        <div className="flex items-center gap-4">
          <div className="flex items-center gap-1">
            <div className="h-3 w-3 rounded-full bg-green-500"></div>
            <span className="text-gray-700">{summary.passedChecks} Passed</span>
          </div>
          <div className="flex items-center gap-1">
            <div className="h-3 w-3 rounded-full bg-red-500"></div>
            <span className="text-gray-700">{summary.failedChecks} Failed</span>
          </div>
          <div className="flex items-center gap-1">
            <div className="h-3 w-3 rounded-full bg-gray-400"></div>
            <span className="text-gray-700">{summary.skippedChecks} Skipped</span>
          </div>
        </div>
      </div>
    </div>
  );
}
