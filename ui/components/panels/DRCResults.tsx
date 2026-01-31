"use client";

/**
 * DRC Results Panel - Design Rule Check Results Display
 *
 * Displays DRC violations with severity levels and clickable locations
 * for navigating to issues in the PCB viewer.
 */

import { useState, useMemo } from "react";
import { cn } from "@/lib/utils";
import {
  AlertTriangle,
  XCircle,
  CheckCircle,
  ChevronDown,
  ChevronRight,
  ExternalLink,
  Filter,
  RefreshCw,
} from "lucide-react";

// ============================================================================
// Types
// ============================================================================

export interface DRCViolation {
  id: string;
  code: string;
  severity: "error" | "warning";
  message: string;
  location: {
    x?: number;
    y?: number;
    layer?: string;
  };
  items?: string[];
  ruleDescription?: string;
}

export interface DRCResult {
  passed: boolean;
  totalViolations: number;
  violations: DRCViolation[];
  warnings: number;
  timestamp: string;
}

interface DRCResultsProps {
  /** DRC results to display */
  results: DRCResult | null;
  /** Loading state */
  isLoading?: boolean;
  /** Optional class name */
  className?: string;
  /** Callback when a violation is clicked */
  onViolationClick?: (violation: DRCViolation) => void;
  /** Callback to re-run DRC */
  onRerunDRC?: () => void;
}

// ============================================================================
// Helper Components
// ============================================================================

const SeverityIcon = ({ severity }: { severity: DRCViolation["severity"] }) => {
  if (severity === "error") {
    return <XCircle className="w-4 h-4 text-red-500 flex-shrink-0" />;
  }
  return <AlertTriangle className="w-4 h-4 text-yellow-500 flex-shrink-0" />;
};

const CodeBadge = ({ code }: { code: string }) => {
  // Color based on violation type
  const colors: Record<string, string> = {
    CLEARANCE: "bg-red-500/20 text-red-400",
    UNCONNECTED: "bg-orange-500/20 text-orange-400",
    SHORT: "bg-red-600/20 text-red-500",
    OVERLAP: "bg-yellow-500/20 text-yellow-400",
    DRILL: "bg-blue-500/20 text-blue-400",
    VIA: "bg-purple-500/20 text-purple-400",
    TRACK: "bg-cyan-500/20 text-cyan-400",
    PAD: "bg-green-500/20 text-green-400",
    SILKSCREEN: "bg-pink-500/20 text-pink-400",
    SOLDER_MASK: "bg-indigo-500/20 text-indigo-400",
    COURTYARD: "bg-amber-500/20 text-amber-400",
    ZONE: "bg-teal-500/20 text-teal-400",
    NET: "bg-slate-500/20 text-slate-400",
  };

  return (
    <span
      className={cn(
        "px-2 py-0.5 rounded text-xs font-mono",
        colors[code] || "bg-slate-500/20 text-slate-400"
      )}
    >
      {code}
    </span>
  );
};

// ============================================================================
// Violation Item Component
// ============================================================================

interface ViolationItemProps {
  violation: DRCViolation;
  isExpanded: boolean;
  onToggle: () => void;
  onClick: () => void;
}

function ViolationItem({ violation, isExpanded, onToggle, onClick }: ViolationItemProps) {
  return (
    <div
      className={cn(
        "border rounded-lg transition-colors",
        violation.severity === "error"
          ? "border-red-500/30 hover:border-red-500/50"
          : "border-yellow-500/30 hover:border-yellow-500/50"
      )}
    >
      <div
        className="flex items-start gap-2 p-3 cursor-pointer hover:bg-slate-800/50"
        onClick={onClick}
      >
        <SeverityIcon severity={violation.severity} />

        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2 flex-wrap">
            <CodeBadge code={violation.code} />
            {violation.location.layer && (
              <span className="text-xs text-slate-500">
                Layer: {violation.location.layer}
              </span>
            )}
          </div>
          <p className="text-sm text-slate-300 mt-1">{violation.message}</p>

          {violation.location.x !== undefined && violation.location.y !== undefined && (
            <p className="text-xs text-slate-500 mt-1 flex items-center gap-1">
              <ExternalLink className="w-3 h-3" />
              Position: ({violation.location.x.toFixed(2)}, {violation.location.y.toFixed(2)}) mm
            </p>
          )}
        </div>

        <button
          onClick={(e) => {
            e.stopPropagation();
            onToggle();
          }}
          className="p-1 hover:bg-slate-700 rounded"
        >
          {isExpanded ? (
            <ChevronDown className="w-4 h-4 text-slate-400" />
          ) : (
            <ChevronRight className="w-4 h-4 text-slate-400" />
          )}
        </button>
      </div>

      {/* Expanded Details */}
      {isExpanded && (
        <div className="px-3 pb-3 border-t border-slate-700/50 pt-2 text-sm">
          {violation.ruleDescription && (
            <div className="mb-2">
              <span className="text-slate-500">Rule: </span>
              <span className="text-slate-300">{violation.ruleDescription}</span>
            </div>
          )}
          {violation.items && violation.items.length > 0 && (
            <div>
              <span className="text-slate-500">Affected items: </span>
              <div className="mt-1 flex flex-wrap gap-1">
                {violation.items.map((item, i) => (
                  <span
                    key={i}
                    className="px-2 py-0.5 bg-slate-700 rounded text-xs text-slate-300"
                  >
                    {item}
                  </span>
                ))}
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

// ============================================================================
// Main Component
// ============================================================================

export function DRCResults({
  results,
  isLoading,
  className,
  onViolationClick,
  onRerunDRC,
}: DRCResultsProps) {
  const [expandedIds, setExpandedIds] = useState<Set<string>>(new Set());
  const [filterSeverity, setFilterSeverity] = useState<"all" | "error" | "warning">("all");
  const [filterCode, setFilterCode] = useState<string | null>(null);

  // Get unique codes for filter
  const uniqueCodes = useMemo(() => {
    if (!results) return [];
    const codes = new Set(results.violations.map((v) => v.code));
    return Array.from(codes).sort();
  }, [results]);

  // Filter violations
  const filteredViolations = useMemo(() => {
    if (!results) return [];
    return results.violations.filter((v) => {
      if (filterSeverity !== "all" && v.severity !== filterSeverity) return false;
      if (filterCode && v.code !== filterCode) return false;
      return true;
    });
  }, [results, filterSeverity, filterCode]);

  const toggleExpanded = (id: string) => {
    setExpandedIds((prev) => {
      const next = new Set(prev);
      if (next.has(id)) {
        next.delete(id);
      } else {
        next.add(id);
      }
      return next;
    });
  };

  // Loading state
  if (isLoading) {
    return (
      <div className={cn("flex items-center justify-center h-48 bg-surface-primary rounded-lg", className)}>
        <div className="text-center">
          <RefreshCw className="w-8 h-8 text-primary-500 animate-spin mx-auto mb-2" />
          <p className="text-sm text-slate-400">Running DRC...</p>
        </div>
      </div>
    );
  }

  // No results state
  if (!results) {
    return (
      <div className={cn("p-4 bg-surface-primary rounded-lg border border-slate-700", className)}>
        <div className="text-center py-8">
          <AlertTriangle className="w-12 h-12 text-slate-600 mx-auto mb-3" />
          <p className="text-slate-400">No DRC results available</p>
          {onRerunDRC && (
            <button
              onClick={onRerunDRC}
              className="mt-4 px-4 py-2 bg-primary-600 text-white rounded-lg hover:bg-primary-700 transition-colors"
            >
              Run DRC Check
            </button>
          )}
        </div>
      </div>
    );
  }

  return (
    <div className={cn("space-y-4 p-4 bg-surface-primary rounded-lg border border-slate-700", className)}>
      {/* Header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-3">
          {results.passed ? (
            <CheckCircle className="w-8 h-8 text-green-500" />
          ) : (
            <XCircle className="w-8 h-8 text-red-500" />
          )}
          <div>
            <h3 className="font-semibold text-white">
              {results.passed ? "DRC Passed" : `${results.totalViolations} Violations Found`}
            </h3>
            <p className="text-xs text-slate-500">
              {new Date(results.timestamp).toLocaleString()}
            </p>
          </div>
        </div>

        {onRerunDRC && (
          <button
            onClick={onRerunDRC}
            className="flex items-center gap-1 px-3 py-1.5 text-sm bg-slate-700 hover:bg-slate-600 rounded-lg transition-colors"
          >
            <RefreshCw className="w-4 h-4" />
            Re-run
          </button>
        )}
      </div>

      {/* Stats */}
      <div className="grid grid-cols-3 gap-3">
        <div className="p-3 bg-slate-800 rounded-lg text-center">
          <div className="text-2xl font-bold text-red-400">
            {results.violations.filter((v) => v.severity === "error").length}
          </div>
          <div className="text-xs text-slate-500">Errors</div>
        </div>
        <div className="p-3 bg-slate-800 rounded-lg text-center">
          <div className="text-2xl font-bold text-yellow-400">{results.warnings}</div>
          <div className="text-xs text-slate-500">Warnings</div>
        </div>
        <div className="p-3 bg-slate-800 rounded-lg text-center">
          <div className="text-2xl font-bold text-slate-300">{uniqueCodes.length}</div>
          <div className="text-xs text-slate-500">Rule Types</div>
        </div>
      </div>

      {/* Filters */}
      {results.violations.length > 0 && (
        <div className="flex items-center gap-2 flex-wrap">
          <Filter className="w-4 h-4 text-slate-500" />

          {/* Severity filter */}
          <select
            value={filterSeverity}
            onChange={(e) => setFilterSeverity(e.target.value as typeof filterSeverity)}
            className="px-2 py-1 text-sm bg-slate-800 border border-slate-700 rounded text-slate-300"
          >
            <option value="all">All Severities</option>
            <option value="error">Errors Only</option>
            <option value="warning">Warnings Only</option>
          </select>

          {/* Code filter */}
          <select
            value={filterCode || ""}
            onChange={(e) => setFilterCode(e.target.value || null)}
            className="px-2 py-1 text-sm bg-slate-800 border border-slate-700 rounded text-slate-300"
          >
            <option value="">All Codes</option>
            {uniqueCodes.map((code) => (
              <option key={code} value={code}>
                {code}
              </option>
            ))}
          </select>

          {/* Results count */}
          <span className="text-xs text-slate-500 ml-auto">
            Showing {filteredViolations.length} of {results.violations.length}
          </span>
        </div>
      )}

      {/* Violations List */}
      {filteredViolations.length > 0 ? (
        <div className="space-y-2 max-h-96 overflow-y-auto">
          {filteredViolations.map((violation) => (
            <ViolationItem
              key={violation.id}
              violation={violation}
              isExpanded={expandedIds.has(violation.id)}
              onToggle={() => toggleExpanded(violation.id)}
              onClick={() => onViolationClick?.(violation)}
            />
          ))}
        </div>
      ) : results.passed ? (
        <div className="text-center py-8 text-green-400">
          <CheckCircle className="w-12 h-12 mx-auto mb-3" />
          <p>All design rule checks passed!</p>
        </div>
      ) : (
        <div className="text-center py-4 text-slate-400">
          No violations match the current filters.
        </div>
      )}
    </div>
  );
}

export default DRCResults;
