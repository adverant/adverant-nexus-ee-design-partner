"use client";

/**
 * ChecklistItem - Individual check display component
 *
 * Shows check name, status badge, violation count, duration, auto-fix badge.
 * Expandable to show violation details.
 * Click violation to highlight component in schematic (future integration).
 */

import { useState, useMemo } from "react";
import { cn } from "@/lib/utils";
import {
  CheckCircle2,
  XCircle,
  AlertTriangle,
  Info,
  Clock,
  Loader2,
  ChevronDown,
  ChevronRight,
  Wrench,
  Shield,
  Minus,
  ExternalLink,
} from "lucide-react";
import {
  ChecklistItemProps,
  CheckStatus,
  ViolationSeverity,
} from "@/types/schematic-quality";
import { ViolationDetail } from "./ViolationDetail";

export function ChecklistItem({
  definition,
  result,
  autoFixEnabled,
  onWaiverRequest,
  onViolationClick,
}: ChecklistItemProps) {
  const [isExpanded, setIsExpanded] = useState(false);

  // Get status icon and color
  const statusConfig = useMemo(() => {
    switch (result.status) {
      case CheckStatus.PASSED:
        return {
          icon: <CheckCircle2 className="h-5 w-5" />,
          color: "text-green-600",
          bgColor: "bg-green-50",
          borderColor: "border-green-200",
          label: "Passed",
        };
      case CheckStatus.FAILED:
        return {
          icon: <XCircle className="h-5 w-5" />,
          color: "text-red-600",
          bgColor: "bg-red-50",
          borderColor: "border-red-200",
          label: "Failed",
        };
      case CheckStatus.WARNING:
        return {
          icon: <AlertTriangle className="h-5 w-5" />,
          color: "text-yellow-600",
          bgColor: "bg-yellow-50",
          borderColor: "border-yellow-200",
          label: "Warning",
        };
      case CheckStatus.RUNNING:
        return {
          icon: <Loader2 className="h-5 w-5 animate-spin" />,
          color: "text-blue-600",
          bgColor: "bg-blue-50",
          borderColor: "border-blue-200",
          label: "Running",
        };
      case CheckStatus.PENDING:
        return {
          icon: <Clock className="h-5 w-5" />,
          color: "text-gray-400",
          bgColor: "bg-gray-50",
          borderColor: "border-gray-200",
          label: "Pending",
        };
      case CheckStatus.SKIPPED:
        return {
          icon: <Minus className="h-5 w-5" />,
          color: "text-gray-400",
          bgColor: "bg-gray-50",
          borderColor: "border-gray-200",
          label: "Skipped",
        };
    }
  }, [result.status]);

  // Get severity badge color
  const severityBadgeColor = useMemo(() => {
    switch (definition.severity) {
      case ViolationSeverity.CRITICAL:
        return "bg-red-100 text-red-800 border-red-300";
      case ViolationSeverity.HIGH:
        return "bg-orange-100 text-orange-800 border-orange-300";
      case ViolationSeverity.MEDIUM:
        return "bg-yellow-100 text-yellow-800 border-yellow-300";
      case ViolationSeverity.LOW:
        return "bg-blue-100 text-blue-800 border-blue-300";
      case ViolationSeverity.INFO:
        return "bg-gray-100 text-gray-800 border-gray-300";
    }
  }, [definition.severity]);

  const hasViolations = result.violationCount > 0;
  const hasContent = hasViolations || result.warnings.length > 0 || result.info.length > 0;

  return (
    <div
      className={cn(
        "rounded-lg border transition-all",
        statusConfig.borderColor,
        statusConfig.bgColor,
        hasContent && "hover:shadow-md"
      )}
    >
      {/* Header - Always visible */}
      <button
        onClick={() => hasContent && setIsExpanded(!isExpanded)}
        className={cn(
          "flex w-full items-center gap-3 p-4 text-left transition-colors",
          hasContent && "cursor-pointer hover:bg-white/50"
        )}
      >
        {/* Expand Icon */}
        {hasContent ? (
          isExpanded ? (
            <ChevronDown className="h-5 w-5 text-gray-400" />
          ) : (
            <ChevronRight className="h-5 w-5 text-gray-400" />
          )
        ) : (
          <div className="w-5" />
        )}

        {/* Status Icon */}
        <div className={statusConfig.color}>{statusConfig.icon}</div>

        {/* Check Name and Category */}
        <div className="min-w-0 flex-1">
          <div className="flex items-center gap-2">
            <h4 className="font-semibold text-gray-900">{definition.name}</h4>
            <span className="text-xs text-gray-500">({definition.category})</span>
          </div>
          <p className="text-sm text-gray-600">{definition.description}</p>
        </div>

        {/* Badges and Stats */}
        <div className="flex items-center gap-2">
          {/* Severity Badge */}
          <span
            className={cn(
              "rounded-md border px-2 py-1 text-xs font-semibold uppercase",
              severityBadgeColor
            )}
          >
            {definition.severity}
          </span>

          {/* Auto-fix Badge */}
          {definition.autoFixable && result.autoFixed && (
            <span className="flex items-center gap-1 rounded-md bg-purple-100 px-2 py-1 text-xs font-medium text-purple-800">
              <Wrench className="h-3 w-3" />
              Auto-fixed ({result.fixedCount})
            </span>
          )}

          {/* Violation Count */}
          {hasViolations && (
            <span className="flex items-center gap-1 rounded-md bg-red-100 px-2 py-1 text-xs font-semibold text-red-800">
              <XCircle className="h-3 w-3" />
              {result.violationCount} violation{result.violationCount !== 1 ? "s" : ""}
            </span>
          )}

          {/* Duration */}
          {result.duration !== undefined && (
            <span className="text-xs text-gray-500">
              {result.duration < 1000
                ? `${result.duration}ms`
                : `${(result.duration / 1000).toFixed(1)}s`}
            </span>
          )}
        </div>
      </button>

      {/* Expanded Content */}
      {isExpanded && hasContent && (
        <div className="border-t bg-white p-4">
          {/* Rationale and Remediation */}
          <div className="mb-4 space-y-3">
            <div>
              <h5 className="mb-1 text-sm font-semibold text-gray-900">Why This Matters</h5>
              <p className="text-sm text-gray-700">{definition.rationale}</p>
            </div>

            <div>
              <h5 className="mb-1 text-sm font-semibold text-gray-900">How to Fix</h5>
              <p className="text-sm text-gray-700">{definition.remediation}</p>
            </div>

            {/* Standards References */}
            {definition.references.length > 0 && (
              <div>
                <h5 className="mb-1 text-sm font-semibold text-gray-900">Standards</h5>
                <div className="flex flex-wrap gap-2">
                  {definition.references.map((ref, idx) => (
                    <span
                      key={idx}
                      className="inline-flex items-center gap-1 rounded-md bg-gray-100 px-2 py-1 text-xs font-mono text-gray-700"
                    >
                      <ExternalLink className="h-3 w-3" />
                      {ref}
                    </span>
                  ))}
                </div>
              </div>
            )}
          </div>

          {/* Violations */}
          {result.violations.length > 0 && (
            <div className="mb-4">
              <h5 className="mb-2 text-sm font-semibold text-gray-900">
                Violations ({result.violations.length})
              </h5>
              <div className="space-y-2">
                {result.violations.map((violation, idx) => (
                  <ViolationDetail
                    key={idx}
                    violation={violation}
                    onWaiverRequest={onWaiverRequest ? () => onWaiverRequest(definition.id) : undefined}
                    onViolationClick={onViolationClick}
                  />
                ))}
              </div>
            </div>
          )}

          {/* Warnings */}
          {result.warnings.length > 0 && (
            <div className="mb-4">
              <h5 className="mb-2 flex items-center gap-2 text-sm font-semibold text-yellow-800">
                <AlertTriangle className="h-4 w-4" />
                Warnings ({result.warnings.length})
              </h5>
              <ul className="space-y-1 text-sm text-yellow-800">
                {result.warnings.map((warning, idx) => (
                  <li key={idx} className="rounded-md bg-yellow-50 p-2">
                    {warning}
                  </li>
                ))}
              </ul>
            </div>
          )}

          {/* Info Messages */}
          {result.info.length > 0 && (
            <div>
              <h5 className="mb-2 flex items-center gap-2 text-sm font-semibold text-blue-800">
                <Info className="h-4 w-4" />
                Information ({result.info.length})
              </h5>
              <ul className="space-y-1 text-sm text-blue-800">
                {result.info.map((info, idx) => (
                  <li key={idx} className="rounded-md bg-blue-50 p-2">
                    {info}
                  </li>
                ))}
              </ul>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
