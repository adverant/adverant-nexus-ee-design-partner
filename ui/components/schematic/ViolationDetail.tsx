"use client";

/**
 * ViolationDetail - Detailed violation info display
 *
 * Shows violation message, component ref, net name, pin, suggested fix,
 * auto-fix badge, and "Request Waiver" button.
 */

import { useState } from "react";
import { cn } from "@/lib/utils";
import {
  AlertCircle,
  AlertTriangle,
  Info,
  XCircle,
  Wrench,
  MessageSquare,
  MapPin,
  Tag,
  Zap,
  FileText,
} from "lucide-react";
import { ViolationDetail as ViolationDetailType, ViolationSeverity } from "@/types/schematic-quality";

interface ViolationDetailProps {
  violation: ViolationDetailType;
  onWaiverRequest?: (violation: ViolationDetailType) => void;
  onViolationClick?: (violation: ViolationDetailType) => void;
  className?: string;
}

export function ViolationDetail({
  violation,
  onWaiverRequest,
  onViolationClick,
  className,
}: ViolationDetailProps) {
  const [showWaiverDialog, setShowWaiverDialog] = useState(false);

  // Get severity icon and color
  const getSeverityIcon = () => {
    switch (violation.severity) {
      case ViolationSeverity.CRITICAL:
        return <XCircle className="h-5 w-5 text-red-600" />;
      case ViolationSeverity.HIGH:
        return <AlertTriangle className="h-5 w-5 text-orange-600" />;
      case ViolationSeverity.MEDIUM:
        return <AlertTriangle className="h-5 w-5 text-yellow-600" />;
      case ViolationSeverity.LOW:
        return <Info className="h-5 w-5 text-blue-600" />;
      case ViolationSeverity.INFO:
        return <Info className="h-5 w-5 text-gray-500" />;
    }
  };

  const getSeverityBadgeColor = () => {
    switch (violation.severity) {
      case ViolationSeverity.CRITICAL:
        return "bg-red-100 text-red-800 border-red-200";
      case ViolationSeverity.HIGH:
        return "bg-orange-100 text-orange-800 border-orange-200";
      case ViolationSeverity.MEDIUM:
        return "bg-yellow-100 text-yellow-800 border-yellow-200";
      case ViolationSeverity.LOW:
        return "bg-blue-100 text-blue-800 border-blue-200";
      case ViolationSeverity.INFO:
        return "bg-gray-100 text-gray-800 border-gray-200";
    }
  };

  const handleClick = () => {
    if (onViolationClick) {
      onViolationClick(violation);
    }
  };

  return (
    <div
      className={cn(
        "rounded-lg border bg-white p-4 shadow-sm transition-all hover:shadow-md",
        onViolationClick && "cursor-pointer hover:border-blue-300",
        className
      )}
      onClick={handleClick}
    >
      {/* Header: Severity Badge + Auto-fix Badge */}
      <div className="mb-3 flex items-center justify-between">
        <div className="flex items-center gap-2">
          {getSeverityIcon()}
          <span className={cn(
            "rounded-md border px-2 py-1 text-xs font-semibold uppercase",
            getSeverityBadgeColor()
          )}>
            {violation.severity}
          </span>
        </div>

        {violation.autoFixable && (
          <div className="flex items-center gap-1 rounded-md bg-purple-100 px-2 py-1 text-xs font-medium text-purple-800">
            <Wrench className="h-3 w-3" />
            Auto-fixable
          </div>
        )}
      </div>

      {/* Violation Message */}
      <div className="mb-3">
        <p className="text-sm font-medium text-gray-900">{violation.message}</p>
      </div>

      {/* Metadata Grid */}
      <div className="mb-3 grid grid-cols-2 gap-3 text-xs">
        {/* Component Reference */}
        {violation.componentRef && (
          <div className="flex items-center gap-2 text-gray-600">
            <Tag className="h-4 w-4 text-gray-400" />
            <span className="font-medium">Component:</span>
            <span className="font-mono text-gray-900">{violation.componentRef}</span>
          </div>
        )}

        {/* Net Name */}
        {violation.netName && (
          <div className="flex items-center gap-2 text-gray-600">
            <Zap className="h-4 w-4 text-gray-400" />
            <span className="font-medium">Net:</span>
            <span className="font-mono text-gray-900">{violation.netName}</span>
          </div>
        )}

        {/* Pin Number */}
        {violation.pinNumber && (
          <div className="flex items-center gap-2 text-gray-600">
            <MapPin className="h-4 w-4 text-gray-400" />
            <span className="font-medium">Pin:</span>
            <span className="font-mono text-gray-900">{violation.pinNumber}</span>
          </div>
        )}

        {/* Symbol Library */}
        {violation.symbolLib && (
          <div className="flex items-center gap-2 text-gray-600">
            <FileText className="h-4 w-4 text-gray-400" />
            <span className="font-medium">Library:</span>
            <span className="font-mono text-gray-900">{violation.symbolLib}</span>
          </div>
        )}

        {/* Property Name */}
        {violation.propertyName && (
          <div className="col-span-2 flex items-center gap-2 text-gray-600">
            <span className="font-medium">Property:</span>
            <span className="font-mono text-gray-900">{violation.propertyName}</span>
          </div>
        )}

        {/* Expected vs Actual */}
        {(violation.expectedValue || violation.actualValue) && (
          <div className="col-span-2 rounded-md bg-gray-50 p-2">
            {violation.expectedValue && (
              <div className="text-gray-600">
                <span className="font-medium">Expected:</span>{" "}
                <span className="font-mono text-gray-900">{violation.expectedValue}</span>
              </div>
            )}
            {violation.actualValue && (
              <div className="text-gray-600">
                <span className="font-medium">Actual:</span>{" "}
                <span className="font-mono text-red-700">{violation.actualValue}</span>
              </div>
            )}
          </div>
        )}

        {/* File Path + Line Number */}
        {violation.filePath && (
          <div className="col-span-2 text-gray-600">
            <span className="font-medium">File:</span>{" "}
            <span className="font-mono text-gray-900">{violation.filePath}</span>
            {violation.lineNumber && (
              <span className="text-gray-500"> (line {violation.lineNumber})</span>
            )}
          </div>
        )}
      </div>

      {/* Suggested Fix */}
      {violation.suggestedFix && (
        <div className="mb-3 rounded-md bg-blue-50 p-3 text-sm">
          <div className="mb-1 flex items-center gap-2 font-medium text-blue-900">
            <Wrench className="h-4 w-4" />
            Suggested Fix
          </div>
          <p className="text-blue-800">{violation.suggestedFix}</p>
        </div>
      )}

      {/* Actions */}
      <div className="flex items-center justify-end gap-2 border-t pt-3">
        {onWaiverRequest && (
          <button
            onClick={(e) => {
              e.stopPropagation();
              onWaiverRequest(violation);
            }}
            className="flex items-center gap-2 rounded-md border border-gray-300 bg-white px-3 py-1.5 text-sm font-medium text-gray-700 transition-colors hover:bg-gray-50"
          >
            <MessageSquare className="h-4 w-4" />
            Request Waiver
          </button>
        )}

        {violation.autoFixable && (
          <button
            onClick={(e) => {
              e.stopPropagation();
              // Auto-fix action would be handled here
              console.log("Auto-fix requested for violation:", violation);
            }}
            className="flex items-center gap-2 rounded-md bg-purple-600 px-3 py-1.5 text-sm font-medium text-white transition-colors hover:bg-purple-700"
          >
            <Wrench className="h-4 w-4" />
            Apply Auto-fix
          </button>
        )}
      </div>
    </div>
  );
}
