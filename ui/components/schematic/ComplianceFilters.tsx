"use client";

/**
 * ComplianceFilters - Filter controls for checklist
 *
 * Filter by status, severity, category, standard.
 * Search bar for text search.
 * "Show only violations" toggle.
 */

import { useState, useCallback } from "react";
import { cn } from "@/lib/utils";
import {
  Search,
  Filter,
  X,
  ChevronDown,
  CheckCircle2,
  XCircle,
  AlertTriangle,
  Clock,
  Minus,
} from "lucide-react";
import {
  ChecklistFilter,
  CheckStatus,
  ViolationSeverity,
  CheckCategory,
  ComplianceStandard,
} from "@/types/schematic-quality";

interface ComplianceFiltersProps {
  filter: ChecklistFilter;
  onFilterChange: (filter: ChecklistFilter) => void;
  className?: string;
}

export function ComplianceFilters({
  filter,
  onFilterChange,
  className,
}: ComplianceFiltersProps) {
  const [showAdvanced, setShowAdvanced] = useState(false);

  const handleSearchChange = useCallback((searchText: string) => {
    onFilterChange({ ...filter, searchText });
  }, [filter, onFilterChange]);

  const toggleStatus = useCallback((status: CheckStatus) => {
    const currentStatuses = filter.status || [];
    const newStatuses = currentStatuses.includes(status)
      ? currentStatuses.filter(s => s !== status)
      : [...currentStatuses, status];
    onFilterChange({ ...filter, status: newStatuses.length > 0 ? newStatuses : undefined });
  }, [filter, onFilterChange]);

  const toggleSeverity = useCallback((severity: ViolationSeverity) => {
    const currentSeverities = filter.severity || [];
    const newSeverities = currentSeverities.includes(severity)
      ? currentSeverities.filter(s => s !== severity)
      : [...currentSeverities, severity];
    onFilterChange({ ...filter, severity: newSeverities.length > 0 ? newSeverities : undefined });
  }, [filter, onFilterChange]);

  const toggleCategory = useCallback((category: CheckCategory) => {
    const currentCategories = filter.category || [];
    const newCategories = currentCategories.includes(category)
      ? currentCategories.filter(c => c !== category)
      : [...currentCategories, category];
    onFilterChange({ ...filter, category: newCategories.length > 0 ? newCategories : undefined });
  }, [filter, onFilterChange]);

  const toggleStandard = useCallback((standard: ComplianceStandard) => {
    const currentStandards = filter.standard || [];
    const newStandards = currentStandards.includes(standard)
      ? currentStandards.filter(s => s !== standard)
      : [...currentStandards, standard];
    onFilterChange({ ...filter, standard: newStandards.length > 0 ? newStandards : undefined });
  }, [filter, onFilterChange]);

  const toggleViolationsOnly = useCallback(() => {
    onFilterChange({ ...filter, hasViolations: !filter.hasViolations });
  }, [filter, onFilterChange]);

  const toggleAutoFixableOnly = useCallback(() => {
    onFilterChange({ ...filter, autoFixable: !filter.autoFixable });
  }, [filter, onFilterChange]);

  const clearAllFilters = useCallback(() => {
    onFilterChange({ searchText: "" });
  }, [onFilterChange]);

  const activeFilterCount = [
    filter.status?.length,
    filter.severity?.length,
    filter.category?.length,
    filter.standard?.length,
    filter.hasViolations ? 1 : 0,
    filter.autoFixable ? 1 : 0,
  ].reduce<number>((sum, count) => sum + (count || 0), 0);

  return (
    <div className={cn("space-y-3 rounded-lg border bg-white p-4 shadow-sm", className)}>
      {/* Search Bar */}
      <div className="relative">
        <Search className="absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-gray-400" />
        <input
          type="text"
          placeholder="Search checks, violations, components..."
          value={filter.searchText || ""}
          onChange={(e) => handleSearchChange(e.target.value)}
          className="w-full rounded-md border border-gray-300 py-2 pl-10 pr-10 text-sm focus:border-blue-500 focus:outline-none focus:ring-1 focus:ring-blue-500"
        />
        {filter.searchText && (
          <button
            onClick={() => handleSearchChange("")}
            className="absolute right-3 top-1/2 -translate-y-1/2 text-gray-400 hover:text-gray-600"
          >
            <X className="h-4 w-4" />
          </button>
        )}
      </div>

      {/* Quick Toggles */}
      <div className="flex flex-wrap items-center gap-2">
        <button
          onClick={toggleViolationsOnly}
          className={cn(
            "rounded-md border px-3 py-1.5 text-sm font-medium transition-colors",
            filter.hasViolations
              ? "border-red-500 bg-red-50 text-red-700"
              : "border-gray-300 bg-white text-gray-700 hover:bg-gray-50"
          )}
        >
          Show Only Violations
        </button>

        <button
          onClick={toggleAutoFixableOnly}
          className={cn(
            "rounded-md border px-3 py-1.5 text-sm font-medium transition-colors",
            filter.autoFixable
              ? "border-purple-500 bg-purple-50 text-purple-700"
              : "border-gray-300 bg-white text-gray-700 hover:bg-gray-50"
          )}
        >
          Auto-fixable Only
        </button>

        <button
          onClick={() => setShowAdvanced(!showAdvanced)}
          className="ml-auto flex items-center gap-2 rounded-md border border-gray-300 bg-white px-3 py-1.5 text-sm font-medium text-gray-700 transition-colors hover:bg-gray-50"
        >
          <Filter className="h-4 w-4" />
          Advanced Filters
          {activeFilterCount > 0 && (
            <span className="rounded-full bg-blue-600 px-2 py-0.5 text-xs text-white">
              {activeFilterCount}
            </span>
          )}
          <ChevronDown className={cn("h-4 w-4 transition-transform", showAdvanced && "rotate-180")} />
        </button>

        {activeFilterCount > 0 && (
          <button
            onClick={clearAllFilters}
            className="flex items-center gap-1 text-sm text-gray-600 hover:text-gray-900"
          >
            <X className="h-4 w-4" />
            Clear All
          </button>
        )}
      </div>

      {/* Advanced Filters */}
      {showAdvanced && (
        <div className="space-y-4 border-t pt-4">
          {/* Status Filter */}
          <div>
            <label className="mb-2 block text-sm font-medium text-gray-700">Status</label>
            <div className="flex flex-wrap gap-2">
              {Object.values(CheckStatus).map((status) => (
                <button
                  key={status}
                  onClick={() => toggleStatus(status)}
                  className={cn(
                    "flex items-center gap-1.5 rounded-md border px-3 py-1.5 text-sm transition-colors",
                    filter.status?.includes(status)
                      ? "border-blue-500 bg-blue-50 text-blue-700"
                      : "border-gray-300 bg-white text-gray-700 hover:bg-gray-50"
                  )}
                >
                  {status === CheckStatus.PASSED && <CheckCircle2 className="h-3.5 w-3.5" />}
                  {status === CheckStatus.FAILED && <XCircle className="h-3.5 w-3.5" />}
                  {status === CheckStatus.WARNING && <AlertTriangle className="h-3.5 w-3.5" />}
                  {status === CheckStatus.RUNNING && <Clock className="h-3.5 w-3.5" />}
                  {status === CheckStatus.PENDING && <Clock className="h-3.5 w-3.5" />}
                  {status === CheckStatus.SKIPPED && <Minus className="h-3.5 w-3.5" />}
                  <span className="capitalize">{status}</span>
                </button>
              ))}
            </div>
          </div>

          {/* Severity Filter */}
          <div>
            <label className="mb-2 block text-sm font-medium text-gray-700">Severity</label>
            <div className="flex flex-wrap gap-2">
              {Object.values(ViolationSeverity).map((severity) => (
                <button
                  key={severity}
                  onClick={() => toggleSeverity(severity)}
                  className={cn(
                    "rounded-md border px-3 py-1.5 text-sm font-medium uppercase transition-colors",
                    filter.severity?.includes(severity)
                      ? severity === ViolationSeverity.CRITICAL
                        ? "border-red-500 bg-red-50 text-red-700"
                        : severity === ViolationSeverity.HIGH
                        ? "border-orange-500 bg-orange-50 text-orange-700"
                        : severity === ViolationSeverity.MEDIUM
                        ? "border-yellow-500 bg-yellow-50 text-yellow-700"
                        : severity === ViolationSeverity.LOW
                        ? "border-blue-500 bg-blue-50 text-blue-700"
                        : "border-gray-500 bg-gray-50 text-gray-700"
                      : "border-gray-300 bg-white text-gray-700 hover:bg-gray-50"
                  )}
                >
                  {severity}
                </button>
              ))}
            </div>
          </div>

          {/* Category Filter */}
          <div>
            <label className="mb-2 block text-sm font-medium text-gray-700">Category</label>
            <div className="flex flex-wrap gap-2">
              {Object.values(CheckCategory).map((category) => (
                <button
                  key={category}
                  onClick={() => toggleCategory(category)}
                  className={cn(
                    "rounded-md border px-3 py-1.5 text-sm transition-colors",
                    filter.category?.includes(category)
                      ? "border-blue-500 bg-blue-50 text-blue-700"
                      : "border-gray-300 bg-white text-gray-700 hover:bg-gray-50"
                  )}
                >
                  {category}
                </button>
              ))}
            </div>
          </div>

          {/* Standard Filter */}
          <div>
            <label className="mb-2 block text-sm font-medium text-gray-700">Standard</label>
            <div className="flex flex-wrap gap-2">
              {Object.values(ComplianceStandard).map((standard) => (
                <button
                  key={standard}
                  onClick={() => toggleStandard(standard)}
                  className={cn(
                    "rounded-md border px-3 py-1.5 text-sm transition-colors",
                    filter.standard?.includes(standard)
                      ? "border-blue-500 bg-blue-50 text-blue-700"
                      : "border-gray-300 bg-white text-gray-700 hover:bg-gray-50"
                  )}
                >
                  {standard}
                </button>
              ))}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
