"use client";

/**
 * ComplianceScoreCard - Large score display with pass/fail stats
 *
 * Shows overall compliance score (0-100) with color gradient,
 * passed/total checks, violation counts by severity, and loading state.
 */

import { useMemo } from "react";
import { cn } from "@/lib/utils";
import { CheckCircle2, XCircle, AlertTriangle, Info, Loader2 } from "lucide-react";
import { ComplianceScoreCardProps, ViolationSeverity } from "@/types/schematic-quality";

export function ComplianceScoreCard({
  score,
  passedChecks,
  totalChecks,
  violationCount,
  violationsBySeverity,
  isLoading,
}: ComplianceScoreCardProps) {
  // Calculate score color (red → yellow → green)
  const scoreColor = useMemo(() => {
    if (score >= 90) return "text-green-600";
    if (score >= 70) return "text-yellow-600";
    return "text-red-600";
  }, [score]);

  const scoreBgColor = useMemo(() => {
    if (score >= 90) return "bg-green-50 border-green-200";
    if (score >= 70) return "bg-yellow-50 border-yellow-200";
    return "bg-red-50 border-red-200";
  }, [score]);

  const scoreRingColor = useMemo(() => {
    if (score >= 90) return "stroke-green-600";
    if (score >= 70) return "stroke-yellow-600";
    return "stroke-red-600";
  }, [score]);

  // Calculate progress percentage
  const progressPercentage = useMemo(() => {
    if (totalChecks === 0) return 0;
    return Math.round((passedChecks / totalChecks) * 100);
  }, [passedChecks, totalChecks]);

  // SVG circle parameters for progress ring
  const radius = 80;
  const circumference = 2 * Math.PI * radius;
  const strokeDashoffset = circumference - (score / 100) * circumference;

  return (
    <div className={cn(
      "rounded-lg border-2 p-6",
      scoreBgColor,
      "transition-all duration-300"
    )}>
      <div className="flex items-start justify-between gap-6">
        {/* Left: Large Score Display */}
        <div className="flex flex-col items-center justify-center">
          {isLoading ? (
            <div className="flex h-48 w-48 items-center justify-center">
              <Loader2 className="h-16 w-16 animate-spin text-gray-400" />
            </div>
          ) : (
            <div className="relative h-48 w-48">
              {/* Background circle */}
              <svg className="h-full w-full -rotate-90 transform">
                <circle
                  cx="96"
                  cy="96"
                  r={radius}
                  stroke="currentColor"
                  strokeWidth="12"
                  fill="transparent"
                  className="text-gray-200"
                />
                {/* Progress circle */}
                <circle
                  cx="96"
                  cy="96"
                  r={radius}
                  stroke="currentColor"
                  strokeWidth="12"
                  fill="transparent"
                  strokeDasharray={circumference}
                  strokeDashoffset={strokeDashoffset}
                  className={cn(scoreRingColor, "transition-all duration-500")}
                  strokeLinecap="round"
                />
              </svg>
              {/* Score text */}
              <div className="absolute inset-0 flex flex-col items-center justify-center">
                <div className={cn("text-6xl font-bold", scoreColor)}>
                  {Math.round(score)}
                </div>
                <div className="text-sm font-medium text-gray-600">
                  Compliance Score
                </div>
              </div>
            </div>
          )}

          {/* Pass/Fail Summary */}
          <div className="mt-4 text-center">
            <div className="text-2xl font-semibold text-gray-900">
              {passedChecks} / {totalChecks}
            </div>
            <div className="text-sm text-gray-600">
              Checks Passed ({progressPercentage}%)
            </div>
          </div>
        </div>

        {/* Right: Violation Stats */}
        <div className="flex-1 space-y-4">
          <div>
            <h3 className="text-lg font-semibold text-gray-900">
              Validation Summary
            </h3>
            <p className="text-sm text-gray-600">
              {isLoading
                ? "Running compliance checks..."
                : violationCount === 0
                ? "All checks passed! No violations found."
                : `Found ${violationCount} violation${violationCount !== 1 ? "s" : ""} across ${totalChecks - passedChecks} failed checks.`}
            </p>
          </div>

          {/* Violations by Severity */}
          <div className="space-y-2">
            <h4 className="text-sm font-medium text-gray-700">
              Violations by Severity
            </h4>

            {/* Critical */}
            <div className="flex items-center justify-between rounded-md bg-white p-3 shadow-sm">
              <div className="flex items-center gap-2">
                <XCircle className="h-5 w-5 text-red-600" />
                <span className="text-sm font-medium text-gray-700">Critical</span>
              </div>
              <span className={cn(
                "text-lg font-bold",
                violationsBySeverity[ViolationSeverity.CRITICAL] > 0 ? "text-red-600" : "text-gray-400"
              )}>
                {violationsBySeverity[ViolationSeverity.CRITICAL] || 0}
              </span>
            </div>

            {/* High */}
            <div className="flex items-center justify-between rounded-md bg-white p-3 shadow-sm">
              <div className="flex items-center gap-2">
                <AlertTriangle className="h-5 w-5 text-orange-600" />
                <span className="text-sm font-medium text-gray-700">High</span>
              </div>
              <span className={cn(
                "text-lg font-bold",
                violationsBySeverity[ViolationSeverity.HIGH] > 0 ? "text-orange-600" : "text-gray-400"
              )}>
                {violationsBySeverity[ViolationSeverity.HIGH] || 0}
              </span>
            </div>

            {/* Medium */}
            <div className="flex items-center justify-between rounded-md bg-white p-3 shadow-sm">
              <div className="flex items-center gap-2">
                <AlertTriangle className="h-5 w-5 text-yellow-600" />
                <span className="text-sm font-medium text-gray-700">Medium</span>
              </div>
              <span className={cn(
                "text-lg font-bold",
                violationsBySeverity[ViolationSeverity.MEDIUM] > 0 ? "text-yellow-600" : "text-gray-400"
              )}>
                {violationsBySeverity[ViolationSeverity.MEDIUM] || 0}
              </span>
            </div>

            {/* Low */}
            <div className="flex items-center justify-between rounded-md bg-white p-3 shadow-sm">
              <div className="flex items-center gap-2">
                <Info className="h-5 w-5 text-blue-600" />
                <span className="text-sm font-medium text-gray-700">Low</span>
              </div>
              <span className={cn(
                "text-lg font-bold",
                violationsBySeverity[ViolationSeverity.LOW] > 0 ? "text-blue-600" : "text-gray-400"
              )}>
                {violationsBySeverity[ViolationSeverity.LOW] || 0}
              </span>
            </div>

            {/* Info */}
            <div className="flex items-center justify-between rounded-md bg-white p-3 shadow-sm">
              <div className="flex items-center gap-2">
                <Info className="h-5 w-5 text-gray-500" />
                <span className="text-sm font-medium text-gray-700">Info</span>
              </div>
              <span className={cn(
                "text-lg font-bold",
                violationsBySeverity[ViolationSeverity.INFO] > 0 ? "text-gray-700" : "text-gray-400"
              )}>
                {violationsBySeverity[ViolationSeverity.INFO] || 0}
              </span>
            </div>
          </div>

          {/* Status Badge */}
          {!isLoading && (
            <div className="pt-2">
              {score >= 90 ? (
                <div className="flex items-center gap-2 rounded-md bg-green-100 px-3 py-2 text-green-800">
                  <CheckCircle2 className="h-5 w-5" />
                  <span className="text-sm font-semibold">Excellent - Ready for Production</span>
                </div>
              ) : score >= 70 ? (
                <div className="flex items-center gap-2 rounded-md bg-yellow-100 px-3 py-2 text-yellow-800">
                  <AlertTriangle className="h-5 w-5" />
                  <span className="text-sm font-semibold">Good - Minor Issues to Address</span>
                </div>
              ) : (
                <div className="flex items-center gap-2 rounded-md bg-red-100 px-3 py-2 text-red-800">
                  <XCircle className="h-5 w-5" />
                  <span className="text-sm font-semibold">Needs Work - Critical Issues Found</span>
                </div>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
