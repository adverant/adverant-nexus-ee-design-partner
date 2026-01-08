"use client";

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
} from "lucide-react";
import { useState } from "react";

interface ValidationPanelProps {
  projectId: string | null;
}

interface ValidationDomain {
  id: string;
  name: string;
  status: "passed" | "warning" | "failed" | "pending";
  score: number;
  issues: number;
  details?: string;
}

const VALIDATION_DOMAINS: ValidationDomain[] = [
  { id: "drc", name: "DRC (Design Rules)", status: "passed", score: 100, issues: 0, details: "All design rules pass" },
  { id: "erc", name: "ERC (Electrical Rules)", status: "passed", score: 100, issues: 0, details: "No electrical rule violations" },
  { id: "ipc", name: "IPC-2221 Standards", status: "passed", score: 98, issues: 2, details: "2 minor deviations (approved)" },
  { id: "si", name: "Signal Integrity", status: "passed", score: 95, issues: 0, details: "All high-speed nets verified" },
  { id: "thermal", name: "Thermal Analysis", status: "warning", score: 82, issues: 1, details: "Q3 junction temp at 78°C (limit: 85°C)" },
  { id: "dfm", name: "DFM (Manufacturing)", status: "passed", score: 98.5, issues: 1, details: "1 via-in-pad warning" },
  { id: "best", name: "Best Practices", status: "passed", score: 96, issues: 2, details: "2 recommendations available" },
  { id: "test", name: "Testability", status: "passed", score: 94, issues: 0, details: "All test points accessible" },
];

const VALIDATORS = [
  { id: "claude", name: "Claude Opus 4", icon: Brain, weight: 0.4, status: "active" },
  { id: "gemini", name: "Gemini 2.5 Pro", icon: Sparkles, weight: 0.3, status: "active" },
  { id: "domain", name: "Domain Experts", icon: Cpu, weight: 0.3, status: "active" },
];

function getStatusIcon(status: ValidationDomain["status"]) {
  switch (status) {
    case "passed":
      return <CheckCircle2 className="w-4 h-4 text-green-400" />;
    case "warning":
      return <AlertTriangle className="w-4 h-4 text-yellow-400" />;
    case "failed":
      return <XCircle className="w-4 h-4 text-red-400" />;
    case "pending":
      return <Clock className="w-4 h-4 text-slate-400 animate-pulse" />;
  }
}

export function ValidationPanel({ projectId }: ValidationPanelProps) {
  const [expandedDomains, setExpandedDomains] = useState<Set<string>>(new Set());

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

  const overallScore = VALIDATION_DOMAINS.reduce((sum, d) => sum + d.score, 0) / VALIDATION_DOMAINS.length;
  const totalIssues = VALIDATION_DOMAINS.reduce((sum, d) => sum + d.issues, 0);

  if (!projectId) {
    return (
      <div className="h-full flex items-center justify-center bg-surface-primary">
        <div className="text-center">
          <CheckCircle2 className="w-12 h-12 text-slate-600 mx-auto mb-3" />
          <p className="text-slate-500">Validation results will appear here</p>
        </div>
      </div>
    );
  }

  return (
    <div className="h-full flex flex-col bg-surface-primary">
      {/* Header */}
      <div className="px-4 py-3 border-b border-slate-700">
        <h2 className="text-sm font-medium text-white mb-2">Multi-LLM Validation</h2>

        {/* Validator Status */}
        <div className="flex items-center gap-2 mb-3">
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
            <span className="text-3xl font-bold text-white">{overallScore.toFixed(1)}</span>
            <span className="text-slate-400 text-sm ml-1">/ 100</span>
          </div>
          <div
            className={cn(
              "px-3 py-1.5 rounded-lg text-sm font-medium",
              overallScore >= 95
                ? "bg-green-500/20 text-green-400"
                : overallScore >= 80
                ? "bg-yellow-500/20 text-yellow-400"
                : "bg-red-500/20 text-red-400"
            )}
          >
            {overallScore >= 95 ? "Approved" : overallScore >= 80 ? "Review Required" : "Not Ready"}
          </div>
        </div>

        <p className="text-xs text-slate-500 mt-2">
          {totalIssues} issues across 8 validation domains
        </p>
      </div>

      {/* Validation Domains */}
      <div className="flex-1 overflow-auto">
        {VALIDATION_DOMAINS.map((domain) => {
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

                {getStatusIcon(domain.status)}

                <div className="flex-1 text-left">
                  <div className="flex items-center justify-between">
                    <span className="text-sm font-medium text-white">{domain.name}</span>
                    <span
                      className={cn(
                        "text-sm font-mono",
                        domain.score >= 95
                          ? "text-green-400"
                          : domain.score >= 80
                          ? "text-yellow-400"
                          : "text-red-400"
                      )}
                    >
                      {domain.score}%
                    </span>
                  </div>

                  {domain.issues > 0 && (
                    <span className="text-xs text-slate-500">
                      {domain.issues} {domain.issues === 1 ? "issue" : "issues"}
                    </span>
                  )}
                </div>
              </button>

              {isExpanded && domain.details && (
                <div className="px-4 py-3 bg-background-primary mx-4 mb-3 rounded-lg">
                  <p className="text-sm text-slate-300">{domain.details}</p>

                  {/* Score Breakdown */}
                  <div className="mt-3 space-y-2">
                    {VALIDATORS.map((v) => (
                      <div key={v.id} className="flex items-center gap-2 text-xs">
                        <span className="text-slate-500 w-24">{v.name}:</span>
                        <div className="flex-1 h-1.5 bg-slate-700 rounded-full overflow-hidden">
                          <div
                            className="h-full bg-primary-500 rounded-full"
                            style={{ width: `${domain.score + (Math.random() - 0.5) * 5}%` }}
                          />
                        </div>
                        <span className="text-slate-400 w-10 text-right">
                          {(domain.score + (Math.random() - 0.5) * 5).toFixed(0)}%
                        </span>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>
          );
        })}
      </div>

      {/* Footer Actions */}
      <div className="px-4 py-3 border-t border-slate-700">
        <button className="w-full py-2 bg-primary-600 hover:bg-primary-500 text-white text-sm font-medium rounded-lg transition-colors">
          Run Full Validation
        </button>
        <button className="w-full mt-2 py-2 bg-transparent hover:bg-surface-secondary text-slate-400 hover:text-white text-sm font-medium rounded-lg border border-slate-700 transition-colors">
          Export Report
        </button>
      </div>
    </div>
  );
}