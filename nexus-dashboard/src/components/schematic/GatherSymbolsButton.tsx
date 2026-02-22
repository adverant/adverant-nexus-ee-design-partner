"use client";

/**
 * GatherSymbolsButton - Symbol Assembly Pre-Generation UI
 *
 * Triggers the symbol assembly pipeline before schematic generation.
 * Shows real-time progress via WebSocket with assembly-specific events.
 */

import { useState, useEffect, useCallback } from "react";
import { cn } from "@/lib/utils";
import {
  Package,
  Download,
  FileText,
  Search,
  CheckCircle2,
  XCircle,
  Loader2,
  AlertTriangle,
} from "lucide-react";
import { useWebSocket } from "@/hooks/useWebSocket";

export interface GatherSymbolsButtonProps {
  projectId: string;
  ideationArtifacts: Array<{ type: string; content: string }>;
  onComplete?: (report: AssemblyReport) => void;
  onError?: (error: string) => void;
  disabled?: boolean;
  className?: string;
}

export interface AssemblyReport {
  success: boolean;
  total_components: number;
  gathered_components: number;
  symbols_found: number;
  datasheets_downloaded: number;
  characterizations_created: number;
  failed_components: Array<{
    part_number: string;
    manufacturer?: string;
    error: string;
  }>;
  operation_id: string;
}

export interface AssemblyProgress {
  phase: string;
  current_component?: string;
  progress_pct: number;
  message: string;
  source?: 'graphrag' | 'kicad' | 'snapeda' | 'ultralibrarian' | 'datasheet' | 'llm';
}

export function GatherSymbolsButton({
  projectId,
  ideationArtifacts,
  onComplete,
  onError,
  disabled = false,
  className,
}: GatherSymbolsButtonProps) {
  const [isGathering, setIsGathering] = useState(false);
  const [operationId, setOperationId] = useState<string | null>(null);
  const [progress, setProgress] = useState<AssemblyProgress | null>(null);
  const [report, setReport] = useState<AssemblyReport | null>(null);
  const [error, setError] = useState<string | null>(null);

  // WebSocket connection for real-time progress
  const { isConnected, lastMessage } = useWebSocket(
    `ws://localhost:3001/ws/projects/${projectId}`,
    {
      enabled: isGathering && operationId !== null,
    }
  );

  // Handle WebSocket progress events
  useEffect(() => {
    if (!lastMessage) return;

    try {
      const event = JSON.parse(lastMessage);

      // Symbol assembly progress events
      if (event.type === 'symbol_assembly_start') {
        setProgress({
          phase: 'analyze',
          progress_pct: 0,
          message: 'Analyzing ideation artifacts...',
        });
      } else if (event.type === 'symbol_assembly_search') {
        setProgress({
          phase: 'search',
          current_component: event.component,
          progress_pct: event.progress_pct || 0,
          message: `Searching ${event.source || 'libraries'} for ${event.component}`,
          source: event.source,
        });
      } else if (event.type === 'symbol_assembly_found') {
        setProgress(prev => ({
          ...prev!,
          message: `Found ${event.component} in ${event.source}`,
          source: event.source,
        }));
      } else if (event.type === 'symbol_assembly_generated') {
        setProgress(prev => ({
          ...prev!,
          message: `Generated symbol for ${event.component}`,
        }));
      } else if (event.type === 'symbol_assembly_datasheet') {
        setProgress(prev => ({
          ...prev!,
          message: `Downloaded datasheet for ${event.component}`,
        }));
      } else if (event.type === 'symbol_assembly_complete') {
        setProgress({
          phase: 'complete',
          progress_pct: 100,
          message: 'Symbol assembly complete',
        });
        setIsGathering(false);

        // Fetch final report
        fetchAssemblyReport(operationId!);
      } else if (event.type === 'symbol_assembly_error') {
        setError(event.message || 'Symbol assembly failed');
        setIsGathering(false);
        onError?.(event.message);
      }
    } catch (err) {
      console.error('Failed to parse WebSocket message:', err);
    }
  }, [lastMessage, operationId, onError]);

  // Fetch final assembly report
  const fetchAssemblyReport = useCallback(async (opId: string) => {
    try {
      const response = await fetch(
        `/api/v1/projects/${projectId}/symbol-assembly/${opId}`
      );
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }
      const data = await response.json();
      setReport(data);
      onComplete?.(data);
    } catch (err) {
      console.error('Failed to fetch assembly report:', err);
      setError('Failed to retrieve assembly report');
    }
  }, [projectId, onComplete]);

  // Start symbol assembly
  const handleGatherSymbols = useCallback(async () => {
    if (!ideationArtifacts || ideationArtifacts.length === 0) {
      setError('No ideation artifacts provided');
      return;
    }

    setIsGathering(true);
    setError(null);
    setReport(null);
    setProgress(null);

    try {
      const response = await fetch(
        `/api/v1/projects/${projectId}/symbol-assembly/gather`,
        {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            ideationArtifacts,
          }),
        }
      );

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.error || `HTTP ${response.status}`);
      }

      const data = await response.json();
      setOperationId(data.operationId);
    } catch (err) {
      console.error('Failed to start symbol assembly:', err);
      setError(err instanceof Error ? err.message : 'Failed to start symbol assembly');
      setIsGathering(false);
      onError?.(err instanceof Error ? err.message : 'Unknown error');
    }
  }, [projectId, ideationArtifacts, onError]);

  // Render progress indicator
  const renderProgress = () => {
    if (!progress) return null;

    const getSourceIcon = () => {
      switch (progress.source) {
        case 'graphrag': return <Search className="w-4 h-4" />;
        case 'kicad': return <Package className="w-4 h-4" />;
        case 'snapeda': return <Download className="w-4 h-4" />;
        case 'ultralibrarian': return <Download className="w-4 h-4" />;
        case 'datasheet': return <FileText className="w-4 h-4" />;
        case 'llm': return <AlertTriangle className="w-4 h-4 text-yellow-500" />;
        default: return <Loader2 className="w-4 h-4 animate-spin" />;
      }
    };

    return (
      <div className="mt-4 p-4 bg-gray-50 rounded-lg border border-gray-200">
        <div className="flex items-center gap-3 mb-2">
          {getSourceIcon()}
          <span className="text-sm font-medium text-gray-700">{progress.message}</span>
        </div>

        {progress.current_component && (
          <div className="text-xs text-gray-500 mb-2">
            Component: {progress.current_component}
          </div>
        )}

        {/* Progress bar */}
        <div className="w-full bg-gray-200 rounded-full h-2 overflow-hidden">
          <div
            className="bg-blue-600 h-full transition-all duration-300"
            style={{ width: `${progress.progress_pct}%` }}
          />
        </div>
        <div className="text-xs text-gray-500 mt-1 text-right">
          {progress.progress_pct}%
        </div>
      </div>
    );
  };

  // Render completion report
  const renderReport = () => {
    if (!report) return null;

    return (
      <div className="mt-4 p-4 bg-green-50 rounded-lg border border-green-200">
        <div className="flex items-center gap-2 mb-3">
          <CheckCircle2 className="w-5 h-5 text-green-600" />
          <span className="font-semibold text-green-800">
            Symbol Assembly Complete
          </span>
        </div>

        <div className="grid grid-cols-2 gap-3 text-sm">
          <div>
            <span className="text-gray-600">Components:</span>
            <span className="ml-2 font-medium">{report.total_components}</span>
          </div>
          <div>
            <span className="text-gray-600">Gathered:</span>
            <span className="ml-2 font-medium text-green-600">
              {report.gathered_components}
            </span>
          </div>
          <div>
            <span className="text-gray-600">Symbols:</span>
            <span className="ml-2 font-medium">{report.symbols_found}</span>
          </div>
          <div>
            <span className="text-gray-600">Datasheets:</span>
            <span className="ml-2 font-medium">{report.datasheets_downloaded}</span>
          </div>
        </div>

        {report.failed_components && report.failed_components.length > 0 && (
          <div className="mt-3 pt-3 border-t border-green-300">
            <div className="flex items-center gap-2 mb-2">
              <AlertTriangle className="w-4 h-4 text-yellow-600" />
              <span className="text-sm font-medium text-yellow-800">
                {report.failed_components.length} component(s) failed
              </span>
            </div>
            <div className="text-xs text-gray-600 space-y-1">
              {report.failed_components.slice(0, 3).map((comp, idx) => (
                <div key={idx}>
                  {comp.part_number}: {comp.error}
                </div>
              ))}
              {report.failed_components.length > 3 && (
                <div className="text-gray-500">
                  ... and {report.failed_components.length - 3} more
                </div>
              )}
            </div>
          </div>
        )}
      </div>
    );
  };

  // Render error
  const renderError = () => {
    if (!error) return null;

    return (
      <div className="mt-4 p-4 bg-red-50 rounded-lg border border-red-200">
        <div className="flex items-center gap-2">
          <XCircle className="w-5 h-5 text-red-600" />
          <span className="text-sm font-medium text-red-800">{error}</span>
        </div>
      </div>
    );
  };

  return (
    <div className={cn("space-y-2", className)}>
      <button
        onClick={handleGatherSymbols}
        disabled={disabled || isGathering || !ideationArtifacts || ideationArtifacts.length === 0}
        className={cn(
          "flex items-center gap-2 px-4 py-2 rounded-md font-medium transition-colors",
          "bg-blue-600 text-white hover:bg-blue-700",
          "disabled:opacity-50 disabled:cursor-not-allowed",
          isGathering && "animate-pulse"
        )}
      >
        {isGathering ? (
          <>
            <Loader2 className="w-4 h-4 animate-spin" />
            Gathering Symbols...
          </>
        ) : (
          <>
            <Package className="w-4 h-4" />
            Gather Symbols & Datasheets
          </>
        )}
      </button>

      {renderProgress()}
      {renderReport()}
      {renderError()}

      {/* WebSocket connection status (debug) */}
      {process.env.NODE_ENV === 'development' && isGathering && (
        <div className="text-xs text-gray-400">
          WS: {isConnected ? '🟢 Connected' : '🔴 Disconnected'}
        </div>
      )}
    </div>
  );
}
