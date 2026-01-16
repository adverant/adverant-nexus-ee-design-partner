"use client";

/**
 * EE Design Partner - Terminal Component
 *
 * Production-ready terminal with real backend integration.
 * Uses Zustand store for state and executes commands via API.
 */

import { useEffect, useRef, useState, KeyboardEvent } from "react";
import { Terminal as TerminalIcon, Maximize2, Minimize2, Loader2, AlertCircle, Wifi, WifiOff } from "lucide-react";
import { cn } from "@/lib/utils";
import { useEEDesignStore, type TerminalEntry } from "@/hooks/useEEDesignStore";
import { useWebSocket } from "@/hooks/useWebSocket";

interface TerminalProps {
  projectId: string | null;
}

export function Terminal({ projectId }: TerminalProps) {
  const terminalRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);
  const [inputValue, setInputValue] = useState("");
  const [historyIndex, setHistoryIndex] = useState(-1);
  const [inputHistory, setInputHistory] = useState<string[]>([]);

  // Store state and actions
  const {
    terminalHistory,
    loading,
    errors,
    isConnected,
    apiError,
    executeCommand,
    checkConnection,
    isTerminalMaximized,
    setTerminalMaximized,
  } = useEEDesignStore();

  // WebSocket connection
  const { isConnected: wsConnected } = useWebSocket({
    autoConnect: true,
    onConnected: () => {
      console.log("[Terminal] WebSocket connected");
    },
    onDisconnected: () => {
      console.log("[Terminal] WebSocket disconnected");
    },
  });

  // Check API connection on mount
  useEffect(() => {
    checkConnection();
  }, [checkConnection]);

  // Auto-scroll to bottom when history changes
  useEffect(() => {
    if (terminalRef.current) {
      terminalRef.current.scrollTop = terminalRef.current.scrollHeight;
    }
  }, [terminalHistory]);

  // Focus input on click
  const handleContainerClick = () => {
    inputRef.current?.focus();
  };

  // Handle form submission
  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!inputValue.trim() || loading.command) return;

    // Add to input history
    setInputHistory((prev) => [...prev.filter((h) => h !== inputValue.trim()), inputValue.trim()]);
    setHistoryIndex(-1);

    // Execute command
    await executeCommand(inputValue);
    setInputValue("");
  };

  // Handle keyboard navigation
  const handleKeyDown = (e: KeyboardEvent<HTMLInputElement>) => {
    if (e.key === "ArrowUp") {
      e.preventDefault();
      if (inputHistory.length > 0) {
        const newIndex = historyIndex < inputHistory.length - 1 ? historyIndex + 1 : historyIndex;
        setHistoryIndex(newIndex);
        setInputValue(inputHistory[inputHistory.length - 1 - newIndex] || "");
      }
    } else if (e.key === "ArrowDown") {
      e.preventDefault();
      if (historyIndex > 0) {
        const newIndex = historyIndex - 1;
        setHistoryIndex(newIndex);
        setInputValue(inputHistory[inputHistory.length - 1 - newIndex] || "");
      } else {
        setHistoryIndex(-1);
        setInputValue("");
      }
    } else if (e.key === "Tab") {
      e.preventDefault();
      // Tab completion could be implemented here
    }
  };

  // Get entry style based on type
  const getEntryStyle = (entry: TerminalEntry) => {
    switch (entry.type) {
      case "input":
        return "text-white";
      case "output":
        return "text-slate-300";
      case "error":
        return "text-red-400";
      case "system":
        return "text-cyan-400";
      default:
        return "text-slate-300";
    }
  };

  // Get entry icon
  const getEntryPrefix = (entry: TerminalEntry) => {
    switch (entry.type) {
      case "input":
        return <span className="text-green-400 mr-2">$</span>;
      case "error":
        return <AlertCircle className="w-4 h-4 text-red-400 mr-2 inline flex-shrink-0" />;
      case "system":
        return <span className="text-cyan-400 mr-2">[SYS]</span>;
      default:
        return null;
    }
  };

  return (
    <div
      className={cn(
        "h-full flex flex-col bg-background-primary border-r border-slate-700",
        isTerminalMaximized && "fixed inset-0 z-50"
      )}
      onClick={handleContainerClick}
    >
      {/* Terminal Header */}
      <div className="flex items-center justify-between px-4 py-2 bg-surface-primary border-b border-slate-700">
        <div className="flex items-center gap-2">
          <TerminalIcon className="w-4 h-4 text-primary-400" />
          <span className="text-sm font-medium text-white">Claude Code Terminal</span>
          {projectId && (
            <span className="text-xs text-slate-500 ml-2">Project: {projectId.slice(0, 8)}...</span>
          )}
        </div>

        <div className="flex items-center gap-2">
          {/* Connection Status */}
          <div
            className={cn(
              "flex items-center gap-1.5 px-2 py-1 rounded text-xs",
              isConnected && wsConnected
                ? "bg-green-500/10 text-green-400"
                : "bg-red-500/10 text-red-400"
            )}
          >
            {isConnected && wsConnected ? (
              <>
                <Wifi className="w-3 h-3" />
                <span>Connected</span>
              </>
            ) : (
              <>
                <WifiOff className="w-3 h-3" />
                <span>Disconnected</span>
              </>
            )}
          </div>

          {/* Loading indicator */}
          {loading.command && (
            <div className="flex items-center gap-1 text-xs text-primary-400">
              <Loader2 className="w-3 h-3 animate-spin" />
              <span>Processing...</span>
            </div>
          )}

          {/* Maximize button */}
          <button
            onClick={(e) => {
              e.stopPropagation();
              setTerminalMaximized(!isTerminalMaximized);
            }}
            className="p-1 rounded hover:bg-surface-secondary transition-colors text-slate-400 hover:text-white"
            aria-label={isTerminalMaximized ? "Minimize terminal" : "Maximize terminal"}
          >
            {isTerminalMaximized ? <Minimize2 className="w-4 h-4" /> : <Maximize2 className="w-4 h-4" />}
          </button>
        </div>
      </div>

      {/* Connection Error Banner */}
      {apiError && (
        <div className="px-4 py-2 bg-red-500/10 border-b border-red-500/20 flex items-center gap-2">
          <AlertCircle className="w-4 h-4 text-red-400 flex-shrink-0" />
          <span className="text-sm text-red-400">{apiError}</span>
          <button
            onClick={() => checkConnection()}
            className="ml-auto text-xs text-red-400 hover:text-red-300 underline"
          >
            Retry
          </button>
        </div>
      )}

      {/* Terminal Content */}
      <div ref={terminalRef} className="flex-1 overflow-auto p-4 font-mono text-sm">
        {terminalHistory.map((entry) => (
          <div key={entry.id} className={cn("mb-2", getEntryStyle(entry))}>
            <div className="flex items-start">
              {getEntryPrefix(entry)}
              <pre className="whitespace-pre-wrap break-words flex-1">{entry.content}</pre>
            </div>
            {entry.status === "pending" && (
              <div className="flex items-center gap-1 mt-1 text-xs text-slate-500">
                <Loader2 className="w-3 h-3 animate-spin" />
                <span>Executing...</span>
              </div>
            )}
          </div>
        ))}

        {/* Command Error */}
        {errors.command && (
          <div className="mb-2 text-red-400 flex items-start gap-2">
            <AlertCircle className="w-4 h-4 flex-shrink-0 mt-0.5" />
            <span>{errors.command}</span>
          </div>
        )}
      </div>

      {/* Input */}
      <form onSubmit={handleSubmit} className="border-t border-slate-700 p-2">
        <div
          className={cn(
            "flex items-center gap-2 bg-surface-secondary rounded-lg px-3 py-2",
            loading.command && "opacity-50"
          )}
        >
          <span className="text-green-400 font-mono">$</span>
          <input
            ref={inputRef}
            type="text"
            value={inputValue}
            onChange={(e) => setInputValue(e.target.value)}
            onKeyDown={handleKeyDown}
            disabled={loading.command}
            placeholder={
              loading.command
                ? "Executing command..."
                : isConnected
                ? "Enter command or describe your requirements..."
                : "Connecting to backend..."
            }
            className="flex-1 bg-transparent border-none outline-none text-white font-mono text-sm placeholder:text-slate-500 disabled:cursor-not-allowed"
            autoFocus
          />
          {loading.command && <Loader2 className="w-4 h-4 text-primary-400 animate-spin" />}
        </div>

        {/* Quick Actions */}
        <div className="flex items-center gap-2 mt-2 px-1">
          <span className="text-xs text-slate-500">Quick:</span>
          <button
            type="button"
            onClick={() => setInputValue("/help")}
            className="text-xs text-slate-400 hover:text-white transition-colors"
            disabled={loading.command}
          >
            /help
          </button>
          <button
            type="button"
            onClick={() => setInputValue("/schematic-gen --help")}
            className="text-xs text-slate-400 hover:text-white transition-colors"
            disabled={loading.command}
          >
            /schematic-gen
          </button>
          <button
            type="button"
            onClick={() => setInputValue("/mapos --help")}
            className="text-xs text-slate-400 hover:text-white transition-colors"
            disabled={loading.command}
          >
            /mapos
          </button>
          <button
            type="button"
            onClick={() => setInputValue("/clear")}
            className="text-xs text-slate-400 hover:text-white transition-colors ml-auto"
            disabled={loading.command}
          >
            Clear
          </button>
        </div>
      </form>
    </div>
  );
}
