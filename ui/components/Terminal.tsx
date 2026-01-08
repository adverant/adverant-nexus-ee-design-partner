"use client";

import { useEffect, useRef, useState } from "react";
import { Terminal as TerminalIcon, Maximize2, Minimize2 } from "lucide-react";

interface TerminalProps {
  projectId: string | null;
}

export function Terminal({ projectId }: TerminalProps) {
  const terminalRef = useRef<HTMLDivElement>(null);
  const [isMaximized, setIsMaximized] = useState(false);
  const [inputValue, setInputValue] = useState("");
  const [history, setHistory] = useState<Array<{ type: "input" | "output"; content: string }>>([
    {
      type: "output",
      content: `Welcome to EE Design Partner - Claude Code Terminal
Type /help for available commands or start with natural language requirements.

Example commands:
  /ee-design analyze-requirements "200A FOC ESC with triple redundant MCUs"
  /schematic-gen power-stage --mosfets=18 --topology=3phase
  /pcb-layout generate --strategy=thermal --layers=10
  /simulate all --gemini-validate
  /firmware-gen stm32h755 --foc --triple-redundant
`,
    },
  ]);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (!inputValue.trim()) return;

    // Add input to history
    setHistory((prev) => [...prev, { type: "input", content: inputValue }]);

    // Process command (placeholder - would connect to backend)
    const response = processCommand(inputValue);
    setHistory((prev) => [...prev, { type: "output", content: response }]);

    setInputValue("");
  };

  const processCommand = (cmd: string): string => {
    if (cmd.startsWith("/help")) {
      return `Available Skills (40+):

Phase 1 - Ideation:
  /research-paper, /patent-search, /market-analysis, /requirements-gen

Phase 2 - Architecture:
  /ee-architecture, /component-select, /bom-optimize, /power-budget

Phase 3 - Schematic:
  /schematic-gen, /schematic-review, /netlist-gen

Phase 4 - Simulation:
  /simulate-spice, /simulate-thermal, /simulate-si, /simulate-rf, /simulate-emc

Phase 5 - PCB Layout:
  /pcb-layout, /pcb-review, /stackup-design, /via-optimize

Phase 6 - Manufacturing:
  /gerber-gen, /dfm-check, /vendor-quote, /panelize

Phase 7 - Firmware:
  /firmware-gen, /hal-gen, /driver-gen, /rtos-config, /build-setup

Phase 8 - Testing:
  /test-gen, /hil-setup, /test-procedure, /coverage-analysis

Phase 9 - Production:
  /manufacture, /assembly-guide, /quality-check, /traceability

Phase 10 - Field Support:
  /debug-assist, /service-manual, /rma-process, /firmware-update

Type any command with --help for usage details.`;
    }

    if (cmd.startsWith("/ee-design")) {
      return `Analyzing requirements...

Detected Design Parameters:
  - Current Rating: 200A continuous
  - Application: FOC (Field-Oriented Control) ESC
  - Architecture: Triple redundant MCUs
  - Safety Level: High (automotive/aerospace grade)

Recommended Architecture:
  - Primary MCU: STM32H755 (ARM Cortex-M7/M4 dual-core)
  - Safety MCU: Infineon AURIX TC377 (ASIL-D)
  - Gate Driver MCU: STM32G474 (high-speed PWM)
  - MOSFETs: 18x SiC (6 per phase, parallel)
  - Topology: 3-phase with shunt current sensing

Next steps:
  1. Run /ee-architecture to generate block diagram
  2. Run /component-select to find optimal parts
  3. Run /power-budget to validate thermal design`;
    }

    return `Processing: ${cmd}
(Connect to backend API for actual execution)`;
  };

  // Auto-scroll to bottom
  useEffect(() => {
    if (terminalRef.current) {
      terminalRef.current.scrollTop = terminalRef.current.scrollHeight;
    }
  }, [history]);

  return (
    <div className="h-full flex flex-col bg-background-primary border-r border-slate-700">
      {/* Terminal Header */}
      <div className="flex items-center justify-between px-4 py-2 bg-surface-primary border-b border-slate-700">
        <div className="flex items-center gap-2">
          <TerminalIcon className="w-4 h-4 text-primary-400" />
          <span className="text-sm font-medium text-white">Claude Code Terminal</span>
          {projectId && (
            <span className="text-xs text-slate-500 ml-2">
              Project: {projectId.slice(0, 8)}...
            </span>
          )}
        </div>
        <button
          onClick={() => setIsMaximized(!isMaximized)}
          className="p-1 rounded hover:bg-surface-secondary transition-colors text-slate-400 hover:text-white"
        >
          {isMaximized ? (
            <Minimize2 className="w-4 h-4" />
          ) : (
            <Maximize2 className="w-4 h-4" />
          )}
        </button>
      </div>

      {/* Terminal Content */}
      <div
        ref={terminalRef}
        className="flex-1 overflow-auto p-4 font-mono text-sm"
      >
        {history.map((entry, i) => (
          <div key={i} className="mb-2">
            {entry.type === "input" ? (
              <div className="flex gap-2">
                <span className="text-green-400">$</span>
                <span className="text-white">{entry.content}</span>
              </div>
            ) : (
              <pre className="text-slate-300 whitespace-pre-wrap">{entry.content}</pre>
            )}
          </div>
        ))}
      </div>

      {/* Input */}
      <form onSubmit={handleSubmit} className="border-t border-slate-700 p-2">
        <div className="flex items-center gap-2 bg-surface-secondary rounded-lg px-3 py-2">
          <span className="text-green-400 font-mono">$</span>
          <input
            type="text"
            value={inputValue}
            onChange={(e) => setInputValue(e.target.value)}
            placeholder="Enter command or describe your requirements..."
            className="flex-1 bg-transparent border-none outline-none text-white font-mono text-sm placeholder:text-slate-500"
          />
        </div>
      </form>
    </div>
  );
}