"use client";

import { useState } from "react";
import { cn } from "@/lib/utils";
import {
  FolderOpen,
  File,
  FileCode,
  FileJson,
  ChevronRight,
  ChevronDown,
  Plus,
  RefreshCw,
  GitBranch,
} from "lucide-react";

interface ProjectBrowserProps {
  activeProject: string | null;
  onProjectSelect: (projectId: string) => void;
}

interface FileNode {
  name: string;
  type: "file" | "folder";
  children?: FileNode[];
  extension?: string;
}

const SAMPLE_PROJECT: FileNode = {
  name: "foc-esc-heavy-lift",
  type: "folder",
  children: [
    {
      name: "hardware",
      type: "folder",
      children: [
        {
          name: "schematic",
          type: "folder",
          children: [
            { name: "main.kicad_sch", type: "file", extension: "kicad_sch" },
            { name: "power-stage.kicad_sch", type: "file", extension: "kicad_sch" },
            { name: "control.kicad_sch", type: "file", extension: "kicad_sch" },
          ],
        },
        {
          name: "pcb",
          type: "folder",
          children: [
            { name: "main.kicad_pcb", type: "file", extension: "kicad_pcb" },
            { name: "stackup.json", type: "file", extension: "json" },
          ],
        },
        {
          name: "bom",
          type: "folder",
          children: [
            { name: "bom.csv", type: "file", extension: "csv" },
            { name: "bom-optimized.csv", type: "file", extension: "csv" },
          ],
        },
      ],
    },
    {
      name: "firmware",
      type: "folder",
      children: [
        {
          name: "src",
          type: "folder",
          children: [
            { name: "main.c", type: "file", extension: "c" },
            { name: "foc.c", type: "file", extension: "c" },
            { name: "hal_config.h", type: "file", extension: "h" },
          ],
        },
        {
          name: "hal",
          type: "folder",
          children: [
            { name: "stm32h755_hal.c", type: "file", extension: "c" },
            { name: "gpio.c", type: "file", extension: "c" },
            { name: "pwm.c", type: "file", extension: "c" },
          ],
        },
        { name: "CMakeLists.txt", type: "file", extension: "txt" },
      ],
    },
    {
      name: "simulation",
      type: "folder",
      children: [
        { name: "spice_results.json", type: "file", extension: "json" },
        { name: "thermal_analysis.json", type: "file", extension: "json" },
      ],
    },
    {
      name: "docs",
      type: "folder",
      children: [
        { name: "requirements.md", type: "file", extension: "md" },
        { name: "architecture.md", type: "file", extension: "md" },
      ],
    },
    { name: "ee-design-config.json", type: "file", extension: "json" },
  ],
};

function getFileIcon(extension?: string) {
  switch (extension) {
    case "json":
      return <FileJson className="w-4 h-4 text-yellow-400" />;
    case "c":
    case "h":
    case "cpp":
    case "ts":
    case "tsx":
      return <FileCode className="w-4 h-4 text-blue-400" />;
    case "kicad_sch":
    case "kicad_pcb":
      return <FileCode className="w-4 h-4 text-green-400" />;
    default:
      return <File className="w-4 h-4 text-slate-400" />;
  }
}

function TreeNode({ node, depth = 0 }: { node: FileNode; depth?: number }) {
  const [isExpanded, setIsExpanded] = useState(depth < 2);

  return (
    <div>
      <button
        onClick={() => node.type === "folder" && setIsExpanded(!isExpanded)}
        className={cn(
          "w-full flex items-center gap-1.5 py-1 px-2 text-sm hover:bg-surface-secondary rounded transition-colors",
          "text-slate-300 hover:text-white"
        )}
        style={{ paddingLeft: `${depth * 12 + 8}px` }}
      >
        {node.type === "folder" ? (
          <>
            {isExpanded ? (
              <ChevronDown className="w-3.5 h-3.5 text-slate-500" />
            ) : (
              <ChevronRight className="w-3.5 h-3.5 text-slate-500" />
            )}
            <FolderOpen className={cn("w-4 h-4", isExpanded ? "text-yellow-400" : "text-slate-400")} />
          </>
        ) : (
          <>
            <span className="w-3.5" />
            {getFileIcon(node.extension)}
          </>
        )}
        <span className="truncate">{node.name}</span>
      </button>

      {node.type === "folder" && isExpanded && node.children && (
        <div>
          {node.children.map((child, i) => (
            <TreeNode key={i} node={child} depth={depth + 1} />
          ))}
        </div>
      )}
    </div>
  );
}

export function ProjectBrowser({ activeProject, onProjectSelect }: ProjectBrowserProps) {
  return (
    <div className="h-full flex flex-col bg-surface-primary border-r border-slate-700">
      {/* Header */}
      <div className="flex items-center justify-between px-3 py-2 border-b border-slate-700">
        <span className="text-sm font-medium text-white">Project</span>
        <div className="flex items-center gap-1">
          <button className="p-1 rounded hover:bg-surface-secondary transition-colors text-slate-400 hover:text-white">
            <Plus className="w-4 h-4" />
          </button>
          <button className="p-1 rounded hover:bg-surface-secondary transition-colors text-slate-400 hover:text-white">
            <RefreshCw className="w-4 h-4" />
          </button>
        </div>
      </div>

      {/* Git Branch */}
      <div className="flex items-center gap-2 px-3 py-2 border-b border-slate-700 text-xs">
        <GitBranch className="w-3.5 h-3.5 text-slate-400" />
        <span className="text-slate-400">main</span>
      </div>

      {/* File Tree */}
      <div className="flex-1 overflow-auto py-2">
        <TreeNode node={SAMPLE_PROJECT} />
      </div>

      {/* Footer */}
      <div className="px-3 py-2 border-t border-slate-700 text-xs text-slate-500">
        164 components | 10 layers
      </div>
    </div>
  );
}