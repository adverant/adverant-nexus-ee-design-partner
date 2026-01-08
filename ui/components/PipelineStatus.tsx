"use client";

import { cn } from "@/lib/utils";
import {
  Lightbulb,
  Building2,
  CircuitBoard,
  Activity,
  Layers,
  Factory,
  Code2,
  TestTube,
  Package,
  HeadphonesIcon,
} from "lucide-react";

const PHASES = [
  { id: "ideation", name: "Ideation", icon: Lightbulb, color: "purple" },
  { id: "architecture", name: "Architecture", icon: Building2, color: "blue" },
  { id: "schematic", name: "Schematic", icon: CircuitBoard, color: "cyan" },
  { id: "simulation", name: "Simulation", icon: Activity, color: "teal" },
  { id: "pcb_layout", name: "PCB Layout", icon: Layers, color: "green" },
  { id: "manufacturing", name: "Manufacturing", icon: Factory, color: "lime" },
  { id: "firmware", name: "Firmware", icon: Code2, color: "yellow" },
  { id: "testing", name: "Testing", icon: TestTube, color: "orange" },
  { id: "production", name: "Production", icon: Package, color: "red" },
  { id: "field_support", name: "Field Support", icon: HeadphonesIcon, color: "pink" },
];

const colorClasses: Record<string, { bg: string; border: string; text: string }> = {
  purple: { bg: "bg-purple-500/20", border: "border-purple-500", text: "text-purple-400" },
  blue: { bg: "bg-blue-500/20", border: "border-blue-500", text: "text-blue-400" },
  cyan: { bg: "bg-cyan-500/20", border: "border-cyan-500", text: "text-cyan-400" },
  teal: { bg: "bg-teal-500/20", border: "border-teal-500", text: "text-teal-400" },
  green: { bg: "bg-green-500/20", border: "border-green-500", text: "text-green-400" },
  lime: { bg: "bg-lime-500/20", border: "border-lime-500", text: "text-lime-400" },
  yellow: { bg: "bg-yellow-500/20", border: "border-yellow-500", text: "text-yellow-400" },
  orange: { bg: "bg-orange-500/20", border: "border-orange-500", text: "text-orange-400" },
  red: { bg: "bg-red-500/20", border: "border-red-500", text: "text-red-400" },
  pink: { bg: "bg-pink-500/20", border: "border-pink-500", text: "text-pink-400" },
};

interface PipelineStatusProps {
  projectId: string | null;
  currentPhase?: string;
  completedPhases?: string[];
}

export function PipelineStatus({
  projectId,
  currentPhase = "ideation",
  completedPhases = [],
}: PipelineStatusProps) {
  if (!projectId) {
    return (
      <div className="h-12 bg-surface-primary border-b border-slate-700 flex items-center justify-center">
        <span className="text-sm text-slate-500">Select or create a project to begin</span>
      </div>
    );
  }

  return (
    <div className="h-12 bg-surface-primary border-b border-slate-700 flex items-center px-4 overflow-x-auto">
      <div className="flex items-center gap-1">
        {PHASES.map((phase, index) => {
          const isCompleted = completedPhases.includes(phase.id);
          const isCurrent = currentPhase === phase.id;
          const colors = colorClasses[phase.color];
          const Icon = phase.icon;

          return (
            <div key={phase.id} className="flex items-center">
              <button
                className={cn(
                  "flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium transition-all",
                  isCurrent && cn(colors.bg, colors.border, colors.text, "border"),
                  isCompleted && "text-green-400",
                  !isCurrent && !isCompleted && "text-slate-500 hover:text-slate-300"
                )}
              >
                <Icon className="w-3.5 h-3.5" />
                <span className="hidden md:inline">{phase.name}</span>
                {isCompleted && (
                  <svg className="w-3 h-3 text-green-400" viewBox="0 0 20 20" fill="currentColor">
                    <path
                      fillRule="evenodd"
                      d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z"
                      clipRule="evenodd"
                    />
                  </svg>
                )}
              </button>

              {index < PHASES.length - 1 && (
                <div
                  className={cn(
                    "w-6 h-0.5 mx-1",
                    isCompleted ? "bg-green-500" : "bg-slate-700"
                  )}
                />
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
}