"use client";

import { useState } from "react";
import * as Tabs from "@radix-ui/react-tabs";
import { cn } from "@/lib/utils";
import {
  CircuitBoard,
  Layers,
  Box,
  Activity,
  Radio,
  Thermometer,
} from "lucide-react";

interface DesignTabsProps {
  projectId: string | null;
}

const TABS = [
  { id: "schematic", label: "Schematic", icon: CircuitBoard },
  { id: "pcb", label: "PCB Layout", icon: Layers },
  { id: "3d", label: "3D Viewer", icon: Box },
  { id: "spice", label: "SPICE", icon: Activity },
  { id: "rf", label: "RF/EMC", icon: Radio },
  { id: "thermal", label: "Thermal", icon: Thermometer },
];

export function DesignTabs({ projectId }: DesignTabsProps) {
  const [activeTab, setActiveTab] = useState("schematic");

  if (!projectId) {
    return (
      <div className="h-full flex items-center justify-center bg-background-primary">
        <div className="text-center">
          <CircuitBoard className="w-12 h-12 text-slate-600 mx-auto mb-3" />
          <p className="text-slate-500">Select a project to view design files</p>
        </div>
      </div>
    );
  }

  return (
    <Tabs.Root
      value={activeTab}
      onValueChange={setActiveTab}
      className="h-full flex flex-col"
    >
      {/* Tab List */}
      <Tabs.List className="flex items-center gap-1 px-2 py-1 bg-surface-primary border-b border-slate-700 overflow-x-auto">
        {TABS.map((tab) => {
          const Icon = tab.icon;
          return (
            <Tabs.Trigger
              key={tab.id}
              value={tab.id}
              className={cn(
                "flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-sm font-medium transition-colors",
                activeTab === tab.id
                  ? "bg-primary-600/20 text-primary-400 border border-primary-600/50"
                  : "text-slate-400 hover:text-white hover:bg-surface-secondary"
              )}
            >
              <Icon className="w-4 h-4" />
              {tab.label}
            </Tabs.Trigger>
          );
        })}
      </Tabs.List>

      {/* Tab Content */}
      <div className="flex-1 overflow-hidden bg-background-primary">
        <Tabs.Content value="schematic" className="h-full">
          <SchematicViewer />
        </Tabs.Content>

        <Tabs.Content value="pcb" className="h-full">
          <PCBViewer />
        </Tabs.Content>

        <Tabs.Content value="3d" className="h-full">
          <ThreeDViewer />
        </Tabs.Content>

        <Tabs.Content value="spice" className="h-full">
          <SPICEViewer />
        </Tabs.Content>

        <Tabs.Content value="rf" className="h-full">
          <RFViewer />
        </Tabs.Content>

        <Tabs.Content value="thermal" className="h-full">
          <ThermalViewer />
        </Tabs.Content>
      </div>
    </Tabs.Root>
  );
}

// Placeholder viewers - would be replaced with actual KiCad WASM, Three.js, etc.

function SchematicViewer() {
  return (
    <div className="h-full flex items-center justify-center">
      <div className="text-center">
        <CircuitBoard className="w-16 h-16 text-cyan-500 mx-auto mb-4" />
        <h3 className="text-lg font-medium text-white mb-2">Schematic Editor</h3>
        <p className="text-sm text-slate-400 max-w-md">
          KiCad WASM schematic editor will render here. Navigate hierarchical sheets,
          place components, and create connections.
        </p>
        <div className="mt-4 flex items-center justify-center gap-4 text-xs text-slate-500">
          <span>3 sheets</span>
          <span>|</span>
          <span>164 components</span>
          <span>|</span>
          <span>ERC: 0 errors</span>
        </div>
      </div>
    </div>
  );
}

function PCBViewer() {
  return (
    <div className="h-full flex items-center justify-center">
      <div className="text-center">
        <Layers className="w-16 h-16 text-green-500 mx-auto mb-4" />
        <h3 className="text-lg font-medium text-white mb-2">PCB Layout Editor</h3>
        <p className="text-sm text-slate-400 max-w-md">
          KiCad WASM PCB editor will render here. View layers, route traces,
          and optimize component placement.
        </p>
        <div className="mt-4 flex items-center justify-center gap-4 text-xs text-slate-500">
          <span>10 layers</span>
          <span>|</span>
          <span>150x100mm</span>
          <span>|</span>
          <span>DRC: 0 errors</span>
        </div>
      </div>
    </div>
  );
}

function ThreeDViewer() {
  return (
    <div className="h-full flex items-center justify-center">
      <div className="text-center">
        <Box className="w-16 h-16 text-purple-500 mx-auto mb-4" />
        <h3 className="text-lg font-medium text-white mb-2">3D Board Viewer</h3>
        <p className="text-sm text-slate-400 max-w-md">
          Three.js 3D viewer will render here. Inspect component placement,
          verify mechanical fit, and export STEP files.
        </p>
      </div>
    </div>
  );
}

function SPICEViewer() {
  return (
    <div className="h-full flex items-center justify-center">
      <div className="text-center">
        <Activity className="w-16 h-16 text-yellow-500 mx-auto mb-4" />
        <h3 className="text-lg font-medium text-white mb-2">SPICE Simulation</h3>
        <p className="text-sm text-slate-400 max-w-md">
          Simulation waveform viewer will render here. DC, AC, transient,
          and Monte Carlo analysis results.
        </p>
        <div className="mt-4 flex items-center justify-center gap-4 text-xs text-slate-500">
          <span>Last run: 2 min ago</span>
          <span>|</span>
          <span>Status: Converged</span>
        </div>
      </div>
    </div>
  );
}

function RFViewer() {
  return (
    <div className="h-full flex items-center justify-center">
      <div className="text-center">
        <Radio className="w-16 h-16 text-orange-500 mx-auto mb-4" />
        <h3 className="text-lg font-medium text-white mb-2">RF/EMC Analysis</h3>
        <p className="text-sm text-slate-400 max-w-md">
          S-parameter plots, field patterns, and EMC compliance results
          will render here.
        </p>
      </div>
    </div>
  );
}

function ThermalViewer() {
  return (
    <div className="h-full flex items-center justify-center">
      <div className="text-center">
        <Thermometer className="w-16 h-16 text-red-500 mx-auto mb-4" />
        <h3 className="text-lg font-medium text-white mb-2">Thermal Analysis</h3>
        <p className="text-sm text-slate-400 max-w-md">
          Thermal FEA results and heat maps will render here. Junction
          temperatures and thermal resistance analysis.
        </p>
        <div className="mt-4 flex items-center justify-center gap-4 text-xs text-slate-500">
          <span>Max Tj: 78°C</span>
          <span>|</span>
          <span>Ambient: 40°C</span>
        </div>
      </div>
    </div>
  );
}