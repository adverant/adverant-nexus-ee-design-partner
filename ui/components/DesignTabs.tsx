"use client";

/**
 * Design Tabs - Main viewer container for schematics, PCB, 3D, and simulations
 *
 * Integrates KiCanvas for schematic/PCB viewing, Three.js for 3D,
 * and specialized panels for simulations and analysis.
 */

import { useState, useEffect, useMemo, useCallback } from "react";
import { useSearchParams, useRouter, usePathname } from "next/navigation";
import * as Tabs from "@radix-ui/react-tabs";
import { cn } from "@/lib/utils";
import {
  CircuitBoard,
  Layers,
  Box,
  Activity,
  Radio,
  Thermometer,
  AlertCircle,
  FileQuestion,
  Loader2,
  ListChecks,
  Package,
  TestTube2,
} from "lucide-react";

// Import real viewer components
import { KiCanvasViewer } from "./viewers/KiCanvasViewer";
import { ThreeDViewer } from "./viewers/ThreeDViewer";
import { MAPOProgress } from "./panels/MAPOProgress";
import { DRCResults, type DRCResult } from "./panels/DRCResults";
import { BOMEditor, type BOMItem } from "./panels/BOMEditor";
import { BenchTesting, type TestPoint } from "./panels/BenchTesting";
import { useEEDesignStore } from "@/hooks/useEEDesignStore";

// ============================================================================
// Types
// ============================================================================

interface DesignTabsProps {
  projectId: string | null;
}

interface FileInfo {
  url: string;
  type: "schematic" | "pcb";
  name: string;
}

// ============================================================================
// Tab Configuration
// ============================================================================

const TABS = [
  { id: "schematic", label: "Schematic", icon: CircuitBoard },
  { id: "pcb", label: "PCB Layout", icon: Layers },
  { id: "3d", label: "3D Viewer", icon: Box },
  { id: "bom", label: "BOM", icon: Package },
  { id: "testing", label: "Testing", icon: TestTube2 },
  { id: "spice", label: "SPICE", icon: Activity },
  { id: "rf", label: "RF/EMC", icon: Radio },
  { id: "thermal", label: "Thermal", icon: Thermometer },
];

// ============================================================================
// API Base URL
// ============================================================================

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:9080";

// ============================================================================
// Placeholder Components (for tabs not yet fully implemented)
// ============================================================================

function PlaceholderViewer({
  icon: Icon,
  title,
  description,
  color,
  stats,
}: {
  icon: React.ElementType;
  title: string;
  description: string;
  color: string;
  stats?: string[];
}) {
  return (
    <div className="h-full flex items-center justify-center">
      <div className="text-center">
        <Icon className={cn("w-16 h-16 mx-auto mb-4", color)} />
        <h3 className="text-lg font-medium text-white mb-2">{title}</h3>
        <p className="text-sm text-slate-400 max-w-md">{description}</p>
        {stats && (
          <div className="mt-4 flex items-center justify-center gap-4 text-xs text-slate-500">
            {stats.map((stat, i) => (
              <span key={i}>
                {stat}
                {i < stats.length - 1 && <span className="ml-4">|</span>}
              </span>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}

// ============================================================================
// Schematic Viewer Tab
// ============================================================================

function SchematicViewer({ projectId }: { projectId: string }) {
  const [schematicUrl, setSchematicUrl] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [generationId, setGenerationId] = useState<string | null>(null);

  // Fetch schematic URL for project
  useEffect(() => {
    async function fetchSchematic() {
      setIsLoading(true);
      setError(null);

      try {
        // Try to get the latest schematic for this project
        const response = await fetch(
          `${API_BASE}/api/v1/projects/${projectId}/schematic/latest`
        );

        if (response.ok) {
          const data = await response.json();
          if (data.success && data.data?.schematicUrl) {
            setSchematicUrl(data.data.schematicUrl);
          } else {
            setSchematicUrl(null);
          }
        } else if (response.status === 404) {
          setSchematicUrl(null);
        } else {
          throw new Error("Failed to fetch schematic");
        }
      } catch (err) {
        setError(err instanceof Error ? err.message : "Failed to load schematic");
      } finally {
        setIsLoading(false);
      }
    }

    fetchSchematic();
  }, [projectId]);

  if (isLoading) {
    return (
      <div className="h-full flex items-center justify-center">
        <div className="text-center">
          <Loader2 className="w-10 h-10 text-primary-500 animate-spin mx-auto mb-3" />
          <p className="text-sm text-slate-400">Loading schematic...</p>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="h-full flex items-center justify-center">
        <div className="text-center">
          <AlertCircle className="w-12 h-12 text-red-500 mx-auto mb-4" />
          <h3 className="text-lg font-medium text-white mb-2">Error Loading Schematic</h3>
          <p className="text-sm text-slate-400">{error}</p>
        </div>
      </div>
    );
  }

  if (!schematicUrl) {
    return (
      <div className="h-full flex items-center justify-center">
        <div className="text-center">
          <FileQuestion className="w-12 h-12 text-slate-600 mx-auto mb-4" />
          <h3 className="text-lg font-medium text-white mb-2">No Schematic Generated</h3>
          <p className="text-sm text-slate-400 max-w-md">
            Use the terminal to generate a schematic with the <code className="text-cyan-400">/schematic-gen</code> command.
          </p>
          {generationId && (
            <div className="mt-6 max-w-md mx-auto">
              <MAPOProgress
                generationId={generationId}
                generationType="schematic"
                onComplete={(success) => {
                  if (success) {
                    // Refresh to get new schematic
                    setGenerationId(null);
                    setIsLoading(true);
                  }
                }}
              />
            </div>
          )}
        </div>
      </div>
    );
  }

  return (
    <KiCanvasViewer
      fileUrl={schematicUrl}
      fileType="schematic"
      className="h-full"
      onLoad={() => console.log("Schematic loaded")}
      onError={(err) => setError(err)}
    />
  );
}

// ============================================================================
// PCB Viewer Tab
// ============================================================================

function PCBViewer({ projectId }: { projectId: string }) {
  const [pcbUrl, setPcbUrl] = useState<string | null>(null);
  const [drcResults, setDrcResults] = useState<DRCResult | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [isDrcLoading, setIsDrcLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [showDrc, setShowDrc] = useState(false);

  useEffect(() => {
    async function fetchPCB() {
      setIsLoading(true);
      setError(null);

      try {
        const response = await fetch(
          `${API_BASE}/api/v1/projects/${projectId}/pcb-layout/latest`
        );

        if (response.ok) {
          const data = await response.json();
          if (data.success && data.data?.pcbUrl) {
            setPcbUrl(data.data.pcbUrl);
            if (data.data?.drcResults) {
              setDrcResults(data.data.drcResults);
            }
          } else {
            setPcbUrl(null);
          }
        } else if (response.status === 404) {
          setPcbUrl(null);
        } else {
          throw new Error("Failed to fetch PCB layout");
        }
      } catch (err) {
        setError(err instanceof Error ? err.message : "Failed to load PCB");
      } finally {
        setIsLoading(false);
      }
    }

    fetchPCB();
  }, [projectId]);

  const handleRunDrc = async () => {
    setIsDrcLoading(true);
    try {
      const response = await fetch(
        `${API_BASE}/api/v1/projects/${projectId}/pcb-layout/drc`,
        { method: "POST" }
      );
      const data = await response.json();
      if (data.success) {
        setDrcResults(data.data);
        setShowDrc(true);
      }
    } catch (err) {
      console.error("DRC failed:", err);
    } finally {
      setIsDrcLoading(false);
    }
  };

  if (isLoading) {
    return (
      <div className="h-full flex items-center justify-center">
        <div className="text-center">
          <Loader2 className="w-10 h-10 text-primary-500 animate-spin mx-auto mb-3" />
          <p className="text-sm text-slate-400">Loading PCB layout...</p>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="h-full flex items-center justify-center">
        <div className="text-center">
          <AlertCircle className="w-12 h-12 text-red-500 mx-auto mb-4" />
          <h3 className="text-lg font-medium text-white mb-2">Error Loading PCB</h3>
          <p className="text-sm text-slate-400">{error}</p>
        </div>
      </div>
    );
  }

  if (!pcbUrl) {
    return (
      <div className="h-full flex items-center justify-center">
        <div className="text-center">
          <Layers className="w-12 h-12 text-slate-600 mx-auto mb-4" />
          <h3 className="text-lg font-medium text-white mb-2">No PCB Layout</h3>
          <p className="text-sm text-slate-400 max-w-md">
            Generate a PCB layout from your schematic using <code className="text-cyan-400">/pcb-layout</code>.
          </p>
        </div>
      </div>
    );
  }

  return (
    <div className="h-full flex">
      {/* PCB Viewer */}
      <div className={cn("flex-1", showDrc && "w-2/3")}>
        <KiCanvasViewer
          fileUrl={pcbUrl}
          fileType="pcb"
          className="h-full"
          onError={(err) => setError(err)}
        />
      </div>

      {/* DRC Panel Toggle */}
      <button
        onClick={() => setShowDrc(!showDrc)}
        className={cn(
          "absolute top-2 left-2 z-20 flex items-center gap-1 px-2 py-1 rounded text-xs",
          drcResults?.passed
            ? "bg-green-500/20 text-green-400"
            : drcResults
            ? "bg-red-500/20 text-red-400"
            : "bg-slate-700 text-slate-300"
        )}
      >
        <ListChecks className="w-4 h-4" />
        DRC {drcResults ? (drcResults.passed ? "✓" : `(${drcResults.totalViolations})`) : ""}
      </button>

      {/* DRC Results Panel */}
      {showDrc && (
        <div className="w-1/3 border-l border-slate-700 overflow-y-auto">
          <DRCResults
            results={drcResults}
            isLoading={isDrcLoading}
            onRerunDRC={handleRunDrc}
            onViolationClick={(v) => {
              // TODO: Navigate to violation location in KiCanvas
              console.log("Navigate to:", v.location);
            }}
          />
        </div>
      )}
    </div>
  );
}

// ============================================================================
// 3D Viewer Tab
// ============================================================================

function ThreeDViewerTab({ projectId }: { projectId: string }) {
  const [modelUrl, setModelUrl] = useState<string | null>(null);
  const [modelFormat, setModelFormat] = useState<"vrml" | "step">("vrml");
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    async function fetchModel() {
      setIsLoading(true);
      setError(null);

      try {
        // Try VRML first (easier to load in browser)
        const vrmlResponse = await fetch(
          `${API_BASE}/api/v1/projects/${projectId}/pcb-layout/latest/board.wrl`
        );

        if (vrmlResponse.ok) {
          const blob = await vrmlResponse.blob();
          const url = URL.createObjectURL(blob);
          setModelUrl(url);
          setModelFormat("vrml");
        } else {
          // Try STEP
          const stepResponse = await fetch(
            `${API_BASE}/api/v1/projects/${projectId}/pcb-layout/latest/board.step`
          );

          if (stepResponse.ok) {
            const blob = await stepResponse.blob();
            const url = URL.createObjectURL(blob);
            setModelUrl(url);
            setModelFormat("step");
          } else {
            setModelUrl(null);
          }
        }
      } catch (err) {
        setError(err instanceof Error ? err.message : "Failed to load 3D model");
      } finally {
        setIsLoading(false);
      }
    }

    fetchModel();

    // Cleanup blob URL on unmount
    return () => {
      if (modelUrl && modelUrl.startsWith("blob:")) {
        URL.revokeObjectURL(modelUrl);
      }
    };
  }, [projectId]);

  if (isLoading) {
    return (
      <div className="h-full flex items-center justify-center">
        <div className="text-center">
          <Loader2 className="w-10 h-10 text-primary-500 animate-spin mx-auto mb-3" />
          <p className="text-sm text-slate-400">Loading 3D model...</p>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="h-full flex items-center justify-center">
        <div className="text-center">
          <AlertCircle className="w-12 h-12 text-red-500 mx-auto mb-4" />
          <h3 className="text-lg font-medium text-white mb-2">Error Loading 3D Model</h3>
          <p className="text-sm text-slate-400">{error}</p>
        </div>
      </div>
    );
  }

  if (!modelUrl) {
    return (
      <div className="h-full flex items-center justify-center">
        <div className="text-center">
          <Box className="w-12 h-12 text-slate-600 mx-auto mb-4" />
          <h3 className="text-lg font-medium text-white mb-2">No 3D Model Available</h3>
          <p className="text-sm text-slate-400 max-w-md">
            Generate a PCB layout first, then export 3D model with <code className="text-cyan-400">/export-3d</code>.
          </p>
        </div>
      </div>
    );
  }

  return (
    <ThreeDViewer
      modelUrl={modelUrl}
      format={modelFormat}
      className="h-full"
      onError={(err) => setError(err)}
    />
  );
}

// ============================================================================
// BOM Editor Tab
// ============================================================================

function BOMEditorTab({ projectId }: { projectId: string }) {
  const [bomItems, setBomItems] = useState<BOMItem[]>([]);
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    async function fetchBOM() {
      setIsLoading(true);
      try {
        const response = await fetch(`${API_BASE}/api/v1/projects/${projectId}/bom`);
        if (response.ok) {
          const data = await response.json();
          if (data.success && data.data?.items) {
            setBomItems(data.data.items);
          }
        }
      } catch (err) {
        console.error("Failed to fetch BOM:", err);
      } finally {
        setIsLoading(false);
      }
    }

    fetchBOM();
  }, [projectId]);

  const handleSave = async (items: BOMItem[]) => {
    await fetch(`${API_BASE}/api/v1/projects/${projectId}/bom`, {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ items }),
    });
  };

  if (isLoading) {
    return (
      <div className="h-full flex items-center justify-center">
        <Loader2 className="w-10 h-10 text-primary-500 animate-spin" />
      </div>
    );
  }

  return (
    <BOMEditor
      projectId={projectId}
      initialItems={bomItems}
      onChange={setBomItems}
      onSave={handleSave}
      className="h-full"
    />
  );
}

// ============================================================================
// Bench Testing Tab
// ============================================================================

function BenchTestingTab({ projectId }: { projectId: string }) {
  const [testPoints, setTestPoints] = useState<TestPoint[]>([]);
  const [pcbLayoutId, setPcbLayoutId] = useState<string>("");
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    async function fetchTestPoints() {
      setIsLoading(true);
      try {
        // Get latest PCB layout ID
        const pcbResponse = await fetch(
          `${API_BASE}/api/v1/projects/${projectId}/pcb-layout/latest`
        );
        if (pcbResponse.ok) {
          const pcbData = await pcbResponse.json();
          if (pcbData.success && pcbData.data?.id) {
            setPcbLayoutId(pcbData.data.id);

            // Get test points for this layout
            const tpResponse = await fetch(
              `${API_BASE}/api/v1/projects/${projectId}/pcb-layout/${pcbData.data.id}/test-points`
            );
            if (tpResponse.ok) {
              const tpData = await tpResponse.json();
              if (tpData.success && tpData.data?.testPoints) {
                setTestPoints(tpData.data.testPoints);
              }
            }
          }
        }
      } catch (err) {
        console.error("Failed to fetch test points:", err);
      } finally {
        setIsLoading(false);
      }
    }

    fetchTestPoints();
  }, [projectId]);

  if (isLoading) {
    return (
      <div className="h-full flex items-center justify-center">
        <Loader2 className="w-10 h-10 text-primary-500 animate-spin" />
      </div>
    );
  }

  if (!pcbLayoutId) {
    return (
      <div className="h-full flex items-center justify-center">
        <div className="text-center">
          <TestTube2 className="w-12 h-12 text-slate-600 mx-auto mb-4" />
          <h3 className="text-lg font-medium text-white mb-2">No PCB Layout</h3>
          <p className="text-sm text-slate-400">
            Generate a PCB layout first to configure bench tests.
          </p>
        </div>
      </div>
    );
  }

  return (
    <BenchTesting
      pcbLayoutId={pcbLayoutId}
      initialTestPoints={testPoints}
      onTestPointsChange={setTestPoints}
      className="h-full"
    />
  );
}

// ============================================================================
// Main Component
// ============================================================================

export function DesignTabs({ projectId }: DesignTabsProps) {
  const searchParams = useSearchParams();
  const router = useRouter();
  const pathname = usePathname();

  // Get initial tab from URL or default to "schematic"
  const urlTab = searchParams.get("tab");
  const [activeTab, setActiveTab] = useState(
    urlTab && TABS.some((t) => t.id === urlTab) ? urlTab : "schematic"
  );

  // Sync tab state with URL
  useEffect(() => {
    if (urlTab && TABS.some((t) => t.id === urlTab) && urlTab !== activeTab) {
      setActiveTab(urlTab);
    }
  }, [urlTab, activeTab]);

  // Update URL when tab changes
  const handleTabChange = useCallback(
    (newTab: string) => {
      setActiveTab(newTab);
      const params = new URLSearchParams(searchParams.toString());
      params.set("tab", newTab);
      if (projectId) {
        params.set("projectId", projectId);
      }
      router.push(`${pathname}?${params.toString()}`, { scroll: false });
    },
    [searchParams, router, pathname, projectId]
  );

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
      onValueChange={handleTabChange}
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
          <SchematicViewer projectId={projectId} />
        </Tabs.Content>

        <Tabs.Content value="pcb" className="h-full relative">
          <PCBViewer projectId={projectId} />
        </Tabs.Content>

        <Tabs.Content value="3d" className="h-full">
          <ThreeDViewerTab projectId={projectId} />
        </Tabs.Content>

        <Tabs.Content value="bom" className="h-full">
          <BOMEditorTab projectId={projectId} />
        </Tabs.Content>

        <Tabs.Content value="testing" className="h-full">
          <BenchTestingTab projectId={projectId} />
        </Tabs.Content>

        <Tabs.Content value="spice" className="h-full">
          <PlaceholderViewer
            icon={Activity}
            title="SPICE Simulation"
            description="Simulation waveform viewer will render here. DC, AC, transient, and Monte Carlo analysis results."
            color="text-yellow-500"
            stats={["Last run: 2 min ago", "Status: Converged"]}
          />
        </Tabs.Content>

        <Tabs.Content value="rf" className="h-full">
          <PlaceholderViewer
            icon={Radio}
            title="RF/EMC Analysis"
            description="S-parameter plots, field patterns, and EMC compliance results will render here."
            color="text-orange-500"
          />
        </Tabs.Content>

        <Tabs.Content value="thermal" className="h-full">
          <PlaceholderViewer
            icon={Thermometer}
            title="Thermal Analysis"
            description="Thermal FEA results and heat maps will render here. Junction temperatures and thermal resistance analysis."
            color="text-red-500"
            stats={["Max Tj: 78°C", "Ambient: 40°C"]}
          />
        </Tabs.Content>
      </div>
    </Tabs.Root>
  );
}
