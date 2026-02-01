"use client";

import { useState, useEffect, useCallback } from "react";
import { useSearchParams, useRouter, usePathname } from "next/navigation";
import { ResizablePanelGroup, ResizablePanel, ResizableHandle } from "@/components/ui/resizable";
import { Terminal } from "@/components/Terminal";
import { ProjectBrowser } from "@/components/ProjectBrowser";
import { DesignTabs } from "@/components/DesignTabs";
import { ValidationPanel } from "@/components/ValidationPanel";
import { PipelineStatus } from "@/components/PipelineStatus";
import { Header } from "@/components/Header";

export default function EEDesignPartner() {
  const searchParams = useSearchParams();
  const router = useRouter();
  const pathname = usePathname();

  // Get initial project from URL or null
  const urlProjectId = searchParams.get("projectId");
  const [activeProject, setActiveProject] = useState<string | null>(urlProjectId);

  // Sync project state with URL on initial load and URL changes
  useEffect(() => {
    if (urlProjectId && urlProjectId !== activeProject) {
      setActiveProject(urlProjectId);
    }
  }, [urlProjectId, activeProject]);

  // Update URL when project changes
  const handleProjectSelect = useCallback(
    (projectId: string | null) => {
      setActiveProject(projectId);

      // Update URL with new projectId
      const params = new URLSearchParams(searchParams.toString());
      if (projectId) {
        params.set("projectId", projectId);
      } else {
        params.delete("projectId");
      }
      router.push(`${pathname}?${params.toString()}`, { scroll: false });
    },
    [searchParams, router, pathname]
  );

  return (
    <div className="h-screen flex flex-col bg-background-primary">
      {/* Header */}
      <Header />

      {/* Pipeline Progress */}
      <PipelineStatus projectId={activeProject} />

      {/* Main Content */}
      <div className="flex-1 overflow-hidden">
        <ResizablePanelGroup direction="horizontal" className="h-full">
          {/* Left Panel - Project Browser */}
          <ResizablePanel defaultSize={15} minSize={10} maxSize={25}>
            <ProjectBrowser
              onProjectSelect={handleProjectSelect}
              activeProject={activeProject}
            />
          </ResizablePanel>

          <ResizableHandle className="w-1 bg-slate-700 hover:bg-primary-500 transition-colors" />

          {/* Center Panel - Terminal + Design Tabs */}
          <ResizablePanel defaultSize={55} minSize={40}>
            <ResizablePanelGroup direction="vertical" className="h-full">
              {/* Terminal (Central Element) */}
              <ResizablePanel defaultSize={40} minSize={20}>
                <Terminal projectId={activeProject} />
              </ResizablePanel>

              <ResizableHandle className="h-1 bg-slate-700 hover:bg-primary-500 transition-colors" />

              {/* Design Tabs */}
              <ResizablePanel defaultSize={60} minSize={30}>
                <DesignTabs projectId={activeProject} />
              </ResizablePanel>
            </ResizablePanelGroup>
          </ResizablePanel>

          <ResizableHandle className="w-1 bg-slate-700 hover:bg-primary-500 transition-colors" />

          {/* Right Panel - Validation */}
          <ResizablePanel defaultSize={30} minSize={20} maxSize={40}>
            <ValidationPanel projectId={activeProject} />
          </ResizablePanel>
        </ResizablePanelGroup>
      </div>
    </div>
  );
}