"use client";

/**
 * Schematics Page - MAPO v3.0 Schematic Generation
 *
 * Full schematic generation pipeline with:
 * 1. Gather Symbols & Datasheets (pre-generation)
 * 2. Generate Schematic (MAPO pipeline)
 * 3. Real-time Quality Checklist (51 compliance checks)
 */

import { useState, useEffect } from "react";
import { useParams } from "next/navigation";
import {
  FileText,
  Zap,
  CheckCircle2,
  AlertTriangle,
  Download,
  Eye
} from "lucide-react";
import { GatherSymbolsButton } from "@/components/schematic/GatherSymbolsButton";
import { SchematicQualityChecklist } from "@/components/schematic/SchematicQualityChecklist";

interface IdeationArtifact {
  id: string;
  type: string;
  content: string;
  created_at: string;
}

export default function SchematicsPage() {
  const params = useParams();
  const projectId = params.projectId as string;

  const [ideationArtifacts, setIdeationArtifacts] = useState<IdeationArtifact[]>([]);
  const [isLoadingArtifacts, setIsLoadingArtifacts] = useState(true);
  const [symbolsGathered, setSymbolsGathered] = useState(false);
  const [assemblyReport, setAssemblyReport] = useState<any>(null);

  const [isGenerating, setIsGenerating] = useState(false);
  const [schematicOperationId, setSchematicOperationId] = useState<string | null>(null);
  const [schematicGenerated, setSchematicGenerated] = useState(false);
  const [schematicPath, setSchematicPath] = useState<string | null>(null);

  // Load ideation artifacts
  useEffect(() => {
    fetchIdeationArtifacts();
  }, [projectId]);

  const fetchIdeationArtifacts = async () => {
    setIsLoadingArtifacts(true);
    try {
      const response = await fetch(`/api/v1/projects/${projectId}/ideation-artifacts`);
      if (!response.ok) throw new Error(`HTTP ${response.status}`);

      const data = await response.json();
      setIdeationArtifacts(data.artifacts || []);
    } catch (err) {
      console.error("Failed to load ideation artifacts:", err);
    } finally {
      setIsLoadingArtifacts(false);
    }
  };

  // Handle symbol assembly completion
  const handleSymbolsGathered = (report: any) => {
    console.log("Symbols gathered:", report);
    setSymbolsGathered(true);
    setAssemblyReport(report);
  };

  // Generate schematic
  const handleGenerateSchematic = async () => {
    if (!ideationArtifacts || ideationArtifacts.length === 0) {
      alert("No ideation artifacts found. Please create design documents first.");
      return;
    }

    setIsGenerating(true);
    setSchematicGenerated(false);
    setSchematicPath(null);

    try {
      const response = await fetch(`/api/v1/projects/${projectId}/schematics/generate`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          ideationArtifacts: ideationArtifacts.map(a => ({
            type: a.type,
            content: a.content
          })),
          autoFix: true, // Enable auto-fix for compliance issues
        }),
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.error || `HTTP ${response.status}`);
      }

      const data = await response.json();
      setSchematicOperationId(data.operationId);

      // Poll for completion (or use WebSocket in production)
      pollSchematicStatus(data.operationId);
    } catch (err) {
      console.error("Failed to generate schematic:", err);
      alert(`Schematic generation failed: ${err instanceof Error ? err.message : 'Unknown error'}`);
      setIsGenerating(false);
    }
  };

  // Poll schematic generation status
  const pollSchematicStatus = async (operationId: string) => {
    const maxAttempts = 120; // 10 minutes max
    let attempts = 0;

    const poll = async () => {
      if (attempts >= maxAttempts) {
        console.error("Schematic generation timed out");
        setIsGenerating(false);
        return;
      }

      try {
        const response = await fetch(
          `/api/v1/projects/${projectId}/schematics/${operationId}`
        );
        if (!response.ok) throw new Error(`HTTP ${response.status}`);

        const data = await response.json();

        if (data.status === "complete") {
          setSchematicGenerated(true);
          setSchematicPath(data.schematicPath);
          setIsGenerating(false);
          return;
        } else if (data.status === "failed") {
          alert(`Schematic generation failed: ${data.error}`);
          setIsGenerating(false);
          return;
        }

        // Still running, poll again
        attempts++;
        setTimeout(poll, 5000); // Poll every 5 seconds
      } catch (err) {
        console.error("Failed to poll schematic status:", err);
        setIsGenerating(false);
      }
    };

    poll();
  };

  // Download schematic
  const handleDownloadSchematic = () => {
    if (!schematicPath) return;
    window.open(`/api/v1/projects/${projectId}/schematics/download?path=${encodeURIComponent(schematicPath)}`, '_blank');
  };

  // View schematic (open in KiCad or viewer)
  const handleViewSchematic = () => {
    if (!schematicPath) return;
    // TODO: Implement schematic viewer or launch KiCad
    alert("Schematic viewer coming soon! For now, download the .kicad_sch file and open in KiCad.");
  };

  return (
    <div className="min-h-screen bg-gray-50 p-6">
      <div className="max-w-7xl mx-auto space-y-6">
        {/* Header */}
        <div className="bg-white rounded-lg shadow-sm p-6 border border-gray-200">
          <h1 className="text-3xl font-bold text-gray-900 mb-2">
            Schematic Generation
          </h1>
          <p className="text-gray-600">
            MAPO v3.0 - AI-Orchestrated Gaming AI for Professional Schematics
          </p>
        </div>

        {/* Ideation Artifacts Summary */}
        <div className="bg-white rounded-lg shadow-sm p-6 border border-gray-200">
          <div className="flex items-center justify-between mb-4">
            <div className="flex items-center gap-2">
              <FileText className="w-5 h-5 text-blue-600" />
              <h2 className="text-xl font-semibold text-gray-900">
                Ideation Artifacts
              </h2>
            </div>
            <span className="text-sm text-gray-500">
              {ideationArtifacts.length} artifact(s) loaded
            </span>
          </div>

          {isLoadingArtifacts ? (
            <div className="text-gray-500">Loading artifacts...</div>
          ) : ideationArtifacts.length === 0 ? (
            <div className="text-yellow-600 flex items-center gap-2">
              <AlertTriangle className="w-4 h-4" />
              No ideation artifacts found. Create design documents first.
            </div>
          ) : (
            <div className="space-y-2">
              {ideationArtifacts.slice(0, 5).map((artifact) => (
                <div key={artifact.id} className="text-sm text-gray-600 flex items-center gap-2">
                  <CheckCircle2 className="w-4 h-4 text-green-600" />
                  {artifact.type}: {artifact.content.substring(0, 80)}...
                </div>
              ))}
              {ideationArtifacts.length > 5 && (
                <div className="text-sm text-gray-500">
                  ... and {ideationArtifacts.length - 5} more
                </div>
              )}
            </div>
          )}
        </div>

        {/* Step 1: Gather Symbols & Datasheets */}
        <div className="bg-white rounded-lg shadow-sm p-6 border border-gray-200">
          <div className="flex items-center gap-2 mb-4">
            <span className="flex items-center justify-center w-8 h-8 rounded-full bg-blue-100 text-blue-600 font-bold">
              1
            </span>
            <h2 className="text-xl font-semibold text-gray-900">
              Gather Symbols & Datasheets
            </h2>
          </div>

          <p className="text-gray-600 mb-4">
            Pre-fetch component symbols and datasheets from GraphRAG, KiCad libraries,
            SnapEDA, UltraLibrarian, and datasheets before schematic generation.
          </p>

          <GatherSymbolsButton
            projectId={projectId}
            ideationArtifacts={ideationArtifacts.map(a => ({ type: a.type, content: a.content }))}
            onComplete={handleSymbolsGathered}
            disabled={ideationArtifacts.length === 0}
          />

          {assemblyReport && (
            <div className="mt-4 p-4 bg-green-50 rounded-lg border border-green-200">
              <div className="grid grid-cols-3 gap-4 text-sm">
                <div>
                  <span className="text-gray-600">Symbols:</span>
                  <span className="ml-2 font-medium text-green-600">
                    {assemblyReport.symbols_found}
                  </span>
                </div>
                <div>
                  <span className="text-gray-600">Datasheets:</span>
                  <span className="ml-2 font-medium text-green-600">
                    {assemblyReport.datasheets_downloaded}
                  </span>
                </div>
                <div>
                  <span className="text-gray-600">Success Rate:</span>
                  <span className="ml-2 font-medium text-green-600">
                    {Math.round((assemblyReport.gathered_components / assemblyReport.total_components) * 100)}%
                  </span>
                </div>
              </div>
            </div>
          )}
        </div>

        {/* Step 2: Generate Schematic */}
        <div className="bg-white rounded-lg shadow-sm p-6 border border-gray-200">
          <div className="flex items-center gap-2 mb-4">
            <span className="flex items-center justify-center w-8 h-8 rounded-full bg-blue-100 text-blue-600 font-bold">
              2
            </span>
            <h2 className="text-xl font-semibold text-gray-900">
              Generate Schematic
            </h2>
          </div>

          <p className="text-gray-600 mb-4">
            Run the MAPO v3.0 Gaming AI pipeline to generate a professional KiCad schematic
            with optimized placement, routing, and standards compliance.
          </p>

          <button
            onClick={handleGenerateSchematic}
            disabled={ideationArtifacts.length === 0 || isGenerating}
            className="flex items-center gap-2 px-6 py-3 rounded-md font-medium transition-colors bg-blue-600 text-white hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed"
          >
            {isGenerating ? (
              <>
                <Zap className="w-5 h-5 animate-pulse" />
                Generating Schematic...
              </>
            ) : (
              <>
                <Zap className="w-5 h-5" />
                Generate Schematic
              </>
            )}
          </button>

          {schematicGenerated && schematicPath && (
            <div className="mt-4 p-4 bg-green-50 rounded-lg border border-green-200">
              <div className="flex items-center gap-2 mb-3">
                <CheckCircle2 className="w-5 h-5 text-green-600" />
                <span className="font-semibold text-green-800">
                  Schematic Generated Successfully
                </span>
              </div>
              <div className="text-sm text-gray-600 mb-3">
                {schematicPath}
              </div>
              <div className="flex gap-2">
                <button
                  onClick={handleDownloadSchematic}
                  className="flex items-center gap-2 px-4 py-2 rounded-md bg-blue-600 text-white hover:bg-blue-700 text-sm"
                >
                  <Download className="w-4 h-4" />
                  Download .kicad_sch
                </button>
                <button
                  onClick={handleViewSchematic}
                  className="flex items-center gap-2 px-4 py-2 rounded-md bg-gray-600 text-white hover:bg-gray-700 text-sm"
                >
                  <Eye className="w-4 h-4" />
                  View Schematic
                </button>
              </div>
            </div>
          )}
        </div>

        {/* Step 3: Quality Checklist */}
        {schematicOperationId && (
          <div className="bg-white rounded-lg shadow-sm p-6 border border-gray-200">
            <div className="flex items-center gap-2 mb-4">
              <span className="flex items-center justify-center w-8 h-8 rounded-full bg-blue-100 text-blue-600 font-bold">
                3
              </span>
              <h2 className="text-xl font-semibold text-gray-900">
                Standards Compliance Checklist
              </h2>
            </div>

            <p className="text-gray-600 mb-4">
              Real-time validation against 51 compliance checks across 6 standards
              (NASA, MIL, IPC, IEC, Best Practices).
            </p>

            <SchematicQualityChecklist
              operationId={schematicOperationId}
              projectId={projectId}
              autoFixEnabled={true}
            />
          </div>
        )}
      </div>
    </div>
  );
}
