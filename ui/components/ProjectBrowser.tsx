"use client";

/**
 * EE Design Partner - Project Browser Component
 *
 * Production-ready file browser with real backend integration.
 * Fetches file tree from API and handles project selection.
 */

import { useEffect, useState, useCallback } from "react";
import { cn } from "@/lib/utils";
import {
  FolderOpen,
  Folder,
  File,
  FileCode,
  FileJson,
  ChevronRight,
  ChevronDown,
  Plus,
  RefreshCw,
  GitBranch,
  Loader2,
  AlertCircle,
  FileText,
  Settings,
  Package,
} from "lucide-react";
import { useEEDesignStore, type FileNode } from "@/hooks/useEEDesignStore";

interface ProjectBrowserProps {
  activeProject: string | null;
  onProjectSelect: (projectId: string) => void;
}

// Get appropriate icon for file type
function getFileIcon(name: string, extension?: string) {
  // By extension
  switch (extension?.toLowerCase()) {
    case "json":
      return <FileJson className="w-4 h-4 text-yellow-400" />;
    case "c":
    case "h":
    case "cpp":
    case "hpp":
      return <FileCode className="w-4 h-4 text-blue-400" />;
    case "ts":
    case "tsx":
    case "js":
    case "jsx":
      return <FileCode className="w-4 h-4 text-blue-400" />;
    case "py":
      return <FileCode className="w-4 h-4 text-green-400" />;
    case "kicad_sch":
      return <FileCode className="w-4 h-4 text-cyan-400" />;
    case "kicad_pcb":
      return <FileCode className="w-4 h-4 text-green-400" />;
    case "md":
    case "txt":
      return <FileText className="w-4 h-4 text-slate-400" />;
    case "csv":
      return <FileText className="w-4 h-4 text-green-400" />;
    case "yaml":
    case "yml":
      return <Settings className="w-4 h-4 text-purple-400" />;
    case "toml":
      return <Settings className="w-4 h-4 text-orange-400" />;
    default:
      break;
  }

  // By filename
  if (name.toLowerCase().includes("package")) {
    return <Package className="w-4 h-4 text-green-400" />;
  }
  if (name.toLowerCase().includes("config")) {
    return <Settings className="w-4 h-4 text-slate-400" />;
  }

  return <File className="w-4 h-4 text-slate-400" />;
}

// Extract extension from filename
function getExtension(name: string): string | undefined {
  const parts = name.split(".");
  if (parts.length > 1) {
    return parts[parts.length - 1];
  }
  return undefined;
}

// Tree node component
function TreeNode({
  node,
  depth = 0,
  expandedFolders,
  onToggle,
  onSelect,
  selectedPath,
}: {
  node: FileNode;
  depth?: number;
  expandedFolders: Set<string>;
  onToggle: (path: string) => void;
  onSelect: (path: string) => void;
  selectedPath: string | null;
}) {
  const isFolder = node.type === "directory";
  const isExpanded = expandedFolders.has(node.path);
  const isSelected = selectedPath === node.path;
  const extension = getExtension(node.name);

  return (
    <div>
      <button
        onClick={() => {
          if (isFolder) {
            onToggle(node.path);
          } else {
            onSelect(node.path);
          }
        }}
        className={cn(
          "w-full flex items-center gap-1.5 py-1 px-2 text-sm rounded transition-colors",
          isSelected
            ? "bg-primary-600/20 text-primary-400"
            : "text-slate-300 hover:text-white hover:bg-surface-secondary"
        )}
        style={{ paddingLeft: `${depth * 12 + 8}px` }}
      >
        {isFolder ? (
          <>
            {isExpanded ? (
              <ChevronDown className="w-3.5 h-3.5 text-slate-500 flex-shrink-0" />
            ) : (
              <ChevronRight className="w-3.5 h-3.5 text-slate-500 flex-shrink-0" />
            )}
            {isExpanded ? (
              <FolderOpen className="w-4 h-4 text-yellow-400 flex-shrink-0" />
            ) : (
              <Folder className="w-4 h-4 text-slate-400 flex-shrink-0" />
            )}
          </>
        ) : (
          <>
            <span className="w-3.5" />
            {getFileIcon(node.name, extension)}
          </>
        )}
        <span className="truncate">{node.name}</span>
        {node.size !== undefined && !isFolder && (
          <span className="text-xs text-slate-600 ml-auto">
            {formatFileSize(node.size)}
          </span>
        )}
      </button>

      {isFolder && isExpanded && node.children && (
        <div>
          {node.children.map((child) => (
            <TreeNode
              key={child.path}
              node={child}
              depth={depth + 1}
              expandedFolders={expandedFolders}
              onToggle={onToggle}
              onSelect={onSelect}
              selectedPath={selectedPath}
            />
          ))}
        </div>
      )}
    </div>
  );
}

// Format file size
function formatFileSize(bytes: number): string {
  if (bytes < 1024) return `${bytes}B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)}KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)}MB`;
}

export function ProjectBrowser({ activeProject, onProjectSelect }: ProjectBrowserProps) {
  const [showProjectList, setShowProjectList] = useState(false);

  // Store state and actions
  const {
    projects,
    fileTree,
    selectedFile,
    expandedFolders,
    loading,
    errors,
    activeProject: storeActiveProject,
    fetchProjects,
    fetchFileTree,
    setActiveProject,
    setSelectedFile,
    toggleFolder,
  } = useEEDesignStore();

  // Fetch projects on mount
  useEffect(() => {
    fetchProjects();
  }, [fetchProjects]);

  // Fetch file tree when active project changes
  useEffect(() => {
    if (activeProject) {
      fetchFileTree(activeProject);
    }
  }, [activeProject, fetchFileTree]);

  // Handle project selection
  const handleProjectSelect = useCallback(
    (projectId: string) => {
      setActiveProject(projectId);
      onProjectSelect(projectId);
      setShowProjectList(false);
    },
    [setActiveProject, onProjectSelect]
  );

  // Handle refresh
  const handleRefresh = useCallback(() => {
    if (activeProject) {
      fetchFileTree(activeProject);
    }
  }, [activeProject, fetchFileTree]);

  // Handle new project (would open a modal in real implementation)
  const handleNewProject = useCallback(() => {
    // In real implementation, this would open a create project modal
    console.log("Open create project modal");
  }, []);

  // Get current project name
  const currentProjectName =
    projects.find((p) => p.id === activeProject)?.name || "Select Project";

  return (
    <div className="h-full flex flex-col bg-surface-primary border-r border-slate-700">
      {/* Header */}
      <div className="flex items-center justify-between px-3 py-2 border-b border-slate-700">
        <button
          onClick={() => setShowProjectList(!showProjectList)}
          className="flex items-center gap-2 text-sm font-medium text-white hover:text-primary-400 transition-colors"
        >
          <span className="truncate max-w-[120px]">{currentProjectName}</span>
          <ChevronDown
            className={cn(
              "w-4 h-4 text-slate-400 transition-transform",
              showProjectList && "rotate-180"
            )}
          />
        </button>
        <div className="flex items-center gap-1">
          <button
            onClick={handleNewProject}
            className="p-1 rounded hover:bg-surface-secondary transition-colors text-slate-400 hover:text-white"
            title="New Project"
          >
            <Plus className="w-4 h-4" />
          </button>
          <button
            onClick={handleRefresh}
            disabled={loading.fileTree}
            className="p-1 rounded hover:bg-surface-secondary transition-colors text-slate-400 hover:text-white disabled:opacity-50"
            title="Refresh"
          >
            <RefreshCw className={cn("w-4 h-4", loading.fileTree && "animate-spin")} />
          </button>
        </div>
      </div>

      {/* Project List Dropdown */}
      {showProjectList && (
        <div className="border-b border-slate-700 bg-background-primary max-h-48 overflow-auto">
          {loading.projects ? (
            <div className="flex items-center justify-center py-4">
              <Loader2 className="w-5 h-5 text-primary-400 animate-spin" />
            </div>
          ) : errors.projects ? (
            <div className="px-3 py-2 text-sm text-red-400">{errors.projects}</div>
          ) : projects.length === 0 ? (
            <div className="px-3 py-2 text-sm text-slate-500">No projects found</div>
          ) : (
            projects.map((project) => (
              <button
                key={project.id}
                onClick={() => handleProjectSelect(project.id)}
                className={cn(
                  "w-full px-3 py-2 text-left text-sm hover:bg-surface-secondary transition-colors",
                  project.id === activeProject
                    ? "bg-primary-600/20 text-primary-400"
                    : "text-slate-300"
                )}
              >
                <div className="font-medium truncate">{project.name}</div>
                {project.description && (
                  <div className="text-xs text-slate-500 truncate">{project.description}</div>
                )}
              </button>
            ))
          )}
        </div>
      )}

      {/* Git Branch (if project selected) */}
      {activeProject && storeActiveProject && (
        <div className="flex items-center gap-2 px-3 py-2 border-b border-slate-700 text-xs">
          <GitBranch className="w-3.5 h-3.5 text-slate-400" />
          <span className="text-slate-400">main</span>
          <span
            className={cn(
              "ml-auto px-1.5 py-0.5 rounded text-[10px]",
              storeActiveProject.status === "completed"
                ? "bg-green-500/20 text-green-400"
                : storeActiveProject.status === "in_progress"
                ? "bg-yellow-500/20 text-yellow-400"
                : "bg-slate-500/20 text-slate-400"
            )}
          >
            {storeActiveProject.status}
          </span>
        </div>
      )}

      {/* File Tree */}
      <div className="flex-1 overflow-auto py-2">
        {!activeProject ? (
          <div className="flex flex-col items-center justify-center h-full text-center px-4">
            <Folder className="w-10 h-10 text-slate-600 mb-3" />
            <p className="text-sm text-slate-500 mb-2">No project selected</p>
            <button
              onClick={() => setShowProjectList(true)}
              className="text-xs text-primary-400 hover:text-primary-300"
            >
              Select a project
            </button>
          </div>
        ) : loading.fileTree ? (
          <div className="flex items-center justify-center h-32">
            <Loader2 className="w-6 h-6 text-primary-400 animate-spin" />
          </div>
        ) : errors.fileTree ? (
          <div className="px-3 py-4 text-center">
            <AlertCircle className="w-8 h-8 text-red-400 mx-auto mb-2" />
            <p className="text-sm text-red-400 mb-2">Failed to load files</p>
            <p className="text-xs text-slate-500 mb-3">{errors.fileTree}</p>
            <button
              onClick={handleRefresh}
              className="text-xs text-primary-400 hover:text-primary-300"
            >
              Retry
            </button>
          </div>
        ) : !fileTree ? (
          <div className="flex flex-col items-center justify-center h-32 text-center px-4">
            <File className="w-8 h-8 text-slate-600 mb-2" />
            <p className="text-sm text-slate-500">No files yet</p>
          </div>
        ) : (
          <TreeNode
            node={fileTree}
            expandedFolders={expandedFolders}
            onToggle={toggleFolder}
            onSelect={setSelectedFile}
            selectedPath={selectedFile}
          />
        )}
      </div>

      {/* Footer Stats */}
      {activeProject && storeActiveProject && (
        <div className="px-3 py-2 border-t border-slate-700 text-xs text-slate-500">
          <div className="flex items-center justify-between">
            <span>Phase: {storeActiveProject.phase}</span>
            {storeActiveProject.completedPhases && (
              <span>{storeActiveProject.completedPhases.length}/10 phases</span>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
