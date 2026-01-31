"use client";

/**
 * KiCanvas Viewer - Web component wrapper for rendering KiCad schematics and PCBs
 *
 * Uses the KiCanvas web component (https://kicanvas.org) to render
 * .kicad_sch and .kicad_pcb files in the browser.
 */

import { useEffect, useRef, useState, useCallback } from "react";
import { cn } from "@/lib/utils";
import { Loader2, AlertCircle, ZoomIn, ZoomOut, RotateCcw, Maximize2 } from "lucide-react";

// ============================================================================
// Types
// ============================================================================

interface KiCanvasViewerProps {
  /** URL to the KiCad file (.kicad_sch or .kicad_pcb) */
  fileUrl: string;
  /** Type of file being viewed */
  fileType: "schematic" | "pcb";
  /** Optional class name for the container */
  className?: string;
  /** Callback when the viewer has loaded */
  onLoad?: () => void;
  /** Callback when an error occurs */
  onError?: (error: string) => void;
  /** Enable/disable controls */
  showControls?: boolean;
}

interface KiCanvasElement extends HTMLElement {
  src?: string;
  zoom?: number;
  fit?: () => void;
}

// ============================================================================
// Component
// ============================================================================

export function KiCanvasViewer({
  fileUrl,
  fileType,
  className,
  onLoad,
  onError,
  showControls = true,
}: KiCanvasViewerProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const kicanvasRef = useRef<KiCanvasElement | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [scriptLoaded, setScriptLoaded] = useState(false);

  // Load KiCanvas script
  useEffect(() => {
    // Check if script is already loaded
    const existingScript = document.querySelector('script[src*="kicanvas"]');
    if (existingScript) {
      setScriptLoaded(true);
      return;
    }

    const script = document.createElement("script");
    script.src = "https://kicanvas.org/kicanvas/kicanvas.js";
    script.type = "module";
    script.async = true;

    script.onload = () => {
      setScriptLoaded(true);
    };

    script.onerror = () => {
      const errorMsg = "Failed to load KiCanvas viewer script";
      setError(errorMsg);
      setIsLoading(false);
      onError?.(errorMsg);
    };

    document.head.appendChild(script);

    return () => {
      // Don't remove script on unmount - other instances may need it
    };
  }, [onError]);

  // Create KiCanvas element when script is loaded
  useEffect(() => {
    if (!scriptLoaded || !containerRef.current || !fileUrl) {
      return;
    }

    // Clear any existing content
    const container = containerRef.current;
    container.innerHTML = "";

    setIsLoading(true);
    setError(null);

    // Create kicanvas-embed element
    const kicanvas = document.createElement("kicanvas-embed") as KiCanvasElement;
    kicanvas.setAttribute("src", fileUrl);
    kicanvas.setAttribute("controls", showControls ? "full" : "none");
    kicanvas.style.width = "100%";
    kicanvas.style.height = "100%";
    kicanvas.style.display = "block";

    // Event listeners
    kicanvas.addEventListener("load", () => {
      setIsLoading(false);
      onLoad?.();
    });

    kicanvas.addEventListener("error", (e: Event) => {
      const errorMsg = `Failed to load ${fileType}: ${(e as CustomEvent).detail || "Unknown error"}`;
      setError(errorMsg);
      setIsLoading(false);
      onError?.(errorMsg);
    });

    kicanvasRef.current = kicanvas;
    container.appendChild(kicanvas);

    // Fallback timeout for load event
    const loadTimeout = setTimeout(() => {
      if (isLoading) {
        setIsLoading(false);
        // Don't set error - KiCanvas might still be working
      }
    }, 5000);

    return () => {
      clearTimeout(loadTimeout);
      if (container.contains(kicanvas)) {
        container.removeChild(kicanvas);
      }
      kicanvasRef.current = null;
    };
  }, [scriptLoaded, fileUrl, fileType, showControls, onLoad, onError, isLoading]);

  // Control handlers
  const handleZoomIn = useCallback(() => {
    if (kicanvasRef.current?.zoom !== undefined) {
      kicanvasRef.current.zoom *= 1.25;
    }
  }, []);

  const handleZoomOut = useCallback(() => {
    if (kicanvasRef.current?.zoom !== undefined) {
      kicanvasRef.current.zoom *= 0.8;
    }
  }, []);

  const handleFitToView = useCallback(() => {
    kicanvasRef.current?.fit?.();
  }, []);

  const handleFullscreen = useCallback(() => {
    containerRef.current?.requestFullscreen?.();
  }, []);

  // Error state
  if (error) {
    return (
      <div className={cn("flex items-center justify-center h-full bg-slate-900", className)}>
        <div className="text-center p-6">
          <AlertCircle className="w-12 h-12 text-red-500 mx-auto mb-4" />
          <h3 className="text-lg font-medium text-white mb-2">Failed to Load {fileType}</h3>
          <p className="text-sm text-slate-400 max-w-md">{error}</p>
          <button
            onClick={() => {
              setError(null);
              setIsLoading(true);
              // Force reload by updating the container
              if (containerRef.current) {
                containerRef.current.innerHTML = "";
              }
            }}
            className="mt-4 px-4 py-2 bg-primary-600 text-white rounded-lg hover:bg-primary-700 transition-colors"
          >
            Retry
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className={cn("relative h-full bg-slate-900", className)}>
      {/* Loading overlay */}
      {isLoading && (
        <div className="absolute inset-0 flex items-center justify-center bg-slate-900/80 z-10">
          <div className="text-center">
            <Loader2 className="w-10 h-10 text-primary-500 animate-spin mx-auto mb-3" />
            <p className="text-sm text-slate-400">Loading {fileType}...</p>
          </div>
        </div>
      )}

      {/* Custom toolbar (optional - KiCanvas has built-in controls) */}
      {showControls && !isLoading && (
        <div className="absolute top-2 right-2 z-20 flex gap-1 bg-slate-800/80 rounded-lg p-1">
          <button
            onClick={handleZoomIn}
            className="p-2 hover:bg-slate-700 rounded-md transition-colors"
            title="Zoom In"
          >
            <ZoomIn className="w-4 h-4 text-slate-300" />
          </button>
          <button
            onClick={handleZoomOut}
            className="p-2 hover:bg-slate-700 rounded-md transition-colors"
            title="Zoom Out"
          >
            <ZoomOut className="w-4 h-4 text-slate-300" />
          </button>
          <button
            onClick={handleFitToView}
            className="p-2 hover:bg-slate-700 rounded-md transition-colors"
            title="Fit to View"
          >
            <RotateCcw className="w-4 h-4 text-slate-300" />
          </button>
          <button
            onClick={handleFullscreen}
            className="p-2 hover:bg-slate-700 rounded-md transition-colors"
            title="Fullscreen"
          >
            <Maximize2 className="w-4 h-4 text-slate-300" />
          </button>
        </div>
      )}

      {/* KiCanvas container */}
      <div ref={containerRef} className="w-full h-full" />
    </div>
  );
}

export default KiCanvasViewer;
