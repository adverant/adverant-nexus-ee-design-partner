"use client";

/**
 * Waveform Viewer - Canvas-based Oscilloscope Waveform Display
 *
 * High-performance waveform visualization with zoom, pan, cursors,
 * measurements, and multi-channel support. Uses HTML5 Canvas for
 * smooth rendering of large datasets.
 */

import { useRef, useEffect, useState, useCallback, useMemo } from "react";
import { cn } from "@/lib/utils";
import {
  ZoomIn,
  ZoomOut,
  Maximize2,
  Move,
  Crosshair,
  RotateCcw,
  Download,
  Settings,
  Eye,
  EyeOff,
  Pause,
  Play,
  ChevronDown,
} from "lucide-react";

// ============================================================================
// Types
// ============================================================================

export interface ChannelConfig {
  name: string;
  label?: string;
  scale: number;
  offset: number;
  unit: string;
  color?: string;
  visible?: boolean;
  coupling?: "ac" | "dc" | "gnd";
  probeAttenuation?: number;
}

export interface WaveformData {
  captureId: string;
  channels: string[];
  sampleRate: number;
  totalSamples: number;
  data: number[][];
  timebase: {
    startTime: number;
    endTime: number;
    unit: string;
  };
  channelConfig: ChannelConfig[];
}

export interface Cursor {
  id: string;
  type: "time" | "voltage";
  position: number; // Normalized position (0-1)
  channel?: string;
  color: string;
}

export interface Measurement {
  type: string;
  channel: string;
  value: number;
  unit: string;
}

interface WaveformViewerProps {
  data: WaveformData | null;
  className?: string;
  onZoomChange?: (start: number, end: number) => void;
  onCursorChange?: (cursors: Cursor[]) => void;
  initialZoom?: { start: number; end: number };
  showGrid?: boolean;
  showMeasurements?: boolean;
  autoScale?: boolean;
  liveMode?: boolean;
  onPause?: () => void;
  onResume?: () => void;
}

// ============================================================================
// Constants
// ============================================================================

const CHANNEL_COLORS = [
  "#FFD700", // Yellow (CH1)
  "#00FFFF", // Cyan (CH2)
  "#FF69B4", // Pink (CH3)
  "#32CD32", // Lime Green (CH4)
  "#FF6347", // Tomato (CH5)
  "#9370DB", // Medium Purple (CH6)
  "#20B2AA", // Light Sea Green (CH7)
  "#FFA500", // Orange (CH8)
];

const GRID_COLOR = "rgba(128, 128, 128, 0.3)";
const AXIS_COLOR = "rgba(255, 255, 255, 0.5)";
const CURSOR_COLORS = ["#FF4444", "#44FF44"];

// ============================================================================
// Helper Functions
// ============================================================================

function formatTime(seconds: number): string {
  if (Math.abs(seconds) < 1e-6) return `${(seconds * 1e9).toFixed(1)}ns`;
  if (Math.abs(seconds) < 1e-3) return `${(seconds * 1e6).toFixed(1)}µs`;
  if (Math.abs(seconds) < 1) return `${(seconds * 1e3).toFixed(2)}ms`;
  return `${seconds.toFixed(3)}s`;
}

function formatVoltage(value: number, unit: string = "V"): string {
  if (Math.abs(value) < 1e-3) return `${(value * 1e6).toFixed(1)}µ${unit}`;
  if (Math.abs(value) < 1) return `${(value * 1e3).toFixed(2)}m${unit}`;
  return `${value.toFixed(3)}${unit}`;
}

function calculateMeasurements(
  data: number[],
  sampleRate: number
): { vpp: number; vrms: number; vmax: number; vmin: number; freq?: number } {
  if (!data || data.length === 0) {
    return { vpp: 0, vrms: 0, vmax: 0, vmin: 0 };
  }

  const vmax = Math.max(...data);
  const vmin = Math.min(...data);
  const vpp = vmax - vmin;

  // Calculate RMS
  const sumSquares = data.reduce((sum, v) => sum + v * v, 0);
  const vrms = Math.sqrt(sumSquares / data.length);

  // Simple frequency detection via zero crossings
  let zeroCrossings = 0;
  const mean = data.reduce((a, b) => a + b, 0) / data.length;
  for (let i = 1; i < data.length; i++) {
    if ((data[i - 1] - mean) * (data[i] - mean) < 0) {
      zeroCrossings++;
    }
  }
  const freq = zeroCrossings > 2 ? (zeroCrossings / 2) * (sampleRate / data.length) : undefined;

  return { vpp, vrms, vmax, vmin, freq };
}

// ============================================================================
// Channel Legend Component
// ============================================================================

interface ChannelLegendProps {
  channels: ChannelConfig[];
  visibleChannels: Set<string>;
  onToggleChannel: (channel: string) => void;
  measurements?: Record<string, { vpp: number; vrms: number; freq?: number }>;
}

function ChannelLegend({
  channels,
  visibleChannels,
  onToggleChannel,
  measurements,
}: ChannelLegendProps) {
  return (
    <div className="flex flex-wrap gap-2 p-2 bg-slate-900/50 rounded-lg">
      {channels.map((ch, i) => {
        const color = ch.color || CHANNEL_COLORS[i % CHANNEL_COLORS.length];
        const isVisible = visibleChannels.has(ch.name);
        const meas = measurements?.[ch.name];

        return (
          <button
            key={ch.name}
            onClick={() => onToggleChannel(ch.name)}
            className={cn(
              "flex items-center gap-2 px-3 py-1.5 rounded-lg transition-all text-xs",
              isVisible ? "bg-slate-800" : "bg-slate-800/50 opacity-50"
            )}
          >
            <div
              className="w-3 h-3 rounded-full"
              style={{ backgroundColor: color, opacity: isVisible ? 1 : 0.3 }}
            />
            <div className="text-left">
              <div className="font-medium" style={{ color: isVisible ? color : "#666" }}>
                {ch.label || ch.name}
              </div>
              {meas && isVisible && (
                <div className="text-[10px] text-slate-500">
                  Vpp: {formatVoltage(meas.vpp, ch.unit)} | RMS: {formatVoltage(meas.vrms, ch.unit)}
                  {meas.freq && ` | ${(meas.freq / 1000).toFixed(1)}kHz`}
                </div>
              )}
            </div>
            {isVisible ? (
              <Eye className="w-3 h-3 text-slate-400" />
            ) : (
              <EyeOff className="w-3 h-3 text-slate-600" />
            )}
          </button>
        );
      })}
    </div>
  );
}

// ============================================================================
// Toolbar Component
// ============================================================================

interface ToolbarProps {
  zoom: { start: number; end: number };
  onZoomIn: () => void;
  onZoomOut: () => void;
  onResetZoom: () => void;
  onFitToScreen: () => void;
  onExport: () => void;
  isPaused: boolean;
  onTogglePause: () => void;
  liveMode: boolean;
  showGrid: boolean;
  onToggleGrid: () => void;
}

function Toolbar({
  zoom,
  onZoomIn,
  onZoomOut,
  onResetZoom,
  onFitToScreen,
  onExport,
  isPaused,
  onTogglePause,
  liveMode,
  showGrid,
  onToggleGrid,
}: ToolbarProps) {
  const zoomLevel = Math.round((1 / (zoom.end - zoom.start)) * 100);

  return (
    <div className="flex items-center gap-2 p-2 bg-slate-900/50 rounded-lg">
      <div className="flex items-center gap-1 border-r border-slate-700 pr-2">
        <button
          onClick={onZoomIn}
          className="p-1.5 hover:bg-slate-700 rounded"
          title="Zoom In"
        >
          <ZoomIn className="w-4 h-4 text-slate-400" />
        </button>
        <button
          onClick={onZoomOut}
          className="p-1.5 hover:bg-slate-700 rounded"
          title="Zoom Out"
        >
          <ZoomOut className="w-4 h-4 text-slate-400" />
        </button>
        <button
          onClick={onResetZoom}
          className="p-1.5 hover:bg-slate-700 rounded"
          title="Reset Zoom"
        >
          <RotateCcw className="w-4 h-4 text-slate-400" />
        </button>
        <button
          onClick={onFitToScreen}
          className="p-1.5 hover:bg-slate-700 rounded"
          title="Fit to Screen"
        >
          <Maximize2 className="w-4 h-4 text-slate-400" />
        </button>
        <span className="text-xs text-slate-500 ml-2">{zoomLevel}%</span>
      </div>

      <div className="flex items-center gap-1 border-r border-slate-700 pr-2">
        <button
          onClick={onToggleGrid}
          className={cn(
            "p-1.5 rounded",
            showGrid ? "bg-slate-700 text-white" : "hover:bg-slate-700 text-slate-400"
          )}
          title="Toggle Grid"
        >
          <Crosshair className="w-4 h-4" />
        </button>
      </div>

      {liveMode && (
        <div className="flex items-center gap-1 border-r border-slate-700 pr-2">
          <button
            onClick={onTogglePause}
            className={cn(
              "flex items-center gap-1 px-2 py-1 rounded text-xs",
              isPaused ? "bg-green-600 text-white" : "bg-yellow-600 text-white"
            )}
          >
            {isPaused ? (
              <>
                <Play className="w-3 h-3" />
                Resume
              </>
            ) : (
              <>
                <Pause className="w-3 h-3" />
                Pause
              </>
            )}
          </button>
        </div>
      )}

      <button
        onClick={onExport}
        className="flex items-center gap-1 px-2 py-1 bg-slate-700 text-slate-300 rounded text-xs hover:bg-slate-600 ml-auto"
      >
        <Download className="w-3 h-3" />
        Export
      </button>
    </div>
  );
}

// ============================================================================
// Main Component
// ============================================================================

export function WaveformViewer({
  data,
  className,
  onZoomChange,
  onCursorChange,
  initialZoom = { start: 0, end: 1 },
  showGrid: initialShowGrid = true,
  showMeasurements = true,
  autoScale = true,
  liveMode = false,
  onPause,
  onResume,
}: WaveformViewerProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const animationRef = useRef<number>();

  // State
  const [zoom, setZoom] = useState(initialZoom);
  const [visibleChannels, setVisibleChannels] = useState<Set<string>>(
    new Set(data?.channels || [])
  );
  const [showGrid, setShowGrid] = useState(initialShowGrid);
  const [isPaused, setIsPaused] = useState(false);
  const [isDragging, setIsDragging] = useState(false);
  const [dragStart, setDragStart] = useState<{ x: number; zoom: typeof zoom } | null>(null);
  const [cursors, setCursors] = useState<Cursor[]>([]);
  const [canvasSize, setCanvasSize] = useState({ width: 0, height: 0 });

  // Calculate measurements for visible channels
  const measurements = useMemo(() => {
    if (!data || !showMeasurements) return {};

    const result: Record<string, { vpp: number; vrms: number; vmax: number; vmin: number; freq?: number }> = {};

    data.channels.forEach((ch, i) => {
      if (visibleChannels.has(ch) && data.data[i]) {
        const startIdx = Math.floor(zoom.start * data.totalSamples);
        const endIdx = Math.floor(zoom.end * data.totalSamples);
        const visibleData = data.data[i].slice(startIdx, endIdx);
        result[ch] = calculateMeasurements(visibleData, data.sampleRate);
      }
    });

    return result;
  }, [data, visibleChannels, zoom, showMeasurements]);

  // ========================================================================
  // Canvas Drawing
  // ========================================================================

  const drawCanvas = useCallback(() => {
    const canvas = canvasRef.current;
    const ctx = canvas?.getContext("2d");
    if (!canvas || !ctx || !data) return;

    const { width, height } = canvas;
    const padding = { top: 20, right: 60, bottom: 30, left: 60 };
    const plotWidth = width - padding.left - padding.right;
    const plotHeight = height - padding.top - padding.bottom;

    // Clear canvas
    ctx.fillStyle = "#0a0a0a";
    ctx.fillRect(0, 0, width, height);

    // Draw grid
    if (showGrid) {
      ctx.strokeStyle = GRID_COLOR;
      ctx.lineWidth = 1;

      // Vertical grid lines
      const numVertLines = 10;
      for (let i = 0; i <= numVertLines; i++) {
        const x = padding.left + (plotWidth / numVertLines) * i;
        ctx.beginPath();
        ctx.moveTo(x, padding.top);
        ctx.lineTo(x, height - padding.bottom);
        ctx.stroke();
      }

      // Horizontal grid lines
      const numHorizLines = 8;
      for (let i = 0; i <= numHorizLines; i++) {
        const y = padding.top + (plotHeight / numHorizLines) * i;
        ctx.beginPath();
        ctx.moveTo(padding.left, y);
        ctx.lineTo(width - padding.right, y);
        ctx.stroke();
      }
    }

    // Draw axes
    ctx.strokeStyle = AXIS_COLOR;
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(padding.left, padding.top);
    ctx.lineTo(padding.left, height - padding.bottom);
    ctx.lineTo(width - padding.right, height - padding.bottom);
    ctx.stroke();

    // Calculate time range
    const totalDuration = data.totalSamples / data.sampleRate;
    const visibleStart = zoom.start * totalDuration;
    const visibleEnd = zoom.end * totalDuration;

    // Draw time labels
    ctx.fillStyle = "#888";
    ctx.font = "10px monospace";
    ctx.textAlign = "center";
    const numTimeLabels = 5;
    for (let i = 0; i <= numTimeLabels; i++) {
      const t = visibleStart + ((visibleEnd - visibleStart) / numTimeLabels) * i;
      const x = padding.left + (plotWidth / numTimeLabels) * i;
      ctx.fillText(formatTime(t), x, height - 10);
    }

    // Draw waveforms
    const startSample = Math.floor(zoom.start * data.totalSamples);
    const endSample = Math.floor(zoom.end * data.totalSamples);
    const visibleSamples = endSample - startSample;

    // Determine decimation factor for performance
    const maxPoints = plotWidth * 2;
    const decimation = Math.max(1, Math.floor(visibleSamples / maxPoints));

    data.channels.forEach((channel, channelIdx) => {
      if (!visibleChannels.has(channel) || !data.data[channelIdx]) return;

      const channelData = data.data[channelIdx];
      const config = data.channelConfig[channelIdx] || { scale: 1, offset: 0, unit: "V" };
      const color = config.color || CHANNEL_COLORS[channelIdx % CHANNEL_COLORS.length];

      // Find data range for auto-scaling
      let dataMin = Infinity;
      let dataMax = -Infinity;
      for (let i = startSample; i < endSample; i += decimation) {
        const val = channelData[i];
        if (val !== undefined) {
          dataMin = Math.min(dataMin, val);
          dataMax = Math.max(dataMax, val);
        }
      }

      // Add some padding to the range
      const range = dataMax - dataMin || 1;
      const scaledMin = dataMin - range * 0.1;
      const scaledMax = dataMax + range * 0.1;

      // Draw voltage labels for this channel
      ctx.fillStyle = color;
      ctx.font = "9px monospace";
      ctx.textAlign = "right";
      const labelOffset = channelIdx * 12;
      ctx.fillText(formatVoltage(scaledMax, config.unit), padding.left - 5, padding.top + 10 + labelOffset);
      ctx.fillText(formatVoltage(scaledMin, config.unit), padding.left - 5, height - padding.bottom - labelOffset);

      // Draw waveform
      ctx.strokeStyle = color;
      ctx.lineWidth = 1.5;
      ctx.beginPath();

      let firstPoint = true;
      for (let i = startSample; i < endSample; i += decimation) {
        const val = channelData[i];
        if (val === undefined) continue;

        const x = padding.left + ((i - startSample) / visibleSamples) * plotWidth;
        const normalizedVal = (val - scaledMin) / (scaledMax - scaledMin);
        const y = height - padding.bottom - normalizedVal * plotHeight;

        if (firstPoint) {
          ctx.moveTo(x, y);
          firstPoint = false;
        } else {
          ctx.lineTo(x, y);
        }
      }
      ctx.stroke();
    });

    // Draw cursors
    cursors.forEach((cursor, i) => {
      const x = padding.left + cursor.position * plotWidth;
      ctx.strokeStyle = cursor.color;
      ctx.lineWidth = 1;
      ctx.setLineDash([5, 3]);
      ctx.beginPath();
      ctx.moveTo(x, padding.top);
      ctx.lineTo(x, height - padding.bottom);
      ctx.stroke();
      ctx.setLineDash([]);

      // Draw cursor time label
      const t = visibleStart + cursor.position * (visibleEnd - visibleStart);
      ctx.fillStyle = cursor.color;
      ctx.font = "10px monospace";
      ctx.textAlign = "center";
      ctx.fillText(formatTime(t), x, padding.top - 5);
    });

    // Draw delta between cursors
    if (cursors.length === 2) {
      const t1 = visibleStart + cursors[0].position * (visibleEnd - visibleStart);
      const t2 = visibleStart + cursors[1].position * (visibleEnd - visibleStart);
      const deltaT = Math.abs(t2 - t1);
      const freq = deltaT > 0 ? 1 / deltaT : 0;

      ctx.fillStyle = "#fff";
      ctx.font = "11px monospace";
      ctx.textAlign = "left";
      ctx.fillText(
        `Δt: ${formatTime(deltaT)} | f: ${freq > 0 ? (freq / 1000).toFixed(2) + "kHz" : "—"}`,
        padding.left + 10,
        padding.top + 15
      );
    }
  }, [data, zoom, visibleChannels, showGrid, cursors]);

  // ========================================================================
  // Event Handlers
  // ========================================================================

  const handleWheel = useCallback(
    (e: WheelEvent) => {
      e.preventDefault();

      const canvas = canvasRef.current;
      if (!canvas) return;

      const rect = canvas.getBoundingClientRect();
      const mouseX = (e.clientX - rect.left) / rect.width;

      // Calculate zoom factor
      const zoomFactor = e.deltaY > 0 ? 1.1 : 0.9;
      const currentRange = zoom.end - zoom.start;
      const newRange = Math.min(1, Math.max(0.001, currentRange * zoomFactor));

      // Keep the point under mouse stationary
      const zoomPoint = zoom.start + mouseX * currentRange;
      let newStart = zoomPoint - mouseX * newRange;
      let newEnd = zoomPoint + (1 - mouseX) * newRange;

      // Clamp to [0, 1]
      if (newStart < 0) {
        newStart = 0;
        newEnd = newRange;
      }
      if (newEnd > 1) {
        newEnd = 1;
        newStart = 1 - newRange;
      }

      setZoom({ start: newStart, end: newEnd });
      onZoomChange?.(newStart, newEnd);
    },
    [zoom, onZoomChange]
  );

  const handleMouseDown = useCallback(
    (e: React.MouseEvent) => {
      if (e.button !== 0) return; // Only left click
      setIsDragging(true);
      setDragStart({ x: e.clientX, zoom: { ...zoom } });
    },
    [zoom]
  );

  const handleMouseMove = useCallback(
    (e: React.MouseEvent) => {
      if (!isDragging || !dragStart) return;

      const canvas = canvasRef.current;
      if (!canvas) return;

      const rect = canvas.getBoundingClientRect();
      const deltaX = (e.clientX - dragStart.x) / rect.width;
      const range = dragStart.zoom.end - dragStart.zoom.start;

      let newStart = dragStart.zoom.start - deltaX * range;
      let newEnd = dragStart.zoom.end - deltaX * range;

      // Clamp
      if (newStart < 0) {
        newStart = 0;
        newEnd = range;
      }
      if (newEnd > 1) {
        newEnd = 1;
        newStart = 1 - range;
      }

      setZoom({ start: newStart, end: newEnd });
    },
    [isDragging, dragStart]
  );

  const handleMouseUp = useCallback(() => {
    setIsDragging(false);
    setDragStart(null);
    if (zoom.start !== initialZoom.start || zoom.end !== initialZoom.end) {
      onZoomChange?.(zoom.start, zoom.end);
    }
  }, [zoom, initialZoom, onZoomChange]);

  const handleDoubleClick = useCallback(
    (e: React.MouseEvent) => {
      const canvas = canvasRef.current;
      if (!canvas) return;

      const rect = canvas.getBoundingClientRect();
      const x = (e.clientX - rect.left) / rect.width;

      // Add cursor at click position
      const newCursor: Cursor = {
        id: `cursor-${Date.now()}`,
        type: "time",
        position: x,
        color: CURSOR_COLORS[cursors.length % CURSOR_COLORS.length],
      };

      const newCursors = [...cursors, newCursor].slice(-2); // Keep max 2 cursors
      setCursors(newCursors);
      onCursorChange?.(newCursors);
    },
    [cursors, onCursorChange]
  );

  // ========================================================================
  // Toolbar Actions
  // ========================================================================

  const handleZoomIn = useCallback(() => {
    const center = (zoom.start + zoom.end) / 2;
    const range = (zoom.end - zoom.start) / 2;
    const newRange = range / 2;
    setZoom({
      start: Math.max(0, center - newRange),
      end: Math.min(1, center + newRange),
    });
  }, [zoom]);

  const handleZoomOut = useCallback(() => {
    const center = (zoom.start + zoom.end) / 2;
    const range = (zoom.end - zoom.start) / 2;
    const newRange = Math.min(0.5, range * 2);
    setZoom({
      start: Math.max(0, center - newRange),
      end: Math.min(1, center + newRange),
    });
  }, [zoom]);

  const handleResetZoom = useCallback(() => {
    setZoom({ start: 0, end: 1 });
    onZoomChange?.(0, 1);
  }, [onZoomChange]);

  const handleExport = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const link = document.createElement("a");
    link.download = `waveform-${data?.captureId || "capture"}.png`;
    link.href = canvas.toDataURL("image/png");
    link.click();
  }, [data]);

  const handleTogglePause = useCallback(() => {
    setIsPaused(!isPaused);
    if (isPaused) {
      onResume?.();
    } else {
      onPause?.();
    }
  }, [isPaused, onPause, onResume]);

  const handleToggleChannel = useCallback((channel: string) => {
    setVisibleChannels((prev) => {
      const next = new Set(prev);
      if (next.has(channel)) {
        next.delete(channel);
      } else {
        next.add(channel);
      }
      return next;
    });
  }, []);

  // ========================================================================
  // Effects
  // ========================================================================

  // Resize observer
  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;

    const resizeObserver = new ResizeObserver((entries) => {
      const entry = entries[0];
      if (entry) {
        const { width, height } = entry.contentRect;
        setCanvasSize({ width: Math.floor(width), height: Math.floor(height) });
      }
    });

    resizeObserver.observe(container);
    return () => resizeObserver.disconnect();
  }, []);

  // Update canvas size
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const dpr = window.devicePixelRatio || 1;
    canvas.width = canvasSize.width * dpr;
    canvas.height = canvasSize.height * dpr;
    canvas.style.width = `${canvasSize.width}px`;
    canvas.style.height = `${canvasSize.height}px`;

    const ctx = canvas.getContext("2d");
    if (ctx) {
      ctx.scale(dpr, dpr);
    }
  }, [canvasSize]);

  // Animation loop
  useEffect(() => {
    if (!data) return;

    const animate = () => {
      drawCanvas();
      if (liveMode && !isPaused) {
        animationRef.current = requestAnimationFrame(animate);
      }
    };

    animate();

    return () => {
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current);
      }
    };
  }, [data, drawCanvas, liveMode, isPaused]);

  // Wheel event listener
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    canvas.addEventListener("wheel", handleWheel, { passive: false });
    return () => canvas.removeEventListener("wheel", handleWheel);
  }, [handleWheel]);

  // Update visible channels when data changes
  useEffect(() => {
    if (data) {
      setVisibleChannels(new Set(data.channels));
    }
  }, [data?.captureId]);

  // ========================================================================
  // Render
  // ========================================================================

  if (!data) {
    return (
      <div
        className={cn(
          "flex flex-col items-center justify-center h-full bg-slate-900 rounded-lg border border-slate-700",
          className
        )}
      >
        <Crosshair className="w-16 h-16 text-slate-600 mb-4" />
        <p className="text-slate-500">No waveform data</p>
        <p className="text-xs text-slate-600 mt-1">Run a capture to view waveforms</p>
      </div>
    );
  }

  return (
    <div className={cn("flex flex-col h-full bg-slate-900 rounded-lg border border-slate-700", className)}>
      {/* Toolbar */}
      <Toolbar
        zoom={zoom}
        onZoomIn={handleZoomIn}
        onZoomOut={handleZoomOut}
        onResetZoom={handleResetZoom}
        onFitToScreen={handleResetZoom}
        onExport={handleExport}
        isPaused={isPaused}
        onTogglePause={handleTogglePause}
        liveMode={liveMode}
        showGrid={showGrid}
        onToggleGrid={() => setShowGrid(!showGrid)}
      />

      {/* Canvas */}
      <div ref={containerRef} className="flex-1 relative overflow-hidden">
        <canvas
          ref={canvasRef}
          className={cn(
            "absolute inset-0",
            isDragging ? "cursor-grabbing" : "cursor-crosshair"
          )}
          onMouseDown={handleMouseDown}
          onMouseMove={handleMouseMove}
          onMouseUp={handleMouseUp}
          onMouseLeave={handleMouseUp}
          onDoubleClick={handleDoubleClick}
        />
      </div>

      {/* Channel Legend */}
      <ChannelLegend
        channels={data.channelConfig}
        visibleChannels={visibleChannels}
        onToggleChannel={handleToggleChannel}
        measurements={measurements}
      />

      {/* Capture Info */}
      <div className="flex items-center justify-between px-3 py-1.5 text-[10px] text-slate-500 border-t border-slate-800">
        <span>
          {data.totalSamples.toLocaleString()} samples @ {(data.sampleRate / 1e6).toFixed(1)} MS/s
        </span>
        <span>
          {formatTime((zoom.end - zoom.start) * data.totalSamples / data.sampleRate)} visible
        </span>
        <span>Capture: {data.captureId.substring(0, 8)}</span>
      </div>
    </div>
  );
}

export default WaveformViewer;
