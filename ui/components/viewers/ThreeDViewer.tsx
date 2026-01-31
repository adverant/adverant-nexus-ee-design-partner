"use client";

/**
 * Three.js 3D Viewer - PCB 3D Model Visualization
 *
 * Renders VRML/WRL files exported from KiCad using Three.js
 * STEP files require server-side conversion to GLTF first.
 */

import { Suspense, useRef, useState, useCallback, useEffect } from "react";
import { Canvas, useThree, useLoader, useFrame } from "@react-three/fiber";
import { OrbitControls, Stage, Environment, Grid, PerspectiveCamera } from "@react-three/drei";
import { cn } from "@/lib/utils";
import {
  Loader2,
  AlertCircle,
  RotateCcw,
  Box,
  Sun,
  Moon,
  Grid3X3,
  Download,
} from "lucide-react";
import * as THREE from "three";
import { VRMLLoader } from "three/examples/jsm/loaders/VRMLLoader.js";

// ============================================================================
// Types
// ============================================================================

interface ThreeDViewerProps {
  /** URL to the 3D model file (VRML/WRL or GLTF) */
  modelUrl: string;
  /** Format of the 3D model */
  format: "vrml" | "wrl" | "gltf" | "glb" | "step";
  /** Optional class name */
  className?: string;
  /** Callback when model is loaded */
  onLoad?: () => void;
  /** Callback on error */
  onError?: (error: string) => void;
}

interface ModelProps {
  url: string;
  format: string;
  onLoad?: () => void;
  onError?: (error: string) => void;
}

// ============================================================================
// VRML Model Loader Component
// ============================================================================

function VRMLModel({ url, onLoad, onError }: ModelProps) {
  const [scene, setScene] = useState<THREE.Object3D | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const loader = new VRMLLoader();

    loader.load(
      url,
      (loadedScene) => {
        // Center the model
        const box = new THREE.Box3().setFromObject(loadedScene);
        const center = box.getCenter(new THREE.Vector3());
        loadedScene.position.sub(center);

        // Scale to reasonable size
        const size = box.getSize(new THREE.Vector3());
        const maxDim = Math.max(size.x, size.y, size.z);
        if (maxDim > 100) {
          const scale = 100 / maxDim;
          loadedScene.scale.multiplyScalar(scale);
        }

        setScene(loadedScene as THREE.Object3D);
        onLoad?.();
      },
      undefined,
      (err) => {
        const errorMsg = `Failed to load VRML model: ${err instanceof Error ? err.message : "Unknown error"}`;
        setError(errorMsg);
        onError?.(errorMsg);
      }
    );

    return () => {
      if (scene) {
        scene.traverse((child) => {
          if (child instanceof THREE.Mesh) {
            child.geometry?.dispose();
            if (child.material instanceof THREE.Material) {
              child.material.dispose();
            } else if (Array.isArray(child.material)) {
              child.material.forEach((m) => m.dispose());
            }
          }
        });
      }
    };
  }, [url, onLoad, onError, scene]);

  if (error) {
    return null;
  }

  if (!scene) {
    return null;
  }

  return <primitive object={scene} />;
}

// ============================================================================
// Placeholder Model (for STEP files that need conversion)
// ============================================================================

function PlaceholderPCB() {
  const meshRef = useRef<THREE.Mesh>(null);

  useFrame((state) => {
    if (meshRef.current) {
      meshRef.current.rotation.y = Math.sin(state.clock.elapsedTime * 0.5) * 0.1;
    }
  });

  return (
    <group>
      {/* PCB Board */}
      <mesh ref={meshRef} position={[0, 0, 0]}>
        <boxGeometry args={[100, 80, 1.6]} />
        <meshStandardMaterial color="#1a472a" metalness={0.1} roughness={0.8} />
      </mesh>

      {/* Sample components */}
      {[...Array(8)].map((_, i) => (
        <mesh
          key={i}
          position={[
            (Math.random() - 0.5) * 80,
            (Math.random() - 0.5) * 60,
            1.5,
          ]}
        >
          <boxGeometry args={[8, 8, 3]} />
          <meshStandardMaterial color="#2a2a2a" metalness={0.5} roughness={0.3} />
        </mesh>
      ))}

      {/* Copper traces (simplified) */}
      <mesh position={[0, 0, 0.85]}>
        <planeGeometry args={[95, 75]} />
        <meshStandardMaterial
          color="#cd7f32"
          metalness={0.8}
          roughness={0.2}
          opacity={0.3}
          transparent
        />
      </mesh>
    </group>
  );
}

// ============================================================================
// Camera Controls Component
// ============================================================================

function CameraController() {
  const { camera } = useThree();

  useEffect(() => {
    camera.position.set(150, 100, 150);
    camera.lookAt(0, 0, 0);
  }, [camera]);

  return null;
}

// ============================================================================
// Loading Fallback
// ============================================================================

function LoadingFallback() {
  return (
    <mesh>
      <sphereGeometry args={[10, 32, 32]} />
      <meshBasicMaterial color="#4a90d9" wireframe />
    </mesh>
  );
}

// ============================================================================
// Main Component
// ============================================================================

export function ThreeDViewer({
  modelUrl,
  format,
  className,
  onLoad,
  onError,
}: ThreeDViewerProps) {
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [showGrid, setShowGrid] = useState(true);
  const [darkMode, setDarkMode] = useState(true);
  const controlsRef = useRef<any>(null);

  const handleLoad = useCallback(() => {
    setIsLoading(false);
    onLoad?.();
  }, [onLoad]);

  const handleError = useCallback(
    (errorMsg: string) => {
      setError(errorMsg);
      setIsLoading(false);
      onError?.(errorMsg);
    },
    [onError]
  );

  const handleResetView = useCallback(() => {
    controlsRef.current?.reset();
  }, []);

  const isVRML = format === "vrml" || format === "wrl";
  const isSTEP = format === "step";

  // Error state
  if (error) {
    return (
      <div className={cn("flex items-center justify-center h-full bg-slate-900", className)}>
        <div className="text-center p-6">
          <AlertCircle className="w-12 h-12 text-red-500 mx-auto mb-4" />
          <h3 className="text-lg font-medium text-white mb-2">Failed to Load 3D Model</h3>
          <p className="text-sm text-slate-400 max-w-md">{error}</p>
          <button
            onClick={() => {
              setError(null);
              setIsLoading(true);
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
    <div className={cn("relative h-full", className)}>
      {/* Loading overlay */}
      {isLoading && (
        <div className="absolute inset-0 flex items-center justify-center bg-slate-900/80 z-10">
          <div className="text-center">
            <Loader2 className="w-10 h-10 text-primary-500 animate-spin mx-auto mb-3" />
            <p className="text-sm text-slate-400">Loading 3D model...</p>
          </div>
        </div>
      )}

      {/* Toolbar */}
      <div className="absolute top-2 right-2 z-20 flex gap-1 bg-slate-800/80 rounded-lg p-1">
        <button
          onClick={handleResetView}
          className="p-2 hover:bg-slate-700 rounded-md transition-colors"
          title="Reset View"
        >
          <RotateCcw className="w-4 h-4 text-slate-300" />
        </button>
        <button
          onClick={() => setShowGrid(!showGrid)}
          className={cn(
            "p-2 rounded-md transition-colors",
            showGrid ? "bg-slate-700" : "hover:bg-slate-700"
          )}
          title="Toggle Grid"
        >
          <Grid3X3 className="w-4 h-4 text-slate-300" />
        </button>
        <button
          onClick={() => setDarkMode(!darkMode)}
          className="p-2 hover:bg-slate-700 rounded-md transition-colors"
          title="Toggle Theme"
        >
          {darkMode ? (
            <Sun className="w-4 h-4 text-slate-300" />
          ) : (
            <Moon className="w-4 h-4 text-slate-300" />
          )}
        </button>
        {modelUrl && (
          <a
            href={modelUrl}
            download
            className="p-2 hover:bg-slate-700 rounded-md transition-colors"
            title="Download Model"
          >
            <Download className="w-4 h-4 text-slate-300" />
          </a>
        )}
      </div>

      {/* Format indicator */}
      <div className="absolute top-2 left-2 z-20 px-2 py-1 bg-slate-800/80 rounded text-xs text-slate-400">
        <Box className="w-3 h-3 inline-block mr-1" />
        {format.toUpperCase()}
        {isSTEP && " (Preview)"}
      </div>

      {/* Three.js Canvas */}
      <Canvas
        shadows
        gl={{ antialias: true, alpha: true }}
        dpr={[1, 2]}
        onCreated={() => {
          // Canvas is ready
          setTimeout(() => setIsLoading(false), 500);
        }}
      >
        <CameraController />

        {/* Lighting */}
        <ambientLight intensity={darkMode ? 0.3 : 0.5} />
        <directionalLight
          position={[50, 50, 50]}
          intensity={darkMode ? 0.8 : 1.2}
          castShadow
          shadow-mapSize={[1024, 1024]}
        />
        <directionalLight position={[-50, 50, -50]} intensity={0.3} />

        {/* Environment */}
        <Environment preset={darkMode ? "night" : "studio"} />

        {/* Grid */}
        {showGrid && (
          <Grid
            args={[200, 200]}
            cellSize={10}
            cellThickness={0.5}
            cellColor={darkMode ? "#404040" : "#808080"}
            sectionSize={50}
            sectionThickness={1}
            sectionColor={darkMode ? "#606060" : "#a0a0a0"}
            fadeDistance={300}
            fadeStrength={1}
            followCamera={false}
            infiniteGrid
          />
        )}

        {/* Model */}
        <Suspense fallback={<LoadingFallback />}>
          {isVRML && modelUrl ? (
            <VRMLModel
              url={modelUrl}
              format={format}
              onLoad={handleLoad}
              onError={handleError}
            />
          ) : (
            <PlaceholderPCB />
          )}
        </Suspense>

        {/* Controls */}
        <OrbitControls
          ref={controlsRef}
          makeDefault
          enableDamping
          dampingFactor={0.1}
          minDistance={50}
          maxDistance={500}
          maxPolarAngle={Math.PI * 0.85}
        />

        {/* Background */}
        <color attach="background" args={[darkMode ? "#0f172a" : "#f1f5f9"]} />
      </Canvas>

      {/* Info panel */}
      <div className="absolute bottom-2 left-2 z-20 text-xs text-slate-500">
        <p>
          Drag to rotate | Scroll to zoom | Shift+drag to pan
        </p>
      </div>
    </div>
  );
}

export default ThreeDViewer;
