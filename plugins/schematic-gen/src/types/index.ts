/**
 * SKiDL Schematic Generator Plugin - Type Definitions
 *
 * Production-ready type definitions for the schematic generation pipeline.
 * These types enforce strict validation and provide comprehensive error handling.
 */

import { z } from 'zod';

// ============================================================================
// Validation Result Types
// ============================================================================

export enum ValidationLevel {
  SKIDL_ERC = 1,
  KICAD_SCH_API = 2,
  KICAD_CLI = 3,
  SPICE = 4
}

export interface ValidationError {
  code: string;
  message: string;
  severity: 'error' | 'warning' | 'info';
  location?: {
    file?: string;
    line?: number;
    component?: string;
    net?: string;
    pin?: string;
  };
  suggestion?: string;
}

export interface ValidationMetrics {
  componentCount: number;
  wireCount: number;
  netCount: number;
  wireComponentRatio: number;
  unconnectedPins: number;
  powerNets: number;
  signalNets: number;
}

export interface LevelValidationResult {
  level: ValidationLevel;
  levelName: string;
  passed: boolean;
  errors: ValidationError[];
  warnings: ValidationError[];
  metrics: Partial<ValidationMetrics>;
  duration: number;
  timestamp: string;
}

export interface ValidationPipelineResult {
  success: boolean;
  levels: LevelValidationResult[];
  overallScore: number;
  expertReviews: ExpertReviewResult[];
  summary: string;
}

// ============================================================================
// Expert Agent Types
// ============================================================================

export interface ExpertCheck {
  id: string;
  description: string;
  category: string;
  passed: boolean;
  details?: string;
  severity: 'critical' | 'major' | 'minor';
}

export interface ExpertReviewResult {
  expertId: string;
  expertName: string;
  role: string;
  checks: ExpertCheck[];
  passed: boolean;
  score: number;
  recommendations: string[];
  timestamp: string;
}

export interface ExpertAgentConfig {
  id: string;
  name: string;
  role: string;
  expertise: string[];
  checks: ExpertCheckDefinition[];
  validationFocus: Record<string, string[]>;
}

export interface ExpertCheckDefinition {
  id: string;
  description: string;
  category: string;
  severity: 'critical' | 'major' | 'minor';
  validator: (schematic: SchematicData) => ExpertCheckResult;
}

export interface ExpertCheckResult {
  passed: boolean;
  details?: string;
  evidence?: Record<string, unknown>;
}

// ============================================================================
// Schematic Data Types
// ============================================================================

export interface Position2D {
  x: number;
  y: number;
}

export interface ComponentData {
  reference: string;
  value: string;
  symbol: string;
  footprint: string;
  position: Position2D;
  rotation: number;
  properties: Record<string, string>;
  pins: PinData[];
  manufacturer?: string;
  partNumber?: string;
}

export interface PinData {
  number: string;
  name: string;
  type: PinType;
  position: Position2D;
  connected: boolean;
  netName?: string;
}

export type PinType =
  | 'input'
  | 'output'
  | 'bidirectional'
  | 'tri_state'
  | 'passive'
  | 'power_input'
  | 'power_output'
  | 'open_collector'
  | 'open_emitter'
  | 'no_connect';

export interface NetData {
  name: string;
  connections: NetConnection[];
  netClass?: string;
  properties: NetProperties;
}

export interface NetConnection {
  componentRef: string;
  pinNumber: string;
  pinName: string;
}

export interface NetProperties {
  impedance?: number;
  maxCurrent?: number;
  maxVoltage?: number;
  differentialPair?: string;
  netType?: 'power' | 'ground' | 'signal' | 'differential';
}

export interface WireData {
  startPoint: Position2D;
  endPoint: Position2D;
  netName: string;
}

export interface LabelData {
  text: string;
  position: Position2D;
  type: 'local' | 'global' | 'hierarchical' | 'power';
}

export interface SchematicSheet {
  id: string;
  name: string;
  pageNumber: number;
  components: ComponentData[];
  nets: NetData[];
  wires: WireData[];
  labels: LabelData[];
  metadata: SheetMetadata;
}

export interface SheetMetadata {
  title: string;
  revision: string;
  date: string;
  author: string;
  company: string;
  comments: string[];
}

export interface SchematicData {
  id: string;
  projectName: string;
  version: string;
  sheets: SchematicSheet[];
  metadata: ProjectMetadata;
  netlist?: NetlistData;
}

export interface ProjectMetadata {
  title: string;
  description: string;
  revision: string;
  date: string;
  author: string;
  company: string;
  targetApplication: string;
}

export interface NetlistData {
  format: 'kicad' | 'spice' | 'pcb';
  components: NetlistComponent[];
  nets: NetlistNet[];
}

export interface NetlistComponent {
  reference: string;
  value: string;
  footprint: string;
  pins: Array<{ number: string; name: string; net: string }>;
}

export interface NetlistNet {
  name: string;
  code: number;
  connections: Array<{ ref: string; pin: string }>;
}

// ============================================================================
// Generation Configuration Types
// ============================================================================

export interface GenerationConfig {
  projectDir: string;
  outputDir: string;
  sheets: SheetConfig[];
  validationLevels: ValidationLevel[];
  enableExpertReview: boolean;
  maxIterations: number;
  targetScore: number;
}

export interface SheetConfig {
  sheetNumber: number;
  name: string;
  skidlModule: string;
  template?: string;
}

export interface SKiDLCircuitConfig {
  name: string;
  description: string;
  libraries: string[];
  subcircuits: SubcircuitConfig[];
  topLevel: TopLevelConfig;
}

export interface SubcircuitConfig {
  name: string;
  function: string;
  parameters: Record<string, unknown>;
}

export interface TopLevelConfig {
  powerNets: string[];
  groundNets: string[];
  interfaceNets: string[];
}

// ============================================================================
// Event Types
// ============================================================================

export type GenerationEvent =
  | { type: 'start'; config: GenerationConfig }
  | { type: 'phase'; phase: string; progress: number }
  | { type: 'sheet_generated'; sheetNumber: number; name: string }
  | { type: 'validation_level'; level: ValidationLevel; result: LevelValidationResult }
  | { type: 'expert_review'; expertId: string; result: ExpertReviewResult }
  | { type: 'complete'; result: GenerationResult }
  | { type: 'error'; error: GenerationError };

export interface GenerationResult {
  success: boolean;
  schematic?: SchematicData;
  filePath?: string;
  validation: ValidationPipelineResult;
  duration: number;
  timestamp: string;
}

export interface GenerationError {
  code: string;
  message: string;
  phase: string;
  stack?: string;
  context?: Record<string, unknown>;
}

// ============================================================================
// Zod Schemas for Runtime Validation
// ============================================================================

export const Position2DSchema = z.object({
  x: z.number(),
  y: z.number()
});

export const ValidationErrorSchema = z.object({
  code: z.string(),
  message: z.string(),
  severity: z.enum(['error', 'warning', 'info']),
  location: z.object({
    file: z.string().optional(),
    line: z.number().optional(),
    component: z.string().optional(),
    net: z.string().optional(),
    pin: z.string().optional()
  }).optional(),
  suggestion: z.string().optional()
});

export const GenerationConfigSchema = z.object({
  projectDir: z.string().min(1),
  outputDir: z.string().min(1),
  sheets: z.array(z.object({
    sheetNumber: z.number().int().positive(),
    name: z.string().min(1),
    skidlModule: z.string().min(1),
    template: z.string().optional()
  })),
  validationLevels: z.array(z.nativeEnum(ValidationLevel)),
  enableExpertReview: z.boolean(),
  maxIterations: z.number().int().positive().max(1000),
  targetScore: z.number().min(0).max(100)
});

// Type guard functions
export function isValidationError(obj: unknown): obj is ValidationError {
  return ValidationErrorSchema.safeParse(obj).success;
}

export function isGenerationConfig(obj: unknown): obj is GenerationConfig {
  return GenerationConfigSchema.safeParse(obj).success;
}

// ============================================================================
// Constants
// ============================================================================

export const VALIDATION_THRESHOLDS = {
  WIRE_COMPONENT_RATIO_MIN: 1.2,
  WIRE_COMPONENT_RATIO_TARGET: 1.5,
  WIRE_COMPONENT_RATIO_CRITICAL: 1.0,
  MAX_UNCONNECTED_PINS: 0,
  MAX_ERC_ERRORS: 0,
  MIN_SPEC_COVERAGE: 90
} as const;

export const EXPERT_WEIGHTS = {
  POWER_ELECTRONICS: 0.35,
  SIGNAL_INTEGRITY: 0.35,
  VALIDATION: 0.30
} as const;