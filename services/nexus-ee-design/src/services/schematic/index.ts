/**
 * Schematic Services Index
 *
 * Exports schematic generation and review services.
 */

export { SchematicGenerator, default } from './schematic-generator';
export type {
  SchematicGeneratorConfig,
  SchematicRequirements,
  PowerRequirements,
  VoltageRail,
  InterfaceRequirement,
  SchematicConstraints,
  GenerationResult,
  BOMEntry,
  NetlistEntry,
  ComponentTemplate,
  ComponentCategory,
  PinDefinition,
  SchematicBlock,
  BlockType,
  BlockConnection
} from './schematic-generator';

export { SchematicReviewer } from './schematic-reviewer';
export type {
  ReviewConfig,
  CustomRule,
  RuleViolation,
  ReviewResult,
  ERCRule
} from './schematic-reviewer';
