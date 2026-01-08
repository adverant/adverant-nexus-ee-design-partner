/**
 * Firmware Services Index
 *
 * Exports firmware generation and related types.
 */

export { FirmwareGenerator, default } from './firmware-generator';
export type {
  FirmwareGeneratorConfig,
  FirmwareRequirements,
  PeripheralRequirement,
  FeatureRequirement,
  GenerationResult,
  HALTemplate,
  PeripheralTemplate,
  DriverTemplate,
  DriverFunctionTemplate
} from './firmware-generator';
