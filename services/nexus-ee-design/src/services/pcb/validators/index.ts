/**
 * PCB Validators Index
 *
 * Exports the validation framework and types for PCB layout validation.
 */

export {
  ValidationFramework,
  default
} from './validation-framework';

export type {
  ValidationDomain,
  ValidationViolation,
  DomainResult,
  ValidationConfig
} from './validation-framework';