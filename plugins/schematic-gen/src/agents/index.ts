/**
 * Expert Agents Index
 *
 * Exports all expert agent modules for the schematic validation pipeline.
 */

export {
  POWER_ELECTRONICS_EXPERT,
  runPowerElectronicsReview
} from './power-electronics-expert';

export {
  SIGNAL_INTEGRITY_EXPERT,
  runSignalIntegrityReview
} from './signal-integrity-expert';

export {
  VALIDATION_EXPERT,
  runValidationReview
} from './validation-expert';

import { runPowerElectronicsReview } from './power-electronics-expert';
import { runSignalIntegrityReview } from './signal-integrity-expert';
import { runValidationReview } from './validation-expert';
import {
  SchematicData,
  ExpertReviewResult,
  EXPERT_WEIGHTS
} from '../types';

/**
 * Run all expert reviews and compute weighted score
 */
export function runAllExpertReviews(schematic: SchematicData): {
  reviews: ExpertReviewResult[];
  overallScore: number;
  passed: boolean;
} {
  const reviews: ExpertReviewResult[] = [];

  // Run each expert review
  const powerReview = runPowerElectronicsReview(schematic);
  reviews.push(powerReview);

  const siReview = runSignalIntegrityReview(schematic);
  reviews.push(siReview);

  const validationReview = runValidationReview(schematic);
  reviews.push(validationReview);

  // Calculate weighted overall score
  const overallScore =
    powerReview.score * EXPERT_WEIGHTS.POWER_ELECTRONICS +
    siReview.score * EXPERT_WEIGHTS.SIGNAL_INTEGRITY +
    validationReview.score * EXPERT_WEIGHTS.VALIDATION;

  // Overall pass requires all experts to pass their critical checks
  const passed = reviews.every(r => r.passed);

  return {
    reviews,
    overallScore,
    passed
  };
}

export default {
  runAllExpertReviews,
  runPowerElectronicsReview,
  runSignalIntegrityReview,
  runValidationReview
};
