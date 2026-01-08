/**
 * PCB Layout Agents Index
 *
 * Exports all 5 competing layout agents for the Ralph Loop tournament.
 */

export { BaseLayoutAgent } from './base-agent';
export type {
  AgentConfig,
  AgentStrategy,
  StrategyWeights,
  AgentParameters,
  PlacementContext,
  AgentFeedback,
  PlacementResult,
  LayoutMetrics
} from './base-agent';

export { ConservativeAgent } from './conservative-agent';
export { AggressiveCompactAgent } from './aggressive-compact-agent';
export { ThermalOptimizedAgent } from './thermal-optimized-agent';
export { EMIOptimizedAgent } from './emi-optimized-agent';
export { DFMOptimizedAgent } from './dfm-optimized-agent';

import { BaseLayoutAgent } from './base-agent';
import { ConservativeAgent } from './conservative-agent';
import { AggressiveCompactAgent } from './aggressive-compact-agent';
import { ThermalOptimizedAgent } from './thermal-optimized-agent';
import { EMIOptimizedAgent } from './emi-optimized-agent';
import { DFMOptimizedAgent } from './dfm-optimized-agent';

/**
 * Agent factory - creates all agents for tournament
 */
export function createAllAgents(): BaseLayoutAgent[] {
  return [
    new ConservativeAgent(),
    new AggressiveCompactAgent(),
    new ThermalOptimizedAgent(),
    new EMIOptimizedAgent(),
    new DFMOptimizedAgent()
  ];
}

/**
 * Agent map by strategy name
 */
export const AgentRegistry = {
  'conservative': ConservativeAgent,
  'aggressive-compact': AggressiveCompactAgent,
  'thermal-optimized': ThermalOptimizedAgent,
  'emi-optimized': EMIOptimizedAgent,
  'dfm-optimized': DFMOptimizedAgent
} as const;

export type AgentStrategyName = keyof typeof AgentRegistry;

/**
 * Create a specific agent by strategy name
 */
export function createAgent(strategy: AgentStrategyName): BaseLayoutAgent {
  const AgentClass = AgentRegistry[strategy];
  return new AgentClass();
}