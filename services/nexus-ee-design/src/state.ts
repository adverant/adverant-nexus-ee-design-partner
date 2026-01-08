/**
 * Shared Application State
 *
 * Holds global singleton instances that need to be shared across modules.
 * This module breaks circular dependencies between index.ts and routes.
 */

import { SkillsEngineClient } from './services/skills/skills-engine-client';

// Global skills engine client instance
let _skillsEngineClient: SkillsEngineClient | null = null;

/**
 * Get the skills engine client instance
 */
export function getSkillsEngineClient(): SkillsEngineClient | null {
  return _skillsEngineClient;
}

/**
 * Set the skills engine client instance
 */
export function setSkillsEngineClient(client: SkillsEngineClient): void {
  _skillsEngineClient = client;
}

/**
 * Clear the skills engine client instance
 */
export function clearSkillsEngineClient(): void {
  _skillsEngineClient = null;
}
