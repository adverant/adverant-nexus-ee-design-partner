/**
 * Symbol Assembly API Routes (MAPO v3.0)
 *
 * Handles symbol gathering, datasheet download, and characterization
 * before schematic generation. Searches GraphRAG first, then external sources.
 *
 * Endpoints:
 * - POST /api/v1/projects/:projectId/symbol-assembly/gather - Trigger symbol assembly
 * - GET /api/v1/projects/:projectId/symbol-assembly/active - List active operations
 * - DELETE /api/v1/projects/:projectId/symbol-assembly/active - Cancel all active operations
 * - GET /api/v1/projects/:projectId/symbol-assembly/:operationId - Get assembly report
 * - DELETE /api/v1/projects/:projectId/symbol-assembly/:operationId - Cancel specific operation
 * - GET /api/v1/projects/:projectId/symbol-assembly/:operationId/symbols - List gathered symbols
 * - GET /api/v1/projects/:projectId/symbol-assembly/:operationId/datasheets - List datasheets
 * - GET /api/v1/projects/:projectId/symbol-assembly/:operationId/download/:filename - Download file
 */

import { Router, Request, Response, NextFunction } from 'express';
import { spawn, ChildProcess } from 'child_process';
import { createHash } from 'crypto';
import { z } from 'zod';
import path from 'path';
import fs from 'fs/promises';
import { ValidationError, NotFoundError } from '../../utils/errors.js';
import { log } from '../../utils/logger.js';
import { config } from '../../config.js';
import { getSchematicWsManager } from '../schematic-ws.js';
import { SchematicEventType } from '../websocket-schematic.js';

/**
 * Process registry: tracks spawned Python processes by operationId
 * so they can be cancelled via DELETE endpoint.
 */
const activeProcesses: Map<string, { process: ChildProcess; projectId: string; startedAt: Date }> = new Map();

/**
 * Validation schemas
 */
const GatherSymbolsSchema = z.object({
  body: z.object({
    ideationArtifacts: z.array(z.object({
      type: z.string(),
      content: z.string(),
    })),
    operationId: z.string().optional(),
    forceRefresh: z.boolean().optional(),
  }),
});

/**
 * Compute a stable hash of artifact content to detect when ideation data has changed.
 * Used to auto-invalidate stale cached assembly reports.
 */
function computeArtifactHash(artifacts: Array<{ type?: string; content?: string }>): string {
  const hash = createHash('sha256');
  for (const a of artifacts) {
    hash.update(a.type || '');
    hash.update('|');
    hash.update(a.content || '');
    hash.update('\n');
  }
  return hash.digest('hex').slice(0, 16);
}

/**
 * Transform a snake_case assembly report from the Python backend
 * into the camelCase format expected by the frontend TypeScript interfaces.
 */
function transformReportToCamelCase(raw: Record<string, unknown>): Record<string, unknown> {
  // Map report-level snake_case fields to camelCase
  const report: Record<string, unknown> = {
    operationId: raw.operationId ?? raw.operation_id,
    projectId: raw.projectId ?? raw.project_id,
    startedAt: raw.startedAt ?? raw.started_at,
    completedAt: raw.completedAt ?? raw.completed_at,
    status: raw.status,
    totalComponents: raw.totalComponents ?? raw.total_components ?? 0,
    symbolsFound: raw.symbolsFound ?? raw.symbols_found ?? 0,
    symbolsFromGraphrag: raw.symbolsFromGraphrag ?? raw.symbols_from_graphrag ?? 0,
    symbolsFromKicad: raw.symbolsFromKicad ?? raw.symbols_from_kicad ?? 0,
    symbolsFromSnapeda: raw.symbolsFromSnapeda ?? raw.symbols_from_snapeda ?? 0,
    symbolsFromUltralibrarian: raw.symbolsFromUltralibrarian ?? raw.symbols_from_ultralibrarian ?? 0,
    symbolsLlmGenerated: raw.symbolsLlmGenerated ?? raw.symbols_llm_generated ?? 0,
    datasheetsDownloaded: raw.datasheetsDownloaded ?? raw.datasheets_downloaded ?? 0,
    characterizationsCreated: raw.characterizationsCreated ?? raw.characterizations_created ?? 0,
    errorsCount: raw.errorsCount ?? raw.errors_count ?? 0,
    artifactHash: raw.artifactHash ?? raw.artifact_hash,
    errors: raw.errors ?? [],
    success: raw.success,
  };

  // Transform each component in the components array
  const rawComponents = raw.components as Array<Record<string, unknown>> | undefined;
  if (Array.isArray(rawComponents)) {
    report.components = rawComponents.map((c) => ({
      partNumber: c.partNumber ?? c.part_number,
      manufacturer: c.manufacturer,
      package: c.package,
      category: c.category,
      symbolFound: c.symbolFound ?? c.symbol_found,
      symbolSource: c.symbolSource ?? c.symbol_source,
      symbolPath: c.symbolPath ?? c.symbol_path,
      datasheetFound: c.datasheetFound ?? c.datasheet_found,
      datasheetPath: c.datasheetPath ?? c.datasheet_path,
      characterizationCreated: c.characterizationCreated ?? c.characterization_created,
      characterizationPath: c.characterizationPath ?? c.characterization_path,
      errors: c.errors,
      success: c.success,
    }));
  } else {
    report.components = [];
  }

  return report;
}

/**
 * Assembly report structure
 */
interface AssemblyReport {
  operationId: string;
  projectId: string;
  startedAt: string;
  completedAt?: string;
  status: 'running' | 'complete' | 'error';
  totalComponents: number;
  symbolsFound: number;
  symbolsGenerated: number;
  datasheetsDownloaded: number;
  characterizationsCreated: number;
  errors: Array<{
    component: string;
    manufacturer?: string;
    partNumber?: string;
    message: string;
    sources: string[];
  }>;
  components: Array<{
    partNumber: string;
    manufacturer: string;
    package: string;
    category: string;
    symbolSource: 'graphrag' | 'kicad' | 'snapeda' | 'ultralibrarian' | 'llm-generated' | 'not-found';
    symbolPath?: string;
    datasheetSource?: string;
    datasheetPath?: string;
    characterizationPath?: string;
  }>;
}

/**
 * Create symbol assembly routes
 */
export function createSymbolAssemblyRoutes(): Router {
  const router = Router({ mergeParams: true });

  /**
   * Trigger symbol assembly
   * POST /api/v1/projects/:projectId/symbol-assembly/gather
   */
  router.post('/gather', async (req: Request, res: Response, next: NextFunction) => {
    try {
      const { projectId } = req.params;
      const validation = GatherSymbolsSchema.safeParse({ body: req.body });

      if (!validation.success) {
        throw new ValidationError('Invalid request body', {
          operation: 'gatherSymbols',
          errors: validation.error.errors,
        });
      }

      const { ideationArtifacts, operationId: providedOperationId, forceRefresh } = validation.data.body;

      // Generate operation ID if not provided
      const { v4: uuidv4 } = await import('uuid');
      const operationId = providedOperationId || uuidv4();

      // Compute content hash for cache invalidation
      const artifactHash = computeArtifactHash(ideationArtifacts);

      log.info('Starting symbol assembly', {
        projectId, operationId,
        artifactCount: ideationArtifacts.length,
        artifactHash,
        forceRefresh: !!forceRefresh,
      });

      // Get WebSocket manager for streaming progress
      const wsManager = getSchematicWsManager();

      // Register operation in WebSocket manager so events aren't dropped
      wsManager.createOperation(projectId, operationId);

      // Create output directory
      const outputDir = path.join(config.artifacts.basePath, projectId, 'symbol-assembly');
      await fs.mkdir(outputDir, { recursive: true });

      // Write artifacts to temp file for Python script
      const artifactsPath = path.join(outputDir, `artifacts-${operationId}.json`);
      await fs.writeFile(artifactsPath, JSON.stringify(ideationArtifacts, null, 2));

      // Spawn Python symbol assembler
      const pythonPath = config.kicad.pythonPath || 'python3';
      const scriptPath = path.join(
        config.kicad.scriptsDir,
        'agents/symbol_assembly/symbol_assembler.py'
      );

      // Fail fast: verify script exists before spawning async process
      try {
        await fs.access(scriptPath);
      } catch {
        throw new Error(
          `Symbol assembler script not found at ${scriptPath}. ` +
          `config.kicad.scriptsDir=${config.kicad.scriptsDir}`
        );
      }

      // Fetch user's external API keys from nexus-auth (BYOK) and merge
      // with system env vars. User keys take precedence over system defaults.
      const externalKeyEnv: Record<string, string> = {};
      const userId = (req.headers['x-user-id'] as string) || '';
      let orgId = (req.headers['x-organization-id'] as string) || '';
      const authToken = req.headers['authorization'] as string || '';

      // Multi-tenant: resolve organization ID from nexus-auth when not provided
      // in request headers. The JWT token doesn't include org_id, so we must
      // query the organizations endpoint to get the user's org for BYOK key isolation.
      if (!orgId && authToken) {
        try {
          const authBaseUrl = process.env.NEXUS_AUTH_URL || 'http://nexus-auth.nexus.svc.cluster.local:9101';
          const orgsRes = await fetch(`${authBaseUrl}/organizations`, {
            headers: {
              'Authorization': authToken,
              'X-Service-Name': 'nexus-ee-design',
              'X-Internal-Request': 'true',
            },
            signal: AbortSignal.timeout(5000),
          });

          if (orgsRes.ok) {
            const orgsData = await orgsRes.json() as {
              organizations?: Array<{ id?: string; ID?: string; name?: string; Name?: string }>;
            };
            const organizations = orgsData.organizations || [];

            if (organizations.length > 0) {
              const org = organizations[0];
              orgId = org.id || org.ID || '';
              log.info('Resolved organization from auth token for BYOK', {
                operationId,
                organizationId: orgId,
                organizationName: org.name || org.Name,
                organizationCount: organizations.length,
              });
            } else {
              log.warn('User has no organizations — BYOK keys unavailable', { operationId });
            }
          } else {
            log.warn('Failed to query organizations endpoint', {
              operationId,
              status: orgsRes.status,
            });
          }
        } catch (orgErr) {
          log.warn('Error resolving organization from auth token', {
            operationId,
            error: (orgErr as Error).message,
          });
        }
      }

      if (orgId && authToken) {
        try {
          const authBaseUrl = process.env.NEXUS_AUTH_URL || 'http://nexus-auth.nexus.svc.cluster.local:9101';
          const wantedProviders = ['nexar', 'snapeda', 'ultralibrarian', 'digikey', 'mouser'];

          // Step 1: List organization keys to get key IDs and provider mappings
          const listRes = await fetch(
            `${authBaseUrl}/auth/external-keys/organizations/${orgId}`,
            {
              headers: {
                'Authorization': authToken,
                'X-Service-Name': 'nexus-ee-design',
                'X-Internal-Request': 'true',
              },
            },
          );

          if (listRes.ok) {
            const listData = await listRes.json() as {
              organization_keys?: Record<string, {
                id: string;
                provider_key: string;
                masked_key?: string;
              }>;
            };
            const orgKeys = listData.organization_keys || {};

            // Step 2: Decrypt each relevant key individually
            for (const providerKey of wantedProviders) {
              const keyEntry = orgKeys[providerKey];
              if (!keyEntry?.id) continue;

              try {
                const decryptRes = await fetch(
                  `${authBaseUrl}/auth/external-keys/${keyEntry.id}/decrypt`,
                  {
                    method: 'POST',
                    headers: {
                      'Authorization': authToken,
                      'Content-Type': 'application/json',
                      'X-Service-Name': 'nexus-ee-design',
                      'X-Internal-Request': 'true',
                    },
                    body: JSON.stringify({ organization_id: orgId }),
                  },
                );

                if (!decryptRes.ok) {
                  log.warn('Failed to decrypt external key', {
                    operationId,
                    providerKey,
                    status: decryptRes.status,
                  });
                  continue;
                }

                const decryptData = await decryptRes.json() as {
                  api_key?: string;
                  provider_key?: string;
                };
                const rawValue = decryptData.api_key || '';
                if (!rawValue) continue;

                // Map provider keys to environment variables the Python script expects
                const mask = (v: string) => v.length > 6 ? `${v.slice(0, 4)}...${v.slice(-2)}(${v.length}c)` : `***`;
                let parseMethod = 'direct';

                switch (providerKey) {
                  case 'nexar': {
                    // Nexar stores OAuth credentials as JSON: {"client_id":"...","client_secret":"..."}
                    try {
                      const parsed = JSON.parse(rawValue);
                      if (parsed.client_id) externalKeyEnv.NEXAR_CLIENT_ID = parsed.client_id;
                      if (parsed.client_secret) externalKeyEnv.NEXAR_CLIENT_SECRET = parsed.client_secret;
                      parseMethod = 'json';
                    } catch {
                      // Fallback: colon-separated format  client_id:client_secret
                      if (rawValue.includes(':')) {
                        const [cid, csec] = rawValue.split(':', 2);
                        externalKeyEnv.NEXAR_CLIENT_ID = cid;
                        externalKeyEnv.NEXAR_CLIENT_SECRET = csec;
                        parseMethod = 'colon-split';
                      } else {
                        externalKeyEnv.NEXAR_CLIENT_ID = rawValue;
                        parseMethod = 'raw-as-client-id';
                      }
                    }
                    break;
                  }
                  case 'snapeda':
                    externalKeyEnv.SNAPEDA_API_KEY = rawValue;
                    break;
                  case 'ultralibrarian':
                    externalKeyEnv.ULTRALIBRARIAN_API_KEY = rawValue;
                    break;
                  case 'digikey': {
                    // DigiKey stores OAuth credentials as JSON: {"client_id":"...","client_secret":"..."}
                    try {
                      const parsed = JSON.parse(rawValue);
                      if (parsed.client_id) externalKeyEnv.DIGIKEY_CLIENT_ID = parsed.client_id;
                      if (parsed.client_secret) externalKeyEnv.DIGIKEY_CLIENT_SECRET = parsed.client_secret;
                      parseMethod = 'json';
                    } catch {
                      if (rawValue.includes(':')) {
                        const [cid, csec] = rawValue.split(':', 2);
                        externalKeyEnv.DIGIKEY_CLIENT_ID = cid;
                        externalKeyEnv.DIGIKEY_CLIENT_SECRET = csec;
                        parseMethod = 'colon-split';
                      } else {
                        parseMethod = 'parse-failed-no-colon';
                      }
                    }
                    break;
                  }
                  case 'mouser':
                    externalKeyEnv.MOUSER_API_KEY = rawValue;
                    break;
                }

                // Diagnostic: log parse result with masked values for debugging credential issues
                const envKeys = Object.entries(externalKeyEnv)
                  .filter(([, v]) => v)
                  .map(([k, v]) => `${k}=${mask(v)}`);
                log.info('Decrypted external key for symbol assembly', {
                  operationId,
                  providerKey,
                  parseMethod,
                  rawValueLength: rawValue.length,
                  extractedKeys: envKeys.slice(-2), // last 2 are the ones just set
                });
              } catch (decryptErr) {
                log.warn('Error decrypting external key', {
                  operationId,
                  providerKey,
                  error: (decryptErr as Error).message,
                });
              }
            }

            if (Object.keys(externalKeyEnv).length > 0) {
              log.info('Loaded user external keys for symbol assembly', {
                operationId,
                userId,
                keys: Object.keys(externalKeyEnv),
              });
            } else {
              log.info('No matching external keys found for symbol assembly providers', {
                operationId,
                userId,
                availableProviders: Object.keys(orgKeys),
                wantedProviders,
              });
            }
          } else {
            log.warn('Failed to list organization external keys', {
              operationId,
              status: listRes.status,
            });
          }
        } catch (err) {
          // Non-fatal: fall back to system env vars
          log.warn('Could not fetch external keys from nexus-auth (using system defaults)', {
            operationId,
            error: (err as Error).message,
          });
        }
      } else if (!orgId) {
        log.warn('No organization ID available — BYOK keys will not be loaded', {
          operationId,
          hasAuthToken: !!authToken,
          hasOrgHeader: !!(req.headers['x-organization-id']),
        });
      }

      // Check for existing completed symbols from a prior run (cross-operation reuse).
      // Strategy: first check assembly_report.json, then fall back to scanning
      // the filesystem for actual gathered files (protects against a failed run
      // overwriting a successful report).
      // Cache is invalidated when: forceRefresh=true OR artifact content hash changed.
      const existingReportPath = path.join(outputDir, 'assembly_report.json');
      const backupReportPath = path.join(outputDir, 'assembly_report.success.json');

      // Helper: check if a cached report's artifact hash matches current artifacts
      const isCacheValid = (report: Record<string, unknown>): boolean => {
        if (forceRefresh) {
          log.info('Force refresh requested — invalidating cached assembly report', { operationId });
          return false;
        }
        const cachedHash = report.artifact_hash as string | undefined;
        if (!cachedHash) {
          log.info('Cached report has no artifact_hash — invalidating (pre-hash report)', { operationId });
          return false;
        }
        if (cachedHash !== artifactHash) {
          log.info('Artifact content changed — invalidating cached assembly report', {
            operationId,
            cachedHash,
            currentHash: artifactHash,
          });
          return false;
        }
        return true;
      };

      // Helper: send reuse response and return early
      const sendReuseResponse = async (report: Record<string, unknown>, symbolCount: number) => {
        const updatedReport = { ...report, operationId, artifact_hash: artifactHash };
        await fs.writeFile(existingReportPath, JSON.stringify(updatedReport, null, 2));

        wsManager.emitProgress(operationId, {
          type: SchematicEventType.SYMBOL_ASSEMBLY_COMPLETE,
          operationId,
          timestamp: new Date().toISOString(),
          progress_percentage: 100,
          current_step: `Reusing ${symbolCount} previously gathered symbols (skipped re-analysis)`,
        });

        res.json({
          success: true,
          data: {
            operationId,
            status: 'complete',
            projectId,
            reused: true,
            previousSymbolsFound: symbolCount,
          },
        });
      };

      // Strategy 1: Check assembly_report.json for status=complete
      try {
        const existingContent = await fs.readFile(existingReportPath, 'utf-8');
        const existingReport = JSON.parse(existingContent) as Record<string, unknown>;
        const components = existingReport.components as unknown[];
        if (existingReport.status === 'complete' && components?.length > 0 && isCacheValid(existingReport)) {
          log.info('Found existing completed assembly report — reusing previous results', {
            operationId,
            previousOperationId: existingReport.operationId,
            symbolsFound: existingReport.symbols_found,
            totalComponents: existingReport.total_components,
            artifactHash,
          });
          await sendReuseResponse(existingReport, (existingReport.symbols_found as number) || components.length);
          return;
        }
      } catch {
        // No existing report or it's incomplete — try fallbacks
      }

      // Strategy 2: Check backup report saved before a failed run overwrote the main one
      try {
        const backupContent = await fs.readFile(backupReportPath, 'utf-8');
        const backupReport = JSON.parse(backupContent) as Record<string, unknown>;
        const components = backupReport.components as unknown[];
        if (backupReport.status === 'complete' && components?.length > 0 && isCacheValid(backupReport)) {
          log.info('Found backup assembly report (main was overwritten by error) — reusing', {
            operationId,
            previousOperationId: backupReport.operationId,
            symbolsFound: backupReport.symbols_found,
          });
          await sendReuseResponse(backupReport, (backupReport.symbols_found as number) || components.length);
          return;
        }
      } catch {
        // No backup report — try filesystem scan
      }

      // Strategy 3: Scan filesystem for gathered symbols even if report is missing/error.
      // A failed run may have overwritten the report, but the actual symbol files
      // and characterizations from previous successful runs still exist on disk.
      // Skip when forceRefresh is requested or when artifact hash changed.
      if (!forceRefresh) try {
        // Check hash marker file to verify symbols on disk match current artifacts
        const hashMarkerPath = path.join(outputDir, '.artifact_hash');
        let filesystemHashValid = false;
        try {
          const storedHash = (await fs.readFile(hashMarkerPath, 'utf-8')).trim();
          filesystemHashValid = storedHash === artifactHash;
          if (!filesystemHashValid) {
            log.info('Filesystem artifact hash mismatch — skipping disk reuse', {
              operationId, storedHash, currentHash: artifactHash,
            });
          }
        } catch {
          // No hash marker means pre-hash symbols — don't reuse
          log.info('No filesystem artifact hash marker — skipping disk reuse', { operationId });
        }

        if (filesystemHashValid) {
          const symbolsDir = path.join(outputDir, 'symbols');
          const charsDir = path.join(outputDir, 'characterizations');
          const datasheetsDir = path.join(outputDir, 'datasheets');

          const symbolFiles = await fs.readdir(symbolsDir).catch(() => [] as string[]);
          const charFiles = await fs.readdir(charsDir).catch(() => [] as string[]);
          const datasheetFiles = await fs.readdir(datasheetsDir).catch(() => [] as string[]);

          const kicadSymbols = symbolFiles.filter(f => f.endsWith('.kicad_sym'));

          if (kicadSymbols.length >= 5) {
            log.info('Filesystem scan found existing gathered symbols — building synthetic report', {
              operationId,
              symbolCount: kicadSymbols.length,
              characterizationCount: charFiles.length,
              datasheetCount: datasheetFiles.length,
            });

            // Build synthetic report from filesystem
            const syntheticComponents = kicadSymbols.map(symFile => {
              const baseName = symFile.replace('.kicad_sym', '');
              const charFile = charFiles.find(f => f.startsWith(baseName) && f.endsWith('.md'));
              const dsFile = datasheetFiles.find(f => f.startsWith(baseName) && f.endsWith('.pdf'));
              return {
                part_number: baseName,
                manufacturer: '',
                category: '',
                symbol_found: true,
                symbol_source: 'recovered-from-disk',
                symbol_path: path.join(symbolsDir, symFile),
                datasheet_found: !!dsFile,
                datasheet_path: dsFile ? path.join(datasheetsDir, dsFile) : null,
                characterization_created: !!charFile,
                characterization_path: charFile ? path.join(charsDir, charFile) : null,
                errors: [],
                success: true,
              };
            });

            const syntheticReport = {
              project_id: projectId,
              operationId,
              status: 'complete',
              started_at: new Date().toISOString(),
              completed_at: new Date().toISOString(),
              total_components: syntheticComponents.length,
              symbols_found: kicadSymbols.length,
              symbols_from_graphrag: 0,
              symbols_from_kicad: 0,
              symbols_from_snapeda: 0,
              symbols_from_ultralibrarian: 0,
              symbols_llm_generated: 0,
              datasheets_downloaded: datasheetFiles.length,
              characterizations_created: charFiles.filter(f => f.endsWith('.md')).length,
              errors_count: 0,
              components: syntheticComponents,
              errors: [],
              success: true,
            };

            await sendReuseResponse(syntheticReport, kicadSymbols.length);
            return;
          }
        }
      } catch {
        // Filesystem scan failed — proceed normally
      }

      // Check if there's already a running operation for this project and abort it
      for (const [existingOpId, entry] of activeProcesses) {
        if (entry.projectId === projectId) {
          log.warn('Killing existing symbol assembly process for same project', {
            existingOperationId: existingOpId,
            newOperationId: operationId,
            projectId,
          });
          try {
            entry.process.kill('SIGTERM');
            setTimeout(() => {
              try { entry.process.kill('SIGKILL'); } catch { /* already dead */ }
            }, 5000);
          } catch { /* process may already be gone */ }
          activeProcesses.delete(existingOpId);
        }
      }

      const python = spawn(pythonPath, [
        scriptPath,
        '--project-id', projectId,
        '--operation-id', operationId,
        '--artifacts-json', artifactsPath,
        '--output-dir', outputDir,
      ], {
        env: {
          ...process.env,
          ...externalKeyEnv,  // User BYOK keys override system defaults
        },
      });

      // Register process so it can be cancelled
      activeProcesses.set(operationId, { process: python, projectId, startedAt: new Date() });

      // Emit initial start event so frontend leaves "Initializing..."
      wsManager.emitProgress(operationId, {
        type: SchematicEventType.SYMBOL_ASSEMBLY_START,
        operationId,
        timestamp: new Date().toISOString(),
        progress_percentage: 0,
        current_step: `Starting symbol assembly for ${ideationArtifacts.length} artifacts...`,
      });

      // Stream progress to WebSocket
      python.stdout.on('data', (data) => {
        const lines = data.toString().split('\n');
        for (const line of lines) {
          if (!line.trim()) continue;

          // Parse PROGRESS: prefix for WebSocket events
          if (line.startsWith('PROGRESS:')) {
            try {
              const event = JSON.parse(line.substring(9));
              wsManager.emitProgress(operationId, event);
            } catch (err) {
              log.warn('Failed to parse progress event', { line, error: err });
            }
          } else {
            // Log other output
            log.debug('Symbol assembler output', { operationId, line });
          }
        }
      });

      python.stderr.on('data', (data) => {
        const stderr = data.toString().trim();
        if (!stderr) return;

        // Python logging writes all levels to stderr by default.
        // Only forward WARNING/ERROR/CRITICAL as error events to the UI.
        const isActualError = /\[(ERROR|CRITICAL|WARNING)\]/.test(stderr)
          || /Traceback \(most recent call last\)/.test(stderr)
          || /^(Error|Exception|RuntimeError|ConnectionError|TimeoutError)/i.test(stderr);

        if (isActualError) {
          log.error('Symbol assembler error', { operation: operationId, stderr } as any);
          wsManager.emitProgress(operationId, {
            type: SchematicEventType.ERROR,
            operationId,
            timestamp: new Date().toISOString(),
            progress_percentage: 0,
            current_step: `Python error: ${stderr.slice(0, 200)}`,
            error_message: stderr,
          });
        } else {
          // INFO/DEBUG logs - just log server-side, don't spam the UI
          log.debug('Symbol assembler log', { operation: operationId, stderr } as any);
        }
      });

      python.on('close', async (code, signal) => {
        // Unregister process
        activeProcesses.delete(operationId);

        if (signal === 'SIGTERM' || signal === 'SIGKILL') {
          log.info('Symbol assembler process was cancelled', { operationId, signal });
          wsManager.emitProgress(operationId, {
            type: SchematicEventType.ERROR,
            operationId,
            timestamp: new Date().toISOString(),
            progress_percentage: 0,
            current_step: 'Symbol assembly was cancelled',
            error_message: 'Operation cancelled by user',
          });
        } else if (code !== 0) {
          log.error('Symbol assembler process failed', { operation: operationId, exitCode: code } as any);
          wsManager.emitProgress(operationId, {
            type: SchematicEventType.ERROR,
            operationId,
            timestamp: new Date().toISOString(),
            progress_percentage: 0,
            current_step: `Symbol assembly failed with exit code ${code}`,
            error_message: `Symbol assembly failed with exit code ${code}`,
          });
        } else {
          log.info('Symbol assembly complete', { operationId });

          // Stamp artifact_hash into the report and filesystem for future cache validation
          try {
            const reportContent = await fs.readFile(existingReportPath, 'utf-8');
            const report = JSON.parse(reportContent);
            if (!report.artifact_hash) {
              report.artifact_hash = artifactHash;
              await fs.writeFile(existingReportPath, JSON.stringify(report, null, 2));
            }
            // Write hash marker for filesystem scan validation
            const hashMarkerPath = path.join(outputDir, '.artifact_hash');
            await fs.writeFile(hashMarkerPath, artifactHash);
            // Save backup of successful report
            await fs.copyFile(existingReportPath, backupReportPath);
          } catch {
            // Non-fatal: hash stamping failed, report still valid
          }

          wsManager.emitProgress(operationId, {
            type: SchematicEventType.SYMBOL_ASSEMBLY_COMPLETE,
            operationId,
            timestamp: new Date().toISOString(),
            progress_percentage: 100,
            current_step: 'Symbol assembly completed successfully',
          });
        }
      });

      python.on('error', (err) => {
        // Unregister process
        activeProcesses.delete(operationId);

        log.error('Failed to start symbol assembler', { operation: operationId, errorMessage: err.message } as any);
        wsManager.emitProgress(operationId, {
          type: SchematicEventType.ERROR,
          operationId,
          timestamp: new Date().toISOString(),
          progress_percentage: 0,
          current_step: `Failed to start symbol assembler: ${err.message}`,
          error_message: `Failed to start symbol assembler: ${err.message}`,
        });
      });

      // Return immediately with operation ID
      res.json({
        success: true,
        data: {
          operationId,
          status: 'running',
          projectId,
        },
      });
    } catch (error) {
      next(error);
    }
  });

  /**
   * List all active symbol assembly operations for a project
   * GET /api/v1/projects/:projectId/symbol-assembly/active
   *
   * IMPORTANT: This route MUST be registered before /:operationId
   * so Express doesn't match "active" as an operationId parameter.
   */
  router.get('/active', async (req: Request, res: Response, next: NextFunction) => {
    try {
      const { projectId } = req.params;

      const operations = Array.from(activeProcesses.entries())
        .filter(([, entry]) => entry.projectId === projectId)
        .map(([opId, entry]) => ({
          operationId: opId,
          projectId: entry.projectId,
          startedAt: entry.startedAt.toISOString(),
          pid: entry.process.pid,
        }));

      res.json({
        success: true,
        data: {
          projectId,
          activeOperations: operations,
          count: operations.length,
        },
      });
    } catch (error) {
      next(error);
    }
  });

  /**
   * Cancel ALL active operations for a project
   * DELETE /api/v1/projects/:projectId/symbol-assembly/active
   */
  router.delete('/active', async (req: Request, res: Response, next: NextFunction) => {
    try {
      const { projectId } = req.params;

      const cancelled: string[] = [];
      for (const [opId, entry] of activeProcesses) {
        if (entry.projectId === projectId) {
          try {
            entry.process.kill('SIGTERM');
            setTimeout(() => {
              try { entry.process.kill('SIGKILL'); } catch { /* already dead */ }
            }, 5000);
          } catch { /* process may already be gone */ }
          activeProcesses.delete(opId);
          cancelled.push(opId);

          log.info('Cancelled active symbol assembly operation', { operationId: opId, projectId });
        }
      }

      res.json({
        success: true,
        data: {
          projectId,
          cancelledOperations: cancelled,
          count: cancelled.length,
        },
      });
    } catch (error) {
      next(error);
    }
  });

  /**
   * Get assembly report
   * GET /api/v1/projects/:projectId/symbol-assembly/:operationId
   */
  router.get('/:operationId', async (req: Request, res: Response, next: NextFunction) => {
    try {
      const { projectId, operationId } = req.params;

      log.debug('Fetching assembly report', { projectId, operationId });

      const reportPath = path.join(
        config.artifacts.basePath,
        projectId,
        'symbol-assembly',
        'assembly_report.json'
      );

      try {
        const reportContent = await fs.readFile(reportPath, 'utf-8');
        const rawReport = JSON.parse(reportContent) as Record<string, unknown>;

        // Verify operation ID matches (check both camelCase and snake_case)
        const reportOpId = rawReport.operationId ?? rawReport.operation_id;
        if (reportOpId !== operationId) {
          throw new NotFoundError('AssemblyReport', operationId, {
            operation: 'getAssemblyReport',
            message: 'Operation ID mismatch',
          } as any);
        }

        // Transform snake_case fields from Python backend to camelCase for frontend
        const report = transformReportToCamelCase(rawReport);

        res.json({
          success: true,
          data: report,
        });
      } catch (err: any) {
        if (err.code === 'ENOENT') {
          throw new NotFoundError('AssemblyReport', operationId, {
            operation: 'getAssemblyReport',
            message: 'Assembly report not found. Operation may still be running.',
          } as any);
        }
        throw err;
      }
    } catch (error) {
      next(error);
    }
  });

  /**
   * Cancel a running symbol assembly operation
   * DELETE /api/v1/projects/:projectId/symbol-assembly/:operationId
   */
  router.delete('/:operationId', async (req: Request, res: Response, next: NextFunction) => {
    try {
      const { projectId, operationId } = req.params;

      log.info('Cancel symbol assembly requested', { projectId, operationId });

      const entry = activeProcesses.get(operationId);
      if (!entry) {
        // Check if there's any active process for this project (frontend may not know the operationId)
        let foundEntry: { operationId: string; process: ChildProcess } | null = null;
        for (const [opId, e] of activeProcesses) {
          if (e.projectId === projectId) {
            foundEntry = { operationId: opId, process: e.process };
            break;
          }
        }

        if (!foundEntry) {
          res.json({
            success: true,
            data: { operationId, status: 'not_found', message: 'No active process found (may have already completed)' },
          });
          return;
        }

        // Kill the found process
        log.info('Killing symbol assembly process by projectId match', {
          requestedOperationId: operationId,
          actualOperationId: foundEntry.operationId,
          projectId,
        });
        try {
          foundEntry.process.kill('SIGTERM');
          setTimeout(() => {
            try { foundEntry!.process.kill('SIGKILL'); } catch { /* already dead */ }
          }, 5000);
        } catch { /* process may already be gone */ }
        activeProcesses.delete(foundEntry.operationId);

        res.json({
          success: true,
          data: { operationId: foundEntry.operationId, status: 'cancelled' },
        });
        return;
      }

      // Kill the process: SIGTERM first, SIGKILL after 5s
      try {
        entry.process.kill('SIGTERM');
        setTimeout(() => {
          try { entry.process.kill('SIGKILL'); } catch { /* already dead */ }
        }, 5000);
      } catch { /* process may already be gone */ }
      activeProcesses.delete(operationId);

      log.info('Symbol assembly process cancelled', { operationId, projectId });

      // Emit cancellation event via WebSocket
      const wsManager = getSchematicWsManager();
      wsManager.emitProgress(operationId, {
        type: SchematicEventType.ERROR,
        operationId,
        timestamp: new Date().toISOString(),
        progress_percentage: 0,
        current_step: 'Symbol assembly was cancelled by user',
        error_message: 'Operation cancelled',
      });

      res.json({
        success: true,
        data: { operationId, status: 'cancelled' },
      });
    } catch (error) {
      next(error);
    }
  });

  /**
   * List gathered symbols
   * GET /api/v1/projects/:projectId/symbol-assembly/:operationId/symbols
   */
  router.get('/:operationId/symbols', async (req: Request, res: Response, next: NextFunction) => {
    try {
      const { projectId, operationId } = req.params;

      log.debug('Listing gathered symbols', { projectId, operationId });

      const symbolsDir = path.join(
        config.artifacts.basePath,
        projectId,
        'symbol-assembly',
        'symbols'
      );

      try {
        const files = await fs.readdir(symbolsDir);
        const symbols = files
          .filter((f) => f.endsWith('.kicad_sym'))
          .map((filename) => ({
            filename,
            path: path.join(symbolsDir, filename),
            relativePath: `symbol-assembly/symbols/${filename}`,
          }));

        res.json({
          success: true,
          data: {
            operationId,
            count: symbols.length,
            symbols,
          },
        });
      } catch (err: any) {
        if (err.code === 'ENOENT') {
          // Directory doesn't exist yet - no symbols gathered
          res.json({
            success: true,
            data: {
              operationId,
              count: 0,
              symbols: [],
            },
          });
          return;
        }
        throw err;
      }
    } catch (error) {
      next(error);
    }
  });

  /**
   * List downloaded datasheets
   * GET /api/v1/projects/:projectId/symbol-assembly/:operationId/datasheets
   */
  router.get('/:operationId/datasheets', async (req: Request, res: Response, next: NextFunction) => {
    try {
      const { projectId, operationId } = req.params;

      log.debug('Listing datasheets', { projectId, operationId });

      const datasheetsDir = path.join(
        config.artifacts.basePath,
        projectId,
        'symbol-assembly',
        'datasheets'
      );

      try {
        const files = await fs.readdir(datasheetsDir);
        const datasheets = files
          .filter((f) => f.endsWith('.pdf'))
          .map((filename) => ({
            filename,
            path: path.join(datasheetsDir, filename),
            relativePath: `symbol-assembly/datasheets/${filename}`,
          }));

        res.json({
          success: true,
          data: {
            operationId,
            count: datasheets.length,
            datasheets,
          },
        });
      } catch (err: any) {
        if (err.code === 'ENOENT') {
          // Directory doesn't exist yet - no datasheets
          res.json({
            success: true,
            data: {
              operationId,
              count: 0,
              datasheets: [],
            },
          });
          return;
        }
        throw err;
      }
    } catch (error) {
      next(error);
    }
  });

  /**
   * Download specific file (symbol, datasheet, or characterization)
   * GET /api/v1/projects/:projectId/symbol-assembly/:operationId/download/:filename
   */
  router.get('/:operationId/download/:filename', async (req: Request, res: Response, next: NextFunction) => {
    try {
      const { projectId, operationId, filename } = req.params;

      log.debug('Downloading symbol assembly file', { projectId, operationId, filename });

      // Validate filename to prevent directory traversal
      if (filename.includes('..') || filename.includes('/') || filename.includes('\\')) {
        throw new ValidationError('Invalid filename', {
          operation: 'downloadFile',
          filename,
        });
      }

      const baseDir = path.join(
        config.artifacts.basePath,
        projectId,
        'symbol-assembly'
      );

      // Determine file type and subdirectory
      let subdirectory: string;
      let contentType: string;
      let extension: string;

      if (filename.endsWith('.kicad_sym')) {
        subdirectory = 'symbols';
        contentType = 'text/plain';
        extension = '.kicad_sym';
      } else if (filename.endsWith('.pdf')) {
        subdirectory = 'datasheets';
        contentType = 'application/pdf';
        extension = '.pdf';
      } else if (filename.endsWith('.md')) {
        subdirectory = 'characterizations';
        contentType = 'text/markdown';
        extension = '.md';
      } else {
        throw new ValidationError('Unsupported file type', {
          operation: 'downloadFile',
          filename,
          supportedTypes: ['.kicad_sym', '.pdf', '.md'],
        });
      }

      const filePath = path.join(baseDir, subdirectory, filename);

      // Verify file exists and is within allowed directory
      const realPath = await fs.realpath(filePath).catch(() => null);
      if (!realPath || !realPath.startsWith(path.join(baseDir, subdirectory))) {
        throw new NotFoundError('File', filename, {
          operation: 'downloadFile',
          message: 'File not found or access denied',
        } as any);
      }

      // Read and send file
      const fileContent = await fs.readFile(realPath);

      res.setHeader('Content-Type', contentType);
      res.setHeader('Content-Disposition', `attachment; filename="${filename}"`);
      res.setHeader('Cache-Control', 'no-cache, no-store, must-revalidate');
      res.send(fileContent);
    } catch (error) {
      next(error);
    }
  });

  return router;
}
