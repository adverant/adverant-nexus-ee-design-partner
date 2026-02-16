/**
 * MAPO Schematic Pipeline Client
 *
 * TypeScript client for calling the Python MAPO schematic generation pipeline.
 * Provides integration between the Node.js API service and the Python ML pipeline.
 */

import { spawn } from "child_process";
import * as path from "path";
import * as fs from "fs/promises";

/**
 * BOM item for schematic generation
 */
export interface BOMItem {
  part_number: string;
  manufacturer?: string;
  reference?: string;
  quantity?: number;
  category?: string;
  value?: string;
  footprint?: string;
  description?: string;
}

/**
 * Connection between component pins
 */
export interface Connection {
  from_ref: string;
  from_pin: string;
  to_ref: string;
  to_pin: string;
  net_name?: string;
}

/**
 * Block diagram structure for hierarchical schematics
 */
export interface BlockDiagram {
  blocks: Record<
    string,
    {
      components: string[];
      external_pins: string[];
    }
  >;
  connections: Array<{
    from_block: string;
    from_pin: string;
    to_block: string;
    to_pin: string;
  }>;
}

/**
 * Expert validation result
 */
export interface ExpertResult {
  expert: string;
  score: number;
  passed: boolean;
  issues: Array<{
    component: string;
    description: string;
    severity: "critical" | "major" | "minor";
    location: string;
  }>;
  suggestions: string[];
  confidence: number;
}

/**
 * Validation report from MAPO pipeline
 */
export interface ValidationReport {
  overall_score: number;
  passed: boolean;
  expert_results: ExpertResult[];
  critical_issues: Array<{
    component: string;
    description: string;
    severity: string;
    expert: string;
  }>;
  recommended_fixes: Array<{
    issue_ref: string;
    fix_type: string;
    description: string;
    kicad_action: string;
    priority: string;
  }>;
  iteration: number;
}

/**
 * Result from the MAPO schematic pipeline
 */
export interface PipelineResult {
  success: boolean;
  schematic_path: string | null;
  sheet_count: number;
  validation_score: number | null;
  validation_passed: boolean;
  symbols_fetched: number;
  symbols_from_cache: number;
  symbols_generated: number;
  iterations: number;
  total_time_seconds: number;
  errors: string[];
}

/**
 * Options for schematic generation
 */
export interface GenerateOptions {
  bom: BOMItem[];
  designIntent: string;
  connections?: Connection[];
  blockDiagram?: BlockDiagram;
  designName?: string;
  referenceImages?: Buffer[];
  skipValidation?: boolean;
  timeout?: number; // milliseconds
}

/**
 * MAPO Schematic Pipeline Client
 *
 * Calls the Python MAPO pipeline for schematic generation and validation.
 */
export class MAPOSchematicClient {
  private pythonPath: string;
  private pipelinePath: string;

  constructor(options?: { pythonPath?: string; pipelinePath?: string }) {
    this.pythonPath = options?.pythonPath || process.env.PYTHON_PATH || "python3";
    this.pipelinePath =
      options?.pipelinePath ||
      path.join(__dirname, "../../python-scripts/mapo_schematic_pipeline.py");
  }

  /**
   * Generate a validated schematic using the MAPO pipeline.
   */
  async generate(options: GenerateOptions): Promise<PipelineResult> {
    const {
      bom,
      designIntent,
      connections,
      blockDiagram,
      designName = "schematic",
      skipValidation = false,
      timeout = 1200000, // 20 minutes default - schematic gen includes LLM + SPICE smoke test + proxy queue wait
    } = options;

    // Create temporary BOM file
    const tempDir = process.env.TEMP_DIR || "/tmp";
    const bomPath = path.join(tempDir, `bom-${Date.now()}.json`);
    const connectionsPath = path.join(tempDir, `connections-${Date.now()}.json`);
    const intentPath = path.join(tempDir, `intent-${Date.now()}.txt`);

    try {
      // Write BOM to temp file
      await fs.writeFile(bomPath, JSON.stringify(bom, null, 2));

      // Write design intent to temp file (avoid E2BIG error from large CLI args)
      await fs.writeFile(intentPath, designIntent, "utf-8");

      // Write connections if provided
      if (connections) {
        await fs.writeFile(connectionsPath, JSON.stringify(connections, null, 2));
      }

      // Build command arguments
      const args = [
        this.pipelinePath,
        "--bom",
        bomPath,
        "--intent-file",
        intentPath,
        "--output",
        designName,
      ];

      if (skipValidation) {
        args.push("--skip-validation");
      }

      // Run the Python pipeline
      const result = await this.runPython(args, timeout);

      return result;
    } finally {
      // Cleanup temp files
      try {
        await fs.unlink(bomPath);
        await fs.unlink(intentPath);
        if (connections) {
          await fs.unlink(connectionsPath);
        }
      } catch {
        // Ignore cleanup errors
      }
    }
  }

  /**
   * Run the Python pipeline and parse the result.
   */
  private runPython(args: string[], timeout: number): Promise<PipelineResult> {
    return new Promise((resolve, reject) => {
      const process = spawn(this.pythonPath, args, {
        env: {
          ...process.env,
          PYTHONUNBUFFERED: "1",
        },
      });

      let stdout = "";
      let stderr = "";

      process.stdout.on("data", (data) => {
        stdout += data.toString();
      });

      process.stderr.on("data", (data) => {
        stderr += data.toString();
        console.error(`[MAPO Pipeline] ${data.toString()}`);
      });

      const timeoutId = setTimeout(() => {
        process.kill();
        reject(new Error(`Pipeline timeout after ${timeout}ms`));
      }, timeout);

      process.on("close", (code) => {
        clearTimeout(timeoutId);

        if (code !== 0) {
          reject(new Error(`Pipeline failed with code ${code}: ${stderr}`));
          return;
        }

        // Parse the JSON result from stdout
        try {
          // Extract JSON from output (look for last JSON object)
          const jsonMatch = stdout.match(/\{[\s\S]*\}$/);
          if (jsonMatch) {
            const result = JSON.parse(jsonMatch[0]) as PipelineResult;
            resolve(result);
          } else {
            // Return a default result if no JSON found
            resolve({
              success: true,
              schematic_path: null,
              sheet_count: 0,
              validation_score: null,
              validation_passed: false,
              symbols_fetched: 0,
              symbols_from_cache: 0,
              symbols_generated: 0,
              iterations: 0,
              total_time_seconds: 0,
              errors: [],
            });
          }
        } catch (e) {
          reject(new Error(`Failed to parse pipeline result: ${e}`));
        }
      });

      process.on("error", (err) => {
        clearTimeout(timeoutId);
        reject(err);
      });
    });
  }

  /**
   * Validate an existing schematic using the MAPO vision validator.
   */
  async validate(
    schematicPath: string,
    designIntent: string
  ): Promise<ValidationReport | null> {
    // Read schematic file
    const schematicContent = await fs.readFile(schematicPath, "utf-8");

    // Call validation endpoint
    const args = [
      path.join(
        path.dirname(this.pipelinePath),
        "validation/schematic_vision_validator.py"
      ),
      schematicPath,
      designIntent,
    ];

    return new Promise((resolve, reject) => {
      const process = spawn(this.pythonPath, args);

      let stdout = "";
      let stderr = "";

      process.stdout.on("data", (data) => {
        stdout += data.toString();
      });

      process.stderr.on("data", (data) => {
        stderr += data.toString();
      });

      process.on("close", (code) => {
        if (code !== 0) {
          reject(new Error(`Validation failed: ${stderr}`));
          return;
        }

        // Look for JSON output file
        const reportPath = schematicPath.replace(/\.kicad_sch$/, ".validation.json");
        fs.readFile(reportPath, "utf-8")
          .then((content) => {
            resolve(JSON.parse(content) as ValidationReport);
          })
          .catch(() => {
            resolve(null);
          });
      });
    });
  }

  /**
   * Search for symbols in the cache/GraphRAG.
   */
  async searchSymbols(
    query: string,
    category?: string,
    limit: number = 10
  ): Promise<
    Array<{
      part_number: string;
      manufacturer: string;
      description: string;
      category: string;
      source: string;
    }>
  > {
    const args = [
      path.join(path.dirname(this.pipelinePath), "graphrag/symbol_indexer.py"),
      "search",
      query,
    ];

    return new Promise((resolve, reject) => {
      const process = spawn(this.pythonPath, args);

      let stdout = "";

      process.stdout.on("data", (data) => {
        stdout += data.toString();
      });

      process.on("close", (code) => {
        if (code !== 0) {
          resolve([]);
          return;
        }

        // Parse results from output
        const results: Array<{
          part_number: string;
          manufacturer: string;
          description: string;
          category: string;
          source: string;
        }> = [];

        // Simple parsing of output lines
        const lines = stdout.split("\n");
        for (const line of lines) {
          const match = line.match(/^\s*-\s*(\S+)\s+\(([^)]+)\)/);
          if (match) {
            results.push({
              part_number: match[1],
              manufacturer: "",
              description: "",
              category: match[2],
              source: "cache",
            });
          }
        }

        resolve(results.slice(0, limit));
      });
    });
  }

  /**
   * Fetch a specific symbol by part number.
   */
  async fetchSymbol(
    partNumber: string,
    manufacturer?: string,
    category: string = "Other"
  ): Promise<{
    part_number: string;
    symbol_sexp: string;
    source: string;
    needs_review: boolean;
  } | null> {
    const args = [
      path.join(
        path.dirname(this.pipelinePath),
        "agents/symbol_fetcher/symbol_fetcher_agent.py"
      ),
      partNumber,
    ];

    if (manufacturer) {
      args.push(manufacturer);
    }
    args.push(category);

    return new Promise((resolve, reject) => {
      const process = spawn(this.pythonPath, args);

      let stdout = "";
      let stderr = "";

      process.stdout.on("data", (data) => {
        stdout += data.toString();
      });

      process.stderr.on("data", (data) => {
        stderr += data.toString();
      });

      process.on("close", (code) => {
        if (code !== 0) {
          resolve(null);
          return;
        }

        // Parse basic info from output
        const sourceMatch = stdout.match(/Source:\s*(\S+)/);
        const needsReviewMatch = stdout.match(/Needs Review:\s*(True|False)/i);

        resolve({
          part_number: partNumber,
          symbol_sexp: "", // Would need to read from cache
          source: sourceMatch?.[1] || "unknown",
          needs_review: needsReviewMatch?.[1]?.toLowerCase() === "true",
        });
      });
    });
  }
}

// Export singleton instance
export const mapoClient = new MAPOSchematicClient();

// Export default
export default MAPOSchematicClient;
