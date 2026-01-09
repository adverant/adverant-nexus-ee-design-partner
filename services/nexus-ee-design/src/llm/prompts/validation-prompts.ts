/**
 * EE Design Partner - Multi-LLM Validation Prompts
 *
 * Production-ready prompts for design validation and consensus:
 * - Domain-specific design review
 * - Multi-validator consensus building
 * - Cross-domain validation
 */

import {
  LLMMessage,
  DesignReview,
  DesignIssue,
  ConsensusInput,
  ConsensusSummary,
  ResolvedIssue,
} from '../types.js';

// ============================================================================
// Types
// ============================================================================

export type ArtifactType = 'schematic' | 'pcb_layout' | 'firmware' | 'simulation' | 'bom';

export type ReviewDomain =
  | 'electrical'
  | 'thermal'
  | 'mechanical'
  | 'signal_integrity'
  | 'power_integrity'
  | 'emc'
  | 'safety'
  | 'manufacturing'
  | 'reliability'
  | 'cost';

export interface DesignArtifact {
  type: ArtifactType;
  id: string;
  name: string;
  version: number;
  summary: string;
  details: Record<string, unknown>;
}

export interface SchematicArtifact extends DesignArtifact {
  type: 'schematic';
  details: {
    componentCount: number;
    netCount: number;
    sheetCount: number;
    powerRails: Array<{ name: string; voltage: number; current: number }>;
    interfaces: string[];
    keyComponents: Array<{ reference: string; value: string; purpose: string }>;
  };
}

export interface PCBArtifact extends DesignArtifact {
  type: 'pcb_layout';
  details: {
    layerCount: number;
    boardSize: { width: number; height: number };
    componentCount: number;
    viaCount: number;
    traceCount: number;
    minTraceWidth: number;
    minClearance: number;
    stackup: string[];
    criticalNets: string[];
  };
}

export interface FirmwareArtifact extends DesignArtifact {
  type: 'firmware';
  details: {
    language: string;
    linesOfCode: number;
    modules: string[];
    rtos?: string;
    flashUsage: number;
    ramUsage: number;
    taskCount?: number;
    interruptCount?: number;
  };
}

export interface SimulationArtifact extends DesignArtifact {
  type: 'simulation';
  details: {
    simulationType: string;
    duration?: number;
    keyMetrics: Record<string, number | string>;
    passed: boolean;
    warnings: string[];
  };
}

// ============================================================================
// Design Review Prompt
// ============================================================================

/**
 * Generate a prompt for domain-specific design review
 */
export function reviewDesignPrompt(
  artifact: DesignArtifact,
  domain: ReviewDomain
): LLMMessage[] {
  const domainExpertise = getDomainExpertise(domain);
  const reviewFocus = getReviewFocus(domain, artifact.type);

  const systemPrompt = `You are an expert ${domain.replace('_', ' ')} engineer with 20+ years of experience in electronics design.

Your expertise includes:
${domainExpertise.map(e => `- ${e}`).join('\n')}

You are reviewing a ${artifact.type.replace('_', ' ')} design from the perspective of ${domain.replace('_', ' ')}.

Review guidelines:
1. Be thorough but prioritize issues by impact
2. Provide specific, actionable feedback
3. Reference industry standards where applicable
4. Consider both current functionality and future maintainability
5. Note any assumptions you're making
6. Rate your confidence in each finding

Severity levels:
- CRITICAL: Will cause failure or safety issue
- MAJOR: Significant performance/reliability impact
- MINOR: Suboptimal but functional
- INFO: Suggestions for improvement

You must respond with a valid JSON object containing your review.`;

  const artifactDetails = formatArtifactDetails(artifact);

  const userPrompt = `Review this ${artifact.type.replace('_', ' ')} design from a ${domain.replace('_', ' ')} perspective:

**Design:** ${artifact.name} (v${artifact.version})
**ID:** ${artifact.id}

**Summary:**
${artifact.summary}

**Details:**
${artifactDetails}

**Review Focus Areas:**
${reviewFocus.map(f => `- ${f}`).join('\n')}

Please provide your review in this JSON format:
{
  "domain": "${domain}",
  "score": number (0-100),
  "passed": boolean (true if score >= 70 and no critical issues),
  "confidence": number (0-1, how confident you are in this review),
  "reasoning": "string - brief explanation of your assessment",
  "issues": [
    {
      "severity": "critical|major|minor|info",
      "category": "string - specific category within ${domain}",
      "description": "string - detailed description of the issue",
      "location": "string - where in the design (component, net, module, etc)",
      "recommendation": "string - specific fix or improvement",
      "reference": "string - standard or best practice reference (optional)"
    }
  ],
  "recommendations": [
    "prioritized list of improvement recommendations"
  ],
  "assumptions": [
    "any assumptions made during review"
  ],
  "questionsForDesigner": [
    "clarifying questions that would help refine the review"
  ]
}`;

  return [
    { role: 'system', content: systemPrompt },
    { role: 'user', content: userPrompt },
  ];
}

/**
 * Validator for design review response
 */
export function validateDesignReviewResponse(response: string): DesignReview | null {
  try {
    const parsed = JSON.parse(response);

    if (
      typeof parsed.score !== 'number' ||
      typeof parsed.passed !== 'boolean' ||
      typeof parsed.domain !== 'string'
    ) {
      return null;
    }

    const issues: DesignIssue[] = Array.isArray(parsed.issues)
      ? parsed.issues.map((i: unknown) => {
          const issue = i as Record<string, unknown>;
          return {
            severity: (issue.severity as DesignIssue['severity']) || 'minor',
            category: String(issue.category || ''),
            description: String(issue.description || ''),
            location: issue.location ? String(issue.location) : undefined,
            recommendation: String(issue.recommendation || ''),
            reference: issue.reference ? String(issue.reference) : undefined,
          };
        })
      : [];

    return {
      domain: parsed.domain,
      score: parsed.score,
      passed: parsed.passed,
      confidence: typeof parsed.confidence === 'number' ? parsed.confidence : 0.8,
      reasoning: String(parsed.reasoning || ''),
      issues,
      recommendations: Array.isArray(parsed.recommendations)
        ? parsed.recommendations.map(String)
        : [],
    };
  } catch {
    return null;
  }
}

// ============================================================================
// Consensus Building Prompt
// ============================================================================

/**
 * Generate a prompt for building consensus from multiple reviews
 */
export function consensusPrompt(reviews: ConsensusInput): LLMMessage[] {
  const systemPrompt = `You are a senior engineering director responsible for synthesizing multiple expert reviews into a cohesive assessment.

Your role is to:
1. Identify areas of agreement and disagreement
2. Resolve conflicts using engineering judgment
3. Weight opinions by relevance and confidence
4. Produce a balanced, actionable final assessment
5. Highlight the most critical findings

Conflict resolution methods:
- UNANIMOUS: All reviewers agree
- MAJORITY: Most reviewers agree
- WEIGHTED: Higher confidence/relevance wins
- EXPERT: Domain-specific expertise takes precedence

You must respond with a valid JSON object containing the consensus.`;

  const reviewSummaries = reviews.reviews.map((r, i) => `
**Review ${i + 1}: ${r.domain}**
- Score: ${r.score}/100 (${r.passed ? 'PASSED' : 'FAILED'})
- Confidence: ${(r.confidence * 100).toFixed(0)}%
- Issues: ${r.issues.length} (${r.issues.filter(i => i.severity === 'critical').length} critical)
- Summary: ${r.reasoning}
`).join('\n');

  const allIssues = reviews.reviews.flatMap(r =>
    r.issues.map(i => ({
      ...i,
      domain: r.domain,
    }))
  );

  const userPrompt = `Build consensus from these expert reviews:

**Artifact:** ${reviews.artifact.type} - ${reviews.artifact.id}
${reviews.artifact.summary}

${reviewSummaries}

**All Issues Found:**
${allIssues.map(i => `- [${i.severity.toUpperCase()}] (${i.domain}) ${i.description}`).join('\n')}

Please synthesize these reviews into a consensus in this JSON format:
{
  "finalScore": number (0-100, weighted average considering confidence),
  "passed": boolean,
  "confidence": number (0-1, overall confidence in consensus),
  "agreementLevel": number (0-1, how much reviewers agreed),
  "resolvedIssues": [
    {
      "issue": "string - the issue being resolved",
      "opinions": [
        { "reviewer": "domain name", "opinion": "their view" }
      ],
      "resolution": "string - final decision",
      "method": "unanimous|majority|weighted|expert"
    }
  ],
  "keyFindings": [
    "most important findings in priority order"
  ],
  "recommendations": [
    "final prioritized recommendations"
  ],
  "riskAssessment": {
    "overallRisk": "low|medium|high|critical",
    "primaryConcerns": ["list of main concerns"],
    "mitigations": ["suggested mitigations"]
  },
  "nextSteps": [
    "recommended next steps for the design team"
  ]
}`;

  return [
    { role: 'system', content: systemPrompt },
    { role: 'user', content: userPrompt },
  ];
}

/**
 * Validator for consensus response
 */
export function validateConsensusResponse(response: string): ConsensusSummary | null {
  try {
    const parsed = JSON.parse(response);

    if (
      typeof parsed.finalScore !== 'number' ||
      typeof parsed.passed !== 'boolean'
    ) {
      return null;
    }

    const resolvedIssues: ResolvedIssue[] = Array.isArray(parsed.resolvedIssues)
      ? parsed.resolvedIssues.map((ri: unknown) => {
          const resolved = ri as Record<string, unknown>;
          return {
            issue: String(resolved.issue || ''),
            opinions: Array.isArray(resolved.opinions)
              ? resolved.opinions.map((o: unknown) => {
                  const opinion = o as Record<string, unknown>;
                  return {
                    reviewer: String(opinion.reviewer || ''),
                    opinion: String(opinion.opinion || ''),
                  };
                })
              : [],
            resolution: String(resolved.resolution || ''),
            method: (resolved.method as ResolvedIssue['method']) || 'weighted',
          };
        })
      : [];

    return {
      finalScore: parsed.finalScore,
      passed: parsed.passed,
      confidence: typeof parsed.confidence === 'number' ? parsed.confidence : 0.8,
      agreementLevel: typeof parsed.agreementLevel === 'number' ? parsed.agreementLevel : 0.7,
      resolvedIssues,
      keyFindings: Array.isArray(parsed.keyFindings) ? parsed.keyFindings.map(String) : [],
      recommendations: Array.isArray(parsed.recommendations)
        ? parsed.recommendations.map(String)
        : [],
    };
  } catch {
    return null;
  }
}

// ============================================================================
// Specialized Review Prompts
// ============================================================================

/**
 * Generate a prompt for safety-critical design review
 */
export function reviewSafetyDesignPrompt(
  artifact: DesignArtifact,
  safetyLevel: 'asil_a' | 'asil_b' | 'asil_c' | 'asil_d' | 'sil_2' | 'sil_3'
): LLMMessage[] {
  const standard = safetyLevel.startsWith('asil') ? 'ISO 26262' : 'IEC 61508';
  const level = safetyLevel.toUpperCase().replace('_', '-');

  const systemPrompt = `You are a functional safety engineer certified in ${standard}.

You are reviewing a design that must comply with ${level} requirements.

Your review must cover:
1. Fault detection and diagnostic coverage
2. Hardware architectural metrics (SPFM, LFM)
3. Dependent failure analysis
4. Safe state definitions
5. Monitoring and watchdog mechanisms
6. Redundancy implementation
7. Error handling and recovery

Be extremely thorough - safety-critical designs require rigorous review.

You must respond with a valid JSON object containing your safety review.`;

  const artifactDetails = formatArtifactDetails(artifact);

  const userPrompt = `Perform a ${standard} ${level} safety review:

**Design:** ${artifact.name} (v${artifact.version})

**Details:**
${artifactDetails}

**Safety Level Required:** ${level}
**Standard:** ${standard}

Provide your safety review in JSON format:
{
  "safetyLevel": "${safetyLevel}",
  "standard": "${standard}",
  "complianceScore": number (0-100),
  "compliant": boolean,
  "safetyMetrics": {
    "spfm": number (Single Point Fault Metric, 0-100%),
    "lfm": number (Latent Fault Metric, 0-100%),
    "diagnosticCoverage": number (0-100%),
    "pmhf": number (Probabilistic Metric for Hardware Failure)
  },
  "safetyMechanisms": [
    {
      "mechanism": "string - safety mechanism name",
      "type": "detection|prevention|mitigation",
      "coverage": "string - what faults it covers",
      "effectiveness": number (0-100%)
    }
  ],
  "violations": [
    {
      "severity": "critical|major|minor",
      "requirement": "string - ${standard} requirement",
      "violation": "string - how it's violated",
      "remediation": "string - how to fix"
    }
  ],
  "recommendations": ["safety improvement recommendations"],
  "safeState": {
    "defined": boolean,
    "description": "string - safe state description",
    "transitionTime": "string - time to reach safe state"
  }
}`;

  return [
    { role: 'system', content: systemPrompt },
    { role: 'user', content: userPrompt },
  ];
}

/**
 * Generate a prompt for DFM (Design for Manufacturing) review
 */
export function reviewDFMPrompt(artifact: PCBArtifact): LLMMessage[] {
  const systemPrompt = `You are a PCB manufacturing engineer with extensive experience in:
- SMT assembly processes
- Through-hole assembly
- Reflow and wave soldering
- AOI and X-ray inspection
- Panelization and depaneling
- Yield optimization

Review the design for manufacturability issues that could cause:
- Assembly defects
- Solder bridging
- Tombstoning
- Insufficient solder joints
- Inspection difficulties
- Test access problems

You must respond with a valid JSON object containing your DFM review.`;

  const userPrompt = `Perform a DFM review:

**PCB Design:** ${artifact.name} (v${artifact.version})

**Board Specifications:**
- Size: ${artifact.details.boardSize.width}mm x ${artifact.details.boardSize.height}mm
- Layers: ${artifact.details.layerCount}
- Components: ${artifact.details.componentCount}
- Min Trace Width: ${artifact.details.minTraceWidth}mm
- Min Clearance: ${artifact.details.minClearance}mm

**Stackup:**
${artifact.details.stackup.map((l, i) => `${i + 1}. ${l}`).join('\n')}

**Critical Nets:**
${artifact.details.criticalNets.map(n => `- ${n}`).join('\n')}

Provide your DFM review in JSON format:
{
  "manufacturabilityScore": number (0-100),
  "assemblyClass": "IPC-A-610 Class 1|2|3",
  "estimatedYield": number (% first-pass yield estimate),
  "issues": [
    {
      "severity": "critical|major|minor",
      "category": "placement|routing|soldermask|silkscreen|drilling|panelization",
      "description": "string - the issue",
      "location": "string - where on the board",
      "impact": "string - manufacturing impact",
      "solution": "string - recommended fix"
    }
  ],
  "designRules": {
    "minTraceOk": boolean,
    "minClearanceOk": boolean,
    "minViaOk": boolean,
    "minAnnularRingOk": boolean,
    "aspectRatioOk": boolean
  },
  "assemblyNotes": ["notes for assembly house"],
  "testStrategy": {
    "ictPossible": boolean,
    "testCoverage": number (% of nets accessible),
    "recommendations": ["test strategy recommendations"]
  },
  "costDrivers": ["factors that increase manufacturing cost"],
  "recommendations": ["prioritized DFM improvements"]
}`;

  return [
    { role: 'system', content: systemPrompt },
    { role: 'user', content: userPrompt },
  ];
}

// ============================================================================
// Helper Functions
// ============================================================================

function getDomainExpertise(domain: ReviewDomain): string[] {
  const expertise: Record<ReviewDomain, string[]> = {
    electrical: [
      'Circuit analysis and simulation',
      'Power distribution and decoupling',
      'Signal routing and termination',
      'Component selection and derating',
      'ESD protection design',
    ],
    thermal: [
      'Thermal simulation and analysis',
      'Heat sink and cooling design',
      'PCB thermal management',
      'Component junction temperature analysis',
      'Thermal via placement',
    ],
    mechanical: [
      'PCB mechanical stress analysis',
      'Vibration and shock resistance',
      'Connector and mounting design',
      'Enclosure thermal management',
      'Environmental protection (IP ratings)',
    ],
    signal_integrity: [
      'High-speed digital design',
      'Impedance control and matching',
      'Crosstalk analysis',
      'Eye diagram interpretation',
      'Return path optimization',
    ],
    power_integrity: [
      'PDN (Power Distribution Network) design',
      'Decoupling strategy',
      'Voltage drop analysis',
      'Transient response',
      'Plane resonance',
    ],
    emc: [
      'EMI/EMC compliance (FCC, CE, CISPR)',
      'Shielding effectiveness',
      'Filtering design',
      'Grounding strategies',
      'Cable and connector emissions',
    ],
    safety: [
      'Functional safety (ISO 26262, IEC 61508)',
      'Electrical safety (UL, IEC 62368)',
      'Fault detection and response',
      'Redundancy implementation',
      'FMEA/FMEDA analysis',
    ],
    manufacturing: [
      'DFM (Design for Manufacturing)',
      'DFA (Design for Assembly)',
      'DFT (Design for Test)',
      'Panelization',
      'Process capability analysis',
    ],
    reliability: [
      'MTBF/MTTF analysis',
      'Derating analysis',
      'Accelerated life testing',
      'Failure mode analysis',
      'Environmental qualification',
    ],
    cost: [
      'BOM cost optimization',
      'Assembly cost analysis',
      'Design-to-cost strategies',
      'Volume manufacturing economics',
      'Second-source availability',
    ],
  };

  return expertise[domain] || [];
}

function getReviewFocus(domain: ReviewDomain, artifactType: ArtifactType): string[] {
  const focusMatrix: Record<ReviewDomain, Record<ArtifactType, string[]>> = {
    electrical: {
      schematic: [
        'Power rail voltage levels and current capacity',
        'Decoupling capacitor placement and values',
        'Component voltage/current ratings',
        'Protection circuit adequacy',
        'Reference designator consistency',
      ],
      pcb_layout: [
        'Power plane integrity',
        'Decoupling capacitor proximity',
        'Current-carrying trace widths',
        'Thermal relief on power connections',
        'Component placement for signal flow',
      ],
      firmware: [
        'GPIO configuration correctness',
        'ADC/DAC scaling factors',
        'PWM frequency and duty cycle limits',
        'Interrupt priority assignments',
        'Watchdog configuration',
      ],
      simulation: [
        'Operating point validity',
        'Transient response stability',
        'Worst-case analysis coverage',
        'Monte Carlo spread',
        'Temperature range coverage',
      ],
      bom: [
        'Component voltage ratings',
        'Power dissipation ratings',
        'Operating temperature range',
        'Package thermal resistance',
        'Derating factors',
      ],
    },
    thermal: {
      schematic: [
        'Power dissipation estimates per component',
        'Thermal interface components',
        'Temperature sensing provisions',
        'Thermal shutdown circuits',
      ],
      pcb_layout: [
        'Thermal via quantity and placement',
        'Copper pour for heat spreading',
        'High-power component spacing',
        'Airflow consideration',
        'Hot spot identification',
      ],
      firmware: [
        'Thermal monitoring implementation',
        'Temperature-based derating',
        'Fan/cooling control algorithms',
        'Thermal shutdown handling',
      ],
      simulation: [
        'Junction temperature predictions',
        'Thermal resistance validation',
        'Transient thermal response',
        'Ambient temperature sensitivity',
      ],
      bom: [
        'Thermal interface materials',
        'Heat sink specifications',
        'Component thermal ratings',
        'High-temperature part selection',
      ],
    },
    // Add more domain/artifact combinations as needed
    mechanical: { schematic: [], pcb_layout: [], firmware: [], simulation: [], bom: [] },
    signal_integrity: { schematic: [], pcb_layout: [], firmware: [], simulation: [], bom: [] },
    power_integrity: { schematic: [], pcb_layout: [], firmware: [], simulation: [], bom: [] },
    emc: { schematic: [], pcb_layout: [], firmware: [], simulation: [], bom: [] },
    safety: { schematic: [], pcb_layout: [], firmware: [], simulation: [], bom: [] },
    manufacturing: { schematic: [], pcb_layout: [], firmware: [], simulation: [], bom: [] },
    reliability: { schematic: [], pcb_layout: [], firmware: [], simulation: [], bom: [] },
    cost: { schematic: [], pcb_layout: [], firmware: [], simulation: [], bom: [] },
  };

  return focusMatrix[domain]?.[artifactType] || [];
}

function formatArtifactDetails(artifact: DesignArtifact): string {
  const details = artifact.details;
  const lines: string[] = [];

  for (const [key, value] of Object.entries(details)) {
    if (Array.isArray(value)) {
      if (value.length > 0 && typeof value[0] === 'object') {
        lines.push(`${formatKey(key)}:`);
        value.slice(0, 10).forEach(item => {
          lines.push(`  - ${JSON.stringify(item)}`);
        });
        if (value.length > 10) {
          lines.push(`  ... and ${value.length - 10} more`);
        }
      } else {
        lines.push(`${formatKey(key)}: ${value.join(', ')}`);
      }
    } else if (typeof value === 'object' && value !== null) {
      lines.push(`${formatKey(key)}: ${JSON.stringify(value)}`);
    } else {
      lines.push(`${formatKey(key)}: ${value}`);
    }
  }

  return lines.join('\n');
}

function formatKey(key: string): string {
  return key
    .replace(/([A-Z])/g, ' $1')
    .replace(/^./, str => str.toUpperCase())
    .trim();
}

// ============================================================================
// Exports
// ============================================================================

export default {
  reviewDesignPrompt,
  validateDesignReviewResponse,
  consensusPrompt,
  validateConsensusResponse,
  reviewSafetyDesignPrompt,
  reviewDFMPrompt,
};
