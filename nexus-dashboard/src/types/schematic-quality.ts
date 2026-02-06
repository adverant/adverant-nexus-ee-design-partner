/**
 * Schematic Quality & Compliance Types (MAPO v3.0)
 *
 * Defines types for the real-time standards compliance checklist UI.
 * Supports 51 checks across 6 compliance standards:
 * - NASA-STD-8739.4 (Workmanship Standards)
 * - MIL-STD-883 (Microelectronics)
 * - IPC-2221 (PCB Design)
 * - IEC 61000 (EMC)
 * - IPC-7351 (Land Patterns)
 * - Professional Best Practices
 */

// ============================================
// Core Enums
// ============================================

/**
 * Compliance standards supported by the validator
 */
export enum ComplianceStandard {
  NASA_STD_8739_4 = 'NASA-STD-8739.4',
  MIL_STD_883 = 'MIL-STD-883',
  IPC_2221 = 'IPC-2221',
  IEC_61000 = 'IEC-61000',
  IPC_7351 = 'IPC-7351',
  BEST_PRACTICES = 'Professional Best Practices'
}

/**
 * Severity levels for compliance violations
 */
export enum ViolationSeverity {
  CRITICAL = 'critical',  // Blocks schematic export
  HIGH = 'high',          // Strongly recommended to fix
  MEDIUM = 'medium',      // Should fix before production
  LOW = 'low',            // Nice to have
  INFO = 'info'           // Informational only
}

/**
 * Check execution status
 */
export enum CheckStatus {
  PENDING = 'pending',
  RUNNING = 'running',
  PASSED = 'passed',
  FAILED = 'failed',
  SKIPPED = 'skipped',
  WARNING = 'warning'
}

/**
 * Check categories for grouping in UI
 */
export enum CheckCategory {
  POWER = 'Power & Grounding',
  CONNECTIVITY = 'Connectivity & Nets',
  SYMBOLS = 'Symbols & Properties',
  PLACEMENT = 'Placement & Layout',
  STANDARDS = 'Standards Compliance',
  DOCUMENTATION = 'Documentation'
}

// ============================================
// Checklist Item Interfaces
// ============================================

/**
 * Single compliance check item definition (static)
 */
export interface ChecklistItemDefinition {
  /** Unique check ID (e.g., "nasa-power-01") */
  id: string;

  /** Human-readable short name */
  name: string;

  /** Full description of what this check validates */
  description: string;

  /** Which standard(s) this check enforces */
  standards: ComplianceStandard[];

  /** Category for UI grouping */
  category: CheckCategory;

  /** Severity if check fails */
  severity: ViolationSeverity;

  /** Detailed explanation of why this matters */
  rationale: string;

  /** How to fix violations */
  remediation: string;

  /** Can this check auto-fix violations? */
  autoFixable: boolean;

  /** Standard-specific rule references (e.g., "NASA-STD-8739.4 §3.2.1") */
  references: string[];
}

/**
 * Runtime check result (dynamic, updated via WebSocket)
 */
export interface ChecklistItemResult {
  /** Check ID (matches ChecklistItemDefinition.id) */
  id: string;

  /** Current execution status */
  status: CheckStatus;

  /** Timestamp when check started */
  startedAt?: string;

  /** Timestamp when check completed */
  completedAt?: string;

  /** Duration in milliseconds */
  duration?: number;

  /** Number of violations found */
  violationCount: number;

  /** Detailed violation messages */
  violations: ViolationDetail[];

  /** Warning messages (non-blocking) */
  warnings: string[];

  /** Info messages */
  info: string[];

  /** Was auto-fix applied? */
  autoFixed: boolean;

  /** Number of items auto-fixed */
  fixedCount: number;
}

/**
 * Detailed violation info
 */
export interface ViolationDetail {
  /** Severity level */
  severity: ViolationSeverity;

  /** Violation message */
  message: string;

  /** Component reference (e.g., "U1", "R5") */
  componentRef?: string;

  /** Net name (e.g., "+3V3", "GND") */
  netName?: string;

  /** Pin number */
  pinNumber?: string;

  /** Symbol library reference */
  symbolLib?: string;

  /** Property name (for property checks) */
  propertyName?: string;

  /** Expected value */
  expectedValue?: string;

  /** Actual value found */
  actualValue?: string;

  /** File path (for symbol lib checks) */
  filePath?: string;

  /** Line number in schematic file */
  lineNumber?: number;

  /** Suggested fix */
  suggestedFix?: string;

  /** Can this violation be auto-fixed? */
  autoFixable: boolean;
}

// ============================================
// Compliance Report Interfaces
// ============================================

/**
 * Full compliance report for a schematic
 */
export interface ComplianceReport {
  /** Report ID */
  reportId: string;

  /** Schematic operation ID */
  operationId: string;

  /** Project ID */
  projectId: string;

  /** When report was generated */
  generatedAt: string;

  /** Overall compliance score (0-100) */
  score: number;

  /** Did all checks pass? */
  passed: boolean;

  /** Total checks run */
  totalChecks: number;

  /** Checks passed */
  passedChecks: number;

  /** Checks failed */
  failedChecks: number;

  /** Checks with warnings */
  warningsChecks: number;

  /** Checks skipped */
  skippedChecks: number;

  /** Total violations found */
  totalViolations: number;

  /** Violations by severity */
  violationsBySeverity: Record<ViolationSeverity, number>;

  /** Results for each check */
  checkResults: ChecklistItemResult[];

  /** Standards coverage (which standards were checked) */
  standardsCoverage: ComplianceStandard[];

  /** Was auto-fix enabled? */
  autoFixEnabled: boolean;

  /** How many items were auto-fixed? */
  autoFixedCount: number;

  /** User-submitted waivers */
  waivers: ComplianceWaiver[];

  /** Export metadata (for audit trail) */
  exportMetadata?: {
    exportedAt: string;
    exportedBy: string;
    format: 'json' | 'pdf' | 'html';
    filePath: string;
  };
}

/**
 * Waiver for a specific violation
 */
export interface ComplianceWaiver {
  /** Waiver ID */
  id: string;

  /** Check ID this waiver applies to */
  checkId: string;

  /** Component reference (if applicable) */
  componentRef?: string;

  /** Net name (if applicable) */
  netName?: string;

  /** Justification for waiver */
  justification: string;

  /** Who submitted the waiver */
  submittedBy: string;

  /** When waiver was submitted */
  submittedAt: string;

  /** Expiration date (optional) */
  expiresAt?: string;

  /** Waiver status */
  status: 'pending' | 'approved' | 'rejected' | 'expired';

  /** Approver (if approved) */
  approvedBy?: string;

  /** Approval timestamp */
  approvedAt?: string;
}

// ============================================
// WebSocket Event Payloads
// ============================================

/**
 * Check started event
 */
export interface ChecklistItemStartEvent {
  checkId: string;
  checkName: string;
  timestamp: string;
}

/**
 * Check passed event
 */
export interface ChecklistItemPassEvent {
  checkId: string;
  checkName: string;
  timestamp: string;
  duration: number;
  info: string[];
}

/**
 * Check failed event
 */
export interface ChecklistItemFailEvent {
  checkId: string;
  checkName: string;
  timestamp: string;
  duration: number;
  violationCount: number;
  violations: ViolationDetail[];
  warnings: string[];
  autoFixed: boolean;
  fixedCount: number;
}

/**
 * Batch check update event (multiple checks updated at once)
 */
export interface ChecklistBatchUpdateEvent {
  updates: ChecklistItemResult[];
  timestamp: string;
}

/**
 * Compliance score update event
 */
export interface ComplianceScoreUpdateEvent {
  score: number;
  passedChecks: number;
  totalChecks: number;
  violationCount: number;
  violationsBySeverity: Record<ViolationSeverity, number>;
  timestamp: string;
}

// ============================================
// UI Component Props
// ============================================

/**
 * Props for SchematicQualityChecklist component
 */
export interface SchematicQualityChecklistProps {
  /** Schematic operation ID to track */
  operationId: string;

  /** Project ID */
  projectId: string;

  /** Show only failed checks? */
  showFailedOnly?: boolean;

  /** Enable auto-fix? */
  autoFixEnabled?: boolean;

  /** Callback when export button clicked */
  onExport?: (report: ComplianceReport) => void;

  /** Callback when waiver submitted */
  onWaiverSubmit?: (checkId: string, justification: string) => void;

  /** Custom CSS class */
  className?: string;
}

/**
 * Props for ChecklistItem component
 */
export interface ChecklistItemProps {
  /** Static definition */
  definition: ChecklistItemDefinition;

  /** Runtime result */
  result: ChecklistItemResult;

  /** Is auto-fix enabled? */
  autoFixEnabled: boolean;

  /** Callback when user requests waiver */
  onWaiverRequest?: (checkId: string) => void;

  /** Callback when user clicks violation detail */
  onViolationClick?: (violation: ViolationDetail) => void;
}

/**
 * Props for ComplianceScoreCard component
 */
export interface ComplianceScoreCardProps {
  /** Current score (0-100) */
  score: number;

  /** Passed checks */
  passedChecks: number;

  /** Total checks */
  totalChecks: number;

  /** Total violations */
  violationCount: number;

  /** Violations by severity */
  violationsBySeverity: Record<ViolationSeverity, number>;

  /** Loading state */
  isLoading: boolean;
}

// ============================================
// Filter/Sort Options
// ============================================

/**
 * Checklist filter options
 */
export interface ChecklistFilter {
  /** Filter by status */
  status?: CheckStatus[];

  /** Filter by severity */
  severity?: ViolationSeverity[];

  /** Filter by category */
  category?: CheckCategory[];

  /** Filter by standard */
  standard?: ComplianceStandard[];

  /** Show only items with violations? */
  hasViolations?: boolean;

  /** Show only auto-fixable items? */
  autoFixable?: boolean;

  /** Text search */
  searchText?: string;
}

/**
 * Checklist sort options
 */
export type ChecklistSortBy =
  | 'severity'
  | 'status'
  | 'name'
  | 'category'
  | 'standard'
  | 'violationCount'
  | 'duration';

export interface ChecklistSort {
  by: ChecklistSortBy;
  direction: 'asc' | 'desc';
}
