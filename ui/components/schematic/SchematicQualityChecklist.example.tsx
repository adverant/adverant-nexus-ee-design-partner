"use client";

/**
 * SchematicQualityChecklist - Example Usage with Realistic Placeholder Data
 *
 * This file demonstrates how to use the SchematicQualityChecklist component
 * with realistic placeholder data for development and testing.
 */

import { useState, useEffect } from "react";
import { SchematicQualityChecklist } from "./SchematicQualityChecklist";
import {
  CheckStatus,
  ViolationSeverity,
  ChecklistItemResult,
  ComplianceReport,
} from "@/types/schematic-quality";

/**
 * Example placeholder data generator
 * Simulates WebSocket events by updating check results over time
 */
export function SchematicQualityChecklistExample() {
  const [operationId] = useState(`op-${Date.now()}`);
  const projectId = "example-project-123";

  const handleExport = (report: ComplianceReport) => {
    console.log("📄 Exporting Compliance Report:");
    console.log(`  Score: ${report.score}/100`);
    console.log(`  Passed: ${report.passedChecks}/${report.totalChecks}`);
    console.log(`  Violations: ${report.totalViolations}`);
    console.log(`  Report ID: ${report.reportId}`);

    // In production, this would:
    // - Generate PDF/JSON export
    // - Save to file system
    // - Upload to cloud storage
    // - Send notification to user

    // For demo, just download as JSON
    const blob = new Blob([JSON.stringify(report, null, 2)], {
      type: "application/json",
    });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `compliance-report-${report.reportId}.json`;
    a.click();
    URL.revokeObjectURL(url);
  };

  const handleWaiverSubmit = (checkId: string, justification: string) => {
    console.log("📋 Waiver Submitted:");
    console.log(`  Check ID: ${checkId}`);
    console.log(`  Justification: ${justification}`);

    // In production, this would:
    // - POST to /api/compliance/waivers
    // - Create waiver record in database
    // - Send notification to approver
    // - Update check status to "waived"

    alert(
      `Waiver submitted for check ${checkId}.\n\nJustification: ${justification}\n\nIn production, this would be sent to an approver.`
    );
  };

  return (
    <div className="min-h-screen bg-gray-100 p-8">
      <div className="mx-auto max-w-7xl">
        {/* Header */}
        <div className="mb-6">
          <h1 className="text-3xl font-bold text-gray-900">
            Schematic Quality Checklist Demo
          </h1>
          <p className="text-gray-600">
            Real-time compliance validation with WebSocket updates
          </p>
        </div>

        {/* Info Card */}
        <div className="mb-6 rounded-lg border border-blue-200 bg-blue-50 p-4">
          <h3 className="mb-2 font-semibold text-blue-900">Demo Mode</h3>
          <p className="text-sm text-blue-800">
            This is a demonstration with placeholder data. In production, the
            component subscribes to WebSocket events from the MAPO v3.0 pipeline
            and updates in real-time as checks execute.
          </p>
          <p className="mt-2 text-sm text-blue-800">
            <strong>Operation ID:</strong> {operationId}
          </p>
        </div>

        {/* Checklist Component */}
        <SchematicQualityChecklist
          operationId={operationId}
          projectId={projectId}
          autoFixEnabled={true}
          onExport={handleExport}
          onWaiverSubmit={handleWaiverSubmit}
        />

        {/* Instructions */}
        <div className="mt-8 rounded-lg border bg-white p-6">
          <h3 className="mb-4 text-lg font-semibold text-gray-900">
            How to Use
          </h3>
          <ol className="list-decimal space-y-2 pl-5 text-sm text-gray-700">
            <li>
              <strong>WebSocket Connection:</strong> Component automatically
              connects to WebSocket and subscribes to operation updates
            </li>
            <li>
              <strong>Real-time Updates:</strong> As checks execute in the MAPO
              pipeline, events stream to the UI and update check status
            </li>
            <li>
              <strong>Filtering:</strong> Use the filter controls to narrow down
              checks by status, severity, category, or search text
            </li>
            <li>
              <strong>Expand Checks:</strong> Click any check item to expand and
              view violation details, rationale, and remediation steps
            </li>
            <li>
              <strong>Request Waiver:</strong> For violations that cannot be
              fixed, click "Request Waiver" and provide justification
            </li>
            <li>
              <strong>Auto-fix:</strong> Checks marked as auto-fixable can be
              automatically corrected by clicking "Apply Auto-fix"
            </li>
            <li>
              <strong>Export Report:</strong> Click "Export Report" to download
              a complete compliance report in JSON or PDF format
            </li>
          </ol>
        </div>

        {/* Integration Notes */}
        <div className="mt-6 rounded-lg border bg-gray-50 p-6">
          <h3 className="mb-4 text-lg font-semibold text-gray-900">
            Integration Notes
          </h3>
          <div className="space-y-4 text-sm text-gray-700">
            <div>
              <strong className="text-gray-900">WebSocket Events:</strong>
              <ul className="mt-1 list-disc pl-5">
                <li>
                  <code className="rounded bg-gray-200 px-1 py-0.5">
                    checklist:item:start
                  </code>{" "}
                  - Check execution started
                </li>
                <li>
                  <code className="rounded bg-gray-200 px-1 py-0.5">
                    checklist:item:pass
                  </code>{" "}
                  - Check passed
                </li>
                <li>
                  <code className="rounded bg-gray-200 px-1 py-0.5">
                    checklist:item:fail
                  </code>{" "}
                  - Check failed with violations
                </li>
                <li>
                  <code className="rounded bg-gray-200 px-1 py-0.5">
                    compliance:score:update
                  </code>{" "}
                  - Overall score updated
                </li>
              </ul>
            </div>

            <div>
              <strong className="text-gray-900">Backend Requirements:</strong>
              <ul className="mt-1 list-disc pl-5">
                <li>
                  WebSocket server at{" "}
                  <code className="rounded bg-gray-200 px-1 py-0.5">
                    /ee-design/ws
                  </code>
                </li>
                <li>
                  Compliance validator service (
                  <code className="rounded bg-gray-200 px-1 py-0.5">
                    schematic-compliance-validator.ts
                  </code>
                  )
                </li>
                <li>
                  API endpoints for export and waiver submission
                </li>
              </ul>
            </div>

            <div>
              <strong className="text-gray-900">Environment Variables:</strong>
              <ul className="mt-1 list-disc pl-5">
                <li>
                  <code className="rounded bg-gray-200 px-1 py-0.5">
                    NEXT_PUBLIC_WS_URL
                  </code>{" "}
                  - WebSocket server URL
                </li>
                <li>
                  <code className="rounded bg-gray-200 px-1 py-0.5">
                    NEXT_PUBLIC_WS_PATH
                  </code>{" "}
                  - WebSocket path (default: /ee-design/ws)
                </li>
              </ul>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

/**
 * Realistic Placeholder Data Generator
 *
 * Simulates check execution over time for development/testing.
 * In production, this data comes from WebSocket events.
 */
export function generatePlaceholderCheckResults(): Map<
  string,
  ChecklistItemResult
> {
  const results = new Map<string, ChecklistItemResult>();

  // Simulate some passed checks
  results.set("nasa-power-02", {
    id: "nasa-power-02",
    status: CheckStatus.PASSED,
    violationCount: 0,
    violations: [],
    warnings: [],
    info: ["All power/ground nets use proper symbols"],
    autoFixed: false,
    fixedCount: 0,
    startedAt: new Date(Date.now() - 5000).toISOString(),
    completedAt: new Date(Date.now() - 4800).toISOString(),
    duration: 200,
  });

  // Simulate a failed check with violations
  results.set("nasa-power-01", {
    id: "nasa-power-01",
    status: CheckStatus.FAILED,
    violationCount: 3,
    violations: [
      {
        severity: ViolationSeverity.HIGH,
        message: "IC U1 missing decoupling capacitor on pin 7 (VCC)",
        componentRef: "U1",
        netName: "VCC_3V3",
        pinNumber: "7",
        suggestedFix:
          "Add 0.1µF ceramic capacitor between pin 7 and GND, placed within 5mm of IC",
        autoFixable: false,
      },
      {
        severity: ViolationSeverity.HIGH,
        message: "IC U5 missing decoupling capacitor on pin 14 (VDDA)",
        componentRef: "U5",
        netName: "VDDA",
        pinNumber: "14",
        suggestedFix:
          "Add 0.1µF ceramic capacitor between pin 14 and GNDA, placed within 5mm of IC",
        autoFixable: false,
      },
      {
        severity: ViolationSeverity.MEDIUM,
        message: "IC U3 decoupling capacitor too far from IC (12mm)",
        componentRef: "U3",
        netName: "VCC_5V",
        actualValue: "12mm",
        expectedValue: "< 5mm",
        suggestedFix: "Move C8 closer to U3, ideally within 5mm",
        autoFixable: false,
      },
    ],
    warnings: [],
    info: [],
    autoFixed: false,
    fixedCount: 0,
    startedAt: new Date(Date.now() - 3000).toISOString(),
    completedAt: new Date(Date.now() - 2500).toISOString(),
    duration: 500,
  });

  // Simulate auto-fixed check
  results.set("nasa-conn-02", {
    id: "nasa-conn-02",
    status: CheckStatus.PASSED,
    violationCount: 0,
    violations: [],
    warnings: [],
    info: ["Auto-fixed 5 net names to match convention"],
    autoFixed: true,
    fixedCount: 5,
    startedAt: new Date(Date.now() - 2000).toISOString(),
    completedAt: new Date(Date.now() - 1800).toISOString(),
    duration: 200,
  });

  // Simulate running check
  results.set("mil-conn-01", {
    id: "mil-conn-01",
    status: CheckStatus.RUNNING,
    violationCount: 0,
    violations: [],
    warnings: [],
    info: [],
    autoFixed: false,
    fixedCount: 0,
    startedAt: new Date().toISOString(),
  });

  return results;
}
