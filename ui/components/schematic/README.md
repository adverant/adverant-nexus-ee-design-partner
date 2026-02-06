# Schematic Quality Checklist Components

**MAPO v3.0 - Real-time Compliance Validation UI**

A comprehensive React component system for displaying real-time schematic compliance validation with WebSocket updates. Supports 51 checks across 6 industry standards (NASA, MIL-STD, IPC, IEC, Best Practices).

---

## Components

### 1. SchematicQualityChecklist.tsx (Main Container)

**Purpose**: Main container component that orchestrates the entire checklist UI.

**Props**:
```typescript
{
  operationId: string;          // Schematic operation ID to track
  projectId: string;            // Project ID
  showFailedOnly?: boolean;     // Show only failed checks initially
  autoFixEnabled?: boolean;     // Enable auto-fix functionality
  onExport?: (report) => void;  // Callback when export button clicked
  onWaiverSubmit?: (checkId, justification) => void; // Callback for waiver submission
  className?: string;
}
```

**Features**:
- WebSocket integration for real-time check updates
- Overall compliance score display
- Filter controls
- Grouped checklist items by standard
- Export report functionality
- Waiver request handling

**WebSocket Events Subscribed**:
- `checklist:item:start` - Check execution started
- `checklist:item:pass` - Check passed
- `checklist:item:fail` - Check failed with violations
- `checklist:batch:update` - Batch update of multiple checks
- `compliance:score:update` - Overall score updated

---

### 2. ComplianceScoreCard.tsx

**Purpose**: Large score display with pass/fail statistics and color gradient.

**Props**:
```typescript
{
  score: number;                // 0-100 compliance score
  passedChecks: number;         // Number of passed checks
  totalChecks: number;          // Total number of checks
  violationCount: number;       // Total violations found
  violationsBySeverity: Record<ViolationSeverity, number>; // Violations by severity
  isLoading: boolean;           // Loading state
}
```

**Visual Design**:
- Circular progress ring with score (0-100)
- Color gradient: Red (< 70) → Yellow (70-89) → Green (≥ 90)
- Violation counts by severity (Critical, High, Medium, Low, Info)
- Status badge (Excellent / Good / Needs Work)

---

### 3. ComplianceFilters.tsx

**Purpose**: Advanced filtering controls for checklist items.

**Props**:
```typescript
{
  filter: ChecklistFilter;                    // Current filter state
  onFilterChange: (filter) => void;           // Filter change callback
  className?: string;
}
```

**Filter Options**:
- **Search**: Text search across checks, violations, components
- **Status**: Pending, Running, Passed, Failed, Warning, Skipped
- **Severity**: Critical, High, Medium, Low, Info
- **Category**: Power, Connectivity, Symbols, Placement, Standards, Documentation
- **Standard**: NASA-STD-8739.4, MIL-STD-883, IPC-2221, IEC-61000, IPC-7351, Best Practices
- **Quick Toggles**: Show only violations, Auto-fixable only

---

### 4. ChecklistItem.tsx

**Purpose**: Individual check display with expandable violation details.

**Props**:
```typescript
{
  definition: ChecklistItemDefinition;  // Static check definition
  result: ChecklistItemResult;          // Runtime result
  autoFixEnabled: boolean;              // Is auto-fix enabled?
  onWaiverRequest?: (checkId) => void;  // Waiver request callback
  onViolationClick?: (violation) => void; // Violation click callback
}
```

**Features**:
- Status icon with color coding
- Severity badge
- Violation count
- Duration display
- Auto-fix badge (if applied)
- Expandable section showing:
  - Rationale (why this matters)
  - Remediation (how to fix)
  - Standards references
  - Violation details
  - Warnings and info messages

---

### 5. ViolationDetail.tsx

**Purpose**: Detailed violation information display.

**Props**:
```typescript
{
  violation: ViolationDetail;              // Violation data
  onWaiverRequest?: (violation) => void;   // Waiver request callback
  onViolationClick?: (violation) => void;  // Click callback
  className?: string;
}
```

**Displays**:
- Severity badge and icon
- Violation message
- Component reference (e.g., "U1", "R5")
- Net name (e.g., "+3V3", "GND")
- Pin number
- Symbol library
- Property name (for property checks)
- Expected vs. Actual values
- File path and line number
- Suggested fix
- Auto-fixable badge
- "Request Waiver" button
- "Apply Auto-fix" button (if auto-fixable)

---

## Usage Example

```tsx
import { SchematicQualityChecklist } from "@/components/schematic";

export default function SchematicTab({ projectId, operationId }) {
  const handleExport = (report) => {
    console.log("Exporting report:", report);
    // Save as JSON/PDF
  };

  const handleWaiverSubmit = (checkId, justification) => {
    console.log("Waiver submitted:", checkId, justification);
    // Submit to API
  };

  return (
    <SchematicQualityChecklist
      operationId={operationId}
      projectId={projectId}
      autoFixEnabled={true}
      onExport={handleExport}
      onWaiverSubmit={handleWaiverSubmit}
    />
  );
}
```

---

## Data Flow

1. **Initialization**: Component loads with 51 checks in PENDING state from `DEFAULT_CHECKLIST`
2. **WebSocket Connection**: Subscribes to `operationId` for real-time updates
3. **Check Execution**: Backend emits events as checks run (`start` → `pass`/`fail`)
4. **State Updates**: Component updates check results and overall score in real-time
5. **Filtering**: User can filter checks by status, severity, category, standard, search text
6. **Export**: User clicks "Export Report" → generates full compliance report with all check results
7. **Waivers**: User clicks "Request Waiver" on violation → submits justification via callback

---

## Styling

All components use:
- **Tailwind CSS** for styling
- **Lucide React** icons
- **Color Scheme**:
  - Green: Passed checks, score ≥ 90
  - Yellow: Warning, score 70-89
  - Red: Failed checks, critical violations, score < 70
  - Blue: Info, running state
  - Gray: Pending, skipped
- **Animations**: Smooth transitions on score changes, spinner on loading

---

## WebSocket Event Format

### Check Started
```typescript
{
  checkId: "nasa-power-01",
  checkName: "Power Decoupling Capacitors",
  timestamp: "2026-02-06T23:10:00Z"
}
```

### Check Failed
```typescript
{
  checkId: "nasa-power-01",
  checkName: "Power Decoupling Capacitors",
  timestamp: "2026-02-06T23:10:02Z",
  duration: 2000,
  violationCount: 3,
  violations: [
    {
      severity: "high",
      message: "IC U1 missing decoupling capacitor on pin 7 (VCC)",
      componentRef: "U1",
      netName: "VCC_3V3",
      pinNumber: "7",
      suggestedFix: "Add 0.1µF ceramic capacitor between pin 7 and GND",
      autoFixable: false
    }
  ],
  warnings: [],
  autoFixed: false,
  fixedCount: 0
}
```

### Score Update
```typescript
{
  score: 85,
  passedChecks: 43,
  totalChecks: 51,
  violationCount: 12,
  violationsBySeverity: {
    critical: 0,
    high: 3,
    medium: 6,
    low: 3,
    info: 0
  },
  timestamp: "2026-02-06T23:10:05Z"
}
```

---

## File Locations

```
ui/
├── components/schematic/
│   ├── SchematicQualityChecklist.tsx   (491 lines)
│   ├── ChecklistItem.tsx               (285 lines)
│   ├── ComplianceScoreCard.tsx         (233 lines)
│   ├── ComplianceFilters.tsx           (279 lines)
│   ├── ViolationDetail.tsx             (230 lines)
│   ├── index.ts                        (11 lines)
│   └── README.md                       (this file)
├── types/
│   └── schematic-quality.ts            (473 lines)
├── data/
│   └── default-checklist.ts            (711 lines)
└── hooks/
    └── useWebSocket.ts                 (513 lines)
```

**Total**: 1,529 lines of component code + 1,697 lines of supporting code = **3,226 lines**

---

## Dependencies

- React 18+
- TypeScript
- Tailwind CSS
- Lucide React (icons)
- Socket.IO Client (WebSocket)
- Next.js (for `@/` path aliases)

---

## Integration Checklist

- [ ] Add `@/types` path alias to `tsconfig.json`
- [ ] Add `@/data` path alias to `tsconfig.json`
- [ ] Add `@/hooks` path alias to `tsconfig.json`
- [ ] Add `@/lib/utils` with `cn()` utility (if not present)
- [ ] Configure WebSocket backend to emit compliance events
- [ ] Add API endpoints for export and waiver submission
- [ ] Test with realistic placeholder data
- [ ] Integrate into schematics tab as collapsible panel

---

## Future Enhancements

1. **Schematic Highlighting**: Click violation → highlight component in KiCanvas viewer
2. **Auto-fix Implementation**: Actually apply fixes to schematic file
3. **Waiver Workflow**: Full approval/rejection workflow with notifications
4. **Report Templates**: PDF export with customizable templates
5. **Historical Tracking**: Show compliance score trends over time
6. **AI Suggestions**: LLM-powered fix suggestions for complex violations
7. **Batch Operations**: Select multiple checks for bulk waiver/auto-fix

---

Built for **MAPO v3.0** - Deep Ideation Integration + Quality Checklist + Symbol Assembly
