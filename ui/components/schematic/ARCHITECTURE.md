# SchematicQualityChecklist - Component Architecture

## Component Hierarchy

```
SchematicQualityChecklist (Main Container)
├── Header
│   ├── Title + Description
│   └── Export Button
│
├── ComplianceScoreCard
│   ├── Circular Progress Ring (SVG)
│   ├── Score Display (0-100)
│   ├── Pass/Fail Counts
│   └── Violations by Severity
│       ├── Critical
│       ├── High
│       ├── Medium
│       ├── Low
│       └── Info
│
├── ComplianceFilters
│   ├── Search Bar
│   ├── Quick Toggles
│   │   ├── Show Only Violations
│   │   └── Auto-fixable Only
│   └── Advanced Filters (Collapsible)
│       ├── Status Filter
│       ├── Severity Filter
│       ├── Category Filter
│       └── Standard Filter
│
└── Checklist Items (Grouped by Standard)
    ├── Standard Section: NASA-STD-8739.4
    │   ├── ChecklistItem
    │   │   ├── Status Icon + Name
    │   │   ├── Severity Badge
    │   │   ├── Violation Count
    │   │   └── Expandable Content
    │   │       ├── Rationale
    │   │       ├── Remediation
    │   │       ├── Standards References
    │   │       ├── ViolationDetail (List)
    │   │       │   ├── Severity + Message
    │   │       │   ├── Component/Net/Pin Info
    │   │       │   ├── Suggested Fix
    │   │       │   └── Actions
    │   │       │       ├── Request Waiver Button
    │   │       │       └── Apply Auto-fix Button
    │   │       ├── Warnings (List)
    │   │       └── Info (List)
    │   └── [More ChecklistItems...]
    │
    ├── Standard Section: MIL-STD-883
    │   └── [ChecklistItems...]
    │
    ├── Standard Section: IPC-2221
    │   └── [ChecklistItems...]
    │
    ├── Standard Section: IEC-61000
    │   └── [ChecklistItems...]
    │
    ├── Standard Section: IPC-7351
    │   └── [ChecklistItems...]
    │
    └── Standard Section: Professional Best Practices
        └── [ChecklistItems...]
```

---

## Data Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                        User Browser                              │
│                                                                  │
│  ┌────────────────────────────────────────────────────────┐    │
│  │    SchematicQualityChecklist Component                  │    │
│  │                                                          │    │
│  │  State:                                                  │    │
│  │  - checkResults: Map<checkId, ChecklistItemResult>      │    │
│  │  - overallScore: number                                 │    │
│  │  - filter: ChecklistFilter                              │    │
│  │  - isValidating: boolean                                │    │
│  │                                                          │    │
│  │  ┌──────────────────────────────────────────────────┐  │    │
│  │  │         useWebSocket Hook                        │  │    │
│  │  │                                                  │  │    │
│  │  │  - Connects to WS server                        │  │    │
│  │  │  - Subscribes to operation events               │  │    │
│  │  │  - Handles incoming events                      │  │    │
│  │  └──────────────────────────────────────────────────┘  │    │
│  │                          ▲                               │    │
│  │                          │                               │    │
│  └──────────────────────────┼───────────────────────────────┘    │
│                             │                                    │
└─────────────────────────────┼────────────────────────────────────┘
                              │
                              │ WebSocket Events
                              │
┌─────────────────────────────┼────────────────────────────────────┐
│                             ▼                                    │
│                    Backend Server                                │
│                                                                  │
│  ┌────────────────────────────────────────────────────────┐    │
│  │          WebSocket Server (/ee-design/ws)              │    │
│  │                                                          │    │
│  │  - Manages client connections                           │    │
│  │  - Routes events to subscribed clients                  │    │
│  │  - Emits compliance events                              │    │
│  └────────────────────────────────────────────────────────┘    │
│                             ▲                                    │
│                             │                                    │
│  ┌────────────────────────────────────────────────────────┐    │
│  │     Compliance Validator Service                        │    │
│  │     (schematic-compliance-validator.ts)                 │    │
│  │                                                          │    │
│  │  - Runs compliance checks                               │    │
│  │  - Emits events for each check                          │    │
│  │  - Calculates overall score                             │    │
│  │                                                          │    │
│  │  Check Functions (51 total):                            │    │
│  │  - nasa-power-01: Power Decoupling Check               │    │
│  │  - nasa-power-02: Power Symbol Check                   │    │
│  │  - mil-conn-01: ESD Protection Check                   │    │
│  │  - ipc-power-01: Trace Width Check                     │    │
│  │  - [48 more checks...]                                  │    │
│  └────────────────────────────────────────────────────────┘    │
│                             ▲                                    │
│                             │                                    │
│  ┌────────────────────────────────────────────────────────┐    │
│  │      MAPO v3.0 Schematic Pipeline                       │    │
│  │                                                          │    │
│  │  - Symbol Resolution                                    │    │
│  │  - Connection Inference                                 │    │
│  │  - Component Placement                                  │    │
│  │  - Wire Routing                                         │    │
│  │  - Smoke Test                                           │    │
│  │  - Compliance Validation ◄── Triggers checks           │    │
│  └────────────────────────────────────────────────────────┘    │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

---

## WebSocket Event Flow

```
Timeline: Check Execution

Backend                          WebSocket                      Frontend
───────                          ─────────                      ────────

Start Check                           │                              │
    │                                 │                              │
    ├─► checklist:item:start ─────────┼──────────────────────────────┤
    │   {                             │                              │
    │     checkId: "nasa-power-01"    │                              │
    │     timestamp: "..."            │                              │
    │   }                             │                              │
    │                                 │                              ├─► Update State
    │                                 │                              │   status = RUNNING
    │                                 │                              │
Execute Check Logic                   │                              │
    │                                 │                              │
    │ (2 seconds pass)                │                              │
    │                                 │                              │
Check Complete                        │                              │
    │                                 │                              │
    ├─► checklist:item:fail ──────────┼──────────────────────────────┤
    │   {                             │                              │
    │     checkId: "nasa-power-01"    │                              │
    │     duration: 2000              │                              │
    │     violationCount: 3           │                              │
    │     violations: [...]           │                              │
    │   }                             │                              │
    │                                 │                              ├─► Update State
    │                                 │                              │   status = FAILED
    │                                 │                              │   violations = [...]
    │                                 │                              │
    ├─► compliance:score:update ──────┼──────────────────────────────┤
    │   {                             │                              │
    │     score: 82                   │                              │
    │     passedChecks: 42            │                              │
    │     totalChecks: 51             │                              │
    │   }                             │                              │
    │                                 │                              ├─► Update State
    │                                 │                              │   score = 82
    │                                 │                              │
    │                                 │                              ▼
    │                                 │                         Re-render UI
    │                                 │                         with updated data
```

---

## State Management

### Component State (Local)

```typescript
// SchematicQualityChecklist.tsx
const [filter, setFilter] = useState<ChecklistFilter>({
  hasViolations: false,
});

const [checkResults, setCheckResults] = useState<Map<string, ChecklistItemResult>>(
  new Map()
);

const [overallScore, setOverallScore] = useState(0);
const [isValidating, setIsValidating] = useState(false);
```

### Computed State (useMemo)

```typescript
// Summary statistics
const summary = useMemo(() => {
  let passedChecks = 0;
  let failedChecks = 0;
  let totalViolations = 0;
  // ... compute from checkResults
  return { passedChecks, failedChecks, totalViolations, ... };
}, [checkResults]);

// Filtered checks
const filteredChecks = useMemo(() => {
  // Apply filter.status, filter.severity, filter.searchText, etc.
  return DEFAULT_CHECKLIST.filter(...);
}, [filter, checkResults]);

// Checks grouped by standard
const checksByStandard = useMemo(() => {
  // Group filteredChecks by standard
  return new Map<ComplianceStandard, ChecklistItemDefinition[]>();
}, [filteredChecks]);
```

### WebSocket State (useWebSocket Hook)

```typescript
const { socket, isConnected } = useWebSocket({
  autoConnect: true,
});

// Subscribe to events
useEffect(() => {
  socket?.on("checklist:item:start", handleCheckStart);
  socket?.on("checklist:item:pass", handleCheckPass);
  socket?.on("checklist:item:fail", handleCheckFail);
  // ...
  return () => {
    socket?.off("checklist:item:start");
    // ... cleanup
  };
}, [socket, isConnected]);
```

---

## Performance Optimizations

### 1. Memoization
- `useMemo` for expensive computations (filtering, grouping)
- `useCallback` for event handlers (prevent re-renders)

### 2. State Updates
- Map-based state for O(1) lookup of check results
- Batch updates via `checklist:batch:update` event

### 3. Rendering
- Conditional rendering (only render expanded content when needed)
- Virtual scrolling (future enhancement for 100+ checks)

### 4. WebSocket
- Single connection shared across all components
- Event batching for high-frequency updates
- Automatic reconnection on disconnect

---

## Error Handling

### WebSocket Errors
```typescript
socket?.on("connect_error", (error) => {
  console.error("[WebSocket] Connection error:", error);
  // Show error banner to user
});

socket?.on("disconnect", (reason) => {
  console.log("[WebSocket] Disconnected:", reason);
  // Show "Offline" indicator
});
```

### Check Execution Errors
```typescript
socket?.on("checklist:item:fail", (event) => {
  // Check failed - display violations
  setCheckResults((prev) => {
    const updated = new Map(prev);
    updated.set(event.checkId, {
      status: CheckStatus.FAILED,
      violations: event.violations,
      // ...
    });
    return updated;
  });
});
```

### User Input Validation
```typescript
// Search text sanitization
const handleSearchChange = (searchText: string) => {
  // Trim whitespace, limit length
  const sanitized = searchText.trim().slice(0, 100);
  onFilterChange({ ...filter, searchText: sanitized });
};
```

---

## Accessibility (WCAG 2.1)

### Keyboard Navigation
- All interactive elements keyboard accessible
- Tab order follows visual flow
- Escape key to close expanded items

### Screen Readers
- ARIA labels on all icons and buttons
- Role attributes on custom controls
- Live regions for dynamic updates

### Visual
- Sufficient color contrast (4.5:1 minimum)
- Icons paired with text labels
- Focus indicators on all interactive elements

### Example ARIA Attributes
```tsx
<button
  aria-label="Expand check details"
  aria-expanded={isExpanded}
  onClick={() => setIsExpanded(!isExpanded)}
>
  {isExpanded ? <ChevronDown /> : <ChevronRight />}
</button>
```

---

## Testing Strategy

### Unit Tests (Jest + React Testing Library)
```typescript
describe("ComplianceScoreCard", () => {
  it("displays correct score", () => {
    render(<ComplianceScoreCard score={85} ... />);
    expect(screen.getByText("85")).toBeInTheDocument();
  });

  it("shows green color for score >= 90", () => {
    render(<ComplianceScoreCard score={95} ... />);
    expect(screen.getByText("95")).toHaveClass("text-green-600");
  });
});
```

### Integration Tests
```typescript
describe("SchematicQualityChecklist Integration", () => {
  it("updates check status on WebSocket event", async () => {
    const { mockSocket } = renderWithWebSocket(
      <SchematicQualityChecklist ... />
    );

    mockSocket.emit("checklist:item:pass", {
      checkId: "nasa-power-01",
      timestamp: "...",
    });

    await waitFor(() => {
      expect(screen.getByText("Passed")).toBeInTheDocument();
    });
  });
});
```

### E2E Tests (Playwright)
```typescript
test("complete compliance workflow", async ({ page }) => {
  await page.goto("/dashboard/ee-design/schematic");
  await page.click("text=Quality Checklist");

  // Wait for checks to complete
  await page.waitForSelector("text=51 checks");

  // Filter by failed only
  await page.click("text=Show Only Violations");
  await expect(page.locator(".checklist-item")).toHaveCount(8);

  // Export report
  const downloadPromise = page.waitForEvent("download");
  await page.click("text=Export Report");
  const download = await downloadPromise;
  expect(download.suggestedFilename()).toContain("compliance-report");
});
```

---

**Built for MAPO v3.0 with real-time WebSocket updates and production-grade architecture**
