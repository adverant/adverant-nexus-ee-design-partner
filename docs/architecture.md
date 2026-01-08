# EE Design Partner - Architecture Documentation

## System Overview

EE Design Partner is a modular, microservices-based architecture for end-to-end hardware/software development automation.

## Core Components

### 1. Backend Service (nexus-ee-design)

**Technology Stack:**
- TypeScript + Node.js 20+
- Express.js REST API
- Socket.IO for real-time updates
- PostgreSQL for metadata
- Redis for job queues (BullMQ)

**Directory Structure:**
```
services/nexus-ee-design/
├── src/
│   ├── index.ts              # Main entry point
│   ├── config.ts             # Configuration management
│   ├── state.ts              # Shared application state
│   ├── api/
│   │   ├── routes.ts         # Main API router
│   │   └── skills-routes.ts  # Skills Engine endpoints
│   ├── services/
│   │   ├── pcb/              # PCB Layout services
│   │   │   ├── ralph-loop.ts           # Tournament orchestrator
│   │   │   ├── validation-framework.ts # 8-domain validator
│   │   │   ├── python-executor.ts      # KiCad automation
│   │   │   └── agents/                 # 5 layout agents
│   │   ├── schematic/        # Schematic services
│   │   │   ├── schematic-generator.ts
│   │   │   └── schematic-reviewer.ts
│   │   ├── simulation/       # Simulation orchestration
│   │   │   └── simulation-orchestrator.ts
│   │   ├── firmware/         # Firmware generation
│   │   │   └── firmware-generator.ts
│   │   ├── validation/       # Multi-LLM validation
│   │   │   └── consensus-engine.ts
│   │   └── skills/           # Skills Engine client
│   │       └── skills-engine-client.ts
│   ├── types/                # TypeScript definitions
│   └── utils/                # Shared utilities
├── python-scripts/           # KiCad automation scripts
└── Dockerfile
```

### 2. Frontend UI

**Technology Stack:**
- Next.js 14 (App Router)
- TypeScript
- Tailwind CSS
- Radix UI components
- XTerm.js for terminal
- React Three Fiber for 3D
- Zustand for state management
- Socket.IO client

**Directory Structure:**
```
ui/
├── app/
│   ├── layout.tsx
│   ├── page.tsx
│   └── globals.css
├── components/
│   ├── Header.tsx
│   ├── PipelineStatus.tsx
│   ├── Terminal.tsx
│   ├── ProjectBrowser.tsx
│   ├── DesignTabs.tsx
│   ├── ValidationPanel.tsx
│   └── ui/
│       └── resizable.tsx
├── hooks/
│   ├── useEEDesignStore.ts
│   └── useWebSocket.ts
└── lib/
    └── utils.ts
```

### 3. Skills Engine Integration

All 40+ skills are defined as markdown files with YAML frontmatter:

```yaml
---
name: pcb-layout
displayName: "PCB Layout Generator"
description: "Generate optimized PCB layouts using multi-agent tournament"
version: 1.0.0
status: published
visibility: organization

allowed-tools:
  - Read
  - Write
  - Edit
  - Bash
  - Task

triggers:
  - /pcb-layout
  - /layout

capabilities:
  - name: generate
    description: Generate PCB layout
    parameters:
      - name: strategy
        type: string
        required: false
        description: "Layout strategy"
---
```

**Skill Categories by Phase:**
- Phase 1: research-paper, patent-search, market-analysis, requirements-gen
- Phase 2: ee-architecture, component-select, bom-optimize, power-budget
- Phase 3: schematic-gen, schematic-review, netlist-gen
- Phase 4: simulate-spice, simulate-thermal, simulate-si, simulate-rf, simulate-emc
- Phase 5: pcb-layout, pcb-review, stackup-design, via-optimize
- Phase 6: gerber-gen, dfm-check, vendor-quote, panelize
- Phase 7: firmware-gen, hal-gen, driver-gen, rtos-config, build-setup
- Phase 8: test-gen, hil-setup, test-procedure, coverage-analysis
- Phase 9: manufacture, assembly-guide, quality-check, traceability
- Phase 10: debug-assist, service-manual, rma-process, firmware-update

## Multi-Agent PCB Layout Tournament (Ralph Loop)

### Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Ralph Loop Orchestrator                   │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐       │
│  │ Conservative │  │   Compact    │  │   Thermal    │       │
│  │    Agent     │  │    Agent     │  │    Agent     │       │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘       │
│         │                 │                 │                │
│         ▼                 ▼                 ▼                │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐       │
│  │     EMI      │  │     DFM      │  │   Python     │       │
│  │    Agent     │  │    Agent     │  │  Executor    │       │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘       │
│         │                 │                 │                │
│         └─────────────────┼─────────────────┘                │
│                           │                                  │
│                           ▼                                  │
│              ┌───────────────────────┐                       │
│              │  Validation Framework │                       │
│              │     (8 Domains)       │                       │
│              └───────────┬───────────┘                       │
│                          │                                   │
│                          ▼                                   │
│              ┌───────────────────────┐                       │
│              │   Consensus Engine    │                       │
│              │  (Claude + Gemini)    │                       │
│              └───────────────────────┘                       │
└─────────────────────────────────────────────────────────────┘
```

### Agent Strategies

| Agent | Focus Areas | Strategy Weights |
|-------|-------------|------------------|
| Conservative | Reliability, DFM | thermal: 0.3, reliability: 0.3, dfm: 0.2, size: 0.1, cost: 0.1 |
| Aggressive Compact | Size, Cost | size: 0.4, cost: 0.3, dfm: 0.2, thermal: 0.1 |
| Thermal Optimized | Heat dissipation | thermal: 0.5, reliability: 0.25, dfm: 0.15, size: 0.1 |
| EMI Optimized | Signal integrity | signal_integrity: 0.4, emi: 0.3, reliability: 0.2, size: 0.1 |
| DFM Optimized | Manufacturing | dfm: 0.5, cost: 0.25, reliability: 0.15, size: 0.1 |

### Validation Domains

1. **DRC** - Design Rule Check (KiCad rules)
2. **ERC** - Electrical Rule Check
3. **IPC-2221** - PCB standards compliance
4. **Signal Integrity** - Impedance, crosstalk, timing
5. **Thermal** - Temperature distribution, hotspots
6. **DFM** - Manufacturability analysis
7. **Best Practices** - Industry conventions
8. **Testing** - Test point accessibility

## Multi-LLM Validation Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      Design Artifact                         │
└──────────────────────────┬──────────────────────────────────┘
                           │
           ┌───────────────┼───────────────┐
           │               │               │
           ▼               ▼               ▼
┌──────────────────┐ ┌──────────────────┐ ┌──────────────────┐
│  Claude Opus 4   │ │  Gemini 2.5 Pro  │ │  Domain Expert   │
│                  │ │  (OpenRouter)    │ │  Validators      │
│  Weight: 0.4     │ │  Weight: 0.3     │ │  Weight: 0.3     │
│                  │ │                  │ │                  │
│  - Primary Gen   │ │  - Cross-check   │ │  - DRC Engine    │
│  - Analysis      │ │  - Alternatives  │ │  - SPICE Verify  │
│  - Code Gen      │ │  - Red Team      │ │  - Thermal FEA   │
└────────┬─────────┘ └────────┬─────────┘ └────────┬─────────┘
         │                    │                    │
         └────────────────────┼────────────────────┘
                              │
                    ┌─────────▼─────────┐
                    │  Consensus Engine │
                    │                   │
                    │  - Vote merging   │
                    │  - Conflict res   │
                    │  - Score calc     │
                    └─────────┬─────────┘
                              │
                    ┌─────────▼─────────┐
                    │  Final Validation │
                    │     Report        │
                    │                   │
                    │  Score: 97.2/100  │
                    └───────────────────┘
```

## Simulation Orchestration

### Container Management

| Simulation | Docker Image | Port | Purpose |
|------------|--------------|------|---------|
| SPICE | ngspice:latest | 9101 | Circuit simulation |
| Thermal | openfoam:latest | 9102 | CFD thermal analysis |
| Thermal FEA | elmerfem:latest | 9103 | Finite element thermal |
| RF/EMC | openems:latest | 9104 | Electromagnetic simulation |
| SI | custom:si | 9105 | Signal integrity analysis |

### Job Queue Architecture

```typescript
interface SimulationJob {
  id: string;
  type: SimulationType;
  projectId: string;
  priority: 'low' | 'normal' | 'high' | 'critical';
  config: SimulationConfig;
  status: 'queued' | 'running' | 'completed' | 'failed';
}
```

Jobs are managed via BullMQ with Redis as the backing store.

## API Endpoints Summary

### Projects
- `GET /api/v1/projects` - List projects
- `POST /api/v1/projects` - Create project
- `GET /api/v1/projects/:id` - Get project
- `POST /api/v1/projects/:id/requirements` - Generate requirements

### Schematic
- `POST /api/v1/projects/:id/schematic/generate` - Generate schematic
- `POST /api/v1/projects/:id/schematic/upload` - Upload schematic
- `POST /api/v1/projects/:id/schematic/:schematicId/validate` - Validate schematic

### Simulation
- `POST /api/v1/projects/:id/simulation/spice` - Run SPICE simulation
- `POST /api/v1/projects/:id/simulation/thermal` - Run thermal simulation
- `POST /api/v1/projects/:id/simulation/signal-integrity` - Run SI analysis
- `POST /api/v1/projects/:id/simulation/rf-emc` - Run RF/EMC simulation
- `GET /api/v1/projects/:id/simulation/:simulationId` - Get results

### PCB Layout
- `POST /api/v1/projects/:id/pcb-layout/generate` - Generate layout
- `GET /api/v1/projects/:id/pcb-layout/:layoutId` - Get layout status
- `POST /api/v1/projects/:id/pcb-layout/:layoutId/validate` - Validate layout

### Manufacturing
- `POST /api/v1/projects/:id/manufacturing/gerbers` - Generate Gerbers
- `POST /api/v1/projects/:id/manufacturing/quote` - Get vendor quote
- `POST /api/v1/projects/:id/manufacturing/order` - Place order

### Firmware
- `POST /api/v1/projects/:id/firmware/generate` - Generate firmware project
- `POST /api/v1/projects/:id/firmware/:firmwareId/hal` - Generate HAL code
- `POST /api/v1/projects/:id/firmware/:firmwareId/driver` - Generate driver

### Skills
- `GET /api/v1/skills` - List all skills
- `POST /api/v1/skills/search` - Search skills
- `GET /api/v1/skills/:skillName` - Get skill details
- `GET /api/v1/skills/phase/:phase` - Get skills by phase
- `POST /api/v1/skills/execute` - Execute skill
- `POST /api/v1/skills/register` - Register new skill
- `GET /api/v1/skills/pipeline/phases` - Get all phases with skills

### Validation
- `POST /api/v1/projects/:id/validate/multi-llm` - Run multi-LLM validation

## WebSocket Events

### Project Events
- `project:updated` - Project metadata changed
- `project:phase:changed` - Pipeline phase changed

### Simulation Events
- `simulation:started` - Simulation job started
- `simulation:progress` - Progress update (0-100)
- `simulation:completed` - Simulation finished
- `simulation:failed` - Simulation failed

### Layout Events
- `layout:started` - Layout generation started
- `layout:iteration` - Tournament iteration completed
- `layout:agent:scored` - Individual agent scored
- `layout:completed` - Layout generation finished

### Validation Events
- `validation:started` - Validation started
- `validation:domain:completed` - Domain validation finished
- `validation:completed` - All validation complete

## Deployment

### Kubernetes Manifests

Located in `/k8s/`:
- `deployment.yaml` - Service deployment
- `service.yaml` - ClusterIP service
- `configmap.yaml` - Configuration
- `hpa.yaml` - Horizontal pod autoscaler

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `PORT` | HTTP server port | 9080 |
| `NODE_ENV` | Environment | development |
| `NEXUS_API_KEY` | API authentication | - |
| `NEXUS_GRAPHRAG_URL` | GraphRAG service URL | http://localhost:9000 |
| `NEXUS_MAGEAGENT_URL` | MageAgent URL | http://localhost:9010 |
| `OPENROUTER_API_KEY` | For Gemini access | - |
| `REDIS_URL` | Redis connection | redis://localhost:6379 |
| `DATABASE_URL` | PostgreSQL connection | - |

## Security Considerations

1. **Authentication**: All API requests require `Authorization: Bearer <token>`
2. **Input Validation**: Zod schemas validate all request bodies
3. **Rate Limiting**: Configurable per endpoint
4. **CORS**: Configured for allowed origins
5. **File Upload**: Size limits and type validation
6. **Sandboxed Execution**: Simulations run in Docker containers