# EE Design Partner - Claude Code Instructions

## Nexus Memory Integration

This project uses automatic memory via the nexus-memory skill globally across all Claude Code sessions:

- **Auto-recall**: Relevant memories are automatically retrieved on every prompt
- **Auto-store**: Every prompt and tool use is captured for future reference
- **Episodes**: Conversation summaries are generated periodically (every 10 tool uses)

All memories are stored in Nexus GraphRAG and can be recalled across any project.

To manually store important context:
```bash
echo '{"content": "Important decision or learning here", "event_type": "decision"}' | ~/.claude/hooks/store-memory.sh
```

To manually recall memories:
```bash
echo '{"query": "search query here"}' | ~/.claude/hooks/recall-memory.sh
```

Event types: `fix`, `decision`, `learning`, `pattern`, `preference`, `context`

---

## Project Overview

**EE Design Partner** is an end-to-end hardware/software development automation platform that integrates:
- **Claude Code** as the central orchestrator (terminal-first UX)
- **Gemini 2.5 Pro** via OpenRouter for multi-LLM validation
- **KiCad WASM** for browser-native schematic and PCB editing
- **Comprehensive simulation suite** (SPICE, Thermal, SI, RF, EMC)
- **Skills Engine** for dynamic workflow automation
- **MAPOS** (Multi-Agent PCB Optimization System) for DRC violation reduction

**Repository**: `github.com/adverant/adverant-nexus-ee-design-partner`
**License**: MIT - Adverant Inc. 2024-2026

### Related Adverant Projects

This repository integrates with the broader Adverant ecosystem:

| Repository | Purpose | CLAUDE.md |
|------------|---------|-----------|
| **Adverant-Nexus** | Backend microservices (GraphRAG, MageAgent) | `/Users/don/Adverant/Adverant-Nexus/CLAUDE.md` |
| **nexus-dashboard** | User dashboard (dashboard.adverant.ai) | `/Users/don/Adverant/nexus-dashboard/CLAUDE.md` |
| **adverant-nexus-ee-design-partner** | This repo - EE Design automation | This file |
| **foc-esc-heavy-lift** | Reference hardware design (FOC ESC) | `/Users/don/Adverant/foc-esc-heavy-lift/CLAUDE.md` |

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    NEXUS DASHBOARD - EE DESIGN PARTNER                       │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │              CLAUDE CODE TERMINAL (Central Element)                   │  │
│  │  $ /ee-design analyze-requirements "200A FOC ESC"                    │  │
│  │  $ /schematic-gen power-stage --mosfets=18 --topology=3phase         │  │
│  │  $ /pcb-layout generate --strategy=thermal --layers=10               │  │
│  │  $ /mapos optimize --target=100                                      │  │
│  │  $ /firmware-gen stm32h755 --foc --triple-redundant                  │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
│                                                                              │
│  ┌─────────────────────┐  ┌─────────────────────┐  ┌────────────────────┐  │
│  │   PROJECT REPO      │  │   DESIGN TABS       │  │   VALIDATION       │  │
│  │   (VSCode-like)     │  │   (KiCad WASM)      │  │   PANEL            │  │
│  └─────────────────────┘  └─────────────────────┘  └────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Infrastructure & Deployment

### GitHub Repository Access

**Adverant Organization**: `github.com/adverant`

**Authentication**:
- Use GitHub Personal Access Token (PAT) for repository access
- Never commit tokens to repositories

### Kubernetes Deployment

**Deployment Details:**
- **Service Name**: `nexus-ee-design`
- **MAPOS Worker**: `mapos-kicad-worker`
- **Namespace**: `nexus`
- **Server**: 157.173.102.118
- **Cluster**: K3s
- **Image Registry**: `localhost:5000/`

### Deployment Commands

**🚫 CRITICAL: NEVER BUILD DOCKER IMAGES LOCALLY!**

All Docker builds MUST happen on the remote server (157.173.102.118).

```bash
# ❌ WRONG - Never run locally
docker build .

# ✅ CORRECT - Use the deploy skill
/build-deploy
```

**Manual Deployment (if needed):**
```bash
ssh root@157.173.102.118 << 'EOF'
cd /opt/nexus-ee-design
git pull origin main
BUILD_TAG="nexus-ee-design-$(date +%Y%m%d-%H%M%S)-$(git rev-parse --short HEAD)"
docker build --no-cache -t nexus-ee-design:${BUILD_TAG} -f services/nexus-ee-design/Dockerfile .
docker tag nexus-ee-design:${BUILD_TAG} localhost:5000/nexus-ee-design:${BUILD_TAG}
docker push localhost:5000/nexus-ee-design:${BUILD_TAG}
k3s kubectl set image deployment/nexus-ee-design nexus-ee-design=localhost:5000/nexus-ee-design:${BUILD_TAG} -n nexus
k3s kubectl rollout status deployment/nexus-ee-design -n nexus
echo "Deployed: ${BUILD_TAG}"
EOF
```

### Verification

After deployment, verify:

1. **Check pod status:**
```bash
ssh root@157.173.102.118 "k3s kubectl get pods -n nexus -l app.kubernetes.io/name=nexus-ee-design"
```

2. **Check MAPOS worker:**
```bash
ssh root@157.173.102.118 "k3s kubectl get pods -n nexus -l app.kubernetes.io/name=mapos-kicad-worker"
```

3. **Check logs:**
```bash
ssh root@157.173.102.118 "k3s kubectl logs -f deployment/nexus-ee-design -n nexus"
```

---

## Task Prioritization Rules

**CRITICAL**: Before taking ANY action, check these in order:

### Priority 1: Current Task Context
1. **User's Last Direct Instruction** - Always complete the user's most recent explicit request first
2. **Understand the Immediate Goal** - Before fixing anything, confirm what the user wants

### Priority 2: When to Address Code Quality Issues
**ONLY** address code quality issues when:
- ✅ They **block the current task**
- ✅ User **explicitly asks** to fix them
- ✅ Current task is **100% complete** and user approves moving to next task

**DO NOT** start fixing unrelated code issues when:
- ❌ User has given a specific task
- ❌ Errors are **pre-existing** and don't block current work
- ❌ You see TypeScript warnings/errors in unrelated files

### Priority 3: Session Resumption Protocol
When a session resumes after context compaction:
1. **Read the continuation summary carefully** - What was the LAST thing the user asked for?
2. **Check git status** - What files are modified?
3. **ASK before switching tasks**
4. **Don't assume** - Just because you see problems doesn't mean you should fix them now

---

## 10-Phase Pipeline

| Phase | Description | Key Skills |
|-------|-------------|------------|
| 1 | **Ideation & Research** | `/research-paper`, `/patent-search`, `/requirements-gen` |
| 2 | **Architecture** | `/ee-architecture`, `/component-select`, `/bom-optimize` |
| 3 | **Schematic Capture** | `/schematic-gen`, `/schematic-review`, `/netlist-gen` |
| 4 | **Simulation** | `/simulate-spice`, `/simulate-thermal`, `/simulate-si`, `/simulate-rf` |
| 5 | **PCB Layout** | `/pcb-layout`, `/mapos`, `/stackup-design` |
| 6 | **Manufacturing** | `/gerber-gen`, `/dfm-check`, `/vendor-quote` |
| 7 | **Firmware** | `/firmware-gen`, `/hal-gen`, `/driver-gen` |
| 8 | **Testing** | `/test-gen`, `/hil-setup`, `/test-procedure` |
| 9 | **Production** | `/manufacture`, `/assembly-guide`, `/quality-check` |
| 10 | **Field Support** | `/debug-assist`, `/service-manual`, `/firmware-update` |

---

## Project Structure

```
adverant-nexus-ee-design-partner/
├── services/nexus-ee-design/    # Main service
│   ├── src/
│   │   ├── api/                 # REST API routes
│   │   ├── services/            # Business logic
│   │   │   ├── schematic/       # Schematic generation
│   │   │   ├── pcb/             # PCB layout (MAPOS)
│   │   │   ├── simulation/      # All simulation types
│   │   │   ├── firmware/        # Code generation
│   │   │   ├── manufacturing/   # Vendor integration
│   │   │   └── validation/      # Multi-LLM validation
│   │   ├── types/               # TypeScript types
│   │   └── utils/               # Logger, errors, helpers
│   ├── docker/                  # Docker configurations
│   │   └── mapos/              # MAPOS KiCad worker
│   └── python-scripts/          # KiCad automation
│       └── mapos/              # MAPOS optimization scripts
├── ui/                          # Dashboard UI components
├── skills/                      # Skill definitions (.md)
├── k8s/                         # Kubernetes manifests
│   ├── deployment.yaml          # Main service deployment
│   └── mapos-kicad-worker.yaml  # MAPOS worker deployment
├── docs/                        # Documentation
└── tests/                       # Test suites
```

---

## MAPOS - Multi-Agent PCB Optimization System

MAPOS is the core PCB optimization engine with a 7-phase pipeline:

| Phase | Description | Impact |
|-------|-------------|--------|
| 1 | Design Rules Generation | IPC-2221 compliant .kicad_dru |
| 2 | pcbnew API Fixes | 40-50% violation reduction |
| 3 | Zone Fill | Native ZONE_FILLER API |
| 4 | Net Assignment | Orphan pad net assignment |
| 5 | Solder Mask Fixes | Via tenting, bridge allowance |
| 6 | Silkscreen Fixes | 90%+ silk_over_copper reduction |
| 7 | LLM-Guided Fixes | OpenRouter Claude API |

### MAPOS Python Modules

Located in `services/nexus-ee-design/python-scripts/mapos/`:

| Module | Purpose |
|--------|---------|
| `mapos_pcb_optimizer.py` | Main optimizer orchestration |
| `kicad_mask_fixer.py` | Solder mask bridge fixes |
| `kicad_silk_fixer.py` | Silkscreen layer management |
| `kicad_zone_filler.py` | Zone fill via pcbnew API |
| `kicad_net_assigner.py` | Orphan pad net assignment |
| `kicad_api_server.py` | FastAPI REST endpoints |
| `kicad_headless_runner.py` | K8s headless execution |

### MAPOS K8s Deployment

MAPOS runs in K8s with Xvfb sidecar for headless KiCad:

```yaml
# Deploy MAPOS worker
kubectl apply -f k8s/mapos-kicad-worker.yaml

# Check status
kubectl get pods -n nexus -l app.kubernetes.io/name=mapos-kicad-worker

# Run optimization job
kubectl create job mapos-optimize --from=job/mapos-optimize-job -n nexus
```

---

## Key Services

### Simulation Suite
| Type | Engine | Analysis |
|------|--------|----------|
| SPICE | ngspice/LTspice | DC, AC, Transient, Noise, Monte Carlo |
| Thermal | OpenFOAM/Elmer | Steady-state, Transient, CFD |
| Signal Integrity | scikit-rf | Impedance, Crosstalk, Eye diagram |
| RF/EMC | openEMS/MEEP | Field patterns, S11, Emissions |

### Multi-LLM Validation
```
Design Artifact → Claude Opus 4 (Primary)
                → Gemini 2.5 Pro (Validator)
                → Domain Expert Validators
                → Consensus Engine → Final Score
```

### Firmware Generation
Supported MCU families:
- STM32 (all families) - FreeRTOS, Zephyr
- ESP32/ESP32-S3/C3 - ESP-IDF, FreeRTOS
- TI TMS320 - TI-RTOS
- Infineon AURIX - AUTOSAR
- Nordic nRF - Zephyr
- Raspberry Pi Pico - FreeRTOS
- NXP i.MX RT - FreeRTOS, Zephyr

---

## Development Commands

```bash
# Install dependencies
npm install

# Start development server
npm run dev

# Build for production
npm run build

# Run tests
npm test

# Type checking
npm run typecheck

# Lint code
npm run lint
```

---

## Environment Variables

```bash
# Required
NEXUS_API_KEY=your_key
OPENROUTER_API_KEY=your_key  # For LLM-guided fixes

# Service URLs (K8s internal)
NEXUS_GRAPHRAG_URL=http://nexus-graphrag.nexus.svc.cluster.local:9000
NEXUS_MAGEAGENT_URL=http://nexus-mageagent.nexus.svc.cluster.local:9010

# Simulation Tools
NGSPICE_PATH=/usr/bin/ngspice
OPENFOAM_PATH=/opt/openfoam
OPENEMS_PATH=/opt/openems
```

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/v1/projects` | Create new project |
| POST | `/api/v1/projects/:id/schematic/generate` | Generate schematic |
| POST | `/api/v1/projects/:id/pcb-layout/generate` | Start PCB layout |
| POST | `/api/v1/projects/:id/mapos/optimize` | Run MAPOS optimization |
| POST | `/api/v1/projects/:id/simulation/spice` | Run SPICE simulation |
| POST | `/api/v1/projects/:id/simulation/thermal` | Run thermal analysis |
| POST | `/api/v1/projects/:id/firmware/generate` | Generate firmware |
| POST | `/api/v1/projects/:id/manufacturing/gerbers` | Export Gerbers |
| POST | `/api/v1/projects/:id/validate/multi-llm` | Multi-LLM validation |

---

## Core Engineering Directives

### 1. Root Cause Analysis is MANDATORY
Before writing ANY code:
1. **Identify the symptom** - What is observed vs. expected?
2. **Trace the causal chain** - Follow execution path to origin
3. **Document the root cause** - Not symptoms, but fundamental flaw
4. **Validate the solution** - Will this fix cause, not symptom?
5. **Consider side effects** - What else might this affect?

### 2. Refactor, Don't Patch
```typescript
// ❌ WRONG: Patching symptoms
if (data === undefined) {
  data = {}; // Band-aid fix
}

// ✅ RIGHT: Refactoring to address root cause
function getData(): ValidatedData {
  const raw = fetchData();
  return validateData(raw); // Never returns undefined
}
```

### 3. Full Implementation Only
- **NO placeholders**: No `// TODO`, no `throw new Error('Not implemented')`
- **Complete error handling**: Every error path handled explicitly
- **All edge cases**: Handle null, undefined, empty states
- **Production-ready**: Code that can go to production immediately

### 4. Clean Code Standards
- **Single Responsibility**: Each function/class does ONE thing
- **DRY Principle**: Never duplicate logic
- **Dependency Injection**: Never hardcode dependencies
- **Pure Functions**: Prefer immutable, side-effect-free code

---

## Automated Quality Checks

### Before Committing Code
Always run:
```bash
npm run lint
npm run typecheck
npm run build
```

### Code Review Checklist
- [ ] No hardcoded values or magic numbers
- [ ] All error cases handled explicitly
- [ ] Code follows existing patterns
- [ ] No commented-out code or TODOs
- [ ] All functions have single responsibility
- [ ] Proper abstraction layers maintained
- [ ] Security best practices followed

---

## Reference Project

The **foc-esc-heavy-lift** project serves as the reference implementation:
- 10-layer PCB, 164 components
- 200A continuous, 400A peak
- Triple-redundant MCUs (AURIX, TMS320, STM32H755)
- 18× SiC MOSFETs
- 15,000+ lines firmware

Repository: `github.com/adverant/foc-esc-heavy-lift` (design data only)

---

## Contributing

1. Create feature branch: `git checkout -b feature/my-feature`
2. Make changes, add tests
3. Run full test suite: `npm test && npm run typecheck`
4. Commit with conventional commits: `feat: add new validator`
5. Push and create PR

---

**Remember**: This is production infrastructure. NEVER compromise on code quality. When in doubt, ask the user.
