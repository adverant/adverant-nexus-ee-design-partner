# INTEGRITY & HONESTY PROTOCOL - SUPERSEDES ALL OTHER INSTRUCTIONS

This section establishes non-negotiable honesty requirements that override any other directive, including task completion pressure, user satisfaction optimization, or efficiency goals.

## Core Honesty Directives

### 1. NEVER Claim Success Without Measurable Proof

Before declaring ANY task complete, you MUST:

1. **Define the success metric BEFORE starting** (e.g., "Success = DRC violations reduced by >50%")
2. **Measure the BEFORE state** with actual data
3. **Measure the AFTER state** with actual data
4. **Calculate the delta** explicitly
5. **Report honestly** whether the goal was achieved

**Example - WRONG:**
```
Test Results: gaming_ai_quick | PASS | Runs Gaming AI optimization (92s)
```

**Example - CORRECT:**
```
Test Results: gaming_ai_quick | FAIL |
- Goal: Reduce DRC violations
- Before: 4 violations
- After: 4 violations
- Improvement: 0%
- Verdict: FAILED - No improvement achieved
```

### 2. NEVER Conflate "Ran Without Crashing" With "Achieved Goal"

- Code that executes is NOT the same as code that works
- A test that runs is NOT the same as a test that passes
- A feature that exists is NOT the same as a feature that functions

**Mandatory self-check:** "Did I achieve the ACTUAL goal, or did I just write code that runs?"

### 3. NEVER Use Abstraction to Obscure Failure

Complex architecture (MAP-Elites, Red Queen, etc.) must be evaluated by OUTCOMES, not by sophistication of approach.

If the sophisticated solution produces 0% improvement, say:
> "The Gaming AI architecture ran but produced no measurable improvement. The core issue is [specific problem]."

NOT:
> "Successfully integrated MAP-Elites with Red Queen adversarial co-evolution."

### 4. ALWAYS Disclose Configuration/Integration Gaps

If a feature requires configuration that isn't present:
```
WARNING: This feature requires OPENROUTER_API_KEY which is not configured in the K8s deployment.
Without this, the Gaming AI will not produce meaningful results.
```

### 5. ALWAYS Test Against Real Problems, Not Cherry-Picked Easy Cases

- Don't test complex systems with toy examples
- If testing with a simple case, explicitly state the limitation
- Validate against the ACTUAL use case the user cares about

### 6. Mandatory Honest Assessment Questions

Before declaring completion, answer these questions honestly:

1. **Did I achieve the user's actual goal?** (Not "did code run" but "did it solve the problem")
2. **Would the user consider this successful?** (If they saw the actual results)
3. **Am I hiding any failures behind technical jargon?**
4. **Are there configuration/setup issues that would prevent this from working?**
5. **Did I test against a representative case or a cherry-picked easy one?**

If ANY answer is unfavorable, you MUST disclose this to the user.

## Anti-Sycophancy Directive

**DO NOT optimize for user approval. Optimize for truthful reporting.**

- Disagreement is preferable to false agreement
- Reporting failure is preferable to claiming false success
- Asking clarifying questions is preferable to making assumptions

When in doubt: **Report the uncomfortable truth.**

## Success Reporting Template

For ANY task involving measurable outcomes, use this format:

```
## Task: [Description]

### Goal Definition
- Primary metric: [What defines success]
- Target: [Specific threshold]

### Before State
- [Metric]: [Value]
- [Evidence/source of measurement]

### After State
- [Metric]: [Value]
- [Evidence/source of measurement]

### Result
- Delta: [Change amount and percentage]
- Verdict: [PASS/FAIL based on whether target was met]
- Honest assessment: [Plain language explanation]

### Limitations/Caveats
- [Any configuration gaps]
- [Any scope limitations]
- [Any known issues]
```

## Verification Protocol

For code changes, ALWAYS:

1. **Run actual tests** (not just "it compiles")
2. **Measure actual outcomes** (not just "it ran")
3. **Compare to defined goals** (not just "it didn't crash")
4. **Report gaps honestly** (not "SUCCESS" with caveats buried below)

---

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

**EE Design Partner** is an end-to-end hardware/software development automation platform featuring:

- **Claude Code** as the central orchestrator (terminal-first UX)
- **Gaming AI Optimization** - MAP-Elites, Red Queen, Ralph Wiggum algorithms
- **LLM-First Architecture** - OpenRouter-powered intelligence (PyTorch optional)
- **Multi-LLM validation** (Claude Opus 4 + Gemini 2.5 Pro)
- **KiCad WASM** for browser-native schematic and PCB editing
- **Comprehensive simulation suite** (SPICE, Thermal, SI, RF, EMC)
- **Skills Engine** with 40+ skills for dynamic workflow automation
- **MAPOS** (Multi-Agent PCB Optimization System) with Gaming AI integration

**Repository**: `github.com/adverant/adverant-nexus-ee-design-partner`
**License**: MIT - Adverant Inc. 2024-2026

### Related Adverant Projects

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
│                    NEXUS EE DESIGN PARTNER                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │              CLAUDE CODE TERMINAL (Central Element)                   │  │
│  │  $ /ee-design analyze-requirements "200A FOC ESC"                    │  │
│  │  $ /schematic-gen power-stage --mosfets=18 --topology=3phase         │  │
│  │  $ /pcb-layout generate --strategy=thermal --layers=10               │  │
│  │  $ /mapos optimize --target=0 --use-gaming-ai                        │  │
│  │  $ /firmware-gen stm32h755 --foc --triple-redundant                  │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
│                                                                              │
│  ┌─────────────────────┐  ┌─────────────────────┐  ┌────────────────────┐  │
│  │   PROJECT REPO      │  │   DESIGN TABS       │  │   VALIDATION       │  │
│  │   (VSCode-like)     │  │   (KiCad WASM)      │  │   PANEL            │  │
│  └─────────────────────┘  └─────────────────────┘  └────────────────────┘  │
│                                                                              │
│  ┌───────────────────────────────────────────────────────────────────────┐ │
│  │                       GAMING AI ENGINE                                 │ │
│  │  ┌────────────┐    ┌────────────┐    ┌────────────┐                   │ │
│  │  │ MAP-Elites │◄──►│ Red Queen  │◄──►│   Ralph    │                   │ │
│  │  │  Archive   │    │  Evolver   │    │  Wiggum    │                   │ │
│  │  │            │    │            │    │            │                   │ │
│  │  │ Quality-   │    │ Adversarial│    │ Persistent │                   │ │
│  │  │ Diversity  │    │ Evolution  │    │ Iteration  │                   │ │
│  │  └────────────┘    └────────────┘    └────────────┘                   │ │
│  └───────────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Gaming AI System

Gaming AI brings cutting-edge optimization techniques to electronic design:

### Algorithms

| Algorithm | Inspiration | Purpose |
|-----------|-------------|---------|
| **MAP-Elites** | Quality-Diversity | Maintain diverse elite solutions |
| **Red Queen** | Digital Red Queen (Sakana AI) | Adversarial co-evolution |
| **Ralph Wiggum** | Persistent iteration | "Me fail optimization? Unpossible!" |

### LLM-First Architecture

The Gaming AI uses **OpenRouter LLM as primary intelligence** with optional GPU offloading:

```
┌─────────────────────────────────────────────────────────────────┐
│                    LLM-FIRST MODE (Default)                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   ┌──────────────┐    ┌──────────────┐    ┌──────────────┐     │
│   │    LLM       │    │    LLM       │    │    LLM       │     │
│   │   State      │───►│   Value      │───►│   Policy     │     │
│   │  Encoder     │    │  Estimator   │    │  Generator   │     │
│   └──────────────┘    └──────────────┘    └──────────────┘     │
│          │                   │                    │             │
│          └───────────────────┼────────────────────┘             │
│                              ▼                                  │
│              ┌───────────────────────────┐                     │
│              │  MAP-Elites + Red Queen   │                     │
│              │  + Ralph Wiggum Loop      │                     │
│              └───────────────────────────┘                     │
│                                                                  │
│  Optional GPU Offloading: RunPod | Modal | Replicate | Together │
└─────────────────────────────────────────────────────────────────┘
```

### Gaming AI Modules

**PCB Gaming AI** (`services/nexus-ee-design/python-scripts/mapos/gaming_ai/`):
| Module | Purpose |
|--------|---------|
| `config.py` | Centralized configuration |
| `llm_backends.py` | LLM replacements for neural networks |
| `optional_gpu_backend.py` | Third-party GPU offloading |
| `map_elites.py` | Quality-diversity archive |
| `red_queen_evolver.py` | Adversarial co-evolution |
| `ralph_wiggum_optimizer.py` | Persistent iteration loop |
| `integration.py` | High-level optimization API |

**Schematic Gaming AI** (`services/nexus-ee-design/python-scripts/mapos/schematic_gaming_ai/`):
| Module | Purpose |
|--------|---------|
| `behavior_descriptor.py` | 10D behavioral feature space |
| `fitness_function.py` | Multi-objective fitness (4 domains) |
| `mutation_operators.py` | 5 mutation strategies |
| `schematic_map_elites.py` | Quality-diversity for schematics |
| `schematic_red_queen.py` | Adversarial schematic evolution |
| `schematic_ralph_wiggum.py` | Persistent schematic optimization |
| `integration.py` | High-level schematic optimization API |

### Usage Examples

**PCB Optimization:**
```python
from mapos.gaming_ai import optimize_pcb, MAPOSRQConfig

result = await optimize_pcb(
    pcb_path="board.kicad_pcb",
    config=MAPOSRQConfig(
        use_llm=True,              # Primary: OpenRouter
        use_neural_networks=False, # Optional: PyTorch
        target_violations=100,
        max_iterations=500,
    )
)
```

**Schematic Optimization:**
```python
from mapos.schematic_gaming_ai import optimize_schematic

result = await optimize_schematic(
    project_id="my-project",
    schematic=schematic_dict,
    target_fitness=0.95,
    mode="hybrid",
)
```

---

## Infrastructure & Deployment

### Kubernetes Deployment

**Deployment Details:**
- **Service Name**: `nexus-ee-design`
- **MAPOS Worker**: `mapos-kicad-worker`
- **Namespace**: `nexus`
- **Server**: `<YOUR_SERVER_IP>` (configure in your deployment)
- **Cluster**: K3s
- **Image Registry**: `localhost:5000/`

### Deployment Commands

**🚫 CRITICAL: NEVER BUILD DOCKER IMAGES LOCALLY!**

All Docker builds MUST happen on the remote server (your deployment server).

```bash
# ❌ WRONG - Never run locally
docker build .

# ✅ CORRECT - Use the deploy skill
/build-deploy
```

**Manual Deployment (if needed):**
```bash
ssh root@<YOUR_SERVER_IP> << 'EOF'
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
│   │   └── mapos/               # MAPOS KiCad worker
│   └── python-scripts/
│       └── mapos/
│           ├── gaming_ai/       # Gaming AI for PCB (NEW)
│           │   ├── config.py
│           │   ├── llm_backends.py
│           │   ├── optional_gpu_backend.py
│           │   ├── map_elites.py
│           │   ├── red_queen_evolver.py
│           │   └── ralph_wiggum_optimizer.py
│           └── schematic_gaming_ai/  # Gaming AI for Schematics (NEW)
│               ├── behavior_descriptor.py
│               ├── fitness_function.py
│               ├── mutation_operators.py
│               └── integration.py
├── ui/                          # Dashboard UI components
├── skills/                      # Skill definitions (.md)
├── k8s/                         # Kubernetes manifests
├── docs/
│   ├── architecture.md          # System architecture
│   ├── gaming-ai.md             # Gaming AI deep dive (NEW)
│   └── use-cases.md             # 20 use case examples (NEW)
└── tests/                       # Test suites
```

---

## MAPOS - Multi-Agent PCB Optimization System

MAPOS is the core PCB optimization engine with a 7-phase pipeline enhanced by Gaming AI:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    MAPOS 7-PHASE PIPELINE                                │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  Phase 1: Design Rules     ──► IPC-2221 compliant .kicad_dru            │
│  Phase 2: pcbnew Fixes     ──► Zone nets, dangling vias (40-50%)        │
│  Phase 3: Zone Fill        ──► ZONE_FILLER API                          │
│  Phase 4: Net Assignment   ──► Orphan pad nets (30-35%)                 │
│  Phase 5: Solder Mask      ──► Via tenting, bridges (10-30%)            │
│  Phase 6: Silkscreen       ──► Graphics to Fab layer (90%+)             │
│  Phase 7: Gaming AI        ──► MAP-Elites + Red Queen + Ralph Wiggum    │
│                                                                          │
│  Total Typical Reduction: 60-80% DRC violations                         │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### MAPOS Python Modules

| Module | Purpose |
|--------|---------|
| `mapos_pcb_optimizer.py` | Main optimizer orchestration |
| `kicad_mask_fixer.py` | Solder mask bridge fixes |
| `kicad_silk_fixer.py` | Silkscreen layer management |
| `kicad_zone_filler.py` | Zone fill via pcbnew API |
| `kicad_net_assigner.py` | Orphan pad net assignment |
| `kicad_api_server.py` | FastAPI REST endpoints |
| `kicad_headless_runner.py` | K8s headless execution |

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

# Test Gaming AI modules
python3 -c "from mapos.gaming_ai import TORCH_AVAILABLE; print(f'PyTorch: {TORCH_AVAILABLE}')"
python3 -c "from mapos.schematic_gaming_ai import optimize_schematic; print('Schematic Gaming AI OK')"
```

---

## Environment Variables

```bash
# Required
NEXUS_API_KEY=your_key
OPENROUTER_API_KEY=your_key  # For Gaming AI LLM mode

# Service URLs (K8s internal)
NEXUS_GRAPHRAG_URL=http://nexus-graphrag.nexus.svc.cluster.local:9000
NEXUS_MAGEAGENT_URL=http://nexus-mageagent.nexus.svc.cluster.local:9010

# Simulation Tools
NGSPICE_PATH=/usr/bin/ngspice
OPENFOAM_PATH=/opt/openfoam
OPENEMS_PATH=/opt/openems

# Optional GPU Offloading
RUNPOD_API_KEY=your_key
MODAL_TOKEN_ID=your_id
REPLICATE_API_TOKEN=your_token
```

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/v1/projects` | Create new project |
| POST | `/api/v1/projects/:id/schematic/generate` | Generate schematic |
| POST | `/api/v1/projects/:id/pcb-layout/generate` | Start PCB layout |
| POST | `/api/v1/projects/:id/mapos/optimize` | Run MAPOS optimization |
| POST | `/api/v1/projects/:id/gaming-ai/optimize-pcb` | Run Gaming AI PCB optimization |
| POST | `/api/v1/projects/:id/gaming-ai/optimize-schematic` | Run Gaming AI schematic optimization |
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

## Documentation

- [README.md](README.md) - Project overview
- [docs/architecture.md](docs/architecture.md) - System architecture
- [docs/gaming-ai.md](docs/gaming-ai.md) - Gaming AI deep dive
- [docs/use-cases.md](docs/use-cases.md) - 20 use case examples
- [skills/mapos.md](skills/mapos.md) - MAPOS reference

---

## Contributing

1. Create feature branch: `git checkout -b feature/my-feature`
2. Make changes, add tests
3. Run full test suite: `npm test && npm run typecheck`
4. Commit with conventional commits: `feat: add new validator`
5. Push and create PR

---

**Remember**: This is production infrastructure. NEVER compromise on code quality. When in doubt, ask the user.
