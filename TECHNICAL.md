# EE Design Partner - Technical Reference

Comprehensive technical documentation for the EE Design Partner plugin.

---

## API Reference

### Base Configuration

| Property | Value |
|----------|-------|
| API Version | v1 |
| Base Path | `/api/v1` |
| Documentation | `/api/docs` |
| OpenAPI Spec | `/api/openapi.json` |

### Project Endpoints

```http
POST /api/v1/projects
```
Create a new project.

```http
POST /api/v1/projects/:id/schematic/generate
```
Generate schematic from requirements and architecture.

```http
POST /api/v1/projects/:id/pcb-layout/generate
```
Generate PCB layout with competing agent tournament.

```http
POST /api/v1/projects/:id/mapos/optimize
```
Run MAPOS optimization pipeline.

### Simulation Endpoints

```http
POST /api/v1/projects/:id/simulation/spice
```
Run SPICE circuit simulation (DC, AC, Transient, Monte Carlo, Noise).

```http
POST /api/v1/projects/:id/simulation/thermal
```
Run thermal analysis (FEA, CFD, Steady-state, Transient).

```http
POST /api/v1/projects/:id/simulation/signal-integrity
```
Run signal integrity analysis (Impedance, Crosstalk, Eye diagram).

```http
POST /api/v1/projects/:id/simulation/rf-emc
```
Run RF/EMC simulation (Field patterns, S-parameters, Emissions).

### Gaming AI Endpoints

```http
POST /api/v1/projects/:id/gaming-ai/optimize-pcb
```
Run PCB optimization with Gaming AI (MAP-Elites, Red Queen, Ralph Wiggum).

```http
POST /api/v1/projects/:id/gaming-ai/optimize-schematic
```
Run schematic optimization with Gaming AI.

```http
GET /api/v1/projects/:id/gaming-ai/archive
```
Retrieve MAP-Elites quality-diversity archive.

---

## MCP Tools

The plugin provides 12 MCP tools for Claude Code integration:

| Tool | Description |
|------|-------------|
| `generate_schematic` | Generate schematic from requirements and architecture |
| `parse_kicad_schematic` | Parse existing KiCad schematic files |
| `generate_pcb_layout` | Generate PCB layout with competing agent tournament |
| `run_spice_simulation` | Run SPICE circuit simulation |
| `run_thermal_simulation` | Run thermal analysis simulation |
| `run_signal_integrity` | Run signal integrity analysis |
| `run_emc_simulation` | Run RF/EMC simulation |
| `export_gerbers` | Export Gerber and drill files |
| `generate_firmware` | Generate firmware scaffolding for target MCU |
| `generate_hal` | Generate HAL layer code |
| `generate_driver` | Generate component driver from datasheet |
| `validate_design` | Run multi-LLM validation on design artifacts |
| `get_vendor_quote` | Get PCB manufacturing quote from vendors |

### MCP Prompts

| Prompt | Description |
|--------|-------------|
| `design_review` | Comprehensive design review prompt |
| `optimization_suggestions` | Get AI suggestions for design optimization |

### MCP Resources

| Resource | Description |
|----------|-------------|
| `component_library` | Component symbol and footprint library |
| `design_rules` | DRC rules and constraints |

---

## Simulation Types

### SPICE Simulation

| Engine | Capabilities |
|--------|--------------|
| ngspice | DC, AC, Transient, Monte Carlo, Noise |
| LTspice | DC, AC, Transient, Monte Carlo, Noise |

### Thermal Simulation

| Engine | Capabilities |
|--------|--------------|
| OpenFOAM | CFD analysis, Transient thermal |
| Elmer | FEA analysis, Steady-state thermal |

### Signal Integrity

| Engine | Capabilities |
|--------|--------------|
| scikit-rf | Impedance analysis, Crosstalk, Eye diagram |

### RF/EMC Simulation

| Engine | Capabilities |
|--------|--------------|
| openEMS | Field patterns, S-parameters |
| MEEP | Electromagnetic simulation, Emissions |

---

## Environment Variables

### Required Variables

| Variable | Description |
|----------|-------------|
| `NEXUS_API_KEY` | Nexus platform access key |
| `ANTHROPIC_API_KEY` | Claude API key for primary LLM |
| `OPENROUTER_API_KEY` | OpenRouter key for Gaming AI LLM mode and Gemini validation |

### Optional Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `NEXUS_GRAPHRAG_URL` | `http://localhost:9000` | GraphRAG service URL |
| `NEXUS_MAGEAGENT_URL` | `http://localhost:9010` | MageAgent service URL |
| `NGSPICE_PATH` | System default | Path to ngspice executable |
| `OPENFOAM_PATH` | System default | Path to OpenFOAM installation |
| `OPENEMS_PATH` | System default | Path to openEMS installation |
| `MAX_CONCURRENT_SIMS` | 4 | Maximum concurrent simulations |
| `LAYOUT_MAX_ITERATIONS` | 500 | Maximum layout optimization iterations |
| `LAYOUT_TARGET_SCORE` | 0.95 | Target fitness score for layout |

---

## Resource Requirements

### Kubernetes Resource Allocation

| Resource | Request | Limit |
|----------|---------|-------|
| CPU | 500m | 4000m |
| Memory | 1Gi | 8Gi |
| Ephemeral Storage | 500Mi | 20Gi |

### Persistent Volumes

| Volume | Mount Path | Size | Persistent |
|--------|------------|------|------------|
| projects | `/data/projects` | 50Gi | Yes |
| output | `/data/output` | 20Gi | Yes |
| tmp | `/tmp` | 5Gi | No |

---

## Service Dependencies

### Required Nexus Services

| Service | Version | Purpose |
|---------|---------|---------|
| nexus-graphrag | >=1.0.0 | Memory and knowledge graph |

### Optional Nexus Services

| Service | Version | Purpose |
|---------|---------|---------|
| nexus-mageagent | >=1.0.0 | Multi-agent orchestration |
| nexus-sandbox | >=1.0.0 | Code execution |

### External Dependencies

| Service | Purpose | Required |
|---------|---------|----------|
| OpenRouter | Gemini 2.5 Pro for validation | Yes |
| Anthropic | Claude API for primary LLM | Yes |

---

## WebSocket Events

The plugin emits the following WebSocket events for real-time updates:

### Simulation Events

| Event | Description |
|-------|-------------|
| `simulation:started` | Simulation job started |
| `simulation:progress` | Progress update with percentage |
| `simulation:completed` | Simulation completed successfully |
| `simulation:failed` | Simulation failed with error |

### Layout Events

| Event | Description |
|-------|-------------|
| `layout:started` | PCB layout generation started |
| `layout:iteration` | Layout optimization iteration update |
| `layout:completed` | Layout generation completed |
| `layout:failed` | Layout generation failed |

### Validation Events

| Event | Description |
|-------|-------------|
| `validation:started` | Design validation started |
| `validation:result` | Individual validation result |
| `validation:completed` | All validations completed |

### Other Events

| Event | Description |
|-------|-------------|
| `firmware:generated` | Firmware scaffolding generated |
| `gerber:exported` | Gerber files exported |

---

## Health Endpoints

| Endpoint | Purpose |
|----------|---------|
| `/health` | General health check |
| `/ready` | Readiness probe for Kubernetes |
| `/live` | Liveness probe for Kubernetes |

---

## Metrics

| Property | Value |
|----------|-------|
| Enabled | Yes |
| Path | `/metrics` |
| Port | 8080 |
| Format | Prometheus |

---

## Logging

| Property | Value |
|----------|-------|
| Level | info |
| Format | JSON |
| Outputs | stdout |

---

## Security Configuration

| Property | Value |
|----------|-------|
| Run as non-root | Yes |
| Read-only root filesystem | No |
| Allow privilege escalation | No |
| Capabilities | Drop ALL |

---

## Skills Reference

### Master Skill

- **Skill Name**: `ee-design-partner`
- **Triggers**: `/ee-design`, `/eda`, `/hardware`

### Complete Sub-Skills List (40+ Skills)

#### Research and Requirements Phase

- `research-paper`
- `patent-search`
- `requirements-gen`

#### Architecture Phase

- `ee-architecture`
- `component-select`
- `bom-optimize`
- `power-budget`

#### Schematic Phase

- `schematic-gen`
- `schematic-review`
- `netlist-gen`

#### Simulation Phase

- `simulate-spice`
- `simulate-thermal`
- `simulate-si`
- `simulate-rf`
- `simulate-emc`
- `simulate-reliability`

#### PCB Phase

- `pcb-layout`
- `pcb-review`
- `stackup-design`
- `via-optimize`

#### Manufacturing Phase

- `gerber-gen`
- `dfm-check`
- `vendor-quote`
- `panelize`

#### Firmware Phase

- `firmware-gen`
- `hal-gen`
- `driver-gen`
- `rtos-config`
- `build-setup`

#### Testing Phase

- `test-gen`
- `hil-setup`
- `test-procedure`
- `coverage-analysis`

#### Production Phase

- `manufacture`
- `assembly-guide`
- `quality-check`
- `traceability`

#### Field Support Phase

- `debug-assist`
- `service-manual`
- `rma-process`
- `firmware-update`

---

## Additional Resources

- [QUICKSTART.md](QUICKSTART.md) - Installation and basic setup
- [USE-CASES.md](USE-CASES.md) - Real-world application examples
- [ARCHITECTURE.md](ARCHITECTURE.md) - System design overview
