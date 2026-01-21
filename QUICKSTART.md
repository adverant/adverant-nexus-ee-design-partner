# EE Design Partner - Quick Start Guide

Get up and running with the EE Design Partner plugin for Nexus in minutes.

---

## Prerequisites

### Required Software

| Software | Version | Purpose |
|----------|---------|---------|
| Node.js | 20+ | Runtime engine |
| Python | 3.11+ | KiCad automation |
| Docker | Latest | Simulation containers |

### Required API Keys

| Key | Provider | Purpose |
|-----|----------|---------|
| `NEXUS_API_KEY` | Adverant | Nexus platform access |
| `ANTHROPIC_API_KEY` | Anthropic | Claude API for primary LLM |
| `OPENROUTER_API_KEY` | OpenRouter | Gemini 2.5 Pro for validation, Gaming AI LLM mode |

---

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/adverant/adverant-nexus-ee-design-partner.git
cd adverant-nexus-ee-design-partner
```

### 2. Install Dependencies

```bash
npm install
```

### 3. Configure Environment

```bash
# Copy example environment file
cp .env.example .env
```

Edit `.env` with your API keys:

```env
# Required API Keys
NEXUS_API_KEY=your_nexus_key
ANTHROPIC_API_KEY=your_anthropic_key
OPENROUTER_API_KEY=your_openrouter_key

# Service URLs (use defaults for local dev)
NEXUS_GRAPHRAG_URL=http://localhost:9000
NEXUS_MAGEAGENT_URL=http://localhost:9010
```

### 4. Start Development Server

```bash
npm run dev
```

The plugin will be available at `http://localhost:8080`.

---

## First Project Setup

### Create a New Project

```bash
/ee-design create-project --name "my-first-project" --type "power-electronics"
```

### Analyze Requirements

```bash
/ee-design analyze-requirements "Motor controller for brushless DC motor, 48V 20A"
```

### Generate Architecture

```bash
/ee-architecture generate --project my-first-project
```

---

## Basic Commands

### Design Phase Commands

| Command | Description |
|---------|-------------|
| `/ee-design` | Master skill trigger for EE Design Partner |
| `/ee-architecture` | Generate system architecture |
| `/component-select` | AI-assisted component selection |
| `/power-budget` | Power budget analysis |

### Schematic Commands

| Command | Description |
|---------|-------------|
| `/schematic-gen` | Generate schematic from requirements |
| `/schematic-review` | AI review of existing schematic |
| `/netlist-gen` | Generate netlist from schematic |

### PCB Commands

| Command | Description |
|---------|-------------|
| `/pcb-layout` | Generate PCB layout with competing agents |
| `/pcb-review` | Design rule and best practice review |
| `/stackup-design` | PCB stackup configuration |
| `/gerber-gen` | Export Gerber and drill files |

### Simulation Commands

| Command | Description |
|---------|-------------|
| `/simulate-spice` | SPICE circuit simulation |
| `/simulate-thermal` | Thermal analysis |
| `/simulate-si` | Signal integrity analysis |
| `/simulate-rf` | RF simulation |
| `/simulate-emc` | EMC analysis |

### MAPOS Commands

| Command | Description |
|---------|-------------|
| `/mapos optimize` | Full Gaming AI optimization |
| `/mapos pre-drc` | Quick pre-DRC fixes |
| `/mapos fill-zones` | Zone fill only |

### Firmware Commands

| Command | Description |
|---------|-------------|
| `/firmware-gen` | Generate firmware scaffolding |
| `/hal-gen` | Generate HAL layer code |
| `/driver-gen` | Generate component drivers |
| `/rtos-config` | RTOS configuration |

### Manufacturing Commands

| Command | Description |
|---------|-------------|
| `/dfm-check` | Design for manufacturing check |
| `/vendor-quote` | Get PCB manufacturing quotes |
| `/panelize` | Panel layout for production |

---

## Verification Steps

### 1. Check Health Endpoints

```bash
# Health check
curl http://localhost:8080/health

# Readiness check
curl http://localhost:8080/ready

# Liveness check
curl http://localhost:8080/live
```

### 2. Verify API Access

```bash
curl http://localhost:8080/api/v1/projects
```

### 3. Check WebSocket Connection

The plugin supports WebSocket events for real-time updates:

- `simulation:started`
- `simulation:progress`
- `simulation:completed`
- `layout:started`
- `layout:iteration`
- `layout:completed`
- `validation:result`

### 4. Verify UI Components

Navigate to `http://localhost:8080/ui` to access:

- **Terminal**: Claude Code terminal integration
- **Project Browser**: VSCode-like GitHub repo browser
- **KiCad Viewer**: KiCad WASM schematic and PCB editor
- **Simulation Panel**: Simulation waveforms and results
- **Validation Panel**: DRC/ERC/Multi-LLM validation results
- **3D Viewer**: Three.js PCB 3D visualization

---

## Next Steps

- Review [USE-CASES.md](USE-CASES.md) for real-world application examples
- See [TECHNICAL.md](TECHNICAL.md) for API reference and configuration details
- Explore [ARCHITECTURE.md](ARCHITECTURE.md) for system design overview

---

## Support

- Documentation: `/docs`
- Issues: [GitHub Issues](https://github.com/adverant/adverant-nexus-ee-design-partner/issues)
- Email: support@adverant.ai
