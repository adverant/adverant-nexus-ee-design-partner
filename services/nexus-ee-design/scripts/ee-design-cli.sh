#!/bin/bash
# EE Design CLI - Terminal Access to EE Design API
# Full read/write/execute/admin access for Claude Code CLI and terminal tools

set -e

# Configuration
EE_DESIGN_URL="${EE_DESIGN_URL:-http://localhost:8080}"
API_PREFIX="${API_PREFIX:-/api/v1}"
CLI_USER="${CLI_USER:-claude-code-cli}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Helper functions
log_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# CLI headers for full access
CLI_HEADERS=(
  -H "X-User-Id: $CLI_USER"
  -H "X-CLI-Access: true"
  -H "X-Claude-Code: true"
  -H "Content-Type: application/json"
)

# API request helper
api_request() {
  local method="$1"
  local endpoint="$2"
  local data="$3"

  local url="${EE_DESIGN_URL}${API_PREFIX}${endpoint}"

  if [ -n "$data" ]; then
    curl -s -X "$method" "${CLI_HEADERS[@]}" -d "$data" "$url"
  else
    curl -s -X "$method" "${CLI_HEADERS[@]}" "$url"
  fi
}

# Commands
cmd_health() {
  log_info "Checking EE Design service health..."
  curl -s "${EE_DESIGN_URL}/health" | jq .
}

cmd_list_projects() {
  log_info "Listing all projects..."
  api_request GET "/projects" | jq .
}

cmd_get_project() {
  local project_id="$1"
  if [ -z "$project_id" ]; then
    log_error "Usage: ee-design-cli get-project <project_id>"
    exit 1
  fi
  api_request GET "/projects/$project_id" | jq .
}

cmd_create_project() {
  local name="$1"
  local description="${2:-EE Design Project}"

  if [ -z "$name" ]; then
    log_error "Usage: ee-design-cli create-project <name> [description]"
    exit 1
  fi

  log_info "Creating project: $name"
  api_request POST "/projects" "{\"name\": \"$name\", \"description\": \"$description\"}" | jq .
}

cmd_generate_schematic() {
  local project_id="$1"
  local requirements="$2"

  if [ -z "$project_id" ] || [ -z "$requirements" ]; then
    log_error "Usage: ee-design-cli generate-schematic <project_id> <requirements>"
    exit 1
  fi

  log_info "Generating schematic for project $project_id..."
  api_request POST "/projects/$project_id/schematic/generate" "{\"requirements\": \"$requirements\"}" | jq .
}

cmd_get_schematic() {
  local project_id="$1"
  local schematic_id="$2"

  if [ -z "$project_id" ] || [ -z "$schematic_id" ]; then
    log_error "Usage: ee-design-cli get-schematic <project_id> <schematic_id>"
    exit 1
  fi

  api_request GET "/projects/$project_id/schematic/$schematic_id" | jq .
}

cmd_download_schematic() {
  local project_id="$1"
  local schematic_id="$2"
  local output_file="${3:-schematic.kicad_sch}"

  if [ -z "$project_id" ] || [ -z "$schematic_id" ]; then
    log_error "Usage: ee-design-cli download-schematic <project_id> <schematic_id> [output_file]"
    exit 1
  fi

  log_info "Downloading schematic to $output_file..."
  curl -s "${CLI_HEADERS[@]}" "${EE_DESIGN_URL}${API_PREFIX}/projects/$project_id/schematic/$schematic_id/schematic.kicad_sch" > "$output_file"
  log_success "Downloaded to $output_file"
}

cmd_generate_pcb() {
  local project_id="$1"
  local schematic_id="$2"

  if [ -z "$project_id" ] || [ -z "$schematic_id" ]; then
    log_error "Usage: ee-design-cli generate-pcb <project_id> <schematic_id>"
    exit 1
  fi

  log_info "Generating PCB layout for schematic $schematic_id..."
  api_request POST "/projects/$project_id/pcb-layout/generate" "{\"schematicId\": \"$schematic_id\"}" | jq .
}

cmd_run_drc() {
  local project_id="$1"
  local layout_id="$2"

  if [ -z "$project_id" ] || [ -z "$layout_id" ]; then
    log_error "Usage: ee-design-cli run-drc <project_id> <layout_id>"
    exit 1
  fi

  log_info "Running DRC on layout $layout_id..."
  api_request POST "/projects/$project_id/pcb-layout/$layout_id/drc" | jq .
}

cmd_export_3d() {
  local project_id="$1"
  local layout_id="$2"
  local format="${3:-step}"
  local output_file="${4:-board.$format}"

  if [ -z "$project_id" ] || [ -z "$layout_id" ]; then
    log_error "Usage: ee-design-cli export-3d <project_id> <layout_id> [format:step|wrl] [output_file]"
    exit 1
  fi

  local ext="step"
  if [ "$format" = "wrl" ] || [ "$format" = "vrml" ]; then
    ext="wrl"
  fi

  log_info "Exporting 3D model ($ext) to $output_file..."
  curl -s "${CLI_HEADERS[@]}" "${EE_DESIGN_URL}${API_PREFIX}/projects/$project_id/pcb-layout/$layout_id/board.$ext" > "$output_file"
  log_success "Exported to $output_file"
}

cmd_get_bom() {
  local project_id="$1"

  if [ -z "$project_id" ]; then
    log_error "Usage: ee-design-cli get-bom <project_id>"
    exit 1
  fi

  api_request GET "/projects/$project_id/bom" | jq .
}

cmd_run_simulation() {
  local project_id="$1"
  local sim_type="$2"
  local params="$3"

  if [ -z "$project_id" ] || [ -z "$sim_type" ]; then
    log_error "Usage: ee-design-cli run-simulation <project_id> <type:spice|thermal|emc> [params_json]"
    exit 1
  fi

  log_info "Running $sim_type simulation..."
  if [ -n "$params" ]; then
    api_request POST "/projects/$project_id/simulation/$sim_type" "$params" | jq .
  else
    api_request POST "/projects/$project_id/simulation/$sim_type" "{}" | jq .
  fi
}

cmd_help() {
  cat << EOF
EE Design CLI - Terminal Access to EE Design API

Usage: ee-design-cli <command> [arguments]

Environment Variables:
  EE_DESIGN_URL    API base URL (default: http://localhost:8080)
  CLI_USER         User ID for API requests (default: claude-code-cli)

Commands:
  health                                    Check service health
  list-projects                             List all projects
  get-project <id>                          Get project details
  create-project <name> [desc]              Create new project

  generate-schematic <proj_id> <requirements>  Generate schematic
  get-schematic <proj_id> <schem_id>          Get schematic details
  download-schematic <proj_id> <schem_id> [file]  Download .kicad_sch

  generate-pcb <proj_id> <schem_id>         Generate PCB layout
  run-drc <proj_id> <layout_id>             Run design rule check
  export-3d <proj_id> <layout_id> [fmt] [file]  Export STEP/VRML

  get-bom <proj_id>                         Get bill of materials
  run-simulation <proj_id> <type> [params]  Run simulation

  help                                      Show this help message

Examples:
  ee-design-cli health
  ee-design-cli create-project "Power Supply" "5V 3A buck converter"
  ee-design-cli generate-schematic abc123 "Design a USB-C power delivery board"
  ee-design-cli export-3d abc123 layout456 step board.step

EOF
}

# Main command dispatcher
main() {
  local command="${1:-help}"
  shift || true

  case "$command" in
    health)             cmd_health ;;
    list-projects)      cmd_list_projects ;;
    get-project)        cmd_get_project "$@" ;;
    create-project)     cmd_create_project "$@" ;;
    generate-schematic) cmd_generate_schematic "$@" ;;
    get-schematic)      cmd_get_schematic "$@" ;;
    download-schematic) cmd_download_schematic "$@" ;;
    generate-pcb)       cmd_generate_pcb "$@" ;;
    run-drc)            cmd_run_drc "$@" ;;
    export-3d)          cmd_export_3d "$@" ;;
    get-bom)            cmd_get_bom "$@" ;;
    run-simulation)     cmd_run_simulation "$@" ;;
    help|--help|-h)     cmd_help ;;
    *)
      log_error "Unknown command: $command"
      cmd_help
      exit 1
      ;;
  esac
}

main "$@"
