# Visual Validation Infrastructure Setup Guide

## Executive Summary

**CRITICAL INFRASTRUCTURE DEPENDENCIES DETECTED**

The MAPO v3.0 visual validation system depends on external services that are **NOT currently running** on this development machine. This document identifies all infrastructure requirements, verifies what's missing, and provides setup instructions.

---

## Infrastructure Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    MAPO Validation Loop                         │
│  (dual_llm_validator.py + ValidationLoop)                       │
└─────────────────────────────────────────────────────────────────┘
                            ↓
        ┌───────────────────────────────────────┐
        │     SchematicImageExtractor           │
        │   (image_extractor.py)                │
        └───────────────────────────────────────┘
                            ↓
        ╔═══════════════════════════════════════╗
        ║   kicad-worker HTTP Service           ║
        ║   Port: 8080                          ║
        ║   Endpoint: /v1/schematic/export      ║
        ╚═══════════════════════════════════════╝
                            ↓
        ┌───────────────────────────────────────┐
        │   kicad-cli (in Docker container)     │
        │   + Xvfb (headless X11 display)       │
        │   + ImageMagick (PNG conversion)      │
        └───────────────────────────────────────┘
                            ↓
        ┌───────────────────────────────────────┐
        │   PNG image bytes (300 DPI)           │
        └───────────────────────────────────────┘
                            ↓
        ╔═══════════════════════════════════════╗
        ║   Opus 4.6 Vision API                 ║
        ║   via OpenRouter                      ║
        ║   API Key: OPENROUTER_API_KEY         ║
        ╚═══════════════════════════════════════╝
```

---

## Infrastructure Status Assessment

### 1. kicad-worker Service

**Status:** ❌ NOT RUNNING

**Expected Location:**
- **Kubernetes:** `http://mapos-kicad-worker:8080` (ClusterIP service in `nexus` namespace)
- **Local Development:** `http://localhost:8080` (Docker container)

**Evidence:**
- `kubectl` command not found on development machine
- No local kicad-worker Docker container running
- Environment variable `KICAD_WORKER_URL` is NOT set

**Configuration Files Found:**
- `/Users/don/Adverant/adverant-nexus-ee-design-partner/k8s/mapos-kicad-worker.yaml` - K8s deployment manifest
- `/Users/don/Adverant/adverant-nexus-ee-design-partner/services/nexus-ee-design/docker/mapos/Dockerfile` - Docker build file

**What the Service Does:**
The kicad-worker is a FastAPI HTTP server (`kicad_api_server.py`) that:
1. Receives `.kicad_sch` file content via POST `/v1/schematic/export`
2. Saves content to temporary `.kicad_sch` file
3. Runs `kicad-cli sch export {format}` to generate PNG/SVG/PDF
4. Returns download URL for the exported image
5. Serves the image via GET `/v1/schematic/download/{export_id}/{filename}`

**HTTP API Contract:**

**POST /v1/schematic/export**
```json
{
  "schematic_content": "<full .kicad_sch file content>",
  "export_format": "png",
  "design_name": "validation_iter0",
  "dpi": 300
}
```

**Response:**
```json
{
  "success": true,
  "export_id": "a1b2c3d4",
  "format": "png",
  "filename": "validation_iter0.png",
  "size_bytes": 524288,
  "download_url": "/v1/schematic/download/a1b2c3d4/validation_iter0.png",
  "errors": [],
  "duration_seconds": 2.3
}
```

**GET /v1/schematic/download/{export_id}/{filename}**
- Returns raw PNG bytes with `Content-Type: image/png`

**GET /health**
```json
{
  "status": "healthy",
  "timestamp": "2026-02-07T12:34:56.789Z"
}
```

---

### 2. kicad-cli Executable

**Status:** ❌ NOT INSTALLED LOCALLY

**Expected Location:** `/usr/bin/kicad-cli` (inside kicad-worker container)

**Evidence:**
- `which kicad-cli` returns: `kicad-cli not found`
- This is expected - kicad-cli runs INSIDE the kicad-worker Docker container, not on host

**What It Does:**
```bash
# Export to SVG
kicad-cli sch export svg -o /output/dir schematic.kicad_sch

# Export to PDF
kicad-cli sch export pdf -o output.pdf schematic.kicad_sch

# PNG is generated via SVG -> ImageMagick conversion
# (KiCad doesn't support direct PNG export for schematics)
convert -density 300 schematic.svg schematic.png
```

**Dependencies:**
- KiCad 8.x (from PPA: `ppa:kicad/kicad-8.0-releases`)
- Xvfb (virtual X11 display for headless operation)
- ImageMagick (for SVG to PNG conversion)

---

### 3. OpenRouter API (Opus 4.6 Vision)

**Status:** ⚠️ UNKNOWN (API key not configured)

**Expected Configuration:**
- Environment variable: `OPENROUTER_API_KEY`
- K8s secret: `nexus-secrets` with key `OPENROUTER_API_KEY`

**Evidence:**
- `printenv | grep OPENROUTER` returns empty
- Kubernetes secrets template found at `/Users/don/Adverant/adverant-nexus-ee-design-partner/k8s/secrets.yaml`
- Code expects API key via environment variable or passed directly to validator

**API Endpoint:**
```
POST https://openrouter.ai/api/v1/chat/completions
Authorization: Bearer sk-or-v1-...
Content-Type: application/json

{
  "model": "anthropic/claude-opus-4.6",
  "messages": [
    {
      "role": "user",
      "content": [
        {
          "type": "image",
          "source": {
            "type": "base64",
            "media_type": "image/png",
            "data": "<base64 PNG data>"
          }
        },
        {
          "type": "text",
          "text": "Analyze this schematic..."
        }
      ]
    }
  ],
  "max_tokens": 4096
}
```

**Fallback Options:**
- Direct Anthropic API (requires `ANTHROPIC_API_KEY`)
- Kimi K2.5 via Moonshot API (alternative vision model)
- Gemini 2.5 Pro via OpenRouter (if Kimi unavailable)

---

## Setup Instructions

### Option 1: Local Development (Recommended for Testing)

**Step 1: Build kicad-worker Docker Image**

```bash
cd /Users/don/Adverant/adverant-nexus-ee-design-partner
docker build \
  -t adverant/kicad-worker:latest \
  -f services/nexus-ee-design/docker/mapos/Dockerfile \
  .
```

**Step 2: Run kicad-worker Container**

```bash
docker run -d \
  --name kicad-worker \
  -p 8080:8080 \
  -e OPENROUTER_API_KEY="sk-or-v1-..." \
  -e DISPLAY=:99 \
  -e KICAD_HEADLESS=1 \
  -v /tmp/kicad-exports:/schematic-data \
  adverant/kicad-worker:latest
```

**Step 3: Verify Service Health**

```bash
curl http://localhost:8080/health
# Expected: {"status": "healthy", "timestamp": "..."}
```

**Step 4: Test Image Export**

```bash
# Create test schematic content
cat > /tmp/test.kicad_sch << 'EOF'
(kicad_sch (version 20230121) (generator eeschema)
  (sheet (at 0 0) (size 297 210) (fields_autoplaced)
    (title "Test Schematic")
    (date "2026-02-07")
  )
)
EOF

# Export via kicad-worker
curl -X POST http://localhost:8080/v1/schematic/export \
  -H "Content-Type: application/json" \
  -d @- << 'PAYLOAD'
{
  "schematic_content": "$(cat /tmp/test.kicad_sch | jq -Rs .)",
  "export_format": "png",
  "design_name": "test",
  "dpi": 300
}
PAYLOAD
```

**Step 5: Configure Python Environment**

```bash
# Set kicad-worker URL for local development
export KICAD_WORKER_URL="http://localhost:8080"

# Set OpenRouter API key
export OPENROUTER_API_KEY="sk-or-v1-..."

# Run visual validator test
cd /Users/don/Adverant/adverant-nexus-ee-design-partner/services/nexus-ee-design/python-scripts
python agents/visual_validator/dual_llm_validator.py /tmp/test_schematic.kicad_sch
```

---

### Option 2: Kubernetes Deployment (Production)

**Prerequisites:**
- Kubernetes cluster running (k3s, minikube, or cloud provider)
- `kubectl` configured with access to cluster
- Docker registry accessible from cluster

**Step 1: Build and Push Image**

```bash
cd /Users/don/Adverant/adverant-nexus-ee-design-partner

# Build image
docker build \
  -t localhost:5000/mapos-kicad-worker:latest \
  -f services/nexus-ee-design/docker/mapos/Dockerfile \
  .

# Push to registry (adjust registry URL as needed)
docker push localhost:5000/mapos-kicad-worker:latest
```

**Step 2: Create Kubernetes Secrets**

```bash
# Set API keys
export OPENROUTER_API_KEY="sk-or-v1-..."
export ANTHROPIC_API_KEY="sk-ant-..."  # optional

# Create secret in cluster
kubectl create namespace nexus --dry-run=client -o yaml | kubectl apply -f -
kubectl create secret generic nexus-secrets \
  --from-literal=OPENROUTER_API_KEY="$OPENROUTER_API_KEY" \
  --from-literal=ANTHROPIC_API_KEY="$ANTHROPIC_API_KEY" \
  -n nexus --dry-run=client -o yaml | kubectl apply -f -
```

**Step 3: Deploy kicad-worker**

```bash
kubectl apply -f k8s/mapos-kicad-worker.yaml
```

**Step 4: Verify Deployment**

```bash
# Check pod status
kubectl get pods -n nexus -l app.kubernetes.io/name=mapos-kicad-worker

# Check service
kubectl get svc -n nexus mapos-kicad-worker

# View logs
kubectl logs -n nexus deployment/mapos-kicad-worker -c kicad-worker

# Test health endpoint
kubectl port-forward -n nexus svc/mapos-kicad-worker 8080:8080 &
curl http://localhost:8080/health
```

**Step 5: Configure Application**

For pods running in the same namespace:
```bash
# Automatic via DNS
export KICAD_WORKER_URL="http://mapos-kicad-worker:8080"
```

For external access:
```bash
# Via port-forward
kubectl port-forward -n nexus svc/mapos-kicad-worker 8080:8080
export KICAD_WORKER_URL="http://localhost:8080"

# OR via Ingress (if configured)
export KICAD_WORKER_URL="https://kicad-worker.your-domain.com"
```

---

## Troubleshooting Guide

### Problem: "Cannot connect to kicad-worker"

**Error:**
```
================================================================================
IMAGE EXTRACTION FAILED
================================================================================
Error: Cannot connect to kicad-worker: Connection refused
KiCad Worker URL: http://mapos-kicad-worker:8080
...
TROUBLESHOOTING:
1. Verify kicad-worker is running: curl http://mapos-kicad-worker:8080/health
```

**Diagnosis Steps:**

1. **Check if service is running:**
   ```bash
   # Docker
   docker ps | grep kicad-worker

   # Kubernetes
   kubectl get pods -n nexus -l app.kubernetes.io/name=mapos-kicad-worker
   ```

2. **Test connectivity:**
   ```bash
   # Docker
   curl http://localhost:8080/health

   # Kubernetes (from within cluster)
   kubectl run -it --rm debug --image=curlimages/curl --restart=Never -- \
     curl http://mapos-kicad-worker:8080/health
   ```

3. **Check logs:**
   ```bash
   # Docker
   docker logs kicad-worker

   # Kubernetes
   kubectl logs -n nexus deployment/mapos-kicad-worker -c kicad-worker
   ```

**Common Fixes:**
- Ensure Docker container is running: `docker start kicad-worker`
- Check port mapping: `-p 8080:8080`
- Verify `KICAD_WORKER_URL` environment variable is set correctly
- For K8s, ensure service and pod are in the same namespace

---

### Problem: "kicad-cli export failed"

**Error:**
```json
{
  "success": false,
  "errors": ["kicad-cli error: /bin/sh: kicad-cli: not found"]
}
```

**Diagnosis:**
- KiCad is not installed in the kicad-worker container
- Container was built incorrectly

**Fix:**
```bash
# Rebuild container with correct Dockerfile
docker build -f services/nexus-ee-design/docker/mapos/Dockerfile -t adverant/kicad-worker:latest .

# Verify KiCad installation
docker run --rm adverant/kicad-worker:latest which kicad-cli
# Should output: /usr/bin/kicad-cli
```

---

### Problem: "PNG conversion failed"

**Error:**
```json
{
  "success": true,
  "errors": ["ImageMagick not installed - PNG conversion unavailable"]
}
```

**Diagnosis:**
- ImageMagick is not installed in container
- SVG export succeeded, but PNG conversion failed

**Fix:**
```bash
# ImageMagick should be installed via Dockerfile
# Verify installation
docker run --rm adverant/kicad-worker:latest which convert
# Should output: /usr/bin/convert

# If missing, rebuild with updated Dockerfile
```

---

### Problem: "Opus 4.6 API call failed"

**Error:**
```python
logger.error(f"Opus analysis error: 401 Unauthorized")
raise RuntimeError("No Opus client available. Set OPENROUTER_API_KEY...")
```

**Diagnosis:**
- `OPENROUTER_API_KEY` is not set or invalid
- OpenRouter API quota exceeded
- Network connectivity issues

**Fix:**
```bash
# Verify API key is set
echo $OPENROUTER_API_KEY
# Should start with: sk-or-v1-

# Test API directly
curl https://openrouter.ai/api/v1/models \
  -H "Authorization: Bearer $OPENROUTER_API_KEY"

# Check API quota on OpenRouter dashboard
# https://openrouter.ai/credits
```

---

### Problem: "Schematic visually unchanged"

**Warning:**
```
result.errors.append("VISUAL_UNCHANGED: Image hash matches previous iteration")
```

**Diagnosis:**
- Fixes were applied to schematic content, but visual output is identical
- Could indicate:
  - Fixes didn't actually change the schematic
  - KiCad rendering is deterministic and ignores certain changes
  - Fix applicator is not working correctly

**Fix:**
- Review applied fixes in logs
- Manually inspect schematic `.kicad_sch` file for changes
- Check `fix_applicator.py` for bugs
- Enable `save_to_disk=True` in SchematicImageExtractor to compare PNGs visually

---

## Environment Variables Reference

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `KICAD_WORKER_URL` | Yes | `http://mapos-kicad-worker:8080` | URL of kicad-worker HTTP service |
| `OPENROUTER_API_KEY` | Yes | None | OpenRouter API key for Opus 4.6 vision |
| `ANTHROPIC_API_KEY` | No | None | Direct Anthropic API key (fallback) |
| `KICAD_HEADLESS` | No | `1` | Enable headless KiCad operation |
| `DISPLAY` | No | `:99` | X11 display for Xvfb |
| `PCB_DATA_DIR` | No | `/data` | KiCad worker data directory |
| `SCHEMATIC_DATA_DIR` | No | `/schematic-data` | Temporary schematic export directory |
| `OUTPUT_DIR` | No | `/output` | Optimization output directory |

---

## Code Integration Points

### Where kicad-worker is Used

**1. SchematicImageExtractor (`image_extractor.py`)**
- Line 34: `KICAD_WORKER_URL = os.environ.get('KICAD_WORKER_URL', 'http://mapos-kicad-worker:8080')`
- Line 210: `export_url = urljoin(self.kicad_worker_url, "/v1/schematic/export")`
- Line 256: `full_download_url = urljoin(self.kicad_worker_url, download_url)`

**2. ValidationLoop (`dual_llm_validator.py`)**
- Line 1074: Creates SchematicImageExtractor instance
- Line 1090: Calls `image_extractor.extract_png()` for each iteration

**3. MAPO Pipeline (`mapo_schematic_pipeline.py`)**
- Line 744: Initializes SchematicImageExtractor with environment URL

**4. ArtifactExporterAgent (`artifact_exporter_agent.py`)**
- Line 60: Config field `kicad_worker_url`
- Line 380, 431, 482: Uses kicad-worker for PDF/SVG/PNG export

---

## Testing the Full Stack

**Complete End-to-End Test:**

```bash
#!/bin/bash
set -e

echo "=== Visual Validation Stack Test ==="

# 1. Start kicad-worker
echo "[1/6] Starting kicad-worker..."
docker run -d --name kicad-worker-test \
  -p 8080:8080 \
  -e OPENROUTER_API_KEY="$OPENROUTER_API_KEY" \
  adverant/kicad-worker:latest
sleep 5

# 2. Health check
echo "[2/6] Testing health endpoint..."
curl -f http://localhost:8080/health || {
  echo "FAILED: kicad-worker not healthy"
  exit 1
}

# 3. Create test schematic
echo "[3/6] Creating test schematic..."
cat > /tmp/test_led.kicad_sch << 'EOF'
(kicad_sch (version 20230121) (generator eeschema)
  (sheet (at 0 0) (size 297 210)
    (title "LED Test Circuit")
  )
  (symbol (lib_id "Device:R") (at 100 100 0)
    (property "Reference" "R1" (at 102 98 0))
    (property "Value" "220R" (at 102 102 0))
  )
  (symbol (lib_id "Device:LED") (at 120 100 0)
    (property "Reference" "D1" (at 122 98 0))
  )
  (wire (pts (xy 90 100) (xy 95 100)))
  (wire (pts (xy 105 100) (xy 115 100)))
)
EOF

# 4. Test image extraction
echo "[4/6] Testing PNG export via kicad-worker..."
export KICAD_WORKER_URL="http://localhost:8080"
cd /Users/don/Adverant/adverant-nexus-ee-design-partner/services/nexus-ee-design/python-scripts

python3 << 'PYTHON'
import asyncio
from agents.visual_validator.image_extractor import SchematicImageExtractor

async def test():
    extractor = SchematicImageExtractor(save_to_disk=True)

    # Read test schematic
    with open('/tmp/test_led.kicad_sch', 'r') as f:
        content = f.read()

    # Extract PNG
    result = await extractor.extract_png(content, design_name="led_test")

    print(f"Success: {result.success}")
    print(f"Image size: {result.image_size_bytes} bytes")
    print(f"Hash: {result.image_hash}")
    print(f"Saved to: {result.image_path}")

    await extractor.close()

asyncio.run(test())
PYTHON

# 5. Test visual validation
echo "[5/6] Testing dual-LLM visual validation..."
python3 agents/visual_validator/dual_llm_validator.py /tmp/test_led.kicad_sch

# 6. Cleanup
echo "[6/6] Cleaning up..."
docker stop kicad-worker-test
docker rm kicad-worker-test

echo "=== Test Complete ==="
```

---

## Deployment Checklist

Before running MAPO v3.0 visual validation in production:

- [ ] kicad-worker Docker image built and pushed to registry
- [ ] Kubernetes deployment applied (`kubectl apply -f k8s/mapos-kicad-worker.yaml`)
- [ ] Pod is running (`kubectl get pods -n nexus`)
- [ ] Service is healthy (`curl http://mapos-kicad-worker:8080/health`)
- [ ] Secrets configured (`kubectl get secret nexus-secrets -n nexus`)
- [ ] `OPENROUTER_API_KEY` is valid and has sufficient credits
- [ ] Environment variable `KICAD_WORKER_URL` set correctly in application
- [ ] Test PNG export via `/v1/schematic/export` endpoint works
- [ ] Test Opus 4.6 vision API call succeeds
- [ ] Full validation loop completes without errors
- [ ] Persistent volume claim created for schematic exports (optional)

---

## Alternative Fallback Strategies

If kicad-worker cannot be deployed, consider these alternatives:

### 1. Local kicad-cli Execution (No HTTP Service)

**Pros:** No Docker/K8s required
**Cons:** Requires KiCad installed on every machine running validation

Modify `dual_llm_validator.py::export_schematic_to_image()` to run `kicad-cli` directly:
```python
subprocess.run(['kicad-cli', 'sch', 'export', 'svg', ...])
```

### 2. Screenshot-Based Validation (No kicad-cli)

**Pros:** Works with any KiCad installation
**Cons:** Requires GUI, not suitable for headless CI/CD

Use Playwright/Selenium to automate KiCad GUI and capture screenshots.

### 3. Text-Based Validation (No Images)

**Pros:** No infrastructure dependencies
**Cons:** NOT visual validation - defeats the purpose

Analyze `.kicad_sch` S-expressions directly without rendering. This is what MAPO v2.x did, and it's what v3.0 is designed to replace.

---

## Conclusion

The MAPO v3.0 visual validation system has a **hard dependency** on the kicad-worker HTTP service. Without it, the system will fail with `ImageExtractionError`.

**Current Status on Development Machine:**
- ❌ kicad-worker: NOT RUNNING
- ❌ kicad-cli: NOT INSTALLED LOCALLY (expected - should be in container)
- ⚠️ OPENROUTER_API_KEY: NOT CONFIGURED
- ❌ Kubernetes: NOT CONFIGURED

**To proceed with visual validation, you must:**
1. Deploy kicad-worker (Docker or K8s)
2. Set `KICAD_WORKER_URL` environment variable
3. Configure `OPENROUTER_API_KEY` for Opus 4.6 vision access

**Without these dependencies, visual validation will fail at the first iteration.**
