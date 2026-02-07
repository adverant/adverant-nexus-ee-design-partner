# Visual Verification Agent - Quick Start Guide

**Get started with visual schematic quality checking in 5 minutes**

---

## Prerequisites

- Python 3.10+ (✓ already installed)
- KiCad 8.x
- OpenRouter API key

---

## Installation (3 steps)

### Step 1: Install Python Dependencies

```bash
cd /Users/don/Adverant/adverant-nexus-ee-design-partner/services/nexus-ee-design/python-scripts
pip install -r requirements.txt
```

This installs:
- `httpx` - API client
- `PyYAML` - Rubric parsing
- Plus existing dependencies

### Step 2: Install KiCad CLI

```bash
brew install kicad
```

Verify installation:
```bash
kicad-cli --version
# Should show: KiCad 8.x
```

### Step 3: Set API Key

```bash
export OPENROUTER_API_KEY='your_openrouter_api_key'
```

Make it permanent:
```bash
echo "export OPENROUTER_API_KEY='your_key'" >> ~/.zshrc
source ~/.zshrc
```

**Get API key**: https://openrouter.ai/keys

---

## Verify Setup

```bash
cd agents/visual_verification
python setup_check.py
```

Expected output:
```
✓ All CRITICAL checks passed
✓ Setup is complete!
```

---

## Usage Examples

### Example 1: Verify Single Schematic

```bash
python visual_verifier.py ../../output/test_1.kicad_sch
```

Output:
```
================================================================================
VISUAL QUALITY REPORT - ✓ PASS
================================================================================
Overall Score: 92.3/100
Production Ready: Yes
Model: anthropic/claude-opus-4-5

Assessment: Professional schematic with good signal flow and layout...

--------------------------------------------------------------------------------
CRITERION SCORES:
--------------------------------------------------------------------------------
  ✓ symbol_overlap              100.0/100
  ✓ wire_crossings               95.0/100
  ✓ signal_flow                  90.0/100
  ...
```

### Example 2: Python API

```python
from agents.visual_verification import verify_schematic

# Async usage
import asyncio

async def main():
    report = await verify_schematic("design.kicad_sch")

    if report.passed:
        print(f"✓ Quality check passed: {report.overall_score:.1f}/100")
    else:
        print(f"✗ Quality check failed: {report.overall_score:.1f}/100")
        for issue in report.issues:
            print(f"  - {issue.criterion_name}: {issue.issue_description}")

asyncio.run(main())
```

### Example 3: MAPO Pipeline Integration

```python
from agents.visual_verification import VisualVerifier

class SchematicPipeline:
    def __init__(self):
        self.visual_verifier = VisualVerifier()

    async def process(self, schematic_path):
        # Run MAPO optimization
        optimized = await self.optimize_layout(schematic_path)

        # Visual quality gate
        report = await self.visual_verifier.verify(optimized)

        if not report.passed:
            raise QualityGateFailure(
                f"Score: {report.overall_score:.1f}/100"
            )

        return optimized
```

---

## Run Examples

```bash
# Basic usage
python example_usage.py 1

# Batch verification
python example_usage.py 2

# Pipeline integration demo
python example_usage.py 3

# Custom rubric
python example_usage.py 4
```

---

## Run Tests

```bash
# All tests
pytest test_visual_verifier.py -v

# With coverage
pytest test_visual_verifier.py --cov=visual_verifier --cov-report=html

# Specific test
pytest test_visual_verifier.py::TestEndToEnd::test_verify_full_flow_excellent
```

---

## Troubleshooting

### Issue: "KiCad CLI not found"

```bash
# Install KiCad
brew install kicad

# Verify
which kicad-cli
```

### Issue: "API key not set"

```bash
# Check current value
echo $OPENROUTER_API_KEY

# Set temporarily
export OPENROUTER_API_KEY='your_key'

# Set permanently
echo "export OPENROUTER_API_KEY='your_key'" >> ~/.zshrc
source ~/.zshrc
```

### Issue: "Module not found: httpx"

```bash
# Install dependencies
cd ../../  # to python-scripts directory
pip install -r requirements.txt
```

### Issue: "convert command not found"

This is optional. Install ImageMagick for better image quality:
```bash
brew install imagemagick
```

Or ignore - the agent will use SVG format instead.

---

## Configuration

### Adjust Pass Threshold

Edit `quality_rubric.yaml`:
```yaml
thresholds:
  pass: 85  # Lower from 90 if too strict
```

### Customize Rubric Weights

Edit `quality_rubric.yaml`:
```yaml
rubric:
  - id: symbol_overlap
    weight: 0.20  # Increase importance
```

### Change Model

Edit `visual_verifier.py` (line 161):
```python
self.model = "anthropic/claude-opus-4-5"  # Current
# self.model = "anthropic/claude-sonnet-4-5"  # Cheaper, faster
```

---

## Cost Estimation

- **Per verification**: ~$0.10-0.30
- **Model**: Claude Opus 4.5 (vision)
- **Depends on**: Image size, complexity

**Optimization tips**:
- Use SVG instead of PNG (smaller)
- Cache results for identical schematics
- Use Sonnet 4.5 for pre-screening

---

## Next Steps

1. ✓ Install dependencies
2. ✓ Verify setup with `setup_check.py`
3. ✓ Run first verification
4. → Integrate into MAPO pipeline
5. → Configure quality thresholds per project
6. → Set up monitoring and logging

---

## Support

- **Documentation**: See `README.md`
- **Implementation Details**: See `IMPLEMENTATION_REPORT.md`
- **Examples**: Run `python example_usage.py`
- **Tests**: Run `pytest test_visual_verifier.py -v`

---

**Ready to start? Run `python setup_check.py` to verify your environment!**
