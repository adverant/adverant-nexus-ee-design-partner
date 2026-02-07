# Visual Verification Agent

**MAPO v3.1 Visual Quality Checker**

Uses Claude Opus 4.5 to visually assess schematic quality and reject poor layouts before manufacturing.

## Overview

The Visual Verification Agent generates high-quality images of KiCad schematics and uses AI vision to score them against a comprehensive quality rubric. This provides an objective, automated quality gate for the MAPO pipeline.

## Features

- **Automated Image Generation**: Exports schematics to PNG/SVG using KiCad CLI
- **Vision-Based Analysis**: Uses Claude Opus 4.5 for visual quality assessment
- **Comprehensive Rubric**: Scores 8 quality dimensions with weighted scoring
- **Production Quality Gate**: Pass threshold of 90/100 ensures only high-quality schematics proceed
- **Detailed Reports**: Identifies specific issues with actionable suggestions

## Quality Rubric

| Criterion | Weight | Pass Criteria |
|-----------|--------|---------------|
| Symbol Overlap | 15% | No overlapping symbols |
| Wire Crossings | 12% | Minimize crossings, use junction dots |
| Signal Flow | 15% | Left-to-right signal flow visible |
| Power Flow | 10% | Top-to-bottom power distribution |
| Functional Grouping | 15% | Related components grouped visually |
| Net Labels | 10% | All labels readable, no overlaps |
| Spacing | 10% | Adequate spacing between components |
| Professional Appearance | 13% | Overall professional, production-ready look |

**Pass Threshold**: 90/100 weighted score

## Installation

### Requirements

```bash
# Python dependencies
pip install httpx pyyaml

# System dependencies
brew install kicad          # KiCad 8.x for kicad-cli
brew install imagemagick    # Optional: for PNG conversion
```

### Environment

```bash
export OPENROUTER_API_KEY="your_openrouter_api_key"
```

## Usage

### Python API

```python
from agents.visual_verification import VisualVerifier

# Initialize verifier
verifier = VisualVerifier()

# Verify schematic
report = await verifier.verify(
    schematic_path="design.kicad_sch",
    output_dir="/tmp/verification"
)

# Check results
if report.passed:
    print(f"✓ PASS: Score {report.overall_score:.1f}/100")
else:
    print(f"✗ FAIL: Score {report.overall_score:.1f}/100")
    for issue in report.issues:
        print(f"  - {issue.criterion_name}: {issue.issue_description}")
```

### Command Line

```bash
# Basic usage
python visual_verifier.py schematic.kicad_sch

# Custom output directory
python visual_verifier.py schematic.kicad_sch /tmp/output

# Exit code: 0 = passed, 1 = failed
```

### Convenience Function

```python
from agents.visual_verification import verify_schematic

report = await verify_schematic("design.kicad_sch")
print(report)
```

## Integration with MAPO Pipeline

### Step 1: Add to Pipeline

```python
from agents.visual_verification import VisualVerifier

class SchematicMapoOptimizer:
    def __init__(self):
        self.visual_verifier = VisualVerifier()

    async def optimize_with_visual_verification(self, schematic_path):
        # Run MAPO optimization
        optimized = await self.mapo_optimize(schematic_path)

        # Visual quality check
        report = await self.visual_verifier.verify(
            schematic_path=optimized,
            output_dir="/tmp/mapo_verification"
        )

        # Reject if quality too low
        if not report.passed:
            raise QualityError(
                f"Visual verification failed: {report.overall_score:.1f}/100\n"
                f"Issues: {len(report.issues)}"
            )

        return optimized, report
```

### Step 2: Quality Gate Configuration

Set pass threshold in `quality_rubric.yaml`:

```yaml
thresholds:
  excellent: 95  # Outstanding quality
  pass: 90       # Production-ready minimum
  acceptable: 70 # Needs improvement
  poor: 50       # Major issues
```

### Step 3: Logging and Monitoring

```python
import logging

logger = logging.getLogger(__name__)

report = await verifier.verify(schematic_path)

logger.info(f"Visual verification: {report.overall_score:.1f}/100")
for issue in report.issues:
    logger.warning(
        f"Quality issue [{issue.criterion_name}]: "
        f"{issue.issue_description} (Score: {issue.score:.1f})"
    )
```

## Report Structure

```python
@dataclass
class VisualQualityReport:
    passed: bool                     # True if overall_score >= 90
    overall_score: float             # 0-100 weighted average
    criterion_scores: Dict[str, float]  # score per rubric item
    issues: List[QualityIssue]       # specific problems identified
    image_path: str                  # path to generated image
    model_used: str                  # "anthropic/claude-opus-4-5"
    timestamp: str                   # ISO format
    overall_assessment: str          # summary from Opus
    production_ready: bool           # direct assessment from Opus
```

### Example Report Output

```
================================================================================
VISUAL QUALITY REPORT - ✗ FAIL
================================================================================
Overall Score: 72.3/100
Production Ready: No
Model: anthropic/claude-opus-4-5
Timestamp: 2026-02-07T14:30:45

Assessment: This schematic has good signal flow but suffers from excessive
wire crossings and cramped component spacing. Requires rework before production.

--------------------------------------------------------------------------------
CRITERION SCORES:
--------------------------------------------------------------------------------
  ✓ symbol_overlap              100.0/100
  ⚠ wire_crossings               60.0/100
  ✓ signal_flow                  90.0/100
  ✓ power_flow                   95.0/100
  ⚠ functional_grouping          75.0/100
  ✓ net_labels                   85.0/100
  ✗ spacing                      50.0/100
  ⚠ professional_appearance      70.0/100

--------------------------------------------------------------------------------
ISSUES IDENTIFIED (5):
--------------------------------------------------------------------------------

1. Wire Crossings (Score: 60.0/100)
   Issue: 18 wire crossings found in power distribution section
   Suggestion: Reroute VCC and GND nets to minimize crossings, use net labels

2. Functional Grouping (Score: 75.0/100)
   Issue: Power regulation components scattered across schematic
   Suggestion: Group U1, C1-C4, L1 together as power supply block

3. Spacing (Score: 50.0/100)
   Issue: Components extremely cramped in upper-right quadrant
   Suggestion: Expand canvas, increase grid spacing to 2 units minimum

4. Professional Appearance (Score: 70.0/100)
   Issue: Overall layout appears rushed and unbalanced
   Suggestion: Balance component distribution, align power symbols

================================================================================
```

## Testing

```bash
# Run test suite
pytest test_visual_verifier.py -v

# Run with coverage
pytest test_visual_verifier.py --cov=visual_verifier --cov-report=html

# Test specific cases
pytest test_visual_verifier.py::TestEndToEnd::test_verify_full_flow_excellent -v
```

## Error Handling

The verifier provides verbose error messages:

```python
try:
    report = await verifier.verify(schematic_path)
except VerificationError as e:
    # Errors include:
    # - KiCad CLI not found
    # - Schematic file not found
    # - Invalid file format
    # - API key missing
    # - API request failed
    # - Timeout exceeded
    logger.error(f"Verification failed: {e}")
```

## Performance

- **Image Generation**: ~2-5 seconds (KiCad CLI + ImageMagick)
- **Opus 4.5 Analysis**: ~10-30 seconds (depends on image size)
- **Total Time**: ~15-35 seconds per schematic

## Limitations

1. **KiCad Dependency**: Requires KiCad 8.x with `kicad-cli` in PATH
2. **Image Size**: Large multi-sheet schematics may exceed vision model limits
3. **Subjectivity**: AI scoring has variance (~±5 points between runs)
4. **Cost**: Each verification costs ~$0.10-0.30 depending on image size

## Troubleshooting

### KiCad CLI Not Found

```bash
# macOS
brew install kicad

# Verify installation
kicad-cli --version
```

### ImageMagick Not Found

```bash
# Optional but recommended
brew install imagemagick

# Verify installation
convert --version
```

### API Key Issues

```bash
# Check environment variable
echo $OPENROUTER_API_KEY

# Test API access
curl -H "Authorization: Bearer $OPENROUTER_API_KEY" \
  https://openrouter.ai/api/v1/models
```

### Low Scores on Good Schematics

Adjust the rubric in `quality_rubric.yaml` to match your standards:

```yaml
thresholds:
  pass: 85  # Lower threshold if scores consistently good but under 90
```

## Future Enhancements

- [ ] Multi-sheet schematic support (stitch images)
- [ ] Historical score tracking (detect quality trends)
- [ ] Custom rubric templates per project type
- [ ] Integration with version control (PR quality gates)
- [ ] Batch verification mode
- [ ] Cost optimization (caching, smaller models for pre-screening)

## License

Copyright © 2026 Adverant. All rights reserved.
