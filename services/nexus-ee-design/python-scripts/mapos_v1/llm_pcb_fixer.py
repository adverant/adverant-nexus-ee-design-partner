#!/usr/bin/env python3
"""
LLM PCB Fixer - Use Claude Opus 4.5 for intelligent PCB DRC violation fixing.

This replaces brittle regex patterns with LLM-guided analysis and fix generation.
The LLM analyzes DRC reports, understands the PCB structure, and generates
targeted S-expression modifications.

NO MOCKS, NO STUBS, NO SHORTCUTS - Real LLM calls and real PCB modifications.

Uses OpenRouter API with Claude Opus 4.5 model.
Integrates fix pattern caching to reduce repeat LLM calls (~$0.30 each).
"""

import os
import sys
import json
import shutil
import subprocess
import argparse
import requests
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass

# Import fix pattern cache for cost savings
try:
    from fix_cache import get_fix_cache, FixPatternCache
    FIX_CACHE_AVAILABLE = True
except ImportError:
    FIX_CACHE_AVAILABLE = False
    get_fix_cache = None


@dataclass
class DRCViolation:
    """Single DRC violation."""
    type: str
    severity: str
    description: str
    positions: List[Tuple[float, float]]
    items: List[str]


@dataclass
class FixProposal:
    """LLM-generated fix proposal."""
    description: str
    modification_type: str
    s_expression_patch: str
    confidence: float
    affected_elements: List[str]


class LLMPCBFixer:
    """
    Use Claude Opus 4.5 to intelligently fix PCB DRC violations.

    The LLM:
    1. Analyzes DRC report to understand violation patterns
    2. Examines relevant sections of the PCB file
    3. Generates S-expression modifications to fix violations
    4. Validates fixes don't introduce new issues
    """

    def __init__(
        self,
        pcb_path: str,
        api_key: Optional[str] = None,
        target_violations: int = 0,
        max_iterations: int = 20
    ):
        self.pcb_path = Path(pcb_path)
        self.target_violations = target_violations
        self.max_iterations = max_iterations
        self.kicad_cli = self._find_kicad_cli()

        # Initialize OpenRouter client for Claude Opus 4.5
        self.api_key = api_key or os.environ.get('OPENROUTER_API_KEY')
        if not self.api_key:
            raise ValueError("OPENROUTER_API_KEY required for LLM-guided fixing")

        self.openrouter_url = "https://openrouter.ai/api/v1/chat/completions"
        self.model = "anthropic/claude-opus-4.5"

        if not self.pcb_path.exists():
            raise FileNotFoundError(f"PCB file not found: {pcb_path}")

        # Output directory
        self.output_dir = self.pcb_path.parent / 'llm_fixer_output'
        self.output_dir.mkdir(exist_ok=True)

        # Load PCB content
        with open(self.pcb_path, 'r') as f:
            self.pcb_content = f.read()

    def _find_kicad_cli(self) -> Optional[str]:
        """Find kicad-cli executable."""
        paths = [
            '/Applications/KiCad/KiCad.app/Contents/MacOS/kicad-cli',
            '/usr/bin/kicad-cli',
            '/usr/local/bin/kicad-cli',
            shutil.which('kicad-cli'),
        ]
        for path in paths:
            if path and Path(path).exists():
                return path
        return None

    def _find_kicad_python(self) -> Optional[str]:
        """Find Python interpreter with pcbnew module available."""
        # Check if pcbnew is available in current Python
        try:
            import pcbnew
            return sys.executable
        except ImportError:
            pass

        # Search common KiCad Python locations
        paths = [
            # macOS
            '/Applications/KiCad/KiCad.app/Contents/Frameworks/Python.framework/Versions/3.9/bin/python3',
            '/Applications/KiCad/KiCad.app/Contents/Frameworks/Python.framework/Versions/3.11/bin/python3',
            # Linux (Ubuntu/Debian with KiCad from PPA)
            '/usr/bin/python3',
            # Container environments
            sys.executable,
        ]

        for path in paths:
            if path and Path(path).exists():
                # Verify pcbnew is importable
                try:
                    result = subprocess.run(
                        [path, '-c', 'import pcbnew; print(pcbnew.Version())'],
                        capture_output=True, text=True, timeout=10
                    )
                    if result.returncode == 0:
                        return path
                except Exception:
                    continue

        return None

    def run_drc(self) -> Dict[str, Any]:
        """Run KiCad DRC and return parsed results."""
        if not self.kicad_cli:
            raise RuntimeError("kicad-cli not found")

        output_path = self.output_dir / 'drc_report.json'

        result = subprocess.run([
            self.kicad_cli, 'pcb', 'drc',
            '--output', str(output_path),
            '--format', 'json',
            '--severity-all',
            str(self.pcb_path)
        ], capture_output=True, text=True, timeout=180)

        if not output_path.exists():
            raise RuntimeError(f"DRC failed: {result.stderr}")

        with open(output_path) as f:
            return json.load(f)

    def analyze_violations_with_llm(self, drc_data: Dict) -> str:
        """Use LLM to analyze DRC violations and suggest fixes.

        Uses fix pattern caching to avoid repeat LLM calls for known violation types.
        """

        # Check cache first for known fix patterns
        violations = drc_data.get('violations', [])
        unconnected = drc_data.get('unconnected_items', [])

        cached_operations = []
        uncached_violations = []

        if FIX_CACHE_AVAILABLE and get_fix_cache is not None:
            cache = get_fix_cache()
            for v in violations:
                cached = cache.get(v)
                if cached:
                    cached_operations.extend(cached)
                    print(f"  Cache hit for violation type: {v.get('type', 'unknown')}")
                else:
                    uncached_violations.append(v)

            # If all violations are cached, return cached operations directly
            if not uncached_violations and cached_operations:
                cache_stats = cache.get_stats()
                print(f"\n  All {len(violations)} violations matched cache patterns!")
                print(f"  Estimated savings: {cache_stats['savings_usd']}")
                return json.dumps({
                    'analysis': 'All violations matched cached fix patterns',
                    'recommended_operations': cached_operations,
                    'from_cache': True,
                    'cache_stats': cache_stats,
                })
        else:
            uncached_violations = violations

        # Prepare violation summary for uncached violations

        # Group by type
        by_type = {}
        for v in violations:
            vtype = v.get('type', 'unknown')
            if vtype not in by_type:
                by_type[vtype] = []
            by_type[vtype].append(v)

        violation_summary = f"""
Total violations: {len(violations)}
Unconnected items: {len(unconnected)}

Violations by type:
"""
        for vtype, items in sorted(by_type.items(), key=lambda x: -len(x[1])):
            violation_summary += f"  {vtype}: {len(items)}\n"
            # Include first 3 examples
            for v in items[:3]:
                desc = v.get('description', 'No description')
                violation_summary += f"    - {desc}\n"

        # Extract relevant PCB sections (first 5000 chars of setup and zones)
        setup_match = self.pcb_content.find('(setup')
        setup_section = self.pcb_content[setup_match:setup_match+2000] if setup_match > 0 else "No setup section found"

        # Get first few zones
        zone_sections = []
        zone_start = 0
        for _ in range(3):
            idx = self.pcb_content.find('(zone ', zone_start)
            if idx < 0:
                break
            end = self.pcb_content.find('\n  )', idx) + 3
            zone_sections.append(self.pcb_content[idx:end])
            zone_start = end

        prompt = f"""You are a PCB design expert analyzing DRC (Design Rule Check) violations in a KiCad PCB file.

## DRC Report Summary
{violation_summary}

## PCB Setup Section (excerpt)
```
{setup_section}
```

## Zone Definitions (first 3)
```
{chr(10).join(zone_sections[:3]) if zone_sections else "No zones found"}
```

## Task
Analyze these violations and recommend which pcbnew API operations to run.

IMPORTANT: Do NOT generate S-expression regex patterns - they corrupt PCB files.
Instead, recommend which of these pcbnew API operations should be run:

Available pcbnew operations:
1. "zone_fill" - Fill all copper zones using ZONE_FILLER API
2. "fix_zone_nets" - Correct zone net assignments using zone.SetNet()
3. "fix_clearances" - Update design settings clearances (min clearance, solder mask, hole clearance)
4. "remove_dangling_vias" - Remove vias not connected to any track
5. "assign_orphan_nets" - Assign nets to unconnected pads based on component type
6. "adjust_trace_width" - Modify trace widths for a specific net (params: {{"net_name": "GND", "width_mm": 2.0}})
7. "adjust_power_traces" - Widen all power traces (GND, VCC, etc.) (params: {{"width_mm": 2.0}})

Output your analysis as JSON:
{{
  "analysis": "Brief analysis of root causes",
  "recommended_operations": [
    {{
      "operation": "operation_name from list above",
      "priority": 1-5 (1=highest),
      "reason": "Why this operation will help",
      "params": {{}},  // Optional parameters
      "estimated_reduction": number
    }}
  ],
  "manual_review_items": [
    "List any issues that require manual PCB layout review"
  ],
  "total_estimated_reduction": number
}}"""

        response = self._call_openrouter(prompt)
        return response

    def _call_openrouter(self, prompt: str, max_tokens: int = 4096, max_retries: int = 3) -> str:
        """
        Call OpenRouter API with Claude Opus 4.5 and exponential backoff retry.

        Implements retry pattern from Nexus Skills Engine for robust API calls.
        """
        if not self.api_key:
            raise RuntimeError("OPENROUTER_API_KEY not configured")

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://adverant.ai",
            "X-Title": "MAPOS PCB Fixer"
        }

        payload = {
            "model": self.model,
            "max_tokens": max_tokens,
            "temperature": 0.7,
            "messages": [
                {
                    "role": "system",
                    "content": """You are an expert PCB designer specializing in KiCad DRC violation fixing.
You have deep knowledge of:
- KiCad S-expression file format
- IPC-2221 clearance standards
- High-voltage design (58V) and high-current (100A) considerations
- DFM best practices for 10-layer boards
Generate precise S-expression modifications following exact KiCad syntax.
Always output valid JSON when requested."""
                },
                {"role": "user", "content": prompt}
            ]
        }

        last_error = None
        for attempt in range(max_retries):
            try:
                # Exponential backoff: 0s, 2s, 4s
                if attempt > 0:
                    import time
                    wait_time = 2 ** attempt
                    print(f"  Retry {attempt}/{max_retries} after {wait_time}s...")
                    time.sleep(wait_time)

                response = requests.post(
                    self.openrouter_url,
                    headers=headers,
                    json=payload,
                    timeout=120
                )

                # Check for rate limiting
                if response.status_code == 429:
                    retry_after = int(response.headers.get('Retry-After', 30))
                    print(f"  Rate limited, waiting {retry_after}s...")
                    import time
                    time.sleep(retry_after)
                    continue

                # Check for server errors (retryable)
                if response.status_code >= 500:
                    last_error = RuntimeError(f"OpenRouter server error: {response.status_code}")
                    continue

                if not response.ok:
                    error_text = response.text
                    # Non-retryable client errors
                    raise RuntimeError(f"OpenRouter call failed: {response.status_code} - {error_text}")

                data = response.json()
                content = data.get('choices', [{}])[0].get('message', {}).get('content')

                if not content:
                    last_error = RuntimeError("Empty response from OpenRouter")
                    continue

                return content

            except requests.Timeout:
                last_error = RuntimeError("OpenRouter request timed out")
                continue
            except requests.ConnectionError as e:
                last_error = RuntimeError(f"OpenRouter connection error: {e}")
                continue
            except requests.RequestException as e:
                last_error = RuntimeError(f"OpenRouter request failed: {e}")
                continue

        # All retries exhausted
        raise last_error or RuntimeError("OpenRouter call failed after all retries")

    def apply_llm_fix(self, fix: Dict) -> bool:
        """
        Apply a single LLM-recommended pcbnew operation.

        DEPRECATED: S-expression regex has been removed.
        This method now routes to pcbnew API operations.
        """
        operation = fix.get('operation', '')
        params = fix.get('params', {})
        reason = fix.get('reason', 'LLM recommended')

        if not operation:
            # Legacy format - skip
            print(f"  Skip: Legacy fix format (no operation specified)")
            return False

        print(f"  Executing pcbnew operation: {operation}")
        print(f"    Reason: {reason}")

        # Route to pcbnew scripts
        script_dir = Path(__file__).parent
        kicad_python = self._find_kicad_python()

        if not kicad_python:
            print(f"    Warning: KiCad Python not found")
            return False

        try:
            if operation == "zone_fill":
                script = script_dir / "kicad_zone_filler.py"
                result = subprocess.run(
                    [kicad_python, str(script), str(self.pcb_path)],
                    capture_output=True, text=True, timeout=120
                )

            elif operation == "fix_zone_nets":
                script = script_dir / "kicad_pcb_fixer.py"
                result = subprocess.run(
                    [kicad_python, str(script), "zone-nets", str(self.pcb_path)],
                    capture_output=True, text=True, timeout=120
                )

            elif operation == "fix_clearances":
                script = script_dir / "kicad_pcb_fixer.py"
                result = subprocess.run(
                    [kicad_python, str(script), "design-settings", str(self.pcb_path)],
                    capture_output=True, text=True, timeout=120
                )

            elif operation == "remove_dangling_vias":
                script = script_dir / "kicad_pcb_fixer.py"
                result = subprocess.run(
                    [kicad_python, str(script), "dangling-vias", str(self.pcb_path)],
                    capture_output=True, text=True, timeout=120
                )

            elif operation == "assign_orphan_nets":
                script = script_dir / "kicad_net_assigner.py"
                result = subprocess.run(
                    [kicad_python, str(script), str(self.pcb_path)],
                    capture_output=True, text=True, timeout=120
                )

            elif operation == "adjust_trace_width":
                script = script_dir / "kicad_trace_adjuster.py"
                net_name = params.get('net_name', 'GND')
                width_mm = params.get('width_mm', 2.0)
                result = subprocess.run(
                    [kicad_python, str(script), "net", str(self.pcb_path), net_name, str(width_mm)],
                    capture_output=True, text=True, timeout=120
                )

            elif operation == "adjust_power_traces":
                script = script_dir / "kicad_trace_adjuster.py"
                width_mm = params.get('width_mm', 2.0)
                result = subprocess.run(
                    [kicad_python, str(script), "power", str(self.pcb_path), "--width", str(width_mm)],
                    capture_output=True, text=True, timeout=120
                )

            elif operation == "add_thermal_vias":
                # This requires more complex pcbnew operations - log for future implementation
                print(f"    Note: {operation} requires custom pcbnew implementation (not yet available)")
                return False

            else:
                print(f"    Unknown operation: {operation}")
                return False

            if result.returncode == 0:
                print(f"    Success: {operation}")
                # Reload PCB content
                with open(self.pcb_path, 'r', encoding='utf-8') as f:
                    self.pcb_content = f.read()
                return True
            else:
                print(f"    Failed: {result.stderr[:200] if result.stderr else 'No error message'}")
                return False

        except subprocess.TimeoutExpired:
            print(f"    Timeout: {operation} took too long")
            return False
        except Exception as e:
            print(f"    Error: {e}")
            return False

    def generate_via_removal_instructions(self, unconnected: List[Dict]) -> str:
        """Use LLM to generate specific via removal instructions."""

        # Extract via positions from unconnected items
        via_positions = []
        for item in unconnected[:20]:  # First 20
            for via_info in item.get('items', []):
                pos = via_info.get('pos', {})
                desc = via_info.get('description', '')
                if 'Via' in desc:
                    via_positions.append({
                        'x': pos.get('x', 0),
                        'y': pos.get('y', 0),
                        'description': desc
                    })

        if not via_positions:
            return ""

        prompt = f"""You are editing a KiCad PCB file. These vias are marked as "unconnected" in DRC:

{json.dumps(via_positions[:10], indent=2)}

The vias appear as S-expressions like:
```
(via (at X Y) (size ...) (drill ...) (layers "F.Cu" "B.Cu") (net N) (uuid "..."))
```

Generate the exact S-expression patterns to remove these dangling vias.

For each via, provide:
1. The (at X Y) pattern to find
2. Whether to remove the entire via block

Output JSON:
{{
  "removals": [
    {{
      "position": [X, Y],
      "pattern_to_remove": "Full via S-expression to remove",
      "reason": "Why this via should be removed"
    }}
  ]
}}"""

        response = self._call_openrouter(prompt, max_tokens=2048)
        return response

    def run_pcbnew_fixes(self) -> Dict[str, Any]:
        """
        Run pcbnew-based fixes using KiCad's Python interpreter.

        These fixes are more reliable than S-expression regex because they
        use KiCad's native API and won't corrupt the file structure.
        """
        script_dir = Path(__file__).parent
        kicad_python = self._find_kicad_python()

        if not kicad_python:
            print("  Warning: KiCad Python not found, skipping pcbnew fixes")
            return {'success': False, 'error': 'KiCad Python not available'}

        results = {}

        # 1. Zone fill
        zone_script = script_dir / "kicad_zone_filler.py"
        if zone_script.exists():
            print("  Running zone fill...")
            try:
                result = subprocess.run(
                    [kicad_python, str(zone_script), str(self.pcb_path)],
                    capture_output=True, text=True, timeout=120
                )
                results['zone_fill'] = {
                    'success': result.returncode == 0,
                    'output': result.stdout[:500] if result.stdout else result.stderr[:500]
                }
                if result.returncode == 0:
                    print("    Zone fill completed")
            except Exception as e:
                results['zone_fill'] = {'success': False, 'error': str(e)}

        # 2. PCB fixes (zone nets, design settings, dangling vias)
        fixer_script = script_dir / "kicad_pcb_fixer.py"
        if fixer_script.exists():
            print("  Running pcbnew fixes...")
            try:
                result = subprocess.run(
                    [kicad_python, str(fixer_script), "all", str(self.pcb_path)],
                    capture_output=True, text=True, timeout=120
                )
                results['pcb_fixer'] = {
                    'success': result.returncode == 0,
                    'output': result.stdout[:500] if result.stdout else result.stderr[:500]
                }
                if result.returncode == 0:
                    print("    pcbnew fixes completed")
            except Exception as e:
                results['pcb_fixer'] = {'success': False, 'error': str(e)}

        # Reload PCB content after pcbnew modifications
        if any(r.get('success', False) for r in results.values()):
            with open(self.pcb_path, 'r', encoding='utf-8') as f:
                self.pcb_content = f.read()
            print("  Reloaded PCB content after pcbnew fixes")

        results['success'] = True
        return results

    def run_iteration(self, iteration: int) -> Dict[str, Any]:
        """Run a single iteration of LLM-guided fixing."""
        print(f"\n{'='*60}")
        print(f"LLM-GUIDED ITERATION {iteration}")
        print(f"{'='*60}")

        # Run DRC
        print("\nRunning DRC...")
        drc_data = self.run_drc()

        violations = drc_data.get('violations', [])
        unconnected = drc_data.get('unconnected_items', [])
        total = len(violations) + len(unconnected)

        print(f"Violations: {len(violations)}, Unconnected: {len(unconnected)}, Total: {total}")

        if total <= self.target_violations:
            return {
                'violations': total,
                'improved': True,
                'done': True
            }

        # Get LLM analysis
        print("\nAsking Opus 4.5 for fix recommendations...")
        analysis = self.analyze_violations_with_llm(drc_data)

        # Parse LLM response
        try:
            # Extract JSON from response
            import re
            json_match = re.search(r'\{[\s\S]*\}', analysis)
            if json_match:
                fix_data = json.loads(json_match.group())
            else:
                print("Failed to parse LLM response as JSON")
                return {'violations': total, 'improved': False, 'done': False}
        except json.JSONDecodeError as e:
            print(f"JSON parse error: {e}")
            return {'violations': total, 'improved': False, 'done': False}

        print(f"\nLLM Analysis: {fix_data.get('analysis', 'No analysis')[:200]}...")
        print(f"Estimated reduction: {fix_data.get('total_estimated_reduction', fix_data.get('estimated_reduction', 0))} violations")

        # Show manual review items if any
        manual_items = fix_data.get('manual_review_items', [])
        if manual_items:
            print(f"\nManual review needed for:")
            for item in manual_items[:5]:
                print(f"  - {item}")

        # Apply recommended pcbnew operations (sorted by priority)
        operations = fix_data.get('recommended_operations', fix_data.get('fixes', []))
        operations_sorted = sorted(operations, key=lambda x: x.get('priority', 5))

        fixes_applied = 0
        for op in operations_sorted:
            if self.apply_llm_fix(op):
                fixes_applied += 1

        # PCB is modified by pcbnew scripts directly (they save automatically)
        if fixes_applied > 0:
            print(f"\nExecuted {fixes_applied} pcbnew operations")

        # Verify improvement
        new_drc = self.run_drc()
        new_total = len(new_drc.get('violations', [])) + len(new_drc.get('unconnected_items', []))

        improvement = total - new_total
        print(f"\nResult: {total} -> {new_total} ({improvement:+d} violations)")

        # Store successful fix patterns in cache for future reuse
        if FIX_CACHE_AVAILABLE and get_fix_cache is not None and improvement > 0:
            cache = get_fix_cache()
            for v in violations:
                # Store fix pattern with success=True if we got improvement
                cache.put(v, operations, success=True)
            print(f"  Cached {len(violations)} fix patterns for future reuse")

        return {
            'violations': new_total,
            'improved': improvement > 0,
            'done': new_total <= self.target_violations
        }

    def run(self) -> Dict[str, Any]:
        """Run the LLM-guided fixing loop."""
        print("=" * 60)
        print("LLM PCB FIXER - Claude Opus 4.5 Guided")
        print("=" * 60)
        print(f"PCB: {self.pcb_path}")
        print(f"Target: {self.target_violations} violations")

        # Backup
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = self.output_dir / f"backup_{timestamp}.kicad_pcb"
        shutil.copy2(self.pcb_path, backup_path)
        print(f"Backup: {backup_path}")

        # Initial state
        initial_drc = self.run_drc()
        initial_total = len(initial_drc.get('violations', [])) + len(initial_drc.get('unconnected_items', []))
        print(f"\nInitial violations: {initial_total}")

        # Phase 1: Run pcbnew-based fixes first (more reliable than regex)
        print("\n[PRE-PROCESSING] Running pcbnew API fixes...")
        pcbnew_results = self.run_pcbnew_fixes()
        if pcbnew_results.get('success'):
            post_pcbnew_drc = self.run_drc()
            post_pcbnew_total = len(post_pcbnew_drc.get('violations', [])) + len(post_pcbnew_drc.get('unconnected_items', []))
            pcbnew_improvement = initial_total - post_pcbnew_total
            print(f"After pcbnew fixes: {post_pcbnew_total} violations ({pcbnew_improvement:+d})")
        else:
            print("pcbnew fixes skipped or failed")

        # Phase 2: Run LLM-guided iterations for remaining issues
        print("\n[LLM PHASE] Starting LLM-guided fixing...")

        # Run iterations
        no_improvement = 0
        for i in range(1, self.max_iterations + 1):
            result = self.run_iteration(i)

            if result['done']:
                print(f"\nTarget reached!")
                break

            if not result['improved']:
                no_improvement += 1
                if no_improvement >= 3:
                    print(f"\nConverged (no improvement for 3 iterations)")
                    break
            else:
                no_improvement = 0

        # Final results
        final_drc = self.run_drc()
        final_total = len(final_drc.get('violations', [])) + len(final_drc.get('unconnected_items', []))

        improvement = initial_total - final_total
        pct = (improvement / initial_total * 100) if initial_total > 0 else 0

        print(f"\n{'='*60}")
        print("FINAL RESULTS")
        print(f"{'='*60}")
        print(f"Initial: {initial_total}")
        print(f"Final: {final_total}")
        print(f"Improvement: {improvement} ({pct:.1f}%)")

        results = {
            'initial_violations': initial_total,
            'final_violations': final_total,
            'improvement': improvement,
            'improvement_pct': pct,
            'success': final_total <= self.target_violations
        }

        results_path = self.output_dir / 'llm_fixer_results.json'
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)

        return results


def main():
    parser = argparse.ArgumentParser(
        description='LLM-guided PCB DRC violation fixer using Claude Opus 4.5'
    )
    parser.add_argument('pcb_path', help='Path to KiCad PCB file')
    parser.add_argument('--target', '-t', type=int, default=0,
                       help='Target violation count')
    parser.add_argument('--max-iterations', '-n', type=int, default=20,
                       help='Maximum iterations')
    parser.add_argument('--api-key', help='OpenRouter API key (or set OPENROUTER_API_KEY)')

    args = parser.parse_args()

    fixer = LLMPCBFixer(
        args.pcb_path,
        api_key=args.api_key,
        target_violations=args.target,
        max_iterations=args.max_iterations
    )

    results = fixer.run()
    return 0 if results['success'] else 1


if __name__ == '__main__':
    sys.exit(main())
