#!/usr/bin/env python3
"""
Multi-Agent EE Validation with Expert Personas

Uses Claude Opus 4.5 with multiple expert personas for comprehensive review:

1. MIT EE Professor - Academic rigor, theoretical correctness
2. Senior PCB Designer - Manufacturing and DFM expertise
3. EMC/EMI Expert - Signal integrity and compliance
4. Reliability Engineer - Failure analysis and MTBF
5. Safety Judge - IEC/UL compliance and hazard analysis

Each persona reviews the design independently, then a synthesis agent
combines their feedback into actionable improvements.

This is the MULTI-AGENT AI-FIRST approach.
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field

try:
    import anthropic
    HAS_ANTHROPIC = True
except ImportError:
    HAS_ANTHROPIC = False


@dataclass
class ExpertPersona:
    """An expert persona for design review."""
    name: str
    title: str
    expertise: List[str]
    review_focus: str
    system_prompt: str


# Define expert personas (role-based, no fictional names)
EXPERT_PERSONAS = [
    ExpertPersona(
        name="Circuit Theory Expert",
        title="MIT EECS Professor Level",
        expertise=["Circuit theory", "Power electronics", "Control systems"],
        review_focus="Theoretical correctness and academic rigor",
        system_prompt="""You are a circuit theory expert with MIT EECS professor-level knowledge in power electronics and motor control systems.

Your review focuses on:
- Theoretical correctness of circuit topology
- Control loop stability (poles, zeros, phase margin)
- Power stage efficiency analysis
- Component stress calculations (voltage, current ratings)
- Thermal design adequacy
- Academic best practices

Apply rigorous engineering standards.
Cite specific equations and design guidelines when noting issues.
Be thorough but constructive with detailed technical feedback."""
    ),

    ExpertPersona(
        name="DFM Expert",
        title="Senior PCB Designer (20+ years experience)",
        expertise=["DFM", "High-speed layout", "Manufacturing"],
        review_focus="Manufacturability and production quality",
        system_prompt="""You are a senior PCB designer with 20+ years of industry experience at tier-1 electronics companies.

Your review focuses on:
- Design for Manufacturing (DFM) issues
- Trace width and spacing for production yield
- Via sizes and aspect ratios
- Solder mask and silkscreen quality
- Assembly considerations (component placement, orientation)
- Panelization and tooling
- Cost optimization

Apply knowledge from thousands of production boards.
Identify what causes yield issues and assembly problems.
Be practical and specific - cite actual manufacturing constraints."""
    ),

    ExpertPersona(
        name="EMC/EMI Expert",
        title="EMC Compliance Engineer",
        expertise=["Signal integrity", "EMC compliance", "High-speed design"],
        review_focus="Electromagnetic compatibility and signal quality",
        system_prompt="""You are an EMC compliance engineer with certification experience for FCC, CE, and automotive (CISPR 25).

Your review focuses on:
- Return path discontinuities
- Differential pair routing quality
- Ground plane splits and gaps
- Decoupling capacitor placement
- High dI/dt loop areas
- Antenna effects from traces
- Shield effectiveness
- Pre-compliance test predictions

Identify issues that cause EMC failures and expensive re-spins.
Be specific about frequencies and emission levels of concern."""
    ),

    ExpertPersona(
        name="Reliability Expert",
        title="Reliability Engineer (Aerospace/Automotive)",
        expertise=["FMEA", "MTBF analysis", "Failure prediction"],
        review_focus="Long-term reliability and failure modes",
        system_prompt="""You are a reliability engineer with aerospace and automotive systems experience.

Your review focuses on:
- Component derating analysis
- Thermal cycling stress
- Solder joint reliability
- Electromigration concerns
- Component wear-out mechanisms
- FMEA (Failure Mode and Effects Analysis)
- MTBF predictions
- Environmental stress factors

Analyze what happens after 10,000 hours of operation.
Identify components and design choices that will fail first.
Suggest redundancy and monitoring for critical functions."""
    ),

    ExpertPersona(
        name="Safety Compliance Expert",
        title="Safety Certification Judge (UL/TUV/CSA)",
        expertise=["IEC 62368", "UL certification", "Hazard analysis"],
        review_focus="Safety compliance and hazard mitigation",
        system_prompt="""You are a safety compliance expert with UL, TUV, and CSA certification body experience.

Your review focuses on:
- IEC 62368-1 compliance for power circuits
- Creepage and clearance distances
- Touch current limits
- Fault condition analysis
- Energy hazard classification
- Fire enclosure requirements
- Protective earthing
- Risk assessment per ISO 14971

Think like a certification body test engineer.
Identify issues that will fail safety testing.
Be specific about standards clauses and test requirements."""
    ),
]


def get_client():
    """Get Anthropic client."""
    api_key = os.environ.get('ANTHROPIC_API_KEY')
    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY not set")
    return anthropic.Anthropic(api_key=api_key)


def read_file(path: str, max_chars: int = 50000) -> str:
    """Read file with size limit."""
    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()
    if len(content) > max_chars:
        content = content[:max_chars] + f"\n... [TRUNCATED at {max_chars} chars] ..."
    return content


def get_expert_review(
    client,
    persona: ExpertPersona,
    schematic_content: str,
    pcb_content: Optional[str],
    design_name: str
) -> Dict[str, Any]:
    """Get review from a single expert persona."""

    files_section = f"SCHEMATIC:\n```\n{schematic_content[:20000]}\n```"
    if pcb_content:
        files_section += f"\n\nPCB:\n```\n{pcb_content[:20000]}\n```"

    user_prompt = f"""Review this electronics design: {design_name}

{files_section}

Provide your expert review covering:
1. CRITICAL ISSUES - Must fix before production
2. WARNINGS - Should fix for quality
3. RECOMMENDATIONS - Best practice improvements
4. POSITIVE ASPECTS - What's done well

Format as JSON with these four sections."""

    response = client.messages.create(
        model="claude-opus-4-5-20251101",
        max_tokens=4096,
        system=persona.system_prompt,
        messages=[{"role": "user", "content": user_prompt}]
    )

    response_text = response.content[0].text

    # Parse JSON from response
    try:
        json_start = response_text.find('{')
        json_end = response_text.rfind('}') + 1
        if json_start >= 0 and json_end > json_start:
            review = json.loads(response_text[json_start:json_end])
        else:
            review = {"raw_review": response_text}
    except json.JSONDecodeError:
        review = {"raw_review": response_text}

    return {
        "expert": persona.name,
        "title": persona.title,
        "focus": persona.review_focus,
        "review": review
    }


def synthesize_reviews(client, reviews: List[Dict], design_name: str) -> Dict[str, Any]:
    """Synthesize all expert reviews into actionable summary."""

    reviews_text = ""
    for r in reviews:
        reviews_text += f"\n\n=== {r['expert']} ({r['title']}) ===\n"
        reviews_text += f"Focus: {r['focus']}\n"
        reviews_text += json.dumps(r['review'], indent=2)

    system_prompt = """You are a chief engineer synthesizing expert design reviews.

Your task:
1. Identify consensus issues (multiple experts flagged)
2. Prioritize by severity and cost to fix
3. Create actionable fix list with specific steps
4. Note areas of disagreement between experts
5. Provide overall design quality score (1-10)

Be concise and actionable. Engineers need to know exactly what to fix."""

    user_prompt = f"""Synthesize these expert reviews for: {design_name}

{reviews_text}

Provide:
1. CONSENSUS_CRITICAL - Issues multiple experts flagged as critical
2. PRIORITIZED_FIXES - Ordered list of fixes with effort estimate
3. EXPERT_DISAGREEMENTS - Areas where experts differed
4. QUALITY_SCORE - Overall design score (1-10) with justification
5. NEXT_STEPS - Immediate actions to take

Format as JSON."""

    response = client.messages.create(
        model="claude-opus-4-5-20251101",
        max_tokens=4096,
        system=system_prompt,
        messages=[{"role": "user", "content": user_prompt}]
    )

    response_text = response.content[0].text

    try:
        json_start = response_text.find('{')
        json_end = response_text.rfind('}') + 1
        if json_start >= 0 and json_end > json_start:
            synthesis = json.loads(response_text[json_start:json_end])
        else:
            synthesis = {"raw_synthesis": response_text}
    except json.JSONDecodeError:
        synthesis = {"raw_synthesis": response_text}

    return synthesis


def run_multi_agent_validation(
    schematic_path: str,
    pcb_path: Optional[str] = None,
    personas: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Run full multi-agent validation.

    Args:
        schematic_path: Path to .kicad_sch file
        pcb_path: Optional path to .kicad_pcb file
        personas: Optional list of persona names to use (default: all)

    Returns:
        Complete validation results with all reviews and synthesis
    """
    client = get_client()

    # Read files
    schematic_content = read_file(schematic_path)
    pcb_content = read_file(pcb_path) if pcb_path else None
    design_name = Path(schematic_path).stem

    # Select personas
    if personas:
        selected_personas = [p for p in EXPERT_PERSONAS if p.name in personas]
    else:
        selected_personas = EXPERT_PERSONAS

    print(f"\nRunning multi-agent validation with {len(selected_personas)} experts...")

    # Get reviews from each expert
    reviews = []
    for persona in selected_personas:
        print(f"  Getting review from {persona.name} ({persona.title})...")
        review = get_expert_review(client, persona, schematic_content, pcb_content, design_name)
        reviews.append(review)

    # Synthesize all reviews
    print("  Synthesizing expert reviews...")
    synthesis = synthesize_reviews(client, reviews, design_name)

    return {
        "design": design_name,
        "schematic": schematic_path,
        "pcb": pcb_path,
        "expert_count": len(reviews),
        "reviews": reviews,
        "synthesis": synthesis
    }


def run_fallback_validation(schematic_path: str, pcb_path: Optional[str]) -> Dict[str, Any]:
    """Fallback when API is not available."""
    return {
        "status": "FALLBACK_MODE",
        "message": "ANTHROPIC_API_KEY not set. Install anthropic SDK and set API key for multi-agent validation.",
        "schematic": schematic_path,
        "pcb": pcb_path,
        "instructions": [
            "1. pip install anthropic",
            "2. export ANTHROPIC_API_KEY=your-key",
            "3. Re-run this script"
        ]
    }


def main():
    parser = argparse.ArgumentParser(
        description='Multi-Agent EE Validation with Expert Personas'
    )
    parser.add_argument(
        '--schematic', '-s',
        type=str,
        required=True,
        help='Path to .kicad_sch file'
    )
    parser.add_argument(
        '--pcb', '-p',
        type=str,
        help='Path to .kicad_pcb file'
    )
    parser.add_argument(
        '--experts',
        type=str,
        nargs='+',
        help='Specific experts to consult (default: all)'
    )
    parser.add_argument(
        '--json',
        action='store_true',
        help='Output as JSON'
    )
    parser.add_argument(
        '--list-experts',
        action='store_true',
        help='List available expert personas'
    )

    args = parser.parse_args()

    if args.list_experts:
        print("\nAvailable Expert Personas:")
        print("=" * 60)
        for p in EXPERT_PERSONAS:
            print(f"\n{p.name}")
            print(f"  Title: {p.title}")
            print(f"  Expertise: {', '.join(p.expertise)}")
            print(f"  Focus: {p.review_focus}")
        return

    if not HAS_ANTHROPIC or not os.environ.get('ANTHROPIC_API_KEY'):
        result = run_fallback_validation(args.schematic, args.pcb)
    else:
        try:
            result = run_multi_agent_validation(args.schematic, args.pcb, args.experts)
        except Exception as e:
            result = {"error": str(e)}

    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print("\n" + "=" * 60)
        print("MULTI-AGENT VALIDATION RESULTS")
        print("=" * 60)

        if "error" in result:
            print(f"\nERROR: {result['error']}")
            return

        if "status" in result and result["status"] == "FALLBACK_MODE":
            print(f"\n{result['message']}")
            for instruction in result.get('instructions', []):
                print(f"  {instruction}")
            return

        print(f"\nDesign: {result['design']}")
        print(f"Experts consulted: {result['expert_count']}")

        print("\n" + "-" * 60)
        print("EXPERT REVIEWS")
        print("-" * 60)

        for review in result['reviews']:
            print(f"\n{review['expert']} ({review['title']})")
            print(f"Focus: {review['focus']}")
            if isinstance(review['review'], dict):
                for key, value in review['review'].items():
                    if key.startswith('raw'):
                        continue
                    print(f"\n  {key}:")
                    if isinstance(value, list):
                        for item in value[:5]:  # Limit output
                            print(f"    - {item}")
                    else:
                        print(f"    {str(value)[:200]}")

        print("\n" + "-" * 60)
        print("SYNTHESIS")
        print("-" * 60)

        synthesis = result.get('synthesis', {})
        if isinstance(synthesis, dict):
            for key, value in synthesis.items():
                if key.startswith('raw'):
                    print(value)
                    continue
                print(f"\n{key}:")
                if isinstance(value, list):
                    for item in value:
                        print(f"  - {item}")
                elif isinstance(value, dict):
                    for k, v in value.items():
                        print(f"  {k}: {v}")
                else:
                    print(f"  {value}")


if __name__ == '__main__':
    main()
