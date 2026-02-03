"""
MAPO v2.1 Schematic - LLM-Guided Red Queen

Adversarial co-evolution for robust schematic designs.
Inspired by the Red Queen hypothesis - designs must constantly
improve just to maintain fitness against evolving challenges.

Author: Nexus EE Design Team
"""

import asyncio
import json
import logging
import random
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import httpx
import numpy as np

from ..core.schematic_state import SchematicState, SchematicSolution
from ..core.config import SchematicMAPOConfig, get_config

logger = logging.getLogger(__name__)


@dataclass
class AdversarialChallenge:
    """
    A challenge created by the Red Queen adversary.
    
    Represents a potential failure mode or stress test for schematics.
    """
    challenge_type: str  # e.g., "power_surge", "missing_connection", "thermal_stress"
    description: str
    severity: float  # 0-1
    target_component: Optional[str] = None
    target_net: Optional[str] = None
    test_conditions: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def difficulty(self) -> float:
        """Challenge difficulty for scoring."""
        return self.severity


@dataclass
class RedQueenResult:
    """Result from a Red Queen adversarial round."""
    challenges_created: int
    challenges_passed: int
    challenges_failed: int
    fitness_delta: float  # Change in fitness
    surviving_solutions: List[SchematicSolution] = field(default_factory=list)
    failed_challenges: List[Tuple[AdversarialChallenge, str]] = field(default_factory=list)


class LLMGuidedRedQueen:
    """
    LLM-Guided Red Queen for adversarial schematic evolution.
    
    Creates intelligent challenges that expose weaknesses in designs,
    forcing continuous improvement. The LLM generates domain-specific
    challenges based on electronics engineering knowledge.
    """
    
    # Challenge types with weights
    CHALLENGE_TYPES = [
        ("power_integrity", 0.25),     # Power rail issues
        ("signal_integrity", 0.20),    # Signal path issues
        ("thermal_stress", 0.15),      # Thermal management
        ("emc_compliance", 0.15),      # EMI/EMC issues
        ("component_failure", 0.10),   # What if component X fails?
        ("missing_protection", 0.10),  # Missing protection circuits
        ("manufacturing", 0.05),       # Assembly/manufacturing issues
    ]
    
    def __init__(self, config: Optional[SchematicMAPOConfig] = None):
        """Initialize Red Queen with configuration."""
        self.config = config or get_config()
        self.adversary_strength = self.config.red_queen_adversary_strength
        self.evolution_rate = self.config.red_queen_evolution_rate
        self._http_client: Optional[httpx.AsyncClient] = None
        
        # Track challenge history to avoid repetition
        self.challenge_history: List[str] = []
        self.successful_challenges: List[AdversarialChallenge] = []
    
    async def _ensure_http_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client."""
        if self._http_client is None or self._http_client.is_closed:
            self._http_client = httpx.AsyncClient(
                timeout=httpx.Timeout(self.config.llm_timeout)
            )
        return self._http_client
    
    async def close(self):
        """Close HTTP client."""
        if self._http_client and not self._http_client.is_closed:
            await self._http_client.aclose()
            self._http_client = None
    
    async def _call_llm(self, prompt: str, system: str = "") -> str:
        """Call LLM via OpenRouter."""
        if not self.config.openrouter_api_key:
            return ""
        
        client = await self._ensure_http_client()
        
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        
        try:
            response = await client.post(
                self.config.openrouter_base_url,
                headers={
                    "Authorization": f"Bearer {self.config.openrouter_api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": self.config.openrouter_model,
                    "messages": messages,
                    "temperature": 0.7,  # Higher for creative challenges
                    "max_tokens": 2048,
                }
            )
            
            if response.status_code == 200:
                data = response.json()
                return data["choices"][0]["message"]["content"]
            return ""
                
        except Exception as e:
            logger.warning(f"LLM call error: {e}")
            return ""
    
    def _select_challenge_type(self) -> str:
        """Select a challenge type based on weights."""
        types, weights = zip(*self.CHALLENGE_TYPES)
        return random.choices(types, weights=weights, k=1)[0]
    
    async def generate_challenges(
        self,
        solution: SchematicSolution,
        num_challenges: int = 3,
    ) -> List[AdversarialChallenge]:
        """
        Generate adversarial challenges for a solution.
        
        LLM analyzes the schematic and creates domain-specific
        challenges that test potential weaknesses.
        """
        state = solution.state
        
        # Build state summary
        components_by_category = {}
        for comp in state.components:
            cat = comp.category
            if cat not in components_by_category:
                components_by_category[cat] = []
            components_by_category[cat].append(f"{comp.reference}: {comp.part_number}")
        
        category_summary = "\n".join(
            f"- {cat}: {', '.join(comps[:5])}" + (f" (+{len(comps)-5} more)" if len(comps) > 5 else "")
            for cat, comps in components_by_category.items()
        )
        
        # Build connection summary
        power_connections = [c for c in state.connections if c.connection_type == "power"]
        signal_connections = [c for c in state.connections if c.connection_type == "signal"]
        
        # Select challenge types
        challenge_types = [self._select_challenge_type() for _ in range(num_challenges)]
        
        prompt = f"""
As a Red Queen adversary, generate {num_challenges} challenges to test this schematic design:

Components by Category:
{category_summary}

Power Connections: {len(power_connections)}
Signal Connections: {len(signal_connections)}
Wiring Completeness: {state.wiring_completeness:.1%}

Current fitness: {solution.fitness:.3f}

Challenge types to generate: {', '.join(challenge_types)}

Previous successful challenges (avoid similar):
{chr(10).join(f"- {c.description}" for c in self.successful_challenges[-5:])}

For each challenge, generate:
1. A specific, testable scenario
2. What could go wrong
3. How to test for this issue

Output as JSON array:
[
  {{
    "challenge_type": "{challenge_types[0]}",
    "description": "...",
    "severity": 0.0-1.0,
    "target_component": "U1" or null,
    "target_net": "VCC" or null,
    "test_conditions": {{"voltage": 5.5, "temperature": 85}}
  }}
]
"""
        
        response = await self._call_llm(
            prompt,
            system="You are a Red Queen adversary testing schematic designs. Create challenging but realistic failure scenarios that expose design weaknesses."
        )
        
        challenges = []
        
        try:
            # Parse JSON from response
            if "```json" in response:
                json_str = response.split("```json")[1].split("```")[0]
            elif "[" in response:
                start = response.index("[")
                end = response.rindex("]") + 1
                json_str = response[start:end]
            else:
                json_str = "[]"
            
            data = json.loads(json_str)
            
            for item in data[:num_challenges]:
                challenge = AdversarialChallenge(
                    challenge_type=item.get("challenge_type", "power_integrity"),
                    description=item.get("description", "Unknown challenge"),
                    severity=min(1.0, max(0.0, item.get("severity", 0.5))),
                    target_component=item.get("target_component"),
                    target_net=item.get("target_net"),
                    test_conditions=item.get("test_conditions", {}),
                )
                challenges.append(challenge)
                
        except Exception as e:
            logger.debug(f"Failed to parse challenges: {e}")
            # Generate default challenges
            for ct in challenge_types[:num_challenges]:
                challenges.append(AdversarialChallenge(
                    challenge_type=ct,
                    description=f"Default {ct} challenge",
                    severity=0.5 * self.adversary_strength,
                ))
        
        return challenges
    
    async def evaluate_challenge(
        self,
        solution: SchematicSolution,
        challenge: AdversarialChallenge,
    ) -> Tuple[bool, str]:
        """
        Evaluate if a solution passes a challenge.
        
        Returns (passed, reason).
        """
        state = solution.state
        
        # Build evaluation prompt
        prompt = f"""
Evaluate if this schematic design passes the following challenge:

Challenge: {challenge.description}
Type: {challenge.challenge_type}
Severity: {challenge.severity:.2f}
Target Component: {challenge.target_component or "N/A"}
Target Net: {challenge.target_net or "N/A"}
Test Conditions: {json.dumps(challenge.test_conditions)}

Schematic Summary:
- Components: {len(state.components)}
- Connections: {len(state.connections)}
- Wiring Completeness: {state.wiring_completeness:.1%}

Does the design adequately address this challenge?
Consider:
1. Are appropriate protection mechanisms in place?
2. Are connections properly made?
3. Would the circuit survive the test conditions?

Output as JSON:
{{"passed": true/false, "reason": "...", "confidence": 0.0-1.0}}
"""
        
        response = await self._call_llm(
            prompt,
            system="You are evaluating schematic designs against failure scenarios. Be critical but fair."
        )
        
        try:
            if "```json" in response:
                json_str = response.split("```json")[1].split("```")[0]
            elif "{" in response:
                start = response.index("{")
                end = response.rindex("}") + 1
                json_str = response[start:end]
            else:
                json_str = '{"passed": false, "reason": "Could not evaluate"}'
            
            data = json.loads(json_str)
            passed = data.get("passed", False)
            reason = data.get("reason", "Unknown")
            
            return passed, reason
            
        except Exception as e:
            logger.debug(f"Failed to parse evaluation: {e}")
            # Default to random based on adversary strength
            passed = random.random() > self.adversary_strength
            return passed, "Evaluation uncertain"
    
    async def run_adversarial_round(
        self,
        solutions: List[SchematicSolution],
        num_challenges: int = 3,
    ) -> RedQueenResult:
        """
        Run a Red Queen adversarial round against a set of solutions.
        
        Creates challenges, tests solutions, and returns results.
        """
        logger.info(f"Starting Red Queen round with {len(solutions)} solutions")
        
        # Generate challenges based on best solution
        best_solution = max(solutions, key=lambda s: s.fitness)
        challenges = await self.generate_challenges(best_solution, num_challenges)
        
        surviving = []
        failed = []
        
        # Test each solution against challenges
        for solution in solutions:
            solution_passed = True
            
            for challenge in challenges:
                passed, reason = await self.evaluate_challenge(solution, challenge)
                
                if not passed:
                    solution_passed = False
                    failed.append((challenge, f"{solution.state.uuid[:8]}: {reason}"))
                    
                    # Track successful challenges
                    if challenge not in self.successful_challenges:
                        self.successful_challenges.append(challenge)
            
            if solution_passed:
                surviving.append(solution)
        
        # Calculate fitness delta
        avg_fitness_before = np.mean([s.fitness for s in solutions])
        avg_fitness_after = np.mean([s.fitness for s in surviving]) if surviving else 0.0
        fitness_delta = avg_fitness_after - avg_fitness_before
        
        result = RedQueenResult(
            challenges_created=len(challenges),
            challenges_passed=len(surviving) * len(challenges),
            challenges_failed=len(failed),
            fitness_delta=fitness_delta,
            surviving_solutions=surviving,
            failed_challenges=failed,
        )
        
        logger.info(
            f"Red Queen round complete: {len(surviving)}/{len(solutions)} survived, "
            f"{len(failed)} challenge failures"
        )
        
        return result
    
    def evolve_adversary(self, round_results: List[RedQueenResult]):
        """
        Evolve the adversary based on round results.
        
        If too many solutions are surviving, increase adversary strength.
        If too few are surviving, decrease it.
        """
        if not round_results:
            return
        
        # Calculate survival rate
        total_solutions = sum(
            r.challenges_passed + r.challenges_failed 
            for r in round_results
        )
        total_passed = sum(r.challenges_passed for r in round_results)
        
        if total_solutions > 0:
            survival_rate = total_passed / total_solutions
            
            # Target survival rate around 50-70%
            if survival_rate > 0.7:
                # Too easy, increase strength
                self.adversary_strength = min(1.0, self.adversary_strength + self.evolution_rate)
                logger.info(f"Red Queen evolved: strength -> {self.adversary_strength:.2f}")
            elif survival_rate < 0.3:
                # Too hard, decrease strength
                self.adversary_strength = max(0.2, self.adversary_strength - self.evolution_rate)
                logger.info(f"Red Queen evolved: strength -> {self.adversary_strength:.2f}")
