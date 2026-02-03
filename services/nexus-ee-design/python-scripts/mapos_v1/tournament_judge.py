#!/usr/bin/env python3
"""
Tournament Judge - Pairwise comparison and Elo ranking for PCB configurations.

This module implements:
1. Pairwise comparison using LLM or heuristics
2. Elo rating system for configuration ranking
3. Tournament bracket for elimination-style selection
4. Multi-criteria judging (DRC, SI, thermal, DFM)

Inspired by Constitutional AI's debate mechanism and competitive self-play.
"""

import os
import sys
import json
import asyncio
import random
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any
from enum import Enum, auto
from pathlib import Path

# Add script directory to path for local imports
SCRIPT_DIR = Path(__file__).parent.resolve()
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from pcb_state import PCBState, DRCResult


class JudgingCriteria(Enum):
    """Criteria for judging configurations."""
    DRC_VIOLATIONS = auto()      # Total DRC violation count
    ERROR_COUNT = auto()         # Serious error count
    UNCONNECTED = auto()         # Unconnected items
    SIGNAL_INTEGRITY = auto()    # SI score (clearance, length matching)
    THERMAL = auto()             # Thermal score (via density, plane coverage)
    MANUFACTURABILITY = auto()   # DFM score (drill sizes, annular ring)
    COST = auto()                # Estimated cost (via count, layer usage)


@dataclass
class JudgingResult:
    """Result of judging a match between configurations."""
    config_a_id: str
    config_b_id: str
    winner_id: str
    confidence: float
    reasoning: str
    criteria_scores: Dict[str, Tuple[float, float]] = field(default_factory=dict)


@dataclass
class EloRating:
    """Elo rating for a configuration."""
    config_id: str
    rating: float = 1200.0
    matches: int = 0
    wins: int = 0
    losses: int = 0

    @property
    def win_rate(self) -> float:
        if self.matches == 0:
            return 0.5
        return self.wins / self.matches


class TournamentJudge:
    """
    Judge for comparing PCB configurations.

    Uses pairwise comparison with LLM guidance or heuristics
    to rank configurations and select the best.
    """

    def __init__(
        self,
        k_factor: float = 32,
        use_llm: bool = False,
        model: str = "claude-opus-4-5-20251101"
    ):
        """
        Initialize tournament judge.

        Args:
            k_factor: Elo K-factor (higher = more volatile ratings)
            use_llm: Whether to use LLM for judging
            model: LLM model to use if use_llm is True
        """
        self.k_factor = k_factor
        self.use_llm = use_llm
        self.model = model

        # Elo ratings for configurations
        self.ratings: Dict[str, EloRating] = {}

        # Match history
        self.match_history: List[JudgingResult] = []

        # Criteria weights (sum to 1.0)
        self.criteria_weights = {
            JudgingCriteria.DRC_VIOLATIONS: 0.35,
            JudgingCriteria.ERROR_COUNT: 0.20,
            JudgingCriteria.UNCONNECTED: 0.15,
            JudgingCriteria.SIGNAL_INTEGRITY: 0.10,
            JudgingCriteria.THERMAL: 0.10,
            JudgingCriteria.MANUFACTURABILITY: 0.10,
        }

    def _get_or_create_rating(self, config_id: str) -> EloRating:
        """Get or create Elo rating for a configuration."""
        if config_id not in self.ratings:
            self.ratings[config_id] = EloRating(config_id=config_id)
        return self.ratings[config_id]

    def _calculate_expected_score(self, rating_a: float, rating_b: float) -> float:
        """Calculate expected score for player A against player B."""
        return 1.0 / (1.0 + 10 ** ((rating_b - rating_a) / 400))

    def _update_elo(
        self,
        winner_id: str,
        loser_id: str
    ) -> Tuple[float, float]:
        """
        Update Elo ratings after a match.

        Args:
            winner_id: ID of winning configuration
            loser_id: ID of losing configuration

        Returns:
            Tuple of (new_winner_rating, new_loser_rating)
        """
        winner = self._get_or_create_rating(winner_id)
        loser = self._get_or_create_rating(loser_id)

        # Calculate expected scores
        expected_winner = self._calculate_expected_score(winner.rating, loser.rating)
        expected_loser = 1.0 - expected_winner

        # Update ratings
        winner.rating += self.k_factor * (1.0 - expected_winner)
        loser.rating += self.k_factor * (0.0 - expected_loser)

        # Update match counts
        winner.matches += 1
        winner.wins += 1
        loser.matches += 1
        loser.losses += 1

        return winner.rating, loser.rating

    def _evaluate_criteria(
        self,
        state: PCBState,
        drc: DRCResult
    ) -> Dict[JudgingCriteria, float]:
        """
        Evaluate a configuration on all criteria.

        Returns scores normalized to 0-1 (higher is better).
        """
        scores = {}

        # DRC violations (lower is better, normalize inversely)
        scores[JudgingCriteria.DRC_VIOLATIONS] = 1.0 / (1.0 + drc.total_violations / 100)

        # Error count (lower is better)
        scores[JudgingCriteria.ERROR_COUNT] = 1.0 / (1.0 + drc.errors / 50)

        # Unconnected items (lower is better)
        scores[JudgingCriteria.UNCONNECTED] = 1.0 / (1.0 + drc.unconnected)

        # Signal integrity (based on clearance violations)
        clearance_violations = drc.violations_by_type.get('clearance', 0)
        scores[JudgingCriteria.SIGNAL_INTEGRITY] = 1.0 / (1.0 + clearance_violations / 50)

        # Thermal (based on via count and zones)
        via_count = len(state.vias)
        zone_count = len(state.zones)
        thermal_base = min(1.0, (via_count + zone_count * 10) / 200)
        scores[JudgingCriteria.THERMAL] = thermal_base

        # Manufacturability (based on DFM violations)
        dfm_types = ['solder_mask_bridge', 'silk_over_copper', 'courtyards_overlap']
        dfm_violations = sum(drc.violations_by_type.get(t, 0) for t in dfm_types)
        scores[JudgingCriteria.MANUFACTURABILITY] = 1.0 / (1.0 + dfm_violations / 50)

        return scores

    def _calculate_weighted_score(
        self,
        criteria_scores: Dict[JudgingCriteria, float]
    ) -> float:
        """Calculate weighted total score from criteria scores."""
        total = 0.0
        for criteria, score in criteria_scores.items():
            weight = self.criteria_weights.get(criteria, 0.0)
            total += weight * score
        return total

    async def judge_match(
        self,
        config_a: PCBState,
        config_b: PCBState
    ) -> JudgingResult:
        """
        Judge a match between two configurations.

        Args:
            config_a: First configuration
            config_b: Second configuration

        Returns:
            JudgingResult with winner and reasoning
        """
        # Get DRC results
        drc_a = config_a.run_drc()
        drc_b = config_b.run_drc()

        # Evaluate criteria
        scores_a = self._evaluate_criteria(config_a, drc_a)
        scores_b = self._evaluate_criteria(config_b, drc_b)

        # Calculate total scores
        total_a = self._calculate_weighted_score(scores_a)
        total_b = self._calculate_weighted_score(scores_b)

        # Determine winner
        if self.use_llm:
            winner_id, confidence, reasoning = await self._llm_judge(
                config_a, config_b, drc_a, drc_b, scores_a, scores_b
            )
        else:
            # Heuristic judging
            if total_a > total_b:
                winner_id = config_a.state_id
                margin = total_a - total_b
            else:
                winner_id = config_b.state_id
                margin = total_b - total_a

            confidence = min(0.95, 0.5 + margin)
            reasoning = self._generate_heuristic_reasoning(
                config_a, config_b, drc_a, drc_b, scores_a, scores_b
            )

        # Create criteria scores dict for result
        criteria_scores = {
            c.name: (scores_a.get(c, 0), scores_b.get(c, 0))
            for c in JudgingCriteria
        }

        result = JudgingResult(
            config_a_id=config_a.state_id,
            config_b_id=config_b.state_id,
            winner_id=winner_id,
            confidence=confidence,
            reasoning=reasoning,
            criteria_scores=criteria_scores
        )

        # Update Elo ratings
        loser_id = config_b.state_id if winner_id == config_a.state_id else config_a.state_id
        self._update_elo(winner_id, loser_id)

        # Record match
        self.match_history.append(result)

        return result

    async def _llm_judge(
        self,
        config_a: PCBState,
        config_b: PCBState,
        drc_a: DRCResult,
        drc_b: DRCResult,
        scores_a: Dict,
        scores_b: Dict
    ) -> Tuple[str, float, str]:
        """Use LLM to judge between configurations."""
        prompt = f"""Compare these two PCB layout configurations:

Configuration A (ID: {config_a.state_id}):
- DRC Violations: {drc_a.total_violations}
- Errors: {drc_a.errors}
- Unconnected: {drc_a.unconnected}
- Components: {len(config_a.components)}
- Vias: {len(config_a.vias)}

Criteria Scores (0-1, higher is better):
{json.dumps({c.name: round(s, 3) for c, s in scores_a.items()}, indent=2)}

Configuration B (ID: {config_b.state_id}):
- DRC Violations: {drc_b.total_violations}
- Errors: {drc_b.errors}
- Unconnected: {drc_b.unconnected}
- Components: {len(config_b.components)}
- Vias: {len(config_b.vias)}

Criteria Scores (0-1, higher is better):
{json.dumps({c.name: round(s, 3) for c, s in scores_b.items()}, indent=2)}

Which configuration is better overall? Consider:
1. Fewer DRC violations is critical (weight: 35%)
2. Fewer errors is important (weight: 20%)
3. All items connected is important (weight: 15%)
4. Signal integrity (weight: 10%)
5. Thermal management (weight: 10%)
6. Manufacturability (weight: 10%)

Return JSON:
{{
  "winner": "A" | "B",
  "confidence": 0.0-1.0,
  "reasoning": "Detailed explanation"
}}
"""

        try:
            api_key = os.environ.get('ANTHROPIC_API_KEY')
            if api_key:
                import anthropic
                client = anthropic.Anthropic(api_key=api_key)
                message = client.messages.create(
                    model=self.model,
                    max_tokens=1024,
                    messages=[{"role": "user", "content": prompt}]
                )
                response = message.content[0].text

                # Parse response
                import re
                json_match = re.search(r'\{[\s\S]*\}', response)
                if json_match:
                    data = json.loads(json_match.group())
                    winner = config_a.state_id if data['winner'] == 'A' else config_b.state_id
                    return winner, data['confidence'], data['reasoning']
        except Exception as e:
            print(f"LLM judging failed: {e}")

        # Fallback to heuristic
        total_a = self._calculate_weighted_score(scores_a)
        total_b = self._calculate_weighted_score(scores_b)
        winner_id = config_a.state_id if total_a > total_b else config_b.state_id
        return winner_id, 0.6, "Fallback heuristic judgment"

    def _generate_heuristic_reasoning(
        self,
        config_a: PCBState,
        config_b: PCBState,
        drc_a: DRCResult,
        drc_b: DRCResult,
        scores_a: Dict,
        scores_b: Dict
    ) -> str:
        """Generate reasoning for heuristic judgment."""
        total_a = self._calculate_weighted_score(scores_a)
        total_b = self._calculate_weighted_score(scores_b)

        winner = "A" if total_a > total_b else "B"
        winner_drc = drc_a if winner == "A" else drc_b
        loser_drc = drc_b if winner == "A" else drc_a

        reasons = []

        # DRC comparison
        if winner_drc.total_violations < loser_drc.total_violations:
            diff = loser_drc.total_violations - winner_drc.total_violations
            reasons.append(f"{diff} fewer DRC violations")

        # Error comparison
        if winner_drc.errors < loser_drc.errors:
            diff = loser_drc.errors - winner_drc.errors
            reasons.append(f"{diff} fewer errors")

        # Unconnected comparison
        if winner_drc.unconnected < loser_drc.unconnected:
            diff = loser_drc.unconnected - winner_drc.unconnected
            reasons.append(f"{diff} fewer unconnected items")

        if reasons:
            return f"Configuration {winner} wins with: {', '.join(reasons)}"
        return f"Configuration {winner} has marginally better overall score ({total_a:.3f} vs {total_b:.3f})"

    def get_rankings(self) -> List[Tuple[str, float]]:
        """Get configurations ranked by Elo rating."""
        sorted_ratings = sorted(
            self.ratings.values(),
            key=lambda r: r.rating,
            reverse=True
        )
        return [(r.config_id, r.rating) for r in sorted_ratings]

    def get_top_configs(self, n: int = 5) -> List[str]:
        """Get top N configuration IDs by rating."""
        rankings = self.get_rankings()
        return [config_id for config_id, _ in rankings[:n]]


class Tournament:
    """
    Tournament bracket for elimination-style selection.

    Runs single-elimination tournament to find best configuration.
    """

    def __init__(
        self,
        judge: TournamentJudge,
        configs: List[PCBState],
        rounds: Optional[int] = None
    ):
        """
        Initialize tournament.

        Args:
            judge: TournamentJudge instance
            configs: List of configurations to compete
            rounds: Number of rounds (auto-calculated if None)
        """
        self.judge = judge
        self.configs = list(configs)
        self.rounds = rounds or self._calculate_rounds(len(configs))

        # Tournament bracket
        self.bracket: List[List[Optional[PCBState]]] = []
        self.results: List[List[JudgingResult]] = []

    def _calculate_rounds(self, n: int) -> int:
        """Calculate number of rounds needed for n competitors."""
        import math
        if n <= 1:
            return 0
        return math.ceil(math.log2(n))

    async def run(self) -> PCBState:
        """
        Run the tournament.

        Returns:
            Winning configuration
        """
        print(f"\nRunning tournament with {len(self.configs)} configurations")

        # Initialize bracket with shuffled configs
        random.shuffle(self.configs)
        current_round = list(self.configs)
        self.bracket.append(current_round)

        round_num = 1
        while len(current_round) > 1:
            print(f"\n  Round {round_num}: {len(current_round)} competitors")

            next_round = []
            round_results = []

            # Pair up configs for matches
            for i in range(0, len(current_round), 2):
                if i + 1 < len(current_round):
                    config_a = current_round[i]
                    config_b = current_round[i + 1]

                    result = await self.judge.judge_match(config_a, config_b)
                    round_results.append(result)

                    # Winner advances
                    winner = config_a if result.winner_id == config_a.state_id else config_b
                    next_round.append(winner)

                    drc_a = config_a._drc_result
                    drc_b = config_b._drc_result
                    print(f"    Match: {config_a.state_id[:8]} ({drc_a.total_violations}) vs "
                          f"{config_b.state_id[:8]} ({drc_b.total_violations}) -> "
                          f"Winner: {winner.state_id[:8]}")
                else:
                    # Odd one out gets bye
                    next_round.append(current_round[i])
                    print(f"    Bye: {current_round[i].state_id[:8]}")

            self.bracket.append(next_round)
            self.results.append(round_results)
            current_round = next_round
            round_num += 1

        winner = current_round[0]
        print(f"\n  Tournament winner: {winner.state_id}")
        print(f"  Final violations: {winner._drc_result.total_violations}")

        return winner

    def get_bracket_summary(self) -> Dict:
        """Get summary of tournament bracket."""
        return {
            'total_rounds': len(self.bracket) - 1,
            'total_matches': sum(len(r) for r in self.results),
            'bracket': [
                [c.state_id[:8] for c in round_configs if c is not None]
                for round_configs in self.bracket
            ]
        }


class SwissTournament:
    """
    Swiss-system tournament for more matches with fewer rounds.

    Each round pairs configs with similar records.
    """

    def __init__(
        self,
        judge: TournamentJudge,
        configs: List[PCBState],
        num_rounds: int = 5
    ):
        self.judge = judge
        self.configs = list(configs)
        self.num_rounds = num_rounds

        # Track wins/losses
        self.records: Dict[str, Tuple[int, int]] = {
            c.state_id: (0, 0) for c in configs
        }

    async def run(self) -> List[PCBState]:
        """
        Run Swiss tournament.

        Returns:
            Configurations sorted by final record
        """
        print(f"\nRunning Swiss tournament: {len(self.configs)} configs, {self.num_rounds} rounds")

        for round_num in range(self.num_rounds):
            print(f"\n  Round {round_num + 1}:")

            # Sort by record (wins - losses)
            sorted_configs = sorted(
                self.configs,
                key=lambda c: self.records[c.state_id][0] - self.records[c.state_id][1],
                reverse=True
            )

            # Pair adjacent configs
            for i in range(0, len(sorted_configs) - 1, 2):
                config_a = sorted_configs[i]
                config_b = sorted_configs[i + 1]

                result = await self.judge.judge_match(config_a, config_b)

                # Update records
                winner_id = result.winner_id
                loser_id = config_b.state_id if winner_id == config_a.state_id else config_a.state_id

                w_wins, w_losses = self.records[winner_id]
                self.records[winner_id] = (w_wins + 1, w_losses)

                l_wins, l_losses = self.records[loser_id]
                self.records[loser_id] = (l_wins, l_losses + 1)

        # Sort by final record
        final_ranking = sorted(
            self.configs,
            key=lambda c: (
                self.records[c.state_id][0],  # wins
                -self.records[c.state_id][1]  # -losses
            ),
            reverse=True
        )

        print(f"\n  Final standings:")
        for i, config in enumerate(final_ranking[:5]):
            wins, losses = self.records[config.state_id]
            drc = config._drc_result
            print(f"    {i+1}. {config.state_id[:8]}: {wins}-{losses} ({drc.total_violations} violations)")

        return final_ranking


async def run_tournament(
    pcb_states: List[PCBState],
    tournament_type: str = "elimination",
    output_json: Optional[str] = None
) -> PCBState:
    """
    Run a tournament to select the best configuration.

    Args:
        pcb_states: List of PCB configurations
        tournament_type: "elimination" or "swiss"
        output_json: Optional path to save results

    Returns:
        Best configuration
    """
    judge = TournamentJudge(k_factor=32, use_llm=False)

    if tournament_type == "swiss":
        tournament = SwissTournament(judge, pcb_states, num_rounds=5)
        rankings = await tournament.run()
        winner = rankings[0]
    else:
        tournament = Tournament(judge, pcb_states)
        winner = await tournament.run()

    print(f"\nFinal Elo Rankings:")
    for config_id, rating in judge.get_rankings()[:10]:
        print(f"  {config_id[:8]}: {rating:.0f}")

    if output_json:
        results = {
            'winner_id': winner.state_id,
            'winner_violations': winner._drc_result.total_violations,
            'elo_rankings': [
                {'id': cid, 'rating': rating}
                for cid, rating in judge.get_rankings()
            ],
            'match_count': len(judge.match_history)
        }
        with open(output_json, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nSaved results to: {output_json}")

    return winner


if __name__ == '__main__':
    import sys
    from pcb_state import PCBState, create_random_modification

    if len(sys.argv) < 2:
        print("Usage: python tournament_judge.py <path_to.kicad_pcb> [num_configs]")
        sys.exit(1)

    pcb_path = sys.argv[1]
    num_configs = int(sys.argv[2]) if len(sys.argv) > 2 else 8

    print(f"\n{'='*60}")
    print("TOURNAMENT JUDGE - PCB Configuration Selection")
    print(f"{'='*60}")

    # Create test configurations
    print(f"\nCreating {num_configs} test configurations...")
    base_state = PCBState.from_file(pcb_path)
    base_drc = base_state.run_drc()
    print(f"Base violations: {base_drc.total_violations}")

    configs = [base_state]
    for i in range(1, num_configs):
        state = base_state.copy()
        for _ in range(random.randint(1, 3)):
            mod = create_random_modification(state)
            state = state.apply_modification(mod)
        state.run_drc()
        configs.append(state)
        print(f"  Config {i}: {state._drc_result.total_violations} violations")

    # Run tournament
    asyncio.run(run_tournament(
        configs,
        tournament_type="elimination",
        output_json=f"{pcb_path.rsplit('.', 1)[0]}_tournament_results.json"
    ))
