#!/usr/bin/env python3
"""
Debate and Critique Coordinator - Multi-agent debate mechanism.

This module orchestrates multi-agent debate for routing decisions,
inspired by CircuitLM research showing that agent debate reduces errors.

Multiple Opus 4.6 agents debate proposals before execution:
1. Proposer Agent suggests modification
2. Critic Agents identify potential issues
3. Refinement round addresses concerns
4. Consensus reached or escalated

Part of the MAPO v2.0 Enhancement: "Opus 4.6 Thinks, Algorithms Execute"
"""

import json
import os
import sys
import asyncio
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any, Callable, Awaitable
from enum import Enum, auto
from pathlib import Path
import time

# Add parent directory to path for local imports
SCRIPT_DIR = Path(__file__).parent.resolve()
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))


class DebateRole(Enum):
    """Roles in the debate process."""
    PROPOSER = auto()    # Agent making the proposal
    CRITIC = auto()      # Agent critiquing the proposal
    MEDIATOR = auto()    # Agent synthesizing critiques
    VOTER = auto()       # Agent voting on acceptance


class CritiqueSeverity(Enum):
    """Severity of critique issues."""
    SUGGESTION = auto()  # Nice to have, not blocking
    CONCERN = auto()     # Should be addressed
    OBJECTION = auto()   # Must be addressed
    BLOCKING = auto()    # Proposal cannot proceed


@dataclass
class Proposal:
    """A proposal to be debated."""
    proposal_id: str
    proposer_name: str
    proposal_type: str  # e.g., "routing_modification", "layer_assignment"
    content: Dict[str, Any]
    reasoning: str
    confidence: float
    timestamp: float = field(default_factory=time.time)


@dataclass
class Critique:
    """A critique of a proposal."""
    critic_name: str
    proposal_id: str
    issues: List[str]
    severity: CritiqueSeverity
    suggestions: List[str]
    approval_vote: bool
    reasoning: str


@dataclass
class Response:
    """Proposer's response to critiques."""
    proposal_id: str
    addressed_issues: List[str]
    unaddressed_issues: List[str]
    modifications: Dict[str, Any]
    revised_confidence: float
    reasoning: str


@dataclass
class Vote:
    """An agent's vote on a proposal."""
    voter_name: str
    proposal_id: str
    vote: bool  # True = accept, False = reject
    reasoning: str
    confidence: float


@dataclass
class DebateOutcome:
    """Outcome of a debate round."""
    proposal: Proposal
    critiques: List[Critique]
    response: Optional[Response]
    votes: List[Vote]
    accepted: bool
    consensus_score: float  # 0-1, how much agreement
    rounds_taken: int
    reasoning: str


@dataclass
class ConsensusResult:
    """Result of multi-round debate."""
    accepted_proposals: List[Proposal]
    rejected_proposals: List[Proposal]
    debates: List[DebateOutcome]
    total_rounds: int
    final_consensus_score: float
    escalated_to_human: bool
    reasoning: str


class DebateAndCritiqueCoordinator:
    """
    Orchestrates multi-agent debate for routing decisions.

    Process:
    1. Proposer Agent suggests modification
    2. Critic Agents identify potential issues
    3. Refinement round addresses concerns
    4. Consensus reached or escalated

    Reduces single-agent blind spots through structured debate.
    """

    def __init__(
        self,
        agents: List[Any],  # List of GeneratorAgent-like objects
        max_rounds: int = 3,
        consensus_threshold: float = 0.6,
        require_unanimous: bool = False
    ):
        """
        Initialize the debate coordinator.

        Args:
            agents: List of agents that can participate in debates
            max_rounds: Maximum debate rounds before escalation
            consensus_threshold: Fraction of votes needed to accept (0-1)
            require_unanimous: If True, all agents must agree
        """
        self.agents = agents
        self.max_rounds = max_rounds
        self.consensus_threshold = consensus_threshold
        self.require_unanimous = require_unanimous
        self.debate_history: List[DebateOutcome] = []

    async def debate_proposal(
        self,
        proposal: Proposal,
        context: Optional[Dict[str, Any]] = None
    ) -> DebateOutcome:
        """
        Run a single debate round on a proposal.

        Args:
            proposal: The proposal to debate
            context: Optional context for the debate

        Returns:
            DebateOutcome with critiques, response, votes, and acceptance
        """
        # Phase 1: Collect critiques from all agents (parallel)
        critique_tasks = [
            self._get_critique(agent, proposal, context)
            for agent in self.agents
            if agent.name != proposal.proposer_name
        ]

        critiques = await asyncio.gather(*critique_tasks, return_exceptions=True)
        critiques = [c for c in critiques if isinstance(c, Critique)]

        # Phase 2: Proposer responds to critiques
        proposer = self._get_agent_by_name(proposal.proposer_name)
        if proposer and critiques:
            response = await self._get_response(proposer, proposal, critiques, context)
        else:
            response = None

        # Phase 3: Collect votes
        vote_tasks = [
            self._get_vote(agent, proposal, critiques, response, context)
            for agent in self.agents
        ]

        votes = await asyncio.gather(*vote_tasks, return_exceptions=True)
        votes = [v for v in votes if isinstance(v, Vote)]

        # Phase 4: Determine acceptance
        accept_votes = sum(1 for v in votes if v.vote)
        total_votes = len(votes)

        if self.require_unanimous:
            accepted = accept_votes == total_votes
        else:
            accepted = (accept_votes / max(1, total_votes)) >= self.consensus_threshold

        consensus_score = accept_votes / max(1, total_votes)

        outcome = DebateOutcome(
            proposal=proposal,
            critiques=critiques,
            response=response,
            votes=votes,
            accepted=accepted,
            consensus_score=consensus_score,
            rounds_taken=1,
            reasoning=self._summarize_debate(proposal, critiques, votes, accepted)
        )

        self.debate_history.append(outcome)
        return outcome

    async def reach_consensus(
        self,
        topic: str,
        proposals: List[Proposal],
        context: Optional[Dict[str, Any]] = None
    ) -> ConsensusResult:
        """
        Multi-round debate until consensus or max rounds.

        Args:
            topic: The topic being debated
            proposals: List of proposals to consider
            context: Optional context

        Returns:
            ConsensusResult with accepted proposals and debate history
        """
        accepted_proposals = []
        rejected_proposals = []
        all_debates = []

        for round_num in range(self.max_rounds):
            # Debate each remaining proposal
            remaining_proposals = [
                p for p in proposals
                if p not in accepted_proposals and p not in rejected_proposals
            ]

            if not remaining_proposals:
                break

            debate_tasks = [
                self.debate_proposal(p, context)
                for p in remaining_proposals
            ]

            debates = await asyncio.gather(*debate_tasks, return_exceptions=True)
            debates = [d for d in debates if isinstance(d, DebateOutcome)]

            all_debates.extend(debates)

            # Categorize outcomes
            for debate in debates:
                if debate.accepted:
                    accepted_proposals.append(debate.proposal)
                elif debate.consensus_score < 0.3:
                    # Strong rejection
                    rejected_proposals.append(debate.proposal)
                # Else: keep for next round (borderline cases)

            # Check if we have enough accepted proposals
            if accepted_proposals:
                break

        # Calculate final consensus
        all_scores = [d.consensus_score for d in all_debates]
        final_score = sum(all_scores) / max(1, len(all_scores))

        # Determine if escalation is needed
        escalated = (
            not accepted_proposals and
            len(rejected_proposals) < len(proposals)
        )

        return ConsensusResult(
            accepted_proposals=accepted_proposals,
            rejected_proposals=rejected_proposals,
            debates=all_debates,
            total_rounds=min(round_num + 1, self.max_rounds),
            final_consensus_score=final_score,
            escalated_to_human=escalated,
            reasoning=self._summarize_consensus(accepted_proposals, all_debates)
        )

    async def _get_critique(
        self,
        agent: Any,
        proposal: Proposal,
        context: Optional[Dict]
    ) -> Critique:
        """Get critique from an agent."""
        try:
            # Check if agent has critique method
            if hasattr(agent, 'critique'):
                result = await agent.critique(proposal.content)
            else:
                # Use heuristic critique
                result = self._heuristic_critique(agent, proposal)

            if isinstance(result, dict):
                return Critique(
                    critic_name=agent.name,
                    proposal_id=proposal.proposal_id,
                    issues=result.get('issues', []),
                    severity=self._parse_severity(result.get('severity', 'CONCERN')),
                    suggestions=result.get('suggestions', []),
                    approval_vote=result.get('approval_vote', True),
                    reasoning=result.get('reasoning', '')
                )
            elif isinstance(result, Critique):
                return result
            else:
                return self._default_critique(agent.name, proposal)

        except Exception as e:
            print(f"Critique from {agent.name} failed: {e}")
            return self._default_critique(agent.name, proposal)

    async def _get_response(
        self,
        proposer: Any,
        proposal: Proposal,
        critiques: List[Critique],
        context: Optional[Dict]
    ) -> Response:
        """Get proposer's response to critiques."""
        try:
            # Synthesize critiques
            all_issues = []
            for c in critiques:
                all_issues.extend(c.issues)

            # Check if agent can respond
            if hasattr(proposer, 'respond_to_critique'):
                critique_summary = {
                    'issues': all_issues,
                    'suggestions': [s for c in critiques for s in c.suggestions],
                    'blocking': any(c.severity == CritiqueSeverity.BLOCKING for c in critiques)
                }
                result = await proposer.respond_to_critique(critique_summary)
            else:
                result = {}

            return Response(
                proposal_id=proposal.proposal_id,
                addressed_issues=result.get('addressed', []),
                unaddressed_issues=result.get('unaddressed', all_issues),
                modifications=result.get('modifications', {}),
                revised_confidence=result.get('confidence', proposal.confidence),
                reasoning=result.get('reasoning', 'No specific response')
            )

        except Exception as e:
            print(f"Response from {proposer.name} failed: {e}")
            return Response(
                proposal_id=proposal.proposal_id,
                addressed_issues=[],
                unaddressed_issues=[i for c in critiques for i in c.issues],
                modifications={},
                revised_confidence=proposal.confidence * 0.8,
                reasoning=f"Failed to respond: {e}"
            )

    async def _get_vote(
        self,
        agent: Any,
        proposal: Proposal,
        critiques: List[Critique],
        response: Optional[Response],
        context: Optional[Dict]
    ) -> Vote:
        """Get vote from an agent."""
        try:
            if hasattr(agent, 'vote'):
                result = await agent.vote(proposal.content, response)
                vote = result if isinstance(result, bool) else True
            else:
                # Heuristic vote based on critiques
                agent_critique = next(
                    (c for c in critiques if c.critic_name == agent.name),
                    None
                )
                if agent_critique:
                    vote = agent_critique.approval_vote
                else:
                    vote = proposal.confidence > 0.5

            return Vote(
                voter_name=agent.name,
                proposal_id=proposal.proposal_id,
                vote=vote,
                reasoning="Based on analysis",
                confidence=0.7
            )

        except Exception as e:
            print(f"Vote from {agent.name} failed: {e}")
            return Vote(
                voter_name=agent.name,
                proposal_id=proposal.proposal_id,
                vote=True,  # Default to accept on error
                reasoning=f"Default vote due to error: {e}",
                confidence=0.3
            )

    def _get_agent_by_name(self, name: str) -> Optional[Any]:
        """Get agent by name."""
        for agent in self.agents:
            if agent.name == name:
                return agent
        return None

    def _parse_severity(self, severity_str: str) -> CritiqueSeverity:
        """Parse severity string to enum."""
        try:
            return CritiqueSeverity[severity_str.upper()]
        except (KeyError, AttributeError):
            return CritiqueSeverity.CONCERN

    def _heuristic_critique(self, agent: Any, proposal: Proposal) -> Dict:
        """Generate heuristic critique when agent doesn't have critique method."""
        return {
            'issues': [],
            'severity': 'SUGGESTION',
            'suggestions': [],
            'approval_vote': True,
            'reasoning': 'No specific critique'
        }

    def _default_critique(self, agent_name: str, proposal: Proposal) -> Critique:
        """Create default critique."""
        return Critique(
            critic_name=agent_name,
            proposal_id=proposal.proposal_id,
            issues=[],
            severity=CritiqueSeverity.SUGGESTION,
            suggestions=[],
            approval_vote=True,
            reasoning="Default approval"
        )

    def _summarize_debate(
        self,
        proposal: Proposal,
        critiques: List[Critique],
        votes: List[Vote],
        accepted: bool
    ) -> str:
        """Summarize debate outcome."""
        accept_count = sum(1 for v in votes if v.vote)
        total_votes = len(votes)
        issue_count = sum(len(c.issues) for c in critiques)

        status = "ACCEPTED" if accepted else "REJECTED"
        return (
            f"Proposal {proposal.proposal_id} {status}. "
            f"Votes: {accept_count}/{total_votes}. "
            f"Issues raised: {issue_count}."
        )

    def _summarize_consensus(
        self,
        accepted: List[Proposal],
        debates: List[DebateOutcome]
    ) -> str:
        """Summarize consensus result."""
        return (
            f"Accepted {len(accepted)} proposals after "
            f"{len(debates)} debate rounds."
        )

    def get_statistics(self) -> Dict[str, Any]:
        """Get debate statistics."""
        if not self.debate_history:
            return {
                'total_debates': 0,
                'acceptance_rate': 0,
                'avg_consensus_score': 0,
                'avg_rounds': 0
            }

        return {
            'total_debates': len(self.debate_history),
            'acceptance_rate': sum(1 for d in self.debate_history if d.accepted) / len(self.debate_history),
            'avg_consensus_score': sum(d.consensus_score for d in self.debate_history) / len(self.debate_history),
            'avg_rounds': sum(d.rounds_taken for d in self.debate_history) / len(self.debate_history),
            'total_critiques': sum(len(d.critiques) for d in self.debate_history),
            'total_votes': sum(len(d.votes) for d in self.debate_history)
        }


# Convenience function
def create_debate_coordinator(
    agents: List[Any],
    max_rounds: int = 3,
    consensus_threshold: float = 0.6
) -> DebateAndCritiqueCoordinator:
    """Create a debate coordinator with given agents."""
    return DebateAndCritiqueCoordinator(
        agents=agents,
        max_rounds=max_rounds,
        consensus_threshold=consensus_threshold
    )


# Main entry point for testing
if __name__ == '__main__':
    async def test_debate():
        """Test the debate coordinator."""
        print("\n" + "="*60)
        print("DEBATE AND CRITIQUE COORDINATOR - Test")
        print("="*60)

        # Create mock agents
        class MockAgent:
            def __init__(self, name: str, bias: float = 0.5):
                self.name = name
                self.bias = bias  # Tendency to approve (0-1)

            async def critique(self, proposal: Dict) -> Dict:
                import random
                approve = random.random() < self.bias
                return {
                    'issues': [] if approve else ['Issue found'],
                    'severity': 'SUGGESTION' if approve else 'CONCERN',
                    'suggestions': ['Consider this'],
                    'approval_vote': approve,
                    'reasoning': f'{self.name} analysis'
                }

            async def vote(self, proposal: Any, response: Any) -> bool:
                import random
                return random.random() < self.bias

        agents = [
            MockAgent("Strategist", 0.7),
            MockAgent("Critic", 0.4),
            MockAgent("Optimizer", 0.6),
        ]

        coordinator = create_debate_coordinator(agents)

        # Create test proposal
        proposal = Proposal(
            proposal_id="test_001",
            proposer_name="Strategist",
            proposal_type="routing_modification",
            content={"action": "reroute", "net": "DATA0"},
            reasoning="Reduce congestion",
            confidence=0.8
        )

        print(f"\nDebating proposal: {proposal.proposal_id}")
        print(f"  Type: {proposal.proposal_type}")
        print(f"  Confidence: {proposal.confidence}")

        outcome = await coordinator.debate_proposal(proposal)

        print(f"\n--- Debate Outcome ---")
        print(f"  Accepted: {outcome.accepted}")
        print(f"  Consensus Score: {outcome.consensus_score:.2f}")
        print(f"  Critiques: {len(outcome.critiques)}")
        print(f"  Votes: {len(outcome.votes)}")
        print(f"  Reasoning: {outcome.reasoning}")

        # Show critique details
        for critique in outcome.critiques:
            print(f"\n  Critique from {critique.critic_name}:")
            print(f"    Issues: {critique.issues}")
            print(f"    Vote: {'Yes' if critique.approval_vote else 'No'}")

        # Show vote details
        print(f"\n  Votes:")
        for vote in outcome.votes:
            print(f"    {vote.voter_name}: {'Accept' if vote.vote else 'Reject'}")

        # Get statistics
        stats = coordinator.get_statistics()
        print(f"\n--- Statistics ---")
        print(f"  Total debates: {stats['total_debates']}")
        print(f"  Acceptance rate: {stats['acceptance_rate']:.0%}")

    asyncio.run(test_debate())
