"""
Schematic Mutation Operators

Implements 5 mutation strategies for intelligent schematic modification:
1. LLM-Guided - Ask Claude to suggest improvements
2. Topology Refinement - Restructure power delivery
3. Component Optimization - Parametric tweaking
4. Interface Hardening - Add protection/isolation
5. Routing Optimization - Rearrange hierarchy
"""

import logging
import random
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple, Callable
from enum import Enum
from copy import deepcopy

from .config import get_schematic_config, MutationConfig

logger = logging.getLogger(__name__)


class MutationStrategy(Enum):
    """Available mutation strategies."""
    LLM_GUIDED = "llm_guided"
    TOPOLOGY_REFINEMENT = "topology_refinement"
    COMPONENT_OPTIMIZATION = "component_optimization"
    INTERFACE_HARDENING = "interface_hardening"
    ROUTING_OPTIMIZATION = "routing_optimization"


@dataclass
class MutationResult:
    """Result of applying a mutation."""
    success: bool
    strategy: MutationStrategy
    description: str
    changes: List[Dict[str, Any]] = field(default_factory=list)
    original_hash: str = ""
    mutated_hash: str = ""


class SchematicMutator:
    """
    Applies mutations to schematics using various strategies.

    The mutator selects strategies based on configured probabilities
    and the current state of the schematic (e.g., stagnation detection).
    """

    def __init__(
        self,
        llm_client: Optional[Any] = None,
        config: Optional[MutationConfig] = None
    ):
        """
        Initialize schematic mutator.

        Args:
            llm_client: Optional LLM client for guided mutations
            config: Mutation configuration
        """
        self.llm_client = llm_client
        self.config = config or get_schematic_config().mutation

        # Strategy weights (normalized probabilities)
        self.strategy_weights = {
            MutationStrategy.LLM_GUIDED: self.config.llm_guided_probability,
            MutationStrategy.TOPOLOGY_REFINEMENT: self.config.topology_refinement_probability,
            MutationStrategy.COMPONENT_OPTIMIZATION: self.config.component_optimization_probability,
            MutationStrategy.INTERFACE_HARDENING: self.config.interface_hardening_probability,
            MutationStrategy.ROUTING_OPTIMIZATION: self.config.routing_optimization_probability,
        }

    def select_strategy(self, stagnation_count: int = 0) -> MutationStrategy:
        """
        Select mutation strategy based on probabilities and stagnation.

        Args:
            stagnation_count: Number of iterations without improvement

        Returns:
            Selected MutationStrategy
        """
        weights = dict(self.strategy_weights)

        # Adjust weights based on stagnation
        if stagnation_count > 20:
            # Increase topology refinement for more drastic changes
            weights[MutationStrategy.TOPOLOGY_REFINEMENT] *= 2.0
            weights[MutationStrategy.LLM_GUIDED] *= 1.5
        elif stagnation_count > 10:
            # Increase LLM guidance
            weights[MutationStrategy.LLM_GUIDED] *= 1.5

        # Normalize weights
        total = sum(weights.values())
        probs = [w / total for w in weights.values()]
        strategies = list(weights.keys())

        return random.choices(strategies, weights=probs, k=1)[0]

    async def mutate(
        self,
        schematic: Dict[str, Any],
        strategy: Optional[MutationStrategy] = None,
        stagnation_count: int = 0,
        validation_results: Optional[Dict[str, Any]] = None
    ) -> Tuple[Dict[str, Any], MutationResult]:
        """
        Apply mutation to schematic.

        Args:
            schematic: Original schematic dictionary
            strategy: Optional specific strategy (random if None)
            stagnation_count: For strategy selection
            validation_results: Current validation issues to address

        Returns:
            (mutated_schematic, mutation_result)
        """
        if strategy is None:
            strategy = self.select_strategy(stagnation_count)

        # Create deep copy to avoid modifying original
        mutated = deepcopy(schematic)
        original_hash = _hash_schematic(schematic)

        logger.debug(f"Applying {strategy.value} mutation")

        try:
            if strategy == MutationStrategy.LLM_GUIDED:
                result = await self._apply_llm_guided(mutated, validation_results)
            elif strategy == MutationStrategy.TOPOLOGY_REFINEMENT:
                result = self._apply_topology_refinement(mutated)
            elif strategy == MutationStrategy.COMPONENT_OPTIMIZATION:
                result = self._apply_component_optimization(mutated)
            elif strategy == MutationStrategy.INTERFACE_HARDENING:
                result = self._apply_interface_hardening(mutated)
            elif strategy == MutationStrategy.ROUTING_OPTIMIZATION:
                result = self._apply_routing_optimization(mutated)
            else:
                result = MutationResult(
                    success=False,
                    strategy=strategy,
                    description=f"Unknown strategy: {strategy}"
                )
        except Exception as e:
            logger.error(f"Mutation failed: {e}")
            result = MutationResult(
                success=False,
                strategy=strategy,
                description=f"Mutation error: {str(e)}"
            )

        result.original_hash = original_hash
        result.mutated_hash = _hash_schematic(mutated)

        return mutated, result

    async def _apply_llm_guided(
        self,
        schematic: Dict[str, Any],
        validation_results: Optional[Dict[str, Any]] = None
    ) -> MutationResult:
        """Apply LLM-guided mutation."""
        if self.llm_client is None:
            return MutationResult(
                success=False,
                strategy=MutationStrategy.LLM_GUIDED,
                description="No LLM client available"
            )

        # Build prompt based on validation issues
        prompt = self._build_llm_prompt(schematic, validation_results)

        try:
            response = await self.llm_client.generate(prompt)
            changes = self._parse_llm_response(response, schematic)

            if changes:
                return MutationResult(
                    success=True,
                    strategy=MutationStrategy.LLM_GUIDED,
                    description=f"LLM suggested {len(changes)} changes",
                    changes=changes
                )
            else:
                return MutationResult(
                    success=False,
                    strategy=MutationStrategy.LLM_GUIDED,
                    description="LLM provided no actionable changes"
                )
        except Exception as e:
            logger.warning(f"LLM mutation failed: {e}")
            return MutationResult(
                success=False,
                strategy=MutationStrategy.LLM_GUIDED,
                description=f"LLM error: {str(e)}"
            )

    def _build_llm_prompt(
        self,
        schematic: Dict[str, Any],
        validation_results: Optional[Dict[str, Any]] = None
    ) -> str:
        """Build prompt for LLM-guided mutation."""
        components = schematic.get("components", [])
        nets = schematic.get("nets", [])

        prompt = f"""Analyze this schematic and suggest ONE specific improvement.

Schematic Summary:
- Components: {len(components)}
- Nets: {len(nets)}
- Sheets: {len(schematic.get('sheets', []))}

"""
        if validation_results:
            violations = validation_results.get("violations", [])
            if violations:
                prompt += "Current Issues:\n"
                for v in violations[:5]:  # Top 5 issues
                    prompt += f"- {v.get('rule', 'Unknown')}: {v.get('message', '')}\n"

        prompt += """
Suggest ONE modification in this exact JSON format:
{
  "action": "add|remove|modify",
  "component_type": "capacitor|resistor|etc",
  "reference": "C1|R1|etc",
  "value": "100nF",
  "reason": "Brief explanation"
}

Only output the JSON, no other text."""

        return prompt

    def _parse_llm_response(
        self,
        response: str,
        schematic: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Parse LLM response and apply changes."""
        import json as json_module

        changes = []
        try:
            # Extract JSON from response
            start = response.find("{")
            end = response.rfind("}") + 1
            if start >= 0 and end > start:
                suggestion = json_module.loads(response[start:end])
                change = self._apply_suggestion(schematic, suggestion)
                if change:
                    changes.append(change)
        except Exception as e:
            logger.warning(f"Failed to parse LLM response: {e}")

        return changes

    def _apply_suggestion(
        self,
        schematic: Dict[str, Any],
        suggestion: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Apply a single LLM suggestion to schematic."""
        action = suggestion.get("action", "").lower()
        components = schematic.setdefault("components", [])

        if action == "add":
            new_component = {
                "reference": suggestion.get("reference", f"X{len(components)+1}"),
                "type": suggestion.get("component_type", "component"),
                "value": suggestion.get("value", ""),
                "footprint": suggestion.get("footprint", ""),
            }
            components.append(new_component)
            return {
                "type": "add",
                "component": new_component,
                "reason": suggestion.get("reason", "")
            }

        elif action == "modify":
            ref = suggestion.get("reference", "")
            for comp in components:
                if comp.get("reference") == ref:
                    old_value = comp.get("value", "")
                    comp["value"] = suggestion.get("value", old_value)
                    return {
                        "type": "modify",
                        "reference": ref,
                        "old_value": old_value,
                        "new_value": comp["value"],
                        "reason": suggestion.get("reason", "")
                    }

        elif action == "remove":
            ref = suggestion.get("reference", "")
            for i, comp in enumerate(components):
                if comp.get("reference") == ref:
                    removed = components.pop(i)
                    return {
                        "type": "remove",
                        "component": removed,
                        "reason": suggestion.get("reason", "")
                    }

        return None

    def _apply_topology_refinement(self, schematic: Dict[str, Any]) -> MutationResult:
        """Apply topology refinement mutation."""
        changes = []
        components = schematic.setdefault("components", [])

        # Find regulators
        regulators = [c for c in components
                      if "regulator" in c.get("type", "").lower()]

        if regulators:
            # Random topology change
            change_type = random.choice([
                "add_bulk_cap",
                "add_input_filter",
                "split_power_domain",
                "upgrade_regulator"
            ])

            if change_type == "add_bulk_cap":
                # Add bulk capacitor near first regulator
                reg = regulators[0]
                new_cap = {
                    "reference": f"C{len(components)+1}",
                    "type": "capacitor",
                    "value": "47uF",
                    "footprint": "Capacitor_SMD:C_1206",
                    "notes": f"Bulk cap for {reg.get('reference', 'regulator')}"
                }
                components.append(new_cap)
                changes.append({
                    "type": "add",
                    "component": new_cap,
                    "reason": "Add bulk capacitor for improved stability"
                })

            elif change_type == "add_input_filter":
                # Add ferrite bead on input
                new_ferrite = {
                    "reference": f"FB{len(components)+1}",
                    "type": "ferrite_bead",
                    "value": "600R@100MHz",
                    "footprint": "Inductor_SMD:L_0805",
                }
                components.append(new_ferrite)
                changes.append({
                    "type": "add",
                    "component": new_ferrite,
                    "reason": "Add input filter for EMC"
                })

        return MutationResult(
            success=len(changes) > 0,
            strategy=MutationStrategy.TOPOLOGY_REFINEMENT,
            description=f"Topology refinement: {len(changes)} changes",
            changes=changes
        )

    def _apply_component_optimization(self, schematic: Dict[str, Any]) -> MutationResult:
        """Apply component optimization mutation."""
        changes = []
        components = schematic.get("components", [])

        # Find capacitors to optimize
        capacitors = [c for c in components
                      if "capacitor" in c.get("type", "").lower()]

        if capacitors and self.config.allow_package_change:
            # Upgrade a random capacitor value
            cap = random.choice(capacitors)
            old_value = cap.get("value", "100nF")

            # Value upgrade map
            upgrades = {
                "100nF": "220nF",
                "100n": "220n",
                "0.1uF": "0.22uF",
                "10uF": "22uF",
                "22uF": "47uF",
            }

            if old_value in upgrades:
                new_value = upgrades[old_value]
                cap["value"] = new_value
                changes.append({
                    "type": "modify",
                    "reference": cap.get("reference", ""),
                    "old_value": old_value,
                    "new_value": new_value,
                    "reason": "Upgrade capacitor for improved filtering"
                })

        return MutationResult(
            success=len(changes) > 0,
            strategy=MutationStrategy.COMPONENT_OPTIMIZATION,
            description=f"Component optimization: {len(changes)} changes",
            changes=changes
        )

    def _apply_interface_hardening(self, schematic: Dict[str, Any]) -> MutationResult:
        """Apply interface hardening mutation."""
        changes = []
        components = schematic.setdefault("components", [])

        # Find interfaces that might need protection
        interfaces = [c for c in components
                      if c.get("type", "").lower() in
                      ["connector", "usb", "can", "rs485", "ethernet"]]

        # Find existing protection
        protection = [c for c in components
                      if any(kw in c.get("type", "").lower()
                             for kw in ["tvs", "esd", "clamp"])]

        # Add protection if interfaces lack it
        if len(interfaces) > len(protection) and self.config.add_esd_protection:
            # Add TVS diode
            new_tvs = {
                "reference": f"D{len(components)+1}",
                "type": "tvs_diode",
                "value": "SMBJ5.0CA",
                "footprint": "Diode_SMD:D_SMB",
                "notes": "ESD protection"
            }
            components.append(new_tvs)
            changes.append({
                "type": "add",
                "component": new_tvs,
                "reason": "Add ESD protection for external interface"
            })

        # Check for CAN without termination
        can_interfaces = [c for c in interfaces if "can" in c.get("type", "").lower()]
        if can_interfaces and self.config.add_termination_resistors:
            # Check for existing 120Ω resistor
            has_termination = any(
                "120" in c.get("value", "") and "resistor" in c.get("type", "").lower()
                for c in components
            )
            if not has_termination:
                new_resistor = {
                    "reference": f"R{len(components)+1}",
                    "type": "resistor",
                    "value": "120R",
                    "footprint": "Resistor_SMD:R_0603",
                    "notes": "CAN bus termination"
                }
                components.append(new_resistor)
                changes.append({
                    "type": "add",
                    "component": new_resistor,
                    "reason": "Add CAN bus termination resistor"
                })

        return MutationResult(
            success=len(changes) > 0,
            strategy=MutationStrategy.INTERFACE_HARDENING,
            description=f"Interface hardening: {len(changes)} changes",
            changes=changes
        )

    def _apply_routing_optimization(self, schematic: Dict[str, Any]) -> MutationResult:
        """Apply routing/hierarchy optimization mutation."""
        changes = []
        sheets = schematic.setdefault("sheets", [{"name": "main"}])
        components = schematic.get("components", [])

        # If single sheet with many components, suggest splitting
        if len(sheets) == 1 and len(components) > 50:
            # Create power supply sheet
            sheets.append({
                "name": "Power Supply",
                "filename": "power_supply.kicad_sch"
            })
            changes.append({
                "type": "add_sheet",
                "sheet": sheets[-1],
                "reason": "Split large schematic for maintainability"
            })

        # If multiple sheets, ensure hierarchy labels exist
        if len(sheets) > 1:
            # Check for hierarchical nets
            nets = schematic.get("nets", [])
            power_nets = [n for n in nets
                         if any(p in n.get("name", "").upper()
                                for p in ["VDD", "VCC", "GND"])]

            if power_nets:
                # Ensure global labels
                for net in power_nets[:3]:
                    net.setdefault("label_type", "global")
                    if net.get("label_type") != "global":
                        net["label_type"] = "global"
                        changes.append({
                            "type": "modify_net",
                            "net": net.get("name", ""),
                            "change": "Upgrade to global label",
                            "reason": "Ensure power nets visible across sheets"
                        })

        return MutationResult(
            success=len(changes) > 0,
            strategy=MutationStrategy.ROUTING_OPTIMIZATION,
            description=f"Routing optimization: {len(changes)} changes",
            changes=changes
        )


def _hash_schematic(schematic: Dict[str, Any]) -> str:
    """Generate hash of schematic for change detection."""
    import hashlib
    import json

    # Sort for consistent hashing
    components = sorted(
        schematic.get("components", []),
        key=lambda c: c.get("reference", "")
    )
    nets = sorted(
        schematic.get("nets", []),
        key=lambda n: n.get("name", "")
    )

    content = {
        "components": components,
        "nets": nets,
        "sheets": schematic.get("sheets", [])
    }

    return hashlib.sha256(
        json.dumps(content, sort_keys=True).encode()
    ).hexdigest()[:16]


async def apply_mutation(
    schematic: Dict[str, Any],
    strategy: Optional[MutationStrategy] = None,
    llm_client: Optional[Any] = None,
    config: Optional[MutationConfig] = None
) -> Tuple[Dict[str, Any], MutationResult]:
    """
    Convenience function to apply mutation.

    Args:
        schematic: Original schematic
        strategy: Optional specific strategy
        llm_client: Optional LLM client
        config: Optional mutation config

    Returns:
        (mutated_schematic, mutation_result)
    """
    mutator = SchematicMutator(llm_client=llm_client, config=config)
    return await mutator.mutate(schematic, strategy=strategy)
