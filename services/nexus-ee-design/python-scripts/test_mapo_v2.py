#!/usr/bin/env python3
"""
MAPO v2.0 Test Script

Tests the MAPO v2.0 module structure and basic functionality.
Compares architecture between v1.0 and v2.0.
"""

import sys
import os

# Add the python-scripts directory to path
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)


def test_v2_imports():
    """Test that all v2.0 modules can be imported."""
    print("=" * 60)
    print("Testing MAPO v2.0 Imports")
    print("=" * 60)

    success_count = 0
    fail_count = 0

    # Test agent imports
    print("\n[Agents]")
    try:
        from mapos_v2.agents import (
            RoutingStrategistAgent,
            CongestionPredictorAgent,
            ConflictResolverAgent,
            SignalIntegrityAdvisorAgent,
            LayerAssignmentStrategistAgent
        )
        print("  ✓ All 5 agent classes imported successfully")
        success_count += 5
    except ImportError as e:
        print(f"  ✗ Agent import failed: {e}")
        fail_count += 5

    # Test core component imports
    print("\n[Core Components]")

    components = [
        ("debate_coordinator", "DebateAndCritiqueCoordinator"),
        ("pathfinder_router", "PathFinderRouter"),
        ("cbs_router", "CBSRouter"),
        ("layer_optimizer", "LayerAssignmentOptimizer"),
        ("multi_agent_optimizer", "MultiAgentOptimizer"),
    ]

    for module_name, class_name in components:
        try:
            module = __import__(f"mapos_v2.{module_name}", fromlist=[class_name])
            cls = getattr(module, class_name)
            print(f"  ✓ {class_name}")
            success_count += 1
        except ImportError as e:
            print(f"  ✗ {class_name}: {e}")
            fail_count += 1

    # Test package-level imports
    print("\n[Package-Level Imports]")
    try:
        import mapos_v2
        print(f"  ✓ mapos_v2 version: {mapos_v2.__version__}")
        print(f"  ✓ {len(mapos_v2.__all__)} exports available")
        success_count += 2
    except ImportError as e:
        print(f"  ✗ Package import failed: {e}")
        fail_count += 2

    print(f"\n[Results] {success_count} passed, {fail_count} failed")
    return fail_count == 0


def test_v2_instantiation():
    """Test that v2.0 components can be instantiated."""
    print("\n" + "=" * 60)
    print("Testing MAPO v2.0 Component Instantiation")
    print("=" * 60)

    success_count = 0
    fail_count = 0

    # Test agent instantiation (without API key - should use defaults)
    print("\n[Agent Instantiation (no API key)]")

    try:
        from mapos_v2.agents import create_routing_strategist
        agent = create_routing_strategist()
        print(f"  ✓ RoutingStrategistAgent created")
        success_count += 1
    except Exception as e:
        print(f"  ✗ RoutingStrategistAgent: {e}")
        fail_count += 1

    try:
        from mapos_v2.agents import create_congestion_predictor
        agent = create_congestion_predictor()
        print(f"  ✓ CongestionPredictorAgent created")
        success_count += 1
    except Exception as e:
        print(f"  ✗ CongestionPredictorAgent: {e}")
        fail_count += 1

    try:
        from mapos_v2.agents import create_si_advisor
        agent = create_si_advisor()
        print(f"  ✓ SignalIntegrityAdvisorAgent created")
        success_count += 1
    except Exception as e:
        print(f"  ✗ SignalIntegrityAdvisorAgent: {e}")
        fail_count += 1

    # Test core component instantiation
    print("\n[Core Component Instantiation]")

    try:
        from mapos_v2.pathfinder_router import create_pathfinder_router
        router = create_pathfinder_router()
        print(f"  ✓ PathFinderRouter created")
        success_count += 1
    except Exception as e:
        print(f"  ✗ PathFinderRouter: {e}")
        fail_count += 1

    try:
        from mapos_v2.cbs_router import create_cbs_router
        router = create_cbs_router()
        print(f"  ✓ CBSRouter created")
        success_count += 1
    except Exception as e:
        print(f"  ✗ CBSRouter: {e}")
        fail_count += 1

    try:
        from mapos_v2.layer_optimizer import create_layer_optimizer
        optimizer = create_layer_optimizer()
        print(f"  ✓ LayerAssignmentOptimizer created")
        success_count += 1
    except Exception as e:
        print(f"  ✗ LayerAssignmentOptimizer: {e}")
        fail_count += 1

    try:
        from mapos_v2.multi_agent_optimizer import create_optimizer
        optimizer = create_optimizer(enable_all_features=False)
        print(f"  ✓ MultiAgentOptimizer created")
        success_count += 1
    except Exception as e:
        print(f"  ✗ MultiAgentOptimizer: {e}")
        fail_count += 1

    # Test stackup presets
    print("\n[Layer Stackup Presets]")
    try:
        from mapos_v2.layer_optimizer import (
            create_2_layer_stackup,
            create_4_layer_stackup,
            create_6_layer_stackup,
            create_8_layer_stackup
        )
        s2 = create_2_layer_stackup()
        s4 = create_4_layer_stackup()
        s6 = create_6_layer_stackup()
        s8 = create_8_layer_stackup()
        print(f"  ✓ 2-layer stackup: {len(s2.layers)} layers")
        print(f"  ✓ 4-layer stackup: {len(s4.layers)} layers")
        print(f"  ✓ 6-layer stackup: {len(s6.layers)} layers")
        print(f"  ✓ 8-layer stackup: {len(s8.layers)} layers")
        success_count += 4
    except Exception as e:
        print(f"  ✗ Stackup presets: {e}")
        fail_count += 4

    print(f"\n[Results] {success_count} passed, {fail_count} failed")
    return fail_count == 0


def compare_v1_v2_architecture():
    """Compare architecture between v1.0 and v2.0."""
    print("\n" + "=" * 60)
    print("MAPO v1.0 vs v2.0 Architecture Comparison")
    print("=" * 60)

    print("""
┌─────────────────────────────────────────────────────────────────┐
│                    MAPO v1.0 Architecture                        │
├─────────────────────────────────────────────────────────────────┤
│  Phase 0: Load & Analyze                                        │
│  Phase 1: Pre-DRC Fixes (deterministic)                         │
│  Phase 2: MCTS Exploration                                      │
│  Phase 3: Evolutionary Optimization                             │
│  Phase 4: Tournament Selection                                  │
│  Phase 5: AlphaFold Refinement                                  │
│                                                                 │
│  Agents: SignalIntegrity, ThermalPower, Manufacturing           │
│  No explicit routing strategy                                   │
│  No layer optimization                                          │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                    MAPO v2.0 Architecture                        │
├─────────────────────────────────────────────────────────────────┤
│  Phase 0: Load & Analyze                                        │
│  Phase 1: LLM Strategic Planning (NEW)                          │
│      - RoutingStrategistAgent                                   │
│      - CongestionPredictorAgent                                 │
│  Phase 2: Pre-DRC Fixes (enhanced with LLM)                     │
│  Phase 3: Negotiation-Based Routing (NEW)                       │
│      - PathFinderRouter (primary)                               │
│      - CBSRouter (fallback for complex)                         │
│  Phase 4: Layer Assignment (NEW)                                │
│      - LayerAssignmentOptimizer (DP)                            │
│      - LayerAssignmentStrategistAgent                           │
│  Phase 5: MCTS Exploration (enhanced with debate)               │
│  Phase 6: Evolutionary Optimization (debate-validated)          │
│  Phase 7: Tournament Selection                                  │
│  Phase 8: AlphaFold Refinement                                  │
│                                                                 │
│  NEW Agents:                                                    │
│    - RoutingStrategistAgent (pre-routing strategy)              │
│    - CongestionPredictorAgent (congestion zones)                │
│    - ConflictResolverAgent (routing conflicts)                  │
│    - SignalIntegrityAdvisorAgent (SI guidance)                  │
│    - LayerAssignmentStrategistAgent (layer hints)               │
│                                                                 │
│  NEW Components:                                                │
│    - DebateAndCritiqueCoordinator (multi-agent debate)          │
│    - PathFinderRouter (negotiation routing)                     │
│    - CBSRouter (conflict-based search)                          │
│    - LayerAssignmentOptimizer (DP optimization)                 │
│    - HybridRouter (PathFinder + CBS)                            │
│                                                                 │
│  Core Philosophy: "Opus 4.5 Thinks, Algorithms Execute"         │
│  All agents use Claude Opus 4.5 via OpenRouter API              │
│  CPU-only algorithms (no GPU required)                          │
└─────────────────────────────────────────────────────────────────┘

Key Improvements:
  1. LLM-first architecture for strategic decisions
  2. Debate-and-critique for complex routing decisions
  3. PathFinder negotiation for iterative conflict resolution
  4. CBS (Conflict-Based Search) for multi-net routing
  5. Layer assignment with DP and LLM hints
  6. Signal integrity awareness throughout
  7. 9-phase pipeline (vs 6 in v1.0)

Research Sources:
  - CircuitLM (arxiv 2601.04505) - debate mechanism
  - OrthoRoute - PathFinder algorithm
  - MAPF literature - CBS adaptation
  - IPC-2141 - SI formulas
""")


def main():
    """Run all tests."""
    print("=" * 60)
    print("MAPO v2.0 Test Suite")
    print("=" * 60)

    all_passed = True

    # Run import tests
    if not test_v2_imports():
        all_passed = False

    # Run instantiation tests
    if not test_v2_instantiation():
        all_passed = False

    # Show architecture comparison
    compare_v1_v2_architecture()

    print("\n" + "=" * 60)
    if all_passed:
        print("All tests PASSED")
    else:
        print("Some tests FAILED")
    print("=" * 60)

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
