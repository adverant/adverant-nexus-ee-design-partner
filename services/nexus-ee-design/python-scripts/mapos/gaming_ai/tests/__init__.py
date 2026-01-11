"""
Gaming AI Test Suite

Comprehensive tests for all Gaming AI components:
- PCB Graph Encoder (GNN)
- Value Network (AlphaZero-style)
- Policy Network (modification selection)
- Dynamics Network (MuZero world model)
- MAP-Elites Archive (quality-diversity)
- Red Queen Evolver (adversarial evolution)
- Ralph Wiggum Optimizer (persistent iteration)
- Training Pipeline (experience replay)
- Integration (complete MAPOS-RQ)

Run all tests:
    pytest gaming_ai/tests/ -v

Run with coverage:
    pytest gaming_ai/tests/ --cov=gaming_ai --cov-report=html

Run specific module:
    pytest gaming_ai/tests/test_map_elites.py -v
"""

__all__ = []
