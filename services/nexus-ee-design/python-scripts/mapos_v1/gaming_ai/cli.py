#!/usr/bin/env python3
"""
MAPOS-RQ Command Line Interface

Gaming AI-enhanced PCB optimization using:
- Digital Red Queen adversarial evolution
- Ralph Wiggum persistent iteration
- AlphaZero-style neural networks

Usage:
    python -m mapos.gaming_ai.cli optimize board.kicad_pcb --target 50
    python -m mapos.gaming_ai.cli train --checkpoint-dir ./checkpoints
    python -m mapos.gaming_ai.cli analyze board.kicad_pcb
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path
from datetime import datetime

# Add parent directories to path
SCRIPT_DIR = Path(__file__).parent
MAPOS_DIR = SCRIPT_DIR.parent
if str(MAPOS_DIR) not in sys.path:
    sys.path.insert(0, str(MAPOS_DIR))


def cmd_optimize(args):
    """Run MAPOS-RQ optimization."""
    from .integration import MAPOSRQOptimizer, MAPOSRQConfig

    pcb_path = Path(args.pcb_path)
    if not pcb_path.exists():
        print(f"Error: PCB file not found: {pcb_path}")
        return 1

    config = MAPOSRQConfig(
        target_violations=args.target,
        max_iterations=args.max_iterations,
        rq_rounds=args.rq_rounds,
        max_stagnation=args.max_stagnation,
        max_duration_hours=args.max_hours,
        use_neural_networks=not args.no_neural,
        use_llm=not args.no_llm,
        checkpoint_path=args.checkpoint,
        output_dir=args.output_dir,
        save_checkpoints=not args.no_save,
        use_git=args.use_git,
    )

    optimizer = MAPOSRQOptimizer(pcb_path, config)

    print(f"\nStarting MAPOS-RQ optimization...")
    print(f"PCB: {pcb_path}")
    print(f"Target: <= {args.target} violations")
    print(f"Max iterations: {args.max_iterations}")
    print()

    result = asyncio.run(optimizer.optimize())

    print(f"\nOptimization complete!")
    print(f"Status: {result.status.name}")
    print(f"Violations: {result.initial_violations} -> {result.final_violations}")
    print(f"Improvement: {result.improvement} ({100 * result.improvement / max(1, result.initial_violations):.1f}%)")
    print(f"Duration: {result.total_duration_seconds / 60:.1f} minutes")

    if result.best_solution_path:
        print(f"Output: {result.best_solution_path}")

    return 0 if result.final_violations <= args.target else 1


def cmd_train(args):
    """Train neural networks."""
    try:
        from .training import TrainingPipeline
    except ImportError:
        print("Error: PyTorch is required for training. Install with: pip install torch")
        return 1

    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    pipeline = TrainingPipeline(
        hidden_dim=args.hidden_dim,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        checkpoint_dir=checkpoint_dir,
    )

    # Load existing checkpoint if resuming
    if args.resume:
        if pipeline.load_checkpoint('latest'):
            print(f"Resumed from epoch {pipeline.epoch}")
        else:
            print("No checkpoint found, starting fresh")

    # Load experience buffer
    buffer_path = Path(args.experience_buffer) if args.experience_buffer else None
    if buffer_path and buffer_path.exists():
        pipeline.buffer.load(buffer_path)
        print(f"Loaded {len(pipeline.buffer)} experiences")
    else:
        print("Warning: No experience buffer provided. Training needs experiences.")
        print("Run optimizations first to collect training data.")
        return 1

    print(f"\nTraining neural networks...")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Buffer size: {len(pipeline.buffer)}")
    print()

    result = pipeline.train(
        num_epochs=args.epochs,
        save_every=args.save_every,
        log_every=args.log_every,
    )

    print(f"\nTraining complete!")
    print(f"Epochs: {result['epochs_trained']}")
    print(f"Best loss: {result['best_loss']:.4f}")
    print(f"Checkpoint: {checkpoint_dir / 'checkpoint_best.pt'}")

    return 0


def cmd_analyze(args):
    """Analyze a PCB file."""
    from .pcb_graph_encoder import PCBGraph
    from .map_elites import BehavioralDescriptor

    pcb_path = Path(args.pcb_path)
    if not pcb_path.exists():
        print(f"Error: PCB file not found: {pcb_path}")
        return 1

    print(f"\nAnalyzing: {pcb_path.name}")
    print("=" * 60)

    # Load PCB state
    try:
        from pcb_state import PCBState
        state = PCBState.from_file(str(pcb_path))
    except ImportError:
        print("Error: Could not import PCBState")
        return 1

    # Basic statistics
    print(f"\nComponents: {len(state.components)}")
    print(f"Traces: {len(state.traces)}")
    print(f"Vias: {len(state.vias)}")
    print(f"Zones: {len(state.zones)}")
    print(f"Nets: {len(state.nets)}")

    # Run DRC
    print("\nRunning DRC...")
    try:
        drc = state.run_drc()
        print(f"Total violations: {drc.total_violations}")
        print(f"Errors: {drc.errors}")
        print(f"Warnings: {drc.warnings}")
        print(f"Unconnected: {drc.unconnected}")
        print(f"Fitness: {drc.fitness_score:.4f}")

        if drc.violations_by_type:
            print("\nViolations by type:")
            for vtype, count in sorted(drc.violations_by_type.items(), key=lambda x: -x[1])[:10]:
                print(f"  {vtype}: {count}")
    except Exception as e:
        print(f"DRC failed: {e}")

    # Behavioral descriptor
    print("\nBehavioral Descriptor:")
    descriptor = BehavioralDescriptor.from_pcb_state(state)
    for key, value in descriptor.to_dict().items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")

    # Graph encoding
    print("\nGraph Representation:")
    graph = PCBGraph.from_pcb_state(state)
    print(f"  Nodes: {graph.num_nodes}")
    print(f"  Edges: {graph.num_edges}")
    print(f"  Node features: {graph.get_node_features().shape}")
    print(f"  Edge features: {graph.get_edge_features().shape}")

    # Neural network encoding (if available)
    try:
        from .pcb_graph_encoder import PCBGraphEncoder
        encoder = PCBGraphEncoder(hidden_dim=256)
        embedding = encoder.encode_graph(graph)
        print(f"  Embedding: {embedding.shape}")
        print(f"  Embedding norm: {embedding.norm().item():.4f}")
    except Exception:
        print("  (Neural encoder not available)")

    # Save analysis if requested
    if args.output:
        output_path = Path(args.output)
        analysis = {
            'pcb_path': str(pcb_path),
            'components': len(state.components),
            'traces': len(state.traces),
            'vias': len(state.vias),
            'zones': len(state.zones),
            'nets': len(state.nets),
            'drc': {
                'total_violations': drc.total_violations if 'drc' in dir() else None,
                'errors': drc.errors if 'drc' in dir() else None,
                'fitness': drc.fitness_score if 'drc' in dir() else None,
            },
            'descriptor': descriptor.to_dict(),
            'graph': {
                'nodes': graph.num_nodes,
                'edges': graph.num_edges,
            },
            'timestamp': datetime.now().isoformat(),
        }

        with open(output_path, 'w') as f:
            json.dump(analysis, f, indent=2)
        print(f"\nAnalysis saved to: {output_path}")

    return 0


def cmd_archive(args):
    """Manage MAP-Elites archive."""
    from .map_elites import MAPElitesArchive

    archive_path = Path(args.archive_path)

    if args.action == 'create':
        archive = MAPElitesArchive()
        archive.save(archive_path)
        print(f"Created empty archive: {archive_path}")

    elif args.action == 'stats':
        if not archive_path.exists():
            print(f"Error: Archive not found: {archive_path}")
            return 1

        archive = MAPElitesArchive.load(archive_path)
        stats = archive.get_statistics()

        print(f"\nArchive Statistics: {archive_path}")
        print("=" * 40)
        print(f"Total cells: {stats.total_cells}")
        print(f"Filled cells: {stats.filled_cells}")
        print(f"Coverage: {stats.coverage:.1%}")
        print(f"Avg fitness: {stats.avg_fitness:.4f}")
        print(f"Max fitness: {stats.max_fitness:.4f}")
        print(f"Total visits: {stats.total_visits}")
        print(f"Diversity: {stats.diversity_score:.4f}")

    elif args.action == 'visualize':
        if not archive_path.exists():
            print(f"Error: Archive not found: {archive_path}")
            return 1

        archive = MAPElitesArchive.load(archive_path)
        print(archive.visualize(args.dim1 or 0, args.dim2 or 1))

    return 0


def main():
    parser = argparse.ArgumentParser(
        description='MAPOS-RQ: Gaming AI-Enhanced PCB Optimization',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Optimize a PCB file
  python -m mapos.gaming_ai.cli optimize board.kicad_pcb --target 50

  # Train neural networks
  python -m mapos.gaming_ai.cli train --experience-buffer ./experiences.json

  # Analyze a PCB file
  python -m mapos.gaming_ai.cli analyze board.kicad_pcb

  # View archive statistics
  python -m mapos.gaming_ai.cli archive stats archive.json
"""
    )

    subparsers = parser.add_subparsers(dest='command', help='Commands')

    # Optimize command
    opt_parser = subparsers.add_parser('optimize', help='Run MAPOS-RQ optimization')
    opt_parser.add_argument('pcb_path', type=str, help='Path to PCB file')
    opt_parser.add_argument('--target', type=int, default=50, help='Target violations (default: 50)')
    opt_parser.add_argument('--max-iterations', type=int, default=100, help='Max iterations (default: 100)')
    opt_parser.add_argument('--rq-rounds', type=int, default=10, help='Red Queen rounds (default: 10)')
    opt_parser.add_argument('--max-stagnation', type=int, default=15, help='Max stagnation (default: 15)')
    opt_parser.add_argument('--max-hours', type=float, default=24.0, help='Max duration hours (default: 24)')
    opt_parser.add_argument('--no-neural', action='store_true', help='Disable neural networks')
    opt_parser.add_argument('--no-llm', action='store_true', help='Disable LLM guidance')
    opt_parser.add_argument('--checkpoint', type=str, help='Path to model checkpoint')
    opt_parser.add_argument('--output-dir', type=str, help='Output directory')
    opt_parser.add_argument('--no-save', action='store_true', help='Disable checkpoint saving')
    opt_parser.add_argument('--use-git', action='store_true', help='Commit progress to git')

    # Train command
    train_parser = subparsers.add_parser('train', help='Train neural networks')
    train_parser.add_argument('--checkpoint-dir', type=str, default='./checkpoints', help='Checkpoint directory')
    train_parser.add_argument('--experience-buffer', type=str, help='Path to experience buffer')
    train_parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    train_parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    train_parser.add_argument('--learning-rate', type=float, default=1e-4, help='Learning rate')
    train_parser.add_argument('--hidden-dim', type=int, default=256, help='Hidden dimension')
    train_parser.add_argument('--save-every', type=int, default=10, help='Save every N epochs')
    train_parser.add_argument('--log-every', type=int, default=1, help='Log every N epochs')
    train_parser.add_argument('--resume', action='store_true', help='Resume from checkpoint')

    # Analyze command
    analyze_parser = subparsers.add_parser('analyze', help='Analyze a PCB file')
    analyze_parser.add_argument('pcb_path', type=str, help='Path to PCB file')
    analyze_parser.add_argument('--output', type=str, help='Save analysis to JSON')

    # Archive command
    archive_parser = subparsers.add_parser('archive', help='Manage MAP-Elites archive')
    archive_parser.add_argument('action', choices=['create', 'stats', 'visualize'], help='Action to perform')
    archive_parser.add_argument('archive_path', type=str, help='Path to archive file')
    archive_parser.add_argument('--dim1', type=int, help='First dimension for visualization')
    archive_parser.add_argument('--dim2', type=int, help='Second dimension for visualization')

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return 0

    if args.command == 'optimize':
        return cmd_optimize(args)
    elif args.command == 'train':
        return cmd_train(args)
    elif args.command == 'analyze':
        return cmd_analyze(args)
    elif args.command == 'archive':
        return cmd_archive(args)

    return 0


if __name__ == '__main__':
    sys.exit(main())
