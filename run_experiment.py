"""
run_experiment.py
Single entry point: trains CRAM, then benchmarks against transformer.

Usage:
    python run_experiment.py                  # full run (2000 steps train + benchmark)
    python run_experiment.py --steps 500      # quick run
    python run_experiment.py --benchmark-only # skip training, just benchmark
    python run_experiment.py --train-only     # skip benchmark
"""

import argparse
import torch
import os
import sys

# Add parent dir to path so imports work from any directory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from cram.configs.cram_config import CRAMConfig
from cram.core.trainer import CRAMTrainer, TrainConfig
from cram.benchmarks.benchmark import BenchmarkRunner, BenchmarkConfig


def main():
    parser = argparse.ArgumentParser(description="CRAM Training + Benchmark")
    parser.add_argument("--steps", type=int, default=2000, help="Training steps")
    parser.add_argument("--bench-steps", type=int, default=1000, help="Benchmark training steps")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seq-len", type=int, default=64)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--benchmark-only", action="store_true")
    parser.add_argument("--train-only", action="store_true")
    parser.add_argument("--output-dir", type=str, default="./cram_runs")
    parser.add_argument("--quick", action="store_true", help="Quick 200-step test run")
    args = parser.parse_args()

    if args.quick:
        args.steps = 200
        args.bench_steps = 200

    os.makedirs(args.output_dir, exist_ok=True)

    # ── Phase 1: Train CRAM ──────────────────────────────────────────
    if not args.benchmark_only:
        print("\n" + "🚀 " * 20)
        print("PHASE 1: CRAM TRAINING")
        print("🚀 " * 20)

        model_config = CRAMConfig.cram_debug()
        model_config.device = args.device

        train_config = TrainConfig(
            total_steps=args.steps,
            batch_size=args.batch_size,
            seq_len=args.seq_len,
            eval_every=max(50, args.steps // 20),
            log_every=max(10, args.steps // 100),
            save_every=max(200, args.steps // 5),
            device=args.device,
            output_dir=args.output_dir,
            run_name="cram_experiment",
        )

        trainer = CRAMTrainer(model_config, train_config)
        metrics = trainer.train()

        # Print loss curve summary
        loss_curve = metrics.get_loss_curve()
        if loss_curve:
            steps = [s for s, _ in loss_curve]
            losses = [v for _, v in loss_curve]

            # Sample 10 points
            n = len(losses)
            sample_every = max(1, n // 10)
            print(f"\n📉 Loss Curve Summary:")
            print(f"   {'Step':>8} | {'Loss':>10} | {'PPL':>10}")
            print(f"   {'─'*8}-+-{'─'*10}-+-{'─'*10}")
            for i in range(0, n, sample_every):
                ppl = min(2**losses[i], 9999)
                print(f"   {steps[i]:>8} | {losses[i]:>10.4f} | {ppl:>10.2f}")

    # ── Phase 2: Benchmark CRAM vs Transformer ───────────────────────
    if not args.train_only:
        print("\n" + "⚔️  " * 20)
        print("PHASE 2: CRAM vs TRANSFORMER BENCHMARK")
        print("⚔️  " * 20)

        bench_config = BenchmarkConfig(
            train_steps=args.bench_steps,
            batch_size=args.batch_size,
            seq_len=args.seq_len,
            n_eval_batches=20,
            device=args.device,
            scaling_lengths=[16, 32, 64, 128, 256],
        )

        runner = BenchmarkRunner(bench_config)
        results = runner.run_all()

        # Save results
        results_path = os.path.join(args.output_dir, "benchmark_results.json")
        runner.save_results(results_path)


if __name__ == "__main__":
    main()
