"""
CRAM vs Transformer Benchmark Suite
Head-to-head comparison on 4 tasks, same compute budget, same data.

Tasks:
    1. Language Modeling       — baseline perplexity
    2. Copy Task               — SAMG memory test
    3. Associative Recall      — relational retrieval  
    4. Long-Range Dependency   — RSP vs attention for distant context

Fairness constraints:
    - Same parameter count (within 5%)
    - Same training steps
    - Same optimizer (AdamW for both — CRAM's AdaMuon advantage is separate)
    - Same batch size and sequence length
    - Same data (same random seed)

Output: detailed comparison table + loss curves side by side
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import time
import json
import os
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from collections import defaultdict

from cram_config import CRAMConfig
from model import CRAMModel
from transformer_baseline import BaselineTransformer, TransformerConfig
from trainer import SyntheticDataset, TrainConfig


# ─────────────────────────────────────────────────────────────────────
# Benchmark Configuration
# ─────────────────────────────────────────────────────────────────────

@dataclass
class BenchmarkConfig:
    # Training budget
    train_steps: int = 1000
    batch_size: int = 16
    seq_len: int = 64
    n_eval_batches: int = 20    # batches per eval
    seed: int = 42

    # Tasks to run
    tasks: List[str] = field(default_factory=lambda: [
        "language_modeling",
        "copy_task",
        "associative_recall",
        "long_range",
    ])

    # Sequence lengths for scaling test
    scaling_lengths: List[int] = field(default_factory=lambda: [32, 64, 128, 256, 512])

    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


# ─────────────────────────────────────────────────────────────────────
# Model Training Utilities (fair — AdamW for both)
# ─────────────────────────────────────────────────────────────────────

def train_model_adamw(
    model: nn.Module,
    data: SyntheticDataset,
    config: BenchmarkConfig,
    task_mix: Dict[str, float],
    verbose: bool = True,
    model_name: str = "model",
) -> Tuple[nn.Module, List[float]]:
    """
    Train any model with AdamW (fair baseline for both CRAM and Transformer).
    Returns trained model and loss history.
    """
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=3e-4, weight_decay=0.1
    )

    # Cosine LR schedule
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.train_steps, eta_min=3e-5
    )

    loss_history = []
    model.train()

    torch.manual_seed(config.seed)

    for step in range(config.train_steps):
        # Sample task according to mix
        r = torch.rand(1).item()
        cumulative = 0.0
        task = "lm"
        for t, frac in task_mix.items():
            cumulative += frac
            if r < cumulative:
                task = t
                break

        # Get batch
        if task == "lm":
            input_ids, labels = data.language_modeling_batch(config.batch_size, config.seq_len)
        elif task == "copy":
            input_ids, labels = data.copy_task_batch(config.batch_size, config.seq_len)
        elif task == "assoc":
            n_pairs = min(8, config.seq_len // 4)
            input_ids, labels = data.associative_recall_batch(
                config.batch_size, n_pairs, query_delay=min(20, config.seq_len // 4)
            )
        elif task == "longrange":
            input_ids, labels = data.long_range_batch(config.batch_size, config.seq_len)
        else:
            input_ids, labels = data.language_modeling_batch(config.batch_size, config.seq_len)

        optimizer.zero_grad()
        out = model(input_ids, labels=labels)
        loss = out['total_loss']
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        loss_history.append(loss.item())

        if verbose and step % 100 == 0:
            smooth = sum(loss_history[-20:]) / min(20, len(loss_history))
            print(f"    [{model_name}] step {step:4d}/{config.train_steps} | loss={smooth:.4f} | lr={scheduler.get_last_lr()[0]:.2e}")

    return model, loss_history


# ─────────────────────────────────────────────────────────────────────
# Evaluation Functions
# ─────────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate_task(
    model: nn.Module,
    data: SyntheticDataset,
    task: str,
    config: BenchmarkConfig,
) -> Dict[str, float]:
    """Evaluate model on a specific task. Returns metrics dict."""
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_valid = 0

    torch.manual_seed(config.seed + 999)  # different seed from training

    for _ in range(config.n_eval_batches):
        if task == "language_modeling":
            input_ids, labels = data.language_modeling_batch(config.batch_size, config.seq_len)
        elif task == "copy_task":
            input_ids, labels = data.copy_task_batch(config.batch_size, config.seq_len)
        elif task == "associative_recall":
            n_pairs = min(8, config.seq_len // 4)
            input_ids, labels = data.associative_recall_batch(
                config.batch_size, n_pairs, query_delay=min(20, config.seq_len // 4)
            )
        elif task == "long_range":
            input_ids, labels = data.long_range_batch(config.batch_size, config.seq_len)
        else:
            continue

        out = model(input_ids, labels=labels)
        total_loss += out['loss'].item()

        # Per-token accuracy on valid positions
        logits = out['logits']
        pred = logits.argmax(-1)
        mask = (labels != -100)
        if mask.sum() > 0:
            total_correct += ((pred == labels) & mask).sum().item()
            total_valid += mask.sum().item()

    avg_loss = total_loss / config.n_eval_batches
    accuracy = total_correct / max(total_valid, 1)
    perplexity = math.exp(min(avg_loss, 20))   # cap at e^20 for display

    return {
        "loss": avg_loss,
        "accuracy": accuracy,
        "perplexity": perplexity,
    }


@torch.no_grad()
def evaluate_complexity_scaling(
    model: nn.Module,
    data: SyntheticDataset,
    lengths: List[int],
    batch_size: int = 4,
    n_warmup: int = 3,
    n_runs: int = 10,
) -> Dict[int, float]:
    """Measure inference time at different sequence lengths."""
    model.eval()
    times = {}

    for L in lengths:
        input_ids = torch.randint(0, data.vocab_size, (batch_size, L), device=data.device)

        # Warmup
        for _ in range(n_warmup):
            _ = model(input_ids)

        # Synchronize before timing
        if data.device.type == 'cuda':
            torch.cuda.synchronize()

        start = time.perf_counter()
        for _ in range(n_runs):
            _ = model(input_ids)
        if data.device.type == 'cuda':
            torch.cuda.synchronize()
        elapsed = (time.perf_counter() - start) / n_runs

        times[L] = elapsed * 1000  # ms

    return times


# ─────────────────────────────────────────────────────────────────────
# Main Benchmark Runner
# ─────────────────────────────────────────────────────────────────────

class BenchmarkRunner:
    """
    Runs CRAM vs Transformer comparison across all tasks and metrics.
    """

    def __init__(self, bench_config: BenchmarkConfig):
        self.bcfg = bench_config
        self.device = torch.device(bench_config.device)

        # Build models
        print(f"\n{'='*60}")
        print("CRAM vs Transformer Benchmark")
        print('='*60)

        # CRAM config (debug size for fast iteration)
        self.cram_config = CRAMConfig.cram_debug()
        self.cram_config.device = bench_config.device
        self.cram_config.max_seq_len = max(bench_config.scaling_lengths) + 10

        # Transformer config — matched to CRAM
        self.tf_config = TransformerConfig.match_cram(self.cram_config)
        self.tf_config.max_seq_len = self.cram_config.max_seq_len

        # Build models
        self.cram = CRAMModel(self.cram_config).to(self.device)
        self.transformer = BaselineTransformer(self.tf_config).to(self.device)

        # Report parameter counts
        cram_params = self.cram.count_parameters()
        tf_params = self.transformer.count_parameters()
        print(f"\n  CRAM params:        {cram_params['total']:>10,}")
        print(f"  Transformer params: {tf_params['total']:>10,}")
        ratio = cram_params['total'] / tf_params['total']
        print(f"  Ratio (CRAM/TF):    {ratio:.3f}x")

        # Shared data generator
        self.data = SyntheticDataset(self.cram_config, self.device)

        # Results storage
        self.results = {}

    def run_training_comparison(self) -> Dict:
        """Train both models on the same data and compare loss curves."""
        print(f"\n{'─'*60}")
        print("Phase 1: Training Comparison (Language Modeling)")
        print('─'*60)

        task_mix = {"lm": 1.0}

        print("\n  Training CRAM...")
        self.cram, cram_losses = train_model_adamw(
            self.cram, self.data, self.bcfg, task_mix,
            verbose=True, model_name="CRAM"
        )

        print("\n  Training Transformer...")
        self.transformer, tf_losses = train_model_adamw(
            self.transformer, self.data, self.bcfg, task_mix,
            verbose=True, model_name="Transformer"
        )

        # Smooth loss curves
        def smooth(losses, w=20):
            out = []
            for i, v in enumerate(losses):
                start = max(0, i - w)
                out.append(sum(losses[start:i+1]) / (i - start + 1))
            return out

        cram_smooth = smooth(cram_losses)
        tf_smooth = smooth(tf_losses)

        result = {
            "cram_losses": cram_losses,
            "tf_losses": tf_losses,
            "cram_final_loss": cram_smooth[-1],
            "tf_final_loss": tf_smooth[-1],
            "cram_improvement": (cram_losses[0] - cram_smooth[-1]) / max(cram_losses[0], 1e-8),
            "tf_improvement": (tf_losses[0] - tf_smooth[-1]) / max(tf_losses[0], 1e-8),
        }
        self.results["training"] = result
        return result

    def run_task_benchmarks(self) -> Dict:
        """Evaluate both trained models on all 4 tasks."""
        print(f"\n{'─'*60}")
        print("Phase 2: Task Benchmarks")
        print('─'*60)

        # Fine-tune both models on task-specific data (100 more steps each)
        task_results = {}

        for task in self.bcfg.tasks:
            print(f"\n  Task: {task}")

            # Map task name to data mix
            mix_map = {
                "language_modeling": {"lm": 1.0},
                "copy_task": {"copy": 1.0},
                "associative_recall": {"assoc": 1.0},
                "long_range": {"longrange": 1.0},
            }
            task_mix = mix_map.get(task, {"lm": 1.0})

            # Fine-tune both (50 steps, same data)
            ftcfg = BenchmarkConfig(
                train_steps=200,
                batch_size=self.bcfg.batch_size,
                seq_len=self.bcfg.seq_len,
                seed=self.bcfg.seed + hash(task) % 100,
            )

            print(f"    Fine-tuning CRAM on {task}...")
            import copy
            cram_ft = copy.deepcopy(self.cram)
            cram_ft, _ = train_model_adamw(cram_ft, self.data, ftcfg, task_mix,
                                            verbose=False, model_name="CRAM")

            print(f"    Fine-tuning Transformer on {task}...")
            tf_ft = copy.deepcopy(self.transformer)
            tf_ft, _ = train_model_adamw(tf_ft, self.data, ftcfg, task_mix,
                                          verbose=False, model_name="TF")

            # Evaluate
            cram_metrics = evaluate_task(cram_ft, self.data, task, self.bcfg)
            tf_metrics = evaluate_task(tf_ft, self.data, task, self.bcfg)

            print(f"    CRAM:        loss={cram_metrics['loss']:.4f}  acc={cram_metrics['accuracy']:.3f}  ppl={cram_metrics['perplexity']:.2f}")
            print(f"    Transformer: loss={tf_metrics['loss']:.4f}  acc={tf_metrics['accuracy']:.3f}  ppl={tf_metrics['perplexity']:.2f}")

            task_results[task] = {
                "cram": cram_metrics,
                "transformer": tf_metrics,
                "cram_wins": cram_metrics['accuracy'] > tf_metrics['accuracy'],
                "accuracy_delta": cram_metrics['accuracy'] - tf_metrics['accuracy'],
                "loss_delta": tf_metrics['loss'] - cram_metrics['loss'],  # positive = CRAM better
            }

        self.results["tasks"] = task_results
        return task_results

    def run_complexity_scaling(self) -> Dict:
        """Measure inference time vs sequence length for both models."""
        print(f"\n{'─'*60}")
        print("Phase 3: Complexity Scaling (Inference Time vs Sequence Length)")
        print('─'*60)

        print("\n  Measuring CRAM inference times...")
        cram_times = evaluate_complexity_scaling(
            self.cram, self.data, self.bcfg.scaling_lengths
        )

        print("  Measuring Transformer inference times...")
        tf_times = evaluate_complexity_scaling(
            self.transformer, self.data, self.bcfg.scaling_lengths
        )

        print(f"\n  {'L':>6} | {'CRAM (ms)':>12} | {'TF (ms)':>12} | {'Ratio TF/CRAM':>14}")
        print(f"  {'─'*6}-+-{'─'*12}-+-{'─'*12}-+-{'─'*14}")
        for L in self.bcfg.scaling_lengths:
            ct = cram_times.get(L, 0)
            tt = tf_times.get(L, 0)
            ratio = tt / ct if ct > 0 else 0
            print(f"  {L:>6} | {ct:>12.2f} | {tt:>12.2f} | {ratio:>14.2f}x")

        # Compute scaling exponents
        lengths = self.bcfg.scaling_lengths
        if len(lengths) >= 2:
            cram_ratios = [cram_times[lengths[i+1]] / cram_times[lengths[i]]
                           for i in range(len(lengths)-1)]
            tf_ratios = [tf_times[lengths[i+1]] / tf_times[lengths[i]]
                         for i in range(len(lengths)-1)]
            # If length doubles, ratio ≈ 2^alpha where alpha is scaling exponent
            import math
            cram_exp = sum(math.log2(r) for r in cram_ratios) / len(cram_ratios)
            tf_exp = sum(math.log2(r) for r in tf_ratios) / len(tf_ratios)
            print(f"\n  CRAM scaling exponent: {cram_exp:.2f} (O(n^{cram_exp:.2f}))")
            print(f"  TF   scaling exponent: {tf_exp:.2f} (O(n^{tf_exp:.2f}))")
        else:
            cram_exp, tf_exp = 1.0, 2.0

        result = {
            "cram_times": cram_times,
            "tf_times": tf_times,
            "cram_exponent": cram_exp,
            "tf_exponent": tf_exp,
        }
        self.results["complexity"] = result
        return result

    def print_final_report(self):
        """Print the complete benchmark report."""
        print(f"\n{'='*60}")
        print("FINAL BENCHMARK REPORT: CRAM vs Transformer")
        print('='*60)

        # Training
        if "training" in self.results:
            r = self.results["training"]
            print(f"\n📈 Training (Language Modeling, {self.bcfg.train_steps} steps)")
            print(f"   CRAM final loss:        {r['cram_final_loss']:.4f}")
            print(f"   Transformer final loss: {r['tf_final_loss']:.4f}")
            delta = r['tf_final_loss'] - r['cram_final_loss']
            winner = "CRAM" if delta > 0 else "Transformer"
            print(f"   Winner: {winner} (Δ={abs(delta):.4f})")

        # Tasks
        if "tasks" in self.results:
            print(f"\n🎯 Task Accuracy Summary")
            print(f"   {'Task':<25} {'CRAM':>8} {'TF':>8} {'Δ':>8} {'Winner'}")
            print(f"   {'─'*25}─{'─'*8}─{'─'*8}─{'─'*8}─{'─'*12}")
            cram_wins = 0
            for task, r in self.results["tasks"].items():
                ca = r["cram"]["accuracy"]
                ta = r["transformer"]["accuracy"]
                d = ca - ta
                winner = "✅ CRAM" if d > 0.01 else ("✅ TF" if d < -0.01 else "Tie")
                if d > 0.01:
                    cram_wins += 1
                print(f"   {task:<25} {ca:>8.3f} {ta:>8.3f} {d:>+8.3f} {winner}")
            print(f"\n   CRAM wins: {cram_wins}/{len(self.results['tasks'])} tasks")

        # Complexity
        if "complexity" in self.results:
            r = self.results["complexity"]
            print(f"\n⚡ Complexity Scaling")
            print(f"   CRAM:        O(n^{r['cram_exponent']:.2f})")
            print(f"   Transformer: O(n^{r['tf_exponent']:.2f})")
            speedup = r['tf_times'].get(max(self.bcfg.scaling_lengths), 1) / \
                      r['cram_times'].get(max(self.bcfg.scaling_lengths), 1)
            print(f"   CRAM speedup at L={max(self.bcfg.scaling_lengths)}: {speedup:.1f}x")

        print(f"\n{'='*60}")

    def run_all(self) -> Dict:
        """Run complete benchmark suite. Returns all results."""
        self.run_training_comparison()
        self.run_task_benchmarks()
        self.run_complexity_scaling()
        self.print_final_report()
        return self.results

    def save_results(self, path: str):
        """Save results to JSON (convert tensors to floats)."""
        def to_serializable(obj):
            if isinstance(obj, dict):
                return {k: to_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [to_serializable(v) for v in obj]
            elif isinstance(obj, torch.Tensor):
                return obj.item()
            elif isinstance(obj, (int, float, bool, str)):
                return obj
            else:
                return str(obj)

        with open(path, 'w') as f:
            json.dump(to_serializable(self.results), f, indent=2)
        print(f"\n  Results saved to: {path}")
