"""
CRAM Training Loop
Full training pipeline with:
    - 5-stage curriculum (module activation schedule)
    - WSAC learning rate schedule
    - CRAMOptimizer (AdaMuon + AdamW)
    - Gradient clipping with MuonClip
    - Live loss/metric tracking
    - Checkpoint saving

Synthetic datasets for fast iteration:
    - Language modeling (next token prediction)
    - Copy task (tests SAMG memory)
    - Associative recall (tests SAMG relational retrieval)
    - Long-range dependency (tests RSP long context)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import time
import json
import os
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple, Iterator
from collections import defaultdict

from cram.configs.cram_config import CRAMConfig
from cram.core.model import CRAMModel
from cram.core.optimizer import CRAMOptimizer, WSACScheduler


# ─────────────────────────────────────────────────────────────────────
# Synthetic Datasets
# ─────────────────────────────────────────────────────────────────────

class SyntheticDataset:
    """
    Fast synthetic datasets — no disk I/O, runs entirely in GPU memory.
    Designed to test specific CRAM capabilities.
    """

    def __init__(self, config: CRAMConfig, device: torch.device):
        self.config = config
        self.device = device
        self.vocab_size = config.vocab_size

    def language_modeling_batch(
        self, batch_size: int, seq_len: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Random token sequences for next-token prediction.
        Baseline task — every model should improve on this.
        """
        tokens = torch.randint(0, self.vocab_size, (batch_size, seq_len + 1),
                               device=self.device)
        return tokens[:, :-1], tokens[:, 1:]

    def copy_task_batch(
        self, batch_size: int, seq_len: int, copy_delay: int = 10
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Copy task: model sees [a b c d ... SEP a b c d ...]
        Must reproduce the first half after SEP token.
        Tests SAMG's ability to store and retrieve sequences.

        Difficulty: copy_delay controls gap between original and copy.
        """
        half = seq_len // 2
        SEP = self.vocab_size - 1   # last token = separator

        # Generate sequence to copy (avoid SEP token)
        original = torch.randint(1, self.vocab_size - 1, (batch_size, half),
                                 device=self.device)
        sep = torch.full((batch_size, 1), SEP, device=self.device)

        # Input:  [original | SEP | original (shifted)]
        input_ids = torch.cat([original, sep, original[:, :-1]], dim=1)
        # Labels: [-100 ... -100 | -100 | original]  (only predict copied part)
        labels = torch.full_like(input_ids, -100)
        labels[:, half + 1:] = original

        return input_ids, labels

    def associative_recall_batch(
        self, batch_size: int, n_pairs: int = 8, query_delay: int = 20
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Associative recall: model sees [k1 v1 k2 v2 ... kN vN ... ki ?]
        Must retrieve vi given ki appeared earlier.
        Tests SAMG's key-value associative memory.

        This is the task where SAMG's graph structure should shine —
        O(log n) retrieval vs transformer's O(n) attention scan.
        """
        # Key-value pairs (keys and values are distinct token ranges)
        keys = torch.randint(1, self.vocab_size // 2,
                             (batch_size, n_pairs), device=self.device)
        values = torch.randint(self.vocab_size // 2, self.vocab_size - 1,
                               (batch_size, n_pairs), device=self.device)

        # Random padding tokens between pairs
        pad_len = query_delay
        padding = torch.randint(1, 10, (batch_size, pad_len), device=self.device)

        # Query: ask for value of a random key
        query_idx = torch.randint(0, n_pairs, (batch_size,))
        query_keys = keys[torch.arange(batch_size), query_idx]   # [B]
        query_values = values[torch.arange(batch_size), query_idx]  # [B]

        # Build sequence: [k1 v1 k2 v2 ... padding ... query_key ?]
        kv_flat = torch.stack([keys, values], dim=2).view(batch_size, -1)  # [B, 2*n_pairs]
        query_tok = query_keys.unsqueeze(1)    # [B, 1]

        input_ids = torch.cat([kv_flat, padding, query_tok], dim=1)

        # Labels: only predict the answer (last token)
        labels = torch.full_like(input_ids, -100)
        labels[:, -1] = query_values

        return input_ids, labels

    def long_range_batch(
        self, batch_size: int, seq_len: int, needle_pos: int = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Long-range dependency: needle in a haystack.
        Model sees a special token at position needle_pos,
        must predict it again at the end of the sequence.
        Tests RSP's long-range memory via slow state.
        """
        if needle_pos is None:
            needle_pos = seq_len // 4   # needle in first quarter

        NEEDLE_OFFSET = self.vocab_size - 10   # special token range

        # Random haystack
        tokens = torch.randint(1, self.vocab_size // 2,
                               (batch_size, seq_len), device=self.device)

        # Plant needle at fixed position
        needle = torch.randint(NEEDLE_OFFSET, self.vocab_size - 1,
                               (batch_size,), device=self.device)
        tokens[:, needle_pos] = needle

        # Task: predict needle at end of sequence
        # We append the needle as the final target
        input_ids = tokens[:, :-1]
        labels = torch.full_like(input_ids, -100)
        # The last non-padding position should predict the needle
        labels[:, -1] = tokens[:, needle_pos]

        return input_ids, labels


# ─────────────────────────────────────────────────────────────────────
# Curriculum Manager
# ─────────────────────────────────────────────────────────────────────

@dataclass
class CurriculumStage:
    name: str
    start_frac: float        # fraction of total steps
    end_frac: float
    # Lambda weights to activate at this stage
    lambda_budget: float = 0.0
    lambda_balance: float = 0.0
    lambda_calib: float = 0.0
    lambda_consist: float = 0.0
    lambda_conf: float = 0.0
    lambda_graph: float = 0.0
    # SAMG surprise threshold
    samg_tau: Optional[float] = None
    # ADR temperature
    adr_temp: Optional[float] = None
    # Data mix (fractions summing to 1)
    data_mix: Dict[str, float] = field(default_factory=lambda: {"lm": 1.0})
    description: str = ""


CURRICULUM = [
    CurriculumStage(
        name="Stage1_RSP_Only",
        start_frac=0.0, end_frac=0.3,
        lambda_budget=0.0, lambda_balance=0.0,
        samg_tau=999.0,   # SAMG frozen (tau=∞ → never surprised → never writes)
        adr_temp=2.0,
        data_mix={"lm": 1.0},
        description="RSP+FFN only. SAMG frozen. Learn basic language modeling."
    ),
    CurriculumStage(
        name="Stage2_SAMG_Activated",
        start_frac=0.3, end_frac=0.55,
        lambda_budget=0.01, lambda_balance=0.01,
        samg_tau=0.7,     # Anneal tau: more writes allowed
        adr_temp=1.5,
        data_mix={"lm": 0.6, "copy": 0.2, "assoc": 0.2},
        description="SAMG activated. Copy + associative recall introduced."
    ),
    CurriculumStage(
        name="Stage3_ADR_All_Paths",
        start_frac=0.55, end_frac=0.75,
        lambda_budget=0.05, lambda_balance=0.02,
        samg_tau=0.5,
        adr_temp=1.0,
        data_mix={"lm": 0.5, "copy": 0.2, "assoc": 0.15, "longrange": 0.15},
        description="All 5 ADR paths active. Temperature annealing."
    ),
    CurriculumStage(
        name="Stage4_All_Modules",
        start_frac=0.75, end_frac=0.92,
        lambda_budget=0.1, lambda_balance=0.05, lambda_calib=0.01,
        lambda_consist=0.01, lambda_conf=0.005,
        samg_tau=0.4,
        adr_temp=0.7,
        data_mix={"lm": 0.4, "copy": 0.2, "assoc": 0.2, "longrange": 0.2},
        description="ALL modules active. SLE reasoning. High-quality task mix."
    ),
    CurriculumStage(
        name="Stage5_Cooldown",
        start_frac=0.92, end_frac=1.0,
        lambda_budget=0.1, lambda_balance=0.05, lambda_calib=0.01,
        lambda_consist=0.01, lambda_conf=0.005,
        samg_tau=0.3,
        adr_temp=0.5,
        data_mix={"lm": 0.3, "copy": 0.25, "assoc": 0.25, "longrange": 0.2},
        description="Cooldown. Checkpoint averaging. Final convergence."
    ),
]


class CurriculumManager:
    """Manages curriculum transitions during training."""

    def __init__(self, model: CRAMModel, total_steps: int):
        self.model = model
        self.total_steps = total_steps
        self.current_stage_idx = -1

    def update(self, step: int) -> Optional[CurriculumStage]:
        """Check if we need to transition to next stage. Returns stage if changed."""
        frac = step / self.total_steps
        for i, stage in enumerate(CURRICULUM):
            if stage.start_frac <= frac < stage.end_frac:
                if i != self.current_stage_idx:
                    self._apply_stage(stage)
                    self.current_stage_idx = i
                    return stage
                return None
        return None

    def _apply_stage(self, stage: CurriculumStage):
        """Apply curriculum stage settings to model."""
        cfg = self.model.config

        # Update loss weights
        cfg.lambda_budget  = stage.lambda_budget
        cfg.lambda_balance = stage.lambda_balance
        cfg.lambda_calib   = stage.lambda_calib
        cfg.lambda_consist = stage.lambda_consist
        cfg.lambda_conf    = stage.lambda_conf
        cfg.lambda_graph   = stage.lambda_graph

        # Update SAMG surprise threshold
        if stage.samg_tau is not None:
            self.model.samg.surprise_tau.data.fill_(stage.samg_tau)

        # Update ADR temperature across all blocks
        if stage.adr_temp is not None:
            for block in self.model.blocks:
                block.adr.temperature = stage.adr_temp

        print(f"\n  🎓 Curriculum → {stage.name}: {stage.description}")


# ─────────────────────────────────────────────────────────────────────
# Metrics Tracker
# ─────────────────────────────────────────────────────────────────────

class MetricsTracker:
    """Tracks and displays training metrics."""

    def __init__(self):
        self.history = defaultdict(list)
        self.step_times = []
        self._last_time = time.time()

    def update(self, step: int, metrics: Dict):
        now = time.time()
        dt = now - self._last_time
        self._last_time = now
        self.step_times.append(dt)

        for k, v in metrics.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            self.history[k].append((step, v))

    def smooth(self, key: str, window: int = 20) -> float:
        """Exponential moving average of metric."""
        vals = [v for _, v in self.history[key][-window:]]
        if not vals:
            return float('nan')
        weights = [0.9 ** (len(vals) - 1 - i) for i in range(len(vals))]
        return sum(w * v for w, v in zip(weights, vals)) / sum(weights)

    def format_row(self, step: int, total: int, metrics: Dict) -> str:
        pct = 100 * step / total
        tokens_per_sec = 0
        if self.step_times:
            avg_dt = sum(self.step_times[-10:]) / len(self.step_times[-10:])
            # approximate tokens/sec
            tokens_per_sec = metrics.get('batch_tokens', 0) / (avg_dt + 1e-8)

        parts = [f"step {step:5d}/{total} ({pct:4.1f}%)"]
        if 'loss' in metrics:
            parts.append(f"loss={metrics['loss']:.4f}")
        if 'task_loss' in metrics:
            parts.append(f"task={metrics['task_loss']:.4f}")
        if 'lr_muon' in metrics:
            parts.append(f"lr={metrics['lr_muon']:.2e}")
        if 'grad_norm' in metrics:
            parts.append(f"gnorm={metrics['grad_norm']:.2f}")
        if 'stage' in metrics:
            parts.append(f"[{metrics['stage']}]")
        if tokens_per_sec > 0:
            parts.append(f"{tokens_per_sec:.0f}tok/s")

        return "  " + " | ".join(parts)

    def save(self, path: str):
        with open(path, 'w') as f:
            json.dump(dict(self.history), f, indent=2)

    def get_loss_curve(self) -> List[Tuple[int, float]]:
        return self.history.get('loss', [])


# ─────────────────────────────────────────────────────────────────────
# Training Configuration
# ─────────────────────────────────────────────────────────────────────

@dataclass
class TrainConfig:
    total_steps: int = 2000
    batch_size: int = 16
    seq_len: int = 64
    eval_every: int = 100
    log_every: int = 10
    save_every: int = 500
    grad_clip: float = 1.0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    dtype: str = "float32"          # float32 for debug; bfloat16 for real training
    output_dir: str = "./cram_runs"
    run_name: str = "cram_debug"


# ─────────────────────────────────────────────────────────────────────
# Main Trainer
# ─────────────────────────────────────────────────────────────────────

class CRAMTrainer:
    """
    Full CRAM training loop.
    Handles curriculum, optimization, logging, checkpointing.
    """

    def __init__(
        self,
        model_config: CRAMConfig,
        train_config: TrainConfig,
    ):
        self.mcfg = model_config
        self.tcfg = train_config
        self.device = torch.device(train_config.device)

        # Model
        print(f"\n{'='*60}")
        print(f"Building CRAM model...")
        self.model = CRAMModel(model_config).to(self.device)
        param_counts = self.model.count_parameters()
        print(f"  Parameters: {param_counts['total']:,}")

        # Optimizer
        print(f"Building CRAMOptimizer (AdaMuon + AdamW)...")
        self.optimizer = CRAMOptimizer(self.model, model_config)

        # Scheduler
        self.scheduler = WSACScheduler(
            self.optimizer,
            total_steps=train_config.total_steps,
            warmup_frac=0.05,
            anneal_start_frac=0.75,
            cooldown_start_frac=0.92,
            min_lr_ratio=0.3,
            peak_lr_muon=model_config.lr_peak,
            peak_lr_adamw=model_config.lr_adamw,
        )

        # Curriculum
        self.curriculum = CurriculumManager(self.model, train_config.total_steps)

        # Data
        self.data = SyntheticDataset(model_config, self.device)

        # Metrics
        self.metrics = MetricsTracker()

        # Output dir
        os.makedirs(train_config.output_dir, exist_ok=True)

    def get_batch(self, stage: CurriculumStage) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample a batch according to current curriculum data mix."""
        r = torch.rand(1).item()
        cumulative = 0.0

        for task, frac in stage.data_mix.items():
            cumulative += frac
            if r < cumulative:
                if task == "lm":
                    return self.data.language_modeling_batch(
                        self.tcfg.batch_size, self.tcfg.seq_len
                    )
                elif task == "copy":
                    return self.data.copy_task_batch(
                        self.tcfg.batch_size, self.tcfg.seq_len
                    )
                elif task == "assoc":
                    n_pairs = min(8, self.tcfg.seq_len // 4)
                    return self.data.associative_recall_batch(
                        self.tcfg.batch_size, n_pairs,
                        query_delay=min(20, self.tcfg.seq_len // 4)
                    )
                elif task == "longrange":
                    return self.data.long_range_batch(
                        self.tcfg.batch_size, self.tcfg.seq_len
                    )

        # Fallback
        return self.data.language_modeling_batch(
            self.tcfg.batch_size, self.tcfg.seq_len
        )

    def train_step(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor,
    ) -> Dict:
        """Single training step. Returns metrics dict."""
        self.model.train()
        self.optimizer.zero_grad()

        out = self.model(input_ids, labels=labels)
        loss = out['total_loss']

        loss.backward()

        # Gradient clipping
        grad_norm = nn.utils.clip_grad_norm_(
            self.model.parameters(), self.tcfg.grad_clip
        )

        self.optimizer.step()
        lr_mult = self.scheduler.step()

        metrics = {
            'loss': loss.item(),
            'task_loss': out['loss'].item(),
            'grad_norm': grad_norm.item() if hasattr(grad_norm, 'item') else float(grad_norm),
            'lr_muon': self.mcfg.lr_peak * lr_mult,
            'lr_mult': lr_mult,
            'batch_tokens': input_ids.numel(),
        }

        # Add auxiliary losses if present
        if 'aux_losses' in out:
            for k, v in out['aux_losses'].items():
                if isinstance(v, torch.Tensor):
                    metrics[f'aux_{k}'] = v.item()

        return metrics

    @torch.no_grad()
    def eval_step(self, task: str = "lm") -> Dict:
        """Run evaluation on a fixed eval batch."""
        self.model.eval()

        if task == "lm":
            input_ids, labels = self.data.language_modeling_batch(
                self.tcfg.batch_size * 2, self.tcfg.seq_len
            )
        elif task == "copy":
            input_ids, labels = self.data.copy_task_batch(
                self.tcfg.batch_size * 2, self.tcfg.seq_len
            )
        elif task == "assoc":
            n_pairs = min(8, self.tcfg.seq_len // 4)
            input_ids, labels = self.data.associative_recall_batch(
                self.tcfg.batch_size * 2, n_pairs
            )

        out = self.model(input_ids, labels=labels)

        # Compute accuracy on valid (non -100) positions
        logits = out['logits']    # [B, L, V]
        B, L, V = logits.shape
        mask = (labels != -100)
        if mask.sum() > 0:
            pred = logits.argmax(-1)
            correct = (pred == labels) & mask
            accuracy = correct.sum().float() / mask.sum().float()
        else:
            accuracy = torch.tensor(0.0)

        return {
            'eval_loss': out['loss'].item(),
            'eval_acc': accuracy.item(),
            'eval_task': task,
        }

    def train(self) -> MetricsTracker:
        """
        Full training loop.
        Returns metrics tracker with complete loss history.
        """
        print(f"\n{'='*60}")
        print(f"Starting training: {self.tcfg.run_name}")
        print(f"  Steps: {self.tcfg.total_steps}")
        print(f"  Batch: {self.tcfg.batch_size} x {self.tcfg.seq_len} tokens")
        print(f"  Device: {self.device}")
        print(f"{'='*60}\n")

        # Initialize curriculum at step 0
        current_stage = CURRICULUM[0]
        self.curriculum._apply_stage(current_stage)

        start_time = time.time()

        for step in range(self.tcfg.total_steps):

            # Check for curriculum transition
            new_stage = self.curriculum.update(step)
            if new_stage is not None:
                current_stage = new_stage

            # Get batch
            input_ids, labels = self.get_batch(current_stage)

            # Train step
            step_metrics = self.train_step(input_ids, labels)
            step_metrics['stage'] = current_stage.name.split('_')[0]

            # Track metrics
            self.metrics.update(step, step_metrics)

            # Log
            if step % self.tcfg.log_every == 0:
                row = self.metrics.format_row(
                    step, self.tcfg.total_steps, step_metrics
                )
                print(row)

            # Eval
            if step % self.tcfg.eval_every == 0 and step > 0:
                eval_lm = self.eval_step("lm")
                eval_copy = self.eval_step("copy")
                eval_assoc = self.eval_step("assoc")
                self.metrics.update(step, {
                    **eval_lm,
                    'copy_acc': eval_copy['eval_acc'],
                    'assoc_acc': eval_assoc['eval_acc'],
                })
                print(f"\n  📊 Eval @ step {step}:")
                print(f"     LM loss={eval_lm['eval_loss']:.4f}  acc={eval_lm['eval_acc']:.3f}")
                print(f"     Copy acc={eval_copy['eval_acc']:.3f}")
                print(f"     Assoc acc={eval_assoc['eval_acc']:.3f}\n")

            # Save checkpoint
            if step % self.tcfg.save_every == 0 and step > 0:
                ckpt_path = os.path.join(
                    self.tcfg.output_dir,
                    f"{self.tcfg.run_name}_step{step}.pt"
                )
                torch.save({
                    'step': step,
                    'model': self.model.state_dict(),
                    'optimizer': self.optimizer.state_dict(),
                    'config': self.mcfg,
                }, ckpt_path)
                print(f"  💾 Saved checkpoint: {ckpt_path}")

        elapsed = time.time() - start_time
        print(f"\n{'='*60}")
        print(f"Training complete in {elapsed:.1f}s ({elapsed/60:.1f}min)")
        smooth_loss = self.metrics.smooth('loss')
        print(f"Final smoothed loss: {smooth_loss:.4f}")
        print(f"{'='*60}")

        # Save metrics
        metrics_path = os.path.join(
            self.tcfg.output_dir,
            f"{self.tcfg.run_name}_metrics.json"
        )
        self.metrics.save(metrics_path)
        print(f"Metrics saved to: {metrics_path}")

        return self.metrics
