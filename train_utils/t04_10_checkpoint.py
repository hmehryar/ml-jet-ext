"""T04-10 deterministic fixed-budget, task-specific best checkpoint driver.

A deliberately separate output namespace and state layout. No access to T04-07.
The caller creates/validates loaders, model and optimizer. Actual HPC production
submission and immutable snapshots remain gated by the T04-10 preflight phase.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import random
import tempfile

from train_utils.t04_telemetry import cuda_reset, epoch_telemetry

import numpy as np
import torch

from train_utils.t04_10_training import run_single_head_epoch


def _rng_state():
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}


def _restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    # torch.load(map_location="cuda") may move CPU RNG state to CUDA.
    # torch.set_rng_state requires a CPU ByteTensor.
    torch.set_rng_state(state["torch"].cpu())
    if state["cuda"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(
            [rng_state.cpu() for rng_state in state["cuda"]]
        )


def _atomic_save(obj, path):
    path = Path(path)
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".tmp", delete=False) as tmp:
        candidate = Path(tmp.name)
    try:
        torch.save(obj, candidate)
        os.replace(candidate, path)
    finally:
        candidate.unlink(missing_ok=True)


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for buf in iter(lambda: f.read(1048576), b""):
            h.update(buf)
    return h.hexdigest()


def fit_fixed_budget(*, model, train_loader, val_loader, optimizer,
                     task, output_dir, config_sha256, epochs=80, device="cpu",
                     precision="fp32", scheduler=None, resume=False,
                     reset_loader_seed=None, microbatch_size=None):
    """Train 80 epochs, select best by target macro-F1, exact state restore.

    `reset_loader_seed(epoch)` should re-seed the dataset shuffling sampler at
    each epoch; required for exact reproducibility with a shuffled DataLoader.
    Only used for first checkpoint validation; no Slurm job submission.
    """
    if type(epochs) is not int or epochs < 1:
        raise ValueError("epochs must be a positive integer")
    if not isinstance(config_sha256, str) or len(config_sha256) != 64:
        raise ValueError("A SHA256 configuration digest is required")
    from models.t04_10_single_head import validate_task
    validate_task(task)
    if getattr(model, "task", None) != task:
        raise ValueError("Model/task mismatch")
    path = Path(output_dir).resolve()
    latest = path / "last.pth"
    best = path / "best.pth"
    history_path = path / "history.json"
    if resume:
        if not latest.is_file() or not history_path.is_file():
            raise FileNotFoundError("Cannot resume without existing checkpoint/history")
        # Loading a serialized checkpoint requires a trusted source. This
        # function only reads the fixed, user-selected local run directory.
        saved = torch.load(latest, map_location=device, weights_only=False)
        if saved["config_sha256"] != config_sha256 or saved["task"] != task:
            raise ValueError("Resume configuration/task mismatch")
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        if scheduler is not None:
            if saved["scheduler"] is None:
                raise ValueError("Scheduler was missing from saved trajectory")
            scheduler.load_state_dict(saved["scheduler"])
        elif saved["scheduler"] is not None:
            raise ValueError("Scheduler mismatch")
        _restore_rng(saved["rng"])
        history = saved["history"]
        start = int(saved["epoch"])
        best_f1 = float(saved["best_f1"])
        if len(history) != start:
            raise ValueError("History/checkpoint epoch disagreement")
        # Recover from a crash between atomic checkpoint and history writes.
        current_history = json.loads(history_path.read_text())
        if current_history != history:
            history_path.write_text(json.dumps(history, indent=2))
        if start >= epochs:
            raise ValueError("Resume target must exceed last saved epoch")
    else:
        path.mkdir(parents=True, exist_ok=False)
        history, start, best_f1 = [], 0, -1.0
    model.to(device)
    # Reconstruct best epoch from existing history; no checkpoint schema changes.
    best_epoch = next((r['epoch'] for r in history
                       if r['val']['macro_f1'] == best_f1), 0)
    for epoch in range(start + 1, epochs + 1):
        if reset_loader_seed is not None:
            reset_loader_seed(epoch)
        epoch_start = cuda_reset(device)
        step_counts = {}
        train = run_single_head_epoch(train_loader, model, task=task, device=device,
                                      precision=precision, optimizer=optimizer,
                                      telemetry_counts=step_counts)
        val = run_single_head_epoch(val_loader, model, task=task, device=device,
                                    precision=precision)
        # Same data keys as T04-04/T04-07, without mutating the checkpoint.
        telemetry = epoch_telemetry(
            epoch_start, device, optimizer,
            {'samples': train['n'], **step_counts}, val,
            microbatch_size=microbatch_size, accumulation=1)
        if scheduler is not None:
            scheduler.step(val["loss"])
        improved = val["macro_f1"] > best_f1
        if improved:
            best_f1 = val["macro_f1"]
            best_epoch = epoch
        row = {"epoch": epoch, "train": train, "val": val,
               "best_val_macro_f1": best_f1}
        history.append(row)
        state = {"schema": 1, "task": task, "epoch": epoch,
                 "config_sha256": config_sha256,
                 "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                 "scheduler": scheduler.state_dict() if scheduler is not None else None,
                 "rng": _rng_state(), "best_f1": best_f1,
                 "history": history}
        if improved:
            _atomic_save(state, best)
        _atomic_save(state, latest)
        with tempfile.NamedTemporaryFile(mode="w", dir=path, suffix=".tmp",
                                         encoding="utf8", delete=False) as temp:
            temp_path = Path(temp.name)
            json.dump(history, temp, indent=2)
        os.replace(temp_path, history_path)
        # Emit only after checkpoint and history are durable, with no
        # modifications to RNG, optimizer, history, or checkpoint schemas.
        print('[T04-04] epoch telemetry: ' +
              json.dumps(telemetry, sort_keys=True, allow_nan=False), flush=True)
        tag = 'alpha' if task == 'alpha_only' else 'q0'
        print(f'[T04-10] Epoch {epoch} / {epochs}: '
              f'val {tag} macro-F1 {val["macro_f1"]:.5f} '
              f'(best {best_f1:.5f} @ {best_epoch})', flush=True)
    return {"epoch": epochs, "best_val_macro_f1": best_f1,
            "best_path": str(best), "last_path": str(latest),
            "last_sha256": _sha256(latest)}
