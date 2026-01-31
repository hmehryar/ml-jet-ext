# scheduler_utils.py

import torch
from torch.optim.lr_scheduler import (
    ReduceLROnPlateau, StepLR, MultiStepLR, CosineAnnealingLR,
    ExponentialLR, OneCycleLR, CosineAnnealingWarmRestarts
)

def create_scheduler(optimizer, cfg, train_loader=None):
    scheduler_type = cfg.scheduler.get("type", "ReduceLROnPlateau")

    if scheduler_type == "ReduceLROnPlateau":
        return ReduceLROnPlateau(
            optimizer,
            mode=cfg.scheduler.get("mode", "max"),
            factor=cfg.scheduler.get("factor", 0.5),
            patience=cfg.scheduler.get("patience", 4),
            # verbose=cfg.scheduler.get("verbose", True),
            min_lr=cfg.scheduler.get("min_lr", 1e-6)
        )
    elif scheduler_type == "StepLR":
        return StepLR(
            optimizer,
            step_size=cfg.scheduler.get("step_size", 10),
            gamma=cfg.scheduler.get("gamma", 0.1)
        )
    elif scheduler_type == "MultiStepLR":
        return MultiStepLR(
            optimizer,
            milestones=cfg.scheduler.get("milestones", [30, 40]),
            gamma=cfg.scheduler.get("gamma", 0.1)
        )
    elif scheduler_type == "CosineAnnealingLR":
        return CosineAnnealingLR(
            optimizer,
            T_max=cfg.scheduler.get("T_max", cfg.epochs),
            eta_min=cfg.scheduler.get("min_lr", 0)
        )
    elif scheduler_type == "CosineAnnealingWarmRestarts":
        return CosineAnnealingWarmRestarts(
            optimizer,
            T_0=cfg.scheduler.get("T_0", 10),
            T_mult=cfg.scheduler.get("T_mult", 2)
        )
    elif scheduler_type == "ExponentialLR":
        return ExponentialLR(
            optimizer,
            gamma=cfg.scheduler.get("gamma", 0.95)
        )
    elif scheduler_type == "OneCycleLR":
        assert train_loader is not None, "train_loader is required for OneCycleLR"
        return OneCycleLR(
            optimizer,
            max_lr=cfg.scheduler.get("max_lr", cfg.learning_rate),
            epochs=cfg.epochs,
            steps_per_epoch=len(train_loader)
        )
    else:
        raise ValueError(f"Unsupported scheduler type: {scheduler_type}")
