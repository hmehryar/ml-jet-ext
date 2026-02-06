import torch
import torch.nn as nn

def _strip_module_prefix(state_dict):
    # Handles DataParallel checkpoints that save keys like "module.xxx"
    if not state_dict:
        return state_dict
    first_key = next(iter(state_dict.keys()))
    if first_key.startswith("module."):
        return {k[len("module."):]: v for k, v in state_dict.items()}
    return state_dict

def load_legacy_checkpoint(legacy_model: nn.Module, ckpt_path: str, map_location="cpu", strict=True):
    """
    Robust loader that supports common checkpoint formats:
      - raw state_dict
      - {"state_dict": ...}
      - {"model_state_dict": ...}
    """
    ckpt = torch.load(ckpt_path, map_location=map_location)
    if isinstance(ckpt, dict):
        if "state_dict" in ckpt:
            sd = ckpt["state_dict"]
        elif "model_state_dict" in ckpt:
            sd = ckpt["model_state_dict"]
        else:
            # sometimes the dict itself is already the state dict
            sd = ckpt
    else:
        sd = ckpt

    sd = _strip_module_prefix(sd)
    legacy_model.load_state_dict(sd, strict=strict)
    return legacy_model

class MLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden_dim: int = 128, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):
        return self.net(x)

class ViTLegacyBranchedAdapter(nn.Module):
    """
    Config 1.b:
      legacy ViT (backbone + legacy heads) frozen
      + new task-specific branches on top of legacy logits bottleneck z0
    """
    def __init__(
        self,
        legacy_model: nn.Module,
        alpha_classes: int = 7,
        q0_classes: int = 5,
        eloss_dim: int = 1,
        branch_hidden_dim: int = 128,
        branch_dropout: float = 0.1,
        use_eloss_branch: bool = True,
    ):
        super().__init__()
        self.legacy = legacy_model

        # Freeze legacy
        for p in self.legacy.parameters():
            p.requires_grad = False
        self.legacy.eval()  # disable dropout/stochastic depth in the frozen model

        # Infer legacy head output dims dynamically
        # Expect legacy forward returns dict with these keys.
        with torch.no_grad():
            self._legacy_alpha_dim = getattr(self.legacy.alpha_head, "out_features", None)
            self._legacy_q0_dim = getattr(self.legacy.q0_head, "out_features", None)
            self._legacy_eloss_dim = getattr(self.legacy.energy_loss_head, "out_features", None)

        if self._legacy_alpha_dim is None or self._legacy_q0_dim is None or self._legacy_eloss_dim is None:
            raise ValueError("Could not infer legacy head dims. Make sure legacy_model has alpha_head/q0_head/energy_loss_head.")

        self.z0_dim = int(self._legacy_alpha_dim + self._legacy_q0_dim + self._legacy_eloss_dim)

        self.use_eloss_branch = use_eloss_branch
        self.alpha_branch = MLP(self.z0_dim, alpha_classes, hidden_dim=branch_hidden_dim, dropout=branch_dropout)
        self.q0_branch    = MLP(self.z0_dim, q0_classes,   hidden_dim=branch_hidden_dim, dropout=branch_dropout)

        if self.use_eloss_branch:
            self.eloss_branch = MLP(self.z0_dim, eloss_dim, hidden_dim=branch_hidden_dim, dropout=branch_dropout)
        else:
            self.eloss_branch = None

    def forward(self, x):
        # legacy is frozen: compute logits without grads
        with torch.no_grad():
            out_legacy = self.legacy(x)
            a = out_legacy["alpha_output"]          # (B, legacy_alpha_dim)
            q = out_legacy["q0_output"]             # (B, legacy_q0_dim)
            e = out_legacy["energy_loss_output"]    # (B, legacy_eloss_dim) usually (B,1)

            # ensure 2D tensors for concat
            if e.dim() == 1:
                e = e.unsqueeze(1)

            z0 = torch.cat([a, q, e], dim=1)        # (B, z0_dim)

        out = {
            "alpha_output": self.alpha_branch(z0),  # (B,7)
            "q0_output": self.q0_branch(z0),        # (B,5)
        }
        if self.use_eloss_branch:
            out["energy_loss_output"] = self.eloss_branch(z0)  # (B,1)
        else:
            # keep key for compatibility if your loop expects it
            out["energy_loss_output"] = e  # legacy eloss logit (frozen)
        return out
