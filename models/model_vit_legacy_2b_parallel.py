import torch
import torch.nn as nn
import torch.nn.functional as F

def _strip_module_prefix(state_dict):
    if not state_dict:
        return state_dict
    first_key = next(iter(state_dict.keys()))
    if first_key.startswith("module."):
        return {k[len("module."):]: v for k, v in state_dict.items()}
    return state_dict


def load_legacy_checkpoint(legacy_model: nn.Module, ckpt_path: str, map_location="cpu", strict=True):
    """
    Supports:
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
            sd = ckpt
    else:
        sd = ckpt

    sd = _strip_module_prefix(sd)
    legacy_model.load_state_dict(sd, strict=strict)
    return legacy_model


class MLPHead(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden_dim: int = 256, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):
        return self.net(x)


class ViTLegacyParallelHeads(nn.Module):
    """
    Config 2.b:
      - load legacy ViT
      - reuse backbone features
      - discard old task heads
      - attach fresh parallel heads for new alpha/q0/eloss tasks
      - backbone frozen by default
    """
    def __init__(
        self,
        legacy_model: nn.Module,
        alpha_classes: int = 7,
        q0_classes: int = 5,
        eloss_dim: int = 1,
        head_hidden_dim: int = 256,
        head_dropout: float = 0.1,
        freeze_backbone: bool = True,
    ):
        super().__init__()

        # reuse the pretrained feature extractor
        self.legacy = legacy_model
        self.features = self.legacy.features

        # infer feature dimension
        feat_dim = getattr(self.features, "num_features", None)
        if feat_dim is None:
            feat_dim = getattr(self.features, "embed_dim", None)
        if feat_dim is None:
            raise ValueError("Could not infer backbone feature dimension from legacy_model.features")

        self.feat_dim = feat_dim
        self.freeze_backbone = freeze_backbone

        if self.freeze_backbone:
            for p in self.features.parameters():
                p.requires_grad = False
            self.features.eval()

        self.alpha_head = MLPHead(
            in_dim=self.feat_dim,
            out_dim=alpha_classes,
            hidden_dim=head_hidden_dim,
            dropout=head_dropout,
        )
        self.q0_head = MLPHead(
            in_dim=self.feat_dim,
            out_dim=q0_classes,
            hidden_dim=head_hidden_dim,
            dropout=head_dropout,
        )
        self.energy_loss_head = MLPHead(
            in_dim=self.feat_dim,
            out_dim=eloss_dim,
            hidden_dim=head_hidden_dim,
            dropout=head_dropout,
        )

    # def extract_features(self, x):
    #     if self.freeze_backbone:
    #         with torch.no_grad():
    #             feats = self.features(x)
    #     else:
    #         feats = self.features(x)
    #     return feats
    def extract_features(self, x):
        # Match the legacy ViT preprocessing exactly
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)

        x = F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)

        if self.freeze_backbone:
            with torch.no_grad():
                feats = self.features.forward_features(x)
        else:
            feats = self.features.forward_features(x)

        # timm ViT forward_features may return [B, N, C] or [B, C]
        if feats.ndim == 3:
            feats = feats[:, 0]   # CLS token

        return feats
    def forward(self, x):
        feats = self.extract_features(x)
        return {
            "alpha_output": self.alpha_head(feats),
            "q0_output": self.q0_head(feats),
            "energy_loss_output": self.energy_loss_head(feats),
        }