"""Classification with a distance penalty on cumulative class probabilities.

This is a gap-weighted squared-CDF (ranked probability) loss, often called
squared-EMD in ordinal classification. It is not squared Wasserstein-2.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F

from train_utils.loss_config import resolve_moment_loss


class CEOrdinalLoss(nn.Module):
    def __init__(self, class_values, ordinal_weight=1.0):
        super().__init__()
        spec = resolve_moment_loss({'type':'ce_ordinal', 'ordinal_weight':ordinal_weight})
        values = torch.as_tensor(class_values, dtype=torch.float64)
        if values.ndim != 1 or len(values) < 2 or not torch.isfinite(values).all():
            raise ValueError('class_values must be a finite one-dimensional ordered grid')
        gaps = values[1:] - values[:-1]
        if not (gaps > 0).all():
            raise ValueError('class_values must be strictly increasing')
        self.register_buffer('class_values', values)
        self.register_buffer('boundary_weights', gaps / (values[-1] - values[0]))
        self.ordinal_weight = spec['ordinal_weight']

    def components(self, logits, targets):
        if (logits.ndim != 2 or logits.shape[1] != len(self.class_values)
                or targets.ndim != 1 or len(targets) != len(logits)
                or targets.dtype != torch.long):
            raise ValueError('Expected logits [B,K] and long class targets [B] on the ordered grid')
        ce = F.cross_entropy(logits, targets)
        cdf = logits.softmax(dim=1).cumsum(dim=1)[:, :-1]
        boundaries = torch.arange(logits.shape[1] - 1, device=logits.device)
        target_cdf = (boundaries[None, :] >= targets[:, None]).to(logits.dtype)
        weights = self.boundary_weights.to(device=logits.device, dtype=logits.dtype)
        ordinal = ((cdf - target_cdf).square() * weights).sum(dim=1).mean()
        return ce + self.ordinal_weight * ordinal, ce, ordinal

    def forward(self, logits, targets):
        return self.components(logits, targets)[0]



class PhysicalMAELoss(nn.Module):
    """Expected normalized |physical_class - true_physical|; optionally add CE.

    Physical class values are registered as a buffer. Calculations use FP32 for
    fp16/bf16 logits and FP64 for FP64 logits, preserving gradient flow to logits.
    """
    def __init__(self, class_values, *, include_ce=False, mae_weight=1.0):
        super().__init__()
        if isinstance(mae_weight, bool) or not isinstance(mae_weight, (int,float)) or not math.isfinite(mae_weight) or mae_weight < 0:
            raise ValueError('mae_weight must be finite and nonnegative')
        values = torch.as_tensor(class_values, dtype=torch.float64)
        if values.ndim != 1 or values.numel()<2 or not torch.isfinite(values).all() or not torch.all(values[1:] > values[:-1]):
            raise ValueError('Physical class grid must be finite, 1D, strictly increasing')
        self.register_buffer('class_values', values)
        self.include_ce = bool(include_ce)
        self.mae_weight = float(mae_weight)

    def components(self, logits, targets):
        if logits.ndim != 2 or logits.shape[1] != self.class_values.numel() or targets.ndim!=1 or targets.shape[0]!=logits.shape[0] or targets.dtype!=torch.long:
            raise ValueError('Logits [B,K] and int64 targets [B] required')
        if logits.device != targets.device: raise ValueError('Logits and targets must share device')
        if not torch.all((targets>=0)&(targets<self.class_values.numel())):
            raise ValueError('Target class index out of physical grid')
        z = logits.to(dtype=torch.float64 if logits.dtype==torch.float64 else torch.float32)
        grid = self.class_values.to(device=logits.device, dtype=z.dtype)
        distances = (grid.unsqueeze(0) - grid[targets].unsqueeze(1)).abs() / (grid[-1]-grid[0])
        expected = (z.softmax(dim=1) * distances).sum(dim=1).mean()
        ce = F.cross_entropy(z, targets) if self.include_ce else None
        total = ce + self.mae_weight*expected if self.include_ce else expected
        return total, ce, expected

    def forward(self, logits, targets):
        return self.components(logits, targets)[0]


def build_moment_criterion(settings=None, device='cpu'):
    from data.moment_loader import ALPHA_VALUES, Q0_VALUES
    spec = resolve_moment_loss(settings)
    grids = {'alpha_output': ALPHA_VALUES, 'q0_output': Q0_VALUES}
    if spec['type'] == 'cross_entropy':
        return {key: nn.CrossEntropyLoss().to(device) for key in grids}
    if spec['type'] == 'ce_ordinal':
        return {key: CEOrdinalLoss(values, spec['ordinal_weight']).to(device)
                for key, values in grids.items()}
    return {key: PhysicalMAELoss(values,
                include_ce=spec['type']=='ce_physical_mae',
                mae_weight=spec.get('mae_weight', 1.0)).to(device)
            for key, values in grids.items()}


def physical_class_metrics(truth, predictions, class_values):
    """Errors of argmax classes in physical units, not errors of class indices."""
    if not truth or len(truth) != len(predictions):
        raise ValueError('Physical metrics require nonempty aligned class lists')
    errors = [float(class_values[p]) - float(class_values[y])
              for y, p in zip(truth, predictions)]
    mae = sum(abs(e) for e in errors) / len(errors)
    return {'mae': mae, 'rmse': math.sqrt(sum(e * e for e in errors) / len(errors)),
            'normalized_mae': mae / float(class_values[-1] - class_values[0])}


def criterion_settings(criterion):
    result = {}
    for key, loss in criterion.items():
        if isinstance(loss, CEOrdinalLoss):
            result[key] = dict(resolve_moment_loss({'type':'ce_ordinal',
                'ordinal_weight':loss.ordinal_weight}), class_values=loss.class_values.cpu().tolist())
        elif isinstance(loss, PhysicalMAELoss):
            result[key] = dict(resolve_moment_loss({'type': 'ce_physical_mae',
                'mae_weight':loss.mae_weight} if loss.include_ce else {'type':'physical_mae'}),
                class_values=loss.class_values.cpu().tolist())
        else:
            result[key] = {'type':'cross_entropy' if isinstance(loss, nn.CrossEntropyLoss)
                            else type(loss).__name__}
    return result
