"""Apply T04-07 loss support to exact T04 pre-existing code. Dry-run default."""
from pathlib import Path
import argparse, hashlib, sys

PATCHES = {}

def replace_once(text, old, new, path):
    count=text.count(old)
    if count!=1: raise RuntimeError(f'{path}: expected exactly one anchor; found {count}: {old[:90]!r}')
    return text.replace(old,new,1)

LOSS_CONFIG = '''"""Typed, backward-compatible moment loss configuration (T04-07)."""
import math

KINDS = ('cross_entropy', 'ce_ordinal', 'physical_mae', 'ce_physical_mae')


def resolve_moment_loss(settings=None):
    if settings is None: settings = {}
    if not isinstance(settings, dict): raise ValueError('moment_loss must be a mapping')
    kind = settings.get('type', 'cross_entropy')
    # Do not silently reinterpret historical configurations.
    if kind not in KINDS: raise ValueError(f'Unsupported moment loss.type: {kind}')
    fields = {'cross_entropy': {'type', 'ordinal_weight'}, 'ce_ordinal': {'type', 'ordinal_weight'},
              'physical_mae': {'type'}, 'ce_physical_mae': {'type', 'mae_weight'}}
    extra = set(settings) - fields[kind]
    if extra: raise ValueError(f'Unknown settings for {kind}: {sorted(extra)}')
    weight = settings.get('ordinal_weight', 1.0) if kind == 'ce_ordinal' else settings.get('mae_weight', 1.0)
    if isinstance(weight, bool) or not isinstance(weight, (int, float)) or not math.isfinite(weight) or weight < 0:
        raise ValueError('Loss coefficient must be finite and nonnegative')
    if kind == 'cross_entropy':
        if settings.get('ordinal_weight', 0.0) != 0 or isinstance(settings.get('ordinal_weight', 0.0), bool):
            raise ValueError('CE requires ordinal_weight=0')
        # Preserve historical serialized CE contract for old checkpoints/manifests.
        return {'type': 'cross_entropy', 'ordinal_weight': 0.0,
                'ordinal_normalization': 'adjacent_physical_gap / full_parameter_range'}
    if kind == 'ce_ordinal':
        return {'type': kind, 'ordinal_weight': float(weight),
                'ordinal_normalization': 'adjacent_physical_gap / full_parameter_range'}
    if kind == 'physical_mae':
        return {'type': kind, 'physical_normalization': 'full_parameter_range',
                'physical_objective': 'expected_absolute_deviation'}
    return {'type': kind, 'mae_weight': float(weight),
            'physical_normalization': 'full_parameter_range',
            'physical_objective': 'expected_absolute_deviation'}
'''

PHYSICAL_CLASS='''

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
'''


def transform(path, s):
    if path == 'train_utils/loss_config.py': return LOSS_CONFIG
    if path == 'train_utils/ordinal_loss.py':
        s = replace_once(s,'\ndef build_moment_criterion(settings=None, device=\'cpu\'):',PHYSICAL_CLASS+'\n\ndef build_moment_criterion(settings=None, device=\'cpu\'):',path)
        old="""    if spec['type'] == 'cross_entropy':
        return {key: nn.CrossEntropyLoss().to(device) for key in grids}
    return {key: CEOrdinalLoss(values, spec['ordinal_weight']).to(device)
            for key, values in grids.items()}"""
        new="""    if spec['type'] == 'cross_entropy':
        return {key: nn.CrossEntropyLoss().to(device) for key in grids}
    if spec['type'] == 'ce_ordinal':
        return {key: CEOrdinalLoss(values, spec['ordinal_weight']).to(device)
                for key, values in grids.items()}
    return {key: PhysicalMAELoss(values,
                include_ce=spec['type']=='ce_physical_mae',
                mae_weight=spec.get('mae_weight', 1.0)).to(device)
            for key, values in grids.items()}"""
        s=replace_once(s,old,new,path)
        old="""        else:
            result[key] = {'type':'cross_entropy' if isinstance(loss, nn.CrossEntropyLoss)
                            else type(loss).__name__}"""
        new="""        elif isinstance(loss, PhysicalMAELoss):
            result[key] = dict(resolve_moment_loss({'type': 'ce_physical_mae',
                'mae_weight':loss.mae_weight} if loss.include_ce else {'type':'physical_mae'}),
                class_values=loss.class_values.cpu().tolist())
        else:
            result[key] = {'type':'cross_entropy' if isinstance(loss, nn.CrossEntropyLoss)
                            else type(loss).__name__}"""
        return replace_once(s,old,new,path)
    if path == 'train_utils/t04_preflight.py':
        s=replace_once(s,"LOSSES = {'cross_entropy', 'ce_ordinal', 'ce_soft_mae'}", "LOSSES = {'cross_entropy', 'ce_ordinal', 'physical_mae', 'ce_physical_mae'}",path)
        old="""    if not isinstance(c.moment_loss, dict) or c.moment_loss.get('type') not in LOSSES:
        raise ValueError('Unsupported moment loss')
    if set(c.moment_loss) - {'type', 'ordinal_weight', 'mae_weight'}:
        raise ValueError('Unknown loss settings')
    if c.moment_loss['type'] == 'ce_ordinal':
        _positive(c.moment_loss.get('ordinal_weight'), 'ordinal_weight', allow_zero=True)
    if c.moment_loss['type'] == 'ce_soft_mae':
        _positive(c.moment_loss.get('mae_weight'), 'mae_weight', allow_zero=True)"""
        new="""    from train_utils.loss_config import resolve_moment_loss
    if not isinstance(c.moment_loss, dict) or c.moment_loss.get('type') not in LOSSES:
        raise ValueError('Unsupported moment loss')
    # Centralized strict schema: wrong coefficient name, NaN, infinities and
    # negative values fail before any new run directory is reserved.
    resolve_moment_loss(c.moment_loss)"""
        return replace_once(s,old,new,path)
    if path == 'train_utils/t04_session_worker.py':
        old="""    if cfg.get('moment_loss', {}).get('type') != 'cross_entropy':
        raise NotImplementedError('Additional losses belong to T04-07')"""
        new="""    from train_utils.loss_config import resolve_moment_loss
    resolve_moment_loss(cfg.get('moment_loss', {'type': 'cross_entropy'}))"""
        s=replace_once(s,old,new,path)
        old="""    runtime['adaptation'] = {"""
        new="""    from train_utils.ordinal_loss import criterion_settings
    runtime['moment_loss'] = cfg['moment_loss']
    runtime['criterion_settings'] = criterion_settings(criterion)
    runtime['adaptation'] = {"""
        return replace_once(s,old,new,path)
    if path == 'train_utils/t04_runner.py':
        old="""    if cfg['moment_loss'].get('type') != 'cross_entropy':
        raise NotImplementedError('Additional losses belong to T04-07')"""
        new="""    from train_utils.loss_config import resolve_moment_loss
    resolve_moment_loss(cfg['moment_loss'])"""
        return replace_once(s,old,new,path)
    if path == 'train_utils/t04_precision.py':
        s=replace_once(s,'from train_utils.ordinal_loss import CEOrdinalLoss, physical_class_metrics',
                      'from train_utils.ordinal_loss import CEOrdinalLoss, PhysicalMAELoss, physical_class_metrics',path)
        old="""        else:
            losses[key] = criterion[key](outputs[key], labels[key])
            if isinstance(criterion[key], torch.nn.CrossEntropyLoss):"""
        new="""        elif isinstance(criterion[key], PhysicalMAELoss):
            losses[key], ce, physical = criterion[key].components(outputs[key], labels[key])
            parts[f'loss_{tag}_ce'] = float(ce.detach()) if ce is not None else 0.
            parts[f'loss_{tag}_ordinal'] = 0.
            parts[f'loss_{tag}_physical_mae'] = float(physical.detach())
        else:
            losses[key] = criterion[key](outputs[key], labels[key])
            if isinstance(criterion[key], torch.nn.CrossEntropyLoss):"""
        return replace_once(s,old,new,path)
    if path == 'train_utils/moment_training.py':
        s=replace_once(s,'from train_utils.ordinal_loss import CEOrdinalLoss, physical_class_metrics, criterion_settings',
                      'from train_utils.ordinal_loss import CEOrdinalLoss, PhysicalMAELoss, physical_class_metrics, criterion_settings',path)
        old="""                else:
                    losses[key] = criterion[key](outputs[key], labels[key])
                    if isinstance(criterion[key], torch.nn.CrossEntropyLoss):"""
        new="""                elif isinstance(criterion[key], PhysicalMAELoss):
                    losses[key], ce, physical = criterion[key].components(outputs[key], labels[key])
                    batch_components[f'loss_{tag}_ce'] = float(ce.detach()) if ce is not None else 0.
                    batch_components[f'loss_{tag}_ordinal'] = 0.
                    batch_components[f'loss_{tag}_physical_mae'] = float(physical.detach())
                else:
                    losses[key] = criterion[key](outputs[key], labels[key])
                    if isinstance(criterion[key], torch.nn.CrossEntropyLoss):"""
        return replace_once(s,old,new,path)
    if path == 'train_utils/t04_session_training.py':
        old="""COLUMNS += [f'{sp}_{tag}_{metric}' for sp in ('train','val')"""
        new="""COLUMNS += [f'{sp}_{tag}_physical_mae_loss' for sp in ('train','val') for tag in ('alpha','q0')]
COLUMNS += [f'{sp}_{tag}_{metric}' for sp in ('train','val')"""
        s=replace_once(s,old,new,path)
        old="""            for metric in ('mae','rmse','normalized_mae'):
                row[f'{split}_{tag}_{metric}'] = metrics[tag].get(metric, '')"""
        new="""            row[f'{split}_{tag}_physical_mae_loss'] = metrics.get(f'loss_{tag}_physical_mae', '')
            for metric in ('mae','rmse','normalized_mae'):
                row[f'{split}_{tag}_{metric}'] = metrics[tag].get(metric, '')"""
        return replace_once(s,old,new,path)
    raise KeyError(path)

PATHS = ['train_utils/loss_config.py','train_utils/ordinal_loss.py','train_utils/t04_preflight.py',
         'train_utils/t04_session_worker.py','train_utils/t04_runner.py',
         'train_utils/t04_precision.py','train_utils/moment_training.py',
         'train_utils/t04_session_training.py']

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo',type=Path,default=Path.cwd())
    p.add_argument('--apply',action='store_true',help='Write code changes; otherwise audit only')
    args=p.parse_args(); repo=args.repo.resolve()
    changes={}
    for rel in PATHS:
        path=repo/rel
        s=path.read_text()
        changes[path]=transform(rel,s)
        compile(changes[path],str(path),'exec')
        print(('PREPARED' if args.apply else 'CHECK'),rel,'SHA256_before',hashlib.sha256(s.encode()).hexdigest()[:12])
    if args.apply:
        for path,s in changes.items(): path.write_text(s)
        print('APPLIED',len(changes),'files (no historical run artifacts modified)')
    else:
        print('DRY RUN: all anchors located and all transformed modules compile')

if __name__=='__main__': main()
