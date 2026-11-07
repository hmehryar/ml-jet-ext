"""T04-only precision and sample-correct gradient accumulation.

Legacy T03 moment_training is deliberately unchanged.
"""
from contextlib import nullcontext
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
from train_utils.moment_training import HEADS, _check_batch
from train_utils.ordinal_loss import CEOrdinalLoss, PhysicalMAELoss, physical_class_metrics


def autocast_context(device, precision):
    if precision not in ('fp32', 'bf16', 'fp16'):
        raise ValueError('Unknown precision')
    if precision == 'fp32':
        return nullcontext()
    if torch.device(device).type != 'cuda':
        raise ValueError('AMP requires CUDA')
    dtype = torch.bfloat16 if precision == 'bf16' else torch.float16
    return torch.autocast(device_type='cuda', dtype=dtype)


def make_scaler(device, precision):
    if precision == 'fp16':
        if torch.device(device).type != 'cuda':
            raise ValueError('FP16 requires CUDA')
        return torch.amp.GradScaler('cuda')
    return None


def accumulation_windows(n_batches, accumulation):
    if type(accumulation) is not int or accumulation < 1 or n_batches < 1:
        raise ValueError('Need positive accumulation and nonempty dataloader')
    return [(i, min(i+accumulation, n_batches)) for i in range(0, n_batches, accumulation)]


def _losses(outputs, labels, criterion, weights):
    _check_batch(outputs, labels)
    losses, parts = {}, {}
    for key, tag in zip(HEADS, ('alpha','q0')):
        if isinstance(criterion[key], CEOrdinalLoss):
            losses[key], ce, ordinal = criterion[key].components(outputs[key], labels[key])
            parts[f'loss_{tag}_ce'] = float(ce.detach())
            parts[f'loss_{tag}_ordinal'] = float(ordinal.detach())
        elif isinstance(criterion[key], PhysicalMAELoss):
            losses[key], ce, physical = criterion[key].components(outputs[key], labels[key])
            parts[f'loss_{tag}_ce'] = float(ce.detach()) if ce is not None else 0.
            parts[f'loss_{tag}_ordinal'] = 0.
            parts[f'loss_{tag}_physical_mae'] = float(physical.detach())
        else:
            losses[key] = criterion[key](outputs[key], labels[key])
            if isinstance(criterion[key], torch.nn.CrossEntropyLoss):
                parts[f'loss_{tag}_ce'] = float(losses[key].detach())
                parts[f'loss_{tag}_ordinal'] = 0.
    combined = sum(float(weights.get(key, 1.)) * losses[key] for key in HEADS)
    if not torch.isfinite(combined).all():
        raise FloatingPointError('Nonfinite T04 training loss')
    return combined, losses, parts


def run_epoch(loader, model, criterion, device, loss_weights=None, optimizer=None,
              *, precision='fp32', accumulation=1, scaler=None):
    """One T04 epoch. Each window computes a sample-mean gradient, including short final windows.

    Requires mean-reduced per-microbatch criterion losses (the T04 criterion contract).
    No optimizer step occurs in validation; FP16 overflow skips are delegated to GradScaler.
    """
    n_batches = len(loader)
    accumulation_windows(n_batches, accumulation)
    training = optimizer is not None
    if precision == 'fp16' and training and scaler is None:
        raise ValueError('FP16 training requires a GradScaler')
    if precision != 'fp16' and scaler is not None:
        raise ValueError('Scaler must only be used for FP16')
    if any(k == 'energy_loss_output' for k in (loss_weights or {})):
        raise ValueError('Moment dataset does not support energy-loss weights')
    model.train(training)
    weights = loss_weights or {}
    totals = {'loss':0., 'loss_alpha':0., 'loss_q0':0.}
    parts_total = {}
    truths = {k:[] for k in HEADS}; predictions = {k:[] for k in HEADS}
    both = total = steps = 0
    completed_steps = skipped_steps = 0
    # Backpropagate each microbatch immediately instead of retaining an entire
    # accumulation window of computation graphs on the accelerator.
    window_batches = 0
    window_samples = 0
    if training:
        optimizer.zero_grad(set_to_none=True)

    def flush():
        nonlocal steps, window_batches, window_samples, completed_steps, skipped_steps
        if not window_batches:
            return
        # For FP16, unscale before averaging the accumulated sample-sum
        # gradients. GradScaler.step() detects nonfinite gradients and skips
        # the underlying optimizer update when necessary.
        if scaler is not None:
            scaler.unscale_(optimizer)
        for group in optimizer.param_groups:
            for parameter in group['params']:
                if parameter.grad is not None:
                    parameter.grad.div_(window_samples)
        if scaler is None:
            optimizer.step()
            completed_steps += 1
        else:
            # Normal optimizer.step may return None, so detect skipped updates
            # using the dynamic loss-scale decrease after scaler.update().
            previous_scale = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            if scaler.get_scale() < previous_scale:
                skipped_steps += 1
            else:
                completed_steps += 1
        optimizer.zero_grad(set_to_none=True)
        steps += 1  # accumulation windows attempted
        window_batches = 0
        window_samples = 0

    with torch.set_grad_enabled(training):
        for images, batch in loader:
            images = images.to(device, non_blocking=True)
            labels = {key:batch[key].to(device, non_blocking=True) for key in HEADS}
            with autocast_context(device, precision):
                outputs = model(images)
                loss, losses, parts = _losses(outputs, labels, criterion, weights)
            b = images.shape[0]; total += b
            totals['loss'] += float(loss.detach())*b
            totals['loss_alpha'] += float(losses['alpha_output'].detach())*b
            totals['loss_q0'] += float(losses['q0_output'].detach())*b
            for key, val in parts.items():
                parts_total[key] = parts_total.get(key,0.)+val*b
            pred = {key: outputs[key].argmax(1) for key in HEADS}
            both += ((pred['alpha_output']==labels['alpha_output']) & (pred['q0_output']==labels['q0_output'])).sum().item()
            for key in HEADS:
                truths[key].extend(labels[key].detach().cpu().tolist())
                predictions[key].extend(pred[key].detach().cpu().tolist())
            if training:
                # Criteria are mean-reduced: accumulate the sum of per-sample
                # losses, and divide gradients by the actual sample count in
                # this window, including any final incomplete window.
                summed_loss = loss * b
                if scaler is None:
                    summed_loss.backward()
                else:
                    scaler.scale(summed_loss).backward()
                window_batches += 1
                window_samples += b
                if window_batches == accumulation:
                    flush()
        if training and window_batches:
            flush()
    result = {key:val/total for key,val in totals.items()}
    result.update({key:val/total for key,val in parts_total.items()})
    result['accuracy'] = both/total
    for key, tag, count in (('alpha_output','alpha',7),('q0_output','q0',5)):
        y,p = truths[key],predictions[key]
        result[tag] = {
            'accuracy': float(accuracy_score(y,p)),
            'precision':float(precision_score(y,p,average='macro',labels=list(range(count)),zero_division=0)),
            'recall':float(recall_score(y,p,average='macro',labels=list(range(count)),zero_division=0)),
            'f1':float(f1_score(y,p,average='macro',labels=list(range(count)),zero_division=0)),
            'confusion_matrix':confusion_matrix(y,p,labels=list(range(count))).tolist(),
        }
        from data.moment_loader import ALPHA_VALUES,Q0_VALUES
        result[tag].update(physical_class_metrics(y,p,ALPHA_VALUES if tag=='alpha' else Q0_VALUES))
        result[f'accuracy_{tag}']=result[tag]['accuracy']
    result['optimizer_steps']=steps
    result['optimizer_steps_completed']=completed_steps
    result['optimizer_steps_skipped']=skipped_steps
    result['microbatches']=n_batches
    result['samples']=total
    result['effective_batch_mean']=total/steps if steps else None
    return result
