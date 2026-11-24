"""T04-only runtime telemetry. Never modifies frozen experiment manifests."""
import json
import os
from pathlib import Path
import resource
import socket
import time
import torch


def cuda_reset(device):
    if torch.device(device).type == 'cuda':
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    return time.perf_counter()


def epoch_telemetry(start, device, optimizer, train, val, microbatch_size, accumulation):
    if torch.device(device).type == 'cuda':
        torch.cuda.synchronize(device)
    seconds = time.perf_counter() - start
    lrs = {group.get('name', 'group_%d' % i): float(group['lr']) for i,group in enumerate(optimizer.param_groups)}
    if len(lrs) != len(optimizer.param_groups):
        raise RuntimeError('Duplicate optimizer group labels')
    count = int(train.get('samples', 0))
    data = {'elapsed_seconds': seconds, 'samples': count,
            'samples_per_second': count / seconds if count and seconds > 0 else None,
            'microbatch_size': microbatch_size, 'gradient_accumulation': accumulation,
            'configured_effective_batch': (microbatch_size * accumulation
                                           if microbatch_size is not None else None),
            'optimizer_steps_attempted': train.get('optimizer_steps'),
            'optimizer_steps_completed': train.get('optimizer_steps_completed'),
            'optimizer_steps_skipped': train.get('optimizer_steps_skipped'),
            'learning_rates': lrs, 'hostname': socket.gethostname(),
            'host_maxrss_kib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            'gpu_name': None, 'cuda_peak_allocated_bytes': None,
            'cuda_peak_reserved_bytes': None}
    if torch.device(device).type == 'cuda':
        data.update(gpu_name=torch.cuda.get_device_name(device),
                    cuda_peak_allocated_bytes=int(torch.cuda.max_memory_allocated(device)),
                    cuda_peak_reserved_bytes=int(torch.cuda.max_memory_reserved(device)))
    return data


def save_session_epoch_telemetry(session, epoch, payload):
    """New, session-local sidecar; never rewrite historical CSV or checkpoint schema."""
    directory = Path(session) / 'telemetry'
    directory.mkdir(exist_ok=True)
    path = directory / ('epoch_%06d.json' % int(epoch))
    with path.open('x') as f:
        json.dump({'epoch': int(epoch), **payload}, f, indent=2, allow_nan=False)
        f.write('\n')
        f.flush()
        os.fsync(f.fileno())
    return path
