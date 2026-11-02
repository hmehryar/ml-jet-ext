"""Apply T04-04 optimizer integration to an exact T04-03 baseline (idempotent).

Run from repo root. Refuses unrecognized source rather than corrupting a checkout.
"""
from pathlib import Path


def replace_once(path, old, new):
    text = path.read_text()
    if new in text:
        print('Already applied:', path)
        return
    hits = text.count(old)
    if hits != 1:
        raise RuntimeError('Expected one baseline anchor in %s; got %s' % (path, hits))
    path.write_text(text.replace(old, new, 1))
    print('Updated:', path)


def main():
    root = Path.cwd()
    worker = root / 'train_utils/t04_session_worker.py'
    preflight = root / 'train_utils/t04_preflight.py'
    if not (root / 'train_utils/t04_optimizers.py').is_file():
        raise RuntimeError('Stage 1 t04_optimizers.py must be installed first')
    replace_once(worker,
        'from train_utils.t04_session_training import train_session\n',
        'from train_utils.t04_session_training import train_session\n'
        'from train_utils.t04_optimizers import build_t04_optimizer, learning_rate_summary\n')
    replace_once(worker,
        "    rates = cfg.get('learning_rates', {})\n    if rates.get('adapter') != rates.get('heads'):\n        raise NotImplementedError('Separate adapter/head learning rates belong to T04-04')\n    if strategy != 'S1' and rates.get('backbone') != rates.get('heads'):\n        raise NotImplementedError('Separate backbone learning rates belong to T04-04')\n    # The optimizer is Adam with one rate across all trainable parameters.\n    return strategy\n",
        "    rates = cfg.get('learning_rates', {})\n"
        "    if cfg.get('optimizer') is None:\n"
        "        # Historical T04-03 behavior remains unchanged for old manifests.\n"
        "        if rates.get('adapter') != rates.get('heads'):\n"
        "            raise NotImplementedError('Separate adapter/head learning rates require optimizer configuration')\n"
        "        if strategy != 'S1' and rates.get('backbone') != rates.get('heads'):\n"
        "            raise NotImplementedError('Separate backbone learning rates require optimizer configuration')\n"
        "    return strategy\n")
    replace_once(worker,
        "    if getattr(model, 'strategy', None) != strategy:\n",
        "    if cfg.get('optimizer') is not None:\n"
        "        spec = cfg['optimizer']\n"
        "        optimizer = build_t04_optimizer(\n"
        "            model, cfg['learning_rates'], optimizer_type=spec['type'],\n"
        "            weight_decay=spec['weight_decay'])\n"
        "    if getattr(model, 'strategy', None) != strategy:\n")
    replace_once(worker,
        "        'optimizer_trainable_parameter_count': sum(\n            p.numel() for group in optimizer.param_groups for p in group['params'])}\n",
        "        'optimizer_trainable_parameter_count': sum(\n            p.numel() for group in optimizer.param_groups for p in group['params']),\n"
        "        'optimizer_type': cfg.get('optimizer', {}).get('type', 'adam') if cfg.get('optimizer') else 'adam',\n"
        "        'optimizer_learning_rates': learning_rate_summary(optimizer) if cfg.get('optimizer') else\n"
        "            {'legacy': float(optimizer.param_groups[0]['lr'])}}\n")
    replace_once(preflight,
        "    split_protocol: str = 'source_disjoint'\n",
        "    split_protocol: str = 'source_disjoint'\n    optimizer: dict | None = None\n")
    replace_once(preflight,
        "    required = set(T04Config.__dataclass_fields__) - {'fold', 'split_protocol'}\n",
        "    required = set(T04Config.__dataclass_fields__) - {'fold', 'split_protocol', 'optimizer'}\n")
    replace_once(preflight,
        "    for key, value in c.learning_rates.items():\n        _positive(value, f'learning_rates.{key}')\n",
        "    for key, value in c.learning_rates.items():\n        _positive(value, f'learning_rates.{key}')\n"
        "    if c.optimizer is not None:\n"
        "        if not isinstance(c.optimizer, dict) or set(c.optimizer) != {'type', 'weight_decay'}:\n"
        "            raise ValueError('optimizer must contain exactly type and weight_decay')\n"
        "        if c.optimizer['type'] not in ('adam', 'adamw'):\n"
        "            raise ValueError('optimizer.type must be adam or adamw')\n"
        "        import math\n"
        "        decay = c.optimizer['weight_decay']\n"
        "        if isinstance(decay, bool) or not isinstance(decay, (int, float)) or not math.isfinite(decay) or decay < 0:\n"
        "            raise ValueError('optimizer.weight_decay must be finite and nonnegative')\n"
        "        if c.optimizer['type'] == 'adam' and decay != 0:\n"
        "            raise ValueError('Legacy Adam requires weight_decay=0')\n")
    print('Integration installed; run tests before preparing NEW immutable experiments.')


if __name__ == '__main__':
    main()
