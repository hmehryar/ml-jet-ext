"""Idempotent, fail-closed source integration for T04-04 stage 2.

Run from project root after stage 1B integration. Never edit prepared-run snapshots.
"""
from pathlib import Path


def replace(path, old, new):
    value = path.read_text()
    if new in value:
        print('Already installed:',path); return
    if value.count(old)!=1:
        raise RuntimeError('Unexpected source anchor at '+str(path))
    path.write_text(value.replace(old,new,1))
    print('Updated:',path)


def main():
    base=Path.cwd()
    if not (base/'train_utils/t04_precision.py').is_file():
        raise RuntimeError('Install t04_precision.py first')
    worker=base/'train_utils/t04_session_worker.py'
    training=base/'train_utils/t04_session_training.py'
    replace(worker,"from train_utils.t04_session_training import train_session\n",
        "from train_utils.t04_session_training import train_session\nfrom train_utils.t04_precision import make_scaler\n")
    replace(worker,
        "    if cfg.get('gradient_accumulation') != 1 or cfg.get('precision') != 'fp32':\n        raise NotImplementedError('Accumulation and AMP belong to T04-04')\n",
        "    if cfg.get('precision') not in ('fp32','bf16','fp16') or type(cfg.get('gradient_accumulation')) is not int or cfg['gradient_accumulation'] < 1:\n        raise ValueError('Invalid precision or gradient accumulation')\n")
    replace(worker,"    criterion = build_moment_criterion(settings['moment_loss'], device=device)\n",
        "    criterion = build_moment_criterion(settings['moment_loss'], device=device)\n    scaler = make_scaler(device, cfg['precision'])\n")
    replace(worker,"        checkpoint_retention=retention)\n",
        "        checkpoint_retention=retention, scaler=scaler,\n        precision=cfg['precision'], accumulation=cfg['gradient_accumulation'])\n")
    replace(training,"from train_utils.moment_training import train_moment_epoch, evaluate_moment\n",
        "from train_utils.moment_training import train_moment_epoch, evaluate_moment\nfrom train_utils.t04_precision import run_epoch\n")
    replace(training,"                  device, scaler=None, loss_weights=None, checkpoint_retention='last_best'):\n",
        "                  device, scaler=None, loss_weights=None, checkpoint_retention='last_best',\n                  precision='fp32', accumulation=1):\n")
    replace(training,"        tr = train_moment_epoch(train_loader, model, criterion, optimizer, device, loss_weights)\n        va = evaluate_moment(val_loader, model, criterion, device, loss_weights)\n",
        "        if precision == 'fp32' and accumulation == 1:\n            tr = train_moment_epoch(train_loader, model, criterion, optimizer, device, loss_weights)\n            va = evaluate_moment(val_loader, model, criterion, device, loss_weights)\n        else:\n            tr = run_epoch(train_loader, model, criterion, device, loss_weights,\n                           optimizer, precision=precision, accumulation=accumulation, scaler=scaler)\n            va = run_epoch(val_loader, model, criterion, device, loss_weights,\n                           precision=precision)\n")
    print('T04 precision installed. Prepare NEW snapshots only after tests pass.')

if __name__=='__main__': main()
