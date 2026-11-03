"""One-shot, fail-closed source patcher for Stage 3F.2B (generic epoch budget).

Run only at repo root. Changes source code, NOT prepared snapshots, run configs,
registry, sessions or Slurm jobs. Refuses unknown input versions.
"""
from pathlib import Path
import ast
import difflib
import sys

TRAIN = Path('train_utils/t04_session_training.py')
WORKER = Path('train_utils/t04_session_worker.py')


def replace_once(text, old, new, where):
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f'{where}: expected exactly one matching snippet; found {count}')
    return text.replace(old, new, 1)


def make_train(src):
    src = replace_once(src,
        'def train_session(*, run, session, mode, target_epochs, seed, min_epochs, patience,',
        'def train_session(*, run, session, mode, target_epochs, seed, min_epochs, patience,', 'training signature')
    src = replace_once(src,
        "precision='fp32', accumulation=1, microbatch_size=None):",
        "precision='fp32', accumulation=1, microbatch_size=None,\n                   protocol_id='early_stop_v1', protocol_hash=None, configured_epochs=None):", TRAIN)
    # Import close to function, avoids touching existing import layout.
    src = replace_once(src,
        "    run, session = Path(run).resolve(), Path(session).resolve()\n    if checkpoint_retention not in ('last_best', 'all'):",
        "    from train_utils.t04_training_termination import (\n"
        "        validate_protocol, may_resume, should_stop)\n"
        "    validate_protocol(protocol_id, protocol_hash, configured_epochs)\n"
        "    run, session = Path(run).resolve(), Path(session).resolve()\n"
        "    if checkpoint_retention not in ('last_best', 'all'):", TRAIN)
    src = replace_once(src,
        "        if state.get('stopped'):\n            raise ResumeError('Cannot resume a training trajectory terminated by early stopping')",
        "        if not may_resume(protocol_id=protocol_id, protocol_hash=protocol_hash,\n"
        "                          epochs=configured_epochs, stopped=bool(state.get('stopped'))):\n"
        "            raise ResumeError('Cannot resume a training trajectory terminated by early stopping')", TRAIN)
    src = replace_once(src,
        '        stop = epoch >= min_epochs and stagnant >= patience',
        '        stop = should_stop(protocol_id=protocol_id, protocol_hash=protocol_hash,\n'
        '                           epoch=epoch, min_epochs=min_epochs, stagnant=stagnant,\n'
        '                           patience=patience, epochs=configured_epochs)', TRAIN)
    # Do not accidentally run a fixed-budget segment past configured max.
    src = replace_once(src,
        '    if start_epoch >= target_epochs:\n        raise ResumeError',
        "    if protocol_id == 'fixed_budget_v2' and target_epochs > configured_epochs:\n"
        "        raise ResumeError('fixed_budget_v2 target exceeds configured epochs')\n"
        '    if start_epoch >= target_epochs:\n        raise ResumeError', TRAIN)
    return src


def make_worker(src):
    src = replace_once(src,
        "    cfg = json.loads((run / 'config.resolved.json').read_text())\n    validate_strategy_settings(cfg)",
        "    cfg = json.loads((run / 'config.resolved.json').read_text())\n"
        "    from train_utils.t04_training_termination import validate_protocol\n"
        "    protocol_id = cfg.get('training_protocol', 'early_stop_v1')\n"
        "    protocol_hash = cfg.get('training_protocol_hash')\n    configured_epochs = cfg['epochs']\n"
        "    validate_protocol(protocol_id, protocol_hash, configured_epochs)\n"
        "    validate_strategy_settings(cfg)", WORKER)
    src = replace_once(src,
        "        patience=cfg['patience'], train_loader=train_loader, val_loader=val_loader,",
        "        patience=cfg['patience'], protocol_id=protocol_id, protocol_hash=protocol_hash,\n        configured_epochs=configured_epochs,\n"
        "        train_loader=train_loader, val_loader=val_loader,", WORKER)
    return src


def main():
    if not (TRAIN.is_file() and WORKER.is_file()):
        raise SystemExit('Run from the ml-jet-ext repository root')
    originals = {TRAIN: TRAIN.read_text(), WORKER: WORKER.read_text()}
    updated = {TRAIN: make_train(originals[TRAIN]), WORKER: make_worker(originals[WORKER])}
    for path, result in updated.items():
        ast.parse(result, filename=str(path))
    if '--check' in sys.argv:
        for path in (TRAIN, WORKER):
            print(''.join(difflib.unified_diff(originals[path].splitlines(True),
                  updated[path].splitlines(True), fromfile=str(path), tofile=str(path))))
        print('CHECK_ONLY source patch applicable; no changes')
        return
    if '--apply' not in sys.argv:
        raise SystemExit('Use --check first, then --apply')
    for path in (TRAIN, WORKER):
        if path.read_text() != originals[path]:
            raise RuntimeError(f'File changed during patch: {path}')
    for path in (TRAIN, WORKER):
        path.write_text(updated[path])
        print('PATCHED', path)


if __name__ == '__main__':
    main()
