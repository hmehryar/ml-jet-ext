"""Stage 5A: CPU-only, fail-closed guard for submitting held T04-10 jobs.

No Slurm commands, GPU loads, or filesystem writes. It validates a prepared
T04-10 run and its frozen matrix identity before job allocation.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

T04_ROOT = Path('/rs/rs_grp_mljet/ml-jet-ext/training_output/T04/t04_10_single_head')
T04_07 = Path('/rs/rs_grp_mljet/ml-jet-ext/training_output/T04/t04_07_top4_production')
MODELS = {'clip', 'siglip'}
TASKS = {'alpha_only', 'q0_only'}
SEEDS = {42, 123, 2026}


def _assert(ok, message):
    if not ok:
        raise ValueError(message)


def guard(prepared_run, mode, *, root=T04_ROOT, verifier=None):
    """Return frozen run metadata after strict verification, never write."""
    _assert(mode in ('fresh', 'resume'), 'Mode must be fresh or resume')
    root = Path(root).resolve()
    run = Path(prepared_run).expanduser().resolve(strict=True)
    _assert(run.is_dir(), 'Prepared run is not a directory')
    _assert(run != root and run.is_relative_to(root), 'Run outside dedicated T04-10 output root')
    _assert(not run.is_relative_to(T04_07.resolve()), 'T04-07 run forbidden')
    _assert((run/'run_manifest.json').is_file(), 'Missing prepared manifest')
    _assert((run/'config.resolved.json').is_file(), 'Missing resolved config')

    manifest = json.loads((run/'run_manifest.json').read_text())
    cfg = json.loads((run/'config.resolved.json').read_text())
    _assert(manifest.get('issue') == 30 and manifest.get('schema') == 1,
            'Not a T04-10 prepared-run manifest')
    _assert(manifest.get('status') == 'prepared', 'Unexpected preparation state')
    _assert(manifest.get('git_branch') == 'feat/t04-10-single-head-pilot',
            'Prepared run came from another branch')
    _assert(isinstance(manifest.get('git_commit'), str) and len(manifest['git_commit']) == 40,
            'Missing source commit')
    model, task, seed = cfg.get('model'), cfg.get('task'), cfg.get('seed')
    moments = cfg.get('moments')
    _assert(model in MODELS and task in TASKS and type(seed) is int and seed in SEEDS,
            'Unknown model, task, or seed')
    _assert(isinstance(moments, list) and 1 <= len(moments) <= 5, 'Invalid moments')
    _assert(cfg.get('epochs') == 80 and cfg.get('training_protocol') == 'fixed_budget_v2',
            'Production training protocol changed')
    _assert(Path(cfg['output_root']).expanduser().resolve() == root,
            'Config output_root is not the dedicated T04-10 root')
    relative = Path(model)/task/f'E{len(moments)}'/f'seed-{seed}'/manifest.get('run_id','')
    _assert(run == root/relative, 'Prepared run does not match canonical T04-10 directory layout')
    _assert(manifest.get('test_set_evaluated') is False, 'Test set evaluation must be disabled')
    overlap = manifest.get('inputs', {}).get('source_overlap')
    _assert(type(overlap) is int and overlap >= 0, 'Missing source overlap disclosure')
    _assert(cfg.get('split_protocol') == 'aggregate_disjoint' or overlap == 0,
            'False source-disjoint claim')
    if verifier is None:
        from train_utils.t04_10_preflight import verify
        verifier = verify
    _assert(verifier(run) is True, 'Immutable snapshot verifier failed')

    # Reject legacy flat-run directories/snapshots; never convert a used run in place.
    _assert((run/'source/train_utils/t04_10_sessions.py').is_file() and
            (run/'source/train_utils/t04_10_session_training.py').is_file(),
            'T04-10 session production source missing; prepare from Stage 6 commit')
    _assert(not (run/'training').exists() and not (run/'training.lock').exists(),
            'Legacy flat-training run is not eligible for Stage 6')
    from train_utils.t04_10_sessions import latest_checkpoint
    pointer = latest_checkpoint(run)
    session_root = run/'sessions'
    if mode == 'fresh':
        _assert(pointer is None and not session_root.exists(),
                'Fresh mode refuses previously used session experiment')
    else:
        _assert(pointer is not None, 'Resume requires verified committed session checkpoint')
        _assert(pointer['epoch'] < cfg['epochs'], 'Resume target must exceed checkpoint epoch')
    return {'status':'GUARD_PASS', 'run':str(run), 'mode':mode,
            'model':model, 'task':task, 'seed':seed, 'source_overlap':overlap,
            'git_commit':manifest['git_commit']}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', required=True)
    p.add_argument('--mode', choices=['fresh','resume'], required=True)
    a=p.parse_args()
    print(json.dumps(guard(a.run, a.mode), sort_keys=True))

if __name__=='__main__':
    main()
