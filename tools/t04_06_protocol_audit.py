"""Read-only fixed_budget_v2 migration audit for the immutable T04-06 registry.

No checkpoint loading, registry changes, Slurm mutations, or submission.
"""
from __future__ import annotations
import argparse
from collections import Counter
import json
from pathlib import Path
from train_utils.t04_training_policy import policy
from tools.t04_06_registry import read, sha
from tools.t04_06_reconcile import query_sacct

DEFAULT = Path('training_output/T04/t04_06/registry.runtime.json')
ACTIVE = frozenset(('submitted', 'running', 'resumed'))


def inspect(doc, slurm=None):
    """Classify safety of each entry without changing the original registry."""
    result = []
    entries = doc['entries']
    if len(entries) != 120:
        raise ValueError('Expected exactly 120 registry experiments')
    active_ids = []
    for ident, entry in sorted(entries.items()):
        if entry['status'] in ACTIVE:
            if not entry['attempts']:
                raise ValueError('Active experiment without Slurm attempt: ' + ident)
            active_ids.append(entry['attempts'][-1]['slurm_job_id'])
    if active_ids and slurm is None:
        raise ValueError('Slurm evidence required for active entries')
    slurm = slurm or {}
    for ident, entry in sorted(entries.items()):
        state = entry['status']
        job_id = entry['attempts'][-1]['slurm_job_id'] if entry['attempts'] else None
        item = {'id':ident, 'state':state, 'last_job_id':job_id,
                'legacy_run':entry['run'], 'v2_action':None,
                'blocker':None, 'observed_slurm':None}
        if state in ACTIVE:
            observed = slurm.get(job_id)
            item['observed_slurm'] = observed
            item['blocker'] = ('ACTIVE_OR_UNRECONCILED: reconcile the legacy attempt first; '
                               'never migrate based on an observed Slurm terminal state alone')
        elif state == 'planned':
            if entry['attempts']:
                item['blocker'] = 'PLANNED_HAS_ATTEMPTS'
            else:
                item['v2_action'] = 'prepare_fresh_v2_after_versioned_registry_creation'
        elif state in ('ready_to_resume', 'completed'):
            if not entry['attempts']:
                item['blocker'] = 'MISSING_PARENT_ATTEMPT'
            else:
                item['v2_action'] = 'verify_parent_checkpoint_then_fork_v2'
        else:
            item['blocker'] = 'REQUIRES_MANUAL_FAILURE_RECOVERY'
        result.append(item)
    return result


def audit(path, query=query_sacct, epochs=80):
    doc = read(Path(path).resolve())
    ids = [e['attempts'][-1]['slurm_job_id'] for e in doc['entries'].values()
           if e['status'] in ACTIVE and e['attempts']]
    observations = query(ids) if ids else {}
    return {'protocol':policy(epochs).protocol_id, 'protocol_sha256':policy(epochs).digest(), 'epochs':epochs,
            'registry_path':str(Path(path).resolve()), 'registry_sha256':sha(path),
            'counts':dict(sorted(Counter(e['status'] for e in doc['entries'].values()).items())),
            'entries':inspect(doc,observations)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=('preview',))
    parser.add_argument('--runtime',type=Path,default=DEFAULT)
    parser.add_argument('--epochs',type=int,default=80,help='Configured training budget for protocol fingerprint')
    parser.add_argument('--include-all',action='store_true',
                        help='Print all 120 records, including untouched planned runs')
    args = parser.parse_args()
    report = audit(args.runtime, epochs=args.epochs)
    print('FIXED_BUDGET_V2_READ_ONLY',report['protocol_sha256'], 'EPOCHS', report['epochs'])
    print('LEGACY_REGISTRY_SHA256',report['registry_sha256'])
    print('COUNTS',json.dumps(report['counts'],sort_keys=True))
    for item in report['entries']:
        if args.include_all or item['state'] != 'planned':
            print(json.dumps(item,sort_keys=True))
    print('PREVIEW_ONLY no filesystem, registry, or Slurm mutations')


if __name__=='__main__':
    main()
