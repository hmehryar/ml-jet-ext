"""Quarantine only the four unsubmitted v2 prepared snapshots, preserving evidence."""
from __future__ import annotations
import argparse
from datetime import datetime,timezone
from pathlib import Path
import json
import os
import re
from tools.t04_06_registry import locked, read
from tools.t04_06_v2_full_horizon import DEFAULT, _require_v2, fence
from train_utils.t04_preflight import verify
from train_utils.t04_sessions import latest_checkpoint

IDS = ('clip_s1_e1_seed123','clip_s1_e1_seed2026','clip_s1_e1_seed42','clip_s1_e2_seed123')

def quarantine(runtime, apply=False):
    runtime=Path(runtime).resolve()
    with locked(runtime.parent/'stage3f_v2.orchestrator'):
        doc=read(runtime)
        _require_v2(doc,runtime)
        targets=[]
        for name in IDS:
            item=doc['entries'][name]
            if item['status']!='planned' or item['attempts']:
                raise RuntimeError('Not an unsubmitted planned entry: '+name)
            run=Path(item['run']).resolve()
            if not run.is_dir():
                raise RuntimeError('Prepared run missing: '+str(run))
            expected=Path(doc['repo']).resolve()/'training_output/T04/t04_06_v2_production'
            if not run.is_relative_to(expected.resolve()):
                raise RuntimeError('Unexpected output root: '+str(run))
            verify(run)
            if (run/'sessions').exists() or latest_checkpoint(run) is not None:
                raise RuntimeError('Sessions/checkpoints present; refusing: '+name)
            if any(runtime.parent.glob('stage3f_v2.'+name+'.*.intent.json')):
                raise RuntimeError('Submission intent found: '+name)
            config=json.loads((run/'config.resolved.json').read_text())
            if config.get('training_protocol')!='fixed_budget_v2':
                raise RuntimeError('Not fixed_budget_v2: '+name)
            targets.append((name,run))
        # Move to a safe quarantine outside the production output root, not delete.
        stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
        quarantine_dir=runtime.parent / ('quarantined_prepared_'+stamp)
        if quarantine_dir.exists():
            raise FileExistsError(quarantine_dir)
        for name,run in targets:
            print('QUARANTINE_CANDIDATE',name,run,'->',quarantine_dir/name)
        if not apply:
            print('CHECK_ONLY no filesystem mutations')
            return
        quarantine_dir.mkdir(mode=0o700)
        for name,run in targets:
            dest=quarantine_dir/name
            os.rename(run,dest)
            print('QUARANTINED',name,dest)
        print('QUARANTINED_WAVE',len(targets),'registry remains unchanged')

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--runtime',type=Path,default=DEFAULT)
    mode=p.add_mutually_exclusive_group(required=True)
    mode.add_argument('--check',action='store_true')
    mode.add_argument('--apply',action='store_true')
    a=p.parse_args()
    quarantine(a.runtime,apply=a.apply)
