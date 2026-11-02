#!/usr/bin/env python3
"""Prepare and launch new immutable T04-04 pilots; never mutate frozen sources."""
import argparse
import os
from pathlib import Path
import subprocess
import sys
from train_utils.t04_preflight import prepare, preflight, verify
from train_utils.t04_sessions import experiment_lock, start_session


def main():
    p=argparse.ArgumentParser()
    sub=p.add_subparsers(dest='action',required=True)
    a=sub.add_parser('prepare');a.add_argument('--config',required=True);a.add_argument('--run-id',required=True)
    b=sub.add_parser('execute');b.add_argument('--run',required=True);b.add_argument('--mode',choices=['fresh','resume'],required=True);b.add_argument('--target-epochs',type=int,required=True)
    args=p.parse_args()
    if args.action=='prepare':
        preflight(args.config)
        run=prepare(args.config,args.run_id)
        verify(run)
        print('PREPARED_RUN='+str(run.resolve()),flush=True)
        return
    run=Path(args.run).expanduser().resolve()
    verify(run)
    with experiment_lock(run):
        session=start_session(run,mode=args.mode,target_epochs=args.target_epochs)
        frozen=run/'source'
        worker=frozen/'train_utils'/'t04_session_worker.py'
        if not worker.is_file():raise FileNotFoundError(worker)
        env=os.environ.copy();env['PYTHONPATH']=str(frozen);env['PYTHONDONTWRITEBYTECODE']='1'
        env['PYTHONUNBUFFERED']='1'
        cmd=[sys.executable,'-B',str(worker),'--prepared-run',str(run),'--session',str(session),'--mode',args.mode,'--target-epochs',str(args.target_epochs)]
        print('EXECUTING',args.mode,run,session,flush=True)
        subprocess.run(cmd,cwd=frozen,env=env,check=True)
        print('SESSION_COMPLETED='+str(session),flush=True)
if __name__=='__main__':main()
