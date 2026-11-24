"""T04-02 isolated session executable; launch from checkout, train from frozen source.

Use --mode fresh for initial session; --mode resume for new immutable session.
The advisory experiment lock is held by this parent for the entire child lifetime.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

from train_utils.t04_preflight import verify
from train_utils.t04_sessions import experiment_lock, start_session, finish_session


def execute(run, mode, target_epochs, internal_session=None):
    run = Path(run).expanduser().resolve()
    verify(run)
    with experiment_lock(run):
        session = start_session(run, mode=mode, target_epochs=target_epochs)
        env = dict(os.environ, PYTHONPATH=str(run / 'source'),
                   PYTHONDONTWRITEBYTECODE='1', MPLBACKEND='Agg')
        args = [sys.executable, '-B', '-m', 'train_utils.t04_session_worker',
                '--prepared-run', str(run), '--session', str(session),
                '--mode', mode, '--target-epochs', str(target_epochs)]
        try:
            result = subprocess.call(args, cwd=str(run / 'source'), env=env)
        except BaseException:
            finish_session(run, session, status='interrupted')
            raise
        if result != 0:
            finish_session(run, session, status='interrupted')
        return result


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument('--prepared-run', required=True)
    p.add_argument('--mode', choices=['fresh','resume'], required=True)
    p.add_argument('--target-epochs', type=int, required=True)
    args = p.parse_args(argv)
    raise SystemExit(execute(args.prepared_run, args.mode, args.target_epochs))


if __name__ == '__main__':
    main()
