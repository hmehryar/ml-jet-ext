"""Execute a T04 prepared run strictly from its frozen source snapshot."""
import argparse
import os
from pathlib import Path
import subprocess
import sys

from train_utils.t04_preflight import verify


def execute(out):
    out = Path(out).expanduser().resolve()
    verify(out)
    if (out / '.execution.lock').exists():
        raise FileExistsError(f'Prepared run already executed: {out}')
    # Exclusive lock; created only after verification, and intentionally retained.
    with (out / '.execution.lock').open('x') as f:
        f.write(str(os.getpid()) + '\n')
    env = dict(os.environ, PYTHONPATH=str(out / 'source'),
               PYTHONDONTWRITEBYTECODE='1', MPLBACKEND='Agg')
    return subprocess.call([sys.executable, '-B', '-m', 'train_utils.t04_runner',
                            '--prepared-run', str(out)], cwd=str(out / 'source'), env=env)


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument('--prepared-run', type=Path, required=True)
    args = p.parse_args(argv)
    raise SystemExit(execute(args.prepared_run))


if __name__ == '__main__':
    main()
