"""Fail-closed, idempotent fix for v2 short Slurm segments. Never edits snapshots."""
from __future__ import annotations
import argparse
import ast
import difflib
from pathlib import Path

FILE = 'train_utils/t04_session_worker.py'
OLD = "    if cfg['min_epochs'] > target_epochs:\n        raise ValueError('target_epochs must be >= min_epochs')\n"
NEW = ("    # min_epochs controls only legacy early-stopping eligibility.\n"
       "    # Fixed-budget v2 can stop a Slurm segment before min_epochs.\n"
       "    if cfg.get('training_protocol', 'early_stop_v1') != 'fixed_budget_v2' and cfg['min_epochs'] > target_epochs:\n"
       "        raise ValueError('target_epochs must be >= min_epochs')\n")

def transform(source):
    ast.parse(source)
    if source.count(NEW) == 1 and source.count(OLD) == 0:
        return source
    if source.count(OLD) != 1 or source.count(NEW) != 0:
        raise RuntimeError('Unrecognized or ambiguous session worker; refusing replacement')
    # Fail-closed: require the earlier protocol integrity validation added by Stage 3F.2B.
    if "validate_protocol(protocol_id, protocol_hash, configured_epochs)" not in source:
        raise RuntimeError('Protocol-hash validation missing from session worker')
    updated = source.replace(OLD, NEW, 1)
    ast.parse(updated)
    return updated

def patch(repo, apply=False):
    path=Path(repo).resolve()/FILE
    before=path.read_text()
    after=transform(before)
    if before==after:
        print('ALREADY_PATCHED',path)
        return 0
    if apply:
        path.write_text(after)
        print('PATCHED',path)
    else:
        print(''.join(difflib.unified_diff(before.splitlines(True),after.splitlines(True), fromfile=str(path),tofile=str(path))),end='')
        print('CHECK_ONLY no changes')
    return 1

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    mode=parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--check', action='store_true')
    mode.add_argument('--apply', action='store_true')
    args=parser.parse_args()
    patch(Path.cwd(), apply=args.apply)
