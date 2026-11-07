"""Issue #29: audit Ceph migration and export relocated YAML *views*.

Never alters historical 120 source YAMLs, plan, runtime, or prepared-run hashes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess

import yaml

from tools.t04_06_registry import read
from tools.t04_06_v2_matrix import RUNTIME, PLAN, verify as verify_matrix
from train_utils.t04_storage import REPO, load_storage, historical_run_path


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def audit(*, check_manifests=False, checksum=False):
    cfg = load_storage()
    doc = read(REPO / RUNTIME)
    verify_matrix(REPO, REPO / PLAN)
    assert len(doc['entries']) == 120
    if any(e['status'] != 'completed' for e in doc['entries'].values()):
        raise RuntimeError('All 120 experiments must be completed')
    expected = set()
    for ident, e in doc['entries'].items():
        physical = historical_run_path(e['run'], storage=cfg)
        if not physical.is_dir() or not (physical / 'run_manifest.json').is_file():
            raise FileNotFoundError(f'{ident}: {physical}')
        expected.add(physical)
        if check_manifests:
            from train_utils.t04_preflight import verify
            verify(physical)
    if len(expected) != 120:
        raise RuntimeError('Duplicate relocated run paths')
    if checksum:
        old = REPO / 'training_output/T04/t04_06_v2_production'
        if not old.is_dir():
            raise FileNotFoundError('Original directory is required for checksum comparison')
        result = subprocess.run(['rsync', '-aHcn', '--itemize-changes', str(old)+'/',
                                 str(cfg.historical_t04_06_v2)+'/'],
                                capture_output=True, text=True, check=True)
        if result.stdout:
            raise RuntimeError('Rsync verification differences:\n'+result.stdout[:4000])
    print('PASS: 120/120 completed runs readable at Ceph storage')
    if check_manifests:
        print('PASS: 120/120 frozen manifests verified')
    if checksum:
        print('PASS: Ceph copy matches PanFS by rsync checksums')
    return doc


def export_yaml_views():
    cfg = load_storage()
    doc = audit()
    root = REPO / 'training_output/T04/t04_06_v2/relocated_yaml_views'
    if root.exists():
        raise FileExistsError('Relocated YAML view directory already exists: '+str(root))
    items = []
    # Views are written into non-repository training_output to keep the frozen
    # YAML files and Git state unchanged. Never feed these to the old launcher.
    root.mkdir(parents=True, exist_ok=False)
    for ident, entry in sorted(doc['entries'].items()):
        original = REPO / entry['config']
        original_bytes = original.read_bytes()
        if digest(original_bytes) != entry['config_sha256']:
            raise ValueError('Original config hash mismatch: '+ident)
        parsed = yaml.safe_load(original_bytes)
        if not isinstance(parsed, dict):
            raise ValueError('Malformed YAML: '+ident)
        parsed['output_root'] = str(cfg.historical_t04_06_v2)
        dest = root / f'{ident}.yml'
        encoded = yaml.safe_dump(parsed, sort_keys=False).encode()
        with dest.open('xb') as out:
            out.write(encoded)
        items.append({'id':ident, 'original_yaml':str(original),
                      'original_sha256':digest(original_bytes),
                      'relocated_yaml':str(dest), 'relocated_sha256':digest(encoded),
                      'recorded_run':entry['run'],
                      'physical_run':str(historical_run_path(entry['run'], storage=cfg))})
    index = {'schema':1, 'purpose':'analysis-only relocated YAML views; NOT frozen training configs',
             'original_plan_sha256':doc['plan_sha256'],
             'original_runtime':str(REPO / RUNTIME), 'entries':items}
    (root / 'migration_index.json').write_text(json.dumps(index, indent=2)+'\n')
    print('EXPORTED', len(items), 'analysis-only relocated YAML views into',root)
    return root


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=['audit','export-yaml-views'])
    p.add_argument('--check-manifests',action='store_true')
    p.add_argument('--checksum',action='store_true')
    args=p.parse_args()
    if args.command=='audit':
        audit(check_manifests=args.check_manifests, checksum=args.checksum)
    else:
        if args.check_manifests or args.checksum:
            p.error('Only audit accepts --check-manifests/--checksum')
        export_yaml_views()

if __name__ == '__main__':
    main()
