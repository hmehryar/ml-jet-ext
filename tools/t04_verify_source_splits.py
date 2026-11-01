#!/usr/bin/env python3
"""Read-only audit of source-jet membership across published ML-JET-Ext splits.

The dataset is never written. A temporary SQLite index is created in --work-dir.
The output report is written ONLY if all checks pass; failures write a diagnostic
JSON alongside the requested report, without asserting source-disjointness.
"""
import argparse
import csv
import hashlib
import json
import sqlite3
import tempfile
from collections import Counter
from pathlib import Path

SPLITS = {'train': 1, 'val': 2, 'test': 4}
EXPECTED = {'train': 11200, 'val': 1400, 'test': 1400}

def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for part in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(part)
    return h.hexdigest()

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset-root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--work-dir', type=Path, default=None, help='Local scratch preferred; SQLite index needs disk space')
    args = p.parse_args()
    root = args.dataset_root.resolve()
    published_file = root / 'splits/split_manifest.json'
    published = json.loads(published_file.read_text())
    configs = set(published.get('configurations', {}))
    if len(configs) != 35:
        raise ValueError('Expected 35 named configurations in split manifest')
    hashes = {}
    for name in SPLITS:
        f = root / 'splits' / f'{name}.csv'
        hashes[f'{name}.csv'] = digest(f)
        if published['split_file_sha256'].get(f'{name}.csv') != hashes[f'{name}.csv']:
            raise ValueError(f'Published SHA256 mismatch for {name}.csv')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    scratch = args.work_dir or args.output.parent
    scratch.mkdir(parents=True, exist_ok=True)
    counts = Counter()
    cross_config_ids = 0
    examples = []
    with tempfile.TemporaryDirectory(prefix='t04-source-audit-', dir=scratch) as temp:
        db = sqlite3.connect(str(Path(temp) / 'sources.sqlite3'))
        db.execute('PRAGMA journal_mode=OFF')
        db.execute('PRAGMA synchronous=OFF')
        db.execute('PRAGMA temp_store=MEMORY')
        db.execute('CREATE TABLE src (source TEXT PRIMARY KEY, config TEXT NOT NULL, mask INTEGER NOT NULL) WITHOUT ROWID')
        for name, bit in SPLITS.items():
            path = root / 'splits' / f'{name}.csv'
            with path.open(newline='') as stream:
                reader = csv.DictReader(stream)
                if not {'agg_file_path', 'config', 'source_file_paths'} <= set(reader.fieldnames or []):
                    raise ValueError(f'{path} missing columns')
                for row in reader:
                    config = row['config']
                    if config not in configs:
                        raise ValueError(f'Unexpected config: {config}')
                    paths = row['source_file_paths'].split('|')
                    if len(paths) != 500 or len(set(paths)) != 500:
                        raise ValueError(f'Expected 500 unique jets in {row["agg_file_path"]}')
                    records = [(source.strip(), config, bit) for source in paths]
                    if any(not source.startswith(config + "_unfolded/") for source, _, _ in records):
                        raise ValueError(f"Source ID does not match its configuration: {config}")
                    if any(not source for source, _, _ in records):
                        raise ValueError('Blank source ID')
                    # Merge masks for recurring source IDs across aggregates; do not
                    # assume that an individual source appears only once overall.
                    db.executemany('''INSERT INTO src (source,config,mask) VALUES (?,?,?)
                        ON CONFLICT(source) DO UPDATE SET mask = mask | excluded.mask''', records)
                    counts[name] += 1
                    if counts[name] % 1000 == 0:
                        print(f'{name}: {counts[name]} aggregates examined', flush=True)
            db.commit()
        for name, expected in EXPECTED.items():
            if counts[name] != expected:
                raise ValueError(f'{name} count {counts[name]} != {expected}')
        overlap_expr = '(mask & (mask - 1)) != 0'
        global_overlap = db.execute(f'SELECT COUNT(*) FROM src WHERE {overlap_expr}').fetchone()[0]
        per_config = {config: 0 for config in sorted(configs)}
        for config, n in db.execute(f'SELECT config,COUNT(*) FROM src WHERE {overlap_expr} GROUP BY config'):
            per_config[config] = n
        if global_overlap:
            for source, config, mask in db.execute(f'SELECT source,config,mask FROM src WHERE {overlap_expr} LIMIT 5'):
                examples.append({'source':source,'config':config,'split_mask':mask})
        db.close()
    evidence = {
        'source_id_method': 'Exact source_file_paths values in published split CSV; SQLite bitmask per unique source ID; no split regeneration',
        'split_sha256': hashes,
        'split_counts': dict(counts),
        'global_cross_split_overlap': global_overlap,
        'per_configuration_cross_split_overlap': per_config,
        'cross_configuration_source_memberships': cross_config_ids,
        'examples': examples[:10],
        'published_split_manifest_sha256': digest(published_file),
    }
    if global_overlap or cross_config_ids:
        diagnostic = args.output.with_suffix('.FAILED.json')
        diagnostic.write_text(json.dumps(evidence,indent=2)+'\n')
        raise SystemExit(f'FAIL: {global_overlap} source IDs span splits; {cross_config_ids} source memberships cross configs. Diagnostic: {diagnostic}. No passing report created.')
    args.output.write_text(json.dumps(evidence, indent=2) + '\n')
    print('PASS: source IDs are disjoint across all three partitions, including all 35 configurations')
    print('Report:', args.output)

if __name__ == '__main__':
    main()
