"""Create a NEW T04 YAML with Ceph output_root; do not mutate source YAML.

This is an explicit generator for new runs, independent of the frozen v2 matrix.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import yaml
from train_utils.t04_storage import future_output_root
from train_utils.t04_preflight import parse_config


def make_new_config(source: Path, destination: Path, experiment: str) -> Path:
    source, destination = Path(source), Path(destination)
    if not source.is_file():
        raise FileNotFoundError(source)
    if destination.resolve(strict=False) == source.resolve(strict=False):
        raise ValueError('Cannot modify the source YAML in place')
    doc = yaml.safe_load(source.read_text())
    if not isinstance(doc, dict) or 'output_root' not in doc:
        raise ValueError('Source must be an existing T04 experiment YAML')
    doc['output_root'] = str(future_output_root(experiment))
    # Validate the complete T04 schema before creating the new file.
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open('x') as fh:
        yaml.safe_dump(doc, fh, sort_keys=False)
    try:
        parse_config(destination)
    except Exception:
        destination.unlink()
        raise
    return destination


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--from-yaml', type=Path, required=True)
    p.add_argument('--to-yaml', type=Path, required=True)
    p.add_argument('--experiment', required=True)
    a = p.parse_args()
    result = make_new_config(a.from_yaml, a.to_yaml, a.experiment)
    print('CREATED', result)
    print('OUTPUT_ROOT', yaml.safe_load(result.read_text())['output_root'])

if __name__ == '__main__':
    main()
