from pathlib import Path
import yaml
from tools.t04_05_smoke import make_config


def test_make_config(tmp_path):
    source = tmp_path / 'source.yml'
    source.write_text(yaml.safe_dump({'model': 'clip', 'strategy': 'S3',
        'moments': [1, 2, 3, 4, 5], 'seed': 42, 'precision': 'bf16',
        'optimizer': {'type': 'adamw'}, 'split_protocol': 'aggregate_disjoint',
        'checkpoint_revision': 'a' * 40}))
    target = tmp_path / 'smoke.yml'
    config = make_config(source, target)
    assert (config['batch_size'], config['gradient_accumulation'], config['epochs']) == (8, 1, 2)
    assert yaml.safe_load(target.read_text())['checkpoint_revision'] == 'a' * 40
