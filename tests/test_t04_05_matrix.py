from pathlib import Path
import pytest
import yaml
from tools.t04_05_matrix import generate


def _setup(tmp_path):
    base = tmp_path / 'experiments/t04/config'
    base.mkdir(parents=True)
    for model in ('clip', 'siglip'):
        (base / ('t04_04_%s_s2_bf16_pilot.yml' % model)).write_text(yaml.safe_dump({
            'model': model, 'strategy': 'S2', 'moments': [1,2,3,4,5],
            'precision': 'bf16', 'split_protocol': 'aggregate_disjoint',
            'optimizer': {'type': 'adamw', 'weight_decay': 0.01},
            'epochs': 1, 'patience': 2,
            'checkpoint': 'pinned', 'checkpoint_revision': 'a'*40,
        }))
    return base


def test_generates_all_eight_no_duplicates(tmp_path):
    _setup(tmp_path)
    files = generate(tmp_path)
    assert len(files) == 8
    assert len({p.name for p in files}) == 8
    for file in files:
        c = yaml.safe_load(file.read_text())
        assert c['moments'] == [1, 2, 3, 4, 5]
        assert c['seed'] == 42
        assert c['epochs'] == 2
        assert c['precision'] == 'bf16'
        assert c['batch_size'] == 1 and c['gradient_accumulation'] == 4
        assert c['optimizer']['type'] == 'adamw'
    with pytest.raises(FileExistsError):
        generate(tmp_path)


def test_refuses_unpinned_or_invalid_template(tmp_path):
    base = _setup(tmp_path)
    file = base / 't04_04_clip_s2_bf16_pilot.yml'
    config = yaml.safe_load(file.read_text())
    config['precision'] = 'fp32'
    file.write_text(yaml.safe_dump(config))
    with pytest.raises(ValueError, match='precision'):
        generate(tmp_path)
