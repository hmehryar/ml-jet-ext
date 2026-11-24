"""T02.2 dispatch tests with synthetic precomputed aggregates."""
from pathlib import Path
from types import SimpleNamespace
import csv
import numpy as np
import pytest
import torch
import yaml

from config import get_config
from data.loader import get_dataloaders


def test_config_and_dispatch(tmp_path):
    root = tmp_path / "dataset"
    (root / "splits").mkdir(parents=True)
    rels = []
    for i in range(3):
        rel = f"cfg/agg500_event_{i:07d}.npy"
        (root / "cfg").mkdir(exist_ok=True)
        np.save(root / rel, np.zeros((5, 32, 32), dtype=np.float32))
        rels.append(rel)
    for fname, rel in zip(("train.csv", "val.csv", "test.csv"), rels):
        with (root / "splits" / fname).open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["agg_file_path", "alpha", "q0", "config"])
            w.writeheader()
            w.writerow(dict(agg_file_path=rel, alpha=0.2, q0=1.0, config="cfg"))
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(dict(dataset_type="moment", dataset_root=str(root),
                                        moments=[1, 2, 5], batch_size=1, num_workers=0,
                                        model_tag="ViT", backbone="vit")))
    cfg = get_config(str(path))
    assert cfg.input_shape == (3, 32, 32)
    assert list(cfg.loss_weights) == ["alpha_output", "q0_output"]
    assert Path(cfg.train_csv) == root / "splits/train.csv"
    train, val, test = get_dataloaders(cfg, device="cpu")
    for dl in (train, val, test):
        image, target = next(iter(dl))
        assert image.shape == (1, 3, 32, 32)
        assert set(target) == {"alpha_output", "q0_output"}


def test_legacy_dispatch_rejects_unknown_mode():
    with pytest.raises(ValueError, match="Unknown dataset_type"):
        get_dataloaders(SimpleNamespace(dataset_type="bad"), device="cpu")


def test_moment_environment_override_precedence(tmp_path, monkeypatch):
    path = tmp_path / 'config.yml'
    raw = dict(dataset_type='moment', dataset_root=str(tmp_path / 'yaml'),
               model_tag='test', backbone='vit_gaussian')
    path.write_text(yaml.safe_dump(raw))
    monkeypatch.setenv('DATASET_ROOT', str(tmp_path / 'old_environment'))
    monkeypatch.setenv('MLJET_DATASET_ROOT', str(tmp_path / 'local'))
    assert get_config(path).dataset_root_dir == str(tmp_path / 'local')
    assert get_config(path, overrides={'dataset_root':str(tmp_path / 'explicit')}).dataset_root_dir == str(tmp_path / 'explicit')
    monkeypatch.delenv('MLJET_DATASET_ROOT')
    assert get_config(path).dataset_root_dir == raw['dataset_root']
    del raw['dataset_root']
    path.write_text(yaml.safe_dump(raw))
    assert get_config(path).dataset_root_dir == str(tmp_path / 'old_environment')


def test_legacy_ignores_moment_environment_override(tmp_path, monkeypatch):
    path = tmp_path / 'legacy.yml'
    path.write_text(yaml.safe_dump(dict(dataset_root=str(tmp_path / 'legacy'),
                                      model_tag='legacy', backbone='vit_gaussian')))
    monkeypatch.setenv('MLJET_DATASET_ROOT', str(tmp_path / 'moment'))
    monkeypatch.setenv('MLJET_OUTPUT_ROOT', str(tmp_path / 'moment_outputs'))
    assert get_config(path).dataset_type == 'legacy'
    assert get_config(path).dataset_root_dir == str(tmp_path / 'legacy')
    assert not get_config(path).output_dir.startswith(str(tmp_path / 'moment_outputs'))


def test_moment_output_environment_precedence(tmp_path, monkeypatch):
    path = tmp_path / 'config.yml'
    path.write_text(yaml.safe_dump(dict(dataset_type='moment', dataset_root=str(tmp_path / 'data'),
                                      model_tag='test', backbone='vit_gaussian',
                                      output_dir=str(tmp_path / 'yaml_outputs'))))
    monkeypatch.setenv('MLJET_OUTPUT_ROOT', str(tmp_path / 'local_outputs'))
    assert get_config(path).output_dir.startswith(str(tmp_path / 'local_outputs'))
    assert get_config(path, overrides={'output_dir':str(tmp_path / 'explicit')}).output_dir.startswith(str(tmp_path / 'explicit'))
