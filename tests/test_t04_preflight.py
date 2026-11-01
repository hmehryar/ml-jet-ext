"""T04-01 CPU-only tests; no real dataset required."""
import csv
import json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

import pytest
import yaml

from train_utils import t04_preflight as p


def fixture(tmp_path, monkeypatch):
    monkeypatch.setattr(p, 'COUNTS', {'train': 4, 'val': 2, 'test': 2})
    root = tmp_path / 'dataset'
    (root / 'splits').mkdir(parents=True)
    hashes = {}
    for split, count in p.COUNTS.items():
        file = root / 'splits' / f'{split}.csv'
        with file.open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=['agg_file_path','alpha','q0','config'])
            writer.writeheader()
            for i in range(count):
                writer.writerow({'agg_file_path':f'c0005_alpha0p20_q_1p0/{split}-{i}.npy',
                                 'alpha':'0.20','q0':'1.0','config':'c0005_alpha0p20_q_1p0'})
        hashes[file.name] = p.sha256(file)
    (root / 'splits/split_manifest.json').write_text(json.dumps({'split_file_sha256':hashes}))
    audit = tmp_path / 'audit.json'
    audit.write_text(json.dumps({'global_cross_split_overlap':0,
                                 'per_configuration_cross_split_overlap':{f'c{i}':0 for i in range(35)},
                                 'source_id_method':'source jet identifiers from aggregate metadata',
                                 'split_sha256':hashes}))
    cfg = dict(model='siglip', checkpoint='google/siglip-base-patch16-224',
               checkpoint_revision='a'*40, strategy='S1', moments=[1,2,3,4,5],
               moment_loss={'type':'cross_entropy'}, precision='fp32',
               learning_rates={'backbone':1e-5,'adapter':1e-4,'heads':1e-4},
               seed=42, epochs=400, min_epochs=75, patience=100,
               scheduler={'type':'ReduceLROnPlateau','monitor':'val_loss','mode':'min'},
               batch_size=4,gradient_accumulation=8,num_workers=2,
               dataset_root=str(root), output_root=str(tmp_path/'runs'),
               gpu_profile='a100-mig-1g.10gb',source_audit_report=str(audit))
    config = tmp_path/'config.yml'
    config.write_text(yaml.safe_dump(cfg))
    return config, cfg, root, audit


def test_preflight_and_prepare(tmp_path, monkeypatch):
    config, cfg, root, audit = fixture(tmp_path,monkeypatch)
    assert p.preflight(config)[1]['counts'] == p.COUNTS
    out = p.prepare(config, 'test-run')
    assert p.verify(out)
    assert (out/'run_manifest.json').is_file()
    with pytest.raises(FileExistsError):
        p.prepare(config, 'test-run')
    (root/'splits/val.csv').write_text('tampered')
    with pytest.raises(RuntimeError, match='Input changed'):
        p.verify(out)


def test_bad_schema_fails(tmp_path, monkeypatch):
    config, cfg, _, _ = fixture(tmp_path,monkeypatch)
    cfg['moments'] = [2,3]
    config.write_text(yaml.safe_dump(cfg))
    with pytest.raises(ValueError, match='cumulative'):
        p.preflight(config)


def test_audit_must_bind_hashes(tmp_path,monkeypatch):
    config, cfg, root, audit = fixture(tmp_path,monkeypatch)
    obj = json.loads(audit.read_text())
    obj['split_sha256']['train.csv'] = '0'*64
    audit.write_text(json.dumps(obj))
    with pytest.raises(ValueError,match='Audit does not match'):
        p.preflight(config)


def test_audit_requires_zero_per_config(tmp_path,monkeypatch):
    config, cfg, root, audit = fixture(tmp_path,monkeypatch)
    obj = json.loads(audit.read_text())
    obj['per_configuration_cross_split_overlap']['c1'] = 1
    audit.write_text(json.dumps(obj))
    with pytest.raises(ValueError,match='overlap mismatch'):
        p.preflight(config)


def test_aggregate_disjoint_accepts_reported_source_reuse(tmp_path, monkeypatch):
    config, cfg, _, audit = fixture(tmp_path, monkeypatch)
    cfg['split_protocol'] = 'aggregate_disjoint'
    config.write_text(yaml.safe_dump(cfg))
    report = json.loads(audit.read_text())
    report['global_cross_split_overlap'] = 3
    report['per_configuration_cross_split_overlap']['c1'] = 3
    audit.write_text(json.dumps(report))
    _, details = p.preflight(config)
    assert details['source_audit_status'] == 'verified_overlap_disclosed'
    assert details['global_cross_split_overlap'] == 3
    assert not details['source_overlap_independence_claim']
    run = p.prepare(config, 'aggregate-overlap')
    manifest = json.loads((run / 'run_manifest.json').read_text())
    assert manifest['configuration']['split_protocol'] == 'aggregate_disjoint'
    assert manifest['inputs']['global_cross_split_overlap'] == 3
    assert p.verify(run)

def test_aggregate_disjoint_rejects_unbound_audit(tmp_path, monkeypatch):
    config, cfg, _, audit = fixture(tmp_path, monkeypatch)
    cfg['split_protocol'] = 'aggregate_disjoint'
    config.write_text(yaml.safe_dump(cfg))
    report = json.loads(audit.read_text())
    report['global_cross_split_overlap'] = 3
    report['per_configuration_cross_split_overlap']['c1'] = 3
    report['split_sha256']['val.csv'] = '0'*64
    audit.write_text(json.dumps(report))
    with pytest.raises(ValueError,match='Audit does not match'):
        p.preflight(config)

def test_source_disjoint_still_rejects_overlap(tmp_path, monkeypatch):
    config, cfg, _, audit = fixture(tmp_path, monkeypatch)
    report = json.loads(audit.read_text())
    report['global_cross_split_overlap'] = 3
    report['per_configuration_cross_split_overlap']['c1'] = 3
    audit.write_text(json.dumps(report))
    with pytest.raises(ValueError,match='Source-disjoint'):
        p.preflight(config)
