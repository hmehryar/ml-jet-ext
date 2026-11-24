"""T04-10 CPU-only safeguards and no production integration side effects."""
import csv
import hashlib
import json
from pathlib import Path
import shutil

import pytest
import yaml

from train_utils import t04_10_preflight as pf


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def fake_run(tmp_path):
    root = tmp_path / 'dataset'
    splits = root / 'splits'
    splits.mkdir(parents=True)
    published = {}
    for name in pf.SPLITS:
        with (splits / name).open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=['agg_file_path','alpha','q0','config'])
            writer.writeheader()
            writer.writerow({'agg_file_path':f'c/{name}.npy','alpha':'0.2','q0':'1.0','config':'c'})
        published[name] = digest(splits / name)
    (splits/'split_manifest.json').write_text(json.dumps({'split_file_sha256':published}))
    audit = tmp_path/'audit.json'
    audit.write_text(json.dumps({'split_sha256':published, 'source_id_method':'fixture',
        'global_cross_split_overlap':3, 'per_configuration_cross_split_overlap':{'c':3}}))
    config = {'model':'clip','task':'alpha_only','checkpoint':'openai/clip-vit-base-patch16',
        'checkpoint_revision':'a'*40,'strategy':'S4','moments':[1,2,3],
        'seed':42,'epochs':80,'precision':'bf16',
        'optimizer':{'type':'adamw','weight_decay':0.01},
        'learning_rates':{'backbone':1e-5,'adapter':1e-4,'heads':1e-4},
        'batch_size':8,'num_workers':2,'dataset_root':str(root),
        'output_root':str(tmp_path/'t04_10_single_head'), 'source_audit_report':str(audit),
        'split_protocol':'aggregate_disjoint','training_protocol':'fixed_budget_v2',
        'gpu_profile':'a100_mig_1g_10gb'}
    config_file = tmp_path/'config.yml'
    config_file.write_text(yaml.safe_dump(config))
    return config, config_file, audit


def test_cpu_preflight_discloses_overlap(tmp_path):
    c,p,_ = fake_run(tmp_path)
    parsed = pf.load_settings(p)
    assert parsed['task'] == 'alpha_only'
    assert pf.inspect_inputs(parsed)['source_overlap'] == 3


def test_source_independence_not_misrepresented(tmp_path):
    c,p,_ = fake_run(tmp_path)
    c['split_protocol'] = 'source_disjoint'
    p.write_text(yaml.safe_dump(c))
    with pytest.raises(ValueError, match='source-disjoint'):
        pf.inspect_inputs(pf.load_settings(p))


def test_no_test_leakage_or_changed_hash(tmp_path):
    c,p,_ = fake_run(tmp_path)
    test = Path(c['dataset_root'])/'splits/test.csv'
    test.write_text(test.read_text() + '\n')
    with pytest.raises(ValueError, match='manifest hash'):
        pf.inspect_inputs(pf.load_settings(p))


@pytest.mark.parametrize('key,value', [
    ('task','joint'), ('model','vit'), ('epochs',79), ('strategy','S5'),
    ('training_protocol','early_stop_v1'), ('precision','fp16'),
    ('moments',[1,3]),('seed',7), ('output_root','/tmp/t04_07_production'),
])
def test_invalid_protocol_rejected(tmp_path,key,value):
    c,p,_ = fake_run(tmp_path)
    c[key] = value
    p.write_text(yaml.safe_dump(c))
    with pytest.raises(ValueError):
        pf.load_settings(p)


def test_prepare_snapshot_separate_and_tamper_evident(tmp_path,monkeypatch):
    cfg,path,audit = fake_run(tmp_path)
    root = tmp_path/'repo'
    for folder in pf.SOURCE_DIRS:
        (root/folder).mkdir(parents=True)
        (root/folder/'__init__.py').write_text('')
    (root/'models/t04_10_single_head.py').write_text('# fixture\n')
    (root/'train_utils/t04_10_runner.py').write_text('# fixture\n')
    monkeypatch.setattr(pf, '_git', lambda _repo,*args:
                        'feat/t04-10-single-head-pilot' if args[0]=='branch'
                        else '' if args[0]=='status' else 'c'*40)
    run = pf.prepare(path,repo_root=root,run_id='test-42')
    assert pf.verify(run)
    assert run.parts[-5:] == ('clip','alpha_only','E3','seed-42','test-42')
    assert json.loads((run/'run_manifest.json').read_text())['test_set_evaluated'] is False
    with pytest.raises(FileExistsError):
        pf.prepare(path,repo_root=root,run_id='test-42')
    (run/'source/train_utils/t04_10_runner.py').write_text('tampered')
    with pytest.raises(ValueError,match='Snapshot'):
        pf.verify(run)


def test_reject_dirty_tracked_worktree(tmp_path,monkeypatch):
    _,path,_ = fake_run(tmp_path)
    root = tmp_path/'repo'
    (root/'models').mkdir(parents=True)
    (root/'train_utils').mkdir()
    (root/'models/t04_10_single_head.py').write_text('')
    (root/'train_utils/t04_10_runner.py').write_text('')
    monkeypatch.setattr(pf, '_git', lambda _repo,*args:
                        'feat/t04-10-single-head-pilot' if args[0]=='branch'
                        else ' M tracked.py' if args[0]=='status' else 'c'*40)
    with pytest.raises(ValueError,match='Tracked working tree'):
        pf.prepare(path,repo_root=root)
