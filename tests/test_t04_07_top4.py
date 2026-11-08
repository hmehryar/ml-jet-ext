"""CPU-only tests for 84-run T04-07 extension, without Ceph or Slurm."""
import importlib.util
import json
from pathlib import Path
from types import ModuleType, SimpleNamespace
import sys
from unittest.mock import Mock
import pytest
import yaml

ROOT=Path(__file__).resolve().parents[1]

def load_module(path, name):
    spec=importlib.util.spec_from_file_location(name, path)
    mod=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def fake_baseline(tmp_path):
    for model,k in (('clip',2),('clip',4),('siglip',4),('siglip',5)):
        for seed in (42,123,2026):
            path=tmp_path/f'experiments/t04/config/t04_06_v2/{model}_s4_e{k}_seed{seed}.yml'
            path.parent.mkdir(parents=True,exist_ok=True)
            d={'model':model,'strategy':'S4','moments':list(range(1,k+1)),
               'seed':seed,'moment_loss':{'type':'cross_entropy'},'epochs':80,
               'precision':'bf16','batch_size':8,'gradient_accumulation':1,
               'training_protocol':'fixed_budget_v2','split_protocol':'aggregate_disjoint',
               'output_root':'/rs/rs_grp_mljet/ml-jet-ext/training_output/T04/t04_06_v2_production',
               'checkpoint':'model/checkpoint','checkpoint_revision':'a'*40}
            path.write_text(yaml.safe_dump(d))


def fake_controller(monkeypatch):
    p=ModuleType('train_utils');p.__path__=[]
    m=ModuleType('train_utils.t04_preflight')
    def parser(path):
        return SimpleNamespace(**yaml.safe_load(Path(path).read_text()))
    m.parse_config=parser
    m.preflight=Mock(return_value=(SimpleNamespace(epochs=80,training_protocol='fixed_budget_v2',training_protocol_hash='ok'),
                                         {'source_overlap_independence_claim':False}))
    m.prepare=Mock();m.verify=Mock()
    sess=ModuleType('train_utils.t04_sessions');sess.latest_checkpoint=Mock(return_value=None)
    monkeypatch.setitem(sys.modules,'train_utils',p)
    monkeypatch.setitem(sys.modules,'train_utils.t04_preflight',m)
    monkeypatch.setitem(sys.modules,'train_utils.t04_sessions',sess)
    return load_module(ROOT/'tools/t04_07_top4_production.py','t04_07_top4_production_test')


def test_exact_84_model_loss_seed_matrix(tmp_path):
    fake_baseline(tmp_path)
    gen=load_module(ROOT/'tools/generate_t04_07_top4.py','t04_07_top4_gen_test')
    configs,plan=gen.build(tmp_path)
    assert len(configs)==84
    assert len({str(p) for p,_ in configs})==84
    assert plan['expected_new_runs']==84 and plan['baseline_reused']==12
    assert {(r['model'],r['moments']) for r in plan['rows']}=={('clip','E4'),('clip','E2'),('siglip','E4'),('siglip','E5')}
    assert len([r for r in plan['rows'] if r['loss']['type']=='physical_mae'])==12
    assert len([r for r in plan['rows'] if r['loss']['type']=='ce_ordinal'])==36
    assert len([r for r in plan['rows'] if r['loss']['type']=='ce_physical_mae'])==36
    for path,raw in configs:
        data=yaml.safe_load(raw)
        assert data['output_root']==gen.OUTPUT
        assert data['training_protocol']=='fixed_budget_v2'
        assert data['epochs']==80
        assert data['moment_loss']['type']!='cross_entropy'


def test_controller_expected_rows_hashes(tmp_path,monkeypatch):
    fake_baseline(tmp_path)
    gen=load_module(ROOT/'tools/generate_t04_07_top4.py','t04_07_top4_gen_test2')
    configs,plan=gen.build(tmp_path)
    for relative,raw in configs:
        path=tmp_path/relative
        path.parent.mkdir(parents=True,exist_ok=True)
        path.write_bytes(raw)
    (tmp_path/gen.TARGET/'top4_plan.json').write_text(json.dumps(plan))
    ctrl=fake_controller(monkeypatch)
    monkeypatch.setattr(ctrl,'REPO',tmp_path)
    rows,digest=ctrl.expected_rows()
    assert len(rows)==84 and len({x['run'] for x in rows})==84
    assert all(x['status']=='planned' for x in rows)
    p=tmp_path/configs[0][0]
    p.write_text(p.read_text()+'\n# accidental edit\n')
    with pytest.raises(RuntimeError,match='hash'):
        ctrl.expected_rows()


def test_held_batch_dry_run_submits_nothing(monkeypatch,capsys):
    c=fake_controller(monkeypatch)
    monkeypatch.setattr(c,'assert_git',lambda:c.COMMIT)
    es=[{'id':f'e{i}','status':'prepared','run':f'/tmp/e{i}'} for i in range(84)]
    monkeypatch.setattr(c,'load',lambda:{'entries':{e['id']:e for e in es}})
    monkeypatch.setattr(c,'observe',lambda e:{'epoch':0,'status':'prepared'})
    monkeypatch.setattr(c.subprocess,'run', Mock(side_effect=AssertionError('No Slurm in dry-run')))
    c.do_submit_wave(limit=20,apply=False)
    assert '"selected_count": 20' in capsys.readouterr().out


def test_batch_refuses_ambiguity(monkeypatch):
    c=fake_controller(monkeypatch)
    monkeypatch.setattr(c,'assert_git',lambda:c.COMMIT)
    monkeypatch.setattr(c,'load',lambda:{'entries':{'a':{'status':'submitting','id':'a'}}})
    with pytest.raises(RuntimeError,match='Ambiguous'):
        c.do_submit_wave(limit=20)


def test_individual_slurm_script_targets_new_namespace():
    data=(ROOT/'slurm/t04_07_top4_production.slurm').read_text()
    assert 't04_07_top4_production/' in data
    assert 't04_07_loss_pilot/' not in data
    assert 'T04_07_PROD_RUN' in data
    assert 'T04_07_PROD_TARGET' in data
    assert 'T04_EXECUTION_PROFILE=cpu1_workers0_v1' in data
