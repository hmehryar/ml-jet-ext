"""Offline CPU-only tests. No scheduler or Ceph access required."""
from __future__ import annotations
import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock
import pytest

MODULE = Path(__file__).resolve().parents[1] / 'tools/t04_07_production.py'


def load(monkeypatch):
    # Substitute only the imports; tests exercise the real controller code.
    parent = ModuleType('train_utils')
    parent.__path__ = []
    pre = ModuleType('train_utils.t04_preflight')
    pre.parse_config = Mock()
    pre.preflight = Mock()
    pre.prepare = Mock()
    pre.verify = Mock()
    sessions = ModuleType('train_utils.t04_sessions')
    sessions.latest_checkpoint = Mock(return_value=None)
    monkeypatch.setitem(sys.modules, 'train_utils', parent)
    monkeypatch.setitem(sys.modules, 'train_utils.t04_preflight', pre)
    monkeypatch.setitem(sys.modules, 'train_utils.t04_sessions', sessions)
    spec = importlib.util.spec_from_file_location('t04_07_controller_under_test', MODULE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_submission_count_fail_closed(monkeypatch):
    m = load(monkeypatch)
    for n in (0,4,21):
        with pytest.raises(ValueError):
            m.submission_preview(n)


def test_preview_refuses_active_and_needs_no_slurm(monkeypatch, tmp_path):
    m = load(monkeypatch)
    monkeypatch.setattr(m, 'assert_git', lambda: m.COMMIT)
    monkeypatch.setattr(m, 'load', lambda: {'entries': {'x':{'status':'submitted'}}})
    monkeypatch.setattr(m.subprocess, 'run', Mock(side_effect=AssertionError('No subprocess!')))
    with pytest.raises(RuntimeError,match='unresolved active'):
        m.submission_preview(1)


def test_preview_refuses_unexpected_session(monkeypatch):
    m = load(monkeypatch)
    monkeypatch.setattr(m, 'assert_git', lambda: m.COMMIT)
    monkeypatch.setattr(m, 'load', lambda: {'entries': {'a':{'status':'prepared','id':'a','run':'/tmp/a'}}})
    monkeypatch.setattr(m, 'observe', lambda e: {'epoch':1,'status':'completed_session'})
    with pytest.raises(RuntimeError,match='unexpected artifacts'):
        m.submission_preview(1)


def test_dry_submit_never_calls_sbatch(monkeypatch, capsys):
    m = load(monkeypatch)
    e={'id':'abc','run':'/tmp/abc','status':'prepared'}
    monkeypatch.setattr(m, 'submission_preview', lambda count, mode='fresh': [e])
    runner=Mock(side_effect=AssertionError('sbatch forbidden in dry-run'))
    monkeypatch.setattr(m.subprocess,'run',runner)
    m.do_submit(count=1,apply=False)
    assert 'NO sbatch' in capsys.readouterr().out
    runner.assert_not_called()


def test_preparation_dry_run_does_not_write(monkeypatch):
    m=load(monkeypatch)
    e={'id':'abc','run':'/this/path/must/not/exist','status':'planned','config':'abc.yml'}
    monkeypatch.setattr(m,'assert_git',lambda:m.COMMIT)
    monkeypatch.setattr(m,'load',lambda:{'entries':{'abc':e}})
    monkeypatch.setattr(m,'preflight',lambda path:(None,None))
    monkeypatch.setattr(m,'prepare',Mock(side_effect=AssertionError('prepare forbidden')))
    m.do_prepare(apply=False)
    m.prepare.assert_not_called()


def test_wrong_commit_blocks_init(monkeypatch):
    m=load(monkeypatch)
    monkeypatch.setattr(m,'assert_git',Mock(side_effect=RuntimeError('wrong commit')))
    with pytest.raises(RuntimeError,match='wrong commit'):
        m.do_init(False)


def test_slurm_script_contains_hold_only_in_controller():
    code=MODULE.read_text()
    script=(MODULE.parents[1]/'slurm/t04_07_production.slurm').read_text()
    assert "'--hold'" in code
    assert 'scontrol release' in code
    assert 'T04_EXECUTION_PROFILE=cpu1_workers0_v1' in script
    assert '--target-epochs "$T04_07_PROD_TARGET"' in script
    assert 't04_07_loss_pilot' in script


def test_full_21_entry_plan_and_cpu_dry_run(monkeypatch, tmp_path, capsys):
    m = load(monkeypatch)
    import json
    import hashlib
    import yaml
    repo = tmp_path / 'repo'
    root = repo / m.CONFIGS
    root.mkdir(parents=True)
    output = tmp_path / 'ceph_fake' / 't04_07_loss_pilot'
    monkeypatch.setattr(m, 'REPO', repo)
    monkeypatch.setattr(m, 'OUTPUT', output)
    monkeypatch.setattr(m, 'STATE', output/'_operations')
    monkeypatch.setattr(m, 'REGISTRY', output/'_operations'/'registry.runtime.json')
    monkeypatch.setattr(m, 'assert_git', lambda: m.COMMIT)
    entries = []
    for seed in m.SEEDS:
        for kind, weight in m.TREATMENTS:
            suffix = kind if weight is None else kind+'_lambda'+str(weight).replace('.', 'p')
            identity = 'clip_s4_e3_seed{}_{}'.format(seed,suffix)
            loss={'type':kind}
            if kind=='ce_ordinal': loss['ordinal_weight']=weight
            if kind=='ce_physical_mae': loss['mae_weight']=weight
            cfg={'model':'clip','strategy':'S4','moments':[1,2,3],
                'seed':seed,'epochs':80,'moment_loss':loss,'batch_size':8,
                'gradient_accumulation':1,'precision':'bf16',
                'split_protocol':'aggregate_disjoint','training_protocol':'fixed_budget_v2',
                'output_root':str(output)}
            path=root/(identity+'.yml')
            body=yaml.safe_dump(cfg,sort_keys=False)
            path.write_text(body)
            entries.append({'id':identity,'seed':seed,'loss':loss,
                            'sha256':hashlib.sha256(body.encode()).hexdigest()})
    (root/'pilot_plan.json').write_text(json.dumps({'new_config_count':21,'baseline_reused':3,'rows':entries}))
    def parse(path):
        return SimpleNamespace(**yaml.safe_load(Path(path).read_text()))
    monkeypatch.setattr(m,'parse_config',parse)
    monkeypatch.setattr(m,'preflight',lambda path:(parse(path),{}))
    rows,phash=m.expected_rows()
    assert len(rows)==21 and len({r['run'] for r in rows})==21
    assert len(phash)==64
    m.do_init(apply=False)
    assert '"expected": 21' in capsys.readouterr().out
    assert not m.REGISTRY.exists()
    assert not output.exists()
    m.do_init(apply=True)
    assert len(m.load()['entries'])==21
    m.do_prepare(apply=False)
    assert all(e['status']=='planned' for e in m.load()['entries'].values())
    assert not any(Path(e['run']).exists() for e in m.load()['entries'].values())


def test_full_dry_run_blocks_existing_registry(monkeypatch, tmp_path):
    m=load(monkeypatch)
    monkeypatch.setattr(m,'assert_git',lambda:m.COMMIT)
    monkeypatch.setattr(m,'expected_rows',lambda:([], 'a'*64))
    monkeypatch.setattr(m,'REGISTRY',tmp_path/'registry.json')
    m.REGISTRY.write_text('{}')
    with pytest.raises(FileExistsError):
        m.do_full_dry_run()


def test_batch_dry_run_never_submits_and_keeps_first_running(monkeypatch, capsys):
    m=load(monkeypatch)
    entries={'initial': {'id':'initial','status':'running','run':'/fake/initial'}}
    entries.update({'p%d'%i:{'id':'p%d'%i,'status':'prepared','run':'/fake/%d'%i} for i in range(20)})
    monkeypatch.setattr(m,'assert_git',lambda:m.COMMIT)
    monkeypatch.setattr(m,'load',lambda:{'entries':entries})
    monkeypatch.setattr(m,'observe',lambda e:{'epoch':0,'status':'prepared'})
    monkeypatch.setattr(m,'do_submit',Mock(side_effect=AssertionError('Forbidden sbatch')))
    m.do_submit_remaining(apply=False, limit=20)
    assert '"selected_count": 20' in capsys.readouterr().out
    m.do_submit.assert_not_called()


def test_batch_refuses_ambiguous_submitting_state(monkeypatch):
    m=load(monkeypatch)
    entries={'initial':{'status':'running'},'bad':{'status':'submitting'}}
    monkeypatch.setattr(m,'assert_git',lambda:m.COMMIT)
    monkeypatch.setattr(m,'load',lambda:{'entries':entries})
    with pytest.raises(RuntimeError, match='Ambiguous'):
        m.do_submit_remaining(apply=False)


def test_batch_blocks_second_wave_if_already_submitted(monkeypatch):
    m=load(monkeypatch)
    entries={'initial':{'status':'running'},'second':{'status':'submitted'}}
    entries.update({'p%d'%i:{'status':'prepared'} for i in range(19)})
    monkeypatch.setattr(m,'assert_git',lambda:m.COMMIT)
    monkeypatch.setattr(m,'load',lambda:{'entries':entries})
    with pytest.raises(RuntimeError,match='Expected exactly one active'):
        m.do_submit_remaining(apply=False)


def test_individual_allow_active_preserves_immutable_checks(monkeypatch):
    m=load(monkeypatch)
    entries={'initial':{'status':'running'},'candidate':{'status':'prepared','id':'candidate','run':'/fake/candidate'}}
    monkeypatch.setattr(m,'assert_git',lambda:m.COMMIT)
    monkeypatch.setattr(m,'load',lambda:{'entries':entries})
    monkeypatch.setattr(m,'observe',lambda e:{'epoch':0,'status':'prepared'})
    assert m.submission_preview(1,'fresh',allow_active=True)[0]['id']=='candidate'
    with pytest.raises(RuntimeError,match='unresolved active'):
        m.submission_preview(1,'fresh')


def test_release_preview_readonly(monkeypatch,capsys):
    m=load(monkeypatch)
    entries={'a':{'id':'a','status':'running','attempts':[{'job_id':'41348227','held':True}]},
             'b':{'id':'b','status':'submitted','attempts':[{'job_id':'41350000','held':True}]}}
    monkeypatch.setattr(m,'load',lambda:{'entries':entries})
    monkeypatch.setattr(m.subprocess,'run',Mock(side_effect=AssertionError('No scheduler')))
    m.do_release_preview()
    out=capsys.readouterr().out
    assert '41348227' in out and '41350000' in out
