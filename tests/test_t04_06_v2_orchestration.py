"""Stage 3F.2C2 isolated tests: no Slurm jobs, GPUs, or dataset needed."""
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import pytest
from tools.t04_06_v2_full_horizon import _target, _require_v2, active_count, fence
from tools.t04_06_v2_reconcile import evidence, decide, _require_v2_registry
from train_utils.t04_training_policy import policy


def config(epochs=80):
    return SimpleNamespace(epochs=epochs,min_epochs=12,patience=16,
        training_protocol='fixed_budget_v2',training_protocol_hash=policy(epochs).digest(),
        output_root='training_output/T04/t04_06_v2_production', precision='bf16')


def test_v2_fresh_targets_follow_configurable_budget(tmp_path):
    cfg=tmp_path/'a.yml';cfg.write_text('a')
    entry={'id':'clip_s1_e1_seed42','config':'a.yml','config_sha256':'a'*64,
           'run':str(tmp_path/'absent'),'status':'planned','attempts':[]}
    with patch('tools.t04_06_v2_full_horizon.sha',return_value='a'*64),\
         patch('tools.t04_06_v2_full_horizon.preflight',return_value=(config(120),{})):
        result=_target(entry,tmp_path,24)
    assert result['mode']=='fresh' and result['target']==24


def test_v2_short_budget_is_not_forced_to_12(tmp_path):
    cfg=tmp_path/'a.yml';cfg.write_text('a')
    entry={'id':'clip_s1_e1_seed42','config':'a.yml','config_sha256':'a'*64,
           'run':str(tmp_path/'absent'),'status':'planned','attempts':[]}
    with patch('tools.t04_06_v2_full_horizon.sha',return_value='a'*64),\
         patch('tools.t04_06_v2_full_horizon.preflight',return_value=(config(5),{})):
        result=_target(entry,tmp_path,24)
    assert result['target']==5


def test_v2_rejects_legacy_configuration(tmp_path):
    cfg=tmp_path/'a.yml';cfg.write_text('a')
    c=config(); c.training_protocol='early_stop_v1'
    entry={'id':'clip_s1_e1_seed42','config':'a.yml','config_sha256':'a'*64,
           'run':str(tmp_path/'absent'),'status':'planned','attempts':[]}
    with patch('tools.t04_06_v2_full_horizon.sha',return_value='a'*64),\
         patch('tools.t04_06_v2_full_horizon.preflight',return_value=(c,{})):
        with pytest.raises(RuntimeError,match='fixed-budget'):
            _target(entry,tmp_path,24)


def _make_evidence(tmp_path, epoch, budget=80, early_stopped=False):
    run=tmp_path/'run'; run.mkdir()
    (run/'config.resolved.json').write_text(json.dumps(dict(
        epochs=budget,training_protocol='fixed_budget_v2',
        training_protocol_hash=policy(budget).digest(),
        output_root='training_output/T04/t04_06_v2_production')))
    (run/'moment_epoch_metrics.csv').write_text('epoch\n'+''.join(f'{i}\n' for i in range(1,epoch+1)))
    session=run/'sessions/session-001';session.mkdir(parents=True)
    (session/'session_manifest.json').write_text(json.dumps({'status':'completed'}))
    (session/'result.json').write_text(json.dumps({'epoch':epoch,'early_stopped':early_stopped}))
    return {'run':str(run)}, dict(epoch=epoch,sha256='a'*64,session='session-001')


def test_v2_completed_only_at_budget(tmp_path):
    entry,pointer=_make_evidence(tmp_path,80)
    with patch('tools.t04_06_v2_reconcile.verify'), patch('tools.t04_06_v2_reconcile.latest_checkpoint',return_value=pointer):
        result=evidence(entry)
    assert result['ok'] and not result['resume_eligible']


def test_v2_partial_returns_ready_to_resume(tmp_path):
    entry,pointer=_make_evidence(tmp_path,24)
    with patch('tools.t04_06_v2_reconcile.verify'), patch('tools.t04_06_v2_reconcile.latest_checkpoint',return_value=pointer):
        result=evidence(entry)
    assert result['resume_eligible'] and not result['ok']


def test_v2_legacy_early_stop_rejected(tmp_path):
    entry,pointer=_make_evidence(tmp_path,20,early_stopped=True)
    with patch('tools.t04_06_v2_reconcile.verify'), patch('tools.t04_06_v2_reconcile.latest_checkpoint',return_value=pointer):
        result=evidence(entry)
    assert not result['ok'] and not result.get('resume_eligible')


def test_v2_fail_closed_runtime_identity(tmp_path):
    p=tmp_path/'registry.json';p.write_text(json.dumps({'kind':'T04-06-v2-plan','training_protocol':'fixed_budget_v2'}))
    doc={'repo':str(tmp_path),'plan_path':str(p)}
    with pytest.raises(RuntimeError,match='independent v2 plan'):
        _require_v2(doc,tmp_path/'registry.runtime.json')


def test_bounded_slots_and_separate_intents(tmp_path):
    doc={'entries':{str(i):{'status':'submitted'} for i in range(4)}}
    assert active_count(doc)==4
    p=fence(tmp_path/'registry.runtime.json',dict(id='clip_s1_e1_seed42',from_epoch=0))
    assert 'stage3f_v2.' in p.name


def test_v2_registry_bootstrap_is_independent(tmp_path):
    from tools.t04_06_v2_matrix import generate, PLAN, RUNTIME
    from tools.t04_06_registry import bootstrap, read
    import yaml
    repo=tmp_path/'repo'
    template_dir=repo/'experiments/t04/config/t04_05_smoke'
    template_dir.mkdir(parents=True)
    for model in ('clip','siglip'):
        sample=dict(model=model,precision='bf16',moments=[1,2,3,4,5],
            batch_size=8,gradient_accumulation=1,split_protocol='aggregate_disjoint',
            min_epochs=12,patience=16)
        (template_dir/f'{model}_s2_e5_seed42.yml').write_text(yaml.safe_dump(sample))
    plan=generate(repo,epochs=80)
    count=bootstrap(repo,plan,repo/RUNTIME,'a'*40)
    assert count==120
    runtime=read(repo/RUNTIME)
    _require_v2(runtime,repo/RUNTIME)
    assert all(e['status']=='planned' for e in runtime['entries'].values())
    assert not (repo/'training_output/T04/t04_06/registry.runtime.json').exists()
    with pytest.raises(FileExistsError):
        bootstrap(repo,plan,repo/RUNTIME,'a'*40)
