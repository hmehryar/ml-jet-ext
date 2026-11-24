"""Stage 3F: no Slurm or real GPU needed for orchestration safety checks."""
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch, Mock
import json
import pytest

from tools.t04_06_full_horizon import active_count, plan, _target, fence, submit_wave


def entry(name, state='planned'):
    return dict(id=name, config='a.yml',config_sha256='a'*64,
                status=state,attempts=[],run='/tmp/a')


def document(entries):
    return {'entries': {e['id']:e for e in entries}, 'repo':'/tmp'}


def config():
    return SimpleNamespace(epochs=80,min_epochs=12,patience=16,scheduler={'type':'reduce_on_plateau'},
                           precision='bf16')


def test_maximum_four_active_and_hard_limit():
    doc=document([entry(f'clip_s1_e{i}_seed42','submitted') for i in range(4)])
    assert active_count(doc)==4
    doc['entries']['overflow']=entry('overflow','resumed')
    with pytest.raises(RuntimeError,match='bound'):
        active_count(doc)


def test_eligible_queue_prioritizes_resumes_and_respects_slots(tmp_path):
    ready=entry('clip_s2_e5_seed42','ready_to_resume')
    fresh=entry('siglip_s1_e1_seed123')
    another=entry('clip_s1_e5_seed42')
    active=entry('clip_s1_e2_seed42','submitted')
    doc=document([ready,fresh,another,active])
    with patch('tools.t04_06_full_horizon.read',return_value=doc), \
         patch('tools.t04_06_full_horizon._target',side_effect=lambda e,repo,n:{'id':e['id'],'from_epoch':0}), \
         patch('tools.t04_06_full_horizon.fence',side_effect=lambda r,x: tmp_path/(x['id']+'.missing')):
        _,chosen=plan(tmp_path/'runtime',scope='all',count=4)
    assert [x['id'] for x in chosen]==['clip_s2_e5_seed42','clip_s1_e5_seed42','siglip_s1_e1_seed123']
    with pytest.raises(ValueError,match='max concurrent'):
        plan(tmp_path/'runtime',count=5)


def test_fresh_target_at_least_min_epochs(tmp_path):
    e=entry('clip_s1_e5_seed42')
    e['run']=str(tmp_path/'missing-run')
    with patch('tools.t04_06_full_horizon.sha',return_value='a'*64), \
         patch('tools.t04_06_full_horizon.preflight',return_value=(config(),{})):
        x=_target(e,tmp_path,2)
    assert x['target']==12 and x['mode']=='fresh' and x['parent_sha256'] is None


def test_resume_parent_mismatch_is_hard_failure(tmp_path):
    e=entry('clip_s1_e5_seed42','ready_to_resume')
    run=tmp_path/'run';run.mkdir()
    e['run']=str(run)
    e['attempts']=[{'slurm_job_id':'12345','checkpoint_sha256':'bad','session':'session-001'}]
    pointer={'sha256':'b'*64,'epoch':12,'session':'session-001'}
    with patch('tools.t04_06_full_horizon.sha',return_value='a'*64), \
         patch('tools.t04_06_full_horizon.preflight',return_value=(config(),{})), \
         patch('tools.t04_06_full_horizon.verify'), \
         patch('tools.t04_06_full_horizon.latest_checkpoint',return_value=pointer):
        (run/'source').mkdir()
        (run/'source/config.original.yml').write_text('frozen')
        (tmp_path/'a.yml').write_text('frozen')
        with pytest.raises(RuntimeError,match='Parent SHA/session'):
            _target(e,tmp_path,24)


def test_previous_submission_intent_blocks_relaunch(tmp_path):
    item={'id':'clip_s1_e5_seed42','from_epoch':12}
    p=fence(tmp_path/'registry.runtime.json',item)
    p.write_text('{}')
    e=entry('clip_s1_e5_seed42','ready_to_resume')
    doc=document([e])
    with patch('tools.t04_06_full_horizon.read',return_value=doc), \
         patch('tools.t04_06_full_horizon._target',return_value=item):
        with pytest.raises(RuntimeError,match='intent'):
            plan(tmp_path/'registry.runtime.json')


def test_held_submit_fences_before_sbatch_and_never_releases(tmp_path):
    item={'id':'clip_s1_e5_seed42','mode':'fresh','from_epoch':0,'target':24,
          'run':str(tmp_path/'run'),'parent_sha256':None}
    Path(item['run']).mkdir()
    script=tmp_path/'train.slurm'
    script.write_text('#SBATCH --cpus-per-task=2\nexport OMP_NUM_THREADS=1 MKL_NUM_THREADS=1\n')
    e=entry(item['id']); doc=document([e]);calls=[]
    def fake_subprocess(command,**kwargs):
        calls.append(command)
        if command[0]=='sbatch':
            assert json.loads(fence(tmp_path/'runtime.json',item).read_text())['phase']=='before_sbatch'
            return Mock(stdout='12345\n')
        return Mock()
    with patch('tools.t04_06_full_horizon.plan',return_value=(doc,[item])), \
         patch('tools.t04_06_full_horizon.verify'), \
         patch('tools.t04_06_full_horizon.latest_checkpoint',return_value=None), \
         patch('tools.t04_06_full_horizon.subprocess.run',side_effect=fake_subprocess), \
         patch('tools.t04_06_full_horizon.transition') as transition_mock:
        submit_wave(tmp_path/'runtime.json','all',4,24,script)
    assert '--hold' in calls[1]
    assert not any(cmd[0]=='scontrol' for cmd in calls)
    assert transition_mock.call_args.args[2]=='submitted'
    assert json.loads(fence(tmp_path/'runtime.json',item).read_text())['phase']=='registered_held'


def test_uncertain_sbatch_retains_fence(tmp_path):
    item={'id':'clip_s1_e5_seed42','mode':'fresh','from_epoch':0,'target':24,
          'run':str(tmp_path/'run'),'parent_sha256':None}
    Path(item['run']).mkdir()
    script=tmp_path/'train.slurm'
    script.write_text('#SBATCH --cpus-per-task=2\nexport OMP_NUM_THREADS=1 MKL_NUM_THREADS=1\n')
    def sbatch(cmd,**kwargs):
        if cmd[0]=='sbatch': raise RuntimeError('scheduler unreachable')
        return Mock()
    with patch('tools.t04_06_full_horizon.plan',return_value=(document([entry(item['id'])]),[item])), \
         patch('tools.t04_06_full_horizon.verify'), \
         patch('tools.t04_06_full_horizon.latest_checkpoint',return_value=None), \
         patch('tools.t04_06_full_horizon.subprocess.run',side_effect=sbatch):
        with pytest.raises(RuntimeError,match='unreachable'):
            submit_wave(tmp_path/'runtime.json','all',4,24,script)
    assert json.loads(fence(tmp_path/'runtime.json',item).read_text())['phase']=='before_sbatch'
