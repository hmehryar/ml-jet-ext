import csv
import json
from pathlib import Path
from unittest.mock import patch
from tools.t04_06_recovery import decision

def entry(status='failed'):
    return {'id':'clip_s2_e5_seed42','status':status,
            'config_sha256':'a'*64,'run':'/not-a-run',
            'attempts':[{'slurm_job_id':'123_0'}]}

def test_blocks_active_and_unknown_scheduler():
    assert 'active registry' in decision(entry('running'))['reason']
    assert 'scheduler terminal' in decision(entry())['reason']
    assert not decision(entry(),slurm_state='RUNNING')['eligible']

def test_blocks_fresh_and_no_checkpoint(tmp_path):
    e=entry(); e['run']=str(tmp_path)
    assert 'no valid' in _run(e,tmp_path,checkpoint=None)['reason']

def _run(e,tmp,checkpoint,session_status='completed',stopped=False,epoch=1,target=3):
    (tmp/'source').mkdir(exist_ok=True)
    (tmp/'source/config.original.yml').write_text('frozen')
    (tmp/'config.resolved.json').write_text(json.dumps({'epochs':target,'precision':'bf16'}))
    (tmp/'run_manifest.json').write_text('{}')
    (tmp/'sessions/session-001').mkdir(parents=True,exist_ok=True)
    (tmp/'sessions/session-001/session_manifest.json').write_text(
        json.dumps({'session':'session-001','status':session_status}))
    (tmp/'sessions/session-001/result.json').write_text(json.dumps({'epoch':epoch,'early_stopped':stopped}))
    with (tmp/'moment_epoch_metrics.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=['epoch'])
        writer.writeheader()
        writer.writerow({'epoch':epoch})
    from tools.t04_06_registry import sha
    e['config_sha256']=sha(tmp/'source/config.original.yml')
    with patch('tools.t04_06_recovery.verify'),patch('tools.t04_06_recovery.latest_checkpoint',return_value=checkpoint):
        return decision(e,slurm_state='FAILED',load_state=lambda *_:
                        {'epoch':epoch,'stopped':False,'optimizer_state_dict':{},
                         'scheduler_state_dict':{},'rng_state':{'present':True}})

def test_valid_closed_session_can_be_reviewed(tmp_path):
    e=entry();e['run']=str(tmp_path)
    pointer={'epoch':1,'sha256':'a'*64,'session':'session-001'}
    result=_run(e,tmp_path,pointer)
    assert result['eligible'] and result['action']=='review_resume'

def test_unfinished_session_blocks(tmp_path):
    e=entry();e['run']=str(tmp_path)
    pointer={'epoch':1,'sha256':'a'*64,'session':'session-001'}
    assert 'unfinished' in _run(e,tmp_path,pointer,session_status='running')['reason']

def test_early_stopping_and_horizon_block(tmp_path):
    e=entry();e['run']=str(tmp_path)
    p={'epoch':1,'sha256':'a'*64,'session':'session-001'}
    assert 'early stopping' in _run(e,tmp_path,p,stopped=True)['reason']
    assert 'horizon' in _run(e,tmp_path,p,stopped=False,target=1)['reason']
