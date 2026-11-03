from pathlib import Path
from unittest.mock import patch
from tools.t04_06_reconcile import parse_sacct, decide
from tools.t04_06_registry import VALID, TRANSITIONS
from tools.t04_06_resume_gate import fence_for


def test_regular_and_array_slurm_ids():
    assert parse_sacct('40941279|COMPLETED|0:0\n40941279.batch|COMPLETED|0:0\n') == {
        '40941279': {'state':'COMPLETED','exit_code':'0:0'}}
    assert '41_0' in parse_sacct('41_0|FAILED|1:0\n41_0.extern|COMPLETED|0:0\n')


def test_partial_segment_decision():
    e={'status':'submitted','attempts':[{'slurm_job_id':'40941279'}]}
    with patch('tools.t04_06_reconcile.evidence',return_value={
        'ok':False,'resume_eligible':True,'epoch':12,
        'checkpoint_sha256':'a'*64,'reason':'valid partial segment'}):
        result=decide(e,{'40941279':{'state':'COMPLETED','exit_code':'0:0'}})
    assert result['action']=='ready_to_resume'
    assert result['epoch']==12


def test_partial_state_machine():
    assert 'ready_to_resume' in VALID
    assert 'ready_to_resume' in TRANSITIONS['running']
    assert 'resumed' in TRANSITIONS['ready_to_resume']
    assert 'completed' not in TRANSITIONS['ready_to_resume']


def test_intent_location(tmp_path):
    assert fence_for(tmp_path/'registry.runtime.json','clip_s2_e5_seed42') == \
        tmp_path/'resume3d2.clip_s2_e5_seed42.intent.json'
