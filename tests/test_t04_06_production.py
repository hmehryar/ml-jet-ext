from unittest.mock import patch
import pytest
from tools.t04_06_production import plan, assert_prior_terminal, candidates


def make_doc():
    entries={}
    for model in ('clip','siglip'):
        for strategy in ('s1','s2','s3','s4'):
            name=f'{model}_{strategy}_e5_seed42'
            entries[name]={'id':name,'status':'planned','attempts':[],'run':'/tmp/'+name}
    entries['clip_s2_e5_seed42']['status']='ready_to_resume'
    return {'entries':entries}


def test_selection_excludes_accepted_experiment():
    doc=make_doc()
    assert len(candidates(doc))==7
    assert all(e['id']!='clip_s2_e5_seed42' for e in candidates(doc))


def test_max_four_and_first_wave_three(tmp_path):
    doc=make_doc()
    with patch('tools.t04_06_production.records',return_value=(doc,candidates(doc))), \
         patch('tools.t04_06_production.intent') as f:
        f.return_value.exists.return_value=False
        assert len(plan(tmp_path/'state',3))==3
        with pytest.raises(ValueError,match='1..4'):
            plan(tmp_path/'state',5)


def test_refuses_unreconciled_previous_wave(tmp_path):
    doc=make_doc()
    e=candidates(doc)[0]
    e['status']='submitted'
    e['attempts']=[{'slurm_job_id':'9001'}]
    with patch('tools.t04_06_production.records',return_value=(doc,candidates(doc))):
        with pytest.raises(RuntimeError,match='not reconciled'):
            plan(tmp_path/'state',3)


def test_verified_terminal_wave_allows_next(tmp_path):
    doc=make_doc()
    e=candidates(doc)[0]
    e['status']='ready_to_resume'
    e['attempts']=[{'slurm_job_id':'9001'}]
    with patch('tools.t04_06_production.records',return_value=(doc,candidates(doc))), \
         patch('tools.t04_06_production.query_sacct',return_value={'9001':{'state':'COMPLETED','exit_code':'0:0'}}), \
         patch('tools.t04_06_production.intent') as f:
        f.return_value.exists.return_value=False
        assert len(plan(tmp_path/'state',4))==4


def test_ambiguous_slurm_blocks_next_wave():
    e={'id':'clip_s1_e5_seed42','status':'ready_to_resume','attempts':[{'slurm_job_id':'9001'}]}
    with patch('tools.t04_06_production.query_sacct',return_value={}):
        with pytest.raises(RuntimeError,match='ambiguous'):
            assert_prior_terminal({},[e])
