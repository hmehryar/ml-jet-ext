import pytest
from tools import t04_06_reconcile as r


def entry(status='submitted'):
    return {'id':'x', 'status': status, 'attempts': [{'id':'attempt','slurm_job_id':'123_0'}],
            'run':'/nonexistent'}


def test_parse_sacct_ignores_steps():
    text='123_0|COMPLETED|0:0\n123_0.batch|COMPLETED|0:0\n123_1|OUT_OF_MEMORY|1:0\n'
    result=r.parse_sacct(text)
    assert result == {'123_0': {'state':'COMPLETED','exit_code':'0:0'},
                      '123_1': {'state':'OUT_OF_MEMORY','exit_code':'1:0'}}


def test_no_false_success_without_artifacts():
    choice=r.decide(entry(), {'123_0':{'state':'COMPLETED','exit_code':'0:0'}})
    assert choice['action']=='none'
    assert 'artifacts insufficient' in choice['reason']


def test_slurm_states_and_missing_are_conservative(monkeypatch):
    assert r.decide(entry(),{})['action']=='none'
    assert r.decide(entry(),{'123_0':{'state':'PENDING','exit_code':'0:0'}})['action']=='none'
    assert r.decide(entry(),{'123_0':{'state':'RUNNING','exit_code':'0:0'}})['action']=='running'
    assert r.decide(entry(),{'123_0':{'state':'OUT_OF_MEMORY','exit_code':'1:0'}})['action']=='failed'
    monkeypatch.setattr(r,'evidence',lambda e: {'ok':True,'reason':'verified','checkpoint_sha256':'a'*64,'epoch':80,'early_stopped':False})
    assert r.decide(entry(),{'123_0':{'state':'COMPLETED','exit_code':'0:0'}})['action']=='completed'


def test_read_only_reconciliation_never_transitions(monkeypatch):
    monkeypatch.setattr(r,'read',lambda p:{'entries': {'x':entry()}})
    monkeypatch.setattr(r,'transition',lambda *a,**kw: pytest.fail('mutated read only'))
    report=r.reconcile('unused',scheduler={'123_0':{'state':'RUNNING','exit_code':'0:0'}})
    assert report[0]['action']=='running'


def test_parse_sacct_single_job():
    """Accept regular Slurm jobs and exclude batch/extern steps."""
    from tools.t04_06_reconcile import parse_sacct

    raw = (
        "40941279|COMPLETED|0:0\n"
        "40941279.batch|COMPLETED|0:0\n"
        "40941279.extern|COMPLETED|0:0\n"
    )

    result = parse_sacct(raw)

    assert result == {
        "40941279": {
            "state": "COMPLETED",
            "exit_code": "0:0",
        }
    }


def test_parse_sacct_array_task():
    """Preserve compatibility with Slurm array-task IDs."""
    from tools.t04_06_reconcile import parse_sacct

    raw = (
        "40941144_0|COMPLETED|0:0\n"
        "40941144_0.batch|COMPLETED|0:0\n"
        "40941144_1|FAILED|1:0\n"
    )

    result = parse_sacct(raw)

    assert result == {
        "40941144_0": {
            "state": "COMPLETED",
            "exit_code": "0:0",
        },
        "40941144_1": {
            "state": "FAILED",
            "exit_code": "1:0",
        },
    }
