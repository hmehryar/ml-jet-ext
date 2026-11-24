"""Tests Slurm-state-aware reservation; never calls the real Slurm scheduler."""
from types import SimpleNamespace
import subprocess
import pytest
from tools.t04_06_v2_full_horizon import _unallocated_active_entries, _cost, _active_profile


def entry(job, profile='cpu1_workers0_v1'):
    return {'status': 'submitted', 'attempts': [{'slurm_job_id': str(job)}],
            'events': [{'note': 'resource_profile=' + profile}]}


def mock_squeue(monkeypatch, output):
    def fake_run(argv, **kwargs):
        assert argv[:4] == ['squeue', '-h', '-j', argv[3]]
        assert kwargs['timeout'] == 15
        return SimpleNamespace(stdout=output)
    monkeypatch.setattr('tools.t04_06_v2_full_horizon.subprocess.run', fake_run)


def test_four_running_jobs_not_subtracted_again(monkeypatch):
    doc = {'entries': {str(i): entry(i) for i in range(1, 5)}}
    mock_squeue(monkeypatch, ''.join(f'{i}|RUNNING|csc1\n' for i in range(1, 5)))
    pending = _unallocated_active_entries(doc)
    resources = [7, 101888, 8]
    for item in pending:
        cost = _cost(_active_profile(item))
        resources = [a-b for a, b in zip(resources, cost)]
    assert pending == []
    assert resources == [7, 101888, 8]


def test_pending_held_jobs_are_reserved(monkeypatch):
    doc = {'entries': {'a': entry(1), 'b': entry(2), 'c': entry(3, 'frozen_config_cpu2_v1')}}
    mock_squeue(monkeypatch, '1|RUNNING|csc1\n2|PENDING|(null)\n3|PENDING|(null)\n')
    pending = _unallocated_active_entries(doc)
    assert [p['attempts'][-1]['slurm_job_id'] for p in pending] == ['2', '3']
    assert sum(_cost(_active_profile(p))[0] for p in pending) == 3


@pytest.mark.parametrize('stdout', [
    '1|RUNNING|other\n',
    '1|COMPLETED|csc1\n',
    '1|CONFIGURING|csc1\n',
    '1|PENDING|csc1\n',
    '',
    '1|RUNNING|csc1\n1|RUNNING|csc1\n',
])
def test_unknown_or_inconsistent_states_fail_closed(monkeypatch, stdout):
    mock_squeue(monkeypatch, stdout)
    with pytest.raises(RuntimeError):
        _unallocated_active_entries({'entries': {'a': entry(1)}})


def test_no_active_jobs_do_not_query_scheduler(monkeypatch):
    def fail(*args, **kwargs):
        raise AssertionError('Unexpected scheduler call')
    monkeypatch.setattr('tools.t04_06_v2_full_horizon.subprocess.run', fail)
    assert _unallocated_active_entries({'entries': {}}) == []


def test_duplicate_slurm_ids_fail_closed(monkeypatch):
    with pytest.raises(RuntimeError, match='duplicate'):
        _unallocated_active_entries({'entries': {'a': entry(1), 'b': entry(1)}})
