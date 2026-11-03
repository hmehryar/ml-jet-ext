"""CPU-only safety tests for the Stage 3F fixed-budget policy audit."""
from copy import deepcopy
import pytest
from train_utils.t04_training_policy import TrainingPolicy, policy
from tools.t04_06_protocol_audit import inspect


def registry(states):
    entries = {}
    for i in range(120):
        state = states.get(i, 'planned')
        attempts = ([{'slurm_job_id': str(50000+i)}]
                    if state != 'planned' else [])
        entries[f'exp_{i:03d}'] = {'status':state,'attempts':attempts,
                                   'run':f'/frozen/exp_{i:03d}'}
    return {'entries':entries}


def test_policy_is_fixed_and_hashed():
    p = policy()
    assert p.epochs == 80 and not p.early_stopping_enabled
    assert len(p.digest()) == 64 and p.digest() == TrainingPolicy(epochs=80).digest()
    assert policy(100).epochs == 100
    assert policy(100).digest() != policy(80).digest()
    with pytest.raises(ValueError):
        TrainingPolicy(epochs=80, early_stopping_enabled=True)


def test_old_early_stop_does_not_stop_new_protocol():
    p = policy()
    assert not p.terminal(epoch=20, old_stopped=True)
    assert not p.terminal(epoch=79, old_stopped=True)
    assert p.terminal(epoch=80, old_stopped=False)
    with pytest.raises(ValueError):
        p.terminal(epoch=81)


def test_active_jobs_block_migration_even_if_sacct_says_completed():
    doc = registry({0:'resumed'})
    before = deepcopy(doc)
    entries = inspect(doc, {'50000':{'state':'COMPLETED','exit_code':'0:0'}})
    assert entries[0]['v2_action'] is None
    assert 'UNRECONCILED' in entries[0]['blocker']
    assert doc == before


def test_completed_and_ready_require_fork_after_checkpoint_verification():
    doc = registry({0:'completed', 1:'ready_to_resume'})
    items = inspect(doc, {})
    assert [x['v2_action'] for x in items[:2]] == ['verify_parent_checkpoint_then_fork_v2']*2
    assert items[2]['v2_action'].startswith('prepare_fresh_v2')


def test_active_requires_scheduler_evidence():
    with pytest.raises(ValueError, match='Slurm evidence'):
        inspect(registry({0:'submitted'}))


def test_failed_is_not_auto_migrated():
    item = inspect(registry({0:'failed'}),{})[0]
    assert item['v2_action'] is None
    assert 'MANUAL' in item['blocker']


def test_wrong_number_of_entries_rejected():
    with pytest.raises(ValueError,match='120'):
        inspect({'entries':{}}, {})
