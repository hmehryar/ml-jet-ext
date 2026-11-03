import pytest
from train_utils.t04_training_policy import policy
from train_utils.t04_training_termination import (
    validate_protocol, may_resume, should_stop)

H = policy().digest()


def test_legacy_early_stopping_preserved():
    assert should_stop(protocol_id='early_stop_v1', epoch=28,
                       min_epochs=12, stagnant=16, patience=16)
    assert not should_stop(protocol_id='early_stop_v1', epoch=27,
                           min_epochs=12, stagnant=15, patience=16)


def test_legacy_stopped_checkpoint_still_rejected():
    assert not may_resume(protocol_id='early_stop_v1', stopped=True)
    assert may_resume(protocol_id='early_stop_v1', stopped=False)


def test_v2_disables_plateau_stop_even_at_80():
    for epoch in (12, 28, 31, 60, 80):
        assert not should_stop(protocol_id='fixed_budget_v2', protocol_hash=H, epochs=80,
                               epoch=epoch, min_epochs=12, stagnant=100, patience=16)


def test_v2_stopped_flag_is_not_early_stop_guard():
    assert may_resume(protocol_id='fixed_budget_v2', protocol_hash=H, epochs=80, stopped=True)


def test_v2_requires_hash():
    with pytest.raises(ValueError):
        validate_protocol('fixed_budget_v2', epochs=80)
    with pytest.raises(ValueError):
        validate_protocol('fixed_budget_v2', 'incorrect', epochs=80)


def test_unknown_protocol_rejected():
    with pytest.raises(ValueError):
        validate_protocol('anything')


def test_v2_never_runs_past_configured_budget():
    with pytest.raises(ValueError):
        should_stop(protocol_id='fixed_budget_v2', protocol_hash=H, epochs=80,
                    epoch=81, min_epochs=12, stagnant=99, patience=16)


def test_custom_epoch_budget_and_hash():
    h = policy(120).digest()
    assert h != policy(80).digest()
    assert not should_stop(protocol_id='fixed_budget_v2', protocol_hash=h, epochs=120,
                           epoch=110, min_epochs=12, stagnant=99, patience=16)
    with pytest.raises(ValueError):
        validate_protocol('fixed_budget_v2', h, epochs=80)
    with pytest.raises(ValueError):
        should_stop(protocol_id='fixed_budget_v2', protocol_hash=h, epochs=120,
                    epoch=121, min_epochs=12, stagnant=99, patience=16)
