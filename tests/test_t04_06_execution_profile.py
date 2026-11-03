import pytest

from train_utils.t04_session_worker import resolve_dataloader_workers


def config(protocol="fixed_budget_v2"):
    return {
        "num_workers": 2,
        "training_protocol": protocol,
    }


def profile_env():
    return {
        "T04_EXECUTION_PROFILE": "cpu1_workers0_v1",
        "T04_DATALOADER_WORKERS": "0",
        "SLURM_CPUS_PER_TASK": "1",
    }


def test_frozen_configuration_is_default():
    assert resolve_dataloader_workers(
        config(), env={}
    ) == (2, "frozen_config")


def test_new_profile_uses_zero_workers():
    cfg = config()
    assert resolve_dataloader_workers(
        cfg, env=profile_env()
    ) == (0, "cpu1_workers0_v1")
    assert cfg["num_workers"] == 2


def test_legacy_protocol_rejected():
    with pytest.raises(ValueError, match="fixed_budget_v2"):
        resolve_dataloader_workers(
            config("early_stop_v1"),
            env=profile_env(),
        )


def test_cpu_mismatch_rejected():
    env = profile_env()
    env["SLURM_CPUS_PER_TASK"] = "2"
    with pytest.raises(ValueError, match="Slurm CPU"):
        resolve_dataloader_workers(config(), env=env)


def test_worker_mismatch_rejected():
    env = profile_env()
    env["T04_DATALOADER_WORKERS"] = "2"
    with pytest.raises(ValueError, match="zero DataLoader"):
        resolve_dataloader_workers(config(), env=env)


def test_unknown_profile_rejected():
    env = profile_env()
    env["T04_EXECUTION_PROFILE"] = "unknown"
    with pytest.raises(ValueError, match="Unknown"):
        resolve_dataloader_workers(config(), env=env)
