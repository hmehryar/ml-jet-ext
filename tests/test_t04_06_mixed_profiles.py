"""Tests for one- and two-CPU production profiles; no jobs submitted."""
from pathlib import Path
from tools.t04_06_v2_full_horizon import _profile, _cost, _reserve, _active_profile, _check_script, PROFILE_ONE, PROFILE_TWO

def test_profile_costs():
    assert _cost(PROFILE_ONE) == (1, 12288, 1)
    assert _cost(PROFILE_TWO) == (2, 12288, 1)

def test_mixed_capacity():
    resources=[15,175616,13]
    for _ in range(4):assert _reserve(resources,PROFILE_TWO)
    for _ in range(7):assert _reserve(resources,PROFILE_ONE)
    assert resources[0]==0
    assert not _reserve(resources,PROFILE_ONE)

def test_profile_selection(tmp_path):
    run=tmp_path/'experiment'
    assert _profile({'run':str(run),'mode':'fresh'})==PROFILE_ONE
    worker=run/'source/train_utils/t04_session_worker.py'
    worker.parent.mkdir(parents=True)
    worker.write_text('def execute(): pass\n')
    assert _profile({'run':str(run),'mode':'resume'})==PROFILE_TWO
    worker.write_text('def resolve_dataloader_workers(): pass\n')
    assert _profile({'run':str(run),'mode':'resume'})==PROFILE_ONE

def test_registry_profile_events():
    assert _active_profile({'events':[]})==PROFILE_TWO
    assert _active_profile({'events':[{'note':'epoch=24 resource_profile=cpu1_workers0_v1'}]})==PROFILE_ONE

def test_slurm_profiles():
    repo=Path(__file__).resolve().parents[1]
    _check_script(repo/'slurm/t04_06_v2_single.slurm',PROFILE_ONE)
    _check_script(repo/'slurm/t04_06_v2_legacy_resume.slurm',PROFILE_TWO)
