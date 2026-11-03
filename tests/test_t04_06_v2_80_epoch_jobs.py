"""80-epoch full-horizon Slurm requests and immutable resume targets."""
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import pytest
from tools import t04_06_v2_full_horizon as launcher
from train_utils.t04_training_policy import policy


def config():
    return SimpleNamespace(epochs=80,min_epochs=12,patience=16,
        training_protocol='fixed_budget_v2',training_protocol_hash=policy(80).digest(),
        output_root='training_output/T04/t04_06_v2_production',precision='bf16')


def test_full_budget_enabled():
    assert launcher.MAX_SEGMENT==80
    assert launcher.MAX_ACTIVE==13


def test_fresh_target_is_80(tmp_path):
    config_file=tmp_path/'a.yml'
    config_file.write_text('demo')
    entry={'id':'clip_s1_e3_seed42','config':'a.yml','config_sha256':'a'*64,
           'run':str(tmp_path/'not_prepared'),'status':'planned','attempts':[]}
    with patch.object(launcher,'sha',return_value='a'*64),\
         patch.object(launcher,'preflight',return_value=(config(),{})):
        result=launcher._target(entry,tmp_path,80)
    assert (result['from_epoch'],result['target'],result['mode'])==(0,80,'fresh')


def test_slurm_production_walltime_and_profile():
    repo=Path(__file__).resolve().parents[1]
    for filename,cpus in (('t04_06_v2_single.slurm',1),('t04_06_v2_legacy_resume.slurm',2)):
        script=repo/'slurm'/filename
        body=script.read_text()
        assert '#SBATCH --time=72:00:00' in body
        assert f'#SBATCH --cpus-per-task={cpus}' in body
        assert '#SBATCH --mem=12G' in body
        assert '#SBATCH --gres=gpu:nvidia_a100_80gb_pcie_1g.10gb:1' in body


def test_count_cannot_exceed_max(tmp_path):
    with pytest.raises(ValueError, match='maximum concurrent'):
        launcher.plan(tmp_path/'nonexistent.json',scope='all',count=14,segment_epochs=80)
    with pytest.raises(ValueError, match='1..80'):
        launcher.plan(tmp_path/'nonexistent.json',scope='all',count=13,segment_epochs=81)
