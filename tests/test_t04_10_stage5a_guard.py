"""Isolation and lifecycle tests; no Slurm, real training, or /rs writes."""
import hashlib
import json
from pathlib import Path
import subprocess

import pytest

from tools.t04_10_stage5a_guard import guard


def fixture_run(tmp_path):
    root=tmp_path/'t04_10_single_head'
    run=root/'clip'/'alpha_only'/'E3'/'seed-42'/'pilot-a'
    run.mkdir(parents=True)
    cfg={'model':'clip','task':'alpha_only','seed':42,'moments':[1,2,3],
         'output_root':str(root),'split_protocol':'aggregate_disjoint',
         'epochs':80,'training_protocol':'fixed_budget_v2'}
    manifest={'schema':1,'issue':30,'status':'prepared','git_branch':'feat/t04-10-single-head-pilot',
              'git_commit':'a'*40,'run_id':'pilot-a','test_set_evaluated':False,
              'inputs':{'source_overlap':103221}}
    (run/'config.resolved.json').write_text(json.dumps(cfg))
    (run/'run_manifest.json').write_text(json.dumps(manifest))
    return root,run,cfg,manifest


def test_accept_fresh_prepared_run(tmp_path):
    root,run,_,_=fixture_run(tmp_path)
    result=guard(run,'fresh',root=root,verifier=lambda p: True)
    assert result['status']=='GUARD_PASS'
    assert result['source_overlap']==103221


def test_reject_outside_root(tmp_path):
    root,run,_,_=fixture_run(tmp_path)
    with pytest.raises(ValueError,match='outside dedicated'):
        guard(run,'fresh',root=tmp_path/'other',verifier=lambda p: True)


def test_reject_modified_manifest_run_id(tmp_path):
    root,run,_,manifest=fixture_run(tmp_path)
    manifest['run_id']='another'
    (run/'run_manifest.json').write_text(json.dumps(manifest))
    with pytest.raises(ValueError,match='canonical'):
        guard(run,'fresh',root=root,verifier=lambda p: True)


def test_reject_false_source_disjoint(tmp_path):
    root,run,cfg,_=fixture_run(tmp_path)
    cfg['split_protocol']='source_disjoint'
    (run/'config.resolved.json').write_text(json.dumps(cfg))
    with pytest.raises(ValueError,match='False source-disjoint'):
        guard(run,'fresh',root=root,verifier=lambda p: True)


def test_reject_fresh_collision(tmp_path):
    root,run,_,_=fixture_run(tmp_path)
    (run/'training').mkdir()
    with pytest.raises(ValueError,match='Fresh mode forbidden'):
        guard(run,'fresh',root=root,verifier=lambda p: True)


def test_resume_requires_checkpoint_history(tmp_path):
    root,run,_,_=fixture_run(tmp_path)
    with pytest.raises(ValueError,match='Resume requires'):
        guard(run,'resume',root=root,verifier=lambda p: True)
    training=run/'training';training.mkdir()
    (training/'last.pth').write_bytes(b'not used')
    (training/'history.json').write_text('[]')
    assert guard(run,'resume',root=root,verifier=lambda p: True)['mode']=='resume'


def test_reject_snapshot_failure(tmp_path):
    root,run,_,_=fixture_run(tmp_path)
    with pytest.raises(ValueError,match='snapshot verifier'):
        guard(run,'fresh',root=root,verifier=lambda p: False)


def test_shell_scripts_safe_and_held():
    repo=Path(__file__).resolve().parents[1]
    for file in ('tools/t04_10_submit_held.sh','slurm/t04_10_single_head_held.slurm'):
        subprocess.run(['bash','-n',str(repo/file)],check=True)
    submit=(repo/'tools/t04_10_submit_held.sh').read_text()
    slurm=(repo/'slurm/t04_10_single_head_held.slurm').read_text()
    assert 'sbatch --parsable --hold' in submit
    assert 'if (( APPLY == 0 ))' in submit
    assert '/t04_10_single_head/slurm_logs/' in slurm
    assert 't04_07' not in slurm.lower()
    assert '#SBATCH --gres=gpu:nvidia_a100_80gb_pcie_1g.10gb:1' in slurm
