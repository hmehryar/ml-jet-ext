import json
from pathlib import Path
from unittest.mock import patch, Mock
import pytest
from tools.t04_06_gate import dry_run, submit_held, intent

def fixture(tmp_path):
    cfg=tmp_path/"config.yml"
    cfg.write_text("frozen")
    run=tmp_path/"run"
    run.mkdir()
    (run/"source").mkdir()
    (run/"source/config.original.yml").write_text("frozen")
    from tools.t04_06_registry import sha
    e=dict(id="clip_s2_e5_seed42",config="config.yml",config_sha256=sha(cfg),
           status="planned",attempts=[],run=str(run))
    doc=dict(repo=str(tmp_path),entries={e["id"]:e})
    spec=Mock(min_epochs=12,epochs=80,split_protocol="aggregate_disjoint",
              precision="bf16",batch_size=8,gradient_accumulation=1)
    return doc,e,cfg,run,spec

def test_dry_run_no_side_effects(tmp_path):
    doc,e,cfg,run,spec=fixture(tmp_path)
    with patch("tools.t04_06_gate.selection",return_value=(doc,e,cfg,run)),\
         patch("tools.t04_06_gate.preflight",return_value=(spec,{})):
        assert dry_run(tmp_path/"registry.runtime.json",e["id"])==12
    assert not intent(tmp_path/"registry.runtime.json",e["id"]).exists()

def test_held_submission_registers_without_release(tmp_path):
    doc,e,cfg,run,spec=fixture(tmp_path)
    script=tmp_path/"gate.slurm";script.write_text("#!/bin/bash\n")
    calls=[]
    def fake_run(cmd,**kwargs):
        calls.append(cmd)
        if cmd[0]=="sbatch":
            return Mock(stdout="1005\n")
        return Mock(stdout="")
    with patch("tools.t04_06_gate.selection",return_value=(doc,e,cfg,run)),\
         patch("tools.t04_06_gate.preflight",return_value=(spec,{})),\
         patch("tools.t04_06_gate.subprocess.run",side_effect=fake_run),\
         patch("tools.t04_06_gate.transition") as trans:
        submit_held(tmp_path/"registry.runtime.json",e["id"],script)
        assert "--hold" in calls[1]
        assert "T04_06_TARGET=12" in next(c for c in calls[1] if c.startswith("--export"))
        assert all(c[0]!="scontrol" for c in calls)
        trans.assert_called_once()
    f=intent(tmp_path/"registry.runtime.json",e["id"])
    assert json.loads(f.read_text())["phase"]=="registered_held"

def test_intent_blocks_duplicate(tmp_path):
    doc,e,cfg,run,spec=fixture(tmp_path)
    script=tmp_path/"gate.slurm";script.write_text("#!/bin/bash\n")
    f=intent(tmp_path/"registry.runtime.json",e["id"])
    f.write_text("{}")
    with pytest.raises(RuntimeError,match="Intent exists"):
        submit_held(tmp_path/"registry.runtime.json",e["id"],script)

def test_submission_uncertainty_retains_fence(tmp_path):
    doc,e,cfg,run,spec=fixture(tmp_path)
    script=tmp_path/"gate.slurm";script.write_text("#!/bin/bash\n")
    def fail(cmd,**kwargs):
        if cmd[0]=="sbatch":
            raise RuntimeError("network failure")
        return Mock()
    with patch("tools.t04_06_gate.selection",return_value=(doc,e,cfg,run)),\
         patch("tools.t04_06_gate.preflight",return_value=(spec,{})),\
         patch("tools.t04_06_gate.subprocess.run",side_effect=fail):
        with pytest.raises(RuntimeError,match="network failure"):
            submit_held(tmp_path/"registry.runtime.json",e["id"],script)
    assert json.loads(intent(tmp_path/"registry.runtime.json",e["id"]).read_text())["phase"]=="before_sbatch"
