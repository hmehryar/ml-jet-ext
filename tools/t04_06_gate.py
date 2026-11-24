"""T04-06 Stage 3D: single-run production gate, deliberately manual release.

No automatic Slurm release, retry or recovery. One held job with durable intent
and identity-bound registry transition; operator must separately authorize release.
"""
import argparse
import json
import os
from pathlib import Path
import re
import subprocess

from tools.t04_06_registry import read, locked, atomic_replace, transition, sha
from train_utils.t04_preflight import preflight, prepare, verify
from train_utils.t04_sessions import latest_checkpoint

DEFAULT_RUNTIME = Path("training_output/T04/t04_06/registry.runtime.json")
IDENTITY = "clip_s2_e5_seed42"
JOB_PATTERN = re.compile(r"^\d+$")


def selection(runtime, identity=IDENTITY):
    doc = read(runtime)
    e = doc['entries'][identity]
    cfg = Path(doc['repo']) / e['config']
    if sha(cfg) != e['config_sha256']:
        raise ValueError("Configuration hash differs from registry")
    if e['status'] != 'planned' or e['attempts']:
        raise ValueError("Run already submitted or no longer planned")
    run = Path(e['run'])
    if run.exists():
        verify(run)
        if (run/'source/config.original.yml').read_bytes() != cfg.read_bytes():
            raise ValueError("Frozen config mismatch")
        if latest_checkpoint(run) or (run/'sessions').exists():
            raise ValueError("Run already has training history")
    return doc, e, cfg, run


def intent(runtime, identity):
    return Path(runtime).resolve().parent / ("gate3d." + identity + ".intent.json")


def dry_run(runtime, identity):
    doc,e,cfg,run=selection(runtime, identity)
    config,inputs=preflight(cfg)
    print("DRY_RUN", identity, "prepared="+str(run.exists()),
          "min_epochs="+str(config.min_epochs),"epochs="+str(config.epochs),
          "split_protocol="+config.split_protocol)
    if config.split_protocol != 'aggregate_disjoint':
        raise ValueError("Unexpected dataset protocol")
    if config.precision != 'bf16' or config.batch_size != 8 or config.gradient_accumulation != 1:
        raise ValueError("T04-06 gated profile differs")
    if intent(runtime,identity).exists():
        raise ValueError("Existing submission intent: manual reconciliation required")
    return config.min_epochs


def prepare_one(runtime, identity):
    with locked(intent(runtime,identity)):
        doc,e,cfg,run=selection(runtime,identity)
        preflight(cfg)
        if not run.exists():
            actual=Path(prepare(cfg,run.name)).resolve()
            if actual != run.resolve():
                raise RuntimeError("Prepared output differs from planned path")
        verify(run)
        print("PREPARED",identity,run)


def submit_held(runtime, identity, script):
    """Held submission only. Manual release after human inspection.

    Fence persists after success and in every uncertain failure case.
    """
    fence=intent(runtime,identity)
    with locked(fence):
        if fence.exists():
            raise RuntimeError("Intent exists; never repeat sbatch automatically")
        doc,e,cfg,run=selection(runtime,identity)
        if not run.exists():
            raise RuntimeError("Prepare immutable run before submission")
        config,_=preflight(cfg)
        if config.min_epochs > config.epochs:
            raise ValueError("Invalid horizon")
        script=Path(script).resolve()
        if not script.is_file():
            raise FileNotFoundError(script)
        subprocess.run(["bash","-n",str(script)],check=True)
        atomic_replace(fence,{"schema":1,"phase":"before_sbatch",
                              "identity":identity,"run":str(run),"job_id":None})
        cmd=["sbatch","--parsable","--hold",
             "--export=ALL,T04_06_RUN="+str(run)+",T04_06_MODE=fresh,T04_06_TARGET="+str(config.min_epochs),
             str(script)]
        try:
            proc=subprocess.run(cmd,cwd=doc['repo'],check=True,text=True,capture_output=True)
            job=proc.stdout.strip().split(";")[0]
            if not JOB_PATTERN.fullmatch(job):
                raise ValueError("Unknown Slurm response")
            atomic_replace(fence,{"schema":1,"phase":"held_submitted",
                                  "identity":identity,"run":str(run),"job_id":job})
            transition(runtime,identity,'submitted',job_id=job,
                       note="Stage 3D one-run held acceptance; manual release required")
            atomic_replace(fence,{"schema":1,"phase":"registered_held",
                                  "identity":identity,"run":str(run),"job_id":job})
            print("HELD_REGISTERED",job,identity)
            print("DO_NOT_RELEASE until verifying registry and Slurm job resources")
        except Exception:
            print("Submission uncertain; inspect intent + Slurm. No automatic retry.")
            raise


def main():
    p=argparse.ArgumentParser()
    p.add_argument("action",choices=["dry-run","prepare-one","submit-held","intent"])
    p.add_argument("--runtime",type=Path,default=DEFAULT_RUNTIME)
    p.add_argument("--id",default=IDENTITY)
    p.add_argument("--script",type=Path,default=Path("slurm/t04_06_single.slurm"))
    a=p.parse_args()
    a.runtime=a.runtime.resolve()
    if a.action == "dry-run":
        dry_run(a.runtime,a.id)
    elif a.action == "prepare-one":
        prepare_one(a.runtime,a.id)
    elif a.action == "submit-held":
        submit_held(a.runtime,a.id,a.script)
    else:
        f=intent(a.runtime,a.id)
        print(f.read_text() if f.exists() else "NO_INTENT")


if __name__=="__main__":
    main()
