import copy
import hashlib
import json
from pathlib import Path
import pytest
from tools.t04_06_registry import atomic_replace, read, transition, validate


def fixture(tmp_path):
    plan=tmp_path/'plan.json'; plan.write_text('{}')
    repo=tmp_path/'repo'; repo.mkdir()
    cfg=repo/'config'; cfg.write_text('model: clip\n')
    entries={}
    for i in range(120):
        name=f'job{i}'
        entries[name]={'id':name,'config':'config', 'config_sha256':hashlib.sha256(cfg.read_bytes()).hexdigest(),
                       'run':str(tmp_path/f'run{i}'),'git_commit':'a'*40,'checkpoint_revision':None,
                       'status':'planned','attempts':[], 'events':[]}
    path=tmp_path/'registry.json'
    atomic_replace(path,dict(schema=1,kind='T04-06-operational',plan_sha256=hashlib.sha256(plan.read_bytes()).hexdigest(),plan_path=str(plan),repo=str(repo),entries=entries))
    return path


def test_registry_lifecycle_and_guardrails(tmp_path):
    reg=fixture(tmp_path)
    a=transition(reg,'job0','submitted',job_id='900_0')
    with pytest.raises(ValueError, match='Illegal transition'):
        transition(reg,'job0','submitted',job_id='900_1')
    transition(reg,'job0','running',attempt_id=a)
    with pytest.raises(ValueError, match='checkpoint'):
        transition(reg,'job0','completed',attempt_id=a)
    transition(reg,'job0','completed',attempt_id=a,checkpoint_sha256='a'*64)
    b=transition(reg,'job0','resumed',job_id='901_0', checkpoint_sha256='a'*64)
    transition(reg,'job0','running',attempt_id=b)
    transition(reg,'job0','failed',attempt_id=b,note='OOM')
    assert len(read(reg)['entries']['job0']['attempts'])==2


def test_duplicate_job_cross_entries_and_config_tamper(tmp_path):
    reg=fixture(tmp_path)
    transition(reg,'job0','submitted',job_id='100_0')
    with pytest.raises(ValueError,match='Duplicate Slurm'):
        transition(reg,'job1','submitted',job_id='100_0')
    (tmp_path/'repo/config').write_text('changed')
    with pytest.raises(ValueError,match='Config changed'):
        read(reg)


def test_atomic_validates_uniqueness(tmp_path):
    reg=fixture(tmp_path)
    doc=read(reg)
    doc['entries']['job1']['run']=doc['entries']['job0']['run']
    with pytest.raises(ValueError,match='Duplicate output'):
        validate(doc)
