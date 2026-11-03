import json
from pathlib import Path
import yaml
import pytest
from tools.t04_06_matrix import plan, generate, verify


def repo(tmp_path):
    base=tmp_path/'experiments/t04/config/t04_05_smoke'
    base.mkdir(parents=True)
    for model in ('clip','siglip'):
        (base/f'{model}_s2_e5_seed42.yml').write_text(yaml.safe_dump(dict(
            model=model,precision='bf16',moments=[1,2,3,4,5],batch_size=8,
            gradient_accumulation=1,split_protocol='aggregate_disjoint',
            checkpoint='pinned',checkpoint_revision='a'*40),sort_keys=False))
    return tmp_path


def test_factorial(tmp_path):
    r=repo(tmp_path)
    entries=plan(r)
    assert len(entries)==120
    assert len({x['run'] for x in entries})==120
    assert set(x['moments'] for x in entries)=={'E1','E2','E3','E4','E5'}
    assert set(x['seed'] for x in entries)=={42,123,2026}


def test_write_verify_and_no_overwrite(tmp_path):
    r=repo(tmp_path)
    reg=generate(r)
    assert len(verify(r,reg))==120
    with pytest.raises(FileExistsError):
        generate(r)
    conf=r/'experiments/t04/config/t04_06/clip_s1_e1_seed42.yml'
    conf.write_text(conf.read_text()+'\n')
    with pytest.raises(ValueError,match='hash'):
        verify(r,reg)


def test_source_template_pinning(tmp_path):
    r=repo(tmp_path)
    reg=generate(r)
    template=r/'experiments/t04/config/t04_05_smoke/clip_s2_e5_seed42.yml'
    template.write_text(template.read_text()+'\n')
    with pytest.raises(ValueError):
        verify(r,reg)
