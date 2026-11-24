from pathlib import Path
import json
import yaml
import pytest
from tools.t04_06_v2_matrix import build,generate,verify,PLAN,OUTPUT,CONFIG_DIR
from train_utils.t04_training_policy import policy


def repo(tmp_path):
    p=tmp_path/'repo'
    d=p/'experiments/t04/config/t04_05_smoke'
    d.mkdir(parents=True)
    for model in ('clip','siglip'):
        # Provide only template fields the planner needs; full schema covered by HPC preflight.
        template=dict(model=model,precision='bf16',moments=[1,2,3,4,5],
                      batch_size=8,gradient_accumulation=1,split_protocol='aggregate_disjoint',
                      min_epochs=12,patience=16)
        (d/f'{model}_s2_e5_seed42.yml').write_text(yaml.safe_dump(template))
    return p


def test_count_and_identity_isolation(tmp_path):
    p=repo(tmp_path)
    doc,files=build(p)
    assert len(doc['entries'])==len(files)==120
    assert len({r['id'] for r in doc['entries']})==120
    assert len({r['run'] for r in doc['entries']})==120
    assert all('/t04_06_v2_production/' in r['run'] and r['run'].endswith('-v2') for r in doc['entries'])
    assert all('/t04_06_production/' not in r['run'] for r in doc['entries'])
    assert list(doc['seeds'])==[42,123,2026]
    assert doc['training_protocol_hash']==policy(80).digest()
    assert not (p/PLAN).exists()


def test_budget_is_configured_not_hardcoded(tmp_path):
    p=repo(tmp_path)
    doc,files=build(p,epochs=120)
    assert doc['epochs']==120
    assert doc['training_protocol_hash']==policy(120).digest()
    assert all(yaml.safe_load(v)['epochs']==120 for v in files.values())
    assert all(yaml.safe_load(v)['training_protocol']=='fixed_budget_v2' for v in files.values())


def test_generate_and_verify_are_immutable(tmp_path):
    p=repo(tmp_path)
    generate(p)
    assert len(verify(p,p/PLAN))==120
    with pytest.raises(FileExistsError):
        generate(p)
    first=next((p/CONFIG_DIR).glob('*.yml'))
    first.write_text(first.read_text()+'\n# tampered\n')
    with pytest.raises(ValueError):
        verify(p,p/PLAN)


def test_template_change_invalidates_existing_plan(tmp_path):
    p=repo(tmp_path)
    generate(p)
    template=p/'experiments/t04/config/t04_05_smoke/clip_s2_e5_seed42.yml'
    template.write_text(template.read_text()+'\n# upstream changed\n')
    with pytest.raises(ValueError):
        verify(p,p/PLAN)


def test_invalid_epoch_budget_rejected(tmp_path):
    p=repo(tmp_path)
    doc,files=build(p,epochs=4)
    assert doc['epochs']==4
    assert all(yaml.safe_load(v)['min_epochs']==4 for v in files.values())
    with pytest.raises(ValueError):
        build(p,epochs=0)


def test_no_legacy_config_mutation(tmp_path):
    p=repo(tmp_path)
    legacy=p/'training_output/T04/t04_06/registry.json'
    legacy.parent.mkdir(parents=True)
    legacy.write_text('DO NOT TOUCH')
    generate(p)
    assert legacy.read_text()=='DO NOT TOUCH'


def _copy_installed_schema_sources(tmp_path):
    """Copy validated current source into an isolated read-only test fixture."""
    from pathlib import Path
    from tools.t04_06_patch_v2_schema import PREFLIGHT, REGISTRY
    root = Path(__file__).resolve().parents[1]
    for name in (PREFLIGHT, REGISTRY):
        target = tmp_path / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((root / name).read_bytes())
    return tmp_path


def test_schema_patcher_preview_is_read_only(tmp_path):
    from tools.t04_06_patch_v2_schema import PREFLIGHT, REGISTRY, patch
    root = _copy_installed_schema_sources(tmp_path)
    before = {name: (root / name).read_bytes() for name in (PREFLIGHT, REGISTRY)}
    assert patch(root, apply=False) == 0
    assert before == {name: (root / name).read_bytes() for name in (PREFLIGHT, REGISTRY)}


def test_schema_patcher_idempotence_guard(tmp_path):
    from tools.t04_06_patch_v2_schema import PREFLIGHT, REGISTRY, patch
    root = _copy_installed_schema_sources(tmp_path)
    before = {name: (root / name).read_bytes() for name in (PREFLIGHT, REGISTRY)}
    assert patch(root, apply=True) == 0  # Already applied is a safe no-op.
    assert patch(root, apply=True) == 0
    assert before == {name: (root / name).read_bytes() for name in (PREFLIGHT, REGISTRY)}


def test_verify_accepts_existing_prepared_run_directory(tmp_path):
    """Immutable matrix verification remains valid after execution begins."""
    p = repo(tmp_path)
    generate(p)

    doc = json.loads((p / PLAN).read_text())
    run = Path(doc["entries"][0]["run"])
    run.mkdir(parents=True)
    (run / "checkpoint.marker").write_text("preserved")

    assert len(verify(p, p / PLAN)) == 120
    assert (run / "checkpoint.marker").read_text() == "preserved"
