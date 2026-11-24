"""Short segment worker patch is strict and idempotent."""
import pytest
from tools.t04_06_patch_v2_short_segment import transform, OLD, NEW

BASE = """def worker(cfg,target_epochs):
    validate_protocol(protocol_id, protocol_hash, configured_epochs)
""" + OLD

def test_short_v2_segments_bypass_legacy_min_epochs():
    result=transform(BASE)
    assert result.count(NEW)==1
    assert transform(result)==result
    # Evaluate only the condition from replacement in an isolated function.
    source="def check(cfg, target_epochs):\n"+NEW
    ns={}
    exec(source,ns)
    ns['check']({'training_protocol':'fixed_budget_v2','min_epochs':12},3)
    ns['check']({'training_protocol':'fixed_budget_v2','min_epochs':12},80)
    with pytest.raises(ValueError):
        ns['check']({'training_protocol':'early_stop_v1','min_epochs':12},3)
    with pytest.raises(ValueError):
        ns['check']({'min_epochs':12},3)
    ns['check']({'min_epochs':12},12)

def test_unknown_source_fail_closed():
    with pytest.raises(RuntimeError):
        transform(BASE.replace(OLD, '    pass\n'))

def test_requires_existing_protocol_validation():
    with pytest.raises(RuntimeError):
        transform(BASE.replace('validate_protocol(protocol_id, protocol_hash, configured_epochs)','pass'))

def test_rejects_duplicated_anchor():
    with pytest.raises(RuntimeError):
        transform(BASE+OLD)
