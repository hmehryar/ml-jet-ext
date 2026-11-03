"""Regression for current three-optional-field preflight and v2 registry routing."""
from pathlib import Path
import ast
import pytest
from tools.t04_06_patch_v2_schema import patch

PREFLIGHT = '''from dataclasses import dataclass
@dataclass(frozen=True)
class T04Config:
    source_audit_report: str
    fold: int | None = None
    split_protocol: str = 'source_disjoint'
    optimizer: dict | None = None

def parse_config(path):
    required = set(T04Config.__dataclass_fields__) - {'fold', 'split_protocol', 'optimizer'}
    if c.min_epochs > c.epochs:
        raise ValueError('min_epochs exceeds epochs')
'''
REGISTRY = '''import json
from pathlib import Path
def _load_plan(repo, plan_path):
    from tools.t04_06_matrix import verify
    # Stage 1's verifier intentionally rejects pre-reserved runs. Bootstrapping
    # is only allowed before any run directory is reserved.
    verify(repo, plan_path)
    plan = json.loads(Path(plan_path).read_text())
    return plan
'''


def tree(tmp_path):
    for name, txt in [('train_utils/t04_preflight.py', PREFLIGHT), ('tools/t04_06_registry.py', REGISTRY)]:
        p = tmp_path / name
        p.parent.mkdir(exist_ok=True)
        p.write_text(txt)
    return tmp_path


def test_apply_and_idempotence(tmp_path):
    root = tree(tmp_path)
    original = [(p, p.read_bytes()) for p in root.rglob('*.py')]
    assert patch(root, apply=False) == 2
    assert all(p.read_bytes() == raw for p, raw in original)
    assert patch(root, apply=True) == 2
    assert patch(root, apply=False) == 0
    assert patch(root, apply=True) == 0
    src = (root/'train_utils/t04_preflight.py').read_text()
    ast.parse(src)
    assert "'optimizer', 'training_protocol', 'training_protocol_hash'" in src
    assert 'validate_protocol(c.training_protocol, c.training_protocol_hash, c.epochs)' in src


def test_fail_closed_if_source_unrecognized(tmp_path):
    root = tree(tmp_path)
    p = root/'train_utils/t04_preflight.py'
    p.write_text(p.read_text().replace("'fold', 'split_protocol', 'optimizer'", "'fold'"))
    with pytest.raises(RuntimeError):
        patch(root, apply=True)
    assert 'T04-06-v2-plan' not in (root/'tools/t04_06_registry.py').read_text()


def test_mixed_already_patched(tmp_path):
    root = tree(tmp_path)
    patch(root, apply=True)
    p = root/'tools/t04_06_registry.py'
    s = p.read_text()
    start = s.index('    plan = json.loads(Path(plan_path).read_text())')
    end = s.index('    plan = json.loads(Path(plan_path).read_text())', start+1)
    from tools.t04_06_patch_v2_schema import ORIGINAL_REGISTRY
    p.write_text(s[:start] + ORIGINAL_REGISTRY + s[end:])
    assert patch(root, apply=False) == 1
    assert patch(root, apply=True) == 1
