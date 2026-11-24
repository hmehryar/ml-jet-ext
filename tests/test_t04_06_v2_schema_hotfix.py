"""Regression coverage for the current fail-closed, idempotent v2 patcher."""
from pathlib import Path
import pytest
from tools.t04_06_patch_v2_schema import patch, ORIGINAL_REGISTRY

PREFLIGHT_SOURCE = """from dataclasses import dataclass
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
"""

REGISTRY_SOURCE = """import json
from pathlib import Path
def _load_plan(repo, plan_path):
""" + ORIGINAL_REGISTRY + """    plan = json.loads(Path(plan_path).read_text())
    return plan
"""


def fixture(root):
    for name, source in (
        ('train_utils/t04_preflight.py', PREFLIGHT_SOURCE),
        ('tools/t04_06_registry.py', REGISTRY_SOURCE),
    ):
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(source)
    return root


def test_all_new(tmp_path):
    root = fixture(tmp_path)
    assert patch(root, apply=False) == 2
    assert patch(root, apply=True) == 2
    assert patch(root, apply=False) == 0
    assert patch(root, apply=True) == 0  # Already applied is idempotent.


def test_partly_already_applied(tmp_path):
    root = fixture(tmp_path)
    assert patch(root, apply=True) == 2
    # Revert only the registry to simulate a mixed deployment.
    registry = root / 'tools/t04_06_registry.py'
    registry.write_text(REGISTRY_SOURCE)
    assert patch(root, apply=False) == 1
    assert patch(root, apply=True) == 1
    assert patch(root, apply=False) == 0


def test_unknown_source_fails_without_writes(tmp_path):
    root = fixture(tmp_path)
    registry = root / 'tools/t04_06_registry.py'
    registry.write_text('unrecognized code')
    preflight = root / 'train_utils/t04_preflight.py'
    before = preflight.read_bytes()
    with pytest.raises((RuntimeError, SyntaxError)):
        patch(root, apply=True)
    assert preflight.read_bytes() == before
