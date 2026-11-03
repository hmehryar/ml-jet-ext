"""Stage 3F.2C1: repair tests written against obsolete PATCHES patcher API.

Updates only two test files. Run --check before --apply. Production files untouched.
"""
import argparse
import ast
import difflib
import os
import tempfile
from pathlib import Path

MATRIX = 'tests/test_t04_06_v2_matrix.py'
HOTFIX = 'tests/test_t04_06_v2_schema_hotfix.py'

MATRIX_TESTS = '''def _copy_installed_schema_sources(tmp_path):
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
'''

HOTFIX_TESTS = '''"""Regression coverage for the current fail-closed, idempotent v2 patcher."""
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
'''


def replace_matrix_tests(text):
    tree = ast.parse(text)
    wanted = {'test_schema_patcher_preview_is_read_only', 'test_schema_patcher_idempotence_guard'}
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in wanted]
    if {n.name for n in nodes} != wanted:
        raise RuntimeError('Expected exactly two legacy schema-patcher tests')
    lines = text.splitlines(keepends=True)
    for node in sorted(nodes, key=lambda n: n.lineno, reverse=True):
        lines[node.lineno - 1:node.end_lineno] = []
    output = ''.join(lines).rstrip() + '\n\n\n' + MATRIX_TESTS
    ast.parse(output)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument('--check', action='store_true')
    modes.add_argument('--apply', action='store_true')
    args = parser.parse_args()
    root = Path.cwd()
    matrix = root / MATRIX
    hotfix = root / HOTFIX
    current_matrix = matrix.read_text()
    current_hotfix = hotfix.read_text()
    if 'from tools.t04_06_patch_v2_schema import PATCHES,patch' not in current_matrix:
        raise RuntimeError('Original matrix tests not found; refusing patch')
    if 'from tools.t04_06_patch_v2_schema import PATCHES, patch' not in current_hotfix:
        raise RuntimeError('Original hotfix tests not found; refusing patch')
    updated = {matrix: replace_matrix_tests(current_matrix), hotfix: HOTFIX_TESTS}
    for path, new in updated.items():
        ast.parse(new)
        old = path.read_text()
        print(''.join(difflib.unified_diff(old.splitlines(True), new.splitlines(True), fromfile=str(path), tofile=str(path))), end='')
    if args.apply:
        for path, new in updated.items():
            fd, temp = tempfile.mkstemp(dir=path.parent, prefix='.t04_testfix_')
            try:
                with os.fdopen(fd, 'w') as f:
                    f.write(new)
                os.replace(temp, path)
            finally:
                if os.path.exists(temp):
                    os.unlink(temp)
        print('PATCHED 2 test files; no production files modified')
    else:
        print('CHECK_ONLY 2 test files; no modifications')


if __name__ == '__main__':
    main()
