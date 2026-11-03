"""Fail-closed, idempotent v2 plan-schema compatibility patch for Stage 3F.2C1.

Only updates the two specific source files; --check never writes.
"""
from __future__ import annotations
import argparse
import ast
import difflib
from pathlib import Path


PREFLIGHT = 'train_utils/t04_preflight.py'
REGISTRY = 'tools/t04_06_registry.py'
OLD_REQUIRED = "required = set(T04Config.__dataclass_fields__) - {'fold', 'split_protocol', 'optimizer'}"
NEW_REQUIRED = "required = set(T04Config.__dataclass_fields__) - {'fold', 'split_protocol', 'optimizer', 'training_protocol', 'training_protocol_hash'}"
VALIDATION = """    if c.training_protocol is not None or c.training_protocol_hash is not None:
        from train_utils.t04_training_termination import validate_protocol
        validate_protocol(c.training_protocol, c.training_protocol_hash, c.epochs)
"""
ORIGINAL_REGISTRY = """    from tools.t04_06_matrix import verify
    # Stage 1's verifier intentionally rejects pre-reserved runs. Bootstrapping
    # is only allowed before any run directory is reserved.
    verify(repo, plan_path)
"""
UPDATED_REGISTRY = """    plan = json.loads(Path(plan_path).read_text())
    if plan.get('kind') == 'T04-06-v2-plan':
        from tools.t04_06_v2_matrix import verify
    elif plan.get('kind') == 'T04-06-plan':
        from tools.t04_06_matrix import verify
    else:
        raise ValueError('Unknown T04-06 plan kind')
    # Bootstrap only before any experiment run is reserved.
    verify(repo, plan_path)
"""


def _once(s, old, new, label):
    a, b = s.count(old), s.count(new)
    if b == 1 and (a == 0 or old in new):
        return s
    if a == 1 and b == 0:
        return s.replace(old, new, 1)
    raise RuntimeError(f'Unexpected or ambiguous {label}: original={a}, updated={b}')


def _preflight(source):
    # Confirm actual dataclass and parse_config exist before rewriting source.
    tree = ast.parse(source)
    klass = next((n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'T04Config'), None)
    parser = next((n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'parse_config'), None)
    if klass is None or parser is None:
        raise RuntimeError('Expected T04Config dataclass and parse_config function')
    fields = [n.target.id for n in klass.body if isinstance(n, ast.AnnAssign) and isinstance(n.target, ast.Name)]
    if not {'fold', 'split_protocol', 'optimizer', 'source_audit_report'} <= set(fields):
        raise RuntimeError('Unexpected T04Config fields')
    field1 = '    training_protocol: str | None = None\n'
    field2 = '    training_protocol_hash: str | None = None\n'
    if field1 not in source and field2 not in source:
        source = _once(source, '    optimizer: dict | None = None\n',
                       '    optimizer: dict | None = None\n' + field1 + field2,
                       'protocol fields')
    elif source.count(field1) != 1 or source.count(field2) != 1:
        raise RuntimeError('Only one protocol field present or duplicates')
    source = _once(source, OLD_REQUIRED, NEW_REQUIRED, 'optional required-field set')
    anchor = "    if c.min_epochs > c.epochs:\n        raise ValueError('min_epochs exceeds epochs')\n"
    source = _once(source, anchor, anchor + VALIDATION, 'protocol validation')
    ast.parse(source)
    return source


def _registry(source):
    tree = ast.parse(source)
    if not any(isinstance(n, ast.FunctionDef) and n.name == '_load_plan' for n in tree.body):
        raise RuntimeError('Missing _load_plan function')
    return _once(source, ORIGINAL_REGISTRY, UPDATED_REGISTRY, 'plan kind dispatch')


def patch(root, apply=False):
    root = Path(root)
    transformations = ((PREFLIGHT, _preflight), (REGISTRY, _registry))
    changes = []
    for filename, fn in transformations:
        path = root / filename
        old = path.read_text()
        new = fn(old)
        changes.append((path, old, new))
    pending = sum(old != new for _, old, new in changes)
    if apply:
        if pending == 0:
            print('ALREADY_PATCHED no modifications')
        else:
            for path, old, new in changes:
                if old != new:
                    path.write_text(new)
                    print('PATCHED', path)
                else:
                    print('ALREADY_PATCHED', path)
    else:
        for path, old, new in changes:
            if old != new:
                print(''.join(difflib.unified_diff(old.splitlines(True), new.splitlines(True),
                       fromfile=str(path), tofile=str(path))), end='')
            else:
                print('ALREADY_PATCHED', path)
        print(f'CHECK_ONLY pending_files={pending}; no modifications')
    return pending


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--check', action='store_true')
    mode.add_argument('--apply', action='store_true')
    a = parser.parse_args()
    patch(Path.cwd(), apply=a.apply)


if __name__ == '__main__':
    main()
