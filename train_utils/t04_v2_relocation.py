"""Allowlisted relocation of T04-06 output_root without changing training provenance.

Hashes in the historical registry refer to pre-migration YAML bytes.
Only the exact output_root field replacement is permitted.
"""
from __future__ import annotations
import hashlib
from pathlib import Path

OLD_ROOT = "training_output/T04/t04_06_v2_production"
NEW_ROOT = "/rs/rs_grp_mljet/ml-jet-ext/training_output/T04/t04_06_v2_production"
OLD_LINE = ("output_root: " + OLD_ROOT).encode()
NEW_LINE = ("output_root: " + NEW_ROOT).encode()


def equivalent_config_bytes(actual: bytes, original: bytes) -> bool:
    """No other YAML differences may be ignored."""
    if actual == original:
        return True
    return (original.count(OLD_LINE) == 1
            and actual == original.replace(OLD_LINE, NEW_LINE, 1))


def matches_original_digest(path, expected_sha256: str) -> bool:
    """Compare with historical SHA-256, allowing only a known path relocation."""
    raw = Path(path).read_bytes()
    if hashlib.sha256(raw).hexdigest() == expected_sha256:
        return True
    if raw.count(NEW_LINE) != 1 or raw.count(OLD_LINE) != 0:
        return False
    historical = raw.replace(NEW_LINE, OLD_LINE, 1)
    return hashlib.sha256(historical).hexdigest() == expected_sha256


def relocated_run_path(historical_run, repo):
    """Read-only map from the historical repo-local run tree to Ceph.

    This DOES NOT modify the registry, manifests, or checkpoints.
    """
    repo = Path(repo).resolve()
    historical_run = Path(historical_run)
    old = repo / OLD_ROOT
    try:
        relative = historical_run.relative_to(old)
    except ValueError as exc:
        raise ValueError(f"Run is outside the historical v2 root: {historical_run}") from exc
    return Path(NEW_ROOT) / relative
