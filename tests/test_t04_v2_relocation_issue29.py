from pathlib import Path
import hashlib
import pytest
from train_utils.t04_v2_relocation import (
    OLD_LINE, NEW_LINE, equivalent_config_bytes,
    matches_original_digest, relocated_run_path,
)

def test_only_root_move_is_accepted(tmp_path):
    old = b"model: clip\n" + OLD_LINE + b"\nseed: 42\n"
    new = old.replace(OLD_LINE, NEW_LINE)
    path = tmp_path / "case.yml"
    path.write_bytes(new)
    digest = hashlib.sha256(old).hexdigest()
    assert equivalent_config_bytes(new, old)
    assert matches_original_digest(path, digest)
    assert not equivalent_config_bytes(new.replace(b"seed: 42", b"seed: 43"), old)
    path.write_bytes(new.replace(b"seed: 42", b"seed: 43"))
    assert not matches_original_digest(path, digest)

def test_other_path_is_not_permitted(tmp_path):
    old = b"output_root: training_output/OTHER\n"
    path = tmp_path / "other.yml"
    path.write_bytes(b"output_root: /rs/anything-else\n")
    assert not matches_original_digest(path, hashlib.sha256(old).hexdigest())

def test_relocated_run_path():
    result = relocated_run_path(
        "/home/repo/training_output/T04/t04_06_v2_production/clip/S1/E1/seed-42/run1",
        "/home/repo")
    assert str(result) == (
        "/rs/rs_grp_mljet/ml-jet-ext/training_output/T04/"
        "t04_06_v2_production/clip/S1/E1/seed-42/run1")
