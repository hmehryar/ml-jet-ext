"""Keep workstation activation settings out of synthetic test fixtures."""
import pytest


@pytest.fixture(autouse=True)
def isolate_machine_overrides(monkeypatch):
    monkeypatch.delenv('MLJET_DATASET_ROOT', raising=False)
    monkeypatch.delenv('MLJET_OUTPUT_ROOT', raising=False)
