"""CPU-only smoke utility unit tests; no Hugging Face network access."""
from argparse import Namespace
from unittest.mock import patch
import pytest
from tools.t04_gpu_smoke import main


def test_reject_invalid_sha():
    with pytest.raises(SystemExit) as e:
        main(['--model','clip','--strategy','S1','--revision','short'])
    assert e.value.code == 2


def test_reject_bad_batch():
    with pytest.raises(SystemExit) as e:
        main(['--model','clip','--strategy','S1','--revision','a'*40,'--batch-size','0'])
    assert e.value.code == 2


def test_creates_exclusive_result(tmp_path):
    report = {'status':'PASS', 'model':'clip'}
    out = tmp_path/'result.json'
    args = ['--model','clip','--strategy','S1','--revision','a'*40,'--output',str(out)]
    with patch('tools.t04_gpu_smoke.run', return_value=report):
        assert main(args) == 0
        assert 'PASS' in out.read_text()
        with pytest.raises(FileExistsError):
            main(args)


def test_oom_returns_nonzero(tmp_path):
    with patch('tools.t04_gpu_smoke.run', return_value={'status':'OOM'}):
        assert main(['--model','siglip','--strategy','S4','--revision','a'*40]) == 2
