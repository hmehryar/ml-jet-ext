import json
from pathlib import Path
import pytest
from train_utils import t04_sessions as s


@pytest.fixture
def run(tmp_path, monkeypatch):
    (tmp_path/'run_manifest.json').write_text('{"ok":true}')
    (tmp_path/'config.resolved.json').write_text('{"epochs":10}')
    monkeypatch.setattr(s, '_validate_run', lambda p: Path(p).resolve())
    return tmp_path


def test_fresh_and_resume_preserves_history(run):
    with s.experiment_lock(run):
        a = s.start_session(run, mode='fresh', target_epochs=3)
        cp = a/'checkpoints'/'epoch_000003.pth'
        cp.write_bytes(b'good checkpoint')
        p = s.publish_checkpoint(run, a, cp, epoch=3)
        s.finish_session(run, a, status='completed')
        first_hash = s._digest(cp)
        b = s.start_session(run, mode='resume', target_epochs=6)
        assert b.name == 'session-002'
        assert json.loads((b/'session_manifest.json').read_text())['parent_checkpoint']['sha256'] == p['sha256']
        cp2 = b/'checkpoints'/'epoch_000006.pth'
        cp2.write_bytes(b'new checkpoint')
        s.publish_checkpoint(run, b, cp2, epoch=6)
        s.finish_session(run, b, status='completed')
        assert s.latest_checkpoint(run)['epoch'] == 6
        assert cp.read_bytes() == b'good checkpoint'
        assert s._digest(cp) == first_hash


def test_fresh_refuses_existing(run):
    with s.experiment_lock(run):
        s.start_session(run, mode='fresh', target_epochs=4)
        with pytest.raises(s.SessionError):
            s.start_session(run, mode='fresh', target_epochs=4)


def test_resume_requires_published_checkpoint(run):
    with s.experiment_lock(run):
        s.start_session(run, mode='fresh', target_epochs=4)
        with pytest.raises(s.SessionError):
            s.start_session(run, mode='resume', target_epochs=5)


def test_corruption_detected(run):
    with s.experiment_lock(run):
        a = s.start_session(run, mode='fresh', target_epochs=4)
        cp = a/'checkpoints'/'epoch_000001.pth'
        cp.write_bytes(b'ok')
        s.publish_checkpoint(run, a, cp, epoch=1)
        cp.write_bytes(b'tampered')
        with pytest.raises(s.SessionError, match='SHA-256'):
            s.latest_checkpoint(run)
        with pytest.raises(s.SessionError):
            s.start_session(run, mode='resume', target_epochs=5)


def test_nonblocking_exclusive_lock(run):
    with s.experiment_lock(run):
        with pytest.raises(s.SessionError, match='Another execution'):
            with s.experiment_lock(run):
                pass
    with s.experiment_lock(run):
        pass


def test_checkpoint_epoch_must_increase(run):
    with s.experiment_lock(run):
        a = s.start_session(run, mode='fresh', target_epochs=3)
        one = a/'checkpoints'/'one.pth'; one.write_bytes(b'1')
        s.publish_checkpoint(run, a, one, epoch=1)
        with pytest.raises(s.SessionError, match='monotonically'):
            s.publish_checkpoint(run, a, one, epoch=1)


def test_parent_target_and_path_guards(run):
    with s.experiment_lock(run):
        a = s.start_session(run, mode='fresh', target_epochs=3)
        cp = a/'checkpoints'/'one.pth'; cp.write_bytes(b'1')
        s.publish_checkpoint(run, a, cp, epoch=1)
        with pytest.raises(s.SessionError):
            s.start_session(run, mode='resume', target_epochs=1)
        with pytest.raises(s.SessionError):
            s.start_session(run, mode='resume', target_epochs=3, parent='wrong')
        with pytest.raises(s.SessionError):
            s.publish_checkpoint(run, a, run/'config.resolved.json', epoch=2)


def test_finalize_once(run):
    with s.experiment_lock(run):
        a = s.start_session(run, mode='fresh', target_epochs=2)
        s.finish_session(run, a, status='interrupted')
        with pytest.raises(s.SessionError):
            s.finish_session(run, a, status='completed')
