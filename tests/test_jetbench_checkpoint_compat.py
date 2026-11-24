"""Read historical NumPy scalar metadata without unrestricted unpickling."""
import io
import pickle
import zipfile

import numpy as np
import pytest
import torch

from models.jetbench_transfer import read_trusted_jetbench_checkpoint


@pytest.mark.parametrize('historical', [False, True])
def test_numpy_scalar_checkpoint(tmp_path, historical):
    stream = io.BytesIO()
    torch.save({'epoch': 3, 'metrics': {'accuracy': np.float64(0.25)},
                'model_state_dict': {'weight': torch.ones(2)}}, stream)
    path = tmp_path / 'checkpoint.pth'
    if historical:
        with zipfile.ZipFile(stream) as source, zipfile.ZipFile(path, 'w') as target:
            for info in source.infolist():
                data = source.read(info.filename)
                if info.filename.endswith('/data.pkl'):
                    data = data.replace(b'cnumpy._core.multiarray\nscalar\n',
                                        b'cnumpy.core.multiarray\nscalar\n')
                    assert b'cnumpy.core.multiarray\nscalar\n' in data
                target.writestr(info, data)
    else:
        path.write_bytes(stream.getvalue())
    checkpoint, state = read_trusted_jetbench_checkpoint(path)
    assert checkpoint['metrics']['accuracy'] == 0.25
    assert torch.equal(state['weight'], torch.ones(2))


class Unexpected:
    pass


def test_unrecognized_globals_still_rejected(tmp_path):
    path = tmp_path / 'unexpected.pth'
    torch.save({'unexpected': Unexpected()}, path)
    with pytest.raises(pickle.UnpicklingError):
        read_trusted_jetbench_checkpoint(path)
