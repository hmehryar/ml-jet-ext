from pathlib import Path
import numpy as np
import pandas as pd
import pytest
import torch
from data.moment_loader import MomentJetDataset, get_moment_dataloaders


def create_fixture(root):
    config = 'c0005_alpha0p20_q_1p0'
    (root/config).mkdir(parents=True)
    (root/'splits'/'five_fold').mkdir(parents=True)
    rows = []
    for i in range(8):
        rel = f'{config}/agg500_event_{i:07d}.npy'
        a = np.stack([np.full((32,32), j/10, np.float32) for j in range(1,6)])
        np.save(root/rel, a)
        rows.append(dict(agg_file_path=rel, alpha=.2, q0=1.0, config=config, source_file_paths='unused'))
    for name, indices in [('train',[0,1,2,3]), ('val',[4,5]), ('test',[6,7]),
                          ('five_fold/fold_0_train',[0,1,2]), ('five_fold/fold_0_val',[3,4,5])]:
        pd.DataFrame([rows[i] for i in indices]).to_csv(root/'splits'/f'{name}.csv', index=False)
    return rows


def test_moment_selection_and_labels(tmp_path):
    create_fixture(tmp_path)
    dataset = MomentJetDataset(tmp_path/'splits/train.csv', tmp_path, moments=(1,3,5), validate_files=True, check_normalization=True)
    image, labels = dataset[0]
    assert image.shape == (3,32,32) and image.dtype == torch.float32
    assert torch.allclose(image[:,0,0], torch.tensor([.1,.3,.5]))
    assert labels['alpha_output'].item() == 0 and labels['q0_output'].item() == 0
    assert labels['alpha_output'].ndim == 0


def test_loaders_and_folds(tmp_path):
    create_fixture(tmp_path)
    a,b,c = get_moment_dataloaders(tmp_path, moments=[1,2], batch_size=2, num_workers=0, pin_memory=False)
    assert [len(ds.dataset) for ds in [a,b,c]] == [4,2,2]
    assert next(iter(a))[0].shape == (2,2,32,32)
    a,b,c = get_moment_dataloaders(tmp_path, moments=[1], batch_size=2, num_workers=0, fold=0)
    assert [len(ds.dataset) for ds in [a,b,c]] == [3,3,2]


def test_reject_duplicate_paths(tmp_path):
    rows = create_fixture(tmp_path)
    pd.DataFrame([rows[0],rows[0]]).to_csv(tmp_path/'splits/train.csv', index=False)
    with pytest.raises(ValueError, match='Duplicate'):
        MomentJetDataset(tmp_path/'splits/train.csv',tmp_path)


def test_reject_invalid_moments(tmp_path):
    create_fixture(tmp_path)
    with pytest.raises(ValueError, match='moments'):
        MomentJetDataset(tmp_path/'splits/train.csv',tmp_path, moments=[1,1])
