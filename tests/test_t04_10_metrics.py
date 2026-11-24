"""#31: deterministic rank and ordinal accuracy tests, no real backbone download."""
import json
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from train_utils.t04_10_metrics import CLASS_VALUES, ClassificationMetrics, ordered_predictions
from train_utils.t04_10_training import run_single_head_epoch


@pytest.mark.parametrize('task,num', [('alpha_only', 7), ('q0_only', 5)])
def test_hand_calculated_topk_and_tolerance(task, num):
    acc = ClassificationMetrics(task)
    logits = torch.zeros((3, num))
    logits[0, 0] = 10  # truth 0, exact top-1
    logits[1, 1] = 10  # truth 2, near top-1 but top-2 includes truth
    logits[1, 2] = 9
    logits[2, num - 1] = 10  # truth 0, distant miss
    logits[2, 1] = 9  # ensure truth 0 is not in top-2
    acc.update(logits, torch.tensor([0, 2, 0]))
    m = acc.compute()
    assert m['top1_accuracy'] == pytest.approx(1/3)
    assert m['top2_accuracy'] == pytest.approx(2/3)
    assert m['top3_accuracy'] >= m['top2_accuracy']
    assert m['within_1_class_accuracy'] == pytest.approx(2/3)
    assert m['within_2_class_accuracy'] == pytest.approx(2/3)
    step = CLASS_VALUES[task][1] - CLASS_VALUES[task][0]
    expected_mae = (step + (num-1)*step)/3
    assert m['physical_mae'] == pytest.approx(expected_mae)


def test_tie_break_lower_class_index():
    rank = ordered_predictions(torch.tensor([[1., 1., 1., 2., 2.]]))
    assert rank.tolist() == [[3, 4, 0, 1, 2]]


def test_weighted_batches_and_fp32_bf16():
    x = torch.tensor([[4., 0., 0., 0., 0.], [0., 5., 0., 0., 0.],
                      [0., 0., 6., 0., 0.]])
    y = torch.tensor([0, 1, 2])
    for dtype in (torch.float32, torch.bfloat16):
        a = ClassificationMetrics('q0_only')
        a.update(x[:2].to(dtype), y[:2]); a.update(x[2:].to(dtype), y[2:])
        assert a.compute()['top1_accuracy'] == 1
        assert a.compute()['physical_mae'] == 0


@pytest.mark.parametrize('task,classes', [('alpha_only', 7), ('q0_only', 5)])
def test_all_k_monotonic(task, classes):
    a = ClassificationMetrics(task, ks=tuple(range(1, classes+1)))
    a.update(torch.randn(25, classes), torch.arange(25) % classes)
    vals = [a.compute()[f'top{k}_accuracy'] for k in range(1, classes+1)]
    assert vals == sorted(vals)
    assert vals[-1] == 1.


def test_invalid_inputs():
    with pytest.raises(ValueError): ClassificationMetrics('q0_only', ks=(6,))
    with pytest.raises(ValueError): ordered_predictions(torch.tensor([[float('nan'), 2.]]))
    a = ClassificationMetrics('q0_only')
    with pytest.raises(ValueError): a.update(torch.zeros(1, 7), torch.tensor([0]))
    with pytest.raises(ValueError): a.update(torch.zeros(1, 5), torch.tensor([5]))


class _ConstantModel(nn.Module):
    task='q0_only'
    def forward(self, x):
        return {'q0_output': torch.stack((x[:,0], x[:,1], x[:,2], x[:,3], x[:,4]), 1)}


def _val_loader():
    class DS(torch.utils.data.Dataset):
        def __len__(self): return 3
        def __getitem__(self, i):
            return torch.nn.functional.one_hot(torch.tensor(i), 5).float(), {'q0_output': torch.tensor(i)}
    return DataLoader(DS(), batch_size=2, shuffle=False)


def test_epoch_and_export(tmp_path):
    path = tmp_path/'validation.json'
    result = run_single_head_epoch(_val_loader(), _ConstantModel(), task='q0_only',
        device='cpu', prediction_export_path=path, sample_ids=['a','b','c'],
        export_metadata={'run_id':'toy','architecture':'siglip','strategy':'S1',
                         'moments':[1,2,3],'checkpoint_sha256':'0'*64,'split':'val'})
    assert result['accuracy'] == result['top1_accuracy'] == 1.
    doc = json.loads(path.read_text())
    assert doc['metadata']['split'] == 'val'
    assert len(doc['predictions']) == 3
    assert doc['predictions'][2]['sample_id'] == 'c'
    assert len(doc['predictions'][0]['probabilities']) == 5
    with pytest.raises(FileExistsError):
        run_single_head_epoch(_val_loader(), _ConstantModel(), task='q0_only',
            device='cpu', prediction_export_path=path, sample_ids=['a','b','c'],
            export_metadata={'run_id':'toy','architecture':'siglip','strategy':'S1',
                             'moments':[1,2,3],'checkpoint_sha256':'0'*64,'split':'val'})
    with pytest.raises(ValueError):
        run_single_head_epoch(_val_loader(), _ConstantModel(), task='q0_only',
            device='cpu', prediction_export_path=tmp_path/'test.json', sample_ids=['a','b','c'],split='test')


@pytest.mark.skipif(not torch.cuda.is_available(),reason='requires CUDA')
def test_cuda_bf16():
    if not torch.cuda.is_bf16_supported(): pytest.skip('BF16 unsupported')
    a=ClassificationMetrics('q0_only')
    with torch.autocast('cuda',dtype=torch.bfloat16):
        x=torch.randn(8,5,device='cuda')
        a.update(x,torch.arange(8,device='cuda')%5)
    assert a.compute()['top1_accuracy'] <= a.compute()['top2_accuracy']
