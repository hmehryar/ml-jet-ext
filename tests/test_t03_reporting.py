import csv
from pathlib import Path
import tempfile
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from train_utils.t03_reporting import generate_t03_report

class Model(nn.Module):
    def __init__(self):
        super().__init__(); self.fc=nn.Linear(4,12)
    def forward(self,x):
        y=self.fc(x);return {'alpha_output':y[:,:7],'q0_output':y[:,7:]}

class Dataset(torch.utils.data.Dataset):
    def __len__(self):return 14
    def __getitem__(self,i):return torch.ones(4)*i/10,{'alpha_output':torch.tensor(i%7),'q0_output':torch.tensor(i%5)}

def test_t03_reporting():
    with tempfile.TemporaryDirectory() as d:
        p=Path(d); model=Model()
        torch.save({'epoch':2,'model_state_dict':model.state_dict(),'moments':[1,2]},p/'best_model.pth')
        with (p/'moment_epoch_metrics.csv').open('w',newline='') as f:
          fields=['epoch','train_loss','val_loss','train_alpha_loss','val_alpha_loss','train_q0_loss','val_q0_loss','train_alpha_accuracy','val_alpha_accuracy','train_q0_accuracy','val_q0_accuracy','train_joint_accuracy','val_joint_accuracy','val_alpha_macro_f1','val_q0_macro_f1','lr']
          w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
          for i in (1,2): w.writerow({k: i*0.1 for k in fields}|{'epoch':i})
        out=generate_t03_report(model,DataLoader(Dataset(),batch_size=5),torch.device('cpu'),p,expected_samples=14)
        assert out['num_validation_samples']==14
        assert len(list((p/'plots').glob('*.pdf')))==len(list((p/'plots').glob('*.png')))
        assert (p/'best_model_summary.json').is_file()
        print('PASS reporting and figure generation:',len(list((p/'plots').glob('*.png'))),'PNG files')
