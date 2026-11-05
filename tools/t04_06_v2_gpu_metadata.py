"""Write append-only physical GPU allocation evidence without changing frozen config."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import torch


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--pool',choices=('a100','h200'),required=True)
    p.add_argument('--run',type=Path,required=True)
    args=p.parse_args()
    run=args.run.resolve()
    if not (run/'run_manifest.json').is_file():
        raise RuntimeError('Not a T04 prepared experiment')
    cfg=json.loads((run/'config.resolved.json').read_text())
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable')
    name=torch.cuda.get_device_name(0)
    expected=('A100','MIG') if args.pool=='a100' else ('H200',)
    if any(x not in name for x in expected) or not torch.cuda.is_bf16_supported():
        raise RuntimeError('Incorrect GPU/BF16 for '+args.pool+': '+name)
    if cfg.get('precision')!='bf16' or cfg.get('training_protocol')!='fixed_budget_v2':
        raise RuntimeError('Unexpected frozen training protocol')
    record={
        'schema':1, 'timestamp_utc':datetime.now(timezone.utc).isoformat(),
        'requested_pool':args.pool,
        'actual_gpu_name':name,
        'frozen_config_gpu_profile':cfg['gpu_profile'],
        'gpu_profile_note':'Frozen config profile describes original A100 benchmark; physical GPU recorded here',
        'pytorch_version':torch.__version__,
        'pytorch_cuda_version':torch.version.cuda,
        'bf16_supported':torch.cuda.is_bf16_supported(),
        'gpu_memory_bytes':torch.cuda.get_device_properties(0).total_memory,
        'slurm_job_id':os.environ.get('SLURM_JOB_ID'),
        'slurm_job_partition':os.environ.get('SLURM_JOB_PARTITION'),
        'slurm_node':os.environ.get('SLURMD_NODENAME'),
        'slurm_cpus_per_task':os.environ.get('SLURM_CPUS_PER_TASK'),
        'execution_profile':os.environ.get('T04_EXECUTION_PROFILE'),
        'dataloader_workers':os.environ.get('T04_DATALOADER_WORKERS'),
    }
    path=run/'hardware_execution.json'
    with path.open('x') as f:
        json.dump(record,f,indent=2,sort_keys=True)
        f.write('\n')
        f.flush()
        os.fsync(f.fileno())
    print('GPU_PROFILE_VERIFIED',args.pool,name,'JOB',record['slurm_job_id'],flush=True)

if __name__=='__main__':
    main()
