"""用小模型验证真实检查点入口的三rank原子发布与配置快照。"""
import argparse
import json
import os
from pathlib import Path
import torch
import torch.distributed as dist
from openpi.models.pi0_config import Pi0Config
from train import save_checkpoint


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--project', type=Path, required=True)
    args = ap.parse_args()
    project = args.project.resolve()
    device = torch.device('cuda', int(os.environ['LOCAL_RANK']))
    torch.cuda.set_device(device)
    dist.init_process_group('nccl', device_id=device)
    model = torch.nn.Linear(2, 2).to(device)
    model.config = Pi0Config(pi05=True, action_horizon=30, ut_window=30)
    optimizer = torch.optim.AdamW(model.parameters())
    optimizer.optim = optimizer
    output = project/'experiments/preflight_20261114/checkpoint_contract'
    if dist.get_rank() == 0:
        output.mkdir(parents=True, exist_ok=True)
    dist.barrier()
    seconds, final = save_checkpoint(model, optimizer, output, 1, {'project': str(project)}, dist.get_rank())
    if dist.get_rank() == 0:
        assert len(list(final.glob('rank_*.pt'))) == 3
        assert json.loads((final/'model_config.json').read_text())['action_horizon'] == 30
        assert (final/'norm_stats.json').is_file() and (final/'split.json').is_file()
        result = {'atomic_checkpoint_complete': True, 'rank_shards': 3, 'model_config_window': 30,
                  'norm_and_split_snapshots': True, 'seconds': seconds}
        (output/'verification.json').write_text(json.dumps(result, indent=2))
        print(json.dumps(result, indent=2))
    dist.barrier()
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
