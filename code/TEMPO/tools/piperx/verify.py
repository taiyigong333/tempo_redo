"""针对本次改动执行动作契约、融合梯度与精确 batch 回归。"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from openpi.models_pytorch.pi0_pytorch import SAM2CrossAttnFusion
from data import ExactBatchSampler
from optimizer import adamw_update


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--project', type=Path, required=True)
    args = ap.parse_args()
    project = args.project.resolve()
    split = json.loads((project / 'configs/split.json').read_text())
    assert not set(split['train_episodes']) & set(split['val_episodes'])
    assert len(split['train_episodes']) + len(split['val_episodes']) == 393
    torch.manual_seed(0)
    fusion = SAM2CrossAttnFusion(16, 8, 4, num_heads=4)
    image, motion = torch.randn(2, 4, 16), torch.randn(2, 4, 8)
    output = fusion(image, motion)
    torch.testing.assert_close(output, image, rtol=0, atol=0)
    output.square().sum().backward()
    gate_grad = fusion.gate.grad.abs().item()
    assert gate_grad > 0
    schedules = []
    for rank in range(3):
        sampler = ExactBatchSampler(1000, 128, 2, rank, 3, 0, 3)
        schedules.append(list(sampler))
    assert all(len(schedule) == 66 for schedule in schedules)
    for update in range(3):
        indices = [idx for schedule in schedules for chunk in schedule[update*22:(update+1)*22] for idx in chunk]
        assert len(indices) == len(set(indices)) == 128
    # 以线性模型验证不等长rank与micro-batch加权，梯度应等于单卡128均值。
    x, y = torch.randn(128, 5), torch.randn(128, 2)
    reference = torch.nn.Linear(5, 2)
    ((reference(x)-y)**2).mean().backward()
    gradient = torch.zeros_like(reference.weight.grad)
    for local in np.array_split(np.arange(128), 3):
        model = torch.nn.Linear(5, 2)
        model.load_state_dict(reference.state_dict())
        for chunk in np.array_split(local, 22):
            loss = ((model(x[chunk])-y[chunk])**2).mean()
            (loss * (3*len(chunk)/128)).backward()
        gradient += model.weight.grad/3
    torch.testing.assert_close(gradient, reference.weight.grad, rtol=1e-5, atol=1e-6)
    resumed = list(ExactBatchSampler(1000, 128, 2, 0, 3, 2, 3))
    assert resumed == schedules[0][44:]
    for rank in range(3):
        sampler = list(ExactBatchSampler(1000, 128, 1, rank, 3, 0, 1))
        assert len(sampler) == 43 and max(map(len, sampler)) == 1
        assert sum(x[0] >= 0 for x in sampler) == [43, 43, 42][rank]
    value = torch.nn.Parameter(torch.randn(100))
    master = value.detach().clone()
    first, second = torch.zeros_like(master), torch.zeros_like(master)
    reference_optimizer = torch.optim.AdamW([value], lr=5e-5, betas=(.9, .95), eps=1e-8, weight_decay=1e-10)
    for step in range(1, 6):
        grad = torch.randn_like(value).bfloat16()
        value.grad = grad.float()
        reference_optimizer.step()
        adamw_update(master, grad, first, second, step, 5e-5, (.9, .95), 1e-8, 1e-10, chunk_size=17)
    torch.testing.assert_close(master, value, rtol=1e-6, atol=1e-7)
    result = {'gpu_master_adamw_matches_torch_reference': True, 'micro_batch_one_exact': True, 'split_disjoint': True, 'fusion_identity_at_init': True, 'fusion_gate_grad': gate_grad,
              'global_batch_exact': 128, 'weighted_gradient_matches_reference': True,
              'resume_sampler_matches': True}
    (project / 'experiments/preflight_20261114/regression.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
