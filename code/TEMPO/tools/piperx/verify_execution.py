"""在真实样本上比较执行优化前后的完整前向与全部可训练梯度。"""
import argparse
import dataclasses
import json
import os
from pathlib import Path

import jax
import safetensors.torch
import torch
from torch.utils.data import default_collate
from openpi.models.pi0_config import Pi0Config
from openpi.models.model import Observation
from openpi.models_pytorch.pi0_pytorch import PI0Pytorch
from data import PiperDataset


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--project', type=Path, required=True)
    ap.add_argument('--compact-prefix', action='store_true')
    ap.add_argument('--checkpoint', type=Path, required=True)
    args = ap.parse_args()
    project = args.project.resolve()
    os.environ['PALIGEMMA_TOKENIZER_PATH'] = str(project.parent/'base_model/pi0.5_base/paligemma_tokenizer.model')
    cfg = Pi0Config(pi05=True, dtype='bfloat16', action_horizon=30, obs_history=3,
                    use_sam2_fusion=True, use_action_history=True, action_history_dim=7,
                    predict_ut=True, ut_dim=64, ut_window=30)
    model = PI0Pytorch(cfg)
    safetensors.torch.load_model(model, args.checkpoint/'model.safetensors')
    model = model.cuda().train()
    model.paligemma_with_expert.gemma_expert.lm_head.requires_grad_(False)
    model.gradient_checkpointing_enable()
    dataset = PiperDataset(project, project/'experiments/preflight_20261114/ut')
    batch = default_collate([dataset[50]])
    batch.pop('sample_weight')
    batch = jax.tree.map(lambda x: x.cuda(), batch)
    observation = Observation.from_dict(batch)
    torch.manual_seed(123)
    noise, times = torch.randn(1, 1, 64, device='cuda'), torch.full((1,), .5, device='cuda')
    torch.manual_seed(456)
    with torch.autocast('cuda', dtype=torch.bfloat16):
        baseline = model(observation, batch['actions'], noise=noise, time=times).float().mean()
    baseline.backward()
    reference = {name: p.grad.detach().clone() for name, p in model.named_parameters() if p.grad is not None}
    model.zero_grad(set_to_none=True)
    model.config = dataclasses.replace(cfg, compact_prefix=args.compact_prefix, checkpoint_joint_forward=False)
    torch.manual_seed(456)
    with torch.autocast('cuda', dtype=torch.bfloat16):
        optimized = model(observation, batch['actions'], noise=noise, time=times).float().mean()
    optimized.backward()
    error_squared, reference_squared, maximum = 0., 0., 0.
    for name, parameter in model.named_parameters():
        if name in reference:
            difference = parameter.grad.float()-reference[name].float()
            error_squared += float(difference.square().sum())
            reference_squared += float(reference[name].float().square().sum())
            maximum = max(maximum, float(difference.abs().max()))
    relative_l2 = (error_squared/max(reference_squared, 1e-30))**.5
    result = {'compact_prefix': args.compact_prefix, 'checkpoint_joint_forward': False, 'baseline_loss': float(baseline), 'optimized_loss': float(optimized),
              'loss_abs_difference': float((baseline-optimized).abs()),
              'gradient_relative_l2': relative_l2, 'gradient_max_abs_difference': maximum,
              'gradient_tensors_compared': len(reference), 'passed': relative_l2 < .01 and abs(float(baseline-optimized)) < .002}
    (project/f'experiments/preflight_20261114/execution_equivalence_compact_{args.compact_prefix}.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    if not result['passed']:
        raise ValueError('执行优化数值对照未通过')


if __name__ == '__main__':
    main()
