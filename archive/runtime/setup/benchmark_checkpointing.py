"""隔离比较外层重算与逐层重算，核对loss、全部梯度和随机状态。"""
from pathlib import Path
import json
import statistics
import sys
import time

import jax
import safetensors.torch
import torch
from torch.utils.data import default_collate

project=Path('/data-minio/taiyi/tempo')
sys.path.insert(0,str(project/'code/TEMPO/tools/piperx'))
from data import PiperDataset
from openpi.models.model import Observation
from openpi.models.pi0_config import Pi0Config
from openpi.models_pytorch.pi0_pytorch import PI0Pytorch

cfg=Pi0Config(pi05=True,dtype='bfloat16',action_horizon=30,obs_history=3,obs_history_stride_s=.333,
              temporal_attn_period=4,use_sam2_fusion=True,use_action_history=True,action_history_dim=7,
              predict_ut=True,ut_dim=64,ut_window=30)
model=PI0Pytorch(cfg)
safetensors.torch.load_model(model,project/'experiments/setup_smoke/policy/step_00000002/model.safetensors')
model=model.cuda().train()
model.paligemma_with_expert.gemma_expert.lm_head.requires_grad_(False)
model.gradient_checkpointing_enable()
dataset=PiperDataset(project,project/'experiments/setup_smoke/ut',split_path=project/'configs/full_data_chunk30/split.json',norm_dir=project/'configs/full_data_chunk30/norm')


def inputs(size):
    batch=default_collate([dataset[50+i] for i in range(size)])
    batch.pop('sample_weight')
    return jax.tree.map(lambda x:x.cuda(),batch)


def step(batch,outer):
    model.zero_grad(set_to_none=True)
    model.gradient_checkpointing_enabled=outer
    torch.manual_seed(456)
    size=len(batch['actions'])
    noise=torch.randn(size,1,64,device='cuda')
    times=torch.full((size,),.5,device='cuda')
    torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats(); started=time.perf_counter()
    with torch.autocast('cuda',dtype=torch.bfloat16):
        loss=model(Observation.from_dict(batch),batch['actions'],noise=noise,time=times).float().mean()
    loss.backward(); torch.cuda.synchronize()
    return loss.detach(),time.perf_counter()-started,torch.cuda.max_memory_allocated()/2**30

batch=inputs(1)
base_loss,_,_=step(batch,True)
reference={name:p.grad.detach().cpu() for name,p in model.named_parameters() if p.grad is not None}
base_rng=torch.cuda.get_rng_state().clone()
base_cpu_rng=torch.get_rng_state().clone()
loss,_,_=step(batch,False)
different=[]
for name,p in model.named_parameters():
    if name in reference:
        if p.grad is None or not torch.equal(p.grad.detach().cpu(),reference[name]):
            different.append(name)
    elif p.grad is not None:
        different.append(name)
report={'batch1_baseline_loss':float(base_loss),'batch1_layer_only_loss':float(loss),
        'loss_exact':torch.equal(base_loss,loss),'gradient_tensors_compared':len(reference),
        'different_gradient_tensors':different,'cuda_rng_exact':torch.equal(base_rng,torch.cuda.get_rng_state()),
        'cpu_rng_exact':torch.equal(base_cpu_rng,torch.get_rng_state()),'model_source_modified':False}
del reference,batch
model.zero_grad(set_to_none=True); torch.cuda.empty_cache()
batch=inputs(32); rows=[]
for outer in (True,False,True,False):
    try:
        loss,seconds,peak=step(batch,outer)
        row={'outer_checkpoint':outer,'seconds':seconds,'peak_allocated_gib':peak,'loss':float(loss)}
    except torch.OutOfMemoryError:
        model.zero_grad(set_to_none=True); torch.cuda.empty_cache()
        row={'outer_checkpoint':outer,'oom':True}
    rows.append(row); print(json.dumps(row),flush=True)
report['batch32_benchmarks']=rows
report['passed']=(report['loss_exact'] and not different and report['cuda_rng_exact'] and report['cpu_rng_exact'])
report['resource_condition']='formal MVAE running concurrently on GPU 3'
(project/'docs/checkpoint_execution_benchmark.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
print(json.dumps(report,ensure_ascii=False),flush=True)
