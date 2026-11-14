"""验证每张指定GPU的反向传播、跨卡通信与项目必需补丁。"""
import inspect
import json
import os
from pathlib import Path
import torch
import torch.distributed as dist
from transformers.models.siglip import modeling_siglip

local_rank = int(os.environ['LOCAL_RANK'])
device = torch.device('cuda', local_rank)
torch.cuda.set_device(device)
dist.init_process_group('nccl', device_id=device)
assert '_temporal_attention' in inspect.getsource(modeling_siglip)
x = torch.randn(32,32,device=device,requires_grad=True)
(x@x).sum().backward()
assert torch.isfinite(x.grad).all()
value = torch.tensor(float(dist.get_rank()+1),device=device)
dist.all_reduce(value)
assert value.item() == dist.get_world_size()*(dist.get_world_size()+1)/2
report = {'rank': dist.get_rank(), 'gpu': torch.cuda.get_device_name(device),
          'cuda_backward': True, 'nccl_all_reduce': True, 'transformers_patch': True}
reports = [None]*dist.get_world_size()
dist.all_gather_object(reports, report)
if dist.get_rank() == 0:
    Path('../../docs/environment_gpu_verification.json').write_text(json.dumps(reports,indent=2))
    print(json.dumps(reports),flush=True)
dist.destroy_process_group()
