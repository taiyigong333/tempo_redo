import os
import torch
import torch.distributed as dist
from torch.distributed.optim import ZeroRedundancyOptimizer
d=torch.device('cuda',int(os.environ['LOCAL_RANK']))
torch.cuda.set_device(d)
dist.init_process_group('nccl',device_id=d)
params=[torch.nn.Parameter(torch.ones(8,device=d,dtype=t)) for t in (torch.bfloat16,torch.float32)]
reference=[torch.nn.Parameter(x.detach().clone()) for x in params]
kw=dict(lr=5e-5,betas=(.9,.95),eps=1e-8,weight_decay=1e-10)
opt=ZeroRedundancyOptimizer(params,optimizer_class=torch.optim.AdamW,**kw)
ref=torch.optim.AdamW(reference,**kw)
for step in range(3):
    for x,y in zip(params,reference):
        x.grad=torch.full_like(x,.25*(step+1))
        y.grad=x.grad.clone()
    opt.step()
    ref.step()
    for x,y in zip(params,reference):
        torch.testing.assert_close(x,y,rtol=0,atol=0)
state=opt.optim.state_dict()
opt.optim.load_state_dict(state)
assert all(v.device.type=='cuda' for s in opt.optim.state.values() for k,v in s.items() if isinstance(v,torch.Tensor) and k!='step')
if dist.get_rank()==0: print('PASS mixed-dtype sharded AdamW equals torch AdamW over 3 updates and local state reload',flush=True)
dist.destroy_process_group()
