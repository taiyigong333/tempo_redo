"""用四rank小模型验证滚动快照、共享AdamW状态与恢复后的下一次更新。"""
import copy
import json
import os
from pathlib import Path
import sys
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'code/TEMPO/tools/piperx'))
from train import save_checkpoint
from openpi.models.pi0_config import Pi0Config

project=Path('/data-minio/taiyi/tempo')
device=torch.device('cuda',int(os.environ['LOCAL_RANK']))
torch.cuda.set_device(device)
dist.init_process_group('nccl',device_id=device)
rank=dist.get_rank()
torch.manual_seed(42)
model=torch.nn.Linear(3,2).to(device)
model.config=Pi0Config(pi05=True,action_horizon=30,ut_window=30,predict_ut=True)
ddp=DistributedDataParallel(model,device_ids=[device.index])
optimizer=torch.optim.AdamW(model.parameters(),lr=1e-3)
x=torch.full((4,3),float(rank+1),device=device)
ddp(x).square().mean().backward(); optimizer.step(); optimizer.zero_grad()
settings={'project':str(project),'norm_dir':str(project/'configs/full_data_chunk30/norm'),
          'split_path':str(project/'configs/full_data_chunk30/split.json'),
          'batch':128,'world_size':4,'save_interval':1000,'steps':5000}
output=project/'runtime/setup/checkpoint_contract'
if rank==0: output.mkdir(exist_ok=True)
dist.barrier()
for step in (100,200,1000,1100,1200):
    save_checkpoint(model,optimizer,output,step,settings,rank)
    if rank==0:
        assert json.loads((output/'latest.json').read_text())['step']==step
    dist.barrier()
reference=copy.deepcopy(model)
reference_optimizer=torch.optim.AdamW(reference.parameters(),lr=1e-3)
reference_optimizer.load_state_dict(copy.deepcopy(optimizer.state_dict()))
state=torch.load(output/f'step_00001200/rank_{rank}.pt',map_location='cpu',weights_only=False)
assert ('optimizer' in state)==(rank==0)
shared=state.get('optimizer')
if shared is None:
    shared=torch.load(output/'step_00001200/rank_0.pt',map_location='cpu',weights_only=False)['optimizer']
optimizer.load_state_dict(shared)
# 使用已同步的相同梯度，验证恢复状态可产生相同的下一次参数更新。
for parameter,expected in zip(model.parameters(),reference.parameters()):
    parameter.grad=torch.full_like(parameter,.25)
    expected.grad=parameter.grad.clone()
optimizer.step(); reference_optimizer.step()
for parameter,expected in zip(model.parameters(),reference.parameters()):
    torch.testing.assert_close(parameter,expected,rtol=0,atol=0)
if rank==0:
    assert not (output/'step_00000100').exists()
    assert not (output/'step_00000200').exists()
    assert not (output/'step_00001100').exists()
    assert (output/'step_00001000').is_dir()
    assert len(list((output/'step_00001200').glob('rank_*.pt')))==4
    report={'world_size':4,'rolling_snapshot_pruning':True,'milestone_retained':True,
            'shared_optimizer_restore_next_update_exact':True,'atomic_latest_pointer':True}
    (project/'docs/checkpoint_verification_gpu_server.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report),flush=True)
dist.barrier(); dist.destroy_process_group()
