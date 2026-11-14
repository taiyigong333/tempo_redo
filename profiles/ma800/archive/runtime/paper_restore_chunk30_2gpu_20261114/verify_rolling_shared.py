from pathlib import Path
import sys, os, json, tempfile, dataclasses
import torch
import torch.distributed as dist
repo=Path('/data/zhoushuangfan/chuanjun_gong/tempo/code/TEMPO')
sys.path.insert(0,str(repo/'tools/piperx'))
from train import save_checkpoint
from pipeline import cards_available, Pipeline
from upload_results import Publisher
from unittest.mock import patch
import pipeline

device=torch.device('cuda',int(os.environ['LOCAL_RANK']))
torch.cuda.set_device(device)
dist.init_process_group('nccl',device_id=device)
rank=dist.get_rank()
paths=[tempfile.mkdtemp(prefix='rolling_check_') if rank==0 else None]
dist.broadcast_object_list(paths,src=0)
p=Path(paths[0]); output=p/'policy'
if rank==0:
    output.mkdir(); (p/'norm').mkdir()
    (p/'norm/norm_stats.json').write_text('{}'); (p/'split.json').write_text('{}')
dist.barrier()
@dataclasses.dataclass
class Config:
    action_dim:int=32
class Tiny(torch.nn.Linear):
    def __init__(self):
        super().__init__(2,2); self.config=Config()
model=Tiny().to(device)
opt=torch.optim.AdamW(model.parameters())
model(torch.ones(1,2,device=device)).sum().backward(); opt.step()
settings=dict(project=str(p),norm_dir=str(p/'norm'),split_path=str(p/'split.json'),save_interval=1000,steps=5000)
for step in (100,200,1000,1100,1200):
    save_checkpoint(model,opt,output,step,settings,rank)
    pointer=json.loads((output/'latest.json').read_text())
    assert pointer['step']==step
    state=torch.load(Path(pointer['path'])/f'rank_{rank}.pt',map_location='cpu',weights_only=False)
    opt.load_state_dict(state['optimizer']); assert state['step']==step
assert not (output/'step_00000100').exists() and not (output/'step_00000200').exists()
assert (output/'step_00001000/model.safetensors').is_file()
assert not (output/'step_00001100').exists() and (output/'step_00001200/rank_1.pt').is_file()
if rank==0:
    snapshot=[dict(memory_total_mib=81920,memory_used_mib=11306,utilization_percent=100,processes=[dict(pid=999)])]
    assert cards_available(snapshot,1024,68000)
    assert not cards_available(snapshot,1024,72000)
    check=Pipeline.__new__(Pipeline)
    check.config=dict(health_interval_seconds=1,stall_timeout_seconds=1800,gpus=[1,2])
    check.last_health=0; check.root=p; check.stage='policy'; check.state='running'; check.runtime=p
    (output/'metrics.jsonl').write_text('{"loss":0.1}\n')
    class Child:
        pid=os.getpid()
        def poll(self): return None
    check.child=Child()
    check.stop_child=lambda: (_ for _ in ()).throw(AssertionError('healthy shared training was stopped'))
    with patch.object(pipeline,'gpu_snapshot',return_value=snapshot): check.health_check(force=True)
    result=json.loads((p/'health_checks.jsonl').read_text().splitlines()[-1])
    assert result['healthy'] and result['external_gpu_pids']==[999]
    (p/'aux').mkdir(); (p/'aux/mvae').mkdir(); (p/'aux/decoder').mkdir()
    for stage in ('mvae','decoder'):
        for name in ('best.pt','last.pt','model_config.json','latest_status.json'):
            (p/'aux'/stage/name).write_text('{}')
    (p/'aux/auxiliary_usage.md').write_text('usage')
    config=p/'config.json'
    config.write_text(json.dumps(dict(project=str(repo.parents[1]),experiment_root=str(p/'aux'),runtime_directory=str(p/'runtime'),norm_dir=str(p/'norm'),split_path=str(p/'split.json'),policy=dict(steps=5000,save_interval=1000))))
    publisher=Publisher(config,'test/repo',p/'token',auxiliary_only=True)
    assert publisher.complete()
    manifest=publisher.stage()
    assert len(manifest['files'])==14
    assert publisher.prefix.endswith('/auxiliary')
    print('PASS two-rank atomic rolling checkpoints, AdamW restore, rolling retention, thousand-step retention, shared GPU admission, no stop for external GPU process, auxiliary bundle',flush=True)
dist.barrier(); dist.destroy_process_group()
