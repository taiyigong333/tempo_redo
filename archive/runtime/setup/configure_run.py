from pathlib import Path
import csv
import json
import shutil
import subprocess

root=Path('/data-minio/taiyi/tempo')
repo=root/'code/TEMPO'
backup=root/'runtime/setup/before_paper_restore'
backup.mkdir(exist_ok=True)
for name in ('tools/piperx/train.py','tools/piperx/pipeline.py','tools/piperx/train_aux.py','src/openpi/models/pi0_config.py','src/openpi/models_pytorch/pi0_pytorch.py'):
    dest=backup/name
    dest.parent.mkdir(parents=True,exist_ok=True)
    if not dest.exists():
        shutil.copy2(repo/name,dest)
subprocess.run(['tar','xzf',str(root/'runtime/setup/gpu_training_update.tgz'),'-C',str(repo)],check=True)
c=json.loads((root/'configs/gpu_server_chunk30_4gpu.json').read_text())
name='gpu_paper_chunk30_4gpu_20261114'
c.update(status='authorized',experiment_root=str(root/'experiments'/name),runtime_directory=str(root/'runtime'/name),tmux_session=name,
         minimum_free_memory_mib=68000,require_exclusive_gpus=False)
c['policy'].update(micro_batch=32,accumulation_steps=1,workers=4,save_interval=1000,latest_save_interval=100,status_interval=100,
                   optimizer='torch.optim.AdamW (official native parameter/state dtypes)',checkpoint_joint_forward=True)
c['policy'].pop('compact_prefix',None)
c['aux_execution']={'resident_data':True,'fused_adamw':True,'compile_forward':True}
c['model_contract']={'action_dim':32,'robot_action_dim':7,'ut_dim':64,'flow_positions':1,'action_horizon':30,'ut_window':30,
                     'source_baseline':'dd8342a1234e7166b856ffec2d7917b51cce05e4'}
c['modelscope']={'repo_id':'taiyigong3333/tempo','repo_type':'dataset','path_prefix':'gpu/'+name,
                 'save_upload_interval':1000,'token_file':str(root/'.secrets/modelscope_token')}
config=root/'configs'/f'{name}.json'
config.write_text(json.dumps(c,ensure_ascii=False,indent=2))
experiment=Path(c['experiment_root']); experiment.mkdir(parents=True,exist_ok=True)
(experiment/'logs').mkdir(exist_ok=True)
(experiment/'config.json').write_text(config.read_text())
print(config)
