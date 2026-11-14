"""辅助训练完成后发布可独立加载的MVAE、decoder和归一化配置。"""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
import torch

parser=argparse.ArgumentParser()
parser.add_argument('--config',type=Path,required=True)
args=parser.parse_args()
config=json.loads(args.config.read_text())
root=Path(config['experiment_root'])
project=Path(config['project'])
while True:
    complete=True
    for stage in ('mvae','decoder'):
        status=root/stage/'latest_status.json'
        complete=complete and status.is_file() and json.loads(status.read_text()).get('state')=='completed'
    if complete:
        break
    time.sleep(20)
for stage in ('mvae','decoder'):
    saved=torch.load(root/stage/'best.pt',map_location='cpu',weights_only=False)
    cfg=saved['model_cfg']
    assert not saved.get('test_only') and (cfg['window'],cfg['action_dim'],cfg['latent_dim'])==(30,7,64)
    (root/stage/'model_config.json').write_text(json.dumps(cfg,indent=2))
(root/'auxiliary_usage.md').write_text('''# TEMPO辅助模型

在全部393条PiperX轨迹上训练，MVAE100轮，decoder80轮，动作窗口30，u_t维度64，机器人动作7维。best.pt按训练loss选择，last.pt保存末轮与优化器状态。

MVAE的video_enc输入池化SAM2特征[B,30,256,1,1]，其均值mu_v用作策略的u_t监督。动作分支使用检查点中的action_mean和action_std。decoder用UtActionDecoder.from_pretrained加载，输入[B,64]，输出[B,30,7]。动作输出处于policy_norm空间，使用configs/norm_stats.json反归一化，并将前6维相对量加回窗口起点state，夹爪使用绝对开口。

source目录包含匹配的模型源码，configs/split.json记录全部训练轨迹。
''')
uploader=project/'code/TEMPO/.tools/uploader/bin/python'
subprocess.run([str(uploader),str(project/'code/TEMPO/tools/piperx/upload_results.py'),
                '--config',str(args.config.resolve()),'--token-file',config['modelscope']['token_file'],
                '--auxiliary-only'],check=True)
