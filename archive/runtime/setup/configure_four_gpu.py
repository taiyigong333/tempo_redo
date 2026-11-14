from pathlib import Path
import csv
import json
import subprocess
root = Path('/data-minio/taiyi/tempo')
old = root/'configs/gpu_server_chunk30.json'
c = json.loads(old.read_text())
c.update(experiment_root=str(root/'experiments/formal_chunk30_4gpu'),
         runtime_directory=str(root/'runtime/formal_chunk30_4gpu'), tmux_session='tempo_chunk30_4gpu',
         gpus=[3,4,6,7], gpu_selection='fixed_cuda_indices', require_exclusive_gpus=False,
         min_free_memory_mib=50000)
rows = subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid','--format=csv,noheader'],text=True)
ids = {int(row[0]): row[1].strip() for row in csv.reader(rows.splitlines())}
c['gpu_uuids'] = [ids[i] for i in c['gpus']]
c['policy']['accumulation_steps'] = 2
new = root/'configs/gpu_server_chunk30_4gpu.json'
new.write_text(json.dumps(c,ensure_ascii=False,indent=2))
old.unlink()
p=root/'code/TEMPO/tools/piperx/pipeline.py'
s=p.read_text()
s=s.replace("contended = record['child_alive'] and bool(record['external_gpu_pids'])", "contended = (record['child_alive'] and bool(record['external_gpu_pids'])\n                     and self.config.get('require_exclusive_gpus', True))")
s=s.replace("if not cards_available(snapshot, self.config['max_idle_memory_mib']):", "# 共享模式仅检查启动显存余量，独占模式兼容既有流程。\n            available = (cards_available(snapshot, self.config['max_idle_memory_mib'])\n                         if self.config.get('require_exclusive_gpus', True) else\n                         all(gpu['memory_total_mib']-gpu['memory_used_mib'] >= self.config['min_free_memory_mib']\n                             for gpu in snapshot))\n            if not available:")
s=s.replace("'rank_0.pt', 'rank_1.pt']", "] + [f'rank_{rank}.pt' for rank in range(len(self.config['gpus']))]")
s=s.replace('等待两卡独占', '等待指定GPU满足启动显存条件').replace('tmux内等待两卡独占', 'tmux内等待指定GPU满足启动显存条件')
p.write_text(s)
p=root/'start_train.sh'
s=p.read_text().replace('gpu_server_chunk30.json','gpu_server_chunk30_4gpu.json').replace('formal_chunk30_2gpu','formal_chunk30_4gpu')
p.write_text(s)
p=root/'handoff.md'
s=p.read_text().replace('gpu_server_chunk30.json','gpu_server_chunk30_4gpu.json')
s=s.replace('默认启动时自动寻找两张无计算进程的空闲GPU，micro16，累积4，有效batch128。缺少两张空闲GPU时每10分钟等待', '用户明确指定CUDA 3、4、6、7，micro16，累积2，有效batch128。允许与当前已有任务共享，启动时每张卡需至少50000MiB空闲，未满足时每10分钟等待')
p.write_text(s)
print('four GPU configuration applied')
