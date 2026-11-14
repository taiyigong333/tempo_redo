from pathlib import Path
import json, os, signal, time
import psutil
runtime=Path('/data/zhoushuangfan/chuanjun_gong/tempo/runtime/paper_restore_chunk30_2gpu_20261114')
while True:
    state=json.loads((runtime/'latest_status.json').read_text())
    if state['state'] in ('completed','failed'):
        break
    # 仅阻止训练失败之后的自动重试，运行中的训练保持继续执行。
    if state['state']=='recovering' and state.get('child_pid') is None:
        (runtime/'manual_attention.json').write_text(json.dumps(state,ensure_ascii=False,indent=2))
        process=psutil.Process(state['controller_pid'])
        if 'tools/piperx/pipeline.py' in ' '.join(process.cmdline()):
            os.kill(process.pid,signal.SIGTERM)
        break
    time.sleep(15)
