#!/usr/bin/env bash
set -euo pipefail
project=/data-minio/taiyi/tempo
source "$project/runtime/clockwise_setup_20261116/env.sh"
cd "$project/code/TEMPO"
config="$project/configs/gpu_clockwise_b234_chunk30_4gpu_20261116.json"
# 启动前完整核对本实验路径，防止数据准备尚未完成。
.venv/bin/python - "$config" <<'PY'
import json,sys
from pathlib import Path
import numpy as np
c=json.loads(Path(sys.argv[1]).read_text())
s=json.loads(Path(c['split_path']).read_text())
assert len(s['train_episodes'])==c['dataset']['train_episodes'] and not s['val_episodes']
report=json.loads(Path('/data-minio/taiyi/tempo/docs/clockwise_data_validation_20261116.json').read_text())
assert (report['episodes'],report['frames'])==(289,30285)
for path in (Path(c['pretrained_dir'])/'model.safetensors',Path(c['tokenizer']),Path(c['norm_dir'])/'norm_stats.json'):
    assert path.is_file(), path
frames=0
for ep in s['train_episodes']:
    root=Path(c['rgb_cache'])/f'episode_{ep:06d}'
    state=np.load(root/'state.npy',mmap_mode='r')
    frames+=len(state)
    for key in ('head','left_wrist'):
        assert np.load(root/f'{key}.npy',mmap_mode='r').shape==(len(state),224,224,3)
    assert np.load(root/'action.npy',mmap_mode='r').shape==state.shape
    assert np.load(Path(c['sam2_cache'])/f'episode_{ep:06d}.npy',mmap_mode='r').shape==(len(state),256,8,8)
assert frames==c['dataset']['frames']
print('训练输入检查通过',frames)
PY
session=gpu_clockwise_b234_chunk30_4gpu_20261116
logs="$project/experiments/$session/logs"
# 每个程序保留原监督及文件锁机制，重入时补齐缺少的窗口。
for kind in train modelscope_publish auxiliary_publish; do
    if tmux list-windows -t "$session" -F '#{window_name}' 2>/dev/null | grep -qx "$kind"; then
        continue
    fi
    case "$kind" in
        train) program='.venv/bin/python -u tools/piperx/pipeline.py' ;;
        modelscope_publish) program=".tools/uploader/bin/python -u tools/piperx/upload_results.py --token-file $project/.secrets/modelscope_token" ;;
        auxiliary_publish) program='.venv/bin/python -u tools/piperx/publish_auxiliary.py' ;;
    esac
    command="source $project/runtime/clockwise_setup_20261116/env.sh; cd $project/code/TEMPO; exec $program --config $config >> $logs/$kind.log 2>&1"
    if tmux has-session -t "$session" 2>/dev/null; then
        tmux new-window -t "$session" -n "$kind" "bash -c $(printf '%q' "$command")"
    else
        tmux new-session -d -s "$session" -n "$kind" "bash -c $(printf '%q' "$command")"
    fi
done
printf '%s\n' "训练会话 $session"
