#!/usr/bin/env bash
set -euo pipefail
project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$project_dir/env.sh"
cd "$project_dir/code/TEMPO"
config="${1:-$project_dir/configs/gpu_paper_chunk30_4gpu_20261114.json}"
mapfile -t settings < <(.venv/bin/python - "$config" <<'PY'
import json
from pathlib import Path
import sys
c=json.loads(Path(sys.argv[1]).read_text()); p=Path(c['project'])
for file in (p/'caches/pi05_base_pytorch/model.safetensors',Path(c['tokenizer']),Path(c['split_path']),
             Path(c['norm_dir'])/'norm_stats.json',p/'docs/data_validation_gpu_server.json',
             Path(c['modelscope']['token_file'])):
    if not file.is_file(): raise SystemExit(f'缺少训练输入 {file}')
s=json.loads(Path(c['split_path']).read_text())
assert len(s['train_episodes'])==393 and not s['val_episodes']
for ep in s['train_episodes']:
    assert (p/f'caches/sam2_npy/episode_{ep:06d}.npy').is_file()
print(c['tmux_session']); print(c['experiment_root']); print(c['modelscope']['token_file'])
PY
)
[[ "${#settings[@]}" -eq 3 ]]
session="${settings[0]}"
experiment="${settings[1]}"
token_file="${settings[2]}"
mkdir -p "$experiment/logs"
printf -v train 'source %q; cd %q; exec .venv/bin/python -u tools/piperx/pipeline.py --config %q >> %q 2>&1' \
    "$project_dir/env.sh" "$project_dir/code/TEMPO" "$config" "$experiment/logs/tmux.log"
if ! tmux has-session -t "$session" 2>/dev/null; then
    tmux new-session -d -s "$session" -n train "bash -c $(printf '%q' "$train")"
fi
# 各发布程序持有独立文件锁，重复启动脚本可以安全补齐缺少的窗口。
for kind in modelscope_publish auxiliary_publish; do
    if tmux list-windows -t "$session" -F '#{window_name}' | grep -qx "$kind"; then
        continue
    fi
    if [[ "$kind" == modelscope_publish ]]; then
        printf -v command 'cd %q; exec .tools/uploader/bin/python -u tools/piperx/upload_results.py --config %q --token-file %q >> %q 2>&1' \
            "$project_dir/code/TEMPO" "$config" "$token_file" "$experiment/logs/modelscope_publish.log"
    else
        printf -v command 'source %q; cd %q; exec .venv/bin/python -u tools/piperx/publish_auxiliary.py --config %q >> %q 2>&1' \
            "$project_dir/env.sh" "$project_dir/code/TEMPO" "$config" "$experiment/logs/auxiliary_publish.log"
    fi
    tmux new-window -t "$session" -n "$kind" "bash -c $(printf '%q' "$command")"
done
echo "训练与发布会话已启动 $session"
echo "状态文件 $project_dir/runtime/latest_status.json"
