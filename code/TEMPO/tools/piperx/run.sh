#!/usr/bin/env bash
set -euo pipefail
# 从脚本位置解析项目目录，避免依赖调用者当前工作目录。
repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
project_dir="$(cd "$repo_dir/../.." && pwd)"
cd "$repo_dir"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-3}"
export JAX_PLATFORMS=cpu
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export NO_PROXY="localhost,127.0.0.1"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
if [[ -n "${WANDB_PROXY:-}" ]]; then
  export HTTP_PROXY="$WANDB_PROXY"
  export HTTPS_PROXY="$WANDB_PROXY"
fi
stage="${1:?请指定 mvae、encode、decoder 或 policy}"
shift
case "$stage" in
  mvae|decoder)
    exec .venv/bin/torchrun --standalone --nproc_per_node="$NPROC_PER_NODE" tools/piperx/train_aux.py --project "$project_dir" --stage "$stage" "$@"
    ;;
  encode)
    exec .venv/bin/torchrun --standalone --nproc_per_node="$NPROC_PER_NODE" tools/piperx/encode.py --project "$project_dir" "$@"
    ;;
  policy)
    exec .venv/bin/torchrun --standalone --nproc_per_node="$NPROC_PER_NODE" tools/piperx/train.py --project "$project_dir" "$@"
    ;;
  *)
    echo "未知阶段 $stage" >&2
    exit 2
    ;;
esac
