#!/usr/bin/env bash
set -euo pipefail
cd /data-minio/taiyi/tempo
source env.sh
# 等待本次数据迁移完整，避免读取尚在传输的轨迹。
while [[ "$(find data/data11_7/piperx_rotating_red_square -name '*.mp4' | wc -l)" -ne 786 ]] || \
      [[ "$(find data/data11_7/piperx_rotating_red_square -name '*.parquet' | wc -l)" -ne 393 ]]; do
    sleep 10
done
cd code/TEMPO
export CUDA_VISIBLE_DEVICES=3,4,6,7 OMP_NUM_THREADS=4
.venv/bin/torchrun --standalone --nproc_per_node=4 tools/piperx/prepare_gpu_server.py --project ../..
.venv/bin/python tools/piperx/compute_full_norm.py --project ../.. --output ../../configs/full_data_chunk30
