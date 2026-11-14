#!/usr/bin/env bash
set -euo pipefail
cd /data-minio/taiyi/tempo
source env.sh
mkdir -p experiments/setup_smoke
code/TEMPO/.venv/bin/python - <<'PY'
from pathlib import Path
import torch
p=Path('experiments/gpu_paper_chunk30_4gpu_20261114/mvae/best.pt')
s=torch.load(p,map_location='cpu',weights_only=False)
s['test_only']=True
s['smoke_source']=str(p.resolve())
torch.save(s,'experiments/setup_smoke/mvae.pt')
PY
cd code/TEMPO
export CUDA_VISIBLE_DEVICES=3,4,6,7 NPROC_PER_NODE=4
bash tools/piperx/run.sh encode --mvae ../../experiments/setup_smoke/mvae.pt --output ../../experiments/setup_smoke/ut
bash tools/piperx/run.sh policy --mode preflight --steps 2 --batch 128 --micro-batch 32 --workers 4 \
    --ut-dir ../../experiments/setup_smoke/ut --output ../../experiments/setup_smoke/policy \
    --split-path ../../configs/full_data_chunk30/split.json --norm-dir ../../configs/full_data_chunk30/norm \
    --eval-interval 0 --save-interval 1000 --latest-save-interval 100 --status-interval 1
bash tools/piperx/run.sh policy --mode preflight --steps 3 --batch 128 --micro-batch 32 --workers 4 \
    --ut-dir ../../experiments/setup_smoke/ut --output ../../experiments/setup_smoke/policy \
    --resume ../../experiments/setup_smoke/policy/step_00000002 \
    --split-path ../../configs/full_data_chunk30/split.json --norm-dir ../../configs/full_data_chunk30/norm \
    --eval-interval 0 --no-save --status-interval 1
