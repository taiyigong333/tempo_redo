#!/usr/bin/env bash
# 由文件所在位置定位项目，让SSH、tmux与交互终端使用同一环境。
TEMPO_PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export TEMPO_PROJECT_DIR
export PATH="$TEMPO_PROJECT_DIR/code/TEMPO/.tools/bin:$PATH"
export UV_PYTHON_INSTALL_DIR="$TEMPO_PROJECT_DIR/code/TEMPO/.tools/python"
export JAX_PLATFORMS=cpu
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export OPENPI_DATA_HOME=/data-minio/taiyi/base_model/openpi_cache
export PALIGEMMA_TOKENIZER_PATH=/data-minio/taiyi/base_model/openpi_cache/big_vision/paligemma_tokenizer.model
export SAM2_REPO="$TEMPO_PROJECT_DIR/code/TEMPO/sam2"
export SAM2_CKPT="$TEMPO_PROJECT_DIR/code/TEMPO/checkpoints/sam2.1_hiera_tiny.pt"
export WANDB_ENTITY=gongcj-beijing
export WANDB_PROJECT=tempo-piperx-gpu-server
export WANDB_PROXY=http://127.0.0.1:9890
export HTTP_PROXY="$WANDB_PROXY"
export HTTPS_PROXY="$WANDB_PROXY"
export NO_PROXY=localhost,127.0.0.1
if [[ -f "$TEMPO_PROJECT_DIR/.secrets/wandb.env" ]]; then
    source "$TEMPO_PROJECT_DIR/.secrets/wandb.env"
fi
source "$TEMPO_PROJECT_DIR/code/TEMPO/.venv/bin/activate"
