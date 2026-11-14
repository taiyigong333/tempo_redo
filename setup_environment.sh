#!/usr/bin/env bash
set -euo pipefail
project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$project_dir/code/TEMPO"
export UV_PYTHON_INSTALL_DIR="$PWD/.tools/python"
export UV_LINK_MODE=copy UV_HTTP_TIMEOUT=120 SAM2_BUILD_CUDA=0
.tools/bin/uv python install 3.11.15
.tools/bin/uv sync --locked --python 3.11.15
# TEMPO的时序注意力依赖该补丁，每次依赖重装后重新应用。
cp -rf src/openpi/models_pytorch/transformers_replace/* .venv/lib/python3.11/site-packages/transformers/
.tools/bin/uv pip check
JAX_PLATFORMS=cpu .venv/bin/python -c 'import inspect; from transformers.models.siglip import modeling_siglip; assert "_temporal_attention" in inspect.getsource(modeling_siglip); import sam2; print("TEMPO与SAM2导入通过")'
