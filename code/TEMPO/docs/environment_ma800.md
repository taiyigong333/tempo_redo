# ma800 环境配置与交接

## 目标与位置

- 更新日期为 2026-11-14。
- 服务器为 `ma800`，项目位于 `/data/zhoushuangfan/chuanjun_gong/tempo/code/TEMPO`。
- 使用 uv 管理项目目录下的 `.venv`，Python 3.11.17。
- 按用户要求，解释器本体和标准库完整保存在 `.venv/.python/cpython-3.11.17-linux-x86_64-gnu/`。`.venv/bin/python` 指向该目录，`sys.base_prefix` 和标准库路径均已验证位于项目 `.venv` 内。
- Python 运行时由 uv 下载的 CPython 完整副本构成，使用 `uv venv --allow-existing --python <项目内解释器> .venv` 绑定该解释器。后续直接运行 `uv sync --locked` 维护依赖。
- 硬件为 3 张 NVIDIA A800 80GB，驱动 535.309.01。
- Python 包默认使用清华镜像，配置保存在 `pyproject.toml`。
- 保留原锁文件的已有依赖版本。SAM2 所需 `hydra-core` 和 `iopath` 加入项目依赖，随 `uv sync` 安装。
- 清华源缺少 PyAV 14.4.0 的预编译包，单独配置华为镜像的同版本 Linux x86_64 wheel。
- Git 源码依赖与 SAM2 权重使用各项目官方来源。

## 使用环境

```bash
cd /data/zhoushuangfan/chuanjun_gong/tempo/code/TEMPO
export PATH="$HOME/.local/bin:$PATH"
source .venv/bin/activate
export LEROBOT_VIDEO_BACKEND=pyav
export SAM2_REPO="$PWD/sam2"
export SAM2_CKPT="$PWD/checkpoints/sam2.1_hiera_tiny.pt"
```

数据集位置根据实际数据设置 `HF_LEROBOT_HOME`。从项目根目录执行工具时，SAM2 路径也可使用代码内置默认值。

## 同步与补丁

```bash
uv sync --locked
cp -rf src/openpi/models_pytorch/transformers_replace/* .venv/lib/python3.11/site-packages/transformers/
```

Transformers 重装或升级后重新复制项目补丁，TEMPO 的时序注意力依赖该补丁。

SAM2 源码位于项目内 `sam2/`，使用提交 `2b90b9f5ceec907a1c18123530e92e794ad901a4`。TEMPO 的提取工具会将源码目录加入 Python 搜索路径。

## 执行记录

已完成安装，已安装 245 个包，原锁文件中已有包的版本全部保留。

### 实际验证结果

- `uv sync --locked` 成功，国内 HTTPS 镜像配置与锁文件一致。
- `uv pip check` 通过，245 个已安装包的声明依赖兼容。
- Python 3.11.17 的解释器、标准库、项目第三方包均确认位于项目 `.venv` 内。
- PyTorch 2.7.1+cu126、Torchvision 0.22.1+cu126、Transformers 4.53.2。
- 3 张 A800 分别通过 CUDA 矩阵运算与反向传播验证。
- JAX 0.5.3 检测到 CUDA GPU，JIT 矩阵运算通过。
- Transformers 项目补丁文件逐一比对通过，SigLIP 时序注意力补丁已启用。
- `pi05_yam_tempo_ut_dynamic_handover` 的 `PI0Pytorch` 成功在 CPU 上构建，共 3,636,568,881 个参数。
- SAM2.1 tiny 成功加载权重，在 GPU 上完成两帧时序特征提取，输出 `(2, 256, 8, 8)` 且数值有限。
- 训练与服务入口均通过 `uv run --locked python ... --help` 检查。
- PyAV 14.4.0 与 OpenCV 4.11.0 导入成功。

### 后续实验所需材料

数据集、π0.5 基座权重和正式训练结果需按实验任务准备。构建 u_t 配置时，仓库提示 `caches/ut_decoder/dynamic_handover_v2_all/best.pt` 待生成，需按 README 的 `tools/ut/train_ut_decoder.py` 流程生成后执行对应动作采样。当前已完成环境与最小运行验证，完整训练和真实数据评测尚待实验阶段执行。

### 安装过程与复核

大文件从阿里云镜像分段下载，并与原锁文件的 SHA-256 逐一核对后安装。其余包使用清华镜像，PyAV 使用华为镜像。下载过程临时使用本地 wheel 路径与清华 HTTP 下载地址，最终已恢复为项目持久配置中的 HTTPS 镜像锁文件，并再次执行 `uv sync --locked` 验证。

SAM2 权重 SHA-256 为 `7402e0d864fa82708a20fbd15bc84245c2f26dff0eb43a4b5b93452deb34be69`。



## 2026-11-14 TEMPO训练准备更新

W&B固定为0.24.2以支持新版密钥，uv sync --locked和uv pip check通过，243个包兼容。数据已完成全量验证，基座已转PyTorch，SAM2缓存与三卡短测完成。正式方案、时长与结果见项目根docs/实验方案.md和docs/test_report.json，最新状态见项目根handoff.md，独立代码修改见tempo_code_changes.md。正式训练等待用户确认。


## 2026-11-14 两卡全量正式任务

用户已授权CUDA1、2独占训练，全部393条轨迹参与，完整执行方案在项目根docs/两卡全量训练执行方案.md。归一化新版本在configs/full_data_chunk30_2gpu_20261114/norm。tmux会话tempo_chunk30_2gpu_20261114已启动监督程序，每10分钟等待显卡、每30分钟健康巡检，状态runtime/latest_status.json，W&B服务器专用出口127.0.0.1:17892已核验账户连接。
