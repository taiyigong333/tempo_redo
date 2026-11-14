from pathlib import Path
import hashlib
import json
p=Path('/data-minio/taiyi/tempo')
cpath=p/'configs/gpu_paper_chunk30_4gpu_20261114.json'
c=json.loads(cpath.read_text()); root=Path(c['experiment_root'])
text='''# GPU-Server TEMPO 训练交接

更新于2026-11-14。用户要求在本人全部数据上复现论文，CUDA卡号3、4、6、7，chunk30，使用独立W&B项目，维护最新权重与每100步状态，每1000步上传ModelScope。正式流水线已在服务器tmux启动，当前阶段以runtime/latest_status.json为准。

## 项目与入口

项目根 `/data-minio/taiyi/tempo`，源码 `code/TEMPO`，独立uv环境 `.venv`，解释器 `.tools/python`。Python3.11.15、PyTorch2.7.1+cu126、Transformers4.53.2、W&B0.24.2，244个包兼容检查通过。

配置 `configs/gpu_paper_chunk30_4gpu_20261114.json`，执行 `bash start_train.sh` 可启动或补齐训练与上传窗口。tmux会话 `gpu_paper_chunk30_4gpu_20261114`，包含train、modelscope_publish、auxiliary_publish三个窗口。已有窗口复用，程序各自持有文件锁。

正式实验 `experiments/gpu_paper_chunk30_4gpu_20261114`。状态 `runtime/latest_status.json`，控制日志位于实验logs/tmux.log，分阶段日志位于logs/mvae.log、decoder.log和policy.log。服务器任务通过本机代理9890记录W&B，网络出口与训练进程均运行在服务器。

## 论文复现约定

模型配置 `src/openpi/models/pi0_config.py` 与前向 `src/openpi/models_pytorch/pi0_pytorch.py` 已与官方提交dd8342a1234e7166b856ffec2d7917b51cce05e4核对。SHA256分别为bc19200f7d62aacaa3ee6bfbd32b4e45689d81ad2d8a5f5d4c1087ec5af06961和d3fd6dabc1bdef852517f576cf4e56ce040c0e2cc2487b17346a0673d10e7d88。

模型合同为action_dim32、u_t64、flow_positions1、action_horizon30、ut_window30，机器人有效动作7维。使用官方原生torch.optim.AdamW及模型原始混合参数精度，BF16 autocast。原版SAM2融合包含零gate与零输出投影，训练中的gate梯度可能为0，按官方初始化执行。论文效果仍需训练后评测。

基座使用用户提供的 `/data-minio/taiyi/base_model/openpi_cache/openpi-assets/checkpoints/pi05_base`，已转换到 `caches/pi05_base_pytorch`。分词器使用同一base_model/openpi_cache/big_vision目录下文件。

## 数据

从ModelScope数据集taiyigong3333/piperx_rotation_red_block下载piperx_rotating_red_square_tempo.zip，压缩包CRC、1187个文件的SHA256内容清单均验证。清单指纹80143ff22ceb1c0a32c9c044b846bb4e283f3c49967e5b1d8219a90daa215908。

全量393轨迹、44398帧、786视频、两路224×224相机、30Hz、7维动作，下一帧state与当前action最大误差0。原始数据 `data/data11_7/piperx_rotating_red_square`，RGB缓存 `caches/piperx_rgb_v1`，SAM2缓存 `caches/sam2_npy`。四卡全量缓存实际227.65秒。

全部393轨迹进入训练，验证轨迹0。全量30帧归一化重新计算，位于configs/full_data_chunk30，前6个关节按窗口起点state相对化，夹爪使用绝对开口。数据和归一化验证记录位于docs/dataset_modelscope.json与docs/data_validation_gpu_server.json。

## 训练参数

- MVAE100轮，decoder80轮，global batch64，latent64，beta0，window30，AdamW lr1e-4、wd1e-4、余弦退火、clip1。
- 辅助训练启用GPU驻留数据、fused AdamW及forward编译，保留模型模块键名和接口。
- 策略5000步，全局batch128，四卡每卡micro32，梯度累积1，workers4。显存不足时监督程序缩小micro，采样器维持global128。
- 策略AdamW betas(.9,.95)、eps1e-8、wd1e-10，clip1，seed42，warmup500，lr5e-5至5e-6余弦衰减。
- 每100步发布最新状态并原子保存最新可恢复检查点。新快照发布后清理前一滚动快照，1000、2000、3000、4000、5000里程碑保留。
- 标准DDP各rank的AdamW状态相同，新检查点仅在rank_0.pt保存一份优化器状态，所有rank独立保存随机状态。恢复入口兼容旧逐rank优化器格式。训练计算期间权重、梯度和优化器状态均位于GPU。
- 启动检查每卡68000MiB空闲。运行后记录外部进程，按进度和数值检查健康状态。每30分钟巡检，阶段失败可从完整检查点恢复。

## W&B与ModelScope

本项目专用W&B为gongcj-beijing/tempo-piperx-gpu-server，与ma800分开。实际运行链接在阶段latest_status.json中。凭据位于项目.secrets目录，仅本机私有权限保存。

ModelScope数据集taiyigong3333/tempo，统一前缀 `gpu/gpu_paper_chunk30_4gpu_20261114/`。policy/step_00001000等目录逐千步发布完整权重、共享优化器状态、各rank随机状态与配置。辅助模型在auxiliary/目录发布，包含MVAE、decoder的best/last及独立加载源码、归一化和使用说明。最终汇总日志和实验材料。

上传工具复用参考聊天的方法，使用ModelScope1.40.3、可续传缓存、独立检查点任务和提交后大小/SHA256回读核验。三个初始配置文件已实际提交到gpu前缀并回读通过，详见docs/modelscope_upload_verification.json。用户已明确授权上传配套模型配置、归一化、划分、日志和源码。

## 验证与接续

四卡CUDA反向及NCCL、JAX GPU、SAM2时序特征提取均通过。四rank小模型检查点验证覆盖原子latest发布、滚动清理、千步保留及共享AdamW恢复后下一次更新逐元素一致，记录在docs/checkpoint_verification_gpu_server.json。

正式辅助训练已运行并记录有限loss。独立全模型短测位于experiments/setup_smoke，用于验证micro32、保存和恢复，测试标签带test_only标记。实际进度和短测结论需读取对应日志与退出码，禁止将短测权重作为正式u_t来源。

接续时先读取runtime/latest_status.json与分阶段最新metrics.jsonl。使用同一start_train.sh可补齐缺失窗口。先检查现有tmux与PID，再处理故障。
'''
(p/'handoff.md').write_text(text)
changes='''# GPU-Server TEMPO 独立改动记录

2026-11-14，论文复现模型基线为dd8342a1234e7166b856ffec2d7917b51cce05e4。

## 模型与训练入口

- 从参考聊天恢复官方pi0_config.py与pi0_pytorch.py，SHA256与官方源码一致。训练使用action_dim32、u_t64、flow_positions1、chunk30及有效7维动作。
- tools/piperx/train.py 采用原生torch.optim.AdamW，支持目标环境分词器和独立W&B项目。支持100步原子滚动快照和1000步里程碑。
- 检查点格式v2将DDP重复的优化器状态集中保存到rank_0.pt，各rank保留随机状态，降低周期性写盘量。兼容既有逐rank优化器快照恢复。四卡小模型恢复后的下一步更新逐元素一致。
- tools/piperx/train_aux.py 复用参考聊天GPU驻留数据、fused AdamW与forward编译，支持独立W&B项目。
- tools/piperx/pipeline.py 支持四卡完成性检查，使用目标服务器既有代理和配置路径，读取batch、保存周期及辅助轮次，按用户要求记录外部GPU进程和训练健康状态。
- tools/piperx/upload_results.py 读取gpu目录前缀并检查全部四rank文件，复用参考实现的续传、并发和远端校验。
- tools/piperx/publish_auxiliary.py 在辅助阶段结束后生成加载配置和使用说明并发布模型。

## 环境与数据

- pyproject.toml和uv.lock将sam2本地依赖纳入uv，配置SAM2_BUILD_CUDA=0，训练使用CUDA版PyTorch。
- tools/piperx/prepare_gpu_server.py 复用数据验证和官方SAM2特征编码，提供可恢复的四卡缓存准备。
- env.sh、setup_environment.sh和start_train.sh提供环境激活、可重复安装和服务器tmux入口。
- configs/gpu_paper_chunk30_4gpu_20261114.json保存CUDA 3、4、6、7及所有正式配置。

## 实际验证

依赖244包兼容检查、脚本语法、四卡CUDA反向/NCCL、JAX GPU、SAM2、全量数据与归一化验证通过。模型源码官方哈希一致。每百步滚动保存和四卡共享优化器恢复小模型验证通过。ModelScope gpu目录三个初始文件实际上传并回读校验通过。完整策略短测结果接续记录于handoff和docs。
'''
(p/'tempo_code_changes_gpu_server.md').write_text(changes)
(p/'tempo_code_changes.md').write_text(changes)
(root/'config.json').write_text(cpath.read_text())
print('project handoff and change log updated')
