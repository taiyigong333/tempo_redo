# TEMPO 双服务器复现工程

收录 2026-11-14 从 ma800 与 GPU-Server 实际工作目录采集的源码、训练配置、环境锁文件、实验记录和辅助脚本。项目目录结构继续使用 `code/TEMPO`、`configs`、`docs`。

## 版本选择

| 分支 | 对应服务器 | 正式配置 | GPU |
|---|---|---|---|
| `main` | GPU-Server | `configs/gpu_paper_chunk30_4gpu_20261114.json` | CUDA 3、4、6、7 |
| `ma800` | ma800 | `configs/paper_restore_chunk30_2gpu_20261114.json` | CUDA 1、2 |

`main` 采用 GPU-Server 的四卡训练与共享优化器检查点版本。主分支的 `profiles/ma800` 保存两卡版本的差异源码、配置、文档及恢复守护脚本，可直接浏览。ma800 完整源码位于 `ma800` 分支，部署时使用该完整分支。

```bash
git clone --branch main git@github.com:taiyigong333/tempo_redo.git tempo-gpu
# 在ma800另建独立目录检出对应版本。
git clone --branch ma800 git@github.com:taiyigong333/tempo_redo.git tempo-ma800
```

## 复现约定

- TEMPO 上游基线为 `dd8342a1234e7166b856ffec2d7917b51cce05e4`，基座为 π0.5。
- 策略动作接口32维，机器人有效动作7维，u_t 64维，action horizon与MVAE窗口均为30。
- 两台服务器当前核心模型配置与前向源码一致，正式训练沿用官方模型结构与原生 AdamW。
- 全部393条轨迹、44398帧用于训练，全局batch128。历史314/79划分仅作为早期实验记录保留。
- 每100步更新状态并保存最新可恢复检查点，每1000步保存里程碑并上传ModelScope。
- GPU-Server 使用四卡micro32、累积1，ma800 使用两卡micro32、累积2。

## 使用前准备

配置保留服务器原始绝对路径。GPU-Server 项目根为 `/data-minio/taiyi/tempo`，ma800 项目根为 `/data/zhoushuangfan/chuanjun_gong/tempo`。新目录部署时应核对正式配置中的 `project`、`experiment_root`、`runtime_directory`、数据划分、归一化、基座和分词器路径。恢复现有训练还需原实验目录和匹配rank数量的检查点。

仓库收录源码、配置、归一化统计、轨迹划分及已有验证报告。数据集、基座、SAM2权重、训练缓存、实验权重、uv工具和虚拟环境需在目标机器准备。凭据由目标机器既有私有文件提供。

GPU-Server 现有环境下的启动入口如下。先检查已有tmux与状态文件，再决定是否启动。

```bash
cd /data-minio/taiyi/tempo
bash start_train.sh configs/gpu_paper_chunk30_4gpu_20261114.json
```

首次安装需先准备 `code/TEMPO/.tools/bin/uv`，再运行 `setup_environment.sh`。该脚本安装项目内Python3.11.15、锁定依赖并应用Transformers时序注意力补丁。发布进程还使用 `.tools/uploader` 中的ModelScope环境，版本与准备方式见使用说明。

ma800 的启动、现有恢复守护与环境说明见 `docs/integration/使用说明.md`。`archive/runtime` 保存历史准备和验证脚本，其中含原始服务器路径与特定阶段行为，运行前先阅读用途。该目录中的 `.complete.json` 和动态运行状态由服务器重新生成。

## 文档导航

- `docs/训练前应当考虑的问题集合.md` 汇总训练目标、数据、动作、模型、恢复和部署的启动前问题。

- `docs/原项目改动及训练部署影响.md` 逐项说明相对官方基线的改动及训练、部署和模型结构影响。

- `docs/integration/使用说明.md` 记录服务器入口、外部依赖与凭据文件位置。
- `docs/integration/整合记录.md` 记录采集来源、文件取舍、版本差异和验证范围。
- `docs/integration/经验与教训.md` 记录环境、复现约束、吞吐优化、检查点和发布实践。
- `docs/integration/交接.md` 给后续接手者提供当前状态和继续工作入口。
- `docs/integration/*_source_manifest.json` 保存两台服务器采集文件的SHA256，`source_comparison.json` 保存逐文件差异。
- 原始 `handoff.md` 与 `tempo_code_changes*.md` 按时间保留，最新约定以正式配置及本目录整合说明为准。
- 官方训练与推理资料保存在 `code/TEMPO/README.md`。

上游源码来自 [TEMPO](https://github.com/tempo-robot/TEMPO/tree/dd8342a1234e7166b856ffec2d7917b51cce05e4) 与 [SAM2](https://github.com/facebookresearch/sam2/tree/2b90b9f5ceec907a1c18123530e92e794ad901a4)。许可证保存在各自源码目录，Gemma许可证同时保留。
