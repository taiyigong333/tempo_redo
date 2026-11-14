# GPU-Server TEMPO 独立改动记录

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


## 完整策略保存恢复实测

独立四卡micro32、global128、累积1完成step1、2，保存完整step2检查点，并实际恢复完成step3，退出码0。检查点约20GB，保存28.04秒，峰值分配显存约61.80GiB每卡。更新耗时14.42、13.60、14.61秒，测量时正式MVAE同时占用同四张GPU。这组数字用于流程验证，正式策略吞吐以正式日志为准。短测W&B为h1ksq8v1。详见docs/policy_smoke_gpu_server.json。

所有正式阶段使用独立W&B项目gongcj-beijing/tempo-piperx-gpu-server。当前正式MVAE run为8o8ivcfb，后续自动执行encode、decoder与5000步策略训练。首次1000步正式策略检查点尚未生成，上传任务已在等待。


## Policy显存目标与重算对照

用户要求policy训练实际显存约80%，MVAE和decoder继续高吞吐。现有policy为四卡micro32、累积1、global128，独立完整策略短测峰值已分配61.80GiB每卡。实际nvidia-smi占用还包括预留显存及CUDA上下文，正式policy启动后需按实际峰值核验。

本轮独立对照仅调整运行时梯度检查点标志，模型源码保持官方哈希。真实样本loss、830个梯度张量及CPU/CUDA随机状态逐元素一致。批量32关闭全部外层重算在独立测试进程中两次OOM，因此该候选未进入正式配置。正式训练继续使用现有完整重算策略。对照结果见docs/checkpoint_execution_benchmark.json。
