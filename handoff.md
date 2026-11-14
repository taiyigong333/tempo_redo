# ma800 TEMPO项目交接

更新日期为2026-11-14。正式训练于北京时间05:19自动启动，MVAE100轮与decoder80轮已完成，06:21起在CUDA1、2上进行主策略全量微调，监督进程持续运行。当前执行依据docs/两卡全量训练执行方案.md与configs/formal_chunk30_2gpu_20261114.json。

## 最新用户要求

- 正式训练使用tmux，结束后自行关闭，并保留日志。
- 使用第2、第3张卡。用户已明确确认对应CUDA编号1、2。
- 两卡独占。显卡占用时每10分钟自动探测，继续等待。
- 每半小时巡检实验健康，每100step发布最新状态。
- 策略每2000步保存，最终步也保存。
- 全部393条轨迹参与训练，训练过程监控训练loss。
- 预测chunk30，使用ma800本地基座的PyTorch转换版本，W&B记录。
- 训练状态驻GPU，优先梯度累积。
- 维护本文件及独立tempo_code_changes.md。

以上要求覆盖早先三卡、314/79划分与1000步保存提案。

## 服务器与路径

SSH别名ma800，10.160.4.185:2223，用户zhoushuangfan。项目根/data/zhoushuangfan/chuanjun_gong/tempo，源码code/TEMPO，环境code/TEMPO/.venv。torch2.7.1+cu126、Transformers4.53.2、W&B0.24.2。

- 原始数据data/data11_7/piperx_rotating_red_square_tempo.zip，已解压同级piperx_rotating_red_square。
- 实际数据393轨迹、44398帧、786视频、30Hz、7D、head/left_wrist两相机。
- action[t]等于state[t+1]，前阶段已全量读取核验。
- 基座来源/data/zhoushuangfan/chuanjun_gong/base_model/pi0.5_base。
- PyTorch基座caches/pi05_base_pytorch/model.safetensors。
- 已有全393轨迹SAM2缓存caches/sam2/piperx_rotating_red_square与caches/sam2_npy。
- 无损RGB/state/action缓存caches/piperx_rgb_v1。

源码基线dd8342a1234e7166b856ffec2d7917b51cce05e4。开始时pyproject、uv.lock及environment_ma800.md已有环境改动，本次保留。旧三卡短测方案与结果在docs/实验方案.md和docs/test_report.json，当前执行以本文件开头指向的两卡全量方案为准。

## 全量归一化已重算

configs/full_data_chunk30_2gpu_20261114/split.json包含393个train_episodes和空val_episodes。

configs/full_data_chunk30_2gpu_20261114/norm/norm_stats.json使用全部44398帧及30帧动作窗口拟合。六关节按窗口起点state相对化，夹爪绝对值，保存mean/std/q01/q99。manifest记录393训练轨迹、0验证轨迹、44398帧和window30。

正式policy和decoder显式传入这版split/norm。旧314/79统计保留于configs/norm供既有记录引用。用户更新数据要求后，执行的是全量统计计算和正式任务部署。

## 当前正式配置

实验目录experiments/formal_chunk30_2gpu_20261114。

- CUDA1 UUID GPU-c30db0fb-bf1f-603e-911c-ab38facc71bb。
- CUDA2 UUID GPU-9b07f235-c0fb-943d-446b-0c6939574039。
- 两卡global128，每卡micro32，累积2，每rank64。
- 完整π0.5+TEMPO-MOT/ACT/u_t全量微调，horizon/window30，ut_dim64。
- BF16计算，FP32主权重和动量在GPU分片。
- 5000更新，warmup500，lr5e-5→5e-6，AdamW(.9,.95)，eps1e-8，wd1e-10，clip1，seed42。
- compact_prefix=false，checkpoint_joint_forward=false，内部逐层重算启用。
- 策略保存2000/4000/5000，状态每100step，eval_interval=0。
- MVAE100轮、decoder80轮，二者global64、lr1e-4、wd1e-4、余弦、clip1。
- 辅助best模型按训练loss选择，全部393轨迹用于训练。

早期三卡共享测量曾预测16～24秒/策略更新。06:24正式日志实测稳定约12.7秒/更新，约10样本/秒，两卡峰值67.7/67.4GiB。5000步策略计算约17.7小时，保存耗时另计。当前micro32，OOM时监督程序减半micro并保持global128。64/卡候选包尚未应用。

## 已启动的tmux与自动调度

会话tempo_chunk30_2gpu_20261114，监督进程初始PID163884，源代码tools/piperx/pipeline.py。

根状态runtime/latest_status.json，同时镜像到runtime/formal_chunk30_2gpu_20261114/latest_status.json及实验目录status.json。监督程序先等待两卡空闲，再运行mvae→encode→decoder→policy。

启动条件为两卡无CUDA计算进程、显存占用低于1024MiB、利用率≤5%，每600秒检查。首次检查两卡均被其他任务占约40GB，当时训练子进程为空。05:19下一轮探测发现两卡空闲并自动启动MVAE，首次运行已确认两张卡只存在本实验GPU进程。等待状态记录next_probe_at。

每1800秒由服务器监督程序健康巡检，记录至runtime/formal_chunk30_2gpu_20261114/health_checks.jsonl。检查进程、指标更新时间、数值有限性及外部GPU争用。异常时结束本实验子进程，从完整checkpoint恢复，阶段最多3次执行尝试，重试等待600秒。

完整实验正常完成后，监督程序及专用代理退出，tmux会话自动结束。日志保留于实验logs/tmux.log及各阶段mvae/encode/decoder/policy.log。

## W&B与网络

entity=gongcj-beijing，project=tempo-piperx-ma800，group=formal_chunk30_2gpu_20261114。各阶段启动后，run地址写入其latest_status.json及日志。

服务器专用出口http://127.0.0.1:17892已启动，初始代理PID163891，W&B API已回读账户成功。代理复用服务器已有mihomo与节点配置，生成仅允许W&B及云存储相关域名的本机监听配置。私有配置在/home/zhoushuangfan/.cache/tempo_wandb_proxy，权限限制，源码和项目文档不含凭据。

旧的已授权本地SSH临时出口17891仍作为备用，原工具会话15311。当前tmux实际选择服务器17892，正常执行依赖服务器端组件。

## 最新检查点

正式任务已进入policy阶段，策略检查点将位于experiments/formal_chunk30_2gpu_20261114/policy/step_00002000、step_00004000、step_00005000。

旧短测完整step6位于experiments/preflight_20261114/ddp_b128_m4_gpu_outer/step_00000006，属于先前3卡配置。正式训练从指定基座与正式辅助模型开始。

## 代码与注意事项

独立修改记录tempo_code_changes.md包含本次新增pipeline/runtime_status/wandb_proxy/compute_full_norm、NPROC_PER_NODE多卡入口、100step状态、全量数据与训练loss监控支持。

历史已验证的执行优化为关闭外层重复重算，真实样本loss与830个梯度张量差0。前缀压缩联合试验梯度差2.98%，当前compact_prefix=false。SAM2融合原先双零初始化已修复为零gate与随机输出投影。

恢复时保持world_size2及global128，正常恢复沿用相同micro-batch。完整模型和优化器分片原子发布后更新latest.json，checkpoint附norm/split/model_config。阶段指标写入metrics.jsonl，恢复时保留故障记录并处理检查点后的日志尾段。

## 接续操作

先读runtime/latest_status.json、tmux会话、logs/tmux.log和health_checks.jsonl，判断是等待、运行、恢复、完成还是失败。当前任务已授权，后续应继续维护既有运行。全部数据要求与两卡独占要求以本文件最新用户要求为准。


## 05:35运行回读

GPU1、2仅有本实验175710/175711进程，各约1462MiB，正在小型MVAE辅助阶段。W&B run ni076muo，group=formal_chunk30_2gpu_20261114，MVAE第31轮训练loss0.0009806。实际全量配置、两卡与在线指标已回读。实时进度继续以runtime/latest_status.json为准。

## 05:39定时巡检已执行

服务器监督程序已在05:39:15实际执行首个30分钟定时巡检。mvae子进程存活，最新指标距检查约11秒，loss等数值有限，两卡仅有本实验进程，external_gpu_pids为空，healthy=true。tmux该窗口remain-on-exit已明确设为off，程序结束后窗口自动关闭。


## 2026-11-14 06:20 后续交接补充

用户明确要求在 ChatGPT 中创建每30分钟运行的定时任务，检查训练失败并恢复，训练完成后上传到 ModelScope 数据集 taiyigong3333/tempo。此前已经部署的每1800秒监督程序运行在 ma800 的 tmux 内。ChatGPT 定时任务尚未创建，当前会话运行规则禁止为单次作业的完成监控创建应用自动任务，不能把服务器监督程序称为已完成用户的 ChatGPT 定时任务要求。

本次实时回读 runtime/latest_status.json，updated_at 为2026-11-14T06:20:36+08:00，阶段 policy，state=running，策略阶段初始化，world_size=2，total_steps=5000，controller_pid=204828，child_pid=243438。W&B run https://wandb.ai/gongcj-beijing/tempo-piperx-ma800/runs/8wxmmo6x 。MVAE与decoder阶段已通过，主策略是否完成首个优化器更新尚需后续回读确认。

前一轮的GPU常驻数据、fused AdamW及辅助forward编译已部署，MVAE从70轮检查点恢复后完成100轮。测量约27.87秒/轮降至13.4秒/轮，decoder约7.6秒/轮。上述测量沿用前一轮交接记录，当前回读没有重新测量。

64/卡、累积1和视觉逐层checkpoint的更新包已放在 runtime/formal_chunk30_2gpu_20261114/full_gpu_policy_update.tgz，但前一轮边界切换超时，未应用。运行控制器仍使用32/卡、累积2配置，global128。不能将本地64版本配置当作当前已运行参数。

ModelScope 上传尚未实现或验证，SDK隔离环境、私密登录、上传清单、完成触发、重试和远端校验仍待落实。凭据来自用户消息，禁止写入项目文档或训练日志。


## 2026-11-14 06:30 上传部署与汇报要求

用户最新要求每小时向其汇报并监督训练。本轮已实际回读正式日志，06:29前后策略完成第31步，loss1.27297，步耗时12.76秒。服务器监督程序继续每1800秒检查并恢复。ChatGPT应用定时任务仍未创建，当前会话的应用规则限制为单次作业的完成监控创建应用自动任务。不要声称每小时聊天通知已经安排成功。

ModelScope独立环境 runtime/modelscope_uploader/.venv 已安装modelscope1.40.3、modelscope-hub0.7.1。使用用户提供的凭据完成HubApi.login并核验taiyigong3333/tempo数据集访问。认证材料仅位于服务器用户私有缓存，目录700、文件600，未写入项目或上传内容。

新增 tools/piperx/upload_results.py，已部署到服务器。现有tmux增加modelscope_publish窗口，其进程268540等待policy.complete.json、policy completed状态及2000/4000/5000三个完整检查点。上传时创建稳定快照，checkpoint采用同盘硬链接。上传目录为runs/formal_chunk30_2gpu_20261114，内容包括三份完整策略checkpoint、MVAE/decoder、norm/split/config、训练日志、指标、交接与修改记录及关键源码。上传清单包含SHA256，远端校验文件数、大小及可用SHA256。禁用删除远端内容，开启SDK断点续传，失败30分钟重试。完成后窗口退出，训练窗口结束后tmux随最后窗口退出。

上传状态 runtime/formal_chunk30_2gpu_20261114/modelscope_publish/latest_status.json，日志 experiments/formal_chunk30_2gpu_20261114/logs/modelscope_publish.log，快照及manifest位于同runtime下modelscope_publish/bundle。check-only已实际返回authentication=ok、dataset_access=ok、training_complete=false。语法检查及临时合成产物工程验证通过，实际大文件上传仍待训练完成。未进行模型评测。

本地configs/formal_chunk30_2gpu_20261114.json已重新同步服务器实际32/卡配置。前述64/卡更新包仍未部署，避免把候选参数作为当前运行参数。当前显存峰值约67.7GiB，保持现有正式训练连续进行。


## 2026-11-14 逐检查点即时上传（覆盖此前全部完成后才上传）

用户最新要求第2000、4000、5000步的每个完整检查点分别立即上传，并每小时检查上传效果。已将upload_results.py改为三个独立发布worker，分别有锁、状态、manifest及SDK续传缓存。每个worker每60秒检查原子发布的目录及latest.json，确认两rank文件与model/config/norm/split齐全且非空后开始上传。2000/4000步上传期间策略继续训练，较慢的上传不会阻止后续检查点启动上传。最多三个worker并行，每个SDK上传并发4。

目标路径分别为数据集taiyigong3333/tempo下的runs/formal_chunk30_2gpu_20261114/policy/step_00002000、step_00004000、step_00005000。各自manifest.json包含文件大小和SHA256。上传后回读远端列表，校验大小和可用SHA256才标记completed，失败30分钟后续传重试。

最终worker在全部三个检查点远端校验成功且训练完整结束后上传辅助模型、配置、日志、文档、关键源码及聚合manifest。最终元数据上传不重新传三份大检查点，聚合清单仍覆盖这些远端文件。全部成功后上传窗口自行退出。

上传窗口已单独重启为PID280713，训练控制器204828与训练窗口持续运行。三个检查点状态位于runtime/formal_chunk30_2gpu_20261114/modelscope_publish/step_0000xxxx/latest_status.json，对应progress.json记录SDK上传计数，root latest_status.json记录聚合阶段。当前检查点尚未生成，三个worker等待中。实际大文件上传尚未执行。

最小工程验证覆盖2000步提前上传、4000步未就绪等待、独立快照与状态、硬链接、最终元数据汇总和远端校验，通过。服务器语法检查通过。06:43:31的正式100step状态已实际更新，loss0.78790，步耗时12.74秒，符合每100step发布要求。06:26:31半小时巡检为healthy=true，两卡计算利用率99%/100%，无外部GPU进程。

用户仍要求ChatGPT应用定时任务。当前会话规则限制为单次训练与上传监控创建应用自动任务，该任务尚未创建。当前会话可继续人工监督和按小时汇报，不能声称跨会话聊天通知已配置。


## 原版对照核查的接续注意

用户询问修改是否影响模型效果与部署，已只读核查服务器实际git diff及官方推理入口。SAM2融合初始化修正会改变训练行为，自定义FP32主权重GPUAdamW会改变数值轨迹，7D/chunk30及相对动作空间需要部署配置对齐。当前checkpoint是标准safetensors加自定义训练目录，原create_trained_policy需要匹配配置和显式norm_stats或assets目录，ut_decoder_path等绝对路径需要迁移。详细事实已追加tempo_code_changes.md。未进行正式模型部署测试或效果评测。当前每个早期checkpoint单独上传时，辅助decoder仍在最终上传阶段才发布，这一部署材料缺口需要如实说明，不能声称每份早期远端checkpoint可独立部署。


## 2026-11-14 用户要求还原后重新训练（覆盖先前运行信息）

用户要求在本人数据上复现论文效果，保持模型结构和输出维度，chunk固定30。新实验为paper_restore_chunk30_2gpu_20261114，CUDA1、2，tmux同名。策略从本地转换基座重新初始化，旧策略训练产物保留在原目录。

服务器src/openpi/models/pi0_config.py和src/openpi/models_pytorch/pi0_pytorch.py已逐字节恢复至git HEAD基线dd8342a1234e7166b856ffec2d7917b51cce05e4，git diff --exit-code确认无差异。恢复范围包括SAM2输出投影双零初始化、完整prefix、原始重计算路径和原始训练图像预处理。删除训练入口使用的额外模型配置开关。所有恢复前文件备份于runtime/paper_restore_chunk30_2gpu_20261114/before_restore。

策略直接使用torch.optim.AdamW，与官方入口使用相同参数精度和状态类型，betas(.9,.95)、eps1e-8、wd1e-10。自定义GPUAdamW已从训练调用链移除。试验过PyTorch ZeroRedundancyOptimizer，最小测试发现其不支持混合BF16/FP32参数，因此正式运行采用直接AdamW，分片方案未投入训练。检查点rank_N.pt现在保存标准AdamW状态，该新实验采用原生恢复格式。

维度核对为action_dim32、u_t64、flow_positions1、action_horizon30、ut_window30，机器人端有效动作7维。此约定与原官方单臂u_t配置一致，仅窗口按用户要求为30。每卡micro16、累积4、global128，5000步，保存2000/4000/5000。全部393轨迹及全量norm继续使用。

本次自主选择复用原实验已完成的MVAE100轮、u_t编码和decoder80轮，复制至新实验目录。这些辅助模型保持既有7D、latent64、window30，其架构未涉及本轮策略代码还原。未重新进行辅助训练。保留本地词表、基座转换兼容修复、数据适配与日志监督代码。

模型恢复与入口语法检查通过。已启动策略训练，W&B为https://wandb.ai/gongcj-beijing/tempo-piperx-ma800/runs/a1nfwov0。启动确认和首批实际更新应以接续记录及实时policy/metrics.jsonl为准，不能把初始化状态当成更新成功。

用户已明确回复“继续沿用”授权上传至ModelScope数据集taiyigong3333/tempo。新tmux内modelscope_publish窗口已启动，目标runs/paper_restore_chunk30_2gpu_20261114，每2000/4000/5000步完整checkpoint独立上传，当前等待检查点。服务器每1800秒巡检及恢复沿用。实际上传、完整检查点恢复与任务效果尚未在本轮验证。

注意原始SAM2融合使用零gate和零输出投影，恢复后该残差分支的初始gate梯度可能为零。本轮严格保持原始实现，不能将先前修改后得到的非零gate梯度当成本轮验证标准。恢复代码不等于已经验证论文效果，最终任务成功率仍需评测。


### 新训练首步实际验证

北京时间2026-11-14 07:01，策略完成第1次更新，loss1.2163374、grad_norm0.9708378、16.19秒/步，两卡峰值各32.72GiB，SAM2 gate梯度为0，符合恢复后的原始双零初始化行为。实际模型合同日志已输出32维动作接口、64维u_t和chunk30。后续以新实验policy/metrics.jsonl为准。


## 2026-11-14 提高单卡批量与吞吐

用户要求提高显存利用率和训练效率。正式配置已由每卡micro16、累积4调整为每卡micro32、累积2，全局batch128。chunk30、32维动作接口、64维u_t、7维机器人动作与官方模型源码沿用。训练算法继续使用原生torch.optim.AdamW。

旧配置实际稳定约14.8～15.0秒/步，torch峰值44.45GiB/卡，驱动显存约46.3GiB/卡，GPU计算利用率100%。调整前最后记录step13，尚未生成首个完整checkpoint，因此停止旧策略并从基座重启。旧policy目录由现有recover_outputs归档至实验interrupted_attempts下，保留日志与指标。MVAE、u_t、decoder继续复用。

当前监督进程298358，策略launcher298446，tmux训练窗口train_micro32。新W&B为https://wandb.ai/gongcj-beijing/tempo-piperx-ma800/runs/zaqa28oq。上传窗口进程293294持续运行，目标目录沿用runs/paper_restore_chunk30_2gpu_20261114。

runtime/paper_restore_chunk30_2gpu_20261114/micro32_change_record.json记录调整前最后五步及参数，config_before_micro32.json保存配置备份。micro-batch划分变化会改变随机数分配及浮点归约次序，未宣称逐位数值一致。实际吞吐收益需要由新配置连续更新验证，不能只依据显存占用下结论。


### micro32实测确认

新配置已实际完成6次优化器更新，排除首步后5步耗时中位数14.29671秒，对比调整前最后5步中位数14.87260秒，吞吐提升4.028%。第6步loss1.2548223、grad_norm0.940291，torch峰值61.80GiB/卡，驱动显存CUDA1为66218MiB、CUDA2为65838MiB，约64GiB/80GiB，两卡计算利用率100%。这属于短程吞吐测量，完整训练速度以持续日志为准。git diff --exit-code再次确认两份官方模型源码无差异。当前继续使用micro32/累积2/global128，未尝试micro64。

验证详情在runtime/paper_restore_chunk30_2gpu_20261114/micro32_verification.json。当前第2000、4000、5000步保存与上传约定继续执行。


## 2026-11-14 改为每1000步保存并上传（最新要求）

用户明确要求每1000步保存并上传一次。当前正式配置policy.save_interval=1000，完整检查点及独立上传计划为1000、2000、3000、4000、5000步，沿用ModelScope数据集taiyigong3333/tempo与runs/paper_restore_chunk30_2gpu_20261114路径。此规则覆盖此前2000/4000/5000安排。

pipeline.py启动参数改为读取policy.save_interval，旧配置未指定时仍默认2000。upload_results.py统一从policy.steps与save_interval计算上传步数，最终步始终包含。检查点等待、最终完成条件、汇总manifest和发布README共享同一计划。五个独立worker并发等待，保留各自续传、锁、状态与远端校验，慢上传可与后续检查点发布并行。

旧运行的保存间隔固定在启动参数中，没有运行中更新入口。调整时已到step20但尚无完整checkpoint，故停止旧策略与上传等待进程，现有recover_outputs归档旧policy后从基座重启。更改记录在runtime/paper_restore_chunk30_2gpu_20261114/save1000_change_record.json，源码与配置备份在before_save1000目录。micro32、累积2、global128、chunk30、官方模型源码和输出维度沿用。

语法检查通过。临时合成产物测试验证1000/2000间隔计划、不能整除时最终步包含、每个检查点就绪判断、下一步等待、全部五个上传完成后聚合manifest及完成条件。新配置已部署并启动，正式大文件保存上传需等到第1000步后验证。


新间隔运行已确认实际命令含--save-interval 1000，五个上传worker均为waiting_for_checkpoint。重启后第1步loss1.2238607、grad_norm0.9437041，实际更新成功。当前W&B为https://wandb.ai/gongcj-beijing/tempo-piperx-ma800/runs/9sfyj5af，监督进程303043、策略launcher303135。两份官方模型源码git diff仍为空。第1000步预计按约14.3秒/步在约4小时后产生，实际大文件保存与上传待验证。


## 2026-11-14 17:14 训练中止原因核查

实际日志确认09:41:46半小时巡检发现CUDA1、2出现外部GPU进程524861、524862，随即按两卡独占规则向本实验torchrun303135发送SIGTERM。此前训练loss和梯度有限，最近记录step618、loss0.13678077。日志没有显示本次因OOM或数值异常退出。此事件由服务器监督进程触发，与本地电脑关闭无关。

截至17:14，监督进程303043和上传进程303046仍在tmux内。状态waiting_for_gpu，每600秒探测，外部进程仍各占约11GB，因此一直等待两卡独占。未修改用户已有独占约束，未终止其他任务，未重新启动训练。

本轮在首个1000步检查点之前被终止，policy目录没有step检查点，latest.json不存在，上传仍等待检查点。监督程序直接停止子进程之前没有紧急保存，这属于现有恢复流程缺口，不能宣称可以从618步继续。资源恢复后现有逻辑会归档当前policy并从基座重训，历史指标保留。用户当前询问原因，本轮只检查与记录。


## 2026-11-14 百步最新检查点、共享GPU与辅助模型独立发布（覆盖此前资源规则）

用户明确要求维护每100步更新的最新训练参数，已开始使用显卡后不要因其他任务主动退出，可获得较多算力时允许共享；同时要求立即上传MVAE和decoder供后续使用。

train.py新增--latest-save-interval，当前100。每100次优化器更新保存完整模型、两rank原生AdamW状态、CPU/CUDA/NumPy/Python随机状态以及model_config/norm/split。先写.incomplete目录，双rank写完后原子发布step目录，再原子更新policy/latest.json。新版本完整发布后清理上一个非千步滚动版本，每1000步与最终步永久保留，继续按千步独立上传。保存间隔与latest间隔通过配置传入，未改模型结构、前向、输出维度、chunk30和优化器算法。

pipeline.py移除外部GPU进程触发不健康和停止训练的条件，仍记录external_gpu_pids。停滞和非有限数值检查保留。共享准入按配置minimum_free_memory_mib=68000判断目标两卡可用显存，基于micro32实测峰值61.80GiB预留驱动工作区。GPU瞬时利用率不作为共享启动阻断。用户要求的计算资源使用解释以本节为准，早先两卡完全独占条件作废。

当前仍用CUDA1、2、micro32/累积2/global128。旧618步只有指标、没有检查点，按现有恢复逻辑归档后从基座重新启动。监督进程843166，策略launcher843277，tmux窗口train_rolling100。现有千步上传窗口继续等待1000/2000/3000/4000/5000。

upload_results.py新增--auxiliary-only，独立锁、快照和状态，不依赖策略完成。目标为ModelScope数据集taiyigong3333/tempo下runs/paper_restore_chunk30_2gpu_20261114/auxiliary。上传mvae与decoder各best.pt/last.pt、从best.pt导出的model_config、阶段状态、全量norm/split、controller配置、匹配mvae.py及ut_decoder.py源码、使用说明和SHA256 manifest。四份辅助检查点均完成严格加载检查，decoder用64维零输入实际推理得到有限[1,30,7]输出。四份best/last实际epoch分别为MVAE99和decoder79（从0计数）。

两rank临时微型模型回归通过，覆盖100/200/1000/1100/1200步原子保存、AdamW状态加载、滚动旧版本清理、1000步归档保留；共享模式有外部GPU进程且指标正常时不会调用stop_child；辅助独立发布材料14文件生成通过。三份修改脚本语法检查通过。实际全量模型首个100步检查点需运行后核查，不能将微型工程验证当成正式权重已保存。

辅助发布窗口auxiliary_publish已启动，状态runtime/paper_restore_chunk30_2gpu_20261114/modelscope_publish/auxiliary/latest_status.json，日志logs/modelscope_auxiliary.log。辅助使用说明同步本地docs/MVAE和decoder使用说明.md。


## 用户指定ModelScope文件夹ma800（覆盖辅助发布的早先路径）

用户进一步要求放在名为ma800的文件夹下。当前辅助权重的实际发布目标改为taiyigong3333/tempo数据集根目录ma800/，其下为mvae/best.pt、mvae/last.pt、decoder/best.pt、decoder/last.pt，以及model_config、configs、source、README.md与manifest.json。配置auxiliary_upload_prefix=ma800，独立辅助worker已切换目标。旧目标上传尚未提交完成时已停止，旧缓存保留在runtime/modelscope_publish/auxiliary_previous_destination_*用于追溯。千步策略检查点继续使用既有runs/paper_restore_chunk30_2gpu_20261114/policy路径。

共享GPU策略已实际完成首批更新，第7步loss1.2273862、约18.61秒/步，GPU与外部任务共存仍继续更新。新W&B为https://wandb.ai/gongcj-beijing/tempo-piperx-ma800/runs/xkexxach。首个完整百步滚动检查点尚待训练达到100步生成。辅助模型ma800目录的实际远端发布校验状态需继续回读。


## 2026-11-14 17:37 上传完成与用户结束条件

辅助模型已完整发布到ModelScope数据集taiyigong3333/tempo的ma800/目录，worker状态completed，verified_files=15，完成时间17:37:39。四个LFS文件全部成功，15文件提交，0失败，0远端删除。manifest SHA256为5e5b337d8f55e157ad1517783d178ba8ed70b75fb631257663600f026d3c03a0。

同次回读确认共享GPU训练实际完成第24步，loss1.1924624、grad_norm0.884122、18.93秒/步。用户要求辅助上传完成且服务器已训练时结束当前操作，此条件已满足。首个百步正式检查点尚未生成，百步滚动保存路径已通过两rank微型工程验证并部署。

用户另外要求每1小时定时检查训练终止并报告原因，由用户决定后续。应用定时任务未创建，当前会话执行规则禁止为单次作业的进度、失败或完成建立应用自动任务。不要声称小时聊天通知已设置。服务器原有30分钟巡检与故障恢复逻辑仍在，未在本轮取消自动重试，也不能把它描述成由用户逐次批准恢复。


### 后续恢复改为用户决策

为落实用户“我来决定后续怎么做”，配置已改为max_stage_attempts=1及recovery_requires_user_decision=true。由于当前控制器已在内存中加载旧配置，另启tmux窗口manual_recovery_guard，每15秒检查控制器状态，仅当state=recovering且child_pid为空时记录manual_attention.json并终止已无训练子进程的控制器，阻止下一次自动重试。当前训练未重启或中断。该守护不发送聊天通知。此前本节关于自动恢复仍开启的描述已由本补充覆盖。小时应用通知仍未创建。
