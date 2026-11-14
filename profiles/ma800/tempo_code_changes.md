# TEMPO代码改动记录

更新日期为2026-11-14。本文件独立于handoff.md，记录本次ma800训练准备的修改、原因与验证结果。当前正式任务已授权，tmux监督程序已在两卡空闲后启动MVAE训练。

## 基线与范围

源码目录 `/data/zhoushuangfan/chuanjun_gong/tempo/code/TEMPO`，基线提交 `dd8342a1234e7166b856ffec2d7917b51cce05e4`。

开始时pyproject.toml、uv.lock、docs/environment_ma800.md已有环境配置改动，sam2目录已存在。总体git diff包含这些原有内容。本次保留项目目录与环境，在tools/piperx新增任务入口。

## 官方文件修改

### examples/convert_jax_model_to_pytorch.py

三处π0.5自适应归一化判断使用传入的pi05配置。原实现依赖路径是否包含pi05，本次pi0.5_base触发KeyError。修复后实际转换成功，产物为项目caches/pi05_base_pytorch/model.safetensors。

### src/openpi/models/tokenizer.py

PaligemmaTokenizer支持PALIGEMMA_TOKENIZER_PATH，读取本地基座自带词表。原默认下载行为仍由缺省配置提供。实际数据入口、策略训练与采样已验证。

### src/openpi/models/pi0_config.py

增加compact_prefix与checkpoint_joint_forward执行开关，缺省值分别为false和true。字段源于本项目历史performance实现，本次显式选择false/false。

### src/openpi/models_pytorch/pi0_pytorch.py

1. SAM2CrossAttnFusion保留零gate，输出投影使用正常随机初始化。原gate和输出投影同时为零时，二者梯度均为零，阻断运动分支学习。回归证实初始残差为零且gate梯度非零，策略短测也记录到有效gate梯度。
2. 可选压缩整批不可见prefix列，保留有效token顺序与位置编号。本次联合压缩试验出现2.9788%梯度相对L2差，推荐配置compact_prefix=false。
3. 可关闭包裹整段Gemma的外层重复checkpoint，内部逐层checkpoint继续运行。真实样本固定noise/time/增强随机种子，loss差0，全部830个梯度张量差0。
4. forward按self.training选择图像训练增强或验证路径，支持确定性eval。最终held-out flow评估与动作采样通过。

### tools/ut/train_mvae.py

从实际训练数据推断action_dim，替换14D硬编码。新增--window，默认16，本次30。归一化的缺省分支也使用实际维度。实际三卡训练入口使用同一官方MVAE结构并通过7D/30帧短测。

### tools/ut/encode_ut_windows.py

窗口长度从model.cfg.window读取，替代固定16，输出长度断言同步。正式本任务走tools/piperx/encode.py，后者已实际三卡编码393轨迹。原工具通过语法检查。

### tools/ut/train_ut_decoder.py

窗口从MVAE配置读取。新增可选--delta-joint-actions，每个窗口使用state[t0]相对化六关节，匹配策略DeltaActions与norm。原缺省调用语义保留。本任务走三卡入口train_aux.py，归一化相同，实际短测通过。原工具通过语法检查。

### pyproject.toml、uv.lock

wandb>=0.19.1固定为wandb==0.24.2，支持用户提供的新版密钥。uv解析输出更新W&B并移除其旧docker-pycreds/setproctitle依赖。曾临时安装0.31诊断兼容性，随后用uv sync --locked恢复锁定环境。最终243包通过uv pip check，torch/Transformers等训练依赖保持已记录版本。

## 新增任务入口

### tools/piperx/prepare.py

逐轨迹校验Parquet帧号、episode、时间戳、7D有限数值与两路视频帧数，生成无损RGB/state/action的npy缓存。固定seed0按轨迹80/20划分，训练集拟合norm，窗口由参数指定为30。

已实际验证393轨迹、44398帧、786视频，action[t]与state[t+1]最大误差0。split为314训练/79验证，本次默认划分已在参数方案披露。

### tools/piperx/data.py

复用官方YamInputs、DeltaActions、Normalize、TokenizePrompt和PadStatesAndActions。保留两路三帧历史、10桶10秒30Hz动作历史与尾帧u_t padding。每进程64个mmap数组LRU限制文件句柄，降低重复解码开销。模型边界state/actions转float32，图像/整数token/布尔mask使用各自dtype。

ExactBatchSampler按更新步确定全局样本顺序，global128按43/43/42分卡，micro4累积11次并按尾批样本数加权。micro1时短rank增加零权重占位，维持同步次数。回归验证无重复样本、全局梯度加权及恢复顺序。

### tools/piperx/optimizer.py

GPUAdamW按参数量均衡分片GPU上的FP32主权重与FP32一、二阶动量，计算权重采用原项目BF16/关键层FP32。FP32梯度临时转换和更新按4,194,304元素块处理，限制峰值。合并同dtype权重广播至各rank。

AdamW公式、betas、eps、weight_decay和偏置校正与torch AdamW一致，5次更新数值回归通过。全部训练状态驻GPU。checkpoint写盘时生成序列化快照。

### tools/piperx/train.py

三卡全量微调入口，缺省preflight最多20次更新，正式运行需显式选择formal。参数源自项目TEMPO配置。首次micro同步建立梯度bucket view，后续保留view并no_sync累积。冻结不参与策略前向的expert.lm_head，减少无用DDP桶。

支持GPU分片优化器、逐rank RNG与step恢复，验证rank参数同步。保存模型、每rank状态、规范化路径配置，并随新检查点保存model_config/norm/split快照。所有rank完成后原子发布目录和latest.json。

每500步及末步调用验证函数。W&B只由rank0上报loss、lr、梯度范数、显存、吞吐、checkpoint耗时和验证指标。正式入口核验正式u_t manifest与decoder路径，测试产物含test_only。

micro1、micro2、micro4及两次恢复均实际通过。micro4稳态29.41964秒/更新，峰值31.05285GiB/卡，完整checkpoint约44GB，保存31.6506秒。最终额外1步验证通过。

### tools/piperx/train_aux.py

复用官方PoEMVAE、ActionDecoder与VideoLatentDecoder，三卡global64按22/21/21精确加权。30帧、7D、latent64、beta0、池化SAM2，MVAE100轮、decoder80轮，lr1e-4、weight_decay1e-4、余弦、clip1。支持轮次边界恢复、held-out验证、best/last与W&B。

MVAE三卡5步成功。decoder首次因分位归一化产生float64标签而反向失败，显式转换float32后，三卡5步与验证成功。短测epoch和loss仅用于流程验证。

### tools/piperx/encode.py

三卡按episode独立编码MVAE均值标签，输出官方npz与训练npy，manifest记录来源、窗口、维度和test_only。实际393轨迹三卡编码完成，24.2088秒。

### tools/piperx/evaluate.py

固定seed的held-out flow评估，按完整u_t窗口计数，恢复评估前随机状态与train/eval状态。最终策略验证抽样8帧，其中4个完整窗口，flow MSE2.049576。对应测试辅助模型，此数值用于流程核验。

### tools/piperx/run.sh

统一三卡mvae/encode/decoder/policy显式阶段入口，解析项目路径，使用项目内环境和可配置W&B代理。通过脚本语法检查。正式阶段执行设计记录在docs/实验方案.md。

## 新增验证工具

- smoke_aux.py提供5步MVAE/decoder诊断及独立测试标签，生成test_only产物。
- verify.py验证split互斥、融合恒等与非零梯度、精确global128、micro1占位、采样恢复和AdamW公式，实际通过。
- verify_execution.py比较真实样本前向和全部梯度，记录通过的外层重算优化与未选用的联合prefix压缩试验。
- verify_checkpoint.py用小模型调用真实三rank保存入口，验证原子发布、model_config window30、norm/split快照，实际通过。
- train.py的动作契约检查实际调用项目采样与UtActionDecoder，输出(1,30,7)且有限。

## 尝试、恢复与当前边界

初期测试遇到其他用户任务占用约40GB/卡，整模型FP32方案两次OOM。未修改他人任务。曾短暂加入CPU卸载尝试，用户要求GPU状态后已停止该测试并移除CPUAdamW及相关参数，改用上述GPU分片主权重方案。

本地首次语法检查受默认Python缓存路径权限限制，改用可写临时缓存后通过。初次macOS归档带入的._元数据已清理，后续归档关闭该元数据。真实源码语法检查通过。

所有正式模型、完整训练指标和真实任务成功率将在用户确认后取得。本次测试与正式目录独立，详细产物见docs/test_report.json，正式参数提案见configs/formal_chunk30_v1.json。

### docs/environment_ma800.md

追加2026-11-14训练准备更新，记录W&B0.24.2、243包兼容检查、数据/基座/三卡测试完成，以及项目根方案、报告和交接文件的位置。


## 2026-11-14 两卡全量正式任务更新

用户指定CUDA1、2独占、全部393轨迹训练、策略2000步及末步保存、100step状态与半小时巡检。

### tools/piperx/compute_full_norm.py

使用全393轨迹的既有state/action缓存重新计算30帧统计，生成独立版本configs/full_data_chunk30_2gpu_20261114，train393、val0、44398帧。保持六关节窗口起点相对化与绝对夹爪，保存manifest。

### tools/piperx/runtime_status.py

新增Asia/Shanghai时间戳与原子JSON状态发布，阶段文件latest_status.json由主rank写入。

### tools/piperx/pipeline.py

新增服务器tmux正式监督入口。进程锁防重复启动，绑定CUDA1、2及UUID，每600秒检查两卡独占条件，每1800秒巡检进程、指标新鲜度/有限性和资源争用。顺序运行MVAE、编码、decoder与policy，日志独立保存。阶段异常等待600秒后恢复，最多3次尝试；policy OOM可依次将micro16降至8/4，global128。处理中断checkpoint与日志尾段并归档故障记录，完成后释放子进程/专用代理并退出tmux。

实际启动tmux会话tempo_chunk30_2gpu_20261114，首次发布waiting_for_gpu和next_probe_at，初始占用约40GB/卡，训练child_pid为空。首条服务器巡检日志已落盘。

### tools/piperx/wandb_proxy.py

复用服务器现有mihomo与代理配置，在用户私有缓存生成仅监听127.0.0.1:17892的W&B出口，限定W&B及云存储相关域名，配置文件权限600、目录700。由监督进程管理生命周期，保留既有授权出口作备用。服务器W&B账户API回读成功。

### tools/piperx/run.sh

新增NPROC_PER_NODE环境配置，默认兼容此前三卡入口，当前监督程序显式设置2和CUDA_VISIBLE_DEVICES=1,2。

### tools/piperx/data.py、train.py、train_aux.py

新增显式split_path/norm_dir参数，支持全量数据版本。policy的eval_interval=0用于当前正式配置，辅助模型在val为空时按训练loss选择best。正式W&B按实验根目录group分组，并记录全量数据路径与rank数。

新增status_interval默认100，policy与辅助模型每100优化器step原子更新状态。policy完整checkpoint保存当前所用norm/split快照，完成状态与指标写入阶段最新文件。辅助best.pt采用临时文件原子替换。保存频率由当前正式命令指定2000，最终5000同样保存。

当前修改做了源码语法检查与任务部署回读。用户指定全量训练后，执行了归一化重算、代理账户连接核验、tmux启动和等待状态检查。两卡吞吐与显存使用依据既有测量预测，实际值由正式训练记录。


实际10分钟探测已于05:19发现两卡空闲并自动启动正式MVAE。05:35回读确认CUDA1、2仅有本实验进程、全量393轨迹训练、W&B ni076muo在线。辅助训练完成阶段恢复边界初始化epoch=start-1，支持last.pt已到计划末轮时正常完成状态发布。

05:39:15服务器定时健康巡检实际运行，确认阶段进程正常、指标有限、两卡独占，healthy=true。tmux窗口选项remain-on-exit=off已生效，确保监督程序退出后关闭会话窗口。


## 2026-11-14 GPU吞吐优化及ModelScope发布

### 已部署的辅助训练优化

train_aux.py和data.py支持完整辅助窗口驻GPU、fused AdamW及forward编译，pipeline.py将这些开关配置化，并在编译失败时回退普通forward。保持FP32计算、有效batch64、MVAE100轮与decoder80轮。从MVAE第70轮完整检查点恢复后完成训练，稳定轮耗时由约27.87秒降至13.4秒。decoder已完成80轮。

主策略实际运行每卡micro32、梯度累积2、global128。正式日志稳定约12.7秒/更新，峰值67.7/67.4GiB。64/卡与视觉逐层checkpoint仅作为未部署候选保存在full_gpu_policy_update.tgz，本地源码中的对应改动尚未应用到运行进程。

### tools/piperx/upload_results.py

新增完成后发布入口，独立SDK环境运行。只有完整的最终训练状态、阶段完成marker和三个完整检查点均存在时才创建上传快照。checkpoint硬链接节省空间，日志/配置使用复制快照。输出manifest记录大小与SHA256，发布到用户指定数据集的本实验子目录。凭据从项目外私有文件读取，文本材料在上传前扫描凭据格式。SDK断点续传、四并发上传、30分钟失败重试、远端大小及可用SHA256校验均已接入。上传成功后状态原子发布并退出tmux窗口。

最小验证实际执行了远端语法编译、只读认证和数据集访问核验。本地临时合成文件验证了完成条件、硬链接、快照复用、数据集上传参数及远端校验，模拟远端大小错误时正确拒绝标记成功。未实际上传训练结果，等待5000步完成后触发。未运行模型测试或评测。

上传窗口已加入现有训练tmux，服务器训练控制器及其30分钟健康恢复流程继续执行。ChatGPT应用定时任务尚未创建。


## 2026-11-14 逐检查点发布变更

upload_results.py新增checkpoint_step与三个独立发布worker，检查latest.json步数以及原子检查点目录完整性后即可开始上传。各worker有独立锁、缓存、进度和manifest，线程池允许后续检查点在早期上传未完成时启动。最终发布阶段改为上传日志/辅助模型/元数据，并聚合三个已上传检查点的manifest，避免再次上传大文件。最终条件同时检查策略完成状态、控制器完成状态和阶段marker。

已运行临时合成产物工程验证，确认2000步提前就绪时能上传、4000步继续等待、硬链接复用、独立状态，以及最终元数据聚合清单远端完整性校验。服务器语法检查通过。仅重启上传等待窗口，训练进程未重启。当前尚无正式策略检查点，实际大文件上传待产生后验证。


## 原版对照的效果与部署影响核查

本次按服务器实际git diff与运行入口核查，基线dd8342a1234e7166b856ffec2d7917b51cce05e4。服务器当前pi0_config.py只增加compact_prefix与checkpoint_joint_forward，尚无候选vision_layer_checkpointing。本地候选源码不能视为当前运行版本。

- SAM2CrossAttnFusion初始化有实质训练行为修改。原版gate和out_proj同时为零，本次保留零gate，out_proj使用正常随机初始化，让gate获得非零梯度。这项修正由本次实施判断加入，已经用于正式训练。其对最终机器人任务效果的影响尚未评测。
- 本次按数据与用户要求使用单臂7D、chunk30、动作历史7D和u_t64，MVAE/decoder窗口同步改为30。原项目包含7D和u_t64配置，原u_t示例窗口为16。动作标签使用窗口起点实测state的六关节相对量，夹爪为绝对量，再用本次全量统计归一化。部署必须使用相同norm、六关节AbsoluteActions、30步decoder以及相同图像/SAM2/历史输入。
- GPUAdamW是新增实现，采用GPU分片FP32主权重与动量，计算权重保留项目混合精度。更新公式曾通过小规模torch AdamW数值回归，但训练数值轨迹与原版torch AdamW入口不能宣称逐位相同。optimizer分片恢复依赖本次训练入口和world_size2。未使用CPU训练状态卸载。
- 外层重复activation checkpoint关闭，内部逐层重算保留。历史固定样本/随机数比较得到loss差0、830个梯度张量差0。辅助forward编译与fused AdamW属于执行后端变化，仍可能产生浮点舍入差。prefix压缩当前关闭，64/卡与视觉逐层checkpoint候选尚未生效。
- 前向图像预处理由固定train=True改为self.training，当前model.train()训练路径仍启用增强。转换脚本按pi05显式配置识别自适应归一化，tokenizer支持基座本地词表路径。
- 主策略权重是标准model.safetensors，参数模块命名保持原模型，但本次目录把norm_stats.json/model_config.json/split.json直接放于step目录。原create_trained_policy默认从checkpoint/assets/<asset_id>读取统计，使用传入train_config构建模型，未自动读取本次model_config.json。原版默认配置不能直接视为本次部署配置。部署需要构造匹配的7D/chunk30/u_t配置、显式传norm_stats或整理assets，并将ut_decoder_path和词表路径改为部署机器实际路径。
- 当前上传worker把checkpoint和配置上传，辅助decoder/MVAE会在最终元数据阶段补齐。因此第2000/4000步远端checkpoint刚发布时，完整独立推理材料尚未承诺齐备。若用户要求每个检查点即可独立部署，需要把对应decoder与部署入口同批发布。

本轮核查为源码与配置检查，遵守用户当前不做模型测试或评测的要求。最终正式checkpoint的原入口加载和真实机器人效果均尚未验证。此前短测的动作形状验证仅覆盖早期测试产物。


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
