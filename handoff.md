# GPU-Server TEMPO 训练交接

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


## 完整策略保存恢复实测

独立四卡micro32、global128、累积1完成step1、2，保存完整step2检查点，并实际恢复完成step3，退出码0。检查点约20GB，保存28.04秒，峰值分配显存约61.80GiB每卡。更新耗时14.42、13.60、14.61秒，测量时正式MVAE同时占用同四张GPU。这组数字用于流程验证，正式策略吞吐以正式日志为准。短测W&B为h1ksq8v1。详见docs/policy_smoke_gpu_server.json。

所有正式阶段使用独立W&B项目gongcj-beijing/tempo-piperx-gpu-server。当前正式MVAE run为8o8ivcfb，后续自动执行encode、decoder与5000步策略训练。首次1000步正式策略检查点尚未生成，上传任务已在等待。


## Policy显存目标与重算对照

用户要求policy训练实际显存约80%，MVAE和decoder继续高吞吐。现有policy为四卡micro32、累积1、global128，独立完整策略短测峰值已分配61.80GiB每卡。实际nvidia-smi占用还包括预留显存及CUDA上下文，正式policy启动后需按实际峰值核验。

本轮独立对照仅调整运行时梯度检查点标志，模型源码保持官方哈希。真实样本loss、830个梯度张量及CPU/CUDA随机状态逐元素一致。批量32关闭全部外层重算在独立测试进程中两次OOM，因此该候选未进入正式配置。正式训练继续使用现有完整重算策略。对照结果见docs/checkpoint_execution_benchmark.json。


## 2026-11-15 完成核查与旧产物整理

训练于2026-11-15 03:49:57完成5000步，最后loss为0.011362534，稳态6.585秒/步。ModelScope各1000至5000步、辅助模型和汇总材料于03:54:12前全部完成上传。已再次回读全部远端材料，大小和可获取SHA256核验通过，报告docs/completion_upload_verification_20261115.json。

最终检查点仍位于 `/data-minio/taiyi/tempo/experiments/gpu_paper_chunk30_4gpu_20261114/policy/step_00005000`，包含model.safetensors、model_config、norm、split、config、共享AdamW状态和四rank随机状态。最终模型全部文件SHA256再次与上传清单一致，safetensors结构及rank状态读取通过。decoder、MVAE、u_t、数据、缓存、环境和配置保留原路径。

step1000、2000、3000、4000、setup_smoke、检查点小模型验证目录及上传bundle副本已按原相对目录移入 `/data-minio/taiyi/tempo/remove`。上传完成状态保留，状态文件新增local_bundle_archive指向归档副本。移动清单位于remove/move_manifest_20261115.json，记录所有原路径与目标路径。此次使用同盘rename，文件内容与硬链接关系保留。

续训使用policy/latest.json所指step5000完整状态及原四rank设置。模型部署还需model_config.json所指的decoder/best.pt、归一化和项目所需SAM2配置与分词器。当前训练效果记录属于训练指标，真实任务效果待评测。


## 用户授权删除训练缓存与本地优化器状态（20261115_224613）

已删除caches/pi05_base_pytorch、caches/piperx_rgb_v1以及最终step5000中的rank_0.pt至rank_3.pt。删除前再次回读ModelScope并逐文件核对4份rank状态的大小和SHA256。最终策略权重、model_config、norm、decoder、MVAE、u_t、SAM2特征缓存、原始数据及环境继续保留。

本地step5000目录现在用于部署。恢复完整续训时，先从ModelScope数据集taiyigong3333/tempo的gpu/gpu_paper_chunk30_4gpu_20261114/policy/step_00005000/下载rank_0.pt至rank_3.pt到该目录，再从原始数据重新生成RGB缓存。RGB缓存来自数据解码，当前ModelScope实验上传内容为模型和配置。清理前核查发现原始JAX基座目录/data-minio/taiyi/base_model/openpi_cache/openpi-assets/checkpoints/pi05_base已不存在。未来需要初始PyTorch基座时，需重新下载原始基座并转换，完整step5000断点恢复使用其自身model.safetensors。

数据恢复时复用tools/piperx/prepare.py中的prepare_episode函数即可重建RGB/state/action缓存。现有prepare_gpu_server.py使用已完成的SAM2报告跳过缓存，RGB删除后需要显式重建，不能仅凭SAM2完成标记推断RGB完整。

项目占用从51.26GiB降至19.37GiB。清理记录为docs/cleanup_requested_20261115_224613.json。runtime/latest_status.json记录local_resume_state_available=false，训练已完成状态保留。


## 2026-11-16 顺时针batch2、3、4新实验



在 GPU-Server 的 /data-minio/taiyi/tempo 上，复用参考实验的训练方式，使用 batch2、3、4 顺时针数据从 π0.5 基座训练。数据来源 ModelScope taiyigong3333/tempo/grab_rotating_red_square_clockwise_20261116，共289条、30285帧、578视频，任务文本 grab the rotating red square。action继承转换流程的下一帧观测状态代理，末帧保持。

## 执行设计

实验名 gpu_clockwise_b234_chunk30_4gpu_20261116，GPU3、4、6、7，MVAE100轮、decoder80轮、policy5000步，chunk30、global batch128、micro32、seed42，优化器和调度沿用参考配置。沿用参考实验全部轨迹用于训练、无验证集的约定，任务效果需后续评测。

基座 /data-minio/taiyi/basemodel/openpi_cache/openpi-assets/checkpoints/pi05_base，转换为 caches/pi05_base_pytorch。新数据、RGB缓存、SAM2缓存、归一化、辅助模型、策略输出均使用新实验路径。W&B沿用 tempo-piperx-gpu-server 项目并以实验名分组，ModelScope发布前缀 gpu/gpu_clockwise_b234_chunk30_4gpu_20261116。

## 当前状态

已完成读取两段参考聊天、核对服务器资源与基座，已定位原工具硬编码的缓存路径和393轨迹判断。数据已下载到GPU-Server，877个文件的大小与SHA256全部通过。最小适配已部署，全部工具语法检查通过。已启动tmux clockwise_prepare_20261116，当前转换π0.5基座，完成后自动进行四卡SAM2缓存、全量归一化、MVAE、encode、decoder、policy及发布。轨迹长度62至210帧，全部满足30帧窗口。模型源码两文件SHA256与参考实验一致。

## 变更范围与兼容

工具通过可选环境变量 TEMPO_RGB_CACHE、TEMPO_SAM2_CACHE、TEMPO_PRETRAINED_DIR 选择数据缓存，未设置时保持原路径。准备和统计工具另支持 TEMPO_DATASET_ROOT、TEMPO_PREPARE_REPORTS、TEMPO_VALIDATION_REPORT。监督程序从实验配置向子进程传递路径，encode完成判断和发布材料轨迹数从配置读取。SAM2准备跳过前额外检查RGB存在，修复此前清理缓存后可能被错误跳过的问题。模型实现保持原版本。

## 路径和接续

启动或恢复使用 bash /data-minio/taiyi/tempo/start_clockwise_train.sh。该入口检查新数据缓存与清单后创建三个tmux窗口，复用监督和上传程序。正式状态为 experiments/gpu_clockwise_b234_chunk30_4gpu_20261116/status.json，最新汇总为 runtime/latest_status.json。准备阶段日志位于 runtime/clockwise_setup_20261116，下载和准备进程分别记录download.exit与setup.exit。归一化位于configs/clockwise_b234_chunk30_20261116，RGB和SAM2位于caches/clockwise_b234_20261116。原工具备份位于runtime/clockwise_setup_20261116/original_tools。

本次沿用参考实验无验证集和固定100/80辅助训练轮数。尚无训练完成或实机任务成功率结论。W&B和ModelScope由参考流程记录和发布，不在文档内保存凭据。

## 数据形式与保存策略核对

已按论文https://arxiv.org/html/2609.16864v1和项目src/openpi/policies/yam_dual_policy.py核对。全部289条、2023个边界样本通过官方LoadActionHistory和LoadSam2Tokens比较。SAM2特征为T×256×8×8，输入为64×256；动作历史10桶×30帧，截止t-1，原始单位、越界补0、整桶pad。图像历史为t-20、t-10、t，30Hz、224×224；机器人7维，policy填充32维；前六维窗口起点相对动作、夹爪绝对值。u_t为正式MVAE视频编码均值，窗口30、维度64，尾29帧掩码。u_t生成后由monitor_training.py自动执行真实PiperDataset与官方历史、SAM2、u_t变换的逐项比较。报告为docs/clockwise_data_contract_20261116.json，完整loader报告另存clockwise_full_loader_contract_20261116.json。

真实原始HDF5已抽查三个batch，均只含puppet观测状态、图像和时间戳。当前action沿用先前已转换数据集的next_observed_state，论文中真实命令的来源语义存在差异，已向用户明确说明。项目支持单臂7维及缺失相机mask。

用户再次要求沿用参考保存策略。已核对save_checkpoint函数与原文件逐字符一致，每100步原子发布可恢复快照，完成后清理上一滚动快照，保留1000至5000各千步。共享优化器保存rank_0.pt，各rank保存独立随机状态。逐千步上传并回读大小和SHA256，辅助模型另行发布，最终补齐日志配置。MVAE和decoder保持best/last及优化器状态。

训练已启动，MVAE W&B为https://wandb.ai/gongcj-beijing/tempo-piperx-gpu-server/runs/82p3m55x。当前进度应读取服务器状态，禁止沿用此段历史轮数推断实时进度。

## 正式数据加载验证通过

MVAE100轮完成，正式u_t生成完毕，decoder80轮训练已启动。monitor_training.py已在真实PiperDataset上验证2023个边界样本，与项目官方LoadActionHistory、LoadSam2Tokens、LoadUtTargets逐元素一致，同时独立重算前六关节相对化、夹爪绝对值、分位归一化、32维补零、图像t-20/t-10/t与缺失相机mask，全部通过。完整报告docs/clockwise_full_loader_contract_20261116.json。decoder W&B为https://wandb.ai/gongcj-beijing/tempo-piperx-gpu-server/runs/hrxc05xt。


## Policy 正式训练与首个检查点

Policy 已在 GPU3、4、6、7 启动，W&B 为 https://wandb.ai/gongcj-beijing/tempo-piperx-gpu-server/runs/h4cwvft1。

micro32、global128、累积1，稳态约6.6秒/步，每卡峰值已分配61.80GiB。首个step100检查点已实际读取验证，837个safetensors权重张量、830份优化器参数状态，rank0含共享优化器，四rank均含独立随机状态，batch128、world_size4核对通过。报告为 docs/clockwise_first_checkpoint_verification_20261116.json。

辅助模型15个文件已于2026-11-16 01:46:44完成ModelScope上传和校验，前缀为 gpu/gpu_clockwise_b234_chunk30_4gpu_20261116/auxiliary，manifest SHA256 为 af10124729678aa1d67ca728419138c520e2b4f3495419b7b42d70f74d73cdb5。

监督进程自动巡检、重试与续训，monitor_training.py 观察完整训练及发布完成并生成 docs/clockwise_completion_20261116.json。只读监测曾重启以持久化输出，日志为 runtime/clockwise_setup_20261116/monitor.log，训练与上传持续运行。实际进度以服务器status和metrics为准。


## 用户要求结束当前等待并安排部署

用户于2026-11-16明确要求训练正常后结束当前聊天，并设置训练完成后的下载与部署任务。最后检查策略已完成step300，loss为0.42635357，稳态6.565秒/步，step300检查点已原子发布，旧step200滚动快照已清理。监督PID341939、torchrun PID384234仍正常。MVAE与decoder完成，辅助模型上传校验完成。当前聊天只读监测进程已停止，正式训练监督和发布程序继续运行。

已通过Codex定时任务工具创建一次性heartbeat，ID为tempo-step5000，名称为“部署顺时针TEMPO step5000”，状态ACTIVE，计划2026-11-16北京时间11:30执行，COUNT=1。这是根据约6.6秒/步和保存开销选定的部署时间。执行时核验step5000和配套上传完整，尚在正常训练时在该次执行中等待产物就绪。

部署目标为SSH gh4090piper的/data/gcj_data/fast/baseline/tempo_full，已只读核查其trained_model/deployment.json、deployment_rtc_paper.json、scripts/start_service.sh、serve_wired.sh。定时任务将从本实验ModelScope前缀下载step5000权重、模型配置、归一化、训练配置、划分和校验清单，以及本实验decoder。复用现有源码、环境、SAM2与tokenizer，适配default/rtc_paper配置，执行严格加载、预热、健康、握手及录制反馈验证，记录回退方式与交接。服务地址为192.168.50.2:8002。任务详情可通过automation_update view tempo-step5000回读。

自动化已成功创建。一次非必要的提示措辞精简更新被自动审核拒绝，原已创建任务及其配置继续有效，因此保留原任务内容。

## 动作表示按论文及官方项目核对

用户再次要求动作表示参考论文。已核查论文图2的robot joint action、TEMPO-ACT的proprioceptive command定义，以及官方dd8342a源码。论文正文未详细规定全部绝对/相对预处理，具体转换以该版本项目配置和变换为依据。

官方src/openpi/training/config.py中的LeRobotYamDataConfig默认use_delta_joint_actions=True，单臂采用make_bool_mask(6,-1)，输入DeltaActions、输出AbsoluteActions。src/openpi/transforms.py的DeltaActions把整个动作窗口的前六关节统一减去当前state，夹爪保持绝对值。官方LoadActionHistory使用原始action的历史桶均值。

已读取GPU-Server实际源码与归一化清单，当前训练采用同样的关节空间约定。state为绝对关节角和绝对夹爪开口，原始action为绝对关节目标，policy/decoder动作空间为相对窗口起点的六关节加绝对夹爪。MVAE内部对原始绝对动作进行z-score处理，decoder使用与policy一致的相对动作及分位归一化。部署时恢复为绝对关节目标。后续部署须继续对齐这一表示与本实验归一化。

本次核对确认当前绝对/相对及关节空间处理与官方对应配置对齐，训练继续执行现有配置。源数据使用下一实测状态代理命令的限制继续记录。

核对来源为https://arxiv.org/html/2609.16864v1、https://github.com/tempo-robot/TEMPO/blob/dd8342a1234e7166b856ffec2d7917b51cce05e4/src/openpi/training/config.py#L430，以及同提交src/openpi/transforms.py和src/openpi/policies/yam_dual_policy.py。

## 完整训练数据补充上传

用户要求构建的训练数据一并上传。自动审批首次要求明确载荷和外部目的地，用户随后明确确认按完整范围上传。授权范围包含约10.4GiB的实际训练输入、RGB/状态/动作缓存、SAM2特征、u_t标签，以及归一化、划分、构建脚本和验证报告。目标为taiyigong3333/tempo下gpu/gpu_clockwise_b234_chunk30_4gpu_20261116/training_data。

上传程序runtime/clockwise_setup_20261116/publish_training_data.py已通过语法检查，在tmux clockwise_training_data_publish_20261116执行。使用低CPU优先级，四组数据分别无损打包为source_dataset.tar.gz、rgb.tar.gz、sam2.tar.gz、ut.tar.gz。原始输入共2901文件，其中source_dataset为877，RGB为1156，SAM2为289，u_t为579。input_manifest.json记录每个内部文件的路径、大小与SHA256，每个压缩包均流式读取并逐文件校验。manifest.json记录实际上传文件的大小与SHA256，提交后强制回读逐文件大小及SHA256匹配。

状态路径为runtime/gpu_clockwise_b234_chunk30_4gpu_20261116/training_data_publish/latest_status.json，日志runtime/clockwise_setup_20261116/training_data_publish.log，退出码training_data_publish.exit。完成后生成docs/clockwise_training_data_upload_20261116.json。当前以状态文件为准，尚待完成上传核验。

## 完整训练数据上传完成

2026-11-16 03:06:47北京时间上传完成，退出码0。taiyigong3333/tempo的gpu/gpu_clockwise_b234_chunk30_4gpu_20261116/training_data目录共332个文件，3543668574字节，约3.30GiB，逐文件远端大小和SHA256全部回读匹配。四个无损归档涵盖2901个原始输入文件、11162259644字节，约10.40GiB；全部归档内部文件已逐个读取核对大小和SHA256。

manifest SHA256为38bd609d451421e344da1a419f063b05b96fea44990673feee79809da9bede1e。归档分别为source_dataset.tar.gz、rgb.tar.gz、sam2.tar.gz、ut.tar.gz，另附归一化、划分、构建与训练工具、验证报告、逐文件输入清单及恢复说明。原始转换数据在同仓库原前缀下也继续可用。

最终验证记录为docs/clockwise_training_data_upload_20261116.json及runtime/gpu_clockwise_b234_chunk30_4gpu_20261116/training_data_publish/latest_status.json，本地已保存对应验证结果和完整上传manifest。此项用户要求已完成。


# 顺时针 batch2、3、4 模型部署

更新于2026年11月16日，北京时间。

## 训练与来源

实验为 `gpu_clockwise_b234_chunk30_4gpu_20261116`，训练于11:22:30完成5000步，最后训练loss为0.00951744。最终检查点于11:27:05完成上传，实验汇总于11:27:14上传完成。训练使用289条轨迹、30285帧，全部进入训练集合。ModelScope来源为 `taiyigong3333/tempo` 下的 `gpu/gpu_clockwise_b234_chunk30_4gpu_20261116`。

策略权重SHA256为 `033e62e6160c2642bbec0683b130e048814ae68157266c4231bf8fde112923af`。配套decoder SHA256为 `0d7e280260da90cd20c5d8cbdf04f61369a012fa07a01eae8612ff0c34351222`。

匿名下载返回404后，使用GPU-Server现有认证环境从ModelScope核验下载。中转链路较慢，最终使用仅限具体模型文件的短时链接，在4090直连并通过四路范围请求续传。账号凭据保存在原服务器，短时链接经加密标准输入传递。4090接收后独立复核全部12个文件大小和SHA256。

## 当前部署

- 目标为SSH `gh4090piper` 的 `/data/gcj_data/fast/baseline/tempo_full`。
- 文件位于 `trained_model/tempo_deploy/gpu/gpu_clockwise_b234_chunk30_4gpu_20261116`。
- `trained_model/deployment.json` 与 `deployment_rtc_paper.json` 均已适配新策略及其decoder。
- 正式服务为 `192.168.50.2:8002`，tmux会话为 `tempo_full_inference`，当前启用 `rtc_paper`。
- RTC采用5次去噪、beta5、最近10次延迟最大值估计，30Hz、30步、7维输出。
- 沿用现有公共模型名称 `pi05_piperx_tempo_full_chunk30` 和GUI入口。实际握手的checkpoint路径已核验为本次新实验。
- 下载的model_config保留发布时内容，运行入口将decoder路径映射为4090上的配套文件。

## 输入输出表示

双相机RGB为224×224，图像历史为t−20、t−10、t。当前state为六个绝对关节角和绝对夹爪开口。历史取300帧实测反馈，补零后聚合为10个30帧桶。策略使用64维u_t，decoder输出窗口起点相对的六关节动作和绝对夹爪，反归一化后恢复绝对关节目标，最终输出float32[30,7]。

本批训练action采用下一实测状态代理，原始采集缺少下发命令。真实任务效果待现场验收。

## 实际验证

- 现有全量部署11项回归通过。
- 两个部署配置文件检查通过，权重包含837个张量。
- 8003候选与8002正式服务均完成严格权重加载，以及原生和RTC预热。
- PiperX有线客户端在两个端口分别完成三模式9批、36帧录制反馈验证，输出均为有限float32[30,7]。
- 正式8002三模式往返P50分别约244.6、245.2、306.0毫秒，RTC P95约316.6毫秒。
- 正式服务健康接口、模型握手及推理会话预约和释放通过。

连续RTC模拟调度的两次60秒目标测试均提前停止。第一次完成2批、6帧和14个规划目标，第二次完成1批、1帧和8个规划目标，两次错误均为“异步动作缓冲耗尽或观测过期”。安全违规与真实机器人指令均为0。失败发生时同一GPU的其他训练进程占用约36900MiB，本服务占用约8764MiB。当前0.5秒时效阈值下的连续运行仍需复测。

报告位于4090的 `trained_model/runtime/gpu_clockwise_b234_chunk30_4gpu_20261116/deployment_verification.json`，客户端原始日志位于 `client/logs/tempo_full/clockwise_b234_20261116`。两次停止证据均已保留。

## 启动

4090服务器使用以下入口。

```bash
cd /data/gcj_data/fast/baseline/tempo_full
bash trained_model/scripts/start_service.sh rtc_paper
```

PiperX使用以下图形入口。

```bash
cd /home/cv/gcj/project/baseline_fast/tempo-lora
bash run_gui_full_rtc_paper.sh
```

## 回退

原两个配置完整备份在 `trained_model/runtime/gpu_clockwise_b234_chunk30_4gpu_20261116/before`。原模型资产保存在原实验目录。需要回退时，在4090恢复两个配置，再使用原启动入口重启。

```bash
cd /data/gcj_data/fast/baseline/tempo_full
cp trained_model/runtime/gpu_clockwise_b234_chunk30_4gpu_20261116/before/deployment.json trained_model/deployment.json
cp trained_model/runtime/gpu_clockwise_b234_chunk30_4gpu_20261116/before/deployment_rtc_paper.json trained_model/deployment_rtc_paper.json
bash trained_model/scripts/start_service.sh rtc_paper --restart
```

## 回退

原两个配置完整备份在 `trained_model/runtime/gpu_clockwise_b234_chunk30_4gpu_20261116/before`。原模型资产保存在原实验目录。需要回退时，在4090恢复两个配置，再使用原启动入口重启。

```bash
cd /data/gcj_data/fast/baseline/tempo_full
cp trained_model/runtime/gpu_clockwise_b234_chunk30_4gpu_20261116/before/deployment.json trained_model/deployment.json
cp trained_model/runtime/gpu_clockwise_b234_chunk30_4gpu_20261116/before/deployment_rtc_paper.json trained_model/deployment_rtc_paper.json
bash trained_model/scripts/start_service.sh rtc_paper --restart
```


## 2026-11-18 已上传训练权重清理

GPU-Server项目/data-minio/taiyi/tempo已按用户要求清理上传成功的策略权重及发布/下载副本，共21个路径。删除前回读ModelScope并核对大小与SHA256。基座、数据、缓存、配置、环境、MVAE/decoder、u_t及优化器/随机续训状态保留，训练依赖清单核验一致。本地策略续训前需按各policy/local_weights_status.json补回对应model.safetensors。清理前146.52GiB，清理后97.15GiB，实际减少49.38GiB。记录为/data-minio/taiyi/tempo/docs/cleanup_weights_20261118.json。


## 2026-11-18 按从base重训范围继续清理

用户明确只需保留从base重新训练的条件。已清理上传且SHA256核验通过的优化器、随机状态和辅助last快照；保留预训练base、tokenizer、SAM2、训练数据/缓存、u_t、归一化/划分配置、代码环境及MVAE/decoder best权重。当前顺时针配置的从base初始化依赖检查通过，未启动新训练。本轮删除44个路径，占用从97.15GiB降至32.81GiB。报告为/data-minio/taiyi/tempo/docs/cleanup_base_only_20261118.json。


## 2026-11-18 history90八卡诊断暂停并迁移ma800

八卡诊断因ZeroRedundancyOptimizer不支持FP32/BF16混合参数而失败，完成0步。诊断进程已退出，正式训练未启动。用户改为ma800三卡，GPU-Server本实验保持暂停，禁止自动重试。日志runtime/history90_setup_20261118/preflight.log，状态paused_status.json。
