from pathlib import Path
import json

root = Path('/data-minio/taiyi/tempo')
repo = root / 'code/TEMPO'
p = repo / 'tools/piperx/train.py'
s = p.read_text()
s = s.replace("os.environ['PALIGEMMA_TOKENIZER_PATH'] = str(project.parent / 'base_model/pi0.5_base/paligemma_tokenizer.model')", "# 允许服务器配置提供分词器，兼容旧项目目录的默认位置。\n    os.environ.setdefault('PALIGEMMA_TOKENIZER_PATH', str(project.parent / 'base_model/pi0.5_base/paligemma_tokenizer.model'))")
s = s.replace("project='tempo-piperx-ma800'", "project=os.environ.get('WANDB_PROJECT', 'tempo-piperx-ma800')")
s = s.replace("sample = default_collate([validation[50]])", "sample = default_collate([(validation if validation is not None else dataset)[50]])")
p.write_text(s)
p = repo / 'tools/piperx/train_aux.py'
s = p.read_text().replace("project='tempo-piperx-ma800'", "project=os.environ.get('WANDB_PROJECT', 'tempo-piperx-ma800')")
p.write_text(s)
p = repo / 'tools/piperx/pipeline.py'
s = p.read_text()
s = s.replace("self.env.update(CUDA_VISIBLE_DEVICES=", "self.env.update(WANDB_ENTITY=self.config['wandb']['entity'],\n                        WANDB_PROJECT=self.config['wandb']['project'],\n                        PALIGEMMA_TOKENIZER_PATH=self.config['tokenizer'],\n                        CUDA_VISIBLE_DEVICES=")
s = s.replace("snapshot = gpu_snapshot(self.config['gpus'])\n            if [gpu['uuid']", "# 自动模式只选择没有计算进程的两张卡，并在本轮流水线内保持选择稳定。\n            if not self.config['gpus']:\n                available = [gpu for gpu in gpu_snapshot(range(torch_device_count()))\n                             if cards_available([gpu], self.config['max_idle_memory_mib'])]\n                if len(available) >= 2:\n                    selected = available[:2]\n                    self.config['gpus'] = [gpu['index'] for gpu in selected]\n                    self.config['gpu_uuids'] = [gpu['uuid'] for gpu in selected]\n                    self.env['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, self.config['gpus']))\n                    self.env['NPROC_PER_NODE'] = '2'\n                    write_json(self.runtime/'resolved_config.json', self.config)\n                else:\n                    self.state = 'waiting_for_gpu'\n                    self.publish(available_cards=len(available))\n                    logging.info('等待两张空闲GPU，当前空闲%d张，10分钟后复查', len(available))\n                    self.wait(self.config['gpu_probe_interval_seconds'])\n                    continue\n            snapshot = gpu_snapshot(self.config['gpus'])\n            if [gpu['uuid']")
s = s.replace("return command\n        if stage == 'encode':", "command += ['--epochs', str(self.config[stage]['epochs'])]\n            return command\n        if stage == 'encode':", 1)
s = s.replace("'--steps', str(self.config['policy']['steps']), '--batch', '128',", "'--steps', str(self.config['policy']['steps']), '--batch', str(self.config['policy']['global_batch']),")
s = s.replace("'--save-interval', '2000', '--status-interval', '100', '--eval-interval', '0'", "'--save-interval', str(self.config['policy']['save_interval']), '--status-interval', str(self.config['policy']['status_interval']), '--eval-interval', '0'")
s = s.replace("self.proxy_process, self.proxy = start_proxy(self.config['wandb']['native_proxy'])", "if self.config['wandb'].get('native_proxy'):\n                    self.proxy_process, self.proxy = start_proxy(self.config['wandb']['native_proxy'])")
s = s.replace("class Pipeline:", "def torch_device_count():\n    # 使用驱动枚举，监督进程无需初始化CUDA上下文。\n    return len(subprocess.check_output(['nvidia-smi', '--query-gpu=index', '--format=csv,noheader'], text=True).splitlines())\n\n\nclass Pipeline:")
p.write_text(s)
config = {
    'status': 'ready_after_verification', 'project': str(root),
    'experiment_root': str(root/'experiments/formal_chunk30_2gpu'),
    'runtime_directory': str(root/'runtime/formal_chunk30_2gpu'),
    'tmux_session': 'tempo_chunk30_2gpu', 'gpus': [], 'gpu_uuids': [],
    'gpu_selection': 'first_two_idle', 'gpu_probe_interval_seconds': 600,
    'health_interval_seconds': 1800, 'stall_timeout_seconds': 3600,
    'max_idle_memory_mib': 1024, 'max_stage_attempts': 3,
    'split_path': str(root/'configs/full_data_chunk30/split.json'),
    'norm_dir': str(root/'configs/full_data_chunk30/norm'),
    'jax_checkpoint': '/data-minio/taiyi/base_model/openpi_cache/openpi-assets/checkpoints/pi05_base',
    'tokenizer': '/data-minio/taiyi/base_model/openpi_cache/big_vision/paligemma_tokenizer.model',
    'dataset': {'train_episodes': 393, 'validation_episodes': 0, 'frames': 44398, 'window': 30},
    'policy': {'steps': 5000, 'global_batch': 128, 'micro_batch': 16,
               'accumulation_steps': 4, 'save_interval': 2000, 'status_interval': 100,
               'eval_interval': 0, 'warmup_steps': 500, 'peak_lr': 5e-5, 'end_lr': 5e-6,
               'optimizer': 'GPU-sharded FP32-master AdamW', 'cpu_offload': False,
               'betas': [.9,.95], 'eps': 1e-8, 'weight_decay': 1e-10, 'clip_norm': 1,
               'compact_prefix': False, 'checkpoint_joint_forward': False, 'seed': 42},
    'mvae': {'epochs': 100, 'global_batch': 64, 'latent_dim': 64, 'beta': 0, 'window': 30},
    'decoder': {'epochs': 80, 'global_batch': 64, 'window': 30},
    'wandb': {'entity': 'gongcj-beijing', 'project': 'tempo-piperx-gpu-server',
              'fallback_proxy': 'http://127.0.0.1:9890', 'native_proxy': None},
}
(root/'configs').mkdir(exist_ok=True)
(root/'configs/gpu_server_chunk30.json').write_text(json.dumps(config,ensure_ascii=False,indent=2))
(root/'docs').mkdir(exist_ok=True)
(root/'handoff.md').write_text('''# GPU-Server TEMPO 交接\n\n2026-11-14，配置进行中。项目根为 `/data-minio/taiyi/tempo`，源码位于 `code/TEMPO`。\n\n## 目标与约束\n\n沿用最新参考聊天的全量393轨迹、44398帧，chunk30，PyTorch全量微调，GPU驻留的FP32主权重与AdamW状态。训练5000步，每2000步及最终保存，每100步更新状态。MVAE100轮、decoder80轮，无验证集。\n\n使用本服务器已有JAX权重转换基座。独立uv环境位于源码目录`.venv`，解释器位于源码目录`.tools/python`。\n\n## 训练设置\n\n机器可读配置为 `configs/gpu_server_chunk30.json`。默认启动时自动寻找两张无计算进程的空闲GPU，micro16，累积4，有效batch128。缺少两张空闲GPU时每10分钟等待，tmux监督进程每30分钟检查运行状态。训练日志保存在实验目录logs，状态保存在runtime/latest_status.json。\n\n当前请求为完成配置并达到可启动状态，正式流水线尚未启动。独立短测产物位于experiments/setup_smoke，不用于正式训练。\n\n## 进行中\n\n环境安装、全量缓存准备、JAX转换、W&B验证、最小训练验证。完成后更新实际结果。\n''')
(root/'tempo_code_changes_gpu_server.md').write_text('''# GPU-Server独立改动记录\n\n迁移ma800已验证的TEMPO源码与PiperX工具，源码基线dd8342a1234e7166b856ffec2d7917b51cce05e4。既有改动详见tempo_code_changes.md。\n\n- tools/piperx/train.py 支持环境变量提供分词器路径与W&B项目。无验证集时的动作形状检查读取训练数据。\n- tools/piperx/train_aux.py 支持W&B项目环境变量。\n- tools/piperx/pipeline.py 支持启动时自动选择两张空闲GPU，使用目标服务器现有代理，传递分词器与W&B设置，读取配置中的训练轮次和保存周期。\n- configs/gpu_server_chunk30.json 保存本服务器路径和全量训练设置。\n\n验证结果待完成后补充。\n''')
print('adapted')
