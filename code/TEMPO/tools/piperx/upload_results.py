"""各检查点完整发布后立即上传，训练结束后补齐日志与实验元数据。"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import fcntl
import hashlib
import json
import logging
import os
from pathlib import Path
import re
import shutil
import time
from zoneinfo import ZoneInfo


def timestamp():
    return datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(timespec='seconds')


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2))
    temporary.replace(path)


def digest(path):
    checksum = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            checksum.update(block)
    return checksum.hexdigest()


def checkpoint_steps(config):
    # 保存与上传共享配置，并始终包含最终步，避免更改间隔后漏传。
    policy = config['policy']
    total, interval = int(policy['steps']), int(policy.get('save_interval', 2000))
    if policy.get('save_steps') is not None:
        return tuple(sorted(set(map(int, policy['save_steps'])) | {total}))
    if total <= 0 or interval <= 0:
        raise ValueError('训练步数和保存间隔必须大于0')
    return tuple(range(interval, total, interval)) + (total,)


class Publisher:
    def __init__(self, config_path, repo_id, token_file, checkpoint_step=None, auxiliary_only=False):
        self.config_path = config_path
        self.config = json.loads(config_path.read_text())
        self.project = Path(self.config['project'])
        self.root = Path(self.config['experiment_root'])
        self.runtime = Path(self.config['runtime_directory'])
        self.work = self.runtime / 'modelscope_publish'
        self.step = checkpoint_step
        self.auxiliary_only = auxiliary_only
        if auxiliary_only:
            self.work /= 'auxiliary'
        if self.step is not None:
            self.work /= f'step_{self.step:08d}'
        self.work.mkdir(parents=True, exist_ok=True)
        self.lock = (self.work / 'publish.lock').open('w')
        fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        self.repo_id = repo_id
        self.token_file = token_file
        self.bundle = self.work / 'bundle'
        self.prefix = self.config.get('modelscope', {}).get('path_prefix', 'runs/' + self.root.name).strip('/')
        if auxiliary_only:
            self.prefix += '/auxiliary'
        if self.step is not None:
            self.prefix += f'/policy/step_{self.step:08d}'

    def status(self, state, **details):
        value = dict(state=state, updated_at=timestamp(), repo_id=self.repo_id,
                     path_in_repo=self.prefix, **details)
        write_json(self.work / 'latest_status.json', value)
        logging.info('%s', json.dumps(value, ensure_ascii=False))

    def complete(self):
        # 检查点在两rank写完后原子发布，早期检查点可在后续训练期间上传。
        if self.auxiliary_only:
            return all((self.root / stage / name).is_file()
                       for stage in ('mvae', 'decoder') for name in ('best.pt', 'last.pt'))
        try:
            pointer = json.loads((self.root / 'policy/latest.json').read_text())
            steps = (self.step,) if self.step is not None else checkpoint_steps(self.config)
            if pointer['step'] < max(steps):
                return False
            for step in steps:
                published = self.work / f'step_{step:08d}' / 'latest_status.json'
                if (self.step is None and self.config.get('remove_local_after_upload') and published.exists()
                        and json.loads(published.read_text()).get('state') == 'completed'):
                    continue
                checkpoint = self.root / 'policy' / f'step_{step:08d}'
                for name in ['model.safetensors', 'model_config.json', 'config.json', 'norm_stats.json', 'split.json'] + [
                        f'rank_{rank}.pt' for rank in range(len(self.config['gpus']))]:
                    if not (checkpoint / name).is_file() or (checkpoint / name).stat().st_size == 0:
                        return False
            if self.step is None:
                progress = json.loads((self.root / 'policy/latest_status.json').read_text())
                controller = json.loads((self.root / 'status.json').read_text())
                if progress['state'] != 'completed' or controller['state'] != 'completed':
                    return False
                if not (self.runtime / 'policy.complete.json').is_file():
                    return False
                for step in steps:
                    result = self.work / f'step_{step:08d}' / 'latest_status.json'
                    if json.loads(result.read_text())['state'] != 'completed':
                        return False
            return True
        except (OSError, KeyError, json.JSONDecodeError):
            return False

    def stage(self):
        ready = self.work / 'bundle.ready.json'
        if ready.exists():
            return json.loads((self.bundle / 'manifest.json').read_text())
        self.status('staging')
        self.bundle.mkdir(exist_ok=True)

        def add(source, target, immutable=False):
            if not source.is_file():
                raise FileNotFoundError(source)
            target = self.bundle / target
            target.parent.mkdir(parents=True, exist_ok=True)
            if immutable:
                # 大检查点使用同盘硬链接，避免额外复制约百GB训练状态。
                if not target.exists():
                    os.link(source, target)
            else:
                shutil.copyfile(source, target)

        if self.step is not None:
            checkpoint = self.root / 'policy' / f'step_{self.step:08d}'
            for source in sorted(checkpoint.iterdir()):
                if source.suffix in ('.json', '.pt', '.safetensors'):
                    add(source, source.name, immutable=True)
            return self.manifest(ready)
        if self.auxiliary_only:
            for stage in ('mvae', 'decoder'):
                for name in ('best.pt', 'last.pt', 'model_config.json', 'latest_status.json'):
                    source = self.root / stage / name
                    add(source, Path(stage) / name, immutable=source.suffix == '.pt')
            add(Path(self.config['norm_dir']) / 'norm_stats.json', 'configs/norm_stats.json')
            add(Path(self.config['split_path']), 'configs/split.json')
            add(self.config_path, 'configs/controller.json')
            for relative in ('tools/ut/mvae.py', 'src/openpi/models_pytorch/ut_decoder.py'):
                add(self.project / 'code/TEMPO' / relative, Path('source') / relative)
            add(self.root / 'auxiliary_usage.md', 'README.md')
            return self.manifest(ready)
        for stage in ('mvae', 'decoder'):
            for name in ('best.pt', 'last.pt', 'config.json', 'metrics.jsonl', 'latest_status.json'):
                source = self.root / stage / name
                if source.exists():
                    add(source, source.relative_to(self.root), immutable=source.suffix == '.pt')
        for name in ('latest.json', 'latest_status.json', 'metrics.jsonl', 'config.json'):
            add(self.root / 'policy' / name, Path('policy') / name)
        for source in sorted((self.root / 'logs').glob('*.log')):
            add(source, source.relative_to(self.root))
        for name in ('config.json', 'status.json'):
            add(self.root / name, name)
        add(self.config_path, 'configs/controller.json')
        add(Path(self.config['split_path']), 'configs/split.json')
        add(Path(self.config['norm_dir']) / 'norm_stats.json', 'configs/norm_stats.json')
        add(self.runtime / 'health_checks.jsonl', 'runtime/health_checks.jsonl')
        add(self.root / 'ut/manifest.json', 'ut/manifest.json')
        for name in ('handoff.md', 'tempo_code_changes.md'):
            source = self.project / name
            if source.exists():
                add(source, Path('docs') / name)
        tools = self.project / 'code/TEMPO/tools/piperx'
        for source in sorted(tools.iterdir()):
            if source.suffix in ('.py', '.sh'):
                add(source, Path('source/tools/piperx') / source.name)
        for relative in ('src/openpi/models/pi0_config.py',
                         'src/openpi/models_pytorch/pi0_pytorch.py',
                         'src/openpi/models_pytorch/gemma_pytorch.py'):
            add(self.project / 'code/TEMPO' / relative, Path('source') / relative)
        readme = '\n'.join([
            '# TEMPO PiperX 全量微调结果', '',
            f"本次实验使用{self.config['dataset']['train_episodes']}条训练轨迹、{self.config['dataset']['frames']}帧，预测chunk为{self.config['dataset']['window']}。",
            'π0.5与TEMPO策略全量微调，CUDA '+','.join(map(str,self.config['gpus']))+f"，有效batch{self.config['policy']['global_batch']}，共{self.config['policy']['steps']}个优化器更新。",
            'policy目录保存第' + '、'.join(map(str, checkpoint_steps(self.config)))
            + '步的完整模型、共享优化器状态与各rank、随机状态及归一化配置。',
            'mvae和decoder目录包含辅助模型，configs、logs和manifest.json记录训练配置、指标与文件校验信息。',
            '本次按用户要求使用全部数据训练，模型评测结果尚未生成。', '',
            '训练代码、日志及配置中的服务器路径保留用于追溯，部署时需要按实际环境调整。',
            '恢复训练需要匹配原rank数量及兼容的训练环境。', '',
        ])
        (self.bundle / 'README.md').write_text(readme)
        return self.manifest(ready)

    def manifest(self, ready):
        files = []
        for path in sorted(self.bundle.rglob('*')):
            if not path.is_file() or path == self.bundle / 'manifest.json':
                continue
            if path.suffix in ('.json', '.jsonl', '.log', '.md', '.py', '.sh', '.txt'):
                content = path.read_text()
                if re.search(r'wandb_v1_[A-Za-z0-9_-]+|ms-[0-9a-f]{8}-[0-9a-f-]{27,}', content):
                    raise ValueError('上传材料发现凭据格式，需清理后继续')
            files.append(dict(path=path.relative_to(self.bundle).as_posix(),
                              size=path.stat().st_size, sha256=digest(path)))
        if self.step is None and not self.auxiliary_only:
            # 汇总清单覆盖已分别上传的检查点，最终只传日志和元数据。
            for step in checkpoint_steps(self.config):
                source = self.work / f'step_{step:08d}' / 'bundle/manifest.json'
                checkpoint_manifest = json.loads(source.read_text())
                prefix = f'policy/step_{step:08d}/'
                files.extend(dict(row, path=prefix + row['path'])
                             for row in checkpoint_manifest['files'])
                files.append(dict(path=prefix + 'manifest.json', size=source.stat().st_size,
                                  sha256=digest(source)))
        manifest = dict(experiment=self.root.name, created_at=timestamp(),
                        repository=self.repo_id, path_in_repo=self.prefix, files=files)
        write_json(self.bundle / 'manifest.json', manifest)
        write_json(ready, dict(created_at=timestamp(), files=len(files),
                               bytes=sum(row['size'] for row in files)))
        return manifest

    def upload(self):
        from modelscope.hub.api import HubApi
        api = HubApi()
        api.login(self.token_file.read_text().strip())
        manifest = self.stage()
        self.status('uploading', files=len(manifest['files']))

        def progress(record):
            # 回调只保存计数，避免把远端响应或认证上下文写入公开材料。
            keys = ('event', 'committed_files', 'committed_bytes', 'total_files', 'total_bytes', 'elapsed')
            write_json(self.work / 'progress.json', {key: record[key] for key in keys if key in record})

        api.upload_folder(repo_id=self.repo_id, repo_type='dataset', folder_path=self.bundle,
                          path_in_repo=self.prefix, revision='master', max_workers=4,
                          use_cache=True, disable_tqdm=True, sync_remote_repo=False,
                          tracker_path=self.work / 'upload_cache', progress_callback=progress,
                          commit_message='Upload ' + self.prefix)
        self.status('verifying')
        remote = {row.path.lstrip('/'): row for row in api._api.list_repo_files(
            self.repo_id, 'dataset', recursive=True)}
        for entry in manifest['files'] + [dict(path='manifest.json',
                                               size=(self.bundle / 'manifest.json').stat().st_size,
                                               sha256=digest(self.bundle / 'manifest.json'))]:
            key = self.prefix + '/' + entry['path']
            item = remote.get(key)
            if item is None or item.size != entry['size']:
                raise RuntimeError('远端文件缺失或大小不匹配 ' + key)
            if not item.sha256 or item.sha256.lower() != entry['sha256']:
                raise RuntimeError('远端SHA256不匹配 ' + key)
        self.status('completed', verified_files=len(manifest['files']) + 1,
                    dataset_url='https://www.modelscope.cn/datasets/' + self.repo_id,
                    manifest_sha256=digest(self.bundle / 'manifest.json'))

    def cleanup_uploaded_checkpoint(self):
        if self.step is None or not self.config.get('remove_local_after_upload'):
            return
        manifest = json.loads((self.bundle/'manifest.json').read_text())
        status_path = self.work/'latest_status.json'
        result = json.loads(status_path.read_text())
        if result.get('local_checkpoint_removed'):
            return
        assert result['state'] == 'completed'
        # 节点同时是最新checkpoint时，先等下一滚动快照接管，或等待训练结束。
        while True:
            pointer = json.loads((self.root/'policy/latest.json').read_text())
            progress = json.loads((self.root/'policy/latest_status.json').read_text())
            if pointer['step'] > self.step or progress['state'] in ('completed', 'failed'):
                break
            time.sleep(10)
        checkpoint = self.root/'policy'/f'step_{self.step:08d}'
        candidates = []
        for row in manifest['files']:
            relative = Path(row['path'])
            assert not relative.is_absolute() and '..' not in relative.parts
            if relative.suffix not in ('.pt', '.safetensors'):
                continue
            for root in (checkpoint, self.bundle):
                path = root/relative
                if path.exists():
                    assert not path.is_symlink() and path.stat().st_size == row['size']
                    candidates.append(path)
        # 所有文件已由upload逐个验真，完成后解除训练目录和硬链接发布副本。
        for path in candidates:
            path.unlink()
        result.update(local_checkpoint_removed=True, local_removed_at=timestamp(),
                      removed_weight_paths=[str(path) for path in candidates])
        write_json(status_path, result)
        if pointer['step'] == self.step:
            pointer.update(local_available=False, modelscope_path=self.prefix, repository=self.repo_id)
            write_json(self.root/'policy/latest.json', pointer)

    def run(self, check_only=False):
        if check_only:
            from modelscope.hub.api import HubApi
            api = HubApi()
            api.login(self.token_file.read_text().strip())
            api._api.get_repo(self.repo_id, 'dataset')
            print(json.dumps(dict(authentication='ok', dataset_access='ok',
                                  training_complete=self.complete()), ensure_ascii=False))
            return
        result = self.work / 'latest_status.json'
        if result.exists() and json.loads(result.read_text()).get('state') == 'completed':
            self.cleanup_uploaded_checkpoint()
            return
        self.status('waiting_for_checkpoint' if self.step is not None else 'waiting_for_training')
        while not self.complete():
            time.sleep(60)
        while True:
            try:
                current = json.loads(result.read_text()) if result.exists() else {}
                if current.get('state') != 'completed':
                    self.upload()
                self.cleanup_uploaded_checkpoint()
                return
            except Exception as error:
                # 上传与网络重试独立于训练进程，训练状态始终保留在GPU。
                delay = int(self.config.get('upload_retry_seconds', 1800))
                current = json.loads(result.read_text()) if result.exists() else {}
                if current.get('state') == 'completed':
                    current.update(cleanup_error_type=type(error).__name__, cleanup_retry_after_seconds=delay)
                    write_json(result, current)
                else:
                    self.status('retrying', error_type=type(error).__name__, retry_after_seconds=delay)
                logging.error('发布或清理尚未完成，异常类型 %s，%d秒后重试', type(error).__name__, delay)
                deadline = time.monotonic() + delay
                while time.monotonic() < deadline:
                    time.sleep(min(60, deadline-time.monotonic()))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--repo-id', default='taiyigong3333/tempo')
    parser.add_argument('--token-file', type=Path, default=Path.home() / '.cache/tempo_modelscope/token')
    parser.add_argument('--check-only', action='store_true')
    parser.add_argument('--auxiliary-only', action='store_true')
    args = parser.parse_args()
    # W&B专用出口限制了域名，ModelScope上传使用国内直连。
    for name in ('HTTP_PROXY', 'HTTPS_PROXY', 'ALL_PROXY', 'http_proxy', 'https_proxy', 'all_proxy'):
        os.environ.pop(name, None)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    if args.auxiliary_only:
        Publisher(args.config, args.repo_id, args.token_file, auxiliary_only=True).run(check_only=args.check_only)
        return
    if args.check_only:
        Publisher(args.config, args.repo_id, args.token_file).run(check_only=True)
        return
    coordinator = Publisher(args.config, args.repo_id, args.token_file)
    coordinator.status('waiting_for_checkpoints')
    # 每个检查点有独立锁、状态和续传缓存，慢上传不会阻塞后续检查点开始。
    steps = checkpoint_steps(coordinator.config)
    with ThreadPoolExecutor(max_workers=len(steps)) as executor:
        workers = [Publisher(args.config, args.repo_id, args.token_file, step)
                   for step in steps]
        futures = [executor.submit(worker.run) for worker in workers]
        for future in futures:
            future.result()
    coordinator.run()


if __name__ == '__main__':
    main()
