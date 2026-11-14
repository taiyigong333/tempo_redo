"""tmux内等待两卡独占并执行正式训练，持续维护状态与定时巡检日志。"""
import argparse
import csv
from datetime import datetime, timedelta
import fcntl
import json
import logging
import math
import os
import psutil
from pathlib import Path
import signal
import subprocess
import time

from runtime_status import timestamp, write_json
from wandb_proxy import reachable, start_proxy


def gpu_snapshot(indices):
    query = subprocess.check_output([
        'nvidia-smi', '--query-gpu=index,uuid,memory.used,memory.total,utilization.gpu',
        '--format=csv,noheader,nounits'], text=True, timeout=20)
    devices = {}
    for row in csv.reader(query.splitlines()):
        index, uuid, used, total, utilization = [value.strip() for value in row]
        devices[int(index)] = dict(index=int(index), uuid=uuid, memory_used_mib=int(used),
                                   memory_total_mib=int(total), utilization_percent=int(utilization), processes=[])
    query = subprocess.check_output([
        'nvidia-smi', '--query-compute-apps=gpu_uuid,pid,used_memory',
        '--format=csv,noheader,nounits'], text=True, timeout=20)
    by_uuid = {device['uuid']: device for device in devices.values()}
    for row in csv.reader(query.splitlines()):
        uuid, pid, memory = [value.strip() for value in row]
        if uuid in by_uuid:
            by_uuid[uuid]['processes'].append({'pid': int(pid), 'memory_mib': int(memory) if memory.isdigit() else None})
    return [devices[index] for index in indices]


def cards_available(snapshot, max_idle_mib, minimum_free_mib=None):
    # 共享模式按训练所需显存准入，GPU利用率瞬时值无法代表可分得的算力。
    if minimum_free_mib is not None:
        return all(gpu['memory_total_mib'] - gpu['memory_used_mib'] >= minimum_free_mib
                   for gpu in snapshot)
    return all(not gpu['processes'] and gpu['memory_used_mib'] < max_idle_mib
               and gpu['utilization_percent'] <= 5 for gpu in snapshot)


class Pipeline:
    def __init__(self, path):
        self.config_path = path.resolve()
        self.config = json.loads(path.read_text())
        self.project = Path(self.config['project'])
        self.root = Path(self.config['experiment_root'])
        self.repo = self.project/'code/TEMPO'
        self.runtime = Path(self.config['runtime_directory'])
        self.logs = self.root/'logs'
        self.runtime.mkdir(parents=True, exist_ok=True)
        self.logs.mkdir(parents=True, exist_ok=True)
        self.lock = (self.runtime/'pipeline.lock').open('w')
        fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        self.lock.write(str(os.getpid()))
        self.lock.flush()
        self.child = None
        self.proxy_process = None
        self.proxy = self.config['wandb']['fallback_proxy']
        self.stage = 'initializing'
        self.state = 'initializing'
        self.last_health = time.monotonic()
        self.micro = self.config['policy']['micro_batch']
        self.env = os.environ.copy()
        self.env.update(CUDA_VISIBLE_DEVICES=','.join(map(str, self.config['gpus'])),
                        NPROC_PER_NODE=str(len(self.config['gpus'])), OMP_NUM_THREADS='4',
                        JAX_PLATFORMS='cpu', NO_PROXY='localhost,127.0.0.1',
                        PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True')
        self.publish()

    def publish(self, **details):
        progress = {}
        candidate = self.root/self.stage/'latest_status.json'
        if candidate.is_file():
            progress = json.loads(candidate.read_text())
        payload = dict(experiment=self.root.name, experiment_root=str(self.root), state=self.state,
                       stage=self.stage, updated_at=timestamp(), controller_pid=os.getpid(),
                       child_pid=self.child.pid if self.child and self.child.poll() is None else None,
                       tmux_session=self.config['tmux_session'], gpu_indices=self.config['gpus'],
                       gpu_uuids=self.config['gpu_uuids'], stage_progress=progress,
                       wandb_proxy=self.proxy, proxy_pid=self.proxy_process.pid if self.proxy_process else None,
                       status_interval_steps=100, **details)
        for path in (self.root/'status.json', self.runtime/'latest_status.json', self.project/'runtime/latest_status.json'):
            write_json(path, payload)

    def stop_child(self):
        if self.child and self.child.poll() is None:
            os.killpg(self.child.pid, signal.SIGTERM)
            try:
                self.child.wait(timeout=60)
            except subprocess.TimeoutExpired:
                os.killpg(self.child.pid, signal.SIGKILL)
                self.child.wait(timeout=15)

    def health_check(self, force=False):
        if not force and time.monotonic()-self.last_health < self.config['health_interval_seconds']:
            return
        self.last_health = time.monotonic()
        metrics = self.root/self.stage/'metrics.jsonl'
        if not metrics.exists():
            metrics = self.logs/f'{self.stage}.log'
        age = time.time()-metrics.stat().st_mtime if metrics.exists() else None
        record = dict(checked_at=timestamp(), state=self.state, stage=self.stage,
                      child_alive=bool(self.child and self.child.poll() is None),
                      progress_age_seconds=age, gpus=gpu_snapshot(self.config['gpus']))
        owned = set()
        if record['child_alive']:
            try:
                parent = psutil.Process(self.child.pid)
                owned = {parent.pid, *(process.pid for process in parent.children(recursive=True))}
            except psutil.NoSuchProcess:
                pass
        record['external_gpu_pids'] = [process['pid'] for gpu in record['gpus']
                                       for process in gpu['processes'] if process['pid'] not in owned]
        record['metrics_finite'] = True
        if metrics.suffix == '.jsonl' and metrics.is_file():
            lines = metrics.read_text().splitlines()
            if lines:
                row = json.loads(lines[-1])
                record['metrics_finite'] = all(math.isfinite(float(row[key])) for key in ('loss', 'train_loss', 'monitor_loss')
                                               if row.get(key) is not None)
        stalled = record['child_alive'] and age is not None and age > self.config['stall_timeout_seconds']
        record['healthy'] = not stalled and record['metrics_finite']
        with (self.runtime/'health_checks.jsonl').open('a') as stream:
            stream.write(json.dumps(record, ensure_ascii=False)+'\n')
        logging.info('半小时巡检 %s', json.dumps(record, ensure_ascii=False))
        if not record['healthy']:
            logging.error('巡检发现训练停滞或非有限指标，停止本实验子进程并从完整检查点恢复')
            self.stop_child()

    def wait(self, seconds):
        deadline = time.monotonic()+seconds
        while time.monotonic() < deadline:
            time.sleep(min(5, max(0, deadline-time.monotonic())))
            self.health_check()

    def wait_for_resources(self):
        while True:
            snapshot = gpu_snapshot(self.config['gpus'])
            if [gpu['uuid'] for gpu in snapshot] != self.config['gpu_uuids']:
                raise RuntimeError('GPU编号与已授权UUID对应关系变化')
            now = datetime.fromisoformat(timestamp())
            next_probe = (now+timedelta(seconds=self.config['gpu_probe_interval_seconds'])).isoformat()
            if not cards_available(snapshot, self.config['max_idle_memory_mib'],
                                   self.config.get('minimum_free_memory_mib')):
                self.state = 'waiting_for_gpu'
                self.publish(gpus=snapshot, next_probe_at=next_probe)
                logging.info('等待训练所需GPU资源，下次检查 %s，当前 %s', next_probe, json.dumps(snapshot))
                self.wait(self.config['gpu_probe_interval_seconds'])
                continue
            if not reachable(self.proxy):
                self.state = 'waiting_for_wandb'
                self.publish(gpus=snapshot, next_probe_at=next_probe)
                logging.warning('W&B出口暂不可达，10分钟后重试')
                self.wait(self.config['gpu_probe_interval_seconds'])
                continue
            return

    def command(self, stage):
        run = ['bash', 'tools/piperx/run.sh', stage]
        shared = ['--split-path', self.config['split_path'], '--norm-dir', self.config['norm_dir']]
        if stage in ('mvae', 'decoder'):
            command = run+['--output', str(self.root/stage), '--status-interval', '100']+shared
            for option, flag in (('resident_data', '--resident-data'), ('fused_adamw', '--fused-adamw'), ('compile_forward', '--compile-forward')):
                if self.config.get('aux_execution', {}).get(option):
                    command.append(flag)
            if stage == 'decoder':
                command += ['--ut-dir', str(self.root/'ut')]
            if (self.root/stage/'last.pt').is_file():
                command += ['--resume', str(self.root/stage/'last.pt')]
            return command
        if stage == 'encode':
            return run+['--mvae', str(self.root/'mvae/best.pt'), '--output', str(self.root/'ut')]
        command = run+['--mode', 'formal', '--ut-dir', str(self.root/'ut'),
                       '--decoder', str(self.root/'decoder/best.pt'), '--output', str(self.root/'policy'),
                       '--steps', str(self.config['policy']['steps']), '--batch', '128',
                       '--micro-batch', str(self.micro), '--workers', str(self.config['policy'].get('workers', 2)),
                       '--save-interval', str(self.config['policy'].get('save_interval', 2000)),
                       '--latest-save-interval', str(self.config['policy'].get('latest_save_interval', 100)),
                       '--status-interval', '100', '--eval-interval', '0']+shared
        latest = self.root/'policy/latest.json'
        if latest.is_file():
            pointer = json.loads(latest.read_text())
            command += ['--resume', pointer['path']]
        return command

    def complete(self, stage):
        if stage == 'encode':
            metadata = json.loads((self.root/'ut/manifest.json').read_text())
            return metadata['episodes'] == 393 and metadata['window'] == 30 and not metadata['test_only']
        if stage in ('mvae', 'decoder'):
            return all((self.root/stage/file).is_file() for file in ('best.pt', 'last.pt')) and json.loads(
                (self.root/stage/'latest_status.json').read_text())['state'] == 'completed'
        pointer = json.loads((self.root/'policy/latest.json').read_text())
        checkpoint = Path(pointer['path'])
        required = ['model.safetensors', 'model_config.json', 'norm_stats.json', 'split.json', 'rank_0.pt', 'rank_1.pt']
        return pointer['step'] == self.config['policy']['steps'] and all((checkpoint/name).is_file() for name in required)

    def recover_outputs(self, stage):
        output = self.root/stage
        checkpoint = output/'latest.json'
        if stage in ('mvae', 'decoder'):
            resumable = (output/'last.pt').is_file()
        else:
            resumable = checkpoint.is_file()
        if stage != 'encode' and output.exists() and (output/'metrics.jsonl').exists() and not resumable:
            archive = self.root/'interrupted_attempts'
            archive.mkdir(exist_ok=True)
            output.rename(archive/f'{stage}_{time.time_ns()}')
        if stage == 'policy' and output.exists():
            recovery = output/'recovery'
            for incomplete in output.glob('.step_*.incomplete'):
                recovery.mkdir(exist_ok=True)
                incomplete.rename(recovery/f'{time.time_ns()}_{incomplete.name}')
            if checkpoint.is_file() and (output/'metrics.jsonl').is_file():
                saved_step = json.loads(checkpoint.read_text())['step']
                records = (output/'metrics.jsonl').read_text().splitlines()
                kept = [line for line in records if json.loads(line)['step'] <= saved_step]
                if len(kept) != len(records):
                    recovery.mkdir(exist_ok=True)
                    (output/'metrics.jsonl').rename(recovery/f'metrics_{time.time_ns()}.jsonl')
                    (output/'metrics.jsonl').write_text('\n'.join(kept)+'\n')
                    # W&B历史不可回退，重放检查点后的更新使用新run保留清晰来源。
                    identity = output/'wandb_id.txt'
                    if identity.is_file():
                        identity.rename(recovery/f'wandb_id_{time.time_ns()}.txt')

    def run_stage(self, stage):
        self.stage = stage
        marker = self.runtime/f'{stage}.complete.json'
        if marker.is_file() and self.complete(stage):
            return
        for attempt in range(1, self.config['max_stage_attempts']+1):
            self.wait_for_resources()
            self.recover_outputs(stage)
            command = self.command(stage)
            self.state = 'running'
            self.publish(attempt=attempt, command=command)
            logging.info('启动阶段 %s，第%d次，命令 %s', stage, attempt, command)
            log_path = self.logs/f'{stage}.log'
            with log_path.open('a') as log:
                log.write(f'\n[{timestamp()}] attempt={attempt}\n')
                log.flush()
                self.child = subprocess.Popen(command, cwd=self.repo, env=self.env,
                                              stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                while self.child.poll() is None:
                    time.sleep(5)
                    self.health_check()
                    self.publish(attempt=attempt)
            code = self.child.returncode
            self.child = None
            if code == 0 and self.complete(stage):
                write_json(marker, {'completed_at': timestamp(), 'stage': stage})
                logging.info('阶段 %s 完成', stage)
                return
            text = log_path.read_text()[-30000:].lower()
            if stage == 'policy' and ('out of memory' in text or 'outofmemoryerror' in text) and self.micro > 4:
                self.micro //= 2
                logging.warning('显存不足，micro-batch调整为%d，全局batch128', self.micro)
            retry_delay = self.config['gpu_probe_interval_seconds']
            if stage in ('mvae', 'decoder') and self.config.get('aux_execution', {}).get('compile_forward') and any(
                    term in text for term in ('dynamo', 'inductor', 'cudagraph', 'compile_fx')):
                self.config['aux_execution']['compile_forward'] = False
                retry_delay = 5
                logging.warning('编译后端异常，恢复普通forward并保留GPU驻留缓存和融合优化器')
            self.state = 'recovering'
            self.publish(attempt=attempt, exit_code=code, micro_batch=self.micro)
            logging.error('阶段 %s 退出码 %s，10分钟后检查并恢复', stage, code)
            self.wait(retry_delay)
        raise RuntimeError(f'{stage}连续失败，请查看阶段日志')

    def run(self):
        try:
            try:
                self.proxy_process, self.proxy = start_proxy(self.config['wandb']['native_proxy'])
                logging.info('服务器W&B专用出口已就绪 %s', self.proxy)
            except Exception as error:
                logging.warning('专用出口初始化结果 %s，采用已授权备用出口', type(error).__name__)
            self.env.update(WANDB_PROXY=self.proxy, HTTP_PROXY=self.proxy, HTTPS_PROXY=self.proxy)
            self.health_check(force=True)
            for stage in ('mvae', 'encode', 'decoder', 'policy'):
                self.run_stage(stage)
            self.stage = 'completed'
            self.state = 'completed'
            self.publish(completed_at=timestamp(), final_checkpoint=str(self.root/'policy/step_00005000'))
            logging.info('正式实验完成，日志已保留，tmux会话将自动退出')
        except BaseException as error:
            self.state = 'failed'
            self.publish(error_type=type(error).__name__, error=str(error))
            logging.exception('实验监督进程退出')
            raise
        finally:
            self.stop_child()
            if self.proxy_process and self.proxy_process.poll() is None:
                self.proxy_process.terminate()
                try:
                    self.proxy_process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    self.proxy_process.kill()
                    self.proxy_process.wait(timeout=5)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', type=Path, required=True)
    args = ap.parse_args()
    def interrupted(signum, frame):
        raise InterruptedError(f'监督进程收到信号{signum}')
    signal.signal(signal.SIGTERM, interrupted)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    Pipeline(args.config).run()


if __name__ == '__main__':
    main()
