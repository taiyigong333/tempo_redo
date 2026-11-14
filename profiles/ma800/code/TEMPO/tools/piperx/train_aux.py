"""复用官方MVAE/decoder结构的多卡辅助训练，支持独立短测和断点恢复。"""
import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import Dataset, DataLoader
import wandb

from data import ExactBatchSampler
from runtime_status import stage_status
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'ut'))
from mvae import MVAEConfig, PoEMVAE, ActionDecoder, VideoLatentDecoder
from openpi.shared import normalize
from openpi.transforms import Normalize


class AuxDataset(Dataset):
    def __init__(self, project, episodes, ut_dir=None, norm_dir=None):
        self.episodes, self.index, self.values = episodes, [], {}
        self.ut_dir = ut_dir
        normalizer = Normalize(normalize.load(Path(norm_dir or project/'configs/norm')), use_quantiles=True)
        self.normalize = normalizer
        for ep in episodes:
            root = project / f'caches/piperx_rgb_v1/episode_{ep:06d}'
            latent = np.load(project / f'caches/sam2_npy/episode_{ep:06d}.npy').mean((2, 3))
            action, state = np.load(root/'action.npy'), np.load(root/'state.npy')
            ut = np.load(ut_dir/f'episode_{ep:06d}.npy') if ut_dir else None
            self.values[ep] = latent, action, state, ut
            self.index.extend((ep, t) for t in range(len(action)-29))

    def materialize(self, device):
        # 固定窗口一次性整理后驻留GPU，摊销逐样本组装、锁页和小块传输开销。
        latents, actions, targets = [], [], []
        for ep in self.episodes:
            latent, action, state, ut = self.values[ep]
            windows = np.lib.stride_tricks.sliding_window_view(latent, 30, axis=0).transpose(0, 2, 1)
            chunks = np.lib.stride_tricks.sliding_window_view(action, 30, axis=0).transpose(0, 2, 1).copy()
            if self.ut_dir:
                chunks[:, :, :6] -= state[:len(chunks), None, :6]
                chunks = self.normalize({'actions': chunks})['actions'].astype(np.float32)
                targets.append(ut)
            latents.append(np.ascontiguousarray(windows[:, :, :, None, None]))
            actions.append(chunks)
        arrays = {'latent': np.concatenate(latents), 'action': np.concatenate(actions)}
        if targets:
            arrays['ut'] = np.concatenate(targets)
        return {key: torch.from_numpy(value).to(device) for key, value in arrays.items()}

    def __len__(self):
        return len(self.index)

    def __getitem__(self, i):
        ep, t = self.index[i]
        latent, action, state, ut = self.values[ep]
        chunk = action[t:t+30].copy()
        if self.ut_dir:
            chunk[:, :6] -= state[t, :6]
            chunk = self.normalize({'actions': chunk})['actions'].astype(np.float32)
        return {'latent': latent[t:t+30, :, None, None].copy(), 'action': chunk,
                'ut': ut[t].copy() if ut is not None else np.zeros(64, np.float32)}


class DecoderPair(torch.nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.sam_dec, self.act_dec = VideoLatentDecoder(cfg), ActionDecoder(cfg)

    def forward(self, ut, latent, action):
        sam = torch.nn.functional.mse_loss(self.sam_dec(ut), latent)
        act = torch.nn.functional.mse_loss(self.act_dec(ut), action)
        return {'loss': sam+act, 'mse_latent': sam, 'mse_action': act}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--project', type=Path, required=True)
    ap.add_argument('--stage', choices=['mvae', 'decoder'], required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--ut-dir', type=Path)
    ap.add_argument('--split-path', type=Path)
    ap.add_argument('--norm-dir', type=Path)
    ap.add_argument('--epochs', type=int)
    ap.add_argument('--resident-data', action='store_true')
    ap.add_argument('--fused-adamw', action='store_true')
    ap.add_argument('--compile-forward', action='store_true')
    ap.add_argument('--test-steps', type=int)
    ap.add_argument('--resume', type=Path)
    ap.add_argument('--status-interval', type=int, default=100)
    args = ap.parse_args()
    project, output = args.project.resolve(), args.output.resolve()
    device = torch.device('cuda', int(os.environ['LOCAL_RANK']))
    torch.cuda.set_device(device)
    dist.init_process_group('nccl', device_id=device)
    rank, world = dist.get_rank(), dist.get_world_size()
    torch.manual_seed(42)
    cfg = MVAEConfig(window=30, action_dim=7, sam_grid=1, latent_dim=64, beta=0)
    split = json.loads(Path(args.split_path or project/'configs/split.json').read_text())
    train = AuxDataset(project, split['train_episodes'], args.ut_dir, args.norm_dir)
    val = AuxDataset(project, split['val_episodes'], args.ut_dir, args.norm_dir)
    if args.stage == 'mvae':
        actions = np.concatenate([v[1] for v in train.values.values()])
        mean, std = actions.mean(0).astype(np.float32), (actions.std(0)+1e-6).astype(np.float32)
        mean_t, std_t = torch.tensor(mean, device=device), torch.tensor(std, device=device)
        model = PoEMVAE(cfg).to(device)
    else:
        if not args.ut_dir:
            raise ValueError('decoder需要正式u_t标签目录')
        model = DecoderPair(cfg).to(device)
    ddp = DistributedDataParallel(model, device_ids=[device.index])
    optimizer = torch.optim.AdamW(ddp.parameters(), lr=1e-4, weight_decay=1e-4, fused=args.fused_adamw)
    epochs = args.epochs or (100 if args.stage == 'mvae' else 80)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    start, best = 0, float('inf')
    if args.resume:
        saved = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(saved['model'])
        optimizer.load_state_dict(saved['optimizer'])
        scheduler.load_state_dict(saved['scheduler'])
        start, best = saved['epoch']+1, saved['best']
        torch.set_rng_state(saved['rank_rng'][rank]['cpu'].cpu())
        torch.cuda.set_rng_state(saved['rank_rng'][rank]['cuda'].cpu())
    if args.fused_adamw:
        # 旧断点的参数组标志随load恢复，需要显式选择当前执行后端。
        for group in optimizer.param_groups:
            group['fused'] = True
            group['foreach'] = False
        for item in optimizer.state.values():
            if 'step' in item:
                item['step'] = item['step'].to(device)
    if args.compile_forward:
        # 仅编译forward函数，保持checkpoint的模块键名和解码器接口。
        model.forward = torch.compile(model.forward, mode='reduce-overhead')
    if rank == 0:
        output.mkdir(parents=True, exist_ok=True)
        if (output/'metrics.jsonl').exists() and not args.resume:
            raise FileExistsError('已有训练指标需要恢复或使用独立目录')
        run = wandb.init(project='tempo-piperx-ma800', name=output.name, group=output.parent.name,
                         config={'stage': args.stage, 'epochs': epochs, 'batch': 64, 'lr': 1e-4,
                                 'window': 30, 'latent_dim': 64, 'beta': 0, 'test_only': bool(args.test_steps),
                                 'train_episodes': len(split['train_episodes']), 'validation_episodes': len(split['val_episodes']),
                                 'world_size': world, 'split_path': str(Path(args.split_path or project/'configs/split.json').resolve()),
                                 'norm_dir': str(Path(args.norm_dir or project/'configs/norm').resolve())},
                         dir=str(output), id=(output/'wandb_id.txt').read_text().strip() if args.resume else None,
                         resume='must' if args.resume else None)
        run.config.update({'resident_data': args.resident_data, 'fused_adamw': args.fused_adamw,
                           'compile_forward': args.compile_forward}, allow_val_change=True)
        (output/'wandb_id.txt').write_text(run.id)
        stage_status(output, args.stage, 'initializing', epoch=start, total_epochs=epochs, wandb_url=run.url)
    dist.barrier()
    steps_per_epoch = len(train)//64
    sampler = ExactBatchSampler(len(train), 64, 64, rank, world, start*steps_per_epoch, epochs*steps_per_epoch, seed=42)
    resident = train.materialize(device) if args.resident_data else None
    loader = iter(sampler) if resident is not None else iter(DataLoader(train, batch_sampler=sampler, num_workers=0, pin_memory=True))
    val_loader = DataLoader(torch.utils.data.Subset(val, list(range(rank, len(val), world))), batch_size=64)
    executed = 0
    if rank == 0:
        stage_status(output, args.stage, 'running', step=start*steps_per_epoch, epoch=start,
                     total_epochs=epochs, total_steps=epochs*steps_per_epoch, wandb_url=run.url)
    epoch = start-1
    for epoch in range(start, epochs):
        t0 = time.perf_counter()
        ddp.train()
        totals = torch.zeros(3, device=device)
        for local_step in range(steps_per_epoch):
            if resident is not None:
                indices = torch.tensor(next(loader), dtype=torch.long, device=device)
                batch = {key: value.index_select(0, indices) for key, value in resident.items()}
            else:
                batch = {k: v.to(device, non_blocking=True) for k, v in next(loader).items()}
            if args.compile_forward:
                torch.compiler.cudagraph_mark_step_begin()
            if args.stage == 'mvae':
                result = ddp(batch['latent'], (batch['action']-mean_t)/std_t)
            else:
                result = ddp(batch['ut'], batch['latent'], batch['action'])
            optimizer.zero_grad(set_to_none=True)
            # 按各rank真实样本数恢复64样本的全局均值，兼容两卡与三卡。
            (result['loss']*world*len(batch['action'])/64).backward()
            torch.nn.utils.clip_grad_norm_(ddp.parameters(), 1, error_if_nonfinite=True)
            optimizer.step()
            totals += torch.stack([result['loss'].detach()*len(batch['action']),
                                   result['mse_action'].detach()*len(batch['action']),
                                   torch.tensor(len(batch['action']), device=device)])
            executed += 1
            global_step = epoch*steps_per_epoch+local_step+1
            if rank == 0 and global_step%args.status_interval == 0:
                progress = dict(step=global_step, epoch=epoch+1, total_epochs=epochs,
                                total_steps=epochs*steps_per_epoch, loss=float(result['loss'].detach()),
                                lr=optimizer.param_groups[0]['lr'], wandb_url=run.url)
                stage_status(output, args.stage, 'running', **progress)
                print(json.dumps(progress), flush=True)
            if args.test_steps and executed >= args.test_steps:
                break
        ddp.eval()
        validation = torch.zeros(2, device=device)
        with torch.no_grad():
            for val_step, batch in enumerate(val_loader):
                batch = {k: v.to(device) for k, v in batch.items()}
                result = model(batch['latent'], (batch['action']-mean_t)/std_t) if args.stage == 'mvae' else model(batch['ut'], batch['latent'], batch['action'])
                validation[0] += result['loss']*len(batch['action'])
                validation[1] += len(batch['action'])
                if args.test_steps and val_step >= 1:
                    break
        dist.all_reduce(totals)
        dist.all_reduce(validation)
        scheduler.step()
        metric = float(validation[0]/validation[1]) if len(val) else float(totals[0]/totals[2])
        monitor_source = 'validation_loss' if len(val) else 'training_loss'
        improved = metric < best
        best = min(best, metric)
        rng = {'cpu': torch.get_rng_state(), 'cuda': torch.cuda.get_rng_state()}
        rank_rng = [None]*world
        dist.all_gather_object(rank_rng, rng)
        if rank == 0:
            row = {'epoch': epoch+1, 'train_loss': float(totals[0]/totals[2]),
                   'train_action_mse': float(totals[1]/totals[2]), 'val_loss': metric if len(val) else None, 'monitor_loss': metric, 'monitor_source': monitor_source,
                   'lr': optimizer.param_groups[0]['lr'], 'seconds': time.perf_counter()-t0,
                   'train_updates': executed, 'global_update': (epoch+1)*steps_per_epoch,
                   'samples_per_second': float(totals[2])/(time.perf_counter()-t0),
                   'resident_data': args.resident_data, 'fused_adamw': args.fused_adamw,
                   'compile_forward': args.compile_forward}
            print(json.dumps(row), flush=True)
            with (output/'metrics.jsonl').open('a') as stream:
                stream.write(json.dumps(row)+'\n')
            wandb.log(row, step=epoch+1)
            state = dict(model=model.state_dict(), model_cfg=asdict(cfg), optimizer=optimizer.state_dict(),
                         scheduler=scheduler.state_dict(), epoch=epoch, best=best, split=split,
                         rank_rng=rank_rng, test_only=bool(args.test_steps), monitor_source=monitor_source)
            if args.stage == 'mvae':
                state.update(action_mean=mean, action_std=std)
            else:
                state.update(sam_dec=model.sam_dec.state_dict(), act_dec=model.act_dec.state_dict(),
                             action_space='policy_norm', norm_stats_dir=str(Path(args.norm_dir or project/'configs/norm').resolve()))
            temp = output/'.checkpoint.tmp'
            torch.save(state, temp)
            temp.replace(output/'last.pt')
            if improved:
                best_temp = output/'.best.tmp'
                torch.save(state, best_temp)
                best_temp.replace(output/'best.pt')
        dist.barrier()
        if args.test_steps:
            break
    if rank == 0:
        stage_status(output, args.stage, 'finishing', epoch=epoch+1, total_epochs=epochs, wandb_url=run.url)
        run.finish()
        stage_status(output, args.stage, 'completed', epoch=epoch+1, total_epochs=epochs,
                     step=start*steps_per_epoch+executed, total_steps=epochs*steps_per_epoch, wandb_url=run.url)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
