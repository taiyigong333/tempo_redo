"""只做受限辅助模型测试，生成的标签仅用于性能测试。"""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'ut'))
from mvae import MVAEConfig, PoEMVAE, ActionDecoder, VideoLatentDecoder


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--project', type=Path, required=True)
    args = ap.parse_args()
    project = args.project.resolve()
    output = project / 'experiments/preflight_20261114'
    (project / 'caches/sam2_npy').mkdir(exist_ok=True)
    (output / 'ut').mkdir(exist_ok=True)
    cfg = MVAEConfig(window=30, action_dim=7, sam_grid=1, latent_dim=64, beta=0)
    torch.manual_seed(42)
    device = torch.device('cuda:0')
    model = PoEMVAE(cfg).to(device)
    norm = json.loads((project / 'configs/norm/norm_stats.json').read_text())['norm_stats']['actions']
    q01, q99 = np.array(norm['q01']), np.array(norm['q99'])
    rows, pool, actions = [], [], []
    for cache in sorted((project / 'caches/sam2/piperx_rotating_red_square').glob('episode_*.npz')):
        with np.load(cache) as z:
            latent, action, state = z['latent'], z['action'], z['state']
        np.save(project / 'caches/sam2_npy' / f'{cache.stem}.npy', latent)
        if len(pool) < 64:
            for t in range(min(len(latent)-29, 64-len(pool))):
                pool.append(latent[t:t+30].mean((2, 3))[:, :, None, None])
                chunk = action[t:t+30].copy()
                chunk[:, :6] -= state[t, :6]
                actions.append(2*(chunk-q01)/(q99-q01+1e-6)-1)
    latent = torch.tensor(np.stack(pool), device=device)
    action = torch.tensor(np.stack(actions), dtype=torch.float32, device=device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    for step in range(5):
        torch.cuda.synchronize()
        begin = time.perf_counter()
        result = model(latent, action)
        opt.zero_grad(set_to_none=True)
        result['loss'].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        opt.step()
        torch.cuda.synchronize()
        rows.append({'stage': 'mvae', 'step': step, 'loss': float(result['loss']), 'seconds': time.perf_counter()-begin})
    torch.save({'model': model.state_dict(), 'model_cfg': asdict(cfg), 'test_only': True}, output / 'mvae_smoke.pt')
    model.eval()
    encode_start = time.perf_counter()
    with torch.no_grad():
        for cache in sorted((project / 'caches/sam2_npy').glob('episode_*.npy')):
            pooled = np.load(cache).mean((2, 3))
            windows = np.stack([pooled[t:t+30, :, None, None] for t in range(len(pooled)-29)])
            targets = []
            for batch in range(0, len(windows), 128):
                mu, _ = model.video_enc(torch.tensor(windows[batch:batch+128], device=device))
                targets.append(mu.float().cpu().numpy())
            np.save(output / 'ut' / cache.name, np.concatenate(targets))
        mu, _ = model.video_enc(latent)
    act_dec, sam_dec = ActionDecoder(cfg).to(device), VideoLatentDecoder(cfg).to(device)
    params = list(act_dec.parameters())+list(sam_dec.parameters())
    opt = torch.optim.AdamW(params, lr=1e-4, weight_decay=1e-4)
    for step in range(5):
        torch.cuda.synchronize()
        begin = time.perf_counter()
        loss = torch.nn.functional.mse_loss(act_dec(mu), action) + torch.nn.functional.mse_loss(sam_dec(mu), latent)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1)
        opt.step()
        torch.cuda.synchronize()
        rows.append({'stage': 'decoder', 'step': step, 'loss': float(loss), 'seconds': time.perf_counter()-begin})
    report = {'test_only': True, 'window': 30, 'encoded_episodes': len(list((output/'ut').glob('*.npy'))),
              'encoding_seconds': time.perf_counter()-encode_start, 'steps': rows}
    (output / 'aux_smoke.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
