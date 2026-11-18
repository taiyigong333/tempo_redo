"""三卡并行编码正式或测试MVAE标签，保持原工具npz格式并提供npy缓存。"""
import argparse
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch
import torch.distributed as dist
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'ut'))
from mvae import MVAEConfig, PoEMVAE


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--project', type=Path, required=True)
    ap.add_argument('--mvae', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    project, output = args.project.resolve(), args.output.resolve()
    device = torch.device('cuda', int(os.environ['LOCAL_RANK']))
    torch.cuda.set_device(device)
    dist.init_process_group('nccl', device_id=device)
    rank, world = dist.get_rank(), dist.get_world_size()
    saved = torch.load(args.mvae, map_location='cpu', weights_only=False)
    cfg = MVAEConfig(**saved['model_cfg'])
    model = PoEMVAE(cfg).to(device).eval()
    model.load_state_dict(saved['model'])
    output.mkdir(parents=True, exist_ok=True)
    files = sorted(Path(os.environ.get('TEMPO_SAM2_CACHE', project/'caches/sam2_npy')).glob('episode_*.npy'))
    begin = time.perf_counter()
    with torch.no_grad():
        for source in files[rank::world]:
            pooled = np.load(source).mean((2, 3))
            windows = np.stack([pooled[t:t+cfg.window, :, None, None] for t in range(len(pooled)-cfg.window+1)])
            result = []
            for start in range(0, len(windows), 128):
                mu, _ = model.video_enc(torch.tensor(windows[start:start+128], device=device))
                result.append(mu.float().cpu().numpy())
            latent = np.concatenate(result)
            np.save(output/source.name, latent)
            np.savez(output/f'{source.stem}.npz', u_window=latent, ep_len=np.int32(len(pooled)))
    dist.barrier()
    if rank == 0:
        manifest = {'mvae': str(args.mvae.resolve()), 'window': cfg.window, 'latent_dim': cfg.latent_dim,
                    'episodes': len(files), 'test_only': saved.get('test_only', False),
                    'seconds': time.perf_counter()-begin}
        (output/'manifest.json').write_text(json.dumps(manifest, indent=2))
        print(json.dumps(manifest, indent=2))
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
