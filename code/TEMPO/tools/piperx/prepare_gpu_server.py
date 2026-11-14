"""复用项目数据校验和官方SAM2编码器，生成本服务器的训练缓存。"""
import argparse
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch
import torch.distributed as dist

from prepare import prepare_episode
from runtime_status import write_json
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sam2_encoder import build_sam2_tiny, preprocess_video_to_sam2_input, stream_extract


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--project', type=Path, required=True)
    args = parser.parse_args()
    project = args.project.resolve()
    device = torch.device('cuda', int(os.environ['LOCAL_RANK']))
    torch.cuda.set_device(device)
    dist.init_process_group('nccl', device_id=device)
    rank, world = dist.get_rank(), dist.get_world_size()
    source = project/'data/data11_7/piperx_rotating_red_square'
    episodes = [json.loads(line) for line in (source/'meta/episodes.jsonl').read_text().splitlines()]
    rgb = project/'caches/piperx_rgb_v1'
    latent_dir = project/'caches/sam2_npy'
    reports = project/'runtime/setup/episodes'
    for directory in (rgb, latent_dir, reports):
        directory.mkdir(parents=True, exist_ok=True)
    model = build_sam2_tiny(device=str(device))
    started = time.perf_counter()
    for episode in episodes[rank::world]:
        index, length = episode['episode_index'], episode['length']
        target = latent_dir/f'episode_{index:06d}.npy'
        report = reports/f'episode_{index:06d}.json'
        # 仅完整校验过的本机产物允许跳过，中断传输留下的文件会重新生成。
        if report.is_file() and target.is_file():
            array = np.load(target, mmap_mode='r')
            if array.shape == (length, 256, 8, 8):
                continue
        validation = prepare_episode((source, rgb, episode))
        video = np.load(rgb/f'episode_{index:06d}/head.npy')
        inputs = preprocess_video_to_sam2_input(video)
        with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
            latent = stream_extract(model, inputs, grid=8, use_full_frame_mask=True)
        values = latent.float().cpu().numpy()
        assert values.shape == (length, 256, 8, 8) and np.isfinite(values).all()
        # 临时文件完成写入后原子发布，支持准备阶段中断后的安全重跑。
        temporary = target.with_suffix('.tmp')
        with temporary.open('wb') as stream:
            np.save(stream, values)
        temporary.replace(target)
        write_json(report, validation)
        print(json.dumps({'rank': rank, 'episode': index, 'frames': length,
                          'elapsed_seconds': time.perf_counter()-started}), flush=True)
    dist.barrier()
    if rank == 0:
        rows = [json.loads((reports/f"episode_{ep['episode_index']:06d}.json").read_text()) for ep in episodes]
        summary = {'episodes': len(rows), 'frames': sum(row['frames'] for row in rows),
                   'videos': len(rows)*2, 'max_next_state_error': max(row['next_state_max_error'] for row in rows),
                   'sam2_shape': ['T', 256, 8, 8], 'sam2_dtype': 'float32',
                   'seconds': time.perf_counter()-started, 'world_size': world}
        write_json(project/'docs/data_validation_gpu_server.json', summary)
        print(json.dumps(summary), flush=True)
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
