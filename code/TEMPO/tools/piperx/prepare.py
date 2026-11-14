"""验证 PiperX 全量数据，并生成无损 RGB 缓存与训练集归一化。"""
import argparse
import concurrent.futures
import json
from pathlib import Path
import time

import av
import numpy as np
import pyarrow.parquet as pq
from openpi.shared import normalize


def prepare_episode(args):
    root, out, ep = args
    idx, length = ep['episode_index'], ep['length']
    name = f'episode_{idx:06d}'
    table = pq.read_table(root / 'data/chunk-000' / f'{name}.parquet')
    action = np.asarray(table['action'].to_pylist(), dtype=np.float32)
    state = np.asarray(table['observation.state'].to_pylist(), dtype=np.float32)
    assert len(action) == length and action.shape == state.shape == (length, 7)
    assert np.isfinite(action).all() and np.isfinite(state).all()
    np.testing.assert_array_equal(table['frame_index'].to_numpy(), np.arange(length))
    np.testing.assert_array_equal(table['episode_index'].to_numpy(), np.full(length, idx))
    np.testing.assert_allclose(table['timestamp'].to_numpy(), np.arange(length) / 30, atol=2e-5)
    proxy_error = float(np.max(np.abs(action[:-1] - state[1:])))
    target = out / name
    target.mkdir(parents=True, exist_ok=True)
    for key in ['head', 'left_wrist']:
        video = root / 'videos/chunk-000' / f'observation.images.{key}' / f'{name}.mp4'
        with av.open(str(video)) as container:
            frames = np.stack([f.to_ndarray(format='rgb24') for f in container.decode(video=0)])
        assert frames.shape == (length, 224, 224, 3)
        np.save(target / f'{key}.npy', frames)
    np.save(target / 'state.npy', state)
    np.save(target / 'action.npy', action)
    return {'episode_index': idx, 'frames': length, 'next_state_max_error': proxy_error}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--project', type=Path, required=True)
    ap.add_argument('--window', type=int, default=30)
    args = ap.parse_args()
    project = args.project.resolve()
    root = project / 'data/data11_7/piperx_rotating_red_square'
    out = project / 'caches/piperx_rgb_v1'
    out.mkdir(parents=True, exist_ok=True)
    episodes = [json.loads(x) for x in (root / 'meta/episodes.jsonl').read_text().splitlines()]
    t0 = time.perf_counter()
    with concurrent.futures.ProcessPoolExecutor(max_workers=12) as pool:
        results = list(pool.map(prepare_episode, [(root, out, e) for e in episodes]))
    rng = np.random.default_rng(0)
    perm = rng.permutation(sorted(e['episode_index'] for e in episodes))
    count_val = round(len(episodes) * .2)
    split = {'train_episodes': sorted(map(int, perm[count_val:])), 'val_episodes': sorted(map(int, perm[:count_val]))}
    (project / 'configs/split.json').write_text(json.dumps(split, indent=2))
    states, chunks = [], []
    for ep in split['train_episodes']:
        directory = out / f'episode_{ep:06d}'
        state, action = np.load(directory / 'state.npy'), np.load(directory / 'action.npy')
        states.append(state)
        # 与策略 DeltaActions 一致，每个窗口的六关节相对当前实测状态，夹爪绝对值。
        indices = np.minimum(np.arange(len(action))[:, None] + np.arange(args.window), len(action) - 1)
        chunk = action[indices].copy()
        chunk[:, :, :6] -= state[:, None, :6]
        chunks.append(chunk.reshape(-1, 7))
    stats = {}
    for key, values in [('state', states), ('actions', chunks)]:
        values = np.concatenate(values)
        stats[key] = normalize.NormStats(mean=values.mean(0), std=values.std(0),
                                         q01=np.quantile(values, .01, axis=0), q99=np.quantile(values, .99, axis=0))
    normalize.save(project / 'configs/norm', stats)
    report = {'episodes': len(results), 'frames': sum(r['frames'] for r in results), 'videos': len(results)*2,
              'max_next_state_error': max(r['next_state_max_error'] for r in results),
              'train_episodes': len(split['train_episodes']), 'val_episodes': len(split['val_episodes']),
              'seconds': time.perf_counter()-t0, 'episodes_detail': results}
    (project / 'experiments/preflight_20261114/data_validation.json').write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k != 'episodes_detail'}, indent=2))


if __name__ == '__main__':
    main()
