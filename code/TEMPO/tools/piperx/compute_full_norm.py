"""按用户要求，用全部轨迹计算30帧动作空间统计并保存独立版本。"""
import argparse
import json
import os
from pathlib import Path

import numpy as np
from openpi.shared import normalize
from runtime_status import timestamp, write_json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--project', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    project, output = args.project.resolve(), args.output.resolve()
    source = Path(os.environ.get('TEMPO_DATASET_ROOT', project/'data/data11_7/piperx_rotating_red_square'))
    episodes = [json.loads(line) for line in (source/'meta/episodes.jsonl').read_text().splitlines()]
    indices = sorted(entry['episode_index'] for entry in episodes)
    states, actions = [], []
    for index in indices:
        directory = Path(os.environ.get('TEMPO_RGB_CACHE', project/'caches/piperx_rgb_v1'))/f'episode_{index:06d}'
        state = np.load(directory/'state.npy')
        action = np.load(directory/'action.npy')
        # 与正式策略相同，窗口起点六关节相对化，夹爪保持绝对开口。
        frames = np.minimum(np.arange(len(action))[:, None]+np.arange(30), len(action)-1)
        chunk = action[frames].copy()
        chunk[:, :, :6] -= state[:, None, :6]
        states.append(state)
        actions.append(chunk.reshape(-1, 7))
    statistics = {}
    for name, parts in [('state', states), ('actions', actions)]:
        values = np.concatenate(parts)
        statistics[name] = normalize.NormStats(mean=values.mean(0), std=values.std(0),
                                             q01=np.quantile(values, .01, axis=0),
                                             q99=np.quantile(values, .99, axis=0))
    normalize.save(output/'norm', statistics)
    write_json(output/'split.json', {'train_episodes': indices, 'val_episodes': []})
    manifest = {'updated_at': timestamp(), 'train_episodes': len(indices), 'val_episodes': 0,
                'frames': sum(len(state) for state in states), 'window': 30,
                'source': str(source), 'norm_dir': str(output/'norm'),
                'joint_action_space': 'delta_from_window_start_state', 'gripper_action_space': 'absolute'}
    write_json(output/'normalization_manifest.json', manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
