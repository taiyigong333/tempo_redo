"""无损缓存数据入口，复用官方数据变换，限制每进程 mmap 文件句柄数。"""
from collections import OrderedDict
import json
from pathlib import Path

import numpy as np
from torch.utils.data import Dataset

from openpi import transforms
from openpi.models.tokenizer import PaligemmaTokenizer
from openpi.policies.yam_dual_policy import YamInputs
from openpi.models.model import ModelType
from openpi.shared import normalize


class PiperDataset(Dataset):
    def __init__(self, project, ut_dir, subset='train', window=30, split_path=None, norm_dir=None):
        self.project = Path(project)
        self.ut_dir = Path(ut_dir)
        self.window = window
        self.maps = OrderedDict()
        split = json.loads(Path(split_path or self.project/'configs/split.json').read_text())
        self.index = []
        for ep in split[f'{subset}_episodes']:
            length = len(np.load(self.project / f'caches/piperx_rgb_v1/episode_{ep:06d}/state.npy'))
            # 保留尾帧，沿用官方 u_t padding 掩码语义。
            self.index.extend((ep, t) for t in range(length))
        self.transform = transforms.compose([
            YamInputs(model_type=ModelType.PI05),
            transforms.DeltaActions(transforms.make_bool_mask(6, -1)),
            transforms.Normalize(normalize.load(Path(norm_dir or self.project/'configs/norm')), use_quantiles=True),
            transforms.TokenizePrompt(PaligemmaTokenizer(200), discrete_state_input=True),
            transforms.PadStatesAndActions(32),
        ])

    def load(self, path):
        path = str(path)
        if path in self.maps:
            self.maps.move_to_end(path)
            return self.maps[path]
        array = np.load(path, mmap_mode='r')
        self.maps[path] = array
        if len(self.maps) > 64:
            self.maps.popitem(last=False)
        return array

    def __len__(self):
        return len(self.index)

    def __getitem__(self, index):
        weight = np.float32(index >= 0)
        ep, t = self.index[max(index, 0)]
        root = self.project / f'caches/piperx_rgb_v1/episode_{ep:06d}'
        state, action = self.load(root / 'state.npy'), self.load(root / 'action.npy')
        frames = np.maximum(0, np.array([t-20, t-10, t]))
        images = {key: self.load(root / f'{key}.npy')[frames].copy() for key in ['head', 'left_wrist']}
        past = np.arange(t-300, t)
        valid = past >= 0
        hist = np.zeros((300, 7), dtype=np.float32)
        hist[valid] = action[past[valid]]
        latent = self.load(self.project / f'caches/sam2_npy/episode_{ep:06d}.npy')[t]
        ut = self.load(self.ut_dir / f'episode_{ep:06d}.npy')
        data = dict(images=images, state=state[t].copy(),
                    actions=action[np.minimum(t + np.arange(self.window), len(action)-1)].copy(),
                    prompt='grab the rotating red square',
                    sam2_tokens=latent.transpose(1, 2, 0).reshape(64, 256).copy(),
                    action_history=hist.reshape(10, 30, 7).mean(1),
                    action_history_is_pad=(~valid).reshape(10, 30).all(1),
                    ut_target=ut[t].copy() if t < len(ut) else np.zeros(64, np.float32),
                    ut_target_is_pad=np.bool_(t >= len(ut)))
        result = self.transform(data)
        # 分位统计可能为float64，模型边界统一为float32并保留图像/掩码原dtype。
        result["state"] = result["state"].astype(np.float32)
        result["actions"] = result["actions"].astype(np.float32)
        return {**result, "sample_weight": weight}


class ExactBatchSampler:
    """按更新步确定全局顺序，三卡不整除的 batch 也按样本数精确加权。"""
    def __init__(self, size, batch, micro, rank, world, start, steps, seed=42):
        self.size, self.batch, self.micro = size, batch, micro
        self.rank, self.world = rank, world
        self.start, self.steps, self.seed = start, steps, seed
        self.rank_counts = [len(x) for x in np.array_split(np.arange(batch), world)]
        self.accum = int(np.ceil(max(self.rank_counts) / micro))

    def __iter__(self):
        permutations = {}
        for step in range(self.start, self.steps):
            indices = []
            for pos in range(step*self.batch, (step+1)*self.batch):
                epoch, offset = divmod(pos, self.size)
                if epoch not in permutations:
                    permutations = {epoch: np.random.default_rng(self.seed + epoch).permutation(self.size)}
                indices.append(int(permutations[epoch][offset]))
            local = np.array_split(np.array(indices), self.world)[self.rank]
            for chunk in np.array_split(local, self.accum):
                # 不整除且micro=1时，较短rank用零权重样本保持同步次数一致。
                yield chunk.tolist() if len(chunk) else [-1]

    def __len__(self):
        return (self.steps-self.start)*self.accum
