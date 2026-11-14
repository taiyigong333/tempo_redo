"""GPU分片AdamW，FP32主权重/动量与BF16计算权重始终驻留GPU。"""
import math

import torch
import torch.distributed as dist


@torch.no_grad()
def adamw_update(master, gradient, first, second, step, lr, betas, eps, weight_decay, chunk_size=4_194_304):
    """分块转FP32梯度与更新，限制临时张量显存，公式与torch AdamW一致。"""
    beta1, beta2 = betas
    correction1 = 1-beta1**step
    correction2 = math.sqrt(1-beta2**step)
    arrays = [tensor.reshape(-1) for tensor in (master, gradient, first, second)]
    for begin in range(0, master.numel(), chunk_size):
        value, grad, m, v = [tensor[begin:begin+chunk_size] for tensor in arrays]
        grad = grad.float()
        value.mul_(1-lr*weight_decay)
        m.lerp_(grad, 1-beta1)
        v.mul_(beta2).addcmul_(grad, grad, value=1-beta2)
        denominator = v.sqrt().div_(correction2).add_(eps)
        value.addcdiv_(m, denominator, value=-lr/correction1)


class GPUAdamW:
    def __init__(self, parameters):
        self.parameters = [parameter for parameter in parameters if parameter.requires_grad]
        world, rank = dist.get_world_size(), dist.get_rank()
        sizes = [0]*world
        self.owners = [[] for _ in range(world)]
        for index in sorted(range(len(self.parameters)), key=lambda i: self.parameters[i].numel(), reverse=True):
            owner = min(range(world), key=sizes.__getitem__)
            self.owners[owner].append(index)
            sizes[owner] += self.parameters[index].numel()
        self.local_indices = self.owners[rank]
        self.masters = [self.parameters[i].detach().float().clone() for i in self.local_indices]
        self.states = [{} for _ in self.masters]
        self.param_groups = [{'lr': 5e-5, 'betas': (.9, .95), 'eps': 1e-8, 'weight_decay': 1e-10}]
        self.optim = self

    def zero_grad(self, set_to_none=False):
        for parameter in self.parameters:
            if parameter.grad is not None:
                if set_to_none:
                    parameter.grad = None
                else:
                    parameter.grad.zero_()

    @torch.no_grad()
    def step(self):
        group = self.param_groups[0]
        for index, master, state in zip(self.local_indices, self.masters, self.states, strict=True):
            gradient = self.parameters[index].grad
            if gradient is None:
                continue
            if not state:
                state.update(first=torch.zeros_like(master), second=torch.zeros_like(master), step=0)
            state['step'] += 1
            adamw_update(master, gradient, state['first'], state['second'], state['step'], **group)
            self.parameters[index].copy_(master)
        for owner, indices in enumerate(self.owners):
            for dtype in (torch.bfloat16, torch.float32):
                parameters = [self.parameters[i] for i in indices if self.parameters[i].dtype == dtype]
                if not parameters:
                    continue
                flat = torch._utils._flatten_dense_tensors(parameters)
                dist.broadcast(flat, src=owner)
                for parameter, value in zip(parameters, torch._utils._unflatten_dense_tensors(flat, parameters), strict=True):
                    parameter.copy_(value)

    def state_dict(self):
        return {'masters': self.masters, 'states': self.states, 'param_groups': self.param_groups,
                'indices': self.local_indices, 'world_size': dist.get_world_size()}

    def load_state_dict(self, state):
        if state['indices'] != self.local_indices or state['world_size'] != dist.get_world_size():
            raise ValueError('恢复时GPU优化器分片需保持一致')
        for parameter, value in zip(self.masters, state['masters'], strict=True):
            parameter.copy_(value)
        self.states = [{key: value.to(self.masters[0].device) if isinstance(value, torch.Tensor) else value
                        for key, value in item.items()} for item in state['states']]
        self.param_groups = state['param_groups']
