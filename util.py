import os
from time import time
os.environ["IMAGEIO_FFMPEG_EXE"] = "/usr/bin/ffmpeg"
from moviepy import ImageSequenceClip
from collections import OrderedDict
import jax
import math
import random
import numpy as np
import torch 
from torch import nn
from torch.distributed import init_process_group
from torch.nn import functional as F
import wandb

def compute_lambda_returns(rwds, values, gamma, lambda_):
    advantages = torch.zeros_like(rwds).to(rwds.device) # (B, T)
    # values (B, T+1) / rwds (B, T)
    for t in reversed(range(rwds.size(1))): # T-1 부터 0 까지
        # rware는 'terminated'가 없으므로 next_nonterminal = 1.0 (항상 1)
        # (만약 'terminated'를 구현했다면 (1.0 - terminated_dones[:, t])를 곱해야 함)
        next_values = values[:, t + 1] # V(s_{t+1})
        delta = rwds[:, t] + gamma * next_values - values[:, t]
        advantages[:, t] =  delta +  gamma * lambda_ *  (advantages[:,t+1] if t +1 < rwds.size(1) else 0)
    lambda_return = advantages + values[:, :-1]
    return lambda_return, advantages

class LossWithIntermediateLosses:
    def __init__(self, **kwargs):
        self.loss_total = sum(kwargs.values())
        self.intermediate_losses = {k: v.item() for k, v in kwargs.items()}

    def __truediv__(self, value):
        for k, v in self.intermediate_losses.items():
            self.intermediate_losses[k] = v / value
        self.loss_total = self.loss_total / value
        return self
    
def get_lr(it, warmup_iters, lr_decay_iters, wm_learning_rate, min_lr):
    # 1) linear warmup for warmup_iters steps
    if it < warmup_iters:
        return wm_learning_rate * it / warmup_iters
    # 2) if it > lr_decay_iters, return min learning rate
    if it > lr_decay_iters:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # coeff ranges 0..1
    return min_lr + coeff * (wm_learning_rate - min_lr)


def set_seed(seed: int):
    """
    Sets the seed for all relevant random number generators to ensure
    reproducibility or controlled diversity.
    """
    # Set the seed for Python's built-in random module
    random.seed(seed)
    
    # Set the seed for NumPy
    np.random.seed(seed)
    
    # Set the seed for PyTorch on CPU
    torch.manual_seed(seed)
    
    # Set the seed for PyTorch on GPU (if available)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # for multi-GPU
        
        # These two are often needed for full reproducibility
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def setup_environment(config,seed=None):
    """
    DDP, wandb, 시드, 장치 설정 등 환경을 초기화합니다.
    """
    ddp = int(os.environ.get('RANK', -1)) != -1
    if ddp:
        init_process_group(backend="nccl")
        ddp_rank = int(os.environ["RANK"])
        ddp_local_rank = int(os.environ['LOCAL_RANK'])
        ddp_world_size = int(os.environ["WORLD_SIZE"])
        device = f'cuda:{ddp_local_rank}'
        torch.cuda.set_device(device)
        master_process = ddp_rank == 0
        seed_offset = ddp_rank if seed is None else seed
        
    else:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        ddp_rank = 0
        ddp_local_rank = 0
        ddp_world_size = 1
        master_process = True
        seed_offset = int(f'{time():.10f}'[-9:][::-1]) if seed is None else seed

    if master_process:
        os.makedirs(config['path']['out_dir'], exist_ok=True)
        wandb.init(**config['wandb'], config=config)
    
    torch.manual_seed(1337 + seed_offset)
    np.random.seed(1337 + seed_offset)
    torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
    torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
    
    return device, master_process, ddp_rank, ddp_local_rank, ddp_world_size

def setup_dataloaders(config,train_dataset,val_dataset, ddp_world_size):
    """
    데이터셋과 데이터로더를 설정합니다.
    """
    # tr = transform if config['params']['dataset']['interpolate'] is None else transform_interpolate
    # train_dataset = VAE_dataset(tr, "train", **config['params']['dataset'])
    # val_dataset = VAE_dataset(tr, "val", **config['params']['dataset'])

    train_sampler = None
    val_sampler = None
    train_shuffle = True

    if ddp_world_size > 1:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, shuffle=True)
        val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset, shuffle=False) if val_dataset is not None else None
        train_shuffle = False # Sampler가 셔플을 담당

    train_loader = torch.utils.data.DataLoader(
        train_dataset, 
        batch_size=config['params']['batch_size'], 
        sampler=train_sampler,
        num_workers=config['params']['num_worker'],
        shuffle=train_shuffle,
        pin_memory=True,
        drop_last=True
    )
    if val_dataset is not None:
        val_loader = torch.utils.data.DataLoader(
            val_dataset, 
            batch_size=config['params']['batch_size'], 
            sampler=val_sampler,
            num_workers=config['params']['num_worker'],
            shuffle=False,
            pin_memory=True,
            drop_last=True
        )
    else: val_loader = None
    max_iter_per_epoch = len(train_dataset) // ddp_world_size // config['params']['batch_size']
    
    return train_loader, val_loader, train_sampler, max_iter_per_epoch


def get_param_groups(net,lr,weight_decay):
    decay,no_decay = [],[]
    for name, param in net.named_parameters():
            if param.requires_grad:
                if name.endswith(".bias") or "norm" in name.lower() :
                    no_decay.append(param)
                else:
                    decay.append(param)
    return [{'params': decay, 'lr': lr, 'weight_decay': weight_decay},
        {'params': no_decay, 'lr': lr, 'weight_decay': 0.0},
    ]

def build_optimizer(net,lr, weight_decay, betas, eps):
    params = get_param_groups(net,lr,weight_decay)
    optimizer = torch.optim.AdamW(params,betas = tuple(betas),eps=eps)
    return optimizer

def remove_actor2_prefix(state_dict):
    new_state_dict = {k[len('actor2.'):]: v for k, v in state_dict.items() if k.startswith('actor2.')}
    return new_state_dict

def remove_actor1_prefix(state_dict):
    new_state_dict = {k[len('actor1.'):]: v for k, v in state_dict.items() if k.startswith('actor1.')}
    return new_state_dict

def load_partner_weights(partner_model, state_dict):
    """
    nn.LSTM으로 학습된 state_dict를 nn.LSTMCell을 쓰는 모델에 로드하기 위해 키를 변환합니다.
    """
    new_state_dict = {}
    for k, v in state_dict.items():
        # LSTM Weight 이름 매핑: 'lstm.weight_ih_l0' -> 'lstm.weight_ih'
        if 'lstm' in k and '_l0' in k:
            new_k = k.replace('_l0', '')
        else:
            new_k = k
        new_state_dict[new_k] = v
        
    partner_model.load_state_dict(new_state_dict)
    return partner_model

def t2j(tensor):
    """Torch Tensor -> JAX Array (Zero-Copy, Modern Way)"""
    # 1. PyTorch 텐서가 연속적인 메모리인지 확인 (DLPack 필수)
    tensor = tensor.contiguous()
    
    # 2. JAX로 변환 (캡슐 없이 바로 변환 가능)
    # from_dlpack은 __dlpack__ 메서드가 구현된 객체를 직접 받습니다.
    return jax.dlpack.from_dlpack(tensor)

def j2t(jax_array):
    """JAX Array -> Torch Tensor (Zero-Copy, Modern Way)"""
    # 1. Torch로 변환 (역시 캡슐 없이 바로 변환 가능)
    return torch.utils.dlpack.from_dlpack(jax_array)

def make_movie(obs,path):
    clip = ImageSequenceClip(list(obs), fps=10)
    clip.write_gif(path, fps=10)
