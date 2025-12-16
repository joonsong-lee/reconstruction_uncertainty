import math
from dataclasses import dataclass
from collections import OrderedDict

from rotary_embedding_torch import RotaryEmbedding
import torch
import torch.nn as nn
from torch.nn import functional as F


class LayerNorm(nn.Module):
    """LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False"""

    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)


class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        # regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        # flash attention make GPU go brrrrr but support is only in PyTorch >= 2.0
        self.flash = hasattr(torch.nn.functional, "scaled_dot_product_attention")
        self.rope = RotaryEmbedding(config.n_embd // config.n_head, cache_max_seq_len = 400)

    def forward(self, x,key_cache = None,value_cache = None):
        (
            B,
            T,
            C,
        ) = x.size()  # batch size, sequence length, embedding dimensionality (n_embd)
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k1, v1 = self.c_attn(x).split(self.n_embd, dim=2)
        if key_cache is not None :
            key_cache[:,-T:] = k1
            value_cache[:,-T:] = v1
            k=key_cache
            v = value_cache
            true_T = k.size(1)
        else:
            true_T = T
            k = k1
            v = v1
        k = k.view(B, true_T, self.n_head, C // self.n_head).transpose(
            1, 2
        )  # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)

        v = v.view(B, true_T, self.n_head, C // self.n_head).transpose(
            1, 2
        )  # (B, nh, T, hs)
        if key_cache is not None :
            q = self.rope.rotate_queries_or_keys(q,offset=true_T - T)
            k = self.rope.rotate_queries_or_keys(k)
        else:
            q = self.rope.rotate_queries_or_keys(q)
            k = self.rope.rotate_queries_or_keys(k)
        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
            # efficient attention using Flash Attention CUDA kernels
        if key_cache is not None:
            y = torch.nn.functional.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=None,
                dropout_p=self.dropout if self.training else 0,
                is_causal=False,)
        else :
            y = torch.nn.functional.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=None,
            dropout_p=self.dropout if self.training else 0,
            is_causal=True,
        )
        y = (
            y.transpose(1, 2).contiguous().view(B, T, C)
        )  # re-assemble all head outputs side by side
        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y,k1,v1



class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class Block(nn.Module):
    def __init__(self, config,kv_cache=False):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)
        self.config = config
        self.kv_cache = kv_cache
        if kv_cache:
            self.keys=torch.zeros((config.batch_size,config.frame_length,config.n_embd))
            self.values=torch.zeros((config.batch_size,config.frame_length,config.n_embd))
            self.cnt = 0

    def forward(self, x):
        if self.kv_cache and self.cnt!=0 and not self.training : #when using kv cache and not the first step
            y,k,v = self.attn(self.ln_1(x),self.keys[:,:self.cnt+x.shape[1]],self.values[:,:self.cnt+x.shape[1]])
            self.keys[:,self.cnt:self.cnt+x.shape[1]] = k
            self.values[:,self.cnt:self.cnt+x.shape[1]] = v
            self.cnt+=x.shape[1]
        elif self.kv_cache and self.cnt==0 and not self.training: #when it is first step
            y,k,v = self.attn(self.ln_1(x))
            shape = x.shape[1]
            self.keys[:,:shape] = k
            self.values[:,:shape] = v
            self.keys = self.keys.to(x.device)
            self.values = self.values.to(x.device)
            self.cnt = shape
        else :
            y,_,_ = self.attn(self.ln_1(x))
        x= x+y
        x = x + self.mlp(self.ln_2(x))
        return x


@dataclass
class GPTConfig:
    frame_length: int
    n_layer: int
    n_head: int
    n_embd: int
    input_size: int
    dropout: float
    bias: bool
    batch_size:int


class GPT(nn.Module):
    def __init__(self, config,is_kv=False):
        super().__init__()
        self.config = config
        self.in_proj = nn.Linear(config.input_size, config.n_embd)
        self.transformer = nn.ModuleDict(
            dict(
                #wte_u = nn.Embedding(config.action_size, config.n_embd),
                drop=nn.Dropout(config.dropout),
                h=nn.ModuleList([Block(config,is_kv) for _ in range(config.n_layer)]),
                ln_f=LayerNorm(config.n_embd, bias=config.bias),
            )
        )
        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight"):
                torch.nn.init.normal_(
                    p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer)
                )
        self.is_kv = is_kv

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def compute_context_vector(self, X): 
        device = X.device
        b, t, n_embd = X.size()
        # forward the GPT model itself
        # from: https://stackoverflow.com/questions/61026393/pytorch-concatenate-rows-in-alternate-order
        x = self.transformer.drop(X )
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        return x
    
    def reset_cache_with_burnin(self,X):
        assert self.transformer.h[0].kv_cache==True, "kv cache should be enabled to use this function"
        for block in self.transformer.h:
            block.cnt = 0 #all the block contatins its own step count, reset it to 0
        context = self.compute_context_vector(X) #kv will be automatically updated in each block
        return context

    def reset_cache_without_burnin(self):
        for block in self.transformer.h:
            block.cnt = 0

    def compute_context_vector_with_cache(self, X): 
        assert self.transformer.h[0].kv_cache==True, "kv cache should be enabled to use this function"
        x1= X.unsqueeze(1) if len(X.shape)==2 else X  # (B,1,n_embd)
        device = X.device
        x1 = self.transformer.drop(x1 )
        for block in self.transformer.h:
            x1 = block(x1)
        x1 = self.transformer.ln_f(x1)
        return x1

    def forward(self, X):
        X = self.in_proj(X)
        if self.is_kv and not self.training:
            x = self.compute_context_vector_with_cache(X)
        else:
            x = self.compute_context_vector(X) # reward (when including reward in context)
        return x
 