import math
import jax.numpy as jnp
import jax
import flax.nnx as nnx
from typing import NamedTuple


#from google deepmind' gemma https://github.com/google-deepmind/gemma/blob/main/gemma/gm/math/_positional_embeddings.py
def apply_rope(
    inputs: jax.Array,
    positions: jax.Array,
    *,
    base_frequency: int = 10000,
    scale_factor: float = 1.0,
    rope_proportion: float = 1.0,
) -> jax.Array:
  """Applies RoPE.

  Let B denote batch size, L denote sequence length, N denote number of heads,
  and H denote head dimension. Note that H must be divisible by 2.

  Args:
    inputs: Array of shape [B, L, N, H].
    positions:  Array of shape [B, L].
    base_frequency: Base frequency used to compute rotations.
    scale_factor: The scale factor used for positional interpolation, allowing
      an expansion of sequence length beyond the pre-trained context length.
    rope_proportion: The proportion of the head dimension to apply RoPE to.

  Returns:
    Array of shape [B, L, N, H].
  """
  head_dim = inputs.shape[-1]
  rope_angles = int(rope_proportion * head_dim // 2)
  nope_angles = head_dim // 2 - rope_angles
  freq_exponents = (
      (2.0 / head_dim) * jnp.arange(0, rope_angles, dtype=jnp.float32)
  )
  timescale = jnp.pad(
      base_frequency**freq_exponents,
      (0, nope_angles),
      mode='constant',
      constant_values=(0, jnp.inf),
  )

  sinusoid_inp = (
      positions[..., jnp.newaxis] / timescale[jnp.newaxis, jnp.newaxis, :]
  )
  sinusoid_inp = sinusoid_inp[..., jnp.newaxis, :]
  if scale_factor < 1.0:
    raise ValueError(f'scale_factor must be >= 1.0, got {scale_factor}')
  sinusoid_inp /= scale_factor

  sin = jnp.sin(sinusoid_inp)
  cos = jnp.cos(sinusoid_inp)

  first_half, second_half = jnp.split(inputs, 2, axis=-1)
  first_part = first_half * cos - second_half * sin
  second_part = second_half * cos + first_half * sin
  out = jnp.concatenate([first_part, second_part], axis=-1)
  return out.astype(inputs.dtype)

class CausalSelfAttention(nnx.Module):
    def __init__(self, config, rngs):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nnx.Linear(in_features=config.n_embd, out_features=3 * config.n_embd, use_bias=config.bias, rngs=rngs)
        # output projection
        self.c_proj = nnx.Linear(in_features=config.n_embd, out_features=config.n_embd, use_bias=config.bias, rngs=rngs)
        # regularization
        self.attn_dropout = nnx.Dropout(rate=config.dropout)
        self.residual_dropout = nnx.Dropout(rate=config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        self.head_dim = config.n_embd // config.n_head
        self.max_seq_len = config.frame_length

    def __call__(self, x, kv_cache=None, cache_index=None):
        """
        Args:
            x: input tensor (B, T, C)
            kv_cache: tuple of (cached_k, cached_v) each with shape (B, max_seq_len, n_head, head_dim)
                     or None if not using cache
            cache_index: current position in cache (scalar int), required when using kv_cache
        
        Returns:
            y: output tensor (B, T, C)
            new_kv_cache: tuple of (new_k, new_v) for caching (same shape as input cache)
        """
        B, T, C = x.shape
        
        # Calculate query, key, values
        q, k, v = jnp.split(self.c_attn(x), [self.n_embd, self.n_embd * 2], axis=-1)
        k = k.reshape(B, T, self.n_head, self.head_dim)  # (B, T, nh, hs)
        q = q.reshape(B, T, self.n_head, self.head_dim)  # (B, T, nh, hs)
        v = v.reshape(B, T, self.n_head, self.head_dim)  # (B, T, nh, hs)
        
        # Compute positions for RoPE
        if kv_cache is not None:
            # Using cache: positions start from cache_index
            # Use lax.iota instead of jnp.arange for JIT compatibility with traced values
            positions = jax.lax.iota(jnp.int32, T) + cache_index
            positions = jnp.broadcast_to(positions[jnp.newaxis, :], (B, T))
        else:
            positions = jnp.broadcast_to(jnp.arange(T)[jnp.newaxis, :], (B, T))
        
        # Apply RoPE
        q = apply_rope(q, positions)
        k = apply_rope(k, positions)
        
        # Handle KV cache with fixed-size buffer and dynamic slicing
        if kv_cache is not None:
            cached_k, cached_v = kv_cache
            max_seq_len = cached_k.shape[1]
            
            # Update cache at current position using dynamic_update_slice
            # cached_k shape: (B, max_seq_len, n_head, head_dim)
            # k shape: (B, T, n_head, head_dim)
            new_cached_k = jax.lax.dynamic_update_slice(cached_k, k, (0, cache_index, 0, 0))
            new_cached_v = jax.lax.dynamic_update_slice(cached_v, v, (0, cache_index, 0, 0))
            
            # Use full cache and apply attention mask to ignore invalid positions
            k_for_attn = new_cached_k  # (B, max_seq_len, n_head, head_dim)
            v_for_attn = new_cached_v  # (B, max_seq_len, n_head, head_dim)
            
            # Create mask: valid positions are [0, cache_index + T)
            valid_len = cache_index + T
            cache_positions = jax.lax.iota(jnp.int32, max_seq_len)  # [0, 1, 2, ..., max_seq_len-1]
            # mask is True for invalid positions (to be masked out)
            attn_mask = cache_positions[jnp.newaxis, :] >= valid_len  # (1, max_seq_len)
            
            new_kv_cache = (new_cached_k, new_cached_v)
        else:
            k_for_attn = k
            v_for_attn = v
            new_kv_cache = None
            attn_mask = None
        
        # Swap axes for attention: (B, T, nh, hs) -> (B, nh, T, hs)
        q = q.swapaxes(1, 2)
        k_attn = k_for_attn.swapaxes(1, 2)
        v_attn = v_for_attn.swapaxes(1, 2)
        
        # Causal self-attention
        if kv_cache is not None:
            # With cache: use mask to ignore invalid (future) positions in cache
            # attn_mask shape: (1, max_seq_len) -> expand to (B, 1, T, max_seq_len)
            # True means "mask out" so we set those to -inf
            mask = jnp.where(attn_mask[jnp.newaxis, :, jnp.newaxis, :], 
                             jnp.finfo(q.dtype).min, 0.0)  # (1, 1, 1, max_seq_len)
            # Manual attention since we need custom mask
            scale = 1.0 / jnp.sqrt(self.head_dim)
            attn_weights = jnp.einsum('bhqd,bhkd->bhqk', q, k_attn) * scale  # (B, nh, T, max_seq_len)
            attn_weights = attn_weights + mask
            attn_weights = jax.nn.softmax(attn_weights, axis=-1)
            y = jnp.einsum('bhqk,bhkd->bhqd', attn_weights, v_attn)  # (B, nh, T, hs)
        else:
            # Without cache: use causal mask
            y = jax.nn.dot_product_attention(q, k_attn, v_attn, is_causal=True)
        y = y.swapaxes(1, 2).reshape(B, T, C)  # (B, T, C)
        y = self.c_proj(y)

        return y, new_kv_cache



class MLP(nnx.Module):
    def __init__(self, config,rngs):
        super().__init__()
        self.c_fc = nnx.Linear(in_features=config.n_embd, out_features=4 * config.n_embd, use_bias=config.bias,rngs=rngs)
        self.c_proj = nnx.Linear(in_features=4 * config.n_embd, out_features=config.n_embd, use_bias=config.bias,rngs=rngs)
        self.dropout = nnx.Dropout(rate=config.dropout)

    def __call__(self, x):
        x = self.c_fc(x)
        x = nnx.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class Block(nnx.Module):
    def __init__(self, config, rngs):
        super().__init__()
        self.ln_1 = nnx.LayerNorm(config.n_embd, epsilon=1e-5, use_fast_variance=False, rngs=rngs)
        self.attn = CausalSelfAttention(config, rngs=rngs)
        self.ln_2 = nnx.LayerNorm(config.n_embd, epsilon=1e-5, use_fast_variance=False, rngs=rngs)
        self.mlp = MLP(config, rngs=rngs)
        self.config = config

    def __call__(self, x, kv_cache=None, cache_index=None):
        """
        Args:
            x: input tensor (B, T, C)
            kv_cache: KV cache for this block or None
            cache_index: current position in cache (scalar int)
        
        Returns:
            x: output tensor (B, T, C)
            new_kv_cache: updated KV cache
        """
        attn_out, new_kv_cache = self.attn(self.ln_1(x), kv_cache=kv_cache, cache_index=cache_index)
        x = x + attn_out
        x = x + self.mlp(self.ln_2(x))
        return x, new_kv_cache



class GPTConfig(NamedTuple):
    frame_length: int
    n_layer: int
    n_head: int
    n_embd: int
    input_size: int
    dropout: float
    bias: bool
    batch_size:int


class GPT(nnx.Module):
    def __init__(self, config, rng, is_kv=False):
        super().__init__()
        self.config = config
        self.in_proj = nnx.Linear(config.input_size, config.n_embd, rngs=rng)
        self.drop = nnx.Dropout(rate=config.dropout, rngs=rng)
        self.ln_f = nnx.LayerNorm(config.n_embd, epsilon=1e-5, use_fast_variance=False, rngs=rng)
        
        # Create blocks using nnx.List for Flax NNX 0.12.0+ compatibility
        blocks_list = []
        for i in range(config.n_layer):
            layer_rng = nnx.Rngs(jax.random.fold_in(rng.default.key.value, i))
            blocks_list.append(Block(config, rngs=layer_rng))
        self.blocks = nnx.List(blocks_list)
        
        self.num_layers = config.n_layer
        self.is_kv = is_kv
        self.head_dim = config.n_embd // config.n_head
        self.n_head = config.n_head
        self.max_seq_len = config.frame_length

    def init_kv_cache(self, batch_size):
        """Initialize fixed-size KV cache for all layers.
        
        Args:
            batch_size: batch size for the cache
        
        Returns:
            List of (k_cache, v_cache) tuples for each layer,
            each with shape (batch_size, max_seq_len, n_head, head_dim)
        """
        cache = []
        for _ in range(self.num_layers):
            k_cache = jnp.zeros((batch_size, self.max_seq_len, self.n_head, self.head_dim))
            v_cache = jnp.zeros((batch_size, self.max_seq_len, self.n_head, self.head_dim))
            cache.append((k_cache, v_cache))
        return cache
    
    def __call__(self, X, kv_cache=None, cache_index=None):
        """
        Forward pass with optional KV cache support.
        
        Args:
            X: input tensor (B, T, input_size)
            kv_cache: List of (k_cache, v_cache) for each layer, or None for no caching
                     Each cache has shape (B, max_seq_len, n_head, head_dim)
            cache_index: current position in cache (scalar int), required when using kv_cache
        
        Returns:
            output: transformed tensor (B, T, n_embd)
            new_kv_cache: updated KV cache list (only returned if kv_cache is not None)
        """
        X = self.in_proj(X)
        X = self.drop(X)
        
        if kv_cache is not None:
            # Use KV cache - process each block sequentially
            new_kv_cache = []
            for i, block in enumerate(self.blocks):
                layer_cache = kv_cache[i]
                X, layer_new_cache = block(X, kv_cache=layer_cache, cache_index=cache_index)
                new_kv_cache.append(layer_new_cache)
            
            X = self.ln_f(X)
            return X, new_kv_cache
        else:
            # No KV cache - standard forward pass
            for block in self.blocks:
                X, _ = block(X, kv_cache=None, cache_index=None)
            
            X = self.ln_f(X)
            return X
 