import os
import jax
import jax.numpy as jnp
from jax.scipy.special import digamma
import flax.nnx as nnx
import optax
from orbax import checkpoint as ocp
from recon_module.transformer import GPT, GPTConfig

#from https://github.com/teddykoker/evidential-learning-pytorch/tree/main


def build_model(config, task, is_kv=False, rngs=None):
    """JAX/Flax NNX version of build_model"""
    if rngs is None:
        rngs = nnx.Rngs(0)
    
    opt_config = config['params']['optimizer']
    scheduler_config = config['params'].get('scheduler', {})
    
    # Create model
    if task == 'classification':
        model = Evidential_reconstruction_classification(is_kv, rngs=rngs, **config['params']['model'])
    elif task == 'no_evidential_classification':
        model = Noevidential_classification(is_kv, rngs=rngs, **config['params']['model'])
    else:
        raise ValueError(f"Unknown task type: {task}")
    
    # Create learning rate schedule
    if config['params'].get('decay_lr', False):
        warmup_steps = scheduler_config.get('warmup_iters', 100)
        decay_steps = scheduler_config.get('lr_decay_iters', 10000)
        init_lr = opt_config['lr']
        end_lr = scheduler_config.get('min_lr', 1e-6)
        
        lr_schedule = optax.warmup_cosine_decay_schedule(
            init_value=0.0,
            peak_value=init_lr,
            warmup_steps=warmup_steps,
            decay_steps=decay_steps,
            end_value=end_lr,
        )
    else:
        lr_schedule = opt_config['lr']
    
    # Create optimizer with schedule
    tx = optax.chain(
        optax.clip_by_global_norm(config['params'].get('grad_clip', 1.0)),
        optax.adamw(learning_rate=lr_schedule, weight_decay=opt_config.get('weight_decay', 0.0),),
    )
    optimizer = nnx.Optimizer(model, tx,wrt=nnx.Param)
    
    # Load checkpoint if exists
    total_iter = 0
    epoch = 0
    iter_num = 0
    best_val_loss = float('1e9')
    
    if config['path'].get('ckpt_path') is not None and os.path.exists(config['path']['ckpt_path']):
        print("Loading model from checkpoint:", config['path']['ckpt_path'])
        checkpointer = ocp.StandardCheckpointer()
        ckpt = checkpointer.restore(os.path.abspath(config['path']['ckpt_path']))
        # Restore model and optimizer states (with partial loading for compatibility)
        model_graphdef, model_abstract = nnx.split(model)
        model_pure_dict = nnx.to_pure_dict(model_abstract)
        ckpt_model = ckpt['model']
        # Only update keys that exist in both checkpoint and current model
        def update_nested_dict(target, source):
            for key in source:
                if key in target:
                    if isinstance(source[key], dict) and isinstance(target[key], dict):
                        update_nested_dict(target[key], source[key])
                    else:
                        target[key] = source[key]
        update_nested_dict(model_pure_dict, ckpt_model)
        nnx.replace_by_pure_dict(model_abstract, model_pure_dict)
        model = nnx.merge(model_graphdef, model_abstract)
        
        opt_graphdef, opt_abstract = nnx.split(optimizer)
        opt_pure_dict = nnx.to_pure_dict(opt_abstract)
        ckpt_opt = ckpt['optimizer']
        update_nested_dict(opt_pure_dict, ckpt_opt)
        nnx.replace_by_pure_dict(opt_abstract, opt_pure_dict)
        optimizer = nnx.merge(opt_graphdef, opt_abstract)
        #nnx.update(model, nnx.State.from_pure_dict(ckpt['model']))
        #nnx.update(optimizer, nnx.State.from_pure_dict(ckpt['optimizer']))
        total_iter = ckpt.get('total_iter', 0)
        epoch = ckpt.get('epoch', 0)
        iter_num = ckpt.get('iter', 0)
        best_val_loss = ckpt.get('best_val_loss', float('1e9'))
    
    return model, optimizer, total_iter, epoch, iter_num, best_val_loss

# class NormalInvGamma(nn.Module):
#     def __init__(self, in_features, out_units):
#         super().__init__()
#         self.dense = nn.Linear(in_features, out_units * 4)
#         self.out_units = out_units

#     def evidence(self, x):
#         return F.softplus(x)

#     def forward(self, x):
#         out = self.dense(x)
#         mu, logv, logalpha, logbeta = torch.split(out, self.out_units, dim=-1)
#         v = self.evidence(logv)
#         alpha = self.evidence(logalpha) + 1
#         beta = self.evidence(logbeta)
#         return mu, v, alpha, beta
    
# def nig_nll(gamma, v, alpha, beta, y):
#     two_beta_lambda = 2 * beta * (1 + v)
#     t1 = 0.5 * (torch.pi / v).log()
#     t2 = alpha * two_beta_lambda.log()
#     t3 = (alpha + 0.5) * (v * (y - gamma) ** 2 + two_beta_lambda).log()
#     t4 = alpha.lgamma()
#     t5 = (alpha + 0.5).lgamma()
#     nll = t1 - t2 + t3 + t4 - t5
#     return nll.mean()

class Dirichlet(nnx.Module):
    def __init__(self, in_features, out_units, rngs):
        super().__init__()
        self.dense = nnx.Linear(in_features, out_units, rngs=rngs)
        self.out_units = out_units

    def evidence(self, x):
        return nnx.softplus(x)

    def __call__(self, x):
        out = self.dense(x)
        alpha = self.evidence(out) + 1
        return alpha

# Eq. (5) from https://arxiv.org/abs/1806.01768:
# Sum of squares loss

def dirichlet_reg(alpha: jnp.ndarray, y: jnp.ndarray):
    # dirichlet parameters after removal of non-misleading evidence (from the label)
    alpha = y + (1 - y) * alpha

    # uniform dirichlet distribution
    beta = jnp.ones_like(alpha)

    sum_alpha = alpha.sum(-1)
    sum_beta = beta.sum(-1)

    # JAX version: use jax.lax.lgamma instead of torch lgamma
    t1 = jax.lax.lgamma(sum_alpha) - jax.lax.lgamma(sum_beta)
    t2 = (jax.lax.lgamma(alpha) - jax.lax.lgamma(beta)).sum(-1)
    t3 = alpha - beta
    # JAX version: use jax.scipy.special.digamma and expand_dims
    t4 = digamma(alpha) - jnp.expand_dims(digamma(sum_alpha), -1)

    kl = t1 - t2 + (t3 * t4).sum(-1)
    return kl.mean()


def dirichlet_mse(alpha: jnp.ndarray, y: jnp.ndarray):
    sum_alpha = alpha.sum(-1, keepdims=True)
    p = alpha / sum_alpha
    t1 = jnp.power(y - p, 2).sum(-1)
    t2 = ((p * (1 - p)) / (sum_alpha + 1)).sum(-1)
    mse = t1 + t2
    return mse.mean()


def evidential_classification(alpha, y, lamb=0.01):
    num_classes = alpha.shape[-1]
    y = jax.nn.one_hot(y, num_classes)
    mse = dirichlet_mse(alpha, y)
    reg = dirichlet_reg(alpha, y)
    total = mse + lamb * reg
    return total, mse, reg


# def evidential_regression(dist_params, y, lamb=1.0):
#     return nig_nll(*dist_params, y) + lamb * nig_reg(*dist_params, y)

# # Normal Inverse Gamma regularization
# # from https://arxiv.org/abs/1910.02600:
# # > we formulate a novel evidence regularizer, L^R_i
# # > scaled on the error of the i-th prediction
# def nig_reg(gamma, v, alpha, _beta, y):
#     reg = (y - gamma).abs() * (2 * v + alpha)
#     return reg.mean()

# def evidential_regression(dist_params, y, lamb=1.0):
#     nll= nig_nll(*dist_params, y)
#     reg = nig_reg(*dist_params, y)
#     mse = (y - dist_params[0]).pow(2).mean()
#     total = nll + lamb * reg + 10*mse  # adding mse loss for stability
#     return total, nll, reg, mse

# class Evidential_reconstruction(nn.Module):
#     def __init__(self, is_kv,in_features, out_units,gpt_config,**kwargs):
#         super().__init__()
#         self.nig = NormalInvGamma(in_features, out_units)
#         data_gpt_config = GPTConfig(**gpt_config)
#         self.gpt = GPT(data_gpt_config,is_kv)

#     def uncertainty(self,mu, v, alpha, beta):
#         epistemic_uncertainty = beta / (v * (alpha - 1))
#         aleatoric_uncertainty = beta / (alpha - 1)
#         return epistemic_uncertainty, aleatoric_uncertainty

#     def forward(self, x,y=None):
#         x = self.gpt(x)
#         mu, v, alpha, beta = self.nig(x)
#         if y is not None:
#             total_loss,nll_loss,reg,mse_loss = evidential_regression((mu, v, alpha, beta), y)
#             return (mu, v, alpha, beta), (total_loss, nll_loss, reg, mse_loss)
#         else:
#             epistemic_uncertainty, aleatoric_uncertainty = self.uncertainty(mu, v, alpha, beta)
#             return mu, epistemic_uncertainty, aleatoric_uncertainty

class Evidential_reconstruction_classification(nnx.Module):
    def __init__(self, is_kv, in_features, out_units, num_classes, rngs, gpt_config):
        super().__init__()
        self.dri = Dirichlet(in_features, out_units * num_classes, rngs)
        self.num_classes = num_classes
        self.out_units = out_units
        self.is_kv = is_kv
        data_gpt_config = GPTConfig(**gpt_config)
        self.gpt = GPT(data_gpt_config, rng=rngs, is_kv=is_kv)

    def uncertainty(self, alpha):
        S = alpha.sum(axis=-1, keepdims=True)
        epistemic_uncertainty = self.num_classes / S
        p = alpha / S
        aleatoric_uncertainty = (p * (1 - p) / (1 + S)).sum(axis=-1, keepdims=True)
        return epistemic_uncertainty, aleatoric_uncertainty

    def init_kv_cache(self, batch_size):
        """Initialize KV cache for inference."""
        return self.gpt.init_kv_cache(batch_size)

    def __call__(self, x, y=None, kv_cache=None, cache_index=None):
        """
        Forward pass with optional KV cache support.
        
        Args:
            x: input tensor (B, T, input_size)
            y: target tensor for training, or None for inference
            kv_cache: List of (k_cache, v_cache) for each layer, or None
            cache_index: current position in cache (scalar int), required when using kv_cache
        
        Returns:
            For training (y is not None):
                mu, alpha_reshaped, (total_loss, mse_loss, reg_loss)
            For inference with kv_cache:
                prob, epistemic_uncertainty, aleatoric_uncertainty, new_kv_cache
            For inference without kv_cache:
                prob, epistemic_uncertainty, aleatoric_uncertainty
        """
        if kv_cache is not None:
            x, new_kv_cache = self.gpt(x, kv_cache=kv_cache, cache_index=cache_index)
        else:
            x = self.gpt(x)
            new_kv_cache = None
        
        raw_alpha = self.dri(x)
        alpha = raw_alpha.reshape(-1, self.num_classes)
        alpha_reshaped = raw_alpha.reshape(*x.shape[:-1], self.out_units, self.num_classes)
        S = alpha_reshaped.sum(axis=-1, keepdims=True)
        prob = alpha_reshaped / S  # probability for each class
        mu = jnp.argmax(prob, axis=-1)  # predicted class indices
        
        if y is not None:
            total_loss, mse_loss, reg_loss = evidential_classification(alpha, y.flatten())
            return mu, alpha_reshaped, (total_loss, mse_loss, reg_loss)
        else:
            epistemic_uncertainty, aleatoric_uncertainty = self.uncertainty(alpha_reshaped)
            if new_kv_cache is not None:
                return prob, epistemic_uncertainty, aleatoric_uncertainty, new_kv_cache
            else:
                return prob, epistemic_uncertainty, aleatoric_uncertainty

class Noevidential_classification(nnx.Module):
    def __init__(self, is_kv, in_features, out_units, num_classes, rngs, gpt_config):
        super().__init__()
        self.out_proj = nnx.Linear(in_features, out_units * num_classes, rngs=rngs)
        self.num_classes = num_classes
        self.out_units = out_units
        data_gpt_config = GPTConfig(**gpt_config)
        self.gpt = GPT(data_gpt_config, rng=rngs)

    def __call__(self, x, y=None):
        x = self.gpt(x)
        logit = self.out_proj(x)
        mu = logit.reshape(*x.shape[:-1], self.out_units, self.num_classes)
        mu = jnp.argmax(mu, axis=-1)
        if y is not None:
            # JAX version: use optax.softmax_cross_entropy_with_integer_labels
            logit_flat = logit.reshape(-1, self.num_classes)
            y_flat = y.flatten().astype(jnp.int32)
            loss = optax.softmax_cross_entropy_with_integer_labels(logit_flat, y_flat).mean()
        else:
            loss = None
        return mu, loss

