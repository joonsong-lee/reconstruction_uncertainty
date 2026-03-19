import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.8"  # GPU 메모리 80%만 사용

import jax
import jax.numpy as jnp
import flax.nnx as nnx
import optax
import numpy as np
from functools import partial
from tqdm import tqdm
import wandb
from omegaconf import OmegaConf
from orbax import checkpoint as ocp
import distrax

from jaxmarl import make
from jaxmarl.environments.overcooked_v2.overcooked import OvercookedV2
from jaxmarl.environments.overcooked_v2.layouts import Layout

from alg.jax_ppo import ActorCriticRNN, ActorCriticRNN_combine, ScannedRNN
from recon_module.evdiential import build_model, evidential_classification
from envs.ovc.visualizer import seq_to_seq_viz


class OnlineBufferJax:
    """JAX-compatible online buffer for reconstruction training."""
    def __init__(self, max_len=2000):
        self.max_len = max_len
        self.ptr = 0
        self.size = 0
        self.storage = [None] * max_len

    def push(self, obs, g):
        """Push batch of observations and global states."""
        # obs: (batch, seq, features), g: (batch, seq, features)
        for i in range(obs.shape[0]):
            data = (np.array(obs[i]), np.array(g[i]))
            self._add_single(data)

    def _add_single(self, data):
        self.storage[self.ptr] = data
        self.ptr = (self.ptr + 1) % self.max_len
        self.size = min(self.size + 1, self.max_len)

    def sample(self, batch_size):
        indices = np.random.choice(self.size, batch_size, replace=False)
        batch = [self.storage[idx] for idx in indices]
        batch_obs, batch_g = zip(*batch)
        return jnp.stack(batch_obs), jnp.stack(batch_g)

    def __len__(self):
        return self.size


def load_network_from_checkpoint(ckpt_path, config, rngs):
    """Load network from checkpoint saved with CheckpointManager."""
    network = ActorCriticRNN(**config['params']['model'], rngs=rngs)
    
    if ckpt_path is not None and os.path.exists(ckpt_path):
        print(f"Loading checkpoint from: {ckpt_path}")
        
        checkpointer = ocp.AsyncCheckpointer(ocp.StandardCheckpointHandler())
        manager = ocp.CheckpointManager(
            os.path.abspath(ckpt_path),
            checkpointer,
            ocp.CheckpointManagerOptions(max_to_keep=2, create=False)
        )
        
        latest_step = manager.latest_step()
        if latest_step is not None:
            print(f"Restoring from step: {latest_step}")
            
            tx = optax.chain(
                optax.clip_by_global_norm(config['params'].get('max_grad_norm', 0.5)),
                optax.adamw(**config["params"]["optimizer"]),
            )
            dummy_optimizer = nnx.Optimizer(network, tx, wrt=nnx.Param)
            
            current_state = nnx.state(network)
            optimizer_state = nnx.state(dummy_optimizer)
            template = {
                'model': current_state.to_pure_dict(),
                'optimizer': optimizer_state.to_pure_dict(),
                'iter': 0,
            }
            
            ckpt = manager.restore(
                latest_step,
                args=ocp.args.StandardRestore(template)
            )
            
            restored_params = ckpt['model']
            graphdef, abstract_state = nnx.split(network)
            nnx.replace_by_pure_dict(abstract_state, restored_params)
            network = nnx.merge(graphdef, abstract_state)
            
            print("Checkpoint loaded successfully!")
        else:
            print("No checkpoint steps found in manager.")
    else:
        print("No checkpoint found, using randomly initialized network.")
    
    return network


def load_partner_networks(partner_dir, config, rngs, num_partners):
    """Load multiple partner networks from checkpoints."""
    partner_networks = []
    partner_keys = ['best_model', 'iter_saver']
    
    for i in range(num_partners):
        for k in partner_keys:
            ckpt_path = os.path.join(partner_dir, f'{k}_{i*1000}')
            if os.path.exists(ckpt_path):
                rng_key = jax.random.fold_in(rngs(), i * len(partner_keys) + partner_keys.index(k))
                partner_rngs = nnx.Rngs(rng_key)
                partner = load_network_from_checkpoint(ckpt_path, config, partner_rngs)
                partner_networks.append(partner)
                print(f"Loaded partner {i} checkpoint {k}")
    
    return partner_networks


@jax.jit
def compute_gae(rewards, values, dones, gamma=0.99, lambda_=0.95):
    """Compute Generalized Advantage Estimation."""
    T = rewards.shape[0]
    lastgaelam = jnp.zeros(rewards.shape[1])
    
    values_padded = jnp.concatenate([values, jnp.zeros((1,) + values.shape[1:])], axis=0)
    dones_padded = jnp.concatenate([dones, jnp.ones((1,) + dones.shape[1:])], axis=0)
    
    def gae_step(lastgaelam, t):
        idx = T - 1 - t
        nextnonterminal = 1.0 - dones_padded[idx + 1]
        nextvalue = values_padded[idx + 1]
        delta = rewards[idx] + gamma * nextvalue * nextnonterminal - values[idx]
        lastgaelam = delta + gamma * lambda_ * nextnonterminal * lastgaelam
        return lastgaelam, lastgaelam
    
    _, advantages_reversed = jax.lax.scan(gae_step, lastgaelam, jnp.arange(T))
    advantages = advantages_reversed[::-1]
    
    returns = advantages + values
    return returns, advantages


def make_fcp_and_recon_local_train(config, partner_config, recon_config):
    """Create FCP + Recon + Local joint training function using ActorCriticRNN_combine."""
    env = OvercookedV2(
        layout='demo_cook_simple', 
        agent_view_size=2,
        random_agent_positions=True, 
        random_reset=False,
        max_steps=config['params']['max_episode_length'],
        sample_recipe_on_delivery=True,
        negative_rewards=True
    )
    
    batch_size = config['params']['batch_size']
    episode_length = config['params']['max_episode_length']
    
    # Get observation shape from env
    obs_shape = env._get_obs_shape()  # (H, W, C) for local obs
    obs_h, obs_w, obs_c = obs_shape
    # Global observation shape (for recon)
    global_obs_shape = (env.height, env.width, 40 + env.indicate_successful_delivery)
    global_h, global_w, global_c = global_obs_shape

    def run_episode(rng, agent, recon_model, partner_networks):
        """Run single episode collecting data for FCP + Recon + Local training.
        
        Note: Always uses KV cache for faster inference and JIT compatibility.
        """
        num_partners = len(partner_networks)
        partner_lstm_dim = partner_config['params']['model']['lstm_dim']
        
        def get_partner_action_all(partner_hstate_h, partner_hstate_c, obs, rng_key):
            """Run ALL partners and return stacked results.
            
            This avoids calling nnx.Module inside jax.lax.switch.
            """
            all_new_h = []
            all_new_c = []
            all_actions = []
            
            for i in range(num_partners):
                hstate = (partner_hstate_h[jnp.newaxis, :], partner_hstate_c[jnp.newaxis, :])
                new_hstate, pi, _ = partner_networks[i](
                    hstate, obs[jnp.newaxis, jnp.newaxis, ...]
                )
                action = pi.sample(seed=jax.random.fold_in(rng_key, i)).squeeze()
                all_new_h.append(new_hstate[0].squeeze(0))
                all_new_c.append(new_hstate[1].squeeze(0))
                all_actions.append(action)
            
            return jnp.stack(all_new_h), jnp.stack(all_new_c), jnp.stack(all_actions)
        
        def select_partner_result(partner_idx, all_h, all_c, all_actions):
            """Select result for specific partner index."""
            return all_h[partner_idx], all_c[partner_idx], all_actions[partner_idx]
        
        def get_partner_action(partner_idx, partner_hstate_h, partner_hstate_c, obs, rng_key):
            """Get action from a specific partner given its index."""
            all_h, all_c, all_actions = get_partner_action_all(
                partner_hstate_h, partner_hstate_c, obs, rng_key
            )
            return select_partner_result(partner_idx, all_h, all_c, all_actions)
        
        def _env_step(carry, step_idx):
            """Single environment step with KV cache support."""
            env_state, last_obs, agent_hstate, partner_hstates, kv_cache, rng, done, partner_ids = carry

            rng, _rng = jax.random.split(rng)
            obs_agent1 = last_obs['agent_1']  # (batch, H, W, C) - our learning agent's local obs
            
            # Flatten obs for recon model input
            obs_flat = obs_agent1.reshape(batch_size, 1, -1)  # (batch, 1, H*W*C)
            
            # Apply reconstruction model with KV cache (always use for JIT compatibility)
            recon_prob, epistemic_unc, aleatoric_unc, new_kv_cache = recon_model(
                obs_flat, kv_cache=kv_cache, cache_index=step_idx
            )
            
            # Convert probability to predicted class (reconstruction output)
            recon_mu = jnp.argmax(recon_prob, axis=-1)  # (batch, 1, out_units)
            recon_obs = recon_mu.squeeze(1).reshape(batch_size, global_h, global_w, global_c)
            recon_obs = recon_obs.astype(jnp.float32)
            
            # Reshape uncertainties
            epistemic_unc = epistemic_unc.squeeze(1).squeeze(-1).reshape(batch_size, global_h, global_w, global_c)
            aleatoric_unc = aleatoric_unc.squeeze(1).squeeze(-1).reshape(batch_size, global_h, global_w, global_c)
            
            # Mean over channel dimension to get single channel uncertainties
            epistemic_unc = epistemic_unc.mean(axis=-1, keepdims=True)  # (batch, H, W, 1)
            aleatoric_unc = aleatoric_unc.mean(axis=-1, keepdims=True)  # (batch, H, W, 1)
            
            # x1: Recon obs + uncertainties (batch, global_H, global_W, C+2)
            recon_input = jnp.concatenate([recon_obs, epistemic_unc, aleatoric_unc], axis=-1)
            
            # x2: Local obs (batch, obs_H, obs_W, C)
            local_input = obs_agent1
            
            # Forward pass through agent network (ActorCriticRNN_combine)
            # agent takes x1 (recon) and x2 (local) as separate inputs
            agent_hstate, pi, value = agent(agent_hstate, recon_input[jnp.newaxis, :], local_input[jnp.newaxis, :])
            
            action_agent1 = pi.sample(seed=_rng).squeeze(0)
            log_prob = pi.log_prob(action_agent1).squeeze(0)
            value = value.squeeze(0)
            
            # SELECT ACTION for agent_0 (partner)
            rng, _rng = jax.random.split(rng)
            obs_agent0 = last_obs['agent_0']
            rng_keys = jax.random.split(_rng, batch_size)
            
            partner_h, partner_c = partner_hstates
            new_partner_h, new_partner_c, action_agent0 = jax.vmap(get_partner_action)(
                partner_ids, partner_h, partner_c, obs_agent0, rng_keys
            )
            partner_hstates_new = (new_partner_h, new_partner_c)
            
            # Prepare actions for environment
            env_act = {
                'agent_0': action_agent0,
                'agent_1': action_agent1
            }

            # STEP ENV
            rng, _rng = jax.random.split(rng)
            rng_step = jax.random.split(_rng, batch_size)

            obsv, next_env_state, reward, next_done, info = jax.vmap(
                env.step, in_axes=(0, 0, 0)
            )(rng_step, env_state, env_act)
            
            # Get global observation for reconstruction target
            global_obs = jax.vmap(env.get_obs_default, in_axes=(0,))(env_state)
            global_agent1 = global_obs[:, 1]

            # Store outputs
            transition = {
                'recon_obs': recon_input,  # Recon input for agent (x1)
                'local_obs': local_input,  # Local obs for agent (x2)
                'raw_obs': obs_agent1,  # Raw obs for recon buffer
                'global_obs': global_agent1,  # Global obs for recon target
                'action': action_agent1,
                'reward': reward['agent_1'],
                'shaped_reward': info['shaped_reward']['agent_1'],
                'log_prob': log_prob,
                'value': value,
                'done': next_done['__all__'],
            }
            
            new_carry = (next_env_state, obsv, agent_hstate, partner_hstates_new, new_kv_cache, rng, next_done['__all__'], partner_ids)
            return new_carry, transition

        # Reset environments
        rng, _reset_rng = jax.random.split(rng)
        reset_keys = jax.random.split(_reset_rng, batch_size)
        initial_obs, initial_env_state = jax.vmap(env.reset, in_axes=(0,))(reset_keys)
        
        # Initialize hidden states (combine uses lstm_dim * 2)
        agent_lstm_dim = agent.rnn.hidden_size
        agent_hstate = ScannedRNN.initialize_carry(batch_size, agent_lstm_dim)
        partner_hstate = ScannedRNN.initialize_carry(batch_size, partner_lstm_dim)
        
        # Initialize KV cache for recon model (always use for JIT compatibility)
        kv_cache = recon_model.init_kv_cache(batch_size)
        
        # Random partner selection
        rng, _rng = jax.random.split(rng)
        partner_ids = jax.random.randint(_rng, (batch_size,), 0, num_partners)
        
        # Initial carry
        initial_done = jnp.zeros((batch_size,), dtype=jnp.bool_)
        initial_carry = (initial_env_state, initial_obs, agent_hstate, partner_hstate, kv_cache, rng, initial_done, partner_ids)
        
        # Run episode with step indices
        _, transitions = nnx.scan(
            lambda carry, step_idx: _env_step(carry, step_idx),
            in_axes=(nnx.Carry, 0),
            out_axes=(nnx.Carry, 0),
            length=episode_length
        )(initial_carry, jnp.arange(episode_length))
        
        return transitions

    # JIT compile the episode runner for faster execution
    run_episode_jit = nnx.jit(run_episode)

    return run_episode_jit, env


@nnx.jit
def ppo_update_step_combine(agent, optimizer, recon_obs, local_obs, actions, old_log_probs, advantages, returns, clip_eps=0.2, entropy_coef=0.01):
    """Single PPO update step for FCP agent using ActorCriticRNN_combine.
    
    Args:
        recon_obs: (T, batch, H, W, C) - reconstruction observation (x1)
        local_obs: (T, batch, H, W, C) - local observation (x2)
        actions: (T, batch)
        old_log_probs: (T, batch)
        advantages: (T, batch)
        returns: (T, batch)
    """
    def loss_fn(agent):
        batch_size = recon_obs.shape[1]
        lstm_dim = agent.rnn.hidden_size
        hstate = ScannedRNN.initialize_carry(batch_size, lstm_dim)
        
        def step_fn(hstate, inputs):
            recon_t, local_t = inputs
            hstate, pi, value = agent(hstate, recon_t[jnp.newaxis, :], local_t[jnp.newaxis, :])
            return hstate, (pi.logits.squeeze(0), value.squeeze(0))
        
        _, (all_logits, all_values) = jax.lax.scan(step_fn, hstate, (recon_obs, local_obs))
        
        # Clip logits to prevent NaN
        all_logits = jnp.clip(all_logits, -20.0, 20.0)
        
        # Policy loss
        pi = distrax.Categorical(logits=all_logits)
        log_probs = pi.log_prob(actions)
        
        # Clip log_probs difference to prevent extreme ratios
        log_ratio = jnp.clip(log_probs - old_log_probs, -10.0, 10.0)
        ratio = jnp.exp(log_ratio)
        clipped_ratio = jnp.clip(ratio, 1 - clip_eps, 1 + clip_eps)
        policy_loss = -jnp.minimum(ratio * advantages, clipped_ratio * advantages).mean()
        
        # Value loss with clipping
        value_loss = ((all_values - returns) ** 2).mean()
        
        # Entropy bonus
        entropy = pi.entropy().mean()
        
        total_loss = policy_loss + 0.5 * value_loss - entropy_coef * entropy
        return total_loss, (policy_loss, value_loss, entropy)
    
    grad_fn = nnx.value_and_grad(loss_fn, has_aux=True)
    (total_loss, (policy_loss, value_loss, entropy)), grads = grad_fn(agent)
    
    # Clip gradients to prevent NaN propagation
    grads = jax.tree.map(lambda g: jnp.clip(g, -1.0, 1.0), grads)
    
    optimizer.update(agent, grads)
    
    return total_loss, policy_loss, value_loss, entropy


@nnx.jit
def recon_train_step(recon_model, recon_optimizer, X, Y, lamb=0.01):
    """Single reconstruction model training step."""
    def loss_fn(model):
        mu, alpha_reshaped, (total_loss, mse_loss, reg_loss) = model(X, Y)
        return total_loss, (mse_loss, reg_loss)
    
    grad_fn = nnx.value_and_grad(loss_fn, has_aux=True)
    (total_loss, (mse_loss, reg_loss)), grads = grad_fn(recon_model)
    recon_optimizer.update(recon_model, grads)
    
    return total_loss, mse_loss, reg_loss


def main():
    # Load config
    config = OmegaConf.load('./configs/ovc_demo.yaml')
    partner_config = config.partners
    recon_config = config.recon_module
    config = config.fcp_recon_local  # Use fcp_recon_local config
    recon_config['path']['ckpt_path'] = config['path']['recon_ckpt_path']
    # Update recon config batch size
    recon_config['params']['model']['gpt_config']['batch_size'] = config['params']['batch_size']
    
    config = OmegaConf.to_container(config, resolve=True)
    partner_config = OmegaConf.to_container(partner_config, resolve=True)
    recon_config = OmegaConf.to_container(recon_config, resolve=True)
    
    # Initialize wandb
    wandb.init(**config['wandb'], config=config)
    
    # Setup
    base_seed = config['params']['base_seed']
    rng = jax.random.PRNGKey(base_seed)
    
    # Create directories
    os.makedirs(config['path']['out_dir'], exist_ok=True)
    
    # Initialize online buffer for reconstruction
    buffer = OnlineBufferJax(config['params']['buffer_size'])
    
    # Initialize models
    rng, init_rng = jax.random.split(rng)
    rngs = nnx.Rngs(init_rng)
    
    # Load reconstruction model (trainable)
    rng, recon_rng = jax.random.split(rng)
    recon_rngs = nnx.Rngs(recon_rng)
    recon_model, recon_optimizer, _, _, _, _ = build_model(
        recon_config, 
        task='classification', 
        is_kv=True, 
        rngs=recon_rngs
    )
    
    # Create FCP-Recon-Local agent using ActorCriticRNN_combine
    agent = ActorCriticRNN_combine(**config['params']['model'], rngs=rngs)
    lstm_dim = config['params']['model']['lstm_dim']
    
    # Create optimizer for FCP agent
    tx = optax.chain(
        optax.clip_by_global_norm(config['params']['max_grad_norm']),
        optax.adamw(**config['params']['optimizer']),
    )
    optimizer = nnx.Optimizer(agent, tx, wrt=nnx.Param)
    
    # Load checkpoint if exists
    start_iter = 0
    if config['path'].get('ckpt_path') is not None and os.path.exists(config['path']['ckpt_path']):
        print(f"Loading agent checkpoint from: {config['path']['ckpt_path']}")
        checkpointer = ocp.StandardCheckpointer()
        ckpt = checkpointer.restore(config['path']['ckpt_path'])
        
        graphdef, abstract_state = nnx.split(agent)
        nnx.replace_by_pure_dict(abstract_state, ckpt['model'])
        agent = nnx.merge(graphdef, abstract_state)
        
        graphdef_opt, abstract_state_opt = nnx.split(optimizer)
        nnx.replace_by_pure_dict(abstract_state_opt, ckpt['optimizer'])
        optimizer = nnx.merge(graphdef_opt, abstract_state_opt)
        
        start_iter = ckpt.get('iter', 0) + 1
    
    # Load partner networks
    rng, partner_rng = jax.random.split(rng)
    partner_rngs = nnx.Rngs(partner_rng)
    partner_networks = load_partner_networks(
        config['path']['partner_dir'],
        partner_config,
        partner_rngs,
        config['params']['num_partners']
    )
    
    # Create training function
    run_episode_fn, env = make_fcp_and_recon_local_train(config, partner_config, recon_config)
    
    # Checkpoint manager
    checkpointer = ocp.StandardCheckpointer()
    
    # Training loop
    best_return = -1e9
    best_recon_loss = 1e9
    
    for iter_num in tqdm(range(start_iter, config['params']['num_iterations'])):
        rng, train_rng = jax.random.split(rng)
        
        # Collect episode (always uses KV cache for fast rollout inference)
        transitions = run_episode_fn(train_rng, agent, recon_model, partner_networks)
        
        # Push data to reconstruction buffer
        raw_obs = transitions['raw_obs']
        global_obs = transitions['global_obs']
        
        raw_obs_flat = raw_obs.transpose(1, 0, 2, 3, 4).reshape(
            config['params']['batch_size'], config['params']['max_episode_length'], -1
        )
        global_obs_flat = global_obs.transpose(1, 0, 2, 3, 4).reshape(
            config['params']['batch_size'], config['params']['max_episode_length'], -1
        )
        buffer.push(raw_obs_flat, global_obs_flat)
        
        # Compute returns and advantages for PPO
        rewards = transitions['reward'] + transitions['shaped_reward']
        returns, advantages = compute_gae(
            rewards,
            transitions['value'],
            transitions['done'].astype(jnp.float32),
            gamma=config['params']['loss']['gamma'],
            lambda_=config['params']['loss']['lambda_'],
        )
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # PPO updates with both recon_obs and local_obs
        for _ in range(10):
            total_loss, policy_loss, value_loss, entropy = ppo_update_step_combine(
                agent, optimizer,
                transitions['recon_obs'],
                transitions['local_obs'],
                transitions['action'],
                transitions['log_prob'],
                advantages,
                returns,
                entropy_coef=config['params']['loss']['entropy_weight'],
            )
        
        # Reconstruction model training
        if buffer.size >= config['params']['buffer_size']:
            X, Y = buffer.sample(config['params']['batch_size'])
            recon_total_loss, recon_mse, recon_reg = recon_train_step(
                recon_model, recon_optimizer, X, Y
            )
        else:
            recon_total_loss, recon_mse, recon_reg = 1.0, 1.0, 1.0
        
        # Logging
        mean_reward = transitions['reward'].sum(axis=0).mean()
        
        if (iter_num + 1) % config['log_interval'] == 0:
            log_dict = {
                'fcp_recon_local_return': float(mean_reward),
                'fcp_recon_local_loss_actions': float(policy_loss),
                'fcp_recon_local_loss_values': float(value_loss),
                'fcp_recon_local_loss_entropy': float(entropy),
                'fcp_recon_local_total_loss': float(recon_total_loss),
                'fcp_recon_local_reg': float(recon_reg),
                'fcp_recon_local_mse': float(recon_mse),
            }
            wandb.log(log_dict, step=iter_num)
        
        # Save checkpoints
        if (iter_num + 1) % config['params']['save_interval'] == 0:
            # Save FCP agent
            agent_ckpt = {
                'model': nnx.state(agent).to_pure_dict(),
                'optimizer': nnx.state(optimizer).to_pure_dict(),
                'iter': iter_num,
            }
            checkpointer.save(
                os.path.abspath(os.path.join(config['path']['out_dir'], f'model_iter_{iter_num}')),
                agent_ckpt,
                force=True
            )
            
            # Save recon model
            recon_ckpt = {
                'model': nnx.state(recon_model).to_pure_dict(),
                'optimizer': nnx.state(recon_optimizer).to_pure_dict(),
            }
            checkpointer.save(
                os.path.abspath(os.path.join(config['path']['out_dir'], f'recon_model_iter_{iter_num}')),
                recon_ckpt,
                force=True
            )
            
            # Visualization
            if buffer.size >= config['params']['batch_size']:
                X_viz, Y_viz = buffer.sample(1)
                prob, epistemic_unc, aleatoric_unc = recon_model(X_viz)
                mu_pred = jnp.argmax(prob, axis=-1)
                
                global_c = 40 + env.indicate_successful_delivery
                
                y_obs = np.array(Y_viz[0])
                mu_sample = np.array(mu_pred[0])
                aleatoric_sample = np.array(aleatoric_unc[0].squeeze(-1).reshape(-1, env.height, env.width, global_c).mean(axis=-1))
                epistemic_sample = np.array(epistemic_unc[0].squeeze(-1).reshape(-1, env.height, env.width, global_c).mean(axis=-1))
                
                seq_to_seq_viz(
                    env, y_obs, mu_sample, aleatoric_sample, epistemic_sample,
                    filename=os.path.join(config['path']['out_dir'], f'recon_iter_{iter_num}')
                )
        
        # Save best FCP model
        if mean_reward > best_return:
            best_return = float(mean_reward)
            agent_ckpt = {
                'model': nnx.state(agent).to_pure_dict(),
                'optimizer': nnx.state(optimizer).to_pure_dict(),
                'iter': iter_num,
            }
            checkpointer.save(
                os.path.abspath(os.path.join(config['path']['out_dir'], 'best_model')),
                agent_ckpt,
                force=True
            )
        
        # Save best recon model
        if isinstance(recon_total_loss, float) and recon_total_loss < best_recon_loss:
            best_recon_loss = recon_total_loss
            recon_ckpt = {
                'model': nnx.state(recon_model).to_pure_dict(),
                'optimizer': nnx.state(recon_optimizer).to_pure_dict(),
            }
            checkpointer.save(
                os.path.abspath(os.path.join(config['path']['out_dir'], 'best_recon_model')),
                recon_ckpt,
                force=True
            )
    
    # Save final models
    agent_ckpt = {
        'model': nnx.state(agent).to_pure_dict(),
        'optimizer': nnx.state(optimizer).to_pure_dict(),
        'iter': iter_num,
    }
    checkpointer.save(
        os.path.abspath(os.path.join(config['path']['out_dir'], 'final_model')),
        agent_ckpt,
        force=True
    )
    
    recon_ckpt = {
        'model': nnx.state(recon_model).to_pure_dict(),
        'optimizer': nnx.state(recon_optimizer).to_pure_dict(),
    }
    checkpointer.save(
        os.path.abspath(os.path.join(config['path']['out_dir'], 'final_recon_model')),
        recon_ckpt,
        force=True
    )
    
    wandb.finish()


if __name__ == "__main__":
    main()
