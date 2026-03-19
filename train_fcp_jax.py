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
from jaxmarl.environments import overcooked_v2_layouts
from jaxmarl.environments.overcooked_v2.layouts import Layout

from alg.jax_ppo import ActorCriticRNN, ScannedRNN


custom_layout_str = """
WXWWWWWWWXW
0    R    0
1   APA   1
2    R    2
WBWWWWWWWBW
"""


def unbatchify(x: jnp.ndarray, agent_list, num_envs, num_actors):
    x = x.reshape((num_actors, num_envs))
    return {a: x[i] for i, a in enumerate(agent_list)}


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
    """Load multiple partner networks from checkpoints.
    
    Loads best_model and iter_saver (last iteration) for each partner,
    matching the save structure from train_partners_jax.py.
    """
    partner_networks = []
    # Matches train_partners_jax.py save structure: best_model_{i} and iter_saver_{i}
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
    
    # Convert to nnx.List for proper JIT tracing
    return nnx.List(partner_networks)


def make_fcp_train(config, partner_config):
    """Create FCP training function."""
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
    agent_lstm_dim = config['params']['model']['lstm_dim']
    partner_lstm_dim = partner_config['params']['model']['lstm_dim']

    def train(rng, agent, partner_networks, optimizer):
        """
        FCP training loop using pure JAX.
        """
        num_partners = len(partner_networks)
        
        def get_all_partner_actions(partner_hstates, obs, rng_key):
            """Run ALL partners on ALL batch elements, then select per-element.
            
            Args:
                partner_hstates: tuple of (h, c) each with shape (batch, hidden)
                obs: (batch, H, W, C)
                rng_key: random key
            
            Returns:
                all_new_h: (num_partners, batch, hidden)
                all_new_c: (num_partners, batch, hidden)
                all_actions: (num_partners, batch)
            """
            partner_h, partner_c = partner_hstates
            all_new_h = []
            all_new_c = []
            all_actions = []
            
            for i in range(num_partners):
                # hstate: (batch, hidden) -> (1, batch, hidden) for seq dim
                hstate = (partner_h[jnp.newaxis, :, :], partner_c[jnp.newaxis, :, :])
                # obs: (batch, H, W, C) -> (1, batch, H, W, C) for seq dim
                obs_in = obs[jnp.newaxis, :, :, :, :]
                
                new_hstate, pi, _ = partner_networks[i](hstate, obs_in)
                # pi.sample() returns (1, batch), squeeze to get (batch,)
                action_raw = pi.sample(seed=jax.random.fold_in(rng_key, i))
                action = jnp.squeeze(action_raw)  # (batch,)
                
                # new_hstate is ((1, batch, hidden), (1, batch, hidden))
                new_h = jnp.squeeze(new_hstate[0])  # (batch, hidden)
                new_c = jnp.squeeze(new_hstate[1])  # (batch, hidden)
                
                all_new_h.append(new_h)
                all_new_c.append(new_c)
                all_actions.append(action)
            
            # Stack: (num_partners, batch, hidden) or (num_partners, batch)
            return jnp.stack(all_new_h, axis=0), jnp.stack(all_new_c, axis=0), jnp.stack(all_actions, axis=0)
        
        def select_per_batch(partner_ids, all_h, all_c, all_actions):
            """Select partner result for each batch element.
            
            Args:
                partner_ids: (batch,) - index of partner to use for each batch element
                all_h: (num_partners, batch, hidden)
                all_c: (num_partners, batch, hidden)
                all_actions: (num_partners, batch)
            
            Returns:
                selected_h: (batch, hidden)
                selected_c: (batch, hidden)
                selected_actions: (batch,)
            """
            # Transpose to (batch, num_partners, ...) for easier indexing
            all_h_t = jnp.moveaxis(all_h, 0, 1)  # (batch, num_partners, hidden)
            all_c_t = jnp.moveaxis(all_c, 0, 1)  # (batch, num_partners, hidden)
            all_actions_t = jnp.moveaxis(all_actions, 0, 1)  # (batch, num_partners)
            
            # Use vmap to select for each batch element
            def select_single(h_all_partners, c_all_partners, actions_all_partners, pid):
                # h_all_partners: (num_partners, hidden)
                # actions_all_partners: (num_partners,)
                # pid: scalar
                return h_all_partners[pid], c_all_partners[pid], actions_all_partners[pid]
            
            selected_h, selected_c, selected_actions = jax.vmap(select_single)(
                all_h_t, all_c_t, all_actions_t, partner_ids
            )
            return selected_h, selected_c, selected_actions
        
        def _env_step(carry, unused):
            """Single environment step."""
            env_state, last_obs, agent_hstate, partner_hstates, rng, done, partner_ids = carry

            # SELECT ACTION for agent_1 (learning agent)
            rng, _rng = jax.random.split(rng)
            obs_agent1 = last_obs['agent_1']
            
            # Forward pass through agent network
            # obs_agent1: (batch, H, W, C) -> add seq dim: (1, batch, H, W, C)
            agent_hstate, pi, value = agent(agent_hstate, obs_agent1[jnp.newaxis, :])
            # pi.sample() returns (seq, batch) = (1, batch)
            action_agent1 = pi.sample(seed=_rng).squeeze(0)  # (batch,)
            log_prob = pi.log_prob(action_agent1[jnp.newaxis, :]).squeeze(0)  # (batch,)
            value = value.squeeze(0)  # (batch,)
            
            # SELECT ACTION for agent_0 (partner)
            rng, _rng = jax.random.split(rng)
            obs_agent0 = last_obs['agent_0']
            
            # Run ALL partners on ALL batch elements (no vmap)
            all_h, all_c, all_actions = get_all_partner_actions(
                partner_hstates, obs_agent0, _rng
            )
        
            # Select per-batch-element using advanced indexing (no vmap)
            new_partner_h, new_partner_c, action_agent0 = select_per_batch(
                partner_ids, all_h, all_c, all_actions
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

            # Store outputs
            transition = {
                'obs': obs_agent1,
                'action': action_agent1,
                'reward': reward['agent_1'],
                'shaped_reward': info['shaped_reward']['agent_1'],
                'log_prob': log_prob,
                'value': value,
                'done': next_done['__all__'],
            }
            
            new_carry = (next_env_state, obsv, agent_hstate, partner_hstates_new, rng, next_done['__all__'], partner_ids)
            return new_carry, transition

        # Reset environments
        rng, _reset_rng = jax.random.split(rng)
        reset_keys = jax.random.split(_reset_rng, batch_size)
        initial_obs, initial_env_state = jax.vmap(env.reset, in_axes=(0,))(reset_keys)
        
        # Initialize hidden states
        agent_hstate = ScannedRNN.initialize_carry(batch_size, agent_lstm_dim)
        partner_hstate = ScannedRNN.initialize_carry(batch_size, partner_lstm_dim)
        
        # Random partner selection
        rng, _rng = jax.random.split(rng)
        partner_ids = jax.random.randint(_rng, (batch_size,), 0, num_partners)
        
        # Initial carry
        initial_done = jnp.zeros((batch_size,), dtype=jnp.bool_)
        initial_carry = (initial_env_state, initial_obs, agent_hstate, partner_hstate, rng, initial_done, partner_ids)
        
        # Run episode
        _, transitions = nnx.scan(
            lambda carry, _: _env_step(carry, _),
            in_axes=(nnx.Carry, None),
            out_axes=(nnx.Carry, 0),
            length=episode_length
        )(initial_carry, None)
        
        return transitions

    # JIT the training function
    train_jit = nnx.jit(train)

    return train_jit, env


@jax.jit
def compute_gae(rewards, values, dones, gamma=0.99, lambda_=0.95):
    """Compute Generalized Advantage Estimation."""
    T = rewards.shape[0]
    lastgaelam = jnp.zeros(rewards.shape[1])
    
    # Pad values and dones for easy indexing without bounds checking
    # Append zeros at the end for "next" values
    values_padded = jnp.concatenate([values, jnp.zeros((1,) + values.shape[1:])], axis=0)
    dones_padded = jnp.concatenate([dones, jnp.ones((1,) + dones.shape[1:])], axis=0)  # terminal at end
    
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


@nnx.jit
def ppo_update_step(agent, optimizer, obs, actions, old_log_probs, advantages, returns, clip_eps=0.2, entropy_coef=0.01):
    """Single PPO update step.
    
    Args:
        obs: (T, batch, H, W, C)
        actions: (T, batch)
        old_log_probs: (T, batch)
        advantages: (T, batch)
        returns: (T, batch)
    """
    # Clip inputs to prevent NaN
    obs = jnp.clip(obs, -10.0, 10.0)
    advantages = jnp.clip(advantages, -10.0, 10.0)
    returns = jnp.clip(returns, -100.0, 100.0)
    old_log_probs = jnp.clip(old_log_probs, -20.0, 0.0)
    
    def loss_fn(agent):
        import distrax
        
        # Re-initialize hidden state
        batch_size = obs.shape[1]
        lstm_dim = agent.rnn.hidden_size
        hstate = ScannedRNN.initialize_carry(batch_size, lstm_dim)
        
        # Process all timesteps - use nnx.scan instead of jax.lax.scan
        def step_fn(carry, obs_t):
            hstate = carry
            # obs_t: (batch, H, W, C), add seq dim for ActorCriticRNN
            new_hstate, pi, value = agent(hstate, obs_t[jnp.newaxis, :])
            logits = jnp.clip(pi.logits.squeeze(0), -20.0, 20.0)
            return new_hstate, (logits, value.squeeze(0))
        
        final_hstate, (all_logits, all_values) = nnx.scan(
            step_fn,
            in_axes=(nnx.Carry, 0),
            out_axes=(nnx.Carry, 0),
            length=obs.shape[0]
        )(hstate, obs)
        
        # Policy loss
        pi = distrax.Categorical(logits=all_logits)
        log_probs = pi.log_prob(actions)
        
        # Clip log_probs difference to prevent extreme ratios
        log_ratio = jnp.clip(log_probs - old_log_probs, -10.0, 10.0)
        ratio = jnp.exp(log_ratio)
        clipped_ratio = jnp.clip(ratio, 1 - clip_eps, 1 + clip_eps)
        policy_loss = -jnp.minimum(ratio * advantages, clipped_ratio * advantages).mean()
        
        # Value loss
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


def main():
    # Load config
    config = OmegaConf.load('./configs/ovc_demo.yaml')
    partner_config = config.partners
    config = config.fcp
    config = OmegaConf.to_container(config, resolve=True)
    partner_config = OmegaConf.to_container(partner_config, resolve=True)
    
    # Initialize wandb
    wandb.init(**config['wandb'], config=config)
    
    # Setup
    base_seed = config['params']['base_seed']
    rng = jax.random.PRNGKey(base_seed)
    
    # Create directories
    os.makedirs(config['path']['out_dir'], exist_ok=True)
    
    # Initialize agent
    rng, init_rng = jax.random.split(rng)
    rngs = nnx.Rngs(init_rng)
    
    agent = ActorCriticRNN(**config['params']['model'], rngs=rngs)
    
    # Create optimizer
    tx = optax.chain(
        optax.clip_by_global_norm(config['params']['max_grad_norm']),
        optax.adamw(**config['params']['optimizer']),
    )
    optimizer = nnx.Optimizer(agent, tx, wrt=nnx.Param)
    
    # Load checkpoint if exists (using CheckpointManager for numbered checkpoints)
    start_iter = 0
    ckpt_path = config['path'].get('ckpt_path', None)
    if ckpt_path is not None and os.path.exists(ckpt_path):
        print(f"Loading FCP agent checkpoint from: {ckpt_path}")
        
        ckpt_checkpointer = ocp.AsyncCheckpointer(ocp.StandardCheckpointHandler())
        ckpt_manager_restore = ocp.CheckpointManager(
            os.path.abspath(ckpt_path),
            ckpt_checkpointer,
            ocp.CheckpointManagerOptions(max_to_keep=5, create=False)
        )
        
        latest_step = ckpt_manager_restore.latest_step()
        if latest_step is not None:
            print(f"Restoring from step: {latest_step}")
            
            current_state = nnx.state(agent)
            optimizer_state = nnx.state(optimizer)
            template = {
                'model': current_state.to_pure_dict(),
                'optimizer': optimizer_state.to_pure_dict(),
                'iter': 0,
            }
            
            ckpt = ckpt_manager_restore.restore(
                latest_step,
                args=ocp.args.StandardRestore(template)
            )
            
            # Restore model state
            graphdef, abstract_state = nnx.split(agent)
            nnx.replace_by_pure_dict(abstract_state, ckpt['model'])
            agent = nnx.merge(graphdef, abstract_state)
            
            # Restore optimizer state
            opt_graphdef, opt_abstract_state = nnx.split(optimizer)
            nnx.replace_by_pure_dict(opt_abstract_state, ckpt['optimizer'])
            optimizer = nnx.merge(opt_graphdef, opt_abstract_state)
            
            start_iter = ckpt.get('iter', 0) + 1
            print(f"Checkpoint loaded! Resuming from iteration {start_iter}")
        else:
            print("No checkpoint steps found in manager.")
    
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
    train_fn, env = make_fcp_train(config, partner_config)
    
    # Checkpoint manager
    checkpointer = ocp.AsyncCheckpointer(ocp.StandardCheckpointHandler())
    ckpt_manager = ocp.CheckpointManager(
        os.path.abspath(config['path']['out_dir']),
        checkpointer,
        ocp.CheckpointManagerOptions(max_to_keep=5, create=True)
    )
    
    # Training loop
    best_return = -1e9
    
    for iter_num in tqdm(range(start_iter, config['params']['num_iterations'])):
        rng, train_rng = jax.random.split(rng)
        
        # Collect episode
        transitions = train_fn(train_rng, agent, partner_networks, optimizer)
        
        # Check for NaN in transitions
        for key, val in transitions.items():
            if jnp.any(jnp.isnan(val)):
                print(f"NaN detected in transitions['{key}'] at iter {iter_num}")
            if jnp.any(jnp.isinf(val)):
                print(f"Inf detected in transitions['{key}'] at iter {iter_num}")
        
        # Compute returns and advantagesㅇ
        rewards = transitions['reward'] + transitions['shaped_reward']
        returns, advantages = compute_gae(
            rewards,
            transitions['value'],
            transitions['done'].astype(jnp.float32),
            gamma=config['params']['loss']['gamma'],
            lambda_=config['params']['loss']['lambda_'],
        )
        
        # Check for NaN in returns/advantages
        if jnp.any(jnp.isnan(returns)):
            print(f"NaN detected in returns at iter {iter_num}")
        if jnp.any(jnp.isnan(advantages)):
            print(f"NaN detected in advantages at iter {iter_num}")
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # PPO updates
        for ppo_iter in range(10):
            total_loss, policy_loss, value_loss, entropy = ppo_update_step(
                agent, optimizer,
                transitions['obs'],
                transitions['action'],
                transitions['log_prob'],
                advantages,
                returns,
                entropy_coef=config['params']['loss']['entropy_weight'],
            )
            # Check for NaN in loss (only check once per outer iteration)
            if ppo_iter == 0:
                if jnp.isnan(total_loss) or jnp.isinf(total_loss):
                    print(f"NaN/Inf in total_loss at iter {iter_num}")
                if jnp.isnan(policy_loss) or jnp.isinf(policy_loss):
                    print(f"NaN/Inf in policy_loss at iter {iter_num}")
        
        # Logging
        mean_reward = transitions['reward'].sum(axis=0).mean()
        
        if (iter_num + 1) % config['log_interval'] == 0:
            wandb.log({
                'fcp_return': float(mean_reward),
                'fcp_loss_policy': float(policy_loss),
                'fcp_loss_value': float(value_loss),
                'fcp_entropy': float(entropy),
            }, step=iter_num)
        
        # Save checkpoint
        if (iter_num + 1) % config['params']['save_interval'] == 0:
            ckpt = {
                'model': nnx.state(agent).to_pure_dict(),
                'optimizer': nnx.state(optimizer).to_pure_dict(),
                'iter': iter_num,
            }
            ckpt_manager.save(iter_num, ckpt)
        
        # Save best model
        if mean_reward > best_return:
            best_return = float(mean_reward)
            ckpt = {
                'model': nnx.state(agent).to_pure_dict(),
                'optimizer': nnx.state(optimizer).to_pure_dict(),
                'iter': iter_num,
            }
            best_checkpointer = ocp.StandardCheckpointer()
            best_checkpointer.save(
                os.path.abspath(os.path.join(config['path']['out_dir'], 'best_model')),
                ckpt,
                force=True
            )
    
    # Save final model
    ckpt = {
        'model': nnx.state(agent).to_pure_dict(),
        'optimizer': nnx.state(optimizer).to_pure_dict(),
        'iter': iter_num,
    }
    final_checkpointer = ocp.StandardCheckpointer()
    final_checkpointer.save(
        os.path.abspath(os.path.join(config['path']['out_dir'], 'final_model')),
        ckpt,
        force=True
    )
    
    wandb.finish()


if __name__ == "__main__":
    main()
