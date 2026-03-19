import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

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

from jaxmarl import make
from jaxmarl.environments.overcooked_v2.overcooked import OvercookedV2
from jaxmarl.environments import overcooked_v2_layouts
from jaxmarl.environments.overcooked_v2.layouts import Layout

from alg.jax_ppo import ActorCriticRNN, ScannedRNN
from recon_module.evdiential import build_model


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


def load_recon_model_from_checkpoint(ckpt_path, config, rngs):
    """Load reconstruction model from checkpoint."""
    model, _, _, _, _, _ = build_model(config, task='classification', is_kv=True, rngs=rngs)
    
    if ckpt_path is not None and os.path.exists(ckpt_path):
        print(f"Loading recon model from: {ckpt_path}")
        checkpointer = ocp.StandardCheckpointer()
        ckpt = checkpointer.restore(ckpt_path)
        
        restored_params = ckpt['model']
        graphdef, abstract_state = nnx.split(model)
        nnx.replace_by_pure_dict(abstract_state, restored_params)
        model = nnx.merge(graphdef, abstract_state)
        
        print("Recon model loaded successfully!")
    
    return model


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
            ckpt_path = os.path.join(partner_dir, f'{k}_{i}')
            if os.path.exists(ckpt_path):
                rng_key = jax.random.fold_in(rngs.default.key.value, i * len(partner_keys) + partner_keys.index(k))
                partner_rngs = nnx.Rngs(rng_key)
                partner = load_network_from_checkpoint(ckpt_path, config, partner_rngs)
                partner_networks.append(partner)
                print(f"Loaded partner {i} checkpoint {k}")
    
    return partner_networks


def make_fcp_recon_train(config, partner_config, recon_config):
    """Create FCP-Recon training function."""
    layout = Layout.from_string(custom_layout_str)
    env = OvercookedV2(
        layout=layout, 
        agent_view_size=2,
        random_agent_positions=True, 
        random_reset=False,
        max_steps=config['params']['max_episode_length'],
        sample_recipe_on_delivery=True
    )
    
    # Get observation shape from env
    obs_shape = env.observation_shape  # (H, W, C)
    obs_h, obs_w, obs_c = obs_shape
    # Global observation shape
    global_obs_shape = env.global_observation_shape  # (H, W, C)
    global_h, global_w, global_c = global_obs_shape

    def train(rng, agent, recon_model, partner_networks, optimizer):
        """
        FCP-Recon training loop using pure JAX.
        """
        batch_size = config['params']['batch_size']
        episode_length = config['params']['max_episode_length']
        num_partners = len(partner_networks)
        
        def _env_step(carry, unused):
            """Single environment step."""
            env_state, last_obs, agent_hstate, partner_hstates, rng, done, partner_ids = carry

            rng, _rng = jax.random.split(rng)
            obs_agent0 = last_obs['agent_0']  # (batch, H, W, C)
            
            # Apply reconstruction model
            obs_flat = obs_agent0.reshape(batch_size, 1, -1)  # (batch, 1, H*W*C)
            # recon_prob: (batch, 1, out_units, num_classes) - probability distribution
            # epistemic_unc: (batch, 1, out_units, 1)
            # aleatoric_unc: (batch, 1, out_units, 1)
            recon_prob, epistemic_unc, aleatoric_unc = recon_model(obs_flat)
            
            # Convert probability to predicted class (reconstruction output)
            recon_mu = jnp.argmax(recon_prob, axis=-1)  # (batch, 1, out_units)
            recon_obs = recon_mu.squeeze(1).reshape(batch_size, global_h, global_w, global_c)  # (batch, H, W, C)
            recon_obs = recon_obs.astype(jnp.float32)  # Convert to float for concatenation
            
            # Combine uncertainty and mean to single channel
            # (batch, 1, out_units, 1) -> squeeze -> (batch, 1, out_units) -> squeeze -> (batch, out_units)
            uncertainty = (epistemic_unc + aleatoric_unc).squeeze(-1).squeeze(1)  # (batch, out_units)
            uncertainty = uncertainty.reshape(batch_size, global_h, global_w, global_c)
            uncertainty = uncertainty.mean(axis=-1, keepdims=True)  # (batch, H, W, 1)
            
            # Concatenate recon obs and uncertainty: (batch, H, W, C+1)
            agent_input = jnp.concatenate([recon_obs, uncertainty], axis=-1)
            
            # Forward pass through agent network (ActorCriticRNN)
            agent_hstate, pi, value = agent(agent_hstate, agent_input[jnp.newaxis, :])
            
            action_agent0 = pi.sample(seed=_rng).squeeze(0)
            log_prob = pi.log_prob(action_agent0).squeeze(0)
            value = value.squeeze(0)
            
            # SELECT ACTION for agent_1 (partner)
            rng, _rng = jax.random.split(rng)
            obs_agent1 = last_obs['agent_1']
            
            partner_hstates_new, pi_partner, _ = partner_networks[0](
                partner_hstates, obs_agent1[jnp.newaxis, :]
            )
            action_agent1 = pi_partner.sample(seed=_rng).squeeze()
            
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
                'obs': agent_input,  # Use recon input as obs
                'action': action_agent0,
                'reward': reward['agent_0'],
                'shaped_reward': info['shaped_reward']['agent_0'],
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
        agent_lstm_dim = agent.rnn.hidden_size
        agent_hstate = ScannedRNN.initialize_carry(batch_size, agent_lstm_dim)
        partner_hstate = ScannedRNN.initialize_carry(batch_size, partner_config['params']['model']['lstm_dim'])
        
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

    return train, env


@nnx.jit
def compute_gae(rewards, values, dones, gamma=0.99, lambda_=0.95):
    """Compute Generalized Advantage Estimation."""
    T = rewards.shape[0]
    advantages = jnp.zeros_like(rewards)
    lastgaelam = jnp.zeros(rewards.shape[1])
    
    def gae_step(carry, t):
        lastgaelam = carry
        nextnonterminal = 1.0 - dones[t + 1] if t < T - 1 else jnp.zeros_like(dones[0])
        nextvalue = values[t + 1] if t < T - 1 else jnp.zeros_like(values[0])
        delta = rewards[t] + gamma * nextvalue * nextnonterminal - values[t]
        lastgaelam = delta + gamma * lambda_ * nextnonterminal * lastgaelam
        return lastgaelam, lastgaelam
    
    # Scan backwards
    _, advantages_reversed = jax.lax.scan(
        gae_step,
        lastgaelam,
        jnp.arange(T - 1, -1, -1)
    )
    advantages = advantages_reversed[::-1]
    
    returns = advantages + values
    return returns, advantages


def ppo_update_step(agent, optimizer, obs, actions, old_log_probs, advantages, returns, clip_eps=0.2, entropy_coef=0.01):
    """Single PPO update step.
    
    Args:
        obs: (T, batch, H, W, C)
        actions: (T, batch)
        old_log_probs: (T, batch)
        advantages: (T, batch)
        returns: (T, batch)
    """
    def loss_fn(agent):
        import distrax
        
        # Re-initialize hidden state
        batch_size = obs.shape[1]
        lstm_dim = agent.rnn.hidden_size
        hstate = ScannedRNN.initialize_carry(batch_size, lstm_dim)
        
        # Process all timesteps
        def step_fn(hstate, obs_t):
            # obs_t: (batch, H, W, C), add seq dim for ActorCriticRNN
            hstate, pi, value = agent(hstate, obs_t[jnp.newaxis, :])
            return hstate, (pi.logits.squeeze(0), value.squeeze(0))
        
        _, (all_logits, all_values) = jax.lax.scan(step_fn, hstate, obs)
        
        # Policy loss
        pi = distrax.Categorical(logits=all_logits)
        log_probs = pi.log_prob(actions)
        ratio = jnp.exp(log_probs - old_log_probs)
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
    optimizer.update(grads)
    
    return total_loss, policy_loss, value_loss, entropy


def main():
    # Load config
    config = OmegaConf.load('./configs/ovc.yaml')
    partner_config = config.partners
    recon_config = config.recon_module
    config = config.fcp_recon
    
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
    
    # Initialize models
    rng, init_rng = jax.random.split(rng)
    rngs = nnx.Rngs(init_rng)
    
    # Load reconstruction model
    rng, recon_rng = jax.random.split(rng)
    recon_rngs = nnx.Rngs(recon_rng)
    recon_model = load_recon_model_from_checkpoint(
        recon_config['path'].get('ckpt_path'),
        recon_config,
        recon_rngs
    )
    
    # Create FCP-Recon agent using ActorCriticRNN
    agent = ActorCriticRNN(**config['params']['model'], rngs=rngs)
    lstm_dim = config['params']['model']['lstm_dim']
    
    # Create optimizer
    tx = optax.chain(
        optax.clip_by_global_norm(config['params']['max_grad_norm']),
        optax.adamw(**config['params']['optimizer']),
    )
    optimizer = nnx.Optimizer(agent, tx, wrt=nnx.Param)
    
    # Load checkpoint if exists
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
    train_fn, env = make_fcp_recon_train(config, partner_config, recon_config)
    
    # Checkpoint manager
    checkpointer = ocp.StandardCheckpointer()
    
    # Training loop
    best_return = -1e9
    
    for iter_num in tqdm(range(config['params']['num_iterations'])):
        rng, train_rng = jax.random.split(rng)
        
        # Collect episode
        transitions = train_fn(train_rng, agent, recon_model, partner_networks, optimizer)
        
        # Compute returns and advantages
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
        
        # PPO updates
        for _ in range(10):
            total_loss, policy_loss, value_loss, entropy = ppo_update_step(
                agent, optimizer,
                transitions['obs'],
                transitions['action'],
                transitions['log_prob'],
                advantages,
                returns,
                entropy_coef=config['params']['loss']['entropy_weight'],
            )
        
        # Logging
        mean_reward = transitions['reward'].sum(axis=0).mean()
        
        if (iter_num + 1) % config['log_interval'] == 0:
            wandb.log({
                'fcp_recon_return': float(mean_reward),
                'fcp_recon_loss_policy': float(policy_loss),
                'fcp_recon_loss_value': float(value_loss),
                'fcp_recon_entropy': float(entropy),
            }, step=iter_num)
        
        # Save checkpoint
        if (iter_num + 1) % config['params']['save_interval'] == 0:
            ckpt = {
                'model': nnx.state(agent).to_pure_dict(),
                'optimizer': nnx.state(optimizer).to_pure_dict(),
                'iter': iter_num,
            }
            checkpointer.save(
                os.path.join(config['path']['out_dir'], f'model_iter_{iter_num}'),
                ckpt,
                force=True
            )
        
        # Save best model
        if mean_reward > best_return:
            best_return = float(mean_reward)
            ckpt = {
                'model': nnx.state(agent).to_pure_dict(),
                'optimizer': nnx.state(optimizer).to_pure_dict(),
                'iter': iter_num,
            }
            checkpointer.save(
                os.path.join(config['path']['out_dir'], 'best_model'),
                ckpt,
                force=True
            )
    
    # Save final model
    ckpt = {
        'model': nnx.state(agent).to_pure_dict(),
        'optimizer': nnx.state(optimizer).to_pure_dict(),
        'iter': iter_num,
    }
    checkpointer.save(
        os.path.join(config['path']['out_dir'], 'final_model'),
        ckpt,
        force=True
    )
    
    wandb.finish()


if __name__ == "__main__":
    main()
