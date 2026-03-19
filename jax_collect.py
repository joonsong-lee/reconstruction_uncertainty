import os

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import jax
import jax.numpy as jnp
import flax.nnx as nnx
import optax
import numpy as np
from argparse import ArgumentParser
from functools import partial
from tqdm import tqdm
from omegaconf import OmegaConf
from orbax import checkpoint as ocp

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


def make_collect(config):
    """Create a collection function similar to make_train in train_partners_jax.py"""
    layout = config['env']['grid'] if config['env']['grid'] in overcooked_v2_layouts.keys() else Layout.from_string(config['env']['grid'], config['env'].get('possible_recipes', None))
    env_config = {k: v for k, v in config['env'].items() if k != 'grid'}
    env = OvercookedV2(layout=layout, **env_config)

    def collect(rng, network, save_path, batch_size, episode_length, iteration):
        """
        Collect episodes using pure JAX/Flax NNX.
        
        Args:
            rng: JAX random key
            network: ActorCriticRNN network
            save_path: Path to save collected data
            batch_size: Number of parallel environments
            episode_length: Length of each episode
            iteration: Current iteration number for seeding
        """
        
        # Initialize hidden state for RNN
        initial_hstate = ScannedRNN.initialize_carry(
            batch_size * 2, config['params']['model']['lstm_dim']
        )

        def _env_step(carry, unused):
            """Single environment step."""
            env_state, last_obs, hstate, rng, done = carry

            # SELECT ACTION
            rng, _rng = jax.random.split(rng)
            obs_batch = jnp.stack([last_obs[a] for a in env.agents]).reshape(
                -1, *env.observation_space().shape
            )
            
            # Forward pass through network
            hstate, pi, value = network(hstate, obs_batch[jnp.newaxis, :])
            action = pi.sample(seed=_rng)
            log_prob = pi.log_prob(action)
            action = action.squeeze()
            value = value.squeeze()
            log_prob = log_prob.squeeze()
            
            # Unbatchify actions for environment
            env_act = unbatchify(action, env.agents, batch_size, 2)
            env_act = {k: v.flatten() for k, v in env_act.items()}

            # STEP ENV
            rng, _rng = jax.random.split(rng)
            rng_step = jax.random.split(_rng, batch_size)

            obsv, next_env_state, reward, next_done, info = jax.vmap(
                env.step, in_axes=(0, 0, 0)
            )(rng_step, env_state, env_act)

            # Get global observations for reconstruction training
            global_obs = jax.vmap(env.get_obs_default, in_axes=(0,))(env_state)
            
            # Store outputs
            outputs = (
                last_obs['agent_1'],  # obs1: (batch_size, H, W, C)
                global_obs[:, 1],      # global1: (batch_size, H, W, C)
            )
            
            new_carry = (next_env_state, obsv, hstate, rng, next_done['__all__'])
            return new_carry, outputs

        # Reset environments
        rng, _reset_rng = jax.random.split(rng)
        reset_keys = jax.random.split(_reset_rng, batch_size)
        initial_obs, initial_env_state = jax.vmap(env.reset, in_axes=(0,))(reset_keys)
        
        # Initial carry
        initial_done = jnp.zeros((batch_size,), dtype=jnp.bool_)
        initial_carry = (initial_env_state, initial_obs, initial_hstate, rng, initial_done)
        
        # Run episode using nnx.scan
        _, outputs = nnx.scan(
            lambda carry, _: _env_step(carry, _),
            in_axes=(nnx.Carry, None),
            out_axes=(nnx.Carry, 0),
            length=episode_length
        )(initial_carry, None)
        
        obs1, g1 = outputs  # (episode_length, batch_size, H, W, C)
        
        return obs1, g1

    return collect, env


def save_episode_data(obs1, g1, save_path, base_seed, batch_size, episode_length):
    """Save collected episode data to disk."""
    np_obs1 = np.array(obs1, copy=True)  # (episode_length, batch_size, H, W, C)
    np_g1 = np.array(g1, copy=True)
    
    for i in range(batch_size):
        seed = base_seed + i
        o1 = np.memmap(
            os.path.join(save_path, 'obs', f'{seed}.npy'),
            mode='w+', dtype=np.uint8, shape=(episode_length, 5, 5, 38)
        )
        g1_file = np.memmap(
            os.path.join(save_path, 'global', f'{seed}.npy'),
            mode='w+', dtype=np.uint8, shape=(episode_length, 5, 11, 40)
        )
        o1[:] = np_obs1[:, i]
        g1_file[:] = np_g1[:, i]
        o1.flush()
        g1_file.flush()


def load_network_from_checkpoint(ckpt_path, config, rngs):
    """Load network from checkpoint saved with CheckpointManager."""
    network = ActorCriticRNN(**config['params']['model'], rngs=rngs)
    
    if ckpt_path is not None and os.path.exists(ckpt_path):
        print(f"Loading checkpoint from: {ckpt_path}")
        
        # Use CheckpointManager to restore (same as used in train_partners_jax.py)
        checkpointer = ocp.AsyncCheckpointer(ocp.StandardCheckpointHandler())
        manager = ocp.CheckpointManager(
            os.path.abspath(ckpt_path),
            checkpointer,
            ocp.CheckpointManagerOptions(max_to_keep=2, create=False)
        )
        
        # Get latest checkpoint step
        latest_step = manager.latest_step()
        if latest_step is not None:
            print(f"Restoring from step: {latest_step}")
            
            # Create a dummy optimizer matching the same structure as train_partners_jax.py
            # Must use optax.chain with same transforms as during training
            tx = optax.chain(
                optax.clip_by_global_norm(config['params'].get('max_grad_norm', 0.5)),
                optax.adamw(**config["params"]["optimizer"]),
            )
            dummy_optimizer = nnx.Optimizer(network, tx, wrt=nnx.Param)
            
            # Create a template matching the saved checkpoint structure
            current_state = nnx.state(network)
            optimizer_state = nnx.state(dummy_optimizer)
            template = {
                'model': current_state.to_pure_dict(),
                'optimizer': optimizer_state.to_pure_dict(),
                'iter': 0,
            }
            
            # Restore with template to handle device placement
            ckpt = manager.restore(
                latest_step,
                args=ocp.args.StandardRestore(template)
            )
            
            # Update network parameters from restored checkpoint using official API
            # Reference: https://flax.readthedocs.io/en/latest/nnx/nnx_basics.html
            restored_params = ckpt['model']
            
            # Get graphdef and abstract_state from current network
            graphdef, abstract_state = nnx.split(network)
            
            # Replace the abstract_state with restored pure dict
            nnx.replace_by_pure_dict(abstract_state, restored_params)
            
            # Merge back to get the updated network
            network = nnx.merge(graphdef, abstract_state)
            
            print("Checkpoint loaded successfully!")
        else:
            print("No checkpoint steps found in manager.")
    else:
        print("No checkpoint found, using randomly initialized network.")
    
    return network


def main(args):
    # Load config
    config = OmegaConf.load(f'./configs/{args.env}.yaml')
    config = config.partners
    config = OmegaConf.to_container(config, resolve=True)
    
    # Setup
    base_seed = args.base_num
    print(f'Base seed: {base_seed}')
    
    # Create directories
    os.makedirs(config['path']['out_dir'], exist_ok=True)
    save_path = config['path'].get('save_dir', None)
    if save_path is not None:
        os.makedirs(save_path, exist_ok=True)
        os.makedirs(os.path.join(save_path, 'obs'), exist_ok=True)
        os.makedirs(os.path.join(save_path, 'global'), exist_ok=True)
    
    # Initialize network
    rng = jax.random.PRNGKey(base_seed)
    rng, init_rng = jax.random.split(rng)
    rngs = nnx.Rngs(init_rng)
    
    # Load network from checkpoint if available
    ckpt_path = config['path'].get('ckpt_path', None)
    network = load_network_from_checkpoint(ckpt_path, config, rngs)
    
    # Create collection function
    collect_fn, env = make_collect(config)
    
    # JIT compile the collection function
    @nnx.jit
    def jit_collect(rng, network):
        return collect_fn(
            rng, network, save_path,
            config['params']['batch_size'],
            config['params']['max_episode_length'],
            0
        )
    
    # Collection loop
    batch_size = config['params']['batch_size']
    episode_length = config['params']['max_episode_length']
    num_iterations = config['params']['num_iterations']
    
    for iteration in tqdm(range(num_iterations)):
        rng, collect_rng = jax.random.split(rng)
        
        # Collect episode
        obs1, g1 = jit_collect(collect_rng, network)
        
        # Save data if save_path is provided
        if save_path is not None:
            current_seed = base_seed + iteration * batch_size
            save_episode_data(obs1, g1, save_path, current_seed, batch_size, episode_length)
    
    print(f"Collection complete! Saved {num_iterations * batch_size} episodes to {save_path}")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--env', type=str, default='ovc')
    parser.add_argument('--base_num', type=int, default=0)
    args = parser.parse_args()
    main(args)
