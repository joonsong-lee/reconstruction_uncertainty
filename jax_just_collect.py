import os
# 2. Import torch and immediately check if it can see the GPUs.
#os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
import torch.nn.functional as F
from argparse import ArgumentParser
from time import time,sleep
from dataclasses import dataclass
from functools import partial
from typing import List
import multiprocessing as mp
import numpy as np

from tqdm import tqdm
import wandb
from omegaconf import OmegaConf
from alg.cnn_mappo import mappo_lstm_stateful as lstm

from util import *
import jax
#jax.config.update("jax_platform_name", "cpu")
from jaxmarl import make
from jaxmarl.environments.overcooked_v2.layouts import Layout
import jax.numpy as jnp
from functools import partial
from jax.experimental import io_callback


custom_layout_str = """
WXWWWWWWWXW
0    R    0
1   APA   1
2    R    2
WBWWWWWWWBW
"""

def torch_callback(obs1,obs2,agent):
    device = next(agent.parameters()).device
    # Callback runs on host (outside JIT). Inputs arrive as (possibly read-only) NumPy views from JAX.
    # Copy to ensure writable memory before torch.from_numpy to avoid UserWarning about non-writable arrays.
    with torch.inference_mode(),torch.no_grad():
        obs1_ = j2t(obs1).float().permute(0,3,1,2).to(device)
        obs2_ = j2t(obs2).float().permute(0,3,1,2).to(device)
        logit, logit2, value = agent(obs1_,obs2_)
        d = Categorical(logits=logit.squeeze().float())
        d2 = Categorical(logits=logit2.squeeze().float())
        action_ = d.sample()
        log_prob = d.log_prob(action_)
        action2_ = d2.sample()
        log_prob2 = d2.log_prob(action2_)
        actions = torch.cat([action_.unsqueeze(1),action2_.unsqueeze(1)],dim=1)
    return t2j(actions),t2j(log_prob),t2j(value.squeeze()),t2j(log_prob2)

@partial(jax.jit, static_argnames=("env", "episode_length", "callback_fn","save_path","batch_size"))
def run_episode_p(env, key, episode_length, callback_fn,save_path,batch_size):
    """ JIT-compiled function to run one full episode. """
    
    def _step_fn(carry, _):
        key, state, done, obs1, obs2 = carry

        # 1. Select Action
        key,_step_key = jax.random.split(key)
        step_key = jax.random.split(_step_key,num=batch_size)
        def step_if_not_done(current_state,current_obs1,current_obs2):
            # Shapes/dtypes expected from callback
            result_shape_dtype = (
                jax.ShapeDtypeStruct((batch_size,2), jnp.int32),      # policy_action
                jax.ShapeDtypeStruct((batch_size,), jnp.float32),     # logprob
                jax.ShapeDtypeStruct((batch_size,), jnp.float32),      # value
                jax.ShapeDtypeStruct((batch_size,), jnp.float32),     # logprob2
            )
            # Wrapper so we only pass JAX arrays (render_state, hx, cx); vae & ac captured (static)
            policy_action,  log_prob,value,log_prob2 = io_callback(
                callback_fn,
                result_shape_dtype,
                obs1,
                obs2,
                ordered=True  # Ensures steps execute in order.
            )
            action_dict = {'agent_0': policy_action[:,0], 'agent_1': policy_action[:,1]}
            obs, next_state, reward, next_done, info = jax.vmap(env.step,in_axes=(0,0,0))(step_key, current_state, action_dict)
            next_obs1 = obs['agent_0']
            next_obs2 = obs['agent_1']
            return (
                next_obs1,
                next_obs2,
                next_state,
                reward['agent_0'],
                info['shaped_reward']['agent_0'],
                info['shaped_reward']['agent_1'],
                next_done['__all__'],
                policy_action,
                log_prob,
                value,
                log_prob2,
            )

        def skip_step_if_done(current_state,current_obs1,current_obs2):
            """Return the same state and zero rewards if the episode has ended."""
            result_shape_dtype = (
                jax.ShapeDtypeStruct((batch_size,2), jnp.int32),      # policy_action
                jax.ShapeDtypeStruct((batch_size,), jnp.float32),     # logprob
                jax.ShapeDtypeStruct((batch_size,), jnp.float32),      # value
                jax.ShapeDtypeStruct((batch_size,), jnp.float32),     # logprob2
            )
            # Wrapper so we only pass JAX arrays (render_state, hx, cx); vae & ac captured (static)
            policy_action,  log_prob,value,log_prob2 = io_callback(
                callback_fn,
                result_shape_dtype,
                obs1,
                obs2,
                ordered=True  # Ensures steps execute in order.
            )
            return (
                current_obs1,
                current_obs2,
                current_state,
                jnp.zeros((batch_size,),dtype=jnp.float32),          # Zero reward
                jnp.zeros((batch_size,),dtype=jnp.float32),          # Zero shaped reward
                jnp.zeros((batch_size,),dtype=jnp.float32),          # Zero shaped reward for agent 1
                jnp.ones((batch_size,), dtype=jnp.bool),         # Done remains True
                jnp.zeros((batch_size,2),dtype=jnp.int32),          # Keep previous actions
                jnp.zeros((batch_size,), dtype=jnp.float32),       # Dummy logit
                value,
                jnp.zeros((batch_size,), dtype=jnp.float32),     # Dummy logit2
            )
        

        next_obs1,next_obs2,next_state, reward, shaped_reward,shaped_reward2, next_done, chosen_actions,log_prob,value,log_prob2= jax.lax.cond(
            done.all(),
            skip_step_if_done,
            step_if_not_done,
            state,  # Pass the current state to the chosen function,
            obs1,
            obs2
        )
        if save_path is not None:
            g = jax.vmap(env.get_obs_default,in_axes=(0,))(state)
        else:
            g = [jnp.array(0),jnp.array(0)]
        g1 = g[:,0]
        new_carry = (key, next_state, next_done, next_obs1,next_obs2)
        outputs = (obs1,g1)
        return new_carry, outputs

    # Initial state
    key, _reset_key = jax.random.split(key)
    reset_key = jax.random.split(_reset_key,batch_size)
    initial_obs, initial_state = jax.vmap(env.reset,in_axes=(0,))(reset_key)
    initial_obs1 = initial_obs['agent_0']
    initial_obs2 = initial_obs['agent_1']
    
    # Run the scan over the episode length
    initial_carry = (key, initial_state, jnp.zeros((batch_size,), dtype=jnp.bool), initial_obs1,initial_obs2)
    _, outputs = jax.lax.scan(_step_fn, initial_carry, None, length=episode_length)
    obs1,g1 = outputs
    return obs1, g1


def collect_episode(callback_fn,env,batch_size,seed,save_path,episode_length=400):
    with torch.no_grad():
        key = jax.random.PRNGKey(seed)
        key = jax.device_put(key, jax.devices()[0])
        obss1,g1 = run_episode_p(env, key, episode_length,callback_fn,save_path,batch_size)
    if save_path is not None:
        np_obss1 = np.array(obss1, copy=True).squeeze()
        globals = np.array(g1, copy=True).squeeze()
        for i in range(batch_size):
            seed = seed + i
            o1 = np.memmap(os.path.join(save_path,'obs', f'{seed}.npy'), mode='w+', dtype=np.uint8, shape=(episode_length,5,5,39))
            g1 = np.memmap(os.path.join(save_path,'global', f'{seed}.npy'), mode='w+', dtype=np.uint8, shape=(episode_length, 5,8,41))
            o1[:] = np_obss1.squeeze()[:,i]
            g1[:] = globals.squeeze()[:,i]
            o1.flush()
            g1.flush()
    


def main(args):
    config = OmegaConf.load(f'./configs/{args.env}.yaml')
    config = config.partners
    config = OmegaConf.to_container(config, resolve=True)
       #  if not ddp, we are running on a single gpu, and one process
    base_seed = config['params']['base_seed']
    print(f'Base seed: {base_seed}')
    #config['path']['ckpt_path'] = config['path']['out_dir']+'/final_model.pt'
    set_seed(base_seed)
    device = "cuda:0"
    agent = lstm(**config['params']['algorithm'],device=device).to(device)
    start_iter = 0
    ckpt = None
    callback_fn = partial(torch_callback, agent=agent)
    torch.backends.cuda.matmul.allow_tf32 = True  # allow tf32 on matmul
    torch.backends.cudnn.allow_tf32 = True  # allow tf32 on cudnn
    best_return = -1e9
    os.makedirs(config['path']['out_dir'], exist_ok=True)
    save_path = config['path']['save_dir']
    if save_path is not None:
        os.makedirs(save_path, exist_ok=True)
        os.makedirs(os.path.join(save_path,'obs'), exist_ok=True)
        os.makedirs(os.path.join(save_path,'global'), exist_ok=True)
    if config['path']['ckpt_path'] is not None:
        ckpt = torch.load(config['path']['ckpt_path'], map_location=device)
        agent.load_state_dict(ckpt['agent_state_dict'])
    agent = agent.to(device)
    agent.eval()
    #l =Layout.from_string(custom_layout_str)
    env = make("overcooked_v2", layout='test_time_simple', agent_view_size = 2,random_agent_positions=True,random_reset = False,
               max_steps=config['params']['max_episode_length'],sample_recipe_on_delivery=False,indicate_successful_delivery=True,negative_rewards=True)
    env.layout.agent_positions = env.layout.agent_positions[::-1]  # Swap starting positions
    for iter in tqdm(range(start_iter,config['params']['num_iterations'])):
        collect_episode(callback_fn,env,config['params']['batch_size'], base_seed+iter*config['params']['batch_size'],
                        save_path, config['params']['max_episode_length'])
        

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--env', type=str, default='ovc')
    args = parser.parse_args()
    main(args)




