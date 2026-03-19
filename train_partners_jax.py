import os

os.environ["IMAGEIO_FFMPEG_EXE"] = "/usr/bin/ffmpeg"
# 2. Import torch and immediately check if it can see the GPUs.
#os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
import jax

jax.config.update("jax_default_matmul_precision", "tensorfloat32")
import jax.numpy as jnp
from jaxmarl.environments.overcooked_v2.overcooked import OvercookedV2
from jaxmarl.environments import overcooked_v2_layouts
from jaxmarl.environments.overcooked_v2.layouts import Layout
import flax.nnx as nnx

import optax
from orbax import checkpoint as ocp
#os.environ['CUDA_VISIBLE_DEVICES'] = '1,2'
from argparse import ArgumentParser

import numpy as np
import wandb
from omegaconf import OmegaConf


from alg.jax_ppo import ActorCriticRNN,ScannedRNN
from util import make_movie
from typing import NamedTuple


class Transition(NamedTuple):
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    info: jnp.ndarray

def batchify(x: dict, agent_list, num_actors):
    x = jnp.stack([x[a] for a in agent_list])
    return x.reshape((num_actors, -1))


def unbatchify(x: jnp.ndarray, agent_list, num_envs, num_actors):
    x = x.reshape((num_actors, num_envs))
    return {a: x[i] for i, a in enumerate(agent_list)}

def make_train(config):
    layout = config['env']['grid'] if config['env']['grid'] in overcooked_v2_layouts.keys() else Layout.from_string(config['env']['grid'],config['env']['possible_recipes'])
    config['env'].pop('grid')
    env = OvercookedV2(layout=layout, **config['env'])

    rew_shaping_anneal = optax.linear_schedule(
        init_value=1.0, end_value=0.0, transition_steps=config["params"]["reward_shaping_iters"]
    )

    def train(rng,base_num):

        # INIT NETWORK
        

        rng, _rng = jax.random.split(rng)
        rngs = nnx.Rngs(_rng)
        network = ActorCriticRNN(**config['params']['model'],rngs=rngs)
        tx = optax.chain(
            optax.clip_by_global_norm(config['params']['max_grad_norm']),
            optax.adamw(**config["params"]["optimizer"]),
        )
        # graphdef, params, other_variables = nnx.split(network, nnx.Param, ...)
        # train_state = TrainState.create(
        #     apply_fn=graphdef.apply,
        #     params=params,
        #     other_variables=other_variables,
        #     tx=tx
        #     )
        optimizer = nnx.Optimizer(network,tx,wrt=nnx.Param)
        # INIT ENV
        
        initial_hstate = ScannedRNN.initialize_carry(
            config['params']['batch_size']*2, config['params']['model']['lstm_dim']
        )
        checkpointer = ocp.AsyncCheckpointer(ocp.StandardCheckpointHandler())
        iter_manager = ocp.CheckpointManager(
            os.path.abspath(os.path.join(config['path']['out_dir'],f'iter_saver_{base_num}')), checkpointer, ocp.CheckpointManagerOptions(max_to_keep=2, create=True))
        best_manager = ocp.CheckpointManager(
            os.path.abspath(os.path.join(config['path']['out_dir'],f'best_model_{base_num}')), checkpointer, ocp.CheckpointManagerOptions(max_to_keep=2, create=True))
        
        # TRAIN LOOP
        def callback(model_state,opt_state,manager,metric,current_iter):
            ckpt = {'model':model_state.to_pure_dict(),'optimizer':opt_state.to_pure_dict(),'iter':current_iter}
            wandb_metric = {
                        k: v.item() for k, v in metric.items()
                    }
            manager.save(current_iter,args=ocp.args.StandardSave(ckpt))
            wandb.log(wandb_metric, step=current_iter)
            #make_movie(obs, os.path.join(config['path']['out_dir'], f'partner_episode_{base_num}_iter_{current_iter}.gif'))

        def _update_step(carry, unused):
            # COLLECT TRAJECTORIES
            network, optimizer, initial_hstate, rng, step, best_return = carry
            def _env_step(network,runner_state, unused):
                (
                    env_state,
                    last_obs,
                    hstate,
                    rng,
                    step,
                ) = runner_state

                # SELECT ACTION
                rng, _rng = jax.random.split(rng)
                obs_batch = jnp.stack([last_obs[a] for a in env.agents]).reshape(
                    -1, *env.observation_space().shape
                )
                hstate, pi, value = network(hstate, obs_batch[np.newaxis,:])
                action = pi.sample(seed=_rng)
                log_prob = pi.log_prob(action)
                action = action.squeeze()
                value = value.squeeze()
                log_prob = log_prob.squeeze()
                env_act = unbatchify(action, env.agents, config['params']['batch_size'], 2)
                env_act = {k: v.flatten() for k, v in env_act.items()}

                # STEP ENV
                rng, _rng = jax.random.split(rng)
                rng_step = jax.random.split(_rng, config["params"]["batch_size"])

                obsv, env_state, reward, done, info = jax.vmap(
                    env.step, in_axes=(0, 0, 0)
                )(rng_step, env_state, env_act)

                anneal_factor = rew_shaping_anneal(step)
                shaped_reward = info['shaped_reward']['agent_0']+info['shaped_reward']['agent_1']
                reward = jax.tree_util.tree_map(
                    lambda x, y: x + y * anneal_factor, reward, info["shaped_reward"]
                )
                info["shaped_reward"] = shaped_reward
                info['original_reward'] = reward['agent_0']
                transition = Transition(
                    action.squeeze(),
                    value.squeeze(),
                    batchify(reward,env.agents,config['params']['batch_size']*2).squeeze(),
                    log_prob.squeeze(),
                    obs_batch,
                    info,
                )
                runner_state = (
                    env_state,
                    obsv,
                    hstate,
                    rng,
                    step,
                )
                return runner_state, transition
            
            rng, _rng = jax.random.split(rng)
            reset_rng = jax.random.split(_rng, config['params']["batch_size"])
            initial_obsv, initial_env_state = jax.vmap(env.reset, in_axes=(0,))(reset_rng)
            runner_state=(
                initial_env_state,
                initial_obsv,
                initial_hstate,
                rng,
                step,
            )
            runner_state, traj_batch = nnx.scan(_env_step,in_axes=(None,nnx.Carry,None),out_axes=(nnx.Carry,0),length = config['params']['max_episode_length'])(network,runner_state, None) #T,B,H,W,C

            # compute last value for gae
            env_state, last_obs, hstate, rng, step= (
                runner_state
            )
            obs_batch = jnp.stack([last_obs[a] for a in env.agents]).reshape(
                    -1, *env.observation_space().shape
                )
            _, _, last_val = network(hstate, obs_batch[np.newaxis,:])
            last_val = last_val.squeeze()

            def _calculate_gae(traj_batch, last_val):
                def _get_advantages(gae_and_next_value, transition):
                    gae, next_value = gae_and_next_value
                    value, reward = (
                        transition.value,
                        transition.reward,
                    )
                    delta = reward + config['params']['loss']['gamma'] * next_value  - value
                    gae = (
                        delta
                        + config['params']['loss']['gamma'] * config['params']['loss']['lambda_'] * gae
                    )
                    return (gae, value), gae

                _, advantages = jax.lax.scan(
                    _get_advantages,
                    (jnp.zeros_like(last_val), last_val),
                    traj_batch,
                    reverse=True,
                    unroll=16,
                )
                return advantages, advantages + traj_batch.value

            advantages, targets = _calculate_gae(traj_batch, last_val)
            # UPDATE NETWORK
            def _update_epoch(carry, unused):
                network, optimizer, update_state = carry
                def _update_minbatch(carry, batch_info):
                    network, optimizer, dummy_carry = carry
                    init_hstate, traj_batch, advantages, targets = batch_info

                    def _loss_fn(network,init_hstate, traj_batch, gae, targets):
                        # RERUN NETWORK
                        ac_in = traj_batch.obs
                        _, pi, value = network(jax.tree_util.tree_map(lambda x: x.squeeze(), init_hstate), ac_in,)
                        log_prob = pi.log_prob(traj_batch.action)

                        # CALCULATE VALUE LOSS
                        # value_pred_clipped = traj_batch.value + (
                        #     value - traj_batch.value
                        # ).clip(-config['params']['algorithm']['clip'], config['params']['algorithm']['clip'])
                        value_losses = jnp.square(value - targets)
                        value_loss = 0.5 * value_losses.mean()
                        # value_losses_clipped = jnp.square(value_pred_clipped - targets)
                        # value_loss = (
                        #     0.5 * jnp.maximum(value_losses, value_losses_clipped).mean()
                        # )


                        # CALCULATE ACTOR LOSS
                        ratio = jnp.exp(log_prob - traj_batch.log_prob)
                        gae = (gae - gae.mean()) / (gae.std() + 1e-8)
                        loss_actor1 = ratio * gae
                        loss_actor2 = (
                            jnp.clip(
                                ratio,
                                1.0 - config['params']['algorithm']['clip'],
                                1.0 + config['params']['algorithm']['clip'],
                            )
                            * gae
                        )
                        loss_actor = -jnp.minimum(loss_actor1, loss_actor2)
                        loss_actor = loss_actor.mean()
                        entropy = pi.entropy().mean()

                        total_loss = (
                            loss_actor
                            + config['params']['loss']['value_coef'] * value_loss
                            - config['params']['loss']['entropy_weight'] * entropy
                        )
                        return total_loss, (value_loss, loss_actor, entropy)

                    grad_fn = nnx.value_and_grad(_loss_fn, has_aux=True)
                    total_loss, grads = grad_fn(
                         network, init_hstate, traj_batch, advantages, targets
                    )
                    optimizer.update(network, grads)
                    return (network, optimizer, dummy_carry), total_loss

                init_hstate, traj_batch, advantages, targets, rng = (
                    update_state
                )
                rng, _rng = jax.random.split(rng)

                init_hstate = jax.tree_util.tree_map(lambda x: jnp.reshape(x, (1, config['params']["batch_size"]*2, -1)), init_hstate)
                batch = (
                    init_hstate,
                    traj_batch,
                    advantages.squeeze(),
                    targets.squeeze(),
                )
                permutation = jax.random.permutation(_rng, config['params']["batch_size"]*2)

                shuffled_batch = jax.tree_util.tree_map(
                    lambda x: jnp.take(x, permutation, axis=1), batch
                )

                minibatches = jax.tree_util.tree_map(
                    lambda x: jnp.swapaxes(
                        jnp.reshape(
                            x,
                            [x.shape[0], config['params']['algorithm']['num_minibatch'], -1]
                            + list(x.shape[2:]),
                        ),
                        1,
                        0,
                    ),
                    shuffled_batch,
                )
                _,total_loss = nnx.scan(_update_minbatch,in_axes=(nnx.Carry,0),out_axes=(nnx.Carry,0))((network,optimizer,jnp.zeros((1,))), minibatches)
                update_state = (
                    jax.tree_util.tree_map(lambda x: x.squeeze(), init_hstate),
                    traj_batch,
                    advantages,
                    targets,
                    rng,
                )
                return (network, optimizer, update_state), total_loss

            update_state = (
                initial_hstate,
                traj_batch,
                advantages,
                targets,
                rng,
            )
            (network, optimizer, update_state), loss_info = nnx.scan(_update_epoch,length=config['params']['algorithm']['num_epochs'],in_axes=(nnx.Carry,None),out_axes=(nnx.Carry,0))((network,optimizer,update_state), None)
            def _trigger_callback_iter():
                metric = {
                    'partner_return': jnp.sum(traj_batch.info['original_reward'].squeeze(),axis=0).mean(),
                    'partner_shaped_return': jnp.sum(traj_batch.info['shaped_reward'].squeeze(),axis=0).mean(),
                    'partner_entropy': loss_info[-1][-1].mean(),
                }
                jax.debug.callback(callback,
                                   nnx.state(network),
                                   nnx.state(optimizer),
                                   iter_manager,
                                   metric,
                                   step,
                                   )
            def _trigger_callback_best():
                metric = {
                    'best_partner_return': jnp.sum(traj_batch.info['original_reward'].squeeze(),axis=0).mean(),
                    'best_partner_shaped_return': jnp.sum(traj_batch.info['shaped_reward'].squeeze(),axis=0).mean(),
                    'best_partner_entropy': loss_info[-1][-1].mean(),
                }
                jax.debug.callback(callback,
                                   nnx.state(network),
                                   nnx.state(optimizer),
                                   best_manager,
                                   metric,
                                   step,
                                   )
            jax.lax.cond(step % config['params']['save_interval'] == 0,
                         _trigger_callback_iter,
                         lambda : None,)
            jax.lax.cond(jnp.sum(traj_batch.info['original_reward'].squeeze(),axis=0).mean() > best_return,
                            _trigger_callback_best,
                            lambda : None,)
            current_step = step + 1
            best_return = jnp.maximum(jnp.sum(traj_batch.info['original_reward'].squeeze(),axis=0).mean(),best_return)
            carry = (network, optimizer, initial_hstate, rng, current_step, best_return)
            return carry, None
        
        # network와 optimizer를 carry에 포함시켜 nnx.scan 사용
        initial_carry = (network, optimizer, initial_hstate, rng, jnp.array(0, dtype=jnp.int32), jnp.array(-1e9))
        
        # nnx.jit으로 전체 scan을 감싸서 최적화
        @nnx.jit
        def run_training(carry, xs):
            return nnx.scan(
                _update_step,
                in_axes=(nnx.Carry, None),
                out_axes=(nnx.Carry, 0),
                length=config['params']['num_iterations']
            )(carry, xs)
        
        final_carry, _ = run_training(initial_carry, None)
        
        return None

    return train

def main(args):
    # LOAD CONFIG
    config = OmegaConf.load(f'./configs/{args.env}.yaml')
    config = config.partners
    config = OmegaConf.to_container(config, resolve=True)
    os.makedirs(config['path']['out_dir'], exist_ok=True)
    # INIT WANDB
    wandb.init(**config['wandb'], config=config)
    # SETUP TRAINING
    rng = jax.random.PRNGKey(args.base_num)
    rngs,_ = jax.random.split(rng)
    train_fn = make_train(config)
    out = train_fn(rngs,args.base_num)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--env', type=str, default='ovc_counter_circuit')
    parser.add_argument('--base_num', type=int, default=0)
    args = parser.parse_args()
    main(args)