import os
# 2. Import torch and immediately check if it can see the GPUs.
#os.environ['CUDA_VISIBLE_DEVICES'] = ''
import torch
from torch.distributions.categorical import Categorical
import torch.nn.functional as F
from functools import partial
import numpy as np


from tqdm import tqdm
import wandb
from omegaconf import OmegaConf
from alg.cnn_mappo import backbone as lstm
from alg.cnn_mappo import cell_backbone as partner
from alg.ppo import ppo

from util import *
import jax
#jax.config.update("jax_platform_name", "cpu")
from jaxmarl import make
from jaxmarl.environments.overcooked_v2.layouts import Layout
import jax.numpy as jnp
from functools import partial
from torch.func import functional_call, vmap, stack_module_state
from jax.experimental import io_callback


custom_layout_str = """
WXWWWWWWWXW
0    R    0
1   APA   1
2    R    2
WBWWWWWWWBW
"""


def get_batch_params(all_params, partner_indices):
    return {k: v[partner_indices] for k, v in all_params.items()}


def inference_core_vmap(base_model, batch_params, batch_buffers, obs, hx, cx):
    """
    base_model: 모델의 구조를 가진 객체 (껍데기). vmap 내부에서 고정됨.
    batch_params: (Batch, ...) 형태의 가중치 뭉치
    obs, hx, cx: (Batch, ...) 형태의 데이터
    """

    # [중요] 단일 샘플(하나의 파트너 가중치 + 하나의 데이터)을 처리하는 함수 정의
    def call_single(params, buffers, x, h, c):
        # 여기서 base_model이 사용됩니다!
        # base_model이라는 '틀'에, params라는 '내용물'을 끼워넣고, (x, h, c)를 입력으로 넣는 함수입니다.
        # functional_call(모델껍데기, (가중치, 버퍼), 입력데이터)
        return functional_call(base_model, (params, buffers), (x, h, c))
    # vmap 설정
    # params, buffers: 0번 차원(Batch)에 따라 병렬화
    # obs: 0번 차원 병렬화
    # hx, cx: 0번 차원 병렬화 (Batch axis)
    compute_batch = vmap(call_single, in_dims=(0, 0, 0, 0, 0))

    return compute_batch(batch_params, batch_buffers, obs, hx, cx)

def torch_callback(obs1,hx1,cx1,obs2,hx2,cx2,partner_ids,agent,partner_base_model,partner_params,partner_buffers):
    device = next(agent.parameters()).device
    # Callback runs on host (outside JIT). Inputs arrive as (possibly read-only) NumPy views from JAX.
    # Copy to ensure writable memory before torch.from_numpy to avoid UserWarning about non-writable arrays.
    with torch.inference_mode(),torch.no_grad():
        obs1_ = j2t(obs1).float().permute(0,3,1,2).to(device)
        obs2_ = j2t(obs2).float().permute(0,3,1,2).to(device)
        hx1_ = j2t(hx1).float().to(device)
        cx1_ = j2t(cx1).float().to(device)
        hx2_ = j2t(hx2).float().to(device)
        cx2_ = j2t(cx2).float().to(device)
        p_ids = j2t(partner_ids).long().to(device)
        logit, value,(next_h1,next_c1) = agent(obs1_, hx1_, cx1_)
        batch_params = get_batch_params(partner_params, p_ids)
        batch_buffer = get_batch_params(partner_buffers, p_ids)
        logits_partner, next_h2, next_c2 = inference_core_vmap(
            partner_base_model, batch_params, batch_buffer, obs2_, hx2_, cx2_
        )
        d = Categorical(logits=logit.squeeze().float())
        d2 = Categorical(logits=logits_partner.float())
        action_ = d.sample()
        action2_ = d2.sample()
        log_prob = d.log_prob(action_)
        actions = torch.cat([action_.unsqueeze(1),action2_.unsqueeze(1)],dim=1)
    return t2j(actions),t2j(log_prob),t2j(value.squeeze()), t2j(next_h1), t2j(next_c1), t2j(next_h2), t2j(next_c2)


@partial(jax.jit, static_argnames=("env", "episode_length", "callback_fn","batch_size"))
def run_episode_p(env, key, episode_length, callback_fn,batch_size):

    """ JIT-compiled function to run one full episode. """
    
    def _step_fn(carry, _):
        key, state, done, obs1, obs2, hx1, cx1, hx2, cx2,partner_ids = carry

        # 1. Select Action
        key,_step_key = jax.random.split(key)
        step_key = jax.random.split(_step_key,num=batch_size)
        def step_if_not_done(current_state,current_obs1,current_obs2):
            # Shapes/dtypes expected from callback
            result_shape_dtype = (
                jax.ShapeDtypeStruct((batch_size,2), jnp.int32),      # policy_action
                jax.ShapeDtypeStruct((batch_size,), jnp.float32),     # logprob
                jax.ShapeDtypeStruct((batch_size,), jnp.float32),      # value,
                jax.ShapeDtypeStruct((1,batch_size, 256), jnp.float32),  # hx
                jax.ShapeDtypeStruct((1,batch_size, 256), jnp.float32),  # cx
                jax.ShapeDtypeStruct((batch_size, 256), jnp.float32),  # hx_partner
                jax.ShapeDtypeStruct((batch_size, 256), jnp.float32),  # cx_partner
            )
            # Wrapper so we only pass JAX arrays (render_state, hx, cx); vae & ac captured (static)
            policy_action, log_prob,value,hx1_,cx1_,hx2_,cx2_= io_callback(
                callback_fn,
                result_shape_dtype,
                obs1,hx1,cx1,
                obs2,hx2,cx2,
                partner_ids,
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
                next_done['__all__'],
                policy_action[:,0],
                log_prob,
                value,
                hx1_,
                cx1_,
                hx2_,
                cx2_,
                partner_ids,
            )

        def skip_step_if_done(current_state,current_obs1,current_obs2):
            """Return the same state and zero rewards if the episode has ended."""
            result_shape_dtype = (
                jax.ShapeDtypeStruct((batch_size,2), jnp.int32),      # policy_action
                jax.ShapeDtypeStruct((batch_size,), jnp.float32),     # logprob
                jax.ShapeDtypeStruct((batch_size,), jnp.float32),      # value,
                jax.ShapeDtypeStruct((1,batch_size, 256), jnp.float32),  # hx
                jax.ShapeDtypeStruct((1,batch_size, 256), jnp.float32),  # cx
                jax.ShapeDtypeStruct((batch_size, 256), jnp.float32),  # hx_partner
                jax.ShapeDtypeStruct((batch_size, 256), jnp.float32),  # cx_partner
            )
            # Wrapper so we only pass JAX arrays (render_state, hx, cx); vae & ac captured (static)
            policy_action, log_prob,value,hx1_,cx1_,hx2_,cx2_= io_callback(
                callback_fn,
                result_shape_dtype,
                obs1,hx1,cx1,
                obs2,hx2,cx2,
                partner_ids,
                ordered=True  # Ensures steps execute in order.
            )
            return (
                current_obs1,
                current_obs2,
                current_state,
                jnp.zeros((batch_size,),dtype=jnp.float32),          # Zero shaped reward
                jnp.zeros((batch_size,),dtype=jnp.float32),          
                jnp.ones((batch_size,), dtype=jnp.bool), 
                jnp.zeros((batch_size,),dtype=jnp.int32),          # Keep previous actions
                jnp.zeros((batch_size,),dtype=jnp.float32),              # Dummy logit
                value,
                hx1_,
                cx1_,
                hx2_,
                cx2_,
                partner_ids,
            )
        

        next_obs1,next_obs2,next_state, reward, shaped_reward, next_done, chosen_actions,log_prob,value, hx1,cx1,hx2,cx2,partner_ids = jax.lax.cond(
            done.all(),
            skip_step_if_done,
            step_if_not_done,
            state,  # Pass the current state to the chosen function,
            obs1,
            obs2
        )
        new_carry = (key, next_state, next_done, next_obs1,next_obs2, hx1,cx1,hx2,cx2,partner_ids)
        outputs = (obs1,reward, chosen_actions, shaped_reward, log_prob, value)
        return new_carry, outputs

    # Initial state
    key, _reset_key = jax.random.split(key)
    reset_key = jax.random.split(_reset_key,num=batch_size)
    initial_obs, initial_state = jax.vmap(env.reset,in_axes=(0,))(reset_key)
    initial_obs1 = initial_obs['agent_0']
    initial_obs2 = initial_obs['agent_1']
    initial_dones = jnp.zeros((batch_size,), dtype=jnp.bool)
    initial_hx1 = jnp.zeros((1,batch_size,256),dtype=jnp.float32)
    initial_cx1 = jnp.zeros((1,batch_size,256),dtype=jnp.float32)
    initial_hx2 = jnp.zeros((batch_size,256),dtype=jnp.float32)
    initial_cx2 = jnp.zeros((batch_size,256),dtype=jnp.float32)

    # Run the scan over the episode length
    partner_ids = jax.random.randint(key, (batch_size,), 0, batch_size)
    initial_carry = (key, initial_state, initial_dones, initial_obs1,initial_obs2, initial_hx1, initial_cx1, initial_hx2, initial_cx2, partner_ids)
    _, outputs = jax.lax.scan(_step_fn, initial_carry, None, length=episode_length)
    obs1,rewards, actions, shaped_rewards,log_probs,values = outputs
    return obs1,rewards, actions, shaped_rewards,log_probs, values


def collect_episode(env,callback_fn,batch_size,seed,episode_length=400):
    with torch.no_grad():
        key = jax.random.PRNGKey(seed)
        obss1,rewards, actions, shaped_rewards,log_probs,values = run_episode_p(env, key, episode_length,callback_fn,batch_size)
    return j2t(obss1[:-1]),j2t(actions[:-1]),j2t(rewards[:-1]),j2t(shaped_rewards[:-1]),j2t(log_probs[:-1]),j2t(values)


def process_episodes(obss,acts,model,old_log_probs,adv,device):
    hx = torch.zeros((1,obss.shape[0],model.lstm_dim)).to(device)
    cx = torch.zeros((1,obss.shape[0],model.lstm_dim)).to(device)
    entropy = torch.zeros_like(old_log_probs).to(device)
    logit_actions, values,(_,_) = model(obss, hx, cx)
    d = Categorical(logits=logit_actions.squeeze())
    log_probs = d.log_prob(acts)
    entropy = d.entropy()
    ratio = (log_probs - old_log_probs).exp()
    loss1 = ratio * adv
    loss2 = torch.clamp(ratio, 1 - 0.1, 1 + 0.1) * adv
    loss_action = -torch.min(loss1, loss2).mean()
    return loss_action, values.squeeze(), entropy



def main(args):
    config = OmegaConf.load(f'./configs/{args.env}.yaml')
    partner_config = config.partners
    config = config.fcp
    config = OmegaConf.to_container(config, resolve=True)
       #  if not ddp, we are running on a single gpu, and one process
    wandb.init(**config['wandb'],config = config)
    base_seed = config['params']['base_seed']
    set_seed(base_seed)
    device = "cuda:0"
    partner_pool = []
    fcp_keys = ['best_model.pt','model_iter_199.pt','model_iter_999.pt']
    for i in range(config['params']['num_partners']):
        for k in fcp_keys:
            t = torch.load(os.path.join(config['path']['partner_dir'],f'ovc_{i}',k), map_location='cpu')['agent_state_dict']
            t= remove_actor2_prefix(t)
            p = partner(**partner_config['params']['algorithm'],device=device).to(device)
            p.eval()
            load_partner_weights(p,t)
            partner_pool.append(p)
    partner_base_model = partner(**partner_config['params']['algorithm'],device=device).to('meta')
    partner_base_model.lstm.flatten_parameters = lambda: None
    all_params,all_buffers = stack_module_state(partner_pool)
    agent = ppo(**config['params']['algorithm'],device=device).to(device)
    l =Layout.from_string(custom_layout_str)
    env = make("overcooked_v2", layout=l, agent_view_size = 2,random_agent_positions=True,random_reset = False,
               max_steps=config['params']['max_episode_length'],sample_recipe_on_delivery=True)
    start_iter = 0
    ckpt = None
    torch.backends.cuda.matmul.allow_tf32 = True  # allow tf32 on matmul
    torch.backends.cudnn.allow_tf32 = True  # allow tf32 on cudnn
    best_return = -1e9
    os.makedirs(config['path']['out_dir'], exist_ok=True)
    optimizer = build_optimizer(agent,**config['params']['optimizer'])
    if config['path']['ckpt_path'] is not None:
        ckpt = torch.load(config['path']['ckpt_path'], map_location=device)
        agent.load_state_dict(ckpt['agent_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        start_iter = ckpt['iter'] + 1
    agent = agent.to(device)
    agent.eval()
    callback_fn = partial(torch_callback,agent=agent,
                          partner_params=all_params,
                          partner_base_model=partner_base_model,
                          partner_buffers=all_buffers)
    for iter in tqdm(range(start_iter,config['params']['num_iterations'])):
        obss,acts,rwds,shaped1,log_probs,values = collect_episode(env,callback_fn,config['params']['batch_size'],
                                                                 base_seed + iter*config['params']['max_episode_length'],config['params']['max_episode_length'])
        obss = torch.from_numpy(obss).float().permute(1,0,4,2,3).to(device)
        acts = torch.from_numpy(acts).squeeze().long().permute(1,0).to(device)
        rwds = torch.from_numpy(rwds).squeeze().float().permute(1,0).to(device)
        shaped1 = torch.from_numpy(shaped1).squeeze().float().permute(1,0).to(device)
        log_probs = torch.from_numpy(log_probs).squeeze().float().permute(1,0).to(device)
        values = torch.from_numpy(values).squeeze().float().permute(1,0).to(device)
        with torch.no_grad():
            lambda_return,adv1= compute_lambda_returns(
                rwds= rwds+shaped1,
                values= values,
                gamma=config['params']['loss']['gamma'],
                lambda_=config['params']['loss']['lambda_'],
            )
            mean_rewards = rwds.sum(dim=1).mean()
            adv1 = (adv1 - adv1.mean()) / (adv1.std() + 1e-8)
        agent.train()
        for i in range(10):
            loss_actions, means_values, entropy = process_episodes(obss, acts, agent, log_probs, adv1, device)
            loss_values = F.mse_loss(means_values, lambda_return)
            loss_entropy = entropy.mean()
            losses = LossWithIntermediateLosses(
                loss_actions=loss_actions,
                loss_values=loss_values*0.5,
                loss_entropy=-config['params']['loss']['entropy_weight'] * loss_entropy,
            )
            loss_total_step = losses.loss_total
            optimizer.zero_grad()
            loss_total_step.backward()
            torch.nn.utils.clip_grad_norm_(agent.parameters(), config['params']['max_grad_norm'])
            optimizer.step()
        if (iter+1)% config['log_interval'] == 0:
            wandb.log({
                'fcp_return': mean_rewards.item(),
                'fcp_loss_actions': losses.intermediate_losses['loss_actions'],
                'fcp_loss_values': losses.intermediate_losses['loss_values'],
                'fcp_loss_entropy': losses.intermediate_losses['loss_entropy'],
            },step=iter)
        if  (iter+1) % config['params']['save_interval'] == 0:
            torch.save({
                'agent_state_dict': agent.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'iter': iter,
            }, os.path.join(config['path']['out_dir'], f'model_iter_{iter}.pt'))
        if mean_rewards.item() > best_return:
            best_return = mean_rewards.item()
            torch.save({
                'agent_state_dict': agent.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'iter': iter,
            }, os.path.join(config['path']['out_dir'], 'best_model.pt'))
    torch.save({
        'agent_state_dict': agent.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'iter': iter,
    }, os.path.join(config['path']['out_dir'], 'final_model.pt'))
    wandb.finish()





