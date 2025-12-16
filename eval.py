from contextlib import nullcontext
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

from recon_module.evdiential import build_model

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

class working_agent(nn.Module):
    def __init__(self, agent,recon_module,atype,back_type,shape,ctx):
        super().__init__()
        self.agent = agent
        self.recon_module = recon_module
        self.atype = atype
        self.ctx = ctx
        self.back_type = back_type
        self.shape = shape
        self.device = next(agent.parameters()).device

    def forward(self,obs,hx,cx):
        if self.atype =='ev':
            obs1_ = obs.float().reshape(obs.shape[0],1,-1).to(self.device)
            with self.ctx:
                recon_obs,epismetic,aleatoric = self.recon_module(obs1_)
            epismetic = epismetic.squeeze(-1).reshape(obs1_.shape[0],*self.shape).to(self.device)
            aleatoric = aleatoric.squeeze(-1).reshape(obs1_.shape[0],*self.shape).to(self.device)
            epismetic = epismetic.mean(dim=-1,keepdim=True)
            aleatoric = aleatoric.mean(dim=-1,keepdim=True)
            recon_obs = recon_obs.reshape(obs1_.shape[0],*self.shape).to(self.device)
            obs1_ = torch.cat([recon_obs, epismetic,aleatoric], dim=-1)
            obs1_ = obs1_.permute(0,3,1,2)
        elif self.atype =='noev':
            obs1_ = obs.float().reshape(obs.shape[0],1,-1).to(self.device)
            with self.ctx:
                recon_obs,_ = self.recon_module(obs1_)
            recon_obs = recon_obs.reshape(obs1_.shape[0],*self.shape).to(self.device)
            obs1_ = recon_obs.permute(0,3,1,2).float()
        else:
            obs1_ = obs.float().permute(0,3,1,2).to(self.device)
        if self.atype == 'sp':
            logit, next_hx,next_cx = self.agent(obs1_, hx, cx)
            value=None
        elif self.back_type == 'lstm':
            logit, value,(next_hx,next_cx) = self.agent(obs1_, hx, cx)
        else:
            logit, value = self.agent(obs1_)
            next_hx, next_cx = hx, cx

        return logit, value,(next_hx,next_cx)
            

        


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
        #obs1_ = j2t(obs1).float().permute(0,3,1,2).to(device)
        obs2_ = j2t(obs2).float().permute(0,3,1,2).to(device)
        hx1_ = j2t(hx1).float().to(device)
        cx1_ = j2t(cx1).float().to(device)
        hx2_ = j2t(hx2).float().to(device)
        cx2_ = j2t(cx2).float().to(device)
        p_ids = j2t(partner_ids).long().to(device)
        logit, value,(next_h1,next_c1) = agent(j2t(obs1), hx1_, cx1_)
        batch_params = get_batch_params(partner_params, p_ids)
        batch_buffer = get_batch_params(partner_buffers, p_ids)
        logits_partner, next_h2, next_c2 = inference_core_vmap(
            partner_base_model, batch_params, batch_buffer, 
            obs2_, hx2_, cx2_
        )
        d = Categorical(logits=logit.squeeze().float())
        d2 = Categorical(logits=logits_partner.float())
        action_ = d.sample()
        action2_ = d2.sample()
        next_h1 = t2j(next_h1)
        next_c1 = t2j(next_c1)
        next_h2 = t2j(next_h2)
        next_c2 = t2j(next_c2)
        actions = t2j(torch.cat([action_.unsqueeze(1),action2_.unsqueeze(1)],dim=1))
    return actions, next_h1,next_c1,next_h2,next_c2

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
                jax.ShapeDtypeStruct((1,batch_size, 256), jnp.float32),  # hx
                jax.ShapeDtypeStruct((1,batch_size, 256), jnp.float32),  # cx
                jax.ShapeDtypeStruct((batch_size, 256), jnp.float32),  # hx_partner
                jax.ShapeDtypeStruct((batch_size, 256), jnp.float32),  # cx_partner
            )
            # Wrapper so we only pass JAX arrays (render_state, hx, cx); vae & ac captured (static)
            policy_action, hx1_,cx1_,hx2_,cx2_= io_callback(
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
                next_done['__all__'],
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
                jax.ShapeDtypeStruct((1,batch_size, 256), jnp.float32),  # hx
                jax.ShapeDtypeStruct((1,batch_size, 256), jnp.float32),  # cx
                jax.ShapeDtypeStruct((batch_size, 256), jnp.float32),  # hx_partner
                jax.ShapeDtypeStruct((batch_size, 256), jnp.float32),  # cx_partner
            )
            # Wrapper so we only pass JAX arrays (render_state, hx, cx); vae & ac captured (static)
            policy_action, hx1_,cx1_,hx2_,cx2_= io_callback(
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
                jnp.ones((batch_size,), dtype=jnp.bool), 
                hx1_,
                cx1_,
                hx2_,
                cx2_,
                partner_ids,
            )
        

        next_obs1,next_obs2,next_state, reward, next_done, hx1,cx1,hx2,cx2,partner_ids = jax.lax.cond(
            done.all(),
            skip_step_if_done,
            step_if_not_done,
            state,  # Pass the current state to the chosen function,
            obs1,
            obs2
        )
        new_carry = (key, next_state, next_done, next_obs1,next_obs2, hx1,cx1,hx2,cx2,partner_ids)
        outputs = reward
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
    return outputs

def collect_episode(env,callback_fn,batch_size,seed,episode_length=400):
    with torch.no_grad():
        key = jax.random.PRNGKey(seed)
        rewards = run_episode_p(env, key, episode_length,callback_fn,batch_size)
    return j2t(rewards)


def main(args): 
    config = OmegaConf.load(f'./configs/{args.env}.yaml')
    partner_config = config.partners
    recon_config = config.recon_module
    config = config.fcp if args.atype == 'fcp' else config.fcp_recon
    config = OmegaConf.to_container(config, resolve=True)
    config['params']['batch_size']= 4
    if args.atype !='fcp':
        recon_config['params']['model']['gpt_config']['batch_size'] = config['params']['batch_size']
        recon_config['path']['ckpt_path'] = config['path']['recon_ckpt_path']
    base_seed = config['params']['base_seed']
    set_seed(base_seed)
    device = "cuda:0"
    partner_pool = []
    if args.env=='ovc':
        from alg.cnn_mappo import cell_backbone as partner
    else:
        from alg.cnn_mappo_paper import cell_backbone as partner
    for i in range(config['params']['num_partners'],config['params']['num_partners']+4):
        t = torch.load(os.path.join(config['path']['partner_dir'],f'ovc_{i}','best_model.pt'), map_location='cpu')['agent_state_dict']
        t= remove_actor2_prefix(t)
        p = partner(**partner_config['params']['algorithm'],device=device).to(device)
        p.eval()
        load_partner_weights(p,t)
        partner_pool.append(p)
    partner_base_model = partner(**partner_config['params']['algorithm'],device=device).to('meta')
    partner_base_model.lstm.flatten_parameters = lambda: None
    all_params,all_buffers = stack_module_state(partner_pool)
    if args.env=='ovc':
        l =Layout.from_string(custom_layout_str)
        env = make("overcooked_v2", layout=l, agent_view_size = 2,random_agent_positions=True,random_reset = False,
                max_steps=config['params']['max_episode_length'],sample_recipe_on_delivery=True)
        if args.atype=='fcp':
            from alg.ppo import ppo
        elif args.backtype=='lstm':
            from alg.ppo import ppo_large as ppo
        else:
            from alg.ppo import ppo_linear as ppo
    else:
        env = make("overcooked_v2", layout='test_time_simple', agent_view_size = 2,random_agent_positions=True,random_reset = False,
               max_steps=config['params']['max_episode_length'],sample_recipe_on_delivery=False,indicate_successful_delivery=True,negative_rewards=True)
        env.layout.agent_positions = env.layout.agent_positions[::-1]  # Swap starting positions
        if args.atype=='fcp':
            from alg.ppo_paper import ppo
        elif args.backtype=='lstm':
            from alg.ppo_paper import ppo_large as ppo
        else:
            from alg.ppo_paper import ppo_linear as ppo
    if args.atype != 'sp':
        agent = ppo(**config['params']['algorithm'],device=device).to(device)
    else:
        agent = partner(**partner_config['params']['algorithm'],device=device).to(device)
    if args.atype in ['ev','noev']:
        task = 'no_evidential_classification' if args.atype=='noev' else 'classification'
        recon_model,_,_,_,_,_,_ = build_model(recon_config,device,dtype=torch.float16,device_type='cuda',task=task,is_kv = True)
    else:
        recon_model = None
    shape = (5,11,40) if args.env=='ovc' else (5,8,41)
    agent_net = working_agent(agent,recon_model,args.atype,args.backtype,shape,torch.amp.autocast(device_type='cuda', dtype=torch.float16) if args.atype in ['ev','noev'] else nullcontext())
    start_iter = 0 
    ckpt = None
    torch.backends.cuda.matmul.allow_tf32 = True  # allow tf32 on matmul
    torch.backends.cudnn.allow_tf32 = True  # allow tf32 on cudnn
    if args.atype != 'sp':
        ckpt = torch.load(config['path']['ckpt_path'], map_location=device)
        agent.load_state_dict(ckpt['agent_state_dict'])
    else:
        t = torch.load(config['path']['ckpt_path'], map_location='cpu')['agent_state_dict']
        t= remove_actor1_prefix(t)
        load_partner_weights(agent,t)
    agent = agent.to(device)
    agent.eval()
    callback_fn = partial(torch_callback,agent=agent_net,
                          partner_params=all_params,
                          partner_base_model=partner_base_model,
                          partner_buffers=all_buffers)
    sum_return = []
    for iter in tqdm(range(100)):
        rwds = collect_episode(env,callback_fn,config['params']['batch_size'],
                                                                 base_seed + iter*config['params']['max_episode_length'],config['params']['max_episode_length'])
        rwds = rwds.squeeze().float().permute(1,0)
        for b in range(rwds.shape[0]):
            sum_return.append(rwds[b].sum().item())
    mean_return = np.mean(sum_return)
    std_return = np.std(sum_return)
    print(f"Mean Return: {mean_return}, Std Return: {std_return}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--env',type=str,default='ovc')
    parser.add_argument('--atype',type=str,default='fcp')
    parser.add_argument('--backtype',type=str,default='lstm')
    args = parser.parse_args()
    main(args)