import os
# 2. Import torch and immediately check if it can see the GPUs.
#os.environ['CUDA_VISIBLE_DEVICES'] = ''
import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
import torch.nn.functional as F
from argparse import ArgumentParser
from time import time
from dataclasses import dataclass
from functools import partial
from typing import List
import multiprocessing as mp
import numpy as np

from tqdm import tqdm
import wandb
from omegaconf import OmegaConf
from alg.mappo import mappo_lstm as lstm

from util import *
from envs.robotic_warehouse.rware.warehouse import (
    Warehouse, 
    RewardType, 
    ObservationType, 
    Direction,
    ImageLayer # <get_true_global_state_vector>를 추가했다면 필요 없지만,
               # <get_true_global_state>를 추가했다면 필요합니다.
)

custom_layout_str = """
...g...
..x.x..
..x.x..
..x.x..
..x.x..
...g...
"""

@dataclass
class episode_data:
    """Processed batch data for training"""
    enc: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
 

def rollout(env,agents,seed):
    obs,_ = env.reset(seed)
    done = False
    obss = []
    obss2 = []
    rewards = []
    actions = []
    actions2 = []
    globals = []
    globals2 = []
    log_probs = []
    log_probs2 = []
    values = []
    while not done:
        obss.append(obs[0])
        obss2.append(obs[1])
        g = env.get_global_state()
        globals.append(g[0])
        globals2.append(g[1])
        obs_agent = torch.from_numpy(obs[0]).to(agents.device).unsqueeze(0).float()
        obs_agent2 = torch.from_numpy(obs[1]).to(agents.device).unsqueeze(0).float()
        logits_action, logits_action2, value = agents(obs_agent,obs_agent2)
        d = Categorical(logits=logits_action.squeeze())
        d2 = Categorical(logits=logits_action2.squeeze())
        action = d.sample()
        action2 = d2.sample()
        log_probs.append(d.log_prob(action).item())
        log_probs2.append(d2.log_prob(action2).item())
        values.append(value.item())
        action_list = [action.cpu().numpy(), action2.cpu().numpy()]
        actions.append(action_list[0])
        actions2.append(action_list[1])
        obs, reward, done,truncated, info = env.step(np.array(action_list))
        rewards.append(reward[0])
    obs_agent = torch.from_numpy(obs[0]).to(agents.device).unsqueeze(0).float()
    obs_agent2 = torch.from_numpy(obs[1]).to(agents.device).unsqueeze(0).float()
    logits_action, logits_action2, value = agents(obs_agent,obs_agent2)
    d = Categorical(logits=logits_action.squeeze())
    d2 = Categorical(logits=logits_action2.squeeze())
    action = d.sample()
    action2 = d2.sample()
    values.append(value.item())


    return np.array(obss), np.array(obss2), np.array(actions), np.array(actions2), np.array(rewards), np.array(globals), np.array(globals2), np.array(log_probs),  np.array(log_probs2), np.array(values)

def collect_episode(model_config,state_dict,device,queue,worker_id,seed,save_path,episode_length=500):
    env =Warehouse(
    shelf_columns=1, 
    column_height=1, 
    shelf_rows=1,
    layout=custom_layout_str,
    n_agents=2,
    msg_bits=0,
    sensor_range=1,
    request_queue_size=5,
    max_inactivity_steps=None,
    max_steps=500,
    reward_type=RewardType.GLOBAL,
    observation_type=ObservationType.FLATTENED,
    normalised_coordinates=True
    )
    with torch.no_grad():
        agent = lstm(**model_config,device=device).to(device)
        agent.load_state_dict(state_dict)
        agent.reset_hidden()
        agent.eval()
        obss, obss2, actions, actions2, rewards, globals, globals2, log_probs, log_probs2, values = rollout(env,agent,seed)
    o1 = np.memmap(os.path.join(save_path,'obs', f'{seed}_0.npy'), mode='w+', dtype=np.uint8, shape=(episode_length,71 ))
    o2 = np.memmap(os.path.join(save_path,'obs', f'{seed}_1.npy'), mode='w+', dtype=np.uint8, shape=(episode_length,71 ))
    g_size = 7*env.n_agents+len(env.shelfs)*3
    g1 = np.memmap(os.path.join(save_path,'global', f'{seed}_0.npy'), mode='w+', dtype=np.uint8, shape=(episode_length, g_size ))
    g2 = np.memmap(os.path.join(save_path,'global', f'{seed}_1.npy'), mode='w+', dtype=np.uint8, shape=(episode_length, g_size ))
    o1[:] = obss.squeeze()[:]
    o2[:] = obss2.squeeze()[:]
    g1[:] = globals.squeeze()[:]
    g2[:] = globals2.squeeze()[:]
    o1.flush()
    o2.flush()
    g1.flush()
    g2.flush()
    queue.put((worker_id,obss,obss2,actions,actions2,rewards,log_probs,log_probs2,values))
    queue.close()


def process_episodes(obss,obss2,acts,acts2,model,old_log_probs,old_log_probs2,adv,device):
    model.reset_hidden(batch_size=obss.shape[0])
    log_probs = torch.zeros_like(old_log_probs).to(device)
    log_probs2 = torch.zeros_like(old_log_probs).to(device)
    values = torch.zeros_like(old_log_probs).to(device)
    entropy = torch.zeros_like(old_log_probs).to(device)
    entropy2 = torch.zeros_like(old_log_probs).to(device)
    logit_actions, logit_actions2, means_values = model(obss, obss2)
    d = Categorical(logits=logit_actions.squeeze())
    d2 = Categorical(logits=logit_actions2.squeeze())
    log_probs = d.log_prob(acts)
    log_probs2 = d2.log_prob(acts2)
    entropy = d.entropy()
    entropy2 = d2.entropy()
    values = means_values.squeeze()
    ratio = (log_probs - old_log_probs).exp()
    ratio2 = (log_probs2 - old_log_probs2).exp()
    loss1 = ratio * adv
    loss2 = torch.clamp(ratio, 1 - 0.1, 1 + 0.1) * adv
    loss1_2 = ratio2 * adv
    loss2_2 = torch.clamp(ratio2, 1 - 0.1, 1 + 0.1) * adv
    loss_action = -torch.min(loss1, loss2).mean()
    loss_action2 = -torch.min(loss1_2, loss2_2).mean()
    entropy = (entropy + entropy2)/2
    loss_actions = (loss_action + loss_action2)/2
    return loss_actions, values, entropy



if __name__ == "__main__":    
    config = OmegaConf.load('./configs/rware.yaml')
    config = config.partners
    config = OmegaConf.to_container(config, resolve=True)
       #  if not ddp, we are running on a single gpu, and one process
    ddp_world_size = 1
    ddp_local_rank = 0
    wandb.init(**config['wandb'],config = config)
    mp.set_start_method('spawn', force=True)
    base_seed = config['params']['base_seed']
    set_seed(base_seed)
    device = "cuda:0"
    agent = lstm(**config['params']['algorithm'],device=device).to(device)
    start_iter = 0
    ckpt = None
    if config['path']['ckpt_path'] is not None:
        ckpt = torch.load(config['path']['ckpt_path'], map_location=device)
        agent.load_state_dict(ckpt['agent_state_dict'])
        start_iter = ckpt['iter'] + 1
    agent = agent.to(device)
    agent.eval()
    partner_dict = []
    torch.backends.cuda.matmul.allow_tf32 = True  # allow tf32 on matmul
    torch.backends.cudnn.allow_tf32 = True  # allow tf32 on cudnn
    best_return = -1e9
    os.makedirs(config['path']['out_dir'], exist_ok=True)
    save_path = config['path']['save_dir']
    os.makedirs(save_path, exist_ok=True)
    os.makedirs(os.path.join(save_path,'obs'), exist_ok=True)
    os.makedirs(os.path.join(save_path,'global'), exist_ok=True)
    optimizer = build_optimizer(agent,ckpt ,**config['params']['optimizer'],num_agent=-1)
    
    for iter in tqdm(range(start_iter,config['params']['num_iterations'])):
        queue = mp.Queue()
        processes = []
        state_dict = agent.state_dict() 
        cpu_dict = {k: v.cpu() for k, v in state_dict.items()}  # Move to CPU for process sharing
        workers = [i for i in range(config['params']['num_workers'])] 
        for worker_id in range(config['params']['num_workers']):
            p = mp.Process(target=collect_episode, args=(config['params']['algorithm'], cpu_dict, f'cuda:{worker_id%2}',
                                                          queue, worker_id, base_seed+iter*config['params']['num_workers']+worker_id,save_path,
                                                          config['params']['max_episode_length']))
            p.start()
            processes.append(p)
        obss,obss2,acts,acts2,rwds,log_probs,values,log_probs2 = [],[],[],[],[],[],[],[]
        while len(workers) > 0:
            worker_id, obs,obs2,act,act2,rwd,log_prob,log_prob2,value = queue.get()
            obss.append(obs)
            obss2.append(obs2)
            acts.append(act)
            acts2.append(act2)
            rwds.append(rwd)
            log_probs.append(log_prob)
            values.append(value)
            log_probs2.append(log_prob2)
            workers.remove(worker_id)
        for p in processes:
            p.join()
        obss = torch.from_numpy(np.stack(obss)).to(device)
        obss2 = torch.from_numpy(np.stack(obss2)).to(device)
        acts = torch.from_numpy(np.stack(acts)).squeeze().long().to(device)
        acts2 = torch.from_numpy(np.stack(acts2)).squeeze().long().to(device)
        rwds = torch.from_numpy(np.stack(rwds)).squeeze().float().to(device)
        log_probs = torch.from_numpy(np.stack(log_probs)).squeeze().float().to(device)
        values = torch.from_numpy(np.stack(values)).squeeze().float().to(device)
        log_probs2 = torch.from_numpy(np.stack(log_probs2)).squeeze().float().to(device)
        with torch.no_grad():
            lambda_return,adv= compute_lambda_returns(
                rwds= rwds,
                values= values,
                gamma=config['params']['loss']['gamma'],
                lambda_=config['params']['loss']['lambda_'],
            )
            mean_rewards = rwds.sum(dim=1).mean()
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        agent.train()
        for i in range(10):
            loss_actions, means_values, entropy = process_episodes(obss,obss2, acts,acts2, agent, log_probs,log_probs2, adv, device)
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
                'partner_return': mean_rewards.item(),
                'partner_loss_actions': losses.intermediate_losses['loss_actions'],
                'partner_loss_values': losses.intermediate_losses['loss_values'],
                'partner_loss_entropy': losses.intermediate_losses['loss_entropy'],
            },step=iter)
        if  (iter+1) % config['params']['save_interval'] == 0:
            torch.save({
                'agent_state_dict': state_dict,
                'optimizer_state_dict': optimizer.state_dict(),
                'iter': iter,
            }, os.path.join(config['path']['out_dir'], f'model_iter_{iter}.pt'))
        if mean_rewards.item() > best_return:
            best_return = mean_rewards.item()
            torch.save({
                'agent_state_dict': state_dict,
                'optimizer_state_dict': optimizer.state_dict(),
                'iter': iter,
            }, os.path.join(config['path']['out_dir'], 'best_model.pt'))
    torch.save({
        'agent_state_dict': state_dict,
        'optimizer_state_dict': optimizer.state_dict(),
        'iter': iter,
    }, os.path.join(config['path']['out_dir'], 'final_model.pt'))
    wandb.finish()





