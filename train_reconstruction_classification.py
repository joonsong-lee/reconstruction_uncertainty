import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
from recon_module.data import rc_dataset
from recon_module.evdiential import build_model
from util import *
from envs.ovc.visualizer import seq_to_seq_viz

from jaxmarl import make
from jaxmarl.environments.overcooked_v2.layouts import Layout

import argparse
from contextlib import nullcontext
import os
from time import time
import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group,broadcast,barrier,reduce,broadcast
import torch.distributed as dist
from tqdm import tqdm,trange
from omegaconf import OmegaConf
import wandb


class rc_trainer:
    def __init__(self,model,optimizer,scaler,ctx,train_sampler,train_loader,val_loader,config,local_rank,ddp_world_size,best_val_loss,device):
        self.model = model
        self.optimizer = optimizer
        self.scaler = scaler
        self.ctx = ctx
        self.config = config
        self.device = device
        self.train_sampler = train_sampler
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.local_rank = local_rank
        self.ddp_world_size = ddp_world_size
        self.master_process = self.local_rank == 0
        self.best_val_loss = best_val_loss
        self.early_stop = torch.tensor([0],device=device)
        self.save = 0
        self.out_dir = config['path']['out_dir']
        if self.master_process:
            custom_layout_str = """
WXWWWWWWWXW
0    R    0
1   APA   1
2    R    2
WBWWWWWWWBW
"""
            l =Layout.from_string(custom_layout_str)
            self.env = make("overcooked_v2", layout=l, agent_view_size = 2,random_agent_positions=True,random_reset = False,
                max_steps=400,sample_recipe_on_delivery=True,indicate_successful_delivery=False,negative_rewards=False)

            print(self.env.height,self.env.width)
            # self.env = make("overcooked_v2", layout='test_time_simple', agent_view_size = 2,random_agent_positions=True,random_reset = False,
            #    max_steps=400,sample_recipe_on_delivery=False,indicate_successful_delivery=True,negative_rewards=True)
            # self.env.layout.agent_positions = self.env.layout.agent_positions[::-1]  # Swap starting positions

    def train_step(self, iternum,data):
        X,Y = data
        with self.ctx:
            mu,alpha,(total_loss,mse,reg) = self.model(X.to(self.device), Y.to(self.device))
            if self.config['params']['decay_lr']:
                lr = get_lr(iternum, **self.config['params']['scheduler'])
                for param_group in self.optimizer.param_groups:
                    param_group["lr"] = lr
        self.scaler.scale(total_loss).backward()
        if self.config['params']['grad_clip'] != 0.0:
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config['params']['grad_clip'])
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad(set_to_none=True)
        return total_loss,reg,mse

    def log_and_save(self, metrics_avg, epoch, iter_num, total_iter):
        wandb.log({
            f"val_loss": metrics_avg['val_loss_total'].item(),
            f"val_mse": metrics_avg['val_mse'].item(),
            f"val_reg_loss": metrics_avg['val_loss_reg'].item(),
            f"val_epistemic_uncertainty": metrics_avg['val_epistemic_uncertainty'].item(),
            f"val_aleatoric_uncertainty": metrics_avg['val_aleatoric_uncertainty'].item(),
        }, step=total_iter)

        checkpoint = {
            "model": self.model.state_dict() if self.ddp_world_size == 1 else self.model.module.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scaler": self.scaler.state_dict(),
            "iter": total_iter,
            "epoch": epoch,
            'iter_num': iter_num,
            "best_val_loss": self.best_val_loss,
        }
        if ((total_iter//self.config['eval_every']) % self.config['save_interval'] == 0):
            torch.save(
                checkpoint,
                os.path.join(self.config['path']['out_dir'], f"wm_epoch_{epoch}_iter_{total_iter}.pt")
            )
        if metrics_avg['val_loss_total'] < self.best_val_loss:
            self.best_val_loss = metrics_avg['val_loss_total']
            torch.save(checkpoint, os.path.join(self.config['path']['out_dir'], 'best_model.pt')) 
            wandb.run.summary['best_val_loss'] = self.best_val_loss
            self.early_stop.mul_(0)
        else:
            self.early_stop.add_(1)

    def evaluate_and_save(self, epoch, iter_num, total_iter,partial=False):
        self.model.eval()
        metrics_avg = {
            'val_loss_total': torch.tensor(0.0, device=self.device), 'val_mse': torch.tensor(0.0, device=self.device),
            'val_loss_nll':torch.tensor(0.0, device=self.device), 'val_loss_reg':torch.tensor(0.0, device=self.device),
            'val_epistemic_uncertainty':torch.tensor(0.0, device=self.device), 'val_aleatoric_uncertainty':torch.tensor(0.0, device=self.device)
        }
        with torch.no_grad(), self.ctx:
            for j,(X, Y) in enumerate(self.val_loader):
                mu,alpha,(total_loss,mse_loss,reg) = self.model(X.to(self.device), Y.to(self.device))
                epistemic_uncertainty, aleatoric_uncertainty = self.model.module.uncertainty(alpha) if self.ddp_world_size>1 else self.model.uncertainty(alpha)
                metrics_avg['val_loss_total'] += total_loss
                metrics_avg['val_mse'] += mse_loss
                metrics_avg['val_loss_reg'] += reg
                metrics_avg['val_epistemic_uncertainty'] += epistemic_uncertainty.mean()
                metrics_avg['val_aleatoric_uncertainty'] += aleatoric_uncertainty.mean()
                if partial and (j+1)>=self.config['eval_step']:
                    break
        # 평균 계산
        for key in metrics_avg:
            metrics_avg[key] /= (j+1)*self.ddp_world_size
        # val_loss = metrics_avg['val_loss'], device=self.device)
        # val_mse = torch.tensor(metrics_avg['val_mse'], device=self.device)
        # val_epistemic_uncertainty = torch.tensor(metrics_avg['epistemic_uncertainty'], device=self.device)
        # val_aleatoric_uncertainty = torch.tensor(metrics_avg['aleatoric_uncertainty'], device=self.device)
        if self.ddp_world_size > 1:
            barrier()
            reduce(metrics_avg['val_loss_total'], dst=0)
            reduce(metrics_avg['val_mse'], dst=0)
            reduce(metrics_avg['val_loss_nll'], dst=0)
            reduce(metrics_avg['val_loss_reg'], dst=0)
            reduce(metrics_avg['val_epistemic_uncertainty'], dst=0)
            reduce(metrics_avg['val_aleatoric_uncertainty'], dst=0)
        if self.master_process:
            self.log_and_save(metrics_avg, epoch, iter_num, total_iter)
            if ((total_iter//self.config['eval_every']) % self.config['save_interval'] == 0):
                rand_idx = np.random.randint(0, X.shape[0])
                y_obs = Y[rand_idx].cpu().numpy()
                mu_sample = mu[rand_idx].cpu().numpy()
                aleatoric_sample = aleatoric_uncertainty[rand_idx].reshape(400,5,11,-1).mean(dim=-1).cpu().numpy()
                epistemic_sample = epistemic_uncertainty[rand_idx].reshape(400,5,11,-1).mean(dim=-1).cpu().numpy()
                print(aleatoric_sample.shape, epistemic_sample.shape)
                seq_to_seq_viz(self.env,y_obs,mu_sample,aleatoric_sample,epistemic_sample,filename=f'{self.out_dir}/recon_iter{total_iter}')

        if self.ddp_world_size > 1:
            broadcast(self.early_stop, src=0)
        self.model.train()
        return 

    def run(self,start_epoch,resume_iter,prev_total_iter):
        total_iter = prev_total_iter
        for epoch in trange(start_epoch, self.config['params']['epochs']):
            if self.train_sampler is not None:
                self.train_sampler.set_epoch(epoch)
            self.model.train()
            for iter, (data) in enumerate(self.train_loader):
                if (iter+1)%self.config['eval_every']==0 and total_iter>0:
                    self.evaluate_and_save(epoch, iter, total_iter,partial=True)
                    if self.early_stop[0] >= self.config['early_stop']:
                        break
                total_loss,reg, mse = self.train_step(total_iter,data)
                if (total_iter % self.config['log_interval'] == 0 and self.local_rank ==0):
                    wandb.log({
                        f"loss_total": total_loss.item(),
                        f"loss_reg": reg.item(),
                        f"loss_mse": mse.item()
                    }, step=total_iter)
                total_iter += 1
            resume_iter = 0
            self.evaluate_and_save(epoch, iter, total_iter,partial=False)
            
        return


def main(args):
    config = OmegaConf.load('./configs/'+args.env+'.yaml').recon_module
    config = OmegaConf.to_container(config, resolve=True)
    train_dataset = rc_dataset('classification','train',**config['params']['dataset'])
    val_dataset = rc_dataset('classification','val',**config['params']['dataset'])
    device, master_process, ddp_rank, ddp_local_rank, ddp_world_size = setup_environment(config,0)
    torch.cuda.set_device(ddp_local_rank)
    train_loader, val_loader, train_sampler, max_iter = setup_dataloaders(config, train_dataset,val_dataset,ddp_world_size)
    device_type = "cuda" if "cuda" in device else "cpu"  # for later use in torch.autocast
    # note: float16 data type will automatically use a GradScaler
    dtype = "float16"
    ptdtype = {
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }[dtype]
    ctx = (
        nullcontext()
        if device_type == "cpu"
        else torch.amp.autocast(device_type=device_type, dtype=ptdtype)
    )
    model,optimizer,scaler,total_iter,epoch,iter_num,best_val_loss = build_model(config,device,dtype,device_type,task='classification',is_kv = False)
    model = model.to(device)
    # wrap model into DDP container
    if ddp_world_size > 1:
        model = DDP(model, device_ids=[ddp_local_rank])
    trainer = rc_trainer(model,optimizer,scaler,ctx,train_sampler,train_loader,val_loader,config,ddp_local_rank,ddp_world_size,best_val_loss,device)
    trainer.run(epoch,iter_num,total_iter)
    if ddp_world_size>1:
        destroy_process_group()
    wandb.finish()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='wm Training')
    parser.add_argument('--env', type=str, default='ovc', help='ovc')
    parser.add_argument('--local_rank', type=int, default=0, help='local rank for distributed training')
    args = parser.parse_args()
    main(args)