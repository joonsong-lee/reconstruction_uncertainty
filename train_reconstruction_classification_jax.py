import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import argparse
from functools import partial
import numpy as np
from tqdm import tqdm, trange
from omegaconf import OmegaConf
import wandb

import jax
import jax.numpy as jnp
import flax.nnx as nnx
import optax
from orbax import checkpoint as ocp


from recon_module.data import rc_dataset
from recon_module.evdiential import build_model
from envs.ovc.visualizer import seq_to_seq_viz

from jaxmarl import make
from jaxmarl.environments.overcooked_v2.layouts import Layout


def numpy_collate(batch):
    """Collate function that returns numpy arrays instead of torch tensors."""
    if isinstance(batch[0], np.ndarray):
        return np.stack(batch)
    elif isinstance(batch[0], (tuple, list)):
        transposed = zip(*batch)
        return [numpy_collate(samples) for samples in transposed]
    else:
        return np.array(batch)


def create_dataloader(dataset, batch_size, shuffle=True, num_workers=0):
    """Create a simple dataloader that yields numpy arrays."""
    from torch.utils.data import DataLoader
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=numpy_collate,
        drop_last=True,
    )


@nnx.jit
def _train_step(model, optimizer, X, Y):
    """Single training step (jitted)."""
    def loss_fn(model):
        mu, alpha, (total_loss, mse, reg) = model(X, Y)
        return total_loss, (mu, alpha, mse, reg)
    
    grad_fn = nnx.value_and_grad(loss_fn, has_aux=True)
    (total_loss, (mu, alpha, mse, reg)), grads = grad_fn(model)
    optimizer.update(model,grads)
    
    return total_loss, mse, reg


@nnx.jit
def _eval_step(model, X, Y):
    """Single evaluation step (jitted)."""
    mu, alpha, (total_loss, mse, reg) = model(X, Y)
    epistemic_uncertainty, aleatoric_uncertainty = model.uncertainty(alpha)
    return mu, alpha, total_loss, mse, reg, epistemic_uncertainty, aleatoric_uncertainty


class RCTrainer:
    def __init__(self, model, optimizer, train_loader, val_loader, config, best_val_loss):
        self.model = model
        self.optimizer = optimizer
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.best_val_loss = best_val_loss
        self.early_stop_count = 0
        self.out_dir = config['path']['out_dir']
        
        # Checkpoint manager
        self.checkpointer = ocp.StandardCheckpointer()
        self.ckpt_dir = os.path.abspath(os.path.join(self.out_dir, 'checkpoints'))
        os.makedirs(self.ckpt_dir, exist_ok=True)
        
        self.env = make(
            "overcooked_v2", layout='demo_cook_simple', agent_view_size=2,
            random_agent_positions=True, random_reset=False,
            max_steps=400, sample_recipe_on_delivery=True,
            indicate_successful_delivery=False, negative_rewards=True
        )
        print(self.env.height, self.env.width)

    def train_step(self, X, Y):
        """Single training step."""
        return _train_step(self.model, self.optimizer, X, Y)

    def eval_step(self, X, Y):
        """Single evaluation step."""
        return _eval_step(self.model, X, Y)

    def log_and_save(self, metrics_avg, epoch, iter_num, total_iter):
        wandb.log({
            "val_loss": float(metrics_avg['val_loss_total']),
            "val_mse": float(metrics_avg['val_mse']),
            "val_reg_loss": float(metrics_avg['val_loss_reg']),
            "val_epistemic_uncertainty": float(metrics_avg['val_epistemic_uncertainty']),
            "val_aleatoric_uncertainty": float(metrics_avg['val_aleatoric_uncertainty']),
        }, step=total_iter)

        # Save checkpoint
        ckpt = {
            'model': nnx.state(self.model).to_pure_dict(),
            'optimizer': nnx.state(self.optimizer).to_pure_dict(),
            'iter': total_iter,
            'epoch': epoch,
            'iter_num': iter_num,
            'best_val_loss': self.best_val_loss,
        }
        
        if ((total_iter // self.config['eval_every']) % self.config['save_interval'] == 0):
            save_path = os.path.join(self.ckpt_dir, f"ckpt_epoch_{epoch}_iter_{total_iter}")
            self.checkpointer.save(save_path, ckpt, force=True)
        
        if metrics_avg['val_loss_total'] < self.best_val_loss:
            self.best_val_loss = metrics_avg['val_loss_total']
            best_path = os.path.join(self.ckpt_dir, "best_model")
            self.checkpointer.save(best_path, ckpt, force=True)
            wandb.run.summary['best_val_loss'] = self.best_val_loss
            self.early_stop_count = 0
        else:
            self.early_stop_count += 1

    def evaluate_and_save(self, epoch, iter_num, total_iter, partial=False):
        """Evaluate model and save checkpoint."""
        metrics_avg = {
            'val_loss_total': 0.0,
            'val_mse': 0.0,
            'val_loss_reg': 0.0,
            'val_epistemic_uncertainty': 0.0,
            'val_aleatoric_uncertainty': 0.0,
        }
        
        num_batches = 0
        last_batch = None
        
        for j, (X, Y) in enumerate(self.val_loader):
            X = jnp.array(X)
            Y = jnp.array(Y)
            
            mu, alpha, total_loss, mse, reg, epistemic_unc, aleatoric_unc = self.eval_step(X, Y)
            
            metrics_avg['val_loss_total'] += float(total_loss)
            metrics_avg['val_mse'] += float(mse)
            metrics_avg['val_loss_reg'] += float(reg)
            metrics_avg['val_epistemic_uncertainty'] += float(epistemic_unc.mean())
            metrics_avg['val_aleatoric_uncertainty'] += float(aleatoric_unc.mean())
            num_batches += 1
            
            last_batch = (X, Y, mu, alpha, epistemic_unc, aleatoric_unc)
            
            if partial and (j + 1) >= self.config['eval_step']:
                break
        
        # Average metrics
        for key in metrics_avg:
            metrics_avg[key] /= num_batches
        
        self.log_and_save(metrics_avg, epoch, iter_num, total_iter)
        
        # Visualization
        if ((total_iter // self.config['eval_every']) % self.config['save_interval'] == 0) and last_batch is not None:
            X, Y, mu, alpha, epistemic_unc, aleatoric_unc = last_batch
            rand_idx = np.random.randint(0, X.shape[0])
            y_obs = np.array(Y[rand_idx])
            mu_sample = np.array(mu[rand_idx])
            aleatoric_sample = np.array(aleatoric_unc[rand_idx].reshape(400, 5, 11, -1).mean(axis=-1))
            epistemic_sample = np.array(epistemic_unc[rand_idx].reshape(400, 5, 11, -1).mean(axis=-1))
            seq_to_seq_viz(
                self.env, y_obs, mu_sample, aleatoric_sample, epistemic_sample,
                filename=f'{self.out_dir}/recon_iter{total_iter}'
            )

    def run(self, start_epoch, resume_iter, prev_total_iter):
        total_iter = prev_total_iter
        
        for epoch in trange(start_epoch, self.config['params']['epochs']):
            for iter_num, (X, Y) in enumerate(tqdm(self.train_loader, desc=f"Epoch {epoch}")):
                # Convert to JAX arrays
                X = jnp.array(X)
                Y = jnp.array(Y)
                
                # Evaluate periodically
                if (iter_num + 1) % self.config['eval_every'] == 0 and total_iter > 0:
                    self.evaluate_and_save(epoch, iter_num, total_iter, partial=True)
                    if self.early_stop_count >= self.config['early_stop']:
                        print(f"Early stopping at epoch {epoch}, iter {iter_num}")
                        return
                
                # Training step (lr schedule is handled by optax)
                total_loss, mse, reg = self.train_step(X, Y)
                
                # Log training metrics
                if total_iter % self.config['log_interval'] == 0:
                    wandb.log({
                        "loss_total": float(total_loss),
                        "loss_reg": float(reg),
                        "loss_mse": float(mse),
                    }, step=total_iter)
                
                total_iter += 1
            
            # End of epoch evaluation
            self.evaluate_and_save(epoch, iter_num, total_iter, partial=False)


def main(args):
    # Load config
    config = OmegaConf.load('./configs/' + args.env + '.yaml').recon_module
    config = OmegaConf.to_container(config, resolve=True)
    
    # Create output directory
    os.makedirs(config['path']['out_dir'], exist_ok=True)
    
    # Initialize wandb
    wandb.init(
        **config['wandb'],
        config=config,
    )
    
    # Create datasets
    train_dataset = rc_dataset('classification', 'train', **config['params']['dataset'])
    val_dataset = rc_dataset('classification', 'val', **config['params']['dataset'])
    
    # Create dataloaders
    train_loader = create_dataloader(
        train_dataset,
        batch_size=config['params']['batch_size'],
        shuffle=True,
        num_workers=config['params'].get('num_workers', 0),
    )
    val_loader = create_dataloader(
        val_dataset,
        batch_size=config['params']['batch_size'],
        shuffle=False,
        num_workers=config['params'].get('num_workers', 0),
    )
    
    # Create model and optimizer
    rng = jax.random.PRNGKey(config.get('seed', 42))
    rngs = nnx.Rngs(rng)
    
    model, optimizer, total_iter, epoch, iter_num, best_val_loss = build_model(
        config, task='classification', is_kv=False, rngs=rngs
    )
    
    # Create trainer and run
    trainer = RCTrainer(
        model, optimizer, train_loader, val_loader, config, best_val_loss
    )
    trainer.run(epoch, iter_num, total_iter)
    
    wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Reconstruction Classification Training (JAX)')
    parser.add_argument('--env', type=str, default='ovc', help='Environment config name')
    args = parser.parse_args()
    main(args)
