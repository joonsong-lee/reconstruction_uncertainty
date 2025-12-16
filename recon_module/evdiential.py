import os

import torch
from torch import nn
import torch.nn.functional as F
from recon_module.transformer import GPT, GPTConfig
from util import build_optimizer
#from https://github.com/teddykoker/evidential-learning-pytorch/tree/main


def load_model(ckpt_path,device,task,is_kv = False, **model_config):
    if task == 'regression':
        model = Evidential_reconstruction(is_kv, **model_config)
    elif task == 'classification':
        model = Evidential_reconstruction_classification(is_kv, **model_config)
    elif task == 'no_evidential_classification':
        model = Noevidential_classification(is_kv, **model_config)
    else:
        raise ValueError("Unknown task type")

    assert os.path.exists(ckpt_path),"ckpt path is wrong"
    checkpoint = torch.load(ckpt_path, map_location=device)
    state_dict = checkpoint['model']
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)

    if device is not None:
        model.to(device)

    return model, checkpoint

def build_model(config,device,dtype,device_type,task,is_kv = False):
    scaler = torch.amp.GradScaler(device,enabled=(dtype == "float16")) #torch.cuda.amp.gradscaler is deprecated. instead of it, using this code is recommended
    # optimizer
    opt_config = config['params']['optimizer']
    
    if config['path']['ckpt_path'] is not None:
        print("Loading model from checkpoint:", config['path']['ckpt_path'])
        model, checkpoint = load_model(config['path']['ckpt_path'], device,task,is_kv,**config['params']['model'])
        optimizer = build_optimizer(model,**opt_config)
        opt_dict = checkpoint['optimizer']
        optimizer.load_state_dict(opt_dict)
        for param_group in optimizer.param_groups:
            param_group['lr'] = opt_config['lr']
    else:
        if task == 'regression':
            model = Evidential_reconstruction(is_kv, **config['params']['model'])
        elif task == 'classification':
            model = Evidential_reconstruction_classification(is_kv, **config['params']['model'])
        elif task == 'no_evidential_classification':
            model = Noevidential_classification(is_kv, **config['params']['model'])
        else:
            raise ValueError("Unknown task type")
        optimizer = build_optimizer(model,**opt_config)
        checkpoint = {}
    total_iter = checkpoint.get('total_iter',0)
    epoch = checkpoint.get('epoch',0)
    iter = checkpoint.get('iter',0)
    best_val_loss = checkpoint.get('best_val_loss', float('1e9'))
    # initialize a GradScaler. If enabled=False scaler is a no-op
    checkpoint = None  # free up memory
    return model, optimizer, scaler, total_iter,epoch,iter,best_val_loss

class NormalInvGamma(nn.Module):
    def __init__(self, in_features, out_units):
        super().__init__()
        self.dense = nn.Linear(in_features, out_units * 4)
        self.out_units = out_units

    def evidence(self, x):
        return F.softplus(x)

    def forward(self, x):
        out = self.dense(x)
        mu, logv, logalpha, logbeta = torch.split(out, self.out_units, dim=-1)
        v = self.evidence(logv)
        alpha = self.evidence(logalpha) + 1
        beta = self.evidence(logbeta)
        return mu, v, alpha, beta
    
def nig_nll(gamma, v, alpha, beta, y):
    two_beta_lambda = 2 * beta * (1 + v)
    t1 = 0.5 * (torch.pi / v).log()
    t2 = alpha * two_beta_lambda.log()
    t3 = (alpha + 0.5) * (v * (y - gamma) ** 2 + two_beta_lambda).log()
    t4 = alpha.lgamma()
    t5 = (alpha + 0.5).lgamma()
    nll = t1 - t2 + t3 + t4 - t5
    return nll.mean()

class Dirichlet(nn.Module):
    def __init__(self, in_features, out_units):
        super().__init__()
        self.dense = nn.Linear(in_features, out_units)
        self.out_units = out_units

    def evidence(self, x):
        return F.softplus(x)

    def forward(self, x):
        out = self.dense(x)
        alpha = self.evidence(out) + 1
        return alpha

# Eq. (5) from https://arxiv.org/abs/1806.01768:
# Sum of squares loss

def dirichlet_reg(alpha, y):
    # dirichlet parameters after removal of non-misleading evidence (from the label)
    alpha = y + (1 - y) * alpha

    # uniform dirichlet distribution
    beta = torch.ones_like(alpha)

    sum_alpha = alpha.sum(-1)
    sum_beta = beta.sum(-1)

    t1 = sum_alpha.lgamma() - sum_beta.lgamma()
    t2 = (alpha.lgamma() - beta.lgamma()).sum(-1)
    t3 = alpha - beta
    t4 = alpha.digamma() - sum_alpha.digamma().unsqueeze(-1)

    kl = t1 - t2 + (t3 * t4).sum(-1)
    return kl.mean()


def dirichlet_mse(alpha, y):
    sum_alpha = alpha.sum(-1, keepdims=True)
    p = alpha / sum_alpha
    t1 = (y - p).pow(2).sum(-1)
    t2 = ((p * (1 - p)) / (sum_alpha + 1)).sum(-1)
    mse = t1 + t2
    return mse.mean()


def evidential_classification(alpha, y, lamb=0.1):
    num_classes = alpha.shape[-1]
    y = F.one_hot(y, num_classes)
    mse = dirichlet_mse(alpha, y)
    reg = dirichlet_reg(alpha, y)
    total = mse + lamb * reg
    return total, mse, reg


def evidential_regression(dist_params, y, lamb=1.0):
    return nig_nll(*dist_params, y) + lamb * nig_reg(*dist_params, y)

# Normal Inverse Gamma regularization
# from https://arxiv.org/abs/1910.02600:
# > we formulate a novel evidence regularizer, L^R_i
# > scaled on the error of the i-th prediction
def nig_reg(gamma, v, alpha, _beta, y):
    reg = (y - gamma).abs() * (2 * v + alpha)
    return reg.mean()

def evidential_regression(dist_params, y, lamb=1.0):
    nll= nig_nll(*dist_params, y)
    reg = nig_reg(*dist_params, y)
    mse = (y - dist_params[0]).pow(2).mean()
    total = nll + lamb * reg + 10*mse  # adding mse loss for stability
    return total, nll, reg, mse

class Evidential_reconstruction(nn.Module):
    def __init__(self, is_kv,in_features, out_units,gpt_config,**kwargs):
        super().__init__()
        self.nig = NormalInvGamma(in_features, out_units)
        data_gpt_config = GPTConfig(**gpt_config)
        self.gpt = GPT(data_gpt_config,is_kv)

    def uncertainty(self,mu, v, alpha, beta):
        epistemic_uncertainty = beta / (v * (alpha - 1))
        aleatoric_uncertainty = beta / (alpha - 1)
        return epistemic_uncertainty, aleatoric_uncertainty

    def forward(self, x,y=None):
        x = self.gpt(x)
        mu, v, alpha, beta = self.nig(x)
        if y is not None:
            total_loss,nll_loss,reg,mse_loss = evidential_regression((mu, v, alpha, beta), y)
            return (mu, v, alpha, beta), (total_loss, nll_loss, reg, mse_loss)
        else:
            epistemic_uncertainty, aleatoric_uncertainty = self.uncertainty(mu, v, alpha, beta)
            return mu, epistemic_uncertainty, aleatoric_uncertainty

class Evidential_reconstruction_classification(nn.Module):
    def __init__(self, is_kv,in_features, out_units,num_classes,gpt_config):
        super().__init__()
        self.dri = Dirichlet(in_features, out_units* num_classes)
        self.num_classes = num_classes
        self.out_units = out_units
        data_gpt_config = GPTConfig(**gpt_config)
        self.gpt = GPT(data_gpt_config,is_kv)

    def uncertainty(self,alpha):
        S = alpha.sum(dim=-1, keepdim=True)
        epistemic_uncertainty = self.num_classes / S
        p = alpha / S
        aleatoric_uncertainty = (p*(1-p)/(1+S)).sum(dim=-1, keepdim=True)
        return epistemic_uncertainty, aleatoric_uncertainty

    def forward(self, x,y=None):
        x = self.gpt(x)
        raw_alpha = self.dri(x)
        alpha = raw_alpha.view(-1,self.num_classes)
        with torch.no_grad():
            alpha_reshaped = raw_alpha.view(*x.shape[:-1],self.out_units,self.num_classes)
            S = alpha_reshaped.sum(dim=-1, keepdim=True)
            mu = alpha_reshaped / S
            mu = torch.argmax(mu,dim=-1)
        if y is not None:
            total_loss,mse_loss, reg_loss = evidential_classification(alpha, y.flatten())
            return mu,alpha_reshaped, (total_loss, mse_loss, reg_loss)
        else:
            epistemic_uncertainty, aleatoric_uncertainty = self.uncertainty(alpha_reshaped)
            return mu, epistemic_uncertainty, aleatoric_uncertainty

class Noevidential_classification(nn.Module):
    def __init__(self, is_kv,in_features, out_units,num_classes,gpt_config):
        super().__init__()
        self.out_proj = nn.Linear(in_features, out_units* num_classes)
        self.num_classes = num_classes
        self.out_units = out_units
        data_gpt_config = GPTConfig(**gpt_config)
        self.gpt = GPT(data_gpt_config,is_kv)

    def forward(self, x,y=None):
        x = self.gpt(x)
        logit = self.out_proj(x)
        mu = logit.view(*x.shape[:-1],self.out_units,self.num_classes)
        mu = torch.argmax(mu,dim=-1)
        loss = F.cross_entropy(logit.view(-1,self.num_classes),y.view(-1)) if y is not None else None
        return mu,loss

