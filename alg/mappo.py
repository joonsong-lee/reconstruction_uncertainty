import numpy as np
import torch
from torch.distributions.categorical import Categorical
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

class backbone(nn.Module):
    def __init__(self, latent_dim,lstm_dim,out_dim,device) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.lstm_dim = lstm_dim
        self.out_dim = out_dim
        self.lstm = nn.LSTM(latent_dim, lstm_dim, batch_first=True)
        self.linear = nn.Linear(lstm_dim, out_dim)
        self.device = device
        self.hx = None
        self.cx = None

    def reset_hidden(self, batch_size=1):
        if batch_size == 1:
            self.hx = torch.zeros((1, self.lstm_dim), device=self.device)
            self.cx = torch.zeros((1, self.lstm_dim), device=self.device)
        else:
            self.hx = torch.zeros((1,batch_size, self.lstm_dim), device=self.device)
            self.cx = torch.zeros((1,batch_size, self.lstm_dim), device=self.device)
    
    def forward(self, inputs) :
        x = inputs
        if self.hx is None or self.cx is None:
            self.reset_hidden(x.shape[0])
        latent,(self.hx, self.cx) = self.lstm(x, (self.hx, self.cx))
        latent = F.tanh(latent)
        out = self.linear(latent)
        return out,latent


class mappo_lstm(nn.Module):
    def __init__(self, latent_dim,lstm_dim,action_dim,device) -> None:
        super().__init__()
        self.actor1 = backbone(latent_dim,lstm_dim,action_dim,device)
        self.actor2 = backbone(latent_dim,lstm_dim,action_dim,device)
        self.critic = nn.Linear(lstm_dim*2,1)
        self.latent_dim = latent_dim
        self.lstm_dim = lstm_dim
        self.action_dim = action_dim
        self.device = device

    def reset_hidden(self, batch_size=1):
        self.actor1.reset_hidden(batch_size)
        self.actor2.reset_hidden(batch_size)

    def __repr__(self) -> str:
        return "actor_critic"


    def forward(self, obs1,obs2) :
        if self.actor1.hx is None or self.actor1.cx is None:
            self.reset_hidden(obs1.shape[0])
        logits_actions1,hx1 = self.actor1(obs1)
        logits_actions2,hx2 = self.actor2(obs2)
        critic_inputs = torch.cat([hx1,hx2],dim=-1)
        means_values = self.critic(critic_inputs)
        return logits_actions1, logits_actions2, means_values