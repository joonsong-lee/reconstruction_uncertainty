import numpy as np
import torch
from torch.distributions.categorical import Categorical
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

class backbone(nn.Module):
    def __init__(self, latent_dim,lstm_dim,out_dim,device) -> None:
        super().__init__()
        # self.conv1 = nn.Conv2d(39, 32, kernel_size=3, stride=1, padding=1) 
        # self.conv2 = nn.Conv2d(32,16, kernel_size=3, stride=1, padding=1)
        self.conv1 = nn.Conv2d(39, 32, kernel_size=1, stride=1) 
        self.conv2 = nn.Conv2d(32,16, kernel_size=1, stride=1)
        self.conv3 = nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1) 
        self.latent_dim = latent_dim
        self.lstm_dim = lstm_dim
        self.out_dim = out_dim
        self.lstm = nn.LSTM(latent_dim, lstm_dim, batch_first=True)
        self.linear = nn.Linear(lstm_dim, out_dim)
        self.device = device
    #     self.hx = None
    #     self.cx = None

    # def reset_hidden(self, batch_size=1):
    #     if batch_size == 1:
    #         self.hx = torch.zeros((1, self.lstm_dim), device=self.device)
    #         self.cx = torch.zeros((1, self.lstm_dim), device=self.device)
    #     else:
    #         self.hx = torch.zeros((1,batch_size, self.lstm_dim), device=self.device)
    #         self.cx = torch.zeros((1,batch_size, self.lstm_dim), device=self.device)
    
    def forward(self, inputs,hx,cx) :
        x = inputs if len(inputs.shape)<=4 else inputs.reshape(-1,inputs.shape[2],inputs.shape[3],inputs.shape[4])
        x = F.tanh(self.conv1(x))
        x = F.tanh(self.conv2(x))
        x = F.tanh(self.conv3(x))
        x = x.contiguous().view(inputs.shape[0],inputs.shape[1],-1) if len(inputs.shape)>4 else x.contiguous().view(inputs.shape[0],1,-1)
        # if self.hx is None or self.cx is None:
        #     self.reset_hidden(inputs.shape[0])
        latent,(hx_, cx_) = self.lstm(x, (hx, cx))
        latent = F.tanh(latent)
        out = self.linear(latent)
        return out,(hx_,cx_),latent


class ppo(nn.Module):
    def __init__(self, latent_dim,lstm_dim,action_dim,device) -> None:
        super().__init__()
        self.actor = backbone(latent_dim,lstm_dim,action_dim,device)
        self.critic = nn.Linear(lstm_dim,1)
        self.latent_dim = latent_dim
        self.lstm_dim = lstm_dim
        self.action_dim = action_dim
        self.device = device

    # def reset_hidden(self, batch_size=1):
    #     self.actor1.reset_hidden(batch_size)
    #     self.actor2.reset_hidden(batch_size)

    def forward(self, obs1,hx1,cx1):
        # if self.actor1.hx is None or self.actor1.cx is None:
        #     self.reset_hidden(obs1.shape[0])
        logits_actions,(hx,cx),latent = self.actor(obs1,hx1,cx1)
        means_values = self.critic(latent)
        return logits_actions, means_values, (hx, cx)
    

class backbone_large(nn.Module):
    def __init__(self, latent_dim,lstm_dim,out_dim,added_dim,device) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(41+added_dim, 32*((added_dim//40)+1), kernel_size=1, stride=1) 
        self.conv2 = nn.Conv2d(32*((added_dim//40)+1),16*((added_dim//40)+1), kernel_size=1, stride=1)
        self.conv3 = nn.Conv2d(16*((added_dim//40)+1),16, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(16,8, kernel_size=3, stride=1, padding=1)
        self.latent_dim = latent_dim
        self.lstm_dim = lstm_dim
        self.out_dim = out_dim
        self.lstm = nn.LSTM(latent_dim, lstm_dim, batch_first=True)
        self.linear = nn.Linear(lstm_dim, out_dim)
        self.device = device
    #     self.hx = None
    #     self.cx = None

    # def reset_hidden(self, batch_size=1):
    #     if batch_size == 1:
    #         self.hx = torch.zeros((1, self.lstm_dim), device=self.device)
    #         self.cx = torch.zeros((1, self.lstm_dim), device=self.device)
    #     else:
    #         self.hx = torch.zeros((1,batch_size, self.lstm_dim), device=self.device)
    #         self.cx = torch.zeros((1,batch_size, self.lstm_dim), device=self.device)
    
    def forward(self, inputs,hx,cx) :
        x = inputs if len(inputs.shape)<=4 else inputs.reshape(-1,inputs.shape[2],inputs.shape[3],inputs.shape[4])
        x = F.tanh(self.conv1(x))
        x = F.tanh(self.conv2(x))
        x = F.tanh(self.conv3(x))
        x = F.tanh(self.conv4(x))
        x = x.contiguous().view(inputs.shape[0],inputs.shape[1],-1) if len(inputs.shape)>4 else x.contiguous().view(inputs.shape[0],1,-1)
        # if self.hx is None or self.cx is None:
        #     self.reset_hidden(inputs.shape[0])
        latent,(hx_, cx_) = self.lstm(x, (hx, cx))
        latent = F.tanh(latent)
        out = self.linear(latent)
        return out,(hx_,cx_),latent


class ppo_large(nn.Module):
    def __init__(self, latent_dim,lstm_dim,action_dim,added_dim,device) -> None:
        super().__init__()
        self.actor = backbone_large(latent_dim,lstm_dim,action_dim,added_dim,device)
        self.critic = nn.Linear(lstm_dim,1)
        self.latent_dim = latent_dim
        self.lstm_dim = lstm_dim
        self.action_dim = action_dim
        self.device = device

    # def reset_hidden(self, batch_size=1):
    #     self.actor1.reset_hidden(batch_size)
    #     self.actor2.reset_hidden(batch_size)

    def forward(self, obs1,hx1,cx1):
        # if self.actor1.hx is None or self.actor1.cx is None:
        #     self.reset_hidden(obs1.shape[0])
        logits_actions,(hx,cx),latent = self.actor(obs1,hx1,cx1)
        means_values = self.critic(latent)
        return logits_actions, means_values, (hx, cx)


class backbone_linear(nn.Module):
    def __init__(self, latent_dim,lstm_dim,out_dim,added_dim,device) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(41+added_dim, 32*((added_dim//40)+1), kernel_size=1, stride=1) 
        self.conv2 = nn.Conv2d(32*((added_dim//40)+1),16*((added_dim//40)+1), kernel_size=1, stride=1)
        self.conv3 = nn.Conv2d(16*((added_dim//40)+1),16, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(16,8, kernel_size=3, stride=1, padding=1)
        self.latent_dim = latent_dim
        self.out_dim = out_dim
        self.linear = nn.Linear(latent_dim, out_dim)
        self.device = device

    
    def forward(self, inputs) :
        x = inputs if len(inputs.shape)<=4 else inputs.reshape(-1,inputs.shape[2],inputs.shape[3],inputs.shape[4])
        x = F.tanh(self.conv1(x))
        x = F.tanh(self.conv2(x))
        x = F.tanh(self.conv3(x))
        x = F.tanh(self.conv4(x))
        x = x.contiguous().view(inputs.shape[0],inputs.shape[1],-1) if len(inputs.shape)>4 else x.contiguous().view(inputs.shape[0],1,-1)
        out = self.linear(x)
        return out,x


class ppo_linear(nn.Module):
    def __init__(self, latent_dim,lstm_dim,action_dim,added_dim,device) -> None:
        super().__init__()
        self.actor = backbone_linear(latent_dim,lstm_dim,action_dim,added_dim,device)
        self.critic = nn.Linear(latent_dim,1)
        self.latent_dim = latent_dim
        self.action_dim = action_dim
        self.device = device

    def forward(self, obs1):
        logits_actions,latent = self.actor(obs1)
        means_values = self.critic(latent)
        return logits_actions, means_values


class backbone_cond(nn.Module):
    def __init__(self, latent_dim,lstm_dim,out_dim,device) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(38, 32, kernel_size=3, stride=1, padding=1) 
        self.conv2 = nn.Conv2d(32,16, kernel_size=3, stride=1, padding=1)
        self.latent_dim = latent_dim+1
        self.lstm_dim = lstm_dim
        self.out_dim = out_dim
        self.lstm = nn.LSTM(latent_dim+1, lstm_dim, batch_first=True)
        self.linear = nn.Linear(lstm_dim, out_dim)
        self.device = device
    #     self.hx = None
    #     self.cx = None

    # def reset_hidden(self, batch_size=1):
    #     if batch_size == 1:
    #         self.hx = torch.zeros((1, self.lstm_dim), device=self.device)
    #         self.cx = torch.zeros((1, self.lstm_dim), device=self.device)
    #     else:
    #         self.hx = torch.zeros((1,batch_size, self.lstm_dim), device=self.device)
    #         self.cx = torch.zeros((1,batch_size, self.lstm_dim), device=self.device)
    
    def forward(self, inputs,c,hx,cx) :
        x = inputs if len(inputs.shape)<=4 else inputs.reshape(-1,inputs.shape[2],inputs.shape[3],inputs.shape[4])
        x = F.tanh(self.conv1(x))
        x = F.tanh(self.conv2(x))
        x = x.contiguous().view(inputs.shape[0],inputs.shape[1],-1) if len(inputs.shape)>4 else x.contiguous().view(inputs.shape[0],1,-1)
        # if self.hx is None or self.cx is None:
        #     self.reset_hidden(inputs.shape[0])
        c = c.unsqueeze(1) if len(c.shape)<3 else c
        x = torch.cat([x,c],dim=-1)
        print(x.shape,c.shape)
        latent,(hx_, cx_) = self.lstm(x, (hx, cx))
        latent = F.tanh(latent)
        out = self.linear(latent)
        return out,(hx_,cx_),latent




class ppo_cond(nn.Module):
    def __init__(self, latent_dim,lstm_dim,action_dim,device,**kwargs) -> None:
        super().__init__()
        self.actor = backbone_cond(latent_dim,lstm_dim,action_dim,device)
        self.critic = nn.Linear(lstm_dim,1)
        self.latent_dim = latent_dim
        self.lstm_dim = lstm_dim
        self.action_dim = action_dim
        self.device = device

    # def reset_hidden(self, batch_size=1):
    #     self.actor1.reset_hidden(batch_size)
    #     self.actor2.reset_hidden(batch_size)

    def forward(self, obs1,c,hx1,cx1):
        # if self.actor1.hx is None or self.actor1.cx is None:
        #     self.reset_hidden(obs1.shape[0])
        logits_actions,(hx,cx),latent = self.actor(obs1,c,hx1,cx1)
        means_values = self.critic(latent)
        return logits_actions, means_values, (hx, cx)