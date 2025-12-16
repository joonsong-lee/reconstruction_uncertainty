import numpy as np
import torch
from torch.distributions.categorical import Categorical
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

class VmapLSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        # nn.LSTMCell과 동일한 파라미터 이름 사용
        self.weight_ih = nn.Parameter(torch.randn(4 * hidden_size, input_size))
        self.weight_hh = nn.Parameter(torch.randn(4 * hidden_size, hidden_size))
        self.bias_ih = nn.Parameter(torch.randn(4 * hidden_size))
        self.bias_hh = nn.Parameter(torch.randn(4 * hidden_size))

    def forward(self, input, state):
        # input: (input_size) or (batch, input_size)
        # state: ((hidden_size), (hidden_size)) or batch version
        hx, cx = state
        
        # vmap 내부에서는 배치가 까져서 1D Tensor로 들어올 수 있음
        # F.linear는 1D 입력도 잘 처리함 (Out_dim,) 형태로 출력
        gates = (F.linear(input, self.weight_ih, self.bias_ih) +
                 F.linear(hx, self.weight_hh, self.bias_hh))
        
        ingate, forgetgate, cellgate, outgate = gates.chunk(4, dim=-1) # 마지막 차원 기준 분할

        ingate = torch.sigmoid(ingate)
        forgetgate = torch.sigmoid(forgetgate)
        cellgate = torch.tanh(cellgate)
        outgate = torch.sigmoid(outgate)

        cy = (forgetgate * cx) + (ingate * cellgate)
        hy = outgate * torch.tanh(cy)

        return hy, cy
    

class cell_backbone(nn.Module):
    def __init__(self, latent_dim,lstm_dim,action_dim,device,**kwargs) -> None:
        super().__init__()
        # self.conv1 = nn.Conv2d(38, 32, kernel_size=3, stride=1, padding=1) 
        # self.conv2 = nn.Conv2d(32,16, kernel_size=3, stride=1, padding=1)
        self.conv1 = nn.Conv2d(39, 32, kernel_size=1, stride=1) 
        self.conv2 = nn.Conv2d(32,16, kernel_size=1, stride=1)
        self.conv3 = nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1) 
        self.latent_dim = latent_dim
        self.lstm_dim = lstm_dim
        self.out_dim = action_dim
        self.lstm = VmapLSTMCell(latent_dim, lstm_dim)
        self.linear = nn.Linear(lstm_dim, action_dim)
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
        x = inputs if len(inputs.shape)<=4 else inputs.view(-1,inputs.shape[2],inputs.shape[3],inputs.shape[4])
        x = F.tanh(self.conv1(x))
        x = F.tanh(self.conv2(x))
        x = F.tanh(self.conv3(x))
        if len(inputs.shape)>4:
            x = x.contiguous().view(inputs.shape[0],inputs.shape[1],-1)
        elif len(inputs.shape)==4:
            x = x.contiguous().view(inputs.shape[0],-1)
        else:
            x = x.reshape(-1)
        # if self.hx is None or self.cx is None:
        #     self.reset_hidden(inputs.shape[0])
        hx_, cx_ = self.lstm(x, (hx, cx))
        latent = F.tanh(hx_)
        out = self.linear(latent)
        return out,hx_,cx_


class backbone(nn.Module):
    def __init__(self, latent_dim,lstm_dim,action_dim,device,**kwargs) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(39, 32, kernel_size=1, stride=1) 
        self.conv2 = nn.Conv2d(32,16, kernel_size=1, stride=1)
        self.conv3 = nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1) 
        self.latent_dim = latent_dim
        self.lstm_dim = lstm_dim
        self.out_dim = action_dim
        self.lstm = nn.LSTM(latent_dim, lstm_dim, batch_first=True)
        self.linear = nn.Linear(lstm_dim, action_dim)
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
        x = inputs if len(inputs.shape)<=4 else inputs.view(-1,inputs.shape[2],inputs.shape[3],inputs.shape[4])
        x = F.tanh(self.conv1(x))
        x = F.tanh(self.conv2(x))
        x = F.tanh(self.conv3(x))
        x = x.contiguous().view(inputs.shape[0],inputs.shape[1],-1) if len(inputs.shape)>3 else x.view(1,-1)
        # if self.hx is None or self.cx is None:
        #     self.reset_hidden(inputs.shape[0])
        latent,(hx_, cx_) = self.lstm(x, (hx, cx))
        latent = F.tanh(latent)
        out = self.linear(latent)
        return out,(hx_,cx_),latent


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

    # def reset_hidden(self, batch_size=1):
    #     self.actor1.reset_hidden(batch_size)
    #     self.actor2.reset_hidden(batch_size)

    def forward(self, obs1,obs2,hx1,cx1,hx2,cx2):
        # if self.actor1.hx is None or self.actor1.cx is None:
        #     self.reset_hidden(obs1.shape[0])
        logits_actions1,(hx1,cx1),latent1 = self.actor1(obs1,hx1,cx1)
        logits_actions2,(hx2,cx2),latent2 = self.actor2(obs2,hx2,cx2)
        critic_inputs = torch.cat([latent1,latent2],dim=-1)
        means_values = self.critic(critic_inputs)
        return logits_actions1, logits_actions2, means_values, (hx1, cx1), (hx2, cx2)
    


class backbone_stateful(nn.Module):
    def __init__(self, latent_dim,lstm_dim,action_dim,device,**kwargs) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(39, 32, kernel_size=1, stride=1) 
        self.conv2 = nn.Conv2d(32,16, kernel_size=1, stride=1)
        self.conv3 = nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1) 
        self.latent_dim = latent_dim
        self.lstm_dim = lstm_dim
        self.out_dim = action_dim
        self.lstm = nn.LSTM(latent_dim, lstm_dim, batch_first=True)
        self.linear = nn.Linear(lstm_dim, action_dim)
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
        x = inputs if len(inputs.shape)<=4 else inputs.reshape(-1,inputs.shape[2],inputs.shape[3],inputs.shape[4])
        x = F.tanh(self.conv1(x))
        x = F.tanh(self.conv2(x))
        x = F.tanh(self.conv3(x))
        x = x.contiguous().view(inputs.shape[0],inputs.shape[1],-1) if len(inputs.shape)>4 else x.reshape(inputs.shape[0],1,-1)
        if self.hx is None or self.cx is None:
            self.reset_hidden(inputs.shape[0])
        latent,(self.hx, self.cx) = self.lstm(x, (self.hx, self.cx))
        latent = F.tanh(latent)
        out = self.linear(latent)
        return out,latent


class mappo_lstm_stateful(nn.Module):
    def __init__(self, latent_dim,lstm_dim,action_dim,device) -> None:
        super().__init__()
        self.actor1 = backbone_stateful(latent_dim,lstm_dim,action_dim,device)
        self.actor2 = backbone_stateful(latent_dim,lstm_dim,action_dim,device)
        self.critic = nn.Linear(lstm_dim*2,1)
        self.latent_dim = latent_dim
        self.lstm_dim = lstm_dim
        self.action_dim = action_dim
        self.device = device

    def reset_hidden(self, batch_size=1):
        self.actor1.reset_hidden(batch_size)
        self.actor2.reset_hidden(batch_size)

    def forward(self, obs1,obs2):
        if self.actor1.hx is None or self.actor1.cx is None:
            self.reset_hidden(obs1.shape[0])
        logits_actions1,latent1 = self.actor1(obs1)
        logits_actions2,latent2 = self.actor2(obs2)
        critic_inputs = torch.cat([latent1,latent2],dim=-1)
        means_values = self.critic(critic_inputs)
        return logits_actions1, logits_actions2, means_values