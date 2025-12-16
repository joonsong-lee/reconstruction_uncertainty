import math
import os

import numpy as np
import pickle
import torch
import torch.utils.data as data


def rc_data_provider(data_dir,mode):
    datas = []
    #combs = np.array(os.listdir(os.path.join(data_dir,'combined')))
    if mode=='train':
        with open(os.path.join(data_dir,f'train.pkl'), 'rb') as f:
            combs = pickle.load(f)
    elif(mode=='val'):
        with open(os.path.join(data_dir,f'val.pkl'), 'rb') as f:
            combs = pickle.load(f)
    for com in combs :
        if com.endswith('npy') :#and (int(com.split('_')[0])%100000)>36000 :#nd not com.startswith('itere') :
            datas.append(com)
    print(len(datas))
    return datas


class rc_dataset(data.Dataset):
    def __init__(self, task, mode,data_dir):
        super(rc_dataset, self).__init__()
        self.datas = rc_data_provider(data_dir,mode)
        self.data_dir = data_dir
        self.task = task
        if task == 'regression':
            self.maxes = np.clip(np.array([[[[ 1,  1,  1,  1,  1,  1,  1,  3,  3,  3,  1,  1,  1,  1,  1,
           1,  1,  3,  3,  3,  1,  1,  1,  1,  0,  1,  1,  1,  1,  1,
           1,  3,  3,  3,  0,  0,  3,  3,  3, 19]]]]),1,None).astype(np.float16)

    def __len__(self):
        return len(self.datas)
    
    def __getitem__(self, idx):
        try:
            obs = np.memmap(os.path.join(self.data_dir,'obs',self.datas[idx]), mode='r',dtype = np.uint8, shape=(400,5,5,39))
            global_ = np.memmap(os.path.join(self.data_dir,'global',self.datas[idx]), mode='r',dtype = np.uint8, shape=(400,5,8,41))
        except:
            temp = np.random.randint(len(self.datas))
            obs = np.memmap(os.path.join(self.data_dir,'obs',self.datas[temp]), mode='r',dtype = np.uint8, shape=(400,5,5,39))
            global_ = np.memmap(os.path.join(self.data_dir,'global',self.datas[temp]), mode='r',dtype = np.uint8, shape=(400,5,8,41))
            print(self.datas[idx])
        obs = np.array(obs).astype(np.float16)
        if self.task == 'regression':
            global_ = np.array(global_).astype(np.float16)
        elif self.task == 'classification':
            global_ = np.array(global_).astype(np.int64)
        if self.task == 'regression':
            obs = obs / self.maxes[:,:,:,:38]
            global_ = global_ / self.maxes
        obs = torch.from_numpy(obs).view(400, -1)
        global_ = torch.from_numpy(global_).view(400, -1)
        
        return obs,global_
    
class OnlineBuffer:
    def __init__(self, max_len=2000):
        self.max_len = max_len
        self.ptr = 0
        self.size = 0
        
        # free-allocation
        self.storage = [None] * max_len

    def push(self, obs, g):
        if len(obs.shape)==3:
            for i in range(obs.shape[0]):
                data = (obs[i].detach().cpu(), g[i].detach().cpu())
                self._add_single(data)
        else:
            data = (obs.detach().cpu(), g.detach().cpu())
            self._add_single(data)

    def _add_single(self, data):
        # overwrite
        self.storage[self.ptr] = data
        
        # circular update pointer
        self.ptr = (self.ptr + 1) % self.max_len
        self.size = min(self.size + 1, self.max_len)

    def sample(self, batch_size,device):
        # random sampling index
        indices = np.random.choice(self.size, batch_size, replace=False)
        batch = [self.storage[idx] for idx in indices]
        batch_obs, batch_g = zip(*batch)
        return torch.stack(batch_obs).to(device), torch.stack(batch_g).to(device)

    def __len__(self):
        return self.size