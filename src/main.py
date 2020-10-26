import shutil
import numpy as np
import torch
from stable_baselines3.common.vec_env.subproc_vec_env import SubprocVecEnv
import gym
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
from pathlib import Path
import time
import math

from model import Model

WeightPath = Path(__file__).parent.parent/'weights/rnn'
NumProcess = 8
NumStep = 128
NumEpoch = 4
NumEpisode = 4000
BatchSize = 256
ImgSize = (1, 80, 80)
lr = 2.5e-4
weight_decay = 0
epsilon = 0.1
gru_size = 256

def T(x, cuda=True):
    if x.dtype in (np.int8, np.int16, np.int32, np.int64, np.bool):
        x = torch.from_numpy(x.astype(np.int64))
    elif x.dtype in (np.float32, np.float64):
        x = torch.from_numpy(x.astype(np.float32))
    if cuda:
        x = x.pin_memory().cuda(non_blocking=True)
    return x

def make_env(seed):
    env = gym.make("Pong-v0")
    env.seed(seed)
    return env

def prepro(imgs):
    if len(imgs.shape) == 3:
        imgs = np.expand_dims(imgs, 0)
    imgs = imgs[:, 35:195]
    imgs = imgs[:, ::2, ::2, 0]
    imgs = np.expand_dims(imgs, 1)
    return imgs.astype(np.float)

def calculate_target_value(reward, next_value):
    res = torch.zeros(NumStep, NumProcess, device='cuda')
    not_done = (reward == 0).float()
    res[-1] = next_value*not_done[-1]*0.99 + reward[-1]*(1-not_done[-1])
    for step in reversed(range(NumStep-1)):
        res[step] = res[step+1]*not_done[step]*0.99 + reward[step]
    return res

def main():
    reward_sum = 0
    agent_time = 0
    n_updates = NumEpisode*NumEpoch*math.ceil(NumProcess*NumStep/BatchSize)
    lrs = np.linspace(lr, 0, num=n_updates, endpoint=False)
    epsilons = np.linspace(epsilon, 0, num=n_updates, endpoint=False)
    global_step = 0
    envs = SubprocVecEnv([lambda: make_env(i) for i in range(NumProcess)])
    model_gpu = torch.jit.script(Model(gru_size).cuda())
    optimizer = torch.optim.Adam(model_gpu.parameters(), weight_decay=weight_decay)
    sampler = BatchSampler(SubsetRandomSampler(range(NumProcess*NumStep)),BatchSize,drop_last=False)
    storage_sz = (NumStep, NumProcess)
    storage = {
        'reward':T(np.zeros(storage_sz), cuda=True),
        'p':T(np.zeros(storage_sz), cuda=True),
        'value':T(np.zeros(storage_sz), cuda=True),
        'x':T(np.zeros([*storage_sz, *ImgSize]), cuda=True),
        'action':T(np.zeros(storage_sz), cuda=True),
        'hidden': T(np.zeros([*storage_sz, gru_size]), cuda=True),
    }

    # model_gpu.eval()
    x = T(prepro(imgs=envs.reset()), cuda=True)
    hidden_pre = T(np.zeros([NumProcess, gru_size]), cuda=True)
    with torch.no_grad():
        p, value, hidden_suf = model_gpu(x, hidden_pre)

    start = time.time()

    for episode in range(NumEpisode):
        agent_start = time.time()

        # model_gpu.eval()
        for step in range(NumStep):
            action = (torch.rand(NumProcess, device='cuda') > p) + 2
            storage['hidden'][step] = hidden_pre
            storage['x'][step] = x
            storage['action'][step] = action
            storage['p'][step] = (action==2).float()*p + (action==3).float()*(1-p)
            storage['value'][step] = value
            x, reward, done, info = envs.step(action.cpu().numpy())
            storage['reward'][step] = T(reward, cuda=True)
            hidden_suf[done] = T(np.zeros(gru_size), cuda=True)
            x = T(prepro(x), cuda=True)
            with torch.no_grad():
                hidden_pre = hidden_suf
                p, value, hidden_suf = model_gpu(x, hidden_pre)

        agent_end = time.time()
        agent_time += (agent_end - agent_start)

        target_value = calculate_target_value(storage['reward'], value)
        advantage = target_value - storage['value']
        target_value_flatten = target_value.view(-1)
        advantage_flatten = advantage.view(-1)
        hidden_flatten = storage['hidden'].view(-1, gru_size)
        x_flatten = storage['x'].view(-1, *ImgSize)
        action_flatten = storage['action'].view(-1)
        p_flatten = storage['p'].view(-1)

        # model_gpu.train()
        for epoch in range(NumEpoch):
            for idx in sampler:
                for pg in optimizer.param_groups:
                    pg['lr'] = lrs[global_step]
                p_new, value_new, _ = model_gpu(x_flatten[idx], hidden_flatten[idx])
                p_new = (action_flatten[idx]==2).float()*p_new + (action_flatten[idx]==3).float()*(1-p_new)
                ratio = p_new / p_flatten[idx]
                policy_loss = -torch.sum(torch.min(
                    ratio * advantage_flatten[idx],
                    torch.clamp(ratio, 1-epsilon, 1+epsilon) * advantage_flatten[idx]
                ))
                value_loss = torch.sum((target_value_flatten[idx] - value_new)**2)
                loss = policy_loss + value_loss
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                global_step += 1

        reward_sum += storage['reward'].sum().item()

        if (episode+1) % 100 == 0:
            torch.save(model_gpu.state_dict(), WeightPath/f'{episode+1}')
        if (episode+1) % 10 == 0:
            end = time.time()
            print(f'reward_sum:{reward_sum}, agent_time:{agent_time}, learn_time:{end-start-agent_time}')
            reward_sum = 0
            start = end
            agent_time = 0
        
if __name__ == '__main__':
    shutil.rmtree(WeightPath)
    WeightPath.mkdir(exist_ok=True)
    main()