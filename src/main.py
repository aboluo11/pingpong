import shutil
import numpy as np
import torch
from stable_baselines3.common.vec_env.subproc_vec_env import SubprocVecEnv
import gym
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
from pathlib import Path
import time
import math
import argparse

from model import Model

torch.backends.cudnn.benchmark = True

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
    else:
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
    return imgs.astype(np.float32)

def calculate_target_value(reward, next_value):
    res = torch.zeros(NumStep, NumProcess, device='cuda', dtype=torch.float32)
    not_done = (reward == 0).float()
    res[-1] = next_value*not_done[-1]*0.99 + reward[-1]*(1-not_done[-1])
    for step in reversed(range(NumStep-1)):
        res[step] = res[step+1]*not_done[step]*0.99 + reward[step]
    return res

def init_state(model, envs):
    model.eval()
    x = T(prepro(imgs=envs.reset()), cuda=True)
    hidden_pre = T(np.zeros([NumProcess, gru_size], dtype=np.float32), cuda=True)
    with torch.no_grad():
        p, value, hidden_suf = model(x, hidden_pre)
    return p, value, hidden_suf, hidden_pre, x

def init_storage():
    storage_sz = (NumStep, NumProcess)
    return {
        'reward':T(np.zeros(storage_sz, dtype=np.float32), cuda=True),
        'p':T(np.zeros(storage_sz, dtype=np.float32), cuda=True),
        'value':T(np.zeros(storage_sz, dtype=np.float32), cuda=True),
        'x':T(np.zeros([*storage_sz, *ImgSize], dtype=np.float32), cuda=True),
        'action':T(np.zeros(storage_sz, dtype=np.float32), cuda=True),
        'hidden': T(np.zeros([*storage_sz, gru_size], dtype=np.float32), cuda=True),
    }

def agent_play(model, storage, envs, p, value, hidden_suf, hidden_pre, x):
    model.eval()
    for step in range(NumStep):
        action = (torch.rand(NumProcess, device='cuda', dtype=torch.float32) > p) + 2
        storage['hidden'][step] = hidden_pre
        storage['x'][step] = x
        storage['action'][step] = action
        storage['p'][step] = (action==2).float()*p + (action==3).float()*(1-p)
        storage['value'][step] = value
        x, reward, done, info = envs.step(action.cpu().numpy())
        storage['reward'][step] = T(reward.astype(np.float32), cuda=True)
        hidden_suf[done] = torch.zeros(gru_size, dtype=torch.float32, device='cuda')
        x = T(prepro(x), cuda=True)
        with torch.no_grad():
            hidden_pre = hidden_suf
            p, value, hidden_suf = model(x, hidden_pre)
    return p, value, hidden_suf, hidden_pre, x

def eval():
    env = make_env(0)
    model = torch.jit.script(Model(256))
    weight_path = WeightPath/str(NumEpisode)
    model.load_state_dict(torch.load(weight_path))
    model.eval()
    x = env.reset()
    hidden = torch.zeros(1, 256, dtype=torch.float32)
    ep = 0
    reward_sum = 0
    win = 0
    lose = 0
    steps = 0
    try:
        t1 = time.time()
        while ep < 10:
            steps += 1
            x = prepro(np.expand_dims(x, 0))
            with torch.no_grad():
                p, value, hidden = model(T(x, cuda=False), hidden)
            action = 2 if p > 0.5 else 3
            x, reward, done, info = env.step(action)
            reward_sum += reward
            if reward == 1:
                win += 1
            elif reward == -1:
                lose += 1
            if done:
                ep += 1
                x = env.reset()
        t2 = time.time()
        print(f'reward:{reward_sum}, steps:{steps}')
        print(f'win:{win}')
        print(f'lose:{lose}')
        print(f'win/lose:{win/lose}')
        print(f"fps={steps/(t2-t1)}")
    finally:
        env.close()


def main():
    reward_sum = 0
    agent_time = 0
    n_updates = NumEpisode*NumEpoch*math.ceil(NumProcess*NumStep/BatchSize)
    lrs = np.linspace(lr, 0, num=n_updates, endpoint=False)
    # epsilons = np.linspace(epsilon, 0, num=n_updates, endpoint=False)
    global_step = 0
    envs = SubprocVecEnv([lambda: make_env(i) for i in range(NumProcess)])
    model = torch.jit.script(Model(gru_size).cuda())
    optimizer = torch.optim.Adam(model.parameters(), weight_decay=weight_decay)
    sampler = BatchSampler(SubsetRandomSampler(range(NumProcess*NumStep)),BatchSize,drop_last=False)

    storage = init_storage()
    p, value, hidden_suf, hidden_pre, x = init_state(model, envs)

    start = time.time()
    for episode in range(NumEpisode):
        agent_start = time.time()
        p, value, hidden_suf, hidden_pre, x = agent_play(model, storage, envs, p, value, hidden_suf, hidden_pre, x)
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

        model.train()
        for epoch in range(NumEpoch):
            for idx in sampler:
                for pg in optimizer.param_groups:
                    pg['lr'] = lrs[global_step]
                p_new, value_new, _ = model(x_flatten[idx], hidden_flatten[idx])
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

        if (episode+1) % 1000 == 0:
            torch.save(model.state_dict(), WeightPath/f'{episode+1}')
        if (episode+1) % 10 == 0:
            end = time.time()
            print(f'reward_sum:{reward_sum}, agent_time:{agent_time}, learn_time:{end-start-agent_time}')
            reward_sum = 0
            start = end
            agent_time = 0
    eval()
        
if __name__ == '__main__':
    t1 = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument('--weight_path', required=True, type=str)
    args = parser.parse_args()
    WeightPath = Path(__file__).parent.parent/f'weights/{args.weight_path}'
    if WeightPath.exists():
        shutil.rmtree(WeightPath)
    WeightPath.mkdir(exist_ok=True)
    main()
    t2 = time.time()
    print(f'total time: {t2-t1}')