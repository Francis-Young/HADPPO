# -*- coding: utf-8 -*-
"""
Created on Tue Apr 25 10:03:49 2023

@author: yf
"""


import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class ActorSoftmax(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=32, kernel_size=2, stride=2):
        super(ActorSoftmax, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1,out_channels = 1, kernel_size = 3, padding = 1)
        self.conv2 = nn.Conv1d(in_channels=1,out_channels = 1, kernel_size = 7, padding = 1)
        self.pool = nn.MaxPool1d(kernel_size, stride)
        self.fc1 = nn.Linear(input_dim//stride - kernel_size, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
    def forward(self,x):
        x = x.unsqueeze(0)
        x = self.conv1(x)
        x = self.conv2(x) 
        x = self.pool(x)
        x = torch.flatten(x, start_dim=1) 
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
class Critic(nn.Module):
    def __init__(self,input_dim,output_dim,hidden_dim=32):
        super(Critic,self).__init__()
        assert output_dim == 1 # critic must output a single value
        self.conv1 = nn.Conv1d(in_channels=1,out_channels = 1, kernel_size = 3, padding = 1)
        self.conv2 = nn.Conv1d(in_channels=1,out_channels = 1, kernel_size = 7, padding = 1)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(input_dim//2 - 2, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
    def forward(self,x):
        x = x.unsqueeze(0)
        x = self.conv1(x)
        x = self.conv2(x) 
        x = self.pool(x)
        x = torch.flatten(x, start_dim=1) 
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        value = self.fc3(x)
        return value
    
"""定义经验回放"""

import random
from collections import deque
class ReplayBufferQue:
    '''DQN的经验回放池，每次采样batch_size个样本'''
    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self.buffer = deque(maxlen=self.capacity)
    def push(self,transitions):
        self.buffer.append(transitions)
    def sample(self, batch_size: int, sequential: bool = False):
        if batch_size > len(self.buffer):
            batch_size = len(self.buffer)
        if sequential: # sequential sampling
            rand = random.randint(0, len(self.buffer) - batch_size)
            batch = [self.buffer[i] for i in range(rand, rand + batch_size)]
            return zip(*batch)
        else:
            batch = random.sample(self.buffer, batch_size)
            return zip(*batch)
    def clear(self):
        self.buffer.clear()
    def __len__(self):
        return len(self.buffer)

class PGReplay(ReplayBufferQue):
    '''继承ReplayBufferQue，重写sample方法
    '''
    def __init__(self):
        self.buffer = deque()
    def sample(self):
        ''' sample all the transitions
        '''
        batch = list(self.buffer)
        return zip(*batch)

"""定义agent"""
import torch
from torch.distributions import Categorical
class Agent:
    def __init__(self,cfg) -> None:
        self.nvec = [7]*cfg.D*cfg.U
        self.gamma = cfg.gamma
        self.device = torch.device(cfg.device) 
        self.actor = ActorSoftmax(cfg.n_states, sum(self.nvec), hidden_dim = cfg.actor_hidden_dim).to(self.device)
        self.critic = Critic(cfg.n_states,1,hidden_dim=cfg.critic_hidden_dim).to(self.device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=cfg.actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=cfg.critic_lr)
        self.memory = PGReplay()
        self.k_epochs = cfg.k_epochs # update policy for K epochs
        self.eps_clip = cfg.eps_clip # clip parameter for PPO
        self.entropy_coef = cfg.entropy_coef # entropy coefficient
        self.sample_count = 0
        self.update_freq = cfg.update_freq
        self.mask = torch.tensor((([0]*2)+([-999]*(cfg.num_actions - 2)))*cfg.D*cfg.U) # mask out the invaild actions

    def sample_action(self,state):
        self.sample_count += 1
        state = torch.tensor(state, device=self.device, dtype=torch.float32).unsqueeze(dim=0)
        logits = self.actor(state) # get the logits from the actor network
        logits += self.mask # decease the logit of invalid actions in each dimension
        split_logits = torch.split(logits, self.nvec, dim=1) # split the logits into chunks according to the action space size
        multi_categoricals = [Categorical(logits=logits) for logits in split_logits] # create a list of categorical distributions for each chunk
        action = torch.stack([categorical.sample() for categorical in multi_categoricals]) # sample an action from each categorical distribution 
        logprob = torch.stack([categorical.log_prob(a) for a, categorical in zip(action, multi_categoricals)]) # calculate the log probability of each action      
        self.log_probs = logprob.detach()
        return action.detach().cpu().numpy()
    @torch.no_grad()
    def predict_action(self,state):
        state = torch.tensor(state, device=self.device, dtype=torch.float32).unsqueeze(dim=0)
        logits = self.actor(state)
        logits += self.mask
        split_logits = torch.split(logits, self.nvec, dim=1)
        multi_categoricals = [Categorical(logits=logits) for logits in split_logits]
        action = torch.stack([categorical.sample() for categorical in multi_categoricals])
        return action.detach().cpu().numpy()
    def update(self):
        # update policy every n steps
        if self.sample_count % self.update_freq != 0:
            return
        old_states, old_actions, old_log_probs, old_rewards, old_dones = self.memory.sample()
        # convert to tensor
        old_states = torch.tensor(np.array(old_states), device=self.device, dtype=torch.float32)
        old_actions = torch.tensor(np.array(old_actions), device=self.device, dtype=torch.float32)
        # monte carlo estimate of state rewards
        returns = []
        discounted_sum = 0
        for reward, done in zip(reversed(old_rewards), reversed(old_dones)):
            if done:
                discounted_sum = 0
            discounted_sum = reward + (self.gamma * discounted_sum)
            returns.insert(0, discounted_sum)
        # Normalizing the rewards:
        returns = torch.tensor(returns, device=self.device, dtype=torch.float32)
        returns = (returns - returns.mean()) / (returns.std() + 1e-5) # 1e-5 to avoid division by zero
        for _ in range(self.k_epochs):
            # compute advantage
            values = [self.critic(o) for o in old_states] # detach to avoid backprop through the critic
            values = torch.stack(values).squeeze()
            advantage = returns - values.detach()
            # get action probabilities
            
            
            logits = [self.actor(o)+self.mask for o in old_states]
            logits = torch.stack(logits).squeeze()

            split_logits = torch.split(logits, self.nvec, dim=1)
            multi_categoricals = [Categorical(logits=logits) for logits in split_logits]
            action = torch.stack([categorical.sample() for categorical in multi_categoricals])
            new_log_probs = torch.stack([categorical.log_prob(a) for a, categorical in zip(action, multi_categoricals)])
            entropy = torch.stack([categorical.entropy() for categorical in multi_categoricals])
            
            if not torch.is_tensor(old_log_probs):
                old_log_probs = torch.cat(old_log_probs, dim=1)
            ratio = torch.exp(new_log_probs - old_log_probs) # old_log_probs must be detached

            ratio = torch.sum(ratio, dim=0)
            # compute surrogate loss
            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * advantage
            # compute actor loss
            actor_loss = -torch.min(surr1, surr2).mean() + self.entropy_coef * entropy.mean()
            # compute critic loss
            critic_loss = (returns - values).pow(2).mean()
            # take gradient step
            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()
            actor_loss.backward()
            critic_loss.backward()
            self.actor_optimizer.step()
            self.critic_optimizer.step()
        self.memory.clear()

"""定义训练"""
import copy
def train(cfg, env, agent):
    ''' 训练
    '''
    print("开始训练")
    rewards = []  # 记录所有回合的奖励
    steps = []
    best_ep_reward = 0 # 记录最大回合奖励
    output_agent = None
    for i_ep in range(cfg.train_eps):
        ep_reward = 0  # 记录一回合内的奖励
        ep_step = 0
        state = env.reset()  # 重置环境，返回初始状态
        for _ in range(cfg.max_steps):
            ep_step += 1
            action = agent.sample_action(state)  # 选择动作
            next_state, reward, done, truncated, info = env.step(action)  # 更新环境，返回transition
            agent.memory.push((state, action,agent.log_probs,reward,done))  # 保存transition
            state = next_state  # 更新下一个状态
            agent.update()  # 更新智能体
            ep_reward += reward  # 累加奖励
            if done:
                break
        if (i_ep+1)%cfg.eval_per_episode == 0:
            sum_eval_reward = 0
            for _ in range(cfg.eval_eps):
                eval_ep_reward = 0
                state = env.reset()
                for _ in range(cfg.max_steps):
                    action = agent.predict_action(state)  # 选择动作
                    next_state, reward, done, truncated, info = env.step(action)  # 更新环境，返回transition
                    state = next_state  # 更新下一个状态
                    eval_ep_reward += reward  # 累加奖励
                    if done:
                        break
                sum_eval_reward += eval_ep_reward
            mean_eval_reward = sum_eval_reward/cfg.eval_eps
            if mean_eval_reward >= best_ep_reward:
                best_ep_reward = mean_eval_reward
                output_agent = copy.deepcopy(agent)
                print(f"回合：{i_ep+1}/{cfg.train_eps}，奖励：{ep_reward:.2f}，评估奖励：{mean_eval_reward:.2f}，最佳评估奖励：{best_ep_reward:.2f}，更新模型！")
            else:
                print(f"回合：{i_ep+1}/{cfg.train_eps}，奖励：{ep_reward:.2f}，评估奖励：{mean_eval_reward:.2f}，最佳评估奖励：{best_ep_reward:.2f}")
        steps.append(ep_step)
        rewards.append(ep_reward)
    print("完成训练")
    env.close()
    return output_agent,{'rewards':rewards}

def test(cfg, env, agent):
    print("开始测试")
    rewards = []  # 记录所有回合的奖励
    steps = []
    for i_ep in range(cfg.test_eps):
        ep_reward = 0  # 记录一回合内的奖励
        ep_step = 0
        state = env.reset()  # 重置环境，返回初始状态
        for _ in range(cfg.max_test_steps):
            ep_step+=1
            action = agent.predict_action(state)  # 选择动作
            next_state, reward, done, truncated, info = env.step(action)  # 更新环境，返回transition
            state = next_state  # 更新下一个状态
            ep_reward += reward  # 累加奖励
            if done:         
                break
        steps.append(ep_step)
        rewards.append(ep_reward)
        print(f"回合：{i_ep+1}/{cfg.test_eps}，奖励：{ep_reward:.2f}")
    print("完成测试")
    env.close()
    return {'rewards':rewards}

"""定义环境"""
import gym

def env_agent_config(cfg):
    env = gym.make(cfg.env_name) # 创建环境
    cfg.D = env.action_space.shape[0]  #动作空间的长
    cfg.U = env.action_space.shape[1]  #动作空间的宽
    n_states = env.observation_space.shape[0] 
    n_actions = cfg.D * cfg.U
    print(f"状态空间维度：{n_states}，动作空间维度：{n_actions}")
    # 更新n_states和n_actions到cfg参数中
    setattr(cfg, 'n_states', n_states)
    setattr(cfg, 'n_actions', n_actions) 
    agent = Agent(cfg)
    return env,agent
"""设置参数"""

import matplotlib.pyplot as plt
import seaborn as sns
class Config:
    def __init__(self) -> None:
        self.env_name = "HospitalEnv-v3" 
        self.new_step_api = False # 是否用gym的新api
        self.algo_name = "PPO" 
        self.mode = "train" # train or test
        self.device = "cpu" # device to use
        self.train_eps = 100 # 训练的回合数
        self.test_eps = 10 # 测试的回合数
        self.max_steps = 200 # 每个训练回合的最大步数
        self.max_test_steps = 1000 # 每个测试回合的最大步数
        self.eval_eps = 5 # 评估的回合数
        self.eval_per_episode = 5 # 评估的频率 
        self.gamma = 0.8 # 折扣因子
        self.k_epochs = 4 # 更新策略网络的次数
        self.actor_lr = 0.003 # actor网络的学习率
        self.critic_lr = 0.003 # critic网络的学习率
        self.eps_clip = 0.2 # epsilon-clip
        self.entropy_coef = 0.01 # entropy的系数
        self.update_freq = 100 # 更新频率
        self.actor_hidden_dim = 128 # actor网络的隐藏层维度
        self.critic_hidden_dim = 128 # critic网络的隐藏层维度
        self.num_actions = 7 # 每维空间可选的动作数量


def smooth(data, weight=0.9):  
    '''平滑曲线
    '''
    last = data[0] 
    smoothed = []
    for point in data:
        smoothed_val = last * weight + (1 - weight) * point  # 计算平滑值
        smoothed.append(smoothed_val)                    
        last = smoothed_val                                
    return smoothed

def plot_rewards(rewards,cfg, tag='train'):
    ''' 画图
    '''
    sns.set()
    plt.figure()  
    plt.title(f"{tag}ing curve on {cfg.device} of {cfg.algo_name} for {cfg.env_name}")
    plt.xlabel('epsiodes')
    plt.plot(rewards, label='rewards')
    plt.plot(smooth(rewards), label='smoothed')
    plt.legend()


"""开始训练"""

# 获取参数
cfg = Config() 
# 训练
env, agent = env_agent_config(cfg)
best_agent,res_dic = train(cfg, env, agent)
 
plot_rewards(res_dic['rewards'], cfg, tag="train")  
# In[] 测试
res_dic = test(cfg, env, best_agent)
plot_rewards(res_dic['rewards'], cfg, tag="test")  
# 保存模型
torch.save(best_agent, "model_ppo16.pt")
