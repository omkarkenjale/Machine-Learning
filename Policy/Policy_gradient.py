#!/usr/bin/env python
# coding: utf-8

# # REINFORCEMENT LEARNING  
# [Link](https://github.com/pytorch/examples/blob/master/reinforcement_learning/reinforce.py)

# ### Environment dynamics
# - State - (Cart Position, Cart Velocity, Pole Angle, Pole Velocity)
# - Action - Move Left or Move Right
# - Reward function - +1 for every incremental timestep
# - Termination - if the pole falls over too far or the cart moves more then 2.4 units away from center

# Things we need to know clearly to understand below:
# 
# What is an environment? What is a state? What is an action? What is an episode? What is a policy?

# [Explaining RL](https://flappybird.io/)

# In[1]:


# OpenAI gym
import gym
env = gym.make('CartPole-v1')


# [OpenAI gym](https://gym.openai.com/envs/#classic_control)

# In[9]:


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[3]:


class Policy(nn.Module):
    def __init__(self):
        super(Policy, self).__init__()
        self.affine1 = nn.Linear(4, 24)
        self.affine2 = nn.Linear(24, 36)
        self.dropout = nn.Dropout(p=0.6)
        self.affine3 = nn.Linear(36, 2)
        
        self.saved_log_probs = []
        self.rewards = []
    
    def forward(self, x):
        x = self.affine1(x)
        #x = self.dropout(x)
        x = self.affine2(x)
        x = self.dropout(x)
        x = F.relu(x)
        action_scores = self.affine3(x)
        return F.softmax(action_scores, dim=1)

policy = Policy()
optimizer = optim.Adam(policy.parameters(), lr=1e-2)
eps = np.finfo(np.float32).eps.item()


# In[4]:


from torch.distributions import Categorical
def select_action(state):
    state = torch.from_numpy(state).float().unsqueeze(0)
    probs = policy(state) # FORWARD PASS
    m = Categorical(probs) # we are sampling from a distribution to add some exploration to the policy's behavior. 
    action = m.sample()
    policy.saved_log_probs.append(m.log_prob(action))
    return action.item()


# In[5]:


gamma = 0.99 # discount factor
def finish_episode_and_update():
    R = 0
    policy_loss = []
    returns = []
    for r in policy.rewards[::-1]:
        R = r + gamma * R
        returns.insert(0, R)
    print("R", R)
    #print(returns)
    returns = torch.tensor(returns)
    returns = (returns - returns.mean()) / (returns.std() + eps)
    for log_prob, R in zip(policy.saved_log_probs, returns):
        policy_loss.append(-log_prob * R)
    optimizer.zero_grad()
    policy_loss = torch.cat(policy_loss).sum()
    policy_loss.backward() # backward pass
    optimizer.step()
    del policy.rewards[:]
    del policy.saved_log_probs[:]


# In[6]:


def main():
    num_episodes = 1000
    reward_epd = []
    for _ in range(num_episodes):
        state = env.reset() # reset() resets the environment
        episode_reward = 0 
        for t in range(1, 10000): # no of steps 
            action = select_action(state)
            state, reward, done, _ = env.step(action)
            env.render() # show 
            policy.rewards.append(reward)
            episode_reward += reward
            if done:
                print("episode ended!")
                break
        reward_epd.append(episode_reward)

        finish_episode_and_update()
    return reward_epd


# Some News article covering RL in games
# 
# [1](https://www.bbc.com/news/technology-40287270)
# 
# [2](https://www.theverge.com/2019/10/30/20939147/deepmind-google-alphastar-starcraft-2-research-grandmaster-level)
# 
# [3](https://bdtechtalks.com/2019/04/17/openai-five-neural-networks-dota-2/)
# 
# [4](https://www.vox.com/future-perfect/2019/9/20/20872672/ai-learn-play-hide-and-seek)
# 
# [5](https://www.wired.com/story/a-robot-teaches-itself-to-play-jenga/)
# 

# reward function failure modes(https://openai.com/blog/faulty-reward-functions/)

# 
# ### [Good resource to read on policy optimization/deep RL](https://spinningup.openai.com/en/latest/spinningup/rl_intro3.html)

# In[7]:


reward = main()


# In[15]:


fig, ((ax1)) = plt.subplots();

ax1.plot(reward)
ax1.set_title('Episode Vs Reward')
ax1.set_xlabel('Number of episodes'); ax1.set_ylabel('Episode Reward')
fig.tight_layout(pad=2)
plt.show()
fig.savefig('results.png')


# In[ ]:




