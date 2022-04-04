# coding:utf-8
"""
ネズミ学習問題のQラーニングプログラム
Copyright(c) 2018 Koji Makino and Hiromitsu Nishizaki All Rights Reserved.
"""
import numpy as np
 
def random_action():
    return np.random.choice([0, 1])
 
def get_action(next_state, episode):
    epsilon = 0.5 * (1 / (episode + 1))           #徐々に最適行動のみをとる、ε-greedy法
    if epsilon <= np.random.uniform(0, 1):
        a = np.where(q_table[next_state]==q_table[next_state].max())[0]
        next_action = np.random.choice(a)
    else:
        next_action = random_action()
    return next_action

def step(state, action):
    reward = 0
    if state==0:
        if action==0:
            state = 1
        else:
            state = 0
    else:
        if action==0:
            state = 0
        else:
            state = 1
            reward = 1
    return state, reward

def update_Qtable(q_table, state, action, reward, next_state):
    gamma = 0.9
    alpha = 0.5
    next_maxQ=max(q_table[next_state])
    q_table[state, action] = (1 - alpha) * q_table[state, action] +\
            alpha * (reward + gamma * next_maxQ)
   
    return q_table

max_number_of_steps = 5  #1試行のstep数
num_episodes = 10  #総試行回数
q_table = np.zeros((2, 2))
 
for episode in range(num_episodes):  #試行数分繰り返す
    state = 0
    episode_reward = 0
 
    for t in range(max_number_of_steps):  #1試行のループ
        action = get_action(state, episode)    # a_{t+1} 
        next_state, reward = step(state, action)
        print(state, action, reward)
        episode_reward += reward  #報酬を追加
        q_table = update_Qtable(q_table, state, action, reward, next_state)
        state = next_state

    print('episode : %d total reward %d' %(episode+1, episode_reward))
    print(q_table)
