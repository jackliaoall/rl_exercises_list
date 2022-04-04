# coding:utf-8
"""
倒立振子のQラーニングプログラム（学習済みQ値を読み込んでテスト）
Copyright(c) 2018 Koji Makino and Hiromitsu Nishizaki All Rights Reserved.
"""
import gym
import numpy as np
import time

def digitize_state(observation):
    p, v, a, w = observation
    d = num_digitized
    pn = np.digitize(p, np.linspace(-2.4, 2.4, d+1)[1:-1])
    vn = np.digitize(v, np.linspace(-3.0, 3.0, d+1)[1:-1])
    an = np.digitize(a, np.linspace(-0.5, 0.5, d+1)[1:-1])
    wn = np.digitize(w, np.linspace(-2.0, 2.0, d+1)[1:-1])
    return pn + vn*d + an*d**2 + wn*d**3

def get_action(next_state, episode):
    epsilon = 0
    if epsilon <= np.random.uniform(0, 1):
        a = np.where(q_table[next_state]==q_table[next_state].max())[0]
        next_action = np.random.choice(a)
    else:
        next_action = np.random.choice([0, 1])
    return next_action

def update_Qtable(q_table, state, action, reward, next_state):
    gamma = 0.99
    alpha = 0.5
    next_maxQ=max(q_table[next_state])
    q_table[state, action] = (1 - alpha) * q_table[state, action] +\
            alpha * (reward + gamma * next_maxQ)
    return q_table

env = gym.make('CartPole-v0')
max_number_of_steps = 200  #1試行のstep数
num_episodes = 1000  #総試行回数
num_digitized = 6  #分割数
#q_table = np.random.uniform(low=-1, high=1, size=(num_digitized**4, env.action_space.n))
q_table = np.loadtxt('Qvalue.txt')

for episode in range(num_episodes):  #試行数分繰り返す
    observation = env.reset()
    state = digitize_state(observation)
    action = np.argmax(q_table[state])
    episode_reward = 0
 
    for t in range(max_number_of_steps):  #1試行のループ
        if episode %10 == 0:
            env.render()
        observation, reward, done, info = env.step(action)
        if done and t < max_number_of_steps-1:
            reward -= max_number_of_steps  #棒が倒れたら罰則
        episode_reward += reward  #報酬を追加
        next_state = digitize_state(observation)  #t+1での観測状態を、離散値に変換
#        q_table = update_Qtable(q_table, state, action, reward, next_state)
        action = get_action(next_state, episode)    # a_{t+1} 
        state = next_state
        if done:
            break
    print('episode:', episode, 'R:', episode_reward)
np.savetxt('Qvalue.txt', q_table)
