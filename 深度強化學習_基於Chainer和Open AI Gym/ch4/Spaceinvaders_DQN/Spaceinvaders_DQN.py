#-*- coding: utf-8 -*-
"""
Open AI Gym(Atari)スペースインベーダーゲームの学習（DQN利用）
Copyright(c) 2018 Hiromitsu Nishizaki and Koji Makino All Rrights Reserved.
"""
from __future__ import print_function
import chainer
import chainer.functions as F
import chainer.links as L
import chainerrl
import numpy as np
import gym

# Q関数
class QFunction(chainer.Chain):
    def __init__(self):
        super(QFunction, self).__init__()        
        with self.init_scope():
            self.conv1 = L.Convolution2D(3, 16, (11,9), 1, 0)  # 1層目の畳み込み層（チャンネル数は16）
            self.conv2 = L.Convolution2D(16, 32, (11,9), 1, 0) # 2層目の畳み込み層（チャンネル数は32）
            self.conv3 = L.Convolution2D(32, 64, (10,9), 1, 0) # 2層目の畳み込み層（チャンネル数は64）
            self.l4 = L.Linear(14976, 6) # アクションは6通り

    def __call__(self, x):             
        h1 = F.max_pooling_2d(F.relu(self.conv1(x)), ksize=2, stride=2)
        h2 = F.max_pooling_2d(F.relu(self.conv2(h1)), ksize=2, stride=2) 
        h3 = F.max_pooling_2d(F.relu(self.conv3(h2)), ksize=2, stride=2) 
        return chainerrl.action_value.DiscreteActionValue(self.l4(h3))

def random_action():
    return np.random.choice([0, 1, 2, 3, 4, 5]) 

# メイン関数
def main():    

    # 強化学習のパラメータ
    gamma = 0.99
    alpha = 0.5
    # max_number_of_steps = 20  #1試行のstep数
    num_episodes = 100  # 総試行回数

    # DQNのセットアップ
    q_func = QFunction()
    optimizer = chainer.optimizers.Adam(eps=1e-3)
    optimizer.setup(q_func)
    explorer = chainerrl.explorers.LinearDecayEpsilonGreedy(
        start_epsilon=1.0, end_epsilon=0.1, decay_steps=num_episodes*100, random_action_func=random_action)
    replay_buffer = chainerrl.replay_buffer.ReplayBuffer(capacity=10 ** 6)
    phi = lambda x: x.astype(np.float32, copy=False)
    agent = chainerrl.agents.DQN(
        q_func, optimizer, replay_buffer, gamma, explorer,
        replay_start_size=5000, minibatch_size = 100, update_interval=50, target_update_interval=2000, phi=phi)

    outdir = 'result'
    env = gym.make('SpaceInvaders-v0')
#    env = gym.wrappers.Monitor(env, outdir) # プレイの様子の動画データを保存（MP4形式）
    chainerrl.misc.env_modifiers.make_reward_filtered(env, lambda x: x * 0.01) # 報酬値を1以下にする

    # エピソードの試行＆強化学習スタート
    for episode in range(1, num_episodes + 1):  #試行数分繰り返す
        done = False
        reward = 0
        observation = env.reset()        
        observation = np.asarray(observation.transpose(2, 0, 1), dtype=np.float32) # 画像データの次元変換
        while not done:
            if episode % 10 == 0:
                env.render()
            action = agent.act_and_train(observation, reward)
            observation, reward, done, info = env.step(action)
            observation = np.asarray(observation.transpose(2, 0, 1), dtype=np.float32)
            print(action, reward, done, info)
        agent.stop_episode_and_train(observation, reward, done)        
        print('Episode {}: statistics: {}, epsilon {}'.format(episode, agent.get_statistics(), agent.explorer.epsilon)) 
        if episode % 10 == 0: # 10エピソード毎にエージェントモデルを保存
            agent.save('agent_spaceinvaders_' + str(episode))

if __name__ == '__main__':
    main()
