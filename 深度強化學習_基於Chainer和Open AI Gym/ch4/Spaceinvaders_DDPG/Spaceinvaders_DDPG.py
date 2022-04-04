#-*- coding: utf-8 -*-
"""
スペースインベーダーゲームの学習（DDPGを利用）
Copyright(c) Hiromitsu Nishizaki and Koji Makino All Rrights Reserved.
"""
from __future__ import print_function
import chainer
import chainer.functions as F
import chainer.links as L
import chainerrl
from chainerrl.agents.ddpg import DDPG
from chainerrl.agents.ddpg import DDPGModel
from chainerrl import distribution

import numpy as np
import gym

# ポリシーネットワーク
class PolicyNetwork(chainer.Chain):
    def __init__(self):
        super(PolicyNetwork, self).__init__()        
        with self.init_scope():
            self.conv1 = L.Convolution2D(3, 16, (11, 9), 1, 0)  # 1層目の畳み込み層（チャンネル数は16）
            self.conv2 = L.Convolution2D(16, 32, (11, 9), 1, 0) # 2層目の畳み込み層（チャンネル数は32）
            self.conv3 = L.Convolution2D(32, 64, (10, 9), 1, 0) # 2層目の畳み込み層（チャンネル数は64）
            self.l4 = L.Linear(14976, 6) # アクションは6通り

    def __call__(self, x):             
        h1 = F.max_pooling_2d(F.relu(self.conv1(x)), ksize=2, stride=2)
        h2 = F.max_pooling_2d(F.relu(self.conv2(h1)), ksize=2, stride=2) 
        h3 = F.max_pooling_2d(F.relu(self.conv3(h2)), ksize=2, stride=2)
        h4 = F.tanh(self.l4(h3)) # -1〜+1に納める
        return distribution.ContinuousDeterministicDistribution(h4) #連続値（6次元のベクトル）を返す

# Q関数
class QFunction(chainer.Chain):
    def __init__(self):
        super(QFunction, self).__init__()        
        with self.init_scope():
            self.conv1 = L.Convolution2D(3, 16, (11, 9), 1, 0)  # 1層目の畳み込み層（チャンネル数は16）
            self.conv2 = L.Convolution2D(16, 32, (11, 9), 1, 0) # 2層目の畳み込み層（チャンネル数は32）
            self.conv3 = L.Convolution2D(32, 64, (10, 9), 1, 0) # 2層目の畳み込み層（チャンネル数は32）            
            self.l4 = L.Linear(14876, 1000) # 状態を1000次元に変換
            self.l5 = L.Linear(1000+6, 1) # 1000+6（6は状態次元数）

    def __call__(self, s, action):
        h1 = F.max_pooling_2d(F.relu(self.conv1(s)), ksize=2, stride=2)
        h2 = F.max_pooling_2d(F.relu(self.conv2(h1)), ksize=2, stride=2) 
        h3 = F.max_pooling_2d(F.relu(self.conv3(h2)), ksize=2, stride=2) 
        h4 = F.tanh(self.l4(h3)) # -1〜+1に納める
        h5 = F.concat((h4, action), axis=1) # 状態と行動を結合
        return self.l5(h5) # 状態と行動からQ値を求める

# メイン関数
def main():    

    # 強化学習のパラメータ
    gamma = 0.995
    num_episodes = 100  #総試行回数

    # DDPGセットアップ
    q_func = QFunction() # Q関数
    policy = PolicyNetwork() # ポリシーネットワーク
    model = DDPGModel(q_func=q_func, policy=policy)
    optimizer_p = chainer.optimizers.Adam(alpha=1e-4)
    optimizer_q = chainer.optimizers.Adam(alpha=1e-3)
    optimizer_p.setup(model['policy'])
    optimizer_q.setup(model['q_function'])

    explorer = chainerrl.explorers.AdditiveOU(sigma=1.0) # sigmaで付与するノイズの強さを設定
    replay_buffer = chainerrl.replay_buffer.ReplayBuffer(capacity=10 ** 6)
    phi = lambda x: x.astype(np.float32, copy=False)

    agent = DDPG(model, optimizer_p, optimizer_q, replay_buffer, gamma=gamma,
                 explorer=explorer, replay_start_size=1000,
                 target_update_method='soft',
                 target_update_interval=1, 
                 update_interval=4, 
                 soft_update_tau=0.01,
                 n_times_update=1,
                 phi=phi, gpu=-1, minibatch_size=200)

    def reward_filter(r): # 報酬値を小さくする（0〜1の範囲になるようにする）
        return r * 0.01

    outdir = 'result'
    chainerrl.misc.set_random_seed(0)
    env = gym.make('SpaceInvaders-v0') #スペースインベーダーの環境呼び出し
    env.seed(0)
    chainerrl.misc.env_modifiers.make_reward_filtered(env, reward_filter)
    env = gym.wrappers.Monitor(env, outdir) # 動画を保存
    
    # エピソードの試行＆強化学習スタート
    for episode in range(1, num_episodes + 1):  #試行数分繰り返す
        done = False
        reward = 0
        n_steps = 0
        total_reward = 0
        obs = env.reset()        
        obs = np.asarray(obs.transpose(2, 0, 1), dtype=np.float32)
        while not done:            
            action = agent.act_and_train(obs, reward) # actionは連続値
            action = F.argmax(action).data # 出力値が最大の行動を選択
            obs, reward, done, info = env.step(action) # actionを実行
            total_reward += reward
            n_steps += 1
            obs = np.asarray(obs.transpose(2, 0, 1), dtype=np.float32)
            print('{0:4d}: action {1}, reward {2}, done? {3}, {4}'.format(n_steps, action, reward, done, info))
        agent.stop_episode_and_train(obs, reward, done)        
        print('Episode {0:4d}: total reward {1}, n_steps {2}, statistics: {3}'.format(episode, total_reward, n_steps, agent.get_statistics())) 
        if episode % 10 == 0:
            agent.save('agent_DDPG_spaceinvaders_' + str(episode))

if __name__ == '__main__':
    main()
