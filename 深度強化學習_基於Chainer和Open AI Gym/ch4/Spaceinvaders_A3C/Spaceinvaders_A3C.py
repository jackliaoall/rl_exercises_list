#-*- coding: utf-8 -*-
"""
スペースインベーダーゲームの学習（A3Cバージョン）
Copyright(c) Hiromitsu Nishizaki and Koji Makino All Rrights Reserved.
"""
from __future__ import print_function
import chainer
import chainer.functions as F
import chainer.links as L
import chainerrl
import numpy as np
import gym
import logging

# 型変換用関数
def phi(obs):
    return obs.astype(np.float32)

# A3C FeedForward Softmax
class A3CLSTMSoftmax(chainer.Chain, chainerrl.agents.a3c.A3CModel, chainerrl.recurrent.RecurrentChainMixin):
    
    # ポリシーネットワークと評価関数でCNN部分は共通化する
    def __init__(self):
        super(A3CLSTMSoftmax, self).__init__()        
        with self.init_scope():
            self.conv1 = L.Convolution2D(3, 16, (11, 9), 1, 0)  # 1層目の畳み込み層（チャンネル数は16）
            self.conv2 = L.Convolution2D(16, 32, (11, 9), 1, 0) # 2層目の畳み込み層（チャンネル数は32）
            self.conv3 = L.Convolution2D(32, 64, (10, 9), 1, 0) # 2層目の畳み込み層（チャンネル数は64）
            self.l4p = L.LSTM(14976, 1024) #ポリシーネットワーク
            self.l4v = L.LSTM(14976, 1024) #バリューネットワーク
            self.l5p = L.Linear(1024, 1024)
            self.l5v = L.Linear(1024, 1024)            
            self.pi = chainerrl.policies.SoftmaxPolicy(L.Linear(1024, 6)) # ポリシーネットワーク
            self.v = L.Linear(1024, 1) # バリューネットワーク

    def pi_and_v(self, state):
        state = np.asarray(state.transpose(0, 3, 1, 2), dtype=np.float32)
        h1 = F.max_pooling_2d(F.relu(self.conv1(state)), ksize=2, stride=2)
        h2 = F.max_pooling_2d(F.relu(self.conv2(h1)), ksize=2, stride=2) 
        h3 = F.max_pooling_2d(F.relu(self.conv3(h2)), ksize=2, stride=2) #ここまでは共通
        h4p = self.l4p(h3)
        h4v = self.l4v(h3)
        h5p = F.relu(self.l5p(h4p)) 
        h5v = F.relu(self.l5p(h4v))
        pout = self.pi(h5p) #ポリシーネットワークの出力
        vout = self.v(h5v) #バリューネットワークの出力
        return pout, vout


# メイン関数
def main():    

    # 初期設定（プロセス数は8）
    n_process = 8
    outdir = 'result'

    # スペースインベーダー環境の設定
    chainerrl.misc.set_random_seed(0)
    process_seeds = np.arange(n_process)

    def make_env(process_idx, test=False):
        env = gym.make('SpaceInvaders-v0')
        process_seed = int(process_seeds[process_idx])
        if not test:
            chainerrl.misc.env_modifiers.make_reward_filtered(env, lambda x: x * 0.01)
        if process_idx == 0 and not test:
            env = gym.wrappers.Monitor(env, outdir)
        env_seed = 2 ** 32 - 1 - process_seed if test else process_seed
        env.seed(env_seed)
        return env
    
    
    # 強化学習のパラメータ        
    num_episodes = 100000  #試行回数

    # DQNのセットアップ
    model = A3CLSTMSoftmax()
    optimizer = chainerrl.optimizers.rmsprop_async.RMSpropAsync(lr=0.001, eps=0.1, alpha=0.99)
    optimizer.setup(model)
    
    agent = chainerrl.agents.a3c.A3C(
        model, optimizer, t_max=8, gamma=0.995, beta=0.1, phi=phi)
    
    # DEBUG用にログを表示
    gym.logger.set_level(0)
    logging.basicConfig(level=logging.DEBUG)

    # エピソードの試行＆強化学習スタート（トレーナーを利用）
    chainerrl.experiments.train_agent_async(
            agent=agent,
            outdir=outdir,
            processes=n_process,
            make_env=make_env,
            profile=True,
            steps=1000000,
            eval_interval=None,
            max_episode_len=num_episodes,
            logger=gym.logger)

if __name__ == '__main__':
    main()
