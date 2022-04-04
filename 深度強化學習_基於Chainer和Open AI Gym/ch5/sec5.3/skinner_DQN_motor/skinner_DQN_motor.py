# coding:utf-8
"""
RaspberryPi用
ネズミ学習問題用プログラム（センサとRCサーボモータ）
Copyright(c) 2018 Koji Makino and Hiromitsu Nishizaki All Rights Reserved.
"""
import numpy as np
import chainer
import chainer.functions as F
import chainer.links as L
import chainerrl
import time
import RPi.GPIO as GPIO
import Adafruit_PCA9685

pwm = Adafruit_PCA9685.PCA9685()
pwm.set_pwm_freq(60)
pwm.set_pwm(0, 0, 375) # サーボモータを初期位置へ
time.sleep(1)

class QFunction(chainer.Chain):
    def __init__(self, obs_size, n_actions, n_hidden_channels=2):
        super(QFunction, self).__init__()
        with self.init_scope():
            self.l1=L.Linear(obs_size, n_hidden_channels)
            self.l2=L.Linear(n_hidden_channels, n_hidden_channels)
            self.l3=L.Linear(n_hidden_channels, n_actions)

    def __call__(self, x, test=False):
        h1 = F.tanh(self.l1(x))
        h2 = F.tanh(self.l2(h1))
        y = chainerrl.action_value.DiscreteActionValue(self.l3(h2))
        return y

def random_action():
    return np.random.choice([0, 1])
 
def step(state, action):
    reward = 0
    if state==0:
        if action==0:
            pwm.set_pwm(0, 0, 150) 
            time.sleep(1)
#            if GPIO.output(13)==1:#電源スイッチが入力されれば
            GPIO.output(16, 1)#電源LED ON
            state = 1
            pwm.set_pwm(0, 0, 375) # サーボモータを初期位置へ
            time.sleep(1)
        else:
            pwm.set_pwm(0, 0, 600) 
            time.sleep(1)
            state = 0
            pwm.set_pwm(0, 0, 375) # サーボモータを初期位置へ
            time.sleep(1)
    else:
        if action==0:
            pwm.set_pwm(0, 0, 150) 
            time.sleep(1)
#            if GPIO.output(13)==1:#電源スイッチが入力されれば
            GPIO.output(16, 0)#電源LED OFF
            state = 0
            pwm.set_pwm(0, 0, 375) # サーボモータを初期位置へ
            time.sleep(1)
        else:
            pwm.set_pwm(0, 0, 600) 
            time.sleep(1)
#            if GPIO.output(15)==1:#商品スイッチが入力されれば
            GPIO.output(18, 1)#商品LED ON
            state = 1
            reward = 1
            pwm.set_pwm(0, 0, 375) # サーボモータを初期位置へ
            time.sleep(1)
            GPIO.output(18, 0)#商品LED OFF
    return np.array([state]), reward

gamma = 0.9
alpha = 0.5
max_number_of_steps = 5  #1試行のstep数
num_episodes = 10  #総試行回数

q_func = QFunction(1, 2)
optimizer = chainer.optimizers.Adam(eps=1e-2)
optimizer.setup(q_func)
explorer = chainerrl.explorers.LinearDecayEpsilonGreedy(start_epsilon=1.0, end_epsilon=0.1, decay_steps=num_episodes, random_action_func=random_action)
replay_buffer = chainerrl.replay_buffer.ReplayBuffer(capacity=10 ** 6)
phi = lambda x: x.astype(np.float32, copy=False)
agent = chainerrl.agents.DQN(
    q_func, optimizer, replay_buffer, gamma, explorer,
    replay_start_size=500, update_interval=1, target_update_interval=100, phi=phi)
#agent.load('agent')

# ラズパイGPIO関係のセットアップ
GPIO.setmode(GPIO.BOARD)
GPIO.setup(16, GPIO.OUT)
GPIO.setup(18, GPIO.OUT)
GPIO.setup(13, GPIO.IN)
GPIO.setup(15, GPIO.IN)

for episode in range(num_episodes):  #試行数分繰り返す
    state = np.array([0])
    R = 0
    reward = 0
    done = True
 
    for t in range(max_number_of_steps):  #1試行のループ
        action = agent.act_and_train(state, reward)
        next_state, reward = step(state, action)
        print(state, action, reward)
        R += reward  #報酬を追加
        state = next_state
    agent.stop_episode_and_train(state, reward, done)

    print('episode : %d total reward %d' %(episode+1, R))
ser.close()
#agent.save('agent')
