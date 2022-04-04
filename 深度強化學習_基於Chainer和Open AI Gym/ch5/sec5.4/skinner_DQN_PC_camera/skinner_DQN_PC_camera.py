# coding:utf-8
"""
PC-Arduino用
ネズミ学習問題のDQNプログラム（カメラ）
Copyright(c) 2018 Koji Makino and Hiromitsu Nishizaki All Rights Reserved.
"""
import numpy as np
import chainer
import chainer.functions as F
import chainer.links as L
import chainerrl
import time
import serial
import cv2

ser = serial.Serial(13)

class QFunction(chainer.Chain):
    def __init__(self):
        super(QFunction, self).__init__()        
        with self.init_scope():
            self.conv1 = L.Convolution2D(1, 8, 5, 1, 0)  # 1層目の畳み込み層（チャンネル数は8）
            self.conv2 = L.Convolution2D(8, 16, 5, 1, 0) # 2層目の畳み込み層（チャンネル数は16）
            self.l3 = L.Linear(400, 2) # アクションは2通り

    def __call__(self, x):        
        h1 = F.max_pooling_2d(F.relu(self.conv1(x)), ksize=2, stride=2)
        h2 = F.max_pooling_2d(F.relu(self.conv2(h1)), ksize=2, stride=2) 
        return chainerrl.action_value.DiscreteActionValue(self.l3(h2))

def random_action():
    return np.random.choice([0, 1])
 
def step(state, action):
    reward = 0
    if action==0:
        ser.write(b"a")
    else:
        ser.write(b"b")
    time.sleep(1.0)
    state = int(ser.read());
    reward = int(ser.read());
    return np.array([state]), reward

# USBカメラから画像を取得（ラズパイ用）
def capture(ndim=3):
    cap = cv2.VideoCapture(0)
    _, frame = cap.read()
    cap.release()
    cx = frame.shape[1] // 2  # 中心画素の取得 
    cy = frame.shape[0] // 2
    xoffset = 10
    yoffset = -10
    frame = frame[cy+yoffset-150:cy+yoffset+150, cx+xoffset-150:cx+xoffset+150] # 中央付近の300x300画素を切り出し
    img = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (32, 32)) #グレー化＋32x32にリサイズ    
    env = np.asarray(img, dtype=np.float32)
    if ndim == 3:
        return env[np.newaxis, :, :] # 2次元→3次元行列（replay用）
    else:
        return env[np.newaxis, np.newaxis, :, :] # 4次元行列（判定用）

gamma = 0.9
alpha = 0.5
max_number_of_steps = 5  #1試行のstep数
num_episodes = 50  #総試行回数

q_func = QFunction()
optimizer = chainer.optimizers.Adam(eps=1e-2)
optimizer.setup(q_func)
explorer = chainerrl.explorers.LinearDecayEpsilonGreedy(start_epsilon=1.0, end_epsilon=0.1, decay_steps=num_episodes, random_action_func=random_action)
replay_buffer = chainerrl.replay_buffer.ReplayBuffer(capacity=10 ** 6)
phi = lambda x: x.astype(np.float32, copy=False)
agent = chainerrl.agents.DQN(
    q_func, optimizer, replay_buffer, gamma, explorer,
    replay_start_size=500, update_interval=1, target_update_interval=100, phi=phi)
#agent.load('agent')

for episode in range(num_episodes):  #試行数分繰り返す
    state = np.array([0])
    R = 0
    reward = 0
    done = True
    ser.write(b"c")
 
    for t in range(max_number_of_steps):  #1試行のループ
        camera_state = capture(ndim=3)
        action = agent.act_and_train(camera_state, reward)
        next_state, reward = step(state, action)
        print(camera_state, action, reward)
        R += reward  #報酬を追加
        state = next_state
    agent.stop_episode_and_train(capture(ndim=3), reward, done)

    print('episode : %d total reward %d' %(episode+1, R))
ser.close()
#agent.save('agent')
