# -*- coding: utf-8 -*-
"""
カメラから入力画像を読み込んで数字分類のためのCNN学習プログラム（28x28画像使用）
Copyright(c) 2018 Koji Makino and Hiromitsu Nishizaki All Rights Reserved.
"""
import numpy as np
import chainer
import chainer.functions as F
import chainer.links as L
import chainer.initializers as I
from chainer import training
from chainer.training import extensions
import cv2


class MyChain(chainer.Chain):
    def __init__(self):
        super(MyChain, self).__init__()
        with self.init_scope():
#            self.conv1=L.Convolution2D(1, 16, 5, 1, 0) # 1層目の畳み込み層（チャンネル数は16）
#            self.conv2=L.Convolution2D(16, 64, 5, 1, 0) # 2層目の畳み込み層（チャンネル数は64）
            self.conv1=L.Convolution2D(1, 4, 3, 1, 1) # 1層目の畳み込み層（チャンネル数は4）
            self.conv2=L.Convolution2D(4, 8, 3, 1, 1) # 2層目の畳み込み層（チャンネル数は8）
            self.l3=L.Linear(None, 10) #クラス分類用
        
    def __call__(self, x):
        h1 = F.max_pooling_2d(F.relu(self.conv1(x)), ksize=2, stride=2) # 最大値プーリングは2×2，活性化関数はReLU
        h2 = F.max_pooling_2d(F.relu(self.conv2(h1)), ksize=2, stride=2) 
        return self.l3(h2)      

# ニューラルネットワークの登録
model = L.Classifier(MyChain(), lossfun=F.softmax_cross_entropy)
chainer.serializers.load_npz("result/CNN.model", model)

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()            
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    xp = int(frame.shape[1]/2)
    yp = int(frame.shape[0]/2)
    d = 28*2
    cv2.rectangle(gray, (xp-d, yp-d), (xp+d, yp+d), color=0, thickness=2)
    cv2.imshow('gray', gray)
    if cv2.waitKey(10) == 113:
        break
    gray = cv2.resize(gray[yp-d:yp + d, xp-d:xp + d],(28, 28))
    img = np.zeros((28,28), dtype=np.float32)
    img[np.where(gray>64)]=1
    img = 1-np.asarray(img,dtype=np.float32)  # 0〜1に正規化してみる
    img = img[np.newaxis, np.newaxis, :, :] # 4次元行列に変換（1x1x8x8，バッチ数xチャンネル数x縦x横）
    x = chainer.Variable(img)
    y = model.predictor(x)
    c = F.softmax(y).data.argmax()
    print(c)
                    
cap.release()
