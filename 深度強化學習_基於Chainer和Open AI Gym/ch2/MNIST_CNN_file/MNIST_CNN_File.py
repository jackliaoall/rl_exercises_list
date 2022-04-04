# -*- coding: utf-8 -*-
"""
ファイルから入力画像を読み込んで数字分類のためのCNN学習プログラム
Copyright(c) 2018 Koji Makino and Hiromitsu Nishizaki All Rights Reserved.
"""
import numpy as np
import chainer
import chainer.functions as F
import chainer.links as L
import chainer.initializers as I
from chainer import training
from chainer.training import extensions
from PIL import Image # 追加


class MyChain(chainer.Chain):
    def __init__(self):
        super(MyChain, self).__init__(
            conv1=L.Convolution2D(1, 16, 3, 1, 1), # 1層目の畳み込み層（チャンネル数は16）
            conv2=L.Convolution2D(16, 64, 3, 1, 1), # 2層目の畳み込み層（チャンネル数は32）
            l3=L.Linear(None, 10), #クラス分類用
        )

    def __call__(self, x):
        h1 = F.max_pooling_2d(F.relu(self.conv1(x)), ksize=2, stride=2) # 最大値プーリングは2×2，活性化関数はReLU
        h2 = F.max_pooling_2d(F.relu(self.conv2(h1)), ksize=2, stride=2) 
        y = self.l3(h2)
        return y        

# ニューラルネットワークの登録
model = L.Classifier(MyChain(), lossfun=F.softmax_cross_entropy)
chainer.serializers.load_npz("result/CNN.model", model)

# ★追加：ファイルからの画像の読み込み
img = Image.open("number/2.png")
img = img.convert('L') # グレースケール変換
img = img.resize((8, 8)) # 8x8にリサイズ

img = 16.0 - np.asarray(img, dtype=np.float32) / 16.0 # 白黒反転，0〜15に正規化，array化
img = img[np.newaxis, np.newaxis, :, :] # 4次元行列に変換（1x1x8x8，バッチ数xチャンネル数x縦x横）
x = chainer.Variable(img)
y = model.predictor(x)
c = F.softmax(y).data.argmax()
print(c)
                    

