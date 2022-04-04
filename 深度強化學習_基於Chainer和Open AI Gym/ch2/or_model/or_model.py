# -*- coding: utf-8 -*-
"""
論理演算子ORの学習プログラム（モデル入力）
Copyright(c) 2018 Koji Makino and Hiromitsu Nishizaki All Rights Reserved.
"""
import numpy as np
import chainer
import chainer.functions as F
import chainer.links as L
import chainer.initializers as I
from chainer import training
from chainer.training import extensions

class MyChain(chainer.Chain):
    def __init__(self):
        super(MyChain, self).__init__()
        with self.init_scope():
            self.l1 = L.Linear(2, 3)#入力2，中間層3
            self.l2 = L.Linear(3, 2)#中間層3，出力2
    def __call__(self, x):
        h1 = F.relu(self.l1(x))#ReLU関数
        y = self.l2(h1)
        return y        

# データの作成
test = np.array(([0,0], [0,1], [1,0], [1,1], [0.7,0.8], [0.2,0.4], [0.9,0.2]), dtype=np.float32)

# ニューラルネットワークの登録
model = L.Classifier(MyChain(), lossfun=F.softmax_cross_entropy)
chainer.serializers.load_npz("result/out.model", model)
# 学習結果の評価
for i in range(len(test)):
    x = chainer.Variable(test[i].reshape(1,2))
    result = F.softmax(model.predictor(x))
    print("input: {}, result: {}".format(test[i], result.data.argmax()))

