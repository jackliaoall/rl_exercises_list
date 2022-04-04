# -*- coding: utf-8 -*-
"""
論理演算子ORの学習プログラム（重みの表示）
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

# ニューラルネットワークの登録
model = L.Classifier(MyChain(), lossfun=F.softmax_cross_entropy)
chainer.serializers.load_npz("result/out.model", model)
print (model.predictor.l1.W.data) # ノードの重み
print (model.predictor.l1.b.data) # バイアスの重み

