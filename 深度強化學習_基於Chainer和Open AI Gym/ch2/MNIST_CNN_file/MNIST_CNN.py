# -*- coding: utf-8 -*-
"""
MNISTを用いた数字分類のためのCNN学習プログラム
Copyright(c) 2018 Koji Makino and Hiromitsu Nishizaki All Rights Reserved.
"""
import numpy as np
import chainer
import chainer.functions as F
import chainer.links as L
import chainer.initializers as I
from chainer import training
from chainer.training import extensions
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

class MyChain(chainer.Chain):
    def __init__(self):
        super(MyChain, self).__init__()
        with self.init_scope():
            self.conv1=L.Convolution2D(1, 16, 3, 1, 1) # 1層目の畳み込み層（チャンネル数は16）
            self.conv2=L.Convolution2D(16, 64, 3, 1, 1) # 2層目の畳み込み層（チャンネル数は64）
            self.l3=L.Linear(256, 10) #クラス分類用
    def __call__(self, x):
        h1 = F.max_pooling_2d(F.relu(self.conv1(x)), 2, 2) # 最大値プーリングは2×2，活性化関数はReLU
        h2 = F.max_pooling_2d(F.relu(self.conv2(h1)), 2, 2) 
        y = self.l3(h2)
        return y        

epoch = 20
batchsize = 100

# データの作成
digits = load_digits()
data_train, data_test, label_train, label_test = train_test_split(digits.data, digits.target, test_size=0.2)
data_train = data_train.reshape((len(data_train), 1, 8, 8))#1x8x8の行列に変更
data_test = data_test.reshape((len(data_test), 1, 8, 8))#
data_train = (data_train).astype(np.float32)
data_test = (data_test).astype(np.float32)
train = chainer.datasets.TupleDataset(data_train, label_train)
test = chainer.datasets.TupleDataset(data_test, label_test)

# ニューラルネットワークの登録
model = L.Classifier(MyChain(), lossfun=F.softmax_cross_entropy)
optimizer = chainer.optimizers.Adam()
optimizer.setup(model)

# イテレータの定義
train_iter = chainer.iterators.SerialIterator(train, batchsize)# 学習用
test_iter = chainer.iterators.SerialIterator(test, batchsize, repeat=False, shuffle=False)# 評価用

# アップデータの登録
updater = training.StandardUpdater(train_iter, optimizer)

# トレーナーの登録
trainer = training.Trainer(updater, (epoch, 'epoch'))

# 学習状況の表示や保存
trainer.extend(extensions.LogReport())#ログ
trainer.extend(extensions.Evaluator(test_iter, model))# エポック数の表示
trainer.extend(extensions.PrintReport(['epoch', 'main/loss', 'validation/main/loss','main/accuracy', 'validation/main/accuracy', 'elapsed_time'] ))#計算状態の表示
#trainer.extend(extensions.dump_graph('main/loss'))#ニューラルネットワークの構造
#trainer.extend(extensions.PlotReport(['main/loss', 'validation/main/loss'], 'epoch',file_name='loss.png'))#誤差のグラフ
#trainer.extend(extensions.PlotReport(['main/accuracy', 'validation/main/accuracy'],'epoch', file_name='accuracy.png'))#精度のグラフ
#trainer.extend(extensions.snapshot(), trigger=(100, 'epoch'))# 学習再開のためのスナップショット出力

#chainer.serializers.load_npz("result/snapshot_iter_500", trainer)#再開用

# 学習開始
trainer.run()

# 途中状態の保存
chainer.serializers.save_npz("result/CNN.model", model)

