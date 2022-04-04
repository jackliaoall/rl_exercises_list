# -*- coding: utf-8 -*-
"""
MNISTを用いた数字分類のためのCNN学習プログラム（28x28画像使用）
Copyright(c) 2018 Koji Makino and Hiromitsu Nishizaki All Rights Reserved.
"""
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

epoch = 20
batchsize = 100

# データの作成
train, test = chainer.datasets.get_mnist(ndim=3)

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
trainer.extend(extensions.Evaluator(test_iter, model))# エポック数の表示
#trainer.extend(extensions.dump_graph('main/loss'))#ニューラルネットワークの構造
#trainer.extend(extensions.PlotReport(['main/loss', 'validation/main/loss'], 'epoch',file_name='loss.png'))#誤差のグラフ
#trainer.extend(extensions.PlotReport(['main/accuracy', 'validation/main/accuracy'],'epoch', file_name='accuracy.png'))#精度のグラフ
trainer.extend(extensions.LogReport())#ログ
trainer.extend(extensions.PrintReport(['epoch', 'main/loss', 'validation/main/loss','main/accuracy', 'validation/main/accuracy', 'elapsed_time'] ))#計算状態の表示
trainer.extend(extensions.snapshot(), trigger=(10, 'epoch'))# エポック毎にトレーナーの状態（モデルパラメータも含む）を保存する（なんらかの要因で途中で計算がとまっても再開できるように）
#chainer.serializers.load_npz("result/snapshot_iter_1437", trainer)

# 学習開始
trainer.run()

# モデルの保存
chainer.serializers.save_npz("result/CNN.model", model)
