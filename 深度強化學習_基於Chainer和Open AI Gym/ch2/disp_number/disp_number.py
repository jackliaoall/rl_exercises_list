# -*- coding: utf-8 -*-
"""
MNISTの画像を表示するプログラム
Copyright(c) 2018 Koji Makino and Hiromitsu Nishizaki All Rights Reserved.
"""
from sklearn.datasets import load_digits
import matplotlib.pyplot as plt
digits = load_digits()
images_and_labels = list(zip(digits.images, digits.target))
for index, (image, label) in enumerate(images_and_labels[:10]):
    plt.subplot(2, 5, index + 1)
    plt.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
    plt.axis('off')
    plt.title('Training: %i' % label)
plt.show()

print(digits.data)
print(digits.target)
print(digits.data.shape)
print(digits.data[0])
print(digits.data.reshape((len(digits.data), 1, 8, 8))[0])
