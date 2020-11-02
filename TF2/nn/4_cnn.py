#https://zhuanlan.zhihu.com/p/60900902

import matplotlib.pyplot as plt
import tensorflow
import tensorflow.keras as K
import numpy as np
from tensorflow.keras import layers

(x_train, y_train), (x_test, y_test) = K.datasets.mnist.load_data()
x_train = x_train.reshape((-1,28,28,1))
x_test = x_test.reshape((-1,28,28,1))
print(x_train.shape, ' ', y_train.shape)
print(x_test.shape, ' ', y_test.shape)





