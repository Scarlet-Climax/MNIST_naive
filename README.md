# 跑例程一时爽，一直跑例程一直爽

[TOC]

这篇文章教你怎么用keras来XJB搭一个（对服饰图像分类的）MNIST模型

## 导入tf库

首先是导入tensorflow和keras

```py
import tensorflow as tf
from tensorflow import keras

import numpy as np
import matplotlib.pyplot as plt
```

## 获取数据集

然后我们从指定地点获取训练集和测试集

```py
fashion_mnist = keras.datasets.fashion_mnist

(train_images, train_labels), (test_images,
                               test_labels) = fashion_mnist.load_data()
```

## 预处理

### 标签映射

训练集中的标签只是0-9的数字（见下表），所以需要一个映射来看懂数字指的是什么服饰

标签|类别
--|--
0|T恤/上衣
1|裤子
2|套衫
3|裙子
4|外套
5|凉鞋
6|衬衫
7|运动鞋
8|包
9|靴子

```py
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
```

### 数据处理

康一眼图像，发现像素值介于0到255之间：
```py
plt.figure()
plt.imshow(train_images[0])
plt.colorbar()
plt.grid(False)
```

那么就要把他们缩小到0到1之间

```py
train_images = train_images / 255.0
test_images = test_images / 255.0
```

## 模型搭建

### 设置层

啥都别说，先看模型

```py
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation=tf.nn.relu),
    keras.layers.Dense(10, activation=tf.nn.softmax)
])
```

大概可以看出来，我们这个模型有三层

1. 第一层相当于把28*28的二维数组“展平”成一维数组
2. 第二层相当于构建一个128个神经元（感知器，以ReLU为激活函数）的全连接层
3. 第三层相当于构建一个具有10个概率得分的数组，其和为1

### 编译模型

训练前再设置几个属性

- 损失函数：描述准确率，这里是交叉熵
- 优化器：根据模型看到的数据及其损失函数更新模型的方式，这里是Adam
- 指标：用于监控训练和测试步骤，这里是准确率

```py
model.compile(optimizer=tf.train.AdamOptimizer(),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
```

## 训练模型

8说了，开冲

```py
model.fit(train_images, train_labels, epochs=5)
```

## 评估准确率

看看测试集在模型上的表现

```py
test_loss, test_acc = model.evaluate(test_images, test_labels)
print('Test accuracy:', test_acc)
```
## 学（mo）习（gai）时间

### 改epochs

先试了一下训练20个epochs：
```py
model.fit(train_images, train_labels, epochs=20)
```

效果：
```
...
Epoch 20/20
60000/60000 [==============================] - 2s 30us/step - loss: 0.1791 - acc: 0.9329
10000/10000 [==============================] - 0s 18us/step
Test accuracy: 0.888
```

### 改模型层数

多加一层128个感知器的全连接层
```py
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation=tf.nn.relu),
	keras.layers.Dense(128, activation=tf.nn.relu),
    keras.layers.Dense(10, activation=tf.nn.softmax)
])
```
效果：
```
Epoch 5/5
60000/60000 [==============================] - 2s 36us/step - loss: 0.2869 - acc: 0.8938
10000/10000 [==============================] - 0s 21us/step
Test accuracy: 0.8736
```

### 改感知器个数

把中间层改成512个感知器
```py
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(512, activation=tf.nn.relu),
    keras.layers.Dense(10, activation=tf.nn.softmax)
])
```
效果：
```
Epoch 5/5
60000/60000 [==============================] - 5s 80us/step - loss: 0.2825 - acc: 0.8951
10000/10000 [==============================] - 0s 34us/step
Test accuracy: 0.8762
```

### 总结
- 改epochs：慢，nb
- 改模型层数：快，菜
- 改模型大小：稍慢，还行

就一句话（雾

> 改编不是乱编，戏说不是胡说