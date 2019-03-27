# -*- coding: utf-8 -*-
# @Time    : 2018/8/15 10:27
# @Author  : Lemon_shark
# @Email   : jiping_cehn@163.com
# @File    : FM.py
# @Software: PyCharm Community Edition

# coding:UTF-8

from __future__ import division
from math import exp
from numpy import *
from random import normalvariate  # 正态分布
from datetime import datetime
import pandas as pd
import numpy as np
import tensorflow as tf

from itertools import count
from collections import defaultdict
from scipy.sparse import csr
import numpy as np
import pandas as pd
import numpy as np
from sklearn.feature_extraction import DictVectorizer
import tensorflow as tf
from tqdm import tqdm_notebook as tqdm


trainData = 'diabetes_train.txt'   #请换为自己文件的路径
testData = 'diabetes_test.txt'

def preprocessData(data):
    feature=np.array(data.iloc[:,:-1])   #取特征
    label=data.iloc[:,-1].map(lambda x: 1 if x==1 else -1) #取标签并转化为 +1，-1
    #将数组按行进行归一化
    zmax, zmin = feature.max(axis=0), feature.min(axis=0)
    feature = (feature - zmin) / (zmax - zmin)
    label=np.array(label)

    return feature,label

def sigmoid(inx):
    #return 1. / (1. + exp(-max(min(inx, 15.), -15.)))
    return 1.0 / (1 + exp(-inx))

def SGD_FM(dataMatrix, classLabels, k, iter):
    '''
    :param dataMatrix:  特征矩阵
    :param classLabels: 类别矩阵
    :param k:           辅助向量的大小
    :param iter:        迭代次数
    :return:
    '''
    # dataMatrix用的是mat, classLabels是列表
    m, n = shape(dataMatrix)   #矩阵的行列数，即样本数和特征数
    alpha = 0.01
    # 初始化参数
    # w = random.randn(n, 1)#其中n是特征的个数
    w = zeros((n, 1))      #一阶特征的系数
    w_0 = 0.
    v = normalvariate(0, 0.2) * ones((n, k))   #即生成辅助向量，用来训练二阶交叉特征的系数

    for it in range(iter):
        for x in range(m):  # 随机优化，每次只使用一个样本
            # 二阶项的计算
            inter_1 = dataMatrix[x] * v
            inter_2 = multiply(dataMatrix[x], dataMatrix[x]) * multiply(v, v)  #二阶交叉项的计算
            interaction = sum(multiply(inter_1, inter_1) - inter_2) / 2.       #二阶交叉项计算完成

            p = w_0 + dataMatrix[x] * w + interaction  # 计算预测的输出，即FM的全部项之和
            loss = 1-sigmoid(classLabels[x] * p[0, 0])    #计算损失

            w_0 = w_0 +alpha * loss * classLabels[x]

            for i in range(n):
                if dataMatrix[x, i] != 0:
                    w[i, 0] = w[i, 0] +alpha * loss * classLabels[x] * dataMatrix[x, i]
                    for j in range(k):
                        v[i, j] = v[i, j]+ alpha * loss * classLabels[x] * (
                        dataMatrix[x, i] * inter_1[0, j] - v[i, j] * dataMatrix[x, i] * dataMatrix[x, i])
        print("第{}次迭代后的损失为{}".format(it, loss))

    return w_0, w, v


def getAccuracy(dataMatrix, classLabels, w_0, w, v):
    m, n = shape(dataMatrix)
    allItem = 0
    error = 0
    result = []
    for x in range(m):   #计算每一个样本的误差
        allItem += 1
        inter_1 = dataMatrix[x] * v
        inter_2 = multiply(dataMatrix[x], dataMatrix[x]) * multiply(v, v)
        interaction = sum(multiply(inter_1, inter_1) - inter_2) / 2.
        p = w_0 + dataMatrix[x] * w + interaction  # 计算预测的输出

        pre = sigmoid(p[0, 0])
        result.append(pre)

        if pre < 0.5 and classLabels[x] == 1.0:
            error += 1
        elif pre >= 0.5 and classLabels[x] == -1.0:
            error += 1
        else:
            continue

    return float(error) / allItem


def batcher(X_, y_=None, batch_size=-1):
    n_samples = X_.shape[0]

    if batch_size == -1:
        batch_size = n_samples
    if batch_size < 1:
       raise ValueError('Parameter batch_size={} is unsupported'.format(batch_size))

    for i in range(0, n_samples, batch_size):
        upper_bound = min(i + batch_size, n_samples)
        ret_x = X_[i:upper_bound]
        ret_y = None
        if y_ is not None:
            ret_y = y_[i:i + batch_size]
            yield (ret_x, ret_y)


train=pd.read_csv(trainData)
test = pd.read_csv(testData)
dataTrain, labelTrain = preprocessData(train)
dataTest, labelTest = preprocessData(test)

x_train = dataTrain.copy()
y_train = labelTrain.copy().astype(np.int32)

x_test = dataTest.copy()
y_test = labelTest.copy().astype(np.int32)

base = np.min(y_train)
y_train = y_train-base
y_test = y_test-base

n,p = x_train.shape

k =4

y_train = [0 if x==0 else 1 for x in y_train]
y_test  = [0 if x==0 else 1 for x in y_test]

y_train = np.array(y_train)
y_test = np.array(y_test)

x = tf.placeholder(tf.float32,shape=(None,p),name='x')
y = tf.placeholder(tf.int32,shape = (None),name='y')


w0 = tf.Variable(tf.zeros([1]))
w = tf.Variable(tf.zeros([p]))

v = tf.Variable(tf.random_normal([k,p],mean=0,stddev=0.01))

#y_hat = tf.Variable(tf.zeros([n,1]))

linear_terms = tf.add(w0,tf.reduce_sum(tf.multiply(w,x),1,keep_dims=True)) # n * 1
pair_interactions = 0.5 * tf.reduce_sum(
    tf.subtract(
        tf.pow(
            tf.matmul(x,tf.transpose(v)),2),
        tf.matmul(tf.pow(x,2),tf.transpose(tf.pow(v,2)))
    ),axis = 1 , keep_dims=True)


y_hat = tf.add(linear_terms,pair_interactions)

y_out_prob = tf.nn.softmax(y_hat)


cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=y_hat)
loss = tf.reduce_mean(cross_entropy)
#
# lambda_w = tf.constant(0.001,name='lambda_w')
# lambda_v = tf.constant(0.001,name='lambda_v')
#
# l2_norm = tf.reduce_sum(
#     tf.add(
#         tf.multiply(lambda_w,tf.pow(w,2)),
#         tf.multiply(lambda_v,tf.pow(v,2))
#     )
# )

# error = tf.reduce_mean(tf.square(y.astype(np.float32)-y_hat))
# loss = tf.add(error,l2_norm)


train_op = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(loss)


epochs = 50
batch_size = 20

# Launch the graph
init = tf.global_variables_initializer()


with tf.Session() as sess:
    sess.run(init)

    for epoch in tqdm(range(epochs), unit='epoch'):
        perm = np.random.permutation(x_train.shape[0])
        # iterate over batches
        for bX, bY in batcher(x_train[perm], y_train[perm], batch_size):
            _,t = sess.run([train_op,loss], feed_dict={x: bX.reshape(-1, p), y: bY})
            # print(t)
            # print(y_hat)
        errors = []
        for bX, bY in batcher(x_test, y_test):
            # errors.append(sess.run(error, feed_dict={x: bX.reshape(-1, p), y: bY.reshape(-1, 1)}))
            _,test_loss = sess.run([train_op,loss],feed_dict={x: bX.reshape(-1, p), y: bY})

            # errors.append(sess.run(error, feed_dict={x: bX.reshape(-1, p), y: bY.reshape(-1, 1)}))
        # print(errors)
        print('train loss:'+str(t)+'test error;'+str(errors))
    # RMSE = np.sqrt(np.array(errors).mean())
    # print (RMSE)




date_startTrain = datetime.now()
print    ("开始训练")
w_0, w, v = SGD_FM(mat(dataTrain), labelTrain, 20, 200)
print(
    "训练准确性为：%f" % (1 - getAccuracy(mat(dataTrain), labelTrain, w_0, w, v)))
date_endTrain = datetime.now()
print(
"训练用时为：%s" % (date_endTrain - date_startTrain))
print("开始测试")
print(
    "测试准确性为：%f" % (1 - getAccuracy(mat(dataTest), labelTest, w_0, w, v)))
