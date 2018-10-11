# -*- coding: utf-8 -*-
"""
Created on Wed Sep  7 2018
@author: 29970
"""
import tensorflow as tf
def lenet_forward(x,train,regularizer):
    #卷积层
    conv1_w=get_weight([5,5,1,32],regularizer)
    conv1_b=get_bias([32])
    conv1=conv2d(x,conv1_w)
    relu1=tf.nn.relu(tf.nn.bias_add(conv1,conv1_b))
    pool1=max_pool(relu1)
    conv2_w=get_weight([5,5,32,64],regularizer)
    conv2_b=get_bias([64])
    conv2=conv2d(pool1,conv2_w)
    relu2=tf.nn.relu(tf.nn.bias_add(conv2,conv2_b))
    pool2=max_pool(relu2)
    #进入全连接层
    #1.将卷积后的结果转化为全连接层的输入  
    pool2_shape=pool2.get_shape().as_list()#将shape数据变为列表方便后面进行调用
    nodes=(pool2_shape[1]*pool2_shape[2])*pool2_shape[3]#拉直后的长度
    reshaped=tf.reshape(pool2,[pool2_shape[0],nodes])#即是输入的batch*特征
    fc1_w=get_weight([nodes,512],regularizer)
    fc1_b=get_bias([512])
    fc1=tf.nn.relu(tf.matmul(reshaped,fc1_w)+fc1_b)
    if train:fc1=tf.nn.dropout(fc1,0.5)#在第一个全连接层有dropout
    fc2_w=get_weight([512,10],regularizer)
    fc2_b=get_bias([10])
    return tf.matmul(fc1,fc2_w)+fc2_b
def get_weight(shape,regularizer):
    w=tf.Variable(tf.truncated_normal(shape,stddev=0.1))
    if regularizer!=None:
        tf.add_to_collection('losses',tf.contrib.layers.l2_regularizer(regularizer)(w))
    '''losses是一个name这样就可以拿去其他文件调用，add_to_collection是把变量加入列表，add_n是把列表内元素相加，
    get_collection将变量从集合中取出变成一个列表
    '''
    return w
def get_bias(shape):
     return tf.zeros(shape,tf.float32)
def conv2d(x,w):
    return tf.nn.conv2d(x,w,strides=[1,1,1,1],padding='SAME')
def max_pool(x):
    return tf.nn.max_pool(x,[1,2,2,1],[1,2,2,1],padding='SAME')
    
    
        

    