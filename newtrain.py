# -*- coding: utf-8 -*-
"""
Created on Wed Sep  7  2018
@author: 29970
"""
import tensorflow as tf
import net
import os
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
#定义训练过程中的超参数
batch_size=100
learning_rate_base=0.005
learning_rate_decay=0.99
regularizer=0.0001 #正则化的权重
steps=50000#最大迭代数
moving_average_decay=0.99
model_save_path='./model/'
model_name='mnist_model'
#训练过程
def train(mnist):
    x=tf.placeholder(tf.float32,[batch_size,28,28,1])
    y_=tf.placeholder(tf.float32,[None,10])
    y=net.lenet_forward(x,True,regularizer)
    global_step=tf.Variable(0,trainable=False)#记录当前步数,0表示什么意思,初始化第0步？
    ce=tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y,labels=tf.argmax(y_,1))
    cem=tf.reduce_mean(ce)
    loss=cem+tf.add_n(tf.get_collection('losses'))
    learning_rate=tf.train.exponential_decay(learning_rate_base,global_step,
    mnist.train.num_examples/batch_size,learning_rate_decay,staircase=True)
    train_step=tf.train.GradientDescentOptimizer(learning_rate).minimize(loss,
    global_step=global_step)
    ema=tf.train.ExponentialMovingAverage(moving_average_decay,global_step)
    ema_op=ema.apply(tf.trainable_variables()) #对所有参数进行指数滑动平均取值。
    with tf.control_dependencies([train_step,ema_op]):
         train_op=tf.no_op(name='train')
         '''
         把两个Operation合在一起
         '''
    saver=tf.train.Saver()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        ckpt=tf.train.get_checkpoint_state(model_save_path)
        if  ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess,ckpt.model_checkpoint_path)
            '''
            ckpt的作用是为了生成一个端点，保证这个项目不会被突然中断，get_checkpoint_state这个函数
            是去找该文件下是否有断点，ckpt.model_checkpoint_path表示模型储存的位置，会自己去寻找，然后会将之前训练的参数重载
            '''
        for i in range(steps):
            xs,ys=mnist.train.next_batch(batch_size)
            reshaped_xs=np.reshape(xs,[batch_size,28,28,1])
            #开始训练
            _,loss_value,step=sess.run([train_op,loss,global_step],feed_dict={x:reshaped_xs,y_:ys})
            if i%100==0:
                print('after',step,'steps',',loss on training  is',loss_value)
                saver.save(sess,os.path.join(model_save_path,model_name),global_step=global_step)
                '''
            这里的saver是为了每隔多少步保存一下当前的训练好的数据
                '''
def main():
    mnist=input_data.read_data_sets('./data/',one_hot=True)
    train(mnist)
if __name__=='__main__':#要用双下划线
    main()
    
    
    
    
    
    
    
    
    
    
    
    