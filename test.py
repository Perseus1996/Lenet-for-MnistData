# -*- coding: utf-8 -*-
"""
Created on Wed Sep  7
@author: 29970
"""
import time
import tensorflow as tf
import data
import readdata
import net
import newtrain
import numpy as np
test_interval_secs=5
def test(mnist):
    with tf.Graph().as_default() as g:
        '''
        创造一个相当于新的计算图进而分配新的计算空间？
        '''
        x=tf.placeholder(tf.float32,[mnist.tes.num_examples,
        28,28,1])
        y_=tf.placeholder(tf.float32,[None,10])
        y=net.lenet_forward(x,False,None)
        
        ema=tf.train.ExponentialMovingAverage(newtrain.learning_rate_decay)
        ema_restore=ema.variables_to_restore()
        saver=tf.train.Saver(ema_restore)
        '''
        这里为什么要重载ema
        '''
        correct_prediction=tf.equal(tf.argmax(y,1),tf.argmax(y_,1))
        accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
        '''
        tf.cast将bool型转换为tf.float
        '''
        while True:
            with tf.Session() as sess :
                ckpt=tf.train.get_checkpoint_state(newtrain.model_save_path)
                if ckpt and ckpt.model_checkpoint_path:
                    saver.restore(sess,ckpt.model_checkpoint_path)
                    '''
                    参数重载saver.restore,将模型重载到这个会话当中
                    '''
                    global_step=ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
                    reshaped_x=np.reshape(mnist.test.images,[mnist.test.num_examples,28,28,1])
                    accuracy_score=sess.run(accuracy,feed_dict={x:reshaped_x,y_:mnist.test.lables})
                    print('after',global_step,'steps',',accuracy  is',accuracy_score)
                else:
                    print('No checkpoint found')
                    return
def main():
            mnist=input_data.read_data_sets('.data',one_hot=True)
            test(mnist)
if __name__='__main__':
            main()
                   
                     
                    
                    
                    
                    
                    
                    
                    
                