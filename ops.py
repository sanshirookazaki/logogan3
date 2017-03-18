#!/usr/bin/env python
# -*- coding: utf-8 -*-
import math
import numpy as np 
import tensorflow as tf

from tensorflow.python.framework import ops

from utils import *

def binary_cross_entropy(preds, targets, name=None):
    """Computes binary cross entropy given `preds`.

    For brevity, let `x = `, `z = targets`.  The logistic loss is

        loss(x, z) = - sum_i (x[i] * log(z[i]) + (1 - x[i]) * log(1 - z[i]))

    Args:
        preds: A `Tensor` of type `float32` or `float64`.
        targets: A `Tensor` of the same type and shape as `preds`.
    """
    eps = 1e-12
    with ops.op_scope([preds, targets], name, "bce_loss") as name:
        preds = ops.convert_to_tensor(preds, name="preds")
        targets = ops.convert_to_tensor(targets, name="targets")
        return tf.reduce_mean(-(targets * tf.log(preds + eps) +
                              (1. - targets) * tf.log(1. - preds + eps)))

def conv_cond_concat(x, y):
    """Concatenate conditioning vector on feature map axis."""
    x_shapes = x.get_shape()
    y_shapes = y.get_shape()
    return tf.concat(3, [x, y*tf.ones([x_shapes[0], x_shapes[1], x_shapes[2], y_shapes[3]])])

def conv2d(input_, output_dim, 
           k_h=5, k_w=5, d_h=2, d_w=2, stddev=0.02,
           name="conv2d"):
    with tf.variable_scope(name):
        w = tf.get_variable('w', [k_h, k_w, input_.get_shape()[-1], output_dim],        #[5,5,3,64]
                            initializer=tf.truncated_normal_initializer(stddev=stddev))
        conv = tf.nn.conv2d(input_, w, strides=[1, d_h, d_w, 1], padding='SAME')        #[64,64,64,3],[5,5,3,64],[1,2,2,1]

        biases = tf.get_variable('biases', [output_dim], initializer=tf.constant_initializer(0.0))  #[64]
        conv = tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape()) #conv+bias , conv.shape
        mean, variance = tf.nn.moments(conv, [0, 1, 2]) #add
        conv = tf.nn.batch_normalization(conv, mean, variance, None, None, 1e-5) # add
        #print conv

        return conv

def deconv2d(input_, output_shape,                      # input = h0, out = [64,8,8,64*8] ,name='g_h1', with_w=True
             k_h=5, k_w=5, d_h=2, d_w=2, stddev=0.02,
             name="deconv2d", with_w=False):
    with tf.variable_scope(name):
        # filter : [height, width, output_channels, in_channels]
        w = tf.get_variable('w', [k_h, k_h, output_shape[-1], input_.get_shape()[-1]],
                            initializer=tf.random_normal_initializer(stddev=stddev))
        
        try:
            deconv = tf.nn.conv2d_transpose(input_, w, output_shape=output_shape,  # value , filter , out_shape , stride 
                                strides=[1, d_h, d_w, 1])
            #print deconv
            

        # Support for verisons of TensorFlow before 0.7.0
        except AttributeError:
            deconv = tf.nn.deconv2d(input_, w, output_shape=output_shape,
                                strides=[1, d_h, d_w, 1])

        biases = tf.get_variable('biases', [output_shape[-1]], initializer=tf.constant_initializer(0.0))
        deconv = tf.reshape(tf.nn.bias_add(deconv, biases), deconv.get_shape())
        print deconv
        if with_w:
            return deconv, w, biases
        else:
            return deconv

def lrelu(x, leak=0.2, name="lrelu"):
    with tf.variable_scope(name):
        f1 = 0.5 * (1 + leak) # 0.6
        f2 = 0.5 * (1 - leak) # 0.4
        return f1 * x + f2 * abs(x) # 0.6 * x + 0.4 * |x|  

def linear(input_, output_size, scope=None, stddev=0.02, bias_start=0.0, with_w=False):   # z=[none,100] ,  64 , 'g_h0_lin' , with_w = True
    shape = input_.get_shape().as_list()    # 

    with tf.variable_scope(scope or "Linear"):
        matrix = tf.get_variable("Matrix", [shape[1], output_size], tf.float32,  # get_var => create or reurn 100,64 分散0.02  
                                 tf.random_normal_initializer(stddev=stddev))
        bias = tf.get_variable("bias", [output_size],
            initializer=tf.constant_initializer(bias_start))                    #[64] 0.0 分散
        if with_w:
            return tf.matmul(input_, matrix) + bias, matrix, bias       # in * matrix+bias , matrix , bias
        else:
            return tf.matmul(input_, matrix) + bias                    #  in * matrix , bias
