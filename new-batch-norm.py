#!/usr/bin/env python2
# -*- coding: UTF-8 -*-
# File: batch_norm.py
# Author: Yuxin Wu <ppwwyyxx@gmail.com>

import tensorflow as tf
from copy import copy

from ._common import layer_register

__all__ = ['BatchNorm']


# http://stackoverflow.com/questions/33949786/how-could-i-use-batch-normalization-in-tensorflow
# TF batch_norm only works for 4D tensor right now: #804
# decay: 0.999 not good for resnet, torch use 0.9 by default
# eps: torch: 1e-5. Lasagne: 1e-4
@layer_register()
def BatchNorm(x, use_local_stat=True, decay=0.9, epsilon=1e-5):
    """
    Batch normalization layer as described in:
    Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift
    http://arxiv.org/abs/1502.03167
    Notes:
    Whole-population mean/variance is calculated by a running-average mean/variance, with decay rate 0.999
    Epsilon for variance is set to 1e-5, as is torch/nn: https://github.com/torch/nn/blob/master/BatchNormalization.lua

    x: BCHW or BC tensor
    use_local_stat: bool. whether to use mean/var of this batch or the running
    average. Usually set to True in training and False in testing
    """

    shape = x.get_shape().as_list()
    n_out = shape[1]  # channel
    assert len(shape) in [2, 4]
    if len(shape) == 2:
        batch_mean, batch_var = tf.nn.moments(x, [0], name='moments',
                                              keep_dims=True)
        batch_mean.set_shape([1, n_out])
        batch_var.set_shape([1, n_out])
    else:
        batch_mean, batch_var = tf.nn.moments(x, [0, 2, 3], name='moments',
                                              keep_dims=True)
        batch_mean.set_shape([1, n_out, 1, 1])
        batch_var.set_shape([1, n_out, 1, 1])
    param_shape = batch_mean.get_shape()
    beta = tf.get_variable('beta', param_shape)
    gamma = tf.get_variable(
        'gamma', param_shape,
        initializer=tf.constant_initializer(1.0))


    ema = tf.train.ExponentialMovingAverage(decay=decay)
    ema_apply_op = ema.apply([batch_mean, batch_var])
    ema_mean, ema_var = ema.average(batch_mean), ema.average(batch_var)

    if use_local_stat:
        with tf.control_dependencies([ema_apply_op]):
            return tf.nn.batch_normalization(
                x, batch_mean, batch_var, beta, gamma, epsilon, 'bn')
    else:
        batch = tf.cast(tf.shape(x)[0], tf.float32)
        mean, var = ema_mean, ema_var * batch / (batch - 1) # unbiased variance estimator
        return tf.nn.batch_normalization(
            x, mean, var, beta, gamma, epsilon, 'bn')
