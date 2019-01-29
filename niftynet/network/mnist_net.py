from __future__ import absolute_import, print_function

import tensorflow as tf
import functools

from niftynet.layer.convolution import ConvolutionalLayer as Conv
from niftynet.layer.downsample import DownSampleLayer as Pooling
from niftynet.network.base_net import BaseNet

class mnist_net(BaseNet):
    """
    A mnist network for 2D classification

    """
    
    def __init__(self,
                 num_classes,
                 w_initializer=None,
                 w_regularizer=None,
                 b_initializer=None,
                 b_regularizer=None,
                 acti_func='relu',
                 name='mnist_net'):
        BaseNet.__init__(self,
                         num_classes=num_classes,
                         name=name)

        self.n_fea = [32, 64, 128, 256, 512, 1024]
        net_params = {'padding': 'VALID',
                      'with_bias': True,
                      'with_bn': False,
                      'acti_func': acti_func,
                      'w_initializer': w_initializer,
                      'b_initializer': b_initializer,
                      'w_regularizer': w_regularizer,
                      'b_regularizer': b_regularizer}

        self.conv1_params = {'kernel_size': 5, 'stride': 2}
        self.conv2_params = {'kernel_size': 3, 'stride': 2}

        self.deconv_params = {'kernel_size': 2, 'stride': 2}
        self.pooling_params = {'kernel_size': 2, 'stride': 2}

        self.conv1_params.update(net_params)
        self.conv2_params.update(net_params)
        self.deconv_params.update(net_params)
        #self.nclasses = 10
        self.Conv = functools.partial(Conv,
                                      w_initializer=w_initializer,
                                      w_regularizer=w_regularizer,
                                      b_initializer=b_initializer,
                                      b_regularizer=b_regularizer,
                                      preactivation=True,
                                      acti_func=acti_func)

    def layer_op(self, images, is_training=True, **unused_kwargs):
        '''
        conv1 [-1, 28, 28, 32]
        conv2 [-1, 14, 14, 64]
        flatten
        dropout
        dense

        :param images:
        :param is_training:
        :param unused_kwargs:
        :return:
        '''
        # conv_1 = Conv(self.n_fea[0], **self.conv1_params)(images)
        print("images shape: ", images.shape)
        conv1 = self.Conv(self.n_fea[0], acti_func=None, with_bn=False)
        conv_1 = conv1(images, is_training)

        down_1 = Pooling(func='MAX', **self.pooling_params)(conv_1)
        conv_2 = Conv(self.n_fea[1], **self.conv2_params)(down_1)
        down_2 = Pooling(func='MAX', **self.pooling_params)(conv_2)
        fc1 = tf.contrib.layers.flatten(down_2)
        fc1 = tf.layers.dense(fc1, 1024)
        fc1 = tf.layers.dropout(fc1, rate=0.25, training=False)
        output_tensor = tf.layers.dense(fc1, self.num_classes)
        #output_tensor = tf.contrib.layers.flatten(output_tensor)
        #output_tensor = tf.argmax(output_tensor, axis=1, name="prediction")
        tf.logging.info('output shape %s', output_tensor.shape)
        return output_tensor
