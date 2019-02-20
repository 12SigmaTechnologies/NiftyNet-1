# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function

from niftynet.layer.convolution import ConvolutionalLayer
from niftynet.network.base_net import BaseNet
import tensorflow as tf

class MaligNet(BaseNet):
    def __init__(self,
                 num_classes,
                 w_initializer=None,
                 w_regularizer=None,
                 b_initializer=None,
                 b_regularizer=None,
                 acti_func='prelu',
                 name='MaligNet'):

        super(MaligNet, self).__init__(
            num_classes=num_classes,
            w_initializer=w_initializer,
            w_regularizer=w_regularizer,
            b_initializer=b_initializer,
            b_regularizer=b_regularizer,
            acti_func=acti_func,
            name=name)

        self.hidden_features = [64, 96, 128]
        self.num_classes = num_classes
        self.initializers = {'w': w_initializer, 'b': b_initializer}
        self.regularizers = {'w': w_regularizer, 'b': b_regularizer}





    def layer_op(self, images, is_training=True, **unused_kwargs):
        conv_1a = ConvolutionalLayer(self.hidden_features[0],
                                         kernel_size=3,
                                         w_initializer=self.initializers['w'],
                                         w_regularizer=self.regularizers['w'],
                                         b_initializer=self.initializers['b'],
                                         b_regularizer=self.regularizers['b'],
                                         acti_func='relu',
                                         name='conv1a')
        flow = conv_1a(images, is_training) #name='conv1a'

        print("conv1a shape = ", flow.shape)

        conv_1b = ConvolutionalLayer(self.hidden_features[0],
                                         kernel_size=3,
                                         w_initializer=self.initializers['w'],
                                         w_regularizer=self.regularizers['w'],
                                         b_initializer=self.initializers['b'],
                                         b_regularizer=self.regularizers['b'],
                                         acti_func='relu',
                                         name='conv1b')

        flow = conv_1b(flow, is_training) #name='conv1b'

        print("conv1b shape = ", flow.shape)

        conv_1c = ConvolutionalLayer(self.hidden_features[0],
                                     kernel_size=3,
                                     stride=2,
                                     w_initializer=self.initializers['w'],
                                     w_regularizer=self.regularizers['w'],
                                     b_initializer=self.initializers['b'],
                                     b_regularizer=self.regularizers['b'],
                                     acti_func='relu',
                                     name='conv1c')

        flow = conv_1c(flow, is_training) #name='conv1c'

        print("conv1c shape = ", flow.shape)

        conv_2a = ConvolutionalLayer(self.hidden_features[0],
                                     kernel_size=3,
                                     w_initializer=self.initializers['w'],
                                     w_regularizer=self.regularizers['w'],
                                     b_initializer=self.initializers['b'],
                                     b_regularizer=self.regularizers['b'],
                                     acti_func='relu',
                                     name='conv2a')

        flow = conv_2a(flow, is_training) #name='conv2a'

        print("conv2a shape = ", flow.shape)

        conv_2b = ConvolutionalLayer(self.hidden_features[0],
                                     kernel_size=3,
                                     w_initializer=self.initializers['w'],
                                     w_regularizer=self.regularizers['w'],
                                     b_initializer=self.initializers['b'],
                                     b_regularizer=self.regularizers['b'],
                                     acti_func='relu',
                                     name='conv2b')

        flow = conv_2b(flow, is_training) #name='conv2b'

        print("conv2b shape = ", flow.shape)

        conv_2c = ConvolutionalLayer(self.hidden_features[0],
                                     kernel_size=3,
                                     stride=2,
                                     w_initializer=self.initializers['w'],
                                     w_regularizer=self.regularizers['w'],
                                     b_initializer=self.initializers['b'],
                                     b_regularizer=self.regularizers['b'],
                                     acti_func='relu',
                                     name='conv2c')

        flow = conv_2c(flow, is_training) #name='conv2c'
        print("conv2c shape = ", flow.shape)

        conv_3a = ConvolutionalLayer(self.hidden_features[1],
                                     kernel_size=2,
                                     stride=2,
                                     w_initializer=self.initializers['w'],
                                     w_regularizer=self.regularizers['w'],
                                     b_initializer=self.initializers['b'],
                                     b_regularizer=self.regularizers['b'],
                                     acti_func='relu',
                                     name='conv3a')

        flow = conv_3a(flow, is_training) #name='conv3a'
        print("conv3a shape = ", flow.shape)

        conv_3b = ConvolutionalLayer(self.hidden_features[1],
                                     kernel_size=2,
                                     stride=2,
                                     padding='VALID',
                                     w_initializer=self.initializers['w'],
                                     w_regularizer=self.regularizers['w'],
                                     b_initializer=self.initializers['b'],
                                     b_regularizer=self.regularizers['b'],
                                     acti_func='relu',
                                     name='conv3b')

        flow = conv_3b(flow, is_training)
                      #name='conv3b'
        print("conv3b shape = ", flow.shape)


        conv_3c = ConvolutionalLayer(self.hidden_features[1],
                                     kernel_size=2,
                                     stride=2,
                                     padding='VALID',
                                     w_initializer=self.initializers['w'],
                                     w_regularizer=self.regularizers['w'],
                                     b_initializer=self.initializers['b'],
                                     b_regularizer=self.regularizers['b'],
                                     acti_func='relu',
                                     name='conv1b')

        flow = conv_3c(flow, is_training)
                      #name='conv3c'
        print("conv3c shape = ", flow.shape)

        conv_4a = ConvolutionalLayer(self.hidden_features[2],
                                     kernel_size=3,
                                     w_initializer=self.initializers['w'],
                                     w_regularizer=self.regularizers['w'],
                                     b_initializer=self.initializers['b'],
                                     b_regularizer=self.regularizers['b'],
                                     acti_func='relu',
                                     name='conv4a')
        flow = conv_4a(flow, is_training)
                      #name='conv4a'
        print("conv4a shape = ", flow.shape)

        flow = tf.layers.dropout(flow, 0.5, name='drop4a')

        conv_4b = ConvolutionalLayer(self.hidden_features[2],
                                     kernel_size=2,
                                     w_initializer=self.initializers['w'],
                                     w_regularizer=self.regularizers['w'],
                                     b_initializer=self.initializers['b'],
                                     b_regularizer=self.regularizers['b'],
                                     acti_func='relu',
                                     name='conv4b')
        flow = conv_4b(flow, is_training)
                      #name='conv4b'
        print("conv4b shape = ", flow.shape)

        flow = tf.layers.dropout(flow, 0.5, name='drop4b')
        conv_4c = ConvolutionalLayer(self.hidden_features[0],
                                     kernel_size=1,
                                     w_initializer=self.initializers['w'],
                                     w_regularizer=self.regularizers['w'],
                                     b_initializer=self.initializers['b'],
                                     b_regularizer=self.regularizers['b'],
                                     acti_func='relu',
                                     name='conv4c')
        flow = conv_4c(flow, is_training)  #name='conv4c'
        print("conv4c shape = ", flow.shape)

        flow = tf.layers.dropout(flow, 0.5, name='drop4c')
        conv_4d = ConvolutionalLayer(self.num_classes,
                                         kernel_size=1,
                                         w_initializer=self.initializers['w'],
                                         w_regularizer=self.regularizers['w'],
                                         b_initializer=self.initializers['b'],
                                         b_regularizer=self.regularizers['b'],
                                         acti_func=None,
                                         name='conv_output')
        flow = conv_4d(flow, is_training)  #name='conv4d'
        print("conv4d shape = ", flow.shape)
        return flow
