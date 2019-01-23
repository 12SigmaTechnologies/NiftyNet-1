# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function

import tensorflow as tf
import numpy as np
from collections import OrderedDict

from niftynet.layer.base_layer import TrainableLayer, Layer
from niftynet.layer.convolution import ConvolutionalLayer as Conv
from niftynet.layer.crop import CropLayer as Crop
from niftynet.layer.deconvolution import DeconvolutionalLayer as DeConv
from niftynet.layer.downsample import DownSampleLayer as Pooling
from niftynet.layer.elementwise import ElementwiseLayer as ElementWise
from niftynet.layer.linear_resize import LinearResizeLayer as Resize
from niftynet.network.base_net import BaseNet


class UNet2D(BaseNet):
    """
    A reimplementation of 2D UNet:
        Ronneberger et al., U-Net: Convolutional Networks for Biomedical
        Image Segmentation, MICCAI '15

    Modified to be fully convolutional
    """

    def __init__(self,
                 num_classes=2,
                 w_initializer=None,
                 w_regularizer=None,
                 b_initializer=None,
                 b_regularizer=None,
                 acti_func='relu',
                 keep_prob=1.,
                 num_channels=1, # gray scale
                 num_layers=3,
                 features_root=16,
                 filter_size=3,
                 pool_size=2,
                 padding='SAME',
                 name='UNet2D'):
        """

        :param num_classes:
        :param w_initializer:
        :param w_regularizer:
        :param b_initializer:
        :param b_regularizer:
        :param acti_func:
        :param keep_prob: dropout probability tensor
        :param num_channels: number of channels in the input image, e.g. grayscale or RGB
        :param num_layers: number of layers in the net
        :param features_root: number of features in the first layer
        :param filter_size: size of the convolution filter
        :param pool_size: size of the max pooling operation
        :param padding: 'VALID' or 'SAME'
        :param name:
        """
        BaseNet.__init__(self,
                         num_classes=num_classes,
                         name=name)
        # self.n_fea = [64, 128, 256, 512, 1024]
        self.num_channels = num_channels
        self.num_layers = num_layers
        self.features_root = features_root
        self.filter_size = filter_size
        self.pool_size = pool_size
        self.keep_prob = keep_prob
        self.padding = padding


        # net_params = {'padding': 'VALID',
        #               'with_bias': True,
        #               'with_bn': False,
        #               'acti_func': acti_func,
        #               'w_initializer': w_initializer,
        #               'b_initializer': b_initializer,
        #               'w_regularizer': w_regularizer,
        #               'b_regularizer': b_regularizer}
        #
        # self.conv_params = {'kernel_size': 3, 'stride': 1}
        # self.deconv_params = {'kernel_size': 2, 'stride': 2}
        # self.pooling_params = {'kernel_size': 2, 'stride': 2}
        #
        # self.conv_params.update(net_params)
        # self.deconv_params.update(net_params)

    def layer_op(self, images, is_training=True, **unused_kwargs):
        """Define structure of unet

        :param images: input tensor, shape [?,nx,ny,channels]
        :param is_training:
        :param unused_kwargs:
        :return:
        """
        tf.logging.info(
            "Layers {layers}, features {features}, filter size {filter_size}x{filter_size}, "
            "pool size: {pool_size}x{pool_size}, padding: {padding}".format(
                layers=self.num_layers,
                features=self.features_root,
                filter_size=self.filter_size,
                pool_size=self.pool_size,
                padding=self.padding))

        # Placeholder for the input image
        in_node = images

        weights = []
        biases = []
        convs = []
        pools = OrderedDict()
        deconv = OrderedDict()
        dw_h_convs = OrderedDict()
        up_h_convs = OrderedDict()

        fs = self.filter_size
        ps = self.pool_size

        # down layers
        for layer in range(0, self.num_layers):
            features = 2 ** layer * self.features_root
            stddev = np.sqrt(2 / (fs ** 2 * features))
            if layer == 0:
                name = 'layer{}_down_w1'.format(layer)
                w1 = weight_variable([fs, fs, self.num_channels, features], stddev, name=name)
            else:
                name = 'layer{}_down_w1'.format(layer)
                w1 = weight_variable([fs, fs, features // 2, features], stddev, name=name)

            name = 'layer{}_down_w2'.format(layer)
            w2 = weight_variable([fs, fs, features, features], stddev, name=name)
            name = 'layer{}_down_b1'.format(layer)
            b1 = bias_variable([features], name=name)
            name = 'layer{}_down_b2'.format(layer)
            b2 = bias_variable([features], name=name)

            conv1 = conv2d(in_node, w1, self.keep_prob, padding=self.padding)
            tmp_h_conv = tf.nn.relu(conv1 + b1)
            conv2 = conv2d(tmp_h_conv, w2, self.keep_prob, padding=self.padding)
            dw_h_convs[layer] = tf.nn.relu(conv2 + b2)

            weights.append((w1, w2))
            biases.append((b1, b2))
            convs.append((conv1, conv2))

            if layer < self.num_layers - 1:
                pools[layer] = max_pool(dw_h_convs[layer], ps, padding=self.padding)
                in_node = pools[layer]

        in_node = dw_h_convs[self.num_layers - 1]

        # up layers
        for layer in range(self.num_layers - 2, -1, -1):
            features = 2 ** (layer + 1) * self.features_root
            stddev = np.sqrt(2 / (fs ** 2 * features))

            name = 'layer{}_up_wd'.format(layer)
            wd = weight_variable_devonc([ps, ps, features // 2, features], stddev, name=name)
            name = 'layer{}_up_bd'.format(layer)
            bd = bias_variable([features // 2], name=name)
            h_deconv = tf.nn.relu(deconv2d(in_node, wd, ps, padding=self.padding) + bd)
            h_deconv_concat = crop_and_concat(dw_h_convs[layer], h_deconv)
            deconv[layer] = h_deconv_concat

            name = 'layer{}_up_w1'.format(layer)
            w1 = weight_variable([fs, fs, features, features // 2], stddev, name=name)
            name = 'layer{}_up_w2'.format(layer)
            w2 = weight_variable([fs, fs, features // 2, features // 2], stddev, name=name)
            name = 'layer{}_up_b1'.format(layer)
            b1 = bias_variable([features // 2], name=name)
            name = 'layer{}_up_b2'.format(layer)
            b2 = bias_variable([features // 2], name=name)

            conv1 = conv2d(h_deconv_concat, w1, self.keep_prob, padding=self.padding)
            h_conv = tf.nn.relu(conv1 + b1)
            conv2 = conv2d(h_conv, w2, self.keep_prob, padding=self.padding)
            in_node = tf.nn.relu(conv2 + b2)
            up_h_convs[layer] = in_node

            weights.append((w1, w2))
            biases.append((b1, b2))
            convs.append((conv1, conv2))

        # Output Map
        name = 'final_w'
        weight = weight_variable([1, 1, self.features_root, self.num_classes], stddev, name=name)
        name = 'final_b'
        bias = bias_variable([self.num_classes], name=name)
        conv = conv2d(in_node, weight, tf.constant(1.0), padding=self.padding)
        output_tensor = tf.nn.relu(conv + bias)

        tf.logging.info('output shape %s', output_tensor.shape)
        return output_tensor


class TwoLayerConv(TrainableLayer):
    """
    Two convolutional layers, number of output channels are ``n_chns`` for both
    of them.

    --conv--conv--
    """

    def __init__(self, n_chns, conv_params):
        TrainableLayer.__init__(self, name='TwoConv')
        self.n_chns = n_chns
        self.conv_params = conv_params

    def layer_op(self, input_tensor):
        output_tensor = Conv(self.n_chns, **self.conv_params)(input_tensor)
        output_tensor = Conv(self.n_chns, **self.conv_params)(output_tensor)
        return output_tensor


class CropConcat(Layer):
    """
    This layer concatenates two input tensors,
    the first one is cropped and resized to match the second one.

    This layer assumes the same amount of differences
    in every spatial dimension in between the two tensors.
    """

    def __init__(self, name='crop_concat'):
        Layer.__init__(self, name=name)

    def layer_op(self, tensor_a, tensor_b):
        """
        match the spatial shape and concatenate the tensors
        tensor_a will be cropped and resized to match tensor_b.

        :param tensor_a:
        :param tensor_b:
        :return: concatenated tensor
        """
        crop_border = (tensor_a.shape[1] - tensor_b.shape[1]) // 2
        tensor_a = Crop(border=crop_border)(tensor_a)
        output_spatial_shape = tensor_b.shape[1:-1]
        tensor_a = Resize(new_size=output_spatial_shape)(tensor_a)
        return ElementWise('CONCAT')(tensor_a, tensor_b)


# TODO: refactor the following ino Layers
def weight_variable(shape, stddev=0.1, name=None):
    initial = tf.truncated_normal(shape, stddev=stddev)
    return tf.get_variable(name, initializer=initial)

def weight_variable_devonc(shape, stddev=0.1, name=None):
    initial = tf.truncated_normal(shape, stddev=stddev)
    return tf.get_variable(name, initializer=initial)

def bias_variable(shape, name=None):
    initial = tf.constant(0.1, shape=shape)
    return tf.get_variable(name, initializer=initial)

def conv2d(x, W,keep_prob_, padding='VALID'):
    conv_2d = tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding=padding)
    return tf.nn.dropout(conv_2d, keep_prob_)

def deconv2d(x, W,stride, padding='VALID'):
    x_shape = tf.shape(x)
    output_shape = tf.stack([x_shape[0], x_shape[1]*2, x_shape[2]*2, x_shape[3]//2])
    return tf.nn.conv2d_transpose(x, W, output_shape, strides=[1, stride, stride, 1], padding=padding)

def max_pool(x,n, padding='VALID'):
    # NB. If max pool is 'SAME' then the upconv'ed image might be larger than the original image
    # TODO: for 'SAME' padding, the input image size should be constrainted.
    return tf.nn.max_pool(x, ksize=[1, n, n, 1], strides=[1, n, n, 1], padding='VALID')

def crop_and_concat(x1,x2):
    x1_shape = tf.shape(x1)
    x2_shape = tf.shape(x2)
    # offsets for the top left corner of the crop
    offsets = [0, (x1_shape[1] - x2_shape[1]) // 2, (x1_shape[2] - x2_shape[2]) // 2, 0]
    size = [-1, x2_shape[1], x2_shape[2], -1]
    x1_crop = tf.slice(x1, offsets, size)
    return tf.concat([x1_crop, x2], 3)
