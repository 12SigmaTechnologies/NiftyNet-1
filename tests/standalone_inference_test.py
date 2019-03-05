import tensorflow as tf
import os
from niftynet.io.image_reader import ImageReader
from niftynet.engine.sampler_resize_v2 import ResizeSampler
from niftynet.network.lung_nodule_malig_net import MaligNet
import numpy as np
os.environ["CUDA_VISIBLE_DEVICES"]='1'


##### Address of the model to be restored
check_point_location='/home/niftynet/models/dense_vnet_abdominal_ct/models/model.ckpt-3000'
#####

##### Create a sampler

data_param = {'image': {'path_to_search': '/home/qke/toy_dataset/data',
                        'filename_contains': '20160119001637_9A6E99BD6E244B9692EAF365E2D3B_20160119180739_9_0', 'spatial_window_size': (32, 32, 32)}}

reader = ImageReader().initialise(data_param)

sampler = ResizeSampler(
    reader=reader,
    data_param=data_param,
    batch_size=1,
    shuffle_buffer=True,
    queue_length=35)

#####

with tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)) as sess:

    sampler.run_threads(sess, tf.train.Coordinator(), num_threads=1)

    from niftynet.network.dense_vnet import DenseVNet
    data_dict = sampler.pop_batch_op()
    #net_logits = DenseVNet(num_classes=9)(data_dict['image'])
    net_logits = MaligNet(num_classes=2)(data_dict['image'])

    # restore the variables
    saver = tf.train.Saver()
    saver.restore(sess, check_point_location)

    net_logits = sess.run(net_logits)
    print(net_logits.shape)