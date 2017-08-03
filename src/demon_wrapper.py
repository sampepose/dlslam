import tensorflow as tf
import numpy as np
import os
from matplotlib import pyplot as plt
from .demon.python.depthmotionnet.networks_original import *


# TODO: Fix this shit to accept TF images
def prepare_input_data(img1, img2, data_format):
    """Creates the arrays used as input from the two images."""
    # scale images if necessary
    if img1.shape[2] != 256 or img1.shape[1] != 192:
        img1 = tf.image.resize_bilinear(img1, [192, 256])
    if img2.shape[2] != 256 or img2.shape[1] != 192:
        img2 = tf.image.resize_bilinear(img2, [192, 256])
    img2_2 = tf.image.resize_bilinear(img2, [48, 64])

    if data_format == 'channels_first':
        img1 = tf.transpose(img1, perm=[0, 3, 1, 2])
        img2 = tf.transpose(img2, perm=[0, 3, 1, 2])
        img2_2 = tf.transpose(img2_2, perm=[0, 3, 1, 2])
        image_pair = tf.concat([img1, img2], 1)
    else:
        image_pair = tf.concat([img1, img2], 3)

    result = {
        'image_pair': image_pair,
        'image1': img1,
        'image2_2': img2_2
    }
    return result


#
# DeMoN has been trained for specific internal camera parameters.
#
# If you use your own images try to adapt the intrinsics by cropping
# to match the following normalized intrinsics:
#
#  K = (0.89115971  0           0.5)
#      (0           1.18821287  0.5)
#      (0           0           1  ),
#  where K(1,1), K(2,2) are the focal lengths for x and y direction.
#  and (K(1,3), K(2,3)) is the principal point.
#  The parameters are normalized such that the image height and width is 1.
#


demon_home = './src/demon/'
weights_dir = demon_home + 'weights'


class Demon:
    def __init__(self, session):
        if tf.test.is_gpu_available(True):
            data_format = 'channels_first'
        else:  # running on cpu requires channels_last data format
            data_format = 'channels_last'

        # init networks
        self.bootstrap_net = BootstrapNet(session, data_format)
        self.iterative_net = IterativeNet(session, data_format)
        self.refine_net = RefinementNet(session, data_format)

        # load weights
        saver = tf.train.Saver()
        saver.restore(session, os.path.join(weights_dir, 'demon_original'))

    def forward(self, session, img1, img2):
        if tf.test.is_gpu_available(True):
            data_format = 'channels_first'
        else:  # running on cpu requires channels_last data format
            data_format = 'channels_last'

        input_data = prepare_input_data(img1, img2, data_format)
        image_pair = session.run(input_data['image_pair'])
        image1 = session.run(input_data['image1'])
        image2_2 = session.run(input_data['image2_2'])

        results = {}

        # run the network
        result = self.bootstrap_net.eval(image_pair, image2_2)
        results['bootstrap'] = result
        for i in range(3):
            result = self.iterative_net.eval(
                image_pair,
                image2_2,
                result['predict_depth2'],
                result['predict_normal2'],
                result['predict_rotation'],
                result['predict_translation']
            )
            results['iterative%d' % i] = result
        rotation = result['predict_rotation']
        translation = result['predict_translation']
        result = self.refine_net.eval(image1, result['predict_depth2'])
        results['refinement'] = result
        return results
