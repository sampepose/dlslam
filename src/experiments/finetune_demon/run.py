import os
import sys
CUR_PATH = os.path.dirname(os.path.realpath(__file__))

from ...inputs.batcher import svkitti_batch_demon
from ...hyperparams import create_hyperparams
from ...demon.python.depthmotionnet.train import finetune

import tensorflow as tf

hyp = create_hyperparams()
# we need to invert depth since that's what demon predicts
hyp.depth_encoding = 'inv'
hyp.bs = 8

# Get the batch of data
image1, image2, depth, flow, valid, normals, normals_from_downsampled, r_rel, t_rel, off_h, off_w = svkitti_batch_demon(
    '/projects/katefgroup/datasets/svkitti/val.txt', hyp)

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
with tf.Session(config=config) as sess:
    coord = tf.train.Coordinator()
    tf.train.start_queue_runners(coord=coord)

    learning_rate = 1.0e-9
    max_iterations = 100000
    log_dir = os.path.join(CUR_PATH, 'logs')
    weights_dir = './src/demon/weights'

    # Train the network
    finetune(sess, image1, image2, depth, normals, normals_from_downsampled, flow, r_rel, t_rel,
             learning_rate, max_iterations, weights_dir, log_dir)
