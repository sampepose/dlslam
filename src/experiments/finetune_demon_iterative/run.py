import os
import sys
CUR_PATH = os.path.dirname(os.path.realpath(__file__))

from ...inputs.batcher import svkitti_batch_demon
from ...hyperparams import create_hyperparams
from ...demon_train import finetune_iterative

import tensorflow as tf

hyp = create_hyperparams()
# we need to invert depth since that's what demon predicts
hyp.depth_encoding = 'inv'
hyp.bs = 8

# Get the batch of data
image1_train, image2_train, depth_train, flow_train, valid_train, normals_train, normals_from_downsampled_train, r_rel_train, t_rel_train, mm_train, off_h_train, off_w_train = svkitti_batch_demon(
    '/projects/katefgroup/datasets/svkitti/train.txt', hyp)
image1_val, image2_val, depth_val, flow_val, valid_val, normals_val, normals_from_downsampled_val, r_rel_val, t_rel_val, mm_val, off_h_val, off_w_val = svkitti_batch_demon(
    '/projects/katefgroup/datasets/svkitti/val.txt', hyp)

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
with tf.Session(config=config) as sess:
    coord = tf.train.Coordinator()
    tf.train.start_queue_runners(coord=coord)

    """ TRAINING """
    gt = {
        'image1': image1_train,
        'image2': image2_train,
        'depth': depth_train,
        'normals': normals_from_downsampled_train,
        'flow': flow_train,
        'rotation': r_rel_train,
        'translation': t_rel_train,
        'motion_mask': mm_train,
    }
    val = {
        'image1': image1_val,
        'image2': image2_val,
        'depth': depth_val,
        'normals': normals_from_downsampled_val,
        'flow': flow_val,
        'rotation': r_rel_val,
        'translation': t_rel_val,
    }
    config = {
        'weights_dir': './src/demon/weights/demon_original',
        'bootstrap_weights': './src/experiments/finetune_demon_bootstrap/logs/extrinsicfinetuned-weights-11000',
        'log_dir': os.path.join(CUR_PATH, 'logs/extrinsics/'),
        # 'plot_zero_motion': False,
    }

    # Train the network
    finetune_iterative(sess, gt, config, val=val)
