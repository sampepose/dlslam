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
image1_val, image2_val, depth_val, flow_val, valid_val, normals_val, normals_from_downsampled_val, r_rel_val, t_rel_val, _, _, mm_val, off_h_val, off_w_val = svkitti_batch_demon(
    '/projects/katefgroup/datasets/svkitti/val.txt', hyp)

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
with tf.Session(config=config) as sess:
    coord = tf.train.Coordinator()
    tf.train.start_queue_runners(coord=coord)

    """ TRAINING """
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
        'weights_dir': './src/experiments/finetune_unsupervised_demon_iterative/logs/1e-4/finetuned-weights-17100',
        'log_dir': os.path.join(CUR_PATH, 'logs/1e-4/'),
        'resume_finetune': True,
        'unsupervised': True,
        # 'restart_global_step': True,
        'learning_rate': 1.0e-4,
        'max_iterations': 30000,
    }

    # Train the network
    finetune_iterative(sess, val, config)
