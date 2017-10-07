import os
import math
import sys
CUR_PATH = os.path.dirname(os.path.realpath(__file__))

from ...inputs.batcher import svkitti_batch_demon
from ...hyperparams import create_hyperparams
from ...demon_wrapper import demon_bir_forward

import imageio
from scipy.misc import imsave
from scipy.io import savemat
import tensorflow as tf
import numpy as np

slim = tf.contrib.slim

hyp = create_hyperparams()
# we need to invert depth since that's what demon predicts
hyp.depth_encoding = 'inv'
hyp.bs = 1

mode = 'val'
if mode == 'val':
    count = 400
else:
    count = 228

# Get the batch of data
image1, image2, depth1, depth2, flow_val, _, valid_val, normals_val, normals_from_downsampled_val, r_rel_val, t_rel_val, p1, p2, mm_val, off_h_val, off_w_val = svkitti_batch_demon(
    '/projects/katefgroup/datasets/svkitti/one_' + mode + '_sequence.txt', hyp, shuffle=False, center_crop=True)

gt_images = []
gt_depth = []
predicted_depth = []

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
with tf.Session(config=config) as sess:
    coord = tf.train.Coordinator()
    tf.train.start_queue_runners(coord=coord)

    # Collect gt images and poses
    for i in range(count):
        # Collect gt
        image1_, image2_, d1_, d2_ = sess.run([image1, image2, depth1, depth2])
        gt_images.append((image1_[0, ...] + 0.5) * 255.)
        gt_images.append((image2_[0, ...] + 0.5) * 255.)
        gt_depth.append(d1_[0, ...])
        gt_depth.append(d2_[0, ...])

tf.reset_default_graph()
with tf.Session(config=config) as sess:
    # Collect predictions
    image_1_placeholder = tf.placeholder(dtype=tf.float32, shape=(1, 192, 256, 3), name='image_1_placeholder')
    image_2_placeholder = tf.placeholder(dtype=tf.float32, shape=(1, 192, 256, 3), name='image_2_placeholder')

    def load_weights(session):
        # Load finetuned bootstrap and iterative weights
        weights_dir = './src/experiments/finetune_demon_iterative/logs/extrinsics/finetuned-weights-8800'
        vars_to_restore = slim.get_variables_to_restore(exclude=["netRefine"])
        pretrained_saver = tf.train.Saver(vars_to_restore)
        pretrained_saver.restore(session, weights_dir)

        # Load refinement weights
        weights_dir = './src/demon/weights/demon_original'
        vars_to_restore = slim.get_variables_to_restore(include=["netRefine"])
        refinement_saver = tf.train.Saver(vars_to_restore)
        refinement_saver.restore(session, weights_dir)

    _, _, refinement_predictions = demon_bir_forward(sess, image_1_placeholder, image_2_placeholder, load_weights, hyp)
    for i in range(len(gt_images) - 1):
        img1 = np.expand_dims(gt_images[i] * 1. / 255 - 0.5, 0)
        img2 = np.expand_dims(gt_images[i + 1] * 1. / 255 - 0.5, 0)

        feed_dict = {
            image_1_placeholder: img1,
            image_2_placeholder: img2,
        }
        refinement_predictions_ = sess.run(refinement_predictions, feed_dict=feed_dict)
        predicted_depth.append(np.transpose(refinement_predictions_['depth'][0, ...], [1, 2, 0]))

# Save images as gif
stacked_gt_depth = np.concatenate([np.array(gt_depth), np.array(gt_depth), np.array(gt_depth)], axis=3)
stacked_pred_depth = np.concatenate([np.array(predicted_depth), np.array(predicted_depth), np.array(predicted_depth)], axis=3)
# normalize to 0...1
stacked_gt_depth = (stacked_gt_depth - stacked_gt_depth.min()) / (stacked_gt_depth.max() - stacked_gt_depth.min())
stacked_pred_depth = (stacked_pred_depth - stacked_pred_depth.min()) / (stacked_pred_depth.max() - stacked_pred_depth.min())
horiz_stack = np.concatenate((np.array(gt_images)[:-1, ...], stacked_gt_depth[:-1, ...] * 255., stacked_pred_depth * 255.), axis=2)
arr = np.split(horiz_stack, horiz_stack.shape[0], axis=0)
arr = [np.squeeze(a) for a in arr]
imageio.mimsave(CUR_PATH + '/' + mode + 'image.gif', arr)

imsave('gtdepth-1.png', stacked_gt_depth[0, ...])
imsave('preddepth-1.png', stacked_pred_depth[0, ...])
imsave('gtimage-1.png', gt_images[0])

imsave('gtdepth-2.png', stacked_gt_depth[10, ...])
imsave('preddepth-2.png', stacked_pred_depth[10, ...])
imsave('gtimage-2.png', gt_images[10])

imsave('gtdepth-3.png', stacked_gt_depth[54, ...])
imsave('preddepth-3.png', stacked_pred_depth[54, ...])
imsave('gtimage-3.png', gt_images[54])


