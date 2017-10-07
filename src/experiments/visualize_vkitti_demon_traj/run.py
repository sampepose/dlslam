import os
import math
import sys
CUR_PATH = os.path.dirname(os.path.realpath(__file__))

from ...inputs.batcher import svkitti_batch_demon
from ...hyperparams import create_hyperparams
from ...demon_wrapper import demon_forward

import imageio
from scipy.misc import imsave
from scipy.io import savemat
import tensorflow as tf
import numpy as np

hyp = create_hyperparams()
# we need to invert depth since that's what demon predicts
hyp.depth_encoding = 'inv'
hyp.bs = 1

mode = 'train'
count = 228
if mode == 'val':
    count = 400

# Get the batch of data
image1, image2, depth_val, _, flow_val, _, valid_val, normals_val, normals_from_downsampled_val, r_rel_val, t_rel_val, p1, p2, mm_val, off_h_val, off_w_val = svkitti_batch_demon(
    '/projects/katefgroup/datasets/svkitti/one_' + mode + '_sequence.txt', hyp, shuffle=False, center_crop=True)

def ominus(a, b):
    return np.linalg.inv(a).dot(b)

def axangle2mat(axis, angle):
    ''' Rotation matrix for rotation angle `angle` around `axis`
    From: http://en.wikipedia.org/wiki/Rotation_matrix#Axis_and_angle
    '''
    x, y, z = axis
    c = math.cos(angle); s = math.sin(angle); C = 1-c
    xs = x*s;   ys = y*s;   zs = z*s
    xC = x*C;   yC = y*C;   zC = z*C
    xyC = x*yC; yzC = y*zC; zxC = z*xC
    return np.array([
            [ x*xC+c,   xyC-zs,   zxC+ys ],
            [ xyC+zs,   y*yC+c,   yzC-xs ],
            [ zxC-ys,   yzC+xs,   z*zC+c ]])

def pose_to_extrinsics(p):
    R = np.matrix.transpose(p[0:3, 0:3])
    t = (-R).dot(p[0:3, 3])
    return R, t

def extrinsics_to_pose(R, t):
    R = np.matrix.transpose(R)
    t = (-R).dot(t)
    return R, t

gt_images = []
gt_poses = []
supervised_pred_pose = []
unsupervised_pred_points = []

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
with tf.Session(config=config) as sess:
    coord = tf.train.Coordinator()
    tf.train.start_queue_runners(coord=coord)

    # Collect gt images and poses
    for i in range(count):
        # Collect gt
        image1_, image2_, p1_, p2_ = sess.run([image1, image2, p1[0, ...], p2[0, ...]])
        gt_images.append((image1_[0, ...] + 0.5) * 255.)
        gt_images.append((image2_[0, ...] + 0.5) * 255.)

        gt_poses.append(p1_)
        gt_poses.append(p2_)

        # a, b, c, d, e, f = sess.run([image1[0, ...], image2[0, ...], flow_val[0, ...], p1[0, ...], p2[0, ...], depth_val[0, ...]])
        # np.save('image1.npy', a)
        # np.save('image2.npy', b)
        # np.save('flow.npy', c)
        # np.save('rel_pose.npy', ominus(e, d))
        # np.save('depth.npy', f)
        # print 'saved'
        # sys.exit()



    # Save gt points as trajectory (use first pt as origin)
    centered_points = []
    for i in range(len(gt_poses) - 1):
        rel = ominus(gt_poses[i + 1], gt_poses[i])
        centered_points.append(rel)

tf.reset_default_graph()
with tf.Session(config=config) as sess:
    # Collect predictions
    image_1_placeholder = tf.placeholder(dtype=tf.float32, shape=(1, 192, 256, 3), name='image_1_placeholder')
    image_2_placeholder = tf.placeholder(dtype=tf.float32, shape=(1, 192, 256, 3), name='image_2_placeholder')
    weights_dir = './src/experiments/finetune_demon_iterative/logs/extrinsics/finetuned-weights-8800'
    bootstrap_predictions, iterative_predictions = demon_forward(sess, image_1_placeholder, image_2_placeholder, weights_dir, hyp)
    for i in range(len(gt_images) - 1):
        img1 = np.expand_dims(gt_images[i] * 1. / 255 - 0.5, 0)
        img2 = np.expand_dims(gt_images[i + 1] * 1. / 255 - 0.5, 0)

        feed_dict = {
            image_1_placeholder: img1,
            image_2_placeholder: img2,
        }
        iterative_predictions_ = sess.run(bootstrap_predictions, feed_dict=feed_dict)

        # rot angle-axis -> rot mat
        pred_r = iterative_predictions_['rotation'][0, ...]
        angle = np.linalg.norm(pred_r)
        pred_r_mat = axangle2mat(pred_r / angle, angle)

        # extrinsics -> pose
        R = np.matrix.transpose(pred_r_mat)
        t = np.dot(-R, iterative_predictions_['translation'][0, ...])

        # relative to absolute
        # R = np.matrix.transpose(pred_r_mat.dot(supervised_pred_pose[-1][0:3, 0:3]))
        # t = -R.dot(t) + supervised_pred_pose[-1][0:3, 3]

        # to pose mat
        P = np.eye(4)
        P[0:3, 0:3] = R
        P[0:3, 3] = t

        # if i == 0:
            # P = ominus(P, np.eye(4))

        # prediction is relative to last predicted point
        supervised_pred_pose.append(P)

tf.reset_default_graph()
with tf.Session(config=config) as sess:
    # Collect predictions
    image_1_placeholder = tf.placeholder(dtype=tf.float32, shape=(1, 192, 256, 3), name='image_1_placeholder')
    image_2_placeholder = tf.placeholder(dtype=tf.float32, shape=(1, 192, 256, 3), name='image_2_placeholder')
    weights_dir = './src/experiments/finetune_unsupervised_demon_iterative/logs/1e-4/finetuned-weights-19000'
    _, iterative_predictions = demon_forward(sess, image_1_placeholder, image_2_placeholder, weights_dir, hyp)
    for i in range(len(gt_images) - 1):
        img1 = np.expand_dims(gt_images[i] * 1. / 255 - 0.5, 0)
        img2 = np.expand_dims(gt_images[i + 1] * 1. / 255 - 0.5, 0)

        feed_dict = {
            image_1_placeholder: img1,
            image_2_placeholder: img2,
        }
        iterative_predictions_ = sess.run(iterative_predictions, feed_dict=feed_dict)

        # rot angle-axis -> rot mat
        pred_r = iterative_predictions_['rotation'][0, ...]
        angle = np.linalg.norm(pred_r)
        pred_r_mat = axangle2mat(pred_r / angle, angle)

        # extrinsics -> pose
        R = np.matrix.transpose(pred_r_mat)
        t = np.dot(-R, iterative_predictions_['translation'][0, ...])

        # relative to absolute
        # R = np.matrix.transpose(pred_r_mat.dot(unsupervised_pred_points[-1][0:3, 0:3]))
        # t = -R.dot(t) + unsupervised_pred_points[-1][0:3, 3]

        # to pose mat
        P = np.eye(4)
        P[0:3, 0:3] = R
        P[0:3, 3] = t

        # prediction is relative to last predicted point
        unsupervised_pred_points.append(P)

# Save points as trajectory
savemat(CUR_PATH + '/' + mode + '-points.mat', {'gt': centered_points, 'sup_pred': supervised_pred_pose, 'unsup_pred': unsupervised_pred_points})

# Save images as gif
imageio.mimsave(CUR_PATH + '/' + mode + 'image.gif', gt_images)
