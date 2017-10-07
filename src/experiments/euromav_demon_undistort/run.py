"""
python3 src/experiments/euromav_demon_undistort/run.py
"""

import math
import sys
sys.path.append('../../..')
import os
CUR_PATH = os.path.dirname(os.path.realpath(__file__))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import scipy.io
import scipy.misc
import tensorflow as tf

from src.inputs import batcher as bat
from src.hyperparams import create_hyperparams
from src.demon_wrapper import Demon
from src.losses import rtLoss
from src.extras import ominus

import cv2


def unit_vector(data, axis=None, out=None):
    """Return ndarray normalized by length, i.e. Euclidean norm, along axis.

    >>> v0 = numpy.random.random(3)
    >>> v1 = unit_vector(v0)
    >>> numpy.allclose(v1, v0 / numpy.linalg.norm(v0))
    True
    >>> v0 = numpy.random.rand(5, 4, 3)
    >>> v1 = unit_vector(v0, axis=-1)
    >>> v2 = v0 / numpy.expand_dims(numpy.sqrt(numpy.sum(v0*v0, axis=2)), 2)
    >>> numpy.allclose(v1, v2)
    True
    >>> v1 = unit_vector(v0, axis=1)
    >>> v2 = v0 / numpy.expand_dims(numpy.sqrt(numpy.sum(v0*v0, axis=1)), 1)
    >>> numpy.allclose(v1, v2)
    True
    >>> v1 = numpy.empty((5, 4, 3))
    >>> unit_vector(v0, axis=1, out=v1)
    >>> numpy.allclose(v1, v2)
    True
    >>> list(unit_vector([]))
    []
    >>> list(unit_vector([1]))
    [1.0]

    """
    if out is None:
        data = np.array(data, dtype=np.float64, copy=True)
        if data.ndim == 1:
            data /= math.sqrt(np.dot(data, data))
            return data
    else:
        if out is not data:
            out[:] = np.array(data, copy=False)
        data = out
    length = np.atleast_1d(np.sum(data * data, axis))
    np.sqrt(length, length)
    if axis is not None:
        length = np.expand_dims(length, axis)
    data /= length
    if out is None:
        return data


def rotation_matrix(angle, direction, point=None):
    """Return matrix to rotate about axis defined by point and direction.

    >>> R = rotation_matrix(math.pi/2, [0, 0, 1], [1, 0, 0])
    >>> numpy.allclose(numpy.dot(R, [0, 0, 0, 1]), [1, -1, 0, 1])
    True
    >>> angle = (random.random() - 0.5) * (2*math.pi)
    >>> direc = numpy.random.random(3) - 0.5
    >>> point = numpy.random.random(3) - 0.5
    >>> R0 = rotation_matrix(angle, direc, point)
    >>> R1 = rotation_matrix(angle-2*math.pi, direc, point)
    >>> is_same_transform(R0, R1)
    True
    >>> R0 = rotation_matrix(angle, direc, point)
    >>> R1 = rotation_matrix(-angle, -direc, point)
    >>> is_same_transform(R0, R1)
    True
    >>> I = numpy.identity(4, numpy.float64)
    >>> numpy.allclose(I, rotation_matrix(math.pi*2, direc))
    True
    >>> numpy.allclose(2, numpy.trace(rotation_matrix(math.pi/2,
    ...                                               direc, point)))
    True

    """
    sina = math.sin(angle)
    cosa = math.cos(angle)
    direction = unit_vector(direction[:3])
    # rotation matrix around unit vector
    R = np.diag([cosa, cosa, cosa])
    R += np.outer(direction, direction) * (1.0 - cosa)
    direction *= sina
    R += np.array([[0.0,         -direction[2],  direction[1]],
                   [direction[2], 0.0,          -direction[0]],
                   [-direction[1], direction[0],  0.0]])
    M = np.identity(4)
    M[:3, :3] = R
    if point is not None:
        # rotation not around origin
        point = np.array(point[:3], dtype=np.float64, copy=False)
        M[:3, 3] = point - np.dot(R, point)
    return M


hyp = create_hyperparams()
hyp.bs = 1  # only give us 2 image pairs at time t and t + 1 per batch
hyp.dataset_location = './records/'

num_tfrecords = 50  # let's just see how we do for the first 50 image pairs

# Load the validation dataset
# i_v1: image at t
# i_v2: image at t + 1
# p_v1: camera pose at t
# p_v2: camera pose at t + 1
# NOTE WE DO NOT CROP THESE YET, WE NEED TO UNDISTORT THEM FIRST!!
(i_v1, i_v2, _, _, p_v1, p_v2) = bat.euromav_batch(
    hyp.dataset_location + 'records.txt', hyp, shuffle=False, crop=False)

# Variables to hold the predicted and gt camera pose over time
pose_gt = []
rel_transform_gt = []
rel_transform_pred = []

init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())


def undistort(img):
    img = img[0, :, :, :]

    old_w = 752
    old_h = 480
    new_w = 374
    new_h = 240
    intrinsic_mat = np.array([
        [458.654, 0.0, 367.215],
        [0.0, 457.296, 248.375],
        [0.0, 0.0, 1.0],
    ])
    distortion_coeffs = np.array([-0.28340811, 0.07395907, 0.00019359, 1.76187114e-05])
    undistorted = cv2.undistort(img, intrinsic_mat, distortion_coeffs)

    # center crop so intrinsics match demon
    top_left_x = old_w // 2 - new_w // 2
    top_left_y = old_h // 2 - new_h // 2

    img = img[top_left_y:top_left_y + new_h, top_left_x:top_left_x + new_w]
    img = scipy.misc.imresize(img, [192, 256])
    img = np.expand_dims(img, 0)
    return img.astype(np.float32, copy=False) * 1. / 255 - 0.5


with tf.Session() as sess:
    # setup
    sess.run(init_op)
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)
    demon = Demon(sess)

    for i in range(num_tfrecords):
        print('Predicting record: %d' % i)

        # Track gt camera pose matrix (absolute camera motion from origin)
        pose_gt.append(sess.run(p_v1)[0, :, :])

        # Track gt relative transformation matrix (relative camera motion between image pair)
        rel_transform_gt.append(ominus(p_v2, p_v1))

        # Undistort ground truth images
        i_v1_undistorted = tf.py_func(undistort, [i_v1], tf.float32)
        i_v2_undistorted = tf.py_func(undistort, [i_v2], tf.float32)

        # demon forward pass
        predictions = demon.forward(sess, i_v1_undistorted, i_v2_undistorted)

        # Calculate rotation matrix from demon predicted rotation axis
        rot_pred = predictions['iterative2']['predict_rotation'][0, :]
        theta = np.linalg.norm(rot_pred)
        rot_pred /= theta
        pred_transform_mat = rotation_matrix(theta, rot_pred)

        # We need to scale the demon translation and add to the transformation matrix
        scaling = 0.001
        pred_transform_mat[0:3, 3:] = predictions['iterative2']['predict_translation'].T * scaling

        # Track predicted transformation matrix (relative camera motion between image pair)
        rel_transform_pred.append(pred_transform_mat)

        # Save first image pair and depth prediction
        if i == 0 or i == 1 or i == 2 or i == 3 or i == 25 or i == num_tfrecords - 1:
            plt.imsave(CUR_PATH + '/depth_' + str(i) + '.png', predictions['refinement']
                       ['predict_depth0'].squeeze(), cmap='Greys')
            scipy.misc.imsave(CUR_PATH + '/img0_' + str(i) + '_distorted.png',
                              sess.run(i_v1[0, :, :, :]))
            scipy.misc.imsave(CUR_PATH + '/img1_' + str(i) + '_distorted.png',
                              sess.run(i_v2[0, :, :, :]))
            scipy.misc.imsave(CUR_PATH + '/img0_' + str(i) + '_undistorted.png',
                              sess.run(i_v1_undistorted[0, :, :, :]))
            scipy.misc.imsave(CUR_PATH + '/img1_' + str(i) + '_undistorted.png',
                              sess.run(i_v2_undistorted[0, :, :, :]))

    # SAVE GROUND TRUTH TO MAT
    origin = np.array([0., 0., 0., 1.0])
    first_pt = None
    points = []
    for pose in pose_gt:
        pt = np.dot(pose, origin)
        pt /= pt[3]
        if first_pt is None:
            first_pt = pt[:-1]
        pt = pt[:-1] - first_pt
        points.append(pt)

    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    zs = [point[2] for point in points]
    scipy.io.savemat(CUR_PATH + '/pose-gt.mat', dict(x_gt=xs, y_gt=ys, z_gt=zs))

    # SAVE PREDICTIONS TO MAT
    # Use the first point of ground truth as origin
    origin = np.array([xs[0], ys[0], zs[0], 1.0])
    points = []
    prev_transform = None
    first_pt = None
    for rel_transform in rel_transform_pred:
        if prev_transform is not None:
            rel_transform = np.dot(rel_transform, prev_transform)
        prev_transform = rel_transform
        pt = np.dot(rel_transform, origin)
        pt /= pt[3]

        # switch z and y coords
        change_coords = np.array([[-1., 0., 0.], [0., 0., 1.], [0., 1., 0.]])
        changed_pt = np.dot(change_coords, pt[:-1])

        # Subtract first point to center the data
        if first_pt is None:
            first_pt = changed_pt
        changed_pt = changed_pt - first_pt

        points.append(changed_pt)

    x_p = [point[0] for point in points]
    y_p = [point[1] for point in points]
    z_p = [point[2] for point in points]
    scipy.io.savemat(CUR_PATH + '/pose-pred.mat', dict(x_p=x_p, y_p=y_p, z_p=z_p))

    from src.extras import split_rt

    # SAVE ERRORS TO MAT
    rtds_nomotion = []
    rtdas_nomotion = []
    rtds = []
    rtas = []
    translation_norms = []
    for i in range(len(rel_transform_pred)):
        pose_pred = np.expand_dims(rel_transform_pred[i], 0)
        rot_pred_ = tf.constant(pose_pred, dtype=tf.float32)

        rtd, rta, _, _ = rtLoss(rot_pred_, rel_transform_gt[i])
        rtds.append(sess.run(rtd))
        rtas.append(sess.run(rta))

        # Print translation error, norm of translation
        rel_t = rel_transform_gt[i]
        r_, t_ = split_rt(rel_t)
        translation_norms.append(sess.run(tf.norm(t_)))

        pred_r, pred_t = split_rt(rot_pred_)
        print('GT: ', sess.run(t_), 'Predicted:', sess.run(pred_t))

        no_mot_transform = tf.constant(
            [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=tf.float32)
        no_mot_transform = tf.expand_dims(no_mot_transform, 0)
        rtd, rta, _, _ = rtLoss(no_mot_transform, rel_transform_gt[i])
        rtds_nomotion.append(sess.run(rtd))
        rtdas_nomotion.append(sess.run(rta))

    scipy.io.savemat(CUR_PATH + '/errors.mat', dict(rtds=rtds, rtas=rtas,
                                                    rtds_nomot=rtds_nomotion, rtdas_nomot=rtdas_nomotion, trans_norm=translation_norms))

    coord.request_stop()
    coord.join(threads)
