""" euromav_demon_test: Run DeMoN on euromav dataset and visualize the camera trajectory

python3 src/experiments/euromav_demon_rectify/run.py
"""

import math
import os

CUR_PATH = os.path.dirname(os.path.realpath(__file__))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import scipy.misc
import tensorflow as tf

from src.demon_wrapper import Demon
from src.extras import ominus


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


# Load the two images
i_v1 = tf.constant(scipy.misc.imread(CUR_PATH + '/stereo/left_rectified.png'), dtype=tf.float32)
i_v2 = tf.constant(scipy.misc.imread(CUR_PATH + '/stereo/right_rectified.png'), dtype=tf.float32)
p_v1 = tf.constant([[332.38475539,    0.,          367.41543198,   0., ],
                    [0.,          332.38475539,  252.17110062,    0., ],
                    [0.,            0.,            1.,            0., ],
                    [0.,            0.,            0.,            1., ]], dtype=tf.float32)
p_v2 = tf.constant([[332.38475539,    0.,          367.41543198,  -36.58819665, ],
                    [0.,          332.38475539,  252.17110062,    0., ],
                    [0.,            0.,            1.,            0., ],
                    [0.,            0.,            0.,            1., ]], dtype=tf.float32)
# batch inputs
i_v1 = tf.expand_dims(i_v1, 0)
i_v2 = tf.expand_dims(i_v2, 0)
p_v1 = tf.expand_dims(p_v1, 0)
p_v2 = tf.expand_dims(p_v2, 0)

# scale input images
i_v1 = tf.cast(i_v1, tf.float32) * 1. / 255 - 0.5
i_v2 = tf.cast(i_v2, tf.float32) * 1. / 255 - 0.5

init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

with tf.Session() as sess:
    # setup
    sess.run(init_op)
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)
    demon = Demon(sess)

    print(sess.run(tf.reduce_max(i_v1)), sess.run(tf.reduce_min(i_v1)))
    print(sess.run(tf.reduce_max(i_v2)), sess.run(tf.reduce_min(i_v2)))

    # Save our gt rgb for sanity check
    scipy.misc.imsave(CUR_PATH + '/img0_0.png', sess.run(i_v1[0, :, :, :]))
    scipy.misc.imsave(CUR_PATH + '/img1_0.png', sess.run(i_v2[0, :, :, :]))

    # demon forward pass
    predictions = demon.forward(sess, i_v1, i_v2)

    # Calculate rotation matrix from demon predicted rotation axis
    rot_pred = predictions['iterative2']['predict_rotation'][0, :]
    theta = np.linalg.norm(rot_pred)
    rot_pred /= theta
    pred_transform_mat = rotation_matrix(theta, rot_pred)
    # We need to scale the demon translation and add to the transformation matrix
    scaling = 0.1
    pred_transform_mat[0:3, 3:] = predictions['iterative2']['predict_translation'].T * scaling
    print(pred_transform_mat)

    plt.imsave(CUR_PATH + '/depth_pred.png', predictions['refinement']
               ['predict_depth0'].squeeze(), cmap='Greys')

    coord.request_stop()
    coord.join(threads)
