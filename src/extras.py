import tensorflow as tf
import math
import numpy as np
from tensorflow.python.framework import ops
# import hyperparams as hyp
from PIL import Image
from scipy.misc import imsave
from math import pi
from skimage.draw import *
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
# from flow_transformer import *
EPS = 1e-6

# cool stuff
import sys
import time


def spinning_cursor():
    while True:
        for cursor in '|/-\\':
            yield cursor


spinner = spinning_cursor()

# ref: https://en.wikipedia.org/wiki/Conversion_between_quaternions_and_Euler_angles


def quat_to_euler(q):  # tf array [qw,qx,qy,qz] as the data is in this format
    shape = q.get_shape()
    bs = int(shape[0])
    w = q[:, 0]
    x = q[:, 1]
    y = q[:, 2]
    z = q[:, 3]
    ysqr = y * y
    # roll (x-axis rotation)
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + ysqr)
    roll = tf.reshape(tf.atan(t0 / t1), [bs, 1])  # atan2 to atan to get angles in +/-pi

    # pitch (y-axis rotation)
    t2 = +2.0 * (w * y - z * x)
    t2 = tf.where(tf.greater(t2, 1), tf.zeros_like(t2) + 1, t2)
    t2 = tf.where(tf.less(t2, -1), tf.zeros_like(t2) - 1, t2)
    pitch = tf.reshape(tf.asin(t2), [bs, 1])
    # yaw (z-axis rotation)
    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (ysqr + z * z)
    yaw = tf.reshape(tf.atan(t3 / t4), [bs, 1])  # atan2 to atan to get angles in +/-pi
    return (roll, pitch, yaw)
# ref: https://en.wikipedia.org/wiki/Conversion_between_quaternions_and_Euler_angles


def quat2rotmat(q):  # tf array [qw,qx,qy,qz] as the data is in this format
    shape = q.get_shape()
    bs = int(shape[0])
    q0 = tf.reshape(q[:, 0], [bs, 1])
    q1 = tf.reshape(q[:, 1], [bs, 1])
    q2 = tf.reshape(q[:, 2], [bs, 1])
    q3 = tf.reshape(q[:, 3], [bs, 1])
    # normalization
    z = tf.concat([tf.square(q0), tf.square(q1), tf.square(q2), tf.square(q3)], 1)
    #z=tf.sqrt(tf.reduce_sum(z, 1, keep_dims=True))
    z = tf.reduce_sum(z, 1, keep_dims=True)
    q0 = tf.div(q0, z)
    q1 = tf.div(q1, z)
    q2 = tf.div(q2, z)
    q3 = tf.div(q3, z)

    t1 = tf.reshape((q0 * q0) + (q1 * q1) - (q2 * q2) - (q3 * q3), [bs, 1])
    t2 = tf.reshape(2.0 * (q1 * q2 - q0 * q3), [bs, 1])
    t3 = tf.reshape(2.0 * (q0 * q2 + q1 * q3), [bs, 1])
    R1 = tf.reshape(tf.concat([t1, t2, t3], 1), [bs, 1, 3])
    t1 = tf.reshape(2.0 * (q1 * q2 + q0 * q3), [bs, 1])
    t2 = tf.reshape((q0 * q0) - (q1 * q1) + (q2 * q2) - (q3 * q3), [bs, 1])
    t3 = tf.reshape(2.0 * (q2 * q3 - q0 * q1), [bs, 1])
    R2 = tf.reshape(tf.concat([t1, t2, t3], 1), [bs, 1, 3])
    t1 = tf.reshape(2.0 * (q1 * q3 - q0 * q2), [bs, 1])
    t2 = tf.reshape(2.0 * (q0 * q1 + q2 * q3), [bs, 1])
    t3 = tf.reshape((q0 * q0) - (q1 * q1) - (q2 * q2) + (q3 * q3), [bs, 1])
    R3 = tf.reshape(tf.concat([t1, t2, t3], 1), [bs, 1, 3])
    R = tf.reshape(tf.concat([R1, R2, R3], 1), [bs, 3, 3])
    return R
    # R=tf.array([])


def pose2mat(p):
    shape = p.get_shape()
    bs = int(shape[0])
    q = p[:, 3:7]
    t = tf.reshape(p[:, 0:3], [bs, 3])
    R = quat2rotmat(q)
    T = merge_rt(R, t)
    return T


def stop_execution(t, msg=''):
    def f(t):
        # print msg
        exit()
        return t
    return tf.py_func(f, [t], t.dtype)


def print_shape(t):
    print(t.name, t.get_shape().as_list())


def print_shape2(t, msg=''):
    def f(A):
        # print np.shape(A), msg
        return A
    return tf.py_func(f, [t], t.dtype)


def split_rt(rt):
    shape = rt.get_shape()
    bs = int(shape[0])
    r = tf.slice(rt, [0, 0, 0], [-1, 3, 3])
    t = tf.reshape(tf.slice(rt, [0, 0, 3], [-1, 3, 1]), [bs, 3])
    return r, t


def split_intrinsics(k):
    shape = k.get_shape()
    bs = int(shape[0])
    # fx = tf.slice(k,[0,0,0],[-1,1,1])
    # print_shape(fx)
    # fy = tf.slice(k,[0,1,0],[-1,1,1])
    # print_shape(fy)
    # x0 = tf.slice(k,[0,0,3],[-1,1,1])
    # print_shape(x0)
    # y0 = tf.slice(k,[0,1,3],[-1,1,1])
    # print_shape(y0)

    # fy = tf.reshape(tf.slice(k,[0,0,0],[-1,1,1]),[bs])
    # fx = tf.reshape(tf.slice(k,[0,1,1],[-1,1,1]),[bs])
    # y0 = tf.reshape(tf.slice(k,[0,0,2],[-1,1,1]),[bs])
    # x0 = tf.reshape(tf.slice(k,[0,1,2],[-1,1,1]),[bs])
    fx = tf.reshape(tf.slice(k, [0, 0, 0], [-1, 1, 1]), [bs])
    fy = tf.reshape(tf.slice(k, [0, 1, 1], [-1, 1, 1]), [bs])
    x0 = tf.reshape(tf.slice(k, [0, 0, 2], [-1, 1, 1]), [bs])
    y0 = tf.reshape(tf.slice(k, [0, 1, 2], [-1, 1, 1]), [bs])
    return fx, fy, x0, y0


def merge_rt(r, t):
    shape = r.get_shape()
    bs = int(shape[0])
    bottom_row = tf.tile(tf.reshape(tf.stack([0., 0., 0., 1.]), [1, 1, 4]),
                         [bs, 1, 1], name="bottom_row")
    rt = tf.concat(axis=2, values=[r, tf.expand_dims(t, 2)], name="rt_3x4")
    rt = tf.concat(axis=1, values=[rt, bottom_row], name="rt_4x4")
    return rt


def random_crop(t, crop_h, crop_w, h, w):
    def off_h(): return tf.random_uniform([], minval=0, maxval=(h - crop_h - 1), dtype=tf.int32)

    def off_w(): return tf.random_uniform([], minval=0, maxval=(w - crop_w - 1), dtype=tf.int32)

    def zero(): return tf.constant(0)
    offset_h = tf.cond(tf.less(crop_h, h - 1), off_h, zero)
    offset_w = tf.cond(tf.less(crop_w, w - 1), off_w, zero)
    t_crop = tf.slice(t, [offset_h, offset_w, 0], [crop_h, crop_w, -1], name="cropped_tensor")
    return t_crop, offset_h, offset_w


def near_topleft_crop(t, crop_h, crop_w, h, w, amount):
    # take a random crop/pad somewhere in [-amount,amount]

    def get_rand(): return tf.random_uniform([], minval=0, maxval=amount, dtype=tf.int32)

    def get_zero(): return tf.constant(0)

    # pad a bit
    pad_h = tf.cond(tf.greater(amount, 0), get_rand, get_zero)
    pad_w = tf.cond(tf.greater(amount, 0), get_rand, get_zero)
    # t = tf.pad(t, [[pad_h, pad_h], [pad_w, pad_w], [0, 0]], "SYMMETRIC")
    t = tf.pad(t, [[pad_h, pad_h], [pad_w, pad_w], [0, 0]])
    t = tf.slice(t, [0, 0, 0], [h, w, -1])

    # crop a bit
    offset_h = tf.cond(tf.less(crop_h, h), get_rand, get_zero)
    offset_w = tf.cond(tf.less(crop_w, w), get_rand, get_zero)
    t_crop = tf.slice(t, [offset_h, offset_w, 0], [crop_h, crop_w, -1], name="cropped_tensor")
    return t_crop, offset_h, offset_w


def topleft_crop(t, crop_h, crop_w, h, w):
    offset_h = 0
    offset_w = 0
    t_crop = tf.slice(t, [offset_h, offset_w, 0], [crop_h, crop_w, -1], name="cropped_tensor")
    return t_crop, offset_h, offset_w


def bottomright_crop(t, crop_h, crop_w, h, w):
    offset_h = h - crop_h
    offset_w = w - crop_w
    t_crop = tf.slice(t, [offset_h, offset_w, 0], [crop_h, crop_w, -1], name="cropped_tensor")
    return t_crop, offset_h, offset_w


def bottomleft_crop(t, crop_h, crop_w, h, w):
    offset_h = h - crop_h
    offset_w = 0
    t_crop = tf.slice(t, [offset_h, offset_w, 0], [crop_h, crop_w, -1], name="cropped_tensor")
    return t_crop, offset_h, offset_w


def bottomcenter_crop(t, crop_h, crop_w, h, w):
    offset_h = h - crop_h
    offset_w = (w - crop_w) / 2
    t_crop = tf.slice(t, [offset_h, offset_w, 0], [crop_h, crop_w, -1], name="cropped_tensor")
    return t_crop, offset_h, offset_w


def norm(x):
    # x should be B x H
    # returns size B
    return tf.sqrt(tf.reduce_sum(tf.square(x), axis=1))


def compute_distance(transform):
    """
    Compute the distance of the translational component of a 4x4 homogeneous matrix.
    """
    # return numpy.linalg.norm(transform[0:3,3])

    r, t = split_rt(transform)
    # t should now be bs x 3
    return norm(t)


def compute_angle_3x3(R):
    return tf.acos(tf.minimum(1., tf.maximum(-1., (tf.trace(R) - 1.) / 2.)))


def compute_angle(transform):
    """
    Compute the rotation angle from a 4x4 homogeneous matrix.
    """
    # an invitation to 3-d vision, p 27
    # return numpy.arccos( min(1,max(-1, (numpy.trace(transform[0:3,0:3]) - 1)/2) ))
    r, t = split_rt(transform)
    return compute_angle_3x3(r)
    # return tf.acos(tf.minimum(1.,tf.maximum(-1.,(tf.trace(r)-1.)/2.)))


def compute_t_diff(rt1, rt2):
    """
    Compute the difference between the magnitudes of the translational components of the two transformations.
    """
    t1 = tf.reshape(tf.slice(rt1, [0, 0, 3], [-1, 3, 1]), [-1, 3])
    t2 = tf.reshape(tf.slice(rt2, [0, 0, 3], [-1, 3, 1]), [-1, 3])
    # each t should now be bs x 3
    mag_t1 = tf.sqrt(tf.reduce_sum(tf.square(t1), axis=1))
    mag_t2 = tf.sqrt(tf.reduce_sum(tf.square(t2), axis=1))
    return tf.abs(mag_t1 - mag_t2)


def compute_t_ang(rt1, rt2):
    """
    Compute the angle between the translational components of two transformations.
    """
    t1 = tf.reshape(tf.slice(rt1, [0, 0, 3], [-1, 3, 1]), [-1, 3])
    t2 = tf.reshape(tf.slice(rt2, [0, 0, 3], [-1, 3, 1]), [-1, 3])
    # each t should now be bs x 3
    mag_t1 = tf.sqrt(tf.reduce_sum(tf.square(t1), axis=1))
    mag_t2 = tf.sqrt(tf.reduce_sum(tf.square(t2), axis=1))
    dot = tf.reduce_sum(t1 * t2, axis=1)
    return tf.acos(dot / (mag_t1 * mag_t2 + EPS))


def safe_inverse(a):
    """
    safe inverse for rigid transformations
    should be equivalent to
      a_inv = tf.matrix_inverse(a)
    for well-behaved matrices
    """
    bs = tf.shape(a)[0]
    R, T = split_rt(a)
    R_transpose = tf.transpose(R, [0, 2, 1])
    bottom_row = tf.tile(tf.reshape(tf.stack([0., 0., 0., 1.]), [1, 1, 4]), tf.stack([bs, 1, 1]))
    inv = tf.concat(axis=2, values=[R_transpose, -tf.matmul(R_transpose, tf.expand_dims(T, 2))])
    inv = tf.concat(axis=1, values=[inv, bottom_row])
    return inv


def ominus(a, b):
    """
    Compute the relative 3D transformation between a and b.

    Input:
    a -- first pose (homogeneous 4x4 matrix)
    b -- second pose (homogeneous 4x4 matrix)

    Output:
    Relative 3D transformation from a to b.
    https://github.com/liruihao/tools-for-rgbd-SLAM-evaluation/blob/master/evaluate_rpe.py
    """
    with tf.name_scope("ominus"):
        a_inv = safe_inverse(a)
        return tf.matmul(a_inv, b)


def sincos2r(sina, sinb, sing, cosa, cosb, cosg):
    # i am using Tait-Bryan angles here, so that
    # alph corresponds to x
    # beta corresponds to y
    # gamm corresponds to z
    shape = sina.get_shape()
    one = tf.ones_like(sina, name="one")
    zero = tf.zeros_like(sina, name="zero")
    Rx = tf.reshape(tf.stack([one, zero, zero,
                              zero, cosa, -sina,
                              zero, sina, cosa],
                             axis=1), [-1, 3, 3])
    Ry = tf.reshape(tf.stack([cosb, zero, sinb,
                              zero, one, zero,
                              -sinb, zero, cosb],
                             axis=1), [-1, 3, 3])
    Rz = tf.reshape(tf.stack([cosg, -sing, zero,
                              sing, cosg, zero,
                              zero, zero, one],
                             axis=1), [-1, 3, 3])
    # Rz, Ry, Rx order works with r2abg_v3 and abg2r_v2
    # i like it because of this tutorial
    # http://danceswithcode.net/engineeringnotes/rotations_in_3d/rotations_in_3d_part1.html
    # Rcam = tf.matmul(tf.matmul(Rz,Ry),Rx,name="Rcam")

    # Rx, Ry, Rz order works with r2abg_v2, r2abg, and abg2r
    # i like it because it matches with Matlab
    Rcam = tf.matmul(tf.matmul(Rx, Ry), Rz, name="Rcam")
    return Rcam


def sinabg2r(sina, sinb, sing):
    cosa = tf.sqrt(1 - tf.square(sina) + EPS)
    cosb = tf.sqrt(1 - tf.square(sinb) + EPS)
    cosg = tf.sqrt(1 - tf.square(sing) + EPS)
    return sincos2r(sina, sinb, sing, cosa, cosb, cosg)


def abg2r(a, b, g):
    sina = tf.sin(a)
    sinb = tf.sin(b)
    sing = tf.sin(g)
    cosa = tf.cos(a)
    cosb = tf.cos(b)
    cosg = tf.cos(g)
    return sincos2r(sina, sinb, sing, cosa, cosb, cosg)


def abg2r_v2(a, b, g):
    shape = a.get_shape()
    bs = int(shape[0])
    return tf.reshape(euler2mat(tf.expand_dims(g, 1),
                                tf.expand_dims(b, 1),
                                tf.expand_dims(a, 1)),
                      [bs, 3, 3])


# def r2abg_bs1((r)):
#     r00 = r[0, 0]
#     r10 = r[1, 0]
#     r11 = r[1, 1]
#     r12 = r[1, 2]
#     r20 = r[2, 0]
#     r21 = r[2, 1]
#     r22 = r[2, 2]
#
#     # singular = sy < 1e-6
#     # if  not singular :
#     #     x = math.atan2(R[2,1] , R[2,2])
#     #     y = math.atan2(-R[2,0], sy)
#     #     z = math.atan2(R[1,0], R[0,0])
#     # else :
#     #     x = math.atan2(-R[1,2], R[1,1])
#     #     y = math.atan2(-R[2,0], sy)
#     #     z = 0
#
#     sy = tf.sqrt(r00 * r00 + r10 * r10)
#     x = tf.cond(sy > 1e-6, lambda: tf.atan2(r21, r22), lambda: tf.atan2(-r12, r11))
#     y = tf.cond(sy > 1e-6, lambda: tf.atan2(-r20, sy), lambda: tf.atan2(-r20, sy))
#     z = tf.cond(sy > 1e-6, lambda: tf.atan2(r10, r00), lambda: tf.identity(0.0))
#     xyz = tf.stack([x, y, z], axis=0)
#     return xyz
#
#
# def r2abg_v2_bs1((r)):
#     r00 = r[0, 0]
#     r10 = r[1, 0]
#     r11 = r[1, 1]
#     r12 = r[1, 2]
#     r20 = r[2, 0]
#     r21 = r[2, 1]
#     r22 = r[2, 2]
#
#     x = tf.atan2(r21, r22)
#     y = tf.atan2(-r20, tf.sqrt(r21 * r21 + r22 * r22))
#     z = tf.atan2(r10, r00)
#
#     xyz = tf.stack([x, y, z], axis=0)
#     return xyz
#
#
# def r2abg_v3_bs1((r)):
#     r11 = r[0, 0]
#     r21 = r[1, 0]
#     r22 = r[1, 1]
#     r23 = r[1, 2]
#     r31 = r[2, 0]
#     r32 = r[2, 1]
#     r33 = r[2, 2]
#
#     z = tf.atan2(r21, r11)
#     y = -tf.asin(r31)
#     x = tf.atan2(r32, r33)
#
#     xyz = tf.stack([x, y, z], axis=0)
#     return xyz


def r2abg(r):
    # r is 3x3.
    # get alpha, beta, gamma
    # i copied a nice python function from the internet, copied from matlab
    xyz = tf.map_fn(r2abg_bs1, (r), dtype=tf.float32)
    x, y, z = tf.unstack(xyz, axis=1)
    return x, y, z


def r2abg_v2(r):
    # r is 3x3.
    # get alpha, beta, gamma
    # i copied a nice python function from the internet, copied from matlab
    xyz = tf.map_fn(r2abg_v2_bs1, (r), dtype=tf.float32)
    x, y, z = tf.unstack(xyz, axis=1)
    return x, y, z


def r2abg_v3(r):
    # http://danceswithcode.net/engineeringnotes/rotations_in_3d/rotations_in_3d_part1.html
    # r is 3x3.
    # get alpha, beta, gamma
    xyz = tf.map_fn(r2abg_v3_bs1, (r), dtype=tf.float32)
    x, y, z = tf.unstack(xyz, axis=1)
    return x, y, z


def zrt2flow_helper(Z, rt12, fy, fx, y0, x0):
    r12, t12 = split_rt(rt12)
    # if hyp.dataset_name == 'KITTI' or hyp.dataset_name=='KITTI2':
    #     flow = zrt2flow_kitti(Z, r12, t12, fy, fx, y0, x0)
    # else:
    flow, XYZ2 = zrt2flow(Z, r12, t12, fy, fx, y0, x0)
    return flow, XYZ2


def zrt2flow_kitti(Z, R, T, fy, fx, y0, x0):
    shape = Z.get_shape()
    bs = int(shape[0])

    def ed(x): return tf.expand_dims(x, axis=0)

    def upk(x): return tf.unstack(x, axis=0)

    def upked(x): return map(ed, upk(x))
    Zu = upked(Z)
    Ru = upked(R)
    Tu = upked(T)
    fxu = upk(fx)
    fyu = upk(fy)
    x0u = upked(x0)
    y0u = upked(y0)
    result1 = []
    result2 = []
    for i in range(bs):
        Zs = Zu[i]
        Rs = Ru[i]
        Ts = Tu[i]
        fxs = fxu[i]
        fys = fyu[i]
        x0s = x0u[i]
        y0s = y0u[i]
        r1 = zrt2flow(Zs, Rs, Ts, fys, fxs, y0s, x0s)
        result1.append(r1)
        # result2.append(r2)
    flow = tf.concat(axis=0, values=result1)
    # XYZ2 = tf.concat(0, result2)
    return flow


def zrt2flow(Z, R, T, fy, fx, y0, x0):
    with tf.variable_scope("zrt2flow"):
        shape = Z.get_shape()
        bs = int(shape[0])
        h = int(shape[1])
        w = int(shape[2])

        # R is B x 3 x 3
        # T is B x 3. let's make it B x H*W x 3
        T = tf.tile(tf.expand_dims(T, axis=1), [1, h * w, 1])  # B x H*W x 3
        # fy, fx, y0, x0 are B

        # get pointcloud1
        [grid_x1, grid_y1] = meshgrid2D(bs, h, w)
        Z = tf.reshape(Z, [bs, h, w], name="Z")
        XYZ = Camera2World(grid_x1, grid_y1, Z, fx, fy, x0, y0)
        # XYZ is B x H*W x 3

        # transform pointcloud1 using r and t, to estimate pointcloud2
        # first we need to transpose XYZ, so that 3 is the inner dim
        XYZ_t = tf.transpose(XYZ, perm=[0, 2, 1], name="XYZ_t")  # B x 3 x H*W
        # now we can multiply
        XYZ_rot_t = tf.matmul(R, XYZ_t, name="XYZ_rot_t")  # B x 3 x H*W
        # and untranspose.
        XYZ_rot = tf.transpose(XYZ_rot_t, perm=[0, 2, 1], name="XYZ_rot")  # B x H*W x 3
        # add in the T
        XYZ2 = XYZ_rot + T  # B x H*W x 3

        # project pointcloud2 down, so that we get the 2D location of all of these pixels
        [X2, Y2, Z2] = tf.split(axis=2, num_or_size_splits=3, value=XYZ2, name="splitXYZ")
        x2y2_flat = World2Camera(X2, Y2, Z2, fx, fy, x0, y0)
        [x2_flat, y2_flat] = tf.split(axis=2, num_or_size_splits=2,
                                      value=x2y2_flat, name="splitxyz_flat")

        # subtract the new 2D locations from the old ones to get optical flow
        x1_flat = tf.reshape(grid_x1, [bs, -1, 1], name="x1")
        y1_flat = tf.reshape(grid_y1, [bs, -1, 1], name="y1")
        flow_flat = tf.concat(axis=2, values=[x2_flat - x1_flat,
                                              y2_flat - y1_flat], name="flow_flat")
        flow = tf.reshape(flow_flat, [bs, h, w, 2], name="flow")
        return flow, XYZ2


def Camera2World(x, y, Z, fx, fy, x0, y0):
    shape = Z.get_shape()
    bs = int(shape[0])
    h = int(shape[1])
    w = int(shape[2])

    # the intrinsics are shaped [bs]
    # we need them to be shaped [bs,h,w]
    fy = tf.tile(tf.expand_dims(fy, 1), [1, h * w])
    fx = tf.tile(tf.expand_dims(fx, 1), [1, h * w])
    fy = tf.reshape(fy, [bs, h, w])
    fx = tf.reshape(fx, [bs, h, w])
    y0 = tf.tile(tf.expand_dims(y0, 1), [1, h * w])
    x0 = tf.tile(tf.expand_dims(x0, 1), [1, h * w])
    y0 = tf.reshape(y0, [bs, h, w])
    x0 = tf.reshape(x0, [bs, h, w])

    X = (Z / fx) * (x - x0)
    Y = (Z / fy) * (y - y0)
    pointcloud = tf.stack([tf.reshape(X, [bs, -1]),
                           tf.reshape(Y, [bs, -1]),
                           tf.reshape(Z, [bs, -1])],
                          axis=2, name="world_pointcloud")
    return pointcloud


def Camera2World_p(x, y, Z, fx, fy):
    shape = Z.get_shape()
    bs = int(shape[0])
    h = int(shape[1])
    w = int(shape[2])

    # the intrinsics are shaped [bs]
    # we need them to be shaped [bs,h,w]
    fy = tf.tile(tf.expand_dims(fy, 1), [1, h * w])
    fx = tf.tile(tf.expand_dims(fx, 1), [1, h * w])
    fy = tf.reshape(fy, [bs, h, w])
    fx = tf.reshape(fx, [bs, h, w])

    X = (Z / fx) * x
    Y = (Z / fy) * y
    pointcloud = tf.stack([tf.reshape(X, [bs, -1]),
                           tf.reshape(Y, [bs, -1]),
                           tf.reshape(Z, [bs, -1])],
                          axis=2, name="world_pointcloud")
    return pointcloud


def World2Camera(X, Y, Z, fx, fy, x0, y0):
    shape = Z.get_shape()
    bs = int(shape[0])
    h = int(shape[1])
    w = int(shape[2])

    # the intrinsics are shaped [bs]
    # we need them to be shaped [bs,h*w,1]
    fy = tf.tile(tf.expand_dims(fy, 1), [1, h * w])
    fx = tf.tile(tf.expand_dims(fx, 1), [1, h * w])
    y0 = tf.tile(tf.expand_dims(y0, 1), [1, h * w])
    x0 = tf.tile(tf.expand_dims(x0, 1), [1, h * w])
    fy = tf.reshape(fy, [bs, -1, 1])
    fx = tf.reshape(fx, [bs, -1, 1])
    y0 = tf.reshape(y0, [bs, -1, 1])
    x0 = tf.reshape(x0, [bs, -1, 1])

    x = (X * fx) / (Z + EPS) + x0
    y = (Y * fy) / (Z + EPS) + y0
    proj = tf.concat(axis=2, values=[x, y], name="camera_projection")
    return proj


def World2Camera_p(X, Y, Z, fx, fy):
    shape = Z.get_shape()
    bs = int(shape[0])
    h = int(shape[1])
    w = int(shape[2])

    # the intrinsics are shaped [bs]
    # we need them to be shaped [bs,h*w,1]
    fy = tf.tile(tf.expand_dims(fy, 1), [1, h * w])
    fx = tf.tile(tf.expand_dims(fx, 1), [1, h * w])
    fy = tf.reshape(fy, [bs, -1, 1])
    fx = tf.reshape(fx, [bs, -1, 1])

    x = (X * fx) / (Z + EPS)
    y = (Y * fy) / (Z + EPS)
    proj = tf.concat(axis=2, values=[x, y], name="camera_projection")
    return proj


def World2Camera_single(X, Y, Z, fx, fy):
    shape = Z.get_shape()
    bs = int(shape[0])
    x = (X * fx) / (Z + EPS)
    y = (Y * fy) / (Z + EPS)
    # print_shape(x)
    proj = tf.stack(axis=1, values=[x, y], name="camera_projection")
    return proj


def Camera2World_single(x, y, Z, fx, fy):
    shape = Z.get_shape()
    bs = int(shape[0])

    X = (Z / fx) * x
    Y = (Z / fy) * y

    point = tf.concat([tf.reshape(X, [bs, -1]),
                       tf.reshape(Y, [bs, -1]),
                       tf.reshape(Z, [bs, -1])],
                      axis=1, name="world_point")
    return point


def atan2(y, x):
    with tf.variable_scope("atan2"):
        angle = tf.where(tf.greater(x, 0.0), tf.atan(y / x), tf.zeros_like(x))
        angle = tf.where(tf.greater(y, 0.0), 0.5 * np.pi - tf.atan(x / y), angle)
        angle = tf.where(tf.less(y, 0.0), -0.5 * np.pi - tf.atan(x / y), angle)
        angle = tf.where(tf.less(x, 0.0), tf.atan(y / x) + np.pi, angle)
        angle = tf.where(tf.logical_and(tf.equal(x, 0.0), tf.equal(y, 0.0)),
                         np.nan * tf.zeros_like(x), angle)
        indices = tf.where(tf.less(angle, 0.0))
        updated_values = tf.gather_nd(angle, indices) + (2 * np.pi)
        update = tf.SparseTensor(indices, updated_values, angle.get_shape())
        update_dense = tf.sparse_tensor_to_dense(update)
        return angle + update_dense


def atan2_ocv(y, x):
    with tf.variable_scope("atan2_ocv"):
        # constants
        DBL_EPSILON = 2.2204460492503131e-16
        atan2_p1 = 0.9997878412794807 * (180 / np.pi)
        atan2_p3 = -0.3258083974640975 * (180 / np.pi)
        atan2_p5 = 0.1555786518463281 * (180 / np.pi)
        atan2_p7 = -0.04432655554792128 * (180 / np.pi)
        ax, ay = tf.abs(x), tf.abs(y)
        c = tf.where(tf.greater_equal(ax, ay), tf.div(ay, ax + DBL_EPSILON),
                     tf.div(ax, ay + DBL_EPSILON))
        c2 = tf.square(c)
        angle = (((atan2_p7 * c2 + atan2_p5) * c2 + atan2_p3) * c2 + atan2_p1) * c
        angle = tf.where(tf.greater_equal(ax, ay), angle, 90.0 - angle)
        angle = tf.where(tf.less(x, 0.0), 180.0 - angle, angle)
        angle = tf.where(tf.less(y, 0.0), 360.0 - angle, angle)
        return angle


def normalize(tensor, a=0, b=1):
    with tf.variable_scope("normalize"):
        return tf.div(tf.multiply(tf.subtract(tensor, tf.reduce_min(tensor)), b - a),
                      tf.subtract(tf.reduce_max(tensor), tf.reduce_min(tensor)))


def cart_to_polar_ocv(x, y, angle_in_degrees=False):
    with tf.variable_scope("cart_to_polar_ocv"):
        v = tf.sqrt(tf.add(tf.square(x), tf.square(y)))
        ang = atan2_ocv(y, x)
        scale = 1 if angle_in_degrees else np.pi / 180
        return v, tf.multiply(ang, scale)


def cart_to_polar(x, y, angle_in_degrees=False):
    with tf.variable_scope("cart_to_polar"):
        v = tf.sqrt(tf.add(tf.square(x), tf.square(y)))
        ang = atan2(y, x)
        scale = 180 / np.pi if angle_in_degrees else 1
        return v, tf.multiply(ang, scale)


def flow2color(flow):
    with tf.variable_scope("flow2color"):
        shape = flow.get_shape()
        bs, h, w, c = shape
        maxFlow = 100.0  # tf.maximum(20.0,tf.reduce_max(flow))
        # maxFlow = tf.maximum(50.0,tf.reduce_max(flow))
        # maxFlow = 40.0 #tf.maximum(20.0,tf.reduce_max(flow))
        # maxFlow = tf.maximum(20.0,tf.reduce_max(flow))
        # maxFlow = 5.0 #tf.maximum(20.0,tf.reduce_max(flow))
        # maxFlow = tf.maximum(5.0,tf.reduce_max(flow))
        flow = tf.concat(axis=2, values=[tf.concat(
            axis=3, values=[maxFlow * tf.ones([bs, h, 1, 1]), -maxFlow * tf.ones([bs, h, 1, 1])]), flow])
        flow = tf.concat(axis=2, values=[tf.concat(
            axis=3, values=[-maxFlow * tf.ones([bs, h, 1, 1]), maxFlow * tf.ones([bs, h, 1, 1])]), flow])
        flow = tf.concat(axis=2, values=[tf.concat(
            axis=3, values=[maxFlow * tf.ones([bs, h, 1, 1]), tf.zeros([bs, h, 1, 1])]), flow])
        flow = tf.concat(axis=2, values=[tf.concat(
            axis=3, values=[-maxFlow * tf.ones([bs, h, 1, 1]), tf.zeros([bs, h, 1, 1])]), flow])
        flow = tf.concat(axis=2, values=[tf.concat(
            axis=3, values=[tf.zeros([bs, h, 1, 1]), maxFlow * tf.ones([bs, h, 1, 1])]), flow])
        flow = tf.concat(axis=2, values=[tf.concat(
            axis=3, values=[tf.zeros([bs, h, 1, 1]), -maxFlow * tf.ones([bs, h, 1, 1])]), flow])
        flow = tf.concat(axis=2, values=[tf.concat(
            axis=3, values=[tf.zeros([bs, h, 1, 1]), tf.zeros([bs, h, 1, 1])]), flow])
        flow = tf.concat(axis=2, values=[tf.zeros([bs, h, 1, 2]), flow])
        flow = tf.concat(axis=2, values=[maxFlow * tf.ones([bs, h, 1, 2]), flow])
        flow = tf.concat(axis=2, values=[-maxFlow * tf.ones([bs, h, 1, 2]), flow])
        fx, fy = flow[:, :, :, 0], flow[:, :, :, 1]
        fx = tf.clip_by_value(fx, -maxFlow, maxFlow)
        fy = tf.clip_by_value(fy, -maxFlow, maxFlow)
        v, ang = cart_to_polar_ocv(fx, fy)
        h = normalize(tf.multiply(ang, 180 / np.pi))
        s = tf.ones_like(h)
        v = normalize(v)
        hsv = tf.stack([h, s, v], 3)
        rgb = tf.image.hsv_to_rgb(hsv) * 255
        rgb = tf.slice(rgb, [0, 0, 10, 0], [-1, -1, -1, -1])
        # rgb = rgb[0,0,1:,:]
        return tf.cast(rgb, tf.uint8)


def meshgrid2D(bs, height, width):
    with tf.variable_scope("meshgrid2D"):
        grid_x = tf.matmul(tf.ones(shape=tf.stack([height, 1])),
                           tf.transpose(tf.expand_dims(tf.linspace(0.0, width - 1, width), 1), [1, 0]))
        grid_y = tf.matmul(tf.expand_dims(tf.linspace(0.0, height - 1, height), 1),
                           tf.ones(shape=tf.stack([1, width])))
        grid_x = tf.tile(tf.expand_dims(grid_x, 0), [bs, 1, 1], name="grid_x")
        grid_y = tf.tile(tf.expand_dims(grid_y, 0), [bs, 1, 1], name="grid_y")
        return grid_x, grid_y


def warper(frame, flow, name="warper", reuse=False):
    with tf.variable_scope(name):
        shape = flow.get_shape()
        bs, h, w, c = shape
        if reuse:
            tf.get_variable_scope().reuse_variables()
        warp, occ = transformer(frame, flow, (int(h), int(w)))
        return warp, occ


def meshGridFlat(batchSize, height, width):
    with tf.name_scope('meshGridFlat'):
        # This should be equivalent to:
        #  x_t, y_t = np.meshgrid(np.linspace(-1, 1, width),
        #                np.linspace(-1, 1, height))
        #  ones = np.ones(np.prod(x_t.shape))
        #  grid = np.vstack([x_t.flatten(), y_t.flatten(), ones])
        x_t = tf.matmul(tf.ones(shape=tf.stack([height, 1])),
                        tf.transpose(tf.expand_dims(tf.linspace(-1.0, 1.0, width), 1), [1, 0]))
        y_t = tf.matmul(tf.expand_dims(tf.linspace(-1.0, 1.0, height), 1),
                        tf.ones(shape=tf.stack([1, width])))

        x_t_flat = tf.reshape(x_t, (1, -1))
        y_t_flat = tf.reshape(y_t, (1, -1))

        ones = tf.ones_like(x_t_flat)
        grid = tf.concat(axis=0, values=[x_t_flat, y_t_flat, ones])

        baseGrid = tf.expand_dims(grid, 0)
        grids = []
        for i in range(batchSize):
            grids.append(baseGrid)
        identityGrid = tf.concat(axis=0, values=grids)

        return identityGrid


def flowTransformGrid(flow):
    with tf.name_scope("flowTransformGrid"):
        flowShape = flow.get_shape()
        batchSize = flowShape[0]
        height = flowShape[1]
        width = flowShape[2]

        identityGrid = meshGridFlat(batchSize, height, width)

        flowU = tf.slice(flow, [0, 0, 0, 0], [-1, -1, -1, 1])
        flowV = tf.slice(flow, [0, 0, 0, 1], [-1, -1, -1, 1])

        # scale it to normalized range [-1,1]
        flowU = tf.reshape((flowU * 2) / tf.cast(width, tf.float32),
                           shape=tf.stack([batchSize, 1, -1]))
        flowV = tf.reshape((flowV * 2) / tf.cast(height, tf.float32),
                           shape=tf.stack([batchSize, 1, -1]))
        zeros = tf.zeros(shape=tf.stack([batchSize, 1, height * width]))

        flowScaled = tf.concat(axis=1, values=[flowU, flowV, zeros])

        return identityGrid + flowScaled


def flowSamplingDensity(flow):
    def flatGridToDensity(u, v, h, w):
        with tf.name_scope("flatGridToDensity"):
            u = tf.squeeze(tf.clip_by_value(u, 0, w - 1))
            v = tf.squeeze(tf.clip_by_value(v, 0, h - 1))

            ids = tf.cast(u + (v * w), tf.int32)

            uniques, _, counts = tf.unique_with_counts(ids)
            densityMap = tf.sparse_to_dense(uniques, [h * w], counts, validate_indices=False)
            densityMap = tf.reshape(densityMap, [1, h, w])
            return densityMap

    with tf.name_scope("flowSamplingDensity"):
        flowShape = flow.get_shape()
        batchSize = flowShape[0].value
        h = flowShape[1].value
        w = flowShape[2].value

        grid = flowTransformGrid(flow)

        densities = []
        for it in range(0, batchSize):
            bGrid = tf.slice(grid, [it, 0, 0], [1, -1, -1])
            u = tf.cast(tf.floor((bGrid[:, 0, :] + 1) * w / 2), tf.int32)
            v = tf.cast(tf.floor((bGrid[:, 1, :] + 1) * h / 2), tf.int32)

            da = flatGridToDensity(u, v, h, w)
            #db = flatGridToDensity(u+1,v,h,w)
            #dc = flatGridToDensity(u,v+1,h,w)
            #dd = flatGridToDensity(u+1,v+1,h,w)
            densities.append(da)

        out = tf.concat(axis=0, values=densities)
        return out


def angleGrid(bs, h, w, y0, x0):
    grid_x, grid_y = meshgrid2D(bs, h, w)
    y0 = tf.tile(tf.reshape(y0, [bs, 1, 1]), [1, h, w])
    x0 = tf.tile(tf.reshape(x0, [bs, 1, 1]), [1, h, w])
    grid_y = grid_y - y0
    grid_x = grid_x - x0
    angleGrid = atan2_ocv(grid_y, grid_x)
    return angleGrid


def angle2color(angles):
    v = tf.ones_like(angles)
    h = normalize(tf.multiply(angles, 180 / np.pi))
    s = tf.ones_like(h)
    # v = normalize(v)
    hsv = tf.stack([h, s, v], 3)
    rgb = tf.image.hsv_to_rgb(hsv) * 255
    return tf.cast(rgb, tf.uint8)


def pseudoFlowColor(angles, depth):
    v = 1 / depth
    h = normalize(tf.multiply(angles, 180 / np.pi))
    s = tf.ones_like(h)
    v = normalize(v)
    hsv = tf.stack([h, s, v], 3)
    rgb = tf.image.hsv_to_rgb(hsv) * 255
    return tf.cast(rgb, tf.uint8)


def resFlowColor(flow, angles, depth, tz):
    fx, fy = flow[:, :, :, 0], flow[:, :, :, 1]
    v, fang = cart_to_polar_ocv(fx, fy)
    # fang = atan2_ocv(fy,fx)
    # angles = fang-angles
    # angles = fang
    # v = 1/tf.square(fang-angles)
    # ax, ay = angles[:, :, :, 0], angles[:, :, :, 1]
    v = tz / depth
    # v = tf.ones_like(depth)
    h = normalize(tf.multiply(angles, 180 / np.pi))
    s = tf.ones_like(h)
    v = normalize(v)
    hsv = tf.stack([h, s, v], 3)
    rgb = tf.image.hsv_to_rgb(hsv) * 255
    return tf.cast(rgb, tf.uint8)


def mynormalize(d):
    dmin = tf.reduce_min(d)
    dmax = tf.reduce_max(d)
    d = (d - dmin) / (dmax - dmin)
    return d


def normalize_within_ex(d):
    return tf.map_fn(mynormalize, (d), dtype=tf.float32)


def oned2color(d, norm=True):
    # convert a 1chan input to a 3chan image output
    # (it's not very colorful yet)
    if norm:
        d = mynormalize(d)
    return tf.cast(tf.tile(255 * d, [1, 1, 1, 3]), tf.uint8)


def paired_oned2color(d1, d2, norm=True):
    # convert a two 1chan inputs to two 3chan image outputs,
    # normalized together
    # (it's not very colorful yet)
    if norm:
        dmin = tf.reduce_min(tf.concat(axis=3, values=[d1, d2]))
        dmax = tf.reduce_max(tf.concat(axis=3, values=[d1, d2]))
    else:
        dmin = 0
        dmax = 1
    d1 = tf.cast(tf.tile(255 * ((d1 - dmin) / (dmax - dmin)), [1, 1, 1, 3]), tf.uint8)
    d2 = tf.cast(tf.tile(255 * ((d2 - dmin) / (dmax - dmin)), [1, 1, 1, 3]), tf.uint8)
    return d1, d2


def triple_oned2color(d1, d2, d3, zeros=False):
    # normalize the three together, and don't let zeros ruin it

    cat = tf.concat(axis=3, values=[d1, d2, d3])

    if not zeros:
        zero_mask = tf.cast(tf.equal(cat, 0.0), tf.float32)

        notzero_indices = tf.squeeze(tf.where(tf.not_equal(cat, 0.0)))
        nonzero_max = tf.reduce_max(tf.gather_nd(cat, notzero_indices))

        # set zeros to the max value
        cat = cat + zero_mask * nonzero_max

    # normalize to [0,1]
    dmin = tf.reduce_min(cat)
    dmax = tf.reduce_max(cat)
    cat = (cat - dmin) / (dmax - dmin)
    cat = tf.cast(255 * cat, tf.uint8)

    [d1, d2, d3] = tf.split(cat, 3, axis=3)
    d1 = tf.tile(d1, [1, 1, 1, 3])
    d2 = tf.tile(d2, [1, 1, 1, 3])
    d3 = tf.tile(d3, [1, 1, 1, 3])
    return d1, d2, d3


def preprocess_color(x):
    return tf.cast(x, tf.float32) * 1. / 255 - 0.5


def preprocess_depth(x):
    return tf.cast(x, tf.float32)


def preprocess_valid(x):
    return 1 - tf.cast(x, tf.float32)


def back2color(i):
    return tf.cast((i + 0.5) * 255, tf.uint8)


def zdrt2flow_fc(Z, dp, R, T, scale, fy, fx):
    with tf.variable_scope("zdrt2flow_fc"):
        shape = Z.get_shape()
        bs = int(shape[0])
        h = int(shape[1])
        w = int(shape[2])

        # x is a 3D point
        # p is x's pivot point in 3D
        # dp is the 3D delta from the x to p
        # p = (x + dp)
        # x' = R(x-p) + p + t
        #    = R(x - (x+dp)) + (x+dp) + t
        #    = R(x -  x-dp ) +  x+dp  + t
        #    = R(      -dp ) +  x+dp  + t
        #    = R(-dp) + x + dp + t

        # note that if dp=0 everywhere, then every point has a different pivot (i.e., itself).
        # what we desire is for everything (on the object) to have the SAME pivot.
        # so, ideally, each dp would point to the center.
        # i think this is all fine, i'm just using the wrong function for what i'm doing right now.

        # put the delta pivots into world coordinates
        [dpx, dpy, dpz] = tf.unstack(dp, axis=2)
        dpx = tf.reshape(dpx, [bs, h, w], name="dpx")
        dpy = tf.reshape(dpy, [bs, h, w], name="dpy")
        dpz = tf.reshape(dpz, [bs, h, w], name="dpz")
        XYZ_dp = Camera2World_p(dpx, dpy, dpz, fx, fy)

        # rotate the negative delta pivots
        XYZ_dp_rot = tf.matmul(R, -tf.expand_dims(XYZ_dp, 3))
        XYZ_dp_rot = tf.reshape(XYZ_dp_rot, [bs, h * w, 3])

        # create a pointcloud for the scene
        [grid_x1, grid_y1] = meshgrid2D(bs, h, w)
        Z = tf.reshape(Z, [bs, h, w], name="Z")
        XYZ = Camera2World_p(grid_x1, grid_y1, Z, fx, fy)

        # add it all up
        XYZ_transformed = XYZ_dp_rot + XYZ + XYZ_dp + T

        # project down, so that we get the 2D location of all of these pixels
        [X2, Y2, Z2] = tf.split(axis=2, num_or_size_splits=3,
                                value=XYZ_transformed, name="splitXYZ")
        x2y2_flat = World2Camera_p(X2, Y2, Z2, fx, fy)
        [x2_flat, y2_flat] = tf.split(axis=2, num_or_size_splits=2,
                                      value=x2y2_flat, name="splitxyz_flat")

        # subtract the new 2D locations from the old ones to get optical flow
        x1_flat = tf.reshape(grid_x1, [bs, -1, 1], name="x1")
        y1_flat = tf.reshape(grid_y1, [bs, -1, 1], name="y1")
        u = tf.reshape(x2_flat - x1_flat, [bs, h, w, 1])
        v = tf.reshape(y2_flat - y1_flat, [bs, h, w, 1])
        flow = tf.concat(axis=3, values=[u, v], name="flow")

        return flow, u, v, XYZ_transformed


def zprt2flow(Z, P, R, T, fy, fx):
    # i might want to change this later, but right now let's assume that P is Bx3
    # that is, there is a single pivot for everything in the scene.
    # the pivot, P, is already in world coordinates
    with tf.variable_scope("zprt2flow"):
        shape = Z.get_shape()
        bs = int(shape[0])
        h = int(shape[1])
        w = int(shape[2])

        # Xt = R(X-P) + P + T

        # R is B x 3 x 3
        # P and T are both B x 3
        P = tf.expand_dims(P, axis=1)  # B x 1 x 3
        T = tf.expand_dims(T, axis=1)  # B x 1 x 3
        P = tf.tile(P, [1, h * w, 1])  # B x H*W x 3
        T = tf.tile(T, [1, h * w, 1])  # B x H*W x 3

        # get pointcloud1
        [grid_x1, grid_y1] = meshgrid2D(bs, h, w)
        Z = tf.reshape(Z, [bs, h, w], name="Z")
        XYZ = Camera2World_p(grid_x1, grid_y1, Z, fx, fy)
        # XYZ is B x H*W x 3

        # transform pointcloud1 using r and t, to estimate pointcloud2
        # first we need to transpose XYZ-P, so that 3 is the inner dim
        XYZ_t = tf.transpose(XYZ - P, perm=[0, 2, 1], name="XYZ_t")  # B x 3 x H*W
        # now we can multiply
        XYZ_rot_t = tf.matmul(R, XYZ_t, name="XYZ_rot_t")  # B x 3 x H*W
        # and untranspose.
        XYZ_rot = tf.transpose(XYZ_rot_t, perm=[0, 2, 1], name="XYZ_rot")  # B x H*W x 3
        # add in the P and T
        XYZ2 = XYZ_rot + P + T  # B x H*W x 3
        # project pointcloud2 down, so that we get the 2D location of all of these pixels
        [X2, Y2, Z2] = tf.split(axis=2, num_or_size_splits=3, value=XYZ2, name="splitXYZ")
        x2y2_flat = World2Camera_p(X2, Y2, Z2, fx, fy)
        [x2_flat, y2_flat] = tf.split(axis=2,
                                      num_or_size_splits=2,
                                      value=x2y2_flat, name="splitxyz_flat")

        # subtract the new 2D locations from the old ones to get optical flow
        x1_flat = tf.reshape(grid_x1, [bs, -1, 1], name="x1")
        y1_flat = tf.reshape(grid_y1, [bs, -1, 1], name="y1")
        flow_flat = tf.concat(axis=2,
                              values=[x2_flat - x1_flat, y2_flat - y1_flat],
                              name="flow_flat")
        flow = tf.reshape(flow_flat, [bs, h, w, 2], name="flow")
        return flow, XYZ2


def p2flow_centered(P, h, w, fy, fx):
    # pared down zprt2flow_centered. puts the pivot at the center of the image
    with tf.variable_scope("p2flow_centered"):
        shape = P.get_shape()
        bs = int(shape[0])

        [x, y] = get_xy_from_3d(P, fy, fx)
        x = tf.cast(x, tf.float32)
        y = tf.cast(y, tf.float32)
        x = x - (w / 2)
        y = y - (h / 2)

        flow_x = tf.tile(tf.reshape(x, [bs, 1, 1, 1]), [1, h, w, 1])
        flow_y = tf.tile(tf.reshape(y, [bs, 1, 1, 1]), [1, h, w, 1])
        flow = tf.concat(axis=3, values=[flow_x, flow_y])
        return flow


def p2flow_centered_3D(P, h, w, fy, fx):
    # pared down zprt2flow_centered. puts the pivot at the center of the image
    with tf.variable_scope("p2flow_centered"):
        shape = P.get_shape()
        bs = int(shape[0])

        [x, y] = get_xy_from_3d(P, fy, fx)
        x = tf.cast(x, tf.float32)
        y = tf.cast(y, tf.float32)
        x = x - (w / 2)
        y = y - (h / 2)

        flow_x = tf.tile(tf.reshape(x, [bs, 1, 1, 1]), [1, h, w, 1])
        flow_y = tf.tile(tf.reshape(y, [bs, 1, 1, 1]), [1, h, w, 1])
        flow = tf.concat(axis=3, values=[flow_x, flow_y])
        return flow


def zt2flow_p(Z, t, fy, fx):
    with tf.variable_scope("zt2flow_p"):
        shape = Z.get_shape()
        bs = int(shape[0])
        h = int(shape[1])
        w = int(shape[2])

        # create a pointcloud for the scene
        [grid_x1, grid_y1] = meshgrid2D(bs, h, w)
        Z = tf.reshape(Z, [bs, h, w], name="Z")
        XYZ = Camera2World_p(grid_x1, grid_y1, Z, fx, fy)

        # add it up
        XYZ_transformed = XYZ + t

        # project down, so that we get the 2D location of all of these pixels
        [X2, Y2, Z2] = tf.split(axis=2, num_or_size_splits=3,
                                value=XYZ_transformed, name="splitXYZ")
        x2y2_flat = World2Camera_p(X2, Y2, Z2, fx, fy)
        [x2_flat, y2_flat] = tf.split(axis=2, num_or_size_splits=2,
                                      value=x2y2_flat, name="splitxyz_flat")

        # subtract the new 2D locations from the old ones to get optical flow
        x1_flat = tf.reshape(grid_x1, [bs, -1, 1], name="x1")
        y1_flat = tf.reshape(grid_y1, [bs, -1, 1], name="y1")
        u = tf.reshape(x2_flat - x1_flat, [bs, h, w, 1])
        v = tf.reshape(y2_flat - y1_flat, [bs, h, w, 1])
        flow = tf.concat(axis=3, values=[u, v], name="flow")

        return flow, XYZ_transformed, x2y2_flat


def zt2flow(Z, t, fy, fx, y0, x0):
    with tf.variable_scope("zt2flow_p"):
        shape = Z.get_shape()
        bs = int(shape[0])
        h = int(shape[1])
        w = int(shape[2])

        # create a pointcloud for the scene
        [grid_x1, grid_y1] = meshgrid2D(bs, h, w)
        Z = tf.reshape(Z, [bs, h, w], name="Z")
        XYZ = Camera2World(grid_x1, grid_y1, Z, fx, fy, x0, y0)

        # add it up
        XYZ_transformed = XYZ + t

        # project down, so that we get the 2D location of all of these pixels
        [X2, Y2, Z2] = tf.split(axis=2, num_or_size_splits=3,
                                value=XYZ_transformed, name="splitXYZ")
        x2y2_flat = World2Camera(X2, Y2, Z2, fx, fy, x0, y0)
        [x2_flat, y2_flat] = tf.split(axis=2, num_or_size_splits=2,
                                      value=x2y2_flat, name="splitxyz_flat")

        # subtract the new 2D locations from the old ones to get optical flow
        x1_flat = tf.reshape(grid_x1, [bs, -1, 1], name="x1")
        y1_flat = tf.reshape(grid_y1, [bs, -1, 1], name="y1")
        u = tf.reshape(x2_flat - x1_flat, [bs, h, w, 1])
        v = tf.reshape(y2_flat - y1_flat, [bs, h, w, 1])
        flow = tf.concat(axis=3, values=[u, v], name="flow")

        return flow


def zcom2flow(Z, com1, com2, fy, fx):
    with tf.variable_scope("zcom2flow_fc"):
        # com1 and com2 are translations from the curr pixel to the com.
        # attribute motion to the center of mass changing place.
        t = com2 - com1
        return zt2flow_p(Z, t, fy, fx)


# def xy2mask((x, y, im)):
#     # call this from a tf.map_fn in, e.g., paired_xy2target
#     with tf.variable_scope("xy2mask"):
#         # im is H x W x 3
#         shape = im.get_shape()
#         h = int(shape[0])
#         w = int(shape[1])
#
#         x = tf.cast(x, tf.int32)
#         y = tf.cast(y, tf.int32)
#
#         x = tf.squeeze(x)
#         y = tf.squeeze(y)
#
#         x = tf.maximum(1, x)
#         x = tf.minimum(x, w - 1)
#         y = tf.maximum(1, y)
#         y = tf.minimum(y, h - 1)
#
#         # make a mask with zeros along the cross,
#         # and mult the image with that.
#         left = tf.ones([h, x - 1, 1])
#         vert = tf.zeros([h, 1, 1])
#         right = tf.ones([h, w - x, 1])
#         mask = tf.concat([left, vert, right], axis=1)
#
#         top = tf.slice(mask, [0, 0, 0], [y - 1, -1, -1])
#         horz = tf.zeros([1, w, 1])
#         bottom = tf.slice(mask, [y, 0, 0], [-1, -1, -1])
#
#         mask = tf.concat([top, horz, bottom], axis=0, name="mask")
#         # print_shape(mask)
#         return mask


# def xy2dot_mask((x, y, im)):
#     # generates an image (sized like im) with a single 1 at the xy
#     # call this from a tf.map_fn in, e.g., paired_xy2target
#     # don't try to pass in h,w instead of im
#     with tf.variable_scope("xy2dot_mask"):
#         shape = im.get_shape()
#         h = int(shape[0])
#         w = int(shape[1])
#
#         x = tf.cast(x, tf.int32)
#         y = tf.cast(y, tf.int32)
#
#         x = tf.squeeze(x)
#         y = tf.squeeze(y)
#
#         # make sure x,y are within the bounds
#         x = tf.maximum(1, x)
#         x = tf.minimum(x, w - 1)
#         y = tf.maximum(1, y)
#         y = tf.minimum(y, h - 1)
#
#         # make a mask of zeros
#         mask = tf.zeros([h, w, 1])
#         # add 1 at the specified coordinate
#         indices = [[tf.cast(y, tf.int64), tf.cast(x, tf.int64), 0]]
#         values = [1.0]
#         shape = [h, w, 1]
#         delta = tf.SparseTensor(indices, values, shape)
#
#         mask = mask + tf.sparse_tensor_to_dense(delta)
#
#         # print_shape(mask)
#         return mask


def paired_xyz2target(xyz1, xyz2, im, fy, fx):
    # xyz1 and xyz2 should be B x 3
    with tf.variable_scope("paired_xyz2target"):
        [x1, y1] = get_xy_from_3d(xyz1, fy, fx)
        [x2, y2] = get_xy_from_3d(xyz2, fy, fx)
        return paired_xy2target(x1, y1, x2, y2, im)


def xyz2target(xyz, im, fy, fx):
    # paints a target (a cross) on the image,
    # at the specified xyz location.
    # xyz should be B x 3
    with tf.variable_scope("xyz2target"):
        [x, y] = get_xy_from_3d(xyz, fy, fx)
        return xy2target(x, y, im)


def xyz2target_grid(xyz, im, fy, fx):
    # xyz should be B x 3
    with tf.variable_scope("xyz2target"):
        [x, y] = get_xy_from_3d(xyz, fy, fx)
        im = xy2target(x, y, im)
        op = 0.1
        im = xy2target(x * 0.5, y, im, opacity=op)
        im = xy2target(x, y * 0.5, im, opacity=op)
        im = xy2target(x * 1.5, y, im, opacity=op)
        im = xy2target(x, y * 1.5, im, opacity=op)
        im = xy2target(x * 0.5, y * 0.5, im, opacity=op)
        im = xy2target(x * 1.5, y * 1.5, im, opacity=op)
        return im


def paired_xy2target(x1, y1, x2, y2, im):
    with tf.variable_scope("paired_xy2target"):
        # print_shape(im)

        # mark the target with a cross of zeros
        mask1 = tf.map_fn(xy2mask, [x1, y1, im], dtype=tf.float32)
        mask2 = tf.map_fn(xy2mask, [x2, y2, im], dtype=tf.float32)

        # print_shape(mask1)
        # mask1 = xy2mask(x1,y1,im)
        # mask2 = xy2mask(x2,y2,im)
        mask1_3 = tf.tile(mask1, [1, 1, 1, 3])
        mask2_3 = tf.tile(mask2, [1, 1, 1, 3])

        # put the image into [0,1]
        im = im + 0.5

        # cross out both targets
        im = im * mask1_3
        im = im * mask2_3

        # paint the second mask in red
        zero = tf.zeros_like(mask2)
        mask2_3 = tf.concat([(1 - mask2), zero, zero], axis=3)
        im = im + mask2_3

        # restore the image to [-0.5,0.5]
        im = im - 0.5
        return im


def xy2target(x, y, im, opacity=1.0):
    with tf.variable_scope("xy2target"):
        # print_shape(im)

        # mark the target with a cross of zeros
        mask = tf.map_fn(xy2mask, [x, y, im], dtype=tf.float32)

        # print_shape(mask)
        mask_3 = tf.tile(mask, [1, 1, 1, 3])

        # put the image into [0,1]
        im = im + 0.5

        # cross out the target
        im = im * (1 - (1 - mask_3) * opacity)

        # restore the image to [-0.5,0.5]
        im = im - 0.5
        return im


def zcom2com_fc(Z, com, fy, fx):
    with tf.variable_scope("zcom2com_fc"):
        shape = Z.get_shape()
        bs = int(shape[0])
        h = int(shape[1])
        w = int(shape[2])

        # create a pointcloud for the scene
        [grid_x, grid_y] = meshgrid2D(bs, h, w)
        Z = tf.reshape(Z, [bs, h, w], name="Z")
        XYZ = Camera2World_p(grid_x, grid_y, Z, fx, fy)

        # move every point (on the toy, at least) to the com
        XYZ_transformed = XYZ + com

        # project down, so that we get the 2D location of all of these pixels
        [X2, Y2, Z2] = tf.split(axis=2, num_or_size_splits=3,
                                value=XYZ_transformed, name="splitXYZ")
        x2y2_flat = World2Camera_p(X2, Y2, Z2, fx, fy)
        [x2_flat, y2_flat] = tf.split(axis=2, num_or_size_splits=2,
                                      value=x2y2_flat, name="splitxyz_flat")

        # subtract the new 2D locations from the old ones to get optical flow
        [grid_x1, grid_y1] = meshgrid2D(bs, h, w)
        x1_flat = tf.reshape(grid_x1, [bs, -1, 1], name="x1")
        y1_flat = tf.reshape(grid_y1, [bs, -1, 1], name="y1")
        u = tf.reshape(x2_flat - x1_flat, [bs, h, w, 1])
        v = tf.reshape(y2_flat - y1_flat, [bs, h, w, 1])
        flow = tf.concat(axis=3, values=[u, v], name="flow")
        return flow, XYZ_transformed


def offsetbbox(o1b, off_h, off_w, crop_h, crop_w, dotrim=False):
    o1b = tf.cast(o1b, tf.int32)
    # first get all 4 edges
    Ls = tf.slice(o1b, [0, 0], [-1, 1])
    Ts = tf.slice(o1b, [0, 1], [-1, 1])
    Rs = tf.slice(o1b, [0, 2], [-1, 1])
    Bs = tf.slice(o1b, [0, 3], [-1, 1])
    # next, offset by crop
    Ls = Ls - off_w
    Rs = Rs - off_w
    Ts = Ts - off_h
    Bs = Bs - off_h
    # finally, trim boxes if they go past edges
    if dotrim:
        assert False
        Ls = tf.maximum(Ls, 0)
        Rs = tf.maximum(Rs, 0)
        Ts = tf.maximum(Ts, 0)
        Bs = tf.maximum(Bs, 0)
        Ls = tf.minimum(Ls, crop_w)
        Rs = tf.minimum(Rs, crop_w)
        Ts = tf.minimum(Ts, crop_h)
        Bs = tf.minimum(Bs, crop_h)
    # then repack
    o1b = tf.concat(axis=1, values=[Ls, Ts, Rs, Bs])
    o1b = tf.cast(o1b, tf.int64)
    return o1b


def avg2(a, b):
    return (a + b) / 2.0


def decode_labels(mask, palette, num_images=1):
    n_classes = len(palette)
    n, h, w, c = mask.shape
    assert(n >= num_images), 'Batch size %d should be greater or equal than number of images to save %d.' % (
        n, num_images)
    outputs = np.zeros((num_images, h, w, 3), dtype=np.uint8)
    for i in range(num_images):
        img2 = Image.new('RGB', (len(mask[i, 0]), len(mask[i])))
        pixels = img2.load()
        for j_, j in enumerate(mask[i, :, :, 0]):
            for k_, k in enumerate(j):
                assert 0 < k <= 14
                pixels[k_, j_] = palette[int(k) - 1]
        outputs[i] = np.array(img2)
    return outputs

# we may want to easily change the way we encode/decode depth
# brox encodes it as 1/z
# eigen encodes it as log(z)


def encode_depth_inv(x):
    return 1 / (x + EPS)


def decode_depth_inv(x):
    return 1 / (x + EPS)


def encode_depth_log(x):
    return tf.log(x + EPS)
    # return tf.exp(x+EPS)


def decode_depth_log(x):
    return tf.exp(x + EPS)
    # # return tf.log(x+EPS)
    # # # x_safe = tf.where(x < EPS, x, tf.ones_like(x))
    # indices = tf.squeeze(tf.where(x > EPS*10))
    # updated_values = tf.log(tf.gather_nd(x, indices))
    # update = tf.SparseTensor(indices, updated_values, x.get_shape())
    # update_dense = tf.sparse_tensor_to_dense(update)
    # return update_dense


def encode_depth_id(x):
    return x


def decode_depth_id(x):
    return x


def encode_depth(x, encoding):
    if encoding == 'id':
        return encode_depth_id(x)
    elif encoding == 'log':
        return encode_depth_log(x)
    elif encoding == 'inv':
        return encode_depth_inv(x)
    else:
        assert False


def decode_depth(x, encoding):
    if encoding == 'id':
        return decode_depth_id(x)
    elif encoding == 'log':
        return decode_depth_log(x)
    elif encoding == 'inv':
        return decode_depth_inv(x)
    else:
        assert False
    return x


def match(xs, ys):  # sort of like a nested zip
    result = {}
    for i, (x, y) in enumerate(zip(xs, ys)):
        if type(x) == type([]):
            subresult = match(x, y)
            result.update(subresult)
        else:
            result[x] = y
    return result


def feed_from(inputs, variables, sess):
    return match(variables, sess.run(inputs))


def feed_from2(inputs1, variables1, inputs2, variables2, sess):
    match([variables1, variables2], sess.run([inputs1, inputs2]))


def poses2rots_v2(poses):
    rx = tf.slice(poses, [0, 0], [-1, 1])
    ry = tf.slice(poses, [0, 1], [-1, 1]) + math.pi / 2.0
    rz = tf.slice(poses, [0, 2], [-1, 1])
    rots = sinabg2r(tf.sin(rz), tf.sin(ry), tf.sin(rx))
    return rots


def poses2rots(poses):
    rxryrz = tf.slice(poses, [0, 6], [-1, 3])
    rx = tf.slice(rxryrz, [0, 0], [-1, 1])
    ry = tf.slice(rxryrz, [0, 1], [-1, 1]) + math.pi / 2.0
    rz = tf.slice(rxryrz, [0, 2], [-1, 1])
    rots = sinabg2r(tf.sin(rz), tf.sin(ry), tf.sin(rx))
    return rots


def decompress_seg(num, seg):
    '''input: HxWx1, with max value being seg
    '''
    pass


def masks2boxes(masks):
    pass


def seg2masksandboxes(num, seg):
    #'gt_masks': 'masks of instances in this image. (instance-level masks), of shape (N, image_height, image_width)',
    #'gt_boxes': 'bounding boxes and classes of instances in this image, of shape (N, 5), each entry is (x1, y1, x2, y2)'
    # the fifth feature of each box is the class id

    masks = decompress_seg(num, seg)
    boxes = masks2boxes(masks)

    masks = []
    boxes = []
    # assert False #not yet implemented
    return masks, boxes


# def selectmask(cls_idx, masks):
#     def select_single((idx, mask)):
#         # idx is (), mask is 14x14x2
#         return tf.slice(mask, tf.stack([0, 0, idx]), [-1, -1, 1])
#     return tf.map_fn(select_single, [cls_idx, masks],
#                      parallel_iterations=128, dtype=tf.float32)


# def select_by_last(indices, items):
#     axis = len(items.get_shape())
#
#     def select_single((idx, item)):
#         start = tf.stack([0 for i in range(axis - 2)] + [idx])
#         end = [-1 for i in range(axis - 2)] + [1]
#         return tf.squeeze(tf.slice(item, start, end))
#     return tf.map_fn(select_single, [indices, items],
#                      parallel_iterations=128, dtype=items.dtype)


# def extract_gt_boxes((counts, segs)):
#     n = tf.squeeze(counts)  # bs is 1
#     seg = tf.squeeze(segs)  # bs is 1
#
#     def extract_instance(idx):
#         mask = tf.equal(seg, idx + 1)
#         bbox = tf.concat(axis=0, values=[mask2bbox(mask), [0]])
#         return bbox
#     it = tf.range(0, n)
#     result = tf.map_fn(extract_instance, it, dtype=tf.int32)
#     return result


def vis_detections(im, class_name, dets, thresh=0.3):
    """Draw detected bounding boxes."""
    inds = np.where(dets[:, -1] >= thresh)[0]
    if len(inds) == 0:
        return
    im = im[0, :, :, :]
    for i in inds:
        bbox = dets[i, :4]
        score = dets[i, -1]
        print ("%.2f confident that there is a %s at " % (score, class_name))
        print(bbox)
        y = int(bbox[0])
        x = int(bbox[1])

        y2 = int(bbox[2])
        x2 = int(bbox[3])
        im[x:x2, y, 0] = 255
        im[x:x2, y, 1] = 0
        im[x:x2, y, 1] = 0
        im[x:x2, y2, 0] = 255
        im[x:x2, y2, 1] = 0
        im[x:x2, y2, 1] = 0

        im[x, y:y2, 0] = 255
        im[x, y:y2, 1] = 0
        im[x, y:y2, 2] = 0
        im[x2, y:y2, 0] = 255
        im[x2, y:y2, 1] = 0
        im[x2, y:y2, 2] = 0
    return im  # imsave(out_file, im)


def box_correspond_np(pred_boxes, gt_boxes, thresh=0.0):
    # print 'box correspond'
    # print np.shape(pred_boxes)
    # print np.shape(gt_boxes)
    overlaps = bbox_overlaps(
        np.ascontiguousarray(pred_boxes, dtype=np.float),
        np.ascontiguousarray(gt_boxes, dtype=np.float))  # Nx32
    dets_argmax_overlaps = overlaps.argmax(axis=1)
    dets_max_overlaps = overlaps.max(axis=1)
    matchmask = dets_max_overlaps > thresh

    return dets_argmax_overlaps, matchmask


def box_correspond(pred_boxes, gt_boxes, thresh=0.75):
    dets, mask = tf.py_func(box_correspond_np, [pred_boxes, gt_boxes, thresh], (tf.int64, tf.bool))
    #isempty = tf.greater(tf.shape(gt_boxes)[0], 0)
    dets = tf.reshape(dets, (tf.shape(pred_boxes)[0],))
    masks = tf.reshape(mask, (tf.shape(pred_boxes)[0],))
    return dets, masks


def tf_nms(dets, thresh):
    keep = tf.py_func(lambda a, b, c: np.array(nms(a, b, c)).astype(np.int64),
                      [dets, thresh, True],
                      tf.int64, stateful=False)
    return tf.cast(keep, tf.int32)


def get_good_detections(feats, probs, bboxdeltas, rois, many_classes=True,
                        conf_thresh=0.8, nms_thresh=0.3):
    boxes = tf_rois_and_deltas_to_bbox(bboxdeltas, rois)
    if many_classes:
        cls_ind = 7
        cls_boxes = boxes[:, 4 * cls_ind:4 * (cls_ind + 1)]
        cls_probs = probs[:, cls_ind]
    else:
        cls_boxes = boxes
        cls_probs = probs

    indices = tf.squeeze(tf.where(cls_probs > conf_thresh))

    feats = tf.gather(feats, indices)
    cls_probs = tf.gather(cls_probs, indices)
    cls_boxes = tf.gather(cls_boxes, indices)
    rois = tf.gather(rois, indices)

    # cls_boxes is [N x 4]
    # cls_probs is [N]; make it [N x 1] so we can concat
    cls_probs = tf.expand_dims(cls_probs, axis=-1)

    dets = tf.concat([cls_boxes, cls_probs], axis=1)
    keep = tf_nms(dets, nms_thresh)

    dets = tf.gather(dets, keep)
    feats = tf.gather(feats, keep)
    probs = tf.gather(probs, keep)
    deltas = tf.gather(bboxdeltas, keep)
    rois = tf.gather(rois, keep)

    # return dets, feats, probs, deltas
    # return dets, feats
    dets = tf.slice(dets, [0, 0], [-1, 4])
    return dets, feats


def pose_cls2angle(rxc, ryc, rzc):
    def __f(x): return tf.cast(tf.argmax(x, axis=1), tf.float32)
    xi = __f(rxc)
    yi = __f(ryc)
    zi = __f(rzc)
    rx = xi / 36 * 2 * pi - pi
    ry = yi / 36 * 2 * pi - pi
    rz = zi / 36 * 2 * pi - pi
    return tf.stack([rx, ry, rz], axis=1)


def pose_angle2cls(rots):
    rx = rots[:, 0]
    ry = rots[:, 1]
    rz = rots[:, 2]

    def __round(w): return tf.cast(tf.round((w + pi) * 36 / (2 * pi)), tf.int32)
    binx = __round(rx)
    biny = __round(ry)
    binz = __round(rz)
    # now make it categorical

    def __oh(w): return tf.one_hot(w, depth=36, axis=1)
    clsx = __oh(binx)
    clsy = __oh(biny)
    clsz = __oh(binz)
    return clsx, clsy, clsz


def box2corners(bbox):
    L, T, R, B = bbox
    lb = np.array([L, B])
    lt = np.array([L, T])
    rb = np.array([R, B])
    rt = np.array([R, T])
    return [lb, lt, rb, rt]


# def drawbox((lb, lt, rb, rt), c, canvas, scale=1):
#     drawline(lb, lt, c, canvas, scale)
#     drawline(rb, rt, c, canvas, scale)
#     drawline(lb, rb, c, canvas, scale)
#     drawline(lt, rt, c, canvas, scale)


def drawbox2(bbox, canvas, c, scale=1):
    corners = box2corners(bbox)
    drawbox([corner * scale for corner in corners], c, canvas, scale)


def plotcuboid2d(corners, canvas, c, scale=1):
    for box in corners:
        # there are twelve edges
        for i1, i2 in [(0, 1), (0, 2), (0, 4),
                       (1, 3), (1, 5), (2, 3),
                       (2, 6), (3, 7), (4, 5),
                       (4, 6), (5, 7), (6, 7)]:
            p1 = box[i1] * scale
            p2 = box[i2] * scale
            drawline(p1, p2, c, canvas, scale)


def masked_stats(masks, field):
    mean, var = tf.nn.weighted_moments(field, [1, 2], masks)
    return mean, tf.sqrt(var + EPS)


def pcaembed(img, clip=True):
    H, W, K = np.shape(img)
    pixelskd = np.reshape(img, (H * W, K))
    P = PCA(3)
    # only fit a small subset for efficiency
    P.fit(np.random.permutation(pixelskd)[:4096])
    pixels3d = P.transform(pixelskd)
    out_img = np.reshape(pixels3d, (H, W, 3))
    if clip:
        std = np.std(out_img)
        mu = np.mean(out_img)
        out_img = np.clip(out_img, mu - std * 2, mu + std * 2)
    out_img -= np.min(out_img)
    out_img /= np.max(out_img)
    out_img *= 255
    return out_img


def tsneembed(img, clip=True):
    H, W, K = np.shape(img)
    pixelskd = np.reshape(img, (H * W, K))
    P = TSNE(3, perplexity=10.0)
    pixels3d = P.fit_transform(pixelskd)
    out_img = np.reshape(pixels3d, (H, W, 3))
    if clip:
        std = np.std(out_img)
        mu = np.mean(out_img)
        out_img = np.clip(out_img, mu - std * 2, mu + std * 2)
    out_img -= np.min(out_img)
    out_img /= np.max(out_img)
    out_img *= 255
    return out_img


def get_mean_3D_com_in_mask(XYZ, mask):
    print_shape(XYZ)
    shape = XYZ.get_shape()
    bs = int(shape[0])
    [X, Y, Z] = tf.split(axis=2, num_or_size_splits=3, value=XYZ)
    mask = tf.reshape(mask, [bs, -1, 1])
    mean_X = tf.reduce_sum(X * mask, axis=[1, 2]) / (EPS + tf.reduce_sum(mask, axis=[1, 2]))
    mean_Y = tf.reduce_sum(Y * mask, axis=[1, 2]) / (EPS + tf.reduce_sum(mask, axis=[1, 2]))
    mean_Z = tf.reduce_sum(Z * mask, axis=[1, 2]) / (EPS + tf.reduce_sum(mask, axis=[1, 2]))
    com = tf.stack([mean_X, mean_Y, mean_Z], axis=1)
    return com


def get_xy_from_3d(p, fy, fx):
    shape = p.get_shape()
    bs = int(shape[0])
    [X, Y, Z] = tf.unstack(p, axis=1)
    xyt_flat = World2Camera_single(X, Y, Z, fx, fy)
    [x, y] = tf.split(axis=1, num_or_size_splits=2, value=xyt_flat)
    x = tf.reshape(x, [bs])
    y = tf.reshape(y, [bs])
    return x, y

# def get_xyz_from_2D(x, y, Z, fy, fx, y0, x0):
#     # print_shape(Z)
#     shape = Z.get_shape()
#     bs = int(shape[0])
#     h = int(shape[1])
#     w = int(shape[2])
#     zero = tf.zeros([bs, h, w, 1])
#     zero = tf.zeros([h, w, 1])
#     mask = xy2dot_mask((x, y, zero))
#     mask = tf.expand_dims(mask,0)
#     XYZ = Camera2World_single(x,y,Z,fy,fx,y0,x0)
#     XYZ = get_mean_3D_in_mask(XYZ, mask)
#     return XYZ


def spin(image, depth, com, R, fy, fx):
    shape = image.get_shape()
    bs = int(shape[0])
    h = int(shape[1])
    w = int(shape[2])

    # generate 3 images: before R, orig, and after R; all centered
    T = tf.zeros([bs, 3])
    [R_inverse, T_inverse] = split_rt(safe_inverse(merge_rt(R, T)))

    spin_flow_12, _ = zprt2flow(depth, com, R, T, fy, fx)
    spin_flow_21, _ = zprt2flow(depth, com, R_inverse, T_inverse, fy, fx)

    image2 = xyz2target_grid(com, image, fy, fx)

    image3, occ1 = warper(image2, spin_flow_12)
    image1, occ3 = warper(image2, spin_flow_21)

    image1 = image1 * occ1
    image3 = image3 * occ3
    image1 = tf.reshape(image1, [bs, h, w, 3])
    image3 = tf.reshape(image3, [bs, h, w, 3])

    # translate so the com is at h/2, w/2
    center12 = p2flow_centered(com, h, w, fy, fx)
    image2, _ = warper(image2, center12)
    image1, _ = warper(image1, center12)
    image3, _ = warper(image3, center12)

    _, occ = warper(image2, -center12)
    image2 *= occ
    image1 *= occ
    image3 *= occ

    return image1, image2, image3


def split_tta(Z_b, Z_g, v_b, v_g, tab_val, toy_val, arm_val):
    md = (Z_b - Z_g)
    md_sum = tf.summary.histogram("md", md)

    tab = tf.where(md < tab_val, tf.ones_like(md), tf.zeros_like(md))
    toy = tf.where(md > toy_val, tf.ones_like(md), tf.zeros_like(md))
    arm = tf.where(md > arm_val, tf.ones_like(md), tf.zeros_like(md))
    arm = arm * v_b * v_g
    tab = tab * v_b * v_g
    tab_or_arm = tf.where(tab + arm > 0, tf.ones_like(tab), tf.zeros_like(tab))
    toy = (toy * (1 - tab_or_arm)) * v_b * v_g

    return tab, toy, arm


def average_in_mask(x, mask):
    shape = x.get_shape()
    bs = int(shape[0])
    h = int(shape[1])
    w = int(shape[2])
    c = int(shape[3])

    mask_c = tf.tile(mask, [1, 1, 1, c])
    x_keep = x * mask_c
    x = tf.reduce_sum(x_keep, axis=[1, 2]) / tf.reduce_sum(mask + EPS, axis=[1, 2])

    return x


def get_com_free(Z1, Zb, mask, fy, fx):
    shape = Z1.get_shape()
    bs = int(shape[0])
    h = int(shape[1])
    w = int(shape[2])
    [grid_x, grid_y] = meshgrid2D(bs, h, w)
    Z = tf.reshape(Z1, [bs, h, w])
    Zb = tf.reshape(Zb, [bs, h, w])
    XYZ = Camera2World_p(grid_x, grid_y, Z, fx, fy)
    XYZb = Camera2World_p(grid_x, grid_y, Zb, fx, fy)
    com_on_obj = get_mean_3D_com_in_mask(XYZ, mask)
    com_on_table = get_mean_3D_com_in_mask(XYZb, mask)
    # com_free = com_on_table
    com_free = com_on_obj
    # # com_free = (com_on_table+com_on_obj)/2

    # # let's try something else:
    # [grid_x,grid_y] = meshgrid2D(bs, h, w)
    # x = average_in_mask(tf.expand_dims(grid_x,3), mask)
    # y = average_in_mask(tf.expand_dims(grid_y,3), mask)
    # z = average_in_mask(Z1, mask)

    # # x = tf.Print(x, [x], 'x')
    # # y = tf.Print(y, [y], 'y')

    # x = tf.reshape(x,[-1])
    # y = tf.reshape(y,[-1])
    # z = tf.reshape(z,[-1])

    # com_free = Camera2World_single(x,y,z,fx,fy)

    return com_free

# Calculates rotation matrix to euler angles
# The result is the same as MATLAB except the order
# of the euler angles ( x and z are swapped ).


def rotationMatrixToEulerAngles(R):
    R = R[0]
    sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])

    singular = sy < 1e-6

    if not singular:
        x = math.atan2(R[2, 1], R[2, 2])
        y = math.atan2(-R[2, 0], sy)
        z = math.atan2(R[1, 0], R[0, 0])
    else:
        x = math.atan2(-R[1, 2], R[1, 1])
        y = math.atan2(-R[2, 0], sy)
        z = 0

    return np.array([x, y, z])


def triple_sin(a, b, g):
    sina = tf.sin(a)
    sinb = tf.sin(b)
    sing = tf.sin(g)
    return sina, sinb, sing


def triple_cos(a, b, g):
    cosa = tf.cos(a)
    cosb = tf.cos(b)
    cosg = tf.cos(g)
    return cosa, cosb, cosg


def sincos_norm(sin, cos):
    n = norm(tf.stack([sin, cos], axis=1))
    sin = sin / n
    cos = cos / n
    return sin, cos


def euler2mat(z, y, x):
    # https://github.com/tinghuiz/SfMLearner/blob/master/utils.py
    """Converts euler angles to rotation matrix
     TODO: remove the dimension for 'N' (deprecated for converting all source
           poses altogether)
     Reference: https://github.com/pulkitag/pycaffe-utils/blob/master/rot_utils.py#L174

    Args:
        z: rotation angle along z axis (in radians) -- size = [B, N]
        y: rotation angle along y axis (in radians) -- size = [B, N]
        x: rotation angle along x axis (in radians) -- size = [B, N]
    Returns:
        Rotation matrix corresponding to the euler angles -- size = [B, N, 3, 3]
    """
    B = tf.shape(z)[0]
    N = 1
    z = tf.clip_by_value(z, -np.pi, np.pi)
    y = tf.clip_by_value(y, -np.pi, np.pi)
    x = tf.clip_by_value(x, -np.pi, np.pi)

    # Expand to B x N x 1 x 1
    z = tf.expand_dims(tf.expand_dims(z, -1), -1)
    y = tf.expand_dims(tf.expand_dims(y, -1), -1)
    x = tf.expand_dims(tf.expand_dims(x, -1), -1)

    zeros = tf.zeros([B, N, 1, 1])
    ones = tf.ones([B, N, 1, 1])

    cosz = tf.cos(z)
    sinz = tf.sin(z)
    rotz_1 = tf.concat([cosz, -sinz, zeros], axis=3)
    rotz_2 = tf.concat([sinz,  cosz, zeros], axis=3)
    rotz_3 = tf.concat([zeros, zeros, ones], axis=3)
    zmat = tf.concat([rotz_1, rotz_2, rotz_3], axis=2)

    cosy = tf.cos(y)
    siny = tf.sin(y)
    roty_1 = tf.concat([cosy, zeros, siny], axis=3)
    roty_2 = tf.concat([zeros, ones, zeros], axis=3)
    roty_3 = tf.concat([-siny, zeros, cosy], axis=3)
    ymat = tf.concat([roty_1, roty_2, roty_3], axis=2)

    cosx = tf.cos(x)
    sinx = tf.sin(x)
    rotx_1 = tf.concat([ones, zeros, zeros], axis=3)
    rotx_2 = tf.concat([zeros, cosx, -sinx], axis=3)
    rotx_3 = tf.concat([zeros, sinx, cosx], axis=3)
    xmat = tf.concat([rotx_1, rotx_2, rotx_3], axis=2)

    rotMat = tf.matmul(tf.matmul(xmat, ymat), zmat)
    return rotMat

# def masked_min_max(x, mask):
#     # uh, this doesn't really work
#     indices = tf.squeeze(tf.where(tf.greater(mask, 0.0)))
#     min = tf.reduce_min(tf.gather_nd(x,indices))
#     # print_shape(min)
#     max = tf.reduce_max(tf.gather_nd(x,indices))
#     # print_shape(max)
#     return min, max


# def masked_min_max((x, mask)):
#     # this doesn't work!
#     # use the separate max/min things below
#     indices = tf.squeeze(tf.where(tf.greater(mask, 0.0)))
#     min = tf.reduce_min(tf.gather_nd(x, indices))
#     max = tf.reduce_max(tf.gather_nd(x, indices))
#     return min, max


# def masked_min((x, mask)):
#     indices = tf.squeeze(tf.where(tf.greater(mask, 0.0)))
#     min = tf.reduce_min(tf.gather_nd(x, indices))
#     return min


# def masked_max((x, mask)):
#     indices = tf.squeeze(tf.where(tf.greater(mask, 0.0)))
#     max = tf.reduce_max(tf.gather_nd(x, indices))
#     return max


def crop_and_resize_around_com(com, image, sideLength, fy, fx):
    # this func is flexible enough to crop AND resize,
    # but right now it actually just crops
    # crop_around_com (below) is more straightforward

    shape = image.get_shape()
    bs = int(shape[0])
    h = int(shape[1])
    w = int(shape[2])

    # get the xy of the com
    com_x, com_y = get_xy_from_3d(com, fy, fx)

    # tensorflow wants the coords normalized for height/width
    y1 = (com_y - sideLength / 2) / h
    y2 = (com_y + sideLength / 2) / h
    x1 = (com_x - sideLength / 2) / w
    x2 = (com_x + sideLength / 2) / w
    boxes = tf.stack([y1, x1, y2, x2], axis=1)

    # one box per batch ind
    box_ind = tf.cast(tf.range(0, bs), tf.int32)

    size = tf.constant([sideLength, sideLength], tf.int32)

    # crop and resize!
    u_image = tf.image.crop_and_resize(image, boxes, box_ind, size)

    return u_image


def crop_around_com(com, image, sideLength, fy, fx):
    # get the xy of the com
    com_x, com_y = get_xy_from_3d(com, fy, fx)

    # com_x = tf.Print(com_x, [com_x], 'com_x')
    # com_y = tf.Print(com_y, [com_y], 'com_y')

    # set it as the center of our glimpse. (y then x)
    offsets = tf.stack([com_y, com_x], axis=1)

    # glimpse!
    u_image = tf.image.extract_glimpse(image,
                                       [sideLength, sideLength],
                                       offsets,
                                       centered=False,
                                       normalized=False)
    return u_image


def basic_blur(image):
    blur_kernel = tf.transpose(tf.constant([[[[1. / 16, 1. / 8, 1. / 16],
                                              [1. / 8, 1. / 4, 1. / 8],
                                              [1. / 16, 1. / 8, 1. / 16]]]],
                                           dtype=tf.float32), perm=[3, 2, 1, 0])
    image = tf.pad(image, [[0, 0], [1, 1], [1, 1], [0, 0]], "SYMMETRIC")
    image = tf.nn.conv2d(image, blur_kernel, strides=[1, 1, 1, 1], padding="VALID")
    return image
