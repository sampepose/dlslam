import tensorflow as tf
import math
from src.utils import *

EPS = 1e-6


def rtLoss(rt_e, rt_g):
    with tf.variable_scope("rtLoss"):
        rt_eg = ominus(rt_e, rt_g)
        rtd = tf.reduce_mean(compute_distance(rt_eg))
        rta = tf.reduce_mean(compute_angle(rt_eg))
        td = tf.reduce_mean(compute_t_diff(rt_e, rt_g))
        ta = tf.reduce_mean(compute_t_ang(rt_e, rt_g))
        return rtd, rta, td, ta


def safe_rtLoss(rt_e, rt_g):
    with tf.variable_scope("rtLoss"):
        rt_eg = ominus(rt_e, rt_g)
        rtd = 0.0
        rta = 0.0
        rtd = tf.reduce_mean(compute_distance(rt_eg))
        rta = tf.reduce_mean(compute_angle(rt_eg))
        return rtd, rta


def radian_l1Loss(e, g):
    with tf.variable_scope("radian_l1Loss"):
        pi = math.pi
        l = tf.reduce_mean(pi / 2 - tf.abs(tf.abs(e - g) - pi / 2))
        return l


def radian_l1Loss_1D(e, g):
    # e and g should be shaped [bs, n]
    # this will return a [bs] tensor
    with tf.variable_scope("radian_l1Loss"):
        pi = math.pi
        l = tf.reduce_mean(pi / 2 - tf.abs(tf.abs(e - g) - pi / 2), axis=1)
        return l


def l1Loss(e, g):
    with tf.variable_scope("l1Loss"):
        l = tf.reduce_mean(tf.abs(e - g))
        return l


def masked_l1Loss(e, g, valid):
    with tf.variable_scope("masked_l1Loss"):
        l = tf.reduce_mean(tf.abs(e - g), axis=3, keep_dims=True)
        l = l * valid
        l = tf.reduce_sum(l) / tf.reduce_sum(valid + EPS)
        return l


def masked_hingeLoss(e, g, valid, slack):
    with tf.variable_scope("masked_hingeLoss"):
        diff = tf.abs(e - g)
        penalty = tf.where(diff > slack,
                           diff - slack,
                           tf.zeros_like(diff))
        l = tf.reduce_mean(penalty, axis=3, keep_dims=True)
        l = l * valid
        l = tf.reduce_sum(l) / tf.reduce_sum(valid + EPS)
        return l


def l2Loss(e, g):
    with tf.variable_scope("l2Loss"):
        l = tf.reduce_mean(tf.square(e - g))
        return l


def masked_l2Loss(e, g, valid):
    with tf.variable_scope("masked_l2Loss"):
        l = tf.reduce_mean(tf.square(e - g), axis=3, keep_dims=True)
        l = l * valid
        l = tf.reduce_sum(l) / tf.reduce_sum(valid + EPS)
        return l


def siLoss(e, g, lamb=0.5):
    # assume e and g are log values
    with tf.variable_scope("siLoss"):
        shape = e.get_shape()
        _, h, w, _ = shape
        n = int(h * w)
        d = e - g
        term1 = tf.reduce_sum(tf.square(d), axis=[1, 2, 3]) / n
        term2 = tf.square(tf.reduce_sum(d, axis=[1, 2, 3])) / (n * n)
        l = term1 - lamb * term2
        l = tf.reduce_mean(l)
        return l


def masked_siLoss(e, g, valid, lamb=0.5):
    with tf.variable_scope("masked_siLoss"):
        shape = e.get_shape()
        _, h, w, _ = shape
        n = tf.reduce_sum(valid, axis=[1, 2, 3])
        d = e - g
        d = d * valid
        term1 = tf.reduce_sum(tf.square(d), axis=[1, 2, 3]) / n
        term2 = tf.square(tf.reduce_sum(d, axis=[1, 2, 3])) / (n * n)
        l = term1 - lamb * term2
        l = tf.reduce_mean(l)
        return l


def huberLoss(labels, predictions, delta=1.0):
    residual = tf.abs(predictions - labels)
    condition = tf.less(residual, delta)
    small_res = 0.5 * tf.square(residual)
    large_res = delta * residual - 0.5 * tf.square(delta)
    return tf.reduce_mean(tf.where(condition, small_res, large_res))


def warped_z_loss(Z1_transformed, Z2, flow):
    bs = int(shape[0])
    h = int(shape[1])
    w = int(shape[2])
    # we want to check if Z2 and Z1_transformed are similar
    # but first we have to warp Z2 to the coordinate frame of Z1
    Z2 = tf.reshape(Z2, [bs, h, w, 1])
    Z2_warped, _ = warper(Z2, flow)
    Z_diff = tf.abs(Z1_transformed - Z2_warped)
    xyzl = l1Loss(Z1_transformed, Z2_warped)
    return xyzl


def smoothLoss2D(flow):
    with tf.name_scope("smoothLoss2D"):
        shape = flow.get_shape()
        bs = shape[0]
        h = shape[1]
        w = shape[2]
        kernel = tf.transpose(tf.constant([[[[0, 0, 0], [0, 1, -1], [0, 0, 0]]],
                                           [[[0, 0, 0], [0, 1, 0], [0, -1, 0]]]],
                                          dtype=tf.float32), perm=[3, 2, 1, 0],
                              name="kernel")
        [u, v] = tf.unstack(flow, axis=3)
        u = tf.expand_dims(u, 3, name="u")
        v = tf.expand_dims(v, 3, name="v")
        diff_u = tf.nn.conv2d(u, kernel, [1, 1, 1, 1], padding="SAME", name="diff_u")
        diff_v = tf.nn.conv2d(v, kernel, [1, 1, 1, 1], padding="SAME", name="diff_v")
        diffs = tf.concat(axis=3, values=[diff_u, diff_v], name="diffs")

        # make mask with ones everywhere but the bottom and right borders
        mask = tf.ones([bs, h - 1, w - 1, 1], name="mask")
        mask = tf.concat(axis=1, values=[mask, tf.zeros([bs, 1, w - 1, 1])], name="mask2")
        mask = tf.concat(axis=2, values=[mask, tf.zeros([bs, h, 1, 1])], name="mask3")
        loss = tf.reduce_mean(tf.abs(diffs * mask), name="loss")
        return loss


def smoothLoss1D(u):
    with tf.name_scope("smoothLoss1D"):
        shape = u.get_shape()
        bs, h, w, c = shape
        kernel = tf.transpose(tf.constant([[[[0, 0, 0], [0, 1, -1], [0, 0, 0]]],
                                           [[[0, 0, 0], [0, 1, 0], [0, -1, 0]]]],
                                          dtype=tf.float32), perm=[3, 2, 1, 0],
                              name="kernel")
        diff = tf.nn.conv2d(u, kernel, [1, 1, 1, 1], padding="SAME", name="diff")

        # make mask with ones everywhere but the bottom and right borders
        mask = tf.ones([bs, h - 1, w - 1, 1], name="mask")
        mask = tf.concat(axis=1, values=[mask, tf.zeros([bs, 1, w - 1, 1])], name="mask2")
        mask = tf.concat(axis=2, values=[mask, tf.zeros([bs, h, 1, 1])], name="mask3")
        loss = tf.reduce_mean(tf.abs(diff * mask), name="loss")
        return loss


def smooth_loss(u, valid=None):
    # u can be any number of channels
    with tf.name_scope("smooth_loss"):
        shape = u.get_shape()
        bs, h, w, c = shape
        kernel = tf.transpose(tf.constant([[[[0, 0, 0], [0, 1, -1], [0, 0, 0]]],
                                           [[[0, 0, 0], [0, 1, 0], [0, -1, 0]]]],
                                          dtype=tf.float32), perm=[3, 2, 1, 0],
                              name="kernel")
        kernel = tf.tile(kernel, [1, 1, int(c), 1])
        diff = tf.nn.conv2d(u, kernel, [1, 1, 1, 1], padding="SAME", name="diff")

        # make mask with ones everywhere but the bottom and right borders
        mask = tf.ones([bs, h - 1, w - 1, 1])
        mask = tf.concat(axis=1, values=[mask, tf.zeros([bs, 1, w - 1, 1])])
        mask = tf.concat(axis=2, values=[mask, tf.zeros([bs, h, 1, 1])])
        if valid is not None:
            mask = mask * valid
        loss = tf.reduce_sum(tf.abs(diff * mask)) / tf.reduce_sum(mask + EPS)
        return loss


def masked_smoothLoss_multichan(u, valid):
    with tf.name_scope("smoothLoss_multichan"):
        shape = u.get_shape()
        bs, h, w, d = shape
        kernel = tf.transpose(tf.constant([[[[0, 0, 0], [0, 1, -1], [0, 0, 0]]],
                                           [[[0, 0, 0], [0, 1, 0], [0, -1, 0]]]],
                                          dtype=tf.float32), perm=[3, 2, 1, 0],
                              name="kernel")
        kernel = tf.tile(kernel, [1, 1, int(d), 1])
        diff = tf.nn.conv2d(u, kernel, [1, 1, 1, 1], padding="SAME", name="diff")

        # make mask with ones everywhere but the bottom and right borders
        mask = tf.ones([bs, h - 1, w - 1, 1], name="mask")
        mask = tf.concat(axis=1, values=[mask, tf.zeros([bs, 1, w - 1, 1])], name="mask2")
        mask = tf.concat(axis=2, values=[mask, tf.zeros([bs, h, 1, 1])], name="mask3")
        mask = mask * valid
        loss = tf.reduce_sum(tf.abs(diff * mask)) / tf.reduce_sum(mask + EPS)
        return loss


def masked_smoothLoss1D(u, valid):
    with tf.name_scope("masked_smoothLoss1D"):
        shape = u.get_shape()
        bs = shape[0]
        h = shape[1]
        w = shape[2]
        kernel = tf.transpose(tf.constant([[[[0, 0, 0], [0, 1, -1], [0, 0, 0]]],
                                           [[[0, 0, 0], [0, 1, 0], [0, -1, 0]]]],
                                          dtype=tf.float32), perm=[3, 2, 1, 0],
                              name="kernel")
        diff = tf.nn.conv2d(u, kernel, [1, 1, 1, 1], padding="SAME", name="diff")

        # make mask with ones everywhere but the bottom and right borders
        mask = tf.ones([bs, h - 1, w - 1, 1], name="mask")
        mask = tf.concat(axis=1, values=[mask, tf.zeros([bs, 1, w - 1, 1])], name="mask2")
        mask = tf.concat(axis=2, values=[mask, tf.zeros([bs, h, 1, 1])], name="mask3")
        mask = mask * valid
        loss = tf.reduce_sum(tf.abs(diff * mask)) / tf.reduce_sum(mask + EPS)
        return loss


def canon_loss(XYZ, mask, y0, x0):
    with tf.variable_scope("canon_loss"):
        shape = XYZ.get_shape()
        bs, hw, _ = shape
        print_shape(XYZ)
        print_shape(y0)
        print_shape(x0)

        mask_flat = tf.reshape(tf.tile(mask, [1, 1, 1, 3]), [bs, -1, 3])
        [X, Y, Z] = tf.split(axis=2, num_or_size_splits=3, value=XYZ * mask_flat)

        y0 = tf.tile(tf.expand_dims(y0, 1), [1, hw])
        x0 = tf.tile(tf.expand_dims(x0, 1), [1, hw])
        fy = tf.reshape(fy, [bs, -1, 1])
        fx = tf.reshape(fx, [bs, -1, 1])
        y0 = tf.reshape(y0, [bs, -1, 1])
        x0 = tf.reshape(x0, [bs, -1, 1])

        loss_x = tf.reduce_sum(tf.abs(X - x0)) / tf.reduce_sum(mask + EPS)
        loss_y = tf.reduce_sum(tf.abs(Y - y0)) / tf.reduce_sum(mask + EPS)

        l = loss_x + loss_y
        return l


def com_centroid_loss(Z, com, mask, fy, fx):
    with tf.variable_scope("com_centroid_loss"):
        shape = Z.get_shape()
        bs = int(shape[0])
        h = int(shape[1])
        w = int(shape[2])

        # create a pointcloud for the scene
        [grid_x, grid_y] = meshgrid2D(bs, h, w)
        Z = tf.reshape(Z, [bs, h, w], name="Z")
        XYZ = Camera2World_p(grid_x, grid_y, Z, fx, fy)

        # move every point (on the toy, at least) to the com
        XYZt = XYZ + com

        # get the dist from the com to the centroid of the toy points
        mask = tf.reshape(tf.tile(mask, [1, 1, 1, 3]), [bs, h * w, 3])
        XYZ_masked = XYZ * mask
        XYZt_masked = XYZt * mask

        l1 = tf.reduce_sum(tf.abs(XYZ_masked - XYZt_masked)) / (EPS + tf.reduce_sum(mask))

        return l1


def com_spread_loss(Z, com, mask, fy, fx):
    with tf.variable_scope("com_spread_loss"):
        shape = Z.get_shape()
        bs = int(shape[0])
        h = int(shape[1])
        w = int(shape[2])

        # create a pointcloud for the scene
        [grid_x, grid_y] = meshgrid2D(bs, h, w)
        Z = tf.reshape(Z, [bs, h, w], name="Z")
        XYZ = Camera2World_p(grid_x, grid_y, Z, fx, fy)

        # move every point (on the toy, at least) to the com
        XYZt = XYZ + com
        [Xt, Yt, Zt] = tf.split(axis=2, num_or_size_splits=3, value=XYZt)

        # get the dist from the com to the centroid of the toy points
        # mask = tf.reshape(tf.tile(mask,[1,1,1,3]),[1,h*w,3])
        # XYZ_masked = XYZ*mask
        # XYZt_masked = XYZt*mask

        mask = tf.reshape(mask, [bs, h * w, 1])
        # grab the XYZ's on the toy
        # indices = tf.where(tf.greater(mask, 0.5))
        # toy_Xt = tf.gather_nd(Xt, indices)
        # toy_Yt = tf.gather_nd(Yt, indices)
        # toy_Zt = tf.gather_nd(Zt, indices)

        # toy_Xt = tf.SparseTensor(indices, toy_Xt, toy_Xt.get_shape())
        # toy_Yt = tf.SparseTensor(indices, toy_Yt, toy_Yt.get_shape())
        # toy_Zt = tf.SparseTensor(indices, toy_Zt, toy_Zt.get_shape())

        # # print_shape(toy_Xt)

        # vXt = tf.nn.moments(toy_Xt, axes=[0])
        # vYt = tf.nn.moments(toy_Yt, axes=[0])
        # vZt = tf.nn.moments(toy_Zt, axes=[0])
        # loss = tf.reduce_mean(vXt + vYt + vZt)

        binary_mask = tf.where(tf.greater(mask, 0.5), tf.ones_like(mask), tf.zeros_like(mask))
        mean_x = tf.reduce_sum(Xt * binary_mask) / (EPS + tf.reduce_sum(binary_mask))
        mean_y = tf.reduce_sum(Yt * binary_mask) / (EPS + tf.reduce_sum(binary_mask))
        mean_z = tf.reduce_sum(Zt * binary_mask) / (EPS + tf.reduce_sum(binary_mask))

        mean_diff_x = tf.reduce_sum(Xt - mean_x) / (EPS + tf.reduce_sum(binary_mask))
        mean_diff_y = tf.reduce_sum(Yt - mean_y) / (EPS + tf.reduce_sum(binary_mask))
        mean_diff_z = tf.reduce_sum(Zt - mean_z) / (EPS + tf.reduce_sum(binary_mask))

        loss = tf.sqrt(tf.square(mean_diff_x) + tf.square(mean_diff_y) + tf.square(mean_diff_z))
        return loss


def generalized_loss(truth, pred, alpha, c, sum_indices, avg_indices, mask=None):
    # https://arxiv.org/pdf/1701.03077.pdf
    # alpha = 2 is l2 loss
    # alpha = 1 is huber loss?
    # alpha = 0 is cauchy loss
    # alpha = -inf is welsch loss
    assert c > 0
    assert c < float('inf')

    x = truth - pred
    invc = 1.0 / c
    xinvcsq = tf.square(x * invc)
    if alpha == 0.0:
        l = tf.log(xinvcsq * 0.5 + 1.0)
    elif alpha == -float('inf'):
        l = 1.0 - tf.exp(-0.5 * xinvcsq)
    else:
        zalpha = max(1, 2 - alpha)
        l = zalpha / alpha * (tf.pow(xinvcsq / zalpha + 1.0, alpha * 0.5) - 1.0)
    if mask:
        l *= mask
        l = (tf.reduce_sum(l, avg_indices, keep_dims=True) /
             (tf.reduce_sum(mask, avg_indices, keep_dims=True) + EPS))
    else:
        l = tf.reduce_mean(l, avg_indices, keep_dims=True)
    l = tf.reduce_sum(l, sum_indices, keep_dims=True)
    l = tf.squeeze(l)
    return l


def binaryloss(pred, gt):
    gtone = gt
    gtzero = 1.0 - gt
    l = -(gtone * tf.log(pred + EPS) + gtzero * tf.log(1.0 - pred + EPS))
    l = tf.reduce_mean(l)
    return l
