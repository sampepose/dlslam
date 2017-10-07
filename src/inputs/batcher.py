import tensorflow as tf

import os
import sys
sys.path.append('..')

from src.utils import print_shape, random_crop, encode_depth, ominus
from src.depth_to_normals import surface_normals
from src.inputs.readers import *
import lmbspecialops as sops

def blendswap_batch(dataset, hyp, shuffle=True):
    with tf.device('/cpu:0'):
        with open(dataset) as f:
            content = f.readlines()
        records = [hyp.dataset_location + line.strip() for line in content]
        nRecords = len(records)
        print ('found %d records' % nRecords)
        for record in records:
            assert os.path.isfile(record), 'Record at %s was not found' % record

        queue = tf.train.string_input_producer(records, shuffle=shuffle)

        (h, w, i1, i2, d1, f12, relativeTranslation1to2, relativeRotation1to2) = read_and_decode_blendswap(queue)

        i1 = tf.cast(i1, tf.float32) * 1. / 255 - 0.5
        i2 = tf.cast(i2, tf.float32) * 1. / 255 - 0.5
        d1 = tf.cast(d1, tf.float32)

        demon_height = 192
        demon_width = 256
        demon_fx = 0.89115971
        demon_fy = 1.18821287
        fx = 0.46875
        fy = 0.8333333333

        # Calculate crop width/height given (d_fx)(d_w) = (s_x)(s_w) relationship (same for height)
        crop_width = int(round((demon_fx * demon_width) / fx))  # 487
        crop_height = int(round((demon_fy * demon_height) / fy))  # 274

        # image tensors need to be cropped. we'll do them all at once.
        allCat = tf.concat(axis=2, values=[i1, i2,
                                           d1,
                                           f12],
                           name="allCat")

        # image tensors need to be cropped. we'll do them all at once.
        print_shape(allCat)
        allCat_crop, off_h, off_w = random_crop(
            allCat, crop_height, crop_width, h, w)
        print_shape(allCat_crop)

        # We need to reshape the crop to match the demon dimensions of 256 x 192
        allCat_crop = tf.image.resize_images(allCat_crop, [demon_height, demon_width])

        # Split out each channel properly
        i1 = tf.slice(allCat_crop, [0, 0, 0], [-1, -1, 3], name="i1")
        i2 = tf.slice(allCat_crop, [0, 0, 3], [-1, -1, 3], name="i2")
        d1 = tf.slice(allCat_crop, [0, 0, 6], [-1, -1, 1], name="d1")
        f12 = tf.slice(allCat_crop, [0, 0, 7], [-1, -1, 2], name="f1")

        # Normalize flow so displacement by the image size corresponds to 1
        # f12 = f12 / [demon_width, demon_height]

        # Normalize translation so ||t||2 == 1
        t_norm = tf.norm(relativeTranslation1to2)
        relativeTranslation1to2 = relativeTranslation1to2 / t_norm

        # Which means we need to scale the depth by the same factor
        d1 = d1 / t_norm

        # Encode depth (we want inverse depth)
        d1 = encode_depth(d1, hyp.depth_encoding)  # encode

        """ Compute normals from gt depth """
        depth_resize = tf.image.resize_images(tf.expand_dims(
            d1, 0), [48, 64], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        depth_resize_nchw = tf.transpose(depth_resize, perm=[0, 3, 1, 2])
        # Use intrinsics of demon
        intrinsics = tf.constant([[0.89115971, 1.18821287, 0.5, 0.5]], dtype=tf.float32)
        normals_from_downsampled = sops.depth_to_normals(
            depth_resize_nchw, intrinsics, inverse_depth=True)

        # unsure why, but ~2% of values are NaN
        normals_from_downsampled = tf.where(tf.is_nan(normals_from_downsampled), tf.zeros_like(
            normals_from_downsampled), normals_from_downsampled)

        # Remove BS dimension and transpose back to NHWC
        normals_from_downsampled = tf.transpose(tf.squeeze(normals_from_downsampled), perm=[1, 2, 0])

        """
        i1: image_1
        i2: image_2
        d1: depth_1
        f12: flow 1 -> 2
        v1: valid flow map 1
        r_rel: relative camera rotation from p1 to p2
        t_rel: relative camera translation from p1 to p2
        """
        batch = tf.train.batch([i1, i2,
                                d1,
                                f12,
                                normals_from_downsampled,
                                relativeRotation1to2, relativeTranslation1to2,
                                off_h, off_w],
                               num_threads=1,
                               batch_size=hyp.bs,
                               dynamic_pad=True)
        return batch


def svkitti_batch(dataset, hyp, shuffle=True):
    with open(dataset) as f:
        content = f.readlines()
    records = [hyp.dataset_location + line.strip() for line in content]
    nRecords = len(records)
    print ('found %d records' % nRecords)
    for record in records:
        assert os.path.isfile(record), 'Record at %s was not found' % record

    queue = tf.train.string_input_producer(records, shuffle=shuffle)

    (h, w, i1, i2, d1, d2, f12, f23, v1, v2, p1, p2, m1, m2) = read_and_decode_svkitti(queue)

    i1 = tf.cast(i1, tf.float32) * 1. / 255 - 0.5
    i2 = tf.cast(i2, tf.float32) * 1. / 255 - 0.5
    d1 = tf.cast(d1, tf.float32)
    d2 = tf.cast(d2, tf.float32)
    v1 = tf.cast(v1, tf.float32)  # 1 at non-sky pixels
    v2 = tf.cast(v2, tf.float32)
    m1 = tf.cast(m1, tf.float32) * 1. / 255  # these are stored in [0,255], and 255 means moving.
    m2 = tf.cast(m2, tf.float32) * 1. / 255
    # d1 = d1*v1 # put 0 depth at invalid spots
    # d2 = d2*v2
    d1 = encode_depth(d1, hyp.depth_encoding)  # encode
    d2 = encode_depth(d2, hyp.depth_encoding)

    # image tensors need to be cropped. we'll do them all at once.
    allCat = tf.concat(axis=2, values=[i1, i2,
                                       d1, d2,
                                       f12, f23,
                                       v1, v2,
                                       m1, m2],
                       name="allCat")

    # image tensors need to be cropped. we'll do them all at once.
    print_shape(allCat)
    allCat_crop, off_h, off_w = random_crop(allCat, hyp.h, hyp.w, h, w)
    print_shape(allCat_crop)
    i1 = tf.slice(allCat_crop, [0, 0, 0], [-1, -1, 3], name="i1")
    i2 = tf.slice(allCat_crop, [0, 0, 3], [-1, -1, 3], name="i2")
    d1 = tf.slice(allCat_crop, [0, 0, 6], [-1, -1, 1], name="d1")
    d2 = tf.slice(allCat_crop, [0, 0, 7], [-1, -1, 1], name="d2")
    f12 = tf.slice(allCat_crop, [0, 0, 8], [-1, -1, 2], name="f12")
    f23 = tf.slice(allCat_crop, [0, 0, 10], [-1, -1, 2], name="f23")
    v1 = tf.slice(allCat_crop, [0, 0, 12], [-1, -1, 1], name="v1")
    v2 = tf.slice(allCat_crop, [0, 0, 13], [-1, -1, 1], name="v2")
    m1 = tf.slice(allCat_crop, [0, 0, 14], [-1, -1, 1], name="m1")
    m2 = tf.slice(allCat_crop, [0, 0, 15], [-1, -1, 1], name="m2")

    """
    i1: image_1
    i2: image_2
    d1: depth_1
    d2: depth_2
    f12: flow 1 -> 2
    f23: ?????
    v1: valid flow map 1
    v2: valid flow map 2
    p1: camera pose 1
    p2: camera pose 2
    m1: motion mask 1 (1 at moving objects)
    m2: motion mask 2
    """
    batch = tf.train.batch([i1, i2,
                            d1, d2,
                            f12, f23,
                            v1, v2,
                            p1, p2,
                            m1, m2,
                            off_h, off_w],
                           batch_size=hyp.bs,
                           dynamic_pad=True)
    return batch


# def rotation_from_matrix(R):
#     """ https://en.wikipedia.org/wiki/Rotation_matrix#Rotation_matrix_from_axis_and_angle """
#     b = R[0, 1]
#     c = R[0, 2]
#     d = R[1, 0]
#     f = R[1, 2]
#     g = R[2, 0]
#     h = R[2, 1]
#     axis = tf.stack([h - f, c - g, d - b])
#     angle = tf.acos(tf.clip_by_value(0.5 * (tf.trace(R) - 1.0), -1.0, 1.0))
#     return axis, angle

def rotation_from_matrix(R):
    return tf.py_func(py_rotation_from_matrix, [R], tf.float32)


import numpy as np
import math


def mat2quat(M):
    ''' Calculate quaternion corresponding to given rotation matrix

    Parameters
    ----------
    M : array-like
      3x3 rotation matrix

    Returns
    -------
    q : (4,) array
      closest quaternion to input matrix, having positive q[0]

    Notes
    -----
    Method claimed to be robust to numerical errors in M

    Constructs quaternion by calculating maximum eigenvector for matrix
    K (constructed from input `M`).  Although this is not tested, a
    maximum eigenvalue of 1 corresponds to a valid rotation.

    A quaternion q*-1 corresponds to the same rotation as q; thus the
    sign of the reconstructed quaternion is arbitrary, and we return
    quaternions with positive w (q[0]).

    References
    ----------
    * http://en.wikipedia.org/wiki/Rotation_matrix#Quaternion
    * Bar-Itzhack, Itzhack Y. (2000), "New method for extracting the
      quaternion from a rotation matrix", AIAA Journal of Guidance,
      Control and Dynamics 23(6):1085-1087 (Engineering Note), ISSN
      0731-5090

    Examples
    --------
    >>> import numpy as np
    >>> q = mat2quat(np.eye(3)) # Identity rotation
    >>> np.allclose(q, [1, 0, 0, 0])
    True
    >>> q = mat2quat(np.diag([1, -1, -1]))
    >>> np.allclose(q, [0, 1, 0, 0]) # 180 degree rotn around axis 0
    True

    '''
    # Qyx refers to the contribution of the y input vector component to
    # the x output vector component.  Qyx is therefore the same as
    # M[0,1].  The notation is from the Wikipedia article.
    Qxx, Qyx, Qzx, Qxy, Qyy, Qzy, Qxz, Qyz, Qzz = M.flat
    # Fill only lower half of symmetric matrix
    K = np.array([
        [Qxx - Qyy - Qzz, 0,               0,               0],
        [Qyx + Qxy,       Qyy - Qxx - Qzz, 0,               0],
        [Qzx + Qxz,       Qzy + Qyz,       Qzz - Qxx - Qyy, 0],
        [Qyz - Qzy,       Qzx - Qxz,       Qxy - Qyx,       Qxx + Qyy + Qzz]]
    ) / 3.0
    # Use Hermitian eigenvectors, values for speed
    vals, vecs = np.linalg.eigh(K)
    # Select largest eigenvector, reorder to w,x,y,z quaternion
    q = vecs[[3, 0, 1, 2], np.argmax(vals)]
    # Prefer quaternion with positive w
    # (q * -1 corresponds to same rotation as q)
    if q[0] < 0:
        q *= -1
    return q


def quat2angle_axis(quat, identity_thresh=None):
    ''' Convert quaternion to rotation of angle around axis

    Parameters
    ----------
    quat : 4 element sequence
       w, x, y, z forming quaternion
    identity_thresh : None or scalar, optional
       threshold below which the norm of the vector part of the
       quaternion (x, y, z) is deemed to be 0, leading to the identity
       rotation.  None (the default) leads to a threshold estimated
       based on the precision of the input.

    Returns
    -------
    theta : scalar
       angle of rotation
    vector : array shape (3,)
       axis around which rotation occurs

    Examples
    --------
    >>> theta, vec = quat2angle_axis([0, 1, 0, 0])
    >>> np.allclose(theta, np.pi)
    True
    >>> vec
    array([ 1.,  0.,  0.])

    If this is an identity rotation, we return a zero angle and an
    arbitrary vector

    >>> quat2angle_axis([1, 0, 0, 0])
    (0.0, array([ 1.,  0.,  0.]))

    Notes
    -----
    A quaternion for which x, y, z are all equal to 0, is an identity
    rotation.  In this case we return a 0 angle and an  arbitrary
    vector, here [1, 0, 0]
    '''
    w, x, y, z = quat
    vec = np.asarray([x, y, z])
    if identity_thresh is None:
        try:
            identity_thresh = np.finfo(vec.dtype).eps * 3
        except ValueError:  # integer type
            identity_thresh = FLOAT_EPS * 3
    n = math.sqrt(x * x + y * y + z * z)
    if n < identity_thresh:
        # if vec is nearly 0,0,0, this is an identity rotation
        return 0.0, np.array([1.0, 0, 0])
    return (vec / n) * (2 * math.acos(w))


def py_rotation_from_matrix(matrix):
    """Return rotation angle and axis from rotation matrix.

    >>> angle = (random.random() - 0.5) * (2*math.pi)
    >>> direc = np.random.random(3) - 0.5
    >>> point = np.random.random(3) - 0.5
    >>> R0 = rotation_matrix(angle, direc, point)
    >>> angle, direc, point = rotation_from_matrix(R0)
    >>> R1 = rotation_matrix(angle, direc, point)
    >>> is_same_transform(R0, R1)
    True

    """
    return np.float32(quat2angle_axis(mat2quat(matrix)))


def svkitti_batch_demon(dataset, hyp, shuffle=True, center_crop=False, include_poses=False):
    with tf.device('/cpu:0'):
        with open(dataset) as f:
            content = f.readlines()
        records = [hyp.dataset_location + line.strip() for line in content]
        nRecords = len(records)
        print ('found %d records' % nRecords)
        # for record in records:
        #     assert os.path.isfile(record), 'Record at %s was not found' % record

        queue = tf.train.string_input_producer(records, shuffle=shuffle)

        (h, w, i1, i2, d1, d2, f12, f23, v1, v2, p1, p2, m1, m2) = read_and_decode_svkitti(queue)

        i1 = tf.cast(i1, tf.float32) * 1. / 255 - 0.5
        i2 = tf.cast(i2, tf.float32) * 1. / 255 - 0.5
        d1 = tf.cast(d1, tf.float32)
        d2 = tf.cast(d2, tf.float32)
        v1 = tf.cast(v1, tf.float32)  # 1 at non-sky pixels
        v2 = tf.cast(v2, tf.float32)
        # these are stored in [0,255], and 255 means moving.
        m1 = tf.cast(m1, tf.float32) * 1. / 255
        m2 = tf.cast(m2, tf.float32) * 1. / 255
        # d1 = d1*v1 # put 0 depth at invalid spots
        # d2 = d2*v2

        demon_height = 192
        demon_width = 256
        svkitti_height = 375
        svkitti_width = 1242
        demon_fx = 0.89115971
        demon_fy = 1.18821287
        svkitti_fx = 725. / svkitti_width
        svkitti_fy = 725. / svkitti_height

        # Calculate crop width/height given (d_fx)(d_w) = (s_x)(s_w) relationship (same for height)
        crop_width = int(round((demon_fx * demon_width) / svkitti_fx))  # 390.8
        crop_height = int(round((demon_fy * demon_height) / svkitti_fy))  # 118.0

        # f12 /= [svkitti_width, svkitti_height]
        # f23 /= [svkitti_width, svkitti_height]

        # image tensors need to be cropped. we'll do them all at once.
        allCat = tf.concat(axis=2, values=[i1, i2,
                                           d1, d2,
                                           f12, f23,
                                           v1, v2,
                                           m1, m2],
                           name="allCat")

        # image tensors need to be cropped. we'll do them all at once.
        if center_crop:
            off_h = ((h - crop_height - 1) / tf.constant(2))
            off_w = ((w - crop_width - 1) / tf.constant(2))
            allCat_crop = tf.slice(allCat, [off_h, off_w, 0],[crop_height, crop_width, -1], name="cropped_tensor")
        else:
            print_shape(allCat)
            allCat_crop, off_h, off_w = random_crop(
                allCat, crop_height, crop_width, h, w)
            print_shape(allCat_crop)

        # We need to reshape the crop to match the demon dimensions of 256 x 192
        allCat_crop = tf.image.resize_images(allCat_crop, [demon_height, demon_width])

        # Split out each channel properly
        i1 = tf.slice(allCat_crop, [0, 0, 0], [-1, -1, 3], name="i1")
        i2 = tf.slice(allCat_crop, [0, 0, 3], [-1, -1, 3], name="i2")
        d1 = tf.slice(allCat_crop, [0, 0, 6], [-1, -1, 1], name="d1")
        d2 = tf.slice(allCat_crop, [0, 0, 7], [-1, -1, 1], name="d2")
        f12 = tf.slice(allCat_crop, [0, 0, 8], [-1, -1, 2], name="f12")
        f23 = tf.slice(allCat_crop, [0, 0, 10], [-1, -1, 2], name="f23")
        v1 = tf.slice(allCat_crop, [0, 0, 12], [-1, -1, 1], name="v1")
        v2 = tf.slice(allCat_crop, [0, 0, 13], [-1, -1, 1], name="v2")
        m1 = tf.slice(allCat_crop, [0, 0, 14], [-1, -1, 1], name="m1")
        m2 = tf.slice(allCat_crop, [0, 0, 15], [-1, -1, 1], name="m2")

        # Normalize flow so displacement by the image size corresponds to 1
        f12 = f12 / [demon_width, demon_height]
        f23 = f23 / [demon_width, demon_height]

        """ Calculate relative camera motion from pose 1 to pose 2 """
        # Note: We use negative poses as vkitti coordinate sysem is not what demon is trained on:
        # vkitti: +x points right, +y points down, +z points forwards
        # demon: +x points left, +y points down, +z points forwards
        # transformation = tf.constant([[-1., 0., 0., 0.,],
        #                               [0., 1., 0., 0.,],
        #                               [0., 0., 1., 0.,],
        #                               [0., 0., 0., 1.,]], dtype=tf.float32)
        # p1 = tf.matmul(tf.matmul(transformation, p1), transformation)
        # p2 = tf.matmul(tf.matmul(transformation, p2), transformation)
        rel_rt = ominus(tf.expand_dims(p2, 0), tf.expand_dims(p1, 0))[0, ...]
        rel_r = rel_rt[0:3, 0:3]
        rel_t = rel_rt[0:3, 3]

        # Important!! Convert from pose to extrinsic matrix
        rel_r = tf.matrix_transpose(rel_r)
        rel_t = tf.matmul(-rel_r, tf.expand_dims(rel_t, 1))[:, 0]

        # Convert rotation matrix to rotation about axis (norm encodes angle of rotation)
        rel_r = rotation_from_matrix(rel_r)
        rel_r.set_shape([3, ])

        # Normalize translation so ||t||2 == 1
        t_norm = tf.norm(rel_t)
        rel_t = rel_t / t_norm

        # Which means we need to scale the depth by the same factor
        d1 = d1 / t_norm
        d2 = d2 / t_norm

        """ Encode depth (we want inverse depth) """
        d1 = encode_depth(d1, hyp.depth_encoding)
        d2 = encode_depth(d2, hyp.depth_encoding)

        """ Compute normals from gt depth """
        depth_resize = tf.image.resize_images(tf.expand_dims(
            d1, 0), [48, 64], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        depth_nchw = tf.expand_dims(tf.transpose(d1, perm=[2, 0, 1]), 0)
        depth_resize_nchw = tf.transpose(depth_resize, perm=[0, 3, 1, 2])
        # Use intrinsics of demon
        intrinsics = tf.constant([[0.89115971, 1.18821287, 0.5, 0.5]], dtype=tf.float32)
        normals = sops.depth_to_normals(depth_nchw, intrinsics, inverse_depth=True)
        normals_from_downsampled = sops.depth_to_normals(
            depth_resize_nchw, intrinsics, inverse_depth=True)

        # unsure why, but ~2% of values are NaN
        normals = tf.where(tf.is_nan(normals), tf.zeros_like(normals), normals)
        normals_from_downsampled = tf.where(tf.is_nan(normals_from_downsampled), tf.zeros_like(
            normals_from_downsampled), normals_from_downsampled)

        # Remove BS dimension and transpose back to NHWC
        normals = tf.transpose(tf.squeeze(normals), perm=[1, 2, 0])
        normals_from_downsampled = tf.transpose(tf.squeeze(normals_from_downsampled), perm=[1, 2, 0])

        """
        i1: image_1
        i2: image_2
        d1: depth_1
        f12: flow 1 -> 2
        v1: valid flow map 1
        rel_r: relative camera rotation from p1 to p2
        rel_t: relative camera translation from p1 to p2
        """
        batch = tf.train.batch([i1, i2,
                                d1,
                                d2,
                                f12,
                                f23,
                                v1,
                                normals,
                                normals_from_downsampled,
                                rel_r, rel_t,
                                p1, p2,
                                m1,
                                off_h, off_w],
                               num_threads=16,
                               batch_size=hyp.bs,
                               dynamic_pad=True)
        return batch


def euromav_batch(dataset, hyp, shuffle=True, crop=True):
    with open(dataset) as f:
        content = f.readlines()
    records = [hyp.dataset_location + line.strip() for line in content]
    nRecords = len(records)
    print ('found %d records' % nRecords)
    for record in records:
        assert os.path.isfile(record), 'Record at %s was not found' % record

    queue = tf.train.string_input_producer(records, shuffle=shuffle)

    (h, w, i1_l, i2_l, i1_r, i2_r, p1, p2) = read_and_decode_euromav(queue)

    # Scale images to [-0.5, 0.5]
    i1_l = tf.cast(i1_l, tf.float32) * 1. / 255 - 0.5
    i2_l = tf.cast(i2_l, tf.float32) * 1. / 255 - 0.5
    i1_r = tf.cast(i1_r, tf.float32) * 1. / 255 - 0.5
    i2_r = tf.cast(i2_r, tf.float32) * 1. / 255 - 0.5

    old_w = 752
    old_h = 480
    new_w = 374
    new_h = 240

    top_left_x = old_w // 2 - new_w // 2
    top_left_y = old_h // 2 - new_h // 2

    if crop:
        # Center crop images
        i1_l = tf.image.crop_to_bounding_box(i1_l, top_left_y, top_left_x, new_h, new_w)
        i2_l = tf.image.crop_to_bounding_box(i2_l, top_left_y, top_left_x, new_h, new_w)

    # Grayscale to RGB
    i1_l = tf.tile(i1_l, [1, 1, 3])
    i2_l = tf.tile(i2_l, [1, 1, 3])

    # d1 = d1*v1 # put 0 depth at invalid spots
    # d2 = d2*v2
    # image tensors need to be cropped. we'll do them all at once.
    # allCat = tf.concat(axis=2,values=[i1,i2],
    #                    name="allCat")

    # # image tensors need to be cropped. we'll do them all at once.
    # print_shape(allCat)
    off_h = 0
    off_w = 0
    # allCat = tf.slice(allCat,[off_h,off_w,0],[-1,-1,-1],name="cropped_tensor")
    # # allCat_crop, off_h, off_w = random_crop(allCat,crop_h,crop_w,480,752)
    # print_shape(allCat)
    # i1 = tf.slice(allCat, [0,0,0], [-1,-1,1], name="i1")
    # i2 = tf.slice(allCat, [0,0,1], [-1,-1,1], name="i2")

    batch = tf.train.batch([i1_l, i2_l,
                            i1_r, i2_r,
                            p1, p2],
                           batch_size=hyp.bs,
                           dynamic_pad=True)
    return batch
