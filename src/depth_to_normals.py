#!/usr/bin/env python2

from scipy.misc import imsave
import numpy as np
import tensorflow as tf

from utils import meshgrid2D, print_shape, decode_depth, Camera2World


def surface_normals(hyp, depth, valid, fy, fx, y0, x0):
    '''
    input: BxHxWx1
    output: BxHxWx3 #xyz
    '''
    with tf.name_scope("surface_normals"):
        shape = depth.get_shape()
        bs = int(shape[0])
        h = int(shape[1])
        w = int(shape[2])

        # blur_kernel = tf.transpose(tf.constant([[[[1./16,1./8,1./16],
        #                                           [1./8,1./4,1./8],
        #                                           [1./16,1./8,1./16]]]],
        #                                        dtype=tf.float32),perm=[3,2,1,0])
        # depth = tf.nn.conv2d(depth, blur_kernel, strides=[1,1,1,1], padding="SAME")
        kernel = tf.transpose(tf.constant([[[[0, 0, 0], [-1, 0, 1], [0, 0, 0]]],
                                           [[[0, -1, 0], [0, 0, 0], [0, 1, 0]]]],
                                          dtype=tf.float32), perm=[3, 2, 1, 0], name="kernel")

        Z = decode_depth(depth, hyp.depth_encoding)
        diff = tf.nn.conv2d(Z, kernel, [1, 1, 1, 1], padding="SAME", name="diff")

        [x, y] = meshgrid2D(bs, h, w)
        # x = tf.expand_dims(x,3)
        # y = tf.expand_dims(y,3)

        z = tf.squeeze(Z, axis=3)

        print_shape(Z)
        print_shape(x)

        # for "minus", we pad on the LEFT, so that x-1 goes to the x spot.
        x_m = tf.pad(x, [[0, 0], [0, 0], [1, 0]], hyp.pad)
        # for "plus", we pad on the RIGHT
        x_p = tf.pad(x, [[0, 0], [0, 0], [0, 1]], hyp.pad)
        # similar for the y direction
        y_m = tf.pad(y, [[0, 0], [1, 0], [0, 0]], hyp.pad)
        y_p = tf.pad(y, [[0, 0], [0, 1], [0, 0]], hyp.pad)

        z_x_m = tf.pad(z, [[0, 0], [0, 0], [1, 0]], hyp.pad)
        z_x_p = tf.pad(z, [[0, 0], [0, 0], [0, 1]], hyp.pad)
        z_y_m = tf.pad(z, [[0, 0], [1, 0], [0, 0]], hyp.pad)
        z_y_p = tf.pad(z, [[0, 0], [0, 1], [0, 0]], hyp.pad)

        x_m = tf.slice(x_m, [0, 0, 0], [bs, h, w])
        x_p = tf.slice(x_p, [0, 0, 0], [bs, h, w])
        z_x_m = tf.slice(z_x_m, [0, 0, 0], [bs, h, w])
        z_x_p = tf.slice(z_x_p, [0, 0, 0], [bs, h, w])

        y_m = tf.slice(y_m, [0, 0, 0], [bs, h, w])
        y_p = tf.slice(y_p, [0, 0, 0], [bs, h, w])
        z_y_m = tf.slice(z_y_m, [0, 0, 0], [bs, h, w])
        z_y_p = tf.slice(z_y_p, [0, 0, 0], [bs, h, w])

        # A = tf.reshape(Camera2World(y_m,x,z_y_m,fx,fy,x0,y0),[bs,h,w,3])
        # B = tf.reshape(Camera2World(y_p,x,z_y_p,fx,fy,x0,y0),[bs,h,w,3])
        # C = tf.reshape(Camera2World(y,x_m,z_x_m,fx,fy,x0,y0),[bs,h,w,3])
        # D = tf.reshape(Camera2World(y,x_p,z_x_p,fx,fy,x0,y0),[bs,h,w,3])

        A = tf.reshape(Camera2World(x_m, y, z_x_m, fx, fy, x0, y0), [bs, h, w, 3])
        B = tf.reshape(Camera2World(x_p, y, z_x_p, fx, fy, x0, y0), [bs, h, w, 3])
        C = tf.reshape(Camera2World(x, y_m, z_y_m, fx, fy, x0, y0), [bs, h, w, 3])
        D = tf.reshape(Camera2World(x, y_p, z_y_p, fx, fy, x0, y0), [bs, h, w, 3])

        cross = tf.cross(A - B, C - D)

        # [grid_x1,grid_y1] = meshgrid2D(bs, h, w)
        # Z = tf.reshape(Z,[bs,h,w],name="Z")
        # XYZ = Camera2World(grid_x1,grid_y1,Z,fx,fy,x0,y0)
        # XYZ = tf.reshape(XYZ,[bs,h,w,3])
        # print_shape(XYZ)
        # [X,Y,_] = tf.split(3, 3, XYZ)

        # [kernel_y, kernel_x] = tf.split(3, 2, kernel)
        # print_shape(kernel_x)
        # diff_y = tf.nn.conv2d(Y,kernel_y,[1,1,1,1],padding="SAME")
        # diff_x = tf.nn.conv2d(X,kernel_x,[1,1,1,1],padding="SAME")
        # diff_denom = tf.abs(2*tf.concat(3,[diff_y,diff_x]))
        # diff = diff/diff_denom
        # print_shape(diff)

        # normals = tf.concat(3,[-diff,tf.ones([bs,h,w,1])])
        # print_shape(normals)
        normals = cross

        mag = tf.sqrt(hyp.eps + tf.reduce_sum(tf.square(normals), axis=3, keep_dims=True))
        normednormals = normals / mag

        mag_max = tf.reduce_max(mag)
        mag_min = tf.reduce_min(mag)
        mag = tf.tile(mag, [1, 1, 1, 3])

        normals_z = ((normednormals + 1) / 2) * valid
        # normals_z = ((normednormals+1)/2)
        nmax = tf.reduce_max(normals_z)
        nmin = tf.reduce_min(normals_z)
        normals_z = tf.cast(normals_z * 255, tf.uint8)

        # [nX,nY,nZ] = tf.split(3,3,normals_z)
        # normals_z = tf.concat(3,[nY,nX,nZ])

        # nmax = tf.reduce_max(normednormals)
        # nmin = tf.reduce_min(normednormals)
    return normednormals, nmax, nmin, mag, mag_max, mag_min, normals_z


def test():
    from inputs.batcher import svkitti_batch_demon
    from hyperparams import create_hyperparams

    hyp = create_hyperparams()
    # don't encode depth!
    hyp.depth_encoding = 'id'
    hyp.bs = 1

    # Get the batch of data
    image1, image2, depth, flow, valid, normals, r_rel, t_rel, off_h, off_w = svkitti_batch_demon(
        '/projects/katefgroup/datasets/svkitti/val.txt', hyp)

    with tf.Session() as sess:
        coord = tf.train.Coordinator()
        tf.train.start_queue_runners(coord=coord)

        img1_, img2_, depth_, normals_ = sess.run([image1, image2, depth, normals])
        imsave('/tmp/img_0.png', img1_[0, :, :, :])
        imsave('/tmp/img_1.png', img2_[0, :, :, :])
        imsave('/tmp/depth_0.png', np.squeeze(depth_))
        imsave('/tmp/nor_0.png', normals_[0, :, :, :])


if __name__ == '__main__':
    test()
    # test_simple()
