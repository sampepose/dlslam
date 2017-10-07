import numpy as np
import scipy.misc
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
plt.rc('axes', titlesize=8)
plt.rcParams["figure.figsize"] = (12.8, 9.6)


def flow_to_image(flow):
    """
    Convert flow into middlebury color code image
    :param flow: optical flow map
    :return: optical flow image in middlebury color
    """
    UNKNOWN_FLOW_THRESH = 1e7
    u = flow[:, :, 0]
    v = flow[:, :, 1]

    maxu = -999.
    maxv = -999.
    minu = 999.
    minv = 999.

    idxUnknow = (abs(u) > UNKNOWN_FLOW_THRESH) | (abs(v) > UNKNOWN_FLOW_THRESH)
    u[idxUnknow] = 0
    v[idxUnknow] = 0

    maxu = max(maxu, np.max(u))
    minu = min(minu, np.min(u))

    maxv = max(maxv, np.max(v))
    minv = min(minv, np.min(v))

    rad = np.sqrt(u ** 2 + v ** 2)
    maxrad = max(-1, np.max(rad))

    print "max flow: %.4f\nflow range:\nu = %.3f .. %.3f\nv = %.3f .. %.3f" % (maxrad, minu, maxu, minv, maxv)

    u = u / (maxrad + np.finfo(float).eps)
    v = v / (maxrad + np.finfo(float).eps)

    def compute_color(u, v):
        """
        compute optical flow color map
        :param u: optical flow horizontal map
        :param v: optical flow vertical map
        :return: optical flow in color code
        """
        [h, w] = u.shape
        img = np.zeros([h, w, 3])
        nanIdx = np.isnan(u) | np.isnan(v)
        u[nanIdx] = 0
        v[nanIdx] = 0

        def make_color_wheel():
            """
            Generate color wheel according Middlebury color code
            :return: Color wheel
            """
            RY = 15
            YG = 6
            GC = 4
            CB = 11
            BM = 13
            MR = 6

            ncols = RY + YG + GC + CB + BM + MR

            colorwheel = np.zeros([ncols, 3])

            col = 0

            # RY
            colorwheel[0:RY, 0] = 255
            colorwheel[0:RY, 1] = np.transpose(np.floor(255 * np.arange(0, RY) / RY))
            col += RY

            # YG
            colorwheel[col:col + YG, 0] = 255 - np.transpose(np.floor(255 * np.arange(0, YG) / YG))
            colorwheel[col:col + YG, 1] = 255
            col += YG

            # GC
            colorwheel[col:col + GC, 1] = 255
            colorwheel[col:col + GC, 2] = np.transpose(np.floor(255 * np.arange(0, GC) / GC))
            col += GC

            # CB
            colorwheel[col:col + CB, 1] = 255 - np.transpose(np.floor(255 * np.arange(0, CB) / CB))
            colorwheel[col:col + CB, 2] = 255
            col += CB

            # BM
            colorwheel[col:col + BM, 2] = 255
            colorwheel[col:col + BM, 0] = np.transpose(np.floor(255 * np.arange(0, BM) / BM))
            col += + BM

            # MR
            colorwheel[col:col + MR, 2] = 255 - np.transpose(np.floor(255 * np.arange(0, MR) / MR))
            colorwheel[col:col + MR, 0] = 255

            return colorwheel

        colorwheel = make_color_wheel()
        ncols = np.size(colorwheel, 0)

        rad = np.sqrt(u**2 + v**2)

        a = np.arctan2(-v, -u) / np.pi

        fk = (a + 1) / 2 * (ncols - 1) + 1

        k0 = np.floor(fk).astype(int)

        k1 = k0 + 1
        k1[k1 == ncols + 1] = 1
        f = fk - k0

        for i in range(0, np.size(colorwheel, 1)):
            tmp = colorwheel[:, i]
            col0 = tmp[k0 - 1] / 255
            col1 = tmp[k1 - 1] / 255
            col = (1 - f) * col0 + f * col1

            idx = rad <= 1
            col[idx] = 1 - rad[idx] * (1 - col[idx])
            notidx = np.logical_not(idx)

            col[notidx] *= 0.75
            img[:, :, i] = np.uint8(np.floor(255 * col * (1 - nanIdx)))

        return img

    img = compute_color(u, v)

    idx = np.repeat(idxUnknow[:, :, np.newaxis], 3, axis=2)
    img[idx] = 0

    return np.uint8(img)


def fundamental(K, R, t):
    tmp = np.dot(np.dot(K, np.matrix.transpose(R)), t)
    cross_mat = np.array([[0.0, -tmp[2], tmp[1]],
                          [tmp[2], 0.0, -tmp[0]],
                          [-tmp[1], tmp[0], 0.0]])
    K_inv_trans = np.matrix.transpose(np.linalg.inv(K))
    return np.dot(np.dot(np.dot(K_inv_trans, R), np.matrix.transpose(K)), cross_mat)


# Load the inputs
imageA = np.load('image1.npy')
imageB = np.load('image2.npy')
depth = np.load('depth.npy')
flow = np.load('flow.npy')
rel_pose = np.load('rel_pose.npy')

_, height, width = flow.shape
print 'height', height, 'width', width

print 'ImageA shape', imageA.shape, 'min', imageA.min(), 'max', imageA.max()
print 'ImageB shape', imageB.shape, 'min', imageB.min(), 'max', imageB.max()
print 'Depth shape', depth.shape, 'min', depth.min(), 'max', depth.max()
print 'Flow shape', flow.shape, 'min', flow.min(), 'max', flow.max()
print 'Rel pose shape', rel_pose.shape, 'min', rel_pose.min(), 'max', rel_pose.max()

# translation = -translation
# rotation = -rotation
# transformation = np.array([[-1., 0., 0.],
#                           [0., 1., 0.],
#                           [0., 0., 1.],], dtype=np.float32)
# translation = transformation.dot(translation)
# rotation = transformation.dot(rotation).dot(transformation)

# import transforms3d.derivations.angle_axes
# theta = np.linalg.norm(rotation)
# rot_mat = np.array(transforms3d.derivations.angle_axes.angle_axis2mat(
#     theta, rotation / theta)).astype(np.float32)
# K = np.array([[0.89115971 * 256, 0.0, 0.5],
#               [0.0, 1.18821287 * 1.92, 0.5],
#               [0.0, 0.0, 1.0]])
# F = fundamental(K, rot_mat, translation)
# print rot_mat

# P1 = np.hstack((K, np.zeros((3, 1))))

# r = np.matrix.transpose(rot_mat)
# t = np.dot(-r, np.reshape(translation, (3, 1)))
# P2 = np.dot(K, np.hstack((r, t)))

# Important!! Convert from pose to extrinsic matrix
rot_mat = np.matrix.transpose(rel_pose[0:3, 0:3])
translation = (-rot_mat).dot(rel_pose[0:3, 3])

# make translation a unit vector
translation /= np.linalg.norm(translation)

import transforms3d.axangles
rotation, angle = transforms3d.axangles.mat2axangle(rot_mat)
rotation = np.array(rotation, dtype=np.float32) * angle

# depth_ = np.zeros((height, width, 1))
# for y in range(height):
#     for x in range(width):
#         ptA = (x, y)
#         flowX = flow[0, y, x]
#         flowY = flow[1, y, x]
#         ptB = (x + (flowX * width), y + (flowY * height))

#         A0 = P1[0, 0:3] - ptA[0] * P1[2, 0:3]
#         A1 = P1[1, 0:3] - ptA[1] * P1[2, 0:3]
#         A2 = P2[0, 0:3] - ptB[0] * P2[2, 0:3]
#         A3 = P2[1, 0:3] - ptB[1] * P2[2, 0:3]
#         b = np.array([-P1[0, 3] + ptA[0] * P1[2, 3],
#                       -P1[1, 3] + ptA[1] * P1[2, 3],
#                       -P2[0, 3] + ptB[0] * P2[2, 3],
#                       -P2[1, 3] + ptB[1] * P2[2, 3]])

#         # A0 = ptA[0] * P1[2, :] - P1[0, :]
#         # A1 = ptA[1] * P1[2, :] - P1[1, :]
#         # A2 = ptB[0] * P2[2, :] - P2[0, :]
#         # A3 = ptB[1] * P2[2, :] - P2[1, :]
#         A = np.vstack((A0, A1, A2, A3))
#         # b = np.zeros((4, 1))
#         #
#         # u, s, vh = np.linalg.svd(A)
#         # X = vh[np.argmax(s)].T
#         # depth_[y, x, 0] = X[2] / (X[-1] + 1.0e-30)
#         # print X

#         X, _, _, _ = np.linalg.lstsq(A, b)
#         depth_[y, x, 0] = X[2]

# depth_ = (depth_ - depth_.min()) / (depth_.max() - depth_.min())

# import scipy.misc
# scipy.misc.imsave('depth_.png', np.squeeze(depth_))


""" Attempt to get depth from flow """
import lmbspecialops as sops
import tensorflow as tf

# flow = np.transpose(np.transpose(flow, [1, 2, 0]) * [width, height], [2, 0, 1])
flow_tf = tf.constant(np.expand_dims(np.transpose(flow, [2, 0, 1]), 0), dtype=tf.float32)
rotation_tf = tf.constant(rotation)
translation_tf = tf.constant(translation)
intrinsics = tf.constant([[0.89115971, 1.18821287, 0.5, 0.5]], dtype=tf.float32)

depth_from_flow_inv = sops.flow_to_depth(
    flow=flow_tf,
    intrinsics=intrinsics,
    rotation=rotation_tf,
    translation=translation_tf,
    normalized_flow=True,
    inverse_depth=True,
)
depth_from_flow = sops.flow_to_depth(
    flow=flow_tf,
    intrinsics=intrinsics,
    rotation=rotation_tf,
    translation=translation_tf,
    normalized_flow=True,
    inverse_depth=False,
)

dff_inv = None
dff = None
with tf.Session() as sess:
    dff_inv = sess.run(depth_from_flow_inv)
    dff = sess.run(depth_from_flow)

print dff_inv.min(), dff_inv.max(), dff_inv.mean(), dff_inv.std()
print dff.min(), dff.max(), dff.mean(), dff.std()

# dff[dff > dff.std() * 3] = dff.std() * 3
# dff_inv[dff_inv > dff_inv.std() * 3] = dff_inv.std() * 3

# dff_inv[dff_inv > 2.0] = 2.0

fig = plt.figure()
plt.boxplot(dff.flatten())
fig.suptitle('Inv. depth from flow\n' +'Median: %f, Mean: %f, Std: %f' % (np.median(dff), np.mean(dff), np.std(dff)))
fig.savefig('boxplot.png')

""" Visualize the inputs """
fig = plt.figure()

subplot = plt.subplot2grid((3, 3), (0, 0), colspan=1)
subplot.set_title('Image A')
plt.imshow(imageA + 0.5)

subplot = plt.subplot2grid((3, 3), (0, 2), colspan=1)
subplot.set_title('Image B')
plt.imshow(imageB + 0.5)

subplot = plt.subplot2grid((3, 3), (1, 0), colspan=1)
subplot.set_title('Flow')
plt.imshow(flow_to_image(flow))
# plt.scatter([22, 23, 26, 44, 41, 26, 50, 29], [
# 8, 10, 15, 23, 24, 26, 31, 34], c='r', s=1, marker='s')

subplot = plt.subplot2grid((3, 3), (2, 0), colspan=1)
subplot.set_title('Inv. Depth (gt)')
plt.imshow(np.squeeze(1.0 / depth), cmap='Greys')
# plt.scatter([22, 23, 26, 44, 41, 26, 50, 29], [
# 8, 10, 15, 23, 24, 26, 31, 34], c='r', s=1, marker='s')

subplot = plt.subplot2grid((3, 3), (2, 1), colspan=1)
subplot.set_title('Inv. Depth (flow_to_depth, inverse_depth=True)')
plt.imshow(np.squeeze(dff_inv), cmap='Greys')

subplot = plt.subplot2grid((3, 3), (2, 2), colspan=1)
subplot.set_title('Depth (flow_to_depth, inverse_depth=False)')
plt.imshow(np.squeeze(dff), cmap='Greys')

fig.savefig('visualization.png')

""" Scale images to [-0.5, 0.5] """
imageA = imageA * 1. / 255 - 0.5
imageB = imageB * 1. / 255 - 0.5

""" Construct histograms for inputs """
fig = plt.figure()

subplot = plt.subplot2grid((3, 3), (0, 0), colspan=1)
subplot.set_title('Image A')
plt.hist(imageA.flatten(), bins='auto')

subplot = plt.subplot2grid((3, 3), (0, 2), colspan=1)
subplot.set_title('Image B')
plt.hist(imageB.flatten(), bins='auto')

subplot = plt.subplot2grid((3, 3), (1, 1), colspan=1)
subplot.set_title('Flow')
plt.hist(flow.flatten(), bins='auto')

subplot = plt.subplot2grid((3, 3), (2, 0), colspan=1)
subplot.set_title('Inv. Depth (gt)')
d = 1.0 / depth.flatten()
dd = (d - d.min()) / (d.max() - d.min())
plt.hist(dd, bins='auto', range=[0.0, 1.5])

subplot = plt.subplot2grid((3, 3), (2, 1), colspan=1)
subplot.set_title('Inv. Depth (flow_to_depth, inverse_depth=True)')
plt.hist(dff_inv.flatten(), bins='auto', range=[0.0, 1.5])

subplot = plt.subplot2grid((3, 3), (2, 2), colspan=1)
subplot.set_title('Depth (flow_to_depth, inverse_depth=False)')
plt.hist(dff.flatten(), bins='auto', range=[0.0, 150.0])

fig.savefig('histograms.png')
