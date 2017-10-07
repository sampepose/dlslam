import sys
sys.path.append('../src')

from inputs.batcher import blendswap_batch
from utils import warper
from hyperparams import create_hyperparams

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import scipy.misc
import tensorflow as tf

plt.rc('axes', titlesize=8)
plt.rcParams["figure.figsize"] = (12.8, 9.6)

def np_flow_to_image(flow):
    """
    Convert flow into middlebury color code image
    :param flow: optical flow map
    :return: optical flow image in middlebury color
    """
    u = flow[:, :, 0]
    v = flow[:, :, 1]

    maxu = -999.
    maxv = -999.
    minu = 999.
    minv = 999.

    idxUnknow = (abs(u) > 1e7) | (abs(v) > 1e7)
    u[idxUnknow] = 0
    v[idxUnknow] = 0

    maxu = max(maxu, np.max(u))
    minu = min(minu, np.min(u))

    maxv = max(maxv, np.max(v))
    minv = min(minv, np.min(v))

    rad = np.sqrt(u ** 2 + v ** 2)
    maxrad = max(-1, np.max(rad))

    u = u / (maxrad + np.finfo(float).eps)
    v = v / (maxrad + np.finfo(float).eps)

    img = compute_color(u, v)

    idx = np.repeat(idxUnknow[:, :, np.newaxis], 3, axis=2)
    img[idx] = 0

    return np.uint8(img)


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

hyp = create_hyperparams()
hyp.dataset_name = 'Blendswap'
hyp.dataset_location = '/projects/katefgroup/datasets/synthetic_renderings/demon_scenes_annotated/tfrecords/'
# we need to invert depth since that's what demon predicts
hyp.depth_encoding = 'inv'
hyp.bs = 1

# Get the batch of data
(image1_train, image2_train, depth_train,
flow_train, normals_train, r_rel_train,
t_rel_train, off_h, off_w) = blendswap_batch('/projects/katefgroup/datasets/synthetic_renderings/demon_scenes_annotated/train.txt', hyp)


""" See how many records have zero flow (this means the camera motion was too large!) """
total_count = 3
count_zero = 0

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
# with tf.Session(config=config) as sess:
#     coord = tf.train.Coordinator()
#     threads = tf.train.start_queue_runners(coord=coord)
#     for i in range(total_count / hyp.bs):
#         (imageA, imageB, inv_depth,
#         flow, normals, r_rel, t_rel) = sess.run([image1_train, image2_train,
#                                        depth_train, flow_train,
#                                        normals_train,
#                                        r_rel_train, t_rel_train])
#         for j in range(hyp.bs):
#             if np.all(flow[j, ...] == 0):
#                 count_zero += 1
#     coord.request_stop()
#     coord.join(threads)
# print float(count_zero) / total_count

""" Visualize one record """
with tf.Session(config=config) as sess:
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)
    warped_tf, _ = warper(image2_train, flow_train)
    (imageA, imageB, inv_depth,
    flow, normals, r_rel, t_rel, warped) = sess.run([image1_train, image2_train,
                                   depth_train, flow_train, normals_train,
                                   r_rel_train, t_rel_train, warped_tf])
    coord.request_stop()
    coord.join(threads)

# Unpack single example from batch
imageA = imageA[0, ...]
imageB = imageB[0, ...]
warped = warped[0, ...]
flow = flow[0, ...]
inv_depth = inv_depth[0, ...]
normals = normals[0, ...]
r_rel = r_rel[0, ...]
t_rel = t_rel[0, ...]

print("Blendswap TFRecord data:")
print("ImageA: %s\tImageB: %s" % (imageA.shape, imageB.shape))
print("Flow: %s\tDepth: %s\tNormals: %s" % (flow.shape, inv_depth.shape, normals.shape))
print("R: %s\tt: %s" % (r_rel.shape, t_rel.shape))

""" Visualize the inputs """
fig = plt.figure()

subplot = plt.subplot2grid((3, 3), (0, 0), colspan=1)
subplot.set_title('Image A')
plt.imshow(imageA + 0.5)

subplot = plt.subplot2grid((3, 3), (0, 1), colspan=1)
subplot.set_title('Warped')
plt.imshow(warped + 0.5)

subplot = plt.subplot2grid((3, 3), (0, 2), colspan=1)
subplot.set_title('Image B')
plt.imshow(imageB + 0.5)

subplot = plt.subplot2grid((3, 3), (1, 0), colspan=1)
subplot.set_title('Flow')
plt.imshow(np_flow_to_image(flow))

subplot = plt.subplot2grid((3, 3), (1, 1), colspan=1)
subplot.set_title('Flow Legend')
plt.imshow(scipy.misc.imread('./flow-map.png'))

subplot = plt.subplot2grid((3, 3), (2, 0), colspan=1)
subplot.set_title('Inv. Depth (gt)')
plt.imshow(np.squeeze(inv_depth), cmap='Greys')

print("+x left, +y up, +z backwards coord system")
print("Rotation: ", r_rel)
print("Translation: ", t_rel)

fig.savefig('visualization.png')