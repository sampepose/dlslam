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

# Get the batch of data
image1, image2, d1, d2, flow12, flow23, valid_val, normals_val, normals_from_downsampled_val, r_rel_val, t_rel_val, p1, p2, mm_val, off_h_val, off_w_val = svkitti_batch_demon(
    '/projects/katefgroup/datasets/svkitti/morning.txt', hyp, shuffle=False, center_crop=True)

def prepare_gt(gt):
    # Create image gt and resize
    image_pair = tf.concat([gt['image1'], gt['image2']], 3)
    image2_2 = tf.image.resize_images(gt['image2'], [48, 64])

    

    # Rearrange all gt from (BS, H, W, K) to (BS, K, H, W)
    image_pair = tf.transpose(image_pair, perm=[0, 3, 1, 2])
    image2_2 = tf.transpose(image2_2, perm=[0, 3, 1, 2])
    depth = tf.transpose(depth, perm=[0, 3, 1, 2])
    normals = tf.transpose(normals, perm=[0, 3, 1, 2])
    flow = tf.transpose(flow, perm=[0, 3, 1, 2])

    return image_pair, image2_2, depth, normals, flow, gt['rotation'], gt['translation']

def rmse(depth1,depth2):
    """
    Computes the root min square errors between the two depth maps.
    Takes preprocessed depths (no nans, infs and non-positive values)

    depth1:  one depth map
    depth2:  another depth map

    Returns: 
        RMSE(log)

    """
    diff = depth1 - depth2
    num_pixels = float(diff.size)
    
    if num_pixels == 0:
        return np.nan
    else:
        return np.sqrt(np.sum(np.square(diff)) / num_pixels)

def scale_invariant(depth1,depth2):
    """
    Computes the scale invariant loss based on differences of logs of depth maps.
    Takes preprocessed depths (no nans, infs and non-positive values)

    depth1:  one depth map
    depth2:  another depth map

    Returns: 
        scale_invariant_distance

    """
    valid_mask = compute_valid_depth_mask(depth2, depth1)
    depth1 = depth1[valid_mask]
    depth2 = depth2[valid_mask]
    # sqrt(Eq. 3)
    assert(np.all(np.isfinite(depth1) & np.isfinite(depth2) & (depth1 > 0) & (depth2 > 0)))
    log_diff = np.log(depth1) - np.log(depth2)
    num_pixels = float(log_diff.size)
    
    if num_pixels == 0:
        return np.nan
    else:
        return np.sqrt(np.sum(np.square(log_diff)) / num_pixels - np.square(np.sum(log_diff)) / np.square(num_pixels))

def compute_valid_depth_mask(d1, d2=None):
    """Computes the mask of valid values for one or two depth maps
    
    Returns a valid mask that only selects values that are valid depth value 
    in both depth maps (if d2 is given).
    Valid depth values are >0 and finite.
    """
    if d2 is None:
        valid_mask = np.isfinite(d1)
        valid_mask[valid_mask] = (d1[valid_mask] > 0)
    else:
        valid_mask = np.isfinite(d1) & np.isfinite(d2)
        valid_mask[valid_mask] = (d1[valid_mask] > 0) & (d2[valid_mask] > 0)
    return valid_mask

def epe(flow1, flow2):
    """Computes the average endpoint error between the two flow fields"""
    diff = flow1 - flow2
    epe = np.sqrt(diff[0, 0,:,:]**2 + diff[0, 1,:,:]**2)
    # mask out invalid epe values
    valid_mask = compute_valid_depth_mask(epe) 
    epe = epe[valid_mask]
    if epe.size > 0:
        return np.mean(epe)
    else:
        return np.nan

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

gt_images = []
gt_depth = []
gt_flow = []
gt_R = []
gt_t = []

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
with tf.Session(config=config) as sess:
    coord = tf.train.Coordinator()
    tf.train.start_queue_runners(coord=coord)

    # Resize ground truths and rearrange to (BS, K, H, W)
    d1 = tf.transpose(tf.image.resize_images(d1, [48, 64], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR), perm=[0, 3, 1, 2])
    d2 = tf.transpose(tf.image.resize_images(d2, [48, 64], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR), perm=[0, 3, 1, 2])
    flow12 = tf.transpose(tf.image.resize_images(flow12, [48, 64], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR), perm=[0, 3, 1, 2])
    flow23 = tf.transpose(tf.image.resize_images(flow23, [48, 64], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR), perm=[0, 3, 1, 2])

    # Collect gt images and poses
    for i in range(800):
        # Collect gt
        image1_, image2_, p1_, p2_, d1_, d2_, f12_, f23_ = sess.run([image1, image2, p1, p2, d1, d2, flow12, flow23])
        gt_images.append((image1_ + 0.5) * 255.)
        gt_images.append((image2_ + 0.5) * 255.)
        gt_R.append(p1_[:, 0:3, 0:3])
        gt_R.append(p2_[:, 0:3, 0:3])
        gt_t.append(p1_[:, 0:3, 3])
        gt_t.append(p2_[:, 0:3, 3])
        gt_depth.append(d1_)
        gt_depth.append(d2_)
        gt_flow.append(f12_)
        gt_flow.append(f23_)

sup_epe = []
sup_scale_inv_depth = []
sup_r_l2 = []
sup_t_l2 = []
sup_depth = []

tf.reset_default_graph()
with tf.Session(config=config) as sess:
    # Collect predictions
    image_1_placeholder = tf.placeholder(dtype=tf.float32, shape=(1, 192, 256, 3), name='image_1_placeholder')
    image_2_placeholder = tf.placeholder(dtype=tf.float32, shape=(1, 192, 256, 3), name='image_2_placeholder')
    weights_dir = './src/experiments/finetune_demon_iterative/logs/extrinsics/finetuned-weights-8800'
    _, iterative_predictions = demon_forward(sess, image_1_placeholder, image_2_placeholder, weights_dir, hyp)
    for i in range(len(gt_images) - 1):
        img1 = gt_images[i] * 1. / 255 - 0.5
        img2 = gt_images[i + 1] * 1. / 255 - 0.5

        feed_dict = {
            image_1_placeholder: img1,
            image_2_placeholder: img2,
        }
        iterative_predictions_ = sess.run(iterative_predictions, feed_dict=feed_dict)

        # rot angle-axis -> rot mat
        pred_r = iterative_predictions_['rotation'][0, ...]
        angle = np.linalg.norm(pred_r)
        pred_r_mat = axangle2mat(pred_r / angle, angle)

        # Collect predictions
        sup_depth.append(iterative_predictions_['depth'])

        # Calculate metrics
        sup_epe.append(epe(gt_flow[i], iterative_predictions_['flow']))
        sup_scale_inv_depth.append(1.0 / scale_invariant(gt_depth[i], 1.0 / iterative_predictions_['depth']))
        sup_r_l2.append(rmse(gt_R[i], iterative_predictions_['rotation']))
        sup_t_l2.append(rmse(gt_t[i], iterative_predictions_['translation']))

print("Supervised:")
print("Avg. EPE: ", np.mean(sup_epe))
print("Avg. Scale Inv. Depth: ", np.mean(sup_scale_inv_depth))
print("Avg. Rotation L2: ", np.mean(sup_r_l2))
print("Avg. Translation L2: ", np.mean(sup_t_l2))

unsup_epe = []
unsup_scale_inv_depth = []
unsup_r_l2 = []
unsup_t_l2 = []
unsup_depth = []

tf.reset_default_graph()
with tf.Session(config=config) as sess:
    # Collect predictions
    image_1_placeholder = tf.placeholder(dtype=tf.float32, shape=(1, 192, 256, 3), name='image_1_placeholder')
    image_2_placeholder = tf.placeholder(dtype=tf.float32, shape=(1, 192, 256, 3), name='image_2_placeholder')
    weights_dir = './src/experiments/finetune_unsupervised_demon_iterative/logs/1e-4/finetuned-weights-19000'
    _, iterative_predictions = demon_forward(sess, image_1_placeholder, image_2_placeholder, weights_dir, hyp)
    for i in range(len(gt_images) - 1):
        img1 = gt_images[i] * 1. / 255 - 0.5
        img2 = gt_images[i + 1] * 1. / 255 - 0.5

        feed_dict = {
            image_1_placeholder: img1,
            image_2_placeholder: img2,
        }
        iterative_predictions_ = sess.run(iterative_predictions, feed_dict=feed_dict)

        # rot angle-axis -> rot mat
        pred_r = iterative_predictions_['rotation'][0, ...]
        angle = np.linalg.norm(pred_r)
        pred_r_mat = axangle2mat(pred_r / angle, angle)

        # Collect predictions
        unsup_depth.append(iterative_predictions_['depth'])

        # Calculate metrics
        unsup_epe.append(epe(gt_flow[i], iterative_predictions_['flow']))
        unsup_scale_inv_depth.append(1.0 / scale_invariant(gt_depth[i], 1.0 / iterative_predictions_['depth']))
        unsup_r_l2.append(rmse(gt_R[i], iterative_predictions_['rotation']))
        unsup_t_l2.append(rmse(gt_t[i], iterative_predictions_['translation']))

print("Unsupervised:")
print("Avg. EPE: ", np.mean(unsup_epe))
print("Avg. Scale Inv. Depth: ", np.mean(unsup_scale_inv_depth))
print("Avg. Rotation L2: ", np.mean(unsup_r_l2))
print("Avg. Translation L2: ", np.mean(unsup_t_l2))

def preprocess_depth(d):
    d_ = np.array(d)
    return (d_ - d_.min()) / (d_.max() - d_.min())

def sidebyside(d1, d2, d3):
    d1 = preprocess_depth([np.transpose(img[0, ...], [1, 2, 0]) for img in d1])
    d2 = preprocess_depth([np.transpose(img[0, ...], [1, 2, 0]) for img in d2])
    d3 = preprocess_depth([np.transpose(img[0, ...], [1, 2, 0]) for img in d3])
    return [np.hstack((a, b, c)) for a,b,c in zip(d1,d2,d3)]

# Save images as gif
# imageio.mimsave(CUR_PATH + '/image.gif', [img[0, ...] for img in gt_images])
imageio.mimsave(CUR_PATH + '/depth.gif', sidebyside(gt_depth, sup_depth, unsup_depth))
# imageio.mimsave(CUR_PATH + '/depth.gif', [preprocess_depth(np.transpose(img[0, ...], [1, 2, 0])) for img in gt_depth])
# imageio.mimsave(CUR_PATH + '/depth-sup.gif', [preprocess_depth(np.transpose(img[0, ...], [1, 2, 0])) for img in sup_depth])
# imageio.mimsave(CUR_PATH + '/depth-unsup.gif', [preprocess_depth(np.transpose(img[0, ...], [1, 2, 0])) for img in unsup_depth])


