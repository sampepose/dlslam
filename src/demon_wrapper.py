import os
import sys
sys.path.append('./src/demon/python/depthmotionnet')

import tensorflow as tf
import numpy as np

from networks_original import *

def prepare_gt(image1, image2):
    # Create image gt and resize
    image_pair = tf.concat([image1, image2], 3)
    image2_2 = tf.image.resize_images(image2, [48, 64])

    # Rearrange all gt from (BS, H, W, K) to (BS, K, H, W)
    image_pair = tf.transpose(image_pair, perm=[0, 3, 1, 2])
    image2_2 = tf.transpose(image2_2, perm=[0, 3, 1, 2])

    return image_pair, image2_2

def demon_forward(session, image1, image2, load_weights, hyp):
    image_pair, image2_2 = prepare_gt(image1, image2)

    raw_bootstrap_predictions = BootstrapNet(session, image_pair, image2_2).forward()
    bootstrap_predictions = {
        'depth': raw_bootstrap_predictions['predict_depth2'],
        'normals': raw_bootstrap_predictions['predict_normal2'],
        'flow': raw_bootstrap_predictions['predict_flow2'],
        'flow_conf': raw_bootstrap_predictions['predict_flow_conf2'],
        'rotation': raw_bootstrap_predictions['predict_rotation'],
        'translation': raw_bootstrap_predictions['predict_translation'],
    }
    
    # Construct iterative network and get predictions
    raw_iterative_predictions = IterativeNet(session, image_pair, image2_2,
                                             bootstrap_predictions['depth'],
                                             bootstrap_predictions['normals'],
                                             bootstrap_predictions['rotation'],
                                             bootstrap_predictions['translation']).forward()
    iterative_predictions = {
        'depth': raw_iterative_predictions['predict_depth2'],
        'normals': raw_iterative_predictions['predict_normal2'],
        'flow': raw_iterative_predictions['predict_flow2'],
        'flow_conf': raw_iterative_predictions['predict_flow_conf2'],
        'rotation': raw_iterative_predictions['predict_rotation'],
        'translation': raw_iterative_predictions['predict_translation'],
    }

    load_weights(session)

    return [bootstrap_predictions, iterative_predictions]

def demon_bir_forward(session, image1, image2, load_weights, hyp):
    image_pair, image2_2 = prepare_gt(image1, image2)

    raw_bootstrap_predictions = BootstrapNet(session, image_pair, image2_2).forward()
    bootstrap_predictions = {
        'depth': raw_bootstrap_predictions['predict_depth2'],
        'normals': raw_bootstrap_predictions['predict_normal2'],
        'flow': raw_bootstrap_predictions['predict_flow2'],
        'flow_conf': raw_bootstrap_predictions['predict_flow_conf2'],
        'rotation': raw_bootstrap_predictions['predict_rotation'],
        'translation': raw_bootstrap_predictions['predict_translation'],
    }
    
    # Construct iterative network and get predictions
    raw_iterative_predictions = IterativeNet(session, image_pair, image2_2,
                                             bootstrap_predictions['depth'],
                                             bootstrap_predictions['normals'],
                                             bootstrap_predictions['rotation'],
                                             bootstrap_predictions['translation']).forward()
    iterative_predictions = {
        'depth': raw_iterative_predictions['predict_depth2'],
        'normals': raw_iterative_predictions['predict_normal2'],
        'flow': raw_iterative_predictions['predict_flow2'],
        'flow_conf': raw_iterative_predictions['predict_flow_conf2'],
        'rotation': raw_iterative_predictions['predict_rotation'],
        'translation': raw_iterative_predictions['predict_translation'],
    }

    image1_t = tf.transpose(image1, perm=[0, 3, 1, 2])
    refinement_predictions = RefinementNet(session, image1_t, iterative_predictions['depth']).forward()

    load_weights(session)

    return [bootstrap_predictions, iterative_predictions, refinement_predictions]