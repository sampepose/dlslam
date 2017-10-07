import os
import sys
sys.path.append('./src/demon/python/depthmotionnet')

import tensorflow as tf
import numpy as np

from networks_original import *
from losses import l1Loss
from utils import axangle2mat, zrt2flow, warper, flow_to_image

slim = tf.contrib.slim


def prepare_gt(gt):
    # Create image gt and resize
    image_pair = tf.concat([gt['image1'], gt['image2']], 3)
    image2_2 = tf.image.resize_images(gt['image2'], [48, 64])

    # Resize ground truths and rearrange to (BS, K, H, W)
    depth = tf.image.resize_images(gt['depth'], [48, 64], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    normals = tf.image.resize_images(gt['normals'], [48, 64], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    flow = tf.image.resize_images(gt['flow'], [48, 64], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

    # Rearrange all gt from (BS, H, W, K) to (BS, K, H, W)
    image_pair = tf.transpose(image_pair, perm=[0, 3, 1, 2])
    image2_2 = tf.transpose(image2_2, perm=[0, 3, 1, 2])
    depth = tf.transpose(depth, perm=[0, 3, 1, 2])
    normals = tf.transpose(normals, perm=[0, 3, 1, 2])
    flow = tf.transpose(flow, perm=[0, 3, 1, 2])

    return image_pair, image2_2, depth, normals, flow, gt['rotation'], gt['translation']


def l2(prediction, gt, normalize=True):
    """
    Computes the L2 norm, normalized wrt the number of elements (BS * H * W * C)
    """
    # assert all finite
    assert_gt_op = tf.Assert(tf.reduce_all(tf.is_finite(gt)), [gt])
    assert_pred_op = tf.Assert(tf.reduce_all(tf.is_finite(prediction)), [prediction])

    with tf.control_dependencies([assert_gt_op, assert_pred_op]):
        diff = prediction - gt
        if normalize:
            num_pixels = tf.cast(tf.size(diff), tf.float32)
            return tf.sqrt(tf.reduce_sum(tf.square(diff)) / num_pixels)
        else:
            return tf.sqrt(tf.reduce_sum(tf.square(diff)))


def scale_invariant_gradient_loss(prediction, gt):
    def discrete_scale_invariant_gradient(f, h):
        """
        Calculates the discrete scale invariant gradient of f with spacing h
        """
        eps = 1.0e-3

        _, height, width, _ = f.shape.as_list()

        # Pad the input width and height to allow for the spacing
        padded_f = tf.pad(f, [[0, 0], [0, h], [0, h], [0, 0]])

        # f(i + h, j)
        f_ih_j = padded_f[:, 0:height, h:width + h, :]

        # (f(i + h, j) - f(i, j)) / (|f(i + h, j)| + |f(i, j)|)
        i = (f_ih_j - f) / (tf.abs(f_ih_j) + tf.abs(f) + eps)

        # f(i, j + h)
        f_i_jh = padded_f[:, h:height + h, 0:width, :]

        # (f(i, j + h) - f(i, j)) / (|f(i, j + h)| + |f(i, j)|)
        j = (f_i_jh - f) / (tf.abs(f_i_jh) + tf.abs(f) + eps)

        return tf.stack([i, j])

    all_losses = []
    hs = [1, 2, 4, 8, 16]
    for h in hs:
        pred_grad = discrete_scale_invariant_gradient(prediction, h)
        gt_grad = discrete_scale_invariant_gradient(gt, h)
        all_losses.append(l2(pred_grad, gt_grad, normalize=False))
    return tf.reduce_sum(tf.accumulate_n(all_losses))


def loss(predictions, gt, summarize=True):
    """
    Returns the total loss for the DeMoN architecture

    predictions: dictionary with 'depth', 'normals', 'flow', 'flow_conf', 'rotation', 'translation'
    gt: dictionary with 'depth', 'normals', 'flow', 'rotation', and 'translation'
    """
    depth_loss = l1Loss(predictions['depth'], gt['depth'])
    normal_loss = l2(predictions['normals'], gt['normals'])
    flow_loss = l2(predictions['flow'], gt['flow'])

    # Calculate gt flow confidence and loss from flow prediction
    gt_flow_conf = tf.exp(-tf.abs(predictions['flow'] - gt['flow']))
    flow_conf_loss = l1Loss(predictions['flow_conf'], gt_flow_conf)

    rotation_loss = l2(predictions['rotation'], gt['rotation'])
    translation_loss = l2(predictions['translation'], gt['translation'])

    # Calculate scale invariant gradient losses
    flow_si_loss = scale_invariant_gradient_loss(predictions['flow'], gt['flow'])
    depth_si_loss = scale_invariant_gradient_loss(predictions['depth'], gt['depth'])

    summaries = {}
    if summarize:
        summaries = {
            'depth': tf.summary.scalar('loss: inv depth l1', depth_loss),
            'normals': tf.summary.scalar('loss: normals l2', normal_loss),
            'flow': tf.summary.scalar('loss: flow l2', flow_loss),
            'flow_conf': tf.summary.scalar('loss: flow conf l1', flow_conf_loss),
            'rotation': tf.summary.scalar('loss: rotation l2', rotation_loss),
            'translation': tf.summary.scalar('loss: translation l2', translation_loss),
            'flow_grad': tf.summary.scalar('loss: flow gradient', flow_si_loss),
            'depth_grad': tf.summary.scalar('loss: depth gradient', depth_si_loss),
        }

    # Add the weighted loss to tf internal list of losses
    losses = [depth_loss,
              normal_loss,
              flow_loss,
              flow_conf_loss,
              rotation_loss,
              translation_loss,
              flow_si_loss,
              depth_si_loss
              ]
    total_loss = tf.losses.compute_weighted_loss(losses, [300,
                                                    100,
                                                    1000,
                                                    1000,
                                                    160,
                                                    15,
                                                    1000,
                                                    1500
                                                    ])

    if summarize:
        summaries['total'] = tf.summary.scalar('loss: total', total_loss)

    # Get all losses (weighted sum + regularization)
    return total_loss, losses, summaries


def unsupervised_loss(image1_2, image2_2, pred_depth, pred_R, pred_T, summarize=True):
    fy = tf.fill([image1_2.shape.as_list()[0]], 1.18821287 * 192)
    fx = tf.fill([image1_2.shape.as_list()[0]], 0.89115971 * 256)
    y0 = x0 = tf.fill([image1_2.shape.as_list()[0]], 0.5)

    # Convert predicted *extrinsics* to pose
    pose_R = tf.transpose(pred_R, perm=[0, 2, 1])
    pose_t = tf.matmul(tf.matrix_inverse(-pred_R), tf.expand_dims(pred_T, 2))[:, :, 0]

    flow, _ = zrt2flow(1.0 / pred_depth, pose_R, pose_t, fy, fx, y0, x0)
    warped_image2, _ = warper(image2_2, flow)
    loss = l1Loss(warped_image2, image1_2)
    if summarize:
        tf.summary.image('image 1', tf.expand_dims(image1_2[0, ...], 0))
        tf.summary.image('image 2', tf.expand_dims(image2_2[0, ...], 0))
        tf.summary.image('flow', tf.expand_dims(flow_to_image(flow[0, ...]), 0))
        tf.summary.image('image 2 warped', tf.expand_dims(warped_image2[0, ...], 0))
        tf.summary.image('inv. depth', tf.expand_dims(pred_depth[0, ...], 0))
        tf.summary.scalar('total_loss', loss)
    return loss


def finetune_bootstrap(session, gt, config, val=None):
    """

    gt: dictionary of ground-truth values,
        {
            'image1',      ((BS, 192, 256, 3) scaled between -0.5, 0.5)
            'image2',      ((BS, 192, 256, 3) scaled between -0.5, 0.5)
            'depth',       (BS, 192, 256, 1)
            'normals',     (BS, 192, 256, 3)
            'flow',        (BS, 192, 256, 2)
            'rotation',    (BS, 3)
            'translation', (BS, 3)
        }

    config: dictionary, with default values of,
        {
            # REQUIRED
            'weights_dir': None, (an absolute path to the directory containing 'demon_original' weights)
            'log_dir': None, (an absolute path to a directory for logging training information and finetuned weights)

            # OPTIONAL
            'learning_rate': 1.0e-4,
            'max_iterations': 15000,
            'resume_finetune': False, (True if we are loading weights from previous fine-tuning)
        }

    val: optional validation data, matching the format of `gt`
    """
    # Validate input configuration and set defaults
    if 'weights_dir' not in config:
        raise ValueError('`weights_dir` not included in config dictionary')
    if 'log_dir' not in config:
        raise ValueError('`log_dir` not included in config dictionary')
    if 'learning_rate' not in config:
        config['learning_rate'] = 1.0e-4
    if 'max_iterations' not in config:
        config['max_iterations'] = 15000
    if 'resume_finetune' not in config:
        config['resume_finetune'] = False

    # Prepare gt: resize, rearrange to (BS, K, H, W)
    image_pair_t, image2_2_t, depth_t, normals_t, flow_t, rotation_t, translation_t = prepare_gt(gt)
    if val is not None:
        image_pair_v, image2_2_v, depth_v, normals_v, flow_v, rotation_v, translation_v = prepare_gt(val)

    # Define all gt placeholders
    image_pair_placeholder = tf.placeholder(dtype=tf.float32, shape=(32, 6, 192, 256), name='image_pair_placeholder')
    image2_2_placeholder = tf.placeholder(dtype=tf.float32, shape=(32, 3, 48, 64), name='image2_2_placeholder')
    depth_gt_placeholder = tf.placeholder(dtype=tf.float32, shape=(32, 1, 48, 64), name='depth_gt')
    normals_gt_placeholder = tf.placeholder(dtype=tf.float32, shape=(32, 3, 48, 64), name='normals_gt')
    flow_gt_placeholder = tf.placeholder(dtype=tf.float32, shape=(32, 2, 48, 64), name='flow_gt')
    rotation_gt_placeholder = tf.placeholder(dtype=tf.float32, shape=(32, 3), name='rotation_gt')
    translation_gt_placeholder = tf.placeholder(dtype=tf.float32, shape=(32, 3), name='translation_gt')

    # Construct bootstrap network and get predictions
    bootstrap_predictions = BootstrapNet(session, image_pair_placeholder, image2_2_placeholder).forward()

    # Define loss from predictions
    loss_predictions = {
        'depth': bootstrap_predictions['predict_depth2'],
        'normals': bootstrap_predictions['predict_normal2'],
        'flow': bootstrap_predictions['predict_flow2'],
        'flow_conf': bootstrap_predictions['predict_flow_conf2'],
        'rotation': bootstrap_predictions['predict_rotation'],
        'translation': bootstrap_predictions['predict_translation'],
    }
    loss_gt = {
        'depth': depth_gt_placeholder,
        'normals': normals_gt_placeholder,
        'flow': flow_gt_placeholder,
        'rotation': rotation_gt_placeholder,
        'translation': translation_gt_placeholder,
    }
    total_loss, individual_losses, loss_summaries = loss(loss_predictions, loss_gt)

    def get_train_op():
        # Define and initialize optimizer
        optimizer = tf.train.AdamOptimizer(learning_rate=config['learning_rate'])
        train_op = slim.learning.create_train_op(total_loss, optimizer, global_step=global_step, summarize_gradients=True)
        return train_op

    if config['resume_finetune']:
        global_step = tf.Variable(0, name='global_step', trainable=False)
        train_op = get_train_op()
        pretrained_saver = tf.train.Saver()
        pretrained_saver.restore(session, config['weights_dir'])
    else:
        vars_to_restore = slim.get_variables_to_restore(include=["netFlow1", 'netDM1'])
        pretrained_saver = tf.train.Saver(vars_to_restore)
        pretrained_saver.restore(session, config['weights_dir'])

        temp = set(tf.all_variables())
        global_step = tf.Variable(0, name='global_step', trainable=False)
        train_op = get_train_op()
        session.run(tf.initialize_variables(set(tf.all_variables()) - temp))

    # Create writer to write training information
    writer_t = tf.summary.FileWriter(config['log_dir'] + '/train/', session.graph)
    writer_v = tf.summary.FileWriter(config['log_dir'] + '/val/', session.graph)
    writer_t.flush()
    writer_v.flush()

    # Create saver for saving fine-tuned weights
    training_saver = tf.train.Saver(max_to_keep=None)

    # Add visualization of ground truth and predictions
    tf.summary.image('gt depth', tf.expand_dims(tf.transpose(depth_gt_placeholder, perm=[0, 2, 3, 1])[0, :, :, :], 0))
    tf.summary.image('gt normals', tf.expand_dims(tf.transpose(normals_gt_placeholder, perm=[0, 2, 3, 1])[0, :, :, :], 0))
    tf.summary.image('pred depth', tf.expand_dims(tf.transpose(bootstrap_predictions['predict_depth2'], perm=[0, 2, 3, 1])[0, :, :, :], 0))
    tf.summary.image('pred normals', tf.expand_dims(tf.transpose(bootstrap_predictions['predict_normal2'], perm=[0, 2, 3, 1])[0, :, :, :], 0))
    all_summaries = tf.summary.merge_all()

    # Use loaded global step if we have one
    start_step = 0
    global_step_ = session.run(global_step)
    if global_step_ != 0:
        start_step = global_step_

    # Optimize for i=start_step...max_iterations
    for step in range(start_step, config['max_iterations']):
        # Get training data
        _image_pair_t, _image2_2_t, _depth_t, _normals_t, _flow_t, _rotation_t, _translation_t, _mm = session.run([image_pair_t, image2_2_t, depth_t, normals_t, flow_t, rotation_t, translation_t, gt['motion_mask']])

        # Forward pass, compute losses, backwards pass for training data
        feed_dict = {
            image_pair_placeholder: _image_pair_t,
            image2_2_placeholder: _image2_2_t,
            depth_gt_placeholder: _depth_t,
            normals_gt_placeholder: _normals_t,
            flow_gt_placeholder: _flow_t,
            rotation_gt_placeholder: _rotation_t,
            translation_gt_placeholder: _translation_t,
        }
        _, _loss, _summaries = session.run([train_op, total_loss, all_summaries], feed_dict=feed_dict)
        writer_t.add_summary(_summaries, step)
        writer_t.flush()

        # Forward pass, compute losses for validation data
        if val is not None:
            # Get validation data
            _image_pair_v, _image2_2_v, _depth_v, _normals_v, _flow_v, _rotation_v, _translation_v = session.run([image_pair_v, image2_2_v, depth_v, normals_v, flow_v, rotation_v, translation_v])
            
            # Forward pass, compute losses for validation data
            feed_dict = {
                image_pair_placeholder: _image_pair_v,
                image2_2_placeholder: _image2_2_v,
                depth_gt_placeholder: _depth_v,
                normals_gt_placeholder: _normals_v,
                flow_gt_placeholder: _flow_v,
                rotation_gt_placeholder: _rotation_v,
                translation_gt_placeholder: _translation_v,
            }
            _val_summaries = session.run(all_summaries, feed_dict=feed_dict)
            writer_v.add_summary(_val_summaries, step)
            writer_v.flush()

        if step % 10 == 0:
            print("%d\tloss:%f" % (step, _loss))

        if step % 200 == 0 or step == config['max_iterations'] - 1:
            save_path = training_saver.save(session, config['log_dir'] + 'finetuned-weights', global_step=step)
            print("Model saved in file: %s at step %d" % (save_path, step))


def finetune_iterative(session, gt, config, val=None):
    """
    Finetune the iterative network. Operates as described in the DeMoN paper: each prediction batch of the
    bootstrap network is appended to the input batch of the iterative network. For bootstrap batch sizes of 8,
    we use a total batch size of 32 as input to the iterative network. The predictions of the iterative network
    are used as input to the next pass of the iterative network. This allows the training process to train up
    to 4 iterations at once.

    gt: dictionary of ground-truth values,
        {
            'image1',      ((BS, 192, 256, 3) scaled between -0.5, 0.5)
            'image2',      ((BS, 192, 256, 3) scaled between -0.5, 0.5)
            'depth',       (BS, 192, 256, 1)
            'normals',     (BS, 192, 256, 3)
            'flow',        (BS, 192, 256, 2)
            'rotation',    (BS, 3)
            'translation', (BS, 3)
        }

    config: dictionary, with default values of,
        {
            # REQUIRED
            'weights_dir': None, (an absolute path to the directory containing 'demon_original' weights. Set `resume_finetune` if resuming finetuning.)
            'log_dir': None, (an absolute path to a directory for logging training information and finetuned weights)

            # OPTIONAL
            'learning_rate': 1.0e-5,
            'max_iterations': 15000,
            'plot_zero_motion': False, (zero-motion prediction is a good baseline for R/t. Adds this baseline to R/t plots if True.)
            'bootstrap_weights': None, (optional path to load only the bootstrap weights. `weights_dir` is used to load the iterative weights.)
            'resume_finetune': False, (True if we are loading weights from previous fine-tuning)
            'unsupervised': False, (if True, we train unsupervised using photometric loss. We convert predicted depth/egomotion to flow and warp the gt image.)
        }

    val: optional validation data, matching the format of `gt`
    """
    # Validate input configuration and set defaults
    if 'weights_dir' not in config:
        raise ValueError('`weights_dir` not included in config dictionary')
    if 'log_dir' not in config:
        raise ValueError('`log_dir` not included in config dictionary')
    if 'learning_rate' not in config:
        config['learning_rate'] = 1.0e-4
    if 'max_iterations' not in config:
        config['max_iterations'] = 15000
    if 'plot_zero_motion' not in config:
        config['plot_zero_motion'] = False
    if 'bootstrap_weights' not in config:
        config['bootstrap_weights'] = None
    if 'resume_finetune' not in config:
        config['resume_finetune'] = False
    if 'unsupervised' not in config:
        config['unsupervised'] = False
    if 'restart_global_step' not in config:
        config['restart_global_step'] = False

    # Prepare gt: resize, rearrange to (BS, K, H, W)
    image_pair_t, image2_2_t, depth_t, normals_t, flow_t, rotation_t, translation_t = prepare_gt(gt)
    if val is not None:
        image_pair_v, image2_2_v, depth_v, normals_v, flow_v, rotation_v, translation_v = prepare_gt(val)

    # Placeholders for input to bootstrap network and gt predictions
    image_pair_bootstrap_placeholder = tf.placeholder(dtype=tf.float32, shape=(8, 6, 192, 256), name='image_pair_bootstrap_placeholder')
    image2_2_bootstrap_placeholder = tf.placeholder(dtype=tf.float32, shape=(8, 3, 48, 64), name='image2_2_bootstrap_placeholder')
    depth_gt_placeholder = tf.placeholder(dtype=tf.float32, shape=(32, 1, 48, 64), name='depth_gt_placeholder')
    normals_gt_placeholder = tf.placeholder(dtype=tf.float32, shape=(32, 3, 48, 64), name='normals_gt_placeholder')
    flow_gt_placeholder = tf.placeholder(dtype=tf.float32, shape=(32, 2, 48, 64), name='flow_gt_placeholder')
    rotation_gt_placeholder = tf.placeholder(dtype=tf.float32, shape=(32, 3), name='rotation_gt_placeholder')
    translation_gt_placeholder = tf.placeholder(dtype=tf.float32, shape=(32, 3), name='translation_gt_placeholder')

    # Construct bootstrap network and get predictions
    raw_bootstrap_predictions = BootstrapNet(session, image_pair_bootstrap_placeholder, image2_2_bootstrap_placeholder).forward()
    bootstrap_predictions = {
        'depth': raw_bootstrap_predictions['predict_depth2'],
        'normals': raw_bootstrap_predictions['predict_normal2'],
        'flow': raw_bootstrap_predictions['predict_flow2'],
        'flow_conf': raw_bootstrap_predictions['predict_flow_conf2'],
        'rotation': raw_bootstrap_predictions['predict_rotation'],
        'translation': raw_bootstrap_predictions['predict_translation'],
    }

    # Placeholders for input to iterative network (which happens to be the output of the bootstrap net)
    image_pair_iterative_placeholder = tf.placeholder(dtype=tf.float32, shape=(32, 6, 192, 256), name='image_pair_iterative_placeholder')
    image2_2_iterative_placeholder = tf.placeholder(dtype=tf.float32, shape=(32, 3, 48, 64), name='image2_2_iterative_placeholder')
    depth_iterative_placeholder = tf.placeholder(dtype=tf.float32, shape=(32, 1, 48, 64), name='depth_iterative_placeholder')
    normals_iterative_placeholder = tf.placeholder(dtype=tf.float32, shape=(32, 3, 48, 64), name='normals_iterative_placeholder')
    rotation_iterative_placeholder = tf.placeholder(dtype=tf.float32, shape=(32, 3), name='rotation_iterative_placeholder')
    translation_iterative_placeholder = tf.placeholder(dtype=tf.float32, shape=(32, 3), name='translation_iterative_placeholder')

    # Construct iterative network and get predictions
    raw_iterative_predictions = IterativeNet(session, image_pair_iterative_placeholder, image2_2_iterative_placeholder,
                                             depth_iterative_placeholder, normals_iterative_placeholder, rotation_iterative_placeholder,
                                             translation_iterative_placeholder).forward()
    iterative_predictions = {
        'depth': raw_iterative_predictions['predict_depth2'],
        'normals': raw_iterative_predictions['predict_normal2'],
        'flow': raw_iterative_predictions['predict_flow2'],
        'flow_conf': raw_iterative_predictions['predict_flow_conf2'],
        'rotation': raw_iterative_predictions['predict_rotation'],
        'translation': raw_iterative_predictions['predict_translation'],
    }

    # Loss, optimizer, and training op
    iterative_gt = {
        'depth': depth_gt_placeholder,
        'normals': normals_gt_placeholder,
        'flow': flow_gt_placeholder,
        'rotation': rotation_gt_placeholder,
        'translation': translation_gt_placeholder,
    }

    if config['unsupervised']:
        image1_1 = tf.transpose(image_pair_iterative_placeholder[:, 0:3, :, :], perm=[0, 2, 3, 1])
        image1_1 = tf.image.resize_images(image1_1, [48, 64])
        image2_1 = tf.transpose(image_pair_iterative_placeholder[:, 3:6, :, :], perm=[0, 2, 3, 1])
        image2_1 = tf.image.resize_images(image2_1, [48, 64])
        depth_pred = tf.transpose(iterative_predictions['depth'], perm=[0, 2, 3, 1])
        total_loss = unsupervised_loss(image1_1,
                                       image2_1,
                                       depth_pred,
                                       axangle2mat(iterative_predictions['rotation']),
                                       iterative_predictions['translation'])
        tf.summary.image('gt depth', tf.expand_dims(tf.transpose(depth_gt_placeholder[0, ...], perm=[1, 2, 0]), 0))
    else:
        total_loss, _, summaries = loss(iterative_predictions, iterative_gt)

    def get_train_op(global_step):
        optimizer = tf.train.AdamOptimizer(learning_rate=config['learning_rate'])
        train_op = slim.learning.create_train_op(total_loss, optimizer, global_step=global_step, summarize_gradients=True)
        return train_op

    if config['resume_finetune']:
        # Create training op variables (BEFORE loading weights)
        global_step = tf.Variable(0, name='global_step', trainable=False)
        train_op = get_train_op(global_step)

        # Load all variables (including optimizer) from `weights_dir`
        pretrained_saver = tf.train.Saver()
        pretrained_saver.restore(session, config['weights_dir'])

        print('Loaded finetuned weights: %s' % config['weights_dir'])
    else:
        if config['bootstrap_weights']:
            # Load all bootstrap variables from `bootstrap_weights`...
            vars_to_restore = slim.get_variables_to_restore(include=['netFlow1', 'netDM1'])
            bootstrap_saver = tf.train.Saver(vars_to_restore)
            bootstrap_saver.restore(session, config['bootstrap_weights'])
            print('Loaded bootstrap weights: %s' % config['bootstrap_weights'])

            # ...and all iterative variables from `weights_dir`
            vars_to_restore = slim.get_variables_to_restore(include=['netFlow2', 'netDM2'])
            iterative_saver = tf.train.Saver(vars_to_restore)
            iterative_saver.restore(session, config['weights_dir'])
            print('Loaded iterative weights: %s' % config['weights_dir'])
        else:
            # Load all variables (except optimizer) from `weights_dir`
            vars_to_restore = slim.get_variables_to_restore(include=['global_step'])
            bootstrap_iterative_saver = tf.train.Saver(vars_to_restore)
            bootstrap_iterative_saver.restore(session, config['weights_dir'])
            print('Loaded bootstrap+iterative weights: %s' % config['weights_dir'])

        # Create training op variables (AFTER loading weights)
        temp = set(tf.all_variables())
        global_step = tf.Variable(0, name='global_step', trainable=False)
        train_op = get_train_op(global_step)
        session.run(tf.initialize_variables(set(tf.all_variables()) - temp))

    # Create writer to write training information
    if config['unsupervised']:
        writer_t = tf.summary.FileWriter(config['log_dir'] + '/unsupervised/', session.graph)
    else:
        writer_t = tf.summary.FileWriter(config['log_dir'] + '/train/', session.graph)
    writer_t.flush()
    
    if val is not None:
        writer_v = tf.summary.FileWriter(config['log_dir'] + '/val/', session.graph)
        writer_v.flush()

    # Construct numpy arrays for holding cumulative inputs to iterative network
    image_pair_iterative_np_train = np.zeros((32, 6, 192, 256), dtype=np.float32)
    image2_2_iterative_np_train = np.zeros((32, 3, 48, 64), dtype=np.float32)
    depth_iterative_np_train = np.zeros((32, 1, 48, 64), dtype=np.float32)
    normals_iterative_np_train = np.zeros((32, 3, 48, 64), dtype=np.float32)
    rotation_iterative_np_train = np.zeros((32, 3), dtype=np.float32)
    translation_iterative_np_train = np.zeros((32, 3), dtype=np.float32)
    # validation
    image_pair_iterative_np_val = np.zeros((32, 6, 192, 256), dtype=np.float32)
    image2_2_iterative_np_val = np.zeros((32, 3, 48, 64), dtype=np.float32)
    depth_iterative_np_val = np.zeros((32, 1, 48, 64), dtype=np.float32)
    normals_iterative_np_val = np.zeros((32, 3, 48, 64), dtype=np.float32)
    rotation_iterative_np_val = np.zeros((32, 3), dtype=np.float32)
    translation_iterative_np_val = np.zeros((32, 3), dtype=np.float32)

    # Construct numpy arrays for holding cumulative gt values
    depth_gt_np_train = np.zeros((32, 1, 48, 64), dtype=np.float32)
    flow_gt_np_train = np.zeros((32, 2, 48, 64), dtype=np.float32)
    normals_gt_np_train = np.zeros((32, 3, 48, 64), dtype=np.float32)
    rotation_gt_np_train = np.zeros((32, 3), dtype=np.float32)
    translation_gt_np_train = np.zeros((32, 3), dtype=np.float32)
    # validation
    depth_gt_np_val = np.zeros((32, 1, 48, 64), dtype=np.float32)
    flow_gt_np_val = np.zeros((32, 2, 48, 64), dtype=np.float32)
    normals_gt_np_val = np.zeros((32, 3, 48, 64), dtype=np.float32)
    rotation_gt_np_val = np.zeros((32, 3), dtype=np.float32)
    translation_gt_np_val = np.zeros((32, 3), dtype=np.float32)

    # Create saver for saving fine-tuned weights
    training_saver = tf.train.Saver(max_to_keep=None)

    # Visualize inputs, gt, and predictions
    # tf.summary.image('depth_from_flow', tf.transpose(raw_iterative_predictions['depth_from_flow'], perm=[0, 2, 3, 1]))
    all_summaries = tf.summary.merge_all()

    # Show individual loss for a single iteration of iterative network
    writer_individual_losses1_t = tf.summary.FileWriter(config['log_dir'] + '/individual_losses1_train/', session.graph)
    writer_individual_losses2_t = tf.summary.FileWriter(config['log_dir'] + '/individual_losses2_train/', session.graph)
    writer_individual_losses3_t = tf.summary.FileWriter(config['log_dir'] + '/individual_losses3_train/', session.graph)
    writer_individual_losses4_t = tf.summary.FileWriter(config['log_dir'] + '/individual_losses4_train/', session.graph)
    writers_individual_losses_t = [writer_individual_losses1_t, writer_individual_losses2_t, writer_individual_losses3_t, writer_individual_losses4_t]
    if val:
        writer_individual_losses1_v = tf.summary.FileWriter(config['log_dir'] + '/individual_losses1_val/', session.graph)
        writer_individual_losses2_v = tf.summary.FileWriter(config['log_dir'] + '/individual_losses2_val/', session.graph)
        writer_individual_losses3_v = tf.summary.FileWriter(config['log_dir'] + '/individual_losses3_val/', session.graph)
        writer_individual_losses4_v = tf.summary.FileWriter(config['log_dir'] + '/individual_losses4_val/', session.graph)
        writers_individual_losses_v = [writer_individual_losses1_v, writer_individual_losses2_v, writer_individual_losses3_v, writer_individual_losses4_v]
    individual_iterative_depth_prediction = tf.placeholder(dtype=tf.float32, shape=(8, 1, 48, 64), name='individual_iterative_depth_prediction')
    individual_iterative_normals_prediction = tf.placeholder(dtype=tf.float32, shape=(8, 3, 48, 64), name='individual_iterative_normals_prediction')
    individual_iterative_flow_prediction = tf.placeholder(dtype=tf.float32, shape=(8, 2, 48, 64), name='individual_iterative_flow_prediction')
    individual_iterative_flow_conf_prediction = tf.placeholder(dtype=tf.float32, shape=(8, 2, 48, 64), name='individual_iterative_flow_conf_prediction')
    individual_iterative_rotation_prediction = tf.placeholder(dtype=tf.float32, shape=(8, 3), name='individual_iterative_rotation_prediction')
    individual_iterative_translation_prediction = tf.placeholder(dtype=tf.float32, shape=(8, 3), name='individual_iterative_translation_prediction')
    individual_iterative_predictions = {
        'depth': individual_iterative_depth_prediction,
        'normals': individual_iterative_normals_prediction,
        'flow': individual_iterative_flow_prediction,
        'flow_conf': individual_iterative_flow_conf_prediction,
        'rotation': individual_iterative_rotation_prediction,
        'translation': individual_iterative_translation_prediction,
    }
    individual_iterative_image_pair_gt = tf.placeholder(dtype=tf.float32, shape=(8, 6, 192, 256), name='individual_iterative_image_pair_gt')
    individual_iterative_depth_gt = tf.placeholder(dtype=tf.float32, shape=(8, 1, 48, 64), name='individual_iterative_depth_gt')
    individual_iterative_normals_gt = tf.placeholder(dtype=tf.float32, shape=(8, 3, 48, 64), name='individual_iterative_normals_gt')
    individual_iterative_flow_gt = tf.placeholder(dtype=tf.float32, shape=(8, 2, 48, 64), name='individual_iterative_flow_gt')
    individual_iterative_rotation_gt = tf.placeholder(dtype=tf.float32, shape=(8, 3), name='individual_iterative_rotation_gt')
    individual_iterative_translation_gt = tf.placeholder(dtype=tf.float32, shape=(8, 3), name='individual_iterative_translation_gt')
    individual_iterative_gt = {
        'depth': individual_iterative_depth_gt,
        'normals': individual_iterative_normals_gt,
        'flow': individual_iterative_flow_gt,
        'rotation': individual_iterative_rotation_gt,
        'translation': individual_iterative_translation_gt,
    }
    if config['unsupervised']:
        image1_1 = tf.transpose(individual_iterative_image_pair_gt[:, 0:3, :, :], perm=[0, 2, 3, 1])
        image1_1 = tf.image.resize_images(image1_1, [48, 64])
        image2_1 = tf.transpose(individual_iterative_image_pair_gt[:, 3:6, :, :], perm=[0, 2, 3, 1])
        image2_1 = tf.image.resize_images(image2_1, [48, 64])
        depth_pred = tf.transpose(individual_iterative_predictions['depth'], perm=[0, 2, 3, 1])
        individual_iter_loss = unsupervised_loss(image1_1,
                                                 image2_1,
                                                 depth_pred,
                                                 axangle2mat(individual_iterative_predictions['rotation']),
                                                 individual_iterative_predictions['translation'],
                                                 summarize=False)
    else:
        individual_iter_loss, _, _ = loss(individual_iterative_predictions, individual_iterative_gt)
    individual_iter_loss_summary = tf.summary.scalar('all iterations total loss', individual_iter_loss)

    def compute(_image_pair, _image2_2, _depth, _normals, _flow, _rotation, _translation,
                image_pair_iterative_np, image2_2_iterative_np, depth_iterative_np, normals_iterative_np, rotation_iterative_np, translation_iterative_np,
                depth_gt_np, flow_gt_np, normals_gt_np, rotation_gt_np, translation_gt_np,
                writer, writer_individual_losses, backprop=True):
        # Forward pass for bootstrap network predictions
        feed_dict = {
            image_pair_bootstrap_placeholder: _image_pair,
            image2_2_bootstrap_placeholder: _image2_2,
        }
        _bootstrap_predictions = session.run(bootstrap_predictions, feed_dict=feed_dict)

        # Slide first 3 batches (24 elements) to the last 3 slots of iterative np input...
        image_pair_iterative_np[8:32, :, :, :] = image_pair_iterative_np[0:24, :, :, :]
        image2_2_iterative_np[8:32, :, :, :] = image2_2_iterative_np[0:24, :, :, :]
        depth_iterative_np[8:32, :, :, :] = depth_iterative_np[0:24, :, :, :]
        normals_iterative_np[8:32, :, :, :] = normals_iterative_np[0:24, :, :, :]
        rotation_iterative_np[8:32, :] = rotation_iterative_np[0:24, :]
        translation_iterative_np[8:32, :] = translation_iterative_np[0:24, :]

        # ...and prepend our new bootstrap net predictions (into slot 0)
        image_pair_iterative_np[0:8, :, :, :] = _image_pair
        image2_2_iterative_np[0:8, :, :, :] = _image2_2
        depth_iterative_np[0:8, :, :, :] = _bootstrap_predictions['depth']
        normals_iterative_np[0:8, :, :, :] = _bootstrap_predictions['normals']
        rotation_iterative_np[0:8, :] = _bootstrap_predictions['rotation']
        translation_iterative_np[0:8, :] = _bootstrap_predictions['translation']

        # Keep our cumulative gt correct via the same two steps
        depth_gt_np[8:32, :, :, :] = depth_gt_np[0:24, :, :, :]
        flow_gt_np[8:32, :, :, :] = flow_gt_np[0:24, :, :, :]
        normals_gt_np[8:32, :, :, :] = normals_gt_np[0:24, :, :, :]
        rotation_gt_np[8:32, :] = rotation_gt_np[0:24, :]
        translation_gt_np[8:32, :] = translation_gt_np[0:24, :]
        depth_gt_np[0:8, :, :, :] = _depth
        flow_gt_np[0:8, :, :, :] = _flow
        normals_gt_np[0:8, :, :, :] = _normals
        rotation_gt_np[0:8, :] = _rotation
        translation_gt_np[0:8, :] = _translation

        # Forward and backwards passes for iterative network
        feed_dict = {
            # inputs
            image_pair_iterative_placeholder: image_pair_iterative_np,
            image2_2_iterative_placeholder: image2_2_iterative_np,

            # predictions
            depth_iterative_placeholder: depth_iterative_np,
            normals_iterative_placeholder: normals_iterative_np,
            rotation_iterative_placeholder: rotation_iterative_np,
            translation_iterative_placeholder: translation_iterative_np,

            # gt
            depth_gt_placeholder: depth_gt_np,
            normals_gt_placeholder: normals_gt_np,
            flow_gt_placeholder: flow_gt_np,
            rotation_gt_placeholder: rotation_gt_np,
            translation_gt_placeholder: translation_gt_np,
        }

        if backprop:
            _, _iterative_predictions, _loss, _summaries = session.run([train_op, iterative_predictions, total_loss, all_summaries], feed_dict=feed_dict)
        else:
            _iterative_predictions, _loss, _summaries = session.run([iterative_predictions, total_loss, all_summaries], feed_dict=feed_dict)

        # Update np input to iterative network with iterative predictions
        depth_iterative_np[:] = _iterative_predictions['depth']
        normals_iterative_np[:] = _iterative_predictions['normals']
        rotation_iterative_np[:] = _iterative_predictions['rotation']
        translation_iterative_np[:] = _iterative_predictions['translation']

        # Write summaries to disk
        writer.add_summary(_summaries, step)
        writer.flush()

        # Compute individual losses and write to disk
        for i in range(4):
            feed_dict = {
                # predictions
                individual_iterative_depth_prediction: _iterative_predictions['depth'][i*8:(i + 1)*8, ...],
                individual_iterative_normals_prediction: _iterative_predictions['normals'][i*8:(i + 1)*8, ...],
                individual_iterative_flow_prediction: _iterative_predictions['flow'][i*8:(i + 1)*8, ...],
                individual_iterative_flow_conf_prediction: _iterative_predictions['flow_conf'][i*8:(i + 1)*8, ...],
                individual_iterative_rotation_prediction: _iterative_predictions['rotation'][i*8:(i + 1)*8, ...],
                individual_iterative_translation_prediction: _iterative_predictions['translation'][i*8:(i + 1)*8, ...],

                # gt
                individual_iterative_image_pair_gt: image_pair_iterative_np[i*8:(i + 1)*8, ...],
                individual_iterative_depth_gt: depth_gt_np[i*8:(i + 1)*8, ...],
                individual_iterative_normals_gt: normals_gt_np[i*8:(i + 1)*8, ...],
                individual_iterative_flow_gt: flow_gt_np[i*8:(i + 1)*8, ...],
                individual_iterative_rotation_gt: rotation_gt_np[i*8:(i + 1)*8, ...],
                individual_iterative_translation_gt: translation_gt_np[i*8:(i + 1)*8, ...],
            }
            _individual_iter_loss_summary = session.run(individual_iter_loss_summary, feed_dict=feed_dict)
            writer_individual_losses[i].add_summary(_individual_iter_loss_summary, step)
            writer_individual_losses[i].flush()

        return _loss

    depth_pred_at_each_iteration = np.zeros((5, 48, 64, 1))
    normals_pred_at_each_iteration = np.zeros((5, 48, 64, 3))
    depth_at_iter = tf.placeholder(dtype=tf.float32, shape=(5, 48, 64, 1), name='depth_at_iter')
    normals_at_iter = tf.placeholder(dtype=tf.float32, shape=(5, 48, 64, 3), name='normals_at_iter')
    depth_at_iter1_summary = tf.summary.image('iter1 depth', tf.expand_dims(depth_at_iter[0, ...], 0))
    depth_at_iter2_summary = tf.summary.image('iter2 depth', tf.expand_dims(depth_at_iter[1, ...], 0))
    depth_at_iter3_summary = tf.summary.image('iter3 depth', tf.expand_dims(depth_at_iter[2, ...], 0))
    depth_at_iter4_summary = tf.summary.image('iter4 depth', tf.expand_dims(depth_at_iter[3, ...], 0))
    normals_at_iter1_summary = tf.summary.image('iter1 normals', tf.expand_dims(normals_at_iter[0, ...], 0))
    normals_at_iter2_summary = tf.summary.image('iter2 normals', tf.expand_dims(normals_at_iter[1, ...], 0))
    normals_at_iter3_summary = tf.summary.image('iter3 normals', tf.expand_dims(normals_at_iter[2, ...], 0))
    normals_at_iter4_summary = tf.summary.image('iter4 normals', tf.expand_dims(normals_at_iter[3, ...], 0))
    depth_gt_summary = tf.summary.image('depth gt', tf.expand_dims(depth_at_iter[4, ...], 0))
    normals_gt_summary = tf.summary.image('normals gt', tf.expand_dims(normals_at_iter[4, ...], 0))
    image_summaries = tf.summary.merge([depth_gt_summary, depth_at_iter1_summary, depth_at_iter2_summary,
                                                depth_at_iter3_summary, depth_at_iter4_summary,
                                                normals_at_iter1_summary, normals_at_iter2_summary,
                                                normals_at_iter3_summary, normals_at_iter4_summary, normals_gt_summary])

    # Use loaded global step if we have one
    start_step = 0
    if config['restart_global_step'] is False:
        global_step_ = session.run(global_step)
        if global_step_ != 0:
            start_step = global_step_
    else:
        session.run(global_step.assign(0))

    # Optimize for i=0...max_iterations
    for step in range(start_step, config['max_iterations']):
        # Get training data
        _image_pair_t, _image2_2_t, _depth_t, _normals_t, _flow_t, _rotation_t, _translation_t = session.run([image_pair_t, image2_2_t, depth_t, normals_t, flow_t, rotation_t, translation_t])

        # def r(t):
        #     np.transpose(t[0, ...], [1, 2, 0])

        # np.save('imageA.npy', np.transpose(_image_pair_t[0, 0:3, :, :], [1, 2, 0]))
        # np.save('imageB.npy', np.transpose(_image_pair_t[0, 3:6, :, :], [1, 2, 0]))
        # np.save('depth.npy', np.squeeze(_depth_t[0, :, :, :]))
        # np.save('normals.npy', r(_normals_t))
        # np.save('flow.npy', r(_flow_t))
        # np.save('rotation.npy', _rotation_t[0, ...])
        # np.save('translation.npy', _translation_t[0, ...])
        # return


        _loss = compute(_image_pair_t, _image2_2_t, _depth_t, _normals_t, _flow_t, _rotation_t, _translation_t,
                        image_pair_iterative_np_train, image2_2_iterative_np_train, depth_iterative_np_train, normals_iterative_np_train, rotation_iterative_np_train, translation_iterative_np_train,
                        depth_gt_np_train, flow_gt_np_train, normals_gt_np_train, rotation_gt_np_train, translation_gt_np_train, 
                        writer=writer_t, writer_individual_losses=writers_individual_losses_t)

        if val is not None:
            _image_pair_v, _image2_2_v, _depth_v, _normals_v, _flow_v, _rotation_v, _translation_v = session.run([image_pair_v, image2_2_v, depth_v, normals_v, flow_v, rotation_v, translation_v])
            _ = compute(_image_pair_v, _image2_2_v, _depth_v, _normals_v, _flow_v, _rotation_v, _translation_v,
                    image_pair_iterative_np_val, image2_2_iterative_np_val, depth_iterative_np_val, normals_iterative_np_val, rotation_iterative_np_val, translation_iterative_np_val,
                    depth_gt_np_val, flow_gt_np_val, normals_gt_np_val, rotation_gt_np_val, translation_gt_np_val, 
                    writer=writer_v, writer_individual_losses=writers_individual_losses_v, backprop=False)

            idx = max((step - 1) % 4, 0)
            # Capture prediction at each iteration (i, i+1, i+2, i+3) and display when finished
            depth_pred_at_each_iteration[idx, ...] = np.transpose(depth_iterative_np_val[idx * 8, ...], [1, 2, 0])
            normals_pred_at_each_iteration[idx, ...] = np.transpose(normals_iterative_np_val[idx * 8, ...], [1, 2, 0])
            if idx == 3:
                # Add ground truth
                depth_pred_at_each_iteration[4, ...] = np.transpose(depth_gt_np_val[-8, ...], [1, 2, 0])
                normals_pred_at_each_iteration[4, ...] = np.transpose(normals_gt_np_val[-8, ...], [1, 2, 0])
                feed_dict = {
                    depth_at_iter: depth_pred_at_each_iteration,
                    normals_at_iter: normals_pred_at_each_iteration,
                }
                image_summaries_ = session.run(image_summaries, feed_dict=feed_dict)
                writer_v.add_summary(image_summaries_, step)
                writer_v.flush()

        if step % 10 == 0:
            print("%d\tloss:%f" % (step, _loss))

        if step % 100 == 0 or step == config['max_iterations'] - 1:
            save_path = training_saver.save(session, config['log_dir'] + 'finetuned-weights', global_step=step)
            print("Model saved in file: %s at step %d" % (save_path, step))
