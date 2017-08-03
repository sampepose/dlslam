import tensorflow as tf
from encoder_decoder import *
from utils import l1Loss


def DepthNet(im, depth_g, is_train=True, reuse=False):
    with tf.variable_scope("depth"):
        if reuse:
            tf.get_variable_scope().reuse_variables()
        depth_stack = encoder_decoder(im, 1, "DepthNet",
                                      hyp.nLayers_depth,
                                      relu=True,
                                      is_train=is_train,
                                      reuse=reuse)
        depth_e = depth_stack[-1]
        if not is_train:
            depth_e = tf.stop_gradient(depth_e)

        with tf.variable_scope("loss"):
            depth_loss = l1Loss(depth_e, depth_g)

        with tf.name_scope("summ"):
            d1_e_sum = tf.summary.histogram("d1_e", depth_e)
            d1_g_sum = tf.summary.histogram("d1_g", depth_g)

            d1_e_sum2 = tf.summary.scalar("d1_e_mean", tf.reduce_mean(depth_e))
            d1_g_sum2 = tf.summary.scalar("d1_g_mean", tf.reduce_mean(depth_g))

            l_sum = tf.summary.scalar("depth_loss", depth_loss)

            id_depth_e = decode_depth(depth_e, hyp.depth_encoding)
            id_depth_g = decode_depth(depth_g, hyp.depth_encoding)
            id_l1 = l1Loss(id_depth_e, id_depth_g)
            id_l1_sum = tf.summary.scalar("id_l1", id_l1)

        depth_stuff = [depth_loss, depth_g, depth_e]
    return depth_stuff
