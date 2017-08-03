import tensorflow as tf


def feed_from(inputs, variables, sess):
    return match(variables, sess.run(inputs))


def l1Loss(e, g):
    with tf.variable_scope("l1Loss"):
        return tf.reduce_mean(tf.abs(e - g))


def random_crop(t, crop_h, crop_w, h, w):
    def off_h(): return tf.random_uniform([], minval=0, maxval=(h - crop_h - 1), dtype=tf.int32)

    def off_w(): return tf.random_uniform([], minval=0, maxval=(w - crop_w - 1), dtype=tf.int32)

    def zero(): return tf.constant(0)

    offset_h = tf.cond(tf.less(crop_h, h - 1), off_h, zero)
    offset_w = tf.cond(tf.less(crop_w, w - 1), off_w, zero)
    t_crop = tf.slice(t, [offset_h, offset_w, 0], [crop_h, crop_w, -1], name="cropped_tensor")
    return t_crop, offset_h, offset_w


def encode_depth_inv(x):
    return 1 / (x + EPS)


def decode_depth_inv(x):
    return 1 / (x + EPS)


def encode_depth_log(x):
    return tf.log(x + EPS)


def decode_depth_log(x):
    return tf.exp(x + EPS)


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
