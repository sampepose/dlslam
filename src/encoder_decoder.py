import tensorflow as tf
import tensorflow.contrib.slim as slim


def encoder_decoder(inputs, output_depth, name, nLayers, relu=False, std=1e-4, do_decode=True, is_train=True, reuse=False):
    with tf.variable_scope(name):
        if reuse:
            tf.get_variable_scope().reuse_variables()
        encode_stack = []
        decode_stack = []
        shape = inputs.get_shape()
        bs = int(shape[0])
        h = int(shape[1])
        w = int(shape[2])

        with slim.arg_scope([slim.conv2d, slim.conv2d_transpose],
                            padding="VALID",
                            activation_fn=tf.nn.relu,
                            normalizer_fn=slim.batch_norm,
                            normalizer_params={'is_training': is_train,
                                               'decay': 0.97,
                                               'epsilon': 1e-5,
                                               'scale': True,
                                               'updates_collections': None},
                            stride=1,
                            weights_initializer=tf.truncated_normal_initializer(stddev=std),
                            weights_regularizer=slim.l2_regularizer(0.0005)):
            """ ENCODER """
            net = inputs
            chans = 32

            # First, one conv at full res
            net = tf.pad(net, [[0, 0], [1, 1], [1, 1], [0, 0]], hyperparams.pad)
            net = slim.conv2d(net, chans, [3, 3], stride=1, scope='encode_conv0')
            encode_stack.append(net)

            # Creates nLayers with each layer comprising of a conv w/ stride 2 and conv w/ stride 1
            # With each new layer (after 2 convs), the number of channels doubles
            for i in range(nLayers):
                chans = int(chans * 2)
                net = tf.pad(net, [[0, 0], [1, 1], [1, 1], [0, 0]], hyperparams.pad)
                net = slim.conv2d(net, chans, [3, 3], stride=2, scope='encode_conv%d_1' % (i + 1))
                net = tf.pad(net, [[0, 0], [1, 1], [1, 1], [0, 0]], hyperparams.pad)
                net = slim.conv2d(net, chans, [3, 3], stride=1, scope='encode_conv%d_2' % (i + 1))
                if i != nLayers - 1:
                    encode_stack.append(net)
                h = int(h / 2)
                w = int(w / 2)

            """ DECODER """
            if do_decode:
                for i in reversed(range(nLayers)):
                    # predict from these feats
                    net = tf.pad(net, [[0, 0], [1, 1], [1, 1], [0, 0]], hyperparams.pad)
                    pred = slim.conv2d(net, output_depth, [3, 3], stride=1,
                                       activation_fn=None,
                                       normalizer_fn=None,
                                       scope='decode_conv%d' % (i + 1))
                    if relu:
                        pred = tf.nn.relu(pred)
                    decode_stack.append(pred)

                    # deconv the feats
                    chans = int(chans / 2)

                    # unpad
                    net = tf.slice(net, [0, 1, 1, 0], [-1, h, w, -1])

                    h = int(h * 2)
                    w = int(w * 2)
                    if hyperparams.do_classic_deconv:
                        net = slim.conv2d_transpose(net, chans, [4, 4], stride=2,
                                                    padding="SAME", scope="decode_deconv%d" % (i + 1))
                    else:
                        net = tf.image.resize_nearest_neighbor(
                            net, [h, w], name="decode_upsample%d" % (i + 1))
                        net = tf.pad(net, [[0, 0], [1, 1], [1, 1], [0, 0]], hyperparams.pad)
                        net = slim.conv2d(net, chans, [3, 3], stride=1,
                                          scope='decode_deconv%d' % (i + 1))
                    # concat [upsampled pred, deconv, saved conv from earlier]
                    net = tf.concat(axis=3, values=[tf.image.resize_images(pred, [h, w]), net,
                                                    encode_stack.pop()],
                                    name="decode_concat%d" % (i + 1))

            # one last pred at full res
            net = tf.pad(net, [[0, 0], [1, 1], [1, 1], [0, 0]], hyperparams.pad)
            pred = slim.conv2d(net, output_depth, [3, 3], stride=1,
                               activation_fn=None,
                               normalizer_fn=None,
                               scope='decode_conv0')
            if relu:
                pred = tf.nn.relu(pred)
            decode_stack.append(pred)

        return decode_stack
