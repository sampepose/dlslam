import tensorflow as tf
import sys
sys.path.append('..')


def read_and_decode_blendswap(filename_queue):
    compress = tf.python_io.TFRecordOptions(
        compression_type=tf.python_io.TFRecordCompressionType.GZIP)
    reader = tf.TFRecordReader(options=compress)
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(
        serialized_example,
        features={
            'image1': tf.FixedLenFeature([], tf.string),
            'image2': tf.FixedLenFeature([], tf.string),
            'relativeTranslation1to2': tf.FixedLenFeature([3], tf.float32),
            'relativeRotation1to2': tf.FixedLenFeature([3], tf.float32),
            'depth1': tf.FixedLenFeature([], tf.string),
            'flow1to2': tf.FixedLenFeature([], tf.string),
        })

    h, w = 480, 640
    i1 = tf.decode_raw(features['image1'], tf.uint8)
    i2 = tf.decode_raw(features['image2'], tf.uint8)
    d1 = tf.decode_raw(features['depth1'], tf.float32)
    f1 = tf.decode_raw(features['flow1to2'], tf.float32)
    relativeTranslation1to2 = features['relativeTranslation1to2']
    relativeRotation1to2 = features['relativeRotation1to2']

    im_shape = tf.stack([h, w, 3])
    i1 = tf.reshape(i1, im_shape)
    i2 = tf.reshape(i2, im_shape)

    depth_shape = tf.stack([h, w, 1])
    d1 = tf.reshape(d1, depth_shape)

    flow_shape = tf.stack([h, w, 2])
    f1 = tf.reshape(f1, flow_shape)

    relativeTranslation1to2 = tf.reshape(relativeTranslation1to2, [3])
    relativeRotation1to2 = tf.reshape(relativeRotation1to2, [3])

    return (h, w, i1, i2, d1, f1, relativeTranslation1to2, relativeRotation1to2)


def read_and_decode_svkitti(filename_queue):
    compress = tf.python_io.TFRecordOptions(
        compression_type=tf.python_io.TFRecordCompressionType.GZIP)
    reader = tf.TFRecordReader(options=compress)
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(
        serialized_example,
        features={
            'height': tf.FixedLenFeature([], tf.int64),
            'width': tf.FixedLenFeature([], tf.int64),
            'i1_raw': tf.FixedLenFeature([], tf.string),
            'i2_raw': tf.FixedLenFeature([], tf.string),
            'd1_raw': tf.FixedLenFeature([], tf.string),
            'd2_raw': tf.FixedLenFeature([], tf.string),
            'f1_raw': tf.FixedLenFeature([], tf.string),
            'f2_raw': tf.FixedLenFeature([], tf.string),
            'v1_raw': tf.FixedLenFeature([], tf.string),
            'v2_raw': tf.FixedLenFeature([], tf.string),
            'p1_raw': tf.FixedLenFeature([16], tf.float32),
            'p2_raw': tf.FixedLenFeature([16], tf.float32),
            'm1_raw': tf.FixedLenFeature([], tf.string),
            'm2_raw': tf.FixedLenFeature([], tf.string),
        })

    h = tf.cast(features['height'], tf.int32)
    w = tf.cast(features['width'], tf.int32)
    i1 = tf.decode_raw(features['i1_raw'], tf.uint8)
    i2 = tf.decode_raw(features['i2_raw'], tf.uint8)
    d1 = tf.decode_raw(features['d1_raw'], tf.float32)
    d2 = tf.decode_raw(features['d2_raw'], tf.float32)
    f1 = tf.decode_raw(features['f1_raw'], tf.float32)
    f2 = tf.decode_raw(features['f2_raw'], tf.float32)
    v1 = tf.decode_raw(features['v1_raw'], tf.uint8)
    v2 = tf.decode_raw(features['v2_raw'], tf.uint8)
    p1 = tf.cast(features['p1_raw'], tf.float32)
    p2 = tf.cast(features['p2_raw'], tf.float32)
    p2 = tf.cast(features['p2_raw'], tf.float32)
    m1 = tf.decode_raw(features['m1_raw'], tf.uint8)
    m2 = tf.decode_raw(features['m2_raw'], tf.uint8)

    im_shape = tf.stack([h, w, 3])
    i1 = tf.reshape(i1, im_shape)
    i2 = tf.reshape(i2, im_shape)

    depth_shape = tf.stack([h, w, 1])
    d1 = tf.reshape(d1, depth_shape)
    d2 = tf.reshape(d2, depth_shape)

    flow_shape = tf.stack([h, w, 2])
    f1 = tf.reshape(f1, flow_shape)
    f2 = tf.reshape(f2, flow_shape)

    valid_shape = tf.stack([h, w, 1])
    v1 = tf.reshape(v1, valid_shape)
    v2 = tf.reshape(v2, valid_shape)

    pose_shape = tf.stack([4, 4])
    p1 = tf.reshape(p1, pose_shape)
    p2 = tf.reshape(p2, pose_shape)

    mask_shape = tf.stack([h, w, 1])
    m1 = tf.reshape(m1, mask_shape)
    m2 = tf.reshape(m2, mask_shape)

    return (h, w, i1, i2, d1, d2, f1, f2, v1, v2, p1, p2, m1, m2)


def read_and_decode_euromav(filename_queue):
    compress = tf.python_io.TFRecordOptions(
        compression_type=tf.python_io.TFRecordCompressionType.GZIP)
    reader = tf.TFRecordReader(options=compress)
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(
        serialized_example,
        features={
            'height': tf.FixedLenFeature([], tf.int64),
            'width': tf.FixedLenFeature([], tf.int64),
            'prev_img_l': tf.FixedLenFeature([], tf.string),
            'curr_img_l': tf.FixedLenFeature([], tf.string),
            'prev_img_r': tf.FixedLenFeature([], tf.string),
            'curr_img_r': tf.FixedLenFeature([], tf.string),
            'p1_raw': tf.FixedLenFeature([16], tf.float32),
            'p2_raw': tf.FixedLenFeature([16], tf.float32),
        })
    height = features['height']
    width = features['width']

    prev_img_l = tf.decode_raw(features['prev_img_l'], tf.uint8)
    curr_img_l = tf.decode_raw(features['curr_img_l'], tf.uint8)
    prev_img_r = tf.decode_raw(features['prev_img_r'], tf.uint8)
    curr_img_r = tf.decode_raw(features['curr_img_r'], tf.uint8)
    p1 = features['p1_raw']
    p2 = features['p2_raw']
    # rel_pose = tf.decode_raw(features['rel_pose'], tf.float64)

    prev_img_l = tf.reshape(prev_img_l, [480, 752, 1], name='prev_img_l')
    curr_img_l = tf.reshape(curr_img_l, [480, 752, 1], name='curr_img_l')
    prev_img_r = tf.reshape(prev_img_r, [480, 752, 1], name='prev_img_r')
    curr_img_r = tf.reshape(curr_img_r, [480, 752, 1], name='curr_img_r')
    # rel_pose = tf.reshape(rel_pose, [4,4],name='rel_pose')
    pose_shape = tf.stack([4, 4])
    p1 = tf.reshape(p1, pose_shape)
    p2 = tf.reshape(p2, pose_shape)

    # imu=tf.cast(imu,tf.float32)
    # rel_pose=tf.cast(rel_pose,tf.float32)

    return (height, width, prev_img_l, curr_img_l, prev_img_r, curr_img_r, p1, p2)
