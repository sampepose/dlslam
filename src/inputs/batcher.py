import tensorflow as tf

import os
import sys
sys.path.append('..')

from src.utils import *
from src.inputs.readers import *


def svkitti_batch(dataset, hyp, shuffle=True):
    with open(dataset) as f:
        content = f.readlines()
    records = [hyp.dataset_location + line.strip() for line in content]
    nRecords = len(records)
    print ('found %d records' % nRecords)
    for record in records:
        assert os.path.isfile(record), 'Record at %s was not found' % record

    queue = tf.train.string_input_producer(records, shuffle=shuffle)

    (h, w, i1, i2, d1, d2, f12, f23, v1, v2, p1, p2, m1, m2) = read_and_decode_svkitti(queue)

    i1 = tf.cast(i1, tf.float32) * 1. / 255 - 0.5
    i2 = tf.cast(i2, tf.float32) * 1. / 255 - 0.5
    d1 = tf.cast(d1, tf.float32)
    d2 = tf.cast(d2, tf.float32)
    v1 = tf.cast(v1, tf.float32)  # 1 at non-sky pixels
    v2 = tf.cast(v2, tf.float32)
    m1 = tf.cast(m1, tf.float32) * 1. / 255  # these are stored in [0,255], and 255 means moving.
    m2 = tf.cast(m2, tf.float32) * 1. / 255
    # d1 = d1*v1 # put 0 depth at invalid spots
    # d2 = d2*v2
    d1 = encode_depth(d1, hyp.depth_encoding)  # encode
    d2 = encode_depth(d2, hyp.depth_encoding)

    # image tensors need to be cropped. we'll do them all at once.
    allCat = tf.concat(axis=2, values=[i1, i2,
                                       d1, d2,
                                       f12, f23,
                                       v1, v2,
                                       m1, m2],
                       name="allCat")

    # image tensors need to be cropped. we'll do them all at once.
    print_shape(allCat)
    allCat_crop, off_h, off_w = random_crop(allCat, hyp.h, hyp.w, h, w)
    print_shape(allCat_crop)
    i1 = tf.slice(allCat_crop, [0, 0, 0], [-1, -1, 3], name="i1")
    i2 = tf.slice(allCat_crop, [0, 0, 3], [-1, -1, 3], name="i2")
    d1 = tf.slice(allCat_crop, [0, 0, 6], [-1, -1, 1], name="d1")
    d2 = tf.slice(allCat_crop, [0, 0, 7], [-1, -1, 1], name="d2")
    f12 = tf.slice(allCat_crop, [0, 0, 8], [-1, -1, 2], name="f12")
    f23 = tf.slice(allCat_crop, [0, 0, 10], [-1, -1, 2], name="f23")
    v1 = tf.slice(allCat_crop, [0, 0, 12], [-1, -1, 1], name="v1")
    v2 = tf.slice(allCat_crop, [0, 0, 13], [-1, -1, 1], name="v2")
    m1 = tf.slice(allCat_crop, [0, 0, 14], [-1, -1, 1], name="m1")
    m2 = tf.slice(allCat_crop, [0, 0, 15], [-1, -1, 1], name="m2")

    batch = tf.train.batch([i1, i2,
                            d1, d2,
                            f12, f23,
                            v1, v2,
                            p1, p2,
                            m1, m2,
                            off_h, off_w],
                           batch_size=hyp.bs,
                           dynamic_pad=True)
    return batch


def euromav_batch(dataset, hyp, shuffle=True):
    with open(dataset) as f:
        content = f.readlines()
    records = [hyp.dataset_location + line.strip() for line in content]
    nRecords = len(records)
    print ('found %d records' % nRecords)
    for record in records:
        assert os.path.isfile(record), 'Record at %s was not found' % record

    queue = tf.train.string_input_producer(records, shuffle=shuffle)

    (h, w, i1_l, i2_l, i1_r, i2_r, p1, p2) = read_and_decode_euromav(queue)

    # Scale images to [-0.5, 0.5]
    i1_l = tf.cast(i1_l, tf.float32) * 1. / 255 - 0.5
    i2_l = tf.cast(i2_l, tf.float32) * 1. / 255 - 0.5
    i1_r = tf.cast(i1_r, tf.float32) * 1. / 255 - 0.5
    i2_r = tf.cast(i2_r, tf.float32) * 1. / 255 - 0.5

    old_w = 752
    old_h = 480
    new_w = 374
    new_h = 240

    top_left_x = old_w // 2 - new_w // 2
    top_left_y = old_h // 2 - new_h // 2

    # Center crop images
    i1_l = tf.image.crop_to_bounding_box(i1_l, top_left_y, top_left_x, new_h, new_w)
    i2_l = tf.image.crop_to_bounding_box(i2_l, top_left_y, top_left_x, new_h, new_w)

    # Grayscale to RGB
    i1_l = tf.tile(i1_l, [1, 1, 3])
    i2_l = tf.tile(i2_l, [1, 1, 3])

    # d1 = d1*v1 # put 0 depth at invalid spots
    # d2 = d2*v2
    # image tensors need to be cropped. we'll do them all at once.
    # allCat = tf.concat(axis=2,values=[i1,i2],
    #                    name="allCat")

    # # image tensors need to be cropped. we'll do them all at once.
    # print_shape(allCat)
    off_h = 0
    off_w = 0
    # allCat = tf.slice(allCat,[off_h,off_w,0],[-1,-1,-1],name="cropped_tensor")
    # # allCat_crop, off_h, off_w = random_crop(allCat,crop_h,crop_w,480,752)
    # print_shape(allCat)
    # i1 = tf.slice(allCat, [0,0,0], [-1,-1,1], name="i1")
    # i2 = tf.slice(allCat, [0,0,1], [-1,-1,1], name="i2")

    batch = tf.train.batch([i1_l, i2_l,
                            i1_r, i2_r,
                            p1, p2],
                           batch_size=hyp.bs,
                           dynamic_pad=True)
    return batch
