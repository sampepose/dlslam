import tensorflow as tf
import batcher as bat
from ..utils import split_intrinsics


class Inputs:
    def __init__(self, hyp):
        self.i1_g_t = tf.ones((hyp.bs, hyp.h, hyp.w, 3))
        self.i2_g_t = tf.ones((hyp.bs, hyp.h, hyp.w, 3))
        self.d1_g_t = tf.ones((hyp.bs, hyp.h, hyp.w, 1))
        self.d2_g_t = tf.ones((hyp.bs, hyp.h, hyp.w, 1))
        self.v1_t = tf.ones((hyp.bs, hyp.h, hyp.w, 1))
        self.v2_t = tf.ones((hyp.bs, hyp.h, hyp.w, 1))
        self.p1_g_t = tf.ones((hyp.bs, 4, 4))
        self.p2_g_t = tf.ones((hyp.bs, 4, 4))
        self.m1_g_t = tf.zeros((hyp.bs, hyp.h, hyp.w, 1))
        self.m2_g_t = tf.zeros((hyp.bs, hyp.h, hyp.w, 1))
        self.off_h_t = tf.ones((hyp.bs))
        self.off_w_t = tf.ones((hyp.bs))
        self.fy_t = tf.ones((hyp.bs))
        self.fx_t = tf.ones((hyp.bs))
        self.y0_t = tf.ones((hyp.bs))
        self.x0_t = tf.ones((hyp.bs))

        self.i1_g_v = tf.ones((hyp.bs, hyp.h, hyp.w, 3))
        self.i2_g_v = tf.ones((hyp.bs, hyp.h, hyp.w, 3))
        self.d1_g_v = tf.ones((hyp.bs, hyp.h, hyp.w, 1))
        self.d2_g_v = tf.ones((hyp.bs, hyp.h, hyp.w, 1))
        self.v1_v = tf.ones((hyp.bs, hyp.h, hyp.w, 1))
        self.v2_v = tf.ones((hyp.bs, hyp.h, hyp.w, 1))
        self.p1_g_v = tf.ones((hyp.bs, 4, 4))
        self.p2_g_v = tf.ones((hyp.bs, 4, 4))
        self.m1_g_v = tf.zeros((hyp.bs, hyp.h, hyp.w, 1))
        self.m2_g_v = tf.zeros((hyp.bs, hyp.h, hyp.w, 1))
        self.off_h_v = tf.ones((hyp.bs))
        self.off_w_v = tf.ones((hyp.bs))
        self.fy_v = tf.ones((hyp.bs))
        self.fx_v = tf.ones((hyp.bs))
        self.y0_v = tf.ones((hyp.bs))
        self.x0_v = tf.ones((hyp.bs))

        # get a batch from the dataset
        if hyp.dataset_name == 'SVKITTI':
            (self.i1_g_t, self.i2_g_t,
             self.d1_g_t, self.d2_g_t,
             self.f12_g_t, self.f23_g_t,
             self.v1_t, self.v2_t,
             self.p1_g_t, self.p2_g_t,
             self.m1_g_t, self.m2_g_t,
             self.off_h_t, self.off_w_t) = bat.svkitti_batch(hyp.dataset_t,
                                                             hyp.bs, hyp.h, hyp.w,
                                                             shuffle=True)

            (self.i1_g_v, self.i2_g_v,
             self.d1_g_v, self.d2_g_v,
             self.f12_g_v, self.f23_g_v,
             self.v1_v, self.v2_v,
             self.p1_g_v, self.p2_g_v,
             self.m1_g_v, self.m2_g_v,
             self.off_h_v, self.off_w_v) = bat.svkitti_batch(hyp.dataset_v,
                                                             hyp.bs, hyp.h, hyp.w,
                                                             shuffle=True)
            # for svkitti, let's just grab the intrinsics from hyp
            self.fy_t = tf.cast(tf.tile(tf.reshape(hyp.fy, [1]), [hyp.bs]), tf.float32)
            self.fx_t = tf.cast(tf.tile(tf.reshape(hyp.fx, [1]), [hyp.bs]), tf.float32)
            self.y0_t = tf.cast(tf.tile(tf.reshape(hyp.y0, [1]), [hyp.bs]), tf.float32)
            self.x0_t = tf.cast(tf.tile(tf.reshape(hyp.x0, [1]), [hyp.bs]), tf.float32)
            self.fy_v = tf.cast(tf.tile(tf.reshape(hyp.fy, [1]), [hyp.bs]), tf.float32)
            self.fx_v = tf.cast(tf.tile(tf.reshape(hyp.fx, [1]), [hyp.bs]), tf.float32)
            self.y0_v = tf.cast(tf.tile(tf.reshape(hyp.y0, [1]), [hyp.bs]), tf.float32)
            self.x0_v = tf.cast(tf.tile(tf.reshape(hyp.x0, [1]), [hyp.bs]), tf.float32)

        elif hyp.dataset_name == 'VKITTI':
            (self.i1_g_t, self.i2_g_t,
             self.s1_g_t, self.s2_g_t,
             self.d1_g_t, self.d2_g_t,
             self.f12_g_t, self.f23_g_t,
             self.v1_t, self.v2_t,
             self.p1_g_t, self.p2_g_t,
             self.m1_g_t, self.m2_g_t,
             self.off_h_t, self.off_w_t,
             self.o1c_g_t, self.o1i_g_t, self.o1b_g_t, self.o1p_g_t, self.o1o_g_t,
             self.o2c_g_t, self.o2i_g_t, self.o2b_g_t, self.o2p_g_t, self.o2o_g_t,
             self.nc1_t, self.nc2_t, self.carseg1_g_t, self.carseg2_g_t,
             self.dets1_t, self.feats1_t,
             self.dets2_t, self.feats2_t) = bat.vkitti_batch(hyp.dataset_t,
                                                             hyp.bs, hyp.h, hyp.w,
                                                             shuffle=True)

            self.feats1_t = tf.reshape(self.feats1_t, [-1, 7, 7, 1024])
            self.feats2_t = tf.reshape(self.feats2_t, [-1, 7, 7, 1024])
            self.dets1_t = tf.reshape(self.dets1_t, [-1, 4])
            self.dets2_t = tf.reshape(self.dets2_t, [-1, 4])

            (self.i1_g_v, self.i2_g_v,
             self.s1_g_v, self.s2_g_v,
             self.d1_g_v, self.d2_g_v,
             self.f12_g_v, self.f23_g_v,
             self.v1_v, self.v2_v,
             self.p1_g_v, self.p2_g_v,
             self.m1_g_v, self.m2_g_v,
             self.off_h_v, self.off_w_v,
             self.o1c_g_v, self.o1i_g_v, self.o1b_g_v, self.o1p_g_v, self.o1o_g_v,
             self.o2c_g_v, self.o2i_g_v, self.o2b_g_v, self.o2p_g_v, self.o2o_g_v,
             self.nc1_v, self.nc2_v, self.carseg1_g_v, self.carseg2_g_v,
             self.dets1_v, self.feats1_v,
             self.dets2_v, self.feats2_v) = bat.vkitti_batch(hyp.dataset_v,
                                                             hyp.bs, hyp.h, hyp.w,
                                                             shuffle=True)

            self.feats1_v = tf.reshape(self.feats1_v, [-1, 7, 7, 1024])
            self.feats2_v = tf.reshape(self.feats2_v, [-1, 7, 7, 1024])
            self.dets1_v = tf.reshape(self.dets1_v, [-1, 4])
            self.dets2_v = tf.reshape(self.dets2_v, [-1, 4])

            self.O12_t = [[self.o1c_g_t,
                           self.o1i_g_t,
                           self.o1b_g_t,
                           self.o1p_g_t,
                           self.o1o_g_t],
                          [self.o2c_g_t,
                           self.o2i_g_t,
                           self.o2b_g_t,
                           self.o2p_g_t,
                           self.o2o_g_t]]
            self.O12_v = [[self.o1c_g_v,
                           self.o1i_g_v,
                           self.o1b_g_v,
                           self.o1p_g_v,
                           self.o1o_g_v],
                          [self.o2c_g_v,
                           self.o2i_g_v,
                           self.o2b_g_v,
                           self.o2p_g_v,
                           self.o2o_g_v]]

            self.carthings_t = [self.nc1_t, self.nc2_t,
                                self.carseg1_g_t, self.carseg2_g_t]
            self.carthings_v = [self.nc1_v, self.nc2_v,
                                self.carseg1_g_v, self.carseg2_g_v]

            # for vkitti, let's just grab the intrinsics from hyp
            self.fy_t = tf.cast(tf.tile(tf.reshape(hyp.fy, [1]), [hyp.bs]), tf.float32)
            self.fx_t = tf.cast(tf.tile(tf.reshape(hyp.fx, [1]), [hyp.bs]), tf.float32)
            self.y0_t = tf.cast(tf.tile(tf.reshape(hyp.y0, [1]), [hyp.bs]), tf.float32)
            self.x0_t = tf.cast(tf.tile(tf.reshape(hyp.x0, [1]), [hyp.bs]), tf.float32)
            self.fy_v = tf.cast(tf.tile(tf.reshape(hyp.fy, [1]), [hyp.bs]), tf.float32)
            self.fx_v = tf.cast(tf.tile(tf.reshape(hyp.fx, [1]), [hyp.bs]), tf.float32)
            self.y0_v = tf.cast(tf.tile(tf.reshape(hyp.y0, [1]), [hyp.bs]), tf.float32)
            self.x0_v = tf.cast(tf.tile(tf.reshape(hyp.x0, [1]), [hyp.bs]), tf.float32)
        elif hyp.dataset_name == 'TOY':
            (self.i1_g_t, self.i2_g_t,
             self.d1_g_t, self.d2_g_t,
             self.v1_t, self.v2_t,
             self.m1_g_t, self.m2_g_t,
             self.off_h_t, self.off_w_t) = bat.toy_batch(hyp.dataset_t,
                                                         hyp.bs, hyp.h, hyp.w,
                                                         shuffle=hyp.shuffle_train)
            (self.i1_g_v, self.i2_g_v,
             self.d1_g_v, self.d2_g_v,
             self.v1_v, self.v2_v,
             self.m1_g_v, self.m2_g_v,
             self.off_h_v, self.off_w_v) = bat.toy_batch(hyp.dataset_v,
                                                         hyp.bs, hyp.h, hyp.w,
                                                         shuffle=hyp.shuffle_val)
            # grab focals from hyp
            self.fy_t = tf.cast(tf.tile(tf.reshape(hyp.fy, [1]), [hyp.bs]), tf.float32)
            self.fx_t = tf.cast(tf.tile(tf.reshape(hyp.fx, [1]), [hyp.bs]), tf.float32)
            self.fy_v = tf.cast(tf.tile(tf.reshape(hyp.fy, [1]), [hyp.bs]), tf.float32)
            self.fx_v = tf.cast(tf.tile(tf.reshape(hyp.fx, [1]), [hyp.bs]), tf.float32)
        elif hyp.dataset_name == 'STOY':
            (self.i1_g_t, self.i2_g_t,
             self.d1_g_t, self.d2_g_t,
             self.v1_t, self.v2_t,
             self.off_h_t, self.off_w_t) = bat.stoy_batch(hyp.dataset_t,
                                                          hyp.bs, hyp.h, hyp.w,
                                                          shuffle=hyp.shuffle_train)
            (self.i1_g_v, self.i2_g_v,
             self.d1_g_v, self.d2_g_v,
             self.v1_v, self.v2_v,
             self.off_h_v, self.off_w_v) = bat.stoy_batch(hyp.dataset_v,
                                                          hyp.bs, hyp.h, hyp.w,
                                                          shuffle=hyp.shuffle_val)
            rgbK = tf.constant([[614.5777587890625, 0.0, 311.260986328125],
                                [0.0, 614.577880859375, 240.79934692382812],
                                [0.0, 0.0, 1.0]])
            k = tf.tile(tf.cast(tf.reshape(rgbK, [1, 3, 3]),
                                tf.float32), [hyp.bs, 1, 1])
            # grab focals from k
            fy, fx, y0, x0 = split_intrinsics(k)
            self.fy_t = fy * hyp.scale
            self.fx_t = fx * hyp.scale
            self.y0_t = (y0 * hyp.scale) - tf.cast(self.off_h_t, tf.float32)
            self.x0_t = (x0 * hyp.scale) - tf.cast(self.off_w_t, tf.float32)
            self.fy_v = fy * hyp.scale
            self.fx_v = fx * hyp.scale
            self.y0_v = (y0 * hyp.scale) - tf.cast(self.off_h_v, tf.float32)
            self.x0_v = (x0 * hyp.scale) - tf.cast(self.off_w_v, tf.float32)
        elif (hyp.dataset_name == 'VSTOY' or hyp.dataset_name == 'JUMBLE'):
            (self.i1_g_t, self.i2_g_t,
             self.d1_g_t, self.d2_g_t,
             self.v1_t, self.v2_t,
             self.off_h_t, self.off_w_t) = bat.stoy_batch(hyp.dataset_t,
                                                          hyp.bs, hyp.h, hyp.w,
                                                          shuffle=hyp.shuffle_train)
            (self.i1_g_v, self.i2_g_v,
             self.d1_g_v, self.d2_g_v,
             self.v1_v, self.v2_v,
             self.off_h_v, self.off_w_v) = bat.stoy_batch(hyp.dataset_v,
                                                          hyp.bs, hyp.h, hyp.w,
                                                          shuffle=hyp.shuffle_val)
            rgbK = tf.constant([[621.6669921875, 0.0, 309.56036376953125],
                                [0.0, 621.6670532226562, 242.37942504882812],
                                [0.0, 0.0, 1.0]])
            k = tf.tile(tf.cast(tf.reshape(rgbK, [1, 3, 3]),
                                tf.float32), [hyp.bs, 1, 1])
            # grab focals from k
            fy, fx, y0, x0 = split_intrinsics(k)
            self.fy_t = fy * hyp.scale
            self.fx_t = fx * hyp.scale
            self.y0_t = (y0 * hyp.scale) - tf.cast(self.off_h_t, tf.float32)
            self.x0_t = (x0 * hyp.scale) - tf.cast(self.off_w_t, tf.float32)
            self.fy_v = fy * hyp.scale
            self.fx_v = fx * hyp.scale
            self.y0_v = (y0 * hyp.scale) - tf.cast(self.off_h_v, tf.float32)
            self.x0_v = (x0 * hyp.scale) - tf.cast(self.off_w_v, tf.float32)
        elif hyp.dataset_name == 'YCB':
            (self.i1_g_t, self.i2_g_t,
             self.d1_g_t, self.d2_g_t,
             self.v1_t, self.v2_t,
             self.m1_g_t, self.m2_g_t,
             self.p1_g_t, self.p2_g_t,
             self.k1_t, self.k2_t,
             self.off_h_t, self.off_w_t) = bat.ycb_batch(hyp.dataset_t,
                                                         hyp.bs, hyp.h, hyp.w,
                                                         shuffle=hyp.shuffle_train)
            (self.i1_g_v, self.i2_g_v,
             self.d1_g_v, self.d2_g_v,
             self.v1_v, self.v2_v,
             self.m1_g_v, self.m2_g_v,
             self.p1_g_v, self.p2_g_v,
             self.k1_v, self.k2_v,
             self.off_h_v, self.off_w_v) = bat.ycb_batch(hyp.dataset_v,
                                                         hyp.bs, hyp.h, hyp.w,
                                                         shuffle=hyp.shuffle_val)
            fx_t, fy_t, x0_t, y0_t = split_intrinsics(self.k1_t)
            self.fx_t = fx_t * hyp.scale
            self.fy_t = fy_t * hyp.scale
            self.y0_t = (y0_t * hyp.scale) - 240
            self.x0_t = (x0_t * hyp.scale) - 320
            fx_v, fy_v, x0_v, y0_v = split_intrinsics(self.k1_v)
            self.fx_v = fx_v * hyp.scale
            self.fy_v = fy_v * hyp.scale
            self.y0_v = (y0_v * hyp.scale) - 240
            self.x0_v = (x0_v * hyp.scale) - 320
        else:
            assert False

        self.train_inputs = [
            self.i1_g_t, self.i2_g_t,
            self.f12_g_t,
            self.d1_g_t, self.d2_g_t,
            self.v1_t, self.v2_t,
            self.m1_g_t, self.m2_g_t,
            self.p1_g_t, self.p2_g_t,
            self.off_h_t, self.off_w_t,
            self.fy_t, self.fx_t,
            self.y0_t, self.x0_t
        ]

        self.val_inputs = [
            self.i1_g_v, self.i2_g_v,
            self.f12_g_v,
            self.d1_g_v, self.d2_g_v,
            self.v1_v, self.v2_v,
            self.m1_g_v, self.m2_g_v,
            self.p1_g_v, self.p2_g_v,
            self.off_h_v, self.off_w_v,
            self.fy_v, self.fx_v,
            self.y0_v, self.x0_v
        ]
