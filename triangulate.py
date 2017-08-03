import imageio
import numpy as np
import scipy.misc


def read_flow(filename):
    """
    read optical flow from Middlebury .flo file
    :param filename: name of the flow file
    :return: optical flow data in matrix
    """
    f = open(filename, 'rb')
    magic = np.fromfile(f, np.float32, count=1)
    data2d = None

    if 202021.25 != magic:
        print 'Magic number incorrect. Invalid .flo file'
    else:
        w = np.fromfile(f, np.int32, count=1)
        h = np.fromfile(f, np.int32, count=1)
        print "Reading %d x %d flo file" % (h, w)
        data2d = np.fromfile(f, np.float32, count=2 * w * h)
        # reshape data into 3D array (columns, rows, channels)
        data2d = np.resize(data2d, (h[0], w[0], 2))
    f.close()
    return data2d


def save_two_frames_from_video(video_path):
    imageio.plugins.ffmpeg.download()
    vid = imageio.get_reader(video_path, 'ffmpeg')
    img0 = vid.get_data(0)
    img1 = vid.get_data(10)

    # Resize images to (512, 384) so we can get optical flow
    img0 = scipy.misc.imresize(img0, (384, 512))
    img1 = scipy.misc.imresize(img1, (384, 512))

    imageio.imwrite('./img0.png', img0)
    imageio.imwrite('./img1.png', img1)


# save_two_frames_from_video('./rgbd_dataset_freiburg1_xyz-rgb.avi')
# You need to manually get the flow from these two saved images!

h = 375
w = 1242

# in pixels
fx = 725.0 / w  # focal length x
fy = 725.0 / h  # focal length y
cx = 620.5 / w  # optical center x
cy = 187.0 / h  # optical center y

# https://dsp.stackexchange.com/questions/6055/how-does-resizing-an-image-affect-the-intrinsic-camera-matrix
# We resized the images from (640, 480) to (512, 384) which is a 0.8 reduction.
# fx *= 0.8
# fy *= 0.8
# cx *= 0.8
# cy *= 0.8

# r1,1 r1,2 r1,3 t1 r2,1 r2,2 r2,3 t2 r3,1 r3,2 r3,3 t3
C1 = np.array([-0.265882, 0.0721791, -0.9612997, 7.760932, 0.01772801, 0.9973904,
               0.06998566, 112.3845, 0.9638427, 0.001565997, -0.2664678, -0.9046764])
C1 = np.reshape(C1, [3, 4])
C2 = np.array([-0.2656333, 0.0730738, -0.9613007, 7.853128, 0.01958271, 0.9973266,
               0.07040109, 112.3763, 0.9638752, -0.0001239963, -0.2663542, -2.269278])
C2 = np.reshape(C2, [3, 4])

print C1
print C2


def quat_to_mat(quat):
    """
    https://github.com/bistromath/gr-air-modes/blob/master/python/Quaternion.py
    Transform a unit quaternion into its corresponding rotation matrix (to
    be applied on the right side).

    :returns: transform matrix
    :rtype: numpy array

    """
    x, y, z, w = quat
    xx2 = 2 * x * x
    yy2 = 2 * y * y
    zz2 = 2 * z * z
    xy2 = 2 * x * y
    wz2 = 2 * w * z
    zx2 = 2 * z * x
    wy2 = 2 * w * y
    yz2 = 2 * y * z
    wx2 = 2 * w * x

    rmat = np.empty((3, 3), float)
    rmat[0, 0] = 1. - yy2 - zz2
    rmat[0, 1] = xy2 - wz2
    rmat[0, 2] = zx2 + wy2
    rmat[1, 0] = xy2 + wz2
    rmat[1, 1] = 1. - xx2 - zz2
    rmat[1, 2] = yz2 - wx2
    rmat[2, 0] = zx2 - wy2
    rmat[2, 1] = yz2 + wx2
    rmat[2, 2] = 1. - xx2 - yy2

    return rmat


def get_extrinsic_camera_mat(traj):
    """ returns (3x4) extrinsic camera matrix """
    rot = quat_to_mat(traj[-4:])  # rotation mat from quaternion
    trans = np.array([traj[0:3]]).T  # translation col vec

    # build extrinsic from camera pose: http://ksimek.github.io/2012/08/22/extrinsic/
    # rot = rot.T
    # trans = -np.dot(rot, trans)

    return np.concatenate((rot, trans), 1)


def get_intrinsic_camera_mat(fx, fy, cx, cy):
    """ returns (3x3) intrinsic camera matrix K """
    return np.array([
        [fx, 0., cx],
        [0., fy, cy],
        [0., 0., 1.],
    ])


def get_camera_mat(extr, intr):
    """ returns (3, 4) camera matrix given extrinsic and intrinsic """
    return np.dot(intr, extr)


# Camera matrices (3x4)
P0 = get_camera_mat(C1, get_intrinsic_camera_mat(fx, fy, cx, cy))
P1 = get_camera_mat(C2, get_intrinsic_camera_mat(fx, fy, cx, cy))


print 'Camera matrices:'
print P0
print P1

import cv2


def read_vkitti_png_flow(flow_fn):
    "Convert from .png to (h, w, 2) (flow_x, flow_y) float32 array"
    # read png to bgr in 16 bit unsigned short
    bgr = cv2.imread(flow_fn, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
    h, w, _c = bgr.shape
    assert bgr.dtype == np.uint16 and _c == 3
    # b == invalid flow flag == 0 for sky or other invalid flow
    invalid = bgr[..., 0] == 0
    # g,r == flow_y,x normalized by height,width and scaled to [0;2**16 - 1]
    out_flow = 2.0 / (2**16 - 1.0) * bgr[..., 2:0:-1].astype('f4') - 1
    out_flow[..., 0] *= w - 1
    out_flow[..., 1] *= h - 1
    out_flow[invalid] = 0  # or another value (e.g., np.nan)
    return out_flow


def run():
    count = 0
    img0 = imageio.imread('./vkitti0.png')
    img1 = imageio.imread('./vkitti1.png')
    flow = read_vkitti_png_flow('./vkitti_flow0.png')

    print img0.shape
    print img1.shape
    print flow.shape

    height, width, channels = img0.shape

    depth = np.zeros((height, width))

    for y in xrange(height):
        for x in xrange(width):
            xx = x + flow[y, x, 0]
            yy = y + flow[y, x, 1]

            # TODO: The lmb code does this. unsure if we need to.
            # xx = x + flow[y, x, 0] / width
            # yy = y + flow[y, x, 1] / height

            # Uncomment this to use the algorithm described here:
            # http://www.cs.nyu.edu/~fergus/teaching/vision/9_10_Stereo.pdf, slide 101
            # a0 = x * P0[2, :] - P0[0, :]   # (1 x 4) rows
            # a1 = y * P0[2, :] - P0[1, :]
            # a2 = xx * P1[2, :] - P1[0, :]
            # a3 = yy * P1[2, :] - P1[1, :]
            # b0 = 0.0, b1 = 0.0, b2 = 0.0, b3 = 0.0

            # Uncomment this to use the algorithm used here:
            # https://github.com/MasteringOpenCV/code/blob/master/Chapter4_StructureFromMotion/Triangulation.cpp
            # a0 = x * P0[2, :] - P0[0, :]   # (1 x 4) rows
            # a1 = y * P0[2, :] - P0[1, :]
            # a2 = xx * P1[2, :] - P1[0, :]
            # a3 = yy * P1[2, :] - P1[1, :]
            # b0 = -(x * P0[2, 3] - P0[0, 3])
            # b1 = -(y * P0[2, 3] - P0[1, 3])
            # b2 = -(xx * P1[2, 3] - P1[0, 3])
            # b3 = -(yy * P1[2, 3] - P1[1, 3])

            # Uncomment this to use the lmb algorithm:
            a0 = y * P0[2, :] - P0[1, :]
            a1 = P0[0, :] - x * P0[2, :]
            a2 = yy * P1[2, :] - P1[1, :]
            a3 = P1[0, :] - xx * P1[2, :]
            b0 = P0[1, 3] - y * P0[2, 3]
            b1 = x * P0[2, 3] - P0[0, 3]
            b2 = P1[1, 3] - yy * P1[2, 3]
            b3 = xx * P1[2, 3] - P1[0, 3]

            A = np.vstack((a0, a1, a2, a3))  # (4 x 4)
            b = np.vstack((b0, b1, b2, b3))  # (4 x 1)

            X, _, _, _ = np.linalg.lstsq(A, b)

            depth[y, x] = X[2, 0] / X[3, 0]

    # Normalize depthmap to [0, 255]
    print 'depth min/max/avg: ', np.min(depth), np.max(depth), np.mean(depth)
    depth = (depth - depth.min()) / (depth.max() - depth.min())
    print 'depth min/max/avg: ', np.min(depth), np.max(depth), np.mean(depth)
    imageio.imwrite('./depth.png', depth)


run()
