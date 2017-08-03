import tensorflow as tf
import numpy as np
import os
import sys
from transforms3d.quaternions import *
import csv
import cv2
outDir = "../records"
inDir = "../datasets/euromav"
count = 0
if not os.path.exists(outDir):
    os.makedirs(outDir)


def _floats_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _int64s_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def recorder(prev_image_l, curr_image_l, prev_image_r, curr_image_r, height, width, p1, p2):
    global count
    count += 1
    sys.stdout.write("\r processed %d sample" % count)
    sys.stdout.flush()
    recordFile = "%s/sample_%d.tfrecord" % (outDir, count)
    compress = tf.python_io.TFRecordOptions(
        compression_type=tf.python_io.TFRecordCompressionType.GZIP)
    writer = tf.python_io.TFRecordWriter(recordFile, options=compress)
    prev_img_l = prev_image_l.tostring()
    curr_img_l = curr_image_l.tostring()
    prev_img_r = prev_image_r.tostring()
    curr_img_r = curr_image_r.tostring()
    p1_raw = p1.flatten().tolist()
    p2_raw = p2.flatten().tolist()
    example = tf.train.Example(features=tf.train.Features(feature={
        'height': _int64_feature(height),
        'width': _int64_feature(width),
        'prev_img_l': _bytes_feature(prev_img_l),
        'curr_img_l': _bytes_feature(curr_img_l),
        'prev_img_r': _bytes_feature(prev_img_r),
        'curr_img_r': _bytes_feature(curr_img_r),
        'p1_raw': _floats_feature(p1_raw),
        'p2_raw': _floats_feature(p2_raw),
    }))
    writer.write(example.SerializeToString())
    writer.close()


def read_camera_sensor(file_name):  # for camera because not float data string data
    with open(file_name) as csvfile:
        readCSV = csv.reader(csvfile, delimiter=',')
        next(readCSV)
        time = np.array([], dtype=np.float64)
        sensor_reading = np.array([], dtype=np.float64)
        for row in readCSV:
            data = row[1]
            t = np.float64(row[0])
            if (time.size == 0):
                time = t
            else:
                time = np.vstack((time, t))
            if (np.size(sensor_reading) == 0):
                sensor_reading = data
            else:
                sensor_reading = np.vstack((sensor_reading, data))
    return (time, sensor_reading)


def read_sensor(file_name):  # common for imu and vicon
    with open(file_name) as csvfile:
        readCSV = csv.reader(csvfile, delimiter=',')
        next(readCSV)
        time = np.array([], dtype=np.float64)
        sensor_reading = np.array([], dtype=np.float64)
        for row in readCSV:
            data = np.float64(row[1:None])
            t = np.float64(row[0])
            if (time.size == 0):
                time = t
            else:
                time = np.vstack((time, t))
            if (sensor_reading.size == 0):
                sensor_reading = data
            else:
                sensor_reading = np.vstack((sensor_reading, data))
    # print(np.shape(time))
    # print(np.shape(sensor_reading))
    return (time, sensor_reading)
# if __name__ == '__main__':


def write_tfrecords():
    cam_freq = 10  # required camera freq
    stride = (20 / cam_freq)  # given 20hz as cam freq in specs
    n_files = 2  # TOOD: We only want the trajectory for one scene!
    sys.stdout.write("\n Creating TF Records \n --------------------")
    for j in range(1, n_files):
        sys.stdout.write("\n reading vicon room %d \n" % j)
        # 20 Hz camera
        cam_file = inDir + '/v' + str(j) + '/mav0/cam0/data.csv'
        (cam_time, cam_data) = read_camera_sensor(cam_file)
        # 100 Hz vicon
        vicon_file = inDir + '/v' + str(j) + '/mav0/vicon0/data.csv'
        (vicon_time, vicon_data) = read_sensor(vicon_file)
        t_p = 0  # previous time

        img_path_l = inDir + '/v' + str(j) + '/mav0/cam0/data/'
        img_path_r = inDir + '/v' + str(j) + '/mav0/cam1/data/'

        for i in range(1, len(cam_time) - stride):
            # print(np.argmin(abs(vicon_time-imu_time[i,0])), np.min(abs(vicon_time-imu_time[i,0])))
            if (t_p == np.argmin(abs(vicon_time - cam_time[i, 0]))):
                continue
            else:
                t_p = np.argmin(abs(vicon_time - cam_time[i, 0]))
                # sys.stdout.write("%d\n"%np.min(abs(vicon_time-cam_time[i,0])))
                prev_pose = vicon_data[np.argmin(abs(vicon_time - cam_time[i, 0])), None]
                prev_image_l = np.array(cv2.imread(
                    img_path_l + cam_data[i, 0], cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH))
                curr_image_l = np.array(cv2.imread(
                    img_path_l + cam_data[i + stride, 0], cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH))

                prev_image_r = np.array(cv2.imread(
                    img_path_r + cam_data[i, 0], cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH))
                curr_image_r = np.array(cv2.imread(
                    img_path_r + cam_data[i + stride, 0], cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH))

                curr_pose = vicon_data[np.argmin(abs(vicon_time - cam_time[i + stride, 0])), None]
                height = prev_image_l.shape[0]
                width = prev_image_l.shape[1]
                # pose 1
                rot1 = quat2mat(prev_pose[0, 3:7])
                t1 = np.reshape(prev_pose[0, 0:3], [1, 3])
                T1 = np.hstack((rot1, np.transpose(t1)))
                T1 = np.vstack((T1, np.array([0, 0, 0, 1])))
                # pose 2
                rot2 = quat2mat(curr_pose[0, 3:7])
                t2 = np.reshape(curr_pose[0, 0:3], [1, 3])
                T2 = np.hstack((rot2, np.transpose(t2)))
                T2 = np.vstack((T2, np.array([0, 0, 0, 1])))
                T_rel = np.matmul(np.linalg.inv(T2), T1)
                rel_pose = np.hstack((np.transpose(T_rel[0:3, 3]), mat2quat(T_rel[0:3, 0:3])))
                recorder(prev_image_l, curr_image_l, prev_image_r,
                         curr_image_r, height, width, T1, T2)
    print('tf records created in records folder..')


if __name__ == '__main__':
    write_tfrecords()
