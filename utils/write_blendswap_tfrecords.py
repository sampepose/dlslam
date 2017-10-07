import os
import sys
sys.path.append('../src')

import numpy as np
import tensorflow as tf
import transforms3d.axangles
import transforms3d.euler
from scipy.misc import imread, imsave
from utils import zrt2flow, ominus, safe_inverse

import lmbspecialops as sops

root_path = '/projects/katefgroup/datasets/synthetic_renderings/demon_scenes_annotated/'
out_dir_name = 'tfrecords'


def _floats_feature(value):
  return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def write_tfrecord(output_path, image1, image2, relativeTranslation1to2, relativeRotation1to2, depth1, flow1to2):
	# Construct writer
	writer_options = tf.python_io.TFRecordOptions(compression_type=tf.python_io.TFRecordCompressionType.GZIP)
	writer = tf.python_io.TFRecordWriter(output_path, options=writer_options)

	image1 = image1.astype(np.uint8)
	image2 = image2.astype(np.uint8)
	relativeTranslation1to2 = relativeTranslation1to2.astype(np.float32)
	relativeRotation1to2 = relativeRotation1to2.astype(np.float32)
	depth1 = depth1.astype(np.float32)
	flow1to2 = flow1to2.astype(np.float32)

	# Encode data into an Example
	example = tf.train.Example(features=tf.train.Features(feature={
	  'image1': _bytes_feature(image1.tostring()),
  	  'image2': _bytes_feature(image2.tostring()),
  	  'relativeTranslation1to2': _floats_feature(relativeTranslation1to2.flatten().tolist()),
  	  'relativeRotation1to2': _floats_feature(relativeRotation1to2.flatten().tolist()),
	  'depth1': _bytes_feature(depth1.tostring()),
	  'flow1to2': _bytes_feature(flow1to2.tostring()),	
  	}))

  	# Write to disk
	writer.write(example.SerializeToString())
	writer.close()


def main():
	# Gather list of subdirectories in root_path. Each subdirectory represents a scene.
	scene_directories = next(os.walk(root_path))[1]
	if out_dir_name in scene_directories:
		scene_directories.remove(out_dir_name)

	print("Creating TFRecords for %d scenes in: %s\n" % (len(scene_directories), root_path))

	config = tf.ConfigProto()
	config.gpu_options.allow_growth = True
	with tf.Session(config=config) as sess:
		# Iterate through each scene
		total_count = 0
		for idx, directory_name in enumerate(scene_directories):
			directory_path = os.path.join(root_path, directory_name)

			if idx < 5:
				continue

			# Count of image pairs for this scene
			num_pairs = len([name for name in os.listdir(directory_path) if name.endswith('.npz') and name.startswith('meta')])
			total_count += num_pairs

			for i in range(num_pairs):
				print "[{0}/{1}]\tWriting {2}/{3} TFRecords for scene: {4}\r".format(idx, len(scene_directories), i, num_pairs, directory_name),
				sys.stdout.flush()
				metadata = np.load(os.path.join(directory_path, 'meta%05d.npz' % i))

				if i > 5:
					break

				# Extract metadata
				image1_path = metadata['image1'][()]
				image2_path = metadata['image2'][()]
				translation1 = metadata['camera_loc1']
				rotation1 = metadata['camera_euler1']
				translation2 = metadata['camera_loc2']
				rotation2 = metadata['camera_euler2']
				depth1_path = metadata['depth1'][()]
				depth2_path = metadata['depth2'][()]

				# Read images
				image1 = imread(image1_path)[:, :, 0:3]
				image2 = imread(image2_path)[:, :, 0:3]

				# Should be 640 x 480 images
				height, width, _ = image1.shape

				# Read depths
				depth1 = np.load(depth1_path)
				depth2 = np.load(depth2_path)

				# Euler -> rotation matrix
				# Note: Blender uses +x right, +y forwards, +z up coordinate system.
				# We want to use a +x left, +y up, +z backwards coordinate system.
				transformation = np.array([[-1., 0., 0.], [0., 0., 1.], [0., -1., 0.]])
				rotation1_mat = transformation.dot(transforms3d.euler.euler2mat(rotation1[0], rotation1[1], rotation1[2]))
				rotation2_mat = transformation.dot(transforms3d.euler.euler2mat(rotation2[0], rotation2[1], rotation2[2]))
				translation1_transformed = transformation.dot(translation1)
				translation2_transformed = transformation.dot(translation2)

				# Create pose matrices
				p1 = np.eye(4)
				p1[0:3, 0:3] = rotation1_mat
				p1[0:3, 3] = translation1_transformed
				p2 = np.eye(4)
				p2[0:3, 0:3] = rotation2_mat
				p2[0:3, 3] = translation2_transformed
				p1 = np.expand_dims(p1, 0)
				p2 = np.expand_dims(p2, 0)

				# Compute relative camera motion between image pair
				rel_rt = sess.run(ominus(tf.constant(p2, dtype=tf.float32), tf.constant(p1, dtype=tf.float32)))
				relativeRotation1to2Mat = rel_rt[0, 0:3, 0:3]
				relativeTranslation1to2 = rel_rt[0, 0:3, 3]

				# Rotation matrix -> rotation axis (vector of rotation where magnitude is the angle)
				relativeRotation1to2, angle = transforms3d.axangles.mat2axangle(relativeRotation1to2Mat)
				relativeRotation1to2 *= angle

				# Assume blender default sensor values
				sensor_width = 32.0
				sensor_height = 18.0
				f_in_mm = float(metadata['focal_length1'][()]) # Assume same intrinsics for both images
				fx = f_in_mm / sensor_width # Normalized focal length
				fy = f_in_mm / sensor_height

				depth_expanded = tf.constant(np.expand_dims(depth1, 0), dtype=tf.float32)
				rot_expanded = tf.constant(np.expand_dims(relativeRotation1to2Mat, 0), dtype=tf.float32)
				t_expanded = tf.constant(np.expand_dims(relativeTranslation1to2, 0), dtype=tf.float32)
				fx_expanded = tf.constant([fx * width], dtype=tf.float32)
				fy_expanded = tf.constant([fy * height], dtype=tf.float32)
				xo_expanded = tf.constant([0.5 * width], dtype=tf.float32)
				yo_expanded = tf.constant([0.5 * height], dtype=tf.float32)
				flow_alt_tf, _ = zrt2flow(depth_expanded, rot_expanded, t_expanded, fx_expanded, fy_expanded, xo_expanded, yo_expanded)
				flow1to2 = sess.run(flow_alt_tf)

				# Write TFRecord to disk
				write_tfrecord(os.path.join(root_path, out_dir_name, directory_name + ('%05d' % i) + '.tfrecord'),
							   image1, image2, relativeTranslation1to2, relativeRotation1to2, depth1, flow1to2)
			print('\n')
		print("Wrote a total of %d TFRecords" % total_count)
		print("Execute:\n")
		print("cd " + root_path + " && ls tfrecords | shuf | split -l $(expr $(ls tfrecords | wc -l) \* 80 / 100) - out && mv outaa train.txt && mv outab val.txt && cd -\n")
		print("to generate an 80-20 train-test split")

if __name__ == '__main__':
	main()

