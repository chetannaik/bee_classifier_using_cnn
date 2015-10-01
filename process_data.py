__author__ = 'chetannaik'

from PIL import Image
from tqdm import tqdm

import numpy as np
import pandas as pd
import os
import shutil
import time

# config
# np.random.seed(42)
size = (48, 48)
num_channels = 3
num_apis_images = 827
num_bombus_images = 3142
train_data_dir = "dataset/images/train/"
train_labels_file = "dataset/train_labels.csv"


# copy files to appropriate directories based on the class
def copy_files():
	print("-- Copying files to data/train directory")
	# Parameters
	labels = pd.read_csv(train_labels_file)
	labels.index = labels.id
	labels = labels['genus']
	labels = labels.T.to_dict()

	if not os.path.exists("data/train/bombus/"):
		os.makedirs("data/train/bombus/")

	if not os.path.exists("data/train/apis/"):
		os.makedirs("data/train/apis/")

	for infile in tqdm(os.listdir(train_data_dir)):
		if infile == ".DS_Store":
			continue
		if labels[int(infile[:-4])] == 1.0:
			# Bombus
			file_src = train_data_dir + infile
			file_dest = 'data/train/bombus/' + infile
			shutil.copy2(file_src, file_dest)
		elif labels[int(infile[:-4])] == 0.0:
			# Apis
			file_src = train_data_dir + infile
			file_dest = 'data/train/apis/' + infile
			shutil.copy2(file_src, file_dest)


# resize images
def resize_images():
	print("-- Resizing images to {} and copying it to {} directory".format(
		str(size[0])+"x"+str(size[1]), "data/thumbnails"))
	for bee in ['bombus', 'apis']:
		if not os.path.exists("data/thumbnails/" + bee):
			os.makedirs("data/thumbnails/" + bee)
		input_dir = "data/train/" + bee + os.sep
		output_dir = "data/thumbnails/" + bee + os.sep
		index = 0
		for infile in tqdm(os.listdir(input_dir)):
			if infile == ".DS_Store":
				continue
			try:
				im = Image.open(input_dir + infile)
				outfile = "{0:04d}".format(index) + ".jpg"
				im = im.resize(size, Image.ANTIALIAS)
				im.save(output_dir + outfile, "JPEG")
				index += 1
			except IOError:
				print("cannot resize", infile)


# create data matrix
def create_matrix():
	print("-- Creating data matrices")
	if not os.path.exists("data/matrix/"):
		os.makedirs("data/matrix/")

	for bee in ['bombus', 'apis']:
		input_dir = "data/thumbnails/" + bee + os.sep
		output_dir = "data/matrix/"
		output_filename = bee + ".dat"

		with open(output_dir + output_filename, "wb") as outfile:
			for infile in tqdm(os.listdir(input_dir)):
				if infile == ".DS_Store":
					continue

				im = Image.open(input_dir + infile)
				# convert to grayscale
				im_converted = im.convert(mode="L")
				r_vec, g_vec, b_vec, l_vec = [], [], [], []
				for i in range(0, size[0]):
					for j in range(0, size[1]):
						r, g, b = im.getpixel((i, j))
						l = im_converted.getpixel((i, j))
						r_vec.append(r)
						g_vec.append(g)
						b_vec.append(b)
						l_vec.append(l)
				#npa = np.asarray(l_vec, dtype=np.uint8)
				npa = np.asarray(r_vec + g_vec + b_vec, dtype=np.uint8)
				npa.tofile(outfile)


# Reflection about vertical axis
def reflected_image_matrix(data):
	reflected_images = []
	for image in tqdm(data):
		reflected_channels = []
		for channel in image:
			reflected_rows = []
			for row in channel:
				reflected_rows.append(row[::-1].tolist())
			reflected_channels.append(reflected_rows)
		reflected_images.append(reflected_channels)
	return np.asarray(reflected_images, dtype=np.uint8)


# Translation with wrapping
def translate(data, direction=None):
	if direction is None or direction not in ["up", "down", "left", "right"]:
		return None

	trans_images = []
	for image in tqdm(data):
		trans_channels = []
		for channel in image:
			if direction == "right":
				trans_rows = []
				for row in channel:
					trans_rows.append(row[-5:].tolist() + row[:-5].tolist())
				trans_channels.append(trans_rows)
			elif direction == "left":
				trans_rows = []
				for row in channel:
					trans_rows.append(row[5:].tolist() + row[:5].tolist())
				trans_channels.append(trans_rows)
			elif direction == "up":
				tmp1 = channel[5:].tolist() + channel[:5].tolist()
				trans_channels.append(tmp1)
			elif direction == "down":
				tmp1 = channel[-5:].tolist() + channel[:-5].tolist()
				trans_channels.append(tmp1)
		trans_images.append(trans_channels)
	return np.asarray(trans_images, dtype=np.uint8)


def generate_data():
	print("-- Generating synthetic dataset by reflecting and translating images")
	if not os.path.exists("data/generated"):
		os.makedirs("data/generated")

	# read apis and bombus data file
	data_apis_file = open("data/matrix/apis.dat", "rb")
	apis_data = np.fromfile(data_apis_file, dtype=np.uint8)
	apis_data = apis_data.reshape((num_apis_images, num_channels, size[0], size[1]))
	data_apis_file.close()
	data_bombus_file = open("data/matrix/bombus.dat", "rb")
	bombus_data = np.fromfile(data_bombus_file, dtype=np.uint8)
	bombus_data = bombus_data.reshape((num_bombus_images, num_channels, size[0], size[1]))
	data_bombus_file.close()

	# generate reflected images
	refl_apis_data = reflected_image_matrix(apis_data)
	refl_bombus_data = reflected_image_matrix(bombus_data)

	# generate translated images
	orig_trans_apis_data = []
	orig_trans_bombus_data = []
	refl_trans_apis_data = []
	refl_trans_bombus_data = []
	for d in ["up", "down", "left", "right"]:
		orig_trans_apis_data.append(translate(apis_data, d))
		orig_trans_bombus_data.append(translate(bombus_data, d))
		refl_trans_apis_data.append(translate(refl_apis_data, d))
		refl_trans_bombus_data.append(translate(refl_bombus_data, d))
	all_apis_data = np.asarray([apis_data] + [refl_apis_data] + orig_trans_apis_data + refl_trans_apis_data)
	all_bombus_data = np.asarray([bombus_data] + [refl_bombus_data] + orig_trans_bombus_data + refl_trans_bombus_data)
	all_apis_data.tofile("data/generated/all_apis_data.dat")
	all_bombus_data.tofile("data/generated/all_bombus_data.dat")
	print("-- Data files generated into data/generated directory")


def shuffle_data(a, b):
	# inplace shuffle
	p = np.random.permutation(len(a))
	return a[p], b[p]


def undersample(data, sample_size):
	# shuffle and undersample
	p = np.random.permutation(len(data))
	shuffled_data  = data[p]
	return shuffled_data[:sample_size]


def prepare_dataset():
	print("-- Preparing train, test and validation dataset")
	# read in data
	data_apis_file = open("data/generated/all_apis_data.dat", "rb")
	data_bombus_file = open("data/generated/all_bombus_data.dat", "rb")
	apis_data = np.fromfile(data_apis_file, dtype=np.uint8)
	bombus_data = np.fromfile(data_bombus_file, dtype=np.uint8)
	# reshape back to image dimentions (as defined in config)
	apis_data = apis_data.reshape((-1, num_channels, size[0], size[1]))
	bombus_data = bombus_data.reshape((-1, num_channels, size[0], size[1]))
	# fix class imbalance by undersampling
	bombus_data = undersample(bombus_data, len(apis_data))
	# generate image labels based on the class
	y_apis = np.empty(len(apis_data), dtype=np.uint8)
	y_apis.fill(0)
	y_bombus = np.empty(len(bombus_data), dtype=np.uint8)
	y_bombus.fill(1)
	# concatenate all data
	X = np.concatenate((apis_data, bombus_data))
	y = np.concatenate((y_apis, y_bombus))
	# cast X data as floating point (single precision)
	X = X.astype(np.float32)
	# shuffle data
	X, y = shuffle_data(X, y)
	# break into 80:10:10 train:validation:test
	X_test = X[:(len(X) // 10)]
	X_val = X[(len(X) // 10):(2 * (len(X) // 10))]
	X_train = X[(2 * (len(X) // 10)):]
	y_test = y[:(len(y) // 10)]
	y_val = y[(len(y) // 10):(2 * (len(y) // 10))]
	y_train = y[(2 * (len(y) // 10)):]
	# Save sets to disk
	X_train.tofile("data/generated/X_train.dat")
	X_val.tofile("data/generated/X_val.dat")
	X_test.tofile("data/generated/X_test.dat")

	y_train.tofile("data/generated/y_train.dat")
	y_val.tofile("data/generated/y_val.dat")
	y_test.tofile("data/generated/y_test.dat")


def main():
	t_start = time.time()
	copy_files()
	resize_images()
	create_matrix()
	generate_data()
	prepare_dataset()
	print("-- Done!")
	print("Time taken to preprocess data: {}".format(time.time() - t_start))


if __name__ == '__main__':
	main()