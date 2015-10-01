__author__ = 'chetannaik'

import os
import csv

import numpy as np
import pandas as pd
import theano
import theano.tensor as T

import lasagne

from PIL import Image
from tqdm import tqdm

# config
size = (48, 48)
num_channels = 3
num_classes = 2
data_dir = "dataset/"
test_data_dir = data_dir + "images/test/"
result_dir = "results/"
models_dir = "models/"


class Predictor:

    def __init__(self):
        self.network = self.load_network()

    @staticmethod
    def load_network():
        input_var = T.tensor4('inputs')
        network = Predictor.build_cnn(input_var)
        read_filename = "model_beenet_color.npz"
        print("Loading parameters: {}".format(read_filename))
        loaded_params = np.load(models_dir + read_filename)
        lasagne.layers.set_all_param_values(network, loaded_params['arr_0'])
        return network

    @staticmethod
    def build_cnn(input_var=None):
        # Neural network model
        network = lasagne.layers.InputLayer(shape=(None, num_channels, size[0], size[1]), input_var=input_var)
        network = lasagne.layers.Conv2DLayer(
                network, num_filters=32, filter_size=(5, 5),
                nonlinearity=lasagne.nonlinearities.rectify,
                W=lasagne.init.GlorotUniform())
        network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 2))
        network = lasagne.layers.Conv2DLayer(
                network, num_filters=32, filter_size=(5, 5),
                nonlinearity=lasagne.nonlinearities.rectify)
        network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 2))
        network = lasagne.layers.DenseLayer(
                lasagne.layers.dropout(network, p=.2),
                num_units=128,
                nonlinearity=lasagne.nonlinearities.sigmoid)
        network = lasagne.layers.DenseLayer(network,
                num_units=num_classes,
                nonlinearity=lasagne.nonlinearities.softmax)
        return network

    def predict(self, filename):
        input = Predictor.process_input_image(filename)
        prediction = lasagne.layers.get_output(self.network, inputs=input, deterministic=True)
        prediction = prediction.eval()[0]  # Evaluate the Theano object
        return {"apis": prediction[0], "bombus": prediction[1]}

    @staticmethod
    def process_input_image(filename):
        im = Image.open(filename)
        im = im.resize(size)
        r_vec = []
        g_vec = []
        b_vec = []
        for i in range(0, size[0]):
            for j in range(0, size[1]):
                r, g, b = im.getpixel((i, j))
                r_vec.append(r)
                g_vec.append(g)
                b_vec.append(b)
        npa = np.asarray(r_vec + g_vec + b_vec, dtype=np.float32)
        npa = npa.reshape((-1, num_channels, size[0], size[1]))
        return npa

def main(beenet):
    print("-- Looking at bee images and predicting bee type")

    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    submission_data = []
    for infile in tqdm(os.listdir(test_data_dir)):
        if infile == ".DS_Store":
            continue
        filename = test_data_dir + infile
        result = beenet.predict(filename)
        if result['apis'] >= result['bombus']:
            val = 1 - result['apis']
        else:
            val = result['bombus']
        submission_data.append((int(infile[:-4]), val))
    sub_df = pd.DataFrame(submission_data)
    sub_df.columns = ['id', 'genus']
    sub_df.to_csv(result_dir + "submission_temp.csv", index=False)

    sfdf = pd.read_csv(data_dir + "SubmissionFormat.csv")
    pdf = pd.read_csv(result_dir + "submission_temp.csv")
    ppdf = pdf[['id', 'genus']]
    ppdf.index = ppdf.id
    ppdf = ppdf['genus']
    ddf = ppdf.T.to_dict()
    data = [['id', 'genus']]
    for rid, row in tqdm(sfdf.iterrows()):
        fid = int(row['id'])
        val = round(ddf[fid], 2)
        data.append([fid, val])
    with open(result_dir + 'submission.csv', 'wb') as fp:
        a = csv.writer(fp, delimiter=',')
        a.writerows(data)
    print("-- Generated the submission file 'submission.csv'")


if __name__ == '__main__':
    beenet = Predictor()
    main(beenet)
