from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from flask import Flask, request, jsonify
import tensorflow as tf
import scipy.misc as sic
import numpy as np
from functools import partial
from cap.calDepthMap import get_tmap
from lib.model import data_loader, generator, SRGAN, inference_data_loader, save_images, SRResnet
from lib.ops import *
import collections
import atexit
import signal
import math
import time


import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--model_path", default=None, type=str, help="path for model")
args = parser.parse_args()

app = Flask(__name__)

# Defining Placeholder
inputs_raw = tf.placeholder(tf.float32, shape=[None, None, None, 4], name='inputs_image_tmap')
path_LR = tf.placeholder(tf.string, shape=[], name='path_LR')

# Setting default parameters
FLAGS = collections.namedtuple('FLAGS', '')

gen_output = generator(inputs_raw, 3, reuse=False, FLAGS=FLAGS)
print('Finish building the network')

with tf.name_scope('convert_image'):
    # Deprocess the images outputed from the model
    inputs = deprocessLR(inputs_raw)
    outputs = deprocess(gen_output)

    # Convert back to uint8
    converted_inputs = tf.image.convert_image_dtype(inputs, dtype=tf.uint8, saturate=True)
    converted_outputs = tf.image.convert_image_dtype(outputs, dtype=tf.uint8, saturate=True)

with tf.name_scope('encode_image'):
    save_fetch = {
        "path_LR": path_LR,
        "inputs": tf.map_fn(tf.image.encode_png, converted_inputs, dtype=tf.string, name='input_pngs'),
        "outputs": tf.map_fn(tf.image.encode_png, converted_outputs, dtype=tf.string, name='output_pngs')
    }

# Define the weight initiallizer (In inference time, we only need to restore the weight of the generator)
var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='generator')
weight_initiallizer = tf.train.Saver(var_list)

config = tf.ConfigProto()
config.gpu_options.allow_growth = True

sess = tf.Session(config=config)

# Load the pretrained model
print('Loading weights from the pre-trained model')
weight_initiallizer.restore(sess, args.model_path)

def handle_exit():
    print('\nAll files saved in ' + directory)
    generate_output()

atexit.register(handle_exit)
signal.signal(signal.SIGTERM, handle_exit)
signal.signal(signal.SIGINT, handle_exit)

if __name__ == '__main__':
    app.run(debug=True)

