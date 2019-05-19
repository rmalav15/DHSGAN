from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import atexit
import collections
import cv2
import os
import signal
import time

import flask
import numpy as np
import scipy.misc as sic

from werkzeug.utils import secure_filename
from cap.calDepthMap import get_tmap
from lib.model import generator
from lib.ops import *

parser = argparse.ArgumentParser()
parser.add_argument("--model_path",
                    default='/mnt/069A453E9A452B8D/Ram/KAIST/SRGAN_data/'
                            'experiment_clean_reside_pred_g20_SRGAN/model-170000',
                    help="path for model")
parser.add_argument("--output_dir", default='./static/images', help="output folder")
args = parser.parse_args()

app = flask.Flask(__name__)

# Check the output directory to save the checkpoint
if not os.path.exists(args.output_dir):
    os.mkdir(args.output_dir)

# Defining Placeholder
inputs_raw = tf.placeholder(tf.float32, shape=[None, None, None, 4], name='inputs_raw')
path_LR = tf.placeholder(tf.string, shape=[], name='path_LR')

# Setting default parameters
_FLAGS = collections.namedtuple('_FLAGS', 'num_resblock, is_training')
FLAGS = _FLAGS(
    num_resblock=16,
    is_training=False
)

with tf.variable_scope('generator'):
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
        "inputs": converted_inputs,
        "outputs": converted_outputs
    }

# Define the weight initiallizer (In inference time, we only need to restore the weight of the generator)
var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='generator')
weight_initializer = tf.train.Saver(var_list)

# Define the initialization operation
init_op = tf.global_variables_initializer()

config = tf.ConfigProto()
config.gpu_options.allow_growth = True

sess = tf.Session(config=config)

# Load the pretrained model
print('Loading weights from the pre-trained model')
weight_initializer.restore(sess, args.model_path)


def handle_exit():
    print('Exiting. Closing TF session !!!!!!!!!!!!!!!!!!!!')
    sess.close()


# TODO: Remove Scipy Dep
def read_image_as_float(image_path):
    im = sic.imread(image_path).astype(np.float32)
    assert im.shape[2] == 3  # Throw error if GrayScale
    im = im / 255.0
    return im


@app.route('/')
def index():
    return flask.render_template("index.html")


@app.route('/error')
def image_not_found():
    return None


# TODO: Take multiple images
@app.route('/dehaze', methods=['POST'])
def dehaze():
    start = time.time()
    image = flask.request.files['image']
    beta = float(flask.request.form['beta'])
    image_name = None
    image_path = None

    if image:
        image_name = secure_filename(image.filename)
        image_path = os.path.join(args.output_dir, image_name)
        image.save(image_path)
    else:
        return flask.redirect(flask.url_for('image_not_found'))

    dehazed_path = os.path.join(args.output_dir, 'dehazed_' + image_name)
    tmap_path = os.path.join(args.output_dir, 'tmap_' + image_name)
    print("image_path:", image_path)
    print("beta:", beta)

    im = read_image_as_float(image_path=image_path)
    tmap = get_tmap(im, beta=beta)
    input_im = np.concatenate((im, np.expand_dims(tmap, axis=2)), axis=2)

    results = sess.run(save_fetch, feed_dict={inputs_raw: np.expand_dims(input_im, axis=0), path_LR: image_path})
    cv2.imwrite(tmap_path, (tmap*255).astype(np.uint8))
    cv2.imwrite(dehazed_path, cv2.cvtColor(results['outputs'][0], cv2.COLOR_RGB2BGR))

    return flask.jsonify({
        "real": image_path,
        "dehazed": dehazed_path,
        "tmap": tmap_path,
        "time": time.time() - start
    })


atexit.register(handle_exit)
signal.signal(signal.SIGTERM, handle_exit)
signal.signal(signal.SIGINT, handle_exit)

if __name__ == '__main__':
    app.run(debug=True)
