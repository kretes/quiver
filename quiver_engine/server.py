import json
import re
from contextlib import contextmanager

from os import listdir
from os.path import abspath, relpath, dirname, join
import webbrowser

import numpy as np
import keras

from flask import Flask, send_from_directory
from flask.json import jsonify
from flask_cors import CORS

from gevent.wsgi import WSGIServer

from scipy.misc import imsave

from imagenet_utils import decode_predictions
from util import deprocess_image, load_img, load_img_scaled, get_json
from layer_result_generators import get_outputs_generator
from occlusion import occlude_and_predict


def get_app(model, temp_folder='./tmp', input_folder='./'):
    get_evaluation_context = get_evaluation_context_getter()
    single_input_shape = model.get_input_shape_at(0)[1:3]

    app = Flask(__name__)
    app.threaded = True
    CORS(app)

    @app.route('/')
    def home():
        return send_from_directory(
            join(
                dirname(abspath(__file__)),
                'quiverboard/dist'
            ),
            'index.html'
        )

    @app.route('/<path>')
    def get_board_files(path):
        return send_from_directory(join(
            dirname(abspath(__file__)),
            'quiverboard/dist'
        ), path)

    @app.route('/inputs')
    def get_inputs():
        image_regex = re.compile(r".*\.(jpg|png|gif)$")

        return jsonify([
            filename for filename in listdir(input_folder)
            if image_regex.match(filename) != None
        ])

    @app.route('/temp-file/<path>')
    def get_temp_file(path):
        return send_from_directory(abspath(temp_folder), path)

    @app.route('/input-file/<path>')
    def get_input_file(path):
        return send_from_directory(abspath(input_folder), path)

    @app.route('/model')
    def get_config():
        return jsonify(json.loads(model.to_json()))

    @app.route('/occlusion/<input_path>/<no_of_occlusions>/<overlap>')
    def get_occlusion(input_path,no_of_occlusions,overlap):
        input_img = load_img_scaled(input_path, single_input_shape)
        filename = get_output_name(temp_folder, 'occlusion', input_path, "%s_%s" % (no_of_occlusions,overlap))

        with get_evaluation_context():
            occluded_mask = occlude_and_predict(model,input_img,int(no_of_occlusions),float(overlap))
            imsave(filename,occluded_mask)

        return jsonify(filename)

    @app.route('/layer/<layer_name>/<input_path>')
    def get_layer_outputs(layer_name, input_path):

        input_img = load_img(input_path, single_input_shape)
        output_generator = get_outputs_generator(model, layer_name)

        with get_evaluation_context():

            layer_outputs = output_generator(input_img)[0]
            output_files = []

            for z in range(0, layer_outputs.shape[2]):

                img = layer_outputs[:, :, z]
                deprocessed = deprocess_image(img)
                filename = get_output_name(temp_folder, layer_name, input_path, z)
                output_files.append(
                    relpath(
                        filename,
                        abspath(temp_folder)
                    )
                )
                imsave(filename, deprocessed)

        return jsonify(output_files)
    @app.route('/predict/<input_path>')
    def get_prediction(input_path):
        input_img = load_img(input_path, single_input_shape)
        with get_evaluation_context():
            return jsonify(
                json.loads(
                    get_json(
                        decode_predictions(
                            model.predict(input_img)
                        )
                    )
                )
            )


    return app

def run_app(app, port=5000):
    http_server = WSGIServer(('', port), app)
    webbrowser.open_new('http://localhost:' + str(port))
    http_server.serve_forever()

def launch(model, temp_folder='./tmp', input_folder='./', port=5000):
    return run_app(
        get_app(model, temp_folder, input_folder),
        port
    )

def get_output_name(temp_folder, layer_name, input_path, z_idx):
    return temp_folder + '/' + layer_name + '_' + str(z_idx) + '_' + input_path + '.png'

def get_evaluation_context_getter():
    if keras.backend.backend() == 'tensorflow':
        import tensorflow as tf
        return tf.get_default_graph().as_default

    if keras.backend.backend() == 'theano':
        return contextmanager(lambda: (yield))
