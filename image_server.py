from flask import Flask, request, Response
# import jsonpickle
import numpy as np
import cv2
import network
import json
from flask_cors import CORS
from flask import jsonify

from loguru import logger

# Initialize the Flask application
app = Flask(__name__)
CORS(app)

model = network.load_model()

@app.route('/api/', methods=['POST'])
def process_video():
    path = (request.args.get('path'))

    logger.success(path)

    response = network.proccess_video(model, path)

    # Encode response to str
    response = ';'.join([f'{k}:{v}' for k, v in response.items()])
    logger.success(response)

    return response


# start flask app
app.run(host="0.0.0.0", port=5000)
