#!/usr/bin/env python
# -*- coding:utf-8 -*-
from __future__ import print_function
import sys

sys.path.insert(0, 'src')
import transform, numpy as np, os
import tensorflow as tf


BATCH_SIZE = 1
DEVICE = '/gpu:0'

from urllib import request as ul_request
from flask import Flask
from flask import request
from flask import Response
from io import BytesIO
from flask import render_template
from PIL import Image
import cv2


app = Flask(__name__)

class Faster_style(object):
    def __init__(self):
        self._init_model()

    def _init_model(self):
        """
        :param checkpoint_dir:
        :return:
        """
        checkpoint_dir = './checkpoint'
        soft_config = tf.ConfigProto(allow_soft_placement=True)
        soft_config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=soft_config)
        self.img_placeholder = tf.placeholder(tf.float32, shape=[1, 500, 500, 3])

        self.preds = transform.net(self.img_placeholder)
        saver = tf.train.Saver()
        if os.path.isdir(checkpoint_dir):
            ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(self.sess, ckpt.model_checkpoint_path)
            else:
                raise Exception("No checkpoint found...")
        else:
            saver.restore(self.sess, checkpoint_dir)

    def ffwd(self):
        """

        :param image:
        :return: _preds
        """
        X = np.zeros((1, 500, 500, 3), dtype=np.float32)
        assert X[0].shape == self.image.shape, 'Images have different dimensions'
        X[0] = self.image
        _preds = self.sess.run(self.preds, feed_dict={self.img_placeholder: X})

        return _preds

    def faster_style(self, url):

        req = ul_request.urlopen(url)
        arr = np.asarray(bytearray(req.read()), dtype=np.uint8)
        self.image = cv2.imdecode(arr, -1)  # 'load it as it is'
        self.shape = self.image.shape
        self.image = cv2.resize(
            self.image,
            (500, 500),
            interpolation=cv2.INTER_CUBIC
        )
        self.image = np.resize(self.image, (500, 500, 3))
        self.image = self.ffwd()
        return self.image[0], self.shape


@app.before_first_request
def init():
    global styler
    styler = Faster_style()


@app.route("/faster_style", methods=['GET', 'POST'])
def server():
    """
    :return:
    """
    params = request.args
    if 'url' in params:
        import base64
        url = params['url']
        image, shape = styler.faster_style(url)
        image = cv2.resize(
            image,
            (shape[1], shape[0]),
            interpolation=cv2.INTER_CUBIC
        )
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.convertScaleAbs(image)
        # save_img('./222.jpg', image)

        img = Image.fromarray(image)
        # img.save('./333.jpg', 'JPEG')
        out = BytesIO()
        img.save(out, 'JPEG')
        out = base64.b64encode(out.getvalue()).decode('ascii')
        return render_template('index.html',
                               img_stream=out)
    else:
        return 'error params!'


if __name__ == '__main__':
    app.run(port=8003, debug=True, use_reloader=False)