import os

from flask import Flask, render_template
from flaskr.blueprint import train, home, display, mask_rcnn


def create_app():
    # create and configure the app
    app = Flask(__name__)
    app.config.from_pyfile('config.py')
    """
    註冊服務
    """
    register_extensions(app)
    register_blueprint(app)
    return app


def register_extensions(app: Flask):
    """
    加载扩展
    :param app:
    :return:
    """


def register_blueprint(app: Flask):
    """
    加载蓝图
    :param app:
    :return:
    """
    app.register_blueprint(train.bp)
    app.register_blueprint(home.bp)
    app.register_blueprint(display.bp)
    app.register_blueprint(mask_rcnn.bp, url_prefix='/maskrcnn')
    app.add_url_rule('/', endpoint='home')


