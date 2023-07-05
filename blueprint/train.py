from flask import (
    Blueprint, flash, g, redirect, render_template, request, session, url_for
)

bp = Blueprint('chat', __name__)


@bp.route('/train/home')
def home():
    return 'training data!'


@bp.route('/index',methods=['GET'])
def index():
    user='1'
    return render_template('index.html',user=user)
