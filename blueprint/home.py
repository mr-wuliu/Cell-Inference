from flask import (
    Blueprint, flash, g, redirect, render_template, request, session, url_for
)

bp = Blueprint('home', __name__)

# 定义单个模型的路由, 目前主页面默认调用Mask R-CNN的模型
class model:
    home = 'ms_rcnn.home'
    inference = 'ms_rcnn.inference'
    training = 'ms_rcnn.training'
    result = 'ms_rcnn.result'
    pr_page = 'ms_rcnn.pr_page'

@bp.route('/')
def home():
    return render_template('common/base.html', model=model)
