from flask import (
    Blueprint, flash, g, redirect, render_template, request, session, url_for
)

bp = Blueprint('home', __name__)

# 定义单个模型的路由, 目前主页面默认调用Mask R-CNN的模型
class model:
    home = 'home.home'
    inference = 'mask_rcnn.inference'
    training = 'mask_rcnn.training'
    result = 'mask_rcnn.result'
    pr_page = 'mask_rcnn.pr_page'
    matrix = 'mask_rcnn.matrix'

@bp.route('/')
def home():
    return render_template('common/base.html', model=model)
