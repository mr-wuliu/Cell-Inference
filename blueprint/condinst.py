import json

import mmcv
from flask import (
    Blueprint, flash, g, redirect, render_template, request, session, url_for
)
import pyecharts.options as opts
from jinja2 import Markup
from pyecharts.charts import Line
from pyecharts.faker import Faker

# 注册蓝图
bp = Blueprint('condinst', __name__)


# 定义单个模型的路由
class model:
    home = 'condinst.home'
    inference = 'condinst.inference'
    training = 'condinst.training'
    result = 'condinst.result'


# 模型配置文件
config_file: str = 'flaskr/static/model/condinst_r101_fpn_1x_coco.py'
checkpoint_file: str = 'flaskr/static/model/condinst_r101_fpn_ms-poly-90k_coco_instance.90000.pth'
# 缓存
cache_path = 'flaskr/cache/'

# pyecharts配置
REMOTE_HOST = "https://pyecharts.github.io/assets/js"

"""
定义画图方法
"""


def get_data() -> list:
    path = 'mmdetection/condinst_r101/20230703_104440/vis_data/scalars.json'
    json_list = []
    with open(path, 'r') as f:
        for line in f:
            try:
                json_data = json.loads(line)
                # 处理每个JSON对象
                # 访问和操作json_data，表示当前行的JSON对象
                json_list.append(json_data)
            except json.decoder.JSONDecodeError:
                # 解析失败，处理下一行
                continue
    return json_list


def generate_line_chart(json_list, key, graph_title, line_title) -> Line:
    y_data = []
    x_data = []
    # 从list中取出学习率作为y轴
    for d in json_list:
        if key in d:
            y_data.append(d[key])
            x_data.append(d['step'])
    line = (
        Line()
        .add_xaxis(x_data)
        .add_yaxis(line_title, y_data)
        .set_global_opts(title_opts=opts.TitleOpts(title=graph_title))
    )
    return line


"""
页面展示
"""


@bp.route('/')
def home():
    return render_template('condinst/condinst_base.html', model=model)


@bp.route('/inference')
def inference():
    return render_template('condinst/inference.html', model=model)


@bp.route('/training')
def training():
    return render_template('condinst/result.html', model=model)


@bp.route('/result')
def result():
    json_list = get_data()
    line_chart = generate_line_chart(json_list)
    plot = Markup(line_chart.render_embed())
    return render_template('condinst/result.html', line_chart=plot, model=model)


"""
接口请求
"""
# @bp.route('/lr')
# def lr():
#     chart = line.dump_options()
#     return render_template('flaskr/templates/condinst/lr.html', chart_options=chart)
