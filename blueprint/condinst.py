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


def generate_lr_chart(json_list) -> Line:
    y_data = []
    x_data = []
    for d in json_list:
        if 'lr' in d:
            y_data.append(d['lr'])
            x_data.append(d['step'])
    line = (
        Line().add_xaxis(x_data)
        .add_yaxis('Learning Rate', y_data)
    )
    # change size
    line.width = '100%'
    return line


def generate_loss_chart(json_list) -> Line:
    x_data = []
    y_data = []
    y_cls_data = []
    y_bbox_data = []
    y_mask_data = []
    y_centerness_data = []
    for d in json_list:
        if 'loss' in d:
            y_data.append(d['loss'])
            y_cls_data.append(d['loss_cls'])
            y_bbox_data.append(d['loss_bbox'])
            y_mask_data.append(d['loss_mask'])
            y_centerness_data.append(d['loss_centerness'])
            x_data.append(d['step'])
    line = (
        Line().add_xaxis(x_data)
        .add_yaxis('loss', y_data)
        .add_yaxis('loss_cls', y_cls_data)
        .add_yaxis('loss_bbox', y_bbox_data)
        .add_yaxis('loss_mask', y_mask_data)
        .add_yaxis('loss_centerness', y_centerness_data)
    )
    # change size
    line.width = '100%'
    return line


def generate_bbox_map_chart(json_list) -> Line:
    x_data = ['10000', '20000', '30000', '40000', '50000', '60000', '70000', '80000', '90000']
    y_bbox_map_data = []
    y_bbox_map50_data = []
    y_bbox_map75_data = []
    y_bbox_map_small_data = []
    y_bbox_map_medium_data = []
    y_bbox_map_large_data = []
    for d in json_list:
        if 'coco/bbox_mAP' in d:
            y_bbox_map_data.append(d['coco/bbox_mAP'])
            y_bbox_map50_data.append(d['coco/bbox_mAP_50'])
            y_bbox_map75_data.append(d['coco/bbox_mAP_75'])
            y_bbox_map_small_data.append(d['coco/bbox_mAP_s'])
            y_bbox_map_medium_data.append(d['coco/bbox_mAP_m'])
            y_bbox_map_large_data.append(d['coco/bbox_mAP_l'])
    line = (
        Line().add_xaxis(x_data)
        .add_yaxis('mAP', y_bbox_map_data)
        .add_yaxis('mAP_50', y_bbox_map50_data)
        .add_yaxis('mAP_75', y_bbox_map75_data)
        .add_yaxis('mAP_s', y_bbox_map_small_data)
        .add_yaxis('mAP_m', y_bbox_map_medium_data)
        .add_yaxis('mAP_l', y_bbox_map_large_data)
    )
    # change size
    line.width = '100%'
    return line


def generate_seg_map_chart(json_list) -> Line:
    x_data = ['10000', '20000', '30000', '40000', '50000', '60000', '70000', '80000', '90000']
    y_seg_map_data = []
    y_seg_map50_data = []
    y_seg_map75_data = []
    y_seg_map_small_data = []
    y_seg_map_medium_data = []
    y_seg_map_large_data = []
    for d in json_list:
        if 'coco/segm_mAP' in d:
            y_seg_map_data.append(d['coco/segm_mAP'])
            y_seg_map50_data.append(d['coco/segm_mAP_50'])
            y_seg_map75_data.append(d['coco/segm_mAP_75'])
            y_seg_map_small_data.append(d['coco/segm_mAP_s'])
            y_seg_map_medium_data.append(d['coco/segm_mAP_m'])
            y_seg_map_large_data.append(d['coco/segm_mAP_l'])
    line = (
        Line().add_xaxis(x_data)
        .add_yaxis('mAP', y_seg_map_data)
        .add_yaxis('mAP_50', y_seg_map50_data)
        .add_yaxis('mAP_75', y_seg_map75_data)
        .add_yaxis('mAP_s', y_seg_map_small_data)
        .add_yaxis('mAP_m', y_seg_map_medium_data)
        .add_yaxis('mAP_l', y_seg_map_large_data)
    )
    # change size
    line.width = '100%'
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
    loss = generate_loss_chart(json_list)
    loss_plot = Markup(loss.render_embed())
    lr = generate_lr_chart(json_list)
    lr_plot = Markup(lr.render_embed())
    bbox_map = generate_bbox_map_chart(json_list)
    bbox_map_plot = Markup(bbox_map.render_embed())
    seg_map = generate_seg_map_chart(json_list)
    seg_map_plot = Markup(seg_map.render_embed())
    return render_template('condinst/result.html', losses=loss_plot, lr=lr_plot, bbox_map=bbox_map_plot,
                           seg_map=seg_map_plot, model=model)


"""
接口请求
"""
# @bp.route('/lr')
# def lr():
#     chart = line.dump_options()
#     return render_template('flaskr/templates/condinst/lr.html', chart_options=chart)
