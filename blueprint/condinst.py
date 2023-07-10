import json
import os
import subprocess

import mmcv
import numpy as np
import torch
from flask import (
    Blueprint, flash, g, redirect, render_template, request, session, url_for
)

import pyecharts.options as opts
from jinja2 import Markup
from matplotlib import pyplot as plt
from mmcv import Compose, ToTensor, Normalize
from mmdet.apis import init_detector
from mmdet.models import CondInst
from mmengine.visualization import Visualizer
from pyecharts.charts import Line
from pyecharts.faker import Faker

import flaskr.utils as utils

# 注册蓝图
bp = Blueprint('condinst', __name__)
# 创建子进程列表，用于保存正在执行的子进程对象
processes = {}

# 定义单个模型的路由
class model:
    home = 'condinst.home'
    inference = 'condinst.inference'
    training = 'condinst.training'
    result = 'condinst.result'


# 模型配置文件
config_file: str = 'flaskr/static/model/condinst_r101_fpn_ms-poly-90k_coco_instance.py'
checkpoint_file: str = 'flaskr/static/model/condinst_r101_fpn_ms-poly-90k_coco_instance.90000.pth'
# 缓存
cache_path = 'flaskr/cache/'

# pyecharts配置
REMOTE_HOST = "https://pyecharts.github.io/assets/js"
network_model = init_detector(config_file, checkpoint_file, device='cpu')
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


def _forward(x):
    conv1 = network_model.backbone.conv1(x)
    bn1 = network_model.backbone.bn1(conv1)

    layer1 = network_model.backbone.layer1(bn1)
    layer2 = network_model.backbone.layer2(layer1)
    layer3 = network_model.backbone.layer3(layer2)
    layer4 = network_model.backbone.layer4(layer3)

    return conv1, bn1, layer1, layer2, layer3, layer4


def preprocess_img(img,
                   mean=[0.485, 0.456, 0.406],
                   std=[0.229, 0.224, 0.225]):
    image_norm = np.float32(img) / 255
    preprocessing = Compose([
        ToTensor(),
        Normalize(mean=mean, std=std)
    ])
    tensor = preprocessing(image_norm.copy()).unsqueeze(0)
    return tensor


def draw_feature():
    md = network_model
    md.forward = _forward
    visualizer = Visualizer()
    img = "cache/urlmCKGR9WyV.png"
    img = mmcv.imread(os.path.join(img),
                      channel_order='rgb')
    input_tensor = preprocess_img(img)
    features = md(img)
    feat = md(input_tensor)[0]
    input_data = visualizer.draw_featmap(feat, channel_reduction='select_max')
    feature_map = features.squeeze().detach().numpy()  # 转换特征图为可处理的数组
    plt.imshow(feature_map, cmap='hot')
    plt.colorbar()
    plt.title('Feature Map')
    plt.show()
"""
页面展示
"""


@bp.route('/')
def home():
    return render_template('condinst/condinst_base.html', model=model)


@bp.route('/inference')
def inference():
    return render_template('condinst/inference.html', model=model)


@bp.route('/training', methods=['GET', 'POST'])
def training():
    if request.method == 'GET':
        # 模型启动
        return render_template('mask_rcnn/training.html', model=model)
    elif request.method == 'POST':
        key = utils.create_key()
        arguments = {}
        # for arg in request.form.to_dict():
        # 获取上传参数
        """
        num_classes
        lr
        num_workers
        batch_size
        """
        for i in request.form:
            arg_key = i
            arg_val = request.form[i]
            if arg_key == 'num_classes':
                if arg_val == None or arg_val == '':
                    arg_val = 5
            elif arg_key == 'lr':
                if arg_val == None or arg_val == '':
                    arg_val = 0.02
            elif arg_key == 'num_workers':
                if arg_val == None or arg_val == '':
                    arg_val = 2
            elif arg_key == 'batch_size':
                if arg_val == None or arg_val == '':
                    arg_val = 4
            else:
                continue
            arguments[arg_key] = arg_val

        # 读取 config.f 文件
        with open('flaskr/static/model/base/condinst_r101_fpn_ms-poly-90k_coco_instance.f', 'r') as file:
            lines = file.readlines()
        file.close()

        # 修改变量
        for i in range(len(lines)):
            if lines[i].startswith('set_num_classes='):
                lines[i] = 'set_num_classes=' + str(arguments['num_classes']) + '\n'  # 将变量修改为新的值
            if lines[i].startswith('set_lr='):
                lines[i] = 'set_lr=' + str(arguments['lr']) + '\n'
            if lines[i].startswith('set_num_workers='):
                lines[i] = 'set_num_workers=' + str(arguments['num_workers']) + '\n'
            if lines[i].startswith('set_batch_size='):
                lines[i] = 'set_batch_size=' + str(arguments['batch_size']) + '\n'
            if lines[i].startswith('# end var'):
                lines[i] = ''
                break
        # 写入新的 Python 文件
        config = 'flaskr/static/model/config/train_' + key + '.py'
        output = os.path.join(cache_path, 'work_dir_' + key)

        with open(config, 'w') as file:
            file.writelines(lines)

        # 执行脚本
        script = 'mmdetection/tools/train.py'
        args = [config,
                '--work-dir', output]
        # ' > ' + cache_path+key+'/log.txt']
        # 创建文件夹

        if not os.path.exists(output):
            os.mkdir(output)

        # 重定向输出
        log = open(os.path.join(output, 'log.txt'), 'a')
        log.write('begin to run.\n')
        process = subprocess.Popen(['python', script] + args,
                                   stdout=log,
                                   stderr=log)
        # stdout, stderr = process.communicate()
        # res = stdout.decode('utf-8')
        # output, error = process.communicate()
        processes[key] = process
        return render_template('mask_rcnn/training_processing.html', model=model, key=key)


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
@bp.route('/data_handle/res', methods=['GET', 'POST'])
def img_inference_res():
    if request.method == 'POST':
        key = request.json['key']
        print(key)
        img_stream = utils.return_img_stream(
            os.path.join(cache_path, 'inf_' + key + '.png'))

        return img_stream
    return 'error'


@bp.route('/stop_train', methods=['GET', 'POST'])
def stop_train():
    if request.method == 'POST':
        key = request.json['key']
        process = processes[key]
        process.terminate()
        return {'apply': '训练终止!'}


@bp.route('/log/<key>', methods=['GET', 'POST'])
def get_log(key):
    if request.method == 'GET':
        import re
        dir = os.path.join(cache_path, 'work_dir_' + key, 'log.txt')
        with open(dir, 'r') as file:
            log_lines = file.readlines()
        log_content = ''
        for line in log_lines:
            line = line.strip()  # 去除行首尾的空白字符
            line = f'<div>{line}</div>'  # 添加 <div> 标签
            log_content += line
        # log_content = re.sub(r'\n', '<br>', log_content)
        file.close()
        return log_content