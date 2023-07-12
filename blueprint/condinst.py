import json
import os
import subprocess
from flask import (
    Blueprint, flash, g, redirect, render_template, request, session, url_for, jsonify
)

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
    pr_page = 'condinst.pr_page'


# Draw 继承
class Draw(utils.Draw):
    pass


# 模型配置文件
config_file: str = 'flaskr/static/model/condinst_r101_fpn_ms-poly-90k_coco_instance.py'
checkpoint_file: str = 'flaskr/static/model/condinst_r101_fpn_ms-poly-90k_coco_instance.pth'
# 缓存
cache_path = 'flaskr/cache/'

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
        return render_template('condinst/training_processing.html', model=model, key=key)


@bp.route('/result')
def result():
    model_name = 'condinst'
    # 绘制各式图表

    # 遍历文件夹, 搜索日期最新的文件
    path = 'flaskr/static/model/sample/' + model_name

    # 遍历文件夹, 搜索日期最新的文件
    folders = [folder for folder in os.listdir(path) if folder.startswith(tuple(str(i) for i in range(10)))]
    latest_file = max(folders) if folders else ''
    if not latest_file:
        return
    for folder in os.listdir(path):
        if folder.startswith(tuple(str(i) for i in range(10))):
            if latest_file < folder:
                latest_file = folder
    path = os.path.join(path, latest_file, 'vis_data/scalars.json')

    json_list = Draw.get_data(path)
    loss = Draw.generate_loss_chart(json_list)
    loss_plot = Draw.Markup(loss.render_embed())
    lr = Draw.generate_lr_chart(json_list)
    lr_plot = Draw.Markup(lr.render_embed())
    bbox_map = Draw.generate_bbox_map_chart(json_list)
    bbox_map_plot = Draw.Markup(bbox_map.render_embed())
    seg_map = Draw.generate_seg_map_chart(json_list)
    seg_map_plot = Draw.Markup(seg_map.render_embed())

    # 特征图展示
    img_list = []
    img_path = 'img/features/' + model_name
    num_img = 0
    for img_f in os.listdir('flaskr/static/' + img_path):
        if img_f.startswith('combine'):
            elm = (str(num_img), img_path + '/' + img_f)
            num_img += 1
            img_list.append(elm)

    # t_sne 展示
    class t_sne:
        path = 'img/t_sne/CondInst R101 t-sne.png'

    return render_template('condinst/result.html',
                           losses=loss_plot,
                           lr=lr_plot, bbox_map=bbox_map_plot,
                           seg_map=seg_map_plot, img_list=img_list,
                           t_sne=t_sne, loss_title='Condinst Loss',
                           model=model)


@bp.route('/pr_page')
def pr_page(page=1):
    page = 1
    page = str(page)
    img_path: str = "img/pr_perclass"
    bbox_img_list = []
    segm_img_list = []
    # 获取当前脚本的文件名
    script_name = model.home
    if script_name.startswith("cascade"):
        script_name = "cascade"
    model_name = script_name.split('.')[0]
    img_path = img_path + '/' + model_name
    print(img_path)
    bbox_img_num = 0
    segm_img_num = 0
    dir_path = 'flaskr/static/' + img_path + '/coco_error_analysis'
    img_path += '/coco_error_analysis'
    for path in os.listdir(dir_path):
        # print(path)
        if path == 'bbox':
            for bbox_img in os.listdir(dir_path + '/' + path):
                cat = bbox_img.split('/')[-1].split('-')[1]
                if cat == page or (cat == 'allclass' and page == '6'):
                    bbox_img_num += 1
                    bbox_img_dir = (str(bbox_img_num), img_path + '/' + path + '/' + bbox_img)
                    print(bbox_img_dir)
                    bbox_img_list.append(bbox_img_dir)
        elif path == 'segm':
            for segm_img in os.listdir(dir_path + '/' + path):
                cat = segm_img.split('/')[-1].split('-')[1]
                if cat == page or (cat == 'allclass' and page == '6'):
                    segm_img_num += 1
                    segm_img_dir = (str(segm_img_num), img_path + '/' + path + '/' + segm_img)
                    print(segm_img_dir)
                    segm_img_list.append(segm_img_dir)

    img_list = {"bbox_img_list": bbox_img_list, "segm_img_list": segm_img_list}
    img_id_list = {"img_segm": "img_segm", "img_bbox": "img_bbox"}

    return render_template(model_name + '/pr_page.html',
                           img_id_list=img_id_list,
                           img_list=img_list,
                           model=model)


"""
接口请求
"""


@bp.route('/pr_page_update', methods=['GET', 'POST'])
def pr_page_update():
    page = request.args.get('page')
    page = str(page)
    img_path: str = "img/pr_perclass"
    bbox_img_list = []
    segm_img_list = []
    # 获取当前脚本的文件名
    script_name = model.home
    if script_name.startswith("cascade"):
        script_name = "cascade"
    model_name = script_name.split('.')[0]
    img_path = img_path + '/' + model_name
    print(img_path)
    bbox_img_num = 0
    segm_img_num = 0
    dir_path = 'flaskr/static/' + img_path + '/coco_error_analysis'
    img_path += '/coco_error_analysis'
    for path in os.listdir(dir_path):
        # print(path)
        if path == 'bbox':
            for bbox_img in os.listdir(dir_path + '/' + path):
                cat = bbox_img.split('/')[-1].split('-')[1]
                if cat == page or (cat == 'allclass' and page == '6'):
                    bbox_img_num += 1
                    bbox_img_dir = (str(bbox_img_num), img_path + '/' + path + '/' + bbox_img)
                    print(bbox_img_dir)
                    bbox_img_list.append(bbox_img_dir)
        elif path == 'segm':
            for segm_img in os.listdir(dir_path + '/' + path):
                cat = segm_img.split('/')[-1].split('-')[1]
                if cat == page or (cat == 'allclass' and page == '6'):
                    segm_img_num += 1
                    segm_img_dir = (str(segm_img_num), img_path + '/' + path + '/' + segm_img)
                    print(segm_img_dir)
                    segm_img_list.append(segm_img_dir)

    img_list = {"bbox_img_list": bbox_img_list, "segm_img_list": segm_img_list}

    return img_list


@bp.route('/data_handle', methods=['GET', 'POST'])
def img_inference():
    # post 上传图片, get下载图片
    res = {'apply': '', 'key': ''}
    if request.method == 'POST':
        import base64
        if request.form.get('image').endswith('.png'):
            return {'apply': '请选择图片'}

        img_stream: str = request.form.get('image').split(',')[1]
        img = base64.b64decode(img_stream)
        # 图片唯一标识
        key = utils.create_key()
        with open(os.path.join(cache_path, key + '.png'), 'wb') as img_decode:
            img_decode.write(img)
        img_decode.close()

        # 对图片进行推理
        utils.inference(config_file, checkpoint_file, key)
        res['apply'] = 'pass'
        res['key'] = key

        return {'apply': 'pass', 'key': key}
    elif request.method == 'GET':
        res['apply'] = 'ERROR'
        res['key'] = ''
        return {'apply': 'ERROR', 'key': ''}
    else:
        return {'apply': 'error', 'key': ''}


@bp.route('/data_handle/res', methods=['GET', 'POST'])
def img_inference_res():
    if request.method == 'POST':
        key = request.json['key']
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
@bp.route('/metrics/<key>', methods=['GET', 'POST'])
def get_metrics(key):
    if request.method == 'GET':
        # 绘制各式图
        path = 'flaskr/cache/work_dir_'+key
        # 遍历文件夹, 搜索日期最新的文件
        folders = [folder for folder in os.listdir(path) if folder.startswith(tuple(str(i) for i in range(10)))]
        latest_file = max(folders) if folders else ''
        if not latest_file:
            return
        for folder in os.listdir(path):
            if folder.startswith(tuple(str(i) for i in range(10))):
                if latest_file < folder:
                    latest_file = folder
        path = os.path.join(path, latest_file, 'vis_data/scalars.json')

        json_list = Draw.get_data(path)
        loss = Draw.generate_loss_chart(json_list)
        loss_plot = Draw.Markup(loss.render_embed())
        lr = Draw.generate_lr_chart(json_list)
        lr_plot = Draw.Markup(lr.render_embed())
        plot = {
            'loss_plot': loss_plot,
            'lr_plot': lr_plot
        }
        return plot
