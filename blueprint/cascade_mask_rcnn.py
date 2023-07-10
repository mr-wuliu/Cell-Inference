import mmcv
from flask import (
    Blueprint, flash, g, redirect, render_template, request, session, url_for
)
import flaskr.utils as utils
import os
import subprocess

from mmdet.apis import init_detector

# 注册蓝图
bp = Blueprint('cascade_mask_rcnn', __name__)

# 创建子进程列表，用于保存正在执行的子进程对象
processes = {}


# 定义单个模型的路由
class model:
    home = 'cascade_mask_rcnn.home'
    inference = 'cascade_mask_rcnn.inference'
    training = 'cascade_mask_rcnn.training'
    result = 'cascade_mask_rcnn.result'
    pr_page = 'cascade_mask_rcnn.pr_page'


# 模型配置文件
config_file: str = 'flaskr/static/model/cascade-mask-rcnn_x101-64x4d_fpn_1x_coco.py'
checkpoint_file: str = 'flaskr/static/model/epoch_12.pth'
# 缓存
cache_path = 'flaskr/cache/'

# pyecharts配置
REMOTE_HOST = "https://pyecharts.github.io/assets/js"
network_model = init_detector(config_file, checkpoint_file, device='cpu')
"""
页面展示
"""


@bp.route('/')
def home():
    return render_template('cascade_mask_rcnn/cascade_mask_rcnn_base.html', model=model)


@bp.route('/inference')
def inference():
    return render_template('cascade_mask_rcnn/inference.html', model=model)


@bp.route('/pr_page')
def pr_page():
    return render_template('cascade_mask_rcnn/pr_page.html', model=model)


@bp.route('/training', methods=['GET', 'POST'])
def training():
    if request.method == 'GET':
        # 模型启动
        return render_template('cascade_mask_rcnn/training.html', model=model)
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
        with open('flaskr/static/model/base/mask-rcnn_r101_fpn_ms-poly-3x_coco.f', 'r') as file:
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
        return render_template('cascade_mask_rcnn/training_processing.html', model=model, key=key)


@bp.route('/result')
def result():
    # 加载配置
    if request.method == 'GET':
        return render_template('cascade_mask_rcnn/result.html', model=model)


"""
接口请求
"""


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
        with open(cache_path + key + '.png', 'wb') as img_decode:
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
        # img = utils.return_img_stream(os.path.join(cache_path,'res_'+key))
    else:
        return {'apply': 'error', 'key': ''}


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
