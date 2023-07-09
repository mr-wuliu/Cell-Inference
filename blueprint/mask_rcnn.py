import mmcv
from flask import (
    Blueprint, flash, g, redirect, render_template, request, session, url_for
)
import flaskr.utils as utils
import os
import subprocess


# 注册蓝图
bp = Blueprint('mask_rcnn', __name__)

# 创建子进程列表，用于保存正在执行的子进程对象
processes = []

# 定义单个模型的路由
class model:
    home = 'mask_rcnn.home'
    inference = 'mask_rcnn.inference'
    training = 'mask_rcnn.training'
    result = 'mask_rcnn.result'
# 模型配置文件
config_file: str = 'flaskr/static/model/mask-rcnn_r101_fpn_1x_coco.py'
checkpoint_file: str = 'flaskr/static/model/mask-rcnn_r101_fpn_1x_coco.pth'
# 缓存
cache_path = 'flaskr/cache/'

"""
页面展示
"""


@bp.route('/')
def home():
    return render_template('mask_rcnn/mask_rcnn_base.html', model=model)


@bp.route('/inference')
def inference():
    return render_template('mask_rcnn/inference.html', model=model)


@bp.route('/training')
def training():
    # 模型启停

    return 'training'

@bp.route('/result')
def result():
    # 加载配置
    if request.method == 'GET':
        num_class = 5
    
    return render_template('mask_rcnn/result.html', model=model)



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

        #对图片进行推理
        utils.inference(config_file,checkpoint_file,key)
        res['apply'] = 'pass'
        res['key'] = key

        return {'apply': 'pass', 'key':key}
    elif request.method == 'GET':
        res['apply'] = 'ERROR'
        res['key'] = ''
        return {'apply': 'ERROR', 'key':''}
        # img = utils.return_img_stream(os.path.join(cache_path,'res_'+key))
    else:
        return {'apply': 'error', 'key': ''}


@bp.route('/data_handle/res', methods=['GET', 'POST'])
def img_inference_res():
    if request.method == 'POST':
        key = request.json['key']
        print(key)
        img_stream = utils.return_img_stream(
            os.path.join(cache_path,'inf_'+key+'.png'))

        return img_stream
    return 'error'

@bp.route('/start_train',methods=['GET','POST'])
def train_start():
    if request.method == 'POST':
        """
        python .\tools\train.py .\configs\mask_rcnn\mask-rcnn_r101_fpn_ms-poly-3x_coco.py --work-dir mask_rcnn_3x_train
        """

        model_key = utils.create_key()
        script = 'mmdetection/tools/train.py'

        # script = '/flaskr/static/model/mask-rcnn_r101_fpn_ms-poly-3x_coco.py'
        args = ['flaskr/static/model/mask-rcnn_r101_fpn_ms-poly-3x_coco.py',
                '--word-dir', cache_path + '/work_dir_'+model_key]
        process = subprocess.Popen(['python', script] + args )
        processes.append(process)
        return model_key

@bp.route('/stop_train',methods=['GET','POST'])
def train_end():
    pass