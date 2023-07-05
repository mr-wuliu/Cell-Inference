from flask import (
    Blueprint, flash, g, redirect, render_template, request, session, url_for
)
from PIL import Image
import os
import random
import string
from mmdet.apis import init_detector, inference_detector
import mmcv
from mmdet.utils import register_all_modules

import time
import mmengine

# 指定权重与配置文件
config_file: str = 'flaskr/static/model/mask-rcnn_r101_fpn_1x_coco.py'
checkpoint_file: str = 'flaskr/static/model/mask-rcnn_r101_fpn_1x_coco.pth'

bp = Blueprint('display', __name__)
current_file = ''
# 文件上传 form方式
@bp.route('/upload', methods=['GET', 'POST'])
def upload():
    import base64
    if request.method == 'GET':
        return render_template('main/inference.html')
    elif request.method == 'POST':
        global current_file
        # 清空文件夹
        # import shutil
        # shutil.rmtree(os.path.join('flaskr','blueprint','img'))
        # os.mkdir(os.path.join('flaskr','blueprint','img'))
        if 'input-b6a[]' not in request.files:
            return "No file part"
        files = request.files.getlist('input-b6a[]')
        for file in files:
            print(type(file))
            # filename = file.filename
            filename =''.join(random.sample(string.ascii_letters + string.digits, 8)) + '.png'
            current_file = filename
            # numpy 格式
            result = inference(config_file,checkpoint_file,file,filename).get_image()
            result = result[..., ::-1]
            img_stream = Image.fromarray(result.astype('uint8')).convert('RGB')
            # img = base64.b64encode(img_stream).decode()
            ot_pt = 'flaskr/blueprint/img/'+filename
            img_stream.save(ot_pt)
            img = return_img_stream(ot_pt)
            img_row= return_img_stream('flaskr/blueprint/img/'+'row'+filename)

    return render_template('main/inference.html',
                           img_row = img_row,
                           img=img)

@bp.route('/getimg')
def getImg():
    # global current_file
    otpt = 'flaskr/blueprint/img/QmYbt56B.png'
    img = return_img_stream(otpt)
    img_row = return_img_stream((otpt))
    return {'img':img, 'img_row':img_row}


def inference(config_file, checkpoint_file, file, filename):
    # save img to path
    path = os.path.join('flaskr/blueprint/img','row'+filename)
    file.save(path)

    # read img
    img = mmcv.imread(path)

    register_all_modules()
    model = init_detector(config_file, checkpoint_file, device='cuda:0')
    from mmdet.registry import VISUALIZERS
    visualizer = VISUALIZERS.build(model.cfg.visualizer)
    visualizer.dataset_meta = model.dataset_meta

    result = inference_detector(model,img)
    visualizer.add_datasample(
        'result',
        img,
        data_sample=result,
        draw_gt=False,
        wait_time=0,
    )
    return visualizer

def return_img_stream(img_local_path):
    """
    工具函数:
    获取本地图片流
    :param img_local_path:文件单张图片的本地绝对路径
    :return: 图片流
    """
    import base64
    img_stream = ''
    with open(img_local_path, 'rb') as img_f:
        img_stream = img_f.read()
        img_stream = base64.b64encode(img_stream).decode()
    return img_stream



if __name__ == '__main__':
    config_file = './model/mask-rcnn_r101_fpn_1x_coco.py'
    checkpoint_file = './model/mask-rcnn_r101_fpn_1x_coco.pth'
    register_all_modules()
    model = init_detector(config_file, checkpoint_file, device='cuda:0')
    from mmdet.registry import VISUALIZERS

    # init the visualizer(execute this block only once)
    visualizer = VISUALIZERS.build(model.cfg.visualizer)

    visualizer.dataset_meta = model.dataset_meta
    img_path = './img/test.png'
    img = mmcv.imread(img_path)#, channel_order='rgb')

    result = inference_detector(model,img)
    # 进行新的可视化
    # print(result)
    visualizer.add_datasample(
        'result',
        img,
        data_sample=result,
        draw_gt=False,
        wait_time=0,
    )
    visualizer.show()
