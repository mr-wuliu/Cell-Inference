import random
import string
import os
import json
import mmcv
from mmdet.utils import register_all_modules
from mmdet.apis import init_detector, inference_detector
from pyecharts.charts import Line
from jinja2 import Markup as Mk

def create_key():
    return ''.join(random.sample(string.ascii_letters + string.digits, 12))

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

def inference(config_file:str, checkpoint_file:str,
              img_key:str) -> None:
    cache_path = 'flaskr/cache'
    img_path = os.path.join(cache_path,img_key+'.png')
    img = mmcv.imread(img_path)
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
    img_npy = visualizer.get_image()[..., ::-1]
    from PIL import Image
    img_stream = Image.fromarray(img_npy.astype('uint8')).convert('RGB')
    img_stream.save(os.path.join(cache_path,'inf_'+img_key+'.png'))

"""
定义画图方法
"""
class Draw:
    @classmethod
    def Markup(self,data):
        return Mk(data)

    @classmethod
    def get_data(self,path:str ) -> list:
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

    @classmethod
    def generate_lr_chart(self,json_list) -> Line:
        y_data = []
        x_data = []
        for d in json_list:
            if 'lr' in d:
                y_data.append(d['lr'])
                x_data.append(str(d['step']))
        line = (
            Line().add_xaxis(x_data)
            .add_yaxis('Learning Rate', y_data)
        )
        # change size
        line.width = '100%'
        return line

    @classmethod
    def generate_loss_chart(self,json_list) -> Line:
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
                if 'loss_centerness' in d:
                    y_centerness_data.append(d['loss_centerness'])
                else:
                    y_centerness_data.append(d['acc'])
                x_data.append(str(d['step']))
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

    @classmethod
    def generate_bbox_map_chart(self,json_list) -> Line:
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

    @classmethod
    def generate_seg_map_chart(self,json_list) -> Line:
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
if __name__ == '__main__':
    pass