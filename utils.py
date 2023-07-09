import random
import string
import os
import mmcv
from mmdet.utils import register_all_modules
from mmdet.apis import init_detector, inference_detector

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
    img_stream.save(os.path.join(cache_path,'inf_-'+img_key+'.png'))

if __name__ == '__main__':
    pass