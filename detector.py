# Copyright (c) OpenMMLab. All rights reserved.
import asyncio
from argparse import ArgumentParser

from mmdetection.mmdet.apis import (async_inference_detector, inference_detector,
                        init_detector, show_result_pyplot)


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('img', help='Image file')
    parser.add_argument('config', help='Config file')
    parser.add_argument('checkpoint', help='Checkpoint file')
    parser.add_argument('--out-file', default=None, help='Path to output file')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--palette',
        default='coco',
        choices=['coco', 'voc', 'citys', 'random'],
        help='Color palette used for visualization')
    parser.add_argument(
        '--score-thr', type=float, default=0.3, help='bbox score threshold')
    parser.add_argument(
        '--async-test',
        action='store_true',
        help='whether to set async options for async inference.')
    args = parser.parse_args()
    return args


def detect(img, config, checkpoint, out_file, device='cuda:0'):
    # build the model from a config file and a checkpoint file
    model = init_detector(config, checkpoint, device=device)
    # test a single image
    result = inference_detector(model, img)
    # show the results
    show_result_pyplot(
        model,
        img,
        result,
        palette='coco',
        score_thr=0.3,
        out_file=out_file)
    return result

if __name__=='__main__':
    config = 'mmdetection/configs/yolox/yolox_s_8x8_300e_coco.py'
    checkpoint = 'mmdetection/yolox/weights/yolox_s_8x8_300e_coco_20211121_095711-4592a793.pth'

    res = detect('images/car.png', config, checkpoint, out_file=None)
