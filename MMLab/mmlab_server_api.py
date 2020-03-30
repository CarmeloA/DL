from mmdet.apis import init_detector,inference_detector,show_result
import mmcv
from mmcv.runner import load_checkpoint
from mmdet.models import build_detector
import time
import os
import argparse
import cv2
import numpy as np

class Mmdetection:
    def __init__(self):
        self.config = '/home/xt/rcnns/server/mmdetection/configs/retinanet_r50_fpn_1x.py'
        self.checkpoint = '/home/xt/rcnns/server/mmdetection/work_dirs/retinanet_r50_fpn_1x/latest.pth'
        self.model = init_detector(self.config,self.checkpoint)
        self.class_name = self.model.CLASSES
        self.score_thr = 0.3
        print('========mmdetection init complete========')
    
    def recognize_img(self,
                      image_byte=None,
                      image_width=None,
                      image_height=None,
                      image_type=None):
        image_bgr = np.frombuffer(image_byte, dtype=np.uint8)
        image_bgr = image_bgr.reshape((image_height, image_width, 3))
        
        result = inference_detector(self.model,image_bgr)

        assert isinstance(self.class_name, (tuple, list))
        if isinstance(result, tuple):
            bbox_result, segm_result = result
        else:
            bbox_result, segm_result = result, None
        bboxes = np.vstack(bbox_result)

        # draw segmentation masks
        if segm_result is not None:
            segms = mmcv.concat_list(segm_result)
            inds = np.where(bboxes[:, -1] > score_thr)[0]
            for i in inds:
                color_mask = np.random.randint(0, 256, (1, 3), dtype=np.uint8)
                mask = maskUtils.decode(segms[i]).astype(np.bool)
                img[mask] = img[mask] * 0.5 + color_mask * 0.5
        # draw bounding boxes
        labels = [
            np.full(bbox.shape[0], i, dtype=np.int32)
            for i, bbox in enumerate(bbox_result)
        ]
        labels = np.concatenate(labels)
        
        res = []

        if self.score_thr > 0:
            assert bboxes.shape[1] == 5
            scores = bboxes[:, -1]
            inds = scores > self.score_thr
            bboxes = bboxes[inds, :]
            labels = labels[inds]
        for bbox, label in zip(bboxes, labels):
            bbox_int = bbox.astype(np.int32)
            left_top = (bbox_int[0], bbox_int[1])
            right_bottom = (bbox_int[2], bbox_int[3])
            label_text = self.class_name[
                label] if self.class_name is not None else 'cls {}'.format(label)

            rs_dict = {
                'item_name': str(label_text),
                'left_upper_x': int(left_top[0]),
                'left_upper_y': int(left_top[1]),
                'right_down_x': int(right_bottom[0]),
                'right_down_y': int(right_bottom[1]),
                'probability': round(bbox[-1], 2)
            }
            res.append(rs_dict)
        print('res type:',type(res))
        print(res)
        return res
            