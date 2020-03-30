from mmdet.apis import init_detector,inference_detector,show_result
import mmcv
from mmcv.runner import load_checkpoint
from mmdet.models import build_detector
import time
import os
import argparse
import cv2
import numpy as np

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

def parse_args():
    parser = argparse.ArgumentParser(description='Test a demo')
    parser.add_argument('--config',dest='config_file',default='/home/xt/mmdetection/configs/retinanet_x101_64x4d_fpn_1x.py')
    parser.add_argument('--checkpoint',dest='checkpoint_file',default='/home/xt/mmdetection/checkpoint/retinanet_x101_64x4d_fpn_1x/3rd_train/latest.pth')
    parser.add_argument('--path',dest='path',default='/home/xt/Images/test_img2/')
    parser.add_argument('--waitKey',dest='waitKey',default=0)
    args = parser.parse_args()
    return args

args = parse_args()

def get_imgs(img_path):
    imgs = []
    img_suffix = ['.jpg', '.BMP', '.jpeg', '.gif', '.JPEG', '.png']
    for file in os.listdir(img_path):
        if file.split('.')[-1] in img_suffix:
            imgs.append(img_path + file)
    return imgs

def init_model(config_file,checkpoint_file):
    model = init_detector(config_file, checkpoint_file)
    return model

def get_result(img,model):
    result = inference_detector(model, img)
    return result


def show_det_result(img,result):
    show_result(img, result, model.CLASSES, score_thr=0.5, wait_time=args.waitKey)

def calculate_det_area(model,result,score_thr=0.3):
    class_name = model.CLASSES
    print('class_name:',class_name)
    assert isinstance(class_name, (tuple, list))
    if isinstance(result, tuple):
        bbox_result, segm_result = result
    else:
        bbox_result, segm_result = result, None
    bboxes = np.vstack(bbox_result)
    print('bboxes1:',bboxes)

    labels = [
        np.full(bbox.shape[0], i, dtype=np.int32)
        for i, bbox in enumerate(bbox_result)
    ]
    labels = np.concatenate(labels)
    print('labels1:', labels)

    res = []

    if score_thr > 0:
        assert bboxes.shape[1] == 5
        scores = bboxes[:, -1]
        print('scores:',scores)
        inds = scores > score_thr
        print('inds:',inds)
        bboxes = bboxes[inds, :]
        print('bboxes:',bboxes)
        labels = labels[inds]
        print('labels:',labels)

        if bboxes.size != 0:
            for bbox, label in zip(bboxes, labels):
                print('label:', label)
                bbox_int = bbox.astype(np.int32)
                left_top = (bbox_int[0], bbox_int[1])
                right_bottom = (bbox_int[2], bbox_int[3])
                label_text = class_name[
                    label] if class_name is not None else 'cls {}'.format(label)

                left_upper_x = int(left_top[0])
                left_upper_y = int(left_top[1])
                right_down_x = int(right_bottom[0])
                right_down_y = int(right_bottom[1])

                area = (right_down_x - left_upper_x) * (right_down_y - left_upper_y)

                print('area:', area)
                return area, left_upper_x, left_upper_y, right_down_x, right_down_y
        else:
            return 0,0,0,0,0

    

def calculate_gt(label_txt,w,h):
    file = open(label_txt, 'r')
    line = file.readline().split()
    gt_left_x = int(float(line[1])*w)
    gt_left_y = int(float(line[2])*h)
    gt_right_x = int(float(line[3])*w) + gt_left_x
    gt_right_y = int(float(line[4])*h) + gt_left_y
    gt_area = (gt_right_x - gt_left_x) * (gt_right_y - gt_left_y)
    return gt_area,gt_left_x,gt_left_y,gt_right_x,gt_right_y

def calculate_IOU(gt_left_x,gt_right_x,gt_left_y,gt_right_y,left_upper_x,left_upper_y,area,gt_area):
    overlap = abs(abs(gt_right_x-gt_left_x) - abs(gt_left_x - left_upper_x)) * \
        abs(abs(gt_right_y-gt_left_y) - abs(gt_left_y - left_upper_y))
    union = gt_area+area-overlap
    iou = overlap / union
    if iou > 1:
        return 1
    return iou

def demo():
    file = open('/var/samps/VOC2021/ImageSets/Main/2021_test.txt','r')
    lines = file.readlines()
    imgs = [line.strip() for line in lines]
    print(len(imgs))
    model = init_model(args.config_file,args.checkpoint_file)
    iou_l = []

    for img in imgs:
        pic = cv2.imread(img)
        w = pic.shape[1]
        h = pic.shape[0]
        label_txt = img.replace('JPEGImages','labels').replace('.jpg','.txt')
        print(label_txt)
        result = get_result(img,model)
    
        area,left_upper_x,left_upper_y,right_down_x,right_down_y = calculate_det_area(model,result)
        gt_area,gt_left_x,gt_left_y,gt_right_x,gt_right_y = calculate_gt(label_txt,w,h)

        cv2.rectangle(pic,(left_upper_x,left_upper_y),(right_down_x,right_down_y),(255,0,0),thickness=2)
        cv2.rectangle(pic,(gt_left_x,gt_left_y),(gt_right_x,gt_right_y),(0,0,255),thickness=2)
        cv2.imshow('win',pic)

        # iou = (max(area, gt_area) - min(area, gt_area)) / (2 * min(area, gt_area))
        iou = calculate_IOU(gt_left_x,gt_right_x,gt_left_y,gt_right_y,left_upper_x,left_upper_y,area,gt_area)
        iou_l.append(iou)
        print('iou:',iou)
        cv2.waitKey(0)
    print(sum(iou_l)/len(iou_l))


if __name__ == '__main__':
    demo()
    # file = open('/var/samps/VOC2021/ImageSets/Main/2021_test.txt', 'r')
    # lines = file.readlines()
    # imgs = [line.strip() for line in lines]
    # print(len(imgs))
    # model = init_model(args.config_file, args.checkpoint_file)
    # for img in imgs:
    #     result = get_result(img,model)
    #     show_det_result(img,result)
    # for img in os.listdir('/var/samps/VOC2021/JPEGImages/s_bottle/'):
    #     pic = '/var/samps/VOC2021/JPEGImages/s_bottle/' + img
    #     start = time.time()
    #     result = get_result(pic,model)
    #     print('time:',time.time()-start)
    #     show_det_result(pic,result)
    #     print(time.time()-start)

    

