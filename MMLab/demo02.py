import os

path = '/home/xt/save/VOCdevkit/VOC2020/JPEGImages/retinanet101_test_img/'

flag = True


with open('/home/xt/save/VOCdevkit/VOC2007/ImageSets/Main/test.txt','r') as f:
    lines = f.readlines()

with open('/home/xt/save/VOCdevkit/VOC2020/ImageSets/retinanet_r101/test1.txt','a') as f1:
    for img in os.listdir(path):
        if flag:
            f1.write('retinanet101_test_img/'+img.replace('.jpg','')+'\n')
        else:
            f1.write(path+img+'\n')