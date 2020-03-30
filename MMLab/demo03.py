import os
import cv2 as cv

path = '/home/xt/save/VOCdevkit/VOC2020/JPEGImages/'

W = []
H = []

for dir in os.listdir(path):
    print(dir)
    for file in os.listdir(path+dir):
        if (path+dir+'/'+file).endswith('.jpg'):
            print(path+dir+'/'+file)
            img = cv.imread(path+dir+'/'+file)
            h = img.shape[0]
            w = img.shape[1]
            H.append(h)
            W.append(w)
print(max(W))
print(min(W))
print('==========')
print(max(H))
print(min(H))
