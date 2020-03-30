import cv2 as cv
import os

path = '/home/xt/Images/CMEX-T10080S/BMP/'

for root,dirs,files in os.walk(path):
    for i,file in enumerate(files):
        img = cv.imread(path+file)
        print(img.shape)
        cut_image = img[49:600,:]
        cv.imwrite('/home/xt/Images/CMEX-T10080S/new_BMP/'+str(i)+'.jpg',cut_image)