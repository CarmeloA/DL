import os
import random

def createImageSet(path,train_no=2000,imageset_path = '/var/samps/VOC2022/ImageSets/Main/',flag=False):
    l = os.listdir(path)
    random.shuffle(l)
    length = len(l)
    print('length:',length)

    if flag:
        path = path.split('/')[-2] + '/'
        print(path)

    train_index = int(length*0.8)
    test_index = train_index + (length-train_index)//2
    val_index = test_index
    print('train_index:',train_index)
    print('test_index:',test_index)
    print('val_index:',val_index)
    # print(train_index+test_index+val_index)
    train = l[:train_index]
    with open(imageset_path + 'train.txt','a') as train_file:
        for img in train:
            if img.endswith('.jpg'):
                train_file.write(path+img.replace('.jpg','')+'\n')
        train_file.close()
    test = l[train_index:test_index]
    with open(imageset_path + 'test.txt','a') as test_file:
        for img in test:
            if img.endswith('.jpg'):
                test_file.write(path+img.replace('.jpg','')+'\n')
        test_file.close()

    val = l[test_index:]
    with open(imageset_path + 'val.txt','a') as val_file:
        for img in val:
            if img.endswith('.jpg'):
                val_file.write(path+img.replace('.jpg','')+'\n')
        val_file.close()
    print(len(train)+len(test)+len(val))

if __name__ == '__main__':
    path = '/var/samps/VOC2022/JPEGImages/'
    for dir in os.listdir(path):
        createImageSet(path+dir+'/')



