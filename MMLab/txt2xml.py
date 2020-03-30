# txt_to_xml.py
# encoding:utf-8
# 根据一个给定的XML Schema，使用DOM树的形式从空白文件生成一个XML
from xml.dom.minidom import Document
import cv2
import os


def generate_xml(name, split_lines, img_size, class_ind, img_path):
    doc = Document()  # 创建DOM文档对象

    annotation = doc.createElement('annotation')
    doc.appendChild(annotation)

    title = doc.createElement('folder')
    title_text = doc.createTextNode('unnormalknife')
    title.appendChild(title_text)
    annotation.appendChild(title)

    img_name = name + '.jpg'

    title = doc.createElement('filename')
    title_text = doc.createTextNode(img_name)
    title.appendChild(title_text)
    annotation.appendChild(title)

    title = doc.createElement('path')
    title_text = doc.createTextNode(img_path)
    title.appendChild(title_text)
    annotation.appendChild(title)

    source = doc.createElement('source')
    annotation.appendChild(source)

    title = doc.createElement('database')
    title_text = doc.createTextNode('My database')
    title.appendChild(title_text)
    source.appendChild(title)

    title = doc.createElement('annotation')
    title_text = doc.createTextNode('STLX')
    title.appendChild(title_text)
    source.appendChild(title)

    size = doc.createElement('size')
    annotation.appendChild(size)

    title = doc.createElement('width')
    title_text = doc.createTextNode(str(img_size[1]))
    title.appendChild(title_text)
    size.appendChild(title)

    title = doc.createElement('height')
    title_text = doc.createTextNode(str(img_size[0]))
    title.appendChild(title_text)
    size.appendChild(title)

    title = doc.createElement('depth')
    title_text = doc.createTextNode(str(img_size[2]))
    title.appendChild(title_text)
    size.appendChild(title)

    # segmented = doc.createElement('segmented')
    # title_text = doc.createTextNode('0')
    # segmented.appendChild(title_text)
    # annotation.appendChild(segmented)

    title = doc.createElement('segmented')
    title_text = doc.createTextNode('0')
    title.appendChild(title_text)
    annotation.appendChild(title)

    for split_line in split_lines:
        line = split_line.strip().split()
        print('line:', line)
        if int(line[0]) in class_ind:
            # for fusion image
            w = int(float(line[3]) * img_size[1])
            h = int(float(line[4]) * img_size[0])
            xmin = int(float(line[1]) * img_size[1])
            ymin = int(float(line[2]) * img_size[0])
            xmax = xmin + w
            ymax = ymin + h

            # for single item img
            # xmin = line[1]
            # ymin = line[2]
            # xmax = line[3]
            # ymax = line[4]

            object = doc.createElement('object')
            annotation.appendChild(object)

            title = doc.createElement('name')
            title_text = doc.createTextNode(class_dict[int(line[0])])
            title.appendChild(title_text)
            object.appendChild(title)

            title = doc.createElement('pose')
            title_text = doc.createTextNode('Unspecified')
            title.appendChild(title_text)
            object.appendChild(title)

            title = doc.createElement('truncated')
            title_text = doc.createTextNode('0')
            title.appendChild(title_text)
            object.appendChild(title)

            title = doc.createElement('difficult')
            title_text = doc.createTextNode('0')
            title.appendChild(title_text)
            object.appendChild(title)

            bndbox = doc.createElement('bndbox')
            object.appendChild(bndbox)
            title = doc.createElement('xmin')
            title_text = doc.createTextNode(str(int(xmin)))
            title.appendChild(title_text)
            bndbox.appendChild(title)
            title = doc.createElement('ymin')
            title_text = doc.createTextNode(str(int(ymin)))
            title.appendChild(title_text)
            bndbox.appendChild(title)
            title = doc.createElement('xmax')
            title_text = doc.createTextNode(str(int(xmax)))
            title.appendChild(title_text)
            bndbox.appendChild(title)
            title = doc.createElement('ymax')
            title_text = doc.createTextNode(str(int(ymax)))
            title.appendChild(title_text)
            bndbox.appendChild(title)

    # 将DOM对象doc写入文件
    f = open('/var/samps/VOC2022/Annotations/unnormalknife/' + name + '.xml', 'w')
    f.write(doc.toprettyxml(indent=''))
    f.close()


if __name__ == '__main__':
    class_dict = {0: 'umbrella', 1: 'pliers', 2: 'charger', 3: 'cellphone', 4: 'laptop', 5: 'watch', 6: 'keys',
                  7: 'fruits',
                  8: 'cup', 9: 'book', 10: 'shoes', 11: 'bottle', 12: 'glassbos', 13: 'screwdriver', 14: 'spoon_fork',
                  15: 'lighter',
                  16: 'bigknife', 17: 'smallknife', 18: 'unnormalknife', 19: 'knife', 20: 'gun', 21: 'rifle',
                  22: 'shotguns', 23: 'bullet',
                  24: 'cosmetics', 25: 'fireworks', 26: 'charging-treasure', 27: 'compressed-gas',
                  28: 'cigarette-lighter', 29: 'tortoise',
                  30: 'snake', 31: 'poultry', 32: 'mammals', 33: 'lizard', 34: 'plant', 35: 'bracelet', 36: 'necklace',
                  37: 'gem'}

    # class_ind=('Pedestrian', 'Car', 'Cyclist')
    class_ind = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
                 11, 12, 13, 14, 15, 16, 17,
                 18, 19, 20, 21, 22, 23, 24,
                 25, 26, 27, 28, 29, 30, 31,
                 32, 33, 34, 35, 36, 37)
    cur_dir = os.getcwd()
    # labels_dir=os.path.join(cur_dir,'Labels')
    labels_dir = '/var/samps/VOC2022/labels/unnormalknife'
    for parent, dirnames, filenames in os.walk(labels_dir):  # 分别得到根目录，子目录和根目录下文件
        for file_name in filenames:
            print('filename:', file_name)
            full_path = os.path.join(parent, file_name)  # 获取文件全路径
            f = open(full_path)
            split_lines = f.readlines()
            print(split_lines)
            name = file_name[:-4]  # 后四位是扩展名.txt，只取前面的文件名
            print('name:', name)
            img_name = name + '.jpg'
            print(img_name)
            img_path = os.path.join('/var/samps/VOC2022/JPEGImages/unnormalknife/', img_name)  # 路径需要自行修改
            print('img_path:', img_path)
            img_size = cv2.imread(img_path).shape
            generate_xml(name, split_lines, img_size, class_ind, img_path)
print('all txts has converted into xmls')