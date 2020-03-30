<h1 size=10>Ubuntu16.04+CUDA9.0+Anaconda3.0配置mmdetection运行环境</h1>
## 1.创建虚拟环境

conda create -n open-mmlab python=3.7

source activate open-mmlab

## 2.安装Cython

conda install Cython

## 3.安装mmcv

git clone https://github.com/open-mmlab/mmcv.git

cd mmcv

pip install .

## 4.安装pytorch

conda pytorch torchvision cudatoolkit=9.0 -c pytorch

## 5.安装mmdetection

git clone https://github.com/open-mmlab/mmdetection.git

cd mmdetection

python setup.py build

python setup.py install

## 6.测试

(1).新建demo.py，导入需要的接口

from mmdet.apis import init_detector,inference_detector,show_result

(2).选择网络及权重

例：config_file = '/home/xt/mmdetection/configs/cascade_rcnn_r50_fpn_1x.py'

​       checkpoint_file = '/home/xt/mmdetection/work_dirs/cascade_rcnn_r50_fpn_1x/epoch_1.pth'

(3).构建模型

model = init_detector(config_file,checkpoint_file)

(4).测试并展示结果

result = inference_detector(model,img)

show_result(img,result,model.CLASSES)