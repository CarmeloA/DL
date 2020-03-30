# 深度学习框架交接文档

- ### Faster-RCNN说明

- **重点说明**

  - 目前项目中采用的版本是Faster-RCNN_TF只带有VGG16模型的目标检测框架(考虑到识别速率的问题，所以选用网络深度比较浅的VGG16)。
  - 另一个版本tf-faster-rcnn-master带有Resnet101，可作为辅助选项(源码修改较少)。

- Faster-RCNN_TF运行说明
  - 部署位置
    - 192.168.10.110：/home/xt/PycharmProjects/test/Faster-RCNN_TF-T
  
  - Gitlib链接
    - http://gitlab.stlx.com.cn:8088/dev/rcnns.git 
      http://gitlab.stlx.com.cn:8088/dev/rcnns/tree/master/Faster-RCNN_TF-T 

  - 训练数据存放位置
      - 存放在/home/xt/PycharmProjects/test/data/VOCdevkit，目录结构如下:
~~~
        |__ VOCdevkit   # 按VOC格式制作的数据集
          |__ VOC2007
             |__ Annotations    
                |-- others    # 存放xml文件
                |-- knife
                |-- xxx
                |--  •
                |--  •
                |--  •
             |__ ImageSets
                |-- Main    # 存放划分好的数据集的txt文件
             |__ JPEGImages
                |-- others    # 存放源图片
                |-- knife
                |-- xxx
                |--  •
                |--  •
                |--  •  
~~~
  - 运行环境
    - python3.5 
      ###重点说明###
      - [此版本的faster需要在python3.5的环境下运行，在110虚拟机上已创建好虚拟环境]，执行命令
        ：source activate python35 进入虚拟环境，以下所有程序均在此虚拟环境下运行
    - cuda9.0

  - 需要安装的依赖
    - Cython
    - easydict
    - numpy
    - pandas
    - Pillow
    - opencv-python
    - matplotlib
    - tensorflow-gpu==1.5.0
    注：除了tensorflow-gpu安装指定版本外，其他依赖项安装最新版本即可

  - 主要模块

    - 训练模块：

        配置数据及预训练模型，并训练VGG16网络，得到权重文件
        - 配置选项说明：
          训练配置参数在Faster-RCNN_TF/tools/train_net.py中， 如图 ![](picture/train_config.jpg)
          * 其中几个重要参数讲解：
            1) --device  选择利用GPU或CPU训练，默认选用GPU
            2) --device_id  选用第几块显卡来训练，默认选用第0块显卡
            3) --iters  训练的迭代次数 ，默认为1000000
            4) --weights 指定初始化参数所使用的预训练模型(目前只有Imagenet)的地址，预训练模型文件在110虚拟机上所在地址为/home/xt/    PycharmProjects/data/pretrain_model/VGG_imagenet.npy
            5) --names 指定类别文件，文件书写格式为，如图 ![](picture/names.jpg)。默认为Faster-RCNN_TF-T/data/names/names37.txt文件，如需添加类别，按此文件书写格式添加即可。
            6) --data，VOC数据集VOCdevkit所在目录。
          
        - 训练
          * 训练演示
            • train_net.py中已设置默认配置参数，只需在终端输入：python train_net.py即可
              如需修改其他参数，指定参数并赋值，例如：python train_net.py --iters 500000 --names /home/xt/PycharmProjects/test/Faster-RCNN_TF-T/data/names/names25.txt --data /home/xt/PycharmProjects/test/data/VOCdevkit

          * 训练完成生成文件说明
            • 权重文件保存的地址在/../Faster-RCNN_TF/output/faster_rcnn_end2end/voc_2007_trainval/中，如图 ![](picture/model.jpg)
            • 其中.ckpt文件存放训练得到的权重文件，也是验证与识别时所使用的文件
            • .meta文件存放tensorflow所使用的静态流程图

        - 调参说明
          * 训练阶段调参
            • 重要参数需在/../Faster-RCNN_TF/lib/fast_rcnn/config.py中更改，主要参数涉及到RPN网络及fast-rcnn网络
            • 可调参数解析：
              __C.TRAIN.LEARNING_RATE：学习率，调节梯度下降的步伐。可调范围（1,0.1,0.001,0.0001,0.00001，…）,最佳为0.0001
              __C.TRAIN.MOMENTUM：随机梯度下降中动量的选择。可调范围（0.9~0.95）
              __C.TRAIN.STEPSIZE：迭代次数的选择，一般按学习率的大小来设置，如果学习率设置较低，迭代次数需设置大些，学习率设置较高，                    迭代次数适当降低，否则会导致过拟合
              __C.TRAIN.DISPLAY：迭代多少次时，展示RPN网络及fast-rcnn网络的loss值，一般设置为20
              __C.TRAIN.BATCH_SIZE：RPN网络产生的图片上感兴趣区域的数量，可选（64,128,256）
              __C.TRAIN.SNAPSHOT_ITERS：迭代多少次保存一次模型（比较重要）
              __C.TRAIN.RPN_POSITIVE_OVERLAP：在RPN网络选取提案框时，计算IOU时判断是正例的阈值，建议可调范围（0.5~0.7），阈值越                                  高，给fast-rcnn网络提供的正例质量越高，但会引起正负例不均衡，导致模型的过拟合。阈值越低，给fast-rcnn网络提供的正例                  质量越低，识别效果越差。
              其他参数可依据具体情况进行调节。

          * 测试阶段调参
            • 同样参数在/../Faster-RCNN_TF/lib/fast_rcnn/config.py中更改，主要参数涉及到RPN网络
              __C.TEST.RPN_POST_NMS_TOP_N：在RPN提案框中，经过NMS（非极大值抑制）之后，排名前N的提案框输入到fast-rcnn网络进行识别。默认N设置为2000，由于TOP-100即可提供高质量的提案框，同时为了降低识别速率，故可以将N设为100，经测试识别效果基本没差别，识别速率提升明显。如果测试时识别效果不理想，可将N值调大。

    - 验证模块：
        评定权重文件的好坏，通过统计计算真实标签与预测的标签得到各个类别的mAP值
        - 配置选项说明：
          验证配置参数在Faster-RCNN_TF/tools/test_net.py中， 如图 ![](picture/test_net.jpg)
          * 重要参数讲解：
          1) --weights：指定用于验证模型存放的地址
          2) --names：指定类别文件，文件书写格式为，如图 ![](picture/names.jpg)。默认为Faster-RCNN_TF-T/data/names/names37.txt文件，如需添加类别，按此文件书写格式添加即可。
          3) --data：VOC数据集VOCdevkit目录所在地址

        - 验证：
          * 验证演示
          • test_net.py中已设置默认配置参数，只需在终端输入：python test_net.py即可
            如需修改其他参数，指定参数并赋值，例如：python test_net.py --weights /home/xt/PycharmProjects/test/Faster-RCNN_TF-T/output/faster_rcnn_end2end/voc_2007_trainval1.3/VGGnet_fast_rcnn_iter_200000.ckpt --data /home/xt/PycharmProjects/test/data/VOCdevkit
          
          * 验证完成生成模型说明
          • 验证完成后会生成AP.txt文件，里面记录各个类别的AP值及计算mAP值，用来衡量识别结果
          
    - 识别模块：
        用得到的权重文件进行目标检测(用于小批量观察结果，大批量建议使用识别评估系统)
        - 配置选项说明：
        识别配置参数在Faster-RCNN_TF/tools/demo.py中，如图 ![](picture/demo.jpg)
        * 重要参数讲解：
        1) --model：指定用于识别模型存放的地址
        2) --path：存放识别图片的文件夹路径

        - 识别
        * 识别演示
        • demo.py中已设置默认参数，只需在终端输入：python demo.py即可
          如果要修改其他参数，指定参数并赋值，例如：python demo.py --model /home/xt/PycharmProjects/test/Faster-RCNN_TF-T/output/faster_rcnn_end2end/voc_2007_trainval1.3/VGGnet_fast_rcnn_iter_200000.ckpt



  - tf-faster-rcnn-master运行说明
    - 部署位置
      - 192.168.10.110：/home/xt/PycharmProjects/test/tf-faster-rcnn-master
    
    - Gitlib链接
      - http://gitlab.stlx.com.cn:8088/dev/rcnns.git 
        http://gitlab.stlx.com.cn:8088/dev/rcnns/tree/master/tf-faster-rcnn-master 
        

    - 运行环境
      - python3.5 
        ###重点说明###
        - 此版本的faster需要在python3.5的环境下运行，在110虚拟机上已创建好虚拟环境，执行命令
          ：source activate python35 进入虚拟环境，以下所有程序均在此虚拟环境下运行(所需依赖均在此虚拟环境下)
      - cuda9.0

    - 需要安装的依赖
      - Cython
      - easydict
      - numpy
      - pandas
      - Pillow
      - opencv-python
      - matplotlib
      - tensorflow-gpu==1.5.0
      注：除了tensorflow-gpu安装指定版本外，其他依赖项安装最新版本即可 

    - 主要模块
      - 训练模块

         配置数据及预训练模型，并训练VGG16网络，得到权重文件
        - 配置选项说明：
          训练配置参数在Faster-RCNN_TF/tools/trainval_net.py中， 如图 ![](picture/tf_trainval.jpg)
          * 其中几个重要参数讲解(与Faster-RCNN_TF版本重复的参数不做说明)：
            1) --net  配置使用的主干网络，可选vgg16, res50, res101, res152, mobile
          
        - 训练
          * 训练演示
            • 与Faster-RCNN_TF版本训练方式相同

          * 训练完成生成文件说明
            • 权重文件保存的地址在/../tf-faster-rcnn-master/output/xxx/voc_2007_trainval/中，其中xxx为所选的主干网络名称
            • 其中.ckpt.data-00000-of-00001某个ckpt的数据文件，保存每个变量的取值，保存的是网络的权值，偏置，操作等等
            • .ckpt.index ：某个ckpt的index文件 二进制 或者其他格式 不可直接查看
            • .ckpt.meta：某个ckpt的meta数据  二进制 或者其他格式 不可直接查看，保存了TensorFlow计算图的结构信息

      - 验证模块
          评定权重文件的好坏，通过统计计算真实标签与预测的标签得到各个类别的mAP值
        - 配置选项说明：
          验证配置参数在Faster-RCNN_TF/tools/test_net.py中
          * 重要参数讲解：
          1) --model：指定用于验证模型存放的地址
          2) --net：指定网络主干，需与验证模型的主干网络相同
          3) --config：指定想要配置的参数，各种参数的配置不同，不同的配置完成不同的功能

        - 验证：
          * 验证演示
          • test_net.py中已设置默认配置参数，只需在终端输入：python test_net.py即可
            如需修改其他参数，指定参数并赋值，例如：python test_net.py --model /home/xt/PycharmProjects/test/tf-faster-rcnn-master/output/res101/voc_2007_trainval/default/res101_faster_rcnn_iter_850000.ckpt --net re101
          
      - 识别模块       
          用得到的权重文件进行目标检测
        - 配置选项说明：
        识别配置参数在tf-faster-rcnn-master/tools/demo.py中，如图 ![](picture/tf_demo.jpg)
        * 重要参数讲解：
        1) --net：指定主干网络，默认res101
        2) --model：在output/xxx/voc_2007_trainval/default/文件夹下所调用权重文件的名称，默认为         res101_faster_rcnn_iter_1000000.ckpt
        3) --path：指定需要识别的图片文件夹
        4) --save_path：指定文件夹用于保存识别结果图片

        - 识别
        * 识别演示
        • demo.py中已设置默认参数，只需在终端输入：python demo.py即可
          如果要修改其他参数，指定参数并赋值，例如：python demo.py --net res101 --model /home/xt/PycharmProjects/test/tf-faster-rcnn-master/output/res101/voc_2007_trainval/default/res101_faster_rcnn_iter_1000000.ckpt --path /home/xt/PycharmProjects/test/tf-faster-rcnn-master/data/plot_image/

### Yolov2说明

  - YOLOv2运行说明
    - 部署位置
      - 192.168.10.110：/home/xt/darknet
      - 192.168.10.136：/home/xt/1/darknet
      - 192.168.10.138：/home/xt/darknet
      - 192.168.10.108: /home/xt/darknet
    
    - 官网地址
      - https://pjreddie.com/darknet/yolov2/

    - 训练数据存放位置
      - 均存放在各个虚拟机的darknet/scripts/VOCdevkit/文件夹下，目录结构如下:
~~~
        |__ VOCdevkit   # 按VOC格式制作的数据集
          |__ VOC2007
             |__ Annotations    
                |-- others    # 存放xml文件
                |-- knife
                |-- xxx
                |--  •
                |--  •
                |--  •
             |__ ImageSets
                |-- Main    # 存放划分好的数据集的txt文件
             |__ JPEGImages
                |-- others    # 存放源图片
                |-- knife
                |-- xxx
                |--  •
                |--  •
                |--  •
             |__ labels    # [此文件夹是通过vco_label.py生成]
                |-- others    # 存放yolov2训练所需的txt文件
                |-- knife
                |-- xxx
                |--  •
                |--  •
                |--  •
~~~
    - 运行环境
      - cuda8.0及以上版本
      - python 3.0以上版本

    - 需要安装的依赖
      - Cython
      - numpy
      - pandas
      - Pillow
      - opencv-python
      - matplotlib
    
    - 数据集准备说明
        由于将数据集各类别细分处理，现数据集按类别存放在不同的文件夹下，如图 ![](picture/Data.jpg)，将制作好的voc数据集放到指定目录下，运行darknet/scripts/voc_label.py，该脚本已做修改，用以将生成的label文件夹按类别细分存放txt文件并在darknet/scripts/下生成训练所需的2007_test.txt、2007_train.txt、2007_val.txt。[修改好的voc_label.py脚本存放在138虚拟机上]。

    - 训练
        终端进入到darknet目录下，执行：./darknet detector train cfg/voc.data cfg/xxx.cfg  # xxx为所使用的网络结构

        - voc.data 存放于darknet/cfg/目录下，用于记录所需要的5个参数
          - classes  # 训练分类的个数
          - train  # 数据阶段生成的2007_train.txt文件路径
          - valid  # 数据阶段生成的2007_val.txt文件路径
          - names  # voc.names文件路径
          - backup  # 模型存放路径

        - voc.names 存放于darknet/data/目录下，记录类别名字

        - xxx.cfg , xxx为所使用的的网络结构
          其中classes为类别数量，最后一层的卷积层的卷积核个数即filters参数，需要根据classes数量来进行调整，计算公式为：filters=5*(classes+5)

          