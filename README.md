# Image-classification [基于keras实现]
 
## 目录结构
- Models：存放各图像分类backbone的代码
- layers：Models中可能会共同调用的一些组件
- Test
  - data_set：5种花分类测试数据集
    - dateset_after_split: 切分后的数据集
    - raw : 原始数据集
    - split_data.py 切分数据集的脚本
  - paper：复现模型对应的论文
  - Utils.py ：训练工具类
  - train.py : 训练脚本


## 目前已经完成的模型：

1. [GoogLeNet](https://github.com/Kitten-Rec/Image-classification/blob/master/Models/GoogLeNet/GoogLeNet.py)
2. [ResNeXt](https://github.com/Kitten-Rec/Image-classification/blob/master/Models/ResNeXt/resnext.py)
3. [SE_ResNeXt](https://github.com/Kitten-Rec/Image-classification/blob/master/Models/SE_ResNeXt/SE_ResNeXt.py)
4. [SE_ResNet](https://github.com/Kitten-Rec/Image-classification/blob/master/Models/SE_ResNet/SE_ResNet.py)
5. [SK_ResNet](https://github.com/Kitten-Rec/Image-classification/tree/master/Models/SK_ResNet)
6. [ResNeSt](https://github.com/Kitten-Rec/Image-classification/blob/master/Models/ResNeSt/ResNeSt.py)
7. [ShuffleNetV1](https://github.com/Kitten-Rec/Image-classification/blob/master/Models/ShuffleNet/ShuffleNetV1.py) & [ShuffleNetV2](https://github.com/Kitten-Rec/Image-classification/blob/master/Models/ShuffleNet/ShuffleNetV2.py)

note: 模型基本上都是采用继承子类模型的方式

持续更新中···
