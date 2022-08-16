import tensorflow as tf
import tensorflow.keras
from tensorflow.keras import layers, Model

from layers.GlobalAveragePooling2D import GlobalAveragePooling2D
from tensorflow.keras import Sequential
from layers.GroupConv import GroupConv2D

"""ref: https://github.com/zhanghang1989/ResNeSt/tree/master/resnest/torch/models
        https://github.com/xiaoqiang765/backbone/blob/main/ResNest/resnest/resnest.py
"""


class rSoftMax(layers.Layer):
    def __init__(self, radix, cardinality):
        super(rSoftMax, self).__init__()
        assert radix > 0, cardinality > 0
        self.radix = radix
        self.cardinality = cardinality

    def call(self, inputs, **kwargs):  # inputs shape(B,1,1,C)
        x = inputs
        batch = x.shape[0]
        if self.radix > 1:
            # x ==> (batch, self.cardinality, self.radix, -1) ==> (batch, self.radix, self.cardinality, -1)
            x = layers.Permute((0, 3, 1, 2))(x)
            x = tf.reshape(x, [batch, self.cardinality, self.radix, -1])
            x = layers.Permute((0, 2, 1, 3))(x)  # [batch, self.radix, self.cardinality, -1]
            # 在self.radix维度上进行softmax
            x = layers.Softmax(axis=1)(x)  # (batch, self.radix, self.cardinality, 1)
            x = tf.reshape(x, [batch, 1, 1, -1])  # (batch, 1, 1, self.radix*self.cardinality) 即 (B, 1, 1, C)
        else:
            x = tf.keras.activations.sigmoid(x)  # (batch, 1, 1, channels*radix)
        return x


class SplitAttentionConv2d(layers.Layer):
    def __init__(self, in_channels, channels, kernel_size, stride=(1, 1), padding=(0, 0),
                 groups=1, bias=True,
                 radix=2, reduction_factor=4, norm_layer=None, **kwargs):
        super(SplitAttentionConv2d, self).__init__()
        self.radix = radix
        self.cardinality = groups
        self.channels = channels
        # 最小为32
        inter_channels = max(channels * radix // reduction_factor, 32)  # 中间层输出通道数
        self.conv = Sequential([
            layers.ZeroPadding2D(padding=padding),
            # #############分组卷积
            GroupConv2D(input_channels=in_channels, output_channels=channels * radix, kernel_size=kernel_size,
                        strides=stride, groups=groups * radix, use_bias=bias, **kwargs)
        ])

        self.use_bn = norm_layer is not None
        if self.use_bn:
            self.bn0 = norm_layer()

        self.relu = layers.ReLU()
        self.fc1 = GroupConv2D(input_channels=channels * radix, output_channels=inter_channels,
                               kernel_size=1, groups=self.cardinality)
        if self.use_bn:
            self.bn1 = norm_layer()
        self.fc2 = GroupConv2D(input_channels=inter_channels, output_channels=channels * radix,
                               kernel_size=1, groups=self.cardinality)
        self.rsoftmax = rSoftMax(radix, groups)

    def call(self, inputs, **kwargs):  # inputs shape (B, H, W, C)
        x = self.conv(inputs)
        if self.use_bn:
            x = self.bn0(x)
        x = self.relu(x)

        batch, rchannel = x.shape[:2]
        if self.radix > 1:
            splited = layers.Lambda(tf.split(),
                                    arguments={'axis': 3, 'num_or_size_splits': rchannel // self.radix})(x)
            gap = sum(splited)
        else:
            gap = x

            # 全局平均池化
            gap = GlobalAveragePooling2D()(gap)  # (batch, 1, 1, channels)
            # gap = layers.GlobalAveragePooling2D()(gap)  # (batch, channels)
            gap = self.fc1(gap)

            if self.use_bn:
                gap = self.bn1(gap)
            gap = self.relu(gap)  # (batch, 1, 1, inter_channels)
            atten = self.fc2(gap)  # (batch, 1, 1, channels*radix)
            atten = self.rsoftmax(atten)  # (B, 1, 1, channels*radix)

        if self.radix > 1:
            attens = layers.Lambda(tf.split(),
                                   arguments={'axis': 3, 'num_or_size_splits': rchannel // self.radix})(atten)
            out = sum([att * split for (att, split) in zip(attens, splited)])
        else:
            out = atten * x  # (batch, h, w, channels*radix), radix = 1 ==> # (batch, h, w, channels)
        return out


class ResNest_Block(layers.Layer):
    """ResNet Bottleneck"""
    expansion = 4  # 决定了self.conv3的输出通道数out_channels = planes*expansion

    def __init__(self, planes, stride=1, downsample=None,
                 radix=1, cardinality=1, bottleneck_width=64,
                 avd=False, avd_first=False, is_first=False,
                 norm_layer=None):
        super(ResNest_Block, self).__init__()
        group_width = int(planes * (bottleneck_width / 64.)) * cardinality  # 中间层输出通道数 一个分支的通道数
        self.conv1 = layers.Conv2D(filters=group_width, kernel_size=1, use_bias=False)
        self.bn1 = norm_layer()
        self.radix = radix
        self.avd = avd and (stride > 1 or is_first)  # 每个模块在第一个残差块里将上一个模块的通道数翻倍，并将高和宽减半
        self.avd_first = avd_first  # 决定了池化层添加在self.conv1层后，还是添加在self.conv2层后
        if self.avd:
            self.avd_layer = Sequential([
                layers.ZeroPadding2D(padding=(1, 1)),
                layers.AveragePooling2D(pool_size=(3, 3), strides=stride)
            ])
            stride = 1
        if radix >= 1:
            self.conv2 = SplitAttentionConv2d(group_width, group_width, kernel_size=3,
                                              stride=(stride, stride), padding=(1, 1),
                                              groups=cardinality, bias=False,
                                              radix=radix, norm_layer=norm_layer)
        else:
            # ###########################分组卷积 只有一个分支

            self.conv2 = Sequential([
                layers.ZeroPadding2D(padding=(1, 1)),
                GroupConv2D(input_channels=group_width, output_channels=group_width, kernel_size=3,
                            strides=(stride, stride), groups=cardinality, use_bias=False)
            ])

            self.bn2 = norm_layer()

        self.conv3 = layers.Conv2D(planes * self.expansion, kernel_size=1, use_bias=False)
        self.bn3 = norm_layer()
        self.relu = layers.ReLU()
        self.downsample = downsample
        self.stride = stride

    def call(self, inputs, **kwargs):
        # print(inputs.shape)
        x = inputs  # inputs shape (B, H, W, C)
        residual = x

        out = self.conv1(x)  # out shape: (batch, group_width, h, w)
        out = self.bn1(out)
        if self.avd and self.avd_first:
            out = self.avd_layer(out)  # 池化层。长宽减半  ==> (batch, group_width, h/2, w/2)

        # 进行池化后：h_, w_ = h/2, w/2，否则h_, w_ = h, w
        out = self.conv2(out)  # out shape: (batch, h_, w_, C)

        if self.radix == 0:
            # 单层conv后接 BN, Dropout, relu
            out = self.bn2(out)
            out = self.relu(out)

        if self.avd and not self.avd_first:
            out = self.avd_layer(out)

        out = self.conv3(out)  # out shape: (batch, planes * self.expansion, h_, w_)
        out = self.bn3(out)

        # 对输入x进行形状改变，使之与out shape相同
        if self.downsample is not None:
            residual = self.downsample(residual)

        # print("out.shape: ", out.shape)
        # print("residual.shape: ", residual.shape)
        out += residual
        # 在relu之间进行叠加
        return self.relu(out)  # out shape: (batch, planes * self.expansion, h_, w_)


# ResNest模型
# class ResNeSt(tensorflow.keras.layers.Layer):
class ResNeSt(tensorflow.keras.Model):
    """ResNet Variants

        Parameters
        ----------
        block : Block
            Class for the residual block. Options are BasicBlockV1, BottleneckV1.
        nlayers : list of int
            Numbers of layers in each block
        classes : int, default 1000
            Number of classification classes.
        dilated : bool, default False
            Applying dilation strategy to pretrained ResNet yielding a stride-8 model,
            typically used in Semantic Segmentation.
        norm_layer : object
            Normalization layer used in backbone network (default: :class:`mxnet.gluon.nn.BatchNorm`;
            for Synchronized Cross-GPU BatchNormalization).

        Reference:

            - He, Kaiming, et al. "Deep residual learning for image recognition." Proceedings of the IEEE conference on computer vision and pattern recognition. 2016.

            - Yu, Fisher, and Vladlen Koltun. "Multi-scale context aggregation by dilated convolutions."
        """

    def __init__(self, block, layers_list, in_channels=3, radix=1, groups=1, bottleneck_width=64,
                 num_classes=4, deep_stem=False, stem_width=64, avg_down=False,
                 avd=False, avd_first=False,
                 final_drop=0.0, norm_layer=layers.BatchNormalization):
        self.cardinality = groups
        self.bottleneck_width = bottleneck_width
        # ResNet-D params
        self.inplanes = stem_width * 2 if deep_stem else 64
        self.avg_down = avg_down
        # ResNeSt params
        self.radix = radix
        self.avd = avd
        self.avd_first = avd_first
        super(ResNeSt, self).__init__()

        # ResNet第一个模块
        if deep_stem:
            # 3个3x3卷积层替代1个7x7卷积层
            self.conv1 = Sequential([
                layers.ZeroPadding2D(padding=(1, 1)),
                layers.Conv2D(stem_width, kernel_size=3, stride=(2, 2), use_bias=False),  # 长宽减半
                norm_layer(),
                layers.ReLU(),
                layers.ZeroPadding2D(padding=(1, 1)),
                layers.Conv2D(stem_width, kernel_size=3, use_bias=False),
                norm_layer(),
                layers.ReLU(),
                layers.ZeroPadding2D(padding=(1, 1)),
                layers.Conv2D(self.inplanes, kernel_size=3, use_bias=False)
            ])
        else:
            self.conv1 = Sequential([
                layers.ZeroPadding2D(padding=(3, 3)),
                layers.Conv2D(self.inplanes, kernel_size=7, strides=(2, 2), use_bias=False)  # 长宽减半
            ])

        self.bn1 = norm_layer()
        self.relu = layers.ReLU()
        self.maxpool = Sequential([  # 长宽减半
            layers.ZeroPadding2D(padding=(1, 1)),
            layers.MaxPool2D(pool_size=(3, 3), strides=2)
        ])

        # ResNet的后四个模块，每个模块都是ResNet残差块，这里换成了ResNest块-Bottleneck
        # 经过每个残差块后，长宽减半（除第一个残差块外），通道加倍
        self.layer1 = self._make_layer(block, 64, layers_list[0], norm_layer=norm_layer, is_first=False)
        self.layer2 = self._make_layer(block, 128, layers_list[1], stride=2, norm_layer=norm_layer)
        self.layer3 = self._make_layer(block, 256, layers_list[2], stride=2, norm_layer=norm_layer)
        self.layer4 = self._make_layer(block, 512, layers_list[3], stride=2, norm_layer=norm_layer)

        # 最后，与GoogLeNet一样，加入全局平均池化层后接上全连接层输出。
        self.avgpool = layers.GlobalAveragePooling2D()  # 全局平均池化层 输出两维
        self.drop = layers.Dropout(final_drop) if final_drop > 0.0 else None
        self.fc = layers.Dense(units=num_classes, activation='softmax', use_bias=False)

    # ResNet残差网络
    def _make_layer(self, block, planes, num_blocks, stride=1, norm_layer=None, is_first=True):
        downsample = None
        # Bottleneck网络输出通道数： planes * expansion
        if stride != 1 or self.inplanes != planes * block.expansion:
            # 此时需要下采样，使得通道数一致
            down_layer = []
            # ceil_mode=True: 计算输出信号大小的时候，会使用向上取整，代替默认的向下取整的操作
            # count_include_pad=False：计算平均池化时，将不包括padding的0
            if self.avg_down:
                down_layer.append(
                    layers.AveragePooling2D(pool_size=(stride, stride), strides=stride)
                )

                down_layer.append(
                    layers.Conv2D(planes * block.expansion, kernel_size=1, strides=stride, use_bias=False))
            else:
                down_layer.append(layers.Conv2D(planes * block.expansion, kernel_size=1, strides=(stride, stride),
                                                use_bias=False))
                down_layer.append(layers.BatchNormalization(axis=3))

            down_layer.append(norm_layer())
            downsample = Sequential(down_layer)

        nlayers = []
        # 每个模块在第一个残差块里将通道数变为 planes * block.expansion，并将高和宽减半  ==> 由downsample层完成
        # planes, stride = 1, downsample = None,
        # radix = 1, cardinality = 1, bottleneck_width = 64,
        # avd = False, avd_first = False, is_first = False,
        # norm_layer = None
        nlayers.append(block(planes, stride=stride, downsample=downsample,
                             radix=self.radix, cardinality=self.cardinality,
                             bottleneck_width=self.bottleneck_width,
                             avd=self.avd, avd_first=self.avd_first,
                             is_first=is_first, norm_layer=norm_layer
                             ))

        self.inplanes = planes * block.expansion
        for i in range(1, num_blocks):
            # 该block输出通道数out_channels = planes * block.expansion
            nlayers.append(block(planes, stride=1, downsample=None,
                                 radix=self.radix, cardinality=self.cardinality,
                                 bottleneck_width=self.bottleneck_width,
                                 avd=self.avd, avd_first=self.avd_first,
                                 is_first=False, norm_layer=norm_layer
                                 ))

        return Sequential(nlayers)

    # def call(self, inputs, **kwargs):
    def call(self, inputs, training=None, mask=None):
        x = inputs  # x shape: (b, c, h, w)
        x = self.conv1(x)  # x shape: (b, self.inplanes, h//2, w//2)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)  # x shape: (b, self.inplanes, h//4, w//4)

        # 经过每个残差块后，长宽减半（除第一个残差块外），通道加倍（除第一个残差块外）
        x = self.layer1(x)  # x shape: (b, 256, h//4, w//4)         , 256 = 64 * block.expansion
        x = self.layer2(x)  # x shape: (b, 512, h//8, w//8)         , 512 = 128 * block.expansion
        x = self.layer3(x)  # x shape: (b, 1024, h//16, w//16)      , 1024 = 256 * block.expansion
        x = self.layer4(x)  # x shape: (b, 2048, h//32, w//32)      , 2048 = 512 * block.expansion

        x = self.avgpool(x)  # x shape: (b, 2048)
        if self.drop:
            x = self.drop(x)
        x = self.fc(x)  # x shape: (b, num_classes)
        return x


# def resnest50(input_shape, classes):
#     input = input_shape
#     x = ResNeSt(ResNest_Block, [3, 4, 6, 3], num_classes=classes)(input)
#     model = Model(inputs=input, outputs=x)
#     return model
def resnest50(classes):
    return ResNeSt(ResNest_Block, [3, 4, 23, 3], num_classes=classes)


def resnest101(classes):
    return ResNeSt(ResNest_Block, [3, 4, 23, 3], num_classes=classes)


def resnest152(classes):
    return ResNeSt(ResNest_Block, [3, 8, 36, 3], num_classes=classes)


# if __name__ == '__main__':
#     input_shape = layers.Input(shape=(224, 224, 3))
#     model = resnest50(input_shape=input_shape, classes=5)
#     model.summary()
