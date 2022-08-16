import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Sequential
from tensorflow.keras import layers

import tensorflow.keras.backend as K

from layers.GroupConv import GroupConv2D


class ShuffleV2Block(layers.Layer):
    def __init__(self, inp, oup, mid_channels, ksize, stride):
        super(ShuffleV2Block, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        self.mid_channels = mid_channels
        self.ksize = ksize
        pad = ksize // 2
        self.pad = pad
        self.inp = inp

        outputs = oup - inp

        branch_main = [
            # pw
            layers.Conv2D(mid_channels, 1, 1, use_bias=False),
            layers.BatchNormalization(),
            layers.ReLU(),
            # dw
            layers.ZeroPadding2D(padding=pad),
            GroupConv2D(mid_channels, mid_channels, ksize, (stride, stride), groups=mid_channels, use_bias=False),
            layers.BatchNormalization(),
            # pw-linear
            layers.Conv2D(outputs, 1, 1, use_bias=False),
            layers.BatchNormalization(),
            layers.ReLU()
        ]
        self.branch_main = Sequential(branch_main)

        if stride == 2:
            branch_proj = [
                # dw
                layers.ZeroPadding2D(padding=pad),
                GroupConv2D(inp, inp, ksize, stride, groups=inp, use_bias=False),
                layers.BatchNormalization(),
                # pw-linear
                layers.Conv2D(inp, 1, 1, use_bias=False),
                layers.BatchNormalization(),
                layers.ReLU()
            ]
            self.branch_proj = Sequential(branch_proj)
        else:
            self.branch_proj = None

    def call(self, inputs, **kwargs):  # old_x shape : (B, H, W, C)
        old_x = inputs
        if self.stride == 1:
            x_proj, x = self.channel_shuffle(old_x)
            return layers.Concatenate(axis=-1)([x_proj, self.branch_main(x)])
        elif self.stride == 2:  # stride=2没有shuffle!!!但是论文图上有
            x_proj = old_x
            x = old_x
            return layers.Concatenate()([self.branch_proj(x_proj), self.branch_main(x)])

    def channel_shuffle(self, x):
        batchsize, height, width, num_channels = x.shape
        assert (num_channels % 4 == 0)

        x = layers.Reshape(target_shape=(height, width, num_channels // 2, 2))(x)  # (B, H, W, C/2, 2)
        x = K.permute_dimensions(x, (4, 0, 1, 2, 3))  # (2, B, H, W, C/2)
        return x[0], x[1]  # shuffle & split


class ShuffleNetV2(keras.Model):
    def __init__(self, classes, model_size='1.5x'):
        super(ShuffleNetV2, self).__init__()
        print('model size is ', model_size)

        self.stage_repeats = [4, 8, 4]
        self.model_size = model_size
        if model_size == '0.5x':
            self.stage_out_channels = [-1, 24, 48, 96, 192, 1024]
        elif model_size == '1.0x':
            self.stage_out_channels = [-1, 24, 116, 232, 464, 1024]
        elif model_size == '1.5x':
            self.stage_out_channels = [-1, 24, 176, 352, 704, 1024]
        elif model_size == '2.0x':
            self.stage_out_channels = [-1, 24, 244, 488, 976, 2048]
        else:
            raise NotImplementedError

        # building first layer
        input_channel = self.stage_out_channels[1]
        self.first_conv = Sequential([
            layers.ZeroPadding2D(padding=(1, 1)),
            layers.Conv2D(input_channel, 3, 2, use_bias=False),
            layers.BatchNormalization(),
            layers.ReLU()
        ])

        self.maxpool = Sequential([
            layers.ZeroPadding2D(padding=1),
            layers.MaxPool2D(pool_size=3, strides=2)
        ])

        self.features = []
        for idxstage in range(len(self.stage_repeats)):
            numrepeat = self.stage_repeats[idxstage]
            output_channel = self.stage_out_channels[idxstage + 2]

            for i in range(numrepeat):
                if i == 0:  # 每个stage的第一个block通道变2倍，剩下的block通道数不变
                    self.features.append(ShuffleV2Block(input_channel, output_channel,
                                                        mid_channels=output_channel // 2, ksize=3,
                                                        stride=2))  # 为啥这里是 mid_channels=output_channel // 2 在shufflenetv1中是整除4
                else:
                    self.features.append(ShuffleV2Block(input_channel // 2, output_channel,
                                                        mid_channels=output_channel // 2, ksize=3, stride=1))

                input_channel = output_channel

        self.features = Sequential(self.features)

        self.conv_last = Sequential([
            layers.Conv2D(self.stage_out_channels[-1], 1, 1, use_bias=False),
            layers.BatchNormalization(),
            layers.ReLU()
        ])

        self.globalpool = layers.GlobalAveragePooling2D()  # (batch_size, channels)
        if self.model_size == '2.0x':
            self.dropout = layers.Dropout(0.2)
        self.classifier = layers.Dense(units=classes, use_bias=False, activation="softmax")

    def call(self, inputs, training=None, mask=None):
        x = inputs
        x = self.first_conv(x)
        x = self.maxpool(x)
        x = self.features(x)
        x = self.conv_last(x)
        x = self.globalpool(x)
        if self.model_size == '2.0x':  # 只给2.0x用dropout层
            x = self.dropout(x)
        x = self.classifier(x)
        return x
